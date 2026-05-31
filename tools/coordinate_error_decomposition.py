from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_grape_point_report import collect_case_groups, safe_float


DEFAULT_EXPERIMENTS = [
    (
        "EMA_BIFPN",
        REPO_ROOT / "outputs/03_global_analysis/selection_reassignment_v1_20260529",
    ),
    (
        "TOPROI_SIMCC_REFINER_V1_FAIR100",
        REPO_ROOT
        / "outputs/06_coordinate_refiner_experiments/ema_bifpn_toproi_simcc_refiner_v1_fair100_backbone_pretrain_20260530/report",
    ),
    (
        "DETACHED_QUERY_SELECTOR_V2_FAIR100",
        REPO_ROOT
        / "outputs/05_selector_experiments/ema_bifpn_detached_query_selector_v2_has_fair100_backbone_pretrain_20260530/report_best_point_l2",
    ),
]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    record_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose GPPoint-DETR picking-point error into box-anchor, offset-expression, "
            "and query-selection components using existing prediction records only."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/03_global_analysis/coordinate_error_decomposition_{datetime.now():%Y%m%d}",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment spec in NAME=record_dir form. Defaults to known record dirs if omitted.",
    )
    parser.add_argument("--split", choices=("valid", "test", "both"), default="test")
    parser.add_argument("--has-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-anchor-ratio", type=float, default=0.12)
    parser.add_argument("--anchor-x-ratio", type=float, default=0.5)
    parser.add_argument("--dominant-threshold-px", type=float, default=15.0)
    parser.add_argument("--better-candidate-margin-px", type=float, default=5.0)
    return parser.parse_args()


def parse_experiments(specs: list[str]) -> list[ExperimentSpec]:
    if not specs:
        out = []
        for name, path in DEFAULT_EXPERIMENTS:
            if (path / "test_prediction_records.json").exists():
                out.append(ExperimentSpec(name=name, record_dir=path))
        if not out:
            raise FileNotFoundError("No default prediction-record directories were found.")
        return out

    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --experiment spec, expected NAME=record_dir: {spec}")
        name, raw_path = spec.split("=", 1)
        out.append(ExperimentSpec(name=name.strip(), record_dir=Path(raw_path).resolve()))
    return out


def load_records(record_dir: Path, split: str) -> list[dict[str, Any]]:
    path = record_dir / f"{split}_prediction_records.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def xyxy_to_xywh(box: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1, y1, max(1e-6, x2 - x1), max(1e-6, y2 - y1)


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def point_l2(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def encode_toproi_offset(
    box_xyxy: list[float],
    point_xy: list[float],
    top_anchor_ratio: float,
    anchor_x_ratio: float,
) -> tuple[float, float]:
    x, y, w, h = xyxy_to_xywh(box_xyxy)
    anchor_x = x + anchor_x_ratio * w
    anchor_y = y + top_anchor_ratio * h
    return (float(point_xy[0]) - anchor_x) / w, (float(point_xy[1]) - anchor_y) / h


def decode_toproi_offset(
    box_xyxy: list[float],
    offset_xy: tuple[float, float],
    top_anchor_ratio: float,
    anchor_x_ratio: float,
) -> tuple[float, float]:
    x, y, w, h = xyxy_to_xywh(box_xyxy)
    anchor_x = x + anchor_x_ratio * w
    anchor_y = y + top_anchor_ratio * h
    return anchor_x + float(offset_xy[0]) * w, anchor_y + float(offset_xy[1]) * h


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([safe_float(row.get(key)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def summarize_l2_rows(rows: list[dict[str, Any]], l2_key: str = "standard_l2") -> dict[str, Any]:
    l2 = np.asarray([safe_float(row.get(l2_key)) for row in rows], dtype=np.float64)
    l2 = l2[np.isfinite(l2)]
    dx = np.asarray([abs(safe_float(row.get("standard_dx"))) for row in rows], dtype=np.float64)
    dy = np.asarray([abs(safe_float(row.get("standard_dy"))) for row in rows], dtype=np.float64)
    if l2.size == 0:
        return {
            "count": 0,
            "mean_l2": 0.0,
            "median_l2": 0.0,
            "p90_l2": 0.0,
            "ppl_sr_30": 0.0,
            "ppl_sr_50": 0.0,
            "mean_abs_dx": 0.0,
            "mean_abs_dy": 0.0,
        }
    return {
        "count": int(l2.size),
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "p90_l2": float(np.quantile(l2, 0.90)),
        "ppl_sr_30": float(np.mean(l2 <= 30.0)),
        "ppl_sr_50": float(np.mean(l2 <= 50.0)),
        "mean_abs_dx": float(dx[np.isfinite(dx)].mean()) if np.isfinite(dx).any() else 0.0,
        "mean_abs_dy": float(dy[np.isfinite(dy)].mean()) if np.isfinite(dy).any() else 0.0,
    }


def classify_coordinate_source(row: dict[str, Any], threshold: float) -> str:
    box_l2 = safe_float(row["pred_box_gt_offset_l2"])
    offset_l2 = safe_float(row["gt_box_pred_offset_l2"])
    if box_l2 >= threshold and offset_l2 < threshold:
        return "box_anchor_dominant"
    if offset_l2 >= threshold and box_l2 < threshold:
        return "offset_expression_dominant"
    if box_l2 >= threshold and offset_l2 >= threshold:
        return "coupled_box_offset"
    return "low_error_or_interaction"


def build_record_lookup(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(record["image_id"]): record for record in records}


def standard_decomposition_rows(
    records: list[dict[str, Any]],
    has_threshold: float,
    iou_threshold: float,
    top_anchor_ratio: float,
    anchor_x_ratio: float,
    dominant_threshold: float,
) -> list[dict[str, Any]]:
    correct_pairs, _, _ = collect_case_groups(records, iou_threshold, has_threshold)
    lookup = build_record_lookup(records)
    rows: list[dict[str, Any]] = []

    for case in correct_pairs:
        record = lookup[int(case["image_id"])]
        gt = record["gt_instances"][int(case["gt_index"])]
        pred = record["pred_instances"][int(case["pred_index"])]
        gt_box = [float(v) for v in gt["bbox_xyxy"]]
        pred_box = [float(v) for v in pred["bbox_xyxy"]]
        gt_point = [float(v) for v in gt["picking_point"]]
        pred_point = [float(v) for v in pred["picking_point"]]
        gt_offset = encode_toproi_offset(gt_box, gt_point, top_anchor_ratio, anchor_x_ratio)
        pred_offset = encode_toproi_offset(pred_box, pred_point, top_anchor_ratio, anchor_x_ratio)
        pred_box_gt_offset_point = decode_toproi_offset(pred_box, gt_offset, top_anchor_ratio, anchor_x_ratio)
        gt_box_pred_offset_point = decode_toproi_offset(gt_box, pred_offset, top_anchor_ratio, anchor_x_ratio)
        pred_anchor_point = decode_toproi_offset(pred_box, (0.0, 0.0), top_anchor_ratio, anchor_x_ratio)
        gt_anchor_point = decode_toproi_offset(gt_box, (0.0, 0.0), top_anchor_ratio, anchor_x_ratio)

        row = {
            "image_id": int(case["image_id"]),
            "file_name": case.get("file_name", record.get("file_name", "")),
            "gt_index": int(case["gt_index"]),
            "pred_index": int(case["pred_index"]),
            "iou": safe_float(case.get("iou")),
            "pred_score": safe_float(pred.get("score")),
            "has_score": safe_float(pred.get("has_picking_score")),
            "gt_area": safe_float(gt.get("area")),
            "standard_dx": float(pred_point[0] - gt_point[0]),
            "standard_dy": float(pred_point[1] - gt_point[1]),
            "standard_l2": point_l2(pred_point, gt_point),
            "pred_box_gt_offset_l2": point_l2(pred_box_gt_offset_point, gt_point),
            "gt_box_pred_offset_l2": point_l2(gt_box_pred_offset_point, gt_point),
            "pred_anchor_only_l2": point_l2(pred_anchor_point, gt_point),
            "gt_anchor_prior_l2": point_l2(gt_anchor_point, gt_point),
            "gt_offset_x": gt_offset[0],
            "gt_offset_y": gt_offset[1],
            "pred_offset_x": pred_offset[0],
            "pred_offset_y": pred_offset[1],
            "offset_abs_error_x": abs(pred_offset[0] - gt_offset[0]),
            "offset_abs_error_y": abs(pred_offset[1] - gt_offset[1]),
        }
        row["coordinate_source"] = classify_coordinate_source(row, dominant_threshold)
        rows.append(row)
    return rows


def visible_gt_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records for gt in record.get("gt_instances", []) if bool(gt.get("has_picking")))


def best_candidate_for_gt(
    record: dict[str, Any],
    gt_index: int,
    has_threshold: float,
    iou_threshold: float,
) -> dict[str, Any] | None:
    gt = record["gt_instances"][gt_index]
    gt_box = [float(v) for v in gt["bbox_xyxy"]]
    gt_point = [float(v) for v in gt["picking_point"]]
    best: dict[str, Any] | None = None
    for pred_index, pred in enumerate(record.get("pred_instances", [])):
        if safe_float(pred.get("has_picking_score")) < has_threshold:
            continue
        iou = box_iou_xyxy([float(v) for v in pred["bbox_xyxy"]], gt_box)
        if iou < iou_threshold:
            continue
        pred_point = [float(v) for v in pred["picking_point"]]
        candidate = {
            "pred_index": int(pred_index),
            "iou": float(iou),
            "l2": point_l2(pred_point, gt_point),
            "score": safe_float(pred.get("score")),
            "has_score": safe_float(pred.get("has_picking_score")),
        }
        if best is None or candidate["l2"] < best["l2"]:
            best = candidate
    return best


def selection_diagnostic_rows(
    records: list[dict[str, Any]],
    standard_rows: list[dict[str, Any]],
    has_threshold: float,
    iou_threshold: float,
    better_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    standard_by_gt = {
        (int(row["image_id"]), int(row["gt_index"])): row
        for row in standard_rows
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        for gt_index, gt in enumerate(record.get("gt_instances", [])):
            if not bool(gt.get("has_picking")):
                continue
            key = (int(record["image_id"]), int(gt_index))
            standard = standard_by_gt.get(key)
            best = best_candidate_for_gt(record, gt_index, has_threshold, iou_threshold)
            row = {
                "image_id": int(record["image_id"]),
                "file_name": record.get("file_name", ""),
                "gt_index": int(gt_index),
                "has_standard_pair": standard is not None,
                "standard_l2": safe_float(standard.get("standard_l2")) if standard else None,
                "standard_pred_index": int(standard["pred_index"]) if standard else None,
                "has_oracle_candidate": best is not None,
                "oracle_l2": safe_float(best.get("l2")) if best else None,
                "oracle_pred_index": int(best["pred_index"]) if best else None,
                "oracle_iou": safe_float(best.get("iou")) if best else None,
            }
            row["better_candidate_exists"] = bool(
                standard is not None
                and best is not None
                and safe_float(standard["standard_l2"]) - safe_float(best["l2"]) >= better_margin
            )
            rows.append(row)

    visible_count = len(rows)
    standard_count = sum(1 for row in rows if row["has_standard_pair"])
    oracle_count = sum(1 for row in rows if row["has_oracle_candidate"])
    better_count = sum(1 for row in rows if row["better_candidate_exists"])
    oracle_l2 = np.asarray(
        [safe_float(row["oracle_l2"]) for row in rows if row["has_oracle_candidate"]],
        dtype=np.float64,
    )
    summary = {
        "visible_gt_count": int(visible_count),
        "standard_pair_count": int(standard_count),
        "coverage_error_count": int(visible_count - standard_count),
        "oracle_iou50_count": int(oracle_count),
        "oracle_iou50_recall": float(oracle_count / visible_count) if visible_count else 0.0,
        "oracle_iou50_mean_l2": float(oracle_l2.mean()) if oracle_l2.size else 0.0,
        "oracle_iou50_p90_l2": float(np.quantile(oracle_l2, 0.90)) if oracle_l2.size else 0.0,
        "better_candidate_count": int(better_count),
        "better_candidate_rate_of_standard": float(better_count / standard_count) if standard_count else 0.0,
    }
    return rows, summary


def coordinate_summary(rows: list[dict[str, Any]], dominant_threshold: float) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["coordinate_source"]] = source_counts.get(row["coordinate_source"], 0) + 1
    summary = {
        "standard": summarize_l2_rows(rows, "standard_l2"),
        "pred_box_gt_offset": summarize_values(rows, "pred_box_gt_offset_l2"),
        "gt_box_pred_offset": summarize_values(rows, "gt_box_pred_offset_l2"),
        "pred_anchor_only": summarize_values(rows, "pred_anchor_only_l2"),
        "gt_anchor_prior": summarize_values(rows, "gt_anchor_prior_l2"),
        "offset_abs_error_x": summarize_values(rows, "offset_abs_error_x"),
        "offset_abs_error_y": summarize_values(rows, "offset_abs_error_y"),
        "dominant_threshold_px": dominant_threshold,
        "coordinate_source_counts": source_counts,
    }
    total = max(1, len(rows))
    summary["coordinate_source_rates"] = {key: value / total for key, value in source_counts.items()}
    return summary


def decide_direction(coord: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    standard_count = max(1, int(selection.get("standard_pair_count", 0)))
    coverage_gap = int(selection.get("coverage_error_count", 0))
    better_rate = safe_float(selection.get("better_candidate_rate_of_standard"))
    source_rates = coord.get("coordinate_source_rates", {})
    offset_rate = safe_float(source_rates.get("offset_expression_dominant", 0.0))
    coupled_rate = safe_float(source_rates.get("coupled_box_offset", 0.0))
    box_rate = safe_float(source_rates.get("box_anchor_dominant", 0.0))
    pred_box_gt_offset_mean = safe_float(coord["pred_box_gt_offset"]["mean"])
    gt_box_pred_offset_mean = safe_float(coord["gt_box_pred_offset"]["mean"])

    if better_rate >= 0.15:
        decision = "selection_first"
        rationale = "A material fraction of standard pairs already has a better IoU-matched visible candidate."
    elif coverage_gap >= max(20, int(0.12 * standard_count)):
        decision = "detector_or_visibility_first"
        rationale = "Too many visible GT instances fail to form standard pairs before coordinate refinement."
    elif offset_rate + coupled_rate >= 0.45 or gt_box_pred_offset_mean > pred_box_gt_offset_mean + 3.0:
        decision = "coordinate_first"
        rationale = "Replacing predicted boxes with GT boxes does not fix the point, so normalized offset expression is the main bottleneck."
    elif box_rate >= 0.30 or pred_box_gt_offset_mean > gt_box_pred_offset_mean + 3.0:
        decision = "box_anchor_calibration_first"
        rationale = "GT offsets decoded on predicted boxes remain inaccurate, so box/top-anchor propagation is dominant."
    else:
        decision = "mixed_low_confidence"
        rationale = "No single source dominates; use this result only as a guardrail before the next experiment."
    return {
        "decision": decision,
        "rationale": rationale,
        "better_candidate_rate": better_rate,
        "offset_or_coupled_rate": offset_rate + coupled_rate,
        "box_anchor_rate": box_rate,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def report_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 2) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" if idx == 0 else "---:" for idx, _ in enumerate(columns)) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            values.append(fmt(value, digits) if isinstance(value, float) else str(value if value is not None else "-"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def flatten_summary_for_table(result: dict[str, Any]) -> dict[str, Any]:
    coord = result["coordinate_summary"]
    sel = result["selection_summary"]
    decision = result["decision"]
    return {
        "experiment": result["experiment"],
        "split": result["split"],
        "decision": decision["decision"],
        "pair": coord["standard"]["count"],
        "standard_mean": coord["standard"]["mean_l2"],
        "standard_p90": coord["standard"]["p90_l2"],
        "pred_box_gt_offset_mean": coord["pred_box_gt_offset"]["mean"],
        "gt_box_pred_offset_mean": coord["gt_box_pred_offset"]["mean"],
        "gt_anchor_prior_mean": coord["gt_anchor_prior"]["mean"],
        "offset_x_err": coord["offset_abs_error_x"]["mean"],
        "offset_y_err": coord["offset_abs_error_y"]["mean"],
        "coverage_gap": sel["coverage_error_count"],
        "better_candidate_count": sel["better_candidate_count"],
        "better_candidate_rate": sel["better_candidate_rate_of_standard"],
        "oracle_mean": sel["oracle_iou50_mean_l2"],
        "oracle_p90": sel["oracle_iou50_p90_l2"],
        "offset_or_coupled_rate": decision["offset_or_coupled_rate"],
        "box_anchor_rate": decision["box_anchor_rate"],
    }


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [flatten_summary_for_table(result) for result in results]
    lines = [
        "# Coordinate Error Decomposition",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "- Protocol: existing prediction records only; no checkpoint loading, no training, no test-time threshold search.",
        "- Decomposition assumes the current `top_center` offset decode: point = top-ROI anchor + normalized offset * box size.",
        "",
        "## Final Read",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['experiment']}` / `{row['split']}`: `{row['decision']}`. "
            f"standard mean={fmt(row['standard_mean'])}, pred-box+GT-offset mean={fmt(row['pred_box_gt_offset_mean'])}, "
            f"GT-box+pred-offset mean={fmt(row['gt_box_pred_offset_mean'])}, better-candidate rate={fmt(row['better_candidate_rate'], 3)}."
        )

    lines.extend(
        [
            "",
            "## Main Decomposition Table",
            "",
        ]
    )
    lines.extend(
        report_table(
            rows,
            [
                ("Experiment", "experiment"),
                ("Split", "split"),
                ("Decision", "decision"),
                ("pair", "pair"),
                ("std mean", "standard_mean"),
                ("std p90", "standard_p90"),
                ("pred box + GT offset", "pred_box_gt_offset_mean"),
                ("GT box + pred offset", "gt_box_pred_offset_mean"),
                ("GT anchor prior", "gt_anchor_prior_mean"),
                ("off x err", "offset_x_err"),
                ("off y err", "offset_y_err"),
                ("coverage gap", "coverage_gap"),
                ("better cand", "better_candidate_count"),
                ("better rate", "better_candidate_rate"),
                ("oracle mean", "oracle_mean"),
            ],
            digits=3,
        )
    )
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `pred box + GT offset`: keeps predicted boxes but replaces the model's point offset with the GT normalized offset. If this stays large, box/top-anchor propagation is hurting coordinates.",
            "- `GT box + pred offset`: keeps the model's normalized offset but decodes it on the GT box. If this stays large, the point offset representation/head is the bottleneck.",
            "- `better candidate`: standard matched query has an IoU>=0.5 visible candidate for the same GT whose point is at least the configured margin better. If frequent, selection/ranking should be fixed before new coordinate losses.",
            "",
            "## Recommendation Rule",
            "",
            "- `coordinate_first` means the next real model should keep spatial point evidence instead of regressing a single bbox-normalized offset vector.",
            "- `box_anchor_calibration_first` means the next no-training step should calibrate top-anchor/box propagation before changing the point head.",
            "- `selection_first` means the candidate pool already contains better points and a selector/reranker is more justified than another coordinate loss.",
            "- `detector_or_visibility_first` means visible GT coverage is the limiting factor.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(
    spec: ExperimentSpec,
    split: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = load_records(spec.record_dir, split)
    standard_rows = standard_decomposition_rows(
        records=records,
        has_threshold=args.has_threshold,
        iou_threshold=args.iou_threshold,
        top_anchor_ratio=args.top_anchor_ratio,
        anchor_x_ratio=args.anchor_x_ratio,
        dominant_threshold=args.dominant_threshold_px,
    )
    selection_rows, selection_summary = selection_diagnostic_rows(
        records=records,
        standard_rows=standard_rows,
        has_threshold=args.has_threshold,
        iou_threshold=args.iou_threshold,
        better_margin=args.better_candidate_margin_px,
    )
    coord_summary = coordinate_summary(standard_rows, args.dominant_threshold_px)
    decision = decide_direction(coord_summary, selection_summary)
    return {
        "experiment": spec.name,
        "record_dir": str(spec.record_dir),
        "split": split,
        "has_threshold": args.has_threshold,
        "iou_threshold": args.iou_threshold,
        "top_anchor_ratio": args.top_anchor_ratio,
        "anchor_x_ratio": args.anchor_x_ratio,
        "coordinate_summary": coord_summary,
        "selection_summary": selection_summary,
        "decision": decision,
        "standard_rows": standard_rows,
        "selection_rows": selection_rows,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = parse_experiments(args.experiment)
    splits = ["valid", "test"] if args.split == "both" else [args.split]
    results = []
    all_standard_rows = []
    all_selection_rows = []

    for spec in experiments:
        for split in splits:
            record_file = spec.record_dir / f"{split}_prediction_records.json"
            if not record_file.exists():
                continue
            result = run_one(spec, split, args)
            results.append({k: v for k, v in result.items() if k not in ("standard_rows", "selection_rows")})
            for row in result["standard_rows"]:
                all_standard_rows.append({"experiment": spec.name, "split": split, **row})
            for row in result["selection_rows"]:
                all_selection_rows.append({"experiment": spec.name, "split": split, **row})

    if not results:
        raise RuntimeError("No prediction records were found for the requested experiments/splits.")

    (args.output_dir / "coordinate_error_decomposition_summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "coordinate_standard_pair_decomposition.csv", all_standard_rows)
    write_csv(args.output_dir / "coordinate_selection_decomposition.csv", all_selection_rows)
    write_report(args.output_dir / "coordinate_error_decomposition_report_zh.md", results)
    print(f"[coordinate-decomposition] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
