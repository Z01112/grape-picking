from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_grape_point_report import collect_case_groups, safe_float


DEFAULT_RECORD_DIR = REPO_ROOT / "outputs/03_global_analysis/selection_reassignment_v1_20260529"
DEFAULT_MAIN_SUMMARY = REPO_ROOT / "outputs/03_global_analysis/post_cleanup_v7_exp2_report_20260525/summary.json"
DEFAULT_EMA_SUMMARY = (
    REPO_ROOT
    / "outputs/02_encoder_experiments/encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526/report/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether better visible picking points already exist in the prediction candidate pool."
    )
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/03_global_analysis/candidate_pool_diagnostic_v1_{datetime.now():%Y%m%d}",
    )
    parser.add_argument("--main-summary", type=Path, default=DEFAULT_MAIN_SUMMARY)
    parser.add_argument("--ema-summary", type=Path, default=DEFAULT_EMA_SUMMARY)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", default="1,3,5,10,20")
    parser.add_argument("--near-iou", type=float, default=0.1)
    return parser.parse_args()


def parse_int_list(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def load_records(record_dir: Path, split: str) -> list[dict[str, Any]]:
    path = record_dir / f"{split}_prediction_records.json"
    if not path.exists():
        raise FileNotFoundError(f"Prediction record file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["primary_checkpoint_split_summary"]["test"]


def flatten_summary(label: str, split_summary: dict[str, Any]) -> dict[str, Any]:
    det = split_summary.get("grape_detection", {})
    has = split_summary.get("has_picking", {})
    point = split_summary.get("picking_point", {})
    size = point.get("size_group_l2_px", {}) or {}
    return {
        "label": label,
        "AP": safe_float(det.get("AP")),
        "F1": safe_float(has.get("f1")),
        "pair": int(point.get("pair_count", 0) or 0),
        "mean_l2": safe_float(point.get("mean_l2_px")),
        "median_l2": safe_float(point.get("median_l2_px")),
        "p90_l2": safe_float(point.get("p90_l2_px")),
        "dx": safe_float(point.get("mean_abs_dx_px", point.get("mae_x_px"))),
        "dy": safe_float(point.get("mean_abs_dy_px", point.get("mae_y_px"))),
        "small_l2": safe_float((size.get("small") or {}).get("mean_l2_px")),
    }


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
    return inter / union if union > 0 else 0.0


def point_case(record: dict[str, Any], gt_idx: int, pred_idx: int, iou: float) -> dict[str, Any]:
    gt = record["gt_instances"][gt_idx]
    pred = record["pred_instances"][pred_idx]
    gt_point = [float(v) for v in gt.get("picking_point", [0.0, 0.0])]
    pred_point = [float(v) for v in pred.get("picking_point", [0.0, 0.0])]
    dx = pred_point[0] - gt_point[0]
    dy = pred_point[1] - gt_point[1]
    return {
        "image_id": int(record["image_id"]),
        "gt_index": int(gt_idx),
        "pred_index": int(pred_idx),
        "gt_area": float(gt.get("area", 0.0)),
        "iou": float(iou),
        "score": safe_float(pred.get("score")),
        "has_score": safe_float(pred.get("has_picking_score")),
        "rank_score": -1,
        "rank_iou": -1,
        "dx_px": float(dx),
        "dy_px": float(dy),
        "l2_px": float(math.hypot(dx, dy)),
    }


def summarize_cases(cases: list[dict[str, Any]], visible_gt_count: int) -> dict[str, Any]:
    l2 = np.asarray([float(item["l2_px"]) for item in cases], dtype=np.float64)
    dx = np.asarray([float(item["dx_px"]) for item in cases], dtype=np.float64)
    dy = np.asarray([float(item["dy_px"]) for item in cases], dtype=np.float64)
    areas = np.asarray([float(item["gt_area"]) for item in cases], dtype=np.float64)
    out = {
        "count": int(len(cases)),
        "recall": float(len(cases) / visible_gt_count) if visible_gt_count else 0.0,
        "mean_l2": float(l2.mean()) if l2.size else 0.0,
        "median_l2": float(np.median(l2)) if l2.size else 0.0,
        "p90_l2": float(np.quantile(l2, 0.90)) if l2.size else 0.0,
        "dx": float(np.mean(np.abs(dx))) if dx.size else 0.0,
        "dy": float(np.mean(np.abs(dy))) if dy.size else 0.0,
        "ppl_sr_30": float(np.mean(l2 <= 30.0)) if l2.size else 0.0,
        "ppl_sr_50": float(np.mean(l2 <= 50.0)) if l2.size else 0.0,
        "small_l2": 0.0,
        "medium_l2": 0.0,
        "large_l2": 0.0,
    }
    if l2.size:
        q1, q2 = np.quantile(areas, [1.0 / 3.0, 2.0 / 3.0])
        groups = {
            "small_l2": l2[areas <= q1],
            "medium_l2": l2[(areas > q1) & (areas <= q2)],
            "large_l2": l2[areas > q2],
        }
        for key, values in groups.items():
            out[key] = float(values.mean()) if values.size else 0.0
    return out


def standard_summary(records: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any], int]:
    correct, _, _ = collect_case_groups(records, 0.5, threshold)
    visible_gt_count = sum(
        1
        for record in records
        for gt in record.get("gt_instances", [])
        if bool(gt.get("has_picking"))
    )
    cases = []
    for item in correct:
        cases.append(
            {
                "image_id": int(item["image_id"]),
                "gt_index": int(item["gt_index"]),
                "pred_index": int(item["pred_index"]),
                "gt_area": float(item["gt_area"]),
                "iou": float(item["iou"]),
                "score": safe_float(item.get("pred_score")),
                "has_score": safe_float(item.get("pred_has_picking_score")),
                "rank_score": -1,
                "rank_iou": -1,
                "dx_px": float(item["dx_px"]),
                "dy_px": float(item["dy_px"]),
                "l2_px": float(item["l2_px"]),
            }
        )
    return summarize_cases(cases, visible_gt_count), visible_gt_count


def candidate_lists(record: dict[str, Any], gt_idx: int, threshold: float) -> list[dict[str, Any]]:
    gt_box = record["gt_instances"][gt_idx]["bbox_xyxy"]
    candidates = []
    for pred_idx, pred in enumerate(record.get("pred_instances", [])):
        if safe_float(pred.get("has_picking_score")) < threshold:
            continue
        iou = box_iou_xyxy(pred.get("bbox_xyxy", [0, 0, 0, 0]), gt_box)
        item = point_case(record, gt_idx, pred_idx, iou)
        item["rank_score_value"] = safe_float(pred.get("score")) * max(safe_float(pred.get("has_picking_score")), 1e-6)
        item["rank_hybrid_value"] = item["rank_score_value"] * (iou + 0.05)
        candidates.append(item)

    by_score = sorted(candidates, key=lambda x: x["rank_score_value"], reverse=True)
    by_iou = sorted(candidates, key=lambda x: x["iou"], reverse=True)
    by_hybrid = sorted(candidates, key=lambda x: x["rank_hybrid_value"], reverse=True)
    rank_score = {(item["pred_index"], item["gt_index"]): idx + 1 for idx, item in enumerate(by_score)}
    rank_iou = {(item["pred_index"], item["gt_index"]): idx + 1 for idx, item in enumerate(by_iou)}
    rank_hybrid = {(item["pred_index"], item["gt_index"]): idx + 1 for idx, item in enumerate(by_hybrid)}
    for item in candidates:
        key = (item["pred_index"], item["gt_index"])
        item["rank_score"] = rank_score[key]
        item["rank_iou"] = rank_iou[key]
        item["rank_hybrid"] = rank_hybrid[key]
    return candidates


def best_from(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return min(items, key=lambda item: float(item["l2_px"]))


def pool_summaries(
    records: list[dict[str, Any]], threshold: float, topk_values: list[int], near_iou: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visible_gt_count = sum(
        1
        for record in records
        for gt in record.get("gt_instances", [])
        if bool(gt.get("has_picking"))
    )
    protocol_cases: dict[str, list[dict[str, Any]]] = {}
    best_rank_rows = []

    for record in records:
        for gt_idx, gt in enumerate(record.get("gt_instances", [])):
            if not bool(gt.get("has_picking")):
                continue
            candidates = candidate_lists(record, gt_idx, threshold)
            any_best = best_from(candidates)
            iou50_best = best_from([item for item in candidates if item["iou"] >= 0.5])
            near_best = best_from([item for item in candidates if item["iou"] >= near_iou])

            for name, case in (
                ("oracle_any_visible", any_best),
                ("oracle_iou50_visible", iou50_best),
                (f"oracle_near_iou{near_iou:.2f}", near_best),
            ):
                if case is not None:
                    protocol_cases.setdefault(name, []).append(case)

            if any_best is not None:
                best_rank_rows.append(
                    {
                        "split_image_id": int(record["image_id"]),
                        "gt_index": int(gt_idx),
                        "best_any_l2": float(any_best["l2_px"]),
                        "best_any_score_rank": int(any_best["rank_score"]),
                        "best_any_iou_rank": int(any_best["rank_iou"]),
                        "best_any_hybrid_rank": int(any_best["rank_hybrid"]),
                        "best_any_iou": float(any_best["iou"]),
                    }
                )

            by_score = sorted(candidates, key=lambda x: x["rank_score_value"], reverse=True)
            by_iou = sorted(candidates, key=lambda x: x["iou"], reverse=True)
            by_hybrid = sorted(candidates, key=lambda x: x["rank_hybrid_value"], reverse=True)
            by_near_score = [item for item in by_score if item["iou"] >= near_iou]
            by_iou50_score = [item for item in by_score if item["iou"] >= 0.5]
            for k in topk_values:
                for protocol, items in (
                    (f"score_top{k}", by_score[:k]),
                    (f"iou_top{k}", by_iou[:k]),
                    (f"hybrid_top{k}", by_hybrid[:k]),
                    (f"near_score_top{k}", by_near_score[:k]),
                    (f"iou50_score_top{k}", by_iou50_score[:k]),
                ):
                    best = best_from(items)
                    if best is not None:
                        protocol_cases.setdefault(protocol, []).append(best)

    rows = []
    for protocol, cases in sorted(protocol_cases.items()):
        row = {"protocol": protocol}
        row.update(summarize_cases(cases, visible_gt_count))
        rows.append(row)
    return rows, best_rank_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def row_by(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    for row in rows:
        if row.get("protocol") == protocol:
            return row
    return {}


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def decision_from(test_rows: list[dict[str, Any]], standard: dict[str, Any]) -> str:
    score5 = row_by(test_rows, "score_top5")
    iou5 = row_by(test_rows, "iou50_score_top5") or row_by(test_rows, "iou_top5")
    oracle_any = row_by(test_rows, "oracle_any_visible")
    std_mean = safe_float(standard.get("mean_l2"), 0.0)
    std_recall = safe_float(standard.get("recall"), 0.0)

    if score5 and score5["recall"] >= std_recall and score5["mean_l2"] <= std_mean - 1.0:
        return "rerank_calibration_first"
    if iou5 and iou5["recall"] >= std_recall and iou5["mean_l2"] <= std_mean - 1.5:
        return "train_selector_first"
    if oracle_any and oracle_any["mean_l2"] <= std_mean - 4.0:
        return "candidate_export_or_selector_required"
    return "coordinate_representation_first"


def make_report(
    output_dir: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    standard_rows: dict[str, dict[str, Any]],
    main_ref: dict[str, Any],
    ema_ref: dict[str, Any],
    decision: str,
    topk_values: list[int],
) -> None:
    test_rows = split_rows["test"]
    std = standard_rows["test"]
    lines = [
        "# Candidate Pool Diagnostic V1",
        "",
        "## Purpose",
        "- No training is used here.",
        "- The diagnostic asks whether accurate visible picking points already exist in the prediction candidate pool.",
        "- `score_topK` uses existing model ranking only; `iou_topK` / `iou50_score_topK` are diagnostic upper bounds that assume instance-local candidate access.",
        "",
        "## References",
        "| Model | AP | F1 | pair | mean L2 | p90 L2 | small L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V7_EXP2_MAIN fair retrain | {fmt(main_ref['AP'],4)} | {fmt(main_ref['F1'],4)} | {main_ref['pair']} | {fmt(main_ref['mean_l2'])} | {fmt(main_ref['p90_l2'])} | {fmt(main_ref['small_l2'])} |",
        f"| EMA_BIFPN report | {fmt(ema_ref['AP'],4)} | {fmt(ema_ref['F1'],4)} | {ema_ref['pair']} | {fmt(ema_ref['mean_l2'])} | {fmt(ema_ref['p90_l2'])} | {fmt(ema_ref['small_l2'])} |",
        f"| EMA_BIFPN records standard | - | - | {std['count']} | {fmt(std['mean_l2'])} | {fmt(std['p90_l2'])} | {fmt(std['small_l2'])} |",
        "",
        "## Test Candidate Pool",
        "| Protocol | recall | pair | mean L2 | median L2 | p90 L2 | dx | dy | SR@30 | SR@50 | small L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = ["standard"]
    for k in topk_values:
        ordered.extend([f"score_top{k}", f"near_score_top{k}", f"iou50_score_top{k}", f"hybrid_top{k}", f"iou_top{k}"])
    ordered.extend(["oracle_iou50_visible", "oracle_near_iou0.10", "oracle_any_visible"])
    lookup = {row["protocol"]: row for row in test_rows}
    lookup["standard"] = std
    for protocol in ordered:
        row = lookup.get(protocol)
        if not row:
            continue
        lines.append(
            f"| {protocol} | {fmt(row.get('recall'),4)} | {int(row.get('count', 0))} | "
            f"{fmt(row.get('mean_l2'))} | {fmt(row.get('median_l2'))} | {fmt(row.get('p90_l2'))} | "
            f"{fmt(row.get('dx'))} | {fmt(row.get('dy'))} | {fmt(row.get('ppl_sr_30'),4)} | "
            f"{fmt(row.get('ppl_sr_50'),4)} | {fmt(row.get('small_l2'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            f"- Diagnostic decision: `{decision}`.",
            "",
            "## Interpretation",
            "- If `score_topK` is already better than standard, a no-training or light calibration reranker is plausible.",
            "- If only `iou_topK` / `iou50_score_topK` improves, the model has useful local candidates but current query ranking does not expose them; use a trained detached selector rather than more point loss.",
            "- If only `oracle_any_visible` improves, useful points exist but are not instance-local; simple geometry reassignment is unlikely to be paper-safe.",
            "- If none improves materially, move to coordinate representation such as keypoint token, ROI refiner, or SimCC/integral heatmap.",
            "",
            "## Files",
            "- `candidate_pool_summary.json`",
            "- `candidate_pool_rows.csv`",
            "- `best_candidate_rank_rows.csv`",
        ]
    )
    (output_dir / "candidate_pool_diagnostic_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk_values = parse_int_list(args.topk)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    standard_rows: dict[str, dict[str, Any]] = {}
    all_rows = []
    all_rank_rows = []

    for split in ("valid", "test"):
        records = load_records(args.record_dir, split)
        standard, visible_gt_count = standard_summary(records, args.threshold)
        standard["protocol"] = "standard"
        standard["visible_gt_count"] = visible_gt_count
        rows, rank_rows = pool_summaries(records, args.threshold, topk_values, args.near_iou)
        for row in rows:
            row["split"] = split
            row["visible_gt_count"] = visible_gt_count
        for row in rank_rows:
            row["split"] = split
        split_rows[split] = rows
        standard_rows[split] = standard
        all_rows.append({"split": split, **standard})
        all_rows.extend(rows)
        all_rank_rows.extend(rank_rows)

    main_ref = flatten_summary("V7_EXP2_MAIN fair retrain", load_test_summary(args.main_summary))
    ema_ref = flatten_summary("EMA_BIFPN report", load_test_summary(args.ema_summary))
    decision = decision_from(split_rows["test"], standard_rows["test"])
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "record_dir": str(args.record_dir.resolve()),
        "threshold": args.threshold,
        "topk": topk_values,
        "near_iou": args.near_iou,
        "decision": decision,
        "references": {
            "main": main_ref,
            "ema_bifpn": ema_ref,
        },
        "standard": standard_rows,
        "candidate_pool": split_rows,
    }

    (args.output_dir / "candidate_pool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "candidate_pool_rows.csv", all_rows)
    write_csv(args.output_dir / "best_candidate_rank_rows.csv", all_rank_rows)
    make_report(args.output_dir, split_rows, standard_rows, main_ref, ema_ref, decision, topk_values)

    print(f"[candidate-pool] decision={decision}")
    print(f"[candidate-pool] wrote {args.output_dir / 'candidate_pool_diagnostic_report_zh.md'}")


if __name__ == "__main__":
    main()
