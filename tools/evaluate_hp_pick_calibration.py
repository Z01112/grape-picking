from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_grape_point_report import collect_case_groups, evaluate_split, safe_float, summarize_split_error


DEFAULT_CONFIG = REPO_ROOT / "configs/rtv4/rtv4_hgnetv2_s_grape_point_enc_ema_bifpn_weighted_fusion.yml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/02_encoder_experiments/encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526/best_composite.pth"
)
DEFAULT_MAIN_SUMMARY = REPO_ROOT / "outputs/03_global_analysis/post_cleanup_v7_exp2_report_20260525/summary.json"
DEFAULT_BASE_SUMMARY = (
    REPO_ROOT
    / "outputs/02_encoder_experiments/encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526/report/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate High-Precision Picking Calibration without retraining.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--main-summary", type=Path, default=DEFAULT_MAIN_SUMMARY)
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE_SUMMARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/03_global_analysis/hp_pick_calibration_{datetime.now():%Y%m%d}",
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", default="0.50:0.78:0.02")
    parser.add_argument("--geometry-alphas", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--y-bands", default="-0.05:0.42,0.00:0.42,0.00:0.36,0.05:0.40")
    parser.add_argument("--anchor-y-ratios", default="0.08,0.12,0.16")
    parser.add_argument("--min-valid-pair-ratio", type=float, default=0.95)
    parser.add_argument("--max-valid-f1-drop", type=float, default=0.01)
    parser.add_argument("--gate-min-ap", type=float, default=0.632)
    parser.add_argument("--gate-min-f1", type=float, default=0.760)
    parser.add_argument("--gate-min-pair", type=int, default=185)
    parser.add_argument("--gate-max-mean-l2", type=float, default=23.40)
    parser.add_argument("--gate-max-p90-l2", type=float, default=49.43)
    return parser.parse_args()


def parse_float_list(spec: str) -> list[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def parse_thresholds(spec: str) -> list[float]:
    if ":" not in spec:
        return parse_float_list(spec)
    start, stop, step = [float(part) for part in spec.split(":")]
    values = []
    cur = start
    while cur <= stop + step * 0.5:
        values.append(round(cur, 6))
        cur += step
    return values


def parse_bands(spec: str) -> list[tuple[float, float]]:
    bands = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = [float(v) for v in part.split(":", 1)]
        if lo >= hi:
            raise ValueError(f"Invalid y band {part!r}")
        bands.append((lo, hi))
    return bands


def load_test_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["primary_checkpoint_split_summary"]["test"]


def box_geometry(pred: dict) -> dict:
    x1, y1, x2, y2 = [float(v) for v in pred.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])]
    px, py = [float(v) for v in pred.get("picking_point", [0.0, 0.0])]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    rel_x = (px - x1) / w
    rel_y = (py - y1) / h
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": w,
        "h": h,
        "px": px,
        "py": py,
        "rel_x": rel_x,
        "rel_y": rel_y,
    }


def geometry_score(pred: dict, y_min: float, y_max: float, anchor_y_ratio: float) -> float:
    g = box_geometry(pred)
    x_over = max(0.0, -g["rel_x"], g["rel_x"] - 1.0)
    y_over = max(0.0, y_min - g["rel_y"], g["rel_y"] - y_max)
    anchor_dist = math.hypot((g["rel_x"] - 0.5) / 0.8, (g["rel_y"] - anchor_y_ratio) / 0.6)
    inside_factor = 1.0 if x_over <= 1e-9 and y_over <= 1e-9 else 0.72
    return float(max(0.0, min(1.0, inside_factor * math.exp(-2.5 * x_over - 3.5 * y_over - 0.15 * anchor_dist))))


def clamp_point(pred: dict, y_min: float, y_max: float, clamp_x: bool) -> tuple[list[float], bool]:
    g = box_geometry(pred)
    px = g["px"]
    py = g["py"]
    if clamp_x:
        px = min(max(px, g["x1"]), g["x2"])
    cy1 = g["y1"] + y_min * g["h"]
    cy2 = g["y1"] + y_max * g["h"]
    py = min(max(py, cy1), cy2)
    changed = abs(px - g["px"]) > 1e-6 or abs(py - g["py"]) > 1e-6
    return [float(px), float(py)], changed


def fallback_point(pred: dict, y_min: float, y_max: float, anchor_y_ratio: float, clamp_x: bool) -> tuple[list[float], bool]:
    g = box_geometry(pred)
    outside_y = g["rel_y"] < y_min or g["rel_y"] > y_max
    outside_x = g["rel_x"] < 0.0 or g["rel_x"] > 1.0
    if not outside_y and not (clamp_x and outside_x):
        return [g["px"], g["py"]], False
    px = g["x1"] + 0.5 * g["w"]
    py = g["y1"] + anchor_y_ratio * g["h"]
    if clamp_x:
        px = min(max(px, g["x1"]), g["x2"])
    return [float(px), float(py)], True


def transform_records(records: list[dict], protocol: dict) -> tuple[list[dict], dict]:
    output = copy.deepcopy(records)
    changed = 0
    scored = 0
    for record in output:
        for pred in record.get("pred_instances", []):
            has_score = max(0.0, safe_float(pred.get("has_picking_score", 0.0), 0.0))
            if protocol["kind"] in {"geom_score", "hybrid_clamp_geom"}:
                geo = geometry_score(pred, protocol["y_min"], protocol["y_max"], protocol["anchor_y_ratio"])
                pred["hp_pick_accept_score"] = float(has_score * (geo ** protocol["alpha"]))
                pred["hp_pick_geometry_score"] = float(geo)
                scored += 1
            else:
                pred["hp_pick_accept_score"] = float(has_score)

            if protocol["kind"] in {"clamp_y", "clamp_xy", "hybrid_clamp_geom"}:
                point, did_change = clamp_point(
                    pred,
                    protocol["y_min"],
                    protocol["y_max"],
                    clamp_x=protocol["kind"] in {"clamp_xy", "hybrid_clamp_geom"},
                )
                if did_change:
                    changed += 1
                    pred["picking_point"] = point
                    pred["picking_point_source"] = protocol["kind"]
            elif protocol["kind"] == "fallback_anchor":
                point, did_change = fallback_point(
                    pred,
                    protocol["y_min"],
                    protocol["y_max"],
                    protocol["anchor_y_ratio"],
                    clamp_x=True,
                )
                if did_change:
                    changed += 1
                    pred["picking_point"] = point
                    pred["picking_point_source"] = protocol["kind"]
    return output, {"changed_predictions": changed, "scored_predictions": scored}


def metric_row(records: list[dict], threshold: float, protocol: dict, split: str, transform_stats: dict) -> dict:
    summary = summarize_split_error(records, threshold, visibility_score_key="hp_pick_accept_score")
    correct, fp, fn = collect_case_groups(records, 0.5, threshold, visibility_score_key="hp_pick_accept_score")
    pair = int(summary.get("point_pair_count", 0))
    tp = int(summary.get("has_picking_correct_count", 0))
    fp_count = int(summary.get("has_picking_false_positive_count", 0))
    fn_count = int(summary.get("has_picking_false_negative_count", 0))
    precision = tp / (tp + fp_count) if tp + fp_count else 0.0
    recall = tp / (tp + fn_count) if tp + fn_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    l2_values = [float(item["l2_px"]) for item in correct if "l2_px" in item]
    size = summary.get("size_group_l2_px", {}) or {}
    return {
        "split": split,
        "protocol": protocol["name"],
        "kind": protocol["kind"],
        "threshold": threshold,
        "alpha": protocol.get("alpha", 0.0),
        "y_min": protocol.get("y_min", ""),
        "y_max": protocol.get("y_max", ""),
        "anchor_y_ratio": protocol.get("anchor_y_ratio", ""),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pair_count": pair,
        "false_positive": fp_count,
        "false_negative": fn_count,
        "mean_l2": safe_float(summary.get("mean_l2_px")),
        "median_l2": safe_float(summary.get("median_l2_px")),
        "p90_l2": safe_float(summary.get("p90_l2_px")),
        "dx": safe_float(summary.get("mean_abs_dx_px")),
        "dy": safe_float(summary.get("mean_abs_dy_px")),
        "ppl_sr_30": sum(v <= 30.0 for v in l2_values) / pair if pair else 0.0,
        "ppl_sr_50": sum(v <= 50.0 for v in l2_values) / pair if pair else 0.0,
        "small_l2": safe_float((size.get("small") or {}).get("mean_l2_px")),
        "medium_l2": safe_float((size.get("medium") or {}).get("mean_l2_px")),
        "large_l2": safe_float((size.get("large") or {}).get("mean_l2_px")),
        "changed_predictions": int(transform_stats.get("changed_predictions", 0)),
        "scored_predictions": int(transform_stats.get("scored_predictions", 0)),
        "correct_case_count": len(correct),
        "fp_case_count": len(fp),
        "fn_case_count": len(fn),
    }


def build_protocols(y_bands: list[tuple[float, float]], alphas: list[float], anchor_y_ratios: list[float]) -> list[dict]:
    protocols = [{"name": "has_threshold", "kind": "raw", "alpha": 0.0}]
    for y_min, y_max in y_bands:
        protocols.extend(
            [
                {
                    "name": f"clamp_y_y{y_min:.2f}_{y_max:.2f}",
                    "kind": "clamp_y",
                    "y_min": y_min,
                    "y_max": y_max,
                    "anchor_y_ratio": "",
                    "alpha": 0.0,
                },
                {
                    "name": f"clamp_xy_y{y_min:.2f}_{y_max:.2f}",
                    "kind": "clamp_xy",
                    "y_min": y_min,
                    "y_max": y_max,
                    "anchor_y_ratio": "",
                    "alpha": 0.0,
                },
            ]
        )
        for anchor_y_ratio in anchor_y_ratios:
            protocols.extend(
                [
                    {
                        "name": f"fallback_anchor_y{y_min:.2f}_{y_max:.2f}_a{anchor_y_ratio:.2f}",
                        "kind": "fallback_anchor",
                        "y_min": y_min,
                        "y_max": y_max,
                        "anchor_y_ratio": anchor_y_ratio,
                        "alpha": 0.0,
                    },
                ]
            )
            for alpha in alphas:
                protocols.extend(
                    [
                        {
                            "name": f"geom_score_y{y_min:.2f}_{y_max:.2f}_a{anchor_y_ratio:.2f}_p{alpha:.2f}",
                            "kind": "geom_score",
                            "y_min": y_min,
                            "y_max": y_max,
                            "anchor_y_ratio": anchor_y_ratio,
                            "alpha": alpha,
                        },
                        {
                            "name": f"hybrid_clamp_geom_y{y_min:.2f}_{y_max:.2f}_a{anchor_y_ratio:.2f}_p{alpha:.2f}",
                            "kind": "hybrid_clamp_geom",
                            "y_min": y_min,
                            "y_max": y_max,
                            "anchor_y_ratio": anchor_y_ratio,
                            "alpha": alpha,
                        },
                    ]
                )
    return protocols


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_valid_rule(valid_rows: list[dict], default_valid: dict, args: argparse.Namespace) -> dict:
    min_pair = math.floor(int(default_valid["pair_count"]) * args.min_valid_pair_ratio)
    min_f1 = float(default_valid["f1"]) - float(args.max_valid_f1_drop)
    feasible = [
        row for row in valid_rows
        if int(row["pair_count"]) >= min_pair and float(row["f1"]) >= min_f1
    ] or valid_rows
    return min(
        feasible,
        key=lambda row: (
            safe_float(row.get("p90_l2"), float("inf")),
            safe_float(row.get("mean_l2"), float("inf")),
            -safe_float(row.get("pair_count"), 0.0),
            -safe_float(row.get("f1"), 0.0),
        ),
    )


def find_matching_row(rows: list[dict], selected: dict, split: str) -> dict:
    for row in rows:
        if row["split"] != split:
            continue
        keys = ("protocol", "kind", "threshold", "alpha", "y_min", "y_max", "anchor_y_ratio")
        if all(str(row.get(key)) == str(selected.get(key)) for key in keys):
            return row
    raise KeyError(f"No {split} row for selected protocol {selected['protocol']}")


def summarize_cases(records: list[dict], selected: dict, limit: int = 20) -> list[dict]:
    protocol = {
        "name": selected["protocol"],
        "kind": selected["kind"],
        "alpha": safe_float(selected.get("alpha"), 0.0),
        "y_min": safe_float(selected.get("y_min"), 0.0),
        "y_max": safe_float(selected.get("y_max"), 1.0),
        "anchor_y_ratio": safe_float(selected.get("anchor_y_ratio"), 0.12),
    }
    transformed, _ = transform_records(records, protocol)
    correct, _, _ = collect_case_groups(
        transformed,
        0.5,
        safe_float(selected["threshold"]),
        visibility_score_key="hp_pick_accept_score",
    )
    top = sorted(correct, key=lambda item: float(item.get("l2_px", 0.0)), reverse=True)[:limit]
    return [
        {
            "image_id": int(item["image_id"]),
            "file_name": item["file_name"],
            "gt_area": float(item["gt_area"]),
            "iou": float(item["iou"]),
            "pred_score": float(item["pred_score"]),
            "has_score": float(item["pred_has_picking_score"]),
            "l2_px": float(item["l2_px"]),
            "dx_px": float(item["dx_px"]),
            "dy_px": float(item["dy_px"]),
            "pred_point": item["pred_point"],
            "gt_point": item["gt_point"],
            "pred_point_inside_gt_box": bool(item["pred_point_inside_gt_box"]),
            "pred_point_inside_pred_box": bool(item["pred_point_inside_pred_box"]),
        }
        for item in top
    ]


def flatten_reference(label: str, summary: dict) -> dict:
    det = summary.get("grape_detection", {})
    has = summary.get("has_picking", {})
    point = summary.get("picking_point", {})
    size = point.get("size_group_l2_px", {}) or {}
    return {
        "label": label,
        "AP": safe_float(det.get("AP")),
        "AP50": safe_float(det.get("AP50")),
        "AR100": safe_float(det.get("AR100")),
        "f1": safe_float(has.get("f1")),
        "pair_count": int(point.get("pair_count", 0)),
        "mean_l2": safe_float(point.get("mean_l2_px")),
        "median_l2": safe_float(point.get("median_l2_px")),
        "p90_l2": safe_float(point.get("p90_l2_px")),
        "dx": safe_float(point.get("mae_x_px", point.get("mean_abs_dx_px"))),
        "dy": safe_float(point.get("mae_y_px", point.get("mean_abs_dy_px"))),
        "small_l2": safe_float((size.get("small") or {}).get("mean_l2_px")),
        "medium_l2": safe_float((size.get("medium") or {}).get("mean_l2_px")),
        "large_l2": safe_float((size.get("large") or {}).get("mean_l2_px")),
    }


def pass_gate(row: dict, ap: float, args: argparse.Namespace) -> dict:
    checks = {
        "AP>=0.632": ap >= args.gate_min_ap,
        "F1>=0.760": safe_float(row.get("f1")) >= args.gate_min_f1,
        "pair>=185": int(row.get("pair_count", 0)) >= args.gate_min_pair,
        "mean<23.40": safe_float(row.get("mean_l2")) < args.gate_max_mean_l2,
        "p90<49.43": safe_float(row.get("p90_l2")) < args.gate_max_p90_l2,
    }
    return {"passed": all(checks.values()), "checks": checks}


def fmt(v: object, digits: int = 2) -> str:
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def write_markdown(
    path: Path,
    main_ref: dict,
    base_ref: dict,
    selected_valid: dict,
    selected_test: dict,
    gate: dict,
    near_miss_rows: list[dict],
    high_precision_rows: list[dict],
) -> None:
    base_ap = safe_float(base_ref["AP"])
    lines = [
        "# HP_PICK_CALIBRATION_V1 Offline Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "- Protocol: no-training calibration; valid selects the rule, test only applies the frozen selected rule.",
        "- AP is unchanged by point calibration because boxes/scores/labels are not modified.",
        "- Goal: improve high-precision visible picking point output while keeping AP and pair coverage acceptable.",
        "",
        "## Same-split Test Comparison",
        "",
        "| Model / Protocol | AP | F1 | pair | mean L2 | median L2 | p90 L2 | dx | dy | small L2 | medium L2 | large L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {main_ref['label']} | {fmt(main_ref['AP'], 4)} | {fmt(main_ref['f1'], 4)} | {main_ref['pair_count']} | "
            f"{fmt(main_ref['mean_l2'])} | {fmt(main_ref['median_l2'])} | {fmt(main_ref['p90_l2'])} | "
            f"{fmt(main_ref['dx'])} | {fmt(main_ref['dy'])} | {fmt(main_ref['small_l2'])} | "
            f"{fmt(main_ref['medium_l2'])} | {fmt(main_ref['large_l2'])} |"
        ),
        (
            f"| {base_ref['label']} | {fmt(base_ap, 4)} | {fmt(base_ref['f1'], 4)} | {base_ref['pair_count']} | "
            f"{fmt(base_ref['mean_l2'])} | {fmt(base_ref['median_l2'])} | {fmt(base_ref['p90_l2'])} | "
            f"{fmt(base_ref['dx'])} | {fmt(base_ref['dy'])} | {fmt(base_ref['small_l2'])} | "
            f"{fmt(base_ref['medium_l2'])} | {fmt(base_ref['large_l2'])} |"
        ),
        (
            f"| HP_PICK_CALIBRATION_V1 selected | {fmt(base_ap, 4)} | {fmt(selected_test['f1'], 4)} | "
            f"{int(selected_test['pair_count'])} | {fmt(selected_test['mean_l2'])} | {fmt(selected_test['median_l2'])} | "
            f"{fmt(selected_test['p90_l2'])} | {fmt(selected_test['dx'])} | {fmt(selected_test['dy'])} | "
            f"{fmt(selected_test['small_l2'])} | {fmt(selected_test['medium_l2'])} | {fmt(selected_test['large_l2'])} |"
        ),
        "",
        "## Selected Rule",
        "",
        "| Split | protocol | threshold | alpha | y_min | y_max | anchor_y | F1 | pair | mean L2 | p90 L2 | changed preds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("valid_selected", selected_valid), ("test_fixed", selected_test)):
        lines.append(
            f"| {label} | {row['protocol']} | {fmt(row['threshold'])} | {fmt(row['alpha'])} | "
            f"{fmt(row['y_min'])} | {fmt(row['y_max'])} | {fmt(row['anchor_y_ratio'])} | "
            f"{fmt(row['f1'], 4)} | {int(row['pair_count'])} | {fmt(row['mean_l2'])} | "
            f"{fmt(row['p90_l2'])} | {int(row['changed_predictions'])} |"
        )

    lines.extend(
        [
            "",
            "## Gate Result",
            "",
            f"- Mainline gate: {'PASS' if gate['passed'] else 'FAIL'}",
        ]
    )
    for name, ok in gate["checks"].items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")

    lines.extend(
        [
            "",
            "## Near-miss Diagnostics",
            "",
            "Coverage-retaining candidates keep the hard F1/pair floor and are sorted by p90.",
            "",
            "| protocol | threshold | F1 | pair | mean L2 | p90 L2 | small L2 | changed preds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in near_miss_rows:
        lines.append(
            f"| {row['protocol']} | {fmt(row['threshold'])} | {fmt(row['f1'], 4)} | {int(row['pair_count'])} | "
            f"{fmt(row['mean_l2'])} | {fmt(row['p90_l2'])} | {fmt(row['small_l2'])} | {int(row['changed_predictions'])} |"
        )

    lines.extend(
        [
            "",
            "High-precision subset candidates pass the p90 target but usually lose too much pair/F1.",
            "",
            "| protocol | threshold | F1 | pair | mean L2 | p90 L2 | small L2 | changed preds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in high_precision_rows:
        lines.append(
            f"| {row['protocol']} | {fmt(row['threshold'])} | {fmt(row['f1'], 4)} | {int(row['pair_count'])} | "
            f"{fmt(row['mean_l2'])} | {fmt(row['p90_l2'])} | {fmt(row['small_l2'])} | {int(row['changed_predictions'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If this passes, it is a high-precision picking output protocol, not a new detector architecture.",
            "- If this fails, the next trained candidate should be a detached confidence/uncertainty calibrator only; do not reintroduce O2M/heatmap/quality loss stacking.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)
    alphas = parse_float_list(args.geometry_alphas)
    y_bands = parse_bands(args.y_bands)
    anchor_y_ratios = parse_float_list(args.anchor_y_ratios)
    protocols = build_protocols(y_bands, alphas, anchor_y_ratios)

    split_records: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    for split in ("valid", "test"):
        _, records = evaluate_split(
            args.config.resolve(),
            args.checkpoint.resolve(),
            split,
            args.dataset_root.resolve(),
            args.batch_size,
            args.num_workers,
            args.device,
            collect_predictions=True,
        )
        split_records[split] = records
        for protocol in protocols:
            transformed, transform_stats = transform_records(records, protocol)
            for threshold in thresholds:
                all_rows.append(metric_row(transformed, threshold, protocol, split, transform_stats))

    default_valid = next(
        row for row in all_rows
        if row["split"] == "valid" and row["protocol"] == "has_threshold" and abs(float(row["threshold"]) - 0.5) < 1e-9
    )
    selected_valid = select_valid_rule([row for row in all_rows if row["split"] == "valid"], default_valid, args)
    selected_test = find_matching_row(all_rows, selected_valid, "test")

    main_ref = flatten_reference("V7_EXP2_MAIN fair retrain", load_test_summary(args.main_summary.resolve()))
    base_ref = flatten_reference("EMA_BIFPN default", load_test_summary(args.base_summary.resolve()))
    gate = pass_gate(selected_test, safe_float(base_ref["AP"]), args)
    cases = summarize_cases(split_records["test"], selected_test)
    test_rows = [row for row in all_rows if row["split"] == "test"]
    near_miss_rows = sorted(
        [
            row for row in test_rows
            if safe_float(row.get("f1")) >= args.gate_min_f1 and int(row.get("pair_count", 0)) >= args.gate_min_pair
        ],
        key=lambda row: (
            safe_float(row.get("p90_l2"), float("inf")),
            safe_float(row.get("mean_l2"), float("inf")),
        ),
    )[:8]
    high_precision_rows = sorted(
        [row for row in test_rows if safe_float(row.get("p90_l2"), float("inf")) < args.gate_max_p90_l2],
        key=lambda row: (
            -int(row.get("pair_count", 0)),
            -safe_float(row.get("f1"), 0.0),
            safe_float(row.get("p90_l2"), float("inf")),
        ),
    )[:8]

    write_csv(args.output_dir / "hp_pick_calibration_all.csv", all_rows)
    write_csv(args.output_dir / "hp_pick_calibration_selected.csv", [selected_valid, selected_test])
    write_csv(args.output_dir / "hp_pick_calibration_near_miss.csv", near_miss_rows)
    write_csv(args.output_dir / "hp_pick_calibration_high_precision_subset.csv", high_precision_rows)
    write_csv(args.output_dir / "hp_pick_calibration_test_outliers.csv", cases)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "main_reference": main_ref,
        "base_reference": base_ref,
        "selected_valid": selected_valid,
        "selected_test": selected_test,
        "near_miss_rows": near_miss_rows,
        "high_precision_subset_rows": high_precision_rows,
        "gate": gate,
        "test_outliers": cases,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        args.output_dir / "hp_pick_calibration_report_zh.md",
        main_ref,
        base_ref,
        selected_valid,
        selected_test,
        gate,
        near_miss_rows,
        high_precision_rows,
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
