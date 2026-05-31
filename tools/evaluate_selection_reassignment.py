from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
    parser = argparse.ArgumentParser(description="Evaluate no-training picking point selection reassignment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--main-summary", type=Path, default=DEFAULT_MAIN_SUMMARY)
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE_SUMMARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/03_global_analysis/selection_reassignment_{datetime.now():%Y%m%d}",
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", default="0.50,0.52,0.54,0.62")
    parser.add_argument("--candidate-min-has", default="0.30,0.50")
    parser.add_argument("--y-bands", default="0.00:0.42")
    parser.add_argument("--x-margins", default="0.10")
    parser.add_argument("--anchor-y-ratios", default="0.12")
    parser.add_argument("--candidate-has-powers", default="1.0")
    parser.add_argument("--box-iou-powers", default="0.0")
    parser.add_argument("--reassign-margins", default="0.00")
    parser.add_argument("--min-valid-pair-ratio", type=float, default=0.98)
    parser.add_argument("--max-valid-f1-drop", type=float, default=0.01)
    parser.add_argument("--gate-min-ap", type=float, default=0.632)
    parser.add_argument("--gate-min-ap50", type=float, default=0.876)
    parser.add_argument("--gate-min-f1", type=float, default=0.760)
    parser.add_argument("--gate-min-pair", type=int, default=185)
    parser.add_argument("--gate-max-mean-l2", type=float, default=23.40)
    parser.add_argument("--gate-min-ppl-sr30", type=float, default=0.78)
    parser.add_argument("--gate-min-ppl-sr50", type=float, default=0.895)
    parser.add_argument("--legacy-diagnostic-gate", action="store_true")
    parser.add_argument("--gate-max-p90-l2", type=float, default=49.43)
    parser.add_argument("--gate-max-small-l2", type=float, default=33.81)
    parser.add_argument("--max-candidates-per-image", type=int, default=48)
    parser.add_argument("--refresh-records", action="store_true")
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


def load_test_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["primary_checkpoint_split_summary"]["test"]


def flatten_reference(label: str, summary: dict[str, Any]) -> dict[str, Any]:
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
        "precision": safe_float(has.get("precision")),
        "recall": safe_float(has.get("recall")),
        "pair_count": int(point.get("pair_count", 0) or 0),
        "mean_l2": safe_float(point.get("mean_l2_px")),
        "median_l2": safe_float(point.get("median_l2_px")),
        "p90_l2": safe_float(point.get("p90_l2_px")),
        "dx": safe_float(point.get("mean_abs_dx_px", point.get("mae_x_px"))),
        "dy": safe_float(point.get("mean_abs_dy_px", point.get("mae_y_px"))),
        "small_l2": safe_float((size.get("small") or {}).get("mean_l2_px")),
        "medium_l2": safe_float((size.get("medium") or {}).get("mean_l2_px")),
        "large_l2": safe_float((size.get("large") or {}).get("mean_l2_px")),
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
    return float(inter / union) if union > 0 else 0.0


def rel_point_to_box(point: list[float], box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return (float(point[0]) - x1) / w, (float(point[1]) - y1) / h


def top_roi_score(point: list[float], box: list[float], y_min: float, y_max: float, x_margin: float, anchor_y: float) -> float:
    rel_x, rel_y = rel_point_to_box(point, box)
    x_over = max(0.0, -x_margin - rel_x, rel_x - (1.0 + x_margin))
    y_over = max(0.0, y_min - rel_y, rel_y - y_max)
    anchor_dist = math.hypot((rel_x - 0.5) / 0.8, (rel_y - anchor_y) / 0.6)
    inside = 1.0 if x_over <= 1e-9 and y_over <= 1e-9 else 0.0
    soft = math.exp(-3.0 * x_over - 4.0 * y_over - 0.12 * anchor_dist)
    return float(max(0.0, min(1.0, (0.70 + 0.30 * inside) * soft)))


def protocol_name(protocol: dict[str, Any]) -> str:
    if protocol["kind"] == "raw":
        return "raw_has"
    return (
        f"reassign_y{protocol['y_min']:.2f}_{protocol['y_max']:.2f}"
        f"_x{protocol['x_margin']:.2f}_a{protocol['anchor_y']:.2f}"
        f"_h{protocol['candidate_min_has']:.2f}_hp{protocol['candidate_has_power']:.2f}"
        f"_bp{protocol['box_iou_power']:.2f}_m{protocol['reassign_margin']:.2f}"
    )


def build_protocols(args: argparse.Namespace) -> list[dict[str, Any]]:
    protocols = [{"kind": "raw", "name": "raw_has"}]
    for y_min, y_max in parse_bands(args.y_bands):
        for x_margin in parse_float_list(args.x_margins):
            for anchor_y in parse_float_list(args.anchor_y_ratios):
                for candidate_min_has in parse_float_list(args.candidate_min_has):
                    for candidate_has_power in parse_float_list(args.candidate_has_powers):
                        for box_iou_power in parse_float_list(args.box_iou_powers):
                            for reassign_margin in parse_float_list(args.reassign_margins):
                                protocol = {
                                    "kind": "reassign",
                                    "y_min": y_min,
                                    "y_max": y_max,
                                    "x_margin": x_margin,
                                    "anchor_y": anchor_y,
                                    "candidate_min_has": candidate_min_has,
                                    "candidate_has_power": candidate_has_power,
                                    "box_iou_power": box_iou_power,
                                    "reassign_margin": reassign_margin,
                                }
                                protocol["name"] = protocol_name(protocol)
                                protocols.append(protocol)
    return protocols


def candidate_score(anchor: dict[str, Any], candidate: dict[str, Any], protocol: dict[str, Any]) -> float:
    point = candidate.get("picking_point", [0.0, 0.0])
    anchor_box = anchor.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    cand_box = candidate.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    has = max(0.0, safe_float(candidate.get("has_picking_score", 0.0), 0.0))
    if has < protocol["candidate_min_has"]:
        return float("-inf")
    geom = top_roi_score(
        point,
        anchor_box,
        protocol["y_min"],
        protocol["y_max"],
        protocol["x_margin"],
        protocol["anchor_y"],
    )
    if geom <= 0.0:
        return float("-inf")
    iou = box_iou_xyxy(anchor_box, cand_box)
    has_term = has ** protocol["candidate_has_power"]
    iou_term = max(iou, 1e-6) ** protocol["box_iou_power"] if protocol["box_iou_power"] > 0 else 1.0
    return float(has_term * geom * iou_term)


def transform_records(
    records: list[dict[str, Any]], protocol: dict[str, Any], max_candidates_per_image: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = copy.deepcopy(records)
    changed = 0
    candidates_checked = 0
    if protocol["kind"] == "raw":
        for record in output:
            for pred in record.get("pred_instances", []):
                pred["selection_accept_score"] = float(pred.get("has_picking_score", 0.0))
                pred["reassigned_point_used"] = False
                pred["picking_point_source_query"] = int(-1)
        return output, {"changed_predictions": 0, "candidates_checked": 0}

    for record in output:
        preds = record.get("pred_instances", [])
        original_points = [list(pred.get("picking_point", [0.0, 0.0])) for pred in preds]
        candidate_indices = sorted(
            range(len(preds)),
            key=lambda idx: (
                safe_float(preds[idx].get("score", 0.0), 0.0)
                * max(safe_float(preds[idx].get("has_picking_score", 0.0), 0.0), 0.01)
            ),
            reverse=True,
        )[: max(1, int(max_candidates_per_image))]
        selected_points = []
        selected_sources = []
        for anchor_idx, anchor in enumerate(preds):
            anchor["selection_accept_score"] = float(anchor.get("has_picking_score", 0.0))
            best_idx = anchor_idx
            best_score = candidate_score(anchor, anchor, protocol)
            if anchor_idx not in candidate_indices:
                candidate_indices_for_anchor = [anchor_idx, *candidate_indices]
            else:
                candidate_indices_for_anchor = candidate_indices
            candidates_checked += len(candidate_indices_for_anchor)
            for cand_idx in candidate_indices_for_anchor:
                candidate = preds[cand_idx]
                score = candidate_score(anchor, candidate, protocol)
                if score > best_score * (1.0 + protocol["reassign_margin"]):
                    best_score = score
                    best_idx = cand_idx
            selected_points.append(list(original_points[best_idx]))
            selected_sources.append(best_idx)

        for idx, pred in enumerate(preds):
            source_idx = selected_sources[idx]
            pred["reassigned_point_used"] = bool(source_idx != idx)
            pred["picking_point_source_query"] = int(source_idx)
            pred["reassigned_point_score"] = float(candidate_score(pred, preds[source_idx], protocol))
            if source_idx != idx:
                changed += 1
                pred["picking_point"] = selected_points[idx]
    return output, {"changed_predictions": changed, "candidates_checked": candidates_checked}


def metric_row(records: list[dict[str, Any]], threshold: float, protocol: dict[str, Any], split: str, stats: dict[str, int]) -> dict[str, Any]:
    summary = summarize_split_error(records, threshold, visibility_score_key="selection_accept_score")
    correct, fp, fn = collect_case_groups(records, 0.5, threshold, visibility_score_key="selection_accept_score")
    tp = int(summary.get("has_picking_correct_count", 0))
    fp_count = int(summary.get("has_picking_false_positive_count", 0))
    fn_count = int(summary.get("has_picking_false_negative_count", 0))
    precision = tp / (tp + fp_count) if tp + fp_count else 0.0
    recall = tp / (tp + fn_count) if tp + fn_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pair = int(summary.get("point_pair_count", 0))
    l2_values = [float(item["l2_px"]) for item in correct if "l2_px" in item]
    size = summary.get("size_group_l2_px", {}) or {}
    reassigned_correct = sum(
        1
        for item in correct
        if item.get("pred_index") is not None
        and records_by_image_lookup(records, item["image_id"], item["pred_index"]).get("reassigned_point_used", False)
    )
    return {
        "split": split,
        "protocol": protocol["name"],
        "kind": protocol["kind"],
        "threshold": threshold,
        "candidate_min_has": protocol.get("candidate_min_has", ""),
        "y_min": protocol.get("y_min", ""),
        "y_max": protocol.get("y_max", ""),
        "x_margin": protocol.get("x_margin", ""),
        "anchor_y": protocol.get("anchor_y", ""),
        "candidate_has_power": protocol.get("candidate_has_power", ""),
        "box_iou_power": protocol.get("box_iou_power", ""),
        "reassign_margin": protocol.get("reassign_margin", ""),
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
        "changed_predictions": int(stats.get("changed_predictions", 0)),
        "candidates_checked": int(stats.get("candidates_checked", 0)),
        "reassigned_correct_pairs": int(reassigned_correct),
        "correct_case_count": len(correct),
        "fp_case_count": len(fp),
        "fn_case_count": len(fn),
    }


def records_by_image_lookup(records: list[dict[str, Any]], image_id: int, pred_idx: int) -> dict[str, Any]:
    for record in records:
        if int(record["image_id"]) == int(image_id):
            preds = record.get("pred_instances", [])
            if 0 <= int(pred_idx) < len(preds):
                return preds[int(pred_idx)]
    return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_valid_rule(valid_rows: list[dict[str, Any]], default_valid: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    min_pair = math.floor(int(default_valid["pair_count"]) * args.min_valid_pair_ratio)
    min_f1 = float(default_valid["f1"]) - float(args.max_valid_f1_drop)
    feasible = [
        row
        for row in valid_rows
        if int(row["pair_count"]) >= min_pair and float(row["f1"]) >= min_f1
    ] or valid_rows
    return min(
        feasible,
        key=lambda row: (
            safe_float(row.get("mean_l2"), float("inf")),
            -safe_float(row.get("ppl_sr_30"), 0.0),
            -safe_float(row.get("ppl_sr_50"), 0.0),
            -safe_float(row.get("pair_count"), 0.0),
            -safe_float(row.get("f1"), 0.0),
            safe_float(row.get("p90_l2"), float("inf")),
        ),
    )


def find_matching_row(rows: list[dict[str, Any]], selected: dict[str, Any], split: str) -> dict[str, Any]:
    keys = (
        "protocol",
        "kind",
        "threshold",
        "candidate_min_has",
        "y_min",
        "y_max",
        "x_margin",
        "anchor_y",
        "candidate_has_power",
        "box_iou_power",
        "reassign_margin",
    )
    for row in rows:
        if row["split"] != split:
            continue
        if all(str(row.get(key)) == str(selected.get(key)) for key in keys):
            return row
    raise KeyError(f"No {split} row for selected protocol {selected['protocol']}")


def pass_gate(row: dict[str, Any], base_ref: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    checks = {
        "AP": safe_float(base_ref.get("AP")) >= args.gate_min_ap,
        "AP50": safe_float(base_ref.get("AP50")) >= args.gate_min_ap50,
        "F1": safe_float(row["f1"]) >= args.gate_min_f1,
        "pair": int(row["pair_count"]) >= args.gate_min_pair,
        "mean_l2": safe_float(row["mean_l2"]) < args.gate_max_mean_l2,
        "ppl_sr_30": safe_float(row["ppl_sr_30"]) >= args.gate_min_ppl_sr30,
        "ppl_sr_50": safe_float(row["ppl_sr_50"]) >= args.gate_min_ppl_sr50,
    }
    if args.legacy_diagnostic_gate:
        checks["p90_l2"] = safe_float(row["p90_l2"]) < args.gate_max_p90_l2
        checks["small_l2"] = safe_float(row["small_l2"]) <= args.gate_max_small_l2
    checks["overall"] = all(checks.values())
    return checks


def fmt(value: Any, digits: int = 2) -> str:
    if value == "":
        return ""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(
    path: Path,
    main_ref: dict[str, Any],
    base_ref: dict[str, Any],
    selected_valid: dict[str, Any],
    selected_test: dict[str, Any],
    gate: dict[str, bool],
    near_miss: list[dict[str, Any]],
    high_precision: list[dict[str, Any]],
) -> None:
    lines = [
        "# Selection Reassignment V1 Offline Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "- Protocol: no-training query-level picking point reassignment; valid selects the rule, test only applies the frozen rule.",
        "- Boxes/scores/labels/has_picking_scores are not modified, so AP is inherited from the EMA_BIFPN base.",
        "- Goal: test whether better visible point candidates already exist in other same-image queries.",
        "",
        "## Same-Split Test Comparison",
        "",
        "| Model / Protocol | AP | F1 | pair | mean L2 | median L2 | p90 L2 | dx | dy | small L2 | PPL-SR@30 | PPL-SR@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| V7_EXP2_MAIN fair retrain | {fmt(main_ref['AP'], 4)} | {fmt(main_ref['f1'], 4)} | {int(main_ref['pair_count'])} | {fmt(main_ref['mean_l2'])} | {fmt(main_ref['median_l2'])} | {fmt(main_ref['p90_l2'])} | {fmt(main_ref['dx'])} | {fmt(main_ref['dy'])} | {fmt(main_ref['small_l2'])} | - | - |",
        f"| EMA_BIFPN default | {fmt(base_ref['AP'], 4)} | {fmt(base_ref['f1'], 4)} | {int(base_ref['pair_count'])} | {fmt(base_ref['mean_l2'])} | {fmt(base_ref['median_l2'])} | {fmt(base_ref['p90_l2'])} | {fmt(base_ref['dx'])} | {fmt(base_ref['dy'])} | {fmt(base_ref['small_l2'])} | - | - |",
        f"| Selection Reassignment V1 selected | {fmt(base_ref['AP'], 4)} | {fmt(selected_test['f1'], 4)} | {int(selected_test['pair_count'])} | {fmt(selected_test['mean_l2'])} | {fmt(selected_test['median_l2'])} | {fmt(selected_test['p90_l2'])} | {fmt(selected_test['dx'])} | {fmt(selected_test['dy'])} | {fmt(selected_test['small_l2'])} | {fmt(selected_test['ppl_sr_30'], 4)} | {fmt(selected_test['ppl_sr_50'], 4)} |",
        "",
        "## Selected Rule",
        "",
        "| Split | protocol | threshold | min_has | y_min | y_max | x_margin | anchor_y | F1 | pair | mean L2 | p90 L2 | changed preds | reassigned correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("valid_selected", selected_valid), ("test_fixed", selected_test)):
        lines.append(
            f"| {label} | {row['protocol']} | {fmt(row['threshold'])} | {fmt(row['candidate_min_has'])} | "
            f"{fmt(row['y_min'])} | {fmt(row['y_max'])} | {fmt(row['x_margin'])} | {fmt(row['anchor_y'])} | "
            f"{fmt(row['f1'], 4)} | {int(row['pair_count'])} | {fmt(row['mean_l2'])} | {fmt(row['p90_l2'])} | "
            f"{int(row['changed_predictions'])} | {int(row['reassigned_correct_pairs'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate Result",
            "",
            f"- Mainline gate: {'PASS' if gate['overall'] else 'FAIL'}",
            f"- AP>=0.632: {'PASS' if gate['AP'] else 'FAIL'}",
            f"- AP50>=0.876: {'PASS' if gate['AP50'] else 'FAIL'}",
            f"- F1>=0.760: {'PASS' if gate['F1'] else 'FAIL'}",
            f"- pair>=185: {'PASS' if gate['pair'] else 'FAIL'}",
            f"- mean<23.40: {'PASS' if gate['mean_l2'] else 'FAIL'}",
            f"- PPL-SR@30>=0.780: {'PASS' if gate['ppl_sr_30'] else 'FAIL'}",
            f"- PPL-SR@50>=0.895: {'PASS' if gate['ppl_sr_50'] else 'FAIL'}",
            f"- diagnostic p90<49.43: {'PASS' if gate.get('p90_l2', False) else 'not gated'}",
            f"- diagnostic small<=33.81: {'PASS' if gate.get('small_l2', False) else 'not gated'}",
            "",
            "## Coverage-Retaining Near Misses",
            "",
            "| protocol | threshold | F1 | pair | mean L2 | p90 L2 | small L2 | changed preds | reassigned correct |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in near_miss[:12]:
        lines.append(
            f"| {row['protocol']} | {fmt(row['threshold'])} | {fmt(row['f1'], 4)} | {int(row['pair_count'])} | "
            f"{fmt(row['mean_l2'])} | {fmt(row['p90_l2'])} | {fmt(row['small_l2'])} | "
            f"{int(row['changed_predictions'])} | {int(row['reassigned_correct_pairs'])} |"
        )
    lines.extend(
        [
            "",
            "## High-Precision Subsets",
            "",
            "| protocol | threshold | F1 | pair | mean L2 | p90 L2 | small L2 | changed preds | reassigned correct |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in high_precision[:12]:
        lines.append(
            f"| {row['protocol']} | {fmt(row['threshold'])} | {fmt(row['f1'], 4)} | {int(row['pair_count'])} | "
            f"{fmt(row['mean_l2'])} | {fmt(row['p90_l2'])} | {fmt(row['small_l2'])} | "
            f"{int(row['changed_predictions'])} | {int(row['reassigned_correct_pairs'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PASS means selection/reranking is a viable no-training picking-first protocol.",
            "- p90 and small-grape L2 are diagnostic risk indicators here, not mainline rejection criteria unless `--legacy-diagnostic-gate` is used.",
            "- FAIL with few useful reassigned correct pairs means the oracle gap is not recoverable by simple geometry-only reassignment; the next step should use richer prediction records or switch to coordinate-first refiner design.",
            "- This report does not claim a new trained model because no weights are changed.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)
    protocols = build_protocols(args)
    main_ref = flatten_reference("V7_EXP2_MAIN fair retrain", load_test_summary(args.main_summary))
    base_ref = flatten_reference("EMA_BIFPN default", load_test_summary(args.base_summary))

    all_rows: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("valid", "test"):
        cache_path = args.output_dir / f"{split}_prediction_records.json"
        if cache_path.exists() and not args.refresh_records:
            records = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"Loaded cached {split} prediction records: {cache_path}", flush=True)
        else:
            print(f"Evaluating {split} predictions once...", flush=True)
            _, records = evaluate_split(
                args.config,
                args.checkpoint,
                split,
                args.dataset_root,
                args.batch_size,
                args.num_workers,
                args.device,
                collect_predictions=True,
            )
            cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            print(f"Cached {split} prediction records: {cache_path}", flush=True)
        split_records[split] = records
        for protocol in protocols:
            print(f"{split}: scoring {protocol['name']}", flush=True)
            transformed, stats = transform_records(records, protocol, args.max_candidates_per_image)
            for threshold in thresholds:
                all_rows.append(metric_row(transformed, threshold, protocol, split, stats))

    default_valid = next(
        row
        for row in all_rows
        if row["split"] == "valid" and row["protocol"] == "raw_has" and abs(float(row["threshold"]) - 0.5) < 1e-9
    )
    selected_valid = select_valid_rule([row for row in all_rows if row["split"] == "valid"], default_valid, args)
    selected_test = find_matching_row(all_rows, selected_valid, "test")
    gate = pass_gate(selected_test, base_ref, args)

    near_miss = sorted(
        [
            row
            for row in all_rows
            if row["split"] == "test"
            and safe_float(row["f1"]) >= args.gate_min_f1
            and int(row["pair_count"]) >= args.gate_min_pair
        ],
        key=lambda row: (safe_float(row["p90_l2"], float("inf")), safe_float(row["mean_l2"], float("inf"))),
    )
    high_precision = sorted(
        [
            row
            for row in all_rows
            if row["split"] == "test"
            and safe_float(row["p90_l2"]) < args.gate_max_p90_l2
            and safe_float(row["mean_l2"]) < args.gate_max_mean_l2
        ],
        key=lambda row: (-int(row["pair_count"]), -safe_float(row["f1"]), safe_float(row["p90_l2"])),
    )

    write_csv(args.output_dir / "selection_reassignment_all.csv", all_rows)
    write_csv(args.output_dir / "selection_reassignment_selected.csv", [selected_valid, selected_test])
    write_csv(args.output_dir / "selection_reassignment_near_miss.csv", near_miss)
    write_csv(args.output_dir / "selection_reassignment_high_precision.csv", high_precision)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "Selection Reassignment V1",
        "main_reference": main_ref,
        "base_reference": base_ref,
        "selected_valid": selected_valid,
        "selected_test": selected_test,
        "gate": gate,
        "near_miss_top": near_miss[:20],
        "high_precision_top": high_precision[:20],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        args.output_dir / "selection_reassignment_report_zh.md",
        main_ref,
        base_ref,
        selected_valid,
        selected_test,
        gate,
        near_miss,
        high_precision,
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
