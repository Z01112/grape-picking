from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grape_point_eval_utils import match_prediction_record, normalize_prediction_record, safe_float


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/12_point_coordinate_diagnostics"
DEFAULT_EMA_VALID = REPO_ROOT / "outputs/08_eval_unification/ema_bifpn_unified_report/valid_prediction_records.json"
DEFAULT_EMA_TEST = REPO_ROOT / "outputs/08_eval_unification/ema_bifpn_unified_report/test_prediction_records.json"
DEFAULT_V7_TEST = REPO_ROOT / "outputs/08_eval_unification/v7_exp2_unified_report/test_prediction_records.json"
DEFAULT_VALID_ANN = REPO_ROOT / "dataset/valid/_annotations.grape_point.json"
DEFAULT_TEST_ANN = REPO_ROOT / "dataset/test/_annotations.grape_point.json"

ANCHORS = {
    "A0_current_top_center": (0.5, 0.12),
    "A1_top_left_ish": (0.35, 0.12),
    "A2_top_right_ish": (0.65, 0.12),
    "A3_upper_middle": (0.5, 0.35),
    "A4_bbox_center": (0.5, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose GPPoint-DETR picking point coordinate mechanism.")
    parser.add_argument("--ema-valid-records", type=Path, default=DEFAULT_EMA_VALID)
    parser.add_argument("--ema-test-records", type=Path, default=DEFAULT_EMA_TEST)
    parser.add_argument("--v7-test-records", type=Path, default=DEFAULT_V7_TEST)
    parser.add_argument("--valid-annotations", type=Path, default=DEFAULT_VALID_ANN)
    parser.add_argument("--test-annotations", type=Path, default=DEFAULT_TEST_ANN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-anchor-ratio", type=float, default=0.12)
    parser.add_argument("--has-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported prediction records payload: {path}")
    return [normalize_prediction_record(item) for item in payload]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def rel_point(point: list[float], box_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return (float(point[0]) - x1) / w, (float(point[1]) - y1) / h


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def quantile_stats(values: list[float]) -> dict:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {key: 0.0 for key in ["count", "mean", "std", "median", "q10", "q25", "q75", "q90", "min", "max"]}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def corr(x_values: list[float], y_values: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return 0.0
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def assign_area_groups(rows: list[dict]) -> tuple[float, float]:
    areas = np.asarray([safe_float(row.get("area", 0.0)) for row in rows], dtype=np.float64)
    if areas.size == 0:
        return 0.0, 0.0
    q1 = float(np.quantile(areas, 1.0 / 3.0))
    q2 = float(np.quantile(areas, 2.0 / 3.0))
    for row in rows:
        area = safe_float(row.get("area", 0.0))
        if area <= q1:
            row["area_group"] = "small"
        elif area <= q2:
            row["area_group"] = "medium"
        else:
            row["area_group"] = "large"
    return q1, q2


def area_group(area: float, q1: float, q2: float) -> str:
    if area <= q1:
        return "small"
    if area <= q2:
        return "medium"
    return "large"


def gt_region(rel_x: float, rel_y: float) -> str:
    labels = []
    if -0.15 <= rel_y <= 0.30:
        labels.append("top_band")
    if 0.30 < rel_y <= 0.55:
        labels.append("upper_middle")
    if rel_x < 0.0 or rel_x > 1.0:
        labels.append("side_outside_x")
    if rel_y > 0.55:
        labels.append("below_middle")
    if not labels:
        labels.append("other")
    return ";".join(labels)


def anchor_residuals(rel_x: float, rel_y: float, top_anchor_ratio: float) -> dict:
    anchors = dict(ANCHORS)
    anchors["A0_current_top_center"] = (0.5, float(top_anchor_ratio))
    out = {}
    best_name = None
    best_norm = None
    for name, (ax, ay) in anchors.items():
        dx = rel_x - ax
        dy = rel_y - ay
        norm = math.hypot(dx, dy)
        out[f"{name}_dx"] = float(dx)
        out[f"{name}_dy"] = float(dy)
        out[f"{name}_residual_norm"] = float(norm)
        if best_norm is None or norm < best_norm:
            best_norm = float(norm)
            best_name = name
    out["best_anchor"] = best_name
    out["best_of_5_residual_norm"] = float(best_norm if best_norm is not None else 0.0)
    return out


def load_coco_gt_rows(path: Path, split: str, top_anchor_ratio: float) -> tuple[list[dict], tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in data.get("images", [])}
    rows = []
    for ann in data.get("annotations", []):
        if safe_float(ann.get("has_picking", 0.0)) <= 0.5:
            continue
        bbox_xywh = [float(v) for v in ann.get("bbox", [0.0, 0.0, 0.0, 0.0])]
        point = ann.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        rel_x, rel_y = rel_point(point, bbox_xyxy)
        dx_top = rel_x - 0.5
        dy_top = rel_y - float(top_anchor_ratio)
        residual = anchor_residuals(rel_x, rel_y, top_anchor_ratio)
        image = image_by_id.get(int(ann.get("image_id", -1)), {})
        area = safe_float(ann.get("area", bbox_xywh[2] * bbox_xywh[3]))
        rows.append(
            {
                "split": split,
                "image_id": int(ann.get("image_id", -1)),
                "file_name": image.get("file_name", ""),
                "annotation_id": int(ann.get("id", -1)),
                "bbox_x": bbox_xywh[0],
                "bbox_y": bbox_xywh[1],
                "bbox_w": bbox_xywh[2],
                "bbox_h": bbox_xywh[3],
                "area": area,
                "gt_point_x": float(point[0]),
                "gt_point_y": float(point[1]),
                "rel_x": float(rel_x),
                "rel_y": float(rel_y),
                "top_center_dx": float(dx_top),
                "top_center_dy": float(dy_top),
                "region": gt_region(rel_x, rel_y),
                "in_top_band": -0.15 <= rel_y <= 0.30,
                "in_upper_middle": 0.30 < rel_y <= 0.55,
                "side_or_outside_x": rel_x < 0.0 or rel_x > 1.0,
                "below_middle": rel_y > 0.55,
                **residual,
            }
        )
    thresholds = assign_area_groups(rows)
    return rows, thresholds


def summarize_gt_distribution(rows: list[dict]) -> dict:
    summary = {
        "count": len(rows),
        "rel_x": quantile_stats([row["rel_x"] for row in rows]),
        "rel_y": quantile_stats([row["rel_y"] for row in rows]),
        "top_center_dx": quantile_stats([row["top_center_dx"] for row in rows]),
        "top_center_dy": quantile_stats([row["top_center_dy"] for row in rows]),
        "regions": dict(Counter(part for row in rows for part in str(row["region"]).split(";"))),
        "area_groups": {},
    }
    for group in ["small", "medium", "large"]:
        group_rows = [row for row in rows if row.get("area_group") == group]
        summary["area_groups"][group] = {
            "count": len(group_rows),
            "rel_x": quantile_stats([row["rel_x"] for row in group_rows]),
            "rel_y": quantile_stats([row["rel_y"] for row in group_rows]),
            "current_anchor_residual": quantile_stats([row["A0_current_top_center_residual_norm"] for row in group_rows]),
            "best_of_5_residual": quantile_stats([row["best_of_5_residual_norm"] for row in group_rows]),
        }
    return summary


def summarize_anchor_coverage(rows: list[dict], split: str) -> tuple[list[dict], dict]:
    output_rows = []
    groups = ["all", "small", "medium", "large"]
    summary = {}
    for group in groups:
        group_rows = rows if group == "all" else [row for row in rows if row.get("area_group") == group]
        current = [row["A0_current_top_center_residual_norm"] for row in group_rows]
        best = [row["best_of_5_residual_norm"] for row in group_rows]
        current_stats = quantile_stats(current)
        best_stats = quantile_stats(best)
        best_counts = Counter(row["best_anchor"] for row in group_rows)
        row = {
            "split": split,
            "area_group": group,
            "count": len(group_rows),
            "current_mean": current_stats["mean"],
            "current_median": current_stats["median"],
            "current_p90": current_stats["q90"],
            "best_of_5_mean": best_stats["mean"],
            "best_of_5_median": best_stats["median"],
            "best_of_5_p90": best_stats["q90"],
            "p90_reduction_abs": current_stats["q90"] - best_stats["q90"],
            "p90_reduction_ratio": (current_stats["q90"] - best_stats["q90"]) / current_stats["q90"] if current_stats["q90"] > 0 else 0.0,
            "current_residual_gt_0.4_rate": float(np.mean(np.asarray(current) > 0.4)) if current else 0.0,
            "best_of_5_residual_gt_0.4_rate": float(np.mean(np.asarray(best) > 0.4)) if best else 0.0,
        }
        for anchor_name in ANCHORS.keys():
            row[f"best_anchor_rate_{anchor_name}"] = best_counts.get(anchor_name, 0) / len(group_rows) if group_rows else 0.0
        output_rows.append(row)
        summary[group] = row
    return output_rows, summary


def build_gt_lookup(rows: list[dict]) -> dict[tuple[int, int], dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(row)
    # Records preserve annotation order from COCO in this dataset path.
    return {(image_id, idx): row for image_id, items in grouped.items() for idx, row in enumerate(items)}


def matched_error_rows(records: list[dict], split: str, model: str, gt_lookup: dict[tuple[int, int], dict], area_thresholds: tuple[float, float], top_anchor_ratio: float, has_threshold: float, iou_threshold: float) -> list[dict]:
    rows = []
    q1, q2 = area_thresholds
    for record in records:
        matched = match_prediction_record(record, iou_threshold=iou_threshold, has_picking_threshold=has_threshold)
        for case in matched["correct_visible_pairs"]:
            gt_idx = int(case["gt_index"])
            pred_idx = int(case["pred_index"])
            gt = record["gt_instances"][gt_idx]
            pred = record["pred_instances"][pred_idx]
            gt_box = [float(v) for v in gt["bbox_xyxy"]]
            pred_box = [float(v) for v in pred["bbox_xyxy"]]
            gt_point = [float(v) for v in gt["picking_point"]]
            pred_point = [float(v) for v in pred["picking_point"]]
            gt_rel_x, gt_rel_y = rel_point(gt_point, gt_box)
            pred_rel_x, pred_rel_y = rel_point(pred_point, pred_box)
            pred_rel_gt_x, pred_rel_gt_y = rel_point(pred_point, gt_box)
            gt_cx, gt_cy = box_center(gt_box)
            pred_cx, pred_cy = box_center(pred_box)
            gt_w = max(1e-6, gt_box[2] - gt_box[0])
            gt_h = max(1e-6, gt_box[3] - gt_box[1])
            pred_w = max(1e-6, pred_box[2] - pred_box[0])
            pred_h = max(1e-6, pred_box[3] - pred_box[1])
            center_err_px = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
            center_err_norm = math.hypot((pred_cx - gt_cx) / gt_w, (pred_cy - gt_cy) / gt_h)
            size_err_w = abs(pred_w - gt_w) / gt_w
            size_err_h = abs(pred_h - gt_h) / gt_h
            area = safe_float(gt.get("area", gt_w * gt_h))
            residual = anchor_residuals(gt_rel_x, gt_rel_y, top_anchor_ratio)
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "image_id": int(record["image_id"]),
                    "file_name": record.get("file_name", ""),
                    "gt_index": gt_idx,
                    "pred_index": pred_idx,
                    "iou": safe_float(case.get("iou", 0.0)),
                    "score": safe_float(pred.get("score", 0.0)),
                    "visible_score": safe_float(pred.get("visible_score", pred.get("has_picking_score", 0.0))),
                    "gt_area": area,
                    "area_group": area_group(area, q1, q2),
                    "gt_rel_x": gt_rel_x,
                    "gt_rel_y": gt_rel_y,
                    "pred_rel_x_pred_box": pred_rel_x,
                    "pred_rel_y_pred_box": pred_rel_y,
                    "pred_rel_x_gt_box": pred_rel_gt_x,
                    "pred_rel_y_gt_box": pred_rel_gt_y,
                    "point_dx_px": safe_float(case.get("dx_px", 0.0)),
                    "point_dy_px": safe_float(case.get("dy_px", 0.0)),
                    "point_l2_px": safe_float(case.get("l2_px", 0.0)),
                    "norm_abs_dx_gt_w": abs(safe_float(case.get("dx_px", 0.0))) / gt_w,
                    "norm_abs_dy_gt_h": abs(safe_float(case.get("dy_px", 0.0))) / gt_h,
                    "box_center_error_px": center_err_px,
                    "box_center_error_norm": center_err_norm,
                    "box_size_error_w_norm": size_err_w,
                    "box_size_error_h_norm": size_err_h,
                    "l2_gt_30": safe_float(case.get("l2_px", 0.0)) > 30.0,
                    "l2_gt_50": safe_float(case.get("l2_px", 0.0)) > 50.0,
                    "high_iou_point_bad": safe_float(case.get("iou", 0.0)) >= 0.85 and safe_float(case.get("l2_px", 0.0)) > 30.0,
                    "low_iou_propagation": 0.5 <= safe_float(case.get("iou", 0.0)) < 0.7 and safe_float(case.get("l2_px", 0.0)) > 30.0,
                    "current_anchor_residual": residual["A0_current_top_center_residual_norm"],
                    "best_anchor": residual["best_anchor"],
                    "best_of_5_residual": residual["best_of_5_residual_norm"],
                    "current_anchor_dx": residual["A0_current_top_center_dx"],
                    "current_anchor_dy": residual["A0_current_top_center_dy"],
                    "gt_region": gt_region(gt_rel_x, gt_rel_y),
                }
            )
    return rows


def summarize_box_error(rows: list[dict], split: str, model: str) -> tuple[list[dict], dict]:
    groups = {
        "all": rows,
        "good_l2_le_30": [row for row in rows if not row["l2_gt_30"]],
        "bad_l2_gt_30": [row for row in rows if row["l2_gt_30"]],
        "bad_l2_gt_50": [row for row in rows if row["l2_gt_50"]],
        "high_iou_ge_085": [row for row in rows if row["iou"] >= 0.85],
        "low_iou_050_070": [row for row in rows if 0.5 <= row["iou"] < 0.7],
    }
    output = []
    for name, group_rows in groups.items():
        row = {
            "split": split,
            "model": model,
            "group": name,
            "count": len(group_rows),
            "mean_l2": quantile_stats([item["point_l2_px"] for item in group_rows])["mean"],
            "p90_l2": quantile_stats([item["point_l2_px"] for item in group_rows])["q90"],
            "mean_iou": quantile_stats([item["iou"] for item in group_rows])["mean"],
            "mean_box_center_error_norm": quantile_stats([item["box_center_error_norm"] for item in group_rows])["mean"],
            "mean_size_error_w_norm": quantile_stats([item["box_size_error_w_norm"] for item in group_rows])["mean"],
            "mean_size_error_h_norm": quantile_stats([item["box_size_error_h_norm"] for item in group_rows])["mean"],
        }
        output.append(row)
    summary = {
        "corr_iou_l2": corr([row["iou"] for row in rows], [row["point_l2_px"] for row in rows]),
        "corr_center_error_l2": corr([row["box_center_error_norm"] for row in rows], [row["point_l2_px"] for row in rows]),
        "corr_size_w_error_l2": corr([row["box_size_error_w_norm"] for row in rows], [row["point_l2_px"] for row in rows]),
        "corr_size_h_error_l2": corr([row["box_size_error_h_norm"] for row in rows], [row["point_l2_px"] for row in rows]),
    }
    return output, summary


def taxonomy_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    bad_rows = [row for row in rows if row["l2_gt_30"]]
    residual_threshold = float(np.quantile([row["current_anchor_residual"] for row in rows], 0.75)) if rows else 0.0
    output = []
    counts = Counter()
    for row in rows:
        labels = []
        if row["point_l2_px"] <= 30.0:
            labels.append("good_success")
        else:
            if row["low_iou_propagation"]:
                labels.append("low_iou_propagation")
            if row["high_iou_point_bad"]:
                labels.append("high_iou_point_bad")
            if row["current_anchor_residual"] >= residual_threshold:
                labels.append("anchor_hard_case")
            if row["area_group"] == "small":
                labels.append("small_grape_hard_case")
            if row["gt_rel_x"] < 0.0 or row["gt_rel_x"] > 1.0 or row["gt_rel_y"] > 0.55:
                labels.append("side_or_outside_case")
            if not labels:
                labels.append("general_point_regression_bad")
        for label in labels:
            counts[label] += 1
        output.append(
            {
                **row,
                "anchor_hard_threshold": residual_threshold,
                "taxonomy_label": ";".join(labels),
            }
        )
    bad_count = len(bad_rows)
    summary = {
        "total_pairs": len(rows),
        "l2_gt_30_count": bad_count,
        "l2_gt_50_count": sum(1 for row in rows if row["l2_gt_50"]),
        "taxonomy_counts": dict(counts),
        "anchor_hard_threshold": residual_threshold,
        "anchor_or_side_bad_rate": (
            sum(1 for row in output if row["point_l2_px"] > 30.0 and ("anchor_hard_case" in row["taxonomy_label"] or "side_or_outside_case" in row["taxonomy_label"])) / bad_count
            if bad_count else 0.0
        ),
        "high_iou_point_bad_rate": counts.get("high_iou_point_bad", 0) / bad_count if bad_count else 0.0,
        "low_iou_propagation_rate": counts.get("low_iou_propagation", 0) / bad_count if bad_count else 0.0,
        "small_grape_hard_rate": counts.get("small_grape_hard_case", 0) / bad_count if bad_count else 0.0,
    }
    return output, summary


def choose_conclusion(anchor_summary: dict, taxonomy_summary: dict, gt_summary: dict) -> tuple[str, str]:
    all_anchor = anchor_summary["test"]["all"]
    p90_reduction = safe_float(all_anchor["p90_reduction_ratio"])
    current_gt04 = safe_float(all_anchor["current_residual_gt_0.4_rate"])
    best_gt04 = safe_float(all_anchor["best_of_5_residual_gt_0.4_rate"])
    high_iou_rate = safe_float(taxonomy_summary["high_iou_point_bad_rate"])
    anchor_or_side_rate = safe_float(taxonomy_summary["anchor_or_side_bad_rate"])
    low_iou_rate = safe_float(taxonomy_summary["low_iou_propagation_rate"])
    regions = gt_summary["test"]["regions"]
    extreme_rate = (
        (regions.get("side_outside_x", 0) + regions.get("below_middle", 0)) / gt_summary["test"]["count"]
        if gt_summary["test"]["count"] else 0.0
    )
    if p90_reduction >= 0.20 and anchor_or_side_rate >= 0.35 and high_iou_rate >= 0.15:
        return (
            "A. 可以做最后一个结构实验：multi-anchor point offset head",
            "best-of-5 anchor residual p90 明显低于 current top-center，且 L2>30 中 anchor/side 难例与 high-IoU 坏点占比都不可忽略，说明单一 top-center anchor 表达是有效瓶颈。",
        )
    if extreme_rate >= 0.35 and safe_float(taxonomy_summary["small_grape_hard_rate"]) >= 0.35:
        return (
            "C. 需要数据/标注层面清理",
            "GT point 极端位置与 small/side/outside 难例占比较高，继续改结构前应先检查采摘点标注语义一致性。",
        )
    if p90_reduction < 0.12 or low_iou_rate >= 0.45:
        return (
            "B. 不建议改坐标结构，优先收束论文",
            "current anchor 覆盖改善空间有限，或 L2>30 主要来自低 IoU box propagation；继续做 multi-anchor 结构收益不确定。",
        )
    if p90_reduction < 0.20:
        return (
            "B. 不建议改坐标结构，优先收束论文",
            f"L2>30 中 anchor/side 难例和 high-IoU 坏点确实不少，但 test best-of-5 anchor residual p90 只降低 {p90_reduction:.2%}，未达到 20% 的明显改善门槛；multi-anchor 可能解释一部分 outlier，但不足以支撑最后一个结构实验。",
        )
    return (
        "B. 不建议改坐标结构，优先收束论文",
        "诊断证据不足以支持最后一个结构实验；multi-anchor coverage 有一定改善但没有同时满足 anchor-hard 与 high-IoU 坏点条件。",
    )


def write_report(path: Path, summary: dict) -> None:
    conclusion = summary["diagnostic_conclusion"]
    test_anchor = summary["anchor_coverage"]["test"]["all"]
    tax = summary["ema_test_taxonomy"]
    test_gt = summary["gt_distribution"]["test"]
    ema_test = summary["ema_test_error_summary"]
    lines = [
        "# Point Coordinate Mechanism Diagnostic",
        "",
        "本报告只做点坐标生成机制诊断：不训练、不生成 checkpoint、不改 decoder/head/loss/matcher。",
        "",
        "## GT Offset Distribution",
        "",
        f"- visible GT test count: {test_gt['count']}",
        f"- rel_x median/q10/q90: {test_gt['rel_x']['median']:.4f} / {test_gt['rel_x']['q10']:.4f} / {test_gt['rel_x']['q90']:.4f}",
        f"- rel_y median/q10/q90: {test_gt['rel_y']['median']:.4f} / {test_gt['rel_y']['q10']:.4f} / {test_gt['rel_y']['q90']:.4f}",
        f"- regions: {json.dumps(test_gt['regions'], ensure_ascii=False)}",
        "",
        "## Anchor Coverage",
        "",
        "| Split | current p90 | best-of-5 p90 | p90 reduction | current >0.4 | best >0.4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["valid", "test"]:
        row = summary["anchor_coverage"][split]["all"]
        lines.append(
            f"| {split} | {row['current_p90']:.4f} | {row['best_of_5_p90']:.4f} | {row['p90_reduction_ratio']:.2%} | {row['current_residual_gt_0.4_rate']:.2%} | {row['best_of_5_residual_gt_0.4_rate']:.2%} |"
        )
    lines += [
        "",
        "## EMA_BIFPN Test Point Error",
        "",
        f"- matched visible pairs: {ema_test['count']}",
        f"- L2>30 count: {tax['l2_gt_30_count']}",
        f"- L2>50 count: {tax['l2_gt_50_count']}",
        f"- high_iou_point_bad rate among L2>30: {tax['high_iou_point_bad_rate']:.2%}",
        f"- low_iou_propagation rate among L2>30: {tax['low_iou_propagation_rate']:.2%}",
        f"- anchor_or_side bad rate among L2>30: {tax['anchor_or_side_bad_rate']:.2%}",
        "",
        "## Outlier Taxonomy",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(tax["taxonomy_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines += [
        "",
        "## 诊断结论",
        "",
        f"**{conclusion['label']}**",
        "",
        conclusion["reason"],
        "",
        "## 必答问题",
        "",
        f"1. 当前 top-center anchor 是否合理？  {'不完全合理；best-of-5 明显降低 residual。' if test_anchor['p90_reduction_ratio'] >= 0.20 else '基本可用；best-of-5 没有形成足够大的 residual p90 改善。'}",
        f"2. single-anchor offset 是否是主要瓶颈？  {'是重要瓶颈。' if conclusion['label'].startswith('A.') else '不是当前证据支持的唯一主瓶颈。'}",
        f"3. 是否值得做最后一个 multi-anchor point offset 实验？  {'值得，只做这一个最小结构实验。' if conclusion['label'].startswith('A.') else '不建议继续开这个结构实验。'}",
        f"4. 如果不值得，论文最终主模型应选 EMA_BIFPN 还是 V7_EXP2_MAIN？  {'若不做结构实验，优先选 EMA_BIFPN；它在统一口径下 pair/F1/AP/mean L2 更稳。' if not conclusion['label'].startswith('A.') else '先不定稿；可用 EMA_BIFPN 作为 multi-anchor 最小实验底座。'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid_gt_rows, valid_area_thresholds = load_coco_gt_rows(args.valid_annotations, "valid", args.top_anchor_ratio)
    test_gt_rows, test_area_thresholds = load_coco_gt_rows(args.test_annotations, "test", args.top_anchor_ratio)
    all_gt_rows = valid_gt_rows + test_gt_rows
    write_csv(args.output_dir / "gt_offset_distribution.csv", all_gt_rows)

    valid_anchor_rows, valid_anchor_summary = summarize_anchor_coverage(valid_gt_rows, "valid")
    test_anchor_rows, test_anchor_summary = summarize_anchor_coverage(test_gt_rows, "test")
    write_csv(args.output_dir / "anchor_coverage_analysis.csv", valid_anchor_rows + test_anchor_rows)

    ema_valid_records = load_records(args.ema_valid_records)
    ema_test_records = load_records(args.ema_test_records)
    ema_valid_rows = matched_error_rows(
        ema_valid_records, "valid", "EMA_BIFPN", build_gt_lookup(valid_gt_rows), valid_area_thresholds, args.top_anchor_ratio, args.has_threshold, args.iou_threshold
    )
    ema_test_rows = matched_error_rows(
        ema_test_records, "test", "EMA_BIFPN", build_gt_lookup(test_gt_rows), test_area_thresholds, args.top_anchor_ratio, args.has_threshold, args.iou_threshold
    )
    pred_rows = ema_valid_rows + ema_test_rows
    if args.v7_test_records.exists():
        v7_test_records = load_records(args.v7_test_records)
        pred_rows.extend(
            matched_error_rows(
                v7_test_records, "test", "V7_EXP2_MAIN_REF", build_gt_lookup(test_gt_rows), test_area_thresholds, args.top_anchor_ratio, args.has_threshold, args.iou_threshold
            )
        )
    write_csv(args.output_dir / "pred_offset_error_by_case.csv", pred_rows)

    box_rows_valid, box_summary_valid = summarize_box_error(ema_valid_rows, "valid", "EMA_BIFPN")
    box_rows_test, box_summary_test = summarize_box_error(ema_test_rows, "test", "EMA_BIFPN")
    write_csv(args.output_dir / "box_error_to_point_error.csv", box_rows_valid + box_rows_test)

    tax_rows, tax_summary = taxonomy_rows(ema_test_rows)
    write_csv(args.output_dir / "outlier_case_taxonomy.csv", tax_rows)

    gt_summary = {
        "valid": summarize_gt_distribution(valid_gt_rows),
        "test": summarize_gt_distribution(test_gt_rows),
    }
    anchor_summary = {
        "valid": valid_anchor_summary,
        "test": test_anchor_summary,
    }
    conclusion_label, conclusion_reason = choose_conclusion(anchor_summary, tax_summary, gt_summary)
    summary = {
        "inputs": {
            "ema_valid_records": str(args.ema_valid_records),
            "ema_test_records": str(args.ema_test_records),
            "valid_annotations": str(args.valid_annotations),
            "test_annotations": str(args.test_annotations),
            "v7_test_records": str(args.v7_test_records) if args.v7_test_records.exists() else None,
        },
        "top_anchor_ratio": args.top_anchor_ratio,
        "gt_distribution": gt_summary,
        "anchor_coverage": anchor_summary,
        "ema_valid_error_summary": {
            "count": len(ema_valid_rows),
            "box_error_correlations": box_summary_valid,
        },
        "ema_test_error_summary": {
            "count": len(ema_test_rows),
            "box_error_correlations": box_summary_test,
        },
        "ema_test_taxonomy": tax_summary,
        "diagnostic_conclusion": {
            "label": conclusion_label,
            "reason": conclusion_reason,
        },
    }
    write_json(args.output_dir / "point_coordinate_mechanism_summary.json", summary)
    write_report(args.output_dir / "point_coordinate_mechanism_summary.md", summary)
    print(f"Wrote point coordinate diagnostics to {args.output_dir}")
    print(conclusion_label)


if __name__ == "__main__":
    main()
