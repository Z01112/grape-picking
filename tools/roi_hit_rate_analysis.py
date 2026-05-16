from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GT = Path("dataset/test/_annotations.grape_point.json")
DEFAULT_REPORT = Path("reports/roi_hit_rate_analysis_zh.md")
DEFAULT_CSV = Path("reports/roi_hit_rate.csv")

ROI_WIDTH_SCALE = 1.08
ROI_Y_MIN = -0.10
ROI_Y_MAX = 0.40


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def normalize_xyxy(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box1: list[float], box2: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box1) + box_area(box2) - inter
    return float(inter / union) if union > 0.0 else 0.0


def box_edge_gap(box1: list[float], box2: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    gap_x = max(ax1 - bx2, bx1 - ax2, 0.0)
    gap_y = max(ay1 - by2, by1 - ay2, 0.0)
    return float(math.hypot(gap_x, gap_y))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean_or_blank(values: list[float]) -> str:
    return "" if not values else f"{sum(values) / len(values):.6f}"


def load_gt(gt_path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        bbox_xyxy = xywh_to_xyxy(ann.get("bbox", [0, 0, 0, 0]))
        point = ann.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
        gt_by_image[image_id].append(
            {
                "ann_id": int(ann.get("id", len(gt_by_image[image_id]))),
                "image_id": image_id,
                "bbox_xyxy": normalize_xyxy(bbox_xyxy),
                "area": float(ann.get("area", box_area(bbox_xyxy))),
                "has_picking": float(ann.get("has_picking", 0.0)) > 0.5,
                "picking_point": [float(point[0]), float(point[1])],
            }
        )

    records = []
    for image_id in sorted(images):
        image = images[image_id]
        records.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name", ""),
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "gt_instances": gt_by_image.get(image_id, []),
                "pred_instances": [],
            }
        )
    return records, images, gt_by_image


def value_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return list(obj)


def extract_point(item: dict[str, Any]) -> list[float] | None:
    for key in ("picking_point", "picking_points", "point", "pred_point"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    return None


def extract_has_score(item: dict[str, Any]) -> float | None:
    for key in ("has_picking_score", "picking_score", "pred_has_picking_score"):
        if key in item:
            return float(item[key])
    if "has_picking" in item:
        return float(item["has_picking"])
    return None


def prediction_box(item: dict[str, Any], bbox_format: str) -> list[float] | None:
    if "bbox_xyxy" in item:
        return normalize_xyxy([float(v) for v in item["bbox_xyxy"]])
    if "box_xyxy" in item:
        return normalize_xyxy([float(v) for v in item["box_xyxy"]])
    if "bbox" not in item:
        return None
    raw = [float(v) for v in item["bbox"]]
    if len(raw) != 4:
        return None
    if bbox_format == "xyxy":
        return normalize_xyxy(raw)
    return normalize_xyxy(xywh_to_xyxy(raw))


def parse_flat_predictions(items: list[dict[str, Any]], bbox_format: str) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if "image_id" not in item:
            continue
        box = prediction_box(item, bbox_format)
        if box is None:
            continue
        pred = {
            "bbox_xyxy": box,
            "score": float(item.get("score", item.get("confidence", 1.0))),
            "category_id": item.get("category_id"),
            "has_picking_score": extract_has_score(item),
            "picking_point": extract_point(item),
        }
        by_image[int(item["image_id"])].append(pred)
    return by_image


def parse_batched_prediction_dict(data: dict[str, Any], bbox_format: str) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_entries = data.get("predictions", data.get("results", data.get("detections")))
    if isinstance(image_entries, list):
        return parse_flat_predictions(image_entries, bbox_format)

    for image_key, output in data.items():
        if not isinstance(output, dict):
            continue
        if not {"boxes", "scores"}.issubset(output.keys()):
            continue
        image_id = int(output.get("image_id", image_key))
        boxes = value_list(output.get("boxes"))
        scores = value_list(output.get("scores"))
        has_scores = value_list(output.get("has_picking_scores"))
        points = value_list(output.get("picking_points"))
        for idx, box in enumerate(boxes):
            item = {
                "image_id": image_id,
                "bbox": box,
                "score": scores[idx] if idx < len(scores) else 1.0,
            }
            if idx < len(has_scores):
                item["has_picking_score"] = has_scores[idx]
            if idx < len(points):
                item["picking_point"] = points[idx]
            parsed = parse_flat_predictions([item], bbox_format)
            by_image[image_id].extend(parsed.get(image_id, []))
    return by_image


def parse_prediction_file(pred_path: Path, bbox_format: str) -> tuple[dict[int, list[dict[str, Any]]], bool]:
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[0], dict) and "gt_instances" in data[0]:
        pred_by_image = {}
        for record in data:
            pred_by_image[int(record["image_id"])] = [
                {
                    "bbox_xyxy": normalize_xyxy([float(v) for v in pred["bbox_xyxy"]]),
                    "score": float(pred.get("score", 1.0)),
                    "has_picking_score": extract_has_score(pred),
                    "picking_point": extract_point(pred),
                }
                for pred in record.get("pred_instances", [])
                if "bbox_xyxy" in pred
            ]
        return pred_by_image, True

    if isinstance(data, list):
        return parse_flat_predictions(data, bbox_format), False
    if isinstance(data, dict):
        for key in ("predictions", "results", "detections", "annotations"):
            if isinstance(data.get(key), list):
                return parse_flat_predictions(data[key], bbox_format), False
        return parse_batched_prediction_dict(data, bbox_format), False
    return {}, False


def enrich_groups(records: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, str]]:
    visible_meta: dict[tuple[int, int], dict[str, Any]] = {}
    areas = []
    for record in records:
        gt_entries = record["gt_instances"]
        total = len(gt_entries)
        boxes = [gt["bbox_xyxy"] for gt in gt_entries]
        for gt_idx, gt in enumerate(gt_entries):
            if not gt["has_picking"]:
                continue
            key = (int(record["image_id"]), int(gt_idx))
            area = float(gt["area"])
            areas.append(area)
            max_neighbor_iou = 0.0
            min_gap_norm = float("inf")
            for other_idx, other_box in enumerate(boxes):
                if other_idx == gt_idx:
                    continue
                max_neighbor_iou = max(max_neighbor_iou, box_iou(gt["bbox_xyxy"], other_box))
                min_gap_norm = min(min_gap_norm, box_edge_gap(gt["bbox_xyxy"], other_box) / max(math.sqrt(max(area, 1.0)), 1.0))
            visible_meta[key] = {
                "area": area,
                "total_grape_count": total,
                "max_neighbor_iou": max_neighbor_iou,
                "min_gap_norm": min_gap_norm if math.isfinite(min_gap_norm) else float("inf"),
            }

    area_threshold = quantile(areas, 1.0 / 3.0)
    crowd_iou_threshold = 0.05
    crowd_gap_threshold = 0.20
    for key, meta in visible_meta.items():
        meta["size_group"] = "small" if float(meta["area"]) <= area_threshold else "medium_large"
        meta["single_multi"] = "single" if int(meta["total_grape_count"]) <= 1 else "multi_adjacent"
        heavy = int(meta["total_grape_count"]) > 1 and (
            float(meta["max_neighbor_iou"]) >= crowd_iou_threshold or float(meta["min_gap_norm"]) <= crowd_gap_threshold
        )
        meta["occlusion_proxy"] = "heavy" if heavy else "light"

    notes = {
        "overall": "所有有可见采摘点且 IoU 匹配到预测 grape bbox 的 GT 实例。",
        "size_group": f"small/medium_large 按 test visible grape 面积 1/3 分位划分，small 阈值 area<={area_threshold:.1f}。",
        "occlusion_proxy": f"light/heavy 使用 GT 几何代理：同图存在邻串且 max_neighbor_iou>={crowd_iou_threshold:.2f} 或 min_gap_norm<={crowd_gap_threshold:.2f} 记为 heavy。",
        "single_multi": "single/multi_adjacent 按图像内 grape GT 数量划分：1 个为 single，>=2 个为 multi_adjacent。",
    }
    return visible_meta, notes


def build_top_local_roi(pred_box_xyxy: list[float], image_w: int, image_h: int) -> list[float]:
    x1, y1, x2, y2 = pred_box_xyxy
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = 0.5 * (x1 + x2)
    roi_w = w * ROI_WIDTH_SCALE
    rx1 = cx - 0.5 * roi_w
    rx2 = cx + 0.5 * roi_w
    ry1 = y1 + ROI_Y_MIN * h
    ry2 = y1 + ROI_Y_MAX * h
    if image_w > 0:
        rx1 = max(0.0, min(rx1, float(image_w)))
        rx2 = max(0.0, min(rx2, float(image_w)))
    if image_h > 0:
        ry1 = max(0.0, min(ry1, float(image_h)))
        ry2 = max(0.0, min(ry2, float(image_h)))
    return [min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2)]


def point_in_box(point: list[float], box: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def outside_distance(point: list[float], box: list[float]) -> tuple[float, float, float]:
    x, y = point
    x1, y1, x2, y2 = box
    dx = 0.0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0.0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    return float(dx), float(dy), float(math.hypot(dx, dy))


def match_records(
    records: list[dict[str, Any]],
    pred_by_image: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
    score_threshold: float,
    grape_category_id: int | None,
) -> list[dict[str, Any]]:
    cases = []
    for record in records:
        image_id = int(record["image_id"])
        gt_entries = record["gt_instances"]
        preds = pred_by_image.get(image_id, [])
        if grape_category_id is not None:
            preds = [pred for pred in preds if pred.get("category_id") in (None, grape_category_id)]
        preds = [pred for pred in preds if float(pred.get("score", 1.0)) >= score_threshold]
        used_gt: set[int] = set()

        for pred_idx, pred in sorted(enumerate(preds), key=lambda item: float(item[1].get("score", 1.0)), reverse=True):
            best_gt_idx = None
            best_iou = -1.0
            for gt_idx, gt in enumerate(gt_entries):
                if gt_idx in used_gt:
                    continue
                iou = box_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx is None or best_iou < iou_threshold:
                continue
            used_gt.add(best_gt_idx)
            gt = gt_entries[best_gt_idx]
            if not gt["has_picking"]:
                continue
            roi = build_top_local_roi(pred["bbox_xyxy"], int(record["width"]), int(record["height"]))
            gt_point = gt["picking_point"]
            hit = point_in_box(gt_point, roi)
            outside_dx, outside_dy, outside_l2 = outside_distance(gt_point, roi)
            pred_point = pred.get("picking_point")
            pred_l2 = pred_abs_dy = ""
            if pred_point is not None:
                dx = float(pred_point[0] - gt_point[0])
                dy = float(pred_point[1] - gt_point[1])
                pred_l2 = float(math.hypot(dx, dy))
                pred_abs_dy = abs(dy)
            cases.append(
                {
                    "image_id": image_id,
                    "file_name": record["file_name"],
                    "gt_index": best_gt_idx,
                    "pred_index": pred_idx,
                    "iou": best_iou,
                    "gt_area": float(gt["area"]),
                    "gt_point": gt_point,
                    "gt_bbox_xyxy": gt["bbox_xyxy"],
                    "pred_bbox_xyxy": pred["bbox_xyxy"],
                    "roi_xyxy": roi,
                    "roi_hit": hit,
                    "outside_dx_px": outside_dx,
                    "outside_dy_px": outside_dy,
                    "outside_l2_px": outside_l2,
                    "pred_point_l2_px": pred_l2,
                    "pred_point_abs_dy_px": pred_abs_dy,
                    "score": float(pred.get("score", 1.0)),
                }
            )
    return cases


def summarize_group(cases: list[dict[str, Any]], visible_count: int, note: str) -> dict[str, Any]:
    hit_count = sum(1 for item in cases if item["roi_hit"])
    miss_cases = [item for item in cases if not item["roi_hit"]]
    miss_pred_l2 = [float(item["pred_point_l2_px"]) for item in miss_cases if item["pred_point_l2_px"] != ""]
    miss_pred_dy = [float(item["pred_point_abs_dy_px"]) for item in miss_cases if item["pred_point_abs_dy_px"] != ""]
    return {
        "visible_gt_count": int(visible_count),
        "matched_visible_gt_count": len(cases),
        "matched_visible_recall": float(len(cases) / visible_count) if visible_count > 0 else 0.0,
        "roi_hit_count": hit_count,
        "roi_miss_count": len(miss_cases),
        "roi_hit_rate": float(hit_count / len(cases)) if cases else 0.0,
        "miss_roi_distance_mean_l2_px": mean_or_blank([float(item["outside_l2_px"]) for item in miss_cases]),
        "miss_roi_distance_mean_abs_dy_px": mean_or_blank([float(item["outside_dy_px"]) for item in miss_cases]),
        "miss_pred_point_mean_l2_px": mean_or_blank(miss_pred_l2),
        "miss_pred_point_mean_abs_dy_px": mean_or_blank(miss_pred_dy),
        "mean_iou": mean_or_blank([float(item["iou"]) for item in cases]),
        "definition_note": note,
    }


def build_summary_rows(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    visible_meta: dict[tuple[int, int], dict[str, Any]],
    notes: dict[str, str],
) -> list[dict[str, Any]]:
    case_by_key = {(int(item["image_id"]), int(item["gt_index"])): item for item in cases}
    rows = []

    all_keys = set(visible_meta.keys())
    rows.append({"group_family": "overall", "group_label": "overall", **summarize_group([case_by_key[k] for k in all_keys if k in case_by_key], len(all_keys), notes["overall"])})

    group_specs = [
        ("size_group", ["small", "medium_large"]),
        ("occlusion_proxy", ["light", "heavy"]),
        ("single_multi", ["single", "multi_adjacent"]),
    ]
    for family, labels in group_specs:
        for label in labels:
            keys = {key for key, meta in visible_meta.items() if meta.get(family) == label}
            rows.append(
                {
                    "group_family": family,
                    "group_label": label,
                    **summarize_group([case_by_key[k] for k in keys if k in case_by_key], len(keys), notes[family]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group_family",
        "group_label",
        "visible_gt_count",
        "matched_visible_gt_count",
        "matched_visible_recall",
        "roi_hit_count",
        "roi_miss_count",
        "roi_hit_rate",
        "miss_roi_distance_mean_l2_px",
        "miss_roi_distance_mean_abs_dy_px",
        "miss_pred_point_mean_l2_px",
        "miss_pred_point_mean_abs_dy_px",
        "mean_iou",
        "definition_note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    gt_path: Path,
    pred_path: Path,
    iou_threshold: float,
    score_threshold: float,
    notes: dict[str, str],
    missing_reason: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ROI hit rate 诊断报告",
        "",
        "## 统计目的",
        "",
        "本报告检查当前 GPPoint-DETR v7_exp2 的 pred-box-based Top Local ROI 是否覆盖对应 GT picking point。",
        "",
        "Top Local ROI 参数固定为：`width_scale=1.08`，`y_min=-0.10`，`y_max=0.40`。",
        "",
        "## 输入",
        "",
        f"- GT 标注：`{gt_path}`",
        f"- 预测结果：`{pred_path}`",
        f"- 匹配 IoU 阈值：`{iou_threshold:.2f}`",
        f"- grape score 阈值：`{score_threshold:.2f}`",
        "",
    ]
    if missing_reason:
        lines.extend(
            [
                "## 当前状态",
                "",
                f"无法统计 ROI hit rate：{missing_reason}",
                "",
                "需要提供完整预测结果 JSON，至少包含每个预测的 `image_id`、`bbox`、`score`。如果还包含 `picking_point`，脚本会额外统计 ROI miss case 中的 predicted point L2 和 `|dy|`。",
                "",
                "本轮不编造结果，因此不能判断 pred-box-based Top Local ROI 是否已经存在覆盖不足问题。",
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    overall = next(row for row in rows if row["group_family"] == "overall")
    hit_rate = float(overall["roi_hit_rate"])
    lines.extend(
        [
            "## 总体结论",
            "",
            f"- visible GT 数：`{overall['visible_gt_count']}`",
            f"- IoU 匹配到预测 grape bbox 的 visible GT 数：`{overall['matched_visible_gt_count']}`",
            f"- Top Local ROI hit rate：`{format_pct(hit_rate)}`",
            f"- ROI miss 数：`{overall['roi_miss_count']}`",
            "",
        ]
    )
    if hit_rate >= 0.90:
        lines.append("从 overall 看，当前 Top Local ROI 覆盖率较高，覆盖不足未表现为主要矛盾；仍需看 heavy / small 等分组。")
    elif hit_rate >= 0.75:
        lines.append("从 overall 看，当前 Top Local ROI 存在一定覆盖缺口，适合继续用分组结果定位是否集中在小串或遮挡场景。")
    else:
        lines.append("从 overall 看，当前 Top Local ROI 覆盖不足较明显，可作为 Teacher-guided ROI 或 ROI sensitivity 实验的重要证据。")
    lines.extend(["", "## 分组结果", ""])

    header = "| group | visible GT | matched | hit | miss | hit rate | miss ROI L2 | miss ROI |dy| | miss pred L2 | miss pred |dy| |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.extend([header, sep])
    for row in rows:
        group = f"{row['group_family']}:{row['group_label']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row["visible_gt_count"]),
                    str(row["matched_visible_gt_count"]),
                    str(row["roi_hit_count"]),
                    str(row["roi_miss_count"]),
                    format_pct(float(row["roi_hit_rate"])),
                    str(row["miss_roi_distance_mean_l2_px"]),
                    str(row["miss_roi_distance_mean_abs_dy_px"]),
                    str(row["miss_pred_point_mean_l2_px"]),
                    str(row["miss_pred_point_mean_abs_dy_px"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 分组定义",
            "",
            f"- {notes['size_group']}",
            f"- {notes['occlusion_proxy']}",
            f"- {notes['single_multi']}",
            "",
            "## 解释边界",
            "",
            "- ROI hit rate 只诊断 Top Local ROI 是否覆盖 GT picking point，不等价于最终 picking point 精度。",
            "- `miss_roi_distance_*` 表示 GT picking point 到 ROI 边界的最短外部距离，只在 miss case 上统计。",
            "- `miss_pred_point_*` 只有在预测 JSON 中包含 predicted picking point 时才会填充。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_missing_csv(path: Path, reason: str) -> None:
    write_csv(
        path,
        [
            {
                "group_family": "unavailable",
                "group_label": "missing_input",
                "definition_note": reason,
            }
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze whether pred-box Top Local ROI covers GT picking points.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--pred-json", type=Path, default=None)
    parser.add_argument("--pred-bbox-format", choices=("xywh", "xyxy"), default="xywh")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--grape-category-id", type=int, default=None)
    args = parser.parse_args()

    if not args.gt.exists():
        reason = f"GT 标注不存在：{args.gt}"
        write_missing_csv(args.out_csv, reason)
        write_report(args.out_md, [], args.gt, args.pred_json or Path("<missing>"), args.iou_threshold, args.score_threshold, {}, reason)
        return 2

    records, _, _ = load_gt(args.gt)
    visible_meta, notes = enrich_groups(records)

    if args.pred_json is None or not args.pred_json.exists():
        reason = "缺少完整预测结果 JSON；当前仓库中没有可直接用于 ROI hit rate 的 per-pred bbox 文件。"
        write_missing_csv(args.out_csv, reason)
        write_report(args.out_md, [], args.gt, args.pred_json or Path("<missing>"), args.iou_threshold, args.score_threshold, notes, reason)
        return 0

    pred_by_image, _ = parse_prediction_file(args.pred_json, args.pred_bbox_format)
    if not pred_by_image:
        reason = f"预测文件无法解析出 bbox 预测：{args.pred_json}"
        write_missing_csv(args.out_csv, reason)
        write_report(args.out_md, [], args.gt, args.pred_json, args.iou_threshold, args.score_threshold, notes, reason)
        return 2

    cases = match_records(records, pred_by_image, args.iou_threshold, args.score_threshold, args.grape_category_id)
    rows = build_summary_rows(records, cases, visible_meta, notes)
    write_csv(args.out_csv, rows)
    write_report(args.out_md, rows, args.gt, args.pred_json, args.iou_threshold, args.score_threshold, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
