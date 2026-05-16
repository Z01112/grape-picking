from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roi_hit_rate_analysis import (
    DEFAULT_GT,
    box_iou,
    enrich_groups,
    load_gt,
    mean_or_blank,
    parse_prediction_file,
    point_in_box,
    outside_distance,
)


DEFAULT_PRED = Path("outputs/grape_point_gppoint_detr_main/predictions/test_predictions.json")
DEFAULT_REPORT = Path("reports/roi_sensitivity_analysis_zh.md")
DEFAULT_CSV = Path("reports/roi_sensitivity.csv")

ROI_VARIANTS = [
    {"variant": "narrow", "width_scale": 0.94, "y_min": -0.14, "y_max": 0.26},
    {"variant": "current", "width_scale": 1.08, "y_min": -0.10, "y_max": 0.40},
    {"variant": "wider", "width_scale": 1.20, "y_min": -0.10, "y_max": 0.40},
    {"variant": "taller", "width_scale": 1.08, "y_min": -0.20, "y_max": 0.55},
    {"variant": "lower", "width_scale": 1.08, "y_min": 0.00, "y_max": 0.55},
]


def build_top_local_roi(
    pred_box_xyxy: list[float],
    image_w: int,
    image_h: int,
    width_scale: float,
    y_min: float,
    y_max: float,
) -> list[float]:
    x1, y1, x2, y2 = pred_box_xyxy
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = 0.5 * (x1 + x2)
    roi_w = w * width_scale
    rx1 = cx - 0.5 * roi_w
    rx2 = cx + 0.5 * roi_w
    ry1 = y1 + y_min * h
    ry2 = y1 + y_max * h
    if image_w > 0:
        rx1 = max(0.0, min(rx1, float(image_w)))
        rx2 = max(0.0, min(rx2, float(image_w)))
    if image_h > 0:
        ry1 = max(0.0, min(ry1, float(image_h)))
        ry2 = max(0.0, min(ry2, float(image_h)))
    return [min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2)]


def collect_matched_visible_cases(
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
            cases.append(
                {
                    "image_id": image_id,
                    "file_name": record["file_name"],
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "gt_index": best_gt_idx,
                    "pred_index": pred_idx,
                    "iou": best_iou,
                    "gt_area": float(gt["area"]),
                    "gt_point": gt["picking_point"],
                    "gt_bbox_xyxy": gt["bbox_xyxy"],
                    "pred_bbox_xyxy": pred["bbox_xyxy"],
                    "score": float(pred.get("score", 1.0)),
                }
            )
    return cases


def apply_roi_variant(cases: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        roi = build_top_local_roi(
            case["pred_bbox_xyxy"],
            int(case["width"]),
            int(case["height"]),
            float(variant["width_scale"]),
            float(variant["y_min"]),
            float(variant["y_max"]),
        )
        hit = point_in_box(case["gt_point"], roi)
        outside_dx, outside_dy, outside_l2 = outside_distance(case["gt_point"], roi)
        item = dict(case)
        item.update(
            {
                "roi_xyxy": roi,
                "roi_hit": hit,
                "outside_dx_px": outside_dx,
                "outside_dy_px": outside_dy,
                "outside_l2_px": outside_l2,
            }
        )
        output.append(item)
    return output


def summarize_group(cases: list[dict[str, Any]], visible_count: int, note: str) -> dict[str, Any]:
    hit_count = sum(1 for item in cases if item["roi_hit"])
    miss_cases = [item for item in cases if not item["roi_hit"]]
    return {
        "visible_gt_count": int(visible_count),
        "matched_visible_gt_count": len(cases),
        "matched_visible_recall": float(len(cases) / visible_count) if visible_count > 0 else 0.0,
        "roi_hit_count": hit_count,
        "roi_miss_count": len(miss_cases),
        "roi_hit_rate": float(hit_count / len(cases)) if cases else 0.0,
        "miss_roi_distance_mean_l2_px": mean_or_blank([float(item["outside_l2_px"]) for item in miss_cases]),
        "miss_roi_distance_mean_abs_dy_px": mean_or_blank([float(item["outside_dy_px"]) for item in miss_cases]),
        "mean_iou": mean_or_blank([float(item["iou"]) for item in cases]),
        "definition_note": note,
    }


def build_rows_for_variant(
    variant: dict[str, Any],
    cases: list[dict[str, Any]],
    visible_meta: dict[tuple[int, int], dict[str, Any]],
    notes: dict[str, str],
) -> list[dict[str, Any]]:
    variant_cases = apply_roi_variant(cases, variant)
    case_by_key = {(int(item["image_id"]), int(item["gt_index"])): item for item in variant_cases}
    rows = []
    all_keys = set(visible_meta.keys())
    rows.append(
        {
            "roi_variant": variant["variant"],
            "width_scale": variant["width_scale"],
            "y_min": variant["y_min"],
            "y_max": variant["y_max"],
            "group_family": "overall",
            "group_label": "overall",
            **summarize_group([case_by_key[key] for key in all_keys if key in case_by_key], len(all_keys), notes["overall"]),
        }
    )
    for family, labels in (
        ("size_group", ["small", "medium_large"]),
        ("occlusion_proxy", ["light", "heavy"]),
        ("single_multi", ["single", "multi_adjacent"]),
    ):
        for label in labels:
            keys = {key for key, meta in visible_meta.items() if meta.get(family) == label}
            rows.append(
                {
                    "roi_variant": variant["variant"],
                    "width_scale": variant["width_scale"],
                    "y_min": variant["y_min"],
                    "y_max": variant["y_max"],
                    "group_family": family,
                    "group_label": label,
                    **summarize_group([case_by_key[key] for key in keys if key in case_by_key], len(keys), notes[family]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "roi_variant",
        "width_scale",
        "y_min",
        "y_max",
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
        "mean_iou",
        "definition_note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    gt_path: Path,
    pred_path: Path,
    iou_threshold: float,
    score_threshold: float,
    notes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ROI sensitivity 诊断报告",
        "",
        "## 统计目的",
        "",
        "本报告只比较不同 Top Local ROI 几何参数下，pred-box-based ROI 对 GT picking point 的覆盖率。该诊断不等价于最终 point 精度，也不声称任何参数会带来最终模型性能提升。",
        "",
        "## 输入",
        "",
        f"- GT 标注：`{gt_path}`",
        f"- 预测结果：`{pred_path}`",
        f"- 匹配 IoU 阈值：`{iou_threshold:.2f}`",
        f"- grape score 阈值：`{score_threshold:.2f}`",
        "",
        "## ROI 参数",
        "",
        "| variant | width_scale | y_min | y_max |",
        "|---|---:|---:|---:|",
    ]
    for variant in ROI_VARIANTS:
        lines.append(f"| {variant['variant']} | {variant['width_scale']:.2f} | {variant['y_min']:.2f} | {variant['y_max']:.2f} |")

    lines.extend(
        [
            "",
            "## Overall 对比",
            "",
            "| variant | hit rate | miss count | miss L2 px | miss abs dy px |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["group_family"] != "overall":
            continue
        lines.append(
            f"| {row['roi_variant']} | {pct(row['roi_hit_rate'])} | {row['roi_miss_count']} | "
            f"{row['miss_roi_distance_mean_l2_px']} | {row['miss_roi_distance_mean_abs_dy_px']} |"
        )

    lines.extend(
        [
            "",
            "## 分组 Hit Rate",
            "",
            "| group | narrow | current | wider | taller | lower |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    group_order = [
        ("overall", "overall"),
        ("size_group", "small"),
        ("size_group", "medium_large"),
        ("occlusion_proxy", "light"),
        ("occlusion_proxy", "heavy"),
        ("single_multi", "single"),
        ("single_multi", "multi_adjacent"),
    ]
    lookup = {(row["roi_variant"], row["group_family"], row["group_label"]): row for row in rows}
    for family, label in group_order:
        cells = []
        for variant in ROI_VARIANTS:
            row = lookup[(variant["variant"], family, label)]
            cells.append(pct(row["roi_hit_rate"]))
        lines.append(f"| {family}:{label} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 分组 Miss Distance",
            "",
            "`miss L2 / |dy|` 只在 ROI miss case 上统计，表示 GT picking point 到 ROI 边界的最短外部距离。",
            "",
            "| group | variant | miss count | miss L2 px | miss abs dy px |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for family, label in group_order:
        for variant in ROI_VARIANTS:
            row = lookup[(variant["variant"], family, label)]
            lines.append(
                f"| {family}:{label} | {variant['variant']} | {row['roi_miss_count']} | "
                f"{row['miss_roi_distance_mean_l2_px']} | {row['miss_roi_distance_mean_abs_dy_px']} |"
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
            "## 结论边界",
            "",
            "- 本报告只说明固定几何 ROI 的覆盖率变化。",
            "- 覆盖率更高并不自动代表最终 picking point 精度更高，因为更大的 ROI 也可能引入邻近果串干扰。",
            "- 任何参数收益都需要后续训练或受控推理实验验证，本报告不能作为最终性能提升结论。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top Local ROI sensitivity diagnostics.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--pred-json", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--pred-bbox-format", choices=("xywh", "xyxy"), default="xywh")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--grape-category-id", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, _, _ = load_gt(args.gt)
    visible_meta, notes = enrich_groups(records)
    pred_by_image, _ = parse_prediction_file(args.pred_json, args.pred_bbox_format)
    matched_cases = collect_matched_visible_cases(
        records,
        pred_by_image,
        args.iou_threshold,
        args.score_threshold,
        args.grape_category_id,
    )

    rows = []
    for variant in ROI_VARIANTS:
        rows.extend(build_rows_for_variant(variant, matched_cases, visible_meta, notes))
    write_csv(args.out_csv, rows)
    write_report(args.out_md, rows, args.gt, args.pred_json, args.iou_threshold, args.score_threshold, notes)
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
