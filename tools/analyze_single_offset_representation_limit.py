from __future__ import annotations

from collections import Counter, defaultdict

from upper_bound_diagnostics_utils import (
    OUT_DIR,
    TOPROI_WIDTH_SCALE,
    TOPROI_Y_MAX,
    TOPROI_Y_MIN,
    area_group,
    area_groups,
    band_for_rely,
    collect_case_groups,
    coco_gt_rows,
    counter_rows,
    discover_models,
    ensure_out_dir,
    in_roi,
    load_records,
    md_table,
    rel_xy,
    safe_float,
    summarize_values,
    top_offset_from_box_and_point,
    write_csv,
    write_json,
)


def main() -> None:
    ensure_out_dir()
    gt_rows = []
    all_coco = []
    for split in ["train", "valid", "test"]:
        split_rows = coco_gt_rows(split)
        all_coco.extend(split_rows)
        q1, q2 = area_groups([r["area"] for r in split_rows])
        for item in split_rows:
            relx, rely = rel_xy(item["bbox_xyxy"], item["picking_point"])
            top_dx, top_dy = top_offset_from_box_and_point(item["bbox_xyxy"], item["picking_point"], 0.12)
            center_dx, center_dy = top_offset_from_box_and_point(item["bbox_xyxy"], item["picking_point"], 0.50)
            gt_rows.append(
                {
                    "split": split,
                    "image_id": item["image_id"],
                    "file_name": item["file_name"],
                    "ann_index": item["ann_index"],
                    "area": item["area"],
                    "area_group": area_group(item["area"], q1, q2),
                    "rel_x": relx,
                    "rel_y": rely,
                    "rel_y_band": band_for_rely(rely),
                    "top_center_dx": top_dx,
                    "top_center_dy": top_dy,
                    "center_dx": center_dx,
                    "center_dy": center_dy,
                    "toproi_local_x": (relx - (0.5 - 0.5 * TOPROI_WIDTH_SCALE)) / TOPROI_WIDTH_SCALE,
                    "toproi_local_y": (rely - TOPROI_Y_MIN) / max(TOPROI_Y_MAX - TOPROI_Y_MIN, 1e-6),
                    "current_toproi_in": in_roi(relx, rely),
                }
            )
    write_csv(OUT_DIR / "offset_distribution.csv", gt_rows)

    models, missing = discover_models()
    ema_info = models["EMA_BIFPN"]
    err_rows = []
    if ema_info.get("has_test_records"):
        records = load_records(ema_info["test_records_path"])
        pairs, fp, fn = collect_case_groups(records, has_picking_threshold=0.5)
        areas = [safe_float(p.get("gt_area", 0.0)) for p in pairs]
        q1, q2 = area_groups(areas)
        for p in pairs:
            gt_box = p.get("gt_bbox_xyxy", [0, 0, 0, 0])
            pred_box = p.get("pred_bbox_xyxy", [0, 0, 0, 0])
            gt_point = p.get("gt_point", [0, 0])
            pred_point = p.get("pred_point", [0, 0])
            relx, rely = rel_xy(gt_box, gt_point)
            pred_relx, pred_rely = rel_xy(pred_box, pred_point)
            pred_gt_relx, pred_gt_rely = rel_xy(gt_box, pred_point)
            row = {
                "model": "EMA_BIFPN",
                "image_id": p.get("image_id"),
                "file_name": p.get("file_name", ""),
                "gt_index": p.get("gt_index"),
                "pred_index": p.get("pred_index"),
                "iou": safe_float(p.get("iou", 0.0)),
                "gt_area": safe_float(p.get("gt_area", 0.0)),
                "area_group": area_group(safe_float(p.get("gt_area", 0.0)), q1, q2),
                "gt_rel_x": relx,
                "gt_rel_y": rely,
                "gt_rel_y_band": band_for_rely(rely),
                "pred_rel_x_to_pred_box": pred_relx,
                "pred_rel_y_to_pred_box": pred_rely,
                "pred_rel_x_to_gt_box": pred_gt_relx,
                "pred_rel_y_to_gt_box": pred_gt_rely,
                "dx_px": safe_float(p.get("dx_px", 0.0)),
                "dy_px": safe_float(p.get("dy_px", 0.0)),
                "l2_px": safe_float(p.get("l2_px", 0.0)),
                "abs_dx_norm_gt_w": abs(safe_float(p.get("dx_px", 0.0))) / max(gt_box[2] - gt_box[0], 1e-6),
                "abs_dy_norm_gt_h": abs(safe_float(p.get("dy_px", 0.0))) / max(gt_box[3] - gt_box[1], 1e-6),
                "l2_gt30": safe_float(p.get("l2_px", 0.0)) > 30.0,
                "l2_gt50": safe_float(p.get("l2_px", 0.0)) > 50.0,
                "high_iou_point_bad": safe_float(p.get("iou", 0.0)) >= 0.85 and safe_float(p.get("l2_px", 0.0)) > 30.0,
                "low_iou_propagation": 0.5 <= safe_float(p.get("iou", 0.0)) < 0.7 and safe_float(p.get("l2_px", 0.0)) > 30.0,
                "high_has_score_point_bad": safe_float(p.get("pred_visible_score", 0.0)) >= 0.8 and safe_float(p.get("l2_px", 0.0)) > 30.0,
                "current_toproi_in": in_roi(relx, rely),
            }
            err_rows.append(row)
    write_csv(OUT_DIR / "offset_error_by_region.csv", err_rows)

    by_group = defaultdict(list)
    for row in err_rows:
        for key in ["all", f"area:{row['area_group']}", f"band:{row['gt_rel_y_band']}", f"iou:{'high>=0.85' if row['iou'] >= 0.85 else 'mid0.7-0.85' if row['iou'] >= 0.7 else 'low0.5-0.7'}"]:
            by_group[key].append(row)
    group_rows = []
    for group, items in sorted(by_group.items()):
        group_rows.append(
            {
                "group": group,
                "count": len(items),
                "mean_L2": summarize_values([i["l2_px"] for i in items]).get("mean", 0.0),
                "p90_L2": summarize_values([i["l2_px"] for i in items]).get("p90", 0.0),
                "L2>30_ratio": sum(i["l2_gt30"] for i in items) / len(items) if items else 0.0,
                "L2>50_ratio": sum(i["l2_gt50"] for i in items) / len(items) if items else 0.0,
                "mean_abs_dx_norm": summarize_values([i["abs_dx_norm_gt_w"] for i in items]).get("mean", 0.0),
                "mean_abs_dy_norm": summarize_values([i["abs_dy_norm_gt_h"] for i in items]).get("mean", 0.0),
            }
        )

    gt_summary = {
        "top_center_dx": summarize_values([r["top_center_dx"] for r in gt_rows]),
        "top_center_dy": summarize_values([r["top_center_dy"] for r in gt_rows]),
        "center_dx": summarize_values([r["center_dx"] for r in gt_rows]),
        "center_dy": summarize_values([r["center_dy"] for r in gt_rows]),
        "rel_x": summarize_values([r["rel_x"] for r in gt_rows]),
        "rel_y": summarize_values([r["rel_y"] for r in gt_rows]),
        "rel_y_band_distribution": counter_rows(Counter(r["rel_y_band"] for r in gt_rows), "rel_y_band"),
        "toproi_out_ratio": sum(not r["current_toproi_in"] for r in gt_rows) / len(gt_rows) if gt_rows else 0.0,
    }
    error_summary = {
        "pair_count": len(err_rows),
        "high_iou_point_bad_count": sum(r["high_iou_point_bad"] for r in err_rows),
        "low_iou_propagation_count": sum(r["low_iou_propagation"] for r in err_rows),
        "high_has_score_point_bad_count": sum(r["high_has_score_point_bad"] for r in err_rows),
        "group_summary": group_rows,
    }
    write_json(OUT_DIR / "single_offset_limit.json", {"gt_offset_distribution": gt_summary, "prediction_error": error_summary, "missing_inputs": missing})
    write_csv(OUT_DIR / "offset_error_group_summary.csv", group_rows)
    md = [
        "# Single-Offset Representation Limit",
        "",
        "This read-only diagnostic compares GT offset geometry with EMA_BIFPN matched visible-pair errors.",
        "",
        "## GT Offset Distribution",
        md_table(
            [
                {"axis": key, **{k: v for k, v in val.items() if k in ["count", "mean", "median", "p10", "p90", "min", "max"]}}
                for key, val in gt_summary.items()
                if isinstance(val, dict) and "mean" in val
            ],
            ["axis", "count", "mean", "median", "p10", "p90", "min", "max"],
        ),
        "",
        "## Error By Region",
        md_table(group_rows, ["group", "count", "mean_L2", "p90_L2", "L2>30_ratio", "L2>50_ratio", "mean_abs_dx_norm", "mean_abs_dy_norm"]),
        "",
        "Key counts:",
        f"- high_iou_point_bad_count: {error_summary['high_iou_point_bad_count']}",
        f"- low_iou_propagation_count: {error_summary['low_iou_propagation_count']}",
        f"- high_has_score_point_bad_count: {error_summary['high_has_score_point_bad_count']}",
    ]
    (OUT_DIR / "single_offset_limit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

