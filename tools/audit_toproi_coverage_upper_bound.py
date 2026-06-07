from __future__ import annotations

from collections import Counter, defaultdict

from upper_bound_diagnostics_utils import (
    OUT_DIR,
    RELAXED_WIDTH_SCALE,
    RELAXED_Y_MAX,
    RELAXED_Y_MIN,
    TOPROI_WIDTH_SCALE,
    TOPROI_Y_MAX,
    TOPROI_Y_MIN,
    area_group,
    area_groups,
    band_for_rely,
    coco_gt_rows,
    counter_rows,
    ensure_out_dir,
    in_roi,
    md_table,
    rel_xy,
    roi_boundary_distance,
    summarize_values,
    write_csv,
    write_json,
)


def main() -> None:
    ensure_out_dir()
    rows = []
    local_rows = []
    all_gt = []
    for split in ["train", "valid", "test"]:
        split_rows = coco_gt_rows(split)
        all_gt.extend(split_rows)
        q1, q2 = area_groups([r["area"] for r in split_rows])
        for item in split_rows:
            relx, rely = rel_xy(item["bbox_xyxy"], item["picking_point"])
            current = in_roi(relx, rely, TOPROI_WIDTH_SCALE, TOPROI_Y_MIN, TOPROI_Y_MAX)
            relaxed = in_roi(relx, rely, RELAXED_WIDTH_SCALE, RELAXED_Y_MIN, RELAXED_Y_MAX)
            upper_half = 0.0 <= relx <= 1.0 and 0.0 <= rely <= 0.5
            dist = roi_boundary_distance(relx, rely, TOPROI_WIDTH_SCALE, TOPROI_Y_MIN, TOPROI_Y_MAX)
            row = {
                **item,
                "rel_x": relx,
                "rel_y": rely,
                "current_toproi_in": current,
                "relaxed_toproi_in": relaxed,
                "upper_half_roi_in": upper_half,
                "gt_centered_oracle_roi_in": True,
                "near_current_boundary": abs(dist) < 0.05,
                "current_boundary_distance": dist,
                "rel_y_band": band_for_rely(rely),
                "area_group": area_group(item["area"], q1, q2),
                "picking_box_available": bool(item.get("picking_bbox")),
            }
            rows.append(row)
            local_rows.append(
                {
                    "split": split,
                    "image_id": item["image_id"],
                    "file_name": item["file_name"],
                    "rel_x": relx,
                    "rel_y": rely,
                    "current_local_x": (relx - (0.5 - 0.5 * TOPROI_WIDTH_SCALE)) / TOPROI_WIDTH_SCALE,
                    "current_local_y": (rely - TOPROI_Y_MIN) / max(TOPROI_Y_MAX - TOPROI_Y_MIN, 1e-6),
                    "current_toproi_in": current,
                    "relaxed_toproi_in": relaxed,
                    "area_group": row["area_group"],
                    "rel_y_band": row["rel_y_band"],
                }
            )

    write_csv(OUT_DIR / "toproi_coverage_by_instance.csv", rows)
    write_csv(OUT_DIR / "toproi_local_xy_distribution.csv", local_rows)

    summary_rows = []
    by_group = defaultdict(list)
    for row in rows:
        for key in ["all", f"split:{row['split']}", f"area:{row['area_group']}", f"band:{row['rel_y_band']}"]:
            by_group[key].append(row)
    for group, items in sorted(by_group.items()):
        n = len(items)
        summary_rows.append(
            {
                "group": group,
                "count": n,
                "in_roi_ratio": sum(bool(i["current_toproi_in"]) for i in items) / n if n else 0.0,
                "out_of_roi_ratio": sum(not bool(i["current_toproi_in"]) for i in items) / n if n else 0.0,
                "near_boundary_ratio": sum(bool(i["near_current_boundary"]) for i in items) / n if n else 0.0,
                "relaxed_in_roi_ratio": sum(bool(i["relaxed_toproi_in"]) for i in items) / n if n else 0.0,
                "upper_half_in_roi_ratio": sum(bool(i["upper_half_roi_in"]) for i in items) / n if n else 0.0,
                "rel_x_median": summarize_values([i["rel_x"] for i in items]).get("median", 0.0),
                "rel_x_p10": summarize_values([i["rel_x"] for i in items]).get("p10", 0.0),
                "rel_x_p90": summarize_values([i["rel_x"] for i in items]).get("p90", 0.0),
                "rel_y_median": summarize_values([i["rel_y"] for i in items]).get("median", 0.0),
                "rel_y_p10": summarize_values([i["rel_y"] for i in items]).get("p10", 0.0),
                "rel_y_p90": summarize_values([i["rel_y"] for i in items]).get("p90", 0.0),
            }
        )

    band_counter = Counter(r["rel_y_band"] for r in rows)
    data = {
        "toproi_definition": {
            "source": "formal GPPoint-DETR TopROI parameters in configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml and dfine_decoder defaults",
            "width_scale": TOPROI_WIDTH_SCALE,
            "y_min_ratio": TOPROI_Y_MIN,
            "y_max_ratio": TOPROI_Y_MAX,
        },
        "summary": summary_rows,
        "band_distribution": counter_rows(band_counter, "rel_y_band"),
    }
    write_json(OUT_DIR / "toproi_coverage_audit.json", data)
    write_csv(OUT_DIR / "toproi_coverage_summary.csv", summary_rows)
    md = [
        "# TopROI Coverage Audit",
        "",
        "This audit uses the point-rich grape_point annotations derived from COCO and does not run any model.",
        "",
        f"Formal current TopROI: width_scale={TOPROI_WIDTH_SCALE}, y=[{TOPROI_Y_MIN}, {TOPROI_Y_MAX}] relative to grape box top.",
        "",
        md_table(summary_rows, ["group", "count", "in_roi_ratio", "out_of_roi_ratio", "near_boundary_ratio", "relaxed_in_roi_ratio", "upper_half_in_roi_ratio", "rel_y_median", "rel_y_p10", "rel_y_p90"]),
        "",
        "Interpretation guardrails:",
        "- If current TopROI covers most GT points but many bad cases remain, current ROI geometry alone is not the main bottleneck.",
        "- If relaxed ROI gains large coverage, future heatmap/integral coordinates should use relaxed ROI rather than the current narrow TopROI.",
        "- Picking box size grouping is skipped when annotations do not expose a picking bbox field.",
    ]
    (OUT_DIR / "toproi_coverage_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
