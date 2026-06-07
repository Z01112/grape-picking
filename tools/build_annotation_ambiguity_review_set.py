from __future__ import annotations

from collections import defaultdict

from upper_bound_diagnostics_utils import (
    OUT_DIR,
    collect_case_groups,
    collect_candidate_l2s,
    discover_models,
    ensure_out_dir,
    in_roi,
    load_records,
    md_table,
    rel_xy,
    safe_float,
    write_csv,
    write_json,
)


def case_label(case: dict, gt_rel_y: float, toproi_in: bool, repeated: int = 1) -> tuple[str, str]:
    l2 = safe_float(case.get("l2_px", 0.0))
    iou = safe_float(case.get("iou", 0.0))
    has_score = safe_float(case.get("pred_visible_score", 0.0))
    if iou >= 0.85 and l2 > 30:
        return "picking_label_questionable", "high IoU but point is far from GT; verify picking point semantics"
    if not toproi_in:
        return "top_roi_mismatch", "GT picking point is outside current TopROI"
    if gt_rel_y > 0.55:
        return "multiple_possible_points", "GT point is below upper grape region; check if label should be lower or stem-related"
    if l2 > 50:
        return "ambiguous_visible", "large point error; verify visible picking point and box association"
    if has_score >= 0.8 and l2 > 30:
        return "clear_visible", "model is confident but point is inaccurate; inspect local visual cues"
    if repeated >= 2:
        return "duplicate_cluster_confusion", "case fails across multiple models"
    return "tiny_or_blurry", "moderate point error; check occlusion, blur, or small cluster"


def main() -> None:
    ensure_out_dir()
    models, missing = discover_models()
    selected = {}
    model_order = ["EMA_BIFPN", "CADA_FULL_FAIR100_FAILED", "CADA_ADAPTER_ONLY_PROBE20", "V7_EXP2_FAIR"]
    all_fail_marks = defaultdict(int)

    model_pairs = {}
    for name in model_order:
        info = models[name]
        if not info.get("has_test_records"):
            continue
        records = load_records(info["test_records_path"])
        pairs, fp, fn = collect_case_groups(records, has_picking_threshold=0.5)
        model_pairs[name] = (records, pairs, fp, fn)
        for case in pairs:
            if safe_float(case.get("l2_px", 0.0)) > 30.0:
                all_fail_marks[(case.get("image_id"), case.get("gt_index"))] += 1
        for case in fn:
            all_fail_marks[(case.get("image_id"), case.get("gt_index"))] += 1

    priority_rows = []
    for name in model_order:
        if name not in model_pairs:
            continue
        records, pairs, fp, fn = model_pairs[name]
        rec_by_id = {r["image_id"]: r for r in records}
        for case in pairs:
            l2 = safe_float(case.get("l2_px", 0.0))
            if l2 <= 30.0 and all_fail_marks[(case.get("image_id"), case.get("gt_index"))] < 2:
                continue
            rec = rec_by_id.get(case.get("image_id"))
            if not rec:
                continue
            gt = rec["gt_instances"][int(case["gt_index"])]
            relx, rely = rel_xy(gt["bbox_xyxy"], gt["picking_point"])
            toproi_ok = in_roi(relx, rely)
            review_label, question = case_label(case, rely, toproi_ok, all_fail_marks[(case.get("image_id"), case.get("gt_index"))])
            failure_type = "L2>50" if l2 > 50 else "high_iou_point_bad" if safe_float(case.get("iou", 0)) >= 0.85 and l2 > 30 else "L2>30"
            key = (case.get("image_id"), case.get("gt_index"))
            row = {
                "split": "test",
                "image_id": case.get("image_id"),
                "file_name": case.get("file_name", ""),
                "grape_gt_id": case.get("gt_index"),
                "picking_gt_id": case.get("gt_index"),
                "gt_grape_bbox": gt.get("bbox_xyxy"),
                "gt_picking_point": gt.get("picking_point"),
                "gt_picking_bbox": gt.get("picking_bbox", ""),
                "model": name,
                "model_pred_point": case.get("pred_point"),
                "model_pred_box": case.get("pred_bbox_xyxy"),
                "l2": l2,
                "has_score": case.get("pred_visible_score"),
                "bbox_iou": case.get("iou"),
                "failure_type": failure_type,
                "suggested_review_label": review_label,
                "review_question": question,
                "toproi_in": toproi_ok,
                "cross_model_failure_count": all_fail_marks[key],
            }
            score = (3 if l2 > 50 else 2 if l2 > 30 else 0) + (2 if failure_type == "high_iou_point_bad" else 0) + all_fail_marks[key]
            row["_priority_score"] = score
            if key not in selected or selected[key]["_priority_score"] < score:
                selected[key] = row
        for case in fn:
            key = (case.get("image_id"), case.get("gt_index"))
            if key in selected:
                continue
            selected[key] = {
                "split": "test",
                "image_id": case.get("image_id"),
                "file_name": case.get("file_name", ""),
                "grape_gt_id": case.get("gt_index"),
                "picking_gt_id": case.get("gt_index"),
                "gt_grape_bbox": case.get("gt_bbox_xyxy"),
                "gt_picking_point": case.get("gt_point"),
                "gt_picking_bbox": "",
                "model": name,
                "model_pred_point": case.get("pred_point"),
                "model_pred_box": case.get("pred_bbox_xyxy"),
                "l2": "",
                "has_score": case.get("pred_visible_score"),
                "bbox_iou": case.get("iou"),
                "failure_type": "GT_has_but_model_no_visible_output",
                "suggested_review_label": "ambiguous_visible",
                "review_question": "GT is visible but matched prediction is not visible; verify has_picking visibility",
                "toproi_in": "",
                "cross_model_failure_count": all_fail_marks[key],
                "_priority_score": 2 + all_fail_marks[key],
            }

    priority_rows = sorted(selected.values(), key=lambda r: (r["_priority_score"], safe_float(r.get("l2", 0.0))), reverse=True)[:200]
    for i, row in enumerate(priority_rows, 1):
        row["review_index"] = i
        row.pop("_priority_score", None)
    write_csv(OUT_DIR / "annotation_ambiguity_review_set.csv", priority_rows)
    manifest = {
        "count": len(priority_rows),
        "selection_sources": model_order,
        "missing_inputs": missing,
        "review_labels": sorted(set(r["suggested_review_label"] for r in priority_rows)),
    }
    write_json(OUT_DIR / "annotation_ambiguity_review_manifest.json", manifest)
    label_counts = defaultdict(int)
    for row in priority_rows:
        label_counts[row["suggested_review_label"]] += 1
    md_rows = [{"suggested_review_label": k, "count": v} for k, v in sorted(label_counts.items(), key=lambda x: -x[1])]
    md = [
        "# Annotation Ambiguity Review Set",
        "",
        "This is a CSV-only review list built from existing prediction records. It does not modify dataset annotations and does not copy images.",
        "",
        md_table(md_rows, ["suggested_review_label", "count"]),
        "",
        "Recommended review protocol:",
        "- Do not delete all hard samples.",
        "- Mark only confirmed annotation mistakes or semantically unresolvable cases.",
        "- Keep genuine hard occlusion cases for realistic evaluation.",
        "- If valid/test labels are corrected, create a clean_eval_v1 and re-evaluate all models under the same records logic.",
    ]
    (OUT_DIR / "annotation_ambiguity_review_set.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

