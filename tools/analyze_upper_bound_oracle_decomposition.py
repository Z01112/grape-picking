from __future__ import annotations

from collections import defaultdict

from upper_bound_diagnostics_utils import (
    OUT_DIR,
    TOP_ANCHOR_RATIO,
    collect_candidate_l2s,
    current_cases_by_gt,
    discover_models,
    ensure_out_dir,
    load_records,
    l2,
    md_table,
    point_from_box_and_top_offset,
    safe_float,
    top_offset_from_box_and_point,
    threshold_metrics,
    write_csv,
    write_json,
)


def summarize_l2(values: list[float]) -> dict:
    if not values:
        return {"pair": 0, "mean_L2": 0.0, "PPL-SR@30": 0.0, "PPL-SR@50": 0.0}
    vals = [float(v) for v in values]
    return {
        "pair": len(vals),
        "mean_L2": sum(vals) / len(vals),
        "PPL-SR@30": sum(v <= 30.0 for v in vals) / len(vals),
        "PPL-SR@50": sum(v <= 50.0 for v in vals) / len(vals),
    }


def run_model(model_name: str, records: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    current = threshold_metrics(records, 0.5)
    current_cases = current_cases_by_gt(records, 0.5)
    rows: list[dict] = []
    image_acc: dict[tuple[str, int], dict] = defaultdict(lambda: {"visible_gt": 0})

    gt_has_l2: list[float] = []
    gt_box_l2: list[float] = []
    gt_box_gt_has_l2: list[float] = []
    cand_iou50_l2: list[float] = []
    perfect_l2: list[float] = []
    has_approx = False
    any_topk_like = False

    for rec in records:
        preds = rec.get("pred_instances", [])
        if len(preds) > 100:
            any_topk_like = True
        for gt_idx, gt in enumerate(rec.get("gt_instances", [])):
            if not bool(gt.get("has_picking", False)):
                continue
            key = (model_name, int(rec["image_id"]))
            image_acc[key]["visible_gt"] += 1
            gt_point = gt.get("picking_point", [0.0, 0.0])
            std = current_cases.get((int(rec["image_id"]), gt_idx))
            std_l2 = safe_float(std.get("l2_px")) if std and bool(std.get("pred_has_picking", False)) else None
            matched_l2 = safe_float(std.get("l2_px")) if std else None
            if std:
                gt_has_l2.append(safe_float(std.get("l2_px")))
                pred_box = std.get("pred_bbox_xyxy", [0, 0, 0, 0])
                pred_point = std.get("pred_point", [0, 0])
                raw_offset = None
                pred_idx = int(std.get("pred_index", -1))
                if 0 <= pred_idx < len(preds):
                    raw_offset = preds[pred_idx].get("raw_picking_offset")
                if raw_offset is None:
                    raw_offset = top_offset_from_box_and_point(pred_box, pred_point, TOP_ANCHOR_RATIO)
                    has_approx = True
                gt_box_point = point_from_box_and_top_offset(gt.get("bbox_xyxy", [0, 0, 0, 0]), raw_offset, TOP_ANCHOR_RATIO)
                if bool(std.get("pred_has_picking", False)):
                    gt_box_l2.append(l2(gt_box_point, gt_point))
                gt_box_gt_has_l2.append(l2(gt_box_point, gt_point))
                image_acc[key].setdefault("current_pairs", 0)
                image_acc[key].setdefault("gt_has_pairs", 0)
                image_acc[key]["gt_has_pairs"] += 1
                if std_l2 is not None:
                    image_acc[key]["current_pairs"] += 1

            iou50 = collect_candidate_l2s(rec, gt, iou_min=0.5, visible_only=False)
            if iou50:
                cand_iou50_l2.append(iou50[0]["l2"])
            any_pred = collect_candidate_l2s(rec, gt, iou_min=None, visible_only=False)
            if any_pred:
                perfect_l2.append(any_pred[0]["l2"])
            rows.append(
                {
                    "model": model_name,
                    "image_id": rec.get("image_id"),
                    "file_name": rec.get("file_name", ""),
                    "gt_index": gt_idx,
                    "standard_l2": "" if std_l2 is None else std_l2,
                    "matched_l2_with_gt_has": "" if matched_l2 is None else matched_l2,
                    "gt_box_l2": "" if not std else (gt_box_l2[-1] if bool(std.get("pred_has_picking", False)) else ""),
                    "gt_box_gt_has_l2": "" if not std else gt_box_gt_has_l2[-1],
                    "candidate_iou50_l2": "" if not iou50 else iou50[0]["l2"],
                    "perfect_selection_l2": "" if not any_pred else any_pred[0]["l2"],
                    "standard_pred_visible": bool(std.get("pred_has_picking", False)) if std else False,
                    "candidate_iou50_count": len(iou50),
                    "candidate_pool_note": "topk_like" if len(preds) > 100 else "final_predictions_only",
                }
            )

    summary = {
        "model": model_name,
        "current_default": current,
        "gt_has_oracle": summarize_l2(gt_has_l2),
        "gt_box_oracle": summarize_l2(gt_box_l2),
        "gt_box_gt_has_oracle": summarize_l2(gt_box_gt_has_l2),
        "candidate_pool_oracle_iou50": summarize_l2(cand_iou50_l2),
        "candidate_pool_available": bool(any_topk_like),
        "candidate_pool_note": "records contain more than 100 predictions in at least one image" if any_topk_like else "records appear to contain final predictions only",
        "perfect_selection_oracle": summarize_l2(perfect_l2),
        "gt_box_offset_approximate": bool(has_approx),
    }

    image_rows = []
    for (model, image_id), item in image_acc.items():
        image_rows.append({"model": model, "image_id": image_id, **item})
    return summary, rows, image_rows


def main() -> None:
    ensure_out_dir()
    models, missing = discover_models()
    summaries = []
    instance_rows = []
    image_rows = []
    for name in ["EMA_BIFPN", "CADA_ADAPTER_ONLY_PROBE20", "CADA_FULL_FAIR100_FAILED", "V7_EXP2_FAIR", "YOLO11_POSE"]:
        info = models[name]
        if not info["has_test_records"]:
            continue
        records = load_records(info["test_records_path"])
        summary, rows, img_rows = run_model(name, records)
        summaries.append(summary)
        instance_rows.extend(rows)
        image_rows.extend(img_rows)

    write_json(OUT_DIR / "oracle_decomposition.json", {"models": summaries, "missing_inputs": missing})
    write_csv(OUT_DIR / "oracle_decomposition_by_instance.csv", instance_rows)
    write_csv(OUT_DIR / "oracle_decomposition_by_image.csv", image_rows)

    table_rows = []
    for item in summaries:
        current = item["current_default"]
        table_rows.append(
            {
                "model": item["model"],
                "current_pair": current["pair"],
                "current_mean_L2": current["mean_L2"],
                "gt_has_pair": item["gt_has_oracle"]["pair"],
                "gt_box_mean_L2": item["gt_box_oracle"]["mean_L2"],
                "gt_box_gt_has_pair": item["gt_box_gt_has_oracle"]["pair"],
                "cand_iou50_pair": item["candidate_pool_oracle_iou50"]["pair"],
                "cand_iou50_mean_L2": item["candidate_pool_oracle_iou50"]["mean_L2"],
                "perfect_mean_L2": item["perfect_selection_oracle"]["mean_L2"],
                "approx_offset": item["gt_box_offset_approximate"],
            }
        )
    md = [
        "# Oracle Decomposition",
        "",
        "This is a read-only upper-bound diagnostic from existing prediction records. No checkpoint was evaluated or trained.",
        "",
        md_table(
            table_rows,
            [
                "model",
                "current_pair",
                "current_mean_L2",
                "gt_has_pair",
                "gt_box_mean_L2",
                "gt_box_gt_has_pair",
                "cand_iou50_pair",
                "cand_iou50_mean_L2",
                "perfect_mean_L2",
                "approx_offset",
            ],
        ),
        "",
        "Notes:",
        "- GT-has keeps predicted boxes and points, then removes visibility decision errors for matched visible grapes.",
        "- GT-box re-decodes the same point offset on the GT box. When records lack raw offsets, the offset is approximated from predicted point relative to predicted box.",
        "- Perfect-selection oracle is not deployable; it only estimates whether the exported prediction set contains a usable point.",
    ]
    if missing:
        md.extend(["", "## Missing Inputs", *[f"- {m}" for m in missing]])
    (OUT_DIR / "oracle_decomposition.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

