from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out):
        return float(default)
    return out


def xyxy_to_xywh(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def xywh_to_xyxy(box_xywh: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def _box_iou_matrix(pred_boxes: list[list[float]], gt_boxes: list[list[float]]) -> torch.Tensor:
    if not pred_boxes or not gt_boxes:
        return torch.zeros((len(pred_boxes), len(gt_boxes)), dtype=torch.float32)
    pred = torch.as_tensor(pred_boxes, dtype=torch.float32)
    gt = torch.as_tensor(gt_boxes, dtype=torch.float32)
    lt = torch.maximum(pred[:, None, :2], gt[:, :2])
    rb = torch.minimum(pred[:, None, 2:], gt[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    gt_area = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (gt[:, 3] - gt[:, 1]).clamp(min=0)
    union = pred_area[:, None] + gt_area - inter
    return inter / union.clamp(min=1e-6)


def build_records_from_coco(coco_dataset: dict, split_dir: Path) -> dict[int, dict]:
    images: dict[int, dict] = {}
    for image in coco_dataset.get("images", []):
        image_id = int(image["id"])
        file_name = str(image.get("file_name", ""))
        images[image_id] = {
            "image_id": image_id,
            "file_name": file_name,
            "width": int(image.get("width", 0)),
            "height": int(image.get("height", 0)),
            "image_path": str((split_dir / file_name).resolve()),
            "gt_instances": [],
            "pred_instances": [],
        }

    for ann in coco_dataset.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in images:
            continue
        bbox_xywh = [float(v) for v in ann.get("bbox", [0.0, 0.0, 0.0, 0.0])]
        point = ann.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        area = safe_float(ann.get("area"), bbox_xywh[2] * bbox_xywh[3])
        images[image_id]["gt_instances"].append(
            {
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "area": area,
                "has_picking": bool(safe_float(ann.get("has_picking", 0.0)) > 0.5),
                "picking_point": [float(point[0]), float(point[1])],
            }
        )
    return images


def prediction_to_instances(prediction: dict) -> list[dict]:
    boxes = prediction.get("boxes", torch.zeros((0, 4))).detach().cpu().to(torch.float32)
    scores = prediction.get("scores", torch.zeros((boxes.shape[0],))).detach().cpu().to(torch.float32)
    labels = prediction.get("labels", torch.zeros((boxes.shape[0],), dtype=torch.int64)).detach().cpu()
    raw_has_scores = prediction.get("raw_has_picking_scores")
    has_scores = prediction.get("has_picking_scores")
    visible_scores = prediction.get("visible_scores")
    points = prediction.get("picking_points")

    if raw_has_scores is None:
        raw_has_scores = has_scores
    if visible_scores is None:
        visible_scores = has_scores
    if has_scores is None:
        has_scores = torch.zeros((boxes.shape[0],), dtype=torch.float32)
    else:
        has_scores = has_scores.detach().cpu().to(torch.float32)
    if raw_has_scores is None:
        raw_has_scores = has_scores
    else:
        raw_has_scores = raw_has_scores.detach().cpu().to(torch.float32)
    if visible_scores is None:
        visible_scores = has_scores
    else:
        visible_scores = visible_scores.detach().cpu().to(torch.float32)
    if points is None:
        points = torch.zeros((boxes.shape[0], 2), dtype=torch.float32)
    else:
        points = points.detach().cpu().to(torch.float32)

    optional_tensor_keys = {
        "point_quality_score": prediction.get("point_quality_scores"),
        "point_final_score": prediction.get("point_final_scores"),
        "point_selector_score": prediction.get("point_selector_scores"),
        "point_selector_final_score": prediction.get("point_selector_final_scores"),
        "point_accept_score": prediction.get("point_accept_scores"),
        "point_accept_final_score": prediction.get("point_accept_final_scores"),
        "point_reliability_score": prediction.get("point_reliability_scores"),
        "point_reliability_final_score": prediction.get("point_reliability_final_scores"),
        "weak_heatmap_score": prediction.get("weak_heatmap_scores"),
    }
    optional_tensor_keys = {
        key: value.detach().cpu().to(torch.float32) if value is not None else None
        for key, value in optional_tensor_keys.items()
    }

    instances: list[dict] = []
    for idx in range(boxes.shape[0]):
        bbox_xyxy = [float(v) for v in boxes[idx].tolist()]
        item = {
            "bbox_xyxy": bbox_xyxy,
            "bbox_xywh": xyxy_to_xywh(bbox_xyxy),
            "score": float(scores[idx].item()),
            "label": int(labels[idx].item()),
            "raw_has_picking_score": float(raw_has_scores[idx].item()),
            "has_picking_score": float(has_scores[idx].item()),
            "visible_score": float(visible_scores[idx].item()),
            "picking_point": [float(v) for v in points[idx].tolist()],
            "source": "gppoint_detr",
        }
        for key, value in optional_tensor_keys.items():
            if value is not None:
                item[key] = float(value[idx].item())
        instances.append(item)
    return instances


def normalize_prediction_record(record: dict) -> dict:
    pred_instances = []
    for pred in record.get("pred_instances", []):
        item = dict(pred)
        if "bbox_xyxy" not in item and "bbox_xywh" in item:
            item["bbox_xyxy"] = xywh_to_xyxy(item["bbox_xywh"])
        if "bbox_xywh" not in item and "bbox_xyxy" in item:
            item["bbox_xywh"] = xyxy_to_xywh(item["bbox_xyxy"])
        if "raw_has_picking_score" not in item:
            item["raw_has_picking_score"] = safe_float(item.get("has_picking_score", item.get("visible_score", 0.0)))
        if "has_picking_score" not in item:
            item["has_picking_score"] = safe_float(item.get("visible_score", item.get("raw_has_picking_score", 0.0)))
        if "visible_score" not in item:
            item["visible_score"] = safe_float(item.get("has_picking_score", item.get("raw_has_picking_score", 0.0)))
        item.setdefault("source", "unknown")
        pred_instances.append(item)

    gt_instances = []
    for gt in record.get("gt_instances", []):
        item = dict(gt)
        if "bbox_xyxy" not in item and "bbox_xywh" in item:
            item["bbox_xyxy"] = xywh_to_xyxy(item["bbox_xywh"])
        if "bbox_xywh" not in item and "bbox_xyxy" in item:
            item["bbox_xywh"] = xyxy_to_xywh(item["bbox_xyxy"])
        if "area" not in item:
            box = item.get("bbox_xywh", [0.0, 0.0, 0.0, 0.0])
            item["area"] = float(box[2]) * float(box[3])
        item["has_picking"] = bool(item.get("has_picking", False))
        gt_instances.append(item)

    out = dict(record)
    out["gt_instances"] = gt_instances
    out["pred_instances"] = pred_instances
    return out


def point_in_box(point_xy: list[float], box_xyxy: list[float], margin: float = 0.0) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def build_point_case(record: dict, gt_idx: int, gt: dict, pred_idx: int, pred: dict, iou: float) -> dict:
    gt_point = gt.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    pred_point = pred.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    gt_point = [float(gt_point[0]), float(gt_point[1])]
    pred_point = [float(pred_point[0]), float(pred_point[1])]
    pred_box = [float(v) for v in pred.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])]
    gt_box = [float(v) for v in gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])]
    dx = float(pred_point[0] - gt_point[0])
    dy = float(pred_point[1] - gt_point[1])
    return {
        "image_id": int(record["image_id"]),
        "file_name": record.get("file_name", ""),
        "image_path": record.get("image_path", ""),
        "iou": float(iou),
        "gt_index": int(gt_idx),
        "pred_index": int(pred_idx),
        "gt_bbox_xyxy": gt_box,
        "pred_bbox_xyxy": pred_box,
        "gt_area": safe_float(gt.get("area", 0.0)),
        "gt_point": gt_point,
        "pred_point": pred_point,
        "pred_raw_has_picking_score": safe_float(pred.get("raw_has_picking_score", 0.0)),
        "pred_has_picking_score": safe_float(pred.get("has_picking_score", 0.0)),
        "pred_visible_score": safe_float(pred.get("visible_score", pred.get("has_picking_score", 0.0))),
        "pred_point_reliability_score": safe_float(pred.get("point_reliability_score", 0.0)),
        "pred_point_reliability_final_score": safe_float(pred.get("point_reliability_final_score", 0.0)),
        "pred_score": safe_float(pred.get("score", 0.0)),
        "pred_source": pred.get("source", "unknown"),
        "dx_px": dx,
        "dy_px": dy,
        "l2_px": float(math.hypot(dx, dy)),
        "pred_point_inside_gt_box": point_in_box(pred_point, gt_box),
        "pred_point_inside_pred_box": point_in_box(pred_point, pred_box),
        "gt_point_inside_pred_box": point_in_box(gt_point, pred_box),
    }


def match_prediction_record(
    record: dict,
    iou_threshold: float = 0.5,
    has_picking_threshold: float = 0.5,
    visibility_score_key: str = "visible_score",
) -> dict:
    record = normalize_prediction_record(record)
    gt_entries = record.get("gt_instances", [])
    pred_entries = record.get("pred_instances", [])
    output = {
        "correct_visible_pairs": [],
        "has_fp_pairs": [],
        "has_fn_pairs": [],
        "matched_pairs": [],
        "unmatched_visible_predictions": [],
        "visible_gt_total": sum(1 for gt in gt_entries if bool(gt.get("has_picking"))),
        "predicted_visible_total": sum(
            1 for pred in pred_entries if safe_float(pred.get(visibility_score_key, pred.get("visible_score", 0.0))) >= has_picking_threshold
        ),
    }
    if not gt_entries:
        output["unmatched_visible_predictions"] = [
            build_point_case(record, -1, {"bbox_xyxy": [0, 0, 0, 0], "picking_point": [0, 0]}, idx, pred, 0.0)
            for idx, pred in enumerate(pred_entries)
            if safe_float(pred.get(visibility_score_key, pred.get("visible_score", 0.0))) >= has_picking_threshold
        ]
        return output
    if not pred_entries:
        return output

    gt_boxes = [item["bbox_xyxy"] for item in gt_entries]
    pred_boxes = [item["bbox_xyxy"] for item in pred_entries]
    pred_scores = torch.as_tensor([safe_float(item.get("score", 0.0)) for item in pred_entries], dtype=torch.float32)
    ious = _box_iou_matrix(pred_boxes, gt_boxes)
    pred_order = torch.argsort(pred_scores, descending=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    for pred_idx in pred_order.tolist():
        best_gt = None
        best_iou = -1.0
        for gt_idx in range(len(gt_entries)):
            if gt_idx in used_gt:
                continue
            iou = float(ious[pred_idx, gt_idx].item())
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt is None or best_iou < iou_threshold:
            continue
        used_gt.add(best_gt)
        used_pred.add(pred_idx)
        gt = gt_entries[best_gt]
        pred = pred_entries[pred_idx]
        gt_visible = bool(gt.get("has_picking", False))
        pred_visible = safe_float(pred.get(visibility_score_key, pred.get("visible_score", 0.0))) >= has_picking_threshold
        case = build_point_case(record, best_gt, gt, pred_idx, pred, best_iou)
        case.update({"gt_has_picking": gt_visible, "pred_has_picking": pred_visible})

        if gt_visible and pred_visible:
            output["correct_visible_pairs"].append(case)
        elif (not gt_visible) and pred_visible:
            output["has_fp_pairs"].append(case)
        elif gt_visible and (not pred_visible):
            output["has_fn_pairs"].append(case)
        output["matched_pairs"].append(case)

    for pred_idx, pred in enumerate(pred_entries):
        if pred_idx in used_pred:
            continue
        if safe_float(pred.get(visibility_score_key, pred.get("visible_score", 0.0))) < has_picking_threshold:
            continue
        output["unmatched_visible_predictions"].append(
            build_point_case(record, -1, {"bbox_xyxy": [0, 0, 0, 0], "picking_point": [0, 0]}, pred_idx, pred, 0.0)
        )
    return output


def collect_case_groups(
    records: list[dict],
    iou_threshold: float = 0.5,
    has_picking_threshold: float = 0.5,
    visibility_score_key: str = "visible_score",
) -> tuple[list[dict], list[dict], list[dict]]:
    correct_pairs, fp_pairs, fn_pairs = [], [], []
    for record in records:
        matched = match_prediction_record(record, iou_threshold, has_picking_threshold, visibility_score_key)
        correct_pairs.extend(matched["correct_visible_pairs"])
        fp_pairs.extend(matched["has_fp_pairs"])
        fp_pairs.extend(matched["unmatched_visible_predictions"])
        fn_pairs.extend(matched["has_fn_pairs"])
    return correct_pairs, fp_pairs, fn_pairs


def summarize_point_cases(cases: list[dict]) -> dict:
    l2_values = [safe_float(item.get("l2_px", 0.0)) for item in cases]
    dx_values = [safe_float(item.get("dx_px", 0.0)) for item in cases]
    dy_values = [safe_float(item.get("dy_px", 0.0)) for item in cases]
    if not l2_values:
        return {
            "pair_count": 0,
            "mean_l2_px": 0.0,
            "median_l2_px": 0.0,
            "p90_l2_px": 0.0,
            "mean_abs_dx_px": 0.0,
            "mean_abs_dy_px": 0.0,
            "ppl_sr_30": 0.0,
            "ppl_sr_50": 0.0,
        }
    l2 = np.asarray(l2_values, dtype=np.float64)
    return {
        "pair_count": int(len(l2_values)),
        "mean_l2_px": float(np.mean(l2)),
        "median_l2_px": float(np.median(l2)),
        "p90_l2_px": float(np.quantile(l2, 0.90)),
        "mean_abs_dx_px": float(np.mean(np.abs(dx_values))),
        "mean_abs_dy_px": float(np.mean(np.abs(dy_values))),
        "ppl_sr_30": float(np.mean(l2 <= 30.0)),
        "ppl_sr_50": float(np.mean(l2 <= 50.0)),
    }


def compute_unified_point_metrics(
    records: list[dict],
    iou_threshold: float = 0.5,
    has_picking_threshold: float = 0.5,
    visibility_score_key: str = "visible_score",
) -> dict:
    visible_gt_total = 0
    matched_grapes = 0
    matched_visible_grapes = 0
    predicted_visible_matched = 0
    predicted_visible_total = 0
    correct_visible = 0
    false_visible_matched = 0
    missed_visible = 0
    correct_pairs: list[dict] = []
    fp_pairs: list[dict] = []
    fn_pairs: list[dict] = []

    for record in records:
        matched = match_prediction_record(record, iou_threshold, has_picking_threshold, visibility_score_key)
        visible_gt_total += int(matched["visible_gt_total"])
        predicted_visible_total += int(matched["predicted_visible_total"])
        matched_pairs = matched["matched_pairs"]
        matched_grapes += len(matched_pairs)
        matched_visible_grapes += sum(1 for case in matched_pairs if bool(case.get("gt_has_picking", False)))
        predicted_visible_matched += sum(1 for case in matched_pairs if bool(case.get("pred_has_picking", False)))
        false_visible_matched += len(matched["has_fp_pairs"])
        missed_visible += len(matched["has_fn_pairs"])
        correct_visible += len(matched["correct_visible_pairs"])
        correct_pairs.extend(matched["correct_visible_pairs"])
        fp_pairs.extend(matched["has_fp_pairs"])
        fp_pairs.extend(matched["unmatched_visible_predictions"])
        fn_pairs.extend(matched["has_fn_pairs"])

    instance_precision = correct_visible / predicted_visible_matched if predicted_visible_matched > 0 else 0.0
    instance_recall = correct_visible / matched_visible_grapes if matched_visible_grapes > 0 else 0.0
    instance_f1 = 0.0 if instance_precision + instance_recall == 0 else (
        2.0 * instance_precision * instance_recall / (instance_precision + instance_recall)
    )
    # The global chain changes the recall denominator from matched-visible GT
    # to all visible GT. Precision stays on the IoU50 instance chain, otherwise
    # DETR's unthresholded top-query exports would make GPPoint reports depend
    # on an arbitrary detector score cutoff that the legacy reports never used.
    global_precision = correct_visible / predicted_visible_matched if predicted_visible_matched > 0 else 0.0
    global_recall = correct_visible / visible_gt_total if visible_gt_total > 0 else 0.0
    global_f1 = 0.0 if global_precision + global_recall == 0 else (
        2.0 * global_precision * global_recall / (global_precision + global_recall)
    )
    detection_visible_recall = matched_visible_grapes / visible_gt_total if visible_gt_total > 0 else 0.0
    point = summarize_point_cases(correct_pairs)

    return {
        "matching": {
            "iou_threshold": float(iou_threshold),
            "has_picking_threshold": float(has_picking_threshold),
            "visibility_score_key": visibility_score_key,
        },
        "instance_chain": {
            "visible_gt_total": int(visible_gt_total),
            "matched_grapes_iou50": int(matched_grapes),
            "matched_visible_grapes": int(matched_visible_grapes),
            "predicted_visible_grapes": int(predicted_visible_matched),
            "correct_visible_grapes": int(correct_visible),
            "has_picking_false_positive": int(false_visible_matched),
            "has_picking_false_negative": int(missed_visible),
            "detection_visible_recall": float(detection_visible_recall),
            "instance_visible_precision": float(instance_precision),
            "instance_visible_recall": float(instance_recall),
            "instance_visible_f1": float(instance_f1),
            "has_picking_precision": float(instance_precision),
            "has_picking_recall": float(instance_recall),
            "has_picking_f1": float(instance_f1),
            "point_pair_count": int(point["pair_count"]),
            "point_mean_l2_px": float(point["mean_l2_px"]),
            "point_median_l2_px": float(point["median_l2_px"]),
            "point_p90_l2_px": float(point["p90_l2_px"]),
            "point_mae_x_px": float(point["mean_abs_dx_px"]),
            "point_mae_y_px": float(point["mean_abs_dy_px"]),
            "ppl_sr_30": float(point["ppl_sr_30"]),
            "ppl_sr_50": float(point["ppl_sr_50"]),
        },
        "global_chain": {
            "visible_gt_total": int(visible_gt_total),
            "predicted_visible_grapes": int(predicted_visible_matched),
            "unmatched_or_unfiltered_predicted_visible_total": int(predicted_visible_total),
            "correct_visible_grapes": int(correct_visible),
            "global_visible_precision": float(global_precision),
            "global_visible_recall": float(global_recall),
            "global_visible_f1": float(global_f1),
        },
        "case_counts": {
            "correct_visible_pairs": len(correct_pairs),
            "false_positive_pairs": len(fp_pairs),
            "false_negative_pairs": len(fn_pairs),
        },
    }
