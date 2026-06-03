"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...rtv4.box_ops import box_iou
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator', 'GrapePointEvaluator']


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            self.summarize_per_category(coco_eval, iou_type)

    def get_category_lookup(self):
        cat_id_to_name = {
            int(cat_id): cat.get("name", str(cat_id))
            for cat_id, cat in getattr(self.coco_gt, "cats", {}).items()
        }
        if not cat_id_to_name and hasattr(self.coco_gt, "dataset"):
            cat_id_to_name = {
                int(cat["id"]): cat.get("name", str(cat["id"]))
                for cat in self.coco_gt.dataset.get("categories", [])
            }
        return cat_id_to_name

    def build_bbox_per_category_metrics(self, coco_eval):
        if not getattr(coco_eval, "eval", None):
            return {}

        precision = coco_eval.eval.get("precision")
        recall = coco_eval.eval.get("recall")
        if precision is None or recall is None:
            return {}

        cat_ids = [int(cat_id) for cat_id in getattr(coco_eval.params, "catIds", [])]
        if not cat_ids:
            return {}

        cat_id_to_name = self.get_category_lookup()
        iou_thrs = np.asarray(coco_eval.params.iouThrs)
        ap50_idx = int(np.argmin(np.abs(iou_thrs - 0.5)))
        area_idx = 0
        max_det_idx = len(coco_eval.params.maxDets) - 1

        metrics = {}
        for class_idx, cat_id in enumerate(cat_ids):
            class_name = cat_id_to_name.get(cat_id, str(cat_id))

            class_precision = precision[:, :, class_idx, area_idx, max_det_idx]
            class_precision = class_precision[class_precision > -1]
            ap = float(np.mean(class_precision)) if class_precision.size else float("nan")

            class_precision_50 = precision[ap50_idx, :, class_idx, area_idx, max_det_idx]
            class_precision_50 = class_precision_50[class_precision_50 > -1]
            ap50 = float(np.mean(class_precision_50)) if class_precision_50.size else float("nan")

            class_recall = recall[:, class_idx, area_idx, max_det_idx]
            class_recall = class_recall[class_recall > -1]
            ar100 = float(np.mean(class_recall)) if class_recall.size else float("nan")

            metrics[class_name] = {
                "category_id": cat_id,
                "AP": ap,
                "AP50": ap50,
                "AR100": ar100,
            }
        return metrics

    def get_per_class_metrics(self, iou_type="bbox"):
        if iou_type != "bbox":
            return {}
        coco_eval = self.coco_eval.get(iou_type)
        if coco_eval is None:
            return {}
        return self.build_bbox_per_category_metrics(coco_eval)

    def get_extra_metrics(self):
        return {}

    def summarize_per_category(self, coco_eval, iou_type):
        if iou_type != "bbox":
            return

        metrics = self.build_bbox_per_category_metrics(coco_eval)
        if not metrics:
            return

        print("Per-class bbox metrics:")
        for class_name, class_metrics in metrics.items():
            print(
                f"  {class_name:<12} AP={class_metrics['AP']:.4f}  "
                f"AP50={class_metrics['AP50']:.4f}  AR100={class_metrics['AR100']:.4f}"
            )

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()


@register()
class GrapePointEvaluator(CocoEvaluator):
    def __init__(self, coco_gt, iou_types, point_iou_threshold=0.5, has_picking_threshold=0.5):
        super().__init__(coco_gt, iou_types)
        self.point_iou_threshold = float(point_iou_threshold)
        self.has_picking_threshold = float(has_picking_threshold)
        self.gt_points_by_image = self._build_gt_lookup()
        self.point_records = []

    def cleanup(self):
        super().cleanup()
        self.point_records = []

    def synchronize_between_processes(self):
        super().synchronize_between_processes()
        gathered = dist_utils.all_gather(self.point_records)
        merged = []
        for part in gathered:
            merged.extend(part)
        self.point_records = merged

    def update(self, predictions):
        super().update(predictions)
        for image_id, prediction in predictions.items():
            self.point_records.append(self._evaluate_points_for_image(int(image_id), prediction))

    def summarize(self):
        super().summarize()
        extra = self.get_extra_metrics().get("grape_point_metrics", {})
        if extra:
            print("Picking point metrics:")
            for key, value in extra.items():
                if isinstance(value, float):
                    print(f"  {key:<28} {value:.4f}")
                else:
                    print(f"  {key:<28} {value}")

    def get_extra_metrics(self):
        if not self.point_records:
            return {"grape_point_metrics": {}}

        # The picking metrics are computed after IoU matching between predicted
        # grape boxes and GT grape boxes. They measure the instance-level chain:
        # grape matched -> visible point predicted -> point error accumulated.
        matched_grapes = sum(int(item["matched_grapes"]) for item in self.point_records)
        visible_gt_total = sum(int(item.get("visible_gt_total", 0)) for item in self.point_records)
        matched_visible_grapes = sum(int(item["matched_visible_grapes"]) for item in self.point_records)
        predicted_visible = sum(int(item["predicted_visible"]) for item in self.point_records)
        correct_visible = sum(int(item["correct_visible"]) for item in self.point_records)
        false_visible = sum(int(item["false_visible"]) for item in self.point_records)
        missed_visible = sum(int(item["missed_visible"]) for item in self.point_records)
        point_pairs = sum(int(item["point_pairs"]) for item in self.point_records)
        sum_abs_x = sum(float(item["sum_abs_x"]) for item in self.point_records)
        sum_abs_y = sum(float(item["sum_abs_y"]) for item in self.point_records)
        sum_l2 = sum(float(item["sum_l2"]) for item in self.point_records)

        # has_picking F1 evaluates the visible-picking-point decision on
        # IoU-matched grape instances.
        precision = float(correct_visible / predicted_visible) if predicted_visible > 0 else 0.0
        recall = float(correct_visible / matched_visible_grapes) if matched_visible_grapes > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
        detection_visible_recall = (
            float(matched_visible_grapes / visible_gt_total) if visible_gt_total > 0 else 0.0
        )
        global_visible_recall = float(correct_visible / visible_gt_total) if visible_gt_total > 0 else 0.0

        metrics = {
            "visible_gt_total": visible_gt_total,
            "matched_grapes_iou50": matched_grapes,
            "matched_visible_grapes": matched_visible_grapes,
            "predicted_visible_grapes": predicted_visible,
            "correct_visible_grapes": correct_visible,
            "detection_visible_recall": detection_visible_recall,
            "global_visible_recall": global_visible_recall,
            "instance_visible_precision": precision,
            "instance_visible_recall": recall,
            "instance_visible_f1": f1,
            "has_picking_precision": precision,
            "has_picking_recall": recall,
            "has_picking_f1": f1,
            "has_picking_false_positive": false_visible,
            "has_picking_false_negative": missed_visible,
            # point_pair_count counts matched grapes where both GT and
            # prediction are visible. mean L2 is the image-plane Euclidean
            # point error, and |dy| is the absolute vertical point error.
            "point_pair_count": point_pairs,
            "point_mae_x_px": float(sum_abs_x / point_pairs) if point_pairs > 0 else 0.0,
            "point_mae_y_px": float(sum_abs_y / point_pairs) if point_pairs > 0 else 0.0,
            "point_mean_l2_px": float(sum_l2 / point_pairs) if point_pairs > 0 else 0.0,
        }
        return {"grape_point_metrics": metrics}

    def _build_gt_lookup(self):
        lookup = {}
        for ann in getattr(self.coco_gt, "dataset", {}).get("annotations", []):
            image_id = int(ann["image_id"])
            bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x, y, w, h = [float(v) for v in bbox]
            point = ann.get("picking_point", [0.0, 0.0])
            if point is None:
                point = [0.0, 0.0]
            lookup.setdefault(image_id, []).append(
                {
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "has_picking": float(ann.get("has_picking", 0.0)),
                    "picking_point": [float(point[0]), float(point[1])],
                }
            )
        return lookup

    def _evaluate_points_for_image(self, image_id: int, prediction: dict):
        gt_entries = self.gt_points_by_image.get(image_id, [])
        visible_gt_total = sum(1 for item in gt_entries if float(item.get("has_picking", 0.0)) > 0.5)
        if not gt_entries:
            return {
                "visible_gt_total": 0,
                "matched_grapes": 0,
                "matched_visible_grapes": 0,
                "predicted_visible": 0,
                "correct_visible": 0,
                "false_visible": 0,
                "missed_visible": 0,
                "point_pairs": 0,
                "sum_abs_x": 0.0,
                "sum_abs_y": 0.0,
                "sum_l2": 0.0,
            }

        pred_boxes = prediction.get("boxes", torch.zeros((0, 4))).detach().cpu().to(torch.float32)
        pred_scores = prediction.get("scores", torch.zeros((pred_boxes.shape[0],))).detach().cpu().to(torch.float32)
        pred_has_scores = prediction.get("has_picking_scores")
        pred_points = prediction.get("picking_points")

        if pred_has_scores is None:
            pred_has_scores = torch.ones((pred_boxes.shape[0],), dtype=torch.float32)
        else:
            pred_has_scores = pred_has_scores.detach().cpu().to(torch.float32)

        if pred_points is None:
            pred_points = torch.zeros((pred_boxes.shape[0], 2), dtype=torch.float32)
        else:
            pred_points = pred_points.detach().cpu().to(torch.float32)

        gt_boxes = torch.as_tensor([item["bbox_xyxy"] for item in gt_entries], dtype=torch.float32)
        gt_has = torch.as_tensor([item["has_picking"] for item in gt_entries], dtype=torch.float32)
        gt_points = torch.as_tensor([item["picking_point"] for item in gt_entries], dtype=torch.float32)

        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            return {
                "visible_gt_total": visible_gt_total,
                "matched_grapes": 0,
                "matched_visible_grapes": 0,
                "predicted_visible": 0,
                "correct_visible": 0,
                "false_visible": 0,
                "missed_visible": int((gt_has > 0.5).sum().item()),
                "point_pairs": 0,
                "sum_abs_x": 0.0,
                "sum_abs_y": 0.0,
                "sum_l2": 0.0,
            }

        ious, _ = box_iou(pred_boxes, gt_boxes)
        pred_order = torch.argsort(pred_scores, descending=True)
        used_gt = set()
        matched_pairs = []

        for pred_idx in pred_order.tolist():
            best_iou = -1.0
            best_gt = None
            for gt_idx in range(gt_boxes.shape[0]):
                if gt_idx in used_gt:
                    continue
                iou = float(ious[pred_idx, gt_idx].item())
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            if best_gt is None or best_iou < self.point_iou_threshold:
                continue
            used_gt.add(best_gt)
            matched_pairs.append((pred_idx, best_gt))

        matched_visible_grapes = 0
        predicted_visible = 0
        correct_visible = 0
        false_visible = 0
        missed_visible = 0
        point_pairs = 0
        sum_abs_x = 0.0
        sum_abs_y = 0.0
        sum_l2 = 0.0

        for pred_idx, gt_idx in matched_pairs:
            gt_visible = bool(gt_has[gt_idx].item() > 0.5)
            pred_visible = bool(pred_has_scores[pred_idx].item() >= self.has_picking_threshold)
            if gt_visible:
                matched_visible_grapes += 1
            if pred_visible:
                predicted_visible += 1

            if gt_visible and pred_visible:
                correct_visible += 1
                pred_point = pred_points[pred_idx]
                gt_point = gt_points[gt_idx]
                diff = pred_point - gt_point
                # These sums become mean |dx|, mean |dy|, and mean L2.
                sum_abs_x += abs(float(diff[0].item()))
                sum_abs_y += abs(float(diff[1].item()))
                sum_l2 += float(torch.linalg.norm(diff, ord=2).item())
                point_pairs += 1
            elif gt_visible and not pred_visible:
                missed_visible += 1
            elif (not gt_visible) and pred_visible:
                false_visible += 1

        return {
            "visible_gt_total": visible_gt_total,
            "matched_grapes": len(matched_pairs),
            "matched_visible_grapes": matched_visible_grapes,
            "predicted_visible": predicted_visible,
            "correct_visible": correct_visible,
            "false_visible": false_visible,
            "missed_visible": missed_visible,
            "point_pairs": point_pairs,
            "sum_abs_x": sum_abs_x,
            "sum_abs_y": sum_abs_y,
            "sum_l2": sum_l2,
        }
