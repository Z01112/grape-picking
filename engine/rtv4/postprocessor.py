"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register
from .point_utils import absolute_points_from_boxes_and_offsets, clamp_points_to_image, assert_offset_roundtrip


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        use_picking_point=False,
        has_picking_threshold=0.5,
        point_offset_mode="center",
        point_top_anchor_ratio=0.0,
        point_anchor_x_ratio=0.5,
        point_decode_clamp_x_abs=None,
        point_decode_clamp_y_min=None,
        point_decode_clamp_y_max=None,
        point_debug_roundtrip=False,
        point_visibility_score_mode="has",
        point_quality_score_alpha=1.0,
        point_selector_score_alpha=1.0,
        point_accept_score_alpha=1.0,
        point_reliability_score_alpha=1.0,
        use_toproi_simcc_refiner=False,
        toproi_simcc_x_min=-0.60,
        toproi_simcc_x_max=0.60,
        toproi_simcc_y_min=-0.35,
        toproi_simcc_y_max=0.45,
        use_toproi_heatmap_refiner=False,
        toproi_heatmap_x_min=-0.60,
        toproi_heatmap_x_max=0.60,
        toproi_heatmap_y_min=-0.35,
        toproi_heatmap_y_max=0.45,
        use_grouped_picking_offsets=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.use_picking_point = use_picking_point
        self.has_picking_threshold = float(has_picking_threshold)
        self.point_offset_mode = str(point_offset_mode)
        self.point_top_anchor_ratio = float(point_top_anchor_ratio)
        self.point_anchor_x_ratio = float(point_anchor_x_ratio)
        self.point_decode_clamp_x_abs = None if point_decode_clamp_x_abs is None else float(point_decode_clamp_x_abs)
        self.point_decode_clamp_y_min = None if point_decode_clamp_y_min is None else float(point_decode_clamp_y_min)
        self.point_decode_clamp_y_max = None if point_decode_clamp_y_max is None else float(point_decode_clamp_y_max)
        self.point_debug_roundtrip = bool(point_debug_roundtrip)
        self.point_visibility_score_mode = str(point_visibility_score_mode).strip().lower()
        self.point_quality_score_alpha = float(point_quality_score_alpha)
        self.point_selector_score_alpha = float(point_selector_score_alpha)
        self.point_accept_score_alpha = float(point_accept_score_alpha)
        self.point_reliability_score_alpha = float(point_reliability_score_alpha)
        self.use_toproi_simcc_refiner = bool(use_toproi_simcc_refiner)
        self.toproi_simcc_x_min = float(toproi_simcc_x_min)
        self.toproi_simcc_x_max = float(toproi_simcc_x_max)
        self.toproi_simcc_y_min = float(toproi_simcc_y_min)
        self.toproi_simcc_y_max = float(toproi_simcc_y_max)
        self.use_toproi_heatmap_refiner = bool(use_toproi_heatmap_refiner)
        self.toproi_heatmap_x_min = float(toproi_heatmap_x_min)
        self.toproi_heatmap_x_max = float(toproi_heatmap_x_max)
        self.toproi_heatmap_y_min = float(toproi_heatmap_y_min)
        self.toproi_heatmap_y_max = float(toproi_heatmap_y_max)
        self.use_grouped_picking_offsets = bool(use_grouped_picking_offsets)
        if self.toproi_simcc_x_max <= self.toproi_simcc_x_min:
            self.toproi_simcc_x_max = self.toproi_simcc_x_min + 1.0
        if self.toproi_simcc_y_max <= self.toproi_simcc_y_min:
            self.toproi_simcc_y_max = self.toproi_simcc_y_min + 1.0
        if self.toproi_heatmap_x_max <= self.toproi_heatmap_x_min:
            self.toproi_heatmap_x_max = self.toproi_heatmap_x_min + 1.0
        if self.toproi_heatmap_y_max <= self.toproi_heatmap_y_min:
            self.toproi_heatmap_y_max = self.toproi_heatmap_y_min + 1.0
        if self.point_visibility_score_mode not in {"has", "quality", "selector", "accept", "reliability"}:
            raise ValueError(
                f"Unsupported point_visibility_score_mode={point_visibility_score_mode!r}; "
                "expected 'has', 'quality', 'selector', 'accept', or 'reliability'."
            )
        self.deploy_mode = False

    @staticmethod
    def _decode_simcc_offsets(
        logits_x: torch.Tensor,
        logits_y: torch.Tensor,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> torch.Tensor:
        dtype = logits_x.dtype
        device = logits_x.device
        x_bins = torch.linspace(float(x_min), float(x_max), logits_x.shape[-1], device=device, dtype=dtype)
        y_bins = torch.linspace(float(y_min), float(y_max), logits_y.shape[-1], device=device, dtype=dtype)
        prob_x = F.softmax(logits_x, dim=-1)
        prob_y = F.softmax(logits_y, dim=-1)
        off_x = (prob_x * x_bins).sum(dim=-1)
        off_y = (prob_y * y_bins).sum(dim=-1)
        return torch.stack((off_x, off_y), dim=-1)

    @staticmethod
    def _decode_heatmap_offsets(
        logits: torch.Tensor,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> torch.Tensor:
        dtype = logits.dtype
        device = logits.device
        height, width = logits.shape[-2:]
        x_bins = torch.linspace(float(x_min), float(x_max), width, device=device, dtype=dtype)
        y_bins = torch.linspace(float(y_min), float(y_max), height, device=device, dtype=dtype)
        prob = F.softmax(logits.flatten(-2), dim=-1).reshape_as(logits)
        prob_x = prob.sum(dim=-2)
        prob_y = prob.sum(dim=-1)
        off_x = (prob_x * x_bins).sum(dim=-1)
        off_y = (prob_y * y_bins).sum(dim=-1)
        return torch.stack((off_x, off_y), dim=-1)

    def extra_repr(self) -> str:
        return (
            f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, '
            f'num_top_queries={self.num_top_queries}, use_picking_point={self.use_picking_point}'
        )

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        boxes_cxcywh = boxes
        bbox_pred = torchvision.ops.box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        selected_boxes_cxcywh = boxes_cxcywh
        raw_has_picking_scores = None
        has_picking_scores = None
        visible_scores = None
        has_picking_flags = None
        picking_points = None
        picking_offsets = None
        grouped_picking_offsets = None
        point_quality_scores = None
        point_final_scores = None
        point_selector_scores = None
        point_selector_final_scores = None
        point_accept_scores = None
        point_accept_final_scores = None
        point_reliability_scores = None
        point_reliability_final_scores = None
        weak_heatmap_scores = None

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            selected_boxes_cxcywh = boxes_cxcywh.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, boxes_cxcywh.shape[-1]),
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                selected_boxes_cxcywh = torch.gather(
                    boxes_cxcywh,
                    dim=1,
                    index=index.unsqueeze(-1).tile(1, 1, boxes_cxcywh.shape[-1]),
                )
            else:
                selected_boxes_cxcywh = boxes_cxcywh

        if self.use_picking_point and 'pred_has_picking' in outputs and 'pred_picking_offsets' in outputs:
            pred_has_picking = outputs['pred_has_picking']
            pred_offsets = outputs['pred_picking_offsets']
            pred_grouped_offsets = outputs.get('pred_grouped_picking_offsets')
            pred_point_quality = outputs.get('pred_point_quality')
            pred_point_selector = outputs.get('pred_point_selector')
            pred_point_accept = outputs.get('pred_point_accept')
            pred_point_reliability = outputs.get('pred_point_reliability')
            pred_weak_heatmap = outputs.get('pred_weak_heatmap_score')
            pred_simcc_x = outputs.get('pred_toproi_simcc_x')
            pred_simcc_y = outputs.get('pred_toproi_simcc_y')
            pred_toproi_heatmap = outputs.get('pred_toproi_heatmap')

            # Keep point outputs aligned with the same top-k grape queries used
            # for labels/scores/boxes.
            if self.use_focal_loss or scores.shape[1] != pred_has_picking.shape[1]:
                pred_has_picking = pred_has_picking.gather(
                    dim=1,
                    index=index.unsqueeze(-1).repeat(1, 1, pred_has_picking.shape[-1]),
                )
                pred_offsets = pred_offsets.gather(
                    dim=1,
                    index=index.unsqueeze(-1).repeat(1, 1, pred_offsets.shape[-1]),
                )
                if pred_grouped_offsets is not None:
                    pred_grouped_offsets = pred_grouped_offsets.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_grouped_offsets.shape[-1]),
                    )
                if pred_point_quality is not None:
                    pred_point_quality = pred_point_quality.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_point_quality.shape[-1]),
                    )
                if pred_point_selector is not None:
                    pred_point_selector = pred_point_selector.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_point_selector.shape[-1]),
                    )
                if pred_point_accept is not None:
                    pred_point_accept = pred_point_accept.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_point_accept.shape[-1]),
                    )
                if pred_point_reliability is not None:
                    pred_point_reliability = pred_point_reliability.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_point_reliability.shape[-1]),
                    )
                if pred_weak_heatmap is not None:
                    pred_weak_heatmap = pred_weak_heatmap.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_weak_heatmap.shape[-1]),
                    )
                if pred_simcc_x is not None:
                    pred_simcc_x = pred_simcc_x.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_simcc_x.shape[-1]),
                    )
                if pred_simcc_y is not None:
                    pred_simcc_y = pred_simcc_y.gather(
                        dim=1,
                        index=index.unsqueeze(-1).repeat(1, 1, pred_simcc_y.shape[-1]),
                    )
                if pred_toproi_heatmap is not None:
                    pred_toproi_heatmap = pred_toproi_heatmap.gather(
                        dim=1,
                        index=index.unsqueeze(-1).unsqueeze(-1).repeat(
                            1, 1, pred_toproi_heatmap.shape[-2], pred_toproi_heatmap.shape[-1]
                        ),
                    )

            if self.use_grouped_picking_offsets and pred_grouped_offsets is not None:
                grouped_picking_offsets = pred_grouped_offsets
                pred_offsets = pred_grouped_offsets
            elif self.use_toproi_simcc_refiner and pred_simcc_x is not None and pred_simcc_y is not None:
                pred_offsets = self._decode_simcc_offsets(
                    pred_simcc_x,
                    pred_simcc_y,
                    self.toproi_simcc_x_min,
                    self.toproi_simcc_x_max,
                    self.toproi_simcc_y_min,
                    self.toproi_simcc_y_max,
                ).to(dtype=pred_offsets.dtype)
            if self.use_toproi_heatmap_refiner and pred_toproi_heatmap is not None:
                pred_offsets = self._decode_heatmap_offsets(
                    pred_toproi_heatmap,
                    self.toproi_heatmap_x_min,
                    self.toproi_heatmap_x_max,
                    self.toproi_heatmap_y_min,
                    self.toproi_heatmap_y_max,
                ).to(dtype=pred_offsets.dtype)

            if any(v is not None for v in (self.point_decode_clamp_x_abs, self.point_decode_clamp_y_min, self.point_decode_clamp_y_max)):
                pred_offsets = pred_offsets.clone()
                if self.point_decode_clamp_x_abs is not None:
                    pred_offsets[..., 0] = pred_offsets[..., 0].clamp(
                        min=-self.point_decode_clamp_x_abs,
                        max=self.point_decode_clamp_x_abs,
                    )
                if self.point_decode_clamp_y_min is not None or self.point_decode_clamp_y_max is not None:
                    min_y = self.point_decode_clamp_y_min if self.point_decode_clamp_y_min is not None else float("-inf")
                    max_y = self.point_decode_clamp_y_max if self.point_decode_clamp_y_max is not None else float("inf")
                    pred_offsets[..., 1] = pred_offsets[..., 1].clamp(min=min_y, max=max_y)

            abs_boxes_cxcywh = selected_boxes_cxcywh * orig_target_sizes.repeat(1, 2).unsqueeze(1)
            if self.point_debug_roundtrip:
                assert_offset_roundtrip(
                    selected_boxes_cxcywh.to(torch.float32),
                    pred_offsets.to(torch.float32),
                    mode=self.point_offset_mode,
                    top_anchor_ratio=self.point_top_anchor_ratio,
                    anchor_x_ratio=self.point_anchor_x_ratio,
                )
            # Decode bbox-normalized picking_offset into a 2D image-coordinate
            # picking_point for every retained query; has_picking then marks
            # which candidate points are valid.
            picking_points = absolute_points_from_boxes_and_offsets(
                abs_boxes_cxcywh,
                pred_offsets,
                mode=self.point_offset_mode,
                top_anchor_ratio=self.point_top_anchor_ratio,
                anchor_x_ratio=self.point_anchor_x_ratio,
            )
            picking_points = clamp_points_to_image(picking_points, orig_target_sizes)

            # The 0.5 threshold in the main config gates valid picking points
            # for evaluation or downstream use, but the candidate coordinates
            # remain available for visualization/debugging.
            raw_has_picking_scores = F.sigmoid(pred_has_picking).squeeze(-1)
            has_picking_scores = raw_has_picking_scores
            picking_offsets = pred_offsets
            if grouped_picking_offsets is None and pred_grouped_offsets is not None:
                grouped_picking_offsets = pred_grouped_offsets
            if pred_point_quality is not None:
                point_quality_scores = F.sigmoid(pred_point_quality).squeeze(-1)
                point_final_scores = raw_has_picking_scores * point_quality_scores.clamp(min=0.0, max=1.0).pow(
                    self.point_quality_score_alpha
                )
                if self.point_visibility_score_mode == "quality":
                    has_picking_scores = point_final_scores
            if pred_point_selector is not None:
                point_selector_scores = F.sigmoid(pred_point_selector).squeeze(-1)
                point_selector_final_scores = raw_has_picking_scores * point_selector_scores.clamp(min=0.0, max=1.0).pow(
                    self.point_selector_score_alpha
                )
                if self.point_visibility_score_mode == "selector":
                    has_picking_scores = point_selector_final_scores
            if pred_point_accept is not None:
                point_accept_scores = F.sigmoid(pred_point_accept).squeeze(-1)
                point_accept_final_scores = raw_has_picking_scores * point_accept_scores.clamp(min=0.0, max=1.0).pow(
                    self.point_accept_score_alpha
                )
                if self.point_visibility_score_mode == "accept":
                    has_picking_scores = point_accept_final_scores
            if pred_point_reliability is not None:
                point_reliability_scores = F.sigmoid(pred_point_reliability).squeeze(-1)
                point_reliability_final_scores = raw_has_picking_scores * point_reliability_scores.clamp(
                    min=0.0,
                    max=1.0,
                ).pow(self.point_reliability_score_alpha)
                if self.point_visibility_score_mode == "reliability":
                    has_picking_scores = point_reliability_final_scores
            if pred_weak_heatmap is not None:
                weak_heatmap_scores = F.sigmoid(pred_weak_heatmap).squeeze(-1)
            visible_scores = has_picking_scores
            has_picking_flags = has_picking_scores >= self.has_picking_threshold

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        zipped = zip(labels, boxes, scores)
        if has_picking_scores is not None:
            zipped = zip(
                labels,
                boxes,
                scores,
                raw_has_picking_scores,
                has_picking_scores,
                visible_scores,
                has_picking_flags,
                picking_points,
                picking_offsets,
                grouped_picking_offsets if grouped_picking_offsets is not None else [None] * len(labels),
                point_quality_scores if point_quality_scores is not None else [None] * len(labels),
                point_final_scores if point_final_scores is not None else [None] * len(labels),
                point_selector_scores if point_selector_scores is not None else [None] * len(labels),
                point_selector_final_scores if point_selector_final_scores is not None else [None] * len(labels),
                point_accept_scores if point_accept_scores is not None else [None] * len(labels),
                point_accept_final_scores if point_accept_final_scores is not None else [None] * len(labels),
                point_reliability_scores if point_reliability_scores is not None else [None] * len(labels),
                point_reliability_final_scores if point_reliability_final_scores is not None else [None] * len(labels),
                weak_heatmap_scores if weak_heatmap_scores is not None else [None] * len(labels),
            )

        for items in zipped:
            if has_picking_scores is None:
                lab, box, sco = items
                result = dict(labels=lab, boxes=box, scores=sco)
            else:
                lab, box, sco, raw_has_score, has_score, visible_score, has_flag, point_xy, offset_xy, grouped_offset_xy, quality_score, final_score, selector_score, selector_final_score, accept_score, accept_final_score, reliability_score, reliability_final_score, heatmap_score = items
                result = dict(
                    labels=lab,
                    boxes=box,
                    scores=sco,
                    raw_has_picking_scores=raw_has_score,
                    has_picking_scores=has_score,
                    visible_scores=visible_score,
                    has_picking=has_flag,
                    picking_points=point_xy,
                    picking_offsets=offset_xy,
                )
                if grouped_offset_xy is not None:
                    result["grouped_picking_offsets"] = grouped_offset_xy
                if quality_score is not None:
                    result["point_quality_scores"] = quality_score
                if final_score is not None:
                    result["point_final_scores"] = final_score
                if selector_score is not None:
                    result["point_selector_scores"] = selector_score
                if selector_final_score is not None:
                    result["point_selector_final_scores"] = selector_final_score
                if accept_score is not None:
                    result["point_accept_scores"] = accept_score
                if accept_final_score is not None:
                    result["point_accept_final_scores"] = accept_final_score
                if reliability_score is not None:
                    result["point_reliability_scores"] = reliability_score
                if reliability_final_score is not None:
                    result["point_reliability_final_scores"] = reliability_final_score
                if heatmap_score is not None:
                    result["weak_heatmap_scores"] = heatmap_score
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
