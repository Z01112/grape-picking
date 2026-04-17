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
        point_decode_clamp_x_abs=None,
        point_decode_clamp_y_min=None,
        point_decode_clamp_y_max=None,
        point_debug_roundtrip=False,
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
        self.point_decode_clamp_x_abs = None if point_decode_clamp_x_abs is None else float(point_decode_clamp_x_abs)
        self.point_decode_clamp_y_min = None if point_decode_clamp_y_min is None else float(point_decode_clamp_y_min)
        self.point_decode_clamp_y_max = None if point_decode_clamp_y_max is None else float(point_decode_clamp_y_max)
        self.point_debug_roundtrip = bool(point_debug_roundtrip)
        self.deploy_mode = False

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
        has_picking_scores = None
        has_picking_flags = None
        picking_points = None
        picking_offsets = None

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

            if self.use_focal_loss or scores.shape[1] != pred_has_picking.shape[1]:
                pred_has_picking = pred_has_picking.gather(
                    dim=1,
                    index=index.unsqueeze(-1).repeat(1, 1, pred_has_picking.shape[-1]),
                )
                pred_offsets = pred_offsets.gather(
                    dim=1,
                    index=index.unsqueeze(-1).repeat(1, 1, pred_offsets.shape[-1]),
                )

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
                )
            picking_points = absolute_points_from_boxes_and_offsets(
                abs_boxes_cxcywh,
                pred_offsets,
                mode=self.point_offset_mode,
                top_anchor_ratio=self.point_top_anchor_ratio,
            )
            picking_points = clamp_points_to_image(picking_points, orig_target_sizes)

            has_picking_scores = F.sigmoid(pred_has_picking).squeeze(-1)
            has_picking_flags = has_picking_scores >= self.has_picking_threshold
            picking_offsets = pred_offsets

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
            zipped = zip(labels, boxes, scores, has_picking_scores, has_picking_flags, picking_points, picking_offsets)

        for items in zipped:
            if has_picking_scores is None:
                lab, box, sco = items
                result = dict(labels=lab, boxes=box, scores=sco)
            else:
                lab, box, sco, has_score, has_flag, point_xy, offset_xy = items
                result = dict(
                    labels=lab,
                    boxes=box,
                    scores=sco,
                    has_picking_scores=has_score,
                    has_picking=has_flag,
                    picking_points=point_xy,
                    picking_offsets=offset_xy,
                )
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
