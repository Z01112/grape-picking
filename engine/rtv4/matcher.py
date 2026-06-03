"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .point_utils import absolute_points_from_boxes_and_offsets

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(
        self,
        weight_dict,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        use_point_cost=False,
        cost_point=0.0,
        point_cost_type='l1',
        point_cost_visible_only=True,
        point_cost_normalized=True,
        point_cost_start_epoch=20,
        point_cost_warmup_epochs=20,
        point_anchor_mode='top_center',
        point_top_anchor_ratio=0.12,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_point = float(weight_dict.get('cost_point', cost_point))

        self.use_point_cost = bool(use_point_cost)
        self.point_cost_type = str(point_cost_type).strip().lower()
        if self.point_cost_type not in ('l1', 'l2'):
            raise ValueError(f"Unsupported point_cost_type: {point_cost_type}")
        self.point_cost_visible_only = bool(point_cost_visible_only)
        self.point_cost_normalized = bool(point_cost_normalized)
        self.point_cost_start_epoch = int(point_cost_start_epoch)
        self.point_cost_warmup_epochs = max(int(point_cost_warmup_epochs), 0)
        self.point_anchor_mode = str(point_anchor_mode)
        self.point_top_anchor_ratio = float(point_top_anchor_ratio)
        self.current_epoch = None

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    def set_epoch(self, epoch) -> None:
        if epoch is None:
            self.current_epoch = None
            return
        self.current_epoch = int(epoch)

    def _point_cost_weight(self) -> float:
        if not self.use_point_cost or self.cost_point <= 0.0:
            return 0.0
        if self.current_epoch is None:
            return float(self.cost_point)
        if self.current_epoch < self.point_cost_start_epoch:
            return 0.0
        if self.point_cost_warmup_epochs <= 0:
            return float(self.cost_point)
        progress = (self.current_epoch - self.point_cost_start_epoch + 1) / float(self.point_cost_warmup_epochs)
        return float(self.cost_point) * max(0.0, min(1.0, progress))

    def _targets_to_point_cost_inputs(self, targets) -> tuple[torch.Tensor, torch.Tensor]:
        target_points, target_valid = [], []
        for target in targets:
            boxes = target["boxes"].to(torch.float32)
            has_picking = target.get("has_picking")
            if has_picking is None:
                visible = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
            else:
                visible = has_picking.to(device=boxes.device, dtype=torch.float32) > 0.5

            offsets = target.get("picking_offsets")
            if offsets is not None:
                points = absolute_points_from_boxes_and_offsets(
                    boxes,
                    offsets.to(device=boxes.device, dtype=torch.float32),
                    mode=self.point_anchor_mode,
                    top_anchor_ratio=self.point_top_anchor_ratio,
                )
            else:
                points = target.get("picking_points")
                if points is None:
                    points = torch.zeros((boxes.shape[0], 2), dtype=torch.float32, device=boxes.device)
                else:
                    points = points.to(device=boxes.device, dtype=torch.float32)
                    if self.point_cost_normalized and points.numel() > 0 and torch.nan_to_num(points).max() > 2.0:
                        size = target.get("size", target.get("orig_size"))
                        if size is not None:
                            size = size.to(device=points.device, dtype=torch.float32)
                            if size.numel() >= 2:
                                # Dataset sizes are [h, w] for transformed targets and [w, h] for orig_size.
                                if "size" in target:
                                    denom = torch.stack((size[1], size[0]))
                                else:
                                    denom = torch.stack((size[0], size[1]))
                                points = points / denom.clamp(min=1.0)

            valid = torch.isfinite(points).all(dim=-1)
            if self.point_cost_visible_only:
                valid = valid & visible
            target_points.append(points)
            target_valid.append(valid)

        if not target_points:
            device = next(iter(targets[0].values())).device if targets else torch.device("cpu")
            return torch.zeros((0, 2), device=device), torch.zeros((0,), dtype=torch.bool, device=device)
        return torch.cat(target_points, dim=0), torch.cat(target_valid, dim=0)

    def _compute_point_cost(self, outputs, out_bbox, targets) -> torch.Tensor | None:
        point_weight = self._point_cost_weight()
        if point_weight <= 0.0 or "pred_picking_offsets" not in outputs:
            return None
        offsets = outputs["pred_picking_offsets"].flatten(0, 1).to(dtype=torch.float32)
        pred_points = absolute_points_from_boxes_and_offsets(
            out_bbox,
            offsets,
            mode=self.point_anchor_mode,
            top_anchor_ratio=self.point_top_anchor_ratio,
        )
        target_points, target_valid = self._targets_to_point_cost_inputs(targets)
        if target_points.numel() == 0:
            return None
        if self.point_cost_type == 'l2':
            cost_point = torch.cdist(pred_points, target_points, p=2)
        else:
            cost_point = torch.cdist(pred_points, target_points, p=1)
        cost_point = cost_point * target_valid.to(cost_point.dtype).unsqueeze(0)
        return point_weight * cost_point

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, allow_point_cost=True):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if allow_point_cost:
            cost_point = self._compute_point_cost(outputs, out_bbox, targets)
            if cost_point is not None:
                C = C + cost_point
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # FIXME，RT-DETR, different way to set NaN
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list
