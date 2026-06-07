"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision
import math

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .point_utils import (
    absolute_points_from_boxes_and_offsets,
    top_roi_local_from_boxes_and_points,
)
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register

import logging

_logger = logging.getLogger(__name__)


@register()
class RTv4Criterion(nn.Module):
    """ This class computes the loss for RT-DETRv4.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 mal_alpha=None,
                 use_uni_set=True,
                 distill_adaptive_params=None,
                 point_loss_type='smooth_l1',
                 point_loss_beta=1.0,
                 wing_loss_omega=10.0,
                 wing_loss_epsilon=2.0,
                 point_coord_weight_x=1.0,
                 point_coord_weight_y=1.0,
                 point_small_grape_area_threshold=0.0,
                 point_small_grape_weight=1.0,
                 dense_positive_iou_thresh=0.0,
                 dense_positive_topk=0,
                 lambda_dense_has=0.0,
                 lambda_dense_point=0.0,
                 top_region_ratio=0.35,
                 top_margin_ratio=0.12,
                 lambda_geo=0.0,
                  point_locality_x_abs_max=1.0,
                  point_locality_y_min=-1.0,
                  point_locality_y_max=1.0,
                  point_locality_power=2.0,
                  point_quality_tau=30.0,
                  point_quality_offset_mode='center',
                  point_quality_top_anchor_ratio=0.0,
                  point_selector_tau=24.0,
                  point_selector_iou_power=1.0,
                  point_selector_dense_iou_thresh=0.10,
                  point_selector_dense_topk=10,
                  point_selector_offset_mode='center',
                  point_selector_top_anchor_ratio=0.0,
                  point_accept_tau=24.0,
                  point_accept_iou_power=1.0,
                  point_accept_iou_thresh=0.50,
                  point_accept_topk=16,
                  point_accept_offset_mode='center',
                  point_accept_top_anchor_ratio=0.0,
                  point_reliability_target='binary_ppl30',
                  point_reliability_tau=30.0,
                  point_reliability_invisible_weight=0.25,
                  point_reliability_offset_mode='center',
                  point_reliability_top_anchor_ratio=0.0,
                  weak_heatmap_sigma_px=30.0,
                  weak_heatmap_offset_mode='center',
                  weak_heatmap_top_anchor_ratio=0.0,
                  toproi_simcc_x_min=-0.60,
                  toproi_simcc_x_max=0.60,
                  toproi_simcc_y_min=-0.35,
                  toproi_simcc_y_max=0.45,
                  toproi_simcc_label_smoothing=0.02,
                  toproi_heatmap_x_min=-0.60,
                  toproi_heatmap_x_max=0.60,
                  toproi_heatmap_y_min=-0.35,
                  toproi_heatmap_y_max=0.45,
                  toproi_heatmap_sigma=1.25,
                  point_o2m_aux_enabled=False,
                  point_o2m_aux_iou_thresh=0.45,
                  point_o2m_aux_topk=2,
                  point_o2m_aux_has_weight=0.20,
                  point_o2m_aux_offset_weight=0.50,
                  use_point_lsd=False,
                  point_lsd_min_improve_px=3.0,
                  point_lsd_teacher_source='best_layer',
                  point_lsd_loss_type='smooth_l1',
                  point_lsd_weight=0.05,
                  point_lsd_offset_mode='top_center',
                  point_lsd_top_anchor_ratio=0.12,
                  point_lsd_anchor_x_ratio=0.5,
                  c2f_grid_size=7,
                  c2f_offset_mode='top_center',
                  c2f_top_anchor_ratio=0.12,
                  c2f_anchor_x_ratio=0.5,
                  c2f_toproi_width_scale=1.08,
                  c2f_toproi_y_min_ratio=-0.10,
                  c2f_toproi_y_max_ratio=0.40,
                  dpo_num_bins_x=96,
                  dpo_num_bins_y=96,
                  dpo_x_min=-1.0,
                  dpo_x_max=1.0,
                  dpo_y_min=-1.0,
                  dpo_y_max=1.0,
                  dpo_soft_sigma=1.5,
                  dpo_expectation_l1_beta=1.0,
                  has_logit_distill_loss_type='mse',
                  ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.base_weight_dict = copy.deepcopy(weight_dict)

        self.distill_adaptive_params = distill_adaptive_params
        self.point_loss_type = str(point_loss_type).lower()
        self.point_loss_beta = float(point_loss_beta)
        self.wing_loss_omega = float(wing_loss_omega)
        self.wing_loss_epsilon = float(wing_loss_epsilon)
        self.point_coord_weight_x = float(point_coord_weight_x)
        self.point_coord_weight_y = float(point_coord_weight_y)
        self.point_small_grape_area_threshold = float(point_small_grape_area_threshold)
        self.point_small_grape_weight = float(point_small_grape_weight)
        self.dense_positive_iou_thresh = float(dense_positive_iou_thresh)
        self.dense_positive_topk = max(int(dense_positive_topk), 0)
        self.lambda_dense_has = float(lambda_dense_has)
        self.lambda_dense_point = float(lambda_dense_point)
        self.top_region_ratio = float(top_region_ratio)
        self.top_margin_ratio = float(top_margin_ratio)
        self.lambda_geo = float(lambda_geo)
        self.point_locality_x_abs_max = float(point_locality_x_abs_max)
        self.point_locality_y_min = float(point_locality_y_min)
        self.point_locality_y_max = float(point_locality_y_max)
        self.point_locality_power = max(float(point_locality_power), 1.0)
        self.point_quality_tau = max(float(point_quality_tau), 1e-6)
        self.point_quality_offset_mode = str(point_quality_offset_mode)
        self.point_quality_top_anchor_ratio = float(point_quality_top_anchor_ratio)
        self.point_selector_tau = max(float(point_selector_tau), 1e-6)
        self.point_selector_iou_power = max(float(point_selector_iou_power), 0.0)
        self.point_selector_dense_iou_thresh = float(point_selector_dense_iou_thresh)
        self.point_selector_dense_topk = max(int(point_selector_dense_topk), 0)
        self.point_selector_offset_mode = str(point_selector_offset_mode)
        self.point_selector_top_anchor_ratio = float(point_selector_top_anchor_ratio)
        self.point_accept_tau = max(float(point_accept_tau), 1e-6)
        self.point_accept_iou_power = max(float(point_accept_iou_power), 0.0)
        self.point_accept_iou_thresh = float(point_accept_iou_thresh)
        self.point_accept_topk = max(int(point_accept_topk), 1)
        self.point_accept_offset_mode = str(point_accept_offset_mode)
        self.point_accept_top_anchor_ratio = float(point_accept_top_anchor_ratio)
        self.point_reliability_target = str(point_reliability_target).strip().lower()
        if self.point_reliability_target not in ('binary_ppl30', 'continuous_exp'):
            raise ValueError(f"Unsupported point_reliability_target: {point_reliability_target}")
        self.point_reliability_tau = max(float(point_reliability_tau), 1e-6)
        self.point_reliability_invisible_weight = max(float(point_reliability_invisible_weight), 0.0)
        self.point_reliability_offset_mode = str(point_reliability_offset_mode)
        self.point_reliability_top_anchor_ratio = float(point_reliability_top_anchor_ratio)
        self.weak_heatmap_sigma_px = max(float(weak_heatmap_sigma_px), 1e-6)
        self.weak_heatmap_offset_mode = str(weak_heatmap_offset_mode)
        self.weak_heatmap_top_anchor_ratio = float(weak_heatmap_top_anchor_ratio)
        self.toproi_simcc_x_min = float(toproi_simcc_x_min)
        self.toproi_simcc_x_max = float(toproi_simcc_x_max)
        self.toproi_simcc_y_min = float(toproi_simcc_y_min)
        self.toproi_simcc_y_max = float(toproi_simcc_y_max)
        if self.toproi_simcc_x_max <= self.toproi_simcc_x_min:
            self.toproi_simcc_x_max = self.toproi_simcc_x_min + 1.0
        if self.toproi_simcc_y_max <= self.toproi_simcc_y_min:
            self.toproi_simcc_y_max = self.toproi_simcc_y_min + 1.0
        self.toproi_simcc_label_smoothing = max(float(toproi_simcc_label_smoothing), 0.0)
        self.toproi_heatmap_x_min = float(toproi_heatmap_x_min)
        self.toproi_heatmap_x_max = float(toproi_heatmap_x_max)
        self.toproi_heatmap_y_min = float(toproi_heatmap_y_min)
        self.toproi_heatmap_y_max = float(toproi_heatmap_y_max)
        if self.toproi_heatmap_x_max <= self.toproi_heatmap_x_min:
            self.toproi_heatmap_x_max = self.toproi_heatmap_x_min + 1.0
        if self.toproi_heatmap_y_max <= self.toproi_heatmap_y_min:
            self.toproi_heatmap_y_max = self.toproi_heatmap_y_min + 1.0
        self.toproi_heatmap_sigma = max(float(toproi_heatmap_sigma), 1e-6)
        self.point_o2m_aux_enabled = bool(point_o2m_aux_enabled)
        self.point_o2m_aux_iou_thresh = float(point_o2m_aux_iou_thresh)
        self.point_o2m_aux_topk = max(int(point_o2m_aux_topk), 0)
        self.point_o2m_aux_has_weight = float(point_o2m_aux_has_weight)
        self.point_o2m_aux_offset_weight = float(point_o2m_aux_offset_weight)
        self.use_point_lsd = bool(use_point_lsd)
        self.point_lsd_min_improve_px = max(float(point_lsd_min_improve_px), 0.0)
        self.point_lsd_teacher_source = str(point_lsd_teacher_source).strip().lower()
        if self.point_lsd_teacher_source != 'best_layer':
            raise ValueError(f"Unsupported point_lsd_teacher_source: {point_lsd_teacher_source}")
        self.point_lsd_loss_type = str(point_lsd_loss_type).strip().lower()
        if self.point_lsd_loss_type not in ('l1', 'smooth_l1', 'mse'):
            raise ValueError(f"Unsupported point_lsd_loss_type: {point_lsd_loss_type}")
        self.point_lsd_weight = float(point_lsd_weight)
        self.point_lsd_offset_mode = str(point_lsd_offset_mode)
        self.point_lsd_top_anchor_ratio = float(point_lsd_top_anchor_ratio)
        self.point_lsd_anchor_x_ratio = float(point_lsd_anchor_x_ratio)
        self.c2f_grid_size = max(int(c2f_grid_size), 2)
        self.c2f_offset_mode = str(c2f_offset_mode)
        self.c2f_top_anchor_ratio = float(c2f_top_anchor_ratio)
        self.c2f_anchor_x_ratio = float(c2f_anchor_x_ratio)
        self.c2f_toproi_width_scale = float(c2f_toproi_width_scale)
        self.c2f_toproi_y_min_ratio = float(c2f_toproi_y_min_ratio)
        self.c2f_toproi_y_max_ratio = float(c2f_toproi_y_max_ratio)
        self.dpo_num_bins_x = max(int(dpo_num_bins_x), 2)
        self.dpo_num_bins_y = max(int(dpo_num_bins_y), 2)
        self.dpo_x_min = float(dpo_x_min)
        self.dpo_x_max = float(dpo_x_max)
        self.dpo_y_min = float(dpo_y_min)
        self.dpo_y_max = float(dpo_y_max)
        if self.dpo_x_max <= self.dpo_x_min:
            self.dpo_x_max = self.dpo_x_min + 1.0
        if self.dpo_y_max <= self.dpo_y_min:
            self.dpo_y_max = self.dpo_y_min + 1.0
        self.dpo_soft_sigma = max(float(dpo_soft_sigma), 1e-6)
        self.dpo_expectation_l1_beta = max(float(dpo_expectation_l1_beta), 1e-6)
        self.has_logit_distill_loss_type = str(has_logit_distill_loss_type).strip().lower()
        if self.has_logit_distill_loss_type not in ('mse', 'smooth_l1'):
            raise ValueError(f"Unsupported has_logit_distill_loss_type: {has_logit_distill_loss_type}")
        if 'loss_dense_has_picking' not in self.weight_dict and self.lambda_dense_has > 0.0:
            self.weight_dict['loss_dense_has_picking'] = self.lambda_dense_has
        if 'loss_dense_picking_offset' not in self.weight_dict and self.lambda_dense_point > 0.0:
            self.weight_dict['loss_dense_picking_offset'] = self.lambda_dense_point
        if 'loss_picking_geo' not in self.weight_dict and self.lambda_geo > 0.0:
            self.weight_dict['loss_picking_geo'] = self.lambda_geo
        if self.point_o2m_aux_enabled:
            self.weight_dict.setdefault('loss_point_o2m_has', self.point_o2m_aux_has_weight)
            self.weight_dict.setdefault('loss_point_o2m_offset', self.point_o2m_aux_offset_weight)
        if self.use_point_lsd:
            self.weight_dict.setdefault('loss_point_lsd', self.point_lsd_weight)
        self._dense_positive_cache = None


    def loss_has_logit_distill(self, outputs, targets, indices, num_boxes, **kwargs):
        student_logits = outputs.get('pred_has_picking')
        teacher_outputs = kwargs.get('teacher_outputs')
        teacher_logits = None
        if isinstance(teacher_outputs, dict):
            teacher_logits = teacher_outputs.get('pred_has_picking')
        if student_logits is None:
            anchor = outputs.get('pred_boxes')
            if anchor is None:
                return {'loss_has_logit_distill': torch.tensor(0.0, device=next(iter(outputs.values())).device)}
            return {'loss_has_logit_distill': anchor.sum() * 0.0}
        if teacher_logits is None:
            return {'loss_has_logit_distill': student_logits.sum() * 0.0}
        teacher_logits = teacher_logits.detach().to(device=student_logits.device, dtype=student_logits.dtype)
        if teacher_logits.shape != student_logits.shape:
            return {'loss_has_logit_distill': student_logits.sum() * 0.0}
        if self.has_logit_distill_loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(student_logits.float(), teacher_logits.float(), reduction='mean', beta=1.0)
        else:
            loss = F.mse_loss(student_logits.float(), teacher_logits.float(), reduction='mean')
        return {'loss_has_logit_distill': loss}


    def loss_distillation(self, outputs, targets, indices, num_boxes, **kwargs):
        student_feature_map = outputs.get('student_distill_output')
        teacher_feature_map = outputs.get('teacher_encoder_output')

        if student_feature_map is None or teacher_feature_map is None:
            return {'loss_distill': torch.tensor(0.0,
                                                 device=student_feature_map.device if student_feature_map is not None else torch.device(
                                                     'cuda'), requires_grad=True)}

        # _logger.info(f"[RTv4Criterion] Student feature map shape: {student_feature_map.shape}")
        # _logger.info(f"[RTv4Criterion] Teacher feature map shape: {teacher_feature_map.shape}")

        if student_feature_map.shape[1] != teacher_feature_map.shape[1]:
            _logger.error(
                f"[RTv4Criterion] Feature dimension mismatch! Student: {student_feature_map.shape[1]}, Teacher: {teacher_feature_map.shape[1]}")
            raise ValueError("Feature dimension mismatch between student and teacher for distillation loss.")

        H_s, W_s = student_feature_map.shape[2:]
        H_t, W_t = teacher_feature_map.shape[2:]

        target_h, target_w = H_s, W_s

        if (H_s, W_s) != (H_t, W_t):
            _logger.warning(
                f"[RTv4Criterion] Resizing teacher feature map from {H_t}x{W_t} to student's {H_s}x{W_s} for distillation.")
            teacher_feature_map = F.interpolate(teacher_feature_map,
                                                size=(target_h, target_w),
                                                mode='bilinear',
                                                align_corners=False)

        student_output_flat = student_feature_map.flatten(2).permute(0, 2, 1)
        teacher_output_flat = teacher_feature_map.flatten(2).permute(0, 2, 1)

        student_output_norm = F.normalize(student_output_flat, p=2, dim=-1)
        teacher_output_norm = F.normalize(teacher_output_flat, p=2, dim=-1)

        cos_sim = F.cosine_similarity(student_output_norm, teacher_output_norm, dim=-1)
        loss_distill = (1 - cos_sim).mean()

        return {'loss_distill': loss_distill}


    def _get_distillation_weight_for_epoch(self) -> float:
        fixed_weight = self.weight_dict.get('loss_distill', 0.0)
        return fixed_weight

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou( \
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, **kwargs):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                    self.fgl_targets_dn = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                    self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                     self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou( \
                box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max + 1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max + 1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                                                                          (F.log_softmax(pred_corners / T, dim=1),
                                                                           F.softmax(target_corners.detach() / T,
                                                                                     dim=1))).sum(-1)
                    if 'is_dn' not in outputs or self.num_pos is None or self.num_neg is None:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, (
                                    (~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                                self.num_pos + self.num_neg)

        return losses

    def loss_has_picking(self, outputs, targets, indices, num_boxes, **kwargs):
        """Binary visibility loss for the matched grape queries.

        has_picking is an instance attribute: it is supervised only after the
        normal Hungarian grape-box matching has paired each query with a GT box.
        """
        if 'pred_has_picking' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_has_picking'][idx].squeeze(-1)
        if src_logits.numel() == 0:
            return {'loss_has_picking': outputs['pred_has_picking'].sum() * 0.0}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_logits.dtype, device=src_logits.device)

        loss = F.binary_cross_entropy_with_logits(src_logits, target_has_picking, reduction='mean')
        return {'loss_has_picking': loss}

    def loss_has_stem(self, outputs, targets, indices, num_boxes, **kwargs):
        """Auxiliary stem visibility loss on matched grape queries only."""
        if 'pred_has_stem' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_has_stem'][idx].squeeze(-1)
        if src_logits.numel() == 0:
            return {'loss_has_stem': outputs['pred_has_stem'].sum() * 0.0}

        target_has_stem = torch.cat(
            [
                t.get('has_stem', torch.zeros_like(t['labels'], dtype=torch.float32))[j]
                for t, (_, j) in zip(targets, indices)
            ],
            dim=0,
        ).to(dtype=src_logits.dtype, device=src_logits.device)

        loss = F.binary_cross_entropy_with_logits(src_logits, target_has_stem, reduction='mean')
        return {'loss_has_stem': loss}

    def _gather_matched_point_examples(self, outputs, targets, indices, offset_key='pred_picking_offsets'):
        """Collect point targets for the same matched queries used by box loss."""
        idx = self._get_src_permutation_idx(indices)
        src_offsets = outputs[offset_key][idx]
        if src_offsets.numel() == 0:
            empty = outputs[offset_key].new_zeros((0, 2))
            empty_1 = outputs[offset_key].new_zeros((0,))
            return empty, empty_1, empty, empty

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        return src_offsets, target_has_picking, target_offsets, target_boxes

    def _collect_dense_positive_examples(
        self,
        outputs,
        targets,
        indices,
        iou_thresh=None,
        topk=None,
        use_cache=True,
    ):
        # Legacy experimental losses, not used by current baseline_replay /
        # v7_exp2 / small_weight configs.
        if use_cache and self._dense_positive_cache is not None:
            return self._dense_positive_cache
        iou_thresh = self.dense_positive_iou_thresh if iou_thresh is None else float(iou_thresh)
        topk = self.dense_positive_topk if topk is None else max(int(topk), 0)

        pred_boxes = outputs.get('pred_boxes')
        pred_has = outputs.get('pred_has_picking')
        pred_offsets = outputs.get('pred_picking_offsets')
        if pred_boxes is None or pred_has is None or pred_offsets is None:
            if use_cache:
                self._dense_positive_cache = None
            return None
        if topk <= 0 or iou_thresh <= 0.0:
            if use_cache:
                self._dense_positive_cache = None
            return None

        batch_indices = []
        query_indices = []
        gt_indices = []
        for batch_id, target in enumerate(targets):
            gt_boxes = target.get('boxes')
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_id].detach()),
                box_cxcywh_to_xyxy(gt_boxes.detach()),
            )
            if ious.numel() == 0:
                continue
            best_ious, best_gt = ious.max(dim=1)
            matched_src = set(int(v) for v in indices[batch_id][0].detach().cpu().tolist())
            for gt_id in range(gt_boxes.shape[0]):
                candidate_pred = torch.nonzero(
                    (best_gt == gt_id) & (best_ious >= iou_thresh),
                    as_tuple=False,
                ).flatten()
                if candidate_pred.numel() == 0:
                    continue
                sorted_pred = candidate_pred[best_ious[candidate_pred].argsort(descending=True)]
                kept = 0
                for pred_id in sorted_pred.tolist():
                    if int(pred_id) in matched_src:
                        continue
                    batch_indices.append(batch_id)
                    query_indices.append(int(pred_id))
                    gt_indices.append(int(gt_id))
                    kept += 1
                    if kept >= topk:
                        break

        if not batch_indices:
            result = {
                'pred_has_logits': pred_has.new_zeros((0,)),
                'pred_offsets': pred_offsets.new_zeros((0, 2)),
                'pred_boxes': pred_boxes.new_zeros((0, 4)),
                'target_has_picking': pred_offsets.new_zeros((0,)),
                'target_offsets': pred_offsets.new_zeros((0, 2)),
                'target_boxes': pred_offsets.new_zeros((0, 4)),
                'target_sizes': pred_offsets.new_zeros((0, 2)),
                'batch_indices': pred_boxes.new_zeros((0,), dtype=torch.long),
                'query_indices': pred_boxes.new_zeros((0,), dtype=torch.long),
            }
            if use_cache:
                self._dense_positive_cache = result
            return result

        batch_tensor = torch.as_tensor(batch_indices, device=pred_boxes.device, dtype=torch.long)
        query_tensor = torch.as_tensor(query_indices, device=pred_boxes.device, dtype=torch.long)
        target_tensor = torch.as_tensor(gt_indices, device=pred_boxes.device, dtype=torch.long)

        pred_has_logits = pred_has[batch_tensor, query_tensor, 0]
        pred_offsets_sel = pred_offsets[batch_tensor, query_tensor]
        pred_boxes_sel = pred_boxes[batch_tensor, query_tensor]

        target_has_list = []
        target_offset_list = []
        target_box_list = []
        target_size_list = []
        for batch_id, gt_id in zip(batch_indices, gt_indices):
            target = targets[batch_id]
            target_has_list.append(target['has_picking'][gt_id])
            target_offset_list.append(target['picking_offsets'][gt_id])
            target_box_list.append(target['boxes'][gt_id])
            target_size_list.append(target['orig_size'])

        result = {
            'pred_has_logits': pred_has_logits,
            'pred_offsets': pred_offsets_sel,
            'pred_boxes': pred_boxes_sel,
            'target_has_picking': torch.stack(target_has_list).to(dtype=pred_offsets_sel.dtype, device=pred_offsets_sel.device),
            'target_offsets': torch.stack(target_offset_list).to(dtype=pred_offsets_sel.dtype, device=pred_offsets_sel.device),
            'target_boxes': torch.stack(target_box_list).to(dtype=pred_offsets_sel.dtype, device=pred_offsets_sel.device),
            'target_sizes': torch.stack(target_size_list).to(dtype=pred_offsets_sel.dtype, device=pred_offsets_sel.device),
            'batch_indices': batch_tensor,
            'query_indices': query_tensor,
        }
        if use_cache:
            self._dense_positive_cache = result
        return result

    def point_o2m_aux_losses(self, outputs, targets, indices):
        """Training-only one-to-many point supervision for aux/pre queries.

        This stays outside self.losses so the final decoder output is not
        directly optimized by extra dense positives.
        """
        if not self.point_o2m_aux_enabled:
            return {}
        if 'pred_has_picking' not in outputs or 'pred_picking_offsets' not in outputs:
            return {}
        dense = self._collect_dense_positive_examples(
            outputs,
            targets,
            indices,
            iou_thresh=self.point_o2m_aux_iou_thresh,
            topk=self.point_o2m_aux_topk,
            use_cache=False,
        )
        if dense is None or dense['pred_has_logits'].numel() == 0:
            zero = outputs['pred_has_picking'].sum() * 0.0
            return {
                'loss_point_o2m_has': zero,
                'loss_point_o2m_offset': zero,
            }

        has_loss = F.binary_cross_entropy_with_logits(
            dense['pred_has_logits'],
            dense['target_has_picking'],
            reduction='mean',
        )
        valid = dense['target_has_picking'] > 0.5
        if valid.any():
            point_loss = self.compute_point_loss(dense['pred_offsets'][valid], dense['target_offsets'][valid])
            coord_weights = torch.as_tensor(
                [self.point_coord_weight_x, self.point_coord_weight_y],
                dtype=point_loss.dtype,
                device=point_loss.device,
            ).view(1, 2)
            offset_loss = (point_loss * coord_weights).sum() / max(int(valid.sum().item()), 1)
        else:
            offset_loss = outputs['pred_picking_offsets'].sum() * 0.0
        return {
            'loss_point_o2m_has': has_loss,
            'loss_point_o2m_offset': offset_loss,
        }

    def loss_picking_offset(self, outputs, targets, indices, num_boxes, **kwargs):
        """Point-offset regression loss for visible picking points only."""
        if 'pred_picking_offsets' not in outputs:
            return {}

        src_offsets, target_has_picking, target_offsets, target_boxes = self._gather_matched_point_examples(outputs, targets, indices)
        if src_offsets.numel() == 0:
            return {'loss_picking_offset': outputs['pred_picking_offsets'].sum() * 0.0}

        # Invisible or undefined picking points should not force a point target.
        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_picking_offset': outputs['pred_picking_offsets'].sum() * 0.0}

        point_loss = self.compute_point_loss(src_offsets[valid], target_offsets[valid])
        # The main GPPoint-DETR config sets y weight higher than x to emphasize
        # the historical vertical drift error (|dy|).
        coord_weights = torch.as_tensor(
            [self.point_coord_weight_x, self.point_coord_weight_y],
            dtype=point_loss.dtype,
            device=point_loss.device,
        ).view(1, 2)
        point_loss = point_loss * coord_weights

        sample_weights = torch.ones((int(valid.sum().item()), 1), dtype=point_loss.dtype, device=point_loss.device)
        if self.point_small_grape_weight > 1.0 and self.point_small_grape_area_threshold > 0.0:
            # Disabled in the main model. Enabled only for the small_weight
            # auxiliary experiment to up-weight small-grape point samples.
            areas = (target_boxes[valid, 2] * target_boxes[valid, 3]).to(dtype=point_loss.dtype)
            small_mask = areas <= self.point_small_grape_area_threshold
            if small_mask.any():
                sample_weights[small_mask] = self.point_small_grape_weight

        loss = (point_loss * sample_weights).sum() / max(int(valid.sum().item()), 1)
        return {'loss_picking_offset': loss}

    def _build_c2f_targets(self, outputs, targets, indices):
        required = ('pred_c2f_grid_logits', 'pred_c2f_cell_residuals', 'pred_boxes')
        if any(key not in outputs for key in required):
            return None
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if src_boxes.numel() == 0:
            return None

        target_has = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(device=src_boxes.device, dtype=torch.float32)
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(device=src_boxes.device, dtype=torch.float32)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(device=src_boxes.device, dtype=torch.float32)
        target_points = absolute_points_from_boxes_and_offsets(
            target_boxes,
            target_offsets,
            mode=self.c2f_offset_mode,
            top_anchor_ratio=self.c2f_top_anchor_ratio,
            anchor_x_ratio=self.c2f_anchor_x_ratio,
        )
        target_local, in_roi = top_roi_local_from_boxes_and_points(
            src_boxes.detach(),
            target_points,
            roi_width_scale=self.c2f_toproi_width_scale,
            roi_y_min_ratio=self.c2f_toproi_y_min_ratio,
            roi_y_max_ratio=self.c2f_toproi_y_max_ratio,
        )
        visible = target_has > 0.5
        valid = visible & in_roi
        grid = self.c2f_grid_size
        clamped = target_local.clamp(min=0.0, max=1.0 - 1e-6)
        cell_x = torch.floor(clamped[:, 0] * grid).to(dtype=torch.long).clamp(0, grid - 1)
        cell_y = torch.floor(clamped[:, 1] * grid).to(dtype=torch.long).clamp(0, grid - 1)
        cell_index = cell_y * grid + cell_x
        center = torch.stack(
            (
                (cell_x.to(dtype=target_local.dtype) + 0.5) / float(grid),
                (cell_y.to(dtype=target_local.dtype) + 0.5) / float(grid),
            ),
            dim=-1,
        )
        residual_target = target_local - center
        return {
            'idx': idx,
            'valid': valid,
            'visible': visible,
            'in_roi': in_roi,
            'cell_index': cell_index,
            'residual_target': residual_target,
            'target_local': target_local,
        }

    def loss_c2f_coarse(self, outputs, targets, indices, num_boxes, **kwargs):
        if 'pred_c2f_grid_logits' not in outputs:
            return {}
        built = self._build_c2f_targets(outputs, targets, indices)
        if built is None:
            return {'loss_c2f_coarse_ce': outputs['pred_c2f_grid_logits'].sum() * 0.0}
        idx = built['idx']
        valid = built['valid']
        logits = outputs['pred_c2f_grid_logits'][idx]
        if not valid.any():
            return {'loss_c2f_coarse_ce': outputs['pred_c2f_grid_logits'].sum() * 0.0}
        loss = F.cross_entropy(logits[valid], built['cell_index'][valid], reduction='mean')
        return {'loss_c2f_coarse_ce': loss}

    def loss_c2f_fine(self, outputs, targets, indices, num_boxes, **kwargs):
        if 'pred_c2f_cell_residuals' not in outputs:
            return {}
        built = self._build_c2f_targets(outputs, targets, indices)
        if built is None:
            return {'loss_c2f_fine_l1': outputs['pred_c2f_cell_residuals'].sum() * 0.0}
        idx = built['idx']
        valid = built['valid']
        residuals = outputs['pred_c2f_cell_residuals'][idx]
        if not valid.any():
            return {'loss_c2f_fine_l1': outputs['pred_c2f_cell_residuals'].sum() * 0.0}
        gather_index = built['cell_index'].view(-1, 1, 1).expand(-1, 1, 2)
        pred = residuals.gather(1, gather_index).squeeze(1)
        loss = F.smooth_l1_loss(pred[valid], built['residual_target'][valid].to(dtype=pred.dtype), reduction='mean')
        return {'loss_c2f_fine_l1': loss}

    def _dpo_soft_target(self, values: torch.Tensor, axis: str) -> torch.Tensor:
        if axis == 'x':
            bins = torch.linspace(
                self.dpo_x_min,
                self.dpo_x_max,
                self.dpo_num_bins_x,
                device=values.device,
                dtype=values.dtype,
            )
        else:
            bins = torch.linspace(
                self.dpo_y_min,
                self.dpo_y_max,
                self.dpo_num_bins_y,
                device=values.device,
                dtype=values.dtype,
            )
        if values.numel() == 0:
            return values.new_zeros((0, bins.numel()))
        if bins.numel() > 1:
            bin_step = (bins[-1] - bins[0]).abs() / (bins.numel() - 1)
        else:
            bin_step = values.new_tensor(1.0)
        sigma = self.dpo_soft_sigma * bin_step.clamp_min(1e-6)
        target = torch.exp(-0.5 * ((bins.view(1, -1) - values.view(-1, 1)) / sigma) ** 2)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def loss_dpo(self, outputs, targets, indices, num_boxes, **kwargs):
        """Distributional point-offset loss for final matched visible queries only."""
        required = ('pred_dpo_logits_x', 'pred_dpo_logits_y', 'pred_dpo_offsets')
        if any(key not in outputs for key in required):
            return {}
        idx = self._get_src_permutation_idx(indices)
        logits_x = outputs['pred_dpo_logits_x'][idx]
        logits_y = outputs['pred_dpo_logits_y'][idx]
        pred_offsets = outputs['pred_dpo_offsets'][idx]
        if logits_x.numel() == 0:
            zero = outputs['pred_dpo_logits_x'].sum() * 0.0
            return {
                'loss_dpo_x': zero,
                'loss_dpo_y': zero,
                'loss_dpo_expectation_l1': zero,
            }

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=pred_offsets.dtype, device=pred_offsets.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=pred_offsets.dtype, device=pred_offsets.device)
        valid = target_has_picking > 0.5
        if not valid.any():
            zero = outputs['pred_dpo_logits_x'].sum() * 0.0
            return {
                'loss_dpo_x': zero,
                'loss_dpo_y': zero,
                'loss_dpo_expectation_l1': zero,
            }

        target_x = self._dpo_soft_target(target_offsets[valid, 0], 'x')
        target_y = self._dpo_soft_target(target_offsets[valid, 1], 'y')
        loss_x = -(target_x * F.log_softmax(logits_x[valid].float(), dim=-1)).sum(dim=-1).mean()
        loss_y = -(target_y * F.log_softmax(logits_y[valid].float(), dim=-1)).sum(dim=-1).mean()
        loss_expectation = F.smooth_l1_loss(
            pred_offsets[valid].float(),
            target_offsets[valid].float(),
            reduction='none',
            beta=self.dpo_expectation_l1_beta,
        ).sum(dim=-1).mean()
        return {
            'loss_dpo_x': loss_x,
            'loss_dpo_y': loss_y,
            'loss_dpo_expectation_l1': loss_expectation,
        }

    def loss_grouped_picking_offset(self, outputs, targets, indices, num_boxes, **kwargs):
        """Offset loss for the grouped single-keypoint picking query branch."""
        if 'pred_grouped_picking_offsets' not in outputs:
            return {}

        src_offsets, target_has_picking, target_offsets, target_boxes = self._gather_matched_point_examples(
            outputs,
            targets,
            indices,
            offset_key='pred_grouped_picking_offsets',
        )
        if src_offsets.numel() == 0:
            return {'loss_grouped_picking_offset': outputs['pred_grouped_picking_offsets'].sum() * 0.0}

        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_grouped_picking_offset': outputs['pred_grouped_picking_offsets'].sum() * 0.0}

        point_loss = self.compute_point_loss(src_offsets[valid], target_offsets[valid])
        coord_weights = torch.as_tensor(
            [self.point_coord_weight_x, self.point_coord_weight_y],
            dtype=point_loss.dtype,
            device=point_loss.device,
        ).view(1, 2)
        point_loss = point_loss * coord_weights

        sample_weights = torch.ones((int(valid.sum().item()), 1), dtype=point_loss.dtype, device=point_loss.device)
        if self.point_small_grape_weight > 1.0 and self.point_small_grape_area_threshold > 0.0:
            areas = (target_boxes[valid, 2] * target_boxes[valid, 3]).to(dtype=point_loss.dtype)
            small_mask = areas <= self.point_small_grape_area_threshold
            if small_mask.any():
                sample_weights[small_mask] = self.point_small_grape_weight

        loss = (point_loss * sample_weights).sum() / max(int(valid.sum().item()), 1)
        return {'loss_grouped_picking_offset': loss}

    def _matched_target_sizes(self, outputs, targets, indices, dtype, device):
        target_sizes = []
        for target, (_, j) in zip(targets, indices):
            size = target.get('orig_size', target.get('size'))
            if size is None:
                size = outputs['pred_boxes'].new_tensor([1.0, 1.0])
            size = size.to(device=device, dtype=dtype).view(1, 2).repeat(len(j), 1)
            target_sizes.append(size)
        if not target_sizes:
            return outputs['pred_boxes'].new_zeros((0, 2), dtype=dtype, device=device)
        return torch.cat(target_sizes, dim=0)

    def _decode_layer_points_for_matched_queries(self, layer_outputs, idx, target_sizes):
        boxes = layer_outputs['pred_boxes'][idx].to(dtype=target_sizes.dtype, device=target_sizes.device)
        offsets = layer_outputs['pred_picking_offsets'][idx].to(dtype=target_sizes.dtype, device=target_sizes.device)
        scale_xyxy = target_sizes.repeat(1, 2)
        points = absolute_points_from_boxes_and_offsets(
            boxes * scale_xyxy,
            offsets,
            mode=self.point_lsd_offset_mode,
            top_anchor_ratio=self.point_lsd_top_anchor_ratio,
            anchor_x_ratio=self.point_lsd_anchor_x_ratio,
        )
        return offsets, points

    def _point_lsd_loss_raw(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.point_lsd_loss_type == 'l1':
            return F.l1_loss(pred, target, reduction='none')
        if self.point_lsd_loss_type == 'mse':
            return F.mse_loss(pred, target, reduction='none')
        return F.smooth_l1_loss(pred, target, reduction='none', beta=max(self.point_loss_beta, 1e-6))

    def loss_point_lsd(self, outputs, targets, indices, num_boxes, **kwargs):
        """Training-only best-layer point localization self-distillation.

        The final matched visible query is supervised toward the detached
        picking offset from the same query's best aux/pre/final layer only when
        that layer is at least point_lsd_min_improve_px better in pixel L2.
        """
        if not self.use_point_lsd:
            return {}
        required = ('pred_picking_offsets', 'pred_boxes')
        if any(key not in outputs for key in required):
            return {}
        aux_layers = [
            item for item in outputs.get('aux_outputs', [])
            if isinstance(item, dict) and all(key in item for key in required)
        ]
        pre = outputs.get('pre_outputs')
        if isinstance(pre, dict) and all(key in pre for key in required):
            aux_layers.append(pre)
        if not aux_layers:
            return {'loss_point_lsd': outputs['pred_picking_offsets'].sum() * 0.0}

        idx = self._get_src_permutation_idx(indices)
        final_offsets = outputs['pred_picking_offsets'][idx]
        if final_offsets.numel() == 0:
            return {'loss_point_lsd': outputs['pred_picking_offsets'].sum() * 0.0}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=final_offsets.dtype, device=final_offsets.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=final_offsets.dtype, device=final_offsets.device)
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=final_offsets.dtype, device=final_offsets.device)
        target_sizes = self._matched_target_sizes(
            outputs,
            targets,
            indices,
            dtype=final_offsets.dtype,
            device=final_offsets.device,
        )

        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_point_lsd': outputs['pred_picking_offsets'].sum() * 0.0}

        scale_xyxy = target_sizes.repeat(1, 2)
        target_points = absolute_points_from_boxes_and_offsets(
            target_boxes * scale_xyxy,
            target_offsets,
            mode=self.point_lsd_offset_mode,
            top_anchor_ratio=self.point_lsd_top_anchor_ratio,
            anchor_x_ratio=self.point_lsd_anchor_x_ratio,
        )

        layer_offsets = []
        layer_l2 = []
        for layer in [outputs] + aux_layers:
            offsets, points = self._decode_layer_points_for_matched_queries(layer, idx, target_sizes)
            layer_offsets.append(offsets)
            layer_l2.append(torch.linalg.norm(points - target_points, ord=2, dim=-1))
        offsets_stack = torch.stack(layer_offsets, dim=1)
        l2_stack = torch.stack(layer_l2, dim=1)

        final_l2 = l2_stack[:, 0]
        best_l2, best_layer_idx = l2_stack.min(dim=1)
        improve = final_l2 - best_l2
        active = valid & (best_layer_idx > 0) & (improve >= self.point_lsd_min_improve_px)
        if not active.any():
            return {'loss_point_lsd': outputs['pred_picking_offsets'].sum() * 0.0}

        row_idx = torch.arange(offsets_stack.shape[0], device=offsets_stack.device)
        teacher_offsets = offsets_stack[row_idx, best_layer_idx].detach()
        loss_raw = self._point_lsd_loss_raw(final_offsets[active], teacher_offsets[active])
        coord_weights = torch.as_tensor(
            [self.point_coord_weight_x, self.point_coord_weight_y],
            dtype=loss_raw.dtype,
            device=loss_raw.device,
        ).view(1, 2)
        loss = (loss_raw * coord_weights).sum() / max(int(active.sum().item()), 1)
        return {'loss_point_lsd': loss}

    def loss_point_quality(self, outputs, targets, indices, num_boxes, **kwargs):
        """Quality calibration for visible matched picking points.

        quality_target = exp(-detach(L2_pixel) / tau).  Detaching the L2 target
        keeps this head as a reliability estimator while loss_picking_offset
        remains the coordinate regression objective.
        """
        if 'pred_point_quality' not in outputs or 'pred_picking_offsets' not in outputs or 'pred_boxes' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_quality_logits = outputs['pred_point_quality'][idx].squeeze(-1)
        src_offsets = outputs['pred_picking_offsets'][idx]
        src_boxes = outputs['pred_boxes'][idx]
        if src_quality_logits.numel() == 0:
            return {'loss_point_quality': outputs['pred_point_quality'].sum() * 0.0}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_quality_logits.dtype, device=src_quality_logits.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_boxes.dtype, device=src_boxes.device)
        target_sizes = torch.cat(
            [
                t['orig_size'].to(device=src_boxes.device, dtype=src_boxes.dtype).view(1, 2).repeat(len(j), 1)
                for t, (_, j) in zip(targets, indices)
            ],
            dim=0,
        )

        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_point_quality': outputs['pred_point_quality'].sum() * 0.0}

        scale_xyxy = target_sizes[valid].repeat(1, 2)
        pred_points = absolute_points_from_boxes_and_offsets(
            src_boxes[valid] * scale_xyxy,
            src_offsets[valid],
            mode=self.point_quality_offset_mode,
            top_anchor_ratio=self.point_quality_top_anchor_ratio,
        )
        target_points = absolute_points_from_boxes_and_offsets(
            target_boxes[valid] * scale_xyxy,
            target_offsets[valid],
            mode=self.point_quality_offset_mode,
            top_anchor_ratio=self.point_quality_top_anchor_ratio,
        )
        l2_pixel = torch.linalg.norm(pred_points - target_points, ord=2, dim=-1).detach()
        quality_target = torch.exp(-l2_pixel / self.point_quality_tau).to(dtype=src_quality_logits.dtype)
        loss = F.binary_cross_entropy_with_logits(src_quality_logits[valid], quality_target, reduction='mean')
        return {'loss_point_quality': loss}

    def loss_point_selector(self, outputs, targets, indices, num_boxes, **kwargs):
        """Detached query-candidate selector supervised by visible point quality.

        Unlike point_quality, this loss includes dense IoU-local candidates in
        addition to the Hungarian match.  It is intended to learn which query
        should expose the final picking point when several predicted grape
        candidates overlap the same visible GT.
        """
        required = ('pred_point_selector', 'pred_picking_offsets', 'pred_boxes')
        if any(key not in outputs for key in required):
            return {}

        selector_groups = []
        offset_groups = []
        box_groups = []
        has_groups = []
        target_offset_groups = []
        target_box_groups = []
        target_size_groups = []

        idx = self._get_src_permutation_idx(indices)
        matched_selector_logits = outputs['pred_point_selector'][idx].squeeze(-1)
        if matched_selector_logits.numel() > 0:
            selector_groups.append(matched_selector_logits)
            offset_groups.append(outputs['pred_picking_offsets'][idx])
            box_groups.append(outputs['pred_boxes'][idx])
            has_groups.append(
                torch.cat([t['has_picking'][j] for t, (_, j) in zip(targets, indices)], dim=0).to(
                    dtype=matched_selector_logits.dtype,
                    device=matched_selector_logits.device,
                )
            )
            target_offset_groups.append(
                torch.cat([t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)], dim=0).to(
                    dtype=outputs['pred_picking_offsets'].dtype,
                    device=outputs['pred_picking_offsets'].device,
                )
            )
            target_box_groups.append(
                torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0).to(
                    dtype=outputs['pred_boxes'].dtype,
                    device=outputs['pred_boxes'].device,
                )
            )
            target_size_groups.append(
                torch.cat(
                    [
                        t['orig_size'].to(device=outputs['pred_boxes'].device, dtype=outputs['pred_boxes'].dtype)
                        .view(1, 2)
                        .repeat(len(j), 1)
                        for t, (_, j) in zip(targets, indices)
                    ],
                    dim=0,
                )
            )

        dense = self._collect_dense_positive_examples(
            outputs,
            targets,
            indices,
            iou_thresh=self.point_selector_dense_iou_thresh,
            topk=self.point_selector_dense_topk,
            use_cache=False,
        )
        if dense is not None and dense['pred_offsets'].numel() > 0:
            selector_groups.append(outputs['pred_point_selector'][dense['batch_indices'], dense['query_indices'], 0])
            offset_groups.append(dense['pred_offsets'])
            box_groups.append(dense['pred_boxes'])
            has_groups.append(dense['target_has_picking'])
            target_offset_groups.append(dense['target_offsets'])
            target_box_groups.append(dense['target_boxes'])
            target_size_groups.append(dense['target_sizes'])

        if not selector_groups:
            return {'loss_point_selector': outputs['pred_point_selector'].sum() * 0.0}

        selector_logits = torch.cat(selector_groups, dim=0)
        pred_offsets = torch.cat(offset_groups, dim=0)
        pred_boxes = torch.cat(box_groups, dim=0)
        target_has = torch.cat(has_groups, dim=0).to(dtype=selector_logits.dtype, device=selector_logits.device)
        target_offsets = torch.cat(target_offset_groups, dim=0).to(dtype=pred_offsets.dtype, device=pred_offsets.device)
        target_boxes = torch.cat(target_box_groups, dim=0).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
        target_sizes = torch.cat(target_size_groups, dim=0).to(dtype=pred_boxes.dtype, device=pred_boxes.device)

        selector_target = selector_logits.new_zeros(selector_logits.shape)
        visible = target_has > 0.5
        if visible.any():
            scale_xyxy = target_sizes[visible].repeat(1, 2)
            pred_points = absolute_points_from_boxes_and_offsets(
                pred_boxes[visible] * scale_xyxy,
                pred_offsets[visible],
                mode=self.point_selector_offset_mode,
                top_anchor_ratio=self.point_selector_top_anchor_ratio,
            )
            target_points = absolute_points_from_boxes_and_offsets(
                target_boxes[visible] * scale_xyxy,
                target_offsets[visible],
                mode=self.point_selector_offset_mode,
                top_anchor_ratio=self.point_selector_top_anchor_ratio,
            )
            l2_pixel = torch.linalg.norm(pred_points - target_points, ord=2, dim=-1)
            iou_matrix, _ = box_iou(
                box_cxcywh_to_xyxy(pred_boxes[visible].detach()),
                box_cxcywh_to_xyxy(target_boxes[visible].detach()),
            )
            aligned_iou = torch.diag(iou_matrix).clamp(min=0.0, max=1.0)
            if self.point_selector_iou_power != 1.0:
                aligned_iou = aligned_iou.pow(self.point_selector_iou_power)
            selector_target[visible] = torch.exp(-l2_pixel.detach() / self.point_selector_tau).to(
                dtype=selector_target.dtype
            ) * aligned_iou.to(dtype=selector_target.dtype)

        loss = F.binary_cross_entropy_with_logits(selector_logits, selector_target.detach(), reduction='mean')
        return {'loss_point_selector': loss}

    def loss_point_accept(self, outputs, targets, indices, num_boxes, **kwargs):
        """Set-aware listwise candidate selector for visible picking points.

        For each visible GT, competing predicted grape candidates are all
        queries with IoU >= point_accept_iou_thresh.  The loss trains only the
        accept head to rank the lower-error candidate higher; boxes, has logits,
        and offsets are used as detached supervision features.
        """
        required = ('pred_point_accept', 'pred_picking_offsets', 'pred_boxes')
        if any(key not in outputs for key in required):
            return {}

        accept_logits = outputs['pred_point_accept'].squeeze(-1)
        pred_offsets = outputs['pred_picking_offsets'].detach()
        pred_boxes = outputs['pred_boxes'].detach()
        total_loss = accept_logits.sum() * 0.0
        group_count = 0

        for batch_idx, target in enumerate(targets):
            if len(target.get('boxes', [])) == 0:
                continue
            target_has = target.get('has_picking')
            if target_has is None:
                continue
            visible_indices = torch.nonzero(target_has.to(device=accept_logits.device) > 0.5, as_tuple=False).flatten()
            if visible_indices.numel() == 0:
                continue

            tgt_boxes = target['boxes'].to(device=pred_boxes.device, dtype=pred_boxes.dtype)
            tgt_offsets = target['picking_offsets'].to(device=pred_offsets.device, dtype=pred_offsets.dtype)
            target_size = target['orig_size'].to(device=pred_boxes.device, dtype=pred_boxes.dtype).view(1, 2)
            ious, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes[batch_idx]), box_cxcywh_to_xyxy(tgt_boxes))

            for gt_idx in visible_indices.tolist():
                candidate_mask = ious[:, gt_idx] >= self.point_accept_iou_thresh
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                if candidate_indices.numel() == 0:
                    continue
                candidate_iou = ious[candidate_indices, gt_idx].clamp(min=1e-6, max=1.0)
                if candidate_indices.numel() > self.point_accept_topk:
                    _, order = torch.topk(candidate_iou, k=self.point_accept_topk, largest=True)
                    candidate_indices = candidate_indices[order]
                    candidate_iou = candidate_iou[order]

                scale_xyxy = target_size.repeat(candidate_indices.numel(), 2)
                pred_points = absolute_points_from_boxes_and_offsets(
                    pred_boxes[batch_idx, candidate_indices] * scale_xyxy,
                    pred_offsets[batch_idx, candidate_indices],
                    mode=self.point_accept_offset_mode,
                    top_anchor_ratio=self.point_accept_top_anchor_ratio,
                )
                target_point = absolute_points_from_boxes_and_offsets(
                    tgt_boxes[gt_idx].view(1, 4) * target_size.repeat(1, 2),
                    tgt_offsets[gt_idx].view(1, 2),
                    mode=self.point_accept_offset_mode,
                    top_anchor_ratio=self.point_accept_top_anchor_ratio,
                )
                l2_pixel = torch.linalg.norm(pred_points - target_point, ord=2, dim=-1).detach()
                target_logit = -l2_pixel / self.point_accept_tau
                if self.point_accept_iou_power > 0.0:
                    target_logit = target_logit + self.point_accept_iou_power * torch.log(candidate_iou)
                target_prob = F.softmax(target_logit, dim=0).detach()
                pred_log_prob = F.log_softmax(accept_logits[batch_idx, candidate_indices], dim=0)
                total_loss = total_loss - (target_prob * pred_log_prob).sum()
                group_count += 1

        if group_count == 0:
            return {'loss_point_accept': accept_logits.sum() * 0.0}
        return {'loss_point_accept': total_loss / group_count}

    def loss_point_reliability(self, outputs, targets, indices, num_boxes, **kwargs):
        """Detached matched-query reliability calibration for usable points.

        This does not introduce a new coordinate objective.  The target is
        derived from the already decoded, detached point error on Hungarian
        matched instances: visible GT with L2 <= tau is reliable, visible GT
        above tau is unreliable, and invisible GT contributes a low-weight
        negative sample.
        """
        required = ('pred_point_reliability', 'pred_picking_offsets', 'pred_boxes')
        if any(key not in outputs for key in required):
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_point_reliability'][idx].squeeze(-1)
        if src_logits.numel() == 0:
            return {'loss_point_reliability': outputs['pred_point_reliability'].sum() * 0.0}

        src_offsets = outputs['pred_picking_offsets'][idx].detach()
        src_boxes = outputs['pred_boxes'][idx].detach()
        target_has = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_logits.dtype, device=src_logits.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_boxes.dtype, device=src_boxes.device)
        target_sizes = torch.cat(
            [
                t['orig_size'].to(device=src_boxes.device, dtype=src_boxes.dtype).view(1, 2).repeat(len(j), 1)
                for t, (_, j) in zip(targets, indices)
            ],
            dim=0,
        )

        reliability_target = src_logits.new_zeros(src_logits.shape)
        sample_weight = src_logits.new_full(src_logits.shape, self.point_reliability_invisible_weight)
        visible = target_has > 0.5
        if visible.any():
            scale_xyxy = target_sizes[visible].repeat(1, 2)
            pred_points = absolute_points_from_boxes_and_offsets(
                src_boxes[visible] * scale_xyxy,
                src_offsets[visible],
                mode=self.point_reliability_offset_mode,
                top_anchor_ratio=self.point_reliability_top_anchor_ratio,
            )
            target_points = absolute_points_from_boxes_and_offsets(
                target_boxes[visible] * scale_xyxy,
                target_offsets[visible],
                mode=self.point_reliability_offset_mode,
                top_anchor_ratio=self.point_reliability_top_anchor_ratio,
            )
            l2_pixel = torch.linalg.norm(pred_points - target_points, ord=2, dim=-1).detach()
            if self.point_reliability_target == 'continuous_exp':
                reliability_target[visible] = torch.exp(-l2_pixel / self.point_reliability_tau).to(
                    dtype=reliability_target.dtype
                )
            else:
                reliability_target[visible] = (l2_pixel <= self.point_reliability_tau).to(
                    dtype=reliability_target.dtype
                )
            sample_weight[visible] = 1.0

        loss = F.binary_cross_entropy_with_logits(
            src_logits,
            reliability_target.detach(),
            weight=sample_weight.detach(),
            reduction='sum',
        ) / sample_weight.sum().clamp(min=1.0)
        return {'loss_point_reliability': loss}

    def loss_weak_heatmap_score(self, outputs, targets, indices, num_boxes, **kwargs):
        """Weak Gaussian confidence for visible matched picking points.

        This is a query-level proxy for top-ROI heatmap supervision.  The label
        is derived only from the matched bbox and existing visible 2D point, and
        the pixel-distance target is detached so this head calibrates reliability
        instead of acting as another coordinate regression objective.
        """
        required = ('pred_weak_heatmap_score', 'pred_picking_offsets', 'pred_boxes')
        if any(key not in outputs for key in required):
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_heatmap_logits = outputs['pred_weak_heatmap_score'][idx].squeeze(-1)
        src_offsets = outputs['pred_picking_offsets'][idx]
        src_boxes = outputs['pred_boxes'][idx]
        if src_heatmap_logits.numel() == 0:
            return {'loss_weak_heatmap_score': outputs['pred_weak_heatmap_score'].sum() * 0.0}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_heatmap_logits.dtype, device=src_heatmap_logits.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        target_boxes = torch.cat(
            [t['boxes'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_boxes.dtype, device=src_boxes.device)
        target_sizes = torch.cat(
            [
                t['orig_size'].to(device=src_boxes.device, dtype=src_boxes.dtype).view(1, 2).repeat(len(j), 1)
                for t, (_, j) in zip(targets, indices)
            ],
            dim=0,
        )

        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_weak_heatmap_score': outputs['pred_weak_heatmap_score'].sum() * 0.0}

        scale_xyxy = target_sizes[valid].repeat(1, 2)
        pred_points = absolute_points_from_boxes_and_offsets(
            src_boxes[valid] * scale_xyxy,
            src_offsets[valid],
            mode=self.weak_heatmap_offset_mode,
            top_anchor_ratio=self.weak_heatmap_top_anchor_ratio,
        )
        target_points = absolute_points_from_boxes_and_offsets(
            target_boxes[valid] * scale_xyxy,
            target_offsets[valid],
            mode=self.weak_heatmap_offset_mode,
            top_anchor_ratio=self.weak_heatmap_top_anchor_ratio,
        )
        l2_pixel = torch.linalg.norm(pred_points - target_points, ord=2, dim=-1).detach()
        denom = 2.0 * self.weak_heatmap_sigma_px * self.weak_heatmap_sigma_px
        heatmap_target = torch.exp(-(l2_pixel * l2_pixel) / denom).to(dtype=src_heatmap_logits.dtype)
        loss = F.binary_cross_entropy_with_logits(src_heatmap_logits[valid], heatmap_target, reduction='mean')
        return {'loss_weak_heatmap_score': loss}

    def _offset_to_simcc_index(self, coord: torch.Tensor, coord_min: float, coord_max: float, num_bins: int) -> torch.Tensor:
        coord = coord.to(torch.float32).clamp(min=coord_min, max=coord_max)
        scale = float(num_bins - 1) / max(float(coord_max - coord_min), 1e-6)
        return torch.round((coord - float(coord_min)) * scale).to(torch.long).clamp(min=0, max=num_bins - 1)

    def loss_toproi_simcc(self, outputs, targets, indices, num_boxes, **kwargs):
        """Independent x/y SimCC coordinate classification for visible picking points.

        The target is derived from the same bbox-normalized `picking_offsets`
        used by GPPoint-DETR.  This changes the point coordinate expression
        without adding new stem, mask, or 3D labels.
        """
        required = ('pred_toproi_simcc_x', 'pred_toproi_simcc_y')
        if any(key not in outputs for key in required):
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_x = outputs['pred_toproi_simcc_x'][idx]
        src_y = outputs['pred_toproi_simcc_y'][idx]
        if src_x.numel() == 0 or src_y.numel() == 0:
            zero = outputs['pred_toproi_simcc_x'].sum() * 0.0 + outputs['pred_toproi_simcc_y'].sum() * 0.0
            return {'loss_toproi_simcc': zero}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_x.dtype, device=src_x.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_x.dtype, device=src_x.device)

        valid = target_has_picking > 0.5
        if not valid.any():
            zero = outputs['pred_toproi_simcc_x'].sum() * 0.0 + outputs['pred_toproi_simcc_y'].sum() * 0.0
            return {'loss_toproi_simcc': zero}

        target_x = self._offset_to_simcc_index(
            target_offsets[valid, 0],
            self.toproi_simcc_x_min,
            self.toproi_simcc_x_max,
            src_x.shape[-1],
        )
        target_y = self._offset_to_simcc_index(
            target_offsets[valid, 1],
            self.toproi_simcc_y_min,
            self.toproi_simcc_y_max,
            src_y.shape[-1],
        )
        loss_x = F.cross_entropy(
            src_x[valid],
            target_x,
            reduction='mean',
            label_smoothing=self.toproi_simcc_label_smoothing,
        )
        loss_y = F.cross_entropy(
            src_y[valid],
            target_y,
            reduction='mean',
            label_smoothing=self.toproi_simcc_label_smoothing,
        )
        weighted = self.point_coord_weight_x * loss_x + self.point_coord_weight_y * loss_y
        return {'loss_toproi_simcc': weighted / max(self.point_coord_weight_x + self.point_coord_weight_y, 1e-6)}

    def _offset_to_heatmap_target(
        self,
        offsets: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = offsets.to(torch.float32)
        target_x = (offsets[:, 0].clamp(self.toproi_heatmap_x_min, self.toproi_heatmap_x_max) - self.toproi_heatmap_x_min)
        target_x = target_x / max(self.toproi_heatmap_x_max - self.toproi_heatmap_x_min, 1e-6) * float(width - 1)
        target_y = (offsets[:, 1].clamp(self.toproi_heatmap_y_min, self.toproi_heatmap_y_max) - self.toproi_heatmap_y_min)
        target_y = target_y / max(self.toproi_heatmap_y_max - self.toproi_heatmap_y_min, 1e-6) * float(height - 1)
        grid_y = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
        grid_x = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
        dist2 = (grid_x - target_x.view(-1, 1, 1)).pow(2) + (grid_y - target_y.view(-1, 1, 1)).pow(2)
        target = torch.exp(-dist2 / (2.0 * self.toproi_heatmap_sigma * self.toproi_heatmap_sigma))
        target = target / target.flatten(1).sum(dim=1).clamp(min=1e-6).view(-1, 1, 1)
        return target.to(dtype=dtype)

    def loss_toproi_heatmap(self, outputs, targets, indices, num_boxes, **kwargs):
        """2D TopROI heatmap coordinate distribution for visible picking points.

        This keeps local ROI spatial evidence instead of collapsing the ROI to a
        query vector before coordinate prediction. Targets are derived only from
        the existing bbox-normalized visible 2D picking offsets.
        """
        if 'pred_toproi_heatmap' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_heatmap = outputs['pred_toproi_heatmap'][idx]
        if src_heatmap.numel() == 0:
            return {'loss_toproi_heatmap': outputs['pred_toproi_heatmap'].sum() * 0.0}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_heatmap.dtype, device=src_heatmap.device)
        target_offsets = torch.cat(
            [t['picking_offsets'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_heatmap.dtype, device=src_heatmap.device)

        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_toproi_heatmap': outputs['pred_toproi_heatmap'].sum() * 0.0}

        heatmap = src_heatmap[valid]
        height, width = heatmap.shape[-2:]
        target = self._offset_to_heatmap_target(
            target_offsets[valid],
            height,
            width,
            heatmap.dtype,
            heatmap.device,
        )
        log_prob = F.log_softmax(heatmap.flatten(1), dim=-1).reshape_as(heatmap)
        loss = -(target * log_prob).flatten(1).sum(dim=1).mean()
        return {'loss_toproi_heatmap': loss}

    def loss_dense_has_picking(self, outputs, targets, indices, num_boxes, **kwargs):
        # Legacy experimental losses, not used by current baseline_replay /
        # v7_exp2 / small_weight configs.
        if 'pred_has_picking' not in outputs or 'pred_picking_offsets' not in outputs:
            return {}
        dense = self._collect_dense_positive_examples(outputs, targets, indices)
        if dense is None or dense['pred_has_logits'].numel() == 0:
            return {'loss_dense_has_picking': outputs['pred_has_picking'].sum() * 0.0}
        loss = F.binary_cross_entropy_with_logits(
            dense['pred_has_logits'],
            dense['target_has_picking'],
            reduction='mean',
        )
        return {'loss_dense_has_picking': loss}

    def loss_dense_picking_offset(self, outputs, targets, indices, num_boxes, **kwargs):
        # Legacy experimental losses, not used by current baseline_replay /
        # v7_exp2 / small_weight configs.
        if 'pred_picking_offsets' not in outputs:
            return {}
        dense = self._collect_dense_positive_examples(outputs, targets, indices)
        if dense is None or dense['pred_offsets'].numel() == 0:
            return {'loss_dense_picking_offset': outputs['pred_picking_offsets'].sum() * 0.0}
        valid = dense['target_has_picking'] > 0.5
        if not valid.any():
            return {'loss_dense_picking_offset': outputs['pred_picking_offsets'].sum() * 0.0}

        point_loss = self.compute_point_loss(dense['pred_offsets'][valid], dense['target_offsets'][valid])
        coord_weights = torch.as_tensor(
            [self.point_coord_weight_x, self.point_coord_weight_y],
            dtype=point_loss.dtype,
            device=point_loss.device,
        ).view(1, 2)
        point_loss = point_loss * coord_weights
        loss = point_loss.sum() / max(int(valid.sum().item()), 1)
        return {'loss_dense_picking_offset': loss}

    def _geometry_penalty(self, pred_offsets: torch.Tensor) -> torch.Tensor:
        if pred_offsets.numel() == 0:
            return pred_offsets.new_zeros((0,))
        x = pred_offsets[:, 0]
        y = pred_offsets[:, 1]
        horizontal_limit = 0.5 + self.top_margin_ratio
        top_y_min = -0.5 - self.top_margin_ratio
        top_y_max = -0.5 + self.top_region_ratio + self.top_margin_ratio
        excess_x = F.relu(x.abs() - horizontal_limit)
        excess_top = F.relu(top_y_min - y)
        excess_bottom = F.relu(y - top_y_max)
        penalty = torch.stack((excess_x, excess_top, excess_bottom), dim=-1)
        return penalty.pow(2.0).sum(dim=-1)

    def loss_picking_geo(self, outputs, targets, indices, num_boxes, **kwargs):
        # Legacy experimental losses, not used by current baseline_replay /
        # v7_exp2 / small_weight configs.
        if 'pred_picking_offsets' not in outputs:
            return {}

        matched_offsets, matched_has, _, _ = self._gather_matched_point_examples(outputs, targets, indices)
        offset_groups = []
        if matched_offsets.numel() > 0:
            matched_valid = matched_has > 0.5
            if matched_valid.any():
                offset_groups.append(matched_offsets[matched_valid])

        dense = self._collect_dense_positive_examples(outputs, targets, indices)
        if dense is not None and dense['pred_offsets'].numel() > 0:
            dense_valid = dense['target_has_picking'] > 0.5
            if dense_valid.any():
                offset_groups.append(dense['pred_offsets'][dense_valid])

        if not offset_groups:
            return {'loss_picking_geo': outputs['pred_picking_offsets'].sum() * 0.0}

        penalties = [self._geometry_penalty(offsets) for offsets in offset_groups if offsets.numel() > 0]
        if not penalties:
            return {'loss_picking_geo': outputs['pred_picking_offsets'].sum() * 0.0}
        loss = torch.cat(penalties, dim=0).mean()
        return {'loss_picking_geo': loss}

    def loss_picking_locality(self, outputs, targets, indices, num_boxes, **kwargs):
        # Legacy experimental losses, not used by current baseline_replay /
        # v7_exp2 / small_weight configs.
        if 'pred_picking_offsets' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_offsets = outputs['pred_picking_offsets'][idx]
        if src_offsets.numel() == 0:
            return {'loss_picking_locality': outputs['pred_picking_offsets'].sum() * 0.0}

        target_has_picking = torch.cat(
            [t['has_picking'][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).to(dtype=src_offsets.dtype, device=src_offsets.device)
        valid = target_has_picking > 0.5
        if not valid.any():
            return {'loss_picking_locality': outputs['pred_picking_offsets'].sum() * 0.0}

        pred = src_offsets[valid]
        pred_x = pred[:, 0]
        pred_y = pred[:, 1]
        excess_x = F.relu(pred_x.abs() - self.point_locality_x_abs_max)
        excess_top = F.relu(self.point_locality_y_min - pred_y)
        excess_bottom = F.relu(pred_y - self.point_locality_y_max)
        penalty = torch.stack((excess_x, excess_top, excess_bottom), dim=-1)
        if self.point_locality_power != 1.0:
            penalty = penalty.pow(self.point_locality_power)
        loss = penalty.sum(dim=-1).mean()
        return {'loss_picking_locality': loss}

    def compute_point_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.point_loss_type == 'l1':
            return F.l1_loss(pred, target, reduction='none')
        if self.point_loss_type == 'smooth_l1':
            beta = max(self.point_loss_beta, 1e-6)
            return F.smooth_l1_loss(pred, target, reduction='none', beta=beta)
        if self.point_loss_type == 'wing':
            diff = (pred - target).abs()
            omega = max(self.wing_loss_omega, 1e-6)
            epsilon = max(self.wing_loss_epsilon, 1e-6)
            c = omega - omega * math.log1p(omega / epsilon)
            return torch.where(
                diff < omega,
                omega * torch.log1p(diff / epsilon),
                diff - c,
            )
        raise ValueError(f"Unsupported point_loss_type: {self.point_loss_type}")

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                       for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self._dense_positive_cache = None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'has_picking': self.loss_has_picking,
            'has_stem': self.loss_has_stem,
            'picking_offset': self.loss_picking_offset,
            'c2f_coarse': self.loss_c2f_coarse,
            'c2f_fine': self.loss_c2f_fine,
            'dpo': self.loss_dpo,
            'grouped_picking_offset': self.loss_grouped_picking_offset,
            'point_lsd': self.loss_point_lsd,
            'picking_locality': self.loss_picking_locality,
            'point_quality': self.loss_point_quality,
            'point_selector': self.loss_point_selector,
            'point_accept': self.loss_point_accept,
            'point_reliability': self.loss_point_reliability,
            'weak_heatmap_score': self.loss_weak_heatmap_score,
            'toproi_simcc': self.loss_toproi_simcc,
            'toproi_heatmap': self.loss_toproi_heatmap,
            'has_logit_distill': self.loss_has_logit_distill,
            'distill': self.loss_distillation,  # NEW: Add distillation loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        if hasattr(self.matcher, 'set_epoch'):
            self.matcher.set_epoch(kwargs.get('epoch', None))
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, allow_point_cost=False)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets, allow_point_cost=False)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        main_only_losses = {'dpo', 'c2f_coarse', 'c2f_fine', 'point_lsd', 'picking_locality', 'point_selector', 'point_accept', 'point_reliability', 'toproi_heatmap', 'has_logit_distill'}
        for loss_name in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            if loss_name == 'distill':
                l_dict = self.get_loss(loss_name, outputs, targets, None, None, **kwargs)
                if 'loss_distill' in l_dict and l_dict['loss_distill'] != 0:
                    dynamic_weight = self._get_distillation_weight_for_epoch()
                    l_dict['loss_distill'] = l_dict['loss_distill'] * dynamic_weight
                losses.update(l_dict)
            else:
                use_uni_set = self.use_uni_set and (loss_name in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else indices
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss_name, outputs, targets, indices_in)
                l_dict = self.get_loss(loss_name, outputs, targets, indices_in, num_boxes_in, **meta, **kwargs)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss in main_only_losses:
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.point_o2m_aux_enabled:
                    l_dict = self.point_o2m_aux_losses(aux_outputs, targets, cached_indices[i])
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                if loss in main_only_losses:
                    continue
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)
            if self.point_o2m_aux_enabled:
                l_dict = self.point_o2m_aux_losses(aux_outputs, targets, cached_indices[-1])
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    if loss in main_only_losses:
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss in main_only_losses:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    if loss in main_only_losses:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou( \
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes',):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum',
                                         avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
               + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
