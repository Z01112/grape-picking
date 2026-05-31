"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from ..core import register

__all__ = ['DFINETransformer']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int]):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None):
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, \
                                                method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu'):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                picking_head,
                picking_offset_head,
                query_pos_head,
                pre_bbox_head,
                pre_picking_head,
                pre_picking_offset_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        # Legacy interface kept for backward compatibility; current GPPoint-DETR
        # point outputs are computed in outer DFINETransformer.forward via
        # _predict_point_branch. The picking_head arguments and dec_out_picking
        # return slots are intentionally left as legacy placeholders here.
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_picking_logits = []
        dec_out_picking_offsets = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_features = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        pre_output = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0 :
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                pre_output = output
                pre_picking_logits = None
                pre_picking_offsets = None
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_features.append(output)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        stacked_picking_logits = torch.stack(dec_out_picking_logits) if dec_out_picking_logits else None
        stacked_picking_offsets = torch.stack(dec_out_picking_offsets) if dec_out_picking_offsets else None
        stacked_features = torch.stack(dec_out_features) if dec_out_features else None

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), stacked_picking_logits, \
               stacked_picking_offsets, torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), \
               pre_bboxes, pre_scores, pre_picking_logits, pre_picking_offsets, stacked_features, pre_output


@register()
class DFINETransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_picking_point_head=False,
                 has_picking_head_layers=1,
                 picking_offset_head_layers=3,
                 use_point_quality_head=False,
                 point_quality_head_layers=2,
                 point_quality_detach_input=False,
                 use_point_selector_head=False,
                 point_selector_head_layers=2,
                 point_selector_detach_input=True,
                 use_point_accept_head=False,
                 point_accept_head_layers=2,
                 point_accept_detach_input=True,
                 use_weak_point_heatmap_head=False,
                 weak_heatmap_head_layers=2,
                 weak_heatmap_detach_input=True,
                 point_head_hidden_scale=1.0,
                 point_local_feature_fusion=False,
                 point_instance_binding_mode='legacy',
                 point_local_feature_level=0,
                 point_local_pool_size=5,
                 point_local_width_scale=1.0,
                 point_local_height_scale=1.0,
                 point_local_top_shift=0.0,
                 point_full_local_width_scale=1.05,
                 point_full_local_height_scale=1.05,
                 point_full_local_y_shift=0.0,
                 point_top_local_width_scale=1.08,
                 point_top_local_y_min_ratio=-0.10,
                 point_top_local_y_max_ratio=0.40,
                 point_fusion_head_layers=2,
                 point_size_condition=False,
                 point_offset_activation='identity',
                 point_offset_min=-0.25,
                 point_offset_max=1.25,
                 point_teacher_roi_mode='none',
                 point_teacher_roi_apply_to='dn_only',
                 point_teacher_roi_detach=True,
                 point_decoupled_roi=False,
                 point_offset_top_local_width_scale=1.08,
                 point_offset_top_local_y_min_ratio=-0.20,
                 point_offset_top_local_y_max_ratio=0.55,
                 use_toproi_simcc_refiner=False,
                 toproi_simcc_head_layers=2,
                 toproi_simcc_detach_input=True,
                 toproi_simcc_bins_x=64,
                 toproi_simcc_bins_y=64,
                 use_toproi_heatmap_refiner=False,
                 toproi_heatmap_size=12,
                 toproi_heatmap_detach_input=True,
                 toproi_heatmap_hidden_scale=0.50,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale*hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.use_picking_point_head = use_picking_point_head
        self.has_picking_head_layers = max(int(has_picking_head_layers), 1)
        self.picking_offset_head_layers = max(int(picking_offset_head_layers), 1)
        self.use_point_quality_head = bool(use_point_quality_head)
        self.point_quality_head_layers = max(int(point_quality_head_layers), 1)
        self.point_quality_detach_input = bool(point_quality_detach_input)
        self.use_point_selector_head = bool(use_point_selector_head)
        self.point_selector_head_layers = max(int(point_selector_head_layers), 1)
        self.point_selector_detach_input = bool(point_selector_detach_input)
        self.use_point_accept_head = bool(use_point_accept_head)
        self.point_accept_head_layers = max(int(point_accept_head_layers), 1)
        self.point_accept_detach_input = bool(point_accept_detach_input)
        self.use_weak_point_heatmap_head = bool(use_weak_point_heatmap_head)
        self.weak_heatmap_head_layers = max(int(weak_heatmap_head_layers), 1)
        self.weak_heatmap_detach_input = bool(weak_heatmap_detach_input)
        self.point_head_hidden_scale = float(point_head_hidden_scale)
        self.point_local_feature_fusion = bool(point_local_feature_fusion)
        self.point_instance_binding_mode = str(point_instance_binding_mode).strip().lower()
        if self.point_instance_binding_mode not in (
            'legacy',
            'none',
            'full',
            'top',
            'full_top',
            'query_box',
            'query_box_top',
        ):
            raise ValueError(f"Unsupported point_instance_binding_mode: {point_instance_binding_mode}")
        self.point_local_feature_level = max(int(point_local_feature_level), 0)
        self.point_local_pool_size = max(int(point_local_pool_size), 1)
        self.point_local_width_scale = float(point_local_width_scale)
        self.point_local_height_scale = float(point_local_height_scale)
        self.point_local_top_shift = float(point_local_top_shift)
        self.point_full_local_width_scale = float(point_full_local_width_scale)
        self.point_full_local_height_scale = float(point_full_local_height_scale)
        self.point_full_local_y_shift = float(point_full_local_y_shift)
        self.point_top_local_width_scale = float(point_top_local_width_scale)
        self.point_top_local_y_min_ratio = float(point_top_local_y_min_ratio)
        self.point_top_local_y_max_ratio = float(point_top_local_y_max_ratio)
        self.point_fusion_head_layers = max(int(point_fusion_head_layers), 1)
        self.point_size_condition = bool(point_size_condition)
        self.point_offset_activation = str(point_offset_activation).strip().lower()
        if self.point_offset_activation not in ('identity', 'sigmoid_range'):
            raise ValueError(f"Unsupported point_offset_activation: {point_offset_activation}")
        self.point_offset_min = float(point_offset_min)
        self.point_offset_max = float(point_offset_max)
        if self.point_offset_max <= self.point_offset_min:
            self.point_offset_max = self.point_offset_min + 1.0
        self.point_teacher_roi_mode = str(point_teacher_roi_mode).strip().lower()
        if self.point_teacher_roi_mode not in ('none', 'dn_jitter'):
            raise ValueError(f"Unsupported point_teacher_roi_mode: {point_teacher_roi_mode}")
        self.point_teacher_roi_apply_to = str(point_teacher_roi_apply_to).strip().lower()
        if self.point_teacher_roi_apply_to not in ('dn_only',):
            raise ValueError(f"Unsupported point_teacher_roi_apply_to: {point_teacher_roi_apply_to}")
        self.point_teacher_roi_detach = bool(point_teacher_roi_detach)
        self.point_decoupled_roi = bool(point_decoupled_roi)
        self.point_offset_top_local_width_scale = float(point_offset_top_local_width_scale)
        self.point_offset_top_local_y_min_ratio = float(point_offset_top_local_y_min_ratio)
        self.point_offset_top_local_y_max_ratio = float(point_offset_top_local_y_max_ratio)
        self.use_toproi_simcc_refiner = bool(use_toproi_simcc_refiner)
        self.toproi_simcc_head_layers = max(int(toproi_simcc_head_layers), 1)
        self.toproi_simcc_detach_input = bool(toproi_simcc_detach_input)
        self.toproi_simcc_bins_x = max(int(toproi_simcc_bins_x), 2)
        self.toproi_simcc_bins_y = max(int(toproi_simcc_bins_y), 2)
        self.use_toproi_heatmap_refiner = bool(use_toproi_heatmap_refiner)
        self.toproi_heatmap_size = max(int(toproi_heatmap_size), 2)
        self.toproi_heatmap_detach_input = bool(toproi_heatmap_detach_input)
        self.toproi_heatmap_hidden_scale = max(float(toproi_heatmap_hidden_scale), 0.125)
        # GPPoint-DETR keeps the detector queries intact and changes only how
        # the per-query point feature is enriched before has/offset prediction.
        self.point_legacy_local_feature = self.point_instance_binding_mode == 'legacy' and self.point_local_feature_fusion
        self.point_use_full_local_feature = self.point_instance_binding_mode in ('full', 'full_top')
        self.point_use_top_local_feature = self.point_instance_binding_mode in ('top', 'full_top', 'query_box_top')
        self.point_use_query_box_binding = self.point_instance_binding_mode in ('query_box', 'query_box_top')

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)
      # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.dec_picking_head = None
        self.dec_picking_offset_head = None
        self.dec_point_quality_head = None
        self.dec_point_selector_head = None
        self.dec_point_accept_head = None
        self.dec_weak_heatmap_head = None
        self.dec_toproi_simcc_x_head = None
        self.dec_toproi_simcc_y_head = None
        self.dec_toproi_heatmap_head = None
        self.pre_picking_head = None
        self.pre_picking_offset_head = None
        self.pre_point_quality_head = None
        self.pre_point_selector_head = None
        self.pre_point_accept_head = None
        self.pre_weak_heatmap_head = None
        self.pre_toproi_simcc_x_head = None
        self.pre_toproi_simcc_y_head = None
        self.pre_toproi_heatmap_head = None
        self.dec_point_local_proj = None
        self.pre_point_local_proj = None
        self.dec_point_full_local_proj = None
        self.pre_point_full_local_proj = None
        self.dec_point_top_local_proj = None
        self.pre_point_top_local_proj = None
        self.dec_point_size_proj = None
        self.pre_point_size_proj = None
        self.dec_point_query_pos_proj = None
        self.pre_point_query_pos_proj = None
        self.dec_point_box_geom_proj = None
        self.pre_point_box_geom_proj = None
        self.dec_point_fusion_head = None
        self.pre_point_fusion_head = None
        if self.use_picking_point_head:
            # Per-grape-query picking heads: has_picking classifies whether the
            # matched grape has a visible picking point, while point_offset
            # regresses a bbox-normalized displacement from the top-center anchor.
            point_hidden_dim = max(int(round(hidden_dim * self.point_head_hidden_scale)), 1)
            point_hidden_dim_scaled = max(int(round(scaled_dim * self.point_head_hidden_scale)), 1)
            self.dec_picking_head = nn.ModuleList(
                [
                    self._make_point_head(hidden_dim, point_hidden_dim, 1, self.has_picking_head_layers, mlp_act)
                    for _ in range(self.eval_idx + 1)
                ]
              + [
                    self._make_point_head(scaled_dim, point_hidden_dim_scaled, 1, self.has_picking_head_layers, mlp_act)
                    for _ in range(num_layers - self.eval_idx - 1)
                ]
            )
            self.dec_picking_offset_head = nn.ModuleList(
                [
                    self._make_point_head(hidden_dim, point_hidden_dim, 2, self.picking_offset_head_layers, mlp_act)
                    for _ in range(self.eval_idx + 1)
                ]
              + [
                    self._make_point_head(scaled_dim, point_hidden_dim_scaled, 2, self.picking_offset_head_layers, mlp_act)
                    for _ in range(num_layers - self.eval_idx - 1)
                ]
            )
            self.pre_picking_head = self._make_point_head(
                hidden_dim,
                point_hidden_dim,
                1,
                self.has_picking_head_layers,
                mlp_act,
            )
            self.pre_picking_offset_head = self._make_point_head(
                hidden_dim,
                point_hidden_dim,
                2,
                self.picking_offset_head_layers,
                mlp_act,
            )
            if self.use_point_quality_head:
                self.dec_point_quality_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, 1, self.point_quality_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, 1, self.point_quality_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_point_quality_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    1,
                    self.point_quality_head_layers,
                    mlp_act,
                )
            if self.use_point_selector_head:
                self.dec_point_selector_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, 1, self.point_selector_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, 1, self.point_selector_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_point_selector_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    1,
                    self.point_selector_head_layers,
                    mlp_act,
                )
            if self.use_point_accept_head:
                accept_geom_dim = 8  # box cxcywh + class score + has logit + predicted offset xy
                self.dec_point_accept_head = nn.ModuleList(
                    [
                        self._make_point_head(
                            hidden_dim * 2 + accept_geom_dim,
                            point_hidden_dim,
                            1,
                            self.point_accept_head_layers,
                            mlp_act,
                        )
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(
                            scaled_dim * 2 + accept_geom_dim,
                            point_hidden_dim_scaled,
                            1,
                            self.point_accept_head_layers,
                            mlp_act,
                        )
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_point_accept_head = self._make_point_head(
                    hidden_dim * 2 + accept_geom_dim,
                    point_hidden_dim,
                    1,
                    self.point_accept_head_layers,
                    mlp_act,
                )
            if self.use_weak_point_heatmap_head:
                self.dec_weak_heatmap_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, 1, self.weak_heatmap_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, 1, self.weak_heatmap_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_weak_heatmap_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    1,
                    self.weak_heatmap_head_layers,
                    mlp_act,
                )
            if self.use_toproi_simcc_refiner:
                # SimCC predicts independent x/y coordinate distributions for
                # the top-ROI picking point.  By default it is detached from the
                # shared query/ROI feature so the refiner can improve point
                # expression without pulling detection or has_picking features.
                self.dec_toproi_simcc_x_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, self.toproi_simcc_bins_x, self.toproi_simcc_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, self.toproi_simcc_bins_x, self.toproi_simcc_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.dec_toproi_simcc_y_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, self.toproi_simcc_bins_y, self.toproi_simcc_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, self.toproi_simcc_bins_y, self.toproi_simcc_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_toproi_simcc_x_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    self.toproi_simcc_bins_x,
                    self.toproi_simcc_head_layers,
                    mlp_act,
                )
                self.pre_toproi_simcc_y_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    self.toproi_simcc_bins_y,
                    self.toproi_simcc_head_layers,
                    mlp_act,
                )
            if self.use_toproi_heatmap_refiner:
                # Unlike query-vector SimCC, this head preserves the TopROI
                # spatial feature map and predicts a 2D probability surface.
                heatmap_hidden_dim = max(int(round(hidden_dim * self.toproi_heatmap_hidden_scale)), 1)
                self.dec_toproi_heatmap_head = nn.ModuleList(
                    [self._make_heatmap_head(hidden_dim, heatmap_hidden_dim, mlp_act) for _ in range(num_layers)]
                )
                self.pre_toproi_heatmap_head = self._make_heatmap_head(hidden_dim, heatmap_hidden_dim, mlp_act)
            if self.point_legacy_local_feature:
                self.pre_point_local_proj = nn.Linear(hidden_dim, hidden_dim)
                self.dec_point_local_proj = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.eval_idx + 1)]
                  + [nn.Linear(hidden_dim, scaled_dim) for _ in range(num_layers - self.eval_idx - 1)]
                )
            if self.point_use_full_local_feature:
                self.pre_point_full_local_proj = nn.Linear(hidden_dim, hidden_dim)
                self.dec_point_full_local_proj = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.eval_idx + 1)]
                  + [nn.Linear(hidden_dim, scaled_dim) for _ in range(num_layers - self.eval_idx - 1)]
                )
            if self.point_use_top_local_feature:
                self.pre_point_top_local_proj = nn.Linear(hidden_dim, hidden_dim)
                self.dec_point_top_local_proj = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.eval_idx + 1)]
                  + [nn.Linear(hidden_dim, scaled_dim) for _ in range(num_layers - self.eval_idx - 1)]
                )
            if self.point_size_condition:
                self.pre_point_size_proj = self._make_point_head(4, point_hidden_dim, hidden_dim, 2, mlp_act)
                self.dec_point_size_proj = nn.ModuleList(
                    [self._make_point_head(4, point_hidden_dim, hidden_dim, 2, mlp_act) for _ in range(self.eval_idx + 1)]
                  + [self._make_point_head(4, point_hidden_dim_scaled, scaled_dim, 2, mlp_act) for _ in range(num_layers - self.eval_idx - 1)]
                )
            if self.point_use_query_box_binding:
                self.pre_point_query_pos_proj = nn.Linear(hidden_dim, hidden_dim)
                self.dec_point_query_pos_proj = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.eval_idx + 1)]
                  + [nn.Linear(hidden_dim, scaled_dim) for _ in range(num_layers - self.eval_idx - 1)]
                )
                self.pre_point_box_geom_proj = self._make_point_head(4, point_hidden_dim, hidden_dim, 2, mlp_act)
                self.dec_point_box_geom_proj = nn.ModuleList(
                    [self._make_point_head(4, point_hidden_dim, hidden_dim, 2, mlp_act) for _ in range(self.eval_idx + 1)]
                  + [self._make_point_head(4, point_hidden_dim_scaled, scaled_dim, 2, mlp_act) for _ in range(num_layers - self.eval_idx - 1)]
                )
            point_feature_parts = (
                1
                + int(self.point_legacy_local_feature)
                + int(self.point_use_full_local_feature)
                + int(self.point_use_top_local_feature)
                + int(self.point_size_condition)
                + 2 * int(self.point_use_query_box_binding)
            )
            if point_feature_parts > 1:
                pre_parts = point_feature_parts
                self.pre_point_fusion_head = self._make_point_head(
                    pre_parts * hidden_dim,
                    point_hidden_dim,
                    hidden_dim,
                    self.point_fusion_head_layers,
                    mlp_act,
                )
                self.dec_point_fusion_head = nn.ModuleList(
                    [
                        self._make_point_head(
                            pre_parts * hidden_dim,
                            point_hidden_dim,
                            hidden_dim,
                            self.point_fusion_head_layers,
                            mlp_act,
                        )
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(
                            pre_parts * scaled_dim,
                            point_hidden_dim_scaled,
                            scaled_dim,
                            self.point_fusion_head_layers,
                            mlp_act,
                        )
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()


        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        if self.dec_picking_head is not None:
            self.dec_picking_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_picking_head[self.eval_idx]])
        if self.dec_picking_offset_head is not None:
            self.dec_picking_offset_head = nn.ModuleList(
                [
                    self.dec_picking_offset_head[i] if i <= self.eval_idx else nn.Identity()
                    for i in range(len(self.dec_picking_offset_head))
                ]
            )
        if self.dec_point_quality_head is not None:
            self.dec_point_quality_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_point_quality_head[self.eval_idx]]
            )
        if self.dec_point_selector_head is not None:
            self.dec_point_selector_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_point_selector_head[self.eval_idx]]
            )
        if self.dec_point_accept_head is not None:
            self.dec_point_accept_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_point_accept_head[self.eval_idx]]
            )
        if self.dec_weak_heatmap_head is not None:
            self.dec_weak_heatmap_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_weak_heatmap_head[self.eval_idx]]
            )
        if self.dec_toproi_simcc_x_head is not None:
            self.dec_toproi_simcc_x_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_toproi_simcc_x_head[self.eval_idx]]
            )
        if self.dec_toproi_simcc_y_head is not None:
            self.dec_toproi_simcc_y_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_toproi_simcc_y_head[self.eval_idx]]
            )
        if self.dec_toproi_heatmap_head is not None:
            self.dec_toproi_heatmap_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_toproi_heatmap_head[self.eval_idx]]
            )

    @staticmethod
    def _make_point_head(input_dim, hidden_dim, output_dim, num_layers, act):
        if int(num_layers) <= 1:
            return nn.Linear(input_dim, output_dim)
        return MLP(input_dim, hidden_dim, output_dim, int(num_layers), act=act)

    @staticmethod
    def _make_heatmap_head(input_dim, hidden_dim, act):
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            get_activation(act),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def _activate_point_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if self.point_offset_activation == 'identity':
            return offsets
        span = self.point_offset_max - self.point_offset_min
        return self.point_offset_min + span * torch.sigmoid(offsets)

    @staticmethod
    def _init_cls_head_bias(head, bias):
        if isinstance(head, nn.Identity):
            return
        if hasattr(head, 'bias') and head.bias is not None:
            init.constant_(head.bias, bias)
            return
        if hasattr(head, 'layers') and len(head.layers) > 0:
            init.constant_(head.layers[-1].bias, bias)

    @staticmethod
    def _init_reg_head_zero(head):
        if isinstance(head, nn.Identity):
            return
        if hasattr(head, 'layers') and len(head.layers) > 0:
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)
            return
        if hasattr(head, 'weight') and hasattr(head, 'bias'):
            init.constant_(head.weight, 0)
            init.constant_(head.bias, 0)

    @staticmethod
    def _box_geometry_features(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        boxes = boxes_cxcywh.to(torch.float32)
        widths = boxes[..., 2].clamp(min=1e-6)
        heights = boxes[..., 3].clamp(min=1e-6)
        areas = widths * heights
        aspect = widths / heights
        return torch.stack(
            (
                widths,
                heights,
                torch.log(areas),
                torch.log(aspect),
            ),
            dim=-1,
        )

    def _build_top_local_rois(self, boxes_cxcywh: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        boxes = boxes_cxcywh.to(torch.float32)
        cx, cy, w, h = boxes.unbind(-1)
        roi_w = (w * self.point_local_width_scale).clamp(min=1e-6)
        roi_h = (h * self.point_local_height_scale).clamp(min=1e-6)
        roi_cy = cy - self.point_local_top_shift * h
        x1 = (cx - 0.5 * roi_w) * feat_w
        y1 = (roi_cy - 0.5 * roi_h) * feat_h
        x2 = (cx + 0.5 * roi_w) * feat_w
        y2 = (roi_cy + 0.5 * roi_h) * feat_h
        x1 = x1.clamp(min=0.0, max=max(float(feat_w - 1), 0.0))
        y1 = y1.clamp(min=0.0, max=max(float(feat_h - 1), 0.0))
        x2 = torch.maximum(x2.clamp(min=0.0, max=float(feat_w)), x1 + 1e-3)
        y2 = torch.maximum(y2.clamp(min=0.0, max=float(feat_h)), y1 + 1e-3)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _build_full_local_rois(self, boxes_cxcywh: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        boxes = boxes_cxcywh.to(torch.float32)
        cx, cy, w, h = boxes.unbind(-1)
        roi_w = (w * self.point_full_local_width_scale).clamp(min=1e-6)
        roi_h = (h * self.point_full_local_height_scale).clamp(min=1e-6)
        roi_cy = cy + self.point_full_local_y_shift * h
        x1 = (cx - 0.5 * roi_w) * feat_w
        y1 = (roi_cy - 0.5 * roi_h) * feat_h
        x2 = (cx + 0.5 * roi_w) * feat_w
        y2 = (roi_cy + 0.5 * roi_h) * feat_h
        x1 = x1.clamp(min=0.0, max=max(float(feat_w - 1), 0.0))
        y1 = y1.clamp(min=0.0, max=max(float(feat_h - 1), 0.0))
        x2 = torch.maximum(x2.clamp(min=0.0, max=float(feat_w)), x1 + 1e-3)
        y2 = torch.maximum(y2.clamp(min=0.0, max=float(feat_h)), y1 + 1e-3)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _build_point_top_rois(
        self,
        boxes_cxcywh: torch.Tensor,
        feat_h: int,
        feat_w: int,
        width_scale: float = None,
        y_min_ratio: float = None,
        y_max_ratio: float = None,
    ) -> torch.Tensor:
        """Build the Top Local ROI used by the GPPoint-DETR point branch.

        The input boxes are normalized cxcywh query boxes. The ROI is centered
        horizontally on the query box but vertically expressed relative to the
        top edge, matching the paper's upper-peduncle local cue.
        This is a fixed geometric-prior window, not an adaptive search module.
        """
        boxes = boxes_cxcywh.to(torch.float32)
        cx, cy, w, h = boxes.unbind(-1)
        width_scale = self.point_top_local_width_scale if width_scale is None else float(width_scale)
        y_min_ratio = self.point_top_local_y_min_ratio if y_min_ratio is None else float(y_min_ratio)
        y_max_ratio = self.point_top_local_y_max_ratio if y_max_ratio is None else float(y_max_ratio)
        roi_w = (w * width_scale).clamp(min=1e-6)
        top_y = cy - 0.5 * h
        y1 = (top_y + y_min_ratio * h) * feat_h
        y2 = (top_y + y_max_ratio * h) * feat_h
        x1 = (cx - 0.5 * roi_w) * feat_w
        x2 = (cx + 0.5 * roi_w) * feat_w
        x1 = x1.clamp(min=0.0, max=max(float(feat_w - 1), 0.0))
        y1 = y1.clamp(min=0.0, max=max(float(feat_h - 1), 0.0))
        x2 = torch.maximum(x2.clamp(min=0.0, max=float(feat_w)), x1 + 1e-3)
        y2 = torch.maximum(y2.clamp(min=0.0, max=float(feat_h)), y1 + 1e-3)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _pool_point_local_features(
        self,
        proj_feats: List[torch.Tensor],
        boxes_cxcywh: torch.Tensor,
        roi_type: str = 'legacy',
        top_roi_params: tuple = None,
    ) -> torch.Tensor:
        level_idx = min(self.point_local_feature_level, len(proj_feats) - 1)
        feat = proj_feats[level_idx]
        bs, _, feat_h, feat_w = feat.shape
        _, query_count, _ = boxes_cxcywh.shape
        if query_count == 0:
            return feat.new_zeros((bs, 0, feat.shape[1]))

        roi_type = str(roi_type).strip().lower()
        if roi_type == 'legacy':
            rois_xyxy = self._build_top_local_rois(boxes_cxcywh, feat_h, feat_w)
        elif roi_type == 'full':
            rois_xyxy = self._build_full_local_rois(boxes_cxcywh, feat_h, feat_w)
        elif roi_type == 'top':
            if top_roi_params is None:
                rois_xyxy = self._build_point_top_rois(boxes_cxcywh, feat_h, feat_w)
            else:
                rois_xyxy = self._build_point_top_rois(boxes_cxcywh, feat_h, feat_w, *top_roi_params)
        else:
            raise ValueError(f"Unsupported roi_type: {roi_type}")
        batch_ids = torch.arange(bs, device=feat.device, dtype=rois_xyxy.dtype).view(bs, 1, 1).expand(bs, query_count, 1)
        rois = torch.cat((batch_ids, rois_xyxy), dim=-1).reshape(-1, 5)
        pooled = torchvision.ops.roi_align(
            feat,
            rois,
            output_size=(self.point_local_pool_size, self.point_local_pool_size),
            spatial_scale=1.0,
            aligned=True,
        )
        pooled = pooled.mean(dim=(-1, -2))
        return pooled.reshape(bs, query_count, feat.shape[1])

    def _pool_point_local_maps(
        self,
        proj_feats: List[torch.Tensor],
        boxes_cxcywh: torch.Tensor,
        roi_type: str = 'top',
        output_size: int = None,
        top_roi_params: tuple = None,
    ) -> torch.Tensor:
        level_idx = min(self.point_local_feature_level, len(proj_feats) - 1)
        feat = proj_feats[level_idx]
        bs, _, feat_h, feat_w = feat.shape
        _, query_count, _ = boxes_cxcywh.shape
        size = self.toproi_heatmap_size if output_size is None else max(int(output_size), 2)
        if query_count == 0:
            return feat.new_zeros((bs, 0, feat.shape[1], size, size))

        roi_type = str(roi_type).strip().lower()
        if roi_type == 'legacy':
            rois_xyxy = self._build_top_local_rois(boxes_cxcywh, feat_h, feat_w)
        elif roi_type == 'full':
            rois_xyxy = self._build_full_local_rois(boxes_cxcywh, feat_h, feat_w)
        elif roi_type == 'top':
            if top_roi_params is None:
                rois_xyxy = self._build_point_top_rois(boxes_cxcywh, feat_h, feat_w)
            else:
                rois_xyxy = self._build_point_top_rois(boxes_cxcywh, feat_h, feat_w, *top_roi_params)
        else:
            raise ValueError(f"Unsupported roi_type: {roi_type}")

        batch_ids = torch.arange(bs, device=feat.device, dtype=rois_xyxy.dtype).view(bs, 1, 1).expand(bs, query_count, 1)
        rois = torch.cat((batch_ids, rois_xyxy), dim=-1).reshape(-1, 5)
        pooled = torchvision.ops.roi_align(
            feat,
            rois,
            output_size=(size, size),
            spatial_scale=1.0,
            aligned=True,
        )
        return pooled.reshape(bs, query_count, feat.shape[1], size, size)

    def _predict_toproi_heatmap(
        self,
        proj_feats: List[torch.Tensor],
        boxes_cxcywh: torch.Tensor,
        heatmap_head,
        roi_boxes_cxcywh: torch.Tensor = None,
        top_roi_params: tuple = None,
    ) -> torch.Tensor:
        if heatmap_head is None:
            return None
        local_boxes_cxcywh = boxes_cxcywh if roi_boxes_cxcywh is None else roi_boxes_cxcywh
        maps = self._pool_point_local_maps(
            proj_feats,
            local_boxes_cxcywh,
            roi_type='top',
            output_size=self.toproi_heatmap_size,
            top_roi_params=top_roi_params,
        )
        bs, query_count, channels, height, width = maps.shape
        if query_count == 0:
            return maps.new_zeros((bs, 0, height, width))
        heatmap_input = maps.detach() if self.toproi_heatmap_detach_input else maps
        logits = heatmap_head(heatmap_input.reshape(bs * query_count, channels, height, width))
        return logits.reshape(bs, query_count, height, width)

    def _fuse_point_feature(
        self,
        hidden: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        proj_feats: List[torch.Tensor],
        legacy_local_proj_head,
        full_local_proj_head,
        top_local_proj_head,
        size_proj_head,
        query_pos_proj_head,
        box_geom_proj_head,
        fusion_head,
        roi_boxes_cxcywh: torch.Tensor = None,
        top_roi_params: tuple = None,
    ) -> torch.Tensor:
        """Fuse query, box geometry, and optional local cues for point heads.

        In query_box_top mode this combines decoder hidden state, top-local ROI
        evidence, query positional embedding, and predicted-box geometry. The
        fused feature is still tied to the same object query, not a new ROI
        detector proposal.
        Conceptually these are query semantic, local visual cue, and geometric
        prior signals.
        """
        parts = [hidden]
        local_boxes_cxcywh = boxes_cxcywh if roi_boxes_cxcywh is None else roi_boxes_cxcywh
        if legacy_local_proj_head is not None:
            legacy_local_feat = self._pool_point_local_features(proj_feats, local_boxes_cxcywh, roi_type='legacy')
            parts.append(legacy_local_proj_head(legacy_local_feat))
        if full_local_proj_head is not None:
            full_local_feat = self._pool_point_local_features(proj_feats, local_boxes_cxcywh, roi_type='full')
            parts.append(full_local_proj_head(full_local_feat))
        if top_local_proj_head is not None:
            top_local_feat = self._pool_point_local_features(
                proj_feats,
                local_boxes_cxcywh,
                roi_type='top',
                top_roi_params=top_roi_params,
            )
            parts.append(top_local_proj_head(top_local_feat))
        if size_proj_head is not None:
            parts.append(size_proj_head(self._box_geometry_features(boxes_cxcywh)))
        if query_pos_proj_head is not None:
            parts.append(query_pos_proj_head(self.query_pos_head(boxes_cxcywh).to(hidden.dtype)))
        if box_geom_proj_head is not None:
            parts.append(box_geom_proj_head(self._box_geometry_features(boxes_cxcywh)))
        if fusion_head is None or len(parts) == 1:
            return hidden
        fused_delta = fusion_head(torch.cat(parts, dim=-1))
        return hidden + fused_delta

    def _predict_point_branch(
        self,
        hidden: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        det_logits: torch.Tensor,
        proj_feats: List[torch.Tensor],
        cls_head,
        offset_head,
        quality_head,
        selector_head,
        accept_head,
        weak_heatmap_head,
        simcc_x_head,
        simcc_y_head,
        toproi_heatmap_head,
        legacy_local_proj_head,
        full_local_proj_head,
        top_local_proj_head,
        size_proj_head,
        query_pos_proj_head,
        box_geom_proj_head,
        fusion_head,
        roi_boxes_cxcywh: torch.Tensor = None,
    ):
        """Predict per-query has_picking logits, point offsets, and optional point reliability."""
        cls_feature = self._fuse_point_feature(
            hidden,
            boxes_cxcywh,
            proj_feats,
            legacy_local_proj_head,
            full_local_proj_head,
            top_local_proj_head,
            size_proj_head,
            query_pos_proj_head,
            box_geom_proj_head,
            fusion_head,
            roi_boxes_cxcywh,
        )
        if self.point_decoupled_roi and top_local_proj_head is not None:
            # v7_exp2_decoupled_roi keeps has_picking on the current narrow Top
            # ROI while giving point_offset a taller Top ROI.  This changes
            # only local visual cue pooling; query semantics, box geometry,
            # losses, and postprocess decoding stay unchanged.
            offset_feature = self._fuse_point_feature(
                hidden,
                boxes_cxcywh,
                proj_feats,
                legacy_local_proj_head,
                full_local_proj_head,
                top_local_proj_head,
                size_proj_head,
                query_pos_proj_head,
                box_geom_proj_head,
                fusion_head,
                roi_boxes_cxcywh,
                (
                    self.point_offset_top_local_width_scale,
                    self.point_offset_top_local_y_min_ratio,
                    self.point_offset_top_local_y_max_ratio,
                ),
            )
        else:
            offset_feature = cls_feature
        has_logits = cls_head(cls_feature)
        picking_offsets = self._activate_point_offsets(offset_head(offset_feature))
        if quality_head is not None:
            # v7_exp2_point_quality_sg uses the same quality target as the
            # original point_quality ablation, but stops quality-loss gradients
            # from flowing back into the shared query/ROI/point features.
            quality_feature = offset_feature.detach() if self.point_quality_detach_input else offset_feature
            quality_logits = quality_head(quality_feature)
        else:
            quality_logits = None
        if selector_head is not None:
            # The selector is trained as a detached candidate-ranking head.  It
            # must not reshape detector queries or point coordinates during the
            # short probe, otherwise it collapses back into another point loss.
            selector_feature = offset_feature.detach() if self.point_selector_detach_input else offset_feature
            selector_logits = selector_head(selector_feature)
        else:
            selector_logits = None
        if accept_head is not None:
            # Set-aware accept scoring sees each candidate together with a
            # detached global candidate-set context and detached geometry.  The
            # listwise accept loss therefore trains only this ranking head.
            accept_base = offset_feature.detach() if self.point_accept_detach_input else offset_feature
            accept_context = accept_base.mean(dim=1, keepdim=True).expand_as(accept_base)
            if det_logits is None:
                det_score = boxes_cxcywh.new_zeros((*boxes_cxcywh.shape[:2], 1))
            else:
                det_score = det_logits.detach().sigmoid().amax(dim=-1, keepdim=True).to(dtype=accept_base.dtype)
            accept_geom = torch.cat(
                (
                    boxes_cxcywh.detach(),
                    det_score,
                    has_logits.detach(),
                    picking_offsets.detach(),
                ),
                dim=-1,
            ).to(dtype=accept_base.dtype)
            accept_logits = accept_head(torch.cat((accept_base, accept_context, accept_geom), dim=-1))
        else:
            accept_logits = None
        if weak_heatmap_head is not None:
            # Query-level weak Gaussian score is supervised from existing bbox +
            # visible 2D point labels.  Detach by default so it calibrates point
            # reliability without changing detector/offset features.
            heatmap_feature = offset_feature.detach() if self.weak_heatmap_detach_input else offset_feature
            weak_heatmap_logits = weak_heatmap_head(heatmap_feature)
        else:
            weak_heatmap_logits = None
        if simcc_x_head is not None and simcc_y_head is not None:
            simcc_feature = offset_feature.detach() if self.toproi_simcc_detach_input else offset_feature
            simcc_x_logits = simcc_x_head(simcc_feature)
            simcc_y_logits = simcc_y_head(simcc_feature)
        else:
            simcc_x_logits = None
            simcc_y_logits = None
        heatmap_logits = self._predict_toproi_heatmap(
            proj_feats,
            boxes_cxcywh,
            toproi_heatmap_head,
            roi_boxes_cxcywh,
            (
                self.point_offset_top_local_width_scale,
                self.point_offset_top_local_y_min_ratio,
                self.point_offset_top_local_y_max_ratio,
            ) if self.point_decoupled_roi else None,
        )
        return (
            has_logits,
            picking_offsets,
            quality_logits,
            selector_logits,
            accept_logits,
            weak_heatmap_logits,
            simcc_x_logits,
            simcc_y_logits,
            heatmap_logits,
        )

    def _get_dn_teacher_roi_boxes(self, denoising_bbox_unact: torch.Tensor, dn_meta: dict) -> torch.Tensor:
        if (
            self.point_teacher_roi_mode != 'dn_jitter'
            or self.point_teacher_roi_apply_to != 'dn_only'
            or denoising_bbox_unact is None
            or dn_meta is None
        ):
            return None
        # DN inputs are target boxes with denoising jitter already applied.  Use
        # them only as the local-ROI pooling box; decoder box geometry and losses
        # keep their original GPPoint-DETR behavior.
        boxes = denoising_bbox_unact.sigmoid()
        return boxes.detach() if self.point_teacher_roi_detach else boxes

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
        if self.pre_picking_head is not None:
            self._init_cls_head_bias(self.pre_picking_head, bias)
        if self.pre_picking_offset_head is not None:
            self._init_reg_head_zero(self.pre_picking_offset_head)
        if self.pre_point_quality_head is not None:
            self._init_reg_head_zero(self.pre_point_quality_head)
        if self.pre_point_selector_head is not None:
            self._init_reg_head_zero(self.pre_point_selector_head)
        if self.pre_point_accept_head is not None:
            self._init_reg_head_zero(self.pre_point_accept_head)
        if self.pre_weak_heatmap_head is not None:
            self._init_reg_head_zero(self.pre_weak_heatmap_head)
        if self.pre_toproi_simcc_x_head is not None:
            self._init_reg_head_zero(self.pre_toproi_simcc_x_head)
        if self.pre_toproi_simcc_y_head is not None:
            self._init_reg_head_zero(self.pre_toproi_simcc_y_head)
        if self.pre_point_fusion_head is not None:
            self._init_reg_head_zero(self.pre_point_fusion_head)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)
        if self.dec_picking_head is not None:
            for head in self.dec_picking_head:
                self._init_cls_head_bias(head, bias)
        if self.dec_picking_offset_head is not None:
            for head in self.dec_picking_offset_head:
                self._init_reg_head_zero(head)
        if self.dec_point_quality_head is not None:
            for head in self.dec_point_quality_head:
                self._init_reg_head_zero(head)
        if self.dec_point_selector_head is not None:
            for head in self.dec_point_selector_head:
                self._init_reg_head_zero(head)
        if self.dec_point_accept_head is not None:
            for head in self.dec_point_accept_head:
                self._init_reg_head_zero(head)
        if self.dec_weak_heatmap_head is not None:
            for head in self.dec_weak_heatmap_head:
                self._init_reg_head_zero(head)
        if self.dec_toproi_simcc_x_head is not None:
            for head in self.dec_toproi_simcc_x_head:
                self._init_reg_head_zero(head)
        if self.dec_toproi_simcc_y_head is not None:
            for head in self.dec_toproi_simcc_y_head:
                self._init_reg_head_zero(head)
        if self.dec_point_fusion_head is not None:
            for head in self.dec_point_fusion_head:
                self._init_reg_head_zero(head)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes, proj_feats

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)

        enc_topk_bbox_unact :torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None

        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes, proj_feats = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        dn_teacher_roi_boxes = self._get_dn_teacher_roi_boxes(denoising_bbox_unact, dn_meta)

        # decoder
        out_bboxes, out_logits, out_picking_logits, out_picking_offsets, out_corners, out_refs, \
            pre_bboxes, pre_logits, pre_picking_logits, pre_picking_offsets, out_hidden, pre_hidden = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_picking_head,
            self.dec_picking_offset_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.pre_picking_head,
            self.pre_picking_offset_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta)

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            if pre_hidden is not None:
                dn_pre_hidden, pre_hidden = torch.split(pre_hidden, dn_meta['dn_num_split'], dim=1)
            else:
                dn_pre_hidden = None

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            if out_hidden is not None:
                dn_out_hidden, out_hidden = torch.split(out_hidden, dn_meta['dn_num_split'], dim=2)
            else:
                dn_out_hidden = None

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
        else:
            dn_pre_hidden = None
            dn_out_hidden = None

        if self.use_picking_point_head:
            pre_picking_logits, pre_picking_offsets, pre_point_quality_logits, pre_point_selector_logits, pre_point_accept_logits, pre_weak_heatmap_logits, pre_toproi_simcc_x_logits, pre_toproi_simcc_y_logits, pre_toproi_heatmap_logits = self._predict_point_branch(
                pre_hidden,
                pre_bboxes,
                pre_logits,
                proj_feats,
                self.pre_picking_head,
                self.pre_picking_offset_head,
                self.pre_point_quality_head,
                self.pre_point_selector_head,
                self.pre_point_accept_head,
                self.pre_weak_heatmap_head,
                self.pre_toproi_simcc_x_head,
                self.pre_toproi_simcc_y_head,
                None,
                self.pre_point_local_proj,
                self.pre_point_full_local_proj,
                self.pre_point_top_local_proj,
                self.pre_point_size_proj,
                self.pre_point_query_pos_proj,
                self.pre_point_box_geom_proj,
                self.pre_point_fusion_head,
            )

            out_picking_logits_list = []
            out_picking_offsets_list = []
            out_point_quality_logits_list = []
            out_point_selector_logits_list = []
            out_point_accept_logits_list = []
            out_weak_heatmap_logits_list = []
            out_toproi_simcc_x_logits_list = []
            out_toproi_simcc_y_logits_list = []
            out_toproi_heatmap_logits_list = []
            final_decoder_layer_idx = out_hidden.shape[0] - 1
            for layer_idx in range(out_hidden.shape[0]):
                heatmap_head_i = None
                if self.dec_toproi_heatmap_head is not None and layer_idx == final_decoder_layer_idx:
                    heatmap_head_i = self.dec_toproi_heatmap_head[layer_idx]
                logits_i, offsets_i, quality_i, selector_i, accept_i, weak_heatmap_i, simcc_x_i, simcc_y_i, heatmap_i = self._predict_point_branch(
                    out_hidden[layer_idx],
                    out_bboxes[layer_idx],
                    out_logits[layer_idx],
                    proj_feats,
                    self.dec_picking_head[layer_idx],
                    self.dec_picking_offset_head[layer_idx],
                    self.dec_point_quality_head[layer_idx] if self.dec_point_quality_head is not None else None,
                    self.dec_point_selector_head[layer_idx] if self.dec_point_selector_head is not None else None,
                    self.dec_point_accept_head[layer_idx] if self.dec_point_accept_head is not None else None,
                    self.dec_weak_heatmap_head[layer_idx] if self.dec_weak_heatmap_head is not None else None,
                    self.dec_toproi_simcc_x_head[layer_idx] if self.dec_toproi_simcc_x_head is not None else None,
                    self.dec_toproi_simcc_y_head[layer_idx] if self.dec_toproi_simcc_y_head is not None else None,
                    heatmap_head_i,
                    self.dec_point_local_proj[layer_idx] if self.dec_point_local_proj is not None else None,
                    self.dec_point_full_local_proj[layer_idx] if self.dec_point_full_local_proj is not None else None,
                    self.dec_point_top_local_proj[layer_idx] if self.dec_point_top_local_proj is not None else None,
                    self.dec_point_size_proj[layer_idx] if self.dec_point_size_proj is not None else None,
                    self.dec_point_query_pos_proj[layer_idx] if self.dec_point_query_pos_proj is not None else None,
                    self.dec_point_box_geom_proj[layer_idx] if self.dec_point_box_geom_proj is not None else None,
                    self.dec_point_fusion_head[layer_idx] if self.dec_point_fusion_head is not None else None,
                )
                out_picking_logits_list.append(logits_i)
                out_picking_offsets_list.append(offsets_i)
                if quality_i is not None:
                    out_point_quality_logits_list.append(quality_i)
                if selector_i is not None:
                    out_point_selector_logits_list.append(selector_i)
                if accept_i is not None:
                    out_point_accept_logits_list.append(accept_i)
                if weak_heatmap_i is not None:
                    out_weak_heatmap_logits_list.append(weak_heatmap_i)
                if simcc_x_i is not None:
                    out_toproi_simcc_x_logits_list.append(simcc_x_i)
                if simcc_y_i is not None:
                    out_toproi_simcc_y_logits_list.append(simcc_y_i)
                if heatmap_i is not None:
                    out_toproi_heatmap_logits_list.append(heatmap_i)
            out_picking_logits = torch.stack(out_picking_logits_list) if out_picking_logits_list else None
            out_picking_offsets = torch.stack(out_picking_offsets_list) if out_picking_offsets_list else None
            out_point_quality_logits = torch.stack(out_point_quality_logits_list) if out_point_quality_logits_list else None
            out_point_selector_logits = torch.stack(out_point_selector_logits_list) if out_point_selector_logits_list else None
            out_point_accept_logits = torch.stack(out_point_accept_logits_list) if out_point_accept_logits_list else None
            out_weak_heatmap_logits = torch.stack(out_weak_heatmap_logits_list) if out_weak_heatmap_logits_list else None
            out_toproi_simcc_x_logits = torch.stack(out_toproi_simcc_x_logits_list) if out_toproi_simcc_x_logits_list else None
            out_toproi_simcc_y_logits = torch.stack(out_toproi_simcc_y_logits_list) if out_toproi_simcc_y_logits_list else None
            out_toproi_heatmap_logits = torch.stack(out_toproi_heatmap_logits_list) if out_toproi_heatmap_logits_list else None

            if dn_out_hidden is not None:
                dn_out_picking_logits_list = []
                dn_out_picking_offsets_list = []
                dn_out_point_quality_logits_list = []
                dn_out_point_selector_logits_list = []
                dn_out_point_accept_logits_list = []
                dn_out_weak_heatmap_logits_list = []
                dn_out_toproi_simcc_x_logits_list = []
                dn_out_toproi_simcc_y_logits_list = []
                dn_out_toproi_heatmap_logits_list = []
                for layer_idx in range(dn_out_hidden.shape[0]):
                    logits_i, offsets_i, quality_i, selector_i, accept_i, weak_heatmap_i, simcc_x_i, simcc_y_i, heatmap_i = self._predict_point_branch(
                        dn_out_hidden[layer_idx],
                        dn_out_bboxes[layer_idx],
                        dn_out_logits[layer_idx],
                        proj_feats,
                        self.dec_picking_head[layer_idx],
                        self.dec_picking_offset_head[layer_idx],
                        self.dec_point_quality_head[layer_idx] if self.dec_point_quality_head is not None else None,
                        self.dec_point_selector_head[layer_idx] if self.dec_point_selector_head is not None else None,
                        self.dec_point_accept_head[layer_idx] if self.dec_point_accept_head is not None else None,
                        self.dec_weak_heatmap_head[layer_idx] if self.dec_weak_heatmap_head is not None else None,
                        self.dec_toproi_simcc_x_head[layer_idx] if self.dec_toproi_simcc_x_head is not None else None,
                        self.dec_toproi_simcc_y_head[layer_idx] if self.dec_toproi_simcc_y_head is not None else None,
                        None,
                        self.dec_point_local_proj[layer_idx] if self.dec_point_local_proj is not None else None,
                        self.dec_point_full_local_proj[layer_idx] if self.dec_point_full_local_proj is not None else None,
                        self.dec_point_top_local_proj[layer_idx] if self.dec_point_top_local_proj is not None else None,
                        self.dec_point_size_proj[layer_idx] if self.dec_point_size_proj is not None else None,
                        self.dec_point_query_pos_proj[layer_idx] if self.dec_point_query_pos_proj is not None else None,
                        self.dec_point_box_geom_proj[layer_idx] if self.dec_point_box_geom_proj is not None else None,
                        self.dec_point_fusion_head[layer_idx] if self.dec_point_fusion_head is not None else None,
                        roi_boxes_cxcywh=dn_teacher_roi_boxes,
                    )
                    dn_out_picking_logits_list.append(logits_i)
                    dn_out_picking_offsets_list.append(offsets_i)
                    if quality_i is not None:
                        dn_out_point_quality_logits_list.append(quality_i)
                    if selector_i is not None:
                        dn_out_point_selector_logits_list.append(selector_i)
                    if accept_i is not None:
                        dn_out_point_accept_logits_list.append(accept_i)
                    if weak_heatmap_i is not None:
                        dn_out_weak_heatmap_logits_list.append(weak_heatmap_i)
                    if simcc_x_i is not None:
                        dn_out_toproi_simcc_x_logits_list.append(simcc_x_i)
                    if simcc_y_i is not None:
                        dn_out_toproi_simcc_y_logits_list.append(simcc_y_i)
                    if heatmap_i is not None:
                        dn_out_toproi_heatmap_logits_list.append(heatmap_i)
                dn_out_picking_logits = torch.stack(dn_out_picking_logits_list) if dn_out_picking_logits_list else None
                dn_out_picking_offsets = torch.stack(dn_out_picking_offsets_list) if dn_out_picking_offsets_list else None
                dn_out_point_quality_logits = torch.stack(dn_out_point_quality_logits_list) if dn_out_point_quality_logits_list else None
                dn_out_point_selector_logits = torch.stack(dn_out_point_selector_logits_list) if dn_out_point_selector_logits_list else None
                dn_out_point_accept_logits = torch.stack(dn_out_point_accept_logits_list) if dn_out_point_accept_logits_list else None
                dn_out_weak_heatmap_logits = torch.stack(dn_out_weak_heatmap_logits_list) if dn_out_weak_heatmap_logits_list else None
                dn_out_toproi_simcc_x_logits = torch.stack(dn_out_toproi_simcc_x_logits_list) if dn_out_toproi_simcc_x_logits_list else None
                dn_out_toproi_simcc_y_logits = torch.stack(dn_out_toproi_simcc_y_logits_list) if dn_out_toproi_simcc_y_logits_list else None
                dn_out_toproi_heatmap_logits = torch.stack(dn_out_toproi_heatmap_logits_list) if dn_out_toproi_heatmap_logits_list else None
            else:
                dn_out_picking_logits, dn_out_picking_offsets, dn_out_point_quality_logits, dn_out_point_selector_logits, dn_out_point_accept_logits, dn_out_weak_heatmap_logits, dn_out_toproi_simcc_x_logits, dn_out_toproi_simcc_y_logits, dn_out_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None

            if dn_pre_hidden is not None:
                dn_pre_picking_logits, dn_pre_picking_offsets, dn_pre_point_quality_logits, dn_pre_point_selector_logits, dn_pre_point_accept_logits, dn_pre_weak_heatmap_logits, dn_pre_toproi_simcc_x_logits, dn_pre_toproi_simcc_y_logits, dn_pre_toproi_heatmap_logits = self._predict_point_branch(
                    dn_pre_hidden,
                    dn_pre_bboxes,
                    dn_pre_logits,
                    proj_feats,
                    self.pre_picking_head,
                    self.pre_picking_offset_head,
                    self.pre_point_quality_head,
                    self.pre_point_selector_head,
                    self.pre_point_accept_head,
                    self.pre_weak_heatmap_head,
                    self.pre_toproi_simcc_x_head,
                    self.pre_toproi_simcc_y_head,
                    None,
                    self.pre_point_local_proj,
                    self.pre_point_full_local_proj,
                    self.pre_point_top_local_proj,
                    self.pre_point_size_proj,
                    self.pre_point_query_pos_proj,
                    self.pre_point_box_geom_proj,
                    self.pre_point_fusion_head,
                    roi_boxes_cxcywh=dn_teacher_roi_boxes,
                )
            else:
                dn_pre_picking_logits, dn_pre_picking_offsets, dn_pre_point_quality_logits, dn_pre_point_selector_logits, dn_pre_point_accept_logits, dn_pre_weak_heatmap_logits, dn_pre_toproi_simcc_x_logits, dn_pre_toproi_simcc_y_logits, dn_pre_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None
        else:
            out_picking_logits = None
            out_picking_offsets = None
            out_point_quality_logits = None
            out_point_selector_logits = None
            out_point_accept_logits = None
            out_weak_heatmap_logits = None
            out_toproi_simcc_x_logits = None
            out_toproi_simcc_y_logits = None
            out_toproi_heatmap_logits = None
            dn_out_picking_logits, dn_out_picking_offsets, dn_out_point_quality_logits, dn_out_point_selector_logits, dn_out_point_accept_logits, dn_out_weak_heatmap_logits, dn_out_toproi_simcc_x_logits, dn_out_toproi_simcc_y_logits, dn_out_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None
            pre_picking_logits, pre_picking_offsets, pre_point_quality_logits, pre_point_selector_logits, pre_point_accept_logits, pre_weak_heatmap_logits, pre_toproi_simcc_x_logits, pre_toproi_simcc_y_logits, pre_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None
            dn_pre_picking_logits, dn_pre_picking_offsets, dn_pre_point_quality_logits, dn_pre_point_selector_logits, dn_pre_point_accept_logits, dn_pre_weak_heatmap_logits, dn_pre_toproi_simcc_x_logits, dn_pre_toproi_simcc_y_logits, dn_pre_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None


        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if out_picking_logits is not None:
            # Final per-query outputs consumed by criterion and postprocessor.
            out['pred_has_picking'] = out_picking_logits[-1]
            out['pred_picking_offsets'] = out_picking_offsets[-1]
            if out_point_quality_logits is not None:
                out['pred_point_quality'] = out_point_quality_logits[-1]
            if out_point_selector_logits is not None:
                out['pred_point_selector'] = out_point_selector_logits[-1]
            if out_point_accept_logits is not None:
                out['pred_point_accept'] = out_point_accept_logits[-1]
            if out_weak_heatmap_logits is not None:
                out['pred_weak_heatmap_score'] = out_weak_heatmap_logits[-1]
            if out_toproi_simcc_x_logits is not None and out_toproi_simcc_y_logits is not None:
                out['pred_toproi_simcc_x'] = out_toproi_simcc_x_logits[-1]
                out['pred_toproi_simcc_y'] = out_toproi_simcc_y_logits[-1]
            if out_toproi_heatmap_logits is not None:
                out['pred_toproi_heatmap'] = out_toproi_heatmap_logits[-1]

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
                out_picking_logits[:-1] if out_picking_logits is not None else None,
                out_picking_offsets[:-1] if out_picking_offsets is not None else None,
                out_point_quality_logits[:-1] if out_point_quality_logits is not None else None,
                out_point_selector_logits[:-1] if out_point_selector_logits is not None else None,
                out_point_accept_logits[:-1] if out_point_accept_logits is not None else None,
                out_weak_heatmap_logits[:-1] if out_weak_heatmap_logits is not None else None,
                out_toproi_simcc_x_logits[:-1] if out_toproi_simcc_x_logits is not None else None,
                out_toproi_simcc_y_logits[:-1] if out_toproi_simcc_y_logits is not None else None,
                None,
            )
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            if pre_picking_logits is not None:
                out['pre_outputs']['pred_has_picking'] = pre_picking_logits
                out['pre_outputs']['pred_picking_offsets'] = pre_picking_offsets
                if pre_point_quality_logits is not None:
                    out['pre_outputs']['pred_point_quality'] = pre_point_quality_logits
                if pre_point_selector_logits is not None:
                    out['pre_outputs']['pred_point_selector'] = pre_point_selector_logits
                if pre_point_accept_logits is not None:
                    out['pre_outputs']['pred_point_accept'] = pre_point_accept_logits
                if pre_weak_heatmap_logits is not None:
                    out['pre_outputs']['pred_weak_heatmap_score'] = pre_weak_heatmap_logits
                if pre_toproi_simcc_x_logits is not None and pre_toproi_simcc_y_logits is not None:
                    out['pre_outputs']['pred_toproi_simcc_x'] = pre_toproi_simcc_x_logits
                    out['pre_outputs']['pred_toproi_simcc_y'] = pre_toproi_simcc_y_logits
                if pre_toproi_heatmap_logits is not None:
                    out['pre_outputs']['pred_toproi_heatmap'] = pre_toproi_heatmap_logits
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                    dn_out_picking_logits,
                    dn_out_picking_offsets,
                    dn_out_point_quality_logits,
                    dn_out_point_selector_logits,
                    dn_out_point_accept_logits,
                    dn_out_weak_heatmap_logits,
                    dn_out_toproi_simcc_x_logits,
                    dn_out_toproi_simcc_y_logits,
                    None,
                )
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                if dn_pre_picking_logits is not None:
                    out['dn_pre_outputs']['pred_has_picking'] = dn_pre_picking_logits
                    out['dn_pre_outputs']['pred_picking_offsets'] = dn_pre_picking_offsets
                    if dn_pre_point_quality_logits is not None:
                        out['dn_pre_outputs']['pred_point_quality'] = dn_pre_point_quality_logits
                    if dn_pre_point_selector_logits is not None:
                        out['dn_pre_outputs']['pred_point_selector'] = dn_pre_point_selector_logits
                    if dn_pre_point_accept_logits is not None:
                        out['dn_pre_outputs']['pred_point_accept'] = dn_pre_point_accept_logits
                    if dn_pre_weak_heatmap_logits is not None:
                        out['dn_pre_outputs']['pred_weak_heatmap_score'] = dn_pre_weak_heatmap_logits
                    if dn_pre_toproi_simcc_x_logits is not None and dn_pre_toproi_simcc_y_logits is not None:
                        out['dn_pre_outputs']['pred_toproi_simcc_x'] = dn_pre_toproi_simcc_x_logits
                        out['dn_pre_outputs']['pred_toproi_simcc_y'] = dn_pre_toproi_simcc_y_logits
                    if dn_pre_toproi_heatmap_logits is not None:
                        out['dn_pre_outputs']['pred_toproi_heatmap'] = dn_pre_toproi_heatmap_logits
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class,
        outputs_coord,
        outputs_picking_logits=None,
        outputs_picking_offsets=None,
        outputs_point_quality_logits=None,
        outputs_point_selector_logits=None,
        outputs_point_accept_logits=None,
        outputs_weak_heatmap_logits=None,
        outputs_toproi_simcc_x_logits=None,
        outputs_toproi_simcc_y_logits=None,
        outputs_toproi_heatmap_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        results = []
        for idx, (a, b) in enumerate(zip(outputs_class, outputs_coord)):
            item = {'pred_logits': a, 'pred_boxes': b}
            if outputs_picking_logits is not None:
                item['pred_has_picking'] = outputs_picking_logits[idx]
            if outputs_picking_offsets is not None:
                item['pred_picking_offsets'] = outputs_picking_offsets[idx]
            if outputs_point_quality_logits is not None:
                item['pred_point_quality'] = outputs_point_quality_logits[idx]
            if outputs_point_selector_logits is not None:
                item['pred_point_selector'] = outputs_point_selector_logits[idx]
            if outputs_point_accept_logits is not None:
                item['pred_point_accept'] = outputs_point_accept_logits[idx]
            if outputs_weak_heatmap_logits is not None:
                item['pred_weak_heatmap_score'] = outputs_weak_heatmap_logits[idx]
            if outputs_toproi_simcc_x_logits is not None and outputs_toproi_simcc_y_logits is not None:
                item['pred_toproi_simcc_x'] = outputs_toproi_simcc_x_logits[idx]
                item['pred_toproi_simcc_y'] = outputs_toproi_simcc_y_logits[idx]
            if outputs_toproi_heatmap_logits is not None:
                item['pred_toproi_heatmap'] = outputs_toproi_heatmap_logits[idx]
            results.append(item)
        return results


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None,
                       outputs_picking_logits=None, outputs_picking_offsets=None,
                       outputs_point_quality_logits=None,
                       outputs_point_selector_logits=None,
                       outputs_point_accept_logits=None,
                       outputs_weak_heatmap_logits=None,
                       outputs_toproi_simcc_x_logits=None,
                       outputs_toproi_simcc_y_logits=None,
                       outputs_toproi_heatmap_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        results = []
        for idx, (a, b, c, d) in enumerate(zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)):
            item = {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_corners': c,
                'ref_points': d,
                'teacher_corners': teacher_corners,
                'teacher_logits': teacher_logits,
            }
            if outputs_picking_logits is not None:
                item['pred_has_picking'] = outputs_picking_logits[idx]
            if outputs_picking_offsets is not None:
                item['pred_picking_offsets'] = outputs_picking_offsets[idx]
            if outputs_point_quality_logits is not None:
                item['pred_point_quality'] = outputs_point_quality_logits[idx]
            if outputs_point_selector_logits is not None:
                item['pred_point_selector'] = outputs_point_selector_logits[idx]
            if outputs_point_accept_logits is not None:
                item['pred_point_accept'] = outputs_point_accept_logits[idx]
            if outputs_weak_heatmap_logits is not None:
                item['pred_weak_heatmap_score'] = outputs_weak_heatmap_logits[idx]
            if outputs_toproi_simcc_x_logits is not None and outputs_toproi_simcc_y_logits is not None:
                item['pred_toproi_simcc_x'] = outputs_toproi_simcc_x_logits[idx]
                item['pred_toproi_simcc_y'] = outputs_toproi_simcc_y_logits[idx]
            if outputs_toproi_heatmap_logits is not None:
                item['pred_toproi_heatmap'] = outputs_toproi_heatmap_logits[idx]
            results.append(item)
        return results
