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
from .point_utils import top_roi_local_from_boxes_and_offsets, top_roi_offsets_from_local_delta
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
                 use_stem_aux=False,
                 stem_aux_type='visibility',
                 stem_aux_loss_weight=0.2,
                 stem_aux_input='query',
                 stem_aux_debug=True,
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
                 use_point_reliability_head=False,
                 point_reliability_head_layers=2,
                 point_reliability_detach_input=True,
                 point_reliability_use_geometry=False,
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
                 use_grouped_picking_query=False,
                 grouped_picking_query_hidden_scale=1.0,
                 grouped_picking_use_toproi=True,
                 grouped_picking_use_box_geometry=True,
                 grouped_picking_residual_offset=False,
                 use_qdpt_lite=False,
                 qdpt_offset_only=True,
                 qdpt_num_heads=4,
                 qdpt_dropout=0.0,
                 qdpt_toproi_levels=[0, 1],
                 qdpt_toproi_map_size=5,
                 qdpt_use_multilevel_toproi=True,
                 qdpt_use_point_prior=True,
                 qdpt_prior_residual=True,
                 qdpt_init_identity=True,
                 qdpt_use_has_distill=True,
                 qdpt_debug_fields=True,
                 use_dpo_head=False,
                 dpo_num_bins_x=96,
                 dpo_num_bins_y=96,
                 dpo_x_min=None,
                 dpo_x_max=None,
                 dpo_y_min=None,
                 dpo_y_max=None,
                 dpo_target_type='soft_ce',
                 dpo_soft_sigma=1.5,
                 dpo_use_expectation_decode=True,
                 dpo_inference_mode='raw_offset',
                 dpo_blend_init=0.0,
                 dpo_debug_fields=True,
                 use_hrpb=False,
                 point_hr_levels=('P2', 'P3'),
                 point_hr_roi_size=5,
                 point_hr_channels=128,
                 point_hr_gate_init=0.0,
                 point_hr_use_p2=True,
                 point_hr_use_p3=True,
                 point_hr_debug=True,
                 use_c2f_ccr=False,
                 c2f_grid_size=7,
                 c2f_roi_size=5,
                 c2f_hidden_dim=256,
                 c2f_gate_init=0.0,
                 c2f_use_toproi_map=True,
                 c2f_residual_tanh=True,
                 c2f_debug=True,
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
        self.use_stem_aux = bool(use_stem_aux)
        self.stem_aux_type = str(stem_aux_type).strip().lower()
        self.stem_aux_loss_weight = float(stem_aux_loss_weight)
        self.stem_aux_input = str(stem_aux_input).strip().lower()
        self.stem_aux_debug = bool(stem_aux_debug)
        if self.stem_aux_type != 'visibility':
            raise ValueError(f"Unsupported stem_aux_type: {stem_aux_type}")
        if self.stem_aux_input != 'query':
            raise ValueError(f"Unsupported stem_aux_input: {stem_aux_input}")
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
        self.use_point_reliability_head = bool(use_point_reliability_head)
        self.point_reliability_head_layers = max(int(point_reliability_head_layers), 1)
        self.point_reliability_detach_input = bool(point_reliability_detach_input)
        self.point_reliability_use_geometry = bool(point_reliability_use_geometry)
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
        self.use_grouped_picking_query = bool(use_grouped_picking_query)
        self.grouped_picking_query_hidden_scale = max(float(grouped_picking_query_hidden_scale), 0.125)
        self.grouped_picking_use_toproi = bool(grouped_picking_use_toproi)
        self.grouped_picking_use_box_geometry = bool(grouped_picking_use_box_geometry)
        self.grouped_picking_residual_offset = bool(grouped_picking_residual_offset)
        self.use_qdpt_lite = bool(use_qdpt_lite)
        self.qdpt_offset_only = bool(qdpt_offset_only)
        self.qdpt_num_heads = max(int(qdpt_num_heads), 1)
        self.qdpt_dropout = float(qdpt_dropout)
        if isinstance(qdpt_toproi_levels, int):
            qdpt_toproi_levels = [qdpt_toproi_levels]
        self.qdpt_toproi_levels = [max(int(level), 0) for level in qdpt_toproi_levels]
        if not self.qdpt_toproi_levels:
            self.qdpt_toproi_levels = [self.point_local_feature_level]
        self.qdpt_toproi_map_size = max(int(qdpt_toproi_map_size), 2)
        self.qdpt_use_multilevel_toproi = bool(qdpt_use_multilevel_toproi)
        self.qdpt_use_point_prior = bool(qdpt_use_point_prior)
        self.qdpt_prior_residual = bool(qdpt_prior_residual)
        self.qdpt_init_identity = bool(qdpt_init_identity)
        self.qdpt_use_has_distill = bool(qdpt_use_has_distill)
        self.qdpt_debug_fields = bool(qdpt_debug_fields)
        self._qdpt_runtime_warnings = set()
        self.use_dpo_head = bool(use_dpo_head)
        self.dpo_num_bins_x = max(int(dpo_num_bins_x), 2)
        self.dpo_num_bins_y = max(int(dpo_num_bins_y), 2)
        self.dpo_x_min = -1.0 if dpo_x_min is None else float(dpo_x_min)
        self.dpo_x_max = 1.0 if dpo_x_max is None else float(dpo_x_max)
        self.dpo_y_min = -1.0 if dpo_y_min is None else float(dpo_y_min)
        self.dpo_y_max = 1.0 if dpo_y_max is None else float(dpo_y_max)
        if self.dpo_x_max <= self.dpo_x_min:
            self.dpo_x_max = self.dpo_x_min + 1.0
        if self.dpo_y_max <= self.dpo_y_min:
            self.dpo_y_max = self.dpo_y_min + 1.0
        self.dpo_target_type = str(dpo_target_type).strip().lower()
        self.dpo_soft_sigma = max(float(dpo_soft_sigma), 1e-6)
        self.dpo_use_expectation_decode = bool(dpo_use_expectation_decode)
        self.dpo_inference_mode = str(dpo_inference_mode).strip().lower()
        if self.dpo_inference_mode not in ('raw_offset', 'dpo_expectation', 'blend'):
            raise ValueError(f"Unsupported dpo_inference_mode: {dpo_inference_mode}")
        self.dpo_debug_fields = bool(dpo_debug_fields)
        self.use_hrpb = bool(use_hrpb)
        self.point_hr_roi_size = max(int(point_hr_roi_size), 2)
        self.point_hr_channels = max(int(point_hr_channels), 8)
        self.point_hr_gate_init = float(point_hr_gate_init)
        self.point_hr_use_p2 = bool(point_hr_use_p2)
        self.point_hr_use_p3 = bool(point_hr_use_p3)
        self.point_hr_debug = bool(point_hr_debug)
        self.use_c2f_ccr = bool(use_c2f_ccr)
        self.c2f_grid_size = max(int(c2f_grid_size), 2)
        self.c2f_roi_size = max(int(c2f_roi_size), 2)
        self.c2f_hidden_dim = max(int(c2f_hidden_dim), 8)
        self.c2f_gate_init = float(c2f_gate_init)
        self.c2f_use_toproi_map = bool(c2f_use_toproi_map)
        self.c2f_residual_tanh = bool(c2f_residual_tanh)
        self.c2f_debug = bool(c2f_debug)
        if isinstance(point_hr_levels, str):
            point_hr_levels = [item.strip() for item in point_hr_levels.split(',') if item.strip()]
        self.point_hr_levels = [str(level).strip().upper() for level in point_hr_levels]
        self.point_hr_level_indices = []
        self.point_hr_unavailable_levels = []
        for level in self.point_hr_levels:
            if level == 'P2':
                self.point_hr_unavailable_levels.append(level)
            elif level == 'P3' and self.point_hr_use_p3:
                self.point_hr_level_indices.append(0)
            elif level == 'P4':
                self.point_hr_level_indices.append(1)
            elif level == 'P5':
                self.point_hr_level_indices.append(2)
        self.point_hr_level_indices = sorted(set(self.point_hr_level_indices))
        self._hrpb_runtime_warnings = set()
        self._hrpb_last_feature_shapes = []
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
        self.dec_stem_head = None
        self.dec_point_quality_head = None
        self.dec_point_selector_head = None
        self.dec_point_accept_head = None
        self.dec_point_reliability_head = None
        self.dec_weak_heatmap_head = None
        self.dec_toproi_simcc_x_head = None
        self.dec_toproi_simcc_y_head = None
        self.dec_toproi_heatmap_head = None
        self.pre_picking_head = None
        self.pre_picking_offset_head = None
        self.pre_stem_head = None
        self.pre_point_quality_head = None
        self.pre_point_selector_head = None
        self.pre_point_accept_head = None
        self.pre_point_reliability_head = None
        self.pre_weak_heatmap_head = None
        self.pre_toproi_simcc_x_head = None
        self.pre_toproi_simcc_y_head = None
        self.pre_toproi_heatmap_head = None
        self.dec_grouped_query_pos_proj = None
        self.dec_grouped_toproi_proj = None
        self.dec_grouped_fusion_head = None
        self.dec_grouped_offset_head = None
        self.pre_grouped_query_pos_proj = None
        self.pre_grouped_toproi_proj = None
        self.pre_grouped_fusion_head = None
        self.pre_grouped_offset_head = None
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
        self.dec_qdpt_query_pos_proj = None
        self.pre_qdpt_query_pos_proj = None
        self.dec_qdpt_attn = None
        self.pre_qdpt_attn = None
        self.dec_qdpt_norm = None
        self.pre_qdpt_norm = None
        self.dec_qdpt_delta_head = None
        self.pre_qdpt_delta_head = None
        self.dec_qdpt_prior_head = None
        self.pre_qdpt_prior_head = None
        self.dec_qdpt_gate = None
        self.pre_qdpt_gate = None
        self.qdpt_level_embed = None
        self.dec_dpo_x_head = None
        self.dec_dpo_y_head = None
        self.pre_dpo_x_head = None
        self.pre_dpo_y_head = None
        self.dpo_blend_alpha = None
        self.point_hr_tower = None
        self.point_hr_delta_head = None
        self.point_hr_gate = None
        self.c2f_roi_encoder = None
        self.c2f_grid_head = None
        self.c2f_residual_head = None
        self.c2f_gate = None
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
            if self.use_stem_aux:
                self.dec_stem_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, 1, self.has_picking_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, 1, self.has_picking_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_stem_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    1,
                    self.has_picking_head_layers,
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
            if self.use_point_reliability_head:
                self.dec_point_reliability_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, 1, self.point_reliability_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(
                            scaled_dim,
                            point_hidden_dim_scaled,
                            1,
                            self.point_reliability_head_layers,
                            mlp_act,
                        )
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_point_reliability_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    1,
                    self.point_reliability_head_layers,
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
            if self.use_grouped_picking_query:
                # K=1 grouped picking query: derive a keypoint-specific token
                # from each grape query, then regress the picking offset from
                # that token plus local TopROI and box geometry cues.  This is
                # intentionally separate from the standard offset_feature path.
                grouped_hidden_dim = max(int(round(hidden_dim * self.grouped_picking_query_hidden_scale)), 1)
                grouped_hidden_dim_scaled = max(int(round(scaled_dim * self.grouped_picking_query_hidden_scale)), 1)
                self.pre_grouped_query_pos_proj = nn.Linear(hidden_dim, hidden_dim)
                self.dec_grouped_query_pos_proj = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.eval_idx + 1)]
                  + [nn.Linear(hidden_dim, scaled_dim) for _ in range(num_layers - self.eval_idx - 1)]
                )
                if self.grouped_picking_use_toproi:
                    self.pre_grouped_toproi_proj = nn.Linear(hidden_dim, hidden_dim)
                    self.dec_grouped_toproi_proj = nn.ModuleList(
                        [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.eval_idx + 1)]
                      + [nn.Linear(hidden_dim, scaled_dim) for _ in range(num_layers - self.eval_idx - 1)]
                    )
                grouped_input_parts = 1 + int(self.grouped_picking_use_toproi)
                grouped_input_dim = grouped_input_parts * hidden_dim + (4 if self.grouped_picking_use_box_geometry else 0)
                grouped_input_dim_scaled = grouped_input_parts * scaled_dim + (4 if self.grouped_picking_use_box_geometry else 0)
                self.pre_grouped_fusion_head = self._make_point_head(
                    grouped_input_dim,
                    grouped_hidden_dim,
                    hidden_dim,
                    2,
                    mlp_act,
                )
                self.dec_grouped_fusion_head = nn.ModuleList(
                    [
                        self._make_point_head(grouped_input_dim, grouped_hidden_dim, hidden_dim, 2, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(grouped_input_dim_scaled, grouped_hidden_dim_scaled, scaled_dim, 2, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.pre_grouped_offset_head = self._make_point_head(
                    hidden_dim,
                    grouped_hidden_dim,
                    2,
                    self.picking_offset_head_layers,
                    mlp_act,
                )
                self.dec_grouped_offset_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, grouped_hidden_dim, 2, self.picking_offset_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, grouped_hidden_dim_scaled, 2, self.picking_offset_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
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
            if self.use_qdpt_lite:
                if hidden_dim % self.qdpt_num_heads != 0:
                    raise ValueError(
                        f"qdpt_num_heads={self.qdpt_num_heads} must divide hidden_dim={hidden_dim}"
                    )
                self.qdpt_level_embed = nn.Embedding(max(self.num_levels, max(self.qdpt_toproi_levels) + 1), hidden_dim)
                self.pre_qdpt_query_pos_proj = nn.Linear(hidden_dim, hidden_dim)
                self.dec_qdpt_query_pos_proj = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
                )
                self.pre_qdpt_attn = nn.MultiheadAttention(
                    hidden_dim,
                    self.qdpt_num_heads,
                    dropout=self.qdpt_dropout,
                    batch_first=True,
                )
                self.dec_qdpt_attn = nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            hidden_dim,
                            self.qdpt_num_heads,
                            dropout=self.qdpt_dropout,
                            batch_first=True,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.pre_qdpt_norm = nn.LayerNorm(hidden_dim)
                self.dec_qdpt_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
                self.pre_qdpt_delta_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    2,
                    self.picking_offset_head_layers,
                    mlp_act,
                )
                self.dec_qdpt_delta_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, 2, self.picking_offset_head_layers, mlp_act)
                        for _ in range(num_layers)
                    ]
                )
                self.pre_qdpt_prior_head = self._make_point_head(4, point_hidden_dim, 2, 2, mlp_act)
                self.dec_qdpt_prior_head = nn.ModuleList(
                    [self._make_point_head(4, point_hidden_dim, 2, 2, mlp_act) for _ in range(num_layers)]
                )
                init_gate = 1.0e-3 if self.qdpt_init_identity else 1.0
                self.pre_qdpt_gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))
                self.dec_qdpt_gate = nn.ParameterList(
                    [nn.Parameter(torch.tensor(init_gate, dtype=torch.float32)) for _ in range(num_layers)]
                )
            if self.use_dpo_head:
                self.pre_dpo_x_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    self.dpo_num_bins_x,
                    self.picking_offset_head_layers,
                    mlp_act,
                )
                self.pre_dpo_y_head = self._make_point_head(
                    hidden_dim,
                    point_hidden_dim,
                    self.dpo_num_bins_y,
                    self.picking_offset_head_layers,
                    mlp_act,
                )
                self.dec_dpo_x_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, self.dpo_num_bins_x, self.picking_offset_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, self.dpo_num_bins_x, self.picking_offset_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.dec_dpo_y_head = nn.ModuleList(
                    [
                        self._make_point_head(hidden_dim, point_hidden_dim, self.dpo_num_bins_y, self.picking_offset_head_layers, mlp_act)
                        for _ in range(self.eval_idx + 1)
                    ]
                  + [
                        self._make_point_head(scaled_dim, point_hidden_dim_scaled, self.dpo_num_bins_y, self.picking_offset_head_layers, mlp_act)
                        for _ in range(num_layers - self.eval_idx - 1)
                    ]
                )
                self.dpo_blend_alpha = nn.Parameter(torch.tensor(float(dpo_blend_init), dtype=torch.float32))
            if self.use_hrpb:
                norm_groups = 8 if self.point_hr_channels % 8 == 0 else 1
                self.point_hr_tower = nn.Sequential(
                    nn.Conv2d(hidden_dim, self.point_hr_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, self.point_hr_channels),
                    get_activation(mlp_act),
                    nn.Conv2d(self.point_hr_channels, self.point_hr_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, self.point_hr_channels),
                    get_activation(mlp_act),
                )
                self.point_hr_delta_head = self._make_point_head(
                    self.point_hr_channels,
                    self.point_hr_channels,
                    2,
                    2,
                    mlp_act,
                )
                self.point_hr_gate = nn.Parameter(torch.tensor(self.point_hr_gate_init, dtype=torch.float32))
            if self.use_c2f_ccr:
                norm_groups = 8 if self.c2f_hidden_dim % 8 == 0 else 1
                self.c2f_roi_encoder = nn.Sequential(
                    nn.Conv2d(hidden_dim, self.c2f_hidden_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, self.c2f_hidden_dim),
                    get_activation(mlp_act),
                    nn.Conv2d(self.c2f_hidden_dim, self.c2f_hidden_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, self.c2f_hidden_dim),
                    get_activation(mlp_act),
                )
                self.c2f_grid_head = nn.Conv2d(self.c2f_hidden_dim, 1, 1)
                self.c2f_residual_head = nn.Conv2d(self.c2f_hidden_dim, 2, 1)
                self.c2f_gate = nn.Parameter(torch.tensor(self.c2f_gate_init, dtype=torch.float32))
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
        if self.dec_point_reliability_head is not None:
            self.dec_point_reliability_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_point_reliability_head[self.eval_idx]]
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
        if self.dec_grouped_query_pos_proj is not None:
            self.dec_grouped_query_pos_proj = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_grouped_query_pos_proj[self.eval_idx]]
            )
        if self.dec_grouped_toproi_proj is not None:
            self.dec_grouped_toproi_proj = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_grouped_toproi_proj[self.eval_idx]]
            )
        if self.dec_grouped_fusion_head is not None:
            self.dec_grouped_fusion_head = nn.ModuleList(
                [nn.Identity()] * (self.eval_idx) + [self.dec_grouped_fusion_head[self.eval_idx]]
            )
        if self.dec_grouped_offset_head is not None:
            self.dec_grouped_offset_head = nn.ModuleList(
                [
                    self.dec_grouped_offset_head[i] if i <= self.eval_idx else nn.Identity()
                    for i in range(len(self.dec_grouped_offset_head))
                ]
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

    def _decode_dpo_offsets(self, logits_x: torch.Tensor, logits_y: torch.Tensor) -> torch.Tensor:
        dtype = logits_x.dtype
        device = logits_x.device
        x_bins = torch.linspace(self.dpo_x_min, self.dpo_x_max, logits_x.shape[-1], device=device, dtype=dtype)
        y_bins = torch.linspace(self.dpo_y_min, self.dpo_y_max, logits_y.shape[-1], device=device, dtype=dtype)
        prob_x = F.softmax(logits_x, dim=-1)
        prob_y = F.softmax(logits_y, dim=-1)
        off_x = (prob_x * x_bins).sum(dim=-1)
        off_y = (prob_y * y_bins).sum(dim=-1)
        return torch.stack((off_x, off_y), dim=-1)

    @staticmethod
    def _dpo_distribution_debug(logits_x: torch.Tensor, logits_y: torch.Tensor):
        prob_x = F.softmax(logits_x.float(), dim=-1)
        prob_y = F.softmax(logits_y.float(), dim=-1)
        entropy_x = -(prob_x * prob_x.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_y = -(prob_y * prob_y.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            'entropy_x': entropy_x,
            'entropy_y': entropy_y,
            'maxprob_x': prob_x.max(dim=-1).values,
            'maxprob_y': prob_y.max(dim=-1).values,
        }

    def _predict_dpo_offsets(self, offset_feature, raw_offsets, dpo_x_head=None, dpo_y_head=None):
        if not self.use_dpo_head or dpo_x_head is None or dpo_y_head is None:
            return None, None, None, None, None
        logits_x = dpo_x_head(offset_feature)
        logits_y = dpo_y_head(offset_feature)
        dpo_offsets = self._decode_dpo_offsets(logits_x, logits_y)
        if self.dpo_blend_alpha is None:
            blend_offsets = raw_offsets
        else:
            blend_offsets = raw_offsets + self.dpo_blend_alpha.to(dtype=raw_offsets.dtype, device=raw_offsets.device) * (
                dpo_offsets.to(dtype=raw_offsets.dtype) - raw_offsets
            )
        debug = self._dpo_distribution_debug(logits_x, logits_y) if self.dpo_debug_fields else None
        return logits_x, logits_y, dpo_offsets.to(dtype=raw_offsets.dtype), blend_offsets, debug

    def _pool_hrpb_roi_features(self, proj_feats: List[torch.Tensor], boxes_cxcywh: torch.Tensor):
        if (
            not self.use_hrpb
            or self.point_hr_tower is None
            or boxes_cxcywh is None
            or not self.point_hr_level_indices
        ):
            return None
        bs, query_count, _ = boxes_cxcywh.shape
        pooled_features = []
        shape_rows = []
        if 'P2' in self.point_hr_unavailable_levels:
            self._hrpb_runtime_warnings.add(
                'P2 unavailable: current HGNetv2 return_idx exposes only P3/P4/P5 to RTv4.'
            )
        for level_idx in self.point_hr_level_indices:
            if level_idx >= len(proj_feats):
                self._hrpb_runtime_warnings.add(f'HRPB requested feature index {level_idx} but only {len(proj_feats)} levels exist.')
                continue
            feat = proj_feats[level_idx]
            _, channels, feat_h, feat_w = feat.shape
            if query_count == 0:
                maps = feat.new_zeros((bs, 0, channels, self.point_hr_roi_size, self.point_hr_roi_size))
            else:
                rois_xyxy = self._build_point_top_rois(
                    boxes_cxcywh,
                    feat_h,
                    feat_w,
                    self.point_offset_top_local_width_scale,
                    self.point_offset_top_local_y_min_ratio,
                    self.point_offset_top_local_y_max_ratio,
                )
                batch_ids = torch.arange(bs, device=feat.device, dtype=rois_xyxy.dtype).view(bs, 1, 1).expand(
                    bs, query_count, 1
                )
                rois = torch.cat((batch_ids, rois_xyxy), dim=-1).reshape(-1, 5)
                pooled = torchvision.ops.roi_align(
                    feat,
                    rois,
                    output_size=(self.point_hr_roi_size, self.point_hr_roi_size),
                    spatial_scale=1.0,
                    aligned=True,
                )
                maps = pooled.reshape(bs, query_count, channels, self.point_hr_roi_size, self.point_hr_roi_size)
            tower_out = self.point_hr_tower(maps.reshape(bs * query_count, channels, self.point_hr_roi_size, self.point_hr_roi_size))
            pooled = tower_out.mean(dim=(-1, -2)).reshape(bs, query_count, self.point_hr_channels)
            pooled_features.append(pooled)
            level_name = {0: 'P3', 1: 'P4', 2: 'P5'}.get(level_idx, f'idx{level_idx}')
            shape_rows.append(
                {
                    'level': level_name,
                    'feature_index': int(level_idx),
                    'stride': int(self.feat_strides[level_idx]) if level_idx < len(self.feat_strides) else None,
                    'feature_shape': [int(v) for v in feat.shape],
                    'roi_map_shape': [int(v) for v in maps.shape],
                    'tower_output_shape': [int(v) for v in tower_out.shape],
                }
            )
        self._hrpb_last_feature_shapes = shape_rows
        if not pooled_features:
            return None
        return torch.stack(pooled_features, dim=0).mean(dim=0)

    def _apply_hrpb_offsets(self, raw_offsets: torch.Tensor, boxes_cxcywh: torch.Tensor, proj_feats: List[torch.Tensor]):
        if (
            not self.use_hrpb
            or raw_offsets is None
            or boxes_cxcywh is None
            or self.point_hr_delta_head is None
            or self.point_hr_gate is None
        ):
            return raw_offsets, None
        hr_feature = self._pool_hrpb_roi_features(proj_feats, boxes_cxcywh)
        if hr_feature is None:
            return raw_offsets, None
        hr_delta = self.point_hr_delta_head(hr_feature).to(dtype=raw_offsets.dtype)
        gate = self.point_hr_gate.to(device=raw_offsets.device, dtype=raw_offsets.dtype)
        final_offsets = raw_offsets + gate * hr_delta
        debug = {
            'raw': raw_offsets,
            'delta': hr_delta,
            'gate': gate.reshape(1),
            'feature_norm': hr_feature.norm(dim=-1, keepdim=True).to(dtype=raw_offsets.dtype),
        }
        return final_offsets, debug

    def _c2f_cell_centers(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid = self.c2f_grid_size
        coord = (torch.arange(grid, device=device, dtype=dtype) + 0.5) / float(grid)
        yy, xx = torch.meshgrid(coord, coord, indexing='ij')
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    def _apply_c2f_ccr_offsets(self, raw_offsets: torch.Tensor, boxes_cxcywh: torch.Tensor, proj_feats: List[torch.Tensor]):
        if (
            not self.use_c2f_ccr
            or raw_offsets is None
            or boxes_cxcywh is None
            or self.c2f_roi_encoder is None
            or self.c2f_grid_head is None
            or self.c2f_residual_head is None
            or self.c2f_gate is None
        ):
            return raw_offsets, None

        c2f_boxes = boxes_cxcywh.detach()
        roi_maps = self._pool_point_local_maps(
            proj_feats,
            c2f_boxes,
            roi_type='top',
            output_size=self.c2f_roi_size,
        ).detach()
        bs, query_count, channels, roi_h, roi_w = roi_maps.shape
        flat_maps = roi_maps.reshape(bs * query_count, channels, roi_h, roi_w)
        encoded = self.c2f_roi_encoder(flat_maps)
        encoded = F.interpolate(
            encoded,
            size=(self.c2f_grid_size, self.c2f_grid_size),
            mode='bilinear',
            align_corners=False,
        )
        grid_logits = self.c2f_grid_head(encoded).flatten(1).reshape(bs, query_count, -1)
        residual_map = self.c2f_residual_head(encoded)
        residual = residual_map.permute(0, 2, 3, 1).reshape(bs, query_count, -1, 2)
        if self.c2f_residual_tanh:
            residual = torch.tanh(residual)
        residual = residual.to(dtype=raw_offsets.dtype) * (0.5 / float(self.c2f_grid_size))

        prob = F.softmax(grid_logits, dim=-1).to(dtype=raw_offsets.dtype)
        centers = self._c2f_cell_centers(raw_offsets.device, raw_offsets.dtype).view(1, 1, -1, 2)
        local_pred = ((centers + residual) * prob.unsqueeze(-1)).sum(dim=2)
        raw_local, _ = top_roi_local_from_boxes_and_offsets(
            c2f_boxes,
            raw_offsets,
            offset_mode='top_center',
            top_anchor_ratio=0.12,
            roi_width_scale=self.point_top_local_width_scale,
            roi_y_min_ratio=self.point_top_local_y_min_ratio,
            roi_y_max_ratio=self.point_top_local_y_max_ratio,
        )
        local_delta = local_pred - raw_local.to(dtype=raw_offsets.dtype)
        delta_offsets = top_roi_offsets_from_local_delta(
            c2f_boxes,
            local_delta,
            roi_width_scale=self.point_top_local_width_scale,
            roi_y_min_ratio=self.point_top_local_y_min_ratio,
            roi_y_max_ratio=self.point_top_local_y_max_ratio,
        ).to(dtype=raw_offsets.dtype)
        gate = self.c2f_gate.to(device=raw_offsets.device, dtype=raw_offsets.dtype)
        refined_offsets = raw_offsets + gate * delta_offsets
        entropy = -(prob * prob.clamp(min=1e-8).log()).sum(dim=-1)
        debug = {
            'raw': raw_offsets,
            'grid_logits': grid_logits,
            'cell_residuals': residual,
            'delta_offsets': delta_offsets,
            'gate': gate.reshape(1),
            'grid_entropy': entropy,
            'toproi_maps_shape': torch.tensor(roi_maps.shape, device=raw_offsets.device),
        }
        return refined_offsets, debug

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

    def _pool_multilevel_point_local_maps(
        self,
        proj_feats: List[torch.Tensor],
        boxes_cxcywh: torch.Tensor,
        output_size: int = None,
        top_roi_params: tuple = None,
    ) -> list:
        """Pool TopROI maps from multiple feature levels for offset-only QDPT."""
        size = self.qdpt_toproi_map_size if output_size is None else max(int(output_size), 2)
        levels = self.qdpt_toproi_levels if self.qdpt_use_multilevel_toproi else [self.point_local_feature_level]
        maps_by_level = []
        for raw_level in levels:
            level_idx = int(raw_level)
            if level_idx >= len(proj_feats):
                self._qdpt_runtime_warnings.add(
                    f"qdpt_toproi_level_{level_idx}_missing_fallback_{min(self.point_local_feature_level, len(proj_feats) - 1)}"
                )
                level_idx = min(self.point_local_feature_level, len(proj_feats) - 1)
            feat = proj_feats[level_idx]
            bs, _, feat_h, feat_w = feat.shape
            _, query_count, _ = boxes_cxcywh.shape
            if query_count == 0:
                maps = feat.new_zeros((bs, 0, feat.shape[1], size, size))
            else:
                rois_xyxy = self._build_point_top_rois(
                    boxes_cxcywh,
                    feat_h,
                    feat_w,
                    *(top_roi_params if top_roi_params is not None else (None, None, None)),
                )
                batch_ids = torch.arange(bs, device=feat.device, dtype=rois_xyxy.dtype).view(bs, 1, 1).expand(
                    bs, query_count, 1
                )
                rois = torch.cat((batch_ids, rois_xyxy), dim=-1).reshape(-1, 5)
                pooled = torchvision.ops.roi_align(
                    feat,
                    rois,
                    output_size=(size, size),
                    spatial_scale=1.0,
                    aligned=True,
                )
                maps = pooled.reshape(bs, query_count, feat.shape[1], size, size)
            maps_by_level.append((level_idx, maps))
        return maps_by_level

    def _tokenize_qdpt_toproi_maps(self, maps_by_level: list, dtype: torch.dtype) -> torch.Tensor:
        tokens = []
        for level_idx, maps in maps_by_level:
            bs, query_count, channels, height, width = maps.shape
            level_tokens = maps.flatten(3).permute(0, 1, 3, 2).to(dtype=dtype)
            if self.qdpt_level_embed is not None:
                level_id = torch.full((1,), int(level_idx), device=maps.device, dtype=torch.long)
                level_embed = self.qdpt_level_embed(level_id).to(dtype=dtype).view(1, 1, 1, channels)
                level_tokens = level_tokens + level_embed
            tokens.append(level_tokens)
        if not tokens:
            raise RuntimeError("QDPT-Lite requires at least one TopROI token level")
        return torch.cat(tokens, dim=2)

    def _apply_qdpt_lite(
        self,
        hidden: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        proj_feats: List[torch.Tensor],
        base_offsets: torch.Tensor,
        query_pos_proj_head,
        attn_head,
        norm_head,
        delta_head,
        prior_head,
        gate,
        roi_boxes_cxcywh: torch.Tensor = None,
        top_roi_params: tuple = None,
    ):
        if (
            not self.use_qdpt_lite
            or hidden is None
            or boxes_cxcywh is None
            or base_offsets is None
            or query_pos_proj_head is None
            or attn_head is None
            or norm_head is None
            or delta_head is None
        ):
            return base_offsets, None

        local_boxes_cxcywh = boxes_cxcywh if roi_boxes_cxcywh is None else roi_boxes_cxcywh
        maps_by_level = self._pool_multilevel_point_local_maps(
            proj_feats,
            local_boxes_cxcywh,
            output_size=self.qdpt_toproi_map_size,
            top_roi_params=top_roi_params,
        )
        local_tokens = self._tokenize_qdpt_toproi_maps(maps_by_level, hidden.dtype)
        query_pos = self.query_pos_head(boxes_cxcywh).to(dtype=hidden.dtype)
        point_token_base = hidden + query_pos_proj_head(query_pos)
        bs, query_count, channels = point_token_base.shape
        if query_count == 0:
            qdpt_delta = base_offsets.new_zeros(base_offsets.shape)
            prior_offset = base_offsets.new_zeros(base_offsets.shape)
            gate_value = gate.to(dtype=base_offsets.dtype) if torch.is_tensor(gate) else base_offsets.new_tensor(float(gate))
            final_offsets = base_offsets + gate_value * (prior_offset + qdpt_delta)
            debug = {
                "base": base_offsets,
                "delta": qdpt_delta,
                "prior": prior_offset,
                "gate": gate_value.reshape(1),
                "point_token_norm": base_offsets.new_zeros((*base_offsets.shape[:2], 1)),
            }
            return final_offsets, debug

        attn_query = point_token_base.reshape(bs * query_count, 1, channels)
        attn_tokens = local_tokens.reshape(bs * query_count, local_tokens.shape[2], channels)
        token_update, _ = attn_head(attn_query, attn_tokens, attn_tokens, need_weights=False)
        point_token = norm_head(point_token_base + token_update.reshape(bs, query_count, channels))
        qdpt_delta = delta_head(point_token)
        prior_offset = (
            prior_head(self._box_geometry_features(boxes_cxcywh)).to(dtype=base_offsets.dtype)
            if self.qdpt_use_point_prior and prior_head is not None
            else torch.zeros_like(base_offsets)
        )
        gate_value = gate.to(device=base_offsets.device, dtype=base_offsets.dtype)
        correction = prior_offset + qdpt_delta if self.qdpt_prior_residual else qdpt_delta
        final_offsets = base_offsets + gate_value * correction
        final_offsets = self._activate_point_offsets(final_offsets)
        debug = {
            "base": base_offsets,
            "delta": qdpt_delta,
            "prior": prior_offset,
            "gate": gate_value.reshape(1),
            "point_token_norm": point_token.norm(dim=-1, keepdim=True),
        }
        return final_offsets, debug

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
        reliability_head,
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
        qdpt_query_pos_proj_head=None,
        qdpt_attn_head=None,
        qdpt_norm_head=None,
        qdpt_delta_head=None,
        qdpt_prior_head=None,
        qdpt_gate=None,
        dpo_x_head=None,
        dpo_y_head=None,
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
        base_picking_offsets = self._activate_point_offsets(offset_head(offset_feature))
        picking_offsets, qdpt_debug = self._apply_qdpt_lite(
            offset_feature,
            boxes_cxcywh,
            proj_feats,
            base_picking_offsets,
            qdpt_query_pos_proj_head,
            qdpt_attn_head,
            qdpt_norm_head,
            qdpt_delta_head,
            qdpt_prior_head,
            qdpt_gate,
            roi_boxes_cxcywh=roi_boxes_cxcywh,
            top_roi_params=(
                self.point_offset_top_local_width_scale,
                self.point_offset_top_local_y_min_ratio,
                self.point_offset_top_local_y_max_ratio,
            ) if self.point_decoupled_roi else None,
        )
        dpo_x_logits, dpo_y_logits, dpo_offsets, dpo_blend_offsets, dpo_debug = self._predict_dpo_offsets(
            offset_feature,
            picking_offsets,
            dpo_x_head,
            dpo_y_head,
        )
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
        if reliability_head is not None:
            # Reliability is a calibration-only head.  By default it sees the
            # already learned point feature but cannot push gradients back into
            # detector, visible-classification, or offset regression branches.
            reliability_feature = offset_feature.detach() if self.point_reliability_detach_input else offset_feature
            reliability_logits = reliability_head(reliability_feature)
        else:
            reliability_logits = None
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
            reliability_logits,
            weak_heatmap_logits,
            simcc_x_logits,
            simcc_y_logits,
            heatmap_logits,
            qdpt_debug,
            dpo_x_logits,
            dpo_y_logits,
            dpo_offsets,
            dpo_blend_offsets,
            dpo_debug,
        )

    def _predict_grouped_picking_offsets(
        self,
        hidden: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        proj_feats: List[torch.Tensor],
        query_pos_proj_head,
        toproi_proj_head,
        fusion_head,
        offset_head,
        base_offsets: torch.Tensor = None,
        roi_boxes_cxcywh: torch.Tensor = None,
    ) -> torch.Tensor:
        if (
            not self.use_grouped_picking_query
            or hidden is None
            or boxes_cxcywh is None
            or query_pos_proj_head is None
            or fusion_head is None
            or offset_head is None
        ):
            return None

        query_pos = self.query_pos_head(boxes_cxcywh).to(dtype=hidden.dtype)
        picking_query = hidden + query_pos_proj_head(query_pos)
        parts = [picking_query]
        local_boxes_cxcywh = boxes_cxcywh if roi_boxes_cxcywh is None else roi_boxes_cxcywh
        if self.grouped_picking_use_toproi and toproi_proj_head is not None:
            local_feat = self._pool_point_local_features(
                proj_feats,
                local_boxes_cxcywh,
                roi_type='top',
                top_roi_params=(
                    self.point_offset_top_local_width_scale,
                    self.point_offset_top_local_y_min_ratio,
                    self.point_offset_top_local_y_max_ratio,
                ) if self.point_decoupled_roi else None,
            )
            parts.append(toproi_proj_head(local_feat).to(dtype=hidden.dtype))
        if self.grouped_picking_use_box_geometry:
            parts.append(self._box_geometry_features(boxes_cxcywh).to(dtype=hidden.dtype))
        grouped_feature = fusion_head(torch.cat(parts, dim=-1))
        grouped_offsets = self._activate_point_offsets(offset_head(grouped_feature))
        if self.grouped_picking_residual_offset and base_offsets is not None:
            grouped_offsets = base_offsets.detach() + grouped_offsets
        return grouped_offsets

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
        if self.pre_point_reliability_head is not None:
            self._init_reg_head_zero(self.pre_point_reliability_head)
        if self.pre_weak_heatmap_head is not None:
            self._init_reg_head_zero(self.pre_weak_heatmap_head)
        if self.pre_toproi_simcc_x_head is not None:
            self._init_reg_head_zero(self.pre_toproi_simcc_x_head)
        if self.pre_toproi_simcc_y_head is not None:
            self._init_reg_head_zero(self.pre_toproi_simcc_y_head)
        if self.pre_point_fusion_head is not None:
            self._init_reg_head_zero(self.pre_point_fusion_head)
        if self.pre_grouped_fusion_head is not None:
            self._init_reg_head_zero(self.pre_grouped_fusion_head)
        if self.pre_grouped_offset_head is not None:
            self._init_reg_head_zero(self.pre_grouped_offset_head)

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
        if self.dec_point_reliability_head is not None:
            for head in self.dec_point_reliability_head:
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
        if self.dec_grouped_fusion_head is not None:
            for head in self.dec_grouped_fusion_head:
                self._init_reg_head_zero(head)
        if self.dec_grouped_offset_head is not None:
            for head in self.dec_grouped_offset_head:
                self._init_reg_head_zero(head)
        if self.qdpt_level_embed is not None:
            init.normal_(self.qdpt_level_embed.weight, std=0.02)
        if self.pre_qdpt_query_pos_proj is not None:
            init.xavier_uniform_(self.pre_qdpt_query_pos_proj.weight)
            init.constant_(self.pre_qdpt_query_pos_proj.bias, 0)
        if self.dec_qdpt_query_pos_proj is not None:
            for head in self.dec_qdpt_query_pos_proj:
                init.xavier_uniform_(head.weight)
                init.constant_(head.bias, 0)
        if self.pre_qdpt_prior_head is not None:
            self._init_reg_head_zero(self.pre_qdpt_prior_head)
        if self.dec_qdpt_prior_head is not None:
            for head in self.dec_qdpt_prior_head:
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
            pre_picking_logits, pre_picking_offsets, pre_point_quality_logits, pre_point_selector_logits, pre_point_accept_logits, pre_point_reliability_logits, pre_weak_heatmap_logits, pre_toproi_simcc_x_logits, pre_toproi_simcc_y_logits, pre_toproi_heatmap_logits, pre_qdpt_debug, pre_dpo_x_logits, pre_dpo_y_logits, pre_dpo_offsets, pre_dpo_blend_offsets, pre_dpo_debug = self._predict_point_branch(
                pre_hidden,
                pre_bboxes,
                pre_logits,
                proj_feats,
                self.pre_picking_head,
                self.pre_picking_offset_head,
                self.pre_point_quality_head,
                self.pre_point_selector_head,
                self.pre_point_accept_head,
                self.pre_point_reliability_head,
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
                self.pre_qdpt_query_pos_proj,
                self.pre_qdpt_attn,
                self.pre_qdpt_norm,
                self.pre_qdpt_delta_head,
                self.pre_qdpt_prior_head,
                self.pre_qdpt_gate,
                self.pre_dpo_x_head,
                self.pre_dpo_y_head,
            )
            pre_picking_offsets, pre_hrpb_debug = self._apply_hrpb_offsets(pre_picking_offsets, pre_bboxes, proj_feats)
            pre_grouped_picking_offsets = self._predict_grouped_picking_offsets(
                pre_hidden,
                pre_bboxes,
                proj_feats,
                self.pre_grouped_query_pos_proj,
                self.pre_grouped_toproi_proj,
                self.pre_grouped_fusion_head,
                self.pre_grouped_offset_head,
                pre_picking_offsets,
            )
            pre_stem_logits = (
                self.pre_stem_head(pre_hidden)
                if self.use_stem_aux and self.pre_stem_head is not None and pre_hidden is not None
                else None
            )

            out_picking_logits_list = []
            out_picking_offsets_list = []
            out_stem_logits_list = []
            out_grouped_picking_offsets_list = []
            out_point_quality_logits_list = []
            out_point_selector_logits_list = []
            out_point_accept_logits_list = []
            out_point_reliability_logits_list = []
            out_weak_heatmap_logits_list = []
            out_toproi_simcc_x_logits_list = []
            out_toproi_simcc_y_logits_list = []
            out_toproi_heatmap_logits_list = []
            out_dpo_x_logits_list = []
            out_dpo_y_logits_list = []
            out_dpo_offsets_list = []
            out_dpo_blend_offsets_list = []
            final_qdpt_debug = None
            final_dpo_debug = None
            final_hrpb_debug = None
            final_c2f_debug = None
            final_decoder_layer_idx = out_hidden.shape[0] - 1
            for layer_idx in range(out_hidden.shape[0]):
                heatmap_head_i = None
                if self.dec_toproi_heatmap_head is not None and layer_idx == final_decoder_layer_idx:
                    heatmap_head_i = self.dec_toproi_heatmap_head[layer_idx]
                logits_i, offsets_i, quality_i, selector_i, accept_i, reliability_i, weak_heatmap_i, simcc_x_i, simcc_y_i, heatmap_i, qdpt_debug_i, dpo_x_i, dpo_y_i, dpo_offsets_i, dpo_blend_offsets_i, dpo_debug_i = self._predict_point_branch(
                    out_hidden[layer_idx],
                    out_bboxes[layer_idx],
                    out_logits[layer_idx],
                    proj_feats,
                    self.dec_picking_head[layer_idx],
                    self.dec_picking_offset_head[layer_idx],
                    self.dec_point_quality_head[layer_idx] if self.dec_point_quality_head is not None else None,
                    self.dec_point_selector_head[layer_idx] if self.dec_point_selector_head is not None else None,
                    self.dec_point_accept_head[layer_idx] if self.dec_point_accept_head is not None else None,
                    self.dec_point_reliability_head[layer_idx] if self.dec_point_reliability_head is not None else None,
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
                    self.dec_qdpt_query_pos_proj[layer_idx] if self.dec_qdpt_query_pos_proj is not None else None,
                    self.dec_qdpt_attn[layer_idx] if self.dec_qdpt_attn is not None else None,
                    self.dec_qdpt_norm[layer_idx] if self.dec_qdpt_norm is not None else None,
                    self.dec_qdpt_delta_head[layer_idx] if self.dec_qdpt_delta_head is not None else None,
                    self.dec_qdpt_prior_head[layer_idx] if self.dec_qdpt_prior_head is not None else None,
                    self.dec_qdpt_gate[layer_idx] if self.dec_qdpt_gate is not None else None,
                    self.dec_dpo_x_head[layer_idx] if self.dec_dpo_x_head is not None else None,
                    self.dec_dpo_y_head[layer_idx] if self.dec_dpo_y_head is not None else None,
                )
                offsets_i, hrpb_debug_i = self._apply_hrpb_offsets(offsets_i, out_bboxes[layer_idx], proj_feats)
                stem_i = (
                    self.dec_stem_head[layer_idx](out_hidden[layer_idx])
                    if self.use_stem_aux and self.dec_stem_head is not None
                    else None
                )
                c2f_debug_i = None
                if layer_idx == final_decoder_layer_idx:
                    offsets_i, c2f_debug_i = self._apply_c2f_ccr_offsets(offsets_i, out_bboxes[layer_idx], proj_feats)
                grouped_offsets_i = self._predict_grouped_picking_offsets(
                    out_hidden[layer_idx],
                    out_bboxes[layer_idx],
                    proj_feats,
                    self.dec_grouped_query_pos_proj[layer_idx] if self.dec_grouped_query_pos_proj is not None else None,
                    self.dec_grouped_toproi_proj[layer_idx] if self.dec_grouped_toproi_proj is not None else None,
                    self.dec_grouped_fusion_head[layer_idx] if self.dec_grouped_fusion_head is not None else None,
                    self.dec_grouped_offset_head[layer_idx] if self.dec_grouped_offset_head is not None else None,
                    offsets_i,
                )
                out_picking_logits_list.append(logits_i)
                out_picking_offsets_list.append(offsets_i)
                if stem_i is not None:
                    out_stem_logits_list.append(stem_i)
                if grouped_offsets_i is not None:
                    out_grouped_picking_offsets_list.append(grouped_offsets_i)
                if quality_i is not None:
                    out_point_quality_logits_list.append(quality_i)
                if selector_i is not None:
                    out_point_selector_logits_list.append(selector_i)
                if accept_i is not None:
                    out_point_accept_logits_list.append(accept_i)
                if reliability_i is not None:
                    out_point_reliability_logits_list.append(reliability_i)
                if weak_heatmap_i is not None:
                    out_weak_heatmap_logits_list.append(weak_heatmap_i)
                if simcc_x_i is not None:
                    out_toproi_simcc_x_logits_list.append(simcc_x_i)
                if simcc_y_i is not None:
                    out_toproi_simcc_y_logits_list.append(simcc_y_i)
                if heatmap_i is not None:
                    out_toproi_heatmap_logits_list.append(heatmap_i)
                if dpo_x_i is not None and dpo_y_i is not None:
                    out_dpo_x_logits_list.append(dpo_x_i)
                    out_dpo_y_logits_list.append(dpo_y_i)
                    out_dpo_offsets_list.append(dpo_offsets_i)
                    out_dpo_blend_offsets_list.append(dpo_blend_offsets_i)
                if layer_idx == final_decoder_layer_idx:
                    final_qdpt_debug = qdpt_debug_i
                    final_dpo_debug = dpo_debug_i
                    final_hrpb_debug = hrpb_debug_i
                    final_c2f_debug = c2f_debug_i
            out_picking_logits = torch.stack(out_picking_logits_list) if out_picking_logits_list else None
            out_picking_offsets = torch.stack(out_picking_offsets_list) if out_picking_offsets_list else None
            out_stem_logits = torch.stack(out_stem_logits_list) if out_stem_logits_list else None
            out_grouped_picking_offsets = torch.stack(out_grouped_picking_offsets_list) if out_grouped_picking_offsets_list else None
            out_point_quality_logits = torch.stack(out_point_quality_logits_list) if out_point_quality_logits_list else None
            out_point_selector_logits = torch.stack(out_point_selector_logits_list) if out_point_selector_logits_list else None
            out_point_accept_logits = torch.stack(out_point_accept_logits_list) if out_point_accept_logits_list else None
            out_point_reliability_logits = torch.stack(out_point_reliability_logits_list) if out_point_reliability_logits_list else None
            out_weak_heatmap_logits = torch.stack(out_weak_heatmap_logits_list) if out_weak_heatmap_logits_list else None
            out_toproi_simcc_x_logits = torch.stack(out_toproi_simcc_x_logits_list) if out_toproi_simcc_x_logits_list else None
            out_toproi_simcc_y_logits = torch.stack(out_toproi_simcc_y_logits_list) if out_toproi_simcc_y_logits_list else None
            out_toproi_heatmap_logits = torch.stack(out_toproi_heatmap_logits_list) if out_toproi_heatmap_logits_list else None
            out_dpo_x_logits = torch.stack(out_dpo_x_logits_list) if out_dpo_x_logits_list else None
            out_dpo_y_logits = torch.stack(out_dpo_y_logits_list) if out_dpo_y_logits_list else None
            out_dpo_offsets = torch.stack(out_dpo_offsets_list) if out_dpo_offsets_list else None
            out_dpo_blend_offsets = torch.stack(out_dpo_blend_offsets_list) if out_dpo_blend_offsets_list else None

            if dn_out_hidden is not None:
                dn_out_picking_logits_list = []
                dn_out_picking_offsets_list = []
                dn_out_grouped_picking_offsets_list = []
                dn_out_point_quality_logits_list = []
                dn_out_point_selector_logits_list = []
                dn_out_point_accept_logits_list = []
                dn_out_point_reliability_logits_list = []
                dn_out_weak_heatmap_logits_list = []
                dn_out_toproi_simcc_x_logits_list = []
                dn_out_toproi_simcc_y_logits_list = []
                dn_out_toproi_heatmap_logits_list = []
                for layer_idx in range(dn_out_hidden.shape[0]):
                    logits_i, offsets_i, quality_i, selector_i, accept_i, reliability_i, weak_heatmap_i, simcc_x_i, simcc_y_i, heatmap_i, _, _, _, _, _, _ = self._predict_point_branch(
                        dn_out_hidden[layer_idx],
                        dn_out_bboxes[layer_idx],
                        dn_out_logits[layer_idx],
                        proj_feats,
                        self.dec_picking_head[layer_idx],
                        self.dec_picking_offset_head[layer_idx],
                        self.dec_point_quality_head[layer_idx] if self.dec_point_quality_head is not None else None,
                        self.dec_point_selector_head[layer_idx] if self.dec_point_selector_head is not None else None,
                        self.dec_point_accept_head[layer_idx] if self.dec_point_accept_head is not None else None,
                        self.dec_point_reliability_head[layer_idx] if self.dec_point_reliability_head is not None else None,
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
                        self.dec_qdpt_query_pos_proj[layer_idx] if self.dec_qdpt_query_pos_proj is not None else None,
                        self.dec_qdpt_attn[layer_idx] if self.dec_qdpt_attn is not None else None,
                        self.dec_qdpt_norm[layer_idx] if self.dec_qdpt_norm is not None else None,
                        self.dec_qdpt_delta_head[layer_idx] if self.dec_qdpt_delta_head is not None else None,
                        self.dec_qdpt_prior_head[layer_idx] if self.dec_qdpt_prior_head is not None else None,
                        self.dec_qdpt_gate[layer_idx] if self.dec_qdpt_gate is not None else None,
                        None,
                        None,
                        roi_boxes_cxcywh=dn_teacher_roi_boxes,
                    )
                    grouped_offsets_i = self._predict_grouped_picking_offsets(
                        dn_out_hidden[layer_idx],
                        dn_out_bboxes[layer_idx],
                        proj_feats,
                        self.dec_grouped_query_pos_proj[layer_idx] if self.dec_grouped_query_pos_proj is not None else None,
                        self.dec_grouped_toproi_proj[layer_idx] if self.dec_grouped_toproi_proj is not None else None,
                        self.dec_grouped_fusion_head[layer_idx] if self.dec_grouped_fusion_head is not None else None,
                        self.dec_grouped_offset_head[layer_idx] if self.dec_grouped_offset_head is not None else None,
                        offsets_i,
                        roi_boxes_cxcywh=dn_teacher_roi_boxes,
                    )
                    dn_out_picking_logits_list.append(logits_i)
                    dn_out_picking_offsets_list.append(offsets_i)
                    if grouped_offsets_i is not None:
                        dn_out_grouped_picking_offsets_list.append(grouped_offsets_i)
                    if quality_i is not None:
                        dn_out_point_quality_logits_list.append(quality_i)
                    if selector_i is not None:
                        dn_out_point_selector_logits_list.append(selector_i)
                    if accept_i is not None:
                        dn_out_point_accept_logits_list.append(accept_i)
                    if reliability_i is not None:
                        dn_out_point_reliability_logits_list.append(reliability_i)
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
                dn_out_grouped_picking_offsets = torch.stack(dn_out_grouped_picking_offsets_list) if dn_out_grouped_picking_offsets_list else None
                dn_out_point_quality_logits = torch.stack(dn_out_point_quality_logits_list) if dn_out_point_quality_logits_list else None
                dn_out_point_selector_logits = torch.stack(dn_out_point_selector_logits_list) if dn_out_point_selector_logits_list else None
                dn_out_point_accept_logits = torch.stack(dn_out_point_accept_logits_list) if dn_out_point_accept_logits_list else None
                dn_out_point_reliability_logits = torch.stack(dn_out_point_reliability_logits_list) if dn_out_point_reliability_logits_list else None
                dn_out_weak_heatmap_logits = torch.stack(dn_out_weak_heatmap_logits_list) if dn_out_weak_heatmap_logits_list else None
                dn_out_toproi_simcc_x_logits = torch.stack(dn_out_toproi_simcc_x_logits_list) if dn_out_toproi_simcc_x_logits_list else None
                dn_out_toproi_simcc_y_logits = torch.stack(dn_out_toproi_simcc_y_logits_list) if dn_out_toproi_simcc_y_logits_list else None
                dn_out_toproi_heatmap_logits = torch.stack(dn_out_toproi_heatmap_logits_list) if dn_out_toproi_heatmap_logits_list else None
            else:
                dn_out_picking_logits, dn_out_picking_offsets, dn_out_grouped_picking_offsets, dn_out_point_quality_logits, dn_out_point_selector_logits, dn_out_point_accept_logits, dn_out_point_reliability_logits, dn_out_weak_heatmap_logits, dn_out_toproi_simcc_x_logits, dn_out_toproi_simcc_y_logits, dn_out_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None, None, None

            if dn_pre_hidden is not None:
                dn_pre_picking_logits, dn_pre_picking_offsets, dn_pre_point_quality_logits, dn_pre_point_selector_logits, dn_pre_point_accept_logits, dn_pre_point_reliability_logits, dn_pre_weak_heatmap_logits, dn_pre_toproi_simcc_x_logits, dn_pre_toproi_simcc_y_logits, dn_pre_toproi_heatmap_logits, _, _, _, _, _, _ = self._predict_point_branch(
                    dn_pre_hidden,
                    dn_pre_bboxes,
                    dn_pre_logits,
                    proj_feats,
                    self.pre_picking_head,
                    self.pre_picking_offset_head,
                    self.pre_point_quality_head,
                    self.pre_point_selector_head,
                    self.pre_point_accept_head,
                    self.pre_point_reliability_head,
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
                    self.pre_qdpt_query_pos_proj,
                    self.pre_qdpt_attn,
                    self.pre_qdpt_norm,
                    self.pre_qdpt_delta_head,
                    self.pre_qdpt_prior_head,
                    self.pre_qdpt_gate,
                    None,
                    None,
                    roi_boxes_cxcywh=dn_teacher_roi_boxes,
                )
                dn_pre_grouped_picking_offsets = self._predict_grouped_picking_offsets(
                    dn_pre_hidden,
                    dn_pre_bboxes,
                    proj_feats,
                    self.pre_grouped_query_pos_proj,
                    self.pre_grouped_toproi_proj,
                    self.pre_grouped_fusion_head,
                    self.pre_grouped_offset_head,
                    dn_pre_picking_offsets,
                    roi_boxes_cxcywh=dn_teacher_roi_boxes,
                )
            else:
                dn_pre_picking_logits, dn_pre_picking_offsets, dn_pre_grouped_picking_offsets, dn_pre_point_quality_logits, dn_pre_point_selector_logits, dn_pre_point_accept_logits, dn_pre_point_reliability_logits, dn_pre_weak_heatmap_logits, dn_pre_toproi_simcc_x_logits, dn_pre_toproi_simcc_y_logits, dn_pre_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None, None, None
        else:
            out_picking_logits = None
            out_picking_offsets = None
            out_stem_logits = None
            out_grouped_picking_offsets = None
            out_point_quality_logits = None
            out_point_selector_logits = None
            out_point_accept_logits = None
            out_point_reliability_logits = None
            out_weak_heatmap_logits = None
            out_toproi_simcc_x_logits = None
            out_toproi_simcc_y_logits = None
            out_toproi_heatmap_logits = None
            out_dpo_x_logits = None
            out_dpo_y_logits = None
            out_dpo_offsets = None
            out_dpo_blend_offsets = None
            final_dpo_debug = None
            final_hrpb_debug = None
            final_c2f_debug = None
            dn_out_picking_logits, dn_out_picking_offsets, dn_out_grouped_picking_offsets, dn_out_point_quality_logits, dn_out_point_selector_logits, dn_out_point_accept_logits, dn_out_point_reliability_logits, dn_out_weak_heatmap_logits, dn_out_toproi_simcc_x_logits, dn_out_toproi_simcc_y_logits, dn_out_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None, None, None
            pre_picking_logits, pre_picking_offsets, pre_stem_logits, pre_grouped_picking_offsets, pre_point_quality_logits, pre_point_selector_logits, pre_point_accept_logits, pre_point_reliability_logits, pre_weak_heatmap_logits, pre_toproi_simcc_x_logits, pre_toproi_simcc_y_logits, pre_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None, None, None, None
            dn_pre_picking_logits, dn_pre_picking_offsets, dn_pre_grouped_picking_offsets, dn_pre_point_quality_logits, dn_pre_point_selector_logits, dn_pre_point_accept_logits, dn_pre_point_reliability_logits, dn_pre_weak_heatmap_logits, dn_pre_toproi_simcc_x_logits, dn_pre_toproi_simcc_y_logits, dn_pre_toproi_heatmap_logits = None, None, None, None, None, None, None, None, None, None, None


        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if out_picking_logits is not None:
            # Final per-query outputs consumed by criterion and postprocessor.
            out['pred_has_picking'] = out_picking_logits[-1]
            out['pred_picking_offsets'] = out_picking_offsets[-1]
            if out_stem_logits is not None:
                out['pred_has_stem'] = out_stem_logits[-1]
            if out_grouped_picking_offsets is not None:
                out['pred_grouped_picking_offsets'] = out_grouped_picking_offsets[-1]
            if out_point_quality_logits is not None:
                out['pred_point_quality'] = out_point_quality_logits[-1]
            if out_point_selector_logits is not None:
                out['pred_point_selector'] = out_point_selector_logits[-1]
            if out_point_accept_logits is not None:
                out['pred_point_accept'] = out_point_accept_logits[-1]
            if out_point_reliability_logits is not None:
                out['pred_point_reliability'] = out_point_reliability_logits[-1]
            if out_weak_heatmap_logits is not None:
                out['pred_weak_heatmap_score'] = out_weak_heatmap_logits[-1]
            if out_toproi_simcc_x_logits is not None and out_toproi_simcc_y_logits is not None:
                out['pred_toproi_simcc_x'] = out_toproi_simcc_x_logits[-1]
                out['pred_toproi_simcc_y'] = out_toproi_simcc_y_logits[-1]
            if out_toproi_heatmap_logits is not None:
                out['pred_toproi_heatmap'] = out_toproi_heatmap_logits[-1]
            if self.use_qdpt_lite and self.qdpt_debug_fields and final_qdpt_debug is not None:
                out['pred_picking_offsets_base'] = final_qdpt_debug['base']
                out['pred_picking_offsets_qdpt_delta'] = final_qdpt_debug['delta']
                out['pred_picking_offsets_prior'] = final_qdpt_debug['prior']
                out['qdpt_gate'] = final_qdpt_debug['gate']
                out['qdpt_point_token_norm'] = final_qdpt_debug['point_token_norm']
            if self.use_hrpb and self.point_hr_debug and final_hrpb_debug is not None:
                out['pred_picking_offsets_raw'] = final_hrpb_debug['raw']
                out['pred_picking_offsets_hr_delta'] = final_hrpb_debug['delta']
                out['point_hr_gate'] = final_hrpb_debug['gate']
                out['point_hr_feature_norm'] = final_hrpb_debug['feature_norm']
            if self.use_c2f_ccr and self.c2f_debug and final_c2f_debug is not None:
                out['pred_picking_offsets_raw'] = final_c2f_debug['raw']
                out['pred_c2f_grid_logits'] = final_c2f_debug['grid_logits']
                out['pred_c2f_cell_residuals'] = final_c2f_debug['cell_residuals']
                out['pred_c2f_delta_offsets'] = final_c2f_debug['delta_offsets']
                out['c2f_gate'] = final_c2f_debug['gate']
                out['c2f_grid_entropy'] = final_c2f_debug['grid_entropy']
                out['c2f_toproi_maps_shape'] = final_c2f_debug['toproi_maps_shape']
            if out_dpo_x_logits is not None and out_dpo_y_logits is not None:
                out['pred_dpo_logits_x'] = out_dpo_x_logits[-1]
                out['pred_dpo_logits_y'] = out_dpo_y_logits[-1]
                out['pred_dpo_offsets'] = out_dpo_offsets[-1]
                out['pred_dpo_blend_offsets'] = out_dpo_blend_offsets[-1]
                if self.dpo_blend_alpha is not None:
                    out['dpo_blend_alpha'] = self.dpo_blend_alpha.reshape(1)
                if self.dpo_debug_fields and final_dpo_debug is not None:
                    out['dpo_entropy_x'] = final_dpo_debug['entropy_x']
                    out['dpo_entropy_y'] = final_dpo_debug['entropy_y']
                    out['dpo_maxprob_x'] = final_dpo_debug['maxprob_x']
                    out['dpo_maxprob_y'] = final_dpo_debug['maxprob_y']

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
                out_stem_logits[:-1] if out_stem_logits is not None else None,
                out_grouped_picking_offsets[:-1] if out_grouped_picking_offsets is not None else None,
                out_point_quality_logits[:-1] if out_point_quality_logits is not None else None,
                out_point_selector_logits[:-1] if out_point_selector_logits is not None else None,
                out_point_accept_logits[:-1] if out_point_accept_logits is not None else None,
                out_point_reliability_logits[:-1] if out_point_reliability_logits is not None else None,
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
                if pre_stem_logits is not None:
                    out['pre_outputs']['pred_has_stem'] = pre_stem_logits
                if pre_grouped_picking_offsets is not None:
                    out['pre_outputs']['pred_grouped_picking_offsets'] = pre_grouped_picking_offsets
                if pre_point_quality_logits is not None:
                    out['pre_outputs']['pred_point_quality'] = pre_point_quality_logits
                if pre_point_selector_logits is not None:
                    out['pre_outputs']['pred_point_selector'] = pre_point_selector_logits
                if pre_point_accept_logits is not None:
                    out['pre_outputs']['pred_point_accept'] = pre_point_accept_logits
                if pre_point_reliability_logits is not None:
                    out['pre_outputs']['pred_point_reliability'] = pre_point_reliability_logits
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
                    None,
                    dn_out_grouped_picking_offsets,
                    dn_out_point_quality_logits,
                    dn_out_point_selector_logits,
                    dn_out_point_accept_logits,
                    dn_out_point_reliability_logits,
                    dn_out_weak_heatmap_logits,
                    dn_out_toproi_simcc_x_logits,
                    dn_out_toproi_simcc_y_logits,
                    None,
                )
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                if dn_pre_picking_logits is not None:
                    out['dn_pre_outputs']['pred_has_picking'] = dn_pre_picking_logits
                    out['dn_pre_outputs']['pred_picking_offsets'] = dn_pre_picking_offsets
                    if dn_pre_grouped_picking_offsets is not None:
                        out['dn_pre_outputs']['pred_grouped_picking_offsets'] = dn_pre_grouped_picking_offsets
                    if dn_pre_point_quality_logits is not None:
                        out['dn_pre_outputs']['pred_point_quality'] = dn_pre_point_quality_logits
                    if dn_pre_point_selector_logits is not None:
                        out['dn_pre_outputs']['pred_point_selector'] = dn_pre_point_selector_logits
                    if dn_pre_point_accept_logits is not None:
                        out['dn_pre_outputs']['pred_point_accept'] = dn_pre_point_accept_logits
                    if dn_pre_point_reliability_logits is not None:
                        out['dn_pre_outputs']['pred_point_reliability'] = dn_pre_point_reliability_logits
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
        outputs_grouped_picking_offsets=None,
        outputs_point_quality_logits=None,
        outputs_point_selector_logits=None,
        outputs_point_accept_logits=None,
        outputs_point_reliability_logits=None,
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
            if outputs_grouped_picking_offsets is not None:
                item['pred_grouped_picking_offsets'] = outputs_grouped_picking_offsets[idx]
            if outputs_point_quality_logits is not None:
                item['pred_point_quality'] = outputs_point_quality_logits[idx]
            if outputs_point_selector_logits is not None:
                item['pred_point_selector'] = outputs_point_selector_logits[idx]
            if outputs_point_accept_logits is not None:
                item['pred_point_accept'] = outputs_point_accept_logits[idx]
            if outputs_point_reliability_logits is not None:
                item['pred_point_reliability'] = outputs_point_reliability_logits[idx]
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
                       outputs_stem_logits=None,
                       outputs_grouped_picking_offsets=None,
                       outputs_point_quality_logits=None,
                       outputs_point_selector_logits=None,
                       outputs_point_accept_logits=None,
                       outputs_point_reliability_logits=None,
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
            if outputs_stem_logits is not None:
                item['pred_has_stem'] = outputs_stem_logits[idx]
            if outputs_grouped_picking_offsets is not None:
                item['pred_grouped_picking_offsets'] = outputs_grouped_picking_offsets[idx]
            if outputs_point_quality_logits is not None:
                item['pred_point_quality'] = outputs_point_quality_logits[idx]
            if outputs_point_selector_logits is not None:
                item['pred_point_selector'] = outputs_point_selector_logits[idx]
            if outputs_point_accept_logits is not None:
                item['pred_point_accept'] = outputs_point_accept_logits[idx]
            if outputs_point_reliability_logits is not None:
                item['pred_point_reliability'] = outputs_point_reliability_logits[idx]
            if outputs_weak_heatmap_logits is not None:
                item['pred_weak_heatmap_score'] = outputs_weak_heatmap_logits[idx]
            if outputs_toproi_simcc_x_logits is not None and outputs_toproi_simcc_y_logits is not None:
                item['pred_toproi_simcc_x'] = outputs_toproi_simcc_x_logits[idx]
                item['pred_toproi_simcc_y'] = outputs_toproi_simcc_y_logits[idx]
            if outputs_toproi_heatmap_logits is not None:
                item['pred_toproi_heatmap'] = outputs_toproi_heatmap_logits[idx]
            results.append(item)
        return results
