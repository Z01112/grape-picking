"""
RT-DETRv4 encoder with a source-faithful ASFF neck transplant.

Adapted from the local public source code:
- third_party/ASFF-master/models/network_blocks.py
- third_party/ASFF-master/models/yolov3_asff.py

This is a complete neck replacement rather than a plug-in block. It keeps the
three-scale RT-DETR backbone outputs and replaces the original FPN/PAN path
with ASFF fusion at each output level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .hybrid_encoder import HybridEncoder, ConvNormLayer_fuse


class ASFFBlock(nn.Module):
    """
    Adaptive Spatial Feature Fusion block.

    Input order follows the original ASFF implementation:
    - x_level_0: highest stride / strongest semantic feature
    - x_level_1: middle stride feature
    - x_level_2: lowest stride / finest feature
    """

    def __init__(self, level, channels=256, compress_c=16, act='silu'):
        super().__init__()
        self.level = level
        self.channels = channels

        if level == 0:
            self.stride_level_1 = ConvNormLayer_fuse(channels, channels, 3, 2, act=act)
            self.stride_level_2 = ConvNormLayer_fuse(channels, channels, 3, 2, act=act)
        elif level == 1:
            self.compress_level_0 = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            self.stride_level_2 = ConvNormLayer_fuse(channels, channels, 3, 2, act=act)
        elif level == 2:
            self.compress_level_0 = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        else:
            raise ValueError(f'Unsupported ASFF level: {level}')

        self.weight_level_0 = ConvNormLayer_fuse(channels, compress_c, 1, 1, act=act)
        self.weight_level_1 = ConvNormLayer_fuse(channels, compress_c, 1, 1, act=act)
        self.weight_level_2 = ConvNormLayer_fuse(channels, compress_c, 1, 1, act=act)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.expand = ConvNormLayer_fuse(channels, channels, 3, 1, act=act)

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(
                F.max_pool2d(x_level_2, kernel_size=3, stride=2, padding=1)
            )
        elif self.level == 1:
            level_0_resized = F.interpolate(
                self.compress_level_0(x_level_0), scale_factor=2.0, mode='nearest'
            )
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        else:
            level_0_resized = F.interpolate(
                self.compress_level_0(x_level_0), scale_factor=4.0, mode='nearest'
            )
            level_1_resized = F.interpolate(x_level_1, scale_factor=2.0, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), dim=1
        )
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)

        fused = (
            level_0_resized * levels_weight[:, 0:1]
            + level_1_resized * levels_weight[:, 1:2]
            + level_2_resized * levels_weight[:, 2:3]
        )
        return self.expand(fused)


@register()
class HybridEncoderASFF(HybridEncoder):
    __share__ = ['eval_spatial_size', ]

    def __init__(
        self,
        in_channels=[256, 512, 1024],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=0.5,
        depth_mult=0.34,
        act='silu',
        eval_spatial_size=None,
        version='dfine',
        distill_teacher_dim=0,
        asff_compress_c=16,
    ):
        if len(in_channels) != 3:
            raise ValueError('HybridEncoderASFF expects exactly 3 feature levels.')

        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
            distill_teacher_dim=distill_teacher_dim,
        )

        # Replace the stock FPN/PAN path with full ASFF fusion.
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.asff_blocks = nn.ModuleList([
            ASFFBlock(level=0, channels=hidden_dim, compress_c=asff_compress_c, act=act),
            ASFFBlock(level=1, channels=hidden_dim, compress_c=asff_compress_c, act=act),
            ASFFBlock(level=2, channels=hidden_dim, compress_c=asff_compress_c, act=act),
        ])

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        distill_student_output = None

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, src_mask=None, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(
                    -1, self.hidden_dim, h, w
                ).contiguous()

                if (
                    self.training
                    and self.feature_projector is not None
                    and enc_ind == self.encoder_idx_for_distillation
                ):
                    distill_student_output = self.feature_projector(
                        proj_feats[enc_ind].permute(0, 2, 3, 1)
                    ).permute(0, 3, 1, 2)

        p3, p4, p5 = proj_feats
        p5_fused = self.asff_blocks[0](p5, p4, p3)
        p4_fused = self.asff_blocks[1](p5, p4, p3)
        p3_fused = self.asff_blocks[2](p5, p4, p3)
        outs = [p3_fused, p4_fused, p5_fused]

        if self.training and distill_student_output is not None:
            return outs, distill_student_output
        return outs
