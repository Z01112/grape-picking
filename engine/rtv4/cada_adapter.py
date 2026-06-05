from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def _make_norm(channels: int, norm: str) -> nn.Module:
    norm = (norm or "gn").lower()
    if norm == "gn":
        groups = 32
        while channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if norm in {"ln", "layernorm", "layer_norm"}:
        return LayerNorm2d(channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported CADA norm: {norm}")


def _make_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported CADA activation: {name}")


class SELiteAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, activation: str = "silu"):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            _make_activation(activation),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SpatialGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=True),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class CADALevelBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        use_local_conv: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        norm: str = "gn",
        activation: str = "silu",
    ):
        super().__init__()
        self.local = (
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
                nn.Conv2d(channels, channels, 1, bias=True),
                _make_norm(channels, norm),
                _make_activation(activation),
            )
            if use_local_conv
            else nn.Identity()
        )
        self.channel_attn = SELiteAttention(channels, activation=activation) if use_channel_attn else nn.Identity()
        self.spatial_attn = SpatialGate(channels) if use_spatial_attn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class CADAAdapter(nn.Module):
    """Conv-Attention Dense Adapter for same-shape multi-scale encoder outputs."""

    def __init__(
        self,
        num_levels: int,
        channels: int,
        levels: list[int] | tuple[int, ...] | None = None,
        use_local_conv: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        use_cross_scale: bool = True,
        norm: str = "gn",
        activation: str = "silu",
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.channels = int(channels)
        self.levels = list(range(self.num_levels)) if levels is None else [int(v) for v in levels]
        self.active_levels = set(self.levels)
        self.use_cross_scale = bool(use_cross_scale)

        invalid = [level for level in self.levels if level < 0 or level >= self.num_levels]
        if invalid:
            raise ValueError(f"CADA levels out of range: {invalid}; num_levels={self.num_levels}")

        self.level_blocks = nn.ModuleList(
            [
                CADALevelBlock(
                    channels,
                    use_local_conv=use_local_conv,
                    use_channel_attn=use_channel_attn,
                    use_spatial_attn=use_spatial_attn,
                    norm=norm,
                    activation=activation,
                )
                for _ in range(self.num_levels)
            ]
        )
        self.cross_scale_proj = nn.ModuleDict()
        if self.use_cross_scale:
            for dst in range(self.num_levels):
                for src in (dst - 1, dst + 1):
                    if 0 <= src < self.num_levels:
                        self.cross_scale_proj[f"{dst}_from_{src}"] = nn.Conv2d(channels, channels, 1, bias=True)

        self.cada_gamma = nn.Parameter(torch.full((self.num_levels,), float(gate_init)))

    def _resize_like(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="nearest")

    def forward(self, features: list[torch.Tensor], return_debug: bool = False):
        if len(features) != self.num_levels:
            raise ValueError(f"CADA expected {self.num_levels} feature levels, got {len(features)}")

        outputs = []
        debug_rows = []
        for level, feat in enumerate(features):
            if level not in self.active_levels:
                out = feat
                update = torch.zeros_like(feat)
            else:
                update = self.level_blocks[level](feat)
                if self.use_cross_scale:
                    for src in (level - 1, level + 1):
                        if 0 <= src < self.num_levels:
                            proj = self.cross_scale_proj[f"{level}_from_{src}"]
                            update = update + proj(self._resize_like(features[src], feat))
                out = feat + self.cada_gamma[level].view(1, 1, 1, 1) * update
            outputs.append(out)
            if return_debug:
                debug_rows.append(
                    {
                        "level": level,
                        "input_shape": list(feat.shape),
                        "output_shape": list(out.shape),
                        "mean_abs_diff": float((out - feat).abs().mean().detach().cpu().item()),
                        "update_mean_abs": float(update.abs().mean().detach().cpu().item()),
                        "gamma": float(self.cada_gamma[level].detach().cpu().item()),
                        "input_finite": bool(torch.isfinite(feat).all().detach().cpu().item()),
                        "output_finite": bool(torch.isfinite(out).all().detach().cpu().item()),
                    }
                )
        if return_debug:
            return outputs, debug_rows
        return outputs
