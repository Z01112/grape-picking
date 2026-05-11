from __future__ import annotations

from typing import Iterable

import torch


def _normalize_point_offset_mode(mode: str | None) -> str:
    mode = str(mode or "center").strip().lower()
    aliases = {
        "center": "center",
        "center_xy": "center",
        "top_center": "top_center",
        "top_center_y": "top_center",
        "top_y": "top_center",
        "bbox_relative": "bbox_relative",
        "bbox_rel": "bbox_relative",
        "relative_box": "bbox_relative",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported point offset mode: {mode}")
    return aliases[mode]


def point_from_xywh_bbox(box_xywh: Iterable[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x + 0.5 * w, y + 0.5 * h]


def _point_anchor_from_xywh_bbox(
    box_xywh: Iterable[float],
    mode: str = "center",
    top_anchor_ratio: float = 0.0,
) -> list[float]:
    """Return the reference anchor used to encode a picking point.

    top_center uses the horizontal box center and a y position measured from
    the top edge: y_top + top_anchor_ratio * h. This is the geometric prior
    described in the paper.
    """
    x, y, w, h = [float(v) for v in box_xywh]
    mode = _normalize_point_offset_mode(mode)
    if mode == "bbox_relative":
        return [x, y]
    center_x = x + 0.5 * w
    if mode == "center":
        anchor_y = y + 0.5 * h
    else:
        anchor_y = y + float(top_anchor_ratio) * h
    return [center_x, anchor_y]


def offset_from_xywh_bbox(
    box_xywh: Iterable[float],
    point_xy: Iterable[float],
    mode: str = "center",
    top_anchor_ratio: float = 0.0,
) -> list[float]:
    """Encode an absolute point as a width/height-normalized offset."""
    px, py = [float(v) for v in point_xy]
    anchor_x, anchor_y = _point_anchor_from_xywh_bbox(
        box_xywh,
        mode=mode,
        top_anchor_ratio=top_anchor_ratio,
    )
    x, y, w, h = [float(v) for v in box_xywh]
    return [
        (px - anchor_x) / max(w, 1e-6),
        (py - anchor_y) / max(h, 1e-6),
    ]


def absolute_points_from_boxes_and_offsets(
    boxes_cxcywh: torch.Tensor,
    offsets_xy: torch.Tensor,
    mode: str = "center",
    top_anchor_ratio: float = 0.0,
) -> torch.Tensor:
    """Decode normalized offsets into image-coordinate picking points.

    For GPPoint-DETR, mode='top_center' means the offset origin is near the
    upper center of each predicted grape box instead of the bbox center.
    """
    boxes_cxcywh = boxes_cxcywh.to(torch.float32)
    offsets_xy = offsets_xy.to(torch.float32)
    mode = _normalize_point_offset_mode(mode)
    centers = boxes_cxcywh[..., :2].clone()
    if mode == "bbox_relative":
        centers[..., 0] = boxes_cxcywh[..., 0] - 0.5 * boxes_cxcywh[..., 2]
        centers[..., 1] = boxes_cxcywh[..., 1] - 0.5 * boxes_cxcywh[..., 3]
    elif mode == "top_center":
        centers[..., 1] = boxes_cxcywh[..., 1] - 0.5 * boxes_cxcywh[..., 3] + float(top_anchor_ratio) * boxes_cxcywh[..., 3]
    sizes = boxes_cxcywh[..., 2:].clamp(min=1e-6)
    return centers + offsets_xy * sizes


def clamp_points_to_image(points_xy: torch.Tensor, image_sizes_wh: torch.Tensor) -> torch.Tensor:
    points_xy = points_xy.to(torch.float32)
    image_sizes_wh = image_sizes_wh.to(torch.float32)
    out = points_xy.clone()
    max_xy = image_sizes_wh.unsqueeze(-2)
    out[..., 0] = torch.minimum(out[..., 0].clamp_min(0.0), max_xy[..., 0])
    out[..., 1] = torch.minimum(out[..., 1].clamp_min(0.0), max_xy[..., 1])
    return out


def normalized_offsets_from_boxes_and_points(
    boxes_cxcywh: torch.Tensor,
    points_xy: torch.Tensor,
    mode: str = "center",
    top_anchor_ratio: float = 0.0,
) -> torch.Tensor:
    """Generate training targets matching absolute_points_from_boxes_and_offsets."""
    boxes_cxcywh = boxes_cxcywh.to(torch.float32)
    points_xy = points_xy.to(torch.float32)
    mode = _normalize_point_offset_mode(mode)
    centers = boxes_cxcywh[..., :2].clone()
    if mode == "bbox_relative":
        centers[..., 0] = boxes_cxcywh[..., 0] - 0.5 * boxes_cxcywh[..., 2]
        centers[..., 1] = boxes_cxcywh[..., 1] - 0.5 * boxes_cxcywh[..., 3]
    elif mode == "top_center":
        centers[..., 1] = boxes_cxcywh[..., 1] - 0.5 * boxes_cxcywh[..., 3] + float(top_anchor_ratio) * boxes_cxcywh[..., 3]
    sizes = boxes_cxcywh[..., 2:].clamp(min=1e-6)
    return (points_xy - centers) / sizes


def assert_offset_roundtrip(
    boxes_cxcywh: torch.Tensor,
    offsets_xy: torch.Tensor,
    atol: float = 1e-5,
    mode: str = "center",
    top_anchor_ratio: float = 0.0,
) -> None:
    points = absolute_points_from_boxes_and_offsets(
        boxes_cxcywh,
        offsets_xy,
        mode=mode,
        top_anchor_ratio=top_anchor_ratio,
    )
    restored = normalized_offsets_from_boxes_and_points(
        boxes_cxcywh,
        points,
        mode=mode,
        top_anchor_ratio=top_anchor_ratio,
    )
    if not torch.allclose(offsets_xy.to(torch.float32), restored, atol=atol, rtol=0.0):
        raise AssertionError("Point offset round-trip validation failed.")
