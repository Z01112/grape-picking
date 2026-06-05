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
    anchor_x_ratio: float = 0.5,
) -> list[float]:
    """Return the reference anchor used to encode a picking point.

    top_center uses a width-relative x anchor and a y position measured from
    the top edge: y_top + top_anchor_ratio * h. The default x ratio is 0.5,
    preserving the original upper-center grape-picking prior.
    """
    x, y, w, h = [float(v) for v in box_xywh]
    mode = _normalize_point_offset_mode(mode)
    if mode == "bbox_relative":
        return [x, y]
    center_x = x + float(anchor_x_ratio) * w
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
    anchor_x_ratio: float = 0.5,
) -> list[float]:
    """Encode an absolute point as a width/height-normalized offset."""
    px, py = [float(v) for v in point_xy]
    anchor_x, anchor_y = _point_anchor_from_xywh_bbox(
        box_xywh,
        mode=mode,
        top_anchor_ratio=top_anchor_ratio,
        anchor_x_ratio=anchor_x_ratio,
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
    anchor_x_ratio: float = 0.5,
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
        centers[..., 0] = boxes_cxcywh[..., 0] - 0.5 * boxes_cxcywh[..., 2] + float(anchor_x_ratio) * boxes_cxcywh[..., 2]
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
    anchor_x_ratio: float = 0.5,
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
        centers[..., 0] = boxes_cxcywh[..., 0] - 0.5 * boxes_cxcywh[..., 2] + float(anchor_x_ratio) * boxes_cxcywh[..., 2]
        centers[..., 1] = boxes_cxcywh[..., 1] - 0.5 * boxes_cxcywh[..., 3] + float(top_anchor_ratio) * boxes_cxcywh[..., 3]
    sizes = boxes_cxcywh[..., 2:].clamp(min=1e-6)
    return (points_xy - centers) / sizes


def top_roi_local_from_boxes_and_offsets(
    boxes_cxcywh: torch.Tensor,
    offsets_xy: torch.Tensor,
    offset_mode: str = "top_center",
    top_anchor_ratio: float = 0.12,
    anchor_x_ratio: float = 0.5,
    roi_width_scale: float = 1.08,
    roi_y_min_ratio: float = -0.10,
    roi_y_max_ratio: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map bbox-normalized point offsets into TopROI local coordinates.

    The returned local coordinates are in the unclamped TopROI coordinate
    system used by the point branch.  A boolean mask marks points inside
    [0, 1] x [0, 1].  This helper is pure geometry and does not depend on
    detector scores, matching cost, or postprocessing.
    """
    boxes_cxcywh = boxes_cxcywh.to(torch.float32)
    offsets_xy = offsets_xy.to(torch.float32)
    points = absolute_points_from_boxes_and_offsets(
        boxes_cxcywh,
        offsets_xy,
        mode=offset_mode,
        top_anchor_ratio=top_anchor_ratio,
        anchor_x_ratio=anchor_x_ratio,
    )
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    roi_w = (w * float(roi_width_scale)).clamp(min=1e-6)
    box_top = cy - 0.5 * h
    roi_x1 = cx - 0.5 * roi_w
    roi_y1 = box_top + float(roi_y_min_ratio) * h
    roi_h = (float(roi_y_max_ratio) - float(roi_y_min_ratio)) * h
    roi_h = roi_h.clamp(min=1e-6)
    local_x = (points[..., 0] - roi_x1) / roi_w
    local_y = (points[..., 1] - roi_y1) / roi_h
    local = torch.stack((local_x, local_y), dim=-1)
    in_roi = (
        torch.isfinite(local).all(dim=-1)
        & (local[..., 0] >= 0.0)
        & (local[..., 0] <= 1.0)
        & (local[..., 1] >= 0.0)
        & (local[..., 1] <= 1.0)
    )
    return local, in_roi


def top_roi_local_from_boxes_and_points(
    boxes_cxcywh: torch.Tensor,
    points_xy: torch.Tensor,
    roi_width_scale: float = 1.08,
    roi_y_min_ratio: float = -0.10,
    roi_y_max_ratio: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map absolute point coordinates into a query box TopROI."""
    boxes_cxcywh = boxes_cxcywh.to(torch.float32)
    points_xy = points_xy.to(torch.float32)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    roi_w = (w * float(roi_width_scale)).clamp(min=1e-6)
    box_top = cy - 0.5 * h
    roi_x1 = cx - 0.5 * roi_w
    roi_y1 = box_top + float(roi_y_min_ratio) * h
    roi_h = (float(roi_y_max_ratio) - float(roi_y_min_ratio)) * h
    roi_h = roi_h.clamp(min=1e-6)
    local_x = (points_xy[..., 0] - roi_x1) / roi_w
    local_y = (points_xy[..., 1] - roi_y1) / roi_h
    local = torch.stack((local_x, local_y), dim=-1)
    in_roi = (
        torch.isfinite(local).all(dim=-1)
        & (local[..., 0] >= 0.0)
        & (local[..., 0] <= 1.0)
        & (local[..., 1] >= 0.0)
        & (local[..., 1] <= 1.0)
    )
    return local, in_roi


def top_roi_offsets_from_local_delta(
    boxes_cxcywh: torch.Tensor,
    local_delta_xy: torch.Tensor,
    roi_width_scale: float = 1.08,
    roi_y_min_ratio: float = -0.10,
    roi_y_max_ratio: float = 0.40,
) -> torch.Tensor:
    """Convert a TopROI-local delta into bbox-normalized offset delta."""
    boxes_cxcywh = boxes_cxcywh.to(torch.float32)
    local_delta_xy = local_delta_xy.to(torch.float32)
    _, _, w, h = boxes_cxcywh.unbind(-1)
    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    roi_w_over_box_w = float(roi_width_scale)
    roi_h_over_box_h = float(roi_y_max_ratio) - float(roi_y_min_ratio)
    scale = torch.stack(
        (
            local_delta_xy.new_full(w.shape, roi_w_over_box_w),
            local_delta_xy.new_full(h.shape, roi_h_over_box_h),
        ),
        dim=-1,
    )
    return local_delta_xy * scale


def assert_offset_roundtrip(
    boxes_cxcywh: torch.Tensor,
    offsets_xy: torch.Tensor,
    atol: float = 1e-5,
    mode: str = "center",
    top_anchor_ratio: float = 0.0,
    anchor_x_ratio: float = 0.5,
) -> None:
    points = absolute_points_from_boxes_and_offsets(
        boxes_cxcywh,
        offsets_xy,
        mode=mode,
        top_anchor_ratio=top_anchor_ratio,
        anchor_x_ratio=anchor_x_ratio,
    )
    restored = normalized_offsets_from_boxes_and_points(
        boxes_cxcywh,
        points,
        mode=mode,
        top_anchor_ratio=top_anchor_ratio,
        anchor_x_ratio=anchor_x_ratio,
    )
    if not torch.allclose(offsets_xy.to(torch.float32), restored, atol=atol, rtol=0.0):
        raise AssertionError("Point offset round-trip validation failed.")
