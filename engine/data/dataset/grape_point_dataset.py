from __future__ import annotations

import torch

from .coco_dataset import CocoDetection
from ...core import register
from ...rtv4.point_utils import normalized_offsets_from_boxes_and_points


@register()
class GrapePointCocoDetection(CocoDetection):
    """Grape-only COCO dataset with optional per-instance picking point metadata.

    Expected annotation extras on each grape annotation:
    - has_picking: float in {0, 1}
    - picking_point: [x, y] in original image pixels
    - picking_offset: [dx, dy] normalized by grape bbox width/height

    The transform stack may ignore custom tensor fields, so this dataset keeps
    point supervision in augmentation-stable forms:
    - picking_offsets stay relative to the grape box
    - picking_points stay in original image coordinates for evaluation
    """

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        regenerate_point_offsets=False,
        point_offset_mode="center",
        point_top_anchor_ratio=0.0,
        point_anchor_x_ratio=0.5,
    ):
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
        )
        self.regenerate_point_offsets = bool(regenerate_point_offsets)
        self.point_offset_mode = str(point_offset_mode)
        self.point_top_anchor_ratio = float(point_top_anchor_ratio)
        self.point_anchor_x_ratio = float(point_anchor_x_ratio)

    def __getitem__(self, idx):
        image, target = super().load_item(idx)
        if self.regenerate_point_offsets:
            target = self._refresh_point_offsets(target)
        point_targets = self._pop_point_targets(target)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        target.update(point_targets)
        _, target = self._ensure_point_targets(image, target)
        return image, target

    def _refresh_point_offsets(self, target: dict) -> dict:
        boxes = target.get("boxes")
        points = target.get("picking_points")
        has_picking = target.get("has_picking")
        if boxes is None or points is None or has_picking is None:
            return target
        if boxes.numel() == 0:
            target["picking_offsets"] = torch.zeros((0, 2), dtype=torch.float32)
            return target

        boxes = boxes.to(dtype=torch.float32)
        points = points.to(dtype=torch.float32)
        has_picking = has_picking.to(dtype=torch.float32)

        boxes_cxcywh = torch.stack(
            (
                0.5 * (boxes[:, 0] + boxes[:, 2]),
                0.5 * (boxes[:, 1] + boxes[:, 3]),
                (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6),
                (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6),
            ),
            dim=-1,
        )
        offsets = normalized_offsets_from_boxes_and_points(
            boxes_cxcywh,
            points,
            mode=self.point_offset_mode,
            top_anchor_ratio=self.point_top_anchor_ratio,
            anchor_x_ratio=self.point_anchor_x_ratio,
        )
        offsets = torch.where((has_picking > 0.5).unsqueeze(-1), offsets, torch.zeros_like(offsets))
        target["picking_offsets"] = offsets
        return target

    @staticmethod
    def _pop_point_targets(target: dict):
        return {
            "has_picking": target.pop("has_picking", None),
            "has_stem": target.pop("has_stem", None),
            "picking_offsets": target.pop("picking_offsets", None),
            "picking_points": target.pop("picking_points", None),
        }

    @staticmethod
    def _ensure_point_targets(image, target: dict):
        num_instances = int(target.get("boxes", torch.zeros((0, 4))).shape[0])
        device = target["boxes"].device if "boxes" in target else torch.device("cpu")

        if "has_picking" not in target:
            target["has_picking"] = torch.zeros((num_instances,), dtype=torch.float32, device=device)
        elif target["has_picking"] is None:
            target["has_picking"] = torch.zeros((num_instances,), dtype=torch.float32, device=device)
        else:
            target["has_picking"] = target["has_picking"].to(dtype=torch.float32, device=device)

        if "has_stem" not in target:
            target["has_stem"] = torch.zeros((num_instances,), dtype=torch.float32, device=device)
        elif target["has_stem"] is None:
            target["has_stem"] = torch.zeros((num_instances,), dtype=torch.float32, device=device)
        else:
            target["has_stem"] = target["has_stem"].to(dtype=torch.float32, device=device)

        if "picking_offsets" not in target:
            target["picking_offsets"] = torch.zeros((num_instances, 2), dtype=torch.float32, device=device)
        elif target["picking_offsets"] is None:
            target["picking_offsets"] = torch.zeros((num_instances, 2), dtype=torch.float32, device=device)
        else:
            target["picking_offsets"] = target["picking_offsets"].to(dtype=torch.float32, device=device)

        if "picking_points" not in target:
            target["picking_points"] = torch.zeros((num_instances, 2), dtype=torch.float32, device=device)
        elif target["picking_points"] is None:
            target["picking_points"] = torch.zeros((num_instances, 2), dtype=torch.float32, device=device)
        else:
            target["picking_points"] = target["picking_points"].to(dtype=torch.float32, device=device)

        return image, target
