"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

from .rtv4 import RTv4

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .rtv4_criterion import RTv4Criterion
from .point_utils import (
    point_from_xywh_bbox,
    offset_from_xywh_bbox,
    absolute_points_from_boxes_and_offsets,
    normalized_offsets_from_boxes_and_points,
    assert_offset_roundtrip,
)

from .dinov3_teacher import DINOv3TeacherModel
