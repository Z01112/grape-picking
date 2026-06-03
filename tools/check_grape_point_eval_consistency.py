from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.rtv4.point_utils import absolute_points_from_boxes_and_offsets, normalized_offsets_from_boxes_and_points
from tools.grape_point_eval_utils import compute_unified_point_metrics


def xywh_to_cxcywh(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x + 0.5 * w, y + 0.5 * h, w, h]


def check_roundtrip(ann_path: Path) -> dict:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    boxes = []
    points = []
    for ann in data.get("annotations", []):
        if float(ann.get("has_picking", 0.0)) <= 0.5:
            continue
        point = ann.get("picking_point")
        if not point:
            continue
        boxes.append(xywh_to_cxcywh([float(v) for v in ann.get("bbox", [0.0, 0.0, 0.0, 0.0])]))
        points.append([float(point[0]), float(point[1])])
    if not boxes:
        return {"annotation_path": str(ann_path), "visible_points": 0, "max_roundtrip_error_px": 0.0}

    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    points_t = torch.as_tensor(points, dtype=torch.float32)
    offsets = normalized_offsets_from_boxes_and_points(
        boxes_t,
        points_t,
        mode="top_center",
        top_anchor_ratio=0.12,
        anchor_x_ratio=0.5,
    )
    decoded = absolute_points_from_boxes_and_offsets(
        boxes_t,
        offsets,
        mode="top_center",
        top_anchor_ratio=0.12,
        anchor_x_ratio=0.5,
    )
    errors = torch.linalg.norm(decoded - points_t, dim=-1)
    return {
        "annotation_path": str(ann_path),
        "visible_points": int(points_t.shape[0]),
        "max_roundtrip_error_px": float(errors.max().item()),
        "mean_roundtrip_error_px": float(errors.mean().item()),
    }


def check_synthetic_record() -> dict:
    record = {
        "image_id": 1,
        "file_name": "synthetic.jpg",
        "image_path": "synthetic.jpg",
        "width": 400,
        "height": 300,
        "gt_instances": [
            {"bbox_xyxy": [0, 0, 100, 100], "bbox_xywh": [0, 0, 100, 100], "area": 10000, "has_picking": True, "picking_point": [50, 20]},
            {"bbox_xyxy": [200, 0, 300, 100], "bbox_xywh": [200, 0, 100, 100], "area": 10000, "has_picking": True, "picking_point": [250, 20]},
            {"bbox_xyxy": [0, 150, 100, 250], "bbox_xywh": [0, 150, 100, 100], "area": 10000, "has_picking": False, "picking_point": [0, 0]},
        ],
        "pred_instances": [
            {"bbox_xyxy": [2, 2, 98, 98], "score": 0.99, "raw_has_picking_score": 0.9, "has_picking_score": 0.9, "visible_score": 0.9, "picking_point": [53, 24]},
            {"bbox_xyxy": [202, 2, 298, 98], "score": 0.95, "raw_has_picking_score": 0.1, "has_picking_score": 0.1, "visible_score": 0.1, "picking_point": [250, 20]},
            {"bbox_xyxy": [2, 152, 98, 248], "score": 0.90, "raw_has_picking_score": 0.8, "has_picking_score": 0.8, "visible_score": 0.8, "picking_point": [50, 180]},
            {"bbox_xyxy": [300, 200, 360, 260], "score": 0.70, "raw_has_picking_score": 0.7, "has_picking_score": 0.7, "visible_score": 0.7, "picking_point": [330, 230]},
        ],
    }
    metrics = compute_unified_point_metrics([record], iou_threshold=0.5, has_picking_threshold=0.5)
    instance = metrics["instance_chain"]
    global_chain = metrics["global_chain"]
    expected = {
        "visible_gt_total": 2,
        "matched_visible_grapes": 2,
        "correct_visible_grapes": 1,
        "has_picking_false_positive": 1,
        "has_picking_false_negative": 1,
        "point_pair_count": 1,
        "global_predicted_visible_total": 3,
    }
    observed = {
        "visible_gt_total": int(instance["visible_gt_total"]),
        "matched_visible_grapes": int(instance["matched_visible_grapes"]),
        "correct_visible_grapes": int(instance["correct_visible_grapes"]),
        "has_picking_false_positive": int(instance["has_picking_false_positive"]),
        "has_picking_false_negative": int(instance["has_picking_false_negative"]),
        "point_pair_count": int(instance["point_pair_count"]),
        "global_predicted_visible_total": int(global_chain["unmatched_or_unfiltered_predicted_visible_total"]),
    }
    passed = expected == observed and math.isclose(float(instance["point_mean_l2_px"]), 5.0, rel_tol=0.0, abs_tol=1e-6)
    return {"passed": passed, "expected": expected, "observed": observed, "metrics": metrics}


def main() -> int:
    ann_path = REPO_ROOT / "dataset" / "valid" / "_annotations.grape_point.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")
    payload = {
        "roundtrip": check_roundtrip(ann_path),
        "synthetic_record": check_synthetic_record(),
    }
    payload["passed"] = (
        payload["roundtrip"]["max_roundtrip_error_px"] <= 1e-4
        and bool(payload["synthetic_record"]["passed"])
    )
    out_path = REPO_ROOT / "outputs" / "debug" / "eval_consistency_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
