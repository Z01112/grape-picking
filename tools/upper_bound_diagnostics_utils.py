from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from grape_point_eval_utils import (
    _box_iou_matrix,
    collect_case_groups,
    compute_unified_point_metrics,
    match_prediction_record,
    normalize_prediction_record,
    safe_float,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "upper_bound_diagnostics_v1"
INDEX_DIR = REPO_ROOT / "outputs" / "_index"

TOP_ANCHOR_RATIO = 0.12
TOPROI_WIDTH_SCALE = 1.08
TOPROI_Y_MIN = -0.10
TOPROI_Y_MAX = 0.40
RELAXED_WIDTH_SCALE = 1.20
RELAXED_Y_MIN = -0.15
RELAXED_Y_MAX = 0.55


MODEL_SPECS = [
    {
        "name": "EMA_BIFPN",
        "priority": 1,
        "base_dir": REPO_ROOT / "outputs/03_unified_evaluation/eval_unification/ema_bifpn_unified_report",
    },
    {
        "name": "CADA_ADAPTER_ONLY_PROBE20",
        "priority": 2,
        "base_dir": REPO_ROOT
        / "outputs/01_mainline_results/candidate_cada_v1/ema_bifpn_cada_v1_adapter_only_probe20/report",
    },
    {
        "name": "CADA_FULL_FAIR100_FAILED",
        "priority": 3,
        "base_dir": REPO_ROOT
        / "outputs/05_failed_experiments/08_other_failed/ema_bifpn_cada_v1_adapter_only_full_fair100/report",
    },
    {
        "name": "V7_EXP2_FAIR",
        "priority": 4,
        "base_dir": REPO_ROOT / "outputs/03_unified_evaluation/eval_unification/v7_exp2_unified_report",
    },
    {
        "name": "YOLO11_POSE",
        "priority": 5,
        "base_dir": REPO_ROOT / "outputs/03_unified_evaluation/eval_unification/yolo_pose_unified_report",
    },
]


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        out.append("|" + "|".join(vals) + "|")
    return "\n".join(out) + "\n"


def load_records(path: Path) -> list[dict]:
    data = read_json(path, default=[])
    if isinstance(data, dict):
        data = data.get("records", data.get("predictions", []))
    if not isinstance(data, list):
        return []
    return [normalize_prediction_record(item) for item in data]


def discover_models() -> tuple[dict[str, dict], list[str]]:
    models: dict[str, dict] = {}
    missing: list[str] = []
    for spec in MODEL_SPECS:
        base = Path(spec["base_dir"])
        info = dict(spec)
        info["summary_path"] = base / "summary.json"
        info["test_records_path"] = base / "test_prediction_records.json"
        info["valid_records_path"] = base / "valid_prediction_records.json"
        info["train_records_path"] = base / "train_prediction_records.json"
        info["summary"] = read_json(info["summary_path"], default={})
        info["has_test_records"] = info["test_records_path"].exists()
        info["has_valid_records"] = info["valid_records_path"].exists()
        if not info["summary_path"].exists():
            missing.append(f"{spec['name']}: missing summary.json at {info['summary_path']}")
        if not info["has_test_records"]:
            missing.append(f"{spec['name']}: missing test_prediction_records.json at {info['test_records_path']}")
        models[spec["name"]] = info
    return models, missing


def visible_score(pred: dict) -> float:
    return safe_float(pred.get("visible_score", pred.get("has_picking_score", pred.get("raw_has_picking_score", 0.0))))


def l2(a: list[float], b: list[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def box_iou(a: list[float], b: list[float]) -> float:
    return float(_box_iou_matrix([a], [b])[0, 0].item())


def point_from_box_and_top_offset(box_xyxy: list[float], offset: list[float], top_anchor_ratio: float = TOP_ANCHOR_RATIO) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return [x1 + 0.5 * w + float(offset[0]) * w, y1 + top_anchor_ratio * h + float(offset[1]) * h]


def top_offset_from_box_and_point(box_xyxy: list[float], point: list[float], top_anchor_ratio: float = TOP_ANCHOR_RATIO) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return [(float(point[0]) - (x1 + 0.5 * w)) / w, (float(point[1]) - (y1 + top_anchor_ratio * h)) / h]


def rel_xy(box_xyxy: list[float], point: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return (float(point[0]) - x1) / w, (float(point[1]) - y1) / h


def top_roi_bounds(width_scale: float = TOPROI_WIDTH_SCALE, y_min: float = TOPROI_Y_MIN, y_max: float = TOPROI_Y_MAX) -> tuple[float, float, float, float]:
    x_min = 0.5 - 0.5 * float(width_scale)
    x_max = 0.5 + 0.5 * float(width_scale)
    return x_min, x_max, float(y_min), float(y_max)


def in_roi(relx: float, rely: float, width_scale: float = TOPROI_WIDTH_SCALE, y_min: float = TOPROI_Y_MIN, y_max: float = TOPROI_Y_MAX) -> bool:
    x_min, x_max, yy_min, yy_max = top_roi_bounds(width_scale, y_min, y_max)
    return x_min <= relx <= x_max and yy_min <= rely <= yy_max


def roi_boundary_distance(relx: float, rely: float, width_scale: float = TOPROI_WIDTH_SCALE, y_min: float = TOPROI_Y_MIN, y_max: float = TOPROI_Y_MAX) -> float:
    x_min, x_max, yy_min, yy_max = top_roi_bounds(width_scale, y_min, y_max)
    return min(abs(relx - x_min), abs(x_max - relx), abs(rely - yy_min), abs(yy_max - rely))


def area_groups(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    vals = sorted(values)
    q1 = vals[int((len(vals) - 1) / 3)]
    q2 = vals[int((len(vals) - 1) * 2 / 3)]
    return float(q1), float(q2)


def area_group(area: float, q1: float, q2: float) -> str:
    if area <= q1:
        return "small"
    if area <= q2:
        return "medium"
    return "large"


def summarize_values(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0}
    arr = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        pos = p * (len(arr) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return arr[lo]
        return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)
    return {
        "count": len(arr),
        "mean": float(mean(arr)),
        "median": float(median(arr)),
        "p10": q(0.10),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "min": arr[0],
        "max": arr[-1],
    }


def threshold_metrics(records: list[dict], threshold: float) -> dict:
    metrics = compute_unified_point_metrics(records, has_picking_threshold=threshold)
    pairs, _, _ = collect_case_groups(records, has_picking_threshold=threshold)
    l2_vals = [safe_float(c.get("l2_px", 0.0)) for c in pairs]
    inst = metrics["instance_chain"]
    glob = metrics["global_chain"]
    return {
        "threshold": float(threshold),
        "F1": inst["has_picking_f1"],
        "pair": inst["point_pair_count"],
        "global_visible_recall": glob["global_visible_recall"],
        "mean_L2": inst["point_mean_l2_px"],
        "median_L2": inst["point_median_l2_px"],
        "p90_L2": inst["point_p90_l2_px"],
        "PPL-SR@30": inst["ppl_sr_30"],
        "PPL-SR@50": inst["ppl_sr_50"],
        "L2>30_count": int(sum(v > 30.0 for v in l2_vals)),
        "L2>50_count": int(sum(v > 50.0 for v in l2_vals)),
    }


def flatten_visible_gts(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        for gi, gt in enumerate(rec.get("gt_instances", [])):
            if not bool(gt.get("has_picking", False)):
                continue
            rows.append({"record": rec, "gt_index": gi, "gt": gt})
    return rows


def current_cases_by_gt(records: list[dict], threshold: float = 0.5) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for rec in records:
        matched = match_prediction_record(rec, has_picking_threshold=threshold)
        for case in matched["matched_pairs"]:
            if bool(case.get("gt_has_picking", False)):
                out[(int(rec["image_id"]), int(case["gt_index"]))] = case
    return out


def collect_candidate_l2s(rec: dict, gt: dict, iou_min: float | None = None, visible_only: bool = False) -> list[dict]:
    rows = []
    gt_point = gt.get("picking_point", [0.0, 0.0])
    gt_box = gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    for pi, pred in enumerate(rec.get("pred_instances", [])):
        iou = box_iou(pred.get("bbox_xyxy", [0, 0, 0, 0]), gt_box)
        if iou_min is not None and iou < iou_min:
            continue
        if visible_only and visible_score(pred) < 0.5:
            continue
        rows.append(
            {
                "pred_index": pi,
                "iou": iou,
                "l2": l2(pred.get("picking_point", [0, 0]), gt_point),
                "score": safe_float(pred.get("score", 0.0)),
                "visible_score": visible_score(pred),
                "pred": pred,
            }
        )
    rows.sort(key=lambda x: x["l2"])
    return rows


def load_coco(split: str) -> dict:
    grape_point_path = REPO_ROOT / "dataset" / split / "_annotations.grape_point.json"
    coco_path = REPO_ROOT / "dataset" / split / "_annotations.coco.json"
    # The raw Roboflow COCO export is detection-only in this workspace.  The
    # picking point fields live in the derived grape_point annotation file, so
    # point diagnostics must prefer it while still accepting a point-rich COCO
    # file if one is introduced later.
    if grape_point_path.exists():
        return read_json(grape_point_path, default={})
    return read_json(coco_path, default={})


def coco_gt_rows(split: str) -> list[dict]:
    coco = load_coco(split)
    images = {int(img["id"]): img for img in coco.get("images", [])}
    rows = []
    for ann_idx, ann in enumerate(coco.get("annotations", [])):
        if safe_float(ann.get("has_picking", 0.0)) <= 0.5:
            continue
        image = images.get(int(ann.get("image_id", -1)), {})
        bbox = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = bbox
        point = ann.get("picking_point", [None, None])
        if not point or point[0] is None:
            continue
        rows.append(
            {
                "split": split,
                "image_id": int(ann.get("image_id", -1)),
                "file_name": image.get("file_name", ""),
                "ann_index": ann_idx,
                "bbox_xywh": bbox,
                "bbox_xyxy": [x, y, x + w, y + h],
                "area": safe_float(ann.get("area"), w * h),
                "picking_point": [float(point[0]), float(point[1])],
                "picking_bbox": ann.get("picking_bbox", ann.get("picking_box", "")),
            }
        )
    return rows


def band_for_rely(rely: float) -> str:
    if rely < 0.30:
        return "top"
    if rely <= 0.55:
        return "upper_middle"
    return "below_middle"


def counter_rows(counter: Counter, key_name: str = "group") -> list[dict]:
    total = sum(counter.values())
    return [
        {key_name: key, "count": count, "ratio": (count / total if total else 0.0)}
        for key, count in counter.most_common()
    ]
