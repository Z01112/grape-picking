from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_ROOT = Path(r"D:\Projects\ultralytics-rtdetr")


@dataclass
class Instance:
    box_xyxy: np.ndarray
    keypoint_xy: np.ndarray
    visible: bool


@dataclass
class Prediction:
    box_xyxy: np.ndarray
    keypoint_xy: np.ndarray
    box_conf: float
    kpt_conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-Pose as a grape picking-point baseline.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_YOLO_ROOT
        / "runs/yolo-pose/grape_yolo_pose_b8_e100_20260531_125612/weights/best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_YOLO_ROOT / "dataset/grape_yolo_pose/data.yaml",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU used by YOLO prediction/val.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for instance-bound point pairing.")
    parser.add_argument("--pred-conf", type=float, default=0.001, help="Low YOLO predict conf; filtering is done later.")
    parser.add_argument("--box-conf-threshold", type=float, default=0.25)
    parser.add_argument("--kpt-conf-threshold", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--skip-yolo-val", action="store_true")
    parser.add_argument(
        "--main-summary",
        type=Path,
        default=REPO_ROOT / "outputs/03_global_analysis/post_cleanup_v7_exp2_report_20260525/summary.json",
    )
    parser.add_argument(
        "--ema-bifpn-summary",
        type=Path,
        default=REPO_ROOT
        / "outputs/02_encoder_experiments/encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526/report/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/07_external_baselines/yolo_pose_grape_point_{datetime.now():%Y%m%d_%H%M%S}",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def split_name(name: str) -> str:
    return "valid" if name == "val" else name


def image_files(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def xywhn_to_xyxy(values: list[float], width: int, height: int) -> np.ndarray:
    cx, cy, bw, bh = values
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def load_gt(label_path: Path, width: int, height: int) -> list[Instance]:
    if not label_path.exists():
        return []
    out: list[Instance] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        vals = [float(v) for v in parts]
        box = xywhn_to_xyxy(vals[1:5], width, height)
        kpt = np.asarray([vals[5] * width, vals[6] * height], dtype=np.float32)
        visible = vals[7] > 0.5
        out.append(Instance(box_xyxy=box, keypoint_xy=kpt, visible=visible))
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def summarize_errors(errors: list[dict[str, float]]) -> dict:
    if not errors:
        return {
            "pair_count": 0,
            "mean_l2_px": None,
            "median_l2_px": None,
            "p90_l2_px": None,
            "mae_x_px": None,
            "mae_y_px": None,
            "ppl_sr_30": None,
            "ppl_sr_50": None,
        }
    l2 = np.asarray([e["l2"] for e in errors], dtype=np.float32)
    dx = np.asarray([abs(e["dx"]) for e in errors], dtype=np.float32)
    dy = np.asarray([abs(e["dy"]) for e in errors], dtype=np.float32)
    return {
        "pair_count": int(len(errors)),
        "mean_l2_px": float(l2.mean()),
        "median_l2_px": float(np.median(l2)),
        "p90_l2_px": float(np.percentile(l2, 90)),
        "mae_x_px": float(dx.mean()),
        "mae_y_px": float(dy.mean()),
        "ppl_sr_30": float((l2 <= 30.0).mean()),
        "ppl_sr_50": float((l2 <= 50.0).mean()),
    }


def load_rt_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data["primary_checkpoint_split_summary"]["test"]
    det = test.get("grape_detection", {})
    has = test.get("has_picking", {})
    point = test.get("picking_point", {})
    return {
        "AP": safe_float(det.get("AP")),
        "AP50": safe_float(det.get("AP50")),
        "has_f1": safe_float(has.get("f1")),
        "pair_count": int(point.get("pair_count") or 0),
        "mean_l2_px": safe_float(point.get("mean_l2_px")),
        "ppl_sr_30": safe_float(point.get("ppl_sr_30")),
        "ppl_sr_50": safe_float(point.get("ppl_sr_50")),
    }


def yolo_val_metrics(model: YOLO, args: argparse.Namespace) -> dict:
    if args.skip_yolo_val:
        return {}
    metrics = model.val(
        data=str(args.data),
        split="val" if args.split == "valid" else args.split,
        imgsz=args.imgsz,
        device=args.device,
        batch=8,
        workers=0,
        iou=args.iou,
        max_det=args.max_det,
        project=str(args.output_dir),
        name="ultralytics_val",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    return {
        "box_AP": safe_float(getattr(metrics.box, "map", None)),
        "box_AP50": safe_float(getattr(metrics.box, "map50", None)),
        "pose_AP": safe_float(getattr(metrics.pose, "map", None)),
        "pose_AP50": safe_float(getattr(metrics.pose, "map50", None)),
    }


def predict_by_image(model: YOLO, image_dir: Path, args: argparse.Namespace) -> dict[str, list[Prediction]]:
    pred_map: dict[str, list[Prediction]] = {}
    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.pred_conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        project=str(args.output_dir),
        name="ultralytics_predict",
        exist_ok=True,
        stream=True,
        verbose=False,
    )
    for result in results:
        path = Path(result.path)
        preds: list[Prediction] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            if result.keypoints is not None:
                kpts_xy = result.keypoints.xy.detach().cpu().numpy()
                if result.keypoints.conf is not None:
                    kpts_conf = result.keypoints.conf.detach().cpu().numpy()
                else:
                    kpts_conf = np.ones((len(boxes), 1), dtype=np.float32)
            else:
                kpts_xy = np.zeros((len(boxes), 1, 2), dtype=np.float32)
                kpts_conf = np.zeros((len(boxes), 1), dtype=np.float32)
            for box, conf, kpt_xy, kpt_conf in zip(boxes, confs, kpts_xy, kpts_conf):
                preds.append(
                    Prediction(
                        box_xyxy=box.astype(np.float32),
                        keypoint_xy=kpt_xy[0].astype(np.float32),
                        box_conf=float(conf),
                        kpt_conf=float(kpt_conf[0]),
                    )
                )
        pred_map[path.name] = preds
    return pred_map


def evaluate_point_metrics(
    images: list[Path],
    label_dir: Path,
    pred_map: dict[str, list[Prediction]],
    args: argparse.Namespace,
) -> dict:
    visible_gt_count = 0
    gt_count = 0
    pred_has_count = 0
    false_positive = 0
    errors: list[dict[str, float]] = []
    case_rows: list[dict[str, Any]] = []

    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
        gt = load_gt(label_dir / f"{image_path.stem}.txt", width, height)
        gt_count += len(gt)
        visible_gt_count += sum(1 for item in gt if item.visible)
        matched_gt: set[int] = set()
        preds = [
            p
            for p in pred_map.get(image_path.name, [])
            if p.box_conf >= args.box_conf_threshold and p.kpt_conf >= args.kpt_conf_threshold
        ]
        preds.sort(key=lambda p: p.box_conf * p.kpt_conf, reverse=True)
        pred_has_count += len(preds)

        for pred in preds:
            best_idx = -1
            best_iou = 0.0
            for idx, target in enumerate(gt):
                if idx in matched_gt:
                    continue
                iou = box_iou(pred.box_xyxy, target.box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0 or best_iou < args.match_iou:
                false_positive += 1
                continue
            matched_gt.add(best_idx)
            target = gt[best_idx]
            if not target.visible:
                false_positive += 1
                continue
            delta = pred.keypoint_xy - target.keypoint_xy
            l2 = float(np.linalg.norm(delta))
            row = {
                "image": image_path.name,
                "iou": float(best_iou),
                "box_conf": float(pred.box_conf),
                "kpt_conf": float(pred.kpt_conf),
                "dx": float(delta[0]),
                "dy": float(delta[1]),
                "l2": l2,
            }
            errors.append(row)
            case_rows.append(row)

    true_positive = len(errors)
    false_negative = max(visible_gt_count - true_positive, 0)
    precision = true_positive / pred_has_count if pred_has_count else 0.0
    recall = true_positive / visible_gt_count if visible_gt_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    point = summarize_errors(errors)
    return {
        "gt_count": int(gt_count),
        "visible_gt_count": int(visible_gt_count),
        "predicted_has_count": int(pred_has_count),
        "has_picking": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
        },
        "picking_point": point,
        "cases": sorted(case_rows, key=lambda x: x["l2"], reverse=True)[:30],
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["model", "AP", "AP50", "has_F1", "pair", "mean_L2", "PPL-SR@30", "PPL-SR@50"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return "-"
            if key == "pair":
                return str(int(value))
            return f"{float(value):.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    fmt("AP"),
                    fmt("AP50"),
                    fmt("has_F1"),
                    fmt("pair"),
                    fmt("mean_L2"),
                    fmt("PPL-SR@30"),
                    fmt("PPL-SR@50"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.weights.exists():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"data yaml not found: {args.data}")

    data = load_yaml(args.data)
    root = Path(data.get("path", args.data.parent))
    split = split_name(args.split)
    image_rel = data["val" if split == "valid" else split]
    image_dir = root / image_rel
    label_dir = root / "labels" / split
    images = image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    model = YOLO(str(args.weights))
    val_metrics = yolo_val_metrics(model, args)
    pred_map = predict_by_image(model, image_dir, args)
    point_metrics = evaluate_point_metrics(images, label_dir, pred_map, args)

    yolo_row = {
        "model": f"YOLO11n-pose {split}",
        "AP": val_metrics.get("box_AP"),
        "AP50": val_metrics.get("box_AP50"),
        "has_F1": point_metrics["has_picking"]["f1"],
        "pair": point_metrics["picking_point"]["pair_count"],
        "mean_L2": point_metrics["picking_point"]["mean_l2_px"],
        "PPL-SR@30": point_metrics["picking_point"]["ppl_sr_30"],
        "PPL-SR@50": point_metrics["picking_point"]["ppl_sr_50"],
    }
    rows = []
    main_ref = load_rt_summary(args.main_summary)
    if main_ref is not None:
        rows.append(
            {
                "model": "V7_EXP2_MAIN fair retrain",
                "AP": main_ref["AP"],
                "AP50": main_ref["AP50"],
                "has_F1": main_ref["has_f1"],
                "pair": main_ref["pair_count"],
                "mean_L2": main_ref["mean_l2_px"],
                "PPL-SR@30": main_ref["ppl_sr_30"],
                "PPL-SR@50": main_ref["ppl_sr_50"],
            }
        )
    ema_ref = load_rt_summary(args.ema_bifpn_summary)
    if ema_ref is not None:
        rows.append(
            {
                "model": "EMA_BIFPN",
                "AP": ema_ref["AP"],
                "AP50": ema_ref["AP50"],
                "has_F1": ema_ref["has_f1"],
                "pair": ema_ref["pair_count"],
                "mean_L2": ema_ref["mean_l2_px"],
                "PPL-SR@30": ema_ref["ppl_sr_30"],
                "PPL-SR@50": ema_ref["ppl_sr_50"],
            }
        )
    rows.append(yolo_row)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "data": str(args.data),
        "split": split,
        "image_count": len(images),
        "thresholds": {
            "predict_conf": args.pred_conf,
            "box_conf": args.box_conf_threshold,
            "keypoint_conf": args.kpt_conf_threshold,
            "match_iou": args.match_iou,
            "nms_iou": args.iou,
        },
        "yolo_native_metrics": val_metrics,
        "yolo_picking_metrics": point_metrics,
        "comparison_rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# YOLO-Pose Grape Picking-Point Baseline",
        "",
        "This report converts YOLO-Pose predictions into the same instance-bound visible picking-point protocol used by GPPoint-DETR.",
        "",
        f"- split: `{split}`",
        f"- images: `{len(images)}`",
        f"- GT instances: `{point_metrics['gt_count']}`",
        f"- visible picking GT: `{point_metrics['visible_gt_count']}`",
        f"- box threshold: `{args.box_conf_threshold}`",
        f"- keypoint threshold: `{args.kpt_conf_threshold}`",
        f"- matching IoU: `{args.match_iou}`",
        "",
        "## Main Table",
        "",
        markdown_table(rows),
        "",
        "## YOLO Native Metrics",
        "",
        "These are Ultralytics native metrics and should not replace the picking-point table above.",
        "",
        "```json",
        json.dumps(val_metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Worst Paired Point Errors",
        "",
    ]
    for case in point_metrics["cases"][:10]:
        report.append(
            f"- `{case['image']}` L2={case['l2']:.2f}, dx={case['dx']:.2f}, dy={case['dy']:.2f}, "
            f"IoU={case['iou']:.3f}, box_conf={case['box_conf']:.3f}, kpt_conf={case['kpt_conf']:.3f}"
        )
    (args.output_dir / "comparison_report_zh.md").write_text("\n".join(report), encoding="utf-8")
    print(args.output_dir / "summary.json")
    print(args.output_dir / "comparison_report_zh.md")
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
