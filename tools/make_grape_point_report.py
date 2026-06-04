from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.misc import MetricLogger, dist_utils
from engine.rtv4.box_ops import box_iou
from engine.solver import TASKS
from tools.grape_point_eval_utils import (
    build_records_from_coco,
    compute_unified_point_metrics,
    prediction_to_instances,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_baseline_replay.yml"
DEFAULT_POINT_V2_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v6_baseline_replay" / "report" / "summary.json"
DEFAULT_BBOX_BASELINE_SUMMARY = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full report for grape-point experiments.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-cases", type=int, default=6)
    parser.add_argument("--point-v2-summary", "--baseline-summary", dest="point_v2_summary", type=Path, default=DEFAULT_POINT_V2_SUMMARY)
    parser.add_argument("--bbox-baseline-summary", type=Path, default=DEFAULT_BBOX_BASELINE_SUMMARY)
    parser.add_argument("--report-mode", default="point_full")
    parser.add_argument("--primary-label", default="point_run")
    parser.add_argument("--reference-label", default="baseline_replay")
    parser.add_argument("--report-title", default="grape point 实验中文结论")
    parser.add_argument("--change-note", action="append", default=None)
    parser.add_argument(
        "--save-prediction-records",
        action="store_true",
        help="Save train/valid/test prediction records for later no-checkpoint offline sweeps.",
    )
    return parser.parse_args()


def safe_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float(default)


def maybe_load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_point_config(config_path: Path) -> dict:
    payload = load_config(str(config_path))
    return payload if isinstance(payload, dict) else {}


def point_checkpoint_cfg(config_payload: dict) -> dict:
    cfg = config_payload.get("point_checkpointing", {})
    return cfg if isinstance(cfg, dict) else {}


def compute_composite_score(grape_ap: float, has_f1: float, point_l2_px: float, cfg: dict) -> float:
    alpha = safe_float(cfg.get("alpha", 0.35), 0.35)
    beta = safe_float(cfg.get("beta", 0.20), 0.20)
    norm_px = max(safe_float(cfg.get("point_error_norm_px", 40.0), 40.0), 1e-6)
    if not all(math.isfinite(v) for v in (grape_ap, has_f1, point_l2_px)):
        return float("nan")
    return grape_ap + alpha * has_f1 - beta * (point_l2_px / norm_px)


def summarize_dataset(ann_path: Path) -> dict:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", [])
    visible = sum(1 for ann in annotations if safe_float(ann.get("has_picking", 0.0)) > 0.5)
    return {
        "images": len(payload.get("images", [])),
        "grape_annotations": len(annotations),
        "grapes_with_picking": visible,
        "grapes_without_picking": len(annotations) - visible,
    }


def load_log_records(log_path: Path) -> list[dict]:
    by_epoch: dict[int, dict] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        by_epoch[int(payload["epoch"])] = payload
    return [by_epoch[k] for k in sorted(by_epoch.keys())]


def build_epoch_rows(records: list[dict], cfg: dict) -> list[dict]:
    rows = []
    for rec in records:
        bbox = rec.get("test_coco_eval_bbox", [])
        per_class = rec.get("test_coco_eval_bbox_per_class", {}).get("grape", {})
        point = rec.get("test_grape_point_metrics", {})
        point_ckpt = rec.get("test_point_checkpoint_metrics", {})

        grape_ap = safe_float(point_ckpt.get("valid_grape_AP"), safe_float(per_class.get("AP"), bbox[0] if len(bbox) > 0 else 0.0))
        grape_ap50 = safe_float(point_ckpt.get("valid_grape_AP50"), safe_float(per_class.get("AP50"), bbox[1] if len(bbox) > 1 else 0.0))
        grape_ar100 = safe_float(point_ckpt.get("valid_grape_AR100"), safe_float(per_class.get("AR100"), bbox[8] if len(bbox) > 8 else 0.0))
        has_precision = safe_float(point_ckpt.get("valid_has_picking_precision"), point.get("has_picking_precision"))
        has_recall = safe_float(point_ckpt.get("valid_has_picking_recall"), point.get("has_picking_recall"))
        has_f1 = safe_float(point_ckpt.get("valid_has_picking_F1"), point.get("has_picking_f1"))
        point_mae_x = safe_float(point_ckpt.get("valid_point_MAE_x_px"), point.get("point_mae_x_px"))
        point_mae_y = safe_float(point_ckpt.get("valid_point_MAE_y_px"), point.get("point_mae_y_px"))
        point_l2 = safe_float(point_ckpt.get("valid_point_mean_L2_px"), point.get("point_mean_l2_px"))
        composite = safe_float(point_ckpt.get("composite_score"), compute_composite_score(grape_ap, has_f1, point_l2, cfg))

        rows.append(
            {
                "epoch": int(rec["epoch"]),
                "train_lr": safe_float(rec.get("train_lr")),
                "train_loss": safe_float(rec.get("train_loss")),
                "train_loss_bbox": safe_float(rec.get("train_loss_bbox")),
                "train_loss_giou": safe_float(rec.get("train_loss_giou")),
                "train_loss_fgl": safe_float(rec.get("train_loss_fgl")),
                "train_loss_ddf": safe_float(rec.get("train_loss_ddf")),
                "train_loss_has_picking": safe_float(rec.get("train_loss_has_picking")),
                "train_loss_picking_offset": safe_float(rec.get("train_loss_picking_offset")),
                "train_loss_dense_has_picking": safe_float(rec.get("train_loss_dense_has_picking")),
                "train_loss_dense_picking_offset": safe_float(rec.get("train_loss_dense_picking_offset")),
                "train_loss_picking_geo": safe_float(rec.get("train_loss_picking_geo")),
                "valid_grape_AP": grape_ap,
                "valid_grape_AP50": grape_ap50,
                "valid_grape_AR100": grape_ar100,
                "valid_has_picking_precision": has_precision,
                "valid_has_picking_recall": has_recall,
                "valid_has_picking_f1": has_f1,
                "valid_has_picking_false_positive": int(point.get("has_picking_false_positive", 0)),
                "valid_has_picking_false_negative": int(point.get("has_picking_false_negative", 0)),
                "valid_point_pair_count": int(point.get("point_pair_count", 0)),
                "valid_point_mae_x_px": point_mae_x,
                "valid_point_mae_y_px": point_mae_y,
                "valid_point_mean_l2_px": point_l2,
                "composite_score": composite,
            }
        )
    return rows


def best_epoch(rows: list[dict], key: str, mode: str = "max") -> dict | None:
    valid_rows = [row for row in rows if math.isfinite(safe_float(row.get(key), float("nan")))]
    if not valid_rows:
        return None
    if mode == "min":
        return min(valid_rows, key=lambda item: safe_float(item.get(key), float("inf")))
    return max(valid_rows, key=lambda item: safe_float(item.get(key), float("-inf")))


def tail_mean_std(rows: list[dict], key: str, window: int = 10) -> dict:
    tail = rows[-window:] if len(rows) >= window else rows
    values = np.asarray([safe_float(item.get(key), float("nan")) for item in tail], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def write_results_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(rows: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in rows], label="train_loss", color="#1f77b4", linewidth=2)
    axes[0, 0].plot(epochs, [row["train_loss_bbox"] for row in rows], label="bbox_loss", color="#ff7f0e")
    axes[0, 0].plot(epochs, [row["train_loss_giou"] for row in rows], label="giou_loss", color="#2ca02c")
    axes[0, 0].set_title("Train Total / BBox Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["train_loss_has_picking"] for row in rows], label="has_picking_loss", color="#9467bd")
    axes[0, 1].plot(epochs, [row["train_loss_picking_offset"] for row in rows], label="point_offset_loss", color="#d62728")
    if any(abs(safe_float(row.get("train_loss_dense_has_picking"))) > 1e-12 for row in rows):
        axes[0, 1].plot(epochs, [row["train_loss_dense_has_picking"] for row in rows], label="dense_has_loss", color="#17becf")
    if any(abs(safe_float(row.get("train_loss_dense_picking_offset"))) > 1e-12 for row in rows):
        axes[0, 1].plot(epochs, [row["train_loss_dense_picking_offset"] for row in rows], label="dense_point_loss", color="#bcbd22")
    if any(abs(safe_float(row.get("train_loss_picking_geo"))) > 1e-12 for row in rows):
        axes[0, 1].plot(epochs, [row["train_loss_picking_geo"] for row in rows], label="geo_loss", color="#ff9896")
    axes[0, 1].plot(epochs, [row["train_loss_fgl"] for row in rows], label="fgl_loss", color="#8c564b")
    axes[0, 1].plot(epochs, [row["train_loss_ddf"] for row in rows], label="ddf_loss", color="#7f7f7f")
    axes[0, 1].set_title("Point Branch / Localization Loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["valid_grape_AP"] for row in rows], label="grape AP", color="#1f77b4", linewidth=2)
    axes[1, 0].plot(epochs, [row["valid_grape_AP50"] for row in rows], label="grape AP50", color="#17becf", linewidth=2)
    axes[1, 0].plot(epochs, [row["valid_grape_AR100"] for row in rows], label="grape AR100", color="#2ca02c", linewidth=2)
    axes[1, 0].set_title("Valid Grape Detection")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [row["valid_has_picking_precision"] for row in rows], label="precision", color="#9467bd")
    axes[1, 1].plot(epochs, [row["valid_has_picking_recall"] for row in rows], label="recall", color="#d62728")
    axes[1, 1].plot(epochs, [row["valid_has_picking_f1"] for row in rows], label="F1", color="#2ca02c", linewidth=2)
    axes[1, 1].set_title("Valid has_picking")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(epochs, [row["valid_point_mean_l2_px"] for row in rows], label="mean L2 px", color="#8c564b", linewidth=2)
    axes[2, 0].plot(epochs, [row["valid_point_mae_x_px"] for row in rows], label="MAE x", color="#e377c2")
    axes[2, 0].plot(epochs, [row["valid_point_mae_y_px"] for row in rows], label="MAE y", color="#7f7f7f")
    axes[2, 0].set_title("Valid Point Error")
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(epochs, [row["composite_score"] for row in rows], label="composite_score", color="#ff7f0e", linewidth=2)
    axes[2, 1].set_title("Valid Composite Score")
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def best_point_l2_state_summary(checkpoint_state: dict | None) -> dict:
    state = checkpoint_state.get("best_point_l2", {}) if isinstance(checkpoint_state, dict) else {}
    metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
    pair_count = int(metrics.get("valid_point_pair_count", 0) or 0)
    epoch = int(state.get("epoch", -1) or -1)
    score = safe_float(state.get("score"), float("nan"))
    is_valid = epoch >= 0 and pair_count > 0 and math.isfinite(score)
    return {
        "best_point_l2_valid_epoch": epoch if is_valid else -1,
        "best_point_l2_pair_count": pair_count,
        "whether_best_point_l2_is_valid": bool(is_valid),
        "best_point_l2_score": score if is_valid else float("nan"),
    }


def find_checkpoint_paths(run_dir: Path, checkpoint_state: dict | None = None) -> dict[str, Path]:
    candidates = {}
    point_l2_state = best_point_l2_state_summary(checkpoint_state)
    for name in (
        "best_composite.pth",
        "best_point_l2.pth",
        "best_has_picking_f1.pth",
        "best_grape_ap.pth",
        "best_stg2.pth",
        "best_stg1.pth",
        "last.pth",
    ):
        if name == "best_point_l2.pth" and not point_l2_state["whether_best_point_l2_is_valid"]:
            continue
        for path in (run_dir / name, run_dir / "checkpoints" / name):
            if path.exists():
                candidates[name] = path.resolve()
                break
    return candidates


def select_primary_checkpoint(run_dir: Path, explicit: Path | None, checkpoint_state: dict | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = find_checkpoint_paths(run_dir, checkpoint_state=checkpoint_state)
    for name in (
        "best_composite.pth",
        "best_point_l2.pth",
        "best_has_picking_f1.pth",
        "best_grape_ap.pth",
        "best_stg2.pth",
        "best_stg1.pth",
        "last.pth",
    ):
        if name in candidates:
            return candidates[name]
    raise FileNotFoundError(f"No checkpoint was found in {run_dir}")


def flatten_split_metrics(stats: dict, cfg: dict) -> dict:
    grape = stats.get("coco_eval_bbox_per_class", {}).get("grape", {})
    point = stats.get("grape_point_metrics", {})
    bbox = stats.get("coco_eval_bbox", [])
    grape_ap = safe_float(grape.get("AP"), bbox[0] if len(bbox) > 0 else 0.0)
    grape_ap50 = safe_float(grape.get("AP50"), bbox[1] if len(bbox) > 1 else 0.0)
    grape_ar100 = safe_float(grape.get("AR100"), bbox[8] if len(bbox) > 8 else 0.0)
    has_f1 = safe_float(point.get("has_picking_f1"))
    point_l2 = safe_float(point.get("point_mean_l2_px"))
    return {
        "grape_AP": grape_ap,
        "grape_AP50": grape_ap50,
        "grape_AR100": grape_ar100,
        "has_picking_precision": safe_float(point.get("has_picking_precision")),
        "has_picking_recall": safe_float(point.get("has_picking_recall")),
        "has_picking_f1": has_f1,
        "has_picking_false_positive": int(point.get("has_picking_false_positive", 0)),
        "has_picking_false_negative": int(point.get("has_picking_false_negative", 0)),
        "point_pair_count": int(point.get("point_pair_count", 0)),
        "point_mae_x_px": safe_float(point.get("point_mae_x_px")),
        "point_mae_y_px": safe_float(point.get("point_mae_y_px")),
        "point_mean_l2_px": point_l2,
        "composite_score": compute_composite_score(grape_ap, has_f1, point_l2, cfg),
    }


def structured_split_summary(stats: dict, cfg: dict, error_summary: dict | None = None) -> dict:
    flat = flatten_split_metrics(stats, cfg)
    error_summary = error_summary or {}
    return {
        "grape_detection": {
            "AP": flat["grape_AP"],
            "AP50": flat["grape_AP50"],
            "AR100": flat["grape_AR100"],
        },
        "has_picking": {
            "precision": flat["has_picking_precision"],
            "recall": flat["has_picking_recall"],
            "f1": flat["has_picking_f1"],
            "false_positive": flat["has_picking_false_positive"],
            "false_negative": flat["has_picking_false_negative"],
        },
        "picking_point": {
            "pair_count": flat["point_pair_count"],
            "mae_x_px": flat["point_mae_x_px"],
            "mae_y_px": flat["point_mae_y_px"],
            "mean_l2_px": flat["point_mean_l2_px"],
            "median_l2_px": safe_float(error_summary.get("median_l2_px"), float("nan")),
            "p90_l2_px": safe_float(error_summary.get("p90_l2_px"), float("nan")),
            "ppl_sr_30": safe_float(error_summary.get("ppl_sr_30"), float("nan")),
            "ppl_sr_50": safe_float(error_summary.get("ppl_sr_50"), float("nan")),
            "mean_abs_dx_px": safe_float(error_summary.get("mean_abs_dx_px"), float("nan")),
            "mean_abs_dy_px": safe_float(error_summary.get("mean_abs_dy_px"), float("nan")),
            "size_group_l2_px": error_summary.get("size_group_l2_px", {}),
            "quality_aligned": error_summary.get("quality_aligned"),
        },
        "composite_score": flat["composite_score"],
    }


def _build_gt_by_image(coco_dataset: dict, split_dir: Path) -> dict[int, dict]:
    images = {
        int(image["id"]): {
            "image_id": int(image["id"]),
            "file_name": image.get("file_name", f"{int(image['id'])}.jpg"),
            "width": int(image.get("width", 0)),
            "height": int(image.get("height", 0)),
            "image_path": str((split_dir / image.get("file_name", "")).resolve()),
            "gt_instances": [],
            "pred_instances": [],
        }
        for image in coco_dataset.get("images", [])
    }
    for ann in coco_dataset.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in images:
            continue
        x, y, w, h = [float(v) for v in ann.get("bbox", [0.0, 0.0, 0.0, 0.0])]
        point = ann.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
        images[image_id]["gt_instances"].append(
            {
                "bbox_xyxy": [x, y, x + w, y + h],
                "bbox_xywh": [x, y, w, h],
                "area": float(w * h),
                "has_picking": bool(safe_float(ann.get("has_picking", 0.0)) > 0.5),
                "picking_point": [float(point[0]), float(point[1])],
            }
        )
    return images


def _prediction_to_instances(prediction: dict) -> list[dict]:
    boxes = prediction.get("boxes", torch.zeros((0, 4))).detach().cpu().to(torch.float32)
    scores = prediction.get("scores", torch.zeros((boxes.shape[0],))).detach().cpu().to(torch.float32)
    labels = prediction.get("labels", torch.zeros((boxes.shape[0],), dtype=torch.int64)).detach().cpu()
    has_scores = prediction.get("has_picking_scores")
    raw_has_scores = prediction.get("raw_has_picking_scores")
    visible_scores = prediction.get("visible_scores")
    quality_scores = prediction.get("point_quality_scores")
    final_scores = prediction.get("point_final_scores")
    selector_scores = prediction.get("point_selector_scores")
    selector_final_scores = prediction.get("point_selector_final_scores")
    accept_scores = prediction.get("point_accept_scores")
    accept_final_scores = prediction.get("point_accept_final_scores")
    reliability_scores = prediction.get("point_reliability_scores")
    reliability_final_scores = prediction.get("point_reliability_final_scores")
    weak_heatmap_scores = prediction.get("weak_heatmap_scores")
    raw_offsets = prediction.get("raw_picking_offsets")
    dpo_offsets = prediction.get("dpo_picking_offsets")
    dpo_blend_offsets = prediction.get("dpo_blend_picking_offsets")
    dpo_entropy_x = prediction.get("dpo_entropy_x")
    dpo_entropy_y = prediction.get("dpo_entropy_y")
    dpo_maxprob_x = prediction.get("dpo_maxprob_x")
    dpo_maxprob_y = prediction.get("dpo_maxprob_y")
    if has_scores is None:
        has_scores = torch.zeros((boxes.shape[0],), dtype=torch.float32)
    else:
        has_scores = has_scores.detach().cpu().to(torch.float32)
    if raw_has_scores is None:
        raw_has_scores = has_scores
    else:
        raw_has_scores = raw_has_scores.detach().cpu().to(torch.float32)
    if visible_scores is None:
        visible_scores = has_scores
    else:
        visible_scores = visible_scores.detach().cpu().to(torch.float32)
    if quality_scores is not None:
        quality_scores = quality_scores.detach().cpu().to(torch.float32)
    if final_scores is not None:
        final_scores = final_scores.detach().cpu().to(torch.float32)
    if selector_scores is not None:
        selector_scores = selector_scores.detach().cpu().to(torch.float32)
    if selector_final_scores is not None:
        selector_final_scores = selector_final_scores.detach().cpu().to(torch.float32)
    if accept_scores is not None:
        accept_scores = accept_scores.detach().cpu().to(torch.float32)
    if accept_final_scores is not None:
        accept_final_scores = accept_final_scores.detach().cpu().to(torch.float32)
    if reliability_scores is not None:
        reliability_scores = reliability_scores.detach().cpu().to(torch.float32)
    if reliability_final_scores is not None:
        reliability_final_scores = reliability_final_scores.detach().cpu().to(torch.float32)
    if weak_heatmap_scores is not None:
        weak_heatmap_scores = weak_heatmap_scores.detach().cpu().to(torch.float32)
    if raw_offsets is not None:
        raw_offsets = raw_offsets.detach().cpu().to(torch.float32)
    if dpo_offsets is not None:
        dpo_offsets = dpo_offsets.detach().cpu().to(torch.float32)
    if dpo_blend_offsets is not None:
        dpo_blend_offsets = dpo_blend_offsets.detach().cpu().to(torch.float32)
    if dpo_entropy_x is not None:
        dpo_entropy_x = dpo_entropy_x.detach().cpu().to(torch.float32)
    if dpo_entropy_y is not None:
        dpo_entropy_y = dpo_entropy_y.detach().cpu().to(torch.float32)
    if dpo_maxprob_x is not None:
        dpo_maxprob_x = dpo_maxprob_x.detach().cpu().to(torch.float32)
    if dpo_maxprob_y is not None:
        dpo_maxprob_y = dpo_maxprob_y.detach().cpu().to(torch.float32)
    points = prediction.get("picking_points")
    if points is None:
        points = torch.zeros((boxes.shape[0], 2), dtype=torch.float32)
    else:
        points = points.detach().cpu().to(torch.float32)

    instances = []
    for idx in range(boxes.shape[0]):
        item = {
            "bbox_xyxy": [float(v) for v in boxes[idx].tolist()],
            "score": float(scores[idx].item()),
            "label": int(labels[idx].item()),
            "raw_has_picking_score": float(raw_has_scores[idx].item()),
            "has_picking_score": float(has_scores[idx].item()),
            "visible_score": float(visible_scores[idx].item()),
            "picking_point": [float(v) for v in points[idx].tolist()],
        }
        if quality_scores is not None:
            item["point_quality_score"] = float(quality_scores[idx].item())
        if final_scores is not None:
            item["point_final_score"] = float(final_scores[idx].item())
        if selector_scores is not None:
            item["point_selector_score"] = float(selector_scores[idx].item())
        if selector_final_scores is not None:
            item["point_selector_final_score"] = float(selector_final_scores[idx].item())
        if accept_scores is not None:
            item["point_accept_score"] = float(accept_scores[idx].item())
        if accept_final_scores is not None:
            item["point_accept_final_score"] = float(accept_final_scores[idx].item())
        if reliability_scores is not None:
            item["point_reliability_score"] = float(reliability_scores[idx].item())
        if reliability_final_scores is not None:
            item["point_reliability_final_score"] = float(reliability_final_scores[idx].item())
        if weak_heatmap_scores is not None:
            item["weak_heatmap_score"] = float(weak_heatmap_scores[idx].item())
        if raw_offsets is not None:
            item["raw_picking_offset"] = [float(v) for v in raw_offsets[idx].tolist()]
        if dpo_offsets is not None:
            item["dpo_picking_offset"] = [float(v) for v in dpo_offsets[idx].tolist()]
        if dpo_blend_offsets is not None:
            item["dpo_blend_picking_offset"] = [float(v) for v in dpo_blend_offsets[idx].tolist()]
        if dpo_entropy_x is not None:
            item["dpo_entropy_x"] = float(dpo_entropy_x[idx].item())
        if dpo_entropy_y is not None:
            item["dpo_entropy_y"] = float(dpo_entropy_y[idx].item())
        if dpo_maxprob_x is not None:
            item["dpo_maxprob_x"] = float(dpo_maxprob_x[idx].item())
        if dpo_maxprob_y is not None:
            item["dpo_maxprob_y"] = float(dpo_maxprob_y[idx].item())
        instances.append(item)
    return instances


@torch.no_grad()
def evaluate_split(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    collect_predictions: bool = False,
) -> tuple[dict, list[dict]]:
    dist_utils.setup_distributed(seed=0)
    try:
        with tempfile.TemporaryDirectory(prefix="point_v2_report_", dir=str(REPO_ROOT / "outputs")) as tmp_dir:
            cfg = YAMLConfig(
                str(config_path),
                resume=str(checkpoint_path),
                device=device,
                use_amp=False,
                output_dir=tmp_dir,
            )
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            split_dir = (dataset_root / split).resolve()
            ann_path = split_dir / "_annotations.grape_point.json"
            cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(split_dir.as_posix())
            cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = str(ann_path.as_posix())
            cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = batch_size
            cfg.yaml_cfg["val_dataloader"]["num_workers"] = num_workers

            solver = TASKS[cfg.yaml_cfg["task"]](cfg)
            solver.eval()
            model = solver.ema.module if solver.ema else solver.model
            criterion = solver.criterion
            postprocessor = solver.postprocessor
            data_loader = solver.val_dataloader
            evaluator = solver.evaluator
            eval_device = solver.device

            model.eval()
            criterion.eval()
            evaluator.cleanup()
            metric_logger = MetricLogger(delimiter="  ")
            header = f"Eval-{split}:"
            records_by_image = build_records_from_coco(evaluator.coco_gt.dataset, split_dir)

            for samples, targets in metric_logger.log_every(data_loader, 10, header):
                samples = samples.to(eval_device)
                targets = [{k: v.to(eval_device) for k, v in t.items()} for t in targets]
                outputs = model(samples)
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessor(outputs, orig_target_sizes)
                results_map = {int(target["image_id"].item()): output for target, output in zip(targets, results)}
                evaluator.update(results_map)

                if collect_predictions:
                    for image_id, output in results_map.items():
                        if image_id in records_by_image:
                            records_by_image[image_id]["pred_instances"] = prediction_to_instances(output)

            metric_logger.synchronize_between_processes()
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()

            stats = {}
            stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            stats["coco_eval_bbox_per_class"] = evaluator.get_per_class_metrics("bbox")
            stats.update(evaluator.get_extra_metrics())
            solver.cleanup()
    finally:
        dist_utils.cleanup()

    records = [records_by_image[k] for k in sorted(records_by_image.keys())] if collect_predictions else []
    return stats, records


def match_prediction_record(
    record: dict,
    iou_threshold: float,
    has_picking_threshold: float,
    visibility_score_key: str = "has_picking_score",
) -> dict:
    gt_entries = record.get("gt_instances", [])
    pred_entries = record.get("pred_instances", [])
    output = {
        "correct_visible_pairs": [],
        "has_fp_pairs": [],
        "has_fn_pairs": [],
        "matched_pairs": [],
    }
    if not gt_entries or not pred_entries:
        return output

    gt_boxes = torch.as_tensor([item["bbox_xyxy"] for item in gt_entries], dtype=torch.float32)
    pred_boxes = torch.as_tensor([item["bbox_xyxy"] for item in pred_entries], dtype=torch.float32)
    pred_scores = torch.as_tensor([item["score"] for item in pred_entries], dtype=torch.float32)
    ious, _ = box_iou(pred_boxes, gt_boxes)
    pred_order = torch.argsort(pred_scores, descending=True)
    used_gt = set()

    for pred_idx in pred_order.tolist():
        best_gt = None
        best_iou = -1.0
        for gt_idx in range(len(gt_entries)):
            if gt_idx in used_gt:
                continue
            iou = float(ious[pred_idx, gt_idx].item())
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt is None or best_iou < iou_threshold:
            continue
        used_gt.add(best_gt)
        gt = gt_entries[best_gt]
        pred = pred_entries[pred_idx]
        gt_visible = bool(gt["has_picking"])
        pred_visible = bool(float(pred.get(visibility_score_key, 0.0)) >= has_picking_threshold)
        case = _build_point_case(record, best_gt, gt, pred_idx, pred, best_iou)
        case.update(
            {
                "gt_has_picking": gt_visible,
                "pred_has_picking": pred_visible,
            }
        )

        if gt_visible and pred_visible:
            output["correct_visible_pairs"].append(case)
        elif (not gt_visible) and pred_visible:
            output["has_fp_pairs"].append(case)
        elif gt_visible and (not pred_visible):
            output["has_fn_pairs"].append(case)

        output["matched_pairs"].append(case)

    return output


def collect_case_groups(
    records: list[dict],
    iou_threshold: float,
    has_picking_threshold: float,
    visibility_score_key: str = "has_picking_score",
) -> tuple[list[dict], list[dict], list[dict]]:
    correct_pairs, fp_pairs, fn_pairs = [], [], []
    for record in records:
        matched = match_prediction_record(record, iou_threshold, has_picking_threshold, visibility_score_key=visibility_score_key)
        correct_pairs.extend(matched["correct_visible_pairs"])
        fp_pairs.extend(matched["has_fp_pairs"])
        fn_pairs.extend(matched["has_fn_pairs"])
    return correct_pairs, fp_pairs, fn_pairs


def _point_in_box(point_xy: list[float], box_xyxy: list[float], margin: float = 0.0) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _build_point_case(record: dict, gt_idx: int, gt: dict, pred_idx: int, pred: dict, iou: float) -> dict:
    gt_point = gt.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    pred_point = pred.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    gt_point = [float(gt_point[0]), float(gt_point[1])]
    pred_point = [float(pred_point[0]), float(pred_point[1])]
    pred_box = [float(v) for v in pred.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])]
    dx = float(pred_point[0] - gt_point[0])
    dy = float(pred_point[1] - gt_point[1])
    return {
        "image_id": int(record["image_id"]),
        "file_name": record["file_name"],
        "image_path": record["image_path"],
        "iou": float(iou),
        "gt_index": int(gt_idx),
        "pred_index": int(pred_idx),
        "gt_bbox_xyxy": [float(v) for v in gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])],
        "pred_bbox_xyxy": pred_box,
        "gt_area": float(gt.get("area", 0.0)),
        "gt_point": gt_point,
        "pred_point": pred_point,
        "pred_has_picking_score": float(pred.get("has_picking_score", 0.0)),
        "pred_point_quality_score": float(pred.get("point_quality_score", 0.0)),
        "pred_point_final_score": float(pred.get("point_final_score", 0.0)),
        "pred_point_accept_score": float(pred.get("point_accept_score", 0.0)),
        "pred_point_accept_final_score": float(pred.get("point_accept_final_score", 0.0)),
        "pred_point_reliability_score": float(pred.get("point_reliability_score", 0.0)),
        "pred_point_reliability_final_score": float(pred.get("point_reliability_final_score", 0.0)),
        "pred_weak_heatmap_score": float(pred.get("weak_heatmap_score", 0.0)),
        "pred_score": float(pred.get("score", 0.0)),
        "dx_px": dx,
        "dy_px": dy,
        "l2_px": float(math.hypot(dx, dy)),
        "pred_point_inside_gt_box": bool(_point_in_box(pred_point, gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]), margin=0.0)),
        "pred_point_inside_pred_box": bool(_point_in_box(pred_point, pred_box, margin=0.0)),
        "gt_point_inside_pred_box": bool(_point_in_box(gt_point, pred_box, margin=0.0)),
    }


def _summarize_point_cases(cases: list[dict], visible_gt_count: int | None = None) -> dict:
    l2_values = [float(item.get("l2_px", 0.0)) for item in cases if "l2_px" in item]
    dx_values = [float(item.get("dx_px", 0.0)) for item in cases if "dx_px" in item]
    dy_values = [float(item.get("dy_px", 0.0)) for item in cases if "dy_px" in item]
    iou_values = [float(item.get("iou", 0.0)) for item in cases if "iou" in item]
    pred_point_inside_gt = [1.0 if bool(item.get("pred_point_inside_gt_box", False)) else 0.0 for item in cases]
    pred_point_inside_pred = [1.0 if bool(item.get("pred_point_inside_pred_box", False)) else 0.0 for item in cases]
    gt_point_inside_pred = [1.0 if bool(item.get("gt_point_inside_pred_box", False)) else 0.0 for item in cases]

    summary = {
        "count": len(cases),
        "mean_l2_px": float(np.mean(l2_values)) if l2_values else 0.0,
        "median_l2_px": float(np.median(l2_values)) if l2_values else 0.0,
        "p90_l2_px": float(np.quantile(np.asarray(l2_values), 0.90)) if l2_values else 0.0,
        "mean_abs_dx_px": float(np.mean(np.abs(dx_values))) if dx_values else 0.0,
        "mean_abs_dy_px": float(np.mean(np.abs(dy_values))) if dy_values else 0.0,
        "mean_iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "pred_point_inside_gt_box_rate": float(np.mean(pred_point_inside_gt)) if pred_point_inside_gt else 0.0,
        "pred_point_inside_pred_box_rate": float(np.mean(pred_point_inside_pred)) if pred_point_inside_pred else 0.0,
        "gt_point_inside_pred_box_rate": float(np.mean(gt_point_inside_pred)) if gt_point_inside_pred else 0.0,
    }
    if visible_gt_count is not None:
        summary["visible_gt_count"] = int(visible_gt_count)
        summary["candidate_recall"] = float(len(cases) / visible_gt_count) if visible_gt_count > 0 else 0.0
    return summary


def collect_cross_instance_mismatch_cases(records: list[dict], correct_pairs: list[dict]) -> list[dict]:
    record_lookup = {int(record["image_id"]): record for record in records}
    mismatches = []

    for case in correct_pairs:
        own_l2 = float(case.get("l2_px", 0.0))
        if own_l2 < 6.0:
            continue

        record = record_lookup.get(int(case["image_id"]))
        if record is None:
            continue

        pred_point = case.get("pred_point", [0.0, 0.0])
        best_other = None
        for other_idx, other_gt in enumerate(record.get("gt_instances", [])):
            if other_idx == int(case["gt_index"]):
                continue
            if not bool(other_gt.get("has_picking")):
                continue

            other_point = other_gt.get("picking_point", [0.0, 0.0])
            other_dist = float(math.hypot(pred_point[0] - other_point[0], pred_point[1] - other_point[1]))
            in_other_box = _point_in_box(pred_point, other_gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]), margin=2.0)
            score = own_l2 - other_dist + (8.0 if in_other_box else 0.0)
            candidate = {
                "other_gt_index": int(other_idx),
                "other_gt_point": [float(other_point[0]), float(other_point[1])],
                "other_gt_bbox_xyxy": [float(v) for v in other_gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])],
                "other_gt_distance_px": other_dist,
                "cross_instance_score": score,
                "pred_point_inside_other_gt": bool(in_other_box),
            }
            if best_other is None or candidate["cross_instance_score"] > best_other["cross_instance_score"]:
                best_other = candidate

        if best_other is None:
            continue
        if not (best_other["pred_point_inside_other_gt"] or best_other["other_gt_distance_px"] + 2.0 < own_l2):
            continue

        enriched = dict(case)
        enriched.update(best_other)
        mismatches.append(enriched)

    mismatches.sort(
        key=lambda item: (
            float(item.get("cross_instance_score", float("-inf"))),
            float(item.get("l2_px", float("-inf"))),
        ),
        reverse=True,
    )
    return mismatches


def summarize_split_error(
    records: list[dict],
    has_picking_threshold: float,
    visibility_score_key: str = "has_picking_score",
) -> dict:
    correct_pairs, fp_pairs, fn_pairs = collect_case_groups(
        records,
        0.5,
        has_picking_threshold,
        visibility_score_key=visibility_score_key,
    )
    l2_values = [float(item["l2_px"]) for item in correct_pairs if "l2_px" in item]
    dx_values = [float(item["dx_px"]) for item in correct_pairs if "dx_px" in item]
    dy_values = [float(item["dy_px"]) for item in correct_pairs if "dy_px" in item]

    size_groups = {"small": [], "medium": [], "large": []}
    area_values = [float(item["gt_area"]) for item in correct_pairs]
    if area_values:
        q1, q2 = np.quantile(np.asarray(area_values, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
        for item in correct_pairs:
            area = float(item["gt_area"])
            if area <= q1:
                size_groups["small"].append(float(item["l2_px"]))
            elif area <= q2:
                size_groups["medium"].append(float(item["l2_px"]))
            else:
                size_groups["large"].append(float(item["l2_px"]))

    return {
        "point_pair_count": len(l2_values),
        "mean_l2_px": float(np.mean(l2_values)) if l2_values else 0.0,
        "median_l2_px": float(np.median(l2_values)) if l2_values else 0.0,
        "p90_l2_px": float(np.quantile(np.asarray(l2_values), 0.90)) if l2_values else 0.0,
        "ppl_sr_30": float(np.mean(np.asarray(l2_values) <= 30.0)) if l2_values else 0.0,
        "ppl_sr_50": float(np.mean(np.asarray(l2_values) <= 50.0)) if l2_values else 0.0,
        "mean_abs_dx_px": float(np.mean(np.abs(dx_values))) if dx_values else 0.0,
        "mean_abs_dy_px": float(np.mean(np.abs(dy_values))) if dy_values else 0.0,
        "has_picking_correct_count": len(correct_pairs),
        "has_picking_false_positive_count": len(fp_pairs),
        "has_picking_false_negative_count": len(fn_pairs),
        "size_group_l2_px": {
            name: {
                "count": len(values),
                "mean_l2_px": float(np.mean(values)) if values else 0.0,
                "median_l2_px": float(np.median(values)) if values else 0.0,
            }
            for name, values in size_groups.items()
        },
    }


def summarize_decoupled_point_diagnostics(records: list[dict], has_picking_threshold: float) -> dict:
    correct_pairs, _, _ = collect_case_groups(records, 0.5, has_picking_threshold)
    visible_gt_count = 0
    oracle_iou50_cases = []
    oracle_point_in_gt_box_cases = []
    oracle_any_visible_cases = []

    for record in records:
        gt_entries = record.get("gt_instances", [])
        pred_entries = record.get("pred_instances", [])
        visible_preds = [
            (idx, pred)
            for idx, pred in enumerate(pred_entries)
            if float(pred.get("has_picking_score", 0.0)) >= has_picking_threshold
        ]
        if gt_entries and pred_entries:
            gt_boxes = torch.as_tensor([item["bbox_xyxy"] for item in gt_entries], dtype=torch.float32)
            pred_boxes = torch.as_tensor([item["bbox_xyxy"] for item in pred_entries], dtype=torch.float32)
            pred_gt_ious, _ = box_iou(pred_boxes, gt_boxes)
        else:
            pred_gt_ious = None

        for gt_idx, gt in enumerate(gt_entries):
            if not bool(gt.get("has_picking")):
                continue
            visible_gt_count += 1

            best_any = None
            best_point_in_gt_box = None
            best_iou50 = None

            for pred_idx, pred in visible_preds:
                iou = float(pred_gt_ious[pred_idx, gt_idx].item()) if pred_gt_ious is not None else 0.0
                case = _build_point_case(record, gt_idx, gt, pred_idx, pred, iou)

                if best_any is None or case["l2_px"] < best_any["l2_px"]:
                    best_any = case
                if case["pred_point_inside_gt_box"] and (best_point_in_gt_box is None or case["l2_px"] < best_point_in_gt_box["l2_px"]):
                    best_point_in_gt_box = case
                if iou >= 0.5 and (best_iou50 is None or case["l2_px"] < best_iou50["l2_px"]):
                    best_iou50 = case

            if best_any is not None:
                oracle_any_visible_cases.append(best_any)
            if best_point_in_gt_box is not None:
                oracle_point_in_gt_box_cases.append(best_point_in_gt_box)
            if best_iou50 is not None:
                oracle_iou50_cases.append(best_iou50)

    return {
        "visible_gt_count": int(visible_gt_count),
        "standard_match": _summarize_point_cases(correct_pairs, visible_gt_count=visible_gt_count),
        "iou_conditioned": {
            "ge_0.50": _summarize_point_cases([item for item in correct_pairs if float(item.get("iou", 0.0)) >= 0.50]),
            "0.50_to_0.70": _summarize_point_cases([item for item in correct_pairs if 0.50 <= float(item.get("iou", 0.0)) < 0.70]),
            "0.70_to_0.85": _summarize_point_cases([item for item in correct_pairs if 0.70 <= float(item.get("iou", 0.0)) < 0.85]),
            "ge_0.70": _summarize_point_cases([item for item in correct_pairs if float(item.get("iou", 0.0)) >= 0.70]),
            "ge_0.85": _summarize_point_cases([item for item in correct_pairs if float(item.get("iou", 0.0)) >= 0.85]),
        },
        "oracle_candidates": {
            "iou_ge_0.50_visible": _summarize_point_cases(oracle_iou50_cases, visible_gt_count=visible_gt_count),
            "point_in_gt_box_visible": _summarize_point_cases(oracle_point_in_gt_box_cases, visible_gt_count=visible_gt_count),
            "any_visible_pred": _summarize_point_cases(oracle_any_visible_cases, visible_gt_count=visible_gt_count),
        },
        "note_zh": "oracle 指标只用于诊断误差来源，不替代正式 test 指标。若 oracle(any visible pred) 明显优于标准匹配，说明当前损失中包含较多 bbox / 排序 / 关联传播误差；若 oracle 仍不理想，则主矛盾更偏向 point 本身未学稳。",
    }


def _select_unique_cases(entries: list[dict], top_k: int) -> list[dict]:
    selected = []
    seen = set()
    for entry in entries:
        key = (entry["image_id"], entry["gt_index"], entry["pred_index"])
        if key in seen:
            continue
        selected.append(entry)
        seen.add(key)
        if len(selected) >= top_k:
            break
    return selected


def _draw_cross(draw: ImageDraw.ImageDraw, xy: list[float], color: str, size: int = 6, width: int = 2) -> None:
    x, y = xy
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def _draw_circle(draw: ImageDraw.ImageDraw, xy: list[float], color: str, radius: int = 5, width: int = 2) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def render_case_image(record: dict, focus: dict, out_path: Path, has_picking_threshold: float, style: str = "clean") -> None:
    image = Image.open(record["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if style == "debug":
        for gt in record.get("gt_instances", []):
            draw.rectangle(gt["bbox_xyxy"], outline="#00c7d9", width=2)
            if gt["has_picking"]:
                _draw_circle(draw, gt["picking_point"], "#00ff66", radius=4, width=2)

        for pred in record.get("pred_instances", []):
            draw.rectangle(pred["bbox_xyxy"], outline="#ff9f1c", width=1)
            if float(pred.get("has_picking_score", 0.0)) >= has_picking_threshold:
                _draw_cross(draw, pred.get("picking_point", [0.0, 0.0]), "#ff375f", size=5, width=2)

    draw.rectangle(focus["gt_bbox_xyxy"], outline="#00ff66", width=4)
    draw.rectangle(focus["pred_bbox_xyxy"], outline="#ff375f", width=4)
    if "other_gt_bbox_xyxy" in focus:
        draw.rectangle(focus["other_gt_bbox_xyxy"], outline="#ffb000", width=3)
    if focus.get("gt_has_picking"):
        _draw_circle(draw, focus["gt_point"], "#00ff66", radius=6, width=3)
    if "other_gt_point" in focus:
        _draw_circle(draw, focus["other_gt_point"], "#ffb000", radius=5, width=2)
    if focus.get("pred_has_picking"):
        _draw_cross(draw, focus["pred_point"], "#ff375f", size=7, width=3)
    elif style == "clean":
        px, py = focus["pred_bbox_xyxy"][0], focus["pred_bbox_xyxy"][1]
        draw.text((px + 2, max(py - 14, 0)), "pred_has=0", fill="#b0b0b0", font=font)
    if focus.get("gt_has_picking") and focus.get("pred_has_picking"):
        draw.line((*focus["gt_point"], *focus["pred_point"]), fill="#ffd166", width=3)
    if focus.get("pred_has_picking") and "other_gt_point" in focus:
        draw.line((*focus["pred_point"], *focus["other_gt_point"]), fill="#ffb000", width=2)

    lines = [
        f"image_id={focus['image_id']}  iou={focus['iou']:.3f}",
        f"gt_has={int(focus['gt_has_picking'])}  pred_has={int(focus['pred_has_picking'])}  pred_has_score={focus['pred_has_picking_score']:.3f}",
        f"pred_score={focus['pred_score']:.3f}",
    ]
    if "l2_px" in focus:
        lines.append(f"dx={focus['dx_px']:.1f}px  dy={focus['dy_px']:.1f}px  l2={focus['l2_px']:.1f}px")
    if "other_gt_distance_px" in focus:
        lines.append(
            f"nearby_gt_dist={focus['other_gt_distance_px']:.1f}px  in_other_gt={int(bool(focus.get('pred_point_inside_other_gt', False)))}"
        )

    box_x, box_y = 8, 8
    line_h = 15
    box_w = max(draw.textlength(line, font=font) for line in lines) + 10
    box_h = line_h * len(lines) + 8
    draw.rectangle((box_x, box_y, box_x + box_w, box_y + box_h), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((box_x + 5, box_y + 4 + idx * line_h), line, fill="white", font=font)

    image.save(out_path)


def generate_qualitative_cases(report_dir: Path, test_records: list[dict], has_picking_threshold: float, top_k: int) -> dict:
    case_root = report_dir / "qualitative_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    debug_root = report_dir / "debug_vis"
    debug_root.mkdir(parents=True, exist_ok=True)

    record_lookup = {int(record["image_id"]): record for record in test_records}
    correct_pairs, fp_pairs, fn_pairs = collect_case_groups(test_records, 0.5, has_picking_threshold)
    cross_instance_pairs = collect_cross_instance_mismatch_cases(test_records, correct_pairs)

    categories = {
        "best_cases": sorted(correct_pairs, key=lambda item: item.get("l2_px", float("inf"))),
        "worst_cases": sorted(correct_pairs, key=lambda item: item.get("l2_px", float("-inf")), reverse=True),
        "has_picking_false_positive_cases": sorted(fp_pairs, key=lambda item: item.get("pred_has_picking_score", 0.0), reverse=True),
        "has_picking_false_negative_cases": sorted(fn_pairs, key=lambda item: item.get("pred_score", 0.0), reverse=True),
        "x_bias_large_samples": sorted(correct_pairs, key=lambda item: abs(item.get("dx_px", 0.0)), reverse=True),
        "y_bias_large_samples": sorted(correct_pairs, key=lambda item: abs(item.get("dy_px", 0.0)), reverse=True),
        "cross_instance_mismatch_samples": cross_instance_pairs,
    }

    summary = {}
    for name, entries in categories.items():
        category_dir = case_root / name
        category_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = debug_root / name
        debug_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for rank, focus in enumerate(_select_unique_cases(entries, top_k), start=1):
            record = record_lookup.get(int(focus["image_id"]))
            if record is None:
                continue
            out_path = category_dir / f"rank{rank:02d}_image_{focus['image_id']}.png"
            debug_path = debug_dir / f"rank{rank:02d}_image_{focus['image_id']}.png"
            render_case_image(record, focus, out_path, has_picking_threshold, style="clean")
            render_case_image(record, focus, debug_path, has_picking_threshold, style="debug")
            outputs.append(
                {
                    "image_id": int(focus["image_id"]),
                    "file_name": focus["file_name"],
                    "clean_output": str(out_path.resolve()),
                    "debug_output": str(debug_path.resolve()),
                    "iou": safe_float(focus.get("iou")),
                    "l2_px": safe_float(focus.get("l2_px"), float("nan")),
                }
            )
        summary[name] = outputs

    return {
        "correct_visible_pair_count": len(correct_pairs),
        "has_picking_false_positive_count": len(fp_pairs),
        "has_picking_false_negative_count": len(fn_pairs),
        "cross_instance_mismatch_count": len(cross_instance_pairs),
        "clean_vis_root": str(case_root.resolve()),
        "debug_vis_root": str(debug_root.resolve()),
        "categories": summary,
    }


def _save_histogram(values: list[float], out_path: Path, title: str, xlabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if values:
        ax.hist(values, bins=min(25, max(8, int(len(values) ** 0.5))), color=color, alpha=0.85, edgecolor="white")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_error_analysis(report_dir: Path, test_records: list[dict], has_picking_threshold: float) -> dict:
    error_dir = report_dir / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)

    correct_pairs, fp_pairs, fn_pairs = collect_case_groups(test_records, 0.5, has_picking_threshold)
    cross_instance_pairs = collect_cross_instance_mismatch_cases(test_records, correct_pairs)

    l2_values = [float(item["l2_px"]) for item in correct_pairs if "l2_px" in item]
    dx_values = [float(item["dx_px"]) for item in correct_pairs if "dx_px" in item]
    dy_values = [float(item["dy_px"]) for item in correct_pairs if "dy_px" in item]

    _save_histogram(l2_values, error_dir / "point_l2_histogram.png", "Point L2 Error Histogram", "L2 error (px)", "#8c564b")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if dx_values:
        axes[0].hist(dx_values, bins=min(25, max(8, int(len(dx_values) ** 0.5))), color="#e377c2", alpha=0.85, edgecolor="white")
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)
    if dy_values:
        axes[1].hist(dy_values, bins=min(25, max(8, int(len(dy_values) ** 0.5))), color="#7f7f7f", alpha=0.85, edgecolor="white")
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)
    axes[0].set_title("X Bias Distribution")
    axes[1].set_title("Y Bias Distribution")
    axes[0].set_xlabel("dx (px)")
    axes[1].set_xlabel("dy (px)")
    for ax in axes:
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(error_dir / "xy_bias_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    size_groups = {"small": [], "medium": [], "large": []}
    area_values = [float(item["gt_area"]) for item in correct_pairs]
    if area_values:
        q1, q2 = np.quantile(np.asarray(area_values, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
        for item in correct_pairs:
            area = float(item["gt_area"])
            if area <= q1:
                size_groups["small"].append(float(item["l2_px"]))
            elif area <= q2:
                size_groups["medium"].append(float(item["l2_px"]))
            else:
                size_groups["large"].append(float(item["l2_px"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ["small", "medium", "large"],
        [statistics.mean(values) if values else 0.0 for values in size_groups.values()],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    ax.set_title("Point Error by Grape Size Group")
    ax.set_ylabel("Mean L2 error (px)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(error_dir / "point_error_by_size_group.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        ["correct", "has_fp", "has_fn"],
        [len(correct_pairs), len(fp_pairs), len(fn_pairs)],
        color=["#2ca02c", "#d62728", "#9467bd"],
    )
    axes[0].set_title("has_picking Group Counts")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(
        ["correct_visible"],
        [statistics.mean(l2_values) if l2_values else 0.0],
        color=["#8c564b"],
    )
    axes[1].set_title("Point Error on Correct Visible Matches")
    axes[1].set_ylabel("Mean L2 error (px)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(error_dir / "has_picking_group_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    clean_case_root = error_dir / "clean_vis"
    clean_case_root.mkdir(parents=True, exist_ok=True)
    record_lookup = {int(record["image_id"]): record for record in test_records}
    categories = {
        "top_worst_cases": sorted(correct_pairs, key=lambda item: item.get("l2_px", float("-inf")), reverse=True),
        "has_picking_false_positive_cases": sorted(fp_pairs, key=lambda item: item.get("pred_has_picking_score", 0.0), reverse=True),
        "has_picking_false_negative_cases": sorted(fn_pairs, key=lambda item: item.get("pred_score", 0.0), reverse=True),
        "x_bias_large_cases": sorted(correct_pairs, key=lambda item: abs(item.get("dx_px", 0.0)), reverse=True),
        "y_bias_large_cases": sorted(correct_pairs, key=lambda item: abs(item.get("dy_px", 0.0)), reverse=True),
    }
    case_outputs = {}
    for name, entries in categories.items():
        category_dir = clean_case_root / name
        category_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for rank, focus in enumerate(_select_unique_cases(entries, 6), start=1):
            record = record_lookup.get(int(focus["image_id"]))
            if record is None:
                continue
            out_path = category_dir / f"rank{rank:02d}_image_{focus['image_id']}.png"
            render_case_image(record, focus, out_path, has_picking_threshold, style="clean")
            outputs.append(str(out_path.resolve()))
        case_outputs[name] = outputs

    return {
        "point_pair_count": len(l2_values),
        "mean_l2_px": float(np.mean(l2_values)) if l2_values else 0.0,
        "median_l2_px": float(np.median(l2_values)) if l2_values else 0.0,
        "p90_l2_px": float(np.quantile(np.asarray(l2_values), 0.90)) if l2_values else 0.0,
        "mean_abs_dx_px": float(np.mean(np.abs(dx_values))) if dx_values else 0.0,
        "mean_abs_dy_px": float(np.mean(np.abs(dy_values))) if dy_values else 0.0,
        "has_picking_correct_count": len(correct_pairs),
        "has_picking_false_positive_count": len(fp_pairs),
        "has_picking_false_negative_count": len(fn_pairs),
        "cross_instance_mismatch_count": len(cross_instance_pairs),
        "size_group_l2_px": {
            name: {
                "count": len(values),
                "mean_l2_px": float(np.mean(values)) if values else 0.0,
                "median_l2_px": float(np.median(values)) if values else 0.0,
            }
            for name, values in size_groups.items()
        },
        "clean_vis_cases": case_outputs,
        "scene_tag_note_zh": "当前标注里没有可直接解析的遮挡/梗方向/单串多串标签，本轮未做场景标签统计。",
    }


def load_checkpoint_metric_state(run_dir: Path) -> dict | None:
    for path in (run_dir / "point_checkpoint_metrics.json", run_dir / "logs" / "point_checkpoint_metrics.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def plot_results_overview(
    rows: list[dict],
    split_metrics: dict,
    checkpoint_metrics: dict,
    reference_summary: dict | None,
    out_path: Path,
    primary_label: str = "point_run",
    reference_label: str = "point_v2",
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    splits = ["train", "valid", "test"]
    ax1.plot(splits, [split_metrics[s]["grape_detection"]["AP"] for s in splits], marker="o", label="grape AP", color="#1f77b4")
    ax1.plot(splits, [split_metrics[s]["has_picking"]["f1"] for s in splits], marker="o", label="has_picking F1", color="#2ca02c")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Primary Checkpoint Across Splits")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(splits), dtype=np.float64)
    width = 0.24
    ax2.bar(x - width, [split_metrics[s]["picking_point"]["mean_l2_px"] for s in splits], width=width, label="mean L2", color="#8c564b")
    ax2.bar(x, [split_metrics[s]["picking_point"]["median_l2_px"] for s in splits], width=width, label="median L2", color="#e377c2")
    ax2.bar(x + width, [split_metrics[s]["picking_point"]["p90_l2_px"] for s in splits], width=width, label="p90 L2", color="#7f7f7f")
    ax2.set_xticks(x, splits)
    ax2.legend()
    ax2.set_title("Point Error Across Splits")
    ax2.set_ylabel("L2 error (px)")
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    labels = [reference_label, primary_label]
    if reference_summary is not None:
        prev = reference_summary.get("primary_checkpoint_split_summary", {}).get("test", {})
        grape_values = [safe_float(prev.get("grape_detection", {}).get("AP")), split_metrics["test"]["grape_detection"]["AP"]]
        f1_values = [safe_float(prev.get("has_picking", {}).get("f1")), split_metrics["test"]["has_picking"]["f1"]]
    else:
        grape_values = [0.0, split_metrics["test"]["grape_detection"]["AP"]]
        f1_values = [0.0, split_metrics["test"]["has_picking"]["f1"]]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.32
    ax3.bar(x - 0.16, grape_values, width=width, label="grape AP", color="#1f77b4")
    ax3.bar(x + 0.16, f1_values, width=width, label="has_picking F1", color="#2ca02c")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Test Detection / Classification")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    if reference_summary is not None:
        prev = reference_summary.get("primary_checkpoint_split_summary", {}).get("test", {})
        prev_point = prev.get("picking_point", {})
        l2_values = [safe_float(prev_point.get("mean_l2_px")), split_metrics["test"]["picking_point"]["mean_l2_px"]]
        median_values = [safe_float(prev_point.get("median_l2_px")), split_metrics["test"]["picking_point"]["median_l2_px"]]
        p90_values = [safe_float(prev_point.get("p90_l2_px")), split_metrics["test"]["picking_point"]["p90_l2_px"]]
    else:
        l2_values = [0.0, split_metrics["test"]["picking_point"]["mean_l2_px"]]
        median_values = [0.0, split_metrics["test"]["picking_point"]["median_l2_px"]]
        p90_values = [0.0, split_metrics["test"]["picking_point"]["p90_l2_px"]]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    ax4.bar(x - width, l2_values, width=width, label="mean L2", color="#8c564b")
    ax4.bar(x, median_values, width=width, label="median L2", color="#e377c2")
    ax4.bar(x + width, p90_values, width=width, label="p90 L2", color="#7f7f7f")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_title(f"Test Point Error vs {reference_label}")
    ax4.set_ylabel("L2 error (px)")
    ax4.grid(axis="y", alpha=0.3)
    ax4.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_comparison_report_zh(
    report_path: Path,
    primary_name: str,
    split_metrics: dict,
    checkpoint_metrics: dict,
    reference_summary: dict | None,
    bbox_summary: dict | None,
    test_decoupled_summary: dict | None = None,
    test_unified_metrics: dict | None = None,
    report_title: str = "point 实验中文结论",
    primary_label: str = "point_run",
    reference_label: str = "point_v2",
    change_notes: list[str] | None = None,
) -> None:
    primary_test = split_metrics["test"]
    notes = list(change_notes or [])
    if not notes:
        notes = [
            "继续固定为 grape bbox detection + per-grape has_picking / point offset 回归，不回退到独立 picking bbox。",
            "优先围绕 point head 与当前 grape 实例的绑定关系做最小可行修改。",
            "报告默认保留 clean 可视化、关键误差图和 train/valid/test 三套指标，便于直接做论文分析。",
        ]
    lines = [
        f"# {report_title}",
        "",
        "## 本次改动点",
    ]
    lines.extend([f"- {item}" for item in notes])
    lines.extend(
        [
        "",
        "## 推荐主 checkpoint",
        f"- 当前推荐主结果使用 `{primary_name}`。",
        "- 选择理由：它在 grape AP、has_picking F1 和 point 误差之间更均衡，适合作为论文主线候选结果。",
        "",
        "## 本轮结果",
        f"- test grape AP/AP50/AR100 = {primary_test['grape_detection']['AP']:.4f} / {primary_test['grape_detection']['AP50']:.4f} / {primary_test['grape_detection']['AR100']:.4f}",
        f"- test has_picking precision/recall/F1 = {primary_test['has_picking']['precision']:.4f} / {primary_test['has_picking']['recall']:.4f} / {primary_test['has_picking']['f1']:.4f}",
        f"- test point pair_count / MAE_x / MAE_y / mean L2 / median L2 / p90 L2 = {int(primary_test['picking_point']['pair_count'])} / {primary_test['picking_point']['mae_x_px']:.2f} / {primary_test['picking_point']['mae_y_px']:.2f} / {primary_test['picking_point']['mean_l2_px']:.2f} / {primary_test['picking_point']['median_l2_px']:.2f} / {primary_test['picking_point']['p90_l2_px']:.2f} px",
        "",
        ]
    )
    if reference_summary is not None:
        prev = reference_summary.get("primary_checkpoint_split_summary", {}).get("test", {})
        lines.extend(
            [
                f"## 相比 {reference_label}",
                f"- grape AP 变化：{primary_test['grape_detection']['AP'] - safe_float(prev.get('grape_detection', {}).get('AP')):+.4f}",
                f"- has_picking F1 变化：{primary_test['has_picking']['f1'] - safe_float(prev.get('has_picking', {}).get('f1')):+.4f}",
                f"- point mean L2 变化：{primary_test['picking_point']['mean_l2_px'] - safe_float(prev.get('picking_point', {}).get('mean_l2_px')):+.2f} px",
                f"- median L2 变化：{primary_test['picking_point']['median_l2_px'] - safe_float(prev.get('picking_point', {}).get('median_l2_px')):+.2f} px",
                f"- p90 L2 变化：{primary_test['picking_point']['p90_l2_px'] - safe_float(prev.get('picking_point', {}).get('p90_l2_px')):+.2f} px",
                "",
            ]
        )
    qa = primary_test.get("picking_point", {}).get("quality_aligned")
    if qa:
        lines.extend(
            [
                "## point quality 对齐口径",
                f"- 使用 `has_picking_score * point_quality_score` 作为可采判断分数时，test pair_count / mean L2 / median L2 / p90 L2 = {int(qa.get('point_pair_count', qa.get('count', 0)))} / {safe_float(qa.get('mean_l2_px')):.2f} / {safe_float(qa.get('median_l2_px')):.2f} / {safe_float(qa.get('p90_l2_px')):.2f} px。",
                "- 该口径只用于判断质量头是否能过滤低质量点，不替代默认 has_picking 主评估口径。",
                "",
            ]
        )
    if test_unified_metrics is not None:
        instance = test_unified_metrics.get("instance_chain", {})
        global_chain = test_unified_metrics.get("global_chain", {})
        lines.extend(
            [
                "## 统一采摘点评估口径",
                "",
                (
                    "| chain | instance_f1 | global_visible_recall | global_f1 | visible GT | "
                    "pair_count | mean_L2 | PPL-SR@30 | PPL-SR@50 |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| instance-chain | {safe_float(instance.get('instance_visible_f1')):.4f} | "
                    f"{safe_float(global_chain.get('global_visible_recall')):.4f} | "
                    f"{safe_float(global_chain.get('global_visible_f1')):.4f} | "
                    f"{int(instance.get('visible_gt_total', 0))} | {int(instance.get('point_pair_count', 0))} | "
                    f"{safe_float(instance.get('point_mean_l2_px')):.2f} | "
                    f"{safe_float(instance.get('ppl_sr_30')):.4f} | {safe_float(instance.get('ppl_sr_50')):.4f} |"
                ),
                (
                    f"| global-chain | {safe_float(instance.get('instance_visible_f1')):.4f} | "
                    f"{safe_float(global_chain.get('global_visible_recall')):.4f} | "
                    f"{safe_float(global_chain.get('global_visible_f1')):.4f} | "
                    f"{int(global_chain.get('visible_gt_total', 0))} | {int(instance.get('point_pair_count', 0))} | "
                    f"{safe_float(instance.get('point_mean_l2_px')):.2f} | "
                    f"{safe_float(instance.get('ppl_sr_30')):.4f} | {safe_float(instance.get('ppl_sr_50')):.4f} |"
                ),
                "",
                "- instance-chain 对应旧 GPPoint-DETR 口径：先 IoU50 匹配 grape，再评价 visible 判断和点误差。",
                "- global-chain 把所有 visible GT 作为召回分母，precision 仍在 IoU50 instance chain 上计算，避免 DETR top-query 数量影响横向比较。",
                "",
            ]
        )
    if bbox_summary is not None:
        bbox_grape_ap = safe_float(bbox_summary.get("test_eval", {}).get("per_class", {}).get("grape", {}).get("AP"))
        lines.extend(
            [
                "## 相比旧 bbox_baseline",
                f"- grape AP 变化：{primary_test['grape_detection']['AP'] - bbox_grape_ap:+.4f}",
                "- 旧 baseline 的 picking 是 bbox AP，本轮不再与其直接做 picking AP 对比，而改看 has_picking F1 和 point 误差。",
                "",
            ]
        )
    if test_decoupled_summary is not None:
        standard = test_decoupled_summary.get("standard_match", {})
        iou_conditioned = test_decoupled_summary.get("iou_conditioned", {})
        oracle = test_decoupled_summary.get("oracle_candidates", {})
        iou85 = iou_conditioned.get("ge_0.85", {})
        oracle_iou50 = oracle.get("iou_ge_0.50_visible", {})
        oracle_gt_box = oracle.get("point_in_gt_box_visible", {})
        oracle_any = oracle.get("any_visible_pred", {})
        lines.extend(
            [
                "## 解耦评估",
                f"- visible GT(有采摘点) 数 = {int(test_decoupled_summary.get('visible_gt_count', 0))}；标准匹配 pair recall = {safe_float(standard.get('candidate_recall')):.4f}。",
                f"- 标准匹配下 pred point 落在 GT box 内的比例 = {safe_float(standard.get('pred_point_inside_gt_box_rate')):.4f}；IoU>=0.85 条件下 mean L2 = {safe_float(iou85.get('mean_l2_px')):.2f} px。",
                f"- oracle(iou>=0.5 的 visible pred) recall / mean L2 = {safe_float(oracle_iou50.get('candidate_recall')):.4f} / {safe_float(oracle_iou50.get('mean_l2_px')):.2f} px。",
                f"- oracle(pred point 落在 GT box 内) recall / mean L2 = {safe_float(oracle_gt_box.get('candidate_recall')):.4f} / {safe_float(oracle_gt_box.get('mean_l2_px')):.2f} px。",
                f"- oracle(全图任意 visible pred) recall / mean L2 = {safe_float(oracle_any.get('candidate_recall')):.4f} / {safe_float(oracle_any.get('mean_l2_px')):.2f} px。",
                "- 解读：如果 oracle(any visible pred) 明显优于标准匹配，说明当前误差里有较多 bbox / 排序 / 关联传播项；如果这些 oracle 指标仍然不理想，主矛盾就更像是 point 自身没有学稳。",
                "",
            ]
        )
    lines.append("## 各 best checkpoint 用途")
    for name, metrics in checkpoint_metrics.items():
        valid = metrics.get("valid", {})
        test = metrics.get("test", {})
        lines.append(
            f"- `{name}`: valid grape AP={safe_float(valid.get('grape_AP')):.4f}, valid has_picking F1={safe_float(valid.get('has_picking_f1')):.4f}, valid point L2={safe_float(valid.get('point_mean_l2_px')):.2f}px; test grape AP={safe_float(test.get('grape_AP')):.4f}, test has_picking F1={safe_float(test.get('has_picking_f1')):.4f}, test point L2={safe_float(test.get('point_mean_l2_px')):.2f}px"
        )
    lines.extend(
        [
            "",
            "## 剩余问题",
            "- 邻近串误关联如果仍明显，下一步继续加强局部实例特征约束，而不是回退到独立 picking bbox 检测。",
            "- 如果 dy 依然明显高于 dx，下一步优先继续优化纵向局部约束和顶部先验，而不是盲目继续堆损失项。",
            "- 如果 small grape 的 L2 仍显著高于 medium/large，再单独做 small grape 专项，而不是本轮同时叠太多模块。",
            "- 如果 grape AP 还没有明显追回，说明 point 分支与主检测之间仍然存在任务冲突，需要继续微调权重或 phase2 策略。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.config = args.config.resolve()
    args.dataset_root = args.dataset_root.resolve()
    if args.report_dir is None:
        args.report_dir = args.run_dir / "report"
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.run_dir / "log.txt"
    if not log_path.exists():
        alt = args.run_dir / "logs" / "log.txt"
        if alt.exists():
            log_path = alt
        else:
            raise FileNotFoundError(f"log.txt not found under {args.run_dir}")

    config_payload = load_point_config(args.config)
    point_cfg = point_checkpoint_cfg(config_payload)
    checkpoint_state = load_checkpoint_metric_state(args.run_dir)
    primary_checkpoint = select_primary_checkpoint(args.run_dir, args.checkpoint, checkpoint_state=checkpoint_state)
    named_checkpoints = find_checkpoint_paths(args.run_dir, checkpoint_state=checkpoint_state)
    if primary_checkpoint.name not in named_checkpoints:
        named_checkpoints[primary_checkpoint.name] = primary_checkpoint

    rows = build_epoch_rows(load_log_records(log_path), point_cfg)
    write_results_csv(rows, args.report_dir / "results.csv")
    plot_training_curves(rows, args.report_dir / "training_curves.png")

    primary_split_stats = {}
    primary_split_records = {}
    primary_split_error = {}
    primary_split_decoupled = {}
    primary_split_unified = {}
    has_picking_threshold = safe_float(config_payload.get("PostProcessor", {}).get("has_picking_threshold", 0.5), 0.5)
    for split in ("train", "valid", "test"):
        stats, records = evaluate_split(
            args.config,
            primary_checkpoint,
            split,
            args.dataset_root,
            args.batch_size,
            args.num_workers,
            args.device,
            collect_predictions=True,
        )
        primary_split_stats[split] = stats
        primary_split_records[split] = records
        if args.save_prediction_records:
            records_path = args.report_dir / f"{split}_prediction_records.json"
            records_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        primary_split_error[split] = summarize_split_error(records, has_picking_threshold)
        primary_split_unified[split] = compute_unified_point_metrics(
            records,
            iou_threshold=0.5,
            has_picking_threshold=has_picking_threshold,
            visibility_score_key="visible_score",
        )
        if any(
            "point_final_score" in pred
            for record in records
            for pred in record.get("pred_instances", [])
        ):
            primary_split_error[split]["quality_aligned"] = summarize_split_error(
                records,
                has_picking_threshold,
                visibility_score_key="point_final_score",
            )
        primary_split_decoupled[split] = summarize_decoupled_point_diagnostics(records, has_picking_threshold)

    checkpoint_metrics = {}
    for ckpt_name, ckpt_path in named_checkpoints.items():
        if ckpt_path == primary_checkpoint:
            checkpoint_metrics[ckpt_name] = {
                "path": str(ckpt_path),
                "valid": flatten_split_metrics(primary_split_stats["valid"], point_cfg),
                "test": flatten_split_metrics(primary_split_stats["test"], point_cfg),
            }
            continue
        valid_stats, _ = evaluate_split(
            args.config, ckpt_path, "valid", args.dataset_root, args.batch_size, args.num_workers, args.device, collect_predictions=False
        )
        test_stats, _ = evaluate_split(
            args.config, ckpt_path, "test", args.dataset_root, args.batch_size, args.num_workers, args.device, collect_predictions=False
        )
        checkpoint_metrics[ckpt_name] = {
            "path": str(ckpt_path),
            "valid": flatten_split_metrics(valid_stats, point_cfg),
            "test": flatten_split_metrics(test_stats, point_cfg),
        }

    split_summaries = {
        split: structured_split_summary(stats, point_cfg, error_summary=primary_split_error.get(split))
        for split, stats in primary_split_stats.items()
    }

    qualitative_summary = generate_qualitative_cases(args.report_dir, primary_split_records["test"], has_picking_threshold, args.top_k_cases)
    error_summary = generate_error_analysis(args.report_dir, primary_split_records["test"], has_picking_threshold)

    point_v2_summary = maybe_load_json(args.point_v2_summary)
    bbox_summary = maybe_load_json(args.bbox_baseline_summary)
    plot_results_overview(
        rows,
        split_summaries,
        checkpoint_metrics,
        point_v2_summary,
        args.report_dir / "results_overview.png",
        primary_label=args.primary_label,
        reference_label=args.reference_label,
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_mode": args.report_mode,
        "report_file_guide_zh": {
            "summary.json": "point 主汇总。优先看这里，包含 train/valid/test 三套指标、best checkpoint 有效性、参考基线对比和与旧 bbox 基线的差异。",
            "results.csv": "逐 epoch 指标总表。重点看 grape AP、has_picking F1、point mean L2 和 composite_score。",
            "training_curves.png": "训练过程曲线图。用于判断是否收敛、分支 loss 是否失衡。",
            "results_overview.png": "总览图。重点看 train/valid/test 对比，以及参考基线 vs 当前实验的 test 对比。",
            "qualitative_cases/": "clean_vis 主展示。默认只画匹配到的 GT/Pred grape、GT/Pred point 和误差线。",
            "debug_vis/": "调试可视化。保留更多候选框/预测，仅用于排查误关联。",
            "error_analysis/": "点误差统计图。看 L2 直方图、x/y 偏差、按 grape 尺寸分组误差，以及重点 clean_vis 样本。",
            "comparison_report_zh.md": "中文实验结论。适合直接整理进论文。"
        },
        "task_definition_zh": {
            "grape": "继续做 bbox 检测",
            "picking": "固定为与 grape 绑定的 has_picking + point offset 回归，不再作为独立 bbox 检测类"
        },
        "config": str(args.config),
        "run_dir": str(args.run_dir),
        "report_dir": str(args.report_dir),
        "primary_checkpoint": str(primary_checkpoint),
        "primary_checkpoint_name": primary_checkpoint.name,
        "composite_formula": {
            "formula": "valid_grape_AP + alpha * valid_has_picking_F1 - beta * (valid_point_mean_l2_px / point_error_norm_px)",
            "alpha": safe_float(point_cfg.get("alpha", 0.35), 0.35),
            "beta": safe_float(point_cfg.get("beta", 0.20), 0.20),
            "point_error_norm_px": safe_float(point_cfg.get("point_error_norm_px", 40.0), 40.0),
        },
        "data_summary": {
            "train": summarize_dataset(args.dataset_root / "train" / "_annotations.grape_point.json"),
            "valid": summarize_dataset(args.dataset_root / "valid" / "_annotations.grape_point.json"),
            "test": summarize_dataset(args.dataset_root / "test" / "_annotations.grape_point.json"),
        },
        "best_validation_epochs": {
            "grape_ap": best_epoch(rows, "valid_grape_AP", mode="max"),
            "grape_ap50": best_epoch(rows, "valid_grape_AP50", mode="max"),
            "has_picking_f1": best_epoch(rows, "valid_has_picking_f1", mode="max"),
            "point_mean_l2": best_epoch([row for row in rows if row["valid_point_pair_count"] > 0], "valid_point_mean_l2_px", mode="min"),
            "composite_score": best_epoch(rows, "composite_score", mode="max"),
        },
        **best_point_l2_state_summary(checkpoint_state),
        "convergence_last10": {
            "valid_grape_AP": tail_mean_std(rows, "valid_grape_AP", window=10),
            "valid_has_picking_F1": tail_mean_std(rows, "valid_has_picking_f1", window=10),
            "valid_point_mean_l2_px": tail_mean_std(rows, "valid_point_mean_l2_px", window=10),
            "composite_score": tail_mean_std(rows, "composite_score", window=10),
        },
        "checkpoint_metric_state": checkpoint_state,
        "primary_checkpoint_split_summary": split_summaries,
        "checkpoint_comparison": checkpoint_metrics,
        "reference_comparison": {
            args.reference_label: point_v2_summary,
            "bbox_baseline": bbox_summary,
        },
        "split_error_summary": primary_split_error,
        "unified_point_metrics": primary_split_unified,
        "decoupled_point_summary": primary_split_decoupled,
        "qualitative_cases": qualitative_summary,
        "error_analysis": error_summary,
    }

    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_comparison_report_zh(
        args.report_dir / "comparison_report_zh.md",
        primary_checkpoint.name,
        split_summaries,
        checkpoint_metrics,
        point_v2_summary,
        bbox_summary,
        test_decoupled_summary=primary_split_decoupled.get("test"),
        test_unified_metrics=primary_split_unified.get("test"),
        report_title=args.report_title,
        primary_label=args.primary_label,
        reference_label=args.reference_label,
        change_notes=args.change_note,
    )

    print(f"[report] wrote {summary_path}")
    print(f"[report] wrote {args.report_dir / 'results.csv'}")
    print(f"[report] wrote {args.report_dir / 'training_curves.png'}")
    print(f"[report] wrote {args.report_dir / 'results_overview.png'}")
    print(f"[report] wrote {args.report_dir / 'comparison_report_zh.md'}")


if __name__ == "__main__":
    main()
