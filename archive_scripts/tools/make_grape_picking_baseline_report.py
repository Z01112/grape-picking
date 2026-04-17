from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_picking_baseline.yml"


def read_config_output_dir(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = config.get("output_dir")
    if output_dir:
        return (REPO_ROOT / output_dir).resolve()
    return (REPO_ROOT / "outputs" / "baseline_main").resolve()


DEFAULT_RUN_DIR = read_config_output_dir(DEFAULT_CONFIG)
DEFAULT_REPORT_DIR = DEFAULT_RUN_DIR / "report"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best_stg2.pth"

PALETTE = [
    "#e76f51",
    "#2a9d8f",
    "#264653",
    "#f4a261",
    "#457b9d",
    "#8d99ae",
]

COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a thesis-friendly visual report for the current grape+picking experiment."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples-per-split", type=int, default=6)
    parser.add_argument("--comparison-count", type=int, default=6)
    parser.add_argument("--failure-count", type=int, default=6)
    parser.add_argument("--best-case-count", type=int, default=6)
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--ann-file-name", default="_annotations.rtv4.json")
    parser.add_argument("--full-report", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--reuse-existing-eval", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_placeholder_image(path: Path, title: str, subtitle: str = "") -> None:
    image = Image.new("RGB", (1280, 720), color=(248, 248, 246))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 80), title, fill="#264653", font=font)
    if subtitle:
        draw.text((40, 120), subtitle, fill="#555555", font=font)
    image.save(path)


def prune_report_dir(report_dir: Path) -> None:
    removable = (
        "class_focus_report.md",
        "confusion_matrix.png",
        "grape_report.json",
        "grape_report.md",
        "index.html",
        "per_class_detailed_metrics.json",
        "picking_report.json",
        "picking_report.md",
        "samples_test.jpg",
        "samples_train.jpg",
        "samples_valid.jpg",
        "split_class_balance.png",
        "test_best_cases.jpg",
        "test_eval.txt",
        "test_failure_cases.jpg",
        "thesis_ready_summary.md",
        "train_labels_overview.png",
    )
    for name in removable:
        path = report_dir / name
        if path.exists():
            path.unlink()

    for dirname in ("_runtime", "_runtime_eval", "eval_test"):
        path = report_dir / dirname
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def median_or_zero(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def metrics_from_eval_array(values: list[float]) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(COCO_METRIC_NAMES)}


def normalize_per_class_eval(per_class_payload: dict | None) -> dict[str, dict]:
    payload = per_class_payload or {}
    normalized = {}
    for class_name, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        normalized[class_name] = {
            "category_id": int(metrics.get("category_id", -1)),
            "AP": float(metrics.get("AP", 0.0)),
            "AP50": float(metrics.get("AP50", 0.0)),
            "AR100": float(metrics.get("AR100", 0.0)),
        }
    return normalized


def load_log_records(log_path: Path) -> tuple[list[dict], str | None]:
    records: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        raise ValueError(f"No JSON log records found in {log_path}")

    primary_records: list[dict] = []
    last_epoch = -1
    trim_note = None
    for idx, record in enumerate(records):
        epoch = int(record.get("epoch", -1))
        if primary_records and epoch <= last_epoch:
            trim_note = (
                "检测到 log.txt 在主训练结束后又发生了 epoch 回跳，"
                f"已自动只保留原始主训练段用于出报告。截断前最后正常 epoch={last_epoch}，"
                f"回跳后的首个 epoch={epoch}，日志行序号={idx + 1}。"
            )
            break
        primary_records.append(record)
        last_epoch = epoch

    return primary_records, trim_note


def resolve_run_artifact(run_dir: Path, *relative_candidates: str) -> Path:
    for candidate in relative_candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    return run_dir / relative_candidates[0]


def build_training_summary(records: list[dict]) -> dict:
    epochs = [int(record["epoch"]) for record in records]
    losses = [float(record.get("train_loss", 0.0)) for record in records]
    lrs = [float(record.get("train_lr", 0.0)) for record in records]
    eval_metrics = [metrics_from_eval_array(record["test_coco_eval_bbox"]) for record in records]
    per_class_records = [
        normalize_per_class_eval(record.get("test_coco_eval_bbox_per_class"))
        for record in records
    ]

    best_ap_idx = max(range(len(eval_metrics)), key=lambda idx: eval_metrics[idx]["AP"])
    best_ap50_idx = max(range(len(eval_metrics)), key=lambda idx: eval_metrics[idx]["AP50"])
    final_metrics = eval_metrics[-1]
    per_class_names = sorted({name for record in per_class_records for name in record})
    per_class_best = {}
    per_class_history = {}
    for class_name in per_class_names:
        history = []
        for epoch, metrics in zip(epochs, per_class_records):
            class_metrics = metrics.get(class_name)
            if not class_metrics:
                continue
            history.append(
                {
                    "epoch": epoch,
                    "AP": float(class_metrics["AP"]),
                    "AP50": float(class_metrics["AP50"]),
                    "AR100": float(class_metrics["AR100"]),
                }
            )
        if not history:
            continue
        best_class_ap = max(history, key=lambda item: item["AP"])
        best_class_ap50 = max(history, key=lambda item: item["AP50"])
        per_class_history[class_name] = history
        per_class_best[class_name] = {
            "source": "epoch_log",
            "selection_metric": "valid_AP",
            "best_ap_epoch": int(best_class_ap["epoch"]),
            "best_ap": float(best_class_ap["AP"]),
            "best_ap50_epoch": int(best_class_ap50["epoch"]),
            "best_ap50": float(best_class_ap50["AP50"]),
            "best_epoch_metrics": {
                "AP": float(best_class_ap["AP"]),
                "AP50": float(best_class_ap["AP50"]),
                "AR100": float(best_class_ap["AR100"]),
            },
        }

    return {
        "epochs": epochs,
        "losses": losses,
        "lrs": lrs,
        "eval_metrics": eval_metrics,
        "per_class_history": per_class_history,
        "per_class_best": per_class_best,
        "best_ap_epoch": epochs[best_ap_idx],
        "best_ap": eval_metrics[best_ap_idx]["AP"],
        "best_ap50_epoch": epochs[best_ap50_idx],
        "best_ap50": eval_metrics[best_ap50_idx]["AP50"],
        "final_epoch": epochs[-1],
        "final_metrics": final_metrics,
    }


def load_split_annotations(dataset_root: Path, split: str, ann_file_name: str) -> dict:
    return read_json(dataset_root / split / ann_file_name)


def collect_dataset_stats(dataset_root: Path, ann_file_name: str) -> dict:
    payloads = {}
    splits = {}
    train_payload = load_split_annotations(dataset_root, "train", ann_file_name)
    categories = {
        int(category["id"]): category["name"]
        for category in train_payload.get("categories", [])
    }

    for split in ("train", "valid", "test"):
        payload = load_split_annotations(dataset_root, split, ann_file_name)
        payloads[split] = payload
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        image_lookup = {int(image["id"]): image for image in images}
        per_class = Counter()
        per_image_counts = Counter()
        width_values = defaultdict(list)
        height_values = defaultdict(list)
        area_ratios = defaultdict(list)

        for ann in annotations:
            image = image_lookup[int(ann["image_id"])]
            class_name = categories[int(ann["category_id"])]
            bbox = ann["bbox"]
            width = float(bbox[2])
            height = float(bbox[3])
            area_ratio = (width * height) / float(image["width"] * image["height"])
            per_class[class_name] += 1
            per_image_counts[int(ann["image_id"])] += 1
            width_values[class_name].append(width)
            height_values[class_name].append(height)
            area_ratios[class_name].append(area_ratio)

        splits[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "per_class": dict(per_class),
            "per_image_counts": dict(per_image_counts),
            "width_values": {k: list(v) for k, v in width_values.items()},
            "height_values": {k: list(v) for k, v in height_values.items()},
            "area_ratios": {k: list(v) for k, v in area_ratios.items()},
        }

    return {
        "categories": categories,
        "splits": splits,
        "payloads": payloads,
    }


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def build_split_index(payload: dict, categories: dict[int, str]) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    images_by_id = {int(image["id"]): image for image in payload.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        label = int(ann["category_id"])
        annotations_by_image[image_id].append(
            {
                "label": label,
                "class_name": categories[label],
                "box": xywh_to_xyxy(ann["bbox"]),
                "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
                "id": int(ann["id"]),
            }
        )
    return images_by_id, annotations_by_image


def build_color_map(categories: dict[int, str]) -> dict[str, str]:
    ordered = [categories[idx] for idx in sorted(categories)]
    return {
        name: PALETTE[idx % len(PALETTE)]
        for idx, name in enumerate(ordered)
    }


def plot_training_curves(records: list[dict], output_path: Path) -> None:
    epochs = [int(record["epoch"]) for record in records]

    def values(key: str) -> list[float]:
        return [float(record.get(key, 0.0)) for record in records]

    ap = [float(record["test_coco_eval_bbox"][0]) for record in records]
    ap50 = [float(record["test_coco_eval_bbox"][1]) for record in records]
    ap75 = [float(record["test_coco_eval_bbox"][2]) for record in records]
    ar100 = [float(record["test_coco_eval_bbox"][8]) for record in records]
    lr = values("train_lr")
    per_class_records = [
        normalize_per_class_eval(record.get("test_coco_eval_bbox_per_class"))
        for record in records
    ]
    class_names = sorted({name for record in per_class_records for name in record})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    axes[0, 0].plot(epochs, values("train_loss"), color="#264653", linewidth=2)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(epochs, values("train_loss_bbox"), label="bbox", color="#e76f51", linewidth=2)
    axes[0, 1].plot(epochs, values("train_loss_giou"), label="giou", color="#f4a261", linewidth=2)
    axes[0, 1].set_title("Box Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, values("train_loss_mal"), label="mal", color="#2a9d8f", linewidth=2)
    axes[0, 2].plot(epochs, values("train_loss_fgl"), label="fgl", color="#457b9d", linewidth=2)
    axes[0, 2].set_title("Class / Local Losses")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, ap, label="AP", color="#264653", linewidth=2)
    axes[1, 0].plot(epochs, ap50, label="AP50", color="#2a9d8f", linewidth=2)
    axes[1, 0].plot(epochs, ap75, label="AP75", color="#e76f51", linewidth=2)
    axes[1, 0].plot(epochs, ar100, label="AR100", color="#8d99ae", linewidth=2)
    axes[1, 0].set_title("Validation AP")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    for idx, class_name in enumerate(class_names):
        class_ap = []
        class_ap50 = []
        for record in per_class_records:
            metrics = record.get(class_name, {})
            class_ap.append(float(metrics.get("AP", 0.0)))
            class_ap50.append(float(metrics.get("AP50", 0.0)))
        color = PALETTE[idx % len(PALETTE)]
        axes[1, 1].plot(epochs, class_ap, label=f"{class_name} AP", color=color, linewidth=2)
        axes[1, 1].plot(epochs, class_ap50, label=f"{class_name} AP50", color=color, linewidth=1.5, linestyle="--")
    axes[1, 1].set_title("Per-Class Validation")
    axes[1, 1].set_xlabel("Epoch")
    if class_names:
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "Per-class epoch metrics will appear\nafter the next training run.", ha="center", va="center")
        axes[1, 1].set_axis_off()

    axes[1, 2].plot(epochs, lr, color="#6d597a", linewidth=2)
    axes[1, 2].set_title("Learning Rate")
    axes[1, 2].set_xlabel("Epoch")

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.25)

    fig.suptitle("Training Curves Overview", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_results_overview(records: list[dict], output_path: Path) -> None:
    epochs = [int(record["epoch"]) for record in records]

    def values(key: str) -> list[float]:
        return [float(record.get(key, 0.0)) for record in records]

    overall_metrics = [metrics_from_eval_array(record["test_coco_eval_bbox"]) for record in records]
    ap = [item["AP"] for item in overall_metrics]
    ap50 = [item["AP50"] for item in overall_metrics]
    ar100 = [item["AR100"] for item in overall_metrics]
    loss = values("train_loss")
    bbox_loss = values("train_loss_bbox")
    giou_loss = values("train_loss_giou")
    cls_loss = values("train_loss_mal")
    fgl_loss = values("train_loss_fgl")
    lr = values("train_lr")

    per_class_records = [
        normalize_per_class_eval(record.get("test_coco_eval_bbox_per_class"))
        for record in records
    ]
    class_names = sorted({name for record in per_class_records for name in record})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0, 0].plot(epochs, loss, color="#1d3557", linewidth=2.4)
    axes[0, 0].set_title("Train Total Loss")
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(epochs, bbox_loss, label="bbox", color="#e76f51", linewidth=2.2)
    axes[0, 1].plot(epochs, giou_loss, label="giou", color="#f4a261", linewidth=2.2)
    axes[0, 1].plot(epochs, cls_loss, label="mal", color="#2a9d8f", linewidth=2.2)
    axes[0, 1].plot(epochs, fgl_loss, label="fgl", color="#457b9d", linewidth=2.2)
    axes[0, 1].set_title("Core Train Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[0, 2].plot(epochs, ap, label="AP", color="#1d3557", linewidth=2.4)
    axes[0, 2].plot(epochs, ap50, label="AP50", color="#2a9d8f", linewidth=2.4)
    axes[0, 2].plot(epochs, ar100, label="AR100", color="#6d597a", linewidth=2.4)
    axes[0, 2].set_title("Overall Validation")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend(frameon=False, fontsize=8)

    if class_names:
        for idx, class_name in enumerate(class_names):
            color = PALETTE[idx % len(PALETTE)]
            class_ap = [float(record.get(class_name, {}).get("AP", 0.0)) for record in per_class_records]
            class_ap50 = [float(record.get(class_name, {}).get("AP50", 0.0)) for record in per_class_records]
            class_ar100 = [float(record.get(class_name, {}).get("AR100", 0.0)) for record in per_class_records]
            axes[1, 0].plot(epochs, class_ap, label=f"{class_name}", color=color, linewidth=2.8)
            axes[1, 1].plot(epochs, class_ap50, label=f"{class_name}", color=color, linewidth=2.8)
            axes[1, 2].plot(epochs, class_ar100, label=f"{class_name}", color=color, linewidth=2.8)

            axes[1, 0].annotate(f"{class_ap[-1]:.3f}", (epochs[-1], class_ap[-1]), color=color, fontsize=8)
            axes[1, 1].annotate(f"{class_ap50[-1]:.3f}", (epochs[-1], class_ap50[-1]), color=color, fontsize=8)
            axes[1, 2].annotate(f"{class_ar100[-1]:.3f}", (epochs[-1], class_ar100[-1]), color=color, fontsize=8)

        axes[1, 0].set_title("Per-Class AP")
        axes[1, 1].set_title("Per-Class AP50")
        axes[1, 2].set_title("Per-Class AR100")
        for ax in axes[1]:
            ax.set_xlabel("Epoch")
            ax.legend(frameon=False, fontsize=8)
    else:
        note = "Per-class epoch curves will appear\nafter the next training run."
        axes[1, 0].text(0.5, 0.5, note, ha="center", va="center", fontsize=11)
        axes[1, 0].set_title("Per-Class AP")
        axes[1, 1].plot(epochs, lr, color="#6d597a", linewidth=2.4)
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 2].text(0.5, 0.5, "Current run used old epoch log format.\nNew runs will fill grape/picking curves.", ha="center", va="center", fontsize=10)
        axes[1, 2].set_title("Per-Class Curves")

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.22)

    fig.suptitle("Results Overview", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_results_csv(records: list[dict], output_path: Path) -> None:
    per_class_records = [
        normalize_per_class_eval(record.get("test_coco_eval_bbox_per_class"))
        for record in records
    ]
    class_names = sorted({name for record in per_class_records for name in record})

    fieldnames = [
        "epoch",
        "train/total_loss",
        "train/mal_loss",
        "train/bbox_loss",
        "train/giou_loss",
        "train/fgl_loss",
        "metrics/AP",
        "metrics/AP50",
        "metrics/AP75",
        "metrics/AR100",
        "metrics/AP_small",
        "metrics/AP_medium",
        "metrics/AP_large",
        "lr",
    ]
    for class_name in class_names:
        fieldnames.extend(
            [
                f"metrics/{class_name}_AP",
                f"metrics/{class_name}_AP50",
                f"metrics/{class_name}_AR100",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record, per_class in zip(records, per_class_records):
            overall = metrics_from_eval_array(record["test_coco_eval_bbox"])
            row = {
                "epoch": int(record["epoch"]),
                "train/total_loss": float(record.get("train_loss", 0.0)),
                "train/mal_loss": float(record.get("train_loss_mal", 0.0)),
                "train/bbox_loss": float(record.get("train_loss_bbox", 0.0)),
                "train/giou_loss": float(record.get("train_loss_giou", 0.0)),
                "train/fgl_loss": float(record.get("train_loss_fgl", 0.0)),
                "metrics/AP": overall["AP"],
                "metrics/AP50": overall["AP50"],
                "metrics/AP75": overall["AP75"],
                "metrics/AR100": overall["AR100"],
                "metrics/AP_small": overall["AP_small"],
                "metrics/AP_medium": overall["AP_medium"],
                "metrics/AP_large": overall["AP_large"],
                "lr": float(record.get("train_lr", 0.0)),
            }
            for class_name in class_names:
                class_metrics = per_class.get(class_name, {})
                row[f"metrics/{class_name}_AP"] = class_metrics.get("AP", "")
                row[f"metrics/{class_name}_AP50"] = class_metrics.get("AP50", "")
                row[f"metrics/{class_name}_AR100"] = class_metrics.get("AR100", "")
            writer.writerow(row)


def plot_train_label_overview(dataset_stats: dict, output_path: Path) -> None:
    train_stats = dataset_stats["splits"]["train"]
    class_names = list(train_stats["per_class"].keys())
    class_counts = [train_stats["per_class"][name] for name in class_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axes[0, 0].bar(class_names, class_counts, color=[PALETTE[idx] for idx, _ in enumerate(class_names)])
    axes[0, 0].set_title("Train Class Counts")
    axes[0, 0].set_ylabel("Boxes")

    for idx, class_name in enumerate(class_names):
        axes[0, 1].scatter(
            train_stats["width_values"][class_name],
            train_stats["height_values"][class_name],
            s=10,
            alpha=0.35,
            color=PALETTE[idx],
            label=class_name,
        )
    axes[0, 1].set_title("Box Width vs Height")
    axes[0, 1].set_xlabel("Width (px)")
    axes[0, 1].set_ylabel("Height (px)")
    axes[0, 1].legend()

    for idx, class_name in enumerate(class_names):
        area_percent = [value * 100.0 for value in train_stats["area_ratios"][class_name]]
        axes[1, 0].hist(area_percent, bins=30, alpha=0.5, color=PALETTE[idx], label=class_name)
    axes[1, 0].set_title("Box Area Ratio")
    axes[1, 0].set_xlabel("Box area / image area (%)")
    axes[1, 0].legend()

    annotations_per_image = list(train_stats["per_image_counts"].values())
    axes[1, 1].hist(
        annotations_per_image,
        bins=min(25, max(5, len(set(annotations_per_image)))),
        color="#457b9d",
    )
    axes[1, 1].set_title("Annotations per Image")
    axes[1, 1].set_xlabel("Boxes per image")

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.2)

    fig.suptitle("Dataset Overview (train split)", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_split_class_balance(dataset_stats: dict, categories: dict[int, str], output_path: Path) -> None:
    split_names = ["train", "valid", "test"]
    class_names = [categories[idx] for idx in sorted(categories)]
    x = np.arange(len(split_names))
    width = 0.35 if len(class_names) <= 2 else 0.25

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for idx, class_name in enumerate(class_names):
        counts = [dataset_stats["splits"][split]["per_class"].get(class_name, 0) for split in split_names]
        offsets = x + (idx - (len(class_names) - 1) / 2.0) * width
        ax.bar(offsets, counts, width=width, label=class_name, color=PALETTE[idx % len(PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_ylabel("Boxes")
    ax.set_title("Class Balance by Split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, color=(245, 245, 245))
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    offset_x = (size[0] - image.width) // 2
    offset_y = (size[1] - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas


def draw_detections_on_image(
    image_path: Path,
    detections: list[dict],
    categories: dict[int, str],
    color_map: dict[str, str],
    show_scores: bool = False,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_name = categories[int(det["label"])]
        color = color_map[class_name]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = class_name
        if show_scores and "score" in det:
            label = f"{class_name} {det['score']:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        top = max(0, int(y1) - text_h - 6)
        draw.rectangle([x1, top, x1 + text_w + 8, top + text_h + 6], fill=color)
        draw.text((x1 + 4, top + 3), label, fill="white", font=font)
    return image


def choose_sample_images(payload: dict, samples_per_split: int, seed: int) -> list[int]:
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    images = payload.get("images", [])
    ranked = []
    for image in images:
        anns = annotations_by_image.get(int(image["id"]), [])
        class_count = len({int(ann["category_id"]) for ann in anns})
        ranked.append((class_count, len(anns), image))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top = ranked[: max(samples_per_split * 3, samples_per_split)]
    random.Random(seed).shuffle(top)
    selected = top[:samples_per_split]
    selected.sort(key=lambda item: item[1], reverse=True)
    return [int(item[2]["id"]) for item in selected]


def build_gt_sample_grid(
    dataset_root: Path,
    split: str,
    images_by_id: dict[int, dict],
    gt_by_image: dict[int, list[dict]],
    categories: dict[int, str],
    output_path: Path,
    selected_image_ids: list[int],
) -> None:
    color_map = build_color_map(categories)
    tile_size = (420, 280)
    cols = 2
    rows = max(1, math.ceil(len(selected_image_ids) / cols))
    header_h = 34
    canvas = Image.new("RGB", (cols * tile_size[0], rows * (tile_size[1] + header_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, image_id in enumerate(selected_image_ids):
        row = idx // cols
        col = idx % cols
        image_info = images_by_id[image_id]
        image_path = dataset_root / split / image_info["file_name"]
        rendered = draw_detections_on_image(
            image_path=image_path,
            detections=gt_by_image.get(image_id, []),
            categories=categories,
            color_map=color_map,
            show_scores=False,
        )
        rendered = fit_image(rendered, tile_size)
        x = col * tile_size[0]
        y = row * (tile_size[1] + header_h)
        canvas.paste(rendered, (x, y + header_h))

        per_class = Counter(det["class_name"] for det in gt_by_image.get(image_id, []))
        caption = f"{image_info['file_name']} | " + ", ".join(f"{k}:{v}" for k, v in sorted(per_class.items()))
        draw.rectangle([x, y, x + tile_size[0], y + header_h], fill=(38, 70, 83))
        draw.text((x + 8, y + 9), caption[:72], fill="white", font=font)

    canvas.save(output_path)


def init_inference_solver(
    config_path: Path,
    dataset_root: Path,
    ann_file_name: str,
    checkpoint_path: Path,
    report_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=seed)
    cfg = YAMLConfig(
        str(config_path),
        resume=str(checkpoint_path),
        device=device,
        use_amp=False,
        output_dir=str(report_dir / "_runtime"),
        val_dataloader={
            "dataset": {
                "img_folder": str((dataset_root / "test").resolve()),
                "ann_file": str((dataset_root / "test" / ann_file_name).resolve()),
            },
            "total_batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        },
    )
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    return solver


@torch.no_grad()
def collect_predictions(
    solver,
    score_thr: float,
) -> dict[int, list[dict]]:
    model = solver.ema.module if getattr(solver, "ema", None) else solver.model
    model.eval()

    predictions_by_image: dict[int, list[dict]] = {}
    for samples, targets in solver.val_dataloader:
        samples = samples.to(solver.device)
        targets = [{k: v.to(solver.device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = solver.postprocessor(outputs, orig_target_sizes)

        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            scores = result["scores"].detach().cpu()
            labels = result["labels"].detach().cpu()
            boxes = result["boxes"].detach().cpu()
            keep = scores >= score_thr

            detections = []
            for score, label, box in zip(scores[keep], labels[keep], boxes[keep]):
                detections.append(
                    {
                        "label": int(label.item()),
                        "score": float(score.item()),
                        "box": [float(v) for v in box.tolist()],
                    }
                )
            detections.sort(key=lambda item: item["score"], reverse=True)
            predictions_by_image[image_id] = detections

    return predictions_by_image


def box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return safe_div(inter_area, union)


def greedy_match(gt: list[dict], pred: list[dict], iou_thr: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, float, int, int]] = []
    for gt_idx, gt_det in enumerate(gt):
        for pred_idx, pred_det in enumerate(pred):
            iou = box_iou_xyxy(gt_det["box"], pred_det["box"])
            if iou >= iou_thr:
                candidates.append((iou, pred_det.get("score", 0.0), gt_idx, pred_idx))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, _, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    unmatched_gt = [idx for idx in range(len(gt)) if idx not in matched_gt]
    unmatched_pred = [idx for idx in range(len(pred)) if idx not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def analyze_predictions(
    images_by_id: dict[int, dict],
    gt_by_image: dict[int, list[dict]],
    pred_by_image: dict[int, list[dict]],
    categories: dict[int, str],
    iou_thr: float,
) -> dict:
    num_classes = len(categories)
    bg_idx = num_classes
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    image_summaries: list[dict] = []

    for image_id, image_info in images_by_id.items():
        gt = gt_by_image.get(image_id, [])
        pred = pred_by_image.get(image_id, [])
        matches, unmatched_gt, unmatched_pred = greedy_match(gt, pred, iou_thr=iou_thr)

        correct = 0
        cls_error = 0
        match_details = []
        for gt_idx, pred_idx, iou in matches:
            gt_label = gt[gt_idx]["label"]
            pred_label = pred[pred_idx]["label"]
            confusion[gt_label, pred_label] += 1
            same_class = gt_label == pred_label
            if same_class:
                correct += 1
            else:
                cls_error += 1
            match_details.append(
                {
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "iou": round(float(iou), 4),
                    "same_class": same_class,
                }
            )

        for gt_idx in unmatched_gt:
            confusion[gt[gt_idx]["label"], bg_idx] += 1
        for pred_idx in unmatched_pred:
            confusion[bg_idx, pred[pred_idx]["label"]] += 1

        summary = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "gt_count": len(gt),
            "pred_count": len(pred),
            "correct": correct,
            "cls_error": cls_error,
            "missed": len(unmatched_gt),
            "extra": len(unmatched_pred),
            "error_score": len(unmatched_gt) * 3 + cls_error * 2 + len(unmatched_pred),
            "matches": match_details,
        }
        image_summaries.append(summary)

    class_metrics = []
    for class_idx in sorted(categories):
        class_name = categories[class_idx]
        tp = int(confusion[class_idx, class_idx])
        gt_total = int(confusion[class_idx, :].sum())
        pred_total = int(confusion[:, class_idx].sum())
        wrong_class = int(confusion[class_idx, :].sum() - confusion[class_idx, class_idx] - confusion[class_idx, bg_idx])
        missed = int(confusion[class_idx, bg_idx])
        background_fp = int(confusion[bg_idx, class_idx])
        confusion_fp = int(confusion[:, class_idx].sum() - confusion[class_idx, class_idx] - confusion[bg_idx, class_idx])
        precision = safe_div(tp, pred_total)
        recall = safe_div(tp, gt_total)
        f1 = safe_div(2 * precision * recall, precision + recall)
        class_metrics.append(
            {
                "class_id": class_idx,
                "class_name": class_name,
                "tp": tp,
                "gt_total": gt_total,
                "pred_total": pred_total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "wrong_class": wrong_class,
                "missed": missed,
                "background_fp": background_fp,
                "confusion_fp": confusion_fp,
            }
        )

    image_summaries.sort(key=lambda item: (item["error_score"], item["gt_count"], item["pred_count"]), reverse=True)
    return {
        "confusion_matrix": confusion,
        "class_metrics": class_metrics,
        "image_summaries": image_summaries,
    }


def plot_confusion_matrix(
    confusion: np.ndarray,
    categories: dict[int, str],
    output_path: Path,
) -> None:
    labels = [categories[idx] for idx in sorted(categories)] + ["background"]
    normalized = confusion.astype(float)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sums, out=np.zeros_like(normalized, dtype=float), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, matrix, title, fmt in [
        (axes[0], confusion, "Confusion Matrix (counts)", "d"),
        (axes[1], normalized, "Confusion Matrix (row-normalized)", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                text = format(value, fmt)
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_test_class_metrics(class_metrics: list[dict], eval_summary: dict | None, output_path: Path) -> None:
    if class_metrics:
        class_names = [item["class_name"] for item in class_metrics]
        precision = [item["precision"] for item in class_metrics]
        recall = [item["recall"] for item in class_metrics]
        f1 = [item["f1"] for item in class_metrics]
    else:
        class_names = sorted(((eval_summary or {}).get("per_class") or {}).keys())
        precision = [0.0 for _ in class_names]
        recall = [0.0 for _ in class_names]
        f1 = [0.0 for _ in class_names]

    if not class_names:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "No test metrics are available.", ha="center", va="center", fontsize=13)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    ap = []
    ap50 = []
    for class_name in class_names:
        metrics = (eval_summary or {}).get("per_class", {}).get(class_name, {})
        ap.append(float(metrics.get("AP", 0.0)))
        ap50.append(float(metrics.get("AP50", 0.0)))

    x = np.arange(len(class_names))
    width = 0.16

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if class_metrics:
        ax.bar(x - 2 * width, precision, width=width, label="Precision", color="#264653")
        ax.bar(x - width, recall, width=width, label="Recall", color="#2a9d8f")
        ax.bar(x, f1, width=width, label="F1", color="#f4a261")
        ax.bar(x + width, ap, width=width, label="AP", color="#457b9d")
        ax.bar(x + 2 * width, ap50, width=width, label="AP50", color="#e76f51")
    else:
        width = 0.28
        ax.bar(x - width / 2, ap, width=width, label="AP", color="#457b9d")
        ax.bar(x + width / 2, ap50, width=width, label="AP50", color="#e76f51")
        ax.text(
            0.5,
            0.95,
            "This chart was recovered from saved COCO eval artifacts.\nPrecision / Recall / F1 are unavailable because the original checkpoint was overwritten after resume.",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test-Class Performance Summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_breakdown(class_metrics: list[dict], output_path: Path) -> None:
    if not class_metrics:
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.5,
            0.60,
            "Prediction-level error breakdown is unavailable in recovery mode.",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.text(
            0.5,
            0.42,
            "The original 100-epoch eval artifacts were preserved, but the checkpoint was overwritten after an accidental resume.",
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    class_names = [item["class_name"] for item in class_metrics]
    correct = [item["tp"] for item in class_metrics]
    wrong_class = [item["wrong_class"] for item in class_metrics]
    missed = [item["missed"] for item in class_metrics]
    confusion_fp = [item["confusion_fp"] for item in class_metrics]
    background_fp = [item["background_fp"] for item in class_metrics]

    x = np.arange(len(class_names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].bar(x, correct, label="Correct", color="#2a9d8f")
    axes[0].bar(x, wrong_class, bottom=correct, label="Wrong class", color="#f4a261")
    axes[0].bar(x, missed, bottom=np.array(correct) + np.array(wrong_class), label="Missed", color="#e76f51")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].set_title("Ground-Truth Outcomes")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, correct, label="Correct", color="#2a9d8f")
    axes[1].bar(x, confusion_fp, bottom=correct, label="Confusion FP", color="#f4a261")
    axes[1].bar(
        x,
        background_fp,
        bottom=np.array(correct) + np.array(confusion_fp),
        label="Background FP",
        color="#e76f51",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_title("Prediction Outcomes")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def choose_best_case_ids(image_summaries: list[dict], count: int) -> list[int]:
    candidates = [
        item
        for item in image_summaries
        if item["error_score"] == 0 and item["gt_count"] > 0
    ]
    candidates.sort(key=lambda item: (item["gt_count"], item["pred_count"]), reverse=True)
    return [item["image_id"] for item in candidates[:count]]


def choose_failure_case_ids(image_summaries: list[dict], count: int) -> list[int]:
    candidates = [item for item in image_summaries if item["error_score"] > 0]
    candidates.sort(key=lambda item: (item["error_score"], item["missed"], item["extra"], item["cls_error"]), reverse=True)
    return [item["image_id"] for item in candidates[:count]]


def build_comparison_grid(
    dataset_root: Path,
    split: str,
    images_by_id: dict[int, dict],
    gt_by_image: dict[int, list[dict]],
    pred_by_image: dict[int, list[dict]],
    categories: dict[int, str],
    selected_image_ids: list[int],
    image_summaries: dict[int, dict],
    output_path: Path,
) -> None:
    if not selected_image_ids:
        placeholder = Image.new("RGB", (1200, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(placeholder)
        draw.text((20, 80), "No images available for this view.", fill="black", font=ImageFont.load_default())
        placeholder.save(output_path)
        return

    color_map = build_color_map(categories)
    panel_size = (380, 250)
    gap = 14
    rows = len(selected_image_ids)
    header_h = 38
    subheader_h = 22
    tile_h = header_h + subheader_h + panel_size[1]
    tile_w = panel_size[0] * 2 + gap * 3
    canvas = Image.new("RGB", (tile_w, rows * tile_h), color=(248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, image_id in enumerate(selected_image_ids):
        image_info = images_by_id[image_id]
        image_path = dataset_root / split / image_info["file_name"]
        gt_image = draw_detections_on_image(
            image_path=image_path,
            detections=gt_by_image.get(image_id, []),
            categories=categories,
            color_map=color_map,
            show_scores=False,
        )
        pred_image = draw_detections_on_image(
            image_path=image_path,
            detections=pred_by_image.get(image_id, []),
            categories=categories,
            color_map=color_map,
            show_scores=True,
        )
        gt_image = fit_image(gt_image, panel_size)
        pred_image = fit_image(pred_image, panel_size)

        summary = image_summaries.get(image_id, {})
        x = 0
        y = idx * tile_h
        draw.rectangle([x, y, x + tile_w, y + header_h], fill=(38, 70, 83))
        caption = (
            f"{image_info['file_name']} | gt={summary.get('gt_count', 0)} pred={summary.get('pred_count', 0)} "
            f"| ok={summary.get('correct', 0)} cls={summary.get('cls_error', 0)} "
            f"miss={summary.get('missed', 0)} extra={summary.get('extra', 0)}"
        )
        draw.text((x + 10, y + 11), caption[:150], fill="white", font=font)

        draw.text((x + gap + 8, y + header_h + 4), "Ground Truth", fill="#264653", font=font)
        draw.text((x + gap * 2 + panel_size[0] + 8, y + header_h + 4), "Prediction", fill="#264653", font=font)
        canvas.paste(gt_image, (x + gap, y + header_h + subheader_h))
        canvas.paste(pred_image, (x + gap * 2 + panel_size[0], y + header_h + subheader_h))

    canvas.save(output_path)


def export_predictions_json(
    images_by_id: dict[int, dict],
    predictions_by_image: dict[int, list[dict]],
    categories: dict[int, str],
    output_path: Path,
) -> None:
    export_payload = []
    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        detections = []
        for det in predictions_by_image.get(image_id, []):
            detections.append(
                {
                    "class_id": int(det["label"]),
                    "class_name": categories[int(det["label"])],
                    "score": round(float(det["score"]), 6),
                    "box_xyxy": [round(float(v), 3) for v in det["box"]],
                }
            )
        export_payload.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "predictions": detections,
            }
        )
    save_json(output_path, export_payload)


def build_summary_json(
    training_summary: dict,
    dataset_stats: dict,
    eval_summary: dict | None,
    prediction_analysis: dict | None,
    class_best_validation_known: dict | None,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    train_stats = dataset_stats["splits"]["train"]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_mode": "full" if args.full_report else "minimal",
        "report_file_guide_zh": {
            "summary.json": "总汇总文件。优先看这里，里面包含最佳验证轮次、test checkpoint 说明、overall 指标，以及 grape/picking 的分类汇总。",
            "per_image_test_summary.json": "逐图统计文件。每张测试图都有 gt_count、pred_count、correct、missed、extra、error_score，后续查难图主要看它。",
        "predictions_test.json": "测试集逐图预测结果。每张图包含预测框、类别和分数，适合后续做误检分析或二次统计。",
        "training_curves.png": "训练过程曲线图。适合快速看 loss、验证指标随 epoch 的变化。",
        "results_overview.png": "类似 results.png 的总览图。更强调 grape/picking 曲线和整体趋势。",
        "test_class_metrics.png": "分类别指标图。重点看 grape 和 picking 的 precision、recall、F1、AP、AP50。",
        "error_breakdown.png": "分类别错误分解图。重点看 missed 和 background false positives。",
        "test_gt_vs_pred.jpg": "测试集代表样例对比图。用于直观看预测效果。",
        },
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_name": args.checkpoint.name,
        "checkpoint_role": describe_checkpoint_role(args.checkpoint),
        "device": args.device,
        "score_threshold": args.score_thr,
        "iou_threshold": args.iou_thr,
        "train": {
            "images": train_stats["images"],
            "annotations": train_stats["annotations"],
            "per_class": train_stats["per_class"],
        },
        "best_validation": {
            "best_ap_epoch": training_summary["best_ap_epoch"],
            "best_ap": training_summary["best_ap"],
            "best_ap50_epoch": training_summary["best_ap50_epoch"],
            "best_ap50": training_summary["best_ap50"],
        },
        "best_validation_per_class": training_summary.get("per_class_best", {}),
        "final_validation": training_summary["final_metrics"],
        "class_best_validation_known": class_best_validation_known,
        "test_eval_note": (
            "test_eval.txt and test_eval below are produced by running train.py --test-only on the checkpoint above."
        ),
        "test_eval": eval_summary,
        "class_summary": build_class_focus_payloads(
            training_summary=training_summary,
            eval_summary=eval_summary,
            prediction_analysis=prediction_analysis,
            args=args,
        ),
    }
    if getattr(args, "report_recovery_note_zh", None):
        payload["report_recovery_note_zh"] = args.report_recovery_note_zh
    if getattr(args, "eval_source_note_zh", None):
        payload["eval_source_note_zh"] = args.eval_source_note_zh
    if prediction_analysis is not None:
        payload["prediction_analysis"] = {
            "class_metrics": prediction_analysis["class_metrics"],
            "confusion_matrix": prediction_analysis["confusion_matrix"].tolist(),
            "hardest_images": prediction_analysis["image_summaries"][: min(10, len(prediction_analysis["image_summaries"]))],
        }
    save_json(output_path, payload)


def build_thesis_summary_markdown(
    report_dir: Path,
    dataset_stats: dict,
    training_summary: dict,
    eval_summary: dict | None,
    prediction_analysis: dict | None,
    args: argparse.Namespace,
) -> None:
    train_stats = dataset_stats["splits"]["train"]
    valid_stats = dataset_stats["splits"]["valid"]
    test_stats = dataset_stats["splits"]["test"]

    lines = [
        "# Grape + Picking Baseline Thesis Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Config: `{args.config.resolve()}`",
        f"Checkpoint: `{args.checkpoint.resolve()}`",
        f"Checkpoint role: {describe_checkpoint_role(args.checkpoint)}",
        f"Score threshold for visualization/error analysis: `{args.score_thr}`",
        f"IoU threshold for matching/error analysis: `{args.iou_thr}`",
        "",
        "## Dataset",
        f"- Train: {train_stats['images']} images, {train_stats['annotations']} boxes",
        f"- Valid: {valid_stats['images']} images, {valid_stats['annotations']} boxes",
        f"- Test: {test_stats['images']} images, {test_stats['annotations']} boxes",
    ]
    for class_name, count in sorted(train_stats["per_class"].items()):
        lines.append(f"- Train `{class_name}` boxes: {count}")

    lines.extend(
        [
            "",
            "## Validation Summary",
            f"- Best validation AP: {training_summary['best_ap']:.4f} at epoch {training_summary['best_ap_epoch']}",
            f"- Best validation AP50: {training_summary['best_ap50']:.4f} at epoch {training_summary['best_ap50_epoch']}",
            f"- Final validation AP/AP50/AP75: "
            f"{training_summary['final_metrics']['AP']:.4f} / "
            f"{training_summary['final_metrics']['AP50']:.4f} / "
            f"{training_summary['final_metrics']['AP75']:.4f}",
        ]
    )

    if eval_summary:
        overall = eval_summary.get("overall", {})
        lines.extend(
            [
                "",
                "## Test Summary",
                "- `test_eval.txt` is the test-set evaluation of the checkpoint listed above, not a history of all epochs.",
                f"- Overall AP/AP50/AP75/AR100: "
                f"{overall.get('AP', 0.0):.3f} / {overall.get('AP50', 0.0):.3f} / "
                f"{overall.get('AP75', 0.0):.3f} / {overall.get('AR100', 0.0):.3f}",
            ]
        )
        for class_name, metrics in sorted(eval_summary.get("per_class", {}).items()):
            lines.append(
                f"- {class_name}: AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}, AR100={metrics['AR100']:.4f}"
            )

    if prediction_analysis:
        lines.extend(["", "## Prediction Error Analysis"])
        for item in prediction_analysis["class_metrics"]:
            lines.append(
                f"- {item['class_name']}: precision={item['precision']:.4f}, recall={item['recall']:.4f}, "
                f"F1={item['f1']:.4f}, missed={item['missed']}, background_fp={item['background_fp']}, "
                f"wrong_class={item['wrong_class']}"
            )

        hardest = prediction_analysis["image_summaries"][: min(5, len(prediction_analysis["image_summaries"]))]
        if hardest:
            lines.extend(["", "## Hardest Test Images"])
            for item in hardest:
                lines.append(
                    f"- {item['file_name']}: gt={item['gt_count']}, pred={item['pred_count']}, "
                    f"ok={item['correct']}, cls_err={item['cls_error']}, "
                    f"missed={item['missed']}, extra={item['extra']}"
                )

        lines.extend(
            [
                "",
                "## Interpretation",
                "- The grape category is much stronger than the picking category in both AP and qualitative examples.",
                "- Small-object sensitivity remains the main bottleneck, especially for partially visible picking points.",
                "- The report images in this folder can be cited directly in the thesis to illustrate both strengths and failure cases.",
            ]
        )

    save_text(report_dir / "thesis_ready_summary.md", "\n".join(lines) + "\n")


def run_split_eval(
    config_path: Path,
    dataset_root: Path,
    ann_file_name: str,
    checkpoint_path: Path,
    run_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    split: str,
) -> tuple[dict | None, str]:
    if not checkpoint_path.exists():
        return None, f"Checkpoint not found, skipped {split} evaluation."

    split_dir = dataset_root / split
    eval_output_dir = run_dir / "report" / f"_runtime_eval_{split}"
    shutil.rmtree(eval_output_dir, ignore_errors=True)
    cmd = [
        sys.executable,
        "train.py",
        "-c",
        str(config_path),
        "--test-only",
        "-r",
        str(checkpoint_path),
        "-d",
        device,
        "-u",
        f"val_dataloader.dataset.img_folder='{split_dir.as_posix()}'",
        f"val_dataloader.dataset.ann_file='{(split_dir / ann_file_name).as_posix()}'",
        f"output_dir='{eval_output_dir.as_posix()}'",
        f"val_dataloader.total_batch_size={batch_size}",
        f"val_dataloader.num_workers={num_workers}",
        "HGNetv2.pretrained=False",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    shutil.rmtree(eval_output_dir, ignore_errors=True)
    if result.returncode != 0:
        return {"error": f"Evaluation failed with exit code {result.returncode}"}, text
    return parse_eval_stdout(text), text


def run_test_eval(
    config_path: Path,
    dataset_root: Path,
    ann_file_name: str,
    checkpoint_path: Path,
    run_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[dict | None, str]:
    return run_split_eval(
        config_path=config_path,
        dataset_root=dataset_root,
        ann_file_name=ann_file_name,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        split="test",
    )


def parse_eval_stdout(text: str) -> dict:
    summary: dict[str, object] = {"overall": {}, "per_class": {}}
    patterns = {
        "AP": r"Average Precision\s+\(AP\)\s+@\[ IoU=0.50:0.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)",
        "AP50": r"Average Precision\s+\(AP\)\s+@\[ IoU=0.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)",
        "AP75": r"Average Precision\s+\(AP\)\s+@\[ IoU=0.75\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)",
        "AR100": r"Average Recall\s+\(AR\)\s+@\[ IoU=0.50:0.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)",
    }
    overall: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            overall[key] = float(match.group(1))
    summary["overall"] = overall

    per_class = {}
    for class_name, ap, ap50, ar100 in re.findall(
        r"^\s*([A-Za-z0-9_+-]+)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)\s+AR100=([0-9.]+)",
        text,
        flags=re.MULTILINE,
    ):
        per_class[class_name] = {
            "AP": float(ap),
            "AP50": float(ap50),
            "AR100": float(ar100),
        }
    summary["per_class"] = per_class
    return summary


def mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def find_metric_index(values, target: float, name: str) -> int:
    for idx, value in enumerate(values):
        if math.isclose(float(value), target, rel_tol=1e-9, abs_tol=1e-9):
            return idx
    raise ValueError(f"Could not find {name}={target} in evaluation artifact.")


def resolve_eval_artifact(run_dir: Path, split: str) -> Path:
    candidates = (
        run_dir / "logs" / f"eval_{split}" / "eval.pth",
        run_dir / f"eval_{split}" / "eval.pth",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Existing eval artifact was not found for split='{split}' under: {run_dir}")


def load_eval_summary_from_artifact(eval_path: Path, categories: dict[int, str]) -> dict:
    payload = torch.load(eval_path, map_location="cpu", weights_only=False)
    precision = np.asarray(payload["precision"])
    recall = np.asarray(payload["recall"])
    params = payload["params"]

    iou_thrs = [float(v) for v in params.iouThrs]
    cat_ids = [int(v) for v in params.catIds]
    area_labels = list(params.areaRngLbl)
    max_dets = list(params.maxDets)

    iou50_idx = find_metric_index(iou_thrs, 0.50, "IoU")
    iou75_idx = find_metric_index(iou_thrs, 0.75, "IoU")
    area_all_idx = area_labels.index("all")
    max_det_100_idx = max_dets.index(100)

    summary = {
        "overall": {
            "AP": mean_valid(precision[:, :, :, area_all_idx, max_det_100_idx]),
            "AP50": mean_valid(precision[iou50_idx : iou50_idx + 1, :, :, area_all_idx, max_det_100_idx]),
            "AP75": mean_valid(precision[iou75_idx : iou75_idx + 1, :, :, area_all_idx, max_det_100_idx]),
            "AR100": mean_valid(recall[:, :, area_all_idx, max_det_100_idx]),
        },
        "per_class": {},
        "source": str(eval_path.resolve()),
        "source_type": "existing_eval_artifact",
    }

    for class_index, category_id in enumerate(cat_ids):
        class_name = categories.get(category_id, str(category_id))
        summary["per_class"][class_name] = {
            "AP": mean_valid(precision[:, :, class_index, area_all_idx, max_det_100_idx]),
            "AP50": mean_valid(precision[iou50_idx : iou50_idx + 1, :, class_index, area_all_idx, max_det_100_idx]),
            "AR100": mean_valid(recall[:, class_index, area_all_idx, max_det_100_idx]),
        }

    return summary


def describe_checkpoint_role(checkpoint_path: Path) -> str:
    name = checkpoint_path.name.lower()
    if name == "best_stg2.pth":
        return "This is the stage-2 checkpoint with the best validation AP. The default report uses it for test evaluation."
    if name == "best_stg1.pth":
        return "This is the stage-1 checkpoint with the best validation AP."
    if name == "last.pth":
        return "This is the last saved checkpoint, not necessarily the best one."
    return "This is a user-specified checkpoint."


def infer_checkpoint_epoch_hint(checkpoint_path: Path, training_summary: dict) -> int | None:
    name = checkpoint_path.name.lower()
    if name == "best_stg2.pth":
        return int(training_summary.get("best_ap_epoch", -1))
    if name == "last.pth":
        return int(training_summary.get("final_epoch", -1))
    return None


def collect_class_best_validation_known(
    config_path: Path,
    dataset_root: Path,
    ann_file_name: str,
    run_dir: Path,
    training_summary: dict,
    args: argparse.Namespace,
) -> dict:
    candidate_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in (run_dir / "best_stg2.pth", run_dir / "last.pth", args.checkpoint):
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            candidate_paths.append(resolved)
            seen.add(resolved)

    candidates: list[dict] = []
    for checkpoint_path in candidate_paths:
        eval_summary, _ = run_split_eval(
            config_path=config_path,
            dataset_root=dataset_root,
            ann_file_name=ann_file_name,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split="valid",
        )
        candidates.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_role": describe_checkpoint_role(checkpoint_path),
                "epoch_known": infer_checkpoint_epoch_hint(checkpoint_path, training_summary),
                "valid_eval": eval_summary,
            }
        )

    per_class: dict[str, dict] = {}
    class_names: set[str] = set()
    for candidate in candidates:
        valid_eval = candidate.get("valid_eval") or {}
        class_names.update((valid_eval.get("per_class") or {}).keys())

    for class_name in sorted(class_names):
        ranked: list[tuple[float, dict, dict]] = []
        for candidate in candidates:
            valid_eval = candidate.get("valid_eval") or {}
            metrics = (valid_eval.get("per_class") or {}).get(class_name)
            if metrics:
                ranked.append((float(metrics.get("AP", -1.0)), candidate, metrics))
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0], reverse=True)
        _, best_candidate, best_metrics = ranked[0]
        per_class[class_name] = {
            "selection_metric": "valid_AP",
            "best_epoch_known": best_candidate.get("epoch_known"),
            "best_checkpoint": best_candidate.get("checkpoint"),
            "best_checkpoint_name": best_candidate.get("checkpoint_name"),
            "best_checkpoint_role": best_candidate.get("checkpoint_role"),
            "best_valid_metrics": {
                "AP": float(best_metrics.get("AP", 0.0)),
                "AP50": float(best_metrics.get("AP50", 0.0)),
                "AR100": float(best_metrics.get("AR100", 0.0)),
            },
        }

    return {
        "note_zh": (
            "该字段用于补充总体 best_ap_epoch 之外的类别最佳轮次。"
            "它仅在当前保留下来的候选 checkpoint 中比较 valid 集分类别 AP，"
            "默认比较 best_stg2.pth 和 last.pth。"
        ),
        "selection_metric": "valid_AP",
        "candidates": candidates,
        "per_class": per_class,
    }


def build_class_focus_payloads(
    training_summary: dict,
    eval_summary: dict | None,
    prediction_analysis: dict | None,
    args: argparse.Namespace,
) -> dict[str, dict]:
    eval_by_name = (eval_summary or {}).get("per_class", {})
    pred_by_name = {
        item["class_name"]: item
        for item in (prediction_analysis or {}).get("class_metrics", [])
    }
    class_names = sorted(set(eval_by_name) | set(pred_by_name))
    checkpoint_note = describe_checkpoint_role(args.checkpoint)
    exact_class_best = training_summary.get("per_class_best", {})

    payloads: dict[str, dict] = {}
    class_best_lookup = getattr(args, "class_best_validation_known", {}) or {}
    class_best_lookup = class_best_lookup.get("per_class", {})
    for class_name in class_names:
        test_metrics = eval_by_name.get(class_name, {})
        pred_metrics = pred_by_name.get(class_name, {})
        tp = int(pred_metrics.get("tp", 0))
        gt_total = int(pred_metrics.get("gt_total", 0))
        pred_total = int(pred_metrics.get("pred_total", 0))
        wrong_class = int(pred_metrics.get("wrong_class", 0))
        missed = int(pred_metrics.get("missed", 0))
        background_fp = int(pred_metrics.get("background_fp", 0))
        confusion_fp = int(pred_metrics.get("confusion_fp", 0))
        payloads[class_name] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "class_name": class_name,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_name": args.checkpoint.name,
            "checkpoint_role": checkpoint_note,
            "test_eval_note": (
                "The values in test_eval.txt come from running train.py --test-only "
                "on the checkpoint above. They are not a full training-history summary."
            ),
            "best_validation": {
                "best_ap_epoch": training_summary["best_ap_epoch"],
                "best_ap": training_summary["best_ap"],
                "best_ap50_epoch": training_summary["best_ap50_epoch"],
                "best_ap50": training_summary["best_ap50"],
            },
            "test_coco_metrics": {
                "AP": float(test_metrics.get("AP", 0.0)),
                "AP50": float(test_metrics.get("AP50", 0.0)),
                "AR100": float(test_metrics.get("AR100", 0.0)),
            },
            "best_validation_for_this_class": exact_class_best.get(class_name) or class_best_lookup.get(class_name),
            "threshold_analysis": {
                "score_threshold": args.score_thr,
                "iou_threshold": args.iou_thr,
                "precision": float(pred_metrics.get("precision", 0.0)),
                "recall": float(pred_metrics.get("recall", 0.0)),
                "f1": float(pred_metrics.get("f1", 0.0)),
                "tp": tp,
                "gt_total": gt_total,
                "pred_total": pred_total,
                "false_negative_total": max(gt_total - tp, 0),
                "false_positive_total": max(pred_total - tp, 0),
                "missed": missed,
                "wrong_class": wrong_class,
                "background_fp": background_fp,
                "confusion_fp": confusion_fp,
            },
        }
    return payloads


def build_class_focus_reports(
    report_dir: Path,
    training_summary: dict,
    eval_summary: dict | None,
    prediction_analysis: dict | None,
    args: argparse.Namespace,
) -> None:
    payloads = build_class_focus_payloads(
        training_summary=training_summary,
        eval_summary=eval_summary,
        prediction_analysis=prediction_analysis,
        args=args,
    )
    save_json(report_dir / "per_class_detailed_metrics.json", payloads)

    combined_lines = [
        "# Class-Focused Test Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Checkpoint: `{args.checkpoint.resolve()}`",
        f"Checkpoint role: {describe_checkpoint_role(args.checkpoint)}",
        "- `test_eval.txt` is the test-set evaluation of the checkpoint above.",
        (
            f"- Best validation AP epoch: `{training_summary['best_ap_epoch']}`; "
            f"best validation AP50 epoch: `{training_summary['best_ap50_epoch']}`."
        ),
        "",
    ]

    if not payloads:
        combined_lines.append("No class-focused metrics are available because test evaluation or prediction analysis was skipped.")
        save_text(report_dir / "class_focus_report.md", "\n".join(combined_lines) + "\n")
        return

    for class_name, payload in payloads.items():
        coco_metrics = payload["test_coco_metrics"]
        threshold_metrics = payload["threshold_analysis"]
        lines = [
            f"# {class_name.capitalize()} Detailed Report",
            "",
            f"Generated: {payload['generated_at']}",
            f"Checkpoint: `{payload['checkpoint']}`",
            f"Checkpoint role: {payload['checkpoint_role']}",
            payload["test_eval_note"],
            "",
            "## Test COCO Metrics",
            f"- AP: `{coco_metrics['AP']:.4f}`",
            f"- AP50: `{coco_metrics['AP50']:.4f}`",
            f"- AR100: `{coco_metrics['AR100']:.4f}`",
            "",
            "## Threshold-Based Detection Analysis",
            f"- Score threshold: `{threshold_metrics['score_threshold']}`",
            f"- IoU threshold: `{threshold_metrics['iou_threshold']}`",
            f"- Precision: `{threshold_metrics['precision']:.4f}`",
            f"- Recall: `{threshold_metrics['recall']:.4f}`",
            f"- F1: `{threshold_metrics['f1']:.4f}`",
            f"- TP / GT / Pred: `{threshold_metrics['tp']}` / `{threshold_metrics['gt_total']}` / `{threshold_metrics['pred_total']}`",
            f"- False negatives: `{threshold_metrics['false_negative_total']}`",
            f"- False positives: `{threshold_metrics['false_positive_total']}`",
            f"- Missed: `{threshold_metrics['missed']}`",
            f"- Wrong class: `{threshold_metrics['wrong_class']}`",
            f"- Background FP: `{threshold_metrics['background_fp']}`",
            f"- Confusion FP: `{threshold_metrics['confusion_fp']}`",
        ]
        if class_name == "picking":
            lines.extend(
                [
                    "",
                    "## Focus Note",
                    "- `picking` is the current bottleneck class, so in practice you should prioritize AP50, AR100, precision, recall, F1, and background false positives.",
                ]
            )
        save_json(report_dir / f"{class_name}_report.json", payload)
        save_text(report_dir / f"{class_name}_report.md", "\n".join(lines) + "\n")

        combined_lines.extend(
            [
                f"## {class_name}",
                f"- Test COCO: AP=`{coco_metrics['AP']:.4f}`, AP50=`{coco_metrics['AP50']:.4f}`, AR100=`{coco_metrics['AR100']:.4f}`",
                (
                    f"- Threshold metrics: precision=`{threshold_metrics['precision']:.4f}`, "
                    f"recall=`{threshold_metrics['recall']:.4f}`, F1=`{threshold_metrics['f1']:.4f}`, "
                    f"TP=`{threshold_metrics['tp']}`, GT=`{threshold_metrics['gt_total']}`, Pred=`{threshold_metrics['pred_total']}`, "
                    f"FP=`{threshold_metrics['false_positive_total']}`, FN=`{threshold_metrics['false_negative_total']}`"
                ),
                "",
            ]
        )

    save_text(report_dir / "class_focus_report.md", "\n".join(combined_lines) + "\n")


def build_html_report(
    report_dir: Path,
    training_summary: dict,
    dataset_stats: dict,
    eval_summary: dict | None,
    prediction_analysis: dict | None,
    args: argparse.Namespace,
) -> None:
    train_stats = dataset_stats["splits"]["train"]
    valid_stats = dataset_stats["splits"]["valid"]
    test_stats = dataset_stats["splits"]["test"]

    class_rows = []
    for class_name, count in sorted(train_stats["per_class"].items()):
        area_values = train_stats["area_ratios"].get(class_name, [])
        class_rows.append(
            "<tr>"
            f"<td>{escape(class_name)}</td>"
            f"<td>{count}</td>"
            f"<td>{median_or_zero(train_stats['width_values'].get(class_name, [])):.1f}</td>"
            f"<td>{median_or_zero(train_stats['height_values'].get(class_name, [])):.1f}</td>"
            f"<td>{median_or_zero(area_values) * 100.0:.3f}%</td>"
            "</tr>"
        )

    eval_block = "<div class='card'><h2>Test Metrics</h2><p>Test eval was skipped.</p></div>"
    if eval_summary:
        overall = eval_summary.get("overall", {})
        per_class = eval_summary.get("per_class", {})
        per_class_rows = "".join(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{metrics['AP']:.4f}</td>"
            f"<td>{metrics['AP50']:.4f}</td>"
            f"<td>{metrics['AR100']:.4f}</td>"
            "</tr>"
            for name, metrics in sorted(per_class.items())
        )
        eval_block = (
            "<div class='card'>"
            "<h2>Test Metrics</h2>"
            f"<p>Checkpoint: <code>{escape(args.checkpoint.name)}</code>. {escape(describe_checkpoint_role(args.checkpoint))}</p>"
            f"<p>Overall: AP={overall.get('AP', 0.0):.3f}, AP50={overall.get('AP50', 0.0):.3f}, "
            f"AP75={overall.get('AP75', 0.0):.3f}, AR100={overall.get('AR100', 0.0):.3f}</p>"
            "<table><thead><tr><th>Class</th><th>AP</th><th>AP50</th><th>AR100</th></tr></thead>"
            f"<tbody>{per_class_rows}</tbody></table>"
            "</div>"
        )

    analysis_block = ""
    if prediction_analysis:
        rows = "".join(
            "<tr>"
            f"<td>{escape(item['class_name'])}</td>"
            f"<td>{item['precision']:.4f}</td>"
            f"<td>{item['recall']:.4f}</td>"
            f"<td>{item['f1']:.4f}</td>"
            f"<td>{item['missed']}</td>"
            f"<td>{item['background_fp']}</td>"
            f"<td>{item['wrong_class']}</td>"
            "</tr>"
            for item in prediction_analysis["class_metrics"]
        )
        analysis_block = (
            "<div class='card'>"
            "<h2>Prediction Analysis</h2>"
            f"<p>Visualization threshold: score &gt;= {args.score_thr:.2f}. Matching threshold: IoU &gt;= {args.iou_thr:.2f}.</p>"
            "<table>"
            "<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Missed</th><th>BG FP</th><th>Wrong Class</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grape Picking Baseline Report</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Segoe UI", sans-serif;
      background: #f7f7f5;
      color: #1f2933;
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    .hero {{
      background: linear-gradient(135deg, #264653, #2a9d8f);
      color: white;
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .card {{
      background: white;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid #e6e6e6;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #ececec;
      text-align: left;
      font-size: 14px;
    }}
    .muted {{
      color: #52606d;
    }}
    .links a {{
      margin-right: 16px;
      color: #264653;
      text-decoration: none;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>Grape + Picking Baseline Report</h1>
    <p class="muted" style="color: rgba(255,255,255,0.85);">
      Best val AP50: {training_summary['best_ap50']:.4f} at epoch {training_summary['best_ap50_epoch']} |
      Best val AP: {training_summary['best_ap']:.4f} at epoch {training_summary['best_ap_epoch']} |
      Final val AP/AP50: {training_summary['final_metrics']['AP']:.4f} / {training_summary['final_metrics']['AP50']:.4f}
    </p>
    <p class="muted" style="color: rgba(255,255,255,0.85);">
      Report assets: <code>summary.json</code>, <code>class_focus_report.md</code>, <code>picking_report.md</code>, <code>predictions_test.json</code>, <code>per_image_test_summary.json</code>
    </p>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Split Summary</h2>
      <table>
        <thead><tr><th>Split</th><th>Images</th><th>Annotations</th></tr></thead>
        <tbody>
          <tr><td>train</td><td>{train_stats['images']}</td><td>{train_stats['annotations']}</td></tr>
          <tr><td>valid</td><td>{valid_stats['images']}</td><td>{valid_stats['annotations']}</td></tr>
          <tr><td>test</td><td>{test_stats['images']}</td><td>{test_stats['annotations']}</td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h2>Train Label Stats</h2>
      <table>
        <thead><tr><th>Class</th><th>Boxes</th><th>Median W</th><th>Median H</th><th>Median Area Ratio</th></tr></thead>
        <tbody>
          {''.join(class_rows)}
        </tbody>
      </table>
    </div>
  </div>

  {eval_block}
  {analysis_block}

  <div class="card links" style="margin-top: 20px;">
    <h2>Report Files</h2>
    <a href="summary.json">summary.json</a>
    <a href="per_class_detailed_metrics.json">per_class_detailed_metrics.json</a>
    <a href="class_focus_report.md">class_focus_report.md</a>
    <a href="grape_report.md">grape_report.md</a>
    <a href="picking_report.md">picking_report.md</a>
    <a href="thesis_ready_summary.md">thesis_ready_summary.md</a>
    <a href="predictions_test.json">predictions_test.json</a>
    <a href="per_image_test_summary.json">per_image_test_summary.json</a>
    <a href="test_eval.txt">test_eval.txt</a>
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Training Curves</h2>
      <img src="training_curves.png" alt="training curves">
  </div>

  <div class="grid" style="margin-top: 20px;">
    <div class="card">
      <h2>Dataset Overview</h2>
      <img src="train_labels_overview.png" alt="dataset overview">
    </div>
    <div class="card">
      <h2>Split Class Balance</h2>
      <img src="split_class_balance.png" alt="split class balance">
    </div>
  </div>

  <div class="grid" style="margin-top: 20px;">
    <div class="card">
      <h2>Confusion Matrix</h2>
      <img src="confusion_matrix.png" alt="confusion matrix">
    </div>
    <div class="card">
      <h2>Test Class Metrics</h2>
      <img src="test_class_metrics.png" alt="test class metrics">
    </div>
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Error Breakdown</h2>
    <img src="error_breakdown.png" alt="error breakdown">
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Train Samples (Ground Truth)</h2>
    <img src="samples_train.jpg" alt="train samples">
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Valid Samples (Ground Truth)</h2>
    <img src="samples_valid.jpg" alt="valid samples">
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Test Samples (Ground Truth)</h2>
    <img src="samples_test.jpg" alt="test samples">
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Representative Test Cases: Ground Truth vs Prediction</h2>
    <img src="test_gt_vs_pred.jpg" alt="test gt vs pred">
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Best Test Cases</h2>
    <img src="test_best_cases.jpg" alt="best test cases">
  </div>

  <div class="card" style="margin-top: 20px;">
    <h2>Failure Cases</h2>
    <img src="test_failure_cases.jpg" alt="failure cases">
  </div>
</body>
</html>
"""
    (report_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)

    dataset_root = args.dataset_root.resolve()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    report_dir = args.report_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if not args.full_report:
        prune_report_dir(report_dir)

    log_path = resolve_run_artifact(run_dir, "log.txt", "logs/log.txt")
    records, log_trim_note = load_log_records(log_path)
    training_summary = build_training_summary(records)
    dataset_stats = collect_dataset_stats(dataset_root, ann_file_name=args.ann_file_name)
    categories = dataset_stats["categories"]
    args.report_recovery_note_zh = log_trim_note

    export_results_csv(records, report_dir / "results.csv")
    plot_training_curves(records, report_dir / "training_curves.png")
    plot_results_overview(records, report_dir / "results_overview.png")

    split_indexes = {}
    for split in ("train", "valid", "test"):
        payload = dataset_stats["payloads"][split]
        images_by_id, gt_by_image = build_split_index(payload, categories)
        split_indexes[split] = (images_by_id, gt_by_image)
        if args.full_report:
            selected_ids = choose_sample_images(payload, samples_per_split=args.samples_per_split, seed=args.seed + len(split))
            build_gt_sample_grid(
                dataset_root=dataset_root,
                split=split,
                images_by_id=images_by_id,
                gt_by_image=gt_by_image,
                categories=categories,
                output_path=report_dir / f"samples_{split}.jpg",
                selected_image_ids=selected_ids,
            )

    if args.full_report:
        plot_train_label_overview(dataset_stats, report_dir / "train_labels_overview.png")
        plot_split_class_balance(dataset_stats, categories, report_dir / "split_class_balance.png")

    eval_summary = None
    eval_text = "Test eval was skipped."
    class_best_validation_known = None
    args.eval_source_note_zh = None
    if args.reuse_existing_eval:
        eval_test_path = resolve_eval_artifact(run_dir, "test")
        eval_summary = load_eval_summary_from_artifact(eval_test_path, categories)
        eval_text = (
            "Recovered from existing test evaluation artifact.\n"
            f"Source: {eval_test_path.resolve()}\n"
            "This report reuses the original saved eval result instead of re-running test-only evaluation."
        )
        args.eval_source_note_zh = (
            "本次 report 直接复用了训练完成时已保存的 test 评估结果，"
            "没有再次调用当前 checkpoint 重跑 test-only。"
        )
        args.class_best_validation_known = None
    elif not args.skip_eval:
        eval_summary, eval_text = run_test_eval(
            config_path=config_path,
            dataset_root=dataset_root,
            ann_file_name=args.ann_file_name,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        class_best_validation_known = collect_class_best_validation_known(
            config_path=config_path,
            dataset_root=dataset_root,
            ann_file_name=args.ann_file_name,
            run_dir=run_dir,
            training_summary=training_summary,
            args=args,
        )
        args.class_best_validation_known = class_best_validation_known
    if args.full_report:
        save_text(report_dir / "test_eval.txt", eval_text)

    prediction_analysis = None
    predictions_by_image: dict[int, list[dict]] = {}
    if not args.skip_predictions and checkpoint_path.exists():
        solver = init_inference_solver(
            config_path=config_path,
            dataset_root=dataset_root,
            ann_file_name=args.ann_file_name,
            checkpoint_path=checkpoint_path,
            report_dir=report_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        predictions_by_image = collect_predictions(solver, score_thr=args.score_thr)
        dist_utils.cleanup()
        shutil.rmtree(report_dir / "_runtime", ignore_errors=True)

        test_images_by_id, test_gt_by_image = split_indexes["test"]
        prediction_analysis = analyze_predictions(
            images_by_id=test_images_by_id,
            gt_by_image=test_gt_by_image,
            pred_by_image=predictions_by_image,
            categories=categories,
            iou_thr=args.iou_thr,
        )

        image_summary_lookup = {
            item["image_id"]: item
            for item in prediction_analysis["image_summaries"]
        }
        save_json(report_dir / "per_image_test_summary.json", prediction_analysis["image_summaries"])
        export_predictions_json(
            images_by_id=test_images_by_id,
            predictions_by_image=predictions_by_image,
            categories=categories,
            output_path=report_dir / "predictions_test.json",
        )

        plot_test_class_metrics(
            class_metrics=prediction_analysis["class_metrics"],
            eval_summary=eval_summary,
            output_path=report_dir / "test_class_metrics.png",
        )
        plot_error_breakdown(
            class_metrics=prediction_analysis["class_metrics"],
            output_path=report_dir / "error_breakdown.png",
        )
        representative_ids = choose_sample_images(
            dataset_stats["payloads"]["test"],
            samples_per_split=args.comparison_count,
            seed=args.seed + 99,
        )
        build_comparison_grid(
            dataset_root=dataset_root,
            split="test",
            images_by_id=test_images_by_id,
            gt_by_image=test_gt_by_image,
            pred_by_image=predictions_by_image,
            categories=categories,
            selected_image_ids=representative_ids,
            image_summaries=image_summary_lookup,
            output_path=report_dir / "test_gt_vs_pred.jpg",
        )

        if args.full_report:
            plot_confusion_matrix(
                confusion=prediction_analysis["confusion_matrix"],
                categories=categories,
                output_path=report_dir / "confusion_matrix.png",
            )
            best_case_ids = choose_best_case_ids(prediction_analysis["image_summaries"], count=args.best_case_count)
            build_comparison_grid(
                dataset_root=dataset_root,
                split="test",
                images_by_id=test_images_by_id,
                gt_by_image=test_gt_by_image,
                pred_by_image=predictions_by_image,
                categories=categories,
                selected_image_ids=best_case_ids,
                image_summaries=image_summary_lookup,
                output_path=report_dir / "test_best_cases.jpg",
            )

            failure_case_ids = choose_failure_case_ids(prediction_analysis["image_summaries"], count=args.failure_count)
            build_comparison_grid(
                dataset_root=dataset_root,
                split="test",
                images_by_id=test_images_by_id,
                gt_by_image=test_gt_by_image,
                pred_by_image=predictions_by_image,
                categories=categories,
                selected_image_ids=failure_case_ids,
                image_summaries=image_summary_lookup,
                output_path=report_dir / "test_failure_cases.jpg",
            )
    else:
        for name in ("predictions_test.json", "per_image_test_summary.json"):
            path = report_dir / name
            if path.exists():
                path.unlink()
        plot_test_class_metrics(
            class_metrics=[],
            eval_summary=eval_summary,
            output_path=report_dir / "test_class_metrics.png",
        )
        plot_error_breakdown(
            class_metrics=[],
            output_path=report_dir / "error_breakdown.png",
        )
        save_placeholder_image(
            report_dir / "test_gt_vs_pred.jpg",
            "Prediction comparison is unavailable in recovery mode.",
            "The original 100-epoch checkpoint was overwritten after an accidental resume, so per-image visualizations could not be reconstructed.",
        )

    build_summary_json(
        training_summary=training_summary,
        dataset_stats=dataset_stats,
        eval_summary=eval_summary,
        prediction_analysis=prediction_analysis,
        class_best_validation_known=class_best_validation_known,
        args=args,
        output_path=report_dir / "summary.json",
    )
    if args.full_report:
        build_class_focus_reports(
            report_dir=report_dir,
            training_summary=training_summary,
            eval_summary=eval_summary,
            prediction_analysis=prediction_analysis,
            args=args,
        )
        build_thesis_summary_markdown(
            report_dir=report_dir,
            dataset_stats=dataset_stats,
            training_summary=training_summary,
            eval_summary=eval_summary,
            prediction_analysis=prediction_analysis,
            args=args,
        )
        build_html_report(
            report_dir=report_dir,
            training_summary=training_summary,
            dataset_stats=dataset_stats,
            eval_summary=eval_summary,
            prediction_analysis=prediction_analysis,
            args=args,
        )

    print(f"[done] report written to: {report_dir}")
    if args.full_report:
        print(f"[open]  {report_dir / 'index.html'}")


if __name__ == "__main__":
    main()

