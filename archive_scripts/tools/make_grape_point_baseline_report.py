from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.solver.det_engine import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact report for grape-point baseline runs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_baseline.yml")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_log_records(log_path: Path) -> list[dict]:
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return sorted(records, key=lambda item: int(item["epoch"]))


def safe_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float(default)


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


def build_epoch_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        bbox = rec.get("test_coco_eval_bbox", [])
        per_class = rec.get("test_coco_eval_bbox_per_class", {}).get("grape", {})
        point = rec.get("test_grape_point_metrics", {})
        rows.append(
            {
                "epoch": int(rec["epoch"]),
                "train_lr": safe_float(rec.get("train_lr")),
                "train_loss": safe_float(rec.get("train_loss")),
                "train_loss_bbox": safe_float(rec.get("train_loss_bbox")),
                "train_loss_has_picking": safe_float(rec.get("train_loss_has_picking")),
                "train_loss_picking_offset": safe_float(rec.get("train_loss_picking_offset")),
                "valid_grape_AP": safe_float(per_class.get("AP", bbox[0] if len(bbox) > 0 else 0.0)),
                "valid_grape_AP50": safe_float(per_class.get("AP50", bbox[1] if len(bbox) > 1 else 0.0)),
                "valid_grape_AR100": safe_float(per_class.get("AR100", bbox[8] if len(bbox) > 8 else 0.0)),
                "valid_has_picking_precision": safe_float(point.get("has_picking_precision")),
                "valid_has_picking_recall": safe_float(point.get("has_picking_recall")),
                "valid_has_picking_f1": safe_float(point.get("has_picking_f1")),
                "valid_has_picking_false_positive": int(point.get("has_picking_false_positive", 0)),
                "valid_has_picking_false_negative": int(point.get("has_picking_false_negative", 0)),
                "valid_point_pair_count": int(point.get("point_pair_count", 0)),
                "valid_point_mae_x_px": safe_float(point.get("point_mae_x_px")),
                "valid_point_mae_y_px": safe_float(point.get("point_mae_y_px")),
                "valid_point_mean_l2_px": safe_float(point.get("point_mean_l2_px")),
            }
        )
    return rows


def best_epoch(rows: list[dict], key: str, mode: str = "max") -> dict | None:
    if not rows:
        return None
    valid_rows = [row for row in rows if not math.isnan(safe_float(row.get(key)))]
    if not valid_rows:
        return None
    if mode == "max":
        return max(valid_rows, key=lambda item: safe_float(item[key]))
    return min(valid_rows, key=lambda item: safe_float(item[key]))


def mean_std(rows: list[dict], key: str, window: int = 10) -> dict:
    tail = rows[-window:] if len(rows) >= window else rows
    values = np.asarray([safe_float(row.get(key)) for row in tail], dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def write_results_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(rows: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in rows], label="train_loss", color="#1f77b4")
    axes[0, 0].plot(epochs, [row["train_loss_has_picking"] for row in rows], label="has_picking_loss", color="#ff7f0e")
    axes[0, 0].plot(epochs, [row["train_loss_picking_offset"] for row in rows], label="point_offset_loss", color="#2ca02c")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, [row["valid_grape_AP"] for row in rows], label="grape AP", color="#1f77b4")
    axes[0, 1].plot(epochs, [row["valid_grape_AP50"] for row in rows], label="grape AP50", color="#17becf")
    axes[0, 1].set_title("Valid Grape Detection")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, [row["valid_has_picking_precision"] for row in rows], label="precision", color="#9467bd")
    axes[1, 0].plot(epochs, [row["valid_has_picking_recall"] for row in rows], label="recall", color="#d62728")
    axes[1, 0].plot(epochs, [row["valid_has_picking_f1"] for row in rows], label="F1", color="#2ca02c")
    axes[1, 0].set_title("Valid has_picking")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, [row["valid_point_mean_l2_px"] for row in rows], label="mean L2 px", color="#8c564b")
    axes[1, 1].plot(epochs, [row["valid_point_mae_x_px"] for row in rows], label="MAE x", color="#e377c2")
    axes[1, 1].plot(epochs, [row["valid_point_mae_y_px"] for row in rows], label="MAE y", color="#7f7f7f")
    axes[1, 1].set_title("Valid Point Error")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results_overview(rows: list[dict], test_stats: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[:, :2])
    epochs = [row["epoch"] for row in rows]
    ax1.plot(epochs, [row["valid_grape_AP"] for row in rows], label="valid grape AP", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, [row["valid_has_picking_f1"] for row in rows], label="valid has_picking F1", color="#2ca02c", linewidth=2)
    ax1.plot(epochs, [row["valid_point_mean_l2_px"] for row in rows], label="valid point mean L2 px", color="#d62728", linewidth=2)
    ax1.set_title("Point Baseline Overview")
    ax1.set_xlabel("Epoch")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    grape = test_stats["coco_eval_bbox_per_class"]["grape"]
    point = test_stats["grape_point_metrics"]
    ax2.bar(
        ["AP", "AP50", "AR100", "F1"],
        [safe_float(grape["AP"]), safe_float(grape["AP50"]), safe_float(grape["AR100"]), safe_float(point["has_picking_f1"])],
        color=["#1f77b4", "#17becf", "#9467bd", "#2ca02c"],
    )
    ax2.set_title("Test Scores")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.bar(
        ["L2 px", "MAE x", "MAE y"],
        [
            safe_float(point["point_mean_l2_px"]),
            safe_float(point["point_mae_x_px"]),
            safe_float(point["point_mae_y_px"]),
        ],
        color=["#8c564b", "#e377c2", "#7f7f7f"],
    )
    ax3.set_title("Test Point Error")
    ax3.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_test_eval(config_path: Path, checkpoint_path: Path, batch_size: int, num_workers: int, device: str) -> dict:
    dist_utils.setup_distributed(seed=0)
    with tempfile.TemporaryDirectory(prefix="point_report_eval_", dir=str(REPO_ROOT / "outputs")) as tmp_dir:
        cfg = YAMLConfig(
            str(config_path),
            resume=str(checkpoint_path),
            device=device,
            use_amp=False,
            output_dir=tmp_dir,
        )
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = "./dataset/test"
        cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = "./dataset/test/_annotations.grape_point.json"
        cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = batch_size
        cfg.yaml_cfg["val_dataloader"]["num_workers"] = num_workers

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        module = solver.ema.module if solver.ema else solver.model
        stats, _ = evaluate(module, solver.criterion, solver.postprocessor, solver.val_dataloader, solver.evaluator, solver.device)
        solver.cleanup()
    dist_utils.cleanup()
    return stats


def find_reference_bbox_summary() -> dict | None:
    path = REPO_ROOT / "outputs" / "baseline_20260407" / "report" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(args: argparse.Namespace, rows: list[dict], test_stats: dict) -> dict:
    train_stats = summarize_dataset(args.dataset_root / "train" / "_annotations.grape_point.json")
    valid_stats = summarize_dataset(args.dataset_root / "valid" / "_annotations.grape_point.json")
    test_dataset_stats = summarize_dataset(args.dataset_root / "test" / "_annotations.grape_point.json")

    grape_best = best_epoch(rows, "valid_grape_AP", mode="max")
    grape_ap50_best = best_epoch(rows, "valid_grape_AP50", mode="max")
    has_f1_best = best_epoch(rows, "valid_has_picking_f1", mode="max")
    point_l2_best = best_epoch([row for row in rows if row["valid_point_pair_count"] > 0], "valid_point_mean_l2_px", mode="min")
    last_row = rows[-1]

    reference_bbox_summary = find_reference_bbox_summary()
    reference_compare = None
    if reference_bbox_summary is not None:
        ref_grape = reference_bbox_summary.get("test_eval", {}).get("per_class", {}).get("grape", {})
        reference_compare = {
            "run_dir": str((REPO_ROOT / "outputs" / "baseline_20260407").resolve()),
            "grape_bbox_test_AP": ref_grape.get("AP"),
            "grape_bbox_test_AP50": ref_grape.get("AP50"),
            "grape_bbox_test_AR100": ref_grape.get("AR100"),
            "note_zh": "这里只能把 grape 检测能力和旧 bbox-baseline 对齐比较；新的 picking 点任务不再和旧的 picking bbox AP 直接可比。",
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_mode": "minimal_point",
        "report_file_guide_zh": {
            "summary.json": "点任务主汇总。优先看这里，包含 valid 最佳轮次、test 结果、grape 检测和 picking 点任务的核心结论。",
            "results.csv": "逐 epoch 指标表。主要看 valid_grape_AP、valid_has_picking_F1、valid_point_mean_l2_px。",
            "training_curves.png": "训练与验证曲线图。看是否收敛、后期是否还在明显上涨。",
            "results_overview.png": "point baseline 总览图。左侧是 epoch 曲线，右侧是 test 核心指标和点误差。"
        },
        "task_definition_zh": {
            "grape": "继续做 bbox 检测",
            "picking": "不再做独立 bbox 检测，改为与 grape 绑定的 has_picking + point offset 回归",
        },
        "config": str(args.config.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_name": args.checkpoint.name,
        "device": args.device,
        "data_summary": {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_dataset_stats,
        },
        "validation_note_zh": "log.txt 中的 test_* 字段其实对应训练过程里的验证集(valid)评估，不是 test 集。",
        "best_validation": {
            "grape_ap_epoch": grape_best["epoch"] if grape_best else None,
            "grape_ap": grape_best["valid_grape_AP"] if grape_best else None,
            "grape_ap50_epoch": grape_ap50_best["epoch"] if grape_ap50_best else None,
            "grape_ap50": grape_ap50_best["valid_grape_AP50"] if grape_ap50_best else None,
            "has_picking_f1_epoch": has_f1_best["epoch"] if has_f1_best else None,
            "has_picking_f1": has_f1_best["valid_has_picking_f1"] if has_f1_best else None,
            "point_l2_best_epoch": point_l2_best["epoch"] if point_l2_best else None,
            "point_mean_l2_px": point_l2_best["valid_point_mean_l2_px"] if point_l2_best else None,
        },
        "last_validation": last_row,
        "convergence_last10": {
            "valid_grape_AP": mean_std(rows, "valid_grape_AP", window=10),
            "valid_has_picking_F1": mean_std(rows, "valid_has_picking_f1", window=10),
            "valid_point_mean_l2_px": mean_std(rows, "valid_point_mean_l2_px", window=10),
        },
        "test_eval": {
            "grape_detection": test_stats.get("coco_eval_bbox_per_class", {}).get("grape", {}),
            "grape_bbox_overall": {
                "AP": safe_float(test_stats.get("coco_eval_bbox", [0.0])[0] if test_stats.get("coco_eval_bbox") else 0.0),
                "AP50": safe_float(test_stats.get("coco_eval_bbox", [0.0, 0.0])[1] if test_stats.get("coco_eval_bbox") else 0.0),
                "AR100": safe_float(test_stats.get("coco_eval_bbox", [0.0] * 9)[8] if test_stats.get("coco_eval_bbox") else 0.0),
            },
            "picking_point": test_stats.get("grape_point_metrics", {}),
        },
        "reference_bbox_baseline": reference_compare,
    }
    return summary


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

    if args.checkpoint is None:
        for candidate in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
            p = args.run_dir / candidate
            if p.exists():
                args.checkpoint = p
                break
        if args.checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {args.run_dir}")
    args.checkpoint = args.checkpoint.resolve()

    rows = build_epoch_rows(load_log_records(log_path))
    test_stats = run_test_eval(args.config, args.checkpoint, args.batch_size, args.num_workers, args.device)

    summary = build_summary(args, rows, test_stats)

    results_csv = args.report_dir / "results.csv"
    summary_json = args.report_dir / "summary.json"
    training_curves_png = args.report_dir / "training_curves.png"
    results_overview_png = args.report_dir / "results_overview.png"

    write_results_csv(rows, results_csv)
    plot_training_curves(rows, training_curves_png)
    plot_results_overview(rows, test_stats, results_overview_png)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[report] wrote {summary_json}")
    print(f"[report] wrote {results_csv}")
    print(f"[report] wrote {training_curves_png}")
    print(f"[report] wrote {results_overview_png}")


if __name__ == "__main__":
    main()
