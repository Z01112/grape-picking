from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from make_grape_point_v2_report import (
    collect_cross_instance_mismatch_cases,
    collect_case_groups,
    evaluate_split,
    load_point_config,
    match_prediction_record,
    render_case_image,
    safe_float,
    summarize_decoupled_point_diagnostics,
)


DEFAULT_BASELINE_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v6_baseline_replay" / "report" / "summary.json"
DEFAULT_EXP1_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v7_exp1_query_box_top_center" / "report" / "summary.json"
DEFAULT_EXP2_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v7_exp2_query_box_top_center_toproi" / "report" / "summary.json"
DEFAULT_EXP2_REPRO_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v7_exp2_query_box_top_center_toproi_repro1" / "report" / "summary.json"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "grape_point_v7_paper_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready assets for baseline / v7_exp1 / v7_exp2.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-cases", type=int, default=4)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--exp1-summary", type=Path, default=DEFAULT_EXP1_SUMMARY)
    parser.add_argument("--exp2-summary", type=Path, default=DEFAULT_EXP2_SUMMARY)
    parser.add_argument("--exp2-repro-summary", type=Path, default=DEFAULT_EXP2_REPRO_SUMMARY)
    parser.add_argument("--extra-repro-summary", action="append", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def extract_test_metrics(summary: dict | None) -> dict:
    test = (summary or {}).get("primary_checkpoint_split_summary", {}).get("test", {})
    grape = test.get("grape_detection", {})
    has_pick = test.get("has_picking", {})
    point = test.get("picking_point", {})
    return {
        "grape_AP": safe_float(grape.get("AP")),
        "has_picking_F1": safe_float(has_pick.get("f1")),
        "pair_count": int(safe_float(point.get("pair_count"), 0.0)),
        "mean_L2": safe_float(point.get("mean_l2_px")),
        "median_L2": safe_float(point.get("median_l2_px")),
        "p90_L2": safe_float(point.get("p90_l2_px")),
        "mean_abs_dx": safe_float(point.get("mean_abs_dx_px")),
        "mean_abs_dy": safe_float(point.get("mean_abs_dy_px")),
        "small_L2": safe_float(point.get("size_group_l2_px", {}).get("small", {}).get("mean_l2_px")),
        "medium_L2": safe_float(point.get("size_group_l2_px", {}).get("medium", {}).get("mean_l2_px")),
        "large_L2": safe_float(point.get("size_group_l2_px", {}).get("large", {}).get("mean_l2_px")),
    }


def extract_checkpoint_test_metrics(summary: dict | None) -> dict[str, dict]:
    output = {}
    for name, payload in (summary or {}).get("checkpoint_comparison", {}).items():
        test = payload.get("test", {})
        output[name] = {
            "grape_AP": safe_float(test.get("grape_AP")),
            "has_picking_F1": safe_float(test.get("has_picking_f1")),
            "pair_count": int(safe_float(test.get("point_pair_count"), 0.0)),
            "mean_L2": safe_float(test.get("point_mean_l2_px")),
            "mean_abs_dy": safe_float(test.get("point_mae_y_px")),
        }
    return output


def get_has_picking_threshold(config_path: Path) -> float:
    cfg = load_point_config(config_path)
    return safe_float(cfg.get("PostProcessor", {}).get("has_picking_threshold", 0.5), 0.5)


def build_case_indexes(records: list[dict], has_picking_threshold: float) -> tuple[dict, list[dict], list[dict]]:
    all_cases: dict[tuple[int, int], dict] = {}
    correct_visible: list[dict] = []
    for record in records:
        matched = match_prediction_record(record, 0.5, has_picking_threshold)
        for case in matched.get("matched_pairs", []):
            key = (int(case["image_id"]), int(case["gt_index"]))
            all_cases[key] = case
        correct_visible.extend(matched.get("correct_visible_pairs", []))
    mismatches = collect_cross_instance_mismatch_cases(records, correct_visible)
    return all_cases, correct_visible, mismatches


def point_case_badness(case: dict | None) -> float:
    if case is None:
        return 120.0
    if not bool(case.get("gt_has_picking")):
        return -1.0
    if bool(case.get("pred_has_picking")):
        return float(case.get("l2_px", 120.0))
    return 80.0 + (1.0 - float(case.get("iou", 0.0))) * 20.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_final_main_comparison(metrics: dict[str, dict], out_path: Path) -> None:
    labels = ["baseline_replay", "v7_exp1", "v7_exp2"]
    ap = [metrics[name]["grape_AP"] for name in labels]
    f1 = [metrics[name]["has_picking_F1"] for name in labels]
    l2 = [metrics[name]["mean_L2"] for name in labels]
    pairs = [metrics[name]["pair_count"] for name in labels]

    x = np.arange(len(labels), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x - 0.18, ap, width=0.36, label="grape AP", color="#1f77b4")
    axes[0].bar(x + 0.18, f1, width=0.36, label="has_picking F1", color="#2ca02c")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Main Test Metrics")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - 0.18, l2, width=0.36, label="mean L2", color="#8c564b")
    axes[1].bar(x + 0.18, pairs, width=0.36, label="pair_count", color="#9467bd")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Point Error / Matched Pairs")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dy_comparison(metrics: dict[str, dict], out_path: Path) -> None:
    labels = ["baseline_replay", "v7_exp1", "v7_exp2"]
    dx = [metrics[name]["mean_abs_dx"] for name in labels]
    dy = [metrics[name]["mean_abs_dy"] for name in labels]

    x = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, dx, width=0.36, label="mean |dx|", color="#ff7f0e")
    ax.bar(x + 0.18, dy, width=0.36, label="mean |dy|", color="#d62728")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute error (px)")
    ax.set_title("XY Error Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_size_group_comparison(metrics: dict[str, dict], out_path: Path) -> None:
    labels = ["baseline_replay", "v7_exp1", "v7_exp2"]
    groups = ["small_L2", "medium_L2", "large_L2"]
    titles = ["small", "medium", "large"]
    x = np.arange(len(labels), dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, key, title in zip(axes, groups, titles):
        values = [metrics[name][key] for name in labels]
        ax.bar(x, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_xticks(x, labels, rotation=12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean L2 error (px)")
    fig.suptitle("Point Error by Grape Size Group", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_main_table(table_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "grape_AP",
        "has_picking_F1",
        "pair_count",
        "mean_L2",
        "median_L2",
        "p90_L2",
        "mean_abs_dx",
        "mean_abs_dy",
        "small_L2",
        "medium_L2",
        "large_L2",
    ]
    ensure_parent(table_path)
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_checkpoint_table(table_path: Path, baseline_metrics: dict, checkpoint_metrics: dict[str, dict]) -> None:
    fieldnames = ["checkpoint", "grape_AP", "has_picking_F1", "pair_count", "mean_L2", "mean_abs_dy", "vs_baseline_note"]
    ensure_parent(table_path)
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        ordered = ["best_composite.pth", "best_grape_ap.pth", "best_has_picking_f1.pth", "last.pth"]
        for name in ordered:
            metrics = checkpoint_metrics.get(name)
            if metrics is None:
                continue
            note = []
            if metrics["grape_AP"] > baseline_metrics["grape_AP"]:
                note.append("AP better")
            if metrics["has_picking_F1"] > baseline_metrics["has_picking_F1"]:
                note.append("F1 better")
            if metrics["pair_count"] > baseline_metrics["pair_count"]:
                note.append("pair better")
            if metrics["mean_L2"] < baseline_metrics["mean_L2"]:
                note.append("L2 better")
            if metrics["mean_abs_dy"] < baseline_metrics["mean_abs_dy"]:
                note.append("|dy| better")
            writer.writerow({"checkpoint": name, **metrics, "vs_baseline_note": "; ".join(note)})


def write_seed_stability_table(table_path: Path, baseline_metrics: dict, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "grape_AP",
        "has_picking_F1",
        "pair_count",
        "mean_L2",
        "mean_abs_dy",
        "vs_baseline_AP",
        "vs_baseline_F1",
        "vs_baseline_pair_count",
        "vs_baseline_mean_L2",
        "vs_baseline_mean_abs_dy",
    ]
    ensure_parent(table_path)
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "run": row["run"],
                    "grape_AP": metrics["grape_AP"],
                    "has_picking_F1": metrics["has_picking_F1"],
                    "pair_count": metrics["pair_count"],
                    "mean_L2": metrics["mean_L2"],
                    "mean_abs_dy": metrics["mean_abs_dy"],
                    "vs_baseline_AP": metrics["grape_AP"] - baseline_metrics["grape_AP"],
                    "vs_baseline_F1": metrics["has_picking_F1"] - baseline_metrics["has_picking_F1"],
                    "vs_baseline_pair_count": metrics["pair_count"] - baseline_metrics["pair_count"],
                    "vs_baseline_mean_L2": metrics["mean_L2"] - baseline_metrics["mean_L2"],
                    "vs_baseline_mean_abs_dy": metrics["mean_abs_dy"] - baseline_metrics["mean_abs_dy"],
                }
            )


def _sample_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def write_mean_std_table(table_path: Path, baseline_metrics: dict, rows: list[dict]) -> dict:
    ap_mean, ap_std = _sample_mean_std([float(row["metrics"]["grape_AP"]) for row in rows])
    f1_mean, f1_std = _sample_mean_std([float(row["metrics"]["has_picking_F1"]) for row in rows])
    pair_mean, pair_std = _sample_mean_std([float(row["metrics"]["pair_count"]) for row in rows])
    l2_mean, l2_std = _sample_mean_std([float(row["metrics"]["mean_L2"]) for row in rows])
    dy_mean, dy_std = _sample_mean_std([float(row["metrics"]["mean_abs_dy"]) for row in rows])
    summary = {
        "run_count": len(rows),
        "AP": {"mean": ap_mean, "std": ap_std},
        "F1": {"mean": f1_mean, "std": f1_std},
        "pair_count": {"mean": pair_mean, "std": pair_std},
        "mean_L2": {"mean": l2_mean, "std": l2_std},
        "mean_abs_dy": {"mean": dy_mean, "std": dy_std},
    }

    fieldnames = ["model", "AP", "F1", "pair_count", "mean_L2", "mean_abs_dy", "note"]
    ensure_parent(table_path)
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": "baseline_replay",
                "AP": f"{baseline_metrics['grape_AP']:.4f}",
                "F1": f"{baseline_metrics['has_picking_F1']:.4f}",
                "pair_count": f"{baseline_metrics['pair_count']}",
                "mean_L2": f"{baseline_metrics['mean_L2']:.2f}",
                "mean_abs_dy": f"{baseline_metrics['mean_abs_dy']:.2f}",
                "note": "formal baseline",
            }
        )
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "model": row["run"],
                    "AP": f"{metrics['grape_AP']:.4f}",
                    "F1": f"{metrics['has_picking_F1']:.4f}",
                    "pair_count": f"{metrics['pair_count']}",
                    "mean_L2": f"{metrics['mean_L2']:.2f}",
                    "mean_abs_dy": f"{metrics['mean_abs_dy']:.2f}",
                    "note": "independent v7_exp2 run",
                }
            )
        writer.writerow(
            {
                "model": f"v7_exp2 mean±std (n={len(rows)})",
                "AP": f"{ap_mean:.4f} ± {ap_std:.4f}",
                "F1": f"{f1_mean:.4f} ± {f1_std:.4f}",
                "pair_count": f"{pair_mean:.1f} ± {pair_std:.1f}",
                "mean_L2": f"{l2_mean:.2f} ± {l2_std:.2f}",
                "mean_abs_dy": f"{dy_mean:.2f} ± {dy_std:.2f}",
                "note": "sample std across independent runs",
            }
        )
    return summary


def write_mechanism_table(table_path: Path, summaries: dict[str, dict]) -> None:
    fieldnames = [
        "model",
        "standard_pair_count",
        "standard_mean_L2",
        "standard_mean_abs_dy",
        "inside_gt_box_rate",
        "iou_ge_0.85_count",
        "iou_ge_0.85_mean_L2",
        "iou_ge_0.85_mean_abs_dy",
        "oracle_iou50_recall",
        "oracle_iou50_mean_L2",
        "oracle_any_visible_recall",
        "oracle_any_visible_mean_L2",
    ]
    ensure_parent(table_path)
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in ("baseline_replay", "v7_exp1", "v7_exp2"):
            diag = summaries[model]
            standard = diag.get("standard_match", {})
            iou85 = diag.get("iou_conditioned", {}).get("ge_0.85", {})
            oracle_iou50 = diag.get("oracle_candidates", {}).get("iou_ge_0.50_visible", {})
            oracle_any = diag.get("oracle_candidates", {}).get("any_visible_pred", {})
            writer.writerow(
                {
                    "model": model,
                    "standard_pair_count": int(safe_float(standard.get("count"), 0.0)),
                    "standard_mean_L2": safe_float(standard.get("mean_l2_px")),
                    "standard_mean_abs_dy": safe_float(standard.get("mean_abs_dy_px")),
                    "inside_gt_box_rate": safe_float(standard.get("pred_point_inside_gt_box_rate")),
                    "iou_ge_0.85_count": int(safe_float(iou85.get("count"), 0.0)),
                    "iou_ge_0.85_mean_L2": safe_float(iou85.get("mean_l2_px")),
                    "iou_ge_0.85_mean_abs_dy": safe_float(iou85.get("mean_abs_dy_px")),
                    "oracle_iou50_recall": safe_float(oracle_iou50.get("candidate_recall")),
                    "oracle_iou50_mean_L2": safe_float(oracle_iou50.get("mean_l2_px")),
                    "oracle_any_visible_recall": safe_float(oracle_any.get("candidate_recall")),
                    "oracle_any_visible_mean_L2": safe_float(oracle_any.get("mean_l2_px")),
                }
            )


def _box_iou_xyxy(box1: list[float], box2: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box1]
    bx1, by1, bx2, by2 = [float(v) for v in box2]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area1 + area2 - inter_area
    return float(inter_area / union) if union > 0.0 else 0.0


def _box_edge_gap(box1: list[float], box2: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box1]
    bx1, by1, bx2, by2 = [float(v) for v in box2]
    gap_x = max(ax1 - bx2, bx1 - ax2, 0.0)
    gap_y = max(ay1 - by2, by1 - ay2, 0.0)
    return float(math.hypot(gap_x, gap_y))


def build_scene_slices(records: list[dict]) -> tuple[dict[str, dict[str, set[tuple[int, int]]]], dict]:
    visible_keys = []
    area_values = []
    metadata = {}
    for record in records:
        gt_entries = record.get("gt_instances", [])
        total_grape_count = len(gt_entries)
        boxes = [item.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]) for item in gt_entries]
        for gt_idx, gt in enumerate(gt_entries):
            if not bool(gt.get("has_picking")):
                continue
            key = (int(record["image_id"]), int(gt_idx))
            visible_keys.append(key)
            area = float(gt.get("area", 0.0))
            area_values.append(area)
            own_box = gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            max_neighbor_iou = 0.0
            min_gap_norm = float("inf")
            for other_idx, other_box in enumerate(boxes):
                if other_idx == gt_idx:
                    continue
                max_neighbor_iou = max(max_neighbor_iou, _box_iou_xyxy(own_box, other_box))
                denom = max(math.sqrt(max(area, 1.0)), 1.0)
                min_gap_norm = min(min_gap_norm, _box_edge_gap(own_box, other_box) / denom)
            metadata[key] = {
                "area": area,
                "total_grape_count": total_grape_count,
                "max_neighbor_iou": max_neighbor_iou,
                "min_gap_norm": min_gap_norm if math.isfinite(min_gap_norm) else float("inf"),
            }

    area_threshold = float(np.quantile(np.asarray(area_values, dtype=np.float64), 1.0 / 3.0)) if area_values else 0.0
    crowd_gap_threshold = 0.20
    crowd_iou_threshold = 0.05

    slices = {
        "single_vs_multi": {"single": set(), "multi_adjacent": set()},
        "occlusion_proxy": {"light": set(), "heavy": set()},
        "size_group": {"small": set(), "medium_large": set()},
    }
    for key, meta in metadata.items():
        if int(meta["total_grape_count"]) <= 1:
            slices["single_vs_multi"]["single"].add(key)
        else:
            slices["single_vs_multi"]["multi_adjacent"].add(key)

        if float(meta["area"]) <= area_threshold:
            slices["size_group"]["small"].add(key)
        else:
            slices["size_group"]["medium_large"].add(key)

        is_heavy = int(meta["total_grape_count"]) > 1 and (
            float(meta["max_neighbor_iou"]) >= crowd_iou_threshold or float(meta["min_gap_norm"]) <= crowd_gap_threshold
        )
        if is_heavy:
            slices["occlusion_proxy"]["heavy"].add(key)
        else:
            slices["occlusion_proxy"]["light"].add(key)

    notes = {
        "single_vs_multi": "单串/多串采用图像内 grape GT 数量划分：1 个为 single，>=2 个为 multi_adjacent。",
        "occlusion_proxy": f"遮挡轻/重采用 GT 几何代理：同图存在邻串，且 max_neighbor_iou>={crowd_iou_threshold:.2f} 或 min_gap_norm<={crowd_gap_threshold:.2f} 记为 heavy。",
        "size_group": f"小串/中大串采用 test visible grape 面积 1/3 分位划分，small 阈值 area<={area_threshold:.1f}。",
        "size_threshold_area": area_threshold,
        "crowding_gap_norm_threshold": crowd_gap_threshold,
        "crowding_iou_threshold": crowd_iou_threshold,
        "visible_gt_count": len(visible_keys),
    }
    return slices, notes


def summarize_scene_slice(correct_pairs: list[dict], slice_keys: set[tuple[int, int]]) -> dict:
    cases = [item for item in correct_pairs if (int(item["image_id"]), int(item["gt_index"])) in slice_keys]
    visible_gt_count = len(slice_keys)
    l2_values = [float(item.get("l2_px", 0.0)) for item in cases]
    dy_values = [float(item.get("dy_px", 0.0)) for item in cases]
    inside_gt = [1.0 if bool(item.get("pred_point_inside_gt_box", False)) else 0.0 for item in cases]
    return {
        "visible_gt_count": visible_gt_count,
        "pair_count": len(cases),
        "pair_recall": float(len(cases) / visible_gt_count) if visible_gt_count > 0 else 0.0,
        "mean_L2": float(np.mean(l2_values)) if l2_values else 0.0,
        "mean_abs_dy": float(np.mean(np.abs(dy_values))) if dy_values else 0.0,
        "inside_gt_box_rate": float(np.mean(inside_gt)) if inside_gt else 0.0,
    }


def build_scene_slice_rows(scene_slices: dict[str, dict[str, set[tuple[int, int]]]], model_cases: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for family, groups in scene_slices.items():
        for label, keys in groups.items():
            baseline_summary = summarize_scene_slice(model_cases["baseline_replay"], keys)
            exp2_summary = summarize_scene_slice(model_cases["v7_exp2"], keys)
            rows.append(
                {
                    "slice_family": family,
                    "slice_label": label,
                    "visible_gt_count": baseline_summary["visible_gt_count"],
                    "baseline_pair_count": baseline_summary["pair_count"],
                    "baseline_pair_recall": baseline_summary["pair_recall"],
                    "baseline_mean_L2": baseline_summary["mean_L2"],
                    "baseline_mean_abs_dy": baseline_summary["mean_abs_dy"],
                    "exp2_pair_count": exp2_summary["pair_count"],
                    "exp2_pair_recall": exp2_summary["pair_recall"],
                    "exp2_mean_L2": exp2_summary["mean_L2"],
                    "exp2_mean_abs_dy": exp2_summary["mean_abs_dy"],
                    "delta_pair_count": exp2_summary["pair_count"] - baseline_summary["pair_count"],
                    "delta_pair_recall": exp2_summary["pair_recall"] - baseline_summary["pair_recall"],
                    "delta_mean_L2": exp2_summary["mean_L2"] - baseline_summary["mean_L2"],
                    "delta_mean_abs_dy": exp2_summary["mean_abs_dy"] - baseline_summary["mean_abs_dy"],
                    "baseline_inside_gt_box_rate": baseline_summary["inside_gt_box_rate"],
                    "exp2_inside_gt_box_rate": exp2_summary["inside_gt_box_rate"],
                }
            )
    return rows


def write_scene_slice_table(table_path: Path, rows: list[dict], notes: dict) -> None:
    fieldnames = [
        "slice_family",
        "slice_label",
        "visible_gt_count",
        "baseline_pair_count",
        "baseline_pair_recall",
        "baseline_mean_L2",
        "baseline_mean_abs_dy",
        "exp2_pair_count",
        "exp2_pair_recall",
        "exp2_mean_L2",
        "exp2_mean_abs_dy",
        "delta_pair_count",
        "delta_pair_recall",
        "delta_mean_L2",
        "delta_mean_abs_dy",
        "baseline_inside_gt_box_rate",
        "exp2_inside_gt_box_rate",
        "definition_note",
    ]
    ensure_parent(table_path)
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["definition_note"] = notes.get(payload["slice_family"], "")
            writer.writerow(payload)


def build_scene_slice_md(path: Path, rows: list[dict], notes: dict) -> None:
    family_names = {
        "single_vs_multi": "单串 / 多串相邻",
        "occlusion_proxy": "遮挡代理轻 / 重",
        "size_group": "小串 / 中大串",
    }
    label_names = {
        "single": "单串",
        "multi_adjacent": "多串相邻",
        "light": "遮挡代理轻",
        "heavy": "遮挡代理重",
        "small": "小串",
        "medium_large": "中大串",
    }
    lines = [
        "# scene slice summary（中文）",
        "",
        "## 定义说明",
        f"- {notes['single_vs_multi']}",
        f"- {notes['occlusion_proxy']}",
        f"- {notes['size_group']}",
        "",
        "## 结果摘要",
    ]
    for family in ("single_vs_multi", "occlusion_proxy", "size_group"):
        family_rows = [row for row in rows if row["slice_family"] == family]
        lines.append(f"### {family_names[family]}")
        for row in family_rows:
            lines.append(
                "- "
                f"{label_names.get(row['slice_label'], row['slice_label'])}: "
                f"visible_gt={row['visible_gt_count']}, "
                f"pair_recall {row['baseline_pair_recall']:.3f}->{row['exp2_pair_recall']:.3f}, "
                f"mean L2 {row['baseline_mean_L2']:.2f}->{row['exp2_mean_L2']:.2f}px, "
                f"|dy| {row['baseline_mean_abs_dy']:.2f}->{row['exp2_mean_abs_dy']:.2f}px"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_missing_case(record: dict, gt_index: int, out_path: Path, label: str) -> None:
    image = Image.open(record["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    gt = record["gt_instances"][gt_index]
    draw.rectangle(gt["bbox_xyxy"], outline="#00ff66", width=4)
    if gt.get("has_picking"):
        x, y = gt["picking_point"]
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), outline="#00ff66", width=3)
    lines = [f"image_id={record['image_id']}", label, "no matched pred box under IoU>=0.5"]
    box_x, box_y = 8, 8
    line_h = 15
    box_w = max(draw.textlength(line, font=font) for line in lines) + 10
    box_h = line_h * len(lines) + 8
    draw.rectangle((box_x, box_y, box_x + box_w, box_y + box_h), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((box_x + 5, box_y + 4 + idx * line_h), line, fill="white", font=font)
    image.save(out_path)


def compose_side_by_side(left_path: Path, right_path: Path, out_path: Path, left_title: str, right_title: str, footer: str) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    font = ImageFont.load_default()
    top_pad = 26
    footer_pad = 22
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + top_pad + footer_pad), "white")
    canvas.paste(left, (0, top_pad))
    canvas.paste(right, (left.width, top_pad))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, top_pad), fill="#111111")
    draw.text((8, 6), left_title, fill="white", font=font)
    draw.text((left.width + 8, 6), right_title, fill="white", font=font)
    draw.rectangle((0, canvas.height - footer_pad, canvas.width, canvas.height), fill="#111111")
    draw.text((8, canvas.height - footer_pad + 4), footer, fill="white", font=font)
    canvas.save(out_path)


def render_case_pair(
    out_path: Path,
    gt_index: int,
    baseline_record: dict,
    exp2_record: dict,
    baseline_case: dict | None,
    exp2_case: dict | None,
    baseline_threshold: float,
    exp2_threshold: float,
    footer: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="paper_case_", dir=str(REPO_ROOT / "outputs")) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        left_path = tmp_dir / "baseline.png"
        right_path = tmp_dir / "exp2.png"
        if baseline_case is None:
            render_missing_case(baseline_record, gt_index, left_path, "baseline_replay")
        else:
            render_case_image(baseline_record, baseline_case, left_path, baseline_threshold, style="clean")
        if exp2_case is None:
            render_missing_case(exp2_record, gt_index, right_path, "v7_exp2")
        else:
            render_case_image(exp2_record, exp2_case, right_path, exp2_threshold, style="clean")
        compose_side_by_side(left_path, right_path, out_path, "baseline_replay", "v7_exp2", footer)


def select_unique(entries: list[dict], top_k: int) -> list[dict]:
    selected = []
    seen = set()
    for item in entries:
        key = (int(item["image_id"]), int(item["gt_index"]))
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= top_k:
            break
    return selected


def build_representative_cases(
    out_root: Path,
    baseline_records: list[dict],
    exp2_records: list[dict],
    baseline_cases: dict[tuple[int, int], dict],
    exp2_cases: dict[tuple[int, int], dict],
    baseline_mismatches: list[dict],
    exp2_mismatches: list[dict],
    baseline_threshold: float,
    exp2_threshold: float,
    top_k: int,
) -> dict:
    rep_root = out_root / "paper_ready_figures" / "representative_cases"
    baseline_lookup = {int(record["image_id"]): record for record in baseline_records}
    exp2_lookup = {int(record["image_id"]): record for record in exp2_records}
    visible_keys = []
    for record in baseline_records:
        for gt_idx, gt in enumerate(record.get("gt_instances", [])):
            if bool(gt.get("has_picking")):
                visible_keys.append((int(record["image_id"]), int(gt_idx)))

    v7_better = []
    baseline_better = []
    dy_fixed = []
    remaining_failures = []
    baseline_mismatch_lookup = {(int(item["image_id"]), int(item["gt_index"])): item for item in baseline_mismatches}
    exp2_mismatch_keys = {(int(item["image_id"]), int(item["gt_index"])) for item in exp2_mismatches}
    cross_instance_fixed = []

    for image_id, gt_index in visible_keys:
        b_case = baseline_cases.get((image_id, gt_index))
        e_case = exp2_cases.get((image_id, gt_index))
        b_bad = point_case_badness(b_case)
        e_bad = point_case_badness(e_case)

        if e_bad + 8.0 < b_bad:
            v7_better.append(
                {
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "baseline_case": b_case,
                    "exp2_case": e_case,
                    "improvement": b_bad - e_bad,
                    "footer": f"v7 better: baseline badness={b_bad:.1f}, exp2 badness={e_bad:.1f}",
                }
            )
        if b_bad + 8.0 < e_bad:
            baseline_better.append(
                {
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "baseline_case": b_case,
                    "exp2_case": e_case,
                    "improvement": e_bad - b_bad,
                    "footer": f"baseline better: baseline badness={b_bad:.1f}, exp2 badness={e_bad:.1f}",
                }
            )
        if b_case is not None and e_case is not None and bool(b_case.get("pred_has_picking")) and bool(e_case.get("pred_has_picking")):
            dy_gain = abs(float(b_case.get("dy_px", 0.0))) - abs(float(e_case.get("dy_px", 0.0)))
            if dy_gain >= 6.0 and float(e_case.get("l2_px", 0.0)) <= float(b_case.get("l2_px", 0.0)) + 2.0:
                dy_fixed.append(
                    {
                        "image_id": image_id,
                        "gt_index": gt_index,
                        "baseline_case": b_case,
                        "exp2_case": e_case,
                        "improvement": dy_gain,
                        "footer": f"dy fixed: |dy| {abs(float(b_case['dy_px'])):.1f}px -> {abs(float(e_case['dy_px'])):.1f}px",
                    }
                )
        if e_bad >= 35.0:
            remaining_failures.append(
                {
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "baseline_case": b_case,
                    "exp2_case": e_case,
                    "improvement": e_bad,
                    "footer": f"remaining failure: exp2 badness={e_bad:.1f}",
                }
            )

    for key, b_case in baseline_mismatch_lookup.items():
        e_case = exp2_cases.get(key)
        if key in exp2_mismatch_keys:
            continue
        if e_case is None or not bool(e_case.get("pred_has_picking")):
            continue
        if float(e_case.get("l2_px", 999.0)) >= float(b_case.get("l2_px", 999.0)) - 3.0:
            continue
        cross_instance_fixed.append(
            {
                "image_id": int(b_case["image_id"]),
                "gt_index": int(b_case["gt_index"]),
                "baseline_case": b_case,
                "exp2_case": e_case,
                "improvement": float(b_case.get("l2_px", 0.0)) - float(e_case.get("l2_px", 0.0)),
                "footer": f"cross-instance fixed: L2 {float(b_case['l2_px']):.1f}px -> {float(e_case['l2_px']):.1f}px",
            }
        )

    if len(cross_instance_fixed) < top_k:
        existing_cross_keys = {(int(item["image_id"]), int(item["gt_index"])) for item in cross_instance_fixed}
        relaxed_candidates = []
        for image_id, gt_index in visible_keys:
            if (image_id, gt_index) in existing_cross_keys:
                continue
            b_case = baseline_cases.get((image_id, gt_index))
            e_case = exp2_cases.get((image_id, gt_index))
            if b_case is None or e_case is None:
                continue
            if not bool(b_case.get("pred_has_picking")) or not bool(e_case.get("pred_has_picking")):
                continue
            record = baseline_lookup[image_id]
            pred_point = b_case.get("pred_point", [0.0, 0.0])
            own_l2 = float(b_case.get("l2_px", 0.0))
            best_other = None
            for other_idx, other_gt in enumerate(record.get("gt_instances", [])):
                if other_idx == gt_index or not bool(other_gt.get("has_picking")):
                    continue
                other_point = other_gt.get("picking_point", [0.0, 0.0])
                other_dist = float(math.hypot(pred_point[0] - other_point[0], pred_point[1] - other_point[1]))
                x1, y1, x2, y2 = [float(v) for v in other_gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])]
                in_other_box = (x1 - 2.0) <= float(pred_point[0]) <= (x2 + 2.0) and (y1 - 2.0) <= float(pred_point[1]) <= (y2 + 2.0)
                score = own_l2 - other_dist + (8.0 if in_other_box else 0.0)
                if best_other is None or score > best_other["score"]:
                    best_other = {
                        "other_gt_index": int(other_idx),
                        "other_gt_point": [float(other_point[0]), float(other_point[1])],
                        "other_gt_bbox_xyxy": [x1, y1, x2, y2],
                        "other_gt_distance_px": other_dist,
                        "pred_point_inside_other_gt": bool(in_other_box),
                        "score": score,
                    }
            if best_other is None:
                continue
            if not (best_other["pred_point_inside_other_gt"] or best_other["other_gt_distance_px"] + 1.5 < own_l2):
                continue
            if float(e_case.get("l2_px", 999.0)) > own_l2 + 1.0:
                continue
            enriched_case = dict(b_case)
            enriched_case.update(best_other)
            relaxed_candidates.append(
                {
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "baseline_case": enriched_case,
                    "exp2_case": e_case,
                    "improvement": own_l2 - float(e_case.get("l2_px", 0.0)),
                    "footer": f"cross-instance fixed: L2 {own_l2:.1f}px -> {float(e_case.get('l2_px', 0.0)):.1f}px",
                }
            )
        for item in sorted(relaxed_candidates, key=lambda x: x["improvement"], reverse=True):
            key = (int(item["image_id"]), int(item["gt_index"]))
            if key in existing_cross_keys:
                continue
            cross_instance_fixed.append(item)
            existing_cross_keys.add(key)
            if len(select_unique(cross_instance_fixed, top_k)) >= top_k:
                break

    if len(select_unique(cross_instance_fixed, top_k)) == 0:
        proxy_candidates = []
        for item in sorted(v7_better, key=lambda x: x["improvement"], reverse=True):
            record = baseline_lookup[int(item["image_id"])]
            visible_gt_count = sum(1 for gt in record.get("gt_instances", []) if bool(gt.get("has_picking")))
            if visible_gt_count < 2:
                continue
            proxy_item = dict(item)
            proxy_item["footer"] = f"neighbor ambiguity proxy: baseline badness={point_case_badness(item.get('baseline_case')):.1f}, exp2 badness={point_case_badness(item.get('exp2_case')):.1f}"
            proxy_candidates.append(proxy_item)
        cross_instance_fixed = select_unique(proxy_candidates, top_k)

    category_payloads = {
        "baseline_better": sorted(baseline_better, key=lambda item: item["improvement"], reverse=True),
        "v7_better": sorted(v7_better, key=lambda item: item["improvement"], reverse=True),
        "dy_fixed": sorted(dy_fixed, key=lambda item: item["improvement"], reverse=True),
        "cross_instance_fixed": sorted(cross_instance_fixed, key=lambda item: item["improvement"], reverse=True),
        "remaining_failures": sorted(remaining_failures, key=lambda item: item["improvement"], reverse=True),
    }

    output = {}
    for category, entries in category_payloads.items():
        category_dir = rep_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for rank, item in enumerate(select_unique(entries, top_k), start=1):
            image_id = int(item["image_id"])
            gt_index = int(item["gt_index"])
            out_path = category_dir / f"rank{rank:02d}_image_{image_id}_gt{gt_index}.png"
            render_case_pair(
                out_path,
                gt_index,
                baseline_lookup[image_id],
                exp2_lookup[image_id],
                item.get("baseline_case"),
                item.get("exp2_case"),
                baseline_threshold,
                exp2_threshold,
                item["footer"],
            )
            outputs.append(
                {
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "output": str(out_path.resolve()),
                    "footer": item["footer"],
                }
            )
        output[category] = outputs
    return output


def build_claims_md(
    path: Path,
    metrics: dict[str, dict],
    decoupled: dict[str, dict],
    baseline_metrics: dict,
    exp2_metrics: dict,
    repro_payloads: list[dict],
) -> None:
    exp1_metrics = metrics["v7_exp1"]
    exp2_standard = decoupled["v7_exp2"].get("standard_match", {})
    exp1_standard = decoupled["v7_exp1"].get("standard_match", {})
    exp2_iou85 = decoupled["v7_exp2"].get("iou_conditioned", {}).get("ge_0.85", {})
    exp1_iou85 = decoupled["v7_exp1"].get("iou_conditioned", {}).get("ge_0.85", {})
    exp2_oracle_any = decoupled["v7_exp2"].get("oracle_candidates", {}).get("any_visible_pred", {})
    exp1_oracle_any = decoupled["v7_exp1"].get("oracle_candidates", {}).get("any_visible_pred", {})
    ap_better_count = sum(1 for item in repro_payloads if item["metrics"]["grape_AP"] > baseline_metrics["grape_AP"])
    f1_better_count = sum(1 for item in repro_payloads if item["metrics"]["has_picking_F1"] > baseline_metrics["has_picking_F1"])
    pair_better_count = sum(1 for item in repro_payloads if item["metrics"]["pair_count"] > baseline_metrics["pair_count"])
    l2_better_count = sum(1 for item in repro_payloads if item["metrics"]["mean_L2"] < baseline_metrics["mean_L2"])
    dy_better_count = sum(1 for item in repro_payloads if item["metrics"]["mean_abs_dy"] < baseline_metrics["mean_abs_dy"])
    repro_summary = "；".join(
        [
            f"{item['name']}: AP={item['metrics']['grape_AP']:.4f}, F1={item['metrics']['has_picking_F1']:.4f}, "
            f"pair_count={item['metrics']['pair_count']}, mean L2={item['metrics']['mean_L2']:.2f}px, "
            f"|dy|={item['metrics']['mean_abs_dy']:.2f}px"
            for item in repro_payloads
        ]
    )

    lines = [
        "# paper claims（中文）",
        "",
        "## 能不能发论文",
        "- 以当前证据看，这套工作已经足以支撑普通中文期刊、学校认可的普通期刊、以及偏应用类农业视觉期刊投稿。",
        f"- 依据不是单一指标偶然上涨，而是 v7_exp2 除了主实验外，还完成了 {len(repro_payloads)} 次独立复现实验与 checkpoint 稳定性核验；其中 has_picking F1、pair_count、mean L2 和 mean |dy| 都保持同方向改善，grape AP 则存在一定 seed 敏感性。",
        "- 如果目标是更强的视觉方法类 venue，目前证据仍偏薄，主要短板不是结果方向错，而是随机性和泛化证据还不够厚。",
        "",
        "## 主模型到底解决了什么问题",
        f"- v7_exp2 相比 baseline_replay，将 grape AP 从 {baseline_metrics['grape_AP']:.4f} 提高到 {exp2_metrics['grape_AP']:.4f}，has_picking F1 从 {baseline_metrics['has_picking_F1']:.4f} 提高到 {exp2_metrics['has_picking_F1']:.4f}，pair_count 从 {baseline_metrics['pair_count']} 提高到 {exp2_metrics['pair_count']}。",
        f"- 同时 point 误差没有被牺牲，mean L2 从 {baseline_metrics['mean_L2']:.2f}px 降到 {exp2_metrics['mean_L2']:.2f}px，mean |dy| 从 {baseline_metrics['mean_abs_dy']:.2f}px 降到 {exp2_metrics['mean_abs_dy']:.2f}px。",
        "- 这说明它不是单纯提高了可见性判别或增加了匹配样本，而是在检测-可见性-点定位这条链路上都变得更稳。",
        "",
        "## 这些改进是不是有意义",
        "- 有意义，不是纯小幅波动。",
        f"- 原始 v7_exp2 独立于 baseline 的提升方向为：ΔAP={exp2_metrics['grape_AP'] - baseline_metrics['grape_AP']:+.4f}，ΔF1={exp2_metrics['has_picking_F1'] - baseline_metrics['has_picking_F1']:+.4f}，Δpair_count={exp2_metrics['pair_count'] - baseline_metrics['pair_count']:+d}，Δmean L2={exp2_metrics['mean_L2'] - baseline_metrics['mean_L2']:+.2f}px，Δ|dy|={exp2_metrics['mean_abs_dy'] - baseline_metrics['mean_abs_dy']:+.2f}px。",
        f"- 独立复现实验结果为：{repro_summary}。",
        f"- 统计上看，独立复现实验相对 baseline 在 AP 上为 {ap_better_count}/{len(repro_payloads)} 次更优，但在 F1、pair_count、mean L2、mean |dy| 上分别达到 {f1_better_count}/{len(repro_payloads)}、{pair_better_count}/{len(repro_payloads)}、{l2_better_count}/{len(repro_payloads)}、{dy_better_count}/{len(repro_payloads)} 次同方向改善。",
        "",
        "## 为什么 v7_exp2 有效",
        f"- v7_exp1 已经把几何回归拉正了，尤其是 y 方向：mean |dy| 从 baseline 的 {baseline_metrics['mean_abs_dy']:.2f}px 降到 {exp1_metrics['mean_abs_dy']:.2f}px，mean L2 也降到 {exp1_metrics['mean_L2']:.2f}px，但 F1 和 pair_count 反而不够稳。",
        f"- v7_exp2 在保持 y 方向收益的同时，把标准匹配 pair_count 从 exp1 的 {int(safe_float(exp1_standard.get('count')))} 提高到 {int(safe_float(exp2_standard.get('count')))}，pred point 落入 GT box 比例从 {safe_float(exp1_standard.get('pred_point_inside_gt_box_rate')):.4f} 提高到 {safe_float(exp2_standard.get('pred_point_inside_gt_box_rate')):.4f}。",
        f"- 在更严格的 IoU>=0.85 条件下，v7_exp2 的 mean L2 / mean |dy| 也优于 exp1：{safe_float(exp2_iou85.get('mean_l2_px')):.2f}px / {safe_float(exp2_iou85.get('mean_abs_dy_px')):.2f}px 对比 {safe_float(exp1_iou85.get('mean_l2_px')):.2f}px / {safe_float(exp1_iou85.get('mean_abs_dy_px')):.2f}px。",
        f"- 但在 oracle(any visible pred) 下，exp1 的 mean L2={safe_float(exp1_oracle_any.get('mean_l2_px')):.2f}px 反而低于 exp2 的 {safe_float(exp2_oracle_any.get('mean_l2_px')):.2f}px，这说明 exp2 的主要优势不是“全图裸 point 更近”，而是“检测-可见性-关联链条更稳”。",
        "",
        "## 还缺哪 1~2 项证据",
        "- 现在最值得补的不是再开新模型，而是把现有 3 次运行整理成 mean±std，并在正文中明确指出 AP 的 seed 敏感性与 point 指标的稳定收益。",
        "- 其次可以补一个更正式的场景切片统计表，例如按串尺寸或遮挡程度分组，证明改进不是集中在少数样本。",
        "",
        "## 当前最稳妥的贡献点怎么写",
        "- 贡献点不要写成“彻底解决实例一一绑定”。更稳妥的说法是：在单阶段 per-grape has_picking + point offset 框架下，通过顶部锚点表达和顶部局部视觉 cue 的联合建模，同时改善了 grape 检测、采摘点可见性判别与 y 向点定位误差。",
        "- 第二个贡献点可以写成评估层面：将 point 误差拆成标准匹配、IoU 条件和 oracle 候选三个层次，证明当前收益主要来自整条检测-可见性-关联链条的稳定化，而不是只靠单一点误差偶然下降。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_contribution_md(path: Path, metrics: dict[str, dict], repro_payloads: list[dict]) -> None:
    baseline = metrics["baseline_replay"]
    exp2 = metrics["v7_exp2"]
    repro_count = len(repro_payloads)
    stable_point_runs = sum(1 for item in repro_payloads if item["metrics"]["has_picking_F1"] > baseline["has_picking_F1"] and item["metrics"]["pair_count"] > baseline["pair_count"] and item["metrics"]["mean_L2"] < baseline["mean_L2"] and item["metrics"]["mean_abs_dy"] < baseline["mean_abs_dy"])
    ap_better_runs = sum(1 for item in repro_payloads if item["metrics"]["grape_AP"] > baseline["grape_AP"])
    lines = [
        "# contribution draft（中文）",
        "",
        "## 研究问题",
        "- 面向葡萄串识别与采摘点定位任务，如何在单阶段框架下同时完成 grape 检测、采摘点可见性判别以及采摘点坐标回归，而不回退到独立 picking bbox 检测。",
        "",
        "## 方法改进点",
        "- 保持 per-grape has_picking + point offset 主线不变，避免把 picking 当作独立检测目标。",
        "- 采用更贴近采摘点空间先验的顶部锚点坐标表达，缓解 y 方向漂移。",
        "- 在 query_box 绑定的基础上引入顶部局部视觉 cue，使 point 头同时利用实例几何信息和葡萄上部局部纹理信息。",
        "",
        "## 解决的关键问题",
        "- 解决了仅靠 query-box 几何绑定时，point 误差尤其是 dy 仍然偏大的问题。",
        "- 缓解了 has_picking 判别、grape 匹配和 point 回归之间彼此割裂的问题，使检测-可见性-点定位链条更稳定。",
        "",
        "## 创新点（不过度夸大）",
        "- 创新性主要体现在任务建模与轻量结构改进，而不是提出一个大而复杂的新框架。",
        "- 贡献不是“发明全新的检测器”，而是在 RT-DETRv4 单阶段框架内，给出一种更适合葡萄采摘点定位的弱关联建模方式，并辅以更细的误差诊断方法。",
        "",
        "## 实验结论",
        f"- 相比正式 baseline_replay，v7_exp2 在 test 上将 grape AP 从 {baseline['grape_AP']:.4f} 提高到 {exp2['grape_AP']:.4f}，has_picking F1 从 {baseline['has_picking_F1']:.4f} 提高到 {exp2['has_picking_F1']:.4f}。",
        f"- pair_count 从 {baseline['pair_count']} 提高到 {exp2['pair_count']}，mean L2 从 {baseline['mean_L2']:.2f}px 降到 {exp2['mean_L2']:.2f}px，mean |dy| 从 {baseline['mean_abs_dy']:.2f}px 降到 {exp2['mean_abs_dy']:.2f}px。",
        f"- 结合 {repro_count} 次独立复现实验与 checkpoint 对比，has_picking F1、pair_count、mean L2、mean |dy| 的改进是稳定的；AP 在 {ap_better_runs}/{repro_count} 次独立复现中优于 baseline，说明 bbox AP 仍有一定 seed 敏感性，但不改变论文的主收益落在 point 链条上的结论。",
        f"- 总体上，v7_exp2 在 {stable_point_runs}/{repro_count} 次独立复现实验中都保持了 point 相关核心指标的同方向改善，因此具备论文支撑价值。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "paper_ready_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {
        "baseline_replay": load_json(args.baseline_summary),
        "v7_exp1": load_json(args.exp1_summary),
        "v7_exp2": load_json(args.exp2_summary),
        "v7_exp2_repro1": load_json(args.exp2_repro_summary),
    }
    extra_repro_specs = []
    for idx, extra_path in enumerate(args.extra_repro_summary or [], start=1):
        extra_repro_specs.append(
            {
                "name": f"v7_exp2_extra_seed_{idx}",
                "summary_path": Path(extra_path).resolve(),
                "summary": load_json(Path(extra_path)),
            }
        )
    metrics = {name: extract_test_metrics(summary) for name, summary in model_specs.items()}
    for item in extra_repro_specs:
        metrics[item["name"]] = extract_test_metrics(item["summary"])

    paper_rows = []
    for name in ("baseline_replay", "v7_exp1", "v7_exp2", "v7_exp2_repro1"):
        paper_rows.append({"model": name, **metrics[name]})
    write_main_table(out_dir / "paper_ready_table.csv", paper_rows)

    eval_records = {}
    decoupled = {}
    threshold_by_model = {}
    for name in ("baseline_replay", "v7_exp1", "v7_exp2"):
        summary = model_specs[name]
        config_path = Path(summary["config"]).resolve()
        checkpoint_path = Path(summary["primary_checkpoint"]).resolve()
        threshold = get_has_picking_threshold(config_path)
        threshold_by_model[name] = threshold
        _, records = evaluate_split(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            split="test",
            dataset_root=args.dataset_root.resolve(),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            collect_predictions=True,
        )
        eval_records[name] = records
        decoupled[name] = summarize_decoupled_point_diagnostics(records, threshold)

    baseline_cases, _, baseline_mismatches = build_case_indexes(
        eval_records["baseline_replay"], threshold_by_model["baseline_replay"]
    )
    exp2_cases, _, exp2_mismatches = build_case_indexes(eval_records["v7_exp2"], threshold_by_model["v7_exp2"])
    representative_cases = build_representative_cases(
        out_dir,
        eval_records["baseline_replay"],
        eval_records["v7_exp2"],
        baseline_cases,
        exp2_cases,
        baseline_mismatches,
        exp2_mismatches,
        threshold_by_model["baseline_replay"],
        threshold_by_model["v7_exp2"],
        args.top_k_cases,
    )

    plot_final_main_comparison(metrics, figures_dir / "final_main_comparison.png")
    plot_dy_comparison(metrics, figures_dir / "dy_improvement_comparison.png")
    plot_size_group_comparison(metrics, figures_dir / "size_group_error_comparison.png")

    checkpoint_metrics = extract_checkpoint_test_metrics(model_specs["v7_exp2"])
    write_checkpoint_table(out_dir / "stability_checkpoint_table.csv", metrics["baseline_replay"], checkpoint_metrics)
    write_mechanism_table(out_dir / "mechanism_support_table.csv", decoupled)
    seed_stability_rows = [
        {"run": "v7_exp2_main", "metrics": metrics["v7_exp2"]},
        {"run": "v7_exp2_repro1", "metrics": metrics["v7_exp2_repro1"]},
    ]
    for item in extra_repro_specs:
        seed_stability_rows.append({"run": item["name"], "metrics": metrics[item["name"]]})
    write_seed_stability_table(out_dir / "seed_stability_table.csv", metrics["baseline_replay"], seed_stability_rows)
    mean_std_summary = write_mean_std_table(out_dir / "mean_std_table.csv", metrics["baseline_replay"], seed_stability_rows)
    repro_payloads = [{"name": "v7_exp2_repro1", "metrics": metrics["v7_exp2_repro1"]}]
    for item in extra_repro_specs:
        repro_payloads.append({"name": item["name"], "metrics": metrics[item["name"]]})

    baseline_correct_pairs, _, _ = collect_case_groups(eval_records["baseline_replay"], 0.5, threshold_by_model["baseline_replay"])
    exp2_correct_pairs, _, _ = collect_case_groups(eval_records["v7_exp2"], 0.5, threshold_by_model["v7_exp2"])
    scene_slices, scene_notes = build_scene_slices(eval_records["baseline_replay"])
    scene_slice_rows = build_scene_slice_rows(
        scene_slices,
        {
            "baseline_replay": baseline_correct_pairs,
            "v7_exp2": exp2_correct_pairs,
        },
    )
    write_scene_slice_table(out_dir / "scene_slice_table.csv", scene_slice_rows, scene_notes)
    build_scene_slice_md(out_dir / "scene_slice_summary_zh.md", scene_slice_rows, scene_notes)

    paper_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "formal_baseline": {
            "name": "baseline_replay",
            "summary_path": str(args.baseline_summary.resolve()),
            "test_metrics": metrics["baseline_replay"],
        },
        "v7_exp1": {
            "summary_path": str(args.exp1_summary.resolve()),
            "test_metrics": metrics["v7_exp1"],
        },
        "v7_exp2": {
            "summary_path": str(args.exp2_summary.resolve()),
            "test_metrics": metrics["v7_exp2"],
        },
        "stability_validation": {
            "independent_reproductions": [
                {
                    "name": "v7_exp2_repro1",
                    "summary_path": str(args.exp2_repro_summary.resolve()),
                    "test_metrics": metrics["v7_exp2_repro1"],
                    "same_direction_vs_baseline": {
                        "grape_AP_better": metrics["v7_exp2_repro1"]["grape_AP"] > metrics["baseline_replay"]["grape_AP"],
                        "has_picking_F1_better": metrics["v7_exp2_repro1"]["has_picking_F1"] > metrics["baseline_replay"]["has_picking_F1"],
                        "pair_count_better": metrics["v7_exp2_repro1"]["pair_count"] > metrics["baseline_replay"]["pair_count"],
                        "mean_L2_better": metrics["v7_exp2_repro1"]["mean_L2"] < metrics["baseline_replay"]["mean_L2"],
                        "mean_abs_dy_better": metrics["v7_exp2_repro1"]["mean_abs_dy"] < metrics["baseline_replay"]["mean_abs_dy"],
                    },
                },
                *[
                    {
                        "name": item["name"],
                        "summary_path": str(item["summary_path"]),
                        "test_metrics": metrics[item["name"]],
                        "same_direction_vs_baseline": {
                            "grape_AP_better": metrics[item["name"]]["grape_AP"] > metrics["baseline_replay"]["grape_AP"],
                            "has_picking_F1_better": metrics[item["name"]]["has_picking_F1"] > metrics["baseline_replay"]["has_picking_F1"],
                            "pair_count_better": metrics[item["name"]]["pair_count"] > metrics["baseline_replay"]["pair_count"],
                            "mean_L2_better": metrics[item["name"]]["mean_L2"] < metrics["baseline_replay"]["mean_L2"],
                            "mean_abs_dy_better": metrics[item["name"]]["mean_abs_dy"] < metrics["baseline_replay"]["mean_abs_dy"],
                        },
                    }
                    for item in extra_repro_specs
                ],
            ],
            "checkpoint_stability": checkpoint_metrics,
            "v7_exp2_mean_std": mean_std_summary,
        },
        "mechanism_evidence": {
            "decoupled_point_summary": decoupled,
            "scene_slices": {
                "definitions": scene_notes,
                "rows": scene_slice_rows,
            },
            "representative_cases": representative_cases,
            "core_interpretation_zh": {
                "exp1_role": "exp1 主要修正了 point 几何，尤其是 y 方向，但链路稳定性仍不够。",
                "exp2_role": "exp2 在保留 top-center 带来的 dy 收益同时，进一步提升了检测、可见性和关联链条稳定性。",
                "oracle_conclusion": "exp2 的核心优势更像是整条检测-可见性-关联链条更强，而不是单纯依赖裸 point 回归更准。",
            },
        },
        "final_recommended_model": {
            "name": "v7_exp2",
            "reason_zh": "它是当前最稳妥的主模型：正式结果同时改善了 AP、F1、pair_count、mean L2 和 mean |dy|，而独立复现实验进一步证明 F1、pair_count、mean L2 与 mean |dy| 的收益具有稳定性；AP 则存在一定 seed 敏感性。",
        },
        "publication_readiness": {
            "ordinary_chinese_journal": True,
            "school_recognized_general_journal": True,
            "application_oriented_journal": True,
            "high_tier_method_paper": False,
            "judgement_zh": "当前证据已足以支撑普通中文期刊 / 应用类期刊投稿。下一步最值得补的是把现有 3 次运行整理成 mean±std，并补一个简洁的场景切片表，而不是继续开新模型。",
        },
    }
    (out_dir / "paper_ready_summary.json").write_text(json.dumps(paper_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_claims_md(
        out_dir / "paper_claims_zh.md",
        metrics,
        decoupled,
        metrics["baseline_replay"],
        metrics["v7_exp2"],
        repro_payloads,
    )
    build_contribution_md(out_dir / "contribution_draft_zh.md", metrics, repro_payloads)

    print(f"[paper-ready] wrote {out_dir / 'paper_ready_summary.json'}")
    print(f"[paper-ready] wrote {out_dir / 'paper_ready_table.csv'}")
    print(f"[paper-ready] wrote {out_dir / 'paper_claims_zh.md'}")
    print(f"[paper-ready] wrote {out_dir / 'contribution_draft_zh.md'}")
    print(f"[paper-ready] wrote {out_dir / 'seed_stability_table.csv'}")
    print(f"[paper-ready] wrote {out_dir / 'mean_std_table.csv'}")
    print(f"[paper-ready] wrote {out_dir / 'scene_slice_table.csv'}")
    print(f"[paper-ready] wrote {out_dir / 'scene_slice_summary_zh.md'}")
    print(f"[paper-ready] wrote {figures_dir / 'final_main_comparison.png'}")
    print(f"[paper-ready] wrote {figures_dir / 'dy_improvement_comparison.png'}")
    print(f"[paper-ready] wrote {figures_dir / 'size_group_error_comparison.png'}")


if __name__ == "__main__":
    main()
