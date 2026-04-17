from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINT_V2_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v2_main" / "report" / "summary.json"
DEFAULT_BBOX_BASELINE_SUMMARY = REPO_ROOT / "outputs" / "baseline_20260407" / "report" / "summary.json"
DEFAULT_EXP_SUMMARIES = {
    "exp0": REPO_ROOT / "outputs" / "grape_point_v4_exp0_query_only" / "report" / "summary.json",
    "exp1": REPO_ROOT / "outputs" / "grape_point_v4_exp1_full_roi" / "report" / "summary.json",
    "exp2": REPO_ROOT / "outputs" / "grape_point_v4_exp2_top_roi" / "report" / "summary.json",
    "exp3": REPO_ROOT / "outputs" / "grape_point_v4_exp3_full_top_roi" / "report" / "summary.json",
    "exp4": REPO_ROOT / "outputs" / "grape_point_v4_exp4_full_top_roi_yw" / "report" / "summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate point_v4 experiment summaries.")
    parser.add_argument("--suite-dir", type=Path, default=REPO_ROOT / "outputs" / "grape_point_v4_suite" / "report")
    parser.add_argument("--point-v2-summary", type=Path, default=DEFAULT_POINT_V2_SUMMARY)
    parser.add_argument("--bbox-baseline-summary", type=Path, default=DEFAULT_BBOX_BASELINE_SUMMARY)
    parser.add_argument("--exp-summary", action="append", default=None, help="Override experiment summary in NAME=PATH format.")
    return parser.parse_args()


def load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def extract_test_metrics(summary: dict | None) -> dict | None:
    if not isinstance(summary, dict):
        return None
    test = summary.get("primary_checkpoint_split_summary", {}).get("test", {})
    grape = test.get("grape_detection", {})
    has_pick = test.get("has_picking", {})
    point = test.get("picking_point", {})
    if not test:
        return None
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


def choose_best_experiment(results: dict[str, dict], point_v2_metrics: dict | None) -> str | None:
    candidates = []
    for name, metrics in results.items():
        if metrics is None:
            continue
        l2 = metrics["mean_L2"]
        grape_ap = metrics["grape_AP"]
        has_f1 = metrics["has_picking_F1"]
        pair_count = metrics["pair_count"]
        if not np.isfinite(l2):
            continue
        penalty = 0.0
        if grape_ap < 0.63:
            penalty += (0.63 - grape_ap) * 100.0
        if has_f1 < 0.78:
            penalty += (0.78 - has_f1) * 120.0
        if point_v2_metrics is not None and pair_count < int(round(0.9 * point_v2_metrics["pair_count"])):
            penalty += float(int(round(0.9 * point_v2_metrics["pair_count"])) - pair_count) * 0.12
        candidates.append((l2 + penalty, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def plot_overview(point_v2_metrics: dict | None, exp_results: dict[str, dict], out_path: Path) -> None:
    labels = []
    grape_values = []
    f1_values = []
    l2_values = []
    median_values = []
    p90_values = []
    pair_values = []

    if point_v2_metrics is not None:
        labels.append("point_v2")
        grape_values.append(point_v2_metrics["grape_AP"])
        f1_values.append(point_v2_metrics["has_picking_F1"])
        l2_values.append(point_v2_metrics["mean_L2"])
        median_values.append(point_v2_metrics["median_L2"])
        p90_values.append(point_v2_metrics["p90_L2"])
        pair_values.append(point_v2_metrics["pair_count"])

    for name in ("exp0", "exp1", "exp2", "exp3", "exp4"):
        metrics = exp_results.get(name)
        if metrics is None:
            continue
        labels.append(name)
        grape_values.append(metrics["grape_AP"])
        f1_values.append(metrics["has_picking_F1"])
        l2_values.append(metrics["mean_L2"])
        median_values.append(metrics["median_L2"])
        p90_values.append(metrics["p90_L2"])
        pair_values.append(metrics["pair_count"])

    x = np.arange(len(labels), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(x - 0.18, grape_values, width=0.36, label="grape AP", color="#1f77b4")
    axes[0, 0].bar(x + 0.18, f1_values, width=0.36, label="has_picking F1", color="#2ca02c")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].set_title("Test Detection / Classification")
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.25, l2_values, width=0.25, label="mean L2", color="#8c564b")
    axes[0, 1].bar(x, median_values, width=0.25, label="median L2", color="#e377c2")
    axes[0, 1].bar(x + 0.25, p90_values, width=0.25, label="p90 L2", color="#7f7f7f")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].set_title("Test Point Error")
    axes[0, 1].set_ylabel("L2 error (px)")
    axes[0, 1].legend()

    axes[1, 0].bar(x - 0.18, [exp_results.get(label, point_v2_metrics)["mean_abs_dx"] if label != "point_v2" else point_v2_metrics["mean_abs_dx"] for label in labels], width=0.36, label="mean |dx|", color="#ff7f0e")
    axes[1, 0].bar(x + 0.18, [exp_results.get(label, point_v2_metrics)["mean_abs_dy"] if label != "point_v2" else point_v2_metrics["mean_abs_dy"] for label in labels], width=0.36, label="mean |dy|", color="#d62728")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].set_title("XY Bias")
    axes[1, 0].set_ylabel("Absolute error (px)")
    axes[1, 0].legend()

    axes[1, 1].bar(x, pair_values, color="#9467bd")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].set_title("Point Pair Count")
    axes[1, 1].set_ylabel("Matched pairs")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report_md(report_path: Path, point_v2_metrics: dict | None, exp_results: dict[str, dict], best_exp: str | None) -> None:
    lines = [
        "# point_v4 实验组中文结论",
        "",
        "## 总览",
        "- point_v4 从 point_v2 重新起线，只围绕实例绑定增强做最小可行修改。",
        "- 本轮对比重点是 query only / full ROI / top ROI / full+top ROI，以及在最完整 ROI 基础上的 mild y-weight + locality penalty。",
        "",
    ]
    if point_v2_metrics is not None:
        lines.append(
            f"- point_v2 基线：grape AP={point_v2_metrics['grape_AP']:.4f}, has_picking F1={point_v2_metrics['has_picking_F1']:.4f}, pair_count={point_v2_metrics['pair_count']}, mean L2={point_v2_metrics['mean_L2']:.2f}px"
        )
    if best_exp and exp_results.get(best_exp) is not None:
        best = exp_results[best_exp]
        lines.extend(
            [
                "",
                "## 当前自动推荐实验",
                f"- 推荐 `{best_exp}` 作为 point_v4 当前最值得细看的实验。",
                f"- 它的 test 指标：grape AP={best['grape_AP']:.4f}, has_picking F1={best['has_picking_F1']:.4f}, pair_count={best['pair_count']}, mean L2={best['mean_L2']:.2f}px, median L2={best['median_L2']:.2f}px, p90 L2={best['p90_L2']:.2f}px。",
            ]
        )
    lines.extend(["", "## 各实验摘要"])
    for name in ("exp0", "exp1", "exp2", "exp3", "exp4"):
        metrics = exp_results.get(name)
        if metrics is None:
            lines.append(f"- `{name}`: 未找到 summary.json，暂未纳入汇总。")
            continue
        lines.append(
            f"- `{name}`: grape AP={metrics['grape_AP']:.4f}, has_picking F1={metrics['has_picking_F1']:.4f}, pair_count={metrics['pair_count']}, mean L2={metrics['mean_L2']:.2f}px, median L2={metrics['median_L2']:.2f}px, p90 L2={metrics['p90_L2']:.2f}px, |dx|={metrics['mean_abs_dx']:.2f}px, |dy|={metrics['mean_abs_dy']:.2f}px, small/medium/large L2={metrics['small_L2']:.2f}/{metrics['medium_L2']:.2f}/{metrics['large_L2']:.2f}px"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.suite_dir = args.suite_dir.resolve()
    args.suite_dir.mkdir(parents=True, exist_ok=True)

    exp_paths = dict(DEFAULT_EXP_SUMMARIES)
    for item in args.exp_summary or []:
        name, raw_path = item.split("=", 1)
        exp_paths[name.strip()] = Path(raw_path.strip()).resolve()

    point_v2_summary = load_json(args.point_v2_summary)
    bbox_summary = load_json(args.bbox_baseline_summary)
    point_v2_metrics = extract_test_metrics(point_v2_summary)

    exp_results = {}
    for name, path in exp_paths.items():
        exp_results[name] = extract_test_metrics(load_json(path))

    best_exp = choose_best_experiment(exp_results, point_v2_metrics)
    plot_overview(point_v2_metrics, exp_results, args.suite_dir / "results_overview.png")
    build_report_md(args.suite_dir / "comparison_report_zh.md", point_v2_metrics, exp_results, best_exp)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "suite_dir": str(args.suite_dir),
        "point_v2_summary": str(args.point_v2_summary.resolve()),
        "bbox_baseline_summary": str(args.bbox_baseline_summary.resolve()),
        "point_v2_metrics": point_v2_metrics,
        "bbox_baseline_summary_payload": bbox_summary,
        "experiments": {
            name: {
                "summary_path": str(exp_paths[name]),
                "test_metrics": metrics,
            }
            for name, metrics in exp_results.items()
        },
        "recommended_best_experiment": best_exp,
    }
    (args.suite_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
