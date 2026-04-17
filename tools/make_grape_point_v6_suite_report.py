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
DEFAULT_BASELINE_REPLAY_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v6_baseline_replay" / "report" / "summary.json"
DEFAULT_POINT_V2_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v2_main" / "report" / "summary.json"
DEFAULT_EXP_SUMMARIES = {
    "exp1": REPO_ROOT / "outputs" / "grape_point_v6_exp1_instance_binding" / "report" / "summary.json",
    "exp2": REPO_ROOT / "outputs" / "grape_point_v6_exp2_instance_binding_relative" / "report" / "summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate point_v6 experiment summaries.")
    parser.add_argument("--suite-dir", type=Path, default=REPO_ROOT / "outputs" / "grape_point_v6_suite" / "report")
    parser.add_argument("--baseline-replay-summary", type=Path, default=DEFAULT_BASELINE_REPLAY_SUMMARY)
    parser.add_argument("--point-v2-summary", type=Path, default=DEFAULT_POINT_V2_SUMMARY)
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
        "grape_AP50": safe_float(grape.get("AP50")),
        "grape_AR100": safe_float(grape.get("AR100")),
        "has_picking_precision": safe_float(has_pick.get("precision")),
        "has_picking_recall": safe_float(has_pick.get("recall")),
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
        "cross_instance_mismatch_count": int(
            safe_float(
                summary.get("qualitative_cases", {}).get("cross_instance_mismatch_count"),
                summary.get("error_analysis", {}).get("cross_instance_mismatch_count", 0),
            )
        ),
    }


def choose_best_experiment(results: dict[str, dict], baseline_metrics: dict | None) -> str | None:
    valid = [(name, metrics) for name, metrics in results.items() if metrics is not None and np.isfinite(metrics["mean_L2"])]
    if not valid:
        return None
    if baseline_metrics is None:
        return min(valid, key=lambda item: item[1]["mean_L2"])[0]

    ranked: list[tuple[float, str]] = []
    for name, metrics in valid:
        score = metrics["mean_L2"]
        if metrics["grape_AP"] < baseline_metrics["grape_AP"] - 0.005:
            score += (baseline_metrics["grape_AP"] - 0.005 - metrics["grape_AP"]) * 100.0
        if metrics["has_picking_F1"] < baseline_metrics["has_picking_F1"] - 0.010:
            score += (baseline_metrics["has_picking_F1"] - 0.010 - metrics["has_picking_F1"]) * 120.0
        if metrics["pair_count"] < baseline_metrics["pair_count"]:
            score += float(baseline_metrics["pair_count"] - metrics["pair_count"]) * 0.10
        ranked.append((score, name))
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def plot_overview(baseline_metrics: dict | None, exp_results: dict[str, dict], out_path: Path) -> None:
    labels = []
    grape_values = []
    f1_values = []
    l2_values = []
    median_values = []
    p90_values = []
    pair_values = []
    dx_values = []
    dy_values = []

    if baseline_metrics is not None:
        labels.append("baseline_replay")
        grape_values.append(baseline_metrics["grape_AP"])
        f1_values.append(baseline_metrics["has_picking_F1"])
        l2_values.append(baseline_metrics["mean_L2"])
        median_values.append(baseline_metrics["median_L2"])
        p90_values.append(baseline_metrics["p90_L2"])
        pair_values.append(baseline_metrics["pair_count"])
        dx_values.append(baseline_metrics["mean_abs_dx"])
        dy_values.append(baseline_metrics["mean_abs_dy"])

    for name in ("exp1", "exp2"):
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
        dx_values.append(metrics["mean_abs_dx"])
        dy_values.append(metrics["mean_abs_dy"])

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

    axes[1, 0].bar(x - 0.18, dx_values, width=0.36, label="mean |dx|", color="#ff7f0e")
    axes[1, 0].bar(x + 0.18, dy_values, width=0.36, label="mean |dy|", color="#d62728")
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


def build_report_md(report_path: Path, baseline_metrics: dict | None, exp_results: dict[str, dict], best_exp: str | None) -> None:
    lines = [
        "# point_v6 中文结论",
        "",
        "## 总览",
        "- point_v6 固定从 point_v2 重新起线，不继承 point_v3 / point_v4 / point_v5 的失败主改动。",
        "- 本轮只做两项核心变化：更强的 per-grape instance binding，以及更稳定的 bbox-relative 坐标回归表达。",
        "- 不回退到 picking bbox，也不引入 two-stage ROI detector。",
        "",
    ]
    if baseline_metrics is not None:
        lines.append(
            f"- baseline_replay：grape AP={baseline_metrics['grape_AP']:.4f}, has_picking F1={baseline_metrics['has_picking_F1']:.4f}, pair_count={baseline_metrics['pair_count']}, mean L2={baseline_metrics['mean_L2']:.2f}px, median L2={baseline_metrics['median_L2']:.2f}px, p90 L2={baseline_metrics['p90_L2']:.2f}px, |dx|={baseline_metrics['mean_abs_dx']:.2f}px, |dy|={baseline_metrics['mean_abs_dy']:.2f}px。"
        )
    if best_exp and exp_results.get(best_exp) is not None:
        best = exp_results[best_exp]
        lines.extend(
            [
                "",
                "## 当前自动推荐实验",
                f"- 推荐 `{best_exp}` 作为 point_v6 当前最值得继续的实验。",
                f"- test 指标：grape AP={best['grape_AP']:.4f}, has_picking F1={best['has_picking_F1']:.4f}, pair_count={best['pair_count']}, mean L2={best['mean_L2']:.2f}px, median L2={best['median_L2']:.2f}px, p90 L2={best['p90_L2']:.2f}px, |dx|={best['mean_abs_dx']:.2f}px, |dy|={best['mean_abs_dy']:.2f}px。",
            ]
        )
        if baseline_metrics is not None:
            lines.append(
                f"- 相比 baseline_replay：Δgrape AP={best['grape_AP'] - baseline_metrics['grape_AP']:+.4f}, Δhas_picking F1={best['has_picking_F1'] - baseline_metrics['has_picking_F1']:+.4f}, Δpair_count={best['pair_count'] - baseline_metrics['pair_count']:+d}, Δmean L2={best['mean_L2'] - baseline_metrics['mean_L2']:+.2f}px, Δ|dy|={best['mean_abs_dy'] - baseline_metrics['mean_abs_dy']:+.2f}px。"
            )
    lines.extend(["", "## 各实验摘要"])
    for name in ("exp1", "exp2"):
        metrics = exp_results.get(name)
        if metrics is None:
            lines.append(f"- `{name}`: 未找到 summary.json，暂未纳入汇总。")
            continue
        lines.append(
            f"- `{name}`: grape AP={metrics['grape_AP']:.4f}, has_picking F1={metrics['has_picking_F1']:.4f}, pair_count={metrics['pair_count']}, mean L2={metrics['mean_L2']:.2f}px, median L2={metrics['median_L2']:.2f}px, p90 L2={metrics['p90_L2']:.2f}px, |dx|={metrics['mean_abs_dx']:.2f}px, |dy|={metrics['mean_abs_dy']:.2f}px, small/medium/large L2={metrics['small_L2']:.2f}/{metrics['medium_L2']:.2f}/{metrics['large_L2']:.2f}px, cross-instance mismatch={metrics['cross_instance_mismatch_count']}。"
        )
    lines.extend(
        [
            "",
            "## 关注点",
            "- 重点看 mean L2 和 |dy| 是否一起下降，而不是只看一个点误差均值。",
            "- 如果 pair_count 上升但 mean L2 没降，说明 has_picking 召回改善了，但点位仍不稳定。",
            "- 如果 cross-instance mismatch 仍高，主矛盾就还在邻串误关联，而不是单纯的局部坐标回归精度。",
        ]
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

    baseline_summary = load_json(args.baseline_replay_summary)
    point_v2_summary = load_json(args.point_v2_summary)
    baseline_metrics = extract_test_metrics(baseline_summary)
    point_v2_metrics = extract_test_metrics(point_v2_summary)

    exp_results = {}
    for name, path in exp_paths.items():
        exp_results[name] = extract_test_metrics(load_json(path))

    best_exp = choose_best_experiment(exp_results, baseline_metrics)
    plot_overview(baseline_metrics, exp_results, args.suite_dir / "results_overview.png")
    build_report_md(args.suite_dir / "comparison_report_zh.md", baseline_metrics, exp_results, best_exp)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "suite_dir": str(args.suite_dir),
        "baseline_replay_summary": str(args.baseline_replay_summary.resolve()),
        "point_v2_summary": str(args.point_v2_summary.resolve()),
        "baseline_replay_metrics": baseline_metrics,
        "point_v2_metrics": point_v2_metrics,
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
