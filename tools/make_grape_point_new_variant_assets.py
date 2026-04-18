from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_grape_point_v2_report import collect_case_groups, evaluate_split, safe_float
from make_grape_point_v7_paper_assets import (
    build_scene_slices,
    extract_test_metrics,
    get_has_picking_threshold,
    load_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v6_baseline_replay" / "report" / "summary.json"
DEFAULT_EXP1_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v7_exp1_query_box_top_center" / "report" / "summary.json"
DEFAULT_EXP2_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v7_exp2_query_box_top_center_toproi" / "report" / "summary.json"
DEFAULT_NEW_SUMMARY = REPO_ROOT / "outputs" / "grape_point_v7_exp2_tight_toproi" / "report" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison assets for the tight top-local ROI variant.")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--exp1-summary", type=Path, default=DEFAULT_EXP1_SUMMARY)
    parser.add_argument("--exp2-summary", type=Path, default=DEFAULT_EXP2_SUMMARY)
    parser.add_argument("--new-summary", type=Path, default=DEFAULT_NEW_SUMMARY)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--new-label", default="new_variant")
    parser.add_argument("--variant-title", default="new variant 中文结论")
    return parser.parse_args()


def plot_overview(metrics_by_name: dict[str, dict], out_path: Path, new_label: str) -> None:
    labels = ["baseline_replay", "v7_exp1", "v7_exp2", new_label]
    x = np.arange(len(labels), dtype=np.float64)
    grape_ap = [metrics_by_name[name]["grape_AP"] for name in labels]
    f1 = [metrics_by_name[name]["has_picking_F1"] for name in labels]
    l2 = [metrics_by_name[name]["mean_L2"] for name in labels]
    dy = [metrics_by_name[name]["mean_abs_dy"] for name in labels]
    pair_count = [metrics_by_name[name]["pair_count"] for name in labels]
    small = [metrics_by_name[name]["small_L2"] for name in labels]
    medium = [metrics_by_name[name]["medium_L2"] for name in labels]
    large = [metrics_by_name[name]["large_L2"] for name in labels]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(x - 0.18, grape_ap, width=0.36, label="grape AP", color="#1f77b4")
    axes[0, 0].bar(x + 0.18, f1, width=0.36, label="has_picking F1", color="#2ca02c")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Detection / Classification")
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.18, l2, width=0.36, label="mean L2", color="#8c564b")
    axes[0, 1].bar(x + 0.18, dy, width=0.36, label="mean |dy|", color="#d62728")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_title("Point Error")
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].bar(x, pair_count, color="#9467bd")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_title("Pair Count")
    axes[1, 0].grid(axis="y", alpha=0.3)

    width = 0.24
    axes[1, 1].bar(x - width, small, width=width, label="small L2", color="#ff7f0e")
    axes[1, 1].bar(x, medium, width=width, label="medium L2", color="#e377c2")
    axes[1, 1].bar(x + width, large, width=width, label="large L2", color="#7f7f7f")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_title("Size-group Point Error")
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_scene_table_rows(
    scene_slices: dict[str, dict[str, set[tuple[int, int]]]],
    model_cases: dict[str, list[dict]],
    new_label: str,
) -> list[dict]:
    rows = []
    for family, groups in scene_slices.items():
        for label, keys in groups.items():
            baseline_summary = summarize_scene_slice(model_cases["baseline_replay"], keys)
            exp2_summary = summarize_scene_slice(model_cases["v7_exp2"], keys)
            new_summary = summarize_scene_slice(model_cases[new_label], keys)
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
                    "new_pair_count": new_summary["pair_count"],
                    "new_pair_recall": new_summary["pair_recall"],
                    "new_mean_L2": new_summary["mean_L2"],
                    "new_mean_abs_dy": new_summary["mean_abs_dy"],
                    "delta_vs_exp2_pair_count": new_summary["pair_count"] - exp2_summary["pair_count"],
                    "delta_vs_exp2_pair_recall": new_summary["pair_recall"] - exp2_summary["pair_recall"],
                    "delta_vs_exp2_mean_L2": new_summary["mean_L2"] - exp2_summary["mean_L2"],
                    "delta_vs_exp2_mean_abs_dy": new_summary["mean_abs_dy"] - exp2_summary["mean_abs_dy"],
                }
            )
    return rows


def summarize_scene_slice(correct_pairs: list[dict], slice_keys: set[tuple[int, int]]) -> dict:
    cases = [item for item in correct_pairs if (int(item["image_id"]), int(item["gt_index"])) in slice_keys]
    visible_gt_count = len(slice_keys)
    l2_values = [float(item.get("l2_px", 0.0)) for item in cases]
    dy_values = [float(item.get("dy_px", 0.0)) for item in cases]
    return {
        "visible_gt_count": visible_gt_count,
        "pair_count": len(cases),
        "pair_recall": float(len(cases) / visible_gt_count) if visible_gt_count > 0 else 0.0,
        "mean_L2": float(np.mean(l2_values)) if l2_values else 0.0,
        "mean_abs_dy": float(np.mean(np.abs(dy_values))) if dy_values else 0.0,
    }


def write_new_variant_scene_csv(path: Path, rows: list[dict], notes: dict) -> None:
    import csv

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
        "new_pair_count",
        "new_pair_recall",
        "new_mean_L2",
        "new_mean_abs_dy",
        "delta_vs_exp2_pair_count",
        "delta_vs_exp2_pair_recall",
        "delta_vs_exp2_mean_L2",
        "delta_vs_exp2_mean_abs_dy",
        "definition_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["definition_note"] = notes.get(payload["slice_family"], "")
            writer.writerow(payload)


def write_report(
    path: Path,
    metrics_by_name: dict[str, dict],
    scene_rows: list[dict],
    notes: dict,
    new_label: str,
    variant_title: str,
) -> None:
    baseline = metrics_by_name["baseline_replay"]
    exp2 = metrics_by_name["v7_exp2"]
    new = metrics_by_name[new_label]
    exp1 = metrics_by_name["v7_exp1"]
    family_titles = {
        "single_vs_multi": "单串 / 多串相邻",
        "occlusion_proxy": "遮挡轻 / 重（几何代理）",
        "size_group": "小串 / 中大串",
    }
    label_titles = {
        "single": "单串",
        "multi_adjacent": "多串相邻",
        "light": "遮挡轻",
        "heavy": "遮挡重",
        "small": "小串",
        "medium_large": "中大串",
    }

    lines = [
        f"# {variant_title}",
        "",
        "## 核心指标",
        f"- baseline_replay: AP={baseline['grape_AP']:.4f}, F1={baseline['has_picking_F1']:.4f}, pair_count={baseline['pair_count']}, mean L2={baseline['mean_L2']:.2f}px, |dy|={baseline['mean_abs_dy']:.2f}px。",
        f"- v7_exp1: AP={exp1['grape_AP']:.4f}, F1={exp1['has_picking_F1']:.4f}, pair_count={exp1['pair_count']}, mean L2={exp1['mean_L2']:.2f}px, |dy|={exp1['mean_abs_dy']:.2f}px。",
        f"- v7_exp2: AP={exp2['grape_AP']:.4f}, F1={exp2['has_picking_F1']:.4f}, pair_count={exp2['pair_count']}, mean L2={exp2['mean_L2']:.2f}px, |dy|={exp2['mean_abs_dy']:.2f}px。",
        f"- {new_label}: AP={new['grape_AP']:.4f}, F1={new['has_picking_F1']:.4f}, pair_count={new['pair_count']}, mean L2={new['mean_L2']:.2f}px, |dy|={new['mean_abs_dy']:.2f}px。",
        "",
        "## 相对 v7_exp2 变化",
        f"- ΔAP={new['grape_AP'] - exp2['grape_AP']:+.4f}",
        f"- ΔF1={new['has_picking_F1'] - exp2['has_picking_F1']:+.4f}",
        f"- Δpair_count={new['pair_count'] - exp2['pair_count']:+d}",
        f"- Δmean L2={new['mean_L2'] - exp2['mean_L2']:+.2f}px",
        f"- Δ|dy|={new['mean_abs_dy'] - exp2['mean_abs_dy']:+.2f}px",
        f"- size-group L2: small/medium/large = {new['small_L2']:.2f}/{new['medium_L2']:.2f}/{new['large_L2']:.2f}px",
        "",
        "## 场景切片说明",
        f"- {notes['single_vs_multi']}",
        f"- {notes['occlusion_proxy']}",
        f"- {notes['size_group']}",
        "",
        "## 场景切片结果",
    ]
    for family in ("single_vs_multi", "occlusion_proxy", "size_group"):
        lines.append(f"### {family_titles[family]}")
        for row in [item for item in scene_rows if item["slice_family"] == family]:
            lines.append(
                "- "
                f"{label_titles.get(row['slice_label'], row['slice_label'])}: "
                f"v7_exp2 pair_recall {row['exp2_pair_recall']:.3f} -> {row['new_pair_recall']:.3f}, "
                f"mean L2 {row['exp2_mean_L2']:.2f} -> {row['new_mean_L2']:.2f}px, "
                f"|dy| {row['exp2_mean_abs_dy']:.2f} -> {row['new_mean_abs_dy']:.2f}px"
            )
        lines.append("")

    worth_multiseed = (
        new["mean_L2"] < exp2["mean_L2"]
        and new["mean_abs_dy"] < exp2["mean_abs_dy"]
        and new["has_picking_F1"] >= exp2["has_picking_F1"] - 0.01
        and new["pair_count"] >= exp2["pair_count"] - 10
    )
    heavy_rows = [row for row in scene_rows if row["slice_family"] == "occlusion_proxy" and row["slice_label"] == "heavy"]
    multi_rows = [row for row in scene_rows if row["slice_family"] == "single_vs_multi" and row["slice_label"] == "multi_adjacent"]
    heavy_or_multi_improved = any(row["delta_vs_exp2_mean_L2"] < 0.0 or row["delta_vs_exp2_mean_abs_dy"] < 0.0 for row in heavy_rows + multi_rows)
    lines.extend(
        [
            "## 判断",
            "- 当前候选是否值得继续多 seed："
            + ("值得，建议继续做多 seed。" if worth_multiseed and heavy_or_multi_improved else "暂不值得直接做多 seed。"),
            "- 如果这版没有同时压低 mean L2 / |dy|，且复杂场景也没有改善，就不建议继续扩 seed。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_summary = load_json(args.baseline_summary)
    exp1_summary = load_json(args.exp1_summary)
    exp2_summary = load_json(args.exp2_summary)
    new_summary = load_json(args.new_summary)

    metrics_by_name = {
        "baseline_replay": extract_test_metrics(baseline_summary),
        "v7_exp1": extract_test_metrics(exp1_summary),
        "v7_exp2": extract_test_metrics(exp2_summary),
        args.new_label: extract_test_metrics(new_summary),
    }

    run_dir = Path(new_summary["run_dir"]).resolve()
    out_summary = run_dir / "new_variant_summary.json"
    out_overview = run_dir / "new_variant_results_overview.png"
    out_scene = run_dir / "new_variant_scene_slice_table.csv"
    out_report = run_dir / "new_variant_comparison_report_zh.md"

    plot_overview(metrics_by_name, out_overview, args.new_label)

    record_specs = {
        "baseline_replay": baseline_summary,
        "v7_exp2": exp2_summary,
        args.new_label: new_summary,
    }
    eval_records = {}
    thresholds = {}
    for name, summary in record_specs.items():
        config_path = Path(summary["config"]).resolve()
        checkpoint_path = Path(summary["primary_checkpoint"]).resolve()
        thresholds[name] = get_has_picking_threshold(config_path)
        _, records = evaluate_split(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            split="test",
            dataset_root=args.dataset_root.resolve(),
            collect_predictions=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        eval_records[name] = records

    scene_slices, scene_notes = build_scene_slices(eval_records["baseline_replay"])
    model_cases = {}
    for name in ("baseline_replay", "v7_exp2", args.new_label):
        correct_pairs, _, _ = collect_case_groups(eval_records[name], 0.5, thresholds[name])
        model_cases[name] = correct_pairs
    scene_rows = build_scene_table_rows(scene_slices, model_cases, args.new_label)
    write_new_variant_scene_csv(out_scene, scene_rows, scene_notes)
    write_report(out_report, metrics_by_name, scene_rows, scene_notes, args.new_label, args.variant_title)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_summary_path": str(args.baseline_summary.resolve()),
        "exp1_summary_path": str(args.exp1_summary.resolve()),
        "exp2_summary_path": str(args.exp2_summary.resolve()),
        "new_summary_path": str(args.new_summary.resolve()),
        "metrics": metrics_by_name,
        "scene_slice_notes": scene_notes,
        "scene_slice_rows": scene_rows,
        "outputs": {
            "results_overview": str(out_overview),
            "scene_slice_table": str(out_scene),
            "comparison_report": str(out_report),
        },
    }
    out_summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[new-variant] wrote {out_summary}")
    print(f"[new-variant] wrote {out_overview}")
    print(f"[new-variant] wrote {out_scene}")
    print(f"[new-variant] wrote {out_report}")


if __name__ == "__main__":
    main()
