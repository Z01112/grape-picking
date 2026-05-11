from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from make_grape_point_report import evaluate_split
from make_grape_point_paper_assets import (
    build_case_indexes,
    build_scene_slices,
    extract_test_metrics,
    get_has_picking_threshold,
    load_json,
    point_case_badness,
    render_case_pair,
    select_unique,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_READY_DIR = REPO_ROOT / "outputs" / "grape_point_v7_paper_ready"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "grape_point_v7_sci_ready"
DEFAULT_STRENGTHENING_SUMMARY = REPO_ROOT / "outputs" / "grape_point_gppoint_detr_small_weight" / "new_variant_summary.json"
DEFAULT_LOCAL_RTDETR_JSON = Path(
    r"D:\Projects\ultralytics-rtdetr\runs\rtdetr\grape_baseline_b8_e100_20260407_224008\best_per_class_metrics.json"
)
DEFAULT_POSE_REFERENCE_URL = "https://docs.ultralytics.com/tasks/pose/"
DEFAULT_RTDETR_REFERENCE_URL = "https://docs.ultralytics.com/models/rtdetr/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCI-ready assets for the v7 grape-point project.")
    parser.add_argument("--paper-ready-dir", type=Path, default=DEFAULT_PAPER_READY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strengthening-summary", type=Path, default=DEFAULT_STRENGTHENING_SUMMARY)
    parser.add_argument("--local-rtdetr-json", type=Path, default=DEFAULT_LOCAL_RTDETR_JSON)
    parser.add_argument("--pose-reference-url", default=DEFAULT_POSE_REFERENCE_URL)
    parser.add_argument("--rtdetr-reference-url", default=DEFAULT_RTDETR_REFERENCE_URL)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-cases", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_final_mean_std_rows(paper_summary: dict) -> tuple[list[dict], dict[str, str]]:
    baseline = paper_summary["formal_baseline"]["test_metrics"]
    mean_std = paper_summary["stability_validation"]["v7_exp2_mean_std"]
    repros = paper_summary["stability_validation"]["independent_reproductions"]
    total_runs = 1 + len(repros)

    same_direction = {
        "AP": 1 + sum(bool(item["same_direction_vs_baseline"]["grape_AP_better"]) for item in repros),
        "F1": 1 + sum(bool(item["same_direction_vs_baseline"]["has_picking_F1_better"]) for item in repros),
        "pair_count": 1 + sum(bool(item["same_direction_vs_baseline"]["pair_count_better"]) for item in repros),
        "mean_L2": 1 + sum(bool(item["same_direction_vs_baseline"]["mean_L2_better"]) for item in repros),
        "mean_abs_dy": 1 + sum(bool(item["same_direction_vs_baseline"]["mean_abs_dy_better"]) for item in repros),
    }
    stability_tag = {
        "AP": "seed_sensitive" if same_direction["AP"] < total_runs else "stable_gain",
        "F1": "stable_gain" if same_direction["F1"] == total_runs else "partly_stable",
        "pair_count": "stable_gain" if same_direction["pair_count"] == total_runs else "partly_stable",
        "mean_L2": "stable_gain" if same_direction["mean_L2"] == total_runs else "partly_stable",
        "mean_abs_dy": "stable_gain" if same_direction["mean_abs_dy"] == total_runs else "partly_stable",
    }

    rows = [
        {
            "metric": "AP",
            "baseline": f"{baseline['grape_AP']:.4f}",
            "v7_exp2_mean": f"{mean_std['AP']['mean']:.4f}",
            "v7_exp2_std": f"{mean_std['AP']['std']:.4f}",
            "direction_vs_baseline": f"{same_direction['AP']}/{total_runs} runs better",
            "stability_tag": stability_tag["AP"],
            "comment": "bbox AP improves in most runs, but still shows seed sensitivity.",
        },
        {
            "metric": "F1",
            "baseline": f"{baseline['has_picking_F1']:.4f}",
            "v7_exp2_mean": f"{mean_std['F1']['mean']:.4f}",
            "v7_exp2_std": f"{mean_std['F1']['std']:.4f}",
            "direction_vs_baseline": f"{same_direction['F1']}/{total_runs} runs better",
            "stability_tag": stability_tag["F1"],
            "comment": "has_picking F1 improves consistently.",
        },
        {
            "metric": "pair_count",
            "baseline": f"{baseline['pair_count']}",
            "v7_exp2_mean": f"{mean_std['pair_count']['mean']:.1f}",
            "v7_exp2_std": f"{mean_std['pair_count']['std']:.1f}",
            "direction_vs_baseline": f"{same_direction['pair_count']}/{total_runs} runs better",
            "stability_tag": stability_tag["pair_count"],
            "comment": "standard-match usable pairs increase consistently.",
        },
        {
            "metric": "mean_L2",
            "baseline": f"{baseline['mean_L2']:.2f}",
            "v7_exp2_mean": f"{mean_std['mean_L2']['mean']:.2f}",
            "v7_exp2_std": f"{mean_std['mean_L2']['std']:.2f}",
            "direction_vs_baseline": f"{same_direction['mean_L2']}/{total_runs} runs better",
            "stability_tag": stability_tag["mean_L2"],
            "comment": "point localization error decreases consistently.",
        },
        {
            "metric": "|dy|",
            "baseline": f"{baseline['mean_abs_dy']:.2f}",
            "v7_exp2_mean": f"{mean_std['mean_abs_dy']['mean']:.2f}",
            "v7_exp2_std": f"{mean_std['mean_abs_dy']['std']:.2f}",
            "direction_vs_baseline": f"{same_direction['mean_abs_dy']}/{total_runs} runs better",
            "stability_tag": stability_tag["mean_abs_dy"],
            "comment": "y-direction error reduction is the most stable gain.",
        },
    ]
    return rows, stability_tag


def build_stability_analysis_md(path: Path, rows: list[dict], paper_summary: dict) -> None:
    checkpoint = paper_summary["stability_validation"]["checkpoint_stability"]
    lines = [
        "# 稳定性分析（中文）",
        "",
        "## 总结",
    ]
    for row in rows:
        lines.append(
            f"- {row['metric']}: baseline={row['baseline']}, v7_exp2 mean±std={row['v7_exp2_mean']} ± {row['v7_exp2_std']}，{row['direction_vs_baseline']}，判定为 {row['stability_tag']}。"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "- 稳定收益指标：has_picking F1、pair_count、mean L2、|dy|。这些指标在 3 次独立运行中都保持相对 baseline 的同方向改善。",
            "- seed sensitive 指标：grape AP。当前 3 次运行中有 2 次优于 baseline，1 次低于 baseline，因此不应把 AP 写成最强主结论。",
            "- checkpoint 层面也支持主判断：best_composite、best_grape_ap、last 都维持了 F1 / pair_count / mean L2 / |dy| 的较优趋势，说明主收益不是单一 checkpoint 偶然出现。",
            "",
            "## checkpoint 补充",
            f"- best_composite: AP={checkpoint['best_composite.pth']['grape_AP']:.4f}, F1={checkpoint['best_composite.pth']['has_picking_F1']:.4f}, pair_count={checkpoint['best_composite.pth']['pair_count']}, mean L2={checkpoint['best_composite.pth']['mean_L2']:.2f}, |dy|={checkpoint['best_composite.pth']['mean_abs_dy']:.2f}",
            f"- last: AP={checkpoint['last.pth']['grape_AP']:.4f}, F1={checkpoint['last.pth']['has_picking_F1']:.4f}, pair_count={checkpoint['last.pth']['pair_count']}, mean L2={checkpoint['last.pth']['mean_L2']:.2f}, |dy|={checkpoint['last.pth']['mean_abs_dy']:.2f}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_external_baselines(paper_summary: dict, local_rtdetr_json: dict, pose_reference_url: str, rtdetr_reference_url: str) -> list[dict]:
    our_metrics = paper_summary["v7_exp2"]["test_metrics"]
    grape_metrics = local_rtdetr_json.get("classes", {}).get("grape", {})
    picking_metrics = local_rtdetr_json.get("classes", {}).get("picking", {})
    source_local = str(DEFAULT_LOCAL_RTDETR_JSON.resolve())
    return [
        {
            "method": "ours_v7_exp2",
            "family": "single-stage per-grape point localization",
            "task_formulation": "grape bbox + has_picking + point localization",
            "reproduction_status": "local_main_model",
            "grape_metric": f"AP={our_metrics['grape_AP']:.4f}",
            "picking_metric": (
                f"F1={our_metrics['has_picking_F1']:.4f}; "
                f"mean L2={our_metrics['mean_L2']:.2f}px; |dy|={our_metrics['mean_abs_dy']:.2f}px"
            ),
            "direct_comparability": "main reference",
            "takeaway": "Main model focuses on per-grape visibility and point localization rather than tiny-box detection AP.",
            "source": paper_summary["v7_exp2"]["summary_path"],
        },
        {
            "method": "ultralytics_rtdetr_dual_box_local",
            "family": "Ultralytics RT-DETR detection",
            "task_formulation": "grape bbox + picking bbox independent detection",
            "reproduction_status": "locally_reproduced",
            "grape_metric": f"AP={float(grape_metrics.get('AP', 0.0)):.4f}",
            "picking_metric": f"picking AP={float(picking_metrics.get('AP', 0.0)):.4f}",
            "direct_comparability": "partial",
            "takeaway": "Strong grape detection, but tiny picking-box AP remains low and does not solve per-grape point association.",
            "source": source_local,
        },
        {
            "method": "ultralytics_pose_reference",
            "family": "Ultralytics YOLO pose/keypoint",
            "task_formulation": "object keypoint / pose estimation",
            "reproduction_status": "official_reference_not_reproduced",
            "grape_metric": "N/A",
            "picking_metric": "official pose metrics use mAP(P), not has_picking F1 + point L2",
            "direct_comparability": "low",
            "takeaway": "Closest common family to point localization, but it does not directly encode per-grape has_picking and uses a different evaluation protocol.",
            "source": pose_reference_url,
        },
        {
            "method": "rtdetr_reference_docs",
            "family": "Ultralytics RT-DETR official docs",
            "task_formulation": "end-to-end object detection",
            "reproduction_status": "official_reference_not_reproduced",
            "grape_metric": "official docs report generic object-detection AP",
            "picking_metric": "N/A",
            "direct_comparability": "low",
            "takeaway": "Useful as an external detector family reference, but not aligned with our point localization objective.",
            "source": rtdetr_reference_url,
        },
    ]


def build_external_comparison_md(path: Path, external_rows: list[dict], pose_reference_url: str, rtdetr_reference_url: str) -> None:
    lines = [
        "# 外部对比（中文）",
        "",
        "## 为什么只选 1~2 条外部对照",
        "- 这轮不追求大规模复现，而是优先选择最能回答任务合理性的常见方案。",
        "- 因此保留一条本地复现的 Ultralytics RT-DETR 双框检测基线，以及一条官方 pose/keypoint 参考路线。",
        "",
        "## 核心结论",
        "- 本地复现的 Ultralytics RT-DETR 双框基线在 grape 检测上很强，但 picking 作为独立小框时 AP 只有 0.2760，说明 tiny-box detection 本身并不适合作为主线目标。",
        "- keypoint / pose 路线在方法家族上与我们的 point localization 更接近，但官方评估口径是 pose mAP(P)，并不直接对应 has_picking F1 + point L2，因此更适合作为任务家族参考，而不是严格数值对位。",
        "- 因此我们当前最合理的对外定位不是“比所有检测器都高”，而是“相较常见双框检测更适合解决 per-grape visibility + point localization 的组合任务”。",
        "",
        "## 来源",
        f"- Ultralytics pose docs: [{pose_reference_url}]({pose_reference_url})",
        f"- Ultralytics RT-DETR docs: [{rtdetr_reference_url}]({rtdetr_reference_url})",
        f"- Local reproduced Ultralytics RT-DETR metrics: [best_per_class_metrics.json](<{external_rows[1]['source']}>)",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def scene_row_lookup(scene_rows: list[dict[str, str]], family: str, label: str) -> dict[str, str]:
    for row in scene_rows:
        if row["slice_family"] == family and row["slice_label"] == label:
            return row
    raise KeyError(f"Missing scene slice row for {family}/{label}")


def build_error_attribution_md(
    path: Path,
    paper_summary: dict,
    scene_rows: list[dict[str, str]],
    strengthening_summary: dict | None = None,
) -> None:
    baseline = paper_summary["formal_baseline"]["test_metrics"]
    exp1 = paper_summary["v7_exp1"]["test_metrics"]
    exp2 = paper_summary["v7_exp2"]["test_metrics"]
    decoupled = paper_summary["mechanism_evidence"]["decoupled_point_summary"]
    row_multi = scene_row_lookup(scene_rows, "single_vs_multi", "multi_adjacent")
    row_heavy = scene_row_lookup(scene_rows, "occlusion_proxy", "heavy")
    row_small = scene_row_lookup(scene_rows, "size_group", "small")

    lines = [
        "# 误差归因（中文）",
        "",
        "## exp1 为什么只能部分改善",
        f"- exp1 相对 baseline，将 mean L2 从 {baseline['mean_L2']:.2f}px 降到 {exp1['mean_L2']:.2f}px，|dy| 从 {baseline['mean_abs_dy']:.2f}px 降到 {exp1['mean_abs_dy']:.2f}px，说明 top-center 的几何表达确实缓解了 y 方向漂移。",
        f"- 但 exp1 的 has_picking F1 从 {baseline['has_picking_F1']:.4f} 降到 {exp1['has_picking_F1']:.4f}，pair_count 从 {baseline['pair_count']} 降到 {exp1['pair_count']}，说明“点更会回归”并没有自动变成“整条检测-可见性-关联链条更稳”。",
        f"- decoupled 诊断也支持这一点：exp1 在 oracle(any visible pred) 下的 mean L2={decoupled['v7_exp1']['oracle_candidates']['any_visible_pred']['mean_l2_px']:.2f}px，甚至优于 exp2，但标准匹配 pair_count 反而更低。",
        "",
        "## exp2 为什么形成完整收益",
        f"- exp2 在保留 exp1 的低 |dy| 基础上，把 F1 提升到 {exp2['has_picking_F1']:.4f}，pair_count 提升到 {exp2['pair_count']}，说明 query_box + top local cue 的联合建模把检测、可见性和点定位链条重新连起来了。",
        f"- 在更严格的 IoU>=0.85 条件下，exp2 的 mean L2 / |dy| 为 {decoupled['v7_exp2']['iou_conditioned']['ge_0.85']['mean_l2_px']:.2f}px / {decoupled['v7_exp2']['iou_conditioned']['ge_0.85']['mean_abs_dy_px']:.2f}px，优于 exp1 的 {decoupled['v7_exp1']['iou_conditioned']['ge_0.85']['mean_l2_px']:.2f}px / {decoupled['v7_exp1']['iou_conditioned']['ge_0.85']['mean_abs_dy_px']:.2f}px。",
        f"- 这说明 exp2 的优势不只是“裸 point 更近”，而是让更正确的 grape query 拿到了更有用的顶部局部视觉线索。",
        "",
        "## 当前 residual error 更像什么",
        f"- 复杂场景关联与链条不稳仍然存在：多串相邻场景下，exp2 pair_recall={float(row_multi['exp2_pair_recall']):.3f}，heavy 遮挡下 pair_recall={float(row_heavy['exp2_pair_recall']):.3f}，都明显低于单串和轻遮挡场景。",
        f"- 小串仍是主要短板：small 场景下 exp2 mean L2={float(row_small['exp2_mean_L2']):.2f}px，|dy|={float(row_small['exp2_mean_abs_dy']):.2f}px，且 pair_recall 只有 {float(row_small['exp2_pair_recall']):.3f}。",
        f"- bbox 误差并非零影响，但已不是唯一主因：exp2 在 oracle_iou50 条件下 mean L2={decoupled['v7_exp2']['oracle_candidates']['iou_ge_0.50_visible']['mean_l2_px']:.2f}px，优于标准匹配 {decoupled['v7_exp2']['standard_match']['mean_l2_px']:.2f}px，说明 bbox/匹配链条仍会传播误差；但即使 IoU>=0.85，mean L2 仍有 {decoupled['v7_exp2']['iou_conditioned']['ge_0.85']['mean_l2_px']:.2f}px，说明纯 point 误差也没有完全解决。",
        "",
        "## 总判断",
        "- 当前 residual error 更像是“复杂场景关联链条 + small grape 表达不足 + 尚未完全压净的 dy 残差”的叠加，而不是单一的 grape bbox 不准。",
    ]
    if strengthening_summary is not None:
        sw = strengthening_summary["metrics"]["small_weight"]
        sw_small = scene_row_lookup(strengthening_summary["scene_slice_rows"], "size_group", "small")
        sw_heavy = scene_row_lookup(strengthening_summary["scene_slice_rows"], "occlusion_proxy", "heavy")
        lines.extend(
            [
                "",
                "## 候选 A（small-grape weighted point loss）补充",
                f"- 候选 A 的整体走势是“补 small 和 heavy，但不是无代价替代”：AP {exp2['grape_AP']:.4f} -> {sw['grape_AP']:.4f}，F1 {exp2['has_picking_F1']:.4f} -> {sw['has_picking_F1']:.4f}，pair_count {exp2['pair_count']} -> {sw['pair_count']}，mean L2 {exp2['mean_L2']:.2f} -> {sw['mean_L2']:.2f}px，|dy| {exp2['mean_abs_dy']:.2f} -> {sw['mean_abs_dy']:.2f}px。",
                f"- 它确实打中了 small grape：pair_recall {float(sw_small['exp2_pair_recall']):.3f} -> {float(sw_small['new_pair_recall']):.3f}，mean L2 {float(sw_small['exp2_mean_L2']):.2f} -> {float(sw_small['new_mean_L2']):.2f}px，|dy| {float(sw_small['exp2_mean_abs_dy']):.2f} -> {float(sw_small['new_mean_abs_dy']):.2f}px。",
                f"- 对 heavy occlusion 也有部分帮助：mean L2 {float(sw_heavy['exp2_mean_L2']):.2f} -> {float(sw_heavy['new_mean_L2']):.2f}px，|dy| {float(sw_heavy['exp2_mean_abs_dy']):.2f} -> {float(sw_heavy['new_mean_abs_dy']):.2f}px，但 pair_recall 没有继续提升。",
                "- 因此候选 A 更适合被写成“针对 small / heavy 难例的定向补强证据”，而不是直接替换 v7_exp2 的主模型版本。",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sci_claims_md(path: Path, paper_summary: dict, strengthening_summary: dict | None = None) -> None:
    mean_std = paper_summary["stability_validation"]["v7_exp2_mean_std"]
    lines = [
        "# SCI claims（中文）",
        "",
        "## 当前真正问题是什么",
        "- 真正的问题不是单纯把 grape 检出来，而是在同一单阶段框架里同时完成 grape 检测、采摘点可见性判别和采摘点定位，并尽量减少由实例重复和邻串干扰带来的链式误差。",
        "",
        "## 为什么独立 picking bbox 不合适",
        "- 独立 picking bbox 会把采摘点变成第二类小目标，容易出现重复框、漏框和与葡萄串的误关联。",
        "- 本地复现的 Ultralytics RT-DETR 双框基线也支持这一判断：grape AP 很强，但 picking AP 只有 0.2760，说明 tiny-box detection 不是这项任务最稳的主线。",
        "",
        "## 为什么 per-grape has_picking + point localization 更合理",
        "- 每个 grape query 直接判断 has_picking 并回归 point，可以天然保留“一个葡萄串一套可见性与坐标”的结构，避免后处理再做硬绑定。",
        "- 这条建模路线也更接近采摘任务的实际需求，因为最终控制量是点位而不是第二个独立框。",
        "",
        "## 为什么 v7_exp2 有效",
        "- top-center 先压低了 dy，query_box + top local cue 再把检测、可见性和 point 链条连起来，所以收益最终同时落在 F1、pair_count、mean L2 和 |dy| 上。",
        "",
        "## 哪些收益是稳定的",
        f"- 3 次独立运行下，v7_exp2 的 F1={mean_std['F1']['mean']:.4f} ± {mean_std['F1']['std']:.4f}，pair_count={mean_std['pair_count']['mean']:.1f} ± {mean_std['pair_count']['std']:.1f}，mean L2={mean_std['mean_L2']['mean']:.2f} ± {mean_std['mean_L2']['std']:.2f}px，|dy|={mean_std['mean_abs_dy']['mean']:.2f} ± {mean_std['mean_abs_dy']['std']:.2f}px，稳定优于 baseline。",
        "",
        "## 哪些结论不能夸大",
        "- grape AP 仍有 seed sensitivity，不能写成绝对稳定主结论。",
        "- v7_exp2 也没有在 small grape 和 heavy occlusion 上全面解决误差，这两类仍然是当前主要残差来源。",
    ]
    if strengthening_summary is not None:
        sw = strengthening_summary["metrics"]["small_weight"]
        sw_small = scene_row_lookup(strengthening_summary["scene_slice_rows"], "size_group", "small")
        lines.extend(
            [
                "",
                "## 定向补强实验说明了什么",
                f"- 候选 A（small-grape weighted point loss）进一步证明 residual error 里确实有一部分来自 small grape 表达不足：small 场景 pair_recall {float(sw_small['exp2_pair_recall']):.3f} -> {float(sw_small['new_pair_recall']):.3f}，mean L2 {float(sw_small['exp2_mean_L2']):.2f} -> {float(sw_small['new_mean_L2']):.2f}px。",
                f"- 但这版没有成为新主模型，因为它的整体取舍是 AP {paper_summary['v7_exp2']['test_metrics']['grape_AP']:.4f} -> {sw['grape_AP']:.4f}，|dy| {paper_summary['v7_exp2']['test_metrics']['mean_abs_dy']:.2f} -> {sw['mean_abs_dy']:.2f}px；因此更适合作为问题驱动稿件里的“ targeted strengthening evidence ”，而不是替代主结论。",
            ]
        )
    lines.extend(
        [
            "",
            "## 离 SCI 三区还差什么",
            "- 还差把主模型、定向补强、外部对比和误差归因收成一套更统一的主表与主图；方法方向本身已经明确，不需要再大范围试错。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sci_contribution_md(path: Path) -> None:
    lines = [
        "# SCI contribution draft（中文）",
        "",
        "## 研究问题",
        "- 在葡萄串识别与采摘点定位场景中，如何在单阶段 RT-DETRv4 框架内同时实现 grape 检测、采摘点可见性判断与采摘点坐标定位，并避免独立 picking bbox 带来的实例重复与误关联问题。",
        "",
        "## 方法改进点",
        "- 将任务统一为 per-grape has_picking + point localization，而不是独立 picking bbox 检测。",
        "- 采用顶部锚点坐标表达抑制 y 方向漂移。",
        "- 在 query_box 绑定基础上引入顶部局部视觉 cue，增强每个 grape query 的点定位判别能力。",
        "",
        "## 解决的关键问题",
        "- 缓解了 dual-box 方案中 tiny picking box 检测不稳和后续实例关联困难的问题。",
        "- 降低了 dy 残差，并提升了 has_picking 判别与标准匹配链条的稳定性。",
        "",
        "## 创新点",
        "- 创新点主要体现在任务建模与轻量结构改进，而不是引入复杂的 two-stage 或大规模模块堆叠。",
        "- 进一步提供了解耦式误差诊断，把 point 误差拆成标准匹配、IoU 条件和 oracle 候选三个层次。",
        "",
        "## 实验结论",
        "- v7_exp2 在正式 test 上同时改善了 has_picking F1、pair_count、mean L2 和 |dy|。",
        "- 3 次独立运行证明主收益集中在 point 链条相关指标，AP 存在 seed 敏感性，因此论文主结论应聚焦 visibility + point localization 的稳定收益。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sci_storyline_md(path: Path, strengthening_summary: dict | None = None) -> None:
    lines = [
        "# SCI storyline（中文）",
        "",
        "## 问题驱动主线",
        "- 葡萄采摘点不是第二个独立目标，而是附着在 grape 实例上的可见局部点位。",
        "- 因此单纯把 picking 当独立 bbox 检测，会在小目标、邻串相邻和复杂遮挡场景下放大误检、重复检和误关联。",
        "",
        "## 方法转折",
        "- 将任务重构为 per-grape has_picking + point localization 后，模型开始具备更符合任务结构的表达。",
        "- 但仅靠几何绑定仍不足以形成完整收益，因此进一步引入顶部锚点表达和顶部局部视觉 cue。",
        "",
        "## 证据链",
        "- 主实验说明 v7_exp2 相对 baseline 在 visibility 判别和 point 指标上同时变好。",
        "- 多 seed 与 checkpoint 稳定性说明这些收益不是单次偶然。",
        "- 解耦评估与场景切片说明收益主要来自检测-可见性-关联链条稳定化，同时也暴露出 small grape 和 heavy occlusion 仍是剩余难点。",
    ]
    if strengthening_summary is not None:
        sw = strengthening_summary["metrics"]["small_weight"]
        lines.extend(
            [
                "- 候选 A 进一步表明：当 point loss 对 small grape 定向加权后，small / heavy 场景的误差可以继续下降，但会带来 AP 与整体 |dy| 的新取舍，因此它更像补强证据而不是替换主线。",
            ]
        )
    lines.extend(
        [
            "",
            "## 不夸大的结论",
            "- 当前方法不是彻底解决所有实例绑定问题，而是在单阶段框架内更合理地组织 visibility 与 point prediction。",
            f"- 现阶段最稳的叙事结构是“v7_exp2 作为主模型，small_weight 作为困难场景补强证据”，而不是继续把每个补丁都包装成新主模型。",
            "- 稿件最稳的卖点不是“全面超越所有 detector”，而是“给出一种更适合葡萄采摘点任务结构的单阶段建模与诊断方式”。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sci_ready_summary(
    paper_summary: dict,
    stability_rows: list[dict],
    external_rows: list[dict],
    out_dir: Path,
    strengthening_summary: dict | None = None,
) -> dict:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "formal_baseline": paper_summary["formal_baseline"],
        "current_main_model": paper_summary["v7_exp2"],
        "stable_metrics": [row["metric"] for row in stability_rows if row["stability_tag"] == "stable_gain"],
        "seed_sensitive_metrics": [row["metric"] for row in stability_rows if row["stability_tag"] == "seed_sensitive"],
        "external_comparison": external_rows,
        "remaining_risks": [
            "small grape remains the hardest scene slice",
            "heavy occlusion / multi-adjacent scenes still have large residual errors",
            "AP cannot be claimed as fully stable across seeds",
        ],
        "outputs": {
            "final_mean_std_table": str((out_dir / "final_mean_std_table.csv").resolve()),
            "stability_analysis": str((out_dir / "stability_analysis_zh.md").resolve()),
            "external_baseline_table": str((out_dir / "external_baseline_table.csv").resolve()),
            "external_comparison": str((out_dir / "external_comparison_zh.md").resolve()),
            "error_attribution": str((out_dir / "error_attribution_zh.md").resolve()),
            "scene_slice_table": str((out_dir / "scene_slice_table.csv").resolve()),
            "representative_cases": str((out_dir / "representative_cases").resolve()),
            "sci_claims": str((out_dir / "sci_claims_zh.md").resolve()),
            "sci_contribution_draft": str((out_dir / "sci_contribution_draft_zh.md").resolve()),
            "sci_storyline": str((out_dir / "sci_storyline_zh.md").resolve()),
        },
    }
    if strengthening_summary is not None:
        summary["strengthening_variant"] = {
            "name": "small-grape weighted point loss",
            "summary_path": strengthening_summary["new_summary_path"],
            "metrics": strengthening_summary["metrics"]["small_weight"],
            "judgement": (
                "Useful as targeted strengthening evidence for small grape / heavy occlusion, "
                "but not strong enough to replace v7_exp2 as the main model."
            ),
        }
    return summary


def build_sci_representative_cases(
    out_dir: Path,
    baseline_summary: dict,
    exp2_summary: dict,
    dataset_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    top_k: int,
) -> dict:
    baseline_config = Path(baseline_summary["config"]).resolve()
    baseline_ckpt = Path(baseline_summary["primary_checkpoint"]).resolve()
    exp2_config = Path(exp2_summary["config"]).resolve()
    exp2_ckpt = Path(exp2_summary["primary_checkpoint"]).resolve()

    _, baseline_records = evaluate_split(
        config_path=baseline_config,
        checkpoint_path=baseline_ckpt,
        split="test",
        dataset_root=dataset_root.resolve(),
        collect_predictions=True,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    _, exp2_records = evaluate_split(
        config_path=exp2_config,
        checkpoint_path=exp2_ckpt,
        split="test",
        dataset_root=dataset_root.resolve(),
        collect_predictions=True,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    baseline_threshold = get_has_picking_threshold(baseline_config)
    exp2_threshold = get_has_picking_threshold(exp2_config)

    baseline_cases, baseline_correct, baseline_mismatches = build_case_indexes(baseline_records, baseline_threshold)
    exp2_cases, exp2_correct, exp2_mismatches = build_case_indexes(exp2_records, exp2_threshold)
    baseline_lookup = {int(record["image_id"]): record for record in baseline_records}
    exp2_lookup = {int(record["image_id"]): record for record in exp2_records}
    scene_slices, _ = build_scene_slices(baseline_records)
    small_keys = scene_slices["size_group"]["small"]

    baseline_fixed = []
    dy_fixed = []
    for key, e_case in exp2_cases.items():
        b_case = baseline_cases.get(key)
        if e_case is None:
            continue
        b_bad = point_case_badness(b_case)
        e_bad = point_case_badness(e_case)
        if e_bad + 8.0 < b_bad:
            baseline_fixed.append(
                {
                    "image_id": int(e_case["image_id"]),
                    "gt_index": int(e_case["gt_index"]),
                    "baseline_case": b_case,
                    "exp2_case": e_case,
                    "score": b_bad - e_bad,
                    "footer": f"baseline fixed: baseline badness={b_bad:.1f}, exp2 badness={e_bad:.1f}",
                }
            )
        if b_case is not None and bool(b_case.get("pred_has_picking")) and bool(e_case.get("pred_has_picking")):
            dy_gain = abs(float(b_case.get("dy_px", 0.0))) - abs(float(e_case.get("dy_px", 0.0)))
            if dy_gain >= 6.0 and float(e_case.get("l2_px", 0.0)) <= float(b_case.get("l2_px", 0.0)) + 2.0:
                dy_fixed.append(
                    {
                        "image_id": int(e_case["image_id"]),
                        "gt_index": int(e_case["gt_index"]),
                        "baseline_case": b_case,
                        "exp2_case": e_case,
                        "score": dy_gain,
                        "footer": f"dy improved: |dy| {abs(float(b_case['dy_px'])):.1f}px -> {abs(float(e_case['dy_px'])):.1f}px",
                    }
                )

    cross_instance_remaining = []
    for item in exp2_mismatches:
        key = (int(item["image_id"]), int(item["gt_index"]))
        cross_instance_remaining.append(
            {
                "image_id": int(item["image_id"]),
                "gt_index": int(item["gt_index"]),
                "baseline_case": baseline_cases.get(key),
                "exp2_case": item,
                "score": float(item.get("cross_instance_score", 0.0)),
                "footer": (
                    f"cross-instance mismatch remains: "
                    f"L2={float(item.get('l2_px', 0.0)):.1f}px, "
                    f"other_gt_dist={float(item.get('other_gt_distance_px', 0.0)):.1f}px"
                ),
            }
        )

    small_grape_remaining = []
    for key in small_keys:
        e_case = exp2_cases.get(key)
        b_case = baseline_cases.get(key)
        e_bad = point_case_badness(e_case)
        if e_bad < 35.0:
            continue
        image_id, gt_index = key
        small_grape_remaining.append(
            {
                "image_id": int(image_id),
                "gt_index": int(gt_index),
                "baseline_case": b_case,
                "exp2_case": e_case,
                "score": e_bad,
                "footer": f"small grape remaining failure: exp2 badness={e_bad:.1f}",
            }
        )

    payloads = {
        "baseline_fixed": sorted(baseline_fixed, key=lambda item: item["score"], reverse=True),
        "dy_fixed": sorted(dy_fixed, key=lambda item: item["score"], reverse=True),
        "cross_instance_remaining": sorted(cross_instance_remaining, key=lambda item: item["score"], reverse=True),
        "small_grape_remaining": sorted(small_grape_remaining, key=lambda item: item["score"], reverse=True),
    }

    rep_root = out_dir / "representative_cases"
    ensure_dir(rep_root)
    output = {}
    for category, entries in payloads.items():
        category_dir = rep_root / category
        ensure_dir(category_dir)
        output_entries = []
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
            output_entries.append(
                {
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "output": str(out_path.resolve()),
                    "footer": item["footer"],
                }
            )
        output[category] = output_entries
    return output


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    ensure_dir(out_dir)

    paper_summary = load_json(args.paper_ready_dir / "paper_ready_summary.json")
    scene_rows = load_csv_rows(args.paper_ready_dir / "scene_slice_table.csv")
    local_rtdetr_json = json.loads(args.local_rtdetr_json.read_text(encoding="utf-8"))
    strengthening_summary = None
    if args.strengthening_summary and args.strengthening_summary.exists():
        strengthening_summary = load_json(args.strengthening_summary)

    baseline_summary = load_json(Path(paper_summary["formal_baseline"]["summary_path"]))
    exp2_summary = load_json(Path(paper_summary["v7_exp2"]["summary_path"]))

    stability_rows, _ = build_final_mean_std_rows(paper_summary)
    write_csv(
        out_dir / "final_mean_std_table.csv",
        stability_rows,
        ["metric", "baseline", "v7_exp2_mean", "v7_exp2_std", "direction_vs_baseline", "stability_tag", "comment"],
    )
    build_stability_analysis_md(out_dir / "stability_analysis_zh.md", stability_rows, paper_summary)

    external_rows = build_external_baselines(
        paper_summary,
        local_rtdetr_json,
        args.pose_reference_url,
        args.rtdetr_reference_url,
    )
    write_csv(
        out_dir / "external_baseline_table.csv",
        external_rows,
        ["method", "family", "task_formulation", "reproduction_status", "grape_metric", "picking_metric", "direct_comparability", "takeaway", "source"],
    )
    build_external_comparison_md(out_dir / "external_comparison_zh.md", external_rows, args.pose_reference_url, args.rtdetr_reference_url)

    shutil.copy2(args.paper_ready_dir / "scene_slice_table.csv", out_dir / "scene_slice_table.csv")
    build_error_attribution_md(out_dir / "error_attribution_zh.md", paper_summary, scene_rows, strengthening_summary)
    representative_cases = build_sci_representative_cases(
        out_dir=out_dir,
        baseline_summary=baseline_summary,
        exp2_summary=exp2_summary,
        dataset_root=args.dataset_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k_cases,
    )

    build_sci_claims_md(out_dir / "sci_claims_zh.md", paper_summary, strengthening_summary)
    build_sci_contribution_md(out_dir / "sci_contribution_draft_zh.md")
    build_sci_storyline_md(out_dir / "sci_storyline_zh.md", strengthening_summary)

    sci_summary = build_sci_ready_summary(paper_summary, stability_rows, external_rows, out_dir, strengthening_summary)
    sci_summary["representative_cases"] = representative_cases
    sci_summary["scene_slice_table_source"] = str((args.paper_ready_dir / "scene_slice_table.csv").resolve())
    sci_summary["local_external_baseline_source"] = str(args.local_rtdetr_json.resolve())
    (out_dir / "sci_ready_summary.json").write_text(
        json.dumps(sci_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[sci-ready] wrote {out_dir / 'final_mean_std_table.csv'}")
    print(f"[sci-ready] wrote {out_dir / 'stability_analysis_zh.md'}")
    print(f"[sci-ready] wrote {out_dir / 'external_baseline_table.csv'}")
    print(f"[sci-ready] wrote {out_dir / 'external_comparison_zh.md'}")
    print(f"[sci-ready] wrote {out_dir / 'error_attribution_zh.md'}")
    print(f"[sci-ready] wrote {out_dir / 'scene_slice_table.csv'}")
    print(f"[sci-ready] wrote {out_dir / 'representative_cases'}")
    print(f"[sci-ready] wrote {out_dir / 'sci_claims_zh.md'}")
    print(f"[sci-ready] wrote {out_dir / 'sci_contribution_draft_zh.md'}")
    print(f"[sci-ready] wrote {out_dir / 'sci_storyline_zh.md'}")
    print(f"[sci-ready] wrote {out_dir / 'sci_ready_summary.json'}")


if __name__ == "__main__":
    main()
