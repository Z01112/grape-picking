from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


BASELINE = {
    "name": "V7_EXP2_MAIN",
    "summary": REPO_ROOT / "outputs/03_global_analysis/post_cleanup_v7_exp2_report_20260525/summary.json",
}

EXPERIMENTS = [
    {
        "name": "EMA_FUSION",
        "summary": REPO_ROOT / "outputs/02_encoder_experiments/encoder_ema_fusion_100e_backbone_pretrain_20260526/report/summary.json",
        "decision": "mechanism_only",
        "role": "detector_recall_candidate",
        "reason": "AP/F1/pair improved, but default mean L2 and small/medium L2 regressed.",
    },
    {
        "name": "EMA_BIFPN",
        "summary": REPO_ROOT / "outputs/02_encoder_experiments/encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526/report/summary.json",
        "decision": "candidate_pending",
        "role": "next_mainline_base",
        "reason": "AP/F1/pair and mean L2 improved, but p90 and small-object L2 regressed.",
    },
    {
        "name": "POINT_O2M_AUX",
        "summary": REPO_ROOT / "outputs/02_mechanism_experiments/point_o2m_aux_100e_backbone_pretrain_20260527/report/summary.json",
        "decision": "mechanism_only",
        "role": "point_error_mechanism",
        "reason": "Mean/p90 L2 improved strongly, but F1 and pair_count fell below mainline coverage.",
    },
    {
        "name": "POINT_QUALITY_ALIGN",
        "summary": REPO_ROOT / "outputs/02_mechanism_experiments/point_quality_align_100e_backbone_pretrain_20260527/report/summary.json",
        "decision": "mechanism_only",
        "role": "quality_filter_mechanism",
        "reason": "F1/pair improved slightly, but default AP/mean L2/p90 do not support mainline replacement.",
    },
    {
        "name": "PICKING_DN_TEACHER_ROI",
        "summary": REPO_ROOT / "outputs/02_mechanism_experiments/picking_dn_teacher_roi_100e_backbone_pretrain_20260527/report/summary.json",
        "decision": "mainline_reject",
        "role": "rejected_mechanism",
        "reason": "No stable AP/F1/point-localization gain under default test protocol.",
    },
    {
        "name": "POINT_DECOUPLED_WEAK_HEATMAP_120E",
        "summary": REPO_ROOT / "outputs/02_sci_main_experiments/point_decoupled_weak_heatmap_120e_continue20_from_best_20260528/report/summary.json",
        "decision": "mainline_reject",
        "role": "weak_label_mechanism",
        "reason": "Valid improved locally, but formal test AP/F1/pair/mean L2/p90 regressed against V7_EXP2_MAIN.",
    },
    {
        "name": "EMA_BIFPN_POINT_DECOUPLED_V1",
        "summary": REPO_ROOT / "outputs/02_combined_experiments/ema_bifpn_point_decoupled_v1_100e_backbone_pretrain_20260529/report/summary.json",
        "decision": "mainline_reject",
        "role": "failed_combo",
        "reason": "The EMA_BIFPN detector gain did not survive the decoupled point/O2M combination; F1, pair_count, mean L2, and p90 all regressed.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a formal failed-experiment audit table for GPPoint-DETR.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/03_global_analysis/failed_experiment_audit_{datetime.now():%Y%m%d}",
    )
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE["summary"])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def split_metrics(summary: dict, split: str = "test") -> dict:
    data = summary["primary_checkpoint_split_summary"][split]
    det = data.get("grape_detection", {})
    has = data.get("has_picking", {})
    point = data.get("picking_point", {})
    size = point.get("size_group_l2_px", {}) or {}
    quality = point.get("quality_aligned", {}) or {}
    return {
        "AP": float(det.get("AP", 0.0)),
        "AP50": float(det.get("AP50", 0.0)),
        "AR100": float(det.get("AR100", 0.0)),
        "has_precision": float(has.get("precision", 0.0)),
        "has_recall": float(has.get("recall", 0.0)),
        "has_f1": float(has.get("f1", 0.0)),
        "pair_count": int(point.get("pair_count", 0)),
        "mean_l2": float(point.get("mean_l2_px", 0.0)),
        "median_l2": float(point.get("median_l2_px", 0.0)),
        "p90_l2": float(point.get("p90_l2_px", 0.0)),
        "dx": float(point.get("mean_abs_dx_px", point.get("mae_x_px", 0.0))),
        "dy": float(point.get("mean_abs_dy_px", point.get("mae_y_px", 0.0))),
        "small_l2": float((size.get("small") or {}).get("mean_l2_px", 0.0)),
        "medium_l2": float((size.get("medium") or {}).get("mean_l2_px", 0.0)),
        "large_l2": float((size.get("large") or {}).get("mean_l2_px", 0.0)),
        "quality_pair_count": int(quality.get("point_pair_count", quality.get("count", 0)) or 0),
        "quality_mean_l2": float(quality.get("mean_l2_px", 0.0) or 0.0),
        "quality_p90_l2": float(quality.get("p90_l2_px", 0.0) or 0.0),
    }


def delta(value: float, baseline: float) -> float:
    return float(value) - float(baseline)


def fmt(value: float | int, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def cleanup_hint(decision: str) -> str:
    if decision == "candidate_pending":
        return "keep checkpoints until next combo result"
    if decision == "mechanism_only":
        return "keep reports/config/logs; checkpoint cleanup allowed after user confirmation"
    return "keep reports/config/logs; checkpoint cleanup recommended after user confirmation"


def build_rows(baseline_summary: Path) -> tuple[dict, list[dict]]:
    baseline = split_metrics(load_json(baseline_summary))
    rows = []
    for exp in EXPERIMENTS:
        summary = load_json(exp["summary"])
        metrics = split_metrics(summary)
        row = {
            "name": exp["name"],
            "decision": exp["decision"],
            "role": exp["role"],
            **metrics,
            "delta_AP": delta(metrics["AP"], baseline["AP"]),
            "delta_has_f1": delta(metrics["has_f1"], baseline["has_f1"]),
            "delta_pair_count": int(metrics["pair_count"] - baseline["pair_count"]),
            "delta_mean_l2": delta(metrics["mean_l2"], baseline["mean_l2"]),
            "delta_p90_l2": delta(metrics["p90_l2"], baseline["p90_l2"]),
            "delta_small_l2": delta(metrics["small_l2"], baseline["small_l2"]),
            "reason": exp["reason"],
            "cleanup_hint": cleanup_hint(exp["decision"]),
            "summary": str(exp["summary"]),
        }
        rows.append(row)
    return baseline, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, baseline: dict, rows: list[dict]) -> None:
    lines = [
        "# Failed Experiment Audit",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "- Reference: `V7_EXP2_MAIN fair retrain`, formal test protocol, `has_picking_threshold=0.5`.",
        f"- Reference metrics: AP {baseline['AP']:.4f}, has F1 {baseline['has_f1']:.4f}, pair {baseline['pair_count']}, mean L2 {baseline['mean_l2']:.2f}, p90 L2 {baseline['p90_l2']:.2f}.",
        "",
        "| Experiment | Decision | AP | dAP | F1 | dF1 | pair | dpair | mean L2 | dmean | p90 L2 | dp90 | small L2 | dsmall | Reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['decision']} | {row['AP']:.4f} | {row['delta_AP']:+.4f} | "
            f"{row['has_f1']:.4f} | {row['delta_has_f1']:+.4f} | {row['pair_count']} | {row['delta_pair_count']:+d} | "
            f"{row['mean_l2']:.2f} | {row['delta_mean_l2']:+.2f} | {row['p90_l2']:.2f} | {row['delta_p90_l2']:+.2f} | "
            f"{row['small_l2']:.2f} | {row['delta_small_l2']:+.2f} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Cleanup dry-run",
            "",
            "| Experiment | Cleanup policy |",
            "|---|---|",
        ]
    )
    for row in rows:
        lines.append(f"| {row['name']} | {row['cleanup_hint']} |")
    lines.extend(
        [
            "",
            "## Next mainline gate",
            "",
            "- `EMA_BIFPN_POINT_DECOUPLED_V1` failed the gate and is no longer a mainline candidate.",
            "- Next work should return to the strongest isolated evidence: keep `EMA_BIFPN` as the detector-side candidate and use point quality only as a post-hoc high-precision protocol, not as an extra training loss.",
            "- Any new trained candidate must pass: test mean L2 < 23.78 or p90 L2 < 49.43, has F1 >= 0.755, pair_count >= 180, AP >= 0.632.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline, rows = build_rows(args.baseline_summary.resolve())
    write_csv(args.output_dir / "failed_experiment_audit.csv", rows)
    write_markdown(args.output_dir / "failed_experiment_audit_zh.md", baseline, rows)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
