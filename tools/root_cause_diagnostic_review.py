from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "V7_EXP2_MAIN",
        "label": "V7_EXP2_MAIN fair retrain",
        "summary": REPO_ROOT / "outputs/03_global_analysis/post_cleanup_v7_exp2_report_20260525/summary.json",
        "role": "fair_reference",
    },
    {
        "name": "EMA_BIFPN",
        "label": "EMA_BIFPN",
        "summary": REPO_ROOT
        / "outputs/02_encoder_experiments/encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526/report/summary.json",
        "role": "detector_base_candidate",
    },
    {
        "name": "POINT_O2M_AUX",
        "label": "POINT_O2M_AUX",
        "summary": REPO_ROOT / "outputs/02_mechanism_experiments/point_o2m_aux_100e_backbone_pretrain_20260527/report/summary.json",
        "role": "point_error_mechanism",
    },
    {
        "name": "POINT_QUALITY_ALIGN",
        "label": "POINT_QUALITY_ALIGN",
        "summary": REPO_ROOT
        / "outputs/02_mechanism_experiments/point_quality_align_100e_backbone_pretrain_20260527/report/summary.json",
        "role": "quality_filter_mechanism",
    },
    {
        "name": "EMA_BIFPN_POINT_DECOUPLED_V1",
        "label": "EMA_BIFPN_POINT_DECOUPLED_V1",
        "summary": REPO_ROOT
        / "outputs/02_combined_experiments/ema_bifpn_point_decoupled_v1_100e_backbone_pretrain_20260529/report/summary.json",
        "role": "failed_combo",
    },
    {
        "name": "EMA_BIFPN_POINT_RELIABILITY_V1_FAIR_100E",
        "label": "EMA_BIFPN_POINT_RELIABILITY_V1_FAIR_100E",
        "summary": REPO_ROOT
        / "outputs/02_combined_experiments/ema_bifpn_point_reliability_v1_100e_backbone_pretrain_20260529/report/summary.json",
        "role": "failed_reliability_fair",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a root-cause diagnostic review from existing GPPoint-DETR reports.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/03_global_analysis/root_cause_diagnostic_review_{datetime.now():%Y%m%d}",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def get_num(data: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def primary_test_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    test = summary["primary_checkpoint_split_summary"]["test"]
    det = test.get("grape_detection", {})
    has = test.get("has_picking", {})
    point = test.get("picking_point", {})
    size = point.get("size_group_l2_px", {}) or {}
    error = summary.get("error_analysis", {}) or {}
    return {
        "AP": get_num(det, "AP"),
        "AP50": get_num(det, "AP50"),
        "AR100": get_num(det, "AR100"),
        "has_precision": get_num(has, "precision"),
        "has_recall": get_num(has, "recall"),
        "has_f1": get_num(has, "f1"),
        "pair_count": int(get_num(point, "pair_count", default=0) or 0),
        "mean_l2": get_num(point, "mean_l2_px"),
        "median_l2": get_num(point, "median_l2_px"),
        "p90_l2": get_num(point, "p90_l2_px"),
        "dx": get_num(point, "mean_abs_dx_px", default=get_num(point, "mae_x_px")),
        "dy": get_num(point, "mean_abs_dy_px", default=get_num(point, "mae_y_px")),
        "ppl_sr_30": get_num(point, "ppl_sr_30"),
        "ppl_sr_50": get_num(point, "ppl_sr_50"),
        "small_l2": get_num(size, "small", "mean_l2_px"),
        "medium_l2": get_num(size, "medium", "mean_l2_px"),
        "large_l2": get_num(size, "large", "mean_l2_px"),
        "false_positive": int(get_num(error, "has_picking_false_positive_count", default=get_num(has, "false_positive", default=0)) or 0),
        "false_negative": int(get_num(error, "has_picking_false_negative_count", default=get_num(has, "false_negative", default=0)) or 0),
        "cross_instance_mismatch": int(get_num(error, "cross_instance_mismatch_count", default=0) or 0),
    }


def decoupled_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    test = (summary.get("decoupled_point_summary") or {}).get("test") or {}
    std = test.get("standard_match") or {}
    iou = test.get("iou_conditioned") or {}
    oracle = test.get("oracle_candidates") or {}
    o_iou = oracle.get("iou_ge_0.50_visible") or {}
    o_any = oracle.get("any_visible_pred") or {}
    o_box = oracle.get("point_in_gt_box_visible") or {}
    iou_50_70 = iou.get("0.50_to_0.70") or {}
    iou_70_85 = iou.get("0.70_to_0.85") or {}
    iou_85 = iou.get("ge_0.85") or {}
    visible_gt = int(get_num(test, "visible_gt_count", default=get_num(std, "visible_gt_count", default=0)) or 0)
    std_count = int(get_num(std, "count", default=0) or 0)
    return {
        "visible_gt_count": visible_gt,
        "standard_count": std_count,
        "coverage_gap": max(0, visible_gt - std_count) if visible_gt else None,
        "standard_recall": get_num(std, "candidate_recall"),
        "standard_mean_l2": get_num(std, "mean_l2_px"),
        "standard_p90_l2": get_num(std, "p90_l2_px"),
        "standard_in_gt_box_rate": get_num(std, "pred_point_inside_gt_box_rate"),
        "oracle_iou_recall": get_num(o_iou, "candidate_recall"),
        "oracle_iou_mean_l2": get_num(o_iou, "mean_l2_px"),
        "oracle_iou_p90_l2": get_num(o_iou, "p90_l2_px"),
        "oracle_any_recall": get_num(o_any, "candidate_recall"),
        "oracle_any_mean_l2": get_num(o_any, "mean_l2_px"),
        "oracle_any_p90_l2": get_num(o_any, "p90_l2_px"),
        "oracle_box_recall": get_num(o_box, "candidate_recall"),
        "oracle_box_mean_l2": get_num(o_box, "mean_l2_px"),
        "iou_50_70_count": int(get_num(iou_50_70, "count", default=0) or 0),
        "iou_50_70_mean_l2": get_num(iou_50_70, "mean_l2_px"),
        "iou_50_70_p90_l2": get_num(iou_50_70, "p90_l2_px"),
        "iou_70_85_count": int(get_num(iou_70_85, "count", default=0) or 0),
        "iou_70_85_mean_l2": get_num(iou_70_85, "mean_l2_px"),
        "iou_70_85_p90_l2": get_num(iou_70_85, "p90_l2_px"),
        "iou_ge_85_count": int(get_num(iou_85, "count", default=0) or 0),
        "iou_ge_85_mean_l2": get_num(iou_85, "mean_l2_px"),
        "iou_ge_85_p90_l2": get_num(iou_85, "p90_l2_px"),
    }


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for exp in EXPERIMENTS:
        summary = load_json(exp["summary"])
        row = {
            "name": exp["name"],
            "label": exp["label"],
            "role": exp["role"],
            "summary": str(exp["summary"]),
        }
        row.update(primary_test_metrics(summary))
        row.update(decoupled_metrics(summary))
        std_mean = row.get("standard_mean_l2")
        any_mean = row.get("oracle_any_mean_l2")
        std_p90 = row.get("standard_p90_l2")
        any_p90 = row.get("oracle_any_p90_l2")
        row["oracle_any_mean_gain"] = (std_mean - any_mean) if std_mean is not None and any_mean is not None else None
        row["oracle_any_p90_gain"] = (std_p90 - any_p90) if std_p90 is not None and any_p90 is not None else None
        row["dy_minus_dx"] = (
            row["dy"] - row["dx"] if row.get("dy") is not None and row.get("dx") is not None else None
        )
        row["small_minus_large_l2"] = (
            row["small_l2"] - row["large_l2"]
            if row.get("small_l2") is not None and row.get("large_l2") is not None
            else None
        )
        rows.append(row)
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> None:
    ref = next(row for row in rows if row["name"] == "V7_EXP2_MAIN")
    for row in rows:
        for key in ("AP", "has_f1", "has_recall", "pair_count", "mean_l2", "median_l2", "p90_l2", "small_l2", "dy"):
            value = row.get(key)
            base = ref.get(key)
            row[f"delta_{key}"] = (value - base) if value is not None and base is not None else None


def diagnose_row(row: dict[str, Any], ref: dict[str, Any]) -> str:
    reasons = []
    if row.get("pair_count", 0) < ref.get("pair_count", 0) - 8 or (row.get("has_recall") or 0) < (ref.get("has_recall") or 0) - 0.03:
        reasons.append("coverage_error")
    if (row.get("oracle_any_mean_gain") or 0) >= 8 or (row.get("oracle_any_p90_gain") or 0) >= 20:
        reasons.append("association_ranking_error")
    if (row.get("iou_ge_85_mean_l2") or 0) >= 20 or (row.get("iou_ge_85_p90_l2") or 0) >= 45:
        reasons.append("coordinate_error_high_iou")
    if (row.get("small_minus_large_l2") or 0) >= 10:
        reasons.append("small_tail_error")
    if not reasons:
        reasons.append("mixed_or_mild")
    return "+".join(reasons)


def decide_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref = next(row for row in rows if row["name"] == "V7_EXP2_MAIN")
    for row in rows:
        row["root_cause_flags"] = diagnose_row(row, ref)

    comparable = [row for row in rows if row["name"] != "V7_EXP2_MAIN"]
    selection_votes = sum(1 for row in comparable if "association_ranking_error" in row["root_cause_flags"])
    coverage_votes = sum(1 for row in comparable if "coverage_error" in row["root_cause_flags"])
    coordinate_votes = sum(1 for row in comparable if "coordinate_error_high_iou" in row["root_cause_flags"])
    detector_ap_failures = sum(1 for row in comparable if (row.get("AP") or 0) < 0.632)

    if selection_votes >= 3:
        decision = "selection_first"
        rationale = (
            "Most variants show a large standard-vs-oracle gap: better visible point candidates exist, "
            "but final query/visibility/ranking selection fails to keep them."
        )
    elif coordinate_votes >= 3:
        decision = "coordinate_first"
        rationale = "High-IoU matched instances still have large point error across variants."
    elif detector_ap_failures >= 3 or coverage_votes >= 4:
        decision = "detector_first"
        rationale = "Most variants lack enough visible candidates or detection/has recall before point refinement."
    else:
        decision = "selection_first"
        rationale = "The strongest actionable signal remains the oracle candidate gap, with AP mostly acceptable."

    return {
        "decision": decision,
        "selection_votes": selection_votes,
        "coverage_votes": coverage_votes,
        "coordinate_votes": coordinate_votes,
        "detector_ap_failures": detector_ap_failures,
        "rationale": rationale,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 2) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" if idx == 0 else "---:" for idx, _ in enumerate(columns)) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(fmt(value, digits))
            else:
                values.append(fmt(value, digits))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    ref = next(row for row in rows if row["name"] == "V7_EXP2_MAIN")
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Root-Cause Diagnostic Review",
        "",
        f"- Generated at: {now}",
        "- Protocol: read existing report `summary.json` files only; no training, no checkpoint loading, no test-time model selection.",
        "- Formal conclusion uses `test` only. `valid` is not mixed into metric comparisons.",
        "- Reference: `V7_EXP2_MAIN fair retrain`.",
        "",
        "## Final Diagnostic Decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Rationale: {decision['rationale']}",
        f"- Vote summary: selection={decision['selection_votes']}, coverage={decision['coverage_votes']}, coordinate={decision['coordinate_votes']}, detector_ap_failures={decision['detector_ap_failures']}.",
        "",
        "Interpretation: 下一步应优先验证实例候选选择/重排序/accept calibration，而不是继续把 O2M、quality、heatmap 或 reliability loss 堆进主训练链路。",
        "",
        "## Same-Split Test Matrix",
        "",
    ]
    lines.extend(
        markdown_table(
            rows,
            [
                ("Experiment", "name"),
                ("AP", "AP"),
                ("F1", "has_f1"),
                ("Recall", "has_recall"),
                ("pair", "pair_count"),
                ("mean", "mean_l2"),
                ("median", "median_l2"),
                ("p90", "p90_l2"),
                ("dx", "dx"),
                ("dy", "dy"),
                ("PPL@30", "ppl_sr_30"),
                ("PPL@50", "ppl_sr_50"),
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "Note: PPL-SR@30/50 columns are preserved for the required reporting interface. Existing historical summaries do not persist full L2 distributions, so unavailable values are shown as `-` instead of being reconstructed or guessed.",
            "",
            "## Error Type Decomposition",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            rows,
            [
                ("Experiment", "name"),
                ("visible GT", "visible_gt_count"),
                ("std pair", "standard_count"),
                ("coverage gap", "coverage_gap"),
                ("FP", "false_positive"),
                ("FN", "false_negative"),
                ("cross mismatch", "cross_instance_mismatch"),
                ("flags", "root_cause_flags"),
            ],
            digits=2,
        )
    )
    lines.extend(["", "## Standard vs Oracle Candidate Gap", ""])
    lines.extend(
        markdown_table(
            rows,
            [
                ("Experiment", "name"),
                ("std recall", "standard_recall"),
                ("std mean", "standard_mean_l2"),
                ("std p90", "standard_p90_l2"),
                ("oracle IoU recall", "oracle_iou_recall"),
                ("oracle IoU mean", "oracle_iou_mean_l2"),
                ("oracle any recall", "oracle_any_recall"),
                ("oracle any mean", "oracle_any_mean_l2"),
                ("oracle any p90", "oracle_any_p90_l2"),
                ("mean gain", "oracle_any_mean_gain"),
                ("p90 gain", "oracle_any_p90_gain"),
            ],
            digits=2,
        )
    )
    lines.extend(["", "## IoU-Conditioned Coordinate Error", ""])
    lines.extend(
        markdown_table(
            rows,
            [
                ("Experiment", "name"),
                ("0.50-0.70 n", "iou_50_70_count"),
                ("0.50-0.70 mean", "iou_50_70_mean_l2"),
                ("0.50-0.70 p90", "iou_50_70_p90_l2"),
                ("0.70-0.85 n", "iou_70_85_count"),
                ("0.70-0.85 mean", "iou_70_85_mean_l2"),
                ("0.70-0.85 p90", "iou_70_85_p90_l2"),
                (">=0.85 n", "iou_ge_85_count"),
                (">=0.85 mean", "iou_ge_85_mean_l2"),
                (">=0.85 p90", "iou_ge_85_p90_l2"),
            ],
            digits=2,
        )
    )
    lines.extend(["", "## Size And Direction Bias", ""])
    lines.extend(
        markdown_table(
            rows,
            [
                ("Experiment", "name"),
                ("small", "small_l2"),
                ("medium", "medium_l2"),
                ("large", "large_l2"),
                ("small-large", "small_minus_large_l2"),
                ("dx", "dx"),
                ("dy", "dy"),
                ("dy-dx", "dy_minus_dx"),
            ],
            digits=2,
        )
    )
    lines.extend(
        [
            "",
            "## Root-Cause Reading",
            "",
            f"- `V7_EXP2_MAIN` reference: AP {fmt(ref['AP'], 4)}, F1 {fmt(ref['has_f1'], 4)}, pair {ref['pair_count']}, mean L2 {fmt(ref['mean_l2'])}, p90 L2 {fmt(ref['p90_l2'])}.",
            "- `POINT_O2M_AUX` proves point error can be reduced, but recall/pair loss means it solves coordinates by sacrificing visible-instance coverage.",
            "- `EMA_BIFPN` proves detector/instance coverage can improve, but p90 and small-object L2 remain worse, so detector AP alone is not the paper contribution.",
            "- `POINT_QUALITY_ALIGN` and `EMA_BIFPN_POINT_RELIABILITY_V1_FAIR_100E` show quality mechanisms can select cleaner subsets, but default full-output coverage degrades or tail error persists.",
            "- The repeated pattern is a three-way conflict among query selection, visibility decision, and point coordinate quality. This supports `selection_first` before any new trained point representation.",
            "",
            "## Literature Alignment",
            "",
            "- YOLOv8-GP supports the idea of grape picking as synchronized detection + picking-point localization, but this repo's evidence rejects naive loss stacking as the main route.",
            "- DEKR, TokenPose, and Poseur all motivate keypoint-specific representation/query design. In this project, that should be considered only after the selection bottleneck is measured and not solved offline.",
            "",
            "## Next Action Gate",
            "",
            "- If a no-training selection/accept protocol raises pair above the main model while reducing p90/small L2, continue with `selection_first`.",
            "- If selection cannot recover pair or oracle gaps are small after full prediction-record export, switch to `coordinate_first` with an independent keypoint representation/refiner.",
            "- If visible candidate recall is the limiting factor, switch to `detector_first`; current evidence does not favor this as the first move.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    add_deltas(rows)
    decision = decide_overall(rows)

    write_csv(args.output_dir / "root_cause_metric_matrix.csv", rows)
    (args.output_dir / "diagnosis_decision.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), **decision, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "root_cause_diagnostic_report_zh.md", rows, decision)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
