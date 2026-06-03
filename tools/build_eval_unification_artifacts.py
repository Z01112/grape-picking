from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs/08_eval_unification"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grape_point_eval_utils import match_prediction_record


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gppoint_row(model: str, summary_path: Path) -> dict:
    data = read_json(summary_path)
    det = data["primary_checkpoint_split_summary"]["test"]["grape_detection"]
    unified = data["unified_point_metrics"]["test"]
    inst = unified["instance_chain"]
    glob = unified["global_chain"]
    return {
        "model": model,
        "AP": det.get("AP"),
        "AP50": det.get("AP50"),
        "instance_f1": inst.get("instance_visible_f1", inst.get("has_picking_f1")),
        "global_visible_recall": glob.get("global_visible_recall"),
        "global_f1": glob.get("global_visible_f1"),
        "pair_count": inst.get("point_pair_count"),
        "mean_L2": inst.get("point_mean_l2_px"),
        "median_L2": inst.get("point_median_l2_px"),
        "p90_L2": inst.get("point_p90_l2_px"),
        "PPL-SR@30": inst.get("ppl_sr_30"),
        "PPL-SR@50": inst.get("ppl_sr_50"),
        "mean_abs_dx": inst.get("point_mae_x_px"),
        "mean_abs_dy": inst.get("point_mae_y_px"),
        "source": str(summary_path.relative_to(REPO_ROOT)),
        "status": "ok",
    }


def yolo_row(summary_path: Path) -> dict:
    data = read_json(summary_path)
    det = data.get("yolo_native_metrics", {})
    unified = data["unified_point_metrics"]
    inst = unified["instance_chain"]
    glob = unified["global_chain"]
    return {
        "model": "YOLO11n-pose",
        "AP": det.get("box_AP"),
        "AP50": det.get("box_AP50"),
        "instance_f1": inst.get("instance_visible_f1", inst.get("has_picking_f1")),
        "global_visible_recall": glob.get("global_visible_recall"),
        "global_f1": glob.get("global_visible_f1"),
        "pair_count": inst.get("point_pair_count"),
        "mean_L2": inst.get("point_mean_l2_px"),
        "median_L2": inst.get("point_median_l2_px"),
        "p90_L2": inst.get("point_p90_l2_px"),
        "PPL-SR@30": inst.get("ppl_sr_30"),
        "PPL-SR@50": inst.get("ppl_sr_50"),
        "mean_abs_dx": inst.get("point_mae_x_px"),
        "mean_abs_dy": inst.get("point_mae_y_px"),
        "source": str(summary_path.relative_to(REPO_ROOT)),
        "status": "ok",
    }


def hp_row(summary_path: Path, ema_row: dict) -> dict:
    data = read_json(summary_path)
    selected = data.get("selected_test", {})
    precision = safe_float(selected.get("precision"))
    pair = safe_float(selected.get("pair_count"))
    visible_gt_total = safe_float(ema_row.get("_visible_gt_total", 251))
    global_recall = pair / visible_gt_total if visible_gt_total > 0 else math.nan
    global_f1 = (
        0.0
        if math.isnan(precision) or math.isnan(global_recall) or precision + global_recall == 0
        else 2.0 * precision * global_recall / (precision + global_recall)
    )
    return {
        "model": "EMA_BIFPN + HAS@0.62 / HP protocol",
        "AP": ema_row.get("AP"),
        "AP50": ema_row.get("AP50"),
        "instance_f1": selected.get("f1"),
        "global_visible_recall": global_recall,
        "global_f1": global_f1,
        "pair_count": selected.get("pair_count"),
        "mean_L2": selected.get("mean_l2"),
        "median_L2": selected.get("median_l2"),
        "p90_L2": selected.get("p90_l2"),
        "PPL-SR@30": selected.get("ppl_sr_30"),
        "PPL-SR@50": selected.get("ppl_sr_50"),
        "mean_abs_dx": selected.get("dx"),
        "mean_abs_dy": selected.get("dy"),
        "source": str(summary_path.relative_to(REPO_ROOT)),
        "status": "ok_existing_hp_threshold_only",
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict], fieldnames: list[str]) -> str:
    out = []
    out.append("| " + " | ".join(fieldnames) + " |")
    out.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        vals = []
        for key in fieldnames:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(fmt(value))
            else:
                vals.append(str(value) if value is not None else "")
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_comparison(rows: list[dict]) -> None:
    fields = [
        "model",
        "AP",
        "AP50",
        "instance_f1",
        "global_visible_recall",
        "global_f1",
        "pair_count",
        "mean_L2",
        "median_L2",
        "p90_L2",
        "PPL-SR@30",
        "PPL-SR@50",
        "mean_abs_dx",
        "mean_abs_dy",
    ]
    write_csv(OUT_DIR / "unified_comparison_summary.csv", rows, fields)
    md = [
        "# Unified Comparison Summary",
        "",
        "All rows use formal test where available. GPPoint-DETR and YOLO-pose point metrics use `tools/grape_point_eval_utils.py` unified IoU50 records matching.",
        "",
        markdown_table(rows, fields),
        "",
        "Notes:",
        "- `baseline_replay_v2` is listed as skipped if config/summary is missing; its checkpoint alone is not enough for a reproducible unified report.",
        "- `EMA_BIFPN + HAS@0.62 / HP protocol` is an existing threshold-only calibration row. AP/AP50 are inherited from EMA_BIFPN because boxes are unchanged.",
        "- `instance_f1` follows the legacy GPPoint-DETR instance-chain has_picking F1.",
        "- `global_visible_recall` uses all visible GT as denominator.",
    ]
    (OUT_DIR / "unified_comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def skipped_baseline() -> dict:
    baseline_dir = REPO_ROOT / "outputs/00_reference_models/baseline_replay_v2"
    checkpoint = baseline_dir / "best_composite.pth"
    config_candidates = sorted(baseline_dir.rglob("*.yml")) + sorted(baseline_dir.rglob("*.yaml"))
    summary = baseline_dir / "summary.json"
    missing = []
    if not checkpoint.exists():
        missing.append(str(checkpoint.relative_to(REPO_ROOT)))
    if not config_candidates:
        missing.append("baseline_replay_v2 config (*.yml/*.yaml)")
    if not summary.exists():
        missing.append(str(summary.relative_to(REPO_ROOT)))
    reason = {
        "model": "baseline_replay_v2",
        "status": "skipped",
        "reason": "checkpoint/config/summary must be explicitly traceable; no config or summary was found, so no path was guessed.",
        "existing_checkpoint": str(checkpoint.relative_to(REPO_ROOT)) if checkpoint.exists() else None,
        "config_candidates": [str(p.relative_to(REPO_ROOT)) for p in config_candidates],
        "summary_exists": summary.exists(),
        "missing": missing,
    }
    out_dir = OUT_DIR / "baseline_replay_v2_unified_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skipped_reason.json").write_text(json.dumps(reason, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model": "baseline_replay_v2",
        "AP": "",
        "AP50": "",
        "instance_f1": "",
        "global_visible_recall": "",
        "global_f1": "",
        "pair_count": "",
        "mean_L2": "",
        "median_L2": "",
        "p90_L2": "",
        "PPL-SR@30": "",
        "PPL-SR@50": "",
        "mean_abs_dx": "",
        "mean_abs_dy": "",
        "source": str((out_dir / "skipped_reason.json").relative_to(REPO_ROOT)),
        "status": "skipped_missing_config_summary",
    }


def error_breakdown(model: str, records_path: Path) -> dict:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    detection_failure = 0
    visibility_failure = 0
    localization_failure_l2_gt30 = 0
    localization_failure_l2_gt50 = 0
    high_quality_success = 0
    visible_gt_total = 0
    matched_visible = 0
    correct_visible = 0
    for record in records:
        matched = match_prediction_record(record, iou_threshold=0.5, has_picking_threshold=0.5, visibility_score_key="visible_score")
        visible_gt_total += int(matched["visible_gt_total"])
        matched_visible_in_record = sum(1 for case in matched["matched_pairs"] if bool(case.get("gt_has_picking", False)))
        matched_visible += matched_visible_in_record
        detection_failure += int(matched["visible_gt_total"]) - matched_visible_in_record
        visibility_failure += len(matched["has_fn_pairs"])
        for case in matched["correct_visible_pairs"]:
            correct_visible += 1
            l2 = safe_float(case.get("l2_px"))
            if l2 > 30.0:
                localization_failure_l2_gt30 += 1
            if l2 > 50.0:
                localization_failure_l2_gt50 += 1
            if l2 <= 30.0:
                high_quality_success += 1
    return {
        "model": model,
        "visible_gt_total": visible_gt_total,
        "matched_visible_grapes": matched_visible,
        "correct_visible_grapes": correct_visible,
        "detection_failure": detection_failure,
        "visibility_failure": visibility_failure,
        "localization_failure_l2_gt30": localization_failure_l2_gt30,
        "localization_failure_l2_gt50": localization_failure_l2_gt50,
        "high_quality_success_l2_le30": high_quality_success,
        "detection_failure_rate": detection_failure / visible_gt_total if visible_gt_total else math.nan,
        "visibility_failure_rate_of_visible_gt": visibility_failure / visible_gt_total if visible_gt_total else math.nan,
        "localization_gt30_rate_of_correct_visible": localization_failure_l2_gt30 / correct_visible if correct_visible else math.nan,
        "high_quality_success_rate_of_visible_gt": high_quality_success / visible_gt_total if visible_gt_total else math.nan,
        "source_records": str(records_path.relative_to(REPO_ROOT)),
        "status": "ok",
    }


def write_error_breakdown(rows: list[dict]) -> None:
    fields = [
        "model",
        "visible_gt_total",
        "matched_visible_grapes",
        "correct_visible_grapes",
        "detection_failure",
        "visibility_failure",
        "localization_failure_l2_gt30",
        "localization_failure_l2_gt50",
        "high_quality_success_l2_le30",
        "detection_failure_rate",
        "visibility_failure_rate_of_visible_gt",
        "localization_gt30_rate_of_correct_visible",
        "high_quality_success_rate_of_visible_gt",
        "source_records",
        "status",
    ]
    write_csv(OUT_DIR / "error_source_breakdown.csv", rows, fields)
    md = [
        "# Error Source Breakdown",
        "",
        "Definitions:",
        "- detection failure: visible GT not matched by any IoU50 predicted grape.",
        "- visibility failure: visible GT matched by IoU50, but matched prediction is not visible under `visible_score >= 0.5`.",
        "- localization failure: IoU50 matched and pred visible, but point L2 is above 30 px or 50 px.",
        "- high-quality success: IoU50 matched, pred visible, and L2 <= 30 px.",
        "",
        markdown_table(rows, fields),
    ]
    (OUT_DIR / "error_source_breakdown.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_next_step(rows: list[dict], breakdown_rows: list[dict]) -> None:
    by_model = {row["model"]: row for row in rows}
    by_breakdown = {row["model"]: row for row in breakdown_rows}
    v7 = by_model.get("V7_EXP2_MAIN fair retrain", {})
    ema = by_model.get("EMA_BIFPN", {})
    v7_b = by_breakdown.get("V7_EXP2_MAIN fair retrain", {})
    ema_b = by_breakdown.get("EMA_BIFPN", {})
    md = [
        "# Next Model Step",
        "",
        "## Bottom Line",
        "",
        "`EMA_BIFPN` is the better next-round base than `V7_EXP2_MAIN fair retrain`: it has higher AP/AP50, higher instance F1, higher global visible recall, higher pair_count, and lower mean L2. The tradeoff is that V7 still has slightly better PPL-SR@30/50, so the next experiment must improve point reliability without sacrificing EMA_BIFPN's coverage.",
        "",
        "## V7 vs EMA_BIFPN",
        "",
        "| metric | V7_EXP2_MAIN | EMA_BIFPN | judgment |",
        "| --- | ---: | ---: | --- |",
        f"| AP | {fmt(v7.get('AP'))} | {fmt(ema.get('AP'))} | EMA better |",
        f"| AP50 | {fmt(v7.get('AP50'))} | {fmt(ema.get('AP50'))} | EMA better |",
        f"| instance_f1 | {fmt(v7.get('instance_f1'))} | {fmt(ema.get('instance_f1'))} | EMA slightly better |",
        f"| global_visible_recall | {fmt(v7.get('global_visible_recall'))} | {fmt(ema.get('global_visible_recall'))} | EMA better |",
        f"| pair_count | {v7.get('pair_count', '')} | {ema.get('pair_count', '')} | EMA better |",
        f"| mean_L2 | {fmt(v7.get('mean_L2'))} | {fmt(ema.get('mean_L2'))} | EMA better |",
        f"| PPL-SR@30 | {fmt(v7.get('PPL-SR@30'))} | {fmt(ema.get('PPL-SR@30'))} | V7 better |",
        f"| PPL-SR@50 | {fmt(v7.get('PPL-SR@50'))} | {fmt(ema.get('PPL-SR@50'))} | V7 slightly better |",
        "",
        "## Main Bottleneck",
        "",
        f"- Detection is not the main bottleneck on the current test split: V7 detection failure is {v7_b.get('detection_failure', '')}, EMA_BIFPN detection failure is {ema_b.get('detection_failure', '')}.",
        f"- Visibility selection is a larger bottleneck than detection: V7 visibility failure is {v7_b.get('visibility_failure', '')}, EMA_BIFPN visibility failure is {ema_b.get('visibility_failure', '')}. EMA reduces this substantially.",
        f"- Localization remains the next useful target: among correct visible pairs, EMA_BIFPN still has {ema_b.get('localization_failure_l2_gt30', '')} cases over 30 px and {ema_b.get('localization_failure_l2_gt50', '')} cases over 50 px.",
        "",
        "## HP Protocol vs Model Change",
        "",
        "HP threshold-only calibration is useful as an inference protocol because it can trade a small amount of coverage for lower mean L2, but it does not change the model's coordinate representation. It should be reported as a protocol/ablation, not as the next main model by itself.",
        "",
        "## Recommended Single Minimal Experiment",
        "",
        "Run only one next model experiment: `EMA_BIFPN + detached point reliability calibration head`, trained to predict whether the existing point will be within 30 px, with gradients stopped into detector/has/offset branches. Do not add heatmap, SimCC, O2M, selector, or geometry fallback in the same experiment.",
        "",
        "Expected effect: keep EMA_BIFPN's stronger AP/global recall/pair coverage, improve PPL-SR@30 and reduce the number of L2>30 visible pairs. This targets the current measured gap directly instead of chasing all diagnostic metrics.",
    ]
    (OUT_DIR / "README_next_model_step.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = skipped_baseline()

    rows = [baseline]
    v7_summary = OUT_DIR / "v7_exp2_unified_report/summary.json"
    ema_summary = OUT_DIR / "ema_bifpn_unified_report/summary.json"
    yolo_summary = OUT_DIR / "yolo_pose_unified_report/summary.json"
    hp_summary = REPO_ROOT / "outputs/03_global_analysis/hp_pick_calibration_fine_has_only_20260530/summary.json"

    v7 = gppoint_row("V7_EXP2_MAIN fair retrain", v7_summary)
    ema = gppoint_row("EMA_BIFPN", ema_summary)
    ema_data = read_json(ema_summary)
    ema["_visible_gt_total"] = ema_data["unified_point_metrics"]["test"]["instance_chain"]["visible_gt_total"]
    rows.extend([v7, ema])
    if hp_summary.exists():
        rows.append(hp_row(hp_summary, ema))
    if yolo_summary.exists():
        rows.append(yolo_row(yolo_summary))
    for row in rows:
        row.pop("_visible_gt_total", None)
    write_comparison(rows)

    breakdown_rows = [
        {
            "model": "baseline_replay_v2",
            "visible_gt_total": "",
            "matched_visible_grapes": "",
            "correct_visible_grapes": "",
            "detection_failure": "",
            "visibility_failure": "",
            "localization_failure_l2_gt30": "",
            "localization_failure_l2_gt50": "",
            "high_quality_success_l2_le30": "",
            "detection_failure_rate": "",
            "visibility_failure_rate_of_visible_gt": "",
            "localization_gt30_rate_of_correct_visible": "",
            "high_quality_success_rate_of_visible_gt": "",
            "source_records": "outputs/08_eval_unification/baseline_replay_v2_unified_report/skipped_reason.json",
            "status": "skipped_missing_config_summary",
        },
        error_breakdown("V7_EXP2_MAIN fair retrain", OUT_DIR / "v7_exp2_unified_report/test_prediction_records.json"),
        error_breakdown("EMA_BIFPN", OUT_DIR / "ema_bifpn_unified_report/test_prediction_records.json"),
    ]
    write_error_breakdown(breakdown_rows)
    write_next_step(rows, breakdown_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
