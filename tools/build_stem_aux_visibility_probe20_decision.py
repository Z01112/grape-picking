from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_stem_aux_visibility_v1_probe20"
REPORT_DIR = RUN_DIR / "report"
EMA_REPORT_DIR = REPO_ROOT / "outputs" / "01_mainline_results" / "ema_bifpn_new1804_fair100" / "report"
BASELINE_REPORT_DIR = REPO_ROOT / "outputs" / "02_baselines" / "rtdetrv4_new1804_baseline_fair100" / "report"
YOLO_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "02_baselines"
    / "yolo11n_pose_new1804_b8_e100_20260607_181020"
    / "summary.json"
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def count_l2(records_path: Path) -> tuple[int | None, int | None]:
    if not records_path.exists():
        return None, None
    records = load_json(records_path)
    if isinstance(records, dict) and "records" in records:
        records = records["records"]
    if not isinstance(records, list):
        return None, None
    gt30 = 0
    gt50 = 0
    for record in records:
        for pred in record.get("pred_instances", []):
            value = safe_float(pred.get("l2_px"))
            if not math.isfinite(value):
                continue
            if value > 30:
                gt30 += 1
            if value > 50:
                gt50 += 1
    if gt30 == 0 and gt50 == 0:
        return None, None
    return gt30, gt50


def extract_metrics(label: str, summary_path: Path, is_candidate: bool = False) -> dict[str, Any]:
    summary = load_json(summary_path)
    if summary is None:
        return {"model": label, "status": "missing", "summary_path": str(summary_path)}
    test = nested(summary, ["primary_checkpoint_split_summary", "test"], {})
    det = test.get("grape_detection", {}) if isinstance(test, dict) else {}
    has = test.get("has_picking", {}) if isinstance(test, dict) else {}
    point = test.get("picking_point", {}) if isinstance(test, dict) else {}
    unified = nested(summary, ["unified_point_metrics", "test"], {})
    instance = unified.get("instance_chain", {}) if isinstance(unified, dict) else {}
    global_chain = unified.get("global_chain", {}) if isinstance(unified, dict) else {}
    gt30, gt50 = count_l2(summary_path.parent / "test_prediction_records.json")
    return {
        "model": label,
        "status": "ok",
        "summary_path": str(summary_path),
        "checkpoint": summary.get("primary_checkpoint_name", ""),
        "AP": safe_float(det.get("AP")),
        "AP50": safe_float(det.get("AP50")),
        "F1": safe_float(has.get("f1", instance.get("instance_visible_f1"))),
        "global_visible_recall": safe_float(global_chain.get("global_visible_recall")),
        "global_f1": safe_float(global_chain.get("global_visible_f1")),
        "pair": safe_float(point.get("pair_count", instance.get("point_pair_count"))),
        "mean_L2": safe_float(point.get("mean_l2_px", instance.get("point_mean_l2_px"))),
        "median_L2": safe_float(point.get("median_l2_px", instance.get("point_median_l2_px"))),
        "p90_L2": safe_float(point.get("p90_l2_px", instance.get("point_p90_l2_px"))),
        "PPL-SR@30": safe_float(point.get("ppl_sr_30", instance.get("ppl_sr_30"))),
        "PPL-SR@50": safe_float(point.get("ppl_sr_50", instance.get("ppl_sr_50"))),
        "L2>30_count": gt30,
        "L2>50_count": gt50,
        "mean_abs_dx": safe_float(point.get("mean_abs_dx_px", point.get("mae_x_px", instance.get("point_mae_x_px")))),
        "mean_abs_dy": safe_float(point.get("mean_abs_dy_px", point.get("mae_y_px", instance.get("point_mae_y_px")))),
        "is_candidate": is_candidate,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "status",
        "checkpoint",
        "AP",
        "AP50",
        "F1",
        "global_visible_recall",
        "global_f1",
        "pair",
        "mean_L2",
        "median_L2",
        "p90_L2",
        "PPL-SR@30",
        "PPL-SR@50",
        "L2>30_count",
        "L2>50_count",
        "mean_abs_dx",
        "mean_abs_dy",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def decide(candidate: dict[str, Any], ema: dict[str, Any]) -> tuple[str, list[str], bool]:
    if candidate.get("status") != "ok":
        return "blocked_no_candidate_summary", ["Candidate summary missing."], False
    if ema.get("status") != "ok":
        return "reference_missing_review_manually", ["EMA_BIFPN_NEW1804 reference missing."], False
    notes = []
    ap = safe_float(candidate.get("AP"))
    f1 = safe_float(candidate.get("F1"))
    pair = safe_float(candidate.get("pair"))
    mean_l2 = safe_float(candidate.get("mean_L2"))
    ppl30 = safe_float(candidate.get("PPL-SR@30"))
    ema_f1 = safe_float(ema.get("F1"))
    ema_pair = safe_float(ema.get("pair"))
    ema_l2 = safe_float(ema.get("mean_L2"))
    ema_ppl30 = safe_float(ema.get("PPL-SR@30"))
    gates = {
        "AP>=0.620": ap >= 0.620,
        "F1>=EMA_BIFPN_NEW1804": f1 >= ema_f1,
        "pair>=EMA_BIFPN_NEW1804": pair >= ema_pair,
        "mean_L2<=EMA+0.3": mean_l2 <= ema_l2 + 0.3,
        "PPL@30>=EMA-0.005": ppl30 >= ema_ppl30 - 0.005,
    }
    for key, value in gates.items():
        notes.append(f"{key}: {value}")
    all_pass = all(gates.values())
    if all_pass and (f1 > ema_f1 or pair > ema_pair) and mean_l2 <= ema_l2 + 0.3:
        return "ready_for_fair100_from_unified_pretrained", notes, True
    if all_pass:
        return "probe20_passed_but_needs_coverage_gain_review", notes, True
    return "probe20_failed_cleanup_checkpoints", notes, False


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        extract_metrics("STEM_AUX_VISIBILITY_V1_PROBE20", REPORT_DIR / "summary.json", is_candidate=True),
        extract_metrics("EMA_BIFPN_NEW1804_FAIR100", EMA_REPORT_DIR / "summary.json"),
        extract_metrics("RTDETRV4_BASELINE_NEW1804_FAIR100", BASELINE_REPORT_DIR / "summary.json"),
        extract_metrics("YOLO11n-pose NEW1804", YOLO_SUMMARY),
    ]
    write_csv(REPORT_DIR / "stem_aux_visibility_probe20_summary.csv", rows)
    decision, notes, keep_checkpoint = decide(rows[0], rows[1])
    payload = {
        "decision": decision,
        "keep_checkpoint": keep_checkpoint,
        "candidate": rows[0],
        "references": rows[1:],
        "decision_notes": notes,
    }
    (REPORT_DIR / "stem_aux_visibility_probe20_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "model",
        "AP",
        "AP50",
        "F1",
        "global_visible_recall",
        "pair",
        "mean_L2",
        "median_L2",
        "p90_L2",
        "PPL-SR@30",
        "PPL-SR@50",
        "mean_abs_dx",
        "mean_abs_dy",
    ]
    lines = [
        "# STEM_AUX_VISIBILITY_V1_PROBE20 Decision",
        "",
        f"- Decision: `{decision}`",
        f"- Keep checkpoint: `{keep_checkpoint}`",
        "- Probe route: EMA_BIFPN_NEW1804 architecture + stem visibility auxiliary, warm-start from EMA_BIFPN_NEW1804 `best_composite.pth`, 20 epochs.",
        "- This is a mechanism probe, not a fair100 paper result.",
        "",
        "## Gate Notes",
        *[f"- {note}" for note in notes],
        "",
        "## Same-Dataset Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row['model']} | missing |  |  |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model", "")),
                    fmt(row.get("AP")),
                    fmt(row.get("AP50")),
                    fmt(row.get("F1")),
                    fmt(row.get("global_visible_recall")),
                    fmt(row.get("pair"), 1),
                    fmt(row.get("mean_L2"), 2),
                    fmt(row.get("median_L2"), 2),
                    fmt(row.get("p90_L2"), 2),
                    fmt(row.get("PPL-SR@30")),
                    fmt(row.get("PPL-SR@50")),
                    fmt(row.get("mean_abs_dx"), 2),
                    fmt(row.get("mean_abs_dy"), 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Next Action",
            "- If decision is `ready_for_fair100_from_unified_pretrained`, the next fair100 must train from HGNetv2 unified pretrained, not from this probe checkpoint.",
            "- If decision is `probe20_failed_cleanup_checkpoints`, keep report/records/logs and remove generated `.pth` files from this run.",
        ]
    )
    (REPORT_DIR / "stem_aux_visibility_probe20_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_DIR / "stem_aux_visibility_probe20_decision.md")


if __name__ == "__main__":
    main()
