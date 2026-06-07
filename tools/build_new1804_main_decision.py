from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "01_mainline_results" / "ema_bifpn_new1804_fair100"
REPORT_DIR = RUN_DIR / "report"


REFERENCE_SUMMARIES = [
    (
        "V7_EXP2_MAIN fair retrain",
        REPO_ROOT / "outputs" / "03_unified_evaluation" / "eval_unification" / "v7_exp2_unified_report" / "summary.json",
    ),
    (
        "EMA_BIFPN old dataset",
        REPO_ROOT / "outputs" / "03_unified_evaluation" / "eval_unification" / "ema_bifpn_unified_report" / "summary.json",
    ),
    (
        "CADA adapter-only probe20",
        REPO_ROOT
        / "outputs"
        / "01_mainline_results"
        / "candidate_cada_v1"
        / "ema_bifpn_cada_v1_adapter_only_probe20"
        / "report"
        / "summary.json",
    ),
    (
        "CADA fair100 old dataset",
        REPO_ROOT
        / "outputs"
        / "05_failed_experiments"
        / "08_other_failed"
        / "ema_bifpn_cada_v1_adapter_only_full_fair100"
        / "report"
        / "summary.json",
    ),
    (
        "YOLO11n-pose unified",
        REPO_ROOT / "outputs" / "03_unified_evaluation" / "eval_unification" / "yolo_pose_unified_report" / "summary.json",
    ),
]


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
            l2 = pred.get("l2_px")
            if l2 is None:
                continue
            value = safe_float(l2)
            if value > 30:
                gt30 += 1
            if value > 50:
                gt50 += 1
    if gt30 == 0 and gt50 == 0:
        # Most reports store records without per-pred l2; avoid implying zero.
        return None, None
    return gt30, gt50


def extract_metrics(label: str, summary_path: Path, is_candidate: bool = False) -> dict[str, Any]:
    summary = load_json(summary_path)
    if summary is None:
        return {
            "model": label,
            "summary_path": str(summary_path),
            "status": "missing",
        }

    test = nested(summary, ["primary_checkpoint_split_summary", "test"], {})
    det = test.get("grape_detection", {}) if isinstance(test, dict) else {}
    has = test.get("has_picking", {}) if isinstance(test, dict) else {}
    point = test.get("picking_point", {}) if isinstance(test, dict) else {}
    unified = nested(summary, ["unified_point_metrics", "test"], {})
    instance = unified.get("instance_chain", {}) if isinstance(unified, dict) else {}
    global_chain = unified.get("global_chain", {}) if isinstance(unified, dict) else {}

    report_dir = summary_path.parent
    gt30, gt50 = count_l2(report_dir / "test_prediction_records.json")

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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def decide(candidate: dict[str, Any], old_ema: dict[str, Any] | None) -> tuple[str, list[str]]:
    notes: list[str] = []
    if candidate.get("status") != "ok":
        return "blocked_no_new1804_summary", ["new1804 summary.json not found or unreadable."]
    if old_ema is None or old_ema.get("status") != "ok":
        return "reference_missing_review_manually", ["Old EMA_BIFPN reference summary missing; only absolute metrics are available."]

    ap_drop = safe_float(old_ema.get("AP")) - safe_float(candidate.get("AP"))
    f1_delta = safe_float(candidate.get("F1")) - safe_float(old_ema.get("F1"))
    pair_delta = safe_float(candidate.get("pair")) - safe_float(old_ema.get("pair"))
    l2_delta = safe_float(candidate.get("mean_L2")) - safe_float(old_ema.get("mean_L2"))
    ppl30_delta = safe_float(candidate.get("PPL-SR@30")) - safe_float(old_ema.get("PPL-SR@30"))

    notes.append(f"AP delta vs old EMA_BIFPN: {-ap_drop:.4f}")
    notes.append(f"F1 delta vs old EMA_BIFPN: {f1_delta:.4f}")
    notes.append(f"pair delta vs old EMA_BIFPN: {pair_delta:.1f}")
    notes.append(f"mean L2 delta vs old EMA_BIFPN: {l2_delta:.3f}px")
    notes.append(f"PPL-SR@30 delta vs old EMA_BIFPN: {ppl30_delta:.4f}")

    if ap_drop <= 0.005 and f1_delta >= -0.005 and pair_delta >= -3 and l2_delta < 0 and ppl30_delta > 0:
        return "mainline_candidate", notes
    if pair_delta > 0 and l2_delta < 0:
        return "point_task_candidate_review_ap_f1", notes
    if f1_delta > 0 and pair_delta > 0 and l2_delta >= 0:
        return "coverage_improved_localization_worse", notes
    return "not_better_than_old_ema_reference", notes


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = REPORT_DIR / "summary.json"
    rows = [extract_metrics("EMA_BIFPN_NEW1804_FAIR100", candidate_path, is_candidate=True)]
    for label, path in REFERENCE_SUMMARIES:
        rows.append(extract_metrics(label, path))

    write_csv(REPORT_DIR / "new1804_main_summary.csv", rows)

    old_ema = next((row for row in rows if row["model"] == "EMA_BIFPN old dataset" and row.get("status") == "ok"), None)
    decision, notes = decide(rows[0], old_ema)
    payload = {
        "decision": decision,
        "candidate": rows[0],
        "references": rows[1:],
        "decision_notes": notes,
    }
    (REPORT_DIR / "new1804_main_decision.json").write_text(
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
        "# NEW1804 EMA_BIFPN fair100 decision",
        "",
        f"- Decision: `{decision}`",
        "- Training route: EMA_BIFPN main architecture, new `datasets/` 1804-image split, HGNetv2 unified pretrained, 100 epochs, no old checkpoint tuning.",
        "- Comparison note: old references are from the previous dataset/eval assets, so use them as historical anchors rather than same-dataset ablations.",
        "",
        "## Main Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    lines.extend(["", "## Decision Notes", ""])
    lines.extend(f"- {note}" for note in notes)
    missing = [row for row in rows if row.get("status") != "ok"]
    if missing:
        lines.extend(["", "## Missing References", ""])
        lines.extend(f"- {row['model']}: {row['summary_path']}" for row in missing)

    (REPORT_DIR / "new1804_main_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[new1804-decision] wrote {REPORT_DIR / 'new1804_main_decision.md'}")


if __name__ == "__main__":
    main()
