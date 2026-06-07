from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_new1804_main_decision import extract_metrics, fmt, write_csv

BASELINE_REPORT_DIR = REPO_ROOT / "outputs" / "02_baselines" / "rtdetrv4_new1804_baseline_fair100" / "report"


def main() -> None:
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        extract_metrics("RTDETRV4_BASELINE_NEW1804_FAIR100", BASELINE_REPORT_DIR / "summary.json", is_candidate=True),
        extract_metrics(
            "EMA_BIFPN_NEW1804_FAIR100",
            REPO_ROOT / "outputs" / "01_mainline_results" / "ema_bifpn_new1804_fair100" / "report" / "summary.json",
        ),
        extract_metrics(
            "V7_EXP2_MAIN old dataset",
            REPO_ROOT / "outputs" / "03_unified_evaluation" / "eval_unification" / "v7_exp2_unified_report" / "summary.json",
        ),
        extract_metrics(
            "EMA_BIFPN old dataset",
            REPO_ROOT / "outputs" / "03_unified_evaluation" / "eval_unification" / "ema_bifpn_unified_report" / "summary.json",
        ),
    ]
    write_csv(BASELINE_REPORT_DIR / "new1804_baseline_comparison.csv", rows)

    baseline = rows[0]
    ema = rows[1]
    decision = "blocked_missing_summary"
    notes: list[str] = []
    if baseline.get("status") == "ok" and ema.get("status") == "ok":
        ap_delta = baseline["AP"] - ema["AP"]
        f1_delta = baseline["F1"] - ema["F1"]
        pair_delta = baseline["pair"] - ema["pair"]
        l2_delta = baseline["mean_L2"] - ema["mean_L2"]
        ppl30_delta = baseline["PPL-SR@30"] - ema["PPL-SR@30"]
        notes = [
            f"Baseline AP delta vs EMA_BIFPN_NEW1804: {ap_delta:.4f}",
            f"Baseline F1 delta vs EMA_BIFPN_NEW1804: {f1_delta:.4f}",
            f"Baseline pair delta vs EMA_BIFPN_NEW1804: {pair_delta:.1f}",
            f"Baseline mean L2 delta vs EMA_BIFPN_NEW1804: {l2_delta:.3f}px",
            f"Baseline PPL-SR@30 delta vs EMA_BIFPN_NEW1804: {ppl30_delta:.4f}",
        ]
        if ap_delta >= -0.003 and f1_delta >= -0.005 and pair_delta >= -3 and l2_delta <= 0 and ppl30_delta >= 0:
            decision = "baseline_matches_or_beats_ema_bifpn"
        elif ap_delta < -0.003 and f1_delta < -0.005 and pair_delta < -3 and l2_delta > 0:
            decision = "ema_bifpn_clearly_better_on_new1804"
        else:
            decision = "mixed_tradeoff_review"

    payload = {
        "decision": decision,
        "baseline": baseline,
        "comparisons": rows[1:],
        "decision_notes": notes,
    }
    (BASELINE_REPORT_DIR / "new1804_baseline_decision.json").write_text(
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
        "# NEW1804 RT-DETRv4 baseline decision",
        "",
        f"- Decision: `{decision}`",
        "- Baseline route: current RT-DETRv4/GPPoint-DETR main config without EMA_BIFPN encoder enhancement.",
        "- Fairness: same `datasets/` split, same HGNetv2 pretrained initialization, 100 epochs, no old checkpoint tuning.",
        "",
        "## Main Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        if row.get("status") == "ok":
            lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    if notes:
        lines.extend(["", "## Decision Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    missing = [row for row in rows if row.get("status") != "ok"]
    if missing:
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(f"- {row['model']}: {row['summary_path']}" for row in missing)
    (BASELINE_REPORT_DIR / "new1804_baseline_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[baseline-decision] wrote {BASELINE_REPORT_DIR / 'new1804_baseline_decision.md'}")


if __name__ == "__main__":
    main()
