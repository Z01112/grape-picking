from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_b1_pointca_pam_priority123 import (  # noqa: E402
    CSV_FIELDS,
    extract_metrics,
    fmt,
    fmt_int,
    safe_float,
    write_csv,
)


OUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "b0_b1_pointca_pam_ablation"

MODEL_SPECS = [
    {
        "label": "RT-DETRv4_GPHead_baseline",
        "role": "external_internal_baseline",
        "summary": REPO_ROOT / "outputs" / "02_baselines" / "rtdetrv4_new1804_baseline_fair100" / "report" / "summary.json",
    },
    {
        "label": "B0_EMA_BIFPN",
        "role": "main_reference",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "ema_bifpn_new1804_fair100" / "report" / "summary.json",
    },
    {
        "label": "B0_PointCA",
        "role": "new_ablation",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b0_point_cross_attn_fair100" / "report" / "summary.json",
    },
    {
        "label": "B0_PAM",
        "role": "new_ablation",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b0_pam_fair100" / "report" / "summary.json",
    },
    {
        "label": "B0_PointCA_PAM",
        "role": "new_candidate",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b0_point_cross_attn_pam_fair100" / "report" / "summary.json",
    },
    {
        "label": "B1",
        "role": "ablation",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b1_backbone_new1804_fair100" / "report" / "summary.json",
    },
    {
        "label": "B1_PAM",
        "role": "ablation",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b1_pam_fair100" / "report" / "summary.json",
    },
    {
        "label": "B1_PointCA",
        "role": "ablation",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b1_point_cross_attn_new1804_fair100" / "report" / "summary.json",
    },
    {
        "label": "B1_PointCA_PAM",
        "role": "candidate",
        "summary": REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_b1_point_cross_attn_pam_fair100" / "report" / "summary.json",
    },
    {
        "label": "YOLO11n_pose",
        "role": "external_baseline",
        "summary": REPO_ROOT / "outputs" / "02_baselines" / "yolo11n_pose_new1804_b8_e100_20260607_181020" / "summary.json",
    },
]


def finite(value: Any) -> bool:
    value = safe_float(value)
    return math.isfinite(value)


def delta(lhs: dict[str, Any], rhs: dict[str, Any], field: str) -> float:
    if lhs.get("status") != "ok" or rhs.get("status") != "ok":
        return float("nan")
    return safe_float(lhs.get(field)) - safe_float(rhs.get(field))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def md_table(rows: list[dict[str, Any]]) -> str:
    cols = ["model", "status", "AP", "AP50", "F1", "pair", "FN", "FP", "mean_L2", "p90_L2", "PPL-SR@30", "PPL-SR@50"]
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            if col in {"model", "status"}:
                values.append(str(row.get(col, "")))
            elif col in {"pair", "FN", "FP"}:
                values.append(fmt_int(row.get(col)))
            else:
                values.append(fmt(row.get(col)))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def decision(rows: list[dict[str, Any]]) -> str:
    lookup = {row["model"]: row for row in rows}
    b0 = lookup.get("B0_EMA_BIFPN", {})
    b0_full = lookup.get("B0_PointCA_PAM", {})
    b1_full = lookup.get("B1_PointCA_PAM", {})
    if b0_full.get("status") != "ok":
        return "pending_b0_pointca_pam_report"
    if b1_full.get("status") != "ok" or b0.get("status") != "ok":
        return "pending_reference_report"

    b0_full_beats_b1_pair = safe_float(b0_full.get("pair")) >= safe_float(b1_full.get("pair"))
    b0_full_beats_b1_l2 = safe_float(b0_full.get("mean_L2")) <= safe_float(b1_full.get("mean_L2"))
    b0_full_keeps_b0_precision = safe_float(b0_full.get("mean_L2")) <= safe_float(b0.get("mean_L2")) + 0.3 and safe_float(b0_full.get("PPL-SR@30")) >= safe_float(b0.get("PPL-SR@30")) - 0.01
    b0_full_improves_b0_coverage = safe_float(b0_full.get("F1")) > safe_float(b0.get("F1")) and safe_float(b0_full.get("pair")) > safe_float(b0.get("pair"))
    if b0_full_beats_b1_pair and b0_full_beats_b1_l2:
        return "b0_pointca_pam_preferred_over_b1_full"
    if b0_full_keeps_b0_precision and b0_full_improves_b0_coverage:
        return "b0_pointca_pam_precision_coverage_balanced_candidate"
    if b0_full_improves_b0_coverage:
        return "b0_pointca_pam_coverage_gain_with_precision_cost"
    return "b0_pointca_pam_not_better_than_existing_routes"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [extract_metrics(spec) for spec in MODEL_SPECS]
    write_csv(OUT_DIR / "b0_b1_pointca_pam_ablation_table.csv", rows, CSV_FIELDS)

    lookup = {row["model"]: row for row in rows}
    comparisons = [
        ("B0_PointCA_vs_B0", "B0_PointCA", "B0_EMA_BIFPN"),
        ("B0_PAM_vs_B0", "B0_PAM", "B0_EMA_BIFPN"),
        ("B0_Full_vs_B0", "B0_PointCA_PAM", "B0_EMA_BIFPN"),
        ("B0_Full_vs_B1_Full", "B0_PointCA_PAM", "B1_PointCA_PAM"),
        ("B1_Full_vs_B0_Full", "B1_PointCA_PAM", "B0_PointCA_PAM"),
    ]
    delta_rows = []
    for name, lhs_name, rhs_name in comparisons:
        lhs = lookup.get(lhs_name, {})
        rhs = lookup.get(rhs_name, {})
        status = "ok" if lhs.get("status") == "ok" and rhs.get("status") == "ok" else "missing_input"
        delta_rows.append({
            "comparison": name,
            "lhs": lhs_name,
            "rhs": rhs_name,
            "status": status,
            "AP_delta": delta(lhs, rhs, "AP"),
            "AP50_delta": delta(lhs, rhs, "AP50"),
            "F1_delta": delta(lhs, rhs, "F1"),
            "pair_delta": delta(lhs, rhs, "pair"),
            "FN_delta": delta(lhs, rhs, "FN"),
            "FP_delta": delta(lhs, rhs, "FP"),
            "mean_L2_delta": delta(lhs, rhs, "mean_L2"),
            "p90_L2_delta": delta(lhs, rhs, "p90_L2"),
            "PPL-SR@30_delta": delta(lhs, rhs, "PPL-SR@30"),
            "PPL-SR@50_delta": delta(lhs, rhs, "PPL-SR@50"),
        })
    write_csv(
        OUT_DIR / "b0_b1_pointca_pam_delta_table.csv",
        delta_rows,
        [
            "comparison",
            "lhs",
            "rhs",
            "status",
            "AP_delta",
            "AP50_delta",
            "F1_delta",
            "pair_delta",
            "FN_delta",
            "FP_delta",
            "mean_L2_delta",
            "p90_L2_delta",
            "PPL-SR@30_delta",
            "PPL-SR@50_delta",
        ],
    )

    payload = {
        "decision": decision(rows),
        "models": rows,
        "comparisons": delta_rows,
    }
    write_json(OUT_DIR / "b0_b1_pointca_pam_ablation_summary.json", payload)

    md = [
        "# B0/B1 PointCA PAM Ablation Summary",
        "",
        "This table compares B0 and B1 combinations under the same current default dataset and unified report format.",
        "",
        "## Main Table",
        "",
        md_table(rows),
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Generated Files",
        "",
        "- `b0_b1_pointca_pam_ablation_table.csv`",
        "- `b0_b1_pointca_pam_delta_table.csv`",
        "- `b0_b1_pointca_pam_ablation_summary.json`",
    ]
    (OUT_DIR / "b0_b1_pointca_pam_ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
