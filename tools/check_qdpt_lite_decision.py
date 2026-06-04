from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grape_point_eval_utils import collect_case_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether QDPT-Lite probe20 passes the EMA_BIFPN gate.")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-records", type=Path, required=True)
    parser.add_argument("--ema-summary", type=Path, required=True)
    parser.add_argument("--ema-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="EMA_BIFPN_QDPT_LITE_V1_PROBE20")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_metrics(summary: dict, split: str = "test") -> dict:
    item = summary["primary_checkpoint_split_summary"][split]
    det = item.get("grape_detection", {})
    has = item.get("has_picking", {})
    point = item.get("picking_point", {})
    return {
        "AP": float(det.get("AP", 0.0) or 0.0),
        "AP50": float(det.get("AP50", 0.0) or 0.0),
        "F1": float(has.get("f1", 0.0) or 0.0),
        "pair": int(point.get("pair_count", 0) or 0),
        "mean_L2": float(point.get("mean_l2_px", 0.0) or 0.0),
        "median_L2": float(point.get("median_l2_px", 0.0) or 0.0),
        "p90_L2": float(point.get("p90_l2_px", 0.0) or 0.0),
        "PPL-SR@30": float(point.get("ppl_sr_30", 0.0) or 0.0),
        "PPL-SR@50": float(point.get("ppl_sr_50", 0.0) or 0.0),
        "mean_abs_dx": float(point.get("mean_abs_dx_px", point.get("mae_x_px", 0.0)) or 0.0),
        "mean_abs_dy": float(point.get("mean_abs_dy_px", point.get("mae_y_px", 0.0)) or 0.0),
    }


def l2_counts(records: list[dict]) -> dict:
    correct, _, _ = collect_case_groups(records, iou_threshold=0.5, has_picking_threshold=0.5)
    l2_values = [float(item.get("l2_px", 0.0) or 0.0) for item in correct]
    return {
        "L2>30_count": sum(1 for value in l2_values if value > 30.0),
        "L2>50_count": sum(1 for value in l2_values if value > 50.0),
    }


def fmt(value) -> str:
    if isinstance(value, int):
        return str(value)
    try:
        value = float(value)
    except Exception:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = split_metrics(load_json(args.candidate_summary))
    ema = split_metrics(load_json(args.ema_summary))
    candidate.update(l2_counts(load_json(args.candidate_records)))
    ema.update(l2_counts(load_json(args.ema_records)))

    checks = {
        "AP_drop<=0.003": candidate["AP"] >= ema["AP"] - 0.003,
        "F1_drop<=0.005": candidate["F1"] >= ema["F1"] - 0.005,
        "pair_drop<=3": candidate["pair"] >= ema["pair"] - 3,
        "mean_L2_lower": candidate["mean_L2"] < ema["mean_L2"],
        "PPL30_higher": candidate["PPL-SR@30"] > ema["PPL-SR@30"],
        "L2gt30_lower": candidate["L2>30_count"] < ema["L2>30_count"],
        "p90_not_higher": candidate["p90_L2"] <= ema["p90_L2"],
    }
    strong_checks = {
        "mean_L2_drop>=0.5": ema["mean_L2"] - candidate["mean_L2"] >= 0.5,
        "PPL30_gain>=0.01": candidate["PPL-SR@30"] - ema["PPL-SR@30"] >= 0.01,
        "L2gt30_drop>=3": ema["L2>30_count"] - candidate["L2>30_count"] >= 3,
    }
    passed = all(checks.values())
    strong = passed and all(strong_checks.values())
    decision = "strong_mainline_candidate" if strong else ("mainline_candidate" if passed else "reject_or_mechanism_only")
    payload = {
        "candidate_label": args.label,
        "decision": decision,
        "passed_hard_gate": passed,
        "passed_strong_gate": strong,
        "candidate": candidate,
        "ema_bifpn": ema,
        "hard_checks": checks,
        "strong_checks": strong_checks,
    }
    args.output_dir.joinpath("qdpt_lite_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = ["model", "AP", "AP50", "F1", "pair", "mean_L2", "median_L2", "p90_L2", "PPL-SR@30", "PPL-SR@50", "L2>30_count", "L2>50_count", "mean_abs_dx", "mean_abs_dy", "decision"]
    with args.output_dir.joinpath("qdpt_lite_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"model": "EMA_BIFPN default", **ema, "decision": "reference"})
        writer.writerow({"model": args.label, **candidate, "decision": decision})

    lines = [
        "# QDPT-Lite Decision",
        "",
        f"- decision: `{decision}`",
        f"- hard gate passed: `{passed}`",
        f"- strong gate passed: `{strong}`",
        "",
        "## Hard Gate",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Comparison", "", "| model | AP | AP50 | F1 | pair | mean L2 | p90 L2 | PPL@30 | PPL@50 | L2>30 | L2>50 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, metrics in (("EMA_BIFPN default", ema), (args.label, candidate)):
        lines.append(
            f"| {name} | {fmt(metrics['AP'])} | {fmt(metrics['AP50'])} | {fmt(metrics['F1'])} | {metrics['pair']} | "
            f"{fmt(metrics['mean_L2'])} | {fmt(metrics['p90_L2'])} | {fmt(metrics['PPL-SR@30'])} | "
            f"{fmt(metrics['PPL-SR@50'])} | {metrics['L2>30_count']} | {metrics['L2>50_count']} |"
        )
    lines.extend([
        "",
        "## Required Answers",
        "- Object query / matcher / postprocessor: unchanged by this experiment; QDPT-Lite is an offset-only decoder path.",
        "- has_picking path: kept on the original coverage feature path; has-logit distill is used only as a stabilizer.",
        "- Initial output closeness: see the smoke report identity section.",
        "- Improvement source: this probe exposes standard final offsets only; QDPT debug tensors are validated in smoke, not exported into public prediction records.",
        "- Stage 2 decision: only consider fair training or UAOL/CQPC if the hard gate passes.",
    ])
    args.output_dir.joinpath("qdpt_lite_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
