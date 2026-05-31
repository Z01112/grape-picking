from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a GPPoint-DETR experiment passes the picking-first gate.")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, default=None, help="Optional EMA_BIFPN summary for PPL-SR comparison.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--reference-label", default="V7_EXP2_MAIN fair retrain")
    parser.add_argument("--base-label", default="EMA_BIFPN")
    parser.add_argument("--min-ap", type=float, default=0.624)
    parser.add_argument("--min-ap50", type=float, default=0.876)
    parser.add_argument("--min-has-f1", type=float, default=0.7615)
    parser.add_argument("--min-pair-count", type=int, default=183)
    parser.add_argument("--target-pair-count", type=int, default=185)
    parser.add_argument("--max-mean-l2", type=float, default=24.50)
    parser.add_argument("--target-mean-l2", type=float, default=23.40)
    parser.add_argument("--min-ppl-sr30", type=float, default=None)
    parser.add_argument("--min-ppl-sr50", type=float, default=None)
    parser.add_argument("--max-p90-l2", type=float, default=49.43, help="Diagnostic only; not a hard gate by default.")
    parser.add_argument("--legacy-p90-gate", action="store_true", help="Use p90 as a hard gate for historical experiments.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def test_summary(summary: dict) -> dict:
    if "primary_checkpoint_split_summary" in summary:
        return summary["primary_checkpoint_split_summary"]["test"]
    if "selected_test" in summary and "base_reference" in summary:
        base = summary.get("base_reference", {})
        selected = summary.get("selected_test", {})
        return {
            "grape_detection": {
                "AP": base.get("AP"),
                "AP50": base.get("AP50"),
                "AR100": base.get("AR100"),
            },
            "has_picking": {
                "f1": selected.get("f1"),
            },
            "picking_point": {
                "pair_count": selected.get("pair_count"),
                "mean_l2_px": selected.get("mean_l2"),
                "median_l2_px": selected.get("median_l2"),
                "p90_l2_px": selected.get("p90_l2"),
                "ppl_sr_30": selected.get("ppl_sr_30"),
                "ppl_sr_50": selected.get("ppl_sr_50"),
                "mae_x_px": selected.get("dx"),
                "mae_y_px": selected.get("dy"),
                "size_group_l2_px": {
                    "small": {"mean_l2_px": selected.get("small_l2")},
                    "medium": {"mean_l2_px": selected.get("medium_l2")},
                    "large": {"mean_l2_px": selected.get("large_l2")},
                },
            },
        }
    raise KeyError("summary must contain primary_checkpoint_split_summary or selected_test/base_reference")


def metrics(summary: dict) -> dict:
    test = test_summary(summary)
    det = test.get("grape_detection", {})
    has = test.get("has_picking", {})
    point = test.get("picking_point", {})
    size = point.get("size_group_l2_px", {}) or {}
    return {
        "AP": finite_float(det.get("AP"), 0.0),
        "AP50": finite_float(det.get("AP50"), 0.0),
        "AR100": finite_float(det.get("AR100"), 0.0),
        "has_f1": finite_float(has.get("f1"), 0.0),
        "pair_count": int(point.get("pair_count", 0) or 0),
        "mean_l2": finite_float(point.get("mean_l2_px"), 0.0),
        "median_l2": finite_float(point.get("median_l2_px")),
        "p90_l2": finite_float(point.get("p90_l2_px")),
        "ppl_sr_30": finite_float(point.get("ppl_sr_30")),
        "ppl_sr_50": finite_float(point.get("ppl_sr_50")),
        "dx": finite_float(point.get("mae_x_px", point.get("mean_abs_dx_px"))),
        "dy": finite_float(point.get("mae_y_px", point.get("mean_abs_dy_px"))),
        "small_l2": finite_float((size.get("small") or {}).get("mean_l2_px")),
        "medium_l2": finite_float((size.get("medium") or {}).get("mean_l2_px")),
        "large_l2": finite_float((size.get("large") or {}).get("mean_l2_px")),
    }


def pass_or_skip(value: float, threshold: float | None, op: str) -> dict:
    if threshold is None or not math.isfinite(threshold):
        return {"passed": None, "status": "skipped", "threshold": threshold}
    if not math.isfinite(value):
        return {"passed": None, "status": "missing", "threshold": threshold}
    if op == ">=":
        passed = value >= threshold
    elif op == "<":
        passed = value < threshold
    else:
        raise ValueError(op)
    return {"passed": bool(passed), "status": "pass" if passed else "fail", "threshold": threshold}


def ppl_threshold(args: argparse.Namespace, reference: dict, base: dict | None, key: str) -> float | None:
    explicit = args.min_ppl_sr30 if key == "ppl_sr_30" else args.min_ppl_sr50
    if explicit is not None:
        return explicit
    values = [reference.get(key, float("nan"))]
    if base is not None:
        values.append(base.get(key, float("nan")))
    values = [float(v) for v in values if math.isfinite(float(v))]
    return max(values) if values else None


def bool_gate(checks: dict[str, dict]) -> bool:
    hard_statuses = [item["passed"] for item in checks.values()]
    return all(status is not False for status in hard_statuses)


def fmt(value: float | int, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    candidate = metrics(load_json(args.candidate_summary))
    reference = metrics(load_json(args.reference_summary))
    base = metrics(load_json(args.base_summary)) if args.base_summary else None

    ppl30_min = ppl_threshold(args, reference, base, "ppl_sr_30")
    ppl50_min = ppl_threshold(args, reference, base, "ppl_sr_50")

    hard_checks = {
        "AP": pass_or_skip(candidate["AP"], args.min_ap, ">="),
        "AP50": pass_or_skip(candidate["AP50"], args.min_ap50, ">="),
        "has_f1": pass_or_skip(candidate["has_f1"], args.min_has_f1, ">="),
        "pair_count": pass_or_skip(candidate["pair_count"], args.min_pair_count, ">="),
        "mean_l2": pass_or_skip(candidate["mean_l2"], args.max_mean_l2, "<"),
        "ppl_sr_30": pass_or_skip(candidate["ppl_sr_30"], ppl30_min, ">="),
        "ppl_sr_50": pass_or_skip(candidate["ppl_sr_50"], ppl50_min, ">="),
    }
    if args.legacy_p90_gate:
        hard_checks["p90_l2"] = pass_or_skip(candidate["p90_l2"], args.max_p90_l2, "<")

    diagnostic_checks = {
        "target_pair_count": pass_or_skip(candidate["pair_count"], args.target_pair_count, ">="),
        "target_mean_l2": pass_or_skip(candidate["mean_l2"], args.target_mean_l2, "<"),
        "p90_l2": pass_or_skip(candidate["p90_l2"], args.max_p90_l2, "<"),
    }
    passed = bool_gate(hard_checks)
    strong = passed and all(item["passed"] is not False for item in diagnostic_checks.values())
    decision = "strong_mainline_candidate" if strong else ("mainline_candidate" if passed else "mechanism_only_or_reject")

    comparable = ["AP", "AP50", "AR100", "has_f1", "pair_count", "mean_l2", "ppl_sr_30", "ppl_sr_50"]
    diagnostics = ["median_l2", "p90_l2", "dx", "dy", "small_l2", "medium_l2", "large_l2"]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate_mode": "picking_first",
        "candidate_label": args.candidate_label,
        "reference_label": args.reference_label,
        "base_label": args.base_label if base is not None else None,
        "decision": decision,
        "passed": passed,
        "strong_candidate": strong,
        "hard_gate": {
            "AP": f">= {args.min_ap:g}",
            "AP50": f">= {args.min_ap50:g}",
            "has_f1": f">= {args.min_has_f1:g}",
            "pair_count": f">= {args.min_pair_count}",
            "mean_l2": f"< {args.max_mean_l2:g}",
            "ppl_sr_30": f">= {ppl30_min:g}" if ppl30_min is not None else "skipped: unavailable in reference/base summary",
            "ppl_sr_50": f">= {ppl50_min:g}" if ppl50_min is not None else "skipped: unavailable in reference/base summary",
            "p90_l2": f"< {args.max_p90_l2:g}" if args.legacy_p90_gate else "diagnostic only",
        },
        "target_not_hard_gate": {
            "pair_count": f">= {args.target_pair_count}",
            "mean_l2": f"< {args.target_mean_l2:g}",
            "p90_l2": f"< {args.max_p90_l2:g}",
        },
        "candidate": candidate,
        "reference": reference,
        "base": base,
        "delta": {key: candidate[key] - reference[key] for key in comparable + diagnostics if key in candidate and key in reference},
        "hard_gate_pass": hard_checks,
        "diagnostic_pass": diagnostic_checks,
    }

    output_dir = args.output_dir or args.candidate_summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mainline_gate_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Picking-first Mainline Gate Result",
        "",
        f"- Candidate: `{args.candidate_label}`",
        f"- Reference: `{args.reference_label}`",
        f"- Decision: `{decision}`",
        "- Gate mode: picking-first; `p90_l2` is diagnostic unless `--legacy-p90-gate` is set.",
        "",
        "## Paper Main Metrics",
        "",
        "| Metric | Candidate | Reference | Delta | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ("AP", "AP50", "has_f1", "pair_count", "mean_l2", "ppl_sr_30", "ppl_sr_50"):
        check = hard_checks.get(key, {})
        gate_status = check.get("status", "-")
        c = candidate[key]
        r = reference[key]
        d = c - r if math.isfinite(float(c)) and math.isfinite(float(r)) else float("nan")
        if key == "pair_count":
            lines.append(f"| {key} | {int(c)} | {int(r)} | {int(d):+d} | {gate_status} |")
        else:
            lines.append(f"| {key} | {fmt(c)} | {fmt(r)} | {fmt(d)} | {gate_status} |")

    lines.extend(
        [
            "",
            "## Diagnostic Appendix Metrics",
            "",
            "| Metric | Candidate | Reference | Delta | Note |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for key in ("median_l2", "p90_l2", "dx", "dy", "small_l2", "medium_l2", "large_l2"):
        c = candidate[key]
        r = reference[key]
        d = c - r if math.isfinite(float(c)) and math.isfinite(float(r)) else float("nan")
        note = "diagnostic only"
        if key in diagnostic_checks:
            note = diagnostic_checks[key]["status"]
        lines.append(f"| {key} | {fmt(c)} | {fmt(r)} | {fmt(d)} | {note} |")

    lines.extend(["", "## Hard Gate", "", "| Gate | Status |", "|---|---|"])
    for key, result in hard_checks.items():
        lines.append(f"| {key} | {result['status']} |")
    (output_dir / "mainline_gate_result_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
