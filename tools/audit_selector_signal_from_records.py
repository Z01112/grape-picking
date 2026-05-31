from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_grape_point_report import safe_float, summarize_split_error


DEFAULT_REPORT_DIR = (
    REPO_ROOT
    / "outputs/05_selector_experiments/"
    / "ema_bifpn_detached_query_selector_v2_has_fair100_backbone_pretrain_20260530/report_best_point_l2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether detached selector prediction records contain a useful picking selection signal."
    )
    parser.add_argument("--valid-records", type=Path, default=DEFAULT_REPORT_DIR / "valid_prediction_records.json")
    parser.add_argument("--test-records", type=Path, default=DEFAULT_REPORT_DIR / "test_prediction_records.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_REPORT_DIR / "summary.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/03_global_analysis/selector_signal_audit_20260531",
    )
    parser.add_argument("--thresholds", default="0.30:0.80:0.02")
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--min-pair-ratio", type=float, default=0.98)
    parser.add_argument("--max-f1-drop", type=float, default=0.002)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float_list(spec: str) -> list[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def parse_thresholds(spec: str) -> list[float]:
    if ":" not in spec:
        return parse_float_list(spec)
    start, stop, step = [float(part) for part in spec.split(":")]
    values = []
    cur = start
    while cur <= stop + step * 0.5:
        values.append(round(cur, 6))
        cur += step
    return values


def score_stats(records: list[dict], field: str) -> dict:
    values = []
    for record in records:
        for pred in record.get("pred_instances", []):
            if field in pred:
                values.append(float(pred[field]))
    if not values:
        return {"count": 0, "unique_rounded": 0, "min": None, "max": None, "mean": None, "std": None}
    rounded = {round(v, 6) for v in values}
    return {
        "count": len(values),
        "unique_rounded": len(rounded),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def add_score(records: list[dict], rule: str, alpha: float, output_key: str = "selector_audit_score") -> list[dict]:
    for record in records:
        for pred in record.get("pred_instances", []):
            has_score = max(0.0, safe_float(pred.get("has_picking_score", 0.0), 0.0))
            selector_score = max(0.0, min(1.0, safe_float(pred.get("point_selector_score", 1.0), 1.0)))
            selector_final = max(0.0, safe_float(pred.get("point_selector_final_score", 0.0), 0.0))
            if rule == "raw_has":
                value = has_score
            elif rule == "selector_only":
                value = selector_score
            elif rule == "selector_final":
                value = selector_final
            elif rule == "has_times_selector_alpha":
                value = has_score * (selector_score ** alpha)
            else:
                raise ValueError(rule)
            pred[output_key] = float(value)
    return records


def metrics(records: list[dict], threshold: float, score_key: str = "selector_audit_score") -> dict:
    summary = summarize_split_error(records, threshold, visibility_score_key=score_key)
    tp = int(summary.get("has_picking_correct_count", 0))
    fp = int(summary.get("has_picking_false_positive_count", 0))
    fn = int(summary.get("has_picking_false_negative_count", 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    size = summary.get("size_group_l2_px", {}) or {}
    pair = int(summary.get("point_pair_count", 0))
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pair_count": pair,
        "mean_l2": safe_float(summary.get("mean_l2_px")),
        "median_l2": safe_float(summary.get("median_l2_px")),
        "p90_l2": safe_float(summary.get("p90_l2_px")),
        "dx": safe_float(summary.get("mean_abs_dx_px")),
        "dy": safe_float(summary.get("mean_abs_dy_px")),
        "ppl_sr_30": safe_float(summary.get("ppl_sr_30"), 0.0),
        "ppl_sr_50": safe_float(summary.get("ppl_sr_50"), 0.0),
        "small_l2": safe_float((size.get("small") or {}).get("mean_l2_px")),
        "medium_l2": safe_float((size.get("medium") or {}).get("mean_l2_px")),
        "large_l2": safe_float((size.get("large") or {}).get("mean_l2_px")),
    }


def run_grid(records: list[dict], split: str, thresholds: list[float], alphas: list[float]) -> list[dict]:
    rows = []
    rules = ["raw_has", "selector_only", "selector_final", "has_times_selector_alpha"]
    for rule in rules:
        rule_alphas = [0.0] if rule != "has_times_selector_alpha" else alphas
        for alpha in rule_alphas:
            scored = add_score(records, rule, alpha)
            for threshold in thresholds:
                row = metrics(scored, threshold)
                row.update({"split": split, "rule": rule, "alpha": alpha})
                rows.append(row)
    return rows


def select_valid_rule(valid_rows: list[dict], baseline: dict, min_pair_ratio: float, max_f1_drop: float) -> dict:
    min_pair = math.floor(int(baseline["pair_count"]) * min_pair_ratio)
    min_f1 = float(baseline["f1"]) - max_f1_drop
    feasible = [
        row
        for row in valid_rows
        if int(row["pair_count"]) >= min_pair and float(row["f1"]) >= min_f1
    ]
    if not feasible:
        feasible = valid_rows
    return min(
        feasible,
        key=lambda row: (
            safe_float(row.get("mean_l2"), float("inf")),
            -safe_float(row.get("ppl_sr_30"), 0.0),
            -safe_float(row.get("ppl_sr_50"), 0.0),
            -safe_float(row.get("pair_count"), 0.0),
            -safe_float(row.get("f1"), 0.0),
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def row_md(label: str, row: dict) -> str:
    return (
        f"| {label} | {row['rule']} | {fmt(row['alpha'], 2)} | {fmt(row['threshold'], 2)} | "
        f"{fmt(row['f1'])} | {int(row['pair_count'])} | {fmt(row['mean_l2'], 2)} | "
        f"{fmt(row['ppl_sr_30'])} | {fmt(row['ppl_sr_50'])} | {fmt(row['p90_l2'], 2)} |"
    )


def write_report(path: Path, payload: dict) -> None:
    stats = payload["score_stats"]
    selected_valid = payload["selected_valid"]
    selected_test = payload["selected_test"]
    baseline_valid = payload["baseline_valid"]
    baseline_test = payload["baseline_test"]
    verdict = payload["verdict"]
    lines = [
        "# Selector Signal Audit",
        "",
        f"- Generated at: {payload['generated_at']}",
        "- Scope: no-training audit from existing valid/test prediction records.",
        "- Rule: select score rule on valid only, then freeze the same rule on formal test.",
        "",
        "## Score Distribution",
        "",
        "| Split | Field | count | unique | min | max | mean | std |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, split_stats in stats.items():
        for field, field_stats in split_stats.items():
            lines.append(
                f"| {split_name} | {field} | {field_stats['count']} | {field_stats['unique_rounded']} | "
                f"{fmt(field_stats['min'], 6)} | {fmt(field_stats['max'], 6)} | "
                f"{fmt(field_stats['mean'], 6)} | {fmt(field_stats['std'], 6)} |"
            )
    lines.extend(
        [
            "",
            "## Picking-First Metrics",
            "",
            "| Protocol | rule | alpha | threshold | F1 | pair | mean L2 | PPL-SR@30 | PPL-SR@50 | p90 L2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            row_md("valid raw_has@0.50", baseline_valid),
            row_md("valid selected", selected_valid),
            row_md("test raw_has@0.50", baseline_test),
            row_md("test selected", selected_test),
            "",
            "## Verdict",
            "",
            f"- Decision: `{verdict['decision']}`",
            f"- Selector score is constant: `{verdict['selector_score_constant']}`",
            f"- Selected rule improves test mean L2 over raw_has@0.50: `{verdict['test_mean_improved']}`",
            f"- Selected rule keeps test pair/F1 within audit bounds: `{verdict['test_coverage_kept']}`",
            "",
            "## Next Action",
            "",
            "- Do not continue the same detached selector head as the next training direction unless the selector output path is first fixed and validated to be non-constant.",
            "- Keep `EMA_BIFPN_HP_PICK_PROTOCOL_HAS062` as the current mainline candidate.",
            "- The next model change should target point coordinate representation or a verified selector output path, but it must start with an interface check that proves the new score is non-constant on a mini validation batch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    valid_records = read_records(args.valid_records)
    test_records = read_records(args.test_records)
    thresholds = parse_thresholds(args.thresholds)
    alphas = parse_float_list(args.alphas)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = run_grid(valid_records, "valid", thresholds, alphas)
    test_rows = run_grid(test_records, "test", thresholds, alphas)
    all_rows = valid_rows + test_rows

    baseline_valid = next(
        row for row in valid_rows
        if row["rule"] == "raw_has" and abs(float(row["threshold"]) - 0.5) < 1e-9
    )
    baseline_test = next(
        row for row in test_rows
        if row["rule"] == "raw_has" and abs(float(row["threshold"]) - 0.5) < 1e-9
    )
    selected_valid = select_valid_rule(valid_rows, baseline_valid, args.min_pair_ratio, args.max_f1_drop)
    selected_test = next(
        row
        for row in test_rows
        if row["rule"] == selected_valid["rule"]
        and abs(float(row["alpha"]) - float(selected_valid["alpha"])) < 1e-9
        and abs(float(row["threshold"]) - float(selected_valid["threshold"])) < 1e-9
    )

    score_stats_payload = {
        "valid": {
            "has_picking_score": score_stats(valid_records, "has_picking_score"),
            "point_selector_score": score_stats(valid_records, "point_selector_score"),
            "point_selector_final_score": score_stats(valid_records, "point_selector_final_score"),
        },
        "test": {
            "has_picking_score": score_stats(test_records, "has_picking_score"),
            "point_selector_score": score_stats(test_records, "point_selector_score"),
            "point_selector_final_score": score_stats(test_records, "point_selector_final_score"),
        },
    }
    selector_constant = (
        score_stats_payload["valid"]["point_selector_score"]["unique_rounded"] <= 1
        and score_stats_payload["test"]["point_selector_score"]["unique_rounded"] <= 1
    )
    test_mean_improved = safe_float(selected_test["mean_l2"]) < safe_float(baseline_test["mean_l2"])
    test_coverage_kept = (
        int(selected_test["pair_count"]) >= math.floor(int(baseline_test["pair_count"]) * args.min_pair_ratio)
        and safe_float(selected_test["f1"]) >= safe_float(baseline_test["f1"]) - args.max_f1_drop
    )
    decision = "selector_signal_reject" if selector_constant or not (test_mean_improved and test_coverage_kept) else "selector_signal_candidate"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "valid_records": str(args.valid_records),
        "test_records": str(args.test_records),
        "summary": str(args.summary),
        "baseline_valid": baseline_valid,
        "baseline_test": baseline_test,
        "selected_valid": selected_valid,
        "selected_test": selected_test,
        "score_stats": score_stats_payload,
        "verdict": {
            "decision": decision,
            "selector_score_constant": selector_constant,
            "test_mean_improved": test_mean_improved,
            "test_coverage_kept": test_coverage_kept,
        },
    }
    write_csv(args.output_dir / "selector_signal_audit_all.csv", all_rows)
    (args.output_dir / "selector_signal_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "selector_signal_audit_report.md", payload)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
