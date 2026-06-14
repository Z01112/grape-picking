from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grape_point_eval_utils import collect_case_groups  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "unified_checkpoint_selection_v1"

METRIC_FIELDS = [
    "AP",
    "AP50",
    "F1",
    "pair",
    "global_visible_recall",
    "mean_L2",
    "median_L2",
    "p90_L2",
    "PPL30",
    "PPL50",
    "L2gt30",
    "L2gt50",
]
CSV_FIELDS = [
    "checkpoint_name",
    "checkpoint_path",
    "selection_role",
    "valid_AP",
    "valid_AP50",
    "valid_F1",
    "valid_pair",
    "valid_global_visible_recall",
    "valid_mean_L2",
    "valid_median_L2",
    "valid_p90_L2",
    "valid_PPL30",
    "valid_PPL50",
    "valid_L2gt30",
    "valid_L2gt50",
    "test_AP",
    "test_AP50",
    "test_F1",
    "test_pair",
    "test_global_visible_recall",
    "test_mean_L2",
    "test_median_L2",
    "test_p90_L2",
    "test_PPL30",
    "test_PPL50",
    "test_L2gt30",
    "test_L2gt50",
    "valid_composite",
    "passes_hard_filter",
    "missing_metrics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified valid-based checkpoint selection reports.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--main-checkpoint-policy", default="composite", choices=["composite"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def nested(payload: dict[str, Any] | None, path: list[str], default: Any = None) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def records_l2_counts(path: Path) -> tuple[int | None, int | None, dict[str, float] | None]:
    payload = load_json(path)
    if isinstance(payload, dict):
        payload = payload.get("records") or payload.get("predictions") or []
    if not isinstance(payload, list) or not payload:
        return None, None, None
    correct, _, _ = collect_case_groups(payload)
    l2_values = [safe_float(item.get("l2_px")) for item in correct if math.isfinite(safe_float(item.get("l2_px")))]
    if not l2_values:
        return 0, 0, None
    return (
        sum(1 for value in l2_values if value > 30),
        sum(1 for value in l2_values if value > 50),
        {
            "PPL30": sum(1 for value in l2_values if value <= 30) / len(l2_values),
            "PPL50": sum(1 for value in l2_values if value <= 50) / len(l2_values),
            "p90_L2": sorted(l2_values)[min(len(l2_values) - 1, int(math.ceil(0.9 * len(l2_values)) - 1))],
        },
    )


def normalize_checkpoint_name(name: str) -> str:
    return name if name.endswith((".pth", ".pt")) else f"{name}.pth"


def role_for_checkpoint(name: str, main_name: str, metric_bests: dict[str, str] | None = None) -> str:
    if name == main_name:
        return "main_selected"
    mapping = {
        "best_grape_ap.pth": "best_grape_ap",
        "best_has_picking_f1.pth": "best_has_picking_f1",
        "best_point_l2.pth": "best_point_l2",
        "last.pth": "last",
    }
    if name in mapping:
        return mapping[name]
    metric_bests = metric_bests or {}
    if metric_bests.get("valid_PPL30") == name:
        return "best_ppl30"
    if metric_bests.get("valid_p90_L2") == name:
        return "best_p90"
    if metric_bests.get("valid_L2gt30") == name:
        return "best_l2gt30"
    return "other"


def map_checkpoint_metrics(
    checkpoint_name: str,
    checkpoint_path: str,
    comparison: dict[str, Any],
    summary: dict[str, Any],
    report_dir: Path,
    is_main: bool,
) -> dict[str, Any]:
    row = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "selection_role": "",
    }
    for split in ["valid", "test"]:
        src = comparison.get(split, {}) if isinstance(comparison, dict) else {}
        row[f"{split}_AP"] = safe_float(src.get("grape_AP"))
        row[f"{split}_AP50"] = safe_float(src.get("grape_AP50"))
        row[f"{split}_F1"] = safe_float(src.get("has_picking_f1"))
        row[f"{split}_pair"] = safe_float(src.get("point_pair_count"))
        row[f"{split}_global_visible_recall"] = float("nan")
        row[f"{split}_mean_L2"] = safe_float(src.get("point_mean_l2_px"))
        row[f"{split}_median_L2"] = float("nan")
        row[f"{split}_p90_L2"] = float("nan")
        row[f"{split}_PPL30"] = float("nan")
        row[f"{split}_PPL50"] = float("nan")
        row[f"{split}_L2gt30"] = float("nan")
        row[f"{split}_L2gt50"] = float("nan")

    if is_main:
        for split in ["valid", "test"]:
            test = nested(summary, ["primary_checkpoint_split_summary", split], {})
            det = test.get("grape_detection", {}) if isinstance(test, dict) else {}
            has = test.get("has_picking", {}) if isinstance(test, dict) else {}
            point = test.get("picking_point", {}) if isinstance(test, dict) else {}
            unified = nested(summary, ["unified_point_metrics", split], {})
            global_chain = unified.get("global_chain", {}) if isinstance(unified, dict) else {}
            row[f"{split}_AP"] = safe_float(det.get("AP", row.get(f"{split}_AP")))
            row[f"{split}_AP50"] = safe_float(det.get("AP50", row.get(f"{split}_AP50")))
            row[f"{split}_F1"] = safe_float(has.get("f1", row.get(f"{split}_F1")))
            row[f"{split}_pair"] = safe_float(point.get("pair_count", row.get(f"{split}_pair")))
            row[f"{split}_global_visible_recall"] = safe_float(global_chain.get("global_visible_recall"))
            row[f"{split}_mean_L2"] = safe_float(point.get("mean_l2_px", row.get(f"{split}_mean_L2")))
            row[f"{split}_median_L2"] = safe_float(point.get("median_l2_px"))
            row[f"{split}_p90_L2"] = safe_float(point.get("p90_l2_px"))
            row[f"{split}_PPL30"] = safe_float(point.get("ppl_sr_30"))
            row[f"{split}_PPL50"] = safe_float(point.get("ppl_sr_50"))
            gt30, gt50, extra = records_l2_counts(report_dir / f"{split}_prediction_records.json")
            if gt30 is not None:
                row[f"{split}_L2gt30"] = float(gt30)
            if gt50 is not None:
                row[f"{split}_L2gt50"] = float(gt50)
            if extra:
                row[f"{split}_PPL30"] = extra["PPL30"]
                row[f"{split}_PPL50"] = extra["PPL50"]
                row[f"{split}_p90_L2"] = extra["p90_L2"]

    missing = []
    for field in CSV_FIELDS:
        if field.startswith("valid_") or field.startswith("test_"):
            value = row.get(field)
            if isinstance(value, float) and math.isnan(value):
                missing.append(field)
    row["missing_metrics"] = ";".join(missing)
    return row


def scan_checkpoint_rows(run_dir: Path, report_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    main_name = summary.get("primary_checkpoint_name") or Path(str(summary.get("primary_checkpoint", "best_composite.pth"))).name
    comparison = summary.get("checkpoint_comparison") if isinstance(summary.get("checkpoint_comparison"), dict) else {}
    names = set(comparison.keys())
    names.update(path.name for path in run_dir.glob("*.pth"))
    if not names and summary.get("weights"):
        names.add(Path(str(summary.get("weights"))).name)
    if not names:
        names.add(main_name or "no_checkpoint_found")
    rows = []
    for name in sorted(names):
        norm_name = normalize_checkpoint_name(name)
        comp = comparison.get(name) or comparison.get(norm_name) or {}
        path = comp.get("path") if isinstance(comp, dict) else None
        if path is None:
            local = run_dir / norm_name
            path = str(local) if local.exists() else ""
        rows.append(map_checkpoint_metrics(norm_name, path, comp, summary, report_dir, norm_name == main_name))
    return rows


def load_baseline_valid(summary_path: Path | None) -> dict[str, float] | None:
    if summary_path is None or not summary_path.exists():
        return None
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        return None
    test = nested(summary, ["primary_checkpoint_split_summary", "valid"], {})
    det = test.get("grape_detection", {}) if isinstance(test, dict) else {}
    has = test.get("has_picking", {}) if isinstance(test, dict) else {}
    point = test.get("picking_point", {}) if isinstance(test, dict) else {}
    return {
        "AP": safe_float(det.get("AP")),
        "F1": safe_float(has.get("f1")),
        "pair": safe_float(point.get("pair_count")),
        "PPL30": safe_float(point.get("ppl_sr_30")),
        "L2gt30": float("nan"),
    }


def hard_filter(row: dict[str, Any], baseline: dict[str, float] | None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    ap = safe_float(row.get("valid_AP"))
    f1 = safe_float(row.get("valid_F1"))
    pair = safe_float(row.get("valid_pair"))
    ppl30 = safe_float(row.get("valid_PPL30"))
    l2gt30 = safe_float(row.get("valid_L2gt30"))
    if baseline:
        checks = [
            ("valid_AP", ap, ">=", max(safe_float(baseline.get("AP")) - 0.005, 0.0)),
            ("valid_F1", f1, ">=", safe_float(baseline.get("F1")) - 0.003),
            ("valid_pair", pair, ">=", safe_float(baseline.get("pair")) - 2),
            ("valid_PPL30", ppl30, ">=", safe_float(baseline.get("PPL30")) - 0.005),
        ]
        base_l2gt30 = safe_float(baseline.get("L2gt30"))
        if math.isfinite(base_l2gt30):
            checks.append(("valid_L2gt30", l2gt30, "<=", base_l2gt30 + 2))
    else:
        checks = [
            ("valid_AP", ap, ">=", 0.60),
            ("valid_F1", f1, ">=", 0.85),
            ("valid_PPL30", ppl30, ">=", 0.85),
        ]
    for name, value, op, threshold in checks:
        if not math.isfinite(value):
            failures.append(f"{name}=missing")
        elif op == ">=" and value < threshold:
            failures.append(f"{name} {value:.4f} < {threshold:.4f}")
        elif op == "<=" and value > threshold:
            failures.append(f"{name} {value:.4f} > {threshold:.4f}")
    return not failures, failures


def minmax_norm(value: float, values: list[float], higher_is_better: bool = True) -> float:
    clean = [item for item in values if math.isfinite(item)]
    if not math.isfinite(value):
        return 0.0 if higher_is_better else 1.0
    if len(clean) <= 1:
        return 1.0
    lo, hi = min(clean), max(clean)
    if abs(hi - lo) < 1e-12:
        return 1.0
    norm = (value - lo) / (hi - lo)
    return norm if higher_is_better else 1.0 - norm


def assign_composites(rows: list[dict[str, Any]], baseline: dict[str, float] | None) -> dict[str, Any]:
    values = {metric: [safe_float(row.get(f"valid_{metric}")) for row in rows] for metric in ["AP", "F1", "PPL30", "mean_L2", "L2gt30"]}
    any_hard = False
    for row in rows:
        passes, failures = hard_filter(row, baseline)
        row["passes_hard_filter"] = passes
        row["hard_filter_failures"] = ";".join(failures)
        any_hard = any_hard or passes
        composite = (
            0.20 * minmax_norm(safe_float(row.get("valid_AP")), values["AP"], True)
            + 0.25 * minmax_norm(safe_float(row.get("valid_F1")), values["F1"], True)
            + 0.20 * minmax_norm(safe_float(row.get("valid_PPL30")), values["PPL30"], True)
            - 0.20 * minmax_norm(safe_float(row.get("valid_mean_L2")), values["mean_L2"], False)
            - 0.15 * minmax_norm(safe_float(row.get("valid_L2gt30")), values["L2gt30"], False)
        )
        row["valid_composite"] = composite
    candidates = [row for row in rows if row["passes_hard_filter"]] or rows
    selected = max(candidates, key=lambda item: safe_float(item.get("valid_composite")))
    return {
        "main_checkpoint": selected["checkpoint_name"],
        "no_checkpoint_passed_hard_filter": not any_hard,
    }


def metric_bests(rows: list[dict[str, Any]], main_name: str) -> list[dict[str, Any]]:
    specs = [
        ("AP", "max"),
        ("AP50", "max"),
        ("F1", "max"),
        ("pair", "max"),
        ("global_visible_recall", "max"),
        ("mean_L2", "min"),
        ("median_L2", "min"),
        ("p90_L2", "min"),
        ("PPL30", "max"),
        ("PPL50", "max"),
        ("L2gt30", "min"),
        ("L2gt50", "min"),
    ]
    out = []
    for split in ["valid", "test"]:
        for metric, direction in specs:
            field = f"{split}_{metric}"
            valid_rows = [row for row in rows if math.isfinite(safe_float(row.get(field)))]
            if not valid_rows:
                out.append(
                    {
                        "metric_name": metric,
                        "split": split,
                        "direction": direction,
                        "best_checkpoint": "missing",
                        "best_value": "",
                        "main_checkpoint_value": "",
                        "delta_from_main": "",
                        "can_be_used_as_paper_main_result": False,
                    }
                )
                continue
            best = (max if direction == "max" else min)(valid_rows, key=lambda row: safe_float(row.get(field)))
            main = next((row for row in rows if row["checkpoint_name"] == main_name), rows[0])
            best_value = safe_float(best.get(field))
            main_value = safe_float(main.get(field))
            delta = best_value - main_value if math.isfinite(best_value) and math.isfinite(main_value) else float("nan")
            out.append(
                {
                    "metric_name": metric,
                    "split": split,
                    "direction": direction,
                    "best_checkpoint": best["checkpoint_name"],
                    "best_value": best_value,
                    "main_checkpoint_value": main_value,
                    "delta_from_main": delta,
                    "can_be_used_as_paper_main_result": best["checkpoint_name"] == main_name,
                }
            )
    return out


def update_summary(summary_path: Path, summary: dict[str, Any], main_name: str, no_hard: bool, dry_run: bool) -> None:
    summary["checkpoint_selection"] = {
        "main_checkpoint": main_name,
        "selection_basis": "valid_composite",
        "paper_main_result_allowed": True,
        "metric_bests_are_tradeoff_only": True,
        "no_metric_stitching": True,
        "no_checkpoint_passed_hard_filter": no_hard,
    }
    if not dry_run:
        write_json(summary_path, summary)


def append_comparison_note(report_dir: Path, dry_run: bool) -> None:
    path = report_dir / "comparison_report_zh.md"
    if not path.exists() or dry_run:
        return
    text = path.read_text(encoding="utf-8")
    note = (
        "\n\n## Checkpoint 选择口径\n\n"
        "本报告主结果来自同一个 checkpoint；不同 checkpoint 的单项最优仅用于分析多任务权衡，"
        "不用于论文主表拼接。主 checkpoint 选择只基于 valid 指标，test 仅用于最终评估。\n"
    )
    if "不同 checkpoint 的单项最优仅用于分析多任务权衡" not in text:
        path.write_text(text.rstrip() + note, encoding="utf-8")


def write_markdown_reports(
    report_dir: Path,
    run_dir: Path,
    rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    main_name: str,
    no_hard: bool,
    dry_run: bool,
) -> None:
    def table(headers: list[str], body: list[list[Any]]) -> list[str]:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in body:
            lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
        return lines

    best_ap = next((row["best_checkpoint"] for row in best_rows if row["split"] == "valid" and row["metric_name"] == "AP"), "")
    best_l2 = next((row["best_checkpoint"] for row in best_rows if row["split"] == "valid" and row["metric_name"] == "mean_L2"), "")
    best_f1 = next((row["best_checkpoint"] for row in best_rows if row["split"] == "valid" and row["metric_name"] == "F1"), "")
    lines = [
        "# Unified Checkpoint Selection Report",
        "",
        "## 主 checkpoint 选择结果",
        f"- Main selected checkpoint: `{main_name}`",
        f"- Run dir: `{run_dir}`",
        f"- Selection basis: `valid_composite`",
        f"- No checkpoint passed hard filter: `{no_hard}`",
        "",
        "## valid-based selection",
        "主 checkpoint 选择只使用 valid 指标。test 指标只用于最终评估，不允许用 test 反选 checkpoint。",
        "",
        "## 不允许指标拼接",
        "正式论文主表必须使用 main_selected checkpoint 的全部 test 指标；不同 checkpoint 的单项 best 只能作为补充分析或消融讨论，不能拼接成论文主表。",
        "",
        "## trade-off 分析",
        f"- best AP checkpoint: `{best_ap}`",
        f"- best F1 checkpoint: `{best_f1}`",
        f"- best point mean L2 checkpoint: `{best_l2}`",
        f"- best AP 与 best point checkpoint 是否一致: `{best_ap == best_l2}`",
        f"- best F1 与 best point checkpoint 是否一致: `{best_f1 == best_l2}`",
        f"- best composite 是否折中结果: `{main_name not in {best_ap, best_f1, best_l2}}`",
        "",
        "## 推荐论文报告口径",
        f"- Main table: use `{main_name}` only.",
        "- Appendix/trade-off table: can report checkpoint_metric_bests.csv and checkpoint_all_metrics.csv.",
        "- Do not report stitched metrics assembled from different checkpoints.",
        "",
        "## Checkpoint Metrics",
    ]
    body = [
        [
            row["checkpoint_name"],
            row["selection_role"],
            row.get("valid_AP"),
            row.get("valid_F1"),
            row.get("valid_pair"),
            row.get("valid_mean_L2"),
            row.get("valid_PPL30"),
            row.get("valid_composite"),
            row.get("passes_hard_filter"),
        ]
        for row in rows
    ]
    lines.extend(table(["checkpoint", "role", "valid_AP", "valid_F1", "valid_pair", "valid_mean_L2", "valid_PPL30", "valid_composite", "hard"], body))
    if not dry_run:
        (report_dir / "checkpoint_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    trade_lines = [
        "# Checkpoint Trade-off Notes",
        "",
        "- Metric-specific best checkpoints are trade-off analysis only.",
        "- Paper main result must use the same main_selected checkpoint across all metrics.",
        f"- Main selected checkpoint: `{main_name}`",
        f"- best_AP_vs_best_L2_conflict: `{best_ap != best_l2}`",
        f"- best_F1_vs_best_L2_conflict: `{best_f1 != best_l2}`",
    ]
    if not dry_run:
        (report_dir / "checkpoint_tradeoff_notes.md").write_text("\n".join(trade_lines) + "\n", encoding="utf-8")


def standardize_one(run_dir: Path, report_dir: Path, baseline_summary: Path | None, dry_run: bool = False) -> dict[str, Any]:
    summary_path = report_dir / "summary.json"
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        payload = {
            "run_dir": str(run_dir),
            "report_dir": str(report_dir),
            "status": "missing_summary",
            "decision": "blocked_by_missing_metrics",
        }
        if not dry_run:
            write_json(report_dir / "checkpoint_selection_summary.json", payload)
        return payload

    rows = scan_checkpoint_rows(run_dir, report_dir, summary)
    baseline = load_baseline_valid(baseline_summary)
    selection = assign_composites(rows, baseline)
    main_name = selection["main_checkpoint"]
    best_rows = metric_bests(rows, main_name)
    best_by_metric = {f"{row['split']}_{row['metric_name']}": row["best_checkpoint"] for row in best_rows}
    for row in rows:
        row["selection_role"] = role_for_checkpoint(row["checkpoint_name"], main_name, best_by_metric)
    no_hard = bool(selection["no_checkpoint_passed_hard_filter"])

    if not dry_run:
        write_csv(report_dir / "checkpoint_all_metrics.csv", rows, CSV_FIELDS)
        write_csv(
            report_dir / "checkpoint_metric_bests.csv",
            best_rows,
            [
                "metric_name",
                "split",
                "direction",
                "best_checkpoint",
                "best_value",
                "main_checkpoint_value",
                "delta_from_main",
                "can_be_used_as_paper_main_result",
            ],
        )
        write_json(
            report_dir / "checkpoint_selection_summary.json",
            {
                "run_dir": str(run_dir),
                "report_dir": str(report_dir),
                "main_checkpoint": main_name,
                "selection_basis": "valid_composite",
                "no_checkpoint_passed_hard_filter": no_hard,
                "paper_main_result_allowed": True,
                "metric_bests_are_tradeoff_only": True,
                "no_metric_stitching": True,
                "missing_metric_rows": [
                    {"checkpoint_name": row["checkpoint_name"], "missing_metrics": row.get("missing_metrics", "")}
                    for row in rows
                    if row.get("missing_metrics")
                ],
            },
        )
    write_markdown_reports(report_dir, run_dir, rows, best_rows, main_name, no_hard, dry_run)
    update_summary(summary_path, summary, main_name, no_hard, dry_run)
    append_comparison_note(report_dir, dry_run)
    return {
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "status": "ok",
        "main_checkpoint": main_name,
        "no_checkpoint_passed_hard_filter": no_hard,
        "checkpoint_count": len(rows),
        "missing_metric_rows": sum(1 for row in rows if row.get("missing_metrics")),
    }


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or (args.run_dir / "report")
    result = standardize_one(args.run_dir, report_dir, args.baseline_summary, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
