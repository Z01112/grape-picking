from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grape_point_eval_utils import compute_unified_point_metrics
from tools.make_grape_point_report import evaluate_split, summarize_split_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze continue20 coverage/localization metric trajectory.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ema-summary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-test", action="store_true")
    return parser.parse_args()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def l2_fail_count(pair_count: int, ppl_sr: float) -> int:
    if pair_count <= 0 or not math.isfinite(ppl_sr):
        return 0
    return int(pair_count - round(pair_count * ppl_sr))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_split_summary(payload: dict[str, Any], split: str) -> dict[str, Any]:
    split_summary = payload.get("primary_checkpoint_split_summary", {})
    if isinstance(split_summary, dict) and isinstance(split_summary.get(split), dict):
        return split_summary[split]
    if isinstance(payload.get(split), dict):
        return payload[split]
    return {}


def extract_metrics(payload: dict[str, Any], split: str = "test") -> dict[str, Any] | None:
    split_summary = _get_split_summary(payload, split)
    if not split_summary:
        return None

    det = split_summary.get("grape_detection", {})
    has = split_summary.get("has_picking", {})
    point = split_summary.get("picking_point", {})
    unified = payload.get("unified_point_metrics", {}).get(split, {})
    global_chain = unified.get("global_chain", {}) if isinstance(unified, dict) else {}
    point_unified = unified.get("point", {}) if isinstance(unified, dict) else {}

    pair = int(point.get("pair_count", point_unified.get("point_pair_count", 0)) or 0)
    ppl30 = safe_float(point.get("ppl_sr_30", point_unified.get("ppl_sr_30")))
    ppl50 = safe_float(point.get("ppl_sr_50", point_unified.get("ppl_sr_50")))
    return {
        "AP": safe_float(det.get("AP")),
        "AP50": safe_float(det.get("AP50")),
        "instance_f1": safe_float(has.get("f1")),
        "pair_count": pair,
        "global_visible_recall": safe_float(global_chain.get("global_visible_recall")),
        "mean_L2": safe_float(point.get("mean_l2_px", point_unified.get("point_mean_l2_px"))),
        "median_L2": safe_float(point.get("median_l2_px", point_unified.get("point_median_l2_px"))),
        "p90_L2": safe_float(point.get("p90_l2_px", point_unified.get("point_p90_l2_px"))),
        "PPL-SR@30": ppl30,
        "PPL-SR@50": ppl50,
        "L2>30_count": l2_fail_count(pair, ppl30),
        "L2>50_count": l2_fail_count(pair, ppl50),
    }


def checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint(\d+)\.pth", path.name)
    if match:
        return int(match.group(1))
    if path.name == "last.pth":
        return 19
    return None


def find_trace_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    by_epoch: dict[int, Path] = {}
    for path in sorted(run_dir.glob("checkpoint*.pth")):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            by_epoch[epoch] = path.resolve()
    last = run_dir / "last.pth"
    if last.exists():
        epoch = checkpoint_epoch(last)
        if epoch is not None and epoch not in by_epoch:
            by_epoch[epoch] = last.resolve()
    items = sorted(by_epoch.items(), key=lambda item: item[0])
    return items


def split_metrics_from_records(stats: dict[str, Any], records: list[dict], has_threshold: float = 0.5) -> dict[str, Any]:
    error = summarize_split_error(records, has_threshold)
    unified = compute_unified_point_metrics(
        records,
        iou_threshold=0.5,
        has_picking_threshold=has_threshold,
        visibility_score_key="visible_score",
    )
    det_stats = stats.get("coco_eval_bbox", [])
    det = {
        "AP": safe_float(det_stats[0] if len(det_stats) > 0 else None),
        "AP50": safe_float(det_stats[1] if len(det_stats) > 1 else None),
    }
    instance_chain = unified.get("instance_chain", {})
    global_chain = unified.get("global_chain", {})
    point = unified.get("point", {})
    pair = int(point.get("point_pair_count", error.get("point_pair_count", 0)) or 0)
    ppl30 = safe_float(point.get("ppl_sr_30", error.get("ppl_sr_30")))
    ppl50 = safe_float(point.get("ppl_sr_50", error.get("ppl_sr_50")))
    return {
        "AP": det["AP"],
        "AP50": det["AP50"],
        "F1": safe_float(instance_chain.get("instance_visible_f1")),
        "pair": pair,
        "global_visible_recall": safe_float(global_chain.get("global_visible_recall")),
        "mean_L2": safe_float(point.get("point_mean_l2_px", error.get("mean_l2_px"))),
        "median_L2": safe_float(point.get("point_median_l2_px", error.get("median_l2_px"))),
        "p90_L2": safe_float(point.get("point_p90_l2_px", error.get("p90_l2_px"))),
        "PPL-SR@30": ppl30,
        "PPL-SR@50": ppl50,
        "L2>30_count": l2_fail_count(pair, ppl30),
        "L2>50_count": l2_fail_count(pair, ppl50),
    }


def evaluate_checkpoint(
    config: Path,
    checkpoint: Path,
    split: str,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, Any]:
    stats, records = evaluate_split(
        config,
        checkpoint,
        split,
        dataset_root,
        batch_size,
        num_workers,
        device,
        collect_predictions=True,
    )
    metrics = split_metrics_from_records(stats, records)
    metrics["records_count"] = len(records)
    return metrics


def ema_split_reference(summary_path: Path, split: str) -> dict[str, Any]:
    payload = load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Cannot load EMA summary: {summary_path}")
    if split == "test":
        metrics = extract_metrics(payload)
        if metrics is None:
            raise ValueError(f"Cannot extract EMA test metrics: {summary_path}")
        return {
            "AP": safe_float(metrics.get("AP")),
            "AP50": safe_float(metrics.get("AP50")),
            "F1": safe_float(metrics.get("instance_f1")),
            "pair": int(metrics.get("pair_count", 0) or 0),
            "global_visible_recall": safe_float(metrics.get("global_visible_recall")),
            "mean_L2": safe_float(metrics.get("mean_L2")),
            "median_L2": safe_float(metrics.get("median_L2")),
            "p90_L2": safe_float(metrics.get("p90_L2")),
            "PPL-SR@30": safe_float(metrics.get("PPL-SR@30")),
            "PPL-SR@50": safe_float(metrics.get("PPL-SR@50")),
            "L2>30_count": metrics.get("L2>30_count"),
            "L2>50_count": metrics.get("L2>50_count"),
        }
    split_summary = payload.get("primary_checkpoint_split_summary", {}).get(split, {})
    det = split_summary.get("grape_detection", {})
    has = split_summary.get("has_picking", {})
    point = split_summary.get("picking_point", {})
    unified = payload.get("unified_point_metrics", {}).get(split, {})
    global_chain = unified.get("global_chain", {}) if isinstance(unified, dict) else {}
    pair = int(point.get("pair_count", 0) or 0)
    ppl30 = safe_float(point.get("ppl_sr_30"))
    ppl50 = safe_float(point.get("ppl_sr_50"))
    return {
        "AP": safe_float(det.get("AP")),
        "AP50": safe_float(det.get("AP50")),
        "F1": safe_float(has.get("f1")),
        "pair": pair,
        "global_visible_recall": safe_float(global_chain.get("global_visible_recall")),
        "mean_L2": safe_float(point.get("mean_l2_px")),
        "median_L2": safe_float(point.get("median_l2_px")),
        "p90_L2": safe_float(point.get("p90_l2_px")),
        "PPL-SR@30": ppl30,
        "PPL-SR@50": ppl50,
        "L2>30_count": l2_fail_count(pair, ppl30),
        "L2>50_count": l2_fail_count(pair, ppl50),
    }


def pareto_pass(row: dict[str, Any], ref: dict[str, Any]) -> bool:
    return (
        safe_float(row["F1"]) >= safe_float(ref["F1"])
        and int(row["pair"]) >= int(ref["pair"])
        and safe_float(row["mean_L2"]) <= safe_float(ref["mean_L2"])
        and safe_float(row["PPL-SR@30"]) >= safe_float(ref["PPL-SR@30"])
        and int(row["L2>30_count"]) <= int(ref["L2>30_count"])
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "epoch",
        "AP",
        "AP50",
        "F1",
        "pair",
        "global_visible_recall",
        "mean_L2",
        "median_L2",
        "p90_L2",
        "PPL-SR@30",
        "PPL-SR@50",
        "L2>30_count",
        "L2>50_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    value = safe_float(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], split: str, ref: dict[str, Any], pareto_rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Continue20 Metric Trajectory ({split})",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Epoch selection is based on valid metrics; test metrics are reference only.",
        "",
        "## EMA_BIFPN Reference",
        "",
        "| AP | AP50 | F1 | pair | mean_L2 | p90_L2 | PPL-SR@30 | PPL-SR@50 | L2>30_count |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(ref['AP'])} | {fmt(ref['AP50'])} | {fmt(ref['F1'])} | {ref['pair']} | {fmt(ref['mean_L2'])} | {fmt(ref['p90_L2'])} | {fmt(ref['PPL-SR@30'])} | {fmt(ref['PPL-SR@50'])} | {ref['L2>30_count']} |",
        "",
        "## Trajectory",
        "",
        "| epoch | AP | AP50 | F1 | pair | global_visible_recall | mean_L2 | median_L2 | p90_L2 | PPL-SR@30 | PPL-SR@50 | L2>30 | L2>50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {fmt(row['AP'])} | {fmt(row['AP50'])} | {fmt(row['F1'])} | {row['pair']} | {fmt(row['global_visible_recall'])} | {fmt(row['mean_L2'])} | {fmt(row['median_L2'])} | {fmt(row['p90_L2'])} | {fmt(row['PPL-SR@30'])} | {fmt(row['PPL-SR@50'])} | {row['L2>30_count']} | {row['L2>50_count']} |"
        )
    lines.extend(["", "## Pareto Result", ""])
    if pareto_rows:
        lines.append("Pareto checkpoint exists under the defined valid-metric gate.")
        for row in pareto_rows:
            lines.append(f"- epoch {row['epoch']}: F1={fmt(row['F1'])}, pair={row['pair']}, mean_L2={fmt(row['mean_L2'])}, PPL-SR@30={fmt(row['PPL-SR@30'])}, L2>30={row['L2>30_count']}")
    else:
        lines.append("No epoch simultaneously satisfies F1/pair/mean_L2/PPL-SR@30/L2>30 against EMA_BIFPN.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tradeoff_label(rows: list[dict[str, Any]], ref: dict[str, Any]) -> str:
    coverage_improves = any(safe_float(row["F1"]) >= safe_float(ref["F1"]) and int(row["pair"]) >= int(ref["pair"]) for row in rows)
    localization_improves = any(safe_float(row["mean_L2"]) <= safe_float(ref["mean_L2"]) and safe_float(row["PPL-SR@30"]) >= safe_float(ref["PPL-SR@30"]) for row in rows)
    both = any(pareto_pass(row, ref) for row in rows)
    if both:
        return "continue_training_has_pareto_checkpoint"
    if coverage_improves and not localization_improves:
        return "coverage_improves_but_localization_worsens"
    if localization_improves and not coverage_improves:
        return "localization_improves_but_coverage_drops"
    if coverage_improves and localization_improves:
        return "coverage_and_localization_improve_in_different_epochs"
    return "both_fail"


def monotonic_worsening(rows: list[dict[str, Any]], key: str) -> bool:
    values = [safe_float(row[key]) for row in rows if math.isfinite(safe_float(row[key]))]
    if len(values) < 2:
        return False
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = find_trace_checkpoints(args.run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No trace checkpoints found in {args.run_dir}")

    valid_ref = ema_split_reference(args.ema_summary, "valid")
    test_ref = ema_split_reference(args.ema_summary, "test")

    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for epoch, ckpt in checkpoints:
        print(f"[trace] evaluating valid epoch={epoch} checkpoint={ckpt.name}")
        valid_metrics = evaluate_checkpoint(args.config, ckpt, "valid", args.dataset_root, args.batch_size, args.num_workers, args.device)
        valid_metrics["epoch"] = epoch
        valid_metrics["checkpoint"] = str(ckpt)
        valid_rows.append(valid_metrics)
        if args.eval_test:
            print(f"[trace] evaluating test epoch={epoch} checkpoint={ckpt.name}")
            test_metrics = evaluate_checkpoint(args.config, ckpt, "test", args.dataset_root, args.batch_size, args.num_workers, args.device)
            test_metrics["epoch"] = epoch
            test_metrics["checkpoint"] = str(ckpt)
            test_rows.append(test_metrics)

    valid_rows.sort(key=lambda row: int(row["epoch"]))
    test_rows.sort(key=lambda row: int(row["epoch"]))
    valid_pareto = [row for row in valid_rows if pareto_pass(row, valid_ref)]
    test_pareto = [row for row in test_rows if pareto_pass(row, test_ref)]
    decision = tradeoff_label(valid_rows, valid_ref)

    write_csv(args.output_dir / "continue20_metric_trajectory.csv", valid_rows)
    write_markdown(args.output_dir / "continue20_metric_trajectory.md", valid_rows, "valid", valid_ref, valid_pareto)
    if test_rows:
        write_csv(args.output_dir / "continue20_metric_trajectory_test.csv", test_rows)
        write_markdown(args.output_dir / "continue20_metric_trajectory_test.md", test_rows, "test", test_ref, test_pareto)

    best_valid_f1 = max(valid_rows, key=lambda row: safe_float(row["F1"]))
    best_valid_l2 = min(valid_rows, key=lambda row: safe_float(row["mean_L2"]))
    best_valid_ppl30 = max(valid_rows, key=lambda row: safe_float(row["PPL-SR@30"]))

    decision_lines = [
        "# Continue Training Tradeoff Decision",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Decision: `{decision}`",
        "- Selection split: valid. Test is reported for reference only.",
        "",
        "## Required Answers",
        "",
        f"1. 普通 continue training 是否真实提升 visible coverage？{'是' if any(safe_float(row['F1']) >= safe_float(valid_ref['F1']) and int(row['pair']) >= int(valid_ref['pair']) for row in valid_rows) else '否'}。valid 最高 F1 epoch={best_valid_f1['epoch']}，F1={fmt(best_valid_f1['F1'])}，pair={best_valid_f1['pair']}；EMA_BIFPN valid F1={fmt(valid_ref['F1'])}，pair={valid_ref['pair']}。",
        f"2. 点精度下降是否随 epoch 单调发生？{'是' if monotonic_worsening(valid_rows, 'mean_L2') else '否'}。valid 最低 mean L2 epoch={best_valid_l2['epoch']}，mean_L2={fmt(best_valid_l2['mean_L2'])}；最高 PPL-SR@30 epoch={best_valid_ppl30['epoch']}，PPL-SR@30={fmt(best_valid_ppl30['PPL-SR@30'])}。",
        f"3. 是否存在一个中间 epoch 同时优于 EMA_BIFPN？{'是' if valid_pareto else '否'}。",
        f"4. 如果存在，是否应把该 checkpoint 作为新的主模型候选？{'存在 Pareto checkpoint，可进入 test 复核后考虑。' if valid_pareto else '不存在 valid Pareto checkpoint，不应把 continue20 中间 checkpoint 直接作为新主模型。'}",
        f"5. 如果不存在，下一步是否需要设计 coverage-preserving point refinement，而不是继续结构改造？{'是。当前轨迹显示 coverage 与 localization 没有在同一 epoch 同时达标，应优先考虑保覆盖的点精修/数据监督，而不是继续堆结构。' if not valid_pareto else '暂不需要，先复核 Pareto checkpoint 的 test 表现。'}",
        "",
        "## Best Valid Epochs",
        "",
        f"- Best F1: epoch {best_valid_f1['epoch']}, F1={fmt(best_valid_f1['F1'])}, pair={best_valid_f1['pair']}, mean_L2={fmt(best_valid_f1['mean_L2'])}, PPL-SR@30={fmt(best_valid_f1['PPL-SR@30'])}",
        f"- Best mean L2: epoch {best_valid_l2['epoch']}, F1={fmt(best_valid_l2['F1'])}, pair={best_valid_l2['pair']}, mean_L2={fmt(best_valid_l2['mean_L2'])}, PPL-SR@30={fmt(best_valid_l2['PPL-SR@30'])}",
        f"- Best PPL-SR@30: epoch {best_valid_ppl30['epoch']}, F1={fmt(best_valid_ppl30['F1'])}, pair={best_valid_ppl30['pair']}, mean_L2={fmt(best_valid_ppl30['mean_L2'])}, PPL-SR@30={fmt(best_valid_ppl30['PPL-SR@30'])}",
    ]
    if valid_pareto:
        decision_lines.extend(["", "## Valid Pareto Checkpoints", ""])
        for row in valid_pareto:
            decision_lines.append(f"- epoch {row['epoch']}: F1={fmt(row['F1'])}, pair={row['pair']}, mean_L2={fmt(row['mean_L2'])}, PPL-SR@30={fmt(row['PPL-SR@30'])}, L2>30={row['L2>30_count']}")
    if test_rows:
        decision_lines.extend(["", "## Test Reference At Same Epochs", ""])
        if test_pareto:
            decision_lines.append("Test also has Pareto checkpoints against EMA_BIFPN reference:")
            for row in test_pareto:
                decision_lines.append(f"- epoch {row['epoch']}: F1={fmt(row['F1'])}, pair={row['pair']}, mean_L2={fmt(row['mean_L2'])}, PPL-SR@30={fmt(row['PPL-SR@30'])}, L2>30={row['L2>30_count']}")
        else:
            decision_lines.append("No evaluated test epoch simultaneously beats EMA_BIFPN on F1/pair/mean_L2/PPL-SR@30/L2>30.")

    (args.output_dir / "continue_training_tradeoff_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    (args.output_dir / "continue_training_tradeoff_decision.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "decision": decision,
                "valid_reference": valid_ref,
                "test_reference": test_ref,
                "valid_pareto_epochs": valid_pareto,
                "test_pareto_epochs": test_pareto,
                "valid_rows": valid_rows,
                "test_rows": test_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'continue_training_tradeoff_decision.md'}")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
