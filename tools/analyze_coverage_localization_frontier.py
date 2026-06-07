from __future__ import annotations

from upper_bound_diagnostics_utils import (
    OUT_DIR,
    discover_models,
    ensure_out_dir,
    load_records,
    md_table,
    threshold_metrics,
    write_csv,
    write_json,
)


def is_dominated(row: dict, others: list[dict]) -> bool:
    for other in others:
        if other is row:
            continue
        no_worse = (
            other["F1"] >= row["F1"]
            and other["pair"] >= row["pair"]
            and other["PPL-SR@30"] >= row["PPL-SR@30"]
            and other["mean_L2"] <= row["mean_L2"]
        )
        strictly = (
            other["F1"] > row["F1"]
            or other["pair"] > row["pair"]
            or other["PPL-SR@30"] > row["PPL-SR@30"]
            or other["mean_L2"] < row["mean_L2"]
        )
        if no_worse and strictly:
            return True
    return False


def main() -> None:
    ensure_out_dir()
    models, missing = discover_models()
    thresholds = [round(v / 100.0, 2) for v in range(5, 100, 5)]
    rows = []
    best_rows = []
    for name, info in models.items():
        if not info.get("has_test_records"):
            continue
        records = load_records(info["test_records_path"])
        model_rows = []
        for th in thresholds:
            row = {"model": name, **threshold_metrics(records, th)}
            rows.append(row)
            model_rows.append(row)
        if not model_rows:
            continue
        best_rows.append({"model": name, "criterion": "best_by_f1", **max(model_rows, key=lambda r: (r["F1"], r["pair"], r["PPL-SR@30"]))})
        positive_l2 = [r for r in model_rows if r["pair"] > 0]
        if positive_l2:
            best_rows.append({"model": name, "criterion": "best_by_mean_l2", **min(positive_l2, key=lambda r: (r["mean_L2"], -r["pair"]))})
            best_rows.append({"model": name, "criterion": "best_by_ppl30", **max(positive_l2, key=lambda r: (r["PPL-SR@30"], r["pair"], r["F1"]))})

    pareto = [dict(row, pareto=True) for row in rows if not is_dominated(row, rows)]
    for row in rows:
        row["pareto"] = not is_dominated(row, rows)
    write_csv(OUT_DIR / "coverage_localization_frontier.csv", rows)
    write_csv(OUT_DIR / "coverage_localization_best_points.csv", best_rows)
    write_json(
        OUT_DIR / "coverage_localization_frontier.json",
        {"frontier": rows, "best_points": best_rows, "pareto_points": pareto, "missing_inputs": missing},
    )

    ema_default = next((r for r in rows if r["model"] == "EMA_BIFPN" and abs(r["threshold"] - 0.5) < 1e-9), None)
    dominating_ema = []
    if ema_default:
        for row in rows:
            if (
                row["pair"] >= ema_default["pair"]
                and row["F1"] >= ema_default["F1"]
                and row["mean_L2"] <= ema_default["mean_L2"]
                and row["PPL-SR@30"] >= ema_default["PPL-SR@30"]
                and row is not ema_default
            ):
                dominating_ema.append(row)

    md = [
        "# Coverage-Localization Frontier",
        "",
        "Threshold sweep over visible_score / has_picking_score from 0.05 to 0.95. This is threshold-only analysis from existing records.",
        "",
        "## Best Points",
        md_table(best_rows, ["model", "criterion", "threshold", "F1", "pair", "global_visible_recall", "mean_L2", "PPL-SR@30", "PPL-SR@50", "L2>30_count"]),
        "",
        "## Pareto Points",
        md_table(pareto[:80], ["model", "threshold", "F1", "pair", "global_visible_recall", "mean_L2", "PPL-SR@30", "L2>30_count"]),
        "",
        "## EMA_BIFPN Dominance Check",
    ]
    if dominating_ema:
        md.append("Some threshold points are non-worse than EMA_BIFPN@0.5 on F1/pair/mean_L2/PPL@30:")
        md.append(md_table(dominating_ema, ["model", "threshold", "F1", "pair", "mean_L2", "PPL-SR@30", "L2>30_count"]))
    else:
        md.append("No threshold-only point simultaneously improves EMA_BIFPN@0.5 on F1, pair, mean_L2, and PPL@30.")
    if missing:
        md.extend(["", "## Missing Inputs", *[f"- {m}" for m in missing]])
    (OUT_DIR / "coverage_localization_frontier.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

