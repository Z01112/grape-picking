from __future__ import annotations

from pathlib import Path

from upper_bound_diagnostics_utils import OUT_DIR, INDEX_DIR, ensure_out_dir, md_table, read_json, write_json


def get_model(items: list[dict], name: str) -> dict:
    for item in items:
        if item.get("model") == name:
            return item
    return {}


def main() -> None:
    ensure_out_dir()
    oracle = read_json(OUT_DIR / "oracle_decomposition.json", default={"models": []})
    toproi = read_json(OUT_DIR / "toproi_coverage_audit.json", default={"summary": []})
    frontier = read_json(OUT_DIR / "coverage_localization_frontier.json", default={"frontier": [], "pareto_points": []})
    offset = read_json(OUT_DIR / "single_offset_limit.json", default={})
    review = read_json(OUT_DIR / "annotation_ambiguity_review_manifest.json", default={})
    missing_inputs = []
    for item in [oracle, frontier, offset, review]:
        missing_inputs.extend(item.get("missing_inputs", []))

    ema_oracle = get_model(oracle.get("models", []), "EMA_BIFPN")
    current = ema_oracle.get("current_default", {})
    gt_has = ema_oracle.get("gt_has_oracle", {})
    gt_box = ema_oracle.get("gt_box_oracle", {})
    cand = ema_oracle.get("candidate_pool_oracle_iou50", {})
    top_all = next((r for r in toproi.get("summary", []) if r.get("group") == "all"), {})
    offset_pred = offset.get("prediction_error", {})
    gt_dist = offset.get("gt_offset_distribution", {})

    current_pair = current.get("pair", 0)
    current_l2 = current.get("mean_L2", 0.0)
    has_pair_gain = gt_has.get("pair", 0) - current_pair
    gt_box_l2_gain = current_l2 - gt_box.get("mean_L2", current_l2)
    cand_l2_gain = current_l2 - cand.get("mean_L2", current_l2)
    in_roi_ratio = top_all.get("in_roi_ratio", 0.0)
    relaxed_gain = top_all.get("relaxed_in_roi_ratio", 0.0) - in_roi_ratio
    high_iou_bad = offset_pred.get("high_iou_point_bad_count", 0)
    low_iou_bad = offset_pred.get("low_iou_propagation_count", 0)
    toproi_out_ratio = gt_dist.get("toproi_out_ratio", 0.0)

    bottlenecks = []
    bottlenecks.append({"bottleneck": "visibility_has_picking", "evidence_score": has_pair_gain, "evidence": f"GT-has oracle adds {has_pair_gain} pairs over current."})
    bottlenecks.append({"bottleneck": "bbox_propagation", "evidence_score": gt_box_l2_gain, "evidence": f"GT-box oracle mean L2 gain is {gt_box_l2_gain:.3f}px."})
    bottlenecks.append({"bottleneck": "candidate_selection", "evidence_score": cand_l2_gain, "evidence": f"IoU50 candidate oracle mean L2 gain is {cand_l2_gain:.3f}px."})
    bottlenecks.append({"bottleneck": "toproi_geometry", "evidence_score": toproi_out_ratio + relaxed_gain, "evidence": f"Current TopROI out ratio is {toproi_out_ratio:.3f}; relaxed ROI gain is {relaxed_gain:.3f}."})
    bottlenecks.append({"bottleneck": "single_offset_point_representation", "evidence_score": high_iou_bad, "evidence": f"High-IoU point-bad cases: {high_iou_bad}; low-IoU propagation cases: {low_iou_bad}."})
    bottlenecks = sorted(bottlenecks, key=lambda x: x["evidence_score"], reverse=True)

    if in_roi_ratio >= 0.85 and high_iou_bad > 0 and (cand_l2_gain >= 0.5 or high_iou_bad >= low_iou_bad):
        route = "ROI_HEATMAP_INTEGRAL_V1"
        route_reason = "Current TopROI covers most GT points, while high-IoU point errors remain; this points to coordinate representation rather than detector boxes."
    elif relaxed_gain >= 0.08:
        route = "RELAXED_ROI_HEATMAP_V1"
        route_reason = "Relaxed ROI substantially improves GT point coverage, so current TopROI is too narrow for a heatmap/integral coordinate route."
    elif review.get("count", 0) >= 100 and toproi_out_ratio >= 0.15:
        route = "STEM_STRUCTURAL_SUPERVISION_V1"
        route_reason = "The review set is large and TopROI misses are non-trivial; structural stem cues should be prioritized after manual label review."
    elif cand_l2_gain < 0.3 and gt_box_l2_gain < 0.3 and high_iou_bad == 0:
        route = "STOP_OFFSET_FAMILY"
        route_reason = "Oracles show limited usable signal for box, has, candidate, or high-IoU point improvements."
    else:
        route = "ROI_HEATMAP_INTEGRAL_V1"
        route_reason = "The remaining failure profile is dominated by point localization under usable instance matches; offset-family tweaks have repeatedly failed."

    decision = {
        "recommended_next_unique_experiment": route,
        "route_reason": route_reason,
        "bottleneck_ranking": bottlenecks,
        "key_evidence": {
            "ema_current_pair": current_pair,
            "ema_current_mean_l2": current_l2,
            "gt_has_pair_gain": has_pair_gain,
            "gt_box_l2_gain": gt_box_l2_gain,
            "candidate_iou50_l2_gain": cand_l2_gain,
            "current_toproi_in_roi_ratio": in_roi_ratio,
            "relaxed_roi_gain": relaxed_gain,
            "high_iou_point_bad_count": high_iou_bad,
            "low_iou_propagation_count": low_iou_bad,
        },
        "already_rejected_routes": [
            "RELCAL/reliability scalar",
            "PAM point-aware matcher cost02 fair100",
            "PAR point-aware rerank",
            "POINT_LSD",
            "grouped query inference",
            "point-only refine",
            "point-feature refine hasdistill",
            "C2F/HRPB/QDPT/DPO/PCGrad as current mainline routes",
            "CADA full fair100 as a direct main model due to coverage loss",
        ],
        "not_yet_genuinely_tried": [
            "ROI heatmap/integral coordinate head with formally validated ROI geometry",
            "stem structural supervision after label review",
            "PALD external teacher with available teacher features and non-zero smoke loss",
        ],
        "missing_inputs": sorted(set(missing_inputs)),
    }
    write_json(OUT_DIR / "next_modeling_decision.json", decision)
    md = [
        "# Next Modeling Decision From Upper-Bound Diagnostics",
        "",
        f"Decision: **{route}**",
        "",
        route_reason,
        "",
        "## Bottleneck Ranking",
        md_table(bottlenecks, ["bottleneck", "evidence_score", "evidence"]),
        "",
        "## Rejected Routes",
        *[f"- {r}" for r in decision["already_rejected_routes"]],
        "",
        "## Not Yet Genuinely Tried",
        *[f"- {r}" for r in decision["not_yet_genuinely_tried"]],
    ]
    (OUT_DIR / "next_modeling_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    report = [
        "# Upper-Bound Diagnostics Report",
        "",
        "Scope: existing records and annotations only. No training, no checkpoint evaluation, no model/criterion/matcher/postprocessor changes.",
        "",
        "## Main Answers",
        f"1. Single-offset route local upper bound: {'likely reached' if route in ['ROI_HEATMAP_INTEGRAL_V1', 'RELAXED_ROI_HEATMAP_V1', 'STOP_OFFSET_FAMILY'] else 'not fully proven'} under current repeated failed offset-family probes.",
        f"2. Largest bottleneck ranking: {', '.join(b['bottleneck'] for b in bottlenecks[:3])}.",
        "3. CADA fair100 became high-precision low-coverage because its retained visible pairs improved L2/PPL while F1 and pair_count fell sharply, indicating coverage-localization trade-off rather than a deployable mainline gain.",
        "4. HRPB/C2F did not improve because post-offset/TopROI refinements did not solve the core instance-bound coordinate-generation bottleneck and in some runs worsened y/localization stability.",
        "5. Threshold-only Pareto evidence is in coverage_localization_frontier.md; use it to decide whether HP protocol is only a reporting protocol or a real model gain.",
        f"6. Modeling paradigm switch evidence: {route_reason}",
        f"7. Recommended next route: {route}.",
        f"8. ROI choice: {'current TopROI' if route == 'ROI_HEATMAP_INTEGRAL_V1' else 'relaxed ROI' if route == 'RELAXED_ROI_HEATMAP_V1' else 'defer until review'} based on coverage audit.",
        "9. Stem supervision needs a stem_aux visible segment annotation tied to the grape instance; do not label all branches.",
        "10. PALD needs an external teacher checkpoint/features and a smoke test proving stable non-zero distillation loss before any training.",
        "",
        "## Key Evidence",
        md_table([decision["key_evidence"]], list(decision["key_evidence"].keys())),
    ]
    (OUT_DIR / "upper_bound_diagnostics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT_DIR / "upper_bound_diagnostics_summary.json", decision)
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Upper-Bound Diagnostics V1",
                "",
                "Read-only diagnostic bundle for GPPoint-DETR picking point upper-bound analysis.",
                "",
                "Generated artifacts:",
                "- oracle_decomposition.*",
                "- toproi_coverage_audit.*",
                "- coverage_localization_frontier.*",
                "- single_offset_limit.*",
                "- annotation_ambiguity_review_set.*",
                "- next_modeling_decision.*",
                "- upper_bound_diagnostics_report.md",
                "",
                "No training or checkpoint generation was performed by these scripts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    missing_md = ["# Missing Inputs", ""]
    if decision["missing_inputs"]:
        missing_md.extend(f"- {m}" for m in decision["missing_inputs"])
    else:
        missing_md.append("No blocking missing inputs for completed diagnostics.")
    (OUT_DIR / "missing_inputs.md").write_text("\n".join(missing_md) + "\n", encoding="utf-8")
    (INDEX_DIR / "upper_bound_diagnostics_v1_index.md").write_text(
        f"# Upper-Bound Diagnostics V1 Index\n\n- Output directory: `{OUT_DIR.relative_to(OUT_DIR.parents[2])}`\n- Decision: `{route}`\n- Main report: `{(OUT_DIR / 'upper_bound_diagnostics_report.md').relative_to(OUT_DIR.parents[2])}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

