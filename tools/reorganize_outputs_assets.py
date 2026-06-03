import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
INDEX = OUTPUTS / "_index"
CONFIG_DIR = ROOT / "configs" / "rtv4"


STANDARD_DIRS = {
    "_index": "Global indexes, path mappings, directory standards, and experiment master tables.",
    "01_mainline_results": "Current paper mainline models and important main references.",
    "02_baselines": "External baselines and early baselines used for comparison.",
    "03_unified_evaluation": "Unified evaluation outputs, threshold protocols, and comparison summaries.",
    "04_diagnostics": "Mechanism diagnostics and audits that are not failed model experiments.",
    "05_failed_experiments": "Experiments explicitly rejected from the mainline, organized by route.",
    "06_data_supervision": "Data supervision, relabel, stem_aux, ambiguous, and clean-eval assets.",
    "07_paper_assets": "Paper tables, figures, visualizations, appendix material, and result drafts.",
    "90_legacy_misc": "Historical experiments with reference value but non-standard or mixed provenance.",
    "99_uncertain_review": "Directories that need human review before classification.",
}

FAILED_SUBDIRS = [
    "01_reliability_relcal",
    "02_postprocess_rerank",
    "03_matching_pam",
    "04_point_lsd",
    "05_grouped_query",
    "06_point_refine",
    "07_selector_o2m_heatmap_simcc",
    "08_other_failed",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return path.rglob("*")


def has_any(path: Path, predicates: Iterable) -> bool:
    for p in iter_files(path):
        if p.is_file() and any(pred(p) for pred in predicates):
            return True
    return False


def count_files(path: Path, predicate=None) -> int:
    n = 0
    for p in iter_files(path):
        if p.is_file() and (predicate is None or predicate(p)):
            n += 1
    return n


def contents_summary(path: Path) -> str:
    if not path.exists():
        return "missing"
    files = [p for p in iter_files(path) if p.is_file()]
    dirs = [p for p in path.iterdir() if p.is_dir()] if path.exists() else []
    pth = sum(1 for p in files if p.suffix.lower() == ".pth")
    summaries = sum(1 for p in files if p.name == "summary.json")
    records = sum(1 for p in files if "prediction_records" in p.name)
    reports = sum(1 for p in files if p.suffix.lower() in {".md", ".csv", ".json", ".png", ".jpg", ".jpeg", ".txt", ".log"})
    return f"{len(files)} files, {len(dirs)} child dirs, {pth} checkpoints, {summaries} summary.json, {records} prediction_records-like, {reports} report/index/media files"


def audit_reference_models() -> List[Dict[str, Any]]:
    ref_root = OUTPUTS / "00_reference_models"
    rows = []
    if not ref_root.exists():
        return rows
    for d in sorted([p for p in ref_root.iterdir() if p.is_dir()]):
        has_checkpoint = has_any(d, [lambda p: p.suffix.lower() == ".pth"])
        has_config = has_any(d, [lambda p: p.suffix.lower() in {".yml", ".yaml"} or p.name.lower().startswith("config")])
        has_summary = has_any(d, [lambda p: p.name == "summary.json"])
        has_records = has_any(d, [lambda p: "prediction_records" in p.name])
        if has_checkpoint and has_config and has_summary and has_records:
            category = "01_mainline_results" if "v7" in d.name.lower() or "gppoint" in d.name.lower() else "02_baselines"
            reason = "contains checkpoint, config, summary, and prediction records; traceable enough for formal reference"
            confidence = "high"
        elif has_checkpoint:
            category = "90_legacy_misc"
            reason = "contains checkpoint but lacks full config/summary/prediction records; keep as historical asset, not formal comparison"
            confidence = "high"
        else:
            category = "99_uncertain_review"
            reason = "no checkpoint or formal report found; keep for human review"
            confidence = "medium"
        rows.append({
            "dir_name": d.name,
            "original_path": rel(d),
            "contents_summary": contents_summary(d),
            "has_checkpoint": has_checkpoint,
            "has_config": has_config,
            "has_summary": has_summary,
            "has_prediction_records": has_records,
            "suggested_category": category,
            "suggested_new_path": f"outputs/{category}/reference_models/{d.name}",
            "reason": reason,
            "confidence": confidence,
        })
    return rows


def base_move_plan() -> List[Dict[str, Any]]:
    mapping = [
        ("00_reference_models/baseline_replay_v2", "90_legacy_misc/reference_models/baseline_replay_v2", "90_legacy_misc", "reference checkpoint only; lacks config/summary/prediction records", "high"),
        ("00_reference_models/gppoint_detr_v7_exp2_current", "90_legacy_misc/reference_models/gppoint_detr_v7_exp2_current", "90_legacy_misc", "reference checkpoint only; formal V7 comparison lives in unified evaluation", "high"),
        ("01_main_model", "01_mainline_results/main_model_training_archive", "01_mainline_results", "contains V7/fair main model training reports and checkpoints", "medium"),
        ("02_combined_experiments", "90_legacy_misc/combined_experiments_archive", "90_legacy_misc", "mixed historical combined experiments; retain but do not treat as current mainline", "medium"),
        ("02_encoder_experiments", "90_legacy_misc/encoder_experiments_archive", "90_legacy_misc", "mixed encoder ablations including EMA_BIFPN source checkpoint; unified EMA result is indexed separately", "medium"),
        ("02_mechanism_experiments", "05_failed_experiments/07_selector_o2m_heatmap_simcc/mechanism_experiments_archive", "05_failed_experiments", "historical mechanism routes such as O2M/quality/DN were rejected from mainline", "medium"),
        ("02_point_refine_experiments", "05_failed_experiments/06_point_refine/point_refine_experiments_archive", "05_failed_experiments", "historical point refine route rejected from mainline", "medium"),
        ("02_sci_main_experiments", "90_legacy_misc/sci_main_experiments_archive", "90_legacy_misc", "mixed SCI-era experiments with historical value and inconsistent final status", "medium"),
        ("02_supervision_experiments", "05_failed_experiments/07_selector_o2m_heatmap_simcc/supervision_experiments_archive", "05_failed_experiments", "heatmap/dense auxiliary supervision routes rejected from mainline", "medium"),
        ("03_global_analysis", "04_diagnostics/global_analysis", "04_diagnostics", "global analysis and HP calibration diagnostics", "high"),
        ("04_tail_coordinate_experiments", "99_uncertain_review/tail_coordinate_experiments", "99_uncertain_review", "early tail-coordinate experiment needs human review before failed/mainline classification", "medium"),
        ("05_selector_experiments", "05_failed_experiments/07_selector_o2m_heatmap_simcc/selector_experiments_archive", "05_failed_experiments", "selector route stopped/rejected", "medium"),
        ("06_coordinate_refiner_experiments", "05_failed_experiments/07_selector_o2m_heatmap_simcc/coordinate_refiner_experiments_archive", "05_failed_experiments", "SimCC/heatmap/refiner routes rejected from mainline", "medium"),
        ("07_external_baselines", "02_baselines/external_baselines_archive", "02_baselines", "external and same-protocol baseline results", "high"),
        ("08_eval_unification", "03_unified_evaluation/eval_unification", "03_unified_evaluation", "unified metrics and comparison reports", "high"),
        ("09_model_improvement/ema_bifpn_relcal_v1_probe20", "05_failed_experiments/01_reliability_relcal/relcal_v1_probe20", "05_failed_experiments", "RELCAL reliability scalar route explicitly failed", "high"),
        ("10_candidate_diagnostics", "04_diagnostics/candidate_selection", "04_diagnostics", "candidate-level selection diagnostics", "high"),
        ("11_point_aware_rerank", "05_failed_experiments/02_postprocess_rerank/par_v1", "05_failed_experiments", "PAR/rerank route explicitly failed", "high"),
        ("12_point_coordinate_diagnostics", "04_diagnostics/point_coordinate_mechanism", "04_diagnostics", "point coordinate mechanism diagnostics", "high"),
        ("14_point_aware_matching", "05_failed_experiments/03_matching_pam", "05_failed_experiments", "PAM point-aware matcher route rejected", "high"),
        ("15_stem_aux_annotation", "06_data_supervision/stem_aux_annotation", "06_data_supervision", "stem_aux annotation package and data supervision assets", "high"),
        ("16_code_mechanism_audit", "04_diagnostics/code_mechanism_audit", "04_diagnostics", "code mechanism audit", "high"),
        ("18_worst_case_frequency_audit", "04_diagnostics/worst_case_frequency", "04_diagnostics", "worst-case frequency and dataset review diagnostics", "high"),
        ("21_external_method_review", "04_diagnostics/external_method_review", "04_diagnostics", "external literature/code method review", "high"),
        ("22_point_lsd_layer_oracle", "04_diagnostics/point_lsd_layer_oracle", "04_diagnostics", "POINT_LSD feasibility diagnostic, not a failed experiment", "high"),
        ("23_point_lsd", "05_failed_experiments/04_point_lsd", "05_failed_experiments", "POINT_LSD route failed", "high"),
        ("24_grouped_picking_query", "05_failed_experiments/05_grouped_query/grouped_picking_query_v1", "05_failed_experiments", "grouped offset inference route failed; raw-offset result preserved as negative/ablation asset", "high"),
        ("25_grouped_aux_control", "05_failed_experiments/05_grouped_query/grouped_aux_control", "05_failed_experiments", "grouped auxiliary control showed no independent positive route", "high"),
        ("26_continue_training_tradeoff", "04_diagnostics/continue_training_tradeoff", "04_diagnostics", "coverage/localization tradeoff diagnostic", "high"),
        ("27_coverage_preserving_point_refine", "05_failed_experiments/06_point_refine/point_only_refine", "05_failed_experiments", "point-only refine failed", "high"),
        ("28_point_feature_refine_hasdistill", "05_failed_experiments/06_point_refine/point_feature_refine_hasdistill", "05_failed_experiments", "point-feature refine with has distill failed", "high"),
        ("point_v2_report_xv_ta8nd", "90_legacy_misc/point_v2_report_xv_ta8nd", "90_legacy_misc", "early point_v2 report with historical reference value", "medium"),
        ("debug", "90_legacy_misc/debug", "90_legacy_misc", "debug artifacts; not a formal experiment", "medium"),
    ]
    rows = []
    for src_rel, dst_rel, category, reason, confidence in mapping:
        src = OUTPUTS / src_rel
        dst = OUTPUTS / dst_rel
        if src.exists():
            rows.append({
                "original_path": rel(src),
                "proposed_path": rel(dst),
                "category": category,
                "reason": reason,
                "confidence": confidence,
                "action": "move" if confidence in {"high", "medium"} else "review",
            })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def md_table(rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["|" + "|".join(fieldnames) + "|", "|" + "|".join(["---"] * len(fieldnames)) + "|"]
    for row in rows:
        vals = []
        for k in fieldnames:
            v = row.get(k, "")
            vals.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines) + "\n"


def write_reference_audit(rows: List[Dict[str, Any]]) -> None:
    fields = ["dir_name", "original_path", "contents_summary", "has_checkpoint", "has_config", "has_summary", "has_prediction_records", "suggested_category", "suggested_new_path", "reason", "confidence"]
    write_csv(INDEX / "reference_models_audit.csv", rows, fields)
    text = "# Reference Models Audit\n\n"
    text += "Formal reference directories must include checkpoint, config, summary, and prediction records. Checkpoint-only directories are retained as legacy assets, not formal paper comparison sources.\n\n"
    text += md_table(rows, fields)
    (INDEX / "reference_models_audit.md").write_text(text, encoding="utf-8")


def write_dry_run(rows: List[Dict[str, Any]]) -> None:
    fields = ["original_path", "proposed_path", "category", "reason", "confidence", "action"]
    write_csv(INDEX / "reorganize_outputs_dry_run.csv", rows, fields)
    text = "# Outputs Reorganization Dry Run\n\n"
    text += "No files are deleted. `move` rows are high/medium confidence. Low-confidence rows should stay in review.\n\n"
    text += md_table(rows, fields)
    (INDEX / "reorganize_outputs_dry_run.md").write_text(text, encoding="utf-8")


def move_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mappings = []
    for row in rows:
        if row["action"] != "move":
            continue
        src = ROOT / row["original_path"]
        dst = ROOT / row["proposed_path"]
        if not src.exists():
            row = dict(row)
            row["action_taken"] = "skipped_missing"
            mappings.append(row)
            continue
        if dst.exists():
            row = dict(row)
            row["action_taken"] = "skipped_target_exists"
            mappings.append(row)
            continue
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
        row = dict(row)
        row["action_taken"] = "moved"
        mappings.append(row)

    # Remove empty source containers only after moving their children.
    for maybe_empty in ["outputs/09_model_improvement", "outputs/00_reference_models"]:
        p = ROOT / maybe_empty
        if p.exists() and p.is_dir() and not any(p.iterdir()):
            p.rmdir()
            mappings.append({
                "original_path": maybe_empty,
                "proposed_path": "",
                "category": "empty_source_container",
                "reason": "empty source container removed after all child artifacts were moved",
                "confidence": "high",
                "action": "remove_empty_dir",
                "action_taken": "removed_empty_dir",
            })
    return mappings


def tree_text(root: Path, max_depth: int = 3) -> str:
    lines = []
    base_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        p = Path(current)
        depth = len(p.parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{p.name}/")
        if depth < max_depth:
            for f in sorted(files)[:20]:
                lines.append(f"{indent}  {f}")
            if len(files) > 20:
                lines.append(f"{indent}  ... {len(files) - 20} more files")
        dirs.sort()
    return "\n".join(lines) + "\n"


def find_summary_metrics(summary_path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    candidates = []
    for key in ["selected_test_metrics", "test_metrics", "primary_checkpoint_split_summary"]:
        val = data.get(key)
        if isinstance(val, dict):
            candidates.append(val.get("test", val))
    for key in ["test", "formal_test"]:
        if isinstance(data.get(key), dict):
            candidates.append(data[key])
    candidates.append(data)
    for val in candidates:
        if not isinstance(val, dict):
            continue
        gd = val.get("grape_detection", {}) if isinstance(val.get("grape_detection"), dict) else {}
        hp = val.get("has_picking", {}) if isinstance(val.get("has_picking"), dict) else {}
        pp = val.get("picking_point", {}) if isinstance(val.get("picking_point"), dict) else {}
        unified = val.get("unified_point_metrics", {}) if isinstance(val.get("unified_point_metrics"), dict) else {}
        out = {
            "AP": gd.get("AP", val.get("AP", val.get("ap"))),
            "AP50": gd.get("AP50", val.get("AP50", val.get("ap50"))),
            "F1": hp.get("f1", val.get("has_picking_f1", unified.get("instance_f1"))),
            "pair": pp.get("pair_count", val.get("point_pair_count", unified.get("pair_count"))),
            "mean_L2": pp.get("mean_l2_px", val.get("point_mean_l2_px", unified.get("mean_l2"))),
            "PPL30": pp.get("ppl_sr_30", val.get("PPL-SR@30", val.get("ppl30", unified.get("ppl_sr_30")))),
            "PPL50": pp.get("ppl_sr_50", val.get("PPL-SR@50", val.get("ppl50", unified.get("ppl_sr_50")))),
        }
        if any(v is not None for v in out.values()):
            return out
    return {}


def classify_experiment_path(path: Path) -> Tuple[str, str, str, str, str]:
    r = rel(path)
    lower = r.lower()
    if "/01_mainline_results/" in lower:
        return "01_mainline_results", "mainline", "candidate_or_reference", "main_result", "mainline/reference archive"
    if "/02_baselines/" in lower:
        return "02_baselines", "baseline", "reference", "baseline", "baseline comparison"
    if "/03_unified_evaluation/" in lower:
        return "03_unified_evaluation", "unified", "available", "diagnostic", "unified comparison/evaluation"
    if "/04_diagnostics/" in lower:
        return "04_diagnostics", "diagnostic", "complete", "diagnostic", "mechanism/data/code diagnostic"
    if "/05_failed_experiments/" in lower:
        return "05_failed_experiments", "failed", "rejected", "negative_result", "failed or stopped route"
    if "/06_data_supervision/" in lower:
        return "06_data_supervision", "data", "review", "data_review", "data supervision/relabel asset"
    if "/90_legacy_misc/" in lower:
        return "90_legacy_misc", "legacy", "legacy", "legacy", "historical asset"
    if "/99_uncertain_review/" in lower:
        return "99_uncertain_review", "uncertain", "uncertain", "uncertain", "needs human review"
    return "unknown", "unknown", "unknown", "uncertain", "not classified"


def make_master_index() -> List[Dict[str, Any]]:
    rows = []
    for summary in sorted(OUTPUTS.rglob("summary.json")):
        if "/_index/" in rel(summary):
            continue
        exp_dir = summary.parent
        if exp_dir.name == "report":
            exp_dir = exp_dir.parent
        category, sub, status, usage, note = classify_experiment_path(exp_dir)
        metrics = find_summary_metrics(summary)
        rows.append({
            "experiment_name": exp_dir.name,
            "category": category,
            "sub_category": sub,
            "status": status,
            "original_path": "",
            "current_path": rel(exp_dir),
            "AP": metrics.get("AP", ""),
            "AP50": metrics.get("AP50", ""),
            "F1": metrics.get("F1", ""),
            "pair": metrics.get("pair", ""),
            "mean_L2": metrics.get("mean_L2", ""),
            "PPL30": metrics.get("PPL30", ""),
            "PPL50": metrics.get("PPL50", ""),
            "decision": status,
            "paper_usage": usage,
            "notes": note,
        })
    return rows


def write_master_index(rows: List[Dict[str, Any]]) -> None:
    fields = ["experiment_name", "category", "sub_category", "status", "original_path", "current_path", "AP", "AP50", "F1", "pair", "mean_L2", "PPL30", "PPL50", "decision", "paper_usage", "notes"]
    write_csv(INDEX / "experiment_master_index.csv", rows, fields)
    text = "# Experiment Master Index\n\n"
    text += "This index is generated from `summary.json` files after outputs reorganization. Use it as the first lookup table before opening individual reports.\n\n"
    text += md_table(rows, fields)
    (INDEX / "experiment_master_index.md").write_text(text, encoding="utf-8")


def write_standard_docs(mappings: List[Dict[str, Any]], dry_rows: List[Dict[str, Any]], ref_rows: List[Dict[str, Any]]) -> None:
    ensure_dir(INDEX)
    std = "# Current Outputs Standard\n\n"
    for name, desc in STANDARD_DIRS.items():
        std += f"## `{name}`\n\n{desc}\n\n"
        if name == "05_failed_experiments":
            std += "Subdirectories:\n\n" + "\n".join(f"- `{s}`" for s in FAILED_SUBDIRS) + "\n\n"
    (INDEX / "current_outputs_standard.md").write_text(std, encoding="utf-8")
    (OUTPUTS / "README_OUTPUTS.md").write_text(std + "## New Experiment Placement\n\n- Mainline model results -> `01_mainline_results/`\n- External or historical baselines -> `02_baselines/`\n- Unified evaluation -> `03_unified_evaluation/`\n- Mechanism diagnostics -> `04_diagnostics/`\n- Failed experiments -> `05_failed_experiments/`\n- Data supervision -> `06_data_supervision/`\n- Paper figures/tables -> `07_paper_assets/`\n- Historical miscellaneous -> `90_legacy_misc/`\n- Uncertain items -> `99_uncertain_review/`\n", encoding="utf-8")
    (INDEX / "README_OUTPUTS.md").write_text((OUTPUTS / "README_OUTPUTS.md").read_text(encoding="utf-8"), encoding="utf-8")

    for name, desc in STANDARD_DIRS.items():
        d = OUTPUTS / name
        ensure_dir(d)
        if name == "_index":
            continue
        readme = f"# {name}\n\n{desc}\n\n"
        readme += "Use `outputs/_index/experiment_master_index.csv` and `outputs/_index/output_path_mapping.csv` to locate experiments and historical paths.\n\n"
        if name == "05_failed_experiments":
            readme += "Do not continue training routes stored here unless a new user-approved plan explicitly reopens them.\n\n"
        elif name == "04_diagnostics":
            readme += "Diagnostics here explain mechanisms and failure causes; they are not failed experiments by themselves.\n\n"
        elif name == "01_mainline_results":
            readme += "Only put current paper mainline candidates and formal main references here.\n\n"
        (d / "README.md").write_text(readme, encoding="utf-8")

    moved = [m for m in mappings if m.get("action_taken") == "moved"]
    skipped = [m for m in mappings if str(m.get("action_taken", "")).startswith("skipped")]
    uncertain = [r for r in dry_rows if r["category"] == "99_uncertain_review"]
    pth_count = count_files(OUTPUTS, lambda p: p.suffix.lower() == ".pth")
    final = "# Outputs Reorganization Final Report\n\n"
    final += "## 1. Reference Models\n\n"
    final += md_table(ref_rows, ["dir_name", "original_path", "has_checkpoint", "has_config", "has_summary", "has_prediction_records", "suggested_category", "suggested_new_path", "reason", "confidence"])
    final += "\n## 2. Final Classification Standard\n\n"
    final += "The active top-level outputs standard is: `_index`, `01_mainline_results`, `02_baselines`, `03_unified_evaluation`, `04_diagnostics`, `05_failed_experiments`, `06_data_supervision`, `07_paper_assets`, `90_legacy_misc`, `99_uncertain_review`.\n\n"
    final += "## 3. Moved Directories\n\n" + md_table(moved, ["original_path", "proposed_path", "category", "reason", "confidence", "action_taken"])
    final += "\n## 4. Retained / Skipped Directories\n\n" + md_table(skipped, ["original_path", "proposed_path", "category", "reason", "confidence", "action_taken"])
    final += "\n## 5. Uncertain Review\n\n" + md_table(uncertain, ["original_path", "proposed_path", "reason", "confidence", "action"])
    final += f"\n## 6. Remaining Checkpoints\n\nDetected `{pth_count}` `.pth` files under `outputs` after reorganization. They were not deleted. Checkpoint-bearing legacy/reference folders are explicitly indexed.\n\n"
    final += "## 7. YAML Handling\n\nThis run did not move or delete YAML files. It only generated `configs/rtv4/config_inventory.csv` and `configs/rtv4/delete_candidate_configs.md`.\n\n"
    final += "## 8. Future Placement\n\nNew mainline results go to `01_mainline_results`; baselines to `02_baselines`; unified reports to `03_unified_evaluation`; diagnostics to `04_diagnostics`; failed routes to `05_failed_experiments`; data-supervision assets to `06_data_supervision`; paper tables/figures to `07_paper_assets`; legacy or unclear assets to `90_legacy_misc` or `99_uncertain_review`.\n"
    (INDEX / "reorganize_outputs_final_report.md").write_text(final, encoding="utf-8")


def config_inventory() -> None:
    rows = []
    delete_candidates = []
    active_keep = {
        "rtv4_hgnetv2_s_grape_point_main.yml",
        "rtv4_hgnetv2_s_grape_point_enc_ema_bifpn_weighted_fusion.yml",
        "rtv4_hgnetv2_s_grape_point_enc_ema_fusion.yml",
        "rtv4_hgnetv2_s_grape_point_ema_bifpn_hp_pick_protocol.yml",
    }
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        name = path.name
        lower = name.lower()
        likely = "unknown"
        related = ""
        candidate = "keep"
        reason = "active or reference configuration"
        if name in active_keep:
            likely = "main_or_protocol"
            related = name.replace(".yml", "")
        elif any(k in lower for k in ["relcal", "pam", "grouped", "point_lsd", "point_only_refine", "point_feature_refine", "continue20", "selector"]):
            likely = "failed_or_control_experiment"
            related = name.replace(".yml", "")
            candidate = "delete_candidate_after_user_confirm"
            reason = "appears tied to a failed/control route; keep for reproducibility until user confirms deletion"
            delete_candidates.append((name, reason))
        else:
            likely = "legacy_or_ablation"
            related = name.replace(".yml", "")
            candidate = "review"
            reason = "not classified as current main config; review before deletion"
        rows.append({
            "file_name": name,
            "likely_usage": likely,
            "related_experiment": related,
            "keep_or_delete_candidate": candidate,
            "reason": reason,
        })
    write_csv(CONFIG_DIR / "config_inventory.csv", rows, ["file_name", "likely_usage", "related_experiment", "keep_or_delete_candidate", "reason"])
    text = "# Delete Candidate Configs\n\nNo YAML files were deleted in this run. These files are candidates only if the user later confirms cleanup and no active script/index requires them.\n\n"
    for name, reason in delete_candidates:
        text += f"- `{name}`: {reason}\n"
    (CONFIG_DIR / "delete_candidate_configs.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    ensure_dir(INDEX)
    ref_rows = audit_reference_models()
    dry_rows = base_move_plan()
    write_reference_audit(ref_rows)
    write_dry_run(dry_rows)
    config_inventory()
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "rows": len(dry_rows), "reference_rows": len(ref_rows)}, ensure_ascii=False, indent=2))
        return
    mappings = move_rows(dry_rows)
    write_csv(INDEX / "output_path_mapping.csv", mappings, ["original_path", "proposed_path", "category", "reason", "confidence", "action", "action_taken"])
    (INDEX / "output_path_mapping.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    (INDEX / "after_reorganize_tree.txt").write_text(tree_text(OUTPUTS, max_depth=3), encoding="utf-8")
    master_rows = make_master_index()
    write_master_index(master_rows)
    write_standard_docs(mappings, dry_rows, ref_rows)
    print(json.dumps({"mode": "execute", "moved": sum(1 for m in mappings if m.get("action_taken") == "moved"), "master_rows": len(master_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
