from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from tools.pald_smoke_utils import load_checkpoint_for_tuning, load_sample, set_no_pretrained, split_image_rows


DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_ema_bifpn_cada_v1_adapter_only_probe20.yml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "90_legacy_misc"
    / "encoder_experiments_archive"
    / "encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526"
    / "best_composite.pth"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "01_mainline_results"
    / "candidate_cada_v1"
    / "ema_bifpn_cada_v1_adapter_only_probe20"
)
CORE_KEYS = ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets", "aux_outputs", "pre_outputs"]
DISABLED_BRANCH_FLAGS = [
    "use_c2f_ccr",
    "use_toproi_simcc_refiner",
    "use_toproi_heatmap_refiner",
    "use_grouped_picking_query",
    "use_qdpt_lite",
    "use_dpo_head",
    "use_hrpb",
    "use_point_quality_head",
    "use_point_selector_head",
    "use_point_accept_head",
    "use_point_reliability_head",
    "use_weak_point_heatmap_head",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train smoke and trainable audit for CADA adapter-only probe20.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_visible_row(split: str = "valid") -> dict:
    for row in split_image_rows(split, None):
        if any(bool(ann.get("has_picking", ann.get("picking_point") is not None)) for ann in row.get("annotations", [])):
            return row
    raise RuntimeError(f"No visible picking annotations found in split={split}.")


def missing_only_cada(missing: list[str]) -> bool:
    if not missing:
        return True
    return all("cada_adapter" in key or "cada_" in key for key in missing)


def optimizer_param_names(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> list[str]:
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    names = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            names.append(id_to_name.get(id(param), "<unknown>"))
    return sorted(names)


def normalize_encoder_output(value):
    if isinstance(value, tuple) and len(value) == 2:
        return value[0]
    return value


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    report: dict = {
        "stage": "CADA_V1_ADAPTER_ONLY_PRETRAIN_AUDIT",
        "scope": "pre-train audit and one-batch smoke only",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "device": str(device),
    }

    try:
        cfg = YAMLConfig(str(args.config), device=str(device), use_amp=False)
        set_no_pretrained(cfg)
        model = cfg.model.to(device)
        load_info = load_checkpoint_for_tuning(model, args.checkpoint)
        optimizer = cfg.optimizer
    except Exception as exc:
        report["exception"] = repr(exc)
        return write_outputs(args.output_dir, report, "blocked_by_config_or_checkpoint")

    missing = load_info.get("missing", [])
    shape_mismatch = load_info.get("shape_mismatch", [])
    report["checkpoint"] = {
        "matched_count": load_info.get("matched_count"),
        "missing_count": len(missing),
        "missing_all_cada": missing_only_cada(missing),
        "missing_sample": missing[:80],
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch": shape_mismatch[:40],
    }

    trainable_rows = []
    frozen_group_counts = {
        "backbone": 0,
        "hybrid_encoder_non_cada": 0,
        "decoder": 0,
        "class_head": 0,
        "bbox_head": 0,
        "has_head": 0,
        "offset_head": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        row = {
            "name": name,
            "shape": "x".join(map(str, param.shape)),
            "requires_grad": bool(param.requires_grad),
            "numel": int(param.numel()),
            "is_cada": "cada" in name,
        }
        if param.requires_grad:
            trainable_rows.append(row)
        else:
            if name.startswith("backbone"):
                frozen_group_counts["backbone"] += param.numel()
            elif name.startswith("encoder") and "cada" not in name:
                frozen_group_counts["hybrid_encoder_non_cada"] += param.numel()
            elif "score_head" in name:
                frozen_group_counts["class_head"] += param.numel()
            elif "bbox_head" in name:
                frozen_group_counts["bbox_head"] += param.numel()
            elif "picking_head" in name and "offset" not in name:
                frozen_group_counts["has_head"] += param.numel()
            elif "picking_offset_head" in name:
                frozen_group_counts["offset_head"] += param.numel()
            elif name.startswith("decoder"):
                frozen_group_counts["decoder"] += param.numel()
            else:
                frozen_group_counts["other"] += param.numel()

    optimizer_names = optimizer_param_names(model, optimizer)
    report["trainable"] = {
        "trainable_param_count": len(trainable_rows),
        "trainable_param_numel": sum(int(row["numel"]) for row in trainable_rows),
        "trainable_all_cada": all(row["is_cada"] for row in trainable_rows),
        "optimizer_param_count": len(optimizer_names),
        "optimizer_param_numel": sum(param.numel() for group in optimizer.param_groups for param in group["params"]),
        "optimizer_all_cada": all("cada" in name for name in optimizer_names),
        "optimizer_param_names": optimizer_names,
        "frozen_group_numel": frozen_group_counts,
    }

    decoder_cfg = cfg.yaml_cfg.get("DFINETransformer", {})
    report["disabled_branch_flags"] = {key: bool(decoder_cfg.get(key, False)) for key in DISABLED_BRANCH_FLAGS}
    report["disabled_branches_ok"] = not any(report["disabled_branch_flags"].values())

    try:
        row = first_visible_row("valid")
        sample, target = load_sample(row, device, args.image_size)
        report["sample"] = {"image_id": int(row["image"]["id"]), "file_name": row["image"]["file_name"]}
        model.train()
        torch.manual_seed(20260605)
        if sample.is_cuda:
            torch.cuda.manual_seed_all(20260605)
        outputs = model(sample, targets=[target])
        report["core_fields_present"] = all(key in outputs for key in CORE_KEYS)
        report["finite_outputs"] = all(
            torch.isfinite(outputs[key]).all().detach().cpu().item()
            for key in ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets"]
            if key in outputs
        )

        with torch.no_grad():
            backbone_feats = model.backbone(sample)
            encoder_output, cada_debug = model.encoder(backbone_feats, return_cada_debug=True)
            encoder_output = normalize_encoder_output(encoder_output)
        feature_rows = []
        shape_ok = True
        gamma_values = []
        for idx, debug in enumerate(cada_debug or []):
            gamma_values.append(float(debug.get("gamma", 0.0)))
            feature_rows.append(
                {
                    "level": idx,
                    "input_shape": "x".join(map(str, debug.get("input_shape", []))),
                    "output_shape": "x".join(map(str, debug.get("output_shape", []))),
                    "mean_abs_diff": debug.get("mean_abs_diff", ""),
                    "update_mean_abs": debug.get("update_mean_abs", ""),
                    "gamma": debug.get("gamma", ""),
                    "input_finite": debug.get("input_finite", ""),
                    "output_finite": debug.get("output_finite", ""),
                }
            )
            shape_ok = shape_ok and debug.get("input_shape") == debug.get("output_shape")
        write_csv(
            args.output_dir / "cada_adapter_only_feature_shapes.csv",
            feature_rows,
            ["level", "input_shape", "output_shape", "mean_abs_diff", "update_mean_abs", "gamma", "input_finite", "output_finite"],
        )
        report["cada"] = {
            "feature_shape_ok": bool(shape_ok),
            "gamma_values": gamma_values,
            "gamma_close_to_1e_3": all(abs(value - 1e-3) < 1e-8 for value in gamma_values),
        }

        model.zero_grad(set_to_none=True)
        loss = outputs["pred_picking_offsets"].mean() + outputs["pred_has_picking"].mean() * 0.01 + outputs["pred_boxes"].mean() * 0.01
        loss.backward()
        grad_rows = []
        for name, param in model.named_parameters():
            has_grad = bool(param.grad is not None and param.grad.detach().abs().sum().cpu().item() > 0)
            if param.requires_grad or has_grad:
                grad_rows.append(
                    {
                        "name": name,
                        "requires_grad": bool(param.requires_grad),
                        "is_cada": "cada" in name,
                        "has_nonzero_grad": has_grad,
                    }
                )
        write_csv(args.output_dir / "cada_adapter_only_grad_audit.csv", grad_rows, ["name", "requires_grad", "is_cada", "has_nonzero_grad"])
        report["backward"] = {
            "loss": float(loss.detach().cpu().item()),
            "grad_rows": len(grad_rows),
            "cada_internal_has_grad": any(row["is_cada"] and row["has_nonzero_grad"] and "cada_gamma" not in row["name"] for row in grad_rows),
            "cada_gamma_has_grad": any(row["is_cada"] and row["has_nonzero_grad"] and "cada_gamma" in row["name"] for row in grad_rows),
            "non_cada_has_grad": any((not row["is_cada"]) and row["has_nonzero_grad"] for row in grad_rows),
        }
    except Exception as exc:
        report["exception"] = repr(exc)
        return write_outputs(args.output_dir, report, "blocked_by_forward_or_backward")

    write_csv(
        args.output_dir / "trainable_param_audit.csv",
        trainable_rows,
        ["name", "shape", "requires_grad", "numel", "is_cada"],
    )

    checks = {
        "checkpoint_ok": len(shape_mismatch) == 0 and missing_only_cada(missing),
        "optimizer_only_cada": bool(report["trainable"]["trainable_all_cada"] and report["trainable"]["optimizer_all_cada"]),
        "disabled_branches_ok": bool(report["disabled_branches_ok"]),
        "core_fields_present": bool(report.get("core_fields_present")),
        "finite_outputs": bool(report.get("finite_outputs")),
        "feature_shape_ok": bool(report.get("cada", {}).get("feature_shape_ok")),
        "gamma_init_ok": bool(report.get("cada", {}).get("gamma_close_to_1e_3")),
        "cada_internal_grad_ok": bool(report.get("backward", {}).get("cada_internal_has_grad")),
        "non_cada_no_grad": not bool(report.get("backward", {}).get("non_cada_has_grad")),
    }
    report["checks"] = checks
    decision = "ready_for_cada_adapter_only_probe20" if all(checks.values()) else "blocked_by_pretrain_audit"
    return write_outputs(args.output_dir, report, decision)


def write_outputs(output_dir: Path, report: dict, decision: str) -> int:
    report["decision"] = decision
    write_json(output_dir / "trainable_param_audit.json", report)
    lines = [
        "# CADA Adapter-Only Trainable Parameter Audit",
        "",
        f"Decision: `{decision}`",
        "",
        "## Checks",
    ]
    for key, value in report.get("checks", {}).items():
        lines.append(f"- `{key}`: {value}")
    if "checks" not in report:
        lines.append(f"- exception: `{report.get('exception', '')}`")
    lines.extend(
        [
            "",
            "## Trainable Scope",
            f"- trainable_param_count: {report.get('trainable', {}).get('trainable_param_count', 'n/a')}",
            f"- trainable_param_numel: {report.get('trainable', {}).get('trainable_param_numel', 'n/a')}",
            f"- optimizer_param_count: {report.get('trainable', {}).get('optimizer_param_count', 'n/a')}",
            f"- optimizer_param_numel: {report.get('trainable', {}).get('optimizer_param_numel', 'n/a')}",
            f"- trainable_all_cada: {report.get('trainable', {}).get('trainable_all_cada', 'n/a')}",
            f"- optimizer_all_cada: {report.get('trainable', {}).get('optimizer_all_cada', 'n/a')}",
            "",
            "## Checkpoint",
            f"- missing_count: {report.get('checkpoint', {}).get('missing_count', 'n/a')}",
            f"- missing_all_cada: {report.get('checkpoint', {}).get('missing_all_cada', 'n/a')}",
            f"- shape_mismatch_count: {report.get('checkpoint', {}).get('shape_mismatch_count', 'n/a')}",
            "",
            "## CADA",
            f"- gamma_values: `{report.get('cada', {}).get('gamma_values', [])}`",
            f"- cada_internal_has_grad: {report.get('backward', {}).get('cada_internal_has_grad', 'n/a')}",
            f"- cada_gamma_has_grad: {report.get('backward', {}).get('cada_gamma_has_grad', 'n/a')}",
            "",
            "## Disabled Branch Flags",
            f"`{report.get('disabled_branch_flags', {})}`",
            "",
            "Training is allowed only when decision is `ready_for_cada_adapter_only_probe20`.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trainable_param_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if decision == "ready_for_cada_adapter_only_probe20" else 1


if __name__ == "__main__":
    raise SystemExit(main())
