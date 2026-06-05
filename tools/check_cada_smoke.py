from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from tools.pald_smoke_utils import load_checkpoint_for_tuning, load_sample, set_no_pretrained, split_image_rows


DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_enc_ema_bifpn_weighted_fusion.yml"
DEFAULT_SMOKE_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_ema_bifpn_cada_v1_smoke.yml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "90_legacy_misc"
    / "encoder_experiments_archive"
    / "encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526"
    / "best_composite.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_cada_v1" / "smoke"
DECISION_DIR = REPO_ROOT / "outputs" / "01_mainline_results" / "candidate_cada_v1"
CORE_KEYS = ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets", "aux_outputs", "pre_outputs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check CADA without training or checkpoint generation.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--smoke-config", type=Path, default=DEFAULT_SMOKE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--equivalence-tol", type=float, default=1e-8)
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


def load_model(config: Path, checkpoint: Path, device: torch.device):
    cfg = YAMLConfig(str(config), device=str(device), use_amp=False)
    set_no_pretrained(cfg)
    model = cfg.model.to(device)
    load_info = load_checkpoint_for_tuning(model, checkpoint)
    return cfg, model, load_info


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a is None or b is None or tuple(a.shape) != tuple(b.shape):
        return None
    return float((a - b).abs().mean().detach().cpu().item())


def output_finite(outputs: dict) -> bool:
    for key in ("pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets"):
        value = outputs.get(key)
        if value is None or not bool(torch.isfinite(value).all().detach().cpu().item()):
            return False
    return True


def normalize_encoder_output(value):
    if isinstance(value, tuple) and len(value) == 2:
        return value[0]
    return value


def encoder_features_with_debug(model: torch.nn.Module, sample: torch.Tensor):
    model.eval()
    with torch.no_grad():
        backbone_feats = model.backbone(sample)
        encoder_output, cada_debug = model.encoder(backbone_feats, return_cada_debug=True)
        encoder_output = normalize_encoder_output(encoder_output)
    return list(encoder_output), cada_debug or []


def run_model_forward(model: torch.nn.Module, sample: torch.Tensor, target: dict, seed: int = 1234):
    model.train()
    torch.manual_seed(seed)
    if sample.is_cuda:
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        return model(sample, targets=[target])


def cada_param_rows(model: torch.nn.Module) -> list[dict]:
    rows = []
    for name, param in model.named_parameters():
        if "cada_adapter" not in name and "cada_" not in name:
            continue
        grad = param.grad
        grad_nonzero = bool(grad is not None and grad.detach().abs().sum().cpu().item() > 0)
        rows.append(
            {
                "name": name,
                "shape": "x".join(map(str, param.shape)),
                "requires_grad": bool(param.requires_grad),
                "numel": int(param.numel()),
                "grad_nonzero": grad_nonzero,
                "is_gamma": "cada_gamma" in name,
            }
        )
    return rows


def grad_summary(rows: list[dict]) -> dict:
    return {
        "cada_param_count": len(rows),
        "cada_trainable_count": sum(1 for row in rows if row["requires_grad"]),
        "cada_grad_nonzero_count": sum(1 for row in rows if row["grad_nonzero"]),
        "cada_gamma_grad_nonzero_count": sum(1 for row in rows if row["is_gamma"] and row["grad_nonzero"]),
        "cada_param_numel": sum(int(row["numel"]) for row in rows),
    }


def missing_only_cada(missing: list[str]) -> bool:
    if not missing:
        return True
    return all("cada_adapter" in key or "cada_" in key for key in missing)


def import_checks() -> dict:
    checks = {}
    for module_name in ("tools.make_grape_point_report", "tools.export_grape_point_predictions"):
        try:
            importlib.import_module(module_name)
            checks[module_name] = True
        except Exception as exc:
            checks[module_name] = str(exc)
    return checks


def write_probe_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# EMA_BIFPN_CADA_V1_PROBE20 Plan",
                "",
                "Scope: plan only. Do not run this during CADA smoke.",
                "",
                "Output directory: `outputs/01_mainline_results/candidate_cada_v1/ema_bifpn_cada_v1_probe20/`.",
                "",
                "Shared settings:",
                "- Warm-start from EMA_BIFPN best checkpoint.",
                "- Train 20 epoch probe.",
                "- `use_ema: false`.",
                "- `cada_gate_init: 1e-3` for probe so adapter internals receive gradient.",
                "- Freeze backbone.",
                "- Do not change matcher, postprocessor, has threshold, or output field semantics.",
                "- Do not enable PALD, HRPB, QDPT, DPO, PCGrad, PAM, PAR, RELCAL, POINT_LSD, grouped query, selector, accept, O2M, SimCC, heatmap, or stem_aux.",
                "",
                "Strategy A: `adapter_only`",
                "- Train CADA adapter plus point branch.",
                "- Keep detection/class/bbox/has heads frozen or very low risk.",
                "- Risk: lower instability, but benefit may be small.",
                "",
                "Strategy B: `adapter_plus_encoder_light`",
                "- Train CADA adapter plus selected HybridEncoder projection/fusion layers.",
                "- Keep backbone frozen and keep matcher/postprocessor unchanged.",
                "- Risk: higher coverage drift; benefit may better match structural change.",
                "",
                "Probe pass criteria:",
                "- AP drop <= 0.003.",
                "- F1 >= EMA_BIFPN - 0.005.",
                "- pair >= EMA_BIFPN - 3.",
                "- mean L2 lower than EMA_BIFPN.",
                "- PPL-SR@30 higher than EMA_BIFPN.",
                "- L2>30 lower than EMA_BIFPN.",
                "- p90 L2 not higher than EMA_BIFPN.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    row = first_visible_row("valid")
    sample, target = load_sample(row, device, args.image_size)

    report: dict = {
        "stage": "CADA_V1_SMOKE",
        "scope": "smoke only; no training, no test evaluation, no checkpoint generation",
        "base_config": str(args.base_config),
        "smoke_config": str(args.smoke_config),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "image_id": int(row["image"]["id"]),
        "file_name": row["image"]["file_name"],
    }

    try:
        base_cfg, base_model, base_load = load_model(args.base_config, args.checkpoint, device)
        smoke_cfg, smoke_model, smoke_load = load_model(args.smoke_config, args.checkpoint, device)
        report["base_load"] = {
            "matched_count": base_load.get("matched_count"),
            "missing_count": len(base_load.get("missing", [])),
            "shape_mismatch_count": len(base_load.get("shape_mismatch", [])),
            "shape_mismatch": base_load.get("shape_mismatch", [])[:40],
        }
        smoke_missing = smoke_load.get("missing", [])
        report["smoke_load"] = {
            "matched_count": smoke_load.get("matched_count"),
            "missing_count": len(smoke_missing),
            "missing_all_cada": missing_only_cada(smoke_missing),
            "missing_sample": smoke_missing[:80],
            "shape_mismatch_count": len(smoke_load.get("shape_mismatch", [])),
            "shape_mismatch": smoke_load.get("shape_mismatch", [])[:40],
        }
    except Exception as exc:
        report["exception"] = repr(exc)
        return write_failure(report, "blocked_by_checkpoint_load_issue")

    try:
        base_outputs = run_model_forward(base_model, sample, target, seed=20260605)
        smoke_outputs = run_model_forward(smoke_model, sample, target, seed=20260605)
        report["base_keys"] = sorted(base_outputs.keys())
        report["smoke_keys"] = sorted(smoke_outputs.keys())
        report["base_core_present"] = all(key in base_outputs for key in CORE_KEYS)
        report["smoke_core_present"] = all(key in smoke_outputs for key in CORE_KEYS)
        report["output_finite"] = bool(output_finite(base_outputs) and output_finite(smoke_outputs))
        report["output_diffs"] = {
            key: tensor_diff(base_outputs.get(key), smoke_outputs.get(key))
            for key in ("pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets")
        }

        base_features, _ = encoder_features_with_debug(base_model, sample)
        smoke_features, cada_debug = encoder_features_with_debug(smoke_model, sample)
        feature_rows = []
        shape_ok = len(base_features) == len(smoke_features) == len(cada_debug)
        encoder_diffs = []
        for idx, (base_feat, smoke_feat) in enumerate(zip(base_features, smoke_features)):
            diff = tensor_diff(base_feat, smoke_feat)
            encoder_diffs.append(diff)
            debug = cada_debug[idx] if idx < len(cada_debug) else {}
            feature_rows.append(
                {
                    "level": idx,
                    "input_shape": "x".join(map(str, debug.get("input_shape", []))),
                    "output_shape": "x".join(map(str, debug.get("output_shape", []))),
                    "base_shape": "x".join(map(str, base_feat.shape)),
                    "smoke_shape": "x".join(map(str, smoke_feat.shape)),
                    "mean_abs_diff": debug.get("mean_abs_diff", ""),
                    "base_vs_cada_output_mean_abs_diff": diff,
                    "gamma": debug.get("gamma", ""),
                    "input_finite": debug.get("input_finite", ""),
                    "output_finite": debug.get("output_finite", ""),
                }
            )
            shape_ok = shape_ok and tuple(base_feat.shape) == tuple(smoke_feat.shape)
            if debug:
                shape_ok = shape_ok and debug.get("input_shape") == debug.get("output_shape")
        write_csv(
            args.output_dir / "cada_feature_shapes.csv",
            feature_rows,
            [
                "level",
                "input_shape",
                "output_shape",
                "base_shape",
                "smoke_shape",
                "mean_abs_diff",
                "base_vs_cada_output_mean_abs_diff",
                "gamma",
                "input_finite",
                "output_finite",
            ],
        )
        report["feature_shape_ok"] = bool(shape_ok)
        report["encoder_output_mean_abs_diffs"] = encoder_diffs
        report["cada_gamma_values"] = [row.get("gamma") for row in feature_rows]
    except Exception as exc:
        report["exception"] = repr(exc)
        return write_failure(report, "blocked_by_shape_mismatch")

    try:
        smoke_model.train()
        smoke_model.zero_grad(set_to_none=True)
        torch.manual_seed(20260605)
        if sample.is_cuda:
            torch.cuda.manual_seed_all(20260605)
        train_outputs = smoke_model(sample, targets=[target])
        loss = train_outputs["pred_picking_offsets"].mean()
        loss.backward()
        param_rows = cada_param_rows(smoke_model)
        write_csv(
            args.output_dir / "cada_trainable_params.csv",
            param_rows,
            ["name", "shape", "requires_grad", "numel", "grad_nonzero", "is_gamma"],
        )
        report["backward_loss"] = float(loss.detach().cpu().item())
        report["grad_summary"] = grad_summary(param_rows)
    except Exception as exc:
        report["exception"] = repr(exc)
        return write_failure(report, "blocked_by_no_cada_grad")

    report["import_checks"] = import_checks()
    report["matcher_modified_by_cada"] = False
    report["postprocessor_modified_by_cada"] = False
    report["postprocessor_output_fields_unchanged_by_design"] = True

    equivalence_ok = all(
        value is not None and value <= args.equivalence_tol
        for value in report["output_diffs"].values()
    ) and all(value is not None and value <= args.equivalence_tol for value in report["encoder_output_mean_abs_diffs"])
    checkpoint_ok = (
        len(base_load.get("shape_mismatch", [])) == 0
        and len(smoke_load.get("shape_mismatch", [])) == 0
        and missing_only_cada(smoke_missing)
    )
    cada_grad_ok = report["grad_summary"]["cada_gamma_grad_nonzero_count"] > 0
    import_ok = all(value is True for value in report["import_checks"].values())
    smoke_ok = bool(
        checkpoint_ok
        and report["base_core_present"]
        and report["smoke_core_present"]
        and report["output_finite"]
        and report["feature_shape_ok"]
        and equivalence_ok
        and cada_grad_ok
        and import_ok
    )

    report["checks"] = {
        "checkpoint_ok": checkpoint_ok,
        "core_fields_ok": bool(report["base_core_present"] and report["smoke_core_present"]),
        "finite_ok": bool(report["output_finite"]),
        "feature_shape_ok": bool(report["feature_shape_ok"]),
        "mainline_equivalence_ok": bool(equivalence_ok),
        "cada_grad_ok": bool(cada_grad_ok),
        "import_ok": bool(import_ok),
    }

    if smoke_ok:
        decision = "ready_for_cada_probe20"
    elif not checkpoint_ok:
        decision = "blocked_by_checkpoint_load_issue"
    elif not report["feature_shape_ok"]:
        decision = "blocked_by_shape_mismatch"
    elif not equivalence_ok:
        decision = "blocked_by_mainline_equivalence_failure"
    elif not cada_grad_ok:
        decision = "blocked_by_no_cada_grad"
    else:
        decision = "blocked_by_other_smoke_failure"

    report["decision"] = decision
    write_outputs(report)
    if decision == "ready_for_cada_probe20":
        write_probe_plan(DECISION_DIR / "cada_probe20_plan.md")
    return 0 if decision == "ready_for_cada_probe20" else 1


def write_failure(report: dict, decision: str) -> int:
    report["decision"] = decision
    write_outputs(report)
    return 1


def write_outputs(report: dict) -> None:
    smoke_dir = Path(report.get("output_dir", DEFAULT_OUTPUT_DIR))
    decision = report["decision"]
    write_json(smoke_dir / "cada_smoke_report.json", report)
    write_json(DECISION_DIR / "cada_smoke_decision.json", {"decision": decision, "checks": report.get("checks", {})})
    lines = [
        "# CADA V1 Smoke Report",
        "",
        f"Decision: `{decision}`",
        "",
        "Scope: smoke only; no training, no test evaluation, no checkpoint generation.",
        "",
        "## Check Summary",
    ]
    for key, value in report.get("checks", {}).items():
        lines.append(f"- `{key}`: {value}")
    if "checks" not in report:
        lines.append(f"- exception: `{report.get('exception', '')}`")
    lines.extend(
        [
            "",
            "## Checkpoint",
            f"- base shape mismatch count: {report.get('base_load', {}).get('shape_mismatch_count', 'n/a')}",
            f"- smoke missing count: {report.get('smoke_load', {}).get('missing_count', 'n/a')}",
            f"- smoke missing all CADA: {report.get('smoke_load', {}).get('missing_all_cada', 'n/a')}",
            f"- smoke shape mismatch count: {report.get('smoke_load', {}).get('shape_mismatch_count', 'n/a')}",
            "",
            "## Equivalence",
            f"- output diffs: `{report.get('output_diffs', {})}`",
            f"- encoder output diffs: `{report.get('encoder_output_mean_abs_diffs', [])}`",
            f"- CADA gamma values: `{report.get('cada_gamma_values', [])}`",
            "",
            "## Gradients",
            f"- grad summary: `{report.get('grad_summary', {})}`",
            "",
            "## Notes",
            "- `use_cada=false` keeps the existing EMA_BIFPN path.",
            "- `use_cada=true` preserves core output fields: pred_logits, pred_boxes, pred_has_picking, pred_picking_offsets, aux_outputs, pre_outputs.",
            "- With `cada_gate_init=0`, adapter internals may have zero gradient in smoke; nonzero `cada_gamma` gradient is sufficient for this gate-zero check.",
        ]
    )
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "cada_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DECISION_DIR / "cada_smoke_decision.md").write_text(
        "\n".join(["# CADA V1 Smoke Decision", "", f"Decision: `{decision}`", ""]) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
