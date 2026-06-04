from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_ema_bifpn_qdpt_lite_v1.yml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "90_legacy_misc"
    / "encoder_experiments_archive"
    / "encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526"
    / "best_composite.pth"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "01_mainline_results"
    / "candidate_ema_bifpn_qdpt_lite_v1_probe20"
    / "smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test QDPT-Lite before training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--identity-warn-threshold", type=float, default=0.05)
    return parser.parse_args()


def module_state_dict(model):
    return dist_utils.de_parallel(model).state_dict()


def named_trainable(model):
    return {name for name, param in dist_utils.de_parallel(model).named_parameters() if param.requires_grad}


def grad_summary(model):
    allowed_with_grad = []
    frozen_with_grad = []
    no_grad_trainable = []
    for name, param in dist_utils.de_parallel(model).named_parameters():
        has_grad = param.grad is not None and torch.isfinite(param.grad).all().item()
        if param.requires_grad and has_grad:
            allowed_with_grad.append(name)
        elif param.requires_grad and param.grad is None:
            no_grad_trainable.append(name)
        elif (not param.requires_grad) and param.grad is not None:
            frozen_with_grad.append(name)
    return {
        "trainable_with_grad_count": len(allowed_with_grad),
        "trainable_without_grad_count": len(no_grad_trainable),
        "frozen_with_grad_count": len(frozen_with_grad),
        "sample_trainable_with_grad": allowed_with_grad[:30],
        "sample_trainable_without_grad": no_grad_trainable[:30],
        "sample_frozen_with_grad": frozen_with_grad[:30],
    }


def make_markdown(report: dict) -> str:
    lines = [
        "# QDPT-Lite Smoke Report",
        "",
        "## Status",
        f"- passed: `{report['passed']}`",
        f"- config: `{report['config']}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- device: `{report['device']}`",
        "",
        "## Interface",
    ]
    for key, value in report["interface"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Identity Check"])
    for key, value in report["identity"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Loss / Gradients"])
    for key, value in report["losses"].items():
        lines.append(f"- {key}: `{value}`")
    for key, value in report["gradients"].items():
        lines.append(f"- {key}: `{value}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings"])
        for item in report["warnings"]:
            lines.append(f"- {item}")
    if report.get("errors"):
        lines.extend(["", "## Errors"])
        for item in report["errors"]:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": False,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "interface": {},
        "identity": {},
        "losses": {},
        "gradients": {},
        "warnings": [],
        "errors": [],
    }

    if not args.config.exists():
        report["errors"].append(f"missing config: {args.config}")
    if not args.checkpoint.exists():
        report["errors"].append(f"missing checkpoint: {args.checkpoint}")
    if report["errors"]:
        args.output_dir.joinpath("qdp_lite_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        args.output_dir.joinpath("qdp_lite_smoke_report.md").write_text(make_markdown(report), encoding="utf-8")
        return 1

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    try:
        cfg = YAMLConfig(str(args.config), tuning=str(args.checkpoint), device=args.device, use_amp=False)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver._setup()
        model = solver.model
        teacher = solver.has_logit_teacher_model
        criterion = solver.criterion
        postprocessor = solver.postprocessor
        device = solver.device
        optimizer = cfg.optimizer
        trainable = named_trainable(model)
        report["interface"]["trainable_param_count"] = sum(
            p.numel() for _, p in dist_utils.de_parallel(model).named_parameters() if p.requires_grad
        )
        report["interface"]["trainable_name_count"] = len(trainable)
        report["interface"]["optimizer_param_groups"] = len(optimizer.param_groups)
        report["interface"]["teacher_present"] = teacher is not None

        loader = cfg.train_dataloader
        samples, targets = next(iter(loader))
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.eval()
        teacher.eval()
        with torch.no_grad():
            student_eval = model(samples)
            teacher_eval = teacher(samples)
        required = ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets"]
        for key in required:
            report["interface"][f"has_{key}"] = key in student_eval
        debug_keys = [
            "pred_picking_offsets_base",
            "pred_picking_offsets_qdpt_delta",
            "pred_picking_offsets_prior",
            "qdpt_gate",
            "qdpt_point_token_norm",
        ]
        for key in debug_keys:
            report["interface"][f"has_{key}"] = key in student_eval
        if hasattr(dist_utils.de_parallel(model).decoder, "_qdpt_runtime_warnings"):
            report["warnings"].extend(sorted(dist_utils.de_parallel(model).decoder._qdpt_runtime_warnings))

        offset_diff = (student_eval["pred_picking_offsets"] - teacher_eval["pred_picking_offsets"]).abs()
        report["identity"] = {
            "mean_abs_offset_diff": float(offset_diff.mean().item()),
            "max_abs_offset_diff": float(offset_diff.max().item()),
            "warn_threshold": float(args.identity_warn_threshold),
            "within_warn_threshold": bool(offset_diff.mean().item() <= args.identity_warn_threshold),
        }

        post_result = postprocessor(student_eval, torch.stack([t["orig_size"] for t in targets], dim=0))
        report["interface"]["postprocessor_result_count"] = len(post_result)
        report["interface"]["postprocessor_has_picking_points"] = bool(
            len(post_result) > 0 and "picking_points" in post_result[0]
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_train = teacher(samples)
        outputs = model(samples, targets=targets)
        loss_dict = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1, teacher_outputs=teacher_train)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        report["losses"] = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
        report["losses"]["total_loss"] = float(total_loss.detach().cpu().item())
        report["losses"]["has_logit_distill_nonzero"] = bool(report["losses"].get("loss_has_logit_distill", 0.0) > 0.0)
        report["gradients"] = grad_summary(model)
        teacher_has_grad = any(p.grad is not None for p in teacher.parameters())
        report["gradients"]["teacher_has_grad"] = bool(teacher_has_grad)

        bad_trainable = [
            name for name in trainable
            if not (
                "qdpt" in name
                or "picking_offset_head" in name
            )
        ]
        report["gradients"]["unexpected_trainable_params"] = bad_trainable[:50]
        finite_losses = all(math.isfinite(v) for v in report["losses"].values() if isinstance(v, float))
        report["passed"] = bool(
            all(report["interface"].get(f"has_{key}", False) for key in required)
            and report["interface"].get("postprocessor_has_picking_points", False)
            and report["losses"].get("has_logit_distill_nonzero", False)
            and report["gradients"].get("trainable_with_grad_count", 0) > 0
            and report["gradients"].get("frozen_with_grad_count", 0) == 0
            and not report["gradients"].get("teacher_has_grad", True)
            and not bad_trainable
            and finite_losses
        )
    except Exception as exc:
        report["errors"].append(repr(exc))
        report["passed"] = False
    finally:
        dist_utils.cleanup()

    args.output_dir.joinpath("qdp_lite_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_dir.joinpath("qdp_lite_smoke_report.md").write_text(make_markdown(report), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
