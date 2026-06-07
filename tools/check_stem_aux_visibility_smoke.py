from __future__ import annotations

import argparse
import csv
import json
import py_compile
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS  # noqa: F401 - register model/dataset components


BASE_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_ema_bifpn_new1804_fair100.yml"
SMOKE_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_ema_bifpn_new1804_stem_aux_visibility_v1_smoke.yml"
OUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "stem_aux_visibility_v1_smoke"
DECISIONS = {
    "ready_for_stem_aux_visibility_probe20",
    "blocked_by_missing_has_stem",
    "blocked_by_dataset_field_issue",
    "blocked_by_no_stem_grad",
    "blocked_by_mainline_equivalence_failure",
    "blocked_by_other_smoke_failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check STEM_AUX_VISIBILITY_V1 without training.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--smoke-config", type=Path, default=SMOKE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {k: tensor_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_shape(v) for v in value[:3]]
    return str(type(value).__name__)


def all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    return True


def make_cfg(config_path: Path, device: str) -> YAMLConfig:
    return YAMLConfig(
        str(config_path),
        train_dataloader={"total_batch_size": 1, "num_workers": 0, "drop_last": False},
        val_dataloader={"total_batch_size": 1, "num_workers": 0, "drop_last": False},
        use_amp=False,
        use_ema=False,
        device=device,
    )


def move_targets(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in target.items()}
        for target in targets
    ]


def write_trainable_csv(path: Path, model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, param in model.named_parameters():
        is_stem = "stem" in name
        rows.append(
            {
                "name": name,
                "shape": "x".join(str(v) for v in param.shape),
                "requires_grad": bool(param.requires_grad),
                "is_stem_aux": bool(is_stem),
                "numel": int(param.numel()),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def safe_imports(status: dict[str, Any]) -> None:
    from engine.rtv4.postprocessor import PostProcessor  # noqa: F401
    import tools.make_grape_point_report  # noqa: F401
    import tools.export_grape_point_predictions  # noqa: F401

    status["checks"]["postprocessor_import"] = True
    status["checks"]["make_grape_point_report_import"] = True
    status["checks"]["export_grape_point_predictions_import"] = True


def build_probe_plan(out_dir: Path) -> None:
    text = """# STEM_AUX_VISIBILITY_V1 Probe20 Plan

Next experiment: `EMA_BIFPN_NEW1804_STEM_AUX_VISIBILITY_V1_PROBE20`

Suggested output directory:
`outputs/01_mainline_results/candidate_stem_aux_visibility_v1_probe20/`

Scope:
- Start from the unified NEW1804 EMA_BIFPN mainline configuration.
- Enable only `use_stem_aux=true`.
- Keep matcher, postprocessor, picking offset decoding, dataset, and `has_picking_threshold` unchanged.
- Train 20 epochs as a mechanism probe, not fair100.

Probe success gate:
- `F1 >= 0.8780`
- `pair >= 234`
- `mean L2 <= 13.63`
- `PPL-SR@30 >= 0.8967`
- `AP >= 0.620`

Decision rule:
- If F1/pair improve and mean L2/PPL-SR@30 are basically preserved, promote to fair100.
- If coverage does not improve or point accuracy drops clearly, stop V1 and keep it as a diagnostic auxiliary route.
"""
    (out_dir / "stem_aux_visibility_v1_probe_plan.md").write_text(text, encoding="utf-8")


def write_reports(out_dir: Path, status: dict[str, Any], decision: str, batch_preview: dict[str, Any]) -> None:
    if decision not in DECISIONS:
        decision = "blocked_by_other_smoke_failure"
    write_json(out_dir / "stem_aux_visibility_smoke_summary.json", status)
    write_json(out_dir / "stem_aux_visibility_smoke_decision.json", {"decision": decision})
    write_json(out_dir / "stem_aux_batch_preview.json", batch_preview)

    md = [
        "# STEM_AUX_VISIBILITY_V1 Smoke Report",
        "",
        f"- Passed: `{status['passed']}`",
        f"- Decision: `{decision}`",
        f"- Device: `{status['device']}`",
        f"- Base config: `{status['base_config']}`",
        f"- Smoke config: `{status['smoke_config']}`",
        "",
        "## Checks",
    ]
    for key, value in status.get("checks", {}).items():
        md.append(f"- {key}: `{value}`")
    if status.get("shape_checks"):
        md.extend(["", "## Shape Checks"])
        for key, value in status["shape_checks"].items():
            md.append(f"- {key}: `{value}`")
    if status.get("loss_keys"):
        md.extend(["", "## Loss Keys"])
        for key in status["loss_keys"]:
            md.append(f"- `{key}`")
    if status.get("errors"):
        md.extend(["", "## Errors"])
        for err in status["errors"]:
            md.append(f"- {err}")
    (out_dir / "stem_aux_visibility_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    decision_md = [
        "# STEM_AUX_VISIBILITY_V1 Smoke Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Answers",
        f"- Dataset has `has_stem`: `{status['checks'].get('dataset_has_stem', False)}`",
        f"- `pred_has_stem` forward field present: `{status['checks'].get('pred_has_stem_present', False)}`",
        f"- `loss_has_stem` finite: `{status['checks'].get('loss_has_stem_finite', False)}`",
        f"- Stem head gradients present: `{status['checks'].get('stem_head_grad', False)}`",
        f"- Core output shapes unchanged: `{status['checks'].get('core_shapes_unchanged', False)}`",
        "",
        "No training, checkpoint, test evaluation, matcher change, postprocessor change, or dataset file edit was performed.",
    ]
    (out_dir / "stem_aux_visibility_smoke_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")
    if decision == "ready_for_stem_aux_visibility_probe20":
        build_probe_plan(out_dir)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    status: dict[str, Any] = {
        "passed": False,
        "decision": "blocked_by_other_smoke_failure",
        "device": device_name,
        "base_config": str(args.base_config.resolve()),
        "smoke_config": str(args.smoke_config.resolve()),
        "checks": {},
        "shape_checks": {},
        "errors": [],
    }
    batch_preview: dict[str, Any] = {}
    decision = "blocked_by_other_smoke_failure"

    try:
        dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)

        for rel in [
            "engine/data/dataset/coco_dataset.py",
            "engine/data/dataset/grape_point_dataset.py",
            "engine/rtv4/dfine_decoder.py",
            "engine/rtv4/rtv4_criterion.py",
            "tools/check_stem_aux_visibility_smoke.py",
        ]:
            py_compile.compile(str(REPO_ROOT / rel), doraise=True)
        status["checks"]["py_compile"] = True

        base_cfg = make_cfg(args.base_config.resolve(), device_name)
        smoke_cfg = make_cfg(args.smoke_config.resolve(), device_name)
        status["checks"]["base_config_load"] = True
        status["checks"]["smoke_config_load"] = True

        loader = smoke_cfg.train_dataloader
        images, targets = next(iter(loader))
        status["checks"]["dataset_batch"] = True
        if not targets or any("has_stem" not in t for t in targets):
            decision = "blocked_by_missing_has_stem"
            raise RuntimeError("dataset target is missing has_stem")
        has_stem_pos = sum(float(t["has_stem"].sum().item()) for t in targets)
        has_picking_pos = sum(float(t.get("has_picking", torch.zeros_like(t["has_stem"])).sum().item()) for t in targets)
        if has_stem_pos <= 0:
            decision = "blocked_by_dataset_field_issue"
            raise RuntimeError("sampled smoke batch has no positive has_stem target")
        status["checks"]["dataset_has_stem"] = True
        status["checks"]["has_stem_positive_batch"] = True
        batch_preview = {
            "images": tensor_shape(images),
            "target_count": len(targets),
            "target_0": {k: tensor_shape(targets[0].get(k)) for k in ["boxes", "labels", "has_picking", "has_stem", "picking_offsets", "picking_points"]},
            "batch_has_stem_positive": has_stem_pos,
            "batch_has_picking_positive": has_picking_pos,
        }

        images = images.to(device)
        targets_device = move_targets(targets, device)
        base_model = base_cfg.model.to(device)
        base_model.train()
        with torch.no_grad():
            base_outputs = base_model(images, targets=targets_device)
        base_core_shapes = {key: list(base_outputs[key].shape) for key in ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets"] if key in base_outputs}
        base_has_stem = "pred_has_stem" in base_outputs
        del base_outputs
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        smoke_model = smoke_cfg.model.to(device)
        smoke_criterion = smoke_cfg.criterion.to(device)
        smoke_model.train()
        smoke_criterion.train()
        smoke_outputs = smoke_model(images, targets=targets_device)
        core_fields = ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets", "aux_outputs", "pre_outputs"]
        missing_base = [key for key in core_fields[:4] if key not in base_core_shapes]
        missing_smoke = [key for key in core_fields if key not in smoke_outputs]
        if missing_base or missing_smoke:
            decision = "blocked_by_mainline_equivalence_failure"
            raise RuntimeError(f"missing core fields base={missing_base}, smoke={missing_smoke}")
        if base_has_stem:
            decision = "blocked_by_mainline_equivalence_failure"
            raise RuntimeError("use_stem_aux=false base output unexpectedly contains pred_has_stem")
        if "pred_has_stem" not in smoke_outputs:
            decision = "blocked_by_other_smoke_failure"
            raise RuntimeError("use_stem_aux=true output is missing pred_has_stem")
        status["checks"]["pred_has_stem_present"] = True
        status["checks"]["forward_base"] = True
        status["checks"]["forward_smoke"] = True
        for key in core_fields[:4]:
            same_shape = base_core_shapes[key] == list(smoke_outputs[key].shape)
            status["shape_checks"][key] = {
                "base": base_core_shapes[key],
                "smoke": list(smoke_outputs[key].shape),
                "same": same_shape,
            }
            if not same_shape:
                decision = "blocked_by_mainline_equivalence_failure"
                raise RuntimeError(f"core shape changed for {key}")
        expected_stem_shape = list(smoke_outputs["pred_has_picking"].shape)
        actual_stem_shape = list(smoke_outputs["pred_has_stem"].shape)
        status["shape_checks"]["pred_has_stem"] = {"expected": expected_stem_shape, "actual": actual_stem_shape}
        if actual_stem_shape != expected_stem_shape:
            decision = "blocked_by_other_smoke_failure"
            raise RuntimeError("pred_has_stem shape mismatch")
        status["checks"]["core_shapes_unchanged"] = True
        if not all_finite(smoke_outputs):
            raise RuntimeError("smoke model output contains NaN/Inf")
        status["checks"]["outputs_finite"] = True

        loss_dict = smoke_criterion(smoke_outputs, targets_device, epoch=0)
        status["loss_keys"] = sorted(loss_dict.keys())
        for key in ["loss_has_stem", "loss_has_picking", "loss_picking_offset"]:
            if key not in loss_dict:
                raise RuntimeError(f"{key} missing from criterion output")
            if not all_finite(loss_dict[key]):
                raise RuntimeError(f"{key} is NaN/Inf")
            status["checks"][f"{key}_finite"] = True
        total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        if not all_finite(total_loss):
            raise RuntimeError("total loss is NaN/Inf")
        total_loss.backward()
        status["checks"]["one_batch_backward"] = True
        stem_grads = [
            name for name, param in smoke_model.named_parameters()
            if "stem" in name and param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all()
        ]
        status["stem_grad_param_count"] = len(stem_grads)
        status["stem_grad_param_sample"] = stem_grads[:10]
        if not stem_grads:
            decision = "blocked_by_no_stem_grad"
            raise RuntimeError("no finite gradient found on stem head parameters")
        status["checks"]["stem_head_grad"] = True

        trainable_rows = write_trainable_csv(out_dir / "stem_aux_trainable_params.csv", smoke_model)
        status["trainable_param_count"] = int(sum(row["numel"] for row in trainable_rows if row["requires_grad"]))
        status["stem_param_count"] = int(sum(row["numel"] for row in trainable_rows if row["is_stem_aux"]))
        safe_imports(status)
        status["passed"] = True
        decision = "ready_for_stem_aux_visibility_probe20"
    except Exception as exc:
        if decision == "blocked_by_other_smoke_failure" and "has_stem" in str(exc):
            decision = "blocked_by_missing_has_stem"
        status["errors"].append(str(exc))
        status["traceback"] = traceback.format_exc()
    finally:
        status["decision"] = decision
        write_reports(out_dir, status, decision, batch_preview)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist_utils.cleanup()
    if not status["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
