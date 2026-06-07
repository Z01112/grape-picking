from __future__ import annotations

import json
import math
import sys
import traceback
import argparse
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS  # noqa: F401 - imports registered solver/model modules


CONFIG_PATH = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_ema_bifpn_new1804_fair100.yml"
OUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "new1804_training_smoke"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, list):
        return [tensor_shape(v) for v in value[:3]]
    if isinstance(value, dict):
        return {k: tensor_shape(v) for k, v in value.items()}
    return str(type(value).__name__)


def is_finite_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value.detach()).all().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-batch smoke check for NEW1804 grape-point training configs.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "passed": False,
        "config": str(config_path),
        "checks": {},
        "errors": [],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    batch_preview: dict[str, Any] = {}
    try:
        dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
        cfg = YAMLConfig(
            str(config_path),
            train_dataloader={"total_batch_size": 1, "num_workers": 0, "drop_last": False},
            val_dataloader={"total_batch_size": 1, "num_workers": 0, "drop_last": False},
            use_amp=False,
            device=status["device"],
        )
        status["checks"]["yaml_config_load"] = True
        train_loader = cfg.train_dataloader
        status["checks"]["dataset_build"] = True
        images, targets = next(iter(train_loader))
        status["checks"]["dataloader_one_batch"] = True
        required_target_keys = ["boxes", "labels", "has_picking", "picking_offsets", "picking_points"]
        missing_keys = sorted({key for target in targets for key in required_target_keys if key not in target})
        if missing_keys:
            raise RuntimeError(f"missing target keys: {missing_keys}")
        status["checks"]["target_fields_present"] = True
        batch_preview = {
            "images": tensor_shape(images),
            "target_count": len(targets),
            "target_0": {key: tensor_shape(targets[0].get(key)) for key in required_target_keys},
            "target_0_has_picking_sum": float(targets[0]["has_picking"].sum().item()) if len(targets) else 0.0,
        }
        device = torch.device(status["device"])
        model = cfg.model.to(device)
        criterion = cfg.criterion.to(device)
        images = images.to(device)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        model.train()
        criterion.train()
        outputs = model(images, targets=targets)
        core_fields = ["pred_logits", "pred_boxes", "pred_has_picking", "pred_picking_offsets", "aux_outputs", "pre_outputs"]
        missing_outputs = [key for key in core_fields if key not in outputs]
        if missing_outputs:
            raise RuntimeError(f"missing model output fields: {missing_outputs}")
        status["checks"]["model_forward"] = True
        loss_dict = criterion(outputs, targets, epoch=0)
        status["loss_keys"] = sorted(loss_dict.keys())
        if "loss_has_picking" not in loss_dict:
            raise RuntimeError("loss_has_picking missing from criterion output")
        if not is_finite_tensor(loss_dict["loss_has_picking"]):
            raise RuntimeError("loss_has_picking is not finite")
        status["checks"]["loss_has_picking_finite"] = True
        if any(float(t["has_picking"].sum().item()) > 0.0 for t in targets):
            if "loss_picking_offset" not in loss_dict:
                raise RuntimeError("loss_picking_offset missing from criterion output")
            if not is_finite_tensor(loss_dict["loss_picking_offset"]):
                raise RuntimeError("loss_picking_offset is not finite")
            status["checks"]["loss_picking_offset_finite"] = True
        total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        if not torch.isfinite(total_loss.detach()).all():
            raise RuntimeError("total loss is not finite")
        total_loss.backward()
        status["checks"]["one_batch_backward"] = True
        import tools.make_grape_point_report  # noqa: F401
        import tools.export_grape_point_predictions  # noqa: F401
        from engine.rtv4.postprocessor import PostProcessor  # noqa: F401
        status["checks"]["imports"] = True
        status["passed"] = True
    except Exception as exc:
        status["errors"].append(str(exc))
        status["traceback"] = traceback.format_exc()
    finally:
        write_json(out_dir / "training_smoke_summary.json", status)
        write_json(out_dir / "batch_preview_shapes.json", batch_preview)
        md = [
            "# New1804 Training Smoke Report",
            "",
            f"- Passed: `{status['passed']}`",
            f"- Config: `{config_path}`",
            f"- Device: `{status['device']}`",
            "",
            "## Checks",
        ]
        for key, value in status.get("checks", {}).items():
            md.append(f"- {key}: `{value}`")
        if status.get("loss_keys"):
            md.extend(["", "## Loss Keys", *[f"- {key}" for key in status["loss_keys"]]])
        if status["errors"]:
            md.extend(["", "## Errors", *[f"- {err}" for err in status["errors"]]])
        (out_dir / "training_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist_utils.cleanup()
    if not status["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
