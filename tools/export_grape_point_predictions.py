from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils


DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_main.yml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "debug" / "grape_point_predictions.json"


def xyxy_to_xywh(box_xyxy: torch.Tensor) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def select_checkpoint_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    ema = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return clean_state_dict(ema["module"])
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return clean_state_dict(checkpoint["model"])
    if isinstance(checkpoint, dict):
        return clean_state_dict(checkpoint)
    raise TypeError("Unsupported checkpoint format; expected a dict-like state.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GPPoint-DETR per-prediction JSON for ROI hit-rate analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--legacy-list-output", action="store_true", help="Write the old top-level list format.")
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required. The exporter no longer defaults to an old checkpoint; "
            "pass an explicit .pth file to keep experiment provenance clear."
        )
    checkpoint_path = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    split_dir = dataset_root / args.split
    ann_path = split_dir / "_annotations.grape_point.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    dist_utils.setup_distributed(seed=0)
    predictions: list[dict] = []

    try:
        with tempfile.TemporaryDirectory(prefix="prediction_export_", dir=str(REPO_ROOT / "outputs")) as tmp_dir:
            cfg = YAMLConfig(
                str(config_path),
                device=args.device,
                use_amp=False,
                output_dir=tmp_dir,
            )
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(split_dir.as_posix())
            cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = str(ann_path.as_posix())
            cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = int(args.batch_size)
            cfg.yaml_cfg["val_dataloader"]["num_workers"] = int(args.num_workers)

            device = torch.device(args.device)
            model = cfg.model.to(device)
            postprocessor = cfg.postprocessor.to(device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state = select_checkpoint_state(checkpoint)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint is not compatible with config. missing={len(missing)}, unexpected={len(unexpected)}"
                )

            model.eval()
            postprocessor.eval()
            has_picking_threshold = float(getattr(postprocessor, "has_picking_threshold", 0.5))
            data_loader = cfg.val_dataloader

            for samples, targets in data_loader:
                samples = samples.to(device)
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                outputs = model(samples)
                orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
                results = postprocessor(outputs, orig_target_sizes)

                for target, result in zip(targets, results):
                    image_id = int(target["image_id"].item())
                    boxes = result.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu().to(torch.float32)
                    scores = result.get("scores", torch.zeros((boxes.shape[0],), device=device)).detach().cpu().to(torch.float32)
                    labels = result.get("labels", torch.zeros((boxes.shape[0],), dtype=torch.int64, device=device)).detach().cpu()
                    raw_has_scores = result.get("raw_has_picking_scores")
                    has_scores = result.get("has_picking_scores")
                    visible_scores = result.get("visible_scores")
                    has_flags = result.get("has_picking")
                    points = result.get("picking_points")
                    quality_scores = result.get("point_quality_scores")
                    final_scores = result.get("point_final_scores")
                    accept_scores = result.get("point_accept_scores")
                    accept_final_scores = result.get("point_accept_final_scores")
                    reliability_scores = result.get("point_reliability_scores")
                    reliability_final_scores = result.get("point_reliability_final_scores")
                    weak_heatmap_scores = result.get("weak_heatmap_scores")
                    has_quality_scores = quality_scores is not None
                    has_accept_scores = accept_scores is not None
                    has_reliability_scores = reliability_scores is not None
                    has_weak_heatmap_scores = weak_heatmap_scores is not None
                    if raw_has_scores is None:
                        raw_has_scores = has_scores
                    if visible_scores is None:
                        visible_scores = has_scores
                    if has_scores is None:
                        has_scores = torch.zeros((boxes.shape[0],), dtype=torch.float32)
                    else:
                        has_scores = has_scores.detach().cpu().to(torch.float32)
                    if raw_has_scores is None:
                        raw_has_scores = has_scores
                    else:
                        raw_has_scores = raw_has_scores.detach().cpu().to(torch.float32)
                    if visible_scores is None:
                        visible_scores = has_scores
                    else:
                        visible_scores = visible_scores.detach().cpu().to(torch.float32)
                    if has_flags is None:
                        has_flags = visible_scores >= has_picking_threshold
                    else:
                        has_flags = has_flags.detach().cpu().to(torch.bool)
                    if points is None:
                        points = torch.zeros((boxes.shape[0], 2), dtype=torch.float32)
                    else:
                        points = points.detach().cpu().to(torch.float32)
                    if quality_scores is None:
                        quality_scores = torch.ones((boxes.shape[0],), dtype=torch.float32)
                    else:
                        quality_scores = quality_scores.detach().cpu().to(torch.float32)
                    if final_scores is None:
                        final_scores = has_scores * quality_scores
                    else:
                        final_scores = final_scores.detach().cpu().to(torch.float32)
                    if accept_scores is None:
                        accept_scores = torch.zeros((boxes.shape[0],), dtype=torch.float32)
                    else:
                        accept_scores = accept_scores.detach().cpu().to(torch.float32)
                    if accept_final_scores is None:
                        accept_final_scores = has_scores * accept_scores
                    else:
                        accept_final_scores = accept_final_scores.detach().cpu().to(torch.float32)
                    if reliability_scores is None:
                        reliability_scores = torch.zeros((boxes.shape[0],), dtype=torch.float32)
                    else:
                        reliability_scores = reliability_scores.detach().cpu().to(torch.float32)
                    if reliability_final_scores is None:
                        reliability_final_scores = has_scores * reliability_scores
                    else:
                        reliability_final_scores = reliability_final_scores.detach().cpu().to(torch.float32)
                    if weak_heatmap_scores is None:
                        weak_heatmap_scores = torch.zeros((boxes.shape[0],), dtype=torch.float32)
                    else:
                        weak_heatmap_scores = weak_heatmap_scores.detach().cpu().to(torch.float32)

                    for idx in range(boxes.shape[0]):
                        score = float(scores[idx].item())
                        if score < args.score_threshold:
                            continue
                        item = {
                            "image_id": image_id,
                            "category_id": int(labels[idx].item()),
                            "bbox": xyxy_to_xywh(boxes[idx]),
                            "score": score,
                            "raw_has_picking_score": float(raw_has_scores[idx].item()),
                            "has_picking_score": float(has_scores[idx].item()),
                            "visible_score": float(visible_scores[idx].item()),
                            "has_picking": bool(has_flags[idx].item()),
                            "picking_point": [float(v) for v in points[idx].tolist()],
                        }
                        if has_quality_scores:
                            item["point_quality_score"] = float(quality_scores[idx].item())
                            item["final_score"] = float(final_scores[idx].item())
                        if has_accept_scores:
                            item["point_accept_score"] = float(accept_scores[idx].item())
                            item["point_accept_final_score"] = float(accept_final_scores[idx].item())
                        if has_reliability_scores:
                            item["point_reliability_score"] = float(reliability_scores[idx].item())
                            item["point_reliability_final_score"] = float(reliability_final_scores[idx].item())
                        if has_weak_heatmap_scores:
                            item["weak_heatmap_score"] = float(weak_heatmap_scores[idx].item())
                        predictions.append(item)
    finally:
        dist_utils.cleanup()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.legacy_list_output:
        payload = predictions
    else:
        payload = {
            "metadata": {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "score_threshold": float(args.score_threshold),
                "has_picking_threshold": float(locals().get("has_picking_threshold", 0.5)),
                "num_predictions": int(len(predictions)),
            },
            "predictions": predictions,
        }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
