from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig


DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_enc_ema_bifpn_weighted_fusion.yml"
DEFAULT_EMA_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "90_legacy_misc"
    / "encoder_experiments_archive"
    / "encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526"
    / "best_composite.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "pald_cache_smoke_stage1"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_annotations(split: str) -> dict:
    ann_path = REPO_ROOT / "dataset" / split / "_annotations.grape_point.json"
    return json.loads(ann_path.read_text(encoding="utf-8"))


def split_image_rows(split: str, limit: int | None = None) -> list[dict]:
    data = load_annotations(split)
    rows = []
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    for image in data.get("images", []):
        rows.append({"split": split, "image": image, "annotations": anns_by_image.get(int(image["id"]), [])})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def count_split_images() -> dict[str, int]:
    counts = {}
    for split in ("train", "valid", "test"):
        try:
            counts[split] = len(load_annotations(split).get("images", []))
        except FileNotFoundError:
            counts[split] = 0
    return counts


def _pil_to_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = raw.view(size, size, 3).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    return tensor.unsqueeze(0).to(device)


def load_sample(row: dict, device: torch.device, image_size: int = 640) -> tuple[torch.Tensor, dict]:
    split = row["split"]
    image = row["image"]
    image_path = REPO_ROOT / "dataset" / split / image["file_name"]
    pil = Image.open(image_path)
    width, height = float(image.get("width", pil.width)), float(image.get("height", pil.height))
    sample = _pil_to_tensor(pil, image_size, device)

    labels, boxes, has_picking, picking_offsets, picking_points = [], [], [], [], []
    for ann in row.get("annotations", []):
        x, y, w, h = [float(v) for v in ann["bbox"]]
        labels.append(0)
        boxes.append([(x + 0.5 * w) / width, (y + 0.5 * h) / height, w / width, h / height])
        visible = bool(ann.get("has_picking", bool(ann.get("picking_point"))))
        has_picking.append(1.0 if visible else 0.0)
        point = ann.get("picking_point") or [float("nan"), float("nan")]
        picking_points.append([float(point[0]) / width, float(point[1]) / height] if visible else [0.0, 0.0])
        if visible and w > 0 and h > 0:
            px, py = float(point[0]), float(point[1])
            top_anchor_y = y + 0.12 * h
            picking_offsets.append([(px - (x + 0.5 * w)) / w, (py - top_anchor_y) / h])
        else:
            picking_offsets.append([0.0, 0.0])

    target = {
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "has_picking": torch.tensor(has_picking, dtype=torch.float32, device=device),
        "picking_offsets": torch.tensor(picking_offsets, dtype=torch.float32, device=device),
        "picking_points": torch.tensor(picking_points, dtype=torch.float32, device=device),
        "orig_size": torch.tensor([height, width], dtype=torch.float32, device=device),
        "size": torch.tensor([image_size, image_size], dtype=torch.float32, device=device),
        "image_id": torch.tensor([int(image["id"])], dtype=torch.int64, device=device),
        "file_name": image["file_name"],
        "image_path": str(image_path),
    }
    return sample, target


def set_no_pretrained(cfg: YAMLConfig) -> None:
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False


def load_checkpoint_for_tuning(model: torch.nn.Module, checkpoint: Path) -> dict:
    state = torch.load(str(checkpoint), map_location="cpu")
    source = state["ema"]["module"] if isinstance(state, dict) and "ema" in state else state["model"]
    if any(key.startswith("module.") for key in source):
        source = {key[7:] if key.startswith("module.") else key: value for key, value in source.items()}
    model_state = model.state_dict()
    matched = {}
    missing = []
    shape_mismatch = []
    for key, value in model_state.items():
        if key not in source:
            missing.append(key)
        elif tuple(value.shape) != tuple(source[key].shape):
            shape_mismatch.append(key)
        else:
            matched[key] = source[key]
    model.load_state_dict(matched, strict=False)
    return {"matched_count": len(matched), "missing": missing, "shape_mismatch": shape_mismatch}


def load_ema_model(config: Path, checkpoint: Path, device: torch.device, freeze: bool = True) -> tuple[YAMLConfig, torch.nn.Module, dict]:
    cfg = YAMLConfig(str(config), device=str(device), use_amp=False)
    set_no_pretrained(cfg)
    model = cfg.model.to(device)
    info = load_checkpoint_for_tuning(model, checkpoint)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return cfg, model, info


def internal_features(model: torch.nn.Module, samples: torch.Tensor) -> list[torch.Tensor]:
    backbone_feats = model.backbone(samples)
    encoder_output = model.encoder(backbone_feats)
    if isinstance(encoder_output, tuple):
        encoder_output = encoder_output[0]
    return list(encoder_output)


def decoder_proj_features(model: torch.nn.Module, samples: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    fpn_features = internal_features(model, samples)
    memory, spatial_shapes, proj_feats = model.decoder._get_encoder_input(fpn_features)
    return proj_feats, memory, spatial_shapes


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.shape]


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def estimate_feature_gib(shape: Iterable[int], image_count: int, dtype_bytes: int = 4) -> float:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return float(numel * image_count * dtype_bytes) / float(1024 ** 3)


def visible_match_indices(cfg: YAMLConfig, outputs: dict, target: dict) -> tuple[torch.Tensor, torch.Tensor]:
    matcher = cfg.criterion.matcher
    match_result = matcher(outputs, [target], allow_point_cost=False)
    indices = match_result["indices"] if isinstance(match_result, dict) else match_result
    pred_idx, gt_idx = indices[0]
    pred_idx = pred_idx.to(device=target["has_picking"].device)
    gt_idx = gt_idx.to(device=target["has_picking"].device)
    visible = target["has_picking"][gt_idx] > 0.5
    return pred_idx[visible], gt_idx[visible]


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)
