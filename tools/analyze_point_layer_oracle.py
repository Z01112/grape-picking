from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.misc import MetricLogger, dist_utils
from engine.rtv4.box_ops import box_iou
from engine.rtv4.point_utils import absolute_points_from_boxes_and_offsets
from engine.solver import TASKS


DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_enc_ema_bifpn_weighted_fusion.yml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "02_encoder_experiments"
    / "encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526"
    / "best_composite.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "22_point_lsd_layer_oracle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether decoder aux/pre layers contain better grape picking offsets than final output."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--fix-threshold-px", type=float, default=30.0)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def read_image_info(ann_path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in payload.get("images", [])}


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    return torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")


def make_matcher_targets(targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    matcher_targets: list[dict[str, torch.Tensor]] = []
    for target in targets:
        boxes_xyxy = target["boxes"].to(torch.float32)
        size_hw = target.get("size")
        if size_hw is None:
            size_hw = target.get("orig_size")
        size_hw = size_hw.to(device=boxes_xyxy.device, dtype=torch.float32)
        width = size_hw[1].clamp(min=1.0)
        height = size_hw[0].clamp(min=1.0)
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
        denom = torch.stack((width, height, width, height))
        boxes_cxcywh = boxes_cxcywh / denom
        matcher_targets.append(
            {
                "boxes": boxes_cxcywh,
                "labels": target["labels"],
                "has_picking": target.get("has_picking", torch.zeros((boxes_xyxy.shape[0],), device=boxes_xyxy.device)),
                "picking_points": target.get("picking_points", torch.zeros((boxes_xyxy.shape[0], 2), device=boxes_xyxy.device)),
                "picking_offsets": target.get("picking_offsets", torch.zeros((boxes_xyxy.shape[0], 2), device=boxes_xyxy.device)),
                "size": size_hw,
                "orig_size": target.get("orig_size", size_hw),
            }
        )
    return matcher_targets


def target_boxes_to_original_xyxy(target: dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = target["boxes"].to(torch.float32)
    size_hw = target.get("size")
    orig_wh = target.get("orig_size")
    if size_hw is None or orig_wh is None:
        return boxes
    size_hw = size_hw.to(device=boxes.device, dtype=torch.float32)
    orig_wh = orig_wh.to(device=boxes.device, dtype=torch.float32)
    scale_x = orig_wh[0].clamp(min=1.0) / size_hw[1].clamp(min=1.0)
    scale_y = orig_wh[1].clamp(min=1.0) / size_hw[0].clamp(min=1.0)
    scale = torch.stack((scale_x, scale_y, scale_x, scale_y))
    return boxes * scale


def decode_points_for_layer(
    boxes_cxcywh: torch.Tensor,
    offsets: torch.Tensor,
    orig_size_wh: torch.Tensor,
    point_offset_mode: str,
    point_top_anchor_ratio: float,
    point_anchor_x_ratio: float,
) -> torch.Tensor:
    abs_boxes = boxes_cxcywh.to(torch.float32) * orig_size_wh.to(torch.float32).repeat(2).view(1, 1, 4)
    return absolute_points_from_boxes_and_offsets(
        abs_boxes,
        offsets.to(torch.float32),
        mode=point_offset_mode,
        top_anchor_ratio=point_top_anchor_ratio,
        anchor_x_ratio=point_anchor_x_ratio,
    )


@contextmanager
def decoder_aux_output_mode(model: torch.nn.Module):
    module = dist_utils.de_parallel(model)
    decoder = getattr(module, "decoder", None)
    inner_decoder = getattr(decoder, "decoder", None) if decoder is not None else None
    old_outer_training = bool(getattr(decoder, "training", False)) if decoder is not None else False
    old_inner_training = bool(getattr(inner_decoder, "training", False)) if inner_decoder is not None else False
    old_num_denoising = getattr(decoder, "num_denoising", None) if decoder is not None else None
    try:
        if decoder is not None:
            decoder.training = True
            if hasattr(decoder, "num_denoising"):
                decoder.num_denoising = 0
        if inner_decoder is not None:
            inner_decoder.training = True
        yield
    finally:
        if decoder is not None:
            decoder.training = old_outer_training
            if old_num_denoising is not None:
                decoder.num_denoising = old_num_denoising
        if inner_decoder is not None:
            inner_decoder.training = old_inner_training


def layer_outputs(outputs: dict[str, Any]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    layers: list[tuple[str, dict[str, torch.Tensor]]] = [("final", outputs)]
    for idx, item in enumerate(outputs.get("aux_outputs", []) or []):
        if "pred_picking_offsets" in item and "pred_boxes" in item:
            layers.append((f"aux_{idx}", item))
    pre = outputs.get("pre_outputs")
    if isinstance(pre, dict) and "pred_picking_offsets" in pre and "pred_boxes" in pre:
        layers.append(("pre", pre))
    return layers


def l2_px(point_xy: torch.Tensor, gt_xy: torch.Tensor) -> float:
    delta = point_xy.to(torch.float32) - gt_xy.to(torch.float32)
    return float(torch.linalg.vector_norm(delta).item())


def summarize_rows(rows: list[dict[str, Any]], iou_threshold: float, fix_threshold_px: float) -> dict[str, Any]:
    def compute(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {
                "visible_matched_pair_count": 0,
                "final_mean_l2": None,
                "best_layer_oracle_mean_l2": None,
                "final_ppl_sr_30": None,
                "best_layer_oracle_ppl_sr_30": None,
                "final_l2_gt30_count": 0,
                "best_layer_oracle_l2_gt30_count": 0,
                "final_is_best_ratio": None,
                "improvement_mean": None,
                "improvement_median": None,
                "l2_gt30_fixed_to_l2_le30_ratio": None,
                "best_layer_counts": {},
                "layer_mean_l2": {},
            }
        final_l2 = [safe_float(row["final_l2"]) for row in subset]
        best_l2 = [safe_float(row["best_layer_l2"]) for row in subset]
        improvements = [safe_float(row["improvement"]) for row in subset]
        final_gt30 = [row for row in subset if safe_float(row["final_l2"]) > fix_threshold_px]
        fixed = [row for row in final_gt30 if safe_float(row["best_layer_l2"]) <= fix_threshold_px]
        best_layer_counts: dict[str, int] = {}
        layer_values: dict[str, list[float]] = {}
        for row in subset:
            best_name = str(row.get("best_layer_name", "unknown"))
            best_layer_counts[best_name] = best_layer_counts.get(best_name, 0) + 1
            for layer_name, value in (row.get("layer_l2") or {}).items():
                layer_values.setdefault(str(layer_name), []).append(safe_float(value))
        layer_mean_l2 = {
            layer_name: float(sum(values) / len(values))
            for layer_name, values in sorted(layer_values.items())
            if values
        }
        return {
            "visible_matched_pair_count": len(subset),
            "final_mean_l2": float(sum(final_l2) / len(final_l2)),
            "best_layer_oracle_mean_l2": float(sum(best_l2) / len(best_l2)),
            "final_ppl_sr_30": float(sum(1 for v in final_l2 if v <= fix_threshold_px) / len(final_l2)),
            "best_layer_oracle_ppl_sr_30": float(sum(1 for v in best_l2 if v <= fix_threshold_px) / len(best_l2)),
            "final_l2_gt30_count": int(len(final_gt30)),
            "best_layer_oracle_l2_gt30_count": int(sum(1 for v in best_l2 if v > fix_threshold_px)),
            "final_is_best_ratio": float(sum(1 for row in subset if bool(row["whether_final_is_best"])) / len(subset)),
            "improvement_mean": float(sum(improvements) / len(improvements)),
            "improvement_median": float(torch.median(torch.tensor(improvements, dtype=torch.float32)).item()),
            "l2_gt30_fixed_to_l2_le30_ratio": float(len(fixed) / len(final_gt30)) if final_gt30 else 0.0,
            "best_layer_counts": dict(sorted(best_layer_counts.items())),
            "layer_mean_l2": layer_mean_l2,
        }

    iou50_rows = [row for row in rows if safe_float(row.get("final_iou", 0.0)) >= iou_threshold]
    out = compute(rows)
    out["iou50_subset"] = compute(iou50_rows)
    out["iou50_visible_matched_pair_count"] = len(iou50_rows)
    return out


def row_to_csv(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key, value in list(out.items()):
        if isinstance(value, (list, tuple, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            out[key] = int(value)
    return out


@torch.no_grad()
def analyze_split(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    dataset_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    iou_threshold: float,
    fix_threshold_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix=f"layer_oracle_{split}_", dir=str(output_dir)) as tmp_dir:
        cfg = YAMLConfig(
            str(config_path),
            resume=str(checkpoint_path),
            device=device,
            use_amp=False,
            output_dir=tmp_dir,
        )
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        split_dir = (dataset_root / split).resolve()
        ann_path = split_dir / "_annotations.grape_point.json"
        image_info = read_image_info(ann_path)
        cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(split_dir.as_posix())
        cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = str(ann_path.as_posix())
        cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = batch_size
        cfg.yaml_cfg["val_dataloader"]["num_workers"] = num_workers

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        model = solver.ema.module if solver.ema else solver.model
        criterion = solver.criterion
        matcher = criterion.matcher
        data_loader = solver.val_dataloader
        eval_device = solver.device
        point_offset_mode = str(getattr(solver.postprocessor, "point_offset_mode", "top_center"))
        point_top_anchor_ratio = float(getattr(solver.postprocessor, "point_top_anchor_ratio", 0.12))
        point_anchor_x_ratio = float(getattr(solver.postprocessor, "point_anchor_x_ratio", 0.5))

        model.eval()
        criterion.eval()

        rows: list[dict[str, Any]] = []
        metric_logger = MetricLogger(delimiter="  ")
        header = f"LayerOracle-{split}:"
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(eval_device)
            targets = [{k: v.to(eval_device) for k, v in t.items()} for t in targets]
            matcher_targets = make_matcher_targets(targets)
            with decoder_aux_output_mode(model):
                outputs = model(samples)

            if "aux_outputs" not in outputs:
                raise RuntimeError("Model forward did not return aux_outputs under decoder aux output mode.")
            indices = matcher(outputs, matcher_targets, allow_point_cost=False)["indices"]
            layers = layer_outputs(outputs)
            layer_names = [name for name, _ in layers]

            for batch_idx, (pred_indices, gt_indices) in enumerate(indices):
                if pred_indices.numel() == 0:
                    continue
                target = targets[batch_idx]
                has_picking = target.get("has_picking", torch.zeros_like(gt_indices, dtype=torch.float32))
                gt_points = target.get("picking_points")
                if gt_points is None:
                    continue
                orig_size_wh = target["orig_size"].to(torch.float32)
                gt_boxes_orig = target_boxes_to_original_xyxy(target)
                final_boxes_orig = cxcywh_to_xyxy(outputs["pred_boxes"][batch_idx : batch_idx + 1]).squeeze(0)
                final_boxes_orig = final_boxes_orig * orig_size_wh.repeat(2).view(1, 4)
                ious, _ = box_iou(final_boxes_orig, gt_boxes_orig)
                image_id = int(target["image_id"].flatten()[0].item())
                info = image_info.get(image_id, {})
                file_name = str(info.get("file_name", ""))
                image_path = str((dataset_root / split / file_name).resolve()) if file_name else ""

                for pred_idx_t, gt_idx_t in zip(pred_indices.tolist(), gt_indices.tolist()):
                    if float(has_picking[gt_idx_t].item()) <= 0.5:
                        continue
                    gt_point = gt_points[gt_idx_t].to(torch.float32)
                    layer_l2: dict[str, float] = {}
                    layer_points: dict[str, list[float]] = {}
                    layer_offsets: dict[str, list[float]] = {}
                    missing_layer = False
                    for layer_name, layer_out in layers:
                        offsets = layer_out.get("pred_picking_offsets")
                        boxes = layer_out.get("pred_boxes")
                        if offsets is None or boxes is None or pred_idx_t >= offsets.shape[1] or pred_idx_t >= boxes.shape[1]:
                            missing_layer = True
                            continue
                        pred_points = decode_points_for_layer(
                            boxes[batch_idx : batch_idx + 1],
                            offsets[batch_idx : batch_idx + 1],
                            orig_size_wh,
                            point_offset_mode=point_offset_mode,
                            point_top_anchor_ratio=point_top_anchor_ratio,
                            point_anchor_x_ratio=point_anchor_x_ratio,
                        ).squeeze(0)
                        point = pred_points[pred_idx_t]
                        layer_l2[layer_name] = l2_px(point, gt_point)
                        layer_points[layer_name] = [float(point[0].item()), float(point[1].item())]
                        layer_offsets[layer_name] = [
                            float(offsets[batch_idx, pred_idx_t, 0].item()),
                            float(offsets[batch_idx, pred_idx_t, 1].item()),
                        ]
                    if "final" not in layer_l2:
                        continue
                    final_l2 = layer_l2["final"]
                    best_layer_name = min(layer_l2.keys(), key=lambda name: (layer_l2[name], 0 if name == "final" else 1))
                    best_layer_l2 = layer_l2[best_layer_name]
                    row = {
                        "split": split,
                        "image_id": image_id,
                        "file_name": file_name,
                        "image_path": image_path,
                        "gt_index": int(gt_idx_t),
                        "pred_query_index": int(pred_idx_t),
                        "final_iou": float(ious[pred_idx_t, gt_idx_t].item()),
                        "is_iou50": bool(float(ious[pred_idx_t, gt_idx_t].item()) >= iou_threshold),
                        "gt_point": [float(gt_point[0].item()), float(gt_point[1].item())],
                        "gt_bbox_xyxy": [float(v) for v in gt_boxes_orig[gt_idx_t].detach().cpu().tolist()],
                        "pred_bbox_xyxy": [float(v) for v in final_boxes_orig[pred_idx_t].detach().cpu().tolist()],
                        "available_layers": layer_names,
                        "missing_any_layer": bool(missing_layer),
                        "layer_l2": layer_l2,
                        "layer_points": layer_points,
                        "layer_offsets": layer_offsets,
                        "final_offset": layer_offsets.get("final"),
                        "final_l2": float(final_l2),
                        "best_layer_name": best_layer_name,
                        "best_layer_l2": float(best_layer_l2),
                        "improvement": float(final_l2 - best_layer_l2),
                        "whether_final_is_best": bool(final_l2 <= best_layer_l2 + 1e-6),
                        "whether_best_layer_improves_by_5px": bool(final_l2 - best_layer_l2 >= 5.0),
                        "whether_best_layer_turns_L2gt30_to_L2le30": bool(
                            final_l2 > fix_threshold_px and best_layer_l2 <= fix_threshold_px
                        ),
                    }
                    rows.append(row)

        metric_logger.synchronize_between_processes()
        solver.cleanup()

    summary = summarize_rows(rows, iou_threshold=iou_threshold, fix_threshold_px=fix_threshold_px)
    summary.update(
        {
            "split": split,
            "config": str(config_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "dataset_annotation": str((dataset_root / split / "_annotations.grape_point.json").resolve()),
            "layer_names_observed": sorted({name for row in rows for name in row.get("layer_l2", {}).keys()}),
            "decoder_aux_output_mode": {
                "used": True,
                "model_backward": False,
                "optimizer_step": False,
                "checkpoint_written": False,
                "num_denoising_temporarily_set_to_zero": True,
                "reason": "eval forward does not expose aux_outputs; decoder training flags were temporarily enabled under torch.no_grad for diagnostics only.",
            },
        }
    )
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "split",
        "image_id",
        "file_name",
        "gt_index",
        "pred_query_index",
        "final_iou",
        "is_iou50",
        "final_l2",
        "best_layer_name",
        "best_layer_l2",
        "improvement",
        "whether_final_is_best",
        "whether_best_layer_improves_by_5px",
        "whether_best_layer_turns_L2gt30_to_L2le30",
        "gt_point",
        "final_offset",
        "layer_l2",
        "layer_offsets",
        "layer_points",
        "gt_bbox_xyxy",
        "pred_bbox_xyxy",
        "image_path",
    ]
    keys = preferred + sorted({key for row in rows for key in row.keys()} - set(preferred))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv(row))


def make_bad_cases(rows_by_split: dict[str, list[dict[str, Any]]], fix_threshold_px: float) -> list[dict[str, Any]]:
    bad_rows: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        for row in rows:
            if safe_float(row.get("final_l2")) <= fix_threshold_px and not bool(row.get("whether_best_layer_turns_L2gt30_to_L2le30")):
                continue
            label = "final_l2_gt30"
            if bool(row.get("whether_best_layer_turns_L2gt30_to_L2le30")):
                label = "fixable_by_best_layer"
            elif not bool(row.get("is_iou50")):
                label = "non_iou50_hungarian_match"
            bad = dict(row)
            bad["bad_case_label"] = label
            bad_rows.append(bad)
    bad_rows.sort(key=lambda item: (item["split"] != "valid", -safe_float(item.get("final_l2")), -safe_float(item.get("improvement"))))
    return bad_rows


def decision_from_valid(valid_summary: dict[str, Any], use_iou50: bool = True) -> tuple[str, list[str]]:
    stats = valid_summary.get("iou50_subset", {}) if use_iou50 else valid_summary
    final_mean = stats.get("final_mean_l2")
    oracle_mean = stats.get("best_layer_oracle_mean_l2")
    final_gt30 = int(stats.get("final_l2_gt30_count") or 0)
    oracle_gt30 = int(stats.get("best_layer_oracle_l2_gt30_count") or 0)
    fixed_ratio = safe_float(stats.get("l2_gt30_fixed_to_l2_le30_ratio"), 0.0)
    final_is_best_ratio = safe_float(stats.get("final_is_best_ratio"), 1.0)
    reasons: list[str] = []
    mean_gap = 0.0
    if final_mean is not None and oracle_mean is not None:
        mean_gap = safe_float(final_mean) - safe_float(oracle_mean)
        reasons.append(f"valid best-layer oracle mean gap = {mean_gap:.3f} px")
    reasons.append(f"valid L2>30 reduction = {final_gt30 - oracle_gt30}")
    reasons.append(f"valid L2>30 fixed-to-<=30 ratio = {fixed_ratio:.4f}")
    reasons.append(f"valid final_is_best_ratio = {final_is_best_ratio:.4f}")
    can_do = (
        mean_gap >= 0.5
        or (final_gt30 - oracle_gt30) >= 5
        or fixed_ratio >= 0.10
    ) and final_is_best_ratio < 0.90
    if can_do:
        return "A. 可以做 POINT_LSD_V1_PROBE20", reasons
    return "B. 不建议做 POINT_LSD，直接进入 grouped picking query 方案设计", reasons


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any], decision: str, decision_reasons: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Best-Layer Point Oracle Diagnostic")
    lines.append("")
    lines.append("Scope: read-only diagnostic. No training, no checkpoint generation, no matcher/head/postprocessor changes.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- config: `{payload['metadata']['config']}`")
    lines.append(f"- checkpoint: `{payload['metadata']['checkpoint']}`")
    lines.append("- split usage: valid is used for the POINT_LSD decision; test is reference only.")
    lines.append("- matching: final-layer Hungarian indices are fixed; aux/pre layers are evaluated on the same query and GT.")
    lines.append("- decision table below uses the IoU50-visible subset, aligned with the grape-point evaluation meaning of matched visible grapes.")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| split | visible pairs | final mean L2 | oracle mean L2 | final PPL-SR@30 | oracle PPL-SR@30 | final L2>30 | oracle L2>30 | final best ratio | fixed ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, summary in payload["splits"].items():
        stats = summary.get("iou50_subset", {})
        lines.append(
            "| {split} | {pairs} | {final_mean} | {oracle_mean} | {final_ppl} | {oracle_ppl} | {final_gt30} | {oracle_gt30} | {best_ratio} | {fixed_ratio} |".format(
                split=split,
                pairs=format_metric(stats.get("visible_matched_pair_count"), 0),
                final_mean=format_metric(stats.get("final_mean_l2")),
                oracle_mean=format_metric(stats.get("best_layer_oracle_mean_l2")),
                final_ppl=format_metric(stats.get("final_ppl_sr_30")),
                oracle_ppl=format_metric(stats.get("best_layer_oracle_ppl_sr_30")),
                final_gt30=format_metric(stats.get("final_l2_gt30_count"), 0),
                oracle_gt30=format_metric(stats.get("best_layer_oracle_l2_gt30_count"), 0),
                best_ratio=format_metric(stats.get("final_is_best_ratio")),
                fixed_ratio=format_metric(stats.get("l2_gt30_fixed_to_l2_le30_ratio")),
            )
        )
    lines.append("")
    lines.append("## All Hungarian Visible Reference")
    lines.append("")
    lines.append("This section includes visible GTs matched by Hungarian assignment even when the final box IoU is below 0.5; it is useful for criterion-level diagnosis, but not used as the main go/no-go gate.")
    lines.append("")
    lines.append("| split | visible pairs | final mean L2 | oracle mean L2 | final PPL-SR@30 | oracle PPL-SR@30 | final L2>30 | oracle L2>30 | final best ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, summary in payload["splits"].items():
        lines.append(
            "| {split} | {pairs} | {final_mean} | {oracle_mean} | {final_ppl} | {oracle_ppl} | {final_gt30} | {oracle_gt30} | {best_ratio} |".format(
                split=split,
                pairs=format_metric(summary.get("visible_matched_pair_count"), 0),
                final_mean=format_metric(summary.get("final_mean_l2")),
                oracle_mean=format_metric(summary.get("best_layer_oracle_mean_l2")),
                final_ppl=format_metric(summary.get("final_ppl_sr_30")),
                oracle_ppl=format_metric(summary.get("best_layer_oracle_ppl_sr_30")),
                final_gt30=format_metric(summary.get("final_l2_gt30_count"), 0),
                oracle_gt30=format_metric(summary.get("best_layer_oracle_l2_gt30_count"), 0),
                best_ratio=format_metric(summary.get("final_is_best_ratio")),
            )
        )
    lines.append("")
    lines.append("## Decoder Aux Output Mode")
    lines.append("")
    lines.append("- Eval forward normally does not expose aux outputs.")
    lines.append("- The script temporarily enables decoder training flags under `torch.no_grad()` and sets `num_denoising=0` only in memory.")
    lines.append("- It does not call backward, optimizer, EMA update, or checkpoint save.")
    lines.append("")
    lines.append("## Best Layer Distribution")
    lines.append("")
    lines.append("| split | subset | best_layer_counts | layer_mean_l2 |")
    lines.append("|---|---|---|---|")
    for split, summary in payload["splits"].items():
        for subset_name, stats in (("iou50", summary.get("iou50_subset", {})), ("all_hungarian", summary)):
            lines.append(
                "| {split} | {subset} | `{counts}` | `{means}` |".format(
                    split=split,
                    subset=subset_name,
                    counts=json.dumps(stats.get("best_layer_counts", {}), ensure_ascii=False),
                    means=json.dumps(stats.get("layer_mean_l2", {}), ensure_ascii=False),
                )
            )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(decision)
    lines.append("")
    for reason in decision_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    if decision.startswith("A."):
        lines.append("Interpretation: at least one earlier/pre decoder point output contains usable localization signal. POINT_LSD can be tested as a conservative training-only mechanism.")
    else:
        lines.append("Interpretation: aux/pre layers do not expose enough better point coordinates. Training POINT_LSD would likely be low-yield; move to grouped picking query design instead.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_utils.setup_distributed(seed=0)
    try:
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for split in args.splits:
            rows, summary = analyze_split(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                split=split,
                dataset_root=args.dataset_root.resolve(),
                output_dir=output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                iou_threshold=args.iou_threshold,
                fix_threshold_px=args.fix_threshold_px,
            )
            rows_by_split[split] = rows
            summaries[split] = summary
            write_csv(output_dir / f"layer_oracle_by_instance_{split}.csv", rows)

        bad_cases = make_bad_cases(rows_by_split, fix_threshold_px=args.fix_threshold_px)
        write_csv(output_dir / "layer_oracle_bad_cases.csv", bad_cases)
        valid_summary = summaries.get("valid") or next(iter(summaries.values()))
        decision, decision_reasons = decision_from_valid(valid_summary, use_iou50=True)
        payload = {
            "metadata": {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "dataset_root": str(args.dataset_root.resolve()),
                "output_dir": str(output_dir),
                "splits": args.splits,
                "iou_threshold": float(args.iou_threshold),
                "fix_threshold_px": float(args.fix_threshold_px),
                "training_started": False,
                "checkpoint_generated": False,
                "model_code_changed": False,
            },
            "splits": summaries,
            "decision": {
                "result": decision,
                "reasons": decision_reasons,
                "decision_basis": "valid iou50_visible subset",
            },
        }
        (output_dir / "layer_oracle_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_markdown(output_dir / "layer_oracle_summary.md", payload, decision, decision_reasons)
    finally:
        dist_utils.cleanup()
    print(f"Layer oracle diagnostic written to: {output_dir}")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
