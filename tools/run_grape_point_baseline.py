from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.rtv4.point_utils import point_from_xywh_bbox, offset_from_xywh_bbox


DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_baseline.yml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "grape_point_baseline"
DEFAULT_HGNET_STAGE1 = REPO_ROOT / "pretrain" / "hgnetv2" / "PPHGNetV2_B0_stage1.pth"
COMMON_TUNING_CHECKPOINTS = (
    REPO_ROOT / "pretrain" / "rtv4_hgnetv2_s_coco.pth",
    REPO_ROOT / "pretrain" / "rtv4_s_coco.pth",
    REPO_ROOT / "pretrain" / "rtdetrv4_s_coco.pth",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _box_center(box_xyxy: list[float]) -> tuple[float, float]:
    return (0.5 * (box_xyxy[0] + box_xyxy[2]), 0.5 * (box_xyxy[1] + box_xyxy[3]))


def match_picking_to_grape(grapes: list[list[float]], picking_box: list[float]) -> tuple[int | None, dict]:
    if not grapes:
        return None, {}

    pcx, pcy = _box_center(picking_box)
    best_idx = None
    best_meta = None
    for idx, grape_box in enumerate(grapes):
        gcx, gcy = _box_center(grape_box)
        distance_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
        x1, y1, x2, y2 = grape_box
        grape_w = max(x2 - x1, 1e-6)
        grape_h = max(y2 - y1, 1e-6)
        rel_x = (pcx - x1) / grape_w
        rel_y = (pcy - y1) / grape_h
        meta = {
            "distance_sq": float(distance_sq),
            "rel_x": float(rel_x),
            "rel_y": float(rel_y),
            "offset_x": float(rel_x - 0.5),
            "offset_y": float(rel_y),
        }
        if best_meta is None or distance_sq < best_meta["distance_sq"]:
            best_idx = idx
            best_meta = meta
    return best_idx, best_meta or {}

def select_best_picking_for_each_grape(grape_anns: list[dict], picking_anns: list[dict]) -> dict[int, dict]:
    grape_boxes_xyxy = [xywh_to_xyxy(ann["bbox"]) for ann in grape_anns]
    candidates_by_grape: dict[int, list[dict]] = defaultdict(list)

    for picking_ann in picking_anns:
        matched_idx, meta = match_picking_to_grape(grape_boxes_xyxy, xywh_to_xyxy(picking_ann["bbox"]))
        if matched_idx is None:
            continue
        candidate = {
            "ann": picking_ann,
            "meta": meta,
            "point_xy": point_from_xywh_bbox(picking_ann["bbox"]),
        }
        candidates_by_grape[int(matched_idx)].append(candidate)

    selected = {}
    for grape_idx, candidates in candidates_by_grape.items():
        candidates.sort(
            key=lambda item: (
                float(item["meta"].get("distance_sq", 1e18)),
                abs(float(item["meta"].get("offset_y", 0.0))),
                abs(float(item["meta"].get("offset_x", 0.0))),
            )
        )
        selected[int(grape_idx)] = candidates[0]
    return selected


def build_grape_point_payload(payload: dict) -> dict:
    categories = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in payload.get("categories", [])}
    per_image: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"grape": [], "picking": []})

    for ann in payload.get("annotations", []):
        class_name = categories.get(int(ann["category_id"]), str(ann["category_id"]))
        if class_name not in ("grape", "picking"):
            continue
        per_image[int(ann["image_id"])][class_name].append(deepcopy(ann))

    new_payload = deepcopy(payload)
    new_payload["categories"] = [{"id": 0, "name": "grape", "supercategory": "grape"}]
    new_annotations = []

    for image in payload.get("images", []):
        image_id = int(image["id"])
        groups = per_image.get(image_id, {"grape": [], "picking": []})
        grape_anns = groups["grape"]
        selected = select_best_picking_for_each_grape(grape_anns, groups["picking"])

        for grape_idx, grape_ann in enumerate(grape_anns):
            new_ann = deepcopy(grape_ann)
            new_ann["category_id"] = 0

            matched = selected.get(grape_idx)
            if matched is None:
                new_ann["has_picking"] = 0.0
                new_ann["picking_point"] = [0.0, 0.0]
                new_ann["picking_offset"] = [0.0, 0.0]
                new_ann["keypoints"] = [0.0, 0.0, 0.0]
                new_ann["num_keypoints"] = 0
            else:
                point_xy = matched["point_xy"]
                new_ann["has_picking"] = 1.0
                new_ann["picking_point"] = [float(point_xy[0]), float(point_xy[1])]
                new_ann["picking_offset"] = offset_from_xywh_bbox(new_ann["bbox"], point_xy)
                new_ann["keypoints"] = [float(point_xy[0]), float(point_xy[1]), 2.0]
                new_ann["num_keypoints"] = 1
                new_ann["picking_annotation_id"] = int(matched["ann"]["id"])

            new_annotations.append(new_ann)

    new_payload["annotations"] = new_annotations
    return new_payload


def prepare_annotations(dataset_root: Path, output_name: str = "_annotations.grape_point.json") -> None:
    for split in ("train", "valid", "test"):
        ann_path = dataset_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")
        payload = read_json(ann_path)
        out_path = dataset_root / split / output_name
        write_json(out_path, build_grape_point_payload(payload))
        print(f"[prepare] wrote {out_path}")


def read_config_output_dir(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = config.get("output_dir")
    if output_dir:
        return (REPO_ROOT / output_dir).resolve()
    return DEFAULT_OUTPUT_DIR.resolve()


def read_config_val_ann_file(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ann_file = config.get("val_dataloader", {}).get("dataset", {}).get("ann_file")
    if ann_file:
        return Path(ann_file).name
    return "_annotations.grape_point.json"


def run_command(args: list[str], dry_run: bool = False) -> None:
    printable = " ".join(f'"{arg}"' if " " in arg else arg for arg in args)
    print(f"[run] {printable}")
    if dry_run:
        return
    try:
        subprocess.run(args, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[error] command failed with exit code {exc.returncode}") from None


def find_checkpoint(path: str | None, output_dir: Path) -> Path:
    if path:
        ckpt = Path(path)
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    for candidate in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
        for ckpt in (output_dir / candidate, output_dir / "checkpoints" / candidate):
            if ckpt.exists():
                return ckpt

    raise FileNotFoundError("No checkpoint found. Pass --checkpoint explicitly or train first.")


def resolve_tuning_checkpoint(path: str | None) -> Path | None:
    if path:
        tuning_ckpt = Path(path)
        if not tuning_ckpt.is_absolute():
            tuning_ckpt = REPO_ROOT / tuning_ckpt
        if tuning_ckpt.exists():
            return tuning_ckpt
        raise FileNotFoundError(f"Tuning checkpoint not found: {tuning_ckpt}")

    for candidate in COMMON_TUNING_CHECKPOINTS:
        if candidate.exists():
            print(f"[train] using tuning checkpoint: {candidate}")
            return candidate
    return None


def should_disable_backbone_pretrained(has_full_checkpoint: bool) -> bool:
    if has_full_checkpoint:
        return True
    return not DEFAULT_HGNET_STAGE1.exists()


def build_train_command(args: argparse.Namespace) -> list[str]:
    tuning_checkpoint = resolve_tuning_checkpoint(args.tuning)
    disable_backbone_pretrained = should_disable_backbone_pretrained(
        has_full_checkpoint=(tuning_checkpoint is not None or args.resume is not None)
    )

    command = [
        sys.executable,
        "train.py",
        "-c",
        str(args.config),
        "-d",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.use_amp:
        command.append("--use-amp")
    if tuning_checkpoint is not None:
        command.extend(["-t", str(tuning_checkpoint)])
    if args.resume:
        command.extend(["-r", str(Path(args.resume))])

    updates = []
    if args.batch_size is not None:
        updates.extend(
            [
                f"train_dataloader.total_batch_size={args.batch_size}",
                f"val_dataloader.total_batch_size={args.batch_size}",
            ]
        )
    if args.num_workers is not None:
        updates.extend(
            [
                f"train_dataloader.num_workers={args.num_workers}",
                f"val_dataloader.num_workers={args.num_workers}",
            ]
        )
    if args.epochs is not None:
        updates.append(f"epoches={args.epochs}")
    if getattr(args, "output_dir_explicit", False):
        updates.append(f"output_dir='{args.output_dir.as_posix()}'")
    if disable_backbone_pretrained:
        updates.append("HGNetv2.pretrained=False")
    if getattr(args, "config_update", None):
        updates.extend(args.config_update)
    if updates:
        command.extend(["-u", *updates])
    return command


def build_eval_command(args: argparse.Namespace, checkpoint: Path, split: str) -> list[str]:
    split_dir = args.dataset_root / split
    eval_output_dir = args.output_dir / "logs" / f"eval_{split}"
    ann_file_name = read_config_val_ann_file(args.config)

    command = [
        sys.executable,
        "train.py",
        "-c",
        str(args.config),
        "--test-only",
        "-r",
        str(checkpoint),
        "-d",
        args.device,
    ]
    updates = [
        f"val_dataloader.dataset.img_folder='{split_dir.as_posix()}'",
        f"val_dataloader.dataset.ann_file='{(split_dir / ann_file_name).as_posix()}'",
        f"output_dir='{eval_output_dir.as_posix()}'",
        "HGNetv2.pretrained=False",
    ]
    if args.batch_size is not None:
        updates.append(f"val_dataloader.total_batch_size={args.batch_size}")
    if args.num_workers is not None:
        updates.append(f"val_dataloader.num_workers={args.num_workers}")
    if getattr(args, "config_update", None):
        updates.extend(args.config_update)
    command.extend(["-u", *updates])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grape bbox + picking point baseline workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
        subparser.add_argument("--output-dir", type=Path, default=None)
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument("--num-workers", type=int, default=None)
        subparser.add_argument("--config-update", nargs="*", default=None)
        subparser.add_argument("--dry-run", action="store_true")

    for name, help_text in (
        ("prepare", "Prepare grape-point annotations."),
        ("train", "Train the grape-point model."),
        ("eval", "Evaluate a checkpoint on valid or test."),
        ("all", "Prepare annotations, train, then eval on valid and test."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        add_shared_flags(sub)
        if name in ("train", "all"):
            sub.add_argument("--epochs", type=int, default=None)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
            sub.add_argument("--tuning", type=str, default=None)
            sub.add_argument("--resume", type=str, default=None)
        elif name == "eval":
            sub.add_argument("--split", choices=("valid", "test"), default="valid")
            sub.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir_explicit = args.output_dir is not None
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    else:
        args.output_dir = read_config_output_dir(args.config)

    if args.command == "prepare":
        prepare_annotations(args.dataset_root)
        return

    prepare_annotations(args.dataset_root)

    if args.command == "train":
        run_command(build_train_command(args), dry_run=args.dry_run)
        return

    if args.command == "eval":
        checkpoint = find_checkpoint(args.checkpoint, args.output_dir)
        run_command(build_eval_command(args, checkpoint, args.split), dry_run=args.dry_run)
        return

    if args.command == "all":
        run_command(build_train_command(args), dry_run=args.dry_run)
        checkpoint = find_checkpoint(None, args.output_dir)
        run_command(build_eval_command(args, checkpoint, "valid"), dry_run=args.dry_run)
        run_command(build_eval_command(args, checkpoint, "test"), dry_run=args.dry_run)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
