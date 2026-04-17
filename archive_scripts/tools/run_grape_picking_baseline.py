from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_picking_baseline.yml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "baseline_main"
DEFAULT_HGNET_STAGE1 = REPO_ROOT / "pretrain" / "hgnetv2" / "PPHGNetV2_B0_stage1.pth"
COMMON_TUNING_CHECKPOINTS = (
    REPO_ROOT / "pretrain" / "rtv4_hgnetv2_s_coco.pth",
    REPO_ROOT / "pretrain" / "rtv4_s_coco.pth",
    REPO_ROOT / "pretrain" / "rtdetrv4_s_coco.pth",
)


def _write_annotation_variant(
    dataset_root: Path,
    loaded: dict,
    output_name: str,
    mapping_name: str,
    keep_category_names: set[str] | None = None,
) -> Path:
    keep_category_names = None if keep_category_names is None else set(keep_category_names)

    used_ids = set()
    for payload in loaded.values():
        categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in payload.get("categories", [])}
        for ann in payload.get("annotations", []):
            category_name = categories.get(ann["category_id"])
            if keep_category_names is None or category_name in keep_category_names:
                used_ids.add(ann["category_id"])

    base_categories = loaded["train"].get("categories", [])
    ordered_categories = [cat for cat in base_categories if cat["id"] in used_ids]
    id_map = {cat["id"]: new_id for new_id, cat in enumerate(ordered_categories)}

    if not id_map:
        raise ValueError(f"No annotation categories were found for variant: {mapping_name}")

    new_categories = []
    for cat in ordered_categories:
        new_cat = dict(cat)
        new_cat["id"] = id_map[cat["id"]]
        new_categories.append(new_cat)

    mapping_path = dataset_root / mapping_name
    mapping_payload = {
        "old_to_new": {str(old_id): new_id for old_id, new_id in id_map.items()},
        "new_categories": new_categories,
    }
    mapping_path.write_text(json.dumps(mapping_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for split, payload in loaded.items():
        categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in payload.get("categories", [])}
        remapped = dict(payload)
        remapped["categories"] = new_categories
        remapped_annotations = []
        for ann in payload.get("annotations", []):
            category_name = categories.get(ann["category_id"])
            if ann["category_id"] not in id_map:
                continue
            if keep_category_names is not None and category_name not in keep_category_names:
                continue
            new_ann = dict(ann)
            new_ann["category_id"] = id_map[ann["category_id"]]
            remapped_annotations.append(new_ann)
        remapped["annotations"] = remapped_annotations

        out_path = dataset_root / split / output_name
        out_path.write_text(json.dumps(remapped, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[prepare] wrote {out_path}")

    print(f"[prepare] wrote {mapping_path}")
    return mapping_path


def prepare_annotations(dataset_root: Path, output_name: str = "_annotations.rtv4.json") -> Path:
    splits = ("train", "valid", "test")
    input_name = "_annotations.coco.json"
    loaded = {}

    for split in splits:
        ann_path = dataset_root / split / input_name
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        with ann_path.open("r", encoding="utf-8") as f:
            loaded[split] = json.load(f)

    mapping_path = _write_annotation_variant(
        dataset_root=dataset_root,
        loaded=loaded,
        output_name=output_name,
        mapping_name="category_mapping.rtv4.json",
    )
    return mapping_path


def read_config_output_dir(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = config.get("output_dir")
    if output_dir:
        return (REPO_ROOT / output_dir).resolve()
    return DEFAULT_OUTPUT_DIR.resolve()


def read_config_val_ann_file(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ann_file = (
        config.get("val_dataloader", {})
        .get("dataset", {})
        .get("ann_file")
    )
    if ann_file:
        return Path(ann_file).name
    return "_annotations.rtv4.json"


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

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint explicitly or train first."
    )


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
        print("[config] disabling HGNetv2 stage1 preload because a full checkpoint will be loaded.")
        return True

    if DEFAULT_HGNET_STAGE1.exists():
        return False

    print(
        "[config] local HGNetv2 stage1 weights were not found at "
        f"{DEFAULT_HGNET_STAGE1}. Training will run with HGNetv2.pretrained=False."
    )
    return True


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
        updates.extend([
            f"train_dataloader.total_batch_size={args.batch_size}",
            f"val_dataloader.total_batch_size={args.batch_size}",
        ])
    if args.num_workers is not None:
        updates.extend([
            f"train_dataloader.num_workers={args.num_workers}",
            f"val_dataloader.num_workers={args.num_workers}",
        ])
    if args.epochs is not None:
        updates.append(f"epoches={args.epochs}")
    if getattr(args, "output_dir_explicit", False):
        updates.append(f"output_dir='{args.output_dir.as_posix()}'")
    if disable_backbone_pretrained:
        updates.append("HGNetv2.pretrained=False")
    if updates:
        command.extend(["-u", *updates])
    return command


def build_eval_command(
    args: argparse.Namespace,
    checkpoint: Path,
    split: str,
) -> list[str]:
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
    ]
    if args.batch_size is not None:
        updates.append(f"val_dataloader.total_batch_size={args.batch_size}")
    if args.num_workers is not None:
        updates.append(f"val_dataloader.num_workers={args.num_workers}")
    if should_disable_backbone_pretrained(has_full_checkpoint=True):
        updates.append("HGNetv2.pretrained=False")
    command.extend(["-u", *updates])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current RT-DETRv4-S grape+picking baseline workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
        subparser.add_argument("--output-dir", type=Path, default=None)
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument("--num-workers", type=int, default=None)
        subparser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.add_parser("prepare", help="Remap COCO category ids to contiguous ids.")
    add_shared_flags(prepare_parser)

    train_parser = subparsers.add_parser("train", help="Train the current RT-DETRv4-S grape+picking model.")
    add_shared_flags(train_parser)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--tuning", type=str, default=None, help="Optional COCO pretrained checkpoint.")
    train_parser.add_argument("--resume", type=str, default=None, help="Optional resume checkpoint.")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint on the valid or test split.")
    add_shared_flags(eval_parser)
    eval_parser.add_argument("--split", choices=("valid", "test"), default="valid")
    eval_parser.add_argument("--checkpoint", type=str, default=None)

    all_parser = subparsers.add_parser("all", help="Prepare annotations, train, then eval on valid and test.")
    add_shared_flags(all_parser)
    all_parser.add_argument("--epochs", type=int, default=None)
    all_parser.add_argument("--seed", type=int, default=0)
    all_parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    all_parser.add_argument("--tuning", type=str, default=None, help="Optional COCO pretrained checkpoint.")
    all_parser.add_argument("--resume", type=str, default=None, help="Optional resume checkpoint.")

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

