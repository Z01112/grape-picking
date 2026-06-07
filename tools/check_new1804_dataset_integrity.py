from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets"
OUT_DIR = REPO_ROOT / "outputs" / "04_diagnostics" / "new1804_dataset_integrity"
EXPECTED_IMAGES = {"train": 1263, "valid": 361, "test": 180}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out):
        return default
    return out


def point_valid(point: Any) -> bool:
    return isinstance(point, list) and len(point) >= 2 and point[0] is not None and point[1] is not None


def bbox_valid(bbox: Any) -> bool:
    return isinstance(bbox, list) and len(bbox) >= 4


def bbox_issue(bbox: Any, width: float, height: float) -> str:
    if not bbox_valid(bbox):
        return "missing_or_malformed_bbox"
    x, y, w, h = [as_float(v) for v in bbox[:4]]
    issues = []
    if w <= 0 or h <= 0:
        issues.append("non_positive_size")
    if x < 0 or y < 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
        issues.append("bbox_out_of_image")
    if w * h <= 1.0:
        issues.append("tiny_area")
    return ";".join(issues)


def point_issue(point: Any, width: float, height: float) -> str:
    if not point_valid(point):
        return "missing_or_malformed_point"
    x, y = as_float(point[0]), as_float(point[1])
    if x < 0 or y < 0 or x > width + 1e-3 or y > height + 1e-3:
        return "point_out_of_image"
    return ""


def stem_bbox_issue(stem_bbox: Any, width: float, height: float, has_stem: bool) -> str:
    if not has_stem:
        return ""
    if not bbox_valid(stem_bbox):
        return "has_stem_without_valid_stem_bbox"
    return bbox_issue(stem_bbox, width, height)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hard_errors: list[str] = []
    per_split_rows: list[dict] = []
    missing_images: list[dict] = []
    bbox_point_issues: list[dict] = []
    stem_rows: list[dict] = []
    preview: list[dict] = []
    total_images = 0
    total_has_picking = 0
    total_annotations = 0

    for split in ["train", "valid", "test"]:
        split_dir = DATASET_ROOT / split
        ann_path = split_dir / "_annotations.grape_point.json"
        if not split_dir.exists():
            hard_errors.append(f"{split}: missing split directory {split_dir}")
            continue
        if not ann_path.exists():
            hard_errors.append(f"{split}: missing annotation file {ann_path}")
            continue
        payload = read_json(ann_path)
        images = payload.get("images", [])
        anns = payload.get("annotations", [])
        categories = payload.get("categories", [])
        image_files = {
            p.name
            for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }
        images_by_id = {int(img["id"]): img for img in images}
        image_names = [str(img.get("file_name", "")) for img in images]
        missing = [name for name in image_names if name not in image_files]
        extra_images = sorted(image_files - set(image_names))
        for name in missing:
            missing_images.append({"split": split, "file_name": name, "reason": "listed_in_json_missing_on_disk"})
        for name in extra_images[:1000]:
            missing_images.append({"split": split, "file_name": name, "reason": "image_on_disk_not_listed_in_json"})
        if missing:
            hard_errors.append(f"{split}: {len(missing)} annotation images missing on disk")
        expected = EXPECTED_IMAGES[split]
        if len(images) != expected:
            hard_errors.append(f"{split}: expected {expected} images, got {len(images)}")
        category_ids = sorted({int(ann.get("category_id", -999)) for ann in anns})
        category_names = [str(cat.get("name", "")) for cat in categories]
        if category_names != ["grape"]:
            hard_errors.append(f"{split}: expected categories ['grape'], got {category_names}")
        if category_ids != [0]:
            hard_errors.append(f"{split}: single-class remap risk, expected annotation category_id [0], got {category_ids}")

        field_missing = Counter()
        split_has_picking = 0
        split_has_stem = 0
        split_stem_bbox_valid = 0
        for ann_idx, ann in enumerate(anns):
            total_annotations += 1
            image = images_by_id.get(int(ann.get("image_id", -1)))
            if image is None:
                hard_errors.append(f"{split}: annotation {ann_idx} references missing image_id {ann.get('image_id')}")
                continue
            width, height = as_float(image.get("width")), as_float(image.get("height"))
            for field in ["has_picking", "picking_point", "picking_offset", "has_stem", "stem_bbox"]:
                if field not in ann:
                    field_missing[field] += 1
            has_picking = as_float(ann.get("has_picking", 0.0)) > 0.5
            has_stem = as_float(ann.get("has_stem", 0.0)) > 0.5
            if has_picking:
                split_has_picking += 1
                total_has_picking += 1
                if not point_valid(ann.get("picking_point")):
                    bbox_point_issues.append({"split": split, "ann_index": ann_idx, "image_id": ann.get("image_id"), "file_name": image.get("file_name"), "issue": "has_picking_without_valid_picking_point"})
                if not point_valid(ann.get("picking_offset")):
                    bbox_point_issues.append({"split": split, "ann_index": ann_idx, "image_id": ann.get("image_id"), "file_name": image.get("file_name"), "issue": "has_picking_without_valid_picking_offset"})
            stem_issue = stem_bbox_issue(ann.get("stem_bbox"), width, height, has_stem)
            if has_stem:
                split_has_stem += 1
                if not stem_issue:
                    split_stem_bbox_valid += 1
            bi = bbox_issue(ann.get("bbox"), width, height)
            pi = point_issue(ann.get("picking_point"), width, height) if has_picking else ""
            if bi:
                bbox_point_issues.append({"split": split, "ann_index": ann_idx, "image_id": ann.get("image_id"), "file_name": image.get("file_name"), "issue": bi})
            if pi:
                bbox_point_issues.append({"split": split, "ann_index": ann_idx, "image_id": ann.get("image_id"), "file_name": image.get("file_name"), "issue": pi})
            if stem_issue:
                stem_rows.append({"split": split, "ann_index": ann_idx, "image_id": ann.get("image_id"), "file_name": image.get("file_name"), "has_stem": has_stem, "stem_bbox": ann.get("stem_bbox"), "issue": stem_issue})
            if len(preview) < 10:
                preview.append(
                    {
                        "split": split,
                        "image_id": ann.get("image_id"),
                        "file_name": image.get("file_name"),
                        "bbox": ann.get("bbox"),
                        "has_picking": ann.get("has_picking"),
                        "picking_point": ann.get("picking_point"),
                        "picking_offset": ann.get("picking_offset"),
                        "has_stem": ann.get("has_stem"),
                        "stem_bbox": ann.get("stem_bbox"),
                    }
                )
        if field_missing:
            for field, count in field_missing.items():
                hard_errors.append(f"{split}: field {field} missing in {count} annotations")
        per_split_rows.append(
            {
                "split": split,
                "json_images": len(images),
                "disk_images": len(image_files),
                "expected_images": expected,
                "annotations": len(anns),
                "has_picking_count": split_has_picking,
                "has_stem_count": split_has_stem,
                "valid_stem_bbox_count": split_stem_bbox_valid,
                "category_names": "|".join(category_names),
                "annotation_category_ids": "|".join(str(v) for v in category_ids),
                "missing_listed_images": len(missing),
                "extra_disk_images": len(extra_images),
            }
        )
        total_images += len(images)

    if total_images != 1804:
        hard_errors.append(f"total image count expected 1804, got {total_images}")
    if not (2400 <= total_has_picking <= 3100):
        hard_errors.append(f"has_picking total expected about 2700, got {total_has_picking}")
    if bbox_point_issues:
        hard_errors.append(f"bbox/point issue rows: {len(bbox_point_issues)}")
    if stem_rows:
        hard_errors.append(f"stem consistency issue rows: {len(stem_rows)}")

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "passed": len(hard_errors) == 0,
        "hard_error_count": len(hard_errors),
        "hard_errors": hard_errors,
        "total_images": total_images,
        "total_annotations": total_annotations,
        "total_has_picking": total_has_picking,
        "expected_images": EXPECTED_IMAGES,
        "per_split": per_split_rows,
        "notes": [
            "Training config must use datasets/ rather than dataset/.",
            "GrapePointCocoDetection expects category_id 0 when remap_mscoco_category=false and num_classes=1.",
            "stem fields are checked for future use only; this run does not train stem_aux.",
        ],
    }
    write_json(OUT_DIR / "dataset_integrity_summary.json", summary)
    write_json(OUT_DIR / "sample_records_preview.json", preview)
    write_csv(OUT_DIR / "per_split_counts.csv", per_split_rows)
    write_csv(OUT_DIR / "missing_images.csv", missing_images, ["split", "file_name", "reason"])
    write_csv(OUT_DIR / "bbox_point_issues.csv", bbox_point_issues)
    write_csv(OUT_DIR / "stem_point_consistency.csv", stem_rows)

    md = [
        "# New1804 Dataset Integrity Report",
        "",
        f"- Dataset root: `{DATASET_ROOT}`",
        f"- Passed: `{summary['passed']}`",
        f"- Total images: `{total_images}`",
        f"- Total annotations: `{total_annotations}`",
        f"- Total has_picking: `{total_has_picking}`",
        "",
        "## Per Split",
        "|split|json_images|disk_images|expected_images|annotations|has_picking_count|has_stem_count|category_names|category_ids|missing|extra|",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in per_split_rows:
        md.append(
            f"|{row['split']}|{row['json_images']}|{row['disk_images']}|{row['expected_images']}|{row['annotations']}|{row['has_picking_count']}|{row['has_stem_count']}|{row['category_names']}|{row['annotation_category_ids']}|{row['missing_listed_images']}|{row['extra_disk_images']}|"
        )
    md.extend(["", "## Hard Errors"])
    if hard_errors:
        md.extend(f"- {err}" for err in hard_errors)
    else:
        md.append("- None.")
    (OUT_DIR / "dataset_integrity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if hard_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

