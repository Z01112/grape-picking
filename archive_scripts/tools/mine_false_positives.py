from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine false-positive predictions for a target class from report predictions."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", default="picking")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--top-k-images", type=int, default=12)
    parser.add_argument("--top-k-boxes", type=int, default=48)
    parser.add_argument("--thumb-size", type=int, default=320)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def chunked(items: list, size: int) -> list[list]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def build_contact_sheet(cards: list[Image.Image], columns: int = 3, bg=(18, 18, 18)) -> Image.Image:
    if not cards:
        return Image.new("RGB", (800, 240), color=(32, 32, 32))

    widths = [card.width for card in cards]
    heights = [card.height for card in cards]
    card_w = max(widths)
    card_h = max(heights)
    rows = math.ceil(len(cards) / columns)
    canvas = Image.new("RGB", (columns * card_w, rows * card_h), color=bg)

    for idx, card in enumerate(cards):
        row = idx // columns
        col = idx % columns
        x = col * card_w
        y = row * card_h
        canvas.paste(card, (x, y))
    return canvas


def render_image_card(
    image_path: Path,
    fp_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    title: str,
    thumb_size: int,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 255, 80), width=3)
    for box in fp_boxes:
        draw.rectangle(box, outline=(255, 80, 80), width=3)

    image.thumbnail((thumb_size, thumb_size))
    title_bar = Image.new("RGB", (image.width, 48), color=(28, 28, 28))
    title_draw = ImageDraw.Draw(title_bar)
    title_draw.text((10, 14), title, fill=(245, 245, 245))

    card = Image.new("RGB", (image.width, image.height + title_bar.height), color=(18, 18, 18))
    card.paste(title_bar, (0, 0))
    card.paste(image, (0, title_bar.height))
    return card


def render_annotated_image(
    image_path: Path,
    fp_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 255, 80), width=4)
    for box in fp_boxes:
        draw.rectangle(box, outline=(255, 80, 80), width=4)
    return image


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    split_dir = dataset_root / args.split
    ann_path = split_dir / "_annotations.coco.json"
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ann = read_json(ann_path)
    predictions = read_json(predictions_path)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in ann["categories"]}
    if args.class_name not in cat_name_to_id:
        raise ValueError(f"Unknown class name: {args.class_name}")
    class_id = cat_name_to_id[args.class_name]

    image_info = {img["id"]: img for img in ann["images"]}
    gt_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for one in ann["annotations"]:
        if one["category_id"] != class_id:
            continue
        gt_by_image[one["image_id"]].append(xywh_to_xyxy(one["bbox"]))

    fp_records: list[dict] = []
    image_summaries: list[dict] = []

    for entry in predictions:
        image_id = entry["image_id"]
        file_name = entry["file_name"]
        gt_boxes = gt_by_image.get(image_id, [])
        preds = [
            pred for pred in entry["predictions"]
            if pred["class_name"] == args.class_name and float(pred["score"]) >= args.score_thr
        ]
        preds.sort(key=lambda item: float(item["score"]), reverse=True)

        matched_gt = set()
        image_fp_boxes: list[list[float]] = []
        image_fp_scores: list[float] = []

        for pred in preds:
            pred_box = [float(v) for v in pred["box_xyxy"]]
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx is not None and best_iou >= args.iou_thr:
                matched_gt.add(best_gt_idx)
                continue

            image_fp_boxes.append(pred_box)
            image_fp_scores.append(float(pred["score"]))
            fp_records.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "score": float(pred["score"]),
                    "box_xyxy": pred_box,
                    "best_iou_with_gt": best_iou,
                    "gt_count": len(gt_boxes),
                }
            )

        if image_fp_boxes:
            image_summaries.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "fp_count": len(image_fp_boxes),
                    "fp_score_sum": round(sum(image_fp_scores), 6),
                    "fp_score_max": round(max(image_fp_scores), 6),
                    "gt_count": len(gt_boxes),
                    "matched_gt": len(matched_gt),
                    "unmatched_gt": len(gt_boxes) - len(matched_gt),
                    "fp_boxes": image_fp_boxes,
                    "gt_boxes": gt_boxes,
                }
            )

    fp_records.sort(key=lambda item: item["score"], reverse=True)
    image_summaries.sort(key=lambda item: (item["fp_score_sum"], item["fp_count"], item["fp_score_max"]), reverse=True)

    summary = {
        "class_name": args.class_name,
        "score_threshold": args.score_thr,
        "iou_threshold": args.iou_thr,
        "total_false_positive_boxes": len(fp_records),
        "images_with_false_positives": len(image_summaries),
        "mean_fp_per_image": round(
            len(fp_records) / max(1, len(image_summaries)),
            4,
        ),
        "median_fp_per_image": round(
            statistics.median([item["fp_count"] for item in image_summaries]) if image_summaries else 0.0,
            4,
        ),
        "top_false_positive_boxes": fp_records[: args.top_k_boxes],
        "top_false_positive_images": image_summaries[: args.top_k_images],
    }

    (output_dir / f"{args.class_name}_false_positive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cards: list[Image.Image] = []
    top_image_dir = output_dir / "top_images"
    top_image_dir.mkdir(parents=True, exist_ok=True)
    for item in image_summaries[: args.top_k_images]:
        image_path = split_dir / item["file_name"]
        if not image_path.exists():
            continue
        title = (
            f"{item['file_name'][:30]} | fp={item['fp_count']} "
            f"| gt={item['gt_count']} | max={item['fp_score_max']:.3f}"
        )
        cards.append(
            render_image_card(
                image_path=image_path,
                fp_boxes=item["fp_boxes"],
                gt_boxes=item["gt_boxes"],
                title=title,
                thumb_size=args.thumb_size,
            )
        )
        annotated = render_annotated_image(
            image_path=image_path,
            fp_boxes=item["fp_boxes"],
            gt_boxes=item["gt_boxes"],
        )
        annotated.save(top_image_dir / item["file_name"], quality=92)

    sheet = build_contact_sheet(cards, columns=3)
    sheet.save(output_dir / f"{args.class_name}_false_positive_sheet.jpg", quality=92)

    print(json.dumps(
        {
            "summary_path": str((output_dir / f"{args.class_name}_false_positive_summary.json").resolve()),
            "sheet_path": str((output_dir / f"{args.class_name}_false_positive_sheet.jpg").resolve()),
            "total_false_positive_boxes": len(fp_records),
            "images_with_false_positives": len(image_summaries),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
