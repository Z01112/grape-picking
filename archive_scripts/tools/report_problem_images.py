from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize problematic test images from a baseline report."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def box_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def load_picking_gt(dataset_root: Path, split: str) -> tuple[dict[int, dict], dict[int, list[list[float]]]]:
    ann_path = dataset_root / split / "_annotations.coco.json"
    payload = read_json(ann_path)
    categories = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
    images_by_id = {int(image["id"]): image for image in payload.get("images", [])}
    picking_gt_by_image: dict[int, list[list[float]]] = {}
    for ann in payload.get("annotations", []):
        if categories.get(int(ann["category_id"])) != "picking":
            continue
        image_id = int(ann["image_id"])
        picking_gt_by_image.setdefault(image_id, []).append(xywh_to_xyxy(ann["bbox"]))
    return images_by_id, picking_gt_by_image


def load_picking_predictions(report_dir: Path, score_thr: float) -> dict[int, list[list[float]]]:
    payload = read_json(report_dir / "predictions_test.json")
    predictions_by_image: dict[int, list[list[float]]] = {}
    for item in payload:
        image_id = int(item["image_id"])
        for pred in item.get("predictions", []):
            if pred["class_name"] != "picking":
                continue
            if float(pred["score"]) < score_thr:
                continue
            predictions_by_image.setdefault(image_id, []).append([float(v) for v in pred["box_xyxy"]])
    return predictions_by_image


def match_picking(pred_boxes: list[list[float]], gt_boxes: list[list[float]], iou_thr: float) -> tuple[int, int, int]:
    used_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= iou_thr:
            used_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn


def render_problem_sheet(
    dataset_root: Path,
    split: str,
    rows: list[dict],
    output_path: Path,
) -> None:
    if not rows:
        return

    tile_w = 360
    tile_h = 260
    gap = 10
    canvas = Image.new("RGB", (tile_w * 2 + gap * 3, len(rows) * (tile_h + 56 + gap) + gap), color=(245, 245, 244))
    font = ImageFont.load_default()

    for idx, row in enumerate(rows):
        image_path = dataset_root / split / row["file_name"]
        gt_image = Image.open(image_path).convert("RGB")
        pred_image = gt_image.copy()
        gt_draw = ImageDraw.Draw(gt_image)
        pred_draw = ImageDraw.Draw(pred_image)
        for box in row["gt_picking_boxes"]:
            gt_draw.rectangle(box, outline=(39, 174, 96), width=3)
            pred_draw.rectangle(box, outline=(39, 174, 96), width=3)
        for box in row["pred_picking_boxes"]:
            pred_draw.rectangle(box, outline=(231, 76, 60), width=3)
        gt_image.thumbnail((tile_w, tile_h))
        pred_image.thumbnail((tile_w, tile_h))

        top = gap + idx * (tile_h + 56 + gap)
        header = Image.new("RGB", (tile_w * 2 + gap, 48), color=(38, 70, 83))
        header_draw = ImageDraw.Draw(header)
        title = (
            f"{row['image_id']} | gt={row['gt_count']} pred={row['pred_count']} "
            f"correct={row['correct']} missed={row['missed']} extra={row['extra']} "
            f"pick_fp={row['picking_fp']} risk={row['risk_score']:.2f}"
        )
        header_draw.text((8, 16), title[:120], fill="white", font=font)
        canvas.paste(header, (gap, top))

        gt_panel = Image.new("RGB", (tile_w, tile_h), color="white")
        gt_panel.paste(gt_image, ((tile_w - gt_image.width) // 2, (tile_h - gt_image.height) // 2))
        pred_panel = Image.new("RGB", (tile_w, tile_h), color="white")
        pred_panel.paste(pred_image, ((tile_w - pred_image.width) // 2, (tile_h - pred_image.height) // 2))

        left = gap
        right = gap * 2 + tile_w
        canvas.paste(gt_panel, (left, top + 48 + gap))
        canvas.paste(pred_panel, (right, top + 48 + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report_dir = args.report_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (report_dir / "problem_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = read_json(report_dir / "per_image_test_summary.json")
    images_by_id, gt_picking_by_image = load_picking_gt(dataset_root, args.split)
    pred_picking_by_image = load_picking_predictions(report_dir, args.score_thr)

    ranked_rows: list[dict] = []
    for row in per_image_rows:
        image_id = int(row["image_id"])
        pred_boxes = pred_picking_by_image.get(image_id, [])
        gt_boxes = gt_picking_by_image.get(image_id, [])
        tp, fp, fn = match_picking(pred_boxes, gt_boxes, iou_thr=args.iou_thr)
        risk_score = (
            1.3 * float(row.get("extra", 0))
            + 1.1 * float(row.get("missed", 0))
            + 1.5 * float(fp)
            + 0.2 * max(0.0, float(row.get("pred_count", 0)) - float(row.get("gt_count", 0)))
        )
        ranked_rows.append(
            {
                "image_id": image_id,
                "file_name": row["file_name"],
                "gt_count": int(row.get("gt_count", 0)),
                "pred_count": int(row.get("pred_count", 0)),
                "correct": int(row.get("correct", 0)),
                "missed": int(row.get("missed", 0)),
                "extra": int(row.get("extra", 0)),
                "error_score": int(row.get("error_score", 0)),
                "picking_tp": tp,
                "picking_fp": fp,
                "picking_fn": fn,
                "risk_score": round(risk_score, 4),
                "gt_picking_boxes": gt_boxes,
                "pred_picking_boxes": pred_boxes,
                "image_width": int(images_by_id[image_id]["width"]),
                "image_height": int(images_by_id[image_id]["height"]),
            }
        )

    ranked_rows.sort(
        key=lambda item: (
            item["risk_score"],
            item["error_score"],
            item["picking_fp"],
            item["extra"],
            item["missed"],
        ),
        reverse=True,
    )

    top_rows = ranked_rows[: args.top_k]
    export_rows = []
    for row in top_rows:
        export_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"gt_picking_boxes", "pred_picking_boxes"}
            }
        )

    write_json(output_dir / "problem_images.json", export_rows)

    lines = [
        "# Problem Images",
        "",
        f"- source report: `{report_dir}`",
        f"- score threshold: `{args.score_thr}`",
        f"- IoU threshold: `{args.iou_thr}`",
        "",
        "| image_id | file_name | risk | gt | pred | correct | missed | extra | picking_fp |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in export_rows:
        lines.append(
            f"| `{row['image_id']}` | `{row['file_name']}` | `{row['risk_score']:.2f}` | "
            f"`{row['gt_count']}` | `{row['pred_count']}` | `{row['correct']}` | "
            f"`{row['missed']}` | `{row['extra']}` | `{row['picking_fp']}` |"
        )
    write_text(output_dir / "problem_images.md", "\n".join(lines) + "\n")

    render_problem_sheet(
        dataset_root=dataset_root,
        split=args.split,
        rows=top_rows,
        output_path=output_dir / "problem_images.jpg",
    )


if __name__ == "__main__":
    main()
