from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose picking label consistency relative to its matched grape."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples-per-class", type=int, default=9)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return float(ordered[left])
    weight = position - left
    return float(ordered[left] * (1.0 - weight) + ordered[right] * weight)


def vertical_band(rel_y: float) -> str:
    if rel_y < -0.05:
        return "above"
    if rel_y < 0.10:
        return "upper"
    if rel_y < 0.28:
        return "middle"
    return "lower"


def horizontal_band(rel_x: float) -> str:
    if rel_x < 0.33:
        return "left"
    if rel_x > 0.67:
        return "right"
    return "center"


def nearest_direction(rel_x: float, rel_y: float) -> str:
    distances = {
        "up": abs(rel_y),
        "side": min(abs(rel_x), abs(1.0 - rel_x)),
        "down": abs(1.0 - rel_y),
    }
    return min(distances.items(), key=lambda item: item[1])[0]


def match_picking_to_grape(grapes: list[list[float]], picking_box: list[float]) -> tuple[int | None, dict]:
    if not grapes:
        return None, {}
    pcx, pcy = box_center(picking_box)
    best_idx = None
    best_meta = None
    for idx, grape_box in enumerate(grapes):
        gcx, gcy = box_center(grape_box)
        distance_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
        x1, y1, x2, y2 = grape_box
        grape_w = max(x2 - x1, 1e-6)
        grape_h = max(y2 - y1, 1e-6)
        pick_w = max(picking_box[2] - picking_box[0], 1e-6)
        pick_h = max(picking_box[3] - picking_box[1], 1e-6)
        rel_x = (pcx - x1) / grape_w
        rel_y = (pcy - y1) / grape_h
        meta = {
            "distance_sq": distance_sq,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "offset_x": rel_x - 0.5,
            "offset_y": rel_y,
            "width_ratio": pick_w / grape_w,
            "height_ratio": pick_h / grape_h,
            "area_ratio": (pick_w * pick_h) / (grape_w * grape_h),
            "aspect_ratio": pick_h / pick_w,
            "vertical_band": vertical_band(rel_y),
            "horizontal_band": horizontal_band(rel_x),
            "direction_bucket": nearest_direction(rel_x, rel_y),
            "top_coverage_ratio": max(0.0, min(1.0, (picking_box[3] - y1) / grape_h)),
        }
        if best_meta is None or distance_sq < best_meta["distance_sq"]:
            best_idx = idx
            best_meta = meta
    return best_idx, best_meta or {}


def collect_rows(dataset_root: Path, splits: list[str]) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        payload = read_json(dataset_root / split / "_annotations.rtv4.json")
        categories = {int(cat["id"]): cat["name"] for cat in payload["categories"]}
        images = {int(image["id"]): image for image in payload["images"]}
        per_image: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"grape": [], "picking": []})
        for ann in payload["annotations"]:
            class_name = categories[int(ann["category_id"])]
            per_image[int(ann["image_id"])][class_name].append(
                {
                    "box_xyxy": xywh_to_xyxy(ann["bbox"]),
                    "bbox_xywh": [float(v) for v in ann["bbox"]],
                    "id": int(ann["id"]),
                }
            )

        for image_id, groups in per_image.items():
            image_info = images[image_id]
            grape_boxes = [item["box_xyxy"] for item in groups["grape"]]
            for picking in groups["picking"]:
                grape_idx, meta = match_picking_to_grape(grape_boxes, picking["box_xyxy"])
                if grape_idx is None:
                    continue
                rows.append(
                    {
                        "split": split,
                        "image_id": image_id,
                        "file_name": image_info["file_name"],
                        "image_path": str((dataset_root / split / image_info["file_name"]).resolve()),
                        "image_width": int(image_info["width"]),
                        "image_height": int(image_info["height"]),
                        "picking_id": int(picking["id"]),
                        "picking_box_xyxy": [float(v) for v in picking["box_xyxy"]],
                        "picking_box_xywh": [float(v) for v in picking["bbox_xywh"]],
                        "grape_index": int(grape_idx),
                        "grape_box_xyxy": [float(v) for v in grape_boxes[grape_idx]],
                        **meta,
                    }
                )
    return rows


def kmeans(features: np.ndarray, num_clusters: int, seed: int, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(features.shape[0], size=num_clusters, replace=False)
    centers = features[indices].copy()
    labels = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(max_iters):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(num_clusters):
            mask = new_labels == idx
            if mask.any():
                new_centers[idx] = features[mask].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return labels, centers


def summarize_feature(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "mean": float(sum(values) / len(values)) if values else 0.0,
    }


def cluster_and_label(rows: list[dict], seed: int) -> dict:
    feature_names = ["rel_y", "height_ratio", "area_ratio", "aspect_ratio", "offset_x_abs"]
    raw = np.array(
        [
            [
                float(row["rel_y"]),
                float(row["height_ratio"]),
                float(row["area_ratio"]),
                float(row["aspect_ratio"]),
                abs(float(row["offset_x"])),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    means = raw.mean(axis=0, keepdims=True)
    stds = raw.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    normalized = (raw - means) / stds
    labels, centers = kmeans(normalized, num_clusters=3, seed=seed)

    cluster_rows: dict[int, list[dict]] = defaultdict(list)
    for idx, cluster_id in enumerate(labels.tolist()):
        row = rows[idx]
        row["cluster_id"] = int(cluster_id)
        row["offset_x_abs"] = abs(float(row["offset_x"]))
        row["distance_to_center"] = float(np.sqrt(((normalized[idx] - centers[cluster_id]) ** 2).sum()))
        distances = np.sqrt(((normalized[idx][None, :] - centers) ** 2).sum(axis=1))
        nearest = np.sort(distances)
        row["ambiguity_margin"] = float(nearest[1] - nearest[0]) if len(nearest) > 1 else 0.0
        cluster_rows[int(cluster_id)].append(row)

    raw_centers = centers * stds + means
    cluster_stats = {}
    for cluster_id, entries in cluster_rows.items():
        cluster_stats[cluster_id] = {
            "count": len(entries),
            "share": len(entries) / len(rows),
            "center": {
                name: float(raw_centers[cluster_id, idx])
                for idx, name in enumerate(feature_names)
            },
            "median_rel_y": percentile([float(item["rel_y"]) for item in entries], 0.50),
            "median_height_ratio": percentile([float(item["height_ratio"]) for item in entries], 0.50),
            "median_area_ratio": percentile([float(item["area_ratio"]) for item in entries], 0.50),
            "median_aspect_ratio": percentile([float(item["aspect_ratio"]) for item in entries], 0.50),
            "median_offset_x_abs": percentile([abs(float(item["offset_x"])) for item in entries], 0.50),
        }

    cover_cluster = max(
        cluster_stats,
        key=lambda cluster_id: (
            cluster_stats[cluster_id]["median_height_ratio"],
            cluster_stats[cluster_id]["median_area_ratio"],
            cluster_stats[cluster_id]["median_aspect_ratio"],
        ),
    )
    remaining = [cluster_id for cluster_id in cluster_stats if cluster_id != cover_cluster]
    root_cluster = min(remaining, key=lambda cluster_id: cluster_stats[cluster_id]["median_rel_y"])
    middle_cluster = next(cluster_id for cluster_id in remaining if cluster_id != root_cluster)
    cluster_name_map = {
        cover_cluster: "covers_most_stem",
        root_cluster: "near_root",
        middle_cluster: "near_middle",
    }

    localized_name_map = {
        "near_root": "靠近根部",
        "near_middle": "靠近中段",
        "covers_most_stem": "覆盖大部分果梗",
    }

    for row in rows:
        semantic_class = cluster_name_map[int(row["cluster_id"])]
        row["semantic_class"] = semantic_class
        row["semantic_class_zh"] = localized_name_map[semantic_class]
        row["covers_most_stem_proxy"] = semantic_class == "covers_most_stem"

    return {
        "feature_names": feature_names,
        "cluster_stats": cluster_stats,
        "cluster_name_map": {str(k): v for k, v in cluster_name_map.items()},
        "localized_name_map": localized_name_map,
    }


def make_crop(image_path: Path, grape_box: list[float], picking_box: list[float], label: str, tile_size: tuple[int, int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(grape_box, outline="#00c853", width=3)
    draw.rectangle(picking_box, outline="#ff1744", width=3)

    x1, y1, x2, y2 = grape_box
    pad_x = 0.25 * max(x2 - x1, 1.0)
    pad_y = 0.25 * max(y2 - y1, 1.0)
    crop_box = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(image.width, int(x2 + pad_x)),
        min(image.height, int(y2 + pad_y)),
    )
    cropped = image.crop(crop_box)
    canvas = Image.new("RGB", tile_size, color=(248, 248, 248))
    fitted = cropped.copy()
    fitted.thumbnail((tile_size[0], tile_size[1] - 30), Image.Resampling.LANCZOS)
    offset_x = (tile_size[0] - fitted.width) // 2
    offset_y = 26 + (tile_size[1] - 30 - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, tile_size[0], 24], fill=(38, 70, 83))
    draw.text((6, 6), label[:52], fill="white", font=font)
    return canvas


def build_class_montage(rows: list[dict], output_path: Path, title: str) -> None:
    if not rows:
        return
    tile_size = (320, 240)
    cols = 3
    rows_count = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * tile_size[0], rows_count * tile_size[1] + 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, canvas.width, 36], fill=(42, 157, 143))
    draw.text((10, 10), title, fill="white", font=font)

    for idx, row in enumerate(rows):
        tile = make_crop(
            image_path=Path(row["image_path"]),
            grape_box=row["grape_box_xyxy"],
            picking_box=row["picking_box_xyxy"],
            label=(
                f"{Path(row['file_name']).stem[:18]} "
                f"ry={row['rel_y']:.2f} hr={row['height_ratio']:.2f}"
            ),
            tile_size=tile_size,
        )
        x = (idx % cols) * tile_size[0]
        y = 40 + (idx // cols) * tile_size[1]
        canvas.paste(tile, (x, y))
    canvas.save(output_path)


def build_triptych(montage_paths: list[tuple[str, Path]], output_path: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in montage_paths if path.exists()]
    if not images:
        return
    width = max(image.width for _, image in images)
    total_height = sum(image.height for _, image in images)
    canvas = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    y = 0
    for _, image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output_path)


def plot_overview(rows: list[dict], output_path: Path) -> None:
    class_colors = {
        "near_root": "#2a9d8f",
        "near_middle": "#e76f51",
        "covers_most_stem": "#264653",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for class_name in ("near_root", "near_middle", "covers_most_stem"):
        entries = [row for row in rows if row["semantic_class"] == class_name]
        axes[0, 0].scatter(
            [row["rel_y"] for row in entries],
            [row["height_ratio"] for row in entries],
            s=12,
            alpha=0.45,
            label=class_name,
            color=class_colors[class_name],
        )
        axes[0, 1].scatter(
            [row["area_ratio"] for row in entries],
            [row["aspect_ratio"] for row in entries],
            s=12,
            alpha=0.45,
            label=class_name,
            color=class_colors[class_name],
        )

    class_counts = Counter(row["semantic_class_zh"] for row in rows)
    axes[1, 0].bar(class_counts.keys(), class_counts.values(), color=["#2a9d8f", "#e76f51", "#264653"])

    direction_counts = Counter(row["direction_bucket"] for row in rows)
    axes[1, 1].bar(direction_counts.keys(), direction_counts.values(), color=["#457b9d", "#2a9d8f", "#f4a261"])

    axes[0, 0].set_title("rel_y vs height_ratio")
    axes[0, 0].set_xlabel("Relative center y")
    axes[0, 0].set_ylabel("Picking / Grape height ratio")
    axes[0, 0].legend()

    axes[0, 1].set_title("area_ratio vs aspect_ratio")
    axes[0, 1].set_xlabel("Picking / Grape area ratio")
    axes[0, 1].set_ylabel("Aspect ratio (h / w)")

    axes[1, 0].set_title("Semantic class counts")
    axes[1, 1].set_title("Nearest direction to grape boundary")

    for ax in axes.flat:
        ax.grid(alpha=0.25)

    fig.suptitle("Picking Label Consistency Overview", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "center_offset_x": summarize_feature([float(row["offset_x"]) for row in rows]),
        "center_offset_y": summarize_feature([float(row["offset_y"]) for row in rows]),
        "rel_x": summarize_feature([float(row["rel_x"]) for row in rows]),
        "rel_y": summarize_feature([float(row["rel_y"]) for row in rows]),
        "width_ratio": summarize_feature([float(row["width_ratio"]) for row in rows]),
        "height_ratio": summarize_feature([float(row["height_ratio"]) for row in rows]),
        "area_ratio": summarize_feature([float(row["area_ratio"]) for row in rows]),
        "aspect_ratio": summarize_feature([float(row["aspect_ratio"]) for row in rows]),
        "top_coverage_ratio": summarize_feature([float(row["top_coverage_ratio"]) for row in rows]),
        "vertical_band": dict(Counter(row["vertical_band"] for row in rows)),
        "horizontal_band": dict(Counter(row["horizontal_band"] for row in rows)),
        "direction_bucket": dict(Counter(row["direction_bucket"] for row in rows)),
        "semantic_class": dict(Counter(row["semantic_class_zh"] for row in rows)),
    }


def semantic_drift_judgement(rows: list[dict], splits: list[str]) -> dict:
    ambiguous_rate = safe_div(
        sum(1 for row in rows if float(row["ambiguity_margin"]) < 0.20),
        len(rows),
    )
    split_distributions = {}
    proportions = defaultdict(list)
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        counts = Counter(row["semantic_class"] for row in split_rows)
        total = max(len(split_rows), 1)
        split_distributions[split] = {
            key: safe_div(counts.get(key, 0), total)
            for key in ("near_root", "near_middle", "covers_most_stem")
        }
        for key, value in split_distributions[split].items():
            proportions[key].append(value)

    max_gap = max(
        (max(values) - min(values)) if values else 0.0
        for values in proportions.values()
    )
    cover_share = safe_div(sum(1 for row in rows if row["covers_most_stem_proxy"]), len(rows))
    near_middle_rows = [row for row in rows if row["semantic_class"] == "near_middle"]
    near_middle_rel_y_p50 = percentile([float(row["rel_y"]) for row in near_middle_rows], 0.50)

    if ambiguous_rate >= 0.28 or max_gap >= 0.18:
        level = "明显语义漂移"
    elif cover_share >= 0.10 or near_middle_rel_y_p50 < 0.08:
        level = "无明显大范围语义漂移，但存在标签粒度差异"
    elif ambiguous_rate >= 0.14 or cover_share >= 0.25:
        level = "轻到中度语义漂移"
    else:
        level = "无明显语义漂移"

    reasons = []
    if cover_share >= 0.20:
        reasons.append("同一类别同时包含大量细长长框与紧凑短框，标注粒度明显偏宽。")
    if ambiguous_rate >= 0.14:
        reasons.append("有一部分样本位于不同语义簇的边界附近，说明标签口径并不完全稳定。")
    if max_gap >= 0.10:
        reasons.append("不同 split 之间三类标签占比存在可见波动。")
    if near_middle_rel_y_p50 < 0.08:
        reasons.append("所谓“中段”类的中心位置仍集中在葡萄顶部附近，说明当前标签里真正的中段框并不多。")
    if not reasons:
        reasons.append("三类语义占比和空间分布总体稳定，未见大范围标签口径漂移。")

    return {
        "level": level,
        "ambiguous_rate": ambiguous_rate,
        "split_distribution_gap": max_gap,
        "cover_share": cover_share,
        "near_middle_rel_y_p50": near_middle_rel_y_p50,
        "split_distributions": split_distributions,
        "reasons": reasons,
    }


def build_markdown_report(summary: dict, output_path: Path) -> None:
    localized = summary["localized_name_map"]
    drift = summary["semantic_drift_judgement"]
    overall = summary["overall"]
    lines = [
        "# Picking 标注一致性诊断",
        "",
        "## 结论",
        f"- 当前判断：`{drift['level']}`",
        f"- 歧义样本占比（cluster margin < 0.20）：`{drift['ambiguous_rate']:.4f}`",
        f"- 三类标签在不同 split 的最大占比差：`{drift['split_distribution_gap']:.4f}`",
        f"- “覆盖大部分果梗”占比：`{drift['cover_share']:.4f}`",
        "",
        "## 总体统计",
        f"- 总 picking 数：`{overall['count']}`",
        f"- 中心点相对偏移中位数：`dx={overall['center_offset_x']['p50']:.4f}`, `dy={overall['center_offset_y']['p50']:.4f}`",
        f"- 高度比中位数：`{overall['height_ratio']['p50']:.4f}`",
        f"- 面积比中位数：`{overall['area_ratio']['p50']:.4f}`",
        f"- 长宽比中位数：`{overall['aspect_ratio']['p50']:.4f}`",
        "",
        "## 上 / 侧 / 下分布",
    ]
    for key, value in overall["direction_bucket"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## 三类语义统计"])
    for class_name in ("near_root", "near_middle", "covers_most_stem"):
        stats = summary["by_class"][class_name]
        lines.extend(
            [
                f"### {localized[class_name]}",
                f"- 数量：`{stats['count']}`，占比：`{safe_div(stats['count'], overall['count']):.4f}`",
                f"- 中心点中位数：`rel_x={stats['rel_x']['p50']:.4f}`, `rel_y={stats['rel_y']['p50']:.4f}`",
                f"- 高度比中位数：`{stats['height_ratio']['p50']:.4f}`",
                f"- 面积比中位数：`{stats['area_ratio']['p50']:.4f}`",
                f"- 长宽比中位数：`{stats['aspect_ratio']['p50']:.4f}`",
            ]
        )

    lines.extend(["", "## 语义漂移判断依据"])
    for reason in drift["reasons"]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## 解释",
            "- 这里的“覆盖大部分果梗”是一个基于 bbox 高度比、面积比和形状的代理判断，不是像素级 stem segmentation 结论。",
            "- 如果后续要把标签规范收紧，最值得优先统一的是：`picking` 到底标根部短框，还是允许标成长条 stem 框。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.dataset_root.resolve(), args.splits)
    cluster_summary = cluster_and_label(rows, seed=args.seed)
    localized_name_map = cluster_summary["localized_name_map"]

    rows_by_class = {
        class_name: [row for row in rows if row["semantic_class"] == class_name]
        for class_name in ("near_root", "near_middle", "covers_most_stem")
    }
    rows_by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in args.splits
    }

    for class_rows in rows_by_class.values():
        class_rows.sort(key=lambda row: (float(row["distance_to_center"]), abs(float(row["offset_x"]))))

    random.Random(args.seed).shuffle(rows)

    plot_overview(rows, output_dir / "picking_label_consistency_overview.png")

    montage_paths = []
    for class_name, title in (
        ("near_root", "靠近根部"),
        ("near_middle", "靠近中段"),
        ("covers_most_stem", "覆盖大部分果梗"),
    ):
        sample_rows = rows_by_class[class_name][: args.samples_per_class]
        output_path = output_dir / f"samples_{class_name}.jpg"
        build_class_montage(sample_rows, output_path=output_path, title=title)
        montage_paths.append((title, output_path))
    build_triptych(montage_paths, output_dir / "samples_triptych.jpg")

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "splits": args.splits,
        "overall": summarize_rows(rows),
        "by_class": {
            class_name: summarize_rows(class_rows)
            for class_name, class_rows in rows_by_class.items()
        },
        "by_split": {
            split: summarize_rows(split_rows)
            for split, split_rows in rows_by_split.items()
        },
        "cluster_summary": cluster_summary,
        "localized_name_map": localized_name_map,
    }
    summary["semantic_drift_judgement"] = semantic_drift_judgement(rows, args.splits)

    (output_dir / "picking_label_consistency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_markdown_report(summary, output_dir / "picking_label_consistency_report_zh.md")

    representatives = {
        class_name: [
            {
                "split": row["split"],
                "file_name": row["file_name"],
                "rel_x": row["rel_x"],
                "rel_y": row["rel_y"],
                "height_ratio": row["height_ratio"],
                "area_ratio": row["area_ratio"],
                "aspect_ratio": row["aspect_ratio"],
            }
            for row in rows_by_class[class_name][: args.samples_per_class]
        ]
        for class_name in rows_by_class
    }
    (output_dir / "picking_label_representatives.json").write_text(
        json.dumps(representatives, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] picking label consistency report: {output_dir}")


if __name__ == "__main__":
    main()
