from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [round(i / 10, 1) for i in range(1, 10)]
IOU_THRESHOLD = 0.5


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path: Path) -> dict[int, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("iscrowd", 0)):
            continue
        point = ann.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
        by_image[int(ann["image_id"])].append(
            {
                "bbox": xywh_to_xyxy(ann["bbox"]),
                "has": float(ann.get("has_picking", 0.0)) > 0.5,
                "point": [float(point[0]), float(point[1])],
            }
        )
    return dict(by_image)


def load_predictions(path: Path) -> dict[int, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in data:
        has_score = float(pred.get("has_picking_score", 0.0))
        quality_score = pred.get("point_quality_score")
        quality_score = None if quality_score is None else float(quality_score)
        final_score = pred.get("final_score")
        if final_score is None:
            final_score = has_score * quality_score if quality_score is not None else has_score
        point = pred.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
        by_image[int(pred["image_id"])].append(
            {
                "bbox": xywh_to_xyxy(pred["bbox"]),
                "score": float(pred.get("score", 0.0)),
                "has_score": has_score,
                "quality_score": quality_score,
                "final_score": float(final_score),
                "point": [float(point[0]), float(point[1])],
            }
        )
    return dict(by_image)


def build_matches(gt_by_image: dict[int, list[dict]], pred_by_image: dict[int, list[dict]]) -> list[dict]:
    matches: list[dict] = []
    for image_id, gt_list in gt_by_image.items():
        pred_list = pred_by_image.get(image_id, [])
        used_gt = set()
        for pred in sorted(pred_list, key=lambda item: item["score"], reverse=True):
            best_gt = None
            best_iou = -1.0
            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in used_gt:
                    continue
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            if best_gt is None or best_iou < IOU_THRESHOLD:
                continue
            used_gt.add(best_gt)
            gt = gt_list[best_gt]
            matches.append(
                {
                    "image_id": image_id,
                    "gt_has": gt["has"],
                    "gt_point": gt["point"],
                    "pred_point": pred["point"],
                    "has_score": pred["has_score"],
                    "quality_score": pred["quality_score"],
                    "final_score": pred["final_score"],
                }
            )
    return matches


def summarize(matches: list[dict], threshold: float, score_key: str) -> dict:
    matched_visible = predicted_visible = correct_visible = false_positive = false_negative = 0
    l2_values, dy_values = [], []
    for item in matches:
        gt_visible = bool(item["gt_has"])
        pred_visible = float(item[score_key]) >= threshold
        if gt_visible:
            matched_visible += 1
        if pred_visible:
            predicted_visible += 1
        if gt_visible and pred_visible:
            correct_visible += 1
            dx = float(item["pred_point"][0]) - float(item["gt_point"][0])
            dy = float(item["pred_point"][1]) - float(item["gt_point"][1])
            l2_values.append(math.hypot(dx, dy))
            dy_values.append(abs(dy))
        elif (not gt_visible) and pred_visible:
            false_positive += 1
        elif gt_visible and not pred_visible:
            false_negative += 1

    precision = correct_visible / predicted_visible if predicted_visible else 0.0
    recall = correct_visible / matched_visible if matched_visible else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "pair_count": correct_visible,
        "mean_L2": sum(l2_values) / len(l2_values) if l2_values else None,
        "mean_abs_dy": sum(dy_values) / len(dy_values) if dy_values else None,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_visible": predicted_visible,
        "matched_visible": matched_visible,
    }


def load_ap(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    grape = data.get("primary_checkpoint_split_summary", {}).get("test", {}).get("grape_detection", {})
    return {"AP": grape.get("AP"), "AP50": grape.get("AP50")}


def calibration_rows(matches: list[dict], tau: float, bins: int) -> list[dict]:
    rows = []
    bin_items: list[list[dict]] = [[] for _ in range(bins)]
    for item in matches:
        if not bool(item["gt_has"]) or item.get("quality_score") is None:
            continue
        q = max(0.0, min(1.0, float(item["quality_score"])))
        idx = min(bins - 1, int(q * bins))
        dx = float(item["pred_point"][0]) - float(item["gt_point"][0])
        dy = float(item["pred_point"][1]) - float(item["gt_point"][1])
        l2 = math.hypot(dx, dy)
        actual_quality = math.exp(-l2 / tau)
        bin_items[idx].append({"q": q, "actual": actual_quality, "l2": l2, "dy": abs(dy)})
    for idx, items in enumerate(bin_items):
        if not items:
            rows.append(
                {
                    "bin": idx,
                    "count": 0,
                    "mean_pred_quality": "",
                    "mean_actual_quality": "",
                    "calibration_gap": "",
                    "mean_L2": "",
                    "mean_abs_dy": "",
                }
            )
            continue
        mean_pred = sum(item["q"] for item in items) / len(items)
        mean_actual = sum(item["actual"] for item in items) / len(items)
        rows.append(
            {
                "bin": idx,
                "count": len(items),
                "mean_pred_quality": mean_pred,
                "mean_actual_quality": mean_actual,
                "calibration_gap": mean_pred - mean_actual,
                "mean_L2": sum(item["l2"] for item in items) / len(items),
                "mean_abs_dy": sum(item["dy"] for item in items) / len(items),
            }
        )
    return rows


def fmt(value: float | int | None, nd: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{nd}f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate point_quality score calibration and threshold transfer.")
    parser.add_argument("--current-run", type=Path, default=ROOT / "outputs/grape_point_gppoint_detr_main")
    parser.add_argument("--quality-run", type=Path, default=ROOT / "outputs/grape_point_v7_exp2_point_quality")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out-md", type=Path, default=ROOT / "reports/point_quality_eval_zh.md")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "reports/point_quality_threshold_calibration.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "outputs/grape_point_v7_exp2_point_quality/report/point_quality_eval_summary.json")
    parser.add_argument("--tau", type=float, default=30.0)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = []
    for run_dir in (args.current_run, args.quality_run):
        required.extend(
            [
                run_dir / "report/summary.json",
                run_dir / "predictions/valid_predictions.json",
                run_dir / "predictions/test_predictions.json",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(path) for path in missing))

    valid_gt = load_gt(args.dataset_root / "valid/_annotations.grape_point.json")
    test_gt = load_gt(args.dataset_root / "test/_annotations.grape_point.json")
    models = {
        "current": args.current_run,
        "point_quality": args.quality_run,
    }

    csv_rows = []
    results: dict[str, dict] = {}
    for model_name, run_dir in models.items():
        valid_matches = build_matches(valid_gt, load_predictions(run_dir / "predictions/valid_predictions.json"))
        test_matches = build_matches(test_gt, load_predictions(run_dir / "predictions/test_predictions.json"))
        ap = load_ap(run_dir / "report/summary.json")
        results[model_name] = {"AP": ap["AP"], "AP50": ap["AP50"], "valid": {}, "test": {}, "matches": test_matches}
        for score_name, score_key in [("has_score", "has_score"), ("final_score", "final_score")]:
            valid_sweep = []
            for threshold in THRESHOLDS:
                row = summarize(valid_matches, threshold, score_key)
                row["threshold"] = threshold
                valid_sweep.append(row)
                csv_rows.append(
                    {
                        "row_type": "threshold",
                        "model": model_name,
                        "split": "valid",
                        "score": score_name,
                        "threshold_source": "sweep",
                        "threshold": threshold,
                        "AP": "",
                        "AP50": "",
                        **row,
                    }
                )
            best_valid = max(valid_sweep, key=lambda item: (item["F1"], item["precision"], -abs(item["threshold"] - 0.5)))
            fixed = summarize(test_matches, 0.5, score_key)
            tuned = summarize(test_matches, float(best_valid["threshold"]), score_key)
            results[model_name]["valid"][score_name] = best_valid
            results[model_name]["test"][(score_name, "fixed_0.5")] = fixed
            results[model_name]["test"][(score_name, "valid_best")] = tuned
            for source, threshold, row in [("fixed_0.5", 0.5, fixed), ("valid_best", best_valid["threshold"], tuned)]:
                csv_rows.append(
                    {
                        "row_type": "threshold",
                        "model": model_name,
                        "split": "test",
                        "score": score_name,
                        "threshold_source": source,
                        "threshold": threshold,
                        "AP": ap["AP"],
                        "AP50": ap["AP50"],
                        **row,
                    }
                )

        if model_name == "point_quality":
            for row in calibration_rows(test_matches, args.tau, args.bins):
                csv_rows.append(
                    {
                        "row_type": "calibration",
                        "model": model_name,
                        "split": "test",
                        "score": "point_quality_score",
                        **row,
                    }
                )

    current_target_pair = results["current"]["test"][("has_score", "valid_best")]["pair_count"]
    near_rows = []
    for model_name in ("current", "point_quality"):
        test_matches = results[model_name]["matches"]
        for score_name, score_key in [("has_score", "has_score"), ("final_score", "final_score")]:
            sweep = [(thr, summarize(test_matches, thr, score_key)) for thr in THRESHOLDS]
            near_thr, near_metric = min(sweep, key=lambda item: (abs(item[1]["pair_count"] - current_target_pair), -item[1]["F1"]))
            results[model_name]["test"][(score_name, "near_current_pair")] = near_metric | {"threshold": near_thr}
            near_rows.append([model_name, score_name, fmt(near_thr, 1), near_metric["pair_count"], fmt(near_metric["mean_L2"], 2), fmt(near_metric["mean_abs_dy"], 2), fmt(near_metric["F1"])])

    test_rows = []
    for model_name in ("current", "point_quality"):
        for score_name in ("has_score", "final_score"):
            for source in ("fixed_0.5", "valid_best"):
                metric = results[model_name]["test"][(score_name, source)]
                threshold = 0.5 if source == "fixed_0.5" else results[model_name]["valid"][score_name]["threshold"]
                test_rows.append(
                    [
                        model_name,
                        score_name,
                        source,
                        fmt(threshold, 1),
                        fmt(results[model_name]["AP"]),
                        fmt(metric["F1"]),
                        metric["pair_count"],
                        fmt(metric["mean_L2"], 2),
                        fmt(metric["mean_abs_dy"], 2),
                        fmt(metric["precision"]),
                        fmt(metric["recall"]),
                    ]
                )

    valid_rows = []
    for model_name in ("current", "point_quality"):
        for score_name in ("has_score", "final_score"):
            row = results[model_name]["valid"][score_name]
            valid_rows.append([model_name, score_name, fmt(row["threshold"], 1), fmt(row["F1"]), row["pair_count"], fmt(row["mean_L2"], 2), fmt(row["mean_abs_dy"], 2)])

    pq_has = results["point_quality"]["test"][("has_score", "valid_best")]
    pq_final = results["point_quality"]["test"][("final_score", "valid_best")]
    cur = results["current"]["test"][("has_score", "valid_best")]
    conclusion = []
    conclusion.append(
        f"point_quality final_score 相比 current@valid-has：AP {results['point_quality']['AP'] - results['current']['AP']:+.4f}，"
        f"F1 {pq_final['F1'] - cur['F1']:+.4f}，pair_count {pq_final['pair_count'] - cur['pair_count']:+d}，"
        f"mean L2 {pq_final['mean_L2'] - cur['mean_L2']:+.2f}px，|dy| {pq_final['mean_abs_dy'] - cur['mean_abs_dy']:+.2f}px。"
    )
    conclusion.append(
        f"同一模型内 final_score 相比 has_score：F1 {pq_final['F1'] - pq_has['F1']:+.4f}，"
        f"pair_count {pq_final['pair_count'] - pq_has['pair_count']:+d}，"
        f"mean L2 {pq_final['mean_L2'] - pq_has['mean_L2']:+.2f}px，|dy| {pq_final['mean_abs_dy'] - pq_has['mean_abs_dy']:+.2f}px。"
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "model",
        "split",
        "score",
        "threshold_source",
        "threshold",
        "AP",
        "AP50",
        "precision",
        "recall",
        "F1",
        "pair_count",
        "mean_L2",
        "mean_abs_dy",
        "false_positive",
        "false_negative",
        "predicted_visible",
        "matched_visible",
        "bin",
        "count",
        "mean_pred_quality",
        "mean_actual_quality",
        "calibration_gap",
    ]
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    md = [
        "# v7_exp2_point_quality 阈值与校准评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- quality target 口径：`exp(-L2_pixel / {args.tau:g})`。",
        "- 匹配口径：IoU>=0.5，按 detection score 降序一对一匹配；valid 选 best-F1 阈值，固定到 test。",
        "- `has_score` 为原 has_picking_score；`final_score = has_picking_score * point_quality_score`。",
        "",
        "## Valid 选阈值",
        "",
        md_table(["model", "score", "best thr", "valid F1", "valid pair", "valid mean L2", "valid |dy|"], valid_rows),
        "",
        "## Test 结果",
        "",
        md_table(["model", "score", "source", "thr", "AP", "F1", "pair_count", "mean L2", "|dy|", "precision", "recall"], test_rows),
        "",
        "## 相近 pair_count 对比",
        "",
        f"- 目标 pair_count 使用 current@valid-has 的 {current_target_pair}。",
        "",
        md_table(["model", "score", "thr", "pair_count", "mean L2", "|dy|", "F1"], near_rows),
        "",
        "## 结论",
        "",
        *[f"- {item}" for item in conclusion],
        "- 若 final_score 只降低 L2 但明显牺牲 pair_count/F1，应写作质量校准机制，而不是替代 has_picking 的主分数。",
        "",
        f"CSV: `{args.out_csv.relative_to(ROOT).as_posix()}`",
    ]
    args.out_md.write_text("\n".join(md), encoding="utf-8")

    serializable_results = {}
    for model_name, payload in results.items():
        serializable_results[model_name] = {
            "AP": payload["AP"],
            "AP50": payload["AP50"],
            "valid": payload["valid"],
            "test": {f"{key[0]}__{key[1]}": value for key, value in payload["test"].items()},
        }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(serializable_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
