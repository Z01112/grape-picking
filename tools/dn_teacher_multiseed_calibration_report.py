from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
OUT_MD = REPORT_DIR / "dn_teacher_roi_multiseed_calibration_zh.md"
OUT_CSV = REPORT_DIR / "dn_teacher_roi_multiseed_calibration.csv"
OUT_F1_CURVE = REPORT_DIR / "dn_teacher_roi_f1_threshold_curve.png"
OUT_PR_CURVE = REPORT_DIR / "dn_teacher_roi_pr_curve.png"

THRESHOLDS = [round(i / 10, 1) for i in range(1, 10)]
IOU_THRESHOLD = 0.5

MODEL_GROUPS = {
    "current": {
        "label": "current / GPPoint-DETR",
        "runs": {
            "main": ROOT / "outputs/grape_point_gppoint_detr_main",
            "repro1": ROOT / "outputs/grape_point_gppoint_detr_main_repro1",
            "seed2026": ROOT / "outputs/grape_point_gppoint_detr_main_seed2026",
        },
    },
    "dn_teacher_roi": {
        "label": "dn_teacher_roi",
        "runs": {
            "main": ROOT / "outputs/grape_point_v7_exp2_dn_teacher_roi",
            "repro1": ROOT / "outputs/grape_point_v7_exp2_dn_teacher_roi_repro1",
            "seed2026": ROOT / "outputs/grape_point_v7_exp2_dn_teacher_roi_seed2026",
        },
    },
}


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
        image_id = int(ann["image_id"])
        bbox_xyxy = xywh_to_xyxy(ann["bbox"])
        by_image[image_id].append(
            {
                "bbox_xyxy": bbox_xyxy,
                "has_picking": bool(float(ann.get("has_picking", 0.0)) > 0.5),
                "picking_point": [float(v) for v in ann.get("picking_point", [0.0, 0.0])],
            }
        )
    return dict(by_image)


def load_predictions(path: Path) -> dict[int, list[dict]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in items:
        image_id = int(pred["image_id"])
        by_image[image_id].append(
            {
                "bbox_xyxy": xywh_to_xyxy(pred["bbox"]),
                "score": float(pred.get("score", 0.0)),
                "has_picking_score": float(pred.get("has_picking_score", 0.0)),
                "picking_point": [float(v) for v in pred.get("picking_point", [0.0, 0.0])],
            }
        )
    return dict(by_image)


def build_matches(gt_by_image: dict[int, list[dict]], pred_by_image: dict[int, list[dict]]) -> list[dict]:
    matches = []
    for image_id, gt_list in gt_by_image.items():
        pred_list = pred_by_image.get(image_id, [])
        if not gt_list or not pred_list:
            continue
        order = sorted(range(len(pred_list)), key=lambda idx: pred_list[idx]["score"], reverse=True)
        used_gt = set()
        for pred_idx in order:
            pred = pred_list[pred_idx]
            best_gt = None
            best_iou = -1.0
            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in used_gt:
                    continue
                iou = box_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
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
                    "gt_idx": best_gt,
                    "gt_has_picking": gt["has_picking"],
                    "gt_point": gt["picking_point"],
                    "pred_has_picking_score": pred["has_picking_score"],
                    "pred_point": pred["picking_point"],
                }
            )
    return matches


def summarize(matches: list[dict], threshold: float) -> dict:
    matched_visible = predicted_visible = correct_visible = false_positive = false_negative = 0
    l2_values, dy_values = [], []
    for item in matches:
        gt_visible = bool(item["gt_has_picking"])
        pred_visible = float(item["pred_has_picking_score"]) >= threshold
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
    test_summary = data.get("primary_checkpoint_split_summary", {}).get("test", {})
    grape = test_summary.get("grape_detection", {})
    return {"AP": grape.get("AP"), "AP50": grape.get("AP50")}


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def fmt(value: float | int | None, nd: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{nd}f}"


def fmt_pm(mean: float | None, std: float | None, nd: int = 4) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean:.{nd}f}±{std:.{nd}f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def require_files() -> None:
    missing = []
    for group in MODEL_GROUPS.values():
        for run_dir in group["runs"].values():
            for path in [
                run_dir / "report/summary.json",
                run_dir / "predictions/valid_predictions.json",
                run_dir / "predictions/test_predictions.json",
            ]:
                if not path.exists():
                    missing.append(path)
    if missing:
        lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing files required for multi-seed calibration report:\n{lines}")


def main() -> int:
    require_files()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    valid_gt = load_gt(ROOT / "dataset/valid/_annotations.grape_point.json")
    test_gt = load_gt(ROOT / "dataset/test/_annotations.grape_point.json")

    csv_rows = []
    run_results: dict[str, dict[str, dict]] = defaultdict(dict)
    curves: dict[str, dict[str, list[dict]]] = defaultdict(dict)

    for group_name, group in MODEL_GROUPS.items():
        for run_name, run_dir in group["runs"].items():
            valid_matches = build_matches(valid_gt, load_predictions(run_dir / "predictions/valid_predictions.json"))
            test_matches = build_matches(test_gt, load_predictions(run_dir / "predictions/test_predictions.json"))
            valid_sweep = []
            for threshold in THRESHOLDS:
                row = summarize(valid_matches, threshold)
                row["threshold"] = threshold
                valid_sweep.append(row)
                csv_rows.append(
                    {
                        "group": group_name,
                        "run": run_name,
                        "split": "valid",
                        "threshold_source": "sweep",
                        "threshold": threshold,
                        **row,
                        "AP": "",
                        "AP50": "",
                    }
                )

            best_valid = max(valid_sweep, key=lambda item: (item["F1"], item["precision"], -abs(item["threshold"] - 0.5)))
            ap = load_ap(run_dir / "report/summary.json")
            test_fixed = summarize(test_matches, 0.5)
            test_valid = summarize(test_matches, float(best_valid["threshold"]))
            for source, threshold, result in [
                ("fixed_0.5", 0.5, test_fixed),
                ("valid_best", float(best_valid["threshold"]), test_valid),
            ]:
                csv_rows.append(
                    {
                        "group": group_name,
                        "run": run_name,
                        "split": "test",
                        "threshold_source": source,
                        "threshold": threshold,
                        **result,
                        **ap,
                    }
                )

            run_results[group_name][run_name] = {
                "valid_best_threshold": float(best_valid["threshold"]),
                "valid_best_F1": float(best_valid["F1"]),
                "AP": ap["AP"],
                "AP50": ap["AP50"],
                "test_fixed": test_fixed,
                "test_valid": test_valid,
            }
            curves[group_name][run_name] = valid_sweep

    fieldnames = [
        "group",
        "run",
        "split",
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
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    plt.figure(figsize=(8.5, 5.0), dpi=140)
    for group_name, runs in curves.items():
        for run_name, rows in runs.items():
            style = "-" if group_name == "dn_teacher_roi" else "--"
            plt.plot([r["threshold"] for r in rows], [r["F1"] for r in rows], style, marker="o", label=f"{group_name}:{run_name}")
    plt.xlabel("has_picking threshold")
    plt.ylabel("valid F1")
    plt.title("Valid F1-threshold curve")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_F1_CURVE)
    plt.close()

    plt.figure(figsize=(8.5, 5.0), dpi=140)
    for group_name, runs in curves.items():
        for run_name, rows in runs.items():
            style = "-" if group_name == "dn_teacher_roi" else "--"
            plt.plot([r["recall"] for r in rows], [r["precision"] for r in rows], style, marker="o", label=f"{group_name}:{run_name}")
    plt.xlabel("valid recall")
    plt.ylabel("valid precision")
    plt.title("Valid precision-recall curve from threshold sweep")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_PR_CURVE)
    plt.close()

    per_run_rows = []
    for group_name, runs in run_results.items():
        for run_name, result in runs.items():
            valid = result["test_valid"]
            per_run_rows.append(
                [
                    group_name,
                    run_name,
                    fmt(result["valid_best_threshold"], 1),
                    fmt(result["valid_best_F1"]),
                    fmt(result["AP"]),
                    fmt(valid["F1"]),
                    valid["pair_count"],
                    fmt(valid["mean_L2"], 2),
                    fmt(valid["mean_abs_dy"], 2),
                ]
            )

    summary_rows = []
    for group_name, runs in run_results.items():
        for source_key, source_label in [("test_fixed", "fixed_0.5"), ("test_valid", "valid_best")]:
            metrics = {
                "AP": [item["AP"] for item in runs.values()],
                "F1": [item[source_key]["F1"] for item in runs.values()],
                "pair_count": [item[source_key]["pair_count"] for item in runs.values()],
                "mean_L2": [item[source_key]["mean_L2"] for item in runs.values()],
                "mean_abs_dy": [item[source_key]["mean_abs_dy"] for item in runs.values()],
            }
            row = [group_name, source_label]
            for key in ["AP", "F1", "pair_count", "mean_L2", "mean_abs_dy"]:
                mean, std = mean_std(metrics[key])
                row.append(fmt_pm(mean, std, 2 if key in {"pair_count", "mean_L2", "mean_abs_dy"} else 4))
            summary_rows.append(row)

    dn_valid = [row for row in summary_rows if row[0] == "dn_teacher_roi" and row[1] == "valid_best"][0]
    current_valid = [row for row in summary_rows if row[0] == "current" and row[1] == "valid_best"][0]
    current_fixed = [row for row in summary_rows if row[0] == "current" and row[1] == "fixed_0.5"][0]

    md = [
        "# DN Teacher ROI 多 seed calibration 评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 口径：每个 run 在 valid set 上扫描 has_picking 阈值 0.1~0.9，选择 valid best-F1 阈值，再固定到 test。",
        f"- 匹配：IoU≥{IOU_THRESHOLD:.1f}，按 detection score 降序一对一匹配，与 `coco_eval.py` 的 has_picking/point 口径一致。",
        "- AP/AP50 来自对应 run 的 `report/summary.json`，has_picking 阈值不改变 bbox 检测。",
        "",
        "## 单 run：valid-selected threshold 对 test 的结果",
        "",
        md_table(
            ["group", "run", "valid best thr", "valid best F1", "test AP", "test F1", "test pair", "test mean L2", "test |dy|"],
            per_run_rows,
        ),
        "",
        "## mean±std 汇总",
        "",
        md_table(
            ["group", "threshold source", "AP", "F1", "pair_count", "mean L2", "|dy|"],
            summary_rows,
        ),
        "",
        "## 重点对比",
        "",
        md_table(
            ["setting", "AP", "F1", "pair_count", "mean L2", "|dy|"],
            [
                ["current@0.5", *current_fixed[2:]],
                ["current@valid-thr", *current_valid[2:]],
                ["dn_teacher_roi@valid-thr", *dn_valid[2:]],
            ],
        ),
        "",
        "## calibration 图表",
        "",
        f"- F1-threshold curve: `{OUT_F1_CURVE.relative_to(ROOT).as_posix()}`",
        f"- PR curve: `{OUT_PR_CURVE.relative_to(ROOT).as_posix()}`",
        "",
        "## 结论写作建议",
        "",
        "- 若 `dn_teacher_roi@valid-thr` 在 F1、pair_count、mean L2、|dy| 的 mean±std 上同时优于 `current@valid-thr`，且 AP 没有明显劣化，可考虑作为新主模型候选。",
        "- 若主要收益来自阈值校准，或 AP/定位误差存在互相抵消，则应写为辅助机制实验：DN teacher ROI 改善 has_picking score 校准或 recall，但尚不足以替代 GPPoint-DETR/current。",
        "- 最终论文不要把单次代表性 run 的阈值结论替代多 seed mean±std 结论。",
        "",
        f"CSV: `{OUT_CSV.relative_to(ROOT).as_posix()}`",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_F1_CURVE}")
    print(f"wrote {OUT_PR_CURVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
