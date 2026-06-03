from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grape_point_eval_utils import (
    _box_iou_matrix,
    match_prediction_record,
    normalize_prediction_record,
    point_in_box,
    safe_float,
    summarize_point_cases,
)


DEFAULT_INPUTS = [
    (
        "V7_EXP2_MAIN",
        REPO_ROOT / "outputs/08_eval_unification/v7_exp2_unified_report/test_prediction_records.json",
        True,
    ),
    (
        "EMA_BIFPN",
        REPO_ROOT / "outputs/08_eval_unification/ema_bifpn_unified_report/test_prediction_records.json",
        True,
    ),
    (
        "RELCAL_V1_PROBE20_FAILED_REF",
        REPO_ROOT
        / "outputs/09_model_improvement/ema_bifpn_relcal_v1_probe20/report/test_prediction_records.json",
        False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate-level picking point selection diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/10_candidate_diagnostics")
    parser.add_argument("--has-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--include-relcal-reference", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported records payload: {path}")
    return [normalize_prediction_record(item) for item in payload]


def point_l2(pred: dict, gt: dict) -> float:
    pred_point = pred.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    gt_point = gt.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    return float(math.hypot(float(pred_point[0]) - float(gt_point[0]), float(pred_point[1]) - float(gt_point[1])))


def visible(pred: dict, threshold: float) -> bool:
    return safe_float(pred.get("visible_score", pred.get("has_picking_score", 0.0))) >= threshold


def candidate_row(pred_idx: int, pred: dict, gt: dict, iou: float, threshold: float) -> dict:
    pred_point = pred.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    return {
        "pred_index": int(pred_idx),
        "iou": float(iou),
        "l2": point_l2(pred, gt),
        "score": safe_float(pred.get("score", 0.0)),
        "raw_has_picking_score": safe_float(pred.get("raw_has_picking_score", pred.get("has_picking_score", 0.0))),
        "has_picking_score": safe_float(pred.get("has_picking_score", pred.get("visible_score", 0.0))),
        "visible_score": safe_float(pred.get("visible_score", pred.get("has_picking_score", 0.0))),
        "score_visible_product": safe_float(pred.get("score", 0.0))
        * safe_float(pred.get("visible_score", pred.get("has_picking_score", 0.0))),
        "pred_visible": bool(visible(pred, threshold)),
        "point_inside_gt_box": bool(point_in_box(pred_point, gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]))),
        "point_inside_pred_box": bool(point_in_box(pred_point, pred.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]))),
    }


def best_by(rows: list[dict], key: str, reverse: bool = False) -> dict | None:
    if not rows:
        return None
    return sorted(rows, key=lambda item: safe_float(item.get(key, 0.0)), reverse=reverse)[0]


def rank_of_pred(rows: list[dict], pred_index: int | None, key: str, reverse: bool = True) -> int | None:
    if pred_index is None or not rows:
        return None
    sorted_rows = sorted(rows, key=lambda item: safe_float(item.get(key, 0.0)), reverse=reverse)
    for rank, item in enumerate(sorted_rows, start=1):
        if int(item["pred_index"]) == int(pred_index):
            return rank
    return None


def case_from_candidate(record: dict, gt_idx: int, gt: dict, candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    pred = record["pred_instances"][int(candidate["pred_index"])]
    gt_point = gt.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    pred_point = pred.get("picking_point", [0.0, 0.0]) or [0.0, 0.0]
    return {
        "image_id": int(record["image_id"]),
        "file_name": record.get("file_name", ""),
        "gt_index": int(gt_idx),
        "pred_index": int(candidate["pred_index"]),
        "iou": safe_float(candidate.get("iou")),
        "l2_px": safe_float(candidate.get("l2")),
        "dx_px": float(pred_point[0]) - float(gt_point[0]),
        "dy_px": float(pred_point[1]) - float(gt_point[1]),
        "pred_score": safe_float(candidate.get("score")),
        "pred_visible_score": safe_float(candidate.get("visible_score")),
        "pred_raw_has_picking_score": safe_float(candidate.get("raw_has_picking_score")),
    }


def pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def analyze_model(name: str, path: Path, has_threshold: float, iou_threshold: float) -> dict:
    records = load_records(path)
    by_gt_rows: list[dict] = []
    taxonomy_rows: list[dict] = []
    ranking_gap_rows: list[dict] = []
    score_samples_standard: list[dict] = []
    score_samples_all_iou50: list[dict] = []
    standard_cases: list[dict] = []
    oracle_iou50_cases: list[dict] = []
    oracle_iou30_cases: list[dict] = []
    oracle_any_visible_cases: list[dict] = []
    alternate_stats = {
        "standard_l2_gt30_count": 0,
        "standard_l2_gt30_with_iou50_l2_le30_alternate": 0,
        "standard_l2_gt50_count": 0,
        "standard_l2_gt50_with_iou50_l2_le30_alternate": 0,
        "alternate_good_scores": [],
        "alternate_good_visible_scores": [],
        "alternate_good_raw_has_scores": [],
        "alternate_good_rank_by_score": [],
        "alternate_good_rank_by_visible_score": [],
        "best_point_rank_by_score": [],
        "best_point_rank_by_visible_score": [],
    }

    visible_gt_total = 0
    for record in records:
        matched = match_prediction_record(record, iou_threshold=iou_threshold, has_picking_threshold=has_threshold)
        standard_by_gt = {
            int(case["gt_index"]): case
            for case in matched["matched_pairs"]
            if int(case.get("gt_index", -1)) >= 0
        }
        gt_instances = record.get("gt_instances", [])
        pred_instances = record.get("pred_instances", [])
        gt_boxes = [gt["bbox_xyxy"] for gt in gt_instances]
        pred_boxes = [pred["bbox_xyxy"] for pred in pred_instances]
        ious = _box_iou_matrix(pred_boxes, gt_boxes).numpy() if pred_instances and gt_instances else np.zeros((0, 0))
        visible_gt_indices = [idx for idx, gt in enumerate(gt_instances) if bool(gt.get("has_picking", False))]
        visible_gt_total += len(visible_gt_indices)

        for gt_idx in visible_gt_indices:
            gt = gt_instances[gt_idx]
            candidates = [
                candidate_row(pred_idx, pred, gt, float(ious[pred_idx, gt_idx]), has_threshold)
                for pred_idx, pred in enumerate(pred_instances)
            ]
            iou50_candidates = [item for item in candidates if item["iou"] >= 0.5]
            iou30_candidates = [item for item in candidates if item["iou"] >= 0.3]
            any_visible_candidates = [item for item in candidates if item["pred_visible"]]

            best_iou50_l2 = best_by(iou50_candidates, "l2")
            best_iou30_l2 = best_by(iou30_candidates, "l2")
            best_any_visible_l2 = best_by(any_visible_candidates, "l2")
            best_score = best_by(iou50_candidates, "score", reverse=True)
            best_visible_score = best_by(iou50_candidates, "visible_score", reverse=True)
            best_score_visible_product = best_by(iou50_candidates, "score_visible_product", reverse=True)

            if best_iou50_l2 is not None:
                oracle_iou50_cases.append(case_from_candidate(record, gt_idx, gt, best_iou50_l2))
                alternate_stats["best_point_rank_by_score"].append(rank_of_pred(iou50_candidates, best_iou50_l2["pred_index"], "score"))
                alternate_stats["best_point_rank_by_visible_score"].append(
                    rank_of_pred(iou50_candidates, best_iou50_l2["pred_index"], "visible_score")
                )
            if best_iou30_l2 is not None:
                oracle_iou30_cases.append(case_from_candidate(record, gt_idx, gt, best_iou30_l2))
            if best_any_visible_l2 is not None:
                oracle_any_visible_cases.append(case_from_candidate(record, gt_idx, gt, best_any_visible_l2))

            standard = standard_by_gt.get(gt_idx)
            standard_exists = standard is not None
            standard_visible = bool(standard.get("pred_has_picking", False)) if standard_exists else False
            standard_l2 = safe_float(standard.get("l2_px")) if standard_exists and standard_visible else None
            standard_iou = safe_float(standard.get("iou")) if standard_exists else None
            standard_pred_idx = int(standard["pred_index"]) if standard_exists else None

            if standard_exists and standard_visible:
                standard_cases.append(standard)
                std_pred = pred_instances[standard_pred_idx]
                score_samples_standard.append(
                    {
                        "score": safe_float(std_pred.get("score")),
                        "raw_has_picking_score": safe_float(std_pred.get("raw_has_picking_score")),
                        "has_picking_score": safe_float(std_pred.get("has_picking_score")),
                        "visible_score": safe_float(std_pred.get("visible_score")),
                        "score_visible_product": safe_float(std_pred.get("score")) * safe_float(std_pred.get("visible_score")),
                        "l2": standard_l2,
                    }
                )

            for item in iou50_candidates:
                score_samples_all_iou50.append(item)

            good_alt = [item for item in iou50_candidates if item["l2"] <= 30.0 and item["pred_index"] != standard_pred_idx]
            good_alt_best = best_by(good_alt, "l2")
            if standard_exists and standard_visible and standard_l2 is not None and standard_l2 > 30.0:
                alternate_stats["standard_l2_gt30_count"] += 1
                if good_alt_best is not None:
                    alternate_stats["standard_l2_gt30_with_iou50_l2_le30_alternate"] += 1
            if standard_exists and standard_visible and standard_l2 is not None and standard_l2 > 50.0:
                alternate_stats["standard_l2_gt50_count"] += 1
                if good_alt_best is not None:
                    alternate_stats["standard_l2_gt50_with_iou50_l2_le30_alternate"] += 1
            if good_alt_best is not None:
                alternate_stats["alternate_good_scores"].append(good_alt_best["score"])
                alternate_stats["alternate_good_visible_scores"].append(good_alt_best["visible_score"])
                alternate_stats["alternate_good_raw_has_scores"].append(good_alt_best["raw_has_picking_score"])
                alternate_stats["alternate_good_rank_by_score"].append(rank_of_pred(iou50_candidates, good_alt_best["pred_index"], "score"))
                alternate_stats["alternate_good_rank_by_visible_score"].append(
                    rank_of_pred(iou50_candidates, good_alt_best["pred_index"], "visible_score")
                )

            standard_pred = pred_instances[standard_pred_idx] if standard_pred_idx is not None else None
            standard_point_cross = False
            if standard_pred is not None and standard_visible:
                current_l2 = point_l2(standard_pred, gt)
                other_l2 = [
                    point_l2(standard_pred, other_gt)
                    for other_idx, other_gt in enumerate(gt_instances)
                    if other_idx != gt_idx and bool(other_gt.get("has_picking", False))
                ]
                standard_point_cross = bool(other_l2 and (min(other_l2) + 5.0 < current_l2))

            labels = []
            if not iou50_candidates:
                labels.append("no_iou50_candidate")
            if standard_exists and not standard_visible:
                labels.append("matched_visible_false")
            if standard_exists and standard_visible and standard_l2 is not None and standard_l2 > 30.0:
                if any(item["l2"] <= 30.0 for item in iou50_candidates):
                    labels.append("selected_point_bad_but_good_candidate_exists")
                else:
                    labels.append("selected_point_bad_no_good_candidate")
            if standard_exists and standard_visible and standard_iou is not None and standard_iou >= 0.85 and standard_l2 is not None and standard_l2 > 30.0:
                labels.append("high_iou_point_bad")
            if standard_exists and standard_visible and standard_iou is not None and 0.5 <= standard_iou < 0.7 and standard_l2 is not None and standard_l2 > 30.0:
                labels.append("low_iou_propagation")
            if standard_point_cross:
                labels.append("cross_instance_suspected")
            if standard_exists and standard_visible and standard_l2 is not None and standard_l2 <= 30.0:
                labels.append("good_success")
            if not labels:
                labels.append("unclassified")

            best_rank_score = rank_of_pred(iou50_candidates, best_iou50_l2["pred_index"], "score") if best_iou50_l2 else None
            best_rank_visible = rank_of_pred(iou50_candidates, best_iou50_l2["pred_index"], "visible_score") if best_iou50_l2 else None
            row = {
                "model": name,
                "image_id": int(record["image_id"]),
                "file_name": record.get("file_name", ""),
                "gt_index": int(gt_idx),
                "iou50_candidate_count": len(iou50_candidates),
                "iou30_candidate_count": len(iou30_candidates),
                "any_visible_candidate_count": len(any_visible_candidates),
                "standard_pred_index": standard_pred_idx,
                "standard_iou": standard_iou,
                "standard_l2": standard_l2,
                "standard_score": safe_float(standard.get("pred_score")) if standard_exists else None,
                "standard_raw_has_score": safe_float(standard.get("pred_raw_has_picking_score")) if standard_exists else None,
                "standard_has_picking_score": safe_float(standard.get("pred_has_picking_score")) if standard_exists else None,
                "standard_visible_score": safe_float(standard.get("pred_visible_score")) if standard_exists else None,
                "standard_pred_visible": standard_visible,
                "best_iou50_l2": safe_float(best_iou50_l2.get("l2")) if best_iou50_l2 else None,
                "best_iou50_score": safe_float(best_iou50_l2.get("score")) if best_iou50_l2 else None,
                "best_iou50_raw_has_score": safe_float(best_iou50_l2.get("raw_has_picking_score")) if best_iou50_l2 else None,
                "best_iou50_visible_score": safe_float(best_iou50_l2.get("visible_score")) if best_iou50_l2 else None,
                "best_iou30_l2": safe_float(best_iou30_l2.get("l2")) if best_iou30_l2 else None,
                "best_any_visible_l2": safe_float(best_any_visible_l2.get("l2")) if best_any_visible_l2 else None,
                "best_score_candidate_l2": safe_float(best_score.get("l2")) if best_score else None,
                "best_visible_score_candidate_l2": safe_float(best_visible_score.get("l2")) if best_visible_score else None,
                "best_score_visible_product_candidate_l2": safe_float(best_score_visible_product.get("l2")) if best_score_visible_product else None,
                "best_candidate_rank_by_score": best_rank_score,
                "best_candidate_rank_by_visible_score": best_rank_visible,
                "taxonomy_label": ";".join(labels),
            }
            by_gt_rows.append(row)
            taxonomy_rows.append(row)
            if best_iou50_l2 is not None:
                ranking_gap_rows.append(
                    {
                        "model": name,
                        "image_id": int(record["image_id"]),
                        "file_name": record.get("file_name", ""),
                        "gt_index": int(gt_idx),
                        "best_point_l2": safe_float(best_iou50_l2.get("l2")),
                        "best_point_score": safe_float(best_iou50_l2.get("score")),
                        "best_point_visible_score": safe_float(best_iou50_l2.get("visible_score")),
                        "best_point_rank_by_score": best_rank_score,
                        "best_point_rank_by_visible_score": best_rank_visible,
                        "best_score_candidate_l2": safe_float(best_score.get("l2")) if best_score else None,
                        "best_visible_score_candidate_l2": safe_float(best_visible_score.get("l2")) if best_visible_score else None,
                        "standard_l2": standard_l2,
                        "standard_visible": standard_visible,
                    }
                )

    def summary_from_cases(cases: list[dict]) -> dict:
        valid_cases = [case for case in cases if case is not None]
        return summarize_point_cases(valid_cases)

    def avg(values: list[Any]) -> float | None:
        values = [safe_float(v) for v in values if v is not None]
        return float(np.mean(values)) if values else None

    standard_summary = summary_from_cases(standard_cases)
    oracle_iou50_summary = summary_from_cases(oracle_iou50_cases)
    oracle_iou30_summary = summary_from_cases(oracle_iou30_cases)
    oracle_any_visible_summary = summary_from_cases(oracle_any_visible_cases)
    taxonomy_counts = Counter()
    for row in taxonomy_rows:
        for label in str(row["taxonomy_label"]).split(";"):
            taxonomy_counts[label] += 1

    score_correlations = []
    for group_name, samples in (
        ("standard_matched_visible_pairs", score_samples_standard),
        ("all_iou50_candidates", score_samples_all_iou50),
    ):
        neg_l2 = [-safe_float(item.get("l2")) for item in samples]
        score_correlations.append(
            {
                "model": name,
                "sample_group": group_name,
                "sample_count": len(samples),
                "corr_score_neg_l2": pearson([safe_float(item.get("score")) for item in samples], neg_l2),
                "corr_raw_has_neg_l2": pearson([safe_float(item.get("raw_has_picking_score")) for item in samples], neg_l2),
                "corr_visible_score_neg_l2": pearson([safe_float(item.get("visible_score")) for item in samples], neg_l2),
                "corr_score_visible_product_neg_l2": pearson(
                    [safe_float(item.get("score_visible_product")) for item in samples],
                    neg_l2,
                ),
            }
        )

    return {
        "model": name,
        "path": str(path),
        "records": records,
        "by_gt_rows": by_gt_rows,
        "taxonomy_rows": taxonomy_rows,
        "ranking_gap_rows": ranking_gap_rows,
        "score_correlations": score_correlations,
        "summary": {
            "visible_gt_total": visible_gt_total,
            "standard_pair_count": standard_summary["pair_count"],
            "standard_mean_l2": standard_summary["mean_l2_px"],
            "standard_ppl30": standard_summary["ppl_sr_30"],
            "oracle_iou50_pair_count": oracle_iou50_summary["pair_count"],
            "oracle_iou50_mean_l2": oracle_iou50_summary["mean_l2_px"],
            "oracle_iou50_ppl30": oracle_iou50_summary["ppl_sr_30"],
            "oracle_iou30_pair_count": oracle_iou30_summary["pair_count"],
            "oracle_iou30_mean_l2": oracle_iou30_summary["mean_l2_px"],
            "oracle_iou30_ppl30": oracle_iou30_summary["ppl_sr_30"],
            "oracle_any_visible_pair_count": oracle_any_visible_summary["pair_count"],
            "oracle_any_visible_mean_l2": oracle_any_visible_summary["mean_l2_px"],
            "oracle_any_visible_ppl30": oracle_any_visible_summary["ppl_sr_30"],
            "standard_l2_gt30_count": alternate_stats["standard_l2_gt30_count"],
            "standard_l2_gt30_with_iou50_l2_le30_alternate": alternate_stats[
                "standard_l2_gt30_with_iou50_l2_le30_alternate"
            ],
            "standard_l2_gt50_count": alternate_stats["standard_l2_gt50_count"],
            "standard_l2_gt50_with_iou50_l2_le30_alternate": alternate_stats[
                "standard_l2_gt50_with_iou50_l2_le30_alternate"
            ],
            "alternate_good_mean_score": avg(alternate_stats["alternate_good_scores"]),
            "alternate_good_mean_visible_score": avg(alternate_stats["alternate_good_visible_scores"]),
            "alternate_good_mean_raw_has_score": avg(alternate_stats["alternate_good_raw_has_scores"]),
            "alternate_good_mean_rank_by_score": avg(alternate_stats["alternate_good_rank_by_score"]),
            "alternate_good_mean_rank_by_visible_score": avg(alternate_stats["alternate_good_rank_by_visible_score"]),
            "best_point_mean_rank_by_score": avg(alternate_stats["best_point_rank_by_score"]),
            "best_point_mean_rank_by_visible_score": avg(alternate_stats["best_point_rank_by_visible_score"]),
            "taxonomy_counts": dict(taxonomy_counts),
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_conclusion(results: list[dict]) -> tuple[str, str]:
    main = next((item for item in results if item["model"] == "EMA_BIFPN"), results[-1])
    s = main["summary"]
    bad30 = int(s.get("standard_l2_gt30_count", 0))
    bad30_alt = int(s.get("standard_l2_gt30_with_iou50_l2_le30_alternate", 0))
    alt_rate = bad30_alt / bad30 if bad30 else 0.0
    oracle_pair_gain = int(s.get("oracle_iou50_pair_count", 0)) - int(s.get("standard_pair_count", 0))
    oracle_mean_gain = safe_float(s.get("standard_mean_l2")) - safe_float(s.get("oracle_iou50_mean_l2"))
    if bad30 >= 20 and alt_rate >= 0.30 and oracle_pair_gain >= 15 and oracle_mean_gain >= 3.0:
        return (
            "A. 进入 point-aware query reranking/postprocess",
            "EMA_BIFPN 的 bad standard cases 中有较多 IoU>=0.5 且 L2<=30 的候选，oracle_iou50 相比标准输出同时提升 pair 和 mean L2，说明好点已经在候选池里但没有被当前 score/visible_score 选中。",
        )
    if safe_float(s.get("oracle_iou50_mean_l2")) > 23.0 or safe_float(s.get("oracle_iou50_ppl30")) < 0.80:
        return (
            "B. 进入 point localization structure 改进",
            "IoU>=0.5 候选的 oracle 仍然不够好，说明候选池内缺少稳定高精度点，单纯重排序空间有限。",
        )
    return (
        "C. 暂停模型改进，整理论文",
        "候选诊断显示可改空间不足，或收益主要来自牺牲 pair 的筛选协议；当前更适合固定 EMA_BIFPN/HP protocol 作为论文资产。",
    )


def build_markdown(results: list[dict], conclusion: tuple[str, str], skipped: list[dict]) -> str:
    lines = [
        "# Point Candidate Selection Diagnostic",
        "",
        "本报告只使用现有 prediction records 做候选级诊断；未训练、未生成 checkpoint、未改模型结构。",
        "",
        "## Oracle Gap Summary",
        "",
        "| Model | visible GT | standard pair | standard mean L2 | standard PPL-SR@30 | oracle IoU50 pair | oracle IoU50 mean L2 | oracle IoU50 PPL-SR@30 | oracle IoU30 pair | oracle any-visible pair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        s = result["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["model"],
                    fmt(s.get("visible_gt_total"), 0),
                    fmt(s.get("standard_pair_count"), 0),
                    fmt(s.get("standard_mean_l2"), 2),
                    fmt(s.get("standard_ppl30"), 4),
                    fmt(s.get("oracle_iou50_pair_count"), 0),
                    fmt(s.get("oracle_iou50_mean_l2"), 2),
                    fmt(s.get("oracle_iou50_ppl30"), 4),
                    fmt(s.get("oracle_iou30_pair_count"), 0),
                    fmt(s.get("oracle_any_visible_pair_count"), 0),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Bad Case Alternate Candidate Test",
            "",
            "| Model | standard L2>30 | has IoU50 alternate L2<=30 | rate | standard L2>50 | has IoU50 alternate L2<=30 | best point rank by score | best point rank by visible_score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        s = result["summary"]
        bad30 = int(s.get("standard_l2_gt30_count", 0))
        alt30 = int(s.get("standard_l2_gt30_with_iou50_l2_le30_alternate", 0))
        bad50 = int(s.get("standard_l2_gt50_count", 0))
        alt50 = int(s.get("standard_l2_gt50_with_iou50_l2_le30_alternate", 0))
        lines.append(
            f"| {result['model']} | {bad30} | {alt30} | {fmt(alt30 / bad30 if bad30 else None)} | "
            f"{bad50} | {alt50} | {fmt(s.get('best_point_mean_rank_by_score'), 2)} | "
            f"{fmt(s.get('best_point_mean_rank_by_visible_score'), 2)} |"
        )

    lines.extend(["", "## Score-L2 Correlation", ""])
    lines.append("| Model | Sample group | n | corr(score,-L2) | corr(raw_has,-L2) | corr(visible,-L2) | corr(score*visible,-L2) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for result in results:
        for row in result["score_correlations"]:
            lines.append(
                f"| {row['model']} | {row['sample_group']} | {row['sample_count']} | "
                f"{fmt(row.get('corr_score_neg_l2'))} | {fmt(row.get('corr_raw_has_neg_l2'))} | "
                f"{fmt(row.get('corr_visible_score_neg_l2'))} | {fmt(row.get('corr_score_visible_product_neg_l2'))} |"
            )

    lines.extend(["", "## Taxonomy Counts", ""])
    all_labels = sorted({label for result in results for label in result["summary"].get("taxonomy_counts", {})})
    lines.append("| Label | " + " | ".join(result["model"] for result in results) + " |")
    lines.append("|---|" + "|".join("---:" for _ in results) + "|")
    for label in all_labels:
        lines.append("| " + label + " | " + " | ".join(str(result["summary"]["taxonomy_counts"].get(label, 0)) for result in results) + " |")

    if skipped:
        lines.extend(["", "## Skipped Inputs", ""])
        for item in skipped:
            lines.append(f"- `{item['model']}` skipped: `{item['path']}` not found.")

    lines.extend(["", "## Diagnostic Conclusion", "", f"**{conclusion[0]}**", "", conclusion[1], ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    skipped = []
    for name, path, required in DEFAULT_INPUTS:
        if name.startswith("RELCAL") and not args.include_relcal_reference:
            continue
        if not path.exists():
            item = {"model": name, "path": str(path)}
            if required:
                skipped.append(item)
            else:
                skipped.append(item)
            continue
        results.append(analyze_model(name, path, args.has_threshold, args.iou_threshold))

    if not results:
        raise RuntimeError("No input records found.")

    by_gt_rows = [row for result in results for row in result["by_gt_rows"]]
    taxonomy_rows = [row for result in results for row in result["taxonomy_rows"]]
    ranking_gap_rows = [row for result in results for row in result["ranking_gap_rows"]]
    correlation_rows = [row for result in results for row in result["score_correlations"]]

    write_csv(args.output_dir / "candidate_selection_by_gt.csv", by_gt_rows)
    write_csv(args.output_dir / "bad_case_taxonomy.csv", taxonomy_rows)
    write_csv(args.output_dir / "candidate_ranking_gap.csv", ranking_gap_rows)
    write_csv(args.output_dir / "score_l2_correlation.csv", correlation_rows)

    serializable = {
        "generated_from": [result["path"] for result in results],
        "has_picking_threshold": args.has_threshold,
        "iou_threshold": args.iou_threshold,
        "models": [
            {
                "model": result["model"],
                "path": result["path"],
                "summary": result["summary"],
                "score_correlations": result["score_correlations"],
            }
            for result in results
        ],
        "skipped": skipped,
    }
    conclusion = choose_conclusion(results)
    serializable["diagnostic_conclusion"] = {"label": conclusion[0], "reason": conclusion[1]}
    (args.output_dir / "candidate_selection_summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "candidate_selection_summary.md").write_text(
        build_markdown(results, conclusion, skipped),
        encoding="utf-8",
    )

    print(f"Wrote candidate diagnostics to {args.output_dir}")
    print(conclusion[0])


if __name__ == "__main__":
    main()
