# 论文实验资产索引

生成日期：2026-05-16

本索引基于当前 `outputs/` 和 `reports/` 扫描结果整理，用于论文写作、答辩说明和实验复现。本文档只记录资产位置与已有指标，不代表新增实验结论。

## 指标口径

- 主表指标来自各实验 `report/summary.json` 中 `primary_checkpoint_split_summary.test`。
- AP/AP50 为 grape bbox detection 指标。
- F1 为 has_picking 判别指标。
- pair_count、mean L2、|dy| 为 picking point 已匹配可见点上的定位指标。
- 除特别说明外，主表使用各实验 summary 中默认 test 评估口径；validation-selected 阈值结果见 `reports/valid_tuned_threshold_test_eval_zh.md`。
- `tight_toproi` 仅保留负结果摘要，当前未发现完整 checkpoint / prediction 资产。

## 正式实验资产总表

| 实验 | 类型 | 配置文件 | 输出目录 | checkpoint | summary / report | predictions | test 关键指标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_replay | 正式基线 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml`；summary 内记录为历史 `rtv4_hgnetv2_s_grape_point_v2.yml` | `outputs/grape_point_v6_baseline_replay` | `outputs/grape_point_v6_baseline_replay/checkpoints/best_composite.pth`；原始归档副本仍保留在 `_archive_cleanup_20260516/outputs/grape_point_v2_main/checkpoints/best_composite.pth` | `outputs/grape_point_v6_baseline_replay/report/summary.json`；`outputs/grape_point_v6_baseline_replay/report/comparison_report_zh.md`；`outputs/grape_point_v6_baseline_replay/report/results.csv` | `outputs/grape_point_v6_baseline_replay/predictions/test_predictions.json`；`outputs/grape_point_v6_baseline_replay/predictions/valid_predictions.json` | AP 0.6342；F1 0.7209；pair 155；L2 29.27；\|dy\| 22.53 |
| GPPoint-DETR / v7_exp2 current | 主模型 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` | `outputs/grape_point_gppoint_detr_main` | `outputs/grape_point_gppoint_detr_main/best_composite.pth` | `outputs/grape_point_gppoint_detr_main/report/summary.json`；`outputs/grape_point_gppoint_detr_main/report/comparison_report_zh.md`；`outputs/grape_point_gppoint_detr_main/report/results.csv` | `outputs/grape_point_gppoint_detr_main/predictions/test_predictions.json`；`outputs/grape_point_gppoint_detr_main/predictions/valid_predictions.json` | AP 0.6424；F1 0.7661；pair 190；L2 24.87；\|dy\| 16.89 |
| small_weight | 困难场景补强实验 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_small_weight.yml` | `outputs/grape_point_gppoint_detr_small_weight` | `outputs/grape_point_gppoint_detr_small_weight/best_composite.pth` | `outputs/grape_point_gppoint_detr_small_weight/report/summary.json`；`outputs/grape_point_gppoint_detr_small_weight/report/comparison_report_zh.md`；`outputs/grape_point_gppoint_detr_small_weight/report/results.csv` | `outputs/grape_point_gppoint_detr_small_weight/predictions/test_predictions.json`；`outputs/grape_point_gppoint_detr_small_weight/predictions/valid_predictions.json` | AP 0.6291；F1 0.7762；pair 189；L2 23.35；\|dy\| 17.25 |
| v7_exp2_taller_toproi | ROI 消融 / 机制实验 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_taller_toproi.yml` | `outputs/grape_point_v7_exp2_taller_toproi` | `outputs/grape_point_v7_exp2_taller_toproi/best_composite.pth` | `outputs/grape_point_v7_exp2_taller_toproi/report/summary.json`；`outputs/grape_point_v7_exp2_taller_toproi/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_taller_toproi/report/results.csv`；`outputs/grape_point_v7_exp2_taller_toproi/report/scene_slice_table.csv` | `outputs/grape_point_v7_exp2_taller_toproi/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_taller_toproi/predictions/valid_predictions.json` | AP 0.6414；F1 0.7415；pair 175；L2 21.93；\|dy\| 15.38 |
| v7_exp2_dn_teacher_roi | Teacher-guided ROI 机制实验 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_dn_teacher_roi.yml` | `outputs/grape_point_v7_exp2_dn_teacher_roi` | `outputs/grape_point_v7_exp2_dn_teacher_roi/best_composite.pth` | `outputs/grape_point_v7_exp2_dn_teacher_roi/report/summary.json`；`outputs/grape_point_v7_exp2_dn_teacher_roi/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_dn_teacher_roi/report/results.csv` | `outputs/grape_point_v7_exp2_dn_teacher_roi/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_dn_teacher_roi/predictions/valid_predictions.json` | AP 0.6289；F1 0.7574；pair 178；L2 24.27；\|dy\| 15.75 |
| v7_exp2_dn_teacher_roi_light_loss | loss 权重消融 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_dn_teacher_roi_light_loss.yml` | `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss` | `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss/best_composite.pth` | `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss/report/summary.json`；`outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss/report/results.csv` | `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss/predictions/valid_predictions.json` | AP 0.6308；F1 0.7773；pair 192；L2 24.05；\|dy\| 16.03 |
| v7_exp2_decoupled_roi | ROI 分支解耦消融 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_decoupled_roi.yml` | `outputs/grape_point_v7_exp2_decoupled_roi` | `outputs/grape_point_v7_exp2_decoupled_roi/best_composite.pth` | `outputs/grape_point_v7_exp2_decoupled_roi/report/summary.json`；`outputs/grape_point_v7_exp2_decoupled_roi/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_decoupled_roi/report/results.csv`；`outputs/grape_point_v7_exp2_decoupled_roi/report/decoupled_roi_final_report_zh.md` | `outputs/grape_point_v7_exp2_decoupled_roi/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_decoupled_roi/predictions/valid_predictions.json` | AP 0.6203；F1 0.7572；pair 184；L2 25.09；\|dy\| 17.25 |
| v7_exp2_point_quality | point quality 机制实验 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_point_quality.yml` | `outputs/grape_point_v7_exp2_point_quality` | `outputs/grape_point_v7_exp2_point_quality/best_composite.pth` | `outputs/grape_point_v7_exp2_point_quality/report/summary.json`；`outputs/grape_point_v7_exp2_point_quality/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_point_quality/report/results.csv`；`reports/point_quality_eval_zh.md` | `outputs/grape_point_v7_exp2_point_quality/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_point_quality/predictions/valid_predictions.json` | AP 0.6271；F1 0.7389；pair 167；L2 25.80；\|dy\| 16.68 |
| v7_exp2_point_quality_sg | stop-gradient quality 消融 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_point_quality_sg.yml` | `outputs/grape_point_v7_exp2_point_quality_sg` | `outputs/grape_point_v7_exp2_point_quality_sg/best_composite.pth` | `outputs/grape_point_v7_exp2_point_quality_sg/report/summary.json`；`outputs/grape_point_v7_exp2_point_quality_sg/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_point_quality_sg/report/results.csv`；`reports/point_quality_sg_eval_zh.md`；`reports/point_quality_sg_diagnostic_report_zh.md` | `outputs/grape_point_v7_exp2_point_quality_sg/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_point_quality_sg/predictions/valid_predictions.json` | AP 0.6236；F1 0.7609；pair 175；L2 24.23；\|dy\| 16.04 |
| v7_exp2_median_anchor | reference anchor 消融 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_median_anchor.yml` | `outputs/grape_point_v7_exp2_median_anchor` | `outputs/grape_point_v7_exp2_median_anchor/best_composite.pth` | `outputs/grape_point_v7_exp2_median_anchor/report/summary.json`；`outputs/grape_point_v7_exp2_median_anchor/report/comparison_report_zh.md`；`outputs/grape_point_v7_exp2_median_anchor/report/results.csv` | `outputs/grape_point_v7_exp2_median_anchor/predictions/test_predictions.json`；`outputs/grape_point_v7_exp2_median_anchor/predictions/valid_predictions.json` | AP 0.6241；F1 0.7715；pair 184；L2 26.76；\|dy\| 17.57 |
| tight_toproi | 负结果 | 未发现独立配置文件 | 当前活跃 `outputs/` 未发现；负结果摘要位于 `_archive_cleanup_20260516/outputs/history_suites/negative_variants/v7_exp2_tight_toproi` | 未发现 checkpoint | `_archive_cleanup_20260516/outputs/history_suites/negative_variants/v7_exp2_tight_toproi/summary_zh.md` | 未发现 prediction JSON | AP 0.6330；F1 0.7592；pair 175；L2 26.66；\|dy\| 17.46 |

## 多 seed 正式资产位置

以下目录不移动、不删除，作为论文中多 seed 稳定性分析的正式资产。

### GPPoint-DETR current / v7_exp2（多 seed 正式资产）

| run | 输出目录 | checkpoint | summary | predictions | test 指标 |
| --- | --- | --- | --- | --- | --- |
| main | `outputs/grape_point_gppoint_detr_main` | `best_composite.pth` | `report/summary.json` | `predictions/test_predictions.json`；`predictions/valid_predictions.json` | AP 0.6424；F1 0.7661；pair 190；L2 24.87；\|dy\| 16.89 |
| repro1 | `outputs/grape_point_gppoint_detr_main_repro1` | `best_composite.pth` | `report/summary.json` | `predictions/test_predictions.json`；`predictions/valid_predictions.json` | AP 0.6351；F1 0.7391；pair 170；L2 24.95；\|dy\| 16.01 |
| seed2026 | `outputs/grape_point_gppoint_detr_main_seed2026` | `best_composite.pth` | `report/summary.json` | `predictions/test_predictions.json`；`predictions/valid_predictions.json` | AP 0.6224；F1 0.7601；pair 179；L2 23.39；\|dy\| 16.55 |

### DN Teacher ROI（多 seed 正式资产）

| run | 输出目录 | checkpoint | summary | predictions | test 指标 |
| --- | --- | --- | --- | --- | --- |
| main | `outputs/grape_point_v7_exp2_dn_teacher_roi` | `best_composite.pth` | `report/summary.json` | `predictions/test_predictions.json`；`predictions/valid_predictions.json` | AP 0.6289；F1 0.7574；pair 178；L2 24.27；\|dy\| 15.75 |
| repro1 | `outputs/grape_point_v7_exp2_dn_teacher_roi_repro1` | `best_composite.pth` | `report/summary.json` | `predictions/test_predictions.json`；`predictions/valid_predictions.json` | AP 0.6221；F1 0.7505；pair 179；L2 23.63；\|dy\| 17.07 |
| seed2026 | `outputs/grape_point_v7_exp2_dn_teacher_roi_seed2026` | `best_composite.pth` | `report/summary.json` | `predictions/test_predictions.json`；`predictions/valid_predictions.json` | AP 0.6272；F1 0.7435；pair 171；L2 22.17；\|dy\| 16.17 |

多 seed calibration 汇总资产：

- `reports/dn_teacher_roi_multiseed_calibration_zh.md`
- `reports/dn_teacher_roi_multiseed_calibration.csv`
- `reports/dn_teacher_roi_f1_threshold_curve.png`
- `reports/dn_teacher_roi_pr_curve.png`

其中 valid-selected 阈值 mean±std 摘要：

| group | AP | F1 | pair_count | mean L2 | \|dy\| |
| --- | --- | --- | --- | --- | --- |
| current@valid-thr | 0.6333±0.0101 | 0.7850±0.0053 | 202.67±2.08 | 26.55±1.01 | 17.85±0.41 |
| dn_teacher_roi@valid-thr | 0.6261±0.0036 | 0.7816±0.0041 | 201.00±4.58 | 24.10±0.90 | 16.76±0.95 |

## 全局诊断报告索引

| 主题 | 报告 | 表格 / 图 |
| --- | --- | --- |
| ROI hit rate | `reports/roi_hit_rate_analysis_zh.md` | `reports/roi_hit_rate.csv` |
| ROI sensitivity | `reports/roi_sensitivity_analysis_zh.md` | `reports/roi_sensitivity.csv` |
| has_picking 阈值扫描 | `reports/has_picking_threshold_sweep_zh.md` | `reports/has_picking_threshold_sweep.csv` |
| valid-selected threshold test 评估 | `reports/valid_tuned_threshold_test_eval_zh.md` | `reports/valid_tuned_threshold_test_eval.csv` |
| DN teacher ROI 多 seed calibration | `reports/dn_teacher_roi_multiseed_calibration_zh.md` | `reports/dn_teacher_roi_multiseed_calibration.csv`；`reports/dn_teacher_roi_f1_threshold_curve.png`；`reports/dn_teacher_roi_pr_curve.png` |
| DN teacher ROI light loss | `reports/dn_teacher_roi_light_loss_eval_zh.md` | `reports/dn_teacher_roi_light_loss_eval.csv` |
| point quality 原版 | `reports/point_quality_eval_zh.md` | `reports/point_quality_threshold_calibration.csv`；`reports/quality_correlation_report_zh.md`；`reports/quality_correlation.csv`；`reports/quality_usage_strategy_report_zh.md`；`reports/quality_usage_strategy.csv` |
| point quality stop-gradient | `reports/point_quality_sg_eval_zh.md`；`reports/point_quality_sg_diagnostic_report_zh.md` | `reports/point_quality_sg_threshold_calibration.csv`；`reports/point_quality_sg_correlation.csv`；`reports/point_quality_sg_usage_strategy.csv` |
| picking relative position / anchor 统计 | `reports/picking_relative_position_analysis_zh.md` | `reports/picking_relative_position_stats.csv`；`reports/picking_relative_anchor_candidates.csv`；`reports/picking_relative_position_samples.csv`；`reports/picking_relative_rel_x_hist.png`；`reports/picking_relative_rel_y_hist.png`；`reports/picking_relative_position_heatmap.png` |
| 项目风险审查 | `reports/project_risk_check_zh.md` | 无 |

## 写作建议

1. 主模型仍建议写作 `GPPoint-DETR / v7_exp2 current`，对应 `outputs/grape_point_gppoint_detr_main`。
2. `baseline_replay` 是正式基线；其 `best_composite.pth`、summary/report 和 valid/test prediction JSON 已集中到 `outputs/grape_point_v6_baseline_replay/`，归档目录中的历史 checkpoint 仅作为原始副本保留。
3. `small_weight` 只能写作困难场景补强实验，不应替代主模型。
4. `tight_toproi` 只有负结果摘要，适合用于说明 ROI 过窄不可行，不适合作为完整可复现实验。
5. `taller_toproi`、`decoupled_roi`、`dn_teacher_roi`、`point_quality`、`point_quality_sg`、`median_anchor` 均应作为机制诊断或消融，不要直接改写主模型结论。
6. 阈值校准相关结论应引用 `valid_tuned_threshold_test_eval_zh.md` 和 `dn_teacher_roi_multiseed_calibration_zh.md`，不要只用单次 0.5 阈值结果下结论。
