# 论文写作工作区

这份索引只保留当前论文主线真正需要打开的入口，避免后续在母目录里来回翻旧实验。

## 当前主线代码
- 训练入口：[train.py](/D:/Projects/RT-DETR/RT-DETRv4/train.py)
- 统一评估与报表内核：[make_grape_point_v2_report.py](/D:/Projects/RT-DETR/RT-DETRv4/tools/make_grape_point_v2_report.py)
- 新变体对比资产：[make_grape_point_new_variant_assets.py](/D:/Projects/RT-DETR/RT-DETRv4/tools/make_grape_point_new_variant_assets.py)
- 论文资产整理：[make_grape_point_v7_paper_assets.py](/D:/Projects/RT-DETR/RT-DETRv4/tools/make_grape_point_v7_paper_assets.py)
- SCI 资产整理：[make_grape_point_v7_sci_assets.py](/D:/Projects/RT-DETR/RT-DETRv4/tools/make_grape_point_v7_sci_assets.py)

## 当前主线配置
- 正式 baseline：[rtv4_hgnetv2_s_grape_point_v2.yml](/D:/Projects/RT-DETR/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_grape_point_v2.yml)
- `v7_exp1`：[rtv4_hgnetv2_s_grape_point_v7_exp1.yml](/D:/Projects/RT-DETR/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp1.yml)
- 当前主模型 `v7_exp2`：[rtv4_hgnetv2_s_grape_point_v7_exp2.yml](/D:/Projects/RT-DETR/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml)
- 定向补强 `small_weight`：[rtv4_hgnetv2_s_grape_point_v7_exp2_small_weight.yml](/D:/Projects/RT-DETR/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_small_weight.yml)

## 核心结果
- 正式 baseline 报告：[summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v6_baseline_replay/report/summary.json)
- `v7_exp1` 报告：[summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_exp1_query_box_top_center/report/summary.json)
- `v7_exp2` 报告：[summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_exp2_query_box_top_center_toproi/report/summary.json)
- `small_weight` 报告：[summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_exp2_small_weight/report/summary.json)
- `small_weight` 新变体对比：[new_variant_summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_exp2_small_weight/new_variant_summary.json)
- `tight_toproi` 负结果保留：[new_variant_summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_exp2_tight_toproi/new_variant_summary.json)

## 论文资产
- 中文期刊版资产：[grape_point_v7_paper_ready](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_paper_ready)
- SCI 版资产：[grape_point_v7_sci_ready](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready)
- SCI 结论稿：[sci_claims_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/sci_claims_zh.md)
- SCI 贡献草稿：[sci_contribution_draft_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/sci_contribution_draft_zh.md)
- SCI 叙事线：[sci_storyline_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/sci_storyline_zh.md)

## 失败方向的最小证据
- 历史结论总索引：[history_suites/README_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/history_suites/README_zh.md)
- 历史演化表：[history_evolution_table.csv](/D:/Projects/RT-DETR/RT-DETRv4/outputs/history_suites/history_evolution_table.csv)
- `v6_exp1` 结论：[summary_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/history_suites/negative_variants/v6_exp1_instance_binding/summary_zh.md)
- `v6_exp2` 结论：[summary_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/history_suites/negative_variants/v6_exp2_instance_binding_relative/summary_zh.md)

## 当前最值得优先打开的文件
1. [sci_ready_summary.json](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/sci_ready_summary.json)
2. [final_mean_std_table.csv](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/final_mean_std_table.csv)
3. [external_comparison_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/external_comparison_zh.md)
4. [error_attribution_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_sci_ready/error_attribution_zh.md)
5. [new_variant_comparison_report_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/grape_point_v7_exp2_small_weight/new_variant_comparison_report_zh.md)

## 当前保留原则
- 主模型仍以 `v7_exp2` 为准。
- `small_weight` 保留为困难场景定向补强证据。
- `tight_toproi` 只保留负结果结论：[summary_zh.md](/D:/Projects/RT-DETR/RT-DETRv4/outputs/history_suites/negative_variants/v7_exp2_tight_toproi/summary_zh.md)。
- 旧版批处理脚本、临时脚本和中间 checkpoint 已删除。
