# RT-DETRv4 当前清理总结

更新日期：2026-05-25

本轮清理将项目重新收束到 `v7_exp2` 主模型。旧实验输出仍保留在 `outputs/` 中，但旧训练配置和一次性工具不再保留为运行入口。

## 当前保留的运行入口

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
- `tools/make_grape_point_report.py`
- `tools/export_grape_point_predictions.py`
- `train.py`

## 当前主模型

- 模型定位：GPPoint-DETR / v7_exp2 current。
- 结构要点：`query_box_top + Top Local ROI`。
- 点任务形式：per-grape `has_picking + picking_offset`。
- 参考权重：`outputs/00_reference_models/gppoint_detr_v7_exp2_current/best_composite.pth`。

## 已清理方向

- 删除旧配置链：v2、v6、v7_exp1、旧消融配置、旧机制实验配置。
- 删除旧工具：wrapper 报告脚本、一次性审计脚本、旧论文资产生成脚本、旧 ROI/quality/teacher 分析脚本。
- 撤掉 HDPS/SimCC/VIS-DEDUP 对核心代码的运行支持。
- 保留训练可靠性修复：`--use-amp` 不再覆盖 YAML 默认值，named checkpoint 写入真实 epoch。

## 后续约定

下一阶段 encoder 模块实验必须：

- 以当前主配置为基准。
- 单独创建清晰命名的新配置。
- 不复用旧 decoder/HDPS 方向代码。
- 用同一套报告工具输出 AP、F1、pair_count、L2、|dx|、|dy|。
