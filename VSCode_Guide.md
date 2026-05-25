# GPPoint-DETR 当前使用指南

当前主模型已经收敛到 `v7_exp2`，配置入口已合并为一个自包含文件：

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`

旧实验配置、旧 wrapper 工具和一次性分析脚本已经清理。后续实验默认从当前主模型出发，重点转向 encoder 端结构改进。

## 主入口

- `train.py`：训练和测试入口。
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`：当前唯一主配置。
- `tools/make_grape_point_report.py`：统一报告生成工具。
- `tools/export_grape_point_predictions.py`：valid/test 预测导出工具。

## 当前主模型定义

任务不是独立检测 picking 小框，而是：

- `grape`：葡萄串 bbox 检测。
- `has_picking`：每个葡萄串是否存在可采摘点。
- `picking_offset`：相对葡萄串顶部参考点的二维采摘点偏移。

当前主模型结构：

- 基础模型：RT-DETRv4 / RTv4。
- 主线改动：`query_box_top + Top Local ROI`。
- 配置名：`rtv4_hgnetv2_s_grape_point_main.yml`。
- 论文表述：GPPoint-DETR / v7_exp2 current。

## 常用命令

训练当前主模型：

```powershell
Set-Location "D:\Projects\RT-DETR\RT-DETRv4"
.\.venv\Scripts\python.exe train.py --config configs\rtv4\rtv4_hgnetv2_s_grape_point_main.yml
```

测试旧参考 checkpoint：

```powershell
Set-Location "D:\Projects\RT-DETR\RT-DETRv4"
.\.venv\Scripts\python.exe train.py -c configs\rtv4\rtv4_hgnetv2_s_grape_point_main.yml --test-only --resume outputs\00_reference_models\gppoint_detr_v7_exp2_current\best_composite.pth --output-dir outputs\03_global_analysis\eval_v7_exp2_current
```

生成报告：

```powershell
Set-Location "D:\Projects\RT-DETR\RT-DETRv4"
.\.venv\Scripts\python.exe tools\make_grape_point_report.py --config configs\rtv4\rtv4_hgnetv2_s_grape_point_main.yml --run-dir outputs\00_reference_models\gppoint_detr_v7_exp2_current --report-dir outputs\03_global_analysis\v7_exp2_current_report
```

导出 valid/test 预测：

```powershell
Set-Location "D:\Projects\RT-DETR\RT-DETRv4"
.\.venv\Scripts\python.exe tools\export_grape_point_predictions.py --config configs\rtv4\rtv4_hgnetv2_s_grape_point_main.yml --checkpoint outputs\00_reference_models\gppoint_detr_v7_exp2_current\best_composite.pth --split valid --output outputs\03_global_analysis\v7_exp2_current_valid_predictions.json
.\.venv\Scripts\python.exe tools\export_grape_point_predictions.py --config configs\rtv4\rtv4_hgnetv2_s_grape_point_main.yml --checkpoint outputs\00_reference_models\gppoint_detr_v7_exp2_current\best_composite.pth --split test --output outputs\03_global_analysis\v7_exp2_current_test_predictions.json
```

## 后续实验规则

下一阶段 encoder 加模块时：

- 不直接改主配置。
- 新建一个清晰命名的实验配置，例如 `configs/rtv4/rtv4_hgnetv2_s_grape_point_encoder_xxx.yml`。
- 新配置以当前主配置为起点，只改 encoder 相关模块。
- 每次实验都必须保留同口径 valid/test 报告，不能只看单个指标。

核心对比指标：

- grape AP / AP50 / AR100
- has_picking precision / recall / F1
- point pair_count
- point mean L2
- mean |dx| / |dy|

## 不再使用的内容

以下内容已从运行入口中移除：

- v2/v6/v7_exp1 旧配置链。
- baseline/small/top-center wrapper 配置。
- DN teacher、point quality、median anchor、taller ROI 等旧机制实验配置。
- HDPS/VIS-DEDUP 配置和对应核心代码支持。
- 旧的一次性审计、可视化和论文资产生成脚本。

历史实验输出仍可在 `outputs/` 或已有报告中查看，但不再作为后续训练入口。
