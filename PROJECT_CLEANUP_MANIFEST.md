# RT-DETRv4 当前清理清单

更新日期：2026-05-25

本次清理后的目标状态是：以 `v7_exp2` 为当前唯一主模型入口，删除旧实验配置和一次性工具，为下一阶段 encoder 端结构改进腾出清晰目录。

## 保留内容

- `train.py`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
- `engine/`
- `tools/make_grape_point_report.py`
- `tools/export_grape_point_predictions.py`
- `dataset/`
- `pretrain/`
- `outputs/`
- `docs/`
- `reports/`

## 已废弃内容

以下内容不再作为可运行入口保留：

- `configs/rtv4` 下 v2、v6、v7_exp1、旧消融、旧机制实验和 HDPS/VIS-DEDUP 配置。
- `tools` 下旧 wrapper、旧论文资产整理、旧可视化审计、旧阈值/ROI/quality/teacher 分析脚本。
- 核心代码中的 HDPS、SimCC、VIS-DEDUP 专用支持。

## 当前主模型

- 配置：`configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
- 方法定位：`v7_exp2 = query_box_top + Top Local ROI`
- 参考 checkpoint：`outputs/00_reference_models/gppoint_detr_v7_exp2_current/best_composite.pth`

## 验证要求

清理后必须通过：

```powershell
.\.venv\Scripts\python.exe -m py_compile train.py engine\rtv4\dfine_decoder.py engine\rtv4\postprocessor.py engine\rtv4\rtv4.py engine\rtv4\rtv4_criterion.py engine\solver\det_solver.py tools\make_grape_point_report.py tools\export_grape_point_predictions.py
```

```powershell
.\.venv\Scripts\python.exe -c "from engine.core import YAMLConfig; cfg=YAMLConfig('configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml'); print(cfg.model.__class__.__name__)"
```

后续新增实验必须从当前主配置复制或继承，实验完成后再决定是否保留。
