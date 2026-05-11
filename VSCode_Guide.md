# 葡萄采摘点项目使用指南

这份文件只保留现在真正有用的信息，目标很明确：
- 怎么用这套代码继续训练或重做报告
- 怎么快速找到论文写作需要的结果
- 报告里哪些指标最重要，应该怎么解读

## 1. 先记住这几个主入口

如果你平时只打开少量文件，优先看这些：

- `train.py`
  训练总入口。

- `tools/make_grape_point_report.py`
  统一评估和报表内核。

- `tools/make_grape_point_main_report.py`
  当前主模型 `main` 的报告入口。

- `tools/make_grape_point_small_grape_report.py`
  `small_grape` 定向补强版的报告入口。

- `tools/make_grape_point_paper_assets.py`
  论文资产整理脚本。

- `tools/make_grape_point_sci_assets.py`
  SCI 写作资产整理脚本。

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
  当前主模型配置。

- `outputs/grape_point_v7_sci_ready/`
  当前写论文最该看的结果目录。

## 2. 当前主线到底是什么

当前任务定义不是“采摘点小框检测”，而是：
- `grape`：葡萄串 `bbox` 检测
- `picking`：按葡萄串实例建模的 `has_picking + point localization`

当前默认主模型是：
- `main`
- 论文命名：`GPPoint-DETR`
- 实际方法含义：`query_box + top-center + top local cue`

当前几个语义化配置的含义：
- `baseline_replay`
  正式基线。
- `top_center`
  过渡版，用来说明仅靠 `top-center` 为什么收益不完整。
- `main`
  当前主模型。
- `small_grape`
  困难场景定向补强版，不替代主模型。

## 3. 最常用命令

### 3.1 训练主模型

```powershell
.\.venv\Scripts\python.exe train.py --config configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml
```

### 3.2 训练正式 baseline

```powershell
.\.venv\Scripts\python.exe train.py --config configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml
```

### 3.3 训练 `small_grape` 补强版

```powershell
.\.venv\Scripts\python.exe train.py --config configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml
```

### 3.4 重做主模型报告

```powershell
.\.venv\Scripts\python.exe tools/make_grape_point_main_report.py --run-dir outputs/grape_point_gppoint_detr_main
```

### 3.5 重做 `small_grape` 报告

```powershell
.\.venv\Scripts\python.exe tools/make_grape_point_small_grape_report.py --run-dir outputs/grape_point_gppoint_detr_small_weight
```

### 3.6 重新整理论文资产

```powershell
.\.venv\Scripts\python.exe tools/make_grape_point_paper_assets.py
.\.venv\Scripts\python.exe tools/make_grape_point_sci_assets.py
```

## 4. 训练和代码主要看哪里

### 4.1 训练入口

- `train.py`
  总训练入口。以后要继续跑实验，默认从这里进。

### 4.2 当前最相关的模型代码

- `engine/rtv4/dfine_decoder.py`
  当前点分支的核心实现。`query_box`、`top local cue` 等都在这里。

- `engine/rtv4/point_utils.py`
  点坐标表达方式，尤其是 `top-center`。

- `engine/rtv4/rtv4_criterion.py`
  点损失、`small_grape weight`、`x/y` 坐标加权等。

如果后面写论文时要解释“为什么方法有效”，最值得回看的就是这 3 个文件。

### 4.3 配置文件怎么理解

现在日常只需要看 4 个语义化配置：

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_top_center.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml`

它们背后仍然会继承少量旧配置文件，这是正常的，不建议手动去砍配置链。

## 5. outputs 目录怎么看

现在 `outputs/` 已经压缩成论文还需要的最小集合。最重要的是这些目录：

- `outputs/grape_point_v6_baseline_replay`
  正式 baseline 结果。

- `outputs/grape_point_v7_exp1_query_box_top_center`
  `top_center` 过渡版结果。

- `outputs/grape_point_gppoint_detr_main`
  当前主模型 `GPPoint-DETR` 主结果。

- `outputs/grape_point_gppoint_detr_main_repro1`
- `outputs/grape_point_gppoint_detr_main_seed2026`
  主模型多 seed 复现实验。

- `outputs/grape_point_gppoint_detr_small_weight`
  `small_grape` 定向补强结果。

- `outputs/grape_point_v7_paper_ready`
  中文期刊版写作资产。

- `outputs/grape_point_v7_sci_ready`
  SCI 版写作资产。

- `outputs/history_suites`
  历史失败方向最小证据，只用于回顾结论，不再展开跑实验。

## 6. 报告文件怎么读

### 6.1 最优先看 `summary.json`

每个核心 run 最优先看：
- `outputs/.../report/summary.json`

这是单个实验的总汇总，最重要。

### 6.2 单次实验报告里最常看的文件

- `report/summary.json`
  主汇总。

- `report/comparison_report_zh.md`
  中文结论版摘要，适合快速回看。

- `report/results_overview.png`
  图形化总览，适合快速看整体趋势。

- `new_variant_summary.json`
  新变体专用汇总，只有变体对比目录里会有。

### 6.3 论文写作阶段最该看的结果文件

如果已经进入写论文阶段，优先看：

- `outputs/grape_point_v7_sci_ready/sci_ready_summary.json`
- `outputs/grape_point_v7_sci_ready/final_mean_std_table.csv`
- `outputs/grape_point_v7_sci_ready/stability_analysis_zh.md`
- `outputs/grape_point_v7_sci_ready/external_comparison_zh.md`
- `outputs/grape_point_v7_sci_ready/error_attribution_zh.md`
- `outputs/grape_point_v7_sci_ready/sci_claims_zh.md`
- `outputs/grape_point_v7_sci_ready/sci_contribution_draft_zh.md`
- `outputs/grape_point_v7_sci_ready/sci_storyline_zh.md`

## 7. 报告里重点指标是什么

这是现在最重要的部分。以后看报告不要平均用力，重点盯下面这些。

### 7.1 `grape_detection`

核心指标：
- `AP`

怎么理解：
- `AP` 反映葡萄串检测整体质量。
- 但对当前论文主线来说，它不是最稳定的主结论，因为主模型在不同 seed 上 `AP` 有一定敏感性。

结论：
- 要看，但不要把它写成“最稳定贡献点”。

### 7.2 `has_picking`

核心指标：
- `F1`

辅助指标：
- `precision`
- `recall`

怎么理解：
- `F1` 是当前主模型最稳定、最值得写进论文主表的分类收益之一。
- 如果 `precision` 和 `recall` 一高一低，要看 `F1` 是否总体更好，而不是只看单边。

### 7.3 `picking_point`

核心指标：
- `pair_count`
- `mean_l2_px`
- `mean_abs_dy_px`

辅助指标：
- `median_l2_px`
- `p90_l2_px`
- `mean_abs_dx_px`

怎么理解：
- `pair_count`
  代表真正进入点误差统计的有效配对数。太低说明链条没打通。

- `mean_l2_px`
  总体点误差主指标。当前最重要。

- `mean_abs_dy_px`
  当前课题的关键痛点指标。因为历史主矛盾一直是 `dy` 漂移。

- `median_l2_px`
  看典型样本误差水平。

- `p90_l2_px`
  看长尾坏样本是否很严重。

- `mean_abs_dx_px`
  当前不是第一优先级，但可以辅助判断误差是否明显偏向纵向。

### 7.4 `size_group_l2_px`

重点看：
- `small`
- `medium`
- `large`

怎么理解：
- 这是分尺度点误差。
- 当前最该盯的是 `small`，因为 `small grape` 是当前短板之一。
- 如果一个新版本只提升 `small`，但整体略波动，也可能仍然有论文价值。

### 7.5 场景切片

当前固定看的 3 组场景：
- `single / multi_adjacent`
- `light_occlusion / heavy_occlusion`
- `small / medium_large`

怎么理解：
- `multi_adjacent` 和 `heavy_occlusion`
  主要看复杂场景下链条是否更稳。

- `small`
  主要看困难小串是否改善。

对于 `main` 来说：
- 重点结论是整体链条更稳、`F1 / pair_count / mean L2 / |dy|` 更好。

对于 `small_grape` 来说：
- 重点结论是 `small` 和 `heavy occlusion` 更有改善，
- 但它不替代 `main`，只是补强证据。

## 8. 论文里最稳妥的主结论怎么写

当前最稳妥的写法是：

1. 主模型 `main` 的稳定收益主要体现在：
   - `has_picking F1`
   - `pair_count`
   - `mean L2`
   - `|dy|`

2. `AP` 可以写有提升趋势或单次更优，但不能写成最稳稳定结论。

3. `top_center` 的作用是先证明：
   - 单靠点表达改进可以压 `dy`
   - 但它不足以形成完整收益

4. `main` 的作用是：
   - 在保持点定位收益的同时，把整条检测-可见性-定位链条做稳

5. `small_grape` 的定位是：
   - 困难场景定向补强证据
   - 不是替换主模型

## 9. 你平时最常见的工作流

### 9.1 如果你要继续补实验

1. 选语义化配置
2. 用 `train.py` 训练
3. 用对应报告入口重做报告
4. 如果是论文阶段，再重做 `paper_assets` 和 `sci_assets`

### 9.2 如果你现在开始写论文

按这个顺序最省力：

1. 看本文件
2. 看 `outputs/grape_point_v7_sci_ready/sci_ready_summary.json`
3. 看 `final_mean_std_table.csv`
4. 看 `external_comparison_zh.md`
5. 看 `error_attribution_zh.md`
6. 再去写方法、实验和讨论

## 10. 最后一句话版

如果后面你只想最省事地推进论文：

- 跑训练看 `train.py`
- 重做报告看 `tools/make_grape_point_report.py`
- 当前主模型看 `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
- 写论文重点看 `outputs/grape_point_v7_sci_ready/`
- 解读指标优先看 `F1 / pair_count / mean L2 / |dy|`

这几项抓住，基本就不会走偏。
