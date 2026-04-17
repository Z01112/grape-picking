# RT-DETRv4 使用指南

这份文件只保留你现在真正需要的内容：怎么跑 baseline、结果看哪里、目录怎么理解、哪些地方你可以安全改。

## 1. 你平时只需要关心这些文件

- `scripts/run_baseline.ps1`
- `scripts/make_report.ps1`
- `scripts/problem_images.ps1`
- `configs/rtv4/rtv4_hgnetv2_s_grape_picking_baseline.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_picking_base.yml`
- `train.py`
- `tools/make_grape_picking_baseline_report.py`

如果你只是想稳定跑 baseline 并看结果，上面这些就够了。

## 2. 最常用的三条命令

训练：

```powershell
.\scripts\run_baseline.ps1
```

重做报告：

```powershell
.\scripts\make_report.ps1
```

筛问题图片：

```powershell
.\scripts\problem_images.ps1
```

如果 PowerShell 不让你运行脚本，先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 3. 现在脚本默认怎么找 run 目录

这三个脚本都支持两种最省事的方式：

1. 你直接传 `-OutputDir` 或 `-RunDir`
2. 你什么都不传，它会优先读取 `scripts/.last_baseline_run.txt` 里记住的最近一次 baseline 路径

另外，3 个脚本开头都留了非常短的可编辑默认值：

- `run_baseline.ps1` 里看 `$PinnedOutputDir`
- `make_report.ps1` 里看 `$PinnedRunDir`
- `problem_images.ps1` 里看 `$PinnedRunDir`

如果你想以后一直固定跑某个目录，比如 `outputs\baseline_20260406`，就直接改这几个变量，不用在终端里手打。

## 4. 输出目录现在怎么看

现在 baseline 跑完并生成报告后，`outputs/某次运行/` 顶层会尽量只保留最该看的内容：

- `summary.json`
- `results.csv`
- `training_curves.png`
- `results_overview.png`
- `test_class_metrics.png`
- `error_breakdown.png`
- `test_gt_vs_pred.jpg`
- `checkpoints/`
- `logs/`
- `report/`

这样 VSCode 资源管理器第一眼看到的就是总汇总和关键图，不会先被一长串 `pth` 淹没。

### 4.1 顶层文件看什么

- `summary.json`
  这次 run 的主汇总。优先看它。

- `results.csv`
  类似 YOLO 的逐 epoch 表格结果文件，适合用 Excel 直接打开看趋势。

- `training_curves.png`
  训练曲线图，偏完整技术视角。

- `results_overview.png`
  更像 YOLO `results.png` 的总览图。优先突出 `grape/picking` 曲线和整体趋势。

- `test_class_metrics.png`
  分类别指标图，重点看 `picking`。

- `error_breakdown.png`
  分类别错误图，重点看 `picking` 的漏检和背景误检。

- `test_gt_vs_pred.jpg`
  代表样例图，直观看预测效果。

### 4.2 `checkpoints/` 里是什么

- `best_stg1.pth`
- `best_stg2.pth`
- `last.pth`
- `checkpointXXXX.pth`

平时你最常用的是：

- `checkpoints/best_stg2.pth`
- `checkpoints/last.pth`

### 4.3 `logs/` 里是什么

- `logs/log.txt`
  训练逐 epoch 日志。以后每个 epoch 都会记录 overall 验证指标，新的训练还会记录 `grape/picking` 的分类别验证指标。

- `logs/summary/`
  TensorBoard 事件文件。

- `logs/eval/`、`logs/eval_valid/`、`logs/eval_test/`
  训练或独立评估过程中生成的中间评估产物。

### 4.4 `report/` 里是什么

`report/` 保留原始明细文件，方便后处理和二次分析。默认极简模式主要保留：

- `report/summary.json`
- `report/per_image_test_summary.json`
- `report/predictions_test.json`
- `report/training_curves.png`
- `report/test_class_metrics.png`
- `report/error_breakdown.png`
- `report/test_gt_vs_pred.jpg`

其中：

- `report/summary.json` 是顶层 `summary.json` 的原始来源
- `report/per_image_test_summary.json` 用来查哪张图最难
- `report/predictions_test.json` 用来做误检分析和二次统计

## 5. 配置为什么分两层

- `*_base.yml` 放稳定公共项：类别数、数据路径、优化器、损失、基础 dataloader 设置
- `*_baseline.yml` 放这次 baseline 常改项：`output_dir`、epoch、增强、batch size、打印频率

这样以后你只改 baseline，不容易把底层公共配置改坏。

## 6. 最重要的报告怎么读

### 6.1 `summary.json` 是主入口

现在尽量把重点都汇总在 `summary.json`：

- 最佳 overall 验证轮次
- 最终验证结果
- test overall 指标
- `grape / picking` 分类别 test 指标
- `grape / picking` 的 precision / recall / F1 / TP / FP / FN
- 已知的类别最佳验证信息

### 6.2 目前最该盯住哪些 `picking` 指标

- `AP`
- `AP50`
- `AR100`
- `precision`
- `recall`
- `F1`
- `background false positives`

### 6.3 overall 最佳轮次和类别最佳轮次要分开理解

- overall 最佳轮次看 `best_validation`
- 分类别最佳轮次看 `best_validation_per_class` 或 `class_summary` 里的对应字段

注意：

- 旧训练日志只记录了 overall 验证指标，所以旧 run 不一定有“精确的分类别最佳 epoch”
- 新训练开始后，`log.txt` 会逐 epoch 记录 `grape/picking` 的 valid AP、AP50、AR100，这样才能得到真正精确的 `grape_best_epoch` 和 `picking_best_epoch`

### 6.4 为什么 `test` 不一定用最后一轮

默认报告会优先拿 `checkpoints/best_stg2.pth` 做 test 评估，因为它是当前保存逻辑下的 overall 最优 checkpoint。
如果你以后想专门比较 `last.pth`，可以单独指定 checkpoint 重做报告。

## 7. 重要目录怎么理解

- `configs/`
  配置文件目录。你当前主要看 `configs/rtv4/`

- `dataset/`
  数据集目录，通常是 `train / valid / test`

- `engine/`
  官方训练框架核心代码，模型、训练、数据、评估都在这里

- `pretrain/`
  预训练权重目录

- `scripts/`
  你直接在 VSCode 终端里运行的入口脚本

- `tools/`
  `scripts` 背后的 Python 程序

- `outputs/`
  训练结果和报告归档目录

## 8. 重要文件说明

### 训练与配置

- `train.py`
  整个工程的训练/测试总入口

- `configs/rtv4/rtv4_hgnetv2_s_grape_picking_base.yml`
  当前葡萄任务的基础公共配置

- `configs/rtv4/rtv4_hgnetv2_s_grape_picking_baseline.yml`
  当前 baseline 实际运行配置

### 运行脚本

- `scripts/run_baseline.ps1`
  一键完成标注准备、训练、valid/test eval，并在结束后自动生成报告

- `scripts/make_report.ps1`
  为某次 run 重做报告，并自动整理目录结构

- `scripts/problem_images.ps1`
  从报告中筛出干扰大、误检多、漏检多的图

### 报告与分析程序

- `tools/run_grape_picking_baseline.py`
  baseline 训练流程封装，最终会调用 `train.py`

- `tools/make_grape_picking_baseline_report.py`
  报告主程序。负责汇总 `summary`、导出关键图、分析 test 预测

- `tools/report_problem_images.py`
  问题图片专题分析程序

- `tools/mine_false_positives.py`
  专门分析误检

- `tools/diagnose_picking_label_consistency.py`
  专门分析 `picking` 标签一致性

## 9. 以后最常见的工作流

1. 跑训练：`run_baseline.ps1`
2. 打开 `outputs/某次运行/summary.json`
3. 看 `training_curves.png`、`test_class_metrics.png`、`error_breakdown.png`
4. 如果要查难图，再看 `report/per_image_test_summary.json`
5. 如果要筛典型问题图，再跑 `problem_images.ps1`

对你当前课题来说，优先级最高的是：

- `summary.json`
- `test_class_metrics.png`
- `error_breakdown.png`
- `report/per_image_test_summary.json`

.\scripts\run_point_v5.ps1 -Experiment exp2 -ResumeLast