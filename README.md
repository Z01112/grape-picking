# 基于 RT-DETRv4 的葡萄串识别与采摘点定位

这个仓库是我围绕葡萄采摘场景做的模型研究代码与论文资产仓库，而不是 RT-DETRv4 官方仓库镜像。

当前主线任务定义为：
- `grape`：葡萄串 `bbox` 检测
- `picking`：按葡萄串实例建模的 `has_picking + point localization`

也就是说，采摘点不再作为独立小框检测目标，而是由每个 `grape query` 直接判断“是否存在可见采摘点”并回归点位坐标。

## 仓库定位

这个仓库主要服务 3 件事：
- 保存当前论文主线代码与配置
- 保存实验方法、评估脚本和论文写作入口
- 保存论文写作过程中需要直接引用的结论组织方式

当前最重要的工作区索引在：
- [paper_workspace/README_zh.md](./paper_workspace/README_zh.md)

## 当前主模型

当前主模型为 `v7_exp2`，核心组合是：
- `query_box`
- `top-center`
- `top local cue`

相对正式基线 `baseline_replay`，`v7_exp2` 的稳定收益主要体现在：
- `has_picking F1`
- `pair_count`
- `mean L2`
- `|dy|`

其中 `AP` 仍然存在一定 seed sensitivity，因此论文主结论不会把它写成最强稳定结论。

## 关键实验版本

论文真正会反复引用的版本只有这些：
- `baseline_replay`
- `v7_exp1`
- `v7_exp2`
- `small_weight`：作为 `small grape / heavy occlusion` 的定向补强证据

更早期的 `v4 / v5 / v6` 探索版本已经压缩成历史结论，不再保留为完整运行目录。

## 仓库结构

```text
configs/         当前主线实验配置
engine/          模型与损失实现
tools/           统一评估、报表、论文资产脚本
paper_workspace/ 论文写作索引
train.py         训练入口
```

## 数据与结果资产说明

当前 GitHub 仓库默认只同步代码与写作入口，不同步本地数据集和 `outputs/` 结果目录。

暂不上传的内容包括：
- `dataset/`
- `outputs/`
- 大体积权重文件 `*.pth`
- TensorBoard event 文件
- 训练日志和临时缓存
- 大量 debug 可视化中间件

本地数据与结果资产仍然保留在工作区中，用于训练、评估和写论文。

## 本地数据集说明

当前数据划分如下：
- train: `1271` images
- valid: `365` images
- test: `184` images

标注形式围绕葡萄串实例展开，支持：
- grape `bbox`
- per-grape `has_picking`
- picking point

## 快速开始

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train.py --config configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml
```

## 常用入口

- 统一评估主脚本：[tools/make_grape_point_v2_report.py](./tools/make_grape_point_v2_report.py)
- 论文资产脚本：[tools/make_grape_point_v7_paper_assets.py](./tools/make_grape_point_v7_paper_assets.py)
- SCI 资产脚本：[tools/make_grape_point_v7_sci_assets.py](./tools/make_grape_point_v7_sci_assets.py)
- 新变体对比资产：[tools/make_grape_point_new_variant_assets.py](./tools/make_grape_point_new_variant_assets.py)

## 论文写作入口

如果你的目标是直接进入论文写作，优先打开：
- [paper_workspace/README_zh.md](./paper_workspace/README_zh.md)
- 本地 `outputs/grape_point_v7_sci_ready/`
- 本地 `outputs/grape_point_v7_paper_ready/`

## 说明

这个仓库已经从“官方 RT-DETRv4 代码树”整理成“葡萄采摘点论文工作区”，所以 GitHub 首页内容现在以本项目为中心，而不是官方仓库介绍。
