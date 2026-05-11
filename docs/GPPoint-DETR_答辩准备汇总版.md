# GPPoint-DETR 答辩准备汇总版

本文档由以下三份材料汇总整理而成：

- `D:\Chorme\答辩准备.docx`
- `docs/GPPoint-DETR_模型运行与代码改动说明.md`
- `docs/GPPoint-DETR_论文问答与技术定位.md`

用途：用于论文答辩、论文修改、方法章节核对和实验结果解释。本文档不是重新发明叙事，而是把已有材料中的问题、答案、代码依据和论文表述边界统一到一份可复习文档中。

论文题目：

> 《基于RT-DETRv4的葡萄串识别与采摘点定位方法研究》

当前主模型名称：

> GPPoint-DETR

历史实验编号：

> `v7_exp2`

---

## 1. 答辩总口径

本课题的核心不是单纯提高葡萄串检测框 AP，而是针对葡萄采摘任务中“检测到葡萄串之后仍缺少可用采摘点”的问题，将视觉输出从普通目标检测结果推进到面向采摘执行的实例级输出结果。

可直接背诵的总述：

> 本文以 RT-DETRv4 为基础提出 GPPoint-DETR。在保持单阶段检测主干基本不变的前提下，模型为每个葡萄串实例增加 `has_picking` 判别分支和 `point_offset` 点定位分支，并结合 `top-center` 几何先验、`query_box_top` 实例绑定和 Top Local ROI 上部局部特征，使模型能够同时输出葡萄串边界框、采摘点可见性以及图像坐标系下的采摘点位置。实验结果表明，该方法主要改善了 `has_picking F1`、有效配对数量、点定位误差和纵向误差控制，而不是单纯依赖 grape AP 的提升。

答辩时必须坚持的边界：

- 当前模型输出的是二维图像坐标系下的采摘点 `(u, v)`，不是机械臂可直接执行的三维坐标。
- GPPoint-DETR 的稳定贡献主要体现在采摘点链条指标上，不应夸大为所有指标全面显著提升。
- `small_weight` 是困难场景补强实验，不是主模型。
- `tight_toproi` 是负结果实验，用于说明过度收紧局部 ROI 不可行。
- 本文不是提出全新 backbone，而是在 RT-DETRv4 单阶段框架上做实例级采摘点输出扩展。

---

## 2. 论文主要创新点

本文创新点建议稳妥写成三点。

### 2.1 任务定义重构

早期路线为：

```text
grape bbox detection + picking bbox detection
```

当前主线为：

```text
grape bbox detection + per-grape has_picking + per-grape point localization
```

也就是说，采摘点不再被当成与葡萄串平级的独立小目标框，而是被建模为葡萄串实例的可见性属性和点位置输出。

答辩说法：

> 本文的第一个改进是任务建模方式的改变。采摘点并不是普通独立目标，而是依附于具体葡萄串实例的操作点。因此本文把 picking bbox 检测重构为每个 grape 实例上的 `has_picking` 判别和点定位。

### 2.2 GPPoint-DETR 实例级输出扩展

原始 RT-DETRv4 输出：

```text
labels / scores / boxes
```

GPPoint-DETR 输出：

```text
labels / scores / boxes / has_picking / picking_point
```

新增分支：

- `has_picking`：判断当前 grape 实例是否有可见采摘点。
- `point_offset`：预测采摘点相对 grape 框参考点的归一化偏移。

答辩说法：

> GPPoint-DETR 保留 RT-DETRv4 的单阶段检测主干，在每个 decoder query 上增加 `has_picking` 和 `point_offset` 两个实例级输出分支，使一个 grape query 可以同时承担检测、可见性判断和采摘点定位。

### 2.3 上部空间先验与局部视觉 cue

GPPoint-DETR 引入三个关键设计：

- `top-center`：把点回归锚点从 bbox 中心移到葡萄串上部。
- `query_box_top`：把采摘点预测绑定到当前 grape query 和其预测框上部。
- Top Local ROI：从 grape 框上部区域提取局部视觉特征，帮助点分支关注果梗附近区域。

答辩说法：

> 这三个设计共同服务于同一个目标：让采摘点预测既属于当前葡萄串实例，又重点关注葡萄串上部更可能出现果梗和采摘点的位置。

---

## 3. 为什么不用原始 RT-DETRv4 直接检测 picking

原始 RT-DETRv4 是通用目标检测模型，适合输出类别和边界框。如果直接把 `picking` 当作第二类目标检测，就会变成：

```text
grape object + picking object
```

这个做法的问题是，它只能回答“图中哪里可能有 picking 小框”，但不能自然回答：

```text
这一个 picking 属于哪一串葡萄？
这串葡萄有没有可见采摘点？
同一串葡萄是否出现多个 picking 响应？
没有可见采摘点的葡萄串如何表达？
```

答辩标准回答：

> 不直接检测 picking，是因为采摘点不是独立语义目标，而是具体葡萄串实例上的局部操作点。直接用原始 RT-DETRv4 检测 picking 小框会带来小目标不稳定、重复框和实例归属不清的问题。因此本文采用每个 grape 实例输出 `has_picking + point_offset` 的方式，把采摘点变成 grape 实例的属性和点坐标。

---

## 4. 为什么 picking bbox 不合适

`picking bbox` 不适合当前任务的原因如下。

| 问题 | 具体表现 | 对论文的解释 |
|---|---|---|
| 小目标不稳定 | 采摘点区域很小，bbox 容易漏检、重复框或偏移 | picking 更适合用点表达，而不是小框 |
| 实例归属不清 | 一个 picking 框需要额外规则判断属于哪串葡萄 | 后处理关联容易自圆其说 |
| 多串相邻困难 | 邻串靠近时 picking 框容易跨实例误关联 | 需要实例级绑定 |
| 输出不贴近采摘执行 | 机械臂最终需要操作点，不是第二个小框 | 点定位更符合任务目标 |

答辩标准回答：

> picking bbox 的问题不是模型完全不能学，而是任务表达不够合理。采摘点边界本身并不稳定，且 picking 必须依附于某个 grape 实例。把 picking 当独立目标会把“每串葡萄的采摘决策”变成“场景中的小目标检测”，容易出现重复检测和错关联。因此本文改用 per-grape 点定位。

---

## 5. 为什么要用 RT-DETRv4 做主模型

选择 RT-DETRv4 的原因不是因为它天然能解决采摘点，而是因为它适合作为实例级输出扩展的基础。

| 原因 | 说明 |
|---|---|
| 单阶段端到端 | 不需要额外 two-stage ROI detector，推理链路更简洁 |
| query-based 输出 | 每个 query 可以自然承载一个 grape 实例及其附属属性 |
| 检测能力较强 | 适合作为葡萄串 bbox 检测基础 |
| 易扩展输出头 | 可以在 decoder query 上增加 `has_picking` 和 `point_offset` |
| 适合实例级建模 | query 与实例天然对应，有利于 per-grape 输出 |

答辩标准回答：

> 本文使用 RT-DETRv4，是因为它提供了较强的单阶段检测框架和 query-based 实例表示。每个 decoder query 可以对应一个葡萄串实例，因此很适合在其上扩展可见性判断和采摘点偏移回归，而不需要把任务改成复杂的两阶段系统。

---

## 6. GPPoint-DETR 相比原始 RT-DETRv4 改了哪里

保留部分：

- Backbone 基本不变。
- Hybrid Encoder / FPN 基本不变。
- Transformer Decoder 主体基本不变。
- 单阶段 query-based 检测方式不变。

新增或修改部分：

| 模块 | 原始 RT-DETRv4 | GPPoint-DETR |
|---|---|---|
| 输出头 | 分类头 + bbox 头 | 分类头 + bbox 头 + `has_picking` 头 + `point_offset` 头 |
| 点表示 | 无 | `top-center` normalized offset |
| 实例绑定 | 无采摘点绑定 | `query_box_top` |
| 局部 cue | 无 | Top Local ROI |
| 损失函数 | 检测损失 | 检测损失 + BCE 可见性损失 + Wing 点损失 |
| 后处理 | boxes / labels / scores | boxes / labels / scores / has_picking / picking_point |
| 评价指标 | bbox AP | AP + F1 + pair_count + mean L2 + `|dy|` |

结构图：

```text
outputs/grape_point_v7_sci_ready/figures/gppoint_detr_vs_rtdetrv4_structure.png
```

答辩标准回答：

> GPPoint-DETR 没有大改 RT-DETRv4 的 backbone，而是在 decoder query 输出端增加了采摘点相关分支，并在训练、后处理和评价阶段配套支持 picking point。核心改动集中在实例级输出头、点偏移表达、顶部局部特征和评价链条上。

---

## 7. has_picking 和 point_offset 的作用

### 7.1 has_picking

`has_picking` 是一个实例级二分类分支，用于判断当前 grape 实例是否存在可见采摘点。

它解决的问题是：

```text
不是每串葡萄都有可见采摘点。
```

如果没有 `has_picking`，模型可能会被迫对所有葡萄串都预测一个点，导致无效点或误采摘候选。

代码位置：

| 功能 | 文件 |
|---|---|
| 输出分支 | `engine/rtv4/dfine_decoder.py` |
| 标签读取 | `engine/data/dataset/coco_dataset.py` |
| 损失计算 | `engine/rtv4/rtv4_criterion.py` |
| 后处理 sigmoid 与阈值判断 | `engine/rtv4/postprocessor.py` |
| F1 评价 | `engine/data/dataset/coco_eval.py` |

答辩标准回答：

> `has_picking` 用来回答“这串葡萄有没有可见采摘点”。它避免模型对无采摘点或采摘点不可见的葡萄串强行输出有效点，是采摘点链条中的可见性判断环节。

### 7.2 point_offset

`point_offset` 是一个二维回归分支，用于预测采摘点相对 grape 框参考点的归一化偏移。

标签定义：

```math
o_i =
\left(
\frac{u_i-a_i^x}{w_i},
\frac{v_i-a_i^y}{h_i}
\right)
```

推理解码：

```math
\hat{p}_i =
\hat{a}_i + \hat{o}_i \odot (\hat{w}_i,\hat{h}_i)
```

其中：

- `p_i=(u_i,v_i)` 是真实采摘点。
- `a_i` 是参考锚点。
- `w_i,h_i` 是 grape 框宽高。
- 推理时使用预测框对应的锚点和宽高。

代码位置：

| 功能 | 文件 |
|---|---|
| 输出分支 | `engine/rtv4/dfine_decoder.py` |
| offset 标签生成 | `engine/data/dataset/grape_point_dataset.py` |
| offset 编码与解码 | `engine/rtv4/point_utils.py` |
| 损失计算 | `engine/rtv4/rtv4_criterion.py` |
| 解码为 picking point | `engine/rtv4/postprocessor.py` |
| L2 / dx / dy 评价 | `engine/data/dataset/coco_eval.py` |

答辩标准回答：

> `point_offset` 用来回答“采摘点在哪里”。它不是直接回归绝对坐标，而是以 grape 框上部参考点为基准回归归一化偏移，再在后处理中恢复为图像坐标。

---

## 8. 为什么使用 top-center

采摘点通常位于葡萄串上部或果梗附近。如果从 bbox 中心点回归采摘点，模型需要长期学习一个明显向上的纵向偏移，容易导致 `dy` 漂移。

GPPoint-DETR 使用 top-center 锚点：

```math
a_i = (x_i + 0.5w_i,\; y_i + 0.12h_i)
```

当前主模型中：

```yaml
point_offset_mode: top_center
point_top_anchor_ratio: 0.12
point_coord_weight_y: 1.30
```

代码位置：

| 功能 | 文件 |
|---|---|
| 锚点定义 | `engine/rtv4/point_utils.py` |
| 标签偏移生成 | `engine/data/dataset/grape_point_dataset.py` |
| 推理解码 | `engine/rtv4/postprocessor.py` |
| 配置参数 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp1.yml` |

答辩标准回答：

> 采用 top-center 是因为采摘点多数位于葡萄串上部。把回归参考点移到上部，可以缩短模型需要学习的纵向偏移，降低 `dy` 漂移风险。实验中 top-center 明显压低了纵向误差，说明这个几何先验是有效的。

---

## 9. Top Local ROI 的作用

Top Local ROI 是从预测 grape 框上部区域提取的局部视觉特征，用于给采摘点分支提供果梗附近的局部纹理信息。

它不是 two-stage ROI detector，而是在原单阶段框架中为点分支补充上部局部 cue。

关键配置：

```yaml
DFINETransformer:
  point_instance_binding_mode: query_box_top
  point_top_local_width_scale: 1.08
  point_top_local_y_min_ratio: -0.10
  point_top_local_y_max_ratio: 0.40
```

代码位置：

| 功能 | 文件 |
|---|---|
| ROI 参数配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` |
| ROI 构造与特征采样 | `engine/rtv4/dfine_decoder.py` |

答辩标准回答：

> Top Local ROI 的作用是让采摘点分支重点关注葡萄串上部区域。采摘点通常出现在果梗附近，整框或全图特征会引入下部果粒、叶片和背景干扰。Top Local ROI 可以提供更相关的局部视觉信息。

---

## 10. query_box_top 的作用

`query_box_top` 可以理解为 point 分支的实例绑定方式。

它融合三类信息：

| 信息 | 作用 |
|---|---|
| decoder query | 表示当前 grape 实例是谁 |
| predicted box / top-center | 约束采摘点的大致空间范围 |
| Top Local ROI | 提供上部果梗区域局部视觉 cue |

答辩标准回答：

> `query_box_top` 的作用是把采摘点预测绑定到具体 grape 实例上，同时利用预测框上部区域的局部视觉信息。简单说，query 负责“属于哪串葡萄”，box/top-center 负责“点大概在哪个空间范围”，Top Local ROI 负责“看果梗附近的局部线索”。

---

## 11. loss 如何支持 picking point

总损失可概括为：

```math
L = L_{det} + \lambda_{has} L_{has} + \lambda_{point} L_{point}
```

当前主模型配置：

```yaml
loss_has_picking: 1.2
loss_picking_offset: 4.0
wing_loss_omega: 0.25
wing_loss_epsilon: 0.05
point_coord_weight_y: 1.30
```

### 11.1 loss_has_picking

作用：

- 使用 BCEWithLogitsLoss 训练可见性判断。
- 对匹配到 grape 的 query 判断该 grape 是否有可见采摘点。

代码位置：

```text
engine/rtv4/rtv4_criterion.py
```

### 11.2 loss_picking_offset

作用：

- 只在 `has_picking=true` 的可见样本上计算。
- 使用 Wing Loss 训练点偏移。
- y 方向可加权，主模型中 y 权重为 `1.30`。
- `small_weight` 变体中对 small grape 的点损失额外加权。

代码位置：

```text
engine/rtv4/rtv4_criterion.py
```

答辩标准回答：

> loss 部分在原检测损失之外增加了两个采摘点相关损失：一个是 `has_picking` 的二分类损失，另一个是点偏移的 Wing Loss。点偏移损失只对有可见采摘点的样本计算，并对 y 方向误差加权，以针对历史上最明显的纵向漂移问题。

---

## 12. postprocessor 如何支持 picking point

原始后处理主要输出：

```text
labels / scores / boxes
```

GPPoint-DETR 后处理额外输出：

```text
has_picking_scores / has_picking / picking_points / picking_offsets
```

流程：

```text
pred_has_picking -> sigmoid -> has_picking_scores
has_picking_scores >= threshold -> has_picking flag
pred_picking_offsets + predicted boxes -> picking_points
```

代码位置：

```text
engine/rtv4/postprocessor.py
```

答辩标准回答：

> postprocessor 会把 `has_picking` logit 转成概率，并根据预测框和 point offset 解码出图像坐标系下的 picking point。因此最终输出不只是 bbox，还包括每个 grape 实例是否有采摘点以及采摘点坐标。

---

## 13. evaluator 如何支持 picking point

评估流程：

```text
预测 grape box 与 GT grape box 做 IoU 匹配
  -> IoU >= 0.5 认为 grape 匹配成功
  -> 统计 has_picking precision / recall / F1
  -> 当 GT visible 且 Pred visible 时，统计 point error
```

输出指标：

- `grape AP`
- `has_picking precision`
- `has_picking recall`
- `has_picking F1`
- `point_pair_count`
- `mean L2`
- `median L2`
- `p90 L2`
- `mean |dx|`
- `mean |dy|`

代码位置：

```text
engine/data/dataset/coco_eval.py
```

答辩标准回答：

> 点误差不是在全图随便统计的，而是在 grape 框匹配成功后，且真实和预测都认为有可见采摘点时才统计。因此 `pair_count` 反映的是检测、可见性和点定位链条是否真正打通。

---

## 14. baseline_replay、v7_exp2、small_weight、tight_toproi 的区别

| 实验名 | 论文定位 | 关键设置 | 主要结论 |
|---|---|---|---|
| `baseline_replay` | 正式对照基线 | point 路线正式复现实验 | 用作所有主结果的统一比较基准 |
| `v7_exp2` / `GPPoint-DETR` | 当前主模型 | `top-center + query_box_top + Top Local ROI` | F1、pair_count、mean L2、`|dy|` 稳定改善 |
| `small_weight` | 定向补强实验 | 在 point loss 上对 small grape 加权 | small 场景改善，但整体存在 trade-off |
| `tight_toproi` | 负例实验 | 收紧 Top Local ROI | 整体回退，不作为主模型 |

### 14.1 baseline_replay

主要指标：

```text
grape AP = 0.6342
has_picking F1 = 0.7209
pair_count = 155
mean L2 = 29.27
|dy| = 22.53
```

答辩定位：

> `baseline_replay` 是正式对照基线，用于保证后续结果不是和旧历史实验随意比较。

### 14.2 GPPoint-DETR

主要指标：

```text
grape AP = 0.6424
has_picking F1 = 0.7661
pair_count = 190
mean L2 = 24.87
|dy| = 16.89
```

答辩定位：

> GPPoint-DETR 是当前论文主模型。它的核心价值是打通并稳定了 grape detection、has_picking 和 point localization 这条实例级采摘点链条。

### 14.3 small_weight

主要指标：

```text
grape AP = 0.6291
has_picking F1 = 0.7762
pair_count = 189
mean L2 = 23.35
|dy| = 17.25
```

答辩定位：

> `small_weight` 是困难场景补强实验，证明 small grape 是残余难点且可以通过 loss 加权定向缓解。但它带来 AP 和部分整体指标取舍，因此不替代 GPPoint-DETR 主模型。

### 14.4 tight_toproi

主要指标：

```text
grape AP = 0.6330
has_picking F1 = 0.7592
pair_count = 175
mean L2 = 26.66
|dy| = 17.46
```

答辩定位：

> `tight_toproi` 是负结果。它说明局部 ROI 不是越窄越好，过度收紧会丢失必要上下文，因此 GPPoint-DETR 采用适度上部 ROI，而不是极窄搜索区域。

---

## 15. 为什么 AP 没有明显提升但仍说方法有效

AP 主要评价 grape bbox 检测质量，但本文目标是采摘点链条：

```text
grape detection -> has_picking -> point localization
```

因此仅看 AP 不足以评价本文方法。

更关键指标：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| `has_picking F1` | 是否能判断可见采摘点 | 对应“能不能采” |
| `pair_count` | 有效点配对数量 | 对应链条是否打通 |
| `mean L2` | 点定位总体误差 | 对应采摘点位置是否准 |
| `|dy|` | 纵向误差 | 对应历史最严重的上下漂移 |

多 seed 稳定性：

| 指标 | GPPoint-DETR mean ± std |
|---|---:|
| AP | 0.6333 ± 0.0101 |
| F1 | 0.7551 ± 0.0142 |
| pair_count | 179.7 ± 10.0 |
| mean L2 | 24.41 ± 0.88 |
| `|dy|` | 16.49 ± 0.44 |

答辩标准回答：

> AP 不是本文最稳定的主结论，因为它只衡量 grape 框检测。而本文的目标是让模型输出更可用的采摘点链条。GPPoint-DETR 在 `has_picking F1`、`pair_count`、`mean L2` 和 `|dy|` 上表现出更稳定改善，因此方法是有效的，但不能夸大为所有检测指标全面提升。

---

## 16. 数据集与标注问答

### 16.1 数据集是自己采集的吗

答辩标准回答：

> 原始图像来自公开葡萄图像数据源，本文不把图像采集作为贡献。本文主要工作是对公开图像进行清洗、去重、统一尺寸、重新标注和任务标签重构，使其适合葡萄串识别与采摘点定位任务。

### 16.2 数据集规模和划分方式

答辩标准回答：

> 最终纳入实验的图像共 1820 张。图像经过重复检索和人工清洗后，按照训练集、验证集和测试集 7:2:1 的比例划分。所有模型在相同划分下训练和评价。

### 16.3 如何避免数据泄漏

答辩标准回答：

> 在划分数据集之前，先使用重复检索工具和人工检查排除重复或高度相似图像，然后再进行训练、验证和测试划分，以降低重复样本跨集合导致结果偏高的风险。

### 16.4 为什么远距离和过小葡萄串不标注

答辩标准回答：

> 这不是随意漏标，而是基于采摘点定位任务的样本语义筛选。远距离或尺度过小的葡萄串通常无法提供稳定采摘点依据，也不符合后续采摘执行需求。因此本文只标注主体轮廓较清晰、具有采摘点判断意义的葡萄串。

### 16.5 采摘点如何定义

答辩标准回答：

> 采摘点主要选择葡萄果梗靠近葡萄串、具有潜在可采摘意义的区域。若该区域可见且能够明确定位，则标记 `has_picking=1` 并记录 picking point；若采摘点被遮挡、不可辨识或不具备明确标注依据，则不参与点定位监督。

---

## 17. 机械臂应用与局限

### 17.1 采摘点坐标能不能直接给机械臂用

不能直接用于机械臂执行。

当前模型输出的是图像二维坐标：

```text
picking_point = (x, y)
```

真实机械臂执行还需要：

- 相机内参。
- 深度信息。
- 相机外参。
- 手眼标定。
- 从图像坐标到机械臂基坐标系的转换。
- 末端执行器姿态规划。

答辩标准回答：

> 当前模型输出的是二维图像坐标，不是机械臂基坐标系下的三维作业点。它可以作为后续机械臂定位流程的视觉输入，但若要直接控制机械臂，还需要结合深度信息、相机标定和手眼标定完成三维坐标转换。

### 17.2 为什么仍然要输出二维点

答辩标准回答：

> 二维采摘点是连接视觉识别和机械臂定位的中间表示。bbox 只能告诉系统目标区域，而 picking point 能提供更明确的剪切或抓取参考位置，为后续三维定位和路径规划提供输入。

### 17.3 最大不足

当前主要不足：

- small grape 仍然困难。
- heavy occlusion / 多串相邻场景仍有残余误差。
- 采摘点标注具有一定任务依赖性和主观性。
- 当前只输出二维图像点，尚未完成真实机械臂平台上的三维定位和闭环采摘验证。
- AP 存在 seed 敏感性，不宜作为稳定主结论。

答辩标准回答：

> 本文最大不足是 small grape、重遮挡和多串相邻场景仍然困难，同时当前模型只完成二维图像采摘点定位，还没有在真实机械臂平台上完成三维定位和闭环采摘验证。

---

## 18. 代码实现总地图

| 模块 | 代码文件 | 作用 | 论文对应位置 |
|---|---|---|---|
| 训练入口 | `train.py` | 读取配置、构建数据集、模型和训练流程 | 实验设置 |
| 标签读取 | `engine/data/dataset/coco_dataset.py` | 读取 grape bbox、`has_picking`、`picking_point` | 数据集与标签定义 |
| 点标签生成 | `engine/data/dataset/grape_point_dataset.py` | 根据 bbox 和 point 生成 normalized offset | 方法公式 |
| 点编码解码 | `engine/rtv4/point_utils.py` | 实现 center/top-center offset 编解码 | 方法公式 |
| decoder 输出 | `engine/rtv4/dfine_decoder.py` | 新增 `has_picking` 和 `point_offset` 分支，实现 `query_box_top` 与 Top Local ROI | 模型结构 |
| 损失函数 | `engine/rtv4/rtv4_criterion.py` | 计算 BCE 可见性损失、Wing 点损失、y 加权和 small 加权 | 训练目标 |
| 后处理 | `engine/rtv4/postprocessor.py` | 解码 `has_picking` 和 `picking_point` | 推理流程 |
| 评价器 | `engine/data/dataset/coco_eval.py` | 计算 AP、F1、pair_count、L2、dx、dy | 实验指标 |
| 报告脚本 | `tools/make_grape_point_report.py` | 统一生成实验报告 | 结果整理 |
| 主模型报告 | `tools/make_grape_point_main_report.py` | GPPoint-DETR 主模型报告入口 | 论文主结果 |
| baseline 报告 | `tools/make_grape_point_baseline_report.py` | baseline_replay 报告入口 | 对照实验 |
| small 报告 | `tools/make_grape_point_small_grape_report.py` | small_weight 报告入口 | 困难场景补强 |
| SCI 资产 | `tools/make_grape_point_sci_assets.py` | 稳定性、切片、外部对比和写作资产 | 论文写作 |

---

## 19. 关键配置文件

| 实验 | 配置文件 |
|---|---|
| baseline_replay | `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml` |
| top_center / v7_exp1 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp1.yml` |
| GPPoint-DETR / v7_exp2 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` |
| 当前主模型入口 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml` |
| small_weight | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_small_weight.yml` |
| small_weight 当前入口 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml` |

注意：

> 部分历史配置或报告内部仍可能保留旧实验目录名，例如 `grape_point_v7_exp2_query_box_top_center_toproi`。当前论文命名统一使用 GPPoint-DETR，实际对应历史 `v7_exp2`。

---

## 20. 常用结果目录

| 目录 | 含义 |
|---|---|
| `outputs/grape_point_v6_baseline_replay` | 正式 baseline |
| `outputs/grape_point_v7_exp1_query_box_top_center` | top-center 过渡实验 |
| `outputs/grape_point_gppoint_detr_main` | GPPoint-DETR 主模型 |
| `outputs/grape_point_gppoint_detr_main_repro1` | GPPoint-DETR 复现实验 1 |
| `outputs/grape_point_gppoint_detr_main_seed2026` | GPPoint-DETR 复现实验 2 |
| `outputs/grape_point_gppoint_detr_small_weight` | small grape 定向补强实验 |
| `outputs/history_suites/negative_variants/v7_exp2_tight_toproi` | tight_toproi 负结果摘要 |
| `outputs/grape_point_v7_sci_ready` | SCI 写作资产 |
| `outputs/grape_point_v7_paper_ready` | 普通论文写作资产 |

---

## 21. 实验结果如何解释

核心主表：

| 模型 | grape AP | has_picking F1 | pair_count | mean L2 | `|dy|` | 论文定位 |
|---|---:|---:|---:|---:|---:|---|
| baseline_replay | 0.6342 | 0.7209 | 155 | 29.27 | 22.53 | 正式基线 |
| top_center | 0.6335 | 0.6938 | 145 | 24.73 | 17.83 | 证明顶部锚点能压低 dy，但链条不够稳定 |
| GPPoint-DETR | 0.6424 | 0.7661 | 190 | 24.87 | 16.89 | 当前主模型 |
| small_weight | 0.6291 | 0.7762 | 189 | 23.35 | 17.25 | small 场景补强，不替代主模型 |
| tight_toproi | 0.6330 | 0.7592 | 175 | 26.66 | 17.46 | 负结果，说明 ROI 不能过度收紧 |

推荐解释：

- `top_center` 证明几何先验能压低点误差，尤其是 `|dy|`。
- `top_center` 的 F1 和 pair_count 不理想，说明仅靠几何锚点不足。
- GPPoint-DETR 通过 `query_box_top + Top Local ROI` 同时改善 F1、pair_count、mean L2 和 `|dy|`。
- `small_weight` 说明 small grape 是残余难点，并且可以被定向缓解，但存在整体性能取舍。
- `tight_toproi` 说明局部特征不是越窄越好，过度收紧会损失上下文。

---

## 22. 高频问题速答

### 问：一句话说清你的工作

答：

> 我将 RT-DETRv4 从普通葡萄串检测器扩展为面向葡萄串实例的采摘点定位模型，使其能够输出葡萄串框、采摘点可见性以及图像坐标系下的采摘点。

### 问：你的模型叫什么

答：

> 模型命名为 GPPoint-DETR，历史实验版本对应 `v7_exp2`。

### 问：你的论文主要创新点是什么

答：

> 主要创新点包括任务重构、实例级输出分支和上部空间先验。具体来说，本文将 picking bbox 检测改为 per-grape `has_picking + point localization`，在 RT-DETRv4 上增加 `has_picking` 和 `point_offset` 分支，并结合 `top-center`、`query_box_top` 与 Top Local ROI 改善采摘点链条稳定性。

### 问：为什么不用 picking bbox

答：

> 因为 picking 不是独立目标，而是依附于葡萄串的局部采摘位置。独立 bbox 容易出现小目标不稳定、重复框和错关联，而 per-grape 点定位更符合采摘决策。

### 问：相比 RT-DETRv4 改了哪里

答：

> 主干基本不变，主要新增 `has_picking` 分支、`point_offset` 分支、`top-center` 点表达、`query_box_top` 实例绑定和 Top Local ROI 局部特征，并在 loss、postprocessor 和 evaluator 中支持采摘点。

### 问：has_picking 和 point_offset 分别有什么作用

答：

> `has_picking` 判断当前葡萄串是否有可见采摘点，`point_offset` 回归采摘点相对葡萄框上部参考点的归一化偏移，最后解码为图像二维坐标。

### 问：为什么用 top-center

答：

> 采摘点通常位于葡萄串上部或果梗附近。top-center 让回归参考点更接近真实采摘区域，减少模型需要学习的纵向偏移，从而缓解 `dy` 漂移。

### 问：Top Local ROI 有什么作用

答：

> Top Local ROI 给点分支提供葡萄串上部局部视觉信息，让模型更关注果梗附近区域，减少下部果粒、叶片和背景干扰。

### 问：为什么 AP 没有明显提升但仍说有效

答：

> AP 主要衡量 grape bbox 检测，本文目标是采摘点链条。GPPoint-DETR 的稳定收益主要体现在 `has_picking F1`、`pair_count`、`mean L2` 和 `|dy|` 上，因此方法有效，但不能说所有指标全面提升。

### 问：你的采摘点能不能直接给机械臂用

答：

> 不能直接用于机械臂执行。当前输出的是二维图像坐标，需要结合深度信息、相机标定和手眼标定转换为机械臂基坐标系下的三维作业点。

### 问：最大不足是什么

答：

> small grape、重遮挡和多串相邻场景仍然困难；采摘点标注有一定任务主观性；当前还没有完成真实机械臂平台上的三维定位和闭环采摘验证。

### 问：未来工作怎么做

答：

> 可以补充实地数据，增强 small grape 和复杂遮挡场景建模，引入点定位不确定性估计，并结合 RGB-D 或双目相机与手眼标定完成三维采摘点定位验证。

---

## 23. 答辩时最容易误说的地方

不要这样说：

```text
GPPoint-DETR 全面提升了所有指标。
```

建议这样说：

```text
GPPoint-DETR 的稳定收益主要体现在实例级采摘点链条，包括 has_picking F1、pair_count、mean L2 和 |dy|。
```

不要这样说：

```text
模型输出可以直接控制机械臂采摘。
```

建议这样说：

```text
模型输出二维图像采摘点，可作为机械臂三维定位流程的视觉输入，但仍需要深度信息和手眼标定。
```

不要这样说：

```text
small_weight 是最终最优模型。
```

建议这样说：

```text
small_weight 是困难场景补强实验，说明 small grape 可以被定向改善，但因为存在整体性能取舍，不替代 GPPoint-DETR 主模型。
```

不要这样说：

```text
tight_toproi 没用所以不重要。
```

建议这样说：

```text
tight_toproi 是负结果，说明 Top Local ROI 不能简单越窄越好，适度局部上下文对采摘点定位仍然必要。
```

---

## 24. 答辩前推荐复习顺序

1. 先背第 1 节“答辩总口径”。
2. 再看第 2 到 10 节，掌握为什么这样改模型。
3. 接着看第 11 到 13 节，掌握 loss、postprocessor、evaluator 如何支持 picking point。
4. 然后看第 14 到 21 节，掌握实验版本和结果解释。
5. 最后看第 22 到 23 节，练习高频问题速答和避免误说。

---

## 25. 最后一段完整回答模板

如果老师问“你到底做了什么”，可以这样回答：

> 我的工作不是单纯提高葡萄串检测 AP，而是针对葡萄采摘任务中“检测到葡萄串后仍缺少可用采摘点”的问题，对 RT-DETRv4 进行了实例级输出扩展。原始 RT-DETRv4 只能输出类别和边界框，而早期将采摘点作为独立小框检测时容易出现重复框和错关联。因此，我提出 GPPoint-DETR，在每个 grape 实例上增加 `has_picking` 判别和 `point_offset` 点定位分支，并结合 `top-center` 几何先验、`query_box_top` 实例绑定和 Top Local ROI 局部特征，使模型能够输出葡萄串框、是否存在采摘点以及图像坐标系下的采摘点位置。实验结果表明，模型主要提升了 `has_picking F1`、有效配对数量、点定位误差和纵向误差控制，而 AP 不是本文最主要的贡献点。

如果老师问“你的结果为什么有意义”，可以这样回答：

> 普通葡萄串检测只能告诉系统哪里有葡萄，但不能告诉系统从哪里采。本文输出的 picking point 虽然仍是二维图像坐标，但它把检测结果进一步转化为具有操作指向性的视觉结果，可作为后续结合深度信息和手眼标定进行机械臂三维定位的前端输入。因此，本文的意义在于将视觉模型从识别任务推进到更接近采摘执行需求的实例级定位任务。

---

## 26. 源码核对版模型理解报告

本节用于回答“模型到底怎么跑、代码到底改在哪里、论文方法章节能不能对上源码”。下面内容以当前项目源码为准，不再按早期实验口头说法推断。

### 26.1 当前训练链路如何从 train.py 走到模型、loss、postprocessor 和 evaluator

整体链路可以概括为：

```text
train.py
  -> YAMLConfig 解析配置
  -> TASKS[cfg.yaml_cfg["task"]] 创建 solver
  -> solver 构建 dataloader / model / criterion / postprocessor / evaluator
  -> det_engine.train_one_epoch 前向训练并计算 loss
  -> det_engine.evaluate 前向推理、后处理、COCO + picking point 评价
```

关键代码位置：

- `train.py:28` 定义主入口 `main(args)`。
- `train.py:37` 将命令行参数合并到配置。
- `train.py:41` 通过 `YAMLConfig(args.config, **update_dict)` 读取 yaml 配置。
- `train.py:49` 根据配置中的 `task` 创建 solver。
- `train.py:51-54` 根据 `--test-only` 决定执行验证还是训练。
- `engine/core/yaml_config.py:44-47` 根据配置懒加载并实例化 `model`。
- `engine/core/yaml_config.py:82-85` 根据配置实例化 `postprocessor`。
- `engine/core/yaml_config.py:88-91` 根据配置实例化 `criterion`。
- `engine/core/yaml_config.py:138-143` 使用验证集 COCO API 创建 `evaluator`。
- `engine/solver/_solver.py:65-70` 将 model、criterion、postprocessor 放到指定 device。
- `engine/solver/_solver.py:100-104` 构建验证 dataloader 和 evaluator。
- `engine/solver/det_engine.py:77-91` 训练阶段使用 AMP 时执行 `outputs = model(samples, targets=targets)`，再由 `criterion(outputs, targets)` 计算损失。
- `engine/solver/det_engine.py:112-114` 非 AMP 分支同样执行模型前向与 loss 计算。
- `engine/solver/det_engine.py:184-188` 验证阶段执行模型前向并调用 `postprocessor(outputs, orig_target_sizes)`。
- `engine/solver/det_engine.py:194-196` 将后处理结果送入 evaluator。
- `engine/solver/det_engine.py:211-217` 汇总 COCO 指标和 picking point 额外指标。

答辩解释时可以说：

> 训练入口仍是 RT-DETRv4 原有的 `train.py` 配置驱动流程。GPPoint-DETR 没有重写训练框架，而是在数据集字段、decoder 输出、criterion 损失、postprocessor 解码和 evaluator 评价五个位置扩展了采摘点相关逻辑。

### 26.2 原始 RT-DETRv4 的结构是什么

从当前项目代码看，原始 RT-DETRv4 主体仍是单阶段 DETR 系列检测框架：

```text
image
  -> backbone
  -> encoder
  -> transformer decoder
  -> classification head
  -> box regression head
  -> labels / scores / boxes
```

对应代码：

- `engine/rtv4/rtv4.py:13` 定义 `RTv4` 模型类。
- `engine/rtv4/rtv4.py:15` 构造函数注入 `backbone`、`encoder`、`decoder`。
- `engine/rtv4/rtv4.py:27` 开始模型前向。
- `engine/rtv4/rtv4.py:28` 执行 backbone。
- `engine/rtv4/rtv4.py:30` 执行 encoder。
- `engine/rtv4/rtv4.py:39` 执行 decoder。
- `engine/rtv4/rtv4.py:41-45` 返回 decoder 输出。

原始 RT-DETRv4 的核心输出是检测类别与边界框，不包含“某串葡萄是否有可见采摘点”和“采摘点坐标”。因此，如果直接用原始 RT-DETRv4 做 picking，只能把采摘点当成一个额外小目标类别，这就是早期 `picking bbox` 路线。

### 26.3 GPPoint-DETR 相比原始 RT-DETRv4 改了哪些地方

GPPoint-DETR 没有大改 backbone，也没有变成 two-stage ROI detector。它是在每个 grape query 上增加实例级采摘点输出链条：

```text
原始 RT-DETRv4:
query feature -> class + bbox

GPPoint-DETR:
query feature + query box/top cue + top local feature
  -> class + bbox
  -> has_picking
  -> point_offset
  -> picking_point
```

主要改动可以分为五类：

- 数据集字段扩展：每个 grape 实例除 bbox 外，还读取或生成 `has_picking`、`picking_points`、`picking_offsets`。
- decoder 分支扩展：在 decoder 内增加 `has_picking` 分支和 `point_offset` 分支。
- 点表达方式改变：从普通 bbox/center 表达转为基于葡萄框 `top-center` 锚点的 offset。
- 实例绑定增强：使用 `query_box_top`，让点预测更强地依赖当前 query 对应的葡萄框几何位置。
- 局部特征增强：使用 Top Local ROI 从葡萄框上部区域提取局部视觉线索，辅助判断和定位采摘点。

答辩中最稳妥的表述是：

> GPPoint-DETR 的核心不是替换 RT-DETRv4 主干，而是把原来的目标检测 query 扩展为“葡萄串实例级采摘点 query”。每个 query 不仅预测 grape bbox，还预测该实例是否存在可见采摘点以及采摘点相对 top-center 的偏移。

### 26.4 has_picking 分支在哪里定义、计算 loss、后处理和评价

`has_picking` 的作用是判断某个 grape 实例是否存在可见采摘点。它不是新类别，而是 grape 实例上的二分类属性。

数据读取位置：

- `engine/data/dataset/coco_dataset.py:151-156` 从标注中读取 `has_picking`。
- `engine/data/dataset/coco_dataset.py:204-209` 将 `has_picking` 写入 target。
- `engine/data/dataset/grape_point_dataset.py:100-125` 保证 target 中存在默认 point 相关字段。

分支定义位置：

- `engine/rtv4/dfine_decoder.py:491` 使用 `use_picking_point_head` 控制是否启用点分支。
- `engine/rtv4/dfine_decoder.py:605-614` 定义 decoder 层的 `dec_picking_head`。
- `engine/rtv4/dfine_decoder.py:625-631` 定义 encoder/pre 阶段的 `pre_picking_head`。
- `engine/rtv4/dfine_decoder.py:907-934` 在 `_predict_point_branch` 中输出 `has_picking` logit 和 offset。
- `engine/rtv4/dfine_decoder.py:1286` 将最后一层 `pred_has_picking` 写入输出字典。
- `engine/rtv4/dfine_decoder.py:1297-1304` 和 `engine/rtv4/dfine_decoder.py:1315-1321` 将 aux 输出中的 `pred_has_picking` 一并传出。

loss 位置：

- `engine/rtv4/rtv4_criterion.py:323-338` 定义 `loss_has_picking`。
- `engine/rtv4/rtv4_criterion.py:337` 使用 `BCEWithLogitsLoss` 计算二分类损失。
- `engine/rtv4/rtv4_criterion.py:641-642` 将 `has_picking` 注册到 loss map。
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_base.yml:50-63` 在配置中启用 `loss_has_picking` 和对应权重。

后处理位置：

- `engine/rtv4/postprocessor.py:112` 判断是否执行 picking point 后处理。
- `engine/rtv4/postprocessor.py:116-124` 根据 top-k 检测结果同步选取 `pred_has_picking`。
- `engine/rtv4/postprocessor.py:154-155` 对 `has_picking` logit 做 sigmoid 并按阈值转为可见性判断。
- `engine/rtv4/postprocessor.py:179-187` 将 `has_picking` 和 `has_picking_scores` 写入最终结果。

评价位置：

- `engine/data/dataset/coco_eval.py:287` 定义 `GrapePointEvaluator`。
- `engine/data/dataset/coco_eval.py:288-291` 设置 `point_iou_threshold` 和 `has_picking_threshold`。
- `engine/data/dataset/coco_eval.py:392-405` 读取预测中的 box、has score 和 point。
- `engine/data/dataset/coco_eval.py:455-457` 判断 GT 与预测是否为可见采摘点。
- `engine/data/dataset/coco_eval.py:338-340` 计算 precision、recall 和 F1。

### 26.5 point_offset 分支在哪里定义、如何解码为 picking_point、如何计算 loss、如何评价

`point_offset` 的作用是回归采摘点相对葡萄框锚点的归一化偏移，而不是直接回归绝对像素坐标。最终输出的 `picking_point` 是通过 bbox 和 offset 解码得到的二维图像坐标。

数据与标签生成：

- `engine/data/dataset/coco_dataset.py:158-168` 读取 `picking_points`。
- `engine/data/dataset/coco_dataset.py:170-180` 读取 `picking_offsets`。
- `engine/data/dataset/grape_point_dataset.py:32-45` 配置是否重新生成 point offset，以及 offset 模式和 top anchor ratio。
- `engine/data/dataset/grape_point_dataset.py:81-86` 调用 `normalized_offsets_from_boxes_and_points` 生成 offset。
- `engine/data/dataset/grape_point_dataset.py:87` 仅对可见采摘点保留 offset，不可见样本 offset 置零。

分支定义：

- `engine/rtv4/dfine_decoder.py:615-624` 定义 decoder 层的 `dec_picking_offset_head`。
- `engine/rtv4/dfine_decoder.py:632-638` 定义 pre 阶段的 `pre_picking_offset_head`。
- `engine/rtv4/dfine_decoder.py:907-934` 在 `_predict_point_branch` 中预测 offset。
- `engine/rtv4/dfine_decoder.py:1287` 将最后一层 `pred_picking_offsets` 写入输出字典。

offset 与 point 的几何关系：

- `engine/rtv4/point_utils.py:8-22` 规范化 offset 模式名称。
- `engine/rtv4/point_utils.py:30-44` 根据 bbox 计算 anchor，其中 `top_center` 为葡萄框上部中心附近位置。
- `engine/rtv4/point_utils.py:47-63` 根据 bbox 与 point 计算归一化 offset。
- `engine/rtv4/point_utils.py:66-82` 根据 bbox 与 offset 解码绝对 point。
- `engine/rtv4/point_utils.py:79-80` 是 `top_center` 解码的核心公式。
- `engine/rtv4/point_utils.py:85-92` 将解码点裁剪到图像范围内。
- `engine/rtv4/point_utils.py:95-111` 从 boxes 和 points 生成 normalized offsets。

loss 位置：

- `engine/rtv4/rtv4_criterion.py:340-360` 通过 Hungarian 匹配结果收集已匹配 query 的预测 offset、GT has_picking、GT offset 和 GT box。
- `engine/rtv4/rtv4_criterion.py:445-473` 定义 `loss_picking_offset`。
- `engine/rtv4/rtv4_criterion.py:453` 只对 `target_has_picking > 0.5` 的实例计算点定位 loss。
- `engine/rtv4/rtv4_criterion.py:457` 计算基础点损失。
- `engine/rtv4/rtv4_criterion.py:458-463` 支持 x/y 坐标加权。
- `engine/rtv4/rtv4_criterion.py:465-470` 支持 small grape 样本加权。
- `engine/rtv4/rtv4_criterion.py:577-593` 定义 `compute_point_loss`，当前支持 L1、Smooth L1 和 Wing Loss。

后处理与评价：

- `engine/rtv4/postprocessor.py:126-136` 可选地限制 offset 范围。
- `engine/rtv4/postprocessor.py:138` 使用检测框的绝对坐标作为解码基础。
- `engine/rtv4/postprocessor.py:146-152` 调用 `absolute_points_from_boxes_and_offsets` 解码 `picking_points`。
- `engine/rtv4/postprocessor.py:179-187` 将 `picking_points` 写入结果。
- `engine/data/dataset/coco_eval.py:425-443` 先用预测 bbox 与 GT bbox 做 IoU 匹配。
- `engine/data/dataset/coco_eval.py:463-471` 在可见采摘点配对成功时计算 dx、dy 和 L2。
- `engine/data/dataset/coco_eval.py:351-354` 汇总 `point_pair_count`、`point_mae_x`、`point_mae_y` 和 `point_mean_l2`。

### 26.6 top-center、query_box_top、Top Local ROI 分别如何实现

#### top-center

`top-center` 是采摘点几何先验。葡萄串采摘点通常位于葡萄串上方或上部连接区域，因此点 offset 不再围绕 bbox 中心建模，而是围绕 bbox 上部中心附近位置建模。

代码位置：

- `engine/rtv4/point_utils.py:30-44` 根据 bbox 与 `top_anchor_ratio` 计算 anchor。
- `engine/rtv4/point_utils.py:79-80` 用 top-center anchor 解码绝对点坐标。
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp1.yml` 中设置 `point_offset_mode: top_center` 和 `point_top_anchor_ratio: 0.12`。
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` 继承 v7_exp1，因此也使用 top-center。

答辩说法：

> top-center 不是经验乱加，而是把采摘点相对葡萄串上部连接区域的空间先验显式编码进 offset 表达中，从而降低纵向回归难度。

#### query_box_top

`query_box_top` 是实例绑定增强方式。它让 point 分支不仅看 decoder hidden feature，还显式融合当前 query 对应的 bbox 几何信息和 top-center 位置编码。

代码位置：

- `engine/rtv4/dfine_decoder.py:496-506` 解析并检查 `point_instance_binding_mode`。
- `engine/rtv4/dfine_decoder.py:530` 标记是否使用 query box 绑定。
- `engine/rtv4/dfine_decoder.py:663-668` 定义 query position projection。
- `engine/rtv4/dfine_decoder.py:669-673` 定义 box geometry projection。
- `engine/rtv4/dfine_decoder.py:873-905` 在 `_fuse_point_feature` 中融合 hidden、local、query position 和 box geometry。
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` 中设置 `point_instance_binding_mode: query_box_top`。

答辩说法：

> query_box_top 的目的不是重新检测一个采摘点框，而是把采摘点预测绑定到当前 grape query 上，减少相邻葡萄串之间的错关联。

#### Top Local ROI

Top Local ROI 是局部视觉线索增强。它从当前预测葡萄框的上部区域提取局部特征，用于辅助判断采摘点是否可见以及 offset 应该往哪里回归。

代码位置：

- `engine/rtv4/dfine_decoder.py:515-517` 保存 Top Local ROI 的宽度和 y 范围参数。
- `engine/rtv4/dfine_decoder.py:529` 标记是否使用 top local cue。
- `engine/rtv4/dfine_decoder.py:651-656` 定义 local feature projection。
- `engine/rtv4/dfine_decoder.py:829-842` 构造 Top Local ROI，其中：
  - `roi_w = w * point_top_local_width_scale`
  - `top_y = cy - 0.5 * h`
  - `y1 = top_y + y_min * h`
  - `y2 = top_y + y_max * h`
- `engine/rtv4/dfine_decoder.py:844-871` 使用 `torchvision.ops.roi_align` 提取 ROI 特征并池化。
- `engine/rtv4/dfine_decoder.py:893-895` 将 Top Local ROI 特征融合进 point feature。
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` 中设置：
  - `point_use_top_local_cue: True`
  - `point_top_local_width_scale: 1.08`
  - `point_top_local_y_min_ratio: -0.10`
  - `point_top_local_y_max_ratio: 0.40`

答辩说法：

> Top Local ROI 不是 two-stage detector，因为它不生成新候选框，也不单独做 ROI 分类回归；它只是给当前 query 的点预测分支补充葡萄串上部局部特征。

### 26.7 baseline_replay、v7_exp2、small_weight、tight_toproi 的区别

#### baseline_replay

配置入口：

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml`

当前整理后的语义：

```text
baseline_replay -> v2 风格正式对照基线
```

特点：

- 已经不是原始 RT-DETRv4，而是正式 point baseline。
- 含 grape bbox、has_picking、point offset 训练链条。
- 不包含 v7_exp2 的 `query_box_top` 和 Top Local ROI 完整增强。
- 作为论文正式对照基线，而不是旧 report 中历史 point_v2 的随意结果。

#### v7_exp2 / GPPoint-DETR

配置入口：

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml`
- 当前主线语义别名通常也可对应 `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`

特点：

- 继承 top-center 表达。
- 使用 `query_box_top` 实例绑定。
- 使用 Top Local ROI 上部局部特征。
- 是本文主模型 GPPoint-DETR。
- 主要收益体现在 `has_picking F1`、`pair_count`、`mean L2` 和 `|dy|`。
- AP 有 seed 敏感性，不宜作为最稳定主结论。

#### small_weight

配置入口：

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml`
- 其内部语义对应 `v7_exp2_small_weight`。

特点：

- 在 v7_exp2 基础上，只对 small grape 的 point offset loss 加权。
- `point_small_grape_area_threshold: 0.07510009765625`
- `point_small_grape_weight: 1.5`
- 目的是验证 small grape 困难场景能否定向改善。
- 它是辅助补强实验，不替代主模型。

#### tight_toproi

配置原文件已在清理后不作为主配置保留，但负结果总结保存在：

- `outputs/history_suites/negative_variants/v7_exp2_tight_toproi/summary_zh.md`

特点：

- 在 v7_exp2 基础上进一步收紧 Top Local ROI。
- 曾使用更窄更靠上的窗口，例如：
  - `point_top_local_width_scale: 0.94`
  - `point_top_local_y_min_ratio: -0.14`
  - `point_top_local_y_max_ratio: 0.26`
- 结果没有带来主指标收益。
- 论文中可作为负例说明：Top Local ROI 不是越窄越好，过度收紧会丢失必要上下文。

### 26.8 文本版结构图：原始 RT-DETRv4 与 GPPoint-DETR 差异

原始 RT-DETRv4：

```text
Input image
  |
  v
Backbone
  |
  v
Hybrid Encoder
  |
  v
Transformer Decoder Queries
  |
  +--> Class Head ---------------> category / score
  |
  +--> Box Head -----------------> grape bbox
```

早期 picking bbox 路线：

```text
Input image
  |
  v
RT-DETRv4 Detector
  |
  +--> grape bbox
  |
  +--> picking bbox

问题：
  picking 被当作独立小目标，容易出现重复框、漏框、邻串误关联；
  picking bbox 与 grape 实例之间没有天然绑定关系。
```

GPPoint-DETR：

```text
Input image
  |
  v
Backbone
  |
  v
Hybrid Encoder
  |
  v
Transformer Decoder Queries
  |
  +--> Class Head ----------------------> grape category / score
  |
  +--> Box Head ------------------------> grape bbox
  |
  +--> Query-box / top-center geometry
  |          |
  |          v
  |     query_box_top feature
  |
  +--> Top Local ROI feature
  |          |
  |          v
  +--> Point Feature Fusion
             |
             +--> has_picking Head ----> visible picking point or not
             |
             +--> point_offset Head ---> offset relative to top-center
                                         |
                                         v
                                  decode with grape bbox
                                         |
                                         v
                                  picking_point (u, v)
```

一句话区别：

> 原始 RT-DETRv4 是“检测框输出模型”，GPPoint-DETR 是“葡萄串检测 + 实例级可见采摘点定位模型”。它没有把 picking 作为独立类别重新检测，而是把 picking point 绑定到每个 grape query 上。

### 26.9 每个关键模块对应文件路径和大致行号

| 模块 | 文件路径 | 关键行号 | 作用 |
|---|---|---:|---|
| 训练入口 | `train.py` | 28, 37, 41, 49, 51-54 | 读取配置、创建 solver、启动训练或验证 |
| 配置解析 | `engine/core/yaml_config.py` | 44-47, 82-91, 138-143 | 构建 model、postprocessor、criterion、evaluator |
| solver 初始化 | `engine/solver/_solver.py` | 65-70, 100-104 | 将模块放入 device，构建验证 dataloader/evaluator |
| 训练循环 | `engine/solver/det_engine.py` | 77-91, 112-114 | 模型前向和 loss 计算 |
| 验证循环 | `engine/solver/det_engine.py` | 184-196, 211-217 | 后处理、evaluator 更新、指标汇总 |
| 原始模型外壳 | `engine/rtv4/rtv4.py` | 13, 27-45 | backbone -> encoder -> decoder 主链路 |
| COCO 数据读取 | `engine/data/dataset/coco_dataset.py` | 151-180, 182-209 | 读取 has_picking、picking_points、picking_offsets 并过滤无效框 |
| 葡萄点数据集 | `engine/data/dataset/grape_point_dataset.py` | 32-45, 81-87, 100-125 | 重新生成 top-center offset，补齐 point 字段 |
| 点几何工具 | `engine/rtv4/point_utils.py` | 30-44, 66-82, 95-111 | top-center anchor、offset 与绝对点互转 |
| decoder 点分支 | `engine/rtv4/dfine_decoder.py` | 491, 605-638, 907-934, 1286-1287 | 定义并输出 has_picking 与 point_offset |
| query_box_top | `engine/rtv4/dfine_decoder.py` | 496-506, 663-681, 873-905 | 将 query box/top 几何信息融合进 point feature |
| Top Local ROI | `engine/rtv4/dfine_decoder.py` | 515-517, 829-871, 893-895 | 构造上部 ROI，roi_align 提取局部特征 |
| matcher | `engine/rtv4/matcher.py` | 52-117 | 仍按类别和 bbox 做 Hungarian 匹配，不把 point 作为匹配代价 |
| point/visibility loss | `engine/rtv4/rtv4_criterion.py` | 323-338, 340-360, 445-473, 577-593 | 计算 has_picking BCE、point offset loss、Wing loss、small weight |
| 后处理 | `engine/rtv4/postprocessor.py` | 112-155, 179-187 | 选 top-k query，解码 picking_point，输出 has_picking |
| 评价器 | `engine/data/dataset/coco_eval.py` | 287-356, 392-471 | COCO AP + picking F1、pair_count、dx/dy/L2 |
| baseline 配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml` | 全文件 | 正式 baseline_replay 入口 |
| 主模型配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` | 全文件 | GPPoint-DETR 主模型入口 |
| small 补强配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml` | 全文件 | small grape loss 加权辅助实验入口 |

### 26.10 最适合答辩时使用的总结

如果老师追问“你不是只加了几个头吗”，可以这样回答：

> 从结构上看，GPPoint-DETR 确实保持了 RT-DETRv4 的 backbone、encoder 和 decoder 主体，这是为了保证单阶段检测效率和可比性。但本文的关键不是换 backbone，而是把原始检测 query 的输出语义从“类别 + 框”扩展为“葡萄串框 + 可见采摘点属性 + 采摘点坐标”。为此，代码上同时改了数据标签读取、decoder 分支、top-center 点表达、query_box_top 实例绑定、Top Local ROI 特征融合、loss、postprocessor 和 evaluator。也就是说，它不是只在末尾加一个回归头，而是把采摘点作为 grape 实例属性贯穿了训练、推理和评价闭环。

如果老师追问“为什么不直接检测 picking bbox”，可以这样回答：

> 因为 picking 本质上不是与葡萄串平级的独立目标，而是依附于某个葡萄串实例的操作点。直接检测 picking bbox 会带来小目标不稳定、重复框、漏框以及与 grape bbox 关联困难等问题。GPPoint-DETR 改为每个 grape query 预测 `has_picking` 和 `point_offset`，使采摘点天然绑定到葡萄串实例上，更符合采摘任务的输出需求。

如果老师追问“为什么 AP 没明显提升还说有效”，可以这样回答：

> AP 衡量的是葡萄串框检测质量，而本文主要解决的是检测到葡萄串之后采摘点是否可见以及点在哪里的问题。GPPoint-DETR 的稳定收益主要体现在 `has_picking F1`、有效点配对数量、`mean L2` 和 `|dy|`。因此，本文不能夸大为葡萄框 AP 全面提升，而应表述为实例级采摘点定位链条得到增强。

---

## 27. 审稿人视角：代码与论文表述一致性审查

本节核对对象：

- 论文文件：`docs/基于RT-DETRv4的葡萄串识别与采摘点定位方法研究_修订版_v21_GPPoint-DETR.docx`
- 答辩汇总：`docs/GPPoint-DETR_答辩准备汇总版.md`
- 代码与配置：`train.py`、`engine/rtv4/*`、`engine/data/dataset/*`、`configs/rtv4/*`

### 27.1 总体结论

总体一致。论文中关于 GPPoint-DETR 的核心说法与当前代码基本匹配：

- Backbone、Hybrid Encoder/FPN 和 Transformer Decoder 主体没有被替换。
- `has_picking` 与 `point_offset` 确实是围绕 decoder query 的 per-query 输出。
- 主模型 `v7_exp2 / GPPoint-DETR` 确实采用 `top-center + query_box_top + Top Local ROI`。
- `point_offset` 在主模型中确实是相对 top-center 锚点、按 bbox 宽高归一化的二维偏移。
- `has_picking` 阈值为 0.5。
- y 方向点损失权重大于 x 方向。
- `small_weight` 只是辅助补强实验，不是主模型默认设置。
- postprocessor 会把 offset 解码为图像坐标系下的 `picking_points`。
- evaluator 确实计算 `has_picking F1`、`pair_count`、`mean L2` 和 `|dy|` 对应指标。

需要更严谨的一处表述：

> 论文中“当 has_picking 概率超过阈值时，使用 point_offset 解码采摘点坐标”的说法，从语义上可以理解，但从代码顺序看不完全精确。当前 postprocessor 实际上会先为所有保留的 grape query 解码候选 `picking_points`，再根据 `has_picking_scores >= 0.5` 标记哪些点是有效可见采摘点；evaluator 只把 GT 可见且预测可见的匹配样本计入点误差统计。

建议改论文表述，而不是改代码：

> 推理阶段，postprocessor 为每个保留的 grape query 根据预测框和 `point_offset` 解码候选二维采摘点，同时将 `has_picking` logit 经 sigmoid 得到可见性概率；当该概率超过阈值 0.5 时，该候选点被视为有效采摘点，并参与后续点误差统计或下游使用。

修改风险：

- 改论文表述风险低，只是让文字更贴近实现。
- 改代码风险高，因为若改为“只有超过阈值才解码点”，可能影响可视化、调试、统一输出字段和现有评估脚本，且不会带来模型性能收益。

### 27.2 逐项核对

| 核查项 | 结论 | 代码依据 | 建议 |
|---|---|---|---|
| 1. backbone、encoder、decoder 主干基本不变 | 属实 | `engine/rtv4/rtv4.py:27-45` 仍是 backbone -> encoder -> decoder；配置继承 `dfine_hgnetv2_s_coco.yml` 和 `base/rtv4.yml` | 论文可继续使用“主体框架基本不变”，但不要说 decoder 文件完全没改 |
| 2. `has_picking` 和 `point_offset` 是 per-query 输出 | 属实 | `engine/rtv4/dfine_decoder.py:1192-1228` 对每层 query 输出；`1286-1287` 写入 final outputs | 可表述为“基于 decoder query 的实例级输出分支” |
| 3. `point_offset` 是相对 top-center 的归一化偏移 | 属实 | `point_utils.py:66-82` 解码；`95-111` 编码；`v7_exp1.yml` 设置 `point_offset_mode: top_center` 和 `point_top_anchor_ratio: 0.12` | 保持当前公式 |
| 4. top-center 比例、Top Local ROI 范围、`query_box_top` 与配置一致 | 属实 | `v7_exp1.yml` 中 `0.12`；`v7_exp2.yml` 中 `query_box_top`、`1.08/-0.10/0.40` | 保持当前参数描述 |
| 5. y 方向 loss 权重是否 `alpha_y > alpha_x` | 属实 | `v7_exp1.yml` 中 `point_coord_weight_x: 1.0`、`point_coord_weight_y: 1.30`；criterion 中按坐标乘权重 | 可写“对 y 方向误差给予更高权重” |
| 6. `has_picking` 阈值是否 0.5 | 属实 | `PostProcessor.has_picking_threshold=0.5`；`GrapePointEvaluator(..., has_picking_threshold=0.5)` | 保持 |
| 7. `small_weight` 是否只是辅助实验 | 属实 | 主模型入口 `grape_point_main.yml` include `v7_exp2.yml`；`small_grape.yml` include `v7_exp2_small_weight.yml` | 论文不要把 small_weight 写成最终主模型 |
| 8. postprocessor 是否输出图像坐标 `picking_point` | 属实，但字段名为复数 | `postprocessor.py:146-152` 解码并 clamp；`179-187` 输出字段 `picking_points` | 论文写“picking point 坐标”可以，代码字段名应写 `picking_points` |
| 9. evaluator 是否计算 F1、pair_count、mean L2、`|dy|` | 属实 | `coco_eval.py:338-354` 汇总 F1、`point_pair_count`、`point_mean_l2_px`、`point_mae_y_px` | 论文中的 `|dy|` 对应代码 `point_mae_y_px` |

### 27.3 审稿人最可能追问的表述边界

#### “decoder 主干基本不变”怎么说才稳

可以说：

> GPPoint-DETR 保持 RT-DETRv4 的 backbone、encoder 和 transformer decoder 主体结构不变，新增逻辑主要位于 decoder query 输出后的实例级采摘点分支以及配套的损失、后处理和评价流程。

不要说：

> decoder 完全没有改。

原因：

- `engine/rtv4/dfine_decoder.py` 这个文件确实加入了 point head、Top Local ROI 和 query-box feature fusion。
- 但 transformer decoder layer、backbone 和 encoder 主体并未替换。

#### “decoder 后增加分支”怎么说才稳

可以说：

> `has_picking` 与 `point_offset` 使用 decoder 输出的 per-query hidden feature 和对应预测框作为输入，在每个 grape query 上生成实例级可见性与点偏移输出。

更严谨原因：

- final inference 的 `pred_has_picking` 和 `pred_picking_offsets` 来自 decoder 层输出。
- 训练阶段还包含 pre/aux/DN 相关输出用于辅助监督，因此不要把实现说成完全独立于 decoder 内部。

#### “使用阈值后再解码点”怎么改

论文中建议统一改成：

> postprocessor 会为保留 query 解码候选 picking point，并通过 `has_picking` 阈值筛选有效采摘点；评价时只有 GT 可见且预测可见的匹配实例计入点误差。

这样最贴合代码，也避免审稿人抓“代码先解码还是先阈值”的细节。

---

## 28. 代码可读性补强记录：只加注释，不改行为

本次修改原则：

- 不改变模型实际行为。
- 不改动核心训练逻辑。
- 不影响已有实验结果。
- 只增加必要注释、函数说明和变量解释。
- 注释重点围绕论文中会被问到的 `has_picking`、`point_offset`、`top-center`、`query_box_top`、Top Local ROI、y 方向加权和 picking point 评价指标。

### 28.1 修改文件与内容

| 文件 | 修改内容 | 行为影响 |
|---|---|---|
| `engine/rtv4/dfine_decoder.py` | 补充 per-query picking heads、`query_box_top` feature fusion、Top Local ROI 构造、final point outputs 的注释和 docstring | 无 |
| `engine/rtv4/point_utils.py` | 补充 top-center anchor、归一化 offset 编码、offset 解码为图像坐标的 docstring | 无 |
| `engine/rtv4/rtv4_criterion.py` | 补充 `loss_has_picking`、`loss_picking_offset`、只对可见点计算 loss、y 方向加权、small_weight 辅助实验的注释 | 无 |
| `engine/rtv4/postprocessor.py` | 补充 top-k query 对齐、candidate picking point 解码、0.5 阈值筛选有效点的注释 | 无 |
| `engine/data/dataset/coco_eval.py` | 补充 `has_picking F1`、`point_pair_count`、`mean L2`、`|dy|` 统计逻辑注释 | 无 |

### 28.2 diff 摘要

```text
engine/data/dataset/coco_eval.py |  7 +++++++
engine/rtv4/dfine_decoder.py     | 19 +++++++++++++++++++
engine/rtv4/point_utils.py       | 13 +++++++++++++
engine/rtv4/postprocessor.py     |  7 +++++++
engine/rtv4/rtv4_criterion.py    | 12 ++++++++++++
5 files changed, 58 insertions(+)
```

### 28.3 语法检查

已用 `ast.parse` 对以下文件做只读语法检查，结果均为 OK：

- `engine/rtv4/dfine_decoder.py`
- `engine/rtv4/point_utils.py`
- `engine/rtv4/rtv4_criterion.py`
- `engine/rtv4/postprocessor.py`
- `engine/data/dataset/coco_eval.py`

### 28.4 需要同步到论文表述的一句话

论文方法章节中建议把“当 has_picking 概率超过阈值时，使用 point_offset 解码采摘点坐标”改得更严谨：

> postprocessor 会为每个保留的 grape query 根据预测框和 `point_offset` 解码候选二维采摘点，同时将 `has_picking` logit 经 sigmoid 得到可见性概率；当该概率超过阈值 0.5 时，该候选点被视为有效采摘点，并参与后续点误差统计或下游使用。

---

## 29. 方法章与答辩用正式模型报告

本节用于写论文第 3 章“方法”以及答辩时解释模型结构。语言尽量保持正式、清楚，并把每个概念和代码实现对应起来。

### 29.1 项目任务定义

#### 原始任务定义

早期任务被建模为两个并列检测目标：

```text
grape bbox detection + picking bbox detection
```

其中 `grape` 表示葡萄串检测框，`picking` 表示采摘点附近的小目标框。该设定便于直接复用通用目标检测器，但存在一个关键问题：采摘点不是与葡萄串平级的独立物体，而是依附于某个葡萄串实例的操作点。若将 picking 作为独立小框检测，模型需要额外解决 picking 框与 grape 框之间的实例归属问题。

#### 当前任务定义

当前论文主线将任务重构为：

```text
grape bbox detection + per-grape has_picking + per-grape point localization
```

也就是说，模型对每个 grape query 输出三类信息：

```text
(bbox_i, score_i, has_picking_i, picking_point_i)
```

其中：

- `bbox_i` 是第 i 个葡萄串实例的边界框。
- `score_i` 是 grape 检测置信度。
- `has_picking_i` 是该葡萄串是否存在可见采摘点的二分类结果。
- `picking_point_i` 是图像坐标系下的二维采摘点。

这种定义把采摘点从独立检测目标改为 grape 实例的属性与点坐标输出，更符合“先识别葡萄串，再判断能否采摘，并给出采摘点位置”的作业逻辑。

### 29.2 原始 RT-DETRv4 结构

原始 RT-DETRv4 可以概括为一个端到端单阶段检测框架：

```text
Input image
  -> Backbone
  -> Hybrid Encoder / FPN
  -> Transformer Decoder
  -> Classification Head
  -> BBox Head
  -> PostProcessor
  -> labels / scores / boxes
```

各模块作用如下：

| 模块 | 作用 |
|---|---|
| Backbone | 提取输入图像的多尺度视觉特征 |
| Hybrid Encoder / FPN | 融合多尺度特征，增强检测所需的上下文表达 |
| Transformer Decoder | 使用 object queries 形成实例级检测表示 |
| Classification Head | 对每个 query 输出目标类别分数 |
| BBox Head | 对每个 query 输出目标边界框 |
| PostProcessor | 将网络输出转换为最终的检测框、类别和置信度 |

原始结构的输出重点是目标类别与边界框，并不直接包含采摘点可见性和采摘点坐标。因此，若不改任务头，原始 RT-DETRv4 只能通过额外的 picking 类别框来间接表示采摘点。

### 29.3 GPPoint-DETR 的结构改进

GPPoint-DETR 保持 RT-DETRv4 的 backbone、Hybrid Encoder/FPN 和 Transformer Decoder 主体结构基本不变，改动集中在 decoder query 输出后的实例级采摘点链条：

```text
Input image
  -> Backbone
  -> Hybrid Encoder / FPN
  -> Transformer Decoder
  -> Classification Head -> grape class / score
  -> BBox Head -> grape bbox
  -> Point Feature Fusion
       -> has_picking branch
       -> point_offset branch
  -> PostProcessor
       -> has_picking probability
       -> picking_point image coordinate
```

#### has_picking branch

`has_picking branch` 是每个 grape query 上的二分类分支，用于判断该葡萄串是否存在可见采摘点。它解决的是“该串葡萄有没有可用采摘点”的问题，避免模型对所有葡萄串都强行输出有效采摘点。

#### point_offset branch

`point_offset branch` 是每个 grape query 上的二维回归分支，用于预测采摘点相对于参考点的归一化偏移。它解决的是“采摘点在哪里”的问题。最终的采摘点不是独立检测框，而是由 grape bbox 与 offset 共同解码得到。

#### top-center coordinate representation

采摘点通常位于葡萄串上部或果梗附近。若从 bbox 中心回归采摘点，模型需要学习较大的向上偏移，容易造成 y 方向误差。GPPoint-DETR 将参考点移动到葡萄框上部，即 `top-center`，从而缩短纵向回归距离。

#### query_box_top binding

`query_box_top` 是实例绑定方式。它将当前 decoder query 的实例表示、预测框几何信息和 top-center 位置编码引入点分支，使采摘点预测与当前 grape query 绑定，降低相邻葡萄串之间的错关联风险。

#### Top Local ROI

Top Local ROI 从预测 grape 框上部区域提取局部视觉特征，用于给点分支提供果梗附近的局部 cue。它不是 two-stage ROI detector，因为它不生成新的候选框，也不独立执行 ROI 分类或 bbox 回归，只是为当前 query 的采摘点分支补充上部局部特征。

### 29.4 公式说明

#### top-center 参考点

设第 i 个预测 grape bbox 为：

```text
b_i = (x_i, y_i, w_i, h_i)
```

其中 `x_i, y_i` 表示 bbox 左上角坐标，`w_i, h_i` 表示宽和高。top-center 参考点定义为：

```text
a_i = (x_i + 0.5 * w_i, y_i + rho * h_i)
```

主模型中：

```text
rho = point_top_anchor_ratio = 0.12
```

因此，top-center 并不是严格位于 bbox 上边界，而是位于距离上边界 `0.12h_i` 的上部中心位置。

#### point_offset 解码公式

点分支预测归一化偏移：

```text
o_i = (o_x, o_y)
```

最终图像坐标系下的采摘点为：

```text
p_i = (u_i, v_i)
u_i = a_x + o_x * w_i
v_i = a_y + o_y * h_i
```

其中 `a_i=(a_x,a_y)` 是 top-center 参考点。该公式说明 `point_offset` 是相对于 grape bbox 尺度归一化后的二维偏移，最终通过 bbox 尺寸恢复为图像像素坐标。

#### 总损失

GPPoint-DETR 的训练损失可写为：

```text
L = L_det + lambda_has * L_has + lambda_pt * L_pt
```

其中：

- `L_det` 是原有 grape 检测损失，包括分类、bbox、GIoU 等检测相关损失。
- `L_has` 是 `has_picking` 可见性判别损失。
- `L_pt` 是采摘点 offset 回归损失。
- `lambda_has` 和 `lambda_pt` 是对应损失权重。

#### has_picking loss

对于匹配到的 grape query，`has_picking` 使用二元交叉熵损失：

```text
L_has = BCEWithLogits(z_i, y_i)
```

其中 `z_i` 是模型输出的 has_picking logit，`y_i` 是真实可见性标签，取值为 0 或 1。

#### point loss

点损失只对真实存在可见采摘点的实例计算：

```text
L_pt = (1 / N_v) * sum_i [ alpha_x * rho(o_x_i - o_x_i*) + alpha_y * rho(o_y_i - o_y_i*) ]
```

其中：

- `N_v` 是可见采摘点样本数量。
- `o_i` 是预测 offset。
- `o_i*` 是真实 offset。
- `rho(.)` 是 Wing Loss。
- 主模型中 `alpha_x=1.0`，`alpha_y=1.30`，因此 y 方向误差权重大于 x 方向。

Wing Loss 单维形式为：

```text
rho(Delta) =
  omega * ln(1 + |Delta| / epsilon), if |Delta| < omega
  |Delta| - C, otherwise
```

主模型中：

```text
omega = 0.25
epsilon = 0.05
```

#### mean L2

对有效点配对样本，预测点与真实点之间的欧氏距离为：

```text
d_i = sqrt((u_i - u_i*)^2 + (v_i - v_i*)^2)
```

mean L2 定义为：

```text
mean L2 = (1 / N_pair) * sum_i d_i
```

其中 `N_pair` 即 `pair_count`，表示真实可见且预测也判为可见的有效点配对数量。

#### |dy|

纵向绝对误差定义为：

```text
|dy| = (1 / N_pair) * sum_i |v_i - v_i*|
```

该指标用于衡量采摘点在图像 y 方向的漂移程度，是本文实验中最关键的点定位误差指标之一。

### 29.5 各模块解决的问题

| 模块 | 解决的问题 | 论文中建议表述 |
|---|---|---|
| `has_picking` | 解决“是否存在可见采摘点”的判断问题，避免对不可见或不可采样本强行输出有效点 | 可见性判别分支 |
| `point_offset` | 解决“采摘点在哪里”的坐标回归问题，并将点绑定到 grape 实例 | 实例级点定位分支 |
| `top-center` | 解决从 bbox 中心回归采摘点时 y 偏移过大的问题 | 上部几何先验 |
| `query_box_top` | 解决采摘点与具体 grape query 之间的实例归属问题 | 实例绑定增强 |
| Top Local ROI | 解决全局/整框特征中果粒、叶片和背景干扰较多的问题 | 上部局部视觉 cue |

### 29.6 与原始 picking bbox 路线的区别

#### 为什么不再把 picking 当成独立 bbox

将 picking 当成独立 bbox 存在以下问题：

- picking 区域通常很小，边界不稳定，小目标检测难度高。
- 同一葡萄串附近可能产生多个 picking 响应，造成重复框。
- 相邻葡萄串重叠时，picking 框容易与错误的 grape 框关联。
- picking bbox 的 AP 并不能直接说明“每个葡萄串是否有可用采摘点”。
- 后处理仍需要额外规则把 picking 框重新分配给 grape 实例。

因此，picking bbox 路线不是最适合当前任务的主线。

#### 为什么 per-grape has_picking + point localization 更合理

当前任务定义更合理的原因是：

- 采摘点本质上是 grape 实例的操作点，而不是独立物体。
- 每个 grape query 天然对应一个葡萄串实例，便于绑定可见性和点坐标。
- `has_picking` 可以显式处理“无可见采摘点”的样本。
- `point_offset` 可以直接给出二维采摘点，而不是依赖小框中心间接表示。
- 评价指标可以围绕采摘链条展开，包括 F1、pair_count、mean L2 和 `|dy|`。

一句话总结：

> picking bbox 是“把采摘点当成小目标去找”，GPPoint-DETR 是“围绕每个葡萄串实例判断能不能采、从哪里采”。后者更符合葡萄采摘任务的实例级输出需求。

### 29.7 代码对应位置

| 模块 | 文件路径 | 大致行号 | 说明 |
|---|---|---:|---|
| 训练入口 | `train.py` | 28-54 | 读取配置、创建 solver、启动训练或验证 |
| 主模型外壳 | `engine/rtv4/rtv4.py` | 13-45 | 保持 backbone -> encoder -> decoder 主流程 |
| 数据字段读取 | `engine/data/dataset/coco_dataset.py` | 151-180, 204-209 | 读取 `has_picking`、`picking_points`、`picking_offsets` |
| 点标签生成 | `engine/data/dataset/grape_point_dataset.py` | 32-45, 81-87 | 根据 bbox 与 point 生成 top-center offset |
| top-center 编解码 | `engine/rtv4/point_utils.py` | 30-44, 66-82, 95-111 | 计算 top-center 锚点，完成 offset 与 point 互转 |
| decoder 点分支定义 | `engine/rtv4/dfine_decoder.py` | 449-530, 605-638 | 配置并定义 `has_picking` 与 `point_offset` heads |
| Top Local ROI | `engine/rtv4/dfine_decoder.py` | 829-871 | 构造上部 ROI 并用 `roi_align` 提取局部特征 |
| query_box_top 融合 | `engine/rtv4/dfine_decoder.py` | 873-934 | 融合 query hidden、box geometry、query position 和 top-local feature |
| final per-query 输出 | `engine/rtv4/dfine_decoder.py` | 1192-1287 | 输出 `pred_has_picking` 和 `pred_picking_offsets` |
| 匹配器 | `engine/rtv4/matcher.py` | 52-117 | 仍按 grape 类别和 bbox 做 Hungarian 匹配 |
| has_picking loss | `engine/rtv4/rtv4_criterion.py` | 323-338 | 对匹配 query 计算 BCE loss |
| point_offset loss | `engine/rtv4/rtv4_criterion.py` | 445-473 | 仅对可见采摘点计算 offset loss，并支持 y 加权和 small 权重 |
| Wing Loss | `engine/rtv4/rtv4_criterion.py` | 577-593 | 点回归的基础损失函数 |
| 后处理 | `engine/rtv4/postprocessor.py` | 112-187 | top-k query 对齐、解码 `picking_points`、输出 `has_picking` |
| 评价器 | `engine/data/dataset/coco_eval.py` | 287-356, 392-471 | 计算 `has_picking F1`、`pair_count`、mean L2、`|dy|` |
| baseline 配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml` | 全文件 | 正式 baseline 入口 |
| GPPoint-DETR 配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml` | 全文件 | 主模型历史版本入口 |
| 主模型语义入口 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml` | 全文件 | 当前命名后的 GPPoint-DETR 入口 |
| small_weight 配置 | `configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml` | 全文件 | small grape 辅助实验入口 |

### 29.8 方法章最稳妥表述

可直接用于论文方法章的总结：

> 本文提出的 GPPoint-DETR 在 RT-DETRv4 单阶段检测框架基础上，将采摘点从独立小目标检测重构为 grape 实例上的可见性判别与点定位问题。模型保留原始 RT-DETRv4 的 backbone、Hybrid Encoder/FPN 和 Transformer Decoder 主体结构，在每个 decoder query 上增加 `has_picking` 分支和 `point_offset` 分支。其中，`has_picking` 用于判断当前葡萄串是否存在可见采摘点，`point_offset` 用于预测采摘点相对于葡萄框 top-center 参考点的归一化偏移。为增强点预测与葡萄串实例之间的绑定关系，模型进一步引入 `query_box_top` 几何绑定和 Top Local ROI 上部局部特征，使采摘点预测同时受到 query 实例表示、预测框几何位置和果梗附近局部视觉线索约束。最终，postprocessor 将 point offset 解码为图像坐标系下的二维 picking point，并通过 `has_picking` 阈值筛选有效采摘点。
