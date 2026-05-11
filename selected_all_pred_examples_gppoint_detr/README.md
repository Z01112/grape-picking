# GPPoint-DETR 全预测示例

本目录中的输出图来自 `GPPoint-DETR` 主模型：

- 配置：`configs/rtv4/rtv4_hgnetv2_s_grape_point_main.yml`
- checkpoint：`outputs/grape_point_gppoint_detr_main/best_composite.pth`

导出规则：

- 仅保留模型预测结果，不包含任何原始标注框
- 绿色框：预测葡萄串框
- 红色叉：预测采摘点
- 葡萄串框阈值：`score >= 0.50`
- 采摘点显示阈值：`has_picking_score >= 0.50`

每个 `case_xx` 文件夹都包含：

- `input_*.jpg`：原始输入图
- `output_all_pred_*.png`：对应输出图，包含该图中所有保留下来的预测葡萄串框与采摘点
