# RT-DETRv4 项目清理清单

生成日期：2026-05-16

本清单用于低风险整理当前 GPPoint-DETR / RT-DETRv4 项目目录。清理目标是让论文主线代码、正式实验输出和诊断报告更清晰，同时避免破坏现有结果复现。

## 清理原则

1. 不删除 `dataset`、`configs`、`engine`、`tools`、`reports`、`docs` 中的主线文件。
2. 不删除任何 `best*.pth`、`summary.json`、`comparison_report_zh.md`、`predictions/test_predictions.json`、`predictions/valid_predictions.json`。
3. 不修改模型代码逻辑，不启动训练。
4. 缓存和临时文件可直接删除；过期实验、旧报告包、失败输出和重复日志只移动到归档目录。
5. 归档目录为 `_archive_cleanup_20260516/`。

## A. 必须保留文件/目录

### 主线代码与数据

- `train.py`
- `requirements.txt`
- `configs/`
- `engine/`
- `tools/`
- `dataset/`
- `reports/`
- `docs/`
- `pretrain/`
- `VSCode_Guide.md`
- `LICENSE`

### 正式实验输出

以下目录或对应实际输出目录必须保留在 `outputs/` 下：

- `outputs/grape_point_gppoint_detr_main`
- `outputs/grape_point_v6_baseline_replay`
- `outputs/grape_point_gppoint_detr_small_weight`
- `outputs/grape_point_v7_exp2_taller_toproi`
- `outputs/grape_point_v7_exp2_dn_teacher_roi`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss`
- `outputs/grape_point_v7_exp2_decoupled_roi`
- `outputs/grape_point_v7_exp2_point_quality`
- `outputs/grape_point_v7_exp2_point_quality_sg`
- `outputs/grape_point_v7_exp2_median_anchor`

其中每个正式实验中的以下资产需要保留：

- 配置快照或相关配置文件
- `best_composite.pth`
- `summary.json`
- `comparison_report_zh.md`
- `results.csv`
- `predictions/test_predictions.json`
- `predictions/valid_predictions.json`
- `scene_slice_table.csv`
- ROI / threshold / calibration 相关报告和表格

### 正式配置文件

- `configs/rtv4/rtv4_hgnetv2_s_grape_point_baseline_replay.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_small_grape.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_taller_toproi.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_dn_teacher_roi.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_dn_teacher_roi_light_loss.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_decoupled_roi.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_point_quality.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_point_quality_sg.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_median_anchor.yml`
- `configs/rtv4/rtv4_hgnetv2_s_grape_point_v7_exp2_small_weight.yml`

## B. 可归档文件/目录

以下内容不直接删除，移动到 `_archive_cleanup_20260516/`：

### 过期或非主线实验输出

- `outputs/grape_point_v2_main`
- `outputs/grape_point_v7_exp1_query_box_top_center`
- `outputs/grape_point_v7_paper_ready`
- `outputs/grape_point_v7_sci_ready`
- `outputs/history_suites`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss2`

### 输出目录根部的临时日志 / 后台日志 / 训练探针日志

- `outputs/_bg_import_test_stderr.log`
- `outputs/_bg_import_test_stdout.log`
- `outputs/_bg_python_test_stderr.log`
- `outputs/_bg_python_test_stdout.log`
- `outputs/_bg_train_probe_stderr.log`
- `outputs/_bg_train_probe_stdout.log`
- `outputs/dn_teacher_multiseed_pipeline_stderr.log`
- `outputs/dn_teacher_multiseed_pipeline_stdout.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_resume_stderr.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_resume_stdout.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_resume.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_train_foreground.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_train_stderr.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_train_stdout.log`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_train.log`
- `outputs/grape_point_v7_exp2_taller_toproi_train_stderr.log`
- `outputs/grape_point_v7_exp2_taller_toproi_train_stdout.log`

## C. 可直接删除的缓存/临时文件

仅删除以下明显缓存或临时文件，并排除 `.git/`、`.venv/`、`_archive_cleanup_20260516/`：

- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `~$*.docx`
- `*.tmp`
- `Thumbs.db`
- `.DS_Store`

说明：`.venv/` 内部缓存暂不清理，避免影响当前 Python 环境。

## D. 不确定、需要人工确认的文件/目录

以下内容本轮不移动、不删除：

- `outputs/grape_point_gppoint_detr_main_repro1`
- `outputs/grape_point_gppoint_detr_main_seed2026`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_repro1`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_seed2026`
- `selected_all_pred_examples_gppoint_detr/`
- `.codex/`
- `.vscode/`
- `.venv/`
- `paper_assets/`：当前未确认存在。
- `tight_toproi`：当前未确认对应输出目录名称，需要人工确认是否已保存到其他目录。

