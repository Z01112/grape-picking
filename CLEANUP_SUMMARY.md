# RT-DETRv4 项目清理总结

生成日期：2026-05-16

## 本轮清理范围

本轮只做项目文件层面的低风险整理：

- 未修改模型代码逻辑。
- 未启动训练。
- 未删除 `dataset/`。
- 未删除任何正式实验的 `best*.pth`、`summary.json`、`comparison_report_zh.md`、`predictions/test_predictions.json`、`predictions/valid_predictions.json`。
- 过期实验和零散日志均移动到 `_archive_cleanup_20260516/`，没有直接删除。

## 已删除内容

仅删除了仓库内明显缓存目录，且排除了 `.git/`、`.venv/` 和 `_archive_cleanup_20260516/`：

- `__pycache__`
- `tools/__pycache__`
- `engine/__pycache__`
- `engine/backbone/__pycache__`
- `engine/core/__pycache__`
- `engine/data/__pycache__`
- `engine/data/dataset/__pycache__`
- `engine/data/transforms/__pycache__`
- `engine/misc/__pycache__`
- `engine/optim/__pycache__`
- `engine/rtv4/__pycache__`
- `engine/solver/__pycache__`

删除明细记录在：

- `_archive_cleanup_20260516/deleted_cache_items.txt`

## 已移动归档内容

以下过期或非主线实验输出已移动到 `_archive_cleanup_20260516/outputs/`：

- `outputs/grape_point_v2_main`
- `outputs/grape_point_v7_exp1_query_box_top_center`
- `outputs/grape_point_v7_paper_ready`
- `outputs/grape_point_v7_sci_ready`
- `outputs/history_suites`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_light_loss2`

以下输出目录根部的临时日志、后台日志和训练探针日志已移动到 `_archive_cleanup_20260516/outputs_root_logs/`：

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

移动明细记录在：

- `_archive_cleanup_20260516/moved_items.txt`

## 已保留内容

### 主线目录

- `dataset/`
- `configs/`
- `engine/`
- `tools/`
- `reports/`
- `docs/`
- `pretrain/`
- `outputs/`

### 正式实验输出

以下正式实验输出目录已检查，仍保留在 `outputs/`：

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

## 需要人工确认

以下内容本轮未移动、未删除：

- `outputs/grape_point_gppoint_detr_main_repro1`
- `outputs/grape_point_gppoint_detr_main_seed2026`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_repro1`
- `outputs/grape_point_v7_exp2_dn_teacher_roi_seed2026`
- `selected_all_pred_examples_gppoint_detr/`
- `.codex/`
- `.vscode/`
- `.venv/`
- `paper_assets/`：当前未确认存在。
- `tight_toproi`：当前未确认对应输出目录名称，需要确认是否保存在其他路径或仅存在于历史报告中。

这些目录可能包含多 seed 统计证据、人工筛图、IDE 设置或环境文件，建议在确认论文资产归属后再决定是否归档。

## 当前推荐目录结构

建议后续保持如下结构：

```text
RT-DETRv4/
  configs/
    rtv4/
      rtv4_hgnetv2_s_grape_point_v7_exp2.yml
      rtv4_hgnetv2_s_grape_point_v7_exp2_*.yml
  dataset/
  docs/
  engine/
  outputs/
    grape_point_gppoint_detr_main/
    grape_point_v6_baseline_replay/
    grape_point_gppoint_detr_small_weight/
    grape_point_v7_exp2_taller_toproi/
    grape_point_v7_exp2_dn_teacher_roi/
    grape_point_v7_exp2_dn_teacher_roi_light_loss/
    grape_point_v7_exp2_decoupled_roi/
    grape_point_v7_exp2_point_quality/
    grape_point_v7_exp2_point_quality_sg/
    grape_point_v7_exp2_median_anchor/
  reports/
  tools/
  pretrain/
  _archive_cleanup_20260516/
```

## 复查建议

1. 若论文最终需要多 seed 结果，请确认 `*_repro1` 和 `*_seed2026` 输出是否纳入正式资产。
2. 若 `tight_toproi` 是负结果但没有独立输出目录，建议在论文资产表中注明其结果来源文件。
3. 后续新增实验建议统一使用 `outputs/grape_point_v7_exp2_<experiment_name>/`，并在实验结束后保留 `summary.json`、`report/`、`predictions/` 和配置快照。

## 最终复查清理：2026-05-16

本轮再次扫描整个项目目录，排除 `.git/`、`.venv/` 和 `_archive_cleanup_20260516/` 后执行低风险清理。

### 直接删除的缓存/临时文件

- 删除重新生成的 `engine/**/__pycache__/`，共 9 个目录。
- 删除记录：`_archive_cleanup_20260516/final_review_20260516/deleted_cache_items.txt`
- 复查结果：`configs/`、`engine/`、`tools/`、`reports/`、`docs/`、`outputs/`、`dataset/` 下未发现残留 `__pycache__` 或 `.ipynb_checkpoints`。

### 已归档的旧日志

- 已将 `outputs/` 下训练、恢复、导出、launcher、stdout/stderr 等旧日志归档，共 38 个文件。
- 归档目录：`_archive_cleanup_20260516/final_review_20260516/logs/`
- 移动记录：`_archive_cleanup_20260516/final_review_20260516/moved_logs.txt`
- 复查结果：`outputs/` 下未发现残留 `.log` 或 stdout/stderr 临时日志。

### 已归档的重复/临时结果

- 已将 `outputs/` 下 `new_variant_*` 重复产物归档，共 13 个文件。
- 归档目录：`_archive_cleanup_20260516/final_review_20260516/duplicate_temp_results/`
- 移动记录：`_archive_cleanup_20260516/final_review_20260516/moved_duplicate_temp_results.txt`
- 复查结果：`outputs/` 下未发现残留 `new_variant_*` 文件。

### dead code 复查结论

本轮没有删除模型代码中的疑似 dead code，原因如下：

- `engine/rtv4/rtv4_criterion.py` 中的 dense / geo / locality legacy losses 已明确标注为历史实验代码，当前正式配置不调用，但直接删除会降低历史实验可追溯性。
- `tools/make_grape_point_*_report.py` 中存在若干旧报告入口脚本，它们是薄 wrapper，不影响主线，但可能仍用于旧报告复现。
- 未发现可以在不影响历史复现和论文资产索引的前提下安全删除的 Python 源码文件。

建议后续若要进一步清理 dead code，应单独开一次“代码层清理”任务，并先做：

- 全配置引用扫描；
- import / build 检查；
- baseline、v7_exp2、small_weight、median_anchor 的最小推理导出检查；
- 明确是否仍保留历史实验可复现入口。
