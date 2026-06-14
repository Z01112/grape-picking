# GPPoint-DETR Grape Picking Point Workspace

This repository contains the RT-DETRv4 based GPPoint-DETR experiments for grape detection plus instance-bound visible 2D picking point prediction.

The current task definition is:

- detect grape bunch boxes;
- predict whether each detected grape has a visible picking point;
- regress the visible 2D picking point for matched grape instances.

It is not a 3D robot picking coordinate system. Do not describe the 2D point output as a robot executable 3D pose without additional depth or control data.

## Current Data And Main Reference

The current working dataset is the default grape-point thesis dataset under:

- `datasets/train/_annotations.grape_point.json`
- `datasets/valid/_annotations.grape_point.json`
- `datasets/test/_annotations.grape_point.json`

Do not introduce `NEW1804` in new experiment names, output directories, scripts, reports, or thesis-facing text. Treat `NEW1804` as a legacy internal label that remains only in already-generated paths for reproducibility.

Older `dataset/` assets may still exist for historical experiments and the picking-bbox-to-point baseline. Keep old-dataset experiments separate from current `datasets/` experiments, and do not move or rename an active run while training is in progress.

The current main reference is:

- `outputs/01_mainline_results/ema_bifpn_new1804_fair100/report/summary.json`

Thesis-facing name for this result: `GPPoint-DETR / EMA_BIFPN current-dataset fair100`. The path still contains the historical `new1804` token for traceability; do not copy that token into new paper text.

Recent rejected candidates and diagnostics are retained under the output archive standard in:

- `outputs/README_OUTPUTS.md`

## Important Current Conclusions

- `GPPoint-DETR / EMA_BIFPN current-dataset fair100` remains the main reference for new comparisons.
- `YANCHOR_MEDIAN_FAIR100` is rejected as a mainline candidate: it improved AP/F1/pair on formal test but worsened mean L2, p90, PPL@30, L2>30, and mean_abs_dy.
- Old proposal docs may contain historical next-step ideas. Before starting any experiment, check the latest `summary.json`, decision markdown/json, and `docs/experiment_archive_summary.md`.

For new names, use:

- `current_dataset` or no dataset suffix for the default `datasets/` split;
- `olddata` only for the legacy `dataset/` split;
- `legacy` only for pre-current reports that are retained for traceability.

## Common Entry Points

Training scripts live in `tools/`. Most accepted scripts are expected to generate reports after training. Examples:

```powershell
powershell -ExecutionPolicy Bypass -File tools\train_current_rtdetrv4_gphead_baseline_fair100.ps1
powershell -ExecutionPolicy Bypass -File tools\train_current_ema_bifpn_fair100.ps1
powershell -ExecutionPolicy Bypass -File tools\train_olddata_picking_bbox_to_point_baseline_fair100.ps1
```

The `train_current_*` scripts are clean-name aliases for the current `datasets/` split. They intentionally preserve historical output paths where existing reports already use the old internal token.

Stopped or rejected routes such as `stem_aux_visibility`, `stem_spatial_guidance`, `YANCHOR_MEDIAN`, `PAM`, picking-aware rerank, and TopROI heatmap have their results retained under `outputs/` but their one-off training/check scripts are intentionally not kept as active entry points.

Manual report generation and comparison utilities are also under `tools/`, especially:

- `tools/make_grape_point_report.py`
- `tools/grape_point_eval_utils.py`
- `tools/build_eval_unification_artifacts.py`

## Output Placement Rule

Do not create new numbered folders directly under `outputs/` for ordinary experiments. Use:

- `outputs/01_mainline_results/` for paper mainline candidates and important references;
- `outputs/02_baselines/` for baselines;
- `outputs/03_unified_evaluation/` for shared evaluation tables;
- `outputs/04_diagnostics/` for audits and smoke checks;
- `outputs/05_failed_experiments/` for rejected model routes;
- `outputs/06_data_supervision/` for relabeling and annotation-support assets.

See `outputs/README_OUTPUTS.md` before adding new output paths.

## Documentation Notes

Project docs under `docs/` include historical proposals as well as current summaries. Treat files named as proposals or analyses as evidence snapshots, not automatically executable plans.

For model selection, prefer the newest report and decision files over older narrative docs.
