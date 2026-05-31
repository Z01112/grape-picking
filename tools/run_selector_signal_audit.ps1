param(
  [string]$OutputDir = "outputs\03_global_analysis\selector_signal_audit_20260531",
  [string]$ReportDir = "outputs\05_selector_experiments\ema_bifpn_detached_query_selector_v2_has_fair100_backbone_pretrain_20260530\report_best_point_l2"
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

& $Python tools\audit_selector_signal_from_records.py `
  --valid-records (Join-Path $ReportDir "valid_prediction_records.json") `
  --test-records (Join-Path $ReportDir "test_prediction_records.json") `
  --summary (Join-Path $ReportDir "summary.json") `
  --output-dir $OutputDir `
  --thresholds "0.30:0.80:0.02" `
  --alphas "0.25,0.5,0.75,1.0,1.5,2.0"

if ($LASTEXITCODE -ne 0) {
  throw "Selector signal audit failed, exit code: $LASTEXITCODE"
}

Write-Host "Selector signal audit report:"
Write-Host (Join-Path $OutputDir "selector_signal_audit_report.md")
