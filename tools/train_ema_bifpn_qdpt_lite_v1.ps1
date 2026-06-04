param(
  [ValidateSet("smoke", "probe20", "report", "decision", "all")]
  [string]$Mode = "all",
  [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$Name = "EMA_BIFPN_QDPT_LITE_V1_PROBE20"
$Python = ".\.venv\Scripts\python.exe"
$Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_qdpt_lite_v1.yml"
$OutputDir = "outputs\01_mainline_results\candidate_ema_bifpn_qdpt_lite_v1_probe20"
$ReportDir = Join-Path $OutputDir "report"
$SmokeDir = Join-Path $OutputDir "smoke"
$Checkpoint = "outputs\90_legacy_misc\encoder_experiments_archive\encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526\best_composite.pth"
$EmaSummary = "outputs\03_unified_evaluation\eval_unification\ema_bifpn_unified_report\summary.json"
$EmaRecords = "outputs\03_unified_evaluation\eval_unification\ema_bifpn_unified_report\test_prediction_records.json"
$V7Summary = "outputs\03_unified_evaluation\eval_unification\v7_exp2_unified_report\summary.json"

function Run-Step {
  param(
    [string]$Label,
    [scriptblock]$Block
  )
  Write-Host "===== $Name $Label ====="
  & $Block
  if ($LASTEXITCODE -ne 0) {
    throw "$Name $Label failed, exit code: $LASTEXITCODE"
  }
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if ($Mode -in @("smoke", "all")) {
  Run-Step "smoke" {
    & $Python tools\check_qdpt_lite_smoke.py --config $Config --checkpoint $Checkpoint --output-dir $SmokeDir --device $Device
  }
}

if ($Mode -in @("probe20", "all")) {
  $smokeJson = Join-Path $SmokeDir "qdp_lite_smoke_report.json"
  if (-not (Test-Path -LiteralPath $smokeJson)) {
    throw "Smoke report missing: $smokeJson"
  }
  $smoke = Get-Content -LiteralPath $smokeJson -Raw | ConvertFrom-Json
  if (-not $smoke.passed) {
    throw "Smoke did not pass; refusing to train probe20."
  }
  Run-Step "train probe20" {
    & $Python train.py -c $Config -t $Checkpoint -d $Device --use-amp
  }
}

if ($Mode -in @("report", "all")) {
  $best = Join-Path $OutputDir "best_composite.pth"
  if (-not (Test-Path -LiteralPath $best)) {
    $best = Join-Path $OutputDir "best_stg1.pth"
  }
  if (-not (Test-Path -LiteralPath $best)) {
    throw "No report checkpoint found in $OutputDir"
  }
  New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
  Run-Step "unified report" {
    & $Python tools\make_grape_point_report.py `
      --run-dir $OutputDir `
      --report-dir $ReportDir `
      --config $Config `
      --checkpoint $best `
      --device $Device `
      --batch-size 8 `
      --num-workers 0 `
      --save-prediction-records `
      --primary-label $Name `
      --reference-label "V7_EXP2_MAIN fair retrain" `
      --point-v2-summary $V7Summary `
      --report-title "$Name unified report" `
      --change-note "QDPT-Lite offset-only residual point token with MIAT and box-to-point prior; matcher, has threshold, and postprocessor main logic unchanged."
  }
}

if ($Mode -in @("decision", "all")) {
  $candidateSummary = Join-Path $ReportDir "summary.json"
  $candidateRecords = Join-Path $ReportDir "test_prediction_records.json"
  if (-not (Test-Path -LiteralPath $candidateSummary)) {
    throw "Candidate summary missing: $candidateSummary"
  }
  if (-not (Test-Path -LiteralPath $candidateRecords)) {
    throw "Candidate test prediction records missing: $candidateRecords"
  }
  Run-Step "decision" {
    & $Python tools\check_qdpt_lite_decision.py `
      --candidate-summary $candidateSummary `
      --candidate-records $candidateRecords `
      --ema-summary $EmaSummary `
      --ema-records $EmaRecords `
      --output-dir $ReportDir `
      --label $Name
  }
}
