param(
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [string]$OutputDir = "outputs\03_global_analysis\ema_bifpn_hp_pick_protocol_has062_20260531",
    [string]$SelectionOutputDir = "outputs\03_global_analysis\selection_calibration_picking_first_20260531",
    [switch]$RefreshRecords,
    [switch]$FullReport
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ProtocolConfig = Join-Path $RepoRoot "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_hp_pick_protocol.yml"
$BaseRunDir = Join-Path $RepoRoot "outputs\02_encoder_experiments\encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526"
$MainSummary = Join-Path $RepoRoot "outputs\03_global_analysis\post_cleanup_v7_exp2_report_20260525\summary.json"
$BaseSummary = Join-Path $BaseRunDir "report\summary.json"
$SelectionDirAbs = Join-Path $RepoRoot $SelectionOutputDir
$OutputDirAbs = Join-Path $RepoRoot $OutputDir

if (!(Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}
if (!(Test-Path -LiteralPath $ProtocolConfig)) {
    throw "Protocol config not found: $ProtocolConfig"
}
if (!(Test-Path -LiteralPath $BaseRunDir)) {
    throw "EMA_BIFPN run dir not found: $BaseRunDir"
}
if (!(Test-Path -LiteralPath $MainSummary)) {
    throw "Main reference summary not found: $MainSummary"
}

New-Item -ItemType Directory -Path $OutputDirAbs -Force | Out-Null

Write-Host "===== EMA_BIFPN HP_PICK_PROTOCOL_HAS062 =====" -ForegroundColor Cyan
Write-Host "Step 1/3: selection calibration summary"

$SelectionArgs = @(
    (Join-Path $RepoRoot "tools\evaluate_selection_reassignment.py"),
    "--output-dir", $SelectionDirAbs,
    "--thresholds", "0.46:0.70:0.02",
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers
)
if ($RefreshRecords) {
    $SelectionArgs += "--refresh-records"
}
& $Python @SelectionArgs
if ($LASTEXITCODE -ne 0) {
    throw "Selection calibration failed, exit code: $LASTEXITCODE"
}

$SelectionSummary = Join-Path $SelectionDirAbs "summary.json"
if (!(Test-Path -LiteralPath $SelectionSummary)) {
    throw "Selection summary not found: $SelectionSummary"
}

Write-Host "Step 2/3: picking-first gate"
& $Python (Join-Path $RepoRoot "tools\check_mainline_gate.py") `
    --candidate-summary $SelectionSummary `
    --reference-summary $MainSummary `
    --base-summary $BaseSummary `
    --candidate-label "EMA_BIFPN_HP_PICK_PROTOCOL_HAS062" `
    --reference-label "V7_EXP2_MAIN fair retrain" `
    --base-label "EMA_BIFPN default" `
    --min-ap 0.624 `
    --min-ap50 0.876 `
    --min-has-f1 0.7615 `
    --min-pair-count 185 `
    --max-mean-l2 23.40 `
    --min-ppl-sr30 0.780 `
    --min-ppl-sr50 0.895 `
    --output-dir $OutputDirAbs
if ($LASTEXITCODE -ne 0) {
    throw "Mainline gate failed to run, exit code: $LASTEXITCODE"
}

if ($FullReport) {
    Write-Host "Step 3/3: full model report with protocol config"
    & $Python (Join-Path $RepoRoot "tools\make_grape_point_report.py") `
        --run-dir $BaseRunDir `
        --report-dir (Join-Path $OutputDirAbs "full_report") `
        --config $ProtocolConfig `
        --checkpoint (Join-Path $BaseRunDir "best_composite.pth") `
        --device $Device `
        --batch-size $BatchSize `
        --num-workers $NumWorkers `
        --point-v2-summary $MainSummary `
        --primary-label "EMA_BIFPN_HP_PICK_PROTOCOL_HAS062" `
        --reference-label "V7_EXP2_MAIN fair retrain" `
        --report-title "EMA_BIFPN high-precision picking protocol report" `
        --change-note "No training: EMA_BIFPN with has_picking_threshold=0.62 selected on valid." `
        --save-prediction-records
    if ($LASTEXITCODE -ne 0) {
        throw "Full protocol report failed, exit code: $LASTEXITCODE"
    }
}
else {
    Write-Host "Step 3/3: full model report skipped. Use -FullReport when a checkpoint-backed report is required."
}

Write-Host ""
Write-Host "Protocol complete. Main files:" -ForegroundColor Green
Write-Host (Join-Path $SelectionDirAbs "selection_reassignment_report_zh.md")
Write-Host (Join-Path $OutputDirAbs "mainline_gate_result_zh.md")
Write-Host $ProtocolConfig
