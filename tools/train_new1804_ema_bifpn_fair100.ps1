param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [switch]$SkipPrechecks,
    [switch]$SkipReport
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Config = Join-Path $RepoRoot "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_new1804_fair100.yml"
$RunDir = Join-Path $RepoRoot "outputs\01_mainline_results\ema_bifpn_new1804_fair100"
$ReportDir = Join-Path $RunDir "report"
$ReferenceSummary = Join-Path $RepoRoot "outputs\03_unified_evaluation\eval_unification\ema_bifpn_unified_report\summary.json"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Block
    )
    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed, exit code: $LASTEXITCODE"
    }
}

Set-Location $RepoRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python. Use .\.venv\Scripts\python.exe from the project venv."
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Config not found: $Config"
}

if (-not $SkipPrechecks) {
    Run-Step "NEW1804 dataset integrity check" {
        & $Python "tools\check_new1804_dataset_integrity.py"
    }

    Run-Step "NEW1804 training smoke check" {
        & $Python "tools\check_new1804_training_smoke.py"
    }
}

Run-Step "NEW1804 EMA_BIFPN fair100 train from HGNetv2 pretrained" {
    & $Python "train.py" `
        -c $Config `
        -d $Device `
        --use-amp
}

if (-not $SkipReport) {
    Run-Step "NEW1804 unified report" {
        & $Python "tools\make_grape_point_report.py" `
            --run-dir $RunDir `
            --report-dir $ReportDir `
            --config $Config `
            --dataset-root "datasets" `
            --device "cuda:0" `
            --batch-size 8 `
            --num-workers 0 `
            --point-v2-summary $ReferenceSummary `
            --reference-label "EMA_BIFPN old dataset" `
            --primary-label "EMA_BIFPN_NEW1804_FAIR100" `
            --report-title "NEW1804 EMA_BIFPN fair100 中文结论" `
            --change-note "新数据集 datasets/ 1804 images; EMA_BIFPN fair100 from HGNetv2 pretrained; no old checkpoint tuning; no failed branches enabled." `
            --save-prediction-records
    }

    Run-Step "NEW1804 decision table" {
        & $Python "tools\build_new1804_main_decision.py"
    }
}

Write-Host ""
Write-Host "Done. Report dir: $ReportDir" -ForegroundColor Green
