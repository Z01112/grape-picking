param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [switch]$SkipPrechecks,
    [switch]$SkipReport,
    [switch]$KeepPeriodicCheckpoints
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Config = Join-Path $RepoRoot "configs\rtv4\rtv4_hgnetv2_s_grape_point_rtdetrv4_new1804_baseline_fair100.yml"
$RunDir = Join-Path $RepoRoot "outputs\02_baselines\rtdetrv4_new1804_baseline_fair100"
$ReportDir = Join-Path $RunDir "report"
$ReferenceSummary = Join-Path $RepoRoot "outputs\01_mainline_results\ema_bifpn_new1804_fair100\report\summary.json"
$SmokeDir = Join-Path $RepoRoot "outputs\04_diagnostics\new1804_training_smoke\rtdetrv4_baseline"

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

    Run-Step "NEW1804 RT-DETRv4 baseline smoke check" {
        & $Python "tools\check_new1804_training_smoke.py" `
            --config $Config `
            --out-dir $SmokeDir
    }
}

Run-Step "NEW1804 RT-DETRv4 baseline fair100 train from HGNetv2 pretrained" {
    & $Python "train.py" `
        -c $Config `
        -d $Device `
        --use-amp
}

if (-not $SkipReport) {
    Run-Step "NEW1804 RT-DETRv4 baseline unified report" {
        & $Python "tools\make_grape_point_report.py" `
            --run-dir $RunDir `
            --report-dir $ReportDir `
            --config $Config `
            --dataset-root "datasets" `
            --device "cuda:0" `
            --batch-size 8 `
            --num-workers 0 `
            --point-v2-summary $ReferenceSummary `
            --reference-label "EMA_BIFPN_NEW1804_FAIR100" `
            --primary-label "RTDETRV4_BASELINE_NEW1804_FAIR100" `
            --report-title "NEW1804 RT-DETRv4 baseline fair100 中文结论" `
            --change-note "新数据集 datasets/ 1804 images; RT-DETRv4/GPPoint-DETR baseline fair100 from HGNetv2 pretrained; no EMA_BIFPN; no old checkpoint tuning; no failed branches enabled." `
            --save-prediction-records
    }

    Run-Step "NEW1804 baseline decision table" {
        & $Python "tools\build_new1804_baseline_decision.py"
    }
}

if (-not $KeepPeriodicCheckpoints) {
    $resolvedRun = Resolve-Path -LiteralPath $RunDir
    $periodic = Get-ChildItem -LiteralPath $resolvedRun -Filter "checkpoint*.pth" -File -ErrorAction SilentlyContinue
    foreach ($file in $periodic) {
        if ($file.FullName -notlike "$($resolvedRun.Path)*") {
            throw "Refusing to delete outside run dir: $($file.FullName)"
        }
    }
    if ($periodic.Count -gt 0) {
        $periodic | Remove-Item -Force
        Write-Host "Removed $($periodic.Count) periodic checkpoint*.pth files; kept best_* and last.pth." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done. Report dir: $ReportDir" -ForegroundColor Green
