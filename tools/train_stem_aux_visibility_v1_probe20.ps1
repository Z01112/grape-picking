param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [switch]$SkipPrechecks,
    [switch]$SkipReport,
    [switch]$KeepFailedCheckpoints
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Config = Join-Path $RepoRoot "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_new1804_stem_aux_visibility_v1_probe20.yml"
$RunDir = Join-Path $RepoRoot "outputs\01_mainline_results\candidate_stem_aux_visibility_v1_probe20"
$ReportDir = Join-Path $RunDir "report"
$Tuning = Join-Path $RepoRoot "outputs\01_mainline_results\ema_bifpn_new1804_fair100\best_composite.pth"
$ReferenceSummary = Join-Path $RepoRoot "outputs\01_mainline_results\ema_bifpn_new1804_fair100\report\summary.json"

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
if (-not (Test-Path -LiteralPath $Tuning)) {
    throw "EMA_BIFPN_NEW1804 tuning checkpoint not found: $Tuning"
}

if (-not $SkipPrechecks) {
    Run-Step "STEM_AUX_VISIBILITY_V1 smoke check" {
        & $Python "tools\check_stem_aux_visibility_smoke.py" --device cpu
    }
}

Run-Step "STEM_AUX_VISIBILITY_V1 probe20 train from EMA_BIFPN_NEW1804 best_composite" {
    & $Python "train.py" `
        -c $Config `
        -d $Device `
        -t $Tuning `
        --use-amp
}

if (-not $SkipReport) {
    Run-Step "STEM_AUX_VISIBILITY_V1 probe20 unified report" {
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
            --primary-label "STEM_AUX_VISIBILITY_V1_PROBE20" `
            --report-title "NEW1804 Stem Aux Visibility V1 Probe20 中文结论" `
            --change-note "Stem visibility auxiliary branch; warm-start from EMA_BIFPN_NEW1804 best_composite for 20-epoch probe only; matcher/postprocessor/dataset/threshold unchanged; no old failed branches enabled." `
            --save-prediction-records
    }

    Run-Step "STEM_AUX_VISIBILITY_V1 probe20 decision" {
        & $Python "tools\build_stem_aux_visibility_probe20_decision.py"
    }

    $DecisionPath = Join-Path $ReportDir "stem_aux_visibility_probe20_decision.json"
    if (Test-Path -LiteralPath $DecisionPath) {
        $Decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
        if (-not $Decision.keep_checkpoint -and -not $KeepFailedCheckpoints) {
            Write-Host ""
            Write-Host "===== Cleanup failed probe checkpoints =====" -ForegroundColor Cyan
            $Checkpoints = Get-ChildItem -LiteralPath $RunDir -Filter "*.pth" -File -ErrorAction SilentlyContinue
            foreach ($Ckpt in $Checkpoints) {
                Remove-Item -LiteralPath $Ckpt.FullName -Force
                Write-Host "Removed $($Ckpt.FullName)"
            }
        }
    }
}

Write-Host ""
Write-Host "Done. Report dir: $ReportDir" -ForegroundColor Green
