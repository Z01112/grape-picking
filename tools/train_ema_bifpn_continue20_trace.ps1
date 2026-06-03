param(
    [ValidateSet("trace20")]
    [string]$Mode = "trace20",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [switch]$Resume,
    [string]$ResumeCheckpoint = ""
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Block
    )
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed, exit code: $LASTEXITCODE"
    }
}

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Config = Join-Path $RepoRoot "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_continue20_trace.yml"
$WarmStart = Join-Path $RepoRoot "outputs\02_encoder_experiments\encoder_ema_bifpn_weighted_fusion_100e_backbone_pretrain_20260526\best_composite.pth"
$RunDir = Join-Path $RepoRoot "outputs\26_continue_training_tradeoff\ema_bifpn_continue20_trace"
$OutputDir = Join-Path $RepoRoot "outputs\26_continue_training_tradeoff"
$EmaSummary = Join-Path $RepoRoot "outputs\08_eval_unification\ema_bifpn_unified_report\summary.json"

if (!(Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (!(Test-Path -LiteralPath $Config)) { throw "Config not found: $Config" }
if (!(Test-Path -LiteralPath $WarmStart)) { throw "EMA_BIFPN warm-start checkpoint not found: $WarmStart" }
if (!(Test-Path -LiteralPath $EmaSummary)) { throw "EMA_BIFPN unified summary not found: $EmaSummary" }

New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if ($Resume) {
    if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
        $ResumeCheckpoint = Join-Path $RunDir "last.pth"
    } elseif (-not [System.IO.Path]::IsPathRooted($ResumeCheckpoint)) {
        $ResumeCheckpoint = Join-Path $RepoRoot $ResumeCheckpoint
    }
    if (!(Test-Path -LiteralPath $ResumeCheckpoint)) {
        throw "Resume checkpoint not found: $ResumeCheckpoint"
    }
    Run-Step "EMA_BIFPN_CONTINUE20_TRACE resume training" {
        & $Python (Join-Path $RepoRoot "train.py") `
            --config $Config `
            --resume $ResumeCheckpoint `
            --device $Device `
            --output-dir $RunDir
    }
} else {
    Run-Step "EMA_BIFPN_CONTINUE20_TRACE training" {
        & $Python (Join-Path $RepoRoot "train.py") `
            --config $Config `
            --tuning $WarmStart `
            --device $Device `
            --output-dir $RunDir
    }
}

Run-Step "EMA_BIFPN_CONTINUE20_TRACE trajectory analysis" {
    & $Python (Join-Path $RepoRoot "tools\analyze_continue_training_tradeoff.py") `
        --config $Config `
        --run-dir $RunDir `
        --output-dir $OutputDir `
        --ema-summary $EmaSummary `
        --device $Device `
        --batch-size $BatchSize `
        --num-workers $NumWorkers `
        --eval-test
}

Write-Host ""
Write-Host "Done. Main outputs:" -ForegroundColor Green
Write-Host (Join-Path $OutputDir "continue20_metric_trajectory.csv")
Write-Host (Join-Path $OutputDir "continue20_metric_trajectory.md")
Write-Host (Join-Path $OutputDir "continue_training_tradeoff_decision.md")
