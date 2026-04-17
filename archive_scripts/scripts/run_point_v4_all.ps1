# Point v4 all-in-one runner.
# Default behavior:
# - runs exp1 -> exp4 in order
# - skips experiments that already have report\summary.json
# - writes suite comparison report at the end
# Recovery example:
#   .\scripts\run_point_v4_all.ps1 -StartFrom exp2 -ResumeLast
param(
    [ValidateSet("exp0", "exp1", "exp2", "exp3", "exp4")]
    [string]$StartFrom = "exp1",
    [switch]$ResumeLast,
    [switch]$ForceRerun,
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [int]$Epochs = -1,
    [int]$ExtraEpochs = 0,
    [switch]$SkipSuiteReport,
    [string[]]$ConfigUpdates = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$order = @("exp1", "exp2", "exp3", "exp4")
$defaultOutputs = @{
    "exp0" = "outputs\grape_point_v4_exp0_query_only"
    "exp1" = "outputs\grape_point_v4_exp1_full_roi"
    "exp2" = "outputs\grape_point_v4_exp2_top_roi"
    "exp3" = "outputs\grape_point_v4_exp3_full_top_roi"
    "exp4" = "outputs\grape_point_v4_exp4_full_top_roi_yw"
}

function Find-LastCheckpoint {
    param([string]$RunDir)
    foreach ($candidate in @(
        (Join-Path $RunDir 'checkpoints\last.pth'),
        (Join-Path $RunDir 'last.pth')
    )) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

$startIndex = $order.IndexOf($StartFrom)
if ($startIndex -lt 0) {
    throw "Unsupported StartFrom: $StartFrom"
}

$toRun = $order[$startIndex..($order.Length - 1)]
Write-Host "[point_v4 all] plan: $($toRun -join ', ')"

foreach ($exp in $toRun) {
    $runDir = Join-Path $repoRoot $defaultOutputs[$exp]
    $summaryPath = Join-Path $runDir 'report\summary.json'
    $lastCheckpoint = Find-LastCheckpoint -RunDir $runDir
    $hasResumeCheckpoint = -not [string]::IsNullOrWhiteSpace($lastCheckpoint)

    $args = @{
        Experiment = $exp
        OutputDir = $defaultOutputs[$exp]
        Device = $Device
        BatchSize = $BatchSize
        NumWorkers = $NumWorkers
    }
    if ($Epochs -gt 0) {
        $args["Epochs"] = $Epochs
    }
    if ($ExtraEpochs -gt 0) {
        $args["ExtraEpochs"] = $ExtraEpochs
    }
    if ($ResumeLast -and $hasResumeCheckpoint) {
        $args["ResumeLast"] = $true
    }
    if ($ConfigUpdates.Count -gt 0) {
        $args["ConfigUpdates"] = $ConfigUpdates
    }

    $shouldSkip = (Test-Path -LiteralPath $summaryPath) -and -not $ForceRerun -and -not ($ResumeLast -and $hasResumeCheckpoint) -and $Epochs -le 0 -and $ExtraEpochs -le 0
    if ($shouldSkip) {
        Write-Host "[point_v4 all] skip $exp because summary already exists: $summaryPath"
        continue
    }

    if ($ResumeLast -and -not $hasResumeCheckpoint) {
        Write-Host "[point_v4 all] $exp has no last.pth, start fresh."
    }
    elseif ($ResumeLast -and $hasResumeCheckpoint) {
        Write-Host "[point_v4 all] $exp resume from: $lastCheckpoint"
    }

    Write-Host "[point_v4 all] running $exp ..."
    & ".\scripts\run_point_v4.ps1" @args

    if ($LASTEXITCODE -ne 0) {
        throw "[point_v4 all] experiment $exp failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipSuiteReport) {
    Write-Host "[point_v4 all] generating suite report..."
    & ".\scripts\make_point_v4_suite_report.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "[point_v4 all] suite report failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[point_v4 all] done."
