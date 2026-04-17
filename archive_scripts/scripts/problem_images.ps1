# Problem image mining entry.
# If you want this script to always point to one run folder, edit $PinnedRunDir below.
param(
    [string]$RunDir = "",
    [float]$ScoreThr = 0.25,
    [float]$IoUThr = 0.5,
    [int]$TopK = 12
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$PinnedRunDir = ""
$LastRunFile = Join-Path $PSScriptRoot ".last_baseline_run.txt"

function Resolve-BaselineRunDir {
    param([string]$RunDirInput)

    if (-not [string]::IsNullOrWhiteSpace($RunDirInput)) {
        $raw = $RunDirInput.Trim().Trim('"')
        $absoluteSuffixMatch = [regex]::Match($raw, '([A-Za-z]:[\\/].*)$')
        if ($absoluteSuffixMatch.Success) {
            $raw = $absoluteSuffixMatch.Groups[1].Value
        }
        if ([System.IO.Path]::IsPathRooted($raw)) {
            return [System.IO.Path]::GetFullPath($raw)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $raw))
    }

    if (-not [string]::IsNullOrWhiteSpace($PinnedRunDir)) {
        return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PinnedRunDir))
    }

    if (Test-Path -LiteralPath $LastRunFile) {
        $saved = (Get-Content -LiteralPath $LastRunFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if (-not [string]::IsNullOrWhiteSpace($saved) -and (Test-Path -LiteralPath $saved)) {
            return [System.IO.Path]::GetFullPath($saved)
        }
    }

    $latest = Get-ChildItem outputs -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'baseline_*' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No baseline run directory was found under outputs."
    }
    return [System.IO.Path]::GetFullPath($latest.FullName)
}

$resolvedRunDir = Resolve-BaselineRunDir $RunDir
$reportDir = Join-Path $resolvedRunDir 'report'
$summaryPath = Join-Path $reportDir 'summary.json'
$perImagePath = Join-Path $reportDir 'per_image_test_summary.json'
$predictionsPath = Join-Path $reportDir 'predictions_test.json'

if (-not (Test-Path -LiteralPath $resolvedRunDir)) {
    throw "RunDir does not exist: $resolvedRunDir`nExample: .\scripts\problem_images.ps1 -RunDir outputs\baseline_20260406"
}
if (-not (Test-Path -LiteralPath $summaryPath)) {
    throw "summary.json was not found in: $reportDir`nRun .\scripts\make_report.ps1 first."
}
if (-not (Test-Path -LiteralPath $perImagePath)) {
    throw "per_image_test_summary.json was not found in: $reportDir"
}
if (-not (Test-Path -LiteralPath $predictionsPath)) {
    throw "predictions_test.json was not found in: $reportDir"
}

Write-Host "[problem_images] run dir: $resolvedRunDir"

& ".\.venv\Scripts\python.exe" "tools\report_problem_images.py" `
    --dataset-root "dataset" `
    --report-dir $reportDir `
    --score-thr $ScoreThr `
    --iou-thr $IoUThr `
    --top-k $TopK
