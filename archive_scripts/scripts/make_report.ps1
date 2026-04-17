# Baseline report entry.
# If you want this script to always point to one run folder, edit $PinnedRunDir below.
param(
    [string]$RunDir = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [switch]$ReuseExistingEval,
    [switch]$SkipPredictions
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

function Resolve-RunArtifact {
    param(
        [string]$BaseDir,
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        $fullPath = Join-Path $BaseDir $candidate
        if (Test-Path -LiteralPath $fullPath) {
            return $fullPath
        }
    }
    return (Join-Path $BaseDir $Candidates[0])
}

function Save-LastRunDir {
    param([string]$ResolvedRunDir)
    Set-Content -LiteralPath $LastRunFile -Value $ResolvedRunDir -Encoding Ascii
}

function Move-RunItemIntoDir {
    param(
        [string]$SourcePath,
        [string]$DestinationDir,
        [string]$RunDirRoot
    )

    if (-not (Test-Path -LiteralPath $SourcePath)) {
        return
    }

    $resolvedRoot = [System.IO.Path]::GetFullPath($RunDirRoot)
    $resolvedSource = [System.IO.Path]::GetFullPath($SourcePath)
    if (-not $resolvedSource.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to move an item outside the run directory: $resolvedSource"
    }

    New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
    $targetPath = Join-Path $DestinationDir (Split-Path -Leaf $SourcePath)
    if (Test-Path -LiteralPath $targetPath) {
        Remove-Item -LiteralPath $targetPath -Recurse -Force
    }
    Move-Item -LiteralPath $SourcePath -Destination $targetPath
}

function Sync-BaselineRunLayout {
    param([string]$ResolvedRunDir)

    $checkpointsDir = Join-Path $ResolvedRunDir 'checkpoints'
    $logsDir = Join-Path $ResolvedRunDir 'logs'
    $reportDir = Join-Path $ResolvedRunDir 'report'

    New-Item -ItemType Directory -Path $checkpointsDir -Force | Out-Null
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

    $checkpointFiles = Get-ChildItem -LiteralPath $ResolvedRunDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(best_stg1|best_stg2|last)\.pth$' -or $_.Name -match '^checkpoint\d+\.pth$' }
    foreach ($file in $checkpointFiles) {
        Move-RunItemIntoDir -SourcePath $file.FullName -DestinationDir $checkpointsDir -RunDirRoot $ResolvedRunDir
    }

    foreach ($name in @('log.txt')) {
        $source = Join-Path $ResolvedRunDir $name
        if (Test-Path -LiteralPath $source) {
            Move-RunItemIntoDir -SourcePath $source -DestinationDir $logsDir -RunDirRoot $ResolvedRunDir
        }
    }

    foreach ($dirName in @('summary', 'eval', 'eval_valid', 'eval_test')) {
        $sourceDir = Join-Path $ResolvedRunDir $dirName
        if (Test-Path -LiteralPath $sourceDir) {
            Move-RunItemIntoDir -SourcePath $sourceDir -DestinationDir $logsDir -RunDirRoot $ResolvedRunDir
        }
    }

    foreach ($dashboardName in @('summary.json', 'results.csv', 'training_curves.png', 'results_overview.png', 'test_class_metrics.png', 'error_breakdown.png', 'test_gt_vs_pred.jpg')) {
        $target = Join-Path $ResolvedRunDir $dashboardName
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Force
        }
    }
}

$resolvedRunDir = Resolve-BaselineRunDir $RunDir
$logPath = Resolve-RunArtifact -BaseDir $resolvedRunDir -Candidates @('log.txt', 'logs\log.txt')
$checkpointPath = Resolve-RunArtifact -BaseDir $resolvedRunDir -Candidates @('best_stg2.pth', 'checkpoints\best_stg2.pth')
$reportDir = Join-Path $resolvedRunDir 'report'

if (-not (Test-Path -LiteralPath $resolvedRunDir)) {
    throw "RunDir does not exist: $resolvedRunDir`nExample: .\scripts\make_report.ps1 -RunDir outputs\baseline_20260406"
}
if (-not (Test-Path -LiteralPath $logPath)) {
    throw "log.txt was not found in: $resolvedRunDir`nMake sure RunDir points to a baseline output directory."
}
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "best_stg2.pth was not found in: $resolvedRunDir"
}

$command = @(
    ".\.venv\Scripts\python.exe",
    "tools\make_grape_picking_baseline_report.py",
    "--dataset-root", "dataset",
    "--config", "configs\rtv4\rtv4_hgnetv2_s_grape_picking_baseline.yml",
    "--run-dir", $resolvedRunDir,
    "--report-dir", $reportDir,
    "--checkpoint", $checkpointPath,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers
)

if ($ReuseExistingEval) {
    $command += "--reuse-existing-eval"
}
if ($SkipPredictions) {
    $command += "--skip-predictions"
}

& $command[0] $command[1..($command.Length - 1)]

Sync-BaselineRunLayout -ResolvedRunDir $resolvedRunDir
Save-LastRunDir -ResolvedRunDir $resolvedRunDir
Write-Host "[report] organized run dir: $resolvedRunDir"
