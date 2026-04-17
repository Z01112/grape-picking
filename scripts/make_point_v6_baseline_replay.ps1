[CmdletBinding()]
param(
    [string]$RunDir = "",
    [string]$ReportDir = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$PinnedRunDir = "outputs\grape_point_v2_main"
$PinnedReportDir = "outputs\grape_point_v6_baseline_replay\report"
$ConfigPath = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v2.yml"

function Resolve-PathFromRepo {
    param([string]$RawPath, [string]$Fallback)
    $value = $RawPath
    if ([string]::IsNullOrWhiteSpace($value)) {
        $value = $Fallback
    }
    $value = $value.Trim().Trim('"')
    if ([System.IO.Path]::IsPathRooted($value)) {
        return [System.IO.Path]::GetFullPath($value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $value))
}

$resolvedRunDir = Resolve-PathFromRepo -RawPath $RunDir -Fallback $PinnedRunDir
$resolvedReportDir = Resolve-PathFromRepo -RawPath $ReportDir -Fallback $PinnedReportDir

if (-not (Test-Path -LiteralPath $resolvedRunDir)) {
    throw "RunDir does not exist: $resolvedRunDir"
}

New-Item -ItemType Directory -Path $resolvedReportDir -Force | Out-Null

$command = @(
    ".\\.venv\\Scripts\\python.exe",
    "tools\\make_grape_point_v2_report.py",
    "--run-dir", $resolvedRunDir,
    "--report-dir", $resolvedReportDir,
    "--config", $ConfigPath,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--primary-label", "baseline_replay",
    "--reference-label", "point_v2",
    "--report-title", "point_v6 baseline replay 中文结论",
    "--report-mode", "point_v6_baseline_replay",
    "--change-note", "Reuse the existing point_v2 checkpoint and export a fresh report with the current unified evaluator.",
    "--change-note", "No retraining is performed in baseline replay."
)

& $command[0] $command[1..($command.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "[point_v6 baseline_replay] report generation failed with exit code $LASTEXITCODE"
}
