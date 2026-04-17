[CmdletBinding()]
param(
    [string]$SuiteDir = "",
    [string]$BaselineReplaySummary = "",
    [string]$PointV2Summary = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

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

$resolvedSuiteDir = Resolve-PathFromRepo -RawPath $SuiteDir -Fallback "outputs\grape_point_v5_suite\report"
$resolvedBaselineReplaySummary = Resolve-PathFromRepo -RawPath $BaselineReplaySummary -Fallback "outputs\grape_point_v5_baseline_replay\report\summary.json"
$resolvedPointV2Summary = Resolve-PathFromRepo -RawPath $PointV2Summary -Fallback "outputs\grape_point_v2_main\report\summary.json"

$command = @(
    ".\\.venv\\Scripts\\python.exe",
    "tools\\make_grape_point_v5_suite_report.py",
    "--suite-dir", $resolvedSuiteDir,
    "--baseline-replay-summary", $resolvedBaselineReplaySummary,
    "--point-v2-summary", $resolvedPointV2Summary
)

& $command[0] $command[1..($command.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "[point_v5 suite] report generation failed with exit code $LASTEXITCODE"
}
