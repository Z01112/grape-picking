# Point v4 suite report entry.
param(
    [string]$SuiteDir = "outputs\grape_point_v4_suite\report"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedSuiteDir = if ([System.IO.Path]::IsPathRooted($SuiteDir)) {
    [System.IO.Path]::GetFullPath($SuiteDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $SuiteDir))
}

& ".\\.venv\\Scripts\\python.exe" "tools\\make_grape_point_v4_suite_report.py" "--suite-dir" $resolvedSuiteDir

Write-Host "[point_v4 suite] wrote suite report: $resolvedSuiteDir"
