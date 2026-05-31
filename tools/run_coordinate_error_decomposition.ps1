param(
    [string]$OutputDir = "outputs\03_global_analysis\coordinate_error_decomposition_20260531",
    [ValidateSet("valid", "test", "both")]
    [string]$Split = "test"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "tools\coordinate_error_decomposition.py"

if (!(Test-Path -LiteralPath $Py)) {
    throw "Python not found: $Py"
}
if (!(Test-Path -LiteralPath $Script)) {
    throw "Diagnostic script not found: $Script"
}

Write-Host "===== Coordinate Error Decomposition =====" -ForegroundColor Cyan
& $Py $Script --output-dir $OutputDir --split $Split
if ($LASTEXITCODE -ne 0) {
    throw "Coordinate Error Decomposition failed, exit code: $LASTEXITCODE"
}

$ReportPath = Join-Path $Root (Join-Path $OutputDir "coordinate_error_decomposition_report_zh.md")
$SummaryPath = Join-Path $Root (Join-Path $OutputDir "coordinate_error_decomposition_summary.json")
$CsvPath = Join-Path $Root (Join-Path $OutputDir "coordinate_standard_pair_decomposition.csv")

Write-Host ""
Write-Host "Diagnostic complete. Main files:" -ForegroundColor Green
Write-Host $ReportPath
Write-Host $SummaryPath
Write-Host $CsvPath
