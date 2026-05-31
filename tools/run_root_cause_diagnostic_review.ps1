param(
    [string]$OutputDir = "outputs\03_global_analysis\root_cause_diagnostic_review_20260529"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "tools\root_cause_diagnostic_review.py"

if (!(Test-Path -LiteralPath $Py)) {
    throw "Python not found: $Py"
}
if (!(Test-Path -LiteralPath $Script)) {
    throw "Diagnostic script not found: $Script"
}

Write-Host "===== Root-Cause Diagnostic Review =====" -ForegroundColor Cyan
& $Py $Script --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Root-Cause Diagnostic Review failed, exit code: $LASTEXITCODE"
}

$ReportPath = Join-Path $Root (Join-Path $OutputDir "root_cause_diagnostic_report_zh.md")
$MatrixPath = Join-Path $Root (Join-Path $OutputDir "root_cause_metric_matrix.csv")
$DecisionPath = Join-Path $Root (Join-Path $OutputDir "diagnosis_decision.json")

Write-Host ""
Write-Host "Diagnostic complete. Main files:" -ForegroundColor Green
Write-Host $ReportPath
Write-Host $MatrixPath
Write-Host $DecisionPath
