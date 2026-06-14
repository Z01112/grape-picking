param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [switch]$SkipPrechecks,
    [switch]$SkipReport
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LegacyScript = Join-Path $RepoRoot "tools\train_new1804_ema_bifpn_fair100.ps1"

if (-not (Test-Path -LiteralPath $LegacyScript)) {
    throw "Legacy reproducibility script not found: $LegacyScript"
}

Write-Host "Current-dataset alias: GPPoint-DETR / EMA_BIFPN fair100" -ForegroundColor Cyan
Write-Host "This wrapper preserves the historical output path used by completed reports." -ForegroundColor Yellow

& $LegacyScript @PSBoundParameters
exit $LASTEXITCODE

