param(
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [string]$OutputDir = "outputs\03_global_analysis\hp_pick_calibration_$(Get-Date -Format yyyyMMdd)"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

Push-Location $Root
try {
    & $Python tools\evaluate_hp_pick_calibration.py `
        --device $Device `
        --batch-size $BatchSize `
        --num-workers $NumWorkers `
        --output-dir $OutputDir
} finally {
    Pop-Location
}
