param(
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [string]$OutputDir = "outputs\03_global_analysis\selection_calibration_picking_first_20260531",
    [string]$Thresholds = "0.46:0.70:0.02",
    [switch]$RefreshRecords
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$ArgsList = @(
    (Join-Path $RepoRoot "tools\evaluate_selection_reassignment.py"),
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--output-dir", (Join-Path $RepoRoot $OutputDir),
    "--thresholds", $Thresholds
)

if ($RefreshRecords) {
    $ArgsList += "--refresh-records"
}

& $Python @ArgsList

if ($LASTEXITCODE -ne 0) {
    throw "Selection Reassignment V1 failed, exit code: $LASTEXITCODE"
}
