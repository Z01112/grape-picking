param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [switch]$KeepPeriodicCheckpoints
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Config = Join-Path $RepoRoot "configs\rtv4\rtv4_hgnetv2_s_olddata_picking_bbox_to_point_baseline_fair100.yml"
$RunDir = Join-Path $RepoRoot "outputs\02_baselines\olddata_picking_bbox_to_point_baseline_fair100"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Block
    )
    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed, exit code: $LASTEXITCODE"
    }
}

Set-Location $RepoRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python. Use .\.venv\Scripts\python.exe from the project venv."
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Config not found: $Config"
}
foreach ($ann in @(
    "dataset\train\_annotations.rtv4.json",
    "dataset\valid\_annotations.rtv4.json",
    "dataset\test\_annotations.rtv4.json"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $ann))) {
        throw "Old picking-bbox annotation not found: $ann"
    }
}

Write-Host ""
Write-Host "This baseline is detection-only RT-DETRv4 on old dataset/." -ForegroundColor Yellow
Write-Host "Protocol after training: convert predicted picking bbox center to picking point, then associate to grape." -ForegroundColor Yellow
Write-Host "It does not use GPPoint has_picking / picking_offset decoder heads." -ForegroundColor Yellow

Run-Step "OLD-DATA RT-DETRv4 picking-bbox-to-point baseline fair100 train from HGNetv2 pretrained" {
    & $Python "train.py" `
        -c $Config `
        -d $Device `
        --use-amp
}

Run-Step "OLD-DATA RT-DETRv4 picking-bbox-to-point baseline report" {
    & $Python "tools\evaluate_olddata_bbox_to_point_baseline.py" `
        --config $Config `
        --run-dir $RunDir `
        --dataset-root "dataset" `
        --device $Device `
        --batch-size 4 `
        --num-workers 0
}

if (-not $KeepPeriodicCheckpoints) {
    if (Test-Path -LiteralPath $RunDir) {
        $resolvedRun = Resolve-Path -LiteralPath $RunDir
        $periodic = Get-ChildItem -LiteralPath $resolvedRun -Filter "checkpoint*.pth" -File -ErrorAction SilentlyContinue
        foreach ($file in $periodic) {
            if ($file.FullName -notlike "$($resolvedRun.Path)*") {
                throw "Refusing to delete outside run dir: $($file.FullName)"
            }
        }
        if ($periodic.Count -gt 0) {
            $periodic | Remove-Item -Force
            Write-Host "Removed $($periodic.Count) periodic checkpoint*.pth files; kept best_* and last.pth if generated." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Done. Run dir: $RunDir" -ForegroundColor Green
Write-Host "Report dir: $(Join-Path $RunDir 'report')" -ForegroundColor Green
