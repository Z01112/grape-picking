param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [switch]$SkipB1,
    [switch]$SkipPointCA,
    [switch]$SkipPointCAPAM,
    [switch]$SkipB1PAM
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ReferenceSummary = Join-Path $RepoRoot "outputs\01_mainline_results\ema_bifpn_new1804_fair100\report\summary.json"

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

function Run-Report {
    param(
        [string]$Name,
        [string]$RunDir,
        [string]$Config,
        [string]$Label,
        [string]$Note
    )
    $FullRunDir = Join-Path $RepoRoot $RunDir
    $ReportDir = Join-Path $FullRunDir "report"
    $FullConfig = Join-Path $RepoRoot $Config
    if (-not (Test-Path -LiteralPath $FullRunDir)) {
        Write-Host "Skip $Name, missing run dir: $FullRunDir" -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $FullRunDir "best_composite.pth"))) {
        Write-Host "Skip $Name, missing best_composite.pth in $FullRunDir" -ForegroundColor Yellow
        return
    }
    Run-Step "$Name unified report" {
        & $Python "tools\make_grape_point_report.py" `
            --run-dir $FullRunDir `
            --report-dir $ReportDir `
            --config $FullConfig `
            --dataset-root "datasets" `
            --device $Device `
            --batch-size $BatchSize `
            --num-workers 0 `
            --point-v2-summary $ReferenceSummary `
            --reference-label "B0 EMA_BIFPN" `
            --primary-label $Label `
            --report-title "$Label 中文结论" `
            --change-note $Note `
            --save-prediction-records
    }
}

Set-Location $RepoRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

if (-not $SkipB1) {
    Run-Report `
        -Name "B1" `
        -RunDir "outputs\01_mainline_results\candidate_b1_backbone_new1804_fair100" `
        -Config "configs\rtv4\rtv4_hgnetv2_b1_s_grape_point_ema_bifpn_new1804_fair100.yml" `
        -Label "B1_FAIR100" `
        -Note "B1 backbone; no PointCA; no PAM; fair100 from HGNetv2 pretrained; current default dataset."
}

if (-not $SkipPointCA) {
    Run-Report `
        -Name "B1_PointCA" `
        -RunDir "outputs\01_mainline_results\candidate_b1_point_cross_attn_new1804_fair100" `
        -Config "configs\rtv4\rtv4_hgnetv2_b1_s_grape_point_ema_bifpn_new1804_point_cross_attn_fair100.yml" `
        -Label "B1_POINTCA_FAIR100" `
        -Note "B1 backbone plus point cross-attention; no PAM; fair100 from HGNetv2 pretrained; current default dataset."
}

if (-not $SkipPointCAPAM) {
    Run-Report `
        -Name "B1_PointCA_PAM" `
        -RunDir "outputs\01_mainline_results\candidate_b1_point_cross_attn_pam_fair100" `
        -Config "configs\rtv4\rtv4_hgnetv2_b1_s_grape_point_ema_bifpn_new1804_point_cross_attn_pam_fair100.yml" `
        -Label "B1_POINTCA_PAM_FAIR100" `
        -Note "B1 backbone plus point cross-attention and point-aware matching; fair100 from HGNetv2 pretrained; current default dataset."
}

if (-not $SkipB1PAM) {
    Run-Report `
        -Name "B1_PAM" `
        -RunDir "outputs\01_mainline_results\candidate_b1_pam_fair100" `
        -Config "configs\rtv4\rtv4_hgnetv2_b1_s_grape_point_ema_bifpn_pam_fair100.yml" `
        -Label "B1_PAM_FAIR100" `
        -Note "B1 backbone plus point-aware matching; no PointCA; fair100 from HGNetv2 pretrained; current default dataset."
}

Run-Step "Priority 1/3 analysis" {
    & $Python "tools\build_b1_pointca_pam_priority123.py"
}

Write-Host ""
Write-Host "Done. Priority outputs: outputs\04_diagnostics\b1_pointca_pam_priority123" -ForegroundColor Green
