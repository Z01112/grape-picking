param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [int]$BatchSize = 8,
    [switch]$SkipReport
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

function Run-Experiment {
    param(
        [string]$Label,
        [string]$ConfigRel,
        [string]$RunDirRel,
        [string]$ChangeNote
    )

    $Config = Join-Path $RepoRoot $ConfigRel
    $RunDir = Join-Path $RepoRoot $RunDirRel
    $ReportDir = Join-Path $RunDir "report"

    if (-not (Test-Path -LiteralPath $Config)) {
        throw "Config not found: $Config"
    }

    Run-Step "$Label fair100 train from HGNetv2 pretrained" {
        & $Python "train.py" `
            -c $Config `
            -d $Device `
            --use-amp
    }

    if (-not $SkipReport) {
        Run-Step "$Label unified report" {
            & $Python "tools\make_grape_point_report.py" `
                --run-dir $RunDir `
                --report-dir $ReportDir `
                --config $Config `
                --dataset-root "datasets" `
                --device "cuda:0" `
                --batch-size $BatchSize `
                --num-workers 0 `
                --point-v2-summary $ReferenceSummary `
                --reference-label "B0 EMA_BIFPN" `
                --primary-label $Label `
                --report-title "$Label 中文结论" `
                --change-note $ChangeNote `
                --save-prediction-records
        }
    }
}

Set-Location $RepoRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

Run-Experiment `
    -Label "B0_POINTCA_FAIR100" `
    -ConfigRel "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_point_cross_attn_fair100.yml" `
    -RunDirRel "outputs\01_mainline_results\candidate_b0_point_cross_attn_fair100" `
    -ChangeNote "B0 EMA_BIFPN plus PointCA; no B1 backbone; no PAM; fair100 from HGNetv2 pretrained; current default dataset."

Run-Experiment `
    -Label "B0_PAM_FAIR100" `
    -ConfigRel "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_pam_fair100.yml" `
    -RunDirRel "outputs\01_mainline_results\candidate_b0_pam_fair100" `
    -ChangeNote "B0 EMA_BIFPN plus point-aware Hungarian matching; no PointCA; fair100 from HGNetv2 pretrained; current default dataset."

Run-Experiment `
    -Label "B0_POINTCA_PAM_FAIR100" `
    -ConfigRel "configs\rtv4\rtv4_hgnetv2_s_grape_point_ema_bifpn_point_cross_attn_pam_fair100.yml" `
    -RunDirRel "outputs\01_mainline_results\candidate_b0_point_cross_attn_pam_fair100" `
    -ChangeNote "B0 EMA_BIFPN plus PointCA and point-aware Hungarian matching; no B1 backbone; fair100 from HGNetv2 pretrained; current default dataset."

if (-not $SkipReport) {
    Run-Step "B0/B1 PointCA PAM ablation summary" {
        & $Python "tools\build_b0_b1_pointca_pam_ablation_table.py"
    }
}

Write-Host ""
Write-Host "Done. Summary: outputs\04_diagnostics\b0_b1_pointca_pam_ablation\b0_b1_pointca_pam_ablation_summary.md" -ForegroundColor Green
