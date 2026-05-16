$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo

$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$ConfigCurrent = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v7_exp2.yml"
$ConfigDn = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v7_exp2_dn_teacher_roi.yml"
$ReferenceSummary = "outputs\grape_point_gppoint_detr_main\report\summary.json"

function Invoke-LoggedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][scriptblock]$Command
  )
  $logDir = Split-Path -Parent $LogPath
  if ($logDir) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
  }
  & $Command 2>&1 | Tee-Object -FilePath $LogPath
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed, see $LogPath"
  }
}

function Train-DnRunIfNeeded {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Nullable[int]]$Seed = $null
  )
  $best = Join-Path $OutputDir "best_composite.pth"
  if (Test-Path -LiteralPath $best) {
    Write-Host "[$Name] best_composite.pth exists, skip training: $best"
    return
  }

  New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
  $trainLog = Join-Path $OutputDir "train_stdout.log"
  Write-Host "[$Name] start training -> $OutputDir"
  if ($null -eq $Seed) {
    Invoke-LoggedCommand -LogPath $trainLog -Command {
      & $Python train.py -c $ConfigDn -d cuda:0 --use-amp --output-dir $OutputDir
    }
  } else {
    Invoke-LoggedCommand -LogPath $trainLog -Command {
      & $Python train.py -c $ConfigDn -d cuda:0 --use-amp --seed $Seed --output-dir $OutputDir
    }
  }
}

function Build-ReportIfNeeded {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$RunDir
  )
  $summary = Join-Path $RunDir "report\summary.json"
  $best = Join-Path $RunDir "best_composite.pth"
  if (-not (Test-Path -LiteralPath $best)) {
    throw "Missing checkpoint for $Name`: $best"
  }
  if (Test-Path -LiteralPath $summary) {
    Write-Host "[$Name] summary exists, skip report: $summary"
    return
  }
  Write-Host "[$Name] build report"
  Invoke-LoggedCommand -LogPath (Join-Path $RunDir "make_report_stdout.log") -Command {
    & $Python tools\make_grape_point_report.py `
      --run-dir $RunDir `
      --report-dir (Join-Path $RunDir "report") `
      --config $Config `
      --checkpoint $best `
      --device cuda:0 `
      --batch-size 4 `
      --num-workers 0 `
      --primary-label $Name `
      --reference-label current `
      --point-v2-summary $ReferenceSummary `
      --report-title $Name
  }
}

function Export-PredictionsIfNeeded {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Split
  )
  $best = Join-Path $RunDir "best_composite.pth"
  $out = Join-Path $RunDir "predictions\$($Split)_predictions.json"
  if (Test-Path -LiteralPath $out) {
    Write-Host "[$Name][$Split] predictions exist, skip export: $out"
    return
  }
  Write-Host "[$Name][$Split] export predictions"
  Invoke-LoggedCommand -LogPath (Join-Path $RunDir "export_$($Split)_stdout.log") -Command {
    & $Python tools\export_grape_point_predictions.py `
      --config $Config `
      --checkpoint $best `
      --split $Split `
      --output $out `
      --device cuda:0 `
      --batch-size 4 `
      --num-workers 0
  }
}

$DnRuns = @(
  @{ Name = "dn_teacher_roi_main"; Dir = "outputs\grape_point_v7_exp2_dn_teacher_roi"; Seed = $null },
  @{ Name = "dn_teacher_roi_repro1"; Dir = "outputs\grape_point_v7_exp2_dn_teacher_roi_repro1"; Seed = $null },
  @{ Name = "dn_teacher_roi_seed2026"; Dir = "outputs\grape_point_v7_exp2_dn_teacher_roi_seed2026"; Seed = 2026 }
)

$CurrentRuns = @(
  @{ Name = "current_main"; Dir = "outputs\grape_point_gppoint_detr_main"; Config = $ConfigCurrent },
  @{ Name = "current_repro1"; Dir = "outputs\grape_point_gppoint_detr_main_repro1"; Config = $ConfigCurrent },
  @{ Name = "current_seed2026"; Dir = "outputs\grape_point_gppoint_detr_main_seed2026"; Config = $ConfigCurrent }
)

foreach ($run in $DnRuns) {
  Train-DnRunIfNeeded -Name $run.Name -OutputDir $run.Dir -Seed $run.Seed
  Build-ReportIfNeeded -Name $run.Name -Config $ConfigDn -RunDir $run.Dir
  foreach ($split in @("valid", "test")) {
    Export-PredictionsIfNeeded -Name $run.Name -Config $ConfigDn -RunDir $run.Dir -Split $split
  }
}

foreach ($run in $CurrentRuns) {
  Build-ReportIfNeeded -Name $run.Name -Config $run.Config -RunDir $run.Dir
  foreach ($split in @("valid", "test")) {
    Export-PredictionsIfNeeded -Name $run.Name -Config $run.Config -RunDir $run.Dir -Split $split
  }
}

& $Python tools\dn_teacher_multiseed_calibration_report.py
if ($LASTEXITCODE -ne 0) {
  throw "multi-seed calibration report failed"
}
