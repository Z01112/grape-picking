[CmdletBinding()]
param(
    [ValidateSet("exp1", "exp2", "exp3")]
    [string]$Experiment = "exp1",
    [string]$OutputDir = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [int]$Epochs = -1,
    [int]$ExtraEpochs = 0,
    [switch]$SkipReport,
    [switch]$ResumeLast,
    [string]$ResumeCheckpoint = "",
    [string[]]$ConfigUpdates = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$ExperimentMap = @{
    "exp1" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v5_exp1.yml"
        Output = "outputs\grape_point_v5_exp1_dense"
    }
    "exp2" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v5_exp2.yml"
        Output = "outputs\grape_point_v5_exp2_geo"
    }
    "exp3" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v5_exp3.yml"
        Output = "outputs\grape_point_v5_exp3_dense_geo"
    }
}

$spec = $ExperimentMap[$Experiment]
$PinnedOutputDir = $spec.Output
$ConfigPath = Join-Path $repoRoot $spec.Config
$LastRunFile = Join-Path $repoRoot ("scripts\.last_point_v5_{0}_run.txt" -f $Experiment)

function Save-LastRunDir {
    param([string]$RunDir)
    Set-Content -LiteralPath $LastRunFile -Value $RunDir -Encoding Ascii
}

function Resolve-OutputDir {
    param([string]$OutputDirInput)
    if (-not [string]::IsNullOrWhiteSpace($OutputDirInput)) {
        $raw = $OutputDirInput.Trim().Trim('"')
        if ([System.IO.Path]::IsPathRooted($raw)) {
            return [System.IO.Path]::GetFullPath($raw)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $raw))
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PinnedOutputDir))
}

function Resolve-ResumeCheckpoint {
    param(
        [string]$OutputRoot,
        [switch]$UseLast,
        [string]$ResumeCheckpointInput
    )
    if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpointInput)) {
        $raw = $ResumeCheckpointInput.Trim().Trim('"')
        if ([System.IO.Path]::IsPathRooted($raw)) {
            return [System.IO.Path]::GetFullPath($raw)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $raw))
    }
    if ($UseLast) {
        foreach ($candidate in @(
            (Join-Path $OutputRoot 'checkpoints\last.pth'),
            (Join-Path $OutputRoot 'last.pth')
        )) {
            if (Test-Path -LiteralPath $candidate) {
                return [System.IO.Path]::GetFullPath($candidate)
            }
        }
        throw "Resume requested, but last.pth was not found under: $OutputRoot"
    }
    return ""
}

function Resolve-RunLogPath {
    param([string]$OutputRoot)
    foreach ($candidate in @(
        (Join-Path $OutputRoot 'logs\log.txt'),
        (Join-Path $OutputRoot 'log.txt')
    )) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    return ""
}

function Get-ConfiguredEpochs {
    param([string]$ConfigPathInput)
    $content = Get-Content -LiteralPath $ConfigPathInput -Raw
    $match = [regex]::Match($content, '(?m)^\s*epoches\s*:\s*(\d+)\s*$')
    if (-not $match.Success) {
        throw "Could not find 'epoches' in config: $ConfigPathInput"
    }
    return [int]$match.Groups[1].Value
}

function Get-LastEpochFromLog {
    param([string]$LogPath)
    if ([string]::IsNullOrWhiteSpace($LogPath) -or -not (Test-Path -LiteralPath $LogPath)) {
        return -1
    }
    $lastEpoch = -1
    foreach ($line in Get-Content -LiteralPath $LogPath) {
        $match = [regex]::Match($line, '"epoch"\s*:\s*(\d+)')
        if ($match.Success) {
            $lastEpoch = [int]$match.Groups[1].Value
        }
    }
    return $lastEpoch
}

function Resolve-TargetEpochs {
    param(
        [int]$EpochsInput,
        [int]$ExtraEpochsInput,
        [string]$OutputRoot,
        [string]$ConfigPathInput
    )
    if ($EpochsInput -gt 0 -and $ExtraEpochsInput -gt 0) {
        throw "Use either -Epochs or -ExtraEpochs, not both."
    }
    if ($EpochsInput -gt 0) {
        return $EpochsInput
    }
    if ($ExtraEpochsInput -le 0) {
        return -1
    }
    $configuredEpochs = Get-ConfiguredEpochs -ConfigPathInput $ConfigPathInput
    $logPath = Resolve-RunLogPath -OutputRoot $OutputRoot
    $lastEpoch = Get-LastEpochFromLog -LogPath $logPath
    if ($lastEpoch -ge 0) {
        return ($lastEpoch + 1 + $ExtraEpochsInput)
    }
    return ($configuredEpochs + $ExtraEpochsInput)
}

$resolvedOutputDir = Resolve-OutputDir $OutputDir
$resolvedResumeCheckpoint = Resolve-ResumeCheckpoint -OutputRoot $resolvedOutputDir -UseLast:$ResumeLast -ResumeCheckpointInput $ResumeCheckpoint
$resolvedTargetEpochs = Resolve-TargetEpochs -EpochsInput $Epochs -ExtraEpochsInput $ExtraEpochs -OutputRoot $resolvedOutputDir -ConfigPathInput $ConfigPath

Write-Host "[point_v5][$Experiment] output dir: $resolvedOutputDir"
Write-Host "[point_v5][$Experiment] config: $ConfigPath"
if (-not [string]::IsNullOrWhiteSpace($resolvedResumeCheckpoint)) {
    Write-Host "[point_v5][$Experiment] resume checkpoint: $resolvedResumeCheckpoint"
}
if ($resolvedTargetEpochs -gt 0) {
    Write-Host "[point_v5][$Experiment] target epochs: $resolvedTargetEpochs"
}

$command = @(
    ".\\.venv\\Scripts\\python.exe",
    "tools\\run_grape_point_baseline.py",
    "all",
    "--dataset-root", "dataset",
    "--config", $spec.Config,
    "--output-dir", $resolvedOutputDir,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers
)

if (-not [string]::IsNullOrWhiteSpace($resolvedResumeCheckpoint)) {
    $command += @("--resume", $resolvedResumeCheckpoint)
}
if ($resolvedTargetEpochs -gt 0) {
    $command += @("--epochs", $resolvedTargetEpochs)
}
if ($ConfigUpdates.Count -gt 0) {
    $command += @("--config-update")
    $command += $ConfigUpdates
}

& $command[0] $command[1..($command.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "[point_v5][$Experiment] training failed with exit code $LASTEXITCODE"
}

Save-LastRunDir $resolvedOutputDir

if (-not $SkipReport) {
    Write-Host "[point_v5][$Experiment] training finished, generating report..."
    & ".\\scripts\\make_point_v5_report.ps1" `
        -Experiment $Experiment `
        -RunDir $resolvedOutputDir `
        -Device $Device `
        -BatchSize $BatchSize `
        -NumWorkers $NumWorkers
}
