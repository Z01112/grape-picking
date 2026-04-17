# Point v4 report entry.
param(
    [ValidateSet("exp0", "exp1", "exp2", "exp3", "exp4")]
    [string]$Experiment = "exp0",
    [string]$RunDir = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$ExperimentMap = @{
    "exp0" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v4_exp0.yml"
        Output = "outputs\grape_point_v4_exp0_query_only"
        Label = "point_v4_exp0"
        Title = "point_v4 exp0 ????"
        Notes = @(
            "???? point_v2 ???????? ROI ???????",
            "????? point_v4 ?????????? locality penalty ? y-weight?"
        )
    }
    "exp1" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v4_exp1.yml"
        Output = "outputs\grape_point_v4_exp1_full_roi"
        Label = "point_v4_exp1_full"
        Title = "point_v4 exp1 ????"
        Notes = @(
            "? point_v2 ????? query + full ROI feature ????????",
            "??????????????? point ??????????"
        )
    }
    "exp2" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v4_exp2.yml"
        Output = "outputs\grape_point_v4_exp2_top_roi"
        Label = "point_v4_exp2_top"
        Title = "point_v4 exp2 ????"
        Notes = @(
            "? point_v2 ????? query + top ROI feature ????????",
            "???? point head ????? grape ???????????????"
        )
    }
    "exp3" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v4_exp3.yml"
        Output = "outputs\grape_point_v4_exp3_full_top_roi"
        Label = "point_v4_exp3_full_top"
        Title = "point_v4 exp3 ????"
        Notes = @(
            "? point_v2 ??????? full ROI ? top ROI feature?",
            "???????????????????????? point ??????????"
        )
    }
    "exp4" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v4_exp4.yml"
        Output = "outputs\grape_point_v4_exp4_full_top_roi_yw"
        Label = "point_v4_exp4_full_top_yw"
        Title = "point_v4 exp4 ????"
        Notes = @(
            "?? full+top ROI ???????? mild y-weight ??? locality penalty?",
            "??????? loss ?????????? point ???????????"
        )
    }
}

$spec = $ExperimentMap[$Experiment]
$PinnedRunDir = $spec.Output
$LastRunFile = Join-Path $repoRoot ("scripts\.last_point_v4_{0}_run.txt" -f $Experiment)

function Resolve-PointRunDir {
    param([string]$RunDirInput)
    if (-not [string]::IsNullOrWhiteSpace($RunDirInput)) {
        $raw = $RunDirInput.Trim().Trim('"')
        if ([System.IO.Path]::IsPathRooted($raw)) {
            return [System.IO.Path]::GetFullPath($raw)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $raw))
    }
    if (Test-Path -LiteralPath $LastRunFile) {
        $saved = (Get-Content -LiteralPath $LastRunFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if (-not [string]::IsNullOrWhiteSpace($saved) -and (Test-Path -LiteralPath $saved)) {
            return [System.IO.Path]::GetFullPath($saved)
        }
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PinnedRunDir))
}

function Save-LastRunDir {
    param([string]$ResolvedRunDir)
    Set-Content -LiteralPath $LastRunFile -Value $ResolvedRunDir -Encoding Ascii
}

function Move-RunItemIntoDir {
    param(
        [string]$SourcePath,
        [string]$DestinationDir,
        [string]$RunDirRoot
    )
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        return
    }
    $resolvedRoot = [System.IO.Path]::GetFullPath($RunDirRoot)
    $resolvedSource = [System.IO.Path]::GetFullPath($SourcePath)
    if (-not $resolvedSource.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to move an item outside the run directory: $resolvedSource"
    }
    New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
    $targetPath = Join-Path $DestinationDir (Split-Path -Leaf $SourcePath)
    if (Test-Path -LiteralPath $targetPath) {
        Remove-Item -LiteralPath $targetPath -Recurse -Force
    }
    Move-Item -LiteralPath $SourcePath -Destination $targetPath
}

function Sync-PointRunLayout {
    param([string]$ResolvedRunDir)
    $checkpointsDir = Join-Path $ResolvedRunDir 'checkpoints'
    $logsDir = Join-Path $ResolvedRunDir 'logs'
    New-Item -ItemType Directory -Path $checkpointsDir -Force | Out-Null
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

    $checkpointFiles = Get-ChildItem -LiteralPath $ResolvedRunDir -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^(best_stg1|best_stg2|best_grape_ap|best_has_picking_f1|best_point_l2|best_composite|last)\.pth$' -or
            $_.Name -match '^checkpoint\d+\.pth$'
        }
    foreach ($file in $checkpointFiles) {
        Move-RunItemIntoDir -SourcePath $file.FullName -DestinationDir $checkpointsDir -RunDirRoot $ResolvedRunDir
    }

    foreach ($name in @('log.txt', 'point_checkpoint_metrics.json')) {
        $source = Join-Path $ResolvedRunDir $name
        if (Test-Path -LiteralPath $source) {
            Move-RunItemIntoDir -SourcePath $source -DestinationDir $logsDir -RunDirRoot $ResolvedRunDir
        }
    }

    foreach ($dirName in @('summary', 'eval', 'eval_valid', 'eval_test')) {
        $sourceDir = Join-Path $ResolvedRunDir $dirName
        if (Test-Path -LiteralPath $sourceDir) {
            Move-RunItemIntoDir -SourcePath $sourceDir -DestinationDir $logsDir -RunDirRoot $ResolvedRunDir
        }
    }
}

$resolvedRunDir = Resolve-PointRunDir $RunDir
if (-not (Test-Path -LiteralPath $resolvedRunDir)) {
    throw "RunDir does not exist: $resolvedRunDir"
}

$command = @(
    ".\\.venv\\Scripts\\python.exe",
    "tools\\make_grape_point_v4_report.py",
    "--run-dir", $resolvedRunDir,
    "--config", $spec.Config,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--primary-label", $spec.Label,
    "--reference-label", "point_v2",
    "--report-title", $spec.Title,
    "--report-mode", "point_v4_full"
)

foreach ($note in $spec.Notes) {
    $command += @("--change-note", $note)
}

& $command[0] $command[1..($command.Length - 1)]

Sync-PointRunLayout -ResolvedRunDir $resolvedRunDir
Save-LastRunDir -ResolvedRunDir $resolvedRunDir
Write-Host "[point_v4 report][$Experiment] organized run dir: $resolvedRunDir"
