[CmdletBinding()]
param(
    [ValidateSet("baseline_replay", "exp1", "exp2")]
    [string]$Experiment = "exp1",
    [string]$RunDir = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ($Experiment -eq "baseline_replay") {
    & ".\\scripts\\make_point_v6_baseline_replay.ps1" `
        -RunDir $RunDir `
        -Device $Device `
        -BatchSize $BatchSize `
        -NumWorkers $NumWorkers
    exit 0
}

$ExperimentMap = @{
    "exp1" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v6_exp1.yml"
        Output = "outputs\grape_point_v6_exp1_instance_binding"
        Label = "point_v6_exp1_instance_binding"
        Title = "point_v6 exp1 中文结论"
        Notes = @(
            "Based on point_v2 and add explicit query-conditioned instance binding in the has_picking / point heads.",
            "Do not change the point coordinate parameterization in exp1."
        )
    }
    "exp2" = @{
        Config = "configs\rtv4\rtv4_hgnetv2_s_grape_point_v6_exp2.yml"
        Output = "outputs\grape_point_v6_exp2_instance_binding_relative"
        Label = "point_v6_exp2_instance_binding_relative"
        Title = "point_v6 exp2 中文结论"
        Notes = @(
            "Based on exp1 and switch point regression to bbox-relative normalized coordinates.",
            "Decode the normalized point back to image pixel coordinates only at inference / evaluation time."
        )
    }
}

$spec = $ExperimentMap[$Experiment]
$PinnedRunDir = $spec.Output
$LastRunFile = Join-Path $repoRoot ("scripts\.last_point_v6_{0}_run.txt" -f $Experiment)
$BaselineReplaySummary = Join-Path $repoRoot "outputs\grape_point_v6_baseline_replay\report\summary.json"
$FallbackPointV2Summary = Join-Path $repoRoot "outputs\grape_point_v2_main\report\summary.json"

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

$referenceSummary = $FallbackPointV2Summary
$referenceLabel = "point_v2"
if (Test-Path -LiteralPath $BaselineReplaySummary) {
    $referenceSummary = $BaselineReplaySummary
    $referenceLabel = "baseline_replay"
}

$command = @(
    ".\\.venv\\Scripts\\python.exe",
    "tools\\make_grape_point_v6_report.py",
    "--run-dir", $resolvedRunDir,
    "--config", $spec.Config,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--primary-label", $spec.Label,
    "--reference-label", $referenceLabel,
    "--point-v2-summary", $referenceSummary,
    "--report-title", $spec.Title,
    "--report-mode", "point_v6_full"
)

foreach ($note in $spec.Notes) {
    $command += @("--change-note", $note)
}

& $command[0] $command[1..($command.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "[point_v6 report][$Experiment] report generation failed with exit code $LASTEXITCODE"
}

Sync-PointRunLayout -ResolvedRunDir $resolvedRunDir
Save-LastRunDir -ResolvedRunDir $resolvedRunDir
Write-Host "[point_v6 report][$Experiment] organized run dir: $resolvedRunDir"
