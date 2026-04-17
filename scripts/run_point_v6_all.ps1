[CmdletBinding()]
param(
    [ValidateSet("baseline_replay", "exp1", "exp2")]
    [string]$StartFrom = "baseline_replay",
    [switch]$ResumeLast,
    [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$steps = @("baseline_replay", "exp1", "exp2")
$expOutputMap = @{
    "exp1" = "outputs\grape_point_v6_exp1_instance_binding"
    "exp2" = "outputs\grape_point_v6_exp2_instance_binding_relative"
}
$startIndex = [array]::IndexOf($steps, $StartFrom)
if ($startIndex -lt 0) {
    throw "Unknown StartFrom value: $StartFrom"
}
$queue = $steps[$startIndex..($steps.Length - 1)]

foreach ($step in $queue) {
    if ($step -eq "baseline_replay") {
        $summaryPath = Join-Path $repoRoot "outputs\grape_point_v6_baseline_replay\report\summary.json"
        if ((-not $ForceRerun) -and (Test-Path -LiteralPath $summaryPath)) {
            Write-Host "[point_v6 all] baseline_replay already has summary.json, skipping."
            continue
        }
        Write-Host "[point_v6 all] running baseline_replay ..."
        & ".\\scripts\\make_point_v6_baseline_replay.ps1"
        continue
    }

    $expMap = @{
        "exp1" = "outputs\grape_point_v6_exp1_instance_binding\report\summary.json"
        "exp2" = "outputs\grape_point_v6_exp2_instance_binding_relative\report\summary.json"
    }
    $summaryPath = Join-Path $repoRoot $expMap[$step]
    if ((-not $ForceRerun) -and (Test-Path -LiteralPath $summaryPath)) {
        Write-Host "[point_v6 all] $step already has summary.json, skipping."
        continue
    }

    $lastRunFile = Join-Path $repoRoot ("scripts\.last_point_v6_{0}_run.txt" -f $step)
    $defaultRunDir = Join-Path $repoRoot $expOutputMap[$step]
    $resumeSwitch = $false
    if ($ResumeLast) {
        $candidateRoots = @()
        if (Test-Path -LiteralPath $lastRunFile) {
            $savedRun = (Get-Content -LiteralPath $lastRunFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
            if (-not [string]::IsNullOrWhiteSpace($savedRun)) {
                $candidateRoots += $savedRun
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($defaultRunDir)) {
            $candidateRoots += $defaultRunDir
        }
        $candidateRoots = $candidateRoots | Select-Object -Unique
        foreach ($root in $candidateRoots) {
            $lastCandidates = @(
                (Join-Path $root "checkpoints\last.pth"),
                (Join-Path $root "last.pth")
            )
            foreach ($candidate in $lastCandidates) {
                if (Test-Path -LiteralPath $candidate) {
                    $resumeSwitch = $true
                    break
                }
            }
            if ($resumeSwitch) {
                break
            }
        }
    }

    Write-Host "[point_v6 all] running $step ..."
    if ($resumeSwitch) {
        & ".\\scripts\\run_point_v6.ps1" -Experiment $step -ResumeLast
    } else {
        & ".\\scripts\\run_point_v6.ps1" -Experiment $step
    }
}

Write-Host "[point_v6 all] generating suite report ..."
& ".\\scripts\\make_point_v6_suite_report.ps1"
