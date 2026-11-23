[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExperimentArgs
)
$ErrorActionPreference = 'Stop'
$RepoPath = Split-Path -Parent $PSScriptRoot
if (!$Python) {
    $Python = Join-Path $RepoPath '.venv\Scripts\python.exe'
}
$RuntimePath = Join-Path $RepoPath '.runtime'
$env:PYTHONPATH = @($RuntimePath, $RepoPath, (Join-Path $RuntimePath 'gpis_deps')) -join ';'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
Push-Location $RepoPath
try {
    if ($ExperimentArgs.Count -gt 0 -and $ExperimentArgs[0] -eq 'route-free') {
        $BaselineArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -Bu -m Module.acitive_02.route_free @BaselineArgs
    } elseif ($ExperimentArgs.Count -gt 0 -and $ExperimentArgs[0] -eq 'active-a') {
        $ActiveArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -Bu -m Module.acitive_02.active_a @ActiveArgs
    } elseif ($ExperimentArgs.Count -gt 0 -and $ExperimentArgs[0] -eq 'active-timing') {
        $TimingArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -Bu -m Module.acitive_02.active_timing @TimingArgs
    } elseif ($ExperimentArgs.Count -gt 0 -and $ExperimentArgs[0] -eq 'active-review') {
        $ReviewArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -Bu -m Module.acitive_02.active_review @ReviewArgs
    } elseif ($ExperimentArgs.Count -ge 2 -and $ExperimentArgs[0] -eq 'full' -and $ExperimentArgs[1] -eq 'cooperative') {
        $FullArgs = @($ExperimentArgs | Select-Object -Skip 2)
        & $Python -Bu -m Module.acitive_02.cooperative @FullArgs
    } elseif ($ExperimentArgs.Count -ge 2 -and $ExperimentArgs[0] -eq 'full' -and $ExperimentArgs[1] -in @('refine', 'signed')) {
        $FullModule = 'Module.acitive_02.full_' + $ExperimentArgs[1]
        $FullArgs = @($ExperimentArgs | Select-Object -Skip 2)
        & $Python -Bu -m $FullModule @FullArgs
    } elseif ($ExperimentArgs.Count -gt 0 -and $ExperimentArgs[0] -eq 'full') {
        $FullArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -Bu -m Module.acitive_02.full @FullArgs
    } else {
        & $Python -Bu -m Module.acitive_02.experiment @ExperimentArgs
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
