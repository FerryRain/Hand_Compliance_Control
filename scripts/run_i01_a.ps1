[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExperimentArgs
)
$ErrorActionPreference = 'Stop'
$RepoPath = Split-Path -Parent $PSScriptRoot
$RuntimePath = Join-Path $RepoPath '.runtime'
$env:PYTHONPATH = @($RuntimePath, (Join-Path $RuntimePath 'gpis_deps'), $RepoPath) -join ';'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$Python = Join-Path $RepoPath '.venv\Scripts\python.exe'
Push-Location $RepoPath
try {
    if ($ExperimentArgs[0] -eq 'report') {
        $ReportArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.i01_a_report @ReportArgs
    } elseif ($ExperimentArgs[0] -eq 'compare-wrists') {
        $CompareArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.i01_a_wrist_comparison @CompareArgs
    } elseif ($ExperimentArgs[0] -eq 'fair') {
        $FairArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.i01_a_fair @FairArgs
    } elseif ($ExperimentArgs[0] -eq 'fair-report') {
        $FairArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.i01_a_fair_report @FairArgs
    } elseif ($ExperimentArgs[0] -eq 'smooth') {
        $SmoothArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.i01_a_smooth @SmoothArgs
    } elseif ($ExperimentArgs[0] -eq 'smooth-report') {
        $SmoothArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.i01_a_smooth_report @SmoothArgs
    } else {
        & $Python -u -m Module.i04_geometric_traversal.i01_a @ExperimentArgs
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
