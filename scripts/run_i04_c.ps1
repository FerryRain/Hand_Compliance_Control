[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = 'D:\Environment\Anaconda\envs\handcomp\python.exe',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExperimentArgs
)
$ErrorActionPreference = 'Stop'
$RepoPath = Split-Path -Parent $PSScriptRoot
$RuntimePath = Join-Path $RepoPath '.runtime'
$PythonPaths = @($RuntimePath, $RepoPath)
$IsBenchmark = $ExperimentArgs.Count -gt 0 -and $ExperimentArgs[0] -eq 'benchmark'
if ($IsBenchmark -or $ExperimentArgs[0] -eq 'compare-surface') {
    $PythonPaths += Join-Path $RuntimePath 'gpis_deps'
}
$env:PYTHONPATH = $PythonPaths -join ';'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
Push-Location $RepoPath
try {
    if ($IsBenchmark) {
        if (-not $PSBoundParameters.ContainsKey('Python')) {
            $Python = Join-Path $RepoPath '.venv\Scripts\python.exe'
        }
        $BenchmarkArgs = @($ExperimentArgs | Select-Object -Skip 1)
        & $Python -u -m Module.i04_geometric_traversal.fingertip_benchmark @BenchmarkArgs
    } else {
        & $Python -u -m Module.i04_geometric_traversal.fingertip_traversal @ExperimentArgs
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
