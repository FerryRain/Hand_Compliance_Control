param(
    [string]$Python = 'D:\Environment\Anaconda\envs\handcomp\python.exe',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExperimentArgs
)
$ErrorActionPreference = 'Stop'
$RepoPath = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = (Join-Path $RepoPath '.runtime') + ';' + $RepoPath
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
Push-Location $RepoPath
try {
    & $Python -m Module.i04_geometric_traversal.benchmark @ExperimentArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
