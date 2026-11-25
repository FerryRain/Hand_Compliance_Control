# Run a module in the workspace's verified PalmReach-6 release and CUDA runtime.
$parkWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$parkRelease = Join-Path $parkWorkspace '.tmp/active02-main-source-20261123'
$parkEnvironment = Join-Path $parkWorkspace 'i04_experiment_repo'
$parkPython = Join-Path $parkEnvironment '.venv/Scripts/python.exe'
$parkRuntime = Join-Path $parkEnvironment '.runtime'
$parkVars = @{
    PYTHONPATH = "$parkRuntime;$parkRuntime/gpis_deps;$parkRelease"
    PYTHONUTF8 = '1'
    PYTHONDONTWRITEBYTECODE = '1'
    OPENBLAS_NUM_THREADS = '1'
    OMP_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
    NUMEXPR_NUM_THREADS = '1'
}
$parkPrior = @{}
Push-Location -LiteralPath $parkRelease
try {
    foreach ($key in $parkVars.Keys) {
        $parkPrior[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $parkVars[$key], 'Process')
    }
    & $parkPython -B -u @args
    $parkExit = $LASTEXITCODE
} finally {
    foreach ($key in $parkVars.Keys) {
        [Environment]::SetEnvironmentVariable($key, $parkPrior[$key], 'Process')
    }
    Pop-Location
}
exit $parkExit
