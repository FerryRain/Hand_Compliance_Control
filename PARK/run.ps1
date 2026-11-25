# Launch PARK in a release checkout or in the original Windows collection workspace.
$parkArgs = @($args)
for ($i = 0; $i -lt $parkArgs.Count - 1; $i++) {
    if ($parkArgs[$i] -eq '--out') {
        $i++
        $parkArgs[$i] = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($parkArgs[$i])
    }
}
$parkRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$parkNative = Join-Path $parkRoot '.tmp/active02-main-source-20261123'
$parkEnv = Join-Path $parkRoot 'i04_experiment_repo'
$parkInterpreter = 'python'
$parkVars = @{}
if (!(Test-Path -LiteralPath (Join-Path $parkRoot 'Module/palmreach6/__main__.py'))) {
    $parkInterpreter = Join-Path $parkEnv '.venv/Scripts/python.exe'
    $parkRuntime = Join-Path $parkEnv '.runtime'
    $parkVars = @{
        PYTHONPATH = "$parkRoot;$parkRuntime;$parkRuntime/gpis_deps;$parkNative"
        OPENBLAS_NUM_THREADS = '1'
        OMP_NUM_THREADS = '1'
        MKL_NUM_THREADS = '1'
        NUMEXPR_NUM_THREADS = '1'
        PYTHONUTF8 = '1'
    }
}
$parkPrior = @{}
Push-Location -LiteralPath $parkRoot
try {
    foreach ($key in $parkVars.Keys) {
        $parkPrior[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $parkVars[$key], 'Process')
    }
    & $parkInterpreter -B -m PARK @parkArgs
    $parkExit = $LASTEXITCODE
} finally {
    foreach ($key in $parkPrior.Keys) {
        [Environment]::SetEnvironmentVariable($key, $parkPrior[$key], 'Process')
    }
    Pop-Location
}
exit $parkExit
