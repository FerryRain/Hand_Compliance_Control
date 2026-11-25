# Run the final PARK controller in the same environment as its scored trials.
$parkArgs = @($args)
for ($i = 0; $i -lt $parkArgs.Count - 1; $i++) {
    if ($parkArgs[$i] -eq '--out') {
        $i++
        $parkArgs[$i] = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($parkArgs[$i])
    }
}
$parkRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (Test-Path -LiteralPath (Join-Path $parkRoot 'Module/palmreach6/__main__.py')) {
    Push-Location -LiteralPath $parkRoot
    try { & python -B -m PARK.run @parkArgs; $parkExit = $LASTEXITCODE }
    finally { Pop-Location }
    exit $parkExit
}
& (Join-Path $PSScriptRoot '../outputs/PARK/park-python.ps1') (Join-Path $PSScriptRoot 'run.py') @parkArgs
exit $LASTEXITCODE
