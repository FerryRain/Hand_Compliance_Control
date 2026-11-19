$ErrorActionPreference = 'Stop'
$TaskRoot = [IO.Path]::GetFullPath('D:\文档\Hand_Compliance_Control\i04_experiment_repo').TrimEnd('\')
$Plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
$Previous = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ([IO.Path]::GetFullPath($Plan.root).TrimEnd('\') -ne $TaskRoot) { throw 'Unexpected root' }
$LogPath = Join-Path $PSScriptRoot 'deletion_log.jsonl'
$Done = @(Get-Content -LiteralPath $LogPath | ForEach-Object { $_ | ConvertFrom-Json })
$Checks = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'before_native_checks.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($Checks.Count -ne 6 -or @($Checks | Where-Object { $_.returncode -ne 0 }).Count) { throw 'Native checks incomplete' }
$Targets = foreach ($Relative in $Plan.delete_paths) {
    if ($Relative -notmatch '^(Module/(generated|i04_[^/]+|tests|evidence)(/|$)|Module/I04|scripts/|\.runtime_tmp(?:debug)?$)') { throw 'Unexpected scope' }
    $Candidate = Join-Path $TaskRoot $Relative
    if (-not (Test-Path -LiteralPath $Candidate)) { continue }
    $Resolved = (Resolve-Path -LiteralPath $Candidate).ProviderPath.TrimEnd('\')
    if ($Resolved -eq $TaskRoot -or -not $Resolved.StartsWith($TaskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Root escape' }
    if ((Get-Item -LiteralPath $Resolved -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Root reparse point' }
    foreach ($Keep in @($Plan.source_keep) + @($Plan.support_keep)) {
        $Protected = [IO.Path]::GetFullPath((Join-Path $TaskRoot $Keep))
        if ($Protected -eq $Resolved -or $Protected.StartsWith($Resolved + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Protected path $Protected" }
    }
    if ($Relative -eq '.runtime_tmp') {
        foreach ($Child in Get-ChildItem -LiteralPath $Resolved -Force) {
            $ChildResolved = (Resolve-Path -LiteralPath $Child.FullName).ProviderPath
            if (-not $ChildResolved.StartsWith($Resolved + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Child escape' }
            if ($Child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Child reparse point' }
            [PSCustomObject]@{ Relative = '.runtime_tmp/' + $Child.Name; Absolute = $ChildResolved }
        }
    } else {
        [PSCustomObject]@{ Relative = $Relative; Absolute = $Resolved }
    }
}
# All targets are disjoint roots. The common .runtime_tmp parent is removed
# only after this pipeline joins. No worker enumerates another worker's root.
$Count = $Done.Count
$Total = $Count + $Targets.Count + 1
$Failures = [Collections.Generic.List[object]]::new()
$Targets | ForEach-Object -ThrottleLimit 6 -Parallel {
    $Target = $_
    try {
        $Bound = $using:TaskRoot
        $Full = [IO.Path]::GetFullPath($Target.Absolute)
        if ($Full -eq $Bound -or -not $Full.StartsWith($Bound + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Worker path escape' }
        if ([IO.Directory]::Exists($Full)) { [IO.Directory]::Delete($Full, $true) }
        else { [IO.File]::Delete($Full) }
        [PSCustomObject]@{ path = $Target.Relative; deleted = $true; at_utc = [DateTime]::UtcNow.ToString('o') }
    } catch {
        [PSCustomObject]@{ path = $Target.Relative; deleted = $false; error = $_.Exception.Message }
    }
} | ForEach-Object {
    if (-not $_.deleted) { $Failures.Add($_) }
    else { $Count += 1 }
    $_ | ConvertTo-Json -Compress | Add-Content -LiteralPath $LogPath -Encoding utf8
    @{ status = 'DELETING_DISJOINT_AUTHORIZED_I04_ROOTS'; deleted_roots = $Count; total_roots = $Total; last_path = $_.path; free_bytes = ([IO.DriveInfo]::new('D')).AvailableFreeSpace; free_before = $Previous.free_before; workers = 6; failures = $Failures.Count } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Encoding utf8
    if ($Count % 100 -eq 0) { Write-Output "Deleted $Count of $Total paths" }
}
if ($Failures.Count) { $Failures | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_failures.json') -Encoding utf8; throw 'Some cleanup targets failed' }
$RuntimeRoot = Join-Path $TaskRoot '.runtime_tmp'
if (Test-Path -LiteralPath $RuntimeRoot) {
    $Resolved = (Resolve-Path -LiteralPath $RuntimeRoot).ProviderPath
    if ($Resolved -ne ($TaskRoot + '\.runtime_tmp')) { throw 'Unexpected parent path' }
    [IO.Directory]::Delete($Resolved, $false)
    @{ path = '.runtime_tmp'; deleted = $true; at_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Compress | Add-Content -LiteralPath $LogPath -Encoding utf8
    $Count += 1
}
$Remaining = @($Plan.delete_paths | Where-Object { Test-Path -LiteralPath (Join-Path $TaskRoot $_) })
if ($Remaining.Count) { throw 'Planned paths remain' }
@{ status = 'DELETION_COMPLETE_PENDING_FINAL_VERIFICATION'; deleted_roots = $Count; free_before = $Previous.free_before; free_after = ([IO.DriveInfo]::new('D')).AvailableFreeSpace; logical_files_deleted = $Plan.delete_files; logical_bytes_deleted = $Plan.logical_bytes } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Raw
