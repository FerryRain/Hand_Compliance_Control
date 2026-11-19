$ErrorActionPreference = 'Stop'
$TaskRoot = [IO.Path]::GetFullPath('D:\文档\Hand_Compliance_Control\i04_experiment_repo').TrimEnd('\')
$PlanPath = Join-Path $PSScriptRoot 'cleanup_plan.json'
$Plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([IO.Path]::GetFullPath($Plan.root).TrimEnd('\') -ne $TaskRoot) { throw 'Unexpected plan root' }
$Checks = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'before_native_checks.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($Checks.Count -ne 6 -or @($Checks | Where-Object { $_.returncode -ne 0 }).Count) { throw 'Native checks incomplete' }
$Integrity = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'verified_before.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($Integrity.failures.Count) { throw 'Retained data integrity failed' }
$Targets = foreach ($Relative in $Plan.delete_paths) {
    if ($Relative -notmatch '^(Module/(generated|i04_[^/]+|tests|evidence)(/|$)|Module/I04|scripts/|\.runtime_tmp(?:debug)?$)') { throw "Unexpected deletion scope $Relative" }
    $Resolved = (Resolve-Path -LiteralPath (Join-Path $TaskRoot $Relative)).ProviderPath.TrimEnd('\')
    if ($Resolved -eq $TaskRoot -or -not $Resolved.StartsWith($TaskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Path escapes workspace $Resolved" }
    $Item = Get-Item -LiteralPath $Resolved -Force
    if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point $Resolved" }
    foreach ($Keep in @($Plan.source_keep) + @($Plan.support_keep)) {
        $Protected = [IO.Path]::GetFullPath((Join-Path $TaskRoot $Keep))
        if ($Protected -eq $Resolved -or $Protected.StartsWith($Resolved + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Protected dependency $Protected" }
    }
    [PSCustomObject]@{ Relative = $Relative; Absolute = $Resolved }
}
$FreeBefore = ([IO.DriveInfo]::new('D')).AvailableFreeSpace
$LogPath = Join-Path $PSScriptRoot 'deletion_log.jsonl'
if (Test-Path -LiteralPath $LogPath) { throw 'Deletion log already exists; do not duplicate cleanup' }
$TaskTargets = foreach ($Target in $Targets) {
    if ($Target.Relative -eq '.runtime_tmp') {
        foreach ($Child in Get-ChildItem -LiteralPath $Target.Absolute -Force) {
            $ChildResolved = (Resolve-Path -LiteralPath $Child.FullName).ProviderPath
            if (-not $ChildResolved.StartsWith($Target.Absolute + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Child path escaped checked directory' }
            if ($Child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Child reparse point' }
            [PSCustomObject]@{ Relative = '.runtime_tmp/' + $Child.Name; Absolute = $ChildResolved }
        }
    }
    $Target
}
$Count = 0
foreach ($Target in $TaskTargets) {
    @{ status = 'DELETING_AUTHORIZED_I04_ARTIFACTS'; deleted_roots = $Count; total_roots = $TaskTargets.Count; current_path = $Target.Relative; free_bytes = ([IO.DriveInfo]::new('D')).AvailableFreeSpace; free_before = $FreeBefore } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Encoding utf8
    # Use .NET inside this one PowerShell process for millions of NTFS links.
    # Every target is resolved, bounded, and pre-scanned for reparse points.
    # There is no shell handoff, wildcard expansion, or computed external path.
    if ([IO.Directory]::Exists($Target.Absolute)) {
        [IO.Directory]::Delete($Target.Absolute, $true)
    } else {
        [IO.File]::Delete($Target.Absolute)
    }
    $Count += 1
    @{ path = $Target.Relative; deleted = $true; at_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Compress | Add-Content -LiteralPath $LogPath -Encoding utf8
    if ($Count % 100 -eq 0) { Write-Output "Deleted $Count of $($TaskTargets.Count) paths" }
}
$Remaining = @($Targets | Where-Object { Test-Path -LiteralPath $_.Absolute })
if ($Remaining.Count) { throw 'Some deletion targets remain' }
@{ status = 'DELETION_COMPLETE_PENDING_FINAL_VERIFICATION'; deleted_roots = $Count; free_before = $FreeBefore; free_after = ([IO.DriveInfo]::new('D')).AvailableFreeSpace; logical_files_deleted = $Plan.delete_files; logical_bytes_deleted = $Plan.logical_bytes } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_status.json') -Raw
