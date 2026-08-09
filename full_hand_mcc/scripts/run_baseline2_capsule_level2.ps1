[CmdletBinding()]
param(
    [ValidateSet("Diagnostic", "Acceptance")]
    [string]$Mode = "Diagnostic",

    [ValidateRange(0, [int]::MaxValue)]
    [int]$Seed = 42
)

Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath(
    (Join-Path -Path $PSScriptRoot -ChildPath "..\..")
)
$pythonPath = Join-Path -Path $repoRoot -ChildPath ".venv\Scripts\python.exe"
$demoPath = Join-Path -Path $repoRoot -ChildPath "full_hand_mcc\scripts\demo_surface_slide.py"
$initialGraspPath = Join-Path -Path $repoRoot -ChildPath (
    "full_hand_mcc\assets\" +
    "fr3_capsule_100x170_bottom_grasp_high_clearance_v4.npz"
)
$debugDirectory = Join-Path -Path $repoRoot -ChildPath (
    "full_hand_mcc\outputs\debug\20_fr3_planning"
)

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot ".git"))) {
    throw "Repository root is invalid: $repoRoot"
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Validated project Python was not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $demoPath -PathType Leaf)) {
    throw "Baseline-2 demo was not found: $demoPath"
}
if (-not (Test-Path -LiteralPath $initialGraspPath -PathType Leaf)) {
    throw "Required Level-2 initial grasp was not found: $initialGraspPath"
}

New-Item -ItemType Directory -Path $debugDirectory -Force -ErrorAction Stop |
    Out-Null

$modeSlug = $Mode.ToLowerInvariant()
$runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$runStem = "baseline2_capsule_level2_{0}_seed{1}_{2}" -f (
    $modeSlug,
    $Seed,
    $runTimestamp
)
$logPath = Join-Path $debugDirectory ($runStem + ".log")
$planPath = Join-Path $debugDirectory ($runStem + "_plan.npz")
$failurePrefixPath = Join-Path $debugDirectory (
    $runStem + "_failure_prefix.npz"
)

# Both modes freeze the planner pad cone at 40 degrees. Diagnostic retains a
# 50-degree runtime audit with a 10-degree planning margin; Acceptance retains
# its 45-degree runtime audit with a 5-degree planning margin.
if ($Mode -eq "Acceptance") {
    $maxPadAngleDeg = "45"
    $plannerPadAngleMarginDeg = "5"
    $tangentialToleranceMm = "2"
    $maxRuntimeSelfPenetrationMm = "0.01"
    $finalContactRecoveryFrames = "50"
    $minTipSurfaceTravelM = "0.456"
    $minFingerJointExcursionRad = "0.08"
}
else {
    $maxPadAngleDeg = "50"
    $plannerPadAngleMarginDeg = "10"
    $tangentialToleranceMm = "3"
    $maxRuntimeSelfPenetrationMm = "0.05"
    $finalContactRecoveryFrames = "20"
    $minTipSurfaceTravelM = "0.45"
    $minFingerJointExcursionRad = "0.05"
}

$pythonArguments = @(
    "-B"
    "full_hand_mcc\scripts\demo_surface_slide.py"
    "--viewer", "headless"
    "--device", "cuda:0"
    "--seed", $Seed.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    "--planner-state-quantization-rad", "0.0005"
    "--object-shape", "capsule"
    "--object-radius-m", "0.10"
    "--object-half-height-m", "0.17"
    "--collision-mode", "full_robot"
    "--initial-grasp", "full_hand_mcc\assets\fr3_capsule_100x170_bottom_grasp_high_clearance_v4.npz"
    "--object-retreat-azimuth-deg", "-90"
    "--planner", "adaptive_surface_mpc"
    "--axial-travel-m", "0.48"
    "--axial-direction", "1"
    "--palm-travel-ratio", "1"
    "--palm-follow-surface-frame"
    "--palm-surface-frame-gain", "0.59"
    "--palm-surface-frame-late-gain", "0.54"
    "--palm-surface-frame-late-start-m", "0.045"
    "--palm-surface-frame-late-ramp-m", "0.010"
    "--palm-clearance-lift-m", "0.008"
    "--palm-clearance-ramp-m", "0.04"
    "--palm-clearance-use-local-normal"
    "--palm-clearance-secondary-lift-m", "0.0053"
    "--palm-clearance-secondary-start-m", "0.025"
    "--palm-clearance-secondary-ramp-m", "0.010"
    "--palm-terminal-local-offset-mm", "-1.55", "-0.07", "0.56"
    "--palm-terminal-local-offset-start-m", "0.0775"
    "--palm-terminal-local-offset-ramp-m", "0.0025"
    "--palm-terminal-second-local-offset-mm", "0", "0", "0.5"
    "--palm-terminal-second-local-offset-start-m", "0.0921"
    "--palm-terminal-second-local-offset-ramp-m", "0.0024"
    "--finger-meridian-gait-mm", "10"
    "--finger-meridian-gait-start-m", "0.025"
    "--finger-meridian-gait-end-m", "0.100"
    "--finger-meridian-gait-scales", "1", "0", "0", "0"
    "--finger-meridian-correction-mm", "1"
    "--finger-meridian-correction-start-m", "0.034"
    "--finger-meridian-correction-end-m", "0.060"
    "--finger-meridian-correction-scales", "0", "1", "0", "0"
    "--finger-meridian-terminal-correction-mm", "0.7"
    "--finger-meridian-terminal-correction-start-m", "0.09855"
    "--finger-meridian-terminal-correction-end-m", "0.1025"
    "--finger-meridian-terminal-correction-scales", "1", "0", "0", "0"
    "--finger-meridian-terminal-tail-correction-mm", "0.06"
    "--finger-meridian-terminal-tail-correction-start-m", "0.1017"
    "--finger-meridian-terminal-tail-correction-end-m", "0.1035"
    "--finger-meridian-terminal-tail-correction-scales", "1", "0", "0", "0"
    "--finger-meridian-local-phase", "0.16", "0.104", "0.1065", "1.5", "1", "0", "0"
    "--finger-meridian-local-phase", "0.14", "0.106", "0.109", "1", "0", "0", "0"
    "--finger-meridian-local-phase", "0.05", "0.1135", "0.116", "1.5", "1", "0", "0"
    "--mpc-keyframes", "385"
    "--mpc-local-refine-start-m", "0.0986"
    "--mpc-local-refine-end-m", "0.1015"
    "--mpc-local-refine-factor", "2"
    "--mpc-local-refine-window", "0.11595", "0.130", "2"
    "--mpc-auto-rephase-max-mm", "1.2"
    "--mpc-feasibility-bridge-max-mm", "1.5"
    "--mpc-feasibility-bridge-trust-radius-rad", "0.05"
    "--mpc-feasibility-bridge-min-progress-ratio", "0.10"
    "--mpc-feasibility-bridge-target-weight", "3200"
    "--mpc-feasibility-bridge-tip-target-scale", "0.5"
    "--mpc-static-bridge-max-dwell-mm", "1.50"
    "--mpc-static-bridge-max-total-ratio", "0.02"
    "--mpc-static-bridge-progress-tolerance-mm", "6.00"
    "--mpc-recovery-bridge-max-span-mm", "3.00"
    "--mpc-recovery-bridge-max-total-ratio", "0.03"
    "--mpc-recovery-bridge-progress-tolerance-mm", "6.00"
    "--mpc-recovery-bridge-normal-tolerance-mm", "6.50"
    "--mpc-recovery-bridge-min-contact-fingers", "2"
    "--mpc-recovery-bridge-terminal-margin-mm", "20"
    "--mpc-auto-rephase-step-mm", "0.05"
    "--mpc-auto-rephase-decay-mm", "0.02"
    "--mpc-auto-rephase-margin-mm", "0.5"
    "--mpc-auto-refine-min-step-mm", "0.15"
    "--mpc-auto-refine-max-insertions", "96"
    "--mpc-max-nfev", "300"
    "--mpc-progress-tolerance-mm", "4"
    "--mpc-intermediate-progress-tolerance-mm", "4"
    "--mpc-monotonic-tolerance-mm", "0.2"
    "--mpc-normal-tolerance-mm", "3"
    "--mpc-tangential-tolerance-mm", $tangentialToleranceMm
    "--min-planner-contact-fingers", "3"
    "--transient-contact-finger", "0"
    "--transient-contact-start-m", "0.025"
    "--transient-contact-end-m", "0.100"
    "--transient-contact-recovery-start-m", "0.085"
    "--transient-contact-normal-recovery-start-m", "0.085"
    "--mpc-transient-normal-tolerance-mm", "6.5"
    "--mpc-transient-tangential-tolerance-mm", "3.5"
    "--mpc-transient-progress-tolerance-mm", "7.5"
    "--palm-guide-only"
    "--palm-guide-max-drift-mm", "30"
    "--min-arm-clearance-mm", "2"
    "--max-incidental-hand-penetration-mm", "1"
    "--max-incidental-hand-contact-force-n", "24"
    "--max-incidental-hand-total-force-n", "36"
    "--max-pad-angle-deg", $maxPadAngleDeg
    "--planner-pad-angle-margin-deg", $plannerPadAngleMarginDeg
    "--planner-soft-pad-angle-deg", "35"
    "--planner-soft-pad-weight", "24"
    "--planner-soft-pad-softplus-tau", "0.02"
    "--planner-tip-geom-target-mm", "-0.25", "-0.25", "-0.50", "-0.25"
    "--planner-tip-geom-weight", "2200"
    "--planner-tip-geom-inner-cap-mm", "-0.8"
    "--planner-tip-geom-inner-weight", "18000"
    "--planner-protected-self-clearance-mm", "0.10"
    "--planner-protected-self-clearance-weight", "4000"
    "--planner-self-separation-seed-step-rad", "0.005"
    "--contact-failure-window", "20"
    "--min-contact-ratio", "0.75"
    "--min-runtime-contact-fingers", "3"
    "--min-majority-contact-ratio", "0.80"
    "--min-average-contact-fingers", "3.0"
    "--max-zero-contact-frames", "10"
    "--max-individual-contact-loss-frames", "20"
    "--final-contact-recovery-frames", $finalContactRecoveryFrames
    "--max-contact-penetration-mm", "1"
    "--max-runtime-self-penetration-mm", $maxRuntimeSelfPenetrationMm
    "--min-meridian-curvature-ratio", "2"
    "--min-tip-surface-travel-m", $minTipSurfaceTravelM
    "--min-tip-relative-travel-m", "0.004"
    "--min-finger-joint-excursion-rad", $minFingerJointExcursionRad
    "--motion-start", "600"
    "--steps", "4700"
    "--object-approach-frames", "250"
    "--finger-force-n", "3"
    "--finger-max-calibrated-force-n", "12"
    "--finger-admittance-mass-kg", "0.08"
    "--finger-admittance-damping-n-s-m", "18"
    "--finger-admittance-stiffness-n-m", "1000"
    "--finger-force-gain", "1"
    "--finger-force-filter-alpha", "0.25"
    "--finger-contact-on-force-n", "0.15"
    "--finger-contact-off-force-n", "0.08"
    "--finger-max-normal-offset-mm", "3"
    "--finger-max-normal-speed-mm-s", "10"
    "--finger-max-normal-acceleration-m-s2", "0.2"
    "--max-tip-contact-force-n", "25"
    "--max-tip-raw-force-n", "40"
    "--finger-servo-load-scale", "0.5"
    "--runtime-tip-gait-mm", "0"
    "--arm-mcc-correction-rad", "0.003"
    "--wrist-update-decimation", "4"
    "--wrist-damping-ratio", "1"
    "--wrist-max-force-error-n", "5"
    "--wrist-max-torque-error-nm", "0.8"
    "--wrist-max-translation-offset-mm", "3"
    "--wrist-max-rotation-offset-rad", "0.03"
    "--arm-trajectory-tracking-gain", "2"
    "--finger-trajectory-tracking-gain", "0.5"
    "--arm-servo-load-scale", "1"
    "--mpc-failure-prefix-output", $failurePrefixPath
    "--plan-output", $planPath
)

$gitCommit = "unknown"
try {
    $gitCommitOutput = & git -C $repoRoot rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $null -ne $gitCommitOutput) {
        $gitCommit = ($gitCommitOutput | Select-Object -First 1).Trim()
    }
}
catch {
    $gitCommit = "unknown"
}

function ConvertTo-PowerShellLiteral {
    param(
        [AllowEmptyString()]
        [string]$Value
    )

    return "'" + $Value.Replace("'", "''") + "'"
}

$copyableCommand = "& {0} {1}" -f (
    (ConvertTo-PowerShellLiteral -Value $pythonPath),
    (($pythonArguments | ForEach-Object {
        ConvertTo-PowerShellLiteral -Value $_
    }) -join " ")
)

$runHeader = @(
    "[BASELINE2-LEVEL2-RUNNER] mode=$Mode seed=$Seed viewer=headless"
    "[BASELINE2-LEVEL2-RUNNER] repo=$repoRoot"
    "[BASELINE2-LEVEL2-RUNNER] git_commit=$gitCommit"
    "[BASELINE2-LEVEL2-RUNNER] plan=$planPath"
    "[BASELINE2-LEVEL2-RUNNER] failure_prefix=$failurePrefixPath"
    "[BASELINE2-LEVEL2-RUNNER] command=$copyableCommand"
) -join [Environment]::NewLine

$pythonExitCode = -1
Push-Location -LiteralPath $repoRoot -ErrorAction Stop
try {
    $runHeader | Tee-Object -FilePath $logPath
    & $pythonPath @pythonArguments 2>&1 |
        Tee-Object -FilePath $logPath -Append
    $pythonExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($pythonExitCode -ne 0) {
    throw "Baseline-2 Level-2 $Mode run failed with exit code $pythonExitCode. See $logPath"
}

Write-Host "Baseline-2 Level-2 $Mode run passed. Log: $logPath"
