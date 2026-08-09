# FR3 + LEAP whole-hand optimization baseline

This directory implements **Baseline 2** from
[`../PROPOSAL.md`](../PROPOSAL.md): an explicit optimizer plans the wrist and
four fingertips, and the low-level controller executes that plan with Wrist
MCC plus four fingertip-force MCC loops. It is not the proposed main method's
online controller. The main method uses inverse demonstrations, a
wrist-conditioned finger Diffusion Policy, and wrist-only ER-GPIS planning.

The authoritative comparison and sensor semantics are in
[`../CONTROL_STRATEGIES.md`](../CONTROL_STRATEGIES.md).
The frozen Level 1–5 numerical, timing, object-generalization, and video gates
are in [`BASELINE2_ACCEPTANCE.md`](BASELINE2_ACCEPTANCE.md). A run is successful
only at the highest level whose complete gate it passes.

## Scope

The upper optimizer jointly plans five kinematic points under the assembled
Franka FR3 + LEAP Hand model and its joint, reachability, smoothness, and
collision constraints:

1. palm-root guide point;
2. index fingertip pad;
3. middle fingertip pad;
4. ring fingertip pad;
5. thumb fingertip pad.

The palm-root point provides a coarse arm direction and need not contact the
object. The four physical fingertip pads are planned and audited more strictly.
Brief fingertip release or a brief pause is allowed, but most motion frames
must retain majority fingertip contact and the controller must restore stable
support. LEAP Hand contact outside the pads is allowed only within configured
force and penetration limits and never counts as fingertip contact. FR3/object
contact is always a hard failure.

## Low-level controller

### Four Finger MCC loops

Each finger loop runs at 100 Hz by default:

1. read that fingertip's direct 3-D physical force measurement;
2. transform it into the current control frame and calibrate its sign;
3. project it onto the planned local surface normal;
4. integrate one scalar virtual mass-damping-stiffness state;
5. add only the bounded normal offset to the planned Cartesian tip point;
6. solve the four-site finger IK and send bounded LEAP joint commands.

For outward surface normal \(n_i\), inward compliance offset \(x_i\), direct
normal force \(f_i\), and loaded force target \(f_i^*\):

```text
M_i x_ddot_i + B_i x_dot_i + K_i x_i = K_f (f_i* - f_i)
p_tip_cmd_i = p_tip_plan_i - x_i n_i
```

The force loop does not modify the tangential component of the optimizer's
trajectory. Motor loads are recorded only for actuator diagnostics and safety;
they are not the primary contact-force source.

The direct-force module is self-contained. Old task configurations,
motor-force compatibility APIs, and the five historical controller variants
have been removed. The active demo exposes one Baseline-2 controller and its
help contains no `--variant`; read-only motor-load diagnostics remain.

After stable four-pad contact and immediately before motion, each direct normal
force is captured as that finger's loaded operating setpoint. The setpoint is
bounded by `--finger-force-n` and `--finger-max-calibrated-force-n` (currently
3-12 N by default). Filtering, contact hysteresis, acceleration/speed/offset
limits, joint limits, and saturation anti-windup remain active.

### Wrist MCC

The wrist loop runs at 25 Hz by default. It estimates a 6-D wrist wrench from
the seven FR3 external joint torques and the current arm Jacobian, subtracts
the loaded calibration wrench, integrates a bounded Cartesian admittance
reference, and maps that reference to a small seven-joint correction with
damped least squares. Its lower bandwidth and smaller correction limits reduce
conflict with the four local finger loops.

MCC changes the reference pose in response to external force. The simulated
robot still executes position commands through its inner position servos.

## Safety and acceptance

This section is a quick operational summary. The authoritative thresholds,
measurement definitions, object matrix, headless-first sequence, and visual
review checklist are in
[`BASELINE2_ACCEPTANCE.md`](BASELINE2_ACCEPTANCE.md).

The physical audit must report at least:

- per-finger contact ratio, majority-contact ratio, average simultaneous
  contacts, longest loss interval, and terminal `4/4` recovery;
- planned and executed surface progress for every fingertip;
- filtered normal-force peaks and raw 3-D force peaks;
- fingertip and incidental-hand penetration/force;
- FR3/object contact, LEAP self-collision, pad orientation, and joint margin;
- planner completion and runtime stability.

The current two-level fingertip force guard uses a 25 N default hard limit on
the filtered normal force and a separate 40 N default emergency cutoff on any
raw 3-D sample. Neither threshold converts a failed run into a success; both
peaks remain visible in the final summary.

Use `--viewer headless` first. It performs planning, dynamics, contact, travel,
force, collision, and terminal-recovery checks without spending time encoding
a video. Re-run the exact accepted plan with `--viewer video` only after the
full numerical audit passes, then visually inspect fingertip-pad contact,
thumb visibility, arm clearance, penetration, route coverage, and playback
speed before placing anything in `outputs/deliverables/fr3/`.

## Windows environment

Continue using the validated repository virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

The planned migration to the `handcomp` Conda environment is deferred until
the demo is complete.

## Run and verify

From the repository root, inspect the CLI and run the tests:

```powershell
.\.venv\Scripts\python.exe -B `
  full_hand_mcc\scripts\demo_surface_slide.py --help

.\.venv\Scripts\python.exe -B -m unittest discover `
  -s full_hand_mcc\tests -v
```

Run the requested 0.48 m bottom-to-top case numerically before recording it:

```powershell
.\.venv\Scripts\python.exe -B `
  full_hand_mcc\scripts\demo_surface_slide.py `
  --viewer headless --device cuda:0 --seed 42 `
  --object-shape capsule `
  --object-radius-m 0.10 --object-half-height-m 0.17 `
  --collision-mode full_robot `
  --initial-grasp full_hand_mcc\assets\fr3_capsule_100x170_bottom_grasp_high_clearance_v4.npz `
  --planner adaptive_surface_mpc `
  --axial-travel-m 0.48 --axial-direction 1 `
  --palm-guide-only `
  --object-retreat-azimuth-deg -90 `
  --finger-force-n 3 `
  --finger-max-calibrated-force-n 12 `
  --finger-admittance-mass-kg 0.08 `
  --finger-admittance-damping-n-s-m 18 `
  --finger-admittance-stiffness-n-m 1000 `
  --finger-max-normal-offset-mm 3 `
  --max-tip-contact-force-n 25 `
  --max-tip-raw-force-n 40 `
  --arm-mcc-correction-rad 0.003 `
  --wrist-update-decimation 4 `
  --motion-start 350 --steps 4700
```

This command is an acceptance target, not a claim that the complete route
already passes. If planning or dynamics fails, keep the output as diagnosis,
update [`PROCESS.md`](PROCESS.md), and do not produce a delivery video.

## Important files

| Path | Purpose |
| --- | --- |
| `scripts/demo_surface_slide.py` | Planning, simulation, contact/collision/force audit, and optional rendering entry point |
| `../src/mjlab/tasks/leaphand/full_hand_mcc_core.py` | Pure fingertip and wrist admittance dynamics |
| `../src/mjlab/tasks/leaphand/leaphand_direct_force_env.py` | Self-contained LEAP constants, model/sensor construction, and read-only motor-load diagnostics used by the FR3 baseline |
| `../src/mjlab/tasks/leaphand/leaphand_full_hand_mcc_env_cfg.py` | FR3 + LEAP sensor transforms, IK/Jacobian mapping, and runtime controller integration |
| `tests/` | Numerical, data-flow, contact-policy, and source-structure regressions |
| `BASELINE2_ACCEPTANCE.md` | Project-level Level 1–5 success gates and required reports |
| `PROCESS.md` | Detailed chronological work log and current next steps |

## Verified boundary and remaining work

The cleaned direct-force tree is `PASS-NUMERICAL-L1`:

- all unittests pass (`17/17`);
- demo, grasp-search, and grasp-optimization CLI checks exit 0; demo help has
  no old `--variant`;
- the 5 mm/750-step CUDA headless smoke exits 0 with contact ratios
  `[0.9975,1.0,1.0,0.99]`, majority ratio `1.0`, average `3.9875/4`, minimum
  `3/4`, loss streaks `[1,0,0,1]`, and 65 terminal all-contact frames;
- raw force peaks are `[14.936,10.963,26.132,9.721] N`; filtered normal peaks
  are `[13.158,8.862,20.480,8.029] N`;
- FR3/object contact, self penetration, tip penetration, and incidental hand
  contact are all zero; maximum pad angle is `41.44 deg`; travel is `5 mm`;
- no video was generated.

This Level-1 smoke does not validate, and the following remain `NOT RUN`:

- the complete 0.48 m route or contact at the object top;
- visually continuous fingertip-pad sliding over the whole object;
- strongly varying curvature or multiple object families;
- Baseline 2A oracle versus 50/100/200 ms Baseline 2B timing;
- hardware force calibration and real-system safety.

These are active requirements, not optional extensions. Current work and
remaining blockers are tracked in [`PROCESS.md`](PROCESS.md) and
[GitHub issue #7](https://github.com/FerryRain/Hand_Compliance_Control/issues/7).
