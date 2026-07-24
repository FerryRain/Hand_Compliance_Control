# Full-hand MCC surface-sliding demo

This directory contains the Windows/CUDA MJLab demo for full-hand Minimalist
Compliance Control (MCC). The xArm6 moves the hand along a reachable trajectory,
while all 16 LEAP Hand motor-load channels provide four independent fingertip
force estimates.

The five planning points are ordered as:

1. palm-root planning site (kinematic only; physical palm contact is not required);
2. index fingertip;
3. middle fingertip;
4. ring fingertip;
5. thumb fingertip.

Only the four tactile fingertip geoms can collide with the target object.
Non-tip contacts therefore cannot hide a detached fingertip.

## Validated result

The default `adaptive_surface_mpc` run was validated on Windows with an RTX
4090 D and `cuda:0`:

- planned and executed travel: `0.2000 m`;
- adaptive-MPC keyframes / execution frames: `40 / 660`;
- per-point final progress: `[200, 200, 200, 200, 200] mm`;
- maximum plan joint step: `0.00205 rad`;
- maximum progress error: `0.66 mm`;
- maximum planned fingertip normal error: `2.02 mm`;
- physical contact ratios (index, middle, ring, thumb):
  `[1.0, 1.0, 1.0, 1.0]`;
- maximum nonzero motor-force correction: `0.006886 rad`.

The controller does not use natural finger closure. It first records the
22-joint loaded servo deflection that established real four-finger contact,
then transports that preload along the URDF-valid trajectory. During motion,
joint tracking lead compensation rejects arm/finger servo lag. Each finger's
motor-force error is converted into an inward Cartesian displacement and then
mapped through that finger's actual `3 x 4` Jacobian.

## Run on Windows

From the repository root:

```powershell
.\.venv\Scripts\python.exe full_hand_mcc\scripts\demo_surface_slide.py `
  --viewer video --device cuda:0
```

This command solves the 40-keyframe MPC from scratch and writes:

- `full_hand_mcc/outputs/adaptive_mpc_motor_force_feedback.mp4`
- `full_hand_mcc/outputs/adaptive_mpc_motor_force_feedback_plan.npz`

For a live viewer, replace `--viewer video` with `--viewer native`.

Run the core tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s full_hand_mcc\tests -v
```

MJLab runs natively on Windows for this task; WSL is not required. The validated
stack used MuJoCo-Warp, CUDA, and the repository-local `.venv`.

## What the demo checks

The rollout automatically:

1. performs tactile-supervised pre-contact search;
2. requires all four real fingertip sensors to remain in contact;
3. records loaded arm/finger servo offsets and per-finger motor-force baselines;
4. solves all five points jointly against the xArm6 + LEAP Hand model and joint limits;
5. moves 0.20 m through 40 MPC keyframes and 660 rate-limited commands;
6. aborts after prolonged contact loss;
7. rejects the result if any fingertip contact ratio is below 99%;
8. reports the actual motor-force correction so a zero-gain run cannot be
   mislabeled as force feedback.

`--reuse-plan <plan.npz>` may be used only for controller-variant tuning. The
default command intentionally replans from the current calibrated contact pose.

## Five reviewed MCC variants

All five variants completed 0.20 m with four 100% physical contact ratios:

| Variant | Main difference | Video |
|---|---|---|
| `hybrid_force_position` | Tangential position plus normal motor-force feedback | `outputs/adaptive_mpc_motor_force_feedback.mp4` |
| `independent_mcc` | Independent contact-coordinate loops | `outputs/independent_mcc_surface_slide.mp4` |
| `motor_torque_mcc` | Direct motor-torque residual path | `outputs/motor_torque_mcc_surface_slide.mp4` |
| `hierarchical_mcc` | Palm/arm priority with fingertip-relative regulation | `outputs/hierarchical_mcc_surface_slide.mp4` |
| `passivity_tank` | Whole-hand rate limiter and energy tank | `outputs/passivity_tank_surface_slide.mp4` |

The hybrid controller is the recommended default. The other variants are
reviewed alternatives, not claims that one trajectory proves identical
hardware behavior for every controller.

## Object-size generalization

Object dimensions are runtime parameters:

```powershell
.\.venv\Scripts\python.exe full_hand_mcc\scripts\demo_surface_slide.py `
  --viewer video --device cuda:0 `
  --object-radius-m 0.018 `
  --output full_hand_mcc\outputs\generalization_radius18mm.mp4 `
  --plan-output full_hand_mcc\outputs\generalization_radius18mm_plan.npz
```

Reviewed matrix using the same MCC gains:

| Object | Contact acquisition | Travel | Contact ratios | Result |
|---|---:|---:|---|---|
| capsule, radius 18 mm, half-height 235 mm | 500 frames | 0.20 m | `[1,1,1,1]` | pass |
| capsule, radius 20 mm, half-height 235 mm | 500 frames | 0.20 m | `[1,1,1,1]` | pass |
| capsule, radius 22 mm, half-height 235 mm | 500 frames | n/a | thumb did not reach before motion | fail |
| capsule, radius 22 mm, half-height 235 mm | 700 frames | 0.20 m | `[1,1,1,1]` | pass |
| short capsule, radius 20 mm, half-height 110 mm, shifted center | n/a | n/a | initial penetration caused numerical divergence | fail |

Thus the current analytic method generalizes over the tested 18–22 mm radius
range after adaptive contact-acquisition time, but it does **not** yet
generalize to arbitrary object length/placement. The short-object failure is
kept as evidence for a future DP policy that can learn collision-free
pre-contact approach, object-conditioned preload, and contact recovery.

## Important controls

- `--object-radius-m`, `--object-half-height-m`, `--object-center-z-m`:
  object geometry and placement;
- `--axial-travel-m`: requested surface travel;
- `--motion-start`, `--steps`: contact-acquisition and motion windows;
- `--mpc-keyframes`: joint-space surface-MPC resolution;
- `--surface-preload-mm`: inward preload embedded in the planned path;
- `--finger-servo-load-scale`: calibrated contact preload margin;
- `--finger-normal-compliance-mm-per-n`: motor-force feedback gain;
- `--min-contact-ratio`: required physical contact ratio;
- `--contact-failure-window`: consecutive bad frames before immediate abort.

## Known limitations and follow-up

- The palm-root point is a planning coordinate, not a required physical contact.
- Radius generalization is validated; arbitrary geometry generalization is not.
- The 22 mm object needs a longer contact-acquisition window.
- Moving a short object directly into the hand can create deep initial
  penetration. A collision-free approach planner or learned DP approach policy
  is still required.
- Hardware use requires motor torque sign/offset identification, conservative
  effort limits, stale-data handling, and emergency stop logic.

Progress and failure evidence are tracked in
[Issue #3](https://github.com/FerryRain/Hand_Compliance_Control/issues/3) and
[Issue #4](https://github.com/FerryRain/Hand_Compliance_Control/issues/4).
