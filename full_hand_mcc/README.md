# Full-hand MCC surface-sliding demo

This directory contains the Windows/CUDA MJLab demo for full-hand Minimalist
Compliance Control (MCC). The five jointly planned points are:

1. palm-root planning site (kinematic only; palm contact is not required);
2. index fingertip;
3. middle fingertip;
4. ring fingertip;
5. thumb fingertip.

The xArm6 follows a joint-limit-aware surface trajectory and uses calibrated
six-axis external-load feedback. All 16 LEAP Hand motor-load channels provide
four independent fingertip force estimates. Final validation uses
`full_robot` collision mode: arm, wrist, palm, non-tip finger links, and tips
all collide with the object, while dedicated sensors reject every non-tip
object contact.

## Full-robot 200 mm collision baseline

The `adaptive_surface_mpc + hybrid_force_position` baseline was validated
on native Windows with an RTX 4090 D and `cuda:0`. The object is a thick
capsule with 150 mm radius and 260 mm cylindrical half-height.

- planned/executed travel: `0.2000 m`;
- total video/sliding duration: `28 s / 20 s`;
- MPC keyframes/execution frames: `40 / 2000`;
- final progress of all five planning points:
  `[200, 200, 200, 200, 200] mm`;
- maximum plan joint step: `0.00069 rad`;
- minimum planned non-tip/object clearance: `6.91 mm`;
- maximum planned finger-pad angle: `36.76 deg` (limit `45 deg`);
- physical contact ratios (index, middle, ring, thumb):
  `[0.999, 1.0, 1.0, 1.0]`;
- measured contact-point travel:
  `[199.4, 199.3, 199.1, 199.2] mm`;
- maximum finger motor-force correction: `0.008918 rad`;
- maximum bounded arm force-feedback correction: `0.001000 rad`;
- maximum MuJoCo tip/object penetration:
  `[0.0, 0.0, 0.0, 0.0] mm`;
- arm/object and non-tip-hand/object collision frames: `0 / 0`;
- maximum runtime finger-pad angle: `36.56 deg`.

This baseline proves 200 mm reachability, continuous physical tip contact, and
full-robot collision safety. It is not the final active-finger result because
the finger sites move only `0.3-0.6 mm` relative to the palm.

“Inward” is local to each contact point. On the cylindrical side this points
toward the cylinder axis, but the implementation does not force all four pads
to aim at one global center point.

The standalone LeapHand and the xArm-attached hand use different palm axes.
The controller explicitly applies
`[x,y,z]_attached = [y,x,-z]_fixed` and reorders qpos/dofs by joint name before
using the fixed-palm Jacobian. The diagnostic script verifies about 0.01 mm
tip-position agreement and direction cosine `1.0` for all four Jacobians.

## Variable-curvature validation

The latest reviewed demo replaces the long constant-curvature cylinder with a
short, thick capsule (`radius=150 mm`, cylindrical half-height `100 mm`). The
four contacts straddle and traverse the hemisphere-to-cylinder join, so the
planned meridian curvature changes between `6.667 1/m` and `0 1/m`. This is a
deliberate curvature discontinuity/transition test rather than another nearly
constant cylindrical slide.

The accepted active-finger 14-second run uses
`assets/capsule_150x100_cap_transition_v1.npz` and reports:

- planned/executed route: `27.0 / 26.9 mm`;
- physical tip-site surface travel:
  `[24.7, 26.4, 26.0, 26.6] mm`;
- motion of each tip relative to the palm:
  `[5.9, 4.6, 4.1, 4.9] mm`;
- per-finger maximum joint excursions:
  `[0.2954, 0.1222, 0.2188, 0.1705] rad`;
- continuous-contact ratios:
  `[0.9973, 1.0, 0.9982, 1.0]`;
- maximum planned/runtime pad angle:
  `43.47 / 38.58 deg` (limit `45 deg`);
- minimum planned non-tip/object clearance: `10.09 mm`;
- tip penetration, arm/object collision frames, and non-tip-hand/object
  collision frames: all `0`.

The thumb is colored purple only for visual inspection; the visual material
does not change its collision, contact, or controller behavior. Surface travel
is measured from stable body-fixed physical fingertip sites projected onto the
object. The tactile/contact sensors independently prove continuous contact.
This avoids false travel spikes when MuJoCo changes the selected
`ContactSensor.pos` slot on a rounded pad.

## Run on Windows

From the repository root:

```powershell
.\.venv\Scripts\python.exe full_hand_mcc\scripts\demo_surface_slide.py `
  --viewer video --device cuda:0 `
  --collision-mode full_robot `
  --initial-grasp full_hand_mcc\assets\full_robot_pad_contact_self_collision_free_v10.npz `
  --axial-travel-m 0.20 --motion-start 800 --steps 2800 `
  --object-approach-frames 300 `
  --finger-normal-preload-mm 1.0 `
  --finger-servo-load-scale 0.5 `
  --runtime-tip-gait-mm 0 `
  --arm-mcc-correction-rad 0.001 `
  --min-tip-relative-travel-m 0 `
  --min-finger-joint-excursion-rad 0 `
  --plan-output full_hand_mcc\outputs\pad_frame_fixed_end_to_end_200mm_plan.npz `
  --output full_hand_mcc\outputs\full_hand_mcc_end_to_end_200mm.mp4
```

The first run solves the 40-keyframe MPC. To replay the exact reviewed path,
add:

```powershell
--reuse-plan full_hand_mcc\outputs\pad_frame_fixed_end_to_end_200mm_plan.npz
```

Run the variable-curvature transition case:

```powershell
.\.venv\Scripts\python.exe full_hand_mcc\scripts\demo_surface_slide.py `
  --viewer video --device cuda:0 `
  --object-shape capsule `
  --object-radius-m 0.15 --object-half-height-m 0.10 `
  --collision-mode full_robot `
  --initial-grasp full_hand_mcc\assets\capsule_150x100_cap_transition_v1.npz `
  --planner adaptive_surface_mpc `
  --axial-travel-m 0.027 --axial-direction -1 `
  --palm-travel-ratio 0.8 `
  --mpc-keyframes 27 --mpc-max-nfev 260 `
  --min-meridian-curvature-ratio 2 `
  --min-tip-surface-travel-m 0.024 `
  --min-tip-relative-travel-m 0.003 `
  --min-finger-joint-excursion-rad 0.015 `
  --motion-start 300 --steps 1400 `
  --object-approach-frames 200 `
  --finger-normal-preload-mm 1 `
  --finger-servo-load-scale 0.5 `
  --arm-mcc-correction-rad 0.001 `
  --camera-azimuth-deg 100 --camera-distance-m 0.78 `
  --plan-output full_hand_mcc\outputs\capsule_cap_transition_plan.npz `
  --output full_hand_mcc\outputs\capsule_cap_transition.mp4
```

For a live viewer, replace `--viewer video` with `--viewer native`. MJLab runs
natively on Windows for this task; WSL is not required.

Run the Jacobian/frame audit:

```powershell
.\.venv\Scripts\python.exe `
  full_hand_mcc\scripts\diagnose_finger_normal_mapping.py `
  full_hand_mcc\assets\full_robot_pad_contact_self_collision_free_v10.npz
```

Run the core tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s full_hand_mcc\tests -v
```

## Automatic acceptance checks

The demo:

1. performs tactile-supervised contact search with no natural closure;
2. requires four real fingertip contacts before motion;
3. records loaded arm/finger servo offsets and force baselines;
4. solves all five points against the xArm6 + LEAP Hand model and limits;
5. validates every plan frame for pad orientation, self-collision, and
   non-tip/object clearance;
6. uses each finger's motor-force error only along its local surface normal;
7. uses bounded, non-integrating arm external-load feedback around the loaded
   four-contact state;
8. rejects prolonged tip contact loss, contact ratio below 99%, tip
   penetration above 1 mm, or any arm/non-tip object collision;
9. reports real contact travel and both feedback correction magnitudes.

## Controller variants

Five structures are implemented:

| Variant | Main difference | Current status |
|---|---|---|
| `hybrid_force_position` | Tangential position + normal motor-force feedback | accepted |
| `independent_mcc` | Independent contact-coordinate loops | needs final pad/full-collision rerun |
| `motor_torque_mcc` | Direct motor-torque residual path | needs final pad/full-collision rerun |
| `hierarchical_mcc` | Palm/arm priority + fingertip-relative regulation | needs final pad/full-collision rerun |
| `passivity_tank` | Cartesian energy tank + whole-hand rate limit | needs final pad/full-collision rerun |

Videos made before the physical finger-pad, self-collision, and full-robot
collision audits are not accepted as final evidence.

## Generalization and limitations

- Object radius/half-height/pose are runtime parameters, but only the 150 mm
  thick-object case above has passed the final physical-pad and full-robot
  visual audit.
- Earlier 18-22 mm radius tests are retained as numerical experiments and
  require rerunning under the final checks.
- Arbitrary mesh geometry, short-object collision-free approach, and a full
  circumferential orbit are not yet generalized. These failures motivate an
  object-conditioned DP policy for approach, preload, path selection, and
  contact recovery.
- The 200 mm constant-curvature run is mostly arm transport: each tip moves
  only `0.3-0.6 mm` relative to the palm. The variable-curvature run improves
  this to `4.1-5.9 mm` and passes explicit active-finger-motion thresholds.
- Higher-curvature ellipsoid candidates were explored but rejected: the
  current xArm6 + LEAP Hand configuration encountered protected MCP-to-DIP
  self-clearance or link/object constraints after roughly `14-18 mm`.
  They are not presented as successful demos.
- Hardware use still requires motor torque sign/offset identification,
  conservative effort limits, stale-data handling, and an emergency stop.

Progress, rejected candidates, and the accepted pad-side result are tracked in
[Issue #6](https://github.com/FerryRain/Hand_Compliance_Control/issues/6).
