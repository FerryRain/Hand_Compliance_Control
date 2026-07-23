# Full-hand Minimalist Compliance Control

This package extends the existing palm MCC and force-recording fingertip task
into one five-contact controller:

```text
arm motor torque (6) -> palm wrench MCC ----------\
                                                    -> 5 reachable references
finger motor torque (16) -> 4 tip force MCC ------/   -> arm + finger commands
```

The input is exactly five world-frame points in this order:

1. palm root contact site;
2. index fingertip;
3. middle fingertip;
4. ring fingertip;
5. thumb fingertip.

Each point also has an outward object-surface normal. Targets must pass
`FivePointReachabilitySolver`: the solver uses the real 22-DoF MuJoCo model,
the model's joint limits, the palm contact site and all four fingertip sites.
A target is not sent to the controller unless all five FK residuals are below
the configured tolerance.

The visible fingertip collision meshes do not have their kinematic sites on
the mesh surface. The demo therefore establishes real four-finger contact
first, measures the collision-consistent site-to-surface standoff, and moves
both the five surface inputs and the corresponding kinematic references
together. This prevents a nominal IK target from pulling a fingertip through
the object or letting the hand fall back to an unconstrained natural closure.

During sliding, the five-point solver owns the tangential trajectory. The arm
MCC is a bounded correction around that trajectory. Each finger uses its four
motor torque residuals to estimate contact force; the force error adjusts only
that finger's three flexion motors while preserving its side-axis value from
the five-point solution.

## Five versions

| Variant | Main idea | Best use |
|---|---|---|
| `independent_mcc` | Five decoupled Cartesian MCC loops | Simple baseline and gain debugging |
| `motor_torque_mcc` | Adds direct 16-motor torque-error correction after fingertip IK | Fast force response, but torque sign must be calibrated |
| `hierarchical_mcc` | Palm motion has priority; fingertips regulate positions relative to the palm | Large arm motion and weak finger workspace |
| `hybrid_force_position` | Tangential position + normal PI force, with four-finger load balancing | Recommended default for surface sliding |
| `passivity_tank` | Hybrid MCC plus an energy tank and whole-hand rate limit | Hard/unknown objects and conservative hardware tests |

## Windows support

MJLab runs natively on Windows; WSL is not required for this demo. The
validated run used Windows, CUDA, an RTX 4090 D, MuJoCo-Warp and
`device=cuda:0`. Run the command from the repository root. The examples below
assume the project environment already exists at `.venv`; if not, install the
repository dependencies using the main MJLab setup before running the demo.

Check the environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch, mujoco; print(torch.cuda.is_available(), mujoco.__version__)"
```

## Run the demo

From the repository root and the existing `mjlab` environment:

```bash
python full_hand_mcc/scripts/demo_surface_slide.py \
  --variant hybrid_force_position \
  --viewer native \
  --device cuda:0
```

Use `--viewer viser` when a native viewer is inconvenient, or `--device cpu`
for a slow CPU smoke test.

Record the real MJLab/MuJoCo-Warp rollout as a finite MP4:

```bash
python full_hand_mcc/scripts/demo_surface_slide.py \
  --variant hybrid_force_position \
  --viewer video \
  --device cuda:0 \
  --steps 500 \
  --motion-start 100 \
  --slide-speed 0.06 \
  --fps 30 \
  --width 960 \
  --height 720 \
  --finger-force-n 12 \
  --min-contact-ratio 0.99 \
  --output full_hand_mcc/outputs/full_hand_mcc_surface_slide_final.mp4
```

On Windows, use the project virtual environment executable:

```powershell
.\.venv\Scripts\python.exe full_hand_mcc\scripts\demo_surface_slide.py `
  --variant hybrid_force_position --viewer video --device cuda:0 `
  --steps 500 --motion-start 100 --slide-speed 0.06 --fps 30 `
  --width 960 --height 720 --finger-force-n 12 `
  --min-contact-ratio 0.99 `
  --output full_hand_mcc\outputs\full_hand_mcc_surface_slide_final.mp4
```

The demo performs these phases automatically:

1. approach the capsule and require all four real fingertip contact sensors;
2. capture collision-consistent surface/site offsets;
3. rotate all five surface targets tangentially around the capsule;
4. recheck every increment against the 22-DoF model and joint limits;
5. fail immediately on prolonged contact loss and fail at the end if any
   fingertip's contact ratio is below `--min-contact-ratio`.

An unreachable motion increment is bisected and otherwise rejected. A video is
not reported as successful merely because an MP4 was written.

The full-hand task uses a 70 mm-radius, 100 mm-half-height capsule selected by
the real 22-DoF reachability solver. It keeps the arm MCC in a small bounded
region around the validated five-contact pose, so force-reference drift cannot
pull the whole hand away from the object.

The validated 5-second Windows/CUDA run achieved fingertip contact ratios
`[0.9925, 0.9950, 0.9975, 1.0000]` for index, middle, ring and thumb.

Useful controls:

- `--slide-speed`: surface angular speed in rad/s;
- `--motion-start`: first simulation step that moves the surface points;
- `--finger-force-n`: desired motor-estimated force for each fingertip;
- `--min-contact-force-n`: contact-sensor force threshold;
- `--min-contact-ratio`: required per-finger ratio over the sliding phase;
- `--contact-failure-window`: maximum consecutive bad frames before abort;
- `--ik-tolerance-mm`: maximum five-point FK residual.

## Task registration

The task is registered as `Leaphand-Full-Hand-MCC-Control` in
`src/mjlab/tasks/leaphand/__init__.py`. The standalone demo also directly
imports the cfg, so it can be run without going through the task picker.

## Hardware bring-up

Before using a real hand:

1. verify the sign of each `tau_ext = -(tau_motor - tau_bias)` channel;
2. identify per-motor zero offsets with the hand unloaded;
3. start with `passivity_tank`, 0.2 N fingertip force, and 1 N palm force;
4. cap motor effort and stop on stale torque data, joint-limit proximity,
   unexpected contact loss, or reachability rejection;
5. only then tune `hybrid_force_position`.
