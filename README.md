# Hand Compliance Control

The research proposal is
[`PROPOSAL.md`](PROPOSAL.md): scalable inverse demonstration generation,
a wrist-conditioned finger Diffusion Policy, and wrist-only ER-GPIS active
exploration. The Windows/CUDA `full_hand_mcc/` implementation is the explicit
whole-hand optimization **Baseline 2**, not the online architecture of the
main method. Franka FR3 supplies palm-root force feedback and the 16 LEAP Hand
motors supply four independent fingertip feedback channels. Five-point
trajectories (palm root plus four physical fingertip pads) are planned under
the assembled FR3 + LEAP URDF/MJCF constraints.
The two low-level control variants and their sensor semantics are fixed in
[`CONTROL_STRATEGIES.md`](CONTROL_STRATEGIES.md).
The active acceptance policy permits short bounded pauses or fingertip release,
but requires majority fingertip contact over most of the route, per-finger
contact ratios, and stable four-tip recovery at the end. FR3/object contact
remains forbidden.

## Current Windows environment

Per the user decision on 2026-07-28, continue the already validated project
runtime for the remainder of the current demo work:

```powershell
.\.venv\Scripts\python.exe
```

This project `.venv` was created from the `DMtactile` Conda interpreter.
Do not switch the active simulation to another Conda environment mid-run.
Migration to the `handcomp` Conda environment is deferred until the demo is
complete; at that point the user will receive a separate manual setup checklist.

## Project index

- `full_hand_mcc/`: Baseline 2 FR3 + LEAP full-hand MCC, explicit five-point
  optimization, collision/contact auditing, GPU simulation, and video delivery.
- `palm_compliance_control/`: palm/arm MCC reference implementation.
- `mcc_finger_compliance_control/`: motor-force fingertip MCC reference.
- `finger_compliance_control/`: earlier finger compliance reference.
- `minimalist_compliance_control/`: upstream/minimal MCC reference.

Start with [`PROPOSAL.md`](PROPOSAL.md),
[`CONTROL_STRATEGIES.md`](CONTROL_STRATEGIES.md), and
[`PROCESS.md`](PROCESS.md), then read
[`full_hand_mcc/PROCESS.md`](full_hand_mcc/PROCESS.md) before continuing a
development session. Full-hand commands and acceptance rules are documented in
[`full_hand_mcc/README.md`](full_hand_mcc/README.md).

## Video outputs

Videos are classified under `full_hand_mcc/outputs/`:

- `debug/00_smoke_and_probes/`
- `debug/10_legacy_surface_methods/`
- `debug/20_fr3_planning/`
- `reference/accepted_xarm6/`
- `deliverables/fr3/`

Only a numerically and visually accepted FR3 result may enter
`deliverables/fr3/`. See
[`full_hand_mcc/outputs/README.md`](full_hand_mcc/outputs/README.md).

## Legacy hand-only data collection

The remaining section documents the older Linux hand-only data-collection
demo. It is retained as a reference and is not the active FR3 task.

### What the legacy demo does

`src/mjlab/scripts/hand_only_compliance_demo.py` builds and runs a hand-only simulation with the following properties:

- palm frame is oriented upward
- gravity is disabled
- a relatively large object is placed in the palm
- the object rotates randomly in-hand via a ball joint and injected torque
- the finger compliance controller stays active during the rollout
- trajectories are recorded with:
  - hand joint position / velocity
  - tactile/contact force traces
  - control targets
  - applied object torque
  - `T_WH`, `T_WO`, `T_HO`, `T_OH`
- trajectory inversion is exported to a second file
- screenshots and a demo video are saved automatically

## Important files

- `src/mjlab/asset_zoo/robots/leaphand_only.xml`
  - hand-only MuJoCo model that reuses the existing Leap Hand asset meshes already present in this repo
- `src/mjlab/scripts/hand_only_compliance_demo.py`
  - simulation loop, compliance control, rendering, logging, and trajectory inversion export

## How to run

From the repo root:

```bash
cd ~/data/Code2/Research/hand_comliance_control
MUJOCO_GL=egl /home/ferry/anaconda3/envs/isaaclab/bin/python src/mjlab/scripts/hand_only_compliance_demo.py
```

Common options:

```bash
cd ~/data/Code2/Research/hand_comliance_control
MUJOCO_GL=egl /home/ferry/anaconda3/envs/isaaclab/bin/python src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 10 \
  --video-fps 30 \
  --width 1280 \
  --height 720 \
  --seed 7
```

Useful flags:

- `--output-root data/hand_only_compliance`
- `--duration-s 8`
- `--control-decimation 10`
- `--sim-dt 0.002`
- `--no-h5`
- `--no-npz`

## Output layout

Each run writes a timestamped folder under:

```text
data/hand_only_compliance/YYYYMMDD_HHMMSS/
```

Expected artifacts:

- `demo.mp4`
- `screenshot_start.png`
- `screenshot_mid.png`
- `screenshot_end.png`
- `trajectory_forward.npz`
- `trajectory_inverted.npz`
- `trajectory_forward.h5`
- `metadata.json`

## Notes

- The task scope is intentionally **hand-only**.
- No new MuJoCoLab clone is required.
- The hand-only XML reuses the existing `xarm6_leap_hand/assets/*` files already inside this repository.
