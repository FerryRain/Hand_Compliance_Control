# Hand-only compliance data collection

This checkout is being used for a **hand-only** MuJoCoLab-based compliance data-collection task.
The arm is no longer part of the active scope. The current demo uses only the Leap Hand kinematic tree while reusing the existing meshes/assets already stored in this repository.

## Conda environment used

Tested and executed with the existing conda environment:

- **env name:** `isaaclab`
- **python path:** `/home/ferry/anaconda3/envs/isaaclab/bin/python`

Recommended runtime setting on this Linux machine:

- `MUJOCO_GL=egl`

## What the current demo does

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
