# MCC fingertip position-teacher pipeline

This directory is intentionally separate from the legacy FSR pipeline in
`finger_compliance_control/`.

完整中文配置、采集、反演、replay 与 DP 训练说明见：

[`MCC-Finger-Pipeline-Guide.md`](MCC-Finger-Pipeline-Guide.md)

## Visual check

```bash
python mcc_finger_compliance_control/scripts/run_test.py --viewer native
```

If the machine's GLX stack is unreliable, use `--viewer viser`.

The first controller version uses four MCC Cartesian fingertip references and
position-only multi-site IK.  Its measured wrench input is fixed to zero.  The
four simulated fingertip 3-D forces are recorded for supervision and quality
analysis, but are not fed back into the controller yet.  Passive compliance
comes from the low-gain joint position actuators.

## 1. Collect

```bash
python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --device cuda:0 --num-envs 1 --trajectory-length 2500 \
  --max-trajectories 5 --motion-start 350
```

## 2. Invert to an object-fixed trajectory

```bash
python mcc_finger_compliance_control/scripts/invert_trajectories.py \
  --file mcc_finger_compliance_control/data/trajectories/<name>.h5
```

## 3. Geometry replay

```bash
python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file mcc_finger_compliance_control/data/inverted/<name>_inverted.h5 \
  --viewer native --mode teacher
```

Use `--viewer headless` for a finite replay and contact summary.
