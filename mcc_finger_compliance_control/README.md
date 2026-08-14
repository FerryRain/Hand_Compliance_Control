# MCC fingertip position-teacher pipeline

This directory is intentionally separate from the legacy FSR pipeline in
`finger_compliance_control/`.

完整中文配置、采集、反演、replay 与 DP 训练说明见：

[`MCC-Finger-Pipeline-Guide.md`](MCC-Finger-Pipeline-Guide.md)

面向多物体、接触流形 DP、全手触觉和主动探索的分阶段路线见：

[`Tactile-Exploration-DP-Roadmap.md`](Tactile-Exploration-DP-Roadmap.md)

## Visual check

```bash
python mcc_finger_compliance_control/scripts/run_test.py --viewer native
```

If the machine's GLX stack is unreliable, use `--viewer viser`.

使用与数据采集相同的随机姿态、物体旋转和控制参数观察 `0.06–0.12 rad/s` 测试：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/run_test.py \
  --viewer native \
  --device cuda:0 \
  --rotate-object \
  --motion-start 350 \
  --motion-length 1800 \
  --angular-speed-min 0.06 \
  --angular-speed-max 0.12 \
  --initial-orientation-mode uniform \
  --seed 20260716 \
  --print-every 50
```

该命令只用于观察，不写 H5。若 native 窗口出现 GLX 错误，把 `--viewer native` 改为
`--viewer viser`。

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

## 4. Inverse replay 与 DP 部署

以下命令使用验证轨迹 `episode 3` 和当前多轨迹最优模型：

```bash
cd ~/Code/Hand_Compliance_Control
conda activate mjlab

INVERTED="mcc_finger_compliance_control/data/inverted/tactile_raw_random_20260715_225348_relaxed99_inverted.h5"
MODEL="mcc_finger_compliance_control/data/models/relaxed486_lerobot_unet_5k/best.pt"
```

### 4.1 完全 teacher replay

palm pose 和 `q_hand` 都逐帧使用反演数据，用于确认坐标变换与几何重建：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file "${INVERTED}" \
  --episode-id 3 \
  --viewer native \
  --mode teacher \
  --device cuda:0 \
  --max-steps 2150 \
  --contact-threshold 0.05
```

无窗口定量测试时，把 `--viewer native` 改为 `--viewer headless`。

### 4.2 Teacher 历史输入 → DP 手指 action

palm 仍按反演轨迹移动；DP 读取 teacher 历史输入，仿真手指动态执行 DP 预测：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file "${INVERTED}" \
  --model "${MODEL}" \
  --episode-id 3 \
  --mode teacher_dp \
  --viewer native \
  --device cuda:0 \
  --inference-steps 50 \
  --contact-threshold 0.05 \
  --seed 20260716
```

### 4.3 完全闭环 DP

palm 仍按反演轨迹移动，但 DP 输入改为仿真实时 `q_hand`、指尖 3D 力和接触法向：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file "${INVERTED}" \
  --model "${MODEL}" \
  --episode-id 3 \
  --mode live_dp \
  --viewer native \
  --device cuda:0 \
  --inference-steps 50 \
  --contact-threshold 0.05 \
  --seed 20260716
```

先做无窗口定量测试时使用 `--viewer headless`。脚本会将逐帧结果保存到模型目录下的 CSV。
`found_contacts` 表示碰撞几何检测到接触；`loaded_contacts` 表示接触力同时达到
`|F_3D| >= contact-threshold`，两者不要混淆。

### 4.4 录制 live DP MP4

`--viewer video` 使用离屏 RGB 渲染，不依赖 GLFW/GLX 窗口。相机会跟随
`robot/palm_lower`，因此大角度轨迹中手不会移出画面：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file mcc_finger_compliance_control/data/inverted/so3_uniform_v2_255x2500_relaxed999_inverted.h5 \
  --model mcc_finger_compliance_control/data/models/so3_uniform_v2_palm_geometry_absolute_q_dp_25k/best.pt \
  --episode-id 108 \
  --mode live_dp \
  --viewer video \
  --video-output mcc_finger_compliance_control/outputs/live_dp_geometry_ep108.mp4 \
  --video-fps 30 --video-width 960 --video-height 720 \
  --video-camera-distance 0.45 \
  --video-camera-azimuth 45 \
  --video-camera-elevation -10 \
  --device cuda:0 \
  --inference-steps 50 \
  --seed 20260717 \
  --contact-threshold 0.05 \
  --finger-impedance \
  --no-finger-nominal-guard \
  --chunk-execution \
  --dp-replan-interval 10 \
  --max-offset-rate-mm 0.08 \
  --recovery-offset-rate-mm 0.20 \
  --force-error-full-scale 1.5
```
