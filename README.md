# Hand-Only Compliance Control

当前仓库的 **当前任务入口** 是 hand-only MuJoCo 数据采集与演示，不是 arm / full-hand MCC 恢复路径。

## 当前任务范围

本轮 OpenClaw 任务固定为：

- 只做 **hand-only** 仿真；
- **不 clone、不重装**，直接复用现有仓库与已有 MuJoCoLab 安装；
- 保持 **手掌朝上**；
- 保持 **重力关闭**；
- 在掌内放置一个 **相对较大的物体**；
- 让物体在掌内 **随机转动**；
- 手指 **compliance controller 始终激活**；
- 记录轨迹，且至少包含 **`T_HO`**；
- 生成 **trajectory inversion**；
- 保存 **截图** 与 **demo 视频**；
- 将最新运行状态写入 `logs/latest_status.json`。

当前正式入口：

```text
src/mjlab/scripts/hand_only_compliance_demo.py
```

## 环境

已验证环境：

```bash
conda activate handcomp
```

或直接使用绝对 Python：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

> 当前 OpenClaw 自动任务与 README 中的所有示例都默认使用这个 `handcomp` conda 环境。

## 如何运行

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
MUJOCO_GL=egl /home/ferry/data/Anaconda/envs/handcomp/bin/python \
  src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 4.0 \
  --video-fps 20 \
  --output-tag random_inhand
```

默认行为：

- 运行 4 秒 hand-only 仿真；
- 输出 forward trajectory（`npz` / `h5` / `json`）；
- 输出 inversion result（`npz` / `h5` / `json`）；
- 输出 1 个 demo 视频；
- 输出 `start / mid / end` 三张截图；
- 刷新 `logs/latest_status.json`。

## 最新正式 run

```text
20260824T220020_random_inhand_grasp_maintain
```

关键产物：

- `artifacts/datasets/20260824T220020_random_inhand_grasp_maintain_trajectory_forward.h5`
- `artifacts/datasets/20260824T220020_random_inhand_grasp_maintain_trajectory_inversion.h5`
- `artifacts/videos/20260824T220020_random_inhand_grasp_maintain_demo.mp4`
- `screenshots/20260824T220020_random_inhand_grasp_maintain_start.png`
- `screenshots/20260824T220020_random_inhand_grasp_maintain_mid.png`
- `screenshots/20260824T220020_random_inhand_grasp_maintain_end.png`
- `logs/20260824T220020_random_inhand_grasp_maintain_summary.json`
- `logs/latest_status.json`

## 本次运行满足的约束

从 `logs/latest_status.json` 可验证：

- `gravity = [0.0, 0.0, 0.0]`
- object half-size = `[0.038, 0.05, 0.028]` m
- object mass = `0.140 kg`
- trajectory 显式记录 `T_HO`
- inversion 输出 `T_OH`
- 反演误差保持在数值精度量级

本次 22:00 run 的摘要指标：

- `num_steps = 2000`
- `num_video_frames = 81`
- `mean_object_angvel_norm = 11.5106`
- `max_object_angvel_norm = 27.6247`
- `mean_translation_error_m = 7.67e-17`
- `max_rotation_error_fro = 8.08e-15`

## 输出位置约定

- 前向轨迹：`artifacts/datasets/*_trajectory_forward.{npz,h5,json}`
- 反演结果：`artifacts/datasets/*_trajectory_inversion.{npz,h5,json}`
- 视频：`artifacts/videos/*_demo.mp4`
- 截图：`screenshots/*_{start,mid,end}.png`
- 运行摘要：`logs/*_summary.json`
- 最新状态：`logs/latest_status.json`

## 备注

- 当前 scope 明确为 **hand-only**；
- **不要** 把 arm / full-hand MCC 历史路径恢复成当前默认目标；
- `Module/` 下其它 FR3 + Leap 工作可以保留，但不是这项自动任务的当前交付入口。
