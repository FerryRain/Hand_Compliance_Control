# Agent Progress

更新时间：2026-08-24 00:00 Asia/Shanghai

## 本轮完成

- 再次确认并复用了现有仓库 `/home/ferry/data/Code2/Research/hand_comliance_control`；
- 未 clone、未重装 MuJoCoLab；
- 按当前 hand-only 入口 `src/mjlab/scripts/hand_only_compliance_demo.py` 再次执行正式 4 秒采集；
- 成功生成新的 hand-only forward / inversion 轨迹、截图、视频和 `logs/latest_status.json`；
- 刷新根目录 `README.md` 与本进度文件中的“最新正式 run”指向；
- 本次运行继续满足：掌心朝上、重力关闭、掌内较大物体随机转动、手指 compliance 持续激活、轨迹包含 `T_HO`、并完成轨迹反演。

## 当前入口

```text
src/mjlab/scripts/hand_only_compliance_demo.py
```

## 当前最新正式 run

```text
20260824T000036_random_inhand_grasp_maintain
```

关键产物：

- `artifacts/datasets/20260824T000036_random_inhand_grasp_maintain_trajectory_forward.h5`
- `artifacts/datasets/20260824T000036_random_inhand_grasp_maintain_trajectory_inversion.h5`
- `artifacts/videos/20260824T000036_random_inhand_grasp_maintain_demo.mp4`
- `screenshots/20260824T000036_random_inhand_grasp_maintain_start.png`
- `screenshots/20260824T000036_random_inhand_grasp_maintain_mid.png`
- `screenshots/20260824T000036_random_inhand_grasp_maintain_end.png`
- `logs/20260824T000036_random_inhand_grasp_maintain_summary.json`
- `logs/latest_status.json`

## 运行方式

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
MUJOCO_GL=egl /home/ferry/data/Anaconda/envs/handcomp/bin/python \
  src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 4.0 \
  --video-fps 20 \
  --output-tag random_inhand
```

## 本次摘要指标

- `gravity = [0.0, 0.0, 0.0]`
- `num_steps = 2000`
- `num_video_frames = 81`
- `mean_palm_force = 229.7016`
- `max_palm_force = 276.9551`
- `mean_object_angvel_norm = 11.5106`
- `max_object_angvel_norm = 27.6247`
- `mean_translation_error_m = 7.67e-17`
- `max_rotation_error_fro = 8.08e-15`

## 备注

- 当前范围明确为 hand-only；
- arm/full-hand MCC 历史路径不是当前目标；
- `logs/latest_status.json` 现在可作为下一次自动任务的第一读取入口。
