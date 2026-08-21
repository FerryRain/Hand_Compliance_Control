# Agent Progress

更新时间：2026-08-21 14:08 Asia/Shanghai

## 本轮完成

- 确认并复用了现有仓库 `/home/ferry/data/Code2/Research/hand_comliance_control`；
- 未 clone、未重装 MuJoCoLab；
- 检查到原先 cron 指定的 `docs/openclaw_compliance_order.md`、`docs/agent_progress.md`、`logs/latest_status.json` 缺失，并已补齐；
- 将 `src/mjlab/scripts/hand_only_compliance_demo.py` 补成当前 hand-only 正式入口；
- 运行脚本生成新的 hand-only forward / inversion 轨迹、截图、视频和 latest status；
- 更新根目录 `README.md`，写清了运行命令和 conda 环境 `handcomp`。

## 当前入口

```text
src/mjlab/scripts/hand_only_compliance_demo.py
```

## 当前最新正式 run

```text
20260821T140810_random_inhand_grasp_maintain
```

关键产物：

- `artifacts/datasets/20260821T140810_random_inhand_grasp_maintain_trajectory_forward.h5`
- `artifacts/datasets/20260821T140810_random_inhand_grasp_maintain_trajectory_inversion.h5`
- `artifacts/videos/20260821T140810_random_inhand_grasp_maintain_demo.mp4`
- `screenshots/20260821T140810_random_inhand_grasp_maintain_start.png`
- `screenshots/20260821T140810_random_inhand_grasp_maintain_mid.png`
- `screenshots/20260821T140810_random_inhand_grasp_maintain_end.png`
- `logs/20260821T140810_random_inhand_grasp_maintain_summary.json`
- `logs/latest_status.json`

## 运行方式

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 4.0 \
  --video-fps 20 \
  --output-tag random_inhand
```

## 备注

- 当前范围明确为 hand-only；
- arm/full-hand MCC 历史路径不是当前目标；
- `logs/latest_status.json` 现在可作为下一次自动任务的第一读取入口。
