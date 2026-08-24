# OpenClaw Compliance Order

更新时间：2026-08-21

## 当前任务范围

只做 **hand-only compliance** 数据采集与演示，不做 arm / full-hand MCC 恢复。

硬性约束：

- 复用当前仓库已有的 MuJoCoLab 安装；
- 不 clone，不重装；
- 不恢复旧 arm-related 路径作为当前目标；
- 当前可运行入口是 `src/mjlab/scripts/hand_only_compliance_demo.py`。

## 任务要求

hand-only 仿真必须满足：

1. 手掌朝上；
2. 关闭重力；
3. 掌内放置一个相对较大的物体；
4. 让物体在掌内随机转动；
5. 手指柔顺控制器始终保持激活；
6. 记录轨迹，至少包含 `T_HO`；
7. 实现轨迹反演；
8. 保存截图和 demo 视频；
9. 维护根目录 `README.md`，明确写清运行方法和 conda 环境。

## 规范化输出位置

- 前向轨迹：`artifacts/datasets/*_trajectory_forward.{npz,h5,json}`
- 反演结果：`artifacts/datasets/*_trajectory_inversion.{npz,h5,json}`
- 视频：`artifacts/videos/*_demo.mp4`
- 截图：`screenshots/*_{start,mid,end}.png`
- 运行摘要：`logs/*_summary.json`
- 最新状态：`logs/latest_status.json`

## 推荐环境

当前验证通过并写入 README 的环境：

```bash
conda activate handcomp
```

或直接使用：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python
```
