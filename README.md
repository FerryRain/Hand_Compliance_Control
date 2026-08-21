# Hand Compliance Control

当前仓库的**直接可运行入口**是一个 hand-only 的 MuJoCoLab 数据采集/演示脚本，
用于生成“手掌朝上 + 关闭重力 + 掌内大物体随机转动 + 手指柔顺控制持续生效”的
仿真轨迹、截图、视频和 `T_HO` 反演结果。

这次任务**只保留 hand-only 范围**：

- 不使用机械臂；
- 不恢复旧的 full-hand MCC / arm-related 路径；
- 不重新 clone 或重装 MuJoCoLab；
- 直接复用当前仓库里的现有安装与资产。

## 本次已验证的 conda 环境

推荐并已实际跑通：`handcomp`

```bash
conda activate handcomp
```

如果不方便激活环境，也可以直接调用：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

> 本 README 中的 hand-only 入口、截图、视频和最新 `latest_status.json` 都是用
> `/home/ferry/data/Anaconda/envs/handcomp/bin/python` 生成的。

## 先看哪里

如果你是 OpenClaw / cron 任务，优先读这几个文件：

1. `docs/openclaw_compliance_order.md`
2. `docs/agent_progress.md`
3. `logs/latest_status.json`

## hand-only 入口

脚本：

```text
src/mjlab/scripts/hand_only_compliance_demo.py
```

它会做这些事：

- 加载 `src/mjlab/asset_zoo/robots/leaphand_only.xml`
- 使用掌心朝上的 hand-only 模型
- 验证重力为 `0 0 0`
- 在掌内放置一个相对较大的 box 物体
- 对物体持续施加随机平滑扰动，让它在掌内随机转动
- 始终保持手指柔顺控制器激活
- 记录前向轨迹（包含 `T_HO`、`T_OH`、`T_WH`、`T_WO`）
- 计算轨迹反演并输出数值误差检查
- 导出截图、MP4 演示视频、JSON 汇总和 `logs/latest_status.json`

## 运行方法

### 正式 4 秒采集

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 4.0 \
  --video-fps 20 \
  --output-tag random_inhand
```

### 快速 smoke test

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 0.6 \
  --video-fps 10 \
  --output-tag smoke
```

## 输出位置

脚本会按时间戳生成一组标准化文件：

- 前向轨迹：`artifacts/datasets/*_trajectory_forward.{npz,h5,json}`
- 轨迹反演：`artifacts/datasets/*_trajectory_inversion.{npz,h5,json}`
- 视频：`artifacts/videos/*_demo.mp4`
- 截图：`screenshots/*_{start,mid,end}.png`
- 汇总：`logs/*_summary.json`
- 最新状态：`logs/latest_status.json`

其中：

- forward H5 里包含 `T_HO`
- inversion H5 里包含 `T_HO_source` 和反演得到的 `T_OH_inverted`
- `logs/latest_status.json` 永远指向最近一次运行结果，方便后续 cron 直接读取

## 当前最新一次正式结果

以 `logs/latest_status.json` 为准。当前最新正式 run 为：

```text
20260821T160340_random_inhand_grasp_maintain
```

对应关键产物：

- 视频：`artifacts/videos/20260821T160340_random_inhand_grasp_maintain_demo.mp4`
- 截图：
  - `screenshots/20260821T160340_random_inhand_grasp_maintain_start.png`
  - `screenshots/20260821T160340_random_inhand_grasp_maintain_mid.png`
  - `screenshots/20260821T160340_random_inhand_grasp_maintain_end.png`
- 前向 H5：`artifacts/datasets/20260821T160340_random_inhand_grasp_maintain_trajectory_forward.h5`
- 反演 H5：`artifacts/datasets/20260821T160340_random_inhand_grasp_maintain_trajectory_inversion.h5`
- 汇总：`logs/20260821T160340_random_inhand_grasp_maintain_summary.json`

## 最小检查

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m py_compile \
  src/mjlab/scripts/hand_only_compliance_demo.py
```

## 与迁移架构的关系

仓库总体仍处在“wrist planner + finger DP”迁移过程中。相关架构和边界继续以这些
文档为准：

- `ARCHITECTURE.md`
- `CONTROL_STRATEGIES.md`
- `PROCESS.md`
- `mcc_finger_compliance_control/README.md`

但对当前这项任务来说，**唯一需要直接运行的是上面的 hand-only 采集脚本**。
旧采集器、旧 arm/full-hand MCC 路径不再是当前目标，也不应恢复为默认入口。
