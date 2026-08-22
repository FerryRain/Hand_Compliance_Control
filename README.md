# Hand Compliance Control

当前仓库以 [`Module/`](Module/) 为唯一的新架构入口。现阶段已完成 M0–M3 的
FR3+Leap Hand 扩展、M04 整手 MCC，并正式评测 `E05-F-MCC` 与 `E05-H-MCC`。
FR3–Leap 安装间隙已经修正并完成 v2 复测。历史 DP release 的资产与推理已验证完整，
但其 `DP nominal + FullHandMCC` 契约不等于当前 standalone Finger DP，正式 DP E05
仍未评测。

## 当前结论

| 项目 | 状态 |
| --- | --- |
| M0-FR3 full-robot contract | `PASSED` |
| M01-FR3 live robot geometry | `PASSED` |
| M02-FR3 four-finger MCC | `PASSED` |
| M03-FR3 grouped runtime guards | `PASSED` |
| M04-R/W/C/H whole-hand MCC | `PASSED_STRUCTURE / MOUNT_FIXED` |
| E05-F-MCC | `EVALUATED / NOT_MET` |
| E05-H-MCC | `EVALUATED / NOT_MET` |
| historical DP release | `ASSET_COMPLETE / INFERENCE_PASSED` |
| standalone Finger DP E05 | `BLOCKED / NOT_EVALUATED` |

`EVALUATED` 表示冻结实验完整运行；`NOT_MET` 表示有性能阈值未满足，不能写成实验
`FAILED`。mount v2 的 flange/palm mesh distance 为 `0.0206 mm`，且有显式可见 adapter。
复测中 F/H 的连续接触分别为 `100.000%` / `99.981%`，但 peak force 等指标仍需改进；
H 的 worst peak 为 `15.751 N`。完整说明见
[`Module/README.md`](Module/README.md)。

## 先看可视化

打开 [`Module/generated/visual_demo/index.html`](Module/generated/visual_demo/index.html)。
其中包含：

- 23-DoF FR3+Leap 模型和指尖指肚接触 close-up；
- 两个独立视角的 flange–palm mount close-up 与几何距离审计；
- M01 Oracle、M02 Fingertip MCC、M03 runtime guards；
- 两段完整 `15 s` 的 `E05-F-MCC` / `E05-H-MCC` MuJoCo 视频；
- contact、force、二维 traversal、wrist wrench、曲率与突变恢复 dashboard。

视频中的物体固定在 world，palm trajectory 由 FR3 七个关节实际执行；不是 fixed palm、
inverse object motion 或无物理引擎动画。

## 环境与复现

固定使用现有 `handcomp` 环境：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m unittest discover -s Module/tests -v
```

重新运行正式 MCC-only E05：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.module_4_whole_hand_mcc.demo
```

从正式 trace 重建视频，并刷新总 gallery：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.fr3_visual_demo
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.visual_demo
```

## Hand-only compliance cron 入口

当前定时数据采集任务只保留 hand-only 范围：

- 不使用机械臂；
- 不恢复旧的 arm/full-hand MCC 路径；
- 直接复用当前仓库中的现有 MuJoCoLab 安装；
- 使用 `src/mjlab/scripts/hand_only_compliance_demo.py` 作为正式入口。

运行环境：

```bash
conda activate handcomp
```

或直接使用：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

正式 4 秒采集命令：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  src/mjlab/scripts/hand_only_compliance_demo.py \
  --duration-s 4.0 \
  --video-fps 20 \
  --output-tag random_inhand
```

本任务要求并已满足：手掌朝上、关闭重力、掌内较大物体随机转动、finger compliance 持续激活、记录包含 `T_HO` 的轨迹、完成轨迹反演、并输出截图与 demo 视频。

最新正式 run：

```text
20260823T020021_random_inhand_grasp_maintain
```

关键产物：

- `artifacts/datasets/20260823T020021_random_inhand_grasp_maintain_trajectory_forward.h5`
- `artifacts/datasets/20260823T020021_random_inhand_grasp_maintain_trajectory_inversion.h5`
- `artifacts/videos/20260823T020021_random_inhand_grasp_maintain_demo.mp4`
- `screenshots/20260823T020021_random_inhand_grasp_maintain_start.png`
- `screenshots/20260823T020021_random_inhand_grasp_maintain_mid.png`
- `screenshots/20260823T020021_random_inhand_grasp_maintain_end.png`
- `logs/20260823T020021_random_inhand_grasp_maintain_summary.json`
- `logs/latest_status.json`

## 文档入口

- [`Module/MASTER_PLAN.md`](Module/MASTER_PLAN.md)：唯一主任务记录与 Gate；
- [`Module/README.md`](Module/README.md)：目录、模块 API、命令和 demo；
- [`Module/PROTOCOL.md`](Module/PROTOCOL.md)：M0–M3 及 FR3 extensions；
- [`Module/E05_MCC_FR3_V2_PROTOCOL.md`](Module/E05_MCC_FR3_V2_PROTOCOL.md)：mount-fixed MCC 复测协议；
- [`Module/E05_EVALUATION_PLAN.md`](Module/E05_EVALUATION_PLAN.md)：F/H 两层定义；
- [`Module/WHOLE_HAND_COMPLIANCE_DESIGN.md`](Module/WHOLE_HAND_COMPLIANCE_DESIGN.md)：
  resultant/internal force decomposition。
- [`Module/evidence/2026-08-23_DP_STRATEGY_AUDIT.md`](Module/evidence/2026-08-23_DP_STRATEGY_AUDIT.md)：
  历史 DP 资产、checkpoint 推理和当前 E05 兼容性结论。

## 当前边界

- 正式 E05 只有 MCC；DP raw 3 s 试跑仅为 `EVIDENCE_ONLY`，不能得出 MCC-vs-DP 结论；
- 正式 protocol 使用固定物体和 gravity-off，以隔离 contact controller；
- 当前结果是 MuJoCo 动力学仿真，不是硬件验证；
- 历史 fixed-palm `E05-PHY-v3` 仅保留为可追溯 evidence，不再是默认测试入口。
