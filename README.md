# Hand Compliance Control

当前仓库以 [`Module/`](Module/) 为唯一的新架构入口。现阶段已完成 M0–M3 的
FR3+Leap Hand 扩展、M04 整手 MCC，并正式评测 `E05-F-MCC` 与 `E05-H-MCC`。
Finger DP 本轮明确延期：没有实现、训练、运行或指标。

## 当前结论

| 项目 | 状态 |
| --- | --- |
| M0-FR3 full-robot contract | `PASSED` |
| M01-FR3 live robot geometry | `PASSED` |
| M02-FR3 four-finger MCC | `PASSED` |
| M03-FR3 grouped runtime guards | `PASSED` |
| M04-R/W/C/H whole-hand MCC | `BLOCKED`（控制结构通过；80 mm invisible mount 待修） |
| E05-F-MCC | `EVALUATED / NOT_MET` |
| E05-H-MCC | `EVALUATED / NOT_MET` |
| Finger DP | `DEFERRED` |

`EVALUATED` 表示冻结实验完整运行；`NOT_MET` 表示有性能阈值未满足，不能写成实验
`FAILED`。两个 MCC 单元都保持了约 `99.990%` 的连续接触，但 force peak、settling 等
指标仍需改进。提交前复核还发现 link8 到 palm 使用了 80 mm 不可见刚性偏移；当前
E05 数值仅作为 mount 修复前基线，修复后必须重测。完整说明见
[`Module/README.md`](Module/README.md)。

## 先看可视化

打开 [`Module/generated/visual_demo/index.html`](Module/generated/visual_demo/index.html)。
其中包含：

- 23-DoF FR3+Leap 模型和指尖指肚接触 close-up；
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

## 文档入口

- [`Module/MASTER_PLAN.md`](Module/MASTER_PLAN.md)：唯一主任务记录与 Gate；
- [`Module/README.md`](Module/README.md)：目录、模块 API、命令和 demo；
- [`Module/PROTOCOL.md`](Module/PROTOCOL.md)：M0–M3 及 FR3 extensions；
- [`Module/E05_MCC_FR3_PROTOCOL.md`](Module/E05_MCC_FR3_PROTOCOL.md)：冻结的 MCC-only E05；
- [`Module/E05_EVALUATION_PLAN.md`](Module/E05_EVALUATION_PLAN.md)：F/H 两层定义；
- [`Module/WHOLE_HAND_COMPLIANCE_DESIGN.md`](Module/WHOLE_HAND_COMPLIANCE_DESIGN.md)：
  resultant/internal force decomposition。

## 当前边界

- 正式 E05 只有 MCC，不包含 DP，也不能得出 MCC-vs-DP 结论；
- 正式 protocol 使用固定物体和 gravity-off，以隔离 contact controller；
- 当前结果是 MuJoCo 动力学仿真，不是硬件验证；
- 历史 fixed-palm `E05-PHY-v3` 仅保留为可追溯 evidence，不再是默认测试入口。
