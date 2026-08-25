# E05 Exp. 1 归档协议：H-MCC vs H-DP-direct

> 冻结日期：2026-08-24
> 环境：`handcomp`
> 物理：MuJoCo，500 Hz，gravity off
> DP：CUDA-only，50 Hz；CPU fallback 禁止
> 当前定位：`EVALUATED / DESCRIPTIVE_ARCHITECTURE_ABLATION`

> 兼容性说明：文件名保留 `CURRENT` 以免破坏已有结果引用，但该协议只对应已经完成的
> DP-direct 实验，不再代表当前 main architecture。当前三层实验计划见
> [`E05_EVALUATION_PLAN.md`](E05_EVALUATION_PLAN.md)，新的 DPRef 设计见
> [`DP_REFERENCE_GENERATOR_DESIGN.md`](DP_REFERENCE_GENERATOR_DESIGN.md)。

## 问题

在同一 whole-hand architecture 下，只替换 finger controller：

```text
E05-H-MCC = FR3 Wrist MCC + coordinated Fingertip MCC
E05-H-DP-direct = FR3 Wrist MCC + Finger DP-direct + Action Authority Filter
```

这项实验不评测 `E05-F-DP`，也不比较 planner。它只回答 DP 能否直接替代低层 Finger MCC，
不能回答 DP-generated nominal references 是否有价值，也不能回答是否需要显式 finger planning。

## 共享条件

两边必须共享：

- 同一 23-DoF FR3+LEAP plant、central palm mount 和四个 belly-pad contact geoms；
- 同一大尺寸 extreme hfield、initial state、seed、friction 与 observation noise；
- 同一 180 mm Y traversal、二维 X 轨迹、15 s horizon 和 `t=9 s` 的 +4 mm away step；
- 同一 Wrist MCC、contact-force coordinator、joint/actuator limits；
- 同一 M03 `ForceSafetyExecutor`；hard release 的执行权限高于 MCC、DP 和 authority QP；
- 前 1 s 相同且不计分的 contact initializer。

从 `t=1 s` 起唯一替换：

```text
Finger MCC <-> Finger DP-direct + Authority Filter
```

DP-direct 分支没有 Finger MCC fallback。

## 三组 episode

| 名称 | seed | friction | force noise | initial joint noise |
| --- | ---: | ---: | ---: | ---: |
| nominal | 7 | 0.90 | 0 N | 0 rad |
| low_friction | 11 | 0.75 | 0.03 N | 0 rad |
| noisy_observation | 19 | 1.05 | 0.05 N | 0 rad |

历史 MCC-only `noisy_pose` 直接对已接触 finger q 加噪，会在 controller 第一次运行前形成
170–287 N 的深度穿透。该状态不是有效 controller comparison 初态，因此本 paired protocol
保留 sensor/friction perturbation，但冻结 initial joint noise 为 0。

## 已评测的 DP-direct checkpoint

使用 `training_d20/formal_finger_dp_checkpoint.pt`：

- teacher provenance：`SIM_PRIVILEGED_GT_HORIZON_IK_NON_MCC_V1`；
- training pool：20 个 `RAW_VERIFIED` episode，zero replay repair；
- train/validation 按 object+episode disjoint；
- I20 held-out physical gate 为 PASS；磁盘路径仍保留历史名称 `training_d20/`；
- I100 只在 open-loop 略好，closed-loop contact richness、peak force 与 filter intervention 更差，
  因此不按数据量选择 I100。

## 指标与解释

共同报告：contact continuity、`N_c`、zero-contact、force RMSE/P95/peak/violation、4-contact
recovery、Y traversal、palm tracking、joint margin、Wrist-MCC wrench/offset、non-tip contact、
hard-guard frames 与 wall time。

DP-direct 额外报告：CUDA policy latency、authority intervention/failure/constraint violation、collective
motion 与 Wrist-MCC opposition。DP deadline 为 20 ms；MCC controller deadline 为 2 ms。

结果只保留执行状态与数值观察：

```text
execution_status = EVALUATED | BLOCKED
evaluation_semantics = DESCRIPTIVE_ONLY_NO_STRATEGY_PASS_FAIL
```

8 N、20 ms 等冻结数值继续作为 reference limit；超过时报告具体超出量，不给 H-MCC 或
H-DP-direct 设置 Pass/Fail、MET/NOT_MET。

## 输出

正式目录：`Module/generated/e05_h_mcc_vs_dp/`

- `summary.json`、`episodes.csv`：三组逐项 threshold 与 aggregate；
- `<episode>/e05_h_mcc_trace.npz`、`e05_h_dp_trace.npz`：500 Hz 原始 trace；
- `e05_h_mcc_vs_dp_side_by_side.mp4`：15 s 同步 nominal 视频；
- `e05_h_mcc_vs_dp_dashboard.png`：force/contact/wrist/latency；
- `review.html`：统一人工审阅入口。
