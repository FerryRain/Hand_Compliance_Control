# E05 当前评测定义：Finger 层与 Whole-hand 层

> 日期：`2026-08-23`
>
> 环境：`handcomp`
>
> 当前状态：MCC 已评测；DP v1 架构/core 已冻结并实现。Dataset-D 的真实 spatial-inverse
> pipeline raw gate 已通过，但正式非 MCC Dataset-I 尚未生成，因而未训练、未评测。

## 为什么分层

E05 必须区分两个问题：

1. **Finger 层 diagnostic**：规定式 FR3 wrist motion 下，standalone controller 的局部能力；
2. **Whole-hand primary**：共享同一 Wrist MCC 后，只替换 Finger MCC 与 Finger DP，公平
   比较 local/differential contact realization。

## 当前单元

| 单元 | Wrist branch | Finger branch | 当前状态 |
| --- | --- | --- | --- |
| `E05-F-MCC` | prescribed FR3 palm tracking | 4×完整 local force-error MCC | `EVALUATED / NOT_MET` |
| `E05-H-MCC` | resultant Wrist MCC | coordinated internal/differential MCC | `EVALUATED / NOT_MET` |
| `E05-F-DP` | 与 F-MCC 相同的 wrist trajectory | standalone Finger DP diagnostic | `NOT_STARTED` |
| `E05-H-DP` | 与 H-MCC 完全相同的 Wrist MCC | Finger DP + Action Authority Filter | `NOT_STARTED`（primary） |

当前没有合格的 DP checkpoint 或 DP 指标。废弃 synthetic replay 的实现与生成数据已清理；
`generated/visual_demo/spatial_inverse_v1/` 是唯一保留的 Dataset-D pipeline audit，不是 E05-DP。

## MCC 共同物理条件

- 同一 23-DoF FR3+LEAP MuJoCo plant；
- flange adapter 对准 central palm mesh；
- 四个 distal fingertip belly-pad collision geoms；
- 初始 hand q 取自已发布视频 `t=2.000 s` 的真实物理状态；
- 同一 `0.60 × 0.84 m` finger-heterogeneous height field；
- physics `dt=2 ms`、gravity off、四指目标力 `2 N`、tip-force 上限 `8 N`；
- 同一状态、接触测量、M03 guards、日志与 evaluator。

完整场景、seed 和阈值只以
[`E05_MCC_CURRENT_PROTOCOL.md`](E05_MCC_CURRENT_PROTOCOL.md) 为准。

## E05-F-MCC

FR3 高刚度跟踪规定式 palm trajectory，Wrist MCC 关闭。四根手指使用完整 local normal
force error：

```text
e_i = lambda_i_des - lambda_i_meas
M_i dd(delta_i) + D_i d(delta_i) + K_i delta_i = e_i
```

## E05-H-MCC

对 hysteresis-confirmed `A_actual` 构造：

```text
H_A = G_A B_A
e_lambda = lambda_des - lambda_meas
e_resultant = H_W_dagger H_A e_lambda
e_internal = (I - H_W_dagger H_A) e_lambda
```

- resultant error 进入 Wrist MCC；
- active fingers 只接 internal/differential error；
- 未确认接触的 finger 仅执行局部 MAKE/recovery；
- tangential exploration trajectory 仍由 nominal planner branch 主导。

## 当前 MCC 结果

| aggregate | E05-F-MCC | E05-H-MCC |
| --- | ---: | ---: |
| episodes | 3/3 | 3/3 |
| execution | `EVALUATED` | `EVALUATED` |
| performance | `NOT_MET` | `NOT_MET` |
| mean contact continuity | 100.000% | 99.748% |
| mean contacts | 3.752 | 3.474 |
| mean force RMSE | 1.751 N | 1.857 N |
| worst peak force | 18.165 N | 14.886 N |
| mean Y traversal | 170.84 mm | 172.86 mm |

以上数值会由提交前的最终 MCC-only 重跑刷新；正式来源为
`generated/e05_mcc_current/summary.json`，不能从文档手工回填。

## 正式 Primary：E05-H-MCC vs. E05-H-DP

两者必须逐 episode 共享：initial state、wrist planner/reference、Wrist MCC 参数与内部状态
初始化、SurfaceModel、desired force、hard guard、actuator limits、物理参数、seed、时长和
evaluator。唯一允许替换：

```text
coordinated Finger MCC
<->
force-history Finger DP + DP Action Authority Filter
```

Wrist MCC 负责 collective compliance；DP 只负责 differential/local contact realization 与
handover。Action Authority Filter 是确定性权限投影，不得生成 force target 或充当隐藏 MCC。

## DP 进入 E05 前的要求

未来 DP 只有满足以下条件才能进入 E05：

实施顺序与数据 Gate 以 [`M4_DP_GUIDE.md`](M4_DP_GUIDE.md) 为准。

- 完整满足 `DP_CONTROLLER_V1_PROTOCOL.md` 的因果 observation、filtered force history、
  measured-q anchored command chunk、authority QP 与 guard state machine；
- 输入仅包含部署时可获得的因果观测，不使用未来接触点或 Oracle 泄漏；
- 明确加入 fingertip force magnitude/history、finger `q/dq`、contact validity 和 action
  history，并冻结 observation/action schema；
- 完整运行冻结时长，M03 guard authority 与 MCC 对齐；
- DP 不得隐藏使用 Fingertip MCC、FullHandMCC 或 analytical force-error fallback；
- teacher physical replay 必须通过 duration/contact/force/collision/non-tip/guard/authority audit，
  且必须来自真实 forward physical interaction；空间反演保持相同时间顺序，forward q command
  原序用作 proposal，replay force/contact 重新测量；verified inverse 必须占主要比例，不能由
  privileged repair 主导；
- Dataset-D/Direct-MCC 只用于 pipeline/training diagnostic；正式训练主数据必须是 Dataset-I，
  不得把 MCC forward provenance 隐藏成 independent inverse teacher；
- dataset audit 通过后才能训练，held-out train/eval split 冻结后才能运行 E05；
- `E05-F-DP` 只作为 standalone capability diagnostic，不承担论文主结论；
- 失败诊断不能冒充正式结果。

正式 DP 还需报告 contact/force/traversal/latency 之外的：authority intervention probability 与
norm、QP failure/P95 latency、guard takeover rate/duration、contact-normal opposition rate
`rho_opp` 与 opposition energy `E_opp`。若 filter 长期大幅修改 DP nominal action，不得把
最终性能全部归因于 DP。

## 状态语义

- `EVALUATED`：协议、episode、trace 与 evaluator 完整有效；
- `NOT_MET`：方法性能未满足预冻结阈值，不等于 evaluator 失败；
- `DATASET_I_BLOCKED / NOT_EVALUATED`：core 与 Dataset-D diagnostic 可以运行，但正式
  teacher 数据仍禁止训练；
- `FAILED` 只用于模型、协议、日志或 evaluator 自身无效。
