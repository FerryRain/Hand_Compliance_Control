# E05 当前评测定义：Finger 层与 Whole-hand 层

> 日期：`2026-08-23`
>
> 环境：`handcomp`
>
> 当前状态：MCC 已评测；DP 旧实现验收不合格，标记为待重做且不进入当前结果。

## 为什么分层

E05 必须区分两个问题：

1. **Finger 层**：同一 FR3 wrist motion 下，Fingertip MCC 与未来重做的 standalone
   Finger DP 如何维持接触；
2. **Whole-hand 层**：加入 Wrist MCC 和 resultant/internal force coordinator 后，整手
   compliance 是否改善。

## 当前单元

| 单元 | Wrist branch | Finger branch | 当前状态 |
| --- | --- | --- | --- |
| `E05-F-MCC` | prescribed FR3 palm tracking | 4×完整 local force-error MCC | `EVALUATED / NOT_MET` |
| `E05-H-MCC` | resultant Wrist MCC | coordinated internal/differential MCC | `EVALUATED / NOT_MET` |
| `E05-F-DP` | 与 F-MCC 相同的 wrist trajectory | standalone Finger DP | `REWORK_REQUIRED / NOT_EVALUATED` |

当前提交没有 DP checkpoint、训练数据、推理代码或 DP 指标。旧失败运行不得作为正式
`E05-F-DP` 结果。

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

## DP 重做的公平性要求

未来 DP 只有满足以下条件才能进入 E05：

- 与 F-MCC 使用相同的 FR3 wrist trajectories、初始状态、对象和 held-out disturbances；
- 输入仅包含部署时可获得的因果观测，不使用未来接触点或 Oracle 泄漏；
- 明确加入 fingertip force magnitude/history、finger `q/dq`、contact validity 和 action
  history，并冻结 observation/action schema；
- 完整运行冻结时长，M03 guard authority 与 MCC 对齐；
- DP 不得隐藏使用 Fingertip MCC、FullHandMCC 或 analytical force-error fallback；
- 先冻结协议，再采集、训练和评测；失败诊断不能冒充正式结果。

## 状态语义

- `EVALUATED`：协议、episode、trace 与 evaluator 完整有效；
- `NOT_MET`：方法性能未满足预冻结阈值，不等于 evaluator 失败；
- `REWORK_REQUIRED / NOT_EVALUATED`：旧实现未被接纳，当前没有可发表指标；
- `FAILED` 只用于模型、协议、日志或 evaluator 自身无效。
