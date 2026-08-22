# E05 当前评测定义：正式 MCC-only + DP compatibility 边界

> 状态：`EVALUATED`
> 日期：`2026-08-23`
> 环境：`handcomp`
> 当前正式范围：只评测 MCC；历史 DP raw trial 仅为 `EVIDENCE_ONLY`，不填正式指标。

## 1. 两个单元

| 单元 | Wrist branch | Finger branch | 回答的问题 |
| --- | --- | --- | --- |
| `E05-F-MCC` | 规定式 FR3 palm tracking，Wrist MCC off | 完整 local force-error Fingertip MCC | 移动 wrist 下 finger MCC 能否维持接触？ |
| `E05-H-MCC` | 同一 nominal path + resultant Wrist MCC | coordinator 后的 internal Fingertip MCC | 加入全手协调后 wrist/finger 是否互相打架？ |

因此本轮不是 controller-vs-controller 比较，也不形成 MCC-vs-DP 结论；它比较 MCC 的两个
控制层级，并验证整手 analytical baseline 是否已真正接入 FR3。

## 2. 共同条件

- 同一个 `23-DoF` FR3+Leap dynamic plant；
- 同一固定 world object；执行期间 `nmocap=0`；
- 同一 nominal 15 秒 2D traversal、4 mm away step、force target 和 seeds；
- 同一真实 fingertip-body belly pads、`mj_contactForce`、joint-torque wrench estimator；
- 同一 Runtime Guards、日志与 evaluator；
- 正式 MCC evaluator 不读取 DP checkpoint、observation、action、fallback 或隐藏后处理；
  外部历史 release 只在独立 compatibility tool 中读取。

正式参数、三组配对扰动和阈值只以
[`E05_MCC_FR3_V2_PROTOCOL.md`](E05_MCC_FR3_V2_PROTOCOL.md) 为准。

## 3. E05-F-MCC

FR3 高刚度跟踪规定式 palm trajectory；Wrist MCC 关闭。每根 finger 根据 surface target 和
完整误差执行：

```text
e_i = lambda_i_des - lambda_i_meas
M_i dd(delta_i) + D_i d(delta_i) + K_i delta_i = e_i
```

主要指标：contact continuity/count/loss、force RMSE/peak/violation、step recovery、joint
margin、palm tracking、controller latency 和 traversal。

## 4. E05-H-MCC

对 hysteresis-confirmed `A_actual` 构造 normal-force map：

```text
H_A = G_A B_A
e_lambda = lambda_des - lambda_meas
e_resultant = H_W_dagger H_A e_lambda
e_internal = (I - H_W_dagger H_A) e_lambda
```

- hand-side desired wrench 进入 Wrist MCC；
- active fingers 只接 internal/differential error；
- 尚未确认接触的 finger 仍以完整 local error 执行已授权 initial MAKE/recovery；
- wrist wrench 由 FR3 joint constraint torque 和 palm Jacobian 得到，不使用 zero-wrench
  假目标；
- 本任务只激活 collective-normal translation projector，planner 的 tangential path 不被
  Wrist MCC 抵消。

追加指标：wrist Fz tracking、compliance offset、FR3 external torque、map rank/condition、
internal leakage 和 projector transition。

## 5. 状态语义

- `EVALUATED`：三个正式 episode 全部完成，protocol/code hash、trace 和产物有效；
- `MET/NOT_MET`：按冻结阈值描述性能；
- 已完整执行但 peak force 超阈值时，状态是 `EVALUATED / NOT_MET`，不是实验 `FAILED`；
- 只有模型、trace、协议或 episode 无效/不完整才算执行失败。

## 6. 当前结果

| 项目 | E05-F-MCC | E05-H-MCC |
| --- | ---: | ---: |
| episodes | 3/3 | 3/3 |
| execution | `EVALUATED` | `EVALUATED` |
| performance | `NOT_MET` | `NOT_MET` |
| mean contact continuity | 100.000% | 99.981% |
| mean force RMSE | 0.782 N | 1.020 N |
| worst peak force | 11.113 N | 15.751 N |
| mean traversal Y | 174.0 mm | 175.3 mm |
| mean controller P95 | 1.200 ms | 1.270 ms |

F 的未达项为 force-violation probability 和 peak force。H 的未达项为 force RMSE、
force settling、force-violation probability 和 peak force。逐 episode 原始值位于
`generated/e05_mcc_fr3_v2/summary.json` 与 `episodes.csv`。

## 7. DP release compatibility 边界

`dp-capsule-v1` 的 checkpoint、normalization、训练/teacher H5 和部署代码已确认齐全，且
state dict / 10-step diffusion inference 通过。但其 frozen deployment 是
`DP nominal + FullHandMCC`，不是 standalone Finger DP。关闭 MCC 后接入当前 E05 的 3 s
raw trial 最终四指全失触，只能写成 `EVIDENCE_ONLY`。正式 E05-F-DP 仍为
`NOT_EVALUATED`；在用户选择 standalone DP 或 DP+shared-MCC 的公平契约前，不得增加 DP
列。完整审计见 `evidence/2026-08-23_DP_STRATEGY_AUDIT.md`。

## 8. 历史预验证边界

`E05-PHY-v3` 继续登记为 `E05-PRE-FMCC`：固定 palm、反向 mocap object 的旧 finger-only
证据。其协议和结果保持不可变，不能替代本轮 FR3 E05，也没有被新结果覆盖。
