# E05-MCC-FR3-v1 冻结评测协议

> 协议状态：`FROZEN_BEFORE_FORMAL_RUN`  
> 冻结日期：`2026-08-23`  
> 环境：`handcomp`  
> 范围：只评测 `E05-F-MCC` 与 `E05-H-MCC`；不实现、不运行、不报告 DP。

## 1. 两个正式单元

| 单元 | FR3 wrist branch | Leap Hand finger branch |
| --- | --- | --- |
| `E05-F-MCC` | 规定式 palm pose tracking；Wrist MCC 关闭 | explicit surface target + 完整 local force-error Fingertip MCC |
| `E05-H-MCC` | 同一 nominal trajectory + Wrist MCC resultant branch | Contact Force Coordinator + internal/differential Fingertip MCC |

本协议只比较“加入协调 Wrist MCC 前后”的 MCC 控制层级，不形成 MCC-vs-DP 结论。两个单元
共享同一个 FR3–Leap plant、对象、nominal trajectory、传感器、force target、guards 和
evaluator。

## 2. 物理模型与信号

- MuJoCo `3.6.0`，`dt=0.002 s`，`implicitfast`，80 solver iterations；
- 模型严格为 `nq=23, nv=23, nu=23`：7 FR3 + 16 Leap Hand；
- 7 个 arm position actuator 与 16 个 finger position actuator；
- 物体固定在 world，`nmocap=0`；执行期间只移动 FR3/hand；
- 四个 contact geom 是绑定到 `fingertip`、`fingertip_2`、`fingertip_3`、
  `thumb_fingertip` 的 mesh-registered belly pads；rounded head 与 FSR mesh 不参与碰撞；
- 指尖力来自 `mj_contactForce`；实际 contact 使用测量力 hysteresis：进入阈值
  `force >= 0.20 N`，退出阈值 `force < 0.10 N`；
- wrist wrench 来自 FR3 七关节 `qfrc_constraint` 与 palm Jacobian 的 least-squares estimate，
  source 固定为 `FR3_JOINT_CONSTRAINT_TORQUE`；日志保留 rank、condition 和 residual；
- wrench 表达在 world，reference 为 `fr3_palm_control_site`，作用对象为 hand；
- 为与旧 E05 隔离 contact-controller 差异并避免 gravity/tool compensation 成为额外变量，
  正式 run 使用 `gravity=0`。FR3/hand 的质量、惯量、关节动力学、actuator、接触与 torque
  constraint dynamics 仍全部启用；gravity-on 不是本协议结论。

## 3. 任务

每个 episode 为 `15.0 s`，前 `1.0 s` 是接触建立窗口。随后在固定的二维 multi-scale
height field 上执行：

- nominal palm Y 净位移 `180 mm`；
- X 方向叠加 `18 mm` 与 `7 mm` 两个频率的 S-shaped motion；
- surface 尺寸 `0.60 x 0.84 m`，包含长波、斜向短波、cross wave、凸/凹 Gaussian
  features 与窄 tanh ridge；
- surface 最大主曲率约 `62.63 1/m`，最小曲率半径约 `15.97 mm`；
- `t=9.0 s` 时 palm 瞬时远离 surface `4 mm`，检验接触恢复；
- 四指目标法向力均为 `2.0 N`，hard evaluation limit 为 `8.0 N`。

## 4. Coordinator 与 Wrist MCC

只使用 `A_actual` 构造 normal-force basis：

```text
H_A = G_A B_A
e_lambda = lambda_des - lambda_meas
P_resultant = H_W_dagger H_A
N_H = I - P_resultant
```

- `E05-F-MCC` 没有 wrist force branch，Finger MCC 接完整 local error；
- `E05-H-MCC` 中 Wrist MCC 接 hand-side resultant wrench，active Finger MCC 只接
  `N_H e_lambda`；尚未真实接触的指仍用完整 local error 执行已授权的初始 MAKE/recovery；
- coordinator SVD relative tolerance `1e-4`，projector transition blend `25 steps`；
- Wrist MCC 为可执行 6D admittance，本任务用 collective surface-normal translation
  projector；切向由 nominal trajectory 主导，rotation axes 本协议不激活；
- Wrist MCC `50 Hz`，Finger MCC/IK 与 physics `500 Hz`；
- Wrist translation offset limit `12 mm`；finger normal offset limit `15 mm`；
- contact set、rank、condition、internal leakage、wrist/finger offsets 和 saturation 全记录。

## 5. 配对 episodes

两个单元使用完全相同的三组条件：

| episode | seed | friction | force noise | initial joint noise |
| --- | ---: | ---: | ---: | ---: |
| nominal | 7 | 0.90 | 0 N | 0 rad |
| low-friction | 11 | 0.75 | 0.03 N | 0.004 rad |
| noisy-pose | 19 | 1.05 | 0.05 N | 0.006 rad |

任何 guard/safety event 和低性能 episode 都保留在 aggregate 中，不得删除。

## 6. 冻结指标与阈值

`EVALUATED` 表示三组 episode 全部完成、产物和 protocol hash 完整；它与性能 gate 分离。
每个单元只有所有 episode 都满足下面阈值才记为 `PERFORMANCE_MET`：

| 指标 | 阈值 |
| --- | ---: |
| `contact_continuity_probability` | `>= 0.995` |
| `average_contact_count` | `>= 3.0` |
| `zero_contact_time_s` | `<= 0.05 s` |
| `force_rmse_n`（四个目标指，contact loss 也计误差） | `<= 1.0 N` |
| `force_violation_probability` | `<= 0.001` |
| `max_tip_force_n` | `<= 8.0 N` |
| `four_contact_recovery_s` after step | `<= 0.25 s` |
| `force_settling_s` after step | `<= 0.75 s` |
| `traversal_y_m` | `>= 0.16 m` |
| `palm_position_tracking_rmse_m` | `<= 0.008 m` |
| `minimum_arm_joint_margin_rad` | `>= 0.03 rad` |
| `minimum_finger_joint_margin_rad` | `>= 0.03 rad` |
| `controller_latency_p95_s` | `<= 0.002 s` |
| `deadline_miss_probability` | `<= 0.01` |
| `non_tip_contact_count` | `= 0` |

`E05-H-MCC` 追加：

| 指标 | 阈值 |
| --- | ---: |
| `wrist_force_z_rmse_n` | `<= 2.5 N` |
| `max_wrist_compliance_translation_m` | `<= 0.0121 m` |
| `coordinator_internal_leakage_p95_n` | `<= 0.05 N` |

`max_tip_force_n > 8 N` 会记为 `NOT_MET` 并由 guard 记录，但不会把已完整执行的实验状态
改成 `FAILED`。只有模型/协议/trace 无效或 episode 未完成，才是执行失败。

## 7. 正式产物

正式 runner 必须生成：

```text
Module/generated/e05_mcc_fr3_v1/
  summary.json
  episodes.csv
  base_traces.npz
  model_audit.json
  generated_fr3_leap.xml
  README.md
```

可视化必须从 nominal 正式 trace 重放，而不是另跑一个更容易的动画，并写入：

```text
Module/generated/visual_demo/fr3_leap_e05_f_mcc.mp4
Module/generated/visual_demo/fr3_leap_e05_h_mcc.mp4
Module/generated/visual_demo/fr3_leap_mcc_dashboard.png
Module/generated/visual_demo/fr3_leap_model_audit.png
```

正式结果记录 protocol 文件 SHA-256、代码版本、Python/MuJoCo/NumPy 版本和每个 episode 的
原始指标。任何后续参数更改必须新建 `v2`，不能覆盖本协议下的 v1 结果。
