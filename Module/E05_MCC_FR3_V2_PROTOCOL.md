# E05-MCC-FR3-v2 冻结复测协议

> 协议状态：`FROZEN_BEFORE_FORMAL_RUN`
> 冻结日期：`2026-08-23`
> 环境：`handcomp`
> 范围：修复 FR3 flange–Leap palm mount 后，只复测 `E05-F-MCC` 与 `E05-H-MCC`。

## 1. v2 相对 v1 的唯一模型变更

v1 把 hand-only XML 的 `80 mm` world placement 误用为 `fr3v2_link8 -> palm_lower`
child transform，导致 flange mesh 与 palm mesh 间约 `68.82 mm` 的不可见刚性段。v2 冻结：

- `palm_lower` 必须是 `fr3v2_link8` 的直接固定 child；
- child translation 固定为 `[0, 0, 0.0112] m`，orientation 与 v1 相同；
- link8 上必须存在可见、无质量且不参与碰撞的 `fr3_leap_mount_adapter`；
- adapter 为半径 `28 mm`、半长 `5.6 mm` 的 cylinder，覆盖两个 body origin；
- home pose 的 flange/palm mesh signed distance 绝对值必须 `<= 1 mm`；
- body-origin gap 必须 `<= 20 mm`；
- `model_audit.mount_geometrically_closed` 必须为 `true`。

不得通过移动相机、隐藏 arm geom 或只画一条线来满足 mount gate。除该 mount 修复和对应
审计字段外，v1 的 arm/hand home pose、surface-relative contact geometry、控制器参数、任务、
seed、阈值和 evaluator 保持不变。

## 2. 两个正式单元

| 单元 | FR3 wrist branch | Leap Hand finger branch |
| --- | --- | --- |
| `E05-F-MCC` | 规定式 palm pose tracking；Wrist MCC 关闭 | explicit surface target + 完整 local force-error Fingertip MCC |
| `E05-H-MCC` | 同一 nominal trajectory + Wrist MCC resultant branch | Contact Force Coordinator + internal/differential Fingertip MCC |

本协议不评测 DP。若历史 DP strategy 足以运行，必须另建 E05-F-DP protocol，不能把 DP
结果写进本 MCC 复测目录。

## 3. 物理模型与信号

- MuJoCo `3.6.0`，`dt=0.002 s`，`implicitfast`，80 solver iterations；
- `(nq,nv,nu)=(23,23,23)`：7 FR3 + 16 Leap Hand；
- 7 个 arm position actuator 与 16 个 finger position actuator；
- 物体固定在 world，`nmocap=0`，执行期间只移动 FR3/hand；
- 四个 contact geom 绑定 `fingertip`、`fingertip_2`、`fingertip_3`、
  `thumb_fingertip` 的 belly pad；rounded head 与 FSR mesh 不参与碰撞；
- 指尖力来自 `mj_contactForce`；actual contact hysteresis 为进入 `0.20 N`、退出 `0.10 N`；
- wrist wrench 来自 FR3 七关节 `qfrc_constraint` 与 palm Jacobian least-squares estimate；
- wrench 在 world frame 表达，reference 为 `fr3_palm_control_site`，作用对象为 hand；
- 正式复测继续使用 `gravity=0` 以隔离 contact controller，不能外推到 gravity-on/hardware。

## 4. 任务

每个 episode 为 `15.0 s`，前 `1.0 s` 是接触建立窗口。随后在同一固定二维 multi-scale
height field 上执行：

- nominal palm Y 净位移 `180 mm`；
- X 方向叠加 `18 mm` 与 `7 mm` 两个频率的 S-shaped motion；
- surface 尺寸 `0.60 x 0.84 m`；最大主曲率约 `62.63 1/m`；
- `t=9.0 s` 时 palm 瞬时远离 surface `4 mm`；
- 四指目标法向力均为 `2.0 N`，hard evaluation limit 为 `8.0 N`。

## 5. Coordinator 与 Wrist MCC

只使用 `A_actual` 构造 normal-force basis：

```text
H_A = G_A B_A
e_lambda = lambda_des - lambda_meas
P_resultant = H_W_dagger H_A
N_H = I - P_resultant
```

- F：关闭 wrist force branch，Finger MCC 接完整 local error；
- H：Wrist MCC 接 hand-side resultant wrench，active Finger MCC 只接 `N_H e_lambda`；
- 未真实接触的指仅用完整 local error 执行初始 MAKE/recovery；
- coordinator SVD relative tolerance `1e-4`，transition blend `25 steps`；
- Wrist MCC 为 6D admittance，本任务只激活 collective-normal translation projector；
- Wrist MCC `50 Hz`，Finger MCC/IK 与 physics `500 Hz`；
- wrist translation offset limit `12 mm`，finger normal offset limit `15 mm`。

## 6. 配对 episodes

| episode | seed | friction | force noise | initial joint noise |
| --- | ---: | ---: | ---: | ---: |
| nominal | 7 | 0.90 | 0 N | 0 rad |
| low-friction | 11 | 0.75 | 0.03 N | 0.004 rad |
| noisy-pose | 19 | 1.05 | 0.05 N | 0.006 rad |

任何 guard/safety event 和低性能 episode 都必须保留在 aggregate 中。

## 7. 冻结指标与阈值

每个单元只有所有 episode 都满足阈值才记为 `PERFORMANCE_MET`：

| 指标 | 阈值 |
| --- | ---: |
| `contact_continuity_probability` | `>= 0.995` |
| `average_contact_count` | `>= 3.0` |
| `zero_contact_time_s` | `<= 0.05 s` |
| `force_rmse_n` | `<= 1.0 N` |
| `force_violation_probability` | `<= 0.001` |
| `max_tip_force_n` | `<= 8.0 N` |
| `four_contact_recovery_s` | `<= 0.25 s` |
| `force_settling_s` | `<= 0.75 s` |
| `traversal_y_m` | `>= 0.16 m` |
| `palm_position_tracking_rmse_m` | `<= 0.008 m` |
| `minimum_arm_joint_margin_rad` | `>= 0.03 rad` |
| `minimum_finger_joint_margin_rad` | `>= 0.03 rad` |
| `controller_latency_p95_s` | `<= 0.002 s` |
| `deadline_miss_probability` | `<= 0.01` |
| `non_tip_contact_count` | `= 0` |

H 追加：

| 指标 | 阈值 |
| --- | ---: |
| `wrist_force_z_rmse_n` | `<= 2.5 N` |
| `max_wrist_compliance_translation_m` | `<= 0.0121 m` |
| `coordinator_internal_leakage_p95_n` | `<= 0.05 N` |

`EVALUATED` 与性能 verdict 分离。mount gate、protocol/hash、trace 或 episode 不完整才是执行
无效；完整执行但超过性能阈值仍写 `EVALUATED / NOT_MET`。

## 8. 正式产物

```text
Module/generated/e05_mcc_fr3_v2/
  summary.json
  episodes.csv
  base_traces.npz
  model_audit.json
  generated_fr3_leap.xml
  README.md
```

可视化必须从 v2 nominal trace 重放，并更新：

```text
Module/generated/visual_demo/fr3_leap_e05_f_mcc.mp4
Module/generated/visual_demo/fr3_leap_e05_h_mcc.mp4
Module/generated/visual_demo/fr3_leap_mcc_dashboard.png
Module/generated/visual_demo/fr3_leap_model_audit.png
```

正式 summary 必须记录本文件 SHA-256、v2 code bundle SHA-256、环境、mount audit 和全部
episode 原始指标。生成正式结果后本协议不得修改。
