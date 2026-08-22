# Whole-hand Coordinated Compliance Controller

> 状态：`IMPLEMENTED / MODULE_TESTED / E05_MCC_EVALUATED`
> 更新日期：`2026-08-23`
> 固定环境：`handcomp`
> 当前正式实现范围只包含 MCC；历史 DP release 仅完成 compatibility audit，不构成 DP 单元。

## 1. 冻结的核心定义

```text
Wrist MCC  : resultant multi-contact wrench + collective wrist compliance
Finger MCC : internal/differential contact force + local contact compliance
Planner    : tangential exploration intent and contact-transition intent
```

正式 whole-hand analytical baseline 不是相互独立的
`Wrist MCC || 4 x Finger MCC`，而是：

```text
one coordinated multi-contact objective
    -> resultant-wrench branch
    -> internal-force branch
```

任务空间解耦是第一层，wrist-slow/finger-fast 的带宽解耦只是第二层。仅使用不同频率不能
消除稳态 reference conflict。

## 2. 坐标系、作用对象和符号

所有正式日志必须同时记录：

- wrench/force 表达 frame；
- moment reference point；
- force 是 hand-on-object 还是 object-on-hand；
- sensor frame 到 palm frame 的 wrench transform；
- gravity、tool、bias 和 inertial compensation 是否启用。

令 `H` 为 palm frame，`C_i` 为第 `i` 个 contact frame，`r_i^H` 为接触点相对 palm
原点的位置。若 `f_i` 已表达在 palm frame：

```text
G_i = [ I ; [r_i]_x ]
```

若 `f_i` 表达在 contact frame，则必须包含旋转：

```text
G_i = [ R_HCi ; [r_i]_x R_HCi ]
```

本设计固定 `f_i` 表示 **hand-on-object** contact force。对当前真实 contact set
`A_actual`：

```text
w_obj_contact_des  =  G_A f_A_des
w_hand_contact_des = -G_A f_A_des
```

因此 zero wrist wrench 是否正确只取决于 resultant：若 `G_A f_A_des = 0`，非零抓取
squeeze 可以与 zero desired wrist wrench 完全一致。原先“finger 目标力为正时 wrist
不能为零”的说法作废。

FR3 F/T 或 joint-torque estimator 的 raw wrench 只有经过 frame transform、符号统一以及
gravity/tool/inertial/bias compensation 后，才能作为 `w_hand_contact_meas`。正式控制误差为：

```text
e_w_hand = w_hand_contact_des - w_hand_contact_meas
```

不得直接把 simulator contact wrench、未补偿 wrist sensor wrench 和 object-side wrench
混成同一个量。

## 3. Resultant / internal-force decomposition

对 active contacts 堆叠：

```text
e_f = f_A_des - f_A_meas
```

在完整、可实现的 contact-force space 中，可用加权 generalized inverse：

```text
G_W_dagger = W^-1 G^T (G W^-1 G^T)^dagger
P_resultant = G_W_dagger G
N_G         = I - P_resultant

e_resultant = P_resultant e_f
e_internal  = N_G e_f
G e_internal = 0
```

`W` 用于表达 contact reliability、force margin、噪声或 control cost。不得默认所有接触
维度同权，也不得在未检查 rank/condition number 时直接使用普通逆。

### 实际 normal-only Finger MCC

当前 Fingertip MCC 主要沿局部法向产生位移，未必能实现任意三维 contact-force correction。
因此正式实现应在 **controller-realizable contact basis** 中分解：

```text
f_A = B_A lambda_A
H_A = G_A B_A
e_lambda = lambda_A_des - lambda_A_meas

H_W_dagger = W^-1 H_A^T (H_A W^-1 H_A^T)^dagger
N_H = I - H_W_dagger H_A

e_w_obj    =  H_A e_lambda
e_w_hand   = -H_A e_lambda
e_internal = N_H e_lambda
```

这里正负号来自第 2 节固定的 hand-on-object convention；若 Wrist MCC 使用 hand-side
reaction wrench，它必须接收 `e_w_hand`，不能直接使用 object-side 的 `e_w_obj`。

对于 scalar normal-force MCC，`B_A` 由各 active-contact normal 组成；只有控制器能够稳定
执行切向接触力时，才能把相应 tangent basis 加入 `B_A`。unilateral contact、friction cone、
force limit 和 actuator limit 最终应由 constrained weighted allocator/QP 处理，而不是
假设 pseudoinverse 输出天然可执行。

若使用 damping、SVD truncation 或 rank-deficient `H_A`，`H_A e_internal = 0` 只近似成立；
必须记录 residual leakage，而不能宣称完全解耦。

## 4. 两个控制分支

### Wrist MCC

Wrist branch 接收：

```text
X_H_plan(t), e_w_hand, allowed compliance directions
```

并负责：

- palm collective motion 与整体压入深度；
- resultant contact force/torque；
- wrist trajectory 的柔顺执行；
- 持续、低频的 aggregate geometry/load error。

desired wrench 必须由当前 prefix 的 multi-contact objective 生成，并满足：

```text
w_hand_contact_des = -G_A f_A_des
```

如果控制代码选择 object-side convention，则全链路统一改为正号，不能在中间临时翻转。

### Finger MCC

Whole-hand baseline 中，每根 Finger MCC 接收协调后的 local internal correction：

```text
delta_lambda_i_cmd = [N_H e_lambda]_i
```

并负责：

- internal/differential load redistribution；
- 单指局部曲率、roughness 和高频 disturbance；
- 已计划 contact transition 的柔顺 approach/release；
- contact loss 后的局部 recovery execution。

Finger MCC 不得再次积分已分配给 wrist 的 collective component。handover/MAKE/BREAK 的
**决策与执行权限**仍属于 prescribed test transaction 或 contact-mode planner；MCC 只让
已授权 transition 柔顺执行，不能自行改变 contact topology。

### E05-F 与 E05-H 的区别

- `E05-F-MCC`：Wrist MCC 关闭且 wrist 规定式跟踪；没有 wrist force branch 接收
  resultant error，因此 finger controller 使用完整、可实现的 local force error；
- `E05-H-MCC`：必须启用上述 coordinator；Wrist MCC 接收 resultant error，Finger MCC
  只接收 internal/differential correction；
- 当前没有正式 `E05-F-DP/E05-H-DP` 单元；3 s raw compatibility trial 不能填写 DP
  placeholder 或形成 MCC-vs-DP 结论。

## 5. Planner / compliance direction decomposition

对只有平移、且 collective surface normal `n_H` 定义良好的简单场景，可写成：

```text
P_n = n_H n_H^T
P_t = I - P_n

P_t : planner-dominated tangential exploration
P_n : compliance-dominated collective normal motion
```

正式 FR3 控制是 6D twist/wrench hybrid control，不能把上述 `3 x 3` 投影直接复制到 torque
维度。必须冻结 power-consistent motion/wrench selection matrices、rotational compliance
axes 和 frame，并验证 planner 与 wrench loop 不在同一方向互相抵消。

建议的层次为：

```text
Planner    : tangential exploration and nominal wrist motion
Wrist MCC  : selected resultant force/torque and collective compliance
Finger MCC : realizable internal/differential contact regulation
```

完成任务空间解耦后，才冻结第二层的时间尺度。`finger 50–100 Hz`、`wrist 10–30 Hz` 只能
作为初始设计区间，正式频率必须由 simulator timestep、sensor delay、FR3 dynamics 和
stability test 决定。

## 6. Contact-set changes

所有 map 和 projector 使用 `A_actual`，不能使用预测 contact set 覆盖真实状态：

- 未确认 MAKE 的 finger 不进入承载 wrench 的 `G_A/B_A`；
- 新接触达到 force/持续时间确认后，才能加入 active coordinator；
- BREAK 由 transaction 授权，实际释放后再从 active map 移除；
- contact set、rank 或 basis 改变时，对 projector/output 做有界 blending，并 reset/freeze
  相关 integrator，防止瞬时 command jump；
- `rank(H_A)`、condition number、singular values 和 `||H_A e_internal||` 超阈值时进入
  degraded mode 或 `SAFE_HOLD`，阈值需在实现前冻结。

这与 transactional prefix 语义一致：预测 suffix 无权提前改变 coordinator 的 active set。

## 7. Force-space 解耦仍需物理验证

上述分解消除了 reference inconsistency，但它本身不保证 actuation-level 完全正交。wrist
位移会改变各 contact force，finger motion 也可能因 stiffness、Jacobian error 和 saturation
泄漏出 resultant wrench。因此控制器实现前还必须定义/识别 command-to-contact-force
effectiveness，并独立测试：

1. pure resultant wrist command 的 internal-force leakage；
2. pure internal finger command 的 resultant-wrench leakage；
3. steady-state cross-loop chasing 与 integrator windup；
4. contact-map rank change 时的 command continuity；
5. friction/force/joint saturation 下 constrained allocator 的可行性。

若 leakage 超过冻结阈值，应升级为基于 contact stiffness/Jacobian 的 constrained allocator，
而不是只靠降低 wrist bandwidth 掩盖耦合。

## 8. 必须记录的 coordinator 日志

- `A_actual`、contact frames、contact points/normals 与 frame transforms；
- `G_A`、`B_A`、`H_A`、rank、singular values、condition number 和 weight matrix version；
- `f/lambda_des`、`f/lambda_meas`、resultant/internal components；
- object-side/hand-side desired and measured wrench；
- `||H_A e_internal||`、cross-branch leakage 与 reconstruction error；
- planner/wrench selection matrices、wrist/finger command components；
- saturation、anti-windup、projector blending、contact activation 和 safety override events。

## 9. 已实现并执行的独立测试

Module 4C/W/H 已执行以下测试；正式物理结果见 `E05_MCC_FR3_PROTOCOL.md` 和
`generated/e05_mcc_fr3_v1/summary.json`：

1. **Sign/frame test**：对称 squeeze 应满足 `G f_des = 0`；同侧 collective load 应得到
   非零 resultant；变换 contact/palm/sensor frame 后物理 wrench 保持一致；
2. **Projector algebra test**：对各合法 `A_actual`、rank 和 weight，检查 decomposition
   reconstruction、`||H_A e_internal||` 与数值 conditioning；
3. **Realizability test**：projected command 满足 normal/tangent basis、unilateral contact、
   friction、force、joint 和 actuator limits；
4. **Contact transition test**：MAKE confirmation、BREAK、rank change 和 projector blending
   不产生超阈值 command jump 或 stale integrator；
5. **Cross-loop physics test**：分别注入 pure resultant 与 pure internal objective，测量双向
   leakage、steady-state chasing、overshoot 和 recovery；
6. **Planner-selection test**：切向 trajectory 不被 normal wrench loop 抵消，selected
   rotational compliance 也不越过 planner authority。

当前 unit/integration tests 验证了 sign/frame、projector reconstruction、actual-contact
authority、joint-torque wrench recovery、wrist motion sign、23-DoF pose IK 和 grouped guards。
正式 E05 还测量了 leakage、rank/condition、contact recovery 与 force performance；两个
单元均为 `EVALUATED / NOT_MET`，主要剩余问题是 peak force 与 H 分支 settling，而不是
接口或执行链未实现。
