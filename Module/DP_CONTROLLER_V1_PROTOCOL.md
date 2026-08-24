# Finger DP Controller v1 冻结设计协议

> 状态：`ARCHITECTURE_FROZEN_CORE_TESTED_DATASET_D_PIPELINE_PASSED_DATASET_I_BLOCKED`
> 日期：`2026-08-23`
> 环境：`handcomp`
> 实现边界：所有新增代码、数据与结果只能位于 `Module/`
> 禁止项：fast learned force residual、隐藏 Finger MCC fallback、旧 DP 结果复用

本文件冻结 controller/data contract；Dataset-D smoke、Dataset-I pilot、raw/repaired 分池、
contact-richness 指标和扩量顺序见 [`M4_DP_GUIDE.md`](M4_DP_GUIDE.md)。

## 1. 正式控制栈与权限

第一版主方法冻结为：

```text
Wrist Planner                 : global / tangential exploration
Wrist MCC                     : collective compliance
Force-history Finger DP       : differential contact realization + handover
DP Action Authority Filter    : deterministic action authority/safety filter
Hard Runtime Guard            : independent authority revocation and release
```

`DP Action Authority Filter` 不是第二个 finger controller。它不生成接触目标、不调节
目标力、不决定 handover，只把 DP nominal action 投影到已冻结的执行权限与硬约束集合。

## 2. Contact-normal authority decomposition

对真实 active contact set `A_actual`，第 `i` 个 contact 的 palm twist 法向速度为：

```text
nu_i^W = n_i^T (v_H + omega_H x r_i^H)
```

堆叠得到：

```text
nu^W = B_H xi_H
B_H[i] = [n_i^T, -n_i^T [r_i^H]_x]
```

Wrist MCC compliance selection 为 `S_C`，因此：

```text
B_C = B_H S_C
P_C = B_C B_C^dagger
```

Finger DP nominal normal motion 为：

```text
nu^DP = J_n Delta_q^DP
```

authority filter 只限制 Wrist MCC 所拥有的 component：

```text
||P_C J_n Delta_q_safe||_inf <= epsilon_C
```

而 `(I-P_C) J_n Delta_q` 是 DP 的 differential/local motion。`P_C` 只由带迟滞确认的
`A_actual` 构造；预测接触不进入 projector。contact-set 变化后立即重建，旧 projector
没有执行权限。

每个 500 Hz command tick 求解：

```text
min ||Delta_q - Delta_q_DP||_W^2
s.t. collective authority, joint position, delta, velocity,
     acceleration and chunk-seam constraints.
```

QP 失败或约束不可行时不得放行 nominal action；输出 deterministic bounded hold，并记录
solver 状态、最大约束违反量、intervention norm 与 latency。

## 3. Frozen observation

符号冻结：

- `nu_i`：contact-normal motion；
- `sigma_i`：SurfaceModel uncertainty；
- `m_i^force`：force channel validity；
- `m_i^geom`：geometry 来自真实 contact；
- `m_i^surface`：SurfaceModel prediction 是否有效。

每指 observation：

```text
o_i = [
  q_i, dq_i,
  F_i^hist, C_i^hist, M_i^force,hist,
  r_i^H, n_i^H, d_i, sigma_i,
  m_i^geom, m_i^surface,
  f_i^des, e_i^fingerID
]
```

free finger 没有真实 contact geometry。此时 `m_i^geom=0`，`r_i^H,n_i^H,d_i,sigma_i`
来自 SurfaceModel；若查询无效则 `m_i^surface=0`。零力与无效 force 必须通过
`m_i^force` 区分。

global/wrist observation：

```text
past real wrist twist
past Wrist MCC displacement/velocity state
future relative wrist-plan twists
previous actually executed finger command
surface_model_version
```

所有 wrist motion 使用当前 palm frame 的 relative twist；策略不能读取未来 real state、
未来 contact、GT-only repair state 或 evaluator label。

## 4. Force preprocessing 与 encoder

多速率结构冻结为：

```text
500 Hz : physics, servo, raw force, hard guard, command interpolation
100 Hz : anti-aliased force/contact/validity history
20 frames / 200 ms : v1 observation history
50 Hz : initial DP replan target rate
```

raw 500 Hz force 必须先经 causal LPF/anti-alias，再降采样；禁止 `f[::5]`。每指 TCN 输入：

```text
x_i^F(k) = [normalized_f_n_i(k), c_i(k), m_i^force(k)]
X_i^F in R^(20 x 3)
```

四指共享 causal TCN，输出 per-finger force latent，并用 finger-ID embedding 区分。
第一版不手工添加 force derivative、peak 或 loading slope。正式 ablation 比较
`100/200/400 ms`，不得预先声称 200 ms 最优。

## 5. Measured-state anchored relative action chunk

DP 输出未来 `H` 个 joint-command offsets：

```text
A_t = {Delta_q_(t,1), ..., Delta_q_(t,H)}
q_nom_(t+k) = q_meas_t + Delta_q_(t,k)
```

每个新 chunk 都以当前真实 `q_meas_t` 为 anchor，不以上一个预测末状态为 anchor。
chunk 第一帧还受 previous executed command 的 seam/rate/acceleration 约束。relative action
本身不等于连续；正式定义是 `measured-state anchoring + boundary continuity`。

训练 target 是 command imitation：

```text
A_t^teacher = {q_teacher,cmd_(t+k) - q_meas_t}_{k=1..H}
```

禁止以未来 measured q 作为 action label。

v1 默认用通过 authority filter 后、进入 plant 前的 actual issued teacher command 作为
`q_teacher,cmd`；privileged nominal 同时保存，用于 filter intervention audit/ablation。
guard-owned frame 及跨越这些 frame 的 chunk 全部排除。这样策略模仿的是可执行 command，
不是 plant response，也不依赖 authority filter 反复修正同一类系统性越权动作。

## 6. Dataset schema 与 teacher

### 6.1 Forward 必须是真实物理 interaction

禁止再从人工 wrist path 构造 hypothetical object motion。一个 forward episode 至少记录：

```text
T_HO_forward(t)
q_f_meas_forward(t), dq_f_meas_forward(t)
q_f_cmd_forward(t)
F_forward(t), C_forward(t)
r_forward(t), n_forward(t)
timestamps, controller/source provenance
```

`q_f_cmd_forward` 是后续可 replay 的核心；只有 wrist SE(3) 而没有真实多指 command/force/contact
interaction 的轨迹一律不是 inverse demonstration。

### 6.2 Spatial inversion 不是 temporal reversal

v1 正式使用空间角色反演，并保持时间索引：

```text
T_OH_replay(t) = inverse(T_HO_forward(t))
q_f_cmd_proposal,replay(t) = q_f_cmd_forward(t)
```

joint angle 是手的内部构型，因此换 parent/child frame 不意味着 `q -> -q`，也不意味着
`q(t) -> q(T-t)`。只有单独的 temporal-reversal ablation 才使用：

```text
T_OH_temporal(t) = inverse(T_HO_forward(T-t))
q_f_cmd_temporal(t) = q_f_cmd_forward(T-t)
```

代码必须通过独立入口区分 `SPATIAL_ONLY` 与 temporal reversal；v1 默认不得反转时间。

### 6.3 Physical replay 与 fresh measurements

固定 object 后真实执行：

```text
T_OH_replay(t), q_f_cmd_proposal,replay(t)
```

Forward 的 force/contact 只作为期望与 validation reference。训练 observation 必须来自 replay
physics 的新测量：

```text
F_replay,real(t), C_replay,real(t), r_replay,real(t), n_replay,real(t)
```

不得复制 forward force/contact 冒充 replay observation。改变 arm motion、inertia、friction
direction、actuator dynamics 与 contact timing 后，两侧 force 不完全相等是允许的。

### 6.4 Verify、repair 与 provenance

raw proposal 成功时标记 `INVERSE_VERIFIED` 并直接保留 command trajectory。只有小范围失败窗口
才允许 simulator-only privileged **non-MCC** local trajectory optimization/MPC repair，并标记
`ORACLE_REPAIRED`。repair 必须保持窗口边界连续，记录逐帧 mask 与占比；Hard Guard 接管帧
不得作为普通 DP action label。

Teacher 正式冻结为：

```text
verified physical inverse demonstrations
+ privileged non-MCC contact-aware repair oracle
```

每个 replay sample 至少记录：

```text
source episode ID, inversion/time/action mapping, provenance, repair mask
q_f_meas, dq_f_meas
q_arm_meas, dq_arm_meas, q_arm_cmd
q_f_teacher_nominal_cmd, q_f_teacher_executed_cmd
f_des
F_raw, F_filtered, C, force_validity
contact/surface geometry, geometry source, SurfaceModelVersion
X_H_plan, delta_X_H_MCC, X_H_cmd, X_H_real, Wrist MCC state
guard state/reason, authority owner
authority solver success/violation/intervention/latency
new-contact authority-derivative rebase mask
episode termination reason, timestamps
```

Oracle 只存在于 simulator dataset generation，可访问 GT geometry、friction、penetration/contact
state 与 full robot state；部署 observation 不得包含这些 privileged fields。

当 free finger 的 MAKE 被真实 force/contact hysteresis 首次确认时，立即用新
`A_actual` 重建 `P_C`，并只把该指的 authority-filter derivative state 重置为零。上一条
实际执行命令必须保留，不能偷偷替换成 measured q；否则 actuator tracking error 会变成
set-point jump，并破坏 chunk seam/rate/acceleration continuity。该事件逐指记录为
`authority_transition_reset_mask`；它表示 derivative rebase，不表示 command-anchor jump。
v1 的 `A_actual` 使用 force 上/下阈值加时间确认：连续 5 个 500 Hz tick 才 MAKE，连续
5 个低于 release threshold 的 tick 才 BREAK；单帧 solver contact 不得触发 projector/reset。

完整数据门禁至少拒绝：时长不足、contact continuity 不足、tip over-force、non-tip/collision、
任何 recovery/guard takeover、authority QP failure、force sensor invalid，以及 privileged repair
占比过高。inverse proposal 的 SE(3) algebraic closure 只证明 proposal 变换正确，不能代替
physical replay acceptance。

### 6.5 Dataset-D 与 Dataset-I

- `Dataset-D / Direct-MCC diagnostic`：验证 observation、encoder、relative action、trainer、
  evaluator 与 inverse pipeline；不能承担正式方法贡献；
- `Dataset-I / Verified inverse`：真实 forward physical interaction → spatial proposal → fresh
  physical replay → verification → limited non-MCC repair；这是正式 teacher。

逐指 contact-mask agreement 只作为 diagnostic，不是 raw acceptance 的硬门槛。Primary contact
quality 以 `N_c(t)>=1`、zero-contact time、average/minimum contact count 和
`R_contact=sum_t N_c^R/sum_t N_c^F` 描述。Forward demonstration oracle 与 replay repair oracle
必须分别记录：前者可用于生成全部 forward command，后者修改 replay command，只有后者计入
`r_replay_repair`。

当前 3 秒最小闭环已做到：真实 moving-object forward、`SPATIAL_ONLY`、same-time 原始 forward
finger command、FR3+LEAP fixed-object replay、fresh force/contact、zero finger repair；raw replay
gate 通过。但 forward collector 使用 simulator Fingertip MCC，因此只能标记
`DATASET_D_DIAGNOSTIC`，仍不得训练。此前 hypothetical object motion 与临时 IK/linear
repair 的废弃链已从当前实现和生成数据中清理，不再提供训练或复现入口。

数据 ablation：Direct、Inverse only、Inverse+physical verification、Inverse+verification+repair。

## 7. Hard Runtime Guard

定义 signed compression coordinate `s_i(q)`：

```text
s_i increases <=> contact compression increases
J_s_i = partial s_i / partial q_i
```

deterministic hard release：

```text
Delta_q_release = -J_s^dagger delta_s, delta_s > 0
```

再通过 joint/rate saturation。禁止在协议里冻结依赖 normal convention 的裸 `+/-J_n`。

状态机：

```text
INITIALIZE -> BUFFER_FILL -> DP_ACTIVE -> SOFT_RECOVERY
           -> HARD_RELEASE -> SAFE_HOLD -> BUFFER_RESET -> BUFFER_FILL
```

- `SOFT_RECOVERY`：DP authority 收紧、wrist velocity 降低、探索暂停/减弱；
- `HARD_RELEASE`：DP authority 为零，执行 deterministic release；
- `SAFE_HOLD`：force 低于 recovery threshold 并稳定 `T_stable`；
- `BUFFER_RESET`：清除 hard intervention 之前的 force/action history；
- 只有 persistent hard violation、非有限状态或不可恢复模型/传感器错误才终止 episode。

Guard event 必须计入指标，但不能因为一次 event 立即结束仿真或视频。

## 8. Formal evaluation

Primary：

```text
E05-H-MCC vs E05-H-DP
```

两边共享 initial state、wrist planner/reference、Wrist MCC、SurfaceModel、force target、hard
guard、actuator limits、episode、seed 与 evaluator。唯一替换：

```text
Finger MCC <-> Finger DP + DP Action Authority Filter
```

`E05-F-DP` 只作为 standalone capability diagnostic，不承担主结论。

Wrist/Finger opposition 在 contact-normal space 中计算：

```text
nu^W = B_C xi_H^MCC
nu^(F,C) = P_C J_n Delta_q^DP
```

报告 `rho_opp` 和：

```text
E_opp = (1/T) integral [-<nu^(F,C), nu^W>]_+ dt
```

同时必须报告 authority-filter intervention probability/norm、QP failure/latency、guard takeover
rate/duration。若 filter 长期大幅修改 DP action，不得把性能全部归因于 DP。

## 9. Go/No-Go 顺序

1. observation/action/guard/filter unit tests；
2. full-length physical replay 与 dataset audit；
3. held-out prediction/control smoke test；
4. E05-H-DP 完整 paired episodes；
5. 仅在证明 DP feedback bandwidth 不足后，才允许讨论 bounded fast residual。
