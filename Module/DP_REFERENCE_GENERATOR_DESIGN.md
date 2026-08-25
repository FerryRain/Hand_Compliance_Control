# Finger DP Reference Generator 设计稿

> 日期：`2026-08-25`
>
> 状态：`IMPLEMENTED / ROLE_COVERAGE_LIMITED / EXP2_EVALUATED`
>
> 边界：接口已冻结并执行；E05 Exp.2 只描述策略性能，不承担 active-planner 解锁职责。

## 1. 方法定位

Finger DP 不再是 low-level compliance controller，而是 learned multi-finger nominal trajectory
generator：

```text
current hand/contact state + future wrist plan
 -> Finger DP Reference Generator
 -> nominal finger trajectory + contact-transition intent
 -> shared Finger MCC
 -> physical finger command
```

共享控制栈为：

```text
Wrist Planner -> Wrist MCC
Finger DPRef  -> Finger MCC
M03 Guard     -> independent veto/release/hold
```

## 2. 权限划分

| 状态/空间 | DPRef 权限 | Finger MCC 权限 | Wrist MCC 权限 |
| --- | --- | --- | --- |
| active contact tangent | nominal tangential/relative motion | 只做必要的 compliant execution | 无 |
| active contact normal | 不直接调高频 force | coordinated internal/differential force | collective/resultant force |
| free finger | full nominal reposition/approach trajectory | off | 无 |
| MAKE transition | approach reference，直到真实 contact 确认 | contact 后有界接管 normal compliance | collective branch 保持 |
| RELEASE transition | 提出 transition intent | 执行冻结的 compliant release primitive | 保持整体安全 |
| hard safety | 无权限 | 无权限 | 无权限；M03 接管 |

## 3. 第一版双 Head 输出契约（已冻结）

共享 encoder：

```text
h_t = E(o_t, wrist_plan[t:t+H])

Diffusion trajectory head:
  h_t -> q_nom[1:4, t+1:t+H]

Categorical role head:
  h_t -> logits[1:4, KEEP/RELEASE/FREE/MAKE]
```

连续 trajectory 使用 diffusion loss；离散 role 使用 masked, class-balanced cross entropy。Role
one-hot 不拼进 diffusion action vector。Role head 每次 inference 只提出每根手指的下一 role
intention，不能逐 horizon frame 直接控制执行状态。

选择 joint-space nominal chunk 是为了最大程度复用现有 I20 数据和 LEAP Hand FK；执行器再按
真实 contact normal 把 active-contact reference 分成 tangent 与 normal。若后续证据表明
Cartesian fingertip reference 更稳定，再作为版本化替代，不在同一实验中混用两种 action schema。

## 4. Reference/Role Interpreter（已冻结）

执行状态只能沿下列有向状态机变化，不能让分类结果直接覆盖当前 role：

```text
KEEP -> RELEASE -> FREE -> MAKE -> KEEP
```

分类 head 只提出 intention；deterministic interpreter 根据连续多次一致请求、真实接触迟滞、
force、geometry validity、whole-hand contact safety 与 M03 guard 批准或拒绝。

角色语义：

- `KEEP`：保留 DP nominal tangent，normal 交给 coordinated Finger MCC；
- `RELEASE`：冻结/衰减 tangential progression，将 `f_i_des` 从当前值平滑 ramp 到 0，由 MCC
  按 ramp 卸力；contact-loss 经过迟滞确认后才进入 `FREE`；
- `FREE`：DP nominal trajectory 具有完整 finger-space authority；
- `MAKE`：DP 生成 approach；真实接触确认后 blend 到 `KEEP`；
- 任一 RELEASE 若会使确认/可承载 contact 数低于 `N_min`，interpreter 必须否决；
- `MAKE -> KEEP` 只有在 `c_i=1` 且 `f_i>f_min` 持续 `T_make_confirm` 后批准；
- M03 hard guard 始终拥有更高权限，但 guard 事件不被重新解释成 learned role intention。

标准 handover 顺序为：

```text
MAKE(replacement)
 -> replacement contact+force confirmed
 -> RELEASE(old anchor), f_des ramps to zero
 -> old contact loss confirmed
 -> FREE(old anchor)
```

## 5. 输入与历史信息

可复用现有因果 observation：finger `q/dq`、filtered force/contact history、force/geometry
validity、contact geometry、target force、actual wrist history、future wrist plan 和 Wrist MCC
state。含义需改动的字段：

```text
q_nom_prev
q_meas
delta_q_MCC and/or complete Finger_MCC_state
```

Force history 仍用于识别稳定接触、handover、incipient loss 和 finger role，不再要求 DP 学习
500 Hz 的直接 force-release law。

## 6. 数据重标注实现与边界

现有 non-MCC Dataset-I 的 `q_teacher_cmd` 已被重新解释为 nominal reference；现有 I20
DP-direct checkpoint 没有复用为 DPRef checkpoint。实现做了：

1. 从真实 active-contact geometry 分解 teacher motion 的 tangent/normal 成分；
2. 根据经过时间确认的 intentional contact transition 生成 `KEEP/RELEASE/FREE/MAKE` labels；
3. 保存 teacher nominal reference、MCC correction 和最终 issued command 三者；
4. 训练 target 是 nominal reference chunk，不是 MCC 后的 actuator command；
5. 运行同一个 reference interpreter 与 command blending contract。

实际重标注复用了正式 nested I20/I100 RAW_VERIFIED episodes。I100 得到 55,900 samples；有效
role labels 中 KEEP/RELEASE/FREE/MAKE 分别为 `184467/95/26459/913`。validation 中只有
`7640/0/836/20`，即 RELEASE 完全缺失且 MAKE 很少。因此 raw trajectory 可复用，但现有 split
不足以验证 transition/handover 泛化。

### 6.1 Role label 的硬规则

禁止直接从相邻两帧 contact mask 推导 role。每指使用 filtered force + contact hysteresis +
未来 `Delta T_role` 窗口：

```text
stable 1 -> stable 1 : KEEP
stable 1 -> sustained unloading -> stable 0 : RELEASE
stable 0 -> stable 0 : FREE
stable 0 -> sustained approach/loading -> stable 1 : MAKE
```

只有持续卸力/approach 与稳定终态同时满足，才把 transition 作为 intentional label。单帧/短时
MuJoCo flicker、guard takeover、非意图 contact loss、语义不确定窗口全部设置
`role_label_valid=0`，不参与 role cross-entropy。Trajectory label 仍可按独立的数据有效性规则
决定是否保留。

## 7. Exp. 2 与最终 Exp. 3

- Exp. 2：固定 wrist trajectory 下，以普通 Plain MCC 为绝对参考，并严格比较
  `Passive-Hold+MCC`、`Reactive-Heuristic+MCC`、`DPRef+MCC` 三种共享栈 source；已完成接触
  优先重评，不设置策略 Pass/Fail；
- Exp. 3：放在 I05 之后作为 I06，完整 active exploration 下比较
  `Explicit Planner+MCC vs. Wrist-only Planner+DPRef+MCC`，隔离 online finger planning。

详细公平条件、指标和依赖见 [`E05_EVALUATION_PLAN.md`](E05_EVALUATION_PLAN.md)。

## 8. Exp. 2 的共享 reference pipeline

Passive、Reactive、DPRef 三个可归因分支全部经过：

```text
Reference source
 -> same Reference/Role Interpreter
 -> same Resultant/Internal Coordinator
 -> same Wrist MCC + Finger MCC
 -> same M03 Guard
```

- Passive-Hold：保持 nominal finger reference，不主动预测 transition；
- Reactive-Heuristic：只用当前/过去状态和 SurfaceModel，提供低力 normal correction、低 workspace
  margin reposition 与 nearest-surface MAKE；禁止读取 future wrist plan；
- DPRef：使用完全相同的可观测历史，并额外 condition future wrist plan，输出 nominal chunk 与
  role intention。

Exp. 2 已使用同一 interpreter、coordinator、Wrist/Finger MCC、M03 和物理条件运行。Passive 的
contact continuity/平均接触数/`N_c>=2` supported traversal 分别为
`0.972/2.285/89.35 mm`；Reactive 为 `0.973/2.310/86.90 mm`；DPRef 为
`0.988/2.450/126.09 mm`。因此 DPRef 在严格共享栈三者中这三项均最好，但四指同时接触率只有
`14.86%`，低于 Reactive 的 `27.45%`，第四指参与不足。普通 Plain MCC 的绝对参考为
`0.992/3.156/138.87 mm`，说明 shared stack 仍有接触保持差距。MuJoCo 力只作持续高力、
多指高力和 penetration 诊断，不以单个瞬时峰值判策略失败。role validation coverage 也不完整。
这些是策略优劣与适用范围描述，不是 Gate verdict。

实现和证据见：

- [`module_4_finger_dp/dpref_policy.py`](module_4_finger_dp/dpref_policy.py)
- [`module_4_whole_hand_mcc/reference_interpreter.py`](module_4_whole_hand_mcc/reference_interpreter.py)
- [`evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md`](evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)
- [`generated/exp2_dpref_mcc/README.md`](generated/exp2_dpref_mcc/README.md)
- [`generated/e05_exp1_exp2_review/index.html`](generated/e05_exp1_exp2_review/index.html)
