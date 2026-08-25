# I04 重定义与 M06–M12 迁移方案

> 日期：`2026-08-25`
>
> 状态：`REDESIGN_SPEC / IMPLEMENTATION_NOT_STARTED`
>
> 目的：记录 I04 的新实验定义，并明确 M06–M12 在新 I04 中应保留、删除、重构或迁移的职责。
> 本文只新增设计说明，不代表现有 `I04_ORACLE_NEXT_POINT_PROTOCOL.md`、现有 I04 代码或历史
> M06–M12 benchmark 已经完成迁移。现有 Explicit contact-mode implementation 与回归结果继续作为
>历史开发证据保留，不再作为新 I04 的目标架构。

---

## 1. 为什么需要重定义 I04

当前 I04 implementation 把问题定义为：GT Oracle 只给一个无 finger-ID 的 surface point，随后
Explicit planner 自己决定 explorer finger、SLIDE/MAKE/BREAK、contact-mode sequence、WRIST_ADJUST
和局部 realization。这个定义实际上把 I04 变成了一个高维显式 contact-mode planning 问题。

新的 I04 不再测试这个问题。

新的核心问题是：

> 在完整物体 GT 已知、下一步理想几何目标已经由 privileged Oracle 给出的情况下，机器人是否能够
> 在整手移动过程中维持真实 fingertip interaction，并最终把整个物体覆盖一遍？

I04 不研究 unknown-object active exploration，不研究 information gain，也不要求被测方法自己寻找
next-best target。GT Oracle 负责给“正确的下一步几何目标”；被测方法只负责实现这个目标。

新 I04 分为两个互补实验：

1. `I04-A: Wrist-only Oracle`：只给 wrist/palm 的外侧目标位置，比较最基础 Passive MCC 与
   DPRef+MCC 在没有 fingertip target 时的 whole-hand traversal 能力；
2. `I04-B: Full-hand Geometric Oracle`：同时给 wrist 与每根 fingertip 的精确几何目标位置，再用
   传统 trajectory planning + MCC 执行，测试在 high-dimensional target 已知时传统控制能否完成
   whole-object coverage。

二者都禁止 Oracle 输出 desired contact state、contact mode、MAKE/BREAK/KEEP/RELEASE 指令。
真实 contact 只能是执行时 observation、MCC feedback 和最终 evaluation signal，而不能是 Oracle
给出的答案。

---

## 2. 共同的 Privileged GT Coverage Oracle

I04 使用完整、准确、固定版本的 object GT mesh。Oracle 可以读取：

```text
- full object mesh / SDF / normals
- current real robot configuration
- current real fingertip contact points for coverage bookkeeping
- already-covered surface ledger
- robot joint/workspace/collision limits
```

Oracle 的职责是从尚未覆盖区域中选择下一个 physically feasible、具有 continuation 的理想目标。
它属于 privileged benchmark infrastructure，不属于被测控制方法。

建议 Oracle 内部统一构造一个完整的 latent geometric target：

```text
Y_k^GT = (
    p_H,k^*,
    x_1,k^*,
    x_2,k^*,
    x_3,k^*,
    x_4,k^*
)
```

其中：

- `p_H,k^*`：palm/wrist 的下一目标位置；
- `x_i,k^*`：第 `i` 根 fingertip 的下一几何目标位置；
- 不包含 `c_i^*`；
- 不包含 finger role；
- 不包含 contact mode；
- 不包含 MAKE/BREAK sequence。

如果后续发现 palm orientation 必须随完整物体几何改变，可以把 `p_H^*` 扩展为 `X_H^*`，但必须
单独冻结 orientation contract。第一版优先保持接口简单：比较变量是“是否暴露 fingertip geometric
target”，而不是额外改变 wrist orientation 信息。

### 2.1 Palm/Wrist target 必须位于物体外侧

I04-A 绝不把 palm contact 当成任务目标。真实系统中主要接触/力信息来自 fingertips，因此 palm
不需要接触物体，也不需要依赖 palm force feedback。

Oracle 应根据目标 surface region 生成物体外侧的 wrist/palm target，例如概念上：

```text
p_H^* = x_surface^* + d_offset * n_surface
```

其中 `d_offset > 0`，并通过完整 robot/object geometry 保证 palm、arm 与非允许 body 不穿入物体。
具体 offset、frame 和 orientation policy 后续作为 numerical protocol 冻结。

### 2.2 A/B 使用同一个 underlying Oracle，只改变暴露信息

公平比较建议采用同一个 `Y_k^GT`：

```text
I04-A adapter:
    Y_k^GT -> p_H,k^*

I04-B adapter:
    Y_k^GT -> (p_H,k^*, x_1,k^*, x_2,k^*, x_3,k^*, x_4,k^*)
```

因此 I04-A 与 I04-B 的核心信息差只有：

```text
是否显式提供 per-finger geometric targets
```

而不是：

```text
是否显式提供 contact-transition answer
```

---

## 3. I04-A：Wrist-only Oracle

### 3.1 实验目的

I04-A 回答：

> 当 global hand motion 已经由 GT Oracle 正确指定，但 high-dimensional fingers 没有任何显式目标时，
> 最基础的被动 MCC 能否随着 wrist motion 自然维持 interaction？DPRef 是否能够通过主动生成 finger
> reference，在相同 wrist motion 下显著提高 continuous traversal / coverage？

这是后续主方法 `wrist-only planning + learned finger realization` 的直接 oracle-level sanity check。

### 3.2 输入

I04-A 只向被测方法暴露：

```text
current measured robot state
future prescribed wrist trajectory tau_H
fingertip force/contact measurements required by the low-level controller
```

其中 Oracle 只提供：

```text
p_H,k^*
```

然后由简单几何 Wrist Trajectory Generator 生成：

```text
tau_H : p_H,current -> p_H,k^*
```

禁止给：

```text
x_i^*
target finger ID
SLIDE / REPOSITION / MAKE / BREAK
KEEP / RELEASE / FREE / MAKE role answer
contact-mode sequence
privileged witness finger/configuration
```

### 3.3 Wrist 的执行语义

Wrist 只做 prescribed geometric trajectory tracking。

I04-A 不要求 palm 接触物体，也不使用 palm-contact force objective。第一版不启用为了“让 palm 贴住
物体”而设计的 Wrist MCC / resultant-force loop；wrist 的目标是把整只 hand 从一个物体外侧位置带到
下一个物体外侧位置。

真正被测行为发生在这个过程中：

```text
wrist moves
 -> fingers are kinematically carried by the wrist
 -> fingertip controller adapts finger joints
 -> real fingertip contacts slide / persist / disappear / reappear naturally
```

### 3.4 I04-A-MCC：最基础 Passive MCC baseline

MCC branch 必须保持最弱、最基础、最可解释的 passive baseline：

```text
prescribed wrist trajectory
 -> basic fingertip MCC
 -> actuator command
```

要求：

- 不运行 M07 contact-mode planner；
- 不运行 Reactive-Heuristic finger reposition；
- 不给 free finger next-surface target；
- 不给 finger role intention；
- 不做显式 handover planning；
- 不根据 future wrist path 主动预测应该让哪根 finger MAKE/BREAK；
- active finger 只依据真实 fingertip contact/force 做局部 compliant correction；
- free finger 默认保持其 nominal/current reference，不主动寻找新接触。

因此 Passive MCC 能做的是：当 wrist 把 hand 带着移动时，根据局部 fingertip force 被动伸缩/柔顺，
尽可能维持已经存在的接触。若 workspace 不够、接触滑脱或 free finger 没有主动落点，系统不允许用
额外 planner 替它补答案。

### 3.5 I04-A-DPRef：learned finger realization

DPRef branch 接收与 MCC branch 完全相同的 wrist trajectory，并额外使用其正常可观测 hand/contact
history：

```text
current hand/contact state + future wrist plan
 -> DPRef
 -> nominal finger trajectory + learned role intention
 -> shared/basic Finger MCC compliance
```

DPRef 可以主动产生：

```text
- active-contact tangential/relative finger motion
- free-finger reposition
- approach toward a new contact
- learned release/handover intention
```

但这些行为来自 DPRef，不来自 GT Oracle 或 Explicit planner。

I04-A 的核心公平性要求是：

```text
same GT
same wrist targets
same wrist trajectory generator
same initial state
same physics
same force/contact sensing
same safety limits

only finger realization differs:
Passive MCC vs DPRef + MCC
```

---

## 4. I04-B：Full-hand Geometric Oracle

### 4.1 实验目的

I04-B 回答：

> 如果传统控制方法不仅知道 wrist 下一步应该去哪里，而且连四根手指的精确几何目标位置都已经由
> GT Oracle 给出，那么普通 whole-hand trajectory planning + MCC 是否能够完成整物体 coverage，
> 并在真实执行过程中自然维持接触？

这个实验是 privileged geometric upper bound / traditional-control reference。

它不比较“谁更会规划 finger”，因为 finger target 已经由 Oracle 给出；它测试的是：

```text
当 high-dimensional geometric answer 已知时，traditional control 是否有能力 physically realize it。
```

### 4.2 输入

I04-B 暴露：

```text
G_k^full = (
    p_H,k^*,
    x_1,k^*,
    x_2,k^*,
    x_3,k^*,
    x_4,k^*
)
```

明确禁止：

```text
c_i^*
KEEP / RELEASE / FREE / MAKE
SLIDE / MAKE / BREAK primitive
anchor/explorer role
contact-mode sequence
MAKE-before-BREAK plan
```

`x_i^*` 只是一个几何位置，不附带“这里必须处于 CONTACT/ FREE”的离散答案。

### 4.3 为什么不能给 desired contact state

I04-B 要测的一个核心 outcome 就是：

```text
trajectory execution 过程中，机器人自己是否能够保持真实接触。
```

如果 Oracle 直接输出：

```text
finger 1 KEEP
finger 2 MAKE
finger 3 RELEASE
...
```

或者把 `c_i^*` 作为 trajectory hard constraint，那么 benchmark 已经提前泄漏了 contact-maintenance
和 handover 答案，无法再判断传统控制本身是否自然保持 contact。

因此：

```text
contact state = measurement / feedback / evaluation
contact state != Oracle command
contact state != trajectory-planning target
```

Basic Fingertip MCC 仍然可以读取真实 fingertip force/contact 做局部 compliance，这是低层 feedback，
不是 desired contact-mode information。

### 4.4 Traditional Whole-hand Trajectory Planning

给定当前真实 configuration 与 `G_k^full`，trajectory planner 只做传统几何/运动学问题：

```text
current q, p_H
 + terminal wrist/fingertip positions
 -> IK / trajectory generation
 -> smooth whole-hand joint/task-space trajectory
```

允许使用：

```text
- current robot configuration
- full GT geometry for reachability/collision checking
- joint limits
- velocity/acceleration limits
- fingertip-object terminal geometry
- non-tip collision avoidance
- smoothness / path length objective
```

禁止使用：

```text
- planned contact modes
- desired contact occupancy
- MAKE/BREAK timing
- handover sequence
- shadow contact successor search
```

最重要的是：`continuous contact` 不作为 trajectory planner 的 hard constraint。它必须由真实 physics
执行后测量。否则 I04-B 会把要评价的结果直接写进约束。

### 4.5 I04-B 的执行控制

建议执行栈：

```text
GT full-hand geometric target
 -> traditional geometric trajectory generator
 -> basic MCC-compliant execution
 -> hard safety guard
 -> physics
```

这里 MCC 的作用是对 model error / local contact force 做柔顺修正，而不是重新决定 finger target 或
contact topology。

I04-B 核心 cell 不要求 DPRef；它主要作为 `Full-hand Geometric Oracle + Traditional Control + MCC`
参考上界。

---

## 5. 新 I04 中 contact 的权限边界

为避免再次把 task outcome 写回 planner，统一冻结：

### 5.1 可以读取真实 contact 的地方

```text
1. Fingertip MCC 的力/接触 feedback
2. DPRef 的正常 observation/history
3. coverage evaluator
4. contact continuity evaluator
5. diagnostic logging
6. hard hardware safety（仅当确实与硬件安全相关）
```

### 5.2 不允许使用 desired contact 的地方

```text
1. GT target interface
2. I04-A wrist trajectory generator
3. I04-B whole-hand geometric trajectory objective
4. M10 task-level execution certificate
5. privileged route adapter 暴露给 controller 的字段
```

尤其注意：原 I04 的 `MAKE-before-BREAK`、`last-contact veto` 如果被作为 task-level hard guard 使用，
会直接帮助系统维持 contact，从而污染 I04 要测的 contact-continuity outcome。

新 I04 中应区分：

```text
hardware safety guard:
    joint / collision / actuator / force emergency
    -> 可以 veto

task contact continuity:
    whether sum_i c_i_real(t) >= 1
    -> 只测量，不由 guard 保证
```

如果真实机器人部署阶段必须保留某个 contact-loss safety rule，必须单独报告该 intervention，并在所有
cells 完全一致；正式 benchmark 第一版优先不使用 task-level last-contact veto。

---

## 6. M06–M12 的新职责

新 I04 不再需要 `M07 -> M08 -> M09 -> M11/M12` 这一条 Explicit contact-mode search chain。
现有 M06–M12 代码和 benchmark 可以作为历史 baseline evidence 保留，但新 I04 runtime path 应按下表
迁移。

| Module | 新 I04 状态 | 新职责 |
| --- | --- | --- |
| M06 | `KEEP / MODIFY` | 通用 trajectory/prefix executor + fresh-state barrier |
| M07 | `REMOVE_FROM_I04` | 不再枚举 contact modes；历史 Explicit baseline 保留 |
| M08 | `MOVE_TO_ORACLE_SIDE / OPTIONAL` | 若复用，只做 privileged geometric target feasibility，不做 primitive CheapCert |
| M09 | `REDEFINE` | 从 primitive ContinuousOptimize 改为 geometric trajectory generation |
| M10 | `KEEP / SIMPLIFY` | 只做 swept hardware-safety / integrity audit，不保证 task contact topology |
| M11 | `REMOVE_FROM_I04` | 不再 beam-search SLIDE/MAKE/BREAK/WRIST sequence |
| M12 | `MOVE_TO_ORACLE_SIDE` | continuation/coverage viability 属于 privileged GT route，不属于 controller runtime |

下面给出逐模块修改要求。

---

## 7. M06：保留为通用 Transactional Trajectory Executor

M06 的 transaction / short-prefix / barrier 思想仍然有价值，但它不再只接受 Explicit contact primitive。

建议新接口：

```text
TrajectoryPlan
 -> split into short committed prefix
 -> M10 safety audit
 -> M06 execute prefix
 -> fresh measured robot state
 -> continue/replan
```

M06 保留：

```text
- short committed prefix
- participant completion bookkeeping
- timeout / SAFE_HOLD for hardware/control failure
- stale-plan rejection
- fresh real-state barrier
- command provenance
```

M06 删除/弱化：

```text
- 对 ContactModeGraph edge identity 的依赖
- MAKE/BREAK transaction type 的特殊执行授权
- prediction suffix 来自 M11 的假设
- task-level last-contact preservation authority
```

I04-A 中，M06 执行 wrist prefix，同时 MCC/DPRef 在同一时间连续运行 finger control。

I04-B 中，M06 执行 whole-hand trajectory prefix。

---

## 8. M07：从新 I04 执行路径移除

原 M07 的作用：

```text
枚举 15 个 nonempty contact modes
枚举 WRIST / SLIDE / REPOSITION / MAKE / BREAK
检查 mode transition legality
```

这些都不是新 I04 的问题。

因此：

```text
I04-A: 禁止调用 M07
I04-B: 禁止调用 M07
```

现有 `module_7_contact_mode_graph/` 不删除，可保留用于：

```text
- 历史 Explicit baseline
- 旧 I01–I03 / M06–M12 复现
- 后续若需要 explicit high-dimensional planning ablation
```

但新 I04 不再依赖它。

---

## 9. M08：从 primitive CheapCert 迁移为 Oracle-side feasibility（可选）

原 M08 为大量 `SLIDE/MAKE/BREAK/...` candidates 做 cheap screening。新 I04 没有这种 candidate
explosion，因此该职责不再必要。

如果希望保留 M08 编号，可以将它迁移为 privileged `OracleTargetFeasibility`：

### I04-A

检查候选 `p_H^*`：

```text
- FR3 reachability
- wrist/palm/arm collision
- target outside object
- trajectory corridor roughly feasible
```

### I04-B

检查候选 `(p_H^*, x_1^*, ..., x_4^*)`：

```text
- terminal whole-hand IK exists
- joint margin valid
- non-tip collision valid
- geometric target is physically meaningful
```

M08 的结果只用于 GT Oracle 不把明显不可能的 target 交给 benchmark，不向 controller 泄漏 witness
configuration/finger mode。

如果 Oracle 自己已有这套 certification，则 M08 可以完全不进入新 I04。

---

## 10. M09：重定义为 Geometric Trajectory Generator

M09 是新 I04 中唯一需要明显“换任务”的 planning module。

### 10.1 I04-A：WristTrajectoryGenerator

输入：

```text
current wrist/palm position
p_H^*
```

输出：

```text
tau_H = {p_H(t)}
```

要求：

```text
- smooth
- joint/velocity feasible
- palm/arm non-tip collision safe
- 不读取 fingertip target
- 不读取 desired contact mode
- 不为了保持 contact 优化 finger joints
```

### 10.2 I04-B：WholeHandGeometricTrajectoryGenerator

输入：

```text
current robot configuration
p_H^*
x_1^*, x_2^*, x_3^*, x_4^*
```

输出：

```text
tau_Q = {q_arm(t), q_finger(t)}
```

可以用连续 IK、trajectory optimization、task-space interpolation 或 joint-space planning，只要统一冻结。

要求：

```text
- terminal geometric target error within tolerance
- joint/velocity/acceleration limits
- non-tip collision avoidance
- smoothness
- 不枚举 contact mode
- 不优化 MAKE/BREAK sequence
- continuous-contact 不作为 hard constraint
```

建议代码层面不要继续沿用 `ContinuousOptimize(primitive)` 语义。可以保留 M09 ID，但公开类名改为：

```text
WristTrajectoryGenerator
WholeHandGeometricTrajectoryGenerator
```

---

## 11. M10：保留，但改成纯 Safety / Integrity Audit

M10 仍然很有价值，因为 planned trajectory 在真正发给机器人前需要 swept-path audit。

新 M10 可以检查：

```text
- joint limits
- velocity/acceleration limits
- arm/palm/non-tip self/object collision
- trust region / prefix integrity
- stale model/state provenance
- hard force emergency rule（若能在执行前定义）
- command digest / authority
```

新 M10 不应检查或保证：

```text
- expected contact mode
- MAKE confirmation
- BREAK legality
- last-contact preservation
- anchor preservation
- terminal successor contact availability
```

原因：这些 task-level contact rules 会把“能否保持接触”从 evaluation 变成 controller 的硬约束。

M10 的原则变为：

```text
safe to execute != guaranteed to preserve contact
```

---

## 12. M11：从新 I04 移除

原 M11 搜索：

```text
SLIDE / MAKE / BREAK / REPOSITION / WRIST_ADJUST sequences
```

新 I04 不需要这一层。

```text
I04-A:
    finger behavior = Passive MCC or DPRef

I04-B:
    finger targets = GT Oracle
    path realization = conventional trajectory generator
```

因此没有 contact-mode beam search 的对象。

现有 M11 继续作为历史 Explicit baseline / I03 evidence 保留，但不能作为 I04-A/B 的 hidden runtime
fallback。

---

## 13. M12：continuation 思想迁移到 GT Coverage Oracle

“不要给一个会把未来彻底走死的目标”仍然重要，但这是 privileged Oracle 的职责，而不是被测
controller 的职责。

因此 M12 的新位置是：

```text
full GT coverage ledger
 -> candidate next geometric targets
 -> privileged continuation / reachability check
 -> choose next Y_k^GT
```

它可以保证：

```text
- 当前 target 可达
- 后续仍有 reachable uncovered region
- route 不因一个局部 target 进入明显 dead end
```

但 M12 不向 I04-A/B controller 暴露：

```text
- future finger assignment
- contact mode
- handover sequence
- witness configuration
```

如果 GT route 采用全局离线 trajectory/graph certification，则不必保留单独的 M12 runtime module。

---

## 14. 新 I04 的完整执行流程

### 14.1 I04-A-MCC

```text
Full Object GT + real coverage ledger
 -> Privileged Coverage Oracle
 -> next outside-object wrist target p_H^*
 -> M09-A WristTrajectoryGenerator
 -> M10 hardware-safety audit
 -> M06 execute wrist prefix
      + basic Passive Fingertip MCC continuously runs
 -> real physics/contact/force
 -> fresh state + actual touched-surface update
 -> next wrist prefix / next Oracle target
 -> repeat until coverage budget/goal reached
```

### 14.2 I04-A-DPRef

```text
Full Object GT + real coverage ledger
 -> SAME Privileged Coverage Oracle
 -> SAME p_H^*
 -> SAME M09-A WristTrajectoryGenerator
 -> M10 hardware-safety audit
 -> M06 execute wrist prefix
      + DPRef receives future wrist plan
      + DPRef generates nominal finger references / role intentions
      + Finger MCC provides local compliance
 -> real physics/contact/force
 -> fresh state + actual touched-surface update
 -> repeat
```

A-MCC 与 A-DPRef 必须使用 paired target sequence policy；若闭环状态分叉导致 Oracle 必须重新选择 target，
则两边使用同一个 Oracle 函数作用于各自真实状态，并记录 target difficulty/provenance。

### 14.3 I04-B-MCC

```text
Full Object GT + real coverage ledger
 -> Privileged Coverage Oracle
 -> next full-hand geometric target
      (p_H^*, x_1^*, x_2^*, x_3^*, x_4^*)
 -> M09-B WholeHandGeometricTrajectoryGenerator
 -> M10 hardware-safety audit
 -> M06 execute whole-hand trajectory prefix
      + basic MCC local compliance
 -> real physics/contact/force
 -> fresh state + actual touched-surface update
 -> repeat until coverage budget/goal reached
```

I04-B 不使用 DPRef、M07、M11 或 contact-mode fallback。

---

## 15. Goal completion 与 Coverage completion 必须重新定义

旧 I04 使用“某根真实 fingertip 到达 Oracle surface goal”作为每个 goal 的 ARRIVE。新 I04 不再使用
这个定义。

### I04-A target completion

当前 target 是 wrist target，因此单步完成只检查 wrist geometric tracking：

```text
||p_H,real - p_H^*|| <= epsilon_H
```

不要求某根指定 finger 到达某个点。

### I04-B target completion

当前 target 是 full-hand geometric target，因此单步完成检查：

```text
||p_H,real - p_H^*|| <= epsilon_H
and
||x_i,real - x_i^*|| <= epsilon_f    for required geometric targets
```

这里仍然不使用 `c_i_real` 作为 target-arrival 条件。

### Whole-object coverage completion

真正的 I04 任务完成由真实 fingertip contact 产生的 surface coverage 决定：

```text
Coverage(t) = fraction of required GT surface covered by REAL fingertip contacts
```

只有真实 physics contact 可以更新 coverage ledger。planned fingertip path、target position、DPRef prediction
或 Oracle witness 都不能计入 coverage。

可继续使用 mesh geodesic / surface-radius coverage 定义，但必须在新的 numerical protocol 中重新冻结：

```text
- required surface set / area weighting
- contact-to-surface association
- coverage radius
- completion threshold
- total time / path budget
```

---

## 16. 新 I04 的主要指标

### 共同指标

```text
- final real surface coverage
- coverage AUC / coverage per unit wrist path
- time/path length to reach target coverage
- hand-level contact continuity R_contact
- maximum all-finger contact-loss gap T_gap_max
- distribution of zero-contact intervals
- mean / minimum simultaneous contact count
- fingertip force statistics
- joint/workspace margin
- non-tip collision / safety intervention count
- wrist geometric tracking error
- compute latency
```

### I04-A 额外指标

用于直接比较 Passive MCC vs DPRef：

```text
- coverage under identical wrist-path budget
- continuous supported traversal distance
- contact regain count/time
- finger workspace utilization
- DPRef role/transition diagnostics（仅解释，不作为 Oracle input）
```

### I04-B 额外指标

```text
- per-finger target tracking error
- whole-hand geometric trajectory success
- contact continuity despite no desired contact-state command
```

---

## 17. 预期实验解释

最终至少形成三个核心 cell：

| Cell | Wrist target | Finger target | Finger realization |
| --- | --- | --- | --- |
| `I04-A-MCC` | GT | none | basic passive MCC |
| `I04-A-DPREF` | same GT | none | DPRef + MCC |
| `I04-B-MCC` | GT | GT per-finger positions | traditional trajectory planning + MCC |

理想情况下，若观察到：

```text
I04-A-MCC      < I04-A-DPREF ~= I04-B-MCC
```

则支持以下解释：

1. traditional MCC/robot control 在 high-dimensional geometric targets 已知时具备完成 coverage 的物理能力；
2. 仅给 wrist/global motion 时，passive compliance 本身不足以持续解决 finger workspace、contact regain 和
   handover；
3. DPRef 能够在不显式规划 high-dimensional fingers 的情况下恢复大部分 full-hand geometric Oracle 能力。

这比“Explicit contact-mode planner 能否猜出哪根 finger 去一个 surface point”更直接对应项目核心问题。

---

## 18. 与当前仓库实现的迁移关系

当前以下内容均应视为 `LEGACY_EXPLICIT_I04`，在新 I04 完成前不要删除：

```text
I04_ORACLE_NEXT_POINT_PROTOCOL.md
I04_RESUME_CHECKPOINT_2026-08-25.md
i04_oracle_next_point/planner.py 中的 explicit finger/contact-mode search
M07 ContactModeGraph
M08 primitive CheapCert
M09 primitive ContinuousOptimize
M11 Lazy Beam Search
M12 Shadow Succ
现有 {2,4} two-anchor WRIST optimization debug evidence
```

它们仍可用于历史复现和 explicit-planning ablation，但不再是新 I04 implementation blocker。

### 推荐代码迁移顺序

1. 冻结本文为新的 I04 design source of truth；
2. 保留旧 I04 文件，但在标题/README 中标记 legacy explicit development；
3. 新建独立的 I04 runner，不在旧 `planner.py` 上继续叠加条件分支；
4. 先实现 `I04-A-MCC`：GT wrist route + prescribed wrist motion + basic fingertip MCC；
5. 接入同一 wrist route 的 `I04-A-DPREF`；
6. 重构 M09-B，实现 full-hand geometric trajectory planning；
7. 实现 `I04-B-MCC`；
8. 统一 coverage/contact/force evaluator；
9. 再决定是否把 M08/M12 的 privileged feasibility/continuation helper 复用到 Oracle 内部。

在 1–7 完成前，不应继续把当前 `{2,4}` blocker 当作新 I04 的主要工程阻塞。

---

## 19. 新 I04 的一句话冻结定义

```text
I04 uses full object GT to provide ideal geometric traversal targets.

A: expose only an outside-object wrist target and compare
   basic passive MCC vs DPRef+MCC while the wrist carries the hand.

B: expose wrist + per-finger geometric target positions and test
   traditional whole-hand trajectory planning + MCC.

Neither branch receives desired contact states or contact-mode instructions;
continuous fingertip contact and surface coverage are measured outcomes.
```
