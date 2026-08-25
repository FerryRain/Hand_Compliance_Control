# I04 重定义、I01–I03 退役与旧 Mx 能力迁移方案

> 日期：`2026-08-25`
>
> 状态：`REDESIGN_SPEC / IMPLEMENTATION_NOT_STARTED`
>
> 本文是新 I04 的设计迁移说明。它重新定义 I04 的实验问题，并说明历史 I01–I03 与旧 M06–M12 能力在新架构中的去留。
> 现有 `I04_ORACLE_NEXT_POINT_PROTOCOL.md`、I01–I03、M06–M12 Explicit implementation 和物理回归结果继续保留为历史证据，
> 但不再作为新 I04 的目标架构或前置 Gate。

---

# 0. 第一原则：新体系不再以 M06–M12 作为正式架构

**在新的 I04 体系下，不需要继续维护 `M06 -> M07 -> ... -> M12` 这种模块链。**

这些 Mx 编号是围绕旧 `Explicit contact-mode planner` 逐步搭出来的工程分解；它们服务的问题是：

```text
枚举 contact mode
选择 SLIDE / MAKE / BREAK / REPOSITION
优化某一个 primitive
搜索未来 primitive sequence
检查 contact successor
再做 certificate-gated execution
```

新的 I04 已经不再解决这个问题，因此不应该为了复用旧编号而把新系统继续切成 M06–M12。

新 I04 的正式主路径应该直接按**功能层**组织：

```text
1. Privileged GT Coverage Oracle
2. Shared Geometric Trajectory Planner
3. Finger Realization
      A-MCC   : Basic Passive MCC
      A-DPRef : DPRef + Basic MCC
      B-MCC   : planned whole-hand trajectory + Basic MCC
4. Hardware Safety / Trajectory Integrity
5. Trajectory Executor + Fresh-State Feedback
6. Real Contact / Coverage Evaluator
```

也就是说，后文仍出现 `M06/M08/M09/M10/M12` 时，只表示：

> **旧模块中哪些能力值得迁移到新的功能层。**

它不表示新 I04 仍然要保留这些 module ID、目录边界或调用链。

`M07`、`M11` 以及旧 contact-mode 语义应直接退出 active path；其他仍有价值的能力应被**重命名、重组并并入新架构**，而不是继续以旧 Mx 形式存在。

M0–M3 已经完成的 robot/sensor/basic controller 能力可以继续复用，但新的 I04 也不要求继续以 `Mx` 编号作为设计中心。后续若整理仓库，可以把“编号”降级为 historical implementation label，把正式架构改为功能命名。

## 0.1 新 I04 的第一步不是继续调 planner，而是先做架构清理

在实现新的 I04-A/B 之前，第一步必须先完成以下 cleanup：

```text
Step 1: 从 new-I04 active runtime path 移除不需要的旧模块
    - M07 ContactModeGraph
    - M11 Lazy Beam Search
    - runtime M12 ShadowSucc
    - primitive-style M08 CheapCert
    - primitive-style M09 ContinuousOptimize
    - contact-mode-specific M10 rules
    - contact-mode-specific M06 transaction semantics

Step 2: 把仍然需要的能力重新命名并迁移
    old M09 capability
        -> Shared Geometric Trajectory Planner

    old M10 useful capability
        -> Hardware Safety / Trajectory Integrity Checker

    old M06 useful capability
        -> Generic Trajectory Executor + Fresh-State Feedback

    old M08/M12 useful privileged capability
        -> GT Oracle Reachability / Continuation helpers

Step 3: 更新代码依赖
    - new I04 runner 不 import M07/M11 contact-mode planner
    - new planner 不产生 SLIDE/MAKE/BREAK/REPOSITION primitive
    - A-MCC/A-DPRef 共享同一个 wrist SE(3) trajectory planner
    - B 使用同一个 planning framework，并额外加入 fingertip geometric objectives
    - controller path 不读取 desired contact modes

Step 4: 更新仓库文档与状态
    - README / MASTER_PLAN 标记旧 I01–I03、M06–M12 为 LEGACY / EVIDENCE
    - G2/G3 不再作为 new-I04 Gate
    - 旧 Explicit I04 regression 保留但不再继续调参

Step 5: 完成以上迁移后，才开始新的 I04-A/B 正式实现与评测
```

因此当前 `{2,4}` two-anchor blocker、M09 WRIST optimization、BREAK gating、M11 search 等问题，**从现在开始都不应再作为新 I04 的工程优先级**。

新的实现优先级是：

```text
clean old active path
    -> define new functional interfaces
    -> build shared SE(3) trajectory planning
    -> validate basic execution/safety
    -> implement I04-A-MCC
    -> implement I04-A-DPRef
    -> implement I04-B-MCC
```

---

## 1. 新 I04 到底要回答什么

旧 I04 把问题定义成：GT Oracle 只给一个无 finger-ID 的 surface point，然后 Explicit planner 自己决定：

```text
哪个 finger 去
SLIDE / REPOSITION / MAKE / BREAK
什么时候 WRIST_ADJUST
contact-mode sequence
handover sequence
```

这实际上把 I04 变成了一个高维显式 contact-mode planning 问题。

新的 I04 不再测试这个问题。

新的核心问题是：

> 在完整物体 GT 已知、下一步理想几何目标已经由 privileged Oracle 给出的情况下，机器人能否通过正常的轨迹规划和低层控制，
> 在整手移动过程中维持真实 fingertip interaction，并最终覆盖整个物体？

I04 不研究 unknown-object active exploration，也不研究 information gain / next-best-touch selection。
GT Oracle 负责提供“正确的下一目标”；被测方法只负责实现这些目标。

新 I04 分成两个互补实验：

1. **I04-A: Wrist-only Oracle**
   - Oracle 只暴露完整 wrist/palm 目标位姿 `X_H^* ∈ SE(3)`；
   - 比较最基础 Passive MCC 与 DPRef+MCC；
   - 两边使用完全相同的 wrist trajectory；
   - 不给任何 fingertip geometric target。

2. **I04-B: Full-hand Geometric Oracle**
   - Oracle 暴露同一个 wrist pose `X_H^*`，再额外暴露四根 fingertip 的几何目标位置；
   - 使用传统 whole-hand trajectory planning + basic MCC 执行；
   - 测试 high-dimensional geometric target 已知时，传统 robot control 是否能完成 coverage。

两个实验都明确禁止 Oracle 输出：

```text
desired contact state
contact mode
KEEP / RELEASE / FREE / MAKE
SLIDE / REPOSITION / MAKE / BREAK
anchor/explorer role
handover sequence
MAKE-before-BREAK answer
```

真实 contact 只能是 controller feedback、physics outcome、coverage update 和 evaluation signal。

---

# 2. Privileged GT Coverage Oracle

I04 使用完整、准确、固定版本的 object GT mesh / SDF。

Oracle 可以读取：

```text
full object mesh / SDF / normals
current real robot configuration
current real wrist pose
current real fingertip contact points for coverage bookkeeping
already-covered surface ledger
robot joint/workspace/collision limits
```

Oracle 的职责是从尚未覆盖区域中选择下一组 physically meaningful、可达并具有 continuation 的理想几何目标。

建议 Oracle 内部统一构造：

```text
Y_k^GT = (
    X_H,k^*,
    x_1,k^*,
    x_2,k^*,
    x_3,k^*,
    x_4,k^*
)
```

其中：

```text
X_H,k^* = (p_H,k^*, R_H,k^*) ∈ SE(3)
x_i,k^* ∈ R^3
```

注意：**wrist orientation 从第一版开始就是必须的，不存在 position-only 的正式 I04。**

整只手要绕完整物体移动时，不可能依靠一个固定 palm orientation 覆盖侧面、背面、耳朵、底部等区域。
如果只改变 `p_H` 而固定 `R_H`，会人为限制 finger workspace，也会让不同 surface region 的接触几何不合理。

因此新的正式 wrist target 始终是：

```text
X_H^* = (position + orientation)
```

而不是：

```text
p_H^* only
```

具体 `R_H^*` 的构造规则需要在 numerical protocol 中冻结，例如可以由：

```text
local surface normal
local traversal tangent
palm approach axis
hand preferred roll / opposition geometry
```

共同定义，但“不旋转 wrist”不再是合法正式配置。

---

## 2.1 Wrist/Palm target 必须位于物体外侧

I04 不把 palm contact 当成任务目标。

真实系统中主要触觉/力信息来自 fingertips，palm 本身没有必要贴住物体，也不应依赖 palm-contact force 来完成 exploration。

因此 Oracle 生成的 wrist/palm pose 应位于物体外侧，例如位置部分可以概念性写成：

```text
p_H^* = x_region^* + d_offset * n_region
```

其中 `d_offset > 0`。

同时 orientation `R_H^*` 根据该区域局部几何旋转，使 hand 的有效 finger workspace 面向待探索区域。

Oracle 必须保证：

```text
palm outside object
arm/palm/non-tip bodies do not penetrate object
wrist pose is reachable
orientation is kinematically meaningful for the hand
```

---

## 2.2 A/B 使用同一个 underlying Oracle

A/B 最干净的公平设计是共享同一个 latent target：

```text
Y_k^GT = (X_H,k^*, x_1,k^*, x_2,k^*, x_3,k^*, x_4,k^*)
```

然后只通过不同 adapter 暴露不同信息：

```text
I04-A adapter:
    Y_k^GT -> X_H,k^*

I04-B adapter:
    Y_k^GT -> (X_H,k^*, x_1,k^*, x_2,k^*, x_3,k^*, x_4,k^*)
```

因此 A/B 的核心信息差只有：

```text
是否显式暴露 per-finger geometric targets
```

不是：

```text
是否告诉 contact-transition answer
```

---

# 3. 共享 Geometric Trajectory Planning Layer

**Oracle 给目标，不等于机器人直接移动到目标。所有 I04 cells 都必须经过一个明确的 trajectory planning layer。**

统一结构是：

```text
GT Coverage Oracle
        ↓
Geometric Target
        ↓
Shared Geometric Trajectory Planner
        ↓
Nominal trajectory
        ↓
Controller / DPRef realization
        ↓
Safety audit + execution
        ↓
Physics
```

这层 trajectory planner 与 Oracle、MCC、DPRef 是三个不同职责：

```text
Oracle:
    决定“下一目标在哪里”

Trajectory Planner:
    决定“从当前状态怎样平滑、安全地运动到该目标”

Controller / DPRef:
    决定“如何在真实动态和接触误差下执行 nominal trajectory”
```

---

## 3.1 A-MCC 与 A-DPRef 必须共享完全相同的 wrist trajectory planner

I04-A 中，输入是：

```text
current measured robot state
current wrist pose X_H,current
next wrist target pose X_H^*
```

共享 planner 输出：

```text
tau_H = {X_H(t)}_{t=0:T}
```

这里 `X_H(t)` 包含：

```text
position trajectory
orientation trajectory
```

例如 orientation interpolation 可以使用 SO(3)/quaternion interpolation；具体实现必须在正式 protocol 中统一冻结。

**A-MCC 和 A-DPRef 不能各自生成不同 wrist path。**

必须满足：

```text
same X_H^*
same trajectory planner
same tau_H
same timing / velocity / acceleration limits
same collision constraints
same wrist tracking controller
```

唯一主要区别是 finger realization：

```text
A-MCC   : passive fingertip MCC
A-DPRef : DPRef nominal finger motion + fingertip MCC
```

这样才能真正回答 DPRef 是否帮助 wrist-only whole-hand traversal。

---

## 3.2 I04-B 使用同一 planning framework，但增加 fingertip geometric objectives

I04-B 不是“拿到目标点以后直接 joint interpolation”。

它同样经过 geometric trajectory planning，只是目标约束更多：

```text
current robot state
+
X_H^*
+
x_1^*, x_2^*, x_3^*, x_4^*
        ↓
Whole-hand Geometric Trajectory Planner
        ↓
nominal q_arm(t), q_finger(t)
```

A/B 最好共用同一个 planning framework 和基础约束：

```text
same robot model
same collision checker
same joint limits
same velocity/acceleration limits
same time parameterization
same wrist pose objective implementation
same planning tolerances
```

区别是：

```text
A:
    active task objective = wrist pose X_H^*
    no fingertip geometric target

B:
    active task objectives = wrist pose X_H^* + fingertip positions x_1:4^*
```

因此不是两个完全无关的 planner，而是同一 geometric planning layer 的两个 task-specification modes。

---

# 4. I04-A：Wrist-only Oracle

## 4.1 实验目的

I04-A 回答：

> 当 global wrist motion 已经由 GT Oracle 正确指定、并通过共同 trajectory planner 生成完整 SE(3) wrist trajectory，
> 但 high-dimensional fingers 没有任何显式目标时，最基础 passive MCC 能否随着 wrist motion 自然维持 fingertip interaction？
> DPRef 是否能够在相同 wrist trajectory 下主动协调 fingers，从而获得更好的 traversal / coverage？

这是主方法：

```text
wrist-only planning + learned finger realization
```

最直接的 oracle-level sanity check。

---

## 4.2 I04-A 输入

Oracle 只向 A 暴露：

```text
X_H,k^*
```

随后 shared trajectory planner 产生：

```text
tau_H = {X_H(t)}
```

A 的被测 finger method 可以读取正常真实观测，但禁止获得：

```text
x_i^*
target finger ID
privileged future finger configuration
explicit contact mode
SLIDE / REPOSITION / MAKE / BREAK
Oracle-generated role answer
```

---

## 4.3 Wrist 执行语义

Wrist 负责按照共享 trajectory planner 的 `tau_H` 运动。

它不是为了“让 palm 压住 object”，也不需要 palm-contact force objective。

真实被测行为是：

```text
wrist translates + rotates around the object
 -> fingers are kinematically carried by the moving wrist
 -> finger controller adapts finger joints
 -> contacts slide / persist / disappear / reappear naturally
```

---

## 4.4 I04-A-MCC：Basic Passive MCC

MCC branch 使用最基础、最弱、最可解释的 passive fingertip MCC：

```text
shared planned wrist trajectory tau_H
        ↓
wrist tracking
        +
basic passive fingertip MCC
        ↓
physics
```

它不做：

```text
finger trajectory planning
finger target prediction
free-finger reposition planning
contact-mode search
handover planning
future-wrist-conditioned finger action
```

active finger 只根据真实 fingertip force/contact 做被动 local compliance correction。

free finger 默认保持 nominal/current reference，不主动寻找下一接触点。

因此 A-MCC 测的是最基础问题：

> 当 wrist 把整只 hand 带着平移和旋转时，纯 passive MCC 本身能把已有 contact 保持多久、能覆盖多少？

---

## 4.5 I04-A-DPRef

DPRef branch 使用完全相同的 `tau_H`：

```text
current hand/contact history
+
future wrist pose trajectory tau_H
        ↓
DPRef
        ↓
nominal finger trajectory + learned role intention
        ↓
basic/shared Fingertip MCC
        ↓
physics
```

DPRef 可以主动产生：

```text
active-contact tangential/relative finger motion
free-finger reposition
approach to a new surface region
release / handover intention
workspace management
```

但这些行为必须来自 DPRef，而不是 GT Oracle 或 Explicit planner。

---

# 5. I04-B：Full-hand Geometric Oracle

## 5.1 实验目的

I04-B 回答：

> 如果传统 robot control 不仅得到下一 wrist pose，而且连四根 fingertip 的精确几何目标都已经由 GT Oracle 给出，
> 那么传统 whole-hand trajectory planning + passive/basic MCC 是否有能力完成 whole-object coverage？

这是一个 privileged geometric upper bound / traditional-control reference。

它要说明：

```text
如果 high-dimensional geometric answer 已知，
robot + trajectory planning + MCC 本身有没有能力完成任务。
```

---

## 5.2 I04-B 输入

I04-B 暴露：

```text
G_k^full = (
    X_H,k^*,
    x_1,k^*,
    x_2,k^*,
    x_3,k^*,
    x_4,k^*
)
```

其中 `X_H^*` 是完整 SE(3) wrist pose。

明确禁止：

```text
c_i^*
KEEP / RELEASE / FREE / MAKE
SLIDE / MAKE / BREAK primitive
anchor/explorer role
contact-mode sequence
MAKE-before-BREAK sequence
```

`x_i^*` 是纯几何 fingertip target，不携带离散 desired contact label。

这些 fingertip target 可以位于 GT object surface 上，从而定义“下一组应达到的 hand geometry”；
但**执行过程中是否连续保持 contact、何时出现/失去接触，仍然完全由 physics 决定。**

---

## 5.3 Traditional Whole-hand Trajectory Planning

输入：

```text
current measured q / X_H
+
X_H^*
+
x_1:4^*
```

输出：

```text
tau_Q = {
    q_arm(t),
    q_finger(t)
}_{t=0:T}
```

允许使用：

```text
IK / differential IK
trajectory optimization
task-space interpolation
joint-space time parameterization
full GT geometry for collision checking
joint limits
velocity / acceleration limits
smoothness / path-length cost
```

不允许使用：

```text
desired contact occupancy
contact-mode graph
MAKE/BREAK timing
handover sequence
ShadowSucc contact search
continuous-contact hard constraint
```

这里非常重要：

```text
trajectory planner 的任务 = geometric realization
contact continuity = physical outcome
```

不能把“必须始终至少一指 contact”写成 trajectory planner 的 hard task constraint，否则就提前把 I04-B 要测的结果保证掉了。

---

## 5.4 I04-B 的执行控制

```text
GT full-hand geometric target
        ↓
Shared Geometric Trajectory Planning framework
        ↓
nominal whole-hand trajectory
        ↓
basic MCC compliant execution
        ↓
hard hardware safety
        ↓
physics
```

MCC 只做 model error / local force 的柔顺修正，不重新决定 fingertip target，也不决定 contact topology。

---

# 6. Contact 的统一权限边界

## 6.1 可以读取真实 contact 的地方

```text
Fingertip MCC force/contact feedback
DPRef normal observation/history
coverage evaluator
contact-continuity evaluator
diagnostic logging
hard hardware emergency logic
```

## 6.2 不允许 desired contact information 进入的地方

```text
GT target adapter
A wrist trajectory objective
B whole-hand geometric trajectory objective
safety/integrity checker 的 task-level rule
Oracle exposed fields
```

尤其注意：

```text
MAKE-before-BREAK
last-contact veto
anchor preservation
planned contact occupancy
```

如果被作为 task-level runtime guarantee，就会人为帮助 controller 保持 contact，从而污染 evaluation。

新 I04 必须区分：

```text
hardware safety:
    joint / collision / actuator / dangerous force
    -> 可以 veto

task success:
    continuous fingertip contact and coverage
    -> 只测量，不由 guard 保证
```

---

# 7. 旧 M06–M12 能力在新 I04 中的迁移映射

> 本节不是新的模块列表，只是 migration map。

| 旧 Module | 新 I04 是否继续作为独立模块 | 能力迁移到哪里 |
| --- | --- | --- |
| M06 | `NO` | Generic Trajectory Executor + Fresh-State Feedback |
| M07 | `NO` | 不迁移；legacy Explicit only |
| M08 | `NO` | 可选能力并入 GT Oracle geometric feasibility |
| M09 | `NO` | 重构为 Shared Geometric Trajectory Planner |
| M10 | `NO` | 重构为 Hardware Safety / Trajectory Integrity Checker |
| M11 | `NO` | 不迁移；legacy Explicit only |
| M12 | `NO` | continuation 能力并入 GT Oracle |

因此新代码不应该再出现：

```text
new-I04 -> M07 -> M08 -> M09 -> M11/M12 -> M10 -> M06
```

而应该是：

```text
GT Oracle
 -> Shared Geometric Trajectory Planner
 -> Finger Realization
 -> Safety / Integrity
 -> Generic Executor
 -> Real Observation / Coverage
```

---

# 8. 旧 M06：能力迁移到 Generic Trajectory Executor

旧 M06 保留 transaction、short prefix、fresh barrier 的执行工程价值，但不再作为 M06 模块进入新架构。

新接口概念上是：

```text
Planned trajectory
 -> split committed prefix
 -> Safety / Integrity check
 -> Generic Executor execute
 -> fresh measured state
 -> continue / replan
```

保留能力：

```text
short committed prefix
participant completion
stale-plan rejection
timeout / SAFE_HOLD for execution failures
fresh measured q / wrist / force / contact feedback
command provenance
```

删除/弱化：

```text
ContactModeGraph edge identity
MAKE/BREAK transaction authority
M11 prediction suffix dependency
task-level last-contact preservation
```

在 I04-A 中：

```text
Executor primarily executes wrist trajectory prefixes;
MCC or DPRef finger control runs continuously in parallel.
```

在 I04-B 中：

```text
Executor executes whole-hand geometric trajectory prefixes.
```

---

# 9. 旧 M07：直接退出 active path

原 M07 枚举：

```text
15 nonempty contact modes
WRIST / SLIDE / REPOSITION / MAKE / BREAK
mode-transition legality
```

这些都不属于新 I04。

现有 M07 只保留用于 legacy Explicit baseline / historical reproduction。

---

# 10. 旧 M08：可选能力并入 Oracle geometric feasibility

原 M08 是 primitive candidate cheap screen。

新 I04 不再有大量 SLIDE/MAKE/BREAK candidates，因此 runtime M08 没必要。

如果复用其思想，只允许作为 Oracle 内部 geometric feasibility：

### A

```text
X_H^* wrist pose reachable?
arm/palm collision free?
pose outside object?
rough SE(3) trajectory corridor feasible?
```

### B

```text
(X_H^*, x_1:4^*) admits terminal IK?
joint margin valid?
non-tip collision valid?
```

---

# 11. 旧 M09：重构为 Shared Geometric Trajectory Planner

这是新 I04 中最重要的能力迁移。

不再表示：

```text
ContinuousOptimize(SLIDE / MAKE / BREAK / WRIST_ADJUST)
```

正式新组件名建议直接使用：

```text
SharedGeometricTrajectoryPlanner
```

提供两种 task specification。

## 11.1 WristPoseTrajectory mode

输入：

```text
current full robot state
X_H,current
X_H^* ∈ SE(3)
```

输出：

```text
tau_H = {X_H(t)}
```

要求：

```text
position + orientation interpolation
joint feasible
velocity/acceleration feasible
arm/palm non-tip collision safe
smooth / time-parameterized
no fingertip target input
no contact-mode input
no finger optimization for contact preservation
```

A-MCC 和 A-DPRef 必须调用**同一实现、同一参数、同一 trajectory**。

## 11.2 WholeHandGeometricTrajectory mode

输入：

```text
current full robot state
X_H^*
x_1^*, x_2^*, x_3^*, x_4^*
```

输出：

```text
tau_Q = {q_arm(t), q_finger(t)}
```

要求：

```text
wrist pose objective 与 A 使用同一实现
terminal fingertip geometric target error within tolerance
joint/velocity/acceleration limits
non-tip collision avoidance
smoothness / time parameterization
no contact-mode search
no MAKE/BREAK optimization
continuous-contact not a hard planning constraint
```

---

# 12. 旧 M10：重构为 Hardware Safety / Trajectory Integrity Checker

只检查“这条 nominal prefix 是否安全、完整、可执行”，不保证 contact task success。

可以检查：

```text
joint limits
velocity / acceleration limits
self collision
arm/palm/non-tip object collision
trajectory integrity / digest
stale state/model provenance
hard actuator / force emergency constraints
```

不应检查或保证：

```text
expected contact mode
MAKE confirmation
BREAK legality
last-contact preservation
anchor preservation
terminal contact successor
```

核心原则：

```text
safe to execute != guaranteed to preserve contact
```

---

# 13. 旧 M11：直接退出 active path

原 M11 beam-search：

```text
SLIDE / MAKE / BREAK / REPOSITION / WRIST_ADJUST
```

新 I04 没有这种 discrete contact-mode search。

现有实现仅作为 legacy Explicit evidence 保留。

---

# 14. 旧 M12：continuation 能力并入 GT Oracle

“不要给一个当前可达、未来完全走死的 target”依然重要。

但它属于 privileged GT Coverage Oracle：

```text
full GT coverage ledger
 -> candidate Y_k^GT
 -> reachability / collision / continuation check
 -> choose next target
```

Oracle 可以内部使用 privileged future geometry 或 witness configuration，
但不能把下面信息暴露给 controller：

```text
future finger assignment
contact mode
handover sequence
MAKE/BREAK witness
```

---

# 15. 新 I04 完整执行流程

## 15.1 I04-A-MCC

```text
Full Object GT + real coverage ledger
        ↓
Privileged Coverage Oracle
        ↓
next outside-object wrist pose X_H^*
        ↓
Shared Geometric Trajectory Planner
    [WristPoseTrajectory mode]
        ↓
nominal tau_H
        ↓
Hardware Safety / Trajectory Integrity
        ↓
Generic Executor executes wrist trajectory prefix
        +
basic Passive Fingertip MCC continuously runs
        ↓
real physics/contact/force
        ↓
fresh measured state + real surface coverage update
        ↓
next prefix / next Oracle target
```

## 15.2 I04-A-DPRef

```text
Full Object GT + real coverage ledger
        ↓
SAME Privileged Coverage Oracle
        ↓
SAME X_H^*
        ↓
SAME Shared Geometric Trajectory Planner
        ↓
SAME tau_H
        ↓
Hardware Safety / Trajectory Integrity
        ↓
Generic Executor executes wrist prefix
        +
DPRef receives future tau_H
        +
DPRef generates nominal finger references / roles
        +
Fingertip MCC provides local compliance
        ↓
real physics/contact/force
        ↓
fresh measured state + real coverage update
```

## 15.3 I04-B-MCC

```text
Full Object GT + real coverage ledger
        ↓
Privileged Coverage Oracle
        ↓
next full-hand geometric target
    (X_H^*, x_1^*, x_2^*, x_3^*, x_4^*)
        ↓
Shared Geometric Trajectory Planner
    [WholeHandGeometricTrajectory mode]
        ↓
nominal whole-hand trajectory
        ↓
Hardware Safety / Trajectory Integrity
        ↓
Generic Executor executes whole-hand trajectory prefix
        +
basic MCC local compliance
        ↓
real physics/contact/force
        ↓
fresh measured state + real surface coverage update
```

I04-B 不使用 DPRef，也不使用任何 contact-mode fallback。

---

# 16. Target completion 与 Coverage completion

## 16.1 I04-A target completion

A 的目标是 wrist pose，因此单目标完成检查 SE(3) tracking：

```text
||p_H,real - p_H^*|| <= epsilon_p
and
angle(R_H,real, R_H^*) <= epsilon_R
```

不要求某根指定 finger 到达某个 target。

## 16.2 I04-B target completion

```text
||p_H,real - p_H^*|| <= epsilon_p
angle(R_H,real, R_H^*) <= epsilon_R
||x_i,real - x_i^*|| <= epsilon_f
```

这里仍然不把 `c_i_real` 作为 desired target state。

## 16.3 Whole-object coverage

真正任务完成由真实 fingertip contact 对 GT surface 的覆盖决定：

```text
Coverage(t) = fraction of required GT surface covered by REAL fingertip contacts
```

只有真实 physics contact 可以更新 coverage ledger。

不能计入：

```text
planned fingertip path
geometric target positions alone
DPRef prediction
Oracle witness
```

需要重新冻结：

```text
required surface set / area weighting
contact-to-mesh association
coverage radius
completion threshold
time / wrist-path / compute budget
```

---

# 17. 核心指标

共同指标：

```text
final real surface coverage
coverage AUC
coverage per unit wrist SE(3) path / time
time/path to target coverage
hand-level contact continuity
maximum all-finger zero-contact gap
zero-contact interval distribution
simultaneous contact count
fingertip force statistics
joint/workspace margin
non-tip collision / safety intervention
wrist position tracking error
wrist orientation tracking error
trajectory-planning latency
controller / DP inference latency
```

I04-A 额外：

```text
coverage under identical tau_H budget
continuous supported traversal
contact regain count/time
finger workspace utilization
DPRef transition diagnostics
```

I04-B 额外：

```text
per-finger geometric tracking error
whole-hand trajectory planning success rate
contact continuity despite no desired contact-state command
```

---

# 18. 三个核心实验 Cell

| Cell | Wrist target | Finger target | Trajectory planning | Finger realization |
| --- | --- | --- | --- | --- |
| `I04-A-MCC` | GT SE(3) pose | none | shared wrist SE(3) planner | basic passive MCC |
| `I04-A-DPREF` | same GT SE(3) pose | none | **same wrist trajectory** | DPRef + MCC |
| `I04-B-MCC` | GT SE(3) pose | GT per-finger positions | same planning framework + finger objectives | traditional trajectory + MCC |

理想关系：

```text
I04-A-MCC < I04-A-DPREF ~= I04-B-MCC
```

如果观察到这个趋势，可以支持：

1. 给出 high-dimensional geometric targets 时，传统 robot control + MCC 有能力完成较高 coverage；
2. 只给 global wrist pose trajectory 时，纯 passive compliance 不足以长期解决 finger workspace / regain / handover；
3. DPRef 可以在不显式 high-dimensional finger planning 的情况下恢复大部分 full-hand geometric Oracle 能力。

---

# 19. I01–I03 在新架构下是否还必要

旧 I01–I03 都围绕 Explicit contact-mode stack 构造：

```text
I01: fixed contact vs explicit variable contact / handover
I02: explicit finger REPOSITION long vs short prefix
I03: M11 Beam vs Beam + M12 ShadowSucc
```

它们的历史结果仍然有效，但不再是新 I04 prerequisite。

因此：

```text
I01 G2=GO       != new I04 prerequisite
I02=NOT_MET     != new I04 blocker
I03=MET         != new I04 prerequisite
G3=NO_GO        != forbid new I04
```

旧 G2/G3 应在后续 `MASTER_PLAN.md` 中标记为：

```text
LEGACY_EXPLICIT_GATES / EVIDENCE_ONLY_FOR_NEW_I04
```

不能改写历史实验结果，但不能继续阻塞新 I04。

---

# 20. I01：退役为历史实验，只抽取 Passive MCC moving-wrist smoke

旧 I01 的主要科学问题是：

```text
fixed |A|=4
vs
explicit variable contact + planned 4->3->4 handover
```

新 I04 不显式规划 contact mode，因此这个比较不再必要。

保留一个很小的 regression：

```text
R-I01: PASSIVE_MCC_MOVING_WRIST_SMOKE

input:
    short prescribed outside-object SE(3) wrist trajectory
    initial real fingertip contacts

controller:
    basic passive Fingertip MCC

check:
    finite/stable commands
    no numerical failure
    no joint/actuator/hard-force safety violation
    wrist position/orientation tracking works
    contact/force can be measured and logged
```

不要求：

```text
4->3->4 handover
variable-contact advantage
minimum traversal distance
99% contact continuity as pass condition
MAKE-before-BREAK
contact-mode certificate
```

contact continuity 是后续 I04 outcome，而不是该 smoke test 的硬 Gate。

旧文件与结果继续保留为：

```text
LEGACY_EXPLICIT_PHYSICS_EVIDENCE
```

---

# 21. I02：独立实验退休，short-prefix/fresh-state 并入 Generic Executor regression

旧 I02 比较：

```text
BREAK(3) -> REPOSITION(3) 12 mm -> MAKE(3)
LONG vs 3xSHORT
```

这是 Explicit finger transaction fixture。

新 I04-A 没有 explicit finger REPOSITION；新 I04-B 使用 generic whole-hand trajectory。

因此 I02 不再作为正式实验/Gate。

保留的只有执行思想：

```text
R-EXECUTOR-PREFIX:
    planned wrist/whole-hand trajectory
    -> short committed prefix
    -> execute
    -> fresh measured state
    -> continue/replan
```

只验证：

```text
no stale-state execution
correct prefix/fresh-state semantics
command provenance
trajectory continuity
```

不再要求证明 SHORT statistically beats LONG。

旧 `I02=EVALUATED / NOT_MET` 保持原样，但不阻塞新 I04。

---

# 22. I03：runtime experiment 退休，continuation 迁移到 Oracle

旧 I03 测：

```text
M11 Beam
vs
M11 Beam + M12 ShadowSucc
```

它解决的是 Explicit contact-mode planner 的 terminal dead end。

新 I04 不运行这个 discrete search，所以 I03 runtime experiment 不再必要。

保留的思想迁移为：

```text
R-ORACLE-CONTINUATION:
    generated GT target sequence has privileged geometric continuation
```

Oracle 可以使用 full GT、offline route、future wrist/finger geometric witness 来判断 continuation，
但这些 privileged witness 不得暴露给 controller。

旧 `I03=MET` 继续作为 Historical Explicit Planner evidence。

---

# 23. 新 I04 真正需要的前置验证链

```text
R0  Robot / Sensor / Basic Controller
    existing low-level validity

R1  Passive MCC Moving-Wrist SE(3) Smoke
    prescribed translation + rotation
    + basic Fingertip MCC

R2  GT Geometric Target Generator
    A: outside-object wrist SE(3) target reachable/collision-valid
    B: wrist SE(3) + fingertip geometric targets admit terminal IK

R3  Shared Geometric Trajectory Planning
    A: track X_H^*
    B: track X_H^* + x_1:4^*
    no desired contact-mode constraints

R4  Generic Execution/Safety
    Hardware Safety / Trajectory Integrity
    + Generic Executor prefix/fresh-state feedback

R5  GT Oracle Continuation
    target sequence is coverage-relevant and geometrically continuable
```

这些 regression 只验证系统有效性，不预先要求高 coverage 或高 contact continuity。

通过后直接运行：

```text
I04-A-MCC
I04-A-DPREF
I04-B-MCC
```

---

# 24. 最终迁移总表

| 旧 ID | 原作用 | 新 I04 是否需要旧模块本身 | 新位置 |
| --- | --- | --- | --- |
| I01 | Fixed vs explicit variable contact | `NO` | historical evidence + passive MCC SE(3) wrist smoke |
| I02 | Explicit REPOSITION long vs short | `NO` | prefix/fresh-state -> Generic Executor regression |
| I03 | ShadowSucc avoids Explicit dead end | `NO` | continuation -> GT Oracle regression |
| M06 | Transaction/prefix/barrier executor | `NO` | capability -> Generic Trajectory Executor |
| M07 | ContactModeGraph | `NO` | legacy Explicit only |
| M08 | Primitive CheapCert | `NO` | optional capability -> Oracle geometric feasibility |
| M09 | Primitive ContinuousOptimize | `NO` | capability -> Shared Geometric Trajectory Planner |
| M10 | Contact-aware exact audit | `NO` | capability -> Hardware Safety / Trajectory Integrity |
| M11 | Contact-mode beam search | `NO` | legacy Explicit only |
| M12 | Shadow contact viability | `NO` | capability -> Oracle continuation helper |

新的依赖关系：

```text
Robot / Sensor / Basic Controller
        +
GT Coverage Oracle
        +
Shared Geometric Trajectory Planner
        +
Hardware Safety / Trajectory Integrity
        +
Generic Trajectory Executor
        +
DPRef only for I04-A-DPREF
        ↓
       I04
```

不再是：

```text
I01 -> I02 -> I03 -> G3 -> I04
```

---

# 25. 一句话冻结定义

```text
The new I04 no longer uses M06–M12 as its formal architecture.
Legacy Mx modules are either retired or their useful capabilities are migrated into functional layers.
The first engineering step is therefore architecture cleanup, not further tuning of the old Explicit planner.

I04 uses full object GT to generate ideal geometric traversal targets.
The wrist target is always a full outside-object SE(3) pose, including rotation.
Every target must first be converted into a smooth feasible trajectory by a shared geometric trajectory planner.

A exposes only the wrist pose and compares basic passive MCC vs DPRef+MCC under the same planned wrist trajectory.
B exposes the same wrist pose plus per-finger geometric target positions and uses traditional whole-hand trajectory planning + MCC.

Neither branch receives desired contact states or contact-mode instructions.
Continuous fingertip contact and real surface coverage remain measured physical outcomes.
```