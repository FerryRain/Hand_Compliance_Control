# 主任务记录：分模块验证与完整系统实验路线

> 本文件是仓库唯一的项目实施与实验主计划。开始任何模块前必须先检查本文件。

## 当前状态

| 项目 | 当前值 |
| --- | --- |
| 计划版本 | `v1.6` |
| 最近更新 | `2026-08-23` |
| 授权状态 | `AUTHORIZED_M0_M04_MCC_ONLY_E05_F_H` |
| 实施状态 | `M0_M04_IMPLEMENTED_E05_MCC_EVALUATED` |
| 当前活动模块 | `M04_R_MOUNT_GAP_CONFIRMED_AWAITING_FIX` |
| 当前 Gate | `G1_LOCKED_PENDING_MOUNT_FIX_AND_MCC_RETEST` |
| 固定环境 | `handcomp` |
| Python | `/home/ferry/data/Anaconda/envs/handcomp/bin/python` |

本轮已按用户最新授权完成 M0–M3 的 FR3+Leap Hand 扩展、M04 的 MCC 控制链，并只运行
`E05-F-MCC` 与 `E05-H-MCC`。Finger DP 明确延期：未实现、未训练、未运行，也不填写
任何 DP 指标。FR3+Leap 模型为 `7 arm + 16 finger = 23 DoF`；物体固定在 world，wrist
运动由 FR3 关节实际产生，不再用移动物体或 fixed palm 冒充。

两个 E05 单元均完成 3 个冻结 paired episodes，因此执行状态为 `EVALUATED`；二者均有
预冻结性能阈值未满足，所以性能结论为 `NOT_MET`，不是 `FAILED`。本次正式协议关闭
gravity 以隔离接触控制，结果不能外推为 gravity-on 或硬件性能。

已有 `E05-PHY-v3` 仅登记为 `E05-PRE-FMCC` 预验证证据：它验证 fixed-palm 下的真实
指腹、拇指接触和 `15 s / 480 mm` 复杂曲面链路，但不包含 FR3、Wrist MCC 或 Finger DP，
不占正式 E05 单元且不能解锁 Gate G1。其冻结协议、trace 和 `MET/NOT_MET` 性能记录
保持不变。

## 核心推进原则

每个模块必须先单独证明自己工作，才能进入下一层集成：

```text
A. Low-level Control Validation
                 |
                 v
B. Geometry-Oracle Planning Validation
                 |
                 v
C. GPIS Unknown-Object Exploration
                 |
                 v
D. Main Method vs. Explicit Baseline
```

任何上游模块未通过时，不得通过调整下游模块掩盖问题。Full system 失败时，必须能
沿模块证据反查到具体接口、控制器、规划器、证书或执行器。

## Main 与 Explicit Baseline 的隔离边界

后续实现必须把两条方法隔离：

- **Main method**：`GPIS/Oracle Wrist Planner + shared Wrist MCC + Finger DP`；
- **Explicit baseline**：`Contact-Mode Planner + shared Wrist MCC + Contact Force Coordinator
  + explicit fingertip trajectory + Fingertip MCC`；
- **共享部分**：状态契约、SurfaceModel 接口、RobotIO、runtime guards、对象、初始
  状态、wrist motion budget、Wrist MCC、desired resultant-wrench convention、日志和
  evaluator；
- **禁止共享的执行分支**：Fingertip MCC 不能作为 DP 后处理器进入 Main；显式
  fingertip/contact-mode planner 不能成为 Main runtime 的隐藏 fallback。

Baseline 的 Fingertip MCC 只解决高频法向力和小局部法向误差，不负责修复切向规划、
可达性、碰撞、接触拓扑或未来 viability。

## 状态标记

主表只使用以下状态：

- `NOT_STARTED`：没有实现，也没有实验；
- `PROTOCOL_FROZEN`：接口、阈值、对象、seed 和 evaluator 已冻结，但尚未执行；
- `IN_PROGRESS`：已明确授权且正在实现或测试；
- `EVALUATED`：评测已完整执行，方法表现由独立的 `MET/NOT_MET` 指标描述；
- `EVIDENCE_ONLY`：已执行的预验证/诊断证据，不属于当前正式协议，不能解锁 Gate；
- `DEFERRED`：用户明确要求本轮不实现、不评测，且不作为本轮完成条件；
- `BLOCKED`：证据表明当前依赖或模块阻塞；
- `PASSED`：达到预先冻结的通过标准，并已登记证据；
- `FAILED`：模块/前置验证未达到冻结的有效性或安全标准，禁止推进依赖模块；已完整执行
  的比较实验不会仅因某个方法性能差而标成 `FAILED`，而是保持 `EVALUATED` 并报告
  `MET/NOT_MET`。

不得把“代码能 import”“单次视频看起来正常”或“下游暂时能运行”记为 `PASSED`。

## 总任务看板

| 顺序 | ID | 模块/实验 | 依赖 | 状态 |
| ---: | --- | --- | --- | --- |
| 0a | P0 | 原 hand-only 统一测试环境、状态和日志 | 无 | `PASSED` |
| 0b | M0-FR3 | 23-DoF 全机器人状态/动作/wrench/log contract | P0 | `PASSED` |
| 1a | M01 | Oracle SurfaceModel 解析接口 | P0 | `PASSED` |
| 1b | M01-FR3 | live FR3 capsules 与 MuJoCo pad/object narrow phase | M0-FR3, M01 | `PASSED` |
| 2a | M02 | Fingertip MCC：Static/Sliding/Curved | P0, M01 | `PASSED` |
| 2b | M02-FR3 | moving-wrist 四指 MCC、signed coordinated error | M0-FR3, M02 | `PASSED` |
| 3a | M03 | hand-only Runtime Guards | P0 | `PASSED` |
| 3b | M03-FR3 | arm/finger 分组 guards 与全局/局部处置 | M0-FR3, M03 | `PASSED` |
| 4d | M04-DP | Finger DP control-only | M0-FR3, M03-FR3 | `DEFERRED` |
| 4r | M04-R | 23-DoF FR3+Leap MuJoCo plant | M0-FR3 | `BLOCKED`（结构测试通过；80 mm 不可见 mount 待修） |
| 4w | M04-W | palm pose IK、wrench estimator 与 6D Wrist MCC | M04-R, M03-FR3 | `PASSED` |
| 4c | M04-C | Resultant/internal Contact Force Coordinator | M01-FR3, M02-FR3, M04-W | `PASSED` |
| 4h | M04-H | Wrist MCC + coordinated Fingertip MCC integration | M04-C | `PASSED` |
| 5pre | E05-PRE-FMCC | Fixed-palm Fingertip-MCC 物理预验证 | M01–M03 | `EVIDENCE_ONLY` |
| 5a | E05-F-MCC | 规定式 FR3 wrist + 四指 Fingertip MCC | M04-R, M02-FR3 | `EVALUATED / NOT_MET` |
| 5b | E05-H-MCC | FR3 Wrist MCC + coordinator + Fingertip MCC | M04-H | `EVALUATED / NOT_MET` |
| 6 | M06 | Transactional Prefix Executor | G1, M02, M03 | `NOT_STARTED` |
| 7 | M07 | ContactModeGraph | P0 | `NOT_STARTED` |
| 8 | M08 | CheapCert | M01, M07 | `NOT_STARTED` |
| 9 | M09 | ContinuousOptimize | M01, M02, M08 | `NOT_STARTED` |
| 10 | M10 | ExactPrefixAudit | M01, M03, M06, M09 | `NOT_STARTED` |
| 11 | M11 | Lazy Beam Search | M07–M10 | `NOT_STARTED` |
| 12 | M12 | Shadow Terminal Viability | M08, M11 | `NOT_STARTED` |
| 13 | I01 | Oracle Continuous Traversal | M01–M12 | `NOT_STARTED` |
| 14 | I02 | Receding-horizon prefix 对比 | I01 | `NOT_STARTED` |
| 15 | I03 | Terminal viability 对比 | I01, M12 | `NOT_STARTED` |
| 16 | I04 | Oracle Explicit vs. Oracle Main | G3, M04-DP, M04-W, M04-C | `NOT_STARTED` |
| 17 | M13 | GPIS SurfaceModel | G3, M01 | `NOT_STARTED` |
| 18 | M14 | GPIS Active Exploration | M13 | `NOT_STARTED` |
| 19 | I05 | GPIS Main vs. GPIS Baseline | G4, M14 | `NOT_STARTED` |

## Phase 0：统一测试环境

### 目标

只冻结后续所有模块共享的环境、状态、动作、日志和 evaluator，不验证算法。

### 环境约束

- 固定使用 `handcomp`，不新建环境；
- 第一阶段物体固定，不做被推动物体的 SE(3) tracking；
- 对象按以下顺序增加复杂度：

```text
Plane -> Cylinder -> Sphere -> Box/Rounded Box -> Free-form Surface
```

- 在 Plane/Cylinder/Sphere 尚未通过前，不引入 ShapeNet 或复杂自由曲面。

### 统一状态

所有 planner/controller 从同一真实状态格式读取：

```text
z_t = (X_H, q, dq, f_tip, c_tip, E_fail, SurfaceModelVersion)
A_actual(t) = {i | c_tip[i] == CONTACT}
```

如需表面运动量，作为版本化可选字段 `v_surface` 加入，不能由不同模块自行解释。
`A_actual` 只来自真实测量；`A_expected`、预测 contact mode 和 optimizer state 均无权
覆盖它。

连续探索的硬条件是：

```text
for all t: |A_actual(t)| >= 1
```

不是要求四根手指始终全部接触。最后一个旧 anchor 只有在 replacement contact 已经
真实确认后才能 BREAK。

### 必须统一记录的日志

- monotonic timestamp、episode、step 和 seed；
- wrist pose/twist/wrench；
- finger q/dq、上一实际动作和当前命令；
- fingertip positions、forces、contact flags；
- `A_actual` 与预测 contact set；
- planned trajectory、committed prefix 和 prediction-only suffix；
- transaction id、executor state、micro-barrier state；
- blocked reason 与 blocked evidence；
- `SurfaceModelVersion`；
- planning/optimization/audit/execution latency；
- collision distance、joint margin、anchor margin、reach margin；
- safety override、contact event 和 certificate id。

执行状态 `RUNNING/DONE/BLOCKED/CANCELLED` 必须与 fingertip 状态
`CONTACT/FREE` 分开记录。

### Phase 0 通过条件

- schema、单位、坐标系、四元数顺序和采样率有版本号；
- Oracle、GPIS、DP、MCC、planner 和 evaluator 使用相同 contract；
- 一条短 mock episode 可以完整写出并重放日志；
- 所有阈值和 evaluator 在第一次正式实验前冻结。

### M0-FR3 扩展（已通过）

新增 `fr3-leap-modules.v1`，不破坏原 `hand-modules.v1`。它固定记录 `arm_q/dq[7]`、
`finger_q/dq[16]`、palm pose/twist、wrist wrench 的 frame/reference/acting-on/source、arm
external torque、四指 contact force/vector/point/normal/validity，以及 actuator/sensor/controller
状态。`A_actual` 仍只从 measured contact state 导出。JSON 与 JSONL round-trip、shape、单位、
NaN/validity 和 actual-contact authority 均有回归测试。

## Module 1：Oracle SurfaceModel

### 目标

先完全去掉未知几何误差，只验证 planning/control 所需的几何接口。

### 最小接口

```text
querySurface(x)
queryNormal(x)
queryClearance(link_or_swept_geometry)
sampleContactCandidates(finger_id)
queryUncertainty(x)
version
```

Oracle 使用 simulator/analytic ground truth：

```text
uncertainty(x) = 0
m_uncertainty = 1
version = fixed immutable id
```

### 独立测试

1. **Point-surface distance**：随机采样点，对比 analytic/simulator GT；
2. **Surface normal**：报告 `acos(n_pred dot n_GT)`；
3. **Link clearance**：随机 hand configuration，对比 simulator collision engine；
4. **MAKE candidates**：检查点在表面、法向正确、对应 fingertip 基本可达。

### 通过条件

误差阈值必须在运行前冻结。四类测试均通过后，才能把 Oracle 交给控制和规划模块。
这一阶段不做 exploration，也不将 analytic oracle 结果冒充未知物体能力。

展示层可额外使用统一放大的 Stanford Bunny/YCB mesh，但不得取代 Plane/Sphere/Cylinder/
Box/RoundedBox 的解析验收，也不得把展示 scale 当作 YCB 真实物理尺寸实验。

### M01-FR3 扩展（已通过）

`FullRobotGeometryAdapter` 从 live `MjData` 生成 world-frame FR3 link capsules，并把它们送入
同一 Oracle clearance contract；四个 fingertip belly pad 到 fixed object 的精确窄相距离
使用 MuJoCo `mj_geomDistance` 与 witness points。approximate capsule 只用于快速几何查询，
不冒充 exact trajectory certificate。

## Module 2：Fingertip MCC（Explicit Baseline 专用）

### 目标

给定正确 fingertip trajectory `p_plan_i(t)`，验证 analytical controller 能否跟踪
切向运动并稳定调节法向力。

### 控制形式

```text
M_i * dd(delta_d_i) + D_i * d(delta_d_i) + K_i * delta_d_i
    = f_des_i - f_i

p_cmd_i = p_plan_i + delta_d_i * n_i
```

MCC 只修改法向，不改变 planner 的切向意图。所有 offset、速度、加速度和 joint
command 必须限幅并记录。

### Module 2A：Static Contact

- 单指接触固定平面；
- 使用至少三个冻结的 `f_des`；
- 记录 steady-state error、overshoot、settling time、oscillation；
- 核心指标：`force RMSE` 与 `P(force > force_max)`。

### Module 2B：Tangential Sliding

- 使用已知切向轨迹 `p(t) = p0 + s(t) * tangent`；
- MCC 只调法向；
- 报告 tangential tracking error、force RMSE、contact loss 和 slip。

### Module 2C：Curved Surface

- 依次使用 cylinder 和 sphere；
- 法向 `n(t)` 持续变化；
- 验证 `f_n` 对 `f_des` 的跟踪、曲率切换稳定性和 contact continuity。

### 通过条件

2A、2B、2C 必须分别通过预冻结标准。单指或平面成功不能替代曲面验证。

### M02-FR3 扩展（已通过）

原 `FingertipMCC.step(...)` 保持兼容；新增 `step_force_error(...)` 接收 coordinator 已投影的
signed force error。四指 wrapper 使用 moving palm frame、逐指 direction、active-contact mask
和 contact transition reset，避免将固定 palm 假设带入 FR3。真实接触 pad 位于指尖指肚，
包括经过朝下初始化的拇指 pad。

## Module 3：Runtime Guards

### 目标

独立验证可观测条件下的 over-force、joint limit、known self-collision、no-progress 和
suspected non-tip blockage。不得假设存在 non-tip tactile sensor，也不得声称直接测得
未知物体与中间指节的碰撞点/法向。

### 必测案例

- **A — Free motion**：不得误报 `BLOCKED`；
- **B — Middle phalanx blockage**：`f_tip ~= 0`，但 command 有进展而 actual motion
  长时间接近零，应输出 `SUSPECTED_OBJECT_BLOCKAGE`；
- **C — Joint limit**：明确输出 `JOINT_LIMIT`；
- **D — Fingertip over-force**：在冻结 latency 内进入保护；
- **E — Known self-collision**：输出独立硬失败原因。

### 指标

blockage detection rate、false-positive rate、detection latency、over-force response
latency。`NO_PROGRESS` 与 `SUSPECTED_OBJECT_BLOCKAGE` 是局部证据，不得永久写成几何事实。

### M03-FR3 扩展（已通过）

全机器人 guard 独立累计 7-DoF arm stall 与四个 finger group stall，避免整体 norm 掩盖单指
阻塞；同时检查 stale/invalid sensor、tip over-force、wrist wrench、arm external torque、
known collision、arm/finger joint limit、controller limit 与 actuator saturation。处置明确区分
`FINGER_LOCAL` 和 `GLOBAL_SAFE_HOLD`，且仍不伪造未知 non-tip collision 的位置或法向。

## Module 4：FR3+Leap 整手 MCC

### Module 4DP：Finger DP（本轮延期）

用户明确要求本轮不做 DP。该分支保持 `DEFERRED`：没有 policy、checkpoint、训练、推理
或指标，也不把 MCC 的输出冒充 DP。未来只有获得新授权后才冻结 DP protocol。

### Module 4R：FR3+Leap dynamic plant

已实现可控的 MuJoCo 整机模型：

- 7 个 FR3 position actuators 与 16 个 Leap Hand position actuators；
- `(nq,nv,nu)=(23,23,23)`，物体固定在 world，`nmocap=0`；
- palm/flange 显式安装，四个 belly pad 绑定真实 fingertip body；
- 初始拇指关节使 thumb belly 朝向并接触曲面，不再缺失拇指接触；
- 正式 E05 的 wrist trajectory 由 FR3 关节执行。

### Module 4W：FR3 wrist control 与 Wrist MCC

已实现 palm pose 的 damped-least-squares IK、规定式 wrist tracking、通用 6D admittance
Wrist MCC 和基于 arm constraint torque/palm Jacobian 的 wrench estimator。wrench 日志明确
frame、moment reference、acting-on 和符号；本次 E05 仅选择 collective-normal translation
方向，切向二维探索仍由 nominal trajectory 主导。

若 `f_des` 表示 hand-on-object force，则 `w_obj_des=G f_des`、
`w_hand_des=-G f_des`。当 `G f_des=0` 时，internal squeeze 与 zero wrist resultant 并不
冲突；Wrist MCC 不再默认对所有非零 fingertip force 追零。

### Module 4C：Contact Force Coordinator

实现使用 normal-force basis `f=B lambda` 与 effective map `H_A=G_A B_A`：

```text
normal-force error -> resultant component -> Wrist MCC
                   -> internal component  -> Fingertip MCCs
```

weighted generalized inverse 构造 resultant/internal projectors；`A_actual` 只由带迟滞的
真实 contact measurement 决定，未确认 contact 不进入承载 wrench 的 map。实现记录 rank、
singular values、condition、reconstruction error 和 internal leakage，并在 contact-set
transition 时 blend/reset，避免旧 projector 保留执行影响。

### Module 4H：整手集成

- `E05-F-MCC`：关闭 Wrist MCC，FR3 跟踪规定轨迹，四指 MCC 接收完整 local force error；
- `E05-H-MCC`：Wrist MCC 接收 resultant error，active fingers 只接收 internal error；
- 未确认接触的手指只允许局部 approach/recovery，不作为 wrench anchor；
- M03-FR3 guards 在同一 loop 中监测 arm/finger stall、force、wrench、torque、joint、sensor
  和 actuator saturation。

实现、符号和审计细节见 `WHOLE_HAND_COMPLIANCE_DESIGN.md`；复现入口见 `Module/README.md`。

## Experiment 5：MCC-only E05 双层评测

本轮正式矩阵只包含：

| Wrist mode | Fingertip controller | 状态 |
| --- | --- | --- |
| 规定式 FR3 wrist tracking | `E05-F-MCC` | `EVALUATED / NOT_MET` |
| FR3 6D Wrist MCC（normal projector） | `E05-H-MCC` | `EVALUATED / NOT_MET` |

DP 两列已从本轮 protocol 和 evaluator 中移除。正式协议为
`E05_MCC_FR3_PROTOCOL.md`，SHA-256 为
`483bd7ccf98ad5afec9bfb2d2912d2b9882d2a4045a194582b879b8adc51664a`。

### 冻结场景

每个单元使用相同的 3 个 paired episodes（nominal、low-friction、noisy-pose），每段
`15 s`：长程 `180 mm` Y traversal、二维 S 型 X 变化、连续多尺度曲率，以及 `t=9 s`
时沿 collective normal 突然离面 `4 mm` 的恢复扰动。对象、初始 arm/hand pose、nominal
wrist trajectory、噪声、seed、runtime guards、control rate 和 evaluator 均共享。

### 正式结果

| aggregate 指标 | E05-F-MCC | E05-H-MCC |
| --- | ---: | ---: |
| 完成 episodes | 3/3 | 3/3 |
| mean contact continuity | 99.990% | 99.990% |
| mean contact count | 3.750 | 3.710 |
| mean force RMSE | 0.773 N | 1.017 N |
| worst peak force | 11.08 N | 13.19 N |
| mean Y traversal | 173.9 mm | 175.3 mm |
| controller P95 mean | 1.249 ms | 1.263 ms |
| wrist Fz RMSE | — | 1.984 N |
| internal leakage P95 | — | 0.0135 N |

F 未满足 deadline-miss、force-violation probability 和 `8 N` peak 阈值；H 未满足 force
RMSE、force settling（noisy-pose worst `5.998 s`）和 `8 N` peak 阈值。完整运行有效，
所以是 `EVALUATED / NOT_MET`，而不是 `FAILED`。H 在本协议中并未改善 force RMSE；只能
报告其 traversal 略高且 internal leakage 较小，不能据此声称整手 MCC 已优于 F。

### E05-PRE-FMCC：历史预验证

`E05-PHY-v3` 只作为 fixed-palm 的历史工程证据；它不含 FR3/Wrist MCC/DP，也不覆盖本次
正式结果。旧 trace 保留可追溯，但不是当前默认控制测试。

### Gate G1 当前判断

MCC 子门已完成有效评测，但性能为 `NOT_MET`；DP 子门按用户要求延期。因此完整 G1 仍是
No-Go，不自动进入 planner/executor 集成。下一步若继续控制器，应冻结新的 v2 protocol
后针对 peak force 与 recovery settling 修正，不能回写本次冻结结果。

## Module 6：Transactional Prefix Executor

### 目标

在没有 planner 的情况下，人工提交 transaction，验证：

```text
commit short prefix -> execute -> micro barrier -> real snapshot
```

只有当前 transaction 的 committed prefix 有执行权限；prediction suffix 只能 warm start。

### Prefix 类型

- `WRIST_ADJUST`：wrist motion，anchor fingers 使用 Baseline MCC，free fingers hold；
- `FINGER_RECONFIGURE`：wrist hold，selected fingers 异步执行，anchors 使用 Baseline MCC。

### 独立测试

1. **Asynchronous completion**：先完成的 finger 进入 hold，必须等待 micro-barrier；
2. **Blocked finger**：被挡 finger 输出 `BLOCKED`，独立安全手指按规则结束；
3. **Transaction authority**：新 transaction 提交后，旧 transaction 永不恢复权限；
4. **SurfaceModelVersion**：transaction 的 model version 过期时必须拒绝；
5. **Timeout**：planner/executor timeout 使用 `SAFE_HOLD`，不恢复旧 suffix。

下一次 planner root 必须来自 barrier 后的新真实 snapshot。

## Module 7：ContactModeGraph

### 状态与 primitive

四指共有 `2^4 - 1 = 15` 个非空 contact sets。动作集合：

```text
WRIST_ADJUST
SLIDE(i)
REPOSITION(i)
MAKE(i)
BREAK(i)
```

### 纯单元测试

穷举 15 个 mode 与全部 primitive，要求 100% deterministic：

- 不产生 empty contact mode；
- MAKE 只作用于 free finger；
- BREAK 只作用于 contact finger；
- 一个 prefix 最多一次 topology change；
- 同一 prefix 不同时 MAKE 与 BREAK；
- WRIST_ADJUST 与 finger reconfiguration 不同时发生；
- PredictLegal 与 CommitLegal 分离；
- `CommitLegal(BREAK(i))` 依赖真实 replacement contact confirmation。

Handover 必须拆成独立 transaction，例如：

```text
REPOSITION(3) -> MAKE(3) -> BREAK(1)
```

## Module 8：CheapCert

### 目标

快速筛除明显不可行的 `(contact mode, action)`，但不签发执行证书。

### 检查与输出

- mode legality、failed-edge evidence；
- approximate IK/reachability；
- joint/anchor margins；
- coarse collision；
- trust region 与 uncertainty。

输出 `SURVIVE` 或 `REJECT`，以及：

```text
m_anchor, m_joint, m_collision, m_reach, m_uncertainty
```

### 定量验证

随机生成 1,000–10,000 个 candidate edges，以昂贵 exact solver/audit 为参考，报告
TP/FP/FN/TN。首要目标是低 false negative；可以放过垃圾，但不能大量误杀可行解。

## Module 9：ContinuousOptimize

### 第一版边界

暂不使用 information gain，只构造当前 mode edge 的连续轨迹：

- target tracking；
- collision/joint constraints；
- anchor preservation；
- smoothness；
- reachability；
- first-edge commit trust region。

### 分 primitive 测试

分别验证 `SLIDE`、`REPOSITION`、`MAKE`、`BREAK` 和 `WRIST_ADJUST`。对于未完成但仍
安全的 MAKE approach，状态为 `MAKE_PROGRESS(i)`；只有真实 contact 才能改变 topology。

### 指标

optimizer success rate、solve time、final target error、minimum clearance、minimum
joint margin、anchor margin 和 trust-region use。

## Module 10：ExactPrefixAudit

### 权限

只有本模块能签发 `ExecutionCertificate`。CheapCert、optimizer、beam node、suffix 和
ShadowSucc 永远没有执行权限。

### Audit 范围

检查整个 swept committed prefix：joint limit、self-collision、link clearance、phase
exclusivity、anchor assumptions、trust bounds、SurfaceModelVersion 与当前 CommitLegal。

### Adversarial tests

1. 起终点安全但中间碰撞：REJECT；
2. 中途 joint limit：REJECT；
3. replacement contact 未真实确认就 BREAK：REJECT；
4. SurfaceModelVersion 过期：REJECT；
5. 超过 trust region：REJECT；
6. prediction suffix 试图获得权限：REJECT。

## Module 11：Lazy Beam Search

### 目标

搜索有当前 progress 且未来仍有路的 contact-mode sequence，而不是单步贪心。

### 固定搜索顺序

1. shifted suffix warm start；
2. enumerate legal edges；
3. CheapCert；
4. optimize survivors；
5. 按 per-mode quota 保留 diverse top-B；
6. 重复至 horizon `H`。

初期 score 不含 information gain：

```text
S = w_progress * J_progress
  + w_contact  * J_contact
  - w_motion   * J_motion
  - w_risk     * J_risk
  - w_switch   * J_switch
```

### 验证

用 `H=2/3` 同时运行 exhaustive search 与 beam search，比较 score、最优 mode sequence
保留率、计算量和 diversity，以冻结 beam width 与 per-mode quota。

## Module 12：Shadow Terminal Viability

### 目标

拒绝当前合法但会把系统带入无安全 continuation 的 prefix。

### 测试

- **Viable**：`A={1}` 且 finger 2 存在 cheap-feasible MAKE，输出 `VIABLE`；
- **Dead end**：`A={1}` 且其他手指均 joint-limited/colliding/unreachable，输出
  `NONVIABLE`。

terminal state 至少需要一个非平凡 cheap successor；singleton contact mode 必须存在
至少一个 inactive finger 的 cheap-feasible MAKE。Shadow successor 只能预测，不能执行。

## Integration 1：Oracle Continuous Traversal

第一次完整集成 Explicit Baseline，但不做 active exploration。任务仅为沿 GT surface
尽可能连续移动。

### I01-A：Fixed Contact

强制 `|A|=4`，记录 `L_fixed`。尝试可重复地复现约 `47.97 mm` 的历史 planning
failure，但不得把它描述成物理 reach limit。

### I01-B：Variable Contact Mode

改为 `|A|>=1`，其余条件不变，记录 `L_variable`。核心假设是：

```text
fixed-contact infeasible != variable-contact infeasible
```

只有发现真实可行且具备 continuation 的 mode transition，才算支持该假设。预期证据
包括 `4->3->4`、`3->2->3` 等 handover，以及 `L_variable > L_fixed`。

### Gate G2

Variable mode 必须用受控实验真实突破 fixed-contact 局部可行域；轨迹切短本身不能
被解释为扩大 feasible set。

## Integration 2：Receding-Horizon Prefix

固定 variable contact mode，对比：

- long/full execution；
- short committed prefix + snapshot + replan。

报告 execution failure、blocked、collision、prediction error、completion distance 和
replanning latency。结论只能是 prefix execution 是否提高 closed-loop robustness，
不能声称其自动改变 feasible set。

## Integration 3：Terminal Viability

比较 Beam 与 Beam + ShadowSucc，报告 dead-end count、handover availability、minimum
joint margin、maximum traversal 和 recovery count。

### Gate G3 / Geometry-Oracle Baseline

只有 Oracle 下能够长期连续 traversal、维持 `|A_actual|>=1`、没有未解释的碰撞/关节/
force violation，并能报告完整 planning/certificate/executor 证据，才允许进入 GPIS。

Gate G3 报告至少包含：

- Safety：contact continuity、collision、joint/force violations；
- Coordination：MAKE/BREAK/handover success、average contacts；
- Planning：latency、nodes、optimizer/certificate success、blocked ratio；
- Viability：dead ends、successor-finger count、terminal margins。

## Module 13：GPIS SurfaceModel

### 替换原则

只替换：

```text
OracleSurfaceModel -> GPISSurfaceModel
```

planner、controller、executor、certificate 和 evaluator 接口保持不变。

GPIS 提供 mean surface、normal、uncertainty 和 version：

```text
surface: mu(x) = 0
normal:  grad(mu) / ||grad(mu)||
uncertainty: sigma^2(x)
```

collision safety 必须使用 uncertainty-aware margin，例如：

```text
d_safe = d0 + beta * sigma(x)
```

### 验证

先沿给定方向 traversal，不做 active exploration。Oracle 与 GPIS 共享 motion 和
evaluator，比较 collision、contact loss、normal error、candidate reachability 和
traversal distance。

### Gate G4

把 GT 替换为 GPIS 后，系统仍需在冻结安全标准下闭环运行。未通过 G4 时不得开始完整
active exploration 对比。

## Module 14：GPIS Active Exploration

第一版 information objective：

```text
J_info = sum_i sigma^2(x_i)
```

之后才考虑 `logdet(Sigma_X)`。比较 random/fixed exploration 与 uncertainty-guided
planner，报告 uncertainty reduction/time、uncertainty reduction/distance、coverage、
F-score、Chamfer 和 active-contact novelty。

## Integration 4：Oracle Main vs. Oracle Explicit Baseline

两边都使用 GT SurfaceModel，以隔离 planning architecture：

- Explicit：Contact-Mode Planner + shared Wrist MCC + Contact Force Coordinator +
  Fingertip MCC；
- Main：Wrist Planner + shared Wrist MCC + Finger DP。

共享 receding horizon、committed prefix、snapshot、runtime guards、geometry、objects、
sensor definitions 和 evaluator。比较 exploration progress、contact continuity、average
contacts、planning/control compute、handover、joint travel、blocked 和 success rate。

核心问题：显式 finger/contact-mode planning 是否必要？若 Main 接触和 progress 接近或
更好，而在线 latency 明显更低，才支持核心 thesis。

## Integration 5：GPIS Main vs. GPIS Explicit Baseline

最终完整比较：

```text
Explicit = GPIS + Contact-Mode Planner + shared Wrist MCC
         + Contact Force Coordinator + Fingertip MCC
Main     = GPIS + Wrist Planner + shared Wrist MCC + Finger DP
```

报告：

- Reconstruction：F-score、Chamfer、coverage；
- Exploration：uncertainty reduction/sec、coverage/sec、path length、explored area；
- Contact：continuous-contact success、average contacts、contact loss、handover；
- Planning efficiency：mean/P95 latency、solver failure、nodes/candidates、real-time factor。

## 四个最终论文问题

| 问题 | 对应实验 |
| --- | --- |
| Q1a — 规定式 wrist 下的 Fingertip MCC 与协调整手 MCC 分别表现如何？ | 当前 E05-F-MCC + E05-H-MCC |
| Q1b — Finger DP 与 Fingertip MCC 在相同 wrist 条件下分别表现如何？ | `DEFERRED`，未来需单独冻结 DP protocol |
| Q2 — Variable contact modes 是否扩大 feasible region？ | Oracle Fixed vs. Variable Contact |
| Q3 — 是否需要显式高维 finger/contact planning？ | Oracle Explicit vs. Main |
| Q4 — 未知物体上能否高效闭环探索？ | GPIS Explicit vs. Main |

## 四个 Go/No-Go Gates

| Gate | Go 条件 | No-Go 后动作 |
| --- | --- | --- |
| G1 | 当前 MCC 两单元达到 safety/readiness；若恢复 DP 比较，再完成其独立子门 | 当前 MCC 为 `EVALUATED / NOT_MET`，先修复 peak/settling，禁止 planner 集成 |
| G2 | Variable mode 真实突破 fixed-contact 局部可行域 | 检查 mode legality/optimizer/audit，不引入 GPIS |
| G3 | Oracle 下长期连续 traversal 且无 dead end | 修复 planner/executor/certificate，禁止 GPIS 集成 |
| G4 | 换成 GPIS 后仍能安全闭环 | 修复 SurfaceModel/uncertainty margin，禁止 full comparison |

只有通过 G3，才投入主要精力进行 GPIS integration。

## 获得授权后的严格实施顺序

```text
0.  [DONE] M0 full-robot environment/state/action/wrench/log contract
1.  [DONE] M01 Oracle + live FR3 geometry
2.  [DONE] M02 Fingertip MCC + moving-wrist four-finger wrapper
3.  [DONE] M03 hand-only + FR3 grouped Runtime Guards
4.  [DONE] M04-R FR3–Leap plant
5.  [DONE] M04-W 6D Wrist MCC + wrench estimator
6.  [DONE] M04-C/H resultant/internal coordinator and integration
7.  [EVALUATED / NOT_MET] E05-F-MCC + E05-H-MCC
8.  [DEFERRED] Finger DP and DP E05 cells; only after explicit authorization
9.  [G1 NO-GO] Tune MCC under a newly frozen v2 protocol, or obtain a gate revision
10. Transactional Prefix Executor
11. ContactModeGraph
12. CheapCert
13. ContinuousOptimize
14. ExactPrefixAudit
15. LazyBeamSearch
16. ShadowSucc
17. Oracle Continuous Traversal
18. Oracle Explicit vs. Main
19. GPIS SurfaceModel
20. GPIS Active Exploration
21. Full Main vs. Baseline
```

不得因为某个后续模块更容易展示而跳过顺序或 Gate。

## 每个模块开始前的 Protocol Freeze

状态从 `NOT_STARTED` 改为 `IN_PROGRESS` 前，必须先登记：

```text
Module ID:
Authorization:
Owner:
Exact scope:
Inputs/outputs and schema version:
Objects and initial states:
Seeds and episode count:
Metrics:
Numerical pass/fail thresholds:
Evaluator version:
Allowed privileged information:
Expected artifact paths:
Timeout/resource budget:
```

没有冻结 numerical threshold 的实验不能事后根据结果选择通过标准。

## 每个模块完成后的 Evidence Record

只有下面内容完整时才能标记 `PASSED`：

```text
Module ID:
Status:
Commit/worktree provenance:
Environment and dependency versions:
Commands:
Input hashes:
Output artifacts:
Metrics and confidence intervals:
Failure cases:
Pass/fail decision:
Downstream modules unlocked:
```

视频和单个成功 episode 只能作为辅助证据，不能替代统计指标与失败案例。

## 主计划更新规则

1. 每次开始工作前先读取本文件的当前状态、依赖和 Gate；
2. 未获得用户明确授权时，只能修改计划文字，不能实施模块；
3. 一次最多有一个 `IN_PROGRESS` 模块；
4. 状态变化必须同时写入证据位置，不得只口头宣告；
5. 上游 `FAILED/BLOCKED` 时，下游保持 `NOT_STARTED`；
6. 任何架构变更先更新本计划并说明对公平比较和论文问题的影响；
7. 历史 suffix、预测 contact、shadow successor 和过期 SurfaceModel 均无执行权限。
