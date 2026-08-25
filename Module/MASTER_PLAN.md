# 主任务记录：分模块验证与完整系统实验路线

> 本文件是仓库唯一的项目实施与实验主计划。开始任何模块前必须先检查本文件。

## 当前状态

| 项目 | 当前值 |
| --- | --- |
| 计划版本 | `v3.4` |
| 最近更新 | `2026-08-25` |
| 授权状态 | `AUTHORIZED_I04_CORE_PROTOCOL_ONLY` |
| 实施状态 | `I04_NOT_IMPLEMENTED` |
| 当前活动模块 | `I04_CORE_PROTOCOL_FROZEN_NUMERICS_TBD` |
| 当前 Gate | `G1a_ARCHIVED_PRE_RETUNE / G2_GO_BUNNY / G3_NO_GO_I02_NOT_MET` |
| 固定环境 | `handcomp` |
| Python | `/home/ferry/data/Anaconda/envs/handcomp/bin/python` |

本轮已完成 M0–M3 的 FR3+Leap Hand 扩展、M04 的 MCC 控制链和 central-palm mount 修正。
FR3+Leap 模型为 `7 arm + 16 finger = 23 DoF`；MCC 初始 hand q 精确取自原 DP 视频
`t=2.000 s`。mount site 位于 palm mesh XY 分数 `[0.506,0.541]`，interface alignment
error 约 `2.6e-10 m`，不再把偏置 body origin 安到电机区域。

两个 MCC 单元均完成 3 个冻结 paired episodes，因此执行状态为 `EVALUATED`。E05 只比较
策略性能并报告参考限制越界，不再给策略写 `PASS/FAIL` 或 `MET/NOT_MET`。本次正式协议关闭
gravity 以隔离接触控制，结果不能外推为 gravity-on 或硬件性能。

上一版 standalone DP 的数据、因果 observation schema、训练量和短时评测均未通过用户
验收。旧代码、数据、checkpoint、视频和数字不复用。其后的 DP-direct v1 协议冻结了 observation、
force TCN、relative chunk、authority QP、runtime guard、dataset contract 与 diffusion core
已实现并通过单测。此前 hypothetical object motion 与临时 IK/linear repair 的错误链及其
生成数据已清理，不再提供复现入口。新的 3 秒真实
forward physical→spatial inverse→FR3+LEAP physical replay 已通过 raw replay gate：时间不反转、
原 forward finger command 数值原序复用、replay force/contact 重新测量、finger repair 为零。
但该早期 forward collector 使用 simulator Fingertip MCC，所以只能作为 Diag-MCC pipeline
diagnostic。其后已用三条通过整轨迹 hard gate 的 12 s TRAIN episode 在 RTX 4090 D 上完成
Diag-MCC-only learning-chain 验证，并在独立 12 s EVAL episode 上完成 10.5 s
`Finger DP-direct + Wrist MCC` 闭环，`D-Gate=PASS`。随后 non-MCC forward oracle、Dataset-I
I-Pilot20、formal pool、nested I20/I100 CUDA training 与 I20 held-out physical validation 均已
完成；I20 已用于 Exp. 1 DP-direct E05。已有磁盘目录仍保留历史 `d20/d100/pilot20` 名称。

2026-08-24 架构现改为：Finger DP 生成 nominal multi-finger trajectory/reference，shared Finger
MCC 负责高频柔顺执行。旧 DP-direct 不删除，作为 Exp. 1 架构消融。新的双头 DPRef、Role
Interpreter、Dataset-I I20/I100 重标注、CUDA I100 training 和 Exp. 2 已实现并运行。2026-08-25
又加入普通 Plain whole-hand MCC 绝对参考，并把 shared stack 从旧 8 N-priority profile 调整为
contact-priority profile。当前四策略 aggregate 中，Plain 的 continuity/平均接触数/`N_c>=2`
supported traversal 为 `0.992/3.156/138.87 mm`；严格共享栈三者中 DPRef 最好，为
`0.988/2.450/126.09 mm`，但其四指同时接触率仅 `14.86%`。旧 `G1a=PASS` 仅作 pre-retune
provenance；E05 只报告接触性能及持续/多指高力诊断，不形成策略 Gate。validation 仍缺 RELEASE。
Exp. 3 已移出 E05，安排在 I05 之后作为最终 active-planner dimensionality ablation。

旧 fixed-palm `E05-PHY-v3` evaluator、协议、报告和生成结果已经退役。其通用复杂曲面与
MuJoCo scene primitives 仍由当前 FR3+LEAP E05-F/H 使用，因此作为共享环境代码保留，
不再形成独立 MCC 版本或独立实验结论。

2026-08-24 用户明确授权在 DP 尚未就绪的条件下，只沿 Explicit MCC baseline 实现
M06–M12。本轮因此冻结并执行 `M06_M12_MCC_BASELINE_PROTOCOL.md`：七个模块均完成模块级
`EVALUATED / MET`，但该授权只豁免实现顺序，不把 G1 写成通过，也不授权 I01、DP/Main
planner integration、G2/G3 或 GPIS。数值验收使用 analytic plane + deterministic
linearized validation backend；Bunny 仅作为复杂物体展示。

2026-08-24 用户随后明确授权启动 Bunny 物理 I01。`I01_BUNNY_PROTOCOL.md` 已在实现和正式
运行前冻结；该授权只允许 Geometry-Oracle + Explicit MCC baseline 的 I01-A/I01-B，
不使用 DP、不把历史 G1 改写为通过。正式 3-seed paired MuJoCo 结果为
`EVALUATED / MET`：fixed-contact 中位进度 `19.885 mm`，variable-contact 中位进度
`58.429 mm`，优势 `38.544 mm`；variable 3/3 达到 primary pass 并完成实测
`4 -> 3 -> 4` handover，因此本 Bunny 协议下 `G2=GO`。G3 未开始。

2026-08-24 用户在审阅 I01-A/I01-B 区别后明确授权继续评测 I02/I03。实现前已冻结
`I02_I03_BUNNY_PROTOCOL.md`：I02 只比较一次 `12 mm` committed REPOSITION 与
`3 x 4 mm` fresh-snapshot replanning；I03 只比较相同 Beam 是否启用 prediction-only
ShadowSucc terminal filter。两者继续使用 Bunny Geometry-Oracle + Explicit MCC baseline，
DP/DPRef/GPIS 关闭；正式结果产生前不预先宣告 G3。

正式 `3 seeds x 4 cells` 已完成且 source-stability 为 PASS。I02 的 LONG/SHORT 均 3/3
完成共同任务和实测 handover；支持距离中位数为 `101.767/101.808 mm`，末端预测误差中位数
为 `1.476/1.467 mm`。SHORT 虽有每条恰好 3 个 REPOSITION certificate + fresh barrier，
但未达到冻结的 `<=0.8*LONG+0.25 mm` 改善阈值，故 `I02=EVALUATED / NOT_MET`。I03 的普通
Beam 3/3 选择 `SLIDE(3)` 并在真实 barrier 后成为 dead end；ShadowSucc 3/3 选择
`SLIDE(1)`、0 dead end、3/3 handover，支持距离中位优势 `94.014 mm`，故
`I03=EVALUATED / MET`。G3 要求二者同时 MET，因此 `G3=NO_GO`，禁止进入 GPIS。

2026-08-24 用户进一步冻结 I04 的核心定义：I04 从 active exploration 中拆出，命名为
`Oracle Next-Point Whole-Hand Contact Traversal`。固定的是完整 Bunny surface graph、必访集合
和完成条件；每个 `g_k=(x_k*,n_k*,t_k_out,epsilon_g,epsilon_n)` 必须依据 M06 barrier 后的当前
真实手位置/contact state 与剩余节点在线选择，Oracle 不输出 target finger。I04 累计集成 I01
的 variable contact、I02 的短 prefix + real-root replan 和 I03 的 terminal viability；Explicit 与
DPRef/Role 各自决定 finger assignment、contact transition 和 handover，并共享 transactional
execution、真实 barrier、MCC 与 MAKE-before-BREAK guard。ARRIVE 只由真实 mesh-geodesic
contact 与法向一致性判定。当前只冻结核心协议，未授权实现、训练或运行，数值层仍为 TBD；
完整定义见 `I04_ORACLE_NEXT_POINT_PROTOCOL.md`。I04 不包含 uncertainty、frontier、information
gain 或 next-best-touch selection，这些仍只属于 GPIS active exploration。

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

- **Main method**：`GPIS/Oracle Wrist-only Planner + Finger DP Reference Generator + shared
  Wrist MCC + shared Contact Force Coordinator + shared Fingertip MCC`；
- **Explicit baseline**：`Contact-Mode/whole-hand Planner + explicit fingertip trajectory/contact
  modes + the same Wrist MCC + the same Contact Force Coordinator + the same Fingertip MCC`；
- **共享部分**：状态契约、SurfaceModel 接口、RobotIO、runtime guards、对象、初始
  状态、wrist motion budget、Wrist MCC、Fingertip MCC、desired resultant-wrench convention、
  日志和 evaluator；
- **唯一核心变量**：Explicit 在线优化 finger trajectories/contact modes；Main 在线只优化
  wrist trajectory，并由 DPRef 生成 finger nominal references；
- **禁止隐藏分支**：显式 fingertip/contact-mode planner 不能成为 Main runtime fallback，DPRef
  也不能进入 Explicit 分支偷偷替换失败规划。

共享 Fingertip MCC 只解决高频法向力和小局部法向误差，不负责决定切向轨迹、可达性、碰撞、
接触拓扑或未来 viability。旧 DP-direct 分支只保留为 Exp. 1，不属于 Main runtime。

## 状态标记

主表只使用以下状态：

- `NOT_STARTED`：没有实现，也没有实验；
- `PROTOCOL_FROZEN`：接口、阈值、对象、seed 和 evaluator 已冻结，但尚未执行；
- `CORE_PROTOCOL_FROZEN`：问题定义和不可变语义已冻结，但数值阈值、fixture 或 schema 仍待
  单独冻结；该状态不授权实现或实验；
- `IN_PROGRESS`：已明确授权且正在实现或测试；
- `EVALUATED`：评测已完整执行；E05 comparative policies 只报告指标、相对优劣和参考限制越界；
- `EVIDENCE_ONLY`：已执行的预验证/诊断证据，不属于当前正式协议，不能解锁 Gate；
- `DEFERRED`：用户明确要求本轮不实现、不评测，且不作为本轮完成条件；
- `BLOCKED`：证据表明当前依赖或模块阻塞；
- `PASSED`：达到预先冻结的通过标准，并已登记证据；
- `FAILED`：模块/前置验证未达到冻结的有效性或安全标准，禁止推进依赖模块；已完整执行
  的比较实验不会仅因某个方法性能差而标成 `FAILED`；E05 只保持 `EVALUATED` 并描述性能。

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
| 4d | M04-DP-direct | Force-history direct Finger DP + non-MCC verified-inverse Dataset-I | M0-FR3, M03-FR3 | `EVALUATED`（I20/I100 trained；Exp. 1 ablation） |
| 4e | M04-DPRef | Learned multi-finger nominal trajectory/reference generator | M02-FR3, M04-H, M04-DP-direct data | `IMPLEMENTED / ROLE_COVERAGE_LIMITED` |
| 4r | M04-R | 23-DoF FR3+Leap MuJoCo plant | M0-FR3 | `PASSED`（central mount + exact 2 s posture） |
| 4w | M04-W | palm pose IK、wrench estimator 与 6D Wrist MCC | M04-R, M03-FR3 | `PASSED` |
| 4c | M04-C | Resultant/internal Contact Force Coordinator | M01-FR3, M02-FR3, M04-W | `PASSED` |
| 4h | M04-H | Wrist MCC + coordinated Fingertip MCC integration | M04-C | `PASSED` |
| 5pre | E05-PRE-FMCC | Fixed-palm Fingertip-MCC 物理预验证 | M01–M03 | `EVIDENCE_ONLY` |
| 5a | E05-F-MCC | 规定式 FR3 wrist + 四指 Fingertip MCC | M04-R, M02-FR3 | `EVALUATED`（见越界统计） |
| 5b | E05-H-MCC | FR3 Wrist MCC + coordinator + Fingertip MCC | M04-H | `EVALUATED`（Exp. 1） |
| 5c | E05-F-DP-direct | standalone Finger DP-direct diagnostic | M04-DP-direct, M03-FR3 | `NOT_STARTED`（非主结果） |
| 5d | E05-H-DP-direct | shared Wrist MCC + direct Finger DP + authority filter | M04-DP-direct, M04-W | `EVALUATED`（Exp. 1 ablation） |
| 5p | E05-H-PlainMCC | 普通解析 whole-hand MCC 绝对参考 | M04-H | `EVALUATED`（Exp. 2 absolute reference） |
| 5e | E05-H-PassiveMCC | shared Wrist/Finger MCC + passive nominal reference | M04-H | `EVALUATED`（Exp. 2） |
| 5f | E05-H-ReactiveMCC | shared Wrist/Finger MCC + causal heuristic reference | M04-H | `EVALUATED`（Exp. 2） |
| 5g | E05-H-DPRef+MCC | shared Wrist/Finger MCC + learned nominal reference/role | M04-DPRef | `EVALUATED`（Exp. 2；role coverage limited） |
| 6 | M06 | Transactional Prefix Executor | G1a for formal integration, M02, M03 | `EVALUATED / MET`（prior MCC-only module waiver） |
| 7 | M07 | ContactModeGraph | P0 | `EVALUATED / MET`（MCC-only module scope） |
| 8 | M08 | CheapCert | M01, M07 | `EVALUATED / MET`（MCC-only module scope） |
| 9 | M09 | ContinuousOptimize | M01, M02, M08 | `EVALUATED / MET`（linearized validation backend） |
| 10 | M10 | ExactPrefixAudit | M01, M03, M06, M09 | `EVALUATED / MET`（module scope） |
| 11 | M11 | Lazy Beam Search | M07–M10 | `EVALUATED / MET`（H2/H3 module scope） |
| 12 | M12 | Shadow Terminal Viability | M08, M11 | `EVALUATED / MET`（module scope） |
| 13 | I01 | Oracle Continuous Traversal | M01–M12 | `EVALUATED / MET`（Bunny MCC-only；`G2=GO`） |
| 14 | I02 | Receding-horizon prefix 对比 | I01 | `EVALUATED / NOT_MET`（3/3 均完成，但稳健性改善阈值未达到） |
| 15 | I03 | Terminal viability 对比 | I01, M12 | `EVALUATED / MET`（dead end `3 -> 0`） |
| 16 | I04 | Oracle Next-Point Whole-Hand Contact Traversal | I01–I03, M04-H, M04-DPRef | `CORE_PROTOCOL_FROZEN / NUMERICS_TBD` |
| 17 | M13 | GPIS SurfaceModel | G3, M01 | `NOT_STARTED` |
| 18 | M14 | GPIS Active Exploration | M13 | `NOT_STARTED` |
| 19 | I05 | GPIS Main vs. GPIS Baseline | G4, M14 | `NOT_STARTED` |
| 20 | I06 / Exp. 3 | Final active-planner dimensionality ablation | I05 | `NOT_STARTED` |

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

## Module 2：Fingertip MCC（Explicit 与 Main 共享执行层）

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

### Module 4DP-direct：已完成的 direct-controller 架构消融

`DP_CONTROLLER_V1_PROTOCOL.md` 现归档为 Exp. 1 的 DP-direct 协议。其 causal force history、
shared TCN、relative command chunk、Action Authority Filter、non-MCC Dataset-I、CUDA I20/I100
training 和 E05 结果均保留，用于回答“DP 能否直接替代 low-level Finger MCC”。

数据采用 spatial inversion，而非 temporal reversal。`Diag-MCC` 只验证 learning pipeline；正式
Dataset-I 使用 non-MCC privileged forward oracle，并严格区分 zero-repair `RAW_VERIFIED`、
`REPAIRED` 和 `REJECTED_DIAGNOSTIC`。已有磁盘路径中的 `d20/d100/pilot20` 保持不变，但文档
逻辑名称统一为 I20/I100/I-Pilot20。

Exp. 1 已完整评测。当前 DP-direct 的 contact continuity、平均 contact 数、force RMSE 和
traversal 均弱于 MCC，而 wrist force-z RMSE 较低；这不否定 learned finger trajectory
generation。它只说明当前 direct policy 不适合替代高频 analytical compliance。DP-direct
不再作为 Main runtime，也不再作为 planner integration Gate。

### Module 4DPRef：Learned Finger Reference Generator（已实现，role coverage limited）

新的 Main 将 DP 上移为 nominal multi-finger trajectory/reference generator：

```text
hand/contact state + future wrist plan -> shared encoder
 -> diffusion head: DPRef nominal finger trajectory
 -> categorical head: per-finger role intention
 -> shared Fingertip MCC compliant execution
```

active contact 的 nominal tangential/relative motion由 DPRef 生成，coordinated normal/internal
force 由 Finger MCC 调节，collective/resultant force 由 Wrist MCC 调节；free finger 的
reposition/MAKE nominal trajectory 可由 DPRef 完整生成。Role head 与 diffusion head 分开训练；
deterministic interpreter 只允许 `KEEP -> RELEASE -> FREE -> MAKE -> KEEP`。RELEASE 通过
`f_des -> 0` 平滑 ramp 卸力，最后接触 veto 与真实 contact confirmation 始终生效。

已实现 `dpref_dataset.py`、`dpref_policy.py`、`dpref_train.py`、`dpref_reference_sources.py`、
`reference_interpreter.py` 和 Exp. 2 evaluator。现有 I20/I100 RAW_VERIFIED 数据完成了 q_nom +
temporally confirmed role 重标注；旧 DP-direct checkpoint 没有复用。I100 在 RTX 4090 D 上完成
10,000 updates，continuous q_nom validation 已记录；validation 无 RELEASE 且 MAKE 只有 20 个
标签/60% accuracy，因此当前 checkpoint 的 role/hand-over 适用范围受限，不能宣称已泛化。

### Module 4R：FR3+Leap dynamic plant

已实现可控的 MuJoCo 整机模型：

- 7 个 FR3 position actuators 与 16 个 Leap Hand position actuators；
- `(nq,nv,nu)=(23,23,23)`，物体固定在 world，`nmocap=0`；
- flange 轴显式对准 central palm mesh，而不是偏置 body origin；四个 belly pad 绑定真实
  fingertip body；
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

## Experiment 5：E05 的两组控制层描述性评测

E05 现在明确包含两个不同问题；两组都只报告性能和参考限制越界，不设置策略
`PASS/FAIL` 或 `MET/NOT_MET`。完整 active-planner 对照另列为 I06 / Exp. 3。详细协议、
公平条件、指标与统一网页入口只以 `E05_EVALUATION_PLAN.md` 为准。

### Exp. 1：DP-direct 能否替代 Finger MCC（已完成）

```text
E05-H-MCC
vs.
E05-H-DP-direct
```

这是 low-level controller replacement ablation。三个 15 s paired episodes 已完整执行：

| aggregate 指标 | E05-H-MCC | E05-H-DP-direct |
| --- | ---: | ---: |
| mean contact continuity | 87.30% | 66.69% |
| mean contact count | 3.026 | 1.590 |
| mean force RMSE | 1.381 N | 2.232 N |
| worst peak force | 81.35 N | 103.02 N |
| mean Y traversal | 174.2 mm | 158.2 mm |
| controller P95 mean | 1.35 ms | 12.00 ms |

两边均已 `EVALUATED`。当前 DP-direct 没有表现出替代 MCC 的优势，因此保留为架构消融，不再
承担 main method 或 planner integration Gate。相对 8 N 参考限制，MCC/DP-direct worst peak
分别超出 `73.35/95.02 N`。旧协议 `E05_DP_CURRENT_PROTOCOL.md` 与
`DP_CONTROLLER_V1_PROTOCOL.md` 只用于复现 Exp. 1。

### Exp. 2：Plain + Passive / Reactive / DPRef+MCC（接触优先重评已执行）

```text
E05-H-PlainMCC = ordinary analytical whole-hand MCC absolute reference
E05-H-PassiveMCC = passive/hold nominal reference -> shared Finger MCC
E05-H-ReactiveMCC = causal heuristic reference -> same Finger MCC
E05-H-DPRef+MCC  = learned anticipatory reference -> same Finger MCC
```

后三者共享 Wrist MCC、Contact Force Coordinator、Finger MCC gains、force target、M03、wrist
trajectory、初态、对象、扰动、horizon 和 evaluator。唯一变量是 finger nominal reference 的
来源。Reactive 只可读取当前/过去 state 和 SurfaceModel，不可读取 future wrist plan。该实验
隔离 learned anticipation 是否改善 contact richness、handover 与 supported traversal，而不是
再次比较 DP 与 MCC 谁负责高频 force control。Plain 不经过新 Role/ForceSafety wrapper，只作
普通 MCC 绝对参考，不能参与 reference-source 因果归因。

四路均完成 nominal、low-friction、noisy-observation 三个 15 s 条件。Plain/Passive/Reactive/
DPRef 的 aggregate continuity 为 `0.992/0.972/0.973/0.988`，平均接触数为
`3.156/2.285/2.310/2.450`，`N_c>=2` supported Y 为
`138.87/89.35/86.90/126.09 mm`。

性能上，Plain 的绝对接触保持最好。在严格共享栈三者中，DPRef 的 continuity、平均接触数、
`P(N_c>=2/3)` 和 supported traversal 全部最好；相对最佳解析 source 分别提高 1.51 pp、0.139
个接触和 36.75 mm。但 DPRef 四指同时接触率 `14.86%`，低于 Reactive 的 `27.45%`，第四指
参与不足；validation 无 RELEASE 且 MAKE 覆盖很少，所以不能由当前结果声称 handover 泛化。
MuJoCo force 只作诊断：四种策略均无多指同时 `>8 N`，单个瞬时峰值不决定策略优劣。

统一网页与机器数据位于 `generated/e05_exp1_exp2_review/`。

### Shared low-level readiness provenance

- 旧 `G1a shared-compliance readiness = PASS` 只对应 2026-08-24 的 8 N-priority profile；
- 当前 contact-priority Exp.2 profile 不沿用该 verdict，也不以单个 MuJoCo peak 判定策略；
- Passive/Reactive/DPRef 的公平性来自同一次重评中完全共享的 interpreter/coordinator/MCC/M03；
- 原 `G1b` 策略性能 Gate 已退休；DPRef 的优缺点按连续数值报告，不再阻塞后续模块；
- M06–M12 的 MCC-only 模块级授权和结果不改变 G1a，也不等于完成正式 I01 traversal。

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

### M06–M12 本轮模块证据（2026-08-24）

冻结协议：[`M06_M12_MCC_BASELINE_PROTOCOL.md`](M06_M12_MCC_BASELINE_PROTOCOL.md)。
代码分别位于 `module_6_prefix_executor/` 到 `module_12_shadow_viability/`；统一 benchmark 为
`m06_m12_benchmark.py`，可视化为 `m06_m12_visual_demo.py`。

- M06：异步完成、blocked peer、transaction revocation、stale model 与 timeout 五类场景全部
  通过；micro-barrier 必须等待新的 timestamp；anchor command 使用现有 Fingertip MCC；
- M07：穷举 15 个非空 modes、131 个 legal primitive edges，重复枚举 deterministic；
- M08：4096 个 seeded candidates 对 nonlinear exact reference 的 false-negative rate 为 0；
- M09：`SLIDE/REPOSITION/MAKE/BREAK/WRIST_ADJUST` 各 32 个 cases，valid success rate 为 1；
  长 MAKE 正确输出 `MAKE_PROGRESS` 且不改变 topology；
- M10：正例签发 immutable certificate；中点碰撞、中途 joint limit、未确认 BREAK、stale
  version、trust exceed 与 suffix authority 六个 adversarial cases 全部拒绝；
- M11：beam width 8、per-mode quota 2 在 `H=2/3` 均保留 exhaustive optimum，score gap 为 0；
  只有 edge zero 可送 audit，后续 prefix 强制为 prediction suffix；
- M12：singleton viable/dead-end case 正确，按 finger 去重 successor，ShadowSucc 无证书权限。

机器结果与 provenance：
[`generated/m06_m12_mcc_baseline/summary.json`](generated/m06_m12_mcc_baseline/summary.json)；
审阅页：
[`generated/m06_m12_mcc_baseline/index.html`](generated/m06_m12_mcc_baseline/index.html)。集成
smoke 维持非空真实 contact set 并完成 `search -> audit -> MCC executor -> barrier snapshot`，
但它使用模块验证 plant，不是 I01/G2/G3 结果。

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

### I01 Bunny 正式结果（2026-08-24）

冻结协议：[`I01_BUNNY_PROTOCOL.md`](I01_BUNNY_PROTOCOL.md)；证据记录：
[`evidence/2026-08-24_I01_BUNNY_PHYSICS.md`](evidence/2026-08-24_I01_BUNNY_PHYSICS.md)；
审阅页：[`generated/i01_bunny_physics/index.html`](generated/i01_bunny_physics/index.html)。

- paired seeds 为 `7/11/19`，两组共享 Bunny、初态扰动、路径、时间律、MCC、guard 与 evaluator；
- fixed-contact 三次都因持续四指 mode 破坏触发 `FIXED_MODE_LOST`，实际进度中位数
  `19.885 mm`，但始终保持至少一指有效 Bunny 接触；这不是物理 reach limit，也不复现或替代
  历史 `47.97 mm` planning failure；
- variable-contact 三次实际进度为 `58.430/58.423/58.429 mm`，3/3 primary pass，3/3
  完成由 M10 certificate 与 M06 micro barrier 约束的实测 `4 -> 3 -> 4` handover；
- variable 非空接触比例为 `99.956%/99.933%/99.956%`，最大全接触丢失 gap `2 ms`，
  worst valid fingertip force `5.475 N < 8 N`；0 over-force tick、0 non-tip collision tick、
  0 authority violation；共签发 9 张 certificate、完成 9 个 barrier；
- `median(L_variable)-median(L_fixed)=38.544 mm >= 10 mm`，故本冻结 Bunny 协议的
  **G2 判定为 `GO`**。这条 I01 证据自身不决定 G1a/G3；后续独立评测现已得到
  历史 `G1a=ARCHIVED_PRE_RETUNE` 和当前 `G3=NO_GO`。E05 策略结果不再定义为 Gate。

物理碰撞使用从同一 Bunny mesh 生成的 `181 x 181` upper-envelope hfield；可视化和接触
残差审计使用完整变换后三角网格。正式实验为固定物体、gravity-off 的控制隔离仿真，不能
外推到完整非凸三角碰撞、gravity-on、硬件或 sim-to-real。

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

### I02/I03 Bunny 正式结果（2026-08-24）

冻结协议：[`I02_I03_BUNNY_PROTOCOL.md`](I02_I03_BUNNY_PROTOCOL.md)；机器结果：
[`generated/i02_i03_bunny_physics/summary.json`](generated/i02_i03_bunny_physics/summary.json)；
审阅页：[`generated/i02_i03_bunny_physics/index.html`](generated/i02_i03_bunny_physics/index.html)。

- I02-LONG 与 I02-SHORT 均 3/3 达到 `>=100 mm` supported cumulative traversal、3/3
  `4 -> 3 -> 4`；无 over-force、non-tip collision、authority 或 suffix-command violation；
- SHORT 每条恰好有 3 张 REPOSITION certificate 与 3 个 fresh measured barrier；LONG/SHORT
  supported traversal 中位数为 `101.767/101.808 mm`；
- LONG/SHORT 最终 terminal prediction error 中位数为 `1.476/1.467 mm`。冻结改善阈值为
  `0.8 * 1.476 + 0.250 = 1.431 mm`，SHORT 未达到；两组 failure count 又同为 0，故
  **I02=`EVALUATED / NOT_MET`**。这表示短前缀机制有效，但没有证明所要求的稳健性增益；
- I03-BEAM 三次均选 `SLIDE(3)`，真实 terminal joint margin 约 `0.020 rad`，M12 诊断
  `NONVIABLE` 并记录 3 个 dead end；I03-SHADOW 三次均选 `SLIDE(1)`，真实 margin
  `0.0478–0.0480 rad`，有 distinct successors、0 dead end、3/3 handover；
- BEAM/SHADOW supported traversal 中位数为 `7.111/101.125 mm`，优势 `94.014 mm`，故
  **I03=`EVALUATED / MET`**。M12 execution authority 始终为 false；
- 因 I02 与 I03 未同时 MET，**G3=`NO_GO`**。该结论只属于固定已知 Bunny、gravity-off、
  Geometry-Oracle + Explicit MCC baseline，DP/DPRef/GPIS 均未参与。

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

## Integration 4：Oracle Next-Point Whole-Hand Contact Traversal

冻结核心协议见 [`I04_ORACLE_NEXT_POINT_PROTOCOL.md`](I04_ORACLE_NEXT_POINT_PROTOCOL.md)。
I04 不做 GPIS active exploration：固定的是 privileged-certified Bunny surface graph、必访节点
和完成条件。每个 surface goal 根据 barrier 后的真实手位置/contact state、剩余节点和共同
OracleRoute contract 在线选择，但不指定 target finger：

```text
g_k = (x_k*, n_k*, t_k_out, epsilon_g, epsilon_n)
```

Explicit contact-mode/finger planner 与 DPRef/Role 分支各自决定哪个 finger 到达、如何管理
workspace 以及如何 MAKE/BREAK；两边共享 surface graph/selector contract、transactional
prefix、真实 barrier、MCC 和 runtime guards。闭环状态分叉后允许目标顺序随各自真实 state
调整，但所有 required nodes 都必须保留到真实到达，不能丢弃难区域。ARRIVE 必须由 MuJoCo
实测 contact 的 mesh geodesic distance 与 surface-normal alignment 确认，`t_k_out` 只作为后继
方向输入，不进入 arrival condition。Oracle 的 privileged feasibility certificate 可以基于当前
真实 root 证明至少存在一个合法 finger/configuration/transition，但 finger identity 和 witness
不得进入 runtime goal。

执行链累计包含 I01 的 variable contact handover、I02 的 short committed prefix + barrier-state
replan 和 I03 的 terminal continuation。Explicit 使用完整 M07–M12 planner；DPRef proposal 至少
共享 M10 execution audit、M06 transaction/barrier、MCC 与 M03 guard。DPRef 分支中 M08/M12
究竟作为共同 filter 还是 diagnostic-only，留待 numerical/method-boundary protocol 冻结，不能
让 Explicit planner 暗中替 DPRef 选择 finger。

I04 比较的是 given-good-next-point 条件下的 whole-hand contact realization ability；它明确排除
uncertainty、frontier、information gain、next-best-touch selection 和 reconstruction。正式实现
前仍需冻结 mesh/route、数值 tolerance、timeout/seeds、goal schema、DPRef checkpoint 与 trace
evaluator。本阶段没有实现或评测授权。

## Integration 5：GPIS Main vs. GPIS Explicit Baseline

最终完整比较：

```text
Explicit = GPIS + Contact-Mode Planner + shared Wrist MCC
         + Contact Force Coordinator + Fingertip MCC
Main     = GPIS + Wrist-only Planner + Finger DP Reference Generator
         + shared Wrist MCC + Contact Force Coordinator + Fingertip MCC
```

报告：

- Reconstruction：F-score、Chamfer、coverage；
- Exploration：uncertainty reduction/sec、coverage/sec、path length、explored area；
- Contact：continuous-contact success、average contacts、contact loss、handover；
- Planning efficiency：mean/P95 latency、solver failure、nodes/candidates、real-time factor。

## Integration 6 / Exp. 3：最终 active-planner dimensionality ablation

Exp. 3 放在 I05 之后。它使用已经完成验证的 active exploration stack，在匹配的 SurfaceModel、
初态、对象、wrist motion budget、receding-horizon protocol、MCC、guard 和 compute budget 下，
只替换在线 planner 的决策空间：

```text
Explicit = optimize wrist + finger trajectories + contact modes -> shared MCC
Main     = optimize wrist only -> DPRef/Role generates finger references -> shared MCC
```

I05 回答两个完整 GPIS 系统最终效果如何；I06/Exp. 3 则作为最后一个受控消融，专门回答省掉
显式 finger/contact-mode planning 后，exploration/contact quality 损失多少以及 planning latency、
变量数和 solver failure 改善多少。DP-direct 不进入该实验。

## 分层论文问题

| 问题 | 对应实验 |
| --- | --- |
| Q0 — DP 能否直接替代高频 Finger MCC？ | Exp. 1：E05-H-MCC vs. E05-H-DP-direct（架构消融） |
| Q1 — learned anticipatory finger reference 是否优于 passive/reactive compliance？ | Exp. 2：Passive-Hold / Reactive-Heuristic / DPRef+MCC |
| Q2 — Variable contact modes 是否扩大 feasible region？ | Oracle Fixed vs. Variable Contact |
| Q2b — 已给定优质下一表面点时，哪种方法能更好地实现整手连续接触到达？ | I04：Oracle Next-Point Whole-Hand Contact Traversal |
| Q3 — 是否需要显式高维 finger/contact planning？ | I06 / Exp. 3：I05 后的 final active-planner dimensionality ablation |
| Q4 — 未知物体上能否高效闭环探索？ | GPIS Explicit vs. Main |

## 系统级 Go/No-Go Gates

| Gate | Go 条件 | No-Go 后动作 |
| --- | --- | --- |
| G1a（历史） | 2026-08-24 8 N-priority shared stack 的 safety/readiness 审计 | 只保留 provenance；不外推到 contact-priority Exp.2 profile |
| G2 | Variable mode 真实突破 fixed-contact 局部可行域 | 检查 mode legality/optimizer/audit，不引入 GPIS |
| G3 | Oracle 下长期连续 traversal 且无 dead end | 修复 planner/executor/certificate，禁止 GPIS 集成 |
| G4 | 换成 GPIS 后仍能安全闭环 | 修复 SurfaceModel/uncertainty margin，禁止 full comparison |

只有通过 G3，才投入主要精力进行 GPIS integration。

当前 Gate 记录：`G1a=ARCHIVED_PRE_RETUNE`、`G2=GO`（仅上述 Bunny I01 冻结协议）、
`G3=NO_GO`（I02 NOT_MET；I03 MET）、`G4=NOT_STARTED`。

## 获得授权后的严格实施顺序

```text
0.  [DONE] M0 full-robot environment/state/action/wrench/log contract
1.  [DONE] M01 Oracle + live FR3 geometry
2.  [DONE] M02 Fingertip MCC + moving-wrist four-finger wrapper
3.  [DONE] M03 hand-only + FR3 grouped Runtime Guards
4.  [DONE] M04-R FR3–Leap plant
5.  [DONE] M04-W 6D Wrist MCC + wrench estimator
6.  [DONE] M04-C/H resultant/internal coordinator and integration
7.  [EVALUATED] E05-F-MCC + E05-H-MCC historical control evidence and limit observations
8.  [DONE / REMOVED] Remove unaccepted legacy DP code and generated data
9.  [DONE] Freeze causal DP v1 architecture and teacher/evaluation protocol
10. [DONE / DIAGNOSTIC] Real forward + spatial-only raw physical replay Diag-MCC gate
11. [DONE / LONG GPU D-GATE PASS / DIAGNOSTIC] 3×12 s TRAIN + independent 10.5 s DP/Wrist-MCC closed loop
12. [DONE / I-GATE PASS] Non-MCC forward oracle + 12 s zero-repair raw replay
13. [DONE / RAW-GATE PASS] I-Pilot20 + RAW/REPAIRED/REJECTED pools
14. [DONE TO I100 / STOP AT I20] Nested I20/I100 CUDA training and object-disjoint held-out validation
15. [EVALUATED / DESCRIPTIVE ABLATION] Exp. 1 E05-H-MCC vs E05-H-DP-direct
16. [DONE / G1a ARCHIVED PRE-RETUNE] Validate the historical 8 N-priority shared stack
17. [DONE / ROLE COVERAGE LIMITED] Freeze DPRef contract; relabel/audit I20/I100; CUDA train
18. [EVALUATED / CONTACT-PRIORITY DESCRIPTIVE COMPARISON] Exp. 2 Plain + Passive / Reactive / DPRef+MCC
19. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] Transactional Prefix Executor
20. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] ContactModeGraph
21. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] CheapCert
22. [EVALUATED / MET / LINEARIZED VALIDATION BACKEND] ContinuousOptimize
23. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] ExactPrefixAudit
24. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] LazyBeamSearch
25. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] ShadowSucc
26. [EVALUATED / MET / G2 GO / BUNNY MCC-ONLY] Oracle Continuous Traversal
27. [EVALUATED / NOT_MET / BUNNY MCC-ONLY] Receding-Horizon Prefix
28. [EVALUATED / MET / BUNNY MCC-ONLY] Terminal Viability
29. [CORE PROTOCOL FROZEN / NUMERICS TBD / NOT AUTHORIZED TO IMPLEMENT] Oracle Next-Point Whole-Hand Contact Traversal
30. [BLOCKED BY G3] GPIS SurfaceModel
31. [BLOCKED BY G3] GPIS Active Exploration
32. [BLOCKED BY G3] Full GPIS Main vs. Baseline
33. [AFTER I05] I06 / Exp. 3 final active-planner dimensionality ablation
```

不得因为某个后续模块更容易展示而跳过顺序或 Gate。M06–M12 是用户明确授权的 MCC-only
模块实现豁免；随后用户又单独授权并已完成 Bunny I01，所以该实验可独立给出 `G2=GO`。
用户随后单独授权的 I02/I03 已完成；当前 G3=NO_GO，所以不能进入 GPIS。I04 已从 active
exploration 中拆出并只冻结核心协议，不因 G3 状态被重新解释为 GPIS，也没有实现/运行授权。
E05 Exp.1/2 不是系统解锁 Gate；I06/Exp.3 固定在 I05 之后执行。

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
