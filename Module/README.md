# Module：FR3 + LEAP Hand 模块化控制实验

`Module/` 是当前唯一实现入口，固定使用 `handcomp`。M0–M3、整手 MCC、已完成的 DP-direct
架构消融、E05 evaluator，以及 Geometry-Oracle + MCC baseline 的 M06–M12 都在本目录内；
`Module/` 外的历史实现不修改、不删除、也不作为当前证据。

当前 main design 为 `Finger DP Reference Generator -> shared Finger MCC`。DPRef 双头模型、Role
Interpreter、共享执行层和 Exp. 2 均已实现并运行。E05 Exp.1/2 现在只做策略性能描述与参考
限制越界统计，不再给策略设置 `PASS/FAIL` 或 `MET/NOT_MET`。DPRef validation 缺 RELEASE，
所以 checkpoint 的 role coverage 仍受限。Exp.3 已移到 I05 之后，作为最终 active-planner
ablation。用户另行授权的 MCC-only I01–I03 已在 Bunny 物理场景完成，且不使用 DP。
I04 已从 active exploration 中拆出并冻结为 Oracle next-point 整手连续接触到达协议；
Explicit MCC development implementation、专项测试和三轮物理回归已完成，但完整 274-goal
traversal 尚未完成，DPRef 分支也未开始。当前结果只能解释为开发回归，不能宣称 full traversal。

## 当前状态

| 模块/实验 | 状态 | 当前结论 |
| --- | --- | --- |
| M0-FR3 | `PASSED` | 7-DoF FR3 + 16-DoF LEAP 状态、动作、wrench、日志契约 |
| M01-FR3 | `PASSED` | Oracle surface/normal/clearance、live arm capsules、belly-pad distance |
| M02-FR3 | `PASSED` | moving-wrist 四指 Fingertip MCC 与 signed force error |
| M03-FR3 | `PASSED` | arm/finger guards + MCC/DP 共用 deterministic force release |
| FR3+LEAP plant | `PASSED` | flange adapter 连到中央 palm plate；四个 mesh-registered belly pads |
| E05-F/H-MCC | `EVALUATED` | 旧 MCC-only 六条完整 15 s 结果和越界统计保留 |
| Diag-MCC Track D | `D-GATE PASS` | 只证明 learning/execution pipeline，不是正式 teacher |
| Dataset-I Track I | `I/RAW PASS` | non-MCC oracle；I-Pilot20 为 12 raw / 8 rejected / 0 repaired |
| DP-direct I20/I100 | `TRAINED / HELDOUT PASS` | 已有 CUDA checkpoint 与历史兼容路径 |
| Exp. 1 H-MCC vs H-DP-direct | `EVALUATED` | MCC 接触/力性能更好；两者的 8 N 越界量见统一网页 |
| G1a shared execution | `ARCHIVED_PRE_RETUNE` | 旧 8 N-priority profile 的审计保留；当前 Exp.2 使用接触优先 profile |
| M04-DPRef | `IMPLEMENTED / ROLE_COVERAGE_LIMITED` | CUDA I100：连续 reference 已记录；validation 缺 RELEASE，MAKE 仅 20 labels/60% |
| Exp. 2 Plain/Passive/Reactive/DPRef+MCC | `EVALUATED` | Plain 绝对接触最好；严格共享栈三者中 DPRef 的 continuity、平均 contacts 和 supported traversal 最好 |
| I06 / Exp. 3 active-planner ablation | `NOT_STARTED` | 位于 I05 后；不属于 E05，也不由 E05 策略 verdict 解锁 |
| M06–M12 MCC-only module protocol | `EVALUATED / MET` | 七个模块达冻结模块阈值；不改变 G1，不是 I01/G2/G3 |
| I01 Bunny physics | `EVALUATED / MET` | variable 3/3 pass，`58.429 mm`；fixed `19.885 mm`；`G2=GO` |
| I02 Bunny prefix | `EVALUATED / NOT_MET` | LONG/SHORT 均 3/3 pass；误差 `1.476/1.467 mm`，未达冻结改善阈值 |
| I03 Bunny viability | `EVALUATED / MET` | Beam/Shadow dead end `3/0`；支持距离中位数 `7.111/101.125 mm` |
| I04 Oracle next-point traversal | `EXPLICIT DEV IMPLEMENTED / FULL TRAVERSAL INCOMPLETE` | 274 个 required goals；三轮回归最多 7 个，当前阻塞在 two-anchor WRIST optimization |
| G3 | `NO_GO` | I02 未 MET；不进入 GPIS |

E05 的 `EVALUATED` 只表示 evaluator 与 episode 完整有效；策略优劣由连续指标与越界量描述。
I01–I03 等系统实验仍可按各自冻结协议使用模块级 Gate，不要与 E05 策略评价混用。

## 先看可视化

1. **E05 Exp.1 + Exp.2 统一指标、优劣分析、图片和全部视频：**
   [`generated/e05_exp1_exp2_review/index.html`](generated/e05_exp1_exp2_review/index.html)
2. DPRef relabel/training audit：
   [`generated/dpref_v1/training_i100/dpref_training_and_label_audit.png`](generated/dpref_v1/training_i100/dpref_training_and_label_audit.png)
3. 旧 G1a 8 N-priority profile 的历史视频与指标：
   [`generated/g1a_shared_stack/README.md`](generated/g1a_shared_stack/README.md)
4. I01 Bunny 物理对比、连续接触指标和同步视频：
   [`generated/i01_bunny_physics/index.html`](generated/i01_bunny_physics/index.html)
5. M06–M12 每个模块的目的、效果、性能和 Bunny 展示：
   [`generated/m06_m12_mcc_baseline/index.html`](generated/m06_m12_mcc_baseline/index.html)
6. I02/I03 Bunny 的 prefix/viability 同步视频、dashboard 与冻结结论：
   [`generated/i02_i03_bunny_physics/index.html`](generated/i02_i03_bunny_physics/index.html)
7. Exp. 1 原始 trace 页面（provenance）：
   [`generated/e05_h_mcc_vs_dp/review.html`](generated/e05_h_mcc_vs_dp/review.html)
8. E05 首次接触分叉、过力持续时间、Authority Filter 与训练状态覆盖：
   [`generated/e05_h_mcc_vs_dp/diagnostics/review.html`](generated/e05_h_mcc_vs_dp/diagnostics/review.html)
9. M0–M3、中央 mount、自然姿态与 MCC gallery：
   [`generated/visual_demo/index.html`](generated/visual_demo/index.html)
10. Dataset-I 数据、训练与 held-out 索引：
   [`generated/finger_dp_formal_v1/README.md`](generated/finger_dp_formal_v1/README.md)
11. 早期 Diag-MCC D-Gate diagnostic：
   [`generated/whole_hand_dp_long_v1/review.html`](generated/whole_hand_dp_long_v1/review.html)

正式 nominal 中，MCC/DP contact continuity 为 `92.1%/81.6%`，平均 contacts
`3.23/1.81`，peak `25.05/45.94 N`，Y traversal `173.9/174.7 mm`。视频没有截掉 8–10 s
失联或后续过力段。

## 文件结构

```text
Module/
├── README.md                         # 当前结构、状态与复现入口
├── LOCAL_REVIEW.md                   # 所有审阅路径和结果解释
├── MASTER_PLAN.md                    # 唯一主任务/Gate 记录
├── PROTOCOL.md                       # M0–M3 验收协议
├── E05_MCC_CURRENT_PROTOCOL.md       # MCC-only 冻结评测
├── E05_DP_CURRENT_PROTOCOL.md        # Exp. 1 H-MCC/H-DP-direct 归档协议（兼容文件名）
├── E05_EVALUATION_PLAN.md            # E05 Exp. 1/2 描述性评测；Exp. 3 路由到 I06
├── DP_CONTROLLER_V1_PROTOCOL.md      # Exp. 1 DP-direct 归档协议
├── DP_REFERENCE_GENERATOR_DESIGN.md  # DPRef+MCC 已冻结接口、实现和结果边界
├── evidence/
│   ├── 2026-08-24_DPREF_EXP2.md      # 旧 8 N-priority G1a/Exp.2 provenance
│   └── 2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md # 当前接触优先四策略重评
├── I01_BUNNY_PROTOCOL.md             # Bunny fixed/variable 物理对比冻结协议
├── I02_I03_BUNNY_PROTOCOL.md         # Bunny prefix/terminal viability 冻结协议
├── I04_ORACLE_NEXT_POINT_PROTOCOL.md # I04 no-target-finger 核心协议与当前实现边界
├── I04_RESUME_CHECKPOINT_2026-08-25.md # I04 回归证据、阻塞点与续作顺序
├── M4_DP_GUIDE.md                    # Diag-MCC/Dataset-I 与 DPRef 指导
├── WHOLE_HAND_COMPLIANCE_DESIGN.md   # Main/Explicit 共享 Wrist/Finger MCC 数学与职责
├── common/                           # hand/full-robot state 和 JSONL contract
├── fr3_leap/
│   └── model.py                      # 23-DoF MJCF、central mount、belly pads
├── module_1_oracle_surface_model/    # M01：Oracle 与 robot geometry
├── module_2_fingertip_mcc/           # M02：Fingertip MCC
├── module_3_runtime_guards/           # M03：guard、force safety 与 command continuity
├── module_4_whole_hand_mcc/          # Wrist/Finger MCC、Role Interpreter、G1a/Exp.2 runner
├── module_4_finger_dp/               # DP-direct、DPRef、Dataset-I relabel、CUDA training、Exp.1/2
├── module_6_prefix_executor/          # M06：certificate-gated transaction + barrier
├── module_7_contact_mode_graph/       # M07：15 个非空 contact modes
├── module_8_cheap_cert/               # M08：低 false-negative cheap screen
├── module_9_continuous_optimize/      # M09：五类 primitive 连续轨迹
├── module_10_exact_prefix_audit/      # M10：swept audit 与唯一 certificate authority
├── module_11_lazy_beam_search/        # M11：diverse lazy beam + suffix warm start
├── module_12_shadow_viability/        # M12：prediction-only terminal continuation
├── i01_bunny_physics/                 # I01 Bunny scene、runner、benchmark 与可视化
├── i02_i03_bunny_physics/             # I02/I03 planner、物理 runner、benchmark 与可视化
├── i04_oracle_next_point/              # I04 full-mesh graph、Explicit planner/runner/benchmark/demo
├── m06_m12_benchmark.py               # 冻结 benchmark、CSV、trace 与 provenance
├── m06_m12_visual_demo.py             # 同 visual_demo 风格的七模块 gallery
├── e05_physics/                      # 大尺寸连续强起伏 surface/scene
├── tests/                            # 单元、物理 smoke、benchmark 与可视化测试
└── generated/
    ├── visual_demo/                  # M01–M03 gallery
    ├── i01_bunny_physics/            # I01 trace/CSV/summary/dashboard/video/HTML
    ├── i02_i03_bunny_physics/        # I02/I03 12 episodes、paired video/dashboard/HTML
    ├── m06_m12_mcc_baseline/         # M06–M12 summary/CSV/trace/PNG/HTML
    ├── e05_mcc_current/              # MCC-only 结果
    ├── finger_dp_formal_v1/          # Dataset-I、I20/I100（历史路径 d20/d100）、held-out
    ├── dpref_v1/                      # q_nom/role relabel、I100 CUDA checkpoint 与 audit
    ├── g1a_shared_stack/              # 旧 8 N-priority shared-stack 审计 provenance
    ├── exp2_dpref_mcc/                # 四策略接触优先 Exp.2 trace/video/dashboard/summary
    ├── e05_h_mcc_vs_dp/              # Exp. 1 paired direct-DP trace/video/dashboard
    └── e05_exp1_exp2_review/          # Exp.1/2 统一网页、指标表、图片和视频
```

## 环境

从仓库根目录运行：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

Finger DP 神经训练和推理只允许 CUDA；CUDA 不可见时 fail closed，不回退 CPU。MuJoCo physics
与 DAQP 使用各自 backend，不属于神经网络 CPU fallback。

## M0–M3 使用与复现

```bash
$PY -m Module.module_1_oracle_surface_model.demo --seed 7
$PY -m Module.module_1_oracle_surface_model.mesh_demo
$PY -m Module.module_2_fingertip_mcc.demo
$PY -m Module.module_3_runtime_guards.demo
```

M01：

```python
from Module.fr3_leap import build_full_robot
from Module.module_1_oracle_surface_model import FullRobotGeometryAdapter

handles = build_full_robot()
geometry = FullRobotGeometryAdapter(handles)
capsules = geometry.world_capsules(data)
distance, witness = geometry.physics_pad_object_distance(data)
```

M02：

```python
from Module.module_2_fingertip_mcc import FingertipMCC

mcc = FingertipMCC()
command = mcc.step(plan, compression_direction, desired_force, measured_force)
```

若 SurfaceModel normal 指向物体外部，增加压缩的方向为 `-normal`。

M03 共用安全执行层：

```python
from Module.module_3_runtime_guards import ForceSafetyConfig, ForceSafetyExecutor

safety = ForceSafetyExecutor(ForceSafetyConfig(joint_lower_rad=q_min, joint_upper_rad=q_max))
result = safety.step(
    fingertip_force_n=forces,
    force_valid_mask=valid,
    history_ready=history_ready,
    current_q_rad=q,
    signed_compression_jacobian=J_s,
)
```

`J_s Δq > 0` 必须定义为压缩增加；hard release 始终执行 `J_s Δq < 0`。M03 权限高于
Finger MCC、Finger DP 和 authority QP。

## MCC 使用与复现

```bash
# MCC-only 3 paired episodes × F/H cells × 15 s
$PY -m Module.module_4_whole_hand_mcc.demo

# 从保存的 trace 重建视频
MUJOCO_GL=osmesa $PY -m Module.module_4_whole_hand_mcc.visual_demo
```

H-MCC 使用 grasp/contact map 将 force error 分成 resultant 与 internal component：Wrist MCC
调前者，四个 Fingertip MCC 只调后者。详细数学见 `WHOLE_HAND_COMPLIANCE_DESIGN.md`。

## M06–M12：Oracle + Explicit MCC Baseline

冻结协议见 [`M06_M12_MCC_BASELINE_PROTOCOL.md`](M06_M12_MCC_BASELINE_PROTOCOL.md)。执行链为：

```text
ContactModeGraph -> CheapCert -> ContinuousOptimize
                 -> LazyBeamSearch -> ShadowSucc
first edge only -> ExactPrefixAudit -> ExecutionCertificate
                 -> TransactionalPrefixExecutor -> micro barrier -> measured snapshot
                 -> existing Fingertip MCC / runtime guards
```

M07/M08/M09/M11/M12 都只能预测；只有 M10 能签发证书，M06 只执行证书绑定的
`Pi_commit`。`Pi_suffix` 仅供下一轮 shifted warm start；timeout、model-version drift、global
guard 或最后接触丢失均进入 `SAFE_HOLD`。

当前正式本机结果（Intel i7-13700，24 logical CPUs；P95 timing boundary 见 `summary.json`）：

| 模块 | 效果 | P95 latency |
| --- | --- | ---: |
| M06 | 5/5 transaction scenarios，0 authority violation | 0.168 ms / step |
| M07 | 15 modes，131 legal edges，0 invariant violation | 1.157 µs / legality |
| M08 | 4096 candidates，FN rate 0 | 3.626 µs / screen |
| M09 | 5×32 cases，success 100%，target error P95 `2.97e-9 m` | 1.159 ms / optimize |
| M10 | 65 swept samples，6/6 adversaries rejected | 3.031 ms / audit |
| M11 | H2/H3 optimum retention 100%，H3 96 vs. 246 optimized edges | 50.97 / 122.69 ms |
| M12 | 1024 states，viable/dead-end 正确 | 0.203 ms / state |

```bash
# 只跑 M06–M12 的语义/算法测试
$PY -m unittest Module.tests.test_m06_m12_planner -v

# 4096 CheapCert candidates、5×32 optimizer、256 audits、beam/exhaustive、1024 shadow states
$PY -m Module.m06_m12_benchmark

# 读取同一次 summary/traces，生成七张 PNG 与 HTML；包含 Bunny 展示
$PY -m Module.m06_m12_visual_demo --reuse-benchmark
```

机器可读结果为
[`generated/m06_m12_mcc_baseline/summary.json`](generated/m06_m12_mcc_baseline/summary.json)
与 [`performance.csv`](generated/m06_m12_mcc_baseline/performance.csv)。数值验收使用 analytic
plane + deterministic linearized backend；Bunny 只用于展示复杂 mesh 接口。集成 smoke 使用
现有 `FullRobotFingertipMCC`，但不是 FR3 nonlinear IK、长程 MuJoCo traversal 或正式 I01。

## I01：Bunny 上的连续接触物理验证

冻结协议见 [`I01_BUNNY_PROTOCOL.md`](I01_BUNNY_PROTOCOL.md)。I01 使用 FR3+LEAP 的 23-DoF
MuJoCo plant、现有 Fingertip MCC/M03 safety、M10 swept audit 和 M06 transactional executor；
Bunny 固定，DP 关闭。两组共享完全相同的 Bunny、seed、初态、路径和控制阈值，唯一实验变量
是是否允许经证书约束的 contact-mode handover。

| cell | 实际进度中位数 | 非空接触比例 | 最大全失联 gap | worst peak | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed `|A|=4` | `19.885 mm` | `100%` | `0 ms` | `4.736 N` | 0/3 primary pass；`FIXED_MODE_LOST` |
| variable `4->3->4` | `58.429 mm` | `99.933–99.956%` | `2 ms` | `5.475 N` | 3/3 primary pass；3/3 handover |

variable 相对 fixed 的中位优势为 `38.544 mm`，超过冻结的 `10 mm` 门槛；9 张 M10
certificate 对应 9 次 M06 micro barrier，authority violation、over-force tick 和 non-tip
collision tick 都为 0，因此本协议 `EVALUATED / MET`、`G2=GO`。

```bash
# I01 几何、模型和短物理 smoke
$PY -m unittest Module.tests.test_i01_bunny_physics -v

# 正式 2 cells × 3 seeds；写入 summary/CSV/完整 trace
$PY -m Module.i01_bunny_physics.benchmark

# 只读取同一次 summary/trace，重建 dashboard、关键帧、视频和审阅页
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/handcomp-i01-cache \
  $PY -m Module.i01_bunny_physics.visual_demo --reuse
```

审阅入口为
[`generated/i01_bunny_physics/index.html`](generated/i01_bunny_physics/index.html)，机器结果为
[`summary.json`](generated/i01_bunny_physics/summary.json) 和
[`episodes.csv`](generated/i01_bunny_physics/episodes.csv)。benchmark 只有在冻结 acceptance 与
G2 条件都满足时返回 0；`visual_demo --reuse` 不重算性能指标。

碰撞体是从同一完整 Bunny mesh 生成的 `181 x 181` upper-envelope hfield；每个计分接触还要
通过完整三角网格 `<=2.5 mm` 残差审计。该实验是固定物体、gravity-off control-isolation，
不能外推为完整非凸 mesh、gravity-on、硬件或 sim-to-real 性能。旧 `G1a=PASS` 已归档为
pre-retune provenance；当前接触优先 Exp.2 不沿用该 verdict。当前
`G3=NO_GO`；E05 Exp.1/2 不再定义策略 Gate。

## I02/I03：Bunny 上的 Prefix 与 Terminal Viability

冻结协议见 [`I02_I03_BUNNY_PROTOCOL.md`](I02_I03_BUNNY_PROTOCOL.md)。正式实验是
`4 cells x seeds 7/11/19 x 20 s`，统一使用 `110 mm` 往返路径、exact-mesh contact evaluator、
MCC 与 M10→M06 执行权限；DP/DPRef/GPIS 关闭。

| 对比 | 目的 | 结果 | 性能结论 |
| --- | --- | --- | --- |
| I02 LONG vs. SHORT | 一次 12 mm prefix vs. 3×4 mm fresh-root replan | 两组均 3/3 task + handover；SHORT 每条 3 cert/3 barrier | error `1.476 -> 1.467 mm`，未到 `1.431 mm` 阈值；`NOT_MET` |
| I03 BEAM vs. SHADOW | 是否启用 M12 terminal filter | dead end `3 -> 0`；SHADOW 3/3 task + handover | traversal `7.111 -> 101.125 mm`；`MET` |

I03-SHADOW 的真实 terminal joint margin 为 `0.0478–0.0480 rad`，M12
`execution_authority=false`；全部推荐 cell 无 over-force、non-tip collision、authority violation
或 prediction-suffix command。由于 G3 要求 I02/I03 同时 MET，当前 `G3=NO_GO`。

```bash
# 语义/路径单测
$PY -m unittest Module.tests.test_i02_i03_bunny_physics -v

# 正式 12 episodes；写入 source hash、CSV、summary 和完整 NPZ trace
$PY -m Module.i02_i03_bunny_physics.benchmark

# 只回放同一次 trace，不重算 acceptance
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/handcomp-i02-i03-cache \
  $PY -m Module.i02_i03_bunny_physics.visual_demo --reuse
```

审阅页：[`generated/i02_i03_bunny_physics/index.html`](generated/i02_i03_bunny_physics/index.html)；
机器结果：[`summary.json`](generated/i02_i03_bunny_physics/summary.json) 与
[`episodes.csv`](generated/i02_i03_bunny_physics/episodes.csv)。结果只适用于固定已知 Bunny、
gravity-off 的 Geometry-Oracle + Explicit MCC baseline。

## I04：Oracle Next-Point Whole-Hand Contact Traversal

核心协议见
[`I04_ORACLE_NEXT_POINT_PROTOCOL.md`](I04_ORACLE_NEXT_POINT_PROTOCOL.md)。I04 不做 GPIS
active exploration；固定的是完整 Bunny surface graph、必访节点和完成条件。每个目标由 M06
barrier 后的真实手位置/contact state 与剩余节点在线产生：

```text
g_k = (surface point, surface normal, outgoing tangent, geodesic tolerance, normal tolerance)
```

Oracle **不指定 target finger**。Explicit contact-mode/finger planner 与 DPRef/Role 各自决定
finger assignment、workspace management、MAKE/BREAK 和 handover；两边共享 surface graph/
selector contract、transactional execution、真实状态 barrier、MCC 与 measured-contact
MAKE-before-BREAK guard。目标顺序可以随各自闭环真实状态调整，但不能从 remaining-node ledger
中丢弃难区域，只有所有 certified required nodes 都真实到达才完成整只 Bunny。

ARRIVE 只由 MuJoCo 真实 fingertip contact 判定：接触点到目标的完整 Bunny mesh geodesic
distance 在容差内，且真实 contact normal 与目标法向一致。`outgoing tangent` 只作为 controller
input，不进入到达条件。离线 privileged feasibility certificate 可以证明至少存在一个可达
finger/configuration，但 runtime goal 必须剥离 finger identity 和 witness。

I04 累计使用 I01 的 variable contact、I02 的短 prefix + real-state replan 和 I03 的 terminal
viability。Explicit 分支走完整 M07–M12；DPRef proposal 至少共享 M10→M06 execution authority、
MCC 和 M03 guards。I04 只比较 given-good-next-point 下谁到达更多、更快、接触更连续，以及
handover、力、关节余量和计算代价；不使用策略 `MET/NOT_MET`。uncertainty、GPIS frontier、
information gain、next-best-touch 和 reconstruction 仍属于后续 GPIS。

当前已实现完整 Bunny mesh graph、state-conditioned OracleRoute、Explicit M07–M12/M10/M06
执行链、trace/evaluator 与可视化。development 数值为 25 mm coverage、12 mm mesh-geodesic
arrival tolerance 和 55° normal tolerance；它们尚未升级为最终正式数值协议。三轮保存回归中
最多完成 `7/274` goals；最长一轮 90 s 完成 `6/274`，接触连续率 `99.9839%`，但覆盖随后停滞。
详细 provenance、已修问题及当前 two-anchor WRIST blocker 见
[`I04_RESUME_CHECKPOINT_2026-08-25.md`](I04_RESUME_CHECKPOINT_2026-08-25.md)。

```bash
# 专项语义、规划器与 FR3+LEAP 回归
$PY -m unittest -q \
  Module.tests.test_m06_m12_planner \
  Module.tests.test_i04_oracle_next_point \
  Module.tests.test_fr3_leap_model

# 短开发运行；输出写入 ignored generated 目录
$PY -m Module.i04_oracle_next_point.benchmark \
  --profile smoke --duration 45 --goals 5

# 只回放同一次 canonical trace；可将编码器切到 h264_nvenc
MUJOCO_GL=osmesa $PY -m Module.i04_oracle_next_point.visual_demo \
  --reuse --speed 12 --codec h264_nvenc
```

正式入口为 `--profile formal`，但在 WRIST blocker 未解决前不应将其当作完成性实验启动。
generated trace、视频和 checkpoint 默认由 `.gitignore` 排除；Git 中发布的是代码、协议、测试
与可复现命令，不把本机回归二进制伪装成仓库内正式结果。

## Exp. 1 DP-direct 与历史 E05 复现

完整文件说明与命令见
[`module_4_finger_dp/README.md`](module_4_finger_dp/README.md)。最常用命令：

```bash
# 当前全量回归
$PY -m unittest discover -s Module/tests -v

# 从冻结 I20 数据重新 CUDA training（路径保留历史 d20 名称）
$PY -m Module.module_4_finger_dp.formal_train \
  --train Module/generated/finger_dp_formal_v1/scaling/dataset_i_d20_train.npz \
  --validation Module/generated/finger_dp_formal_v1/formal_pool_v1/dataset_i_validation.npz \
  --output Module/generated/finger_dp_formal_v1/training_d20_reproduction \
  --updates 10000 --device cuda:0

# 重跑 Exp. 1 三组 H-MCC/H-DP-direct 物理评测
$PY -m Module.module_4_finger_dp.e05_dp_benchmark \
  --checkpoint Module/generated/finger_dp_formal_v1/training_d20/formal_finger_dp_checkpoint.pt \
  --output Module/generated/e05_h_mcc_vs_dp

# 只重建可视化
MUJOCO_GL=osmesa $PY -m Module.module_4_finger_dp.e05_dp_visual \
  --output Module/generated/e05_h_mcc_vs_dp

# 不重跑控制器，只分析冻结 trace 和 I20 状态覆盖
$PY -m Module.module_4_finger_dp.e05_failure_diagnostics
```

配对 E05 前 1 s 为相同且不计分的 contact initializer；随后只替换 `Finger MCC` 与
`Finger DP-direct + Authority Filter`。两边共享 Wrist MCC、M03 safety、轨迹、初态、扰动和
limits。这是 Exp. 1 架构消融，不是新 main method。

## Exp. 2 DPRef+MCC 与后续 Exp. 3

```text
Exp. 2 controller isolation:
  Plain whole-hand MCC (absolute reference)
  Passive-Hold+MCC
  vs. Reactive-Heuristic+MCC
  vs. DPRef+MCC

I06 / Exp. 3 final active-planner isolation (after I05):
  Explicit Planner+MCC
  vs.
  Wrist-only Planner+DPRef+MCC
```

Exp. 2 的 Passive/Reactive/DPRef 三路共享 Wrist MCC、Finger MCC、Role Interpreter 和 M03，
只更换 nominal finger reference source；Plain 是不经过新 wrapper 的普通 MCC 绝对参考。实验已经
完整运行，以接触连续性和多指接触为主指标，力只作持续/多指高力诊断，不设置策略 Gate。
Exp. 3 不属于 E05：它位于 I05 之后，两边共享全部 low-level controllers，只更换 active planner
是否显式优化 finger trajectories/contact modes。

```bash
# 共享低层安全 gate
$PY -m Module.module_4_whole_hand_mcc.g1a_benchmark

# 复用 RAW_VERIFIED Dataset-I 生成 q_nom + temporally confirmed role labels
$PY -m Module.module_4_finger_dp.dpref_dataset

# 双头 CUDA 训练；CUDA 不可见会 fail closed
$PY -m Module.module_4_finger_dp.dpref_train \
  --updates 10000 --batch-size 128 --device cuda:0

# 四策略、三条件、每条 15 s；DPRef inference 使用 CUDA
$PY -m Module.module_4_finger_dp.exp2_benchmark --device cuda:0

# 从冻结 trace 重建 dashboard 和四条视频，不重跑 controller
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  $PY -m Module.module_4_finger_dp.exp2_visual

# 合并 Exp.1/Exp.2 到同一审阅目录和网页
$PY -m Module.e05_strategy_review
```

完整实现与原始数值见
[`evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md`](evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)。
当前统一解释与可视化以
[`generated/e05_exp1_exp2_review/index.html`](generated/e05_exp1_exp2_review/index.html) 为准。

## 当前结果边界

- 结果来自 gravity-off MuJoCo control-isolation，不是硬件或 sim-to-real 结论；
- 旧 Exp. 1 baseline H-MCC 有严重 transient over-force；旧 G1a 只对应 8 N-priority profile；
  当前 Exp.2 采用接触优先 profile，力峰值只作持续/多指高力诊断；
- Dataset-I 训练位移约 28–36 mm/12 s，E05 为 180 mm/15 s + 4 mm step；160 mm matched
  pilot 未通过 raw gate，因此没有被送入训练；
- I100 没有稳定改善 held-out closed-loop，所以不继续盲目扩 I500/I1000；
- 三组首次持续 contact-count 分叉发生在 DP 接管后 `0.048–0.224 s`；长期漂移是后续放大，
  不是第一次失败；
- I20 虽有 30.1% history window 含 contact transition，但当前 `N_c=1/0` 仅为
  `0.438%/0%`，严重 recovery states 仍明显不足；
- I100 DPRef 的 continuous q_nom validation 合格，但 validation/test 都没有 RELEASE，不能声称
  handover generalization；
- Exp. 2 中 DPRef 在严格共享栈三者里的 continuity、平均接触数与 `N_c>=2` supported traversal
  均最好（`0.988/2.450/126.09 mm`），但四指接触率仅 14.86%，第四指参与仍需改善；
- `G1a=ARCHIVED_PRE_RETUNE`；M06–M12 与 Bunny I01–I03 均已完成；I01 得到 `G2=GO`，
  I02=`NOT_MET`、I03=`MET`，因此 `G3=NO_GO`；
- G2 只说明 variable contact 在该固定 Bunny、gravity-off 场景真实突破 fixed-contact 局部
  可行域。I02 未证明冻结的 short-prefix 稳健性改善，所以 GPIS 仍未解锁；E05 策略性能
  本身不承担解锁职责。
