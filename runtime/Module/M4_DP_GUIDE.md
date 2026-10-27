# M4-DP 实施指导：Verified-Inverse 数据与 DP Reference Generator

> 状态：`DP_DIRECT_ARCHIVED / DPREF_IMPLEMENTED / EXP1_EXP2_EVALUATED`
> 日期：`2026-08-25`
> 环境：`handcomp`
> DP-direct 复现以 [`DP_CONTROLLER_V1_PROTOCOL.md`](DP_CONTROLLER_V1_PROTOCOL.md) 为准；
> 新主方法以 [`DP_REFERENCE_GENERATOR_DESIGN.md`](DP_REFERENCE_GENERATOR_DESIGN.md) 为准。
> 当前结论：DP-direct 只作为 Exp. 1 消融；DPRef 已完成重标注、CUDA I100 训练与 Exp. 2。
> 旧 `G1a=PASS` 已归档为 8 N-priority pre-retune provenance；当前 Exp.2 使用接触优先 profile。
> E05 策略只做描述性评测。role validation coverage 受限，Exp. 3 位于 I05 后。

## 当前架构修正

DP 从 low-level finger controller 上移为 nominal multi-finger trajectory/reference generator：

```text
old ablation : Wrist MCC + Finger DP-direct
new main     : Wrist MCC + (Finger DPRef -> shared Finger MCC)
```

新的实验顺序为：

1. Exp. 1 `H-MCC vs. H-DP-direct`：已完成的低层替代消融；
2. Exp. 2 `Plain / Passive-Hold / Reactive-Heuristic / DPRef+MCC`：已完成接触优先描述性重评；
3. Exp. 3 `Explicit Planner+MCC vs. Wrist-only Planner+DPRef+MCC`：移至 I05 后的 I06 执行。

现有 Dataset-I 已完成 nominal-reference/role 重标注，旧 I20 checkpoint 没有复用；旧
MCC/M03 G1a 审计只对应 pre-retune profile，当前公平性来自后三种策略完全共享同一执行栈。

## 0. 按原章节执行：每一部分具体做什么

本文件真正的实施工作从第 3 部分开始。下面的编号与后文章节完全一致，不再引入另一套步骤
编号。某一部分未通过，就只修这一部分，不得用更多数据、replay repair 或 MCC fallback
掩盖问题。

| 原章节 | 这一部分具体做什么 | 完成时必须留下什么 | 当前状态 |
| --- | --- | --- | --- |
| **1. 当前已经证明什么** | 只确认现有 3 s spatial-inverse 实验的能力边界：它证明物理映射可行，但不是 DP 训练结果，也不是正式 Dataset-I | 当前 forward/replay 指标、provenance 和明确的 `Diag-MCC` 标记 | `DONE` |
| **2. 总实施顺序** | 冻结两条独立工作线：第 3 部分验证 DP learning/execution pipeline，第 4 部分验证 non-MCC generator；D-Gate 和 I-Gate 通过前，不进入 Dataset-I pilot/扩量、正式 Dataset-I training 或 E05 | 固定的执行顺序，以及允许 Diag-MCC diagnostic training、禁止提前正式训练的边界 | `FROZEN` |
| **3. Track D：验证 DP 学习与执行链** | 使用完整 ≥10 s 连续接触 Diag-MCC episode；先做整轨迹 hard gate，再检查 force history、observation、label 和时间戳，最后完成 GPU open-loop overfit 与 `Finger DP-direct + Wrist MCC` closed loop。这里只检查 learnability，不检查正式 Dataset-I generalization | source physical traces、逐 episode acceptance、版本化 causal sample pack、CUDA 审计、checkpoint、10.5 s 闭环视频、force/contact/authority trace，以及 D-Gate verdict | `LONG_GPU_PASS` |
| **4. Track I：实现正式 non-MCC forward generator** | 用 GT geometry、exact contact 和未来 object motion 实现 privileged forward oracle；先跑通单 episode；明确 forward oracle 与 replay repair 的权限差别 | 12 s oracle/replay trace、solve latency、无 Finger MCC provenance 和 I-Gate | `PASS` |
| **5. Dataset-I engineering pilot** | 采集 I-Pilot20；每条执行 zero-repair raw spatial replay，分别进入 `RAW_VERIFIED` 或 `REJECTED_DIAGNOSTIC` | 20 组 verdict；raw/repaired/rejected=`12/0/8` | `RAW-GATE PASS` |
| **6. 数据池与 split** | 冻结 development/train/validation/untouched-test crop；按完整 episode/object 分池，禁止 frame split | object manifest、19 train / 4 validation / 3 test RAW episodes | `PASS` |
| **7. Learning curve 决定规模** | 构建严格 nested I20⊂I100 并用同一验证集训练；不因 I100 open-loop 略好就忽略 closed-loop 退化 | I20/I100 sample pack、CUDA checkpoint、held-out 结果 | `STOP_AT_I20` |
| **8. 正式训练与评测** | 归档 Exp. 1；以 Plain MCC 为绝对参考，比较 Passive-Hold、Reactive-Heuristic 与 DPRef+MCC | Exp. 1/2 证据、CUDA checkpoint、trace、视频与指标 | `EXP1/EXP2_EVALUATED` |
| **9. 数据/执行 Gate 与 E05 描述性结果** | 汇总第 3–8 部分；D/I/Raw/Scale 保证数据流程有效；旧 G1a 单独归档；E05 策略不设 Pass/Fail | 数据 Gate、Exp.1/2 接触指标和持续/多指高力诊断 | `EXP1_EXP2_EVALUATED` |
| **10. 当前下一步** | 补 intentional role validation/test 数据，重点改善 DPRef 第四指参与与四指接触 | role-complete split、CUDA retrain、同协议 Exp. 2 rerun | `FOLLOW_UP_IDENTIFIED` |

总体关系为：

```text
第 3 部分：Diag-MCC 学习链验证 ──┐
                                  ├─> 第 5 部分：20-episode Dataset-I pilot
第 4 部分：non-MCC generator ─────┘
                                      -> 第 6 部分：分池与 split
                                      -> 第 7 部分：逐级扩量
                                      -> 第 8 部分：Exp. 1 归档与 Exp. 2 DPRef+MCC
```

## 1. 当前已经证明什么

当前 3 s 物理实验已经完成：

```text
moving-object forward physical interaction
 -> spatial inversion (same time order)
 -> same recorded finger command
 -> fixed-object FR3+LEAP physical replay
 -> fresh force/contact measurement
```

Replay 没有 Finger IK、Finger MCC 或 force repair，并得到：

| 指标 | Forward | Replay |
| --- | ---: | ---: |
| nonempty-contact continuity | 100% | 100% |
| average contact count | 3.719 | 3.142 |
| maximum fingertip force | 2.693 N | 3.859 N |
| zero-contact time | — | 0 s |
| non-tip contact frames | — | 0 |

此外，`q_cmd_replay[t] == q_cmd_forward[t]`，映射残差为 0；SE(3) residual 约
`1.12e-16`。这证明 spatial inversion 的核心物理假设在一个 episode 上成立。当前问题已从
“inverse 是否成立”转为：

1. DP pipeline 能否学习有效 demonstration；
2. 如何规模化生成不依赖最终 MCC baseline 的 forward demonstrations。

该 episode 的 forward teacher 使用 Fingertip MCC，因此只属于 `Diag-MCC`，不是
论文正式 Dataset-I。

## 2. 总实施顺序

```text
Track D: 1-4 Diag-MCC episodes -> overfit -> closed-loop imitation -> D-Gate
                                                                           \
                                                                            -> both pass
                                                                           /      |
Track I: non-MCC forward oracle -> single physical episode -> I-Gate              v
                                                               I-Pilot20 -> Raw-Gate
                                                                    -> nested scaling
                                                                    -> formal training
```

两条 track 可以交替开发。Diag-MCC diagnostic training 是 Track D 的必要步骤；但在 D-Gate 与
单 episode I-Gate 通过之前，不得开始 Dataset-I I-Pilot20、正式 Dataset-I training 或 E05-DP。

## 3. Track D：先验证 DP 学习与执行链

### 3.1 目的

Diag-MCC 不承担论文 teacher contribution，只回答：

> Force-history Finger DP-direct 的实现能否模仿一个已知有效 controller？

第一轮使用少量完整长 episode，目标是验证 pipeline 与有限 held-out trajectory capability，
不是 Dataset-I/object-level generalization。

> **D-Gate checks controller/pipeline learnability, not generalization.**

### 3.2 必须同时验证的链路

- 500 Hz raw force 的 causal LPF/anti-alias 与 100 Hz history timestamps；
- 200 ms force/contact/validity window 没有未来泄漏或一帧错位；
- observation 中 `q/dq`、contact geometry、target force、wrist plan/real/MCC state 对齐；
- label 是 future issued command chunk，不是 future measured state；
- 每个 chunk 以当前 measured `q` 重新锚定；
- chunk seam、500 Hz interpolation、rate/acceleration limit 连续；
- authority filter 只投影 collective component，不变成隐藏 controller；
- hard guard 接管帧不进入普通 imitation label。

### 3.3 两级 smoke test

#### D1 — Open-loop overfit

必须报告：

- diffusion/train loss；
- command RMSE / maximum error；
- chunk 首帧 seam error；
- authority intervention probability/norm；
- causal timestamp audit。

仅有低 train loss 不算通过。

#### D2 — Closed-loop physical imitation

从 demonstration 的真实初始状态启动，执行 DP，而不是回放 teacher command。报告：

- nonempty-contact continuity 与 zero-contact time；
- average `N_c(t)` 与 contact-time integral；
- force RMSE、maximum、P95、soft-limit duration；
- trajectory/action deviation；
- authority QP failure/intervention；
- hard-guard takeover 与完整 recovery timeline。

如果 1 个 episode 都不能闭环复现，应先检查 observation/label/anchor/interpolation/TCN/filter，
不能归因于 Dataset-I 数据量。

### 3.4 长轨迹 hard gate（冻结）

每条源 episode 为 12 s；接触建立后必须连续保持至少 10 s。完整 episode 还必须满足
`max force < 8 N`、`non-tip=0`、`guard violation=0`。任何一次 whole-hand zero-contact 都让整条
进入 `REJECTED_DIAGNOSTIC`，不得截取失败前的短窗口。只有 `ACCEPTED_TRAIN` 可以贡献样本；
`ACCEPTED_EVAL` 永远不进入训练。

### 3.5 本轮执行结果（2026-08-24）

三个 TRAIN episode 与一个独立 EVAL episode 的 post-contact 连续时长分别为：

```text
TRAIN: 11.646 / 11.560 / 11.628 s
EVAL : 11.662 s
zero-contact / non-tip / guard violations: all 0
maximum force: 6.018 / 4.770 / 5.597 / 5.387 N
```

前三条共生成 `1653` 个 50 Hz causal anchors。每个 anchor 使用 100 Hz × 200 ms force
history 和 50 Hz × 400 ms future command chunk；future leakage 为 0，measured-q anchor
residual 为 `1.39e-17 rad`。

D1 在 `NVIDIA GeForce RTX 4090 D` 上训练 8000 updates：

```text
full-chunk RMSE = 0.002507 rad
first-command RMSE = 0.001565 rad
GPU open-loop inference P95 = 0.405 ms/sample
```

D2 前 1 s 为显式 teacher contact initialization；之后 10.5 s 完全撤销 Finger MCC，只保留
Wrist MCC、Finger DP-direct、incremental Action Authority Filter 和 hard guard：

```text
contact continuity / zero-contact = 1.000 / 0.000 s
average / minimum contacts = 3.998 / 3
maximum / P95 force = 5.615 / 2.205 N
authority solver failures / hard guard / non-tip = 0 / 0 / 0
GPU policy latency P95 = 11.842 ms (<20 ms)
D-Gate = PASS, blocking_reason = NONE
```

Authority QP 限制的是 `q_cmd,new - q_cmd,previous` 的新增 collective motion；已有 position
target preload 不被误当成每 tick 新动作。每接触 collective command velocity 上限为
`10 mm/s`。Opposition rate/energy `0.541 / 1.93e-5` 保留为诊断，不是安全 gate。

该 PASS 仍只证明 Diag-MCC learning/execution pipeline，不是正式 Dataset-I、object-level
generalization 或 E05 论文结论。审阅入口：
[`generated/whole_hand_dp_long_v1/review.html`](generated/whole_hand_dp_long_v1/review.html)。

一键复现：

```bash
MUJOCO_GL=osmesa /home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.module_4_finger_dp.long_gpu_demo --updates 8000 --require-pass
```

## 4. Track I：正式 non-MCC forward generator

### 4.1 Forward oracle 的权限

正式 Dataset-I 的 forward teacher 是 simulator-only privileged demonstration generator：

```text
GT geometry
+ exact contact/normal/friction/state
+ known future forward object trajectory
 -> offline/privileged finger trajectory optimization
 -> q_f_cmd_forward[0:T]
```

可优化：

```text
J = J_contact + J_force + J_joint + J_collision + J_smooth
```

该 oracle 可以在所有 forward demonstration 中存在；部署时完全不存在。它不能直接复用
最终 E05-H-MCC 的 Finger MCC branch，否则主比较会退化成 baseline distillation。

### 4.2 Forward oracle 与 replay repair 不是同一件事

| 概念 | 修改什么 | 允许比例 | 数据含义 |
| --- | --- | ---: | --- |
| Forward oracle | 生成 `q_cmd_forward` | 可为 100% | demonstration generator |
| Raw spatial replay | 使用相同 `q_cmd_forward[t]` | repair 必须为 0% | `RAW_VERIFIED` candidate |
| Replay repair oracle | replay 失败后修改 `q_cmd_replay` | 单独统计 | `REPAIRED` pool |

Replay repair rate 定义为：

```text
r_replay_repair = modified replay frames / all replay frames
```

第一版 DP 主训练只允许 `RAW_VERIFIED`。`REPAIRED` 独立保存，后续仅用于
`RAW` vs. `RAW + REPAIRED` ablation；不得静默混入 raw pool。

### 4.3 后续 contribution validation：不阻塞单 episode I-Gate

Privileged oracle 可以使用，但必须证明容易的 forward process 比直接求解目标任务更容易。
这个论文对照必须在 I-Pilot20 或 scaling 阶段补齐，但不要求为了通过单 episode I-Gate 而先实现
完整 direct target-task generator。后续至少比较：

| Generator | Success rate | Solver time | Replay/repair rate |
| --- | ---: | ---: | ---: |
| Direct fixed-object hand exploration | 报告 | 报告 | 报告 |
| Easy moving-object forward generation | 报告 | 报告 | 报告 |
| Forward → raw spatial inverse | 报告 | transformation time | 报告 |

如果 forward oracle 本身已经是能轻松解决最终固定物体探索的超级控制器，则不能声称 inverse
generation 更 scalable。

### 4.4 I-Gate 实际结果

当前 oracle 为 `SIM_PRIVILEGED_GT_HORIZON_IK_NON_MCC_V1`：使用 GT surface、exact force
objective 和 5-step future object horizon；`mcc_calls=0`。单条 12 s 结果：

| 指标 | Forward | Raw replay |
| --- | ---: | ---: |
| nonempty-contact continuity | 1.000 | 1.000 |
| average contacts | 3.950 | 2.694 |
| maximum force | 6.412 N | 4.647 N |
| non-tip frames | 0 | 0 |

command mapping residual 与 replay repair rate 均为 0；oracle latency mean/P95 为
`6.74/7.05 ms`。因此 `I-Gate=PASS`。原始证据位于
`generated/finger_dp_formal_v1/track_i/igate_dev_crop_0/`。

## 5. Dataset-I engineering pilot

### 5.1 规模与 diversity

第一轮只生成 `20` 个 episode，明确标记为 `PILOT_V1`。建议使用
`5 objects × 4 motion seeds`，而不是同一对象重复
20 次。物体必须足够大，允许 LEAP Hand 在表面形成真实多指接触；具体对象和尺寸在采集前
冻结。

Pilot 用于验证 generator statistics 与 raw replay acceptance，不直接等同于正式 scaling 的
`I20`。Generator 冻结后应重新建立或显式版本化正式 `I20`；pilot 数据不得进入 untouched
test，是否进入 training 必须在 split manifest 中明确记录。

每个 episode 必须保存完整 forward 与 replay causal logs、oracle solve statistics、source
provenance、raw/repaired/rejected classification 和 termination reason。

### 5.2 Raw replay 硬门禁

#### Whole-hand contact

定义：

```text
N_c(t) = sum_i c_i(t)
```

要求 hysteresis 后近乎全过程 `N_c(t) >= 1`；优先报告精确 zero-contact time。若允许数值
dropout，容差必须在运行前冻结。

#### Force

必须同时报告：

- `f_max < 8 N`；
- per-finger/all-contact force P95；
- `T(f > f_soft)`；
- force RMSE 与 transient peaks。

#### Safety

Raw pool 必须满足：

```text
collision = 0
non-tip contact = 0
hard-guard takeover = 0
replay repair = 0
```

因此 `RAW_VERIFIED` 必须同时满足：

```text
q_f_cmd_replay(t) == q_f_cmd_forward(t)
modified replay frames = 0
```

任何 frame 的 replay finger command 被 oracle 修改后，该 episode 都不能再计算 raw acceptance，
只能进入独立 `REPAIRED` pool。不满足安全/接触门禁且未形成合格 repair 的 episode 进入
`REJECTED_DIAGNOSTIC`。Pilot 必须分别报告：

```text
R_raw
R_repaired
R_rejected
```

#### Provenance

必须逐样本确认：

```text
q_f_cmd_replay(t) == q_f_cmd_forward(t)
time_mapping = SAME_T_FORWARD_ORDER
inversion_mode = SPATIAL_ONLY
force/contact source = FRESH_REPLAY_MEASUREMENT
```

### 5.3 Contact richness：primary diagnostic

逐指 contact-mask agreement 保留为 diagnostic，但不作为硬门禁。Spatial inversion 不要求
每根手指在完全相同时间接触；安全 handover 或不同但合法的 contact set 仍可能构成有效数据。

主要 richness 指标为：

```text
R_contact = sum_t N_c_replay(t) / sum_t N_c_forward(t)
```

当前 episode 约为 `3.142 / 3.719 = 0.845`。同时报告 average/minimum contact count、contact
switches、zero-contact time 和逐指 contact probability，避免只用一个 ratio 隐藏 failure。

### 5.4 I-Pilot20 实际结果

当前 I-Pilot20 的历史兼容目录为 `generated/finger_dp_formal_v1/pilot20/`：

```text
RAW_VERIFIED / REPAIRED / REJECTED = 12 / 0 / 8
raw acceptance rate = 0.60
```

未通过 episode 不贡献任何训练 sample；没有把单帧 over-force 或 contact dropout 前的短窗口
截进训练集。

## 6. 数据池与 split

每个 episode 只能属于一个 pool：

```text
RAW_VERIFIED
REPAIRED
REJECTED_DIAGNOSTIC
```

在大规模 generation 或 tuning 前，先冻结四类 object：

```text
generator-development objects
training objects
validation objects
untouched test objects
```

`untouched test objects` 只能在最终 E05 打开，不能依据其表现继续修改 generator、policy 或
threshold。训练、验证、测试必须至少按完整 episode 切分；禁止 random frame split。正式
generalization 还必须按 object 切分：

```text
train objects
seen-object / unseen trajectories
completely unseen objects
```

同一物理 episode 的相邻帧、chunk 和 replay variant 不得跨 split。

### 6.1 当前冻结 split

对象集合在批量生成前写入 `formal_pool_v1/object_split_manifest.json`。当前 RAW pool 为
`19 train / 4 validation / 3 untouched test` episodes；train/validation/test 的 object id 和
episode id 均不相交。训练文件只包含 `RAW_VERIFIED`，repair policy 固定为 `NONE`。

## 7. 通过 learning curve 决定数据规模

Dataset-I 按以下规模递增：

```text
20 -> 100 -> 500 -> 1000+
```

这里的 `20` 是 generator 冻结后的正式 Scaling-20，不是第 5 部分允许反复修改 generator 的
I-Pilot20。正式 scaling dataset 必须 nested：

```text
I20 subset I100 subset I500 subset I1000+
```

每阶段保持 architecture、training compute 定义、validation/test split 和 evaluator 一致，报告：

- held-out nonempty-contact continuity；
- contact richness/retention；
- force violation 与 soft-limit exposure；
- action error 与 authority intervention；
- seen/unseen-object success；
- generator success、solver time 和 replay acceptance rate。

只有当 `N -> 2N` 的性能增益已经趋于饱和，才认为数据量接近足够。禁止先拍脑袋决定
“完整数据集”大小。

### 7.1 当前 I20/I100 scaling 结论

`I20` 含 20 条 episode、11180 个 causal anchors；`I100` 严格包含 I20，含 100 条 episode、
55900 anchors，其中 15 条为已通过 raw gate 的物理 disturbance episodes。两个模型都在 RTX
4090 D 上训练 10000 updates：

| 模型 | validation first/full RMSE | held-out continuity | avg contacts | peak force | authority intervention |
| --- | ---: | ---: | ---: | ---: | ---: |
| I20 | 0.00221 / 0.00465 rad | 1.000 | 2.664 | 5.957 N | 56.6% |
| I100 | 0.00213 / 0.00427 rad | 1.000 | 2.543 | 7.496 N | 65.1% |

I100 只在 open-loop RMSE 上略好，closed-loop contact richness、peak force 与 filter dependence
反而退化。因此没有把“数据更多”写成 Scale-Gate 已通过；Exp. 1 checkpoint 选择 I20。

## 8. 正式训练与评测顺序

```text
Diag-MCC intentional overfit + closed-loop diagnostic
 -> Dataset-I held-out replay validation
 -> Exp. 1 E05-H-MCC vs. E05-H-DP-direct (architecture ablation, complete)
 -> repair shared MCC/M03 safety and command continuity
 -> relabel/audit Dataset-I for nominal reference + contact role
 -> Exp. 2 Passive-Hold / Reactive-Heuristic / DPRef+MCC
```

Exp. 1 的 `E05-H-DP-direct` 与 `E05-H-MCC` 共享 Wrist MCC、wrist reference、initial states、
guards、limits、objects 和 evaluation horizon；唯一替换是 Finger MCC 与 Finger DP-direct。

Exp. 2 的三个分支必须进一步共享 **同一个 Finger MCC** 与 Role Interpreter；唯一替换
passive、causal reactive 或 learned anticipatory nominal-reference source。Reactive branch 不得
读取 future wrist plan。`E05-F-DP` 继续只作为可选 standalone diagnostic，不解锁主方法。

### 8.1 Exp. 1 DP-direct 正式配对结果

三组 15 s episode（nominal、low-friction、noisy-observation）均完整执行，前 1 s 使用相同且
不计分的 contact initializer。随后唯一替换是 Finger MCC 与 Finger DP-direct + Authority Filter；
Wrist MCC、M03 force-safety executor、wrist path、扰动、初态和限制完全共享。

| aggregate mean | E05-H-MCC | E05-H-DP-direct |
| --- | ---: | ---: |
| contact continuity | 0.873 | 0.667 |
| average contacts | 3.026 | 1.590 |
| force RMSE | 1.381 N | 2.232 N |
| worst peak across episodes | 81.35 N | 103.02 N |
| mean Y traversal | 174.2 mm | 158.2 mm |
| controller P95 | 1.35 ms | 12.00 ms |

两边均已 `EVALUATED`。MCC 的 contact continuity、平均接触数、force RMSE、traversal 和延迟
更好；DP-direct 的 wrist force-z RMSE 更低且 CUDA latency 满足 20 ms 周期。相对 8 N 参考
限制，worst peak 分别超出 `73.35/95.02 N`。E05-speed matched inverse pilot（160 mm/12 s、
4 mm step）也因 forward continuity `0.9735`、forward/replay peak `34.66/9.73 N` 被正确拒绝，
说明正式数据只覆盖约 28–36 mm/12 s 是当前主要分布缺口之一。审阅入口：
`generated/e05_h_mcc_vs_dp/review.html`。

### 8.2 冻结 trace failure diagnostic

诊断入口为 `generated/e05_h_mcc_vs_dp/diagnostics/review.html`。采用持续 50 ms 的
`N_c^DP < N_c^MCC` 定义，nominal/low-friction/noisy 的首次分叉为：

```text
1.224 / 1.148 / 1.048 s
= DP 接管后 224 / 148 / 48 ms
```

因此长期误差会继续放大失败，但不是第一次分叉的起点。三组最近一次 policy replan 的 exact
intervention norm 都为 0；pre-divergence window 无 QP failure。Authority Filter 可能影响后续
轨迹，但不能解释这三次最初掉指。v1 trace 没保存完整 pre-projection action，所以报告只给
明确标注的 bounded `r_AF_safe_proxy`；下一版必须同时记录 nominal/safe vectors 才计算正式
`r_AF`。

I20 不是完全没有 contact transition：30.063% 的 200 ms history window 含 gain/loss；但当前
`N_c<=2/1/0` 仅占 `8.014%/0.438%/0%`，说明 severe recovery basin 确实覆盖不足。

Raw peak 也不是纯单 tick artefact：六个 cell/episode 的最长 `F>8 N` 连续段为 8–50 ms。
low-friction MCC 的 81.35 N peak 发生在 `BUFFER_FILL -> ACTIVE`，同时 palm/finger command target
跳变约 21.65 mm/0.030 rad；noisy MCC 同类跳变约 12.99 mm/0.031 rad。nominal/noisy DP 的
hard-release 切换也出现约 32–35 mm palm target reset。因而 shared M03 re-entry/hold command
continuity 是独立于 DP 的优先 blocker。

Nominal 总 Y traversal 几乎相同，但只累计相邻两帧均为 `N_c>=2` 的 positive-Y motion 时，
MCC/DP 占各自 positive motion 的 `76.0%/41.8%`。因此 DP traversal 数字不能被解释成稳定
接触下的等价进度；接触退化先发生，低接触 motion 与 guard intervention 随后改变 traversal。

### 8.3 Exp. 2 Plain + Passive / Reactive / DPRef+MCC（接触优先重评已执行）

```text
E05-H-PlainMCC = ordinary whole-hand MCC absolute reference
E05-H-PassiveMCC = passive/hold nominal reference -> shared Finger MCC
E05-H-ReactiveMCC = causal reactive reference -> same Finger MCC
E05-H-DPRef+MCC  = learned anticipatory reference -> same Finger MCC
```

Exp. 2 没有复用 DP-direct checkpoint。现有 RAW_VERIFIED I20/I100 已重标注为 measured-q
anchored q_nom chunk 与时间确认的 `KEEP/RELEASE/FREE/MAKE`；flicker、guard takeover 和不确定
窗口用 invalid mask。I100 在 RTX 4090 D 上完成 10,000 updates。旧 G1a 只对应 pre-retune
8 N-priority profile；当前 Passive/Reactive/DPRef 在同一次重评中共享完全一致的执行栈。
continuous q_nom validation 已记录，但 validation 没有 RELEASE，MAKE 仅 20 labels/60%，因此
checkpoint 的 role/handover 适用范围受限。

Exp. 2 的四策略 15 s × 3 conditions 已完整运行。Plain/Passive/Reactive/DPRef 的 continuity 为
`0.992/0.972/0.973/0.988`，average contacts 为 `3.156/2.285/2.310/2.450`，`N_c>=2`
supported Y 为 `138.87/89.35/86.90/126.09 mm`。Plain 是接触保持最好的绝对参考；在严格共享
执行栈的三者中，DPRef 的 continuity、平均接触数、`P(N_c>=2/3)` 和 supported traversal 均最好，
但四指同时接触率仅 `14.86%`，低于 Reactive 的 `27.45%`。MuJoCo force 只作诊断：四种策略
均没有多指同时 `>8 N`，不因单指瞬时 peak 给策略 verdict。role coverage 不完整限制了 handover
解读。详见
[`evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md`](evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)。

统一网页与机器数据：
[`generated/e05_exp1_exp2_review/index.html`](generated/e05_exp1_exp2_review/index.html)。

## 9. 数据/执行 Gate 与 E05 评测语义

每个 Gate 都必须输出 `PASS / FAIL / BLOCKED` 和非空的 `blocking_reason`（`PASS` 时写
`NONE`），不能只给一个笼统的 “DP failed”。

| Gate | 回答的问题/通过条件 | 未通过时动作 |
| --- | --- | --- |
| D-Gate | DP pipeline 本身能否学习并闭环执行：1–4 Diag-MCC episode 可过拟合并完成 physical imitation | 修 DP pipeline；不归因于泛化或 Dataset-I 数据量 |
| I-Gate | non-MCC forward oracle 能否在单 episode 产生合法真实 forward command/force/contact，且 provenance 无 Finger MCC fallback | 修 oracle；direct target-task 对照不阻塞此 Gate |
| Raw-Gate | spatial inverse 后、zero replay repair 的 `RAW_VERIFIED` 是否满足 whole-hand contact、force、安全与 provenance 门禁 | 不训练；严格分开 raw/repaired/rejected |
| Scale-Gate | 冻结 generator 后，nested `I20/I100/I500/I1000+` 是否稳定扩量并带来 held-out 增益 | 继续分批采集或修训练分布，不改 untouched test |
| Exp1 protocol readiness | held-out Dataset-I 与 direct-controller protocol 是否足以形成有效 paired ablation | 只修协议，不对策略性能判 Pass/Fail |
| G1a（历史） | 2026-08-24 8 N-priority shared stack 的 safety/readiness 审计 | 只保留 provenance，不外推到当前 contact-priority profile |

Exp.1/2 本身不在表中设置策略 Gate。实验完成后只报告数值、相对优劣、8 N 等参考限制的
超出量，以及数据覆盖限制。

本轮 Gate verdict：

| Gate | Verdict | blocking reason / 解释 |
| --- | --- | --- |
| D-Gate | `PASS` | `NONE` |
| I-Gate | `PASS` | `NONE` |
| Raw-Gate | `PASS` | `NONE`；I-Pilot20 严格分为 12 raw / 0 repaired / 8 rejected |
| Scale-Gate | `STOP_AT_I20` | I100 open-loop 略好，但 held-out physical safety/intervention 退化，不能宣称 scaling gain |
| Exp. 1 authorization | `PASS` | I20 held-out、split、protocol、checkpoint 与 evaluator 均冻结有效 |
| Exp. 1 performance | `EVALUATED` | MCC 接触/力指标更优；峰值相对 8 N 分别超 73.35/95.02 N |
| G1a | `ARCHIVED_PRE_RETUNE` | 旧 8 N-priority profile 的审计保留；当前 Exp.2 不沿用该 verdict |
| DPRef checkpoint | `ROLE_COVERAGE_LIMITED` | validation 缺 RELEASE；MAKE 只有 20 labels，accuracy 60% |
| Exp. 2 performance | `EVALUATED` | Plain 绝对接触最好；严格共享栈三者中 DPRef 的 continuity、平均 contacts 与 supported traversal 最好，但第四指不足 |

## 10. 当前下一步

第 3–8 部分、DPRef 实现和 Exp. 2 都已有实际证据。下一轮建议按以下顺序改进，不盲目扩
I500/I1000：

1. 采集/构建含 intentional MAKE/RELEASE 的 object-disjoint validation/test episodes；不能用
   相邻 contact flicker 伪造 role。
2. 保持 q_nom/role contract 和 frozen test objects 不变，CUDA 重训并通过逐 role coverage。
3. 不针对单点 MuJoCo force peak 过拟合；继续记录持续高力、多指同时高力和超额冲量。
4. 保持 evaluator 不变重跑 Exp. 2，主比较 contact continuity、contact richness 与 supported traversal。
5. planner integration 服从 `MASTER_PLAN.md` 的系统依赖；E05 性能不承担解锁职责。

I04 只做 Oracle 指定 surface goal、但不指定 target finger 的 whole-hand contact traversal；它不
选择探索点。Exp. 3 固定在 I05 后作为 I06 最终消融，不能用 Exp. 2 的固定 wrist 结果或 I04
的 given-good-next-point 结果替代 active planner-level comparison。
