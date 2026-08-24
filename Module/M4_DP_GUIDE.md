# M4-DP 实施指导：先验证学习链，再扩展 Verified-Inverse 数据

> 状态：`ACTIVE_GUIDE / DATASET-D_SMOKE_NEXT / DATASET-I_GENERATOR_IN_DEVELOPMENT`
> 日期：`2026-08-23`
> 环境：`handcomp`
> 控制架构以 [`DP_CONTROLLER_V1_PROTOCOL.md`](DP_CONTROLLER_V1_PROTOCOL.md) 为准；本文件负责实施顺序、数据门禁和实验解释。
> 当前禁止：直接批量采集 Dataset-I、正式 DP 训练、E05-DP 评测、隐藏 Finger MCC fallback。

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

该 episode 的 forward teacher 使用 Fingertip MCC，因此只属于 `Dataset-D diagnostic`，不是
论文正式 Dataset-I。

## 2. 总实施顺序

```text
Track D: 1-4 Dataset-D episodes -> intentional overfit -> closed-loop imitation
                              \
                               -> both gates pass -> scale Dataset-I -> formal training
                              /
Track I: non-MCC forward oracle -> 20-episode pilot -> RAW_VERIFIED pool
```

两条 track 可以交替开发，但在二者各自通过 Gate 之前，不得开始正式扩量或 E05-DP。

## 3. Track D：先验证 DP 学习与执行链

### 3.1 目的

Dataset-D 不承担论文 teacher contribution，只回答：

> Force-history Finger DP 的实现能否模仿一个已知有效 controller？

第一轮只使用 `1–4` 个成功 episode，目标是故意过拟合，而不是泛化。

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

### 4.3 Scalable contribution 的必要对照

Privileged oracle 可以使用，但必须证明容易的 forward process 比直接求解目标任务更容易。
至少比较：

| Generator | Success rate | Solver time | Replay/repair rate |
| --- | ---: | ---: | ---: |
| Direct fixed-object hand exploration | 报告 | 报告 | 报告 |
| Easy moving-object forward generation | 报告 | 报告 | 报告 |
| Forward → raw spatial inverse | 报告 | transformation time | 报告 |

如果 forward oracle 本身已经是能轻松解决最终固定物体探索的超级控制器，则不能声称 inverse
generation 更 scalable。

## 5. Dataset-I engineering pilot

### 5.1 规模与 diversity

第一轮只生成 `20` 个 episode。建议使用 `5 objects × 4 motion seeds`，而不是同一对象重复
20 次。物体必须足够大，允许 LEAP Hand 在表面形成真实多指接触；具体对象和尺寸在采集前
冻结。

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

## 6. 数据池与 split

每个 episode 只能属于一个 pool：

```text
RAW_VERIFIED
REPAIRED
REJECTED_DIAGNOSTIC
```

训练、验证、测试必须至少按完整 episode 切分；禁止 random frame split。正式 generalization
还必须按 object 切分：

```text
train objects
seen-object / unseen trajectories
completely unseen objects
```

同一物理 episode 的相邻帧、chunk 和 replay variant 不得跨 split。

## 7. 通过 learning curve 决定数据规模

Dataset-I 按以下规模递增：

```text
20 -> 100 -> 500 -> 1000+
```

每阶段保持模型、训练预算定义、held-out split 和 evaluator 一致，报告：

- held-out nonempty-contact continuity；
- contact richness/retention；
- force violation 与 soft-limit exposure；
- action error 与 authority intervention；
- seen/unseen-object success；
- generator success、solver time 和 replay acceptance rate。

只有当 `N -> 2N` 的性能增益已经趋于饱和，才认为数据量接近足够。禁止先拍脑袋决定
“完整数据集”大小。

## 8. 正式训练与评测顺序

```text
Dataset-D intentional overfit + closed-loop diagnostic
 -> Dataset-I held-out replay validation
 -> E05-H-MCC vs. E05-H-DP primary
 -> optional E05-F-DP standalone diagnostic
```

`E05-H-DP` 与 `E05-H-MCC` 必须共享 Wrist MCC、wrist reference、initial states、guards、limits、
objects 和 evaluation horizon。唯一替换是 Finger MCC 与 Finger DP + Action Authority Filter。

`E05-F-DP` 不解锁主方法，也不作为 E05-H-DP 的强制前置 gate；它只回答 DP 在没有 Wrist MCC
时的 standalone capability。

## 9. Go/No-Go Gates

| Gate | 通过条件 | 未通过时动作 |
| --- | --- | --- |
| D-Gate | 1–4 Dataset-D episode 可过拟合，并能 closed-loop physical imitation | 修 DP pipeline，不扩 Dataset-I |
| I-Generator Gate | non-MCC forward oracle 在 20-episode pilot 稳定生成 valid forward data | 修 oracle/object/motion distribution |
| I-Raw Gate | RAW_VERIFIED 满足 whole-hand contact、force、安全与 provenance 门禁 | 不训练；raw/repaired/rejected 分池 |
| Scale Gate | 100/500/1000+ learning curve 显示稳定 held-out 增益 | 继续分批采集或修分布 |
| E05 Gate | held-out Dataset-I 验证通过且训练/eval split 冻结 | 禁止 E05-H-DP 主比较 |

## 10. 当前下一步

现在立即推进两项，但不做正式扩量：

1. `Track D`：准备 1–4 个 Dataset-D episode，完成 overfit 与 closed-loop imitation smoke test；
2. `Track I`：实现 non-MCC forward oracle，完成 20-episode pilot 和 RAW_VERIFIED 分类。

只有 D-Gate、I-Generator Gate 与 I-Raw Gate 全部通过，才开始 100→500→1000+ 的 Dataset-I
扩量和正式训练。
