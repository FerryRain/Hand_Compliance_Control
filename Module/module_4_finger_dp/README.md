# M4-DP：DP-direct 归档与 DPRef + shared MCC

当前主方法不是 DP 直接控制关节，而是：

```text
hand/contact history + future wrist plan
 -> DPRef shared encoder
 -> diffusion q_nom chunk + categorical role intention
 -> deterministic Reference/Role Interpreter
 -> shared Finger MCC
 -> LEAP Hand

wrist plan -> shared Wrist MCC -> FR3
```

神经训练和推理固定使用 `handcomp` + CUDA；CUDA 不可见时 fail closed。MuJoCo physics 使用其
自身 backend，不属于神经网络 CPU fallback。

## 当前状态

| 内容 | 状态 | 结论 |
| --- | --- | --- |
| Exp. 1 DP-direct | `EVALUATED` | MCC 的 contact/force 表现更好；DP-direct 保留为架构消融 |
| G1a shared execution | `ARCHIVED_PRE_RETUNE` | 旧 8 N-priority profile 的审计保留；当前接触优先 profile 由 Exp.2 连续指标描述 |
| Dataset-I relabel | `COMPLETE` | I20/I100 生成 measured-q anchored q_nom 与时间确认 role labels |
| DPRef CUDA I100 | `TRAINED / ROLE_COVERAGE_LIMITED` | q_nom 已记录；validation 无 RELEASE，MAKE 20 labels/60% |
| Exp. 2 | `EVALUATED` | Plain 为绝对参考；共享栈三者中 DPRef 的 continuity、平均 contacts 与 supported traversal 最好 |
| I06 / Exp. 3 | `NOT_STARTED` | 位于 I05 后的最终 active-planner ablation，不属于 E05 |

## 文件结构

| 文件 | 用途 |
| --- | --- |
| `contracts.py` | 旧 DP-direct observation shape、单位、validity contract |
| `force_history.py` | 500→100 Hz causal LPF/anti-alias 与 200 ms history |
| `policy.py` | 归档的 DP-direct policy |
| `action_chunk.py` | measured-q anchored relative action chunk |
| `authority_filter.py` | DP-direct authority QP；不在 DPRef 后增加第二个低层 controller |
| `guard_state_machine.py` | M03 shared force-safety executor 兼容入口 |
| `dataset_i_oracle.py` | simulator-only non-MCC forward oracle 与 raw replay |
| `dataset_i_pipeline.py` | RAW/REJECTED、object split 与 causal sample pack |
| `formal_train.py`、`formal_eval.py` | 归档 DP-direct I20/I100 CUDA training/held-out |
| `e05_dp_benchmark.py`、`e05_dp_visual.py` | Exp. 1 DP-direct evaluator 与可视化 |
| `dpref_dataset.py` | 复用 RAW_VERIFIED Dataset-I，生成 q_nom + confirmed role labels |
| `dpref_policy.py` | shared encoder、diffusion trajectory head、separate role head |
| `dpref_train.py` | CUDA-only I100 training、split/coverage/gate audit |
| `dpref_reference_sources.py` | Passive、causal Reactive、CUDA DPRef 三种公平 reference source |
| `exp2_benchmark.py` | Plain + 三种 shared-stack source × 三条件的 Exp. 2 evaluator |
| `dpref_visual.py`、`exp2_visual.py` | 从冻结数据重建 training audit、dashboard 和视频 |

共享执行层在相邻目录：

| 文件 | 用途 |
| --- | --- |
| `../module_4_whole_hand_mcc/reference_interpreter.py` | KEEP/RELEASE/FREE/MAKE FSM、确认与 force ramp |
| `../module_4_whole_hand_mcc/runner.py` | DP nominal tangent + shared Finger/Wrist MCC 物理执行 |
| `../module_4_whole_hand_mcc/g1a_benchmark.py` | shared low-level safety gate |
| `../module_3_runtime_guards/command_continuity.py` | finger/wrist step limiter |
| `../module_3_runtime_guards/force_safety_executor.py` | rapid-load、release、reentry 与 SAFE_HOLD |

## 数据与训练审计

现有 RAW_VERIFIED episodes 可以复用，旧 DP-direct checkpoint 不能复用。role 不是从相邻两帧
contact mask 生硬推导，而是使用 100 ms stable confirmation、300 ms lookahead、force 与
normal-motion evidence；模糊 flicker 被 mask。

| split | episodes | samples | KEEP | RELEASE | FREE | MAKE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I20 train | 20 | 11,180 | 37,276 | 28 | 4,803 | 120 |
| I100 train | 100 | 55,900 | 184,467 | 95 | 26,459 | 913 |
| validation | 4 | 2,236 | 7,640 | 0 | 836 | 20 |
| test | 3 | 1,677 | 6,494 | 0 | 103 | 22 |

I100 在 RTX 4090 D 上完成 10,000 updates（149.28 s）。validation first-command RMSE 为
0.002209 rad，chunk RMSE 为 0.004733 rad；但 validation MAKE accuracy 仅 60%（20 labels），
RELEASE 无样本，所以当前 checkpoint 标记为 `ROLE_COVERAGE_LIMITED`，不能声称 handover
generalization。旧 source summary 中的 `training_status=FAIL` 是历史 gate 字段，不再作为
E05 策略 verdict。

## Exp. 2 结果

Plain whole-hand MCC 是不含新 Role/ForceSafety wrapper 的普通解析绝对参考。Passive、Reactive、
DPRef 三者共享 wrist path、Wrist MCC、resultant/internal coordinator、Finger MCC、Role
Interpreter、M03、robot/object/initial states；这三者只改变 reference source。

| 3 条件 aggregate | Plain | Passive | Reactive | DPRef |
| --- | ---: | ---: | ---: | ---: |
| contact continuity | **0.992** | 0.972 | 0.973 | **0.988†** |
| average contacts | **3.156** | 2.285 | 2.310 | **2.450†** |
| `P(N_c>=2)` | **0.877** | 0.610 | 0.612 | **0.847†** |
| `P(N_c>=3)` | **0.754** | 0.439 | 0.451 | **0.466†** |
| four-contact probability | **0.533** | 0.263 | **0.275†** | 0.149 |
| supported Y (`N_c>=2`) | **138.87 mm** | 89.35 mm | 86.90 mm | **126.09 mm†** |
| worst peak force（诊断） | 11.650 N | 56.774 N | 52.237 N | 4.825 N |
| mean `>8 N` time（诊断） | 0.783 s | 0.092 s | 0.029 s | 0 s |
| multi-pad simultaneous `>8 N` | 0 s | 0 s | 0 s | 0 s |

`†` 只表示 Passive/Reactive/DPRef 严格共享栈子集中的最好结果。DPRef 相对该子集最佳解析源，
continuity 提高 1.51 pp、平均多 0.139 个 contact、supported traversal 多 36.75 mm；但四指同时
接触率比 Reactive 低 12.60 pp，且 role validation coverage 不完整。MuJoCo 力只作诊断：主要看
高力持续时间、多指同时高力与超额冲量，不以单点 peak 判策略失败。

Exp.1/2 的统一网页：
[`../generated/e05_exp1_exp2_review/index.html`](../generated/e05_exp1_exp2_review/index.html)。

## 复现

从仓库根目录运行：

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python

# 共享执行层安全 gate
$PY -m Module.module_4_whole_hand_mcc.g1a_benchmark

# 生成 DPRef label packs
$PY -m Module.module_4_finger_dp.dpref_dataset

# CUDA-only 双头训练
$PY -m Module.module_4_finger_dp.dpref_train \
  --updates 10000 --batch-size 128 --device cuda:0

# 训练与 label audit 图
$PY -m Module.module_4_finger_dp.dpref_visual

# Exp. 2：4 strategies × 3 conditions × 15 s
$PY -m Module.module_4_finger_dp.exp2_benchmark --device cuda:0

# 从保存 trace 重建图与四条视频
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  $PY -m Module.module_4_finger_dp.exp2_visual

# 相关回归
$PY -m unittest Module.tests.test_finger_dp_core \
  Module.tests.test_e05_mcc_full_robot -v
```

Exp. 1 的旧 DP-direct 复现命令仍保留在
[`../DP_CONTROLLER_V1_PROTOCOL.md`](../DP_CONTROLLER_V1_PROTOCOL.md)，不应拿来冒充 DPRef。

## 审阅路径

- 当前接触优先重评证据：
  [`../evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md`](../evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)
- 旧 8 N-priority G1a/Exp.2 provenance：
  [`../evidence/2026-08-24_DPREF_EXP2.md`](../evidence/2026-08-24_DPREF_EXP2.md)
- relabel：
  [`../generated/dpref_v1/relabelled_dataset_i/README.md`](../generated/dpref_v1/relabelled_dataset_i/README.md)
- CUDA training：
  [`../generated/dpref_v1/training_i100/README.md`](../generated/dpref_v1/training_i100/README.md)
- Exp. 2 dashboard：
  [`../generated/exp2_dpref_mcc/exp2_comparison.png`](../generated/exp2_dpref_mcc/exp2_comparison.png)
- 四条视频：
  [`Plain`](../generated/exp2_dpref_mcc/plain_whole_hand_mcc_video.mp4)、
  [`Passive`](../generated/exp2_dpref_mcc/passive_hold_mcc_video.mp4)、
  [`Reactive`](../generated/exp2_dpref_mcc/reactive_heuristic_mcc_video.mp4)、
  [`DPRef`](../generated/exp2_dpref_mcc/dpref_mcc_video.mp4)

## 下一步建议

建议补充 intentional MAKE/RELEASE 的 object-disjoint validation/test episodes并重训，
重点改善 DPRef 的第四指参与率与四指同时接触，而不是针对 MuJoCo 单点峰值过拟合。E05 不负责
解锁 active planner；I04 已独立冻结为 Oracle next-point whole-hand contact traversal，不负责
选择探索点。Exp.3 固定在 I05 后作为 I06 最终消融。
