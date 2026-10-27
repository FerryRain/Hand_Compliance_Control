# Exp. 2 接触优先重评证据

日期：`2026-08-25`
环境：`handcomp`；DPRef 推理使用 `cuda:0 / NVIDIA GeForce RTX 4090 D`。
状态：`EVALUATED`。本实验只描述各策略性能，不设置策略 `PASS/FAIL`。

## 1. 为什么重评

旧 shared stack 以 `8 N` 峰值限制为优先目标，MAKE/recontact 过于保守，Passive、Reactive 和
DPRef 都落在较低的多接触水平。当前协议改为：

- 主指标是 contact continuity、平均接触数、`P(N_c>=2/3/4)` 和带多指接触的移动距离；
- MuJoCo fingertip force 只作诊断，不以单个瞬时峰值否定策略；
- 力侧重点是持续高力、多指同时高力、超额冲量和明显 penetration；
- `8 N` 保留为统一观测参考线，不是策略门槛。

## 2. 调整后的共享执行层

本轮没有修改 DPRef checkpoint，也没有按评测结果重新训练。调整只发生在所有 shared-stack
策略共同使用的执行层：

1. MAKE/recontact 使用独立的低力 acquisition MCC；
2. 首次测到接触、等待 30 ms 确认时仍保留 50% approach，而不是立即冻结；
3. `MAKE -> KEEP` 的目标力 ramp 从真实 measured load 开始；
4. acquisition MCC 的 offset/velocity 状态转交给 KEEP MCC，避免切换时丢失接触；
5. soft-force 事件不再主动 release；只有广泛或严重事件才触发 deterministic hard handling；
6. evaluator 增加最长连续高力、多指同时高力、超额冲量和 `>20 N` 时间。

## 3. 公平比较契约

四种策略均使用同一 FR3+LEAP 场景、15 s wrist path，以及 nominal、low-friction、
noisy-observation 三个配对条件。

- `Plain whole-hand MCC`：旧的普通解析绝对参考，不经过新 Role/ForceSafety wrapper；
- `Passive-Hold + MCC`：共享执行栈，nominal finger reference 保持不动；
- `Reactive-Heuristic + MCC`：同一共享栈，只使用当前/过去状态的因果启发式；
- `DPRef + MCC`：同一共享栈，使用 future wrist plan 生成 nominal chunk/role intention。

因此 Plain 只用于显示绝对基础性能；reference-source 的严格因果比较只在
Passive/Reactive/DPRef 三者内部进行。

## 4. 三条件 aggregate

| 指标 | Plain | Passive | Reactive | DPRef |
| --- | ---: | ---: | ---: | ---: |
| contact continuity | **99.21%** | 97.23% | 97.26% | **98.77%** |
| 平均 contact 数 | **3.156** | 2.285 | 2.310 | **2.450** |
| `P(N_c>=2)` | **87.69%** | 61.02% | 61.23% | **84.70%** |
| `P(N_c>=3)` | **75.37%** | 43.90% | 45.08% | **46.64%** |
| `P(N_c=4)` | **53.30%** | 26.31% | **27.45%** | 14.86% |
| total Y traversal | **174.36 mm** | 166.81 mm | 170.53 mm | **173.84 mm** |
| supported Y, `N_c>=2` | **138.87 mm** | 89.35 mm | 86.90 mm | **126.09 mm** |
| force RMSE（诊断） | 1.910 N | 1.855 N | 1.752 N | 1.570 N |
| worst peak（诊断） | 11.65 N | 56.77 N | 52.24 N | 4.83 N |
| 平均 `>8 N` 时间 | 0.783 s | 0.092 s | 0.029 s | 0 s |
| 平均最长连续 `>8 N` | 0.477 s | 0.043 s | 0.029 s | 0 s |
| 多指同时 `>8 N` | 0 s | 0 s | 0 s | 0 s |
| 平均 `>20 N` 时间 | 0 s | 0.0147 s | 0.0113 s | 0 s |
| `>8 N` 超额冲量 | 0.632 N s | 0.483 N s | 0.381 N s | 0 N s |

## 5. 性能解读

- 相对旧 8 N-priority profile，Passive 的 continuity/平均 contacts/supported distance 分别提高
  `+9.73 pp/+0.774/+37.59 mm`，Reactive 提高 `+11.36 pp/+0.863/+37.56 mm`，DPRef 提高
  `+11.07 pp/+1.031/+79.96 mm`。这确认旧结果偏低主要来自 shared acquisition/transition stack，
  不能归因于 Passive source 本身。
- **Plain MCC** 给出最高 contact continuity、平均接触数和多指接触丰富度，是当前绝对接触
  保持参考；但它是不同执行栈，不能把其优势归因于 reference source。
- **Passive-Hold** 是不带预测的 shared-stack 下限；相对 Plain 平均少 0.871 个接触，
  `N_c>=2` supported distance 少 49.52 mm。
- **Reactive-Heuristic** 相对 Passive 只带来很小变化：平均多 0.026 个接触、总移动多
  3.73 mm，但 `N_c>=2` supported distance 反而少 2.45 mm。
- **DPRef** 在严格共享栈三者中，continuity、平均接触数、`P(N_c>=2/3)` 和 supported distance
  均最好；相对最佳解析 reference source分别提高 1.51 pp、0.139 个接触、23.47 pp、
  1.56 pp 和 36.75 mm。
- DPRef 的主要不足是 `P(N_c=4)` 比 Reactive 低 12.60 pp，第四指 contact probability 仅
  19.19%；当前应优先补足第四指参与和 intentional MAKE/RELEASE 数据，而不是针对单点
  MuJoCo 峰值继续收紧控制器。
- 四种策略均未出现多指同时 `>8 N`。Passive/Reactive 的 50 N 级单指峰值持续时间很短，但
  仍应作为接触切换/penetration 诊断保留，不能解释成真实硬件力值。

## 6. 审阅与复现

统一网页：
[`generated/e05_exp1_exp2_review/index.html`](../generated/e05_exp1_exp2_review/index.html)

Exp. 2 原始结果：

- [`summary.json`](../generated/exp2_dpref_mcc/summary.json)
- [`exp2_comparison.png`](../generated/exp2_dpref_mcc/exp2_comparison.png)
- [`Plain`](../generated/exp2_dpref_mcc/plain_whole_hand_mcc_video.mp4)
- [`Passive`](../generated/exp2_dpref_mcc/passive_hold_mcc_video.mp4)
- [`Reactive`](../generated/exp2_dpref_mcc/reactive_heuristic_mcc_video.mp4)
- [`DPRef`](../generated/exp2_dpref_mcc/dpref_mcc_video.mp4)

从仓库根目录复现：

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m Module.module_4_finger_dp.exp2_benchmark --device cuda:0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  $PY -m Module.module_4_finger_dp.exp2_visual
$PY -m Module.e05_strategy_review
```

旧 [`2026-08-24_DPREF_EXP2.md`](2026-08-24_DPREF_EXP2.md) 只保留为 8 N-priority profile 的历史
provenance，不代表当前接触优先配置和当前 Exp. 2 数值。
