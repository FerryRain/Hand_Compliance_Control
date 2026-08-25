# E05：固定 Wrist 轨迹下的策略性能评测

> 当前状态：`EXP1_EVALUATED / EXP2_EVALUATED`
> 评测语义：只报告性能、相对优劣和参考限制越界；不给策略设置 `PASS/FAIL`、`MET/NOT_MET`，
> 也不以 E05 结果解锁或阻塞后续模块。

统一审阅入口：[`generated/e05_exp1_exp2_review/index.html`](generated/e05_exp1_exp2_review/index.html)。

## 1. E05 的边界

E05 固定：

- wrist trajectory、初始状态、物体、摩擦/噪声条件和评测时长；
- 同一实验内部的 Wrist MCC、guard、执行频率和 actuator limits；
- 只替换 finger controller 或 nominal finger reference source。

因此 E05 回答 controller/reference-source 层的问题，不包含在线 active planning，也不评测
GPIS reconstruction。MuJoCo 接触力只作为诊断指标，`8 N` 是统一参考线而非硬判据。力侧重点为
`>8 N` 占用时间、最长连续段、多指同时高力和超额冲量；单个数值尖峰不会否决策略。

Exp.1 与 Exp.2 使用了不同版本的 shared MCC/guard。只能在各自实验内部比较，不能把两组之间的
峰值变化直接归因于某个 finger 策略。

## 2. Exp.1：Finger MCC vs. DP-direct

比较：

```text
E05-H-MCC       = shared Wrist MCC + coordinated Finger MCC
E05-H-DP-direct = same Wrist MCC + direct Finger DP + authority filter
```

问题：DP 能否直接替代高频 Finger MCC，承担 low-level contact/compliance control？

| 三条件 aggregate | H-MCC | H-DP-direct | 解读 |
|---|---:|---:|---|
| contact continuity | 87.30% | 66.69% | MCC 高 20.62 个百分点 |
| 平均 contact 数 | 3.026 | 1.590 | MCC 多 1.436 个 contact |
| force RMSE | 1.381 N | 2.232 N | MCC 更低 |
| worst peak force | 81.35 N | 103.02 N | 相对 8 N 分别超 73.35 / 95.02 N |
| mean Y traversal | 174.23 mm | 158.23 mm | MCC 多 16.00 mm |
| controller P95 | 1.35 ms | 12.00 ms | MCC 约快 8.9 倍；DP 仍低于 20 ms policy 周期 |
| wrist force-z RMSE | 2.660 N | 1.568 N | DP-direct 在该单项更低 |

性能判断：

- MCC 的优势是多接触保持、局部力跟踪、移动距离和执行延迟；
- DP-direct 的明确优势是较低的 wrist resultant force-z RMSE，且 CUDA 推理仍满足 20 ms 周期；
- DP-direct 的 contact loss、authority intervention 和 force spike 更严重；
- 两种策略都明显超过 8 N，说明 Exp.1 的旧 shared stack 还有共同的冲击/连续性问题；这不是
  DP 独有问题，也不能据此把实验标成失败。

结论仅限于：当前 DP-direct 不如 analytical MCC 适合直接承担高频低层柔顺控制。该结果支持把
DP 上移为 nominal finger trajectory/role generator，但不是对 DP 路线作总体否定。

## 3. Exp.2：Plain 绝对参考 + Passive / Reactive / DPRef

`Plain whole-hand MCC` 是最初的普通解析全手 MCC：不经过新 Role Interpreter/ForceSafety wrapper，
用于显示基础接触保持能力，不参加 reference-source 因果归因。

另外三种策略共享完全相同的：

```text
Reference/Role Interpreter
 -> resultant/internal-force coordinator
 -> Wrist MCC + Finger MCC
 -> runtime guard
```

这三个 shared-stack 分支唯一变量是 nominal reference source：

- `Passive-Hold + MCC`：保持当前 nominal reference，只依靠 MCC 反应；
- `Reactive-Heuristic + MCC`：使用因果局部启发式，不读取未来 wrist trajectory；
- `DPRef/Role + MCC`：读取状态和 future wrist plan，生成 nominal joint chunk 与 role intention。

| 三条件 aggregate | Plain | Passive | Reactive | DPRef |
|---|---:|---:|---:|---:|
| contact continuity | **99.21%** | 97.23% | 97.26% | **98.77%†** |
| 平均 contact 数 | **3.156** | 2.285 | 2.310 | **2.450†** |
| `P(N_c>=2)` | **87.69%** | 61.02% | 61.23% | **84.70%†** |
| `P(N_c>=3)` | **75.37%** | 43.90% | 45.08% | **46.64%†** |
| 四指同时接触 | **53.30%** | 26.31% | **27.45%†** | 14.86% |
| force RMSE（诊断） | 1.910 N | 1.855 N | 1.752 N | 1.570 N |
| worst peak force（诊断） | 11.65 N | 56.77 N | 52.24 N | 4.83 N |
| 平均 `>8 N` 时间（诊断） | 0.783 s | 0.092 s | 0.029 s | 0 s |
| 平均多指同时 `>8 N` | 0 s | 0 s | 0 s | 0 s |
| total Y traversal | **174.36 mm** | 166.81 mm | 170.53 mm | **173.84 mm*** |
| supported Y, `N_c>=2` | **138.87 mm** | 89.35 mm | 86.90 mm | **126.09 mm*** |
| DPRef/reference P95 | — | — | — | 13.54 ms |

`†` 表示仅在严格可比的 Passive/Reactive/DPRef 子集中最好；Plain 是不同执行栈的绝对参考。

策略优劣：

- **Plain MCC**：多指丰富度最高，但平均约 0.783 s 处于单指 `>8 N`；它显示解析基础上限，
  不能与后三者作“只替换 reference source”的因果比较。
- **Passive-Hold**：是不使用预测的 shared-stack 下限；相对 Plain 平均少 0.871 个接触、
  supported distance 少 49.52 mm。
- **Reactive-Heuristic**：相对 Passive 平均多 0.026 个接触、`P(N_c>=3)` 高 1.18 pp、总 traversal
  多 3.73 mm，但 supported `N_c>=2` 少 2.45 mm；提升很小。
- **DPRef**：在严格共享栈的三者中，continuity、平均接触数、`P(N_c>=2/3)` 和 supported distance
  全部最好；相对最佳解析源分别提高 1.51 pp、0.139 个接触和 36.75 mm。代价是四指同时接触率
  低 12.60 pp，载荷明显偏向前三指，尚不能声称可靠 handover。

数据覆盖限制：DPRef validation 没有 `RELEASE` label，`MAKE` 只有 20 个且准确率为 60%。因此
本实验能评价当前轨迹参考的执行性能，但不能声称已经验证 handover / BREAK 泛化。

## 4. Exp.3 不属于 E05

这里的 active planning 指：SurfaceModel/GPIS 已运行，由在线 planner 选择下一段 wrist trajectory，
而不是 E05 中预先固定 wrist trajectory。Exp.3 比较：

```text
Explicit wrist + finger/contact-mode planner + shared MCC
vs.
Wrist-only active planner + DPRef/Role generator + shared MCC
```

它回答“active tactile exploration 是否需要显式规划高维 fingers/contact modes”，属于完整系统的
planner-level ablation。正式位置为 [`MASTER_PLAN.md`](MASTER_PLAN.md) 中 **I05 之后的 I06 / Exp.3**；
I04 已独立冻结为 given-good-next-point 的 Oracle whole-hand contact traversal，不选择探索点；
I05 是完整 GPIS main-vs-baseline 主实验。

## 5. 复现与审阅

统一生成：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.e05_strategy_review
```

统一目录：

```text
Module/generated/e05_exp1_exp2_review/
├── index.html                 # 单页指标、分析、图片和视频
├── summary.json               # 无策略 verdict 的机器可读汇总
├── metrics.csv                # 六种策略的扁平指标
├── exp1_dashboard.png
├── exp1_mcc.mp4
├── exp1_dp_direct.mp4
├── exp1_side_by_side.mp4
├── exp2_dashboard.png
├── exp2_plain.mp4
├── exp2_passive.mp4
├── exp2_reactive.mp4
└── exp2_dpref.mp4
```

原始 source summaries 可能仍含历史 verdict 字段，仅用于 provenance；本文件和统一网页是当前 E05
解释协议。
