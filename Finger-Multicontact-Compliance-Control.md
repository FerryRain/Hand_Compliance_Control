# 面向全指腹多接触的分层柔顺控制方案

## 1. 文档目的

本文档整理当前项目中关于以下问题的设计结论：

- Diffusion Policy（DP）应当输出什么；
- 手掌/机械臂 MCC 与手指柔顺控制如何分工；
- 当接触不仅发生在指尖，而是分布在近段、中段和远端指节时，如何建立动力学模型；
- FSR、指尖 6D 视触觉传感器和电机电流分别能够提供什么信息；
- 如何实现能够容忍 DP 目标轻微穿透的弹簧阻尼/阻抗控制器；
- 当前实现与论文 `Contact-Implicit Model Predictive Control for Dexterous In-hand Manipulation: A Long-Horizon and Robust Approach`（arXiv:2402.18897v2）的关系和差别；
- 后续应当按照什么顺序实现和验证。

参考论文：`/home/rimlab/Desktop/2402.18897v2.pdf`。

---

## 2. 最终任务与控制目标

任务目标是：给定空间中的目标滑动方向，手能够沿未知或部分已知物体表面移动，同时尽可能保持手掌和多根手指的包裹接触。

这里的“稳定接触”不等价于：

- 所有掌部 FSR 永远有值；
- 四根手指的所有指节始终接触；
- 每个近端指节必须接触；
- 所有 FSR 都必须超过同一个阈值。

对于当前任务，更合理的稳定性定义是：

- 大部分时间至少三根手指保持有效接触；
- 单根手指允许在近段、中段、远端之间转移接触；
- 短时 FSR 掉线可以接受，但不能出现长时间整指脱离；
- 接触力不能长期过大；
- 不能出现明显穿透、滑移失控或高频开合振荡；
- 手掌是否贴合需要结合任务几何判断，而不能只看掌部 FSR。

---

## 3. 推荐的整体分层结构

```text
目标滑动方向、表面信息、历史触觉和手状态
                         │
                         ▼
                 Diffusion Policy
                     约 20 Hz
        输出 nominal palm pose + finger q_pre
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Palm/Arm MCC             Finger contact controller
       约 100 Hz                    约 100 Hz
  法向柔顺与切向跟踪          FSR/触觉修正 q_pre
             │                       │
             ▼                       ▼
     机械臂力矩或位置目标        手指关节阻抗控制
                                   更高频率
```

### 3.1 DP 的职责

DP 负责低频、长时间尺度的几何规划：

- 根据目标方向规划未来 palm control pose；
- 根据历史接触和局部表面变化预测 finger pre-shape；
- 提前适应曲率变化，而不是等失去接触后再被动恢复；
- 输出名义参考，而不是必须刚性执行的真实状态。

DP 不应直接负责：

- 高频接触力调节；
- 每个 FSR 的瞬时闭环控制；
- 机械臂关节力矩；
- 碰撞安全和过力保护；
- 机械臂可达域和奇异位形处理。

### 3.2 Palm/Arm MCC 的职责

MCC 接收 DP 给出的 palm nominal target，负责：

- 沿表面切向跟踪 DP 轨迹；
- 在表面法向保持柔顺和接触力；
- 吸收 DP palm target 的小幅法向误差；
- 避免直接将 DP 位置误差转化为刚性穿透；
- 处理机械臂动力学、关节限制和可达性。

推荐将 palm 控制分解为：

- 切向：位置/轨迹跟踪；
- 法向：MCC 力控或导纳；
- 姿态：palm-Z 对齐表面法向；
- wrist yaw：不强制锁死，或只做弱正则化。

### 3.3 手指控制器的职责

手指控制器接收 DP 的 `q_pre`，负责：

- 无接触时向预形状靠近；
- 接触后降低预形状锚定刚度；
- 根据近段、中段和远端 FSR 调整平衡位置；
- 利用真实关节速度阻尼抑制振荡；
- 发生过力时主动回撤；
- 接触丢失时有限速地重新寻找表面；
- 即使 DP 的 `q_pre` 略微位于物体内部，也只产生有限、可控的接触力。

---

## 4. DP 单轨迹部署问题的结论

此前只使用一条轨迹 overfit 时，出现了以下现象：

- teacher palm + DP q_pre：接触能够稳定保持；
- DP palm + DP q_pre，但使用 teacher history：能够较好复现；
- DP full + live history：预测逐渐偏移，随后发生过力和穿透。

这说明 DP 输出头在训练分布内可以拟合教师，主要困难发生在实时观测进入模型之后。

### 4.1 单轨迹数据的影响

单条轨迹中，时间、palm pose、hand q 和 FSR 几乎一一对应。模型可能将随机 FSR 波动当成时间或轨迹位置的强特征：

```text
FSR 出现轻微差异
→ 模型误判当前轨迹位置
→ 输出不同阶段的 palm/q_pre
→ 接触状态进一步改变
→ 下一次观测偏差更大
```

增加不同初始姿态、运动方向、接触波动和曲率变化的数据，可以迫使模型学习更稳定的几何—动作关系，而不是记忆一条 FSR 时间序列。

### 4.2 增加数据不能自动解决的问题

如果存在以下情况，单纯增加正常 teacher 数据仍然不够：

- live FSR 的尺度远超训练范围；
- FSR body 顺序或含义不一致；
- 训练和部署使用不同坐标系；
- 法向定义或符号不一致；
- 训练曲率经过平滑，而部署曲率由短窗口直接计算；
- replay 动力学与数据采集环境明显不同；
- DP 偶发输出没有位姿、速度和接触力安全限制。

因此，多轨迹训练与输入语义对齐必须同时进行。

---

## 5. DP 模块的完整设计规格

### 5.1 DP 在系统中的定位

本项目中的 DP 不是直接控制机器人关节的端到端 policy，而是一个局部表面运动参考生成器。它需要回答两个问题：

1. 为了沿目标方向继续移动，未来一段时间 palm control pose 应该如何变化；
2. 面对即将到来的表面曲率变化，手指应提前形成什么慢速预形状。

DP 不负责高速接触稳定。即使 DP 输出存在几毫米位置误差或小角度姿态误差，MCC 和手指阻抗也应当保证系统不会刚性穿透或立即失去接触。

### 5.2 推荐输入

每个 DP 时刻的状态建议定义为：

\[
o_t=
\left[
d_t,
s_t,
q_{h,t},
T_{op,t},
n_t,
\kappa_t
\right]
\]

其中：

- \(d_t\in\mathbb R^3\)：目标滑动方向；
- \(s_t\)：FSR/触觉状态；
- \(q_{h,t}\in\mathbb R^{16}\)：当前手指关节位置；
- \(T_{op,t}\)：object frame 下的 palm control pose；
- \(n_t\in\mathbb R^3\)：局部表面法向；
- \(\kappa_t\)：局部曲率或曲率 proxy。

当前实现的 48 维输入布局是：

```text
desired_direction                 3
FSR                              16
hand_q                           16
palm relative position            3
palm relative rotation 6D         6
surface normal                     3
curvature proxy                    1
-----------------------------------
total                             48
```

#### FSR 输入的推荐形式

直接输入 16 维原始 FSR 可以保留全指腹压力分布，但容易让模型记忆单个传感器噪声。建议至少保存并比较以下输入方案：

1. `raw16`：归一化并滤波后的 16 维 FSR；
2. `raw16 + mask16`：压力加每个 FSR 的接触开关；
3. `region12`：四指 × 近/中/远三个区域汇总；
4. `region12 + contact4`：区域压力加四指接触状态；
5. 训练时对 raw16 加入噪声、比例漂移和短时 dropout。

第一版可以维持 16 维输入，但必须确保采集、训练和部署使用相同的：

- FSR body 顺序；
- 单位或归一化方式；
- EMA/低通滤波；
- 截断范围；
- 接触阈值；
- 无接触时的零值定义。

### 5.3 推荐输出

每个未来时刻输出：

\[
a_t=
\left[
T^{target}_{op,t},
q^{DP}_{pre,t}
\right]
\]

当前实现的 25 维 action 布局为：

```text
palm target relative position     3
palm target relative rotation 6D  6
finger q_pre                      16
-----------------------------------
total                             25
```

这里的 palm target 和 `q_pre` 都是 nominal reference：

- palm target 交给 MCC；
- q_pre 交给手指接触导纳与阻抗控制器；
- 不能直接写入 robot root pose；
- 不能直接覆盖真实 hand q。

### 5.4 为什么使用 object-relative 坐标

数据采集时物体主动运动、手尽量留在原世界位置。反演后固定物体，相对运动关系应保持不变：

\[
p_{op}=R_o^\top(p_p-p_o)
\]

\[
R_{op}=R_o^\top R_p
\]

其中：

- \(p_o,R_o\)：物体世界位姿；
- \(p_p,R_p\)：palm control point 世界位姿；
- \(p_{op},R_{op}\)：物体坐标系下的手掌位姿。

旋转建议使用 rotation-6D 作为网络输入和输出，避免四元数双覆盖以及旋转向量在大角度处的不连续。

反演只应改变坐标表达，不应改变手和物体的相对几何。如果原始几何 replay 不能精确重合，应优先检查：

- palm body origin 与 palm control point 是否混用；
- MuJoCo `wxyz` 与 SciPy `xyzw` 是否混用；
- 左乘/右乘和主动/被动旋转是否写反；
- object pose 和 palm pose 是否来自同一仿真时刻；
- XML、mesh scale 和初始姿态是否一致。

### 5.5 Palm control point 的定义必须固定

DP 轨迹应围绕实际被控制的 palm control point 建立，而不是任意 palm body origin。当前项目优先使用 palm FSR bodies 的几何中心，是为了让：

- 数据反演点；
- DP 输出点；
- MCC 跟踪点；
- surface normal 查询点；

保持为同一个物理点。

若 DP 预测的是 control point，而执行器控制的是 `palm_lower` root/body origin，需要使用固定局部偏移：

\[
p_{control}=p_{root}+R_{root}r_{offset}
\]

部署时必须反算：

\[
p_{root}=p_{control}-R_{root}r_{offset}
\]

### 5.6 时序窗口

DP 应使用历史窗口而不是单帧状态：

\[
O_t=\{o_{t-H_o+1},\ldots,o_t\}
\]

并预测未来 action chunk：

\[
A_t=\{a_t,a_{t+1},\ldots,a_{t+H_a-1}\}
\]

推荐控制频率分层：

```text
仿真/内层关节控制：数百 Hz 或更高
FSR 接触控制：       100 Hz
DP 推理：             20 Hz
```

若原始数据为 100 Hz，DP stride 可以取 5。一个 8 帧 observation history 对应约 0.4 s 历史。实际 horizon 应根据表面运动速度和控制器滞后确定。

DP 以 receding-horizon 方式执行：每次只执行预测序列中的少量点，然后使用最新触觉重新规划，不能一次无反馈地执行整个长 action chunk。

### 5.7 目标方向条件

训练时的目标方向应从教师轨迹未来位移投影到局部切平面得到：

\[
\Delta p_t=p_{t+L}-p_t
\]

\[
d_t=
\frac{
\Delta p_t-n_t(n_t^\top\Delta p_t)
}{
\left\|\Delta p_t-n_t(n_t^\top\Delta p_t)\right\|
}
\]

部署时必须用用户/任务给定方向替代 teacher direction。当前 replay 若仍从 teacher 数据按时间索引读取 direction，只是在验证轨迹复现，还没有验证任意方向条件控制。

外部给定的世界方向也应先转换到 object frame，并投影到局部切平面。

### 5.8 法向和曲率输入

第一阶段可以假设已知局部法向和曲率，但训练与部署算法必须一致。

曲率 proxy 可以使用法向随弧长的变化：

\[
\kappa_t\approx
\frac{\arccos(n_t^\top n_{t-L})}
{\sum_{i=t-L+1}^{t}\|p_i-p_{i-1}\|}
\]

需要注意：

- 位移太小时禁止直接相除；
- 必须限幅；
- 训练不能使用部署无法获得的未来信息；
- 训练与部署应使用相同 lag；
- 实时端应使用因果滤波；
- 不建议训练端用零相位未来滤波、部署端用三帧未滤波估计。

如果法向和曲率暂时不可靠，可以做输入消融：

- 有 normal + curvature；
- 只有 normal；
- 二者都无，仅使用触觉和历史位姿。

但对于 smooth convex 未知表面泛化，稳定的法向或等价局部几何信息会显著降低学习难度。

### 5.9 q_pre 标签的定义

`q_pre` 应表示手指的慢速几何预形状，而不是实际接触振荡后的瞬时关节位置。

当前从实际 hand q 低通得到 q_pre 的方法可以用于初期实验，但有两个局限：

- 实际 q 已包含接触反力和快速柔顺修正；
- 零相位低通可能使用未来信息，部署时不能严格复现其因果关系。

更理想的数据记录是：

```text
q_pre_nominal       教师的几何预形状
delta_q_contact     FSR/导纳修正
q_eq                内层阻抗平衡参考
q_actual            实际关节位置
```

DP 监督 `q_pre_nominal`，底层控制器负责 `delta_q_contact`。

### 5.10 数据质量筛选

DP 数据必须按完整 trajectory 筛选，不能删除零散坏帧后把前后不连续的状态拼接起来。

当前任务推荐使用：

- 三指 FSR 接触率；
- 至少两指接触率；
- 最长连续三指掉线步数；
- 几何三指接触率（仿真 privileged metric）；
- palm tracking error；
- palm-lost 比例；
- FSR 峰值；
- 关节速度峰值；
- 穿透和过力事件。

不应将“所有 palm FSR 有值”或“所有近端 FSR 有值”作为硬筛选条件。

训练/验证集必须按 trajectory、物体初始姿态或物体实例划分，不能随机按帧划分，否则相邻时间帧泄漏会导致过于乐观的验证结果。

### 5.11 训练增强

为了降低 DP 对偶然 FSR 抖动的敏感性，建议在训练时加入：

- FSR 小幅加性噪声；
- 每个传感器独立比例漂移；
- 整体增益变化；
- 随机 1–5 帧短时 dropout；
- 接触阈值附近的轻微扰动；
- palm pose 毫米级扰动；
- hand q 小角度扰动；
- normal 小角度旋转扰动；
- curvature 限幅范围内扰动。

增强幅度必须来自真实传感器或仿真统计，不能大到破坏物理对应关系。

### 5.12 DP 输出安全投影

DP 输出在进入 MCC 和手指控制器之前应经过安全层：

#### Palm target

- 限制单次平移增量；
- 限制单次旋转增量；
- 限制相对表面的法向深度；
- 保证 rotation-6D 重建后是合法旋转；
- 异常输出时保持上一安全目标；
- MCC 报告过力或 tracking lost 时暂停切向前进。

#### Finger q_pre

- 投影到关节限位；
- 限制每次 q_pre 变化；
- 限制 q_pre 与当前 q 的最大距离；
- 过力时禁止继续向内更新；
- 电流超限时覆盖 DP，执行安全回撤。

这不是让安全层替代 DP，而是避免一次低概率扩散采样异常直接形成危险动作。

### 5.13 Replay 与部署验证顺序

DP 应按以下顺序验证：

1. **原始几何 replay**：只验证反演和 XML 几何；
2. **teacher replay**：teacher palm + teacher q，验证 FSR 重算；
3. **teacher palm + DP q_pre**：隔离手指输出；
4. **DP palm + teacher q/q_pre**：隔离 palm 输出；
5. **DP palm + DP q_pre + teacher history**：验证网络在训练分布内的完整输出；
6. **逐字段 live observation**：依次替换 live q、pose、normal、curvature、FSR；
7. **DP full + live history + hand-only impedance**；
8. **DP full + live history + Palm MCC + finger impedance**；
9. **外部目标方向替代 teacher direction**；
10. **未知初始姿态、未知轨迹和新物体测试**。

逐字段替换非常重要，它可以确定首先导致闭环偏移的是 FSR、曲率还是位姿，而不是把所有问题归因于“模型没有泛化”。

### 5.14 多轨迹数据与闭环恢复数据

高质量 teacher 轨迹能够覆盖正常接触状态，但未必覆盖 policy 自己产生的小偏差。若多轨迹训练后仍存在闭环累积误差，可以加入：

- 从 teacher 状态加入小位姿/关节扰动后的恢复轨迹；
- 由旧 policy rollout，再由 teacher/controller 修正的 DAgger 风格数据；
- 接触丢失后重新贴合的恢复片段；
- 过力后安全回撤并继续滑动的片段。

这些数据的目标不是教 DP 取代底层阻抗，而是让 DP 在偏离 nominal trajectory 后仍能给出合理的慢速几何参考。

### 5.15 DP 成功标准

DP 不能只用离线 action MSE 或 diffusion loss 判断。至少需要同时满足：

- teacher-history replay 的 palm/q_pre 误差足够小；
- live-history replay 不发生误差持续放大；
- 三指接触率保持在任务要求范围；
- FSR/电流不过力；
- palm target 增量平滑；
- 沿目标方向达到指定切向位移；
- 接触丢失次数和最长持续时间受控；
- 对未见初始姿态仍能工作；
- 对 smooth convex 新表面具有可测量的泛化能力。

---

## 6. 多指节接触动力学

### 6.1 多接触模型仍然成立

假设一根手指在近段、中段和远端分别有接触，整根手指受到的外部关节力矩为：

\[
\tau_{\mathrm{ext}}
=
J_p^\top f_p
+J_m^\top f_m
+J_d^\top f_d
\]

其中：

- \(J_p\)：近段接触点相对于手指关节的 Jacobian；
- \(J_m\)：中段接触点 Jacobian；
- \(J_d\)：远端或指尖接触点 Jacobian；
- \(f_p,f_m,f_d\)：各接触点的接触力。

如果指尖传感器输出的是完整 6D wrench，则对应项写为：

\[
\tau_{d}=J_{d,6D}^\top w_d
\]

### 6.2 “近段 1 轴、中段 2 轴、远端 3 轴”的理解

这个理解在直觉上基本正确：一个接触点主要受它上游关节影响，下游关节不会改变该接触点的位置。因此 Jacobian 会自然形成嵌套结构。例如忽略侧摆关节时：

\[
J_p=\begin{bmatrix}* & 0 & 0\end{bmatrix}
\]

\[
J_m=\begin{bmatrix}* & * & 0\end{bmatrix}
\]

\[
J_d=\begin{bmatrix}* & * & *\end{bmatrix}
\]

如果把 LEAP Hand 每指的侧摆/根部自由度纳入，则使用相应的 4 自由度 Jacobian。

但近、中、远三个接触不能被当成三个互相独立的机械臂。近端关节运动会同时改变中段和远端接触状态，因此三个接触弹簧必须联合映射到同一组关节。

### 6.3 多接触系统的一般动力学

整根手指可写为：

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=
\tau+\sum_iJ_i^\top f_i
\]

多接触不会使动力学公式失效，只会增加接触项。真正的困难是：传感器能否区分每个接触点的力，以及多个接触目标是否互相冲突。

---

## 7. 三类传感器能够提供什么

| 信息来源 | 主要优势 | 推荐用途 | 局限 |
|---|---|---|---|
| 指节 FSR | 覆盖近段、中段和远端，结构简单 | 接触开关、相对法向压力、压力分布、过力回撤 | 精度和灵敏度有限，难以测量切向力 |
| 指尖 6D 视触觉 | 指尖 wrench 精度高，可感知切向力和力矩 | 指尖精确力控、滑移检测、接触法向估计 | 只能观测指尖，不能直接观测近段和中段 |
| 电机电流 | 覆盖整根运动链，可转换为关节力矩 | 过载保护、总外力矩估计、传感器交叉校验 | 摩擦、减速器、温漂和电流噪声会污染低力估计 |

### 7.1 第一版最低信息要求

第一版控制器不要求每个接触点都有准确的 3D 力，只需要：

- 关节位置 \(q\)；
- 关节速度 \(\dot q\)；
- DP 输出的 \(q_{\mathrm{pre}}\)；
- 每个指节区域是否接触；
- 每个指节区域的相对压力；
- 每个 FSR pad 在指节上的固定位置和局部法向；
- 关节限位、电流上限和安全回撤条件。

FSR 只要在工作区间内大致单调，就可以用于区间控制，不必立刻换算为牛顿。

### 7.2 FSR 的法向力近似

对第 \(i\) 个 FSR，可以采用：

\[
f_i\approx s_i n_i
\]

其中：

- \(s_i\)：FSR 标量或归一化压力；
- \(n_i\)：FSR pad 的局部法向，经正向运动学转换到世界坐标。

该模型无法恢复切向摩擦力，但足以用于保持法向接触和限制过力。

### 7.3 电机电流的用途

经过标定后：

\[
\tau_{\mathrm{motor}}=K_t I
\]

外部接触力矩可以估计为：

\[
\tau_{\mathrm{ext}}
\approx
\tau_{\mathrm{motor}}
-M(q)\ddot q
-C(q,\dot q)\dot q
-g(q)
-\tau_{\mathrm{friction}}
\]

在低速抚摸的准静态条件下，可先近似：

\[
\tau_{\mathrm{ext}}
\approx
\tau_{\mathrm{motor}}
-g(q)
-\tau_{\mathrm{friction}}
\]

电流不适合作为唯一的接触定位信息，但非常适合检测：

- FSR 无值但电机负载明显增大；
- FSR 卡死或输出异常；
- 整根手指过载；
- DP `q_pre` 过度内收。

---

## 8. 推荐手指控制器：关节阻抗 + 多区域接触导纳

### 8.1 为什么不直接使用三个独立弹簧

如果近段、中段和远端分别独立输出关节动作，会出现：

- 近段为了减力而张开，导致远端失去接触；
- 远端为了补力而闭合，导致近段过力；
- 多个控制器对同一个上游关节给出相反命令；
- 接触切换时 action 不连续，引发振荡。

因此，应当把三个接触区域的误差统一映射到关节空间。

### 8.2 接触区域定义

对每根手指建立三个区域：

```text
proximal: 近段指腹 FSR 的稳健均值或加权均值
middle:   中段指腹 FSR 的稳健均值或加权均值
distal:   远端 FSR；若有 6D 视触觉，则优先使用其法向力
```

记测量值为：

\[
s=\begin{bmatrix}s_p&s_m&s_d\end{bmatrix}^\top
\]

未标定 FSR 时采用区间误差：

\[
e_i=
\begin{cases}
s_{i,\min}-s_i,&s_i<s_{i,\min}\\
0,&s_{i,\min}\le s_i\le s_{i,\max}\\
s_{i,\max}-s_i,&s_i>s_{i,\max}
\end{cases}
\]

### 8.3 法向接触 Jacobian

将三个接触点的法向 Jacobian 堆叠：

\[
J_n=
\begin{bmatrix}
n_p^\top J_p\\
n_m^\top J_m\\
n_d^\top J_d
\end{bmatrix}
\]

这样每个接触误差通过实际运动学结构影响对应关节，而不是手工假定三个控制器完全独立。

### 8.4 外层接触导纳

使用一个联合导纳状态 \(\Delta q\)：

\[
M_a\Delta\ddot q
+D_a\Delta\dot q
+K_a\Delta q
=
B_f e
\]

可以选择：

\[
B_f=J_n^\top
\]

或者在初期使用根据手指结构手工设计的低增益映射矩阵，随后再替换为 Jacobian 映射。

DP 的名义预形状与触觉修正组合为：

\[
q_{\mathrm{eq}}
=q_{\mathrm{pre}}^{DP}+\Delta q
\]

### 8.5 内层关节阻抗

若支持力矩控制：

\[
\tau_f
=
\tau_g
+K_q(q_{\mathrm{eq}}-q)
+D_q(\dot q_{\mathrm{eq}}-\dot q)
\]

若底层是位置伺服，则输出：

\[
q_{\mathrm{cmd}}=q_{\mathrm{eq}}
\]

并使用执行器/XML 内部的 `kp`、`kv` 或真实电机驱动器的位置环实现阻抗近似。

### 8.6 接触状态机

每个区域至少需要以下状态：

1. `FREE`：没有接触，主要跟踪 DP `q_pre`；
2. `SEEKING`：此前有接触但刚刚丢失，以有限速度向内寻找；
3. `CONTACT`：压力位于有效范围，降低 q_pre 锚定刚度；
4. `OVERFORCE`：压力、电流或指尖 wrench 超限，立即回撤；
5. `FAULT`：传感器互相矛盾或持续过载，保持/张开到安全姿态。

接触开关必须使用迟滞：

\[
s_{\mathrm{on}}>s_{\mathrm{off}}
\]

避免 FSR 在阈值附近导致状态反复跳变。

---

## 9. DP 目标穿透时会发生什么

### 9.1 轻微穿透

DP 的 `q_pre` 可以略微位于物体内部。此时实际关节不会刚性到达 `q_pre`，而会停在弹簧力与接触反力的平衡位置：

\[
K_q(q_{\mathrm{eq}}-q)\approx J^\top f_e
\]

这正是将 `q_pre` 定义为“弹簧平衡参考”而不是“真实关节命令”的意义。

### 9.2 持续或严重穿透

阻尼只能抑制瞬态速度，不能消除持续目标误差。若 `q_pre` 位于物体内部过深，稳态接触力仍会随刚度和目标误差增大。

因此必须同时设置：

- `q_pre` 单次变化率上限；
- `q_pre` 相对当前关节的最大偏移；
- 每个接触区域的最大 FSR；
- 指尖 6D wrench 上限；
- 电机电流/估计关节力矩上限；
- \(\Delta q\) 和 \(\Delta\dot q\) 限幅；
- 过力状态下的强制回撤；
- DP 异常输出时保持上一安全目标。

弹簧阻尼是主要柔顺机制，但不能替代安全监督器。

---

## 10. 与参考论文的关系

论文采用三级结构：

1. 高层 contact-implicit MPC 生成关节位置和接触力参考；
2. 中层 contact controller 根据触觉修正参考；
3. 底层 joint impedance controller 执行修正后的关节参考。

论文关节控制公式为：

\[
\tau
=
\tau_g
+K_r(q_d-q)
+D_r(\dot q_d-\dot q)
+J^\top F_{e,\mathrm{ff}}
\]

当前项目的对应关系是：

| 论文模块 | 当前项目对应模块 |
|---|---|
| Contact reference generator | DP 输出 palm pose、finger q_pre |
| Compliance-based contact controller | FSR/视触觉驱动的多区域导纳 |
| Joint impedance controller | 手指关节 PD/阻抗执行器 |
| Object/robot motion controller | Palm/Arm MCC |

当前方案不是完整复现论文的 contact-force optimal controller，而是针对现有传感器和控制接口实现的简化版本。核心思想保持一致：高层参考不能被刚性执行，必须经过触觉反馈修正和底层阻抗。

---

## 11. 当前手指控制器与目标模型的差别

当前 `LeapHandComplianceController` 主要包含：

- FSR EMA；
- 接触开关迟滞；
- FSR 区间误差；
- `D_force * ΔFSR`；
- q_pre 位置锚定；
- `tanh` 输出限制；
- action rate limit。

其中：

\[
-D_{force}\Delta FSR
\]

是压力变化率反馈，不是论文中的真实关节速度阻尼：

\[
-D_q\dot q
\]

因此当前控制器可以抑制部分触觉抖动，但还不能被视为严格的被动关节弹簧阻尼系统。

推荐保留：

- FSR 滤波；
- 区间目标；
- 接触迟滞；
- action/参考变化率限制；
- 无接触时的 q_pre 寻找逻辑。

推荐新增或重构：

- 显式的 \(q_{eq}\) 或 \(\Delta q\) 状态；
- 基于 \(\dot q\) 的关节阻尼；
- 近/中/远多接触的联合映射；
- 电流和指尖 wrench 的安全监督；
- 过力回撤状态；
- DP `q_pre` 的安全投影。

---

## 12. DP 数据标签需要注意的问题

当前层次化数据准备中，`q_pre` 是对实际 hand q 做低通滤波得到的。它可以作为慢速几何预形状的近似，但不严格等于教师控制器内部的弹簧平衡位置。

以后采集数据时，最好分别记录：

- `q_pre_nominal`：教师控制器或高层规划器给出的名义预形状；
- `delta_q_contact`：FSR 柔顺控制产生的接触修正；
- `q_eq`：最终送给内层阻抗的平衡参考；
- `q_actual`：真实关节位置；
- `fsr_raw` 和 `fsr_filtered`；
- `contact_state`；
- `motor_current/estimated_torque`；
- 指尖 6D wrench（如果可用）。

推荐关系：

\[
q_{eq}=q_{pre,nominal}+\Delta q_{contact}
\]

DP 只学习 `q_pre_nominal`，快速接触修正交给底层控制器。这样不会把触觉噪声、瞬时振荡和几何预形状混入同一个监督标签。

---

## 13. FSR、电流和 6D 触觉的融合方案

### 13.1 第一阶段：不做精确力重建

- FSR：归一化、迟滞和区间控制；
- 指尖 6D：只用于远端过力和滑移监督；
- 电流：只用于关节过载保护；
- 不尝试从电流反演每个指节的精确接触力。

这是最稳妥、最容易验证的方案。

### 13.2 第二阶段：估计多区域法向力

已知活动接触区域后，可求解：

\[
\min_{f_p,f_m,f_d\ge0}
\left\|
J_n^\top
\begin{bmatrix}f_p\\f_m\\f_d\end{bmatrix}
-\tau_{ext}
\right\|_W^2
+\lambda
\left\|f-f_{FSR}\right\|^2
\]

如果指尖 6D 传感器能够可靠给出远端法向力，可固定或强约束 \(f_d\)，再估计近段和中段。

### 13.3 第三阶段：滑移和切向控制

- 指尖使用 6D 视触觉估计切向力和摩擦裕度；
- 近/中段使用 FSR 压力中心变化、接触开关序列和关节运动间接检测滑移；
- DP 使用较慢的接触历史预测几何变化；
- 底层控制器只处理快速稳定和过力保护。

---

## 14. 推荐实施顺序

### 阶段 A：多轨迹 DP 基线

1. 使用质量筛选后的 A/B 轨迹重新准备 DP 数据；
2. 训练/验证按 trajectory 划分，不能随机按帧划分；
3. 对 FSR 加入合理噪声、比例扰动和随机短时 dropout；
4. 保持 teacher palm，只部署 DP q_pre；
5. 验证 DP 是否仍然对轻微 FSR 差异敏感。

### 阶段 B：手指阻抗重构

1. 将 DP `q_pre` 解释为 equilibrium/reference；
2. 新增显式 \(q_{eq}\) 或 \(\Delta q\) 状态；
3. 将 FSR 修正作用到 \(q_{eq}\)，而不是直接产生高频 action；
4. 增加真实 \(\dot q\) 阻尼；
5. 增加 q_pre、\(\Delta q\)、关节速度和电流限幅；
6. 先在固定 palm、固定物体条件下验证。

### 阶段 C：多区域联合控制

1. 为每个 FSR pad 建立位置和法向；
2. 计算近、中、远法向 Jacobian；
3. 使用联合映射代替互相独立的手工关节增量；
4. 测试单接触、双接触和三接触切换；
5. 检查近段回撤是否破坏远端接触。

### 阶段 D：加入 Palm MCC

1. DP palm 只作为 nominal target；
2. MCC 法向力控，切向跟踪；
3. palm-Z 对齐法向，wrist yaw 不锁死；
4. 手指控制频率高于 DP；
5. 检查手指接触反力是否将手掌推离表面。

### 阶段 E：真实传感器融合

1. 标定 FSR 的单调区间和迟滞；
2. 标定电流—力矩映射和摩擦；
3. 接入指尖 6D wrench；
4. 增加传感器一致性检查；
5. 最后再尝试多接触力反演。

---

## 15. 验证实验与评价指标

### 15.1 固定物体、固定 palm

验证手指控制器本身：

- 三指有效接触率；
- 单指接触持续率；
- 最长连续掉接触步数；
- FSR 超限占比；
- 关节速度 RMS 和峰值；
- q_pre 阶跃后的过冲和恢复时间；
- 接触后是否持续开合。

### 15.2 固定物体、移动 palm target

验证 DP 与手指控制耦合：

- DP target 小幅穿透时的最大接触力；
- DP target 加入 2–5 mm 位姿扰动后的恢复；
- q_pre 加入小角度扰动后的恢复；
- live FSR 与 teacher FSR 的分布差异；
- DP 输出误差是否随时间累积。

### 15.3 MCC + 手指联合控制

验证手掌和手指是否互相对抗：

- palm tracking error；
- 法向接触力稳定性；
- 手指闭合时 palm 是否被推离；
- MCC 是否因手指高频力变化发生振荡；
- 手指控制是否因 palm 调整反复掉接触。

### 15.4 DP 闭环部署

除了接触率，还必须记录：

- palm prediction error；
- q_pre prediction error；
- 每次 replan 的目标增量；
- live state 相对训练分布的标准化距离；
- 最大 FSR norm；
- 最大电机电流；
- 穿透深度；
- force/slip/contact-loss 事件数量。

单看“有接触”可能把严重穿透误判为成功，因此接触率必须与过力和穿透指标联合使用。

---

## 16. 推荐的初始控制原则

在没有完成精确传感器标定前，建议遵守：

1. 使用 FSR 区间而不是精确目标牛顿值；
2. 接触后降低 q_pre 锚定刚度；
3. 无接触时才较强地跟踪 q_pre；
4. FSR 外层慢于关节内层；
5. 真实关节速度阻尼独立于 FSR 导数反馈；
6. 过力回撤优先级高于 DP 跟踪；
7. DP 输出必须限速、限幅并进行安全投影；
8. 近、中、远接触通过联合 Jacobian 或耦合矩阵协调；
9. 电机电流首先用于安全，不急于用于精确接触力反演；
10. 指尖 6D 触觉首先用于滑移和远端力监督。

---

## 17. 当前推荐结论

当前最适合项目的路线不是要求每个指节都安装精确 6D 力传感器，也不是让 DP 直接输出全部关节动作，而是：

```text
DP 学习慢速几何参考
        +
MCC 负责 palm/arm 柔顺跟踪
        +
多区域 FSR 导纳负责接触平衡位置修正
        +
关节阻抗负责真实弹簧阻尼
        +
电流与指尖 6D 触觉负责安全和精细监督
```

多指节接触不会使动力学公式失效。需要改变的是：从单指尖 Jacobian 扩展为近段、中段、远端的堆叠多接触 Jacobian，并避免将三个接触控制器当作互不相关的独立系统。

第一版应优先证明“归一化 FSR + 多区域联合导纳 + 关节阻抗”能够在不依赖精确力标定的情况下稳定工作。随后再逐步加入电流力矩估计和指尖 6D wrench 融合。
