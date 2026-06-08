# Hand Contact re\-location

> Process：
> 
> Oct 28 \- Nov 03： 
> 
> 
> 
> 

Project Source Code: 



# Intro



在真实世界的接触丰富操作中，机器人手常常需要在保持连续、多点接触的同时，实现手掌与手指相对于物体表面的有目的运动。这类在接触约束下进行的表面相对运动，是多种灵巧操作任务的基础能力，例如对曲面物体进行擦拭清洁、沿表面扫描以获取几何信息，以及在不重新抓取的情况下完成在手重定位。尽管近年来灵巧手硬件与触觉感知技术取得了显著进展，在真实系统中稳定实现“边移动边保持全手接触”仍然面临显著挑战。

这一困难不仅来自局部接触控制本身，更源于多点接触系统所带来的强耦合与协同复杂性。当手在物体表面发生滑动或滚动时，控制器不仅需要在每个接触点上稳定法向交互，以避免局部脱离或过度挤压，还需要在切向方向生成协调一致的相对运动，以实现整体接触重定位。与此同时，系统还必须处理粘滑切换、摩擦参数不确定、接触几何随运动持续变化，以及多指之间通过物体形成的闭链耦合约束。对于全手接触场景而言，问题因此不再是单一接触点的稳定控制，而是如何在不断演化的接触拓扑下，协调手掌与多根手指的运动与受力分配，使系统沿着可行的接触流形持续推进。该过程同时涉及接触模式切换、运动学冗余分配与多接触约束下的动作规划，使得纯基于模型的方法往往对建模误差敏感，并需要针对不同物体、摩擦条件与接触构型进行大量调参与工程化适配。

Learning\-based 方法为这一问题提供了另一条可能路径，但在接触丰富的多指操作中，其应用仍受到训练数据来源的根本限制。一方面，面向真实接触的高保真触觉仿真仍然不足，多点接触、摩擦滑移与柔顺形变在仿真中往往难以稳定复现，导致基于仿真的策略学习难以获得可靠、可迁移的行为。另一方面，若直接依赖真实世界示教，高质量的“全手接触移动”数据本身又极难规模化采集：多点接触下的遥操作不够稳定，手掌与手指沿物体表面的连续运动也难以通过直接牵引示教高效获得。因此，现有方法面临的关键瓶颈并不仅仅是控制或学习算法本身，而在于缺乏一种能够低成本、规模化构造“保持接触的表面运动”监督信号的数据途径。

上面都是讨论性叙事，缺乏基于文献的研究现状。

为此，我们提出一种顺应控制引导的学习框架，其核心并非直接示教复杂的全手接触技能，而是通过一种更易实现的交互过程，合成出原本难以获取的监督信号。我们的基本观察是：相比于直接采集“手在物体表面运动”的示教轨迹，利用顺应控制维持稳定接触，并执行简单、可重复的物体运动原语，要更加容易、稳定且高吞吐。基于这一观察，我们首先设计低层顺应控制器，使系统优先稳定法向接触，同时在切向方向允许受约束的滑移或滚动，从而能够在真实交互中高效采集长时序、连续接触的运动轨迹。在此基础上，我们进一步将这些易获取的交互轨迹系统性地转换为等价的“手相对于物体表面运动”监督信号，并据此训练一个保持接触的运动模型。这样一来，困难的接触技能学习被转化为一个由顺应交互驱动的数据合成问题：顺应控制负责维持接触、吸收局部几何与摩擦扰动，学习模块则利用合成监督在接触流形上生成有效运动，从而在显著降低示教成本的同时提升真实执行中的鲁棒性



Our main contributions are summarized as follows:

- 提出一种顺应控制引导的数据采集策略，通过在稳定多点接触的同时允许沿物体表面的切向运动，实现全手连续接触轨迹的高效采集。

- 提出一种轨迹转换方法，将易于采集的顺应交互轨迹转换为保持接触的手\-\-物体表面相对运动监督信号，从而避免直接示教难以获取的接触保持运动。

- 基于转换后的监督信号训练接触保持运动模型，并在轨迹复现、接触重定位和曲面擦拭任务中验证其有效性。

# Related works

Force Policy: Learning Hybrid Force\-Position Control Policy under Interaction Frame for Contact\-Rich Manipulation
FoAR: Force\-Aware Reactive Policy for Contact\-Rich Robotic Manipulation



（请在做method的过程中慢慢积累）

# Methodology

## Module 1：Compliance Contact Controller

Module 1 是一个 **统一的接触保持与柔顺抓握控制层**：通过触觉闭环生成抓握补偿 $ \Delta \mathbf{q}_{comp} $*，并用慢更新的参考 *$\mathbf{q}_{ref}$抑制冗余漂移；上层 DP 只输出 residual 动作$ \Delta \mathbf{q}_{DP} $，在采集/执行两种模式下仅调节增益与触觉目标区间。

在物体尽量不动的前提下，Module 1 负责 **稳定抓握 \+ 保持持续接触**，把上层 DP 的任务从既要抓稳又要走到目标点解耦为主要管怎么走，从而避免手指瞎动、脱离接触或过度夹紧。

### 接口定义

### 输入

- 手指状态：当前关节$\mathbf{q}_t $ 、速度 $ \dot{\mathbf{q}}_t$

- 触觉：每个 body / 每指的 FSR 读数$\mathbf{s}_t$\(可合成每指标量 $ s_i $）

（$s_i=\omega⋅mean(s_{proximal})+(1−\omega)⋅mean(s_{distal})$?\)

- 上层动作：DP 输出 $\Delta \mathbf{q}_{DP,t}$

- 腕部力/接触力：$f_n$

### 输出

- 手指关节命令：$ \mathbf{q}_{cmd,t}$

- 手掌/机械臂末端命令：笛卡尔阻抗目标$\mathbf{x}^{\star}_t$或速度/力命令

### 手指端compliance

#### a\. 总目标

#### $\mathbf{q}_{cmd,t} = \mathbf{q}_{ref,t} + \Delta \mathbf{q}_{DP,t} + \Delta \mathbf{q}_{comp,t}$

- $ \Delta \mathbf{q}_{DP,t} $：DP输出的指令

- $ \Delta \mathbf{q}_{comp,t} $：Compliance control补偿（防掉、防夹死、抑制抖动）

- $\mathbf{q}_{ref,t}$：稳定接触姿势

#### B\. 触觉补偿：把每指 FSR 维持在区间 \[$s_{\min}, s_{\max}$\]

对每根手指 $i$聚合触觉$s_i$，定义contact：

$e_i =
\begin{cases}
s_{\min}-s_i, & s_i < s_{\min}\\
0, & s_{\min}\le s_i \le s_{\max}\\
s_{\max}-s_i, & s_i > s_{\max}
\end{cases}$

然后生成补偿并映射到该指的末端位置：

$\Delta \mathbf{q}_{comp,i} = \mathrm{clip}\left(\mathbf{K}_{s,i} e_i - \mathbf{D}_{s,i}\dot{s}_i\right)$
*拼接得到整手 *$\Delta \mathbf{q}_{comp,t} $。

为了保持约束还需要加上：

- **rate limit**（每步最大关节变化）

- **饱和**（避免补偿无限加）

- **优先级**：补偿项只作用在“抓握主关节子集”，尽量不干扰任务关节

#### C\. $ \mathbf{q}_{ref}$：用上一帧状态做慢更新锚点

避免每步跳变，同时避免长时间漂到关节极限。

- **接触稳定时**更新：
$\mathbf{q}_{ref,t+1} \leftarrow (1-\alpha)\mathbf{q}_{ref,t} + \alpha \mathbf{q}_t$

- **接触不稳定时**冻结：
$\mathbf{q}_{ref,t+1} \leftarrow \mathbf{q}_{ref,t}$

- **防漂移正则**$ \mathbf{q}_{nom} $*：*
$\mathbf{q}_{ref,t+1} \leftarrow \mathbf{q}_{ref,t+1} + \beta(\mathbf{q}_{nom}-\mathbf{q}_{ref,t+1})$

### 手掌/机械臂末端：现成柔顺控制\(实机用仿真不用MAYBE\)

如果你需要“贴着表面走”，必须在手掌/末端做笛卡尔柔顺：

- 切向：跟踪$\mathbf{x}^{\star}_{tan} $

- 法向：$x^{\star}_n \leftarrow x^{\star}_n + k_f(f_n^\star - f_n)$

这样避免“硬位置推挤”导致顶飞/失去接触。

### 两种模式

#### 模式 A：采集数据（Grasp\-Maintain）

- $\Delta \mathbf{q}_{DP} \approx 0 $

- $s_{\min}$ 稍大

- $ \mathbf{K}_s $较大（更强抓握保持）

#### 模式 B：策略执行（Task\-Execution）

- $ \Delta \mathbf{q}_{DP}$为主要驱动

- 触觉区间更“宽松”（避免补偿压制任务）

- $ \mathbf{K}_s $较小 \+ 更强 rate limit（防 DP 与补偿打架）

- 手掌柔顺启用





\[Screencast from 2026年03月30日 16时17分04秒\.webm\]

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDJkOGM2MjY1NGVlMzAzYzNlMmE4ZGI3ZjgwMzA1ZjlfOTljMWU0YjRkN2M5NDY3MjMxYWI0N2I2MWJkOTM3ZjFfSUQ6NzYyMzMzMzA2ODMzMzUwMTM5NF8xNzgwMzg0MTQ0OjE3ODA0NzA1NDRfVjM)

## Module 2：Compliance\-guided Data Collection \+ Trajectory Inversion

在module1的作用下，保持接触的情况下执行简单物体运动采集“稳定接触下的交互轨迹”，再将轨迹反演为“手相对物体沿表面移动”的示范序列。

## Module 3：Goal\-conditioned Diffusion Policy

输入：目标点（物体表面点，基于 pose 转换）\+ 接触状态（FSR/F/T）\+ 机器人状态；

输出：下一步控制指令（EE 的增量 \+ 手关节残差/抓持强度）；

由 Module 1 执行并稳定接触，并对其施加力

还有一个问题：

这个项目和希哥的思路相似，靠手物相对运动关系出运动学轨迹、以及力觉交互数据。用什么样的传感器去做这个任务？三维FT sensor或者coinFT吗？华立创传感器？

另外一个可能性就是，不管sim2real了。直接在实物里采集数据、实物里跑。实物里用机械臂带动一个物体乱转。

# Experiments

## Experimental Setup

仿真轨迹的定性展示、力觉数据展示；实物轨迹的定性展示、moveto到一些pose的定量精度；一些下游任务，比如exploration、wiping。

问题：有没有baseline可以比？

## Trajectory Inversion Validation

### Proof: 

在当前pipeline下，**物体相对手**的运动轨迹易采且接触稳定；

对轨迹做反演后，能在“物体固定、手运动”的设置下**等价复现**同样的接触迁移，并保持接触质量。

### 实现：

**A\. Demonstration \(easy\-to\-collect\)\.**

让物体相对手执行若干标准 primitive（每个 10–20 s）

- 绕 xyz各个轴旋转（不同角速度）

- 小幅平移 \+ 旋转复合

- 在compliance control下采集 $T_{HO}(t)、f_n(t)、c(t) $等。

**B\. Replay **

固定物体位姿 $T_{WO}^{\text{fix}}$，让手执行反演轨迹$T_{WH}^{\text{replay}}$ 。

## Goal\-Directed Contact Relocalization

### Proof:

证明训练出来的 model（可以尝试用ACP）不仅能 replay，而是能在闭环里实现：
**在保持接触的前提下，把接触 patch 从当前区域移动到目标区域**（接触重定位）。

### 实现：

在物体表面定义目标区域 $\mathcal{R}^\star$。成功条件：接触中心进入 $\mathcal{R}^\star$ 并保持$\Delta t$ \(如 1 s）。

## Downstream Task — Vase Wiping

### Proof:

证明这个接触的任务是有用的，是一个**可迁移的 manipulation primitive**，能显著提升embody任务的成功率：**曲面擦拭覆盖**与**接触力稳定性**。

