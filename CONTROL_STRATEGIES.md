# 两版底层控制策略

本文档定义主方法和显式优化 Baseline 的底层控制结构。后续 session 在修改
控制器、数据格式或实验对比前，必须先读 `PROPOSAL.md` 和本文档。

## MCC 的准确含义

Minimalist Compliance Control（MCC）的核心不是“必须读取电机电流”，而是：

> 使用测得或估计的外力，驱动一个虚拟质量–阻尼–刚度系统，在线积分得到
> 柔顺参考位姿，再通过 IK/Jacobian 生成位置命令。

原实现因为缺少直接力传感器，才采用：

```text
motor current/torque
→ remove model bias and filter
→ Jacobian wrench estimation
→ compliance-reference integration
```

传感器来源不是 MCC 的定义。若已有可靠的 wrist wrench 或真实 fingertip
force，应直接使用，不必再经过不稳定的舵机电流、关节力矩和 Jacobian 反推。

灵巧手电机电流同时包含关节/齿轮摩擦、回差、电流量化、温度、内部位置环、
多关节耦合、线缆/软材料变形和未建模惯性，因此不应把
`motor current ∝ fingertip contact force` 作为两版方案的默认假设。

## 方案一：Finger DP + Wrist MCC（主方法）

控制链：

```text
ER-GPIS wrist-only planning
├─ planned wrist trajectory + measured wrist wrench → Wrist MCC → arm command
└─ planned wrist trajectory + finger state + real fingertip force/tactile
   → Finger DP → finger joint command
```

### 上层输入

ER-GPIS 只输出短时 wrist trajectory，不输出四条显式 fingertip trajectory。
它只需执行机械臂 workspace、collision、velocity 和 joint-limit 等轻量检查。

### Wrist MCC

Wrist MCC 维护柔顺参考状态 `X_H_ref`、`dX_H_ref`：

```text
M_H ddX_H_ref
+ D_H dX_H_ref
+ K_H (X_H_ref - X_H_plan)
= W_H_meas + W_H_cmd
```

积分后的 `X_H_ref` 通过 arm IK 转成关节位置命令。`W_H_meas` 优先来自可靠的
机械臂外力估计或六维 F/T 传感器；必要时才考虑由四指力合成等效 hand wrench。

### Finger DP

建议观测至少包括：

- finger joint position/velocity；
- 四个真实 fingertip force；
- 四指 contact validity；
- tactile image/features（若存在）；
- 历史 finger action；
- 当前及未来 wrist-motion chunk。

DP 输出 finger joint trajectory 或 position increments，并采用 action chunking
与 receding-horizon execution。

真实 fingertip force 在方案一中用于：

1. 判断接触是否存在；
2. 判断接触过强或过弱；
3. 帮助策略判断 wrist 运动后应闭合还是释放；
4. 触发独立的简单安全保护。

### 不叠加 Finger MCC

方案一的 finger command 只由 DP 生成。不得默认执行：

```text
q_f_cmd = q_f_DP + delta_q_f_MCC
```

因为 DP 和 Finger MCC 若同时对相同的 fingertip force 作出反应，会造成过补偿、
抖动、动作语义改变和训练/部署分布偏移。

允许的额外层只是安全 guard，例如力超过硬上限时停止继续闭合、轻微释放或进入
Force Release 状态；这不是第二个完整的手指 MCC。

## 方案二：4× Fingertip-Force MCC + Wrist MCC（Baseline 2）

控制链：

```text
ER-GPIS whole-hand optimizer
→ planned wrist trajectory + four planned fingertip Cartesian trajectories
├─ wrist plan + measured wrist wrench → Wrist MCC → arm command
└─ fingertip plan_i + real fingertip force_i
   → force-based Finger MCC_i
   → corrected Cartesian target_i
   → finger IK/Jacobian
   → finger command_i
```

上层显式优化 wrist 与四个 fingertips。计算瓶颈来自上层多指优化，不来自简单的
底层 admittance。

### Finger MCC 的测量输入

每根手指直接使用真实 fingertip force。默认不使用舵机电流估计接触力。

若物体表面外法向为 `n_i`，需要先通过标定确定力传感器坐标和符号，然后计算：

```text
f_n_i = -n_iᵀ F_tip_i
e_f_i = f_n_i_des - f_n_i
```

符号必须通过接触实验校准，不得仅凭模型坐标系假设。

### 法向 Cartesian admittance

第一版 Finger MCC 只修改规划目标的表面法向分量，不任意改变切向探索轨迹：

```text
m_f dd(delta_d_i)
+ d_f d(delta_d_i)
+ k_r delta_d_i
= k_f e_f_i
```

离散积分：

```text
d(delta_d_i)[t+1] =
    d(delta_d_i)[t] + dd(delta_d_i)[t] dt

delta_d_i[t+1] =
    delta_d_i[t] + d(delta_d_i)[t+1] dt
```

根据经标定的法向符号生成：

```text
p_i_cmd = p_i_plan ± delta_d_i n_i
```

再执行：

```text
p_i_cmd → IK_i → q_i_cmd
```

或：

```text
dq_i_cmd = J_i# (dp_i_plan ± d(delta_d_i) n_i)
```

`delta_d_i`、速度、加速度和单步关节命令必须限幅，并保留 anti-windup、
滤波、接触 hysteresis、关节限制和碰撞安全检查。

### 为什么默认只控制法向

上层 optimizer 已决定沿表面向哪里探索；Finger MCC 只负责保持适当压入力。
若把完整三维力送入各向同性 compliance，切向摩擦也会改变目标，破坏上层的信息
采集轨迹。因此 Baseline 第一版使用标量法向力；三维 compliance 只能作为单独
消融，不能替代默认版本。

### Wrist MCC 的 wrench 来源

优先级建议：

1. wrist 六维 F/T 传感器；
2. 可靠的机械臂 joint-torque 外力估计；
3. 四个真实 fingertip force 合成的等效 hand wrench。

第三种情况下：

```text
F_H = Σ R_(H←i) F_tip_i
tau_H = Σ (p_i - p_H) × R_(H←i) F_tip_i
W_H = [F_H, tau_H]
```

第一版 Baseline 若已有可靠 wrist wrench，应优先直接使用，避免额外的坐标变换
和接触点位置误差。

### Finger/Wrist MCC 带宽分离

四个局部 Finger MCC 和整体 Wrist MCC 可能对同一力变化同时退让，造成接触
丢失。第一版至少使用简单的职责和带宽分离：

- Finger MCC：局部、高频、单指曲率和法向力变化，约 `50–100 Hz`；
- Wrist MCC：整体、低频 hand wrench 与腕部轨迹误差，约 `10–30 Hz`；
- wrist compliance gain 更小/更软，避免和四个手指闭环同时放大；
- 对共同模式力变化增加退让限幅或仲裁，避免四指与 wrist 同时过度释放。

## 两版公平对比

| 项目 | 方案一：主方法 | 方案二：Baseline |
| --- | --- | --- |
| Wrist 控制 | Wrist MCC | Wrist MCC |
| Wrist 参考 | ER-GPIS wrist-only 轨迹 | 显式 whole-hand optimizer 轨迹 |
| Finger 参考 | 无显式 fingertip 轨迹 | 四条 Cartesian fingertip 轨迹 |
| Finger 力反馈 | 真实指尖力进入 DP | 真实指尖力进入 Finger MCC |
| Finger 动作生成 | Diffusion Policy | 法向 Cartesian admittance + IK |
| 手指舵机电流估力 | 不使用 | 不使用 |
| 多指协调 | 离线学习、在线隐式生成 | 在线显式优化 |
| 主要计算成本 | DP inference | whole-hand optimization |

最准确的实验问题是：

> Learned fingertip-force-conditioned compliance 与 analytical
> fingertip-force-based compliance，在接触保持、探索质量、泛化性和在线延迟上
> 如何权衡？

## 实现与验收约束

- 两版共享相同 wrist MCC、机器人模型、对象、wrist trajectory budget 和安全
  门槛，减少不公平变量。
- 指尖接触必须由四个物理 fingertip pad 传感器独立统计；指甲、手背、掌部接触
  不得冒充。
- 允许短时失联或停顿，但大部分移动时间应由大部分指尖接触，并在恢复状态后
  重建稳定支撑。
- FR3/object contact 始终是硬失败；LEAP Hand 附带接触执行独立的力和穿透限制。
- Baseline 2A 允许优化充分收敛；2B 使用 50/100/200 ms 时间预算。
- 两版都报告接触保持率、平均接触数、每指接触率、力稳定性、滑移、恢复、
  joint margin、碰撞、mean/P95/max latency 和 deadline miss rate。
