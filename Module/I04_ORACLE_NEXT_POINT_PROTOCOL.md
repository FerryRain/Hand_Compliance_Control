# I04 — Oracle Next-Point Whole-Hand Contact Traversal

核心协议冻结日期：2026-08-24。实现状态更新：2026-08-25。

## 状态与授权边界

- Module ID：`I04-ORACLE-NEXT-POINT-BUNNY-v1`
- 核心协议状态：`CORE_PROTOCOL_FROZEN`
- Explicit MCC 数值/实现状态：`DEVELOPMENT_V1 / PHYSICAL_REGRESSION_INCOMPLETE`
- DPRef 分支状态：`NOT_STARTED`
- 完整 274-goal traversal：`NOT_COMPLETED`
- 2026-08-24 的原始授权只冻结 I04 的问题、输入、方法边界、实测判据和诊断量；
  2026-08-25 已形成 Explicit MCC development implementation 与三轮物理回归。当前发布不把
  development 数值写成最终正式协议，不授权 DPRef 训练、GPIS 集成、Gate 变更，也不把局部
  回归写成完整 Bunny traversal 成功。当前实现、结果和续作阻塞点见
  [`I04_RESUME_CHECKPOINT_2026-08-25.md`](I04_RESUME_CHECKPOINT_2026-08-25.md)。

I04 的唯一问题是：

> Given a good next surface point, can the hand physically traverse to it while preserving contact?

I04 不回答“哪个未知区域最有信息”。它在完整已知 Bunny mesh 上维护确定性的 traversal-completion
ledger，并依据 barrier 后的真实手状态选择下一个可行 surface goal。它不包含 uncertainty、GPIS
frontier、information gain、next-best-touch selection 或 reconstruction；这些属于后续 GPIS
active exploration。

## 冻结的核心设计选择

Oracle 指定 surface goal，但不指定 target finger：

```text
g_k = (x_k*, n_k*, t_k_out, epsilon_g, epsilon_n)
```

其中 `x_k*` 是 Bunny 表面目标点，`n_k*` 是目标表面法向，`t_k_out` 是到达后的建议离开切向；
`epsilon_g` 和 `epsilon_n` 的数值在正式数值协议中冻结。Oracle 输入中禁止出现 target finger
ID、finger role、contact mode、MAKE/BREAK sequence 或 privileged witness configuration。

`t_k_out` 是两个方法都能读取的 controller input，用来给出连续 traversal 的后继方向；它不
属于 arrival condition，不能把 I04 变成额外的 fingertip 姿态跟踪实验。

成功到达只要求存在某根真实手指完成目标：

```text
exists i: real fingertip i reaches g_k
```

选择 explorer finger、anchor finger、contact mode 和 handover sequence 是被测方法的职责。

## 固定 Bunny surface graph 与实测状态条件目标

固定的是完整 surface graph、必须访问的目标集合和完成条件，不是脱离物理状态的 waypoint
顺序：

```text
z_barrier,k = (X_H, q, dq, f_tip, c_tip, A_actual, E_fail)
U_k         = required but not-yet-ARRIVED surface nodes
g_k         = OracleRoute(G_surface, U_k, z_barrier,k)
```

- `G_surface` 使用完整 Bunny triangle mesh，而不是 I01–I03 的 upper-envelope height field；
- surface samples、法向、面积/曲率信息、必访节点和邻接边来自同一个版本化 mesh fixture；
- 相邻候选按 mesh geodesic connectivity 构造，不能用普通 Euclidean proximity 跨越耳朵两侧、
  薄壳两面或其他测地不相邻区域；
- 每次 goal selection 必须读取 M06 micro-barrier 后的真实 palm pose、joint state、真实 contact
  set、force 和 failure evidence，不能从上一轮 prediction suffix 假设当前手的位置；
- 优先选择邻接且未访问的 privileged-feasible node；局部没有新节点时可以沿已访问的 certified
  bridge/backtracking node 调整，但这些重复节点不计为新增完成；
- target selector 不得永久丢弃难区域。完整 route graph 的每个 required node 都要保持在 `U_k`
  中，直到真实 ARRIVE；只有 `U_k` 为空才算完成整只 Bunny 的 certified reachable surface；
- 只有当前 `g_k` 的实测 ARRIVE 被确认后才更新 ledger 并选择下一个 goal。timeout/recovery
  行为及预算仍属于待冻结数值层，不能在方法之间使用不同规则。

两个方法共享完全相同的 `G_surface`、required set、OracleRoute 算法、初态、object pose 和预算。
因为闭环真实状态可能分叉，实际目标顺序允许随各自 `z_barrier,k` 改变；必须记录每个 goal 的
geodesic step、曲率、feasibility margin 和 bridge/new-node 属性，避免把不同目标难度隐藏起来。
这仍是已知几何上的 state-conditioned traversal planning，不是未知几何的信息探索。

## Privileged feasibility certification

正式 surface graph 在交给被测方法前，必须用完整 Bunny geometry 做离线 privileged
certification。每个 required node 至少存在某个合法 whole-hand configuration 和某根可达手指：

```text
exists (q, X_H, i):
  fingertip_i(q, X_H) reaches x_k*
  and joint/collision/contact constraints are valid
```

相邻 graph edge 还必须通过 route-level connectivity certification。在线选择 `g_k` 时，要基于
当前 `z_barrier,k` 再确认至少存在一个从真实 root 出发的合法到达 realization；不能只依赖
与当前手姿态无关的 pointwise IK：

```text
exists (Pi, contact-mode sequence, i):
  Root(Pi) = z_barrier,k
  and fingertip_i reaches g_k
  and hand-level contact remains nonempty
  and joint/collision/contact constraints are valid
```

正式数值协议需冻结 online feasibility、edge margin 和证书 schema。证书可以在 privileged
内部保存 witness finger/configuration/contact sequence 以便审计，但 runtime goal adapter 必须
剥离这些字段，只向被测方法输出 `g_k`。

因此：

```text
Oracle feasibility certificate != Oracle finger assignment
```

## 实测 ARRIVE 判据

普通三维 Euclidean distance 不能作为 arrival 判据。令 `x_i_real` 和 `n_i_real` 为 MuJoCo
真实 fingertip--Bunny contact 在完整 mesh 上的接触点和法向，则：

```text
ARRIVE(g_k) iff exists i:
  c_i_real = 1
  and d_S(x_i_real, x_k*) <= epsilon_g
  and dot(n_i_real, n_k*) >= cos(epsilon_n)
```

其中 `d_S` 是同一 Bunny mesh 上的 surface geodesic distance。接触点、法向和 contact flag
全部来自物理引擎的真实 contact；planned contact、expected role、DPRef prediction、prediction
suffix 或 privileged certificate witness 都不能补入 ARRIVE。

若多个手指在同一确认窗口到达，ARRIVE 仍只发布一次，但所有参与手指都保留在诊断日志中。

## Hand-level continuous contact

定义：

```text
c_H(t) = 1[sum_i c_i_real(t) >= 1]
```

I04 允许任意非空 fingertip contact set；多指同时接触是性能/解释量，不是合法性的硬要求。
至少报告：

```text
R_contact = (1 / T) * integral c_H(t) dt
T_gap_max = max_j(t_recover_j - t_loss_j)
```

还要保存每次 contact-loss/recovery interval，而不只报告是否曾经断触。

## 共享 MAKE-before-BREAK runtime guard

两个方法必须共享同一个 measured-contact runtime guard、transactional prefix execution、真实
状态 micro-barrier、MCC、force coordinator 和 hard safety guards。

如果 finger `i` 是当前最后一个真实 contact，方法不能仅凭预测的 MAKE 或 role intention 释放
它。只有 replacement finger `j` 的真实 contact（以及待冻结的 force/time confirmation）被确认
后，BREAK/RELEASE(`i`) 才合法：

```text
MAKE(j)
 -> measured contact/force confirmation for j
 -> BREAK or RELEASE(i)
```

显式规划器与 DPRef/Role 都不能绕过该共享 guard；下一 transaction root 只能来自 barrier 后的
真实状态。

## I01–I03 与 M01–M12 的累计集成

I04 不是另起一套 waypoint follower，而是在完整 Bunny mesh 上累计集成已有模块：

```text
M01 full-mesh SurfaceModel + current measured barrier state + remaining-node ledger
 -> state-conditioned no-finger Oracle goal g_k
 -> method-specific finger/contact realization proposal
 -> M10 exact committed-prefix audit
 -> M06 transactional execution
 -> M02/M04 shared Wrist/Finger MCC + M03 runtime guards
 -> real contact ARRIVE / barrier snapshot / replan
```

- I01 的贡献：允许 variable nonempty contact modes，并用真实 MAKE-before-BREAK handover 连续移动；
- I02 的贡献：只执行短 committed prefix，每个 barrier 后从当前真实手位置重新规划；
- I03 的贡献：在提交前检查 terminal continuation，避免为了当前 waypoint 把手送入无后继状态；
- Explicit 分支必须使用完整 `M07 -> M08 -> M09 -> M11/M12 -> M10 -> M06` 链；
- DPRef/Role 分支提出 finger/reference/contact-transition realization，但 committed command 仍必须
  经过共同的 M10 execution audit、M06 transaction/barrier、MCC 和 M03 guards。

M07–M09/M11–M12 在 DPRef 分支中是 shared feasibility/viability filter、diagnostic-only，还是由
DPRef proposal path 完全替代，尚未由本次消息完整冻结；正式 method-boundary protocol 必须明确
这一点，不能暗中让 Explicit planner 替 DPRef 选择 finger。

## Paired methods 与唯一比较变量

```text
I04-EXPLICIT:
  g_k from the same state-conditioned OracleRoute contract
  -> Explicit contact-mode/finger planner
  -> shared transaction/barrier/MCC/runtime guards

I04-DPREF:
  g_k from the same state-conditioned OracleRoute contract
  -> DPRef nominal trajectory + Role intention
  -> shared transaction/barrier/MCC/runtime guards
```

两边共享 SurfaceModel/surface graph、remaining-node semantics、OracleRoute/goal adapter、paired
initial state、wrist motion budget、contact/force definition、MCC gains、guards、evaluator 和
compute accounting。Explicit 分支决定显式 finger assignment/contact mode；DPRef 分支隐式产生
finger assignment/contact transition。禁止 Oracle 把 finger identity 泄漏给任一分支，也禁止
两种方法互相作为 runtime fallback。闭环分叉后不强迫两边使用与各自真实手状态不相容的固定
目标顺序，而是记录共同 selector 在各自状态上产生的全部 goal 与难度证据。

DPRef 的 goal-conditioning schema、frame、checkpoint 和训练数据尚未冻结；它们必须在正式实现
前单独登记。现有或未来 DPRef 均不能因为读取了与 Explicit 不同的 goal 信息而获得优势。

## Finger-assignment diagnostics

定义：

```text
A_ik = 1[waypoint k was reached by finger i]
```

若多个手指共同触发 ARRIVE，允许同一列有多个 `1`。至少记录：

- 每根手指完成的 waypoint 数与连续承担长度；
- explorer/anchor role occupancy；
- finger switch、MAKE、BREAK/RELEASE 和完整 handover sequence；
- handover 发生位置、局部曲率、joint/workspace margin；
- 是否长期只使用同一根手指，以及 workspace 接近极限后的接管行为。

这些量用于解释 whole-hand contact realization behavior，不作为 Oracle 输入或隐藏约束。

## 比较指标与报告方式

I04 只报告连续性能和相对优劣，不使用 `MET/NOT_MET` 或策略 pass/fail 标签。至少比较：

- route completion fraction、连续完成 waypoint 数和最长 uninterrupted geodesic progress；
- 每 waypoint arrival time、arrival geodesic error 和 timeout/recovery；
- `R_contact`、`T_gap_max` 和全部 gap duration distribution；
- MAKE/RELEASE/handover 数量、成功行为和 finger-assignment diagnostics；
- fingertip force、collision、joint/workspace margin、guard/SAFE_HOLD 事件；
- planner/optimizer/audit/DPRef inference/controller 的 mean/P95 latency 与 compute budget；
- transactional certificate、barrier、real-root 与 command-authority provenance。

若一个方法到达更多点但接触中断、力或计算代价更高，报告该 trade-off，不强行压缩成单一
Gate verdict。

## 正式实现前仍需冻结的数值层

以下内容没有在本次核心冻结中擅自决定：

- Bunny scale、pose、gravity/support、完整 mesh 版本及 certified reachable surface subset；
- sampling density、required-node set、state-conditioned OracleRoute policy、route-completion ledger、
  bridge/backtracking 规则、route length accounting 和 geodesic implementation；
- `epsilon_g`、`epsilon_n`、contact/force confirmation time；
- per-waypoint timeout、recovery、episode duration、seeds 和 paired initial states；
- privileged point/edge certificate margins；
- common goal frame/schema、wrist plan contract、DPRef goal conditioning，以及 DPRef 分支中
  M08/M12 的 filter/diagnostic 权限；
- DPRef training/checkpoint provenance、formal cells、trace schema 和可视化输出。

这些数值与实现细节必须在任何 I04 正式 episode 之前追加为版本化 numerical protocol；不得
回改本文件已经冻结的 no-target-finger、geodesic ARRIVE、privileged-but-hidden feasibility、
measured hand contact、shared MAKE-before-BREAK 和非探索边界。
