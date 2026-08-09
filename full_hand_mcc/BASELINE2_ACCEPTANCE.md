# Baseline 2 项目级验收规范

本文档把 `PROPOSAL.md` 中的方案二固化为可重复执行的验收协议。它适用于
`full_hand_mcc/` 的 FR3 + LEAP 显式 whole-hand optimization baseline，
不适用于 proposal 主方法的 Finger DP 训练或 wrist-only ER-GPIS 性能结论。
控制边界以 [`../PROPOSAL.md`](../PROPOSAL.md) 和
[`../CONTROL_STRATEGIES.md`](../CONTROL_STRATEGIES.md) 为准；执行状态以根/子
`PROCESS.md` 的最新检查点为准。

验收按五级递进：

1. Level 1：底层真实力闭环；
2. Level 2：单物体完整长程；
3. Level 3：Optimization Oracle 与 50/100/200 ms time-capped variants；
4. Level 4：多物体、多尺度、多摩擦、多姿态泛化；
5. Level 5：数值与视觉均通过的正式视频交付。

低等级通过不能替代高等级。例如 5 mm smoke 只能支持 Level 1，不能证明
0.48 m、顶部到达、实时优化、泛化或视频交付。

## 1. 不可改变的控制边界

- 上层 optimizer 输出 wrist pose trajectory 和四条 Cartesian fingertip
  trajectories；五个规划点是 palm-root guide 加四个物理 fingertip pads。
- palm-root 只提供大致方向，不需要位于物体表面，也不使用旧的 3 mm palm
  可行球作为项目成功条件。掌部可发生受控接触，但不计入四指接触率。
- 四个 Finger MCC 分别直接使用四路真实 fingertip force，只修改各自目标的
  local surface normal 分量；上层的 tangential trajectory 必须保持不变。
- Wrist MCC 使用 wrist wrench 或可靠的 FR3 external-joint-torque wrench
  estimate，带宽和增益低于 Finger MCC。
- LEAP motor load 只用于 actuator diagnostic/safety，不能作为四路 primary
  fingertip-force feedback。
- FR3 的任何 link 与目标物体接触均为硬失败。LEAP 的非指腹接触可以存在，
  但必须受力/穿透限制且不得主导动作。
- 物体在本 baseline 的第一阶段保持固定；物体运动不属于本规范。

## 2. 统一测量定义

### 2.1 评估窗口

`motion evaluation window` 从 loaded four-pad force setpoint 标定完成且正式
surface motion 开始的第一帧起，到 commanded route 第一次到达终点的帧止。
所有 contact ratio、average contact 和 loss-window 指标只在该运动窗口统计；
终点后的 hold 不能抬高路线接触率。`terminal recovery window` 单独统计末段
连续 `4/4`。approach、初始接触搜索和标定帧不计入接触保持率，但整个 rollout
的碰撞、过力、穿透、NaN/Inf 和 joint-limit safety guard 始终有效。

默认 Finger MCC 为 100 Hz。本文写成“20 帧/0.20 s”的窗口在改变控制频率时
必须按时间换算并向上取整，不能通过降采样缩短真实约束时间。

### 2.2 物理指腹接触

对第 `i` 个手指：

```text
c_i(t) = 1
```

仅当 contact pair 包含该手指指定的 physical fingertip-pad geometry、目标物体，
并且经符号校准的 normal force 不低于 `0.10 N`。指甲、指背、指节、掌部或
FR3 的接触不能替代 `c_i`。

```text
C(t) = c_index + c_middle + c_ring + c_thumb
majority_contact_ratio = mean(C(t) >= 3)
average_contact_count = mean(C(t))
per_finger_contact_ratio_i = mean(c_i(t))
```

### 2.3 路线进度

每个物体先定义一条带方向的 surface route coordinate `s in [0, L]`。四指各自
的 planned/actual pad contact point 投影到同一物体坐标系和同一方向的 `s`。
不能用 FR3 wrist 位移、palm 位移或视频像素位移代替 fingertip progress。

每帧需同时保存：planned `s`、actual `s`、surface geodesic travel、world
travel、tip-relative-to-palm travel 和 tangential cross-track error。MuJoCo
contact slot 跳变不能制造虚假 travel；实际行程以稳定 body-fixed physical pad
site 为主，contact sensor 只证明是否接触。

## 3. 所有动态等级共用的硬门槛

下表是成功门槛，不是建议值。任意一项失败，该 seed/run 即失败。安全停止是
正确的故障处理，但仍不能把该次任务标为成功。

| 类别 | 硬门槛 |
| --- | --- |
| Majority contact | `majority_contact_ratio >= 0.80`，即至少三指接触的帧不少于 80% |
| Average contact | `average_contact_count >= 3.00/4` |
| Per-finger contact | index/middle/ring/thumb 均 `>= 0.75`，不允许用其他三指平均掩盖一指长期离面 |
| Brief majority loss | 连续 `C(t) < 3` 不超过 `0.20 s`（100 Hz 时 20 帧） |
| Brief total release | 连续 `C(t) = 0` 不超过 `0.10 s`（100 Hz 时 10 帧），且之后必须恢复 |
| Individual recovery | 任一指连续离面不超过 `0.20 s`（100 Hz 时 20 帧） |
| Terminal recovery | Level 1 末段连续 `4/4 >= 0.20 s`；Level 2–5 连续 `4/4 >= 0.50 s` |
| Filtered fingertip force | 每指 filtered normal force 始终 `< 25 N`；达到或超过即失败 |
| Raw fingertip force | 每指 raw 3-D magnitude 始终 `< 40 N`；达到或超过即 emergency abort + 失败 |
| FR3/object | runtime contact frames `= 0`；所有插值规划帧 minimum clearance `>= 2.0 mm` |
| Fingertip penetration | 每个 pad/object maximum penetration `<= 1.0 mm` |
| Other LEAP contact | 单个 geometry force `< 24 N`、同帧合力 `< 36 N`、penetration `<= 1.0 mm` |
| Other-contact dominance | 全程非指腹接触冲量 / 四指接触冲量 `<= 0.20`；非指腹接触不计入 `C(t)` |
| Self-collision | runtime significant self-collision count `= 0`，numerical penetration `<= 0.01 mm` |
| Pad orientation | physical pad normal 与 local inward surface normal 的夹角 runtime `<= 45 deg`；规划使用至少 `5 deg` margin，即 `<= 40 deg` |
| Joint safety | position/velocity/command limit violation `= 0`；planner maximum single joint step `<= 0.03 rad` |
| Numerical safety | NaN、Inf、solver exception、unstable integration、unbounded admittance state均 `= 0` |
| Plan audit | 每个 keyframe 与每个插值帧都通过 URDF/MJCF reachability、joint、pad-angle、FR3 clearance 和 self-collision 检查；不得跳帧抽查代替全量检查 |

普通 planned frame 要求四个 pad 位于 nominal contact band。允许的 recovery
必须同时满足：

- 单次 static bridge 覆盖不超过 `1.5 mm`，全路线累计不超过 `2% L`；
- static + genuinely moving recovery 的单次连续跨度不超过 `3.0 mm`，累计不
  超过 `3% L`；
- recovery 中 nominally supported fingertips 不少于 2，且运行时仍受上表
  `>=3` majority aggregate 约束；
- moving recovery 必须有非零 finger joint motion 和至少三指正向 surface
  progress，不能靠复制上一姿态通过；
- 路线最后 `20 mm` 禁止 recovery bridge，必须回到普通进度/接触门槛。

palm guide 的 tracking error 只报告，不因未落在严格位置球内而失败；只要
FR3/LEAP reachability、joint 和 collision safety 通过即可。palm contact 若
发生，归入 `Other LEAP contact`，不能为 fingertip contact 补分。

## 4. 必须报告但不单独决定成功的指标

以下指标必须逐 seed 保存，便于比较方法和发现退化。除非在运行前另行冻结为
硬门槛，否则不能在看到结果后临时挑阈值改变 PASS/FAIL。

- 四指 force target、mean/RMS/P95/max force、force-error MAE/RMSE/P95；
- raw force peak 及时间戳、filtered-force peak 及持续时间；
- per-finger contact-loss 次数、每次持续时间、最长持续时间、恢复时间；
- `C=0/1/2/3/4` 各自帧数和占比；
- planned/actual surface progress、geodesic/world/relative-to-palm travel、
  cross-track error、backtracking distance；
- fingertip tangential speed、normal speed、受控滑移距离；
- 每指 joint excursion、minimum joint margin、command rate；
- Finger MCC normal offset/velocity、各限幅 saturation ratio、anti-windup
  触发次数；
- wrist wrench、Cartesian reference offset、joint correction 和 saturation；
- planner objective、constraint residual、iterations、restarts、fallbacks；
- solve latency 的 mean/P50/P95/P99/max、deadline miss rate、achieved rate；
- minimum FR3 clearance、self-clearance、pad angle 和所有 incidental contact
  geometry/force/impulse；
- object curvature range、curvature ratio、scale、friction、pose、seed；
- wall-clock runtime、GPU/driver/MJLab version、代码 commit 和完整命令。

这些 report-only 指标是 Baseline 2A/2B 和未来 DP 主方法公平比较的基础。

## 5. 分级完成门槛

### Level 1：底层闭环资格

目的：只验证 corrected low-level controller 的传感器数据流、M-B-K 动态、
符号、带宽分离和短程安全，不证明长程规划能力。

完成条件：

1. 当前工作树的全部 unit/structure/contact-policy tests 退出码为 0；demo、
   grasp-search 和 grasp-optimization CLI import/`--help` 退出码为 0，活动
   demo help 中不存在旧 `--variant`；
2. 四指 primary feedback 可追溯到四路 direct fingertip force；motor-derived
   force 只出现在 diagnostic；
3. pure-controller test 中 MCC 对 planned tangential target 的改变量绝对值
   `<= 1e-9 m`，normal offset/speed/acceleration 均不超过配置上限；
4. 在 canonical `R=0.10 m, half-height=0.17 m` capsule 上完成至少 `5 mm`、
   至少 750 simulation steps 的 CUDA headless motion；
5. 四指各自 actual directed surface progress `>= 4.75 mm`，并通过第 3 节
   全部硬门槛；末段 `4/4` 连续至少 20 个 100 Hz evaluated frames；
6. 只保存 numerical summary、plan 和日志；Level 1 不生成交付视频。

当前状态：`PASS-NUMERICAL-L1`。2026-08-10 清理后的工作树已完成 17/17
unittest、三个 CLI 检查和 5 mm/750-step CUDA headless；详细数值见第 9 节。
本次没有生成视频，且结果不能升级为 Level 2–5。

### Level 2：单物体 0.48 m 完整长程

canonical object 固定为 capsule：`radius=0.10 m`、cylindrical
`half-height=0.17 m`，object axis 为 `+z`。请求路线 `L=0.48 m`，从下端
区域连续经过 cylinder 和 upper cap，触摸顶部区域。

一个 seed 的完成条件：

1. upper optimizer 返回完整计划，所有五个 planning points 均来自同一可达
   FR3 + LEAP state；palm-root 为 guide，四个 pads 严格跟踪各自 surface
   targets；
2. 四指各自 actual directed surface progress 均 `>= L - 4 mm = 0.476 m`，
   终点 progress error `<= 4 mm`；各自 accumulated surface travel
   `>= 0.95 L = 0.456 m`；
3. 每个指尖的 `tip-relative-to-palm travel >= 4 mm`，且每根手指至少一个
   joint 的 peak-to-peak excursion `>= 0.08 rad`，排除纯 FR3 刚性搬运；
4. 对 canonical capsule，四个 pad 都必须跨过 `z = +half-height` 的
   cylinder-to-upper-cap seam；终端四指都在 upper cap (`z >= +0.17 m`)，
   四指 contact centroid 达到 `z >= half-height + 0.50 radius = 0.22 m`，
   且至少一个 pad 达到 `z >= half-height + 0.75 radius = 0.245 m`；
5. planned keyframe backtracking 每步 `<= 0.2 mm`，累计 actual backward
   progress `<= 0.01 L = 4.8 mm`；
6. 第 3 节全部硬门槛通过，末段连续 `4/4 >= 0.50 s`，最后 `20 mm` 无
   recovery bridge；
7. 固定 controller gains 和 acceptance thresholds，对 seeds
   `42, 43, 44, 45, 46` 连续执行；五次必须全部通过，不能逐 seed 手调参数
   或替换初始姿态。允许同一个自动 optimizer 根据 seed 求解。

Level 2 先运行五次 headless。任一 seed 失败时不得生成 canonical delivery
video；失败必须记录在 PROCESS，且说明是 planning infeasibility、contact、
force、collision、progress、topology 还是 numerical failure。

### Level 3：Oracle 与 time-capped variants

Level 3 使用 Level-2 五个完全相同的初始状态、路线、controller gains 和 seeds，
只改变上层 optimizer 的求解预算。

#### Baseline 2A：Optimization Oracle

- 不设实时 deadline，但必须记录每次 solve wall time、iterations 和 restarts；
- planner 必须正常报告 convergence，不能把 exhausted/timeout/fallback 标为
  Oracle convergence；
- 所有 feasibility residual 必须在 Level-2 几何门槛内；
- 用两倍 `max_nfev` 或等价更大预算复核，normalized objective 改善不超过
  `1%`，四指终点 Cartesian 变化均不超过 `1.0 mm`；
- 五个 seeds 必须 `5/5` 通过 Level-2 动态硬门槛。

#### Baseline 2B：50/100/200 ms time-capped

每个 budget 都要独立标记 PASS 或 FAIL，不能把 200 ms 的结果写成 50 ms
结果。wall-clock latency 必须覆盖 CPU/GPU synchronization、solver 和 candidate
selection，不得只报告内部 kernel time。

一个 budget 的成功条件：

- 五个相同 seeds 中至少 `4/5` 完成 Level-2 路线与全部动态硬门槛；
- `P95 solve latency <= budget`，deadline miss rate `<= 5%`；
- `max latency <= budget + max(5 ms, 0.10 * budget)`；
- timeout 时只允许复用最后一个已全量验证的 feasible solution 或安全
  hold/retreat；unsafe partial solution 次数必须为 0；
- fallback frame ratio `<= 20%`；
- 相对同 seed Oracle，surface progress 不低于 `95%`，majority contact ratio
  降低不超过 `5 percentage points`，average contact count 降低不超过
  `0.25`；force/collision 门槛不因实时预算放宽。

Level 3 实验完成要求：Oracle 通过，50/100/200 ms 全部完成同条件测量，并且
至少 200 ms variant 达到其独立成功门槛。50 或 100 ms 可以作为有意义的失败
结果用于展示 compute/contact tradeoff，但只有达到上述门槛的 budget 才能称为
“time-capped Baseline 通过”。

### Level 4：泛化资格

Level 4 开始前冻结 Finger/Wrist MCC gains、force thresholds、contact windows、
recovery budgets 和 optimizer objective weights。允许上层读取不同 object
geometry 并自动生成 route/initial grasp；不允许针对单个测试物体手调 controller
或挑选 seed。若无法泛化，必须如实记录，作为 proposal 使用 inverse-trained DP
的动机，不能删掉失败物体后宣称泛化。

每个成功 run 必须通过第 3 节全部硬门槛、Level-2 的主动手指运动门槛、
`actual directed surface progress >= 0.95 L` 和末段 `4/4 >= 0.50 s`。

#### 物体课程

| Family | 必测几何与路线拓扑 | 最小路线要求 |
| --- | --- | --- |
| P：plane | 大平面或远离边缘的厚板中心区域，验证零曲率与法向稳定性 | 单向或弧形 surface route `L >= 0.20 m` |
| C：large-radius cylinder/capsule | `R=0.10/0.15/0.20 m` 的侧面段，验证粗物体与近恒定曲率 | 沿轴/斜向 route `L >= 0.20 m` |
| T：complete capsule top | 至少 canonical `R=0.10 m, h=0.17 m`，从下端跨侧面和上端帽 | `L=0.48 m`，满足 Level-2 seam/top 坐标门槛 |
| H：high-curvature sphere/ellipsoid | sphere 与三轴不等 ellipsoid 均要出现；ellipsoid route 的正曲率比 `>= 2` | pole-to-equator 或跨主曲率方向 route `L >= 0.15 m` |
| B：rounded box | 至少三种 fillet radius，路线必须跨 `face -> rounded edge -> adjacent face` | `L >= 0.15 m`，至少一次明确曲率突变/快速变化 |
| S：superquadric/irregular | 至少一个参数化 superquadric 和一个未用于调参的 irregular mesh | `L >= 0.15 m`，跨至少两个不同曲率区域 |

nominal geometry 在开始 Level 4 前冻结：

- P：有效 central patch 至少 `0.30 x 0.30 m`，pad 路线距任何外边缘
  `>= 50 mm`；
- C：圆柱/胶囊侧面半径使用 `0.10/0.15/0.20 m`，轴向有效长度至少
  `0.25 m`；
- T：使用 Level 2 的 `R=0.10 m, h=0.17 m` canonical capsule；
- H：sphere 使用 `R=0.06/0.08/0.10 m`；ellipsoid 至少使用半轴
  `(0.06,0.09,0.14) m` 和 `(0.08,0.12,0.18) m`；
- B：nominal half extents `(0.10,0.08,0.16) m`，fillet radius 使用
  `5/15/30 mm`；
- S：superquadric nominal axes `(0.10,0.08,0.14) m`，至少测试 exponent
  pairs `(e1,e2)=(0.5,1.5)` 与 `(1.5,0.5)`；irregular mesh 的 bounding-box
  最长边限定在 `0.12–0.30 m`，其文件 hash 在冻结清单中记录。

每个 family 的冻结验收矩阵包含 20 runs：以下四种配置各跑 seeds
`42–46`。

| 配置 | 尺寸 | friction coefficient | 固定物体姿态 |
| --- | --- | --- | --- |
| G0 nominal | `1.00 x` nominal | `0.60` | nominal pose |
| G1 small/low-friction | `0.80 x` | `0.30` | translation `(+10,-10,+10) mm`，rotation `(+15,-10,0) deg` |
| G2 large/high-friction | `1.20 x` | `1.00` | translation `(-10,+10,-10) mm`，rotation `(-15,+10,+15) deg` |
| G3 held-out | 从 `[0.90,1.10]` 用预先保存的 seed 采样 | 从 `[0.40,0.80]` 用同一 seed 采样 | translation 每轴 `[-20,+20] mm`、rotation 每轴 `[-20,+20] deg`；运行前冻结，irregular mesh/shape parameters 不参与调参 |

每个 G 配置内部的 seed-to-shape mapping 在运行前固定，确保“一个 family”不被
单一容易几何代替：

| Family | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
| --- | --- | --- | --- | --- | --- |
| P | straight route | diagonal route | curved route | reverse diagonal | held-out curved route |
| C | `R=0.10 m` | `R=0.15 m` | `R=0.20 m` | `R=0.10 m` | `R=0.20 m` |
| T | canonical route | start azimuth `+15 deg` | start azimuth `-15 deg` | route yaw `+10 deg` | held-out route yaw `-10 deg` |
| H | sphere `R=0.06 m` | sphere `R=0.10 m` | ellipsoid `(0.06,0.09,0.14) m` | ellipsoid `(0.08,0.12,0.18) m` | sphere `R=0.08 m` |
| B | fillet `5 mm` | fillet `15 mm` | fillet `30 mm` | reverse face-edge-face, `5 mm` | held-out edge, `30 mm` |
| S | superquadric `(0.5,1.5)` | superquadric `(1.5,0.5)` | irregular mesh | second route on irregular mesh | held-out superquadric parameters |

G1/G2 的 scale 乘在表中 base geometry 上；G3 使用冻结采样的 scale。完整
120-run manifest 必须在第一次 evaluation 前保存，之后不能因失败替换 route、
mesh、seed 或 shape parameters。

总计至少 `6 families * 20 = 120` 个 frozen evaluation runs。Level 4 完成
条件：

- overall success rate `>= 85%`（至少 102/120）；
- 每个 family success rate `>= 80%`（至少 16/20）；
- G0/G1/G2/G3 每个配置跨 family 的 success rate `>= 80%`；
- 所有 120 runs 的 FR3/object contact event、unsafe partial-plan execution、
  filtered force cutoff 和 raw emergency cutoff 事件均为 0；安全 abort 可防止
  后续伤害，但对应 run 仍计失败；
- 必须分别报告 size、friction、pose、curvature 对 contact、force、latency、
  recovery 和 failure mode 的影响，不只报告总体平均。

### Level 5：视频交付资格

Level 5 不能用视觉印象覆盖数值失败。完整项目视频交付要求 Level 1–4 已满足；
Level 2 通过后可以生成 canonical review candidate，但在 Level 4 通过前只能
标为“single-object candidate”，不能称为最终泛化交付。

正式交付至少包含：

1. canonical 0.48 m bottom-to-top 的无剪切完整 overview；
2. 同一 canonical run 的同步 pad-side/close-up view，能看清 thumb 和四个
   physical pads；
3. 一个 generalization review montage，P/C/H/B/S 每个非 canonical family
   至少放入一个 held-out PASS run；montage 可以在 runs 之间剪切，但每个 run
   的接触运动段必须连续且明确标注 object、seed、scale、friction 和 playback
   speed；
4. 每个片段对应的原始完整 MP4、plan、numerical summary 和 review record。

视频技术门槛：

- 最低 `1280x720`、`30 fps`；物理时间与视频时间比在 `0.75x–1.25x`，不得
  通过高速播放掩盖失联或穿模；slow motion/cut 必须在画面中标注；
- 从接触稳定前至少 1 s 开始，包含完整 motion、顶部/终点和末段至少 1 s
  `4/4` 保持；不能在物体刚碰到手、路线未开始或未到顶部时结束；
- overview 中 FR3、LEAP 和完整物体边界可见；close-up 中 thumb pad 在至少
  80% 的固定抽查帧中可见，任何遮挡区间必须由第二视角覆盖；
- 必须能辨认是指腹内侧沿表面滑动，而不是指甲、指背、掌背或机械臂顶住物体；
- 视觉上不得出现 FR3/物体相交、明显 mesh penetration、teleport、自然闭合
  后离面、长时间原地不动或错误的超快播放。

## 6. Headless-first 执行顺序

每个候选严格按以下顺序处理：

1. 冻结 commit、环境、完整 CLI、seed、object parameters、controller gains、
   acceptance thresholds 和输出目录；
2. 运行 tests/CLI check；
3. `--viewer headless` 完成 planner + 全程 dynamics + final audit；
4. 保存 plan、逐帧 metrics 和 machine-readable summary；任何硬门槛失败即停止，
   不编码视频；
5. 对 PASS plan 以相同 commit/config/seed 运行 `--viewer video`，不得重新搜索
   一个更好但未 headless 审核的 plan；
6. video run 自身再次执行完整 numerical audit；render 模式中的失败仍使视频
   作废；
7. 完成第 7 节视觉审核后才能复制到 `outputs/deliverables/fr3/`。

文件名中的 `final`、`accepted` 或版本号没有验收效力；只有 summary 和 review
record 能改变状态。

## 7. 视频逐帧/抽帧审核清单

### 7.1 必查帧

- motion 前 1 s、开始运动、25%/50%/75% route、cap/edge/curvature transition、
  top/终点和最后 1 s；
- 全程每 `0.5 s` 固定抽一帧；
- numerical log 中每次 `C(t)` 变化、任一指失联/恢复、force P95/max、minimum
  FR3 clearance、maximum penetration、maximum pad angle 的对应帧；
- 每个异常事件前后各 10 帧逐帧查看；若固定抽帧发现不合理，则该区间全部
  逐帧查看，不能换相机角度回避。

### 7.2 每个检查点回答

- 四个接触是否都发生在指定 physical pad，而不是指甲/外侧/指背？
- pad inward direction 是否与 local surface normal 大体一致，而不是机械地
  指向一个全局中心？
- thumb 是否可见；若被遮挡，第二视角是否在同一时刻清楚显示？
- fingertip 是否沿 planned surface direction 连续移动并在短暂失联后恢复？
- palm/其他 LEAP contact 是否只是受控附带接触，没有替代或推开 fingertips？
- FR3 是否与物体保持可见间隙；是否存在穿模、瞬移、抖动或异常弹飞？
- canonical run 是否真正跨过 upper-cap seam 并达到顶部区域？
- 播放速度是否足以观察接触，没有用加速隐藏问题？

任一答案为“否”时视频视觉 FAIL，即使数值 summary PASS，也不能交付。

### 7.3 审核产物

每个正式视频旁必须保存：

- `acceptance.json`：commit、CLI、seed、object、全部硬门槛与 report metrics；
- `review.md`：审核日期、视频时长、物理时长、相机、固定抽帧时间点、事件帧、
  每项视觉结论和最终 PASS/FAIL；
- contact sheet：至少包含开始、四个路线分位、曲率/顶部关键点、结束和所有异常
  事件；
- plan/trajectory 文件的路径与 hash。

## 8. 状态词与失败记录

只使用以下状态，避免把“运行结束”写成“成功”：

- `NOT RUN`：尚未执行；
- `RUNNING`：执行中，不能提前判断；
- `FAIL-SAFE`：未完成任务但安全停止；
- `FAIL-UNSAFE`：出现 collision/force/penetration/numerical hard failure；
- `PASS-NUMERICAL-L<n>`：对应等级 headless 数值通过；
- `PASS-VISUAL`：同一数值通过 run 的视频审核通过；
- `DELIVERABLE`：Level 5 所需文件齐全并位于交付目录。

失败记录至少包含 first failing frame/keyframe、失败门槛、实际值、object/seed、
最后 feasible state、是否触发安全停止和下一项假设。失败不得删除；泛化失败要
进入项目结论，并用于解释为什么 proposal 主方法需要 inverse demonstrations
与 learned finger policy。

## 9. 当前基线

2026-08-10 清理和 direct-force module 解耦后的验收状态为：

| Level | 状态 | 当前证据 |
| --- | --- | --- |
| Level 1 | `PASS-NUMERICAL-L1` | 17/17 unittest；demo/search/optimize CLI exit 0；5 mm/750-step CUDA headless exit 0 |
| Level 2 | `NOT RUN` | 0.48 m、upper-cap seam、top coordinates 尚未完成 |
| Level 3 | `NOT RUN` | Oracle 与 50/100/200 ms timing 尚未运行 |
| Level 4 | `NOT RUN` | 多物体/尺度/摩擦/姿态矩阵尚未运行 |
| Level 5 | `NOT RUN` | 没有清理后交付视频或视觉审核 |

Level-1 GPU summary：

- contact ratio `[0.9975,1.0,1.0,0.99]`；majority ratio `1.0`；average
  contact `3.9875/4`；minimum simultaneous contact `3/4`；
- longest per-finger loss `[1,0,0,1]` frames；terminal `4/4=65` frames；
- raw force peaks `[14.936,10.963,26.132,9.721] N`；filtered normal peaks
  `[13.158,8.862,20.480,8.029] N`；
- FR3/object contact、self penetration、tip penetration、incidental hand
  contact 全为 0；maximum pad angle `41.44 deg`；travel `5 mm`；
- headless exit 0，没有编码视频。

该结果只证明 Level 1。它不证明：

- Level 2 的 0.48 m bottom-to-top、upper-cap seam 和 top coordinates；
- Level 3 的 Oracle convergence 或 50/100/200 ms timing；
- Level 4 的任何多物体/多尺度/摩擦/姿态泛化成功率；
- Level 5 的视觉审核或正式交付。

Level 2–5 仍为 `NOT RUN`，不得根据旧视频、旧文件名或旧结果补记为 PASS。
