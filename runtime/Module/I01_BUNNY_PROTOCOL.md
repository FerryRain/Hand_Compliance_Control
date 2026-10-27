# I01：Bunny 上的 Oracle 连续接触物理验证协议

冻结日期：2026-08-24。授权来自用户本轮明确指令：“先跑 I01，在物理引擎中测试手到底
能不能在 Bunny 物体上移动并保持连续接触”。本协议只使用 Geometry-Oracle + Explicit
MCC baseline；Finger DP、预测 contact 和 prediction suffix 均无执行权限。

## Module ID 与边界

- Module ID：`I01-PHY-BUNNY-v1`
- 状态：`IN_PROGRESS`
- Owner：Codex / local worktree
- Exact scope：固定 Bunny、FR3+LEAP、MuJoCo gravity-off control-isolation；比较
  I01-A fixed four-contact 与 I01-B variable nonempty-contact traversal。
- 不在范围：硬件、sim-to-real、DP/Main planner、GPIS、active exploration、G3。
- G1 仍保持历史 `NO_GO`。本轮用户授权允许先运行 MCC-only I01，但不回写 G1 为通过。

## 几何与初态

- 输入资产：`Module/assets/stanford_bunny.ply`，运行时记录 SHA-256。
- 统一缩放：最长边 `0.300 m`、第二长边至少 `0.180 m`，与 M01 Bunny 展示一致。
- 固定姿态：先按源 `Y-up -> Z-up`，再绕本地 X 轴 `+90 deg` 侧放；完整三角网格只作
  可视几何。
- 物理碰撞：从同一变换后三角网格沿 `-Z` 射线取 upper envelope，生成
  `181 x 181` MuJoCo hfield。hfield 只作为可复现的 Bunny 上表面碰撞近似，不冒充完整
  非凸三角网格碰撞。
- 每个计分接触点必须同时满足：MuJoCo fingertip--Bunny 碰撞、法向力
  `>= 0.20 N`、且到完整 Bunny 三角网格的最近距离 `<= 2.5 mm`。落在 hfield 外围底板
  而不贴近 Bunny mesh 的接触不计入 `A_actual`。
- Bunny 固定在世界坐标；手移动，物体不移动。初态来自相同 FR3 home 与 LEAP natural
  posture，A/B 使用相同 seed 扰动。

## 冻结运行条件

- 环境：`handcomp`；MuJoCo timestep `0.002 s`；gravity `0`；固定物体。
- seeds：`7, 11, 19`，每个 cell 3 个 episode。
- 每条 episode：`3.0 s` contact acquisition，随后最多 `9.0 s` 计分 traversal。
- 目标路径：沿 Bunny upper envelope 的世界 `-X` 方向最多 `60 mm`，相同平滑时间律；
  wrist Z 只由 Oracle upper-envelope 高度变化补偿。
- 目标法向力 `2.0 N/finger`；contact threshold `0.20 N`；hard force limit `8.0 N`。
- 只有实际测量的非空 `A_actual` 可继续执行；过力、非 fingertip 碰撞、NaN、关节越界或
  最后接触丢失触发 `SAFE_HOLD`。

## Cells 与公平条件

### I01-A：Fixed Contact

- 计分开始时必须实测 `A_actual={1,2,3,4}`；整个 prefix 强制 `|A|=4`。
- `L_fixed` 为第一次持续 `>40 ms` 的四指 mode 破坏、guard/over-force stop 或路径终点
  之前的实际沿面进度。

### I01-B：Variable Contact Mode

- 允许任意非空 `A_actual`；其余 plant、初态、Oracle、路径、MCC、guards、时间和阈值与
  I01-A 完全相同。
- topology change 必须由 M10 certificate 绑定的 committed prefix 触发；每个 prefix
  最多一次变化，执行后在 micro barrier 用真实 contact 重测。
- 至少一次 handover 必须由测量证实为 `4 -> 3 -> 4`；BREAK 阶段被释放指法向力应连续
  `<0.20 N` 至少 `40 ms`，同时另外三指保持非空接触；MAKE 完成只按重新测得的力确认。
- `L_variable` 使用与 `L_fixed` 完全相同的停止规则。

## 数值指标与判定

“手能在 Bunny 上移动并保持连续接触”的 primary pass 必须在同一 cell 至少 2/3 episodes
同时满足：

1. 实际 palm 沿路径进度 `>=50 mm`；
2. 计分段有效 Bunny `|A_actual|>=1` 的时间占比 `>=99.0%`；
3. 单次全接触丢失 gap `<=10 ms`；
4. peak valid fingertip force `<=8.0 N`，持续过力 tick 为 0；
5. Bunny 以外 non-tip collision tick 为 0，guard emergency/NaN 为 0。

I01-A 的额外 fixed-contact 指标是四指 mode 占比和 `L_fixed`，不要求它必须完成 60 mm。
I01-B 只有在至少 2/3 episodes 完成测量 `4 -> 3 -> 4` handover 后才能称 variable-mode
机制有效。

Gate G2 只有在下列条件全部满足时才可标记 `GO`：

- `median(L_variable) >= median(L_fixed) + 10 mm`；
- 至少 2/3 I01-B episodes 达到 primary pass 且完成真实 handover；
- I01-A 的停止由相同路径上的 fixed-contact 局部不可行/接触破坏引起，而不是人为缩短轨迹；
- suffix、shadow state 或预测 contact 从未进入 command path。

未达到 G2 不影响本次实验作为有效 `EVALUATED / NOT_MET` 证据保存。

## Schema、evaluator 与生成物

- trace schema：`i01-bunny-trace.v1`，保存每个 physics tick 的 q/dq/command、palm/tip pose、
  raw contact force、mesh-valid contact mask、mode、contact position、mesh residual、guard、
  planner/audit/executor event 与 latency。
- evaluator：`i01-bunny-evaluator.v1`；可视化只读取冻结 trace/summary，不重新定义指标。
- 输出目录：`Module/generated/i01_bunny_physics/`，包含 `summary.json`、`episodes.csv`、
  `traces.npz`、生成 MJCF、canonical Bunny mesh、dashboard、视频/关键帧、`index.html`。
- 资源预算：CPU-only MuJoCo，正式运行不超过 30 分钟；单 episode 超过 5 分钟 wall time
  记为 timeout/`SAFE_HOLD`。
