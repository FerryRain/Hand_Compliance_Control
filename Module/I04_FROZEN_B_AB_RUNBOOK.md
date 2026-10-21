# I04：冻结成功 B 调度的 A-MCC 对照

2026-10-21。B 基准为 `generated/i04b_full_force_grace01_20261020`，代码、原生轨迹、报告和视频保持原字节。
本次 A 使用 `a_frozen_b_mcc`，不是存在行为回归的旧 `matched_mcc` 执行入口。

## 实验目的与用户确认的边界

检验加入逐指表面目标、指腹朝向和联合手指规划的作用。用户明确选择：A 沿用成功 B 的外层调度，
包括规划预计接触人数、参考手指以及对应推进门槛；只隔离手指目标和名义手指规划的差异。

因此本轮协议为 `i04.ab.frozen-b-supervisor.comparison.v1`，不是旧协议中连预测调度信息也禁止的严格 A。
私有外层几何服务在两侧一致；它可用 B 的几何姿态库产生腕目标，但只向 A 的名义参考构造器传递腕位姿。
A 固定公共初态的名义手形；实际四指仍由同一 MCC 和有界全手控制求解器主动运动，并非锁死关节。
两组闭环状态不同后，实际目标序列和腕轨迹可以不同。
外层预计可接触的手指数来自联合手形候选，A不执行该手形，预测可行不保证A实际可行。
因此结果属于冻结外层流程的消融，不代表为A重新设计过的最优腕规划器性能。

## 保持一致的条件

- 同一 FR3+LEAP、固定 Mustard 2.8 倍缩放、256 件碰撞体、零重力、MuJoCo 原生步长 2ms。
- 同一完整初始 qpos、qvel、ctrl、warmstart 和时间；零继承覆盖。
- 原 50 份 B 运行源码保持哈希；复用其控制核、测量、覆盖、外层扫描及补扫算法。
- 先表面扫描，达到97%或搜索耗尽后继续补扫；总仿真预算9000秒，原搜索及恢复预算保持。
- 每指 MCC 目标1N，质量0.08、阻尼70、刚度20，法向偏移上限4mm。
- 原始力>2.5N主动卸力，<1.2N恢复；>12N仅记录，>40N连续200ms停止；≥8N状态不增加覆盖。
- 原四指起步窗口、后续2/3指预计人数及参考手指推进规则保持；逐指参考法向转角使用同一限速公式。
- 原关节范围、步长限制、实态及上一参考双几何检查、非指腹/臂间隙和独立核验。

“同一公式”不表示两组目标、预计接触、参考法向或实际动作时长相同；这些值随各自闭环状态及方法参考变化。
覆盖沿用25mm网格边测地邻域，分母8192顶点、388目标，不等同于物理皮肤压力足迹覆盖率。

## 安装与运行

从仓库根目录执行；依赖见 `requirements-i04-whole-hand.txt`。Windows验证环境为Python3.10、MuJoCo3.6.0。
发布数据使用Git LFS，先执行 `git lfs pull`。本地 `.runtime` 是已有依赖目录，不是控制逻辑补丁。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'
python -m Module.i04_geometric_traversal.b_full_portable verify --case Module/generated/i04b_full_force_grace01_20261020
python -m Module.i04_geometric_traversal.a_frozen_b_mcc --output Module/generated/my_a_frozen_b --seconds 9000
python -m Module.i04_geometric_traversal.ab_frozen_b_delivery audit --case Module/generated/my_a_frozen_b --workers 4
python -m Module.i04_geometric_traversal.ab_frozen_b_delivery video --case Module/generated/my_a_frozen_b
```

输出目录必须不存在。`--seconds 35` 可做短段验证，但短段不是正式整瓶结果。
执行期间可在另一个终端给审计和视频命令加 `--follow`，在最终 `run_complete.json` 出现后结束。
审计从t=0连续重放，分块间不重置物理状态；所有已执行命令同时验证实态和上一参考的几何守护。
视频只读取记录的实际q，不推进物理，包含首尾状态、全部离面、卸力和失败尝试。
正常速度与12倍速采用B相同的20fps、2400×700三视角；B视频直接引用原文件。

## 每次规划的 cost 与时间

A 的 `planning_calls.jsonl` 保存每次观测调用的ID、父ID、仿真请求时间、wall/进程CPU耗时、成功/失败，
并保存实际 `least_squares` 返回的cost、迭代次数与终止状态。父调用时间包含子调用，不得重复求和。
初始化和2ms底层控制循环不属于该规划时间合计。计时观察器不修改目标函数，但记录开销和系统负载可能影响墙钟搜索预算。
外层候选排序、调度代码和顶层日志落盘同样未包围在调用计时中；合计是“已观测调用耗时”，不是完整规划周期的总耗时。
规划期间仿真时间不推进。`wall_elapsed_s`另记包含初始化、规划、原生执行与写盘的实际运行时间；
两次运行均有并行审计与渲染，整体墙钟不能直接当作纯规划计算时间。
其中 `private_supervisor.*` 的23关节求解属于共同外层几何服务；A方法自身的名义参考求解是7关节腕IK。
外层服务也会使用7变量腕种子求解，WRIST标签只标识目标函数；应按父调用ID判断归属，不能只凭变量数分摊A/B开销。
不能把外层服务的联合几何计算误称为A得到了逐指参考，也不能把它从实际计算总开销里隐去。
冻结观察器的`find_patch`原始success检查返回元组，`(None, attempts)`也会被写为true。
报告根据同一条记录中的解元数据是否存在恢复实际接受/拒绝；CSV保留`recorded_success`原值，
`success`用于修正统计。原始JSONL、控制执行和计时不受此报告修正影响。

B原实验没有保存逐调用计算时间、优化器cost或全部被拒绝候选。事件中的 `duration_s` 是计划动作时间，
不能写成计算时间。换面选择分数可从记录重建：扫描阶段为撤离+调姿秒数；补扫阶段再加 `3*(4-预计接触数)`。
贴面路线采用第一个可行解，没有统一的标量路线cost；报告按实际算法注明，不伪造缺失值。

为补充计算分析，可对B每个已记录规划动作的保存状态运行配对静态IK微基准：

```powershell
python -m Module.i04_geometric_traversal.ab_frozen_b_cost_review --case Module/generated/i04b_full_force_grace01_20261020 --output Module/generated/my_ab_review/b_offline_ik
python -m Module.i04_geometric_traversal.ab_frozen_b_report --a Module/generated/my_a_frozen_b --output Module/generated/my_ab_review
```

微基准不执行物理、不更改B；使用保存的命令状态、目标点与滤波法向构造同一静态问题，A只接收终点腕位姿。
输入分布来自B记录，两个方法共用保存的手臂起点；A名义手形取公共初态，B手形种子取该保存状态。
它不是B历史完整搜索的精确复现。建议原生实验、渲染和审计结束后单独运行，以减小负载对计时的影响。
两个方法优化目标不同，原始cost不可直接作高低优劣判断；另报告共同四指位置RMSE等后验指标。
报告同时提供原始事件CSV、A全调用CSV、逐事件配对静态IK CSV、图和JSON。

下载到其他路径后，可使用资产重定位入口（只改变同哈希模型文件的查找位置）：

```powershell
python -m Module.i04_geometric_traversal.ab_frozen_b_portable verify
python -m Module.i04_geometric_traversal.ab_frozen_b_portable smoke --case Module/generated/i04a_frozen_b01_20261021
python -m Module.i04_geometric_traversal.ab_frozen_b_portable audit --case Module/generated/i04a_frozen_b01_20261021 --workers 4
python -m Module.i04_geometric_traversal.ab_frozen_b_portable video --case Module/generated/i04a_frozen_b01_20261021
```

重审计和重渲染使用新的 `downloaded_*` 子目录，保留已发布证据。

## 本次正式文件

- A：`generated/i04a_frozen_b01_20261021`。
- B：`generated/i04b_full_force_grace01_20261020`，冻结不改。
- 对照：`generated/i04_ab_frozen_b_20261021/review.html`；最终指标以完成并核验后的 `comparison.json` 为准。

单次固定物体、seed7结果不代替多种子评价；DP/DPRef继续暂缓。A未完成时仍交付从0到实际停止的完整视频，
不会删去失败段或把提前停止延长成“与B同长”。
