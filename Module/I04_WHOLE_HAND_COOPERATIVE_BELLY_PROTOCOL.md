# Mustard 整手协同指腹接触与自然屈指

日期：2026-10-18。开发协议：`i04-whole-hand-belly.development.v1.6`。
本轮响应“只有一个手指接触、其余手指被动伸直”和“第二个手指姿态僵直”；第二个手指按中指处理。
验证对象为原 Mustard、固定物体/零重力仿真、MCC。DP按用户要求暂缓。

## 范围与状态

当前完成一段局部四指共同接近和60mm参考滑动，未接入整瓶换面调度。
旧 `mustard_natural_fresh02_20261018` 的单领触整瓶原生证据保留，不能作为本控制器的整瓶成功。
新主案 `generated/whole_hand_probe08_20261018` 从唯一声明6mm悬停初态、零覆盖开始；无续接或分块重置。
局部结果不外推一般多物体、硬件、GPIS或DPRef性能，也不是旧冻结A/B协议的公平对比。

## 实现

- `i04_geometric_traversal/whole_hand_belly.py`：四指各自表面目标、共同23DoF IK、自然屈指软先验及整手名义路线。
- `i04_geometric_traversal/whole_hand_belly_probe.py`：原生2ms闭环；四指独立法向MCC，腕与四指共同求解，没有控制意义上的单领触指。
- `i04_geometric_traversal/whole_hand_belly_review.py`：验证原始数据哈希后计算指标，渲染实际保存姿态；不生成新动力学或规划覆盖。

四指均在每一步求解自己的掌侧指腹位置和法向。各指分别接近、调压和调整附近落点；先接触指的剩余悬停间隙连续转入MCC记忆，避免突然向内跳变。
名义腕路线沿四指共同表面目标连续规划，避免腕部来回迁就某一指而掀起其他指。
长指软先验为 `[0,.22,.35,.22]` rad，并加入 `DIP≈0.65×PIP+0.015` 的软耦合及弱跨指协调。
长指任务范围为侧摆 `[-.18,.18]`、MCP `[-.12,.85]`、PIP `[.10,.95]`、DIP `[.07,.75]` rad，再与原物理范围相交。
这些是新开发姿态设计，物理关节范围没有修改。中指不再跟随旧版静止伸直等待参考。

每指目标力1N，MCC虚拟质量0.08、阻尼70、刚度20，法向偏移上限4mm，速度上限0.25mm/s。
混合指腹/边缘接触优先使用合格指腹法向，并根据被拒绝接触法向的切向分量以1mm/s微调落点，累计上限6mm。
混合接触若仍有合格指腹，调压目标0.75N，否则0N；拒绝接触≥0.2N仍不能计入安全指腹或覆盖。
主动卸力在原始力>2.5N启动、<1.2N恢复，包含40ms实际法向速度前瞻；不改变原8N覆盖排除或12N持续20ms硬停。

共同参考以4mm/s推进，需至少三指安全接触、无主动卸力且各指拒绝力<0.2N。
开始/结束交接采用300ms窗：≥三指接触时间占90%以上，每指参与60%以上，当前四指同时贴合。
短暂个别失联不会让全手重新起步，但永久被动第四指无法通过；该窗口只影响运动交接，不放松物理接触分类。

## 物理与证据约束

原物体网格、256件碰撞体、物理尺寸、执行器/摩擦和原始接触测量保持。
掌侧指腹逐contact判据保持：对齐角≤30°、中央椭圆归一化半径≤0.8、表面误差≤0.1mm。
非指腹3mm和臂部1mm原硬几何阈值保持；采用6mm软目标提前避让，实际命令每2ms同时相对实测状态和上一参考做原严格检查。
原生状态记录q、qvel、ctrl、warmstart、完整接触力/点/法向、腹面分类、MCC、各指目标及覆盖。
独立审计从唯一初态逐步 `mj_step`，重新测力和分类，原轨迹分块不重置物理。
覆盖仅由合格实测指腹接触增加，仍为25mm网格边测地邻域；不是皮肤压力足迹面积。

## 最终局部主案

| 检查 | 实测结果 |
| --- | --- |
| 完整仿真时长 | 25.988s |
| 开始共同滑动 | 5.848s |
| 滑动阶段时长（包括卸力/调整） | 20.140s |
| 终止原因 | `COOPERATIVE_LOCAL_STROKE_COMPLETED` |
| 扫描参考进度 | 60mm |
| 至少三指同时安全指腹接触 | 92.6415% |
| 四指同时安全指腹接触 | 59.8411% |
| 原始力峰值 | 3.302455N |
| 臂自碰力 / unsafe状态 | 0 / 0 |
| 规划抬手 / 实际命令几何回退 | 0 / 0 |
| 原生连续独立核验 | 12994状态，通过；一次初始化 |
| 状态、力、覆盖最大误差 | 0；法向浮差≤3.34e−16 |
| 末300ms独立复核 | ≥三指100%；各指100/100/100/60%；最后观测四指接触 |
| 局部覆盖 | 4.327236%；396/8192顶点、17/388目标 |

| 手指 | 滑动阶段接触率 | 首末合格接触点沿世界Y净移 | 全程合格接触期间切向材料累计运动 |
| --- | --- | --- | --- |
| 食指 | 99.0765% | 60.017mm | 82.762mm |
| 中指 | 90.8044% | 60.566mm | 94.024mm |
| 无名指 | 93.8530% | 59.984mm | 88.280mm |
| 拇指 | 67.5670% | 58.096mm | 57.273mm |

净位移和累计材料运动的定义及统计区间不同，不互相冒充；参考60mm也不替代实际运动证据。
中指滑动时侧摆10.199–10.359°、MCP18.691–28.663°、PIP26.213–30.064°、DIP10.247–15.875°。
PIP和DIP保持弯曲且实际变化，MCP变化约10°。拇指仍间歇失联，未达到四指始终贴合。

## 开发反例与验证

probe01–06均保留：包含首触目标跳变、几何悬停可行但接近走廊受阻、共同腕姿摆动、混合侧缘法向和过严连续四指门槛等问题。
其独立原生审计通过仅说明记录一致，超时/开发停止不称成功，不拼接为最终结果。
probe07（v1.5）也完成局部滑动，28.404s、≥三指93.5285%、四指56.3601%、峰2.822431N。
probe08提高四指同时率与拇指参与率，但≥三指率略降、力峰略增，不称所有指标全面优化。
几何拟合本身不计覆盖。最终60mm名义路线的61个姿态均通过原严格几何，实际12994条命令另行逐步独立核验。
38项相关回归已通过（原自然手形/臂缓冲反例/指腹/有界步进32项，新增6项交接连续性及整手参与反例）。

证据入口为主案 `review.html`。`cooperative_full_1x.mp4` 为完整局部时间轴20fps正常速度；三视角分别看整手原物体、长指弯曲、拇指对掌。
中、右视角仅将物体透明并调整渲染光照，手部和运动取核验实际状态；粉点均为合格原生接触。
`cooperative_metrics.json/png`给出接触人数、各指参与、原始力和实际中指运动；`delivery_checks.json`验证引用、编码和运行源一致性。
`independent_native_audit/audit.json`、`manifest.json`、`summary.json`、`trace_*.npz`、初末完整状态及36个源快照可追溯。

复现时使用新输出目录，保留原案：

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) '.runtime')+';'+(Get-Location).Path
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'
& 'D:\Environment\Anaconda\envs\handcomp\python.exe' -m Module.i04_geometric_traversal.whole_hand_belly_probe --output Module/generated/whole_hand_reproduction --geometry Module/generated/whole_hand_geometry_touch02_20261018/geometry.npz --index 8 --seconds 60 --stroke .06
& 'D:\Environment\Anaconda\envs\handcomp\python.exe' -m Module.i04_geometric_traversal.natural_belly_audit Module/generated/whole_hand_reproduction
& 'D:\Environment\Anaconda\envs\handcomp\python.exe' -m Module.i04_geometric_traversal.whole_hand_belly_review Module/generated/whole_hand_reproduction
```

运行源SHA256：`7ce2d9042bba039c4b0b6f9cb327c07931cb14c4a9ab17e4a89e45d561fa0f37`。
整手规划器SHA256：`9e8a1eb58656b22664178273e783dcc630741e5b942cc6f9856059c58c60a8f2`。
冻结模型SHA256：`6aaa2bb3bb302c37944a22913cf944a998cb762273c423de7a0d032943067821`。

后续工作是把四指共同目标和姿态约束接入连续换面/整瓶覆盖调度，并从零执行完整回归；当前未完成此项。
