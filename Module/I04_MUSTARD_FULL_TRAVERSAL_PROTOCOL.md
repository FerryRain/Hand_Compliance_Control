# 新 I04 mustard 全表面触摸遍历

日期：2026-10-13。用户要求实际遍历摸一遍，接触可短暂中断，MCC-only、DP 延后。
冻结协议：`i04-mustard-full-surface.v1`。数值配置和 runtime hash 在正式配对评测前冻结。
旧 Bunny v1、mustard recovery v2 原始记录保留，不改写其冻结协议。

## 完成目标

完成条件是整个闭合 mustard 外表面的实测触摸覆盖，包括瓶身各面、侧缘、底面、肩部、瓶颈和盖。
coverage 使用真实 MuJoCo pad contacts 的 25 mm mesh-geodesic union，normal cosine ≥0.5，
contact threshold=0.2 N、mesh residual ≤5 mm。执行轨迹、达到预算、IK witness、预测点不增加覆盖。
保留全部有体积的物体表面；必访点和 mesh vertices 的分母冻结。未访问区域不能以“不可达”删除。
整表面完成必须满足全部 8192 task vertices 与 388 required points 已真实覆盖，
并在末段确认至少一指安全接触连续 150 ms、所有 pad force ≤8 N；该确认期间仍保留所有保护。

source mesh quality 在轨迹规划前独立审计：闭合主体 8192 vertices / 16380 faces，正体积。
两个附加 components 共 4 faces，volume=0，其中一项 area=0，另一项为重合正反三角形、
area=2.2545e-8 m²（raw area 的约 6.32e-8）。它们不构成物体外表面，在 task surface 中清理；
raw source、物理 256-part V-HACD 完整保留，并同时报告 raw/clean area 与修复记录。
该修复依据零体积重合/退化三角形，不依据控制策略、覆盖结果或可达性。

## 几何与执行

物体保持固定，scale 2.8 已烘焙、runtime=1、gravity-off，23 DoF、dt=2 ms。
放置位置 `[0.30, 0, 0.40] m` 在正式评测前依据整个表面的机器人几何可达性确定，
A/B 使用完全相同的 fixture。使用 pad 实际 ellipsoid 的 support vector（包含切向偏移），
不能使用 box 的 L1 support radius。几何 cache 来源和版本、文件 hash 必须核对。
Oracle 构造外侧 wrist SE(3) 和四个 Cartesian tip positions；不提供 contact labels、role intention 或 joint witness。
A adapter 仅暴露 SE(3)，B 加四个 tip positions。同一 bounded IK、几何碰撞检查与路径时间参数化。
路径包含必要的外侧几何过渡；允许计划过程中真实失联，到目标附近再局部探触，不能一失联就回到旧区域。
MCC 仍按真实力/法向控制。局部恢复使用因果传感历史与已暴露的 wrist frame，不读取未来 tip/GT contact answers。
稳定接触至少一指 150 ms；未恢复或无法完成当前局部目标记录后继续其他未覆盖区域，并保留剩余区域。

持续 >12 N/20 ms、实际非指尖碰撞、自碰撞、关节越界及非有限状态仍独立保护；≥8 N 先卸力。
不为获得覆盖放宽保护，单次失联不判整段失败。报告必要的失联、局部重获、失败目标和累计推进。
视频与全瓶正反/底面覆盖图读取同一次保存状态与接触账本。

## 冻结配置与复现

Formal 使用 seeds 7/11/19，A/B 配对 initial q/qvel/ctrl snapshots，均从同一 5 s acquisition 开始。
maximum treatment budget=9000 s；完成后提前停止。budget 依据 development 中整瓶路径与末段
补扫耗时冻结（6000 s pilot 已触摸全部 required points，但 mesh 小孔仍未全覆盖），
不按某个正式结果修改。预算耗尽不能标作遍历完成，development 续跑不得冒充连续正式 episode。

共同速度上限：wrist 25 mm/s、rotation 0.35 rad/s、quintic joint path 0.8 rad/s、
control command 1.2 rad/s。Cartesian IK waypoints ≤15 mm /0.1 rad，joint edge check ≤0.025 rad。
bounded arm/finger IK，外侧几何转移使用 7-DoF RRT-Connect（最多 1500 iterations /2 s wall time）。
非 pad 几何规划间隙 3 mm；margin 扩大只用于 scratch 查询，mj_step 的原 contact 参数不变。
法向退出受阻时先查询 ±X/±Y/±Z 的 10/20/35 mm 小步退出，必要时直接规划 joint-space 外侧转移。
缓存可行几何 pose 不增加 coverage；相同世界目标仍在当前实测 root 上重新检查路径。

到达后允许 150 ms 失联/未覆盖缓冲，沿已暴露 wrist frame 与初始实测 normal history
生成有界几何接近：1.5 mm/s、最多 8 mm、local timeout=7 s。真实触点存在但目标区域尚未覆盖时，
记录为 Oracle geometric refinement；无触点时记录为 causal endpoint local recovery。
超时继续其他区域。失败规划至少等待 0.5 s physical time 再试；重复目标依次尝试其他 roll/finger
几何 pose；fill 用 20 mm geodesic 内的可行候选，但完成判定始终针对原始未覆盖 vertex。

soft relief：≥8 N 暂停路径并卸力，低于 4 N 后继续；hard stop：>12 N 连续 20 ms、
实际非 pad 碰撞/自碰撞、joint limit 或非有限状态。不可降低这些保护阈值来获得覆盖。
保存每 10 ms 原生状态、实际触点/法向/力、目标/phase 和 command，120 s 保存分段日志与物理检查点。
独立审计重建接触覆盖并集、完整分母、FK、command rate、信息边界、source hash 和配对初态。

Windows 固定 `handcomp`：

```powershell
./scripts/run_i04_full_surface_mcc.ps1 --geometry Module/generated/mustard_true_ellipsoid_geometry07 --output Module/generated/new_i04_mustard_full_surface_formal_20261013 --jobs 3
```

单个记录用 `python -m Module.i04_geometric_traversal.full_review <run-directory> --video` 生成双视角回放与六面覆盖图。
物理为原生 MuJoCo CPU，NVIDIA OpenGL 用于录像；不是实机实验。当前协议只报告固定物体扫描，不推断真实抓持稳定性。
