# I04 mustard 连续贴面滑动

日期：2026-10-16。用户将当前阶段优先级调整为：尽量沿物体连续滑动，规划时考虑接触连续性，必要时才抬起调整方位。仅运行 MCC，DP 延后。

这是新增的 development 协议 `i04-mustard-continuous-sliding.development.v1`。
历史 full-surface v1.2 的结果和控制器保持原始记录；其全覆盖结果不代表本版本已完成整瓶连续滑动。

## 规划与控制

从当前真实接触出发，以当前手腕、指腹中心和椭球姿态构造轨迹。候选目标来自完整闭合表面的未覆盖区域，优先约 11 cm 的局部段，同时惩罚方向反转、法向差异和重复失败。沿 mesh geodesic 路径平滑后投回原表面，约每 2 mm 检查 IK 和碰撞。手腕以相邻法向的最小旋转连续变化；一段内固定主接触手指。可执行前缀至少 25 mm；连续两段没有增加实测覆盖时转入重定位。

手腕目标使用实际 ellipsoid support vector，初始测得预压与姿态连续保留。A 只向共享 IK 暴露 wrist SE(3)，B 另有四个 Cartesian fingertip positions；不传递 Oracle 私有 joint witness。滑动按表面弧长推进，默认 6 mm/s，起步 0.3 s 渐入，并限制标称关节速度 0.8 rad/s。物理控制指令限制为 1.2 rad/s。

每指 MCC 调节实际法向力，目标 2 N。共同腕部低速法向修正用于补偿几何误差，向内速度最多 1.5 mm/s，偏移范围向外 4 mm /向内 6 mm。最近实测法向与 MCC 状态保留最多 100 ms；控制宽限不改变真实 contact/force 日志，也不增加覆盖。主接触丢失 60 ms 后暂停切向进度，保持当前弧长并局部恢复；最多尝试 2 s 后重定位。短失联不等于整段失败。

≥8 N 时暂停推进并卸力，降到 4 N 以下恢复。>12 N 连续 20 ms、实际非指腹碰撞/自碰撞、关节越界、非有限状态仍停止。物体碰撞参数和非指腹 3 mm 规划间隙沿用原模型。

validation03 的 A11/B19 发现力保护触发后仅冻结手腕、用手指卸力仍会继续压入。
validation04 增加从上一已施加控制目标积分的腕部法向退出（30 mm/s），同时保留手指卸力；
降至 4 N 后从当前实测状态重新规划，不回到刚才导致过载的路径。12 N /20 ms 硬停止阈值未改变。
03 的失败记录保留；04 另建目录、全部从新的 acquisition 运行，不混合作为同版结果。

## 必要重定位

只有局部贴面候选无法形成可行段、连续无覆盖进展、局部恢复超时或控制几何检查受阻时进入重定位，事件记录原因。这里的“必要”是有限候选搜索的结论，不是全局不存在贴面路径的数学证明。

先查法向及世界轴方向的 12 /25 /40 mm 小步退出；无直线退出时查碰撞安全的关节空间路径。重触姿态优先保留当前手形与 roll，只做当前法向到新区域的最小旋转；仍不可行时采用全局几何姿态。离面阶段暂停向内压的 MCC，接近和重触时重新启用。外侧 offset 不可达时允许检查实际表面目标的安全路径。重触需至少 150 ms 安全接触，接近最多 6 mm、5 s。不会通过赋值实体关节位置“跳到”下一点。

规划的计算时间未计入仿真运动时钟；这不构成实时控制性能验证。

## 验证口径

场景仍为原尺寸 mustard、原 256 件 V-HACD 碰撞、固定 fixture、无重力、23 DoF、2 ms 原生 MuJoCo 动力学。每个 seed 从新的共同 5 s acquisition 开始。当前 300 s 验证关注连续滑动效果和必要重定位，不以预算耗尽宣称整瓶完成。

每 10 ms 保存实际状态、先前控制、下一控制、力、法向、触点、覆盖和观测阶段。独立审计从日志重建全部 8192 顶点/388 必访点的 25 mm geodesic 覆盖，并检查 FK、控制限速、初态、源码快照；采样重建 MuJoCo 接触力和位置。覆盖只由真实传感器增加。

连续性指标由单个手指的实际接触轨迹计算，不拼接不同手指，不把短缺失填入严格连续段。存在 normals 时，位移投影至局部切平面；0.5 s 窗口内速度至少 1 mm/s，且实际指尖位移也要印证。有效滑动段需持续至少 2 s、切向跨度至少 10 mm。静止保持、纯法向压入和计划轨迹不算实测滑动。该指标证明接触下的切向运动，不能单凭它区分摩擦滑移与指腹滚动。

同时报告：全时段安全接触率、实测切向移动接触率、滑动阶段指标、严格连续段时长/跨度、离面时长、抬起次数、力峰值、实际覆盖。旧新比较使用相同 seed/初态和共同时间前缀。播放记录原始时间的无剪切视频，同时提供 1 倍速局部特写；视频是记录状态 FK 回放，不增加覆盖或接触证据。

## 复现

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) '.runtime')+';'+(Get-Location).Path
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'
$slideHash=(Get-FileHash Module/i04_geometric_traversal/surface_slide.py -Algorithm SHA256).Hash.ToLower()
& 'D:\Environment\Anaconda\envs\handcomp\python.exe' -m Module.i04_geometric_traversal.surface_slide --output Module/generated/mustard_sliding_fresh/B-MCC_seed7 --cell B-MCC --seed 7 --budget 300 --expected-runtime-sha256 $slideHash
& 'D:\Environment\Anaconda\envs\handcomp\python.exe' -m Module.i04_geometric_traversal.sliding_audit Module/generated/mustard_sliding_fresh/B-MCC_seed7
```

修改输出目录后再运行，拒绝覆盖已有 manifest。A/B 与其他 seed 使用相同参数和运行时；保存每次启动的源码 hash。正式全表面连续滑动验收、自由物体和实机均未由这项 development 验证覆盖。
