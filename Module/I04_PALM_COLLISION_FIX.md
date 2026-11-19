# I04 掌部穿模修复（2026-10-27）

2026-11-19 保留版说明：最终接续 B 的轨迹、审核和视频完整保留，前期诊断和失败案例已按用户要求清理。当前入口和文件清单见 [I04 A/B 保留版](I04_AB_RETAINED.md)。下方路径与描述属于修复时的历史记录。

旧 B 的掌部详细网格、四个掌部 FSR 外壳及安装转接件的 `contype/conaffinity` 均为 0。
因此碰撞查询、非指腹间隙残差、命令守护和原生接触核验都看不到这些部件。
原生审计可以精确重放这个有遗漏的模型，不能证明被禁用的掌部几何没有穿透。
旧 B 的 100% 是该模型下的指腹测地邻域覆盖，不能继续作为“全手无穿模完成遍历”的证据。
旧文件保留用于复现与诊断；修复采用新的运行入口和案例目录。

## 已确认的原轨迹问题

原 B 以约 1 秒间隔抽样 4860 个保存状态，启用掌部碰撞的诊断副本在其中 4309 个状态
检测到大于 0.5mm 的凸包重叠。这是抽样统计，不是 2ms 全轨迹的精确时间占比。
最深抽样状态位于约 t=1861s，保守凸包重叠约 59.99mm。
另用实际掌部三角网格顶点对原 Mustard 闭合外表面检查：24224 个掌部顶点进入物体内部超过 1mm，
最深顶点约 47.29mm。因此不是仅由凸包填平开孔引起的误报。

## 修复方法

- 新运行模型在编译前启用原有的六个刚性掌部部件：手腕转接件、掌座和四个 FSR 外壳。
  使用独立碰撞位与物体交互，不增加掌部与指腹之间的自碰撞对。
- 初始全手拟合、连续表面目标拟合、换面候选及路径检查都考虑掌部间隙。
  初始拟合和逐次目标拟合以前没有完整的身体间隙残差，本入口补上该项。
- 规划倾向保留 6mm 间隙，命令检查沿用 3mm 非指腹间隙门槛；
  控制每个 2ms 状态记录实际掌部距离，低于 1mm 时停止推进。
  原有正间隙退出、四指 MCC、失触恢复和真实 `mj_step` 执行继续使用。
- 设置 `margin=gap=6mm`，使正距离进入接触查询而不提前产生接触力。
  真正重叠仍会产生物理接触。[MuJoCo 参数定义](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#body-geom-gap)
- 不移动物体、不缩小或隐藏手掌，不通过修改原轨迹关节角生成“修复画面”。

这里六个掌部部件使用 MuJoCo 的保守凸包碰撞。原几何、质量、惯量、关节和物体 256 件碰撞体保留。
该修复覆盖刚性掌部和安装件；详细手指外壳沿用原模型，指腹接触仍由原四个 pad 判定。
新 B 的物理条件已变化，不能与旧 A 继续当作同物理条件的正式对照。

### 调姿跟踪与主动退让

第一轮新案例 `i04b_palm_safe01_20261027` 从零覆盖执行 5774.788s，覆盖 98.3816%，
8026/8192 顶点及 383/388 目标。2887394 个已记录状态独立核验通过。
它在必要调姿时因真实跟踪滞后触发 `PALM_CLEARANCE_STOP`；最后一步之后仍有 0.992mm 正间隙，未穿透。
这一轮没有完成整瓶，记录与完整视频保留。

新增 `palm_safe_recovery` 在离面调姿和撤离时，用实际接触法向 Jacobian 与实际关节速度
预测未来 120ms 的间隙。预计或当前间隙小于 4mm 时中断路径，逐步生成向外的受限命令；
参考已安全而实测仍滞后时，只在双几何守护通过后保持参考，等待真实伺服跟上。
实测恢复到约 5.8mm 后重新规划，保留原 1mm 停止底线。

相同危险状态的局部回归 `palm_retreat_regression02_20261027`：约 92ms 后从 4.024mm
恢复到 5.838mm，最低 3.822mm；46 个 2ms 状态和全部双命令守护独立核验通过。
这只是恢复回归，不计为整瓶实验。

当前接续案例 `i04b_palm_recovery02_20261027` 保留第一轮的相同已核验前缀，
从 t=5774.600s 的真实 q、速度、上一命令与 warmstart 接续，替换原最后约 188ms 的逼近片段。
没有重置物理状态，没有继承未来覆盖；控制器的名义参考和搜索状态重新建立。
该案例明确记录版本边界，不冒称从零运行的单一版本结果；整条接续轨迹另从初始状态连续核验。

接续版最终在6918.226s完成8192/8192顶点、388/388目标及末端接触保持，
3459113个原生状态、全部6918226次双命令检查通过完整连续独立核验。
最小已记录掌部间隙1.314886mm，13次主动退让成功，中位耗时约40ms仿真时间；
原始仿真接触力峰10.661406N，原8N覆盖排除和40N/200ms停止规则保留。
覆盖仍指合格指腹接触的25mm测地邻域累计，不表示每个网格点都被指腹直接压到。
全时间轴至少三指同时合格接触47.913%，四指同时16.706%；滑动阶段至少一指合格接触98.175%。
最长全手合格接触缺口169.724s；它还包括原判据排除的不合格或高力接触，不直接等于物理离面时长。
这次完成覆盖与掌部防穿模修复，换面连续性仍有优化空间，视频保留全部缺口。

## 运行新版本

从仓库根目录运行，确保数据已通过 Git LFS 下载：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.runtime') + ';' + (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$py = 'D:\Environment\Anaconda\envs\handcomp\python.exe'

& $py -m Module.i04_geometric_traversal.palm_safe_recovery --output Module/generated/my_palm_safe_b --seconds 9000
& $py -m Module.i04_geometric_traversal.palm_safe_audit --case Module/generated/my_palm_safe_b --workers 2
& $py -m Module.i04_geometric_traversal.b_full_portable video --case Module/generated/my_palm_safe_b
& $py 'Module/i04_geometric_traversal/trajectory_replay.py' --case Module/generated/my_palm_safe_b --speed 12
```

输出目录必须不存在。审计可加 `--follow` 随已写完的轨迹分块连续重放。
如只运行仿真而尚未执行审计，先运行
`python -m Module.i04_geometric_traversal.b_full_portable assets --case <新案例目录>`，
生成该新案例的同哈希资产定位表，再用交互回放入口打开。

精确复现本机的版本接续（父案例需完整下载，新输出目录必须不存在）：

```powershell
& $py -m Module.i04_geometric_traversal.palm_safe_recovery --output Module/generated/my_palm_continuation --parent Module/generated/i04b_palm_safe01_20261027 --join-time 5774.60 --seconds 9000
```

`palm_safe_mcc` 保留为第一轮仅有停止保护的入口；新使用者优先选择 `palm_safe_recovery`。
本机接续版完整视频复用经过逐项数据校验的相同前缀画面，只渲染新接续段，再按全局原生采样网格合成。
不插值、不补造姿态或接触；合成步骤重新编码视频，因此压缩字节与旧视频不同。

## 核验

`Module.tests.test_i04_palm_collision` 覆盖漏检复现、真实重叠产生接触力、正间隙无提前力、
编译后身体碰撞掩码更新、原几何/质量/惯量保留、指令拒绝时日志保持对齐，
以及实际靠近/远离速度的预测符号；六项测试通过。

同一条旧 B t≈1861s 的实测/上一参考双守护回归中，旧模型两侧均放行，
新模型两侧均以 `NON_TIP_OBJECT_COLLISION` 拒绝。

35 秒短段 `generated/palm_safe_probe01_20261027` 已完成 17501 个原生状态和双命令守护的独立核验。
最小掌部间隙 5.845mm，记录六次路线/转向尝试；四指同时接触及贴面滑动均已观察。
该短段只覆盖约 5.70%，不是新的整瓶完成结果。

最新整瓶验证案例：`generated/i04b_palm_recovery02_20261027`。
运行状态、最终覆盖及独立核验以该案例的 `progress.json`、`summary.json` 和
`independent_native_audit/audit.json` 为准。

每个案例保存启动时的 `source_snapshot` 和逐文件哈希。
第一轮启动后工作源码补了异常分支的日志对齐清理；它不改变成功 tick 的控制计算。
最新接续案启动后控制源码保持冻结；各版本精确复现以各自快照为准。
接续 manifest 中继承的 `source_trial` 文案描述父入口；实际接续性质以 `resume`、
`fresh_single_version=false` 和 `coverage_inherited=true` 为准。

对比截图中左侧为旧 B 的实际 t=1861s，右侧为新短段的实际 t=15s；
这是两个独立运行的保存状态，不是同一状态被事后移动的前后对比。
