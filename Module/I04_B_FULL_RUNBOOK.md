# I04B：恢复整手遍历与仿真高力容忍

本入口响应2026-10-20“给我完整的B，仿真里面的高力可以适当忽略”的要求。仅运行MCC。
它从共同6mm悬停初态、零覆盖开始，恢复原B的整手执行方式，再对遗漏区域补扫。
这是新的B开发实验；尚未用这些执行条件重新运行A，不能与旧A直接解释为单因素对照结论。

已完成案例`i04b_full_force_grace01_20261020`：4859.048秒达到100%，8192/8192顶点和388/388目标，
2,429,524个原生状态连续独立重放通过。原始力峰9.0952N，最长全手断触62.808秒。

- [完整报告与六方向定位](generated/i04b_full_force_grace01_20261020/review.html)
- [正常速度完整视频：80分59.10秒](generated/i04b_full_force_grace01_20261020/complete_videos/b_complete_1x.mp4)
- [12倍速完整总览：6分45秒](generated/i04b_full_force_grace01_20261020/complete_videos/b_complete_12x.mp4)
- [实验目的、方法、结果与限制](evidence/2026-10-20_I04_B_FULL.md)

## 做了哪些调整

- 恢复起步时的四指稳定接触窗口；后续按局部几何可支持的2/3指人数以及当前参考手指确认推进。
- 恢复每根手指的曲面法向转角限速，正确更新换面使用的参考手指。
- 目标排序只读取现有触点参考，不在每轮排序时突然把整手目标重新投影到表面。
- 先沿整瓶表面寻找贴面路线；达到97%或该阶段搜索耗尽后，用同一物理状态继续补扫遗漏区域。
- 仿真力超过12N只记录警示；超过40N且连续100个2ms采样才停止。原始力不截断、不删除。

MCC每指1N目标、2.5N主动卸力/低于1.2N恢复、关节范围、步长限制、
非指腹/机械臂碰撞检查以及实态/上一参考双几何守护保持原实现。
这些阈值是本次仿真设置，不是机器人实物的力限值。

覆盖仍按原始中央掌侧指腹和25mm网格边测地邻域计算；任一手指原始力达到8N时，
该状态不增加覆盖。完整分母为8192顶点和388目标，最后还需150ms实际合格接触保持。
100%表示这个冻结指标完成，不是皮肤压力足迹覆盖了每一毫米表面。
报告另给出只去掉8N上限排除的覆盖统计，保留实际中央指腹、非指腹/臂碰撞、触点距离、
法向、原0.2N接触检测门槛、25mm测地邻域和完整分母。它用于量化仿真高力对计分的影响，不改写本次控制或严格完成判据。

## 安装和启动

从仓库`main`下载代码与Git LFS数据。环境安装、资产说明见
[原整手运行说明](I04_WHOLE_HAND_RUNBOOK.md)。
Windows已使用Python3.10和MuJoCo3.6.0；物理仿真不需要CUDA或DP checkpoint。
重新渲染中文视频目前使用Windows微软雅黑字体；已生成的视频可直接在其他系统观看。

在仓库根目录、已安装`Module/requirements-i04-whole-hand.txt`的Python环境执行：

```powershell
git lfs pull
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'
python -m Module.i04_geometric_traversal.b_full_mcc --output Module/generated/my_b_full_smoke --seconds 20
python -m Module.i04_geometric_traversal.b_full_mcc_audit --case Module/generated/my_b_full_smoke --workers 4
```

长程实验使用另一个不存在的输出目录：

```powershell
python -m Module.i04_geometric_traversal.b_full_mcc --output Module/generated/my_b_full --seconds 9000
python -m Module.i04_geometric_traversal.b_full_mcc_audit --case Module/generated/my_b_full --workers 4
python -m Module.i04_geometric_traversal.b_full_video --case Module/generated/my_b_full
python -m Module.i04_geometric_traversal.b_full_review --case Module/generated/my_b_full
```

`--seconds`为物理仿真预算，不是计算耗时或保证完成的时刻。必须检查`summary.json`及独立核验结果，
不能仅根据脚本正常退出判定整瓶已完成。输出目录不能复用或覆盖。
几何IK种子来自发布版run10，但不继承它的物理状态、命令或任何覆盖。
新实验会重新规划；其中包含墙钟搜索预算，实际路线可能随计算环境变化。
对已发布轨迹的逐状态复现使用独立`audit`。

## 边运行边核验和制作视频

运行开始并生成`manifest.json`及首个`trace_chunks.json`后，可在另外的终端执行：

```powershell
python -m Module.i04_geometric_traversal.b_full_mcc_audit --case Module/generated/my_b_full --workers 4 --follow
python -m Module.i04_geometric_traversal.b_full_video --case Module/generated/my_b_full --follow
```

独立核验从t=0连续重放所有2ms状态，分块不重置物理；重新计算力、指腹接触、覆盖，
并检查所有实际施加命令的两类几何守护。新核验器只适配本版的高力停止条件及多阶段完成标记。
最终通过记录为`independent_native_audit/audit.json`中的`PASSED_NATIVE_EVIDENCE_AUDIT`。

阶段切换时可能已有中间`summary.json`；实时跟随必须等待`run_complete.json`。
`progress.json`只用于进度显示，不是最终结果。再次核验需选新的`--output-name`。

视频位于`complete_videos/`：

- `b_complete_1x.mp4`：正常速度完整时间轴，20fps，包含全部离面、卸力、恢复及结束状态。
- `b_complete_12x.mp4`：同一完整时间轴的12倍速总览。
- `video_manifest.json`：原始轨迹哈希、取帧规则、首末仿真时刻、视频帧数及SHA-256。

视频只设置原始保存的实际关节状态，不运行新物理、不补造接触或姿态。
粉色点来自实际合格指腹接触；三视图分别显示全景、长指和拇指。
渲染完成不等于独立核验通过，最终以对应案例的核验与报告为准。
`b_full_review`要求最终核验通过，再生成覆盖图、接触/力统计、完整视频解码检查和`review.html`。

## 在另一台电脑重放下载的案例

克隆`main`并执行`git lfs pull`后，先检查本次发布的代码、报告、视频和全部原生轨迹：
本次校验清单599项，共约4.53GB；正常速度视频约1.59GB，12倍速总览约0.22GB。

```powershell
python -m Module.i04_geometric_traversal.b_full_portable verify --case Module/generated/i04b_full_force_grace01_20261020
```

此命令逐文件比较[发布清单](evidence/2026-10-20_I04_B_FULL.json)中的字节数和SHA-256，
可识别尚未下载的LFS指针；它不重新运行物理。

原始`model.xml`中的作者路径保留不改。`b_full_portable`按哈希把公共机器人资产和案例内
257个生成网格重新映射到当前克隆；所有290个模型资产必须有明确映射，不能依赖作者的磁盘目录。

```powershell
python -m Module.i04_geometric_traversal.b_full_portable smoke --case Module/generated/i04b_full_force_grace01_20261020 --states 500
python -m Module.i04_geometric_traversal.b_full_portable audit --case Module/generated/i04b_full_force_grace01_20261020 --workers 4 --output-name downloaded_native_reaudit
python -m Module.i04_geometric_traversal.b_full_portable video --case Module/generated/i04b_full_force_grace01_20261020 --output-name downloaded_complete_videos
```

`smoke`检查前500个原生状态和对应双几何守护，不能代替完整轨迹核验。
完整`audit`仍使用原来的40N/200ms停止政策与8N覆盖排除规则，只改变资产文件路径解析。
生成资产的映射与SHA-256见案例的`portable_assets.json`。

## 数据与版本边界

`manifest.json`冻结本版协议`i04.b.full-mcc.force-grace.v1`、力政策和控制源快照。
`trace_*.npz`保存全部2ms状态、施加前后命令、四指原始力、指腹触点、覆盖和执行阶段。
新增`force_warning_12n_steps`、`force_stop_40n_steps`、`traversal_stage`便于核对警示、终止与阶段切换。
`initial.npz`、`final_state.npz`、`events.json`及分块哈希一起保留。

历史17.62%的B、已完成100%的run10和原A均保留原协议、原力规则及原数据。
它们的区别和已确认的回归原因见[回归检查](evidence/2026-10-20_I04_MCC_REGRESSION.md)。
未来重新做公平A/B时，须将非对照执行条件共同冻结后再运行两组。
