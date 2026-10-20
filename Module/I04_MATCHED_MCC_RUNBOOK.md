# I04A-MCC / I04B-MCC 新执行层运行说明

协议：`i04.ab.whole_hand.v2.mcc.1`。本入口用于2026-10-19开始的新MCC对照运行。
原run10和旧`benchmark`保留原来的协议；DP/DPRef继续暂缓。
2026-10-20复查：此新入口存在已确认的B行为回归，不能视为保留旧B成功行为的基线。
起步/滑动门槛、目标重投影及锚指维护的变化详见[回归诊断](evidence/2026-10-20_I04_MCC_REGRESSION.md)；
[A/B正常速度完整视频](generated/i04_mcc_regression_review_20261020/review.html)已另附，原失败数据保留。
实验目标与共同约束见 [I04_AB_WHOLE_HAND_PROTOCOL.md](I04_AB_WHOLE_HAND_PROTOCOL.md)。

## 1. 这次比较什么

共同任务是从零覆盖开始，尝试遍历完整Mustard，包含瓶身、侧面、肩部、瓶盖与瓶底。
8192顶点、388目标和25mm测地邻域分母固定，只有实际合格的中央掌侧指腹接触才能计分。
完整覆盖要求全部顶点、全部目标和至少150ms安全保持。达到时间/搜索预算而未覆盖完整物体仍保留全部结果。

| 项目 | A-MCC | B-MCC |
| --- | --- | --- |
| 方法侧目标 | 只有完整手掌位姿及参考序列 | 手掌目标和各指表面位置/法向 |
| 名义手指 | 固定为共同声明初态的手指关节值 | 联合几何规划出的多指目标 |
| 腕规划 | 从自己的状态解腕部IK，名义手指固定 | 手掌和手指共同求解 |
| 执行 | 公共四指MCC、姿态先验、卸力、碰撞保护 | 同一函数、同一参数 |

A的“固定名义手形”不等于锁住手指：接近、贴面和重触阶段四指的MCC开启，执行器的23关节有界求解仍可使手指屈伸，
响应实测力、保持自然关节范围并主动接近。它不读取B的逐指GT落点、法向、完整关节参考或几何种子。
离面阶段共用调姿路径执行器与实测卸力规则。
A内部的逐指控制参考来自机器人自身FK；B的逐指参考来自物体有限曲面。这是保留的实验差异。

私有Oracle可读取物体GT和已发布几何姿态库，按本方法实际状态/覆盖寻找下一任务。
A适配器仅向`WristOnlyReference`传入`WristTarget`。两组使用同一目标选择规则，状态分化后实际目标序列可以不同。
安全层私有几何查询只用于碰撞判定、间隙保护和避障，不向A提供用于寻触的表面落点。

## 2. 实际共用的执行与恢复

`matched_mcc.py`的两组直接调用同一个`WholeHandTraversal.contact_step/measure/tick/flush`，
以及同一个`RecoveryWholeHandTraversal.air_move`；原已发布控制源码未改写。
公共MCC为每指1N、质量0.08、阻尼70、刚度20；2ms原生步长和日志；偏移4mm、速度0.25mm/s、加速度0.02m/s²。
主动卸力2.5N、恢复1.2N；≥8N不授予覆盖，>12N持续20ms停止。
双几何守护、非指腹3mm/臂1mm间隙与自然关节范围沿用原实现。

初次接近和换面后接近共用实测人数窗口：最近300ms中至少80%时间有两指合格接触，且当前至少两指。
不读取`expected_contacts`，没有特殊锚指必须接触的门槛。
滑动在实测至少两指、无卸力/混合接触/连续守护拒绝时推进；否则暂停。连续不足两指6秒触发重规划，不直接判整轮失败。
接近预算至少12秒；换面卸力暂停最多40秒，与路径跟踪时间分别记账。
连续16轮没有新增覆盖才触发共享搜索耗尽停止。完整表面搜索、候选失败与各类停止均写入事件/摘要。

法向滤波状态与MCC偏移保留本方法自身历史。实际接触时优先使用符合一致性条件的实测法向，
无实测时A使用自身名义指腹朝向，B使用其规划法向；滤波与限幅算法完全一致。
新区域接近/转移失败的公共重初始化按同一规则执行，不继承另一个方法的接触历史。

当前是seed7单种子实现试验。seeds11/19、DPRef及其固定腕参考重放不在本次完成范围内。
规划含墙钟搜索限额，因此同版本重新规划不承诺逐命令完全相同；已经保存的轨迹可按唯一初态精确物理重放。

## 3. 安装和启动

先按 [已发布版运行说明](I04_WHOLE_HAND_RUNBOOK.md) 下载LFS资产并安装
`Module/requirements-i04-whole-hand.txt`。所有命令从仓库根目录运行。

本轮数据下载后可先执行字节完整性检查（不重新运行物理）：

```powershell
python -m Module.i04_geometric_traversal.matched_mcc_delivery verify
```

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'

# 先产生A及唯一公共初态。输出目录必须不存在。
python -m Module.i04_geometric_traversal.matched_mcc `
  --cell A-MCC --seconds 7000 --output Module/generated/my_i04a_mcc

# B复制A的完整物理初态，而不复制接触、控制历史或覆盖。
python -m Module.i04_geometric_traversal.matched_mcc `
  --cell B-MCC --seconds 7000 --output Module/generated/my_i04b_mcc `
  --initial Module/generated/my_i04a_mcc/initial.npz
```

7000秒是仿真时间上限，不是墙钟耗时估计。规划、仿真、压缩与独立重放另耗计算时间。
当前Oracle离线求解时暂停仿真计时；因此该试验不是实时控制性能测试。
`--local-only --seconds 16`只做局部接近和保持验证，不遍历物体，不能当作整瓶对照结果。
不要在运行中修改控制源码；新参数/修复需要另开从零运行，保留旧失败记录。

## 4. 核验、视频和对照页

```powershell
python -m Module.i04_geometric_traversal.matched_mcc_review audit `
  --case Module/generated/my_i04a_mcc --workers 4
python -m Module.i04_geometric_traversal.matched_mcc_review render `
  --case Module/generated/my_i04a_mcc --speed 24
python -m Module.i04_geometric_traversal.matched_mcc_review report `
  --case Module/generated/my_i04a_mcc

# 可选：不跳帧的正常速度区间节选，明确保留实际起止时间。
python -m Module.i04_geometric_traversal.matched_mcc_review render `
  --case Module/generated/my_i04a_mcc --speed 1 --start 60 --end 70
# 增加视频后重新运行report，将节选加入页面。

# 对B执行上述audit/render/report后，生成统一口径的比较。
python -m Module.i04_geometric_traversal.matched_mcc_review compare `
  --a Module/generated/my_i04a_mcc --b Module/generated/my_i04b_mcc `
  --output Module/generated/my_i04_mcc_comparison
```

审计输出目录必须不存在，防止覆写已有证据。审计从唯一初态连续重放全部原生状态，
使用独立指腹分类器重新计算力、覆盖和实际运动，再检查每条已执行命令的双几何守护。
视频只读取已核验的实际关节状态；加速覆盖完整时间轴，不删除离面、卸力或失败片段。
中文渲染使用Windows微软雅黑字体；查看现成HTML/MP4无需安装仿真环境。

比较工具先检查协议、公共配置哈希、公共源码哈希、完整初态哈希一致，再统计两案共同时间前缀。
完整终态、停止原因及未覆盖区域另行保留，不将不同运行时长的终点直接当作等时效果。
单种子差异不等于统计结论；旧run10开发接续不能替代同版本的新B参照。

## 5. 文件与信息边界

- `initial.npz`：qpos、qvel、ctrl、warmstart和时间；每案覆盖从零开始。
- `manifest.json`：方法、公共参数/源码/初态哈希、输入权限、名义手形和预算。
- `source_snapshot/`：该案实际执行的冻结源码。
- `nominal_wrist_reference.jsonl`：公开手掌位姿序列及路线参数/转移时间，保留SHA-256。
- `trace_*.npz`、`trace_chunks.json`：全部2ms状态、施加命令、力、接触、覆盖、参考来源及分块哈希。
- `events.json`：贴面、换面、重规划和腕部IK拒绝等事件；`summary.json`记录真实停止原因。
- `independent_native_audit/`：完整原生核验与独立预览。
- `recovery_metrics.json`：全手失联、初次接近、首次重触、下次稳定支持和终点截尾分别记账。
  原生段数包含2ms抖动，另报至少100ms/1s/6s的缺口，不能把每次短暂重触都视作稳定恢复。
- `input_boundary_record.json`：公开输入模式和法向来源检查；结合源码边界测试使用，不宣称进程级安全隔离。
- `review.html`、`video_manifest.json`、MP4和图：中文报告与完整时间轴视频。
- `Module/evidence/2026-10-19_I04_MATCHED_MCC.json`：本轮全部交付文件的大小和SHA-256。

原生日志的`public_wrist_pose_world`在贴面阶段记录本次手掌参考；在离面路径中保留最近贴面请求。
离面名义序列及时间应以JSONL为准，实际执行以`command_rad/q_rad`为准，不能把这个单字段误当全程连续腕轨迹。
`input_normal_source`记录贴面MCC的法向选择上下文；离面期间不是独立的逐步法向来源追踪。
离面卸力函数直接读取当时实测原始法向，参见共同执行源码。贴面MCC使用2.5N/1.2N滞回；
离面执行器沿用原版的即时2.5N卸力暂停规则。两组执行同一规则，报告不混淆二者。
未来A-DPRef接入前还需实现完整冻结腕序列的调度重放，不能直接重新跑闭环Oracle后称两条腕输入相同。

边界与回归检查：

```powershell
python -m unittest Module.tests.test_i04_matched_mcc -v
python -m unittest discover -s Module/tests -p "test_i04*.py"
```
