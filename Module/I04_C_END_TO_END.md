# i01-b 端到端计时与部署选择

站点早先标为 `i01-ab`，仿真原始记录使用 `I04-c`。

每轮从初始场景启动独立进程，用外层单调秒表计到 R-GPIS 网格和重建输入关闭文件。启动、模型加载、规划或读取腕轨、在线执行、等待、日志及重建均计入。质量评分和视频生成在此后完成。主实验利用仿真最快可用速度，动作以 2 ms 积分、底层 MCC 以 10 ms 更新。

## 两组配对结果

每组 2 对，初态、MuJoCo 模型、物体表面和底层 MCC 参数逐对校验。两次策略种子为 20261109 和 20261110。时间用中位数和最小至最大值报告。接触覆盖按实际指尖头部落点在瓶面的测地邻域并集计，重建使用每轮实测落点、相同 R-GPIS 设置与固定查询范围。

预设的结束质量门槛是 10 mm 覆盖 ≥ 70%、25 mm 覆盖 ≥ 95%、F-score@5 mm ≥ 40%。门槛在整个动作执行完并写出重建后评估。

| 对照 | 方法 | 实际时间中位数 [范围]（s） | 10 mm 覆盖 | 25 mm 覆盖 | R-GPIS F-score@5 mm | 达标 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 完整任务 | 腕指联合规划＋MCC | 288.75 [285.90–291.61] | 81.67% | 100.00% | 44.20% | 2/2 |
| 完整任务 | Finger Policy＋MCC | 494.11 [483.60–504.61] | 72.36% | 97.62% | 44.16% | 2/2 |
| 同一腕轨 | 固定腕在线指端 IK＋MCC | 198.24 [195.41–201.07] | 81.09% | 99.30% | 44.35% | 2/2 |
| 同一腕轨 | Finger Policy＋MCC | 394.26 [391.92–396.61] | 72.36% | 97.62% | 44.16% | 2/2 |

完整任务两对的 DP 减 MCC 时间为 +197.71 s 和 +213.00 s，MCC 的总耗时中位数少 205.35 s。两种方法有 2/2 对同时满足结束质量门槛。

同一腕轨两对的 DP 减 MCC 时间为 +196.51 s 和 +195.54 s，MCC 的总耗时中位数少 196.03 s。两种方法有 2/2 对同时满足结束质量门槛。

完整任务的几何组在线联合规划 6 自由度腕姿和 16 个指关节。DP 的上游从同一初态、同一已知网格重新生成腕轨，使用 I04-c 联合规划器的已接受 MuJoCo 预演推进规划状态，腕轨生成耗时 136.93 s 和 141.65 s。DP 随后从同一初态执行手指策略，前瞻为腕轨上的 200 ms。该测量对应这一套腕轨上游。

同一腕轨组使用固定的 199483 条腕部命令。几何组读取当前触点与未来约束腕姿，在腕姿锁定后在线求 16 个指关节，再用底层 MCC 执行。两组都读同一份腕轨，逐步核对实际发出的腕部命令。DP 的实时观测与预测均在本轮完成。

报告的全程质量来自离线仿真。DP 记录到分区载荷高于 40 N，单轮最长连续时间为 1.79–1.82 s。试验记录超限后继续完成整条参考，几何组未出现该持续超限。高力触发点与最长持续时间存于逐轮结果。

## 各阶段耗时

表内各项为两轮中位数，单位为秒。规划和推理分项已包含在对应执行阶段。外层总时间还计入进程启动及首次载入依赖，独立取中位数的各项可能有舍入差。

| 方法 | 新生成腕轨 | 运行及记录 | R-GPIS | 运行阶段内的规划或推理 |
| --- | ---: | ---: | ---: | --- |
| 完整任务 MCC | — | 254.34 | 29.74 | 105 次分段请求合计 120.22，其中 1282 次腕指求解 10.09、107 次物理预演 105.36 |
| 完整任务 DP | 139.29 | 326.34 | 26.32 | 3983 次去噪与输出转换合计 139.66 |
| 同腕轨 IK＋MCC | — | 165.13 | 28.43 | 3990 次指端 IK 合计 21.62 |
| 同腕轨 DP | — | 365.27 | 26.94 | 3983 次去噪与输出转换合计 182.70 |

## 选用的 Finger Policy

仿真部署采用发布权重 `single_all_seed20261116`，每 100 ms 同步预测 16 个间隔 50 ms 的动作点，保留 10 步 FP32 DDPM、固定系数缓存与 CUDA Graph。控制端每 10 ms 保存观测到预分配环形缓冲，通过 DTW 与五次插值消费动作块。仅动作块查询跨进程发送到 RTX 4090 D，每条完整腕轨有 3983 次查询。两个进程各用其兼容的 MuJoCo 与推理环境。

CPU 配对微基准中，同输入的 593 次查询历史与 5924 次控制输出逐值一致。观测处理的中位数从 0.533 ms 降到 0.435 ms，动作块安装从 2.269 ms 降到 0.823 ms。首对完整任务的两段原生轨迹各有 40 个分片，与对应归档视频的记录逐个匹配。


## 独立实时节拍试验

两组均使用同一腕轨的前 60 s、同一初态和种子，目标 100 Hz。输出迟到定义为命令准备时已超过其 10 ms 周期，指令间隔以连续两次输出之间的墙时衡量。这里的计数来自仿真控制循环。

| 方法 | 控制周期 | 超过周期的输出 | 指令间隔 ≥20 ms | 最大指令间隔 |
| --- | ---: | ---: | ---: | ---: |
| 固定腕在线 IK＋MCC | 6000 | 99 | 13 | 100.34 ms |
| Finger Policy＋MCC | 6000 | 1651 | 572 | 447.33 ms |

DP 的 572 次 ≥20 ms 间隔中有 566 次结束于动作块查询帧。本次实时节拍测试仅覆盖腕轨前 60 s，所选同步 DP 仍未达到每 10 ms 提供新命令的截止要求。

主实验在 i7-13700 和 RTX 4090 D 上运行，数值库为单线程。固定腕几何组和 DP 分别使用两组不同的 P 核，完整任务 MCC 两轮由操作系统分配，部分 CPU 与 GPU 作业重叠运行。GPU 一次运行一个 DP 作业。结果的波动为每个方法两次试验的观测范围。

## 结果与回放

两项对照初态和底层 MCC 相同，重建提交的是开放表面网格。逐轮指标含四组各两次的端到端结果、100 Hz 节拍和推理诊断。[冻结协议](../data/i01-b/protocol.json) · [逐轮指标](../data/i01-b/results.json)

实验站点提供并排回放、表面误差图和 PLY 下载。首对完整任务回放的 40 个原生分片已与归档逐个校验。[i01-b 回放与重建](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#i01-b)

## 复现

从仓库根目录运行。使用 Python 3.10 的 CPU 仿真环境和配好 CUDA 的 Finger Policy 环境。将 CPU 仿真依赖安装到仓库的 `.runtime` 目录，端到端基准的仿真进程会优先从该目录加载 MuJoCo 3.6。策略子进程使用 Finger Policy 环境自己的依赖。该环境另需安装 `gpytorch==1.15.2` 和 `scikit-image==0.25.2` 以执行 R-GPIS。获取 `single_all.pt` 后复制到 `finger_policy/checkpoints/single_all_seed20261116/best.pt`。启动时按权重清单核对 SHA-256。[CPU 仿真依赖](requirements-i04-whole-hand.txt) · [Finger Policy 安装和权重](../finger_policy/README.md) · [权重清单](../finger_policy/checkpoints/manifest.json)

先生成并审核腕指联合规划的参考轨迹，再提取用于同腕轨对照的指令。生成步骤在基准计时之前。完整任务里的 DP 组会在计时内重新计算自身的腕轨。

```powershell
$GeometryPython = 'C:\path\to\handcomp\python.exe'
$PolicyPython = '.\.venv\Scripts\python.exe'
$Reference = 'Module/generated/i01b-reference'
$Wrist = "$Reference/common_wrist.npz"
$Trials = 'Module/generated/i01b-trials'

& $GeometryPython -m pip install --target .runtime -r Module/requirements-i04-whole-hand.txt
& $PolicyPython -m pip install gpytorch==1.15.2 scikit-image==0.25.2

.\scripts\run_i04_c.ps1 -Python $GeometryPython run --planner continuous --output $Reference
.\scripts\run_i04_c.ps1 -Python $GeometryPython audit --case $Reference
.\scripts\run_i04_c.ps1 -Python $GeometryPython benchmark prepare-wrist --case $Reference --output $Wrist

.\scripts\run_i04_c.ps1 -Python $PolicyPython benchmark suite --root $Trials --wrist $Wrist --repeats 2
.\scripts\run_i04_c.ps1 -Python $PolicyPython benchmark suite --root $Trials --wrist $Wrist --branches fixed-mcc fixed-dp --repeats 1 --seconds 60 --paced
.\scripts\run_i04_c.ps1 -Python $PolicyPython benchmark score --root $Trials
.\scripts\run_i04_c.ps1 -Python $PolicyPython benchmark figures --root $Trials
```

原同腕轨为 199,483 条腕部指令。腕轨 SHA-256 随 `prepare-wrist` 输出，以协议中的 `fixed_wrist_sha256` 为准。[冻结协议](../data/i01-b/protocol.json)
