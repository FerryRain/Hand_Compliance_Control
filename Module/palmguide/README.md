# PalmGuide · 腕部主动触觉探索

PalmGuide 是 Active-1-A 保留的 Facing-4 版本。它利用四指历史实触重建的 GPIS 选择下一块探索面片，同时评估手掌朝向、指腹可达性和接触载荷。上层只输出腕部位姿，冻结的 `single_all` 和 MCC 在执行时调节手指。五个 YCB 物体共用这套参数。[数据发布版本](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/c94e90e6cbe01191de87d175e9c1acfdc34c2873/palmguide)

## 腕部如何选下一步

1. 将新获得的指腹触点和法向加入历史地图，用共享 GPIS 更新有触觉支持的表面。候选区域来自当前可达的表面与观测边界。物体真值网格用于仿真碰撞和离线评分。[在线 GPIS](../wrist_active/online_tactile_mesh.py) · [探索面片](../wrist_active/tactile_field.py)
2. 给候选面片配腕部位姿，根据面片法向、掌面几何和实测手型历史，估计哪根指腹能触到目标、掌部余量有多大、移动需要多久，再按预计实触信息收益选择目标和方向。[面片与掌姿评分](../wrist_active/palm_acquisition.py) · [掌面几何](../wrist_active/palm_surface.py)
3. 在两次全局选向之间，局部反馈每 50 ms 调整腕高、倾角和推进量。实触和载荷也会触发重新选向、恢复支撑或卸载。五个 240 秒归档平均实际下发 282.4 次全局探索方向，约 1.18 Hz。手指策略接收触觉和本体状态，输出关节参考，再由 MCC 执行。[掌向跟踪](../wrist_active/palm_facing.py) · [支撑控制](../wrist_active/palm_support.py) · [Facing-4 入口](../wrist_active/palm_unload.py)

## 复跑

从仓库根目录操作。使用项目的 CUDA Finger Policy 环境。归档运行使用 Python 3.10.18、PyTorch 2.8.0+cu129、MuJoCo 3.6.0、GPyTorch 1.15.2、NumPy 1.26.4、SciPy 1.15.3、trimesh 4.8.3 和 scikit-image 0.25.2。下面的下载命令使用 Hugging Face CLI。数据包包含实际使用的手型历史和 `single_all` 权重，准备程序会核对 33 个签名源码文件、权重、手模型和配对初态。[Finger Policy 环境与部署](../../finger_policy/README.md) · [冻结输入清单](inputs.json) · [源码指纹](source-sha256.json)

```bash
hf download FerryZh/full-hand-active --repo-type dataset --revision c94e90e6cbe01191de87d175e9c1acfdc34c2873 --include "palmguide/inputs/*" "active-2/fixtures.zip" --local-dir palmguide-data
python -m Module.palmguide.prepare --fixtures palmguide-data/active-2/fixtures.zip --inputs palmguide-data/palmguide/inputs
python -m Module.palmguide --object pear --seconds 240 --out runs/palmguide/240s/pear
python -m Module.palmguide --object pear --seconds 600 --out runs/palmguide/600s/pear
```

`--object` 支持 `pear`、`mustard`、`cracker`、`bleach`、`coffee`。每个输出目录对应一次运行，包含 `trace.npz`、`plans.jsonl`、`run.json`、`clock.json` 和三组计时数组。输入中的夹具和早期协议提供与归档一致的种子、物理初态和校验值。规划器获得无物体几何的手模型。主线手指执行器从已发布的 `data/i01-b/finger_policy.json` 读取调度尺度，该数值与归档运行使用的配置相同。

发布验证在干净检出中使用上述入口。五个物体分别从零启动并运行 2 秒，各自的 201 帧在 10 个字段上与对应 600 秒归档逐帧一致，涵盖腕和指关节状态、腕命令与前瞻、指令手型、每指接触、触点、法向、载荷和执行阶段。20 组物体、方法与时长组合的覆盖和网格评分，以及双方全部 240 秒全局决策的信息收益，均已从发布数据复核。

## 对照与数据核验

每个物体有一个配对起点和种子。五物体平均为物体等权。240 秒和 600 秒是仿真运动时间。下表的覆盖采用实触点在真值表面上 25 mm 的测地邻域，重建使用 5 mm 双向 F 分数。[逐物体与均值 CSV](https://huggingface.co/datasets/FerryZh/full-hand-active/blob/c94e90e6cbe01191de87d175e9c1acfdc34c2873/palmguide/metrics/endpoints.csv) · [完整视频与时间曲线](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-1)

| 仿真运动时间 | PalmGuide 覆盖 | Active-1-B 覆盖 | PalmGuide F@5 | Active-1-B F@5 |
| --- | ---: | ---: | ---: | ---: |
| 240 秒 | 77.19% | 43.13% | 85.92% | 61.44% |
| 600 秒 | 97.95% | 64.76% | 97.47% | 77.00% |

Active-2 按实际发出的全局探索目标统计。每段收益取截至下一次选向的新增合格实触信息，包含零收益段。前 240 秒每次全局选向实得信息分别为 9.88 和 3.00 nat，每个运动秒分别为 11.56 和 3.23 nat。Aude 对照使用 Active-1-B 的 LEAP 移植与相同初态。[全局决策与计分口径](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-2) · [基线实现](../active_1_b/README.md)

下面的命令从原始轨迹复算每秒覆盖，用已发布的评分网格复算 F@5，并在 240 秒档从原始触觉和规划日志复算每次全局决策的实际收益。将 `--method` 设为 `active-1-b` 可核验 Aude。数据发布也包含独立的 240 秒和 600 秒原始运行、100 个参与评分的 PLY 和五段双系统对照录像。[数据目录](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/c94e90e6cbe01191de87d175e9c1acfdc34c2873/palmguide)

```bash
hf download FerryZh/full-hand-active --repo-type dataset --revision c94e90e6cbe01191de87d175e9c1acfdc34c2873 --include "palmguide/*" --local-dir palmguide-data
python -m Module.palmguide.verify_scores --data palmguide-data/palmguide --object pear --method palmguide --seconds 240
```
