# PalmReach-6

**PalmReach-6 是主动触觉探索主实验的主实现。** 它对应原始记录中的 `profiled`。全局规划比较整掌运动，局部 active 维持腕部贴合，冻结的 `single_all` 手指策略与 MCC 执行接触适应。配对对照是逐指目标选向的 Facing-4，以及 LEAP 上的 Aude 腕指联合控制。[主实验数据](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/df447aa996be0aaa634a3e319f3a39aa98f11045/palmreach-6) · [对照视频与指标](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/active-methods.html)

## 一次腕部决策

1. 合格指腹触点和法向更新整手共享的 GPIS。根据手型、哪些手指正在接触和候选腕动，查询从 13 段 `single_all` 轨迹拟合的四指条件可达域。[拟合与查询](../wrist_active/policy_reach.py)
2. 比较六个切向运动方向和原地保持。用预期新增触觉信息、至少一指与多指支撑的概率，以及运动用时给整掌动作评分。[整掌评分](../wrist_active/palm_exploration.py)
3. 局部 active 根据实触表面调整腕高与倾角。持续零指轻触 0.75 秒，或单指持续支撑且没有新增实触格达到阈值时，允许全局在当前 GPIS 网格上规划复触。单指停滞阈值在饼干盒为 6 秒，其余物体为 1.5 秒，复触重试间隔均为 0.75 秒。[局部控制](../wrist_active/palm_support.py) · [触发条件](../wrist_active/palm_progress_recontact.py) · [发布参数](../wrist_active/retained_object_parameters.json)

## 复现主实验

在仓库根目录、项目 CUDA Finger Policy 环境中运行。复现使用 Python 3.10、NumPy 1.26.4、SciPy 1.15.3、PyTorch 2.8.0+cu129 和 MuJoCo 3.6.0。准备程序校验冻结源码、可达域、手指权重、手模型和配对初态的 SHA-256。[Finger Policy 环境](../../finger_policy/README.md) · [输入清单](inputs.json) · [控制器指纹](source-sha256.json)

```bash
hf download FerryZh/full-hand-active palmreach-6/inputs/runtime.zip --repo-type dataset --revision df447aa996be0aaa634a3e319f3a39aa98f11045 --local-dir palmreach-data
python -m Module.palmreach6.prepare --archive palmreach-data/palmreach-6/inputs/runtime.zip
python -m Module.palmreach6 --object cracker --seconds 240 --out runs/palmreach6/cracker-240
python -m Module.palmreach6 --object cracker --seconds 600 --out runs/palmreach6/cracker-600
python -m Module.palmreach6 --method facing-4 --object cracker --seconds 240 --out runs/facing4/cracker-240
python -m Module.palmreach6 --method aude --object cracker --seconds 240 --out runs/aude/cracker-240
```

`--object` 可取 `coffee`、`cracker`、`mustard`、`bleach`、`pear`。上述三个方法共用每个物体的 `start-02` 模型、初态和种子。数据包里的 `release/PalmReach-6.zip` 保存选定版本的原始源码、权重和运行指纹。该快照的 Windows 启动脚本沿用开发时的物体初态，正式配对使用上面的 Python 入口。[实验协议](protocol.json) · [饼干盒选参记录](selection.json)

## 正式结果

每个物体一个配对起点，两个时长独立采集，五物体等权平均。C25 是合格实触点在物体真值表面 25 mm 测地邻域内覆盖的面积比例，仅用于离线评价。F@5 是 5 mm 阈值下重建表面的双向 F 分数。[逐次结果与原始指标](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/df447aa996be0aaa634a3e319f3a39aa98f11045/palmreach-6/metrics)

| 方法 | 240 秒 C25 | 240 秒 F@5 | 600 秒 C25 | 600 秒 F@5 |
| --- | ---: | ---: | ---: | ---: |
| **PalmReach-6** | 53.68% | 69.30% | 84.68% | 88.86% |
| Facing-4 | 77.70% | 84.93% | 93.33% | 94.99% |
| Aude | 38.67% | 55.15% | 55.06% | 69.89% |

发布核验从干净检出复跑五物体各 2 秒，将 201 帧的 10 个腕指和接触字段与各自 600 秒归档逐帧比对。13 段训练轨迹包含 16,463 个可达域窗口，重新拟合所得 NPZ 与实测模型的 SHA-256 一致。30 条正式归档均已从实触轨迹复算 C10、C25，并从终点 PLY 复核 F@5。

正式数据按时长、方法和物体分别打包。每包同时包含原生轨迹、规划日志、运行指纹、评分时间序列和终点网格。数据集中另外收录 12 次饼干盒选参、4 次独立起点验证、可达域拟合轨迹、开发消融、逐分钟网格和回放。[正式轨迹](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/df447aa996be0aaa634a3e319f3a39aa98f11045/palmreach-6/formal) · [历史 Facing-4 实现](https://github.com/FerryRain/Hand_Compliance_Control/tree/0599970a287b5aaa259470b1482a54d5f09df092/Module/palmguide)

准备上述运行输入后，可直接从单个正式试验包重算每秒 C10、C25 和终点网格 F@5。

```bash
hf download FerryZh/full-hand-active palmreach-6/formal/240s/palmreach-6/cracker.zip --repo-type dataset --revision df447aa996be0aaa634a3e319f3a39aa98f11045 --local-dir palmreach-data
python -m Module.palmreach6.verify --object cracker --seconds 240 --archive palmreach-data/palmreach-6/formal/240s/palmreach-6/cracker.zip
```

完整数据版本共 208 个文件，逐文件校验和可用于核验单独下载的试验包。[SHA-256 清单](https://huggingface.co/datasets/FerryZh/full-hand-active/blob/df447aa996be0aaa634a3e319f3a39aa98f11045/palmreach-6/SHA256SUMS.txt)
