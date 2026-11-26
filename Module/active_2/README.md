# Active-3 · 五种控制条件的主动探索比较

对每个 YCB 物体，五种控制条件从相同的 `start-02` 初态和种子出发，各自闭环探索 240 秒。梨的种子为 20261120，其余四个物体为 20261122。标准 AUDE 复用 Active Exploration 的五条正式轨迹。原三组共 15 轮，两个附加对照各 5 轮。25 轮使用同一套实触覆盖率与有符号 GPIS 重建评分。[逐物体结果与原三组回放](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-3) · [Active Exploration 的 AUDE 基线](../../PARK/scores.csv)

| 五物体等权均值 | 25 mm 实触覆盖率 | 重建 F@5 mm | ≥2 指有效接触时间 | ≥3 指有效接触时间 |
| --- | ---: | ---: | ---: | ---: |
| AUDE | 38.67% | 55.15% | 42.49% | 24.45% |
| single_all（AUDE 指尖力矩） | 33.10% | 47.52% | 39.38% | 17.29% |
| 仅底层柔顺 | 22.70% | 33.77% | 23.92% | 8.33% |
| AUDE 腕规划 + single_all/MCC | 14.43% | 28.05% | 18.32% | 6.96% |
| AUDE + 掌轴距离约束 | 37.25% | 53.45% | 47.50% | 22.91% |

**AUDE 腕规划 + single_all/MCC**　与标准 AUDE 相比，平均 C25 低 24.24 个百分点，平均 F@5 低 27.10 个百分点。与使用 AUDE 指尖力矩的原 single_all 组相比，≥2 指接触时间占比低 21.06 个百分点，≥3 指占比低 10.33 个百分点。

**AUDE + 掌轴距离约束**　与标准 AUDE 相比，平均 C25 低 1.42 个百分点，平均 F@5 低 1.69 个百分点。两项指标在梨、饼干盒上均高于标准 AUDE，在芥末瓶、漂白剂瓶上均较低。训练中央 95% 区间为 100.76–146.80 mm，中位数为 128.53 mm。

所有均值均用五个物体的未舍入值等权计算。C25 为参考表面上距注册实触点 25 mm 测地范围内的面积占比。F@5 是 5 mm 容差下最终重建网格相对于参考表面的精确率和召回率的调和平均。接触占比以初始帧之后的 24,000 个 100 Hz 采样帧为分母，计入的指腹力 ≥0.2 N、法向模长 &gt;0.99，且全手各区载荷均 &lt;8 N。

## 控制条件

上层均每 0.9 秒按本组实测触觉更新一次 AUDE 腕指联合计划，伺服频率为 500 Hz。

| 运行参数 | 腕部与手指执行 |
| --- | --- |
| `planned` | 标准 AUDE 执行规划的腕姿及手指解，使用原版指尖力矩控制。 |
| `single_all` | 执行规划腕姿，由冻结的同步 `single_all` 提供手指目标，再由 AUDE 指尖力矩执行。芥末瓶的关节指令限幅为 0.015 rad/控制步，其余为 0.020，失触后的额外重试关闭。 |
| `compliance` | 执行规划腕姿，指尖位置跟踪项为零，保留底层法向力、速度阻尼和弱零空间回正。 |
| `aude_mcc` | 执行 AUDE 规划腕姿。手指由原生缓冲式 `single_all` 生成目标，经 MCC 指尖力反馈和位置伺服执行。上层手指解仅记录。 |
| `aude_height` | 保留 AUDE 的腕指联合规划、手指解与直接指尖力矩，仅对腕部沿掌心朝向至已触表面的距离加约束。 |

距离带按每个 50 ms 训练帧的有效指腹接触距离中位数标定。原芥末与扩充训练集各占 50% 权重，分别含 49,348 和 146,539 个有效实触帧。复算脚本读取原芥末与扩充数据的训练划分，并核对发布数据的指纹。距离约束组每次规划根据最近一次有效指腹接触修正腕目标。[训练标定记录](height_calibration.json) · [标定复算脚本](calibrate_height.py)

| 物体 | 修正次数 / 规划次数 | 最大修正（mm） |
| --- | ---: | ---: |
| 梨 | 55/267 | 17.89 |
| 芥末瓶 | 50/267 | 11.48 |
| 饼干盒 | 50/267 | 39.88 |
| 漂白剂瓶 | 2/267 | 2.67 |
| 咖啡罐 | 21/267 | 15.71 |

下表给出实测掌轴距离处于训练区间内的比例。全程以初始状态之后的 24,000 帧为分母，失触时沿用最近触点。“新实触”只以当帧有有效触点的帧为分母。两者均用实际腕姿计算距离。

| 物体 | 标准 AUDE（全程） | 限距组（全程） | 标准 AUDE（新实触） | 限距组（新实触） |
| --- | ---: | ---: | ---: | ---: |
| 梨 | 76.07% | 88.78% | 82.35% | 95.06% |
| 芥末瓶 | 81.30% | 97.99% | 92.71% | 97.44% |
| 饼干盒 | 84.47% | 91.55% | 93.24% | 91.98% |
| 漂白剂瓶 | 97.95% | 99.13% | 97.87% | 98.87% |
| 咖啡罐 | 98.45% | 96.76% | 98.11% | 96.07% |
| 五物体等权均值 | 87.65% | 94.84% | 92.85% | 95.88% |

## 核验发布结果

原三组归档固定于 `active-3-paired`，新增十轮和五组汇总固定于 `active-3-controls`。在仓库根目录使用以下命令核对两份原始轨迹归档及 25 行明细。[扩展结果与校验值](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/active-3-controls/active-3-extended) · [原三组轨迹、重建网格与回放](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/active-3-paired/active-2)

```shell
hf download FerryZh/full-hand-active active-2/trajectories.zip --repo-type dataset --revision active-3-paired --local-dir downloads
hf download FerryZh/full-hand-active active-3-extended/trajectories.zip active-3-extended/trials.csv active-3-extended/results.json --repo-type dataset --revision active-3-controls --local-dir downloads
python -m Module.active_2.verify_extended --base downloads/active-2/trajectories.zip --new downloads/active-3-extended/trajectories.zip --report downloads/active-3-extended
```

## 重新运行

还原正式夹具、评价模型、冻结手指权重和用于逐项比对的 AUDE 原始记录。夹具清单指定手和物体的 MuJoCo 模型、初态、离线参考网格和种子。[实验环境](../../PARK/README.md) · [手指策略权重](../../finger_policy/README.md)

```powershell
hf download FerryZh/full-hand-active --repo-type dataset --revision PARK --include "park/inputs/*" "park/formal/*" "park/comparison/*" --local-dir downloads
python -m PARK.prepare --data downloads/park --comparisons
foreach ($object in 'pear','mustard','cracker','bleach','coffee') {
    foreach ($condition in 'planned','single_all','compliance','aude_mcc','aude_height') {
        python -m Module.active_2 --object $object --condition $condition --out "runs/active-3/$object/$condition"
    }
}
```

如需从已发布的轨迹重新计算正式扩展表，将两个 ZIP 分别解压到含 `runs/` 的目录，运行如下命令。输出写入 `runs/extended-active-3/report/`。

```powershell
Expand-Archive -LiteralPath downloads/active-2/trajectories.zip -DestinationPath runs/original-active-3/runs
Expand-Archive -LiteralPath downloads/active-3-extended/trajectories.zip -DestinationPath runs/extended-active-3/runs
python -m Module.active_2.report_extended --base runs/original-active-3 --root runs/extended-active-3 --recompute
```

CSV 包含逐轮覆盖率、重建质量、接触占比、控制证据和运行用时。用时来自各轮日志，部分任务并行执行，计时受共享计算资源影响。原三组保留梨的 4 倍速回放，每 15 个仿真秒更新一次 GPIS，以及五物体的最终重建对照。
