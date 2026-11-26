# Active-3 · AUDE 腕部限距与完整手指执行

主实验在五个 YCB 物体上比较五种控制条件，共 25 轮。另有五轮保留原版 AUDE 指尖力矩、只限腕部距离的辅助对照。每个物体各组使用相同 `start-02` 初态和一个配对种子，各自闭环探索 240 秒。梨的种子为 20261120，其余为 20261122。原版 AUDE 沿用 Active Exploration 的正式轨迹。[逐物体结果、网格与回放](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-3) · [Active Exploration 基线](../../PARK/scores.csv)

| 主实验五物体等权均值 | 25 mm 实触覆盖率 | 重建 F@5 mm | ≥2 指有效接触时间 | ≥3 指有效接触时间 |
| --- | ---: | ---: | ---: | ---: |
| 原版 AUDE | 38.67% | 55.15% | 42.49% | 24.45% |
| single_all + MCC | 14.43% | 28.05% | 18.32% | 6.96% |
| single_all + MCC（掌轴限距） | 27.65% | 44.80% | 14.66% | 8.13% |
| single_all（AUDE 指尖力矩） | 33.10% | 47.52% | 39.38% | 17.29% |
| 仅底层柔顺 | 22.70% | 33.77% | 23.92% | 8.33% |

同一套完整手指执行器加入腕部限距后，平均 C25 高 13.23 个百分点，平均 F@5 高 16.75 个百分点。≥2 指接触时间占比低 3.66 个百分点，≥3 指占比高 1.17 个百分点。原版 AUDE 的平均 C25 为 38.67%，F@5 为 55.15%。完整手指限距组的两项均值分别为 27.65% 和 44.80%。梨、芥末瓶、饼干盒、漂白剂瓶的 C25 和 F@5 都有提高。咖啡罐的 C25 持平，F@5 由 20.3374% 变为 20.3359%。[五物体完整手指限距前后同步探索视频](https://huggingface.co/datasets/FerryZh/full-hand-active/resolve/active-3-fingerall/active-3-fingerall/exploration.mp4)

## 控制条件

所有组每 0.9 秒根据本组实测触觉进行 AUDE 腕指联合规划，伺服频率为 500 Hz。两组完整手指执行使用相同的冻结 `single_all`、原生缓冲调度、MCC 力反馈和关节位置伺服。腕部限距根据候选腕姿的掌心朝向（掌坐标负 Z 轴），约束到最近一次合格触点的投影距离，多指取中位数，保留 AUDE 规划的腕部朝向。

| 运行参数 | 执行方式 |
| --- | --- |
| `planned` | 执行原版 AUDE 的腕指解和指尖力矩。 |
| `aude_mcc` | 执行 AUDE 腕姿和完整 single_all + MCC。 |
| `aude_mcc_height` | 与 `aude_mcc` 使用同一完整手指执行器，发布腕目标时加入掌轴限距。 |
| `single_all` | 冻结同步策略给出手指目标，由 AUDE 指尖力矩执行。芥末瓶关节指令每控制步限幅 0.015 rad，其余为 0.020 rad。 |
| `compliance` | 执行规划腕姿，关闭指尖位置跟踪，保留底层法向力、速度阻尼和弱零空间回正。 |
| `aude_height`（辅助）| 原版 AUDE 同时执行规划的腕指解和指尖力矩，仅对腕目标施加相同掌轴限距。 |

C25 按参考表面上距已注册实触点 25 mm 测地范围内的面积占比计算。F@5 是最终有符号 GPIS 与参考表面间 5 mm 容差下精确率、召回率的调和平均。有效指腹力至少 0.2 N，法向模长大于 0.99，且全手各区载荷小于 8 N。接触时间占比以初始状态之后的 24,000 个 100 Hz 采样帧为分母。

## 完整手指执行器的实测掌轴距离

训练中央 95% 区间为 100.76–146.80 mm，中位数为 128.53 mm。标定使用原芥末与扩充训练库，各占 50% 权重。全程带内比例采用实际腕姿与最近有效触点，新实触带内比例只统计当帧有合格实触的帧。下表把两组完整手指执行器逐物体配对。目标修正的次数与位移只统计限距组。[训练标定记录](height_calibration.json) · [标定复算脚本](calibrate_height.py)

| 物体 | 修正 / 规划次数 | 最大修正（mm） | 不限距（全程） | 限距（全程） | 不限距（新实触） | 限距（新实触） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 梨 | 119/267 | 25.47 | 67.42% | 83.67% | 97.92% | 78.42% |
| 芥末瓶 | 82/267 | 61.00 | 67.63% | 86.54% | 64.25% | 85.74% |
| 饼干盒 | 23/267 | 27.51 | 54.31% | 96.86% | 59.63% | 100.00% |
| 漂白剂瓶 | 73/267 | 26.04 | 84.21% | 95.61% | 100.00% | 97.25% |
| 咖啡罐 | 3/267 | 12.41 | 98.86% | 99.40% | 99.79% | 99.62% |
| 五物体等权均值 | — | — | 74.49% | 92.42% | 84.32% | 92.21% |

## 辅助对照 · 原版 AUDE 指尖力矩加限距

| 物体 | 25 mm 实触覆盖率 | 重建 F@5 mm | ≥2 指有效接触时间 | ≥3 指有效接触时间 |
| --- | ---: | ---: | ---: | ---: |
| 梨 | 50.26% | 68.45% | 45.58% | 23.75% |
| 芥末瓶 | 33.53% | 51.22% | 34.47% | 14.57% |
| 饼干盒 | 53.50% | 68.22% | 55.59% | 22.08% |
| 漂白剂瓶 | 28.05% | 43.12% | 48.90% | 24.27% |
| 咖啡罐 | 20.93% | 36.25% | 52.94% | 29.89% |
| 五物体等权均值 | 37.25% | 53.45% | 47.50% | 22.91% |

## 下载与核验

三份归档依次包含原三组的 15 轮、已发布扩展的 10 轮，以及完整手指限距的 5 轮。本次 JSON 和 CSV 包含全部 30 轮，分别标出五个主要条件和一个辅助条件。[本次发布](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/active-3-fingerall/active-3-fingerall) · [前十轮扩展](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/active-3-controls/active-3-extended) · [原三组轨迹与回放](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/active-3-paired/active-2)

```shell
hf download FerryZh/full-hand-active active-2/trajectories.zip --repo-type dataset --revision active-3-paired --local-dir downloads
hf download FerryZh/full-hand-active active-3-extended/trajectories.zip --repo-type dataset --revision active-3-controls --local-dir downloads
hf download FerryZh/full-hand-active active-3-fingerall/trajectories.zip active-3-fingerall/trials.csv active-3-fingerall/results.json --repo-type dataset --revision active-3-fingerall --local-dir downloads
python -m Module.active_2.verify_fingerall --base downloads/active-2/trajectories.zip --extended downloads/active-3-extended/trajectories.zip --new downloads/active-3-fingerall/trajectories.zip --report downloads/active-3-fingerall
```

## 重新运行

在仓库根目录还原正式夹具、评价模型、手指权重与基准轨迹，即可按物体重新采集。以下命令运行完整手指执行器的直接配对。[实验环境](../../PARK/README.md) · [手指策略](../../finger_policy/README.md)

```powershell
hf download FerryZh/full-hand-active --repo-type dataset --revision PARK --include "park/inputs/*" "park/formal/*" "park/comparison/*" --local-dir downloads
python -m PARK.prepare --data downloads/park --comparisons
foreach ($object in 'pear','mustard','cracker','bleach','coffee') {
    foreach ($condition in 'aude_mcc','aude_mcc_height') {
        python -m Module.active_2 --object $object --condition $condition --out "runs/active-3/$object/$condition"
    }
}
```

若使用三份发布的轨迹重新评分，将它们分别解压到下列根目录下的 `runs/`。报告写入 `runs/fingerall-active-3/report/`。

```powershell
Expand-Archive -LiteralPath downloads/active-2/trajectories.zip -DestinationPath runs/original-active-3/runs
Expand-Archive -LiteralPath downloads/active-3-extended/trajectories.zip -DestinationPath runs/extended-active-3/runs
Expand-Archive -LiteralPath downloads/active-3-fingerall/trajectories.zip -DestinationPath runs/fingerall-active-3/runs
python -m Module.active_2.report_fingerall --base runs/original-active-3 --extended runs/extended-active-3 --root runs/fingerall-active-3 --recompute
```

以下命令从两组共十条轨迹生成五物体的同步回放。各物体使用固定相机，上下两行显示完整手指不限距和限距时的实测腕指姿态与 C25，以 4 倍速播放。

```powershell
python -m Module.active_2.fingerall_replay --control runs/extended-active-3/runs --new runs/fingerall-active-3/runs --out exploration.mp4
```
