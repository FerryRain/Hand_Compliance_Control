# PARK

PARK 用主动触觉探索物体表面。全局规划依据表面不确定度和四指接触概率选择腕部目标，降低重复触达区域和长时间停留区域的评分。若规划轨迹的起点未通过间隙检查，控制器会从当前实测腕姿重新规划。局部调腕、手指策略和接触控制与配对实验共用原生执行器。[逐物体结果、重建与同步视频](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/active-methods#coverage)

## 正式结果

五个物体各使用一个配对种子。梨为 20261120，咖啡罐、饼干盒、芥末瓶和漂白剂瓶为 20261122。三种方法从相同初态独立运行 240 秒和 600 秒。表中是五个物体的等权平均。C25 表示距实触点 25 mm 测地邻域内的表面积占比，F@5 是 5 mm 容差下的重建 F 分数。

| 方法 | 240 秒 C25 | 240 秒 F@5 | 600 秒 C25 | 600 秒 F@5 |
| --- | ---: | ---: | ---: | ---: |
| PARK | 68.36% | 78.48% | 96.68% | 96.58% |
| PalmReach-6 | 53.68% | 69.30% | 84.68% | 88.86% |
| AUDE | 38.67% | 55.15% | 55.06% | 69.89% |

[逐物体评分](scores.csv) · [试验与源码校验清单](release.json) · [数据归档校验值](archives.json)

## 下载与核验

在 Python 3.10 或更高版本中，下载同名的 PARK 代码与数据。下面的命令还原正式试验、冻结模型和原生输入，并逐项核对 SHA-256。[GitHub Release](https://github.com/FerryRain/Hand_Compliance_Control/releases/tag/PARK) · [Hugging Face 数据](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/PARK/park)

```bash
git clone --branch PARK --depth 1 https://github.com/FerryRain/Hand_Compliance_Control.git
cd Hand_Compliance_Control
hf download FerryZh/full-hand-active --repo-type dataset --revision PARK --include "park/inputs/*" "park/formal/*" --local-dir park-download
python -m PARK.prepare --data park-download/park
python -m PARK verify
```

复现三方法比较时，再还原 PalmReach-6 和 AUDE 的原生记录及评分。

```bash
hf download FerryZh/full-hand-active --repo-type dataset --revision PARK --include "park/comparison/*" --local-dir park-download
python -m PARK.prepare --data park-download/park --comparisons
```

## 运行

规划、实测腕姿重规划、评分和回放都位于本目录。采集环境使用 Python 3.10、NumPy 1.26.4、SciPy 1.15.3、PyTorch 2.8.0+cu129、GPyTorch 和 MuJoCo 3.6.0。正式记录位于 `outputs/PARK/trials/`。用 `--out` 指定新目录即可独立采集。[手指执行环境](../finger_policy/README.md)

```bash
python -m PARK trial --object pear --seconds 600 --out runs/pear
python -m PARK score --object pear --seconds 600 --out runs/pear
```

在原 Windows 实验工作区，可用 `./PARK/run.ps1 verify` 调用已配置的环境。正式数据保留 100 Hz 原生轨迹和采集时的源码摘要；发布清单另外核对当前实现。每个物体均提供三方法同步回放、覆盖率曲线和重建网格。[实验素材](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/PARK/park/atlas)
