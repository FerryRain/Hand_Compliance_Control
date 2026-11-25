# PARK

PARK 是本仓库定稿的主动触觉探索方法。全局规划用 GPIS 不确定度和四指概率探针选择腕部目标，降低重复触达和持续停留区域的评分。当轨迹起点未通过间隙检查时，控制器从当前实测腕姿重新规划。局部调腕、`single_all` 手指策略与 MCC 沿用配对实验的执行器。[逐物体结果、重建与同步回放](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/active-methods#coverage)

## 正式结果

五个物体为咖啡罐、饼干盒、芥末瓶、漂白剂瓶和梨。每个物体一个固定种子，梨为 20261120，其余为 20261122。每种方法从同一初态分别运行 240 秒和 600 秒，五个物体等权平均。C25 是 25 mm 测地邻域的实触面积占比，F@5 是 5 mm 容差下的重建 F 分数。

| 方法 | 240 秒 C25 | 240 秒 F@5 | 600 秒 C25 | 600 秒 F@5 |
| --- | ---: | ---: | ---: | ---: |
| PARK | 68.36% | 78.48% | 96.68% | 96.58% |
| PalmReach-6 | 53.68% | 69.30% | 84.68% | 88.86% |
| AUDE | 38.67% | 55.15% | 55.06% | 69.89% |

[30 条逐轮评分](scores.csv) · [发布清单](release.json) · [归档校验值](archives.json) · [完整实验数据](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/main/park)

## 下载与核验

在 GitHub 的 `park-v3` 标签检出中，用 Python 3.10 或更高版本还原数据。下面的命令从 Hugging Face 下载正式试验和输入。数据也随同名 GitHub Release 提供，GitHub 压缩包的展开路径见 Release 说明。[GitHub Release](https://github.com/FerryRain/Hand_Compliance_Control/releases/tag/park-v3)

```bash
git clone --branch park-v3 --depth 1 https://github.com/FerryRain/Hand_Compliance_Control.git
cd Hand_Compliance_Control
hf download FerryZh/full-hand-active --repo-type dataset --revision park-v3 --include "park/inputs/*" "park/formal/*" --local-dir park-download
python -m PARK.prepare --data park-download/park
python -m PARK.run verify
```

`prepare` 校验归档 SHA-256 后按原始目录恢复文件。若已有同名文件，程序会核对内容。核验覆盖定稿规划器、模型、共享原生控制器和十组正式试验。复现对照分析时，再下载基线归档并加上 `--comparisons`。

```bash
hf download FerryZh/full-hand-active --repo-type dataset --revision park-v3 --include "park/comparison/*" --local-dir park-download
python -m PARK.prepare --data park-download/park --comparisons
```

## 运行与复现

在本项目的 CUDA Finger Policy 环境运行。采集环境使用 Python 3.10、NumPy 1.26.4、SciPy 1.15.3、PyTorch 2.8.0+cu129、GPyTorch 与 MuJoCo 3.6.0。新采集使用单独的输出目录。省略 `--out` 时，入口会复核并复用已还原的匹配试验。[环境说明](../finger_policy/README.md)

```bash
python -m PARK.run trial --object pear --seconds 600 --repeat 0 --out runs/park-new-trial
python -m PARK.run score --object pear --seconds 600 --repeat 0 --out runs/park-new-trial
```

在原 Windows 实验工作区也可以使用 `./PARK/run.ps1 verify`。原始采集签名记录为 `PalmReach-3`、`revision=v3`。签名包含文件相对路径，源码沿用 `PalmReach-3/`、`PARK-3/park3/` 和 `Module/park/` 中的共享组件。数据包含这次实际使用的规划模型、评估 GPIS、20 段训练采集、训练划分与探针拟合输入。原生执行器基于提交 `1cf4e3411b1f946e67a13afb6b52cdcf4b05b573`。

三个方法的回放、实际评分网格、腕姿轨迹以及覆盖和信息收益曲线均可按物体下载。原生轨迹保留 100 Hz 采样，站点腕姿 CSV 为 10 Hz。[归档数据与可视化](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/main/park/atlas)
