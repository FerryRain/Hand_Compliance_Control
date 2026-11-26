# Active-3 · 手指执行方式对主动重建与多指贴合的影响

在五个 YCB 物体上，每种手指执行方式从同一 `start-02` 初态和种子独立闭环探索 240 秒。梨的种子为 20261120，其余物体为 20261122。AUDE 行沿用 Active Exploration 正式比较中的五条 240 秒轨迹。主动调指与仅柔顺两组在相同夹具上运行，三组由同一套覆盖率和有符号 GPIS 重建程序评分。[逐物体比较与回放](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-3) · [Active Exploration 的 AUDE 基线](../../PARK/scores.csv)

| 五物体等权均值 | 25 mm 实触覆盖率 | 重建 F@5 mm | ≥2 指有效接触时间 | ≥3 指有效接触时间 |
| --- | ---: | ---: | ---: | ---: |
| AUDE 规划手指 | 38.67% | 55.15% | 42.49% | 24.45% |
| 冻结的 `single_all` 主动调指 | 33.10% | 47.52% | 39.38% | 17.29% |
| 仅底层指尖柔顺 | 22.70% | 33.77% | 23.92% | 8.33% |

与 AUDE 相比，single_all 的五物体平均覆盖率低 5.57 个百分点，平均 F@5 低 7.62 个百分点。按物体看，梨、饼干盒、咖啡罐的两项指标均高于 AUDE，芥末瓶、漂白剂瓶均较低。与仅柔顺组相比，single_all 的平均 ≥2 指有效接触时间占比高 15.46 个百分点，≥3 指占比高 8.96 个百分点。在梨、饼干盒、漂白剂瓶、咖啡罐上，两项接触占比均高于仅柔顺组。每个物体有一个配对种子，表中均值按五个物体的未舍入分数等权计算。

三组每 0.9 秒按本组实测触觉做一次腕指联合规划。AUDE 组同时执行规划的腕姿和指尖目标。`single_all` 组执行规划腕姿，由冻结的触觉扩散策略给出手指目标。芥末瓶每个控制步的关节指令限幅为 0.015 rad，其余物体为 0.020 rad，失触后的额外重试关闭。仅柔顺组执行规划腕姿，指尖位置跟踪项为零，保留法向力、速度阻尼和弱零空间回正。

C25 计算参考表面上距注册实触点 25 mm 测地范围内的面积占比。F@5 是最终重建网格与参考表面在 5 mm 距离阈值下的精确率和召回率的调和平均。接触占比统计初始帧之后的 24,000 个 100 Hz 采样帧。每帧计入的指腹力至少为 0.2 N、法向模长大于 0.99，且全手各区载荷均小于 8 N。

## 核验发布结果

从轨迹归档核对五组配对记录与发布的 JSON、CSV。命令逐轮输出 C25、F@5、≥2 指与 ≥3 指接触占比、5 mm 精确率和召回率，以及各组的五物体等权均值。六项比例的范围为 0 到 1。这个步骤只需要 Python 和 NumPy。

```shell
hf download FerryZh/full-hand-active active-2/trajectories.zip active-2/trials.csv active-2/results.json --repo-type dataset --revision active-3-paired --local-dir downloads
python -m Module.active_2.verify downloads/active-2/trajectories.zip --report downloads/active-2
```

## 重新运行

在本仓库根目录还原正式夹具、评价模型和用于逐项比对的 AUDE 原始记录。默认夹具含手模型、五个物体的 MuJoCo 物理模型、初态和离线参考网格。`fixtures.json` 指定每个物体的种子与输入指纹。手指策略使用 `single_all_seed20261116` 冻结权重。[实验环境](../../PARK/README.md) · [手指策略权重与安装](../../finger_policy/README.md)

```powershell
hf download FerryZh/full-hand-active --repo-type dataset --revision PARK --include "park/inputs/*" "park/formal/*" "park/comparison/*" --local-dir downloads
python -m PARK.prepare --data downloads/park --comparisons
foreach ($object in 'pear','mustard','cracker','bleach','coffee') {
    foreach ($condition in 'planned','single_all','compliance') {
        python -m Module.active_2 --object $object --condition $condition --out "runs/active-3/$object/$condition"
    }
}
```

要从发布的 15 条轨迹重新计算正式表格，可在同一环境中执行以下命令。表格输出到 `runs/published-active-3/report/`。

```powershell
Expand-Archive -LiteralPath downloads/active-2/trajectories.zip -DestinationPath runs/published-active-3/runs
python -m Module.active_2.report --root runs/published-active-3 --recompute
```

逐轮 CSV 包含覆盖率、重建精确率与召回率、接触指标、在线用时、离线终点评分用时和校验值。用时取自各轮日志，部分任务并行执行，计时受共享计算资源影响。梨的三组回放以 4 倍速同步显示探索与每 15 个仿真秒更新的 GPIS。[轨迹、合并表与回放](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/active-3-paired/active-2)
