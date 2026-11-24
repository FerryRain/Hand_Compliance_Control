# Active-2 · 手指执行方式对主动重建与多指贴合的影响

在五个 YCB 物体上，每个物体用相同初态和种子测试三种手指执行方式，每轮 240 个仿真秒。三组沿用 Active-1-B 的 LEAP 移植和 Aude 腕指联合规划，每 0.9 秒按各自的实测触觉更新腕部目标。规划器的手模型中有 0 个物体几何。重建只使用已经测到的合格指腹触点和法向。[实验页与逐物体合并表](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-2)

| 五物体等权均值 | 25 mm 实触覆盖率 | 重建 F@5 mm | ≥2 指有效接触时间 | ≥3 指有效接触时间 |
| --- | ---: | ---: | ---: | ---: |
| 执行 Aude 规划的手指目标 | 43.13% | 61.44% | 37.30% | 16.26% |
| 由冻结的 `single_all` 给出手指目标 | 42.27% | 59.29% | 48.07% | 26.12% |
| 仅执行底层指尖柔顺 | 24.35% | 39.63% | 22.57% | 9.32% |

`single_all` 替代规划器发给手指的目标后，平均覆盖率低 0.86 个百分点，重建 F 分数低 2.15 个百分点。每个物体只有一个配对种子，这里比较的是实测均值。三组都保留规划器内部的手指变量。

后两组均只执行规划腕姿。仅柔顺组保留 Aude 指尖法向力、速度阻尼和弱零空间回正，关闭指尖位置跟踪。与这一组相比，主动调指使至少两指和至少三指有效接触的平均时间占比分别提高 25.50 和 16.80 个百分点。四个物体的两项接触占比都提高。芥末瓶的两指占比从 47.40% 降至 8.76%，三指从 18.67% 降至 5.20%。

## 复现

使用已登记的五物体夹具、Active-1-B 的仿真环境与 `finger_policy/checkpoints/manifest.json` 中的 `single_all_seed20261116` 权重。夹具结构和依赖见 [Active-1-B](../active_1_b/README.md)，策略权重的获取方式见 [Finger Policy](../../finger_policy/README.md)。

```powershell
foreach ($object in 'pear','mustard','cracker','bleach','coffee') {
    foreach ($condition in 'planned','single_all','compliance') {
        python -m Module.active_2 --fixtures fixtures --object $object --condition $condition --out "runs/active-2/$object/$condition"
    }
}
```

每一轮保存逐次规划、100 Hz 物理轨迹、覆盖率、最终 GPIS 网格与评分。逐轮 CSV 另含联合规划、逐指 GP、IK、策略推理、在线总用时和最终重建评分用时。梨的三组视频同时显示探索过程和每 15 个仿真秒更新的 GPIS。[数据、合并表与视频](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/main/active-2)

下载同目录中的 `trajectories.zip` 后，可用 Python 和 NumPy 从归档核对轨迹指纹、配对初态与种子，并导出逐物体及等权平均的四项指标。输出比例介于 0 和 1 之间。

```shell
python -m Module.active_2.verify trajectories.zip
```
