# Active-1-B · Aude 方法在 LEAP 上的触觉探索基线

每个 YCB 物体从各自已知的四指轻载姿态出发，探索 240 个仿真秒。每物体保留一个初态和种子。在线逐指 GP 依据已经发生的触碰选择切向，联合规划器每 0.9 秒求腕部 6 个增量与 16 个手指关节增量。无预设扫描路线，规划手模型含 0 个物体几何。仿真使用物体几何产生接触，评分和展示可以使用真值表面。

执行器采用 Khadivar 等人无机械臂实现的腕部运动控制、指尖笛卡尔跟踪、法向力和零空间回正，运行于 500 Hz。物理指腹接触为规划提供反馈。LEAP 保留自身的运动学、关节约束和目标手型限制。本基线使用 22 变量规划和定时重规划。作者 Allegro 原系统使用 43 变量 SQP 和到位后重规划。各项结果描述本次 LEAP 移植。[作者发布版本](https://github.com/epfl-lasa/object_exploration/tree/d89a6941fec699768e64fb6acc07439ab822e59f)

| 物体 | 25 mm 实触覆盖率 | 精确率 @5 mm | 召回率 @5 mm | F@5 mm | 双向平均表面误差 |
|---|---:|---:|---:|---:|---:|
| 梨 | 63.53% | 99.34% | 66.56% | 79.71% | 6.26 mm |
| 芥末瓶 | 34.42% | 92.72% | 36.16% | 52.03% | 18.40 mm |
| 饼干盒 | 41.83% | 92.62% | 42.84% | 58.58% | 17.47 mm |
| 漂白剂瓶 | 44.18% | 96.46% | 45.56% | 61.89% | 16.24 mm |
| 咖啡罐 | 31.70% | 95.14% | 38.67% | 54.99% | 34.63 mm |
| **五物体等权平均** | **43.13%** | **95.25%** | **45.96%** | **61.44%** | **18.60 mm** |

25 mm 覆盖率按真值表面面积加权，计算沿表面到已登记指腹实触点距离不超过 25 mm 的比例。合格触点需要有效指腹法向、指腹力至少 0.2 N，且所有分区载荷均低于 8 N。覆盖统计再检查观测与真值的空间距离和法向一致性。重建使用合格实触点和法向拟合有符号 GPIS。F@5 mm 是两个采样表面在 5 mm 容差下精确率与召回率的调和平均。表中平均 F 是五个逐物体 F 的等权均值。双向表面误差以毫米计。

五次运行的平均在线墙钟为 111.37 秒，联合规划为 20.06 秒，其中逐指 GP 选向 5.12 秒、腕指 IK 14.87 秒。最终 GPIS 拟合与表面评分另计，平均 1.39 秒。计时来自并发仿真。本组不调用学习型手指策略。模拟力峰及实际手型偏离见逐物体诊断，目标手型约束不保证物理执行全程遵守。

## 复跑

使用与实验配对的夹具目录，内含 `fixtures.json`、`hand.mjb` 和每个物体的 `model.mjb`、`initial.npz`、`surface.obj`。指纹在数据集的 `fixture-hashes.json`。复跑需要本机持有的匹配夹具。Python 环境包含 MuJoCo 3.6.0、NumPy、SciPy、PyTorch、GPyTorch、trimesh、scikit-image 和 scikit-learn。视频还需 Pillow、imageio、imageio-ffmpeg 和中文字体。可用 CPU 或 GPU 进行离线 GPIS，归档的最终评分使用 CUDA。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
foreach ($object in 'pear','mustard','cracker','bleach','coffee') {
    python -m Module.active_1_b --fixtures fixtures --object $object --out "runs/$object"
}
python -m Module.active_1_b.report --fixtures fixtures --runs runs --out results
python -m Module.active_1_b.video --case fixtures/pear --run runs/pear --output results/pear.mp4 --title "梨 · Active-1-B" --speed 4
```

每个 `runs/<object>` 保存物理轨迹、逐次计划、运行指纹、100 Hz 覆盖率、最终 PLY 及评分。报告程序核对原始记录、夹具、时间轴和指标后，导出逐物体结果、逐秒覆盖和网格。视频每 15 个仿真秒用截至此刻的合格触点重建。最终网格直接使用已评分文件。

[站点](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-1-b) · [五物体数据和视频](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/main/active-1-b)。
