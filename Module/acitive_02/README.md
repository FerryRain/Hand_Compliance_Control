# Active-02 · 梨的主动探索与重建

在 MuJoCo 中固定一颗留出的 YCB 016 梨。两条分支都使用腕部和手指 MCC。`aude-mcc` 执行 Aude 联合规划给出的腕指目标，`aude-single-all-mcc` 执行 Aude 的腕部目标与冻结 `single_all` 的手指目标。

- **Active-02-a** 从匹配的四指接触初态起步。规划查询由本轮实触更新的 GPIS，仿真碰撞和离线评分使用真值几何。手型约束保留长指屈曲与拇指对掌。
- **Active-02-b** 的局部规划使用梨真值碰撞几何。视频中的累计重建仍只使用各次运行的合格触点。

[实验站点](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-02) · [数据与视频](https://huggingface.co/datasets/FerryZh/full-hand-active)

## 运行

在仓库根目录配置可用的 Python 环境，包含 MuJoCo、NumPy、SciPy、PyTorch、GPyTorch、trimesh、scikit-image 和用于视频的 ImageIO 与 Pillow。`single_all` 分支还需要 CUDA 和 [Finger Policy 的发布权重与运行数据](../../finger_policy/README.md)。可用 `-Python` 为 PowerShell 入口传入 Python 路径。

B 的 `--base` 指向 `run` 生成的初始化目录，其中有 `model.mjb` 和 `frame-00000/initial.npz`。生成初态需要 `experiment.py` 中 `PEAR`、`PARTS` 和 `START_SOURCE` 指定的仿真与初态来源。A 的 `--reference` 指向 B 目录，读取同一物理模型与匹配初态。站点发布包提供结果和视频，复跑还需上述输入。

```powershell
.\scripts\run_acitive_02.ps1 run --output Module/generated/active-02-init --starts 0
.\scripts\run_acitive_02.ps1 full cooperative --base Module/generated/active-02-init --output Module/generated/active-02-b-repro --passes 3 --max-seconds 1300
.\scripts\run_acitive_02.ps1 active-a --reference Module/generated/active-02-b-repro --output Module/generated/active-02-a-repro --passes 3 --max-seconds 1300
```

在各运行目录上依次执行 `full score`、`full signed` 和 `full video`。用 `active-timing --a <A目录> --b <B目录> --output <计时目录>` 统计模块用时。用 `python -m Module.acitive_02.site_video --a <A目录> --b <B目录> --output <视频目录>` 生成站点的同步视频。
