# Active-02 · 逐指触觉探索与预设环扫

## 五物体无路标基线

逐指 GP 使用本轮指腹触点、法向和从手部运动得到的自由空间观测选择切向，生成虚拟触点。联合 IK 在不含物体的手模型上求解腕位与 16 个关节目标，腕部和手指均以 MCC 执行。手型约束保留长指屈曲与拇指对掌。选向周期 0.90 秒，虚拟步长 18 mm，物体固定于 MuJoCo 场景中。真值网格用于产生物理触觉和离线评分。

每个物体使用一个配对四指初态、一个种子和 240 仿真秒。腕部 Active 使用原对照中的 credible 版本。其五物体平均 10 mm 实触覆盖为 64.38%，逐指 GP＋腕指 MCC 为 8.44%，历史预设环扫为 64.92%。逐指基线的五物体平均轻载三指时间为 190.21 秒。梨上将名义手指指令交由冻结 single_all 后，覆盖从 16.40% 变为 7.50%。

本地使用 LEAP 和 SciPy 求解器，作者的发布实现使用 Allegro 和 MATLAB。结论针对本次配对的移植与执行系统。[论文](https://doi.org/10.1016/j.robot.2023.104461) · [作者主循环](https://github.com/epfl-lasa/object_exploration/blob/d89a6941fec699768e64fb6acc07439ab822e59f/src/run_problem.m#L448)

复跑主基线需要匹配的本地试验输入。夹具目录含 `fixtures.json`、`hand.mjb`，以及对象子目录中的 `model.mjb`、`initial.npz` 和供离线评分的 `surface.obj`。清单以 SHA-256 绑定规划手、物理模型、初态与种子。梨的策略分支另需本地冻结权重及初态文件中的 75 帧预热观测。本轮使用的 MuJoCo 版本为 3.6.0。

```powershell
.\scripts\run_acitive_02.ps1 route-free --fixtures path/to/fixtures --object coffee --seconds 240 --out Module/generated/route-free-coffee
.\scripts\run_acitive_02.ps1 route-free --fixtures path/to/fixtures --object pear --arm aude-single-all-mcc --out Module/generated/route-free-pear-single-all
```

输出包含 100 Hz 物理轨迹、逐计划的切向与虚拟目标、用时、覆盖曲线、最终 GPIS 网格和评分。报告中低于 8 N 的分区门限适用于五物体比较。[归档结果与视频](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/main/active-02/route-free) · [原采集版本与源码指纹](https://github.com/FerryRain/Hand_Compliance_Control/tree/a6df352a7417d6075351d42b51f98e548d7a796c/Module/acitive_02)。复用已有运行目录并按原源码指纹校验时，请使用这一采集版本。站点收录的[Active-1-B 五物体基线](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active-1-b)采用作者控制律和物理触碰反馈

## 历史梨环扫 A/B

两版都围绕夹具原点预生成三轴的逐圈路标，实触反馈用于局部调整。每版比较联合规划的手指与冻结 single_all 的手指，均采用腕指 MCC。

- **Active-02-a** 用本轮触点重建的局部 GPIS 查询候选间隙和法向，并限制手型。
- **Active-02-b** 在候选姿态上查询梨的真实碰撞几何。用于视频的逐时重建只使用本轮已发生的合格触点。

## 历史 A/B 的运行

在仓库根目录配置可用的 Python 环境，包含 MuJoCo、NumPy、SciPy、PyTorch、GPyTorch、trimesh、scikit-image 和用于视频的 ImageIO 与 Pillow。`single_all` 分支还需要 CUDA 和 [Finger Policy 的发布权重与运行数据](../../finger_policy/README.md)。可用 `-Python` 为 PowerShell 入口传入 Python 路径。

B 的 `--base` 指向 `run` 生成的初始化目录，其中有 `model.mjb` 和 `frame-00000/initial.npz`。生成初态需要 `experiment.py` 中 `PEAR`、`PARTS` 和 `START_SOURCE` 指定的仿真与初态来源。A 的 `--reference` 指向 B 目录，读取同一物理模型与匹配初态。站点发布包提供结果和视频，复跑还需上述输入。

```powershell
.\scripts\run_acitive_02.ps1 run --output Module/generated/active-02-init --starts 0
.\scripts\run_acitive_02.ps1 full cooperative --base Module/generated/active-02-init --output Module/generated/active-02-b-repro --passes 3 --max-seconds 1300
.\scripts\run_acitive_02.ps1 active-a --reference Module/generated/active-02-b-repro --output Module/generated/active-02-a-repro --passes 3 --max-seconds 1300
```

在各运行目录上依次执行 `full score`、`full signed` 和 `full video`。用 `active-timing --a <A目录> --b <B目录> --output <计时目录>` 统计模块用时。用 `python -m Module.acitive_02.site_video --a <A目录> --b <B目录> --output <视频目录>` 生成站点的同步视频。
