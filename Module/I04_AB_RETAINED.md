# I04 A/B 保留版

2026-11-19 按用户要求清理旧 I04 实验。当前入口是 `scripts/run_i04_ab.ps1`，运行的是已知物体几何下的整手规划与 MCC 控制。Active/GPIS 探索不在此保留版中。

## 保留结果

| 方法 | 数据目录 | 实际结论 |
| --- | --- | --- |
| A | `generated/i04a_frozen_b01_20261021` | 当前仍有完整原生轨迹的最佳 A。覆盖 14.6667732%，2025.236 秒，1,012,618 步、103 个分块；未完成整物体遍历。 |
| B | `generated/i04b_palm_recovery02_20261027` | 掌部碰撞修复后的完成结果。覆盖 100%，6918.226 秒，3,459,113 步、348 个分块，8192/8192 顶点、388/388 目标。 |

两案都保留原始轨迹、初末态、模型和碰撞网格、源码快照、独立审核、完整视频和视频来源记录。B 内部的 `parent_provenance` 同时保留原接续来源。

**结论边界**

- B 的 100% 是原协议的 **25 mm 测地邻域覆盖**，不表示每平方毫米都被指腹直接扫过，也不是 GPIS 未知物体探索成功。
- B 在 5774.600 秒有已记录的控制版本交接，全程状态保存在同一案例内。它不是从零执行同一控制版本的新实验。
- A 保留原模型，当时刚性掌部碰撞未启用。A 与掌部修复后的 B 不能作为同物理条件的严格配对消融。
- 更早的 A 曾按旧接触标准报告 100%；那些原生轨迹在 2026-10-28 已被删除。此次不把旧报告升级为可回放的成功数据，也未从残留报告伪造轨迹。
- 本轮没有重新运行数小时的完整遍历。完整运行结论来自保留的原始独立审核；本轮执行文件完整性、短启动、采样回放和短原生重放检查。

## 下载与安装

代码发布到主分支 `main`。视频和历史实验数据存放在 Google Drive，下载入口、完整文件清单和解压说明见 [`data/i04/README.md`](../data/i04/README.md)。GitHub 保留运行必需的模型、初态、几何种子和代码；不需要下载历史轨迹即可启动新实验。使用 Git LFS 获取小型二进制种子，不能用源码 ZIP 中的 LFS 指针替代实际文件。

```powershell
git clone --single-branch --branch main https://github.com/FerryRain/Hand_Compliance_Control.git
cd Hand_Compliance_Control
git lfs install --local
git lfs pull

# 独立 Python 3.10 环境；本实验不需要训练 DP，也不要求 CUDA。
conda create -n i04-ab python=3.10 -y
conda activate i04-ab
python -m pip install -r Module/requirements-i04-whole-hand.txt

# 新电脑显式使用当前环境的 python；此检查不需要历史数据。
.\scripts\run_i04_ab.ps1 -Python python verify --code-only

# 从 Google Drive 下载并解压历史数据到仓库后，再回放。
.\scripts\run_i04_ab.ps1 -Python python replay --method B --speed 12
```

交互回放需要可用的图形桌面/OpenGL。也可以直接在 [Google Drive 视频目录](https://drive.google.com/drive/folders/1TknGR1qpzUxk6Gex-x_Y3YoJJWKstOZA) 观看完整视频。回放读取已保存的真实状态，不启动新探索。

## 运行

在仓库根目录运行，默认 Python 为 `D:\Environment\Anaconda\envs\handcomp\python.exe`。脚本设置本地依赖路径和数值库线程数。不同环境可用 `-Python` 指定 Python 3.10。

```powershell
# 回放已保存的完整 B，12 倍速。B-safe 旧别名仍可用于直接回放器。
.\scripts\run_i04_ab.ps1 replay --method B --speed 12

# 回放最佳 A
.\scripts\run_i04_ab.ps1 replay --method A --speed 12

# 新建 B 实验。输出目录必须不存在，避免覆盖保留结果。
.\scripts\run_i04_ab.ps1 run --method B --output Module/generated/my_b_run --seconds 9000

# 新建 A 实验；该方法已有结果未达到全覆盖。
.\scripts\run_i04_ab.ps1 run --method A --output Module/generated/my_a_run --seconds 9000

# 仅校验代码和运行输入，不需要 Drive 数据
.\scripts\run_i04_ab.ps1 verify --code-only

# 下载所有历史数据和视频后，校验完整保留集
.\scripts\run_i04_ab.ps1 verify

# 分别核对前 500 步原生状态、命令及几何守护
.\scripts\run_i04_ab.ps1 smoke --method A --states 500
.\scripts\run_i04_ab.ps1 smoke --method B --states 500
```

需要重新完整审核时，使用 `audit --method B --output-name new_full_audit` 或对应的 A。新审核目录必须不存在，原审核不会覆盖。该操作耗时较长，日常观看视频不需要运行它。

`run` 从声明初态执行新仿真，`replay` 显示保留轨迹，`smoke` 只核对短原生前缀，`verify` 只核对文件。`--seconds` 是仿真预算，不是完成时限承诺；新 B 从零运行不等同于历史 B 的跨版本接续轨迹，不能提前声称新运行必然完成。运行后先检查新目录的 `summary.json`、停止原因和覆盖，再独立审核新结果。

Linux 可在安装同一依赖后直接使用 Python 入口（回放仍需图形环境）：

```bash
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
python -m Module.i04_geometric_traversal.retained_ab verify --code-only
python -m Module.i04_geometric_traversal.retained_ab replay --method B --speed 12
python -m Module.i04_geometric_traversal.retained_ab run --method B --output Module/generated/my_b_run --seconds 9000
```

Windows/Python 3.10 是本次实际验证环境；Linux 命令使用同一可迁移入口，未另做 Linux 全程复现实验。

## 必须保留的依赖

- `i04_geometric_traversal/` 中的运行、整手规划、MCC、几何/力守护、审核和回放依赖。A 和 B 使用各自原来的力停与接触标准；此次不调参。
- `i04_oracle_next_point/surface_graph.py` 及包入口，供已知表面图使用。
- `generated/whole_hand_geometry_touch02_20261018/geometry.npz`，原初始化几何种子。
- `generated/mustard_natural_fresh02_20261018/` 中经过哈希检查的紧凑 IK 库及来源审核。
- `generated/whole_hand_surface_run10_20261019/` 中紧凑协同 IK 库、共享模型网格和来源元数据。DP 项目也读取其中部分来源文件。这不是另外保留的一份可回放实验。
- `generated/i04b_full_force_grace01_20261020/initial.npz` 与 `manifest.json`，A 的冻结初态和源码基准。这不是完整旧 B 数据。
- `evidence/whole_hand_release_assets_20261019.json` 指向的机器人与物体资产，以及公共模块、环境依赖。

入口自动安装已有的紧凑种子库适配器，不读取已经移除的旧原生轨迹，不继承覆盖。B 的 `palm_safe_mcc.py` 已恢复成成功案例源码快照的精确字节。A/B 原运行源码与对应快照的哈希应全部一致。

旧发布清单仅用于原协议和来源核对，可能列出已按要求删除的旧实验。当前完整文件清单以 `evidence/2026-11-19_I04_AB_RETAINED_FILES.json` 为准（含 Drive 历史数据）；代码与运行输入清单为 `evidence/2026-11-19_I04_AB_CODE_FILES.json`。

删除范围、检查日志及实际释放空间见 `evidence/2026-11-19_I04_AB_CLEANUP.json`。DP、E05、机器人资产及其他非 I04 工作保留。Git 历史未清除，也未执行 LFS 垃圾回收。
