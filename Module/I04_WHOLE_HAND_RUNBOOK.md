# Mustard 整手遍历：下载、查看和运行

发布版本：2026-10-19；分支 `main`。仅 MCC，DP/DPRef 未运行。
完整发布案例是 `Module/generated/whole_hand_surface_run10_20261019`，约3.06 GB（含运行所需小型支持数据）。
代码、报告、图片、全部原始2ms轨迹、三段正式视频及原始核验记录一起保存。

本页命令对应已发布的B开发版本。后续 [I04A/B新目标与共同条件](I04_AB_WHOLE_HAND_PROTOCOL.md)
已定义，公共运行代码迁移及新配对评测尚未完成；本页`run-fresh`不代表已经对齐的新A/B实验入口。

## 1. 下载代码和完整数据

安装 Git 和 [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后执行：

```powershell
git lfs install
git clone --branch main https://github.com/FerryRain/Hand_Compliance_Control.git
cd Hand_Compliance_Control
git lfs pull
```

已有克隆则先切换到本分支、`git pull --ff-only`，再执行 `git lfs pull`。
推荐使用克隆；GitHub 的源码 ZIP 是否包含大文件由仓库归档设置决定。
只出现约130字节的MP4/NPZ文件表示下载到了LFS指针，尚未下载实际数据。
LFS 的文件存储方式见 [GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

## 2. 直接查看报告和视频

无需安装仿真依赖，直接用浏览器打开：

```text
Module/generated/whole_hand_surface_run10_20261019/review.html
```

- `whole_hand_complete_24x.mp4`：完整实际时间轴，24倍速，219.95秒，保留离面、调姿和恢复。
- `whole_hand_details_1x.mp4`：六个方向正常速度节选，24.20秒，明确标注区域间时间跳转。
- `whole_hand_base_clear_1x.mp4`：瓶底正常速度特写，3.95秒；局部视角隐藏遮挡的机械臂和地面，左侧保留原始全景。
- `coverage_verified.png`、`contact_paths_verified.png`、`timeline_verified.png`：覆盖、实际指腹接触点和全程力/接触记录。

GitHub 网页展示 HTML 源码时，请在下载后打开本地 HTML；页面使用相对链接，不依赖原作者电脑路径。

## 3. 安装独立的 CPU 仿真环境

已验证环境为 Windows、Python 3.10、MuJoCo 3.6.0。无需 CUDA 或 DP checkpoint。
在仓库根目录执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r Module/requirements-i04-whole-hand.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'
```

后续命令中的 `python` 指此虚拟环境，可用 `.\.venv\Scripts\python.exe` 替代。
Linux/macOS 可用 Python 3.10 建立 `.venv` 并安装同一依赖；本次没有在这些系统上重新执行完整原生核验。
现成视频不受操作系统限制；重新渲染历史中文视频脚本使用 Windows 微软雅黑字体。

## 4. 检查下载是否完整，以及模型是否能运行

```powershell
python -m Module.i04_geometric_traversal.published_whole_hand verify
python -m Module.i04_geometric_traversal.published_whole_hand smoke --states 500
```

`verify`逐一核对报告、视频、轨迹、支持资产和冻结源码SHA-256，不重新计算物理。
`smoke`从唯一原始初态连续重放前500个原生状态（1秒），检查位置、速度、命令，并在独立进程中检查两类几何守护。
它验证启动、资产路径迁移和短段动力学，不代表重新核验了全部88分钟仿真。

发布适配入口会在内存中把历史 `model.xml` 的本机绝对路径映射到本次克隆里的同字节资产；
原始 XML、清单、控制器与核验报告保留原样。映射和每个资产的哈希见
[`evidence/whole_hand_release_assets_20261019.json`](evidence/whole_hand_release_assets_20261019.json)。

## 5. 对发布轨迹重新执行完整独立核验

```powershell
python -m Module.i04_geometric_traversal.published_whole_hand audit --workers 6 --output-name downloaded_native_reaudit
```

核验全部2,638,518个原生状态、实际指腹接触与覆盖、以及每条命令的实态/上一参考几何守护；从t=0连续重放，分块不重置。
结果写入案例下新目录 `downloaded_native_reaudit/`，不会覆盖已经发布的 `independent_native_audit/`。
再次运行请另取新输出名；这项工作耗时明显长于查看视频，取决于CPU和进程数。

## 6. 用此版规划和控制逻辑启动新实验

先运行30秒检查环境：

```powershell
python -m Module.i04_geometric_traversal.published_whole_hand run-fresh --output Module/generated/my_mustard_v8_smoke --seconds 30
```

需要更长遍历时使用新的目录：

```powershell
python -m Module.i04_geometric_traversal.published_whole_hand run-fresh --output Module/generated/my_mustard_v8_full --seconds 7000
python -m Module.i04_geometric_traversal.published_whole_hand audit --case Module/generated/my_mustard_v8_full --workers 6 --output-name fresh_native_audit
```

`--seconds`是物理仿真预算，不是墙钟时间。输出目录必须不存在；预算耗尽可能仍未覆盖完整物体，查看 `summary.json` 的实际结果。
此入口使用v8残余区域规划、四指MCC、自然屈指、原几何/力判据，从唯一新初态、零覆盖开始。
发布轨迹与旧单领触轨迹只提供几何IK姿态种子，不继承接触、物理状态或覆盖。
该新实验标记为 `v8-fresh-published`，不能当作已发布run10的逐步重新规划复现。
发布前已完成此入口的2秒启动检查，并对生成的1000个状态独立重放核验通过；
这只验证启动及核验流程，新实验尚未重新跑完整瓶遍历。

已发布run10是v4→v6→v7→v8的连续开发接续，控制器切换时刻为2260.000、4102.920、5273.000秒；
其原始命令、状态和递归版本来源全部随本次发布保存。第5节重放能核验这条完整历史轨迹。

## 7. 原始轨迹格式与测试

`summary.json` 的 `trace_chunks` 顺序列出全部266个分块、状态数、时间范围和SHA-256。
`initial.npz` / `final_state.npz`为原生初终态；`trace_0000.npz`至`trace_0265.npz`包含全部2ms观测，没有为了上传降采样。

```python
import json
from pathlib import Path
import numpy as np

case = Path('Module/generated/whole_hand_surface_run10_20261019')
summary = json.loads((case / 'summary.json').read_text(encoding='utf-8'))
for item in summary['trace_chunks']:
    with np.load(case / item['file']) as z:
        print(item['file'], z['time_s'][0], z['time_s'][-1], z['q_rad'].shape)
```

`q_rad`、`command_rad`、`sensor_command_rad`按7个臂关节后16个手关节排列；手指顺序食指、中指、无名指、拇指。
各长指4个坐标为侧摆、MCP、PIP、DIP；`qvel_rad_s`、`qacc_warmstart`使用原生MuJoCo DoF顺序，不能套用上述位置重排。
`safe_belly`是合格指腹标志，`belly_contact_positions_world_m`为实际世界坐标触点，`forces_n`为原始力。
`phase`记录接近、滑动、卸力、撤离、调姿与恢复；`coverage`采用25mm网格边测地邻域，不能解释为皮肤压力足迹。

```powershell
python -m unittest discover -s Module/tests -p 'test_i04_*.py'
```

发布前424项I04回归通过。完整实测覆盖为8192/8192顶点、388/388目标；
仍有最长146.156秒的全手有效接触中断，原始力峰10.000660N。该结果仅适用于固定Mustard、零重力仿真，尚不是无间断贴面滑动。
