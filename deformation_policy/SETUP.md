# 在新机器准备代码、环境和输入

以下命令从主库根目录运行。主库克隆命令在 [开发交接首页](README.md)。系统工具需要 Git、Git LFS、Conda。运行密集 ETac 还需要 NVIDIA GPU、驱动、可用的 CUDA Toolkit 和本机 C++ 构建工具。Windows 使用带 MSVC 和 `nvcc` 的开发终端，Linux 使用与 Toolkit 匹配的主机编译器。仅阅读、编写数据格式和检查命令行帮助可以先在 CPU 机器进行。

## 1. 取得主库运行输入和上游 ETac

```sh
git lfs install --local
git lfs pull
git clone https://github.com/Crxus/ETac.git deformation_policy/.deps/ETac
git -C deformation_policy/.deps/ETac switch --detach 6dc00c5d3dc49a6915076a2e1c63b9cff02179b0
git -C deformation_policy/.deps/ETac rev-parse HEAD
```

`git lfs pull` 用于主库当前提交中的 I04 小型启动种子。上游固定提交提供 `BlindGrasping/propagation-ckpt/linear.pth`、通用 ETac 类和定制几何库。主库中的 LEAP 接口从 `simulation_Etac` 导入，本地的补丁已随项目发布。将上游依赖检出到工作目录内即可，无需将文件复制到另一个项目。[主库 LEAP 接口说明](../simulation_Etac/README.md) · [上游 ETac 固定版本](https://github.com/Crxus/ETac/tree/6dc00c5d3dc49a6915076a2e1c63b9cff02179b0)

设置环境变量指向刚才克隆的上游**根目录**。在新的终端重新设置，或在本目录的环境脚本中维护。

**PowerShell**

```powershell
$env:ETAC_SOURCE_ROOT = (Resolve-Path 'deformation_policy/.deps/ETac').Path
```

**Bash**

```bash
export ETAC_SOURCE_ROOT="$PWD/deformation_policy/.deps/ETac"
```

## 2. 准备触觉仿真环境

原 Windows 触觉环境记录为 Python 3.10、PyTorch 2.8.0 / CUDA 12.8、MuJoCo 3.6.0、PyTorch3D 0.7.9、ETac 定制 Kaolin 0.18.0，原机编译使用 CUDA Toolkit 12.9。下面给出以这些版本为起点的源码安装顺序。新机的编译器和驱动需要实际检查。PyTorch3D v0.7.9 的官方兼容列表未列出 PyTorch 2.8，原项目使用了本地编译，仓库没有附相应的通用二进制包。[原环境记录](../simulation_Etac/WINDOWS.md) · [PyTorch 版本与 CUDA 轮子](https://pytorch.org/get-started/previous-versions/) · [PyTorch3D v0.7.9 构建说明](https://github.com/facebookresearch/pytorch3d/blob/v0.7.9/INSTALL.md)

已有项目的 `handcomp` 环境可先做后面的导入检查。首次安装可将独立环境建在本目录中。

```sh
conda create --prefix ./deformation_policy/.envs/handcomp python=3.10 -y
conda activate ./deformation_policy/.envs/handcomp
nvidia-smi
nvcc --version
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install setuptools==80.9.0 wheel ninja packaging cython numpy pybind11 iopath
python -m pip install mujoco==3.6.0 trimesh==4.8.3 scipy transforms3d plotly pillow glfw
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

上面的 CUDA 检查应为 `True`。PyTorch3D 与定制 Kaolin 含原生扩展，需针对当前解释器和 PyTorch 安装。若环境已有兼容版本且后面的导入与 CUDA 检查通过，可直接复用。全新环境从相应源码安装的命令如下。[上游 ETac 安装说明](https://github.com/Crxus/ETac/blob/6dc00c5d3dc49a6915076a2e1c63b9cff02179b0/README.md#installation)

```sh
git clone --branch v0.7.9 --depth 1 https://github.com/facebookresearch/pytorch3d.git deformation_policy/.deps/pytorch3d
python -m pip install --no-build-isolation ./deformation_policy/.deps/pytorch3d
python -m pip install --no-build-isolation ./deformation_policy/.deps/ETac/ETac/thirdparty/pytorch_kinematics
python -m pip install --no-build-isolation ./deformation_policy/.deps/ETac/ETac/thirdparty/kaolin
python -m pip install --no-build-isolation --no-deps -e ./deformation_policy/.deps/ETac
```

**只检查导入、资源和接口**可以执行以下命令。其中 `compute_sdf` 必须来自 ETac 定制的 Kaolin。第一次实现点表前，建议先区分“原生扩展无法载入”和“点表或物体几何错误”。

```sh
python -B -c "from pathlib import Path; from simulation_Etac import blind_grasping_assets; from simulation_Etac.check_leap import LEAP_XML; from kaolin.metrics.trianglemesh import compute_sdf; from pytorch3d import _C; p = blind_grasping_assets() / 'propagation-ckpt' / 'linear.pth'; assert p.is_file() and LEAP_XML.is_file(); print(LEAP_XML, p)"
python -B finger_policy/pipeline.py --help
```

**需要运行基线时**，下面第一条会在 GPU 与 MuJoCo 上做方块接触检查，第二条打开图形查看器。它们使用原有密集节点。`check_windows` 用 ShadowHand 检查上游环境。LEAP 实验应使用 `check_leap`。

```sh
python -m simulation_Etac.check_leap
python -m simulation_Etac.view_leap
```

## 3. 读取 I04-B 或训练 DP 时再装对应依赖

**I04-B。** 从根目录用下面命令核对 Git 中的代码和启动输入，不需要下载历史轨迹。单独复跑旧 I04 时，可在独立环境安装它的 `requirements-i04-whole-hand.txt`，其中锁定了 NumPy、MuJoCo 等 CPU 仿真依赖。新方块教师可直接导入 `WholeHandStepSolver`，准备 NumPy 和 SciPy 即可使用求解器本身。[I04 依赖](../Module/requirements-i04-whole-hand.txt) · [保留版使用说明](../Module/I04_AB_RETAINED.md)

```sh
python -B -m Module.i04_geometric_traversal.retained_ab verify --code-only
```

**DP。** 新策略要显式安装训练依赖。仓库的 `requirements.txt` 声明 LeRobot 和 h5py，主库 `pyproject.toml` 管理 mjlab、MuJoCo 和 Warp。已跑通 ETac 的本地环境与原 DP 实验环境各自有依赖记录，仓库没有一份两者均已复现的跨平台锁文件。下面给出与上述 Torch 版本一致的候选安装和导入检查。先查看解析结果，以便保留已编译 CUDA 扩展要求的 Torch 版本。[DP 额外依赖](../requirements.txt) · [主库依赖](../pyproject.toml) · [LeRobot 0.4.4 包约束](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml)

```sh
python -m pip install --dry-run -e . -r requirements.txt "torch==2.8.0" "torchvision==0.23.0" "mujoco==3.6.0" "lerobot==0.4.4" "h5py==3.16.0"
python -m pip install -e . -r requirements.txt "torch==2.8.0" "torchvision==0.23.0" "mujoco==3.6.0" "lerobot==0.4.4" "h5py==3.16.0"
python -m pip check
python -B -c "import h5py, torch; from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy; print(torch.__version__, h5py.__version__, torch.cuda.is_available())"
```

训练可以在单独环境读取采集好的 HDF5。闭环时若采集与推理使用两个环境，需要按同一仿真帧传送观测和回收关节参考，记录延迟。若在同一进程中执行，安装 DP 后重跑上节的原生扩展导入。第一次成功时保存实际 `pip freeze`、系统、CUDA、主库和上游提交到该次 `runs/<run_id>/`。

**Windows 的旧 Python 入口。** 在所核对的提交中，`finger_policy/scripts/` 有 37 个 Git 符号链接，目标位于 `mcc_finger_compliance_control/scripts/`。若某个 `.py` 打开只有一行 `../../mcc_finger_compliance_control/...`，说明工作副本将链接取成了文本。优先在本目录的适配器中引用链接对应的实际源文件，需要完整旧流程时用支持符号链接的检出。旧脚本使用同目录的裸模块导入，子进程路径顺序见 `pipeline.py` 的 `runtime_environment()`。单独读到一个文件不代表所有裸导入都已可用。[旧流程入口与路径设置](../finger_policy/pipeline.py)

## 4. 一次干净克隆可以拿到哪些输入

| 需要的东西 | 来源与用途 |
| --- | --- |
| LEAP XML / 网格、三份核心触觉代码、检查与查看器 | 来自主库。方块可直接由示例构建。主库已有四个指尖的指腹/连接件拆分，新项目自己补齐其余合格掌侧面的掩码。[LEAP 读取器](../simulation_Etac/mujoco_leap.py) · [指尖分件](../mcc_finger_compliance_control/assets/true_fingertip_contact_v1/asset_manifest.json) |
| 传播权重、通用 ETac 类、定制几何库 | 来自步骤 1 的上游固定提交。`linear.pth` 是形变传播权重，和待训练的手指动作 checkpoint 用途不同。[ETac 上游权重](https://github.com/Crxus/ETac/tree/6dc00c5d3dc49a6915076a2e1c63b9cff02179b0/BlindGrasping/propagation-ckpt) |
| I04-B 规划、模型、初态与紧凑种子 | 主库和 Git LFS 提供。重放旧完整轨迹才需要其 Drive 历史包，第一条方块试次可以自己生成。[代码与启动输入](../Module/I04_AB_RETAINED.md) · [历史数据下载](../data/i04/README.md) |
| 原指尖 DP 的参考权重、Mustard 和 Expanded 数据 | 主库的下载指南指向 `FingerPolicy_Data`。`policy/` 提供三个历史权重，`data/training/mustard222/` 和 `data/training/expanded/` 提供相应的准备数据与原始记录。新实验可把资源放在 `deformation_policy/.deps/FingerPolicy_Data/`，由本目录配置引用。原数据用于核对来源及重建采集条件，新策略训练使用同步记录了掌侧形变的试次。[权重与训练数据下载](../finger_policy/README.md) |
| 原物体几何、闭环参考和采集配方 | 下载目录的 `assets_3d/` 存储物体几何，`data/evaluation/` 提供参考、请求、初态和运行依赖。复跑旧采集还需对应的任务协议和对象池配置。新方块流程使用本目录自己的配置、试次和 manifest。[原实验输入清单](../finger_policy/README.md) |
| 新触觉数据与策略 | 三档固定点表、整手形变试次、多指节教师、编码器、训练入口及新 checkpoint 都由新项目产出。现有方块演示、旧指尖权重及 I04-B 记录都不能替代这些交付物。[开发顺序和数据合同](README.md) |

开始正式训练前，按配置检查数据来源。`single_common` 与 `flat_common` 都只读取 Mustard 训练试次，`single_all` 同时读取 Mustard 和 Expanded。旧入口的 `--group` 会选择网络与采样方式，当前实现对两个 `common` 组也会加入 Expanded 窗口。形变训练入口需要按新配置显式选择数据集。[三个版本与采样要求](README.md#三个策略版本)
