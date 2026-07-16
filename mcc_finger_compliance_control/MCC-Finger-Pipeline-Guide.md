# 无 FSR 指尖 MCC 数据采集与 Diffusion Policy 流程

本文档对应任务 `Leaphand-Finger-MCC-Position-Control`。这套代码与旧的
`finger_compliance_control`、`palm_compliance_control` 和 combined 控制任务相互独立。

当前控制结构：

- 机械臂：固定世界坐标掌心位姿的 Cartesian 弹簧阻尼参考 + Mink IK。
- 手指：固定预抓握位姿的低刚度位置控制，依靠物理接触产生被动让步。
- 指尖触觉：记录四个完整 fingertip geom 的精确 3D 接触力。
- 力反馈：未启用。3D 力只用于记录、质量筛选和未来 DP 输入。

## 1. 相关文件

环境与控制器：

```text
src/mjlab/tasks/leaphand/leaphand_mcc_finger_env_cfg.py
src/mjlab/tasks/leaphand/__init__.py
```

无 FSR 模型：

```text
src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_tactile.xml
src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml
```

流水线：

```text
mcc_finger_compliance_control/scripts/run_test.py
mcc_finger_compliance_control/scripts/collect_trajectories.py
mcc_finger_compliance_control/scripts/filter_trajectories.py
mcc_finger_compliance_control/scripts/invert_trajectories.py
mcc_finger_compliance_control/scripts/replay_inverted.py
mcc_finger_compliance_control/scripts/dp_dataset.py
mcc_finger_compliance_control/scripts/train_dp.py
```

## 2. 配置 Conda 环境

### 2.1 先理解这套环境需要什么

这套流水线包含两种计算负载：

- MuJoCo-Warp 仿真：NVIDIA GPU 最快，也能使用 CPU，只是并行采集会明显变慢。
- Diffusion Policy 训练：NVIDIA CUDA 最方便；CPU 能完成小规模测试，但正式训练很慢。

本文命令按 Linux/bash 编写，并在 Ubuntu 上验证。Windows 用户建议使用 WSL2；如果直接使用
PowerShell，需要把命令中的反斜杠换行语法改成 PowerShell 对应写法。

这里有三个容易混淆的“CUDA 版本”：

1. `nvidia-smi` 右上角的 CUDA Version：当前显卡驱动能够支持的最高 CUDA 版本。
2. `torch.version.cuda`：安装的 PyTorch wheel 自带的 CUDA 运行时版本。
3. `nvcc --version`：系统单独安装的 CUDA Toolkit 版本。

本项目通常不要求单独安装 `nvcc`，也不要求系统 Toolkit 与 PyTorch wheel 完全同版。不要因为
`nvcc` 不存在就判断 CUDA 不可用；应以 `torch.cuda.is_available()` 和实际短仿真为准。

### 2.2 判断应该选择哪条安装路线

先进入仓库。下面是示例路径，其他用户应替换成自己的仓库位置：

```bash
cd ~/Code/Hand_Compliance_Control
```

检查 NVIDIA 显卡和驱动：

```bash
nvidia-smi
```

- 命令成功：优先使用 2.4 节 NVIDIA 路线。
- 命令不存在、没有 NVIDIA 显卡或驱动不可用：使用 2.5 节 CPU 路线。
- 其他非 NVIDIA 显卡：统一按 CPU 路线运行本项目。
- Apple Silicon/macOS：按 CPU 路线，并在运行命令中使用 `--device cpu`。

快速选择表：

| 机器 | 仿真参数 | DP 参数 | 初次测试环境数 |
|---|---|---|---:|
| NVIDIA + 可用驱动 | `--device cuda:0` | `--device cuda:0` | 1，确认后再增大 |
| 纯 CPU/Apple Silicon | `--device cpu` | `--device cpu` | 1 |

`cuda:0` 表示第一张 NVIDIA GPU；多卡机器可用 `cuda:1` 选择第二张卡。这里的 `--device`
决定仿真或训练张量放在哪个计算设备上，与是否打开可视化窗口是两件事。

### 2.3 创建公共的 Conda 基础环境

如果还没有 Conda，需要先安装 Miniconda 或 Miniforge。之后在仓库根目录执行：

```bash
conda create -n mjlab python=3.10 -y
conda activate mjlab
python -m pip install --upgrade pip
```

确认当前终端确实使用刚创建的环境：

```bash
which python
python --version
```

`which python` 应指向 Conda 的 `envs/mjlab` 目录，Python 应为 3.10.x。如果仍指向系统 Python，
说明 `conda activate mjlab` 没有生效。

以后每次打开新终端，都要重新进入仓库并激活环境：

```bash
cd ~/Code/Hand_Compliance_Control
conda activate mjlab
```

下面的 GPU 和 CPU PyTorch 安装方式只能二选一，不要在同一环境中先后安装两套。

### 2.4 NVIDIA GPU：当前已验证的 CUDA 12.8 路线

若 `nvidia-smi` 显示驱动支持 CUDA 12.8 或更高，可使用开发机已验证的组合：

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install mink==1.1.1 h5py==3.16.0 scipy==1.15.3
```

其中 `pip install -e .` 会以 editable 模式安装当前仓库：修改 `src/` 下的代码后不需要反复重装。
最后一行补充本任务直接使用、但项目基础依赖中没有固定版本的 Mink、H5 和 SciPy 工具。

若驱动较旧，不要强装 `cu128`。打开 PyTorch 官方安装选择器
<https://pytorch.org/get-started/locally/>，选择自己的操作系统、Pip、Python 和驱动支持的 CUDA
版本。先执行官网生成的 PyTorch 命令，再安装本项目：

```bash
# 这只是格式示例；cuXXX 必须替换成官网为当前机器生成的版本，不能原样复制。
python -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX

python -m pip install -e .
python -m pip install mink==1.1.1 h5py==3.16.0 scipy==1.15.3
```

如果官网没有与旧驱动匹配的组合，建议更新 NVIDIA 驱动；暂时无法更新时，可以先使用 CPU
路线验证整个项目。

### 2.5 无 NVIDIA GPU、无 CUDA 或只想先验证流程

CPU 路线不要求 CUDA、`nvidia-smi` 或 `nvcc`。Linux/Windows CPU 使用：

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip install mink==1.1.1 h5py==3.16.0 scipy==1.15.3
```

macOS/Apple Silicon 的 PyTorch 从默认 PyPI 安装，不使用 Linux CPU wheel 索引：

```bash
python -m pip install torch==2.10.0
python -m pip install -e .
python -m pip install mink==1.1.1 h5py==3.16.0 scipy==1.15.3
```

运行后续命令时统一使用 `--device cpu`。CPU 用户建议先设 `--num-envs 1`，不要直接照抄
32 环境、1024 条轨迹的正式采集命令。CPU 可以跑通完整流程，但采集和训练都会慢很多。

### 2.6 检查安装是否成功

运行下面的多行检查命令：

```bash
python - <<'PY'
import h5py
import mink
import mujoco
import torch
import warp

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("torch GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("mujoco:", mujoco.__version__)
print("warp:", warp.__version__)
print("h5py:", h5py.__version__)
print("mink import: OK")
PY
```

NVIDIA 路线预期 `torch GPU available: True`；CPU 路线显示 `False` 是正常的。开发机验证过：

```text
Python 3.10.20
PyTorch 2.10.0+cu128
MuJoCo 3.9.0
Warp 1.12.0
Mink 1.1.1
h5py 3.16.0
```

版本不必逐项完全相同。至少应满足 `pyproject.toml` 中的 Python 3.10–3.13、PyTorch 2.7 或
更高、MuJoCo 3.6 或更高以及 Warp 1.12.0。出现依赖冲突时，优先新建干净 Conda 环境，不要在
已经装有另一套 Torch/CUDA 的环境中反复覆盖。

最后检查当前任务 cfg：

```bash
python -c "from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import mcc_finger_contact_env_cfg; print(mcc_finger_contact_env_cfg(1).scene.num_envs)"
```

输出 `1` 表示 cfg 可以正常加载。接下来再运行第 3 节的可视化测试。

### 2.7 常见环境问题

- `torch.cuda.is_available() == False`：常见原因是装成 CPU Torch、驱动不可用或驱动太旧。
- `Warp CUDA error 100`：Warp 没找到可用的 NVIDIA 驱动；先改成 `--device cpu` 验证。
- `CUDA out of memory`：采集时减小 `--num-envs`，训练时减小 `--batch-size`。
- `No module named mink/h5py`：确认已激活 `mjlab`，再执行对应路线最后一条依赖安装命令。
- `GLXBadContext`：这是窗口/图形驱动问题，不代表无头仿真失败；按第 3 节使用 Viser。
- H5 `Resource temporarily unavailable`：已有进程占用同名文件；退出旧进程并更换 `--filename`。
- `Permission denied` 或缓存目录只读：可在命令前加
  `MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp`，将临时编译缓存写到 `/tmp`。

## 3. 可视化运行 test

Native MuJoCo 窗口：

```bash
cd ~/Code/Hand_Compliance_Control
conda activate mjlab

python mcc_finger_compliance_control/scripts/run_test.py \
  --viewer native \
  --device cuda:0 \
  --print-every 100
```

CPU、Apple Silicon 或没有可用 NVIDIA 驱动时，直接运行：

```bash
python mcc_finger_compliance_control/scripts/run_test.py \
  --viewer native \
  --device cpu \
  --print-every 100
```

若本机出现 `GLXBadContext` 或 `Failed to create context`，使用 Viser：

```bash
python mcc_finger_compliance_control/scripts/run_test.py \
  --viewer viser \
  --device cuda:0 \
  --print-every 100
```

CPU 用户使用 Viser 时也应把该命令改为 `--device cpu`。

终端会打印四指 3D 力、指尖 IK 误差和掌心跟踪误差。关闭 native 窗口时，优先先在终端按
`Ctrl+C`，可减少部分显卡驱动上的 GLX 销毁错误。

## 4. 无头并行数据采集

采集脚本默认保存所有原始轨迹，不在线淘汰。`--num-envs` 控制 CUDA 并行环境数，进度条直接显示在当前终端。

下面是 NVIDIA GPU 的正式采集示例。新人应先把 `--num-envs` 和 `--max-trajectories` 都改小，
确认能运行并生成 H5 后，再启动长时间采集。不同显卡显存不同：32 环境不是强制值，显存不足时
依次尝试 16、8、4 或 1。

```bash
RUN="tactile_raw_random_$(date +%Y%m%d_%H%M%S)"

python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --device cuda:0 \
  --num-envs 32 \
  --trajectory-length 2500 \
  --max-trajectories 1024 \
  --motion-start 350 \
  --record-start-step 350 \
  --motion-length 1800 \
  --angular-speed-min 0.03 \
  --angular-speed-max 0.06 \
  --initial-orientation-jitter-deg 3 \
  --contact-threshold 0.05 \
  --seed 20260715 \
  --filename "${RUN}"
```

CPU/非 NVIDIA 用户先用下面的短任务验证，不要一开始就在 CPU 上采集 1024 条：

```bash
RUN="cpu_smoke_$(date +%Y%m%d_%H%M%S)"

python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --device cpu \
  --num-envs 1 \
  --trajectory-length 500 \
  --max-trajectories 2 \
  --motion-start 150 \
  --record-start-step 150 \
  --motion-length 300 \
  --angular-speed-min 0.03 \
  --angular-speed-max 0.06 \
  --initial-orientation-jitter-deg 3 \
  --seed 42 \
  --filename "${RUN}"
```

这条 CPU 命令只验证环境能推进并写出 H5，不代表生成的数据已经满足四指持续接触质量标准。

说明：

- 前 350 步包括机械臂 preparation 和手指接触建立，不写入训练数据。
- 每条记录包含 2150 帧，其中前 1800 帧物体旋转，后 350 帧停止并稳定。
- 每条轨迹随机初始姿态扰动、旋转轴和角速度。
- `max-trajectories` 最好是 `num-envs` 的整数倍。
- 不要添加 `--online-quality-gate`，否则会退回单环境在线严格淘汰模式。
- 每次使用新的 `RUN`，避免同名 H5 文件锁冲突。
- `RUN` 是当前终端中的临时变量；关闭终端后需要重新设置，或者直接把后续命令中的 `${RUN}`
  换成实际文件名。

原始文件位置：

```text
mcc_finger_compliance_control/data/trajectories/${RUN}.h5
```

### 4.1 离线严格质量筛选

```bash
python mcc_finger_compliance_control/scripts/filter_trajectories.py \
  "mcc_finger_compliance_control/data/trajectories/${RUN}.h5" \
  --contact-threshold 0.05
```

严格条件为：记录区间内每一帧四个 fingertip geom 都存在目标物体碰撞，并且每个指尖
`|F_3D| >= 0.05 N`。

输出：

```text
.../${RUN}_strict4tip.h5
.../${RUN}_strict4tip_report.csv
```

CSV 会给出每条轨迹的四指联合接触率、各指接触率、最小力和首次失联步。

如果用于 `stride=5` 的 DP 训练，不希望因零散的 1–4 帧掉接触删除整条轨迹，可以使用 relaxed
筛选：

```bash
RELAXED="mcc_finger_compliance_control/data/trajectories/${RUN}_relaxed99.h5"

python mcc_finger_compliance_control/scripts/filter_trajectories.py \
  "${RAW}" \
  --output "${RELAXED}" \
  --contact-threshold 0.05 \
  --min-all4-ratio 0.99 \
  --min-per-tip-ratio 0.99 \
  --max-loss-run 5
```

它要求四指同时接触率和每指接触率均至少 99%，同时把单次连续掉接触限制在 5 个仿真帧以内。
正式训练优先使用该数据；100% 严格筛选更适合作为最高质量评测子集。

## 5. 数据反演

反演将“世界中物体运动、手固定”变成“物体坐标系固定、手相对物体运动”。先设置路径：

```bash
FILTERED="mcc_finger_compliance_control/data/trajectories/${RUN}_strict4tip.h5"
INVERTED="mcc_finger_compliance_control/data/inverted/${RUN}_strict4tip_inverted.h5"
```

执行：

```bash
python mcc_finger_compliance_control/scripts/invert_trajectories.py \
  --file "${FILTERED}" \
  --output "${INVERTED}"
```

反演文件额外包含：

- `palm_pose_object`
- `fingertip_pose_object`
- `fingertip_force_object`
- `fingertip_contact_pos_object`
- `fingertip_contact_normal_object`
- `fingertip_curvature_object`
- `planned_palm_angular_velocity_object`
- `palm_twist_object`

`palm_twist_object` 是 palm 相对物体的 6D 速度，前三维为线速度，后三维为角速度。脚本按 episode
使用后向差分计算，只依赖当前帧和历史帧，不读取未来信息。DP 使用的
`fingertip_contact_normal_object` 来自接触传感器记录法向经过坐标变换后的结果；解析胶囊曲率等字段
仍保留用于离线分析，但不再进入当前最小 DP 输入。

## 6. Replay

### 6.1 可视化 teacher geometry replay

```bash
python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file "${INVERTED}" \
  --episode-id 0 \
  --viewer native \
  --mode teacher \
  --device cuda:0
```

若严格筛选后 episode 0 不存在，从质量报告中选择一个 `strict_pass=1` 的原始 episode ID。

### 6.2 无头定量 replay

```bash
python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file "${INVERTED}" \
  --episode-id 0 \
  --viewer headless \
  --mode teacher \
  --device cuda:0 \
  --max-steps 2150 \
  --contact-threshold 0.05
```

Teacher replay 直接写入反演后的 palm root pose 和 `q_hand`，用于验证坐标变换和碰撞几何，不代表最终闭环部署控制器。

Replay 也支持 CPU：把上述命令中的 `--device cuda:0` 改成 `--device cpu` 即可。几何 replay
只有一个环境，CPU 通常足够；第一次运行可能需要等待 Warp 编译 CPU kernel。

## 7. Diffusion Policy 训练

当前任务专用 DP 使用 LeRobot 0.4.4 的 `DiffusionPolicy` 和 conditional 1-D U-Net，不依赖旧 FSR
DP，也不再使用接触点位置、解析曲率或 palm 绝对相对位姿等特权输入。训练定义为：

```text
历史输入（默认 16 个采样点）：
  observation.state：
    q_hand                           16D
    fingertip_force_object          12D
    fingertip_contact_normal_object 12D

  observation.environment_state：
    palm_twist_object                6D

  合计                              46D

输出：
  未来 32 个采样点的 Δq_hand，单步 16D
```

3D 力为零时已经隐式表达无接触，因此最小骨架没有再重复加入 4D contact flag。`stride=5` 时相邻
采样点间隔为 0.05 s：16 点历史从首点到当前点覆盖 0.75 s，32 点预测覆盖未来 1.6 s。

### 7.1 小规模 overfit/流水线测试

```bash
python mcc_finger_compliance_control/scripts/train_dp.py \
  --file "${INVERTED}" \
  --device cuda:0 \
  --steps 3000 \
  --batch-size 128 \
  --stride 5 \
  --obs-horizon 16 \
  --pred-horizon 32 \
  --diffusion-steps 100 \
  --inference-steps 50 \
  --down-dims 128 256 512 \
  --val-ratio 0 \
  --num-workers 0 \
  --save-every 1000 \
  --eval-samples 64
```

无 CUDA 时可以先做一个更小的 CPU 冒烟测试：

```bash
python mcc_finger_compliance_control/scripts/train_dp.py \
  --file "${INVERTED}" \
  --device cpu \
  --steps 100 \
  --batch-size 16 \
  --stride 5 \
  --obs-horizon 4 \
  --pred-horizon 8 \
  --diffusion-steps 10 \
  --inference-steps 10 \
  --down-dims 32 64 128 \
  --num-workers 0 \
  --save-every 100 \
  --eval-samples 8
```

这个命令只用于确认数据读取、反向传播和模型保存都正常，不能用它判断最终模型效果。

### 7.2 正式训练

```bash
python mcc_finger_compliance_control/scripts/train_dp.py \
  --file "${INVERTED}" \
  --device cuda:0 \
  --steps 100000 \
  --batch-size 256 \
  --lr 3e-4 \
  --stride 5 \
  --obs-horizon 16 \
  --pred-horizon 32 \
  --diffusion-steps 100 \
  --inference-steps 100 \
  --down-dims 256 512 1024 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --eval-every 1000 \
  --save-every 10000 \
  --eval-samples 64
```

模型默认保存到：

```text
mcc_finger_compliance_control/data/models/dp_unet_<timestamp>/
```

包含：

- `latest.pt`、`best.pt` 和阶段 checkpoint；
- `dataset_info.json`；
- `metrics.csv`：方便用 pandas、Excel 或其他脚本继续分析；
- `metrics.json`：完整机器可读指标；
- `training_curves.png`：训练结束后的 loss 和生成轨迹 MAE 曲线。

训练/验证按完整 episode 划分，不会把同一轨迹的窗口同时放入两边。`--eval-every` 控制中间
metrics 的记录间隔，`--save-every` 控制完整 checkpoint 的保存间隔，两者不必相同。`best.pt`
按照验证集生成轨迹 MAE 选择，而不是只看 diffusion noise loss。

## 8. 推荐执行顺序

```text
run_test 可视化确认
  -> collect_trajectories 原始并行采集
  -> filter_trajectories 离线筛选
  -> invert_trajectories 坐标反演 + 解析表面特征
  -> replay_inverted teacher 几何验证
  -> train_dp 小规模 overfit
  -> train_dp 正式训练
```

在 teacher replay 未正确复现前，不应把 DP 训练问题归因于模型；应先修复坐标、episode 选择或碰撞几何。
