# MCC fingertip position-teacher pipeline

This directory is intentionally separate from the legacy FSR pipeline in
`finger_compliance_control/`.

完整中文配置、采集、反演、replay 与 DP 训练说明见：

[`MCC-Finger-Pipeline-Guide.md`](MCC-Finger-Pipeline-Guide.md)

面向多物体、接触流形 DP、全手触觉和主动探索的分阶段路线见：

[`Tactile-Exploration-DP-Roadmap.md`](Tactile-Exploration-DP-Roadmap.md)

## Visual check

```bash
python mcc_finger_compliance_control/scripts/run_test.py --viewer native
```

If the machine's GLX stack is unreliable, use `--viewer viser`.

使用与数据采集相同的随机姿态、物体旋转和控制参数观察 `0.06–0.12 rad/s` 测试：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/run_test.py \
  --viewer native \
  --device cuda:0 \
  --rotate-object \
  --motion-start 350 \
  --motion-length 1800 \
  --angular-speed-min 0.06 \
  --angular-speed-max 0.12 \
  --initial-orientation-mode uniform \
  --seed 20260716 \
  --print-every 50
```

该命令只用于观察，不写 H5。若 native 窗口出现 GLX 错误，把 `--viewer native` 改为
`--viewer viser`。

The first controller version uses four MCC Cartesian fingertip references and
position-only multi-site IK.  Its measured wrench input is fixed to zero.  The
four simulated fingertip 3-D forces are recorded for supervision and quality
analysis, but are not fed back into the controller yet.  Passive compliance
comes from the low-gain joint position actuators.

## 1. Collect

```bash
python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --device cuda:0 --num-envs 1 --trajectory-length 2500 \
  --max-trajectories 5 --motion-start 350
```

## 2. Invert to an object-fixed trajectory

```bash
python mcc_finger_compliance_control/scripts/invert_trajectories.py \
  --file mcc_finger_compliance_control/data/trajectories/<name>.h5
```

## 3. Geometry replay

```bash
python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file mcc_finger_compliance_control/data/inverted/<name>_inverted.h5 \
  --viewer native --mode teacher
```

Use `--viewer headless` for a finite replay and contact summary.

## 4. 下载当前 DP 模型与数据

Git 不跟踪大体积 checkpoint/H5。当前可复现版本发布在 GitHub Release
`dp-capsule-v1`，包含：

| Release 文件 | 放置位置 | 用途 |
|---|---|---|
| `best.pt` | `data/models/so3_uniform_v2_palm_geometry_local_planner_absolute_q_dp_25k/best.pt` | 部署 checkpoint |
| `dataset_info.json` | 与 `best.pt` 同目录 | 模型输入输出和训练配置 |
| `metrics.json` | 与 `best.pt` 同目录 | 训练指标 |
| `so3_uniform_v2_255x2500_relaxed999_inverted.h5` | `data/inverted/` | teacher replay、DP 部署初始状态 |
| `so3_uniform_v2_255x2500_relaxed999_palm_geometry_local_planner_dp.h5` | `data/inverted/` | 56 维 palm-frame DP 训练集 |

安装了 GitHub CLI 时，一次下载并放到约定位置：

```bash
cd ~/Code/Hand_Compliance_Control
mkdir -p /tmp/dp-capsule-v1
gh release download dp-capsule-v1 \
  --repo FerryRain/Hand_Compliance_Control \
  --dir /tmp/dp-capsule-v1

MODEL_DIR=mcc_finger_compliance_control/data/models/so3_uniform_v2_palm_geometry_local_planner_absolute_q_dp_25k
DATA_DIR=mcc_finger_compliance_control/data/inverted
mkdir -p "${MODEL_DIR}" "${DATA_DIR}"
cp /tmp/dp-capsule-v1/best.pt "${MODEL_DIR}/best.pt"
cp /tmp/dp-capsule-v1/dataset_info.json "${MODEL_DIR}/dataset_info.json"
cp /tmp/dp-capsule-v1/metrics.json "${MODEL_DIR}/metrics.json"
cp /tmp/dp-capsule-v1/so3_uniform_v2_255x2500_relaxed999_inverted.h5 "${DATA_DIR}/"
cp /tmp/dp-capsule-v1/so3_uniform_v2_255x2500_relaxed999_palm_geometry_local_planner_dp.h5 "${DATA_DIR}/"
```

未安装 `gh` 时，可以在项目的 Releases 页面下载同名文件，再按表格放置。部署参数快照保存在
[`configs/deployment/capsule_dp_mcc_v1.yaml`](configs/deployment/capsule_dp_mcc_v1.yaml)。

## 5. DP replay 与应用部署

先设置公共路径：

```bash
cd ~/Code/Hand_Compliance_Control
conda activate mjlab

INVERTED=mcc_finger_compliance_control/data/inverted/so3_uniform_v2_255x2500_relaxed999_inverted.h5
MODEL=mcc_finger_compliance_control/data/models/so3_uniform_v2_palm_geometry_local_planner_absolute_q_dp_25k/best.pt
```

### 5.1 完全 teacher replay

palm pose 和 `q_hand` 都逐帧来自 H5。这一步只验证反演坐标和碰撞环境：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file "${INVERTED}" \
  --episode-id 216 \
  --viewer headless \
  --mode teacher \
  --device cuda:0 \
  --max-steps 2150 \
  --contact-threshold 0.05
```

### 5.2 Teacher palm 路径上的闭环 DP + FullHandMCC

DP 使用实时手指状态；palm 仍执行 H5 中的教师路径；FullHandMCC 在高频层维持指尖接触：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file "${INVERTED}" \
  --model "${MODEL}" \
  --episode-id 216 \
  --mode live_dp \
  --viewer headless \
  --device cuda:0 \
  --inference-steps 10 \
  --chunk-execution \
  --dp-replan-interval 20 \
  --execution-layer fullhand_mcc \
  --mcc-direction-source oracle \
  --contact-threshold 0.05 \
  --max-steps 2150
```

### 5.3 主动规划 palm 的应用部署

下面不再 replay 教师 palm 路径：上层 active planner 沿胶囊子午线移动 palm，DP 预测手指
nominal pose，FullHandMCC 负责接触恢复和目标力。当前 `oracle` 法向来自解析胶囊，是仿真特权信息；
它用于验证完整 pipeline，不应被描述成无特权真机部署。

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file "${INVERTED}" \
  --model "${MODEL}" \
  --episode-id 216 \
  --mode live_dp \
  --viewer headless \
  --device cuda:0 \
  --max-steps 2150 \
  --inference-steps 10 \
  --chunk-execution \
  --dp-replan-interval 20 \
  --execution-layer fullhand_mcc \
  --mcc-direction-source oracle \
  --contact-threshold 0.05 \
  --palm-source active_capsule \
  --active-palm-surface-speed-mm-s 20 \
  --active-palm-travel-mm 400 \
  --active-palm-direction -1 \
  --active-palm-min-contact-fingers 3
```

`400 mm` 是请求值；planner 会按剩余胶囊子午线长度裁剪，episode 216 的有效路程约为
`313 mm`。CSV 中 `found_contacts` 是几何接触数，`loaded_contacts` 是同时满足
`|F_3D| >= 0.05 N` 的接触数。

## 6. 打开可视化

把上一节部署命令中的 `--viewer headless` 改成 `--viewer native` 即可打开 MuJoCo 窗口，并保留
DP target、法向和实时接触点标记：

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file "${INVERTED}" --model "${MODEL}" \
  --episode-id 216 --mode live_dp --viewer native --device cuda:0 \
  --inference-steps 10 --chunk-execution --dp-replan-interval 20 \
  --execution-layer fullhand_mcc --mcc-direction-source oracle \
  --palm-source active_capsule --active-palm-surface-speed-mm-s 20 \
  --active-palm-travel-mm 400 --active-palm-direction -1 \
  --active-palm-min-contact-fingers 3 --highlight-contacts
```

若 GLFW/GLX 无法创建窗口，将 `native` 改成 `viser`。无 NVIDIA GPU 时将
`--device cuda:0` 改成 `--device cpu`；CPU 可以检查功能，但实时性明显更低。

## 7. 录制跟手视角 MP4

`--viewer video` 使用离屏渲染，相机固定跟随 `robot/palm_lower`：

仓库内示例：[`outputs/active_capsule_dp_mcc_direction-1_400mm.mp4`](outputs/active_capsule_dp_mcc_direction-1_400mm.mp4)。
该次运行完成 `313.0 mm` 可达路径，`>=3` 指接触率为 `98.9%`，四指接触率为
`93.7%`，最大指尖力为 `1.98 N`。

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file "${INVERTED}" --model "${MODEL}" \
  --episode-id 216 --mode live_dp --viewer video --device cuda:0 \
  --video-output mcc_finger_compliance_control/outputs/active_capsule_dp_mcc_direction-1_400mm.mp4 \
  --video-fps 30 --video-width 960 --video-height 720 \
  --video-camera-distance 0.45 --video-camera-azimuth 315 --video-camera-elevation -10 \
  --max-steps 2150 --inference-steps 10 \
  --chunk-execution --dp-replan-interval 20 \
  --execution-layer fullhand_mcc --mcc-direction-source oracle \
  --contact-threshold 0.05 --palm-source active_capsule \
  --active-palm-surface-speed-mm-s 20 --active-palm-travel-mm 400 \
  --active-palm-direction -1 --active-palm-min-contact-fingers 3 \
  --highlight-contacts --seed 42
```

## 8. 训练同架构模型

Release 中较小的 `*_palm_geometry_local_planner_dp.h5` 是直接训练输入：

```bash
python mcc_finger_compliance_control/scripts/train_dp.py \
  --file mcc_finger_compliance_control/data/inverted/so3_uniform_v2_255x2500_relaxed999_palm_geometry_local_planner_dp.h5 \
  --output mcc_finger_compliance_control/data/models/my_capsule_dp \
  --device cuda:0 --steps 25000 --batch-size 256 \
  --stride 5 --obs-horizon 16 --pred-horizon 32 \
  --action-representation absolute_q \
  --diffusion-steps 100 --inference-steps 50 \
  --down-dims 256 512 1024 --seed 20260724
```

正式流水线脚本保留在 `scripts/`：采集、筛选、反演、replay、训练、统一部署、多物体目录与
YCB/碰撞分解工具。早期一次性 A/B、独立 Surface-MCC 部署和烟测入口已经移除，避免与当前
`deploy_dp_inverse.py --execution-layer fullhand_mcc` 混淆。
