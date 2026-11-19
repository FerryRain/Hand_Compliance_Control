# Finger Policy

LEAP Hand 的 Diffusion Policy 工作流，包含 mocap 数据采集、实测关节轨迹导出、训练、闭环仿真评估和 1 倍速回放。默认使用 `single_all`。控制链由上层提供手掌参考，DP 预测手指动作，scheduler 拼接并下发参考，MCC 做末端柔顺修正。

## 1. 下载和准备

在 [Google Drive 交付目录](https://drive.google.com/drive/folders/1XMBF-VGcsxUDkoRjZV8JFCfsbjyNh9ep?usp=drive_link) 下载需要的文件。部署现有模型先取前两个 ZIP 和评估输入，评估输入的解压步骤见第 5 节。

| 文件 | 内容和用途 |
| --- | --- |
| `finger-policy-checkpoints-20261119.zip` | 三组策略各一份 `best.pt`，以及 `finger_policy/checkpoints/manifest.json`。 |
| `ycb-collision-and-runtime-20261119.zip` | 七个部署物体的 YAML、连续表面 OBJ 和预分解凸碰撞件，并附已有的采集几何。 |
| `ycb-all-object-surfaces-20261119.zip` | YCB 标准表面网格、纹理和已有点云，用于查看源几何或新增物体。 |
| `ycb-highres-surfaces-20261119.zip` | 碗、手电钻和大夹钳的高精度扫描网格，用于几何检查或重做分解。 |
| `finger-policy-evaluation-inputs-20261119.zip` | 29 个闭环评估请求、手掌参考 HDF5 和初态。 |
| `finger-policy-evaluation-bootstrap-20261119.zip.part-001` 至 `part-004` | 评估入口初始化共用执行链时读取的基础 checkpoint。 |

ZIP 内路径以 **仓库根目录** 为起点。下面的命令从仓库内任意目录运行。先将 `DOWNLOADS` 改成下载目录。仅下载部分 ZIP 时，校验会检查已下载的文件。

```bash
REPO=$(git rev-parse --show-toplevel)
DOWNLOADS=/path/to/FingerPolicy_YCB_20261119
(
  cd "$DOWNLOADS"
  sha256sum --ignore-missing -c SHA256SUMS.txt
)
unzip "$DOWNLOADS/finger-policy-checkpoints-20261119.zip" -d "$REPO"
unzip "$DOWNLOADS/ycb-collision-and-runtime-20261119.zip" -d "$REPO"

conda activate mjlab
cd "$REPO/finger_policy"
python pipeline.py --help
```

每个物体的 YAML、表面 OBJ 和碰撞件路径，以及 V-HACD 分解参数，见云盘的 `README.txt`。连续表面用于渲染和源网格法向查询，`collision_part_*.obj` 用于仿真碰撞。芥末瓶使用 256 个凸件，其余六个已配置物体各用 128 个。

### 各流程还需要的输入

| 用途 | 原实验工作区中的文件 |
| --- | --- |
| 采集 | `data/protocols/collection/recipes.json` 和任务协议。按协议准备对象池的 `objects/`、`object_manifest.json`、部分任务的初态和轨迹文件，以及协议指定的碰撞后端。 |
| 训练 | `data/old222/single_finger_frames.npz`、`data/old222/episode_split.json`、`data/expanded/manifest.json` 和分指 NPZ、`data/protocols/training.json`、`data/integrity.json`。 |
| 闭环评估 | 云盘评估包解压到 `data/evaluation_inputs/` 后，提供参考索引、请求、命令 HDF5、初态 HDF5 和基础 checkpoint。原工作区也可继续使用 `data/protocols/`。 |
| 回放 | 一次闭环评估的结果 JSON、同名 `.request.json` 和结果所引用的实测轨迹 NPZ。 |

**Git 提供源码，云盘提供三组权重、YCB 几何和闭环评估输入。** 采集协议和完整训练数据仍保存在原实验工作区。迁移采集和训练任务时，按新机器的实际位置恢复原协议引用的路径。部署和训练使用 CUDA，视频回放使用 EGL。命令假定当前环境已经安装本仓库的 mjlab 和相关依赖。

## 2. 模型和训练数据

| 策略 | 结构 | 新增训练样本 | 云盘权重 |
| --- | --- | --- | --- |
| `flat_common` | 一个全手条件 U-Net，联合预测 16 个关节的动作块 | 同一物理试次、同一时间的四指观测和标签 | `checkpoints/flat_common_seed20261116/best.pt`，19,000 步 |
| `single_common` | 长指共享一个 U-Net，拇指用另一个 | 与 `flat_common` 相同的整手试次 | `checkpoints/single_common_seed20261116/best.pt`，19,000 步 |
| `single_all` | 与 `single_common` 相同 | 同一训练物体内可分别从各指的实测试次取样，涵盖单指、多指和整手采集 | `checkpoints/single_all_seed20261116/best.pt`，18,000 步，默认部署 |

三组权重的训练 seed 都是 `20261116`。每指使用 **16 帧 × 45 维** 的历史，每隔 50 ms 取一帧，包含关节和接触位置与法向、运动量、双向关节裕量、指身份和已经发布的掌运动。模型预测 16 个未来关节增量。每个预测步都以同一个当前实测关节角为基准，监督标签来自之后的实测关节角。

全手模型使用四指各自的 33 维信息和共享的 12 维掌运动。分指模型的长指分支在三个长指间共享参数，拇指分支单独建模。训练在 100 个 DDPM 噪声时间步上抽样，部署默认去噪 10 步。分指 CUDA 运行时用两条 stream 并行计算两个分支，单手条件满足时使用 CUDA Graph。

- **旧数据**是原 Mustard 222 条轨迹的实测帧，按基础轨迹和增强组保留 169／20／33 条训练、验证、测试划分。原始来源为 `mcc_finger_compliance_control/data/dp_stratified_222_20261022/`。有这套源 HDF5 时，可用 [single_finger_dp.py](scripts/single_finger_dp.py) 的 `prepare` 子命令生成旧帧数组。
- **新增数据**由 mocap 采集的函数曲面、程序化物体和 YCB 物体导出。`data/expanded/manifest.json` 索引 718 个分指 shard，其中 537 个划入训练、83 个验证、93 个物体留出、5 个排除。`flat_common` 和 `single_common` 的新增训练池来自其中 19 条同步整手试次。
- **物体划分**以对象为单位。新增数据使用糖盒做验证，苹果、梨和午餐肉罐做留出。扩展数据的原始来源和哈希见 `data/provenance.json`，每个 shard 的对象、手指和划分见扩展清单。

## 3. 采集、审核和导出

采集时以 mocap 支撑手掌，按任务协议发布物体运动，手指由 MCC 执行。整手任务使用离线检查过的参考。采集队列写原始 `rollout.h5` 与物体运动计划，并为每次任务写审计结果。候选索引检查原接触和力门、运动完整性、关节裕量以及异常快步。导出阶段重建因果观测并记录实际未来关节角。未通过模仿学习准入的原始记录仍可用于诊断。

| 配方 | 采集入口 |
| --- | --- |
| `single_region` | [collect_contact_wm_feature_motion.py](scripts/collect_contact_wm_feature_motion.py)。单指跨区域，按协议执行预定物体运动。 |
| `single_banana` | [collect_contact_wm_object_waypoints.py](scripts/collect_contact_wm_object_waypoints.py)。单指跟随香蕉的三维位姿序列。 |
| `multi` | [collect_contact_wm_multi_waypoints.py](scripts/collect_contact_wm_multi_waypoints.py)。多个手指在同一试次同步采集。 |
| `whole` | [collect_contact_wm_planned.py](scripts/collect_contact_wm_planned.py)。使用已有手型参考采集整手数据。 |

从 `finger_policy/` 运行下面的单任务示例。已有对应协议及输入时，先查看任务名，再用一个新的 `--run` 名启动。

```bash
python pipeline.py collection-list
python pipeline.py collect single_banana --job banana_3d_roll45 --run banana_new

python pipeline.py index --audit-dirs logs/collection_runs/banana_new \
  --output data/collection_runs/banana_new/candidates.json
python pipeline.py export --manifest data/collection_runs/banana_new/candidates.json \
  --output data/collection_runs/banana_new/shards
```

[run_contact_wm_collection_queue.py](scripts/run_contact_wm_collection_queue.py) 执行和审计队列，[prepare_expanded_finger_candidates.py](scripts/prepare_expanded_finger_candidates.py) 生成候选，[export_wm_finger_candidates.py](scripts/export_wm_finger_candidates.py) 写分指 NPZ 和 `shards/manifest.json`。其中 `observation` 保存逐指观测，`q` 保存实测关节角，`roots` 标出可取出完整历史和未来标签的帧。用 `--prepare-only` 可先生成队列，再以相同配方和 `--run` 加 `--resume` 启动，无需重复指定 `--job`。进度见 `logs/collection_runs/<run>/status.json`。

## 4. 训练

训练从随机初始化开始。冻结协议默认运行 20,000 个更新，每次抽 64 组窗口，旧和新增数据各半。长指和拇指按每根手指的样本数均衡采样。归一化来自旧训练集，按旧验证集和糖盒整手验证集的前四步 MAE 等权选择 `best.pt`。

```bash
python pipeline.py train --group flat_common   --seed 20261116 --run flat_new
python pipeline.py train --group single_common --seed 20261116 --run single_common_new
python pipeline.py train --group single_all    --seed 20261116 --run single_all_new

# 将本次采集通过审核的样本纳入训练
python pipeline.py train --group single_all --seed 20261116 --run single_all_banana \
  --add-shards data/collection_runs/banana_new/shards/manifest.json
```

`--canary-steps 1` 可用一个独立的 `--run` 验证训练和验证接口。新样本仍按原物体划分，两个 common 组只接收能组成同步四指试次的新增样本。权重和训练指标写入 `checkpoints/runs/<run>/`，实际使用的协议、清单和标准输出写入 `logs/training_runs/<run>/`。

入口是 [train_expanded_policy_comparison.py](scripts/train_expanded_policy_comparison.py)。[expanded_policy_data.py](scripts/expanded_policy_data.py) 负责新旧采样与划分，[controlled_policy_ablation.py](scripts/controlled_policy_ablation.py) 负责模型组合、去噪及部署 provider，底层 U-Net 复用仓库中的 [train_dp.py](../mcc_finger_compliance_control/scripts/train_dp.py)。

## 5. 部署评估和原速回放

下载评估输入 ZIP、基础 checkpoint 的四个分卷和 `EVALUATION_SHA256SUMS.txt`，放在第 1 节的下载目录。在同一个终端执行以下命令。

```bash
(
  cd "$DOWNLOADS" &&
  cat finger-policy-evaluation-bootstrap-20261119.zip.part-* > finger-policy-evaluation-bootstrap-20261119.zip &&
  sha256sum -c EVALUATION_SHA256SUMS.txt
)
unzip "$DOWNLOADS/finger-policy-evaluation-inputs-20261119.zip" -d "$REPO"
unzip "$DOWNLOADS/finger-policy-evaluation-bootstrap-20261119.zip" -d "$REPO"
cd "$REPO/finger_policy"
```

29 条参考覆盖原 Mustard 20 条、Mustard 连续长参考和 8 条其他 YCB 参考。每条请求 JSON 指定手掌参考、初态和环境。评估器从 `dataset` 读取 `teachers_inverted.h5` 和 `teachers_commanded_palm.h5`，从 `rows[].source` 读取初始手指目标。请求中的 `checkpoint` 供基础部署链初始化，`--group` 或 `--checkpoint` 选择实际预测指动作的策略。`--seed` 选择打包模型的训练种子，仿真随机种子由请求指定。

当前 29 个请求的 DAgger 教师纠偏和 residual 均关闭。闭环仍运行 MCC 和请求配置的重抓握。历史 DAgger 教师代码与示范数据属于独立训练实验，本评估入口无需加载。

```bash
EVAL_CATALOG=data/evaluation_inputs/catalog.json
python pipeline.py reference-list --catalog "$EVAL_CATALOG"
python pipeline.py evaluate --catalog "$EVAL_CATALOG" --group single_all --reference long_natural \
  --run mustard_review
python pipeline.py evaluate --catalog "$EVAL_CATALOG" --group single_all --reference long_natural \
  --run mustard_new --checkpoint checkpoints/runs/single_all_new/best.pt

python pipeline.py render \
  --result data/evaluation_runs/mustard_review/evaluation/single_all_seed20261116/long_natural.json \
  --output data/evaluation_runs/mustard_review/replay --title "Mustard | single_all"
```

已有请求时，追加 `--canary` 可先测 200 个参考帧。其他长参考包括 `ycb_016_pear_forward`、`ycb_021_bleach_cleanser`、`ycb_003_cracker_box` 和 `ycb_002_master_chef_can`。评估记录完成度、严格真指尖接触率、found、关节裕量和过力。回放读取实测状态，保留实际终止帧，输出 `deployment_review.mp4`。

[evaluate_expanded_independent_job.py](scripts/evaluate_expanded_independent_job.py) 构造请求并汇总，[evaluate_controlled_policy_ablation.py](scripts/evaluate_controlled_policy_ablation.py) 接入所选策略，仿真使用共用的 [deploy_b_gpu_serial_parity.py](../mcc_finger_compliance_control/scripts/deploy_b_gpu_serial_parity.py) 和 MCC。[render_focused_policy_video.py](scripts/render_focused_policy_video.py) 负责渲染。

`python pipeline.py verify` 会检查原工作区的数据哈希、权重和原始采集链接，适用于已恢复完整原实验输入的工作区。
