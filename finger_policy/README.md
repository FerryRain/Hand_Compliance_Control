# Finger Policy

LEAP Hand 的 Diffusion Policy 工作流，包含 mocap 数据采集、实测关节轨迹导出、训练、闭环仿真评估和 1 倍速回放。默认使用 `single_all`。控制链由上层提供手掌参考，DP 预测手指动作，scheduler 拼接并下发参考，MCC 做末端柔顺修正。

## 1. 下载和准备

下载完整的 `FingerPolicy_Data` 文件夹（约 **3.38 GB**），保留目录结构。文件夹顶层有 `policy/`、`assets_3d/`、`data/`，以及 `link_data.py`、`GUIDE.md`、`INVENTORY.md` 和校验清单。[Google Drive 下载入口](https://drive.google.com/drive/folders/1XMBF-VGcsxUDkoRjZV8JFCfsbjyNh9ep?usp=drive_link)

| 下载目录 | 用途 |
| --- | --- |
| `policy/` | 三个可选权重，均为训练种子 20261116。部署时用 `--group` 选择 |
| `assets_3d/` | `YCB/` 内分标准表面、碰撞与运行资产、高精扫描；`collection_shapes/` 保存采集用函数曲面和几何体 |
| `data/training/` | Mustard 222 条和扩充数据的 `prepared/` 供重训直接读取。两个 `raw.zip` 供分析原始采集记录 |
| `data/evaluation/` | `reused_mustard222/` 是原数据的掌参考，`planned_long/` 是另行规划的长参考，`protocol/` 和 `runtime/` 提供初态、请求及加载器依赖 |

在 **Linux 的干净源码仓库根目录**执行。先准备好项目的 `mjlab` 环境和 CUDA，把第一行改成实际下载目录。

```bash
bundle="$HOME/Downloads/FingerPolicy_Data"
(cd "$bundle" && sha256sum --quiet -c SHA256SUMS.txt) &&
  python "$bundle/link_data.py" "$PWD"
```

脚本让仓库直接读取下载目录中的文件。连接成功后请保留下载目录的位置。发布代码中的 Mustard 默认配置会备份为 `ycb_mustard.yaml.before-finger-data`，随后使用与评估轨迹匹配的配置。

```bash
conda activate mjlab
cd finger_policy
python pipeline.py evaluate --group single_all --reference long_natural --run first_check --canary
```

这条命令用扩充分指策略运行 Mustard 长参考的前 200 帧。去掉 `--canary` 并换一个 `--run` 名即可跑全长。部署和重训都直接使用已展开的数据。需要查看原始 HDF5 时，再按需执行对应的一行：

```bash
unzip -n "$bundle/data/training/mustard222/raw.zip" -d "$bundle/data/training/mustard222"
unzip -n "$bundle/data/training/expanded/raw.zip" -d "$bundle/data/training/expanded"
```

部署场景通过物体 YAML 读取表面网格和预分解凸碰撞件。视频回放需要 EGL 和一次本机部署生成的结果。重跑原采集任务还需对应采集协议、对象池配置与初态，详见第 3 节。

## 2. 模型和训练数据

| 策略 | 结构 | 新增训练样本 | 云盘权重 |
| --- | --- | --- | --- |
| `flat_common` | 一个全手条件 U-Net，联合预测 16 个关节的动作块 | 同一物理试次、同一时间的四指观测和标签 | `policy/flat_common.pt`，19,000 步 |
| `single_common` | 长指共享一个 U-Net，拇指用另一个 | 与 `flat_common` 相同的整手试次 | `policy/single_common.pt`，19,000 步 |
| `single_all` | 与 `single_common` 相同 | 同一训练物体内可分别从各指的实测试次取样，涵盖单指、多指和整手采集 | `policy/single_all.pt`，18,000 步，默认部署 |

三组权重的训练 seed 都是 `20261116`。每指使用 **16 帧 × 45 维** 的历史，每隔 50 ms 取一帧，包含关节和接触位置与法向、运动量、双向关节裕量、指身份和已经发布的掌运动。模型预测 16 个未来关节增量。每个预测步都以同一个当前实测关节角为基准，监督标签来自之后的实测关节角。

全手模型使用四指各自的 33 维信息和共享的 12 维掌运动。分指模型的长指分支在三个长指间共享参数，拇指分支单独建模。训练在 100 个 DDPM 噪声时间步上抽样，部署默认去噪 10 步。分指 CUDA 运行时用两条 stream 并行计算两个分支，单手条件满足时使用 CUDA Graph。

- **旧数据**是原 Mustard 222 条轨迹的实测帧，按基础轨迹和增强组保留 169／20／33 条训练、验证、测试划分。原始来源为 `mcc_finger_compliance_control/data/dp_stratified_222_20261022/`。解压 Mustard 原始包后，可将 `$bundle/data/training/mustard222/raw/prepared/` 用作 [single_finger_dp.py](scripts/single_finger_dp.py) 的 `prepare --dataset` 输入，重新生成旧帧数组。
- **新增数据**由 mocap 采集的函数曲面、程序化物体和 YCB 物体导出。`data/expanded/manifest.json` 索引 718 个分指 shard，其中 537 个划入训练、83 个验证、93 个物体留出、5 个排除。`flat_common` 和 `single_common` 的新增训练池来自其中 19 条同步整手试次。
- **物体划分**以对象为单位。新增数据使用糖盒做验证，苹果、梨和午餐肉罐做留出。扩展数据的原始来源见下载目录的 `data/training/source_index.csv`，每个 shard 的对象、手指和划分见扩展清单。

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

按第 1 节连接下载目录后，在仓库的 `finger_policy/` 运行 `python pipeline.py verify --training-data-only`，核对 722 项训练输入。

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

从仓库的 `finger_policy/` 运行。按评估目的选择 `data/evaluation_inputs_full/` 下的清单。

| 清单 | 参考范围 |
| --- | --- |
| `native20-catalog.json` | 原来全量评估的 20 条 Mustard 验证掌轨迹 |
| `train-palm-catalog.json` | 直接复用训练掌运动的 169 条 Mustard 轨迹 |
| `original-long-catalog.json` | 新规划的 9 条长参考，覆盖 Mustard、罐头、盒子、清洁剂瓶、水壶，以及苹果和梨各两个方向 |
| `height-variant-catalog.json` | Mustard 长轨迹的掌间距变体 |
| `catalog.json` | 默认 63 条，包括 20 条验证、33 条留出、9 条新长参考及 1 条高度变体 |

每条请求 JSON 指定手掌参考、初态和环境。评估器从 `dataset` 读取 `teachers_inverted.h5` 和 `teachers_commanded_palm.h5`，从 `rows[].source` 读取初始手指目标。请求中的 `checkpoint` 供基础部署链初始化，`--group` 或 `--checkpoint` 选择实际预测指动作的策略。`--seed` 选择打包模型的训练种子，仿真随机种子由请求指定。

这些请求关闭了 DAgger 教师纠偏和 residual，保留 MCC 和重抓握。

### 手掌参考的来源

下载目录的 `data/evaluation/trajectory_index.csv` 逐条列出 `palm_source_split`、`reused_training_palm`、初态来源和各策略训练时是否见过该物体。

- 原有 20 条来自 Mustard 验证集，训练用该验证集选择 checkpoint。
- 沿用训练集手掌轨迹的 169 条回放标为 `reused_training_palm=True`，放在 `train-palm-catalog.json`。做回归诊断时可用，报告时与留出测试分开。
- 两条 Mustard 长参考以验证集 ep71 初始化，后续采用跨区路线或其高度变体。YCB 长参考由已知物体几何生成，物体训练时已见和手掌参考已见分别标记。
- 4 条 G05 历史 60 秒参考保存在 `historical_60s_spin_tilt/`，附原始命令和初态，索引标为历史档案。

```bash
EVAL_CATALOG=data/evaluation_inputs_full/catalog.json
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

将示例中的参考名改为 `holdout_ep0014` 可选留出测试。回放训练掌路线时，指定 `--catalog data/evaluation_inputs_full/train-palm-catalog.json` 和 `train_palm_epNNNN` 参考。要一次跑完原 20 条参考，见下载目录 `GUIDE.md` 的批量命令。
