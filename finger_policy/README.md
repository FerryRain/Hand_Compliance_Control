# Finger Policy

项目根目录下的分指 Diffusion Policy 工作目录。包含 mocap 数据采集、实测关节标签导出、训练、闭环部署评估和原速回放。默认加载最近一轮五物体视频使用的 `single_all` 联合 checkpoint。

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 采集与规划、候选审核、观测重建、分指模型、训练和评估源码。共享的 MCC、恢复与仿真组件沿用同一工作区的实现。 |
| `data/old222/` | 222 条旧轨迹的预处理帧数组与原始分组划分。 |
| `data/expanded/` | 718 个新增分指 shard 和清单，保留原训练、验证、留出与排除划分。 |
| `data/protocols/` | 31 个采集任务、29 个部署参考和训练配置。 |
| `data/raw_collection/` | 126 份原始采集目录的链接，清单记录原始路径和 rollout SHA-256。 |
| `data/evaluations/` | 五物体三组策略的结果、请求、审计与视频元数据。实测轨迹与 MP4 链接到原档案。 |
| `logs/` | 采集日志及数据准入审核、9 组训练日志、五物体部署和渲染日志。 |
| `checkpoints/` | Flat common、单指 common、single_all 各保留评估种子 `20261116` 的 `best.pt`。`latest_single_all.pt` 指向同一份默认部署权重（18,000 步）。其他种子的原权重在 `data/provenance.json` 记录的训练档案中。 |

分指模型使用一个长指共享 U-Net 与一个拇指 U-Net。每指输入 16 帧 × 45 维，包含接触位置与法向、关节及接触运动、双向关节裕量、手指身份和上层 palm command。输出 16 个未来的 4 维关节增量，以当前实测关节角为锚重建动作，再送入原 scheduler 和 MCC。

分指 CUDA 部署在初始化时捕获一个单手形状的 CUDA Graph。每个去噪步用两条 stream 并行执行长指和拇指 U-Net，结果合并后按原 10 步 DDPM 更新。图捕获不适用时，CUDA 分支用普通双 stream 执行。

## 运行

下面的命令从 `finger_policy/` 运行。本机使用 `mjlab`，采集、训练、部署与视频渲染需要相应的 CUDA 或 EGL 访问。

```bash
cd /home/rimlab/Code/Hand_Compliance_Control/finger_policy
PY=/home/rimlab/miniconda3/envs/mjlab/bin/python
"$PY" pipeline.py verify
"$PY" pipeline.py collection-list
"$PY" pipeline.py reference-list
```

### 1. 采集、审核和导出

选择 `single_region`（跨区域单指）、`single_banana`（三维位姿单指）、`multi`（同步多指）或 `whole`（离线检查过的整手参考）。指定 `--job` 可只采一个任务，省略时运行该配方下全部任务。

```bash
"$PY" pipeline.py collect single_banana --job banana_3d_roll45 --run banana_new
"$PY" pipeline.py index --audit-dirs logs/collection_runs/banana_new \
  --output data/collection_runs/banana_new/candidates.json
"$PY" pipeline.py export --manifest data/collection_runs/banana_new/candidates.json \
  --output data/collection_runs/banana_new/shards
```

采集使用 mocap 固定手掌和预定物体运动，手指仍由原 MCC 逐帧执行。标签来自实测未来关节角。审计和导出保留失败记录，只有通过原准入检查的样本进入训练池。可先用 `collect … --prepare-only` 生成队列并核对冻结源文件，之后在同名运行加 `--resume` 执行。恢复队列前先检查 `logs/collection_runs/<run>/status.json`。

### 2. 训练

```bash
"$PY" pipeline.py train --group single_all --seed 20261116 --run finger_new
# 使用额外导出的实测 shard 训练
"$PY" pipeline.py train --group single_all --seed 20261116 --run finger_with_banana \
  --add-shards data/collection_runs/banana_new/shards/manifest.json
```

默认沿用旧/新各半窗口、每根手指均衡及原物体留出划分。新增样本仍经过相同划分，Flat 和单指 common 的新数据要求四指来自同一实测轨迹。训练权重和指标输出到 `checkpoints/runs/<run>/`，使用的协议、数据清单和标准输出保存到 `logs/training_runs/<run>/`。加 `--canary-steps 1` 可只验证一次参数更新和验证集接口。

### 3. 闭环部署评估与 1 倍速回放

```bash
# 使用已收录的 single_all 权重评估 Mustard 连续长参考
"$PY" pipeline.py evaluate --group single_all --reference long_natural --run mustard_review
# 换用新训练 checkpoint
"$PY" pipeline.py evaluate --group single_all --reference long_natural --run mustard_new \
  --checkpoint checkpoints/runs/finger_new/best.pt
"$PY" pipeline.py render \
  --result data/evaluation_runs/mustard_review/evaluation/single_all_seed20261116/long_natural.json \
  --output data/evaluation_runs/mustard_review/replay --title "Mustard | single_all"
```

评估沿用所选参考的冻结请求、MCC 和既有恢复，不启用额外姿态纠偏。结果记录路径完成度、严格真指尖和 found 接触率、关节裕量以及过力。加 `--canary` 只测试 200 帧参考的接口。原速回放读取实测状态，保留实际终止帧。

Git 提供工作流源码及共享的仿真、MCC 依赖。使用现有权重部署前，按交付包 README 解压 checkpoint 和 YCB 运行资产，并准备对应的冻结评估请求、掌参考和初始状态。重新采集或训练还需要相应的协议或输入数据。上面的复现实验命令假定这些输入已放回项目中对应的位置。原冻结协议里的绝对路径用于溯源，换机器时需按实际路径配置。运行数据、日志、视频和权重都保存在 Git 之外。
