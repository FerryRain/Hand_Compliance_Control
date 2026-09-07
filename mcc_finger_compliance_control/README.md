# MCC fingertip position-teacher pipeline(mustard v4 主线)

本目录独立于 legacy 管线 `finger_compliance_control/`。当前任务主线为
`Leaphand-Finger-MCC-Position-Control` 下的 **mustard 瓶身接触轨迹 DP**:

- **Teacher**:MCC 指尖位置控制器(特权表面 oracle + 四指法向力伺服),手指
  目标为 palm 系内绝对 tip 目标(12D);DP 学习"接触历史 + 未来手掌运动
  → 未来四指 tip 目标"。
- **Action contract(v4)**:`tip_target_palm`(palm 系绝对目标、无跨步累积,
  部署经 IK 解码,MCC 独立叠加法向力偏移)。
- **Observation**:dual-track B2 242D 因果状态(schema
  `contact_geometry_planner_motion_dual_track_v3`)。

详细配置、采集、反演、replay、DAgger 恢复数据与 DP 训练说明:

- [`MCC-Finger-Pipeline-Guide.md`](MCC-Finger-Pipeline-Guide.md) — 全流程 guide
- [`DAgger_Recovery_Data_Guide.md`](DAgger_Recovery_Data_Guide.md) — DAgger 式失触→恢复数据采集(状态 A/B,规划中)
- [`DP_DATA_COLLECTION_GUIDE.md`](DP_DATA_COLLECTION_GUIDE.md) — 数据设计讨论与 O/A 契约
- [`SPEED_UNIFORM_PROGRESS.md`](SPEED_UNIFORM_PROGRESS.md) — 逐日进度与 Gate(§32-44 为 DP/DAgger 记录)

## 当前数据与模型

| 内容 | 路径 |
|---|---|
| v4 训练文件(220 ep,含 ep226) | `data/dp/mustard_randomized_dual_track_v4_220_tip_target.h5` |
| v4 已训模型(219 ep = 220 剔除 226,15k 步) | `data/models/mustard_randomized_dual_track_v4_B2_tiptarget_219_15k/best.pt` |
| 闭环 rollout(DAgger 轮次/接管) | `data/closed_loop_rollouts/{dagger_mustard_*, current_policy}/` |
| 椭圆 palm 轨道与关键节点 | `data/plans/mustard_so3_v1/`(`*_opt.h5` 含 26 关键节点四指抓握) |

以下命令均在 `mcc_finger_compliance_control/scripts/` 目录内执行。Python 解释器:
`/home/rimlab/miniconda3/envs/mjlab/bin/python`(或先 `conda activate mjlab`)。

## 1. 数据采集(数采)

两条主链:批量轨道采集(planner_inverse,四指沿瓶身接触流形)与 dual-track
随机化(执行扰动制造 clean/perturbed 双轨,当前 v4 训练集来源)。

### 1.1 批量轨道采集 + 筛选(推荐)

```bash
cd mcc_finger_compliance_control/scripts

# 第一步:椭圆 palm 轨道批量生成 + 离线关键节点筛选(约 35 min @ 12 workers)
python batch_generate_so3_plans.py \
  --output-dir ../data/plans/mustard_so3_v2 --workers 12

# 第二步:通过离线检查的 plan 并行采集 + 接触率筛选(16 env/批)
python batch_collect_and_filter.py \
  --plans-dir ../data/plans/mustard_so3_v2 \
  --output-dir ../data/trajectories/mustard_so3_v2 \
  --selected-dir ../data/trajectories/mustard_so3_v2/selected \
  --envs-per-batch 16 --min-contact-rate 0.98
```

采集内部逐批 bundle 后调用 `collect_trajectories.py --motion-mode planner_inverse
--planner-file batch.h5`(物体固定、palm 沿轨道反演运动,教师逐帧解四指接触),
并在末端按接触率把合格轨迹导出到 `selected/`。`--execution-randomization` 可在
批量层透传,同一轨道生成 clean/perturbed 双轨标签。

### 1.2 单批/单模式采集(调试或定制)

```bash
python collect_trajectories.py \
  --device cuda:0 --num-envs 16 --trajectory-length 2500 \
  --max-trajectories 16 --motion-mode planner_inverse \
  --planner-file ../data/plans/<suite>/<plan>_opt.h5 \
  --initial-orientation-mode fixed --fixed-motion-start \
  --differential-contact-qp --no-contact-gate \
  --physics-substeps 10 --seed 42 --filename my_run
```

物体随机旋转等早期单轨模式仍可用(`--motion-mode rotation + angular-speed-*`,
见 `MCC-Finger-Pipeline-Guide.md` §4.2)。产物统一写 `data/trajectories/<name>.h5`。

## 2. DP 训练

当前 v4 推荐命令(219 ep、15k 步,含适度增广;增广只作用于训练窗口、验证集恒干净):

```bash
cd mcc_finger_compliance_control/scripts

python train_dp.py \
  --file ../data/dp/mustard_randomized_dual_track_v4_220_tip_target.h5 \
  --output ../data/models/mustard_randomized_dual_track_v4_B2_tiptarget_219_15k \
  --device cuda:0 --steps 15000 --batch-size 256 --lr 1e-5 \
  --stride 5 --obs-horizon 16 --pred-horizon 16 \
  --input-profile B2 \
  --action-field tip_target_palm --action-dim 12 \
  --action-representation absolute_q \
  --diffusion-steps 100 --inference-steps 100 --noise-scheduler DDPM \
  --down-dims 256 512 1024 --kernel-size 5 --n-groups 8 \
  --val-ratio 0.1 --num-workers 4 \
  --contact-dropout-probability 0.10 --max-contact-dropout-steps 3 \
  --qhand-dropout-probability 0.05 --qhand-dropout-max-steps 6 \
  --qhand-dropout-sigma 0.02 \
  --bus-contact-dropout-probability 0.05 \
  --exclude-episode-ids 226 \
  --seed 20260831 --save-every 1000 --eval-every 1000 --eval-samples 64
```

DAgger 混训:先用 `build_dagger_dataset.py` 把 rollout 切为
`action_q_hand`(时间对齐的成功教师 q),再给 `train_dp.py` 加
`--dagger-file <dagger.h5>` 与 `--dagger-sample-ratio`(见 §3 与
`DAgger_Recovery_Data_Guide.md`)。模型目录含 `run_manifest.json`
(完整参数与 episode 划分)、`metrics.csv`、`best.pt`。

## 3. 闭环部署(DP + FullHandMCC)

单条部署(也用于采集 DAgger rollout;此命令与 `collect_dagger_rollouts.py`
内部调用一致):

```bash
cd mcc_finger_compliance_control/scripts

python deploy_dp_inverse.py \
  --file ../data/dp/mustard_randomized_dual_track_v4_220_tip_target.h5 \
  --model ../data/models/mustard_randomized_dual_track_v4_B2_tiptarget_219_15k/best.pt \
  --episode-id 24 --mode live_dp --viewer headless --device cuda:0 \
  --inference-steps 10 \
  --execution-layer fullhand_mcc --mcc-direction-source hybrid \
  --mcc-preset collection_matched_sensor \
  --dp-history-q-source nominal \
  --chunk-execution --dp-replan-interval 10 \
  --max-steps 600 --seed 42 \
  --rollout-h5 ../data/closed_loop_rollouts/<round>/ep024.h5 \
  --report ../data/closed_loop_rollouts/<round>/ep024.csv
```

说明:

- `--mode live_dp`:DP 用实时手指状态闭环预测,`--max-steps` 限制部署长度;
- 物体与初始手型读自 `--file`(H5 attrs),不需要单独传 object;
- `--rollout-h5/--report` 记录因果观测 + 教师 q(csv 逐帧诊断);
- `--live-teacher-takeover-frame N`:第 N 帧起保留物理状态、改执行教师 q
  (失触后打恢复标签的现有机制);
- 失触检测/暂停(状态 A)→ 恢复数据采集流程见
  [`DAgger_Recovery_Data_Guide.md`](DAgger_Recovery_Data_Guide.md)。

批量多 episode rollout:

```bash
python collect_dagger_rollouts.py \
  --file ../data/dp/mustard_randomized_dual_track_v4_220_tip_target.h5 \
  --model ../data/models/mustard_randomized_dual_track_v4_B2_tiptarget_219_15k/best.pt \
  --episodes 24 186 226 \
  --output-dir ../data/closed_loop_rollouts/dagger_mustard_round4 \
  --max-steps 600 --device cuda:0 --seed 20260831
```

## 4. 可视化

- 几何轨道/关键节点检查:`view_inverse_palm_plans.py <plan>_opt.h5`
- 采集轨迹接触诊断:`visualize_orbit_contact.py <traj.h5>`
- 部署看过程:把部署命令 `--viewer headless` 改为 `--viewer native`
  (GLX 不稳用 `viser`);录像用 `--viewer video`(见脚本 `--video-*` 参数)。

## 5. 历史说明

早期 capsule-DP 发布(`so3_uniform_v2`、GitHub Release `dp-capsule-v1`、56D
palm 输入、absolute_q 16D 输出)已被当前 mustard v4 tip-target 契约取代,
代码与文档仍留在仓库(replay/release 命令见 git 历史与旧版 README)。当前
v4 决策记录于 `data/models/.../run_manifest.json` 与 `SPEED_UNIFORM_PROGRESS.md`。
