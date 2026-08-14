# 触觉探索与接触流形 DP：可执行路线图

本文档把当前项目从“多物体教师数据采集”推进到“未知物体表面实时探索”拆成可逐项验收的阶段。它同时说明四套可递进方案、数据质量与规模、数据表征、DP 架构、底层控制器、上层接口，以及可用于 MuJoCo 采集的物体模型数据集。

本文档中的状态标记：

- **已有**：仓库中已经有可运行实现。
- **待补**：进入下一阶段前必须实现。
- **后期**：不阻塞当前基线。

## 1. 任务定义

最终任务不是让 DP 直接承担所有控制，而是让三层系统各自解决合适的时间尺度：

```text
Active surface planner（1–5 Hz）
  给出手掌探索方向、速度、距离和风险预算
                 ↓
Structured tactile DP（5–10 Hz）
  根据触觉/运动历史，在接触流形上选择未来接触移动策略
                 ↓
Differential QP（100–500 Hz）
  把策略投影成满足 Jacobian、关节限位和自碰撞约束的可执行速度
                 ↓
MCC / force controller（200–500 Hz）
  维持法向接触、失触恢复、过力退让和高频稳定
```

核心分工是：

- DP 学“沿接触流形往哪里走”和“多解中选哪一种”。
- QP/MCC 保证当前动作尽量留在接触流形上。
- 上层规划器决定探索目标，不要求 DP 解决机械臂全局可达性。

## 2. 四套递进方案

### 2.1 方案 A：未来关节位置 DP（当前基线）

用途：最快验证数据、反演、模型和部署闭环，不作为最终架构。

当前 palm-frame `contact_geometry` 输入为 50 维：

```text
q_hand                              16
fingertip_contact_pos_palm          12
fingertip_contact_normal_palm       12
fingertip_contact_mask               4
palm_relative_twist_palm             6
```

DP 输出未来 `H × 16` 的手指关节位置，优先使用 `absolute_q`，避免持续积分 `delta_q` 的小偏差。底层 MCC 只允许在 nominal 周围施加有限的法向接触补偿。

建议模型：仓库现有 LeRobot Conditional 1D U-Net；`obs_horizon=16`、`pred_horizon=32`、训练扩散步数 100、部署 DDIM 8–16 步、5–10 Hz 重规划。

优点：现有代码可直接训练。缺点：把曲面运动和关节实现纠缠在一起，跨物体泛化弱，闭环误差容易改变后续历史分布。

### 2.2 方案 B：接触流形上的微分策略 DP（推荐主线）

这套方案更接近 Aude 类方法的思想：局部接触点在切平面内发生小位移，而法向约束、姿态冗余和关节约束由优化器处理。

对每根目标接触手指建立时间连续的接触基：

```text
B_i = [t_i1, t_i2, n_i]
```

先扣除手掌运动带来的刚体速度，再生成教师标签：

```text
v_tip_rel = v_tip - G_i V_palm
u_i_tan   = [t_i1, t_i2]^T v_tip_rel
v_i_n     = n_i^T v_tip_rel
```

DP 每次预测未来窗口：

```text
u_tan[H, 4, 2]          每指切向速度/位移
q_posture[H, 16]        冗余姿态偏好，不是硬关节命令
contact_mode[H, 4]      maintain / weak / lost / recovered / overforce
force_band[H, 4, 2]     期望力区间，可选
normal_future[H, 4, 3]  未来局部法向，可选辅助头
```

QP 在高频循环中求解：

```text
min ||T^T(J qdot + G V_palm) - u_tan||²
  + λ_posture ||N qdot - qdot_posture||²
  + λ_smooth ||qdot - qdot_prev||²
  + λ_slack ||s||²

s.t. 法向接触速度/恢复约束
     关节位置与速度限制
     自碰撞和物体穿透距离限制
     必要时允许带惩罚的 slack
```

MCC 的优先级必须是：先恢复几何接触，再进入目标力区间；过力则退让。不能让力误差补偿无限覆盖 DP 的切向意图。

### 2.3 方案 C：全手触觉版

如果真机升级为指尖视触觉 + 手掌/指腹密集触觉，数据不应展平成固定的大向量，而应保存为带空间位置的 tactile tokens：

```text
link_id, taxel_id
position_palm[3], normal_palm[3]
pressure 或 force_vector[1/3]
d_pressure 或 d_force[1/3]
contact_mask, age, confidence
sensor_type
```

指尖视触觉先由轻量 CNN/ViT 编码；压阻阵列按 link 用 PointNet、稀疏 Transformer 或 GNN 编码；各 link token 再进入时序融合模块。DP 的动作仍建议输出切向策略和姿态偏好，而不是直接输出高频力矩。

仿真阶段先实现 11 个区域级传感器，再细化到 taxel。不要一开始就模拟数百个独立碰撞 geom，否则吞吐量和接触噪声都会恶化。

### 2.4 方案 D：完整主动探索系统

在方案 B/C 稳定后加入局部表面记忆：

```text
surface_memory[N=32..128]
  = {point_palm, normal_palm, confidence, age, support/link id}
```

DP 同时预测未来局部几何变化和 2–4 个多模态切向策略候选；上层依据不确定性、QP 可行性、接触风险和信息增益选候选。4090 实时部署时，先使用单候选 8–12 步 DDIM；多候选可以降低频率或异步生成。

## 3. 数据规模与训练时长

轨迹数量不是唯一指标。有效样本数取决于不同物体、接触区域、曲率变化、运动轴、速度和恢复事件的覆盖。

| 阶段 | 建议轨迹量 | 目的 |
|---|---:|---|
| 单轨迹 overfit | 10–30 | 检查坐标、标签、replay 和模型是否能记住 |
| 同物体闭环 | 300–1,000 | 检查 contact-history covariate shift |
| 基础形状课程 | 2,000–5,000 | 球、胶囊、椭球、圆角盒、圆柱、组合体 |
| 多物体方案 B | 7,500–15,000 | 3–5k 基础形状，2–4k 轴/构型组合，0.5–1.5k 恢复，2–5k 不规则物体 |
| 全手触觉方案 C | 15,000–30,000 标注轨迹 | 另加 10–50 h 或 50k–200k 无标注触觉片段做预训练 |
| 完整探索方案 D | 20,000–50,000 高质量专家轨迹 | 加入未见类别、边缘、凹区、把手、孔洞和材质变化 |

一条 2500 控制步、`dt=0.01 s` 的轨迹相当于 25 秒交互。`stride=5` 后为 20 Hz 数据，适合 5–10 Hz policy 通过滑动窗口读取，但短时失触不能因此被筛选器“隐藏”。

训练时间建议：

- 当前 U-Net 基线：4090 上先跑 100k–300k updates，通常按 6–18 小时预算。
- 结构化时序编码器 + diffusion：先按 12–36 小时预算。
- 是否停止由 object-disjoint validation、闭环接触率和恢复成功率决定，不以“训练了十几个小时”作为成功标准。
- 记录 `noise_loss`、反归一化后的关节/切向误差、接触模式准确率、滚动部署漂移和每对象接触指标。noise loss 只能衡量扩散去噪目标，不能替代实际动作与闭环指标。

## 4. 数据质量分级

### A 类：动作教师数据

- 对“期望接触集合”要求每指和联合接触率均不低于 99%。当前四指尖任务就是四指联合接触。
- 最长联合失触不超过 5 个原始控制帧。
- 没有可见穿透、爆力、IK 跳变或关节瞬间翻转。
- 初始机械臂准备阶段不纳入统计；只从实际 ready/motion/record 帧开始。
- 轨迹必须产生有意义的切向位移、法向变化或手型变化，近静止轨迹不算多样性。

### B 类：稳定支撑数据

允许短暂单指失触，但至少三指或其他指定支撑区稳定，并在很短时间内恢复。只有在模型显式学习 `contact_mode` 时，B 类数据才能混入动作训练；否则它会把“失触后的姿势”当作普通动作。

### C 类：恢复数据

人工制造小幅位置、法向或摩擦扰动，并保留成功恢复的完整前后文。恢复成功、耗时、峰值力和最终接触集合必须标注。

### D 类：失败数据

用于风险/失败预测器，不直接作为动作教师。包括长期失触、不可恢复过力、严重穿透和 QP 不可行。

建议额外记录以下质量量：

- 每指 loaded-contact、几何 contact、联合 contact 和最长 loss run。
- 每指力的 P50/P95/max；力大本身不必一票否决，但高力伴随穿透或振荡必须拒绝。
- `max |Δq|`、`max |Δqdot|`、高频能量和动作 rate-limit 命中率。
- 表面弧长、法向转角、手掌位移/旋转、各主轴与速度桶覆盖。
- QP slack、关节余量、自碰撞余量、恢复次数和恢复时长。

## 5. 坐标系与 H5 表征

部署输入统一使用 palm frame。这样真机只需手内 FK 和触觉外参，不需要物体世界位姿；物体坐标和解析 oracle 只能作为教师生成或离线评估特权信息。

建议 H5 按 episode 保存，并在根属性中声明版本：

```text
attrs:
  schema_version
  object_id, object_family, source_dataset, source_asset_id, source_license
  control_dt, policy_dt, dp_input_frame="palm"
  teacher_controller, quality_class

observation:
  q_hand[T,16], qdot_hand[T,16]
  palm_twist_palm[T,6]
  contact_pos_palm[T,4,3]
  contact_normal_palm[T,4,3]
  contact_force_palm[T,4,3]          # 可用于 MCC/辅助任务；可从 DP 主输入中剔除
  contact_mask[T,4]
  future_palm_command[T,Hp,6]

teacher_action:
  q_nominal[T,16]
  tip_velocity_relative[T,4,3]
  tangent_basis[T,4,3,2]
  tangent_action[T,4,2]
  posture_velocity[T,16]
  contact_mode[T,4]
  force_band[T,4,2]

diagnostic:
  oracle_point/object_normal/curvature
  qp_slack, joint_margin, collision_margin
  object_pose_world, object_motion
```

注意：`q_nominal` 必须是教师的 nominal/pre-shape，不是已经混入高频接触补偿后的 `q_cmd`。否则 DP 会把控制器的瞬时纠错也当成未来策略。

切平面基必须做时间连续化：新一帧的 `t1/t2` 与上一帧选择最接近的符号和方向，否则法向几乎不变时，切向标签也可能突然翻转。

## 6. 上下层接口

上层规划器给 DP：

```text
future_palm_twist[Hp,6]
exploration_direction_palm[3]
desired_distance
speed_limit
risk_level
optional target_region
```

DP/QP 回传上层：

```text
feasible
joint_margin
qp_slack
support_score
contact_loss_risk
policy_uncertainty
recommended_speed_scale
stop_or_replan
```

DP 不负责机械臂全局可达域；QP 或外部 planner 可以拒绝不可达动作，并让上层减速或重规划。

## 7. 当前仓库的端到端执行流程

以下流程先跑现有方案 A。所有命令均从仓库根目录执行。

### Step 0：环境与静态检查

```bash
cd ~/Code/Hand_Compliance_Control
conda activate mjlab

python -m py_compile \
  mcc_finger_compliance_control/scripts/collect_trajectories.py \
  mcc_finger_compliance_control/scripts/invert_trajectories.py \
  mcc_finger_compliance_control/scripts/export_palm_dp.py \
  mcc_finger_compliance_control/scripts/train_dp.py

python -m unittest discover \
  -s mcc_finger_compliance_control/tests \
  -p 'test_*.py'
```

验收：对象配置和 surface oracle 测试通过。

### Step 1：可视化确认单个对象

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --viewer native \
  --device cuda:0 \
  --num-envs 1 \
  --object-id capsule_medium \
  --teacher-controller fullhand_mcc \
  --trajectory-length 2500 \
  --motion-start 1000 \
  --record-start-step 1000 \
  --max-prep-wait-steps 1000 \
  --motion-length 1400 \
  --motion-mode rotation \
  --initial-orientation-mode fixed
```

若 native/GLX 不稳定，将 `--viewer native` 改为 `--viewer viser`。验收：准备阶段结束后四指进入接触；运动过程中没有明显穿透、抖动和手型跳变。

### Step 2：小批量多对象 baseline

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/batch_collect.py \
  --objects capsule_medium ellipsoid_medium sphere_medium rounded_box_medium \
  --motion-modes rotation translation combined \
  --device cuda:0 \
  --num-envs 4 \
  --max-trajectories 8 \
  --trajectory-length 2500 \
  --motion-start 1000 \
  --record-start-step 1000 \
  --max-prep-wait-steps 1000 \
  --motion-length 1400 \
  --initial-orientation-mode fixed \
  --axis-sampling stratified \
  --seed 20260812
```

验收：每个 `object × motion_mode × axis` 至少有一条成功轨迹。失败组合先降低速度/距离或拆成较短的 move-hold 段，不要用同一默认动作范围硬套所有物体。

### Step 3：分析与严格筛选

```bash
python mcc_finger_compliance_control/scripts/analyze_quality.py \
  mcc_finger_compliance_control/data/trajectories/<raw>.h5 \
  --quality-profile strict99 \
  --report mcc_finger_compliance_control/data/trajectories/<raw>_quality.csv

python mcc_finger_compliance_control/scripts/filter_trajectories.py \
  mcc_finger_compliance_control/data/trajectories/<raw>.h5 \
  --min-all4-ratio 0.99 \
  --min-per-tip-ratio 0.99 \
  --max-loss-run 5 \
  --output mcc_finger_compliance_control/data/trajectories/<raw>_strict99.h5
```

验收：同时检查接触率、最长失触、关节跳变和有效运动量。不能只凭四指接触率放行近静止轨迹。

### Step 4：反演并做纯教师 replay

```bash
python mcc_finger_compliance_control/scripts/invert_trajectories.py \
  --file mcc_finger_compliance_control/data/trajectories/<raw>_strict99.h5 \
  --output mcc_finger_compliance_control/data/inverted/<name>_inverted.h5

python mcc_finger_compliance_control/scripts/replay_inverted.py \
  --file mcc_finger_compliance_control/data/inverted/<name>_inverted.h5 \
  --episode-id 0 \
  --viewer native \
  --mode teacher \
  --device cuda:0 \
  --contact-threshold 0.05
```

验收：反演只改变坐标系，不改变手与物体的相对几何；teacher replay 的位置和接触应接近原始采集。

**当前硬阻塞项**：`invert_trajectories.py` 中的曲率仍使用固定胶囊半径/半长解析函数。它会复制采集到的真实接触法向，但 `fingertip_curvature_object` 对非胶囊是错误的。多物体 DP 训练前必须改为：按 `object_id` 调用通用 mesh/primitive surface query，或暂时完全不导出曲率。

### Step 5：导出 palm-frame DP 数据

```bash
python mcc_finger_compliance_control/scripts/export_palm_dp.py \
  --file mcc_finger_compliance_control/data/inverted/<name>_inverted.h5 \
  --output mcc_finger_compliance_control/data/inverted/<name>_palm_geometry.h5 \
  --state-schema contact_geometry
```

验收：属性包含 `dp_input_frame=palm`、`dp_state_schema=contact_geometry`；所有 contact point/normal 和 palm twist 都在 palm frame。

### Step 6：训练现有 U-Net 基线

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python mcc_finger_compliance_control/scripts/train_dp.py \
  --file mcc_finger_compliance_control/data/inverted/<name>_palm_geometry.h5 \
  --output mcc_finger_compliance_control/data/models/<run_name> \
  --device cuda:0 \
  --steps 100000 \
  --batch-size 256 \
  --stride 5 \
  --obs-horizon 16 \
  --pred-horizon 32 \
  --action-representation absolute_q \
  --diffusion-steps 100 \
  --inference-steps 10 \
  --noise-scheduler DDPM \
  --eval-every 1000 \
  --save-every 10000
```

验收顺序：单轨迹 overfit → object-disjoint validation → `teacher_dp` → `live_dp`。训练/验证 episode 不能只是同一物体同一姿态的不同随机种子。

### Step 7：部署评估

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file mcc_finger_compliance_control/data/inverted/<name>_palm_geometry.h5 \
  --model mcc_finger_compliance_control/data/models/<run_name>/best.pt \
  --episode-id 0 \
  --mode live_dp \
  --viewer native \
  --device cuda:0 \
  --inference-steps 10 \
  --chunk-execution \
  --dp-replan-interval 20 \
  --finger-impedance \
  --contact-threshold 0.05 \
  --report mcc_finger_compliance_control/data/models/<run_name>/live_ep0.csv
```

验收：先比较 teacher replay、teacher-history DP 和 live DP，定位误差来自模型、动态执行还是接触历史偏移。目标推理预算是单次重规划低于 100–200 ms；QP/MCC 必须继续高频运行，不能等待 diffusion。

## 8. 方案 B 的实施门槛

按以下顺序实现，不应直接重写整个部署器：

1. **标签导出**：补 `qdot_hand`、未来 palm command、时间连续切向基、`u_tan`、`q_posture` 和 contact mode。
2. **教师信号 replay**：完全不用 DP，把教师 `u_tan` 输入 QP→MCC；接触质量必须接近原始教师。
3. **QP 验收**：记录 slack、不可行率、关节余量和自碰撞余量；不允许用大 slack 掩盖错误法向或错误 Jacobian。
4. **DP 只替换 `u_tan`**：其余姿态和模式仍用教师，隔离切向预测质量。
5. **逐头替换**：再加入 posture、mode、force band 和未来 normal 辅助头。
6. **闭环数据聚合**：收集 policy 偏离后的成功恢复数据，不能只增加同分布 teacher 轨迹。

## 9. 可用于 MuJoCo 数采的物体数据集

来源清单的机器可读版本见 `configs/datasets/object_sources.yaml`。数据本体不要提交 Git；仓库只保存来源、许可证、筛选清单、转换参数和每个资产的 YAML。

### 第一优先级

| 数据集 | 规模/格式 | 当前项目价值 | 接入成本 |
|---|---|---|---|
| Google Scanned Objects 的 MuJoCo 转换 | 1,030 个日用品；OBJ、纹理、MJCF、V-HACD 碰撞块 | 最适合第一批真实物体；已有 MuJoCo 碰撞模型 | 低；需重新核对尺寸、接触参数和模型质量 |
| YCB Object and Model Set | 约 77 个标准操作物体；多分辨率 mesh/纹理 | 可买到同款实物，适合 sim-to-real 和标准 benchmark | 中；需生成稳定 collision decomposition |
| EGAD! | 2,282 train + 49 eval OBJ | 几何复杂度和抓取难度分层清楚，适合曲率课程 | 低到中；需要按手大小缩放、补质量/惯量 |

推荐先从 GSO 中人工筛 30–50 个可单手包裹、无细碎悬空部件、尺度可信的物体，再加入 YCB 10–20 个和 EGAD 49 个 eval 形状。不要一开始导入全部一千个资产。

### 第二优先级

| 数据集 | 规模/格式 | 用途与限制 |
|---|---|---|
| OmniObject3D | 6,000 个真实扫描、190 类，含 textured mesh | 类别丰富；需遵守下载平台条款并批量修网格、缩面 |
| Amazon Berkeley Objects | 7,953 个高质量 glTF 2.0 模型 | 日用品形态丰富且有尺寸/材质元数据；CC BY-NC 4.0，需 GLB→OBJ/MJCF 和类别筛选 |
| ContactDB | 50 个日用品、3,750 个带人类接触图的 mesh | 不是主规模来源，但很适合评估接触区域先验和全手触觉表征；许可为自定义条款 |

### 第三优先级

| 数据集 | 用途与风险 |
|---|---|
| Objaverse / Objaverse-XL | 800k / 10M+，适合后期扩大长尾；尺度、拓扑和物理质量不统一，单物体许可证不同。只下载经过类别、许可证、尺度和网格质量过滤的 manifest 子集。 |
| ShapeNetCore / PartNet / PartNet-Mobility | 适合类别级 CAD 形状和带把手/门盖的关节物体；真实表面细节有限，且需注册或遵守研究条款。关节物体应在静态物体 pipeline 稳定后再接。 |

### 下载与许可原则

1. 优先官方源。Hugging Face 镜像仅作为下载便利，不自动继承再分发权。
2. 每个资产保存 `source_dataset/source_asset_id/source_url/license/attribution`。
3. Objaverse 必须逐资产检查许可证；不能只依据整个集合的 ODC-By 声明。
4. ABO 是 CC BY-NC 4.0，不应混入未来商业部署资产而不做隔离。
5. ContactDB、OmniObject3D、ShapeNet/PartNet 在批量使用前重新阅读并保存当时版本的条款。

## 10. 外部 mesh 的标准接入流程

当前 `object_catalog.py` 只完整支持 primitive 和运行时生成的 rounded box。导入外部 OBJ 前需增加通用 `type: mesh`：

```yaml
geoms:
  - type: mesh
    visual_mesh: assets_external/gso/<asset_id>/model.obj
    collision_meshes_glob: assets_external/gso/<asset_id>/model_collision_*.obj
    scale: [1.0, 1.0, 1.0]
    pos: [0.0, 0.0, 0.0]
    quat: [1.0, 0.0, 0.0, 0.0]
```

每个资产执行：

1. **许可证登记**：生成 manifest，保存原始哈希和来源。
2. **单位归一**：统一为米，保存原始尺度和变换，禁止靠视觉猜尺度。
3. **坐标归一**：移动到合理质心，但保存 `T_source_to_canonical`，不要不可逆覆盖原始 mesh。
4. **网格检查**：有限顶点、法向、连通分量、重复面、自交、孔洞和退化三角形。
5. **视觉简化**：通常保留 20k–80k faces；纹理和碰撞分离。
6. **碰撞分解**：生成 8–32 个凸块作为起点；过多凸块会显著降低并行采集速度。
7. **质量/惯量**：按体积和材料范围设置；mocap 教师环境仍要避免不合理接触响应。
8. **surface query**：oracle 必须查询与碰撞模型一致的表面，并输出最近点、法向、距离和置信度。
9. **单资产 gallery**：检查尺度、初始姿态、法向和碰撞，不直接批量数采。
10. **8 条 pilot**：rotation/translation 分开，按主轴分层抽样；质量合格后才进入大批量。

建议目录：

```text
assets_external/                  # .gitignore，不提交大文件
  gso/<asset_id>/original/
  gso/<asset_id>/processed/
mcc_finger_compliance_control/
  configs/datasets/object_sources.yaml
  configs/objects/gso_<asset_id>.yaml
  manifests/gso_subset_v1.csv     # 来源、许可、hash、scale、split
```

## 11. 数据划分

至少维护四个互斥 split：

- `train_object`：训练物体和姿态。
- `val_unseen_instance`：已见类别、未见实例。
- `test_unseen_category`：未见类别或显著不同拓扑。
- `test_sim2real`：有对应实物、绝不参与训练调参的 YCB/GSO 子集。

同一个 mesh 的不同姿态、缩放或运动轴不能分别放入 train/validation；那只能验证轨迹插值，不能验证物体泛化。相似模型和派生 collision mesh 也必须按源资产 ID 成组划分。

## 12. 当前最近的三个动作

1. 给 `object_catalog.py`、`GeometrySurfaceOracle` 和 gallery 增加真正的外部 mesh 配置支持。
2. 从 GSO MuJoCo 集合挑 10 个手持物体做下载、转换、可视化和 8 条/物体 pilot，而不是立刻下载全部数据集。
3. 同时修复多物体反演中的固定胶囊曲率，随后再开始方案 B 的切向标签导出。

完成这三项后，才进入 30–50 个 GSO + 10–20 个 YCB + 49 个 EGAD eval 的第一版多物体训练集。
