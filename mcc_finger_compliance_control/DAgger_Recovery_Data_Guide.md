# DAgger 恢复数据采集指南

本文只说明如何采集并训练“DP 失触后由专家恢复”的数据。

## 1. 数据采集目标

每条恢复数据由两个片段组成：

~~~text
片段 A：DP 闭环运行
稳定抓握 → 逐渐漂移 → 持续失触 → 保存失触状态 A

片段 B：专家恢复
从状态 A 重新初始化 → 椭圆规划器重新规划 → 数采控制器执行恢复
→ 四指重新接触 → 保持一段稳定尾帧

最终：A + B 合并为一个 recovery episode
~~~

两个片段可以来自两次仿真，不要求使用同一个 simulator instance。但片段 B 必须从片段 A 的失触状态 A 初始化，拼接前必须检查状态连续性。

正式恢复采集统一使用：

~~~text
DP history q = live q
tactile      = 实时 solver contact 锚定、原始表面修正后的 tactile
DP label     = 专家 q_ref 或 expert tip target
~~~

DP 不学习 MCC 补偿。`q_cmd`、`delta_q_comp` 和 `e_servo` 只用于诊断或作为独立执行上下文，不能作为动作标签。

这里的 tactile 不是凸分解碰撞面的原始几何，也不是脱离物理接触独立查询出来的 SDF
“伪接触”。各通道来源固定为：

| 通道 | 来源 |
|---|---|
| contact mask / 接触是否存在 | 当前帧 MuJoCo solver 的真实 fingertip contact |
| 3D force | 当前帧 MuJoCo contact force |
| contact point | solver contact 锚点；有可靠 correspondence 时再局部映射到原始表面 |
| contact normal | 仅在真实 contact 锚点处查询未凸分解 source mesh/SDF，并按传感器约定定向 |

因此它仍然是 **live observation**，只是几何法向不采用凸分解 hull 的面法向。当前
`--dp-tactile-normal-source source_mesh_oracle` 实现的正是“actual-contact-anchored normal”：
它不能凭空创建接触，也不能在失触后向 DP 暴露物体距离或未来表面。

当前实现只修正 normal，contact point 仍来自凸分解 collision。若 convex contact 到原始
表面的映射距离过大或映射前后法向差异过大，该帧应降低 confidence 或从训练窗口中剔除，
不能把它当成高质量 tactile。后续可再补充 contact-point lifting 与一致性 gate，但不能
因此退回 recorded teacher tactile；recorded tactile 与当前 `q_live` 不同步，会产生更严重的
状态语义错误。

## 2. 完整流程

### 步骤一：运行 DP，采集失触历史

不要直接开始批量采集。先用 `deploy_dp_inverse.py` 对一条 episode 做可视化闭环部署，
确认 DP 的确经历“稳定接触 → 逐渐漂移 → 持续失触”，而不是初始化穿模、错误法向、
MCC 参数不一致或接触检测抖动造成的假失败。

#### 1.1 单条可视化闭环测试

~~~bash
cd /home/rimlab/Code/Hand_Compliance_Control
conda activate mjlab

MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file mcc_finger_compliance_control/data/inverted/mustard_v1_239_mesh_normal_inward_inverted.h5 \
  --model mcc_finger_compliance_control/data/models/mustard_v1_239_motion96_kinematic_residual_pred8_25k/best.pt \
  --episode-id 24 \
  --mode live_dp \
  --viewer native \
  --device cuda:0 \
  --execution-layer fullhand_mcc \
  --mcc-preset collection_matched_sensor \
  --dp-history-q-source live \
  --dp-tactile-normal-source source_mesh_oracle \
  --mcc-direction-source hybrid \
  --contact-threshold 0.05 \
  --chunk-execution \
  --dp-replan-interval 10 \
  --inference-steps 100 \
  --dp-samples 1 \
  --seed 20260831 \
  --max-steps 2500 \
  --highlight-contacts \
  --report /tmp/recovery_ep024_visual.csv
~~~

先观察并确认：

- 初始四指能够正常建立接触；
- 接触点高亮与画面中的真实接触一致；
- 失触发生在 DP 接管之后；
- 失触前存在逐步漂移，而不是单帧瞬移；
- MCC 没有产生异常振荡、持续穿透或巨大力峰值。

#### 1.2 保存一条完整失触 rollout

确认单条现象正确后，保持所有控制参数不变，只改成 headless 并保存 H5：

~~~bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
  --file mcc_finger_compliance_control/data/inverted/mustard_v1_239_mesh_normal_inward_inverted.h5 \
  --model mcc_finger_compliance_control/data/models/mustard_v1_239_motion96_kinematic_residual_pred8_25k/best.pt \
  --episode-id 24 \
  --mode live_dp \
  --viewer headless \
  --device cuda:0 \
  --execution-layer fullhand_mcc \
  --mcc-preset collection_matched_sensor \
  --dp-history-q-source live \
  --dp-tactile-normal-source source_mesh_oracle \
  --mcc-direction-source hybrid \
  --contact-threshold 0.05 \
  --chunk-execution \
  --dp-replan-interval 10 \
  --inference-steps 100 \
  --dp-samples 1 \
  --seed 20260831 \
  --max-steps 2500 \
  --rollout-h5 mcc_finger_compliance_control/data/closed_loop_rollouts/recovery_round1/ep024.h5 \
  --report mcc_finger_compliance_control/data/closed_loop_rollouts/recovery_round1/ep024.csv
~~~

检查这条 H5 的帧数、live-q、触觉和控制字段完整后，再进入批量阶段。

#### 1.3 批量扩展到更多失败 episode

`collect_dagger_rollouts.py` 是上述单条命令的批量包装器：

~~~bash
cd /home/rimlab/Code/Hand_Compliance_Control
conda activate mjlab

MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/collect_dagger_rollouts.py \
  --file mcc_finger_compliance_control/data/inverted/mustard_v1_239_mesh_normal_inward_inverted.h5 \
  --model mcc_finger_compliance_control/data/models/mustard_v1_239_motion96_kinematic_residual_pred8_25k/best.pt \
  --episodes 24 30 40 45 51 53 57 59 \
  --output-dir mcc_finger_compliance_control/data/closed_loop_rollouts/recovery_round1 \
  --max-steps 2500 \
  --device cuda:0 \
  --seed 20260831 \
  --inference-steps 100 \
  --dp-replan-interval 10 \
  --dp-samples 1 \
  --dp-history-q-source live \
  --dp-tactile-normal-source source_mesh_oracle
~~~

批量脚本的模型、MCC、history source、法向来源、推理步数和 replan interval 必须与
已经验证的单条部署完全一致。批量阶段只扩大 failure-state 覆盖，不能同时改控制配置。

每条 rollout 至少保存：

- `q_live`、`qvel_live`；
- 实时 solver contact 的点、mask 和力，以及在该 contact 锚点查询的 source-surface 法向；
- palm/object pose 与 twist；
- planner command/phase；
- DP observation、DP reference、`q_cmd`、`delta_q_comp` 和 `e_servo`；
- 失触开始帧和最终选定的 `failure_frame`。

推荐将“有效接触指少于 3 根并持续 20–30 帧”作为第一版失触触发条件。短暂的 1–3 帧接触抖动不应触发专家。

### 步骤二：从失触状态 A 规划恢复轨迹

从 rollout 的 `failure_frame` 提取状态 A：

~~~text
q/qvel
palm pose/twist
object pose/twist
当前 planner 方向与剩余任务
实时指尖接触状态
~~~

专家不是从 A 任意生成一条新的椭圆，而是**沿失触前的原 planner 轨迹继续规划**。
状态 A 除了提供当前位姿，还要提供原轨迹的 phase、运动方向、切向速度和角速度；恢复
规划器在 A 附近对原路径做局部续接/修正，使新轨迹先顺着原运动趋势继续，再逐渐进入
可恢复四指接触的局部椭圆/运动。它也不是读取原 teacher 数据中相同时间戳的 q。

续接边界至少满足：

~~~text
p_new(0)     = p_A
R_new(0)     = R_A
v_new(0)     ≈ v_original(A)
omega_new(0) ≈ omega_original(A)
~~~

即 palm position/orientation 严格连续，线速度和角速度至少一阶连续。接管后的专家
手型参考也必须从当前 `q_live(A)` 平滑开始，再逐渐收敛到优化出的稳定抓握目标，不能在
专家第一帧直接跳到新关键点 q。

实现时建议在 A 后设置约 50–100 个控制帧的五次多项式或 B-spline 过渡段：以原轨迹
在 A 的 pose/twist 为起始边界，以恢复椭圆的首个稳定段为终止边界。不要先把速度清零，
再突然启动一条新轨迹。

恢复规划不是由单个脚本完成，而是两级规划：

1. `generate_manifold_palm_plan.py` 负责 **palm path**：沿原 planner 在 A 点的
   pose/twist 续接局部椭圆轨迹；
2. `optimize_contact_plan.py` 负责 **finger keyframes**：在已经生成的 palm path 上
   离线求稀疏四指接触点和健康抓握姿势。它不负责生成椭圆轨迹；
3. `collect_trajectories.py` 中的接触流形 QP 在相邻关键帧之间生成逐帧手指运动，
   FullHandMCC 再负责实际追踪和接触力调节；
4. `view_inverse_palm_plans.py` 用于可视化检查 palm path 和抓握关键帧。

因此正确调用顺序应当是：

~~~text
state A
  → generate_manifold_palm_plan.py       生成并平滑续接 palm trajectory
  → optimize_contact_plan.py             求稀疏 finger/contact keyframes
  → collect_trajectories.py              QP 插值 + FullHandMCC 执行恢复
~~~

当前缺少一个适配器，把 rollout 的任意 `failure_frame` 转换为轨迹生成器可直接使用的起点。该接口未完成前，不能把原 episode 第一帧冒充状态 A。建议接口：

~~~text
extract_recovery_state.py
  --rollout <rollout.h5>
  --frame <failure_frame>
  --output <state_A.h5>

generate_manifold_palm_plan.py
  --recovery-state <state_A.h5>
  --output <from_A.h5>

optimize_contact_plan.py
  --input <from_A.h5>
  --output <from_A_opt.h5>
  --recovery-state <state_A.h5>
~~~

以上 `--recovery-state` 目前还没有实现。它对两个规划器的作用不同：

- 对 palm planner：提供 A 点的原路径 phase、pose 和 twist，生成连续的椭圆续接；
- 对 contact optimizer：用 `q_live(A)` 和当前四指接触状态 warm-start 第一关键帧，
  后续关键帧再逐步转入优化出的稳定抓握分支。

所以不能只运行现有的：

~~~bash
python mcc_finger_compliance_control/scripts/optimize_contact_plan.py \
  --input from_A.h5 --output from_A_opt.h5 --object-id ycb_mustard
~~~

这条现有命令适合对一条完整的离线 palm plan 做常规抓握筛选，但不包含状态 A 的
恢复边界，也不能单独产生需要的 recovery plan。

恢复轨迹应满足：

- 起点与状态 A 的 palm/object 相对位姿一致；
- 延续原 planner 的 phase、切向、速度和任务方向，不能重新随机选择椭圆方向；
- 接管边界的 palm pose 连续，linear/angular velocity 无明显突变；
- 专家 q reference 从 `q_live(A)` 平滑过渡到恢复目标；
- 手掌轨迹平滑并满足安全距离；
- 关键点四指可达且手型健康；
- 长度足以完成接触恢复，并保留至少一个 DP prediction horizon 的稳定尾段。

### 步骤三：数采控制器执行专家恢复

将专家仿真重置到状态 A，然后让原数据采集逻辑执行 `from_A_opt.h5`：

~~~text
状态 A
→ 必要的控制器 warm-up
→ FullHandMCC/数采接触控制器执行新规划
→ 搜索并恢复接触
→ 四指稳定接触 50–100 帧
→ 保存完整恢复片段
~~~

这里使用 `collect_trajectories.py` 中的教师控制逻辑。专家可以使用仿真特权几何和 surface oracle，因为这些信息只负责生成教师标签；最终 policy observation 不得包含未授权特权信息。

恢复片段必须保存专家任务参考，例如 `q_ref_expert` 或 `tip_x_des_palm`。不要把 `q_live`、`q_cmd` 或 MCC compensation 当作专家动作。

当前 `collect_trajectories.py` 尚未提供完整的任意状态注入接口。需要补充以下调用契约：

~~~text
collect_trajectories.py
  --motion-mode planner_inverse
  --planner-file <from_A_opt.h5>
  --init-h5 <state_A.h5>
  --record-controller-source
  --filename <recovery_name>
~~~

`--init-h5` 至少应恢复 q、qvel、palm/object pose、planner phase。无法恢复的 MCC 滤波/积分状态应清零，并把 warm-up 区间显式记录下来。

### 步骤四：拼接并生成训练窗口

将 DP 片段和专家恢复片段按状态 A 合并：

~~~text
[DP stable → drift → loss at A]
                  +
[A → expert takeover → recovery → stable]
                  ↓
          one recovery episode
~~~

拼接工具建议独立实现为 `build_recovery_dataset.py`：

~~~text
build_recovery_dataset.py
  --rollout <DP rollout.h5>
  --failure-frame <A>
  --recovery <expert recovery.h5>
  --reference-dp <clean training.h5>
  --output <recovery_round1.h5>
~~~

拼接前必须检查：

- A 两侧的 q、qvel 跳变量；
- palm/object 相对位姿误差；
- 控制频率和字段时序；
- tactile/contact 字段坐标系、contact mask 与 source-surface normal 的锚定关系；
- planner phase 和运动方向。
- 新旧 palm trajectory 在 A 处的位置、姿态、线速度和角速度连续性；

超过阈值的片段直接拒绝，不用插值掩盖不连续。

训练切窗规则：

- DP 失触段可以进入 observation history；
- warm-up 段可以进入 history，但没有有效 expert label；
- prediction horizon 必须全部落在有效专家控制段；
- 失败 DP action 不作为 target；
- target 只取专家 `q_ref` 或 expert tip target；
- 保存 `controller_source`、`expert_label_valid`、`failure_frame`、`expert_start_frame` 和 `recovery_confirm_frame`。

## 3. 与 clean 数据混合训练

不要覆盖原 clean H5。通过 `--dagger-file` 混合：

~~~bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/train_dp.py \
  --file mcc_finger_compliance_control/data/inverted/mustard_v1_239_motion96_kinematic_palm_dp.h5 \
  --dagger-file mcc_finger_compliance_control/data/dp/mustard_recovery_round1.h5 \
  --dagger-sample-ratio 0.10 \
  --resume mcc_finger_compliance_control/data/models/mustard_v1_239_motion96_kinematic_residual_pred8_25k/best.pt \
  --output mcc_finger_compliance_control/data/models/mustard_recovery_round1 \
  --device cuda:0 \
  --steps 5000 \
  --batch-size 256 \
  --lr 1e-4 \
  --stride 5 \
  --obs-horizon 16 \
  --pred-horizon 8 \
  --action-representation kinematic_residual_q \
  --diffusion-steps 100 \
  --inference-steps 50 \
  --seed 20260829
~~~

第一轮建议比较 `--dagger-sample-ratio 0.05 / 0.10 / 0.20`。clean validation 和 recovery validation 必须按 episode 分组，不能把同一条恢复轨迹的相邻窗口同时放入训练集和验证集。

## 4. 数据质量要求

一条恢复 episode 合格需要同时满足：

- 失触前包含足够的稳定抓握与漂移历史；
- 状态 A 是真实持续失触，不是单帧检测抖动；
- 专家确实从 A 重新规划，而不是复制同时间 teacher q；
- 恢复轨迹顺着原 planner 续接，接管处没有 palm pose、twist 或 q reference 跳变；
- 恢复段包含接管、搜索、恢复和稳定尾段；
- 恢复后四指接触率建议不低于 98%；
- 手型健康、无持续穿透和异常力峰值；
- 拼接边界状态连续；
- contact-anchored live observation 与 expert target 时序严格对齐；
- 所有 DP 几何输入使用手掌坐标系。

第一轮先采 20–40 条，覆盖拇指失触、其他单指失触、双指失触、切向滑移和手型退化。验证有效后，再用新 policy 重新 rollout 并采下一轮，而不是一直使用旧 checkpoint 生成所有恢复数据。

## 5. 各脚本的具体职责

### `deploy_dp_inverse.py`

单条部署实验的实际执行入口。它加载一个 inverse H5 和 DP checkpoint，让上层 palm
沿给定轨迹运动，DP 生成手指任务参考，FullHandMCC 执行参考并维持接触。

- 输入：inverse H5、checkpoint、episode id 和部署参数；
- 输出：逐帧 CSV 和可选 rollout H5；
- 用于：观察某条轨迹如何从稳定状态漂移到失触，并记录真实 `q_live`、tactile、DP
  reference 和 MCC 执行量；
- 不负责：重新规划恢复轨迹或生成专家恢复标签。

### `collect_dagger_rollouts.py`

`deploy_dp_inverse.py` 的批量前台包装器。它依次运行多个 episode，并检查输出是否完整，
适合批量寻找失败状态 A。

- 输入：inverse H5、checkpoint 和 episode 列表；
- 输出：每个 episode 一组 rollout H5/CSV；
- 正式恢复采集必须使用 `--dp-history-q-source live`；
- 不负责：筛选最终 failure frame、运行专家或拼接恢复数据。

### `generate_manifold_palm_plan.py`

只负责规划 **palm trajectory**。它根据物体原始几何生成保持安全距离的椭圆/流形
手掌位姿轨迹，输出逐帧 `palm_pose_object` 和 `palm_twist_object`。

- 输入：物体、椭圆平面/覆盖范围、速度参数化和初始 palm 条件；
- 输出：palm-only planner H5；
- 用于恢复时：沿原 planner 在状态 A 的 phase、pose 和 twist 继续规划；
- 不负责：求四指关节姿势、执行动力学或维持接触力；
- 当前限制：还不能直接读取 rollout 的任意 failure state A。

### `optimize_contact_plan.py`

在已经生成的 palm trajectory 上，离线求解稀疏四指抓握关键帧。优化考虑原始 mesh
表面接触、指腹方向、手指可达性、manipulability、关节范围、手型和指间关系。

- 输入：`generate_manifold_palm_plan.py` 生成的 planner H5；
- 输出：在原 H5 内容上增加 `grasp_keyframe_q`、接触点、法向、质量指标和 valid mask；
- 关键帧 q 是接触流形控制的边界/姿态参考，不是要求每帧直接追踪的完整轨迹；
- 不负责：生成椭圆 palm path、执行 MuJoCo 或进行实时力控制；
- 当前限制：恢复场景还需用 `q_live(A)` warm-start 第一关键帧。

### `view_inverse_palm_plans.py`

纯几何可视化工具。它把物体固定在 inverse 空间，用不同颜色显示
`palm_pose_object` 轨迹，并可显示每条路径起点处的手模型和初始 q。

- 用于：检查轨迹位置、朝向、覆盖区域和初始手型是否合理；
- 不逐帧播放 `grasp_keyframe_q`，因此不能靠它检查所有关键帧手型；
- 不运行 physics、MCC 或接触率测试，因此可视化通过不代表物理采集合格。

### `collect_trajectories.py`

专家物理执行与原始数据记录入口。`planner_inverse` 模式下，它反演 palm plan 使物体
相对固定手运动，并使用预计算抓握关键帧、接触流形 QP 和 FullHandMCC 生成逐帧教师动作。

- 输入：优化后的 planner H5、物体/环境参数和采集参数；
- 输出：包含 q、专家 reference、接触、力、pose 和控制诊断的 raw H5；
- 接触流形 QP：连接相邻稀疏抓握关键帧；
- FullHandMCC：追踪任务参考并进行法向接触补偿；
- 用于恢复时：从状态 A 执行新规划，记录专家接管到恢复稳定的全过程；
- 当前限制：尚缺完整的 `--init-h5 <state_A.h5>` 状态注入接口。

### `invert_trajectories.py`

把“手固定、物体运动”的 raw 数采结果转换为“物体固定、手沿表面运动”的等价轨迹，
并处理 palm/object 相对 pose、接触点和法向的坐标变换。

- 用于：将专家恢复 raw H5 转成 DP 所需的任务视角；
- 注意：恢复段和 DP 失触段必须采用同一坐标约定和同一帧时序后才能拼接。

### `export_palm_dp.py` / `export_dual_track_v3.py` / `export_task_tip_motion_v4.py`

这些是不同模型版本的数据导出器，不应混用：

- `export_palm_dp.py`：导出旧 palm-frame q-policy 数据；
- `export_dual_track_v3.py`：导出包含 task prior 与 live execution 的 242D 双轨状态；
- `export_task_tip_motion_v4.py`：在双轨状态基础上导出 V4 absolute fingertip target。

恢复数据必须选择与待微调 checkpoint 完全相同的 exporter、schema、action representation、
normalization 和 horizon。

### `build_dagger_dataset.py` / `relabel_dagger_phase_aligned.py`

这两个脚本属于旧的局部 teacher relabel 实验：前者用 rollout 构建 96D DAgger 样本，
后者在原 teacher 轨迹附近进行 phase 对齐。它们不会调用椭圆规划器，也不会产生真实的
专家恢复过程，因此不能替代本 Guide 的 recovery builder。

### `train_dp.py`

训练和微调入口。clean 数据由 `--file` 提供，恢复数据由 `--dagger-file` 提供，
`--dagger-sample-ratio` 控制每个 batch 中恢复样本比例。它不负责把 raw trajectory 自动
转换成训练 schema。

### 尚需补充的两个工具

- `extract_recovery_state.py`：从 DP rollout 的 failure frame 导出完整状态 A；
- `build_recovery_dataset.py`：审计拼接边界，把 DP 失触段与专家恢复段合并，并按
  `expert_label_valid` 生成训练窗口。

在状态 A 注入和 recovery builder 完成之前，可以分别运行 DP rollout、palm planner、
contact optimizer 和专家采集，但还不能把它们可靠地串成最终恢复训练 H5。
