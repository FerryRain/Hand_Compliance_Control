# DAgger 式恢复数据采集 Guide

> 目标读者:需要跑恢复数据采集的合作者。
> 一句话:让当前还不够好的 DP 在闭环部署中自己"找出"会失触的时刻,我们在失触现场用专家(数采控制器)从同一个状态重新抓一次、恢复接触,把这整段"失触 → 恢复"过程变成训练数据,教会 DP 下次自己回来。

## 0. 为什么需要这份数据

现在的训练数据全部来自**专家成功轨迹**。DP 在闭环中一旦漂移、手指脱开瓶身,它进入了自己从未见过的状态分布(接触 mask 全空、几何信息冻结),而训练数据里没有"从这种状态回到接触"的例子——于是越漂越远。

DAgger 的思路:policy 自己跑,在**即将/已经失触**的地方停下来,把现场状态记下来;然后让专家从**同一个现场状态**重新做一次成功的恢复;把"失败前状态 + 专家恢复过程"加入训练集。

## 1. 总体流程(四步)

```text
第1步  部署找失败        闭环跑 DP → 失触达阈值 → 暂停并保存现场 = 状态 A
                                        │
                                        ▼
第2步  规划恢复目标      从状态 A 重新规划一小段椭圆轨迹(起点 = A 的手掌位姿);
                        状态 B = 这段轨迹上第一个四指全部 valid 的关键节点(稳定抓握手型)
                                        │
                                        ▼
第3步  专家恢复采集      采集环境重置到状态 A;数采控制器追踪这段新轨迹,
                        手指从失触状态追回接触,到达下一关键点上的稳定抓握状态 B
                                        │
                                        ▼
第4步  拼接与建集        失触段 + 恢复段拼成完整 episode → 与主训练集同格式 → 混训
```

一次完整恢复数据 = **失触段(第1步产出,只有真实状态)** + **恢复段(第3步产出,状态 + 专家目标)**。

## 2. 第 1 步:部署中找失触 → 状态 A

**做什么**:用当前 DP checkpoint 在一条专家采集轨迹上闭环部署(`deploy_dp_inverse.py`,palm 走轨迹、手指由 DP 预测、FullHandMCC 执行)。一旦检测到"失触":

- **失触定义**:loaded 接触指 `< 3`(力 ≥0.05 N 才算),即同一时刻至少两根手指已脱开;
- **触发阈值**(暂停条件):失触**连续 ≥30 帧(0.3 s)** 或失触期间指尖力缺口持续扩大——先冻结这个默认值,跑几条看触发率再定;
- **触发动作**:停止 replan、记录触发帧(状态 A),把**从 episode 起点到 A 的整段闭环状态**落盘。

**当前状态**:失触计数与 `[TACTILE-HOLD]` 逻辑在 `deploy_dp_inverse.py` 已存在(只挡 replan,不暂停);闭环记录 `--rollout-h5`(含 q、接触 mask、发给 DP 的状态)已可用,且在脚本结束时会整段写盘。

**需要的小改动**:把"失触连续帧数 ≥ 阈值"接成**提前结束闭环并走既有落盘路径**(保证 A 前轨迹不丢),并在输出里记 `splice_frame = A 帧号`。建议加 `--fail-stop-min-contact-fingers 3 --fail-stop-grace-frames 30` 两个参数。

**产物**:每一条失败部署写

- `rollouts/<round>/epXXX.h5` — 失触前闭环状态(`dp_observation_state`、`q_live`、`fingertip_contact_mask`、A 帧号)
- `rollouts/<round>/epXXX.csv` — 逐帧诊断,便于人工确认失触段

**示例命令**:

```bash
cd mcc_finger_compliance_control/scripts

python deploy_dp_inverse.py \
  --file ../data/dp/mustard_randomized_dual_track_v4_220_tip_target.h5 \
  --model ../data/models/mustard_randomized_dual_track_v4_B2_tiptarget_219_15k/best.pt \
  --episode-id 24 --mode live_dp --viewer headless \
  --execution-layer fullhand_mcc \
  --rollout-h5 ../data/closed_loop_rollouts/recovery_round1/ep024.h5 \
  --report   ../data/closed_loop_rollouts/recovery_round1/ep024.csv \
  --fail-stop-min-contact-fingers 3 --fail-stop-grace-frames 30   # [新增参数]
```

（`--file` 是"被部署的专家轨迹"(提供 palm 路径、物体、初始手型);想部署更多条就循环换 `--episode-id`。）

## 3. 第 2 步:规划恢复目标 → 状态 B

**做什么**:从失触段 h5 读出状态 A 那一刻的**手掌位姿 + 物体位姿**(mocap 记录的物体 7-DoF pose、palm 7-DoF pose),以它们为起点**重新规划一小段椭圆轨迹**(覆盖 A 所在瓶身区域、向前延伸一段),再在这段轨迹上解出**关键节点的稳定四指抓握**(接触点/法向/q)。

**状态 B = 这段新轨迹上第一个四指全部 valid 的关键节点**,也就是恢复完成后要到达的稳定抓握手型。

**为什么恢复目标是"追踪轨迹到 B",而不是一个静态手型**:

- 数采专家(`FullHandMCC`)的工作方式是 **palm 沿路径运动,它逐帧从表面几何解出四指该贴在哪**、保持接触并调节法向力——它不接受"给定一个静态手型、手指走过去"这种指令;
- 失触时手指脱开,要找回接触必须依托 palm 路径 + 表面几何作为锚。所以恢复 = 控制器**追踪从 A 出发的新轨迹**,手指在到达关键点 B 之前把接触全部建立回来,在 B 处达成稳定抓握;
- 轨迹从 A 到 B 通常要走一小段(手指重新接近、接触搜索),控制器追踪的过程本身就是要学的"从失触中恢复"。

**用到的规划器**(已存在,产物都是 H5):

| 步骤 | 脚本 | 干什么 | 输入 → 输出 |
|---|---|---|---|
| 1 | `generate_manifold_palm_plan.py` | 生成 palm 椭圆轨道 | 初始 palm/object 位姿 → 含 `palm_pose_object` 的轨道 |
| 2 | `optimize_contact_plan.py` | 在轨道关键节点解四指抓握 | 轨道 → 含 `grasp_keyframe_q / contact_point / normal / valid`(26 节点) |
| 3 | `view_inverse_palm_plans.py` | 可视化检查轨道 + 手型 | `_opt.h5` → MuJoCo 窗口 |

**需要的小改动**:现在 `generate_manifold_palm_plan.py --source` 只取源轨迹**第一帧**做起点。恢复场景需要"从任意中间帧(A)起步",因此:
- 方案 A(推荐):给 `generate_manifold_palm_plan.py` 加 `--start-palm-pose 7 个值 --start-object-pose 7 个值`(直接从 rollout 的 A 帧读数传入);
- 状态 B 取新轨迹上**第一个全部 valid 的关键节点**;轨迹长度建议 600–800 帧,使 B 位于轨迹中前段、从 A 到 B 有足够帧完成"手指重新接近 + 接触搜索",B 之后也留足帧数给训练目标(每样本需未来 16 步 × 5 帧)。

**示例命令**:

```bash
cd mcc_finger_compliance_control/scripts

# 1) 以状态 A 为起点生成一小段椭圆轨道(数值来自 rollout 的 A 帧)
python generate_manifold_palm_plan.py \
  --output ../data/plans/recovery_round1/fromA_s24.h5 \
  --object-id ycb_mustard \
  --frames 800 --angle-deg 55 \
  --path-mode minimum_enclosing_ellipse \
  --ellipse-arc-region calibrated \
  --palm-tangent-sign -1 \
  --start-palm-pose 0.70 0.00 0.80 -3.14 0 0 1 \      # [新增参数]A 帧 palm pose(wxyz 末 4)
  --start-object-pose 0.70 0.00 0.80 1 0 0 0           # [新增参数]A 帧物体 pose

# 2) 解关键节点抓握 → 状态 B 所在
python optimize_contact_plan.py \
  --input ../data/plans/recovery_round1/fromA_s24.h5 \
  --output ../data/plans/recovery_round1/fromA_s24_opt.h5 \
  --object-id ycb_mustard

# 3) 可视化确认轨道贴着瓶身、关键节点手型合理
python view_inverse_palm_plans.py ../data/plans/recovery_round1/fromA_s24_opt.h5
```

## 4. 第 3 步:专家恢复采集

**做什么**:把采集环境**重置到状态 A**(手指 q、物体位姿与 A 帧一致),让数采控制器(`collect_trajectories.py`,特权表面 oracle + MCC)从第 0 帧起**追踪第 2 步新规划的这段椭圆轨迹**;手指从失触状态逐渐追回接触,**到达下一关键点 B 时处于稳定抓握状态**——这就是恢复过程,整段(从 A 到 B 及之后)都作为恢复数据记录下来。

**关键点**:
- 专家能看到真实表面(数采本来就是特权教师),它从 A 的坏姿势**自己找接触**,这正是要学的能力;
- 与主数据集完全同一条采集链 → 输出格式(raw h5、`tip_x_des_palm` 等标签)天然一致,无需特殊转换;
- 初始注入:需要把 rollout 的 A 帧 `q_hand` + `object_pose_world` 传进 reset。**需要的小改动**:给 `collect_trajectories.py` 加 `--init-h5 rollouts/ep024.h5 --init-frame N`(读该帧 q_hand 与物体位姿作初态),其余全部走既有 `planner_inverse` 流程。

**示例命令**:

```bash
cd mcc_finger_compliance_control/scripts

python collect_trajectories.py \
  --device cuda:0 --num-envs 1 --trajectory-length 2500 \
  --motion-mode planner_inverse \
  --planner-file ../data/plans/recovery_round1/fromA_s24_opt.h5 \
  --init-h5 ../data/closed_loop_rollouts/recovery_round1/ep024.h5 --init-frame <A帧号> \  # [新增参数]
  --filename recovery_s24_fromA
```

产物:`data/trajectories/recovery_s24_fromA_*.h5` —— 一条**标准格式**的专家恢复轨迹(建议每条恢复段采集 2–3 条,扰动 seed 略有不同,提高稳健性)。

**人工核对**(打开看或看 CSV):恢复段开头确实从失触状态开始、接触在 ~0.5–2 s 内全部建立、之后沿轨道不失触、质量与主数据集一致(四指接触率 ≥98%)。

## 5. 第 4 步:拼接与建集

**做什么**:把一条恢复数据的**失触段(第1步)** 与 **恢复段(第3步)** 拼成一个连续 episode,导出成与 v4 主训练文件**同 schema** 的训练 H5。

**拼接规则**(核心,务必一致):

- 两段都按 100 Hz(帧长一致),按帧直接连接:失触段帧 `0..N_A` + 恢复段帧 `0..M` → 新 episode 帧 `0..N_A+M`;
- 状态通道(发给 DP 的 242 维因果状态)**逐字段重映射拼接**——失触段来自 rollout 的 `dp_observation_state`,恢复段来自采集 raw 经既有导出链(`invert/export`)得到的同一 schema;
- 专家标签通道(`tip_target_palm` 12D)在**恢复段才有值**,失触段填占位并记录 splice 位置;
- 建集时样本窗口规则:**允许窗口起点上溯到 splice 前若干帧**(让历史覆盖失触帧),但窗口的**未来目标必须全部落在恢复段内**。这样每条拼接 episode 只在交界处与恢复段内产生样本——学到的正是"从失触中恢复"。

**需要的新增**:一个小工具 `build_recovery_samples.py`(拼接 + 按上述规则切窗 + 写 H5),输入 = 失触段 rollout + 恢复段训练格式文件 + splice 帧号,输出 = 与 `--file` 同 schema 的 H5(直接可 `train_dp.py --dagger-file` 或合并进主文件)。

**示例命令**:

```bash
cd mcc_finger_compliance_control/scripts

# 先按主流程把恢复段 raw 转成 v4 同款训练格式(与既有 219 数据同一套导出链)
python export_dual_track_v3.py --file ../data/trajectories/recovery_s24_fromA_*.h5 ...   # 与主数据一致

# 拼接 + 切窗建集
python build_recovery_samples.py \
  --lost-segment ../data/closed_loop_rollouts/recovery_round1/ep024.h5 \   # 第1步产物(状态)
  --recovery-file ../data/dp/recovery_s24_fromA_tiptarget.h5 \             # 第3步产物(状态+标签)
  --splice-frame <A帧号> \
  --output ../data/dp/mustard_recovery_round1.h5                          # [新增脚本]
```

## 6. 数据目录约定(仅 4 个目录)

| 内容 | 目录 | 例子 |
|---|---|---|
| 失触段 rollout | `data/closed_loop_rollouts/recovery_round<N>/` | `ep024.h5/.csv` |
| A→B 轨道 | `data/plans/recovery_round<N>/` | `fromA_s24.h5`、`fromA_s24_opt.h5` |
| 恢复段采集 | `data/trajectories/`(自动) | `recovery_s24_fromA_*.h5` |
| 拼接训练文件 | `data/dp/` | `mustard_recovery_round1.h5` |

训练:`train_dp.py --dagger-file mustard_recovery_round1.h5` 混入下一轮 fine-tune(与既有 DAgger 轮次 `dagger_mustard_nominal_round*` 相同用法)。

## 7. 既有能力 vs 需要的小改动(汇总)

| 环节 | 可直接用 | 需小改/新增 |
|---|---|---|
| 第1步 失触检测 | 失触计数、`[TACTILE-HOLD]`、`--rollout-h5` 闭环记录 | 失触阈值→提前结束 + 记 splice 帧 |
| 第2步 B 规划 | `generate_manifold_palm_plan.py`、`optimize_contact_plan.py`、`view_inverse_palm_plans.py` | 支持以任意帧(A)位姿为起点 |
| 第3步 恢复采集 | `collect_trajectories.py --motion-mode planner_inverse` 全套 | `--init-h5/--init-frame` 初始状态注入 |
| 第4步 拼接建集 | v4 导出链、`train_dp.py --dagger-file` | `build_recovery_samples.py`(拼接+切窗) |

## 8. 每轮采集的核对清单

1. 失触段:A 帧确实处于失触(loaded 接触 <3),不是误触发;
2. A→B 轨道:可视化贴着瓶身、无穿模,`grasp_keyframe_valid` 全 True;
3. 恢复段:四指接触率 ≥98%、恢复建立时间 <2 s、无 over-force、与主数据同格式可导出;
4. 拼接:帧率一致、splice 标记正确、样本窗口规则生效(目标全在恢复段);
5. 一轮建议 ≥20 条恢复 episode(覆盖不同失触形态:单指/双指脱开、漂移方向),再进入训练。
