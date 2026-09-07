# 速度均匀化与 SO(3) 多轴覆盖 — 进度记录

日期:2026-08-26。目标:**在 ycb_mustard 上、SO(3) 随机旋转轴下实现 ≥99% 四指接触率(无 contact gating),生成连续无停车轨迹作为 DP 训练数据**。

## 1. 机制

planner_inverse 管线:
1. `generate_manifold_palm_plan.py` 在物体系规划手掌椭圆轨迹(物体固定)
2. 采集器反演为固定手掌世界系 + 等效物体运动
3. 接触流形微分 QP + MeshNormalOracle 逐帧预测手指 → FullHandMCC/导纳保持接触

规划参数(codex 原始命令,必须保持一致):
`--path-mode minimum_enclosing_ellipse --palm-tangent-sign -1 --ellipse-arc-region calibrated --ellipse-section-fraction 0.85 --seed {seed} --ellipse-azimuth-deg {az} --angle-deg 120 --frames 2500`

## 2. 采集结果汇总

| # | plan | 速度参数化 | 速度 | all4 | 备注 |
|---|---|---|---|---|---|
| 1 | mustard_cap_s310_sec0p85_120deg(原版) | arc_bspline_cosine_ramp(旧) | 0.65× | **99.77%** ✓ | 峰值 0.158°/帧 |
| 2 | 原版 | 旧 | 1.0× | 96.64%(历史) | 未达标 |
| 3 | cap_s310_sec0p85_arclen_ramp | arc_length+ramp0.10 | 1.0× 均匀 | 96.04% | 峰值 0.119°/帧,启动平滑 |
| 4 | cap_s310_az50_sec0p85_arclen_ramp | arc_length+ramp0.10 | 1.0× 均匀 | 96.36% | azimuth 25→50 稳定 |
| 5 | cap_s310_sec0p85_blend05 | 组合弧长 w=0.5 | 1.0× 均匀 | 62.88% | 旋转峰值降但位置波动 2.2× 成新瓶颈 |
| 6 | cap_s311_az25_sec0p85_arclen_ramp | arc_length+ramp0.10 | 1.0× | 15.00% | **初始抓握失败**(第 0 指 20%) |
| 7 | 同上 | arc_length+ramp0.10 | 0.65× | 21.78% | 降速无效 → 非速度问题 |
| 8 | cap_s312_az25_sec0p85_arclen_ramp | arc_length+ramp0.10 | 1.0× | 77.88% | 初始抓握部分失败 |

## 3. 关键发现

### 3.1 轨迹区域由三个参数决定(曾导致长时间漂移)
- `--path-mode` 必须是 `minimum_enclosing_ellipse`(默认是 mesh_latlon,会走到瓶底 z≈0.005!)
- `--palm-tangent-sign -1`:失败 plan 全为 +1 → offset 迭代把手掌推到不同区域
- `--ellipse-section-fraction 0.85`:cap 端区域椭圆(z 0.238-0.263)
- 正确区域:z=[0.238,0.263] y=[0.007,0.130] 瓶盖区

### 3.2 速度均匀化的正确做法(用户要求:规划时让速度统一,不盲目降速)
- 原版速度剖面:位置 max/mean 2.3×,旋转 4.2×(max 0.2434°/帧),前 10% cosine ramp
- `resample_pose_uniform_arc_length`(位置弧长重采样)后:位置 1.0×、旋转 1.8×
- **必须保留 cosine ramp 10%**:全均匀序列第一帧就 0.0951°/帧全速启动 → 第 4 指启动崩溃(51.8%→96.04% 的差别)
- 纯 rotation 弧长重采样:位置爆炸 11×;blend w=0.5:旋转峰值更低(0.085)但位置 2.2× → 第 4 指 63%
- **结论:arc_length+ramp0.10 是当前最优参数化**

### 3.3 速度 vs 初始抓握(双瓶颈归因)
- seed 310:settle 后 4 指接触稳定,速度是唯一瓶颈(0.65× → 99.77%)
- seed 311/312:settle 阶段就建不好四指接触(0.65× 降速无效),手掌一移动即崩
- **SO(3) 多轴 ≥99% 的根本障碍是初始抓握对随机姿态的鲁棒性,不是速度**

## 4. 代码变更清单

`generate_manifold_palm_plan.py`:
- 删除 mesh_latlon 分支(用户指令)+ `enforce_palm_outline_clearance` 辅助函数;CLI `--path-mode` 默认改 `minimum_enclosing_ellipse`
- `resample_pose_uniform_arc_length`:新增 `ramp_fraction` 参数(cosine 软启停)
- 新增 `resample_pose_uniform_blend`(位置+旋转归一化加权弧长)
- 修复 `resample_pose_uniform_rotation` de-sign bug(Slerp 用未 de-sign 的四元数会走 360°-θ 远弧)
- `minimum_enclosing_ellipse` 分支(实际执行路径)接入重采样 + 重算 clearance
- CLI `--time-parameterization {arc_length, rotation, blend}`(default arc_length)

`collect_trajectories.py`(前会话):`--planner-speed-factor`(0.05-4.0)时间参数化;曲率字段 `contact_curvature_k1/k2`;静态字段 `palm_command_pose_object`/`q_nominal`/`fingertip_pose_object`;planner attrs 合并
`build_dp_samples.py`(前会话):按 DP_DATA_COLLECTION_GUIDE 切窗为训练样本

## 5. 下一步(按优先级)

1. **初始抓握鲁棒性**(多轴覆盖的根本障碍):settle 阶段(当前 200 帧)4 指接触建立失败
   - 尝试增加 `--planner-settle-steps`(200 → 500/1000)
   - 检查 `wait_for_four_tip_prep` 的失败退出路径与初始构型(手掌相对 SO(3) 物体的位形)
2. seed 310 打磨:旋转峰值 0.119 → 0.10 以下(ramp 加宽/微降速),目标 ≥99%
3. 批量生成多轴数据:seed 310 + 多 azimuth 已稳定 96%,可作为 DP 数据候选

## 6. 2026-08-26 管线边界确认

用户确认的教师数据管线不是机械臂带手主动运动，而是：

1. 物体固定，在物体系中计算光滑手掌位姿路径；
2. 在稀疏路径点上规划并验证四指稳定抓握姿势；
3. 将整条手掌路径反演成“手在世界系固定、物体运动”的轨迹；
4. 相邻抓握关键帧之间，用物体已知运动和接触流形 QP 生成小步切向指尖运动；
5. 法向由力闭环单独管理：力低时向内加 offset，力高时向外退让；
6. 最终在 `planner_inverse` 环境里采集，统计物体开始运动后的四指接触率。

因此，机械臂 MCC 只用于在准备阶段将手稳定在固定世界位姿，不是采集轨迹的主动运动源。

## 7. 稀疏抓握关键帧 A/B 证据

- `mustard_cap_s310_keyframe_manifold_force_raw.h5`：仅约 12% 运行时关键帧被接受，但 all4 约 96.6--97.0%。
- `mustard_cap_s310_keyframe_manifold_force_v2_raw.h5`：放宽几何判定后 100% 关键帧被接受，但 all4 降到 82.6%，且力 p95 显著升高。
- 结论：“指尖距表面近”不等于“稳定抓握”。运行时最近面 + 独立 IK 会选到折叠、侧触或横向交叉的分支。

## 8. 当前实现决策

`optimize_contact_plan.py` 已有固定物体系的稀疏 knot 投影和姿态正则雏形，但当前仅输出稠密 `finger_q_plan`，且 `collect_trajectories.py` 并未读取该数据。接下来改为：

- planner H5 显式保存稀疏 `frame/q/contact_point_object/normal_object/quality/valid`；
- 关键帧必须同时通过表面距离、指腹朝向、关节限制、屈曲协同、三指横向顺序/间距和参考抓握偏差检查；
- 关键帧 q 只是 QP 的零空间/手型边界，不逐帧直接追踪；
- 中间目标是物体表面上的短程材料点路径，QP 只执行有限速度的切平面微分运动；
- 长时间失触触发局部重规划，不无限追踪过时材料点。

## 9. 稀疏关键帧管线首次物理验证

- `optimize_contact_plan.py` 已输出稀疏关键帧的帧号、q、物体材料接触点、法向、符号距离、指腹朝向误差、屈曲协同误差、横向顺序间距、参考抓握偏差和 valid mask。
- `collect_trajectories.py` 已按当前反演轨迹源帧在相邻有效关键帧之间插值；材料点驱动切平面 QP，q 仅作手型/零空间参考。
- s310、26 个稀疏关键帧的离线检查通过率 100%；首次 CUDA 物理回放 all4=`99.90%`，各指=`[99.97,99.97,99.97,100.00]%`。
- 首次回放力中位数为 `[15.66,14.17,14.87,16.23]N`，明显过高。原因是关键帧 nominal q 与力环同时占用法向自由度。
- 用户确认当前教师数据只训练 q：只要不出现数值爆炸、明显穿透或高频振荡，力超目标不作为淘汰条件。因此不启用尚未证明接触率的 `project_nominal_normal_motion`，保留已证明 all4=99.90% 的接触优先版。
- 不再继续对 s310 单轨迹调参；下一轮直接用 s311/s312 + 瓶身/瓶底不同截面轨迹做共同 A/B，防止过拟合已知成功轨迹。

## 10. 多轨迹稀疏抓握结果

| 轨迹 | 离线有效关键帧 | 物理 all4 | q 单步 p99 / max | 判断 |
|---|---:|---:|---:|---|
| cap s310 | 26/26 | 99.90% | 待补 | 合格 |
| cap s311 | 26/26 | **99.895%** | 0.00167 / 0.00851 rad | 合格，证明不是 s310 单轨迹过拟合 |
| cap s312 | 26/26 | 89.63% | 0.01117 / 0.08331 rad | 前半段初始抓握失败；后 2500 帧 all4=100% |
| body s208 | 9/26 | 待测 | 待测 | 存在长无效区间，需自适应细分 |
| bottom s320 | 6/26 | 待测 | 待测 | 最小横向间距 10.9mm，属困难区域 |

s312 时序定位：失败集中在记录开始后前约 1300 帧，随后四指持续 100%。原因不是轨迹后半段不可达，而是旧 closure-path 建立的真实接触与离线第一关键帧之间没有静止过渡。直接在未接触时追离线关键帧也不行：固定 site 与真实 collision pad 中心的偏差使无名指停在表面外约 12mm。当前修正顺序为：

1. closure-path 先建立真实四指接触并标定 pad 接触点；
2. 物体保持反演第一帧不动；
3. settle 窗口内由流形 QP 将真实抓握平滑过渡到第一关键帧；
4. 过渡后才开始物体运动和记录。

修正后 s312 失败前半段短测（运动 1300 帧，旋转 20.4deg，平移路径 66.5mm）：

- all4=`100.0%`，各指均 `100.0%`；
- q 单步变化 p99=`0.00109rad`，max=`0.00453rad`；
- 这部分在修正前是 s312 接触率最差、q 跳变最大的区间，说明“真实接触标定→静止关键帧过渡→运动”顺序有效。

## 11. 瓶身方位角批量可行性审计

12 条不同方位/方向的瓶身路径离线关键帧有效率：

`s202=50.0%, s203=61.5%, s204=26.9%, s205=73.1%, s206=53.8%, s207=100.0%, s208=34.6%, s209=3.8%, s210=88.5%, s211=100.0%, s212=42.3%, s213=73.1%`。

结论：相同类型的瓶身轨迹对初始 SO(3)/截面方位很敏感；应先物理验证 s207/s211 两条完整可行路径，再把 s210 的少数失败段分段重规划。s204/s209 等低有效率路径不应依靠放宽阈值硬接，应回到手掌路径层调整截面/安全距离。

## 12. 数据质量优先级与 CUDA 多轨迹并行

用户再次确认：当前教师数据只用于学习手指 `qpos`，力值本身不是训练目标。因此质量排序固定为：

1. 运动阶段四指持续接触率，目标 `all4 >= 99%`；
2. 关节轨迹平滑、手型不退化、无高频振荡；
3. 轨迹覆盖范围和方向多样性；
4. 力仅用于判断“是否接触”和在线法向纠偏。超过期望力不淘汰轨迹；只有数值爆炸、明显深穿透或力诱发振荡才判故障。

`filter_trajectories.py` 已核对：它没有最大力拒绝项，力只作为接触下限。因而无需为高接触率轨迹降低关键帧预压。

为避免每条 planner 启一个进程，现已打通多轨迹 bundle：

- `bundle_manifold_plans.py` 将单环境 `(T,1,...)` 掌轨迹堆叠为 `(T,E,...)`；
- 稀疏抓握关键帧同步堆叠为 `(K,E,...)`；
- `collect_trajectories.py` 的反演控制器和关键帧控制器按 `env_id` 读取各自轨迹；
- s207+s211 双环境 CUDA smoke 在运动开始后的 182 帧中，两环境 all4 均为 `99.45%`（仅启动边界 1 帧），且两环境 `q_hand`、关键帧参考和物体位姿均明显不同，证明没有错误广播为同一条轨迹。

该 smoke 原本复用了完整 `5000` 步配置（1000 准备 + 400 静止关键帧过渡 + 3600 记录），确认接口后已主动停止；s207/s211 的各自完整 3600 帧物理结果此前均为 `100%`，不重复浪费时间。

## 13. 未覆盖曲面审计与首轮物理验证（2026-08-27）

旧的五条高接触轨迹并没有覆盖整个 Mustard：

- body s207/s211 主要覆盖 lower/upper body；
- cap s310/s311/s312 主要覆盖 upper body 和 shoulder/neck，真正 cap 只占约 0--1%；
- 因此此前的高接触率不能证明瓶底和窄瓶盖区域已经打通。

为避免继续重复已验证轨迹，新生成了四条曲面区域和截面法向均不同的候选，并把离线抓握求解增加到 20 次投影迭代。物理测试从固定的第 800 步开始，不再由某个环境的四指准备状态阻塞整个并行 bundle。

| 环境/轨迹 | 新覆盖区域 | 掌平移路径 | 掌旋转路径 | 物理 all4 | 各指接触率 | 判断 |
|---|---|---:|---:|---:|---|---|
| lower oblique s402 | bottom 59% + lower body 41% | 309 mm | 136.2 deg | 46.32% | 90.80/100/100/50.36% | 拇指长失触，不合格 |
| lower steep s404 | bottom 47% + lower body 53% | 440 mm | 191.3 deg | 17.60% | 71.08/91.64/62.44/64.52% | 多指不可达且路径过长，不合格 |
| cap oblique s413 | upper 30% + shoulder/neck 52% + true cap 18% | 246 mm | 177.3 deg | **99.56%** | 99.92/100/100/99.64% | 新区域合格；最长单次失触仅 2 帧 |
| bottom longitudinal s414 | bottom 25% + lower body 75% | 371 mm | 150.9 deg | 47.64% | 99.96/85.60/100/62.04% | 拇指主导长失触，不合格 |

四条轨迹的截面法向两两间隔为 `40.0--84.6 deg`，不是同一轨迹的重复测试。s413 证明高曲率 shoulder/neck/true-cap 路径可以超过 98%；但 lower/bottom 家族还不能规模化采集。

下轮筛选规则：

1. 生成阶段拒绝掌平移路径超过约 `300--350 mm` 的局部教师轨迹；
2. 离线稀疏抓握有效率优先要求 `>=95%`，不靠放宽几何阈值把明显不可达路径送进物理仿真；
3. 对 lower/bottom 批量搜索新的 section fraction、tilt、azimuth、方向和 SO(3)，只物理验证排名靠前的新候选；
4. 对 shoulder/neck/cap 生成新的多角度邻域轨迹，验证 s413 不是单一方位偶然成功；
5. 仍不使用力上限淘汰，核心指标保持 `all4 >=99%`（最低可讨论线 95--98%）、q 平滑和手型不退化。

## 14. 新方位批量筛选与扩量判定（2026-08-27）

在未测试过的 section fraction、tilt、azimuth 和独立 uniform-SO(3) 上生成了 12 个候选，其中 11 个通过掌轨迹运动学门控；`cap f96/t25/a240/s515` 因累计旋转 `308.4 deg`、单步旋转 `0.330 deg` 被生成器直接拒绝。11 条候选全部在物理仿真前运行 20 次稀疏接触投影，避免把明显不可达路径送进 CUDA。

新的四条代表性路径并行物理结果：

| 轨迹 | 离线关键帧有效 | 曲面覆盖 | 掌平移/旋转 | 物理 all4 | 最长 all4 失触 | q 单步 p99/max |
|---|---:|---|---:|---:|---:|---:|
| lower f22/t15/a10/s501 | 84.6% | lower body 99% | 191 mm / 79.4 deg | **100.00%** | 0 | 0.00246/0.00506 rad |
| lower f58/t50/a275/s506 | 100% | lower body 100% | 234 mm / 97.4 deg | **99.44%** | 3 | 0.00203/0.00468 rad |
| neck f90/t45/a120/s513 | 76.9% | lower 24% + upper 76% | 177 mm / 95.3 deg | **99.96%** | 1 | 0.00218/0.00494 rad |
| cap f94/t10/a185/s514 | 100% | upper 50% + shoulder/neck 28% + true cap 22% | 190 mm / 139.4 deg | **99.68%** | 1 | 0.00186/0.00573 rad |

四条均通过离线筛选条件 `all4>=99%`、每指接触率 `>=99%`、最长连续失触 `<=3` 帧。它们和旧 s207/s211/s310--312 不是相同轨迹，且补充了新的下瓶身方位、斜截面过渡以及更高比例的真瓶盖区域。当前证据已经支持启动 Mustard 的规模化候选采集，再通过离线筛选只保留高质量轨迹；不再需要逐条手工调到完全 100%。

本轮同时修复 `filter_trajectories.py`：静态 `(E,16)` `q_nominal` 不再被误当作 `(T,E,...)` 展平。多环境筛选后将其按保留帧展开为 `(T_selected,16)`；流式字段保持 `(T_selected,1,...)`。另外把旧的 time-major 交错展平改为 env/episode-major 连续展平。验证后的 episode 边界严格位于 `[2500, 5000, 7500]`，每条 2500 帧连续，不能再出现多环境帧交错污染 DP 时间窗口。四环境筛选文件已成功生成，不再出现 boolean-index 长度不匹配。

## 15. 12 条分层随机轨迹规模化试采（2026-08-27）

沿 Mustard 物体系长轴的 `18%--96%` 高度分层生成 12 条全新轨迹，每条使用不同的截面 tilt、azimuth、运动方向和 independent uniform-SO(3)。全部通过 `translation_path<=350 mm` 等运动学门控后，以 12 个 CUDA 环境完整采集 2500 个运动帧；不做在线淘汰、不用最大力阈值拒绝。

| 轨迹 | all4 | 最长失触 | 各指接触率（食/中/无名/拇） | q 单步 p99/max | 结论 |
|---|---:|---:|---|---:|---|
| f18 s601 | **99.92%** | 1 | 100/100/100/100% | .0026/.0060 | strict99 |
| f25 s602 | 83.04% | 38 | 100/99.9/100/83.1% | .0056/.0111 | 拇指失败 |
| f32 s603 | 1.64% | 1676 | 100/100/100/1.6% | .0058/.0107 | 拇指失败 |
| f40 s604 | 79.20% | 464 | 100/100/100/79.2% | .0030/.0073 | 拇指失败 |
| f48 s605 | 65.08% | 475 | 77.8/95.5/100/75.0% | .0055/.0160 | 多指失败 |
| f56 s606 | **99.44%** | 5 | 100/99.6/99.9/99.9% | .0022/.0059 | strict99 |
| f65 s607 | **99.96%** | 1 | 100/100/100/100% | .0022/.0048 | strict99 |
| f72 s608 | **99.84%** | 2 | 100/100/99.8/100% | .0025/.0071 | strict99 |
| f80 s609 | **99.88%** | 2 | 100/100/100/99.9% | .0016/.0033 | strict99 |
| f86 s610 | 98.16% | 26 | 100/100/98.2/100% | .0032/.0062 | relaxed98 |
| f91 s611 | 92.36% | 102 | 92.4/100/100/100% | .0051/.0142 | 食指失败 |
| f96 s612 | 84.88% | 176 | 100/100/84.9/100% | .0020/.0061 | 无名指失败 |

批量成功率：strict99=`5/12=41.7%`，relaxed98=`6/12=50%`，all4 中位数=`95.26%`。这支持“广泛 raw 采集 + 离线筛选”的路线，但还不支持把所有随机轨迹直接用于 DP。

五条 strict99 轨迹并非只集中在容易的瓶身横截面：

- s601 覆盖 `40% bottom + 60% lower body`；
- s606 覆盖 `48% upper + 35% shoulder/neck + 17% cap`；
- s607/s608 跨 lower/upper body；
- s609 是另一独立 SO(3) 的 lower-body 方位；
- 五条截面法向两两最大分离 `87.4 deg`。

因此当前高质量集合已经包含瓶底、瓶身、瓶肩和真瓶盖的曲面变化。离线关键帧有效率只能作排序特征，不能作硬门槛：s601 离线仅 46.2% 却物理 all4=99.92%，而部分 65% 候选仍会失败；最终标签必须来自物理闭环接触率。

已保存两个 episode-major、无帧交错的数据集：

- `mustard_scale12_physics_relaxed99.h5`：5 条 / 12500 帧；
- `mustard_scale12_physics_relaxed98.h5`：6 条 / 15000 帧。

## 16. 单侧瓶身到瓶底落差轨迹（2026-08-27）

完整横穿瓶底不是唯一的数据来源。为专门采集“平直瓶身 → 底部圆角/落差 → 平坦瓶底”，生成器新增 `--ellipse-arc-center-offset-deg`：对于角度为 `A` 的 bottom arc，设置 `offset=-direction*A/2` 可让轨迹终止在瓶底极值，而不继续横穿到另一侧。

四条不同纵截面/SO(3) 的首次物理测试：

| 轨迹 | 几何覆盖 | 法向累计变化 | all4 | 最长失触 | 各指接触率 | 判断 |
|---|---|---:|---:|---:|---|---|
| a15 s701 | 58% bottom + 42% lower body | 58.0 deg | **98.32%** | 7 | 98.3/100/100/100% | 合格的瓶身→瓶底落差数据 |
| a75 s702 | 38% bottom + 62% lower body | 120.4 deg | 90.04% | 73 | 100/100/90.1/100% | 无名指失触，不合格 |
| a165 s703 | 有效点仍在 lower body | 76.1 deg | 76.24% | 442 | 100/100/100/76.2% | 拇指失触，不合格 |
| a255 s704 | 有效点仍在 lower body | 3.6 deg | **99.08%** | 3 | 100/100/99.2/99.9% | 接触合格，但不是落差正样本 |

因此已证明单侧落差可以采集：s701 达到 relaxed98 且确实覆盖瓶身和瓶底；下一批应围绕 s701 的纵截面邻域扩增，而不是重新尝试完整横穿瓶底。s704 可保留为普通 lower-body 数据，不能标成 bottom-transition。

## 17. 指尖实际接触面审计与背面接触修复（2026-08-27）

重新把 MuJoCo 物理接触点转换到每个 fingertip site 坐标系后发现：旧的“有碰撞且力超过阈值”不能证明物体接触在指腹正面。当前四个触觉 pad 的固定外法向均为 site 局部 `-X`；普通三指 mesh 的正面约在 `x=0mm`、背面约在 `x=20.6mm`，拇指背面约在 `x=29.1mm`。

s701 虽然旧指标 all4=`98.32%`，但无名指接触点 local-x 中位数为 `19.62mm`，关节中值为 `[0.449,-0.029,1.211,1.520]rad`。无名指近端打开而中/远端过度内收，实际长期以背壳接触，因此 s701 **撤销“合格 DP 教师轨迹”结论**。相比之下，s514 三根普通手指接触点 local-x 为 `0.35--0.84mm`，是有效正面接触。

根因和修复：

1. `surface_mcc_finger.py::_pad_normal_target()` 原先从当前 site 的 `±X/±Y/±Z` 中自动挑最接近物体法向的轴，侧面和背面都可能被判为朝向正确；现固定为“pad 局部 `-X` 对齐物体外法向的反方向”，只约束面对表面，不约束绕 pad 法向的 twist。
2. FullHandMCC 新增物理接触点的 site-local 分类。普通三指 `local-x>12mm`、拇指 `local-x>16mm` 视为明确背面接触；控制器把它当作指腹失触并进入恢复，不再在背壳上继续积分加力。
3. 新采集 H5 增加 `fingertip_pad_contact_valid` 和 `fingertip_contact_pos_tip`；`fingertip_contact` 以及在线质量门控都只统计有效指腹接触。
4. `filter_trajectories.py` 可从新字段读取，也可从旧 H5 的世界接触点和 fingertip pose 反算 site-local 接触位置；新增 `--max-back-contact-ratio`，strict teacher 默认不允许背面接触。
5. 稀疏抓握关键帧的最大 comfort deviation 从 `1.20` 收紧到 `0.90rad`，最大 synergy residual 从 `0.85` 收紧到 `0.35rad`；在线大幅手型退化检测同步从 `spread>2.0/residual>0.35` 收紧为 `spread>1.25/residual>0.25`。

回归结果：

- 固定指腹法向在 1000 个随机旋转上的最大向量误差为 `1.31e-15`；
- 旧 s701 env0 的 0/1000/2000 帧均正确识别为 `[食1,中1,无名0,拇1]`，无名指 local-x 分别为 `19.78/20.92/17.12mm`；末帧无名指回到 `0.52mm` 后重新判为有效；
- 旧 s514 每 25 帧抽样的四指 pad-valid 比例均为 `100%`；
- 新离线筛选拒绝 s701，保留 unvisited 文件中的 s513/s514 两条真实正面接触轨迹；
- 三个修改脚本通过 `py_compile` 和 `git diff --check`。当前 conda 环境没有安装 `pytest`，因此未运行 pytest 套件；新的控制闭环仍需做一条 CUDA 可视化/物理回放确认恢复动作没有引入振荡。

## 18. Aude/Khadivar 运动学健康指标与直接几何接触规划（2026-08-27）

根据 Khadivar、Yao、Gao、Aude Billard 的 *Online active and dynamic object shape exploration with a multi-fingered robotic hand*（RAS 166, 2023），手指运动学健康不能定义为“是否接近一条预设 open→grasp 轨迹”。论文的目标函数包含：

\[
Q=\Delta q^T M_q\Delta q+\delta^T M_\delta\delta
+\sum_i\frac{1}{\mu_i}+\sum_i\frac{1}{\eta_i},
\]

其中 \(\mu_i=\sqrt{\det(J_iJ_i^T)}\) 是指尖 manipulability，\(\eta_i=d_{min}(p_i,S_i)/d_{max}(p_i,S_i)\) 是指尖在可达域内的 isotropic reachability。关节限位、期望接触、手掌姿态连续性和手指互碰是约束。论文的 10,000 个随机关节样本用于建立每指 reachability/manipulability 数据集。

当前实现据此完成以下修改：

1. `optimize_contact_plan.py` 不再用 closure-path/surface intersection 初始化接触。第一关键帧从多组结构化关节初值出发，直接联合优化完整四指 q 和原始 mesh 表面接触；后续关键帧以前一解作连续初值。
2. `surface_mcc_finger.py` 为每根手指均匀采样 10,000 个 4-DoF 构型，建立 fingertip reachability convex hull；离线优化时直接计算精确 Jacobian manipulability，不使用论文为实时性采用的 GPR 近似。
3. 几何接触目标、指腹法向、关节变化、参考姿势、三指横向 lane、关节限位、\(\eta\) 和 \(\mu\) 共同进入求解。归一化屈曲平衡只保留为弱防折叠先验，不再作为 closure-based 硬判据。
4. `collect_trajectories.py` 在存在离线关键帧时直接使用第一个已验证几何接触 q 和四个物体材料点建立初始接触；没有关键帧时调用同一直接几何优化并缓存。主控制路径已不再调用 `_closure_surface_targets()`。
5. 新增关键帧/稠密诊断：`isotropic_reachability`、`manipulability`；`bundle_manifold_plans.py` 同步支持这些字段。

s701 离线结果从旧 closure-synergy 判据下的关键帧 `30.8%`，变为：

- 稠密 2500 帧几何可行率 `100%`；
- 稀疏关键帧 `26/26=100%`；
- pad normal error P95=`68.8deg`；
- \(\eta\) P05=`0.052`，\(\mu\) P05=`0.602`；
- 最小三指横向间距 `39.7mm`；
- 最大 q 单步变化 `0.0044rad`。

完整 CUDA 物理回归（3 环境并行、每条 2500 个运动帧、10 physics substeps）：

| 轨迹 | 四指正面接触 | 各指正面接触率 | 最长 all4 失触 | local-x P95 最大值 | q 单步 p99/max |
|---|---:|---|---:|---:|---:|
| s701（旧背壳失败） | **99.40%** | 99.72/99.76/99.88/99.92% | 4 | 0.75mm | .00221/.00883rad |
| s514（成功回归） | **99.80%** | 100/100/100/99.80% | 3 | 1.09mm | .00172/.00505rad |
| s606（成功回归） | **99.76%** | 100/100/99.92/99.84% | 2 | 1.58mm | .00251/.00488rad |

s701 无名指背壳接触已消失：四指接触点 local-x 中位数均为 `0.24--0.43mm`，而旧轨迹无名指为 `19.62mm`。新方法同时没有降低 s514/s606 的原成功接触率，说明它不是只针对 s701 调参。力仍不是教师数据筛选目标，但本轮最大力为 `19.3--60.4N`，没有再出现粗 closure 交点导致的 `2211N` 初始化深穿透。

结论：后续教师轨迹的“健康手型”主指标固定为 surface feasibility + \(\eta\) + \(\mu\) + \(\Delta q\) + joint/collision margins；不再用是否贴合固定抓握轨迹作为有效性标准。最终是否保留仍由 MuJoCo 真实正面 pad 接触率和 q 平滑性决定。

## 19. 旧 strict99 五轨迹在新版规划器上的完整回归（2026-08-27）

为确认 Aude/Khadivar 健康指标、直接 mesh 接触规划和正面 pad 判定没有只在
s701/s514/s606 三条回归轨迹上有效，重新测试了第 15 节中旧控制器下达到
strict99 的五条轨迹：`s601/s606/s607/s608/s609`。

测试配置保持一致：

- 每条轨迹重新运行 20 次直接几何接触投影，`preload=2mm`；
- 5 个 CUDA 环境并行；
- 800 步机械臂准备 + 400 步反演第一帧静止接触建立；
- 随后记录 2500 个运动帧；
- 每控制步 10 个 physics substeps；
- 统计的是有效指腹正面接触，不是任意 fingertip mesh 碰撞；
- 不用最大力阈值淘汰轨迹。

完整物理结果：

| 轨迹 | 离线几何可行 | 四指正面接触 | 各指正面接触率（食/中/无名/拇） | 最长失触 | q 单步 p99/max | 判断 |
|---|---:|---:|---|---:|---:|---|
| s601 | 68.7% | **80.60%** | 99.84/99.84/99.92/80.76% | 339 | .01226/.05492rad | 拇指长失触，不合格 |
| s606 | 100% | **99.84%** | 100/100/99.92/99.92% | 2 | .00347/.00464rad | 合格 |
| s607 | 100% | **99.96%** | 100/100/100/99.96% | 1 | .00244/.00416rad | 合格 |
| s608 | 100% | **99.76%** | 99.84/99.92/100/100% | 2 | .00257/.00395rad | 合格 |
| s609 | 100% | **99.84%** | 99.84/100/100/100% | 2 | .00286/.00546rad | 合格 |

结论：新版控制/规划回归为 `4/5 >=95%`，没有满足用户设定的 `5/5 >=95%`
规模采集启动门槛。s601 的旧 all4=99.92% 不能沿用：新版正面 pad 判定揭示其
拇指路径存在长期不可行段，离线 pad-normal P95 达 169.6deg，与物理失触一致。
因此当前不直接启动大规模采集；批量工具可以准备完成，但必须保留预检门控，
并将最终接受标准放在物理闭环后的离线筛选上。

本轮原始回归数据：

`data/trajectories/regression_aude_s601_s606_s607_s608_s609_full.h5`

## 20. 批量 SO(3) 随机轨迹生成 → 离线筛选 → 并行采集 → 接触率筛选流水线（2026-08-27）

用户目标：批量生成环绕 mustard bottle 的 SO(3) 随机轴轨迹（数量级 256 条），
打通两步数据采集流水线：

1. **第一步**：自动生成批量轨迹并做离线可达性检查，只保留全部关键帧通过判据的；
2. **第二步**：仿真环境并行采集，最后离线筛选四指接触率 ≥98% 的轨迹。

### 20.1 新增脚本与修改

| 文件 | 变更 |
|---|---|
| `optimize_contact_plan.py` | 新增 `--keyframes-only` 筛选模式：跳过 PCHIP 插值与密集逐帧评估，只算 26 个 grasp keyframe 的 IK 投影与判据（采集器真正消费的边界条件），产物为源 plan + grasp_keyframe_*（无 finger_* 密集集）。判据与全帧模式**完全一致**（s601 两模式均为 17/26）。实测耗时瓶颈在 26 次关键帧 IK 求解（≈70s/条），全帧评估只占小头——批量提速靠多进程并行，非该模式。 |
| `batch_generate_so3_plans.py` | **第一步**。默认 16 个环绕区域（section_fraction×tilt×azimuth，继承 sweep_v3 网格 +4 组填充）× 16 个 SO(3) seed = 256 候选；ProcessPool 并行逐条执行 generate → optimize --keyframes-only；输出 `offline_screening.csv`（逐条关键帧有效数/比例/状态）与 `passing_plans.txt`（全部关键帧有效的 plan 列表）；幂等（已有 `_opt.h5` 跳过）；失败产物自动删除（`--keep-all` 保留）。 |
| `batch_collect_and_filter.py` | **第二步**。收集 passing plans → `bundle_manifold_plans.bundle` 分批（默认 16 env/批）→ 逐批 subprocess 调 `collect_trajectories.py`（planner_inverse，批内并行、批间串行防单卡过载）→ 逐 env 算 all4 接触率（`fingertip_collision_found` 且 `|force|≥0.05N`）→ 筛选 ≥98% 逐轨迹导出独立 H5 到 `--selected-dir` + `contact_rates.csv`。 |

### 20.2 关键修正（打通时踩坑）

- **采集必须带 `--fixed-motion-start --no-contact-gate`**：不带时 motion 等待四指接触就绪才启动，prep 阶段三指接触的轨迹（如 s602 第 3 指 sd=+11.7mm）永不启动，报 `Recorded 0/2500 post-prep frames`。回归 attrs `motion_start_semantics=fixed_schedule`、`contact_gated_motion=False` 对应此二参数。
- **collect 对 `--filename` 无条件追加 `.h5`**，且输出写 `<cwd>/mcc_finger_compliance_control/data/trajectories/<filename>.h5`：脚本以项目根为 cwd、传纯文件名，再从固定目录移回 output-dir。
- `_contact_rates`/`_export_episode` 需按 (T, E, ...) 轴切片，`q_nominal` 等 (E, ...) 数组按 env 行取。
- 12 条批量 shell 脚本曾因管道掩蔽 optimize 失败码而静默跳过 s604（单独重跑 100% 正常）；批量脚本需 `set -o pipefail`。

### 20.3 12 条分层重跑（当前判据版本）与旧回归对比

| plan | 旧回归 keyframe | 当前 feasible | 当前 keyframe valid | pad P95° | 备注 |
|---|---:|---:|---:|---:|---|
| s601 f0.18_t20_a30 | 18/26 | 64.6% | 17/26 | 169.1 | 拇指反向，瓶底窄截面 |
| s602 f0.25_t40_a80 | — | 49.8% | 14/26 | 101.1 | pad 反向 |
| s603 f0.32_t55_a130 | — | 100% | **26/26** | 51.6 | |
| s604 f0.40_t25_a180 | — | 100% | **26/26** | 52.0 | 批量脚本误跳，单独补跑 |
| s605 f0.48_t50_a230 | — | 74.7% | 20/26 | 90.1 | |
| s606 f0.56_t65_a280 | 25/26 | 79.7% | 21/26 | 82.9 | 当前判据更严 |
| s607 f0.65_t20_a330 | 26/26 | 100% | **26/26** | 57.3 | |
| s608 f0.72_t40_a45 | 26/26 | 100% | **26/26** | 49.5 | |
| s609 f0.80_t55_a100 | 26/26 | 100% | **26/26** | 51.1 | |
| s610 f0.86_t20_a155 | — | 80.9% | 21/26 | 84.8 | |
| s611 f0.91_t35_a215 | — | 77.1% | 21/26 | 86.4 | |
| s612 f0.96_t15_a275 | — | 29.9% | 9/26 | 106.8 | 瓶盖端外侧距离超标（outside_p95=16.8mm） |

12 条中 5 条全有效（s603/s604/s607/s608/s609）。失败集中在 f≤0.25（瓶底窄截面
拇指 pad 反向）与 f≥0.91（瓶盖端 reachability/外侧距离）。

### 20.4 全链路物理验证（2026-08-27）

用第一步筛选出的 3 条同区域不同 seed 轨迹（f0.18 区域 s602/s603/s604，离线 26/26
全有效）做 3 env 并行真实采集（3700 步、800 准备 + 400 settle + 2500 记录、
10 substeps、seed 42）：

| env | plan | all4 接触率 | 各指 | 筛选 |
|---|---:|---|---:|---|
| 0 | f0.18_t20_a30_s602 | **99.96%** | 100/100/100/100 | ✓ |
| 1 | f0.18_t20_a30_s603 | **100.00%** | 100/100/100/100 | ✓ |
| 2 | f0.18_t20_a30_s604 | **99.88%** | 100/100/100/100 | ✓ |

**3/3 ≥98%，全部通过筛选并导出单轨迹。** 验证了：离线关键帧全有效 ↔ 物理四指
接触率 ≥98% 的对应在批量规模下成立；同一区域不同 SO(3) seed 的轨迹，离线通过的
物理表现均优（s601 是 seed 特定姿态 × 瓶底窄截面的偶发失败，批量筛选中被自然淘汰）。

### 20.5 使用指令

第一步（批量生成 256 条 + 离线筛选，预计 ~35 分钟 @ 12 workers）：

```bash
cd mcc_finger_compliance_control/scripts
/home/rimlab/miniconda3/envs/mjlab/bin/python batch_generate_so3_plans.py \
    --output-dir ../data/plans/mustard_so3_v1 --workers 12
```

第二步（对通过离线检查的 plan 并行采集 + 筛选 ≥98%，16 env/批）：

```bash
/home/rimlab/miniconda3/envs/mjlab/bin/python batch_collect_and_filter.py \
    --passing-list ../data/plans/mustard_so3_v1/passing_plans.txt \
    --output-dir ../data/trajectories/mustard_so3_v1 \
    --selected-dir ../data/trajectories/mustard_so3_v1/selected \
    --envs-per-batch 16 --min-contact-rate 0.98
```

产物：`offline_screening.csv`（第一步）、`passing_plans.txt`（第一步）、
`contact_rates.csv` + `selected/*.h5`（第二步）。

## 21. mustard_so3_v1 256 条离线计划审计与采集前修复（2026-08-28）

批量生成实际尝试 424 条候选，其中 262 条达到 `26/26` 关键帧有效；旧
`passing_plans.txt` 为满足 `target=256`，按文件名排序截取前 256 条。这会漏掉
6 条已通过计划，并使清单只包含 13/16 个区域（遗漏少量 f=0.90/0.91/0.96
瓶盖端样本），因此采集入口改为扫描 `--plans-dir` 的全部 `_opt.h5`，不再依赖
有截断偏差的旧清单。

### 21.1 256 条旧清单统计

- 文件完整、无 NaN、无重复路径名；
- 27 个独立 uniform-SO(3) 初始姿态；
- 掌平移路径 86--320mm，单步最大 0.142mm；
- 姿态累计变化 16.9--174.5deg，单步最大 0.285deg；
- 掌缘最小安全距离 31.0mm；
- pad normal P95 为 46.7--80.4deg，关键帧最小 \(\eta=0.0447\)，最小
  manipulability P05=0.253；
- 24 条累计姿态变化不足 30deg，5 条 posture deviation 超过新版 0.9rad；
- 全部使用 `direction=+1`。因此它们适合作为第一批“正向局部接触变化”数据，
  但不是最终双向数据集；下一轮必须补 `direction=-1` 的物理轨迹。

采集前加强门控后，从全部 262 条关键帧通过计划中保留 231 条：

1. `grasp_keyframe_valid` 必须全部为真；
2. `planner_realized_rotation_path_deg >= 30`；
3. `max(grasp_keyframe_posture_deviation) <= 0.90rad`。

已完成 dry-run：231 条被正确打包为 `14×16 + 1×7` 共 15 个 bundle，所有
bundle 均为 2500 帧，plan name 共 231 个且无重复。

### 21.2 batch_collect_and_filter.py 修复

1. `--max-trajectories` 从错误的固定 `1` 改为当前 bundle 的环境数，否则
   16-env bundle 的 episode 数和采集终止条件不一致；
2. 接触率优先读取 `fingertip_contact`，该字段已合并 collision、力阈值和
   正面 pad-valid；不再把背壳碰撞算作成功；
3. 新增每指接触率、最长连续失触和实际 q 单步跳变筛选；默认分别为
   `>=98%`、`<=10 frames`、`<=0.03rad`；
4. 单 episode 导出保留 `(T,1,...)` 环境轴，`q_nominal` 保持 `(1,16)`，不再
   生成与现有 export/DP 工具不兼容的 `(T,...)`；
5. 新增运动量/手型 preflight CSV 和 `--resume`，中断后复用已完成的
   `batch_*_collected.h5`；
6. plan 路径同时支持项目根、passing-list 相对路径和绝对路径。

回归文件验证：新版离线统计函数在五轨迹 H5 上复现
`s601=80.60%, s606=99.84%, s607=99.96%, s608=99.76%, s609=99.84%`，
最长失触及 q max 与独立分析一致；导出 schema 验证为
`episode_id=(2500,1), q_hand=(2500,1,16), q_nominal=(1,16),`
`palm_command_pose_object=(2500,1,7)`。

### 21.3 原始 H5 数据契约确认

每个记录帧同时保留：

- 手状态：`q_hand/qvel/q_pre/q_ref`；
- 实际触觉：`fingertip_force_world/local`、collision、正面 contact mask、
  世界/指尖局部接触点；
- 教师几何：接触法向、oracle 法向和曲率（只作 teacher/meta，不强制作为
  DP 部署输入）；
- planner 条件：`palm_command_pose_object`、实际 palm/object pose/twist；
- 监督与诊断：目标/参考指尖位置、QP 输出、恢复状态、材料点和 q nominal；
- 时序：episode id/step、record step、实际 motion/record start。

准备阶段不进入最终 H5：800 步机械臂准备和 400 步第一帧静止接触建立后，
每条只保存 2500 个运动帧。因此当前 raw schema 足以支持接触质量筛选以及后续
“接触历史 + 局部 palm planner command → future finger q”训练；训练导出时应明确
排除 oracle/curvature 等特权字段，除非只作为平衡与评估 metadata。

## 22. 瓶底覆盖与原地转腕补充计划（2026-08-28）

针对 231 条主计划中“瓶底平面覆盖不足”和“缺少掌心位置固定、仅手腕转动”
两类空缺，新增了独立补充计划，不修改或覆盖原计划集合。

### 22.1 真正瓶底覆盖

除原有 `26/26` 可行性外，额外使用高分辨率 source mesh 检查最近表面面片：

- 归一化长轴高度 `h <= 0.06`；
- 面片法向与瓶底外法向夹角满足 `dot >= 0.70`。

因此不会把擦过瓶底圆角/边缘的轨迹误计为瓶底覆盖。长距离一次性跨越完整瓶底的
多数候选仅达到 `8/26--21/26`，说明在当前严格手型约束下不能靠放宽标签伪造可达。
改为仍有 2500 帧的局部长窗口后，两条互补轨迹达到严格 `26/26`：

- `bottom_enter_a0_s801_opt.h5`：下瓶身进入平底，平底帧约 59.7%；
- `bottom_leave_a135_s805_opt.h5`：平底离开至另一侧下瓶身，平底帧约 59.9%。

另保留 `bottom_cross_d70_a135_s805_opt.h5`，其约 70.2% 路径位于真实平底区域，
关键帧同样为 `26/26`。三条分别补充“进入落差、离开落差、平底内横移”。

### 22.2 掌心固定的原地转腕

新增 `generate_wrist_twist_plan.py`：从已通过计划中选取健康抓握锚点，只绕
palm-Z（局部表面法向轴）执行 quintic smoothstep 转腕。脚本明确区分两种旋转
中心：`palm_origin` 表示手的位置严格不动、只改变朝向；`control_point` 表示绕
掌心接触控制点转动。补充数据清单采用前者，所有轨迹 palm body 位置数值上恒定。

- 瓶身：原地 `+30deg/-30deg` 两条均为 `26/26`；
- 瓶盖：原始 `-25deg` 仅 `19/26`，改为可拼接的局部 `+12deg/-12deg` 后，
  两个不同抓握锚点共四条原地转腕轨迹均达到 `26/26`。

合格轨迹统一列在
`data/plans/mustard_supplement_smoke/passing_supplemental_plans.txt`，应先独立进行
物理接触采集与 `>=98%` 四指接触筛选，再合并进入 DP 数据集。

### 22.3 提高 DP 窗口内的转腕信息密度

单程 12deg/30deg 虽然有 2500 帧，但对当前最短训练配置
`stride=3, pred_horizon=24` 而言，每个 72 帧预测窗口的平均姿态变化仅约
0.35deg/0.86deg，容易被模型当成静止保持。生成器因此加入 `--cycles`，在掌位置
严格不动的条件下执行平滑的 `0 -> +A -> -A -> 0` 往复周期。

严格离线检查通过的新版本为：

- 瓶身：`A=45deg, cycles=2`，累计 360deg，`26/26`；
- 瓶盖：`A=20deg, cycles=2`，累计 160deg，两个抓握锚点均为 `26/26`。

在 72 帧窗口中，瓶身相对转角中位数/P95 为 `10.03/27.40deg`，瓶盖为
`4.46/12.18deg`；在 144 帧窗口中分别为 `20.21/51.81deg` 和
`8.98/23.02deg`。补充清单已用这三条往复轨迹替换低信息密度的单程版本。

## 23. 正向掌轨迹执行失败复盘与反演必要性（2026-08-28）

当前教师数据先在物体系规划掌轨迹，再将其反演为“机械臂停在一个固定世界目标、
物体按等效 mocap 轨迹运动”。该选择不是单纯为了方便回放，而是此前三种正向执行
方式分别暴露了不同的动力学或可达性瓶颈。

### 23.1 直接覆写 palm root pose / mocap palm

直接逐帧写入掌位姿相当于无限刚度的运动学边界。固定支撑并非不能承受手指反力，
而是接触冲量无法改变下一帧的掌运动，系统没有掌端位移和阻尼来吸收能量。于是以下
三组约束会直接竞争：

1. 手掌按规划位姿强制运动；
2. 手指位置执行器追踪关节目标；
3. 硬接触约束阻止几何穿透。

接触求解器刚把发生穿透的几何推开，下一帧 root pose 又可能将其写回碰撞区，造成
大接触冲量、指关节来回弹动、接触面片切换和 MCC/QP 目标反复修正。降低手指刚度
只能减小力峰值，同时会削弱接触恢复，不能消除这个结构性冲突。

### 23.2 freejoint palm + 6D PD wrench

6D wrench 允许接触反力推动掌部，比 pose overwrite 更符合动力学，但一个悬空
free hand 并不等价于机械臂末端：

- 开启重力时，wrench 控制器必须同时承担整手重量、姿态保持和接触跟踪；
- 掌质心、wrench 施加点和期望腕部控制点不一致会产生平移/转动耦合；
- 接触指数量变化会突然改变闭环等效刚度；
- 高增益可减小轨迹误差，却重新接近无限刚度并激发振荡；
- 低增益较柔顺，但掌会被指尖反力推离规划路径，且产生稳态误差；
- 力/力矩限幅进一步造成轨迹相关、姿态相关的误差。

`run_planned_palm_contact.py` 后来关闭 floating-hand 重力，等价于假设上游机械臂
已承担重力。这可以作为诊断环境，但其惯量和支撑动力学仍不是实际机械臂，且 6D
增益需要随接触构型重新调节。

### 23.3 机械臂带手正向执行掌轨迹

该方案具有真实关节惯量、阻尼和重力补偿，也能通过 MCC 吸收手指反力；其主要
瓶颈转为全局可达性。当前掌轨迹仅根据物体表面和四指几何生成，没有联合优化：

- xArm 关节限位和连续肘部构型；
- 腕部奇异点与 wrist flip；
- 机械臂、手、物体和地面的碰撞；
- 整条路径的可操纵性及连续 IK 分支。

因此单个掌位姿可达不代表整条 SO(3)/瓶底/瓶盖轨迹连续可达。完整解决需要把
`q_arm(t)` 和 `q_hand(t)` 一起纳入全身轨迹优化，计算量和失败模式都会显著增加。

### 23.4 当前反演的数学关系与实际收益

设物体系规划的掌轨迹为 `T_O_P(t)`，机械臂稳定后测得的固定世界掌位姿为
`T_W_P*`，采集器设置：

```text
T_W_O(t) = T_W_P* inv(T_O_P(t))
```

因此理想情况下：

```text
inv(T_W_O(t)) T_W_P* = T_O_P(t)
```

反演只交换哪一侧执行运动，不改变手与物体的相对 SE(3) 几何。当前实现还具有
以下工程收益：

1. 机械臂只需到达并保持一个已验证的稳定姿态，不再要求整条掌轨迹处于其可达域；
2. 手掌不是删除自由度后的纯 fixed base，接触反力仍经真实手掌和机械臂关节传递，
   由固定目标 MCC 提供有限刚度、阻尼与恢复；
3. 机械臂承担真实重力支撑，不需要给 floating hand 构造虚拟重力补偿；
4. 规划阶段可以把全局机械臂可达性与局部四指可达性解耦，后者由 `26/26` 稀疏
   关键帧、关节限位、指腹朝向、横向排序和手型健康门控检查；
5. 采集状态机先准备机械臂，再设置反演第一帧并保持不动，重新建立真实四指接触、
   过渡到第一关键帧，最后才运动和记录，避免初始穿透和旧接触锚点污染。

当前高接触率不是“反演”一个步骤单独产生的，而是精确 SE(3) 反演、真实机械臂
静态柔顺支撑、第一帧 settle、离线抓握筛选、接触流形 QP、MCC 法向维持以及平滑
低速物体轨迹共同作用的结果。

### 23.5 适用边界

反演在相对运动学上等价，但在动力学上不完全等价。mocap 物体不受手指反力推动，
等效为无限质量轨迹源，因此不能用当前数据直接学习：

- 机械臂动作、腕部 wrench 或关节力矩；
- 自由物体受力后的运动；
- 高速情况下的惯性、冲击和真实滑移；
- 正向机械臂跟踪误差导致的力分布变化。

它适合当前低速准静态目标：给定上层 palm 相对运动，根据局部表面和接触历史学习
未来 finger q。当前 DP 不输出机械臂动作，训练也不把力作为监督目标，因此反演的
主要偏差与现阶段学习目标基本解耦。结论是：反演在数学上不是唯一方案，但在当前
工程阶段是有效且接近必要的教师数据解耦手段。

### 23.6 后续同轨迹 A/B/C 验证协议

不能仅依靠理论判断反演数据是否可迁移。后续从已通过物理筛选的轨迹中选择
`5--10` 条，其中必须包含瓶身、瓶盖/肩部、瓶底和原地转腕，并为同一个
`T_O_P(t)` 分别执行：

- **A：当前 planner_inverse**：固定世界掌目标 + mocap 物体运动；
- **B：floating palm**：固定物体 + freejoint palm 的 6D PD wrench 跟踪；
- **C：arm forward**：固定物体 + 机械臂 MCC/IK 正向跟踪，仅保留整条路径可达者。

为避免比较混入控制器差异，三组使用相同指尖规划关键帧、FullHandMCC 参数、物理
步长、接触模型、路径时间参数化和记录窗口。统一报告：

1. 实际相对位姿对 `T_O_P(t)` 的 position/orientation RMS、P95 和 max；
2. 四指同时接触率、各指接触率、最长连续失触和接触切换次数；
3. `q_hand` 的同时间/同弧长 MAE、P95，以及每步 `dq`、高频能量或 qacc；
4. 指尖接触点在掌系/物体系中的路径偏差和 pad 正面接触比例；
5. 力仅作动力学诊断：P50/P95/max、过力伴随穿透或振荡的区间；
6. B 的 palm tracking error/wrench saturation，C 的 IK residual、关节裕量和
   manipulability。

建议先运行 A/C：它直接回答“反演教师 q 是否接近真实机械臂正向执行”。B 主要
用于定位无限刚度与浮动腕动力学之间的差异，不应作为最终真机等价基准。若在低速
可达子集上 A/C 的相对位姿误差小、finger q 接近且接触模式一致，即可继续用反演
规模化采集；若只有 A 高接触而 C 系统性偏离，则需要在数据中加入 palm tracking
扰动，或采集少量 C 数据进行混合训练/微调。

## 24. 瓶底与原地转腕补充轨迹物理采集（2026-08-28）

231 条主计划已完成物理采集，其中 225 条通过严格筛选、6 条被拒绝。随后将第 22
节的补充清单打包为一个 6-env CUDA batch，使用与主批次一致的
`planner_inverse + FullHandMCC`、1ms 物理步长、10ms 控制周期、800 步机械臂
准备和 400 步第一帧 settle。6 条均完整保存 2500 个运动帧并通过筛选：

| plan | all4 | 各指最低接触率 | 最长失触 | dq P99 / max (rad) | 结果 |
|---|---:|---:|---:|---:|---|
| bottom enter s801 | 99.96% | 99.96% | 1 | 0.00334 / 0.00539 | 通过 |
| bottom leave s805 | 99.92% | 99.92% | 2 | 0.00356 / 0.00624 | 通过 |
| bottom cross s805 | 99.80% | 99.84% | 2 | 0.00436 / 0.00686 | 通过 |
| wrist body A45 C2 | 99.76% | 99.76% | 2 | 0.00366 / 0.00471 | 通过 |
| wrist cap A20 C2 s607 | 99.88% | 99.92% | 2 | 0.00263 / 0.00404 | 通过 |
| wrist cap A20 C2 s622 | 99.92% | 99.92% | 2 | 0.00287 / 0.00485 | 通过 |

逐帧复核显示失触均为离散的 1--2 帧事件，没有长时间失触被总体均值掩盖。瓶底
三条轨迹每指最大单关节活动范围约 `0.38--1.12rad`，说明它们不是近静止样本；
转腕轨迹每指最大单关节活动范围约 `0.18--0.54rad`，同样能产生有效手指监督。
实际 planner command 完整覆盖：

- bottom enter：283mm / 150.5deg；
- bottom leave：295mm / 146.9deg；
- bottom cross：444mm / 224.0deg；
- wrist body：位置严格不动，累计转腕 360deg；
- 两条 wrist cap：位置严格不动，各累计转腕 160deg。

固定目标机械臂在接触反力下仍表现出有限柔顺：实际相对位姿对 planner command 的
旋转 P95 为 `0.25--0.59deg`，位置 P95 为 `2.91--6.30mm`。这部分偏差应保留在
训练观测中，而不是用 teacher pose 覆盖实际 palm pose。

力峰值约 `30--51N`，其中 bottom cross 最高；但当前数据只监督 finger q，且这些
区间没有伴随持续失触、明显 q 跳变或数值穿透，因此按既定标准不拒绝，只保留力作
诊断字段。补充数据位于：

```text
data/trajectories/mustard_supplement_v1/
  batch_000_collected.h5
  contact_rates.csv
  selected/*.h5
```

补充 6 条全部合格后，当前总计为 237 条已采集轨迹、231 条严格合格轨迹
（主批次 225 + 补充 6）。后续训练合并时只使用两个 `selected/` 目录，不能把主
批次被拒绝的 6 条重新混入。

## 25. 指尖接触面与背部假接触审计（2026-08-28）

为排除“手指已经蜷缩、但依靠指尖背壳碰撞维持接触标签”的假教师数据，新增
`scripts/audit_fingertip_contact_surface.py`。判定基于每个物理接触点在对应
MCC fingertip site 局部坐标中的位置：三指 `local_x > 12mm`、拇指
`local_x > 16mm` 判为明确背壳接触；局部 Y/Z 不设阈值，因此任务允许的侧边接触
不会被删除。同时报告三段屈曲关节的归一化 synergy spread，专门检查
“手型退化 + 背壳接触”的耦合情况。

审计时发现旧 H5 的 `fingertip_contact_pos_tip` 存在一个记录时序问题：该字段来自
`policy(obs)` 的 pre-step debug，而 `fingertip_contact_pos_world` 和
`fingertip_pose_world` 在 `env.step()` 后记录。接触建立/解除瞬间，上一拍的零
placeholder 会被按下一拍姿态变换，生成米级局部坐标。231 条已选数据中共有
223 个这种错拍指帧，分散于 132 条轨迹；它们不是物理接触点跳变。

审计器和离线筛选现统一使用同一帧的 world contact + fingertip pose 重算局部点。
重算后的结果为：

- 主批次 225 条 selected + 补充 6 条 selected，共 `231/231` 通过；
- 2,310,000 个 finger-frame 中，真实背壳接触为 `0`；
- “手型退化 + 背壳接触”为 `0`；
- 最大 flexion synergy spread 为 `0.528 < 0.75` 健康阈值；
- 侧边接触被保留，不参与拒绝。

对主批次全部 231 条原始 episode 重算后，仍然是原来的 `225/231` 入选，状态没有
任何变化。两个明确的背壳失败样本本来就已被接触率门控拒绝：

- `f0.86_t20_a155_s605_opt`：拇指背壳 601 帧，最长连续 531 帧；
- `f0.86_t20_a155_s618_opt`：拇指背壳 917 帧，最长连续 773 帧。

因此当前 selected 数据不需要再删除轨迹；真正需要修的是接触点字段的帧对齐。
采集器现改为在 post-step 同帧重算并记录 `fingertip_contact_pos_tip`、pad-valid mask
和 `fingertip_contact`，控制器也增加 50mm 指尖 site 半径合理性门控，避免零点或
陈旧点被大负 X 误判为正面/侧面接触。`invert_trajectories.py` 会对旧 H5 同样重算
接触 mask 和局部点，因此旧数据无需重采；重新反演即可得到对齐后的 DP 输入。

审计产物：

```text
data/trajectories/mustard_contact_surface_audit.csv
data/trajectories/mustard_contact_surface_audit_passing.txt
data/trajectories/mustard_contact_surface_audit_rejected.txt
```

## 26. DP palm-frame 坐标契约与同帧接触修复（2026-08-28）

DP 数据接口现统一为 `palm_lower` 瞬时坐标系。物体系反演文件只作为 teacher replay
和中间坐标检查使用，不能再直接传给 `train_dp.py`。最终训练 H5 必须声明：

```text
dp_input_frame = palm
palm_frame_body = palm_lower
action_representation = absolute_q
```

逐帧变换定义为：

```text
point_palm  = R_world_from_palm^T (point_world - palm_position_world)
vector_palm = R_world_from_palm^T vector_world
```

当前 `contact_geometry_planner` 的 56 维输入为：

```text
q_hand                              16  joint rad
fingertip_contact_pos_palm          12  palm-frame point
fingertip_contact_normal_palm       12  palm-frame vector
fingertip_contact_mask               4  scalar/bool
palm_relative_twist_palm             6  palm-frame [linear, angular]
planner_palm_delta_pose_palm         6  current-palm [translation, rotvec]
```

输出保持未来 absolute `q_hand`，每步 16 维。训练器默认已从 `delta_q` 改为
`absolute_q`，但仍保留 legacy A/B 选项。

代码层面的强制检查包括：

1. `invert_trajectories.py` 对旧原始 H5 同帧重算 contact local point 和 mask；
2. `export_palm_dp.py` 拒绝没有同帧修复标记的旧 inverted H5，导出后逐字段检查有限值、
   frame 属性、接触点尺度和单位法向；
3. `dp_dataset.py` 不再提供 object-frame 训练字段，拒绝 object/world 或无明确 frame
   属性的训练 H5；
4. legacy `build_dp_samples.py` 已修正：contact point/normal/force 全部真正旋转到
   palm frame，planner 改成当前掌系 delta pose，监督改成 absolute qpos；
5. GP manifold/PointNet 预训练及 embedding 导出同样要求
   `dp_input_frame=palm, palm_frame_body=palm_lower`。

单轨迹完整回归结果：

- palm contact point 对世界系直接计算的最大误差：`2.28e-8m`；
- palm normal 最大误差：`1.10e-7`；
- palm force 最大误差：`5.56e-7N`；
- planner delta pose 最大误差：`0`；
- 有效接触点到掌心最大距离：`0.170m`（物理合理）；
- 输出 state shape：`(500, 56)`，q shape：`(500, 16)`。

CUDA 2-step LeRobot DP 冒烟训练已通过，checkpoint 明确记录：

```text
input_frame = palm
state_schema = contact_geometry_planner
state_dim = 56
action_representation = absolute_q
```

旧 raw trajectory 无需重采，但旧 inverted 和 palm DP H5 应重新生成；否则训练器会
主动拒绝，避免错拍 contact mask 或隐式 object/world frame 再次混入模型。

主批次 225 条与补充 6 条现在使用显式两阶段流水线处理：

```text
selected raw
  -> invert_trajectories.py --selected-dir（反演中间文件）
  -> export_palm_dp.py（最终 palm-frame DP 文件）
```

反演中间文件：

```text
data/inverted/mustard_v1_231_inverted.h5
```

最终训练文件：

```text
data/inverted/mustard_v1_231_palm_geometry_planner_dp.h5
```

最终文件为 127.9MiB、231 个 episode、每条 2500 帧，共 577,500 帧。检查结果：

- 四指同时有效接触率：`99.9233%`；
- index/middle/ring/thumb：`99.9865/99.9782/99.9872/99.9714%`；
- 有效接触点到掌心距离 P95/max：`167.25/175.00mm`；
- 法向单位长度最大误差：`5.96e-8`；
- 20 帧 planner 目标平移 P95/max：`2.41/5.87mm`；
- planner 旋转 P95/max：`1.43/8.07deg`；
- 文件中不存在任何 `_world` 或 `_object` 笛卡尔训练字段。

## 27. body--cap / cap-top--body 过渡数据并入训练集（2026-08-29）

在原有 `225 + 6 = 231` 条严格数据基础上，追加两批真实物理采集数据：

- `body_to_cap_mer270_A110_p12_i40_test.h5`：4 条 body→cap；
- `mustard_cap_top_to_body_mer270_s802_4x3700_raw.h5`：4 条从四指捏住盖顶
  到瓶身的 cap-top→body。

`mustard_so3_manifold_v1_body_cap11_raw.h5` 经检查为失败采集留下的 0 帧空文件，
没有加入。旧的低接触 `cap_to_body_*_test.h5` 和数组直接倒放文件也没有加入。

反演器新增 `--raw-file`，可把 `(T,E,...)` 并行采集文件中的每个环境独立编号为
一个 episode，并与多个 `selected/` 目录一次性合并。接触面判定也由单独的 local-X
阈值改成“两阶段判定”：先找 positive-X 候选，再查询原始指尖网格最近表面法向；
只有法向以 site `+X` 为主才视为背壳，合法侧边接触保留。

最终产物：

```text
data/inverted/mustard_v1_239_inverted.h5
data/inverted/mustard_v1_239_palm_geometry_planner_dp.h5
```

完整检查结果：

- 239 个唯一 episode，原 231 个 source 全部保留，新增 source 恰好 8 个；
- 每条 2500 帧，共 597,500 帧；
- 四指同时有效接触率 `99.7580%`；
- index/middle/ring/thumb 为
  `99.9128/99.9749/99.9377/99.9289%`；
- 输入仍为 palm-frame `contact_geometry_planner` 56 维，输出仍为 absolute q 16 维；
- planner horizon 仍为 20 帧，所有训练字段有限且不存在 `_world`/`_object` 字段；
- 有效接触法向单位长度最大误差 `1.19e-7`。

## 28. Mustard DP 不收敛诊断与局部运动建模计划（2026-08-30）

### 28.1 数据契约与法向语义检查

将当前 Mustard 数据与此前效果较好的 capsule
`so3_uniform_v2_palm_geometry_local_planner_absolute_q` 逐字段对照后确认，两者的 DP
接口相同，均为 palm-frame `contact_geometry_planner`：

```text
q_hand                              16
fingertip_contact_pos_palm          12
fingertip_contact_normal_palm       12
fingertip_contact_mask               4
palm_relative_twist_palm             6
planner_palm_delta_pose_palm         6
                                    --
state                               56

action: future absolute q_hand      16 / waypoint
```

没有发现字段顺序、维数、palm frame 变换或 action 标签错位。但旧 Mustard 导出中存在
确定的接触法向语义问题：凸分解接触法向、原始 source mesh 法向及 ContactSensor 法向
的极性混用，约 90% 法向与接触力方向相反，并出现 185 次相邻帧近 180 度翻转。现统一
训练/部署约定为 ContactSensor 原生方向
`primary_fingertip_to_object`；原始 mesh outward 法向在导出时取反，并对同一
episode/手指执行时间连续性检查。当前应使用：

```text
data/inverted/mustard_v1_239_mesh_normal_inward_inverted.h5
data/inverted/mustard_v1_239_mesh_normal_inward_palm_geometry_planner_dp.h5
```

修正法向后，用相同网络训练 10k step，Mustard validation MAE 仅改善约 4.4%。因此
法向错误会降低质量，但不是本次泛化不收敛的唯一或主要原因。

### 28.2 已确认的训练现象

旧 Mustard 50k absolute-q 模型表现为典型轨迹级过拟合：

| step | train trajectory MAE | validation trajectory MAE | validation hold-q |
|---:|---:|---:|---:|
| 1k | 0.03117rad | 0.03232rad | 0.01881rad |
| 10k | 0.01064rad | 0.02226rad | 0.01881rad |
| 46k | 0.00488rad | 0.02037rad | 0.01881rad |
| 50k | 0.00475rad | 0.02044rad | 0.01881rad |

训练集能够被拟合，说明 75.85M 参数的 conditional 1-D U-Net 容量足够；但验证
noise loss 在早期最低，随后随训练继续上升，说明继续增加训练时长或网络宽度不会
自动解决问题。239 条长轨迹可切出约十万个窗口，但相邻窗口高度相关，独立运动模式
仍只有约 239 条，不能把窗口数等同于独立数据量。

Mustard 也不是因为 2500 step 太长而在窗口内几乎静止。实测全部数据：

| 未来时间 | capsule 平均 `|delta q|` | Mustard 平均 `|delta q|` | Mustard `<1mrad` 窗口 |
|---:|---:|---:|---:|
| 0.05s | 0.00034rad | 0.00146rad | 33.7% |
| 0.20s | 0.00133rad | 0.00573rad | 0.1% |
| 0.40s | 0.00266rad | 0.01118rad | 0.0% |
| 1.60s | 0.01058rad | 0.03556rad | 0.0% |

每条轨迹关节活动范围中位数为 `0.315rad`，而 capsule 为 `0.108rad`。Mustard 的
0.2s planner command 平均包含 `1.78mm` 平移和 `0.67deg` 旋转。因此当前失败不是
“没有运动监督”，而是 Mustard 的局部状态到未来手型映射更复杂、跨轨迹覆盖不足，
且现有网络没有高效利用局部运动规律。

### 28.3 horizon 与 chunk 的实际影响

当前训练配置为 stride 5、`control_dt=0.01s`，因此 32 个 action waypoint 覆盖
1.6s；planner 只提供未来 0.2s 的一个 delta pose。Mustard 中约 28.6% 的窗口，
前 0.2s 的关节运动方向与完整 1.6s 方向明显不一致，capsule 仅约 0.97%。这是长
horizon 的条件歧义来源。

部署启用 chunk 时，每 10 个控制帧（0.1s）重新推理，action waypoint 间隔为
0.05s，通常只实际执行前约 2 个 waypoint；DTW 可跳过前 0--4 个点，C2 插值也会
使用邻近 waypoint。因此 1.6s 尾部不会被完整执行，但训练仍对 32 点等权计算
diffusion loss，U-Net 也联合生成整条序列，尾部不确定性仍会占据训练容量并影响前段。

只把 horizon 从 32 缩到 8（0.4s），其余数据、切分和网络参数不变，10k 结果为：

```text
H=8 validation MAE          0.01360rad
H=8 hold-q baseline         0.00592rad
H=8 first-4 validation MAE  0.01263rad
H=8 first-4 hold baseline   0.00337rad
```

短 horizon 有改善，但仍未解决问题。旧 H=32 best 模型在 chunk 相关的前 4 点上 MAE
约 `0.01344rad`，hold 仅 `0.00338rad`，所以不能把失败解释为“只有不会执行的尾部
不准”。

### 28.4 当前 LeRobot DP 没有高效使用局部运动的原因

当前 LeRobot conditioning 路径把 `16 x 56` 历史直接 flatten 为 896 维 global
condition；没有显式速度/法向变化率，也没有 causal temporal encoder。U-Net 必须
从轨迹级数据中自行学出“相邻帧求差 -> 估计局部速度 -> 外推未来”。在同一 Mustard
验证集上，仅用历史 q 的最近一步速度做无学习外推即可得到：

```text
0.2s future MAE: 0.00108rad
0.4s future MAE: 0.00272rad
1.6s future MAE: 0.01876rad
```

这证明短期可预测信息已经存在于历史中；当前 DP 的 H=8 MAE `0.01360rad` 远高于
简单速度基线，主要是表征/训练目标没有利用该信息，而非数据在短期不可预测。延长
历史不是首选：当前已有 16 个 stride 点，即 0.8s 历史；改成 32 帧会把展平条件
扩大到 1792 维，并在现有数据量下进一步增加过拟合风险。修正后的全历史 KNN 审计
也未显示 16 -> 32 帧能改善跨轨迹最近邻覆盖。

此前 capsule 上 legacy `delta_q` 不如 absolute q，不能直接否定内部 residual
建模。旧 delta 使用整段 demonstrated joint range 归一化短期小增量：单步 residual
归一化绝对值均值仅约 `0.00105`（capsule）/`0.00151`（Mustard），diffusion 的小
归一化采样误差会被较大的 joint-range scale 放大；若部署时再错误累加还会产生漂移。

### 28.5 修改优先级

第一阶段保持最终接口输出 absolute q，不重新采集数据，完成以下 A/B：

1. 在每个 stride-rate 状态中显式加入 causal 局部运动特征：
   `q_velocity(16)`、`contact_point_velocity_palm(12)`、
   `contact_normal_angular_rate_palm(12)`；仅当前后两帧接触均有效时计算接触导数，
   否则置零并保留 contact mask。
2. 构造确定性局部基线
   `q_base(t+k)=q_t+k*dt*clip(qdot_t)`。Diffusion 在内部只预测
   `teacher_q-q_base`，按 residual 的实际训练标准差/稳健分位数归一化；推理后立即
   加回 `q_base`，对外和 chunk scheduler 仍提供完整 absolute-q 序列，不做跨次
   推理的 delta 积分。
3. 先使用 `pred_horizon=8`。训练指标必须单独记录 first-step、first-4、full
   horizon，并加入 constant-velocity baseline，不能只看 noise loss。
4. 若前两项仍不足，将历史 flatten 替换为轻量 causal TCN：逐帧 MLP 96->128，
   4 个 dilation `1/2/4/8` block，输出 256 维 history latent；planner 单独编码后
   与 history latent 融合，再条件化较小的 U-Net `[128,256,512]`。这比继续扩大当前
   75.85M U-Net 更符合 239 条独立轨迹的数据规模，也满足未来 5--10Hz 实时部署。
5. planner 从单个 `t+0.2s` 目标改为局部 waypoint 序列，例如
   `t+0.05/0.10/0.20/0.40s`。这些是上层 planner 的局部运动意图，不是物体几何
   特权信息；若仍保持 1.6s action horizon，则 planner 条件也必须相应延长。
6. 在 diffusion noise loss 外增加物理 q/velocity 辅助损失，并提高 chunk 会实际
   执行的前 4--8 点权重；训练窗口按未来运动量分桶均衡采样，而不是删除所有慢速
   窗口，以保留稳定保持姿势的能力。

第一阶段验收线：

```text
H=8 first-4 MAE < hold baseline 0.00337rad
H=8 full MAE    < hold baseline 0.00592rad
目标接近无学习速度外推 full MAE 0.00272rad
```

达到离线标准后，再按 `teacher_dp -> chunk live_dp + MCC` 顺序验证接触率。未达到
离线标准前，不应通过继续增加训练时长、混入 capsule 数据或调 MCC 参数掩盖 DP 的
局部预测误差。

## 29. 显式运动表示与运动学残差基线实现（2026-08-30）

### 29.1 新状态数据契约

第一阶段已经实现，未重新采集原始轨迹，而是从修正法向后的 239 条 Mustard 反演数据
重新导出：

```text
data/inverted/mustard_v1_239_motion96_kinematic_palm_dp.h5
```

新 schema 名为 `contact_geometry_planner_motion`，每个时刻为 96 维：

```text
q_hand                                          16   [0:16]
fingertip_contact_pos_palm                      12  [16:28]
fingertip_contact_normal_palm                   12  [28:40]
fingertip_contact_mask                           4  [40:44]
q_velocity                                      16  [44:60]
fingertip_contact_point_velocity_palm           12  [60:72]
fingertip_contact_normal_angular_rate_palm      12  [72:84]
palm_relative_twist_palm                         6  [84:90]
planner_palm_delta_pose_palm                     6  [90:96]
```

运动特征严格按 episode 内因果后向差分计算。当前 stride 为 5、控制周期为 0.01s，
所以运动特征间隔为 0.05s；接触点速度和法向角速度只有在相邻两帧对应手指都有效接触
时才保留，否则置零，接触 mask 仍单独输入。训练导出数据和部署端 teacher-state 的
逐字段对照误差均在浮点容差内：位置最大约 `3e-8`、法向约 `1.2e-7`、点速度约
`9e-7`、法向角速度约 `3.3e-6`。

### 29.2 运动学残差 action

新增内部 action representation：`kinematic_residual_q`。对未来第 k 个 waypoint：

```text
q_base(t+k) = q(t) + k * 0.05s * clip(q_velocity(t), +/-1.0rad/s)
residual(t+k) = q_teacher(t+k) - q_base(t+k)
```

Diffusion 只学习 residual，并使用 residual 自身的稳健分位数尺度归一化；推理后立即
加回 `q_base`。因此模型 checkpoint 内部虽然是 residual，给 chunk scheduler、MCC
和关节执行层的接口仍是未来 absolute q，不会发生跨次推理的增量累积。旧
`absolute_q`、`delta_q` 和 56D 模型保持兼容。

`live_dp` 中 DP history 使用 nominal q 时，同时使用相同 nominal history 计算
q velocity，避免输入中的 q 和 qdot 来自两个不一致闭环；接触点/法向运动仍由实时
触觉历史计算。96D `live_dp + chunk` 已完成 120 帧 CUDA 接口烟测，状态构建、推理、
absolute-q 重建与 chunk 执行均可运行。该烟测仍使用旧 capsule replay/oracle，不能
用于评价 Mustard 接触率，只证明新代码路径没有维度或时序错误。

### 29.3 无学习基线与 2k A/B 结果

同一 validation split、`obs_horizon=16`、`pred_horizon=8`、stride 5 下：

```text
hold-q baseline, full H=8                0.005917 rad
constant-velocity baseline, full H=8     0.002774 rad
hold-q baseline, first 4                 0.003370 rad
constant-velocity baseline, first 4      0.001048 rad
```

使用原 75.85M conditional U-Net 只训练 2k step 的新模型结果：

| step | val full H=8 | val first-4 | val first-step | val final-step |
|---:|---:|---:|---:|---:|
| 500 | 0.003305 | 0.001419 | 0.000574 | 0.007110 |
| 1000 | 0.003058 | 0.001263 | 0.000466 | 0.006701 |
| 1500 | 0.002917 | 0.001238 | 0.000457 | 0.006289 |
| 2000 | 0.002849 | 0.001182 | 0.000419 | 0.006201 |

相对旧的 Mustard H=8 absolute-q 10k 模型，新模型在仅 2k step 时：

```text
full H=8:  0.013595 -> 0.002849 rad  （约 4.77x 改善）
first-4:   0.012626 -> 0.001182 rad  （约 10.68x 改善）
```

它已显著超过 hold-q 验收线，证明显式速度表示和基线残差确实解决了原网络不会利用
局部运动的主要表征问题。不过 2k 模型仍略差于恒速度基线：full 高约 2.7%，first-4
高约 12.7%。曲线到 2k 仍持续下降，因此下一步应先正式训练 10k--25k，并把“是否
稳定超过 constant-velocity baseline”作为选择 checkpoint 的必要条件；在超过之前，
不能仅根据较低 noise loss 宣称模型已经学到曲率导致的非匀速修正。

本阶段训练输出目录：

```text
data/models/mustard_v1_239_motion96_kinematic_residual_pred8_ab2k/
```

该目录只保留数据契约、metrics、manifest 和训练曲线；测试 checkpoint 在接口验证后
删除，避免与后续正式模型混淆。

### 29.4 正式 25k 训练结果

正式模型目录：

```text
data/models/mustard_v1_239_motion96_kinematic_residual_pred8_25k/
```

训练期间 64-window 指标显示，模型在约 5--6k 首次稳定超过恒速度基线，之后 action
MAE 继续下降；20--25k 已进入平台区。25k 时：

```text
train sample MAE      0.001263rad
validation sample MAE 0.001731rad
validation first-4    0.000732rad
validation first-step 0.000251rad
validation final-step 0.003787rad
```

validation noise loss 在 9k 最低，之后从 `0.03035` 上升到 25k 的 `0.04468`，但实际
关节轨迹 MAE 同期仍从 `0.002060` 降到约 `0.00173rad`。因此本任务不能根据 noise
loss 选择 checkpoint；它衡量去噪目标，不等价于最终 absolute-q 重建精度。

为消除每次训练评估只抽 64 个窗口造成的随机误差，在完全相同 validation episodes、
相同随机种子上对 20k、自动 `best.pt`（24k）和 25k 各做 `3 x 256` 窗口复评：

| checkpoint | full H=8 | first-4 | first-step | final-step |
|---:|---:|---:|---:|---:|
| 20k | 0.001782 | 0.000759 | 0.000261 | 0.003861 |
| `best.pt` / 24k | 0.001762 | 0.000753 | 0.000256 | 0.003816 |
| **25k** | **0.001736** | **0.000737** | **0.000252** | **0.003762** |
| constant velocity | 0.002668 | 0.001049 | -- | -- |
| hold q | 0.005920 | 0.003338 | -- | -- |

25k 相对恒速度基线在 full H=8 上改善约 `34.9%`，first-4 改善约 `29.7%`；这证明
Diffusion 已经不只是复制显式速度外推，而是学到了局部非匀速/曲面变化修正。自动
`best.pt` 的 24k 优势是小样本评估噪声造成的，实际部署应明确使用
`checkpoint_0025000.pt`。20k 后提升已经很小，并出现中等 train/validation gap，
继续单纯延长训练预计收益有限；下一步应转入正确 Mustard inverse 环境中的
`teacher_dp -> live_dp + MCC` 接触闭环验证。

## 30. Mustard 无特权部署审计与数据集轨迹验证

### 30.1 部署环境和信息边界修复

旧 `deploy_dp_inverse.py` 无论 H5 中记录什么物体，都会创建 capsule replay 环境；这会
让 Mustard checkpoint 在错误的碰撞几何中运行。部署入口现在从 H5 的 `object_id`、
`object_scale` 读取环境元数据并动态创建对应物体。物体 ID/scale 只用于构造物理环境，
不作为 DP 或 MCC 的观测。

MCC 的默认方向源由解析 oracle 改为 `hybrid`：

```text
有接触：使用实时 ContactSensor 接触法向
短暂失触：有限帧保留上一有效传感器法向
没有历史接触：使用默认抓握闭合方向搜索
```

解析表面 oracle 和旧 `active_capsule` planner 均被标记为特权路径；除非显式传入
`--allow-privileged-surface-oracle`，部署入口会拒绝执行。数据集测试里的 palm teacher
轨迹被视作上层 planner 命令；低层手指控制只读取实时关节状态和 ContactSensor，
不读取物体 mesh 最近点或解析法向。

### 30.2 nominal q 仍是正式闭环定义

正式 `live_dp` 保持：

```text
DP q/qdot history = nominal DP trajectory history
MCC input          = nominal DP target + live tactile/joint feedback
physical command   = MCC-corrected q command
```

原因是若把 `q_live` 直接反馈给 DP，MCC 为维持接触产生的低频关节偏置会被 DP 当作任务
运动继续外推，形成 `DP -> MCC correction -> live q -> DP` 的递归积分。该问题此前已在
Capsule 上验证。新增 `--dp-history-q-source live` 仅用于 A/B 诊断，默认仍是 `nominal`。

### 30.3 数据集轨迹部署结果

使用 25k Mustard 模型、CUDA、chunk execution、10 diffusion inference steps、传感器
法向 hybrid MCC 测试：

| episode / mode | q MAE | q P95 | >=3 指 | 原始 4 指 | 结论 |
|---|---:|---:|---:|---:|---|
| ep24 `teacher_dp` | 0.00640 | 0.00788 | 99.1% | 90.7% | 主要是 1--3 帧传感器/碰撞断续；允许 3 帧后 99.8% |
| ep24 `live_dp`, nominal, replan=10 | 0.13000 | 0.23659 | 83.1% | 21.2% | nominal 自回归长期漂移，失败 |
| ep24 `live_dp`, nominal, replan=20 | 0.05965 | 0.09976 | 92.6% | 36.9% | 改频率不能消除持续失触 |
| ep24 `live_dp`, live-q A/B | 0.14617 | 0.23293 | 98.4% | 87.6% | 接触提高但姿势漂移更大，不能替代 nominal |
| ep73 `teacher_dp` | 0.03164 | 0.04943 | 81.2% | 59.7% | 即使 teacher forcing，低层恢复也破坏轨迹 |

ep24 的 live sensor normal 与 teacher normal 余弦相似度约 `0.985--0.99`，因此失败并非
移除 oracle 后法向完全错误。nominal live 中 DP 轨迹相对 teacher 的偏差由前期毫米级
持续增长，约在 frame 1000 达 `0.126rad`、frame 1800 后超过 `0.27rad`，随后形成持续
失触。live-q A/B 能用物理接触把手留在表面附近，但其 q 误差没有回到 teacher，验证
了“把 MCC 补偿反馈给 DP 会改变任务姿势”的风险。

ep73 暴露另一个独立瓶颈：恢复 offset 多次达到 `0.08rad` 上限且释放过慢。将 offset
释放调快可把 q MAE 从 `0.03164` 降到 `0.02143rad`，但四指接触仍约 `58.1%`；把实时
传感器法向记忆由 5 帧扩展到 20 帧后也只有 `59.8%`。因此单纯增加滤波/记忆或恢复
步幅无法替代可行的接触恢复方向和姿势约束。

### 30.4 当前验收判断和下一步

数据集轨迹的 gate 尚未通过，所以暂不运行经线 planner。否则 planner 失败与 DP/MCC
失败会混在一起，无法归因；而现有 `active_capsule` 经线实现本身还使用已知 capsule
解析几何，也不符合无特权部署要求。

下一版应同时保持 nominal 隔离并解决两个层次的问题：

1. DP：训练时加入多步 self-rollout/scheduled sampling，或增加周期性 absolute-q
   anchor，抑制 kinematic residual 在 nominal 自回归中的长期积分漂移；不能用 raw
   `q_live` 直接掩盖问题。
2. MCC：把接触补偿显式记录为 `delta_q_comp`，部署输入可使用
   `q_task = q_live - delta_q_comp` 做受限状态校正；恢复器还需加入手型/可达性约束，
   避免 ep73 中单纯沿旧法向把关节推到 recovery 上限。
3. 验收顺序保持 `teacher_dp -> nominal live_dp -> sensor-only planner`。只有代表性
   teacher 和 nominal live 均达到目标接触率后，才进入上层主动 planner 测试。

## 31. ep73 执行层与 nominal/live-history 因果对照

为区分 DP 预测、位置执行和 MCC 恢复三个因素，新增只用于诊断的
`--teacher-action-source recorded`。它仍运行 teacher-conditioned DP 并报告预测误差，
但把逐帧 H5 `q_hand` 作为关节目标送入同一位置执行器；它不同于直接写 simulator
state 的几何 replay。

### 31.1 exact state、exact target 与 DP target

| ep73 执行方式 | DP/目标误差 | live q MAE | >=3 指 | 原始 4 指 |
|---|---:|---:|---:|---:|
| 逐帧直接写 exact teacher state | 0 | 0 | 100.0% | 100.0% |
| exact recorded q + position servo | 0 | 0.004760rad | 46.3% | 36.1% |
| teacher-conditioned DP + position servo | 0.000432rad | 0.004617rad | 46.0% | 35.7% |
| exact q + 较高位置增益 K=20 | 0 | 0.003021rad | 61.7% | 52.0% |

exact target 和 DP target 的接触结果几乎相同，因此 `0.000432rad` DP teacher-forcing
误差不是 ep73 失败原因。精确状态 replay 的 100% 接触证明反演几何/teacher 数据正确；
一旦经过物理位置执行器，接触反力带来的约 `0.003--0.005rad` 误差就足以破坏边缘
接触。提高位置刚度有改善但仍不足，而且初始冲击力显著增大，不能作为柔顺部署方案。

### 31.2 MCC 恢复和目标力对照

| MCC 设置 | live q MAE | recovery 时间 | 最大 recovery offset | >=3 指 | 原始 4 指 |
|---|---:|---:|---:|---:|---:|
| 旧 1.0N / 0.08rad 长记忆恢复 | 0.031637 | 22.7% | 0.0800rad | 81.2% | 59.7% |
| 1.0N / 0.02rad 受限快速释放 | 0.004145 | 9.4% | 0.0200rad | 80.5% | 60.4% |
| **1.5N / 0.02rad 受限快速释放** | **0.004415** | **0.37%** | **0.0057rad** | **97.6%** | **88.7%** |

最后一项允许最多 1/2/3/5 帧短暂接触断续后的四指接触率分别为
`94.0/95.1/96.5/98.5%`，力 P95 为 `2.82N`、运行期最大约 `5.24N`。因此 ep73 的
teacher failure 不是 DP 没学会，而是 1N 对执行误差没有足够接触裕量；同时旧恢复器
用大关节 offset 代替接触裕量，导致 q 轨迹被严重改坏。

本轮有效受限参数：

```text
mcc_desired_force                 1.5 N
runtime_loss_frames               8
sensor_normal_memory_frames       20
runtime_recovery_limit            0.02 rad
contact_search_step               0.10 mm/frame
recovery_offset_decay             0.98
recovery_decay_force_ratio        0.5
```

### 31.3 通过 teacher gate 后的 nominal/live-history A/B

在完全相同的 1.5N 受限 MCC 下：

| live_dp history | DP q MAE | live q MAE | >=3 指 | 原始 4 指 | 主要失败 |
|---|---:|---:|---:|---:|---|
| nominal q | 0.121282 | 0.096053 | 94.1% | 30.2% | frame~800 后拇指持续失触（拇指 32.5%） |
| live q | 0.158467 | 0.158505 | 89.9% | 73.5% | 接触尚可但整手任务姿势持续漂移 |

live-history 确实比 nominal-history 更能维持物理接触，证明真实状态反馈对最终闭环是
必要的；但它没有把手拉回 teacher，而是把 MCC/执行偏差当成新的任务运动继续外推。
当前训练样本中只有单一 `q_hand`，等价于干净 teacher 状态，模型从未见过
`q_live != q_nom`，也没有输入可辨别“任务运动”和“低层补偿”。因此直接把现有模型
切到 live q 不能解决漂移。

下一版训练输入应显式拆为：

```text
q_nom, q_live, q_live-q_nom, qdot_live, delta_q_comp,
live contact geometry/history, planner command
```

动作仍输出未来 absolute `q_nom`；MCC 输出 `q_cmd=q_nom+delta_q_comp`，且
`delta_q_comp` 不写回 nominal。训练数据必须包含时序相关的执行扰动和闭环恢复样本，
最好通过 simulator perturbation rollout/DAgger 获取；只对 q tensor 加独立高斯噪声
会让 q、接触点和法向彼此不一致，不能作为最终训练数据。

## 32. 下一阶段执行计划：从干净模仿到真实状态闭环

当前阶段不继续盲目增加干净轨迹，也不进入 active planner。下一阶段按以下 gate 串行
执行，前一项未通过时不得用后一项掩盖问题。

### Step 1：冻结并批量验证低层 MCC

候选低层参数固定为：

```text
mcc_desired_force                 1.5 N
runtime_loss_frames               8
sensor_normal_memory_frames       20
runtime_recovery_limit            0.02 rad
contact_search_step               0.10 mm/frame
recovery_offset_decay             0.98
recovery_decay_force_ratio        0.5
```

在 8--12 条覆盖低/中/高关节运动、不同掌轨迹长度和接触困难度的 validation episode 上
运行 `teacher_dp + sensor-only hybrid MCC`。逐条和汇总验收：

```text
允许 <=3 帧短暂失触后的四指接触率 >= 95%
q_live 对 teacher MAE <= 0.01 rad
force P95 <= 4 N
不存在长时间 recovery offset 顶到 0.02 rad
```

若仅少数轨迹不通过，先按 failure finger、失触连续长度、q tracking error、MCC correction
分解原因；不得直接调整 DP 或 planner。

### Step 2：生成物理一致的扰动恢复数据

保持原 teacher 轨迹作为任务参考 `q_nom`，通过仿真执行器/接触动力学制造而非 tensor
伪造以下扰动：

```text
0.01--0.05 rad 的时序相关关节执行偏差
20--100 帧的低频偏置/扰动
单指短暂失触及重新接触
1--3 mm 量级的接触位置变化
MCC 实际产生和释放的 contact compensation
```

第一阶段规模为 16--32 条代表轨迹、每条 4 个扰动 rollout，约 16--32 万仿真帧。
新 H5 每帧至少记录：

```text
q_nom
q_live
q_live - q_nom
qdot_live
delta_q_comp
live contact point / normal / mask history
planner command
future teacher q_nom label
```

扰动后接触点、法向和力必须来自重新运行的仿真；禁止只修改大 tensor 中的 q。

#### Step 2A（先执行）：无人工扰动的自然闭环 observation rollout

在添加人工扰动前，先使用已有高质量 object/palm/teacher-q 轨迹重新运行
`exact teacher q_prior + 1.5N bounded MCC`。不改变物体轨迹、不注入失败，只记录位置
执行器和 MCC 在正常高接触运行中自然产生的差异：

```text
q_prior          = 当前成功 teacher nominal q
q_cmd            = MCC + rate-limit/EMA 后实际下发的关节目标
delta_q_comp_cmd = q_cmd - q_prior
q_live           = 编码器/仿真实际关节位置
e_servo          = q_live - q_cmd
```

同时重新记录 live contact point/normal/mask、planner command 和 future teacher q label。
MCC 补偿不是需要额外传感器估计的 ground truth，而是控制器自身已知的命令分解；真机上
同样已知 `q_prior` 与最终 `q_cmd`。Step 1 已证明这类自然 rollout 在常规轨迹上可保持
3帧 gap 后 `96.2--99.6%` 四指接触，因此不会主动破坏主数据质量。

先用 Step 2A 数据做小规模双状态模型 A/B。只有自然执行偏差仍不足以覆盖 live 部署
分布时，才进入下述 Step 2B。

#### Step 2B（可选）：受控执行扰动

人工扰动不修改监督 action，只通过 actuator delay、刚度/阻尼随机化、平滑外力或短时
命令偏置产生物理一致的 observation deviation。此数据的质量标准不同于主 teacher
数据：允许短暂失触，但必须在规定时间内由 MCC 恢复，最终回到高接触状态；持续失触、
不可恢复碰撞或超出可达域的 rollout 丢弃。

#### Step 2C（live-q 闭环必需）：current-policy state aggregation / DAgger

自然 teacher rollout 只能覆盖 teacher manifold 附近的执行差异，不能独自解决当前 DP
自回归后访问到的新状态。实测 ep73：

```text
teacher MCC compensation P95       0.018 rad
live-q MCC compensation P95        0.021 rad
live-q DP nominal 对 teacher MAE    0.158 rad
nominal-history 失稳后 comp P95     0.145 rad
```

因此 live-q 的主要 novelty 不只是 MCC compensation，而是 policy 自己的 task prior 已经
漂离 teacher manifold；nominal-history 失稳后，低层补偿也会进入完全未见分布。

完成 Step 2A 的接口/时序校验后，必须运行当前 DP+MCC 收集其真实访问状态，但不能使用
当前 DP 的 action 作为监督标签。第一版在同一已知 palm/object teacher 轨迹上按时间索引
重新标注：

```text
observation = current-policy q_prior/q_cmd/q_live/tactile history
label       = 同一时刻开始的成功 teacher future q
```

只保留 teacher future 从当前状态仍局部可达、且重新执行能恢复高接触的窗口；偏差过大、
不可恢复窗口丢弃。后续扩展未知几何时，改由特权接触优化 teacher 根据当前物体几何和
planner future 在线重求成功 q label。这样数据包含 policy 真正遇到的困难状态，但
Diffusion action 分布仍只包含成功 teacher 动作。

这里的“扰动恢复数据”第一版只用于 **observation robustness / invariance**，不得把失败
执行动作当作 DP action 标签：

```text
输入 observation：扰动后真实 q_live、接触几何和执行偏差
任务 reference：原始干净 q_nom / planner command
监督 action：原始成功 teacher 的 future q_nom
低层恢复：仍由 MCC 完成
```

因此 Diffusion 学到的不是“失败动作也是一种可能模式”，而是“在有限执行偏差下仍应
输出同一条任务层 nominal 轨迹”。失败 rollout 中实际产生的 `q_cmd/q_live` 只作为
观测和诊断，绝不作为 action target。只保留物理有效、扰动幅度受控且最终可恢复的
片段；发生不可恢复自碰撞、持续失触或超出关节可达域的片段直接丢弃。

只有当后续证明 MCC 无法独立处理某类恢复时，才考虑第二阶段的 expert recovery
demonstration。届时必须由优化器/特权 teacher 给出成功恢复标签，增加明确 recovery
mode 条件，并对未见过的扰动方向、幅度和手指组合做 held-out 测试；不能把当前 policy
自己的失败动作混入主 Diffusion 数据集。

#### 重要修正：nominal 是滚动任务先验，不是固定抓握目标

DP 不能对所有 `q_live-q_nom` 都保持不变。实际偏差至少包含两类：

```text
执行偏差：actuator lag、MCC 法向补偿、短暂传感器断续
          -> 不应被递归积分进任务轨迹

任务偏差：未知曲面变化使当前接触点/法向和可达手型发生真实变化
          -> DP 必须根据 live q + tactile history 更新未来 nominal
```

因此 Step 2 数据分成两个监督集合：

1. **nuisance/invariance rollout**：只改变执行层，物体表面和任务轨迹不变；输入使用
   扰动后的 live 状态，标签保持原成功 teacher future q。
2. **surface-adaptation rollout**：改变局部物体/手掌相对运动或表面接触条件；由特权
   teacher/接触优化器从扰动后的真实状态重新求一条成功 future q，标签必须随曲面改变，
   不能仍使用旧 nominal。

部署中的 nominal 是每次 DP replan 后更新的 rolling task prior。建议显式构造：

```text
q_prior       = 上一次 DP 给出的当前任务参考
delta_q_comp  = MCC 已知的低层补偿
q_task_est    = q_live - delta_q_comp
e_track       = q_live - (q_prior + delta_q_comp)
```

DP 以 `q_prior, q_task_est, e_track, delta_q_comp, live tactile history, planner command`
为条件，输出新的 future absolute task q。这样可以利用 live q 适应真实曲面，同时避免把
已知 MCC 补偿误认为新的表面运动趋势。

### Step 3：扩展 DP 状态，保持任务输出不变

输入拆成任务流和执行反馈流：

```text
task stream:     q_nom, nominal velocity, planner command
feedback stream: q_live, q_live-q_nom, qdot_live, delta_q_comp,
                 live tactile geometry history
```

输出继续为未来 absolute `q_nom`，或当前以 nominal 运动学外推为基准的 residual；不得
改成相对 `q_live` 的动作增量。MCC 仍独立生成
`q_cmd=q_nom_DP+delta_q_comp`，补偿不写回 nominal history。

### Step 4：干净/恢复数据混合训练和闭环 A/B

首版保持现有 conditional Diffusion U-Net，按约 `50%` 干净 teacher 窗口和 `50%`
扰动恢复窗口训练 10k--15k。比较：

```text
旧模型 + nominal history
旧模型 + live history（诊断）
新模型 + q_nom/live/error/comp 双状态输入
```

验收以真实闭环为主：teacher_dp 不退化、live_dp 不持续漂移、恢复后 q 回到 nominal、
四指接触提高；noise loss 只用于训练健康检查。

### Step 5：接入无特权 active planner

只有 Step 1--4 在数据集轨迹上通过后，才使用实时传感器信息接入主动 planner。
现有解析 capsule meridian planner 使用已知几何，仍只允许作为 privileged A/B，不作为
最终部署结果。

## 33. Step 1 完成：1.5N 受限 MCC 多轨迹验证

从固定 validation split 选择 8 条覆盖低/中/高关节变化、不同路径长度和旋转范围的
episode。除 bootstrap 前 75 帧外的结果：

| ep | q MAE | q P95 | >=3 指 | 原始 4 指 | <=3帧 gap 后4指 | force P95 | recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.00437 | 0.00621 | 99.8% | 98.6% | 99.6% | 2.14N | 0.00% |
| 40 | 0.00437 | 0.00733 | 99.6% | 94.1% | 99.6% | 2.62N | 0.00% |
| 51 | 0.00683 | 0.00869 | 99.6% | 97.4% | 99.2% | 2.41N | 0.00% |
| 57 | 0.00461 | 0.00810 | 98.3% | 91.8% | 99.3% | 2.66N | 0.00% |
| 73 | 0.00442 | 0.00715 | 97.6% | 88.7% | 96.5% | 2.82N | 0.37% |
| 93 | 0.00460 | 0.00726 | 97.8% | 89.6% | 96.5% | 2.74N | 0.91% |
| 189 | 0.00821 | 0.00994 | 96.3% | 77.9% | 96.2% | 2.85N | 0.49% |
| 238 stress | 0.01514 | 0.02722 | 80.5% | 40.9% | 50.8% | 13.37N | 29.61% |

前 7 条常规 validation 全部通过既定 gate：3帧 gap 后四指接触 `96.2--99.6%`、
q MAE `<0.01rad`、force P95 `<4N`，且 recovery 很少。说明 1.5N 受限 MCC 可冻结为
下一阶段的标准低层控制器。

ep238 是单独的 body-to-cap 极端 stress 轨迹：掌路径约 `0.713m`、累计姿态变化约
`219deg`、单关节范围约 `1.07rad`。精确逐帧写入 teacher state 时其原始几何四指接触
也只有 `94.2%`，loaded 四指更低，且存在很高接触力；因此它本身不满足本阶段的高质量
teacher gate。ep238 不进入首批扰动训练集，保留为后续长距离/极限恢复 stress test，
不能用于否定常规轨迹上的低层参数。

Step 1 状态：**完成**。下一步开始实现 nuisance observation rollout 的记录契约；首批
只改变执行层，action label 始终保持原成功 future q_nom，不写入失败 action。

## 34. Step 2A 开始：因果闭环 rollout H5 记录接口

`deploy_dp_inverse.py` 新增 `--rollout-h5`。记录不是把同一 control call 中尚未执行的
新命令错误地配给当前传感器，而是明确保存两个时刻：

```text
observation at t:
  q_prior_applied(t-1)
  q_cmd_applied(t-1)
  delta_q_comp_applied = q_cmd_applied - q_prior_applied
  q_live(t)
  e_servo(t) = q_live(t) - q_cmd_applied(t-1)
  live contact point / normal / mask / force(t)

new command computed at t:
  q_prior_next(t)
  q_cmd_next(t)

supervision reference:
  teacher_q_hand(t), source episode/time index
```

H5 同时保存 palm-frame live DP state、planner delta pose、palm pose/twist、物体 ID/scale、
normal polarity、stride、obs/pred horizon 和 bootstrap 边界。新增
`--teacher-action-source recorded` 可用 exact recorded teacher q 作为 prior，同时仍运行
DP 仅做误差诊断。

ep24 300 帧 CUDA 烟测结果：

```text
q MAE                         0.00441 rad
>=3 tip contact              98.7%
raw 4-tip contact            96.9%
force max                    2.23 N
delta_q_comp natural P95     0.0172 rad
comp identity max error      0
servo identity max error     0
valid contact normal norm    1.0（浮点容差内）
```

数据契约烟测通过。下一小步是用通过 Step 1 的常规 episode 生成完整 Step 2A rollout，
随后实现 current-policy rollout 的 teacher time-index relabel 和局部可恢复性筛选。

## 35. Step 2A 完成：自然 MCC 补偿闭环数据

使用 exact recorded teacher `q` 作为滚动 task prior，同时让冻结后的 1.5N FullHandMCC
在真实物理闭环中产生 `q_cmd/q_live/contact`，完成 7 条常规 validation episode 的完整
2500 帧 rollout。前 75 帧仅用于历史 bootstrap，质量统计从第 75 帧开始：

| ep | q MAE | q P95 | >=3 指 | 原始4指 | <=3帧 gap 后4指 | force P95 | comp L-inf P95 | servo L-inf P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.00446 | 0.00633 | 99.8% | 98.9% | 99.8% | 2.13N | 0.02395 | 0.02912 |
| 40 | 0.00445 | 0.00759 | 99.5% | 93.7% | 99.3% | 2.63N | 0.02682 | 0.02956 |
| 51 | 0.00843 | 0.01002 | 99.8% | 97.2% | 99.1% | 2.38N | 0.03096 | 0.03484 |
| 57 | 0.00470 | 0.00835 | 98.5% | 91.8% | 98.1% | 2.68N | 0.03280 | 0.03393 |
| 73 | 0.00450 | 0.00717 | 98.0% | 90.1% | 96.6% | 2.77N | 0.02600 | 0.02981 |
| 93 | 0.00471 | 0.00735 | 97.5% | 89.6% | 95.3% | 2.69N | 0.03165 | 0.03332 |
| 189 | 0.00681 | 0.00885 | 96.1% | 77.4% | 94.8% | 2.85N | 0.03417 | 0.03542 |

结论：

- 7 条共 `17,500` 个 control frames；按照当前 `stride=5, obs=16, pred=8` 的因果边界，
  共有 `16,695` 个可构造中心时刻。
- 所有 H5 的补偿恒等式、servo 恒等式和上一周期 prior 对齐误差均为 0；有效接触法向
  模长误差不超过 `2.4e-7`，未发现新的字段错拍。
- ep24/40/51/57 可作为首批 clean-natural rollout。ep73/93 包含少量真实恢复/短失触，
  更适合作为 natural feedback 数据，而不是最干净的 baseline。
- ep189 的 `>=3指` 仍有 96.1%，但原始四指接触只有 77.4%，且无名指/拇指失触更明显；
  保留作 hard-natural rollout，训练时单独分层采样，不能与 clean 数据等权混合。
- 这些 rollout 的 action label 始终是成功 recorded teacher future q；MCC 产生的
  `q_cmd/q_live` 只进入执行反馈流，不会把补偿或短失触动作污染为 Diffusion 标签。

数据位于：

```text
mcc_finger_compliance_control/data/closed_loop_rollouts/natural_teacher/ep*.h5
```

Step 2A 状态：**完成**。下一步是构造 current-policy rollout，并按同一 source episode/time
索引进行 teacher relabel；先在 ep24/40/51/57 上建立可恢复性 gate，再将 ep73/93/189
作为 held-out 难度递增测试。

## 36. Step 2C pilot：ep24 当前策略访问分布与 teacher takeover

使用当前正式 25k Mustard checkpoint、nominal history、chunk replan=10、sensor-only hybrid
MCC 对 ep24 运行完整 `live_dp`：

```text
q MAE / P95                 0.11719 / 0.18961 rad
>=3 指 / 原始4指            58.6% / 15.6%
per-tip found              16.3 / 66.2 / 100.0 / 91.3 %
force max                  13.18 N
```

漂移从约 frame 300--400 开始持续增长；后半段 `q_prior` 已远离 teacher，MCC 为追踪错误
task prior 产生的 `q_cmd-q_prior` 可达到数 rad。因此完整失败 rollout 绝不能直接并入训练，
也不能把其 q_cmd/q_live 当作 action label。

新增仅用于恢复性筛选的 `--live-teacher-takeover-frame`：先保留 live DP 到指定帧时的真实
物理状态，再把 task prior 切到同时间索引 recorded teacher q。DP 推理仍运行作诊断，
但不再控制动作。ep24 结果：

| takeover | 接管时 q MAE | 接管时接触 | 恢复到 q<0.01 且连续25帧4指 | 接管100帧后4指 | 接管后 force max |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.0104 | 4/4 | 2 帧 | 99.8% | 3.31N |
| 400 | 0.0288 | 4/4 | 22 帧 | 99.6% | 8.03N |
| 600 | 0.0543 | 3/4 | 39 帧 | 99.8% | 14.26N |

这证明“time-index teacher relabel”在有限偏差下确实可恢复，不只是按 q 距离做静态猜测；
但恢复所需瞬态力会随偏差快速增大。首版保守 gate 采用 takeover<=400 的分布（约
`q MAE<=0.03rad`、接管时4指仍在、实测恢复 max force<10N）。frame 600 虽能恢复，先作为
hard/recovery validation，不进入首批训练。frame 600 后的完整失败状态全部丢弃。

下一步：将 ep24 中通过 gate 的 live observation history 与同时间索引 future teacher q
打包为 recovery windows；然后对 ep40/51/57 重复 current-policy rollout 和少量 takeover
边界验证，不能只在 ep24 上拟合。

## 37. 部署 MCC 对齐实验：执行层改善接触，但 nominal-only history 会失稳

为验证“live DP 的漂移是否主要由部署 MCC 与数采 MCC 参数不一致造成”，在
`deploy_dp_inverse.py` 中新增独立的
`--mcc-preset collection_matched_sensor`。默认 `current` 行为不变。新 preset 只迁移数采
控制器中可在真实部署获得的低层参数：

```text
finger servo K/D/effort        35 / 2.5 / 35
per-tip force target           [3, 3, 3, 4] N
force integral gain            0.003
normal offset cap              3 mm
command rate / EMA             0.18 rad/step / 0.65
transient loss/search/release  6 frames / 0.20 / 0.25 mm
recontact confirmation         4 frames
```

没有迁移 mesh oracle、已知物体位姿/运动、最近表面投影、manifold QP 或 absolute surface
recovery。运行时法向仍来自实时接触传感器，短暂失触使用 20 帧法向记忆，超出记忆后退回
各指自身的 grasp-closure Jacobian 方向。因此这是 sensor-only 控制器对齐，不是 privileged
teacher 被带入部署。

ep24、相同 checkpoint/seed 下的完整 2500 帧结果：

| 名义动作 / DP q history | q MAE | q P95 | >=3 指 | 原始4指 | force max |
|---|---:|---:|---:|---:|---:|
| recorded teacher / teacher history | 0.00287 | 0.00398 | 100.0% | 99.8% | 6.70N |
| teacher_dp / teacher history | 0.00293 | 0.00399 | 100.0% | 99.8% | 6.87N |
| live_dp / nominal history | 0.27502 | 0.48274 | 12.7% | 5.2% | 92.40N |
| live_dp / live-q history | 0.20042 | 0.34301 | 99.9% | 99.6% | 11.57N |

隔离实验说明：

1. **控制器对齐是有效的。** 给定 recorded teacher 或 teacher-conditioned DP 目标时，新的
   sensor-only 执行层几乎精确复现数采接触；相对冻结的 1.5N 部署基线，ep24 q MAE 从约
   `0.00446rad` 降到 `0.00287rad`。recorded teacher rollout 的
   `delta_q_comp` frame-Linf P95 为 `0.01624rad`，没有额外 recovery offset。
2. **nominal-only history 的因果关系是 DP prior 先漂、然后失触。** frame 150 仍为四指
   接触时，prior-to-teacher MAE 已为 `0.00690rad`，MCC 后 live-to-teacher 为
   `0.00498rad`；frame 175 prior 先增到 `0.01128rad`，随后食指和拇指才失触。3 mm
   力环只能修正局部执行误差，不能替错误任务 prior 无限寻找表面。
3. **把真实 q 暴露给 DP 能保持接触，但旧模型仍发生任务轨迹漂移。** live-q history 下
   四指接触达到 `99.6%`，证明对齐 MCC 本身可以稳定接触；但 q MAE 单调增大，末段达到
   约 `0.35rad`，拇指力约 `10N`。因此“接触没掉”不等于“学会正确表面轨迹”。旧模型
   没看过这种 live-state 分布，会沿一条仍接触但偏离 teacher 的手型分支持续移动。

由此修正 Step 2C/DAgger 的首版状态定义：

```text
observation history: live q + live tactile geometry + planner command
supervision action:  同 source episode/time 的 successful future teacher q
diagnostics only:    q_prior, q_cmd, delta_q_comp, e_servo
```

不能再用 nominal q 替换 live q 来制造表面上稳定的输入，也不能把 MCC 的 `q_cmd/q_live`
当作 action label。首轮只保留 teacher takeover 实测可恢复、接触与力不过度恶化的窗口；
先在 ep24/40/51/57 形成小规模 aggregated dataset，再与 clean teacher 数据混合微调并以
未参与聚合的 episode 做闭环验证。

## 38. Mustard 单物体 DAgger 首版验证（历史 live-q pilot，已由 §39 修正）

> 本节记录首版实验及其失败边界。首版把 MCC 后的 live q 作为 DAgger observation，因而
> 同时混入了低层补偿偏差和 nominal 自回归偏差。当前工作路线已在 §39 改为专门稳定
> nominal history；本节 checkpoint 和 clipped-label 做法不再作为后续训练主线。

### 38.1 已实现的数据契约

新增 `scripts/build_dagger_dataset.py`，把 `deploy_dp_inverse.py --rollout-h5` 产生的真实
闭环状态转换为可直接和 clean teacher H5 混训的恢复数据：

```text
observation q_hand       = live q
observation tactile     = live contact point / normal / mask
observation motion      = live causal q/contact motion feature
observation planner     = 同时间索引的非特权 planner command
action_q_hand           = 同 source episode/time 的 successful teacher q
```

`dp_dataset.py` 通过 H5 的 `action_field=action_q_hand` 将 observation q 与监督 action
明确分离。`train_dp.py` 新增 `--dagger-file`、`--dagger-sample-ratio` 和 `--resume`，并保证：

- DAgger episode 只进入训练集，不污染验证集；
- 复用原 25k checkpoint 的 clean train/validation split；
- 复用原 normalization，避免加入少量恢复数据后改变输入/动作尺度；
- clean teacher 和 DAgger recovery 可通过 weighted sampler 控制混合比例；
- 不把失败的 `q_live`、`q_cmd` 或 MCC 补偿误当成 imitation action。

LeRobot DP 的最终归一化动作会限制在 `[-1, 1]`。这批恢复标签直接按原
`kinematic_residual_q` normalization 表示时，约 `44.2%` 的标量超过该区间，P95 绝对值
约 `4.12`。因此 DAgger episode 当前训练的是**限速后的 teacher 恢复方向**：仅对这些
episode 将归一化 action 截到 `[-1, 1]`；持续 replan 逐步回到 teacher，而不是要求一次
chunk 完成一个训练分布外的大跳变。

### 38.2 rollout 与保守筛选

使用原模型、`collection_matched_sensor`、live-q history、CUDA，对 Mustard clean train
split 中的 ep186/128/43/15 采集 current-policy rollout。ep24 保留为未参与聚合的闭环测试。

筛选 gate：

```text
teacher q MAE <= 0.05 rad
至少 3 指 pad-valid contact
force max <= 12 N
finger synergy spread <= 0.60 rad
normalized joint-limit margin >= 0.10
只填补 <= 3 帧的无效间隙
连续 segment >= max(120, (obs_horizon + pred_horizon) * stride)
```

最终从 4 条 rollout 得到 4 个连续 segment，共 `635` 个原始控制帧；`stride=5`、
`obs=16`、`pred=8` 后只有 `36` 个 DAgger windows。输出：

```text
data/closed_loop_rollouts/dagger_mustard_pilot/
  mustard_dagger_pilot_qmae005.h5
  mustard_dagger_pilot_qmae005.json
```

### 38.3 训练与闭环结果

首先以 DAgger sample ratio `10%`、lr `3e-5` 微调 1000 steps。干净验证集没有明显回归：

```text
step 500   val noise=0.04854   val q MAE=0.001637 rad
step 1000  val noise=0.04908   val q MAE=0.001629 rad
```

`teacher_dp` ep24 仍达到 q MAE `0.00284rad`、原始四指 `99.9%`、max force `6.46N`，证明
模型导出、推理和执行层没有损坏。但是同一模型的真正 `live_dp` 在约 frame 1600 后离开
恢复数据覆盖范围并失稳：

| checkpoint / rollout | q MAE | q P95 | 原始4指 | MCC有效4指 | force max |
|---|---:|---:|---:|---:|---:|
| 原 25k，ep24 held-out | 0.20042 | 0.34301 | 99.6% | 90.6% | 11.57N |
| DAgger 10%，step 500，ep24 | 0.23165 | 0.67132 | 85.2% | 82.4% | 2406.02N |
| DAgger 10%，step 1000，ep24 | 0.27963 | 0.76986 | 78.5% | 72.9% | 1263.70N |
| DAgger 1%，step 500，ep24 | 0.22011 | 0.61726 | 86.8% | 78.5% | 1160.60N |

保守的 `1%`/lr `1e-5` 对照仍没有改善 held-out ep24，说明问题不只是 oversampling 比例。
但在参与聚合的 ep128 前 500 帧上，10%/step500 checkpoint 将 q MAE 从原 rollout 的
`0.0550rad` 降到 `0.0286rad`，并保持原始四指 `98.8%`、MCC 有效四指 `99.3%`。这证明：

1. live observation -> teacher action 的 DAgger 数据接口正确，模型能学到已覆盖状态附近的
   恢复；
2. 只有 4 个 segment / 36 个高度相关窗口，只足以局部拟合，不能泛化到另一条 2500 帧
   轨迹的误差方向和后半段状态；
3. 干净 validation MAE 无法预测自回归稳定性，checkpoint 选择必须加入 held-out
   closed-loop 指标；
4. 当前 pilot checkpoint 不应作为最终模型，也不能立刻和 capsule 合并。下一轮应先扩大
   Mustard current-policy rollout 覆盖，并按迭代模型重新 rollout，再用 held-out episodes
   做早停，而不是继续增加同一小数据集的训练步数。

## 39. 路线修正：只对 DP nominal 自回归链做局部 DAgger

### 39.1 问题拆分

后续实验严格区分三种闭环：

```text
teacher_dp / teacher history
    ground-truth history 锚定，不存在 free-running exposure bias

live_dp / nominal history
    DP prediction -> DP history，自身误差沿 nominal 链累积

live_dp / live-q history
    q_live = q_task + delta_q_MCC + servo error
    除 nominal 漂移外，还会把低层 MCC 补偿误解释为 task motion
```

项目主线改为第二种。低层继续执行：

```text
q_cmd = q_nominal_DP + delta_q_MCC
```

但 DP 的 q history 只使用 `q_nominal_DP`。允许 `q_live != q_nominal`；只要 MCC 补偿有限、
接触/力/手型健康，就不训练 DP 去消灭或外推这个差值。DAgger 唯一目标是：

```text
slightly off-manifold nominal history -> successful future teacher nominal q
```

### 39.2 代码契约修正

`deploy_dp_inverse.py --rollout-h5` 现在同时输出：

```text
live_dp_state          原始物理 live state，仅作执行诊断
dp_observation_state   真正放入 DP causal history 的 state
q_live                 MCC + servo 后的实际 q
q_prior_applied        上一帧 nominal task prior
teacher_q_hand         同时间索引的成功 teacher q
```

rollout 额外记录 `dp_history_q_source`。`build_dagger_dataset.py` 默认读取
`dp_observation_state`，并拒绝不是 `dp_history_q_source=nominal` 的 rollout，防止再次把
live-q compensation 混入 nominal DAgger。状态中的 q velocity 同样来自 nominal history；
live tactile point/normal/mask 仍保留，因为它们是部署时真实可用的传感器信息。

构建器默认 `max_teacher_q_mae_rad` 从 `0.05` 收紧到 `0.015`。segment 起点强制对齐 DP
stride grid，避免从两次推理之间的 held state 开始 `[::stride]` 而造成时序错位。

首版“对所有 DAgger action 归一化后 clip 到 `[-1,1]`”也已撤销。训练现在读取 H5 的显式
`bound_normalized_action`；新的 local DAgger 默认为 false。`train_dp.py` 会在训练前统计
recovery action 范围，默认当超过 `10%` 的标量落在 `[-1,1]` 外时直接终止，要求重新采集
更早、更小的偏差，而不是把远距离 one-shot return 静默截断。

### 39.3 nominal smoke 结果

使用原 25k 模型、ep128、`dp-history-q-source=nominal` 运行 500 帧 CUDA rollout。只保留：

```text
nominal-to-teacher q MAE <= 0.015 rad
至少3指 pad-valid contact
force <= 12 N
健康手型和关节 margin
```

得到：

```text
有效连续帧             178
stride=5 后训练窗口      13
q error median/P95       0.00557 / 0.01380 rad
normalized action >1     4.75%   （首版 live-q pilot 为 44.2%）
normalized action abs P95 0.978
normalized action abs max 1.457
```

CUDA 1-step train contract 已通过，没有 clipping，并保持原 checkpoint split/normalization。
这验证了新数据路径在时序、状态来源、teacher label 和动作尺度上均可训练。

### 39.4 后续执行顺序

1. 用原 Mustard policy 采集 20--40 条多样化 nominal-history rollout；在 `0.003--0.015rad`
   的早期偏差区间形成局部稳定数据，不采已经跑飞的状态。
2. 固定若干未参与 aggregation 的完整 Mustard episodes，checkpoint 只按 held-out
   `live_dp / nominal history` 的 q drift、接触、力和手型健康选择；clean validation MAE
   只作为防回归指标。
3. 得到 `pi_1` 后必须重新 rollout，收集它访问的新状态形成 `D_2`，而不是持续重复训练
   `pi_0` 的恢复窗口。
4. 至少完成两轮 Mustard DAgger 并证明 held-out nominal drift 下降后，再加入 capsule clean
   teacher 数据联合训练；暂时不把 capsule 的 recovery 数据混进首轮验证。

## 40. Mustard nominal-history iterative DAgger 两轮实测（2026-08-31）

### 40.1 聚合数据

固定 20 条仅来自 clean train split 的多样轨迹：

```text
1, 67, 229, 233, 169, 74, 232, 81, 78, 237,
28, 63, 168, 226, 37, 54, 72, 71, 34, 200
```

每轮均运行 600 帧 `live_dp / nominal history / collection_matched_sensor`，构建数据时保持：

```text
nominal-to-teacher q MAE <= 0.015 rad
至少 3 指有效 pad contact
force <= 12 N
健康手型与 joint margin
连续有效段 >= 120 simulation frames
```

结果：

| 数据 | rollout policy | 有效段 | 有效帧 | stride 后窗口 |
|---|---|---:|---:|---:|
| D1 | pi0（原 25k） | 8 | 1310 | 79 |
| D2 | pi1（D1 fine-tune） | 10 | 1646 | 100 |
| D1+D2 | iterative aggregate | 18 | 2956 | 179 |

D2 新增了 ep67、169、226 等 pi1 才访问到的局部状态，证明不能只重复训练 pi0 rollout。
D1+D2 的归一化 recovery action 仅 `1.96%` 超出 `[-1,1]`，abs P95=`0.827`；训练不做
action clipping。批量采集脚本现在会先把 source/model/output 解析为绝对路径，避免
`deploy_dp_inverse.py` 改变工作目录后找不到相对路径。

### 40.2 pi1 与 pi2 训练

两轮都从上一策略继续 fine-tune 1000 steps，clean/DAgger 采样比例保持 `99.5/0.5%`，没有
通过增加 recovery 权重制造表面改善：

```text
pi1: D1, resume pi0, clean val q MAE = 0.001640 rad
pi2: D1+D2, resume pi1, clean val q MAE = 0.001620 rad
```

因此两轮均未破坏 teacher 分布预测精度。模型目录：

```text
data/models/mustard_nominal_dagger_round1_pi1_1k
data/models/mustard_nominal_dagger_round2_pi2_1k
```

### 40.3 held-out 完整闭环结果

以下均为同一仿真、同一 MCC、同一 seed、2500 帧、`dp_history_q=nominal`；ep24/40/51/57
从未进入 D1/D2。指标排除 75 帧 DP history warm-up。

| ep | policy | q MAE rad | q P95 rad | >=3 指 | 4 指 |
|---:|---|---:|---:|---:|---:|
| 24 | pi0 | 0.275* | 0.483* | 12.7%* | 5.2%* |
| 24 | pi1 | 0.120 | 0.242 | 38.2% | 18.4% |
| 24 | pi2 | 0.100 | 0.305 | 91.3% | 22.7% |
| 40 | pi0 | 0.122 | 0.231 | 83.5% | 37.3% |
| 40 | pi1 | 0.077 | 0.183 | 83.4% | 54.9% |
| 40 | pi2 | 0.075 | 0.105 | 84.8% | 44.4% |
| 51 | pi0 | 0.158 | 0.268 | 21.4% | 13.4% |
| 51 | pi1 | 0.214 | 0.487 | 75.4% | 24.1% |
| 51 | pi2 | 0.177 | 0.439 | 84.6% | 43.7% |
| 57 | pi0 | 0.351 | 0.702 | 75.8% | 47.0% |
| 57 | pi1 | 0.200 | 0.386 | 98.8% | 41.7% |
| 57 | pi2 | 0.161 | 0.398 | 87.8% | 47.4% |

`*` 为此前同契约基线报告；其余来自本轮 CSV。结论不是“已经稳定”，而是：iterative local
DAgger 对 nominal autoregressive drift 有因果改善，尤其 ep24/51；但 4 指接触和后段过力
仍不合格，说明 179 个 local windows 尚不足以覆盖完整轨迹。

### 40.4 checkpoint 必须按闭环选

pi2 的 250/500/750/1000 step checkpoints 在 held-out ep24、51 上做了完整闭环。250 step
接触不足；500 step 严重过力；750 step 在 ep51 的 q P95 达 `0.741 rad`；1000 step 在漂移、
接触和过力之间整体最好。因此当前 `pi2/best.pt` 可以保留，但选择依据应是闭环折中，而非
clean validation loss 单一指标。

### 40.5 下一轮边界

下一轮继续使用 pi2 rollout，但不纳入 `q error > 0.015 rad` 的晚期跑飞状态，也不让 DP
拟合 MCC correction。应优先扩充：

1. ep24/51 类在 800--1600 帧附近刚开始偏离前的相邻 train episodes；
2. pi2 相比 pi1 新出现的轻微偏离状态；
3. 保持 action outside fraction <10%，clean validation 不回归；
4. 仍用固定 held-out 完整闭环选择 pi3 checkpoint。

只有 Mustard 的 nominal q drift 和 4 指接触在多条 held-out 上继续稳定改善后，才进入
Mustard + capsule clean joint training；capsule recovery 暂不加入。

### 40.6 第三轮“全轨迹近邻定向采样”失败，pi3 不晋级

为覆盖 ep24/51，曾按 clean train 轨迹的 q、接触几何、局部速度和 planner 统计选择 20 条
近邻（不包含 held-out，也不重复 D1/D2），用 pi2 采集 D3。D3 本身质量看似很好：

```text
18 segments / 4874 frames / 563 new windows
D1+D2+D3 total DAgger windows = 742
normalized action outside [-1,1] = 3.19%
clean val q MAE = 0.001618 rad
```

但完整 held-out 暴露出明显退化：

| ep | pi2 q MAE | pi3 q MAE | pi2 4 指 | pi3 4 指 | pi3 max force |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.100 | 0.196 | 22.7% | 69.0% | 1178 N |
| 40 | 0.075 | 0.131 | 44.4% | 19.8% | 320 N |
| 51 | 0.177 | 0.240 | 43.7% | 37.9% | 1779 N |
| 57 | 0.161 | 0.189 | 47.4% | 26.5% | 1070 N |

pi3 在部分轨迹上学成“维持接触但离开 teacher task hand shape”的解；clean validation 完全
无法识别这一点。因此：

1. pi3 不晋级，当前主模型仍为 `mustard_nominal_dagger_round2_pi2_1k/best.pt`；
2. D3 文件保留用于诊断，但不加入下一版主训练；
3. “全轨迹统计近邻”不是合格的 DAgger state coverage 度量，不能继续沿用；
4. 下一轮采样必须按**偏离发生前的局部历史窗口**聚类/覆盖，并以 held-out 闭环即时淘汰
   checkpoint；不能按整条 teacher 轨迹的均值/方差挑邻居；
5. 在实现局部窗口 coverage 和自动闭环 checkpoint ranking 前，不进入 capsule 联训。

该负结果也验证了本节前面的关键结论：训练曲线、noise loss 和 teacher-forced q MAE 只能做
防回归检查，不能作为 autoregressive policy 的最终模型选择标准。

### 40.7 一次性冻结测试：pi2 的改善可泛化，但尚不均匀

由于 ep24/40/51/57 已被反复用于路线和 checkpoint 决策，它们正式降级为 dev set。另从原
validation split 冻结此前从未运行的 ep30/45/53/59，只进行一次 pi0/pi2 最终比较，之后
不再据此调当前模型：

| ep | policy | q MAE rad | q P95 rad | >=3 指 | 4 指 |
|---:|---|---:|---:|---:|---:|
| 30 | pi0 | 0.199 | 0.450 | 99.9% | 91.4% |
| 30 | pi2 | 0.251 | 0.601 | 97.9% | 83.9% |
| 45 | pi0 | 0.160 | 0.439 | 96.6% | 71.3% |
| 45 | pi2 | 0.154 | 0.404 | 96.8% | 26.2% |
| 53 | pi0 | 0.193 | 0.269 | 12.4% | 7.2% |
| 53 | pi2 | 0.209 | 0.520 | 98.4% | 62.2% |
| 59 | pi0 | 0.301 | 0.694 | 26.1% | 14.7% |
| 59 | pi2 | 0.100 | 0.214 | 62.5% | 33.4% |
| mean | pi0 | 0.213 | 0.463 | 58.7% | 46.2% |
| mean | pi2 | 0.178 | 0.435 | 88.9% | 51.4% |

这证明 pi2 的 nominal DAgger 改善并非只发生在 aggregation episodes：平均 q drift 和接触
均有泛化提升。但 ep30/45 的四指退化、所有策略仍可能产生数百到上千牛峰值，说明当前
模型还不具备可靠部署条件。冻结测试的职责到此结束，不应反复查看这四条再调 pi2。

## 41. DAgger same-time label 的 phase audit（2026-08-31）

新增 `scripts/audit_dagger_phase.py`。对 D1/D2/D3 每个已通过质量门控的 nominal state，在
同一 clean teacher episode 的局部 `t±50` simulation frames 中搜索最近状态，并比较：

```text
q             nominal q
q_tactile     nominal q + live/teacher contact point, normal, mask
full          q + tactile + local planner command
```

rollout 的 `source_episode_step` 是 episode 截取后的局部帧号，而 DP H5 的
`episode_step` 保留原轨迹帧号（Mustard 为 1200--3699）；审计已显式按 episode 内局部索引
对齐，不能直接比较两个字段。平坦区域用极小时间距离正则优先选择同时间帧；只有匹配距离
相对 same-time 至少改善 2% 才记为 identifiable phase。

### 41.1 完整距离结果

| 数据 | 全部 abs phase median/P95 | 全部 `abs>10` | identifiable `abs>10` | identifiable P95 | boundary |
|---|---:|---:|---:|---:|---:|
| D1 | 3 / 13 frames | 9.1% | 13.5% | 15 frames | 0.0% |
| D2 | 3 / 11 frames | 6.0% | 10.3% | 15 frames | 0.0% |
| D3 | 3 / 21 frames | 17.5% | 30.3% | 27 frames | 0.2% |

结论在 q-only、q+tactile、full 三组距离下方向一致；D3 的长尾显著更大，且分布在多数 D3
episodes，并非一两个长 segment 驱动。不同 episode 同时存在 lead 与 lag，因此不能用固定
全局时间偏移修复。该结果支持“D3 same-time supervision 含更多 phase conflict”。

### 41.2 不能把 pi3 退化全部归因于 phase

审计同时比较了 same-time 与 phase-matched teacher 的未来 H8（stride=5）q 标签。对
`identifiable && abs phase>10 frames` 的样本：

| 数据 | conflict samples | future q RMS median | P95 | joint max P95 |
|---|---:|---:|---:|---:|
| D1 | 116 | 0.00707 rad | 0.01479 | 0.06329 rad |
| D2 | 88 | 0.00550 rad | 0.01198 | 0.05157 rad |
| D3 | 708 | 0.00344 rad | 0.00761 | 0.02260 rad |

D3 的冲突样本数量/比例更高，但由于这些轨迹局部运动更慢，单个样本的 q label 冲突幅度
反而小于 D1/D2。因此 phase conflict 是可信嫌疑，不是已经建立的唯一因果解释。

下一步不能简单逐帧 `argmin` 后直接改 label：独立匹配会产生非单调、抖动的 teacher phase。
合格的 A/B 应对每个连续 segment 做带局部速度/单调性约束的 phase path（constrained
DTW/Viterbi），生成 `D3_phase_aligned`，然后从同一个 pi2 checkpoint、相同采样率和 seed
训练：

```text
same-time D3      -> 已知退化的 pi3 baseline
phase-aligned D3 -> causal comparison
```

在这个 A/B 完成前，不启动 D4，也不加入 execution-context 或 scheduled sampling，以免多个
变量同时变化而无法判断 D3 退化的真正原因。

## 42. 双轨数据 + DP 输出格式 A/B（2026-09-01，已授权，计划）

### 42.1 背景与动机

§31.3 确立的闭环失配机制：训练标签 = teacher 执行 q_live，但部署时 MCC 修改 desired →
DP 输出 ≠ 后续输入，自回归外推把"系统性偏差"当成了 task dynamics。结论：需要把
**意图（executed 前目标）**与**执行（executed 状态）**在数据表示上解耦：

- 输入（observation）：真实执行状态 q_live + tactile + planner（部署时可得）
- 输出（label / action）：执行前的意图目标

### 42.2 关键数据事实（已核实）

| 通道 | 采集端 | 部署端 | 备注 |
|---|---|---|---|
| 意图（执行前） | `q_ref`（QP 规划输出，= q_command_t） | DP 输出 q_prior → MCC | raw 文件均有 q_ref 通道 |
| 执行（执行后） | `q_hand`（力控后） | MCC 后 q_live | D1 现有 label |

- 原始 raw 文件里 `q_ref` vs `q_hand` RMS 差 0.0256 rad（强压世界，力控压出的意图-执行差）；
- `invert_trajectories.py:62` 只搬运 `q_hand`，`build_dp_samples.py:93` 把 `qref` 标为
  diagnostics only —— **D1 训练文件丢失了 q_ref 意图通道**；
- 原始 raw 文件均含 `tip_x_des_palm`（(4,3)）通道（collect_trajectories.py:4852），可用于
  Cartesian 意图标签。

### 42.3 双轨数据定义（obs 执行 / label 意图）

```
obs  (输入):  q_live 历史 + tactile(p,n,m) + planner 特征       ← 部署时可得
label(输出):  future 意图目标
```

### 42.4 DP 输出格式 A/B（同一双轨数据、同一 pi2 checkpoint 家族）

- **Variant A — absolute q_target**：label = future q_ref（16D 关节意图），MCC 直接以
  DP 输出为 desired；
- **Variant B — 法向/切向意图解耦 Δp_tangent**：label = palm-frame 指尖切向 delta
  （法向分量由 MCC 的力控拥有，F_n → 1.5N），DP 只输出切向意图；
- 训练后统一在 frozen ep30/45/53/59 上闭环评估（live q 历史 + MCC 叠加），
  与现有 baseline（§31.3：q_rms 0.417 rad，contact3 98.0% / contact4 84.4%）对比。

### 42.5 执行顺序

1. 从原始 raw 恢复 q_ref（及可选 tip 目标）通道，按 episode/step 与重采集 q_live 对齐，
   生成双轨训练文件（不含原始强压 q_hand 作为 label）；
2. 训练端支持两种 action 表示（absolute q_target / Δp_tangent）；
3. 部署端 MCC 接入两种输出格式；
4. A/B 训练两个模型（同一超参、同一数据、同一 seed 族）；
5. frozen 闭环评估 + 与 baseline 对比，登记 §30.4 验收判断。

> 并行状态：Step 2A 全量重采集（§34-35 链路，collect_executed 模式）139/239 完成中；
> 其产物 q_live 作为双轨数据 obs 输入，采集完成后立即导出。

### 42.6 方案修订（2026-09-01）：放弃 1.5N MCC 重放，直接用原始 239 条轨迹（已执行）

**变更**：Step 2A 的 1.5N MCC 重采集（collect_executed）产物不再作为双轨数据 obs 输入，
obs 直接用**原始采集时代执行轨迹 q_hand**（= D1 同一分布），label 从原始 raw 恢复意图通道。

**原因（实测）**：pilot 重采集 170 集 + 检查 ep0-3 后发现 4x3700 强压批次在 1.5N MCC
下重放失稳——ep0-3 的 q_live 偏离 teacher 0.115 rad（rms），拇指关节被推到 2.19 rad
（teacher 全域最大仅 0.911），tip 偏离 120 mm，force 峰值 500 N+（ep30 等正常集仅
0.004 rad / 3 N）。根因：这批 cap-top-to-body 轨迹拇指速度是其他轨迹 4.5 倍
（0.00168 vs 0.00037 rad/frame），1.5N 弱力控 + 接触搜索在快速强压段把拇指推飞。
此类 rollout 不是合法补偿分布，排除后 239→235 集且重采大概率复现失败。

**新定义（v2 文件）**：

```
obs  (输入):  原始采集时代执行 q_hand + tactile(p,n,m) + planner 特征  ← 与 D1 同分布
label(输出):  Variant A = q_ref (16D 关节意图)   /   Variant B = tip_delta_tangent_palm (12D)
```

- 生成脚本：`scripts/export_dual_track.py`（v2 版）——obs 块直接调
  export_palm_dp.export()（与 D1 完全同一管线，obs 分布逐字段一致），意图通道按
  source_trajectory 从原始 raw 恢复 q_ref / tip_x_des_palm，逐 episode 对齐校验
  （inverted q_hand ≡ raw q_hand，rms < 1e-4）；
- **产物**：`data/dp/mustard_v1_239_dual_track_palm_v2.h5`（239 集 / 597500 帧，已导出验证）：
  - q_ref vs q_hand RMS 中位数 0.0235 rad（力控压出的意图-执行差，全 239 集 < 0.05 rad）；
  - tip_delta_tangent_palm：接触帧 99.9% 非零、无接触帧全零（约定：法向归 MCC）；
  - attrs：`action_field=q_ref / action_dim=16 / obs_q_source=original_collection_executed_q_hand`；
- 后续 A/B 只切 `action_field`：A=q_ref(16D)，B=tip_delta_tangent_palm(12D，需训练端参数化)。

**后续步骤**（42.5 不变）：训练端 12D 参数化 → 部署端两格式接入 → A/B 训练 →
frozen ep30/45/53/59 闭环评估 vs baseline。

### 42.7 训练端与部署端两格式接入完成（2026-09-01）

**训练端**（`train_dp.py` / `dp_dataset.py`，已完成并 smoke 验证）：

- 新增 `--action-field`（q_ref | tip_delta_tangent_palm）与 `--action-dim`（16 | 12）
  覆盖；未传时从数据集 attrs 读取（v2 文件 attrs 已声明 q_ref/16）；
- 12D 只允许 absolute_q 表示（normal 分量不参与学习），`kinematic_residual_q`
  分支在 action_dim≠16 时 raise；
- `overfit_metrics` 12D 时跳过 joint-space baseline（hold_current/kinematic → NaN）。

**部署端**（`deploy_dp_inverse.py`，已完成并 smoke 验证）：

- `DPRuntime` 从 checkpoint 读取 `action_dim`/`action_field`（train 写入顶层键），
  校验 12D ⇔ tip_delta_tangent_palm；
- **Variant A（q_ref 16D）**：absolute_q 输出直接作为 desired 进入既有
  MCC 链路 —— 部署代码零分支，仅换 checkpoint；
- **Variant B（tip_delta 12D）**：`_plan`/`offline_teacher` 走
  `tip_delta_to_absolute_q()` 重建：

  ```text
  tip_des[h] = FK(q_base) + delta[h]            （normal 分量由 MCC 力控拥有）
  q_des[h]   = solve_fingertip_targets(tip_des[h], seed=q_des[h-1])
  ```

  逐 waypoint 用前一步解作 seed（resolved-rate IK 兼任时间分支选择，与
  离线约定一致），重建后的 16D chunk 进入与 Variant A 完全相同的
  precontact/track → MCC update 链路。

- 验证：smoke 100 步两模型均训练成功（A: train_mae 0.362 rad；B: train_mae
  0.0145 rad）；offline_teacher ep30 端到端跑通（A: hold_q 0.0086 rad；
  B: FK+IK 重建执行正常）。

**A/B 正式训练**（运行中）：pi2 超参族（batch 256 / lr 1e-5 / DDPM 100 /
down [256,512,1024] / kernel 5 / groups 8 / embed 128 / seed 20260831 /
val_ratio 0.1 / dropout 0.1@3），同一 dual-track v2 数据，仅切 action：

```text
pred_horizon=16（=planner 72 帧 / stride 5 的必要值），stride=5，obs=16
Variant A: data/models/mustard_v1_239_dual_track_variant_a_qref_5k
Variant B: data/models/mustard_v1_239_dual_track_variant_b_tipdelta_5k
5000 步 / save-every 1000（闭环评估可选取多 epoch）
```

**下一步**：frozen ep30/45/53/59 闭环评估（live_dp + fullhand_mcc + mcc-preset
current 冻结 1.5N，dp-history-q nominal），vs §31.3 baseline（q_rms 0.417 rad，
contact3 98.0% / contact4 84.4%），登记 §30.4 验收判断。

### 42.8 A/B 闭环评估与部署链诊断阶梯完成（2026-09-01）

**frozen 闭环评估**（live_dp + fullhand_mcc + mcc-preset current 冻结 1.5N，
dp-history-q nominal，seed 42，ep30/45/53/59，best.pt）：

| 模型 | q_mae | contact3 | contact4 | force_max |
|---|---:|---:|---:|---:|
| Variant B (tip_delta 12D) | 0.370/0.474/0.458/0.548 | 85.4/94.6/90.3/99.3% | 19.8/65.6/49.2/39.6% | 2451/1283/1552/1445 N |
| Variant A (q_ref 16D) | 0.077/0.103/0.165/0.114 | 64.8/80.2/89.2/84.6% | 29.1/60.8/47.3/66.4% | 55/640/1873/1006 N |
| §31.3 baseline | 0.417*(q_rms) | 98.0% | 84.4% | — |

两者闭环均显著劣化：contact4 远低于 baseline，且出现数百~千牛顿级力尖峰
（1.5N 冻结力控设定下为物理失控）。

**部署链诊断阶梯**（同协议，逐层隔离）：

- **S0** `teacher_dp --teacher-action-source recorded`（执行记录 teacher q，
  DP 推理不执行）：4 集 q_mae 0.005–0.008 rad、force_max ≤2.75N、contact4
  71–90% —— **控制器/执行栈正常**；
- **S1** `teacher_dp`（B 部署链 + teacher 观测，无闭环漂移）：4 集 q_mae
  0.020–0.038 rad、force_max ≤4.03N、contact4 73–84% —— **FK+IK 部署链与
  DP 预测（训练分布内）正常**；
- **S2** live_dp（B）：力 1283–2451N、contact4 20–66% —— 崩。

**根因定位：live 闭环观测漂移**。DP 预测误差（dp_horizon_mae）从 teacher 观测
的 0.047 rad 放大到 live 观测的 0.608 rad（12.8×，B），A 亦放大 4×（0.193）。
机制与 §31.3 一致：模型从未见过 q_live≠q_nom 状态、无输入可辨别任务运动与
低层补偿，live 观测分布偏移 → 意图预测失真 → 执行失真意图 → 过力/接触丢失 →
观测进一步偏移的恶性循环。B 比 A 更敏感（FK(q_base=nominal) 基准随漂移错位）。

**判定**：A/B 意图解耦实验在训练/离线层面成立（A 未超 hold-q baseline 属
离线局限；B 离线 0.77mm vs zero-delta 2.74mm），但两者闭环均不达标
（§30.4 未通过）。下一步不修部署链（S1 证明链路正常），而是按 §31.3 结论
解决观测分布：训练输入显式拆分 q_nom / q_live / Δq_comp，并引入
simulator perturbation rollout / DAgger 的闭环扰动恢复样本。

**同场修复**：train_dp.py overfit_metrics 12D 分支由 NaN 改为 zero-delta
hold-tip baseline（误差=期望位移幅值，kinematic 与之重合）；训练曲线标签按
action_unit（rad/m）区分；evaluate_dp_checkpoint.py 读取 checkpoint 顶层
action_dim/action_field，支持 12D 评估。

**数据契约硬障碍（2026-09-01 复核）**：Variant A 的"干净意图"无法从旧数据
恢复。全部 239 个 episode 的 `q_nominal` 通道只是 `planner_h5["q_hand"][0]`
的静态 (1,16) 快照（`collect_trajectories.py` 经 write_static 写入，
per-episode 常量），而 MCC 前真实命令链（closure_target_q →
hybrid_grasp_q_ref α-smoothing → contact_qp → update()）没有逐帧关节位置
记录。旧 `q_ref` 字段是 `update()` 输出后的命令（含力控补偿），不是纯 QP
意图。**Variant A 重训必须重新采集**（部署时记录同周期 q_prior/q_cmd/q_live，
见 §43.1）；Variant B 安全，因为 `tip_x_des_palm`（逐帧）在力控修正前计算。

**MCC 参数对齐修复（2026-09-01，部署侧 deploy_dp_inverse.py）**：对 16 项
采集/部署 MCC 参数逐一核对后修复 2 个真实错配——① update() 调用
`contact_observed` 由 `live_found` 改为 `live_found & (||F||>=0.05N)`
（与采集 `loaded` 语义一致）；② 新增 `natural_flexion_floor=-0.30` 参数
（与采集的 manifold flexion 下限一致），经 CLI `--mcc-natural-flexion-floor`
传入 FullHandMCC 构造。修复后重跑 S2_ORACLE：首帧力尖峰逐集完全相同
（473/1431/1236/1381N，确定性 bootstrap→track 行为，与 MCC 参数无关），
hmae 混合（ep53 0.154 改善 2.2×，ep45 0.386 恶化），found=4 由 65–87%
降至 16–44%（contact_observed 更严格）。**结论：MCC 对齐是必要 hygiene 但
非充分条件，瓶颈在 DP 输入结构**——由 §43.4 的 G(δ) 审计定量证实。

## 43. 双轨部署协议纠偏与因果阶梯复核（2026-09-01）

### 43.1 当前 v2 并非最初定义的完整双状态模型

复核数据、采集端和部署端后确认，§42 的实现把“双轨”简化成了：

```text
input  = q_live + tactile + planner
label  = q_ref（A）或 tip_delta_tangent_palm（B）
```

它只分开了 observation 与 label，网络输入中并没有同时存在 `q_prior` 和
`q_live`，也没有 `q_live-q_prior`、`delta_q_comp` 或 `e_servo`。部署参数
`--dp-history-q-source nominal|live` 仍是二选一，而不是双轨并存。

采集语义还存在第二个问题：`collect_trajectories.py` 当前在
`FullHandMCCController.update()` 和 persistent recovery 完成后，才把
`q_command_t` 写入 debug 字段 `q_ref`。因此旧文件的 `q_ref` 实际是低层控制器修改后的
命令，不是严格的 MCC 前任务意图。Variant A 不能被解释成纯 `q_prior` 教师标签。

完整双轨必须在同一个控制时刻保存并校验：

```text
q_prior                 MCC 前任务意图
q_cmd                   q_prior + 低层修正后的下发命令
q_live                  物理执行结果
delta_q_comp = q_cmd - q_prior
e_servo      = q_live - q_cmd
e_total      = q_live - q_prior = delta_q_comp + e_servo
```

并显式加入双轨运动信息：`qdot_prior`、`qdot_live` 和
`e_qdot=qdot_live-qdot_prior`。平滑加速度只作为后续消融项，不能直接使用逐帧
`qacc` 尖峰。

### 43.2 重新设计的因果阶梯

`scripts/eval_dual_track_ab.sh` 现作为唯一入口；旧
`run_diagnostic_stairs.sh` 仅为兼容包装。当前旧 checkpoint 的诊断顺序为：

```text
S0          recorded teacher action + collection-matched MCC
S1          teacher q/tactile + DP action
S1_SENSOR   teacher q + live ContactSensor tactile
S1_ORACLE   teacher q + actual solver support + source-mesh normal
S2_ORACLE   live q + matched MCC + source-mesh normal
S2_SENSOR   live q + matched MCC + ContactSensor normal
S3          仅在 S2 通过后更换 current MCC 参数
```

这样分别隔离 action/FK-IK、触觉几何、真实状态反馈和 controller transfer，禁止把
多个变化合并成一个 `live_dp` 结果。双轨 checkpoint 在 `live_dp` 中默认禁止
`nominal` history；必须显式传入仅供历史复现实验的 override 才能运行。

### 43.3 ep30 800 帧 CUDA 短测

Variant B、chunk replan=10、inference=100 的结果：

| stage | q MAE | q P95 | >=3 指 | 4 指 | force P95 / max |
|---|---:|---:|---:|---:|---:|
| S0 | 0.0028 | 0.0035 | 100.0% | 97.7% | 4.8 / 5.7N |
| S1 | 0.0097 | 0.0127 | 100.0% | 95.2% | 7.3 / 16.2N |
| S1_SENSOR | 0.0292 | 0.0634 | 96.1% | 79.7% | 16.3 / 27.3N |
| S1_ORACLE | 0.0128 | 0.0229 | 99.9% | 93.4% | 8.7 / 18.6N |
| S2_ORACLE | 0.2811 | 0.4739 | 90.1% | 61.7% | 59.4 / 72.1N |
| S2_SENSOR | 0.2806 | 0.4386 | 92.3% | 66.2% | 370.2 / 824.7N |

这组结果给出两个相互独立的结论：

1. `S1_SENSOR -> S1_ORACLE` 恢复了大部分 teacher-conditioned 性能，说明 live
   solver normal/point 与训练 source-mesh tactile 的域差异确实很大；后续应采用
   actual-contact-anchored 的 source-mesh local lifting，而不是独立 SDF 自行触发接触。
2. `S2_ORACLE` 仍然失败，证明 tactile normal 不是闭环失败的唯一根因。frame 85 时
   live q 仅偏离 teacher 约 0.012rad；frame 90 的 DP target error 已由 S1 的
   0.032rad 放大到 0.078rad，frame 100 达 0.129rad。当前 policy 对小执行偏差是
   expansive mapping，不是局部收缩映射。

因此 §42 旧模型只保留为诊断 baseline。真正下一版不能只在部署时切换
`nominal/live`；必须修复采集语义、导出完整双状态与差分量、重新训练，再按同一阶梯验证。

### 43.4 G(δ) 局部灵敏度审计：thin-manifold 假设被定量证实（2026-09-01）

`scripts/audit_local_sensitivity.py`：在 val 集 256 个窗口上，对每窗口的
q 通道施加 16D 随机单位方向扰动 δ∈{0.003,0.005,0.01,0.015,0.02} rad。
同一窗口的基线输入与所有扰动输入共享同一个 diffusion 采样 seed（批量
`conditional_sample`），隔离纯输入灵敏度。每窗口 4 个随机方向取平均。

两种扰动放置（`--perturb-mode`）：

```text
last  仅末帧 q 偏移，历史保持在 teacher 轨迹上 —— 闭环信息结构：
      手指此刻被外力推离轨迹、接触几何不变，模型必须产出纠偏意图
all   所有帧整体平移 —— 绝对 q 的 gauge 方向对照
```

指标：`G(δ)=||π(o+δu)−π(o)||/δ`（输出漂移 / 关节偏差），
`R(δ)=||π(o+δu)−a*||/||π(o)−a*||`（目标误差放大比）。

**Variant B**（tip_delta_tangent_palm 12D，G 单位 m/rad；base_err p50=4.5mm）：

| δ(rad) | G p50 | G p90 | R p50 | R p90 | R p99 |
|---|---:|---:|---:|---:|---:|
| 0.003 | 1.20 | 1.86 | 1.06 | 1.57 | 1.77 |
| 0.005 | 0.74 | 1.08 | 1.14 | 1.50 | 1.64 |
| 0.010 | 0.36 | 0.54 | 1.08 | 1.42 | 1.65 |
| 0.015 | 0.24 | 0.35 | 1.05 | 1.57 | 1.67 |
| 0.020 | 0.18 | 0.27 | 1.04 | 1.33 | 1.55 |

**Variant A**（q_ref 16D，G 无量纲 rad/rad；base_err p50=0.100 rad）：

| δ(rad) | G p50 | G p90 | R p50 | R p90 | R p99 |
|---|---:|---:|---:|---:|---:|
| 0.003 | 26.8 | 34.7 | 1.00 | 1.26 | 1.48 |
| 0.005 | 16.1 | 21.2 | 1.02 | 1.25 | 1.46 |
| 0.010 | 8.0 | 10.4 | 1.01 | 1.22 | 1.51 |
| 0.015 | 5.4 | 7.0 | 1.02 | 1.25 | 1.43 |
| 0.020 | 4.0 | 5.2 | 1.01 | 1.25 | 1.38 |

结论：

1. **A 模型 G(0.003)=26.8 >> 1**，直接命中 thin-manifold 判据
   （G(0.003)>1 且 G(0.005)>1）；与闭环观测定量吻合——§43.3 实测
   0.012 rad 执行偏差放大到 0.078 rad 目标误差（≈6.5×），G(0.01)=8.0
   的预测值 0.096 rad 为同一量级。局部数据覆盖不足/映射不稳被证实。
2. **G 随 δ 单调衰减**（A: 26.8→4.0；B: 1.20→0.18）：模型在 teacher 点
   邻域存在高增益内核、超出后响应饱和，是"只在流形上见过数据"的典型形态。
   响应最陡区间为 0.003–0.01 rad。
3. **B 模型** G≈1.2 m/rad：0.003 rad 偏差产生 ~3.6mm 输出漂移，与单步
   意图幅度（几 mm）同量级；且 `last`/`all` 两模式增益几乎相同——模型
   无法利用 q 通道区分"当前偏差"与"整体平移"，任何 q 扰动都剧烈改变输出。
4. 尾部窗口 R p99 达 1.4–1.9×，存在少量对扰动极敏感的窗口。

行动指引：训练数据需在 teacher 轨迹周围注入 physics-consistent 执行扰动
（perturbation tube），**优先覆盖 0.003–0.01 rad**（模型增益最陡区间），
扩展到 0.02 rad 使响应区被数据撑大；采集端同时按 §43.1 记录同周期
`q_prior/q_cmd/q_live`，训练端把 q 通道与意图通道拆开（B0/B1/B2 消融）。

## 44. 随机化双轨数采准备（2026-09-02）

### 44.1 信息边界

最终 DP + MCC 不允许读取物体类别、YAML `pregrasp_q`、真实物体位姿、未来轨迹
几何或未接触表面的距离/最近点。由于凸分解接缝会污染 ContactSensor 法向，允许一个
严格受限的触觉仿真例外：

```text
MuJoCo actual fingertip contact found
    -> use that actual contact position as the query anchor
    -> query original high-resolution surface/SDF local normal
    -> provide the corrected normal as fingertip tactile observation
```

该 oracle 不能自行触发接触、不能在失触时搜索表面，也不向 DP/MCC 输出 SDF 距离、
物体姿态或未来几何。当前实现后端是 `MeshNormalOracle` 的原 mesh 局部 PCA 法向，
还不是真正的体素 SDF 梯度；以后可替换后端，但必须保持 actual-contact-anchored 接口。

部署端同时撤销了从 object YAML 读取 `pregrasp_q` 的改动：四指首次稳定接触后只用
实时 `q_live` 建立协同回收参考。解析 capsule oracle、source-surface 曲率/距离和
teacher tactile replay 仍属于显式诊断，不是可部署输入。

### 44.2 采集/部署执行栈审计

Mustard inverted H5 中记录的物理契约为：contact `solref=(-20000,-400)`、
`solimp width=0.002m`、10 physics substeps（1ms）、手指位置执行器
`stiffness/damping/effort=35/2.5/35`、force integral gain `0.003`。部署 replay 已
对齐这些参数，并改成与采集相同的单个 `netforce` ContactSensor 同时产生
found/force/point/normal，启动时 `audit_collection_execution_contract()` 会核对 H5 attrs。

不能照搬且有明确理由的 teacher-only 层为：物体 source geometry surface target、
接触流形 QP、persistent surface recovery、物体专属初始预抓握。部署时这些由 DP 的
task prior + 接触触发式法向 MCC 替代，不能以“参数对齐”为由重新引入特权信息。

### 44.3 新原始 H5 三轨契约

`collect_trajectories.py` 现在同周期记录：

```text
q_prior(t)       teacher planner/QP output before FullHandMCC
q_cmd(t)         final relative-position target actually applied this step
q_live(t+1)      post-physics encoder state (legacy field q_hand)
delta_q_comp(t)  q_cmd(t) - q_prior(t)
e_servo(t+1)     q_live(t+1) - q_cmd(t)
execution_perturbation_q(t)
```

H5 `schema_version=mcc_tip_dual_track_v2`，attrs 明确 pre/post-step 时序。旧 `q_ref`
继续保留用于兼容，但不得再作为纯 task intent；新训练必须使用 `q_prior`。

### 44.4 物理一致执行随机化

新增 `--execution-randomization`。随机量不是直接篡改 tensor observation，而是在
FullHandMCC 后对最终 16D 位置目标加入平滑相关扰动，再由普通 actuator + MuJoCo
产生自洽的 q_live/触觉。扰动向量范数默认 log-uniform 覆盖 `0.003--0.020rad`，
时间常数 `0.35s`、50 帧渐入、25% 并行环境保持 clean control。扰动不直接污染
`q_prior`；后续帧允许高质量 teacher 根据真实偏差作出恢复，因而 label 是成功恢复
动作而不是失败 rollout action。

2-env CUDA smoke（s602 路径开头 300 记录帧）验证：

- 三轨恒等式最大误差 `3.7e-9rad`；
- active env 实际扰动范数 `3.30mrad`，clean env 为 0；
- 两者 `>=3指=100%`，all4=`94.7%/94.0%`；
- 此短测只含完整 2500 帧路径的初始化 300 帧，拇指短失触占比被放大；旧完整 s602
  物理采集 all4=`99.96%`，正式质量判断必须跑完整 2500 帧并离线筛选 `>=98%`。

`batch_collect_and_filter.py` 已可透传随机化参数；正式启动前先用 16 条完整多样化
plan 做 pilot，分别统计 clean/3--10mrad/10--20mrad 三档的 all4、最长失触、
`||q_live-q_prior||` 和手型健康，不能直接无审计地重采全部 231 条。

反演链同步修复：`invert_trajectories.py` 的 selected-raw bundle 白名单现会在所有
输入都具备时保留三轨与随机化字段。2-env smoke 反演得到 600 帧/2 episodes，
`q_prior/q_cmd/q_hand/delta_q_comp/e_servo` 均为 `(T,1,16)`，两条恒等式反演前后
误差为 0；法向仍统一为 palm-DP 所用的 `primary_fingertip_to_object` 约定。正式
训练前还需从 inverted object-frame 数据走既有 palm-frame DP exporter，不能直接训练 raw。

### 44.5 16 条完整 pilot 结果（2026-09-02）

完整随机化 pilot 已采集到：

```text
data/trajectories/mustard_randomized_dual_track_pilot16/
```

这批包含 16 条不同 plan（14 个不同方位/形状 family，另含 2 个同 family 不同 seed），
每条记录 2500 帧；4 个环境为 clean，12 个环境施加 3--20mrad 平滑执行扰动。
离线筛选结果：

```text
selected                         16 / 16
all4 contact min / mean / max   99.92% / 99.958% / 100.00%
minimum single-tip rate         99.92%
longest all4 loss run           1 frame
q_hand max single-step          0.00558 rad
```

因此扰动没有破坏 teacher 成功性，且远高于 98% 筛选门槛。它也不是只存在于 actuator
命令中：与同名无扰动旧轨迹比较，clean 组 `q_live` 差异 P95 平均仅 `0.90mrad`，
3--10mrad 组为 `22.71mrad`，10--20mrad 组为 `21.55mrad`。这表明执行扰动经过
MuJoCo 接触闭环后确实产生了新的 live state/tactile，并触发 teacher 的后续恢复，
形成了需要的 successful perturbation tube。

三轨数据一致性：

```text
max |(q_cmd-q_prior)-delta_q_comp| = 0
max |(q_live-q_cmd)-e_servo|        = 0
contact normal norm median          = 1.0
normal norm absolute error P99      < 6e-8
```

力不是当前训练标签/筛选主目标，但用于排除数值爆炸：全数据 fingertip force P95
`16.49N`、P99 `25.76N`、最大 `47.96N`；同时关节单步变化仍很小，没有出现力导致的
姿态跳变。后续完整采集可以保持当前 `0.003--0.020rad / 25% clean / tau=0.35s`
设置，不需要因接触质量下调扰动。

完整 `passing_plans.txt` dry-run 在当前 `rotation>=30deg / posture<=0.90rad`
门控下得到 `227/256` 条，分为 14 个 16-env batch 和 1 个 3-env batch。dry-run
同时发现批处理原先给每个 subprocess 重复使用同一个 seed，会令相同 env 槽位在各批
重复 clean assignment、扰动幅值和相关方向。现已改为
`batch_seed = base_seed + 104729 * batch_id`，保持可复现且让不同批次覆盖不同执行扰动。

注意：旧 `export_dual_track.py` 仍以历史 `q_ref` 作为 intent 标签。新随机化数据训练前
必须改为直接使用反演后保留下来的 `q_prior/q_cmd/q_hand/delta_q_comp/e_servo`，不得
把兼容字段 `q_ref` 再解释为纯 task intent。

### 44.6 完整采集前 80 条的随机化覆盖审计（2026-09-02）

正式目录当前完成 batch 000--004，共 80 条且全部进入 selected；采集进程在 batch 005
开始处停止。20 条 clean、60 条 active，active 目标范数分布为：

```text
min / P10 / P25 / P50 / P75 / P90 / max (mrad)
3.07 / 3.59 / 4.96 / 7.74 / 12.36 / 16.12 / 19.20
```

其中 38 条位于旧 policy 增益最陡的 3--10mrad 区，22 条位于 10--20mrad 扩展区。
扰动方向协方差 effective rank=`15.97/16`，各关节 RMS 最大/最小仅 `1.026`，说明
16 个关节方向没有明显遗漏。不同 batch 的 seed 已确认分别为 base seed 加质数步进，
幅值/clean 槽位没有跨批重复。

与同名旧无扰动轨迹逐帧比较，物理执行状态变化为：

```text
target group       q_live difference P50 / P95 / mean-max
clean              0.11 / 1.24 / 8.09 mrad
3--10mrad          5.54 / 14.16 / 27.85 mrad
10--20mrad        13.35 / 33.74 / 54.02 mrad
```

`q_prior` 相对旧 teacher reference 的 P95 也分别达到约 `15.4/33.3mrad`，证明
teacher 不是在扰动下机械复制原标签，而是在生成对应的成功恢复意图。真实 contact point
相对旧轨迹 P95 平均变化 `0.45mm`，法向 P95 平均变化 `0.85deg`；四指接触平均
`99.905%`、最低 `99.36%`、最长失触 4 帧、最大 q 单步 `0.01124rad`。因此它是
保持成功接触的局部扰动管，而不是失败/掉触数据。

时间相关性实测 direction cosine：10ms=`0.97`、100ms=`0.74`、350ms=`0.36`、
1s=`0.052`。当前 `tau=0.35s` 能覆盖短时执行扰动，但不能等价于持续数秒的 DP 系统性
bias。第一版仍应保持当前设置完成 227 条并训练/做局部增益与 live 闭环复测；如果仍出现
慢漂移，正确补充项是小规模 `tau=1.5--3s` 或 piecewise-constant structured bias，
而不是把幅值扩大到已经失败的 0.2--0.4rad。

当前 80 条只覆盖按文件排序靠前的 4 个轨迹 family
（f0.18/f0.25/f0.28/f0.32），可以判断执行随机化是否有效，但不能代表完整表面/任务
多样性；需用 `--resume` 完成剩余 147 条。训练采样还应维持约 25% clean / 75% active，
不能无权重地叠加全部旧 clean 数据而稀释 perturbation tube。

### 44.7 v3 双轨重训与 G(delta) 冻结判别（2026-09-02）

227 条正式 randomized 数据已经导出为：

```text
data/dp/mustard_randomized_dual_track_v3_227.h5
```

这次没有混入缺少逐帧双轨通道的旧 239 条数据。episode 重新编号为 `0--226`，
seed=`20260831` 的分层划分为 204 train / 23 val；clean 为 51/6，perturbed 为
153/17。导出前后的关键契约审计为：

```text
episodes / frames                 227 / 567500
clean / perturbed                 57 / 170
state dimension                   242
max |delta_q_comp-(q_cmd-q_prior)| 0
max |e_servo-(q_hand-q_cmd)|       0
qvel vs causal finite difference   cosine=0.98524, RMSE=0.00788 rad/s
tip pose recorded vs FK            < 0.2 micrometre
```

输入 schema `contact_geometry_planner_motion_dual_track_v3` 的固定顺序为：

```text
q_prior 16
contact position / normal / mask  12 + 12 + 4
q_prior_velocity                  16
contact point velocity/rate       12 + 12
q_hand / q_live_velocity/e_qdot   16 + 16 + 16
delta_q_comp / e_servo            16 + 16
palm_twist / planner_delta         6 + 72
```

总计 242D。主 action 是 12D `tip_delta_tangent_palm`；同文件也保存 16D
`q_prior` 绝对意图作为后续消融备用。主标签严格复用 v2 Variant B 定义：

```text
project_tangent(tip_x_des_palm[t+1] - tip_actual_palm[t], normal[t])
```

无接触帧标签置零。有效标签范数 P50/P90/P95/P99 分别为
`6.34/11.67/14.27/22.03mm`，最大 `48.02mm`。

训练完成两个共享 242D 网络形状的 5k 模型：

| model | active input | val MAE | clean val | perturbed val | val noise loss |
|---|---:|---:|---:|---:|---:|
| B2 | 242D 完整双轨 | 1.067mm | 1.092mm | 1.033mm | 0.02164 |
| B0 | 84D 执行观测 mask | 1.122mm | 1.144mm | 1.052mm | 0.02092 |

模型目录：

```text
data/models/mustard_randomized_dual_track_v3_B2_tipdelta_5k/
data/models/mustard_randomized_dual_track_v3_B0_tipdelta_5k/
```

B2 的离线 MAE 略优于 B0，也略优于旧 v2 B 的约 `1.109mm`，说明完整意图流没有
损害 teacher-manifold prediction。但冻结的 G(delta) 判据未通过。256 个 val window、
4 个固定随机方向、相同 diffusion seed 下，delta=`0.003rad` 的结果为：

| model | perturb placement | G p50 | G p90 | R p50 | R p90 | R p99 |
|---|---|---:|---:|---:|---:|---:|
| old v2 B | last | 1.200 | 1.860 | 1.06 | 1.57 | 1.77 |
| v3 B2 | last | 0.989 | 1.383 | 1.04 | 1.31 | 1.72 |
| v3 B2 | all | 0.989 | 1.391 | 1.04 | 1.32 | 1.72 |
| v3 B0 | last | 1.114 | 1.607 | 1.02 | 1.38 | 1.69 |
| v3 B0 | all | 1.113 | 1.606 | 1.02 | 1.38 | 1.69 |

B2 相对旧 B 的 `G(0.003)` 只下降约 17.6%，没有达到预先冻结的 `<0.30` 或下降
至少 70% 标准；`R p99` 也没有达到 `<1.2`。因此按验证协议，没有继续花费约一小时
运行 ep30/45/53/59 的 S2_ORACLE。这里的“未做闭环”是门控的预期行为，不是实验遗漏。

last/all 几乎完全相同是比平均 MAE 更重要的结果：当前模型没有把“仅当前帧发生的
执行偏差”与“整段历史共同偏移”解释成两种不同的时间状态。随机化数据改善了普通
插值精度和 G p90，但没有学成局部收缩恢复场。B2 优于 B0 说明意图/补偿双轨有帮助，
但仅靠把这些通道拼进 condition 不足以保证网络使用其因果关系。

下一轮不能简单延长同一 5k 训练，也不能直接进入大规模 S2。优先诊断/修复顺序为：

1. 在 227 条数据上统计同一 planner phase 附近 `e_servo` 变化与
   `tip_delta_tangent_palm` 纠正方向的条件覆盖，确认是否存在足够的近似配对样本；
2. 对 G 审计增加“真实运动学一致”扰动：同时由 FK/接触重算 tip/contact geometry，
   避免仅修改 q_hand/e_servo 而保持 tactile 不变形成不可能 observation；
3. 若配对覆盖不足，采集同一 plan/phase 的多扰动 seed 或构造小规模 teacher recovery
   pair，而不是继续增加全新 plan；
4. 在 loss 中加入 paired consistency/contraction 约束，显式要求局部执行误差对应的
   输出变化方向，而不能期望 conditional U-Net 从相关数据中自动分离该关系；
5. 新模型先复测同一 G(delta)；通过后才恢复 S2_ORACLE 四集闭环门控。

机器可读 G(delta) 全表保存在两个模型目录的 `G_delta_last.json` 与
`G_delta_all.json` 中。

### 44.8 single-sample vs mean-of-K 判别：分布级收缩已达标，失败在采样端（2026-09-02）

§44.7 的 G(delta) 判据是在"每输入单次采样"下测量的（与部署 replan 的一次性
行为一致）。一个待定问题：`G(0.003)=0.99` 究竟反映"条件分布只学中心/无收缩"，
还是"条件分布本身在收缩、但单次采样噪声主导了点级增益"。为区分两者，
`audit_local_sensitivity.py` 增加 `--samples K`：每个 (window, 扰动输入) 在
同一条 generator 流中独立采 K 条，输出两组指标：

```text
level 0  single-sample    = 第一条独立样本（部署式 one-shot 行为）
level 1  mean-of-K        = K 条独立样本平均（分布级条件均值响应）
```

修复过程中发现并解决一个 8GB 卡 OOM：单次 batch 336（21 输入 × K=16）直接
超显存，分块到 16 行后仍崩溃且存活分配稳定在 ~5.95GiB 与块大小无关。根因是
checkpoint 参数 `requires_grad=True`，每次 `conditional_sample` 都会构建
autograd graph（~2.3GiB）；而 Python 先求值 RHS 再赋值，第 k+1 个 chunk 的
forward 执行时第 k 个 chunk 的 graph 尚未释放，两个 graph 重叠即爆。在 chunk
循环外包 `torch.no_grad()` 后 graph 逐块即时释放，运行通过。诊断脚本见
`/tmp/diag_mem*.py`（不入库）。

128 个 val window、4 方向、last 扰动、K=16 的结果（JSON 见两模型目录
`G_delta_samples16.json`）：

| model | level | G p50 @0.003 | G p90 @0.003 | R p50 | R p90 | R p99 | base_err p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 B2 | single-sample | 0.971 | 1.365 | 0.99 | 1.35 | 1.91 | 3.96mm |
| v3 B2 | mean-of-16 | 0.245 | 0.335 | 1.01 | 1.09 | 1.19 | 3.56mm |
| v3 B0 | single-sample | 1.093 | 1.529 | 1.01 | 1.39 | 1.93 | 3.91mm |
| v3 B0 | mean-of-16 | 0.281 | 0.389 | 1.01 | 1.10 | 1.22 | 3.55mm |

single-sample 行与 §44.7 记录的 0.989 / 1.114 一致（同一语义），旧 audit 的
结论可以原样引用。新信息来自 gap：

1. **分布级收缩达标**。B2 条件均值响应的 `G(0.003)=0.245` 低于冻结的 `<0.30`
   标准，`R p99=1.19` 也低于 `<1.2`；B0 为 0.281 / 1.22，G 达标但 R p99 临界
   超限。§44.7 "没有学成局部收缩恢复场"的判断需要修正为：**条件均值层面已经
   收缩，单样本链路没有收缩**。模型并不是"只能预测中心"——条件分布非零宽度，
   而 mean-of-K 证明了条件均值对离流形扰动有正确的方向响应。
2. **gap 约 4.0× = 采样噪声主导单样本增益**。由 single base_err（0.00434m）
   与 mean base_err（0.00376m）之差反推单样本条件标准差约 2.2mm/步（12D 范数）；
   而 δ=0.003rad 的确定性条件均值位移仅 `0.245×0.003 ≈ 0.74mm`，单样本增益被
   噪声完全淹没。G p90 同样压缩约 4×（1.37→0.34）。
3. **B2 与 B0 同型**，说明该现象与双轨条件流是否完整无关，是 diffusion one-shot
   推理的通有特性。

据此，§44.7 诊断顺序需要重排：在投入配对数据/一致性损失（第 3/4 步）之前，
应先做低成本验证——**deploy 端每个 replan 采 K≈16 条取均值再执行**（噪声降
√K≈4×，直接把单样本增益压回分布级水平），然后在同一 242D B2 上重测
S2_ORACLE 四集门控。若 averaging 闭环仍失败，才回到配对覆盖统计与
paired-consistency loss 路径。该改动只涉及部署脚本，不重训、不换数据。

### 44.9 mean-of-K 真实闭环否证与 tip-delta 契约错误（2026-09-07）

在 `deploy_dp_inverse.py` 增加 `--dp-samples K`，对完全相同的 v3 B2 checkpoint、
ep30、`S2_ORACLE`、`collection_matched_sensor`、live-q、chunk/replan=10 和 seed=42，
只改变每次 replan 的独立 diffusion 样本数，并在归一化 action 空间求均值后执行。
800 帧 CUDA 结果如下：

| K | 4 指接触 | 最长连续非 4 指 | q MAE / P95 | force max | 推理 mean / P95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 37.7% | 323 帧 | 0.444 / 0.532rad | 2700N | 491 / 580ms |
| 4 | 28.6% | 485 帧 | 0.327 / 0.419rad | 2962N | 474 / 495ms |
| 16 | 23.9% | 538 帧 | 0.394 / 0.551rad | 2220N | 505 / 540ms |

K=4/16 没有改善闭环；第一次 replan 的 joint target error 仍为
`0.0566–0.0608rad`（K=1 为 `0.0670rad`），并未随 `sqrt(K)` 消失。因此 §44.8
的 mean-of-K G 下降只证明采样方差可被平均，不能证明条件均值是可执行、可稳定的
teacher action。继续做更多轨迹的 K 对照或共享噪声 G，只能补充诊断，不能修复当前
闭环。

随后直接用 ep30 的**真实 ground-truth `tip_delta_tangent_palm` label**跳过 DP，送入
当前 `tip_delta_to_absolute_q()` 解码器。即使 action 完全无预测误差，frame
75/100/200/500 的首 waypoint q MAE 仍为 `0.054/0.072/0.075/0.061rad`，完整 H=16
q MAE 为 `0.082/0.079/0.086/0.082rad`；而相同窗口内 teacher q 的实际平均运动仅
约 `0.007–0.017rad`。解码后 fingertip 与 teacher future actual tip 的平均误差仍有
`7.7–11.8mm`，大于这些窗口中 teacher fingertip 的实际平均运动
`0.9–4.2mm`。

根因是 action 契约不一致：训练 label 在每个未来时刻 `j` 定义为

```text
tip_delta[j] = tangent_project(tip_x_des[j] - tip_actual[j])
```

它是该未来时刻 teacher 低层控制的瞬时 target/tracking residual。部署却解释为

```text
tip_target[j] = FK(q_live[t]) + predicted_tip_delta[j]
```

即“相对当前时刻 t 的未来 fingertip 位移”。这两个量不相同；每次 replan 还会把
同类 residual 再次加到新的 live tip 上，天然形成累积漂移。`tip-delta` validation
loss 低，只说明小的瞬时 residual 容易拟合，并不说明它能重建 teacher task motion。
这也解释了为什么 B2 离线 loss/G 看似改善，而真实闭环仍在第一次 replan 后迅速偏离。

下一步优先级据此修改为：先修 action label/decoder 契约，再讨论更多 recovery 数据、
paired loss 或采样噪声。DP 仍只负责名义任务运动，MCC 负责法向力；建议将 label 改为
相对同一规划起点的 future task-tip motion（或 absolute future task-tip target），
并用带参考手型正则的逐指 IK 解码。新 label 必须先通过“ground-truth action round-trip”
门控：不经过网络时应能重建 teacher fingertip path，误差显著小于实际窗口运动；门控
通过后才训练和跑 S1/S2。

### 44.10 千牛峰值与 teacher-forcing 结果澄清（2026-09-07）

§44.9 三次 S2 的千牛峰值不能简单归因于 diffusion 随机性，也尚未被证明为纯粹的
collision 穿模。峰值逐帧审计显示：K=1 的 ring 在 frame234 达 2700N，此时
`max|q_cmd-q_live|=0.698rad`、该指 tip target error=33.9mm；K=4 的 middle 在
frame779 达 2962N，分别为 0.778rad/53.7mm；K=16 的 middle 在 frame307 达
2220N，分别为 1.219rad/52.5mm。也就是位置执行器正在持续追踪一个被接触约束挡住的
不可行深层目标。

同时确认一个独立的安全门控缺陷：`FullHandMCCFingerController.update()` 先用
pad-validity 将 `found = raw_found & pad_contact_valid`，随后 hard-overforce 也使用这个
过滤后的 `found`。上述 1--3kN 峰值帧中 `mcc_overforce_active` 仍全部为 0，说明碰撞
落在被判为非有效指腹的区域后，MCC 正确地不把它当成任务接触，却也错误地不把它当成
必须退让的危险碰撞。任务接触 gate 与全手碰撞安全 gate 必须拆开：前者只接受有效 pad，
后者应对任何 raw finger-object 高力立即撤回/冻结 nominal action。

当前 CSV 的 ContactSensor `dist` 在有效碰撞帧均报告 0，无法据此量化 source mesh
penetration，所以“是否穿模”尚未被排除。后续安全审计需额外记录 raw contact geom、
contact 在 fingertip local frame 的位置以及 source-mesh signed distance；在此之前只能
确定千牛峰值来自“不可行位置目标 + 接触约束 + 安全 gate 漏检”，不能断言全部来自
几何穿透。

另需区分不同 checkpoint。§37 的 `teacher_dp + MCC = 99.8%` 使用的是此前
`kinematic_residual_q`/future-absolute-q 接口，不是当前 v3 B2 tip-delta checkpoint；
当前 tip-delta 路径此前的 Variant-B teacher 阶梯是 §43.3 的 S1=95.2%、
S1_ORACLE=93.4%。v3 B2 在 §44.9 前没有执行过 S1/S2。因此旧 99.8% 不能反证当前
tip-delta label/decoder 契约问题。一般而言 teacher forcing 还会每次用 recorded history
重新锚定输入，误差不会回写下一次 condition；即使 teacher-conditioned 接触率高，也不
等于该 action 能作为 free-running nominal state 稳定递推。
