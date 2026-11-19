# LEAP Hand 全手贴合策略开发交接

**首期只用 I04-B 的 Mustard（芥末瓶），比较逐指和整手策略在两种触觉输入下的闭环性能和泛化。** 操作者只给腕部运动指令，策略驱动 LEAP 的 16 个手指关节。用同一批教师示范训练四组模型，检验腕沿瓶身移动并倾转时，各指节能贴多久、过深时能否卸载、局部失触后能否重新贴合，以及手指还剩多少关节余量。

本项目的工作目录是主库根部的 `deformation_policy/`，与 `simulation_Etac/`、`finger_policy/` 和 `Module/` 同级。**所有新增代码、配置、操作说明、数据生成与训练流程都放在这里。** 新实验的原始数据、清单、权重、评估和视频统一写入 `deformation_policy/runs/<run_id>/`。现有目录按下文给出的接口读取。本文交付的是开发规格与接手步骤，五项开发任务见后文。

## 从空工作目录开始

在空工作目录执行以下命令。最后一条应显示原指尖策略的命令行帮助，它只用来确认拿到了正确的仓库。

```sh
git clone --depth 1 --branch main https://github.com/FerryRain/Hand_Compliance_Control.git .
git rev-parse --show-toplevel
git rev-parse HEAD
python -B finger_policy/pipeline.py --help
```

本说明核对的已有代码基线是 `76af384ef32bab28dd7e5e0b23807cf32dc46175`，其中已经包含 `simulation_Etac/`。开发时使用刚克隆到的提交，并把实际 SHA 记入新实验。首次接手先读主库约定和主计划，再按启动说明取得上游 ETac、加载 LEAP 模型。启动说明也列出原 DP 哪些输入需单独获取。[仓库约定](../CLAUDE.md) · [主计划](../Module/MASTER_PLAN.md) · [环境、资源与检查命令](SETUP.md)

下表给出新代码和产物的建议落点。现在这个目录包含交接文档和本地运行目录的 Git 忽略规则，表中的功能随开发逐步实现。

| 本目录内的位置 | 负责的内容 |
| --- | --- |
| `assets/`、`configs/`、`atlas.py`、`tactile.py` | 可版本控制的面掩码、固定点表、配置、生成与逐帧读取。 |
| `scene.py`、`wrist_trajectories.py`、`teacher.py` | 可动手腕、轨迹族、多指节教师和可重复的初始条件。 |
| `collect.py`、`dataset.py`、`encoders.py`、`policy.py`、`train.py` | 同帧采集两种触觉、切窗、编码、选择动作结构并训练四组 DP。 |
| `rollout.py`、`evaluate.py` | 按外部腕指令闭环运行，分别给出常规测试与 Mustard 留出工况的配对结果。 |
| `runs/<run_id>/` | 每次运行独立保存配置、原始试次、样本索引、权重和评估。清单中的路径以所属数据集或运行目录为基准。 |

## 三个策略版本

正式版本名定义动作结构和数据来源，`observation` 单独指定触觉输入。首期运行两个 `common` 版本，每个版本分别训练 `deformation` 和 `force_normal`。四组都从过去 16 帧预测未来 16 步关节增量，使用相同的 Mustard 母试次、窗口和实测动作标签。

| 配置名 | 正式训练数据 | 每一步怎样产生四指动作 | 本期安排 |
| --- | --- | --- | --- |
| `single_common` | 仅 Mustard | 每指输出自己的 4 个关节增量。食指、中指和无名指合批使用长指网络，拇指使用独立网络。 | 训练、闭环评价 |
| `flat_common` | 仅 Mustard | 一个全手策略同时读取四指状态和共享条件，联合输出 16 个关节增量。 | 用相同试次、窗口和预算训练并评价 |
| `single_all` | Mustard + Expanded。Expanded 指其他物体及特殊接触构型的数据。 | 与 `single_common` 使用相同的逐指结构。 | 仅保留定义，待以后开展 Expanded 阶段 |

### 首期四组训练

| 动作结构 | `deformation`（掌侧形变历史） | `force_normal`（原 `finger_policy` 指尖观测） |
| --- | --- | --- |
| 逐指 | `single_common/deformation` | `single_common/force_normal` |
| 整手联合 | `flat_common/deformation` | `flat_common/force_normal` |

`force_normal` 沿用实验名称，在配置里将其观测来源固定为 `finger_policy` 的 `project_history`。通用 `dp_dataset.py` 另有一个同名 schema，两者字段不同。逐指输入含指腹接触点相对手掌的位置、向内法向、接触有效位、接触点速度、法向变化率，以及原有的本体状态和共享腕指令。指腹实测力另供低层 MCC 和审核。`deformation` 用掌侧形变历史替换这 13 维指尖接触特征。同一教师试次同时记录两套观测所需的原始量，四组在新 Mustard 数据上配对重训。已发布的历史权重用于复现旧实验。[原逐指观测字段](../finger_policy/scripts/single_finger_dp.py) · [训练和推理如何组装条件](../finger_policy/scripts/controlled_policy_ablation.py)

**旧采样器需要替换。** `train_expanded_policy_comparison.py` 会创建并读取 `ExpandedWindows`，其 `choose()` 分支使旧 `common` 组也可能抽到 Expanded 样本。在本目录实现只读 Mustard 清单的路径，让四组在仅有这份清单时均可训练和评价。记录实际物体、母轨迹和窗口数。`single_all` 留待以后有 Expanded 数据时启用。[旧训练入口](../finger_policy/scripts/train_expanded_policy_comparison.py) · [旧采样器](../finger_policy/scripts/expanded_policy_data.py)

I04-B 已提供 Mustard 的连续表面、碰撞模型、初态和规划种子。先用 Git 与 Git LFS 中的启动输入生成一条短腕轨迹。已有 B 历史片段时也可提取其腕参考与初态。原 B 教师瞄准四个指腹，在同一瓶体和腕轨迹下执行新的多指节教师，记录其实际动作和两套触觉。[I04-B 保留版](../Module/I04_AB_RETAINED.md) · [历史片段来源](../data/i04/README.md)

## 现有策略怎样采、怎样学、怎样执行

| 环节 | 代码现在做什么 | 新项目如何复用 |
| --- | --- | --- |
| 采集与审核 | 原 mocap 流程按协议施加运动，由 MCC 控制手指，写 `rollout.h5`。候选审核检查指腹接触和力、试次是否完整、关节余量和异常跳步。 | 沿用原始轨迹、审核、导出分层。把训练准入改成多指节的贴合、卸载和恢复质量，保留原始失败轨迹用于诊断。[原采集说明](../finger_policy/README.md) · [整手采集入口](../finger_policy/scripts/collect_contact_wm_planned.py) · [候选审核](../finger_policy/scripts/prepare_expanded_finger_candidates.py) |
| 观测与标签 | 原导出器在 100 Hz 重建 96 维因果观测，每 5 帧取一次。`project_history` 投影出每指 16 帧 × 45 维的输入。标签来自之后实测的关节角。 | 原版组直接复用该重建、投影及分支组装。形变组共享本体、腕条件、时间戳和标签。[原始量与观测重建](../finger_policy/scripts/reconstruct_contact_wm_dp_observation.py) · [逐指投影](../finger_policy/scripts/single_finger_dp.py) · [逐指导出器](../finger_policy/scripts/export_wm_finger_candidates.py) |
| 训练 | 逐指使用长指、拇指两个 U-Net，整手使用一个 U-Net。训练从随机权重开始，以去噪误差优化。 | 原版组复用模型、输入布局、损失、归一化规则和动作解码。四组共享 Mustard 窗口和验证预算，形变组单独接触觉编码器。[训练循环](../finger_policy/scripts/train_expanded_policy_comparison.py) · [模型、条件与损失](../finger_policy/scripts/controlled_policy_ablation.py) · [U-Net 构造](../mcc_finger_compliance_control/scripts/train_dp.py) |
| 执行与修正 | 旧评估保留动作调度、MCC 力反馈和重抓握，关闭 DAgger 与 residual。 | 四组按同一设置评测，替换产生手指参考的 DP，分别记录调度前后的参考、MCC 修正和实测关节。多指节教师只用于生成离线示范。[原部署设置](../finger_policy/README.md#5-部署评估和原速回放) · [评估接线](../finger_policy/scripts/evaluate_controlled_policy_ablation.py) · [低层执行链](../mcc_finger_compliance_control/scripts/batched_b_control_chain.py) |

## 一、建立固定掌侧点表和逐帧形变读取器

**采哪些表面。** 在张开手的参考姿态标注掌面、朝握持空间的掌缘、四指各节从指根到指尖的掌侧外壳。拇指取朝向三根长指的对握面。掌背、指背、缝隙内壁、安装件和连接件均在采样范围外。暴露在掌侧的连接件也排除。独立零件按名称排除，皮肤和连接件共用网格时在原网格上标注三角面。指尖现有 `pad`、`connector` 凸件可用来定位分界，其余部位仍需要逐面标注。连接件继续参加物理碰撞，仅碰连接件时皮肤直接接触面积为零。[手模型 XML](../src/mjlab/asset_zoo/robots/leaphand_only.xml) · [已有指尖分件与哈希](../mcc_finger_compliance_control/assets/true_fingertip_contact_v1/asset_manifest.json) · [分件加载代码](../mcc_finger_compliance_control/scripts/split_fingertip_collision.py)

**怎样定点。** 300、500、1000 是整只手的总数。先给掌区与每个有合格表面的指节留点，再按有效表面积分配余量，在区域内均匀细化。生成一次确定的主点序列，满足 `P300 ⊂ P500 ⊂ P1000`。共同点的 ID、位置、法向永久一致。每一档独立计算代表面积，使该区域内各点权重之和等于区域的合格面积。点表保存 `node_id, region_id, link_id, face_id, barycentric, x_link_m, normal_link, area_m2`、几何版本和哈希。将小型点表、面掩码和生成配置放在 `deformation_policy/assets/`，整条轨迹共用同一版。

**怎样取形变。** 现有 `MujocoLeapETac.deformation(data)` 返回直接压入掩码和世界系位移，单位为米。它按固定方向向 17 个几何体投射射线，在同一几何体内传播方块接触的形变。`read(data)` 包含接触力模型的贡献。新读取器应在合格皮肤上选压入源和接收点，再一次计算同一时刻的场，供三档点表取样。必须能覆盖侧缘等在单一投影中被遮挡的合格面。每个点输出 `u_link = R_world_from_link.T @ u_world`。现有代码用 `geom_xmat`，若点表存的是 body 坐标，要显式处理 geom 到 body 的固定变换。[密集读取器](../simulation_Etac/mujoco_leap.py) · [探针与场景构造](../simulation_Etac/check_leap.py)

**直接接入 I04-B 的 Mustard。** 从 I04-B 场景取 `handles.fixture.mesh`，单独准备几何时可用 `load_fixture('mustard', output_dir)`，把 `output_dir` 设在本次 `runs/<run_id>/` 内。返回的连续表面和 256 个碰撞凸件已经用了同一套旋转、平移，2.8 倍缩放已烘入资产，运行时比例是 1。用 `ObjectModel.initialize_mesh(fixture.mesh)` 建立 ETac 距离查询，在 `fr3_e05_object` 严格对应的物体 body 坐标系中求距离，用 `object_geom_ids` 中的全部凸件维持物理碰撞。首个凸件有自己的 geom 原点，用它的位姿代替瓶体位姿会让触觉和碰撞错位。把适配代码放在本目录的 `tactile.py`。[I04-B 场景与网格变换](../Module/i04_geometric_traversal/fixtures.py) · [网格初始化与距离查询](../simulation_Etac/object_model.py) · [Mustard 资产来源](../assets_external/ycb/collision/006_mustard_bottle/vhacd_256_scaled2p8_objstage/README.md)

**怎样判定接触。** 在瓶身选取已知分离和轻压状态，核对米制距离、内外符号和最近点。单独记录直接压入、传播形变、观测有效性和过深压入。现有直接掩码只覆盖 `0 < d < 0.002 m`，其中物体内部为正。深度达到 2 mm 时该掩码会消失，要保留几何深度用于教师和审核，防止把过深压入当成失触。舒适压入带需另按材料和示范标定。

**交付与验收。** 推荐在本目录实现 `atlas.py`、`tactile.py`，交付三档点表、带区域颜色的正面及侧面图，以及掌侧与 Mustard 瓶身压入、释放的记录。核对点数、子集 ID、区域面积和连接件零采样。无接触时让手腕和关节运动，局部形变应接近零。对整体刚体转动后的同一接触构型，共同节点局部位移应一致。另用同一瓶体检查仅碰连接件和超过原读取带的压入，分别核对直接面积和过深标记。

## 二、给定腕轨迹，生成多指节贴合教师

在本目录的 `scene.py` 复用 I04-B 已有的固定 Mustard、FR3 腕部运动和 LEAP 初态。由外部给腕轨迹，经手指执行器运行真实物理步进，同时记录腕指令、实际腕姿和关节响应。将固定皮肤点映射到当前场景的 LEAP 链节，并按关节名映射 `qpos`、`qvel` 和执行器。旧策略四指关节顺序为 `(1,0,2,3) / (5,4,6,7) / (9,8,10,11) / (12,13,14,15)`。[保留版运行入口](../Module/i04_geometric_traversal/retained_ab.py) · [碰撞修复后的 B 场景](../Module/i04_geometric_traversal/palm_safe_mcc.py) · [关节名定义](../mcc_finger_compliance_control/scripts/surface_mcc_finger.py)

1. **在同一瓶上生成多样的腕轨迹。** 从 I04-B 初态接近瓶身并建立轻接触，先取一小段切向滑动和轻微倾转，再用 `find_patch`、`group_route` 改变瓶身正面、侧面、瓶肩等可达区域，以及触点、滑动方向、初始手型和腕速。安排局部卸载与复触，保留同一瓶体、同一套碰撞参数。B 的 25 mm 测地覆盖率度量物体表面曾被探访的范围，新示范按各手指区域实际贴合来审核。[规划器](../Module/i04_geometric_traversal/whole_hand_surface_planner.py) · [I04-B 保留版](../Module/I04_AB_RETAINED.md)
2. **构建教师目标。** 给每帧标出本腕姿下可达且期望贴合的区域。按区评估直接接触面积、压入带、局部脱离、过深和危险碰撞，加入自然手型和小步平滑项。指尖过深而中节脱离时，求能同时减轻指尖压入、让中节靠近的关节变化。掌区也测量和评估，但掌面位置受外部腕指令约束，记录手指无法弥补的部分。教师可读取已知物体表面、间隙和法向来生成示范。
3. **求解并实测。** 取多指节候选表面点到物体表面的距离或法向压入残差，用其关于 **16 个手指关节** 的雅可比组装 `J, error`。沿用 `WholeHandStepSolver.solve(J, error, prior, lower, upper, ...)` 的有界步进、不可行返回和单步限幅。腕部由外部轨迹驱动。对物理关节上下限分别保留余量，检查实际跟踪后的面积、深度和碰撞再接受该步。I04 的 `contact_measurement` 会把阈值以上的非指腹接触列入 `unsafe`。在新场景按面掩码允许目标皮肤接触，连接件和掌部硬件继续执行原有间隙与碰撞保护。持续不可达时记录区域、原因与实际状态。[有界求解器](../Module/i04_geometric_traversal/whole_hand_step_solver.py) · [指腹运动学示例](../Module/i04_geometric_traversal/whole_hand_kinematics.py) · [原掌部碰撞监测](../Module/i04_geometric_traversal/palm_collision.py)

**交付与验收。** `teacher.py` 和一条新生成的 Mustard 试次。在同一时间轴画出腕指令与实测腕姿、指尖压入、中节直接接触面积、关节实测角及双向余量。示范应出现腕继续沿给定轨迹移动、指尖卸载、中节重新贴上的过程。用实例解释无法同时贴合的区域，记录最小余量和过深时长。

## 三、同步采集全手，并导出训练窗口

在 `collect.py` 为每条教师试次写一份 HDF5，同时记录可重建原版 `finger_policy` 的指腹接触数据，以及固定 1000 点的掌侧形变。300 与 500 点从同帧按 ID 取子集并使用各自的面积权重。`dataset.py` 生成四组共用的窗口索引。下表中 `T` 表示原始控制帧数，`N` 表示整手点数。产物写入 `runs/<run_id>/`。

| 字段与形状 | 含义及使用边界 |
| --- | --- |
| `time_s[T]`、`frame_id[T]`、`episode_id`、`parent_trajectory_id`、`trajectory_family_id`、`object_id` | 记录物理时钟、母轨迹、轨迹族、初态、接触区域、速度、种子、物体/点表哈希和提交号，供重放及按工况分组。 |
| `q_rad[T,16]`、`dq_rad_s[T,16]`、`margin_lo_rad[T,16]`、`margin_hi_rad[T,16]` | 保存实测关节角、速度和物理限位。两端余量分别为 `q−q_min`、`q_max−q`。策略中的 `q_velocity` 沿用原观测器对实测角做的 5 帧后向差分，原始物理速度另存作核对。记录关节名、顺序和指身份。 |
| `palm_pose_world[T,7]`、`object_pose_world[T,7]`、腕参考与指令发布时间、`palm_twist_palm[T,6]`、`published_palm_delta_pose_palm[T,6]` | 以原策略的掌原点和坐标轴为准，给出腕到掌的固定变换及四元数顺序。后两项是共享的 12 维条件。旧采集用已发布的物体轨迹生成等效掌指令，本期固定瓶体，按同一前视时序把已发布的腕目标换为掌系增量。 |
| `u_link_m[T,N,3]`、`obs_valid[T,N]` | 每点局部形变及传感有效性。能看到且未形变的点有效，丢帧或不可读另标。合成触觉的预处理要能在部署端执行。 |
| `pad_point_world[T,4,3]`、`pad_normal_world[T,4,3]`、`pad_physical_found[T,4]` | 四个指腹与物体的实测接触点（m）、从指腹指向物体的向内法向和物理接触有效位。有效位直接取 `pad_physical_found`，指腹力阈值审核与连接件碰撞另行记录。与掌/物体位姿一起保存即可重建接触几何。记录使用的传感通道和缺帧标记。 |
| `contact_position_palm`、`contact_normal_palm`、`contact_valid`、`contact_point_velocity_palm`、`contact_normal_rate_palm` | 从原始序列按原观测器重建并可逐帧导出，每指分别为 3、3、1、3、3 维。`contact_position_palm` 是相对掌原点的位置。训练和在线推理使用相同的变换、接触记忆和差分。 |
| `pad_net_force_world[T,4,3]` | 保存指腹三维合力（N），记录受力侧、坐标和单位。原闭环的 MCC、接触审核和过力评价使用这路读数。原 DP 的输入以 `FIELDS` 定义的 45 维为准。 |
| `direct_gt[T,N]`、`depth_gt_m[T,N]`、区域目标/可达性、过深与碰撞事件 | 仿真真值供教师、采集审核和独立评价。策略所需接触面积或掩码从当时可观测的触觉推得，几何真值单独存。 |
| `q_teacher_ref[T,16]`、`q_command[T,16]`、运行类型与终止原因 | 在离线示范中对齐教师参考、下发命令与实测关节角，复核动作标签。策略闭环单独记录预测值和执行值。 |

**掌系位置、失触及变化率。** 原 `BatchedBObservation.push` 先在输入的固定参考系记住各指最近的有效接触点和法向，再用当前掌位姿变换。位置为 `p_palm = Rᵀ(p_ref − t_palm)`，法向只旋转。失触时 `contact_valid=0`，已建立的点和法向记忆继续投影到当前掌系。首次接触前和复位/预热沿用原状态机。接触点速度取 5 个 100 Hz 帧（0.05 s）的后向差分。法向变化率为前一采样法向与当前法向的叉积除以 0.05 s。两端均有接触时才计算这两项，其余输出零。[原观测器与接触记忆](../mcc_finger_compliance_control/scripts/batched_b_control_chain.py) · [原采集的坐标及指令转换](../mcc_finger_compliance_control/scripts/contact_wm_nominal.py)

在 MuJoCo 端按指腹与瓶体的碰撞对填入等效物理接触读数，将 I04 的物体外法向转为上述向内约定。原采集使用 `pad_point_world`、`pad_normal_world` 和 `pad_physical_found`。区分指腹与连接件接触，记录多接触归并方式。供 MCC 和过力评价使用的三个力分量按受力侧统一、变换到同一坐标系再求合力。记录原部署请求实际采用的法向来源。若该请求开启网格法向修正，保留原始法向、修正配置及修正后的值，离线和在线按同一约定重建。[I04 接触遍历](../Module/i04_geometric_traversal/physics.py) · [原指腹传感通道](../mcc_finger_compliance_control/scripts/collect_contact_wm.py) · [原导出读取的字段](../finger_policy/scripts/reconstruct_contact_wm_dp_observation.py)

每个控制周期在下发动作前读取触觉、关节和当时已发布的腕指令。沿用 100 Hz 原始记录、20 Hz 策略帧，每隔 5 帧取一个策略帧。原采集重建在第 t 帧引用已发布计划的第 t+19 帧目标（末尾截断），新场景应保留等效腕指令的前视时长、发布时间和坐标定义。保留传感器和指令时间戳，同一原始帧可复算原版与形变的两个输入。[旧观测重建](../finger_policy/scripts/reconstruct_contact_wm_dp_observation.py) · [现有切窗代码](../finger_policy/scripts/expanded_policy_data.py)

```text
条件窗口    X(t) = [O(t−15), …, O(t)]                   16 个策略帧，每帧含同一只手
手指标签    Y_f(t,k) = q_f(t+k) − q_f(t), k = 1…16      每指 [16,4]，单位 rad
整手解码    Q_ref(t,k) = q_measured(t) + Y_pred(t,k)    整手 [16,16]
```

每个未来标签都减同一个当前实测角。四组共用一份索引，每个样本的四指、本指与整手触觉都来自同一试次和时刻。在切窗前给母轨迹及其所有裁剪、扰动、镜像和点数版本分配同一划分。固定 `train`、`val`、`test_id` 和 `test_generalization` 四份清单。归一化只在训练集拟合，选模型只用验证集。[旧采样与同帧整手分支](../finger_policy/scripts/expanded_policy_data.py)

**交付与验收。** 交付一份包含接触、短时失触、转腕和重触的双模态试次、一份带分组与哈希的 manifest 和一个可回查的 16 入/16 出样本。用同一串原始读数分别走原 `BatchedBObservation.push → project_history → conditions` 和新采集/在线接口，逐字段对齐 96 维整手中间观测、每指 45 维投影及 flat 输入。报告有效位、失触记忆、两种变化量和双向余量的差异及最大数值误差。在两种观测下核对同一时间戳、腕指令与未来关节标签。切窗留在同一试次及复位区间。

## 四、训练两种动作结构和两种触觉输入的 DP

**原版 `force_normal` 组。** 按 `finger_policy/scripts/single_finger_dp.py::FIELDS` 保留每指的完整 45 维顺序。训练与推理均调用 `project_history`，由 `controlled_policy_ablation.conditions` 投入对应的 U-Net。原码的字段名、维度和几何意义如下。[字段顺序和投影](../finger_policy/scripts/single_finger_dp.py) · [逐指与整手输入组装](../finger_policy/scripts/controlled_policy_ablation.py)

| 每指原版输入 | 维度和语义 |
| --- | --- |
| 本体和身份 | `q` 4、`q_velocity` 4、`joint_lower_margin_rad` 4、`joint_upper_margin_rad` 4、`finger_identity` 4，共 20。 |
| 指尖接触 | `contact_position_palm` 3、`contact_normal_palm` 3、`contact_valid` 1、`contact_point_velocity_palm` 3、`contact_normal_rate_palm` 3，共 13。 |
| 共享腕条件 | `palm_twist_palm` 6、`published_palm_delta_pose_palm` 6，共 12。 |

`single_common/force_normal` 的每个分支只读本指 33 维和共享腕条件 12 维，每指输出 `[16,4]`。食指、中指、无名指合批共享权重，拇指独立。`flat_common/force_normal` 由原 `conditions(x, 'flat')` 的布局读取四指本地块和一份共享腕量，每帧 `4×33+12=144` 维，一个网络联合输出 `[16,16]`。

**形变组只替换指尖接触部分。** 复用上述 20 维本体和 12 维腕条件，以形变编码取代每指的 13 维接触历史。按全手触觉方案，逐指形变组读取本指形变和同帧全手形变摘要，整手形变组联合读取四指及掌侧特征。原逐指基线始终使用原本的本指观测范围。报告本指与跨指信息的可见范围，两列比较包含整手形变传感覆盖带来的差异。两组 DP 的监督都是同一批教师试次的未来实测关节增量。

1. **实现可解释的形变编码。** 在 `encoders.py` 按掌区和各指节，从有效位移估计面积加权接触面积、压入均值/峰值及其历史变化。校准压入正向，记录阈值和有效面积。每指读取本指特征和同帧整手摘要。先用少量窗口调通归一化、掩码、去噪与动作解码。主对照建议先固定为 500 点和这套分区编码，在两种动作结构中一致使用。
2. **再接 PointNet++。** 每点持有参考位置、法向、`u_link_m`、区域身份和有效位。在本链节参考网格上分组和聚合，随后按指和掌区池化得到 `z_finger` 与 `z_hand`。若需要在整手做空间邻域，把各链节点通过当前 FK 变换到共同掌系。来自不同链节的局部坐标不能直接拼成一个欧氏点云。对每一历史帧编码，按原有时间顺序输入 DP，一次计算的全手特征供四指共用。训练端与部署端共用同一编码器。仓库的普通 `SurfacePointNet` 输入是物体 GP 表面点云，可参考张量和池化写法，新的形变 PointNet++ 需要实现。[已有 PointNet](../mcc_finger_compliance_control/scripts/train_surface_pointnet.py)
3. **训练并复现。** 在 `policy.py`、`train.py` 中按动作结构及观测选择分支。原版组直接复用 `configs`、`conditions`、`shared_loss`、`sample`、原版 checkpoint 布局和 `ControlledRuntime` 的加回当前角解码。归一化按原 `CommonWindows` 在本期 Mustard 训练划分上拟合，包括长指共享统计、拇指单独统计、有效位和身份位的处理。保留原指尖短时丢接触增强，形变组按相同手指和时段处理其触觉有效性。形变组只扩展条件输入及其编码器。四组均从头训练，配对窗口、更新预算、种子、动作尺度、去噪步数和 Mustard 验证集选模规则。已有默认值为 100 个训练噪声步和 10 个推理去噪步。保存字段、参数、归一化、指序、点表、划分和实际训练来源。[原训练循环](../finger_policy/scripts/train_expanded_policy_comparison.py) · [模型与原版运行时](../finger_policy/scripts/controlled_policy_ablation.py) · [原接触增强](../finger_policy/scripts/expanded_policy_data.py)

**交付与验收。** 为四种组合分别保存配置、checkpoint、数据清单和训练曲线。检查它们的母试次与窗口 ID 完全一致、物体均为 Mustard，任一组都可单独复载。按指报前 4 步及完整 16 步的关节 MAE，用于定位动作拟合误差。改变 300/500/1000 点数或分区编码与 PointNet++ 时单列附加实验，四组主对照保持相同的形变配置。

## 五、闭环性能和 Mustard 内泛化

在 `rollout.py` 预热 16 帧历史，沿用原 `ControlledRuntime` 的关节增量解码和 `DPChunkScheduler.observe / install / next_command` 的调度顺序。调度器使用 0.05 s 路点、0.01 s 控制周期。操作者提供外部腕参考，两种观测方案的手指参考均经过原闭环的 MCC 力反馈和已配置的重抓握。记录名义手指参考、调度器输出、MCC 修正、实际指令和实测角，注明恢复状态和腕轨迹实际推进情况。[原部署与重抓握配置](../finger_policy/README.md#5-部署评估和原速回放) · [调度器](../mcc_finger_compliance_control/scripts/dp_chunk_scheduler.py) · [实际控制调用](../mcc_finger_compliance_control/scripts/batched_b_control_chain.py)

| 对照或指标 | 怎样判断是否完成用户要的功能 |
| --- | --- |
| 四组配对回放 | 四组从相同 Mustard 初态出发，使用相同外部腕指令、时限、执行器、动作调度、限位、MCC、重抓握配置和评估种子，各自读取闭环产生的触觉。DAgger 和 residual 保持原评估的关闭设置。主对照比较完整执行链。若另分析去掉 MCC 后策略本身的表现，四组用同一消融设置单列结果。 |
| 每个可达区域的持续贴合 | 用采样点代表面积统计接触面积与目标面积比、连续贴合时长及达到预设贴合条件的试次成功率，同时给出各指近/中/远节、指尖和掌区的明细。用几何真值或独立接触读数评价，保存同一时间轴上的区域曲线和回放。 |
| 卸载、恢复与运动质量 | 统计过深压入的深度和时长、失触到稳定恢复的时间及未恢复次数、到两端的最小关节余量、越界次数、腕跟踪误差和控制延迟。以深度描述仿真压入，报告压强或力需要相应测量/标定。 |
| 测试工况与阈值 | 先在开发用的教师示范上固定可达目标区、正常压入带、恢复判定时长和腕跟踪容差。同一套独立几何/物理读数对四组评分，完整保留失败及未恢复试次。 |

**把常规性能和泛化分开报告。** `test_id` 使用训练分布中出现过的工况类型，抽独立的母轨迹与初态实例。`test_generalization` 仍用同一 Mustard，预先完整留出一种或几种训练和调参均未用过的工况，例如整族腕轨迹、一个速度区间、一组初始手型，或一个瓶肩区域与倾转方向的组合。一次留出一个因素便于解释，也可单列组合留出。所有测试工况都要在四组中配对。首期只检验同一芥末瓶上的新工况。后续若做跨物体泛化，可冻结这四组 Mustard 权重和归一化，在新物体上只做闭环评测。实机迁移需单独评价。

**怎样回答比较问题。** 给出四组在两张测试表上的绝对值、母轨迹数、种子数和按母轨迹配对的不确定性。固定动作结构比较全手形变与原版指尖接触观测，再固定触觉比较逐指与整手。为每组报告从常规到留出工况的退化量，覆盖率等越高越好的指标用 `ID − 留出`，过深时长等越低越好的指标用 `留出 − ID`。对比各组退化量，并检查形变收益在 `single` 和 `flat` 上是否一致。

## 第一个工作包与最终交付

**第一个工作包**交付 I04-B Mustard、可驱动的腕、三档带 ID 的点表和一条双模态教师试次。给出同一时间轴的腕倾转、指尖卸载、中节复触、原指尖接触点/法向/有效位和全手形变，并导出一份可以手算核对的 16 入/16 出样本。先完成原版输入的逐字段一致性核对，再按冻结的工况清单扩充 Mustard 数据和训练四组。

**最终交付**使另一台机器能从头安装、生成点表、采集同一批带两种触觉的 Mustard 示范、训练或加载四组模型，并只给腕指令运行闭环。交付可复制命令、提交号、清单及划分、四组配置和 checkpoint、常规与泛化结果表、配对差值和代表性失败回放。把实际跑通的入口写回本目录。现有环境命令见[启动说明](SETUP.md)。

迁移到真手时，还需要能观测相应掌区和指节形变的传感器，或经过标定的可观测映射。多点仿真输出本身不提供实机测量。记录实机可用部位、坐标、时间延迟、幅值与噪声，再决定哪些通道可以用于部署。仿真闭环与实机验证分别记录。
