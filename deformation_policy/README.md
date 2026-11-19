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

| 动作结构 | `deformation` 输入 | `force_normal` 输入 |
| --- | --- | --- |
| 逐指 | `single_common/deformation` | `single_common/force_normal` |
| 整手联合 | `flat_common/deformation` | `flat_common/force_normal` |

同一次教师执行同步记录两种触觉。选择对应的观测列即可生成四组训练输入，旧力策略的权重只作历史参考。形变覆盖掌侧与多指节，力与法向来自四个指腹，两者的覆盖范围随观测方案一起比较。

**旧采样器需要替换。** `train_expanded_policy_comparison.py` 会创建并读取 `ExpandedWindows`，其 `choose()` 分支使旧 `common` 组也可能抽到 Expanded 样本。在本目录实现只读 Mustard 清单的路径，让四组在仅有这份清单时均可训练和评价。记录实际物体、母轨迹和窗口数。`single_all` 留待以后有 Expanded 数据时启用。[旧训练入口](../finger_policy/scripts/train_expanded_policy_comparison.py) · [旧采样器](../finger_policy/scripts/expanded_policy_data.py)

I04-B 已提供 Mustard 的连续表面、碰撞模型、初态和规划种子。先用 Git 与 Git LFS 中的启动输入生成一条短腕轨迹。已有 B 历史片段时也可提取其腕参考与初态。原 B 教师瞄准四个指腹，在同一瓶体和腕轨迹下执行新的多指节教师，记录其实际动作和两套触觉。[I04-B 保留版](../Module/I04_AB_RETAINED.md) · [历史片段来源](../data/i04/README.md)

## 现有策略怎样采、怎样学、怎样执行

| 环节 | 代码现在做什么 | 新项目如何复用 |
| --- | --- | --- |
| 采集与审核 | 原 mocap 流程按协议施加运动，由 MCC 控制手指，写 `rollout.h5`。候选审核检查指腹接触和力、试次是否完整、关节余量和异常跳步。 | 沿用原始轨迹、审核、导出分层。把训练准入改成多指节的贴合、卸载和恢复质量，保留原始失败轨迹用于诊断。[原采集说明](../finger_policy/README.md) · [整手采集入口](../finger_policy/scripts/collect_contact_wm_planned.py) · [候选审核](../finger_policy/scripts/prepare_expanded_finger_candidates.py) |
| 观测与标签 | 原逐指导出器从 100 Hz 记录重建因果观测，每 5 帧取一次。每指得到 16 帧 × 45 维历史，用未来实测关节角生成 16 步动作。 | 复用时间基准、关节顺序和标签定义。新的导出器对同一物理试次和时刻绑定四指、两套触觉与一份手指动作。[逐指字段与投影](../finger_policy/scripts/single_finger_dp.py) · [原逐指导出器](../finger_policy/scripts/export_wm_finger_candidates.py) · [力与法向字段](../mcc_finger_compliance_control/scripts/dp_dataset.py) |
| 训练 | 默认 `single_all` 有长指、拇指两个 U-Net，三根长指合批。旧训练器从随机权重开始，按噪声预测误差训练，保存 `best.pt` 和归一化参数。 | 复用 U-Net 构造、噪声调度、损失和当前关节角解码。改输入维度、窗口加载器、归一化和 checkpoint 格式。[训练入口](../finger_policy/scripts/train_expanded_policy_comparison.py) · [模型与损失](../finger_policy/scripts/controlled_policy_ablation.py) · [U-Net 构造](../mcc_finger_compliance_control/scripts/train_dp.py) |
| 执行与修正 | DP 输出关节参考块，调度器对齐动作并插值，原执行链的 MCC 根据接触反馈修正逐帧命令。 | 记录调度前后的参考和执行器收到的命令。教师在离线采集阶段生成示范，部署时测策略自己的响应。DAgger 已废弃。[动作调度器](../mcc_finger_compliance_control/scripts/dp_chunk_scheduler.py) · [MCC 执行链](../mcc_finger_compliance_control/scripts/deploy_b_gpu_serial_parity.py) |

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

在 `collect.py` 为每条教师试次写一份同时含两种触觉的 HDF5，在 `dataset.py` 生成四组共用的窗口索引。每帧保存固定 1000 点形变，300 与 500 点按 ID 取同帧子集并使用各自面积权重。下表是新格式建议。`T` 为物理帧数，`N` 为整手点数。小试次和结果存入 `runs/<run_id>/`。

| 字段与形状 | 含义及使用边界 |
| --- | --- |
| `time_s[T]`、`frame_id[T]`、`episode_id`、`parent_trajectory_id`、`trajectory_family_id`、`object_id` | 记录物理时钟、母轨迹、轨迹族、初态、接触区域、速度、种子、物体/点表哈希和提交号，供重放及按工况分组。 |
| `q_rad[T,16]`、`dq_rad_s[T,16]`、`margin_lo_rad[T,16]`、`margin_hi_rad[T,16]` | 来自动力学实测状态。`margin_lo=q−q_min`、`margin_hi=q_max−q`，采用模型物理限位。两组有符号余量都送入模型。关节名和指身份表随数据保存。 |
| `wrist_pose_world[T,7]`、`palm_twist_palm[T,6]`、`published_palm_delta_palm[T,6]` | 说明位置单位、四元数顺序和各速度的坐标系。后两项构成共享 12 维条件。用当时已发布的腕指令，记录指令时间戳。 |
| `u_link_m[T,N,3]`、`obs_valid[T,N]` | 每点局部形变及传感有效性。能看到且未形变的点有效，丢帧或不可读另标。合成触觉的预处理要能在部署端执行。 |
| `fingertip_force_palm[T,4,3]`、`fingertip_contact_normal_palm[T,4,3]`、原始有效性 | 四个指腹各自的三维合力（N）及接触单位法向，均在掌坐标系。写明受力方向、法向朝向、接触缺失和读数缺失的编码，在采集和部署中复用。 |
| `direct_gt[T,N]`、`depth_gt_m[T,N]`、区域目标/可达性、过深与碰撞事件 | 仿真真值供教师、采集审核和独立评价。策略所需接触面积或掩码从当时可观测的触觉推得，几何真值单独存。 |
| `q_teacher_ref[T,16]`、`q_command[T,16]`、运行类型与终止原因 | 在离线示范中对齐教师参考、下发命令与实测关节角，复核动作标签。策略闭环单独记录预测值和执行值。 |

MuJoCo 端可复用 I04 的接触遍历、指腹/连接件映射和姿态。现有 `contact_measurement` 把 `wrench[0]` 累加为法向标量。另取指腹与瓶体接触的三个力分量，按接触坐标旋到世界系、统一受力侧、对各指腹求合力，再旋到掌系。I04 返回从物体指向指腹的外法向，旧 DP 的主法向从指腹指向物体。按此约定翻转并归一化，记录哪个指腹有真实接触，无接触时法向置零。供策略使用的法向只来自当前接触，网格查询值另存供教师和独立评价。单点轻压、切向滑动和无接触三种情况可核对符号、切向分量和缺测行为。[I04 接触提取](../Module/i04_geometric_traversal/physics.py) · [已有指腹力与法向记录](../mcc_finger_compliance_control/scripts/collect_contact_wm.py) · [旧 DP 法向方向及来源](../mcc_finger_compliance_control/scripts/invert_trajectories.py) · [掌系导出](../mcc_finger_compliance_control/scripts/export_palm_dp.py)

每个控制周期先在同一物理状态读两种触觉、关节和当时的腕指令，再生成参考并步进。沿用 100 Hz 原始记录、20 Hz 策略帧，每隔 5 个原始帧取一帧。保留传感器时间戳，输入只使用截止当前时刻已到达的数据。每个窗口应能从原始记录逐帧复算。[旧观测重建](../finger_policy/scripts/reconstruct_contact_wm_dp_observation.py) · [现有切窗代码](../finger_policy/scripts/expanded_policy_data.py)

```text
条件窗口    X(t) = [O(t−15), …, O(t)]                   16 个策略帧，每帧含同一只手
手指标签    Y_f(t,k) = q_f(t+k) − q_f(t), k = 1…16      每指 [16,4]，单位 rad
整手解码    Q_ref(t,k) = q_measured(t) + Y_pred(t,k)    整手 [16,16]
```

每个未来标签都减同一个当前实测角。四组共用一份索引，每个样本的四指、本指与整手触觉都来自同一试次和时刻。在切窗前给母轨迹及其所有裁剪、扰动、镜像和点数版本分配同一划分。固定 `train`、`val`、`test_id` 和 `test_generalization` 四份清单。归一化只在训练集拟合，选模型只用验证集。[旧采样与同帧整手分支](../finger_policy/scripts/expanded_policy_data.py)

**交付与验收。** 交付一份双模态试次、一份包含相对路径、分组与哈希的 manifest，以及一个可反查的 16 入/16 出样本。在两个观测分支检查到相同的四指时间戳、双向余量、腕指令和动作标签。切窗留在同一试次及复位区间，学生输入只含已发布腕指令、当前及过去可观测的数据。保留失触、卸载和恢复阶段。

## 四、训练两种动作结构和两种触觉输入的 DP

**四组共享的状态和动作。** 保留原每指 33 维中的 20 维，即关节角、速度、到下限余量、到上限余量和四维指身份，各 4 维。另保留共享的掌运动和已发布腕指令 12 维。原余下 13 维记录接触点、法向、有效性和变化量。新协议在该位置接入对应触觉历史。两种触觉都是观测，监督一律来自同一教师执行后的未来实测关节增量。逐指结构以本指状态和同模态的本指、全手触觉为条件，三根长指合批共享权重，拇指独立，每指输出 `[16,4]`。整手结构按固定顺序读取四指及一份共享条件，由一个网络联合输出 `[16,16]`。[原逐指输入](../finger_policy/scripts/single_finger_dp.py) · [逐指和整手分支](../finger_policy/scripts/controlled_policy_ablation.py)

**`force_normal` 的定义取自仓库。** 每个指腹提供掌系三维力和三维接触法向，保留切向力。新接口沿用这两个触觉字段，保留上面约定的关节状态、双向余量和腕条件。逐指模型读取本指六个值及四指同帧力和法向的摘要，整手模型读取有固定指序的全部四指信号。所有触觉特征只从本组可观测通道提取。记录两种模态的维度、编码、采样频率和延迟。[力与法向 schema](../mcc_finger_compliance_control/scripts/dp_dataset.py) · [向量坐标变换](../mcc_finger_compliance_control/scripts/export_palm_dp.py)

1. **实现可解释的形变编码。** 在 `encoders.py` 按掌区和各指节，从有效位移估计面积加权接触面积、压入均值/峰值及其历史变化。校准压入正向，记录阈值和有效面积。每指读取本指特征和同帧整手摘要。先用少量窗口调通归一化、掩码、去噪与动作解码。主对照建议先固定为 500 点和这套分区编码，在两种动作结构中一致使用。
2. **再接 PointNet++。** 每点持有参考位置、法向、`u_link_m`、区域身份和有效位。在本链节参考网格上分组和聚合，随后按指和掌区池化得到 `z_finger` 与 `z_hand`。若需要在整手做空间邻域，把各链节点通过当前 FK 变换到共同掌系。来自不同链节的局部坐标不能直接拼成一个欧氏点云。对每一历史帧编码，按原有时间顺序输入 DP，一次计算的全手特征供四指共用。训练端与部署端共用同一编码器。仓库的普通 `SurfacePointNet` 输入是物体 GP 表面点云，可参考张量和池化写法，新的形变 PointNet++ 需要实现。[已有 PointNet](../mcc_finger_compliance_control/scripts/train_surface_pointnet.py)
3. **训练并复现。** 在 `policy.py`、`train.py` 增加 `architecture` 和 `observation` 两个选择项，替换旧代码中固定 45 维的检查、采样和归一化。复用 U-Net、噪声调度和 epsilon 预测损失，可沿用 100 个训练噪声步与默认 10 个推理去噪步。四组从头训练，使用相同母窗口、更新预算、种子列表、推理步数和选模规则，记录实际数据曝光量、模型参数量和推理耗时。各组保存 DP 权重、编码器及其参数、归一化、动作尺度、指序、点表/字段版本、划分和种子。可训练编码器随 DP 一起优化。[现有训练循环](../finger_policy/scripts/train_expanded_policy_comparison.py) · [旧模型固定输入和损失](../finger_policy/scripts/controlled_policy_ablation.py)

**交付与验收。** 为四种组合分别保存配置、checkpoint、数据清单和训练曲线。检查它们的母试次与窗口 ID 完全一致、物体均为 Mustard，任一组都可单独复载。按指报前 4 步及完整 16 步的关节 MAE，用于定位动作拟合误差。改变 300/500/1000 点数或分区编码与 PointNet++ 时单列附加实验，四组主对照保持相同的形变配置。

## 五、闭环性能和 Mustard 内泛化

在 `rollout.py` 接入与训练相同的输入、坐标、归一化和 16 帧历史。先积累有效历史或明确标记启动阶段的填充。用当前实测 `q` 将动作块恢复为绝对参考，再按 `DPChunkScheduler.observe / install / next_command` 的顺序接到执行器。默认调度器使用 0.05 s 路点、0.01 s 控制周期，配置要与实际时钟一致。分别记录调度前、调度后、实际下发和实测关节。操作者的腕指令在整个回路持续作为外部输入。[调度器与配置](../mcc_finger_compliance_control/scripts/dp_chunk_scheduler.py)

| 对照或指标 | 怎样判断是否完成用户要的功能 |
| --- | --- |
| 四组配对回放 | 让四个策略分别从完全相同的 Mustard 初态出发，跟踪相同腕指令。固定物理参数、时钟、执行器、动作调度、限位保护和评估种子，各策略闭环读取自己产生的传感反馈。主对照关闭可选 MCC 接触修正，另测修正版时对四组采用同一设置并记录修正量。固定手型与离线教师可作参照线。 |
| 每个可达区域的持续贴合 | 用采样点代表面积统计接触面积与目标面积比、连续贴合时长及达到预设贴合条件的试次成功率，同时给出各指近/中/远节、指尖和掌区的明细。用几何真值或独立接触读数评价，保存同一时间轴上的区域曲线和回放。 |
| 卸载、恢复与运动质量 | 统计过深压入的深度和时长、失触到稳定恢复的时间及未恢复次数、到两端的最小关节余量、越界次数、腕跟踪误差和控制延迟。以深度描述仿真压入，报告压强或力需要相应测量/标定。 |
| 测试工况与阈值 | 先在开发用的教师示范上固定可达目标区、正常压入带、恢复判定时长和腕跟踪容差。同一套独立几何/物理读数对四组评分，完整保留失败及未恢复试次。 |

**把常规性能和泛化分开报告。** `test_id` 使用训练分布中出现过的工况类型，抽独立的母轨迹与初态实例。`test_generalization` 仍用同一 Mustard，预先完整留出一种或几种训练和调参均未用过的工况，例如整族腕轨迹、一个速度区间、一组初始手型，或一个瓶肩区域与倾转方向的组合。一次留出一个因素便于解释，也可单列组合留出。所有测试工况都要在四组中配对。首期只检验同一芥末瓶上的新工况。后续若做跨物体泛化，可冻结这四组 Mustard 权重和归一化，在新物体上只做闭环评测。实机迁移需单独评价。

**怎样回答比较问题。** 给出四组在两张测试表上的绝对值、母轨迹数、种子数和按母轨迹配对的不确定性。固定动作结构比较形变与力/法向，再固定触觉比较逐指与整手。为每组报告从常规到留出工况的退化量，覆盖率等越高越好的指标用 `ID − 留出`，过深时长等越低越好的指标用 `留出 − ID`。对比各组退化量，并检查形变收益在 `single` 和 `flat` 上是否一致。

## 第一个工作包与最终交付

**第一个工作包**交付 I04-B Mustard、可驱动的腕、三档带 ID 的点表和一条双模态教师试次。给出同一时间轴的腕倾转、指尖卸载、中节复触及两种触觉读数，并导出一份可以手算核对的 16 入/16 出样本。随后按冻结的工况清单扩充 Mustard 数据和训练四组。

**最终交付**使另一台机器能从头安装、生成点表、采集同一批带两种触觉的 Mustard 示范、训练或加载四组模型，并只给腕指令运行闭环。交付可复制命令、提交号、清单及划分、四组配置和 checkpoint、常规与泛化结果表、配对差值和代表性失败回放。把实际跑通的入口写回本目录。现有环境命令见[启动说明](SETUP.md)。

迁移到真手时，还需要能观测相应掌区和指节形变的传感器，或经过标定的可观测映射。多点仿真输出本身不提供实机测量。记录实机可用部位、坐标、时间延迟、幅值与噪声，再决定哪些通道可以用于部署。仿真闭环与实机验证分别记录。
