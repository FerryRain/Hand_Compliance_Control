# simulation_Etac

这个目录提供 LEAP Hand 的 MuJoCo 掌侧触觉读取器、两处 ETac 源码修改和示例。当前实现对方块计算密集形变。固定 300、500、1000 点和腕部驱动的手指策略属于下文的开发规格。依赖使用 [Crxus/ETac 的固定版本](https://github.com/Crxus/ETac/tree/6dc00c5d3dc49a6915076a2e1c63b9cff02179b0)。

## 已有代码

| 文件 | 用途 |
| --- | --- |
| [mujoco_leap.py](mujoco_leap.py) | 在掌心和 16 个活动链节的握持侧采样，查询方块表面并在同一链节内传播形变。 |
| [etac.py](etac.py) | 使用 ETac 的法向和切向接触计算，处理零切向速度。 |
| [object_model.py](object_model.py) | 通过 `initialize_mesh()` 接收物体局部坐标系中的 Trimesh，查询有符号距离和最近表面点。 |
| [check_leap.py](check_leap.py)、[view_leap.py](view_leap.py) | 使用本仓库的 LEAP XML 检查各部位或显示参考节点与受压节点。 |
| [check_windows.py](check_windows.py) | 用上游发布的 ShadowHand 与方块检查 Windows 依赖和传播权重。 |

从仓库根目录运行示例。环境安装与数据位置见 [Windows 运行说明](WINDOWS.md)。

```powershell
python -m simulation_Etac.view_leap
```

在现有 MuJoCo 循环中，对同一个 `MjModel` 创建一次读取器，在每次更新 `MjData` 后读取。`surface_nodes()` 返回未形变节点的当前世界位置及法向。`deformation()` 返回直接受压掩码和传播后的世界坐标位移，单位为米。`read()` 则返回链节局部触觉响应，包含接触力模型的贡献。

```python
from simulation_Etac import blind_grasping_assets
from simulation_Etac.mujoco_leap import MujocoLeapETac

reader = MujocoLeapETac(model, blind_grasping_assets() / 'propagation-ckpt')
reference_world, normal_world = reader.surface_nodes(data)
indented, displacement_world = reader.deformation(data)
deformed_world = reference_world + displacement_world.cpu().numpy()
```

## 掌侧固定点表规格（待实现）

**表面范围。** 以张开的参考手型标注掌心这一侧全部外露的握持表面，包括掌面、朝向握持空间的掌缘圆弧，以及四指从指根到指尖各节的掌侧外表面。拇指沿与其余三指对握的一面标注。标注固定在各自链节上，屈指或转腕后仍使用同一批材料点。指背、掌背和关节缝内壁在标注范围外。

**连接件排除。** 独立连接件整件排除，包括朝向掌心的部分。连接件和指腹合并在同一网格时，按三角面标注，只在合格外表面取点。当前四个指尖已有 `pad` 和 `connector` 的拆分，可用来对齐标注。连接件保留物理碰撞，只接触连接件时，皮肤直接接触面积为零。形变传播的压入源也限于已标注的皮肤。其余掌部与指节需要相应的面标注。[已有指尖几何拆分](../mcc_finger_compliance_control/scripts/split_fingertip_collision.py)

**三个分辨率。** 300、500、1000 都是整只手的总点数。三档覆盖相同的合格表面。先保证掌区与各有效指节均有采样，再按有效表面积和空间分布细化。300 点全部属于 500 点，500 点全部属于 1000 点。共同节点的编号、位置和法向一致。每档分别计算每点代表的表面积，以便跨分辨率比较贴合面积。

**数据接口。** 点表保存稳定的 `node_id`、所属链节与区域、参考网格面及面内位置、链节局部位置与法向，以及该分辨率的面积权重。每帧从同一密集形变场取值，用所属链节当前旋转的转置把世界位移转为 `displacement_link_m[N, 3]`，其中 `N` 取 300、500 或 1000。分别保存直接受压掩码和观测有效性。结合节点编号，同一时刻三个版本的共同节点可逐点比对。

## 手指策略需要新做的四项工作

现有逐指 DP 的 33 维本指状态中，20 维是关节角、速度、到下限的余量、到上限的余量和手指身份，各占 4 维。另 13 维描述掌坐标系中的接触点、法向、接触有效性及点和法向的变化。改版保留前述 20 维和另行输入的 12 维掌运动与腕指令，用形变历史替换这 13 维接触观测。训练的动作标签仍来自教师控制下实测的关节运动。沿用逐指结构时，三根长指各有自己的历史和动作，共用一套网络权重并合批推理，拇指使用另一套权重。[现有观测字段](../finger_policy/scripts/single_finger_dp.py) · [逐指推理实现](../finger_policy/scripts/controlled_policy_ablation.py)

| 新开发项 | 输入与产出 | 可复用的基础 |
| --- | --- | --- |
| 多指节贴合教师 | 给定腕部轨迹，只调 16 个手指关节。用各可达指节的接触面积、压入和双向关节余量生成示范，包括指尖卸压、中节重新贴上。 | [I04-B 有界关节求解器](../Module/i04_geometric_traversal/whole_hand_step_solver.py) |
| 整手同步采集与导出 | 在同一试次、同一时刻记录全手形变、实测关节角和速度、双向余量及当时已发布的腕指令。用历史 16 帧作为条件，未来 16 步实测关节角均减去同一当前实测角作为动作标签。 | [现有切窗与同步采样](../finger_policy/scripts/expanded_policy_data.py) |
| 形变编码器和 DP 输入 | 先做分区接触面积、压入量基线，再用 PointNet++ 编码本指和整手形变特征。保留手指身份、关节状态与掌运动，接入长指共享、拇指独立的 DP。 | [现有 DP 条件与去噪损失](../finger_policy/scripts/controlled_policy_ablation.py) |
| 全手闭环部署、评价和纠偏 | 操作者给腕指令，策略滚动输出手指参考。按指节统计贴合时长、过压、失触恢复时间和关节余量。策略持续失触时，从当时状态交给教师恢复并导出接管片段。 | [动作调度器](../mcc_finger_compliance_control/scripts/dp_chunk_scheduler.py) · [恢复数据工具](../recovery_standalone/scripts/build_recovery_dataset.py) |

第一条目标示范是腕部沿方块平移并轻微倾转，在同一时间轴上读出指尖压入减轻、中节恢复接触，以及教师实际执行的关节运动。现有 `deformation()` 只把压入深度大于零且小于 2 mm 的节点标为直接受压。新采集器需要另行记录达到或超过这个深度的节点，才能区分过度压入和失触。扩展到芥末瓶等物体还需要为读取器接入与碰撞模型对齐的连续物体表面。
