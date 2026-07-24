# Full-hand MCC 开发进程

> 本文档是 `full_hand_mcc` 的可恢复实验日志。顶部要求不可删除。每完成一个
> 设计、实现或验证阶段都要及时更新并提交，确保任务中断、上下文压缩或更换
> session 后可以从“当前正在做”继续。每次上下文压缩/摘要恢复后必须重新读
> 根 `PROCESS.md`、本文档和 #7 最新评论，再执行任何代码或仿真命令。

## 固定验收要求

- 默认机器人必须是 Franka FR3（7 DoF）+ LEAP Hand（16 DoF），总计 23 DoF。
- 掌根采用机械臂负载残差反馈的 MCC；四指分别用对应 4 个电机的负载残差估计
  指尖力并反馈。
- 输入五点为掌根 + 四指尖；五点通过真实 MJCF/URDF 关节限制和同时 IK 可达性
  检查。掌根点是控制参考，不要求接触物体。
- 接触必须发生在四个物理指腹外侧。指甲/手背接触、自然闭合、手指离面、穿模
  均判失败。
- 指尖沿物体表面从下端缓慢、连续滑到上端帽面/顶部区域，并保持四指接触。
- 分阶段碰撞：先仅指尖与物体碰撞，再启用 FR3 + 掌部 + 非指尖链节全碰撞；
  最终结果中任何非指尖物体碰撞都判失败。
- 规划必须考虑关节限位、五点误差、指腹法向、相邻姿态平滑、机械臂/物体净距。
- 可继续探索 MPC、多阶段轨迹优化、刚性手掌跟随、晚段手指重构型等方案；所有
  候选必须通过数值和视频审核后才能称为成功。
- 最终要测试明显曲率变化和多个物体。泛化失败要写入
  [#4](https://github.com/FerryRain/Hand_Compliance_Control/issues/4)，并说明
  是否支持后续用 DP 学习指尖跟随。
- 所有正式实验在 Windows GPU MJLab 中运行；视频必须自行抽帧审核后交付。
- 代码、文档、合格视频直接提交 `main`，不创建新分支。

## 相关 issue

- [#7 FR3 迁移与滑到顶部](https://github.com/FerryRain/Hand_Compliance_Control/issues/7)
  —— 当前实现与验收。
- [#4 多物体和大曲率泛化](https://github.com/FerryRain/Hand_Compliance_Control/issues/4)
  —— FR3 主 demo 通过后继续。
- [#1 硬件力矩符号](https://github.com/FerryRain/Hand_Compliance_Control/issues/1)
  —— 仿真结果移植硬件前必须解决。
- [#6 xArm6 指腹/碰撞审核](https://github.com/FerryRain/Hand_Compliance_Control/issues/6)
  —— 可复用的验收方法与历史结果。

## 基线（已合并 main）

提交 `990c99a` 是 xArm6 + LEAP 的已审核基线：

- 胶囊半径 100 mm，表面规划/执行 280 mm。
- 四指接触率约 99.88%–100%。
- 指尖表面位移约 280.5–281.6 mm。
- 机械臂、掌部、非指尖链节对物体碰撞帧为 0。
- 物体穿透为 0，指腹角最大约 47.99°（阈值 50°）。
- 该路线只接近上端帽面边界，未充分进入顶部，因此不能满足当前 #7。

基线视频：
`full_hand_mcc/outputs/capsule_100x170_bottom_to_top_280mm.mp4`

## 当前正在做（2026-07-25）

### 1. FR3 资产与装配

状态：模型装配与编译已通过，待环境/GPU 验证。

- 上游：
  `google-deepmind/mujoco_menagerie/franka_fr3_v2`（Apache-2.0）。
- 已复制 8 个官方碰撞 STL、`LICENSE` 和上游 README 到
  `src/mjlab/asset_zoo/robots/fr3_leap_hand/`。
- 新增 `fr3v2_collision.xml`：保留官方 7 关节运动学、惯量、力矩/角度限制；
  复用低多边形碰撞 STL 作显示和碰撞，避免把大体积 OBJ 全部放入仓库。
- 计划在 `fr3v2_link8` 法兰创建 35 mm 刚性适配器，并通过 `MjSpec.attach`
  装配固定基座版 LEAP Hand。

已遇到并处理：

1. `MjSpec.attach` 默认前缀导致 `fingertip` 等名称找不到：
   改为 `prefix=""`, `suffix=""`。
2. attach 后两个源 XML 的相对 mesh 路径失去各自基准目录：
   已在合并前把所有 mesh 路径解析为绝对路径。
3. 含中文的 `D:\文档` 路径被 MuJoCo C XML 解析器显示为乱码：
   所有编译/GPU 命令必须在 `D:\Code\...\mjlab_full_hand_mcc` 执行。
4. 初次同步时 `Copy-Item -LiteralPath "...\*"` 不展开通配符，导致 8 个 STL
   漏拷贝；改为逐文件 `-LiteralPath` 同步。后续不要用该写法复制二进制资产。

编译检查（2026-07-25）：

```text
nq=23, nv=23, bodies=28, geoms=25
joint order:
fr3v2_joint1..7,
1,0,2,3, 5,4,6,7, 9,8,10,11, 12,13,14,15
FivePointReachabilitySolver: lower.shape=(23,), points.shape=(5,3)
```

### 2. 23 DoF 环境/控制器

状态：核心模型/求解器编译通过；环境运行与三个脚本尚未验收。

已改：

- `ARM_DOF=7`、`HAND_DOF=16`、`TOTAL_DOF=23`。
- FR3 关节名 `fr3v2_joint1..7` 与官方 home pose。
- 机械臂动作、初始化、关节速度、执行器力矩和 bias 力矩切片。
- 手指观测中的 q/tau/bias 偏移。
- 五点 IK 的关节名、维度、7 DoF 掌部 Jacobian。
- 全机碰撞传感器正则兼容 `fr3v2_link0..7_collision`。
- 移除只支持 xArm6 的旧掌部控制器实例；保留更直接的 FR3
  “执行器力矩 - bias 力矩”校准残差 MCC。

尚未改完：

- `demo_surface_slide.py` 中 22、6、`joint6` 周期处理等硬编码。
- `optimize_full_robot_grasp.py` 的 22+3 优化变量切片。
- `search_long_route_arm_branch.py` 的 6 DoF seed/分支逻辑。
- 单元测试的预期维度与 FR3 资产测试。

### 3. 到顶部路线

状态：未开始运行 FR3 规划。

计划：

1. 重新优化 FR3 初始抓取和物体位姿，确保指腹朝向为“接近物体局部内法向”，
   而不是机械地要求完全指向物体中心。
2. 以胶囊表面弧长而非世界直线插值生成多关键帧；每帧向表面内压入小预载深度。
3. 初步把表面行程从 0.28 m 提到约 0.38–0.45 m，并以“所有四指进入上端帽面”
   作为数值终止条件，而不是固定距离即成功。
4. 同时优化 FR3 冗余关节和平滑度，约束机械臂/物体净距。
5. GPU 执行速度降低，镜头必须能清楚看见拇指和顶部接触。

## 验收记录（待填）

以下字段在合格前不能写“通过”：

- 编译与模型关节数：
- 单元测试：
- 规划关键帧/表面行程：
- 四指接触率：
- 四指指腹角：
- 指尖表面实际位移：
- 最终上端帽面/顶部坐标：
- FR3/掌部/非指尖物体碰撞帧：
- 最大穿透：
- GPU 型号与运行命令：
- 视频：
- 抽帧图：
- Codex 视觉审核结论：

## 下一步恢复指令

新 session 从这里继续：

1. 读根 `PROCESS.md`、本文档和 issue #7 最新评论。
2. `git status --short --branch`，确认在 `main`，不要创建分支。
3. 模型编译已经通过；按“尚未改完”列表完成三个脚本的 23/7 参数化。
4. 每完成一项立即更新本文档并提交一个 process 检查点。
