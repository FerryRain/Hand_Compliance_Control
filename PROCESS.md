# 主分支开发进程（持久检查点）

> 本文档是跨 Codex session 的主入口。任何 session 开始工作前，应先读本文档，
> 再读对应子项目的 `PROCESS.md` 和链接的 GitHub issues。每完成一个阶段都要更新并
> 提交本文档；不要只在任务结束时补写。任何上下文压缩/摘要恢复发生后，也必须
> 重新读取根与子项目两份 PROCESS，不能只依赖压缩摘要继续。

## 用户要求（必须保留在文档头部）

1. 以后只在 `main` 更新，不再切换或创建 Codex 功能分支。
2. 主 README 只提供项目目录索引、简要运行命令和代码结构；子目录 README 说明
   具体问题、实现和用法。
3. 仓库根目录和子项目目录都要有 process 文档，记录已完成、未完成、正在做、
   遇到的问题、失败方法、下一步和相关 issue，便于其他 session 快速续接。
4. `full_hand_mcc` 使用 Franka FR3 + LEAP Hand：机械臂关节负载作为掌根 MCC
   反馈，16 个手指电机负载作为四个指尖 MCC 反馈。
5. 测试输入为掌根与四个指尖共五点；所有点必须满足真实 URDF/MJCF 关节约束和
   同时可达性。掌根控制点允许不接触物体。
6. 四个物理指腹必须始终贴着物体表面连续、缓慢滑动；不能自然闭合，不能由
   指甲/手背接触，不能穿模。
7. 路径要从物体下端移动到上端并进入顶部曲面区域；不能只移动很短距离。
8. 先验证“仅四指尖可碰撞”，成功后启用全机器人碰撞；FR3、掌部和非指尖链节
   不能碰撞物体，规划优化中要显式加入机器人/物体位姿和净距约束。
9. 可探索多种 MCC、轨迹优化、MPC 等方法；每种结果都必须审核。继续研究改进
   方法，不局限于预先规定的数量。
10. 要在曲率变化明显和多个物体上验证泛化性；若泛化不足，明确记录，并作为
    后续使用 DP 学习指尖跟随的依据。
11. 允许反复运行 Windows GPU MJLab。最终视频必须由 Codex 自己逐帧/抽帧审核，
    审核不合格不得交付。
12. 新要求和问题要及时写入 GitHub issue；持续处理已有 issue。成功后直接提交
    并推送 `main`。

## 仓库状态

- 运行仓库：`D:\Code\Hand_Compliance_Control\mjlab_full_hand_mcc`
- 当前分支：`main`
- `codex/full-hand-mcc-end-to-end` 已于 2026-07-25 快进合并并推送到 `main`
  （合并后提交为 `990c99a`）。
- 后续不得为本项目创建新分支；允许在 `main` 上分阶段提交持久检查点。

## 子项目进度

| 子目录 | 作用 | 状态 | 详细进程 |
| --- | --- | --- | --- |
| `full_hand_mcc/` | FR3 + LEAP 全手 MCC、五点规划、碰撞审核、视频 | 迁移进行中 | [`full_hand_mcc/PROCESS.md`](full_hand_mcc/PROCESS.md) |
| `palm_compliance_control/` | 原始掌部/机械臂 MCC 参考 | 已有参考实现 | README/源码 |
| `mcc_finger_compliance_control/` | 手指 MCC 参考 | 已有参考实现 | README/源码 |
| `finger_compliance_control/` | 手指顺应控制参考 | 已有参考实现 | README/源码 |
| `minimalist_compliance_control/` | 上游/简化 MCC 方法 | 参考代码 | README/源码 |

## GitHub issue 索引

- [#7 FR3 full-hand MCC：迁移机械臂并滑到物体顶部](https://github.com/FerryRain/Hand_Compliance_Control/issues/7)
  —— 当前主任务。
- [#4 多物体/曲率泛化](https://github.com/FerryRain/Hand_Compliance_Control/issues/4)
  —— 未完成，FR3 单物体验收后继续。
- [#1 硬件力矩符号与五接触接口](https://github.com/FerryRain/Hand_Compliance_Control/issues/1)
  —— 仿真后续硬件问题，仍开放。
- [#3 长距离连续表面滑动](https://github.com/FerryRain/Hand_Compliance_Control/issues/3)
  —— xArm6 版本已关闭；FR3 的更强顶部要求由 #7 追踪。
- [#6 指腹方向、全机碰撞和长路线审核](https://github.com/FerryRain/Hand_Compliance_Control/issues/6)
  —— xArm6 验收记录，已关闭。

## 当前正在做

正在执行 #7：FR3 23 DoF 迁移与 GPU smoke 已完成，当前在修复 0.48 m
跨上端帽面路线的动态手指自碰撞，然后重新运行完整 GPU 仿真和视频审核。

已完成：

- 从 MuJoCo Menagerie 取得官方 FR3 v2 关节、惯量、关节限制和低多边形碰撞网格，
  保留 Apache-2.0 许可证与上游说明。
- 新增轻量 FR3 模型骨架，设计在 `fr3v2_link8` 法兰处刚性装配 LEAP Hand。
- 环境核心的 7/16/23 DoF 常量、FR3 关节命名、机械臂动作项、观测切片、掌根和
  手指负载反馈切片已完成改造；GPU 环境 reset 与单步 action 已通过。
- 掌根五点中的第 0 点改为 `palm_lower` 根部控制点，不要求物理接触。

当前阻塞/问题：

- MuJoCo 原生 XML 解析器不能可靠读取包含中文字符的模型路径。因此源码可在
  `D:\文档` 镜像中补丁化，但编译/GPU 运行必须同步到 ASCII 路径 `D:\Code`。
- 第一次动态装配测试发现 `MjSpec.attach` 默认会添加模型名前缀；已明确使用空
  前后缀保留既有指尖 body/site 名称。
- 第二次编译发现 attach 后相对网格路径丢失；正在把 FR3 与 LEAP 网格路径在
  合并前解析为绝对路径。该修复已完成。
- 2026-07-25 检查点：FR3+LEAP 合并模型已在 `D:\Code` 编译成功，
  `nq=23`、`nv=23`、28 bodies、25 geoms；23 个关节顺序和五点 site 均已确认，
  `FivePointReachabilitySolver` 输出 `(5, 3)`。
- FR3 法兰初始安装方向会让手指向后穿入 link6/link7，最深 49.27 mm；已在安装
  旋转后增加局部 X 轴 180° 翻转，home pose 的机械臂/手自碰撞降为 0。
- demo、抓取优化器、长路线分支搜索器和指腹映射诊断脚本已完成 23/7 参数化，
  10 个 MCC core unittest 全部通过。
- 已生成 `full_hand_mcc/assets/fr3_capsule_100x170_grasp_v1.npz`：
  指尖误差最大 0.043 mm，指腹角 13.65°–38.89°，非指尖最小净距 4.885 mm，
  自碰撞 0。
- 已生成高净空 v2 抓取与 0.48 m 路线。路线使四指全部进入上端帽面，静态路线
  机械臂最小净距 43.669 mm；240° 物体接近路径通过全机/物体净距审查。
- v2 GPU 接近阶段无非指尖/物体碰撞，但首个滑动关键帧发现
  `mcp_joint_geom`–`dip_geom` 动态自碰撞约 0.080 mm，超过 0.05 mm 验收阈值，
  已拒绝该视频。

下一步：

1. 重新优化带至少约 0.4–0.5 mm 静态受保护自碰撞余量的 v3 抓取；不放宽动态
   0.05 mm 阈值掩盖穿模。
2. 从 v3 重建并审核 0.48 m 跨上端帽面的长路线。
3. 运行 GPU MJLab 全程仿真，输出视频并抽帧审核指腹接触、拇指可见、顶部到达、
   全机无物体碰撞和无手指自碰撞；失败则继续改。
4. 补 FR3 回归测试、两级 README，执行多物体/大曲率泛化测试并更新 #7/#4。
5. 分阶段更新两级 PROCESS，提交并直接推送 `main`。
