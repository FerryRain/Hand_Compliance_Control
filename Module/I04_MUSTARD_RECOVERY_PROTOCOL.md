# 新 I04 v2：mustard 接触恢复（MCC-only）

日期：2026-10-13。用户要求从 mustard 开始，允许短暂失联，重点研究丢失后的恢复。
本轮仍仅运行 A-MCC/B-MCC，DPRef 延后。旧 Bunny v1 原始记录与源码 commit 保留。

## 对象与初始化

使用现有 YCB 006 mustard bottle derivative：原始数据已 center/scale 2.8，runtime scale=1。
物理为既有 256-part V-HACD，GT/覆盖为完整 source visual mesh；不对 mustard 使用 SDF physics。
包含全部三个 source mesh components，25 mm geodesic coverage，完整 required set 不删除。
FR3 7 + LEAP 16 DoF，固定物体、gravity-off、dt=2 ms，logging=10 ms。
初始 5 s 共用 acquisition；物体高度按四个实际 pad footprints 校准，初始不穿透。
缓慢 approach 后以相同 MCC/force relief 建立接触；危险初始化不启动 treatment。

## 接触、恢复与安全分别评价

短暂单指或全指失联不终止 episode、不赋策略 FAIL。记录全部 gap、首次重新接触、稳定恢复时间、
恢复后接触数、未恢复/终点截尾，以及恢复后运动和新增覆盖。至少一指正常接触连续 150 ms 才算稳定恢复。

两组共享显式的 measured-history recovery 层，方法名称解释为 MCC + Shared Recovery：

- 全指失联持续 100 ms 后，暂缓当前 nominal trajectory，以真实记录的安全接触 joint state 为恢复参考。
- 返回动作限速，MCC 仅以真实 force/normal 作法向调整；恢复层不读 Oracle、GT tip targets 或未来几何。
- 恢复参考保存实际产生安全接触力的 actuator command，保留弹性 preload；接近历史姿态后可沿
  最近实测 normal 作最多 3 mm 的有界局部 approach，不使用未来几何或显式 contact-mode 搜索。
- 回退参考从上一实际 actuator command 连续推进，避免每个 2 ms 从滞后的测量值重新开始而停滞。
- 稳定恢复后从 fresh state 重新规划并继续探索；8 s 未恢复记录 timeout，继续 episode 而非判整段失败。
- ≥8 N 时先局部向外 force relief，降到 <4 N 后继续原恢复；危险持续 >12 N/20 ms、
  真实非指尖碰撞、自碰撞、关节及数值风险独立停止。关闭恢复消融仍保留相同 force relief。
- 报告暂停时间、实际 wrist 路径、覆盖/时间，不能以长时间 hold 冒充恢复后的任务推进。

A 仍只暴露 wrist SE(3) target；B 额外暴露四个 geometric tip positions。两组共用恢复参数、
初态和运行预算，恢复增强不混写成旧 Pure Passive MCC 的结果。

## 验证流程与物理失联挑战

Smoke 首先验证自然运动。可另启用 `--withdraw-at`：在冻结时间沿当前/历史真实 contact normal
向外移动 wrist 25 mm，真实物理脱离后只解除扰动，不自动把 wrist 送回。恢复动作由 controller 自己产生。
挑战与自然失联结果分别记录；没有产生实际失联的挑战不记作恢复成功。

Formal traversal 使用 seeds 7/11/19、每组总 simulation budget 60 s（含 5 s acquisition），
分别运行自然运动和固定 treatment t=8 s 注入一次的挑战。不把自然失联和撤离失联混成挑战成功。

另做 `--task hold` 接触恢复隔离实验：A-MCC、seeds 7/11/19、25 s 总预算，t=4 s 撤离。
不查询 Oracle，不继续探索；撤离后 hold 撤离位置，不自动返回。分别开启/关闭恢复层，其余条件一致。
这一组用于检查恢复的因果作用，不作为 A/B 遍历比较，也不以 hold 覆盖率评价完整任务。
恢复层使用的仅是因果传感历史，不读取挑战的注入时间、结束信号或目标位置。
初始碰撞、力、source/parts hashes、输入 snapshot、core source hashes、controller/recovery 事件均保存。

本实验是已知 GT 的离线 native MuJoCo physics 评测；策略只报告指标，不用单次 contact loss 或瞬时力峰作策略判决。
