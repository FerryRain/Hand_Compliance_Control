# 新 I04 mustard 接触恢复记录

日期：2026-10-13。用户指定 MCC-only、允许短暂失联、从 mustard 开始。
状态：`EVALUATED / FULL_COVERAGE_INCOMPLETE`，DPRef：`DEFERRED_BY_USER`。

完成 18 次真实 native MuJoCo 运行，其中 18 次达到预算。隔离撤离实验：恢复层开启 3/3 次稳定恢复，关闭 0/3；开启时稳定恢复耗时 3.796–3.874 s。两侧初态及撤离前完整保存历史逐元素一致。

natural / A-MCC：平均接触率 99.57%，覆盖 5.062%，安全接触失联 9 次、稳定恢复 9 次，保护提前停止 0 次。
natural / B-MCC：平均接触率 97.45%，覆盖 3.631%，安全接触失联 47 次、稳定恢复 47 次，保护提前停止 0 次。
challenge / A-MCC：平均接触率 70.16%，覆盖 4.271%，安全接触失联 14 次、稳定恢复 14 次，保护提前停止 0 次。
challenge / B-MCC：平均接触率 82.55%，覆盖 3.917%，安全接触失联 50 次、稳定恢复 50 次，保护提前停止 0 次。
撤离后持续失联段 / A-MCC：稳定恢复 16.394–16.478 s，恢复后到终点新增覆盖 1.937–2.882 个百分点。
撤离后持续失联段 / B-MCC：稳定恢复 4.048–5.562 s，恢复后到终点新增覆盖 0.761–2.705 个百分点。
A 当前以最近两指稳定接触的历史作为恢复参考；后段仅一指接触时，该参考会较旧。恢复速度和更多手指的参与仍需改进，四指同时接触比例在本轮均为 0。

| 运行 | 组 | Seed | treatment s | 接触率 | 覆盖 | 稳定恢复/失联 | 保护停止或预算 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| natural | A-MCC | 7 | 55.00 | 99.54% | 4.635% | 5/5 | TIME_BUDGET_REACHED |
| natural | B-MCC | 7 | 55.00 | 96.17% | 3.668% | 22/22 | TIME_BUDGET_REACHED |
| natural | A-MCC | 11 | 55.00 | 99.60% | 4.713% | 2/2 | TIME_BUDGET_REACHED |
| natural | B-MCC | 11 | 55.00 | 98.15% | 3.621% | 12/12 | TIME_BUDGET_REACHED |
| natural | A-MCC | 19 | 55.00 | 99.56% | 5.839% | 2/2 | TIME_BUDGET_REACHED |
| natural | B-MCC | 19 | 55.00 | 98.01% | 3.603% | 13/13 | TIME_BUDGET_REACHED |
| challenge | A-MCC | 7 | 55.00 | 70.07% | 3.694% | 6/6 | TIME_BUDGET_REACHED |
| challenge | B-MCC | 7 | 55.00 | 69.28% | 2.732% | 33/33 | TIME_BUDGET_REACHED |
| challenge | A-MCC | 11 | 55.00 | 70.28% | 4.640% | 4/4 | TIME_BUDGET_REACHED |
| challenge | B-MCC | 11 | 55.00 | 88.21% | 4.088% | 11/11 | TIME_BUDGET_REACHED |
| challenge | A-MCC | 19 | 55.00 | 70.13% | 4.479% | 4/4 | TIME_BUDGET_REACHED |
| challenge | B-MCC | 19 | 55.00 | 90.15% | 4.930% | 6/6 | TIME_BUDGET_REACHED |
| hold_on | A-MCC | 7 | 20.00 | 81.45% | 1.274% | 1/1 | TIME_BUDGET_REACHED |
| hold_on | A-MCC | 11 | 20.00 | 81.82% | 1.274% | 1/1 | TIME_BUDGET_REACHED |
| hold_on | A-MCC | 19 | 20.00 | 81.42% | 1.274% | 1/1 | TIME_BUDGET_REACHED |
| hold_off | A-MCC | 7 | 20.00 | 28.96% | 1.274% | 0/1 | TIME_BUDGET_REACHED |
| hold_off | A-MCC | 11 | 20.00 | 28.90% | 1.274% | 0/1 | TIME_BUDGET_REACHED |
| hold_off | A-MCC | 19 | 20.00 | 28.93% | 1.274% | 0/1 | TIME_BUDGET_REACHED |


## 方法和局限

两组共享 Basic MCC + SharedContactRecovery。A 仅获取 wrist SE(3)，B 额外获取四指几何位置。
恢复层只读取实测位置、实际下发 command、接触和力的因果历史；不读取 Oracle 或撤离时刻。
每个 seed 共用 5 s acquisition 的 q/qvel/ctrl snapshot。mustard scale 2.8 已烘焙，runtime=1；
使用既有 256-part V-HACD，完整 source mesh 的三个连通分量、388 必访点全部保留。
总表面面积 0.356718412 m²；覆盖只来自真实 pad contacts 的 geodesic union。

全指安全接触缺失不终止运行。100 ms 去抖、150 ms 稳定确认、8 s 恢复 timeout 后继续。
≥8 N force relief 在消融两侧仍保留，持续 >12 N/20 ms 等独立保护不放宽。
traversal 的自然和 t=8 s 撤离各运行 60 s 总预算；hold 隔离实验 25 s、t=4 s 撤离，
撤离后保持撤离位置、无自动返回、无 Oracle、不作遍历比较。

失联次数包括短暂 gap；稳定恢复要求至少一指连续正常接触 150 ms。恢复后运动和新增覆盖在各次运行报告逐事件列出；其后缀可能重叠，不能相加。wrist 路程包含细微接触振荡，探索推进以新增实测覆盖为据。撤离挑战若开始时已无接触，不能算作撤离造成失联。未完成完整 mustard 覆盖；3 seeds、固定物体、gravity-off、GT 离线规划的结果不外推到实机。

运行前 28 项接口、恢复语义、指令推进、覆盖、FR3/MCC 回归检查通过。
各运行 source/manifest/model/256 parts/trace hashes、paired snapshots、因果恢复参考、稳定实测接触、
恢复事件记账与 A/B 信息边界审计均为 VERIFIED。hold 开/关撤离前完整保存历史逐元素相同。
MuJoCo native CPU physics；GPU OpenGL 仅用于记录回放；离线同步规划的 wall latency 单独保存。

冻结协议：[`../I04_MUSTARD_RECOVERY_PROTOCOL.md`](../I04_MUSTARD_RECOVERY_PROTOCOL.md)。
本机汇总报告：[`../generated/new_i04_mustard_recovery_review_20261013/index.html`](../generated/new_i04_mustard_recovery_review_20261013/index.html)。
运行 commit：`558a012d000e4a45702476d9843b34c238594519`；每次 manifest 保存配置、环境和源码 hashes。
旧 Bunny v1 证据保留在 `2026-10-13_NEW_I04_MCC.md`，本轮不覆盖历史记录。
