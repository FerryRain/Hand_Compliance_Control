# _legacy — 退役实验/测试脚本(2026-09-07 归档)

本目录存放已被当前 mustard v4 主线(planner_inverse 批量双轨采集 → v3 导出链 →
tip_target DP → 闭环部署/DAgger)取代或淘汰的实验、测试脚本。**普通 mv 归档,
未删除**:git 历史与文件均可恢复;确认不再需要后可整目录删除。

| 脚本 | 退役原因 |
|---|---|
| `test_palm_free_orbit.py` | 早期 palm-free orbit 独立测试;采集主线早已改 planner_inverse 全手 |
| `run_planned_palm_contact.py` | §22/23 正向路径执行诊断实验(与反演路线互斥);主线用反演 |
| `batch_collect.py` | 跨物体批量采集 v1,被 `batch_collect_and_filter.py`(现役)取代 |
| `analyze_quality.py` | 早期多物体 raw 质量分析;接触率筛选已内嵌 `batch_collect_and_filter.py` |
| `reverse_collected_trajectory.py` | body↔cap 时间反转(239 cap 段时代),v4 随机化数据不再用反转增强 |
| `reverse_plan_time.py` | cap→body 恢复轨迹反转(cap 段时代),同上 |
| `collect_executed_q_live.py` | Step 2A executed_q_live 239 重采集(产物已弃用于双轨,§32 判定) |
| `export_executed_q_live.py` | 上者的导出端,同弃 |
| `export_clean_dual_track.py` | 旧 clean 单轨数据 lift 进双轨 schema(D1/clean 时代) |
| `export_dual_track.py` | v2 版导出(SPEED §L2208);label 用历史 q_ref 有语义缺陷(§L2590),被 `export_dual_track_v3.py` 取代 |

注意:`active_capsule_palm_planner.py`、`dp_chunk_scheduler.py`、
`fingertip_impedance.py`、`surface_manifold_gp.py`、`train_surface_pointnet.py`、
`palm_planner_features.py` 名称看似旧时代产物,但 `deploy_dp_inverse.py` 现役
代码仍直接 import,**不可移走**。
