# 当前 DP 工作流

当前保留范围：Mustard 222 数据、finger-history DP（finger19k）、无机械臂部署。

| 工作流 | 入口 |
|---|---|
| 参考准备 | `replan_healthy_reference.py`、`prepare_variable_collection_references.py` |
| 数据采集 | `run_stratified_variable_collection.py`、`collect_regional_healthy_reference.py` |
| 数据准备 | `prepare_stratified_dp.py`、`rebuild_commanded_palm_dp.py` |
| DP 训练 | `train_dp.py`，分指编码实现 `finger_history_dp.py` |
| 部署评估 | `evaluate_regional_dp.py`，执行链 `deploy_b_gpu_serial_parity.py` |
| 可视化 | `visualize_regional_deployment.py`、`plan_waypoint_viewer.py` |

其余文件作为上述入口的直接或间接依赖保留。部分通用入口仍静态引用旧 RL/DPPO 模块，因此不能按文件名删除；保留这些模块不表示它们是当前训练入口。

旧 DP/RL 实验、新曲面采集、机械臂部署等独立入口已归档至 `mcc_finger_compliance_control/backups/latest_dp_only_20261027_181917`，恢复方式及完整保留原因见该目录。数据、模型、Release 快照保持不变。
