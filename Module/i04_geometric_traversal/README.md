# 新 I04：几何轨迹＋被动 MCC

入口：`python -m Module.i04_geometric_traversal.benchmark --profile formal --output <fresh_directory>`。
冻结协议：[`../I04_GEOMETRIC_MCC_PROTOCOL.md`](../I04_GEOMETRIC_MCC_PROTOCOL.md)。
DPRef 按用户要求延后，本实现只运行 A-MCC/B-MCC。

功能层分别为：GT Coverage Oracle、Shared Geometric Trajectory Planner、Basic Passive MCC、
Hardware Safety/Trajectory Integrity、Generic Trajectory Executor、Measured Coverage。
旧 surface graph 仅作为纯几何工具复用，不运行 legacy contact-mode planner。

每个 seed 只 acquisition 一次，A/B 从完全相同的 q/qvel/ctrl snapshot 出发。
所有覆盖来自真实 physical pad contact；不把 target、预测或 IK witness 计入覆盖。
zero contact 继续运行；硬件危险停止独立记录，提前停止不记作全预算完成。

`review.py` 读取同一次 NPZ/JSON，生成 comparison PNG、HTML、aggregate。
`--video` 使用实际保存 q 的 FK 回放，不重算指标。

Windows 中文工作区通过 MuJoCo VFS 读取二进制 meshes；缺失依赖隔离在 `.runtime/`。
运行脚本为 `scripts/run_new_i04_mcc.ps1`。

本机 21 项接口/覆盖权限/force-sign/机器人与 MCC 回归检查通过。
正式实验结果与 source/config/environment hashes 由 generated 目录的 manifest 保存。
