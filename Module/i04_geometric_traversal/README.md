# 新 I04：mustard 几何轨迹＋MCC＋接触恢复

2026-10-19最新版本为四指协同整瓶遍历：8192/8192顶点、388/388目标覆盖，完整原生重放通过。
源码、报告、视频和完整轨迹一起发布；大文件使用Git LFS。

- [运行和下载](../I04_WHOLE_HAND_RUNBOOK.md)
- [更新日志](../CHANGELOG.md) / [结果与限制](../evidence/2026-10-19_I04_WHOLE_HAND_SURFACE.md)
- [当前协议](../I04_WHOLE_HAND_SURFACE_PROTOCOL.md)

可迁移入口：`python -m Module.i04_geometric_traversal.published_whole_hand --help`。
`verify`检查下载，`smoke`检查短段原生重放，`audit`重放整条轨迹，`run-fresh`启动零覆盖的新实验。
原始run10为开发接续结果；最长有效断触146.156秒仍需优化。以下为2026-10-13基线说明。

入口：`python -m Module.i04_geometric_traversal.benchmark --profile formal --output <fresh_directory>`。
当前冻结协议：[`../I04_MUSTARD_RECOVERY_PROTOCOL.md`](../I04_MUSTARD_RECOVERY_PROTOCOL.md)。
旧 Bunny v1 的协议和首轮证据保留，以旧 commit 复现。
DPRef 按用户要求延后，本实现只运行 A-MCC/B-MCC。

功能层分别为：GT Coverage Oracle、Shared Geometric Trajectory Planner、Basic Passive MCC、
Hardware Safety/Trajectory Integrity、Generic Trajectory Executor、Measured Coverage、SharedContactRecovery。
旧 surface graph 仅作为纯几何工具复用，不运行 legacy contact-mode planner。

每个 seed 只 acquisition 一次，A/B 从完全相同的 q/qvel/ctrl snapshot 出发。
所有覆盖来自真实 physical pad contact；不把 target、预测或 IK witness 计入覆盖。
zero contact 继续运行；硬件危险停止独立记录，提前停止不记作全预算完成。

默认物体 mustard，既有 256-part V-HACD，完整三个 surface components 和 388 goals 保留。
两组共享恢复：失联 100 ms 后回到最近安全接触参考，保留 actuator preload；
最多沿最近实测法向探触 3 mm。至少一指连续正常接触 150 ms 才算稳定恢复，8 s timeout 后继续。
≥8 N 先局部卸力，持续 >12 N/20 ms 等硬件保护保留。恢复层不接收 GT、未来目标或撤离时刻。

`--withdraw-at 8` 注入 25 mm 真实撤离，撤离后不自动返回。
`--task hold --cells A-MCC --duration 25 --withdraw-at 4` 隔离恢复机制，不查询 Oracle，
撤离后保持撤离位置；加 `--no-recovery` 做开关对照（相同 force relief）。
hold 对照不作为 A/B 遍历实验，也不以 hold 覆盖评价探索能力。

`review.py` 读取同一次 NPZ/JSON，生成 comparison PNG、HTML、aggregate。
`--video` 使用实际保存 q 的 FK 回放，不重算指标。

Windows 中文工作区通过 MuJoCo VFS 读取二进制 meshes；缺失依赖隔离在 `.runtime/`。
运行脚本为 `scripts/run_new_i04_mcc.ps1`。

本机 28 项接口/恢复语义与执行/覆盖权限/force-sign/机器人与 MCC 回归检查通过。
正式实验结果与 source/config/environment hashes 由 generated 目录的 manifest 保存。

2026-10-13：mustard 自然/撤离遍历＋hold 恢复开关共 18 次运行全部达到预算，
独立审计通过；hold 开启恢复 3/3、关闭 0/3。完整覆盖未完成、四指参与仍待改善。
见 [`../evidence/2026-10-13_NEW_I04_MUSTARD_RECOVERY.md`](../evidence/2026-10-13_NEW_I04_MUSTARD_RECOVERY.md) 和
[`../generated/new_i04_mustard_recovery_review_20261013/index.html`](../generated/new_i04_mustard_recovery_review_20261013/index.html)。
