# 视频与计划输出分级

`outputs/` 只按验收状态和开发阶段保存结果，不能仅凭文件名中的
`final`、`validated` 或版本号判断其可交付性。

- `debug/00_smoke_and_probes/`：冒烟测试、传感器探针和诊断视频。
- `debug/10_legacy_surface_methods/`：FR3 迁移前或尚未通过当前完整规则的历史方法。
- `debug/20_fr3_planning/`：FR3 规划与动态调试视频，包括被拒绝的 `_debug_fr3_*`。
- `reference/accepted_xarm6/`：已经完成数值与视觉审核、仅供回归比较的 xArm6 基线。
- `deliverables/fr3/`：当前 FR3 + LEAP Hand 完整交付视频。只有全程规划、GPU
  动力学、接触/碰撞指标和 Codex 逐帧视觉审核全部通过后才能写入。

NPZ 计划文件暂时保留在 `outputs/` 根目录，避免破坏历史复现实验命令。新调试视频
应直接写入对应 `debug/` 子目录；正式视频不得从调试目录直接改名冒充交付结果。
