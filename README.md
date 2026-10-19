# FR3 + LEAP Hand 模块化柔顺控制

当前项目唯一入口是 [`Module/`](Module/README.md)。所有当前实现、协议、测试和新增产物均位于
`Module/`；仓库其他目录属于历史或并行工作，不作为本项目的默认入口。

## 当前架构

```text
Wrist trajectory / planner -> Wrist MCC -> FR3

hand/contact history + future wrist plan
 -> Finger DP Reference Generator
 -> nominal finger trajectory + role intention
 -> coordinated Finger MCC
 -> LEAP Hand
```

Finger DP 当前定位为多指 nominal trajectory/role generator，而不是高频低层力控制器。Wrist MCC
负责 collective/resultant compliance，Finger MCC 负责 local/internal compliance。

## 当前实验状态

- 2026-10-19目标更新：[I04A/B新目标与共同条件](Module/I04_AB_WHOLE_HAND_PROTOCOL.md)。
  I04B采用手掌与四指联合规划、主动指腹接触和整瓶遍历；I04A保持wrist-only任务输入。
  非对照部分统一为同一模型、初态、自然手形约束、MCC、保护/恢复和评价规范；公共代码迁移与新配对评测待完成，DPRef暂缓。

- 2026-10-19：Mustard 四指协同整瓶遍历完成，8192/8192顶点、388/388目标覆盖；2,638,518个原生状态连续独立重放通过。
  [运行与下载说明](Module/I04_WHOLE_HAND_RUNBOOK.md) · [更新日志](Module/CHANGELOG.md) ·
  [中文报告](Module/generated/whole_hand_surface_run10_20261019/review.html) ·
  [实验记录](Module/evidence/2026-10-19_I04_WHOLE_HAND_SURFACE.md)。报告、视频和完整轨迹随分支发布，大文件请用 `git lfs pull` 下载。
  本轮是v4→v6→v7→v8连续开发接续，仅MCC、固定Mustard、零重力；最长全手有效接触中断146.156秒，换面连续性仍需优化。

- 2026-10-13历史记录：新 I04 已完成 mustard 的 18 次 MCC + 共享接触恢复运行：允许短暂失联，恢复后继续探索；DPRef 延后。
  hold 撤离对照恢复开/关为 3/3、0/3，完整表面覆盖未完成。
  见 [mustard 报告](Module/generated/new_i04_mustard_recovery_review_20261013/index.html) 和
  [冻结协议](Module/I04_MUSTARD_RECOVERY_PROTOCOL.md)。旧 Bunny 首轮数字另存历史证据。

- M0–M3：FR3+LEAP robot、Oracle SurfaceModel、Fingertip MCC 和 Runtime Guards 已实现；
- Exp.1：`Whole-hand MCC vs. DP-direct` 已完成，作为低层 controller replacement 消融；
- Exp.2：`Plain MCC / Passive / Reactive / DPRef+MCC` 已完成接触优先重评；
- E05 只报告性能与诊断指标，不给策略设置 `PASS/FAIL`；
- MuJoCo fingertip force 只作持续高力、多指同时高力和 penetration 诊断，单个瞬时峰值不决定
  策略优劣；
- Exp.3 不属于固定 wrist 的 E05，位于 I05 后作为最终 active-planner ablation。

Exp.2 三条件 aggregate 中，普通 Plain MCC 是绝对接触保持参考；在严格共享执行栈的
Passive/Reactive/DPRef 三者中，DPRef 的 contact continuity、平均接触数与
`N_c>=2` supported traversal 最好，但第四指参与和四指同时接触仍需改善。

## 首要审阅入口

- [Exp.1 + Exp.2 统一网页](Module/generated/e05_exp1_exp2_review/index.html)
- [当前接触优先重评证据](Module/evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)
- [完整模块结构与复现说明](Module/README.md)
- [主任务与实验顺序](Module/MASTER_PLAN.md)

## 环境与复现

固定使用 `handcomp`：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python

# 当前相关回归
$PY -m unittest Module.tests.test_e05_strategy_review \
  Module.tests.test_e05_mcc_full_robot \
  Module.tests.test_finger_dp_core -v

# Exp.2：DPRef inference 必须使用 CUDA
$PY -m Module.module_4_finger_dp.exp2_benchmark --device cuda:0

# GPU/EGL 可视化重建
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  $PY -m Module.module_4_finger_dp.exp2_visual

# 重建统一网页
$PY -m Module.e05_strategy_review
```

DPRef 训练与推理禁止 CPU fallback；MuJoCo physics 本身仍使用其 physics backend。完整文件结构、
各模块独立复现方式和适用范围以 [`Module/README.md`](Module/README.md) 为准。
