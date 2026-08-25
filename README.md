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
