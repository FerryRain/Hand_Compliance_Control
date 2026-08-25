# M04-H：共享 Whole-hand MCC 执行层

本目录提供 Explicit baseline 与 DPRef main 共同使用的低层执行层：

```text
normal-force error
 -> resultant component -> Wrist MCC
 -> internal component  -> Finger MCC

nominal finger reference + role intention
 -> deterministic Role Interpreter
 -> same Wrist/Finger MCC
 -> M03 guard
```

| 文件 | 用途 |
| --- | --- |
| `coordinator.py` | grasp-map resultant/internal decomposition |
| `wrist_mcc.py` | 6D wrist admittance 与 selected compliance DOFs |
| `robot_control.py` | FR3 pose IK、wrench estimator 与 actuator command |
| `reference_interpreter.py` | KEEP/RELEASE/FREE/MAKE FSM、request confirmation、force ramp |
| `runner.py` | 23-DoF FR3+LEAP 物理 runner；支持可替换 reference source |
| `g1a_benchmark.py` | 旧 8 N-priority shared command/force/guard 审计 |
| `benchmark.py`、`demo.py` | 历史 MCC-only E05 |
| `visual_demo.py` | 从 trace 渲染关键帧和 15 s 视频 |

当前接触优先 Exp.2 的 shared stack 还包含：MAKE/recontact 独立 acquisition MCC、确认期 50%
approach、从 measured load 开始的 force ramp，以及 acquisition→KEEP 的 MCC state transfer。
soft-force 事件不主动 release；MuJoCo force 只用于持续高力、多指同时高力和 penetration 诊断。

复现旧 shared safety 审计与当前相关单测：

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m Module.module_4_whole_hand_mcc.g1a_benchmark
$PY -m unittest Module.tests.test_e05_mcc_full_robot -v
```

旧 `G1a=PASS` 只对应 2026-08-24 的 8 N-priority profile，当前状态为
`ARCHIVED_PRE_RETUNE`。接触优先 Exp.2 以普通 Plain MCC 为绝对参考，并在完全相同的当前 shared
stack 内比较 Passive/Reactive/DPRef；三者不设置策略 Gate。当前证据见
[`../evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md`](../evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)。
