# M03：Runtime Guards 与共享 force safety

检测 joint limit、over-force、self-collision、arm/finger stall、sensor/wrench/torque/actuator
异常，并提供 MCC/DP 共用的 deterministic force-release 执行层。

| 文件 | 用途 |
| --- | --- |
| `guards.py` | hand-only 可观测 stall/force/joint/self-collision guards |
| `full_robot_guards.py` | arm/finger 分组 reason、local/global hold scope |
| `force_safety_executor.py` | 500 Hz soft recovery、hard release、safe hold、buffer reset、re-entry |
| `command_continuity.py` | wrist normal/tangent 与 per-finger command step limiter |
| `demo.py` | 数值复现 |
| `visual_demo.py` | blockage/force/limit 可视化 |

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m Module.module_3_runtime_guards.demo
$PY -m unittest Module.tests.test_finger_dp_core.CommandContinuityTest \
  Module.tests.test_finger_dp_core.GuardStateMachineTest -v
```

`signed_compression_jacobian` 必须满足 `J_s Δq>0` 表示压缩增加；hard release 执行
`J_s Δq<0`。该模块的命令权限高于 Finger MCC、Finger DP 与 Action Authority Filter。
当前 shared stack 还加入 rapid-loading soft damping、zero-command reentry ramp，以及独立的
wrist-normal/wrist-tangent/finger step bounds；G1a 三个 15 s 条件均通过。证据见
[`../generated/g1a_shared_stack/README.md`](../generated/g1a_shared_stack/README.md)。
