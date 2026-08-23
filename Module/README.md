# Module：FR3 + LEAP Hand 模块化控制实验

`Module/` 是当前实现入口。固定使用 `handcomp`，先验证 M0–M3，再验证 FR3+LEAP
整手 MCC。当前 DP 版本未通过验收，不进入本次提交、结果或结论；后续按新协议重做。

## 当前状态

| 模块/实验 | 状态 | 结论 |
| --- | --- | --- |
| M0-FR3 | `PASSED` | 7-DoF FR3 + 16-DoF LEAP 状态、动作、wrench 与日志契约 |
| M01-FR3 | `PASSED` | Oracle、live FR3 capsules 与 MuJoCo pad/object distance |
| M02-FR3 | `PASSED` | moving-wrist 四指 Fingertip MCC 与 signed force error |
| M03-FR3 | `PASSED` | arm/finger 分组 guards、局部 hold 与 global safe hold |
| FR3+LEAP plant | `PASSED` | adapter 对准中央 palm plate；四个 physical belly pads |
| E05-F-MCC | `EVALUATED / NOT_MET` | 完整运行，但 force/latency 阈值未全部满足 |
| E05-H-MCC | `EVALUATED / NOT_MET` | resultant/internal 解耦有效，但 force/recovery 阈值未全部满足 |
| Finger DP | `REWORK_REQUIRED / NOT_EVALUATED` | 旧实现未通过验收，未纳入本次提交 |

`EVALUATED` 表示 evaluator 和完整 episode 有效；`NOT_MET` 表示控制性能未达到预先冻结
阈值，不等于实验执行失败。

## 先看可视化

直接打开 [`generated/visual_demo/index.html`](generated/visual_demo/index.html)。其中包含：

- M01 Oracle、M02 Fingertip MCC、M03 Runtime Guards；
- FR3 flange 到中央 palm plate 的安装审计；
- 原视频 `t=2.000 s` 提取的自然手部姿态；
- E05-F-MCC 与 E05-H-MCC 两段完整 15 秒 MuJoCo 视频；
- contact、force、palm path、wrist wrench、curvature 与恢复指标。

MCC-only 审阅清单见 [`LOCAL_REVIEW.md`](LOCAL_REVIEW.md)。

## 文件结构

```text
Module/
├── README.md                         # 本文件：结构、接口和复现
├── LOCAL_REVIEW.md                   # 当前 MCC-only 可视化与原始结果索引
├── MASTER_PLAN.md                    # 唯一主任务记录和 Gate
├── PROTOCOL.md                       # M0–M3 冻结验收协议
├── E05_MCC_CURRENT_PROTOCOL.md       # 当前 MCC 场景、姿态、seed 与阈值
├── E05_EVALUATION_PLAN.md            # E05-F/H 分层定义；DP 标记为待重做
├── WHOLE_HAND_COMPLIANCE_DESIGN.md   # resultant/internal 数学、符号与职责
├── common/                           # hand-only 与 23-DoF 状态/日志 contract
├── fr3_leap/
│   └── model.py                      # FR3+LEAP MJCF、中央 mount、自然姿态
├── module_1_oracle_surface_model/    # M01：Oracle 与 full-robot geometry
├── module_2_fingertip_mcc/           # M02：单指/四指 analytical MCC
├── module_3_runtime_guards/          # M03：hand-only/full-robot guards
├── module_4_whole_hand_mcc/          # Wrist MCC、coordinator、E05 evaluator
├── e05_physics/                      # 当前异质强起伏 surface/physics
├── tests/                            # 单元与回归测试
└── generated/
    ├── e05_mcc_current/              # summary/CSV/trace/exact MJCF
    ├── local_review/                 # 当前 MCC PNG/MP4/HTML
    └── visual_demo/                  # M0–M4 总 gallery
```

## 环境

所有命令从仓库根目录运行：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

不要新建环境，也不要把临时输出写入 `screenshots/`。

## M0–M3 单独复现

```bash
$PY -m Module.module_1_oracle_surface_model.demo --seed 7
$PY -m Module.module_1_oracle_surface_model.mesh_demo
$PY -m Module.module_2_fingertip_mcc.demo
$PY -m Module.module_3_runtime_guards.demo
```

M01 full-robot geometry：

```python
from Module.fr3_leap import build_full_robot
from Module.module_1_oracle_surface_model import FullRobotGeometryAdapter

handles = build_full_robot()
geometry = FullRobotGeometryAdapter(handles)
capsules_world = geometry.world_capsules(data)
distance, witness = geometry.physics_pad_object_distance(data)
```

M02 Fingertip MCC：

```python
from Module.module_2_fingertip_mcc import FingertipMCC

mcc = FingertipMCC()
command_f = mcc.step(plan, direction, desired_force, measured_force)
command_h = mcc.step_force_error(plan, direction, coordinated_internal_error)
```

`direction` 表示正位移会增加接触力；若 SurfaceModel 给出 outward normal `n`，应传
`-n`。

M03 full-robot guards：

```python
from Module.module_3_runtime_guards import (
    FullRobotGuardConfig, FullRobotGuardObservation, FullRobotRuntimeGuards,
)

decision = FullRobotRuntimeGuards(config).evaluate(observation)
```

输出明确区分 `FINGER_LOCAL` 和 `GLOBAL_SAFE_HOLD`。未知 non-tip blockage 只记录可观测
stall 证据，不伪造碰撞位置或法向。

## 复现当前 MCC

完整物理评测：

```bash
$PY -m Module.module_4_whole_hand_mcc.demo
```

它运行 `3 paired episodes × 2 MCC cells × 15 s`，结果写入
`generated/e05_mcc_current/`。正式结论必须同时读取 `execution_status` 和
`performance_verdict`。

从已保存 trace 重建视频，不重新执行控制器：

```bash
MUJOCO_GL=osmesa $PY -m Module.module_4_whole_hand_mcc.visual_demo
$PY -m Module.visual_demo
```

## 回归测试

```bash
$PY -m unittest discover -s Module/tests -v
```

重点检查 23-DoF 分组、自然姿态、中央 mount、belly-pad parent/orientation、Oracle、MCC、
runtime guards、wrench/internal projector 和 visual-demo 完整性。

## 当前边界

- 结果来自 gravity-off MuJoCo 控制隔离实验，不是硬件结果；
- MCC 完整曲线使用 shadow guards；正式 transactional executor 在 M06；
- 当前 DP 代码、数据、模型和指标均不属于本次提交的已验收内容；
- planner、GPIS 与 full main-vs-baseline 尚未开始，继续受 G1/G3 约束。
