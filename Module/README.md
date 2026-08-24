# Module：FR3 + LEAP Hand 模块化控制实验

`Module/` 是当前实现入口。固定使用 `handcomp`。M0–M3 与 FR3+LEAP MCC 已有正式
结果；Finger DP v1 已冻结架构并开始独立实现。新的真实
`forward physical → spatial inverse → physical replay` 最小链已通过 raw replay gate，但其
forward 暂由 Fingertip MCC 采集，只属于 Dataset-D pipeline diagnostic；正式 Dataset-I 尚未
生成，因此没有启动训练，也没有 E05-DP 性能结论。

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
| Finger DP v1 core | `IMPLEMENTED / UNIT_TESTED` | force history、relative chunk、authority QP、guard、diffusion core |
| Spatial inverse Dataset-D | `RAW_REPLAY_GATE_PASSED` | 3 s 真实 forward、同序原始 q command、固定物体 FR3 replay、fresh force/contact |
| Formal Dataset-I | `NOT_READY` | 仍缺非 MCC forward provenance；禁止用 Dataset-D 冒充正式 teacher |
| E05-H-DP | `NOT_EVALUATED` | 数据未通过前禁止训练和评测 |

`EVALUATED` 表示 evaluator 和完整 episode 有效；`NOT_MET` 表示控制性能未达到预先冻结
阈值，不等于实验执行失败。

## MCC 版本边界

- 唯一当前协议是 `E05_MCC_CURRENT_PROTOCOL.md`；
- 唯一当前正式结果目录是 `generated/e05_mcc_current/`；
- fixed-palm `E05-PHY-v3` 不再作为独立 MCC 版本、协议或实验结论保留；
- `e05_physics/scene.py` 与 `extreme_surface.py` 仅作为当前 FR3+LEAP 评测共享环境保留；
- `Module/` 外的历史目录不属于本模块清理范围，保持不变；
- 旧 DP checkpoint、旧短数据与旧指标不得复用；新实现唯一入口是
  `module_4_finger_dp/` 与 `DP_CONTROLLER_V1_PROTOCOL.md`；
- 只有正式 Dataset-I physical audit 通过后才允许训练或写 E05-DP 结论；Dataset-D 即使 raw
  replay 通过也只验证数据链。

## 先看可视化

直接打开 [`generated/visual_demo/index.html`](generated/visual_demo/index.html)。其中包含：

- M01 Oracle、M02 Fingertip MCC、M03 Runtime Guards；
- FR3 flange 到中央 palm plate 的安装审计；
- 原视频 `t=2.000 s` 提取的自然手部姿态；
- E05-F-MCC 与 E05-H-MCC 两段完整 15 秒 MuJoCo 视频；
- contact、force、palm path、wrist wrench、curvature 与恢复指标。

新的 inverse-data 可视化位于
[`generated/visual_demo/spatial_inverse_v1/`](generated/visual_demo/spatial_inverse_v1/)：视频左侧
是真实 moving-object forward physics，右侧是 fixed-object FR3+LEAP physical replay。画面明确
显示 `SPATIAL_ONLY`、`SAME_ORDER`、两侧 fresh force/contact 与零 finger-command 映射误差。
它是 Dataset-D 数据链审计，不是 DP 策略评测。

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
├── DP_CONTROLLER_V1_PROTOCOL.md       # 新 DP v1 的冻结 observation/action/teacher/eval 规范
├── M4_DP_GUIDE.md                     # Dataset-D smoke、Dataset-I generator、扩量与 Gate 指导
├── WHOLE_HAND_COMPLIANCE_DESIGN.md   # resultant/internal 数学、符号与职责
├── common/                           # hand-only 与 23-DoF 状态/日志 contract
├── fr3_leap/
│   └── model.py                      # FR3+LEAP MJCF、中央 mount、自然姿态
├── module_1_oracle_surface_model/    # M01：Oracle 与 full-robot geometry
├── module_2_fingertip_mcc/           # M02：单指/四指 analytical MCC
├── module_3_runtime_guards/          # M03：hand-only/full-robot guards
├── module_4_whole_hand_mcc/          # Wrist MCC、coordinator、E05 evaluator
├── module_4_finger_dp/                # 新 DP v1；无旧 checkpoint/fallback
│   ├── contracts.py                  # 因果 observation 与 validity contract
│   ├── contact_hysteresis.py         # time-confirmed A_actual MAKE/BREAK
│   ├── force_history.py              # 500→100 Hz causal LPF/anti-alias + 200 ms buffer
│   ├── policy.py                     # shared per-finger TCN + conditional diffusion core
│   ├── action_chunk.py               # measured-q anchored command chunks
│   ├── authority_filter.py           # DAQP contact-normal authority QP 与 opposition metrics
│   ├── guard_state_machine.py        # release/hold/reset 状态机
│   ├── dataset.py                    # HDF5 physical command-imitation schema + hard audit
│   ├── inverse_replay.py             # spatial/temporal 明确分离的 SE(3) proposal
│   ├── spatial_inverse_data.py        # 真 forward physics、同序 q_cmd、raw FR3 replay、pair HDF5/audit
│   ├── spatial_inverse_visual.py      # forward/replay 双画面视频与 dashboard
│   ├── spatial_inverse_demo.py        # 当前 inverse-data 最小闭环入口
│   └── repair_oracle.py              # simulator-only non-MCC replay-repair primitive
├── e05_physics/                      # 共享强起伏 surface/scene；无独立 evaluator
├── tests/                            # 单元与回归测试
└── generated/
    ├── e05_mcc_current/              # summary/CSV/trace/exact MJCF
    ├── local_review/                 # 当前 MCC PNG/MP4/HTML
    └── visual_demo/
        └── spatial_inverse_v1/       # 新 forward/replay pair、视频、dashboard、raw HDF5
```

## 环境

所有命令从仓库根目录运行：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

不要新建环境，也不要把临时输出写入 `screenshots/`。
DP authority filter 使用 `handcomp` 已安装的 `daqp` active-set solver。

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

## 复现 Finger DP v1 与空间反演数据链

先运行快速核心测试：

```bash
$PY -m unittest Module.tests.test_finger_dp_core Module.tests.test_finger_dp_data -v
```

运行当前 3 秒真实 forward→spatial-inverse→physical-replay，并生成 paired HDF5、JSON、PNG
和左右同步 MP4：

```bash
MUJOCO_GL=osmesa $PY -m Module.module_4_finger_dp.spatial_inverse_demo --require-accepted
```

默认输出：

- `generated/visual_demo/spatial_inverse_v1/forward_spatial_inverse_replay.mp4`：左右同步物理视频；
- `forward_replay_pair.h5`：forward/replay 各 1500 个 500 Hz causal samples；
- `forward_replay_audit.png`：force、actual contact count、FR3 tracking 与相反空间运动；
- `summary.json`：raw gate、provenance、mapping 与 training authorization。

当前预期 `raw_spatial_replay_audit.accepted=true`，但同时必须看到
`dataset_class=DATASET_D_DIAGNOSTIC`、`formal_dataset_i_ready=false` 和
`training_allowed=false`。这不是矛盾：前者只证明 inverse replay 机械链可工作，后者阻止
MCC-generated forward 被错误宣传为正式 inverse teacher。当前逐指 contact-mask agreement
约 `84.83%`（非空接触连续性为 `100%`），也必须保留报告，不能把 raw pass 写成逐指动力学
完全等价。

当前仍禁止正式训练。下一步按 [`M4_DP_GUIDE.md`](M4_DP_GUIDE.md) 同时隔离验证两件事：
用 1–4 个 Dataset-D episode 做 intentional-overfit/closed-loop smoke test；开发 non-MCC
forward oracle 并完成 20-episode Dataset-I pilot。两条 Gate 都通过后才扩量与进入
`E05-H-MCC vs E05-H-DP`。

## 回归测试

```bash
$PY -m unittest discover -s Module/tests -v
```

重点检查 23-DoF 分组、自然姿态、中央 mount、belly-pad parent/orientation、Oracle、MCC、
runtime guards、wrench/internal projector 和 visual-demo 完整性。

## 当前边界

- 结果来自 gravity-off MuJoCo 控制隔离实验，不是硬件结果；
- MCC 完整曲线使用 shadow guards；正式 transactional executor 在 M06；
- DP v1 core 是实现/单测状态；Dataset-D raw spatial replay 已验收，正式 Dataset-I、训练模型与
  E05-DP 均未验收；
- planner、GPIS 与 full main-vs-baseline 尚未开始，继续受 G1/G3 约束。
