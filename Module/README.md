# Module 使用说明：M0–M4 与 MCC-only E05

当前版本已经完成 FR3 + Leap Hand 的 MCC 分支。E05 只评测 `E05-F-MCC` 和
`E05-H-MCC`；DP 未实现、未运行，也没有 DP 指标。固定环境为 `handcomp`。

## 当前状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| M0/P0-FR3 | `PASSED` | 23-DoF 分组状态、wrench frame/sign、sensor validity、JSONL |
| M01-FR3 | `PASSED` | live FR3 link capsules、Oracle clearance、MuJoCo pad/object distance |
| M02-FR3 | `PASSED` | moving-wrist finger IK、signed force error、transition reset |
| M03-FR3 | `PASSED` | arm/finger 分组 stall、wrench/torque/limit/sensor/controller guards |
| M04-R/W/C/H | `PASSED_STRUCTURE / MOUNT_REVIEW_OPEN` | 控制链通过；发现 link8→palm 有 80 mm 不可见 mount offset，待下一版修复并重测 |
| E05-F-MCC | `EVALUATED / NOT_MET` | 三个正式 episode 全部执行；force peak 等阈值未全部满足 |
| E05-H-MCC | `EVALUATED / NOT_MET` | 三个正式 episode 全部执行；force RMSE/settling/peak 未全部满足 |

`EVALUATED` 表示实验完整有效；`NOT_MET` 表示控制性能仍需改进，不把已经完成的评测写成
`FAILED`。

### 审阅基线的已知问题

当前 `M04-R` 在 MuJoCo kinematic tree 中确实把 `palm_lower` 作为 `fr3v2_link8` 的固定
child，因此并非自由漂浮；但 v1 使用 `MOUNT_POSITION_LINK8_M=[0,0,0.08]`，中间没有
显式 adapter geometry，视觉上和机械定义上都留下了 `80 mm` 的不可见连接段。该问题
已确认，当前正式 E05 结果只作为修复前基线；修正 mount 后必须重新运行全部 MCC 评测
和视频，不能沿用这里的数值。

## 目录结构

```text
Module/
├── README.md                         # 本文件：结构、API、复现与结果
├── MASTER_PLAN.md                    # 唯一主任务记录和 Gate
├── PROTOCOL.md                       # M0–M3 协议及 FR3 extension
├── E05_MCC_FR3_PROTOCOL.md           # MCC-only E05 冻结协议/阈值/seed
├── E05_EVALUATION_PLAN.md            # E05-F/H 当前定义
├── WHOLE_HAND_COMPLIANCE_DESIGN.md   # resultant/internal 数学与符号
├── fr3_visual_demo.py                # 正式 trace -> FR3 视频/dashboard
├── visual_demo.py                    # M0–M4 总 gallery 生成器
├── common/
│   ├── contracts.py                  # 原 hand-modules.v1
│   └── full_robot_contracts.py       # fr3-leap-modules.v1
├── fr3_leap/
│   └── model.py                      # 7 FR3 + 16 Leap MuJoCo builder/audit
├── module_1_oracle_surface_model/
│   ├── geometry.py / surface_model.py # analytic Oracle primitives/interface
│   ├── mesh_surface.py / mesh_demo.py # Bunny/YCB-style showcase
│   ├── demo.py / visual_demo.py       # 数值验收与图形复现
│   └── robot_geometry.py             # M01-FR3 live-state adapter
├── module_2_fingertip_mcc/
│   ├── controller.py                 # MCC + step_force_error
│   ├── full_robot.py                 # 四指 moving-wrist wrapper
│   └── benchmarks.py / demo.py / visual_demo.py
├── module_3_runtime_guards/
│   ├── guards.py
│   ├── full_robot_guards.py          # arm/finger 分组 guards
│   └── demo.py / visual_demo.py
├── module_4_whole_hand_mcc/
│   ├── coordinator.py                # H_A decomposition
│   ├── wrist_mcc.py                  # 6D Wrist MCC
│   ├── robot_control.py              # palm IK + wrench estimator
│   ├── runner.py / benchmark.py      # E05 runner/evaluator
│   └── demo.py / visual_demo.py       # CLI 与正式 trace renderer
├── e05_physics/                      # fixed-palm PHY-v3 历史实现，不是默认入口
├── assets/                           # Bunny mesh 与 attribution
├── tests/                            # 49 个 unittest 回归测试
├── evidence/                         # 命令、hash、环境和 verdict
└── generated/
    ├── e05_mcc_fr3_v1/
    │   ├── summary.json              # 正式 aggregate/threshold verdict
    │   ├── episodes.csv              # 六个 episode 的逐项指标
    │   ├── base_traces.npz           # 视频和复盘使用的完整 trace
    │   ├── model_audit.json
    │   └── generated_fr3_leap.xml    # 本次实际编译的 exact MJCF
    ├── visual_demo/                  # gallery、PNG、GIF、两段 15 s MP4
    └── e05_physics_v3/               # fixed-palm 历史 evidence
```

`__pycache__/`、根目录 `screenshots/`、IDE 配置和旧 hand-only 采集数据都不是 Module
复现输入，不应提交为实验产物。

## 最快看懂：可视化

直接打开 `Module/generated/visual_demo/index.html`。页面包含：

- M01 Oracle、M02 analytical MCC、M03 guards；
- 23-DoF FR3+Leap 整机与 fingertip-belly close-up；
- `E05-F-MCC`、`E05-H-MCC` 两段完整 15 秒正式 trace 视频；
- contact、force、2D palm path、wrist wrench、curvature 与 recovery dashboard。

重新生成 gallery（不会重跑 E05）：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.visual_demo
```

从正式 trace 重新生成 FR3 视频：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.fr3_visual_demo
```

## 从零复现顺序

所有命令都从仓库根目录运行，且固定使用同一个解释器：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PYTHON=/home/ferry/data/Anaconda/envs/handcomp/bin/python
```

### 1. 单独验证 M01–M03

```bash
$PYTHON -m Module.module_1_oracle_surface_model.demo --seed 7
$PYTHON -m Module.module_2_fingertip_mcc.demo
$PYTHON -m Module.module_3_runtime_guards.demo
```

三个命令均打印 JSON，成功时 `passed: true` 且进程返回 `0`。它们分别复现 Oracle
distance/normal/clearance/candidates、Static/Sliding/Curved MCC，以及 free-motion、stall、
joint-limit、over-force 和 self-collision guards。

### 2. 验证 M0–M4 full-robot contract 与控制单元

```bash
$PYTHON -m unittest \
  Module.tests.test_full_robot_contracts \
  Module.tests.test_fr3_leap_model \
  Module.tests.test_full_robot_modules \
  Module.tests.test_whole_hand_mcc -v
```

该组检查 23-DoF 分组契约、四个 belly pad 的 parent/orientation、live FR3 geometry、
moving-wrist MCC、full-robot guards、projector algebra、palm IK 和 wrench sign。

### 3. 运行短物理 smoke（不覆盖正式结果）

```bash
$PYTHON -m Module.module_4_whole_hand_mcc.demo \
  --quick \
  --output-dir /home/ferry/data/tmp/e05_mcc_fr3_smoke
```

`--quick` 只有 3 秒且输出为 `EVIDENCE_ONLY`；程序禁止把它写进正式默认目录。

### 4. 运行正式 E05-F/H MCC

```bash
$PYTHON -m Module.module_4_whole_hand_mcc.demo
```

这会严格运行 `3 seeds x 2 cells x 15 s`，并重建
`Module/generated/e05_mcc_fr3_v1/{summary.json,episodes.csv,base_traces.npz,
model_audit.json,generated_fr3_leap.xml}`。正式判定看 `summary.json` 中彼此独立的
`execution_status` 与 `performance_verdict`。

### 5. 从正式 trace 重建视频与总页面

```bash
$PYTHON -m Module.fr3_visual_demo
$PYTHON -m Module.visual_demo
```

然后打开 `Module/generated/visual_demo/index.html`。`fr3_visual_demo` 只重放正式 NPZ，
不会悄悄重新运行控制器；`Module.visual_demo` 重建 M01–M03 图并检查 FR3 视频是否齐全。

### 6. 完整回归

```bash
$PYTHON -m unittest discover -s Module/tests -v
```

当前基线应输出 `Ran 49 tests ... OK`。

## 模块用法

### M0：全机器人契约

```python
from Module.common import FullRobotStateSnapshot, FullRobotJsonlLogger
```

状态明确分成 `arm_q[7]` 和 `finger_q[16]`；wrench 同时记录 frame、reference、acting-on
和 estimator。`A_actual` 只由 measured `contact_states` 得到。旧 schema 保持兼容。

### M01：Oracle + live FR3 geometry

```python
from Module.fr3_leap import build_full_robot
from Module.module_1_oracle_surface_model import FullRobotGeometryAdapter

handles = build_full_robot()
adapter = FullRobotGeometryAdapter(handles)
capsules_world = adapter.world_capsules(data)
distance, witness = adapter.physics_pad_object_distance(data)
```

解析几何与 Bunny/YCB showcase 保留；approximate capsule 不冒充 exact mesh certificate。

### M02：Finger MCC

```python
from Module.module_2_fingertip_mcc import FingertipMCC

mcc = FingertipMCC()
cmd_f = mcc.step(plan, direction, desired_force, measured_force)  # E05-F
cmd_h = mcc.step_force_error(plan, direction, internal_error)     # E05-H
```

`direction` 是正位移增加接触力的单位方向；物体 outward normal 为 `n` 时传 `-n`。

### M03：Full-robot guards

```python
from Module.module_3_runtime_guards import (
    FullRobotGuardConfig, FullRobotGuardObservation, FullRobotRuntimeGuards,
)

decision = FullRobotRuntimeGuards(config).evaluate(observation)
```

Arm 与四根 finger 分组积累 stall；输出区分 `FINGER_LOCAL` 与 `GLOBAL_SAFE_HOLD`。未知
non-tip blockage 仍只记录局部证据，不伪造碰撞点或法向。

### M04：整手协调 MCC

```text
normal force error
        |
        v
ContactForceCoordinator (H_A = G_A B_A)
        | resultant                 | internal/differential
        v                           v
FR3 torque wrench -> Wrist MCC          4 x Fingertip MCC
```

- F：Wrist MCC off，四指接完整 local error；
- H：Wrist MCC 调 hand-side resultant，active fingers 只接 `N_H e_lambda`；
- coordinator 只使用 hysteresis-confirmed `A_actual`；
- object 固定无 mocap，FR3 实际执行 palm trajectory；
- Wrist MCC 是 6D 实现，本 E05 只激活 collective-normal translation projector。

## 正式 E05 复现

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.module_4_whole_hand_mcc.demo
```

正式 protocol SHA-256：

```text
483bd7ccf98ad5afec9bfb2d2912d2b9882d2a4045a194582b879b8adc51664a
```

命令、模型审计、代码哈希、视频元数据和逐项 verdict 已登记在
[`evidence/2026-08-23_M0_M04_E05_MCC_FR3.md`](evidence/2026-08-23_M0_M04_E05_MCC_FR3.md)。

三个配对 episode 的 aggregate：

| 指标 | E05-F-MCC | E05-H-MCC |
| --- | ---: | ---: |
| mean contact continuity | 99.990% | 99.990% |
| mean contact count | 3.750 | 3.710 |
| mean force RMSE | 0.773 N | 1.017 N |
| worst peak force | 11.08 N | 13.19 N |
| mean Y traversal | 173.9 mm | 175.3 mm |
| mean controller P95 | 1.249 ms | 1.263 ms |
| H wrist Fz RMSE | — | 1.984 N |
| H internal leakage P95 | — | 0.0135 N |

F 未达：deadline miss（一个 episode 略超）、force-violation probability、8 N peak。H 未达：
force RMSE、force settling（noisy-pose worst `5.998 s`）、8 N peak。两者都保持
`EVALUATED`。

## 测试

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m unittest discover -s Module/tests -v
```

## 边界

- 本轮没有 DP，不能形成 MCC-vs-DP 结论；
- 正式协议关闭 gravity 以隔离 contact control，不能外推为 gravity-on 或硬件；
- `E05_PHYSICS_PROTOCOL.md` 与 `generated/e05_physics_v3/` 是 fixed-palm 历史证据，未被
  新结果覆盖；
- v1 暴露的 force peak/settling 问题只能在新 v2 协议中修正，不能回写本次结果。
