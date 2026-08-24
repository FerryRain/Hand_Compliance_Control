# Finger DP v1

本目录只包含新 v1。它不读取旧 checkpoint、旧短数据，也不把 Fingertip MCC 作为 fallback。
当前状态是
`CORE_UNIT_TESTED / DATASET-D RAW SPATIAL REPLAY PASSED / DATASET-I NOT READY / NOT_TRAINED`。

## 控制链

```text
causal q/dq + 200 ms filtered force/contact/validity
+ contact geometry + real/past and planned/future wrist state
 -> shared per-finger TCN + condition encoder
 -> conditional diffusion relative command chunk
 -> measured-q anchoring
 -> DP Action Authority Filter
 -> 500 Hz interpolation + independent Runtime Guard
```

Wrist MCC 拥有 collective compliance；Finger DP 拥有 differential/local contact realization 与
handover。正式数学与数据/evaluation 边界见
[`../DP_CONTROLLER_V1_PROTOCOL.md`](../DP_CONTROLLER_V1_PROTOCOL.md)；下一步实施顺序、
Dataset-D/Dataset-I 权限、数据门禁和 scaling curve 见
[`../M4_DP_GUIDE.md`](../M4_DP_GUIDE.md)。

## 文件

| 文件 | 用途 |
| --- | --- |
| `contracts.py` | 因果 observation、shape、validity 与 version contract |
| `contact_hysteresis.py` | 真实 force 的 time-confirmed MAKE/BREAK 与 `A_actual` |
| `force_history.py` | 500 Hz causal LPF/anti-alias，降到 100 Hz × 20 frames |
| `policy.py` | shared force TCN、state/wrist encoder、conditional diffusion core |
| `action_chunk.py` | measured-q anchored command-imitation labels 与 seam blending |
| `authority_filter.py` | DAQP 求解的 `P_C J_n Delta q` authority QP 与 opposition metrics |
| `guard_state_machine.py` | soft recovery、signed-compression release、hold、buffer reset |
| `dataset.py` | versioned HDF5 physical replay schema 与 hard acceptance audit |
| `inverse_replay.py` | spatial inversion 与 temporal reversal 的显式独立 API |
| `spatial_inverse_data.py` | moving-object forward physics、原序 q command、fixed-object FR3 raw replay、paired HDF5/audit |
| `spatial_inverse_visual.py` | 同步 forward/replay 视频与 force/contact/dashboard |
| `spatial_inverse_demo.py` | 当前最小闭环的唯一复现入口 |
| `repair_oracle.py` | simulator-only non-MCC local horizon force repair |

## 复现

从仓库根目录：

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m unittest Module.tests.test_finger_dp_core Module.tests.test_finger_dp_data -v
MUJOCO_GL=osmesa $PY -m Module.module_4_finger_dp.spatial_inverse_demo --require-accepted
```

`handcomp` 必须包含 `daqp`（当前环境已安装）；它用于 500 Hz 小规模凸 authority QP，
不是 learned controller 或 Finger MCC fallback。

当前有效的数据链审计输出位于
`Module/generated/visual_demo/spatial_inverse_v1/`：

- `forward_replay_pair.h5`：1500-step forward 与 replay 完整 causal logs；
- `summary.json`：raw gate、source provenance 与 training authorization；
- `forward_replay_audit.png`：两侧 fresh force/contact、palm tracking 与空间反向运动；
- `forward_spatial_inverse_replay.mp4`：左 moving-object forward、右 fixed-object FR3 replay；
- `README.md`：该次运行的边界与 verdict。

严格门禁命令：

```bash
MUJOCO_GL=osmesa $PY -m Module.module_4_finger_dp.spatial_inverse_demo --require-accepted
```

未通过 raw replay gate 时返回码 `2`。当前 3 秒固定配置预期通过：spatial-only、same-time
forward q command、zero finger repair，并重新测量 replay force/contact。但 forward provenance
仍是 simulator-only Fingertip MCC，因此该输出被硬标记为 `Dataset-D diagnostic`、
`formal_dataset_i_ready=false`、`training_allowed=false`；视频不得称为 DP evaluation。

下一步不是直接批量采集：先用 1–4 个 Dataset-D episode 验证 overfit 与 closed-loop imitation，
同时开发 non-MCC forward oracle 并做 20-episode Dataset-I pilot。正式训练继续关闭。
