# 当前 FR3+LEAP / MCC / DPRef 审阅入口

本页只列当前可复现证据。E05 Exp.1/2 均为描述性策略评测：只报告性能、相对优劣和参考限制
越界，不写策略 `PASS/FAIL` 或 `MET/NOT_MET`。Exp.3 已移到 I05 后的 I06，作为最终
active-planner ablation。M4 的实施顺序与结果解释以
[`M4_DP_GUIDE.md`](M4_DP_GUIDE.md) 为准。

## 首要：E05 Exp.1 + Exp.2 统一网页

- 同一网页中的两组指标、优劣分析、dashboard 与全部视频：
  [`index.html`](generated/e05_exp1_exp2_review/index.html)
- 无策略 verdict 的统一机器数据：
  [`summary.json`](generated/e05_exp1_exp2_review/summary.json)、
  [`metrics.csv`](generated/e05_exp1_exp2_review/metrics.csv)
- 复现说明：
  [`README.md`](generated/e05_exp1_exp2_review/README.md)

只允许 Exp.1 内部和 Exp.2 内部比较；两个实验之间的 shared MCC/guard 版本不同。

## DPRef + shared MCC / Exp. 2 当前接触优先证据

- 一页证据与复现：
  [`2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md`](evidence/2026-08-25_EXP2_CONTACT_PRIORITY_RERUN.md)
- Exp. 2 dashboard：
  [`exp2_comparison.png`](generated/exp2_dpref_mcc/exp2_comparison.png)
- 四条完整视频：
  [`Plain`](generated/exp2_dpref_mcc/plain_whole_hand_mcc_video.mp4)、
  [`Passive`](generated/exp2_dpref_mcc/passive_hold_mcc_video.mp4)、
  [`Reactive`](generated/exp2_dpref_mcc/reactive_heuristic_mcc_video.mp4)、
  [`DPRef`](generated/exp2_dpref_mcc/dpref_mcc_video.mp4)
- machine-readable：
  [`summary.json`](generated/exp2_dpref_mcc/summary.json)
- CUDA training 与 role audit：
  [`training README`](generated/dpref_v1/training_i100/README.md)、
  [`training audit`](generated/dpref_v1/training_i100/dpref_training_and_label_audit.png)、
  [`role coverage`](generated/dpref_v1/training_i100/dpref_role_coverage.png)
- 旧 G1a 8 N-priority profile（仅历史 provenance）：
  [`README`](generated/g1a_shared_stack/README.md)、
  [`video`](generated/g1a_shared_stack/g1a_nominal_video.mp4)

| 3 条件 aggregate | Plain MCC | Passive-Hold | Reactive | DPRef |
| --- | ---: | ---: | ---: | ---: |
| contact continuity | 0.992 | 0.972 | 0.973 | 0.988 |
| average contacts | 3.156 | 2.285 | 2.310 | 2.450 |
| `P(N_c>=3)` | 0.754 | 0.439 | 0.451 | 0.466 |
| four-contact probability | 0.533 | 0.263 | 0.275 | 0.149 |
| supported Y (`N_c>=2`) | 138.87 mm | 89.35 mm | 86.90 mm | 126.09 mm |
| worst peak force（诊断） | 11.650 N | 56.774 N | 52.237 N | 4.825 N |
| multi-pad simultaneous `>8 N` | 0 s | 0 s | 0 s | 0 s |

Plain MCC 的绝对接触保持最好。严格共享栈的 Passive/Reactive/DPRef 中，DPRef 的 continuity、
平均接触数、`P(N_c>=2/3)` 和 supported traversal 最好；但其四指接触和第四指参与明显不足。
MuJoCo force 只作为持续/多指高力与 penetration 诊断，单个瞬时 peak 不产生策略 verdict。
validation 没有 RELEASE label，不能声称 handover generalization。

## 首要：I01 Bunny 连续接触物理验证

建议先打开：

- fixed/variable 同步视频、曲线与结论：
  [`index.html`](generated/i01_bunny_physics/index.html)
- dashboard：
  [`i01_bunny_dashboard.png`](generated/i01_bunny_physics/i01_bunny_dashboard.png)
- seed 7 同步视频：
  [`i01_fixed_vs_variable.mp4`](generated/i01_bunny_physics/i01_fixed_vs_variable.mp4)
- 完整指标与逐 episode 表：
  [`summary.json`](generated/i01_bunny_physics/summary.json)、
  [`episodes.csv`](generated/i01_bunny_physics/episodes.csv)
- 预先冻结的协议：
  [`I01_BUNNY_PROTOCOL.md`](I01_BUNNY_PROTOCOL.md)

| 指标 | fixed `|A|=4` | variable `4->3->4` |
| --- | ---: | ---: |
| actual progress median | `19.885 mm` | `58.429 mm` |
| nonempty-contact fraction | `100%` | `99.933–99.956%` |
| maximum all-contact-loss gap | `0 ms` | `2 ms` |
| worst valid fingertip force | `4.736 N` | `5.475 N` |
| primary/mechanism pass | `0/3` / n.a. | `3/3` / `3/3` |

fixed 三次都在约 20 mm 因持续四指 mode 破坏触发 `FIXED_MODE_LOST`；variable 三次都走到
约 58.4 mm，并由测量确认 handover。两组均无 over-force tick、non-tip collision 或 authority
violation。variable 的中位优势 `38.544 mm` 超过冻结门槛 `10 mm`，所以本协议
`EVALUATED / MET`、`G2=GO`。

这是固定 Bunny、gravity-off 的 MuJoCo control-isolation。碰撞使用 Bunny-derived upper-envelope
hfield，并以完整三角网格残差过滤计分接触；它不是硬件、gravity-on 或完整非凸 mesh 的结论，
也不代表 G1a 或 G3 已通过。

## 0. M06–M12 Geometry-Oracle + MCC baseline

建议先打开：

- 七模块总览：
  [`index.html`](generated/m06_m12_mcc_baseline/index.html)
- 完整 machine-readable 指标：
  [`summary.json`](generated/m06_m12_mcc_baseline/summary.json)
- 扁平性能表：
  [`performance.csv`](generated/m06_m12_mcc_baseline/performance.csv)
- 冻结协议：
  [`M06_M12_MCC_BASELINE_PROTOCOL.md`](M06_M12_MCC_BASELINE_PROTOCOL.md)

| 模块 | 目的 | 当前模块结果 |
| --- | --- | --- |
| M06 | certificate-gated prefix、micro-barrier、真实 snapshot | 5/5 transaction 场景，0 authority violation |
| M07 | 15 个非空 contact modes 与 primitive legality | deterministic，empty-mode violation 为 0 |
| M08 | optimizer 前的 optimistic cheap screen | 4096 candidates，false-negative rate 0 |
| M09 | 五类 primitive 的连续轨迹构造 | 5×32 cases，success rate 1.0 |
| M10 | swept exact audit 与唯一证书权限 | 正例签证书，6/6 adversarial cases 拒绝 |
| M11 | diverse lazy beam + shifted suffix warm start | H2/H3 均保留 exhaustive optimum |
| M12 | terminal cheap continuation / dead-end 检测 | viable/dead-end 正确，ShadowSucc 无执行权 |

Intel i7-13700 本机 P95 为：M06 step `0.168 ms`、M07 legality `1.157 µs`、M08 screen
`3.626 µs`、M09 optimize `1.159 ms`、M10 audit `3.031 ms`、M11 H2/H3 search
`50.97/122.69 ms`、M12 viability `0.203 ms`。各 timing boundary 已写入 `summary.json`，不能把
M09 linearized backend 或 M06 synthetic transaction plant 的 latency 外推成 FR3 physics 性能。

执行状态为 `EVALUATED`，模块协议 performance 为 `MET`。数值 acceptance 使用 analytic plane
和明确标注的 linearized validation backend；Bunny 仅用于可视化。集成 smoke 连接现有
`FullRobotFingertipMCC`，但这份 M06–M12 证据本身没有运行正式 I01 或长程 FR3 MuJoCo
traversal。随后单独授权并执行的 Bunny I01 结果见本页首节；G1 仍为 No-Go，G3 未开始。

## 1. Exp. 1：E05-H-MCC vs E05-H-DP-direct

建议先打开：

- 总审阅页：[`review.html`](generated/e05_h_mcc_vs_dp/review.html)
- 15 s 左右同步 nominal 视频：
  [`e05_h_mcc_vs_dp_side_by_side.mp4`](generated/e05_h_mcc_vs_dp/e05_h_mcc_vs_dp_side_by_side.mp4)
- force/contact/Wrist-MCC/latency dashboard：
  [`e05_h_mcc_vs_dp_dashboard.png`](generated/e05_h_mcc_vs_dp/e05_h_mcc_vs_dp_dashboard.png)
- 逐 episode machine-readable 指标：
  [`summary.json`](generated/e05_h_mcc_vs_dp/summary.json)、
  [`episodes.csv`](generated/e05_h_mcc_vs_dp/episodes.csv)

三组 episode 都完整运行 15 s。前 1 s 是双方相同且不计分的初始化，随后只替换 Finger MCC
与 Finger DP-direct + Authority Filter；Wrist MCC、M03 guard、trajectory、initial state 和
limits 共享。

| aggregate | E05-H-MCC | E05-H-DP-direct |
| --- | ---: | ---: |
| mean contact continuity | 87.30% | 66.69% |
| mean contacts | 3.026 | 1.590 |
| mean force RMSE | 1.381 N | 2.232 N |
| worst peak | 81.35 N | 103.02 N |
| mean Y traversal | 174.2 mm | 158.2 mm |
| controller P95 | 1.35 ms | 12.00 ms |
| 相对 8 N 峰值越界 | 73.35 N | 95.02 N |

Nominal 单组更便于直接看：contact continuity `92.1%/81.6%`，average contacts `3.23/1.81`，
peak `25.05/45.94 N`，Y traversal `173.9/174.7 mm`。DP P95 低于 20 ms，但 contact/force/
recovery 和 authority robustness 均弱于 MCC；这里描述的是相对表现，不给策略判失败。

这里的 DP 是 **DP-direct**。新的 Exp. 2 `Passive/Reactive/DPRef+MCC` 已完成，证据在本页首节；
Exp. 3 不属于 E05，固定在 I05 后的 I06 执行。

### 1.1 冻结 trace 的 failure diagnostic

- 总审阅页：
  [`diagnostics/review.html`](generated/e05_h_mcc_vs_dp/diagnostics/review.html)
- 完整机器结果：
  [`diagnostic_summary.json`](generated/e05_h_mcc_vs_dp/diagnostics/diagnostic_summary.json)
- Dataset-I 与 E05 contact-state coverage：
  [`contact_state_coverage.png`](generated/e05_h_mcc_vs_dp/diagnostics/contact_state_coverage.png)

三组首次持续 `N_c^DP < N_c^MCC` 分别发生于 `1.224/1.148/1.048 s`，即 DP 接管后
`224/148/48 ms`。最近一次 replan 的 Authority Filter intervention 均为 0，因此 filter 不是
这三次首次掉指的直接触发器；但 v1 trace 未保存完整 pre-projection vector，报告中的
`r_AF_safe_proxy` 不能冒充精确定义的 `r_AF`。

I20 的 current `N_c<=2/1/0` 比例为 `8.014%/0.438%/0%`，30.063% 的 200 ms history 含
contact transition：训练数据不是完全没有 transition，但严重恢复状态确实稀缺。两种 controller
的过力都包含多 tick 事件；部分峰值与 M03 `BUFFER_FILL/HARD_RELEASE` 切换处 13–32 mm palm
command target jump 同时发生，必须先修共享安全执行层的连续性。

把 Y motion 只统计在相邻两帧均有 `N_c>=2` 时，nominal 的 supported positive-Y fraction 为
MCC `76.0%`、DP `41.8%`；因此 nominal 的总 traversal 虽接近，DP 主要是在低接触状态下继续
移动。这支持“traversal 下降/保持不是首要根因”的判断。

## 2. Dataset-I 与正式 checkpoint

- 数据/训练总索引：
  [`finger_dp_formal_v1/README.md`](generated/finger_dp_formal_v1/README.md)
- I-Gate 12 s 原始证据：
  [`episode_summary.json`](generated/finger_dp_formal_v1/track_i/igate_dev_crop_0/episode_summary.json)
- I-Pilot20 分类（历史路径 `pilot20/`）：
  [`batch_manifest.json`](generated/finger_dp_formal_v1/pilot20/batch_manifest.json)
- frozen object split：
  [`object_split_manifest.json`](generated/finger_dp_formal_v1/formal_pool_v1/object_split_manifest.json)
- I20 CUDA training（历史路径 `training_d20/`）：
  [`training_summary.json`](generated/finger_dp_formal_v1/training_d20/training_summary.json)
- I20 object-disjoint held-out：
  [`closed_loop_summary.json`](generated/finger_dp_formal_v1/heldout_validation/closed_loop_summary.json)
- I100 对照：
  [`training_summary.json`](generated/finger_dp_formal_v1/training_d100/training_summary.json)、
  [`closed_loop_summary.json`](generated/finger_dp_formal_v1/heldout_validation_d100/closed_loop_summary.json)

I-Gate 的 forward/replay continuity 为 `1.000/1.000`，peak `6.412/4.647 N`，mapping residual
与 repair rate 都是 0；forward provenance 明确是 non-MCC privileged oracle。I-Pilot20 为
`12 RAW_VERIFIED / 0 REPAIRED / 8 REJECTED`。

I20/I100 分别有 20/100 episodes、11180/55900 anchors。两者 held-out continuity 都为 1.0；
I20 average contacts/peak 为 `2.664/5.957 N`，I100 为 `2.543/7.496 N`，所以 Exp. 1 选择 I20。

## 3. E05 分布缺口证据

训练 Dataset-I 位移约 28–36 mm/12 s；正式 E05 是 180 mm/15 s，并有 4 mm step。一次
160 mm/12 s matched pilot 的 forward continuity 为 `0.9735`、forward/replay peak 为
`34.66/9.73 N`，因此被 raw gate 正确拒绝，训练样本数为 0：

[`manifest.json`](generated/finger_dp_formal_v1/e05_matched_pilot/e05_match_train0_pilot/manifest.json)

这个失败轨迹只用于 diagnosis，不能为了改善 E05 结果静默放入训练。

## 4. M0–M3 与机器人结构

打开 [`generated/visual_demo/index.html`](generated/visual_demo/index.html)，可查看：

- M01 surface/normal/clearance；
- M02 static/sliding/curved MCC；
- M03 blockage/over-force/joint-limit/self-collision；
- FR3 flange 到中央 palm plate 的安装；
- 四个 tip-body belly pads 和大尺寸强起伏 surface。

单独结构图：

- 自然接触姿态：[`natural_pose_audit.png`](generated/local_review/natural_pose_audit.png)
- 中央掌心安装：[`mount_center_audit.png`](generated/local_review/mount_center_audit.png)

## 5. MCC-only 历史正式结果

- 页面：[`mcc_review.html`](generated/local_review/mcc_review.html)
- E05-F-MCC：[`mcc_f_video.mp4`](generated/local_review/mcc_f_video.mp4)
- E05-H-MCC：[`mcc_h_video.mp4`](generated/local_review/mcc_h_video.mp4)
- summary：[`summary.json`](generated/e05_mcc_current/summary.json)

这些是 `E05_MCC_CURRENT_PROTOCOL.md` 的 MCC-only 结果；正式 MCC/DP 配对应读取第 1 节的新
协议与结果，不能混用阈值或 episode 定义。

## 6. Diag-MCC diagnostic（非正式 E05）

- 长 D-Gate 页面：
  [`review.html`](generated/whole_hand_dp_long_v1/review.html)
- 最小 spatial-inverse 双画面：
  [`forward_spatial_inverse_replay.mp4`](generated/visual_demo/spatial_inverse_v1/forward_spatial_inverse_replay.mp4)

它们分别证明 learning/execution pipeline 与 spatial inversion mechanics；teacher 使用 MCC，
所以不能替代 Dataset-I 或正式 E05。

## 7. 复现

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python

# 当前全量测试
$PY -m unittest discover -s Module/tests -v

# I01 Bunny 独立回归、正式物理 benchmark 与同 trace 可视化
$PY -m unittest Module.tests.test_i01_bunny_physics -v
$PY -m Module.i01_bunny_physics.benchmark
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/handcomp-i01-cache \
  $PY -m Module.i01_bunny_physics.visual_demo --reuse

# M06–M12 独立回归、冻结 benchmark 与同数据可视化
$PY -m unittest Module.tests.test_m06_m12_planner -v
$PY -m Module.m06_m12_benchmark
$PY -m Module.m06_m12_visual_demo --reuse-benchmark

# 重新训练 I20；CUDA-only（路径保留历史 d20 名称）
$PY -m Module.module_4_finger_dp.formal_train \
  --train Module/generated/finger_dp_formal_v1/scaling/dataset_i_d20_train.npz \
  --validation Module/generated/finger_dp_formal_v1/formal_pool_v1/dataset_i_validation.npz \
  --output Module/generated/finger_dp_formal_v1/training_d20_reproduction \
  --updates 10000 --device cuda:0

# 正式 paired E05
$PY -m Module.module_4_finger_dp.e05_dp_benchmark \
  --checkpoint Module/generated/finger_dp_formal_v1/training_d20/formal_finger_dp_checkpoint.pt \
  --output Module/generated/e05_h_mcc_vs_dp

# 只重建视频/dashboard
MUJOCO_GL=osmesa $PY -m Module.module_4_finger_dp.e05_dp_visual \
  --output Module/generated/e05_h_mcc_vs_dp

# 冻结 trace 的 post-hoc failure diagnostic
$PY -m Module.module_4_finger_dp.e05_failure_diagnostics
```

## 8. 当前边界

- gravity-off MuJoCo 不是硬件或 sim-to-real 结论；
- 旧 Exp. 1 H-MCC baseline 有 transient over-force；旧 G1a 只对应 pre-retune profile；
- Dataset-I long/speed matched raw generation 尚未通过；
- I100 没有稳定提升 closed-loop，不继续盲目扩量；
- `G1a=ARCHIVED_PRE_RETUNE`；M06–M12 与 Bunny I01 已完成，且 I01 在冻结范围内给出
  `G2=GO`；当前 `G3=NO_GO`，所以 GPIS 尚未解锁。E05 策略性能不承担解锁职责。
