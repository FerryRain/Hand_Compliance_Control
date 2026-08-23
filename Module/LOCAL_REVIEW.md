# 当前 FR3+LEAP / MCC / DP-v1 审阅入口

本页是统一审阅索引。MCC 是已执行结果；DP 仍未训练/评测。inverse-data 部分提供当前真实
forward→spatial-inverse→physical-replay Dataset-D 数据链审计。废弃 synthetic replay 的
源码与生成数据已从当前 Module 清理，不再提供复现入口。

M4-DP 后续应先做什么、哪些数据能训练以及何时允许扩量，统一以
[`M4_DP_GUIDE.md`](M4_DP_GUIDE.md) 为准。

## 1. FR3–LEAP 结构修正

- 自然接触姿态：
  [`natural_pose_audit.png`](generated/local_review/natural_pose_audit.png)
- 中央掌心安装：
  [`mount_center_audit.png`](generated/local_review/mount_center_audit.png)

当前固定值在 `fr3_leap/model.py`：

- hand reference 是已发布视频 `t=2.000 s` 的精确 16-D `q_hand`；
- palm-mesh XY mount 分数约为 `[0.506, 0.541]`；
- palm/interface site world alignment error 约 `2.6e-10 m`；
- adapter 与 palm mesh 接口距离在 `1 mm` 容差内；
- `palm_lower` 是 `fr3v2_link8` 的直接固定 child。

## 2. 更新后的 MCC 验证

- MCC-only 页面：
  [`mcc_review.html`](generated/local_review/mcc_review.html)
- E05-F-MCC：
  [`mcc_f_video.mp4`](generated/local_review/mcc_f_video.mp4)
- E05-H-MCC：
  [`mcc_h_video.mp4`](generated/local_review/mcc_h_video.mp4)
- 指标 dashboard：
  [`mcc_dashboard.png`](generated/local_review/mcc_dashboard.png)
- 正式 summary：
  [`summary.json`](generated/e05_mcc_current/summary.json)
- 六个 episode：
  [`episodes.csv`](generated/e05_mcc_current/episodes.csv)
- 精确运行 MJCF：
  [`generated_fr3_leap.xml`](generated/e05_mcc_current/generated_fr3_leap.xml)

F/H 均完整运行 `3 × 15 s`。执行状态与性能 verdict 分离：完整运行是
`EVALUATED`，超过冻结阈值则是 `NOT_MET`，不能写成 evaluator `FAILED`。

## 3. M0–M3 总可视化

打开 [`generated/visual_demo/index.html`](generated/visual_demo/index.html)，可同时查看：

- M01 surface/normal/clearance；
- M02 static、sliding、curved-surface MCC；
- M03 blockage/over-force/joint-limit/self-collision guards；
- 当前 central mount、自然姿态与两段 MCC 视频。

## 4. 新真实 spatial-inverse 数据链（不是 DP 评测）

- 左右同步物理视频：
  [`forward_spatial_inverse_replay.mp4`](generated/visual_demo/spatial_inverse_v1/forward_spatial_inverse_replay.mp4)
- fresh force/contact、FR3 tracking 与空间运动 dashboard：
  [`forward_replay_audit.png`](generated/visual_demo/spatial_inverse_v1/forward_replay_audit.png)
- forward/replay 各 1500 个 500 Hz samples：
  [`forward_replay_pair.h5`](generated/visual_demo/spatial_inverse_v1/forward_replay_pair.h5)
- machine-readable provenance/gate：
  [`summary.json`](generated/visual_demo/spatial_inverse_v1/summary.json)
- 目录边界说明：
  [`README.md`](generated/visual_demo/spatial_inverse_v1/README.md)

当前 raw replay gate：`accepted=true`。3 秒内 forward/replay any-contact continuity 都是
`100%`，zero-contact gap `0 s`，平均 contact count 为 `3.719/3.142`；但逐指 contact-mask
entry agreement 只有 `84.83%`，forward contact retention 为 `84.08%`，所以不能把它描述成
逐指完全等价。forward/replay 峰值分别约 `2.693/3.859 N`，non-tip contact frames 为 `0`，
FR3 palm tracking RMSE/maximum 约 `0.306/0.543 mm`。SE(3) residual 约 `1.12e-16`，并且：

```text
q_cmd_replay[t] == q_cmd_forward[t]
maximum mapping residual = 0 rad
replay finger repair = NONE
```

视频左侧是 moving-object 的真实 physics collection，右侧是 fixed-object FR3+LEAP replay；
两侧 force/contact 均来自各自物理测量，不复制 forward label。

但它严格标记为 `Dataset-D diagnostic`、`formal_dataset_i_ready=false`、
`training_allowed=false`，因为 forward collector 当前使用 Fingertip MCC。它证明 inverse
pipeline 已按定义实现，不能冒充正式 non-MCC Dataset-I。

## 5. 复现

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python

# M0–M3 与控制单元回归
$PY -m unittest discover -s Module/tests -v

# 完整重跑 MCC
$PY -m Module.module_4_whole_hand_mcc.demo

# 从正式 trace 重建视频与 gallery
MUJOCO_GL=osmesa $PY -m Module.module_4_whole_hand_mcc.visual_demo
$PY -m Module.visual_demo

# DP v1 core tests + 当前真实空间反演最小闭环
$PY -m unittest Module.tests.test_finger_dp_core Module.tests.test_finger_dp_data -v
MUJOCO_GL=osmesa $PY -m Module.module_4_finger_dp.spatial_inverse_demo --require-accepted
```

最后一个命令当前应返回 `0`，表示 Dataset-D raw replay mechanics 通过。必须同时检查
`summary.json` 中 `training_allowed=false`；正式 Dataset-I 门禁仍保持 fail-closed。

## 6. 边界

- 当前结果是 gravity-off MuJoCo 控制隔离实验，不是硬件结果；
- MCC trace 使用 shadow guards，M06 transactional executor 尚未实现；
- DP v1 core 已实现并单测；Dataset-D spatial replay mechanics 已通过，Dataset-I 尚未生成；
  没有训练 checkpoint，也没有 `E05-H-DP` 指标。
