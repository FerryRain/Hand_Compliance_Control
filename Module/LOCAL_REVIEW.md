# 当前 MCC-only 审阅入口

本页只列出已验收、准备进入 `main` 的 FR3+LEAP、M0–M3 和 MCC 结果。未通过验收的
DP 数据、训练、checkpoint、视频和指标不在本页，也不进入本次提交。

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

## 4. 复现

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
```

## 5. 边界

- 当前结果是 gravity-off MuJoCo 控制隔离实验，不是硬件结果；
- MCC trace 使用 shadow guards，M06 transactional executor 尚未实现；
- DP 当前状态是 `REWORK_REQUIRED / NOT_EVALUATED`，新协议冻结前没有正式指标。
