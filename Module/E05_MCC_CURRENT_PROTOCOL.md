# E05 MCC 当前冻结协议：自然接触姿态与单指异质粗糙面

> 状态：`FROZEN_AND_EVALUATED`  
> 冻结日期：`2026-08-23`  
> 环境：`handcomp`  
> 范围：本协议只授权 MCC；DP 不读取、不实现、不评测。

## 1. 本轮只改变什么

1. Leap Hand 的姿态参考改为已发布 raw-DP 视频 `t=2.000 s` 的真实 `q_hand`，不是
   checkpoint mean，也不是对截图的人工估计；
2. FR3 法兰轴连接 Leap palm 中心区域，不再连接 `palm_lower` 的偏置 body origin；
3. 固定 world height field 增加指间异质性，让相邻手指在不同时间遇到凸包、凹槽和短波；
4. 在上述模型上重新评测 `E05-F-MCC` 和 `E05-H-MCC`。

DP 不参与这次 MCC 运行。未来 DP 必须使用独立冻结协议和独立结果，不能反写本协议的
MCC 数字。

## 2. 冻结自然姿态

参考来源：commit `c5090d6703fd5f502b7411978bd9935ee425b0ed`、checkpoint SHA-256
`89044a1045ae44e28bec129c71998d3b389f08e4349e8a18441ba10bdd073ef0` 的确定性
compatibility run，在 `t=2.000 s` 读取物理状态：

```text
[ 0.21093412,  0.00522920,  0.40359447,  0.52873424,
  0.20371661,  0.00995645,  0.40966995,  0.53162878,
  0.20904370,  0.01274314,  0.40656648,  0.52803646,
 -0.05238612,  1.56150337,  0.38962566,  0.51837131 ] rad
```

这组值是 MCC finger IK 的 posture null-space reference。实际 joint command 仍由解析 MCC、
surface target 与物理反馈产生。

## 3. 冻结中央掌心安装

- palm mesh 局部 XY 范围中的 mount 分数约为 `[0.506, 0.541]`，必须位于中心区
  `[0.38, 0.62]^2`；
- palm mount 点：`[-0.048, -0.032, 0.0112776] m`，Z 位于法兰侧 palm mesh 外表面；
- link8 interface 点：`[0, 0, 0.0112] m`；
- 两个 site 的 world alignment error `<= 1e-9 m`；
- adapter–palm mesh signed distance 绝对值 `<= 1 mm`；
- `palm_lower` 仍是 `fr3v2_link8` 的直接固定 child；
- 显式 adapter 可见但无质量、不参与碰撞。

body-origin gap 和任意两块 mesh 的全局最短距离不再作为安装正确性的替代指标。

## 4. 冻结表面与任务

- MuJoCo `dt=0.002 s`、`implicitfast`、80 iterations、gravity off；
- hfield 约 `2 mm` 网格；试验性 `1 mm` 网格没有改善峰值/接触，未保留；
- hand position actuator `kp=22`、damping ratio `1.5`；
- pad/object contact `solref=0.028 1`；这由 MCC safety precheck 冻结，替代产生
  单帧 `64.75 N` 数值冲击的 `0.020 1`；
- 固定 world height field：`0.60 x 0.84 m`；
- dense profile 高度范围约 `36.27 mm`；
- 最大主曲率约 `95.84 1/m`，最小曲率半径约 `10.43 mm`；
- broad/diagonal waves、短波 cross term、窄 ridge，以及错开的单指 Gaussian bump/pit；
- 每个 episode `15.0 s`，前 `1.0 s` 建立接触；
- palm Y 净位移 `180 mm`，X 叠加 `18 mm + 7 mm` 两频 S path；
- `t=9.0 s` palm 远离表面 `4 mm`，测量失触与恢复；
- 四指目标力 `2 N`，硬评测上限 `8 N`。

## 5. 两个 MCC 单元

| 单元 | wrist | fingers |
| --- | --- | --- |
| `E05-F-MCC` | prescribed FR3 palm pose，Wrist MCC off | 四个完整 local force error Fingertip MCC |
| `E05-H-MCC` | 同一 nominal path + resultant Wrist MCC | active fingers 只调 internal/differential error；未接触指负责 MAKE |

两者使用同一模型、表面、姿态、trajectory、seed、friction/noise 配对和 evaluator。

## 6. 正式 episodes 与阈值

episodes 保持：`nominal(seed=7)`、`low_friction(seed=11, mu=0.75, force noise=0.03N,
q noise=0.004rad)`、`noisy_pose(seed=19, mu=1.05, force noise=0.05N,
q noise=0.006rad)`。

阈值保持与上一轮一致以便纵向比较：contact continuity `>=0.995`、average contacts
`>=3`、zero-contact `<=0.05s`、force RMSE `<=1N`、force violation probability
`<=0.001`、peak force `<=8N`、four-contact recovery `<=0.25s`、force settling
`<=0.75s`、Y traversal `>=0.16m`、palm tracking RMSE `<=8mm`、joint margins
`>=0.03rad`、controller p95 `<=2ms`、deadline miss `<=0.01`、non-tip contacts `=0`。
H 另要求 wrist Fz RMSE `<=2.5N`、wrist compliance translation `<=12.1mm`、
internal leakage p95 `<=0.05N`。

完整运行但超过阈值写 `EVALUATED / NOT_MET`，不能写成执行失败。

## 7. 冻结产物

```text
Module/generated/e05_mcc_current/
  summary.json
  episodes.csv
  base_traces.npz
  model_audit.json
  generated_fr3_leap.xml
  README.md

Module/generated/local_review/
  mcc_f_video.mp4
  mcc_h_video.mp4
  mcc_dashboard.png
  mount_center_audit.png
  natural_pose_audit.png
```

正式运行后不得修改本文件；代码/协议 SHA、全部 episode 原始指标和 trace 必须保留。

## 8. Safety precheck（不作为最终分数）

自然姿态/异质表面的首次全运行发现 H/noisy-pose 在 `t=1.884 s` 出现一个 `2 ms` 的
`64.75 N` 接触流形冲击，前后帧约为 `2.2 N` 与 `3.1 N`。因此先完成接触参数选择，再执行
正式 MCC episodes。
控制/离散组合试验显示：

- 过软 `kp=14, solref=0.035` 可把峰值降到 `3.44 N`，但 contact continuity 仅
  `89.26%`；
- `kp=18, solref=0.028` 的 continuity 为 `98.90%`、峰值 `11.69 N`；
- `kp=22, solref=0.028` 的 continuity 为 `99.50%`、峰值 `9.82 N`，保留；
- 继续到 `solref=0.030` 或 1 mm hfield 均没有继续改善。

这些是 controller-selection evidence，不计入最终三 episode aggregate。完整 tuning 数字写入
`Module/generated/local_review/mcc_tuning.json`，首次 aggregate 保留为
`mcc_precheck_summary.json`。
