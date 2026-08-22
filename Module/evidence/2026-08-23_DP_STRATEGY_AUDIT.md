# 8 月 17 日 DP strategy 完整性与 E05 兼容性审计

> 日期：2026-08-23
>
> 环境：`handcomp` Python
>
> 正式 E05-F-DP：`NOT_EVALUATED`
> 兼容性试跑：`EVIDENCE_ONLY`

## 1. 审计对象

关键历史提交：

| commit | 日期 | 内容 |
| --- | --- | --- |
| `29fddafe` | 2026-08-15 | 恢复 DP/MCC 数据、训练、部署流水线 |
| `80edf001` | 2026-08-17 | active capsule DP-MCC deployment 与 release guide |
| `ce441f3` / `fa35b32` / `2742f399` | 2026-08-17 | 只调整 demo camera；策略契约不变 |

tag/release：`dp-capsule-v1`，名称为 `Capsule fingertip DP + FullHandMCC v1`。

Git tree 只提交代码、配置、文档和 demo 视频；大文件按 README 约定放在 GitHub Release，
不是缺失提交。远端资产清单：

| asset | size | SHA-256 |
| --- | ---: | --- |
| `best.pt` | 910,376,605 B | `89044a1045ae44e28bec129c71998d3b389f08e4349e8a18441ba10bdd073ef0` |
| `dataset_info.json` | 955 B | `89bd66c67074cd0e5bad8ba6755b7086192a266c8842383db01121cbdbe224c5` |
| `metrics.json` | 10,619 B | `659bc982674567a1be79249fe9601f77c7043ad9df2b18b28a3746951dc1f62c` |
| `capsule_dp_mcc_v1.yaml` | 1,547 B | `edba8618f8da91cdd00b322f4c037ae204149665580299a66fc7f93ea40b9033` |
| teacher replay H5 | 682,042,016 B | `0330b3e85363d79a4da7ce5111022a54716384fb8708ae9e38b5df3cab13fc38` |
| DP training H5 | 127,375,376 B | `91795e65ad8d8fa982391ed634452d36d00e6c97e8b92b4743df3916886bc9d4` |

结论：**原 capsule release 的代码与发布资产完整。**

## 2. checkpoint 实际加载

不是只看文件名或 demo 视频。`best.pt` 下载到临时目录后完成了实际加载：

```text
LeRobot                      0.4.4
architecture                 lerobot_diffusion_conditional_unet1d
step                         25000
model tensors                148
parameters                   75,851,152
state_dict missing           []
state_dict unexpected        []
input                        56-D palm-frame history x 16
output                       32 x 16 absolute q
10-step CPU inference        finite output
```

checkpoint 自带 `56-D` normalization statistics；无需另找 scaler。发布指标的
`val_sample_mae_rad` 在 step 25,000 为约 `0.001715 rad`，但这是发布数据分布上的 teacher
prediction metric，不是当前 E05 contact-control 指标。

`handcomp` 原先没有 LeRobot/diffusers。本次只把 LeRobot 0.4.4 和推理依赖放到
`/home/ferry/data/tmp/`；没有修改 Conda 环境。为绕开 LeRobot 0.4.4 对无关 VLA、视频和
hardware policy 的 eager import，临时目录使用了 audit-only lazy-import shim；
`diffusion/configuration_diffusion.py`、`modeling_diffusion.py` 和 checkpoint state dict 未改。

## 3. 为什么不能直接登记为当前 E05-F-DP

发布策略与当前冻结的 standalone Finger DP 边界不一致：

| 项目 | `dp-capsule-v1` | 当前公平 E05-F-DP 所需 |
| --- | --- | --- |
| 角色 | nominal contact-motion generator | finger controller |
| observation | q + contact point/normal/mask + palm twist + future planner delta | 必须重新冻结；不能偷偷增加 oracle/teacher 输入 |
| action | future absolute `q_hand` | 可执行 `q_f_cmd`，含明确 rate/fallback |
| 训练对象 | capsule meridian | 当前 extreme varying-curvature hfield |
| 初始抓持 | capsule grasp distribution | 当前 downward belly-pad grasp |
| 正式 executor | `FullHandMCC` | 若比较 DP vs. MCC，不能把 Fingertip MCC 隐藏在 DP 后处理 |

release config 明确写有 `execution.layer: fullhand_mcc`；其 roadmap 也定义 DP 学 nominal
motion、MCC 负责高频 contact/force。它可以成为未来的 `DP + shared low-level MCC` 方法，
但不能直接回答当前问题“standalone Finger DP 是否优于 analytical Fingertip MCC”。

## 4. raw policy 物理兼容性试跑

仍然做了一个最小试跑，避免只凭文档判断：

- 使用修正后的 FR3 + Leap Hand v2 plant；
- 与 E05-F 相同的 prescribed FR3 wrist、fixed extreme surface 和 belly pads；
- 3 s quick trajectory，前 1 s 为 settling；
- 10-step diffusion，历史 0.75 s，5 Hz replan；
- 使用 release 的 exact historical chunk scheduler；
- **不使用 Fingertip MCC、FullHandMCC 或 force-error 后处理**。

结果：

| 指标 | raw DP trial |
| --- | ---: |
| execution | `EVIDENCE_ONLY` |
| first target vs. current grasp | `1.1562 rad` max delta |
| contact continuity | `6.7%` |
| average contacts | `0.067` |
| zero-contact time | `1.866 s / 2.0 s evaluated` |
| force RMSE | `2.020 N` |
| peak force | `13.156 N` |
| final contacts | `{}` |
| mean / P95 DP inference | `92.4 / 97.4 ms` on CPU（本次运行） |

这证明 raw absolute-q policy 不能直接插入当前 E05 initial grasp。该结果不等于“DP 方法
失败”：它是 out-of-contract / out-of-distribution 兼容性证据，所以不能进入 MCC-vs-DP
正式表格，也不能被标成正式 `FAILED`。

## 5. 复现

下载并校验 checkpoint：

```bash
gh release download dp-capsule-v1 \
  --repo FerryRain/Hand_Compliance_Control \
  --pattern best.pt \
  --dir /home/ferry/data/tmp/dp-capsule-v1-artifacts

sha256sum /home/ferry/data/tmp/dp-capsule-v1-artifacts/best.pt
```

确认 `handcomp` Python 能 import LeRobot 0.4.4 的 diffusion modules 后运行：

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PYTHONPATH=/path/to/lerobot-0.4.4-deps \
  /home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.evidence.run_dp_release_compatibility \
  --checkpoint /home/ferry/data/tmp/dp-capsule-v1-artifacts/best.pt \
  --output /home/ferry/data/tmp/e05_dp_compatibility_trial.json \
  --video /home/ferry/data/tmp/e05_f_dp_raw_compatibility.mp4
```

脚本会强制校验 checkpoint SHA、state fields、schema、action representation 和完整
state dict，并从 commit `2742f399` 只读加载 exact scheduler。当前生成的 3 s H.264 视频为
`960x540`、12 fps、36 frames，SHA-256：

```text
c365d6870ebf632dde5a569ce376a9f98f23c3262aad49f91d08a80f4b1df626
```

## 6. 正式 E05-F-DP 前还缺什么

需要用户先冻结一个方法选择：

1. **standalone Finger DP**：不能使用 analytical Fingertip MCC 后处理；需在当前
   FR3/downward-belly/extreme-surface 分布上训练或 fine-tune，并定义 contact-loss fallback；
2. **DP nominal + shared MCC**：可以直接沿用 release 的方法思想，但实验名必须写成
   `DP+MCC`，baseline 也必须共享同一个低层 MCC，比较的是 nominal generator；
3. 两种都测：分别建 protocol，不能混成一个 DP 数字。

选择冻结前，正式状态保持 `M04-DP BLOCKED / E05-F-DP NOT_EVALUATED`。
