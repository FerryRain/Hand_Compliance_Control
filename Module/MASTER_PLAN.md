# 当前实验计划

以灵巧手触觉实验档案中的实验范围和状态为准。报告、原始文件和复现命令见 [Module](README.md) 和 [实验站点](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site)。

## 第一部分　Policy 控制策略

### 五物体完整 DP 与策略消融

- 完整 DP 使用 `single_all`，评估物体为芥末瓶、留出梨、饼干盒、漂白清洁剂瓶和咖啡罐。保留每个试次实际完成度及提前停止原因。[五物体结果](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#objects)
- 消融比较 `single_all`、`single_common` 和 `flat_common`。站点保留配对条件、逐组指标、等时覆盖和两套运行结果。[策略消融](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#plans)
- 触觉输入消融待做。固定策略结构和腕轨，比较触觉形变与力信号输入下的接触保持情况。

训练、评估和回放入口见 [Finger Policy](../finger_policy/README.md)。

### i01-a 独立腕部规划下的贴合

待验证。腕部规划器只求解腕的位置、朝向和运动时序。固定同一条独立生成的腕轨，比较 Finger Policy＋MCC 和固定名义抓姿＋MCC。记录腕部运动段中至少一指、至少三指、四指同时接触的时间占比，以及最长失接触时长和持续高力。覆盖和 R-GPIS 重建作为补充指标。[站点实验定义](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#i01-a)

### i01-b 联合规划腕轨上的贴合

在 MuJoCo 中用无机械臂的 LEAP Hand 探索固定芥末瓶。两种方法共用底层 MCC。两项已运行的对照均使用腕指联合规划器生成的腕轨。

| 对照 | 方案 | 状态 |
| --- | --- | --- |
| 完整任务 | 腕指联合规划＋MCC 对比现场生成腕轨后运行 Finger Policy＋MCC | 两次配对已运行 |
| 同一腕轨 | 固定腕运动，比较在线指端 IK＋MCC 与 Finger Policy＋MCC | 两次配对已运行 |
| 100 Hz 节拍 | 同一腕轨前 60 秒，测量迟到周期、长指令间隔和最大间隔 | 已运行 |

端到端时间从进程启动计至 R-GPIS 网格写完。10 mm 和 25 mm 覆盖都基于实测指尖头部接触，重建用共用的 R-GPIS 配置和评估范围。站点早先标为 `i01-ab`，本地记录使用 `I04-c`。[报告和复现](I04_C_END_TO_END.md) · [站点配对结果](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#i01-b)

### I04-A/B 历史记录

历史 A 记录腕部参考加手指柔顺。历史 B 记录手掌与四指联合规划。保留两条历史回放及原生审计，以各自原始记录的碰撞模型和覆盖定义解读。[I04 A/B 保留版](I04_AB_RETAINED.md)

## 第二部分　Active 主动探索

| 组别 | 上层 | 底层 | 状态 |
| --- | --- | --- | --- |
| 主方法 | 根据触觉只选择下一段腕部运动 | `single_all`＋MCC | 待做 |
| Aude＋MCC | Aude Billard 团队的触觉探索方法 | MCC | 待做 |
| 腕部 Active＋MCC | 与主方法相同的腕部规划 | MCC | 待做 |

每轮以实测触点更新物体表面。三组从相同物体和初始位姿出发。用相同时间比较重建质量，用相同目标覆盖比较达到目标的实际耗时，同时记录未达标试次。

补充消融待做。固定手指策略和腕部动作空间，比较腕部优化的信息增益。固定纯 MCC，比较两种规划范围的覆盖、重建误差和耗时。[Active 实验定义](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#active)
