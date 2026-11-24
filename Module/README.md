# LEAP Hand 实验与复现

[灵巧手触觉实验档案](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site) 记录当前实验和计划。[主任务记录](MASTER_PLAN.md) 对应站点的各组实验。

| 站点项目 | 复现与证据 |
| --- | --- |
| 五物体完整 DP 与策略消融 | 三个策略权重、腕参考、物体初态、闭环记录及覆盖计算见 [Finger Policy](../finger_policy/README.md)。 |
| `i01-a` 独立腕部规划下的贴合 | 待验证。腕部规划器只输出腕轨与前瞻，以运动时的实测指尖接触、失接触时长和载荷评估。[i01-a 计划](MASTER_PLAN.md#i01-a-独立腕部规划下的贴合)。 |
| `i01-b` 联合规划腕轨上的贴合 | [完整任务、同腕轨、100 Hz 节拍和 R-GPIS 重建](I04_C_END_TO_END.md)。站点早先标为 `i01-ab`，本地原始记录名为 `I04-c`。入口为 `scripts/run_i04_c.ps1 benchmark`。 |
| I04-A/B 历史录像 | [保留版、原生回放和文件校验](I04_AB_RETAINED.md)。入口为 `scripts/run_i04_ab.ps1`。 |
| Active-1 重建与 Active-2 信息获取 | [PalmGuide（Facing-4）的运行与评分入口](palmguide/README.md)，以及[逐指选向、腕指联合规划的 Active-1-B 对照](active_1_b/README.md)。 |
| Active-3 手指执行方式消融 | [五物体的三组配对、接触率、重建效果与复现入口](active_2/README.md)。代码目录沿用归档名。 |
| Active 与触觉输入消融 | 站点列为待做。各组的控制边界和评价条件见 [主任务记录](MASTER_PLAN.md)。 |

## i01-b 数据与复现

- [逐轮结果](../data/i01-b/results.json) · [冻结协议与同腕轨校验值](../data/i01-b/protocol.json)
- [参考腕轨生成与端到端复现](I04_C_END_TO_END.md) · [站点回放与 PLY](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/#i01-b)

完整任务从同一初态执行。DP 组重新生成腕轨，生成时间计入总耗时。同腕轨对照逐步核对发布的腕命令。两组分别用各自的实测指尖接触做 R-GPIS 重建。[端到端评测](I04_C_END_TO_END.md)

## 历史 I04-A/B

历史 A 保留可回放的 14.67% 邻域覆盖轨迹，历史 B 保留掌部修复后的 100% 邻域覆盖轨迹。两条记录使用不同的掌部碰撞条件。完整轨迹、模型、审核及下载说明见 [I04 A/B 保留版](I04_AB_RETAINED.md)。
