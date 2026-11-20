# LEAP Hand 触觉探索实验

实验以灵巧手触觉实验档案为准。站点收录五种物体的 Finger Policy 闭环结果、三种 DP 策略消融、腕部贴合实验 `i01-a` 和 `i01-b`、历史 I04-A/B 回放，以及 Active 实验计划。[灵巧手触觉实验档案](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site)

- `i01-a` 待验证，评估独立生成腕轨时 Finger Policy 能否维持指尖贴合。[i01-a 计划](Module/MASTER_PLAN.md#i01-a-独立腕部规划下的贴合)
- `i01-b` 已运行，使用腕指联合规划得到的腕轨。完整任务和锁定同一腕轨各做两次配对，另有 100 Hz 节拍试验。原始文件使用 `I04-c` 名称。[计时、重建与复现说明](Module/I04_C_END_TO_END.md)
- Finger Policy 提供训练数据、三组策略权重、闭环评估和回放入口。[Finger Policy 使用说明](finger_policy/README.md)
- I04-A/B 保留历史原生轨迹、回放、审核和模型。[历史 I04-A/B](Module/I04_AB_RETAINED.md)
- Active 计划包含腕部 Active＋逐指 DP、Aude＋MCC、腕部 Active＋MCC 和补充消融。[当前实验计划](Module/MASTER_PLAN.md)

## 复现入口

`Module` 记录实验协议和仿真代码。安装、模型资产和原始数据获取方式见各实验的说明。

```powershell
.\scripts\run_i04_c.ps1 benchmark --help
.\scripts\run_i04_ab.ps1 -Python python verify --code-only
python finger_policy/pipeline.py --help
```

完整结果、原始文件和依赖索引见 [Module](Module/README.md)。
