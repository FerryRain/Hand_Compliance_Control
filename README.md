# LEAP Hand 触觉探索实验

**PalmReach-6 是本仓库主动触觉探索主实验的主实现。** 手腕在六个切向运动方向和原地保持之间选动作，用共享 GPIS 和 `single_all` 的四指条件可达域估计信息收益与接触支撑。局部 active 调整腕高和掌面朝向，手指策略与 MCC 执行接触适应。[实现与运行命令](Module/palmreach6/README.md) · [正式实验与对照数据](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/df447aa996be0aaa634a3e319f3a39aa98f11045/palmreach-6)

实验以灵巧手触觉实验档案为准。站点收录五种物体的 Finger Policy 闭环结果、三种 DP 策略消融、腕部贴合实验 `i01-a` 和 `i01-b`、历史 I04-A/B 回放，以及 Active 主动探索实验。[灵巧手触觉实验档案](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site)

- `i01-a` 待验证，评估独立生成腕轨时 Finger Policy 能否维持指尖贴合。[i01-a 计划](Module/MASTER_PLAN.md#i01-a-独立腕部规划下的贴合)
- `i01-b` 已运行，使用腕指联合规划得到的腕轨。完整任务和锁定同一腕轨各做两次配对，另有 100 Hz 节拍试验。原始文件使用 `I04-c` 名称。[计时、重建与复现说明](Module/I04_C_END_TO_END.md)
- Finger Policy 提供训练数据、三组策略权重、闭环评估和回放入口。[Finger Policy 使用说明](finger_policy/README.md)
- I04-A/B 保留历史原生轨迹、回放、审核和模型。[历史 I04-A/B](Module/I04_AB_RETAINED.md)
- Active-1 与 Active-2 在相同的五物体起点比较 PalmReach-6、Facing-4 和 Aude，分析覆盖、重建和实触信息收益。240 秒和 600 秒分别独立运行。[三方法对照](https://tactile-hand-experiment-atlas.spunky-brook-2549.chatgpt.site/active-methods.html) · [历史 PalmGuide / Facing-4 发布](Module/palmguide/README.md)
- Active-3 在相同的 Aude 腕指规划下比较规划手指、`single_all` 主动调指与仅底层柔顺，评估五物体重建和多指接触时间。[实验代码与复现](Module/active_2/README.md) · [15 条轨迹、指标和视频](https://huggingface.co/datasets/FerryZh/full-hand-active/tree/main/active-2)
- Active 计划包含腕部 Active＋逐指 DP、Aude＋MCC、腕部 Active＋MCC 和补充消融。[当前实验计划](Module/MASTER_PLAN.md)

## 复现入口

`Module` 记录实验协议和仿真代码。安装、模型资产和原始数据获取方式见各实验的说明。

```powershell
python -m Module.palmreach6 --help
.\scripts\run_i04_c.ps1 benchmark --help
.\scripts\run_i04_ab.ps1 -Python python verify --code-only
python finger_policy/pipeline.py --help
```

完整结果、原始文件和依赖索引见 [Module](Module/README.md)。
