# recovery_standalone — DP 失触恢复数据采集包

把「DP 闭环失触 → 专家重新规划 → 采集恢复轨迹 → 拼成训练集」整条链路打包成一份
可独立分发的代码。设计意图与验收标准见 `DAgger_Recovery_Data_Guide.md`。

## 依赖边界

本目录**不依赖 `mcc_finger_compliance_control/` 里的任何文件**（见下方「与主仓库的关系」），
但仍然依赖：

| 依赖 | 说明 |
|---|---|
| conda 环境 `mjlab` | 提供 torch / mujoco / mujoco_warp / warp / h5py / scipy / trimesh / yaml |
| `mjlab` 包（editable，指向主仓库 `src/`） | 上游仿真库，含 `mjlab.tasks.leaphand.*` 手部环境定义；通过 `mjlab.pth` 解析，与 cwd 无关 |
| `assets_external/` | 物体碰撞网格。`object_catalog.py` 按 `Path(__file__).parents[2]` 解析，因此本目录必须放在一个包含 `assets_external/` 的目录下（当前：仓库根）。单个物体约 7 MB |
| `src/mjlab/asset_zoo/robots/xarm6_leap_hand/` | 手部 XML 与网格（15 MB），由 leaphand 环境定义引用 |

### 与主仓库的关系

`src/mjlab/tasks/leaphand/leaphand_mcc_finger_env_cfg.py` 需要 `object_catalog`。该文件带有
回退导入：

```python
try:                      # 本目录自带一份（sys.path 里有 scripts/ 时命中）
    from object_catalog import ...
except ImportError:       # 主仓库自带一份
    from mcc_finger_compliance_control.scripts.object_catalog import ...
```

因此两个树各自解析到自己的副本，互不影响；`object_catalog.py` 本身零改动。

## 目录结构

```
recovery_standalone/
├── scripts/
│   ├── batch_run_recovery_pipeline.py   # 7 段编排入口
│   ├── extract_recovery_state.py        # [1] rollout → 状态 A
│   ├── generate_manifold_palm_plan.py   # [2] state A → palm 流形规划
│   ├── optimize_contact_plan.py         # [3] 接触可行性优化
│   ├── collect_trajectories.py          # [4] 专家采集（GPU）
│   ├── invert_trajectories.py           # [5] 反解
│   ├── export_palm_dp.py                # [6] 导出 DP 数据集
│   ├── build_recovery_dataset.py        # [7] 拼接 + 审计
│   └── object_catalog / surface_mcc_finger / dp_motion_features /
│       palm_planner_features             # 上游依赖模块（随包分发）
├── configs/                             # 物体族/物体 YAML（object_catalog 按 parents[1] 解析）
└── data/trajectories/                   # 采集原始输出（collect 默认写入这里）
```

## 用法

全部脚本按绝对路径调用，**不要求特定 cwd**：

```bash
PY=/home/rimlab/miniconda3/envs/mjlab/bin/python
B=/path/to/recovery_standalone/scripts

# 批量：给定 rollout 目录（epNNN.h5 + epNNN.csv 成对），自动找失触帧并跑完 7 段
$PY $B/batch_run_recovery_pipeline.py \
  --rollout-dir   <含 epNNN.h5/epNNN.csv 的目录> \
  --reference-dp  <干净训练集 h5> \
  --output-dir    <输出目录> \
  --object-id ycb_mustard
```

单段调用见 `DAgger_Recovery_Data_Guide.md` 第 5 节；`--help` 有每个参数的说明。

### 环境变量

- `MCC_TRAJECTORY_DIR` — 覆盖 `collect_trajectories.py` 的原始输出目录，默认
  `recovery_standalone/data/trajectories`。想让采集结果落回主仓库的
  `mcc_finger_compliance_control/data/trajectories` 时设置它（batch 编排会同时读取该
  变量的值来决定去哪里找产物）。

## 已知限制（交付前请知晓）

1. **恢复段缺少「接管→搜索→恢复」过程。** 编排默认
   `--record-start-step 160`，即从注入后第 160 步才开始记录，而 `build_recovery_dataset.py`
   的 `recovery_confirm_frame` 实测为 0——恢复段第一帧就已经是四指稳定接触。结果是数据由
   「失触前 80 帧」+「已恢复稳定 50 帧」拼成，中间的失触与接管过程不入库，与 Guide
   「恢复段包含接管、搜索、恢复和稳定尾段」的要求不符。
2. **`--max-q-jump-rad` 被放宽到 0.13。** 默认 0.05 会拒绝全部恢复片段；实测拼接边界的
   帧间 q 跳变是段内正常值的 8–53 倍，属真实不连续（由限制 1 的 160 步空档造成），
   放宽阈值只是让它通过审计而非修复。
3. **失触检测偏松。** 默认 `--failure-trigger-threshold 3`（Guide 建议「有效接触指
   少于 3 根」），实际会接受「丢一指仍在 3 指振动」的片段；部分片段 80 帧 ctx 里
   近乎全程四指接触，几乎没有可学的失触前兆。
4. **训练窗口稀疏。** 130 帧 / stride 5 / obs 16 / pred 8 ⇒ 每条仅 3 个窗口。
5. **无法复跑历史批次。** `recovery_round1/` 输入与 `_intermediate/` 中间产物已被清理，
   已有的 8 条成品无法回溯重跑。
6. **动作标签用执行值。** `action_field='q_hand'` 是专家**实际执行**的关节位置，不是
   `q_ref` 意图值——与主仓库其它 schema 一致，非恢复数据独有。
