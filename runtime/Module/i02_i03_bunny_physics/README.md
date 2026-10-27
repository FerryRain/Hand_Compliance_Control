# I02/I03 Bunny Physics

固定 Bunny 上的 Geometry-Oracle + Explicit MCC 物理集成；DP、DPRef、GPIS 关闭。冻结协议见
[`../I02_I03_BUNNY_PROTOCOL.md`](../I02_I03_BUNNY_PROTOCOL.md)。

## 评测问题

- **I02**：相同 `BREAK(3) -> +X REPOSITION(3) 12 mm -> MAKE(3)` 中，一次 12 mm
  committed prefix 与 `3 x 4 mm`、每段从 fresh measured barrier 重规划相比，是否显著降低
  physical terminal prediction error。
- **I03**：相同 Beam/candidates/score/M09/M10/M06 下，M12 ShadowSucc terminal predicate
  是否能过滤当前合法但没有安全 continuation 的终点。M12 只筛选，不签证书、不执行。

执行权限始终是：

```text
measured root -> M09 -> M10 certificate -> M06 edge 0 -> measured barrier
prediction suffix / M11 / M12 --------------------------------> no command authority
```

`core.py` 提供 Bunny pad-center Oracle、M09/M10、I03 frozen candidate/search；`runner.py`
提供 20 s MuJoCo 物理状态机和 exact-mesh evaluator；`benchmark.py` 运行 12 个正式 episodes；
`visual_demo.py` 只回放已保存的正式 trace。

## 正式结果

| cell | task pass | mechanism | supported traversal median | worst peak | 关键结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| I02-LONG | 3/3 | 3/3 | `101.767 mm` | `5.637 N` | terminal error `1.476 mm` |
| I02-SHORT | 3/3 | 3/3 | `101.808 mm` | `4.825 N` | terminal error `1.467 mm`；每条 3 cert/3 barrier |
| I03-BEAM | 0/3 | 0/3 | `7.111 mm` | `7.422 N` | 3/3 actual `NONVIABLE` dead end |
| I03-SHADOW | 3/3 | 3/3 | `101.125 mm` | `7.010 N` | 0 dead end；actual margin `0.0478–0.0480 rad` |

- I02 冻结改善阈值为 `SHORT <= 0.8 * LONG + 0.25 mm = 1.431 mm`；实际 `1.467 mm`，所以
  `EVALUATED / NOT_MET`。两组都完成不等于证明 SHORT 更稳。
- I03 dead end `3 -> 0`，支持距离中位优势 `94.014 mm`，所以 `EVALUATED / MET`。
- G3 要求 I02/I03 同时 MET，故 `G3=NO_GO`。

## 复现

从仓库根目录运行：

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python

# 快速语义与路径测试
$PY -m unittest Module.tests.test_i02_i03_bunny_physics -v

# 正式 4 cells x 3 seeds；约 3 分钟（本机）
$PY -m Module.i02_i03_bunny_physics.benchmark

# 从相同 summary/NPZ 生成 dashboard、HTML 和两段并排视频
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/handcomp-i02-i03-cache \
  $PY -m Module.i02_i03_bunny_physics.visual_demo --reuse
```

默认输出目录是 `Module/generated/i02_i03_bunny_physics/`：

- `summary.json`：冻结 acceptance、聚合指标、source hashes、环境版本；
- `episodes.csv`：12 个 episode 的平面表；
- `trace_<cell>_seed_<seed>.npz`：每 tick 的 q/dq/command/contact/force/phase/certificate；
- `traces.npz`：合并 trace；
- `index.html`、dashboard、keyframes、两段 MP4：审阅可视化；
- `generated_fr3_leap_bunny.xml` 与 canonical Bunny visual mesh：场景 provenance。

这是固定物体、gravity-off control-isolation 结果，不能外推到未知物体、完整非凸 mesh
collision、gravity-on、硬件或 sim-to-real。
