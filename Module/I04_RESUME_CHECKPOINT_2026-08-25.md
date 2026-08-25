# I04 断电续作检查点（2026-08-25）

## 1. 冻结任务定义

I04 是 **Oracle Next-Point Whole-Hand Contact Traversal**，不是 GPIS active
exploration。Oracle 只提供 Bunny 表面目标，不提供 target finger；被测方法必须从
当前 MuJoCo 真实接触状态决定 finger assignment、surface micro-bridge、MAKE/SLIDE、
handover 和 workspace management。

当前阶段只运行 **Explicit MCC baseline**：

- MCC baseline：ON；
- M01--M12：ON；
- DPRef：OFF（尚未训练好）；
- GPIS / uncertainty / information gain / next-best-touch：OFF；
- 执行权：只有 M10 certificate 可以经 M06 执行；prediction suffix 无执行权；
- 接触拓扑：任何非空 fingertip contact set 合法；
- MAKE-before-BREAK：MAKE 必须保持到 fresh confirmed barrier，瞬时碰撞不算完成。

最终目标仍是完整运行固定 Bunny required set（274 个 25 mm coverage goals），保存
长轨迹并生成视频或一键 replay。当前尚未完成 274 点，不能宣称 full traversal。

## 2. 已实现内容

### Bunny、路线与真实状态

- I04 使用 MuJoCo 3.6 的非凸 mesh SDF 作为真实物理碰撞；I01--I03 的旧 hfield
  路径未被改写。
- Bunny mesh：34,834 vertices、69,451 faces、面积约 0.212093 m²。
- 用 mesh-edge geodesic 构建 274 个固定 required goals，实际最大覆盖半径
  0.0249926 m。
- Oracle goal 不含 finger ID；每个 fresh barrier 都从各根真实接触指自己的 mesh
  root 重新计算 geodesic micro-bridge。
- arrival 使用持续确认的真实 fingertip contact、mesh geodesic tolerance 12 mm 和
  normal tolerance 55°。
- 困难 goal 可以暂缓，但不再从 required ledger 永久删除。

### M01--M12 与物理执行

- M01/M02/M03：FR3+LEAP、4 路 fingertip MCC、force safety、command continuity；
- M06：transactional prefix、fresh measured barrier、timeout SAFE_HOLD；
- M07/M08/M09：contact-mode graph、CheapCert、continuous candidate optimization；
- M10：MuJoCo exact FK、joint/self/link/SDF swept audit；
- M11/M12：lazy beam search 与 shadow successor viability；
- WRIST_ADJUST、SLIDE、MAKE、REPOSITION 都走 M07--M12 和 M10/M06 权限链；
- `mj_step` 后补 `mj_forward`，保证 q/site/contact 来自同一个物理时刻；
- 手指事务锁定证书中的 target normal，避免在线 IK 使用另一侧/另一曲率法向；
- contact-changing MAKE 必须持续通过 fresh barrier；新接触成功后预留 preload
  settling；
- 低冗余根状态要求确认时长和最小锚点力，失败手指有 8 s cooldown；
- 真实连续接触指标仍按 MuJoCo contact 计算，不用 confirmation 掩盖 gap。

### 证据、回放与测试

- benchmark 保存 canonical Bunny、`trace.npz`、`events.json`、`summary.json`；
- visual demo 可生成 MP4、关键帧、dashboard 和 HTML；
- 当前发布相关测试：28/28 通过；
- 已验证旧的短 smoke 能完成 3/3 development goals，并生成过视频；这不是完整
  Bunny 结果。

## 3. 历史问题与处理

1. **早期几何只是 Bunny hfield，不是完整非凸物体。**
   I04 已改为 MuJoCo mesh SDF，耳朵两侧等薄结构由真实非凸 narrow phase 区分。
2. **所有手指共享同一个 geodesic root。**
   已改为每根真实接触指从自己的当前 contact vertex 规划，且每个 barrier 重算。
3. **`mj_step` 后 site/contact 与新 qpos 不原子。**
   已在暴露 observation 前调用 `mj_forward`。
4. **在线 IK 法向和 M10 prefix target normal 不一致。**
   已让参与手指执行证书 metadata 中的 target normal。
5. **弱锚点仍能触发新事务。**
   曾在 0.35 N 的第二锚点上发起 MAKE，随后退化；现增加确认时间、0.75 N 根状态
   阈值和 2.5 N recovery preload。
6. **瞬时 MAKE 被当作成功。**
   旧 M06 在一次碰撞后把 participant 标成 DONE，即使 fresh barrier 已脱离。现增加
   barrier topology revalidation，并加入对应单元测试。
7. **MAKE 停在 SDF 外约 1--2 mm。**
   free MAKE preload 从 2 mm 加深到 4 mm；progress-only completion tolerance 调整为
   2.25 mm，但 contact-changing MAKE 仍必须有真实持续接触。
8. **失败手指 1/3 来回重试。**
   现保存 per-finger cooldown，使其他自由指也能获得恢复机会。
9. **困难 goal 被 failure score 永久排除。**
   已移除永久 `None`，改为 capped penalty；required ledger 不删除 goal。
10. **当前主要未解决问题：双锚点后的 workspace management。**
    单锚点 `{2}` 已多次真实恢复到 `{2,4}`，但自由指可能仍不可达，M11 随后没有
    存活 finger edge，需要可执行的 two-anchor WRIST_ADJUST。

## 4. 三轮已保存物理回归

所有原先在 `/tmp` 的关键证据已复制到：

`Module/generated/i04_oracle_next_point/checkpoints/2026-08-25/`

| 回归 | stop | goals | continuity | max gap | peak force | 主要结论 |
|---|---:|---:|---:|---:|---:|---|
| regression_1 | LAST_CONTACT_LOST @ 42.114 s | 7/274 | 99.9591% | 12 ms | 6.862 N | target normal/root gating 有改善，但仍失去最后接触 |
| regression_2 | LAST_CONTACT_LOST @ 86.616 s | 5/274 | 99.9833% | 12 ms | 6.992 N | persistent MAKE 正确，不再把瞬时碰撞算成功；长期陷入 singleton MAKE |
| regression_3 | MAXIMUM_DURATION_REACHED @ 90 s | 6/274 | 99.9839% | 4 ms | 5.448 N | 不再提前崩溃，真实恢复出大量双指 barrier，但覆盖停滞 |

regression_3 的补充指标：

- covered area fraction：2.4881%；
- force RMSE：0.4655 N；
- 104 个 certified prefixes；
- 14 次 SAFE_HOLD，25 次 planning rejection；
- planner latency：mean 0.4965 s、p95 0.5319 s、max 1.5261 s；
- fresh barriers 中 `{2,4}` 出现 42 次，证明 singleton-to-pair 恢复是真实发生的。

## 5. 当前源码位置与“尚未物理验证”的修改

主要文件：

- `Module/i04_oracle_next_point/surface_graph.py`
- `Module/i04_oracle_next_point/planner.py`
- `Module/i04_oracle_next_point/runner.py`
- `Module/i04_oracle_next_point/benchmark.py`
- `Module/i04_oracle_next_point/visual_demo.py`
- `Module/module_6_prefix_executor/executor.py`
- `Module/fr3_leap/model.py`
- `Module/tests/test_i04_oracle_next_point.py`
- `Module/tests/test_m06_m12_planner.py`
- `Module/tests/test_fr3_leap_model.py`

regression_3 之后又做了以下源码修改，**23 项测试通过，但尚未跑新的物理回归**：

- 允许 two-confirmed-contact 状态请求 WRIST_ADJUST；
- finger-edge rejection 后下一次优先请求 WRIST recovery；
- wrist completion tolerance 调到 1.0 mm；
- contact-changing MAKE 后加入 settling；
- M11 WRIST rejection 输出 enumerated/cheap/optimized/retained 计数。

对 regression_3 的 88.682 s 保存状态做了离线重放：

- root contact set：`{2,4}`；
- free finger 1 nonlinear residual：约 17.18 mm（不可直接 MAKE）；
- free finger 3 nonlinear residual：约 11.98 mm（不可直接 MAKE）；
- active fingers 2/4 的 surface bridge 候选可行；
- WRIST search：`enumerated=9, cheap=1, optimized=0, retained=[0]`。

因此当前最近的阻塞点是：WRIST candidate 通过了 CheapCert，但 M09 optimization 没有
产生 feasible prefix。尚未确定具体 reason（优先检查 `TARGET_ERROR`、`IK_RESIDUAL`、
`ANCHOR_PRESERVATION`、`REACH`），不能再把它笼统归因于 Jacobian 或 MCC。

## 6. 明天恢复后的顺序

1. 先确认工作区和测试，不启动 formal run：

   ```bash
   cd /home/ferry/data/Code2/Research/hand_comliance_control
   git status --short
   /home/ferry/data/Anaconda/envs/handcomp/bin/python -m unittest -q \
     Module.tests.test_m06_m12_planner \
     Module.tests.test_i04_oracle_next_point \
     Module.tests.test_fr3_leap_model
   ```

2. 用 regression_3 的 88.682 s 状态直接打印 M09 WRIST optimization reasons；不要先跑
   90 s 仿真。
3. 只根据该 reason 修 two-anchor WRIST candidate，并先通过 saved-state M10 audit。
4. 跑 30--60 s 回归，门槛是：`{2,4}` 后出现 certified WRIST 或第三根持续接触，且
   coverage 继续增加。
5. 再跑 180--300 s endurance；确认不再 recovery loop 后才运行 274-goal formal。
6. 如果 MCC baseline 在公平 guard 下仍不能完整覆盖，正式报告它停止的位置和连续
   指标，不把未完成序列写成成功。
7. 最终轨迹稳定后再更新 `I04_ORACLE_NEXT_POINT_PROTOCOL.md`、`Module/README.md`、
   `MASTER_PLAN.md`，生成 MP4/HTML 和一键 replay 命令。

## 7. 当前可用命令

正式入口（目前不要在尚未解决 WRIST blocker 时启动）：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.i04_oracle_next_point.benchmark --profile formal
```

使用 canonical generated trace 生成回放：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.i04_oracle_next_point.visual_demo --reuse --speed 12
```

注意：canonical generated trace 不是 regression_3；三轮断电 checkpoint 使用各自目录
中的 `trace.npz/events.json/summary.json`。后续应给 visual demo 增加显式 `--input`
参数，避免通过复制覆盖 canonical trace。

## 8. 停止状态

- 没有正在运行的 benchmark、renderer 或后台训练；
- 三轮关键 `/tmp` 证据已持久化；
- 当前源码可 import，发布前相关测试 28/28 通过；
- 工作区原本已有大量用户/历史未提交改动，恢复时不得 reset 或清理这些文件；
- I04 当前状态是 **paused / incomplete**，不是 full Bunny completed。
