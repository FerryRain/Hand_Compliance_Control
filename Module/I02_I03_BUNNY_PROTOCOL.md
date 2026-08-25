# I02/I03：Bunny 上的短前缀重规划与终端可行性物理协议

冻结日期：2026-08-24。用户在 I01 完成后明确授权：“请继续评测 I02 和 I03”。本协议在任何
I02/I03 实现和正式结果运行前冻结。只使用 Geometry-Oracle + Explicit MCC baseline；DP、
DPRef、GPIS、prediction suffix 和 ShadowSucc 均无执行权限。

## Module ID、状态与范围

- Module IDs：`I02-PHY-BUNNY-v1`、`I03-PHY-BUNNY-v1`
- 协议状态：`PROTOCOL_FROZEN`；评测状态：I02 `EVALUATED / NOT_MET`、I03
  `EVALUATED / MET`、G3 `NO_GO`
- Exact scope：固定 Bunny、FR3+LEAP、MuJoCo gravity-off control-isolation；I02 隔离 committed
  prefix 长度，I03 隔离 terminal ShadowSucc filter。
- I01 的 `G2=GO` 是本协议的输入，不重新比较 fixed/variable feasible set。
- 不在范围：DP/Main planner、GPIS、active exploration、硬件、sim-to-real、gravity-on。
- 本协议本身不改变 G1a；后续已将 E05 策略 G1b 退休，Exp.1/2 改为描述性评测。

## 共享物理场景与 evaluator

- Bunny、姿态、缩放、碰撞和 exact-mesh 接触过滤完全沿用 `I01_BUNNY_PROTOCOL.md`：
  `Module/assets/stanford_bunny.ply`；最长边 `0.300 m`；exact triangle visual/audit mesh；
  `181 x 181` upper-envelope MuJoCo hfield。
- timestep `0.002 s`，gravity `0`，Bunny 固定；seeds `7,11,19`，paired cells。
- 每条 episode `20.0 s`：`0–3 s` contact acquisition；`3–4 s` 沿世界 `-X` 从 `0` 到
  `10 mm`；`4–7 s` 规划/换指 plateau；`7–12 s` 到 `60 mm`；`12–17 s` 返回 `10 mm`；
  `17–20 s` hold。规划累计路径为 `110 mm`，所有 cells 使用同一平滑时间律。
- 目标力 `2.0 N/finger`，contact threshold `0.20 N`，hard limit `8.0 N`。
- 计分接触必须同时满足 MuJoCo fingertip--Bunny force/contact 与 exact-mesh residual
  `<=2.5 mm`；runtime `A_actual` 只来自实测 force/hfield contact。
- 任何 over-force、non-tip Bunny collision、NaN、未授权命令、最后接触丢失 `>10 ms` 或
  executor `SAFE_HOLD` 都记录停止；预测状态永远不能补入 `A_actual`。
- 共同任务通过条件：supported cumulative palm traversal `>=100 mm`、非空 exact-mesh contact
  fraction `>=99.0%`、最大全接触丢失 gap `<=10 ms`、peak valid fingertip force `<=8 N`、
  over-force/non-tip/authority violation 均为 0。

## I02：Receding-Horizon Prefix

### 目的

只回答：在相同 variable-contact handover 和相同终端目标下，短 committed prefix + 真实 barrier
snapshot + replan 是否比一次长 prefix 更能抑制 live nonlinear kinematics 的预测误差。I02
不能被解释为再次扩大 feasible set。

### Paired cells

两组都在 `10 mm` plateau 上执行同一个实测 `4 -> 3 -> 4` handover，参与指固定为 finger 3：

```text
BREAK(3) -> total +X REPOSITION(3) 12 mm -> MAKE(3)
```

- `I02-LONG`：REPOSITION 由一次 `12 mm` M09 prefix 表示；在实测 root 上只优化和 M10 audit
  一次，M06 执行到 micro barrier。
- `I02-SHORT`：同一绝对终点拆为 `3 x 4 mm`；每段结束必须获得 fresh M06 barrier snapshot，
  下一段从该实测 q/wrist/tips/`A_actual` 重新线性化、优化和 M10 audit。
- BREAK、MAKE、最终 contact target、Bunny、路径和 MCC 完全相同；SHORT 不能复用上一段
  prediction suffix，也不能把预测 contact 当作下一段 root。

### I02 指标与判定

报告两组的 execution failure/blocked/collision、最终 12 mm 目标误差、逐 prefix terminal
prediction error、supported cumulative traversal、contact/force、安全事件、certificate/barrier/
replan count、M09/M10/M06 latency。

`I02 performance=MET` 必须全部满足：

1. SHORT 至少 `2/3` episodes 达到共同任务通过条件并完成实测 `4 -> 3 -> 4`；
2. SHORT 每条成功 episode 恰有 3 个 REPOSITION certificate 和 3 个对应 fresh barrier root，
   root authority violation 为 0；
3. SHORT task-pass count 不低于 LONG，且 supported cumulative traversal 中位数不低于 LONG
   超过 `5 mm` 的负差；
4. robustness improvement 至少满足一项：SHORT execution failure count 小于 LONG，或 SHORT
   最终重定位误差中位数 `<= 0.80 * LONG + 0.25 mm`；
5. 所有执行 prefix 都有 authentic M10 certificate，suffix command count 为 0。

## I03：Terminal Viability

### 目的

只回答：在相同 Beam、候选、score、M09、M10、M06 和 MCC 下，M12 ShadowSucc 是否能拒绝
“当前 prefix 合法且可执行、但终点没有安全 continuation”的候选，减少物理 dead end。

### 冻结 decision fixture

- decision root 是 `10 mm` plateau 上的真实四指 barrier state；M11 使用 horizon `1`、beam
  width `8`、per-mode quota `2`。H=1 专门隔离 terminal filter；M11 的 H=2/3 搜索能力已由
  M06–M12 模块协议单独验证。
- 两组共享 live-MuJoCo local kinematics 与 Bunny pad-center Oracle。冻结 SLIDE 候选为：
  `finger1: -3.0 mm X`、`finger2: -2.5 mm X`、`finger3: +4.0 mm X`、
  `finger4: +2.0 mm X`。BREAK 候选只作为未来 handover successor。
- 即时 progress score 使普通 Beam 优先 finger 3；所有被执行的 edge 仍必须通过
  `minimum_joint_margin=0.010 rad` 的 M10 swept audit。
- ShadowSucc 的未来 continuation reserve 冻结为 terminal minimum joint margin
  `>=0.025 rad`，并要求至少一个非平凡 CheapCert survivor。它只返回 viability 与 successor
  fingers，不优化、不审计、不签证书。

### Paired cells

- `I03-BEAM`：M11 不使用 terminal predicate，选择最高即时 score 的 edge；执行后的真实
  barrier state 再由 M12 诊断。若为 `NONVIABLE`，记录 `DEAD_END` 并安全保持。
- `I03-SHADOW`：唯一变化是把 M12 predicate 用于 M11 terminal filtering。M11 仍只把 edge 0
  送 M10；通过 barrier 后必须用真实状态再次得到 `VIABLE`。随后优先对 finger 4 执行完整
  certified `BREAK -> REPOSITION -> MAKE` handover（不可用时选最低 ID 的 legal successor），
  再继续相同 `110 mm` 累计路径。

### I03 指标与判定

报告 dead-end count、predicted/actual viability agreement、successor-finger count、selected
sequence、minimum joint margin、handover success、recovery count、maximum/supported cumulative
traversal、M11/M12/M10/M06 latency、contact/force 与全部 authority evidence。

`I03 performance=MET` 必须全部满足：

1. SHADOW 至少 `2/3` episodes 达到共同任务通过条件；
2. SHADOW actual terminal dead-end count 为 0，至少 `2/3` 完成真实 `4 -> 3 -> 4` handover；
3. SHADOW 相对 BEAM 至少减少 2 个 paired dead ends，且 supported cumulative traversal
   中位数优势 `>=30 mm`；
4. SHADOW 选择终点的 actual minimum joint margin `>=0.025 rad`，并有至少一个 distinct
   successor finger；
5. Shadow execution authority 始终为 false，prediction suffix/unaudited command count 为 0。

## Gate G3 / Geometry-Oracle Baseline

本协议只在以下条件全部满足时把 `G3` 标为 `GO`：

- I02 与 I03 均为 `EVALUATED / MET`；
- I02-SHORT 与 I03-SHADOW 各至少 `2/3` 达到 `>=100 mm` supported cumulative traversal；
- 两个推荐 cell 合计无 over-force、non-tip collision、unexplained joint violation、authority
  violation 或 terminal dead end；
- 每个 topology change 都有 M10 certificate、M06 barrier 和实测 contact confirmation；
- 报告完整 safety、coordination、planning 和 viability 指标以及 timing boundary。

即使 `G3=GO`，它也只表示固定、已知 Bunny 上的 Geometry-Oracle baseline 可以进入下一阶段
协议设计；不等于 GPIS、未知物体、active exploration、hardware 或 Main-vs-Baseline 已通过。

## Schema 与生成物

- trace schema：`i02-i03-bunny-trace.v1`；evaluator：`i02-i03-bunny-evaluator.v1`。
- 输出：`Module/generated/i02_i03_bunny_physics/`，包含每 episode 完整 q/dq/command、真实
  contact/force/mesh residual、planned/actual/cumulative progress、prefix/certificate/barrier、
  search/shadow/audit latency、selected sequence、joint margin、guard 与 authority trace。
- 可视化只读取保存的 summary/traces，不重新定义或计算 acceptance。
- 正式运行 CPU-only MuJoCo，单 episode wall time 上限 5 分钟；超时进入 `SAFE_HOLD`。

## 正式结果登记（不修改上述 acceptance）

2026-08-24 完成 cells `I02-LONG/I02-SHORT/I03-BEAM/I03-SHADOW` × seeds `7/11/19`：

- I02 两组均 3/3 common task pass 与 handover；LONG/SHORT supported traversal 中位数
  `101.767/101.808 mm`，terminal prediction error 中位数 `1.476/1.467 mm`；SHORT 未达到
  冻结的 `1.431 mm` error-improvement threshold，且 failure count 同为 0，故 `I02=NOT_MET`；
- I03 BEAM/SHADOW dead end 为 `3/0`，supported traversal 中位数 `7.111/101.125 mm`，
  SHADOW actual margin `0.0478–0.0480 rad` 且 3/3 handover，故 `I03=MET`；
- 推荐 cells 没有 over-force、non-tip collision、authority violation、Shadow authority 或
  suffix command；但 G3 要求 I02/I03 同时 MET，因此 `G3=NO_GO`。

机器结果与 source hashes：`Module/generated/i02_i03_bunny_physics/summary.json`；证据说明：
`Module/evidence/2026-08-24_I02_I03_BUNNY_PHYSICS.md`。
