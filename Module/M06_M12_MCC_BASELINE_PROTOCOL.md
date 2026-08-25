# M06–M12 Oracle + MCC Baseline 模块验收协议

冻结日期：2026-08-24。固定环境为 `handcomp`，随机种子为 `7`。本协议只授权
Geometry-Oracle + Explicit MCC Baseline 的模块级实现与验证；它不改变
`G1_NO_GO_E05_PERFORMANCE_NOT_MET`，不接入 Finger DP，也不构成 I01–I05 或 GPIS 的
正式实验结论。

## 权限与状态边界

- `A_actual` 只来自执行时测量；预测 mode、MAKE、suffix 和 ShadowSucc 无权覆盖它；
- M07、M08、M09、M11、M12 都没有执行权限；只有 M10 可以签发
  `ExecutionCertificate`，M06 只接受证书与 prefix digest、root contact set、model version
  全部匹配的提交；
- 每个 committed prefix 最多一次 topology change，wrist transaction 与 finger
  transaction 互斥；
- prediction suffix 只可作为下一次搜索的 shifted warm start；timeout、model-version drift、
  全局 guard 或最后接触丢失均进入 `SAFE_HOLD`，旧 suffix 不恢复；
- normal execution state、每个 participant state、contact state 与 micro-barrier state 分开记录。

## M06：Transactional Prefix Executor

固定 `dt=0.01 s`，completion tolerance `0.75 mm`，micro-barrier 至少等待一个新的真实
observation。以下五项必须全部通过且 authority violation 为 0：

1. 两指异步完成时，先完成者 hold，barrier 只在所有 participant terminal 后关闭；
2. 一个 finger `BLOCKED` 时，其余安全 participant 可以完成，最终 snapshot 保留 blocked evidence；
3. 新 transaction 提交后，旧 certificate/prefix 永不恢复；
4. stale `SurfaceModelVersion` 在执行前拒绝，执行中 drift 触发 `SAFE_HOLD`；
5. timeout 触发 `SAFE_HOLD`，prediction suffix 从未进入 command path。

Anchor finger 使用现有 `FullRobotFingertipMCC` 调整 normal force；BREAK finger 在释放阶段不被
MCC 反向拉回。该模块测试的是 transaction 语义和 MCC 接口，不把简化 plant 当作 E05 物理性能。

## M07：ContactModeGraph

穷举四指的 15 个非空 mode 与全部 primitive，两次枚举结果必须逐项相同，且：不生成 empty
mode、MAKE 只作用于 free finger、BREAK/SLIDE 只作用于 contact finger、REPOSITION 只作用于
free finger、一个 prefix 最多一次 topology change、WRIST 与 finger action 不混合。
`CommitLegal(BREAK)` 必须依赖当前真实 replacement contact 的连续确认。

## M08：CheapCert

使用 seed 7 生成 4096 个候选，以非线性 exact reference 判定可行性。冻结目标：

- false-negative rate `<= 1%`；
- 输出 TP/FP/FN/TN 与 anchor/joint/collision/reach/uncertainty 五类 margin；
- 4096 次 screen 的 P95 latency `<= 0.50 ms/candidate`；
- hard failed-edge evidence 只按相同 model version 和 edge key 生效。

CheapCert 可以保守地放过 false positive，但不得签发证书。

## M09：ContinuousOptimize

使用同一个 versioned Oracle 和确定性 linearized hand kinematics，分别测试
`SLIDE/REPOSITION/MAKE/BREAK/WRIST_ADJUST`，每类 32 个 seeded case：

- valid request 的 optimize success rate `>= 95%`；
- success case terminal target error `<= 0.75 mm`；
- anchor error `<= 0.75 mm`，joint/collision margin 非负；
- P95 solve latency `<= 10 ms/candidate`；
- 超过 first-prefix trust radius 的 MAKE 输出 `MAKE_PROGRESS`，不改变真实 topology。

Linearized backend 是模块验证夹具，不冒充完整 FR3 nonlinear IK 或 MuJoCo exact collision。

## M10：ExactPrefixAudit

完整 swept prefix 使用每段至少 9 个检查点。一个正例必须签发证书；以下六个 adversarial case
必须全部拒绝：中点碰撞、中途 joint limit、未确认 replacement 的 BREAK、stale model version、
trust-region exceed、prediction suffix authority attempt。随机有效 prefix audit 的 P95 latency
目标为 `<= 5 ms/prefix`。Endpoint-only 检查不计为通过。

## M11：Lazy Beam Search

固定 `H={2,3}`，默认 beam width `8`、per-mode quota `2`。在冻结的小型 deterministic problem
上与 exhaustive search 对比，要求：

- 最优 sequence retention `=100%`，score gap `<=1e-9`；
- 输出 expanded、cheap-survivor、optimized、retained node 数与 latency；
- shifted suffix 只改变候选优先级，不直接成为 committed prefix；
- 最终输出只有第一条 optimized edge 可送 M10，后续均为 prediction suffix。

## M12：Shadow Terminal Viability

- singleton `{1}` 且 inactive finger 2 有 cheap-feasible MAKE 时必须为 `VIABLE`；
- singleton `{1}` 且所有 inactive fingers 均 joint-limited/colliding/unreachable 时必须为
  `NONVIABLE`；
- 非 singleton terminal state 至少有一个非平凡 cheap successor；
- distinct successor finger 计数按 finger 去重，不按 contact-point sample 数计；
- P95 latency 目标 `<= 1 ms/state`；ShadowSucc 永不产生 certificate。

## 生成物与复现

正式生成目录固定为 `Module/generated/m06_m12_mcc_baseline/`，包含 `summary.json`、性能 CSV、
每模块 PNG 和 `index.html`。数值由 benchmark 生成，可视化只读取同一次 benchmark 输出，不改变
阈值。复现命令在实现完成后登记到 `Module/README.md`。
