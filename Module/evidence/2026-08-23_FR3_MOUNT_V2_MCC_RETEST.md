# FR3–Leap mount v2 与 MCC 复测证据

> 日期：2026-08-23
>
> 环境：`handcomp`
>
> 状态：`UNCOMMITTED_REVIEW_WORK`
> 正式单元：`E05-F-MCC`、`E05-H-MCC`

## 1. 修正内容

v1 的 `palm_lower` 在 kinematic tree 中已经是 `fr3v2_link8` 的固定 child，但错误地把
hand-only XML 的 `80 mm` world placement 沿用为 child transform，形成约 `68.82 mm` 的
不可见 mesh gap。v2 只改变 mount：

- `link8 -> palm_lower` translation：`[0, 0, 0.0112] m`；
- 加入可见、无质量、不参与碰撞的 `fr3_leap_mount_adapter`；
- adapter 半径 `28 mm`、半长 `5.6 mm`；
- 保持 hand orientation、初始接触、任务、controller、seed 和阈值不变。

审计结果：

| 项目 | 结果 |
| --- | ---: |
| direct fixed child | `true` |
| adapter present | `true` |
| body-origin gap | `11.2 mm` |
| flange/palm mesh distance | `0.0206203 mm` |
| mesh-distance limit | `1.0 mm` |
| geometrically closed | `true` |
| plant structure valid | `true` |

专用双视角近景：`Module/generated/visual_demo/fr3_leap_mount_closeup.png`。橙色仅用于在
审计图中突出 adapter；正式视频使用正常材质。

## 2. 冻结与执行

协议在正式复测前冻结：

```text
protocol SHA-256   2858b8b30211d4a83015dd0e1e414abc995e0650f28254788e484d3e8eab5196
code bundle SHA-256 4f75fd375acd98273b416bd77cc61865f4a39b1b2fa8067e0ae33e1d00ebff03
```

执行命令：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python \
  -m Module.module_4_whole_hand_mcc.demo
```

每个单元运行 `3 seeds x 15 s`。全部 episode 完成，模型审计和 trace 完整，因此执行状态均为
`EVALUATED`；性能是否达阈值由 `MET/NOT_MET` 独立表达。

## 3. v2 结果

| aggregate | E05-F-MCC | E05-H-MCC |
| --- | ---: | ---: |
| execution | `EVALUATED` | `EVALUATED` |
| performance | `NOT_MET` | `NOT_MET` |
| mean contact continuity | `100.000%` | `99.981%` |
| mean contact count | `3.748` | `3.709` |
| mean force RMSE | `0.782 N` | `1.020 N` |
| worst peak force | `11.113 N` | `15.751 N` |
| mean force settling | `0.325 s` | `0.647 s` |
| mean Y traversal | `174.035 mm` | `175.283 mm` |
| mean controller P95 | `1.200 ms` | `1.270 ms` |
| mean wrist Fz RMSE | — | `2.006 N` |
| mean internal leakage P95 | — | `0.0121 N` |

F 未达 `force_violation_probability` 与 `max_tip_force_n`。H 未达 `force_rmse_n`、
`force_settling_s`、`force_violation_probability` 与 `max_tip_force_n`。mount 已修正，但
control readiness 仍为 No-Go；尤其 H 的 low-friction peak 为 `15.751 N`，不能因安装修正
而隐藏该性能问题。

## 4. 可视产物

| 文件 | 元数据 / SHA-256 |
| --- | --- |
| `fr3_leap_e05_f_mcc.mp4` | H.264, 960x540, 12 fps, 180 frames, 15 s; `43c046c3c637629b885ce16db04acbefb5d681804d9fbdd19a12ec90a0aa6e1a` |
| `fr3_leap_e05_h_mcc.mp4` | H.264, 960x540, 12 fps, 180 frames, 15 s; `639d65110f6548948c57e510a411bfaa15c5511ab8b7d5c19782fbe89494e6b5` |
| `fr3_leap_mount_closeup.png` | `bb86703e6a81208f1c651de38eff5190c3a0ef1b4aeb8b0b93521464ee0b0588` |

机器校验：

```bash
sha256sum \
  Module/generated/visual_demo/fr3_leap_e05_f_mcc.mp4 \
  Module/generated/visual_demo/fr3_leap_e05_h_mcc.mp4 \
  Module/generated/visual_demo/fr3_leap_mount_closeup.png
```

## 5. 回归

```text
Ran 50 tests in 30.866s
OK
```

新增测试明确检查 parent、adapter、body-origin gap 与 MuJoCo mesh distance。M0、M01、M02、
M03 的既有 full-robot 测试继续通过；mount 修正没有改变这些模块的接口。
