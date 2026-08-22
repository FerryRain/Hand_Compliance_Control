# E05-PHY-v3 冻结协议：真实指腹 frame、拇指朝下接触与长程二维复杂曲面

> 冻结日期：2026-08-22
> 状态：`PROTOCOL_FROZEN_V3`
> 只评测：`explicit fingertip reference + FingertipMCC`
> 排除：Finger DP、DP 指标、DP vs. MCC 结论

本协议是仓库内唯一有效的 E05 控制测试标准。任何 E05 MCC 结论都必须来自本协议定义的
MuJoCo 物理接触、正式 trace 和冻结 evaluator；解析 surrogate、手工轨迹或无物理引擎
结果不能作为通过证据。

评测执行状态与性能判定严格分离：成功生成完整数值和可视化产物即记为
`evaluation_status=COMPLETED`；各指标只输出 `MET/NOT_MET`，用于描述 baseline 表现，
不把 `NOT_MET` 写成实验执行失败。正常完成的正式命令返回退出码 `0`。

## 1. 物理边界

- Leap Hand 掌部固定，wrist-relative motion 由物体的等量反向 mocap 运动实现；
- MuJoCo 负责 16 个 finger joints、position actuators、contact constraints 和 contact force；
- 物体是 kinematic mocap body，不验证机械臂 wrist dynamics 或物体自由 SE(3) dynamics；
- 原 hand、FSR 和 rounded-tip meshes 只渲染、不碰撞；
- 只有四个 mesh-registered belly pads 能与物体发生物理接触；
- 接触力只来自 `mujoco.mj_contactForce`；
- SurfaceModel 使用与 collision height-field 独立查询的解析 ground truth；
- non-tip collision 被关闭，本协议不验证 unknown non-tip blockage。

## 2. Mesh-registered 指腹几何

每根手指使用一个 `24 x 16 x 4 mm` 薄椭球。pad 不再挂在 tip-FSR body，也不再通过
world-down 反算一个局部旋转；它直接固定到真实 physical fingertip body：

```text
index  -> fingertip
middle -> fingertip_2
ring   -> fingertip_3
thumb  -> thumb_fingertip
```

在四个 fingertip body frame 中：local `+Y` 是近端方向，local `-X` 是腹侧 outward
方向。pad 长轴沿 body `+Y`，薄轴/normal 沿 body `-X`。长指 pad center 的 local Y 为
`-24 mm`，thumb 为 `-30 mm`；pad distal edge 与 rounded mesh head 的设计间隔分别至少
`13.5 mm` 和 `20.1 mm`。

控制和记录使用 pad-center `site` 与 `mj_jacSite`。DLS IK 同时跟踪 pad center 和
`pad outward normal = -surface normal`。表面 reference 使用椭球沿当前 surface normal
的 support radius。

几何回归必须满足：

- 四个 proxy 全部为 ellipsoid，完整尺寸 `24 x 16 x 4 mm`；
- pad site/geom 的 parent 是上述真实 fingertip body，而不是 FSR body；
- pad distal edge 到 rounded head 的设计间隔 `>=12 mm`；
- 标称姿态下四个 physical pad outward normals 与 world down 的点积 `>=0.999`；
- 实际 contact-to-head clearance 始终 `>=10 mm`；
- tip-head sphere 不存在，所有物体接触 hand geoms 均属于四个 pads。

## 3. 拇指标称关节姿态

固定的拇指关节角为：

```text
q[12:16] = [0.53481, 1.57006, 0.10087, -0.63505] rad
```

该姿态通过真实 `thumb_fingertip` body kinematics 使 body-local `-X` 指腹法向朝下，不能
通过单独旋转 collision geom 冒充。基础 maintenance 中 thumb physical-contact
probability 必须 `>=0.95`；长程复杂曲面突变前也必须 `>=0.95`。

## 4. 固定物理配置

| 项目 | 固定值 |
| --- | --- |
| 环境 | `handcomp` |
| MuJoCo | `3.6.0` |
| timestep | `0.002 s` |
| gravity | `0 0 0 m/s^2` |
| desired force | `2.0 N` |
| contact threshold | `0.20 N` |
| force limit | `8.0 N` |
| nominal friction | `0.90` |
| IK damping / gain | `0.01 / 0.20` |
| joint command margin | `0.08 rad` |
| nominal seed | `7` |

## 5. 基础 sanity checks：5A/5B/5C

### 5A Maintenance

- plane translation `40 mm`；plane rotation `5 deg`；sphere translation `25 mm`；
- 每场 hand continuity `>=0.999`，average contacts `>=3.90`；
- force RMSE `<=0.30 N`，maximum force `<=8.0 N`；
- thumb contact `>=0.95`，actual contact-to-head clearance `>=10 mm`；
- zero-contact time `<=0.002 s`；minimum joint margin `>=0.05 rad`；
- joint-limit/non-tip contact 均为零。

### 5B Handover

```text
{1,2,3} -> {1,2} -> {1,2,4}
```

要求 anchors 1/2 全程保持、MAKE recovery `<=0.25 s`、zero-contact 为零、最后 `0.5 s`
持续为 `{1,2,4}`、maximum force `<=8.0 N`、minimum joint margin `>=0.05 rad`。

### 5C Robustness

固定 seed `0..23` 的 24 episodes。聚合要求 success `>=0.90`、contact continuity
`>=0.995`、average contacts `>=3.50`、force RMSE `<=0.35 N`、force violation
`<=0.005`，且无 joint-limit/non-tip contact。

## 6. 主挑战：长程二维多尺度连续曲面

物体是 `0.60 x 0.84 m` 的 `301 x 421` MuJoCo height-field。解析 C-infinity surface
同时依赖 `x,y`，由长波、斜向短波、交叉波纹、四个不同尺度的凸/凹 Gaussian feature
以及斜向窄 tanh ridge 构成。冻结的 dense characterization 为：

- height range：约 `22.30 mm`；
- maximum gradient norm：约 `0.775`；
- maximum principal curvature：约 `62.63 1/m`；
- minimum curvature radius：约 `15.97 mm`。

episode 为 `15.0 s`，settling `0.75 s`。物体沿 Y 净移动 `480 mm`，同时在 X 方向叠加
两种频率的摆动形成 S-scan；标称全程 path length 约 `568 mm`。突变前连续段
`[0.75,10.0) s` 必须至少覆盖 `300 mm` path。surface/oracle vertical-ray audit 最大高度
误差必须 `<=0.05 mm`。

连续段按最大主曲率分三类记录：

```text
low      : [0, 10) 1/m
high     : [10, 40) 1/m
extreme  : >= 40 1/m
```

连续段评价阈值：

- hand contact probability `>=0.995`；average contact count `>=3.50`；
- thumb contact probability `>=0.95`；contact-to-head clearance `>=10 mm`；
- relative path length `>=0.30 m`；
- force RMSE `<=0.50 N`；force violation probability `<=0.005`；
- maximum physical force `<=8.0 N`。

## 7. 受控突变与恢复

在 `t=10.0 s`，mocap object 沿 surface normal 瞬时远离手指 `4 mm`，之后继续长程 S-scan。
contact recovery 使用连续 `50 ms` 确认；force settling 使用连续 `100 ms` 四指接触且每指
force 位于 `[1.5,2.5] N`。

恢复评价阈值：

- any-contact recovery `<=0.10 s`；all-finger recovery `<=0.25 s`；
- longest hand-level zero-contact `<=0.15 s`；force settling `<=0.75 s`；
- minimum joint margin `>=0.05 rad`；non-tip contact count `==0`。

快速重新接触不能抵消 over-force 或 force-settling 的 `NOT_MET`；综合性能判定要求所有
阈值同时满足，但无论结果如何，完整产出后评测状态均为 `COMPLETED`。

## 8. 正式可视化与产物

正式命令输出到 `Module/generated/e05_physics_v3/`：

- `summary.json`、`robustness_episodes.csv`、`traces.npz`；
- `pad_geometry_audit.png`：object-hidden underside、长指侧视和拇指接触侧视；
- 基础与 extreme dashboards；
- 五段 MP4，其中 extreme video 为 `15 s`；
- extreme pre-step、post-step、recovered 等关键帧；
- `index.html`，明确展示 geometry audit、thumb contact、path length 与 `MET/NOT_MET`。

正式命令：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.e05_physics.demo
```

协议变更后必须升版本并重跑全部结果；不得用解析结果、旧 trace 或短视频覆盖正式判定。
