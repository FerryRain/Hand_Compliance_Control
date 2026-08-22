# P0 与 Module 1–3 冻结验收协议

冻结日期：2026-08-21。授权范围仅为 P0 最小共享契约和 M01–M03；所有测试固定使用
`handcomp`、SI 单位、右手世界坐标系和随机种子 `7`。

## P0：共享契约

- schema：`hand-modules.v1`；wrist pose 为 `[x,y,z,qw,qx,qy,qz]`；
- mock snapshot 必须 JSON round-trip 等价；
- `A_actual` 必须只由 `contact_states` 计算；
- 非有限值、数组维度不一致和非单位四元数必须拒绝；
- JSONL mock episode 写入并重放后逐条等价。

## M01：Oracle SurfaceModel

- 对象：Plane、Sphere、finite Cylinder、Box、RoundedBox；
- 每种对象随机点数：200；clearance 随机 capsule 数：50；
- point signed-distance 最大绝对误差 `<= 1e-10 m`；
- surface projection residual `<= 1e-9 m`；
- normal 最大角误差 `<= 1e-7 rad`；
- capsule clearance 相对 20,001 点 dense reference 误差 `<= 5e-5 m`；
- 请求 8 个可达 MAKE candidates 时必须返回 8 个，且 surface residual
  `<= 1e-9 m`、reach distance 不超过请求半径。

## M02：Fingertip MCC

统一周期 `dt=0.002 s`，解析接触刚度 `1000 N/m`，评估窗口为最后 `1.0 s`。

- 2A：`f_des={1,2,3} N`；每个目标 force RMSE `<= 0.05 N`，overshoot
  `<= 0.20 N`，`P(f > 3.5 N)=0`；
- 2B：切向最大误差 `<= 1e-9 m`，force RMSE `<= 0.05 N`，settling 后 contact
  loss 为 0；
- 2C：Cylinder 与 Sphere 各自 force RMSE `<= 0.06 N`，切向扰动
  `<= 1e-6 m`，settling 后 contact loss 为 0；
- 所有场景均不得越过配置的 offset/velocity/acceleration limit。

MCC 中传入的 `compliance_direction` 明确定义为“正位移会增加法向接触力”的单位向量；
若 SurfaceModel 返回物体外法向，调用方必须传入其相反方向。

## M01 Mesh Showcase：Bunny/YCB 展示层

这一项是 M01 的展示增强，不替代上述五种解析几何的通过条件：

- 默认资产为 Stanford Bunny 官方重建网格，Y-up 转为 Z-up；
- 使用 uniform scale，使最长边为 `0.30 m`、第二长边至少为 `0.18 m`；
- 水平居中、最低点为 `z=0`，同时画出 10 cm hand-span reference；
- 64 个 surface samples 的 residual `<=1e-8 m`；
- 最近面法向的局部 signed-distance 符号准确率 `>=95%`；
- 返回 12 个有效 contact candidates，并生成可复现 PNG；
- 非 watertight mesh 只声明 near-surface local sign，不声明全局 occupancy 正确。

## M03：Runtime Guards

统一周期 `dt=0.01 s`，stall 阈值 `0.15 s`。

- 200 帧正常 free motion 的 BLOCKED false positive 必须为 0；
- tip force 近零且 command/actual stall 时必须在 `[0.15,0.17] s` 内输出
  `SUSPECTED_OBJECT_BLOCKAGE`；
- `JOINT_LIMIT`、`TIP_OVERFORCE`、`SELF_COLLISION` 必须在一帧内响应；
- 恢复真实运动后 stall timer 必须清零；
- blockage 证据只保留局部观测量，不生成未知碰撞点或碰撞法向。

## 固定复现命令

从仓库根目录运行：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m unittest discover -s Module/tests -v
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.module_1_oracle_surface_model.demo
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.module_2_fingertip_mcc.demo
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.module_3_runtime_guards.demo
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.module_1_oracle_surface_model.mesh_demo
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.visual_demo
```

`Module.visual_demo` 只把上述模块的真实 trace 和 query 结果转成 PNG/GIF/HTML，不新增或
改变数值通过阈值。可视化文件生成失败会由 unittest 检测，但视觉外观本身不是新的
go/no-go gate。

## 2026-08-23：M0–M3 FR3+Leap extension

本节是 additive extension；不改写上面的 `hand-modules.v1` 历史阈值。

### M0-FR3

- schema：`fr3-leap-modules.v1`；canonical joint groups 为 FR3 `7`、Leap `16`；
- palm pose 为 world `[x,y,z,qw,qx,qy,qz]`；
- wrist wrench 必须记录 frame、reference、acting-on-hand 和 estimator source；
- fingertip normal force/vector、contact point/normal 和 validity 分开记录；
- actuator saturation、sensor validity、arm torque、wrist/finger compliance offset 必须存在；
- JSON/JSONL round-trip 等价，`A_actual` 仍只由 measured contact states 推导。

### M01-FR3

- live `MjData` 必须产生 7 个随 arm q 改变的 world-frame planner capsules；
- Oracle clearance 必须有限并带 immutable model version；
- pad/object narrow-phase distance 必须来自 `mj_geomDistance` 并返回 witness points；
- approximate capsule 与 exact MuJoCo geometry 的语义不得混用。

### M02-FR3

- 原 `step(desired, measured)` 行为与阈值保持不变；
- 新 `step_force_error(...)` 必须接受 coordinator 给出的 signed internal error；
- 四指 wrapper 必须拒绝非 `(4,3)/(4,)` 输入；
- contact activation/release 必须 reset 对应 MCC state，防止 stale integrator；
- moving palm 下 fingertip Jacobian 只选择本 finger 的 4 个 DOF。

### M03-FR3

- arm stall 和每根 4-DoF finger stall 分组计时，不允许 norm masking；
- wrist wrench、arm external torque、sensor invalid/stale、actuator saturation、joint limit、
  controller offset 与 robot collision 有独立 reason；
- single-finger block 返回 `FINGER_LOCAL`，arm/wrench/sensor/collision 返回
  `GLOBAL_SAFE_HOLD`；
- 原 M03 的 unknown blockage 边界保持不变，不生成不可测碰撞点/法向。

### FR3 plant structural gate

- exact dimensions `nq=23, nv=23, nu=23`；
- 7 arm + 16 hand actuator；fixed object `nmocap=0`；
- 四个 physical pad parent 必须分别为三个长指 fingertip body 与 thumb fingertip body；
- nominal pose 四个 pad outward-axis 的 world Z 分量均 `< -0.95`；
- home hold 50 physics steps 后状态有限且 gravity-off arm drift `<1e-9 rad`。

FR3 physics E05 的任务、阈值和 hash 不写在本历史协议中，单独冻结于
`E05_MCC_FR3_PROTOCOL.md`。
