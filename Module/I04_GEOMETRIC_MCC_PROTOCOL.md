# 新 I04：MCC 两组几何轨迹覆盖实验

日期：2026-10-13。用户已授权新 I04；随后明确本轮仅运行 MCC，DPRef 延后。
本协议对应 `I04_REDESIGN_M06_M12_MIGRATION.md`，不继承旧 G2/G3 解锁条件。

## 固定问题与方法

- A-MCC：Oracle 只暴露 wrist SE(3) 目标，手指保持初始 nominal，真实有效接触指仅执行法向 MCC。
- B-MCC：同一个 Oracle 额外暴露四个 Cartesian fingertip 目标，共享几何规划框架增加 tip position objectives。
- 不使用 M07/M11、SLIDE/MAKE/BREAK、contact-mode certificate、last-contact veto 或 desired contact occupancy。
- 安全保护仅检查关节、非指尖碰撞、危险持续力和数值有效性；零接触继续执行并如实记录。
- Oracle 使用完整 GT 进行终态几何可达性检查，joint witness 不进入 A 的 runtime target。
- Oracle 用附近其他几何目标的终态 IK、collision 与 midpoint 检查 continuation；不预测或保证 contact。

## Fixture 与数值

完整 canonical Stanford Bunny（既有 0.30/0.18 m scale policy 与 90° side rotation），
FR3 7 + LEAP 16 DoF，MuJoCo 3.6.0 native nonconvex mesh SDF，固定物体、gravity=0。
本机使用 Windows Conda `handcomp`，缺失库安装到仓库隔离 `.runtime/`，不修改全局环境。
CPU 是 native MuJoCo physics backend，不冒充 GPU physics；本轮无神经推理。
规划为 synchronous offline simulation，分别报告 wall latency；60 s 指 simulation budget，
不宣称实机实时运行频率。

- Physics 2 ms；记录 10 ms；初始 acquisition 3 s 两组完全共用同一 q/qvel/ctrl snapshot。
- 初始 acquisition 允许 GT normal 引导 6 mm/s approach 并执行同一 MCC；这一特权规则在 treatment 前结束。
  treatment 开始后 A 的 free fingers 不再读取 GT normal 或执行 approach search。
- Formal：seeds 7/11/19 × 两组 × 60 s（含 acquisition）。Smoke：seed 7 × 两组 × 6 s。
- 两组同对象、seed、初态、controller gains、力阈值、几何规划框架与时间/运动上限。
- Wrist max nominal translation 12 mm/s、rotation 0.18 rad/s；joint nominal 0.6 rad/s。
- 实际 command slew ≤1.2 rad/s；joint margin 0.015 rad；prefix feedback 100 ms。
- MCC desired force 2 N；hardware stop：>12 N 连续 20 ms；非指尖 object collision 独立硬停止。
- Geometric terminal tolerances：wrist 2 mm / 0.04 rad；tip 3 mm。
- 完整表面 geodesic coverage radius 25 mm；完整 required set 不删除困难/不可达节点。
- 只有 solver 实际 pad contact 且 force≥0.20 N、mesh residual≤5 mm、normal alignment cos≥0.5
  可增加 coverage。用 mesh-edge truncated Dijkstra 求顶点面积并集；不使用 planned tips 更新覆盖。

## 报告方式

只报告连续性能，策略不设置 PASS/FAIL。记录实际执行时间、停止原因、真实接触连续率、
平均/每指/四指接触、最长零接触、覆盖面积和 required goals、实际 wrist translation/rotation、
力 RMSE/峰值/持续越界、非指尖碰撞、规划及 Oracle latency、拒绝原因、prefix feedback。
提前 hardware stop 必须报告短于共同预算，不能当作完整 60 s。

启动前保存 profile、配置、环境版本、源码与物体 hashes 和输入初态 hashes。
所有轨迹 NPZ、events JSON、逐集 summary 与 aggregate 写到 ignored generated 目录。
此实验仅支持已知 GT、固定物体、gravity-off 的仿真结论；本轮不作未知物体/硬件推论。

## 复现

Windows：`powershell -File scripts/run_new_i04_mcc.ps1 --profile formal --output Module/generated/new_i04_mcc_formal`
Python：`python -m Module.i04_geometric_traversal.benchmark --profile formal --output <fresh_directory>`

完整表面任务成功要求全部 mesh vertices 的面积覆盖；required goals 同时独立报告。
required samples 全部覆盖但面积仍不完整时明确标注不完整；运行到预算结束本身不表示任务完成。
