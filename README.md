# Mustard 222：采集、DP 训练与闭环部署

本版本是 **2026-10-27 的研究代码与数据快照**。不包含新不规则曲面数据，也没有声称模型已在这些曲面上训练。本次仅提供不带机械臂的手部闭环部署；不包含 FR3 部署入口、配置或实验结果。原始采集仍使用机械臂固定手掌、移动物体的方式，保留原始测量字段。

## 文件结构

```text
README.md                 使用说明（从这里开始）
scripts/run.py            统一入口：校验／采集／训练／部署／可视化
configs/                  冻结划分、来源清单、部署配置、模型索引
data/
  mustard222/             三份训练与部署 HDF5
  raw_selected/           222 条原始合格记录，按 episode 编号命名
  collection_references/  199 条仍保留的规划参考
  planning_seed_bank.npz  健康规划起始手型库
checkpoints/              flat24k.pt
reports/                  数据审核、训练记录、评估摘要、环境和验证结果
runtime/                  仿真库、所需采集／训练／部署代码及模型资产
outputs/                  运行时自动生成，不随包发布
FILES.sha256              包内文件校验清单
```

`runtime/` 保留依赖所需的内部布局，普通使用无需在其中寻找入口。训练和采集输出不混入冻结数据目录。

## 1. 下载与安装

代码通过 Git 获取；Release 只下载 `dataset222.tar.gz`、`policies.tar.gz`、`SHA256SUMS`，放进克隆的目录后解压。

```bash
git clone --branch mustard-dp-222-20261027 --depth 1 \
  https://github.com/FerryRain/Hand_Compliance_Control.git mustard222
cd mustard222
# 将同版本 Release 的两个压缩包与 SHA256SUMS 下载到这里
sha256sum -c SHA256SUMS
tar -xzf dataset222.tar.gz
tar -xzf policies.tar.gz
conda create -n mustard222 python=3.10 -y
conda activate mustard222
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./runtime -r requirements-extra.txt
python scripts/run.py verify
```

已测环境：Linux、Python 3.10、RTX 4060 Ti 8 GB、Torch 2.10.0、MuJoCo 3.9.0、MuJoCo-Warp 3.8.1、Warp 1.12.0。完整环境版本见 `reports/environment.json`。`verify` 在 CPU 上核对 222 个源文件的 SHA256、冻结的数据划分，并加载flat 模型执行一次 `[16,16]` 推理；不要求启动 CUDA 仿真。

命令由 `scripts/run.py` 定位仓库根目录，无需手工替换原机器的 `/home/rimlab/...`。原始 HDF5 与 checkpoint 保持字节不变，其中的旧绝对路径仅保留作来源记录。输出写到 `outputs/`。

## 2. 数据与模型

- **222 条 Mustard 轨迹，333,344 帧，单集 800–6022 帧**；169 train / 20 val / 33 test。
- 175 条带 tube，47 条无扰动。按基础规划组冻结划分，不随机拆窗口；同组衍生轨迹不能跨集合。
- `data/mustard222/`，包含 `teachers_commanded_palm.h5`（训练入口）、`teachers_inverted.h5`（部署参考）、`teachers_motion96.h5` ；划分与来源清单放在 `configs/`。
- 222 个原始合格 `data/raw_selected/epXXXX.h5` 全部打包；199 个仍保存的原规划参考也打包。其余 23 个原规划文件已不在本地，不影响训练或部署，但不能声称可以逐文件重放全部原始采集规划。
- `flat24k`：当前对照用的展平历史条件 DP，24,000 步 checkpoint。
- checkpoint 自带网络配置、归一化、特征顺序和划分；不能使用另一份数据的归一化。

具体路径见 `configs/release.json`，训练记录与闭环评估摘要在 Git 的 reports/ 中提供。既有开发轨迹被反复检查过，不应再次宣称它们是全新独立测试集。

## 3. DP 结构和时间语义

每个历史时刻 **96 维**：

| 特征 | 维数 |
|---|---:|
| 四指 q、q 速度 | 16＋16 |
| 接触点、接触法向、有效性 | 12＋12＋4 |
| 接触点速度、法向角速度 | 12＋12 |
| 实际 palm twist、已发布 palm command | 6＋6 |

历史 16 个样本，每 50 ms 一个，覆盖约 750 ms。预测 16 个未来 waypoint，每点 16 个关节，间隔 50 ms。动作是相对**同一个当前实测 q**的 `anchored_delta_q`，部署逐点反归一化后重建绝对关节目标；不是逐 waypoint 累加增量。

- **flat**：历史条件展平，条件化一维 U-Net。
- U-Net 通道 `[256,512,1024]`；DDPM 训练噪声步 100，部署采样 10 步。不是 100 次部署推理。
- DP 约每 100 ms 重规划；scheduler／MCC 控制 100 Hz，物理子步 1 ms。不要任意缩放时序而继续使用原 checkpoint。

## 4. 如何采集

先复采随包提供的 ep48 健康规划参考：

```bash
python scripts/run.py collect --episode 48          # 无 tube
# tube 对照输出到独立的 _tube 目录：
python scripts/run.py collect --episode 48 --tube
```

采集使用机械臂保持 palm、移动物体；教师手型优化＋原 MCC。手指 Kp=35、Kd=0.2、关节力矩限幅±0.8 N·m，四指目标合力各 10 N。执行目标在每个 10 ms 控制周期的十个物理子步中保持。tube 扰动在物理执行中施加，记录实际 q／接触，不能直接扰动标签。

采集完成后自动保存 `acceptance.json`。合格要求完整、稳定初始化、四指及逐指严格率≥95%、最长四指失触≤10 帧、运动阶段无>100 N、执行一致性检查通过。仿真遇超力继续记录并在结束后拒绝该样本，**不是取消验收，也不是可直接用于真机的安全策略**。

生成新的 Mustard 参考时，先规划再健康手型校验，不能任意修改 palm 指令后继续复用旧 q 标签：

```bash
python runtime/mcc_finger_compliance_control/scripts/generate_manifold_palm_plan.py \
  --output outputs/new_palm.h5 --object-id ycb_mustard --frames 1500 \
  --path-mode minimum_enclosing_ellipse --angle-deg 120 --direction 1 \
  --ellipse-section-fraction 0.5 --ellipse-plane-tilt-deg 20 \
  --ellipse-azimuth-deg 30 --global-ellipse-plane-mode long_axis \
  --palm-tangent-sign -1 --object-rotation-mode uniform_so3 --seed 20261027 \
  --time-parameterization arc_length --palm-clearance-range-m .03 .04
python runtime/mcc_finger_compliance_control/scripts/replan_healthy_reference.py \
  --input outputs/new_palm.h5 --output outputs/new_healthy --mode continuous \
  --bank data/planning_seed_bank.npz
```

只有 `outputs/new_healthy/summary.json` 的 `collection_admissible=true` 才继续。新参考采集的底层入口是 `collect_leap_force_matched.py`，完整参数可查看 `scripts/run.py` 调用的 `run_variable_length_collection.make_command()`；批量采样／子轨迹筛选代码为 `run_stratified_variable_collection.py`，它需要已有采集 manifest，并非无输入即启动的脚本。

重新导出随包的 222 条原始记录：

```bash
python scripts/run.py prepare
```

输出为 `outputs/reexport222/`，原冻结数据不会被覆盖；重新导出后的文件不保证与旧导出逐字节相同。新采样数据应先建立独立来源、规划组和冻结划分，再导出，不能直接追加进旧 HDF5 而继续用旧划分。

## 5. 如何训练

```bash
python scripts/run.py train --model flat24k --steps 25000 --device cuda:0
```

这条命令从头训练 flat 架构，**不是从发布权重续训**。使用 batch64、lr=3e-4、seed42、B2 全96维输入、零 FK loss，输出到 `outputs/train_flat24k/`。已发布 flat24k 来自先训练、再降低学习率续训的阶段流程，不能将上面的从头训练命令当作其逐步复现。原阶段参数见 `reports/training/*/launch.json`／`run_manifest.json`。

按 val 选择 checkpoint，再做闭环验证；不要凭训练 loss 或最新步数替换部署模型。

## 6. 如何部署与可视化

默认单环境、原浮动 palm 部署入口；先跑 120 帧检查依赖，再跑完整参考：

```bash
python scripts/run.py deploy --model flat24k --episode 48 --frames 120
python scripts/run.py deploy --model flat24k --episode 48
python scripts/run.py deploy --source teacher --episode 48
python scripts/run.py view --model flat24k --episode 48
```

执行链是 `DP → scheduler → MCC → 手指位置执行器`，本入口 residual 关闭。可视化创建一个环境。默认保留仿真超力诊断，不用于真实硬件。

## 7. 必须同时报告的指标

- **严格健康率**：真实指尖分割碰撞体上的有效接触、有限有效数值、合力≥0.05 N；排除连接件。该接触指标本身不保证全手构型或自碰撞健康。
- **found率**：当前评估中的指尖或连接件检测到接触，不保证真实指尖接触。
- 四指／逐指比例、最长失触、各部位峰值力、完整执行状态；区分全程和去除76帧 bootstrap 后的结果。

指尖分割资产：`runtime/mcc_finger_compliance_control/assets/true_fingertip_contact_v1`；物理接触必须使用分割后碰撞体，不能回退到指尖＋连接件合并凸包。

各上游模型和库保留其 LICENSE；Mustard 网格来源见 `runtime/assets_external/ycb/README.md`。发布校验清单见 `FILES.sha256`。
