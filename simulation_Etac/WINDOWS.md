# Windows 上运行 LEAP ETac

从 `Hand_Compliance_Control` 仓库根目录执行下面的命令。LEAP 的 XML 和手部网格随本仓库提供。ETac 的通用代码、传播权重与 ShadowHand 示例资产取自 [Crxus/ETac 固定提交](https://github.com/Crxus/ETac/tree/6dc00c5d3dc49a6915076a2e1c63b9cff02179b0)。

## 环境与上游资源

原开发环境使用 Python 3.10、PyTorch 2.8.0（CUDA 12.8）、MuJoCo 3.6.0、PyTorch3D 0.7.9 和 ETac 定制的 Kaolin 0.18.0，编译工具链使用 CUDA 12.9。上游的 Kaolin 提供网格距离查询需要的 `compute_sdf`。还需上游的 `pytorch_kinematics`、`trimesh`、`transforms3d`、`numpy`、`plotly`，以及查看器使用的 Pillow 与 GLFW。依赖源码与安装步骤见 [上游安装说明](https://github.com/Crxus/ETac/blob/6dc00c5d3dc49a6915076a2e1c63b9cff02179b0/README.md#installation)。

新环境可将上游检出到本仓库旁边。若已有该版本的检出，直接把 `ETAC_SOURCE_ROOT` 设为其实际路径即可。

```powershell
git clone https://github.com/Crxus/ETac.git ../ETac-upstream
git -C ../ETac-upstream switch --detach 6dc00c5d3dc49a6915076a2e1c63b9cff02179b0
$env:ETAC_SOURCE_ROOT = (Resolve-Path ../ETac-upstream).Path
```

在已装好上述依赖的 Python 环境中安装上游包。此目录里的两个本地修改由 `simulation_Etac` 自身加载。

```powershell
python -m pip install --no-build-isolation --no-deps -e "$env:ETAC_SOURCE_ROOT"
```

`ETAC_SOURCE_ROOT` 指向含 `BlindGrasping` 子目录的上游源码根目录。未设置该变量时，程序从已安装的 `ETac` 包定位源码。若环境安装的是普通 wheel，可以显式设置变量以定位权重。LEAP 示例会读取 `BlindGrasping/propagation-ckpt/linear.pth`。

## LEAP 形变

```powershell
python -m simulation_Etac.check_leap
python -m simulation_Etac.view_leap
```

检查脚本在内存中移除 FSR 贴片几何，并让方块逐部位接触。密集模式使用 0.8 mm 的掌侧投影间距，原模型记录为 44,471 个候选节点，其中掌部 17,743 个。当前 `deformation()` 将进入 2 mm 接触带内且位于方块内部的节点投影到表面，再在同一链节传播位移。[仓库自带的 LEAP XML](../src/mjlab/asset_zoo/robots/leaphand_only.xml)

查看器循环展示掌心、食指近节、中指远节和拇指内侧。灰点标记参考表面，彩色点和箭头显示形变。图示放大 8 倍，显示上限为 12 mm，原始位移以米输出。空格键暂停或继续，拖动鼠标旋转视角。

## ShadowHand 环境检查

```powershell
python -m simulation_Etac.check_windows
```

这条命令使用上游发布的 ShadowHand、方块与传播权重检查 GPU 几何库、935 个触觉节点输出和 MuJoCo 步进。LEAP 的 300、500、1000 点表使用本项目自己的掌侧定义，见 [点表规格与策略开发清单](README.md)。
