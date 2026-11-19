# I04 视频和历史数据下载

I04 代码、启动所需模型、完整初态及紧凑 IK 种子库保留在 GitHub `main`。视频与历史实验记录存放在 Google Drive：

- [I04 A/B 总目录](https://drive.google.com/drive/folders/1Yq8yGz8CFHmwzlfuiJr7jL2D6Y3-AwiL)
- [历史数据压缩包](https://drive.google.com/drive/folders/1iTa0i0acOOslYgrYJ37g80mg5Ahao9j2)
- [完整视频和接续段视频](https://drive.google.com/drive/folders/1TknGR1qpzUxk6Gex-x_Y3YoJJWKstOZA)

Drive 文件保持账号原有访问权限，没有改为公开分享。新实验可以只使用 GitHub 中的代码和运行输入；回放、完整保留集校验和历史重新审核需要下载对应数据。

## 下载内容

| 数据 | 目录 | 原始结论 |
| --- | --- | --- |
| A 的现存最佳完整记录 | `Module/generated/i04a_frozen_b01_20261021` | 14.6668% 覆盖，未完成；1,012,618 个原生状态 |
| B 的掌部修复完成记录 | `Module/generated/i04b_palm_recovery02_20261027` | 原 25 mm 测地邻域指标 100%；3,459,113 个原生状态；含控制版本接续 |

压缩包包含原始轨迹、初末态、模型/碰撞网格、源码快照、独立审核、控制记录及视频来源信息。MP4 单独放在视频目录，不重复装入 ZIP。两案物理碰撞模型不同，不是严格同条件 A/B 配对。更早已按要求删除的实验不在本次现存档案中。

## 解压和回放

1. 下载所需方法的全部 `I04_A_history_*.zip` 或 `I04_B_history_*.zip`。每个 ZIP 都可独立解压，不需要二进制拼接。
2. 把所有 ZIP 解压到**仓库根目录**。压缩包自带 `Module/generated/...` 路径；不要解压到 `data/i04` 下。
3. 安装 Python 依赖后运行回放：

```powershell
.\scripts\run_i04_ab.ps1 -Python python replay --method B --speed 12
.\scripts\run_i04_ab.ps1 -Python python smoke --method B --states 500
```

改为 `--method A` 可回放或检查 A。完整使用说明见 [I04 A/B 保留版](../../Module/I04_AB_RETAINED.md)。

## 视频保存位置

视频可以直接在 Drive 播放。要在本地执行完整 `verify`，请按以下映射保存原文件名：

| Drive 文件名 | 仓库内保存路径 |
| --- | --- |
| `A_complete_12x.mp4` | `Module/generated/i04a_frozen_b01_20261021/complete_videos/a_complete_12x.mp4` |
| `A_complete_1x.mp4` | `Module/generated/i04a_frozen_b01_20261021/complete_videos/a_complete_1x.mp4` |
| `B_complete_12x.mp4` | `Module/generated/i04b_palm_recovery02_20261027/complete_videos/b_complete_12x.mp4` |
| `B_complete_1x.mp4` | `Module/generated/i04b_palm_recovery02_20261027/complete_videos/b_complete_1x.mp4` |
| `B_tail_12x.mp4` | `Module/generated/i04b_palm_recovery02_20261027/tail_native_videos/b_complete_12x.mp4` |
| `B_tail_1x.mp4` | `Module/generated/i04b_palm_recovery02_20261027/tail_native_videos/b_complete_1x.mp4` |

下载两案全部历史数据和视频后，运行 `verify` 校验原始文件 SHA-256；只下载代码时运行 `verify --code-only`。此校验不等同于重新执行完整物理审核。

```powershell
.\scripts\run_i04_ab.ps1 -Python python verify --code-only
.\scripts\run_i04_ab.ps1 -Python python verify
```

Git 历史中的旧提交可能仍含过往 LFS 指针；本次没有重写已有 Git 历史。当前 `main` 使用本页的 Drive 下载入口。
