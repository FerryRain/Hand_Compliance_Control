# I04 视频和历史数据下载

I04 代码、启动所需模型、完整初态及紧凑 IK 种子库保留在 GitHub `main`。视频与历史实验记录存放在 Google Drive：

- [I04 A/B 总目录](https://drive.google.com/drive/folders/1Yq8yGz8CFHmwzlfuiJr7jL2D6Y3-AwiL)
- [历史数据压缩包](https://drive.google.com/drive/folders/1iTa0i0acOOslYgrYJ37g80mg5Ahao9j2)
- [完整视频和接续段视频](https://drive.google.com/drive/folders/1TknGR1qpzUxk6Gex-x_Y3YoJJWKstOZA)

请使用有权访问该目录的 Google 账号。启动新实验只需要 GitHub 中的代码和运行输入。回放历史轨迹或重新审核时，请下载对应方法的数据。[下载清单](downloads.json)列出文件名、字节数和 SHA-256。

## 下载内容

| 数据 | 目录 | 原始结论 |
| --- | --- | --- |
| A 的现存最佳完整记录 | `Module/generated/i04a_frozen_b01_20261021` | 14.6668% 覆盖，未完成；1,012,618 个原生状态 |
| B 的掌部修复完成记录 | `Module/generated/i04b_palm_recovery02_20261027` | 原 25 mm 测地邻域指标 100%；3,459,113 个原生状态；含控制版本接续 |

压缩包包含原始轨迹、初末态、模型/碰撞网格、源码快照、独立审核、控制记录及视频来源信息。MP4 单独放在视频目录，不重复装入 ZIP。两案物理碰撞模型不同，不是严格同条件 A/B 配对。更早已按要求删除的实验不在本次现存档案中。

## 解压和回放

1. 下载所需方法的全部历史包。A 有 17 个 `I04_A_history_*.zip`，包含 454 个原文件。B 有 84 个 `I04_B_history_*.zip`，包含 751 个原文件。若 Drive 将批量下载包成外层 ZIP，请先解开外层。
2. 将每个历史 ZIP 独立解压到**仓库根目录**。压缩包自带 `Module/generated/...` 路径。
3. 安装 Python 依赖后运行回放：

```powershell
.\scripts\run_i04_ab.ps1 -Python python replay --method B --speed 12
.\scripts\run_i04_ab.ps1 -Python python smoke --method B --states 500
```

改为 `--method A` 可回放或检查 A。完整使用说明见 [I04 A/B 保留版](../../Module/I04_AB_RETAINED.md)。

## 视频保存位置

可在 Drive 播放原速或 12 倍速视频。较大的四个视频按关键帧拆成可独立播放的 MP4，按编号连续观看。分段保留原始编码帧。

| Drive 下载 | 还原后的仓库路径 |
| --- | --- |
| [A 完整 12×，单个 MP4](https://drive.google.com/file/d/1ThUQ7qZmYeTtUixPDcOeFFjBfbNcbME4/view) | `Module/generated/i04a_frozen_b01_20261021/complete_videos/a_complete_12x.mp4` |
| [A 完整原速，7 段](https://drive.google.com/drive/folders/1s5iCem60zKp9Rrh5gLKY-Mvb0QOXMPO3) | `Module/generated/i04a_frozen_b01_20261021/complete_videos/a_complete_1x.mp4` |
| [B 完整 12×，9 段](https://drive.google.com/drive/folders/1cGrPXi85pM2-H7edATTZEXwEqfPGx5FF) | `Module/generated/i04b_palm_recovery02_20261027/complete_videos/b_complete_12x.mp4` |
| [B 完整原速，70 段](https://drive.google.com/drive/folders/1A3m98jnEK6X3U-45FjfAufVhG3KqDdqI) | `Module/generated/i04b_palm_recovery02_20261027/complete_videos/b_complete_1x.mp4` |
| [B 接续段 12×，单个 MP4](https://drive.google.com/file/d/1j1kStLkQcqnMLa4RXdtmQb9053gPfVMP/view) | `Module/generated/i04b_palm_recovery02_20261027/tail_native_videos/b_complete_12x.mp4` |
| [B 接续段原速，9 段](https://drive.google.com/drive/folders/10A_GpO37hZNqWih1O7Y9ny21gHXgSxja) | `Module/generated/i04b_palm_recovery02_20261027/tail_native_videos/b_complete_1x.mp4` |

要还原完整 MP4，下载对应视频目录中的全部分段和 `*_layout.zip`。解开 Drive 的外层下载包，保留里面的 `*_layout.zip`。在仓库根目录运行以下命令，把示例路径换成解压后的下载目录。只需 Python 3.10 或更高版本的标准库。

```powershell
python data/i04/restore_videos.py "D:/Downloads/I04" --video B_complete_1x
```

该命令核对每段 SHA-256，还原并核对原始 MP4 后才写入表中的位置。去掉 `--video B_complete_1x` 会处理全部六个视频，包括两个直接下载的 MP4。已有同内容文件会跳过，遇到不同内容会停止。

下载两案全部历史数据和视频后，运行 `verify` 校验原始文件 SHA-256；只下载代码时运行 `verify --code-only`。此校验不等同于重新执行完整物理审核。

```powershell
.\scripts\run_i04_ab.ps1 -Python python verify --code-only
.\scripts\run_i04_ab.ps1 -Python python verify
```

Git 历史中的旧提交可能仍含过往 LFS 指针；本次没有重写已有 Git 历史。当前 `main` 使用本页的 Drive 下载入口。
