# FR3 + LEAP Hand Compliance Control

当前项目入口是 [`Module/`](Module/README.md)。实现按 M0–M4 分层验证，并在 E05 中分别
评测 Fingertip MCC 与 coordinated whole-hand MCC。

## 当前状态

- 当前唯一 MCC 协议：[`Module/E05_MCC_CURRENT_PROTOCOL.md`](Module/E05_MCC_CURRENT_PROTOCOL.md)
- 当前正式结果：`E05-F-MCC` 与 `E05-H-MCC`，均为 `EVALUATED / NOT_MET`
- Finger DP：`REWORK_REQUIRED / NOT_EVALUATED`
- 本轮不实现、采集、训练或评测 DP，只讨论下一版协议

`EVALUATED / NOT_MET` 表示完整评测有效，但部分预冻结性能阈值未达到，不表示运行失败。

## 复现入口

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m unittest discover -s Module/tests -v
$PY -m Module.module_4_whole_hand_mcc.demo
```

详细结构、接口、视频和逐模块复现方式见 [`Module/README.md`](Module/README.md)。所有新实现
和新产物只允许放在 `Module/` 内；仓库其他历史目录保留但不作为当前入口，也不随本模块
工作修改。
