# M02：Fingertip MCC

实现单指法向 compliance、static/sliding/curved benchmark，以及 moving-wrist 四指 wrapper。

| 文件 | 用途 |
| --- | --- |
| `controller.py` | second-order normal MCC 与 signed force-error API |
| `benchmarks.py` | static、tangential、sphere/cylinder protocols |
| `full_robot.py` | FR3+LEAP 四指 wrapper 和 per-finger Jacobian |
| `demo.py` | 数值复现 |
| `visual_demo.py` | trace 可视化 |

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m Module.module_2_fingertip_mcc.demo
```

`compliance_direction` 的正方向必须增加接触压缩；若输入是物体 outward normal，应传 `-n`。
整手 resultant/internal coordination 在 `../module_4_whole_hand_mcc/`。
