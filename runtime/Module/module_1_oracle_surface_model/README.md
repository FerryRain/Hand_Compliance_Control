# M01：Oracle SurfaceModel

提供解析几何、Bunny/YCB-style mesh 展示和 FR3+LEAP live geometry adapter。

| 文件 | 用途 |
| --- | --- |
| `surface_model.py`、`geometry.py` | plane/sphere/cylinder/box/rounded-box surface、normal、clearance |
| `mesh_surface.py` | 大尺寸 Bunny/YCB-style mesh query 与 candidates |
| `robot_geometry.py` | live FR3 capsules 与 MuJoCo pad/object exact distance |
| `demo.py`、`mesh_demo.py` | 数值复现 |
| `visual_demo.py` | PNG/GIF 可视化 |

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m Module.module_1_oracle_surface_model.demo --seed 7
$PY -m Module.module_1_oracle_surface_model.mesh_demo
```

验收阈值见 `../PROTOCOL.md`；总 gallery 在 `../generated/visual_demo/index.html`。
