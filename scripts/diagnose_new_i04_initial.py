"""Print actual pad/mesh geometry for the shared initial state."""
from pathlib import Path
import json
import mujoco
import numpy as np
import trimesh
from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point.surface_graph import BunnySurfaceGraph
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.physics import make_plant, tip_positions, pad_support
from Module.i04_geometric_traversal.runner import prepare_initial_state

out = Path('Module/generated/new_i04_initial_diagnostic')
out.mkdir(parents=True, exist_ok=True)
bunny = canonical_bunny_heightfield()
h = make_plant(bunny.export_visual_mesh(out / 'bunny.obj'), 0.002)
g = BunnySurfaceGraph(bunny.mesh)
state, audit = prepare_initial_state(h, g, ExperimentConfig())
d = mujoco.MjData(h.model)
d.qpos[:] = state['qpos']
d.qvel[:] = state['qvel']
d.ctrl[:] = state['ctrl']
mujoco.mj_forward(h.model, d)
tips = tip_positions(h, d)
points, distances, triangles = trimesh.proximity.closest_point(g.mesh, tips - h.object_position_m)
normals = g.mesh.face_normals[triangles]
print(json.dumps(audit, indent=2))
print('site signed-distance / actual pad support', distances,
      [pad_support(h, d, i, normals[i]) for i in range(4)])
print('q_hand', d.qpos[h.hand_qpos_adrs])
print('ncon', d.ncon, 'contacts', [(int(d.contact[i].geom1), int(d.contact[i].geom2), float(d.contact[i].dist))
                                 for i in range(d.ncon)])
print('object', h.object_geom_id, 'tips', h.tip_geom_ids)
print('site/geom offsets', tips - d.geom_xpos[h.tip_geom_ids])
