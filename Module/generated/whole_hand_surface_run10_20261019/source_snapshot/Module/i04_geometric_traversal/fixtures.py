"""Fixed object fixtures; source geometry and physics backend stay explicit."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
import trimesh

from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q, FullRobotModelConfig, build_full_robot
from Module.fr3_leap.model import compile_model_xml
from Module.i04_oracle_next_point.surface_graph import BunnySurfaceGraph

ROOT = Path(__file__).resolve().parents[2]
MUSTARD = ROOT / 'assets_external/ycb/collision/006_mustard_bottle/vhacd_256_scaled2p8_objstage'


class SurfaceGraph(BunnySurfaceGraph):
    """Cover every mesh component, including the mustard cap and small sheets."""
    def _farthest_point_cover(self, seed_vertex):
        minimum = np.full(len(self.vertices_m), np.inf)
        owner = np.full(len(self.vertices_m), -1, dtype=np.int32)
        selected = []
        current = int(seed_vertex)
        while True:
            goal_id = len(selected)
            selected.append(current)
            distances = dijkstra(self.adjacency, indices=current, directed=False)
            improved = distances < minimum
            minimum[improved] = distances[improved]
            owner[improved] = goal_id
            disconnected = np.flatnonzero(~np.isfinite(minimum))
            if len(disconnected):
                current = int(disconnected[0])
                continue
            radius = float(minimum.max())
            if radius <= self.coverage_radius_m + 1e-12:
                return np.asarray(selected, dtype=np.int32), owner, radius
            current = int(np.argmax(minimum))


class ObjectPlant:
    """23-DoF handles plus the complete set of physical object geoms."""
    def __init__(self, base, object_geom_ids, fixture):
        self.base = base
        self.object_geom_ids = np.asarray(object_geom_ids, dtype=np.int32)
        self.fixture = fixture

    def __getattr__(self, name):
        return getattr(self.base, name)


@dataclass
class FixedObjectFixture:
    name: str
    mesh: trimesh.Trimesh
    mesh_path: Path
    collision_parts: list[Path]
    metadata: dict

    def export_visual_mesh(self, path):
        path = Path(path)
        self.mesh.export(path)
        return path

    def make_plant(self, dt_s):
        if self.name == 'bunny':
            base = build_full_robot(FullRobotModelConfig(
                surface='bunny', timestep_s=dt_s, gravity_m_s2=0,
                arm_kp=1800, arm_damping_ratio=.9, hand_kp=60, hand_damping_ratio=1.5,
                bunny_visual_mesh_path=str(self.mesh_path), bunny_collision_mode='sdf'))
            base.model.geom_conaffinity[base.object_geom_id] |= 4
            return ObjectPlant(base, [base.object_geom_id], self)
        base = build_full_robot(FullRobotModelConfig(
            surface='plane', timestep_s=dt_s, gravity_m_s2=0,
            arm_kp=1800, arm_damping_ratio=.9, hand_kp=60, hand_damping_ratio=1.5))
        data = mujoco.MjData(base.model)
        data.qpos[base.arm_qpos_adrs] = ARM_HOME_Q
        data.qpos[base.hand_qpos_adrs] = HAND_NATURAL_Q
        mujoco.mj_forward(base.model, data)
        tips = data.site_xpos[base.tip_site_ids]
        # Start on the broad bottle body. Relative placement is fixed before
        # any policy is run; scale 2.8 is already baked into the source asset.
        region = np.array([0.0, 0.10, 0.0])
        ray = np.array([[region[0], region[1], self.mesh.bounds[1, 2] + .1]])
        locations, _, _ = self.mesh.ray.intersects_location(ray, [[0, 0, -1]], multiple_hits=True)
        if not len(locations):
            raise ValueError('Mustard initial region does not intersect the source mesh')
        top = float(locations[:, 2].max())
        position = np.array([tips[:, 0].mean() - region[0],
                             tips[:, 1].mean() - region[1], tips[:, 2].min() - top - .012])
        # A bottle's shoulder is higher than the center ray. Check each
        # entire pad footprint so initialization starts without penetration.
        for geom in base.tip_geom_ids:
            center = data.geom_xpos[geom]
            half = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ base.model.geom_size[geom]
            xs = np.linspace(center[0] - half[0], center[0] + half[0], 5) - position[0]
            ys = np.linspace(center[1] - half[1], center[1] + half[1], 5) - position[1]
            xx, yy = np.meshgrid(xs, ys)
            origins = np.column_stack((xx.ravel(), yy.ravel(),
                                       np.full(xx.size, self.mesh.bounds[1, 2] + .1)))
            hits, _, _ = self.mesh.ray.intersects_location(origins,
                np.tile([0, 0, -1], (len(origins), 1)), multiple_hits=True)
            if len(hits):
                position[2] = min(position[2], center[2] - half[2] - hits[:, 2].max() - .004)
        root = ET.fromstring(base.xml)
        body = root.find(".//body[@name='fr3_e05_object']")
        asset = root.find('asset')
        body.set('pos', ' '.join(str(x) for x in position))
        for child in list(body):
            body.remove(child)
        names = []
        for index, path in enumerate(self.collision_parts):
            mesh_name = f'mustard_collision_mesh_{index}'
            geom_name = 'fr3_e05_object_geom' if index == 0 else f'mustard_collision_geom_{index}'
            names.append(geom_name)
            ET.SubElement(asset, 'mesh', name=mesh_name, file=str(path))
            ET.SubElement(body, 'geom', name=geom_name, type='mesh', mesh=mesh_name,
                          density='0', contype='2', conaffinity='5',
                          friction='0.9 0.01 0.001', solref='0.028 1',
                          solimp='0.90 0.95 0.001', rgba='0.96 0.72 0.08 0', group='4')
        ET.SubElement(asset, 'mesh', name='mustard_visual_mesh', file=str(self.mesh_path))
        texture = MUSTARD / 'material_0.png'
        material_attributes = {}
        if texture.is_file():
            ET.SubElement(asset, 'texture', name='mustard_texture', type='2d', file=str(texture))
            ET.SubElement(asset, 'material', name='mustard_material', texture='mustard_texture',
                          texuniform='false', rgba='1 1 1 1')
            material_attributes = {'material': 'mustard_material'}
        ET.SubElement(body, 'geom', name='mustard_visual', type='mesh', mesh='mustard_visual_mesh',
                      density='0', contype='0', conaffinity='0', group='2',
                      rgba='1 1 1 1' if material_attributes else '0.96 0.72 0.08 1', **material_attributes)
        xml = ET.tostring(root, encoding='unicode')
        model = compile_model_xml(xml)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
        robot_names = [mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in base.robot_geom_ids]
        updated = replace(base, model=model, xml=xml, object_geom_id=ids[0], object_position_m=position,
                          robot_geom_ids=np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                                                     for n in robot_names], dtype=np.int32))
        return ObjectPlant(updated, ids, self)


def load_fixture(name, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if name == 'bunny':
        from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
        bunny = canonical_bunny_heightfield()
        path = bunny.export_visual_mesh(directory / 'canonical_object.obj')
        return FixedObjectFixture(name, bunny.mesh, path, [],
            {'source_sha256': bunny.source_sha256, 'collision_backend': 'native_nonconvex_sdf'})
    if name != 'mustard':
        raise ValueError('Supported objects: mustard, bunny')
    source = MUSTARD / 'visual_scaled.obj'
    mesh = trimesh.load(source, force='mesh', process=True)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    mesh.apply_transform(transform)
    bounds = mesh.bounds.copy()
    translation = np.r_[-0.5 * (bounds[0, :2] + bounds[1, :2]), -bounds[0, 2]]
    mesh.apply_translation(translation)
    transform[:3, 3] = translation
    visual = directory / 'canonical_object.obj'
    mesh.export(visual)
    parts_directory = directory / 'collision_parts'
    parts_directory.mkdir(exist_ok=True)
    parts = []
    for index, source_part in enumerate(sorted(MUSTARD.glob('collision_part_*.obj'))):
        part = trimesh.load(source_part, force='mesh', process=False)
        part.apply_transform(transform)
        target = parts_directory / f'part_{index:03d}.obj'
        part.export(target)
        parts.append(target)
    if len(parts) != 256:
        raise ValueError('The mustard fixture requires the existing 256-part derivative')
    return FixedObjectFixture(name, mesh, visual, parts,
        {'source_sha256': sha256(source.read_bytes()).hexdigest(),
         'source_manifest_sha256': sha256((MUSTARD / 'manifest.json').read_bytes()).hexdigest(),
         'collision_backend': 'vhacd_256', 'source_baked_scale': 2.8, 'runtime_scale': 1.0,
         'source_components': len(mesh.split(only_watertight=False)),
         'canonical_extents_m': mesh.extents.tolist(),
         'license': 'YCB Object and Model Set, CC BY 4.0',
         'source_transform': transform.tolist()})
