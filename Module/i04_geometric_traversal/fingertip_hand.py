"""I04-c physical plant and per-contact measurements in the I04 bottle frame."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from mcc_finger_compliance_control.scripts.split_fingertip_collision import (
    ASSETS, BODIES, GEOMS, replace_tip_collision,
)
from .physics import quaternion, rotation

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / 'src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml'
FIXTURE = ROOT / 'Module/generated/whole_hand_surface_run10_20261019'
SURFACE = FIXTURE / 'canonical_task_surface.obj'
SEED = Path(__file__).with_name('i04c_ep71_grasp.json')
DT = .002
HEAD_FRACTION = .75
MAX_JOINT_DEVIATION = .36
MAX_POSTURE_RMS = .18
MAX_HEAD_ANGLE = 42.
FINGERS = ('index', 'middle', 'ring', 'thumb')
JOINTS = ('1', '0', '2', '3', '5', '4', '6', '7',
          '9', '8', '10', '11', '12', '13', '14', '15')


def reference_grasp():
    seed = json.loads(SEED.read_text(encoding='utf-8'))
    # fixtures.load_fixture rotates the source YCB mesh +90 degrees about X,
    # centers X/Y and places its lowest Z at zero. The same rigid transform
    # takes the finger-policy pose into the retained B collision geometry.
    frame = Rotation.from_euler('x', 90, degrees=True)
    pose = seed['palm_pose_object_wxyz']
    position = frame.apply(pose[:3]) + seed['source_to_i04_translation_m']
    return position, frame * rotation(pose[3:]), np.array(seed['q_hand_rad'])


def build_model(*, surface=SURFACE, collision_parts=None, object_name='mustard', model_name='I04-c'):
    xml = ET.parse(HAND).getroot()
    xml.set('model', model_name)
    xml.find('compiler').set('usethread', 'false')
    ET.SubElement(xml, 'option', timestep=str(DT), gravity='0 0 0', integrator='implicitfast')
    xml.find('default/joint').attrib.update(damping='0.08', frictionloss='0.01', armature='0.001')
    xml.find('default/geom').attrib.update(solref='0.012 1', solimp='0.9 0.95 0.001')
    world, assets = xml.find('worldbody'), xml.find('asset')
    palm = world.find("body[@name='palm_lower']")
    del palm.attrib['euler']
    palm.attrib.update(pos='0 0 0', quat='1 0 0 0')
    contact = xml.find('contact')
    contact.remove(contact.find("exclude[@body2='pip4']"))
    world.append(ET.Element('body', name='palm_target', mocap='true'))
    eq = ET.SubElement(xml, 'equality')
    ET.SubElement(eq, 'weld', name='palm_tracking', body1='palm_target', body2='palm_lower',
                  relpose='0 0 0 1 0 0 0', torquescale='1', solref='0.004 1',
                  solimp='0.995 0.999 0.001')
    files = {}
    for mesh in assets.findall('mesh'):
        source = HAND.parent / mesh.get('file')
        mesh.set('file', source.name)
        files[source.name] = source.read_bytes()

    obj = ET.SubElement(world, 'body', name=object_name)
    if collision_parts is None:
        collision_parts = [FIXTURE / 'collision_parts' / f'part_{index:03d}.obj'
                           for index in range(256)]
    for index, path in enumerate(collision_parts):
        name = f'part_{index:03d}'
        asset_file = name + path.suffix
        files[asset_file] = path.read_bytes()
        ET.SubElement(assets, 'mesh', name=name, file=asset_file)
        ET.SubElement(obj, 'geom', name=name, type='mesh', mesh=name, group='3',
                      rgba='0 0 0 0', contype='2', conaffinity='1', density='0',
                      friction='0.9 0.01 0.001', solref='0.028 1', solimp='0.90 0.95 0.001')
    files[surface.name] = surface.read_bytes()
    ET.SubElement(assets, 'mesh', name='bottle_surface', file=surface.name)
    ET.SubElement(obj, 'geom', name='bottle_visual', type='mesh', mesh='bottle_surface',
                  contype='0', conaffinity='0', density='0', rgba='0.94 0.65 0.09 1')
    for geom in palm.iter('geom'):
        # Positive-distance witnesses support planning. Matching margin/gap
        # keeps physical force activation at zero distance.
        geom.attrib.update(conaffinity='3', margin='0.0025', gap='0.0025')
    actuators = ET.SubElement(xml, 'actuator')
    for name in JOINTS:
        ET.SubElement(actuators, 'position', name='drive_' + name, joint=name, kp='35',
                      kv='1.1', forcerange='-35 35', inheritrange='1')
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'), assets=files)
    return replace_tip_collision(spec).compile()


@dataclass
class Touch:
    forces: np.ndarray
    loads: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    min_fraction: np.ndarray
    incidental: set[str]
    posture_metrics: np.ndarray
    legal_posture: bool

    @property
    def measured_heads(self):
        return (self.forces >= .2) & (np.linalg.norm(self.normals, axis=1) > .99)

    @property
    def eligible(self):
        return self.measured_heads & (self.loads.max() < 8.) & self.legal_posture


class FingertipHand:
    def __init__(self, model, *, object_name='mustard', grasp=None, min_head_fraction=HEAD_FRACTION):
        self.model = model
        self.palm = model.body('palm_lower').id
        self.driver = model.body('palm_target').mocapid[0]
        self.pads = np.array([model.geom(g + '_pad_collision').id for g in GEOMS])
        self.pad_index = {g: i for i, g in enumerate(self.pads)}
        object_body = model.body(object_name).id
        self.objects = set(np.flatnonzero((model.geom_bodyid == object_body) & (model.geom_contype > 0)))
        self.min_head_fraction = min_head_fraction
        self.q = np.array([model.joint(name).qposadr[0] for name in JOINTS])
        self.dof = np.array([model.joint(name).dofadr[0] for name in JOINTS])
        self.limits = np.array([model.joint(name).range for name in JOINTS])
        if (model.njnt, model.nu, model.nmocap) != (17, 16, 1):
            raise ValueError('Expected one free palm and sixteen finger hinges')
        # Express the physical split plane in distal-body coordinates. The
        # compiled mesh frame has been recentered by MuJoCo, the source STL has not.
        source = ET.parse(HAND)
        self.head_axes, self.head_origins, self.head_spans = [], [], []
        for i, g in enumerate(GEOMS):
            kind = 'thumb_fingertip_two_part' if i == 3 else 'fingertip_two_part'
            split = json.loads((ASSETS / kind / 'split_manifest.json').read_text())
            sign = 1 if split['pad_side'] == 'positive' else -1
            axis = sign * np.asarray(split['plane_normal_source_mesh'])
            cut = sign * split['plane_offset_m']
            mesh = trimesh.load(ASSETS / kind / 'pad_convex.obj', force='mesh', process=False)
            span = float((mesh.vertices @ axis).max() - cut)
            geom = source.find(f".//geom[@name='{g}']")
            rot = rotation(np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '))
            body_axis = rot.apply(axis)
            origin = np.fromstring(geom.get('pos'), sep=' ') + body_axis * cut
            self.head_axes.append(body_axis)
            self.head_origins.append(origin)
            self.head_spans.append(span)
        self.tip_bodies = np.array([model.body(b).id for b in BODIES])
        self.head_axes = np.asarray(self.head_axes)
        pos, orient, self.reference_q = reference_grasp() if grasp is None else grasp
        self.inward_in_palm = orient.inv().apply([0., 0., -1.])
        self.palm_geom = model.geom('palm_lower_collision').id
        roots = [model.body(name).id for name in ('mcp_joint', 'mcp_joint_2', 'mcp_joint_3', 'pip_4')]
        self.load_region = np.full(model.ngeom, 4, dtype=int)
        for geom, body in enumerate(model.geom_bodyid):
            while body > 0 and body not in roots:
                body = model.body_parentid[body]
            if body in roots:
                self.load_region[geom] = roots.index(body)
        d = self.initialize(pos, orient, self.reference_q)
        self.object_ids = np.array(sorted(self.objects))
        rotations = d.geom_xmat[self.object_ids].reshape(-1, 3, 3)
        boxes = model.geom_aabb[self.object_ids]
        self.box_centers = d.geom_xpos[self.object_ids] + np.einsum('ijk,ik->ij', rotations, boxes[:, :3])
        self.box_halves = np.einsum('ijk,ik->ij', abs(rotations), boxes[:, 3:])
        def mesh_vertices(geom):
            mesh = model.geom_dataid[geom]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            return model.mesh_vert[first:first + count].astype(float)
        palm_vertices = mesh_vertices(self.palm_geom)
        self.palm_hull = palm_vertices[ConvexHull(palm_vertices).vertices]
        points = np.concatenate([d.geom_xpos[g] + mesh_vertices(g) @ r.T
                                 for g, r in zip(self.object_ids, rotations)])
        self.bottle_hull = points[ConvexHull(points).vertices]

    def palm_gap_bound(self, data):
        """Diagnostic separating-plane lower bound. Contact uses native collision pairs."""
        g = self.palm_geom
        outward = -data.xmat[self.palm].reshape(3, 3) @ self.inward_in_palm
        local = data.geom_xmat[g].reshape(3, 3).T @ outward
        near = (self.palm_hull @ local).min() + data.geom_xpos[g] @ outward
        return max(0., near - (self.bottle_hull @ outward).max())

    def posture(self, data):
        delta = data.qpos[self.q] - self.reference_q
        axes = np.einsum('ijk,ik->ij', data.xmat[self.tip_bodies].reshape(4, 3, 3), self.head_axes)
        inward = data.xmat[self.palm].reshape(3, 3) @ self.inward_in_palm
        cosine = np.min(axes @ inward)
        metrics = np.array([abs(delta).max(), np.sqrt(delta @ delta / 16),
                            np.rad2deg(np.arccos(np.clip(cosine, -1., 1.))), self.palm_gap_bound(data)])
        legal = (metrics[0] <= MAX_JOINT_DEVIATION and metrics[1] <= MAX_POSTURE_RMS
                 and metrics[2] <= MAX_HEAD_ANGLE)
        return metrics, bool(legal)

    def fraction(self, data, finger, point):
        body = self.tip_bodies[finger]
        local = data.xmat[body].reshape(3, 3).T @ (point - data.xpos[body])
        return float((local - self.head_origins[finger]) @ self.head_axes[finger]
                     / self.head_spans[finger])

    def measure(self, data):
        forces = np.zeros(4)
        loads = np.zeros(5)
        points, normals = np.zeros((4, 3)), np.zeros((4, 3))
        fractions = np.full(4, np.inf)
        incidental = set()
        wrench = np.zeros(6)
        for i, c in enumerate(data.contact):
            a, b = int(c.geom1), int(c.geom2)
            if c.dist > 0 or (a in self.objects) == (b in self.objects):
                continue
            other = b if a in self.objects else a
            mujoco.mj_contactForce(self.model, data, i, wrench)
            force = max(float(wrench[0]), 0.)
            loads[self.load_region[other]] += force
            f = self.pad_index.get(other)
            if f is None:
                incidental.add(self.model.geom(other).name)
                continue
            fraction = self.fraction(data, f, c.pos)
            if self.min_head_fraction is not None and fraction < self.min_head_fraction:
                incidental.add(FINGERS[f] + '_proximal_pad')
                continue
            fractions[f] = min(fractions[f], fraction)
            forces[f] += force
            points[f] += force * c.pos
            normals[f] += (force if a in self.objects else -force) * c.frame[:3]
        loaded = forces > 0
        points[loaded] /= forces[loaded, None]
        normals[loaded] /= np.maximum(np.linalg.norm(normals[loaded], axis=1)[:, None], 1e-12)
        metrics, legal = self.posture(data)
        return Touch(forces, loads, points, normals, fractions, incidental, metrics, legal)

    def initialize(self, position, orientation, joints):
        data = mujoco.MjData(self.model)
        data.qpos[:7] = np.r_[position, quaternion(orientation)]
        data.mocap_pos[self.driver] = position
        data.mocap_quat[self.driver] = data.qpos[3:7]
        data.qpos[self.q] = data.ctrl[:] = joints
        mujoco.mj_forward(self.model, data)
        return data
