"""Whole physical fingertip contact, excluding the distal joint connector.

Uses the two previously accepted ordinary/thumb mesh splits. This is an
evaluation metric, independent of legacy strict contact and training rewards.
Source STL coordinates, distal-body coordinates and tip-site coordinates are
different frames; the transforms below explicitly account for all three.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from batched_fullhand_mcc import _quat_matrix
from surface_mcc_finger import HAND_XML, _fixed_hand_model
from torch_hand_fk import TorchHandFK

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'mcc_finger_compliance_control/assets/true_fingertip_contact_v1'
GEOMS = ('fingertip_geom', 'fingertip_2_geom', 'fingertip_3_geom', 'thumb_fingertip_geom')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrueFingertipContact:
    """Vectorized physical fingertip envelope with a strict split gate.

    The 1 mm surface tolerance accommodates solver contact midpoints and mesh
    rounding. It NEVER extends the accepted region across the split plane.
    Artificial cut faces and far-away points are not accepted. Interior solver
    midpoints are allowed: compliant contact can penetrate the fingertip, so
    imposing a thin surface shell would incorrectly reject real contacts.
    No orientation or front-pad x restriction: every exterior fingertip side
    is eligible. This is the accepted physical convex pad envelope, not an
    assertion that the unsimplified visual STL is the collision surface.
    """
    def __init__(self, device='cpu', *, surface_tolerance_m=.001,
                 split_epsilon_m=1e-6, fk=None):
        if not 0 <= surface_tolerance_m <= .003 or not 0 <= split_epsilon_m <= .001:
            raise ValueError('invalid geometric tolerance')
        self.device = torch.device(device)
        self.tolerance = float(surface_tolerance_m)
        self.split_epsilon = float(split_epsilon_m)
        self.fk = (fk or TorchHandFK()).to(self.device).eval()
        manifest = json.loads((ASSETS/'asset_manifest.json').read_text())
        for name, sha in manifest['files'].items():
            if _sha(ASSETS/name) != sha:
                raise ValueError(f'accepted split asset changed: {name}')
        self.parts = []
        self.split_manifests = []
        for kind in ('fingertip_two_part', 'thumb_fingertip_two_part'):
            folder = ASSETS/kind
            split = json.loads((folder/'split_manifest.json').read_text())
            if not split['accepted_interactively'] or _sha(ROOT/split['source']) != split['source_sha256']:
                raise ValueError('split/source STL provenance mismatch')
            normal = np.asarray(split['plane_normal_source_mesh'], dtype=np.float64)
            normal /= np.linalg.norm(normal)
            sign = 1 if split['pad_side'] == 'positive' else -1
            part = dict(normal=torch.tensor(sign*normal, dtype=torch.float32, device=self.device),
                        offset=sign*split['plane_offset_m'])
            for role in ('pad', 'connector'):
                mesh = trimesh.load(folder/f'{role}_convex.obj', force='mesh', process=False)
                equations = ConvexHull(np.asarray(mesh.vertices)).equations
                # Duplicated coplanar triangle facets are unnecessary here.
                equations = np.unique(np.round(equations, 10), axis=0)
                lengths = np.linalg.norm(equations[:, :3], axis=1)
                equations = equations/lengths[:, None]
                centers = -equations[:, 3, None]*equations[:, :3]
                cap = ((np.abs(equations[:, :3] @ normal) > 1-1e-6)
                       & (np.abs(centers @ normal-split['plane_offset_m']) < 1e-7))
                if not cap.any() or cap.all():
                    raise ValueError('cannot identify artificial split cap')
                part[role] = (torch.tensor(equations, dtype=torch.float32, device=self.device),
                              torch.tensor(~cap, device=self.device))
            self.parts.append(part)
            self.split_manifests.append(split)
        spec = mujoco.MjSpec.from_file(str(HAND_XML))
        model = _fixed_hand_model()
        positions, rotations = [], []
        for name in GEOMS:
            geom = spec.geom(name)
            rotation = Rotation.from_quat(np.asarray(geom.quat)[[1, 2, 3, 0]]).as_matrix()
            mesh = spec.mesh(geom.meshname)
            if not np.allclose(mesh.scale, 1):
                raise ValueError('source STL scale changed')
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            mid = int(model.geom_dataid[gid])
            mesh_rotation = Rotation.from_quat(model.mesh_quat[mid][[1, 2, 3, 0]]).as_matrix()
            compiled_rotation = Rotation.from_quat(model.geom_quat[gid][[1, 2, 3, 0]]).as_matrix()
            # MuJoCo recenters and rotates mesh vertices during compilation.
            if not (np.allclose(model.geom_pos[gid], geom.pos+rotation @ model.mesh_pos[mid], atol=1e-7)
                    and np.allclose(compiled_rotation, rotation @ mesh_rotation, atol=1e-7)):
                raise ValueError('source mesh versus compiled geometry transform mismatch')
            positions.append(np.asarray(geom.pos).copy())
            rotations.append(rotation)
        self.geom_position = torch.tensor(np.asarray(positions), dtype=torch.float32, device=self.device)
        self.geom_rotation = torch.tensor(np.asarray(rotations), dtype=torch.float32, device=self.device)

    def definition(self):
        return dict(schema='whole_true_fingertip_contact_v1',
            region='accepted ordinary/thumb physical fingertip convex envelope; connector and artificial cut face excluded; all exterior sides allowed',
            force='at least 0.05 N; per-slot force in live multi-slot evaluation',
            envelope_tolerance_m=self.tolerance, split_epsilon_m=self.split_epsilon,
            tolerance_rule='outside-envelope tolerance never expands across fingertip/connector split; interior solver midpoints remain eligible',
            assets_sha256=_sha(ASSETS/'asset_manifest.json'),
            implementation_sha256=_sha(__file__),
            split_planes=[dict(normal=s['plane_normal_source_mesh'], offset_m=s['plane_offset_m'],
                               side=s['pad_side']) for s in self.split_manifests])

    def source_from_tip_local(self, local):
        """[..., four fingers, xyz], in the original tip SITE frame."""
        body = torch.einsum('fij,...fj->...fi', self.fk.site_rotations, local)+self.fk.site_positions
        return torch.einsum('fji,...fj->...fi', self.geom_rotation, body-self.geom_position)

    @torch.no_grad()
    def source_from_palm(self, q, points):
        """[batch, four fingers, (optional slots), xyz], palm coordinates."""
        _, _, origins, rotations, _ = self.fk.contact_kinematics(q)
        slots = points.ndim == 4
        if slots:
            body = torch.einsum('bfji,bfsj->bfsi', rotations, points-origins[:, :, None])
            return torch.einsum('fji,bfsj->bfsi', self.geom_rotation, body-self.geom_position[:, None])
        body = torch.einsum('bfji,bfj->bfi', rotations, points-origins)
        return torch.einsum('fji,bfj->bfi', self.geom_rotation, body-self.geom_position)

    @torch.no_grad()
    def classify_source(self, points, found, force_n):
        """Source STL points [batch, four, (slots), xyz]; no force averaging."""
        if points.shape[1] != 4 or points.shape[:-1] != found.shape or found.shape != force_n.shape:
            raise ValueError('expected matching [batch, four, optional slots] measurements')
        finite = torch.isfinite(points).all(-1) & torch.isfinite(force_n)
        loaded = found.bool() & finite & (force_n >= .05)
        safe = torch.nan_to_num(points)
        tip_masks, connector_masks, signed_values = [], [], []
        for finger in range(4):
            part = self.parts[1 if finger == 3 else 0]
            value = safe[:, finger]
            signed = (value*part['normal']).sum(-1)-part['offset']
            def envelope(role):
                equations, _ = part[role]
                distance = value @ equations[:, :3].T+equations[:, 3]
                return distance.amax(-1) <= self.tolerance
            tip_masks.append((signed > self.split_epsilon) & envelope('pad'))
            connector_masks.append((signed < -self.split_epsilon) & envelope('connector'))
            signed_values.append(signed)
        tip = loaded & torch.stack(tip_masks, 1)
        connector = loaded & torch.stack(connector_masks, 1)
        return dict(true_fingertip=tip, connector=connector,
                    unmapped_or_split=loaded & ~tip & ~connector,
                    loaded=loaded, signed_tip_distance_m=torch.stack(signed_values, 1))


class ContactSlotObserver:
    """Split collision geoms; independent pad/connector evaluation sensors.

    Legacy controller sensors still aggregate the complete distal body. This
    isolates the collision change from changing pretrained-policy inputs.
    """
    def __init__(self, capacity=1, controller_scope='body', capture_collision=False):
        self.capacity = 1
        self.latest = None
        self.names = []
        self.pad_names = []
        self.safety_peak_force_n = None
        self.controller_scope = controller_scope
        self.capture_collision = capture_collision
        self.collision_names = []

    def configure(self, cfg):
        from dataclasses import replace
        from split_fingertip_collision import configure_split_collision, GEOMS, SITES
        configure_split_collision(cfg, self.controller_scope)
        for site, geom in zip(SITES, GEOMS):
            source = next(s for s in cfg.scene.sensors if s.name == f'{site}_contact')
            pad = replace(source, name=f'{site}_true_tip_contact',
                primary=replace(source.primary, mode='geom', pattern=f'^{geom}_pad_collision$'),
                fields=('found', 'force', 'pos', 'normal'), reduce='netforce', num_slots=1)
            self.pad_names.append(pad.name)
            cfg.scene.sensors = (*cfg.scene.sensors, pad)
            sensor = replace(source, name=f'{site}_connector_contact',
                primary=replace(source.primary, mode='geom', pattern=f'^{geom}_connector_collision$'),
                fields=('found', 'force', 'pos', 'normal'), reduce='netforce', num_slots=1)
            self.names.append(sensor.name)
            cfg.scene.sensors = (*cfg.scene.sensors, sensor)
            if self.capture_collision:
                collision=replace(pad,name=pad.name+'_physical_normal',reduce='maxforce',
                    fields=('found','force','pos','normal','tangent'),global_frame=False,num_slots=1)
                cfg.scene.sensors=(*cfg.scene.sensors,collision)
                self.collision_names.append(collision.name)

    def read(self, env, control_entries):
        sensors = [env.scene[name] for name in self.names]
        pad_sensors = [env.scene[name] for name in self.pad_names]
        for sensor in (*sensors, *pad_sensors):
            sensor.update(0.0)
        connector = [sensor.data for sensor in sensors]
        pads = [sensor.data for sensor in pad_sensors]
        pad_force = torch.stack([x.force.sum(1) for x in pads], 1)
        connector_force = torch.stack([x.force.sum(1) for x in connector], 1)
        self.latest = dict(
            pad_found=torch.stack([(x.found > 0).any(1) for x in pads], 1),
            pad_force=pad_force,
            pad_points=torch.stack([x.pos[:, 0] for x in pads], 1),
            pad_inward=torch.stack([x.normal[:, 0] for x in pads], 1),
            connector_found=torch.stack([(x.found > 0).any(1) for x in connector], 1),
            connector_points=torch.stack([x.pos[:, 0] for x in connector], 1),
            connector_inward=torch.stack([x.normal[:, 0] for x in connector], 1),
            connector_force=connector_force)
        if self.capture_collision:
            collisions=[]
            for name in self.collision_names:
                sensor=env.scene[name];sensor.update(0.0);collisions.append(sensor.data)
            self.latest.update(
                physical_normal_inward=torch.stack([x.normal[:,0] for x in collisions],1),
                physical_normal_found=torch.stack([x.found[:,0]>0 for x in collisions],1),
                physical_contact_point=torch.stack([x.pos[:,0] for x in collisions],1),
                physical_normal_force=torch.stack([x.force[:,0,0].clamp_min(0) for x in collisions],1))
        self.safety_peak_force_n = torch.maximum(
            torch.maximum(pad_force.norm(dim=-1), connector_force.norm(dim=-1)),
            (pad_force+connector_force).norm(dim=-1)).amax(-1)

    def definition(self):
        return dict(schema='split_collision_whole_fingertip_contact_v2',
            collision='accepted pad_convex.obj and connector_convex.obj are separate colliding geoms; original merged mesh has contype=conaffinity=0',
            region='all contacts matched to pad_collision geom; connector_collision never supplies fingertip contact or force',
            force='per-finger pad-only 3D net force magnitude >=0.05N; same aggregation and threshold, connector force excluded',
            position='finite pad contact position; no front-x/radius/separate-envelope restriction',
            sensor_reduce='netforce', measurement='geometry-ID membership, not point inference on merged geometry',
            compliance='original B/221 material/servo parameters retained for both parts',
            safety='100N on pad, connector, and their total net-force magnitude',
            implementation_sha256=_sha(__file__),
            assets_sha256=_sha(ASSETS/'asset_manifest.json'))


def with_true_fingertip_metric(parent, slot_observer=None):
    """Whole-tip coverage uses actual split collision geom IDs, with same load."""
    if slot_observer is None:
        raise ValueError('split-collision contact observer is required')
    class MeasuredResidual(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.evaluate:
                raise ValueError('new collision evaluation does not modify training')
            self.tip_count = torch.zeros(self.chain.batch, 4, device=self.chain.device)
            self.tip_all_count = torch.zeros(self.chain.batch, device=self.chain.device)
            self.connector_count = torch.zeros_like(self.tip_count)
            self.subset_violations = 0
            self.pad_strict_count = torch.zeros_like(self.tip_count)
            self.pad_strict_all_count = torch.zeros_like(self.tip_all_count)

        def observations(self, *args):
            result = super().observations(*args)
            ctx = self.context
            if ctx['frame'] < 76:
                return result
            data = slot_observer.latest
            if data is None:
                raise ValueError('split collision sensors were not refreshed')
            tip = (data['pad_found'] & torch.isfinite(data['pad_points']).all(-1)
                   & torch.isfinite(data['pad_force']).all(-1)
                   & (data['pad_force'].norm(dim=-1) >= .05))
            connector = (data['connector_found'] & torch.isfinite(data['connector_force']).all(-1)
                         & (data['connector_force'].norm(dim=-1) >= .05))
            q = args[0]
            # Verify pure range expansion on the SAME pad-only observation.
            pad_geometry = self.geom.extract(q, ctx['palm_pose'], data['pad_points'],
                torch.zeros_like(data['pad_points']), data['pad_found'], data['pad_force'].norm(dim=-1))
            pad_strict = pad_geometry['strict']
            alive = ctx['active'] | ctx['newly_stopped']
            self.subset_violations += int((pad_strict & ~tip & alive[:, None]).sum())
            if self.subset_violations:
                raise AssertionError('whole-tip contact must include strict contact in the same split-geometry run')
            self.tip_count += (tip & alive[:, None]).float()
            self.tip_all_count += (tip.all(-1) & alive).float()
            self.connector_count += (connector & alive[:, None]).float()
            self.pad_strict_count += (pad_strict & alive[:, None]).float()
            self.pad_strict_all_count += (pad_strict.all(-1) & alive).float()
            return result

        def summary(self):
            result = super().summary()
            denominator = max(1, self.frames-76)
            result.update(true_fingertip_four_rate=(self.tip_all_count/denominator).cpu().tolist(),
                true_fingertip_per_finger=(self.tip_count/denominator).cpu().tolist(),
                connector_per_finger=(self.connector_count/denominator).cpu().tolist(),
                unmapped_contact_per_finger=torch.zeros_like(self.tip_count).cpu().tolist(),
                raw_slot_overflow_frames=torch.zeros_like(self.tip_count).cpu().tolist(),
                strict_not_in_whole_tip_samples=self.subset_violations,
                pad_only_strict_four_rate=(self.pad_strict_all_count/denominator).cpu().tolist(),
                pad_only_strict_per_finger=(self.pad_strict_count/denominator).cpu().tolist(),
                collision_revision='accepted_two_part_hulls_v2')
            result['contact_metric_definitions']['true_fingertip'] = slot_observer.definition()
            result['contact_metric_definitions']['found'] = 'legacy distal-body found; pad or connector contacts'
            result['contact_metric_definitions']['strict'] = 'legacy x/radius/force rules on distal-body force-weighted contact centroid; includes connector contribution'
            return result
    return MeasuredResidual
