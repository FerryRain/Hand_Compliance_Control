"""Physically collect an offline-certified whole-hand reference.

The healthy reference and a virtual inward equilibrium are distinct records.
Neither is written to physical joint state after initialization. Object motion
is the immutable planned command, not a function of measured contact or force.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import collect_contact_wm as collection
import contact_wm_initialization as placement
from collect_contact_wm_interior import InteriorProbe
from collect_contact_wm_motion import MotionProbe
from torch_hand_fk import TorchHandFK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def prepare_virtual_reference(reference, depths):
    """Offline normal displacement at each planned pad witness (CPU only).

    Normals are object-outward. Mobility excludes side swing, just as the
    frozen incremental force interface does. This is a virtual target audit,
    not a claim that an executed penetrating configuration is admissible.
    """
    with h5py.File(reference) as h:
        if not h.attrs['collection_admissible'] or not h['grasp_keyframe_valid'][:].all():
            raise ValueError('Full offline healthy-reference gate required')
        q = torch.tensor(h['q_hand'][:, 0], dtype=torch.float32)
        pose = h['object_command_pose_world'][:]
        relative = h['palm_pose_object'][:, 0]
        points = h['grasp_keyframe_contact_point_object'][:, 0]
        normals = h['grasp_keyframe_normal_object'][:, 0]
        gap = torch.tensor(h['grasp_keyframe_signed_distance'][:, 0], dtype=torch.float32)
    rp = Rotation.from_quat(relative[:, [4, 5, 6, 3]]).as_matrix()
    pp = np.einsum('tji,tfj->tfi', rp, points-relative[:, None, :3])
    nn = torch.tensor(np.einsum('tji,tfj->tfi', rp, normals), dtype=torch.float32)
    nn = nn / nn.norm(dim=-1, keepdim=True)
    fk = TorchHandFK()
    _, _, origin, rot, _ = fk.contact_kinematics(q)
    anchor = torch.einsum('tfji,tfj->tfi', rot, torch.tensor(pp, dtype=torch.float32)-origin)
    valid = torch.ones(q.shape[0], 4, dtype=torch.bool)
    initial = origin+torch.einsum('tfij,tfj->tfi', rot, anchor)
    mobility = torch.tensor([1., 0., 1., 1.])
    values, errors = [], []
    for depth in depths:
        if not 0 < depth <= .01:
            raise ValueError('Virtual depth must be in (0,10] mm')
        candidate = q.clone()
        for _ in range(6):
            _, _, origin, rot, jac = fk.contact_kinematics(candidate, anchor, valid)
            point = origin+torch.einsum('tfij,tfj->tfi', rot, anchor)
            displacement = ((point-initial)*nn).sum(-1)
            error = -depth-gap-displacement
            jn = torch.einsum('tfi,tfij->tfj', nn, jac)
            weighted = jn*mobility
            dq = weighted*(error/(jn*weighted).sum(-1).clamp_min(1e-8))[..., None]
            candidate += dq.clamp(-.05, .05).reshape(-1, 16)
        _, _, origin, rot, _ = fk.contact_kinematics(candidate, anchor, valid)
        point = origin+torch.einsum('tfij,tfj->tfi', rot, anchor)
        errors.append(float((((point-initial)*nn).sum(-1)+depth+gap).abs().max()))
        values.append(candidate.numpy())
    virtual = np.stack(values, 1)
    delta = virtual-q.numpy()[:, None]
    velocity = np.diff(virtual, axis=0)/.01
    report = dict(frames=len(q),depths_m=depths,normal_displacement_error_max_m=errors,
        max_virtual_q_delta_rad=float(abs(delta).max()),
        max_virtual_velocity_rad_s=float(abs(velocity).max()),
        max_side_delta_rad=float(abs(delta[..., [1,5,9,13]]).max()),
        healthy_reference_sha256=sha(reference),actual_state_teleport=False,
        original_q_semantics='offline planned healthy q; inherited seed-only H5 attribute is stale',
        virtual_q_semantics='pre-MCC virtual inward equilibrium, not an executed state or healthy label')
    if max(errors) > 1e-5 or abs(delta).max() > .35 or abs(velocity).max() > 3.:
        raise ValueError('Virtual reference fails frozen continuity/displacement gate: '+str(report))
    return dict(healthy_q=q.numpy(), virtual_q=virtual, object_command_pose=pose), report


class PlannedProbe(InteriorProbe):
    def __init__(self, args, row):
        # Reuse the physical plant and measurement pipeline, not the old
        # single-finger path or its post-initialization height compensation.
        MotionProbe.__init__(self, args, row)
        self.config = replace(self.config, force_servo_max_outward_offset=args.outward_travel_m,
            thumb_max_outward_offset=args.outward_travel_m, force_servo_max_step=args.normal_step_m)
        self.mcc.config = self.config
        self.mcc.incremental_force_compliance = True
        self.mcc.force_command_consistency = False
        self.mcc.normal_joint_mobility = torch.tensor([1.,0.,1.,1.],device=self.device)
        self.mcc.force_setpoint.fill_(args.force_target_n)
        with np.load(args.virtual_reference) as z:
            self.healthy = torch.tensor(z['healthy_q'],device=self.device,dtype=torch.float32)
            self.virtual = torch.tensor(z['virtual_q'],device=self.device,dtype=torch.float32)
            path = torch.tensor(z['object_command_pose'],device=self.device,dtype=torch.float32)
        if self.virtual.shape != (args.frames,self.n,16):
            raise ValueError('Reference/lane count mismatch')
        if not torch.equal(self.virtual, torch.maximum(torch.minimum(self.virtual,self.mcc.upper),self.mcc.lower)):
            raise ValueError('Virtual target exceeds physical joint limits')
        self.object_path = path[:,None,:].repeat(1,self.n,1)
        self.object_path[:,:,:3] += self.origins[None]
        self.published_home = self.object_path[0].clone()
        self.fixed_reference_q = self.virtual[0].clone()
        self.reference_index = -1
        # Initial physical state, including self-collision, independently checked
        # in the exact simulator model. Never start at the virtual target.
        m = self.env.sim.mj_model
        for lane,root in enumerate(self.initial_roots):
            d = mujoco.MjData(m)
            placement.set_palm_static(m,d,self.driver.mocap_id)
            d.qpos[placement.hand_qpos_indices(m)] = root['q']
            p = np.array(root['object_pose'])
            placement.set_object_static(m,d,self.target,p[:3],p[3:])
            if self.free_joint >= 0:
                d.qpos[self.free_q:self.free_q+7] = p
            mujoco.mj_forward(m,d)
            if any(c.dist < -1e-5 for c in d.contact[:d.ncon]):
                raise ValueError('Initial physical root collides, lane '+str(lane))

    def sample(self, phase):
        result = super().sample(phase)
        if hasattr(self,'healthy'):
            i = max(self.reference_index,0)
            result['planned_healthy_q'] = self.healthy[i].expand(self.n,-1).cpu().numpy().copy()
            result['planned_reference_index'] = np.full(self.n,self.reference_index,dtype=np.int32)
        return result

    def prepare_motion(self, meta):
        error = float((self.driver_pose-self.published_home).abs().max())
        if error > 1e-6:
            raise ValueError('Object command handoff discontinuity')
        speed = (self.object_path[1:,:,:3]-self.object_path[:-1,:,:3]).norm(dim=-1)/.01
        if float(speed.max()) > self.args.max_object_command_speed:
            raise ValueError('Planned command speed exceeds frozen bound')
        for lane,job in enumerate(self.jobs):
            job.update(virtual_depth_m=self.args.virtual_depths_m[lane],
                       force_target_n=self.args.force_target_n)
        meta.update(no_future_waypoint_planning=False,
            nominal_reference='offline healthy reference plus frozen virtual inward equilibrium',
            nominal_command_exploration_rad=[0.,0.],
            action_owner='offline healthy q -> virtual inward q -> incremental MCC -> held servo',
            planned_reference=self.args.reference_provenance,
            labels='q/qvel are physical observations; planned_healthy_q is geometric intent; pre_mcc_q is virtual command',
            interior_initialization=self.interior_init_report,
            mcc_candidate_overrides=dict(incremental_force_compliance=True,side_mobility=0.,
                force_target_n=self.args.force_target_n,outward_travel_m=self.args.outward_travel_m,
                normal_step_m=self.args.normal_step_m,
                offset_semantics='current-Jacobian command projection, not integrator'),
            motion_command_max_speed_m_s=float(speed.max()),
            motion_handoff_command_error=error,training_admissible=False)
        np.savez_compressed(collection.episode_folder(self.args,self.row)/'object_path_plan.npz',
                            commanded=self.object_path.cpu().numpy())

    def desired(self, frame, extra=None):
        if extra is not None:
            raise ValueError('Branches require separate full-state validation')
        self.reference_index = frame
        return self.virtual[frame],self.object_path[frame]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    a = p.parse_args(); protocol=json.loads(a.protocol.read_text())
    scripts=Path(__file__).resolve().parent
    for name,expected in protocol['source_hashes'].items():
        if sha(scripts/name)!=expected:raise ValueError('Source changed: '+name)
    for name,expected in protocol['input_hashes'].items():
        if sha(name)!=expected:raise ValueError('Input changed: '+name)
    for name,expected in protocol['collision_provenance'].items():
        if collection.collision_provenance()[name]!=expected:raise ValueError('Backend changed: '+name)
    args=SimpleNamespace(**protocol['collector'],**protocol['controller'])
    args.root=Path(args.root)
    args.reference_provenance=protocol['reference_provenance']
    row=dict(next(r for r in json.loads((args.root/'object_manifest.json').read_text())['objects']
                  if r['object_id']==protocol['object']))
    row.update(motion_backend=args.backend,solver_line_search=args.solver_line_search,
        solver_warmstart=not args.disable_solver_warmstart,constraint_capacity=args.constraint_capacity,
        object_motion_protocol=dict(name=protocol['id'],source_hashes=protocol['source_hashes'],
            protocol_sha256=sha(a.protocol),driver_update_s=.001,object_weld_time_constant_s=.005,
            no_contact_search=True,motion_owner='object',offline_planned=True))
    folder=collection.episode_folder(args,row)
    if folder.exists():raise FileExistsError('Preserve previous evidence: '+str(folder))
    old=collection.Probe
    try:
        collection.Probe=PlannedProbe
        result=collection.run(args,row)
        print(json.dumps(dict(status=result['status'],folder=str(folder)),indent=2))
    finally:collection.Probe=old


if __name__=='__main__':main()
