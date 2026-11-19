"""Fixed interior pregrasp + independent object motion, collection experiment.

An interior target is a virtual equilibrium, not an initial physical state inside
the object. The object approaches it on a fixed ramp while ordinary MCC unloads
the fingers. No force threshold takeover, missing-contact search or future q
trajectory is introduced. Candidate force-command interfaces are opt-in and
recorded in the protocol; deployment defaults remain unchanged.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import collect_contact_wm as collection
import contact_wm_initialization as placement
from collect_contact_wm_prescribed import PrescribedProbe


def reference_for_job(args, finger, seed):
    key = f'{finger}:{seed}'
    if key in args.interior_rows:
        return args.interior_rows[key]
    return args.interior_rows[str(finger)]


def interior_jobs(args, row):
    if getattr(args, 'active_fingers', None) is not None:
        raise ValueError('Interior pool currently contains independent single fingers')
    seeds = getattr(args, 'episode_seeds', [args.seed])
    forces = getattr(args, 'force_targets_n', [getattr(args, 'force_target_n', 10.)])
    jobs = [dict(finger=f, behavior=b, seed=seed+f*101+i, force_target_n=force)
            for force in forces for seed in seeds for f in args.fingers
            for i,b in enumerate(args.behaviors)]
    for job in jobs:
        reference_for_job(args, job['finger'], job['seed'])
    return jobs, None


class InteriorProbe(PrescribedProbe):
    def __init__(self, args, row):
        super().__init__(args, row)
        self.config = replace(self.config,
                              force_servo_max_outward_offset=args.outward_travel_m,
                              thumb_max_outward_offset=args.outward_travel_m,
                              force_servo_max_step=args.normal_step_m)
        self.mcc.config = self.config
        self.mcc.force_command_consistency = getattr(args, 'force_command_consistency', False)
        self.mcc.incremental_force_compliance = getattr(args, 'incremental_force_compliance', False)
        if getattr(args, 'force_target_n', None) is not None:
            self.mcc.force_setpoint.fill_(args.force_target_n)
        for lane,job in enumerate(self.jobs):
            if 'force_target_n' in job:
                self.mcc.force_setpoint[lane].fill_(job['force_target_n'])
        if getattr(args, 'side_mobility', None) is not None:
            if not 0 <= args.side_mobility <= 1:
                raise ValueError('Side mobility must be in [0,1]')
            self.mcc.normal_joint_mobility = torch.tensor(
                [1., args.side_mobility, 1., 1.], device=self.device)
        self.fixed_reference_q = self.q0.clone()
        homes = [reference_for_job(args,j['finger'],j['seed'])['home_pose_local'] for j in self.jobs]
        self.published_home = torch.tensor(homes, dtype=torch.float32, device=self.device)
        self.published_home[:, :3] += self.origins

    def sample(self, phase):
        result = super().sample(phase)
        if getattr(self.args, 'force_command_consistency', False) or getattr(self.args, 'incremental_force_compliance', False):
            diag = getattr(self.mcc, 'force_command_diagnostic', {})
            for name in ('normal_before', 'normal_after', 'requested_signed_step',
                         'scale', 'engaged', 'movable', 'direction', 'enabled'):
                value = diag.get(name, torch.zeros((self.n,4), device=self.device))
                result['mcc_final_' + name] = value.detach().cpu().numpy().astype(np.float32)
        return result

    def initialize(self):
        original_close = self.env.close
        def close():
            try:
                if self.trace:
                    folder = collection.episode_folder(self.args, self.row)
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder/'substeps.npz', **{
                        k: np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
                    mujoco.mj_saveModel(self.env.sim.mj_model, str(folder/'physics_model.mjb'), None)
            finally:
                original_close()
        self.env.close = close
        records = [self.sample(0)]
        start = self.object.clone()
        streak = torch.zeros(self.n, dtype=torch.long, device=self.device)
        # A quintic ramp is prepublished and independent of measured contact.
        # All q/velocity writes happened in the normal initial-state constructor.
        ramp_frames = self.args.approach_frames
        for frame in range(ramp_frames + self.args.home_hold_frames):
            u = min((frame+1)/ramp_frames, 1.)
            blend = u*u*u*(10 + u*(-15 + 6*u))
            pose = torch.lerp(start, self.published_home, blend)
            record = self.step(self.fixed_reference_q, pose, 0, use_mcc=True)
            x = self.observer.latest
            loaded = x['pad_found'] & (x['pad_force'].norm(dim=-1) >= .2)
            streak = torch.where(collection.all_active_loaded(loaded, self.active), streak+1, 0)
            self.initialized = (streak >= 15) & (frame >= ramp_frames-1)
            record['initialized'] = self.initialized.cpu().numpy().copy()
            records.append(record)
            if self.done_reason:
                break
        self.base = self.object.clone()
        self.load_settle_frames_executed = len(records)-1
        self.interior_init_report = dict(
            reference='fixed virtual interior pregrasp, never rebased to measured q',
            ramp_frames=ramp_frames, hold_frames=self.args.home_hold_frames,
            actual_frames=len(records)-1,
            maximum_approach_speed_m_s=1.875*self.args.approach_distance_m/(ramp_frames*.01),
            home_max_abs_error=float((self.object-self.published_home).abs().max()),
            initialized=self.initialized.cpu().tolist(),
            control_state_handoff='preserve MCC offset, previous command and physical state')
        collection.save_json(collection.episode_folder(self.args,self.row)/'interior_initialization.json',
                             self.interior_init_report)
        return records

    def prepare_motion(self, meta):
        super().prepare_motion(meta)
        meta.update(nominal_reference='fixed_virtual_interior_pregrasp',
                    interior_reference_file_sha256=self.args.interior_reference_sha256,
                    interior_static_reference=self.args.interior_rows,
                    interior_initialization=self.interior_init_report,
                    mcc_candidate_overrides=dict(outward_travel_m=self.args.outward_travel_m,
                                                normal_step_m=self.args.normal_step_m,
                                                force_target_n=self.mcc.force_setpoint[0].cpu().tolist(),
                                                force_target_n_per_lane=self.mcc.force_setpoint.cpu().tolist(),
                                                side_mobility=getattr(self.args,'side_mobility',None),
                                                force_command_consistency=getattr(self.args,'force_command_consistency',False),
                                                incremental_force_compliance=getattr(self.args,'incremental_force_compliance',False),
                                                offset_semantics=('diagnostic current-Jacobian projection of command minus nominal; not a control integrator'
                                                  if getattr(self.args,'incremental_force_compliance',False) else 'scalar force integrator')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    opts = parser.parse_args()
    p = json.loads(opts.protocol.read_text())
    scripts = Path(__file__).resolve().parent
    for name, expected in p['source_hashes'].items():
        if hashlib.sha256((scripts/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source differs from protocol: '+name)
    for field, expected in p['collision_provenance'].items():
        if collection.collision_provenance()[field] != expected:
            raise ValueError('Wrong collision implementation: '+field)
    reference = Path(p['interior_reference'])
    if hashlib.sha256(reference.read_bytes()).hexdigest() != p['interior_reference_sha256']:
        raise ValueError('Static reference differs from protocol')
    reference_data = json.loads(reference.read_text())
    args = SimpleNamespace(**p['collector'])
    args.root = Path(args.root)
    args.variant = 'prescribed'
    args.interior_reference_sha256 = p['interior_reference_sha256']
    args.interior_rows = {(f"{r['finger']}:{r['seed']}" if 'seed' in r else str(r['finger'])): r
                          for r in reference_data['rows']}
    for name, value in p['interior_controller'].items():
        setattr(args, name, value)
    manifest = args.root/'object_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != p['object_manifest_sha256']:
        raise ValueError('Manifest changed')
    row = dict(next(r for r in json.loads(manifest.read_text())['objects'] if r['object_id']==p['object']))
    row.update(motion_backend=args.backend, solver_line_search=args.solver_line_search,
               solver_warmstart=not args.disable_solver_warmstart,
               constraint_capacity=args.constraint_capacity,
               object_motion_protocol=dict(name=p['id'], collector_profile=p['collector'],
                   source_hashes=p['source_hashes'], fixed_interior_reference=True,
                   interior_controller=p['interior_controller'],
                   protocol_sha256=hashlib.sha256(opts.protocol.read_bytes()).hexdigest(),
                   driver_update_s=.001, object_weld_time_constant_s=.005,
                   object_gt_surface_height=False, no_finger_waypoints=True,
                   no_contact_search=True, motion_owner='object'))
    folder=collection.episode_folder(args,row)
    if folder.exists():
        raise FileExistsError('Refuse to overwrite '+str(folder))
    original_place = placement.place
    original_jobs = collection.collection_jobs
    def interior_place(model, root, obj, q, finger, palm_id, target_id, seed):
        ref = reference_for_job(args,finger,seed)
        target = np.array(ref['home_pose_local'])
        target[2] -= args.approach_distance_m
        q = np.array(ref['q'])
        data = mujoco.MjData(model)
        placement.set_palm_static(model,data,palm_id)
        data.qpos[placement.hand_qpos_indices(model)] = q
        placement.set_object_static(model,data,target_id,target[:3],target[3:])
        mujoco.mj_forward(model,data)
        if any(c.dist < -1e-5 for c in data.contact[:data.ncon]):
            raise ValueError('Interior-reference initial physical state collides: finger '+str(finger))
        rot = Rotation.from_quat(target[[4,5,6,3]]).as_matrix()
        return q, target[:3], rot, np.array([0.,0.,-1.]), np.zeros(3), dict(
            method='fixed virtual pregrasp; object starts separated and follows fixed approach ramp',
            static_reference=ref, initial_physics_penetration=False,
            initial_object_retreat_m=args.approach_distance_m)
    try:
        placement.place = interior_place
        if hasattr(args,'episode_seeds') or hasattr(args,'force_targets_n'):
            collection.collection_jobs = interior_jobs
        collection.Probe = InteriorProbe
        result = collection.run(args,row)
        print(json.dumps(dict(status=result['status'], folder=str(folder)),indent=2))
    finally:
        placement.place = original_place
        collection.collection_jobs = original_jobs


if __name__ == '__main__':
    main()
