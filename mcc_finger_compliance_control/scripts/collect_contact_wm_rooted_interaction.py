"""Multi-finger WM interactions from checked static roots, not demonstrations.

Only the starting hand pose is planned. A fixed virtual grasp, independent
object motion and optional DP/exploration generate measured transition labels.
No runtime IK, contact search, or feedback-dependent object motion is used.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import collect_contact_wm as collection
import contact_wm_initialization as placement
from collect_contact_wm_interior import InteriorProbe
from collect_contact_wm_motion import MotionProbe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RootedInteractionProbe(InteriorProbe):
    def __init__(self,args,row):
        MotionProbe.__init__(self,args,row)
        self.config=replace(self.config,force_servo_max_outward_offset=args.outward_travel_m,
            thumb_max_outward_offset=args.outward_travel_m,force_servo_max_step=args.normal_step_m)
        self.mcc.config=self.config
        self.mcc.incremental_force_compliance=True;self.mcc.force_command_consistency=False
        self.mcc.normal_joint_mobility=torch.tensor([1.,0.,1.,1.],device=self.device)
        self.mcc.force_setpoint.fill_(args.force_target_n)
        self.fixed_reference_q=self.q0.clone()
        roots=[self.initial_roots[j['root_index']] for j in self.jobs]
        self.published_home=torch.tensor([r['collection_home_object_pose'] for r in roots],device=self.device,dtype=torch.float32)
        self.published_home[:,:3]+=self.origins
        m=self.env.sim.mj_model
        for root in roots:
            d=mujoco.MjData(m);placement.set_palm_static(m,d,self.driver.mocap_id)
            d.qpos[placement.hand_qpos_indices(m)]=root['q']
            pose=np.array(root['object_pose']);placement.set_object_static(m,d,self.target,pose[:3],pose[3:])
            mujoco.mj_forward(m,d)
            if any(c.dist < -1e-5 for c in d.contact[:d.ncon]):
                raise ValueError('Initial physical root penetrates before loading')

    def initialize(self):
        records=super().initialize();original=self.env.close
        def close():
            try:
                if getattr(self,'dp_source',None) is not None:self.dp_source.save(collection.episode_folder(self.args,self.row))
            finally:original()
        self.env.close=close
        return records

    def motion_initialization_check(self):
        """Contact readiness is a label, not an interaction-data filter.

        A fixed, finite loading program must finish without state surgery. No
        contact-rate exception is granted to numerical failures or partial
        initialization. The original initialized mask remains unchanged.
        """
        expected=self.args.approach_frames+self.args.home_hold_frames
        finite=all(bool(torch.isfinite(getattr(self.env.sim.data,k)).all())
                   for k in ('qpos','qvel','qacc','ctrl'))
        endpoint_error=float((self.driver_pose-self.published_home).abs().max())
        ready=(self.load_settle_frames_executed==expected and finite
               and endpoint_error<=1e-6 and self.done_reason is None)
        return dict(ready=ready,policy='finite_completed_loading_not_contact_success',
                    expected_frames=expected,actual_frames=self.load_settle_frames_executed,
                    finite=finite,endpoint_command_error=endpoint_error,
                    contact_ready_per_lane=self.initialized.cpu().tolist(),
                    reason=None if ready else 'incomplete_or_invalid_fixed_loading')

    def prepare_motion(self,meta):
        assert self.args.motion_owner=='object' and not self.args.object_surface_follow
        if float(abs(self.driver_pose-self.published_home).max())>1e-6:
            raise ValueError('Loading-to-motion command jump')
        self.object_path=torch.stack([MotionProbe.prescribed_object_motion(self,i) for i in range(self.args.frames)])
        if float((self.object_path[0]-self.published_home).abs().max())>1e-6:
            raise ValueError('First motion command does not match loading endpoint')
        speed=(self.object_path[1:,:,:3]-self.object_path[:-1,:,:3]).norm(dim=-1)/.01
        if float(speed.max())>self.args.max_object_command_speed:
            raise ValueError('Object speed exceeds protocol')
        MotionProbe.prepare_motion(self,meta)
        for job in self.jobs:job['force_target_n']=self.args.force_target_n
        meta.update(no_future_waypoint_planning=True,
            nominal_reference='frozen_DP' if self.dp_source is not None else 'fixed_virtual_grasp',
            object_path_reference='independent prepublished motion; no runtime height fitting',
            collection_purpose='WM_interactions_not_planned_whole_hand_teacher',
            interior_initialization=self.interior_init_report,
            mcc_candidate_overrides=dict(incremental_force_compliance=True,side_mobility=0.,
                force_target_n=self.args.force_target_n,outward_travel_m=self.args.outward_travel_m,
                normal_step_m=self.args.normal_step_m),
            motion_command_max_speed_m_s=float(speed.max()),training_admissible=False)
        np.savez_compressed(collection.episode_folder(self.args,self.row)/'object_path_plan.npz',
                            commanded=self.object_path.cpu().numpy())

    def reference_and_object(self,frame,extra=None):
        if extra is not None:raise ValueError('Counterfactual snapshots not validated for this entry')
        return self.fixed_reference_q.clone(),self.object_path[frame]

    def desired(self,frame,extra=None):
        return MotionProbe.desired(self,frame,extra)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());scripts=Path(__file__).resolve().parent
    for name,expected in protocol['source_hashes'].items():
        if sha(scripts/name)!=expected:raise ValueError('Source changed: '+name)
    for name,expected in protocol['input_hashes'].items():
        if sha(name)!=expected:raise ValueError('Input changed: '+name)
    for name,expected in protocol['collision_provenance'].items():
        if collection.collision_provenance()[name]!=expected:raise ValueError('Collision backend changed')
    args=SimpleNamespace(**protocol['collector'],**protocol['controller']);args.root=Path(args.root)
    row=dict(next(r for r in json.loads((args.root/'object_manifest.json').read_text())['objects'] if r['object_id']==protocol['object']))
    row.update(motion_backend=args.backend,solver_line_search=args.solver_line_search,
        solver_warmstart=not args.disable_solver_warmstart,constraint_capacity=args.constraint_capacity,
        object_motion_protocol=dict(name=protocol['id'],protocol_sha256=sha(a.protocol),
            source_hashes=protocol['source_hashes'],driver_update_s=.001,object_weld_time_constant_s=.005,
            motion_owner='object',no_contact_search=True,offline_planned=False))
    folder=collection.episode_folder(args,row)
    if folder.exists():raise FileExistsError(folder)
    collection.Probe=RootedInteractionProbe
    result=collection.run(args,row);print(json.dumps(dict(status=result['status'],folder=str(folder)),indent=2))


if __name__=='__main__':main()
