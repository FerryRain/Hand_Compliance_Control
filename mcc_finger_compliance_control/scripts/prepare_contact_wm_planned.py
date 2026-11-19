"""Bridge WM object assets and command paths to the existing healthy planner.

No raw rollout action is labelled as a teacher. Commands provide the object
path and initial seed only; the legacy continuous planner must solve and
certify all new hand references before physical collection is allowed.
"""
import argparse
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import time

import h5py,numpy as np
from scipy.spatial.transform import Rotation
from object_catalog import ObjectConfig,ObjectGeomConfig
from healthy_grasp_planner import HealthConfig,state_report
from replan_healthy_reference import TemporalHealthyPlanner,plan_continuous
from plan_waypoint_viewer import PlanReference


def wm_object_config(root,row,output):
    folder=Path(root)/'objects'/row['object_id'];links=output/'collision_assets'
    links.mkdir(parents=True,exist_ok=True)
    files=row['collision_parts'] if row['backend']=='convex_parts' else ['surface.obj']
    hashes={}
    for i,name in enumerate(files):
        source=(folder/name).resolve();link=links/f'collision_part_{i:04d}.obj'
        if link.exists():
            if link.resolve()!=source:raise ValueError('Conflicting planner mesh asset')
        else:link.symlink_to(source)
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    geom=ObjectGeomConfig(geom_type='mesh',size=(1.,1.,1.),pos=(0.,0.,0.),quat=(1.,0.,0.,0.),
        rgba=(.45,.7,.9,1.),mass_fraction=1.,file=str((folder/'surface.obj').resolve()),
        collision_dir=str(links.resolve()))
    config=ObjectConfig(object_id=row['object_id'],display_name=row['object_id'],family=row['family'],
        body_name='target_ball',mocap=True,total_mass_kg=.2,initial_pos=(0.,0.,0.),
        initial_rot=(1.,0.,0.,0.),contact=dict(solref=[-20000.,-400.],solimp=[.9,.98,.002,.5,2.],
        friction=[1.,.005,.0001],condim=3),collection={},motion={},geoms=(geom,),resolved=row)
    (output/'object_assets.json').write_text(json.dumps(dict(object=row,collision_hashes=hashes,
        original_assets_retained=True,planner_config=asdict(config)),indent=2)+'\n')
    return config


def object_relative_palm(palm,object_pose):
    rp=Rotation.from_quat(palm[..., [4,5,6,3]])
    ro=Rotation.from_quat(object_pose[..., [4,5,6,3]])
    return np.c_[ro.inv().apply(palm[...,:3]-object_pose[...,:3]),
                 (ro.inv()*rp).as_quat()[:,[3,0,1,2]]]


class WMTemporalHealthyPlanner(TemporalHealthyPlanner):
    def calibrate(self):
        """Keep physical/local-health gates; do not impose Mustard q quantiles."""
        c=self.config;lo=np.asarray(self.base.lower);hi=np.asarray(self.base.upper)
        margin=np.maximum(c.physical_joint_margin_rad,c.physical_joint_margin_fraction*(hi-lo))
        self.lower=lo+margin+c.healthy_bound_margin_rad
        self.upper=hi-margin-c.healthy_bound_margin_rad
        self.lower[[3,7,11,15]]=np.maximum(self.lower[[3,7,11,15]],0.)
        self.health_floor=np.tile([[.015],[.15],[.002]],(1,4))
        values=[]
        for q,pose in zip(self.bank_q,self.bank_pose):
            tip=self.base.tip_positions_palm(q)
            health=np.array([self.base._finger_kinematic_health(f,tip[f],self.maps[f]) for f in range(4)])
            jac=self.witness_jacobians(self.geometry.evaluate(q,pose))
            values.append(np.array([health[:,0],health[:,1],np.linalg.svd(jac,compute_uv=False)[:,-1]]))
        self.health_target=np.maximum(np.median(values,axis=0),self.health_floor)
        self.calibration=dict(config=asdict(c),healthy_lower_rad=self.lower.tolist(),
            healthy_upper_rad=self.upper.tolist(),health_floor=self.health_floor.tolist(),
            health_target=self.health_target.tolist(),health_order=['eta','mu','sigma_min_m_per_rad'],
            calibration_bank_states=len(self.bank_q),scope='Physical limits and existing object-agnostic health floors; no Mustard joint-envelope quantiles',
            seeds_only=True,kinematic_only=True,physical_collection_passed=False)


def prepare(folder,lane,root,output):
    with h5py.File(folder/'rollout.h5') as h:
        meta=json.loads(h.attrs['metadata']);q=h['raw_100hz/pre_mcc_q'][1,lane].astype(float)
        palm=h['raw_100hz/palm_pose'][0,lane].astype(float)
        origin=np.array(meta['env_origins'][lane]);palm[:3]-=origin
    with np.load(folder/'object_path_plan.npz') as z:poses=z['commanded'][:,lane].copy()
    poses[:,:3]-=origin
    relative=object_relative_palm(np.tile(palm,(len(poses),1)),poses)
    row=meta['object_geometry'];config=wm_object_config(root,row,output)
    source=output/'candidate_commands.h5'
    with h5py.File(source,'x') as h:
        h['palm_pose_object']=relative[:,None,:]
        h['q_hand']=np.repeat(q[None,None,:],len(relative),axis=0)
        h['object_command_pose_world']=poses
        h.attrs.update(object_id=row['object_id'],object_scale=1.,control_dt=.01,
            healthy_reference_admissible=False,collection_admissible=False,
            finger_contact_plan=False,q_hand_semantics='initialization seed only, not planned or measured teacher labels',
            source_folder=str(folder.resolve()),source_lane=lane)
    bank=output/'initial_seed_bank.npz'
    np.savez(bank,q_hand=q[None,:],palm_pose_object=relative[:1])
    return source,bank,config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--lane',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError('Preserve previous planner evidence')
    a.output.mkdir(parents=True);started=time.monotonic()
    source,bank,config=prepare(a.folder,a.lane,a.root,a.output)
    if a.prepare_only:return
    a.input=source;a.max_reference_velocity_rad_s=2.;a.continue_after_planning_failure=False
    reference=PlanReference.load(source)
    planner=WMTemporalHealthyPlanner(reference.object_id,1.,bank,object_config=config)
    planner.calibration.update(bank_path=str(bank),bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
        non_tip_geom_names=list(planner.geometry.non_tip_names))
    (a.output/'protocol.json').write_text(json.dumps(dict(source=str(source),frames=len(reference.poses),
        maximum_reference_velocity_rad_s=2.,calibration=planner.calibration,
        no_shortened_reference_on_failure=True,full_interpolated_reference_check_required=True,
        physical_admission=False,source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
        for n in ['prepare_contact_wm_planned.py','healthy_grasp_planner.py','replan_healthy_reference.py','true_tip_waypoint_optimizer.py']}),indent=2)+'\n')
    plan_continuous(a,reference,planner,started)


if __name__=='__main__':main()
