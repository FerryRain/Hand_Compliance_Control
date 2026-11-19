"""Plan continuous whole-hand references from new jointly feasible roots.

Only accepted complete paths can proceed to physical collection. Failed paths
remain recorded and do not block attempts on the other objects or roots.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from prepare_contact_wm_multi_crossing import interpolate, program
from prepare_contact_wm_planned import wm_object_config, object_relative_palm, WMTemporalHealthyPlanner
from replan_healthy_reference import plan_continuous
from plan_waypoint_viewer import PlanReference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots',type=Path,required=True)
    ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--pool',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=3000)
    a = ap.parse_args()
    root = json.loads(a.roots.read_text())['roots'][a.index]
    if root['status']!='static_accepted_not_physically_verified' or root['active_fingers']!=[0,1,2,3]:
        raise ValueError('Jointly feasible four-finger initial root required')
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    home=np.array(root['object_pose']);home[2]+=.006
    sign=1 if root['root_attempt_index']==0 else -1
    if 'tomato' in root['object_id']:
        poses=program(home,sign,a.frames)
    else:
        r0=Rotation.from_quat(home[[4,5,6,3]])
        local=np.c_[sign*interpolate([0,.025,-.020,.035,-.025],a.frames),
            interpolate([0,.015,.025,-.010,0],a.frames),np.zeros(a.frames)]
        angles=np.deg2rad(np.c_[interpolate([0,10,-10,15,0],a.frames),
            interpolate([0,-6,8,-4,0],a.frames),sign*interpolate([0,15,35,60,90],a.frames)])
        poses=np.tile(home,(a.frames,1))
        poses[:,:3]+=r0.apply(local)
        poses[:,3:]=(r0*Rotation.from_euler('xyz',angles)).as_quat()[:,[3,0,1,2]]
        poses[0]=home
    relative=object_relative_palm(np.tile(root['palm_pose'],(a.frames,1)),poses)
    row=next(r for r in json.loads((a.pool/'object_manifest.json').read_text())['objects'] if r['object_id']==root['object_id'])
    if row['split']!='train':raise ValueError('Do not plan training on held-out object')
    config=wm_object_config(a.pool,row,a.output)
    source=a.output/'candidate_commands.h5'
    with h5py.File(source,'x') as h:
        h['palm_pose_object']=relative[:,None,:]
        h['q_hand']=np.tile(root['q'],(a.frames,1,1))
        h['object_command_pose_world']=poses
        h.attrs.update(object_id=row['object_id'],object_scale=1.,control_dt=.01,
            collection_admissible=False,healthy_reference_admissible=False,
            q_hand_semantics='initial seed only, never a label',source_root_index=a.index)
    bank=a.output/'initial_seed_bank.npz'
    np.savez(bank,q_hand=np.array(root['q'])[None],palm_pose_object=relative[:1])
    started=time.monotonic()
    planner=WMTemporalHealthyPlanner(row['object_id'],1.,bank,object_config=config)
    planner.calibration.update(bank_path=str(bank),bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
        non_tip_geom_names=list(planner.geometry.non_tip_names))
    a.input=source;a.max_reference_velocity_rad_s=2.;a.continue_after_planning_failure=False
    (a.output/'protocol.json').write_text(json.dumps(dict(root=root,frames=a.frames,
        config=asdict(planner.config),maximum_reference_velocity_rad_s=2.,
        same_MCC=True,physical_admission=False,full_path_gate=True,
        object_command_max_speed_m_s=float(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=-1).max()/.01)),indent=2))
    plan_continuous(a,PlanReference.load(source),planner,started)


if __name__=='__main__':main()
