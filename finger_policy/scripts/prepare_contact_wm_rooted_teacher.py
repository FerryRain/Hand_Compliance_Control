"""Plan whole-hand teachers from static roots and independent object sweeps.

A root is only an initialization seed. All frames must pass the existing
continuous healthy-reference gate; failed plans are retained, never shortened.
"""
import argparse,hashlib,json,time
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from collect_contact_wm_prescribed import PrescribedSweep
from prepare_contact_wm_planned import wm_object_config,object_relative_palm,WMTemporalHealthyPlanner
from plan_waypoint_viewer import PlanReference
from replan_healthy_reference import plan_continuous

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['root','roots','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--object',required=True);p.add_argument('--root-index',type=int,default=0)
 p.add_argument('--frames',type=int,default=1500);p.add_argument('--rate',type=float,default=2.8)
 a=p.parse_args();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(a.output)
 candidates=[r for r in json.loads(a.roots.read_text())['roots'] if r['object_id']==a.object and r['active_fingers']==[0,1,2,3] and r['status']=='static_accepted_not_physically_verified']
 r=candidates[a.root_index];row=next(r for r in json.loads((a.root/'object_manifest.json').read_text())['objects'] if r['object_id']==a.object)
 if row['split']!='train':raise ValueError('Teacher development only on train parents')
 program=PrescribedSweep(r['seed']);values=program.evaluate(np.arange(a.frames)*.01,a.rate)
 pose=np.asarray(r['object_pose'],float);pose[2]+=.006 # restore near-contact endpoint; no virtual inward preload in planner
 rot=Rotation.from_quat(pose[[4,5,6,3]])
 poses=np.tile(pose,(a.frames,1));poses[:,:3]+=rot.apply(np.c_[values[:,:2],np.zeros(a.frames)])
 poses[:,3:]=(rot*Rotation.from_euler('z',values[:,2])).as_quat()[:,[3,0,1,2]]
 speed=float(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1).max()/.01)
 if speed>.3:raise ValueError('Published motion exceeds0.3m/s')
 relative=object_relative_palm(np.tile(r['palm_pose'],(a.frames,1)),poses);q=np.array(r['q'])
 a.output.mkdir(parents=True);source=a.output/'candidate_commands.h5';bank=a.output/'initial_seed_bank.npz'
 with h5py.File(source,'x') as h:
  h['palm_pose_object']=relative[:,None,:];h['q_hand']=np.tile(q,(a.frames,1,1));h['object_command_pose_world']=poses
  h.attrs.update(object_id=a.object,object_scale=1.,control_dt=.01,healthy_reference_admissible=False,collection_admissible=False,finger_contact_plan=False,q_hand_semantics='static seed only; not teacher labels')
 np.savez(bank,q_hand=q[None,:],palm_pose_object=relative[:1])
 config=wm_object_config(a.root,row,a.output);reference=PlanReference.load(source)
 planner=WMTemporalHealthyPlanner(a.object,1.,bank,object_config=config)
 planner.calibration.update(bank_path=str(bank),bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),non_tip_geom_names=list(planner.geometry.non_tip_names))
 a.input=source;a.max_reference_velocity_rad_s=2.;a.continue_after_planning_failure=False
 (a.output/'protocol.json').write_text(json.dumps(dict(root_source=str(a.roots.resolve()),root_source_sha256=hashlib.sha256(a.roots.read_bytes()).hexdigest(),root=r,motion=program.metadata,rate=a.rate,frames=a.frames,max_speed_m_s=speed,maximum_reference_velocity_rad_s=2.,no_shortened_reference_on_failure=True,calibration=planner.calibration,physical_admission=False),indent=2))
 plan_continuous(a,reference,planner,time.monotonic())
if __name__=='__main__':main()
