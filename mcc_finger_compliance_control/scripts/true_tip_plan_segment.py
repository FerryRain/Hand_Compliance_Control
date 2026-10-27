"""Turn a continuous feasible part of a rejected new path into a full plan."""
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator


def write_feasible_segment(source_path, output_path, indices, valid, joint_knots, controller):
    runs=[];start=None
    for i,ok in enumerate(np.r_[valid,False]):
        if ok and start is None:start=i
        if not ok and start is not None:
            if i-start>=6:runs.append((start,i-1))
            start=None
    if not runs:return False
    left,right=max(runs,key=lambda x:indices[x[1]]-indices[x[0]])
    begin,end=int(indices[left]),int(indices[right])
    with h5py.File(source_path) as source:
        poses=np.asarray(source['palm_pose_object']).reshape(-1,7)
        attrs=dict(source.attrs)
        count=len(poses);dt=float(attrs.get('control_dt',.01))
    source_time=np.arange(begin,end+1,dtype=float)
    time=np.linspace(begin,end,count)
    pose=np.zeros((count,7),dtype=np.float32)
    for j in range(3):pose[:,j]=np.interp(time,source_time,poses[begin:end+1,j])
    rotations=Slerp(source_time,Rotation.from_quat(poses[begin:end+1,3:][:,[1,2,3,0]]))(time)
    pose[:,3:]=rotations.as_quat()[:,[3,0,1,2]]
    q=PchipInterpolator(indices[left:right+1],joint_knots[left:right+1],axis=0)(time).astype(np.float32)
    twist=np.zeros((count,6),dtype=np.float32)
    twist[1:,:3]=np.diff(pose[:,:3],axis=0)/dt
    twist[1:,3:]=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt
    tip=np.zeros((count,4,7),dtype=np.float32)
    for i in range(count):
        tip[i,:,:3]=controller.tip_positions_palm(q[i]) @ rotations[i].as_matrix().T+pose[i,:3]
        local=np.asarray([controller.data.site_xmat[s].reshape(3,3) for s in controller.tip_ids])
        tip[i,:,3:]=Rotation.from_matrix(rotations[i].as_matrix()[None] @ local).as_quat()[:,[3,0,1,2]]
    with h5py.File(output_path,'w') as output:
        for k,v in attrs.items():output.attrs[k]=v
        output.attrs['feasible_segment_source_plan']=str(Path(source_path).resolve())
        output.attrs['feasible_segment_source_frame_interval']=(begin,end)
        output.attrs['feasible_segment_source_total_frames']=count
        output.attrs['feasible_segment_time_stretched']=True
        output.attrs['planner_realized_rotation_path_deg']=float(np.degrees(np.linalg.norm((rotations[1:]*rotations[:-1].inv()).as_rotvec(),axis=-1).sum()))
        output.attrs['planner_realized_translation_path_m']=float(np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=-1).sum())
        output['palm_pose_object']=pose[:,None]
        output['palm_twist_object']=twist[:,None]
        output['q_hand']=q[:,None]
        velocity=np.zeros_like(q);velocity[1:]=np.diff(q,axis=0)/dt
        output['qvel']=velocity[:,None]
        output['fingertip_pose_object']=tip[:,None]
        fixed=np.zeros((count,1,7),dtype=np.float32);fixed[...,3]=1
        output['object_pose_world']=fixed
        output['episode_id']=np.zeros((count,1),dtype=np.int32)
        output['record_step']=np.arange(count)[:,None]
    return True
