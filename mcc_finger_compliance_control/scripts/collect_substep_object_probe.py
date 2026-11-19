"""Diagnostic: same 100Hz reference endpoints, continuous 1ms object motion."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def substep_poses(position0, quaternion0, position1, quaternion1, count):
    """Sample (0,1] for each solver step; scalar-first unit quaternions."""
    if count<1:raise ValueError('count must be positive')
    phases=np.arange(1,count+1,dtype=float)/count
    rotations=Rotation.from_quat(np.stack([quaternion0,quaternion1])[:,[1,2,3,0]])
    quaternions=Slerp([0.,1.],rotations)(phases).as_quat()[:,[3,0,1,2]]
    positions=(1-phases[:,None])*np.asarray(position0)+phases[:,None]*np.asarray(position1)
    return np.concatenate([positions,quaternions],axis=-1)


def main():
    from mjlab.envs import ManagerBasedRlEnv
    import torch
    import warp as wp
    import collect_long_oracle_probe as probe
    output=Path(sys.argv[sys.argv.index('--output')+1])
    base=ManagerBasedRlEnv.step
    previous=None
    def interpolated_step(env,*args,**kwargs):
        nonlocal previous
        mid=int(env.scene['target'].data.indexing.mocap_id)
        position=env.sim.data.mocap_pos[:,mid].detach().cpu().numpy().copy()
        quaternion=env.sim.data.mocap_quat[:,mid].detach().cpu().numpy().copy()
        step=getattr(env,'_long_probe_steps',0)
        if previous is None or step<1200:
            previous=(position,quaternion)
            return base(env,*args,**kwargs)
        values=np.stack([substep_poses(previous[0][i],previous[1][i],position[i],quaternion[i],env.cfg.decimation)
                         for i in range(env.num_envs)],axis=1)
        poses=torch.as_tensor(values,device=env.device,dtype=torch.float32)
        physical=env.sim.step;index=0
        def substep():
            nonlocal index
            # Same Warp stream as the existing solver/sensor graph. No races
            # with the graph reading these mocap buffers.
            stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
            with torch.cuda.stream(stream):
                env.sim.data.mocap_pos[:,mid]=poses[index,:,:3]
                env.sim.data.mocap_quat[:,mid]=poses[index,:,3:]
            index+=1
            physical()
        env.sim.step=substep
        try:
            return base(env,*args,**kwargs)
        finally:
            env.sim.step=physical
            previous=(position,quaternion)
    ManagerBasedRlEnv.step=interpolated_step
    try:
        probe.main()
    finally:
        ManagerBasedRlEnv.step=base
        path=output/'summary.json'
        if path.exists():
            data=json.loads(path.read_text())
            data['object_motion_interpolation']=dict(enabled=True,control_hz=100,physics_hz=1000,
                position='linear within each control tick',rotation='shortest-arc SLERP',
                endpoints_and_duration_unchanged=True,preparation_and_settling_unchanged=True,
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                caveat='Changes intra-tick physical motion, not a force-sensor fix or speed optimization')
            path.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
