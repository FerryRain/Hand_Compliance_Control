"""Recorded absolute actuator targets in the original arm-held collection plant.

Separates open-loop command replay fidelity from moving-palm transfer. Restores
all22 joint positions/velocities and starts at source sample1; no new teacher,
MCC, diffusion or residual actions. Uses the same1ms force monitoring as the
support comparisons, which is stricter than the old10ms collector stop check.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import mcc_finger_contact_env_cfg
from audit_long_free_palm import source_row, longest_loss
from invert_trajectories import _relative_pose
from object_catalog import load_object_config
from true_fingertip_contact import ContactSlotObserver
from collect_trajectories import install_collection_substep_safety


def run(args):
    rows = [source_row(eid) for eid in args.ids]
    source = []
    for row in rows:
        if row['status'] != 'reference_completed':
            raise ValueError('Only complete corrected collection references are allowed')
        with h5py.File(Path(row['probe'])/'collected.h5') as h:
            n = args.frames
            if len(h['q']) < n:
                raise ValueError('Do not pad a reference tail')
            source.append(dict(q=np.asarray(h['q'][:n, 0]),
                velocity=np.asarray(h['qvel'][:n, 0]),
                target=np.concatenate((h['arm_q_ref'][:n, 0], h['q_cmd'][:n, 0]), -1),
                object=np.asarray(h['object_pose_world'][:n, 0]),
                palm=_relative_pose(np.asarray(h['object_pose_world'][:n, 0]),
                                    np.asarray(h['palm_pose_world'][:n, 0]))))
    batch = len(source)
    device = 'cuda:0'
    cfg = mcc_finger_contact_env_cfg(num_envs=batch,
        object_config=load_object_config('ycb_mustard'), finger_stiffness=35.,
        finger_damping=2.5, finger_effort_limit=35., physics_substeps=10)
    observer = ContactSlotObserver(controller_scope='pad')
    observer.configure(cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    values = {key:torch.as_tensor(np.stack([r[key] for r in source], 1),
                                 device=device, dtype=torch.float32)
              for key in ('q', 'velocity', 'target', 'object')}
    n = args.frames - 1
    arrays = {key:torch.zeros((n, batch, *shape), device=device, dtype=dtype)
              for key,shape,dtype in [('q',(16,),torch.float32), ('palm_world',(7,),torch.float32),
                ('force',(),torch.float32), ('post_force',(),torch.float32),
                ('strict',(4,),torch.bool), ('found',(4,),torch.bool),
                ('seen',(),torch.bool)]}
    active = torch.ones(batch, dtype=torch.bool, device=device)
    stopped = torch.full((batch,), -1, dtype=torch.long, device=device)
    try:
        env.reset()
        robot = env.scene['robot']
        # Source files were collected as separate single-world runs. The
        # plane terrain assigns a display grid origin to a multi-world batch;
        # remove that offset from the arm mount before restoring world poses.
        root_mocap_id = int(robot.data.indexing.mocap_id)
        env.sim.data.mocap_pos[:, root_mocap_id, :] = 0.
        env.sim.data.mocap_quat[:, root_mocap_id, :] = torch.tensor(
            [1.,0.,0.,0.],device=device)
        robot.write_joint_state_to_sim(values['q'][0], values['velocity'][0])
        robot.set_joint_position_target(values['target'][0])
        robot.write_data_to_sim()
        target_id = int(env.scene['target'].data.indexing.mocap_id)
        palm_id = next(i for i,b in enumerate(robot.data.indexing.bodies)
                       if (b.name or '').endswith('/palm_lower'))
        env.sim.data.mocap_pos[:, target_id, :] = values['object'][0, :, :3]
        env.sim.data.mocap_quat[:, target_id, :] = values['object'][0, :, 3:]
        env.sim.forward()
        restored = _relative_pose(values['object'][0].cpu().numpy(),
                                 robot.data.body_link_pose_w[:,palm_id].cpu().numpy())
        expected = np.stack([r['palm'][0] for r in source])
        initial_error = np.linalg.norm(restored[:,:3]-expected[:,:3],axis=-1)
        if initial_error.max() > .0001:
            raise ValueError(f'Arm restore initial relative-position mismatch: {initial_error.tolist()}')
        frame = 0
        original_step = env.sim.step
        monitor = install_collection_substep_safety(env, observer)
        monitor['enabled'] = True
        for sample in range(1, args.frames):
            frame = sample - 1
            pose = values['object'][sample].clone()
            pose[~active, 2] += 2.
            env.sim.data.mocap_pos[:, target_id, :] = pose[:, :3]
            env.sim.data.mocap_quat[:, target_id, :] = pose[:, 3:]
            env.sim.forward()
            env.sim.sense()
            action = torch.cat((values['target'][sample, :, :6],
                               ((values['target'][sample, :, 6:]-robot.data.joint_pos[:, 6:])/.08)
                               .clamp(-2, 2)), -1)
            monitor['peak'].zero_()
            env.step(action)
            observer.read(env, None)
            x = observer.latest
            arrays['post_force'][frame] = observer.safety_peak_force_n
            arrays['force'][frame] = torch.maximum(monitor['peak'], observer.safety_peak_force_n)
            unsafe = active & (arrays['force'][frame] > 100.)
            arrays['seen'][frame] = active
            stopped = torch.where(unsafe, sample, stopped)
            active &= ~unsafe
            strict = (x['pad_found'] & torch.isfinite(x['pad_points']).all(-1)
                      & torch.isfinite(x['pad_force']).all(-1) & (x['pad_force'].norm(dim=-1)>=.05))
            arrays['strict'][frame] = strict & active[:, None]
            arrays['found'][frame] = (x['pad_found']|x['connector_found']) & active[:, None]
            arrays['q'][frame] = robot.data.joint_pos[:, 6:]
            arrays['palm_world'][frame] = robot.data.body_link_pose_w[:, palm_id]
            if frame % 1000 == 0:
                print(f'[SOURCE-REPLAY] {frame}/{n} active={int(active.sum())}/{batch}', flush=True)
            if not bool(active.any()):
                break
        z = {key:value.cpu().numpy() for key,value in arrays.items()}
        lanes=[]
        for i,eid in enumerate(args.ids):
            seen = z['seen'][:,i]
            loss,first = longest_loss(z['strict'][:,i].all(-1))
            palm = _relative_pose(source[i]['object'][1:][seen], z['palm_world'][seen,i])
            lanes.append(dict(trajectory_id=eid, status='completed' if stopped[i]<0 else 'force_stop',
                stop_task_frame=None if stopped[i]<0 else int(stopped[i]),
                evaluated_frames=n, observed_frames=int(seen.sum()),
                strict_healthy_four_rate=float(z['strict'][:,i].all(-1).mean()),
                found_four_rate=float(z['found'][:,i].all(-1).mean()),
                strict_healthy_per_finger=z['strict'][:,i].mean(0).tolist(),
                found_per_finger=z['found'][:,i].mean(0).tolist(),
                longest_four_loss_frames=loss, first_loss_over5_frame=first,
                peak_force_n=float(z['force'][seen,i].max()),
                post_force_peak_n=float(z['post_force'][seen,i].max()),
                source_q_tracking_mae_rad=float(abs(z['q'][seen,i]-source[i]['q'][1:,6:][seen]).mean()),
                palm_reference_error_max_m=float(np.linalg.norm(palm[:,:3]-source[i]['palm'][1:][seen,:3],axis=-1).max())))
        args.output.mkdir(parents=True,exist_ok=True)
        trace = args.output/f'source_arm_raw_{args.frames}.npz'
        np.savez_compressed(trace,**z,ids=args.ids)
        report=dict(ids=args.ids,lanes=lanes,trace=str(trace),frames=args.frames,
            initial_palm_reference_error_m=initial_error.tolist(),
            controller='Recorded absolute arm/finger actuator targets only; no MCC, DP or RL',
            physics='Original arm-held/moving-mocap-object collection model and materials',
            initialization='Source sample0 all22 q/qvel/actuator targets; replay sample1..end',
            safety='Identical100N1ms substep and post-step net-force gate; stopped tick and tail count as failed')
        (args.output/f'source_arm_raw_{args.frames}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(lanes,indent=2),flush=True)
    finally:
        if 'original_step' in locals():
            env.sim.step = original_step
        torch.cuda.synchronize()
        for name in ('step_graph','forward_graph','reset_graph','sense_graph'):
            setattr(env.sim,name,None)
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids',type=int,nargs='+',default=[36,39,41,43,45])
    parser.add_argument('--frames',type=int,default=6000)
    parser.add_argument('--output',type=Path,
        default=Path('mcc_finger_compliance_control/data/palm_support_alignment_20261013/source_arm'))
    args=parser.parse_args()
    if not 2<=args.frames<=6000:
        parser.error('frames must be2..6000')
    run(args)


if __name__=='__main__':
    main()
