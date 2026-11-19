"""Recorded actuator-command replay with finite or fixed palm support.

No DP or second MCC pass: q_cmd already includes the collection controller.
Source sample 0 initializes measured q/qvel; transition k uses q_cmd[k] and
is compared to measured q[k]. Sources are never modified. No state resets
after initialization. All >100N events are recorded, not truncated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'mcc_finger_compliance_control/data'
SPLIT = DATA/'dp_stratified_222_20261022/episode_split.json'


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(pose):
    pose = np.asarray(pose)
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(pose[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    result[..., :3, 3] = pose[..., :3]
    return result


def pose(transform):
    transform = np.asarray(transform)
    quat = Rotation.from_matrix(transform[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(*transform.shape[:-2], 4)
    return np.concatenate((transform[..., :3, 3], quat[..., [3, 0, 1, 2]]), -1)


def paths(palm, obj, mode):
    """Keep the source world frame/gravity at t=0 and preserve relative pose."""
    p, o = matrix(palm), matrix(obj)
    if mode in ('weld', 'legacy'):
        hand = o[0] @ np.linalg.inv(o) @ p
        target = np.repeat(o[:1], len(o), axis=0)
    elif mode == 'fixed':
        hand = np.repeat(p[:1], len(p), axis=0)
        target = p[0] @ np.linalg.inv(p) @ o
    elif mode == 'fixed_original':
        hand = np.repeat(p[:1], len(p), axis=0)
        target = o
    elif mode in ('prescribed', 'arm'):
        hand, target = p, o
    else:
        raise ValueError(mode)
    if mode != 'fixed_original':
        if not np.allclose(np.linalg.inv(target) @ hand, np.linalg.inv(o) @ p, atol=2e-7):
            raise AssertionError('Relative-pose transport failed')
    return pose(hand), pose(target)


def load(row, limit=0):
    source = Path(row['source'])
    if sha(source) != row['sha256']:
        raise ValueError(f'Source changed: {source}')
    with h5py.File(source) as h:
        n = min(len(h['q_hand']), limit) if limit else len(h['q_hand'])
        def array(k):
            return np.asarray(h[k][:n, 0])
        attrs = {k: h.attrs[k].item() if isinstance(h.attrs[k], np.generic) else h.attrs[k]
                 for k in ('control_dt', 'physics_substeps', 'object_id', 'object_scale',
                           'fullhand_finger_stiffness', 'fullhand_finger_damping',
                           'fullhand_finger_effort_limit', 'contact_stiffness', 'contact_damping',
                           'split_fingertip_collision') if k in h.attrs}
        if not attrs['split_fingertip_collision'] or attrs['control_dt'] != .01:
            raise ValueError('Unexpected data contract')
        force = array('fingertip_force_world')
        strict = ((array('fingertip_collision_found') > .5)
                  & (array('true_fingertip_contact_valid') > .5)
                  & (np.linalg.norm(force, axis=-1) >= .05))
        return dict(row=row, n=n, attributes=attrs, q=array('q_hand'),
                    velocity=array('qvel')[:, -16:], command=array('q_cmd'),
                    full_q=array('q'), full_velocity=array('qvel'), arm_action=array('action')[:, :6],
                    palm=array('palm_pose_world'), obj=array('object_pose_world'),
                    strict=strict, found=array('fingertip_body_collision_found') > .5,
                    force=force, peak=array('contact_safety_peak_force_n'))


def geometry_signature(model):
    import mujoco
    rows=[]
    for g in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
        if not model.geom_contype[g] and not model.geom_conaffinity[g]: continue
        rows.append(dict(name=name,friction=model.geom_friction[g].tolist(),
                         solref=model.geom_solref[g].tolist(),solimp=model.geom_solimp[g].tolist(),
                         contype=int(model.geom_contype[g]),conaffinity=int(model.geom_conaffinity[g])))
    return rows


def preserve_parent_filter(before, after):
    """Fixed root weldid=0 must not enable formerly filtered parent contacts.

    MuJoCo's parent filter exempts world-welded bodies. Reproduce the original
    pair graph explicitly; never disable any formerly enabled contact pair.
    """
    import mujoco
    old=before.compile(); new=after.compile(); added=[]
    if old.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT):
        return added
    def parent_filtered(m,a,b):
        wa,wb=map(int,m.body_weldid[[a,b]])
        pa,pb=map(int,m.body_weldid[m.body_parentid[[wa,wb]]])
        return bool(wa and wb and (wa==pb or wb==pa))
    excluded={frozenset((x.bodyname1,x.bodyname2)) for x in after.excludes}
    for a in range(1,old.nbody):
        for b in range(a+1,old.nbody):
            if not parent_filtered(old,a,b):continue
            na=mujoco.mj_id2name(old,mujoco.mjtObj.mjOBJ_BODY,a)
            nb=mujoco.mj_id2name(old,mujoco.mjtObj.mjOBJ_BODY,b)
            aa=mujoco.mj_name2id(new,mujoco.mjtObj.mjOBJ_BODY,na)
            bb=mujoco.mj_name2id(new,mujoco.mjtObj.mjOBJ_BODY,nb)
            if aa<0 or bb<0:raise ValueError('Support changed body inventory')
            if not parent_filtered(new,aa,bb) and frozenset((na,nb)) not in excluded:
                after.add_exclude(bodyname1=na,bodyname2=nb)
                added.append((na,nb))
    return added


def run(args):
    script_hash=sha(__file__)
    import torch
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensorCfg, ContactMatch
    from mjlab.utils.lab_api.math import quat_error_magnitude
    from replay_inverted import replay_env_cfg
    from object_catalog import load_object_config
    from true_fingertip_contact import ContactSlotObserver
    from palm_support import WeldPalmSupport, KinematicPalmSupport
    from held_relative_target import install_held_relative_targets
    from feedback_self_contact import SelfContactDiagnostic
    from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import _load_mcc_leaphand_spec, mcc_finger_contact_env_cfg

    records=json.loads(SPLIT.read_text())['episodes']
    selected=[next(r for r in records if r['episode_id']==eid) for eid in args.ids]
    sources=[load(r,args.frames) for r in selected]
    a=sources[0]['attributes']
    if any(s['attributes']!=a for s in sources): raise ValueError('Mixed physics in a batch')
    device=args.device;batch=len(sources);length=max(s['n'] for s in sources)
    cfg=replay_env_cfg(num_envs=batch, hand_stiffness=a['fullhand_finger_stiffness'],
        hand_damping=a['fullhand_finger_damping'],hand_effort_limit=a['fullhand_finger_effort_limit'],
        object_config=load_object_config(a['object_id']),physics_substeps=a['physics_substeps'])
    if args.mode=='arm':
        cfg=mcc_finger_contact_env_cfg(num_envs=batch,object_config=load_object_config(a['object_id']),
            finger_stiffness=a['fullhand_finger_stiffness'],finger_damping=a['fullhand_finger_damping'],
            finger_effort_limit=a['fullhand_finger_effort_limit'],physics_substeps=a['physics_substeps'])
    # Copy the collection robot materials BEFORE anatomical tip splitting.
    collection_spec=_load_mcc_leaphand_spec()
    properties={g.name:{k:np.asarray(getattr(g,k)).copy() for k in
        ('friction','solref','solimp','priority','solmix','margin','gap','condim')} for g in collection_spec.geoms}
    original=cfg.scene.entities['robot'].spec_fn
    def matched_spec():
        spec=original()
        for g in spec.geoms:
            for name,value in properties.get(g.name,{}).items():
                if value.ndim: getattr(g,name)[:]=value
                else: setattr(g,name,value.item())
        return spec
    cfg.scene.entities['robot'].spec_fn=matched_spec
    observer=ContactSlotObserver(controller_scope='pad',capture_collision=True)
    observer.configure(cfg)
    if args.mode=='weld': driver=WeldPalmSupport(args.tau)
    elif args.mode in ('legacy','arm'): driver=None
    else: driver=KinematicPalmSupport()
    exclusions=[]
    before_support=cfg.scene.entities['robot'].spec_fn
    if driver is not None: driver.configure(cfg)
    if args.mode in ('fixed','fixed_original','prescribed'):
        after_support=cfg.scene.entities['robot'].spec_fn
        def preserve_spec():
            spec=after_support()
            exclusions[:]=preserve_parent_filter(before_support(),spec)
            return spec
        cfg.scene.entities['robot'].spec_fn=preserve_spec
    cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='all_hand_object',
        primary=ContactMatch(mode='subtree',pattern='^palm_lower$',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='^target_ball$',entity='target'),
        fields=('found','force'),reduce='maxforce',num_slots=1))
    arrays={}
    for name,shape,dtype in [('q',(16,),torch.float32),('command',(16,),torch.float32),
       ('qvel',(16,),torch.float32),('pad_points',(4,3),torch.float32),('pad_inward',(4,3),torch.float32),
       ('pad_found',(4,),torch.bool),('connector_force',(4,3),torch.float32),
       ('palm',(7,),torch.float32),('palm_error_m',(),torch.float32),('palm_error_rad',(),torch.float32),
       ('strict',(4,),torch.bool),('found',(4,),torch.bool),('pad_force',(4,3),torch.float32),
       ('peak',(),torch.float32),('self_peak',(),torch.float32),('held_error',(),torch.float32)]:
        arrays[name]=torch.zeros((length,batch,*shape),device=device,dtype=dtype)
    hand_paths=[];object_paths=[];commands=[];q_sources=[]
    for s in sources:
        h,o=paths(s['palm'],s['obj'],args.mode)
        def pad(v): return np.concatenate((v,np.repeat(v[-1:],length-len(v),axis=0)))
        hand_paths.append(pad(h));object_paths.append(pad(o));commands.append(pad(s['command']));q_sources.append(pad(s['q']))
    hand=torch.tensor(np.stack(hand_paths,1),device=device,dtype=torch.float32)
    objects=torch.tensor(np.stack(object_paths,1),device=device,dtype=torch.float32)
    targets=torch.tensor(np.stack(commands,1),device=device,dtype=torch.float32)
    restore=install_held_relative_targets();env=None
    started=time.monotonic()
    try:
        env=ManagerBasedRlEnv(cfg=cfg,device=device);env.reset()
        robot=env.scene['robot']; origins=env.scene.env_origins
        hand[:,:,:3]+=origins[None];objects[:,:,:3]+=origins[None]
        if driver is not None:driver.bind(env)
        root0=torch.cat((hand[0],torch.zeros((batch,6),device=device)),1)
        if args.mode=='arm': pass
        elif driver is None: robot.write_root_state_to_sim(root0)
        else: driver.write_root_state(root0)
        q0=torch.tensor(np.stack([s['full_q' if args.mode=='arm' else 'q'][0] for s in sources]),device=device)
        v0=torch.tensor(np.stack([s['full_velocity' if args.mode=='arm' else 'velocity'][0] for s in sources]),device=device)
        robot.write_joint_state_to_sim(q0,v0)
        init_target=targets[0]
        if args.mode=='arm':
            init_target=torch.cat((q0[:,:6],init_target),1)
        robot.set_joint_position_target(init_target);robot.write_data_to_sim()
        object_mocap=int(env.scene['target'].data.indexing.mocap_id)
        env.sim.data.mocap_pos[:,object_mocap]=objects[0,:,:3]
        env.sim.data.mocap_quat[:,object_mocap]=objects[0,:,3:]
        env.sim.forward();env.sim.sense()
        ids=robot.data.indexing.body_ids.cpu().numpy() if isinstance(robot.data.indexing.body_ids,torch.Tensor) else np.asarray(robot.data.indexing.body_ids)
        palm_id=next(i for i in ids if (mujoco.mj_id2name(env.sim.mj_model,mujoco.mjtObj.mjOBJ_BODY,int(i)) or '').endswith('/palm_lower'))
        palm_local=list(ids).index(palm_id)
        if args.mode=='arm':
            def is_hand(b):
                while b and b!=palm_id: b=env.sim.mj_model.body_parentid[b]
                return b==palm_id
            ids=np.array([i for i in ids if is_hand(i)])
        mask=np.isin(env.sim.mj_model.geom_bodyid,ids).astype(np.int32)
        own=SelfContactDiagnostic(env,mask)
        original_step=env.sim.step;frame=0;substep=0
        cmd=targets[0]
        control_sensors=[env.scene[n+'_contact'] for n in ('if_tip','mf_tip','rf_tip','th_tip')]
        def record_state(k):
            observer.read(env,[sen.data for sen in control_sensors]);state=observer.latest
            arrays['q'][k]=robot.data.joint_pos[:,-16:]
            arrays['qvel'][k]=robot.data.joint_vel[:,-16:]
            arrays['palm'][k]=robot.data.body_link_pose_w[:,palm_local]
            for name in ('pad_force','pad_found','pad_points','pad_inward','connector_force'):
                arrays[name][k]=state[name]
            arrays['strict'][k]=state['pad_found'] & torch.isfinite(state['pad_points']).all(-1) & (state['pad_force'].norm(dim=-1)>=.05)
            arrays['found'][k]=state['pad_found']|state['connector_found']
        for sen in control_sensors:sen.update(0.)
        record_state(0);arrays['command'][0]=targets[0]
        # Source endpoints are known for REPLAY only; no future teacher data is a policy observation.
        def physical_step():
            nonlocal substep
            substep+=1
            if args.mode=='weld':
                f=substep/float(a['physics_substeps'])
                previous=hand[frame-1];current=hand[frame]
                sign=torch.where((previous[:,3:]*current[:,3:]).sum(-1,keepdim=True)<0,-1.,1.)
                quat=(1-f)*previous[:,3:]+f*current[:,3:]*sign
                quat=quat/quat.norm(dim=-1,keepdim=True)
                # Directly update the anchor, never the integrated hand state.
                env.sim.data.mocap_pos[:,driver.mocap_id]=(1-f)*previous[:,:3]+f*current[:,:3]
                env.sim.data.mocap_quat[:,driver.mocap_id]=quat
            original_step()
            arrays['held_error'][frame]=torch.maximum(arrays['held_error'][frame],
                (robot.data.joint_pos_target[:,-16:]-cmd).abs().amax(-1))
            env.sim.sense()
            for sen in control_sensors:sen.update(0.)
            observer.read(env,[sen.data for sen in control_sensors])
            other=env.scene['all_hand_object'];other.update(0.)
            peak=torch.maximum(observer.safety_peak_force_n,other.data.force.norm(dim=-1).amax(-1))
            arrays['peak'][frame]=torch.maximum(arrays['peak'][frame],peak)
            arrays['self_peak'][frame]=torch.maximum(arrays['self_peak'][frame],own.read())
            if args.mode=='legacy': robot.write_root_state_to_sim(legacy_state)
        env.sim.step=physical_step
        for frame in range(1,length):
            substep=0;cmd=targets[frame]
            # Match source collection's object pose update before each control interval.
            env.sim.data.mocap_pos[:,object_mocap]=objects[frame,:,:3]
            env.sim.data.mocap_quat[:,object_mocap]=objects[frame,:,3:]
            if args.mode=='legacy':
                from palm_support import measured_pose_difference_twist
                twist=measured_pose_difference_twist(hand[frame],hand[frame-1],.01)
                legacy_state=torch.cat((hand[frame],twist),1)
                robot.write_root_state_to_sim(legacy_state)
            elif args.mode not in ('weld','arm'):
                driver.write_root_state(torch.cat((hand[frame],torch.zeros((batch,6),device=device)),1))
            env.sim.forward();env.sim.sense()
            action=(cmd-robot.data.joint_pos[:,-16:])/.08
            if args.mode=='arm':
                arm_action=torch.tensor(np.stack([s['arm_action'][min(frame,s['n']-1)] for s in sources]),device=device)
                action=torch.cat((arm_action,action),1)
            env.step(action)
            env.sim.sense()
            for sen in control_sensors:sen.update(0.)
            observer.read(env,[sen.data for sen in control_sensors]);x=observer.latest
            arrays['peak'][frame]=torch.maximum(arrays['peak'][frame],observer.safety_peak_force_n)
            q=robot.data.joint_pos[:,-16:]
            if not torch.isfinite(q).all():raise FloatingPointError(f'Nonfinite q at {frame}')
            arrays['q'][frame]=q;arrays['command'][frame]=cmd
            arrays['palm'][frame]=robot.data.body_link_pose_w[:,palm_local]
            arrays['palm_error_m'][frame]=(arrays['palm'][frame,:,:3]-hand[frame,:,:3]).norm(dim=-1)
            arrays['palm_error_rad'][frame]=quat_error_magnitude(arrays['palm'][frame,:,3:],hand[frame,:,3:])
            arrays['strict'][frame]=x['pad_found'] & torch.isfinite(x['pad_points']).all(-1) & (x['pad_force'].norm(dim=-1)>=.05)
            arrays['found'][frame]=x['pad_found']|x['connector_found']
            arrays['pad_force'][frame]=x['pad_force']
            record_state(frame)
            if frame%500==0:
                print(f'[SUPPORTED-REPLAY] {args.mode} ids={args.ids} frame={frame}/{length} elapsed={time.monotonic()-started:.1f}s',flush=True)
                save(args.output.with_suffix('.progress.json'),dict(frame=frame,total=length,elapsed_s=time.monotonic()-started))
        values={k:v.cpu().numpy() for k,v in arrays.items()}
        # No padded tail or initialization sample enters any metric.
        lanes=[]
        for i,s in enumerate(sources):
            valid=slice(1,s['n']);qerror=np.abs(values['q'][valid,i]-s['q'][1:])
            contact=values['strict'][valid,i];found=values['found'][valid,i]
            force=np.linalg.norm(values['pad_force'][valid,i],axis=-1)
            source_force=np.linalg.norm(s['force'][1:],axis=-1)
            row=dict(episode_id=s['row']['episode_id'],family=s['row']['family'],source=s['row']['source'],
                frames=s['n'],replayed_transitions=s['n']-1,complete=True,
                source_strict=float(s['strict'][1:].all(-1).mean()),source_found=float(s['found'][1:].all(-1).mean()),
                strict=float(contact.all(-1).mean()),found=float(found.all(-1).mean()),
                strict_per_finger=contact.mean(0).tolist(),found_per_finger=found.mean(0).tolist(),
                q_mae_rad=float(qerror.mean()),q_error_p95_rad=float(np.quantile(qerror,.95)),q_max_error_rad=float(qerror.max()),
                q_mae_first100_rad=float(qerror[:100].mean()),pad_force_p95_n=np.quantile(force,.95,axis=0).tolist(),
                source_pad_force_p95_n=np.quantile(source_force,.95,axis=0).tolist(),
                force_peak_n=float(values['peak'][valid,i].max()),source_force_peak_n=float(s['peak'][1:].max()),
                over100_frames=int((values['peak'][valid,i]>100).sum()),
                self_peak_n=float(values['self_peak'][valid,i].max()),self_over100_frames=int((values['self_peak'][valid,i]>100).sum()),
                palm_error_p95_m=float(np.quantile(values['palm_error_m'][valid,i],.95)),
                palm_error_max_m=float(values['palm_error_m'][valid,i].max()),
                palm_rotation_p95_rad=float(np.quantile(values['palm_error_rad'][valid,i],.95)),
                held_target_max_error_rad=float(values['held_error'][valid,i].max()))
            row['gate']=dict(contact=row['strict']>=row['source_strict']-.02 and row['found']>=row['source_found']-.02,
                q=row['q_mae_rad']<=.01 and row['q_error_p95_rad']<=.03,
                force=bool(np.all(np.quantile(force,.95,axis=0)<=np.maximum(1.25*np.quantile(source_force,.95,axis=0),np.quantile(source_force,.95,axis=0)+2.))) and row['over100_frames']==0 and row['self_over100_frames']==0,
                target=row['held_target_max_error_rad']<=1e-6)
            row['reproduction_gate_passed']=all(row['gate'].values())
            lanes.append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.trace:np.savez_compressed(args.output.with_suffix('.npz'),**values,source_q=np.stack(q_sources,1),reference_palm=hand.cpu().numpy())
        if args.collect_dir is not None:
            args.collect_dir.mkdir(parents=True,exist_ok=True)
            for i,s in enumerate(sources):
                destination=args.collect_dir/f"ep{s['row']['episode_id']:04d}.h5"
                with h5py.File(destination,'x') as h:
                    # One physical transition per row: state BEFORE the next
                    # target update, requested action, measured NEXT state.
                    for name in ('q','qvel','palm','pad_force','pad_found','pad_points','pad_inward','connector_force','strict','found'):
                        for suffix,sl in (('pre',slice(0,s['n']-1)),('next',slice(1,s['n']))):
                            h.create_dataset(f'{name}_{suffix}',data=values[name][sl,i],compression='gzip')
                    for name,data in dict(q_cmd=values['command'][1:s['n'],i],
                        object_pose_pre=objects[:s['n']-1,i].cpu().numpy(),
                        object_pose_command=objects[1:s['n'],i].cpu().numpy(),
                        palm_reference_command=hand[1:s['n'],i].cpu().numpy(),
                        object_force_peak_n=values['peak'][1:s['n'],i],self_force_peak_n=values['self_peak'][1:s['n'],i]).items():
                        h.create_dataset(name,data=data,compression='gzip')
                    h.attrs.update(schema='supported_contact_transition_v1',support=args.mode,control_dt=.01,
                        source=s['row']['source'],source_sha256=sha(s['row']['source']),script_sha256=script_hash,
                        command_source='recorded final actuator target; original MCC already included; no online MCC pass',
                        expert_demonstration=False,initial_warmstart_restored=False,
                        object_support='kinematic mocap; this dataset does not model a freely dynamic object',
                        palm_force_valid=False if args.mode in ('fixed','fixed_original','prescribed') else True,
                        palm_force_caveat='fixed palm and mocap object have prescribed motion; their mutual reaction force is not a physical label',
                        reference_frame='world including lane origin; quaternions wxyz; pad_inward points into object',
                        labels='new physical measurements, not copied from source; retain failures and force events')
        report=dict(mode=args.mode,tau=args.tau,lanes=lanes,elapsed_s=time.monotonic()-started,
            scope='recorded absolute actuator targets; no policy inference or second MCC correction; full source lengths unless frames explicitly set',
            limits='q/qvel restored but source constraint warmstart unavailable; inversion preserves relative pose, not inertial/support equivalence',
            source_physics=a,geometry=geometry_signature(env.sim.mj_model),
            support=driver.report() if driver is not None else dict(kind=args.mode),
            source_hashes={s['row']['source']:sha(s['row']['source']) for s in sources},
            provenance=dict(script_sha256=script_hash,script_unchanged=script_hash==sha(__file__),sources_unchanged=True),contact=observer.definition(),
            explicit_parent_filter_preservation=exclusions)
        report['force_scope']='MuJoCo enabled contacts; fixed palm versus mocap object has no dynamic reaction-force label. Finger-object and dynamic finger self-contact remain measured.'
        if args.mode=='weld':
            report['support']['timing']='1ms interpolation between recorded palm endpoints, replay only'
        save(args.output,report)
        print(json.dumps(dict(mode=args.mode,lanes=lanes,elapsed_s=report['elapsed_s'])),flush=True)
        return report
    finally:
        if env is not None:
            if 'original_step' in locals():env.sim.step=original_step
            env.close()
        restore()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids',type=int,nargs='+',required=True)
    p.add_argument('--mode',choices=('weld','legacy','fixed','fixed_original','prescribed','arm'),required=True)
    p.add_argument('--tau',type=float,default=.005)
    p.add_argument('--frames',type=int,default=0,help='0 means full source lengths')
    p.add_argument('--device',default='cuda:0');p.add_argument('--trace',action='store_true')
    p.add_argument('--collect-dir',type=Path,help='Write measured transition H5s, not expert labels; refuses overwrite')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args)


if __name__=='__main__':main()
