"""Read-only solver-contact attribution at the exact safety peak substep.

No new sensors or control changes. Decode MuJoCo-Warp's existing constraint
forces, retain pad/connector net forces, and attach actual geometry IDs, link
names, joint ancestry and command motion along each contact normal.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force, contact_force_fn
from mjlab.envs import ManagerBasedRlEnv
import true_fingertip_contact
from batched_b_control_chain import BatchedBControlChain

FINGERS=('index','middle','ring','thumb')
JOINTS=('1','0','2','3','5','4','6','7','9','8','10','11','12','13','14','15')


@wp.kernel
def all_object_loads(cone:int, frame:wp.array(dtype=wp.mat33),
    friction:wp.array(dtype=wp.types.vector(length=5,dtype=float)), dim:wp.array(dtype=int),
    address:wp.array2d(dtype=int), efc:wp.array2d(dtype=float), njmax:int,
    count:wp.array(dtype=int), geom:wp.array(dtype=wp.vec2i), world:wp.array(dtype=int),
    objects:wp.array(dtype=int), hand:wp.array(dtype=int), net:wp.array2d(dtype=wp.vec3),
    pointmax:wp.array2d(dtype=float)):
    c=wp.tid()
    if c>=count[0]:return
    a=geom[c][0];b=geom[c][1]
    if a<0 or b<0:return
    h=-1;sign=1.0
    if hand[a]>0 and objects[b]>0:h=a
    elif hand[b]>0 and objects[a]>0:h=b;sign=-1.0
    if h<0:return
    f=contact_force_fn(cone,frame,friction,dim,address,efc,njmax,count,world[c],c,True)
    wp.atomic_add(net,world[c],h,sign*wp.spatial_top(f))
    wp.atomic_max(pointmax,world[c],h,wp.length(wp.spatial_top(f)))


def region(name):
    short=name.split('/')[-1]
    if short.endswith('_pad_collision'):return 'true_fingertip'
    if short.endswith('_connector_collision'):return 'connector'
    if short=='palm_lower_collision':return 'palm'
    return 'other_link'


class Recorder:
    def __init__(self,args):
        self.args=args;self.call=0;self.inside=False;self.phase='physics';self.substep=0
        self.env=None;self.observer=None;self.pre=None;self.mcc=None
        self.peaks=[None]*len(args.ids);self.maxima=np.zeros(len(args.ids))
        self.first_stops=[None]*len(args.ids)
        self.incidental=[None]*len(args.ids);self.incidental_max=np.zeros(len(args.ids))
        self.aggregate=None
        self.streams=None

    def observe(self,env,observer):
        self.env=env;self.observer=observer
        if self.streams is None:
            self.streams=dict(torch_cuda_stream=torch.cuda.current_stream().cuda_stream,
                              warp_cuda_stream=wp.get_stream(env.sim.wp_device).cuda_stream)
        if not self.inside:return
        if self.phase=='physics':self.substep+=1
        offset=0 if self.args.mode in ('source-arm','raw') else 200
        if self.call<offset:return
        force=observer.safety_peak_force_n.detach().cpu().numpy()
        d=env.sim.wp_data;m=env.sim.mj_model
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
        if self.aggregate is None:
            object_mask=np.array([n.startswith('target/') for n in names],dtype=np.int32)
            hand_mask=np.array([n.startswith('robot/') for n in names],dtype=np.int32)
            with wp.ScopedDevice(env.sim.wp_device):
                self.aggregate=dict(objects=wp.array(object_mask,dtype=wp.int32),
                    hand=wp.array(hand_mask,dtype=wp.int32),net=wp.zeros((env.num_envs,m.ngeom),dtype=wp.vec3),
                    pointmax=wp.zeros((env.num_envs,m.ngeom),dtype=float))
            self.other_mask=torch.tensor([n.startswith('robot/') and region(n) in ('palm','other_link')
                                         for n in names],device=env.device)
        a=self.aggregate
        with wp.ScopedDevice(env.sim.wp_device):
            a['net'].zero_()
            a['pointmax'].zero_()
            wp.launch(all_object_loads,dim=d.naconmax,inputs=[env.sim.wp_model.opt.cone,
                d.contact.frame,d.contact.friction,d.contact.dim,d.contact.efc_address,d.efc.force,
                d.njmax,d.nacon,d.contact.geom,d.contact.worldid,a['objects'],a['hand']],outputs=[a['net'],a['pointmax']])
        # Consume the read-only Warp output on its producer stream.
        stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
        with torch.cuda.stream(stream):
            load=wp.to_torch(a['net']).norm(dim=-1)
            other_net=load[:,self.other_mask].amax(-1).cpu().numpy()
            other_point=wp.to_torch(a['pointmax'])[:,self.other_mask].amax(-1).cpu().numpy()
        other=np.maximum(other_net,other_point)
        chosen=[i for i,f in enumerate(force) if f>=80 and
                (f>self.maxima[i] or (f>100 and self.first_stops[i] is None))]
        chosen=sorted(set(chosen+[i for i,f in enumerate(other) if f>100 and f>self.incidental_max[i]]))
        if not chosen:return
        # Extract actual solver forces, never recompute forces from a saved q.
        count=int(d.nacon.numpy()[0])
        with wp.ScopedDevice(env.sim.wp_device):
            ids=wp.array(np.arange(count,dtype=np.int32),dtype=wp.int32)
            force_out=wp.zeros(count,dtype=wp.spatial_vector)
            contact_force(env.sim.wp_model,d,ids,False,force_out)
        local=force_out.numpy()[:,:3]
        frames=d.contact.frame.numpy()[:count]
        world_force=np.einsum('nki,nk->ni',frames,local)
        geom=d.contact.geom.numpy()[:count]
        worlds=d.contact.worldid.numpy()[:count]
        points=d.contact.pos.numpy()[:count]
        distance=d.contact.dist.numpy()[:count]
        anchors=d.xanchor.numpy();axes=d.xaxis.numpy()
        qpos=d.qpos.numpy();qvel=d.qvel.numpy()
        cvel=d.cvel.numpy();subtree_com=d.subtree_com.numpy()
        hand_joint_ids=[next(j for j in range(m.njnt)
            if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or '').split('/')[-1]==name) for name in JOINTS]
        qa=m.jnt_qposadr[hand_joint_ids];va=m.jnt_dofadr[hand_joint_ids]
        x={key:value.detach().cpu().numpy() for key,value in observer.latest.items()}
        for lane in chosen:
            pad=x['pad_force'][lane];connector=x['connector_force'][lane]
            loads=np.stack((np.linalg.norm(pad,axis=-1),np.linalg.norm(connector,axis=-1),
                            np.linalg.norm(pad+connector,axis=-1)),-1)
            finger,kind=np.unravel_index(np.argmax(loads),loads.shape)
            reconstructed_pad=np.zeros((4,3));reconstructed_connector=np.zeros((4,3))
            rows=[]
            solver_q=qpos[lane,qa].copy()
            if self.phase=='physics':solver_q-=float(env.physics_dt)*qvel[lane,va]
            target=self.pre['command'][lane]
            for c in np.flatnonzero(worlds==lane):
                g0,g1=map(int,geom[c])
                if g0<0 or g1<0:continue
                if names[g0].startswith('robot/') and names[g1].startswith('target/'):
                    hand,obj=g0,g1;sign=1.
                elif names[g1].startswith('robot/') and names[g0].startswith('target/'):
                    hand,obj=g1,g0;sign=-1.
                else:continue
                f=world_force[c]*sign
                if np.linalg.norm(f)<.01:continue
                name=names[hand];part=region(name)
                outward=-sign*frames[c,0]
                bid=int(m.geom_bodyid[hand]);body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid)
                ancestors=[];cursor=bid
                while cursor>0:
                    for j in range(m.njnt):
                        if m.jnt_bodyid[j]==cursor:ancestors.append(j)
                    cursor=int(m.body_parentid[cursor])
                columns=[j for j in range(16) if hand_joint_ids[j] in ancestors]
                finger_id=columns[0]//4 if columns else None
                if finger_id is not None:
                    if part=='true_fingertip':reconstructed_pad[finger_id]+=f
                    if part=='connector':reconstructed_connector[finger_id]+=f
                jac=np.stack([np.cross(axes[lane,hand_joint_ids[j]],points[c]-anchors[lane,hand_joint_ids[j]])
                              if j in columns else np.zeros(3) for j in range(16)],-1)
                jn=outward@jac
                cv=cvel[lane,bid];com=subtree_com[lane,m.body_rootid[bid]]
                native_velocity=cv[3:]+np.cross(cv[:3],points[c]-com)
                record=dict(hand_geom=name,object_geom=names[obj],body=body,region=part,
                    finger=FINGERS[finger_id] if finger_id is not None else None,
                    hand_joint_chain=[JOINTS[j] for j in columns],
                    net_force_vector_sensor_convention=f.tolist(),single_contact_force_n=float(np.linalg.norm(f)),
                    normal_force_n=float(local[c,0]),penetration_mm=max(0.,-float(distance[c]))*1000,
                    point_world=points[c].tolist(),outward_world=outward.tolist(),
                    actuator_target_normal_step_mm=float(jn@(target-solver_q))*1000,
                    joint_target_delta_rad=(target-solver_q)[columns].tolist(),
                    instantaneous_joint_normal_velocity_mm_s=float(jn@qvel[lane,va])*1000,
                    pre_solve_contact_normal_velocity_mm_s=float(outward@native_velocity)*1000)
                if self.mcc is not None:
                    record['nominal_target_normal_step_mm']=float(jn@(self.mcc['nominal_q'][lane]-solver_q))*1000
                    record['mcc_change_normal_step_mm']=float(jn@(target-self.mcc['nominal_q'][lane]))*1000
                rows.append(record)
            grouped={}
            for r in rows:
                name=r['hand_geom']
                if name not in grouped:grouped[name]=dict(geom=name,region=r['region'],finger=r['finger'],
                    body=r['body'],hand_joint_chain=r['hand_joint_chain'],force=np.zeros(3),depth=0.)
                grouped[name]['force']+=np.asarray(r['net_force_vector_sensor_convention'])
                grouped[name]['depth']=max(grouped[name]['depth'],r['penetration_mm'])
            geoms=[dict(**{k:v for k,v in item.items() if k not in ('force','depth')},
                        net_force_n=float(np.linalg.norm(item['force'])),penetration_mm=item['depth'])
                   for item in grouped.values()]
            step=self.call-offset+(1 if self.args.mode in ('source-arm','raw') else 0)
            event=dict(trajectory_id=self.args.ids[lane],task_frame=step,
                phase=self.phase,physical_substep=self.substep,
                force_peak_n=float(force[lane]),trigger_finger=FINGERS[finger],
                trigger_component=('pad','connector','pad_plus_connector')[kind],
                pad_net_force_n=loads[:,0].tolist(),connector_net_force_n=loads[:,1].tolist(),
                combined_net_force_n=loads[:,2].tolist(),
                pad_reconstruction_max_error_n=float(abs(reconstructed_pad-pad).max()),
                connector_reconstruction_max_error_n=float(abs(reconstructed_connector-connector).max()),
                other_link_net_force_max_n=float(other_net[lane]),
                other_link_single_contact_max_n=float(other_point[lane]),
                geometry_loads=sorted(geoms,key=lambda r:-r['net_force_n']),
                contacts=sorted(rows,key=lambda r:-r['single_contact_force_n']),
                control=dict(q_pre=self.pre['q'][lane].tolist(),command=target.tolist()),
                signs='Positive normal target displacement/velocity means outward withdrawal; sensor force sign matches existing netforce sensors',
                joint_semantics='Joint ancestry of the colliding link;100N is contact-force stop, not joint torque overload')
            palm_id=next(b for b in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or '').endswith('/palm_lower'))
            object_ids=[b for b in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or '').startswith('target/') and m.body_mocapid[b]>=0]
            positions=d.xpos.numpy();quats=d.xquat.numpy()
            event['control']['q_solver_pre']=solver_q.tolist()
            event['control']['palm_pose_solver_world']=np.r_[positions[lane,palm_id],quats[lane,palm_id]].tolist()
            event['control']['object_pose_world']=np.r_[positions[lane,object_ids[0]],quats[lane,object_ids[0]]].tolist()
            if self.mcc is not None:
                event['mcc']={key:value[lane].tolist() for key,value in self.mcc.items()}
            for name in ('pad_force_n','connector_force_n'):
                if name in self.pre:event['control']['pre_'+name]=self.pre[name][lane].tolist()
            if force[lane]>=80 and force[lane]>self.maxima[lane]:self.maxima[lane]=force[lane];self.peaks[lane]=event
            if force[lane]>100 and self.first_stops[lane] is None:
                self.first_stops[lane]=event
                print('[PEAK PART]',event['trajectory_id'],event['task_frame'],event['trigger_finger'],
                      event['trigger_component'],round(event['force_peak_n'],2),flush=True)
            if other[lane]>100 and other[lane]>self.incidental_max[lane]:
                self.incidental_max[lane]=other[lane];self.incidental[lane]=event


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('source-arm','raw','oracle','dp'),required=True)
    p.add_argument('--ids',type=int,nargs='+',default=[36,39,41,43,45])
    p.add_argument('--frames',type=int,default=4500)
    p.add_argument('--palm-support',choices=('free','kinematic','weld'),default='free')
    p.add_argument('--command-error-limit',type=float,default=None)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    recorder=Recorder(args)
    old_observer=true_fingertip_contact.ContactSlotObserver
    old_step=ManagerBasedRlEnv.step;old_chain_step=BatchedBControlChain.step
    class Observed(old_observer):
        def read(self,env,*values,**kwargs):
            result=super().read(env,*values,**kwargs);recorder.observe(env,self);return result
    def step(env,action):
        recorder.env=env;recorder.inside=False
        if recorder.observer is not None:recorder.observer.read(env,None)
        q=env.scene['robot'].data.joint_pos[:,-16:].detach().cpu().numpy()
        recorder.pre=dict(q=q,command=q+.08*action[:,-16:].detach().cpu().numpy())
        if recorder.observer is not None:
            x=recorder.observer.latest
            recorder.pre['pad_force_n']=x['pad_force'].norm(dim=-1).cpu().numpy()
            recorder.pre['connector_force_n']=x['connector_force'].norm(dim=-1).cpu().numpy()
        recorder.inside=True;recorder.phase='physics';recorder.substep=0
        try:
            result=old_step(env,action)
            recorder.phase='post_control'
            if recorder.observer is not None:recorder.observer.read(env,None)
            return result
        finally:recorder.inside=False;recorder.call+=1
    def chain_step(chain,**kwargs):
        action,diag=old_chain_step(chain,**kwargs)
        keep=('nominal_q','command_q','collision_safety_active','raw_force_magnitude',
              'normal_offset','paired_engaged','paired_infeasible','paired_normal_force_n','palm_normal_step_m')
        recorder.mcc={key:value.detach().cpu().numpy() for key,value in diag.items() if key in keep}
        recorder.mcc['force_filtered']=chain.mcc.force_filtered.detach().cpu().numpy().copy()
        return action,diag
    true_fingertip_contact.ContactSlotObserver=Observed
    ManagerBasedRlEnv.step=step;BatchedBControlChain.step=chain_step
    try:
        if args.mode=='source-arm':
            import replay_source_arm_commands
            replay_source_arm_commands.run(argparse.Namespace(ids=args.ids,frames=args.frames,output=args.output))
        else:
            import audit_long_free_palm
            audit_long_free_palm.run(argparse.Namespace(mode=args.mode,ids=args.ids,frames=args.frames,
                output=args.output,seed=20261013,delay_frames=0,root_velocity='recorded',
                materials='collection' if args.mode=='raw' else 'deployment',projection_live_step=None,
                command_error_limit=args.command_error_limit,palm_support=args.palm_support,
                support_time_constant=.01,record_base=True))
    finally:
        true_fingertip_contact.ContactSlotObserver=old_observer
        ManagerBasedRlEnv.step=old_step;BatchedBControlChain.step=old_chain_step
        result=dict(mode=args.mode,palm_support=args.palm_support,ids=args.ids,
            first_over100=recorder.first_stops,maximum_events=recorder.peaks,
            other_link_over100=recorder.incidental,
            cuda_streams=recorder.streams,
            scope='Actual physical solver contacts at force peaks; control and collision models unchanged')
        (args.output/'force_parts.json').write_text(json.dumps(result,indent=2)+'\n')
        for event in recorder.first_stops:
            if event is not None:
                print('[FORCE-PART]',event['trajectory_id'],event['task_frame'],event['trigger_finger'],
                      event['trigger_component'],round(event['force_peak_n'],2),flush=True)


if __name__=='__main__':main()
