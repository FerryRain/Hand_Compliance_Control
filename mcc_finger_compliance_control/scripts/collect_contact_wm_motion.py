"""Diagnose object-motion semantics: 1ms mocap ramp vs welded dynamic body.

Ordinary MCC regulation and actuator parameters are retained. WM mode disables
threshold takeover and records overload instead of stopping. Legacy behavior
remains explicitly selectable. Physical/measurement checks are independent of
task success; this is not a real-hardware safety configuration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
import collect_contact_wm as collection
from contact_wm_motion_profiles import NORMAL_BEHAVIORS,assign_motion_diversity,normal_excursion_m


def assign_motion_rates(jobs,args):
    speed_range=getattr(args,'motion_rate_range',None)
    for job in jobs:
        rate=(float(np.random.default_rng(job['seed']+47021).uniform(*speed_range))
              if speed_range is not None else float(args.motion_rate))
        job.update(motion_rate=rate,physical_motion_period_s=args.motion_period_s/rate,
                   physical_startup_ramp_s=.3/rate)


@wp.kernel
def contact_depth(count:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),
                  world:wp.array(dtype=int),dist:wp.array(dtype=float),
                  target:wp.array(dtype=int),hand:wp.array(dtype=int),out:wp.array2d(dtype=float)):
    c=wp.tid()
    if c>=count[0]:return
    a=geom[c][0];b=geom[c][1]
    if a<0 or b<0:return
    h=-1
    if target[a]>0 and hand[b]>0:h=hand[b]-1
    elif target[b]>0 and hand[a]>0:h=hand[a]-1
    if h>=0:wp.atomic_max(out,world[c],h,wp.max(0.,-dist[c]))


class MotionProbe(collection.Probe):
    def __init__(self,args,row):
        self.motion_ready=False;self.trace=[];self.phase_now=0
        self.measurement_audit=dict(pad_normal_sign_mismatch_samples=0,
            connector_normal_sign_mismatch_samples=0,max_net_force_discrepancy_n=0.)
        super().__init__(args,row)
        # Independent per-episode clock; do not alter the initialization RNG or
        # jump between speed multipliers inside a trajectory.
        assign_motion_rates(self.jobs,args)
        assign_motion_diversity(self.jobs,getattr(args,'randomize_motion',False))
        import mujoco
        m=self.env.sim.mj_model
        self.target_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'target/target_ball')
        self.joint_q_indices=self.robot.data.indexing.joint_q_adr.cpu().numpy()
        self.joint_v_indices=self.robot.data.indexing.joint_v_adr.cpu().numpy()
        self.free_joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'target/object_free')
        if self.free_joint>=0:
            self.free_q=int(m.jnt_qposadr[self.free_joint]);self.free_v=int(m.jnt_dofadr[self.free_joint])
        self.driver_pose=self.object.clone();self.motion_ready=True
        self.palm_driver_pose=self.palm.clone();self.palm_goal=self.palm.clone()
        self.palm_base=self.palm.clone();self.palm_path=None;self.object_path=None
        target=[];hand=[]
        for g in range(m.ngeom):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            target.append(int(name.startswith('target/')))
            hand.append(1 if name.endswith('_pad_collision') else 2 if name.startswith('robot/') else 0)
        with wp.ScopedDevice(self.env.sim.wp_device):
            self.target_mask=wp.array(target,dtype=int);self.hand_mask=wp.array(hand,dtype=int)
            self.depth=wp.zeros((self.n,2),dtype=float)
            self.mass_times_acc=wp.zeros((self.n,m.nv),dtype=float)
        original=self.env.sim.step
        def substep():
            # mj_step's contacts/forces belong to the pre-integration state.
            # Keep both states; never label that force as a new-state force.
            self.pre_substep_qpos=self.env.sim.data.qpos.cpu().numpy().copy()
            self.pre_substep_qvel=self.env.sim.data.qvel.cpu().numpy().copy()
            self.subtick+=1;fraction=self.subtick/10
            pose=torch.lerp(self.motion_start,self.motion_goal,fraction)
            qa=self.motion_start[:,3:];qb=self.motion_goal[:,3:]
            qb=torch.where((qa*qb).sum(-1,keepdim=True)<0,-qb,qb)
            quat=torch.lerp(qa,qb,fraction);pose[:,3:]=quat/quat.norm(dim=-1,keepdim=True)
            self.driver_pose=pose
            self.env.sim.data.mocap_pos[:,self.target]=pose[:,:3]
            self.env.sim.data.mocap_quat[:,self.target]=pose[:,3:]
            if self.args.motion_owner=='palm':
                palm=torch.lerp(self.palm_start,self.palm_goal,fraction)
                palm[:,3:]/=palm[:,3:].norm(dim=-1,keepdim=True)
                self.palm_driver_pose=palm
                self.env.sim.data.mocap_pos[:,self.driver.mocap_id]=palm[:,:3]
                self.env.sim.data.mocap_quat[:,self.driver.mocap_id]=palm[:,3:]
            try:original()
            finally:self.record_substep()
        self.env.sim.step=substep

    def write_object(self,value):
        if not self.motion_ready:
            super().write_object(value)
            import mujoco
            m=self.env.sim.mj_model;j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'target/object_free')
            if j>=0:
                q=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
                self.env.sim.data.qpos[:,q:q+7]=value;self.env.sim.data.qvel[:,v:v+6]=0
        # During collection only the driver moves. Never reset object qpos/qvel.

    def actual_pose(self):
        if self.free_joint>=0:return self.env.sim.data.qpos[:,self.free_q:self.free_q+7].clone()
        return self.driver_pose.clone()

    def step(self,nominal,object_pose,phase,use_mcc=True):
        self.phase_now=phase;self.subtick=0
        self.motion_start=self.driver_pose.clone();self.motion_goal=object_pose.clone()
        self.palm_start=self.palm_driver_pose.clone()
        if self.args.motion_owner=='palm':self.palm=self.robot.data.root_link_pose_w.clone()
        return super().step(nominal,object_pose,phase,use_mcc=use_mcc and not self.args.no_mcc)

    def finish_initialization(self,records):
        # Freeze the reference BEFORE force loading. Carry its compliance state
        # into motion; rebasing q/resetting MCC afterwards would double count or
        # abruptly remove the preload.
        self.fixed_reference_q=self.robot.data.joint_pos.clone()
        self.load_settle_frames_executed=0
        requested=getattr(self.args,'load_settle_frames',0)
        if requested<=0 or not bool(self.initialized.all()):return records
        remaining=max(0,250-(len(records)-1))
        streak=torch.zeros(self.n,device=self.device,dtype=torch.long)
        for _ in range(min(requested,remaining)):
            record=self.step(self.fixed_reference_q,self.object.clone(),0,use_mcc=True)
            x=self.observer.latest;loaded=x['pad_found']&(x['pad_force'].norm(dim=-1)>=.2)
            streak=torch.where(collection.all_active_loaded(loaded,self.active),streak+1,0)
            self.initialized=streak>=15
            record['initialized']=self.initialized.cpu().numpy().copy()
            records.append(record);self.load_settle_frames_executed+=1
            if self.done_reason:break
        if self.load_settle_frames_executed<15:self.initialized.zero_()
        records[-1]['initialized']=self.initialized.cpu().numpy().copy()
        return records

    def plan_palm_path(self):
        """GT palm-only height-following path; no finger IK or q waypoints."""
        from contact_wm_geometry import surface_anchor
        from scipy.spatial.transform import Rotation
        self.palm_base=self.palm_driver_pose.clone()
        actual=self.actual_pose().cpu().numpy();palms=self.palm_base.cpu().numpy()
        x=self.observer.latest;points=x['pad_physical_points'].cpu().numpy()
        plans=[]
        for lane,job in enumerate(self.jobs):
            f=job['finger'];R=Rotation.from_quat(actual[lane,[4,5,6,3]]).as_matrix()
            anchor=R.T@(points[lane,f]-actual[lane,:3])
            start,_=surface_anchor(self.args.root,self.row,anchor[:2]);path=[]
            sign=1 if job['seed']%2 else -1
            for frame in range(self.args.frames):
                u=frame/max(self.args.frames-1,1);u=u*u*(3-2*u)
                dx=sign*self.args.path_length_m*u
                point,_=surface_anchor(self.args.root,self.row,anchor[:2]+[dx,0.])
                delta=np.array([dx,0.,point[2]-start[2] if self.args.surface_follow else 0.])
                p=palms[lane].copy();p[:3]+=R@delta;path.append(p)
            plans.append(path)
        self.palm_path=torch.tensor(np.array(plans).transpose(1,0,2),device=self.device,dtype=torch.float32)

    def prescribed_object_motion(self,frame,include_normal=True):
        from scipy.spatial.transform import Rotation
        # Compress the analytic path clock, never the physics timestep.
        physical_frame=frame
        frame=frame*getattr(self.args,'motion_rate',1.)
        _,pose=super().desired(frame)
        t=frame*.01;phase=2*np.pi*t/self.args.motion_period_s
        r=min(t/.3,1.);ramp=r*r*(3-2*r)
        for i,job in enumerate(self.jobs):
            if getattr(self.args,'motion_rate_range',None) is not None:
                t=frame*.01*job['motion_rate']/self.args.motion_rate
                phase=2*np.pi*t/self.args.motion_period_s
                r=min(t/.3,1.);ramp=r*r*(3-2*r)
            kind=job['behavior']
            angle=job.get('motion_direction_rad',0.);scale=job.get('motion_amplitude_scale',1.)
            direction=np.array([np.cos(angle),np.sin(angle),0.])
            if kind in ('slide','slide_release') and self.args.fixed_reference:
                sign=1 if job['seed']%2 else -1
                local=direction*(sign*self.args.slide_amplitude_m*scale*np.sin(phase)*ramp)
                pose[i,:3]=self.base[i,:3]+self.rot[i]@torch.tensor(local,device=self.device,dtype=torch.float32)
            elif kind=='ellipse':
                x,y=.020*np.sin(phase),.012*(1-np.cos(phase))
                local=np.array([np.cos(angle)*x-np.sin(angle)*y,np.sin(angle)*x+np.cos(angle)*y,0])*ramp*scale
                pose[i,:3]=self.base[i,:3]+self.rot[i]@torch.tensor(local,device=self.device,dtype=torch.float32)
            elif kind=='rotate':
                # Rotate about the object centre, NOT about the initial
                # fingertip material point (which would cancel the traversal).
                rotation=self.rot[i].cpu().numpy()@Rotation.from_rotvec(direction*(.12*scale*np.sin(phase)*ramp)).as_matrix()
                pose[i,:3]=self.base[i,:3]
                pose[i,3:]=torch.tensor(Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]],device=self.device,dtype=torch.float32)
            if include_normal and kind in ('slide_release','normal_cycle'):
                pose[i,:3]+=self.inward[i]*normal_excursion_m(job,physical_frame,self.args.motion_period_s)
        return pose

    def plan_object_path(self):
        """Adjust object height, keeping its prescribed tangent/rotation motion.

        The world-space pregrasp/contact reference stays fixed. New material
        passes under it; no finger target follows an object-fixed contact point.
        No measured-force feedback or lost-contact search changes this path.
        """
        import mujoco
        from split_fingertip_collision import GEOMS
        from contact_wm_initialization import set_object_static, hand_qpos_indices
        from contact_wm_geometry import solve_clearance_height
        from contact_wm_clearance import FixedTipClearance
        base=self.base.cpu().numpy().astype(float)
        path=np.stack([self.prescribed_object_motion(f,include_normal=False).cpu().numpy() for f in range(self.args.frames)]).astype(float)
        original=path.copy();diagnostics=[];height_spreads=[]
        model=self.env.sim.mj_model
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(model.ngeom)]
        targets=np.array([g for g,n in enumerate(names) if n.startswith('target/') and
                          (model.geom_contype[g] or model.geom_conaffinity[g])])
        initial_qpos=self.env.sim.data.qpos.cpu().numpy()
        initial_mocap_pos=self.env.sim.data.mocap_pos.cpu().numpy()
        initial_mocap_quat=self.env.sim.data.mocap_quat.cpu().numpy()
        q=self.fixed_reference_q.cpu().numpy()
        for lane,job in enumerate(self.jobs):
            fingers=collection.job_fingers(job);finger=fingers[0];data=mujoco.MjData(model)
            data.qpos[:]=initial_qpos[lane]
            data.qpos[hand_qpos_indices(model)]=q[lane]
            data.mocap_pos[:]=initial_mocap_pos[lane]
            data.mocap_quat[:]=initial_mocap_quat[lane]
            pads=[next(g for g,n in enumerate(names) if n.endswith(GEOMS[f]+'_pad_collision')) for f in fingers]
            distance_queries=[FixedTipClearance(model,data,pad,targets) for pad in pads]
            def clearance(pose,fi):
                set_object_static(model,data,self.target,pose[:3],pose[3:])
                mujoco.mj_kinematics(model,data)
                return distance_queries[fi]()
            initial=np.array([clearance(base[lane],fi) for fi in range(len(fingers))])
            if np.max(np.abs(initial))>.003:
                raise ValueError(f'Initial pregrasp clearance inconsistent with initialized contact: lane {lane}, {initial}')
            initial_surface=[]
            for fi in range(len(fingers)):
                initial_pose=base[lane].copy()
                def initial_gap(z):
                    initial_pose[2]=z;return clearance(initial_pose,fi)
                z,_=solve_clearance_height(initial_gap,base[lane,2],base[lane,2]);initial_surface.append(z)
            initial_surface_z=np.array(initial_surface)
            preload_height=base[lane,2]-initial_surface_z
            previous_z=initial_surface_z.copy()
            errors=[];iterations=[];spreads=[]
            for frame in range(self.args.frames):
                pose=path[frame,lane]
                heights=[]
                for fi in range(len(fingers)):
                    def gap_at_height(z):
                        pose[2]=z;return clearance(pose,fi)
                    surface_z,queries=solve_clearance_height(gap_at_height,previous_z[fi],original[frame,lane,2])
                    pose[2]=surface_z
                    errors.append(clearance(pose,fi)-2.e-5);iterations.append(queries)
                    previous_z[fi]=surface_z
                    heights.append(surface_z+preload_height[fi])
                # Preserve the initial mechanical preload, rather than using
                # unreliable penetrating-distance magnitude as a motion step.
                # Multiple fingers share ONE rigid object. The minimax midpoint
                # does not pretend all different ideal heights can be realized.
                pose[2]=.5*(min(heights)+max(heights));spreads.append(max(heights)-min(heights))
                if frame==0:pose[:]=base[lane]
            excursion=np.array([normal_excursion_m(job,f,self.args.motion_period_s) for f in range(self.args.frames)])
            path[:,lane,:3]+=excursion[:,None]*self.inward[lane].cpu().numpy()
            xyz=np.concatenate((base[lane,None,:3],path[:,lane,:3]))
            speed=np.linalg.norm(np.diff(xyz,axis=0),axis=-1)/.01
            row=dict(lane=lane,finger=finger,active_fingers=fingers,behavior=job['behavior'],motion_rate=job['motion_rate'],
                initial_pad_clearance_m=float(initial[0]),initial_pad_clearances_m=initial.tolist(),
                max_clearance_error_m=float(np.max(np.abs(errors))),
                separation_probe_m=2.e-5,initial_preload_height_m=float(preload_height[0]),initial_preload_heights_m=preload_height.tolist(),
                distance_false_contact_corrections=sum(q.corrected_queries for q in distance_queries),
                max_proven_false_contact_gap_m=max(q.max_proven_false_contact_gap for q in distance_queries),
                shared_ideal_height_spread_max_m=max(spreads),
                max_geometry_iterations=max(iterations),
                height_correction_range_m=np.ptp(path[:,lane,2]-original[:,lane,2]).item(),
                path_length_m=np.linalg.norm(np.diff(xyz,axis=0),axis=-1).sum().item(),
                maximum_command_speed_m_s=speed.max().item(),
                initial_command_jump_m=np.linalg.norm(xyz[1]-xyz[0]).item())
            row.update(normal_excursion_min_max_m=[float(excursion.min()),float(excursion.max())],
                motion_direction_rad=job.get('motion_direction_rad',0.),motion_amplitude_scale=job.get('motion_amplitude_scale',1.))
            diagnostics.append(row)
            height_spreads.append(spreads)
            print(f'[WM path] lane {lane} {job["behavior"]}: {row["path_length_m"]:.3f}m, max speed {speed.max():.3f}m/s',flush=True)
        folder=collection.episode_folder(self.args,self.row)
        folder.mkdir(parents=True,exist_ok=True)
        report=dict(method='whole fingertip clearance at positive separation; initial preload height preserved',
            finger_reference='fixed q; no finger IK, loss search or force-dependent object path; one shared Z at minimax midpoint for concurrent fingers',
            bounds=dict(maximum_command_speed_m_s=self.args.max_object_command_speed,initial_command_jump_m=1e-6),lanes=diagnostics)
        report['command_gate']=all(d['maximum_command_speed_m_s']<=self.args.max_object_command_speed and d['initial_command_jump_m']<=1e-6 for d in diagnostics)
        (folder/'object_path_plan.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(folder/'object_path_plan.npz',prescribed=original,commanded=path,ideal_height_spread=np.array(height_spreads).T)
        if not report['command_gate']:
            raise ValueError('Privileged object motion discontinuous/too fast; see object_path_plan.json')
        self.object_path=torch.tensor(path,device=self.device,dtype=torch.float32)

    def prepare_motion(self,meta):
        checkpoint=getattr(self.args,'dp_checkpoint',None)
        amplitude=getattr(self.args,'perturbation_rad',0.)
        self.dp_source=None;self.nominal_packet=None
        if checkpoint is None and amplitude==0:return
        from contact_wm_nominal import FrozenDPNominal,SmoothPerturbation
        if self.args.object_surface_follow and self.object_path is None:self.plan_object_path()
        self.exploration=SmoothPerturbation(self.jobs,self.active,self.args.frames,amplitude,self.device)
        if checkpoint is not None:
            self.dp_source=FrozenDPNominal(checkpoint,self.fixed_reference_q,self.args.seed+71021)
            meta.update(no_dp=False,dp=self.dp_source.metadata,
                nominal_reference='frozen_DP_after_76_fixed_bootstrap_frames',
                action_owner='frozen_DP -> original_chunk_scheduler -> bounded_exploration -> collector_MCC -> held_position_servo')
        else:meta['action_owner']='fixed_reference -> bounded_exploration -> collector_MCC -> held_position_servo'
        meta['object_path_reference']='fixed_post_initialization_q used only to prepare object height path, independent of DP/exploration'
        meta['raw_dp_chunks']='dp_plans.npz: prediction, chunk, anchor, history, normalized and drop; raw_generator_q in H5 is the scheduler-rate nominal'
        meta['nominal_command_exploration_rad']=[-amplitude,amplitude]
        meta['exploration']=dict(kind='smooth_random_joint_offset',maximum_rad=amplitude,
            target_interval_s=.5,interpolation='quintic zero endpoint velocity/acceleration',
            seed_offset=59021,active_only=True,starts_after_dp_bootstrap=checkpoint is not None,
            same_state_branching_supported=False)
        meta['dp_input_alignment']='dp_input_*[k] is pre-step state used for command[k]; q[k] is post-step; root input valid=false'
        self.nominal_packet=dict(raw_generator_q=self.fixed_reference_q.clone(),
            scheduler_nominal_q=self.fixed_reference_q.clone(),exploration_offset_q=torch.zeros_like(self.fixed_reference_q),
            exploration_active=torch.zeros(self.n,device=self.device,dtype=torch.bool))
        if self.dp_source is not None:
            self.nominal_packet.update(dp_input_state=torch.zeros(self.n,96,device=self.device),
                dp_input_valid=torch.zeros(self.n,device=self.device,dtype=torch.bool),
                dp_virtual_palm_pose=torch.zeros(self.n,7,device=self.device),
                dp_virtual_palm_twist=torch.zeros(self.n,6,device=self.device),
                dp_planner_command=torch.zeros(self.n,6,device=self.device),
                dp_future_object_command=torch.zeros(self.n,7,device=self.device),
                dp_replanned=torch.zeros(self.n,device=self.device,dtype=torch.bool),
                dp_active=torch.zeros(self.n,device=self.device,dtype=torch.bool),
                dp_control_frame=torch.full((self.n,),-1,device=self.device))

    def desired(self,frame,extra=None):
        nominal,pose=self.reference_and_object(frame,extra)
        if getattr(self,'nominal_packet',None) is None:return nominal,pose
        if extra is not None:raise ValueError('Snapshot branches do not yet capture DP/exploration state')
        details={}
        if self.dp_source is not None:
            # At the input of frame f, the published target for f+19 is the
            # endpoint twenty control intervals ahead. Hold the last target.
            target=min(frame+self.dp_source.runtime.planner_step_frames-1,self.args.frames-1)
            future=(self.object_path[target] if self.object_path is not None else self.prescribed_object_motion(target))
            x=self.observer.latest
            nominal,details=self.dp_source.step(frame,self.robot.data.joint_pos.clone(),self.palm,
                self.actual_pose(),future,x['pad_physical_points'],x['pad_physical_inward'],x['pad_physical_found'])
            details['dp_future_object_command']=future.clone()
        start=76 if self.dp_source is not None else 0
        offset=self.exploration.values[frame-start] if frame>=start else torch.zeros_like(nominal)
        self.nominal_packet=dict(raw_generator_q=nominal.clone(),scheduler_nominal_q=nominal.clone(),
            exploration_offset_q=offset.clone(),exploration_active=(offset!=0).any(-1),**details)
        return nominal+offset,pose

    def reference_and_object(self,frame,extra=None):
        if self.args.fixed_reference:
            nominal=self.fixed_reference_q.clone()
            if extra is not None:nominal=self.mcc._clamp_q(nominal+extra)
        else:nominal,_=super().desired(frame,extra)
        if self.args.motion_owner=='palm':
            if self.palm_path is None:self.plan_palm_path()
            self.palm_goal=self.palm_path[frame].clone()
            return nominal,self.base.clone()
        if getattr(self.args,'object_surface_follow',False):
            if self.object_path is None:self.plan_object_path()
            return nominal,self.object_path[frame].clone()
        return nominal,self.prescribed_object_motion(frame)

    def snapshot(self):
        from contact_wm_branches import CollectionSnapshot
        return CollectionSnapshot(self)

    def collect_branches(self,root,root_frame,records,meta,folder):
        from contact_wm_branches import collect_branches
        return collect_branches(self,root,root_frame,records,meta,folder)

    def sample(self,phase):
        if self.args.motion_owner=='palm':self.palm=self.robot.data.root_link_pose_w.clone()
        out=super().sample(phase)
        out['object_command_pose']=out['object_pose'].copy()
        out['object_pose']=self.actual_pose().cpu().numpy()
        out['object_solver_cvel']=self.env.sim.data.cvel[:,self.target_body].cpu().numpy().copy()
        out['control_normals_world']=(self.control_normals_world if hasattr(self,'control_normals_world') else -self.inward[:,None,:].expand(-1,4,-1)).cpu().numpy().copy()
        # Hand-only actuator generalized forces, matching the 16 joint labels.
        out['actuator_force']=self.env.sim.data.qfrc_actuator[:,self.robot.data.indexing.joint_v_adr].cpu().numpy().copy()
        out['palm_command_pose']=self.palm_driver_pose.cpu().numpy().copy()
        out['palm_twist']=self.robot.data.root_link_vel_w.cpu().numpy().copy()
        if getattr(self,'nominal_packet',None) is not None:
            out.update({k:v.detach().cpu().numpy().copy() for k,v in self.nominal_packet.items()})
        return out

    def record_substep(self):
        from mujoco_warp._src.support import mul_m
        d=self.env.sim.wp_data
        stream=torch.cuda.ExternalStream(wp.get_stream(self.env.sim.wp_device).cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with wp.ScopedDevice(self.env.sim.wp_device):
            self.depth.zero_()
            wp.launch(contact_depth,dim=d.naconmax,inputs=[d.nacon,d.contact.geom,d.contact.worldid,d.contact.dist,
                self.target_mask,self.hand_mask],outputs=[self.depth])
            mul_m(self.env.sim.wp_model,d,self.mass_times_acc,d.qacc)
        torch.cuda.current_stream().wait_stream(stream)
        x=self.observer.latest
        ma=wp.to_torch(self.mass_times_acc)
        rhs=wp.to_torch(d.qfrc_smooth)+wp.to_torch(d.qfrc_constraint)
        balance=ma-rhs
        self.trace.append(dict(q=self.robot.data.joint_pos.cpu().numpy().copy(),
            qvel=self.robot.data.joint_vel.cpu().numpy().copy(),
            contact_q=self.pre_substep_qpos[:,self.joint_q_indices].copy(),
            contact_qvel=self.pre_substep_qvel[:,self.joint_v_indices].copy(),
            palm_pose=self.robot.data.root_link_pose_w.cpu().numpy().copy(),
            palm_twist=self.robot.data.root_link_vel_w.cpu().numpy().copy(),
            contact_object_pose=self.env.scene['target'].data.root_link_pose_w.cpu().numpy().copy(),
            contact_object_twist=self.env.scene['target'].data.root_link_vel_w.cpu().numpy().copy(),
            object_pose=self.actual_pose().cpu().numpy(),driver_pose=self.driver_pose.cpu().numpy(),
            object_solver_cvel=self.env.sim.data.cvel[:,self.target_body].cpu().numpy().copy(),
            pad_force=x['pad_force'].cpu().numpy().copy(),pad_point=x['pad_physical_points'].cpu().numpy().copy(),
            pad_normal=x['pad_physical_inward'].cpu().numpy().copy(),found=x['pad_physical_found'].cpu().numpy().copy(),
            depth_m=wp.to_torch(self.depth).cpu().numpy().copy(),phase=np.full(self.n,self.phase_now),
            qpos_before=self.pre_substep_qpos,qvel_before=self.pre_substep_qvel,
            qpos_after=self.env.sim.data.qpos.cpu().numpy().copy(),qvel_after=self.env.sim.data.qvel.cpu().numpy().copy(),
            dynamics_balance_max_abs=balance.abs().amax(-1).cpu().numpy().copy(),
            dynamics_balance_relative=(balance.norm(dim=-1)/(ma.norm(dim=-1)+rhs.norm(dim=-1)+1.e-6)).cpu().numpy().copy(),
            # The previous ratio is ill-conditioned near static equilibrium.
            # Retain its components so an audit cannot mistake cancellation
            # of two large forces for a large absolute mechanics violation.
            dynamics_mass_times_acc=ma.cpu().numpy().copy(),
            qfrc_smooth=wp.to_torch(d.qfrc_smooth).cpu().numpy().copy(),
            solver_mass_times_acc=wp.to_torch(d.efc.Ma).cpu().numpy().copy(),
            qfrc_constraint=wp.to_torch(d.qfrc_constraint).cpu().numpy().copy(),
            qfrc_actuator=wp.to_torch(d.qfrc_actuator).cpu().numpy().copy(),
            qacc=wp.to_torch(d.qacc).cpu().numpy().copy(),
            solver_niter=wp.to_torch(d.solver_niter).cpu().numpy().copy(),
            physics_time=self.env.sim.data.time.cpu().numpy().copy()))
        if 'pad_legacy_normal_dot_raw' in x:
            for role in ('pad','connector'):
                self.measurement_audit[role+'_normal_sign_mismatch_samples']+=int(
                    (x[role+'_physical_found'] & (x[role+'_legacy_normal_dot_raw']<-.99)).sum())
            self.measurement_audit['max_net_force_discrepancy_n']=max(
                self.measurement_audit['max_net_force_discrepancy_n'],
                float(x['pad_legacy_net_error_n'].max()),float(x['connector_legacy_net_error_n'].max()))
            for role in ('pad','connector'):
                for suffix in ('physical_geom_ids','physical_contact_id','physical_distance','legacy_normal_dot_raw'):
                    key=role+'_'+suffix;self.trace[-1][key]=x[key].cpu().numpy().copy()
            for key in ('object_physical_geom_ids','object_physical_force','object_physical_point',
                        'object_physical_normal','object_physical_distance'):
                self.trace[-1][key]=x[key].cpu().numpy().copy()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--objects',nargs='+',required=True);p.add_argument('--fingers',nargs='+',type=int,default=[0])
    p.add_argument('--active-fingers',nargs='+',type=int,
        help='2–4 fingers active together in each lane; --fingers otherwise remains separate single-finger lanes')
    p.add_argument('--initial-roots',type=Path,help='Static initial poses from prepare_contact_wm_multi_roots.py')
    p.add_argument('--load-settle-frames',type=int,default=0,
        help='Static fixed-reference MCC loading before motion; approach plus loading still limited to250 frames')
    p.add_argument('--behaviors',nargs='+',default=['slide']);p.add_argument('--frames',type=int,default=400)
    p.add_argument('--branches',action='store_true',
        help='Save full state after frame100; verify repeated continuations then collect bounded action branches')
    p.add_argument('--dp-checkpoint',type=Path,help='Frozen whole-hand DP nominal source; collector MCC unchanged')
    p.add_argument('--perturbation-rad',type=float,default=0.,help='Smooth bounded pre-MCC exploration in [0,.08]rad')
    p.add_argument('--branch-diagnostic-only',action='store_true',
        help='Measure action sensitivity even after strict future-repeat failure; branches excluded from training')
    p.add_argument('--backend',choices=['mocap_ramp','mocap_weld'],required=True);p.add_argument('--no-mcc',action='store_true')
    p.add_argument('--normal-memory-frames',type=int,default=0)
    p.add_argument('--contact-readout',choices=['matched','legacy'],default='matched')
    p.add_argument('--mcc-overforce-mode',choices=['continuous','legacy'],default='continuous',
        help='continuous retains ordinary force regulation but removes extra threshold takeover/hold/projection')
    p.add_argument('--overforce-response',choices=['record','stop'],default='record')
    p.add_argument('--solver-line-search',choices=['parallel','iterative'],default='parallel')
    p.add_argument('--disable-solver-warmstart',action='store_true')
    p.add_argument('--constraint-capacity',type=int,default=None,
        help='Optional per-environment constraint buffer allocation, >=1000; no contact-law or solver-tolerance change')
    p.add_argument('--contact-regulation',choices=['legacy_handoff','unified','reference_compliance'],default='reference_compliance',
        help='reference_compliance: fixed pregrasp plus force feedback only in contact; on loss return offset to zero, no search')
    p.add_argument('--compliance-travel-m',type=float,default=None,
        help='Optional bounded normal offset travel; recorded in effective MCC config')
    p.add_argument('--allow-moving-kinematic-diagnostic',action='store_true',
        help='allow zero-solver-velocity mocap collision motion for explicit legacy diagnosis only')
    p.add_argument('--fixed-reference',action='store_true')
    p.add_argument('--motion-owner',choices=['object','palm'],default='object')
    p.add_argument('--path-length-m',type=float,default=.12)
    p.add_argument('--surface-follow',action='store_true')
    p.add_argument('--object-surface-follow',action='store_true',
        help='Privileged fixed-reference teacher: adjust object height using collision geometry; no finger search')
    p.add_argument('--slide-amplitude-m',type=float,default=.012)
    p.add_argument('--motion-period-s',type=float,default=None,
        help='Base period before speed scaling: normal fixed-reference object collection defaults to 10s; legacy modes to 3.5s')
    p.add_argument('--motion-rate',type=float,default=None,
        help='Explicit constant analytic speed; use 1 for slow baseline. Normal fixed-reference object runs default to random 2–4x.')
    p.add_argument('--motion-rate-range',nargs=2,type=float,metavar=('MIN','MAX'),
        help='Sample one fixed speed multiplier per lane/episode; normal collection profile uses 2 4')
    p.add_argument('--randomize-motion',action='store_true',help='Seeded tangent/rotation direction, amplitude and open-loop release distance')
    p.add_argument('--max-object-command-speed',type=float,default=.10,
        help='Explicit object path preflight speed bound in m/s; no clipping')
    p.add_argument('--device',default='cuda:0');p.add_argument('--friction',type=float,default=1.);p.add_argument('--contact-multiplier',type=float,default=1.)
    p.add_argument('--seed',type=int,default=20261109);p.add_argument('--run-tag',required=True)
    a=p.parse_args();a.root=a.root.resolve()
    if a.constraint_capacity is not None and a.constraint_capacity<1000:
        p.error('--constraint-capacity must be >=1000')
    if not np.isfinite(a.perturbation_rad) or not 0<=a.perturbation_rad<=.08:
        p.error('--perturbation-rad must be in [0,.08]')
    if a.dp_checkpoint is not None or a.perturbation_rad>0:
        if a.branches:p.error('DP/exploration state not yet covered by the branch restore contract')
        if not a.fixed_reference or a.motion_owner!='object':p.error('Nominal sources require fixed initialization reference and object motion')
    if a.dp_checkpoint is not None:
        if not a.dp_checkpoint.is_file():p.error('DP checkpoint missing')
        if sorted(a.active_fingers or [])!=[0,1,2,3]:p.error('Whole-hand DP requires all four active fingers')
        if a.frames<86:p.error('DP needs 76 bootstrap frames and at least one execution interval')
    if a.branch_diagnostic_only and not a.branches:p.error('--branch-diagnostic-only requires --branches')
    if a.branches and (a.frames<160 or not a.fixed_reference or a.motion_owner!='object'):
        p.error('Branch verification requires >=160 frames, fixed reference and object motion')
    normal_object_motion=(a.fixed_reference and a.motion_owner=='object'
                          and set(a.behaviors)<=NORMAL_BEHAVIORS)
    if a.motion_period_s is None:
        a.motion_period_s=10. if normal_object_motion else 3.5
    if not np.isfinite(a.motion_period_s) or a.motion_period_s<=0:
        p.error('--motion-period-s must be finite and positive')
    if a.motion_rate is None:
        if a.motion_rate_range is None and normal_object_motion:
            a.motion_rate_range=[2.,4.]
        a.motion_rate=1.
    elif a.motion_rate_range is not None:
        p.error('Select either --motion-rate or --motion-rate-range')
    if not 0<=a.load_settle_frames<=250:p.error('Loading frames must be in [0,250]')
    if a.active_fingers is not None:
        a.active_fingers=sorted(a.active_fingers)
        if a.motion_owner!='object' or not a.fixed_reference or a.initial_roots is None:
            p.error('Concurrent contact currently requires object motion, fixed reference and --initial-roots')
    if a.motion_owner=='palm' and (not a.fixed_reference or a.behaviors!=['traverse']):
        p.error('Palm diagnostic requires --fixed-reference --behaviors traverse')
    if a.object_surface_follow and (a.motion_owner!='object' or not a.fixed_reference):
        p.error('--object-surface-follow requires --motion-owner object --fixed-reference')
    if not np.isfinite(a.motion_rate) or not 0<a.motion_rate<=16:
        p.error('--motion-rate must be finite and in (0,16]')
    if a.motion_rate!=1 and (a.motion_owner!='object' or not a.fixed_reference):
        p.error('Rate comparison requires fixed hand reference and object motion')
    if a.motion_rate_range is not None:
        lo,hi=a.motion_rate_range
        if not (np.isfinite([lo,hi]).all() and 0<lo<=hi<=16):p.error('Invalid motion-rate range')
        if not a.fixed_reference or a.motion_owner!='object' or not set(a.behaviors)<=NORMAL_BEHAVIORS:
            p.error('Random speed requires fixed-reference normal object-motion profiles')
    if a.randomize_motion and not normal_object_motion:p.error('Motion diversity requires normal object-motion profiles')
    if not np.isfinite(a.max_object_command_speed) or a.max_object_command_speed<=0:
        p.error('--max-object-command-speed must be finite and positive')
    if a.motion_owner=='object' and a.backend!='mocap_weld' and not a.allow_moving_kinematic_diagnostic:
        p.error('Moving object contact dynamics require --backend mocap_weld; mocap_ramp has no solver surface velocity')
    rows=json.loads((a.root/'object_manifest.json').read_text())['objects'];collection.Probe=MotionProbe
    # Save the substep trace before closing the environment via the wrapper.
    original=MotionProbe.initialize
    def initialize(self):
        original_close=self.env.close
        def close():
            if getattr(self,'dp_source',None) is not None:
                self.dp_source.save(collection.episode_folder(a,self.row))
            if self.trace:
                folder=collection.episode_folder(a,self.row)
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'substeps.npz',**{k:np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
                import mujoco
                mujoco.mj_saveModel(self.env.sim.mj_model,str(folder/'physics_model.mjb'),None)
            original_close()
        self.env.close=close
        records=original(self)
        return self.finish_initialization(records)
    MotionProbe.initialize=initialize
    for name in a.objects:
        row=dict(next(r for r in rows if r['object_id']==name));row['motion_backend']=a.backend
        row['solver_line_search']=a.solver_line_search
        row['solver_warmstart']=not a.disable_solver_warmstart
        row['constraint_capacity']=a.constraint_capacity
        if a.motion_owner=='palm':row['palm_support_backend']='weld'
        row['object_motion_protocol']=dict(driver_update_s=.001,object_weld_time_constant_s=.005 if a.backend=='mocap_weld' else None,
            motion_owner=a.motion_owner,palm_path_length_m=a.path_length_m if a.motion_owner=='palm' else None,
            palm_gt_surface_height=a.surface_follow,palm_support_time_constant_s=.005 if a.motion_owner=='palm' else None,
            object_gt_surface_height=a.object_surface_follow,
            mcc_enabled=not a.no_mcc,normal_memory_frames=a.normal_memory_frames,
            contact_readout=a.contact_readout,mcc_overforce_mode=a.mcc_overforce_mode,
            solver_line_search=a.solver_line_search,
            solver_warmstart=not a.disable_solver_warmstart,
            contact_regulation=a.contact_regulation,compliance_travel_m=a.compliance_travel_m,
            load_settle_frames_requested=a.load_settle_frames,initialization_total_frame_cap=250,
            overforce_response=a.overforce_response,
            substep_force_alignment='forces/contact geometry from qpos_before/qvel_before; qpos_after/qvel_after are integrated outputs; H5 endpoint sensors are refreshed by forward',
            nominal_reference='fixed_post_initialization_q' if a.fixed_reference else 'q0_plus_sinusoidal_joint_excitation',
            ramp_s=.3,slide_direction='positive for odd job seed, negative for even',
            slide_amplitude_m=a.slide_amplitude_m if a.fixed_reference else .012,
            period_s=a.motion_period_s if a.fixed_reference else 3.5,
            motion_rate=a.motion_rate,max_object_command_speed_m_s=a.max_object_command_speed,
            motion_rate_range=a.motion_rate_range,speed_sampling='one constant multiplier per trajectory, independent seed; exact multiplier in jobs',
            motion_randomization=a.randomize_motion,
            normal_release='preissued smooth object retreat/return after height compensation; no loss-triggered search',
            physical_motion_period_s=None if a.motion_rate_range else a.motion_period_s/a.motion_rate,
            physical_startup_ramp_s=None if a.motion_rate_range else .3/a.motion_rate,
            physical_control_dt_s=.01,physics_substep_dt_s=.001,
            object_velocity_semantics='dynamic solver body' if a.backend=='mocap_weld' else 'mocap velocity still zero')
        print(json.dumps(collection.run(a,row)),flush=True)
