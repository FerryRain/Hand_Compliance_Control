"""P0 contact dynamics collection: mocap palm, prescribed object, real fingers.

Structured nominal actions -> unchanged batched MCC -> held position actuator.
Single-finger means one active controller, NOT deleting other hand collisions.
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from contact_wm_geometry import target_spec

SCRIPT=Path(__file__).resolve().parent
SOURCE_CONFIG=SCRIPT.parent/'data/mocap_collection_roundtrip_20261109/full_v2_collect.json'
MCC_KEYS=('anchor','anchor_valid','force_filtered','filter_valid','contact_active','force_offset',
          'hold','contact_phase','loss_streak','recontact_streak','transient_offset',
          'previous_command','reference_q','previous_nominal','force_setpoint')
SIM_KEYS=('qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied','time')


def collision_provenance():
    """Record the actual loaded collision implementation, including opt-in copies.

    Version strings alone cannot distinguish a diagnostic backport. This only
    records provenance; it does not select a backend or grant data admission.
    """
    import mujoco_warp._src.collision_gjk as gjk
    path=Path(gjk.__file__).resolve()
    protocol=path.parents[2]/'protocol.json'
    result=dict(gjk_source_path=str(path),gjk_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    convex=path.with_name('collision_convex.py')
    result.update(convex_source_path=str(convex),convex_source_sha256=hashlib.sha256(convex.read_bytes()).hexdigest())
    if protocol.exists():
        result['isolated_backend_protocol']=json.loads(protocol.read_text())
    return result


def job_fingers(job):
    return job.get('active_fingers',[job['finger']])


def episode_folder(args,row):
    active=getattr(args,'active_fingers',None)
    scope='single_finger' if active is None else 'whole_hand' if len(active)==4 else 'multi_finger'
    suffix='' if active is None else '_active'+''.join(map(str,active))
    return args.root/'episodes'/scope/args.run_tag/f'{row["object_id"]}_mu{args.friction:g}_k{args.contact_multiplier:g}{suffix}'


def collection_jobs(args,row):
    active=getattr(args,'active_fingers',None)
    if active is None:
        return [dict(finger=f,behavior=b,seed=args.seed+f*101+i) for f in args.fingers for i,b in enumerate(args.behaviors)],None
    if len(set(active))!=len(active) or not 2<=len(active)<=4 or any(f not in range(4) for f in active):
        raise ValueError('Simultaneous collection needs 2–4 distinct fingers in [0,3]')
    path=getattr(args,'initial_roots',None)
    if path is None:raise ValueError('Simultaneous collection requires a verified static initial-roots manifest')
    roots=json.loads(Path(path).read_text())['roots']
    roots=[r for r in roots if r['object_id']==row['object_id'] and r['active_fingers']==active
           and r['status']=='static_accepted_not_physically_verified']
    if not roots:raise ValueError(f'No static root for {row["object_id"]} / {active}')
    for r in roots:
        if r['mesh_sha256']!=row['mesh_sha256']:raise ValueError('Initial root mesh differs from collection mesh')
        if not np.allclose(r['palm_pose'],[0,0,.6,1,0,0,0],atol=1e-9):raise ValueError('Unsupported root palm frame')
    jobs=[dict(finger=active[0],active_fingers=active,behavior=b,seed=r['seed']+i,
        root_index=k,root_seed=r['seed'],root_id=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest())
        for k,r in enumerate(roots) for i,b in enumerate(args.behaviors)]
    return jobs,roots


def all_active_loaded(loaded,active):
    return (loaded|~active).all(-1)&active.any(-1)


def selected_contacts(contact,active):
    """All requested fingers, never the union or just the first finger."""
    return np.all(contact|~active,axis=-1)&np.any(active,axis=-1)


def controller_state_fields(mcc,q_template):
    """Keep optional controller memory with validity; never drop an entire field
    because its first cold-start value was None. Zeros only encode absent memory,
    not sanitized physical observations, and must be read with the validity bit.
    """
    result={}
    for key in MCC_KEYS:
        value=getattr(mcc,key)
        if isinstance(value,torch.Tensor):result['mcc_'+key]=value.clone()
        elif key in ('previous_command','reference_q','previous_nominal') and value is None:
            result['mcc_'+key]=torch.zeros_like(q_template)
        else:raise TypeError(f'Unrecordable MCC state {key}: {type(value)}')
        result['mcc_'+key+'_valid']=torch.full((len(q_template),),value is not None,
                                              device=q_template.device,dtype=torch.bool)
    return result


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def make_env(root,row,batch,device,friction,contact_multiplier,*,matched_contacts=False):
    from replay_inverted import replay_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch,ContactSensorCfg
    from collection_force_regulation import CollectionForceObserver
    from palm_support import KinematicPalmSupport, WeldPalmSupport
    from replay_supported_collection import preserve_parent_filter
    from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import _load_mcc_leaphand_spec
    cfg=replay_env_cfg(num_envs=batch,physics_substeps=10,hand_stiffness=35.,hand_damping=.2,hand_effort_limit=.8)
    if row.get('constraint_capacity') is not None:
        capacity=int(row['constraint_capacity'])
        if capacity<1000:raise ValueError('Do not reduce the reference constraint buffer')
        cfg.sim.njmax=capacity
    # This selects a CUDA solver implementation, not a different contact law.
    # Keep the legacy choice unless the collection protocol explicitly opts in.
    cfg.sim.ls_parallel=row.get('solver_line_search','parallel')=='parallel'
    if not row.get('solver_warmstart',True):
        cfg.sim.mujoco.disableflags=(*cfg.sim.mujoco.disableflags,'warmstart')
    cfg.scene.entities['target'].spec_fn=lambda:target_spec(root,row,friction,contact_multiplier)
    source=_load_mcc_leaphand_spec();props={g.name:{k:np.asarray(getattr(g,k)).copy() for k in
        ('friction','solref','solimp','priority','solmix','margin','gap','condim')} for g in source.geoms}
    original=cfg.scene.entities['robot'].spec_fn
    def spec():
        s=original()
        for g in s.geoms:
            for k,v in props.get(g.name,{}).items():
                if v.ndim:getattr(g,k)[:]=v
                else:setattr(g,k,v.item())
            g.friction[0]=friction
            g.solref[:]=[-20000*contact_multiplier,-400*contact_multiplier**.5]
        return s
    cfg.scene.entities['robot'].spec_fn=spec
    observer_type=CollectionForceObserver
    if matched_contacts:
        from contact_wm_physics import MatchedCollectionObserver
        observer_type=MatchedCollectionObserver
    obs=observer_type(controller_scope='pad');obs.configure(cfg)
    before=cfg.scene.entities['robot'].spec_fn
    driver=WeldPalmSupport(.005) if row.get('palm_support_backend')=='weld' else KinematicPalmSupport()
    driver.configure(cfg)
    after=cfg.scene.entities['robot'].spec_fn
    def preserve():
        s=after();preserve_parent_filter(before(),s);return s
    cfg.scene.entities['robot'].spec_fn=preserve
    cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='all_hand_object',
        primary=ContactMatch(mode='subtree',pattern='^palm_lower$',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='^target_ball$',entity='target'),
        fields=('found','force'),reduce='maxforce',num_slots=1))
    for f,body in enumerate(('mcp_joint','mcp_joint_2','mcp_joint_3','pip_4')):
        cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name=f'finger_all_{f}',
            primary=ContactMatch(mode='subtree',pattern='^'+body+'$',entity='robot'),
            secondary=ContactMatch(mode='body',pattern='^target_ball$',entity='target'),
            fields=('found','force'),reduce='maxforce',num_slots=1))
    cfg.sim.contact_sensor_maxmatch=512
    env=ManagerBasedRlEnv(cfg=cfg,device=device);env.reset();driver.bind(env)
    return env,driver,obs


class Probe:
    def __init__(self,args,row):
        import mujoco
        from batched_fullhand_mcc import BatchedFullHandMCC
        from surface_mcc_finger import FullHandMCCFingerConfig
        from feedback_self_contact import SelfContactDiagnostic
        self.args=args;self.row=row;self.device=args.device;self.start=time.monotonic()
        self.jobs,self.initial_roots=collection_jobs(args,row)
        self.n=len(self.jobs);self.env,self.driver,self.observer=make_env(args.root,row,self.n,args.device,args.friction,args.contact_multiplier,
            matched_contacts=getattr(args,'contact_readout','legacy')=='matched')
        self.robot=self.env.scene['robot'];self.origins=self.env.scene.env_origins
        target_body=mujoco.mj_name2id(self.env.sim.mj_model,mujoco.mjtObj.mjOBJ_BODY,
            'target/object_driver' if row.get('motion_backend')=='mocap_weld' else 'target/target_ball')
        self.target=int(self.env.sim.mj_model.body_mocapid[target_body])
        self.config=FullHandMCCFingerConfig(**json.loads(SOURCE_CONFIG.read_text())['mcc_config'])
        reference_compliance=getattr(args,'contact_regulation','legacy_handoff') in ('unified','reference_compliance')
        if reference_compliance:
            # Initialized nominal already touches the surface. Loading is
            # produced by force feedback, not an additional 3mm target jump.
            self.config=replace(self.config,nominal_surface_preload=0.)
        if getattr(args,'compliance_travel_m',None) is not None:
            travel=float(args.compliance_travel_m)
            if not 0.<travel<=.02:raise ValueError('Collection compliance travel must be in (0,20] mm')
            self.config=replace(self.config,max_normal_offset=travel,thumb_max_inward_offset=travel,
                force_servo_max_outward_offset=travel,thumb_max_outward_offset=travel)
        self.mcc=BatchedFullHandMCC(self.n,args.device,self.config,
            overforce_takeover=getattr(args,'mcc_overforce_mode','legacy')=='legacy',
            continuous_contact_regulation=reference_compliance)
        self.active=torch.zeros((self.n,4),device=args.device,dtype=torch.bool)
        self.fingers=torch.tensor([j['finger'] for j in self.jobs],device=args.device)
        for lane,job in enumerate(self.jobs):self.active[lane,job_fingers(job)]=True
        self.q0=torch.tensor([.10,0.,.15,.15]*3+[.15,1.5,.15,.15],device=args.device).repeat(self.n,1)
        external_surface=row['family'].startswith('large_')
        under_palm=row.get('initialization')=='under_palm'
        if external_surface and not under_palm:self.q0[:]=torch.tensor([.75,0.,.50,.35]*3+[.5,1.5,.45,.3],device=args.device)
        for i,job in enumerate(self.jobs):
            rng=np.random.default_rng(job['seed']);f=job['finger']
            pose=[.1,0. if f<3 else 1.5,.15,.15] if external_surface else [.85,0. if f<3 else 1.3,.65,.65]
            if under_palm:pose=[.65,0. if f<3 else 1.3,.35,.25]
            self.q0[i,f*4:f*4+4]=torch.tensor(np.array(pose)+rng.uniform(-.04,.04,4),device=args.device)
        self.q0=self.mcc._clamp_q(self.q0);self.command=self.q0.clone();self.nominal=self.q0.clone()
        if self.initial_roots is not None:
            self.q0=torch.tensor([self.initial_roots[j['root_index']]['q'] for j in self.jobs],device=args.device,dtype=torch.float32)
            if not torch.equal(self.q0,self.mcc._clamp_q(self.q0)):raise ValueError('Static root q exceeds actuator joint limits')
        self.palm=torch.zeros((self.n,7),device=args.device);self.palm[:,2]=.6;self.palm[:,3]=1.;self.palm[:,:3]+=self.origins
        self.driver.write_root_state(torch.cat((self.palm,torch.zeros((self.n,6),device=args.device)),1))
        self.robot.write_joint_state_to_sim(self.q0,torch.zeros_like(self.q0));self.robot.set_joint_position_target(self.q0);self.robot.write_data_to_sim()
        # Reject invalid starting geometry before launching any physics steps.
        from contact_wm_initialization import place
        model=self.env.sim.mj_model
        rotations=[];positions=[];directions=[];anchors=[];self.placement=[]
        for i,job in enumerate(self.jobs):
            if self.initial_roots is None:
                q,pos,r,direction,anchor,diagnostic=place(model,args.root,row,self.q0[i].cpu().numpy(),
                    job['finger'],self.driver.mocap_id,self.target,job['seed'])
            else:
                root=self.initial_roots[job['root_index']];q=np.array(root['q']);pose=np.array(root['object_pose'])
                pos=pose[:3];r=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                direction=np.array(root['approach_axis']);anchor=np.zeros(3)
                diagnostic=dict(root,manifest_sha256=hashlib.sha256(Path(args.initial_roots).read_bytes()).hexdigest())
            self.q0[i]=torch.tensor(q,device=args.device)
            positions.append(pos);rotations.append(r);directions.append(direction);anchors.append(anchor);self.placement.append(diagnostic)
        self.command=self.q0.clone();self.nominal=self.q0.clone()
        self.robot.write_joint_state_to_sim(self.q0,torch.zeros_like(self.q0));self.robot.set_joint_position_target(self.q0);self.robot.write_data_to_sim()
        self.anchors=np.array(anchors)
        self.rot=torch.tensor(np.array(rotations),device=args.device,dtype=torch.float32)
        self.inward=torch.tensor(np.array(directions),device=args.device,dtype=torch.float32)
        self.object=torch.tensor(np.c_[positions,Rotation.from_matrix(rotations).as_quat()[:,[3,0,1,2]]],device=args.device,dtype=torch.float32)
        self.object[:,:3]+=self.origins;self.write_object(self.object)
        self.env.sim.forward();self.env.sim.sense();self.observer.read(self.env,None)
        self.base=self.object.clone();self.tick=0;self.done_reason=None;self.initialized=torch.zeros(self.n,device=args.device,dtype=torch.bool)
        self.object_twist=torch.zeros((self.n,6),device=args.device)
        self.peak=torch.zeros(self.n,device=args.device);self.self_peak=self.peak.clone();self.held_error=self.peak.clone();self.substeps=0
        self.own=SelfContactDiagnostic(self.env,np.isin(model.geom_bodyid,self.robot.data.indexing.body_ids.cpu().numpy()).astype(np.int32))
        self.original_step=self.env.sim.step
        def substep():
            self.original_step();self.substeps+=1;self.env.sim.sense();self.observer.read(self.env,None)
            sensor=self.env.scene['all_hand_object'];sensor.update(0.)
            self.peak=torch.maximum(self.peak,torch.maximum(self.observer.safety_peak_force_n,sensor.data.force.norm(dim=-1).amax(-1)))
            self.self_peak=torch.maximum(self.self_peak,self.own.read())
            self.held_error=torch.maximum(self.held_error,(self.robot.data.joint_pos_target-self.command).abs().amax(-1))
            for value in (self.env.sim.data.qpos,self.env.sim.data.qvel,self.env.sim.data.qacc,
                          self.observer.latest['pad_force'],self.observer.latest['connector_force']):
                if not torch.isfinite(value).all():raise RuntimeError('nonfinite_physics_or_measurement')
            if getattr(self.args,'overforce_response','stop')=='stop' and bool(((self.peak>100)|(self.self_peak>100)).any()):
                raise InterruptedError('force_stop_100N')
        self.env.sim.step=substep

    def write_object(self,value):
        self.env.sim.data.mocap_pos[:,self.target]=value[:,:3];self.env.sim.data.mocap_quat[:,self.target]=value[:,3:]

    def sample(self,phase):
        self.env.sim.sense();self.observer.read(self.env,None);x=self.observer.latest
        finger_sensors=[self.env.scene[f'finger_all_{i}'] for i in range(4)]
        for sensor in finger_sensors:sensor.update(0.)
        all_part_found=torch.stack([s.data.found[:,0]>0 for s in finger_sensors],1)
        strict=x['pad_found']&(x['pad_force'].norm(dim=-1)>=.05)
        out=dict(q=self.robot.data.joint_pos.clone(),qvel=self.robot.data.joint_vel.clone(),
            actuator_force=self.env.sim.data.qfrc_actuator[:].clone(),raw_generator_q=self.nominal.clone(),pre_mcc_q=self.nominal.clone(),
            post_mcc_q=self.command.clone(),actuator_target_q=self.robot.data.joint_pos_target.clone(),
            palm_pose=self.palm.clone(),object_pose=self.object.clone(),strict=strict,found=x['pad_found']|x['connector_found'],
            palm_twist=torch.zeros_like(self.object_twist),object_command_twist=self.object_twist.clone(),
            tracking_error=self.robot.data.joint_pos-self.command,
            joint_margin_rad=torch.stack((self.robot.data.joint_pos-self.mcc.lower,self.mcc.upper-self.robot.data.joint_pos),-1),
            pad_net_force_world=x['pad_force'].clone(),pad_representative_point_world=x['pad_points'].clone(),
            pad_point_world=x['pad_physical_points'].clone(),pad_normal_world=x['pad_physical_inward'].clone(),
            pad_force_contact_frame=x['pad_physical_force'].clone(),pad_physical_found=x['pad_physical_found'].clone(),
            connector_net_force_world=x['connector_force'].clone(),connector_found=x['connector_found'].clone(),
            active_finger=self.active.clone(),unintended_finger_contact=all_part_found&~self.active,
            any_part_finger_found=all_part_found,any_part_finger_peak_force_n=torch.stack([s.data.force[:,0].norm(dim=-1) for s in finger_sensors],1),
            sim_time_s=self.env.sim.data.time[:].clone(),
            object_peak_n=self.peak.clone(),self_peak_n=self.self_peak.clone(),held_error=self.held_error.clone(),
            phase=torch.full((self.n,),phase,device=self.device),dt=torch.full((self.n,),self.substeps*.001,device=self.device),
            initialized=self.initialized.clone())
        out.update(controller_state_fields(self.mcc,self.robot.data.joint_pos))
        # Matched tangent basis for reconstructing the strongest-point force.
        out['pad_tangent_world']=(x['pad_physical_tangent'].clone() if 'pad_physical_tangent' in x else
            torch.stack([self.env.scene[n].data.tangent[:,0] for n in self.observer.guard_names['pad']],1).clone())
        for role in ('pad','connector'):
            for suffix in ('physical_geom_ids','physical_contact_id','physical_distance',
                           'legacy_normal_dot_raw','legacy_net_error_n'):
                key=role+'_'+suffix
                if key in x:out[key]=x[key].clone()
        for key in ('object_physical_geom_ids','object_physical_force','object_physical_point',
                    'object_physical_normal','object_physical_distance'):
            if key in x:out[key]=x[key].clone()
        return {k:v.detach().cpu().numpy() for k,v in out.items()}

    def step(self,nominal,object_pose,phase,use_mcc=True):
        from palm_support import measured_pose_difference_twist
        self.object_twist=measured_pose_difference_twist(object_pose,self.object,.01)
        self.nominal=nominal.clone();self.object=object_pose.clone();self.write_object(self.object)
        self.env.sim.forward();self.env.sim.sense();self.observer.read(self.env,None);x=self.observer.latest
        if use_mcc:
            # Sensor normal on the hand points into the object. The force servo
            # takes outward object normals; use the commanded approach axis only
            # while no current physical normal exists.
            fallback=-self.inward[:,None,:].expand(-1,4,-1)
            memory_frames=getattr(self.args,'normal_memory_frames',0)
            continuous=self.mcc.continuous_contact_regulation
            if continuous:memory_frames=max(memory_frames,1)
            if memory_frames:
                if not hasattr(self,'last_contact_normal'):
                    self.last_contact_normal=fallback.clone()
                    self.contact_normal_age=torch.full_like(x['pad_physical_found'],memory_frames+1,dtype=torch.long)
                observed=x['pad_physical_found']
                self.last_contact_normal=torch.where(observed[...,None],-x['pad_physical_inward'],self.last_contact_normal)
                self.contact_normal_age=torch.where(observed,0,self.contact_normal_age+1)
                if continuous:
                    # Hold the last measured direction until another contact
                    # provides a new one. No delayed switch to the initial
                    # approach axis while a displacement offset is held.
                    fallback=self.last_contact_normal
                else:
                    fallback=torch.where((self.contact_normal_age<=memory_frames)[...,None],self.last_contact_normal,fallback)
            outward=torch.where(x['pad_physical_found'][...,None],-x['pad_physical_inward'],fallback)
            self.control_normals_world=outward.clone()
            self.command,_=self.mcc.update(q_live=self.robot.data.joint_pos,palm_pose_world=self.palm,
                force_world=x['pad_force'],found=x['pad_found']&self.active,
                surface_points_world=x['pad_points'],surface_normals_world=outward,nominal_posture_q=nominal,
                contact_points_world=x['pad_points'],collision_normals_outward_world=outward,
                collision_points_world=x['pad_physical_points'])
            self.command=torch.where(self.active[...,None].expand(-1,-1,4).reshape(self.n,16),self.command,self.q0)
        else:self.command=nominal.clone()
        if not torch.isfinite(self.command).all():raise RuntimeError('nonfinite_physics_or_measurement: command')
        self.peak.zero_();self.self_peak.zero_();self.held_error.zero_();self.substeps=0
        try:self.env.step((self.command-self.robot.data.joint_pos)/.08)
        except InterruptedError as e:self.done_reason=str(e)
        self.tick+=1
        return self.sample(phase)

    def initialize(self):
        records=[self.sample(0)];streak=torch.zeros(self.n,device=self.device,dtype=torch.long)
        for frame in range(250):
            x=self.observer.latest;loaded=(x['pad_force'].norm(dim=-1)>=.2)&x['pad_found']&self.active
            hit=all_active_loaded(loaded,self.active)
            pose=self.object.clone();pose[:,:3]-=self.inward*(~hit)[:,None]*.0001
            record=self.step(self.q0,pose,0,use_mcc=False)
            # Completion refers to the recorded endpoint, not the input to
            # the last physics step (which can already have lost contact).
            endpoint=self.observer.latest
            endpoint_loaded=(endpoint['pad_force'].norm(dim=-1)>=.2)&endpoint['pad_found']&self.active
            streak=torch.where(all_active_loaded(endpoint_loaded,self.active),streak+1,0)
            self.initialized=streak>=15
            record['initialized']=self.initialized.cpu().numpy().copy()
            records.append(record)
            if self.done_reason or bool(self.initialized.all()):break
        self.base=self.object.clone();self.mcc.reset()
        return records

    def desired(self,frame,extra=None):
        t=frame*.01;pose=self.base.clone();nom=self.q0.clone()
        for i,j in enumerate(self.jobs):
            f=j['finger'];b=j['behavior'];sign=1 if j['seed']%2 else -1
            ramp=min(t/.3,1.)**2*(3-2*min(t/.3,1.))
            local=np.array([0.,0.,0.])
            if b=='slide':local[0]=sign*.012*np.sin(2*np.pi*t/3.5)*ramp
            if b=='turn':local[:2]=[.009*np.sin(2*np.pi*t/3),.006*(1-np.cos(2*np.pi*t/3))]
            if b=='press_release':local[2]=.001*np.sin(2*np.pi*t/4)-.009*np.sin(np.pi*np.clip((t-1.2)/1.4,0,1))**2
            if b=='pulse':local[0]=sign*.005*np.sin(2*np.pi*t/3)
            pose[i,:3]+=self.rot[i]@torch.tensor(local,device=self.device,dtype=torch.float32)
            if b=='turn':
                relative=Rotation.from_rotvec([.08*np.sin(2*np.pi*t/3),0,.2*np.sin(2*np.pi*t/4)]).as_matrix()
                r=self.rot[i].cpu().numpy()@relative;anchor=self.anchors[i]
                pose[i,:3]+=self.rot[i]@torch.tensor(anchor,device=self.device,dtype=torch.float32)-torch.tensor(r@anchor,device=self.device,dtype=torch.float32)
                pose[i,3:]=torch.tensor(Rotation.from_matrix(r).as_quat()[[3,0,1,2]],device=self.device,dtype=torch.float32)
            # 20Hz candidate, zero-order hold over the five 100Hz servo ticks.
            ta=(frame//5)*.05;amplitude=.035 if b!='pulse' else .07
            ar=min(ta/.3,1.);ar=ar*ar*(3-2*ar)
            delta=amplitude*np.sin(2*np.pi*ta/(1.3 if b=='pulse' else 2.7)+np.arange(4)*.7)*ar
            for active_f in job_fingers(j):
                nom[i,4*active_f:4*active_f+4]+=torch.tensor(delta,device=self.device,dtype=torch.float32)
        if extra is not None:nom+=extra
        return self.mcc._clamp_q(nom),pose

    def snapshot(self):
        return dict(sim={k:getattr(self.env.sim.data,k)[:].clone() for k in SIM_KEYS},
            mcc={k:getattr(self.mcc,k).clone() if isinstance(getattr(self.mcc,k),torch.Tensor) else copy.deepcopy(getattr(self.mcc,k)) for k in MCC_KEYS},
            object=self.object.clone(),command=self.command.clone(),nominal=self.nominal.clone(),tick=self.tick,
            peak=self.peak.clone(),self_peak=self.self_peak.clone(),held_error=self.held_error.clone(),object_twist=self.object_twist.clone(),
            episode_length=self.env.episode_length_buf.clone(),
            actions={k:getattr(self.env.action_manager,k).clone() for k in ('_action','_prev_action','_prev_prev_action')},
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.device))

    def restore(self,s):
        for k,v in s['sim'].items():getattr(self.env.sim.data,k)[:]=v
        for k,v in s['mcc'].items():setattr(self.mcc,k,v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v))
        self.object=s['object'].clone();self.command=s['command'].clone();self.nominal=s['nominal'].clone();self.tick=s['tick'];self.done_reason=None
        for k in ('peak','self_peak','held_error','object_twist'):setattr(self,k,s[k].clone())
        self.env.episode_length_buf[:]=s['episode_length']
        for k,v in s['actions'].items():getattr(self.env.action_manager,k)[:]=v
        torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state(s['cuda_rng'],self.device)
        self.robot.set_joint_position_target(self.command);self.robot.write_data_to_sim()
        self.env.sim.forward();self.env.sim.sense();self.observer.read(self.env,None)
        # Forward may update acceleration buffers; restore warmstart last.
        self.env.sim.data.qacc_warmstart[:]=s['sim']['qacc_warmstart']


def export(path,records,meta):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=set.intersection(*(set(r) for r in records));data={k:np.stack([r[k] for r in records]) for k in sorted(keys)}
    if not all(np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite record')
    with h5py.File(path,'x') as h:
        raw=h.create_group('raw_100hz')
        for k,v in data.items():raw.create_dataset(k,data=v,compression='gzip')
        # 20Hz views retain all five executed commands, not a fictitious
        # constant post-MCC action. Exclude partial terminal control intervals.
        starts=np.arange(0,len(records)-5,5);valid=np.array([np.all(np.abs(data['dt'][s+1:s+6]-.01)<1e-7) for s in starts],dtype=bool)
        starts=starts[valid];view=h.create_group('view_20hz');view['state_index']=starts;view['next_state_index']=starts+5
        view['actuator_q_sequence']=np.stack([data['actuator_target_q'][s+1:s+6] for s in starts]) if len(starts) else np.empty((0,5,len(meta['jobs']),16))
        view['pre_mcc_q_sequence']=np.stack([data['pre_mcc_q'][s+1:s+6] for s in starts]) if len(starts) else np.empty((0,5,len(meta['jobs']),16))
        view['time_s']=np.cumsum(data['dt'][:,0])[starts]
        if len(records)>1:
            contact=data['strict'];events=h.create_group('events')
            events['contact_onset']=(~contact[:-1])&contact[1:];events['contact_loss']=contact[:-1]&(~contact[1:])
            events['recontact']=(~contact[:-1])&contact[1:]&np.maximum.accumulate(contact,axis=0)[:-1]
            events['overload_100n']=(data['object_peak_n']>100)|(data['self_peak_n']>100)
        h.attrs['metadata']=json.dumps(meta)
    active=np.zeros((len(meta['jobs']),4),dtype=bool)
    for lane,job in enumerate(meta['jobs']):active[lane,job_fingers(job)]=True
    n=len(active);selected=selected_contacts(data['strict'],active)
    selected_found=selected_contacts(data['found'],active)
    return dict(path=str(path),steps_100hz=len(records)-1,transitions_20hz=len(starts)*n,
        initialized=np.asarray(records[-1]['initialized']).astype(bool).tolist(),
        active_strict=selected.mean(0).tolist(),all4_strict=data['strict'].all(-1).mean(0).tolist(),
        all4_found=data['found'].all(-1).mean(0).tolist(),active_found=selected_found.mean(0).tolist(),
        active_fingers=active.tolist(),per_finger_strict=data['strict'].mean(0).tolist(),per_finger_found=data['found'].mean(0).tolist(),
        contact_summary_semantics='Includes initial endpoint; active_strict/found require ALL requested fingers; per_finger arrays include passive fingers too. Offline motion audits exclude initial endpoint.',
        unintended_contact_fraction=data['unintended_finger_contact'].any(-1).mean(0).tolist(),
        force_peak=float(data['object_peak_n'].max()),self_peak=float(data['self_peak_n'].max()),
        held_error=float(data['held_error'].max()),contact_loss_events=int((selected[:-1]&~selected[1:]).sum()),
        contact_onset_events=int((~selected[:-1]&selected[1:]).sum()),terminal_reason=meta.get('terminal_reason'))


def motion_initialization_gate(probe):
    """Keep the original contact gate unless a collector defines WM readiness.

    The optional hook cannot bypass a numerical/physical abort. It also must
    not rewrite the initialized contact mask to manufacture teacher success.
    """
    if hasattr(probe,'motion_initialization_check'):
        result=probe.motion_initialization_check()
    else:
        result=dict(ready=bool(probe.initialized.all()),policy='all_active_contact_ready',
                    contact_ready_per_lane=probe.initialized.cpu().tolist(),
                    reason=None if bool(probe.initialized.all()) else 'contact_loading_timeout')
    if probe.done_reason:
        result={**result,'ready':False,'reason':probe.done_reason}
    return result


def run(args,row):
    from held_relative_target import install_held_relative_targets
    undo=install_held_relative_targets();probe=None;folder=episode_folder(args,row)
    folder.mkdir(parents=True,exist_ok=True)
    try:
        probe=Probe(args,row);init=probe.initialize()
        meta=dict(schema='contact_wm_p0_v2',object_id=row['object_id'],split=row['split'],jobs=probe.jobs,
            seed=args.seed,mode=('static_object_direct_palm_path' if getattr(args,'motion_owner','object')=='palm' else 'mocap_fixed_palm_prescribed_object'),action_owner='structured_nominal -> MCC_100hz (mode recorded below) -> held_position_servo_1000hz',
            no_dp=True,no_future_waypoint_planning=True,nonactive_collisions_retained=True,
            nominal_command_exploration_rad=[0.,0.] if getattr(args,'fixed_reference',False) else [.035,.07],mcc_config=asdict(probe.config),placement=probe.placement,
            mcc_overforce_mode=getattr(args,'mcc_overforce_mode','legacy'),
            contact_regulation=getattr(args,'contact_regulation','legacy_handoff'),
            overforce_response=getattr(args,'overforce_response','stop'),
            contact_readout=getattr(args,'contact_readout','legacy'),
            force_convention='pad/connector net world vectors are force ON OBJECT, opposite force on hand; strongest contact frame normal points hand -> object',
            object_geometry=row,
            model_action_contract='closed-loop model conditions on proposed pre-MCC sequence + current MCC state; future executed actuator sequence is a plant-model input or a prediction label, never a causal GPC input',
            physics=dict(friction=args.friction,contact_multiplier=args.contact_multiplier,calibrated=False,hand_kp=35.,hand_kd=.2,effort_nm=.8,
                constraint_capacity_per_env=int(probe.env.sim.wp_data.njmax)),
            collision_implementation=collision_provenance(),
            command_time_alignment='post_mcc_q[k] applied to state[k-1] to produce state[k]; raw_generator fields recorded separately',
            optional_controller_state='mcc_*_valid indicates whether memory exists; zero payload with valid=false is absent memory, never a physical observation',
            coordinate_frame='world including lane origins; store env_origins for local conversion',env_origins=probe.origins.cpu().tolist(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if hasattr(probe.observer,'geom_names'):meta['geom_names']=probe.observer.geom_names
        meta['collection_scope']='single_finger' if probe.initial_roots is None else 'whole_hand' if bool(probe.active.all()) else 'multi_finger'
        meta['active_fingers_per_lane']=[job_fingers(j) for j in probe.jobs]
        meta['load_settle_frames_executed']=getattr(probe,'load_settle_frames_executed',0)
        meta['motion_initialization_check']=motion_initialization_gate(probe)
        meta['requested_motion_frames']=args.frames
        report=dict(object_id=row['object_id'],initialization=export(folder/'initialization.h5',init,meta))
        if not meta['motion_initialization_check']['ready']:
            report['status']='initialization_failed';report['reason']=meta['motion_initialization_check']['reason']
            save_json(folder/'summary.json',report);return report
        if hasattr(probe,'prepare_motion'):probe.prepare_motion(meta)
        records=[probe.sample(1)];records[0]['dt'][:]=0;root=None;root_frame=None
        for f in range(args.frames):
            try:
                nom,pose=probe.desired(f)
                record=probe.step(nom,pose,1)
                if not all(np.isfinite(v).all() for v in record.values()):
                    raise RuntimeError('nonfinite_physics_or_measurement: exported state')
                records.append(record)
            except RuntimeError as error:
                if not str(error).startswith(('nonfinite_physics_or_measurement','physics_buffer_capacity_exhausted')):
                    raise
                # Keep the finite prefix and the raw invalid terminal snapshot
                # separately. Never sanitize NaNs into training labels.
                probe.done_reason=str(error)
                np.savez_compressed(folder/'invalid_physics_state.npz',
                    **{k:getattr(probe.env.sim.data,k)[:].cpu().numpy() for k in ('qpos','qvel','qacc','ctrl')})
                break
            if probe.done_reason:break
            if f==99 and args.branches:root=probe.snapshot();root_frame=f+1
            if f%100==0:print(f'[WM] {row["object_id"]} frame {f}/{args.frames} elapsed {time.monotonic()-probe.start:.1f}s',flush=True)
        meta['terminal_reason']=probe.done_reason
        report.update(status='recorded',rollout=export(folder/'rollout.h5',records,meta))
        if hasattr(probe,'measurement_audit'):
            report['measurement_audit']=probe.measurement_audit
        if root is not None and hasattr(probe,'collect_branches'):
            report.update(probe.collect_branches(root,root_frame,records,meta,folder))
        elif root is not None:
            branches=[];traces=[];branch_folder=args.root/'branches'/args.run_tag/row['object_id'];branch_folder.mkdir(parents=True,exist_ok=True)
            snap_arrays={f'sim_{k}':v.cpu().numpy() for k,v in root['sim'].items()}
            snap_arrays.update({f'mcc_{k}':v.cpu().numpy() for k,v in root['mcc'].items() if isinstance(v,torch.Tensor)})
            snap_arrays.update({f'generator_{k}':v.cpu().numpy() for k,v in root.items() if isinstance(v,torch.Tensor)})
            snap_arrays.update({f'action_manager_{k}':v.cpu().numpy() for k,v in root['actions'].items()})
            snap_arrays['base_object_pose']=probe.base.cpu().numpy();snap_arrays['base_hand_q']=probe.q0.cpu().numpy();snap_arrays['anchors']=probe.anchors
            np.savez_compressed(branch_folder/'root.npz',**snap_arrays)
            save_json(branch_folder/'root.json',dict(root_frame=root_frame,tick=root['tick'],jobs=probe.jobs,object=row,split=row['split'],
                mcc_none_fields=[k for k,v in root['mcc'].items() if v is None],physics=meta['physics'],mcc_config=asdict(probe.config)))
            for bi,amplitude in enumerate([0.,0.,-.03,.03,-.08,.08]):
                probe.restore(root);branch=[probe.sample(2)];branch[0]['dt'][:]=0
                extra=torch.zeros_like(probe.q0)
                for i,j in enumerate(probe.jobs):extra[i,j['finger']*4:j['finger']*4+4]=torch.tensor([0,0,amplitude,amplitude],device=args.device)
                for k in range(60):
                    nom,pose=probe.desired(root_frame+k,extra);branch.append(probe.step(nom,pose,2))
                    if probe.done_reason:break
                root_id=f'{args.run_tag}/{row["object_id"]}/mu{args.friction:g}/k{args.contact_multiplier:g}/seed{args.seed}/frame100'
                bm={**meta,'root_id':root_id,'candidate':bi,'extra_rad':amplitude,'terminal_reason':probe.done_reason}
                branches.append(export(branch_folder/f'candidate_{bi}.h5',branch,bm));traces.append(branch)
            a,b=traces[:2];same_len=len(a)==len(b)
            qa=np.stack([r['q'] for r in a]);qb=np.stack([r['q'] for r in b]);fa=np.stack([r['pad_net_force_world'] for r in a]);fb=np.stack([r['pad_net_force_world'] for r in b])
            if same_len:
                qerr=float(np.abs(qa-qb).max());ferr=float(np.quantile(np.linalg.norm(fa-fb,axis=-1),.95))
                mask=float(np.not_equal(np.stack([r['strict'] for r in a]),np.stack([r['strict'] for r in b])).mean())
            else:qerr=ferr=mask=None
            repeat=dict(same_length=same_len,q_max=qerr,force_p95=ferr,mask_disagreement=mask,
                passed=bool(same_len and qerr<=1e-5 and ferr<=.1 and mask<=.001))
            report.update(branches=branches,snapshot_repeat=repeat)
        report['elapsed_s']=time.monotonic()-probe.start;save_json(folder/'summary.json',report);return report
    finally:
        if probe is not None:probe.env.close()
        undo()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--objects',nargs='+',default=['ellipsoid_000']);p.add_argument('--fingers',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--behaviors',nargs='+',default=['slide','press_release','turn','pulse'])
    p.add_argument('--frames',type=int,default=400);p.add_argument('--branches',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--friction',type=float,default=1.);p.add_argument('--contact-multiplier',type=float,default=1.);p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--run-tag',default='pilot_v1')
    a=p.parse_args();a.root=a.root.resolve();objects=json.loads((a.root/'object_manifest.json').read_text())['objects'];results=[]
    for name in a.objects:
        row=next(r for r in objects if r['object_id']==name)
        if row['split']!='train':raise ValueError('P0 tuning only on train instances')
        result=run(a,row);results.append(result);save_json(a.root/('attempt_'+a.run_tag+'.json'),results);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
