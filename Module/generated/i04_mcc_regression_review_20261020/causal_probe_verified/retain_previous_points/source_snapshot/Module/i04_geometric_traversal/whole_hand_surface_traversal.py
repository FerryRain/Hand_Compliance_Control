"""Fresh whole-mustard traversal with four active palmar contact tasks."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
import trimesh

from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from .whole_hand_belly import contact_clearance_handoff, cooperative_window_ready
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_surface_planner import WholeHandSurfacePlanner
from .natural_belly_traversal import set_kinematics, unit, raster_goals
from .belly_contact import BellyContactConfig, belly_contact_measurement
from .whole_hand_kinematics import fast_belly_ik_terms as belly_ik_terms
from .whole_hand_step_solver import WholeHandStepSolver
from .belly_bounded_step import bounded_belly_step
from .contracts import ExperimentConfig
from .full_surface import build_scene
from .physics import set_joint_state, set_control, joint_state, contact_measurement
from .sliding_safety import arm_self_contacts
from .oracle import MeasuredCoverage


def support_ready(history, expected, *, strict_four=False):
    """A movement handoff only; absent fingers remain active contact tasks."""
    safe=np.asarray(history,dtype=bool)
    if safe.shape!=(150,4):return False
    if strict_four:return cooperative_window_ready(safe)
    needed=min(3,max(2,int(np.asarray(expected).sum())))
    return bool(safe[-1].sum()>=needed and np.mean(safe.sum(axis=1)>=needed)>=.8)


class WholeHandTraversal:
    def __init__(self, output, geometry, seed_case, seconds):
        self.out=Path(output).resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.h,self.graph,_,quality=build_scene(self.out)
        self.planner=WholeHandSurfacePlanner(self.h,self.graph,seed_case)
        self.data=mujoco.MjData(self.h.model);self.scratch=mujoco.MjData(self.h.model)
        with np.load(geometry) as z:seed=z['q'][8].copy()
        q,initial,points,normals=self.planner.fit(seed,clearance=.006)
        if not initial['accepted']:raise RuntimeError('NO_COOPERATIVE_INITIAL_HOVER')
        set_joint_state(self.h,self.data,q);set_control(self.h,self.data,q)
        assert contact_measurement(self.h,self.data,.2)[0].max()<.05
        self.last=q.copy();self.initial_q=q.copy();self.points=points;self.normals=normals
        self.normal_filter=normals.copy();self.anchor=1;self.nominal=q.copy()
        self.clearance=np.full(4,.006);self.offsets=np.zeros(4);self.bias=np.zeros((4,3));self.release=np.zeros(4,bool)
        self.controllers=[FingertipMCC(MCCConfig(virtual_mass=.08,damping=70.,stiffness=20.,dt_s=.002,
            max_offset_m=.004,max_velocity_m_s=.00025,max_acceleration_m_s2=.02)) for _ in range(4)]
        self.coverage=MeasuredCoverage(self.graph,ExperimentConfig());self.bcfg=BellyContactConfig()
        self.duration=seconds;self.logs={};self.chunks=[];self.events=[];self.states=0;self.routes=0;self.lifts=0
        self.rejections=0;self.peak=0.;self.high_steps=0;self.done_hold=0.;self.last_print=perf_counter()
        self.window=deque(maxlen=150);self.goal=-1;self.route_progress=0.;self.geometry_backtracks=0
        self.stop='UNFINISHED';self.force_releases=0
        self.step_solver=WholeHandStepSolver()
        np.savez_compressed(self.out/'initial.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,
            qacc_warmstart=self.data.qacc_warmstart,time=self.data.time)
        (self.out/'model.xml').write_text(self.h.xml,encoding='utf-8')
        manifest=dict(protocol='i04-whole-hand-surface.development.v3',cell='B-MCC',scope='WHOLE_OBJECT_COOPERATIVE',
            dp='DEFERRED_BY_USER',source_hashes=snapshot_sources(self.out),geometry_quality=quality,
            belly_contact=asdict(self.bcfg),coverage_radius_m=.025,required_vertices=8192,required_goals=388,
            posture_lower_rad=self.planner.lower.tolist(),posture_upper_rad=self.planner.upper.tolist(),
            initial_state='One declared cooperative-hand 6mm hover; zero inherited coverage',initial_plan=initial,
            coverage_inherited=False,resume=None,original_collision_geometry=True,dt_s=.002,
            command_geometry_guard_every_native_tick=True,command_reference_guard_every_native_tick=True,
            proactive_release_n=2.5,release_resume_n=1.2,force_relief_n=8.,hard_force_n=12.,hard_force_duration_s=.020,
            requested_duration_s=seconds,controlled_fingers=[0,1,2,3],desired_force_per_finger_n=1.,
            all_four_contact_tasks_at_every_surface_tick=True,leader_finger_field_is_routing_anchor_only=True,
            shared_nominal_wrist_route=True,sliding_speed_m_s=.004,final_stable_dwell_s=.15,
            bounded_solver='Warm active set; original objective/box and KKT checked, original BVLS fallback',
            initial_handoff='300ms: >=3 for90%, each finger60%, current4',
            subsequent_handoff='300ms: geometry-supported quorum2or3 for80%; four tasks stay active',
            seed_case=str(Path(seed_case).resolve()),seed_use='Geometry-only IK seeds; no physical state or coverage inherited',
            geometry_source_sha256=sha256(Path(geometry).read_bytes()).hexdigest())
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def measure(self):
        h,d=self.h,self.data
        f,p,n,a,unsafe,penetration=contact_measurement(h,d,.2)
        bf,bp,bn,ba,bd=belly_contact_measurement(h,d,.2,self.bcfg);arm=arm_self_contacts(h,d)
        rejected=np.asarray(bd['rejected_contact_force_n']);safe=ba&(f<8.)&(rejected<.2)
        if unsafe or arm or f.max()>=8.:safe[:]=False
        return dict(f=f,p=p,n=n,a=a,unsafe=unsafe,penetration=penetration,bf=bf,bp=bp,bn=bn,ba=ba,bd=bd,
                    arm=arm,rejected=rejected,safe=safe)

    def flush(self):
        if not self.logs:return
        name=f'trace_{len(self.chunks):04d}.npz'
        np.savez_compressed(self.out/name,**{k:np.asarray(v) for k,v in self.logs.items()})
        self.chunks.append(dict(file=name,states=len(self.logs['time_s']),start_s=float(self.logs['time_s'][0]),
            end_s=float(self.logs['time_s'][-1]),sha256=sha256((self.out/name).read_bytes()).hexdigest()))
        self.logs={};d=self.data;c=self.coverage
        np.savez_compressed(self.out/'checkpoint.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,qacc_warmstart=d.qacc_warmstart,
            time=d.time,visited_vertices=c.visited_vertices,visited_goals=c.visited_goals)
        (self.out/'trace_chunks.json').write_text(json.dumps(self.chunks,indent=2))
        (self.out/'events.json').write_text(json.dumps(self.events,indent=2))

    def tick(self, command, phase, obs=None):
        h,d,p=self.h,self.data,self.planner
        obs=self.measure() if obs is None else obs
        f,safe=obs['f'],obs['safe'];self.window.append(safe.copy())
        self.coverage.update(obs['bp']-h.object_position_m,obs['bn'],safe)
        self.peak=max(self.peak,float(f.max()));self.high_steps=self.high_steps+1 if f.max()>12. else 0
        def reason(candidate):
            return p.safety.command_collision_reason(joint_state(h,d),candidate) or p.safety.command_collision_reason(self.last,candidate)
        rejection=reason(command)
        if rejection:
            self.geometry_backtracks+=1;self.rejections+=1;accepted=None
            for scale in (.5,.25,.125,0.):
                trial=self.last+scale*(command-self.last)
                if reason(trial) is None:accepted=trial;break
            if accepted is None:
                accepted=p.clearance_exit(joint_state(h,d),self.last)
                if accepted is None or reason(accepted) is not None:raise RuntimeError('NO_GEOMETRY_SAFE_COMMAND:'+rejection)
                phase='CLEARANCE_EXIT'
            command=accepted
        else:self.rejections=0
        assert np.all(command>=p.lower) and np.all(command<=p.upper)
        assert np.max(abs(command-self.last))<=.0024+1e-10
        values=dict(time_s=d.time,q_rad=joint_state(h,d),qvel_rad_s=d.qvel.copy(),qacc_warmstart=d.qacc_warmstart.copy(),
            sensor_command_rad=self.last.copy(),command_rad=command.copy(),forces_n=f,contact_positions_world_m=obs['p'],
            normals_world=obs['n'],active=obs['a'],belly_forces_n=obs['bf'],belly_contact_positions_world_m=obs['bp'],
            belly_normals_world=obs['bn'],belly_active=obs['ba'],rejected_contact_force_n=obs['rejected'],safe_belly=safe,
            phase=phase,leader_finger=self.anchor,mcc_offset_m=self.offsets.copy(),coverage=self.coverage.area_fraction,
            visited_vertex_count=int(self.coverage.visited_vertices.sum()),goal_vertex=self.goal,unsafe_count=len(obs['unsafe']),
            arm_self_force_n=max([item['force_n'] for item in obs['arm']],default=0.),actual_penetration_m=obs['penetration'],
            reference_geometry_backtrack=rejection or '',controlled_fingers=np.ones(4,bool),finger_search_active=~safe,
            simultaneous_safe_contacts=int(safe.sum()),target_points_world=self.points.copy(),target_normals_world=self.normal_filter.copy(),
            target_clearance_m=self.clearance.copy(),tangential_seam_bias_m=self.bias.copy(),stroke_distance_m=self.route_progress,
            contact_tasks_enabled=phase not in ('NECESSARY_REORIENTATION','COOPERATIVE_WITHDRAWAL'))
        for key,value in values.items():self.logs.setdefault(key,[]).append(value)
        self.states+=1
        if obs['unsafe'] or obs['arm']:raise RuntimeError('UNSAFE_NATIVE_CONTACT')
        if self.high_steps>=10:raise RuntimeError('SUSTAINED_DANGEROUS_FORCE')
        self.last=command.copy();set_control(h,d,command);mujoco.mj_step(h.model,d);mujoco.mj_forward(h.model,d)
        self.done_hold=self.done_hold+.002 if self.coverage.visited_vertices.all() and safe.any() else 0.
        if self.states%10000==0:self.flush()
        if self.states%500==0 and (self.out/'development_stop.json').exists():raise RuntimeError('DEVELOPMENT_STOP_REQUESTED')
        if perf_counter()-self.last_print>12.:
            result=dict(time_s=d.time,area=self.coverage.area_fraction,vertices=int(self.coverage.visited_vertices.sum()),
                goals=int(self.coverage.visited_goals.sum()),routes=self.routes,lifts=self.lifts,phase=phase,
                force_n=f.tolist(),safe=safe.tolist(),peak_n=self.peak,goal=self.goal)
            (self.out/'progress.json').write_text(json.dumps(result,indent=2));print('PROGRESS',json.dumps(result),flush=True)
            self.last_print=perf_counter()
        return obs

    def contact_step(self, points, normals, nominal, phase):
        h,p=self.h,self.planner;obs=self.measure();f=obs['f'];safe=obs['safe'];rejected=obs['rejected']
        self.points=np.asarray(points).copy();self.normals=np.asarray(normals).copy();self.nominal=nominal.copy()
        was_release=self.release.copy();self.release|=f>2.5;self.release&=f>=1.2
        self.force_releases+=int(np.count_nonzero(self.release&~was_release))
        if self.release.any():phase='FORCE_RELEASE'
        for i in range(4):
            if obs['ba'][i] and rejected[i]>=.2:
                away=np.zeros(3)
                for record in obs['bd']['records']:
                    if record['finger']==i and not record['geometry_qualified']:
                        n=np.asarray(record['object_outward_normal_world'])
                        away+=record['normal_force_n']*(n-obs['bn'][i]*(n@obs['bn'][i]))
                if np.linalg.norm(away)>1e-8:
                    self.bias[i]+=.000002*unit(away)
                    if np.linalg.norm(self.bias[i])>.006:self.bias[i]*=.006/np.linalg.norm(self.bias[i])
        targets=points+self.bias;errors=[];rows=[];set_kinematics(h,self.scratch,self.last)
        for i in range(4):
            if obs['ba'][i] and obs['bn'][i]@normals[i]>.7:target_n=obs['bn'][i]
            else:target_n=obs['n'][i] if obs['a'][i] and obs['n'][i]@normals[i]>.7 else normals[i]
            angle=np.arccos(np.clip(self.normal_filter[i]@target_n,-1.,1.));blend=min(.06,.35*.002/max(angle,1e-12))
            self.normal_filter[i]=unit((1-blend)*self.normal_filter[i]+blend*target_n);normal=self.normal_filter[i]
            if safe[i] and self.clearance[i]>0.:
                self.clearance[i],self.offsets[i]=contact_clearance_handoff(self.clearance[i],self.offsets[i])
                self.controllers[i].reset(offset_m=self.offsets[i])
            else:self.clearance[i]=max(0.,self.clearance[i]-.003*.002)
            prior=self.offsets[i]
            if self.clearance[i]<.0005:
                desired=(.75 if obs['ba'][i] else 0.) if rejected[i]>=.2 else 1.
                self.offsets[i]=self.controllers[i].step(targets[i],-normal,desired,float(f[i])).offset_m
            if self.release[i]:
                set_kinematics(h,p.data,joint_state(h,self.data))
                _,j,_=belly_ik_terms(h.model,p.data,int(h.tip_geom_ids[i]),targets[i],normal,dof_indices=p.order)
                velocity=float(normal@j[:3]@self.data.qvel[p.order])
                self.offsets[i]=max(-.004,prior-min(.0006,.00004+max(0.,-velocity)*.04))
                self.controllers[i].reset(offset_m=self.offsets[i])
            e,j,_=belly_ik_terms(h.model,self.scratch,int(h.tip_geom_ids[i]),targets[i],normal,
                clearance_m=self.clearance[i]-self.offsets[i],orientation_weight_m=.04,dof_indices=p.order)
            metric=np.outer(normal,normal)+.35*(np.eye(3)-np.outer(normal,normal))
            errors.append(np.r_[metric@e[:3],e[3:]]);rows.append(np.vstack((metric@j[:3],j[3:])))
        pe,pj=p.posture_terms(self.last,nominal,arm_weight=.03,finger_weight=.0025)
        errors.append(pe);rows.append(pj)
        gj,ge=p.arm_buffer_terms(self.last,np.arange(23))
        if len(ge):errors.append(ge);rows.append(gj)
        oe,oj=p.object_buffer_terms(self.last)
        if len(oe):errors.append(oe);rows.append(oj)
        delta,diag=self.step_solver.solve(np.vstack(rows),np.concatenate(errors),self.last,p.lower,p.upper,
            gain=.55,maximum_joint_step=np.r_[np.full(7,.0004),np.full(16,.0008)])
        if not diag['feasible']:raise RuntimeError('NO_COOPERATIVE_STEP:'+diag['reason'])
        return self.tick(self.last+delta,phase,obs)

    def approach(self, patch, clearance=.006, strict_four=False):
        start=self.data.time;root=self.last.copy();self.window.clear();self.bias[:]=0.;self.offsets[:]=0.
        self.clearance[:]=clearance;self.normal_filter=patch['normals'].copy();self.anchor=patch['anchor']
        for controller in self.controllers:controller.reset()
        budget=max(12.,clearance/.003+8.)
        while self.data.time<self.duration and self.data.time-start<budget:
            fraction=min(1.,(self.data.time-start)/max(2.,clearance/.003))
            nominal=(1-fraction)*root+fraction*patch['q']
            obs=self.contact_step(patch['points'],patch['normals'],nominal,'COOPERATIVE_APPROACH')
            if support_ready(self.window,patch['expected_contacts'],strict_four=strict_four) and obs['safe'][self.anchor]:return True
            if self.rejections>300:return False
        return False

    def slide(self, route, phase='COOPERATIVE_SLIDE'):
        start=self.data.time;progress=0.;gap=0.;length=route['distance'][-1]
        self.anchor=int(route['anchor']);self.goal=int(route.get('goal',-1));self.routes+=1
        needed=min(3,max(2,int(np.asarray(route['expected_contacts']).sum(axis=-1).min())))
        while self.data.time<self.duration and self.data.time-start<max(10.,length/.0015+5.):
            k=min(max(0,np.searchsorted(route['distance'],progress)-1),len(route['distance'])-2)
            fraction=(progress-route['distance'][k])/max(1e-9,route['distance'][k+1]-route['distance'][k])
            points=(1-fraction)*route['points'][k]+fraction*route['points'][k+1]
            normals=np.array([unit(n) for n in (1-fraction)*route['normals'][k]+fraction*route['normals'][k+1]])
            nominal=(1-fraction)*route['q'][k]+fraction*route['q'][k+1]
            self.route_progress=progress;obs=self.contact_step(points,normals,nominal,phase)
            safe=obs['safe'];gap=0. if safe.sum()>=needed and safe[self.anchor] else gap+.002
            if safe.sum()>=needed and safe[self.anchor] and not self.release.any() and obs['rejected'].max()<.2 and self.rejections<3:
                angle=max(np.arccos(np.clip(np.sum(route['normals'][k]*route['normals'][k+1],axis=1),-1.,1.)))
                speed=min(.004,.12*(route['distance'][k+1]-route['distance'][k])/max(angle,1e-9))
                progress=min(length,progress+speed*.002)
            if self.done_hold>=.15:return True,'FULL_MEASURED_BELLY_SURFACE'
            if progress>=length-1e-10 and safe[self.anchor]:return True,'ROUTE_FINISHED'
            if gap>6.:return False,'COOPERATIVE_RECONTACT_REPLAN'
            if self.rejections>250:return False,'GEOMETRY_REPLAN'
        return False,'ROUTE_TIME_BUDGET'

    def air_move(self,path,phase):
        progress=0.;start=self.data.time
        while self.data.time<self.duration and self.data.time-start<path.duration_s+8.:
            desired=path.sample(progress);delta=desired-self.last
            command=self.last+delta*min(1.,.0012/max(1e-12,np.max(abs(delta))))
            obs=self.measure()
            if obs['f'].max()>2.5:
                # Withdraw along each measured force normal before continuing
                # the validated joint path; this is still actual plant motion.
                rows=[];errors=[];set_kinematics(self.h,self.scratch,self.last)
                set_kinematics(self.h,self.planner.data,joint_state(self.h,self.data))
                for i in np.flatnonzero(obs['a']):
                    normal=obs['n'][i]
                    _,j,_=belly_ik_terms(self.h.model,self.scratch,int(self.h.tip_geom_ids[i]),obs['p'][i],normal,
                        dof_indices=self.planner.order)
                    _,ja,_=belly_ik_terms(self.h.model,self.planner.data,int(self.h.tip_geom_ids[i]),obs['p'][i],normal,
                        dof_indices=self.planner.order)
                    velocity=float(normal@ja[:3]@self.data.qvel[self.planner.order])
                    outward=min(.0006,.00004+max(0.,-velocity)*.04)
                    rows.append(normal@j[:3]);errors.append(-outward)
                if rows:
                    change,diag=bounded_belly_step(np.vstack(rows),np.asarray(errors),self.last,self.planner.lower,
                        self.planner.upper,gain=1.,maximum_joint_step=.001)
                    if diag['feasible']:command=self.last+change
                self.tick(command,'FORCE_RELEASE',obs)
            else:
                self.tick(command,phase,obs)
                if self.rejections<3:progress+=.002
            if progress>=path.duration_s and np.max(abs(self.last-path.waypoints[-1]))<.002:return True
            if self.rejections>400:return False
        return False

    def prepare_transfer(self, target):
        p=self.planner
        for clearance in (.006,.015,.035):
            withdrawal=p.fit_patch(self.last,self.points,self.normals,anchor=self.anchor,clearance=clearance,
                iterations=80,flexible=False)
            if withdrawal['reason'] is not None:continue
            first=p.path_between(self.last,withdrawal['q'])
            if first is None or first.duration_s>12.:continue
            hover=p.fit_patch(target['q'],target['points'],target['normals'],anchor=target['anchor'],
                clearance=clearance,iterations=75,flexible=False)
            if hover['reason'] is not None or not p.safety.edge_free(hover['q'],target['q']):continue
            middle=p.path_between(withdrawal['q'],hover['q'])
            if middle is not None:return first,middle,hover,clearance
        return None

    def pivot_route(self, degrees):
        anchor=self.anchor;center=self.points[anchor];normal=self.normals[anchor]
        qs=[self.last.copy()];points=[self.points.copy()];normals=[self.normals.copy()]
        for angle in np.linspace(0.,np.radians(degrees),6)[1:]:
            raw=center+Rotation.from_rotvec(angle*normal).apply(self.points-center)
            projected,_,_=trimesh.proximity.closest_point(self.graph.mesh,raw-self.h.object_position_m)
            pts=projected+self.h.object_position_m;ns=np.array([self.planner.smooth_normal(p) for p in projected])
            pts[anchor]=center;ns[anchor]=normal
            patch=self.planner.fit_patch(qs[-1],pts,ns,anchor=anchor,iterations=40,flexible=False)
            if patch['reason'] or not self.planner.safety.edge_free(qs[-1],patch['q']):return None
            qs.append(patch['q']);points.append(pts);normals.append(ns)
        return dict(q=np.array(qs),points=np.array(points),normals=np.array(normals),anchor=anchor,
            distance=np.linspace(0.,abs(np.radians(degrees))*.004/.08,6),expected_contacts=np.ones((5,4),bool),goal=-1)

    def run(self):
        g,p=self.graph,self.planner;failed=np.zeros(len(g.vertices_m),int);pivots=set()
        raster=raster_goals(g);order=np.full(len(g.vertices_m),len(raster));order[raster]=np.arange(len(raster))
        try:
            # Identical accepted initial patch and four-finger contact setup.
            q,diag=p.fit_targets(self.last,self.points,self.normals)
            initial=dict(q=q,points=self.points.copy(),normals=self.normals.copy(),anchor=1,expected_contacts=np.ones(4,bool))
            ok=self.approach(initial,strict_four=True)
            self.events.append(dict(kind='INITIAL_COOPERATIVE_APPROACH',time_s=self.data.time,success=ok))
            while self.data.time<self.duration:
                if self.done_hold>=.15:self.stop='FULL_MEASURED_BELLY_SURFACE';break
                obs=self.measure();safe=obs['safe']
                # Clear an actual positive buffer before trying another route.
                if p.minimum_arm_clearance(self.last)<.002 or p.minimum_nonpad_clearance(self.last)<.0045:
                    begin=self.data.time
                    while self.data.time-begin<1.5 and self.data.time<self.duration:
                        command=p.clearance_exit(joint_state(self.h,self.data),self.last,arm_target_m=.004,object_target_m=.006,preserve_wrist=True)
                        if command is None:break
                        self.tick(command,'CLEARANCE_EXIT')
                        if p.minimum_arm_clearance(self.last)>.0038 and p.minimum_nonpad_clearance(self.last)>.0058:break
                    self.points,self.normals=p.project_pads(self.last);self.normal_filter=self.normals.copy()
                remaining=np.flatnonzero(~self.coverage.visited_vertices)
                candidates=g.required_vertices[~self.coverage.visited_goals]
                if not len(candidates):candidates=remaining
                if not len(candidates):
                    self.contact_step(self.points,self.normals,self.last,'FINAL_SAFE_HOLD');continue
                starts=np.array([g.nearest_vertex(point-self.h.object_position_m)[0] for point in self.points])
                distances=dijkstra(g.adjacency,directed=False,indices=starts)
                ranking=candidates[np.argsort(distances[:,candidates].min(axis=0)+.04*failed[candidates]+.0001*order[candidates])]
                route=None;chosen=None;planning_start=perf_counter()
                for candidate in ranking[:10]:
                    anchor_order=np.argsort(distances[:,candidate])
                    for anchor in anchor_order[:2]:
                        route=p.group_route(self.last,self.points,self.normals,int(anchor),int(candidate))
                        if route is not None:chosen=int(candidate);break
                    if route is not None:break
                    failed[candidate]+=1
                    if perf_counter()-planning_start>15.:break
                if route is not None:
                    before=int(self.coverage.visited_vertices.sum());ok,why=self.slide(route)
                    new=int(self.coverage.visited_vertices.sum())-before
                    self.events.append(dict(kind='COOPERATIVE_SURFACE_ROUTE',time_s=self.data.time,goal=chosen,anchor=route['anchor'],
                        length_m=float(route['distance'][-1]),success=ok,reason=why,new_vertices=new,area=self.coverage.area_fraction))
                    if why=='FULL_MEASURED_BELLY_SURFACE':self.stop=why;break
                    if not self.coverage.visited_vertices[chosen]:failed[chosen]+=1
                    if ok and (new>0 or failed[chosen]<2):continue
                pivoted=False
                if self.measure()['safe'].sum()>=2:
                    for degrees in (15.,-15.,30.,-30.):
                        key=(*np.round(self.points[self.anchor]/.02).astype(int),self.anchor,degrees)
                        if key in pivots:continue
                        pivots.add(key);turn=self.pivot_route(degrees)
                        if turn is None:continue
                        ok,why=self.slide(turn,'COOPERATIVE_CONTACT_PIVOT')
                        self.events.append(dict(kind='COOPERATIVE_CONTACT_PIVOT',time_s=self.data.time,degrees=degrees,success=ok,reason=why))
                        if ok:pivoted=True;break
                if pivoted:continue
                options=[]
                for candidate in ranking[:8]:
                    candidate=int(candidate)
                    patch,attempts=p.find_patch(self.last,g.vertices_m[candidate]+self.h.object_position_m,
                        p.smooth_normal(g.vertices_m[candidate]),budget_s=8.)
                    if patch is None:failed[candidate]+=1;continue
                    transfer=self.prepare_transfer(patch)
                    if transfer is None:failed[candidate]+=1;continue
                    first,middle,hover,clearance=transfer
                    options.append((first.duration_s+middle.duration_s,candidate,patch,transfer))
                    if options[-1][0]<15. or len(options)>=2:break
                if options:
                    _,candidate,patch,transfer=min(options,key=lambda item:item[0]);first,middle,hover,clearance=transfer
                    self.lifts+=1;self.goal=candidate
                    self.events.append(dict(kind='NECESSARY_COOPERATIVE_REORIENTATION',time_s=self.data.time,goal=candidate,
                        duration_s=first.duration_s+middle.duration_s,clearance_m=clearance,expected_contacts=patch['expected_contacts'].tolist()))
                    if self.air_move(first,'COOPERATIVE_WITHDRAWAL') and self.air_move(middle,'NECESSARY_REORIENTATION'):
                        ok=self.approach(patch,clearance)
                        self.events.append(dict(kind='COOPERATIVE_REAPPROACH',time_s=self.data.time,goal=candidate,success=ok))
                        if not ok:failed[candidate]+=1
                    else:failed[candidate]+=1
                else:
                    self.events.append(dict(kind='NO_COOPERATIVE_TRANSFER',time_s=self.data.time,remaining=len(remaining)))
                    # Keep real control and a bounded planning search; never
                    # mark unreachable vertices covered or drop the denominator.
                    self.contact_step(self.points,self.normals,self.last,'COOPERATIVE_RECONTACT')
                    if failed[candidates].min()>=10:self.stop='NO_REACHABLE_COOPERATIVE_ROUTE';break
            if self.stop=='UNFINISHED':self.stop='SIMULATION_BUDGET'
        except RuntimeError as error:self.stop=str(error)
        except KeyboardInterrupt:self.stop='DEVELOPMENT_INTERRUPTED'
        finally:
            self.flush();d=self.data;c=self.coverage
            np.savez_compressed(self.out/'final_state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,qacc_warmstart=d.qacc_warmstart,
                time=d.time,visited_vertices=c.visited_vertices,visited_goals=c.visited_goals)
            result=dict(stop_reason=self.stop,executed_s=float(d.time),full_surface_complete=bool(self.done_hold>=.15),
                coverage_fraction=c.area_fraction,visited_vertices=int(c.visited_vertices.sum()),total_vertices=8192,
                visited_goals=int(c.visited_goals.sum()),total_goals=388,native_states=self.states,routes=self.routes,
                necessary_lifts=self.lifts,peak_force_n=self.peak,force_releases=self.force_releases,geometry_backtracks=self.geometry_backtracks,
                coverage_inherited=False,fresh_single_version=True,resume=None,trace_chunks=self.chunks)
            (self.out/'summary.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--seed-case',required=True);p.add_argument('--seconds',type=float,default=3600.)
    a=p.parse_args();WholeHandTraversal(a.output,a.geometry,a.seed_case,a.seconds).run()
