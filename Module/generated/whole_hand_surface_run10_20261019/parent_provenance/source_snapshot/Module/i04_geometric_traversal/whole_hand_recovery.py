"""Pause-aware physical transfers and exact-checkpoint cooperative continuation.

The continuation contains the parent's complete immutable native prefix. The
independent auditor replays from t=0 through the controller-version boundary;
neither coverage from another trajectory nor a physical teleport is allowed.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra

from .whole_hand_continuity import ContinuityWholeHandTraversal
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_kinematics import fast_belly_ik_terms as belly_ik_terms
from .belly_bounded_step import bounded_belly_step
from .natural_belly_traversal import set_kinematics, raster_goals
from .physics import joint_state


class RecoveryWholeHandTraversal(ContinuityWholeHandTraversal):
    def __init__(self, output, geometry, seed_case, seconds, resume=None):
        super().__init__(output, geometry, seed_case, seconds)
        self.resume_info=None
        if resume is not None:self.restore_prefix(Path(resume).resolve())
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol='i04-whole-hand-surface.development.v6',
            pause_aware_transfer=True,transfer_force_pause_budget_s=40.,
            transfer_failure_refreshes_current_surface=True,
            resume=self.resume_info,coverage_inherited=bool(self.resume_info),
            coverage_inheritance_scope='Same continuously replayable native prefix only' if self.resume_info else None,
            source_hashes=snapshot_sources(self.out))
        if self.resume_info:
            manifest['initial_state']='Original parent t=0 hover; exact physical continuation at declared join time'
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

    def restore_prefix(self, parent):
        pm=json.loads((parent/'manifest.json').read_text())
        ps=json.loads((parent/'summary.json').read_text())
        pa=json.loads((parent/'independent_native_audit/audit.json').read_text())
        assert pa['status']=='PASSED_NATIVE_EVIDENCE_AUDIT' and pa['continuous_replay']
        assert not pm.get('resume'), 'Only one explicit version boundary is supported here'
        parent_xml=(parent/'model.xml').read_text(encoding='utf-8')
        current_xml=(self.out/'model.xml').read_text(encoding='utf-8')
        assert parent_xml.replace(str(parent),'CASE_ROOT')==current_xml.replace(str(self.out),'CASE_ROOT')
        for path in [parent/'canonical_object.obj',*sorted((parent/'collision_parts').glob('*.obj'))]:
            assert path.read_bytes()==(self.out/path.relative_to(parent)).read_bytes()
        # Keep exactly the audited original XML and immutable asset references.
        shutil.copy2(parent/'model.xml',self.out/'model.xml')
        assert ps['trace_chunks'] and not ps['full_surface_complete']
        provenance=self.out/'parent_provenance';provenance.mkdir()
        for name in ('manifest.json','summary.json','initial.npz','final_state.npz'):
            assert sha256((parent/name).read_bytes()).hexdigest()==pa['source_files'][name]
            shutil.copy2(parent/name,provenance/name)
        shutil.copy2(parent/'independent_native_audit/audit.json',provenance/'audit.json')
        shutil.copytree(parent/'source_snapshot',provenance/'source_snapshot')
        for item in ps['trace_chunks']:
            path=parent/item['file'];assert sha256(path.read_bytes()).hexdigest()==item['sha256']
            try:os.link(path,self.out/item['file'])
            except OSError:shutil.copy2(path,self.out/item['file'])
        self.chunks=list(ps['trace_chunks']);self.states=int(ps['native_states'])
        self.events=json.loads((parent/'events.json').read_text())
        shutil.copy2(parent/'initial.npz',self.out/'initial.npz')
        with np.load(parent/'final_state.npz') as z:
            for name in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(self.data,name)[:]=z[name]
            self.data.time=float(z['time'])
            self.coverage.visited_vertices[:]=z['visited_vertices']
            self.coverage.visited_goals[:]=z['visited_goals']
        mujoco.mj_forward(self.h.model,self.data)
        self.last=np.r_[self.data.ctrl[self.h.arm_actuator_ids],self.data.ctrl[self.h.hand_actuator_ids]]
        with np.load(parent/self.chunks[-1]['file']) as z:
            np.testing.assert_array_equal(self.last,z['command_rad'][-1])
            assert abs(self.data.time-float(z['time_s'][-1])-.002)<1e-9
            self.anchor=int(z['leader_finger'][-1]);self.goal=int(z['goal_vertex'][-1])
        self.refresh_surface()
        self.routes=int(ps['routes']);self.lifts=int(ps['necessary_lifts'])
        self.peak=float(ps['peak_force_n']);self.force_releases=int(ps.get('force_releases',0))
        self.geometry_backtracks=int(ps['geometry_backtracks'])
        self.resume_info=dict(parent_case=str(parent),join_time_s=float(self.data.time),
            parent_manifest_sha256=sha256((parent/'manifest.json').read_bytes()).hexdigest(),
            parent_summary_sha256=sha256((parent/'summary.json').read_bytes()).hexdigest(),
            parent_audit_sha256=sha256((provenance/'audit.json').read_bytes()).hexdigest(),
            prefix_native_states=self.states,prefix_chunks=len(self.chunks),
            physical_state_reset=False,independent_trajectories_merged=False,
            previous_protocol=pm['protocol'],controller_state='Reinitialized; physical state and command unchanged')
        self.events.append(dict(kind='EXACT_NATIVE_CONTINUATION',time_s=self.data.time,**self.resume_info))
        shutil.copy2(parent/'final_state.npz',self.out/'checkpoint.npz')
        (self.out/'trace_chunks.json').write_text(json.dumps(self.chunks,indent=2))
        (self.out/'events.json').write_text(json.dumps(self.events,indent=2))

    def refresh_surface(self):
        self.points,self.normals=self.planner.project_pads(self.last)
        self.normal_filter=self.normals.copy();self.bias[:]=0.;self.offsets[:]=0.
        self.clearance[:]=.006;self.release[:]=False;self.window.clear()
        for controller in self.controllers:controller.reset()

    def air_move(self,path,phase):
        progress=0.;start=self.data.time;force_pause=0.;tracking_time=0.
        while self.data.time<self.duration:
            if force_pause>=40. or tracking_time>=path.duration_s+8.:break
            desired=path.sample(progress);delta=desired-self.last
            command=self.last+delta*min(1.,.0012/max(1e-12,np.max(abs(delta))))
            obs=self.measure()
            if obs['f'].max()>2.5:
                rows=[];errors=[];set_kinematics(self.h,self.scratch,self.last)
                set_kinematics(self.h,self.planner.data,joint_state(self.h,self.data))
                for i in np.flatnonzero(obs['a']):
                    normal=obs['n'][i]
                    _,j,_=belly_ik_terms(self.h.model,self.scratch,int(self.h.tip_geom_ids[i]),obs['p'][i],normal,dof_indices=self.planner.order)
                    _,ja,_=belly_ik_terms(self.h.model,self.planner.data,int(self.h.tip_geom_ids[i]),obs['p'][i],normal,dof_indices=self.planner.order)
                    velocity=float(normal@ja[:3]@self.data.qvel[self.planner.order])
                    rows.append(normal@j[:3]);errors.append(-min(.0006,.00004+max(0.,-velocity)*.04))
                if rows:
                    change,diag=bounded_belly_step(np.vstack(rows),np.asarray(errors),self.last,self.planner.lower,
                        self.planner.upper,gain=1.,maximum_joint_step=.001)
                    if diag['feasible']:command=self.last+change
                self.tick(command,'FORCE_RELEASE',obs);force_pause+=.002
            else:
                self.tick(command,phase,obs);tracking_time+=.002
                if self.rejections<3:progress+=.002
            if progress>=path.duration_s and np.max(abs(self.last-path.waypoints[-1]))<.002:
                self.events.append(dict(kind='TRANSFER_SEGMENT_FINISHED',time_s=self.data.time,
                    phase=phase,elapsed_s=self.data.time-start,force_pause_s=force_pause,planned_s=path.duration_s))
                return True
            if self.rejections>400:break
        self.refresh_surface()
        self.events.append(dict(kind='TRANSFER_SEGMENT_REPLAN',time_s=self.data.time,
            phase=phase,elapsed_s=self.data.time-start,force_pause_s=force_pause,progress_s=progress))
        return False

    def prepare_transfer(self,target):
        # An already unloaded hand can approach the next validated hover
        # directly. It must not first return to a stale pre-turn contact patch.
        if self.measure()['f'].max()<.2:
            p=self.planner
            for clearance in (.006,.015,.035):
                hover=p.fit_patch(target['q'],target['points'],target['normals'],anchor=target['anchor'],
                    clearance=clearance,iterations=75,flexible=False)
                if hover['reason'] is not None or not p.safety.edge_free(hover['q'],target['q']):continue
                middle=p.path_between(self.last,hover['q'])
                if middle is not None:return p.path_between(self.last,self.last),middle,hover,clearance
        return super().prepare_transfer(target)

    def run(self):
        g,p=self.graph,self.planner;failed=np.zeros(len(g.vertices_m),int);pivots=set()
        raster=raster_goals(g);order=np.full(len(g.vertices_m),len(raster));order[raster]=np.arange(len(raster))
        try:
            if self.resume_info is None:
                q,diag=p.fit_targets(self.last,self.points,self.normals)
                initial=dict(q=q,points=self.points.copy(),normals=self.normals.copy(),anchor=1,expected_contacts=np.ones(4,bool))
                ok=self.approach(initial,strict_four=True)
                self.events.append(dict(kind='INITIAL_COOPERATIVE_APPROACH',time_s=self.data.time,success=ok))
            while self.data.time<self.duration:
                if self.done_hold>=.15:self.stop='FULL_MEASURED_BELLY_SURFACE';break
                if p.minimum_arm_clearance(self.last)<.002 or p.minimum_nonpad_clearance(self.last)<.0045:
                    begin=self.data.time
                    while self.data.time-begin<1.5 and self.data.time<self.duration:
                        command=p.clearance_exit(joint_state(self.h,self.data),self.last,arm_target_m=.004,object_target_m=.006,preserve_wrist=True)
                        if command is None:break
                        self.tick(command,'CLEARANCE_EXIT')
                        if p.minimum_arm_clearance(self.last)>.0038 and p.minimum_nonpad_clearance(self.last)>.0058:break
                    self.refresh_surface()
                remaining=np.flatnonzero(~self.coverage.visited_vertices)
                candidates=g.required_vertices[~self.coverage.visited_goals]
                if not len(candidates):candidates=remaining
                if not len(candidates):self.contact_step(self.points,self.normals,self.last,'FINAL_SAFE_HOLD');continue
                starts=np.array([g.nearest_vertex(point-self.h.object_position_m)[0] for point in self.points])
                distances=dijkstra(g.adjacency,directed=False,indices=starts)
                ranking=candidates[np.argsort(distances[:,candidates].min(axis=0)+.04*failed[candidates]+.0001*order[candidates])]
                route=None;chosen=None;planning_start=perf_counter()
                # Sliding requires nearby actual support; an unloaded hand goes
                # straight to validated reapproach planning instead.
                if self.measure()['safe'].sum()>=2:
                    for candidate in ranking[:10]:
                        for anchor in np.argsort(distances[:,candidate])[:2]:
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
                    patch,attempts=p.find_patch(self.last,g.vertices_m[candidate]+self.h.object_position_m,p.smooth_normal(g.vertices_m[candidate]),budget_s=8.)
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
                    # Give recovery actual physical time before another costly
                    # geometric search, while preserving all four active tasks.
                    self.refresh_surface();begin=self.data.time
                    while self.data.time-begin<.5 and self.data.time<self.duration:
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
                force_releases_scope='Controller counter after join only' if self.resume_info else 'Whole trajectory',
                coverage_inherited=bool(self.resume_info),fresh_single_version=self.resume_info is None,resume=self.resume_info,trace_chunks=self.chunks)
            (self.out/'summary.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--seed-case',required=True);p.add_argument('--seconds',type=float,default=6000.);p.add_argument('--resume')
    a=p.parse_args();RecoveryWholeHandTraversal(a.output,a.geometry,a.seed_case,a.seconds,a.resume).run()
