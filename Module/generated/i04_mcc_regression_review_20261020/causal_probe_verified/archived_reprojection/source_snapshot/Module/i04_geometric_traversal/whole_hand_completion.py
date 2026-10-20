"""Measured-coverage completion using nearby cooperative surface patches.

Geodesic neighbours only propose contact targets. Coverage remains exclusively
the original native palmar-contact ledger, with its unchanged 25 mm radius.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
import trimesh
from .whole_hand_continuity import ContinuitySurfacePlanner
from .whole_hand_recovery import RecoveryWholeHandTraversal
from .whole_hand_geometry_cache import CommandGeometryCache
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_lineage import restore_verified_prefix
from .oracle import align_vector


class CompletionSurfacePlanner(ContinuitySurfacePlanner):
    def load_cooperative_seeds(self,case):
        case=Path(case);summary=json.loads((case/'summary.json').read_text());qs=[];points=[];fingers=[]
        for item in summary['trace_chunks']:
            source=case/item['file'];assert sha256(source.read_bytes()).hexdigest()==item['sha256']
            with np.load(source) as z:
                ids=np.arange(0,len(z['time_s']),100);safe=z['safe_belly'][ids]
                rows,cols=np.nonzero(safe&(safe.sum(axis=1)>=2)[:,None]);rows=ids[rows]
                qs.append(z['q_rad'][rows]);points.append(z['belly_contact_positions_world_m'][rows,cols]);fingers.append(cols)
        self.seed_q=np.concatenate(qs);self.seed_finger=np.concatenate(fingers);self.seed_tree=cKDTree(np.concatenate(points))

    def group_route(self,root,initial_points,initial_normals,anchor,goal,*,maximum_m=.08):
        start=self.graph.nearest_vertex(initial_points[anchor]-self.h.object_position_m)[0]
        route,route_normals=self.surface_route(start,int(goal),spacing=.002)
        if len(route)<2:return None
        route+=self.h.object_position_m;route[0]=initial_points[anchor];route_normals[0]=initial_normals[anchor]
        points=np.asarray(initial_points).copy();normals=np.asarray(initial_normals).copy()
        qs=[root.copy()];pps=[points.copy()];nns=[normals.copy()];distances=[0.];expected=[]
        for k in range(1,len(route)):
            turn=align_vector(route_normals[k-1],route_normals[k]);guesses=route[k]+turn.apply(points-route[k-1])
            projected,_,_=trimesh.proximity.closest_point(self.graph.mesh,guesses-self.h.object_position_m)
            points=projected+self.h.object_position_m;normals=np.array([self.smooth_normal(p) for p in projected])
            points[anchor]=route[k];normals[anchor]=route_normals[k]
            patch=self.fit_patch(qs[-1],points,normals,anchor=anchor,iterations=55,flexible=False)
            if patch['reason'] is not None or not self.safety.edge_free(qs[-1],patch['q']):break
            points=patch['points'];normals=patch['normals'];length=distances[-1]+float(np.linalg.norm(route[k]-route[k-1]))
            qs.append(patch['q']);pps.append(points.copy());nns.append(normals.copy());distances.append(length);expected.append(patch['expected_contacts'])
            if length>=maximum_m:break
        if len(qs)<2 or distances[-1]<.001:return None
        return dict(q=np.array(qs),points=np.array(pps),normals=np.array(nns),distance=np.array(distances),
            anchor=int(anchor),expected_contacts=np.array(expected),goal=int(goal),complete=len(qs)==len(route))


class CompletionWholeHandTraversal(RecoveryWholeHandTraversal):
    def restore_prefix(self,parent):restore_verified_prefix(self,parent)

    def __init__(self,output,geometry,seed_case,seconds,resume):
        super().__init__(output,geometry,seed_case,seconds,resume)
        self.planner=CompletionSurfacePlanner(self.h,self.graph,seed_case)
        self.planner.load_cooperative_seeds(resume)
        self.command_cache=CommandGeometryCache(self.planner.safety)
        self.planner.safety.command_collision_reason=self.command_cache.command_reason
        self.nominal=self.last.copy();self.attempts=np.zeros(len(self.graph.vertices_m),int)
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol='i04-whole-hand-surface.development.v7',
            completion_contact_target_geodesic_radius_m=.018,minimum_completion_stroke_m=.001,
            seed_use='Own verified native cooperative poses as geometry-only seeds; no new coverage credit',
            cooperative_seed_source=str(Path(resume).resolve()),all_versions_reference_guard_audit_required=True,
            historical_goal_failure_counter_is_not_terminal=True,source_hashes=snapshot_sources(self.out))
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

    def planning_status(self,phase,**extra):
        result=dict(time_s=float(self.data.time),area=self.coverage.area_fraction,vertices=int(self.coverage.visited_vertices.sum()),
            goals=int(self.coverage.visited_goals.sum()),routes=self.routes,lifts=self.lifts,phase=phase,
            force_n=self.measure()['f'].tolist(),safe=self.measure()['safe'].tolist(),peak_n=self.peak,goal=self.goal,**extra)
        (self.out/'progress.json').write_text(json.dumps(result,indent=2));print('PLANNING',json.dumps(result),flush=True)

    def ranked_targets(self):
        g=self.graph;missing=np.flatnonzero(~self.coverage.visited_vertices)
        distance=dijkstra(g.adjacency,directed=False,indices=missing,limit=.018)
        candidates=np.flatnonzero(np.isfinite(distance).any(axis=0));gain=np.isfinite(distance[:,candidates]).sum(axis=0)
        near=np.linalg.norm(g.vertices_m[candidates,None,:]+self.h.object_position_m-self.points[None,:,:],axis=-1).min(axis=1)
        order=np.argsort(near+.025*self.attempts[candidates]-.002*gain)
        # Spatial diversity prevents spending the whole search on neighbours
        # of the same inaccessible geometric anchor.
        chosen=[]
        for i in order:
            v=int(candidates[i])
            if all(np.linalg.norm(g.vertices_m[v]-g.vertices_m[w])>.007 for w in chosen):chosen.append(v)
            if len(chosen)>=32:break
        return chosen,distance

    def expected_new_neighbourhood(self,route,distances):
        ids=[self.graph.nearest_vertex(p-self.h.object_position_m)[0] for p in route['points'][-1]]
        return int(np.isfinite(distances[:,ids]).any(axis=1).sum())

    def run(self):
        no_progress=0;cycles=0
        try:
            while self.data.time<self.duration:
                if self.done_hold>=.15:self.stop='FULL_MEASURED_BELLY_SURFACE';break
                if self.coverage.visited_vertices.all():
                    self.contact_step(self.points,self.normals,self.nominal,'FINAL_SAFE_HOLD');continue
                cycles+=1;before=int(self.coverage.visited_vertices.sum());candidates,distances=self.ranked_targets()
                self.planning_status('PLANNING_COMPLETION',cycle=cycles,remaining=8192-before)
                route=None;chosen=None
                if self.measure()['safe'].sum()>=2:
                    for v in candidates[:8]:
                        anchors=np.argsort(np.linalg.norm(self.points-(self.graph.vertices_m[v]+self.h.object_position_m),axis=1))
                        for anchor in anchors[:2]:
                            proposed=self.planner.group_route(self.last,self.points,self.normals,int(anchor),v)
                            if proposed is not None and self.expected_new_neighbourhood(proposed,distances)>0:
                                route=proposed;chosen=v;break
                        if route is not None:break
                        self.attempts[v]+=1
                if route is not None:
                    ok,why=self.slide(route)
                    added=int(self.coverage.visited_vertices.sum())-before
                    self.events.append(dict(kind='COMPLETION_SURFACE_ROUTE',time_s=self.data.time,goal=chosen,
                        success=ok,reason=why,new_vertices=added,length_m=float(route['distance'][-1])))
                    self.attempts[chosen]+=int(added==0)
                    if added>0:no_progress=0;continue
                options=[]
                for v in candidates:
                    self.planning_status('PLANNING_COMPLETION_PATCH',candidate=v,options=len(options))
                    point=self.graph.vertices_m[v]+self.h.object_position_m
                    patch,attempts=self.planner.find_patch(self.last,point,self.planner.smooth_normal(self.graph.vertices_m[v]),budget_s=25.)
                    if patch is None:self.attempts[v]+=1;continue
                    transfer=self.prepare_transfer(patch)
                    if transfer is None:self.attempts[v]+=1;continue
                    first,middle,hover,clearance=transfer
                    cost=first.duration_s+middle.duration_s+3.*(4-int(patch['expected_contacts'].sum()))
                    options.append((cost,v,patch,transfer))
                    if cost<8. or len(options)>=3:break
                if options:
                    _,v,patch,transfer=min(options,key=lambda option:option[0]);first,middle,hover,clearance=transfer
                    self.lifts+=1;self.goal=v
                    self.events.append(dict(kind='COMPLETION_COOPERATIVE_REORIENTATION',time_s=self.data.time,goal=v,
                        duration_s=first.duration_s+middle.duration_s,clearance_m=clearance,expected_contacts=patch['expected_contacts'].tolist()))
                    if self.air_move(first,'COOPERATIVE_WITHDRAWAL') and self.air_move(middle,'NECESSARY_REORIENTATION'):
                        ok=self.approach(patch,clearance)
                        end=min(self.duration,self.data.time+2.)
                        while self.data.time<end and self.done_hold<.15:
                            self.contact_step(patch['points'],patch['normals'],patch['q'],'COOPERATIVE_RECONTACT')
                        self.events.append(dict(kind='COMPLETION_REAPPROACH',time_s=self.data.time,goal=v,success=ok,
                            new_vertices=int(self.coverage.visited_vertices.sum())-before))
                    self.attempts[v]+=int(self.coverage.visited_vertices.sum()==before)
                else:
                    # Preserve the ongoing approach/MCC state. Repeatedly
                    # resetting a 6 mm clearance would prevent recontact.
                    end=min(self.duration,self.data.time+2.)
                    while self.data.time<end:self.contact_step(self.points,self.normals,self.nominal,'COOPERATIVE_RECONTACT')
                    self.events.append(dict(kind='COMPLETION_SEARCH_RETRY',time_s=self.data.time,cycle=cycles))
                added=int(self.coverage.visited_vertices.sum())-before;no_progress=0 if added else no_progress+1
                if no_progress>=16:self.stop='COMPLETION_SEARCH_EXHAUSTED';break
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
                necessary_lifts=self.lifts,peak_force_n=self.peak,force_releases=self.force_releases,
                force_releases_scope='Controller counter after join only',geometry_backtracks=self.geometry_backtracks,
                coverage_inherited=True,fresh_single_version=False,resume=self.resume_info,trace_chunks=self.chunks,completion_cycles=cycles)
            (self.out/'summary.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--seed-case',required=True);p.add_argument('--seconds',type=float,default=7000.);p.add_argument('--resume',required=True)
    a=p.parse_args();CompletionWholeHandTraversal(a.output,a.geometry,a.seed_case,a.seconds,a.resume).run()
