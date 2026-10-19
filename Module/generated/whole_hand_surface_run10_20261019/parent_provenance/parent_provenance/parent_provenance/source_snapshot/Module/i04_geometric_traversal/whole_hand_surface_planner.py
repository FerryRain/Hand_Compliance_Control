"""Whole-object cooperative patches and continuous four-finger surface paths.

A routing finger names the requested mesh anchor only. Every fit includes all
four palmar targets; neither geometry nor an old seed grants measured coverage.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from Module.fr3_leap import ARM_HOME_Q
from .whole_hand_belly import WholeHandBellyPlanner, cooperative_hand_reference
from .natural_belly_traversal import set_kinematics, unit
from .belly_contact import belly_ik_terms
from .whole_hand_kinematics import fast_belly_ik_terms
from .contracts import WristTarget
from .physics import pose, position_jacobian
from .oracle import align_vector
from .full_surface import JointPath
from .sliding_safety import is_nonadjacent_arm_pair


class WholeHandSurfacePlanner(WholeHandBellyPlanner):
    def __init__(self, handles, graph, seed_case=None):
        super().__init__(handles, graph)
        self.seed_tree=None;self.seed_q=None;self.seed_finger=None
        m=self.h.model
        bodies=sorted(set(map(int,m.geom_bodyid[self.safety._arm_geoms])))
        self.arm_pairs=[(a,b) for k,a in enumerate(bodies) for b in bodies[k+1:]
                        if is_nonadjacent_arm_pair(m,a,b)]
        self.arm_pair_index={pair:k for k,pair in enumerate(self.arm_pairs)}
        if seed_case:
            case=Path(seed_case);audit=json.loads((case/'independent_native_audit/audit.json').read_text())
            source=case/'independent_native_audit/verified_preview.npz'
            assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
            assert sha256(source.read_bytes()).hexdigest()==audit['preview_sha256']
            with np.load(source) as z:
                rows,fingers=np.nonzero(z['belly_active'][::4])
                rows=rows*4
                self.seed_q=z['q_rad'][rows].copy();self.seed_finger=fingers
                points=z['belly_contact_positions_world_m'][rows,fingers]
            self.seed_tree=cKDTree(points)

    def fixed_buffer_terms(self, q, margin=.006):
        """Fixed-size geometric residuals, without changing the physical model."""
        m=self.h.model;d=self.safety.scratch;set_kinematics(self.h,d,q)
        before=m.geom_margin.copy()
        try:
            m.geom_margin[self.h.object_geom_ids]=np.maximum(before[self.h.object_geom_ids],margin+.0001)
            m.geom_margin[self.safety._arm_geoms]=np.maximum(before[self.safety._arm_geoms],margin+.0001)
            mujoco.mj_collision(m,d)
        finally:m.geom_margin[:]=before
        errors=np.zeros(m.nbody+len(self.arm_pairs));jac=np.zeros((len(errors),23))
        objects=set(map(int,self.h.object_geom_ids));pads=set(map(int,self.h.tip_geom_ids))
        for contact in d.contact:
            if contact.dist>=margin:continue
            a,b=int(contact.geom1),int(contact.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
            normal=np.asarray(contact.frame[:3]);body=None
            if a in objects and b not in objects and b not in pads:body=bb
            elif b in objects and a not in objects and a not in pads:body=ba;normal=-normal
            if body is not None:
                slot=body
                if errors[slot]<=3.*(contact.dist-margin):continue
                jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jp,jr,np.asarray(contact.pos),body)
                errors[slot]=3.*(contact.dist-margin);jac[slot]=3.*normal@jp[:,self.order]
            elif (min(ba,bb),max(ba,bb)) in self.arm_pair_index:
                slot=m.nbody+self.arm_pair_index[(min(ba,bb),max(ba,bb))]
                if errors[slot]<=3.*(contact.dist-margin):continue
                ja,jb,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv)),np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,ja,jr,np.asarray(contact.pos),ba)
                mujoco.mj_jac(m,d,jb,jr,np.asarray(contact.pos),bb)
                errors[slot]=3.*(contact.dist-margin);jac[slot]=3.*np.asarray(contact.frame[:3])@(jb-ja)[:,self.order]
        return errors,jac

    def fit_patch(self, root, points, normals, *, anchor=1, clearance=.0002,
                  iterations=55, reproject=0, anchor_weight=1., flexible=True):
        q=np.clip(root,self.lower+1e-9,self.upper-1e-9);points=np.asarray(points).copy();normals=np.asarray(normals).copy()
        for outer in range(reproject+1):
            cached=None;value=None
            def terms(trial):
                nonlocal cached,value
                if cached is not None and np.array_equal(cached,trial):return value
                set_kinematics(self.h,self.data,trial);errors=[];rows=[]
                for finger in range(4):
                    n=normals[finger]
                    e,j,_=fast_belly_ik_terms(self.h.model,self.data,int(self.h.tip_geom_ids[finger]),points[finger],n,
                        clearance_m=clearance,orientation_weight_m=.04,dof_indices=self.order)
                    tangent=1. if finger==anchor else (.12 if flexible else .35)
                    weight=anchor_weight if finger==anchor else 1.
                    metric=weight*(np.outer(n,n)+tangent*(np.eye(3)-np.outer(n,n)))
                    errors.append(np.r_[metric@e[:3],weight*e[3:]])
                    rows.append(np.vstack((metric@j[:3],weight*j[3:])))
                pe,pj=self.posture_terms(trial,root)
                ge,gj=self.fixed_buffer_terms(trial)
                value=(np.concatenate([*errors,pe,ge]),np.vstack([*rows,pj,gj]));cached=trial.copy()
                return value
            fit=least_squares(lambda x:terms(x)[0],q,jac=lambda x:terms(x)[1],bounds=(self.lower,self.upper),
                max_nfev=iterations,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            q=fit.x
            if outer<reproject:
                projected,ns=self.project_pads(q)
                for i in range(4):
                    if i!=anchor:points[i]=projected[i];normals[i]=ns[i]
        set_kinematics(self.h,self.data,q);details=[]
        for i in range(4):
            e,_,d=belly_ik_terms(self.h.model,self.data,int(self.h.tip_geom_ids[i]),points[i],normals[i],
                clearance_m=clearance,dof_indices=self.order)
            details.append(dict(normal_error_m=float(e[:3]@normals[i]),position_error_m=float(np.linalg.norm(e[:3])),
                angle_deg=d['alignment_angle_deg'],qualified_patch=d['geometry_qualified']))
        eligible=np.array([abs(d['normal_error_m'])<.0025 and d['angle_deg']<15. and d['qualified_patch'] for d in details])
        reason=self.safety.collision_reason(q)
        if reason is None and (details[anchor]['position_error_m']>.004 or not eligible[anchor]):reason='ANCHOR_UNREACHED'
        if reason is None and eligible.sum()<2:reason='INSUFFICIENT_COOPERATIVE_PATCH'
        return dict(q=q,points=points,normals=normals,anchor=anchor,
            expected_contacts=eligible,reason=reason,details=details,clearance=clearance)

    def find_patch(self, root, p, n, *, clearance=.0002, budget_s=8., preferred=None):
        begin=perf_counter();seeds=[];attempts=[];best=None;best_score=float('inf')
        if preferred is not None:seeds.append((np.asarray(root),int(preferred),'continuation'))
        if self.seed_tree is not None:
            distances,indices=self.seed_tree.query(p,k=min(64,len(self.seed_q)))
            seen=[]
            for k in np.atleast_1d(indices):
                seed=np.clip(self.seed_q[k],self.lower+1e-9,self.upper-1e-9)
                finger=int(self.seed_finger[k])
                if any(finger==f and np.linalg.norm(seed[:7]-a[:7])<.5 for a,f,_ in seen):continue
                seeds.append((seed,finger,'previous_geometry_seed'));seen.append((seed,finger,''))
                if len(seen)>=5:break
        for roll in (0.,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2,np.pi):
            for finger in (1,0,2,3):
                target,hand=self.target(p,n,np.array([0.,1.,0.]),finger=finger,clearance=clearance,
                    roll=roll,root_hand=cooperative_hand_reference())
                seed=np.asarray(root).copy();seed[7:]=hand
                q,ep,er,_=self.safety.solve(seed,target,iterations=50)
                if ep<.008 and er<.1:seeds.append((q,finger,'cooperative_template'))
                if len(seeds)>=12:break
            if len(seeds)>=12:break
        for seed,finger,kind in seeds:
            points,normals=self.project_pads(seed);points[finger]=p;normals[finger]=n
            patch=self.fit_patch(seed,points,normals,anchor=finger,clearance=clearance,reproject=2,anchor_weight=1.8)
            attempts.append(dict(seed=kind,anchor=finger,reason=patch['reason'],contacts=int(patch['expected_contacts'].sum())))
            if patch['reason'] is None:
                score=np.linalg.norm(patch['q'][:7]-root[:7])+.15*np.linalg.norm(patch['q'][7:]-root[7:])
                score+=.7*(4-patch['expected_contacts'].sum())
                if score<best_score:best=patch;best_score=score
                if patch['expected_contacts'].all() or len(attempts)>=6:break
            if perf_counter()-begin>budget_s and attempts:break
        if best is not None:best['attempts']=attempts
        return best,attempts

    def group_route(self, root, initial_points, initial_normals, anchor, goal, *, maximum_m=.12):
        start=self.graph.nearest_vertex(initial_points[anchor]-self.h.object_position_m)[0]
        route,route_normals=self.surface_route(start,int(goal),spacing=.003)
        if len(route)<2:return None
        route+=self.h.object_position_m;route[0]=initial_points[anchor]
        route_normals[0]=initial_normals[anchor]
        points=np.asarray(initial_points).copy();normals=np.asarray(initial_normals).copy()
        qs=[root.copy()];pps=[points.copy()];nns=[normals.copy()];distances=[0.];expected=[]
        for k in range(1,len(route)):
            turn=align_vector(route_normals[k-1],route_normals[k])
            guesses=route[k]+turn.apply(points-route[k-1])
            projected,_,_=trimesh.proximity.closest_point(self.graph.mesh,guesses-self.h.object_position_m)
            points=projected+self.h.object_position_m
            normals=np.array([self.smooth_normal(p) for p in projected]);points[anchor]=route[k];normals[anchor]=route_normals[k]
            patch=self.fit_patch(qs[-1],points,normals,anchor=anchor,iterations=35,flexible=False)
            if patch['reason'] is not None or not self.safety.edge_free(qs[-1],patch['q']):break
            length=distances[-1]+float(np.linalg.norm(route[k]-route[k-1]))
            qs.append(patch['q']);pps.append(points.copy());nns.append(normals.copy());distances.append(length)
            expected.append(patch['expected_contacts'])
            if length>=maximum_m:break
        if len(qs)<3 or distances[-1]<.006:return None
        return dict(q=np.array(qs),points=np.array(pps),normals=np.array(nns),distance=np.array(distances),
                    anchor=int(anchor),expected_contacts=np.array(expected),goal=int(goal),
                    complete=len(qs)==len(route))

    def path_between(self, root, goal):
        if self.safety.edge_free(root,goal):points=np.array([root,goal])
        else:points=self.safety._rrt(root,goal)
        if points is None:return None
        durations=[]
        for a,b in zip(points[:-1],points[1:]):
            move=spin=0.
            for u in np.linspace(0.,1.,9):
                set_kinematics(self.h,self.data,a+u*(b-a))
                jp,jr=position_jacobian(self.h,self.data,self.h.palm_site_id)
                move=max(move,float(np.linalg.norm(jp@(b-a))));spin=max(spin,float(np.linalg.norm(jr@(b-a))))
            durations.append(max(.3,1.9*np.max(abs(b-a))/.6,1.9*move/.04,1.9*spin/.5))
        set_kinematics(self.h,self.data,root);start=pose(self.h,self.data)
        set_kinematics(self.h,self.data,goal);finish=pose(self.h,self.data)
        return JointPath(points,np.array(durations),start,finish,0.)
