"""Whole-mustard palmar exploration with a bounded, lightly flexed hand.

New development entry; historical trials are immutable. Geometry supplies a
route, never contact or coverage. The original object and force rules remain.
Canonical long-finger order is abduction, MCP flexion, PIP, DIP (not XML order).
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import least_squares
import trimesh

from Module.fr3_leap import ARM_HOME_Q
from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from .contracts import ExperimentConfig, WristTarget
from .full_surface import build_scene, JointPath
from .physics import set_joint_state, set_control, joint_state, pose, rotation, quaternion, contact_measurement, position_jacobian
from .sliding_safety import SlidingSafetyPlanner, arm_self_contacts, is_nonadjacent_arm_pair
from .belly_contact import BellyContactConfig, belly_contact_measurement, belly_ik_terms
from .belly_bounded_step import bounded_belly_step
from .oracle import MeasuredCoverage


def unit(v):
    v=np.asarray(v,float)
    return v/max(float(np.linalg.norm(v)),1e-12)


def route_endpoint_reached(progress,length,safe,contact,endpoint,*,pivot=False):
    """A segment handoff, never a coverage or whole-run completion test.

    Brief losses are normal while sliding. Requiring a new 150 ms contact
    bout at every mesh segment stranded the hand at already reached points.
    The small measured endpoint tolerance applies only to translation.
    """
    if not safe:return False
    if progress>=length-1e-10:return True
    return bool(not pivot and length-progress<=.008 and
                np.linalg.norm(np.asarray(contact)-endpoint)<=.005)


def set_kinematics(h,data,q):
    """Scratch FK/Jacobians only; physical data always uses native dynamics."""
    data.qpos[h.arm_qpos_adrs]=q[:7];data.qpos[h.hand_qpos_adrs]=q[7:]
    mujoco.mj_kinematics(h.model,data);mujoco.mj_comPos(h.model,data)


class NaturalSafetyPlanner(SlidingSafetyPlanner):
    """Identical original contact-distance rules, without scratch dynamics.

    Geometry queries need native collision detection, not a constraint-force
    solve. The runtime's actual plant is unchanged and still uses mj_step.
    """
    def solve(self,q_root,target,*,iterations=100):
        if not isinstance(target,WristTarget):return super().solve(q_root,target,iterations=iterations)
        desired=rotation(target.pose_world[3:]);fixed=np.clip(np.asarray(q_root).copy(),self.lower,self.upper);cached=None
        if not hasattr(self,'_avoid_pairs'):
            bodies=sorted(set(map(int,self.h.model.geom_bodyid[self._arm_geoms])))
            self._avoid_pairs=[(a,b) for k,a in enumerate(bodies) for b in bodies[k+1:] if is_nonadjacent_arm_pair(self.h.model,a,b)]
        avoidance_error=np.zeros(len(self._avoid_pairs));avoidance_jac=np.zeros((len(self._avoid_pairs),7))
        def update(arm):
            nonlocal cached
            if cached is None or not np.array_equal(cached,arm):
                q=fixed.copy();q[:7]=arm;set_kinematics(self.h,self.scratch,q);cached=arm.copy()
                m=self.h.model;before=m.geom_margin.copy();avoidance_error[:]=0.;avoidance_jac[:]=0.;minimum={}
                try:
                    m.geom_margin[self._arm_geoms]=np.maximum(before[self._arm_geoms],.009)
                    mujoco.mj_collision(m,self.scratch)
                finally:m.geom_margin[:]=before
                for c in self.scratch.contact:
                    ba,bb=int(m.geom_bodyid[int(c.geom1)]),int(m.geom_bodyid[int(c.geom2)]);pair=tuple(sorted((ba,bb)))
                    if pair not in self._avoid_pairs or c.dist>=.008:continue
                    index=self._avoid_pairs.index(pair)
                    if float(c.dist)>=minimum.get(index,float('inf')):continue
                    minimum[index]=float(c.dist);ja,jb,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv)),np.zeros((3,m.nv))
                    mujoco.mj_jac(m,self.scratch,ja,jr,np.asarray(c.pos),ba);mujoco.mj_jac(m,self.scratch,jb,jr,np.asarray(c.pos),bb)
                    avoidance_error[index]=3.*(.008-float(c.dist))
                    avoidance_jac[index]=-3.*np.asarray(c.frame[:3])@(jb-ja)[:,self.h.arm_dof_adrs]
        def error(arm):
            update(arm);current=pose(self.h,self.scratch)
            return np.r_[current[:3]-target.pose_world[:3],.10*(rotation(current[3:])*desired.inv()).as_rotvec(),avoidance_error]
        def jacobian(arm):
            update(arm);current=pose(self.h,self.scratch);e=(rotation(current[3:])*desired.inv()).as_rotvec();angle=np.linalg.norm(e)
            skew=np.array([[0.,-e[2],e[1]],[e[2],0.,-e[0]],[-e[1],e[0],0.]])
            coefficient=1/12 if angle<1e-4 else (1-.5*angle/np.tan(.5*angle))/angle**2
            left_inverse=np.eye(3)-.5*skew+coefficient*skew@skew
            jp,jr=position_jacobian(self.h,self.scratch,self.h.palm_site_id)
            return np.vstack((jp[:,:7],.10*left_inverse@jr[:,:7],avoidance_jac))
        best=least_squares(error,fixed[:7],bounds=(self.lower[:7],self.upper[:7]),jac=jacobian,
            max_nfev=iterations,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q=fixed.copy();q[:7]=best.x;set_kinematics(self.h,self.scratch,q);current=pose(self.h,self.scratch)
        return q,float(np.linalg.norm(current[:3]-target.pose_world[:3])),float((rotation(current[3:])*desired.inv()).magnitude()),0.

    def collision_reason(self,q):
        m=self.h.model;objects=set(map(int,self.h.object_geom_ids));tips=set(map(int,self.h.tip_geom_ids))
        margins=m.geom_margin.copy();self._arm_clearances={};self._base_reason=None
        try:
            m.geom_margin[self.h.object_geom_ids]+=.004
            m.geom_margin[self._arm_geoms]=np.maximum(margins[self._arm_geoms],self.arm_clearance_m+1e-6)
            set_kinematics(self.h,self.scratch,q);mujoco.mj_collision(m,self.scratch)
        finally:m.geom_margin[:]=margins
        for c in self.scratch.contact:
            a,b=int(c.geom1),int(c.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
            if is_nonadjacent_arm_pair(m,ba,bb):
                pair=tuple(sorted((ba,bb)));self._arm_clearances[pair]=min(self._arm_clearances.get(pair,float('inf')),float(c.dist))
            if self._base_reason is None:
                if objects.intersection((a,b)):
                    other=b if a in objects else a
                    if other not in tips and c.dist<-.0005:self._base_reason='NON_TIP_OBJECT_COLLISION'
                elif c.dist<-.002 and ba!=bb and m.body_parentid[ba]!=bb and m.body_parentid[bb]!=ba:
                    self._base_reason='SELF_COLLISION'
        if self._base_reason is None:
            for c in self.scratch.contact:
                pair=(int(c.geom1),int(c.geom2))
                if objects.intersection(pair):
                    if not tips.intersection(pair) and c.dist<.003:self._base_reason='NON_TIP_CLEARANCE';break
                    if tips.intersection(pair) and c.dist<-.002:self._base_reason='DEEP_PAD_PENETRATION';break
        if any(d<=0. for d in self._arm_clearances.values()):return 'SHALLOW_ARM_SELF_COLLISION'
        if self._base_reason:return self._base_reason
        if any(d<self.arm_clearance_m for d in self._arm_clearances.values()):return 'ARM_SELF_CLEARANCE'
        return None


def hand_reference(finger=0, broad=False,curved=False):
    """Small splay, no reverse distal flexion; idle fingers clear the surface.

    The thumb is gently opened alongside the palm. It is not forced to oppose
    the other three fingers on one locally flat face.
    """
    hand=np.r_[np.tile([0.,-.08,.025,.025],3),[-.22,-.20,.04,.04]]
    for i in (range(3) if broad else (finger,)):
        hand[4*i:4*i+4]=([.1,1.,.2,.2] if i==3 else ([0.,.2,.4,.3] if curved else [0.,.12,.025,.025]))
    return hand


def posture_bounds(lower,upper):
    lo,hi=np.asarray(lower).copy(),np.asarray(upper).copy()
    lo[7:19]=np.tile([-.18,-.12,0.,0.],3)
    hi[7:19]=np.tile([.18,.65,.65,.60],3)
    lo[19:]=[-.28,-.30,0.,0.]
    hi[19:]=[.4,1.8,1.,1.]
    return np.maximum(lo,lower),np.minimum(hi,upper)


def raster_goals(graph,pitch=.032):
    """Serpentine body bands plus both ends; all original goals retained."""
    vertices=graph.required_vertices;p=graph.vertices_m[vertices];n=graph.normals[vertices]
    axis=np.argmax(abs(n),axis=1);sign=np.where(n[np.arange(len(n)),axis]>=0,1,-1)
    ordered=[]
    for a,s in ((2,1),(0,1),(2,-1),(0,-1),(1,1),(1,-1)):
        ids=np.flatnonzero((axis==a)&(sign==s))
        minor=0 if a==2 else 2;major=1 if a!=1 else 0
        row=np.floor((p[ids,minor]-p[:,minor].min())/pitch).astype(int)
        for count,r in enumerate(np.unique(row)):
            items=ids[row==r]
            items=items[np.argsort(p[items,major])]
            if count%2==0:items=items[::-1]
            ordered.extend(vertices[items].tolist())
    assert len(ordered)==len(vertices) and len(set(ordered))==len(vertices)
    ordered=np.array(ordered,int)
    first_count=int(np.count_nonzero((axis==2)&(sign==1)))
    seed=np.flatnonzero(ordered[:first_count]==vertices[0])
    if len(seed):
        k=int(seed[0]);ordered=np.r_[ordered[k:first_count],ordered[:k],ordered[first_count:]]
    return ordered


class NaturalBellyPlanner:
    def __init__(self,h,graph):
        self.h,self.graph=h,graph
        self.safety=NaturalSafetyPlanner(h,ExperimentConfig())
        self.lower,self.upper=posture_bounds(self.safety.lower,self.safety.upper)
        self.data=mujoco.MjData(h.model)
        self.order=np.r_[h.arm_dof_adrs,h.hand_dof_adrs]
        self.templates={}
        self.failures={}

    def template(self,finger=0,broad=False,curved=False):
        key=(finger,broad,curved)
        if key not in self.templates:
            q=np.r_[ARM_HOME_Q,hand_reference(finger,broad,curved)]
            set_joint_state(self.h,self.data,q)
            palm=pose(self.h,self.data);R=rotation(palm[3:]).as_matrix()
            g=int(self.h.tip_geom_ids[finger])
            self.templates[key]=((self.data.geom_xpos[g]-palm[:3])@R,
                R.T@self.data.geom_xmat[g].reshape(3,3),q[7:].copy())
        return self.templates[key]

    def target(self,p,n,tangent,*,finger=0,broad=False,clearance=.003,roll=0.,curved=False,root_hand=None):
        if root_hand is None:c,M,hand=self.template(finger,broad,curved)
        else:
            hand=np.asarray(root_hand).copy();set_kinematics(self.h,self.data,np.r_[ARM_HOME_Q,hand])
            palm=pose(self.h,self.data);R=rotation(palm[3:]).as_matrix();g=int(self.h.tip_geom_ids[finger])
            c=(self.data.geom_xpos[g]-palm[:3])@R;M=R.T@self.data.geom_xmat[g].reshape(3,3)
        n=unit(n);x=np.asarray(tangent)-n*float(n@tangent)
        if np.linalg.norm(x)<.05:
            basis=np.eye(3)[np.argmin(abs(n))];x=basis-n*(n@basis)
        x=unit(x);x=Rotation.from_rotvec(roll*n).apply(x)
        z=-n;y=unit(np.cross(z,x));x=np.cross(y,z)
        R=np.column_stack((x,y,z))@M.T
        center=np.asarray(p)+(float(self.h.model.geom_size[self.h.tip_geom_ids[finger],2])+clearance)*n
        return WristTarget(np.r_[center-R@c,quaternion(Rotation.from_matrix(R))]),hand

    def solve(self,root,p,n,tangent=(0,1,0),*,finger=0,broad=False,clearance=.003,roll=0.,iterations=65,curved=False,preserve_hand=False):
        target,hand=self.target(p,n,np.asarray(tangent),finger=finger,broad=broad,clearance=clearance,roll=roll,
                               curved=curved,root_hand=np.asarray(root)[7:] if preserve_hand else None)
        seed=np.asarray(root).copy();seed[7:]=hand
        q,ep,er,_=self.safety.solve(seed,target,iterations=iterations)
        reason='WRIST_IK' if ep>.0004 or er>.008 else self.safety.collision_reason(q)
        return q,dict(reason=reason,position_error_m=ep,rotation_error_rad=er,
                      finger=finger,broad=broad,roll_rad=roll,curved=curved,preserve_hand=preserve_hand)

    def find(self,root,p,n,*,tangent=(0,1,0),clearance=.003,budget_s=5.,preferred_roll=0.,require_approach=False):
        started=perf_counter();records=[];best=None;best_score=float('inf');valid=0
        rolls=[preferred_roll]+[preferred_roll+s*np.pi/4 for s in (1,-1,2,-2,3,-3,4)]
        # Neighbouring configurations first. Alternate elbow seeds are finite.
        seeds=[np.asarray(root).copy(),np.r_[ARM_HOME_Q,hand_reference()]]
        for seed in seeds:
            for roll in rolls:
                for finger,curved in ((0,False),(1,False),(2,False),(0,True),(1,True),(2,True),(3,False)):
                    q,diag=self.solve(seed,p,n,tangent,finger=finger,clearance=clearance,roll=roll,curved=curved)
                    if diag['reason'] is None and require_approach:
                        touch,td=self.solve(q,p,n,tangent,finger=finger,clearance=.0002,roll=roll,preserve_hand=True)
                        if td['reason'] is not None or not self.safety.edge_free(q,touch):
                            diag['reason']='HOVER_WITHOUT_SAFE_APPROACH'
                        else:diag['contact_q']=touch.tolist()
                    records.append(diag)
                    if diag['reason'] is None:
                        if not require_approach:return q,diag
                        # A feasible thumb pose can require a full arm loop.
                        # Compare a small set before committing to that pose.
                        valid+=1
                        score=float(np.linalg.norm(q[:7]-root[:7])+.12*np.linalg.norm(q[7:]-root[7:]))
                        if self.safety.edge_free(root,q):score*=.45
                        if score<best_score:best=(q,diag);best_score=score
                        if score<.25 or valid>=6:return best
                    if perf_counter()-started>budget_s:
                        if best is not None:return best
                        return None,dict(reason='FINITE_GEOMETRY_BUDGET',attempts=records)
        if best is not None:return best
        return None,dict(reason='NO_NATURAL_POSTURE',attempts=records)

    def smooth_normal(self,p):
        dist,ind=self.graph._tree.query(p,k=12)
        return unit(np.sum(self.graph.normals[ind]/np.maximum(dist[:,None],.002),axis=0))

    def surface_route(self,start_vertex,goal_vertex,spacing=.003):
        distances,parents=dijkstra(self.graph.adjacency,directed=False,indices=int(start_vertex),return_predecessors=True)
        route=[int(goal_vertex)]
        while route[-1]!=start_vertex:
            node=int(parents[route[-1]])
            if node<0:raise ValueError('Disconnected requested surface')
            route.append(node)
        xyz=self.graph.vertices_m[route[::-1]]
        if len(xyz)<2:return xyz.copy(),np.array([self.smooth_normal(xyz[0])])
        # Resample the actual mesh route, then project smoothing back to mesh.
        for _ in range(2):
            if len(xyz)>2:xyz[1:-1]=.25*xyz[:-2]+.5*xyz[1:-1]+.25*xyz[2:]
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        samples=np.linspace(0,length[-1],max(2,int(np.ceil(length[-1]/spacing))+1))
        raw=np.column_stack([np.interp(samples,length,xyz[:,j]) for j in range(3)])
        projected,_,_=trimesh.proximity.closest_point(self.graph.mesh,raw)
        normals=np.array([self.smooth_normal(p) for p in projected])
        return projected,normals

    def slide_plan(self,root,points,normals,finger,tangent):
        """Continue the pad's tangent frame without a pole/roll discontinuity."""
        from .oracle import align_vector
        qs=[];ts=[];previous=np.asarray(root);old_n=normals[0];t=unit(tangent-old_n*(old_n@tangent))
        for k,(p,n) in enumerate(zip(points,normals)):
            if k:t=unit(align_vector(old_n,n).apply(t));old_n=n
            q,diag=self.solve(previous,p+self.h.object_position_m,n,t,finger=finger,clearance=.0002,iterations=40,preserve_hand=True)
            if diag['reason'] is not None or not self.safety.edge_free(previous,q):
                if len(qs)<3:return None,dict(reason='SURFACE_CONTINUATION_BLOCKED',at=k,detail=diag)
                break
            qs.append(q);ts.append(t.copy());previous=q
        count=len(qs);p=points[:count];n=normals[:count]
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if distance[-1]<.001:return None,dict(reason='NO_SURFACE_ADVANCE')
        return dict(points=p,normals=n,tangents=np.array(ts),q=np.array(qs),distance=distance,
                    finger=finger,complete=count==len(points)),None

    def transfer(self,root,p,n,normal_now,*,budget_s=12.):
        """Minimum tested outward clearance first; every edge is verified."""
        started=perf_counter();records=[];best_path=None;best_diag=None
        root=np.asarray(root).copy();set_joint_state(self.h,self.data,root)
        wrist=pose(self.h,self.data)
        from .oracle import align_vector
        # Carry the current hand's roll forward. A global +Y reset used to
        # provoke an unnecessary 90/180-degree wrist turn on each transfer.
        current_axes=self.data.geom_xmat[self.h.tip_geom_ids[:3]].reshape(3,3,3)
        tangent=unit(align_vector(normal_now,n).apply(np.mean(current_axes[:,:,0],axis=0)))
        for distance in (.006,.015,.035,.065):
            outward=wrist.copy();outward[:3]+=distance*unit(normal_now)
            retreat,ep,er,_=self.safety.solve(root,WristTarget(outward),iterations=60)
            if ep>.0005 or er>.01 or not self.safety.edge_free(root,retreat):continue
            goal,diag=self.find(retreat,p,n,tangent=tangent,clearance=distance,
                budget_s=min(3.,max(.1,budget_s-(perf_counter()-started))),require_approach=True)
            records.append(dict(clearance_m=distance,find=diag))
            if goal is None:
                if perf_counter()-started>budget_s:break
                continue
            if self.safety.edge_free(retreat,goal):path=np.array([root,retreat,goal])
            else:
                middle=self.safety._rrt(retreat,goal)
                if middle is None:
                    if perf_counter()-started>budget_s:break
                    continue
                path=np.vstack((root,middle))
            durations=[]
            for segment,(a,b) in enumerate(zip(path[:-1],path[1:])):
                # Time bounds on joint, wrist translation and rotation.
                move=spin=0.
                from .physics import position_jacobian
                for u in np.linspace(0,1,15):
                    set_joint_state(self.h,self.data,a+u*(b-a))
                    jp,jr=position_jacobian(self.h,self.data,self.h.palm_site_id)
                    move=max(move,np.linalg.norm(jp@(b-a)))
                    spin=max(spin,np.linalg.norm(jr@(b-a)))
                air_speed,air_spin=(.012,.18) if segment==0 else (.04,.5)
                durations.append(max(.3,1.90*np.max(abs(b-a))/.6,1.90*move/air_speed,1.90*spin/air_spin))
            candidate_path=JointPath(path,np.array(durations),wrist,pose(self.h,self.data),perf_counter()-started)
            if best_path is None or candidate_path.duration_s<best_path.duration_s:
                best_path=candidate_path;best_diag=dict(**diag,clearance_m=distance,attempts=records.copy(),
                    tangent_world=Rotation.from_rotvec(diag['roll_rad']*n).apply(tangent).tolist())
            if best_path.duration_s<12. or perf_counter()-started>budget_s:break
        if best_path is not None:return best_path,best_diag
        return None,dict(reason='NO_VALIDATED_TRANSFER',attempts=records)

    def minimum_arm_clearance(self,q,query_m=.006):
        m=self.h.model;s=self.safety;before=m.geom_margin.copy()
        try:
            m.geom_margin[s._arm_geoms]=np.maximum(before[s._arm_geoms],query_m)
            set_kinematics(self.h,s.scratch,q);mujoco.mj_collision(m,s.scratch)
        finally:m.geom_margin[:]=before
        return min((float(c.dist) for c in s.scratch.contact if is_nonadjacent_arm_pair(
            m,int(m.geom_bodyid[int(c.geom1)]),int(m.geom_bodyid[int(c.geom2)]))),default=query_m)

    def minimum_nonpad_clearance(self,q,query_m=.008):
        m=self.h.model;s=self.safety;before=m.geom_margin.copy()
        try:
            m.geom_margin[self.h.object_geom_ids]+=query_m
            set_kinematics(self.h,s.scratch,q);mujoco.mj_collision(m,s.scratch)
        finally:m.geom_margin[:]=before
        objects=set(map(int,self.h.object_geom_ids));tips=set(map(int,self.h.tip_geom_ids));distance=query_m
        for c in s.scratch.contact:
            a,b=int(c.geom1),int(c.geom2)
            if a in objects and b not in objects:other=b
            elif b in objects and a not in objects:other=a
            else:continue
            if other not in tips:distance=min(distance,float(c.dist))
        return distance

    def arm_buffer_terms(self,q,columns,margin_m=.006):
        self.minimum_arm_clearance(q,query_m=margin_m+1e-6)
        m=self.h.model;rows=[];errors=[];pairs={}
        for c in self.safety.scratch.contact:
            ba,bb=int(m.geom_bodyid[int(c.geom1)]),int(m.geom_bodyid[int(c.geom2)])
            if not is_nonadjacent_arm_pair(m,ba,bb) or c.dist>=margin_m:continue
            pair=tuple(sorted((ba,bb)))
            if c.dist>=pairs.get(pair,float('inf')):continue
            pairs[pair]=float(c.dist)
            ja,jb,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv)),np.zeros((3,m.nv))
            mujoco.mj_jac(m,self.safety.scratch,ja,jr,np.asarray(c.pos),ba)
            mujoco.mj_jac(m,self.safety.scratch,jb,jr,np.asarray(c.pos),bb)
            rows.append(4.*np.asarray(c.frame[:3])@(jb-ja)[:,self.order[columns]])
            errors.append(4.*(float(c.dist)-margin_m))
        return np.asarray(rows).reshape(-1,len(columns)),np.asarray(errors)

    def clearance_exit(self,measured,previous,*,arm_target_m=.0012,object_target_m=.0035,preserve_wrist=False):
        """Find a separating command inside the original clearance guard.

        Holding the previous reference is not necessarily valid after servo
        lag has entered a positive clearance buffer. This is an actual
        separating motion, not an exemption from the original threshold.
        """
        safety=self.safety;h=self.h;m=h.model
        safety.collision_reason(previous)
        margins=m.geom_margin.copy()
        try:
            m.geom_margin[safety._arm_geoms]=np.maximum(margins[safety._arm_geoms],arm_target_m+1e-6)
            m.geom_margin[h.object_geom_ids]+=max(.004,object_target_m+1e-6)
            mujoco.mj_collision(m,safety.scratch)
        finally:m.geom_margin[:]=margins
        objects=set(map(int,h.object_geom_ids));tips=set(map(int,h.tip_geom_ids));rows=[];errors=[];arm_rows={};pad_rows=[]
        for ct in safety.scratch.contact:
            a,b=int(ct.geom1),int(ct.geom2)
            ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
            if is_nonadjacent_arm_pair(m,ba,bb) and ct.dist<arm_target_m:
                pair=tuple(sorted((ba,bb)))
                if pair not in arm_rows or ct.dist<arm_rows[pair][0]:
                    ja,jb,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv)),np.zeros((3,m.nv))
                    mujoco.mj_jac(m,safety.scratch,ja,jr,np.asarray(ct.pos),ba)
                    mujoco.mj_jac(m,safety.scratch,jb,jr,np.asarray(ct.pos),bb)
                    arm_rows[pair]=(float(ct.dist),np.asarray(ct.frame[:3])@(jb-ja)[:,self.order])
            if a in objects and b not in objects:other=b;n=np.array(ct.frame[:3])
            elif b in objects and a not in objects:other=a;n=-np.array(ct.frame[:3])
            else:continue
            threshold=.0002 if other in tips else object_target_m
            if ct.dist>=threshold:continue
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
            mujoco.mj_jac(m,safety.scratch,jp,jr,np.asarray(ct.pos),int(m.geom_bodyid[other]))
            if other in tips:pad_rows.append(len(rows))
            rows.append(n@jp[:,self.order]);errors.append(-min(.00025,threshold-float(ct.dist)))
        for distance,row in arm_rows.values():
            rows.append(8.*row);errors.append(-8.*min(.00008,arm_target_m-distance))
        if preserve_wrist and arm_rows:
            for index in pad_rows:errors[index]=0.
            jp,jr=position_jacobian(h,safety.scratch,h.palm_site_id)
            rows.extend(4.*jp);errors.extend(np.zeros(3))
            rows.extend(.4*jr);errors.extend(np.zeros(3))
        if not rows:return None
        delta,diag=bounded_belly_step(np.array(rows),np.array(errors),previous,self.lower,self.upper,
            gain=1.,maximum_joint_step=.0024,damping=.001)
        if not diag['feasible']:return None
        for factor in (1.,.75,.5,.25):
            trial=previous+factor*delta
            if safety.command_collision_reason(measured,trial) is None:return trial
        return None


def run(output,duration_s=1800.,max_routes=1200,resume=None):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    h,g,home,quality=build_scene(out)
    planner=NaturalBellyPlanner(h,g);cfg=ExperimentConfig();bcfg=BellyContactConfig()
    coverage=MeasuredCoverage(g,cfg)
    # A declared new, collision-free hover is the sole physical initialization.
    raster=raster_goals(g);raster_index=np.full(len(g.vertices_m),len(raster),int);raster_index[raster]=np.arange(len(raster))
    vertex=int(raster[0]);point=g.vertices_m[vertex];normal=planner.smooth_normal(point)
    q,initial_plan=planner.find(home,point+h.object_position_m,normal,clearance=.006)
    if q is None:raise RuntimeError('INITIAL_NATURAL_HOVER_UNREACHABLE')
    finger=int(initial_plan['finger']);tangent=Rotation.from_rotvec(initial_plan['roll_rad']*normal).apply(unit(np.array([0.,1.,0.])-normal*normal[1]))
    data=mujoco.MjData(h.model);set_joint_state(h,data,q);set_control(h,data,q)
    inherited=[];parent_summary=None;parent_manifest=None;resume_info=None
    if resume is not None:
        from dataclasses import replace
        from Module.fr3_leap.model import compile_model_xml
        parent=Path(resume).resolve()
        parent_audit=json.loads((parent/'independent_native_audit/audit.json').read_text())
        if parent_audit['status']!='PASSED_NATIVE_EVIDENCE_AUDIT':raise ValueError('Resume requires a passed continuous native audit')
        parent_summary=json.loads((parent/'summary.json').read_text());parent_manifest=json.loads((parent/'manifest.json').read_text())
        for name in ('summary.json','manifest.json','model.xml','final_state.npz','initial.npz'):
            if sha256((parent/name).read_bytes()).hexdigest()!=parent_audit['source_files'][name]:raise ValueError('Resume evidence changed: '+name)
        xml=(parent/'model.xml').read_text(encoding='utf-8');model=compile_model_xml(xml)
        h.base=replace(h.base,model=model,xml=xml);planner=NaturalBellyPlanner(h,g);scratch=mujoco.MjData(model)
        data=mujoco.MjData(model)
        with np.load(parent/'final_state.npz') as z:
            for name in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(data,name)[:]=z[name]
            data.time=float(z['time']);mask=z['visited_vertices'].copy()
        with np.load(parent/'independent_native_audit/verified_preview.npz') as z:
            np.testing.assert_array_equal(mask,z['covered_vertices'])
        coverage.visited_vertices[:]=mask;coverage.visited_goals[:]=mask[g.required_vertices]
        mujoco.mj_forward(h.model,data);q=np.r_[data.ctrl[h.arm_actuator_ids],data.ctrl[h.hand_actuator_ids]]
        f,p,n,a,unsafe,pen=contact_measurement(h,data,.2)
        finger=int(np.argmax(f))
        if a[finger]:
            projected,_,_=trimesh.proximity.closest_point(g.mesh,(p[finger]-h.object_position_m)[None,:])
            point=projected[0];normal=n[finger].copy()
        else:
            center=data.geom_xpos[h.tip_geom_ids[finger]]-h.object_position_m
            projected,_,_=trimesh.proximity.closest_point(g.mesh,center[None,:]);point=projected[0];normal=planner.smooth_normal(point)
        tangent=data.geom_xmat[h.tip_geom_ids[finger]].reshape(3,3)[:,0].copy();vertex=g.nearest_vertex(point)[0]
        for chunk in parent_summary['trace_chunks']:
            source=(parent/chunk['file']).resolve()
            if sha256(source.read_bytes()).hexdigest()!=chunk['sha256']:raise ValueError('Resume native chunk changed')
            inherited.append(dict(chunk,file=os.path.relpath(source,out).replace('\\','/')))
        # A stopped, un-applied observation at the join is replaced by the
        # first continuation observation at the identical native state.
        if abs(inherited[-1]['end_s']-data.time)<1e-8:
            inherited[-1]['skip_last']=inherited[-1].get('skip_last',0)+1
            inherited[-1]['states']-=1;inherited[-1]['end_s']-=.002
        resume_info=dict(parent_case=str(parent),parent_audit_sha256=sha256((parent/'independent_native_audit/audit.json').read_bytes()).hexdigest(),
            parent_manifest_sha256=sha256((parent/'manifest.json').read_bytes()).hexdigest(),
            resume_state_sha256=sha256((parent/'final_state.npz').read_bytes()).hexdigest(),join_time_s=data.time,
            no_physical_motion_at_join=True,scope='Contiguous multi-version development lineage; not a fresh single-version run')
        (out/'run_initial.npz').write_bytes((parent/'final_state.npz').read_bytes())
        (out/'initial.npz').write_bytes((parent/'initial.npz').read_bytes())
    (out/'model.xml').write_text(h.xml,encoding='utf-8')
    if resume is None:np.savez_compressed(out/'initial.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,qacc_warmstart=data.qacc_warmstart,time=data.time)
    sources={};root=Path(__file__).resolve().parents[2]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            file=Path(filename).resolve()
            if file.suffix=='.py' and file.is_relative_to(root/'Module'):
                rel=file.relative_to(root).as_posix();raw=file.read_bytes();sources[rel]=sha256(raw).hexdigest()
                destination=out/'source_snapshot'/rel;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    manifest=dict(protocol='i04-natural-belly-whole.development.v1.11',cell='B-MCC',dp='DEFERRED_BY_USER',
        source_hashes=sources,geometry_quality=quality,belly_contact=asdict(bcfg),coverage_inherited=resume is not None,
        coverage_radius_m=.025,required_vertices=len(g.vertices_m),required_goals=g.required_goal_count,
        initial_state=('Original lineage hover followed by verified exact-state continuation; see resume metadata'
                       if resume is not None else 'New declared natural-hand 6 mm collision-free hover; one physical initialization'),
        initial_plan=initial_plan,posture_lower_rad=planner.lower.tolist(),posture_upper_rad=planner.upper.tolist(),
        original_collision_geometry=True,force_relief_n=8.,hard_force_n=12.,hard_force_duration_s=.020,
        command_geometry_guard_every_native_tick=True,dt_s=.002,requested_duration_s=duration_s,
        free_transfer_wrist_speed_m_s=.04,free_transfer_rotation_rad_s=.5,
        withdrawal_wrist_speed_m_s=.012,sliding_speed_m_s=.008,free_approach_speed_m_s=.025,
        proactive_release_n=3.,release_resume_n=1.5,first_belly_touch_handoff=True,resume=resume_info,
        segment_stable_dwell_s=0.,final_stable_dwell_s=.15,measured_endpoint_tolerance_m=.005,
        transfer_approach_geometry_prechecked=True,compare_transfer_durations=True,active_positive_arm_buffer_exit=True,
        arm_recovery_target_m=.004,arm_recovery_wrist_regularization=True,ik_arm_soft_clearance_m=.008,
        mcc_arm_soft_clearance_m=.006,command_reference_guard_every_native_tick=True,
        approach_nominal_follows_clearance=True,independent_rejected_finger_release=True,
        nonpad_recovery_target_m=.006)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    logs={};events=[];chunks=inherited;last=q.copy();peak=0.;high_steps=0;rejections=0;routes=0;lifts=0
    if resume is not None:
        peak=parent_summary['peak_force_n'];routes=parent_summary['routes'];lifts=parent_summary['necessary_lifts']
        events=json.loads((parent/'events.json').read_text())
        events.append(dict(time_s=data.time,kind='VERIFIED_CONTIGUOUS_CONTROLLER_UPDATE',**resume_info))
    offsets=np.zeros(4);applied=0;phase='INITIAL';stop='UNFINISHED';last_progress=perf_counter()
    scratch=mujoco.MjData(h.model);failed=np.zeros(len(g.vertices_m),int);done_hold=0.
    states=sum(c['states'] for c in inherited);summary_path=out/'progress.json'
    controllers=[FingertipMCC(MCCConfig(virtual_mass=.08,damping=30.,stiffness=40.,dt_s=.002,
        max_offset_m=.004,max_velocity_m_s=.012,max_acceleration_m_s2=1.)) for _ in range(4)]
    nominal_previous=q.copy()
    pivots_tried=set()
    gateways_tried=set()
    relief_active=False;relief_count=0;last_handoff=-np.inf
    force_retries={}
    released_idle_fingers=set()
    np.savez_compressed(out/'global_raster.npz',goal_vertices=raster,goal_points_local_m=g.vertices_m[raster],
                        original_required_vertices=g.required_vertices)

    def flush():
        nonlocal logs
        if not logs:return
        name=f'trace_{len(chunks):04d}.npz';np.savez_compressed(out/name,**{k:np.asarray(v) for k,v in logs.items()})
        chunks.append(dict(file=name,states=len(logs['time_s']),start_s=float(logs['time_s'][0]),end_s=float(logs['time_s'][-1])))
        logs={}
        np.savez_compressed(out/'checkpoint.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,
            qacc_warmstart=data.qacc_warmstart,time=data.time,visited_vertices=coverage.visited_vertices,
            visited_goals=coverage.visited_goals)
        (out/'trace_chunks.json').write_text(json.dumps(chunks,indent=2))
        (out/'events.json').write_text(json.dumps(events,indent=2))

    def measure():
        f,p,n,a,unsafe,pen=contact_measurement(h,data,.2)
        bf,bp,bn,ba,bd=belly_contact_measurement(h,data,.2,bcfg)
        arm=arm_self_contacts(h,data)
        safe=ba & (f<8.) & (np.asarray(bd['rejected_contact_force_n'])<.2)
        if unsafe or arm or f.max()>=8.:safe[:]=False
        return f,p,n,a,unsafe,pen,bf,bp,bn,ba,bd,arm,safe

    def step(command,phase,observation=None,goal_vertex=-1):
        nonlocal last,high_steps,peak,rejections,applied,last_progress,states,done_hold,relief_active,relief_count
        if states%500==0 and (out/'development_stop.json').exists():
            raise RuntimeError('DEVELOPMENT_STOP_REQUESTED')
        obs=measure() if observation is None else observation
        f,p,n,a,unsafe,pen,bf,bp,bn,ba,bd,arm,safe=obs
        coverage.update(bp-h.object_position_m,bn,safe)
        peak=max(peak,float(f.max()));high_steps=high_steps+1 if f.max()>12 else 0
        if f.max()>3. and not relief_active:
            relief_active=True;relief_count+=1
            events.append(dict(time_s=data.time,kind='PROACTIVE_FORCE_RELEASE',forces_n=f.tolist()))
        if relief_active and f.max()<1.5:
            relief_active=False
            events.append(dict(time_s=data.time,kind='FORCE_RELEASE_COMPLETE',forces_n=f.tolist()))
        if relief_active:
            set_kinematics(h,scratch,last);rows=[];errors=[]
            for i in np.flatnonzero(a):
                e,J,detail=belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[i]),p[i],n[i],
                    orientation_weight_m=.04,dof_indices=planner.order)
                e[:3]=-n[i]*.00004;e[3:]=0.
                rows.append(J);errors.append(e)
            if rows:
                delta,rd=bounded_belly_step(np.vstack(rows),np.concatenate(errors),last,planner.lower,planner.upper,
                    gain=1.,maximum_joint_step=.001)
                if not rd['feasible']:raise RuntimeError('NO_BOUNDED_FORCE_RELEASE')
                command=last+delta;phase='FORCE_RELEASE'
        def command_reason(candidate):
            return (planner.safety.command_collision_reason(joint_state(h,data),candidate) or
                    planner.safety.command_collision_reason(last,candidate))
        reason=command_reason(command)
        if reason:
            rejections+=1
            # One scalar preserves the requested correction direction.
            accepted=False
            for fraction in (.5,.25,.125,0.):
                candidate=last+fraction*(command-last)
                if command_reason(candidate) is None:
                    command=candidate;accepted=True;break
            if not accepted:
                escape=planner.clearance_exit(joint_state(h,data),last)
                if escape is None or command_reason(escape) is not None:raise RuntimeError('NO_GEOMETRY_SAFE_COMMAND:'+str(reason))
                command=escape;phase='CLEARANCE_EXIT'
        else:rejections=0
        if np.any(command<planner.lower) or np.any(command>planner.upper) or np.max(abs(command-last))>.002400000001:
            raise RuntimeError('COMMAND_BOUND_VIOLATION')
        values=dict(time_s=data.time,q_rad=joint_state(h,data),qvel_rad_s=data.qvel.copy(),
            qacc_warmstart=data.qacc_warmstart.copy(),sensor_command_rad=last.copy(),command_rad=command.copy(),
            forces_n=f,contact_positions_world_m=p,normals_world=n,active=a,
            belly_forces_n=bf,belly_contact_positions_world_m=bp,belly_normals_world=bn,belly_active=ba,
            rejected_contact_force_n=np.asarray(bd['rejected_contact_force_n']),safe_belly=safe,
            phase=phase,leader_finger=finger,mcc_offset_m=offsets.copy(),coverage=coverage.area_fraction,
            visited_vertex_count=int(coverage.visited_vertices.sum()),goal_vertex=goal_vertex,
            unsafe_count=len(unsafe),arm_self_force_n=max([c['force_n'] for c in arm],default=0.),
            reference_geometry_backtrack=reason or '',actual_penetration_m=pen)
        values['released_idle_fingers']=np.array([i in released_idle_fingers for i in range(4)])
        for k,v in values.items():logs.setdefault(k,[]).append(v)
        states+=1
        if unsafe or arm:raise RuntimeError('UNSAFE_NATIVE_CONTACT')
        if high_steps>=10:raise RuntimeError('SUSTAINED_DANGEROUS_FORCE')
        last=command.copy();set_control(h,data,last);mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
        done_hold=done_hold+.002 if coverage.visited_vertices.all() and safe.any() else 0.
        if states%15000==0:flush()
        if perf_counter()-last_progress>12.:
            progress=dict(time_s=data.time,area=coverage.area_fraction,vertices=int(coverage.visited_vertices.sum()),
                goals=int(coverage.visited_goals.sum()),routes=routes,lifts=lifts,phase=phase,peak_n=peak,force_n=f.tolist())
            summary_path.write_text(json.dumps(progress,indent=2));print('PROGRESS',json.dumps(progress),flush=True);last_progress=perf_counter()
        return obs

    def bounded_move(command,phase):
        delta=command-last;factor=min(1.,.0024/max(np.max(abs(delta)),1e-15))
        return step(last+factor*delta,phase)

    def track(points,normals,qs,distances,*,clearance_start=0.,duration_limit=30.,goal=-1,phase_override=None,require_stable=False):
        nonlocal finger,tangent,normal,point,nominal_previous,last_handoff
        ctrl=controllers[finger]
        if clearance_start>0.:ctrl.reset()
        start=data.time;progress=0.;gap=0.;settled=0.;filter_normal=normal.copy()
        qs=qs.copy();qs[0]=nominal_previous.copy()
        while data.time<duration_s and data.time-start<duration_limit:
            obs=measure();f,p,n,a,unsafe,pen,bf,bp,bn,ba,bd,arm,safe=obs
            # Stop approach at the first qualified belly, including another
            # finger. Continuing to lower the wrist until the nominated finger
            # touches used to overload a finger that had touched first.
            candidates=np.flatnonzero(safe)
            if len(candidates) and data.time-last_handoff>.5:
                other=int(candidates[np.argmax(f[candidates])])
                early=clearance_start>0. and f[other]>.25
                takeover=other!=finger and (f[other]>max(1.5,f[finger]+.5) or
                    (not safe[finger] and gap>.25 and f[other]>.3))
                if early or takeover:
                    finger=other;normal=n[other].copy()
                    projected,_,_=trimesh.proximity.closest_point(g.mesh,(bp[other]-h.object_position_m)[None,:])
                    point=projected[0];set_kinematics(h,scratch,last)
                    _,_,detail=belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[finger]),point+h.object_position_m,normal)
                    offset=-float((np.asarray(detail['support_point_world_m'])-point-h.object_position_m)@normal)
                    controllers[finger].reset(offset_m=float(np.clip(offset,-.004,.004)))
                    offsets[:]=0.;offsets[finger]=float(np.clip(offset,-.004,.004));nominal_previous=last.copy();last_handoff=data.time
                    events.append(dict(time_s=data.time,kind='FIRST_BELLY_HANDOFF' if early else 'BELLY_SUPPORT_HANDOFF',finger=finger,force_n=float(f[finger])))
                    return True,'BELLY_HANDOFF'
            was_relief=relief_active
            gap=0. if safe[finger] else gap+.002
            u=min(progress,distances[-1]);k=min(max(0,int(np.searchsorted(distances,u)-1)),len(distances)-2)
            fraction=(u-distances[k])/max(distances[k+1]-distances[k],1e-9)
            point=(1-fraction)*points[k]+fraction*points[k+1]
            planned_normal=unit((1-fraction)*normals[k]+fraction*normals[k+1])
            nominal=(1-fraction)*qs[k]+fraction*qs[k+1]
            # Native contact normals actively correct the belly orientation.
            normal_target=n[finger] if a[finger] and n[finger]@planned_normal>.7 else planned_normal
            angle=float(np.arccos(np.clip(filter_normal@normal_target,-1,1)))
            blend=min(.06,.18*.002/max(angle,1e-12))
            filter_normal=unit((1-blend)*filter_normal+blend*normal_target);normal=filter_normal.copy()
            far_time=max(0.,clearance_start-.006)/.025
            elapsed=data.time-start
            clearance=max(0.,clearance_start-.025*elapsed) if elapsed<far_time else max(0.,min(clearance_start,.006)-.004*(elapsed-far_time))
            if clearance_start>0.:
                approach_fraction=float(np.clip(1.-clearance/clearance_start,0.,1.))
                nominal=(1.-approach_fraction)*qs[0]+approach_fraction*qs[-1]
            nominal_previous=nominal.copy()
            others=[i for i in range(4) if i!=finger and a[i] and bd['rejected_contact_force_n'][i]>=.2]
            released_idle_fingers.update(others)
            columns=np.concatenate([np.arange(7)]+[np.arange(7+4*i,11+4*i) for i in sorted({finger,*others})])
            weights=np.r_[np.full(7,.02 if clearance_start>0 else .012),
                np.full(len(columns)-7,.006 if clearance_start>0 else .0015)]
            prior_offset=float(offsets[finger])
            offset=ctrl.step(point,-normal,1.,float(f[finger])).offset_m if clearance<.0005 else 0.
            if offset>prior_offset+.004*.002:
                offset=prior_offset+.004*.002;ctrl.reset(offset_m=offset,velocity_m_s=.004)
            offsets[:]=0.;offsets[finger]=offset
            set_kinematics(h,scratch,last)
            error,J,diag=belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[finger]),point+h.object_position_m,
                normal,clearance_m=clearance-offset,orientation_weight_m=.04,dof_indices=planner.order[columns])
            for other in others:
                _,other_J,_=belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[other]),p[other],n[other],
                    orientation_weight_m=.04,dof_indices=planner.order[columns])
                outward=min(.00002,.00002*max(0.,f[other]-.1))
                J=np.vstack((J,n[other]@other_J[:3]));error=np.r_[error,-outward]
            J=np.vstack((J,np.diag(weights)));error=np.r_[error,weights*(last[columns]-nominal[columns])]
            guard_J,guard_error=planner.arm_buffer_terms(last,columns)
            if len(guard_error):J=np.vstack((J,guard_J));error=np.r_[error,guard_error]
            delta,dd=bounded_belly_step(J,error,last[columns],planner.lower[columns],planner.upper[columns],gain=.65,
                maximum_joint_step=np.r_[np.full(7,.0004),np.full(len(columns)-7,.0008)])
            if not dd['feasible']:raise RuntimeError('BOUNDED_NATURAL_STEP:'+dd['reason'])
            command=last.copy();command[columns]+=delta
            # Idle fingers follow a gentle reference; only active motion is
            # solved with contact objectives. No fixed old contact triplet.
            idle=np.setdiff1d(np.arange(23),columns)
            if released_idle_fingers:
                held=np.concatenate([np.arange(7+4*i,11+4*i) for i in released_idle_fingers])
                idle=np.setdiff1d(idle,held)
            command[idle]+=np.clip(nominal[idle]-last[idle],-.0008,.0008)
            step(command,phase_override or ('APPROACH' if clearance>0 else ('SLIDE' if gap<.1 else 'ACTIVE_RECONTACT')),obs,goal)
            if was_relief and not relief_active:
                set_kinematics(h,scratch,last)
                _,_,detail=belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[finger]),point+h.object_position_m,normal)
                offset=-float((np.asarray(detail['support_point_world_m'])-point-h.object_position_m)@normal)
                controllers[finger].reset(offset_m=float(np.clip(offset,-.004,.004)))
                offsets[finger]=float(np.clip(offset,-.004,.004));nominal_previous=last.copy()
                return False,'FORCE_RELIEVED_REPLAN'
            if safe[finger] and f.max()<2. and diag['position_residual_m']<.003 and rejections<3:
                dn=float(np.arccos(np.clip(normals[k]@normals[k+1],-1,1)))
                speed=min(.008,.15*(distances[k+1]-distances[k])/max(dn,1e-6))
                progress=min(distances[-1],progress+speed*.002)
            reached=route_endpoint_reached(progress,distances[-1],safe[finger],
                bp[finger]-h.object_position_m,points[-1],pivot=phase_override=='CONTACT_PIVOT' or require_stable)
            settled=settled+.002 if reached else 0.
            if reached and (not require_stable or settled>=.15):return True,'ROUTE_FINISHED'
            if gap>max(5.,clearance_start/.004+3.):return False,'RECONTACT_REPLAN'
            if rejections>250:return False,'GEOMETRY_REPLAN'
            if done_hold>=.15:return True,'FULL_MEASURED_BELLY_SURFACE'
        return False,'ROUTE_TIME_BUDGET'

    try:
        p=np.vstack((point,point));n=np.vstack((normal,normal));qn,_=planner.solve(last,point+h.object_position_m,normal,tangent,finger=finger,clearance=.0002,preserve_hand=True)
        ok,reason=track(p,n,np.vstack((qn,qn)),np.array([0.,1e-9]),clearance_start=.006 if resume is None else 0.,duration_limit=12.,goal=vertex)
        events.append(dict(time_s=data.time,kind='INITIAL_APPROACH',success=ok,reason=reason))
        while data.time<duration_s and routes<max_routes:
            if (out/'development_stop.json').exists():raise RuntimeError('DEVELOPMENT_STOP_REQUESTED')
            if done_hold>=.15:stop='FULL_MEASURED_BELLY_SURFACE';break
            # Refresh the measured support context at a real native tick.
            # This is also the finite recontact entry after force relief.
            step(last,'CONTACT_CONTEXT')
            # Holding inside a positive clearance buffer is legal but leaves
            # every useful route blocked. Actively leave that buffer before
            # spending time on the same geometry search again.
            buffer_reason=planner.safety.collision_reason(last)
            arm_gap=planner.minimum_arm_clearance(last)
            nonpad_gap=planner.minimum_nonpad_clearance(last)
            if buffer_reason in ('ARM_SELF_CLEARANCE','NON_TIP_CLEARANCE') or arm_gap<.002 or nonpad_gap<.0045:
                recovery_start=data.time;initial_reason=buffer_reason
                while data.time-recovery_start<2. and data.time<duration_s:
                    escape=planner.clearance_exit(joint_state(h,data),last,arm_target_m=.004,object_target_m=.006,preserve_wrist=True)
                    if escape is None:break
                    step(escape,'CLEARANCE_EXIT')
                    if (planner.safety.collision_reason(last) is None and planner.minimum_arm_clearance(last)>=.0038
                            and planner.minimum_nonpad_clearance(last)>=.0058):break
                nominal_previous=last.copy()
                events.append(dict(time_s=data.time,kind='ACTIVE_POSITIVE_BUFFER_EXIT',initial_reason=initial_reason,
                    final_reason=planner.safety.collision_reason(last),duration_s=data.time-recovery_start))
            obs=measure();safe=obs[-1]
            if not safe.any():
                # No live support: derive a planning anchor from the actual
                # pad location, rather than retaining a previous face normal.
                local_center=data.geom_xpos[h.tip_geom_ids[finger]]-h.object_position_m
                projected,_,_=trimesh.proximity.closest_point(g.mesh,local_center[None,:])
                point=projected[0];normal=planner.smooth_normal(point)
            current=int(coverage.last_vertices[finger]) if safe[finger] and coverage.last_vertices[finger]>=0 else g.nearest_vertex(point)[0]
            distance=dijkstra(g.adjacency,directed=False,indices=current)
            remaining=np.flatnonzero(~coverage.visited_vertices)
            goals=g.required_vertices[~coverage.visited_goals]
            candidates=goals if len(goals) else remaining
            if not len(candidates):
                track(np.vstack((point,point)),np.vstack((normal,normal)),np.vstack((nominal_previous,nominal_previous)),
                    np.array([0.,1e-9]),duration_limit=8.,phase_override='FINAL_SAFE_HOLD',require_stable=True)
                continue
            # Prefer a nearby unvisited frontier. The raster remains a weak
            # tie-breaker, rather than repeatedly returning to an old blocked
            # row at the opposite end of the bottle.
            rank=candidates[np.argsort(distance[candidates]+.04*failed[candidates]+.0003*raster_index[candidates])]
            planned=None;goal=None
            for candidate in rank[:12]:
                candidate=int(candidate)
                points,normals=planner.surface_route(current,candidate)
                if len(points)<2:
                    failed[candidate]+=1;continue
                # Preserve the previous surface reference and compliance
                # state across route boundaries; nearest mesh vertices are
                # routing anchors, not permission to jump the physical target.
                points[0]=point;normals[0]=normal
                set_joint_state(h,planner.data,last)
                axis=planner.data.geom_xmat[h.tip_geom_ids[finger]].reshape(3,3)[:,0].copy()
                planned,pdiag=planner.slide_plan(last,points,normals,finger,axis)
                if planned is not None:goal=candidate;break
                failed[candidate]+=1
            if planned is not None:
                routes+=1
                ok,reason=track(planned['points'],planned['normals'],planned['q'],planned['distance'],
                    duration_limit=max(8.,planned['distance'][-1]/.002+4.),goal=goal)
                if not ok or not coverage.visited_vertices[goal]:failed[goal]+=1
                events.append(dict(time_s=data.time,kind='SURFACE_ROUTE',goal_vertex=goal,length_m=float(planned['distance'][-1]),
                    success=ok,reason=reason,area=coverage.area_fraction,vertices=int(coverage.visited_vertices.sum())))
                if reason=='FULL_MEASURED_BELLY_SURFACE':stop=reason;break
                if reason=='FORCE_RELIEVED_REPLAN':
                    force_retries[goal]=force_retries.get(goal,0)+1
                    if force_retries[goal]<=2:continue
                else:force_retries[goal]=0
                # A blocked/repeated prefix needs a different posture, not a
                # fresh local trial or artificially inherited coverage.
                if ok and failed[goal]<3:continue
            pivoted=False
            if measure()[-1][finger]:
                set_kinematics(h,planner.data,last)
                axis=planner.data.geom_xmat[h.tip_geom_ids[finger]].reshape(3,3)[:,0].copy()
                for degrees in (15.,-15.,30.,-30.,60.,-60.):
                    key=(*np.round(point/.02).astype(int).tolist(),finger,degrees)
                    if key in pivots_tried:continue
                    pivots_tried.add(key);angle=np.radians(degrees)
                    proposed=Rotation.from_rotvec(angle*normal).apply(axis)
                    qturn,pd=planner.solve(last,point+h.object_position_m,normal,proposed,finger=finger,clearance=.0002,preserve_hand=True)
                    if pd['reason'] is not None or not planner.safety.edge_free(last,qturn):continue
                    parameter=abs(angle)*.008/.12
                    ok,why=track(np.vstack((point,point)),np.vstack((normal,normal)),np.vstack((nominal_previous,qturn)),
                        np.array([0.,parameter]),duration_limit=abs(angle)/.08+4.,phase_override='CONTACT_PIVOT')
                    events.append(dict(time_s=data.time,kind='CONTACT_PIVOT',requested_angle_deg=degrees,success=ok,reason=why))
                    pivoted=ok
                    if ok:break
            if pivoted:continue
            moved=False;transfer_options=[]
            # A difficult frontier may be reached by landing just beside it
            # with a better finger/posture, then sliding across. A covered
            # landing remains only a route waypoint: it grants no new area.
            anchor=int(rank[0]);near=g.required_vertices[
                np.argsort(np.linalg.norm(g.vertices_m[g.required_vertices]-g.vertices_m[anchor],axis=1))]
            count=int(coverage.visited_vertices.sum());gateways=[]
            for v in near:
                key=(count,anchor,int(v),finger)
                if len(gateways)>=5:break
                if np.linalg.norm(g.vertices_m[v]-g.vertices_m[anchor])>.04:break
                if coverage.visited_vertices[v] and key not in gateways_tried:gateways.append(int(v))
            transfer_rank=list(dict.fromkeys([anchor,*gateways,*rank[1:8].tolist()]))
            for candidate in transfer_rank:
                candidate=int(candidate);p=g.vertices_m[candidate];n=planner.smooth_normal(p)
                path,td=planner.transfer(last,p+h.object_position_m,n,normal,budget_s=8.)
                if path is None:failed[candidate]+=1;continue
                transfer_options.append((path.duration_s,candidate,path,td,p,n))
                if path.duration_s<12.:break
            if transfer_options:
                _,candidate,path,td,p,n=min(transfer_options,key=lambda item:item[0])
                if coverage.visited_vertices[candidate]:gateways_tried.add((count,anchor,candidate,finger))
                lifts+=1;begin=data.time
                events.append(dict(time_s=data.time,kind='NECESSARY_REORIENTATION',goal_vertex=candidate,plan=td,duration_s=path.duration_s))
                while data.time-begin<path.duration_s+.2 and data.time<duration_s:
                    bounded_move(path.sample(data.time-begin),'NECESSARY_REORIENTATION')
                arrival_start=data.time
                while np.max(abs(last-path.waypoints[-1]))>.003 and data.time-arrival_start<2. and data.time<duration_s:
                    bounded_move(path.waypoints[-1],'NECESSARY_REORIENTATION')
                nominal_previous=last.copy()
                finger=int(td['finger']);tangent=np.asarray(td['tangent_world'])
                point=p.copy();normal=n.copy();released_idle_fingers.clear()
                qn,dd=planner.solve(last,p+h.object_position_m,n,tangent,finger=finger,clearance=.0002,preserve_hand=True)
                if dd['reason'] is not None:
                    events.append(dict(time_s=data.time,kind='TRANSFER_APPROACH_REJECTED',goal_vertex=candidate,reason=dd['reason']))
                    failed[candidate]+=1;continue
                ok,reason=track(np.vstack((p,p)),np.vstack((n,n)),np.vstack((qn,qn)),np.array([0.,1e-9]),
                    clearance_start=td['clearance_m'],duration_limit=td['clearance_m']/.004+8.,goal=candidate)
                events.append(dict(time_s=data.time,kind='TRANSFER_APPROACH',goal_vertex=candidate,success=ok,reason=reason))
                failed[candidate]+=0 if ok else 1;moved=True
            if not moved:
                events.append(dict(time_s=data.time,kind='NO_TRANSFER_CANDIDATE',remaining_vertices=len(remaining)))
                # Try the next finite candidate block on the next planning
                # cycle; deferred regions remain in the original denominator.
                if np.min(failed[candidates])>=15:stop='NO_REACHABLE_REMAINING_ROUTE';break
        if stop=='UNFINISHED':stop='SIMULATION_BUDGET' if data.time>=duration_s else 'ROUTE_BUDGET'
    except RuntimeError as error:stop=str(error)
    except KeyboardInterrupt:stop='DEVELOPMENT_INTERRUPTED'
    finally:
        flush()
        np.savez_compressed(out/'final_state.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,
            qacc_warmstart=data.qacc_warmstart,time=data.time,visited_vertices=coverage.visited_vertices,
            visited_goals=coverage.visited_goals)
        result=dict(stop_reason=stop,executed_s=data.time,full_surface_complete=bool(done_hold>=.15),
            coverage_fraction=coverage.area_fraction,visited_vertices=int(coverage.visited_vertices.sum()),
            total_vertices=len(g.vertices_m),visited_goals=int(coverage.visited_goals.sum()),total_goals=g.required_goal_count,
            native_states=states,routes=routes,necessary_lifts=lifts,peak_force_n=peak,coverage_inherited=resume is not None,
            fresh_single_version=resume is None,resume=resume_info,
            force_releases=relief_count,
            trace_chunks=[dict(c,sha256=sha256((out/c['file']).read_bytes()).hexdigest()) for c in chunks])
        (out/'summary.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
    return result


def geometry_survey(output,limit=40):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    h,g,q,quality=build_scene(out)
    planner=NaturalBellyPlanner(h,g)
    records=[];qs=[];vertices=[]
    chosen=np.unique(np.linspace(0,len(g.required_vertices)-1,min(limit,len(g.required_vertices))).astype(int))
    for k,goal in enumerate(chosen):
        v=int(g.required_vertices[goal]);p=g.vertices_m[v]+h.object_position_m;n=planner.smooth_normal(g.vertices_m[v])
        solved,diag=planner.find(q,p,n,clearance=.003,budget_s=4.)
        records.append(dict(goal=int(goal),vertex=v,position_local_m=g.vertices_m[v].tolist(),normal=n.tolist(),**diag))
        if solved is not None:qs.append(solved);vertices.append(v);q=solved
        print('SURVEY',k+1,'/',len(chosen),'success',len(qs),'goal',int(goal),'reason',diag['reason'],flush=True)
    np.savez_compressed(out/'poses.npz',q=np.array(qs),vertices=np.array(vertices))
    (out/'survey.json').write_text(json.dumps(dict(geometry_only=True,coverage_granted=False,quality=quality,records=records),indent=2))
    (out/'model.xml').write_text(h.xml,encoding='utf-8')
    return records


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--survey',type=int)
    ap.add_argument('--seconds',type=float,default=1800.)
    ap.add_argument('--resume')
    args=ap.parse_args()
    if args.survey is not None:geometry_survey(args.output,args.survey)
    else:run(args.output,args.seconds,resume=args.resume)
