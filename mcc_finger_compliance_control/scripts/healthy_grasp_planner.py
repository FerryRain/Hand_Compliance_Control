"""Nonpenetrating reference planning with explicit local motion reserve.

Planning uses a small positive surface gap. Compliance penetration belongs
to force-controlled execution and is never an optimization target here.
This isolated planner does not modify the running frozen collector audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from itertools import combinations,product
import warnings

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog, minimize, OptimizeWarning
from scipy.special import logsumexp

from object_catalog import load_object_config, MeshNormalOracle
from surface_mcc_finger import FullHandMCCFingerController, FullHandMCCFingerConfig, HAND_JOINT_NAMES, GeometrySurfaceOracle
from true_tip_waypoint_optimizer import SplitHullGeometry

DATA=Path(__file__).resolve().parents[1]/'data'
DEFAULT_BANK=DATA/'dp0_true_tip_1500/long_chains/training_only_healthy_seed_bank.npz'


@dataclass(frozen=True)
class HealthConfig:
    target_gap_m: float = .00015
    min_gap_m: float = .00005
    max_gap_m: float = .00080
    non_tip_clearance_m: float = .002
    preferred_non_tip_clearance_m: float = .005
    max_cross_finger_penetration_m: float = .000001
    physical_joint_margin_rad: float = .12
    physical_joint_margin_fraction: float = .08
    distal_bank_upper_quantile: float = .995
    distal_bank_extra_rad: float = .12
    healthy_bound_margin_rad: float = .03
    health_floor_fraction: float = .65
    local_joint_budget_rad: float = .10
    min_normal_retreat_m: float = .001
    min_tangent_motion_m: float = .0005
    least_squares_evaluations: int = 40
    constrained_iterations: int = 25


class StableSplitHullGeometry(SplitHullGeometry):
    """Recheck identical relative geometry at independent world origins.

    Native mesh CCD may leave a zero/empty witness for a near pair, after
    which the legacy query reports the next, more distant object hull.
    Rigidly translating both bodies changes no physical input. Use the
    smallest signed clearance, retaining the corresponding witnesses.
    Never jitter q or average away a detected penetration.
    """
    origins=np.array([[0.,0.,0.],[.1,.1,.1],[-.1,-.1,-.1]])

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        free=np.flatnonzero(self.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        if len(free)!=1:raise ValueError('Expected one free palm in geometry model')
        self.free_address=int(self.model.jnt_qposadr[free[0]])
        self.origin=np.zeros(3)

    def position(self,q,palm_pose):
        rotation=super().position(q,palm_pose)
        if self.origin.any():
            self.data.qpos[self.free_address:self.free_address+3]+=self.origin
            self.data.mocap_pos[self.mocap]+=self.origin
            mujoco.mj_kinematics(self.model,self.data)
        return rotation

    def evaluate(self,q,palm_pose,self_collision=False,non_tip_clearance=False):
        values=[]
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(np.asarray(palm_pose)[[4,5,6,3]]).as_matrix()
        try:
            for origin in self.origins:
                self.origin=origin
                value=super().evaluate(q,palm_pose,self_collision,non_tip_clearance)
                value['object_points']-=rotation @ origin
                value['hand_points_palm']-=origin
                values.append(value)
        finally:
            self.origin=np.zeros(3)
            self.position(q,palm_pose)  # Jacobians use the original palm frame.
        gaps=np.stack([v['distance'] for v in values])
        selected=np.argmin(gaps,axis=0)
        result={}
        for key in ('distance','object_points','top_fraction','pad_surface_normals','cut_cap','hand_points_palm'):
            result[key]=np.stack([v[key] for v in values])[selected,np.arange(4)]
        result['connector_distance']=np.min([v['connector_distance'] for v in values],axis=0)
        result['self_penetration']=max(v['self_penetration'] for v in values)
        result['distance_origin_spread_m']=np.ptp(gaps,axis=0)
        if non_tip_clearance:
            result['non_tip_distance']=np.min([v['non_tip_distance'] for v in values],axis=0)
        return result


def directional_capacity(jacobian, directions, lower_step, upper_step):
    """Local linear max travel for each direction, including redundancy.

    This is a Jacobian/healthy-bound capacity certificate, not a guarantee
    of nonlinear collision-free or force-safe execution.
    """
    jacobian=np.asarray(jacobian,float)
    values=[]
    pairs=np.asarray(list(combinations(range(4),2)))
    pairs=np.repeat(pairs,4,axis=0)
    sides=np.tile(np.asarray(list(product((0,1),repeat=2))),(6,1))
    limits=np.stack((lower_step,upper_step))
    for direction in np.asarray(directions,float):
        eq=np.column_stack((jacobian,-direction))
        # Five variables minus three equalities leaves two degrees of
        # freedom. Enumerate pairs of active joint bounds in one tiny batch.
        a=np.zeros((24,5,5));a[:,:3]=eq
        a[np.arange(24),3,pairs[:,0]]=1.;a[np.arange(24),4,pairs[:,1]]=1.
        b=np.zeros((24,5));b[:,3]=limits[sides[:,0],pairs[:,0]];b[:,4]=limits[sides[:,1],pairs[:,1]]
        invertible=abs(np.linalg.det(a))>1e-12
        if invertible.any():
            candidates=np.linalg.solve(a[invertible],b[invertible,:,None])[...,0]
            feasible=((candidates[:,:4]>=np.asarray(lower_step)-1e-9).all(-1)
                &(candidates[:,:4]<=np.asarray(upper_step)+1e-9).all(-1)&(candidates[:,4]>=-1e-10))
            values.append(max(0.,float(candidates[feasible,4].max())) if feasible.any() else 0.)
        else:
            # Rank-deficient source states are still audited honestly.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',OptimizeWarning)
                result=linprog([0.,0.,0.,0.,-1.],A_eq=eq,b_eq=np.zeros(3),
                    bounds=list(zip(lower_step,upper_step))+[(0.,None)],
                    method='highs',options={'threads':1})
            values.append(max(0.,float(result.x[-1])) if result.success else 0.)
    return np.asarray(values)


class HealthyGraspPlanner:
    def __init__(self, object_id='ycb_mustard', scale=1., bank=DEFAULT_BANK, config=None):
        self.config=config or HealthConfig()
        self.base=FullHandMCCFingerController(FullHandMCCFingerConfig())
        self.maps=self.base._kinematic_health_maps()
        self.geometry=StableSplitHullGeometry(load_object_config(object_id),scale)
        object_config=load_object_config(object_id)
        self.oracle=GeometrySurfaceOracle(object_config,scale=scale,
            mesh_normal_oracle=MeshNormalOracle.from_config(object_config,scale=scale))
        self.dofs=np.asarray([self.geometry.model.jnt_dofadr[mujoco.mj_name2id(
            self.geometry.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in HAND_JOINT_NAMES])
        with np.load(bank) as z:
            self.bank_q=np.asarray(z['q_hand'],float)
            self.bank_pose=np.asarray(z['palm_pose_object'],float)
        self.calibrate()

    def calibrate(self):
        c=self.config
        physical_lower=np.asarray(self.base.lower);physical_upper=np.asarray(self.base.upper)
        margin=np.maximum(c.physical_joint_margin_rad,
                          c.physical_joint_margin_fraction*(physical_upper-physical_lower))
        self.lower=physical_lower+margin;self.upper=physical_upper-margin
        # The two distal flexion joints keep room above the healthy teacher
        # envelope; proximal/opposition joints retain their own physical span.
        distal=np.array([2,3,6,7,10,11,14,15])
        self.upper[distal]=np.minimum(self.upper[distal],
            np.quantile(self.bank_q[:,distal],c.distal_bank_upper_quantile,axis=0)+c.distal_bank_extra_rad)
        self.lower[distal]=np.maximum(self.lower[distal],
            np.quantile(self.bank_q[:,distal],.005,axis=0)-.10)
        self.lower[[3,7,11,15]]=np.maximum(self.lower[[3,7,11,15]],0.)
        self.lower+=c.healthy_bound_margin_rad;self.upper-=c.healthy_bound_margin_rad
        if (self.lower>=self.upper).any():raise ValueError('Empty healthy joint bounds')
        rows=[]
        for q,pose in zip(self.bank_q[::3],self.bank_pose[::3]):
            tip=self.base.tip_positions_palm(q)
            eta_mu=np.array([self.base._finger_kinematic_health(f,tip[f],self.maps[f]) for f in range(4)])
            # Calibrate and evaluate sigma at the same split-pad witness
            # point. A fixed tip site has a different Jacobian and scale.
            geometry=self.geometry.evaluate(q,pose)
            jacobians=self.witness_jacobians(geometry)
            sigma=np.linalg.svd(jacobians,compute_uv=False)[:,-1]
            rows.append(np.array([eta_mu[:,0],eta_mu[:,1],sigma]))
        rows=np.asarray(rows)
        self.health_target=np.maximum(np.median(rows,axis=0),1e-8)
        self.health_floor=np.maximum(np.quantile(rows,.05,axis=0)*c.health_floor_fraction,
                                     np.array([.015,.15,.002])[:,None])
        self.calibration=dict(config=asdict(c),healthy_lower_rad=self.lower.tolist(),
            healthy_upper_rad=self.upper.tolist(),health_order=['eta','mu','sigma_min_m_per_rad'],
            health_floor=self.health_floor.tolist(),health_target=self.health_target.tolist(),
            calibration_bank_states=len(self.bank_q),calibration_stride=3,
            sigma_reference='split true-tip hull witness point, same as candidate evaluation',
            distance_query='Conservative minimum over three rigid world-origin representations; identical q and relative pose',
            scope='Mustard training-bank envelope; local differential capacity, not global optimality')

    def witness_jacobians(self,geometry):
        mujoco.mj_comPos(self.geometry.model,self.geometry.data)
        result=[]
        for f in range(4):
            jac=np.zeros((3,self.geometry.model.nv))
            body=int(self.geometry.model.geom_bodyid[self.geometry.pads[f]])
            mujoco.mj_jac(self.geometry.model,self.geometry.data,jac,None,
                geometry['hand_points_palm'][f],body)
            result.append(jac[:,self.dofs[4*f:4*f+4]])
        return np.asarray(result)

    def state(self,q,pose,capacity=False):
        q=np.asarray(q,float)
        g=self.geometry.evaluate(q,pose,self_collision=True,non_tip_clearance=True)
        tip=self.base.tip_positions_palm(q)
        eta_mu=np.array([self.base._finger_kinematic_health(f,tip[f],self.maps[f]) for f in range(4)])
        jacobians=self.witness_jacobians(g)
        sigma=np.linalg.svd(jacobians,compute_uv=False)[:,-1]
        health=np.array([eta_mu[:,0],eta_mu[:,1],sigma])
        result=dict(geometry=g,health=health,
            joint_margin_rad=np.minimum(q-self.lower,self.upper-q),jacobians=np.asarray(jacobians))
        if capacity:
            normal=self.oracle.normals_at_world(g['object_points'])
            from scipy.spatial.transform import Rotation
            r=Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()
            normal_palm=(r.T @ normal.T).T
            capacities=[]
            for f in range(4):
                n=normal_palm[f];n=n/max(np.linalg.norm(n),1e-12)
                axis=np.eye(3)[np.argmin(abs(n))]
                t1=np.cross(n,axis);t1/=np.linalg.norm(t1);t2=np.cross(n,t1)
                block=slice(4*f,4*f+4)
                lo=np.maximum(self.lower[block]-q[block],-self.config.local_joint_budget_rad)
                hi=np.minimum(self.upper[block]-q[block],self.config.local_joint_budget_rad)
                capacities.append(directional_capacity(jacobians[f],[n,t1,-t1,t2,-t2],lo,hi)
                    if (lo<=0).all() and (hi>=0).all() else np.zeros(5))
            result.update(capacity_m=np.asarray(capacities),normal_object=normal)
        return result

    def constraints(self,s):
        c=self.config;g=s['geometry']
        return np.r_[(g['distance']-c.min_gap_m)/.001,
            (c.max_gap_m-g['distance'])/.001,
            (g['non_tip_distance']-c.non_tip_clearance_m)/.002,
            (c.max_cross_finger_penetration_m-g['self_penetration'])/.001,
            (s['health'][0]-self.health_floor[0])/.03,
            (s['health'][1]-self.health_floor[1])/.2,
            (s['health'][2]-self.health_floor[2])/.005,
            s['joint_margin_rad']/.1]

    def valid(self,s,include_capacity=False):
        good=bool(np.isfinite(self.constraints(s)).all() and self.constraints(s).min()>=-1e-4
                  and not s['geometry']['cut_cap'].any())
        if include_capacity:
            good=good and bool((s['capacity_m'][:,0]>=self.config.min_normal_retreat_m).all()
                and (s['capacity_m'][:,1:]>=self.config.min_tangent_motion_m).all())
        return good

    def quality(self,s,q,previous):
        # Smooth minimum gives weak fingers priority rather than allowing
        # high average manipulability to hide a nearly immobilized finger.
        scores=(s['health']/self.health_target).ravel()
        worst=-float(logsumexp(-8*scores))/8
        fraction=(q-self.lower)/(self.upper-self.lower)
        margin=np.maximum(fraction*(1-fraction),1e-6)
        c=self.config;g=s['geometry']
        return (-worst+.008*np.mean(-np.log(4*margin))
            +.008*np.mean(((q-previous)/.3)**2)
            +.01*np.mean(((g['distance']-c.target_gap_m)/.001)**2)
            +.02*np.mean((np.maximum(c.preferred_non_tip_clearance_m-g['non_tip_distance'],0)/.003)**2))

    def nearest_seeds(self,pose,count=3):
        angular=2*np.arccos(np.clip(abs(self.bank_pose[:,3:] @ np.asarray(pose)[3:]),0,1))
        distance=np.linalg.norm(self.bank_pose[:,:3]-np.asarray(pose)[:3],axis=-1)/.03+angular
        return self.bank_q[np.argsort(distance)[:count]]

    def solve(self,pose,source_q,previous=None,multistart=3):
        c=self.config
        previous=np.asarray(source_q if previous is None else previous,float)
        seeds=[previous,np.asarray(source_q,float),*self.nearest_seeds(pose)]
        seeds=[np.clip(q,self.lower+1e-6,self.upper-1e-6) for q in seeds]
        cache={}
        def state(q):
            key=np.asarray(q,float).tobytes()
            if key not in cache:
                if len(cache)>1000:cache.clear()
                cache[key]=self.state(q,pose)
            return cache[key]
        def residual(q):
            s=state(q);g=s['geometry']
            return np.r_[(g['distance']-c.target_gap_m)/.001,
                3*np.maximum(c.preferred_non_tip_clearance_m-g['non_tip_distance'],0)/.002,
                4*max(g['self_penetration']-c.max_cross_finger_penetration_m,0)/.001,
                np.minimum(self.constraints(s),0)*2.,
                .06*(q-previous)/.3]
        seeds.sort(key=lambda q:float(np.sum(residual(q)**2)))
        candidates=[]
        for seed in seeds[:multistart]:
            fitted=least_squares(residual,seed,bounds=(self.lower,self.upper),
                max_nfev=c.least_squares_evaluations,diff_step=1e-3,
                ftol=1e-5,xtol=1e-5,gtol=1e-5)
            candidates.append(fitted.x)
            refined=minimize(lambda q:self.quality(state(q),q,previous),fitted.x,
                method='SLSQP',bounds=list(zip(self.lower,self.upper)),
                constraints={'type':'ineq','fun':lambda q:self.constraints(state(q))},
                options={'maxiter':c.constrained_iterations,'ftol':1e-5,'eps':1e-4})
            candidates.append(refined.x)
            feasible=sorted([q for q in candidates if self.valid(state(q))],
                            key=lambda q:self.quality(state(q),q,previous))
            for value in feasible[:2]:
                value=np.asarray(value,np.float32).astype(float)
                checked=self.state(value,pose,capacity=True)
                if self.valid(checked,include_capacity=True):
                    checked.update(valid=True,geometric_health_valid=True,
                        objective=self.quality(checked,value,previous))
                    return value,checked
        def rank(q):
            s=state(q);violation=np.minimum(self.constraints(s),0)
            return (not self.valid(s),float(violation@violation),self.quality(s,q,previous))
        # Rank feasible candidates by motion health, not sub-micron solver
        # residual differences. Never mark an infeasible result accepted.
        feasible=[q for q in candidates if self.valid(state(q))]
        q=min(feasible,key=lambda q:self.quality(state(q),q,previous)) if feasible else min(candidates,key=rank)
        q=np.asarray(q,np.float32).astype(float)
        final=self.state(q,pose,capacity=True)
        final['valid']=self.valid(final,include_capacity=True)
        final['geometric_health_valid']=self.valid(final)
        final['objective']=self.quality(final,q,previous)
        return q,final


def state_report(s):
    g=s['geometry']
    row=dict(pad_gap_m=g['distance'].tolist(),non_tip_min_m=float(g['non_tip_distance'].min()),
        cross_finger_penetration_m=float(g['self_penetration']),cut_cap=g['cut_cap'].tolist(),
        health_order=['eta','mu','sigma_min_m_per_rad'],health=s['health'].tolist(),
        min_healthy_joint_margin_rad=float(s['joint_margin_rad'].min()))
    if 'capacity_m' in s:
        row.update(capacity_direction_order=['outward','t1','-t1','t2','-t2'],
            local_capacity_m=s['capacity_m'].tolist())
    for name in ('valid','geometric_health_valid','objective'):
        if name in s:row[name]=s[name]
    return row
