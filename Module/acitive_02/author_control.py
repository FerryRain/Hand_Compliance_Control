"""Run the public Khadivar et al. no-arm control law on a matched LEAP fixture."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from . import active_a, aude, route_free
from .experiment import digest, read_json, write_json
from .route_free_inputs import OBJECTS, load_trial
from .route_free_score import score


SOURCE = 'd89a6941fec699768e64fb6acc07439ab822e59f'
PERIOD = 90
TRACE_FIELDS = ('qpos','wrist_mocap','goal_wrist','goal_joints_rad','finger_torque_nm',
                'points_m','normals_outward','pad_force_n','region_load_n','pad_valid',
                'posture_legal','mode','wall_s')


class PlannerContact(NamedTuple):
    measured_heads: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    loads: np.ndarray


def stable_trf(fun,x0,*,bounds,**options):
    # Express the first translation from its lower bound. The physical initial
    # state and bounds match the archived, numerically stabilized MCC rollout.
    offset = np.zeros_like(x0)
    offset[0] = -bounds[0][0]
    fit = least_squares(lambda x:fun(x-offset),x0+offset,
                        bounds=(bounds[0]+offset,bounds[1]+offset),**options)
    fit.x -= offset
    return fit


def wrist_step(position,rotation,goal,dt):
    displacement=goal.position-position
    position=position+dt*.27*displacement/(np.linalg.norm(displacement)+.002)
    turn=(goal.orientation*rotation.inv()).as_rotvec()
    # The author's quaternion log is half the angle-axis vector.
    increment=(dt*4.5/(np.linalg.norm(turn)+np.pi/180))*turn
    return position,Rotation.from_rotvec(increment)*rotation


def finger_torque(jacobians,velocity,error,normals,in_contact,force_magnitude,rest_term,bias):
    slip=np.einsum('ij,ij->i',error,normals)
    tangent_error=error-slip[:,None]*normals*in_contact[:,None]
    normal_force=np.where(in_contact,np.where(force_magnitude>.5,-.5,.2),.5)
    tip_force=10.*tangent_error-.01*velocity-normal_force[:,None]*normals
    jt=jacobians.transpose(0,2,1)
    gram=jacobians@jt+1.e-10*np.eye(3)
    proj=np.linalg.solve(gram,(jacobians@rest_term)[...,None])[...,0]
    # Sum each digit's 16-DoF null-space rest term, as in the authors' loop.
    return bias+np.einsum('fji,fj->i',jacobians,tip_force-proj)+4.*rest_term


class AuthorExecutor:
    def __init__(self,plant,robot,data,grasp):
        self.plant,self.robot,self.data=plant,robot,data
        model=plant.model
        self.dt=model.opt.timestep
        if not np.isclose(self.dt,.002) or model.nu!=16:
            raise ValueError('The port expects 16 scalar motors and a 2 ms physics tick')
        np.testing.assert_array_equal(model.jnt_dofadr[model.actuator_trnid[:,0]],plant.dof)
        np.testing.assert_allclose(model.actuator_gear[:,0],1.)
        former_torque=data.actuator_force.copy()
        # Change the XML position servos into direct joint torques in memory.
        # Joint/body inertia, physical stops, contact geometry and forces remain those of LEAP.
        model.actuator_gainprm[:]=0.
        model.actuator_gainprm[:,0]=1.
        model.actuator_biasprm[:]=0.
        model.actuator_ctrllimited[:]=0
        data.ctrl[:]=former_torque
        self.position=data.mocap_pos[plant.driver].copy()
        self.rotation=active_a.rotation(data.mocap_quat[plant.driver])
        self.origin=data.qpos[:3].copy()
        self.normals=np.zeros((4,3))
        self.fk=mujoco.MjData(robot.model)
        self.full_jac=np.empty((4,3,model.nv))
        self.wrench=np.empty(6)
        # LEAP long fingers order MCP flexion before abduction.
        self.rest=grasp.bound(np.array([.3,0.,.3,.3]*3+[.5,.4,.4,.4]))
        self.grasp=grasp

    def set_goal(self,posture):
        delta=posture.position-self.origin
        position=self.origin+route_free.limit(delta,.45)
        self.goal=active_a.Posture(position,posture.orientation,self.grasp.bound(posture.joints))
        self.fk.qpos[:7]=[0.,0.,0.,1.,0.,0.,0.]
        self.fk.qpos[self.robot.q]=self.goal.joints
        mujoco.mj_kinematics(self.robot.model,self.fk)
        self.local_tips=self.fk.geom_xpos[self.robot.pads].copy()
        self.target_wrist=np.r_[self.goal.position,active_a.quaternion(self.goal.orientation)]

    def tactile(self):
        count=np.zeros(4,dtype=int)
        position=np.zeros((4,3))
        force=np.zeros((4,3))
        for i,c in enumerate(self.data.contact):
            a,b=int(c.geom1),int(c.geom2)
            if a in self.plant.objects:
                finger=self.plant.pad_index.get(b)
                sign=1.
            elif b in self.plant.objects:
                finger=self.plant.pad_index.get(a)
                sign=-1.
            else:
                continue
            if finger is None or c.efc_address<0:
                continue
            mujoco.mj_contactForce(self.plant.model,self.data,i,self.wrench)
            count[finger]+=1
            position[finger]+=c.pos
            force[finger]+=sign*(c.frame.reshape(3,3).T@self.wrench[:3])
        divisor=np.maximum(count,1)[:,None]
        return count>0,position/divisor,np.linalg.norm(force/divisor,axis=1)

    def physical_observation(self, scored_touch):
        # The authors' no-arm feedback packet uses the solver's current tip
        # contact flag/mean position and the radial site-to-contact normal.
        active,points,_=self.tactile()
        tips=self.data.geom_xpos[self.plant.pads]
        normals=np.zeros_like(points)
        v=tips[active]-points[active]
        normals[active]=v/(np.linalg.norm(v,axis=1,keepdims=True)+1.e-10)
        return PlannerContact(active,points,normals,scored_touch.loads)

    def step(self):
        m,d=self.plant.model,self.data
        self.position,self.rotation=wrist_step(self.position,self.rotation,self.goal,self.dt)
        d.mocap_pos[self.plant.driver]=self.position
        d.mocap_quat[self.plant.driver]=active_a.quaternion(self.rotation)
        mujoco.mj_forward(m,d)
        tips=d.geom_xpos[self.plant.pads]
        live,points,forces=self.tactile()
        delta=tips[live]-points[live]
        self.normals[live]=delta/(np.linalg.norm(delta,axis=1,keepdims=True)+1.e-10)
        for finger,pad in enumerate(self.plant.pads):
            mujoco.mj_jacGeom(m,d,self.full_jac[finger],None,pad)
        target=self.position+self.rotation.apply(self.local_tips)
        velocity=self.full_jac@d.qvel
        q,dq=d.qpos[self.plant.q],d.qvel[self.plant.dof]
        rest_term=.004*(self.rest-q)-.000008*dq
        d.ctrl[:]=finger_torque(self.full_jac[:,:,self.plant.dof],velocity,target-tips,
                               self.normals,live,forces,rest_term,d.qfrc_bias[self.plant.dof])
        mujoco.mj_step(m,d)

    def sample(self,touch,mode,clock):
        d,p=self.data,self.plant
        return (d.qpos.copy(),np.r_[d.mocap_pos[p.driver],d.mocap_quat[p.driver]],
                self.target_wrist,self.goal.joints,d.actuator_force.copy(),touch.points.copy(),
                touch.normals.copy(),touch.forces.copy(),touch.loads.copy(),touch.measured_heads,
                touch.legal_posture,mode,clock)


def run(obj,fixtures,out,seconds,planner_feedback='registered'):
    out=out.resolve()
    paired,initial,case,mesh=load_trial(fixtures,obj)
    plant,data,robot,sensed,touch,state,_=initial
    if robot.objects or robot.surface_queries or plant.surface_queries:
        raise ValueError('The planner must only receive the robot model and observed contacts')
    if sha256(state.tobytes()).hexdigest()!=paired['integration_state']:
        raise ValueError('Initial physics state differs from the paired fixture')
    signature=dict(object=obj,seed=paired['seed'],initial=paired['integration_state'],source=SOURCE,
           code={p.name:digest(p) for p in map(Path,[__file__,aude.__file__,route_free.__file__])},
           hand='LEAP',controller='published no-arm wrist DS and fingertip torque law on LEAP',
           wrist_mcc=False,finger_mcc=False,planner_variables=22,planning_object_geometries=0,
           period=PERIOD,seconds=seconds,servo_hz=500,contact_log_hz=100,planner_feedback=planner_feedback)
    if out.exists():
        archived=read_json(out/'run.json')
        if archived['signature']!=signature or any(digest(out/k)!=v for k,v in archived['hashes'].items()):
            raise ValueError(f'Results at {out} belong to another configuration')
        return
    current=aude.state_posture(robot,sensed)
    grasp=active_a.GraspEnvelope(current.joints,robot.limits)
    ik=route_free.GraspIK(robot,grasp,solver=stable_trf)
    explorer=aude.AudeExplorer(robot,ik,paired['seed'],aude.DEFAULT_STEP_M)
    explorer.directions=route_free.TimedDirections(paired['seed'])
    execute=AuthorExecutor(plant,robot,data,grasp)
    execute.set_goal(current)
    rows=[execute.sample(touch,'initial',0.)]
    observe=execute.physical_observation if planner_feedback=='physical' else (lambda sample:sample)
    observed=observe(touch)
    planner_pad_valid=[observed.measured_heads.copy()]
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage=Path(tmp)/'run'
        stage.mkdir()
        with (stage/'plans.jsonl').open('w',encoding='utf-8') as stream:
            start=perf_counter()
            for tick in range(seconds*100):
                explorer.observe(sensed,observed)
                if tick%PERIOD==0:
                    t0=perf_counter()
                    ik.center=route_free.touch_frame(observed,ik.center)
                    ik.last_fit_ms=[]
                    explorer.directions.elapsed_ms=0.
                    plan=explorer.plan(sensed,observed)
                    execute.set_goal(plan.posture)
                    record=dict(tick=tick,mode=plan.mode,live=plan.valid.tolist(),targets_m=plan.targets.tolist(),
                                directions=plan.direction.tolist(),wrist=execute.target_wrist.tolist(),
                                joints=execute.goal.joints.tolist(),target_error_mm=plan.error_mm,
                                fit_calls=len(ik.last_fit_ms),ik_ms=sum(ik.last_fit_ms),
                                gp_direction_ms=explorer.directions.elapsed_ms,plan_ms=(perf_counter()-t0)*1000)
                    stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
                for _ in range(active_a.CONTROL_STEPS):
                    execute.step()
                mujoco.mj_forward(plant.model,data)
                touch=plant.measure(data)
                active_a.robot_state(plant,robot,data,sensed)
                observed=observe(touch)
                planner_pad_valid.append(observed.measured_heads.copy())
                rows.append(execute.sample(touch,plan.mode,perf_counter()-start))
                if (tick+1)%6000==0:
                    print(obj,(tick+1)*.01,'s',len(explorer.gp.surface),'surface cells',flush=True)
        trace={k:np.asarray(v) for k,v in zip(TRACE_FIELDS,zip(*rows),strict=True)}
        trace['planner_pad_valid']=np.asarray(planner_pad_valid)
        if abs(data.time-seconds)>1.e-6 or not np.isfinite(trace['qpos']).all() or not np.isfinite(trace['finger_torque_nm']).all():
            raise ValueError('Invalid trajectory')
        np.testing.assert_allclose(grasp.bound(trace['goal_joints_rad']),trace['goal_joints_rad'],rtol=0,atol=1e-10)
        np.savez_compressed(stage/'trace.npz',**trace)
        hashes={p:digest(stage/p) for p in ('trace.npz','plans.jsonl')}
        write_json(stage/'run.json',dict(signature=signature,hashes=hashes,trace_sha256=hashes['trace.npz'],
            initial_sha256=paired['integration_state'],seed=paired['seed'],model_sha256=digest(case/'model.mjb'),
            offline_mesh_sha256=digest(mesh),steps=100*seconds,wall_s=float(trace['wall_s'][-1]),
            planning_object_geometries=0,rest_rad=execute.rest.tolist(),actuation='direct torque on original LEAP hinges'))
        stage.rename(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',choices=OBJECTS,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=240)
    parser.add_argument('--planner-feedback',choices=('registered','physical'),default='registered')
    opts=parser.parse_args()
    if opts.seconds<=0:
        parser.error('Duration must be positive')
    torch.set_num_threads(1)
    run(opts.object,opts.fixtures,opts.out,opts.seconds,opts.planner_feedback)
    print(score(opts.out,opts.fixtures/opts.object/'surface.obj'),flush=True)
