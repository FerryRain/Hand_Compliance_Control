"""Run a tactile-direction baseline with the Active-02 grasp envelope and wrist MCC."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import active_a as execution
from . import aude
from . import route_free_inputs as inputs
from .experiment import digest, read_json, write_json
from .surface_scan import limit


class GraspIK(aude.JointIK):
    def __init__(self, hand, grasp, *, solver=None):
        super().__init__(hand, limits=grasp.limits, solver=solver)
        self.center = None
        self.last_fit_ms = []

    def fit(self, start, offsets, targets):
        if self.center is None:
            raise ValueError('The wrist approach reference requires an observed contact')
        began = perf_counter()
        frame = self.center.copy()
        local_start = execution.Posture(start.position-frame,start.orientation,start.joints)
        goal, error, gap = super().fit(local_start,offsets,targets-frame)
        self.last_fit_ms.append((perf_counter()-began)*1000)
        return execution.Posture(goal.position+frame,goal.orientation,goal.joints),error,gap


def touch_frame(touch, previous):
    valid = touch.measured_heads
    if not valid.any() or touch.loads.max()>=8:
        return previous
    center = touch.points[valid].mean(0)
    return center if previous is None else .70*previous+.30*center


class TimedDirections(aude.ExplorationDirections):
    def directions(self, *args):
        began = perf_counter()
        result = super().directions(*args)
        self.elapsed_ms = (perf_counter() - began) * 1000
        return result


def bounded_wrist(previous, position, orientation, origin):
    # Apply the matched command-rate and workspace limits before wrist compliance.
    legal = origin + limit(position - origin, .45)
    translation = previous[:3] + limit(legal - previous[:3], .002)
    old = execution.rotation(previous[3:])
    delta = (execution.rotation(orientation) * old.inv()).as_rotvec()
    return np.r_[translation, execution.quaternion(Rotation.from_rotvec(limit(delta, .0125)) * old)]


def run(obj, arm, folder, seconds, period, fixtures):
    folder = folder.resolve()
    started = perf_counter()
    paired,initial,case,mesh = inputs.load_trial(fixtures,obj)
    plant,data,robot,sensed,touch,state,warm = initial
    sources = [Path(__file__), Path(aude.__file__), Path(inputs.__file__), Path(execution.__file__),
               Path(execution.WristFeedback.command.__code__.co_filename),
               Path(execution.GraspEnvelope.bound.__code__.co_filename)]
    signature = dict(object=obj, arm=arm, seconds=seconds, period=period, seed=paired['seed'],
                     source={f.name:digest(f) for f in sources}, upstream=aude.UPSTREAM,
                     model=paired['model.mjb'], initial=paired['integration_state'],
                     step_m=aude.DEFAULT_STEP_M, wrist_mcc=True, palm_proximity=False,
                     approach_reference='measured contacts', planning_object_geometries=0)
    if folder.exists():
        saved = read_json(folder/'run.json')
        if saved['signature'] != signature or any(digest(folder/n) != h for n,h in saved['hashes'].items()):
            raise ValueError(f'Output differs from requested trial at {folder}')
        return
    if (robot.objects or plant.surface_queries or robot.surface_queries
            or sha256(state.tobytes()).hexdigest() != paired['integration_state']
            or digest(case/'model.mjb') != paired['model.mjb']):
        raise ValueError('Planning scene or paired initial state mismatch')
    live = aude.state_posture(robot, sensed)
    grasp = execution.GraspEnvelope(live.joints, robot.limits)
    ik = GraspIK(robot, grasp)
    planner = aude.AudeExplorer(robot, ik, paired['seed'], aude.DEFAULT_STEP_M)
    planner.directions = TimedDirections(paired['seed'])
    wrist = execution.WristFeedback(palm_proximity=False)
    feedback = execution.GraspFeedback(grasp)
    nominal_wrist = np.r_[live.position, execution.quaternion(live.orientation)]
    workspace = live.position.copy()
    normals = touch.normals.copy()
    rows = [execution.record(data,plant,touch,nominal_wrist,wrist,live.joints,live.joints,'initial',0.)]
    ticks = round(seconds/execution.CONTROL_DT)
    policy = None
    if arm == execution.ARMS[1]:
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy
        if warm is None:
            raise ValueError('The registered finger-policy trial requires a matched tactile history')
        policy = LocalFingerPolicy(paired['seed'], ticks + len(warm))
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as temporary:
        stage = Path(temporary)/'run'
        stage.mkdir()
        with (policy if policy is not None else nullcontext()), (stage/'plans.jsonl').open('w',encoding='utf-8') as log:
            if policy is not None:
                for i,packet in enumerate(warm):
                    policy.command(i,packet)
            began = perf_counter()
            for tick in range(ticks):
                planner.observe(sensed,touch)
                phase = tick % period
                if phase == 0:
                    t0 = perf_counter()
                    ik.center = touch_frame(touch,ik.center)
                    ik.last_fit_ms = []
                    planner.directions.elapsed_ms = 0.
                    plan = planner.plan(sensed,touch)
                    current = aude.state_posture(robot,sensed)
                    origin = execution.Posture(nominal_wrist[:3],execution.rotation(nominal_wrist[3:]),
                                               grasp.bound(current.joints))
                    position,orientation,joints = execution.Leg(origin,plan.posture,plan.mode).samples(period)
                    plan_ms = (perf_counter()-t0)*1000
                    row = dict(tick=tick,mode=plan.mode,accepted_joint_step=plan.mode=='explore',
                               live=plan.valid.tolist(),targets_m=plan.targets.tolist(),directions=plan.direction.tolist(),
                               target_wrist_m=plan.posture.position.tolist(),
                               target_wrist_xyzw=plan.posture.orientation.as_quat().tolist(),
                               joint_target_rad=plan.posture.joints.tolist(),roi_from_touch_m=ik.center.tolist(),
                               gp_length_m=planner.directions.tangent_length.tolist(),surface_cells=len(planner.gp.surface),
                               free_cells=len(planner.gp.free),target_error_mm=plan.error_mm,min_self_gap_m=plan.self_gap_m,
                               plan_ms=plan_ms,gp_direction_ms=planner.directions.elapsed_ms,
                               ik_ms=sum(ik.last_fit_ms),ik_calls=len(ik.last_fit_ms))
                    log.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                next_wrist = bounded_wrist(nominal_wrist,position[phase],orientation[phase],workspace)
                joint_nominal = grasp.bound(joints[phase])
                nominal = joint_nominal
                if policy is not None:
                    ahead_phase = min(phase + execution.LOOKAHEAD_STEPS//execution.CONTROL_STEPS, period-1)
                    future = next_wrist
                    for i in range(phase+1,ahead_phase+1):
                        future = bounded_wrist(future,position[i],orientation[i],workspace)
                    packet = execution.policy_observation(plant,data,wrist.forecast(future))
                    nominal = policy.command(len(warm)+tick,packet)
                normals[touch.measured_heads] = touch.normals[touch.measured_heads]
                segment = execution.blend_wrist(nominal_wrist,next_wrist)
                touch = execution.step_plant(plant,data,feedback,wrist,touch,segment,nominal,normals)
                nominal_wrist = next_wrist
                execution.robot_state(plant,robot,data,sensed)
                rows.append(execution.record(data,plant,touch,next_wrist,wrist,nominal,joint_nominal,
                                             plan.mode,perf_counter()-began))
                if (tick+1)%6000 == 0:
                    print(obj,arm,(tick+1)*execution.CONTROL_DT,'s',len(planner.gp.surface),'contact cells',flush=True)
            online_wall = rows[-1][-1]
        trace = {k:np.asarray(v) for k,v in zip(execution.TRACE_KEYS,zip(*rows),strict=True)}
        if (abs(data.time-seconds)>1e-6 or not np.isfinite(trace['qpos']).all()
                or np.any(trace['palm_proximity_drive_n']) or not np.all(np.diff(trace['wall_s'])>0)):
            raise ValueError('Invalid physics, clock or online access')
        np.testing.assert_allclose(grasp.bound(trace['finger_command'][1:]),trace['finger_command'][1:],rtol=0,atol=1e-9)
        np.savez_compressed(stage/'trace.npz', **trace)
        np.savez_compressed(stage/'native-clock.npz',first_plan_to_post_step_s=trace['wall_s'])
        outputs = ['trace.npz','plans.jsonl','native-clock.npz']
        policy_stats = None
        if policy is not None:
            policy_stats = dict(info=policy.info, timing=policy.report(stage))
            outputs.append('timing.npz')
        scoring_mesh_sha = digest(mesh)
        if scoring_mesh_sha != paired['surface.obj']:
            raise ValueError('Offline evaluation surface does not match the paired fixture')
        hashes = {name:digest(stage/name) for name in outputs}
        write_json(stage/'run.json',dict(signature=signature,hashes=hashes,trace_sha256=hashes['trace.npz'],
              seed=paired['seed'],initial_sha256=paired['integration_state'],model_sha256=paired['model.mjb'],
              planning_object_geometries=0,initial_file_sha256=digest(case/'initial.npz'),
              offline_mesh_sha256=scoring_mesh_sha,steps=ticks,wall_s=online_wall,
              setup_online_and_trace_write_s=perf_counter()-started,policy=policy_stats))
        stage.rename(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',required=True,choices=inputs.OBJECTS)
    parser.add_argument('--arm',choices=execution.ARMS,default=execution.ARMS[0])
    parser.add_argument('--period',type=int,choices=(18,90),default=90)
    parser.add_argument('--seconds',type=int,choices=(240,600),default=240)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True,help='Matched fixture directory with fixtures.json and hand.mjb')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.object,args.arm,args.out,args.seconds,args.period,args.fixtures)
    from .route_free_score import score
    score(args.out,args.fixtures/args.object/'surface.obj')
