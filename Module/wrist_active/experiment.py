"""Run matched, closed-loop wrist Active trials and short counterfactual branches."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch

from ..acitive_02.active_a import robot_state
from ..acitive_02.aude import AudeExplorer, DEFAULT_STEP_M, JointIK
from ..acitive_02.experiment import CONTROL_DT, CONTROL_STEPS, digest, policy_observation
from ..i04_geometric_traversal.fingertip_hand import DT, FingertipHand, HEAD_FRACTION
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, STATE
from ..i04_geometric_traversal.physics import quaternion, rotation
from .history import GENERATED, ROBOT, HandHistory, build_history, read_model
from .planner import HORIZON, LOAD_LIMIT, PERIOD, WristPlanner, interpolate, pose


ROOT = GENERATED / 'gpis_wrist_active_20261121'
FAIR = GENERATED / 'i01_a_fair_20261121'
ACTIVE2 = GENERATED / 'active-02-a_20261121'
ARMS = ('history-dp','history-mcc','rigid-dp','no-gain-dp','aude-mcc')
CASES = [('ellipsoid',s) for s in ('start-01','start-02')]+[('meat-can',s) for s in ('start-01','start-02')]
BUDGET = 240.
PREVIEW = 20
PROBE_RANKS = 4


def save_json(path, content):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(content,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def scenario(obj,start):
    if obj=='pear':
        return ACTIVE2, GENERATED/'i01_a_policy_wrist_20261120/pear/plan/surface.obj'
    case=FAIR/'runs'/obj/start/'shared/fixed-dp_r01'
    return case, case/'surface.obj'


def seed_for(obj,start):
    if obj=='pear':
        return json.loads((ACTIVE2/'protocol.json').read_bytes())['seed']
    return 20261121+int(start[-2:])-1


def code_hashes():
    return {p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('planner.py'),Path(__file__).with_name('history.py'))}


def freeze(root):
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'hand-history.npz').exists():
        save_json(root/'history-sources.json',build_history(root/'hand-history.npz'))
    protocol=dict(method='wrist-only tactile GPIS',budget_s=BUDGET,checkpoint_s=[30,60,120,240],
        cases=CASES,arms=ARMS,additional_pear='same archived Active-02-A integration state, 240 s prefix',
        history_objects=['mustard','cracker'],history_sha256=digest(root/'hand-history.npz'),
        history_sources=json.loads((root/'history-sources.json').read_bytes()),source_sha256=code_hashes(),
        hand_model_sha256=digest(ROBOT),seed_rule='20261121+A/B index (0/1); archived seed on pear',
        physics_dt_s=DT,tactile_dt_s=CONTROL_DT,mcc_dt_s=CONTROL_DT,dp_interval_s=.1,
        committed_preview_s=PREVIEW*CONTROL_DT,scoring_horizon_s=HORIZON,replan_s=PERIOD,
        force_definition='distal head >=0.2 N and unit measured normal; four finger regions and palm each <8 N',
        feasibility='paired contact probability >=0.16, palm gap>=0.8 mm, two support residuals<6.5 mm',
        targets={'coverage_10mm':.70,'reconstruction_fscore_5mm':.40},
        branch_experiment='First feasible decision from 1 s with >=4 candidates. Four evenly spaced feasible score ranks; identical committed prefix, one complete 0.5 s candidate.',
        geometry='plant and offline metrics only; both planners receive an object-free kinematics model',
        sources={})
    for obj,start in [*CASES,('pear','active-02')]:
        case,surface=scenario(obj,start)
        protocol['sources'][obj+'/'+start]={k:digest(p) for k,p in dict(model=case/'model.mjb',initial=case/'initial.npz',offline_surface=surface).items()}
    file=root/'protocol.json'
    if file.exists() and json.loads(file.read_bytes())!=json.loads(json.dumps(protocol)):
        raise ValueError('Frozen protocol differs; use a fresh experiment output')
    save_json(file,protocol)


def setup(obj,start):
    case,_=scenario(obj,start)
    model=read_model(case/'model.mjb')
    data=mujoco.MjData(model)
    with np.load(case/'initial.npz') as d:
        state=d['integration_state']
        warm=d['warm_policy_observations'] if 'warm_policy_observations' in d else None
    mujoco.mj_setState(model,data,state,STATE)
    mujoco.mj_forward(model,data)
    grasp=(data.qpos[:3].copy(),rotation(data.qpos[3:7]),data.qpos[7:].copy())
    plant=FingertipHand(model,object_name=obj,grasp=grasp,min_head_fraction=None if obj=='pear' else HEAD_FRACTION,surface_queries=False)
    robot=FingertipHand(read_model(ROBOT),object_name='fixture',grasp=grasp,surface_queries=False)
    if robot.objects:
        raise ValueError('Unexpected object geometries in planner')
    plant.inward_in_palm[:]=robot.inward_in_palm[:]=(0.,0.,-1.)
    sensed=mujoco.MjData(robot.model)
    robot_state(plant,robot,data,sensed)
    np.testing.assert_allclose(data.geom_xpos[plant.pads],sensed.geom_xpos[robot.pads],atol=1e-7,rtol=0.)
    touch=plant.measure(data)
    if not touch.measured_heads.all() or touch.loads.max()>=LOAD_LIMIT:
        raise ValueError(f'Invalid four-contact start: {touch.forces}, {touch.loads}')
    return plant,data,robot,sensed,touch,state,warm


def run(root,obj,start,arm,seconds,branch=None,pilot=False):
    torch.set_num_threads(1)
    file=root/('pilot' if pilot else 'branches' if branch is not None else 'runs')/obj/start/(
        f'rank-{branch}' if branch is not None else arm)
    file.mkdir(parents=True,exist_ok=True)
    receipt=file/'run.json'
    if receipt.exists():
        saved=json.loads(receipt.read_bytes())
        if (saved['requested_s']!=seconds or saved['source_sha256']!=code_hashes()
                or saved['trace_sha256']!=digest(file/'trace.npz')):
            raise ValueError(f'Existing run differs from requested experiment: {file}')
        return saved
    if not pilot:
        protocol=json.loads((root/'protocol.json').read_bytes())
        case,_=scenario(obj,start)
        source=protocol['sources'][obj+'/'+start]
        if (protocol['source_sha256']!=code_hashes() or digest(case/'model.mjb')!=source['model']
                or digest(case/'initial.npz')!=source['initial'] or digest(ROBOT)!=protocol['hand_model_sha256']
                or digest(root/'hand-history.npz')!=protocol['history_sha256']):
            raise ValueError('Inputs differ from frozen protocol')
    began=perf_counter()
    plant,data,robot,sensed,touch,state,warm=setup(obj,start)
    q_initial=data.qpos[7:].copy()
    wrist=data.qpos[:7].copy()
    feedback=ForceFeedback()
    seed=seed_for(obj,start)
    if arm=='aude-mcc':
        explorer=AudeExplorer(robot,JointIK(robot),seed,DEFAULT_STEP_M)
        planner=None
    else:
        explorer=None
        planner=WristPlanner(robot,HandHistory(root/'hand-history.npz'),arm)
    ticks=round(seconds/CONTROL_DT)
    tape=np.tile(wrist,(ticks+PREVIEW+120,1))
    joint_tape=np.tile(q_initial,(len(tape),1))
    period=90 if explorer else round(PERIOD/CONTROL_DT)
    published=PREVIEW
    history_len=0 if warm is None else len(warm)
    policy=None
    if arm.endswith('-dp'):
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy
        policy=LocalFingerPolicy(seed,ticks+history_len+2)
    records={key:[] for key in ('qpos','palm_command','published_preview','finger_nominal','points_m','normals_outward','pad_valid','pad_force_n','region_load_n','wall_s')}
    def capture(cmd,preview,nom):
        for key,val in zip(records,(data.qpos,cmd,preview,nom,touch.points,touch.normals,touch.measured_heads,touch.forces,touch.loads,perf_counter()-began)):
            records[key].append(np.copy(val))
    capture(wrist,wrist,q_initial)
    normals=np.broadcast_to(rotation(wrist[3:]).apply([0.,0.,1.]),(4,3)).copy()
    plans=[]
    nominal=q_initial.copy()
    branch_data=None
    plan_wall=0.
    setup_s=perf_counter()-began
    end=ticks
    physical_wrist=wrist
    with (policy if policy else nullcontext()):
        if warm is not None and policy:
            for i,packet in enumerate(warm):
                policy.command(i,packet)
        for tick in range(ticks):
            if planner:
                planner.observe(touch,tick)
            else:
                explorer.observe(sensed,touch)
            if tick%period==0 and branch_data is None:
                stamp=perf_counter()
                base=tape[published-1].copy()
                n=period
                if explorer:
                    decision=explorer.plan(sensed,touch)
                    goal=pose(decision.posture.position,decision.posture.orientation)
                    goal_q=decision.posture.joints
                    info=dict(mode=decision.mode,choice=-1,pair=[],gain=0.,extra_gain=0.,feasible=0,candidates=[])
                    fraction=1.
                else:
                    previous=(planner.heading.copy() if planner.heading is not None else None,planner.safe,planner.recover_count)
                    goal,info=planner.plan(sensed,touch,base,tick)
                    if branch is not None and tick>=100 and info['feasible']>=PROBE_RANKS:
                        matrix=np.asarray(info['candidates'])
                        eligible=matrix[matrix[:,1]>0]
                        ordered=eligible[np.argsort(-eligible[:,5],kind='stable')]
                        idx=int(np.rint(branch*(len(ordered)-1)/(PROBE_RANKS-1)))
                        choice=int(ordered[idx,0])
                        planner.heading,planner.safe,planner.recover_count=previous
                        goal,info=planner.plan(sensed,touch,base,tick,forced=choice)
                        n=round(HORIZON/CONTROL_DT)
                        end=published+n
                        branch_data=dict(tick=tick,first_tick=published,last_tick=end,choice=choice,
                            predicted_gain=info['gain'],predicted_extra_gain=info['extra_gain'],pair=info['pair'],
                            state_sha256=sha256(np.ascontiguousarray(data.qpos).tobytes()+np.ascontiguousarray(data.qvel).tobytes()).hexdigest(),
                            prefix_sha256=sha256(tape[:published].tobytes()).hexdigest())
                    goal_q=q_initial
                    fraction=n*CONTROL_DT/HORIZON
                tape[published:published+n]=interpolate(base,goal,fraction*np.arange(1,n+1)/n)
                joint_tape[published:published+n]=joint_tape[published-1]+(goal_q-joint_tape[published-1])*np.arange(1,n+1)[:,None]/n
                published+=n
                if branch_data:
                    tape[published:published+PREVIEW+1]=tape[published-1]
                    published+=PREVIEW+1
                cost=perf_counter()-stamp
                plan_wall+=cost
                plans.append(dict(tick=tick,observed_through=tick,online_cells=len(planner.surface.samples) if planner else None,
                                  last_new_touch=planner.surface.last_observation_tick if planner else None,
                                  goal=goal.tolist(),planning_ms=1000*cost,**info))
            if tick+PREVIEW-1>=published:
                raise RuntimeError('Unpublished DP preview')
            if policy:
                nominal=policy.command(history_len+tick,policy_observation(plant,data,tape[tick+PREVIEW-1]))
            else:
                nominal=joint_tape[tick]
            normals[touch.measured_heads]=touch.normals[touch.measured_heads]
            sub=interpolate(physical_wrist,tape[tick],np.arange(1,CONTROL_STEPS+1)/CONTROL_STEPS)
            for ref in sub:
                data.ctrl[:]=feedback.command(plant,data,nominal,normals,touch,True)
                data.mocap_pos[plant.driver]=ref[:3]
                data.mocap_quat[plant.driver]=ref[3:]
                mujoco.mj_step(plant.model,data)
            physical_wrist=tape[tick]
            mujoco.mj_forward(plant.model,data)
            touch=plant.measure(data)
            robot_state(plant,robot,data,sensed)
            capture(physical_wrist,tape[tick+PREVIEW-1],nominal)
            if tick+1>=end:
                break
        if policy:
            policy.report(file)
    if branch is not None and branch_data is None:
        raise RuntimeError('No common feasible branch point within probe budget')
    trace={key:np.asarray(value) for key,value in records.items()}
    if not np.isfinite(trace['qpos']).all() or abs(data.time-end*CONTROL_DT)>1e-6:
        raise ValueError(f'Invalid final plant state at {data.time}, expected {end*CONTROL_DT}')
    wall=perf_counter()-began
    np.savez_compressed(file/'trace.npz',**trace)
    with (file/'plans.jsonl').open('w',encoding='utf-8') as output:
        for row in plans:
            output.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
    mask=trace['region_load_n'][1:].max(axis=1)<LOAD_LIMIT
    count=trace['pad_valid'][1:].sum(1)
    result=dict(obj=obj,start=start,arm=arm,requested_s=seconds,executed_s=end*CONTROL_DT,seed=seed,
        setup_s=setup_s,wall_to_last_observation_s=wall,planning_s=plan_wall,branch=branch_data,
        source_sha256=code_hashes(),trace_sha256=digest(file/'trace.npz'),plans_sha256=digest(file/'plans.jsonl'),
        integration_state_sha256=sha256(state.tobytes()).hexdigest(),policy=policy.info if policy else None,
        light_two=float(np.mean(mask&(count>=2))),light_three=float(np.mean(mask&(count>=3))),
        decision_count=len(plans),scan_plans=sum(p['mode']=='scan' for p in plans),peak_n=float(trace['region_load_n'].max()))
    save_json(receipt,result)
    print(f'{obj}/{start}/{arm} {branch=} {end*CONTROL_DT:.2f}s; light 2/3 {result["light_two"]:.1%}/{result["light_three"]:.1%}; planning {plan_wall:.2f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','probes','suite'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--object',choices=('ellipsoid','meat-can','pear'),default='ellipsoid')
    parser.add_argument('--start',default='start-01')
    parser.add_argument('--arm',choices=ARMS,default='history-dp')
    parser.add_argument('--seconds',type=float,default=BUDGET)
    parser.add_argument('--branch',type=int,choices=range(PROBE_RANKS))
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.action=='freeze':
        freeze(args.root)
    elif args.action=='run':
        run(args.root,args.object,args.start,args.arm,args.seconds,args.branch,args.pilot)
    elif args.action=='suite':
        for obj,start in CASES:
            run(args.root,obj,start,args.arm,BUDGET)
        if args.arm in ('history-dp','history-mcc'):
            run(args.root,'pear','active-02',args.arm,BUDGET)
    else:
        for obj,start in CASES:
            for rank in range(PROBE_RANKS):
                run(args.root,obj,start,'history-dp',10.,rank)


if __name__=='__main__':
    main()
