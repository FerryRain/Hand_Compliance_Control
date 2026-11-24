"""Physical, paired-budget trials of the wrist sweep, with an optional paced 100 Hz loop."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, sleep

import mujoco
import numpy as np
import torch

from ..acitive_02.active_a import robot_state
from ..acitive_02.experiment import CONTROL_STEPS, digest, policy_observation
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback
from . import experiment
from .buffered_fingers import BufferedFingers
from .history import GENERATED, ROBOT
from .planner import interpolate
from .sweep import DT, PLAN_TICKS, SweepWrist
from .wrist_worker import WristWorker


OUT = GENERATED/'gpis_wrist_sweep_20261121'
PHASES = ('scan','withdraw','transfer','approach')
SMOOTH = ('oblate','ovoid','superellipsoid')


def hashes():
    here = Path(__file__)
    return {name:digest(here.with_name(name)) for name in
            ('sweep.py','buffered_fingers.py','wrist_worker.py','sweep_experiment.py','experiment.py','planner.py')}


def fixture(obj, start):
    base = experiment.FAIR
    try:
        if obj in SMOOTH:
            experiment.FAIR = GENERATED/'i01_a_smooth_20261121'
        case, mesh = experiment.scenario(obj,start)
        return experiment.setup(obj,start), case, mesh
    finally:
        experiment.FAIR = base


def run(folder, obj, start, arm, seconds, paced=False, seed=None):
    torch.set_num_threads(1)
    n = round(seconds/DT)
    seed = experiment.seed_for(obj,start) if seed is None else seed
    signature = dict(obj=obj,start=start,arm=arm,seconds=seconds,paced=paced,seed=seed,code=hashes())
    folder.mkdir(parents=True,exist_ok=True)
    receipt = folder/'run.json'
    if receipt.exists():
        old = json.loads(receipt.read_bytes())
        if old['signature'] != signature or old['trace_sha256'] != digest(folder/'trace.npz'):
            raise ValueError(f'Different rollout already exists in {folder}')
        return old
    began = perf_counter()
    (plant,data,hand,sensed,touch,initial,warm),case,mesh = fixture(obj,start)
    planner = SweepWrist(hand,sensed,touch,gpis=arm!='route-dp',recontact=arm!='no-recontact-dp')
    feedback = ForceFeedback()
    nominal = data.qpos[plant.q].copy()
    history = len(warm) if warm is not None else 0
    policy = (BufferedFingers(seed,n+history,nominal,hand.limits,paced=paced)
              if arm.endswith('-dp') else None)
    wrist = data.qpos[:7].copy()
    normals = touch.normals.copy()
    trace = {k:np.empty((n+1,*np.shape(v)),dtype=np.asarray(v).dtype) for k,v in dict(
        qpos=data.qpos,palm_command=wrist,published_preview=wrist,finger_nominal=nominal,
        points_m=touch.points,normals_outward=touch.normals,pad_valid=touch.measured_heads,
        pad_force_n=touch.forces,region_load_n=touch.loads,wall_s=0.,phase=np.uint8(0)).items()}
    def record(i, ahead, phase):
        for key, value in zip(trace,(data.qpos,wrist,ahead,nominal,touch.points,touch.normals,
                                    touch.measured_heads,touch.forces,touch.loads,perf_counter()-began,phase)):
            trace[key][i] = value
    record(0,wrist,0)
    plan_ms, service_ms, finish_ms = [], np.zeros(n), np.zeros(n)
    log = folder/'plans.jsonl'
    with (policy if policy else nullcontext()), WristWorker(planner,paced) as worker, log.open('w',encoding='utf-8') as stream:
        if policy and warm is not None:
            for i, packet in enumerate(warm):
                policy.command(i,packet)
        worker.observe(sensed,touch,0)
        worker.submit(sensed,touch,0,0)
        first_plan = worker.install(0,initial=True)
        clock = perf_counter()
        for tick in range(n):
            due = clock + tick*DT
            if paced:
                remaining = due-perf_counter()
                if remaining > .0012:
                    sleep(remaining-.0008)
                while perf_counter() < due:
                    pass
            t0 = perf_counter()
            info = worker.update(sensed,touch,tick,n) if tick else first_plan
            if info is not None:
                plan_ms.append(info['planning_ms'])
                stream.write(json.dumps(info,separators=(',',':'))+'\n')
            cmd, ahead = planner.command(touch)
            if policy:
                nominal = policy.command(tick+history,policy_observation(plant,data,ahead))
            normals[touch.measured_heads] = touch.normals[touch.measured_heads]
            for ref in interpolate(wrist,cmd,np.arange(1,CONTROL_STEPS+1)/CONTROL_STEPS):
                data.ctrl[:] = feedback.command(plant,data,nominal,normals,touch,True)
                data.mocap_pos[plant.driver] = ref[:3]
                data.mocap_quat[plant.driver] = ref[3:]
                mujoco.mj_step(plant.model,data)
            wrist = cmd
            mujoco.mj_forward(plant.model,data)
            touch = plant.measure(data)
            robot_state(plant,hand,data,sensed)
            record(tick+1,ahead,PHASES.index(planner.phase()))
            stop = perf_counter()
            service_ms[tick] = 1000*(stop-t0)
            if paced:
                finish_ms[tick] = 1000*(stop-due)
            if (tick+1)%3000 == 0:
                stream.flush()
                s = slice(tick-2998,tick+2)
                light = trace['region_load_n'][s].max(1)<8
                three = np.mean(light & (trace['pad_valid'][s].sum(1)>=3))
                print(f'{obj} {arm} {(tick+1)*DT:g}s | cells {len(planner.surface.samples)} | '
                      f'pass {planner.route.passes} | last 30s light3 {three:.1%} | '
                      f'escapes {planner.escape_count}',flush=True)
        acquired = perf_counter()
        if policy:
            policy.report(folder)
    if abs(data.time-n*DT)>1e-6 or not np.isfinite(trace['qpos']).all():
        raise ValueError('Invalid final simulation state')
    np.savez_compressed(folder/'trace.npz',**trace)
    np.savez_compressed(folder/'control-timing.npz',service_ms=service_ms,plan_ms=plan_ms,
                        finish_from_release_ms=finish_ms if paced else [])
    stats = dict(signature=signature, seed=seed,
                 initial_sha256=sha256(initial.tobytes()).hexdigest(),model_sha256=digest(case/'model.mjb'),
                 initial_file_sha256=digest(case/'initial.npz'),robot_sha256=digest(ROBOT),
                 offline_mesh_sha256=digest(mesh),trace_sha256=digest(folder/'trace.npz'),plan_sha256=digest(log),
                 setup_s=clock-began,acquire_after_setup_s=acquired-clock,steps=n,
                 timing_sha256=digest(folder/'control-timing.npz'),
                 policy=policy.info if policy else None,passes=planner.route.passes,
                 escape_count=planner.escape_count, phase_names=PHASES,late_wrist_handoffs=worker.late,
                 installs=policy.installs if policy else 0, missed_handoffs=policy.missed if policy else 0,
                 skipped_queries=policy.skipped if policy else 0, underrun_ticks=policy.underrun if policy else 0)
    experiment.save_json(receipt,stats)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--series',default='runs')
    p.add_argument('--object',default='pear',choices=('pear','ellipsoid','meat-can',*SMOOTH))
    p.add_argument('--start',default='active-02')
    p.add_argument('--arm',default='sweep-dp',choices=('sweep-dp','sweep-mcc','route-dp','no-recontact-dp'))
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--seconds',type=float,default=240.)
    p.add_argument('--paced',action='store_true')
    args = p.parse_args()
    run(args.root/args.series/args.object/args.start/args.arm,args.object,args.start,args.arm,args.seconds,args.paced,args.seed)


if __name__ == '__main__':
    main()
