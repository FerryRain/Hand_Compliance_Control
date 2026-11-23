"""Time active-02 control modules by rerunning the archived trajectories."""

import argparse
from array import array
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from shutil import copyfile
from time import perf_counter_ns
from unittest.mock import patch

import mujoco
import numpy as np

from . import active_a, cooperative
from .active_gpis import LocalGPIS, ObservedSurface
from .active_planner import ActiveSurfacePlanner, TactileProjection
from .aude import AudeExplorer, JointIK
from .experiment import ARMS, CONTROL_DT, HISTORY_TICKS, START_SOURCE, digest, read_json, write_json
from .surface_scan import TactileSurface
from .wrist import WristFeedback
from ..i04_geometric_traversal import fingertip_local_policy
from ..i04_geometric_traversal.fingertip_hand import FingertipHand
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, Leg, latency_ms


A_ARCHIVE = Path('Module/generated/active-02-a_20261121')
B_ARCHIVE = active_a.REFERENCE


class Timers:
    """Charge nested calls to their own module once."""

    def __init__(self):
        self.pending = []
        self.details = defaultdict(lambda: array('d'))
        self.module_times = defaultdict(float)

    def instrument(self, owner, name, module, label=None):
        function = getattr(owner, name)
        label = label or f'{owner.__name__.split(".")[-1]}.{name}'
        choose_module = module if callable(module) else None

        @wraps(function)
        def timed(*args, **kwargs):
            group = choose_module(kwargs) if choose_module else module
            current = [perf_counter_ns(), 0]
            self.pending.append(current)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter_ns() - current[0]
                self.pending.pop()
                if self.pending:
                    self.pending[-1][1] += elapsed
                exclusive_ms = (elapsed - current[1]) * 1.e-6
                self.details[(group, label)].append(exclusive_ms)
                self.module_times[group] += exclusive_ms / 1000.

        return patch.object(owner, name, timed)

    def report(self):
        return dict(module_seconds=dict(self.module_times), exclusive_functions={
            f'{module}/{label}': latency_ms(values) for (module, label), values in self.details.items()})


def projection_module(arguments):
    return 'joint_planning' if arguments['joint'] else 'wrist_support'


def instrument(timers, module):
    mappings = [
        (AudeExplorer, 'plan', 'active_decision'),
        (AudeExplorer, 'observe', 'active_observation'),
        (JointIK, '__init__', 'hand_reachability_setup'),
        (JointIK, 'fit', 'joint_planning'),
        (Leg, 'samples', 'joint_planning'),
        (WristFeedback, 'command', 'wrist_mcc'),
        (FingertipHand, 'measure', 'contact_measurement'),
        (mujoco, 'mj_step', 'physics'),
        (mujoco, 'mj_forward', 'physics'),
        (module, 'step_plant', 'plant_control'),
        (module, 'policy_observation', 'policy_packet'),
        (module, 'record', 'trace_recording'),
        (np, 'savez_compressed', 'trace_recording'),
    ]
    if module is active_a:
        mappings += [
            (ActiveSurfacePlanner, 'plan', 'active_decision'),
            (ActiveSurfacePlanner, 'observe', 'active_observation'),
            (ActiveSurfacePlanner, 'stabilize', 'wrist_support'),
            (ObservedSurface, 'refresh', 'online_gpis'),
            (ObservedSurface, 'patch', 'online_gpis'),
            (LocalGPIS, 'predict', 'online_gpis'),
            (TactileProjection, 'solve', projection_module),
            (module, 'robot_state', 'sensed_hand_kinematics'),
            (active_a.GraspFeedback, 'command', 'finger_mcc'),
        ]
    else:
        mappings += [
            (cooperative.CooperativeSweep, 'guide', 'active_decision'),
            (cooperative.CooperativeSweep, 'stabilize', 'wrist_support'),
            (cooperative.ContactProjection, 'solve', projection_module),
            (TactileSurface, 'add', 'active_observation'),
            (ForceFeedback, 'command', 'finger_mcc'),
        ]
    return [timers.instrument(owner, name, group) for owner, name, group in mappings]


def archived_policy(actions):
    class ArchivedPolicy:
        def __init__(self, seed, frames):
            self.info = dict(nominal_source='recorded frozen single_all targets')

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def command(self, frame, _packet):
            return actions[max(0, frame - HISTORY_TICKS + 1)]

        def report(self, _case):
            return {}

    return ArchivedPolicy


def match_trace(reference, replay):
    with np.load(reference, allow_pickle=False) as expected, np.load(replay, allow_pickle=False) as measured:
        frames = len(measured['qpos'])
        for key in ('qpos', 'pad_valid', 'finger_nominal', 'finger_command', 'wrist_nominal', 'pad_force_n'):
            np.testing.assert_array_equal(measured[key], expected[key][:frames], err_msg=f'Trace mismatch in {key}')
    return frames


def run_job(job):
    label, reference, output, arm, seconds = job
    module = active_a if label == 'active-02-a' else cooperative
    original = reference / arm
    protocol = read_json(reference / 'protocol.json')
    run = read_json(original / 'run.json')
    signed = read_json(reference / 'signed-results.json')
    final_reconstruction = next(entry['final'] for entry in signed['runs'] if entry['arm'] == arm)
    if seconds is not None:
        protocol['max_s'] = seconds
    trace = original / 'trace.npz'
    with np.load(trace, allow_pickle=False) as observations:
        nominal = observations['finger_nominal'] if arm == ARMS[1] else None
        last_original_sample_s = float(observations['wall_s'][-1])
    scratch = output / f'{label}-{arm}'
    scratch.mkdir()
    for name in ('model.mjb', 'initial.npz', 'planning-hand.mjb'):
        if (reference / name).is_file():
            copyfile(reference / name, scratch / name)
    args = (scratch, arm, protocol)
    if module is cooperative:
        with np.load(START_SOURCE, allow_pickle=False) as start:
            frame = protocol['start']
            args = (scratch, arm, start['input_palm_pose_world'][frame, 0].astype(float),
                    start['input_q_live'][frame, 0].astype(float), protocol)

    timers = Timers()
    with ExitStack() as stack:
        for measurement in instrument(timers, module):
            stack.enter_context(measurement)
        if nominal is not None:
            stack.enter_context(patch.object(fingertip_local_policy, 'LocalFingerPolicy', archived_policy(nominal)))
        replay = module.run_arm(args)
    count = match_trace(trace, scratch / arm / 'trace.npz')
    if seconds is None and round(run['seconds'] / CONTROL_DT) != count - 1:
        raise ValueError('Timing replay did not reach the archived terminal tick')
    original_times = dict(acquisition_wall_s=run['wall_execution_s'], last_sample_wall_s=last_original_sample_s,
                          local_180ms_block=run['planning_timing'], policy=run.get('policy_timing'),
                          policy_startup_s=run.get('policy', {}).get('process_startup_s'),
                          reconstruction_wall_s=final_reconstruction['reconstruction_wall_s'],
                          acquisition_plus_final_mesh_s=run['wall_execution_s'] + final_reconstruction['reconstruction_wall_s'])
    print(f'{label} {arm}: matched {count} archived frames', flush=True)
    report = dict(experiment=label, arm=arm, reference_folder=reference.name, trace_sha256=digest(trace),
                  matched_archive_frames=count, seconds_simulated=replay['seconds'], original=original_times,
                  replay_core_wall_s=replay['wall_execution_s'], **timers.report())
    write_json(scratch / 'module-timing.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, default=A_ARCHIVE)
    parser.add_argument('--b', type=Path, default=B_ARCHIVE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds < CONTROL_DT):
        parser.error('--seconds must cover at least one finite control tick')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scratch = root / 'replays'
    scratch.mkdir()
    jobs = [(label, archive.resolve(), scratch, arm, args.seconds)
            for label, archive in (('active-02-a', args.a), ('active-02-b', args.b)) for arm in ARMS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as workers:
        reports = list(workers.map(run_job, jobs))
    write_json(root / 'module-timing.json', dict(replay_budget_s=args.seconds, concurrent_replay_workers=len(jobs),
                                                mujoco_version=mujoco.__version__, runs=reports))


if __name__ == '__main__':
    main()
