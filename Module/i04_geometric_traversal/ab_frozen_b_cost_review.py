"""Per-event original evidence and explicitly retrospective IK microbenchmarks.

The old B did NOT store per-call computation times or least-squares results.
This tool never fills those missing values with inferred motion durations.
Its optional benchmark is a new, paired static solve using saved references;
it is neither an exact replay of B's historical search nor a new B rollout.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import scipy

from .ab_frozen_b_planning import PlanningRecorder, plain
from .belly_contact import belly_ik_terms
from .contracts import WristTarget
from .full_surface import build_scene
from .matched_mcc import CommonSettings, WristOnlyReference
from .natural_belly_traversal import set_kinematics
from .physics import pose, rotation
from .published_whole_hand import install_asset_adapter
from .whole_hand_adaptive_completion import AdaptiveCompletionPlanner


PLAN_KINDS = {'COOPERATIVE_SURFACE_ROUTE', 'COOPERATIVE_CONTACT_PIVOT',
              'COMPLETION_SURFACE_ROUTE', 'NECESSARY_COOPERATIVE_REORIENTATION',
              'COMPLETION_COOPERATIVE_REORIENTATION'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def event_rows(case):
    events = read(case/'events.json')
    summary = read(case/'summary.json')
    rows = []
    for i, event in enumerate(events):
        if event['kind'] not in PLAN_KINDS:
            continue
        transfer = 'REORIENTATION' in event['kind']
        if transfer:
            start = float(event['time_s'])
            end = start
            for later in events[i+1:]:
                end = float(later['time_s'])
                if later['kind'] in ('COOPERATIVE_REAPPROACH', 'COMPLETION_REAPPROACH', 'TRANSFER_SEGMENT_REPLAN'):
                    break
                if later['kind'] in PLAN_KINDS:
                    break
        else:
            start = float(events[i-1]['time_s']) if i else 0.
            end = float(event['time_s'])
        motion = event.get('duration_s')
        cost = motion
        if event['kind'] == 'COMPLETION_COOPERATIVE_REORIENTATION':
            cost += 3.*(4-sum(event['expected_contacts']))
        rows.append(dict(plan_index=len(rows), event_index=i, kind=event['kind'],
            start_s=start, end_s=min(end, summary['executed_s']),
            interval_source='Logged transfer request to following result' if transfer else 'Previous logged event to route result; may include unlogged clearance motion',
            goal=event.get('goal'), anchor=event.get('anchor'), success=event.get('success'),
            reason=event.get('reason'), new_vertices=event.get('new_vertices'),
            route_length_m=event.get('length_m'), planned_motion_s=motion,
            selected_ranking_cost=cost,
            cost_definition=('transfer seconds + 3*(4-predicted contacts)' if 'COMPLETION_COOPERATIVE' in event['kind']
                             else 'transfer seconds' if transfer else 'First feasible route; no scalar route ranking objective'),
            optimizer_cost=None, original_planning_wall_s=None,
            original_timing_status='NOT_RECORDED_IN_B' if summary['method'] == 'B-MCC' else 'SEE_ONLINE_CALL_LEDGER',
            expected_contacts=event.get('expected_contacts')))
    return rows


def selected_native_rows(case, rows):
    summary = read(case/'summary.json')
    queries = {}
    for row in rows:
        # End state uses the last recorded observation before the event, not
        # an unrecorded future state. All benchmark inputs are independently
        # addressable by original native observation index.
        for endpoint in ('start', 'end'):
            index = max(0, min(summary['native_states']-1,
                               int(round(row[endpoint+'_s']/.002))-(endpoint == 'end')))
            row[endpoint+'_native_index'] = index
            queries.setdefault(index, None)
    offset = 0
    keys = ('time_s', 'sensor_command_rad', 'target_points_world', 'target_normals_world', 'leader_finger')
    for chunk in summary['trace_chunks']:
        count = chunk['states']-int(chunk.get('skip_last', 0))
        indices = [index for index in queries if offset <= index < offset+count]
        if indices:
            path = case/chunk['file']
            assert sha256(path.read_bytes()).hexdigest() == chunk['sha256']
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in keys}
                for index in indices:
                    queries[index] = {key: value[index-offset].copy() for key, value in arrays.items()}
        offset += count
    assert all(value is not None for value in queries.values())
    return queries


def save_tables(out, rows):
    (out/'original_plan_events.json').write_text(json.dumps(plain(rows), indent=2), encoding='utf-8')
    with (out/'original_plan_events.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def benchmark(case, output, limit=None):
    install_asset_adapter()
    case, out = Path(case).resolve(), Path(output).resolve()
    assert out != case and not out.is_relative_to(case)
    out.mkdir(parents=True, exist_ok=False)
    rows = event_rows(case)
    if limit is not None:
        rows = rows[:limit]
    native = selected_native_rows(case, rows)
    save_tables(out, rows)
    # A new scene is used ONLY for static FK/IK. No physical step, coverage
    # update, original model mutation or original case write occurs here.
    scene = out/'benchmark_scene'; scene.mkdir()
    h, graph, _, _ = build_scene(scene)
    planner = AdaptiveCompletionPlanner(h, graph)
    with np.load(case/'initial.npz') as initial:
        initial_q = initial['qpos'][np.r_[h.arm_qpos_adrs, h.hand_qpos_adrs]]
    a = WristOnlyReference(h, initial_q, planner.lower, planner.upper, CommonSettings())
    scratch = mujoco.MjData(h.model)
    records = []
    with PlanningRecorder(out/'benchmark_calls.jsonl', scope='RETROSPECTIVE_STATIC_IK_NOT_ORIGINAL_PLANNING') as recorder:
        recorder.observe_least_squares()
        for row in rows:
            start, end = [native[row[key+'_native_index']] for key in ('start', 'end')]
            root = start['sensor_command_rad']
            reference_q = end['sensor_command_rad']
            points, normals = end['target_points_world'], end['target_normals_world']
            anchor = int(end['leader_finger'])
            set_kinematics(h, scratch, reference_q)
            wrist = pose(h, scratch).copy()
            results = {}
            # Alternating order reduces systematic first-solver cache effects.
            for method in (('A', 'B') if row['plan_index'] % 2 == 0 else ('B', 'A')):
                with recorder.call('benchmark.'+method, plan_index=row['plan_index'],
                                   source_method=read(case/'summary.json')['method']) as measurement:
                    if method == 'A':
                        seed = root.copy(); seed[7:] = a.nominal_hand
                        q, ep, er, _ = a.safety.solve(seed, WristTarget(wrist), iterations=100)
                        np.testing.assert_array_equal(q[7:], seed[7:])
                        reason = 'A_WRIST_IK_UNREACHABLE' if ep > .002 or er > .04 else None
                    else:
                        fit = planner.fit_patch(root, points, normals, anchor=anchor,
                                                iterations=55, flexible=False)
                        q, reason = fit['q'], fit['reason']
                set_kinematics(h, scratch, q)
                actual = pose(h, scratch)
                errors = []
                angles = []
                for finger, geom in enumerate(h.tip_geom_ids):
                    e, _, info = belly_ik_terms(h.model, scratch, int(geom), points[finger], normals[finger],
                                               clearance_m=.0002, dof_indices=planner.order)
                    errors.append(float(np.linalg.norm(e[:3])))
                    angles.append(float(info['alignment_angle_deg']))
                results[method] = dict(wall_s=measurement['wall_s'], cpu_s=measurement['cpu_s'],
                    reason=reason, wrist_position_error_m=float(np.linalg.norm(actual[:3]-wrist[:3])),
                    wrist_orientation_error_rad=float((rotation(actual[3:])*rotation(wrist[3:]).inv()).magnitude()),
                    common_four_pad_position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                    common_four_pad_angle_mean_deg=float(np.mean(angles)),
                    common_four_pad_cost_position_only_m2=float(.5*np.sum(np.square(errors))),
                    arm_motion_rad=float(np.linalg.norm(q[:7]-root[:7])),
                    hand_motion_rad=float(np.linalg.norm(q[7:]-root[7:])))
            record = dict(plan_index=row['plan_index'], event_index=row['event_index'], kind=row['kind'],
                          source_start_s=float(start['time_s']), source_end_s=float(end['time_s']), results=results)
            records.append(record)
            with (out/'paired_static_ik.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if len(records) % 10 == 0:
                print('BENCHMARK', len(records), '/', len(rows), flush=True)
    metadata = dict(status='COMPLETED_RETROSPECTIVE_STATIC_IK_BENCHMARK', plans=len(rows),
        source_case=str(case), source_summary_sha256=sha256((case/'summary.json').read_bytes()).hexdigest(),
        source_events_sha256=sha256((case/'events.json').read_bytes()).hexdigest(),
        original_B_compute_times_available=False,
        input_caveat='Saved command root and endpoint, saved control points and filtered normals; original candidate sets and nominal waypoint targets were not saved',
        scope='One paired static IK request per recorded plan; not historical replanning latency or complete route search cost',
        fairness='Same saved arm root and source reference problem on B-state distribution; A uses the common initial nominal hand, B keeps the saved source hand seed. Same host, sequential solvers, alternating order; A receives endpoint palm pose only',
        native_dynamics_steps=0, original_case_modified=False,
        optimizer_cost_comparability='Own least-squares objectives differ; compare common four-pad position metric separately',
        execution_environment=dict(python=sys.version, platform=platform.platform(),
            processor=platform.processor(), logical_cpus=os.cpu_count(), numpy=np.__version__,
            scipy=scipy.__version__, mujoco=mujoco.__version__,
            threads={name: os.environ.get(name) for name in
                     ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            timing_design='One solve per method/event, alternating order; no repeated timing trials or CPU pinning; residual OS load is uncontrolled'),
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'benchmark_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.case, args.output, args.limit), indent=2))
