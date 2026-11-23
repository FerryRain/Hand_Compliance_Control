"""End-to-end I04-c trials with a fresh process and independently scored output."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, perf_counter_ns

import numpy as np
import trimesh

from .contracts import ExperimentConfig
from .fingertip_hand import DT, SURFACE
from .fingertip_traversal import surface_graph, write_json


BRANCHES = ('full-mcc', 'full-dp', 'fixed-mcc', 'fixed-dp')
QUALITY = {'coverage_10mm': .70, 'coverage_25mm': .95, 'fscore_5mm': .40}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare_wrist(case, output):
    from .fingertip_replay import chunks

    if output.exists():
        raise FileExistsError(output)
    audit = read(case / 'native_audit.json')
    trace = list(chunks(case))
    palm = np.concatenate([block['palm_command'] for block in trace])
    phase = np.concatenate([block['phase'] for block in trace])
    if not audit['native_replay_matched'] or len(palm) != audit['states']:
        raise ValueError('The common wrist requires a complete audited reference')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, palm_command=palm, phase=phase)
    return dict(source=case.name, file=output.as_posix(), steps=len(palm),
                wrist_sha256=sha256(palm.astype('<f8', copy=False).tobytes()).hexdigest())


def read_contacts(case):
    with np.load(case / 'live_head_contacts.npz', allow_pickle=False) as samples:
        return tuple(samples[key] for key in ('point_m', 'normal', 'step', 'max_region_load_n'))


def write_reconstruction(case, surface=SURFACE):
    import torch
    from . import fingertip_r_gpis as r

    started = perf_counter()
    torch.set_num_threads(1)
    points, normals, steps, _ = read_contacts(case)
    if len(points) < 4:
        write_json(case / 'reconstruction.json', dict(status='insufficient_contact_points', points=len(points)))
        return perf_counter() - started
    indices = r.voxel_representatives(points)
    selected = indices[r.farthest_indices(points[indices], r.CONTACT_BUDGET)]
    if len(selected) < 4:
        write_json(case / 'reconstruction.json', dict(status='insufficient_occupied_voxels', points=len(selected)))
        return perf_counter() - started
    workspace = r.bounds_for_contacts(trimesh.load(surface, force='mesh', process=False).vertices)
    try:
        mesh, info = r.reconstruct(points[selected].astype(np.float64), normals[selected].astype(np.float64),
                                  r.bounds_for_contacts(points), workspace)
    except (ValueError, RuntimeError) as error:
        write_json(case / 'reconstruction.json', dict(status='surface_extraction_failed', error=str(error)))
        return perf_counter() - started
    mesh.export(case / 'r_gpis.ply')
    np.savez_compressed(case / 'r_gpis_inputs.npz', point_m=points[selected], normal=normals[selected],
                        step=steps[selected], query_bounds_m=workspace)
    write_json(case / 'reconstruction.json', dict(status='written', occupied_voxels=len(indices), **info))
    return perf_counter() - started


def trial(args):
    if args.paced and args.branch.startswith('full'):
        raise ValueError('Paced measurements use the independently fixed wrist protocol')
    started = perf_counter()
    wake = dict()
    seconds = args.seconds if args.seconds is not None else 9000.
    if args.branch == 'full-mcc':
        from .fingertip_continuous import ContinuousTraversal

        result = ContinuousTraversal(args.output, seconds, collect_contacts=True).run()
    else:
        from .fingertip_online import run_online_trial

        result, wake = run_online_trial(args.branch, args.output, args.wrist, args.seed,
                                       seconds=args.seconds, paced=args.paced)
    rollout_s = perf_counter() - started
    print(f'{args.branch} executed {result["executed_s"]:.3f} simulated seconds, writing R-GPIS', flush=True)
    reconstruct_s = write_reconstruction(args.output)
    # perf_counter is a shared monotonic clock across processes on Windows with Python 3.10+.
    done = perf_counter_ns()
    write_json(args.output / 'task_written.json', dict(written_ns=done, rollout_s=rollout_s,
        reconstruction_s=reconstruct_s, branch=args.branch, seed=args.seed, paced=args.paced,
        states=result['states'], executed_s=result['executed_s'], **wake))


def suite(args):
    if args.cpu_ids:
        import psutil

        psutil.Process().cpu_affinity(args.cpu_ids)
    if args.paced and any(branch.startswith('full') for branch in args.branches):
        raise ValueError('Paced measurements use the independently fixed wrist protocol')
    args.root.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
        seed = args.seed + repeat
        schedule = args.branches if repeat % 2 == 0 else args.branches[::-1]
        for branch in schedule:
            tag = f'{branch}_r{repeat + 1:02d}'
            if args.paced:
                tag += '_paced'
            case = args.root / tag
            stopwatch = case / 'stopwatch.json'
            if stopwatch.exists():
                continue
            if case.exists():
                raise FileExistsError(f'Incomplete trial needs a fresh root: {case}')
            command = [sys.executable, '-u', '-m', 'Module.i04_geometric_traversal.fingertip_benchmark',
                       'trial', '--branch', branch,
                       '--output', str(case), '--seed', str(seed)]
            if args.wrist is not None:
                command.extend(('--wrist', str(args.wrist)))
            if args.seconds is not None:
                command.extend(('--seconds', str(args.seconds)))
            if args.paced:
                command.append('--paced')
            print(f'Start {tag}', flush=True)
            with (args.root / f'{tag}.log').open('wb') as log:
                start_ns = perf_counter_ns()
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                exit_ns = perf_counter_ns()
            if completed.returncode:
                raise RuntimeError(f'Trial {tag} failed, see {tag}.log')
            stamp = read(case / 'task_written.json')
            if not start_ns <= stamp['written_ns'] <= exit_ns:
                raise RuntimeError(f'Invalid external stopwatch for {tag}')
            stamp.update(repeat=repeat + 1, seconds_to_written=(stamp['written_ns'] - start_ns) * 1e-9,
                         seconds_through_process_exit=(exit_ns - start_ns) * 1e-9,
                         cpu_affinity=args.cpu_ids)
            write_json(stopwatch, stamp)
            print(f'End {tag}, {stamp["seconds_to_written"]:.3f} s to written result', flush=True)


def score_case(case, graph, truth, *, quality_gate=QUALITY):
    from .fingertip_replay import chunks
    from .fingertip_surface_report import coverage_score, mesh_accuracy
    from .oracle import MeasuredCoverage

    watch = read(case / 'stopwatch.json')
    summary = read(case / 'summary.json')
    wrist = sha256()
    trace_steps = 0
    for block in chunks(case):
        wrist.update(np.ascontiguousarray(block['palm_command'], dtype='<f8').tobytes())
        trace_steps += len(block['phase'])
    if ('wrist_sha256' in watch and watch['wrist_sha256'] != wrist.hexdigest()) or trace_steps != watch['states']:
        raise ValueError(f'Executed command tape mismatch in {case.name}')
    watch['wrist_sha256'] = wrist.hexdigest()
    points, normals, step, load = read_contacts(case)
    sample_steps = np.r_[np.arange(0, watch['states'], int(10 / DT)), watch['states']]
    ledger = MeasuredCoverage(graph, ExperimentConfig())
    coverage, evidence = coverage_score(ledger, points, normals, step,
                                       load < 8., watch['states'], sample_steps)
    np.savez_compressed(case / 'measured_area.npz', vertex_area_m2=ledger.vertex_area,
                        head_contacts_sha256=digest(case / 'live_head_contacts.npz'), **evidence)
    mesh_path = case / 'r_gpis.ply'
    accuracy = mesh_accuracy(trimesh.load_mesh(mesh_path, process=False), truth) if mesh_path.exists() else None
    main = coverage['all_heads']['area_fraction_by_radius_mm']
    quality = dict(coverage_10mm=main['10'], coverage_25mm=main['25'],
                   fscore_5mm=accuracy['at_mm']['5']['fscore'] if accuracy else 0.)
    result = dict(**watch, coverage=coverage, quality=quality, r_gpis_accuracy=accuracy,
                  time_samples_s=(sample_steps * DT).tolist(), stop_reason=summary['stop_reason'],
                  peak_load_n=max(summary['peak_load_n_fingers_then_palm']),
                  longest_over_40n_s=summary.get('longest_over_40n_s', 0.),
                  first_force_trip=summary.get('first_sustained_overload_stop'))
    if quality_gate is not None:
        result['passed_quality'] = all(quality[name] >= limit for name, limit in quality_gate.items())
    write_json(case / 'score.json', result)
    return result


def score(args):
    from .fingertip_surface_report import MESH_SAMPLES, METRIC_SEED

    graph = surface_graph()
    truth, _ = trimesh.sample.sample_surface(graph.mesh, MESH_SAMPLES, seed=METRIC_SEED + 1)
    rows = [score_case(path.parent, graph, truth) for path in sorted(args.root.glob('*/stopwatch.json'))]
    groups = {}
    for mode in (False, True):
        for branch in BRANCHES:
            records = [row for row in rows if row['branch'] == branch and row['paced'] == mode]
            if not records:
                continue
            numeric = ('seconds_to_written', 'executed_s', 'rollout_s', 'reconstruction_s')
            series = {key: [record[key] for record in records] for key in numeric}
            series.update({name: [record['quality'][name] for record in records] for name in QUALITY})
            groups[branch + ('_paced' if mode else '')] = dict(trials=len(records),
                passes=sum(record['passed_quality'] for record in records),
                metrics={key: dict(median=float(np.median(values)), minimum=min(values), maximum=max(values))
                         for key, values in series.items()})
    pairs = {}
    for mode in (False, True):
        for task in ('full', 'fixed'):
            left = {row['repeat']: row for row in rows if row['branch'] == f'{task}-mcc' and row['paced'] == mode}
            right = {row['repeat']: row for row in rows if row['branch'] == f'{task}-dp' and row['paced'] == mode}
            matches = sorted(left.keys() & right.keys())
            if not matches:
                continue
            for repeat in matches:
                a, b = left[repeat], right[repeat]
                ca = args.root / f'{a["branch"]}_r{repeat:02d}{"_paced" if mode else ""}'
                cb = args.root / f'{b["branch"]}_r{repeat:02d}{"_paced" if mode else ""}'
                if digest(ca / 'initial.npz') != digest(cb / 'initial.npz'):
                    raise ValueError(f'Initial state mismatch in pair {task}/{repeat}')
                ma, mb = read(ca / 'manifest.json'), read(cb / 'manifest.json')
                if any(ma[key] != mb[key] for key in ('model_sha256', 'surface_sha256')):
                    raise ValueError(f'Physical scenario mismatch in pair {task}/{repeat}')
                if ma['controller']['parameters'] != mb['controller']['parameters']:
                    raise ValueError(f'MCC parameters differ in pair {task}/{repeat}')
                if task == 'fixed' and a['wrist_sha256'] != b['wrist_sha256']:
                    raise ValueError(f'Fixed wrist mismatch in repeat {repeat}')
            pairs[task + ('_paced' if mode else '')] = [dict(repeat=repeat,
                dp_minus_mcc_s=right[repeat]['seconds_to_written'] - left[repeat]['seconds_to_written'],
                both_passed_quality=left[repeat]['passed_quality'] and right[repeat]['passed_quality'])
                for repeat in matches]
    write_json(args.root / 'end_to_end_results.json', dict(quality_gate=QUALITY, groups=groups, pairs=pairs, runs=rows))


def figures(args):
    from .fingertip_r_gpis import voxel_representatives
    from .fingertip_surface_report import MESH_SAMPLES, METRIC_SEED, render_summary

    graph = surface_graph()
    truth, _ = trimesh.sample.sample_surface(graph.mesh, MESH_SAMPLES, seed=METRIC_SEED + 1)
    for task in ('full', 'fixed'):
        cases = [args.root / f'{task}-{branch}_r01' for branch in ('mcc', 'dp')]
        branches, meshes, times = [], [], []
        for case in cases:
            record = read(case / 'score.json')
            points, _, _, _ = read_contacts(case)
            branches.append(dict(coverage=record['coverage'], voxels=points[voxel_representatives(points)],
                                 reconstruction=dict(accuracy=record['r_gpis_accuracy'])))
            meshes.append(trimesh.load_mesh(case / 'r_gpis.ply', process=False))
            times.append(np.asarray(record['time_samples_s']))
        np.testing.assert_array_equal(*times)
        output = args.root / f'{task}_figures'
        output.mkdir(exist_ok=True)
        label = 'Joint planning + MCC' if task == 'full' else 'Fixed-wrist finger IK + MCC'
        render_summary(output, times[0], branches, graph, meshes, truth, mcc_label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prepare = actions.add_parser('prepare-wrist')
    prepare.add_argument('--case', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    single = actions.add_parser('trial')
    single.add_argument('--branch', choices=BRANCHES, required=True)
    single.add_argument('--output', type=Path, required=True)
    batch = actions.add_parser('suite')
    batch.add_argument('--root', type=Path, required=True)
    batch.add_argument('--branches', nargs='+', choices=BRANCHES, default=list(BRANCHES))
    batch.add_argument('--repeats', type=int, default=2)
    batch.add_argument('--repeat-start', type=int, default=0)
    batch.add_argument('--cpu-ids', nargs='+', type=int, help='Optional isolated CPU affinity for trials')
    for action in (single, batch):
        action.add_argument('--wrist', type=Path)
        action.add_argument('--seed', type=int, default=20261109)
        action.add_argument('--seconds', type=float)
        action.add_argument('--paced', action='store_true')
    report = actions.add_parser('score')
    report.add_argument('--root', type=Path, required=True)
    plots = actions.add_parser('figures')
    plots.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare-wrist':
        print(json.dumps(prepare_wrist(args.case, args.output), indent=2))
    else:
        {'trial': trial, 'suite': suite, 'score': score, 'figures': figures}[args.action](args)


if __name__ == '__main__':
    main()
