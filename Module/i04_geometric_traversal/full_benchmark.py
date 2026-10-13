"""Frozen A/B MCC evaluation for complete mustard surface coverage."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .full_audit import audit


def benchmark(geometry, output, jobs=3):
    geometry, output = Path(geometry).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'benchmark_manifest.json').exists():
        raise FileExistsError('Use a fresh formal evaluation directory')
    source = Path(__file__).with_name('full_surface.py')
    frozen_hash = sha256(source.read_bytes()).hexdigest()
    manifest = dict(protocol='i04-mustard-full-surface.v1', seeds=[7, 11, 19], cells=['A-MCC', 'B-MCC'],
        budget_s=9000., common_acquisition_s=5., source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(), runtime_sha256=frozen_hash,
        geometry_source=str(geometry), dp='DEFERRED_BY_USER',
        completion='Every one of 8192 task vertices and 388 required points measured covered; final safe contact 150 ms')
    (output/'benchmark_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cases = [(cell, seed) for seed in manifest['seeds'] for cell in ('B-MCC', 'A-MCC')]
    def run(case):
        cell, seed = case
        if sha256(source.read_bytes()).hexdigest() != frozen_hash:
            raise RuntimeError('Frozen controller changed during evaluation')
        directory = output/f'{cell}_seed{seed}'; directory.mkdir()
        command = [sys.executable, '-u', '-m', 'Module.i04_geometric_traversal.full_surface',
            '--formal', '--geometry', str(geometry), '--output', str(directory),
            '--cell', cell, '--seed', str(seed), '--budget', '9000']
        with (directory/'console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{cell} seed{seed} exited {result.returncode}; see {directory}/console.log')
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        verified = audit(directory)
        return dict(cell=cell, seed=seed, directory=str(directory), summary=summary, audit=verified)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run, case): case for case in cases}
        for future in as_completed(futures):
            cell, seed = futures[future]
            try:
                result = future.result(); results.append(result)
                summary = result['summary']
                print('CASE_DONE', cell, seed, summary['stop_reason'],
                    summary['final_covered_area_fraction'], summary['visited_required_goals'], flush=True)
            except Exception as error:
                errors.append(dict(cell=cell, seed=seed, error=str(error)))
                print('CASE_ERROR', cell, seed, str(error), flush=True)
            (output/'benchmark_progress.json').write_text(json.dumps(dict(
                completed=[dict(cell=r['cell'], seed=r['seed'], full=r['summary']['full_surface_complete']) for r in results],
                errors=errors, total=len(cases)), indent=2), encoding='utf-8')
    pairs = []
    for seed in manifest['seeds']:
        directories = [output/f'{cell}_seed{seed}' for cell in ('A-MCC', 'B-MCC')]
        if not all((directory/'summary.json').exists() for directory in directories): continue
        snapshots = [np.load(directory/'initial.npz') for directory in directories]
        for key in ('qpos', 'qvel', 'ctrl'):
            np.testing.assert_array_equal(snapshots[0][key], snapshots[1][key])
        traces = [np.load(directory/'trace.npz') for directory in directories]
        common_time = min(float(t['time_s'][-1]) for t in traces)
        prefix = [float(t['coverage'][max(0, np.searchsorted(t['time_s'], common_time, side='right')-1)]) for t in traces]
        pairs.append(dict(seed=seed, initial_q_qvel_ctrl_exactly_equal=True,
            common_observation_s=common_time, A_coverage_at_common_time=prefix[0], B_coverage_at_common_time=prefix[1]))
    results.sort(key=lambda r: (r['seed'], r['cell']))
    status = dict(protocol=manifest['protocol'], results=results, errors=errors, paired_initial_states=pairs,
        audited_cases=len(results), requested_cases=len(cases),
        full_completed_cases=sum(r['summary']['full_surface_complete'] for r in results), dp='DEFERRED_BY_USER')
    (output/'benchmark_summary.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('BENCHMARK_DONE', json.dumps({key: status[key] for key in (
        'audited_cases', 'requested_cases', 'full_completed_cases', 'errors')}), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args(); result = benchmark(args.geometry, args.output, args.jobs)
    if result['errors']: raise SystemExit(1)


if __name__ == '__main__': main()
