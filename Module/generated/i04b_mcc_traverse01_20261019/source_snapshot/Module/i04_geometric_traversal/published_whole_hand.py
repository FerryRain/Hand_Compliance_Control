"""Portable access to the published Mustard data, with immutable evidence files.

The adapter changes asset path resolution only. Native control, collision
checks, contact classification and the archived replay are not rewritten.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'Module/generated/whole_hand_surface_run10_20261019'
CATALOG = ROOT / 'Module/evidence/2026-10-19_I04_WHOLE_HAND_SURFACE.json'
ASSETS = ROOT / 'Module/evidence/whole_hand_release_assets_20261019.json'
GEOMETRY = ROOT / 'Module/generated/whole_hand_geometry_touch02_20261018/geometry.npz'
SEED = ROOT / 'Module/generated/mustard_natural_fresh02_20261018'
_original_compile = None


def digest(path):
    result = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify():
    catalog = read(CATALOG)
    total = 0
    for item in catalog['files']:
        path = ROOT / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise RuntimeError(f'Missing/incomplete artifact: {item["path"]}; run git lfs pull')
        if digest(path) != item['sha256']:
            raise RuntimeError(f'Artifact hash mismatch: {item["path"]}')
        total += item['bytes']
    audit = read(CASE / 'independent_native_audit/audit.json')
    assert audit['status'] == 'PASSED_NATIVE_EVIDENCE_AUDIT'
    assert audit['full_surface_complete'] and audit['native_states'] == 2638518
    manifest = read(CASE / 'manifest.json')
    for name, expected in manifest['source_hashes'].items():
        assert digest(ROOT / name) == digest(CASE / 'source_snapshot' / name) == expected, name
    for item in read(ASSETS)['assets']:
        assert digest(ROOT / item['path']) == item['sha256'], item['path']
    return dict(status='PASSED_DOWNLOAD_INTEGRITY', files=len(catalog['files']),
                bytes=total, native_states=2638518, covered_vertices=8192, covered_goals=388,
                scope='File integrity and archived audit; this command does not rerun physics')


@lru_cache(maxsize=1)
def asset_map():
    result = {}
    for item in read(ASSETS)['assets']:
        path = ROOT / item['path']
        assert digest(path) == item['sha256'], item['path']
        result[item['original_file']] = path
    return result


def portable_compile(xml):
    root = ET.fromstring(xml)
    mapping = asset_map()
    for element in root.findall('./asset/*[@file]'):
        source = element.get('file')
        # Prefer the verified mapping even when the original author's drive
        # happens to exist. This also exercises relocation on the author host.
        if source in mapping:
            element.set('file', str(mapping[source]))
        elif not Path(source).is_file():
            raise FileNotFoundError(f'Unmapped model asset: {source}')
    return _original_compile(ET.tostring(root, encoding='unicode'))


def install_asset_adapter():
    global _original_compile
    if _original_compile is not None:
        return
    from Module.fr3_leap import model
    original = model.compile_model_xml
    _original_compile = original
    # Imports may already hold an alias. Replace aliases to this one function,
    # not physics algorithms, data, source files or XML evidence on disk.
    for module in list(sys.modules.values()):
        if getattr(module, '__name__', '').startswith('Module.'):
            if getattr(module, 'compile_model_xml', None) is original:
                module.compile_model_xml = portable_compile


def portable_guard_initializer(directory):
    install_asset_adapter()
    from . import whole_hand_parallel_audit as auditor
    original = getattr(auditor, '_published_original_initializer', auditor.initialize_guard_worker)
    original(directory)


def smoke(states=500):
    """Short continuous native replay and a spawned geometry worker check."""
    if not 1 <= states <= 10000:
        raise ValueError('smoke --states must be between 1 and 10000')
    install_asset_adapter()
    import mujoco
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from .sliding_audit import _native_context
    from .whole_hand_parallel_audit import guard_chunk
    summary = read(CASE / 'summary.json')
    chunk = summary['trace_chunks'][0]
    assert digest(CASE / chunk['file']) == chunk['sha256']
    context = _native_context(CASE)
    model, data = context.model, context.data
    with np.load(CASE / 'initial.npz') as initial:
        for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(data, name)[:] = initial[name]
        data.time = float(initial['time'])
    mujoco.mj_forward(model, data)
    errors = dict(qpos=0., qvel=0., previous_command=0.)
    with np.load(CASE / chunk['file']) as trace:
        for k in range(states):
            for label, actual, expected in (
                ('qpos', data.qpos[context.qpos], trace['q_rad'][k]),
                ('qvel', data.qvel, trace['qvel_rad_s'][k]),
                ('previous_command', data.ctrl[context.actuator_ids], trace['sensor_command_rad'][k]),
            ):
                error = float(np.max(np.abs(actual - expected)))
                errors[label] = max(errors[label], error)
                if error > 1e-9:
                    raise AssertionError((k, label, error))
            data.ctrl[context.actuator_ids] = trace['command_rad'][k]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    with ProcessPoolExecutor(max_workers=1, initializer=portable_guard_initializer, initargs=(str(CASE),)) as pool:
        guarded = pool.submit(guard_chunk, str(CASE), chunk, states * .002, True).result()
    assert guarded['commands'] == guarded['reference_guards'] == states
    return dict(status='PASSED_PORTABLE_NATIVE_SMOKE', states=states, time_s=states * .002,
                errors=errors, asset_paths_resolved=len(asset_map()), spawned_guard_worker=guarded,
                full_trajectory_reaudited=False)


def audit(case, workers, output_name):
    install_asset_adapter()
    from . import whole_hand_parallel_audit as auditor
    if output_name in ('independent_native_audit', 'parent_provenance', '.', '..'):
        raise ValueError('Use a new audit output name; archived evidence is immutable')
    if (Path(case) / output_name).exists():
        raise FileExistsError('Audit output already exists; choose a fresh --output-name')
    auditor._published_original_initializer = auditor.initialize_guard_worker
    auditor.initialize_guard_worker = portable_guard_initializer
    return auditor.audit(case, workers=workers, output_name=output_name)


def run_fresh(output, seconds):
    """Start v8 planning at zero coverage; this is a NEW experiment, not run10."""
    install_asset_adapter()
    import numpy as np
    from .whole_hand_adaptive_completion import AdaptiveCompletionTraversal, AdaptiveCompletionPlanner
    from .whole_hand_recovery import RecoveryWholeHandTraversal
    from .whole_hand_geometry_cache import CommandGeometryCache
    from .whole_hand_belly_probe import snapshot_sources

    class FreshTraversal(AdaptiveCompletionTraversal):
        def __init__(self):
            RecoveryWholeHandTraversal.__init__(self, output, GEOMETRY, SEED, seconds, resume=None)
            self.planner = AdaptiveCompletionPlanner(self.h, self.graph, SEED)
            self.planner.load_cooperative_seeds(CASE)
            self.command_cache = CommandGeometryCache(self.planner.safety)
            self.planner.safety.command_collision_reason = self.command_cache.command_reason
            self.nominal = self.last.copy()
            self.attempts = np.zeros(len(self.graph.vertices_m), int)
            manifest = read(self.out / 'manifest.json')
            manifest.update(protocol='i04-whole-hand-surface.development.v8-fresh-published',
                completion_contact_target_geodesic_radius_m=.018, minimum_completion_stroke_m=.001,
                fallback_candidate_clearances_m=[.001, .002, .003],
                seed_use='Published verified poses as geometry-only IK seeds; ZERO inherited physical state or coverage',
                cooperative_seed_source=str(CASE), fresh_single_version=True, coverage_inherited=False,
                published_result_reproduction=False, all_versions_reference_guard_audit_required=True,
                source_hashes=snapshot_sources(self.out))
            (self.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        def run(self):
            result = super().run()
            # The archived v7/v8 runner was continuation-only and hard-coded
            # these reporting fields. This new entry starts from zero.
            result.update(coverage_inherited=False, fresh_single_version=True,
                          resume=None, force_releases_scope='Whole trajectory',
                          published_result_reproduction=False)
            (self.out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result
    return FreshTraversal().run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Verify every uploaded report, video, trace and source hash')
    smoke_parser = sub.add_parser('smoke', help='Short native replay plus a spawned geometry worker')
    smoke_parser.add_argument('--states', type=int, default=500)
    audit_parser = sub.add_parser('audit', help='Full continuous replay; writes a separate audit directory')
    audit_parser.add_argument('--case', type=Path, default=CASE)
    audit_parser.add_argument('--workers', type=int, default=6)
    audit_parser.add_argument('--output-name', default='downloaded_native_reaudit')
    fresh = sub.add_parser('run-fresh', help='New v8 experiment from zero coverage, with geometry-only pose seeds')
    fresh.add_argument('--output', type=Path, required=True)
    fresh.add_argument('--seconds', type=float, default=7000.)
    args = parser.parse_args()
    if args.command == 'verify': result = verify()
    elif args.command == 'smoke': result = smoke(args.states)
    elif args.command == 'audit': result = audit(args.case, args.workers, args.output_name)
    else: result = run_fresh(args.output, args.seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
