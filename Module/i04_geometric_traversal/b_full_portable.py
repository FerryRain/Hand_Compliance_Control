"""Relocate generated B model assets without changing archived XML or physics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .b_full_video import digest, read
from .published_whole_hand import ASSETS, ROOT, asset_map, install_asset_adapter


def register(case):
    """Map all model files to verified shared assets or this case's own assets."""
    case = Path(case).resolve()
    mapping = asset_map()
    shared_sources = {item['original_file'] for item in read(ASSETS)['assets']}
    xml_hash = digest(case / 'model.xml')
    references = [item.get('file') for item in
                  ET.fromstring((case / 'model.xml').read_text(encoding='utf-8')).findall('./asset/*[@file]')]
    catalog_path = case / 'portable_assets.json'
    if catalog_path.exists():
        catalog = read(catalog_path)
        assert catalog['model_sha256'] == xml_hash
    else:
        entries = []
        # These are the only locally generated files in the B scene builder.
        for source in references:
            if source in shared_sources:
                continue
            parts = source.replace('\\', '/').split('/')
            if len(parts) >= 2 and parts[-2] == 'collision_parts' and re.fullmatch(r'part_\d{3}\.obj', parts[-1]):
                relative = f'collision_parts/{parts[-1]}'
            elif parts[-1] == 'canonical_object.obj':
                relative = parts[-1]
            else:
                raise ValueError(f'Unexpected generated asset: {source}')
            path = case / relative
            entries.append(dict(original_file=source, path=relative, sha256=digest(path)))
        catalog = dict(model_sha256=xml_hash, assets=entries,
                       scope='Only asset path resolution changes; XML and mesh bytes are preserved')
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in catalog['assets']:
        path = (case / item['path']).resolve()
        if not path.is_relative_to(case):
            raise ValueError('Asset must stay inside the case directory')
        assert digest(path) == item['sha256'], item['path']
        mapping[item['original_file']] = path
    assert all(source in mapping for source in references)
    install_asset_adapter()
    relocated = sum(item['original_file'].replace('\\', '/').casefold() !=
                    str(mapping[item['original_file']]).replace('\\', '/').casefold()
                    for item in catalog['assets'])
    return dict(model_assets=len(references), case_assets=len(catalog['assets']),
                case_assets_relocated=relocated,
                model_sha256=xml_hash, all_assets_resolved=True)


def initialize_worker(directory):
    register(directory)
    from .whole_hand_parallel_audit import initialize_guard_worker
    initialize_guard_worker(directory)


def smoke(case, states=500):
    """Short native replay and spawned geometry checks using relocated assets."""
    if not 1 <= states <= 10000:
        raise ValueError('states must be between 1 and 10000')
    case = Path(case).resolve()
    assets = register(case)
    from .sliding_audit import _native_context
    from .whole_hand_parallel_audit import guard_chunk
    catalog = read(case / 'summary.json')['trace_chunks']
    chunk = catalog[0]
    assert digest(case / chunk['file']) == chunk['sha256']
    context = _native_context(case)
    model, data = context.model, context.data
    with np.load(case / 'initial.npz') as initial:
        for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(data, key)[:] = initial[key]
        data.time = float(initial['time'])
    mujoco.mj_forward(model, data)
    errors = dict(qpos=0., qvel=0., previous_command=0.)
    with np.load(case / chunk['file']) as trace:
        assert states <= len(trace['time_s'])
        for k in range(states):
            for name, actual, expected in (
                ('qpos', data.qpos[context.qpos], trace['q_rad'][k]),
                ('qvel', data.qvel, trace['qvel_rad_s'][k]),
                ('previous_command', data.ctrl[context.actuator_ids], trace['sensor_command_rad'][k]),
            ):
                error = float(np.max(np.abs(actual - expected)))
                errors[name] = max(errors[name], error)
                assert error <= 1e-9, (k, name, error)
            data.ctrl[context.actuator_ids] = trace['command_rad'][k]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    with ProcessPoolExecutor(max_workers=1, initializer=initialize_worker, initargs=(str(case),)) as pool:
        guard = pool.submit(guard_chunk, str(case), chunk, states * .002, True).result()
    assert guard['commands'] == guard['reference_guards'] == states
    return dict(status='PASSED_RELOCATED_B_NATIVE_SMOKE', states=states, assets=assets,
                errors=errors, geometry_guard=guard, full_trajectory_reaudited=False)


def audit(case, workers=4, output_name='downloaded_native_reaudit'):
    case = Path(case).resolve()
    assets = register(case)
    from . import b_full_mcc_audit as auditor
    auditor.portable_guard_initializer = initialize_worker
    result = auditor.audit(case, workers=workers, follow=False, output_name=output_name)
    source = case / output_name / Path(__file__).name
    source.write_bytes(Path(__file__).read_bytes())
    result.update(asset_relocation=assets, asset_adapter_sha256=digest(source))
    (case / output_name / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def verify_download(case, catalog_path=None):
    """Check published bytes, including LFS objects, without rerunning physics."""
    case = Path(case).resolve()
    catalog_path = catalog_path or ROOT / 'Module/evidence/2026-10-20_I04_B_FULL.json'
    catalog = read(catalog_path)
    assert catalog['protocol'] == 'i04.b.full-mcc.force-grace.v1'
    assert case == (ROOT / catalog['case']).resolve()
    missing, mismatched, total = [], [], 0
    for item in catalog['files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Catalog paths must stay inside the repository')
        if not path.is_file():
            missing.append(item['path'])
        elif path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            mismatched.append(item['path'])
        total += item['bytes']
    if missing or mismatched:
        raise RuntimeError(json.dumps(dict(
            status='DOWNLOAD_INTEGRITY_FAILED', missing=missing, mismatched=mismatched,
            hint='Run git lfs pull for missing or pointer-only large files; retain the published source version'),
            ensure_ascii=False))
    return dict(status='PASSED_B_DOWNLOAD_INTEGRITY', files=len(catalog['files']),
                bytes=total, scope='Published file bytes only; this command does not rerun physics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'assets', 'smoke', 'audit', 'report', 'video'))
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--states', type=int, default=500)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    if args.action == 'verify':
        result = verify_download(args.case)
    elif args.action == 'assets':
        result = register(args.case)
    elif args.action == 'smoke':
        result = smoke(args.case, args.states)
    elif args.action == 'audit':
        result = audit(args.case, args.workers, args.output_name or 'downloaded_native_reaudit')
    else:
        register(args.case)
        if args.action == 'report':
            from .b_full_review import report
            result = report(args.case)
        else:
            from .b_full_video import render
            result = render(args.case, output_name=args.output_name or 'downloaded_complete_videos')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
