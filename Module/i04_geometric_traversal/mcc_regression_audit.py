"""Evidence and media checks for the 2026-10-20 MCC baseline investigation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

from .matched_mcc_delivery import page_check
from .published_whole_hand import ROOT


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prefix(case, until=212.316):
    fields = ('time_s', 'coverage', 'visited_vertex_count', 'forces_n', 'safe_belly', 'leader_finger')
    parts = []
    for chunk in read(case/'summary.json')['trace_chunks']:
        if chunk['start_s'] >= until:
            break
        with np.load(case/chunk['file']) as z:
            # Both timelines use the same half-open physical time interval.
            end = np.searchsorted(z['time_s'], until-1e-8)
            parts.append({key: z[key][:end] for key in fields})
    data = {key: np.concatenate([item[key] for item in parts]) for key in fields}
    return dict(applied_interval_s=[0., until], observations=len(data['time_s']),
                coverage_fraction=float(data['coverage'][-1]), visited_vertices=int(data['visited_vertex_count'][-1]),
                at_least_two_contact_fraction=float(np.mean(data['safe_belly'].sum(axis=1) >= 2)),
                peak_force_n=float(data['forces_n'].max()),
                recorded_routing_anchors=np.unique(data['leader_finger']).tolist())


def video_qa(directory):
    manifest = next(directory.glob('video_*_complete_1x.json'))
    meta = read(manifest)
    path = directory/meta['video']
    assert digest(path) == meta['video_sha256']
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == meta['frames'] and abs(seconds-count/meta['fps']) < .1
    assert meta['full_time_axis'] and meta['speed'] == 1. and meta['time_range_s'][0] == 0.
    reader = imageio.get_reader(path)
    samples = []
    try:
        for capture in (meta['captures'][0], meta['captures'][1], meta['captures'][-1]):
            index = int(Path(capture['file']).stem.rsplit('_', 1)[1])
            frame = reader.get_data(index)
            original = np.asarray(Image.open(directory/capture['file']).convert('RGB'))
            assert frame.shape == original.shape == (700, 2400, 3)
            mae = float(np.mean(abs(frame.astype(float)-original.astype(float))))
            assert mae < 10.
            name = f'decoded_{index:05d}.png'
            Image.fromarray(frame).save(directory/name)
            samples.append(dict(frame=index, native_time_s=capture['time_s'], encoding_mae=mae, file=name))
    finally:
        reader.close()
    return dict(status='PASSED_VIDEO_DECODE_AND_HASH', file=str(path.relative_to(ROOT)), frames=count,
                encoded_seconds=seconds, native_range_s=meta['time_range_s'], sha256=meta['video_sha256'], samples=samples)


def run(output):
    output = output.resolve()
    generated = ROOT/'Module/generated'
    old = generated/'whole_hand_surface_run10_20261019'
    a = generated/'i04a_mcc_traverse01_20261019'
    b = generated/'i04b_mcc_traverse01_20261019'
    old_manifest, b_manifest = read(old/'manifest.json'), read(b/'manifest.json')
    sources = []
    for name, expected in old_manifest['source_hashes'].items():
        assert digest(ROOT/name) == digest(old/'source_snapshot'/name) == expected
        assert b_manifest['source_hashes'][name] == expected
        sources.append(dict(path=name, sha256=expected))
    with np.load(old/'initial.npz') as x, np.load(b/'initial.npz') as y:
        assert x.files == y.files
        assert all(np.array_equal(x[key], y[key]) for key in x.files)
    models = [ET.fromstring((case/'model.xml').read_text()) for case in (old, b)]
    for model in models:
        for element in model.findall('./asset/*[@file]'):
            element.set('file', Path(element.get('file')).name)
    assert ET.tostring(models[0]) == ET.tostring(models[1])
    old_s, b_s = read(old/'summary.json'), read(b/'summary.json')
    with np.load(b/'trace_0010.npz') as z:
        index = int(np.argmin(abs(z['time_s']-212.210)))
        peak_index = int(np.argmax(z['forces_n'].max(axis=1)))
        failure = dict(boundary_s=float(z['time_s'][index]), final_s=float(z['time_s'][-1]),
                       point_jump_mm=(1000*np.linalg.norm(z['target_points_world'][index]-z['target_points_world'][index-1], axis=1)).tolist(),
                       thumb_before_n=float(z['forces_n'][index, 3]), thumb_2ms_later_n=float(z['forces_n'][index+1, 3]),
                       thumb_6ms_later_n=float(z['forces_n'][index+3, 3]),
                       mcc_thumb_offset_m=float(z['mcc_offset_m'][index, 3]),
                       route_progress_stayed_zero=bool(np.all(z['stroke_distance_m'][index:] == 0.)),
                       full_run_peak_time_s=float(z['time_s'][peak_index]),
                       full_run_peak_force_n=float(z['forces_n'][peak_index].max()),
                       source_trace_sha256=digest(b/'trace_0010.npz'))
    probe = read(output/'causal_probe_verified/probe.json')
    assert probe['archived_reproduction_passed']
    result = dict(status='REGRESSION_CONFIRMED_LOCAL_CAUSAL_REPRODUCTION_PASSED',
                  old_control_sources_unchanged=sources, identical_initial_state=True,
                  model_equal_except_asset_paths=True, unchanged_coverage_and_force_predicates=True,
                  old_final={k: old_s[k] for k in ('executed_s', 'coverage_fraction', 'peak_force_n', 'stop_reason')},
                  new_b_final={k: b_s[k] for k in ('executed_s', 'coverage_fraction', 'peak_force_n', 'stop_reason')},
                  common_prefix=dict(old_b=prefix(old), new_b=prefix(b)),
                  failure_boundary=failure, local_causal_probe=probe,
                  limitations=[
                      'Local 1s counterfactual is not a new complete traversal and does not validate a production fix.',
                      'Both counterfactual branches still exceed the 8N coverage-exclusion threshold.',
                      'Other changed gates, speed caps, anchor bookkeeping and the development version schedule are separate confounds.',
                      'No runtime controller was modified in this investigation; the failed paired run remains archived.',
                  ])
    (output/'regression_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    videos = {cell: video_qa(output/f'videos_{cell}') for cell in ('a', 'b')}
    # The page links this check record, so create it before validating links.
    qa = dict(status='VIDEO_PASSED_PAGE_CHECK_PENDING', videos=videos)
    (output/'delivery_checks.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    qa.update(status='PASSED', page=page_check(output/'review.html'))
    (output/'delivery_checks.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], common_prefix=result['common_prefix'], delivery=qa), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
