"""Region priorities use measured true-tip coverage, with parent deduplication."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from regional_contact_coverage import HEIGHT_EDGES


def parent_coverage(rows, kind='strict'):
    cells = [[set() for _ in range(12)] for _ in range(5)]
    heights = [set() for _ in range(5)]
    for row in rows:
        if not row.get('selected'):
            continue
        parent = row['related_plan_group_id']
        coverage = row['measured_coverage'][kind]
        for b in coverage['visited_height_bands']:
            heights[b].add(parent)
        for b, s in coverage['visited_cells']:
            cells[b][s].add(parent)
    return dict(cells=[[len(x) for x in band] for band in cells],
                heights=[len(x) for x in heights], cell_parents=cells)


def region_order(coverage, attempts, reservations=(), cell_goal=2, height_goal=8):
    """Balance cells and heights; active jobs count as temporary reservations."""
    counts = np.asarray(coverage['cells'], float).copy()
    heights = np.asarray(coverage['heights'], float).copy()
    for band, sector in reservations:
        counts[band, sector] += 1
        heights[band] += 1
    keys = []
    for b in range(5):
        for s in range(12):
            deficit = counts[b, s] < cell_goal or heights[b] < height_goal
            keys.append(((not deficit, counts[b, s], attempts.get((b, s), 0),
                          heights[b], b, s), (b, s)))
    return [value for _, value in sorted(keys)]


def region_reached(measured, target, kind='strict'):
    return list(target) in measured[kind]['visited_cells']


def nominal_fingertip_coverage(path, regions):
    """Count dense, certified contact witnesses, never palm-proxy positions."""
    with h5py.File(path) as h:
        if not bool(h.attrs.get('healthy_reference_admissible', False)):
            raise ValueError('Uncertified reference')
        count = len(h['palm_pose_object'])
        points = np.asarray(h['grasp_keyframe_contact_point_object'])[:, 0]
        frames = np.asarray(h['grasp_keyframe_frame_index'])[:, 0]
        if len(points) != count or not np.array_equal(frames, np.arange(count)):
            raise ValueError('Regional scheduling requires dense certified witnesses')
        valid = np.asarray(h['grasp_keyframe_valid'])[:, 0].astype(bool)
        valid &= np.asarray(h['grasp_keyframe_true_tip_geometry_valid'])[:, 0].astype(bool)
        band, sector, finite, off_pole = regions.indices(points)
    return dict(frames=count, band=band, sector=sector,
                valid=finite & off_pole & valid[:, None],
                coverage=regions.coverage(points, np.broadcast_to(valid[:, None], points.shape[:-1])))


def target_windows(nominal, target, minimum=800, maximum=4000, whole_tested=False):
    """Prefer long continuous windows and keep the original physical speed."""
    b, s = target
    hit = ((nominal['band'] == b) & (nominal['sector'] == s)
           & nominal['valid']).any(-1)
    count = len(hit)
    indices = np.flatnonzero(hit)
    if len(indices) < 50 or count < minimum:
        return []
    widths = [count] if not whole_tested else []
    widths += [min(count, maximum), min(count, 3000), min(count, 2000), min(count, 1500), minimum]
    windows = []
    for width in dict.fromkeys(widths):
        if width < minimum or (whole_tested and width == count):
            continue
        for center in np.quantile(indices, [.5, .25, .75]).astype(int):
            start = int(np.clip(center-width//2, 0, count-width))
            pair = (start, start+width)
            if pair not in windows and hit[pair[0]:pair[1]].sum() >= 50:
                windows.append(pair)
    return windows


def targeted_spec(index, seed, target, family):
    """Expand the old mid-body templates with explicit cap/bottom approaches.

    These parameters propose a route. Dense fingertip geometry and subsequent
    measured coverage still decide whether it reaches the requested region.
    """
    b, sector = target
    rng = np.random.default_rng(seed+index*7919+b*101+sector)
    group = 5000+index
    angle = (sector+.5)*30. + rng.uniform(-12., 12.)
    height = float(rng.uniform(HEIGHT_EDGES[b]+.02, HEIGHT_EDGES[b+1]-.02))
    spec = dict(candidate_id=10000+index*2, geometry_group=group, family=family,
        target_speed_m_s=.03 if index % 2 == 0 else .05,
        object_seed=800012+group, clearance_seed=seed+group,
        direction=int(rng.choice([-1, 1])), angle_deg=float(rng.uniform(220., 360.)),
        meridian_deg=angle, azimuth_deg=angle, section_fraction=height,
        tilt_deg=float(rng.uniform(0., 20.)), meridian_sweep_deg=0.,
        meridian_weave_deg=0., palm_yaw_deg=float(rng.uniform(-35., 35.)),
        arc_region='calibrated', target_region=[b,sector])
    if family in ('meridian_arc', 'spatial_spiral'):
        spec.update(section_fraction=-1., angle_deg=float(rng.uniform(160., 230.)),
            arc_region='cap' if b >= 3 else 'bottom' if b == 0 else 'calibrated')
    if family == 'spatial_spiral':
        spec['meridian_sweep_deg'] = float(rng.uniform(90., 220.))*spec['direction']
    elif family == 'direction_sweep':
        spec['meridian_weave_deg'] = float(rng.uniform(15., 30.))
    return spec


def choose_family(rows, attempted_families):
    counts = Counter(r['family'] for r in rows if r.get('selected'))
    families = ('meridian_arc', 'oblique_orbit', 'spatial_spiral', 'direction_sweep')
    return min(families, key=lambda name: (counts[name]+2*attempted_families[name], counts[name], name))


def generate_targeted_palm(spec, folder, env, scripts):
    import json
    import subprocess
    import sys
    from plan_exploration_dataset import time_by_speed
    from run_leap_hardware_evaluation import save
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'generation.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['spec'] != spec:
            raise ValueError('Regional parent identity changed')
        return result
    raw, timed = folder/'geometry.h5', folder/'palm.h5'
    command = [sys.executable, str(scripts/'generate_manifold_palm_plan.py'),
        '--output', str(raw), '--object-id', 'ycb_mustard', '--frames', '2400',
        '--path-mode', 'minimum_enclosing_ellipse', '--angle-deg', str(spec['angle_deg']),
        '--direction', str(spec['direction']), '--ellipse-section-fraction', str(spec['section_fraction']),
        '--ellipse-plane-tilt-deg', str(spec['tilt_deg']), '--ellipse-azimuth-deg', str(spec['azimuth_deg']),
        '--ellipse-meridian-deg', str(spec['meridian_deg']), '--global-ellipse-plane-mode', 'long_axis',
        '--ellipse-arc-region', spec['arc_region'], '--palm-tangent-sign', '-1',
        '--object-rotation-mode', 'uniform_so3', '--seed', str(spec['object_seed']),
        '--time-parameterization', 'arc_length', '--palm-yaw-offset-deg', str(spec['palm_yaw_deg']),
        '--palm-clearance-range-m', '.03', '.04', '--palm-clearance-seed', str(spec['clearance_seed']),
        '--meridian-sweep-deg', str(spec['meridian_sweep_deg']),
        '--meridian-weave-deg', str(spec['meridian_weave_deg'])]
    result = dict(spec=spec, command=command, status='generation_error')
    with (folder/'generation.log').open('w') as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    result['returncode'] = process.returncode
    if process.returncode == 0:
        result['timing'] = time_by_speed(raw, timed, spec['target_speed_m_s'], 60.)
        with h5py.File(timed, 'r+') as h:
            h.attrs.update(regional_target_band=spec['target_region'][0],
                regional_target_sector=spec['target_region'][1], exploration_family=spec['family'],
                related_plan_group_id=f'mustard_exploration_seed{spec.get("collection_seed",20261015)}_geometry{spec["geometry_group"]:04d}')
        result.update(status='palm_candidate', palm_plan=str(timed.resolve()))
    save(result_path, result)
    return result
