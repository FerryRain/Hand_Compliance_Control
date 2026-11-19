"""Measure actual finger actuation and loaded footprint, not command coverage."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from collect_contact_wm import job_fingers


def longest_false(mask):
    edges = np.diff(np.r_[False, ~mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), initial=0))


def audit(folder):
    with h5py.File(folder / 'rollout.h5') as h:
        meta = json.loads(h.attrs['metadata'])
        fields = ['q', 'qvel', 'pre_mcc_q', 'post_mcc_q', 'object_pose', 'object_command_pose',
                  'pad_point_world', 'pad_normal_world', 'pad_net_force_world',
                  'pad_physical_found', 'strict', 'found', 'object_peak_n', 'self_peak_n',
                  'mcc_force_offset', 'joint_margin_rad', 'actuator_target_q', 'dt']
        raw = {k: h['raw_100hz'][k][:] for k in fields}
    rows = []
    finger_jobs = [(lane, {**job, 'finger': finger})
                   for lane, job in enumerate(meta['jobs']) for finger in job_fingers(job)]
    for lane, job in finger_jobs:
        f = job['finger']; js = slice(4*f, 4*f+4)
        pose = raw['object_pose'][1:, lane].astype(float)
        r = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
        force = np.linalg.norm(raw['pad_net_force_world'][1:, lane, f], axis=1)
        loaded = raw['pad_physical_found'][1:, lane, f].astype(bool) & (force >= .05)
        points = r.inv().apply(raw['pad_point_world'][1:, lane, f] - pose[:, :3])[loaded]
        normal = r.inv().apply(raw['pad_normal_world'][1:, lane, f])[loaded]
        strict = raw['strict'][1:, lane, f].astype(bool)
        spans = None; angle = None; cells = 0; new_cells = []
        if len(points) > 2:
            _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
            spans = (np.ptp(points @ axes.T, axis=0)*1000).tolist()
            cells = len(np.unique(np.floor(points/.005).astype(int), axis=0))
            n0 = normal[:min(10,len(normal))].mean(0); n0 /= max(np.linalg.norm(n0), 1e-12)
            normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
            angle = float(np.quantile(np.degrees(np.arccos(np.clip(normal @ n0, -1, 1))), .95))
            previous = set(); frames = np.flatnonzero(loaded)
            for start in range(0, len(loaded), 500):
                here = set(map(tuple, np.floor(points[(frames>=start)&(frames<start+500)]/.005).astype(int)))
                new_cells.append(len(here-previous)); previous |= here
        cfg = meta['mcc_config']; offset = raw['mcc_force_offset'][1:,lane,f]
        inward = cfg.get('thumb_max_inward_offset') if f == 3 else None
        if inward is None: inward = cfg['max_normal_offset']
        outward = cfg.get('thumb_max_outward_offset') if f == 3 else None
        if outward is None: outward = cfg.get('force_servo_max_outward_offset')
        if outward is None: outward = cfg['max_normal_offset']
        command = raw['object_command_pose'][1:,lane]
        incremental=meta.get('mcc_candidate_overrides',{}).get('incremental_force_compliance',False)
        rows.append(dict(lane=lane, finger=f, seed=job['seed'], motion_rate=job['motion_rate'],
            variant=meta.get('collector_variant',meta['nominal_reference']), frames=len(loaded),
            measured_q_range_deg=np.degrees(np.ptp(raw['q'][1:,lane,js],axis=0)).tolist(),
            nominal_q_range_deg=np.degrees(np.ptp(raw['pre_mcc_q'][1:,lane,js],axis=0)).tolist(),
            post_mcc_q_range_deg=np.degrees(np.ptp(raw['post_mcc_q'][1:,lane,js],axis=0)).tolist(),
            strict=float(strict.mean()), found=float(raw['found'][1:,lane,f].mean()),
            longest_strict_loss_frames=longest_false(strict),
            loaded_contact_pca_span_mm=spans,loaded_contact_5mm_cells=cells,
            new_loaded_contact_5mm_cells_each_5s=new_cells,
            loaded_normal_angle_p95_deg=angle,
            object_command_xyz_span_mm=(np.ptp(command[:,:3],axis=0)*1000).tolist(),
            pad_force_p95_n=float(np.quantile(force,.95)), pad_force_peak_100hz_n=float(force.max()),
            object_force_peak_substep_n=float(raw['object_peak_n'][1:,lane].max()),
            self_force_peak_substep_n=float(raw['self_peak_n'][1:,lane].max()),
            mcc_offset_range_mm=[float(offset.min()*1000),float(offset.max()*1000)],
            offset_semantics='current-Jacobian diagnostic projection, not integrated control memory' if incremental else 'scalar force integrator',
            mcc_inward_bound_fraction=None if incremental else float((offset>=inward-1e-5).mean()),
            mcc_outward_bound_fraction=None if incremental else float((offset<=-outward+1e-5).mean()),
            motion_finite=all(np.isfinite(a).all() for a in raw.values()),
            actuator_post_mcc_max_abs_rad=float(np.max(np.abs(raw['actuator_target_q'][:,lane]-raw['post_mcc_q'][:,lane]))),
            actual_actuator_matches_post_mcc_1e6=bool(np.allclose(raw['actuator_target_q'][:,lane],raw['post_mcc_q'][:,lane],rtol=0,atol=1e-6)),
            terminal_reason=meta['terminal_reason']))
    return dict(folder=str(folder), rows=rows,
                limitations='Representative loaded contact footprint is not true mesh area or material-slip distance. Normal angle is not principal curvature. 100Hz pad peaks can miss substep transients; object/self peaks include substeps. No blanket physics or teacher admission.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folders', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reports = [audit(f) for f in args.folders]
    args.output.write_text(json.dumps(reports, indent=2))
    for report in reports:
        for row in report['rows']:
            print(json.dumps(row), flush=True)
