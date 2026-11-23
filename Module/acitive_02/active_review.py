"""Check active-02-a observation provenance and compare it with archived active-02-b."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .active_a import REFERENCE
from .active_planner import GraspEnvelope, LONG_DISTAL
from .experiment import ARMS, CONTROL_DT, digest, read_json, write_json
from ..i04_geometric_traversal.fingertip_hand import JOINTS


def source_observations(trace, voxel, maximum_load):
    valid = trace['pad_valid'] & (trace['region_load_n'].max(axis=1) <= maximum_load)[:, None]
    ticks, fingers = np.nonzero(valid)
    points = trace['points_m'][ticks, fingers]
    normals = trace['normals_outward'][ticks, fingers]
    occupied = np.floor(points / voxel).astype(int)
    _, first = np.unique(occupied, axis=0, return_index=True)
    selected = np.sort(first)
    return points[selected], normals[selected], ticks[selected]


def verify(root, reference, protocol, hand):
    for name in ('model.mjb', 'initial.npz'):
        if digest(root / name) != digest(reference / name):
            raise ValueError(f'Unmatched experimental input {name}')
    if digest(root / 'planning-hand.mjb') != protocol['planning_hand_sha256']:
        raise ValueError('Unexpected planning model hash')
    if digest(reference / 'protocol.json') != protocol['reference_protocol_sha256']:
        raise ValueError('The archived comparison protocol changed')
    for name, expected in protocol['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'The evaluated planning source changed after acquisition: {name}')
    if any(str(hand.geom(i).name).startswith(('part_', 'bottle_')) for i in range(hand.ngeom)):
        raise ValueError('Object geometry found in the planner model')
    forbidden = ('object_mesh_input_to_planner', 'candidate_native_collision_queries', 'wrist_native_proximity')
    if any(protocol[key] for key in forbidden):
        raise ValueError('Object geometry input found in a tactile-only trial')
    dof = np.array([hand.joint(name).qposadr[0] for name in JOINTS])
    limits = np.array([hand.joint(name).range for name in JOINTS])
    reports = []
    for arm in ARMS:
        case = root / arm
        with np.load(case / 'trace.npz', allow_pickle=False) as trace, np.load(
                case / 'online-gpis-observations.npz', allow_pickle=False) as gpis:
            points, normals, ticks = source_observations(trace, protocol['signed_gpis']['voxel_m'],
                                                        protocol['qualified_surface_region_load_n'])
            np.testing.assert_array_equal(gpis['points_m'], points)
            np.testing.assert_array_equal(gpis['normals_outward'], normals)
            np.testing.assert_array_equal(trace['palm_proximity_drive_n'], 0.)
            joints = trace['qpos'][:, dof]
            grasp = GraspEnvelope(joints[0], limits)
            issued = trace['finger_command'][1:]
            np.testing.assert_allclose(issued, grasp.bound(issued), atol=1.e-9, rtol=0.)
            control_rate = np.max(abs(np.diff(issued, axis=0))) / CONTROL_DT
            if control_rate > protocol['maximum_finger_rate_rad_s'] + 1.e-7:
                raise ValueError(f'Issued rate exceeded the configured limit in {case}')
            delta = abs(trace['finger_nominal'] - trace['aude_joint_nominal']).max()
            if (delta == 0.) != (arm == ARMS[0]):
                raise ValueError(f'Finger nominal authority is wrong in {case}')
        plans = [json.loads(line) for line in (case / 'plans.jsonl').read_text(encoding='utf-8').splitlines()]
        plan_ticks = [plan['tick'] for plan in plans]
        count = np.searchsorted(ticks, plan_ticks, side='right')
        np.testing.assert_array_equal(count, [plan['gpis_contact_voxels'] for plan in plans])
        np.testing.assert_array_equal(ticks[count - 1], [plan['last_surface_observation_tick'] for plan in plans])
        report = dict(arm=arm, independently_matched_tactile_voxels=len(ticks), audited_plans=len(plans),
                      maximum_issued_sample_rate_rad_s=float(control_rate),
                      minimum_measured_long_distal_deg=np.rad2deg(joints[1:, LONG_DISTAL].min(axis=0)).tolist(),
                      minimum_measured_thumb_opposition_joint_deg=float(np.rad2deg(joints[1:, 13].min())),
                      measured_any_distal_reverse_pct=float(100 * (joints[1:, LONG_DISTAL] < 0.).any(axis=1).mean()))
        reports.append(report)
    return reports


def pair_summary(root, experiment, matched_seconds):
    results_file = root / 'results.json'
    signed_file = root / 'signed-results.json'
    signed = read_json(signed_file)
    if signed['coverage_results_sha256'] != digest(results_file):
        raise ValueError(f'Inconsistent reconstructed result archive {root}')
    reports = []
    for run, model in zip(read_json(results_file)['runs'], signed['runs']):
        if model['trace_sha256'] != run['trace_sha256'] or model['arm'] != run['arm']:
            raise ValueError(f'Inconsistent paired result in {root}')
        with np.load(root / run['arm'] / 'coverage.npz', allow_pickle=False) as coverage:
            i = round(matched_seconds / CONTROL_DT)
            matched = float(coverage['coverage_10_mm'][i])
        reports.append(dict(arm=run['arm'], duration_sim_s=run['seconds'], coverage_10_mm=run['coverage_10_mm'],
                            coverage_25_mm=run['coverage_25_mm'], qualified_multifinger=run['qualified_at_least_two_contacts'],
                            qualified_multifinger_while_moving=run['qualified_moving_at_least_two_contacts'],
                            coverage_10_mm_from_multifinger=run['coverage_from_at_least_2_contacts_10_mm'],
                            coverage_10_mm_at_matched_time=matched,
                            time_to_95pct_coverage_10_mm=run['coverage_milestones_10_mm']['95'],
                            fscore_at_5mm=model['final']['fscore_5_mm'],
                            symmetric_mean_mm=model['final']['symmetric_mean_mm'],
                            max_region_load_n=run['max_region_load_n'],
                            longest_above_40n_s=run['longest_above_40n_s'],
                            acquisition_plus_final_mesh_s=model['acquisition_plus_final_reconstruction_wall_s']))
    return dict(experiment=experiment, folder=root.name, results_sha256=digest(results_file),
                signed_results_sha256=digest(signed_file), runs=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', default=REFERENCE, type=Path)
    args = parser.parse_args()
    root, reference = args.output.resolve(), args.reference.resolve()
    protocol = read_json(root / 'protocol.json')
    contents = (root / 'planning-hand.mjb').read_bytes()
    hand = mujoco.MjModel.from_binary_path('planning-hand.mjb', {'planning-hand.mjb': contents})
    audit = verify(root, reference, protocol, hand)
    same_time = min(run['seconds'] for path in (root, reference) for run in read_json(path / 'results.json')['runs'])
    comparison = dict(matched_time_sim_s=same_time, tactile_audit=audit,
                      a=pair_summary(root, 'active-02-a', same_time),
                      b=pair_summary(reference, 'active-02-b', same_time))
    write_json(root / 'active-02-comparison.json', comparison)
    print(f'Both arms verified with {hand.ngeom} robot-only planner geoms and matched A/B time {same_time:.2f}s')


if __name__ == '__main__':
    main()
