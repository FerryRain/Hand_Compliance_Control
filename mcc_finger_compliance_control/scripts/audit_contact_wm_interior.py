"""Force authority and physical-state audit for fixed interior pregrasp trials.

CPU collision replay identifies intersecting hand parts at recorded peak frames;
it does not recompute or attribute the recorded GPU force to a particular pair.
"""
import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from audit_contact_wm_prescribed import audit as audit_motion


def audit(folder):
    result = audit_motion(folder)
    model = mujoco.MjModel.from_binary_path(str(folder / 'physics_model.mjb'))
    palm = model.body('robot/palm_lower').id
    palm_mocap = int(model.body_mocapid[palm])
    hinge_dofs = model.jnt_dofadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    with h5py.File(folder / 'rollout.h5') as h:
        raw = {k: h['raw_100hz'][k][:] for k in (
            'q', 'post_mcc_q', 'actuator_force', 'mcc_force_setpoint',
            'pad_net_force_world', 'palm_pose', 'self_peak_n', 'joint_margin_rad')}
    with np.load(folder / 'substeps.npz') as archive:
        data = {k: archive[k] for k in (
            'phase', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after',
            'dynamics_mass_times_acc', 'qfrc_smooth', 'qfrc_constraint',
            'qfrc_actuator', 'pad_force')}
    balance = (data['dynamics_mass_times_acc'] - data['qfrc_smooth']
               - data['qfrc_constraint'])[..., hinge_dofs]
    result['physics'] = dict(
        finite=all(np.isfinite(v).all() for v in data.values()),
        between_substeps_qpos_jump=float(np.max(np.abs(
            data['qpos_before'][1:] - data['qpos_after'][:-1]))),
        between_substeps_qvel_jump=float(np.max(np.abs(
            data['qvel_before'][1:] - data['qvel_after'][:-1]))),
        hand_dynamics_balance_max_abs_nm=float(np.max(np.abs(balance))),
        hand_dynamics_balance_p99_abs_nm=float(np.quantile(np.abs(balance), .99)))
    for row in result['rows']:
        lane, finger = row['lane'], row['finger']
        js = slice(4 * finger, 4 * finger + 4)
        force = np.linalg.norm(raw['pad_net_force_world'][1:, lane, finger], axis=-1)
        torque = raw['actuator_force'][1:, lane, js]
        target = raw['mcc_force_setpoint'][1:, lane, finger]
        margin = raw['joint_margin_rad'][1:, lane, js]
        row.update(
            force_target_n=np.unique(target).tolist(),
            force_tracking_rmse_n=float(np.sqrt(np.mean((force - target) ** 2))),
            force_median_n=float(np.median(force)),
            torque_limit_nm=.8,
            per_joint_torque_saturation_fraction=(np.abs(torque) >= .799).mean(0).tolist(),
            command_measurement_abs_p95_rad=np.quantile(np.abs(
                raw['post_mcc_q'][1:, lane, js] - raw['q'][1:, lane, js]), .95, axis=0).tolist(),
            minimum_joint_margin_rad=float(margin.min()),
            joint_margin_p05_rad=float(np.quantile(margin, .05)))
        init_idx = np.flatnonzero(data['phase'][:, lane] == 0)[-1000:]
        row['initialization_last_second'] = dict(
            force_median_n=float(np.median(np.linalg.norm(data['pad_force'][init_idx, lane, finger], axis=-1))),
            per_joint_torque_saturation_fraction=(np.abs(
                data['qfrc_actuator'][init_idx, lane][:, hinge_dofs[js]]) >= .799).mean(0).tolist())
        # raw sample i contains peaks over the ten preceding physical substeps.
        peak_control = int(np.argmax(raw['self_peak_n'][1:, lane]))
        motion_idx = np.flatnonzero(data['phase'][:, lane] == 1)
        candidates = motion_idx[10 * peak_control:10 * (peak_control + 1)]
        pairs = {}
        d = mujoco.MjData(model)
        d.mocap_pos[palm_mocap] = raw['palm_pose'][0, lane, :3]
        d.mocap_quat[palm_mocap] = raw['palm_pose'][0, lane, 3:]
        for tick in candidates:
            d.qpos[:] = data['qpos_before'][tick, lane]
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            mujoco.mj_collision(model, d)
            for contact in d.contact[:d.ncon]:
                names = tuple(sorted(model.geom(int(g)).name for g in contact.geom))
                if not all(name.startswith('robot/') for name in names):
                    continue
                if contact.dist < 0:
                    pairs[names] = min(pairs.get(names, 0.), float(contact.dist))
        row['self_peak_frame_geometry'] = dict(
            control_frame=peak_control + 1,
            recorded_self_peak_n=float(raw['self_peak_n'][peak_control + 1, lane]),
            intersecting_pairs=[dict(geoms=list(k), depth_mm=-1000*v)
                                for k, v in sorted(pairs.items(), key=lambda p: p[1])],
            limitation='CPU collision check of ten recorded GPU pre-step states; pair-specific GPU forces were not recorded.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folders', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = [audit(folder) for folder in args.folders]
    args.output.write_text(json.dumps(results, indent=2))
    for result in results:
        print(Path(result['folder']).parent.name, result['physics'])
        for row in result['rows']:
            print(row['finger'], 'strict/found', row['strict'], row['found'],
                  'P95/peak/self', row['pad_force_p95_n'], row['object_force_peak_substep_n'],
                  row['self_force_peak_substep_n'], 'torque saturated', row['per_joint_torque_saturation_fraction'])
