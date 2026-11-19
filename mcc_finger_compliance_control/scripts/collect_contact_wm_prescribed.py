"""Single-finger prescribed-object experiment; no motion-time height fitting.

The initialized object pose fixes the relative height once. Published XY/yaw
commands do not depend on q, measured contacts, force, or a surface query.
The nominal pregrasp stays fixed; only the existing MCC adapts the fingers.
An explicit compensated control arm uses the old geometry height follower.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
import torch

import collect_contact_wm as collection
from collect_contact_wm_motion import MotionProbe
from collect_contact_wm_scan import ScanProbe


class PrescribedSweep:
    def __init__(self, seed):
        rng = np.random.default_rng(seed + 83011)
        angle = rng.uniform(-np.pi, np.pi)
        r = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        xy = np.array([[0., 0.], [.05, .02], [-.05, .05],
                       [-.05, -.035], [.06, -.035], [.04, .04]]) @ r.T
        yaw = np.array([0., .10, -.12, .08, -.10, .06])
        self.times = np.array([0., 6., 16., 28., 40., 60.])
        values = np.c_[xy, yaw]
        # Segment endpoints have zero speed/acceleration. Positions remain in
        # the convex interval between nodes. No repeat or discontinuous reset.
        self.spline = BPoly.from_derivatives(self.times,
                          np.stack([values, np.zeros_like(values), np.zeros_like(values)], 1))
        self.metadata = dict(seed=int(seed), base_times_s=self.times.tolist(),
                             xy_yaw_knots=values.tolist(),
                             z='constant initialized height',
                             feedback_dependencies=[], kind='prescribed_nonperiodic_C2_sweep')

    def evaluate(self, physical_time, rate, derivative=0):
        t = np.asarray(physical_time) * rate
        if np.any(t < 0) or np.any(t > 60) or not 2 <= rate <= 4:
            raise ValueError('Outside frozen 60s base clock / 2–4x rate')
        return self.spline(t, nu=derivative) * rate ** derivative


class PrescribedProbe(ScanProbe):
    def prepare_motion(self, meta):
        MotionProbe.prepare_motion(self, meta)
        self.programs = [PrescribedSweep(j['seed']) for j in self.jobs]
        self.scan_base_rotation = self.rot.cpu().numpy()
        self.scan_base_pose = self.base.cpu().numpy()
        for j, p in zip(self.jobs, self.programs):
            j.update(physical_motion_period_s=None, motion_program=p.metadata,
                     collector_variant=self.args.variant)
        meta.update(action_owner='fixed_pregrasp -> ordinary_MCC -> held_position_servo',
                    nominal_reference='fixed_q_frozen_before_static_MCC_loading',
                    collector_variant=self.args.variant,
                    no_active_joint_excitation=True,
                    object_path_reference=('none: independent prescribed XY/yaw and constant initialized Z'
                                           if self.args.variant == 'prescribed' else
                                           'explicit old fixed-q geometric height-compensation control'),
                    no_future_waypoint_planning=True)
        if self.args.variant == 'compensated':
            self.plan_object_path()
        else:
            path = np.stack([self.prescribed_object_motion(f).cpu().numpy()
                             for f in range(self.args.frames)])
            speed = np.linalg.norm(np.diff(path[:, :, :3], axis=0), axis=-1) / .01
            jump = np.linalg.norm(path[0, :, :3] - self.base.cpu().numpy()[:, :3], axis=-1)
            if speed.max() > self.args.max_object_command_speed or jump.max() > 1e-6:
                raise ValueError('Prescribed command exceeds frozen speed / initial continuity bound')
            self.object_path = torch.as_tensor(path, device=self.device)
            folder = collection.episode_folder(self.args, self.row)
            np.savez_compressed(folder / 'object_path_plan.npz', commanded=path, prescribed=path)
            collection.save_json(folder / 'object_path_plan.json', dict(
                method='prepublished object command; no surface/q/contact/force height fitting',
                maximum_speed_m_s=float(speed.max()), initial_jump_m=float(jump.max()),
                programs=[p.metadata for p in self.programs], command_gate=True))

    def prescribed_object_motion(self, frame, include_normal=True):
        pose = self.scan_base_pose.copy()
        for lane, (job, program) in enumerate(zip(self.jobs, self.programs)):
            value = program.evaluate(frame * .01, job['motion_rate'])
            pose[lane, :3] += self.scan_base_rotation[lane] @ np.r_[value[:2], 0.]
            r = self.scan_base_rotation[lane] @ Rotation.from_euler('z', value[2]).as_matrix()
            pose[lane, 3:] = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
        return torch.as_tensor(pose, device=self.device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--variant', choices=['prescribed', 'compensated'], required=True)
    opts = parser.parse_args()
    protocol = json.loads(opts.protocol.read_text())
    scripts = Path(__file__).resolve().parent
    for name, expected in protocol['source_hashes'].items():
        if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source differs from protocol: ' + name)
    for field, expected in protocol['collision_provenance'].items():
        if collection.collision_provenance()[field] != expected:
            raise ValueError('Wrong collision backend: ' + field)
    args = SimpleNamespace(**protocol['collector'])
    args.root = Path(args.root)
    args.variant = opts.variant
    args.run_tag += '_' + opts.variant
    if args.active_fingers is not None or args.dp_checkpoint or args.perturbation_rad or args.branches:
        raise ValueError('Single-finger fixed pregrasp experiment only')
    manifest = args.root / 'object_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol['object_manifest_sha256']:
        raise ValueError('Manifest differs from protocol')
    row = dict(next(r for r in json.loads(manifest.read_text())['objects']
                    if r['object_id'] == protocol['object']))
    row.update(motion_backend=args.backend, solver_line_search=args.solver_line_search,
               solver_warmstart=not args.disable_solver_warmstart,
               constraint_capacity=args.constraint_capacity,
               object_motion_protocol=dict(name=protocol['id'], variant=args.variant,
                   protocol_sha256=hashlib.sha256(opts.protocol.read_bytes()).hexdigest(),
                   source_hashes=protocol['source_hashes'], driver_update_s=.001,
                   object_weld_time_constant_s=.005, motion_owner='object',
                   object_gt_surface_height=args.variant == 'compensated',
                   physical_control_dt_s=.01, physics_substep_dt_s=.001,
                   no_finger_waypoints=True, no_contact_search=True))
    folder = collection.episode_folder(args, row)
    if folder.exists():
        raise FileExistsError('Refuse to overwrite: ' + str(folder))
    collection.Probe = PrescribedProbe
    result = collection.run(args, row)
    print(json.dumps(dict(status=result['status'], folder=str(folder)), indent=2))


if __name__ == '__main__':
    main()
