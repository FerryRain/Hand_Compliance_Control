"""Isolated nonperiodic object-command experiment using the frozen WM collector.

The JSON protocol records the entire collector profile. This entry adds no
finger planning or feedback recovery and does not modify the periodic collector.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import collect_contact_wm as collection
from collect_contact_wm_motion import MotionProbe
from contact_wm_scan_motion import CompoundScan


class ScanProbe(MotionProbe):
    def initialize(self):
        original_close = self.env.close

        def close():
            try:
                if self.trace:
                    folder = collection.episode_folder(self.args, self.row)
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / 'substeps.npz', **{
                        k: np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
                    mujoco.mj_saveModel(self.env.sim.mj_model, str(folder / 'physics_model.mjb'), None)
            finally:
                original_close()

        self.env.close = close
        return self.finish_initialization(super().initialize())

    def prepare_motion(self, meta):
        super().prepare_motion(meta)
        self.programs = [CompoundScan(j['seed'] + 92021) for j in self.jobs]
        for job, program in zip(self.jobs, self.programs):
            job.update(physical_motion_period_s=None, physical_startup_ramp_s=None,
                       startup_semantics='C2 curve starts from zero velocity and acceleration',
                       motion_program=program.metadata)
        self.scan_base_rotation = self.rot.cpu().numpy()
        self.scan_base_pose = self.base.cpu().numpy()
        meta.update(action_owner='fixed_pregrasp -> ordinary_MCC -> held_position_servo',
                    nominal_reference='fixed_q_frozen_before_static_MCC_loading',
                    motion_programs=[p.sampled_report(self.args.frames * .01, j['motion_rate'])
                                     for p, j in zip(self.programs, self.jobs)],
                    motion_program_note='Nonperiodic XY plus rotation; existing geometric Z compensation can alter full-pose smoothness and is audited separately')
        collection.save_json(collection.episode_folder(self.args, self.row) / 'scan_program.json',
                             dict(jobs=self.jobs, commands=meta['motion_programs']))

    def prescribed_object_motion(self, frame, include_normal=True):
        pose = self.scan_base_pose.copy()
        for lane, (job, program) in enumerate(zip(self.jobs, self.programs)):
            v = program.evaluate(frame * .01, job['motion_rate'])
            pose[lane, :3] += self.scan_base_rotation[lane] @ np.r_[v[:2], 0.]
            rotation = self.scan_base_rotation[lane] @ Rotation.from_rotvec(v[2:]).as_matrix()
            pose[lane, 3:] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return torch.as_tensor(pose, device=self.device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--object', required=True)
    opts = parser.parse_args()
    protocol = json.loads(opts.protocol.read_text())
    scripts = Path(__file__).resolve().parent
    for name, expected in protocol['source_hashes'].items():
        if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source differs from frozen scan protocol: ' + name)
    args = SimpleNamespace(**protocol['collector'])
    args.root = Path(args.root)
    args.initial_roots = Path(args.initial_roots)
    for path, expected in [(args.root / 'object_manifest.json', protocol['object_manifest_sha256']),
                           (args.initial_roots, protocol['roots_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Input differs from frozen scan protocol: ' + str(path))
    actual = collection.collision_provenance()
    for field, expected in protocol['collision_provenance'].items():
        if actual[field] != expected:
            raise ValueError('Wrong collision backend: ' + field)
    if opts.object not in protocol['objects']:
        raise ValueError('Object outside frozen scan protocol')
    if not 0 < args.frames <= 3000 or args.motion_rate_range != [2., 4.]:
        raise ValueError('Scan covers at most 30s at the declared 2–4x speed')
    if args.branches or args.dp_checkpoint or args.perturbation_rad or not args.fixed_reference:
        raise ValueError('This experiment holds the nominal pregrasp fixed')
    objects = json.loads((args.root / 'object_manifest.json').read_text())['objects']
    row = dict(next(r for r in objects if r['object_id'] == opts.object))
    row.update(motion_backend=args.backend, solver_line_search=args.solver_line_search,
               solver_warmstart=not args.disable_solver_warmstart,
               constraint_capacity=args.constraint_capacity,
               object_motion_protocol=dict(
                   name=protocol['id'], collector_profile=protocol['collector'],
                   source_hashes=protocol['source_hashes'],
                   protocol_sha256=hashlib.sha256(opts.protocol.read_bytes()).hexdigest(),
                   driver_update_s=.001, object_weld_time_constant_s=.005,
                   motion_owner='object', object_gt_surface_height=True,
                   physical_control_dt_s=.01, physics_substep_dt_s=.001,
                   period_s=None, no_finger_waypoints=True, no_contact_search=True,
                   object_velocity_semantics='dynamic solver body'))
    folder = collection.episode_folder(args, row)
    if folder.exists():
        raise FileExistsError('Refuse to overwrite scan evidence: ' + str(folder))
    collection.Probe = ScanProbe
    result = collection.run(args, row)
    print(json.dumps(dict(object=opts.object, status=result['status'], folder=str(folder)), indent=2))


if __name__ == '__main__':
    main()
