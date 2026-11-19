"""Shared rigid-object pose programs for simultaneous multi-finger collection.

Reuse the existing physical initialization, virtual grasp and MCC unchanged.
The path is expressed in the local environment frame, never per-finger frames.
"""
import json
from pathlib import Path

import numpy as np
import torch

import collect_contact_wm as collection
import collect_contact_wm_rooted_interaction as rooted
from collect_contact_wm_motion import MotionProbe
from collect_contact_wm_virtual_root import VirtualRootProbe


class MultiWaypointProbe(VirtualRootProbe):
    def prepare_motion(self, meta):
        path = Path(self.args.object_waypoints)
        if rooted.sha(path) != self.args.object_waypoints_sha256:
            raise ValueError('Object path hash mismatch')
        with np.load(path) as z:
            command = z['commanded'].copy()
        if command.shape != (self.args.frames, self.n, 7):
            raise ValueError('Shared-object lane count mismatch')
        if not np.isfinite(command).all() or not np.allclose(
                np.linalg.norm(command[..., 3:], axis=-1), 1., atol=1e-6):
            raise ValueError('Invalid rigid pose')
        command[..., :3] += self.origins.cpu().numpy()[None]
        self.object_path = torch.tensor(command, device=self.device, dtype=torch.float32)
        error = float((self.object_path[0]-self.driver_pose).abs().max())
        speed = float((self.object_path[1:, :, :3]-self.object_path[:-1, :, :3]).norm(dim=-1).max()/.01)
        if error > 1e-6 or speed > self.args.max_object_command_speed:
            raise ValueError(f'Invalid handoff/speed: {error}, {speed}')
        MotionProbe.prepare_motion(self, meta)
        for job in self.jobs:
            job['force_target_n'] = self.args.force_target_n
        meta.update(no_future_waypoint_planning=False,
            nominal_reference='fixed virtual grasp; shared object pose waypoints',
            no_finger_waypoints=True, shared_object_per_active_group=True,
            virtual_root_reference=self.virtual_reference_report,
            interior_initialization=self.interior_init_report,
            object_waypoints_sha256=self.args.object_waypoints_sha256,
            motion_handoff_command_error=error, motion_command_max_speed_m_s=speed,
            mcc_candidate_overrides=dict(incremental_force_compliance=True,
                side_mobility=0., force_target_n=self.args.force_target_n,
                outward_travel_m=self.args.outward_travel_m, normal_step_m=self.args.normal_step_m),
            training_admissible=False)
        folder = collection.episode_folder(self.args, self.row)
        np.savez_compressed(folder/'object_path_plan.npz', commanded=command)
        (folder/'object_waypoint_design.json').write_text(path.with_suffix('.json').read_text())


if __name__ == '__main__':
    rooted.RootedInteractionProbe = MultiWaypointProbe
    rooted.main()
