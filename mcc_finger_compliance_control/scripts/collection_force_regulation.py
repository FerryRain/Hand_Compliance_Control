"""Collection force profiles and final arbitration at measured contact points.

Connector contacts are protected collisions, never valid fingertip contacts.
This projection does not change the persistent fingertip Jacobian anchors.
"""
from dataclasses import asdict, replace

import mujoco
import numpy as np

from true_fingertip_contact import ContactSlotObserver


class CollectionForceObserver(ContactSlotObserver):
    """Matched strongest physical contact for each pad and connector geom.

The netforce sensor's normal is a placeholder, not a solver contact normal.
Keep its resultant for ordinary regulation, add matched point/normal/force
only for the final physical collision guard.
    """
    def configure(self, cfg):
        super().configure(cfg)
        self.guard_names = {}
        for role, names in (('pad', self.pad_names), ('connector', self.names)):
            self.guard_names[role] = []
            for name in names:
                source = next(s for s in cfg.scene.sensors if s.name == name)
                sensor = replace(source, name=name+'_force_guard', reduce='maxforce',
                    fields=('found', 'force', 'pos', 'normal', 'tangent'),
                    num_slots=1, global_frame=False)
                cfg.scene.sensors = (*cfg.scene.sensors, sensor)
                self.guard_names[role].append(sensor.name)

    def read(self, env, control_entries):
        import torch
        result = super().read(env, control_entries)
        for role, names in self.guard_names.items():
            values = []
            for name in names:
                sensor = env.scene[name]; sensor.update(0.)
                values.append(sensor.data)
            self.latest.update({
                role+'_physical_found': torch.stack([v.found[:, 0]>0 for v in values], 1),
                role+'_physical_force': torch.stack([v.force[:, 0] for v in values], 1),
                role+'_physical_points': torch.stack([v.pos[:, 0] for v in values], 1),
                role+'_physical_inward': torch.stack([v.normal[:, 0] for v in values], 1)})
        return result


def collection_force_config(config, profile):
    if profile == 'legacy':
        return config
    if profile != 'regulated':
        raise ValueError(f'Unknown collection force profile: {profile}')
    return replace(config, desired_force_per_finger=(6., 6., 6., 8.),
                   force_servo_max_outward_offset=.006,
                   overforce_hard_ratio=2.5, thumb_overforce_hard_ratio=2.5)


def force_profile_description(config, profile):
    return dict(name=profile, revision=3, config=asdict(config),
                connector_warning_n=3. if profile == 'regulated' else None,
                connector_escape_n=6. if profile == 'regulated' else None,
                activation='after accepted legacy settling, at motion start',
                feedback='3D pad net resultant for ordinary regulation; matched strongest physical point/normal/force for the final collision guard',
                hard_stop='100 N physics-substep threshold; actual stop/record response is stored in overforce_response metadata')


def protect_collection_contacts(controller, q_live, q_candidate, palm_pose,
                                measurements, lane):
    """Arbitrate after recovery, so an inward recovery cannot defeat retreat.

Keep tangential intent when possible; above the hard threshold start that
finger from measured q. Enforce the local linear normal constraint, bounded
by joint limits and the existing per-tick joint step. This cannot guarantee
collision avoidance between 10 ms control decisions.
    """
    cfg = controller.config
    q = np.asarray(q_live, dtype=np.float64).reshape(16)
    candidate = np.asarray(q_candidate, dtype=np.float64).reshape(16)
    command = candidate.copy()
    controller._set_q(controller.data, q)
    rows = []
    active = np.zeros(4, dtype=bool)
    hard_active = np.zeros(4, dtype=bool)
    for role in ('pad', 'connector'):
        forces = np.linalg.norm(measurements[f'{role}_physical_force'][lane], axis=-1)
        found = measurements[f'{role}_physical_found'][lane]
        points = controller.points_world_to_palm(measurements[f'{role}_physical_points'][lane], palm_pose)
        outward = controller.vectors_world_to_palm(-measurements[f'{role}_physical_inward'][lane], palm_pose)
        for f in range(4):
            warning = 3. if role == 'connector' else 1.25 * controller.force_setpoint[f]
            hard_limit = 6. if role == 'connector' else 2.5 * controller.force_setpoint[f]
            if not found[f] or not np.isfinite(forces[f]) or forces[f] < warning:
                continue
            if not np.all(np.isfinite(points[f])) or not np.all(np.isfinite(outward[f])):
                continue
            length = np.linalg.norm(outward[f])
            if length < 1.e-6:
                continue
            jac = np.zeros((3, controller.model.nv))
            jac_rot = np.zeros_like(jac)
            mujoco.mj_jac(controller.model, controller.data, jac, jac_rot,
                          points[f], int(controller.tip_body_ids[f]))
            block = slice(4*f, 4*f+4)
            jn = (outward[f] / length) @ jac[:, controller.dof_indices[block]]
            if jn @ jn < 1.e-10:
                continue
            hard = forces[f] >= hard_limit
            rows.append((f, jn, cfg.solver_contact_escape_step if hard else 0.))
            active[f] = True
            hard_active[f] |= hard
    for f in np.flatnonzero(hard_active):
        command[4*f:4*f+4] = q[4*f:4*f+4]
        controller.force_servo_offset[f] = min(controller.force_servo_offset[f], 0.)
        controller.transient_search_offset[f] = 0.
    # Usually one row per finger. Alternating projections also handle a pad
    # and connector touching simultaneously; unresolved constraints are logged.
    for _ in range(4):
        for f, jn, minimum in rows:
            block = slice(4*f, 4*f+4)
            delta = command[block] - q[block]
            gap = minimum - float(jn @ delta)
            if gap > 0.:
                command[block] += gap * jn / float(jn @ jn)
        for f in np.flatnonzero(active):
            block = slice(4*f, 4*f+4)
            command[block] = q[block] + np.clip(command[block] - q[block],
                -min(.08, cfg.action_rate_limit), min(.08, cfg.action_rate_limit))
        command = controller.clamp_joint_positions(command).astype(np.float64)
    margins = [float(jn @ (command[4*f:4*f+4] - q[4*f:4*f+4]) - minimum)
               for f, jn, minimum in rows]
    controller.previous_command = command.copy()
    return command, dict(active=active, hard_active=hard_active,
                         correction=command-candidate,
                         min_constraint_margin_m=min(margins, default=0.))
