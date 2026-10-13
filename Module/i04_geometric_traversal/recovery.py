"""Causal shared contact recovery; contact gaps remain measured outcomes."""
from __future__ import annotations

import numpy as np
from Module.e05_physics.scene import FINGERS
from .physics import joint_state, position_jacobian


class ContactEpisodes:
    """Measure every gap; a single transient touch does not close a gap."""
    def __init__(self, confirmation_s):
        self.confirmation_s = confirmation_s
        self.open_start = None
        self.first_touch = None
        self.stable_since = None
        self.episodes = []

    def observe(self, timestamp, safe_contact_count):
        if safe_contact_count == 0:
            if self.open_start is None:
                self.open_start = timestamp
            self.stable_since = None
            return
        if self.open_start is not None:
            if self.first_touch is None:
                self.first_touch = timestamp
            if self.stable_since is None:
                self.stable_since = timestamp
            if timestamp - self.stable_since + 1e-10 >= self.confirmation_s:
                self.episodes.append({'lost_at_s': self.open_start,
                    'first_touch_at_s': self.first_touch, 'stable_recovered_at_s': timestamp,
                    'recovery_time_s': timestamp - self.open_start,
                    'first_touch_delay_s': self.first_touch - self.open_start,
                    'contact_count_after_recovery': int(safe_contact_count), 'recovered': True})
                self.open_start = self.first_touch = self.stable_since = None

    def result(self, terminal_time):
        result = list(self.episodes)
        if self.open_start is not None:
            result.append({'lost_at_s': self.open_start, 'first_touch_at_s': self.first_touch,
                           'recovered': False, 'censored_at_s': terminal_time,
                           'observed_gap_s': terminal_time - self.open_start})
        return result


class SharedContactRecovery:
    """Both MCC cells receive exactly this measured-state recovery layer.

It remembers a stable measured robot state, releases excessive local force,
and returns gradually after persistent total contact loss. No Oracle or
future fingertip target is available to this class.
"""
    def __init__(self, config):
        self.config = config
        self.mode = 'NORMAL'
        self.last_good_q = None
        self.last_good_time_s = None
        self.return_q = None
        self.previous_command = None
        self.mode_before_relief = 'NORMAL'
        self.last_normals = np.zeros((4, 3))
        self.good_since = None
        self.zero_since = None
        self.recovery_started = None
        self.confirmed_since = None
        self.cooldown_until = 0.0
        self.pending_events = []
        self.pause_steps = 0
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.recovery_timeouts = 0
        self.force_release_count = 0
        self.search_offsets_m = np.zeros(4)
        self.actual_active = np.zeros(4, dtype=bool)

    def observe(self, timestamp, measured_q, forces, normals, active, previous_command=None):
        previous = self.mode
        self.previous_command = np.asarray(measured_q if previous_command is None else previous_command).copy()
        self.actual_active = np.asarray(active).copy()
        self.last_normals[active] = normals[active]
        good = int(np.sum(active & (forces <= self.config.force_relief_n)))
        if self.mode == 'NORMAL' and good >= 2 and np.max(forces) <= 6.0:
            self.good_since = timestamp if self.good_since is None else self.good_since
            if timestamp - self.good_since >= self.config.recovery_confirmation_s:
                # Actual actuator reference produced the measured load;
                # measured q alone drops that elastic preload on restoration.
                reference = measured_q if previous_command is None else previous_command
                self.last_good_q = np.asarray(reference).copy()
                self.last_good_time_s = timestamp
        else:
            self.good_since = None
        self.zero_since = (timestamp if self.zero_since is None else self.zero_since) if good == 0 else None
        if np.max(forces) >= self.config.force_relief_n:
            if self.mode != 'FORCE_RELEASE':
                self.mode_before_relief = self.mode
                self.force_release_count += 1
                self.pending_events.append({'time_s': timestamp, 'kind': 'FORCE_RELIEF_START'})
            self.mode = 'FORCE_RELEASE'
            self.confirmed_since = None
        elif self.mode == 'FORCE_RELEASE' and np.max(forces) < 4.0:
            self.mode = self.mode_before_relief
            self.pending_events.append({'time_s': timestamp, 'kind': 'FORCE_RELIEF_END'})
        if (self.config.recovery_enabled and self.mode == 'NORMAL'
                and self.zero_since is not None and timestamp >= self.cooldown_until):
            if (timestamp - self.zero_since >= self.config.contact_loss_debounce_s
                    and self.last_good_q is not None):
                self.mode = 'RECOVER'
                self.recovery_started = timestamp
                self.recovery_attempts += 1
                self.return_q = self.last_good_q.copy()
                self.search_offsets_m[:] = 0
                self.confirmed_since = None
                self.pending_events.append({'time_s': timestamp, 'kind': 'CONTACT_RECOVERY_START',
                    'reference_time_s': self.last_good_time_s,
                    'reference_command_rad': self.return_q.tolist()})
        if self.mode == 'RECOVER':
            if good >= 1:
                self.confirmed_since = timestamp if self.confirmed_since is None else self.confirmed_since
                if timestamp - self.confirmed_since >= self.config.recovery_confirmation_s:
                    self.mode = 'NORMAL'
                    self.recovery_successes += 1
                    self.cooldown_until = timestamp + self.config.recovery_cooldown_s
                    self.pending_events.append({'time_s': timestamp, 'kind': 'CONTACT_RECOVERY_STABLE',
                        'elapsed_s': timestamp - self.recovery_started, 'contact_count': good})
                    self.recovery_started = None
            else:
                self.confirmed_since = None
            if self.mode == 'RECOVER' and timestamp - self.recovery_started >= self.config.recovery_timeout_s:
                self.mode = 'NORMAL'
                self.recovery_timeouts += 1
                self.cooldown_until = timestamp + self.config.recovery_cooldown_s
                self.pending_events.append({'time_s': timestamp, 'kind': 'CONTACT_RECOVERY_TIMEOUT_CONTINUE'})
                self.recovery_started = None
        if self.mode != 'NORMAL':
            self.pause_steps += 1
        return previous != self.mode and self.mode == 'NORMAL'

    def nominal(self, handles, data, default):
        if self.mode == 'NORMAL':
            return np.asarray(default).copy()
        current = joint_state(handles, data)
        if self.mode == 'FORCE_RELEASE' or self.last_good_q is None:
            return current
        command = self.return_q.copy()
        step = self.config.recovery_joint_speed_rad_s * self.config.dt_s
        # Integrate from the actual applied reference. Re-rooting a 2 ms
        # increment at measured q every tick stalls a damped position servo.
        arm_root = self.previous_command[:7]
        arm_delta = np.clip(command[:7] - arm_root, -step, step)
        jp, jr = position_jacobian(handles, data, handles.palm_site_id)
        translation_speed = np.linalg.norm(jp[:, :7] @ arm_delta) / self.config.dt_s
        rotation_speed = np.linalg.norm(jr[:, :7] @ arm_delta) / self.config.dt_s
        scale = min(1.0, self.config.recovery_wrist_speed_m_s / max(translation_speed, 1e-10),
                    self.config.wrist_rotation_speed_rad_s / max(rotation_speed, 1e-10))
        command[:7] = arm_root + scale * arm_delta
        if np.linalg.norm(current[:7] - self.return_q[:7]) < .06:
            for finger in range(4):
                if np.linalg.norm(self.last_normals[finger]) < .5:
                    continue
                if not self.actual_active[finger]:
                    self.search_offsets_m[finger] = min(.003,
                        self.search_offsets_m[finger] + .004 * self.config.dt_s)
                jp, _ = position_jacobian(handles, data, handles.tip_site_ids[finger])
                columns = np.asarray([int(n) for n in FINGERS[finger].joint_names]) + 7
                jf = jp[:, columns]
                inward = -self.last_normals[finger] * self.search_offsets_m[finger]
                dq = jf.T @ np.linalg.solve(jf @ jf.T + 1e-6 * np.eye(3), inward)
                command[columns] += np.clip(dq, -.08, .08)
        return command

    def relieve_force(self, handles, data, command, forces, normals):
        if self.mode != 'FORCE_RELEASE':
            return command
        current = joint_state(handles, data)
        result = command.copy()
        for finger in range(4):
            if forces[finger] < 4.0 or np.linalg.norm(normals[finger]) < .5:
                continue
            jp, _ = position_jacobian(handles, data, handles.tip_site_ids[finger])
            columns = np.asarray([int(n) for n in FINGERS[finger].joint_names]) + 7
            jf = jp[:, columns]
            outward = normals[finger] * self.config.force_relief_speed_m_s * self.config.dt_s
            dq = jf.T @ np.linalg.solve(jf @ jf.T + 1e-6 * np.eye(3), outward)
            result[columns] = current[columns] + dq
        return result
