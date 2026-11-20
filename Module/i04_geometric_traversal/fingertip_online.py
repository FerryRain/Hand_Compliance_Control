"""Generate wrist commands for Finger DP and solve fingers online on a prescribed wrist."""
from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
from time import perf_counter

import mujoco
import numpy as np

from .belly_bounded_step import bounded_belly_step
from .fingertip_continuous import ContinuousJointPlanner, ContinuousTraversal
from .fingertip_dp_compare import CONTROL_STEPS, LOOKAHEAD_STEPS, TRACK_PHASES, control_pacer, policy_observation
from .fingertip_hand import DT
from .fingertip_plan import Posture
from .fingertip_traversal import Traversal, latency_ms, write_json
from .physics import rotation


class WristPreview(ContinuousTraversal):
    """Use the accepted physical preview to advance the wrist planner's state."""
    def __init__(self, output, seconds, hand, graph):
        super().__init__(output, seconds, hand=hand, graph=graph)
        self.capture_preview_contacts = True
        self.wrists = []
        self.phases = []

    def execute(self, forecast):
        commands, self.feedback = forecast
        reason = None
        count = 0
        for touch in self.preview_contacts:
            self.metrics.update(touch)
            count += 1
            if self.metrics.dwell >= 75:
                reason = 'FULL_MEASURED_HEAD_SURFACE'
                break
        assert count == len(commands) or reason is not None
        self.wrists.append(np.array([np.r_[p, q] for p, q, _, _ in commands[:count]]))
        self.phases.append(np.array([phase for _, _, _, phase in commands[:count]]))
        mujoco.mj_copyData(self.data, self.h.model, self.preview)
        return reason


class FixedWristIK(ContinuousJointPlanner):
    """Project live head targets toward the future wrist and linearize only 16 fingers."""
    interval = 50

    def __init__(self, hand, graph):
        super().__init__(hand, graph)
        self.lower = np.maximum(hand.limits[:, 0] + .02, self.seed.joints - .305)
        self.upper = np.minimum(hand.limits[:, 1] - .02, self.seed.joints + .305)
        self.brake = np.eye(16) * .025
        self.timing = []
        self.start = self.target = self.seed.joints.copy()

    def command(self, i, live, touch, palm):
        if i % self.interval == 0:
            start = perf_counter()
            self.start = self.target
            self.target = self.fit_fingers(live, touch, palm[min(i + self.interval - 1, len(palm) - 1)])
            self.timing.append((perf_counter() - start) * 1000.)
        u = (i % self.interval + 1) / self.interval
        return self.start + u * (self.target - self.start)

    def fit_fingers(self, live, touch, palm):
        position, orientation = palm[:3], rotation(palm[3:])
        root = np.clip(live.qpos[self.h.q], self.lower, self.upper)
        self.set_pose(Posture(position, orientation, root))
        outward = orientation.apply(-self.h.inward_in_palm)
        tips = np.array([self.support(f, outward) for f in range(4)])
        local = rotation(live.qpos[3:7]).inv().apply(touch.points - live.qpos[:3])
        raw = np.where(touch.measured_heads[:, None], position + orientation.apply(local), tips)
        points, normals = self.project(raw)
        state = root.copy()
        for _ in range(4):
            error = [.018 * (state - self.seed.joints), .025 * (state - root)]
            rows = [np.eye(16) * .018, self.brake]
            for f, (point, normal) in enumerate(zip(points, normals)):
                tip = self.support(f, normal)
                body = self.h.tip_bodies[f]
                jp, jr = self.jac(tip, body)
                axis = self.data.xmat[body].reshape(3, 3) @ self.h.head_axes[f]
                metric = .27 * np.eye(3) + .73 * np.outer(normal, normal)
                error.append(metric @ (tip - point + .00055 * normal))
                rows.append(metric @ jp[:, self.h.dof])
                error.append(.010 * (axis + normal))
                rows.append(.010 * np.cross(jr[:, self.h.dof].T, axis).T)
                gap, witness = self.nearest_part(f)
                if witness is not None and gap < .017:
                    physical, direction = witness
                    distal = self.h.fraction(self.data, f, physical) >= .83
                    target_gap = -.00055 if distal else .001
                    jacobian, _ = self.jac(physical, body)
                    error.append(np.array([2.5 * (gap - target_gap)]))
                    rows.append(2.5 * direction[None] @ jacobian[:, self.h.dof])
            delta, info = bounded_belly_step(np.vstack(rows), np.concatenate(error), state,
                self.lower, self.upper, gain=.8, damping=.004, maximum_joint_step=.035)
            if not info['feasible']:
                break
            state = state + delta
            self.set_pose(Posture(position, orientation, state))
            if abs(delta).max() < .0004:
                break
        return np.clip(state, self.start - .065, self.start + .065)


def run_online_trial(branch, output, wrist_file, seed, *, seconds, paced):
    budget = seconds or 9000.
    run = Traversal(output, budget, collect_contacts=True)
    h, d = run.h, run.data
    model = h.model
    wrist_generation_s = 0.
    if branch == 'full-dp':
        started = perf_counter()
        upstream = WristPreview(output / 'wrist_planning', budget, h, run.graph)
        wrist_result = upstream.run()
        if not upstream.wrists:
            raise RuntimeError('The wrist planner produced no commands')
        palm, phases = np.concatenate(upstream.wrists), np.concatenate(upstream.phases)
        np.savez(output / 'generated_wrist.npz', palm_command=palm, phase=phases)
        wrist_generation_s = perf_counter() - started
        print(f'Fresh wrist {len(palm)} steps in {wrist_generation_s:.3f} s ({wrist_result["stop_reason"]})',
              flush=True)
    else:
        if wrist_file is None:
            raise ValueError('Fixed-wrist trials require a common wrist tape')
        with np.load(wrist_file, allow_pickle=False) as tape:
            palm, phases = tape['palm_command'], tape['phase']
    if seconds is not None:
        palm, phases = palm[:run.budget], phases[:run.budget]
    if len(palm) != len(phases) or palm.shape != (len(phases), 7) or not len(palm):
        raise ValueError('Invalid wrist tape')
    wrist_hash = sha256(np.ascontiguousarray(palm, dtype='<f8').tobytes()).hexdigest()
    manifest = json.loads((output / 'manifest.json').read_bytes())
    first_high = None
    longest = 0
    controller = None
    if branch.endswith('dp'):
        from .fingertip_local_policy import LocalFingerPolicy

        controller = LocalFingerPolicy(seed, (len(palm) + CONTROL_STEPS - 1) // CONTROL_STEPS)
        manifest['policy'] = controller.info
        manifest['controller']['dp_inference'] = True
        manifest['wrist_feature_horizon_s'] = LOOKAHEAD_STEPS * DT
        run.planner = None
    else:
        run.planner = FixedWristIK(h, run.graph)
        manifest['planning_variables'] = 16
        manifest['planning'] = dict(mode='fixed_wrist_online_contact_ik', update_s=run.planner.interval * DT)
    manifest.update(branch=branch, wrist_sha256=wrist_hash, wrist_steps=len(palm),
                    sustained_force_behavior='record_and_continue_in_offline_benchmark')
    write_json(output / 'manifest.json', manifest)
    touch = h.measure(d)
    produced_s = []
    with control_pacer(paced) as wait, (controller if controller is not None else nullcontext()):
        started = perf_counter()
        for i, (wrist, phase) in enumerate(zip(palm, phases)):
            control_tick = i % CONTROL_STEPS == 0
            if control_tick:
                wait(started + i * DT)
                outward = rotation(wrist[3:]).apply(-h.inward_in_palm)
                normals = np.broadcast_to(outward, (4, 3))
            if controller is None:
                nominal = run.planner.command(i, d, touch, palm)
            elif control_tick:
                future = palm[min(i + LOOKAHEAD_STEPS - 1, len(palm) - 1)]
                nominal = controller.command(i // CONTROL_STEPS, policy_observation(h, d, future))
            ctrl = run.feedback.command(h, d, nominal, normals, touch, str(phase) in TRACK_PHASES)
            if paced and control_tick:
                produced_s.append(perf_counter() - started)
            d.mocap_pos[h.driver], d.mocap_quat[h.driver], d.ctrl[:] = wrist[:3], wrist[3:], ctrl
            mujoco.mj_step(model, d)
            mujoco.mj_forward(model, d)
            if not np.isfinite(d.qpos).all() or abs(d.time - (i + 1) * DT) > 1.e-6:
                raise RuntimeError(f'Invalid native state at step {i + 1}')
            touch = h.measure(d)
            run.record(touch, wrist, ctrl, phase)
            longest = max(longest, run.metrics.high_steps)
            if first_high is None and run.metrics.high_steps >= 100:
                first_high = dict(step=i + 1, time_s=(i + 1) * DT, region_load_n=touch.loads.tolist())
            if (i + 1) % 10000 == 0:
                print(f'{branch} {(i + 1) * DT:.3f} / {len(palm) * DT:.3f} simulated s', flush=True)
    run.flush()
    cv = run.metrics.coverage
    np.savez_compressed(output / 'coverage.npz', visited_vertices=cv.visited_vertices,
                        visited_goals=cv.visited_goals, required_vertices=cv.graph.required_vertices)
    summary = dict(stop_reason='WRIST_TRAJECTORY_COMPLETE', longest_over_40n_s=longest * DT,
        first_sustained_overload_stop=first_high, live_head_observations=run.contact_tape.save(output),
        **run.metrics.summary())
    if controller is not None:
        summary['control_timing'] = controller.report(output)
    else:
        summary['control_timing'] = dict(fixed_wrist_ik=latency_ms(run.planner.timing))
    write_json(output / 'summary.json', summary)
    extra = dict(wrist_sha256=wrist_hash, wrist_generation_s=wrist_generation_s)
    if paced:
        dt = CONTROL_STEPS * DT
        finish = np.asarray(produced_s)
        gaps = np.diff(finish)
        late = finish - dt * np.arange(len(finish))
        np.savez_compressed(output / 'paced_timing.npz', issued_s=finish, target_s=dt * np.arange(len(finish)))
        extra['paced_control'] = dict(ticks=len(finish), late_ticks=int(np.sum(late > dt)),
            gaps_at_least_20ms=int(np.sum(gaps >= 2 * dt)), max_gap_ms=float(gaps.max(initial=0.) * 1000.),
            output_lateness_ms=latency_ms(late * 1000.))
    return summary, extra
