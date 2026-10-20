"""Fresh complete B traversal with restored execution and simulation force grace.

The user's 2026-10-20 request permits simulated force peaks to be tolerated.
Raw forces, original contact/coverage classification, and geometry checks remain
unchanged. This version changes the force STOP rule only: >40N for 200ms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .physics import joint_state, set_control
from .published_whole_hand import CASE, GEOMETRY, SEED, install_asset_adapter
from .whole_hand_adaptive_completion import AdaptiveCompletionPlanner
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_completion import CompletionWholeHandTraversal
from .whole_hand_geometry_cache import CommandGeometryCache
from .whole_hand_recovery import RecoveryWholeHandTraversal


PROTOCOL = 'i04.b.full-mcc.force-grace.v1'
HARD_FORCE_N = 40.
HARD_FORCE_STEPS = 100


class FullBTraversal(CompletionWholeHandTraversal):
    def __init__(self, output, seconds=9000.):
        self.wall_start = perf_counter()
        RecoveryWholeHandTraversal.__init__(self, output, GEOMETRY, SEED, seconds, resume=None)
        self.planner = AdaptiveCompletionPlanner(self.h, self.graph, SEED)
        print('Loading published geometry-only pose seeds; zero inherited coverage', flush=True)
        self.planner.load_cooperative_seeds(CASE)
        self.command_cache = CommandGeometryCache(self.planner.safety)
        self.planner.safety.command_collision_reason = self.command_cache.command_reason
        self.nominal = self.last.copy()
        self.attempts = np.zeros(len(self.graph.vertices_m), int)
        self.stage = 'surface_sweep'
        self.force_warning_steps = 0
        self.force_warning_observations = 0
        self.sweep_stop = None
        manifest = json.loads((self.out/'manifest.json').read_text())
        manifest.update(
            protocol=PROTOCOL, method='B-MCC', cell='B-MCC',
            source_hashes=snapshot_sources(self.out),
            seed_use='Published geometry-only IK pose seeds; no physical state, command history or coverage inherited',
            cooperative_seed_source=str(CASE), coverage_inherited=False, fresh_single_version=True, resume=None,
            force_stop_policy=dict(warning_n=12., stop_n=HARD_FORCE_N,
                                   consecutive_samples=HARD_FORCE_STEPS, duration_s=.2,
                                   scope='Simulation only; explicitly authorized force tolerance'),
            hard_force_n=HARD_FORCE_N, hard_force_duration_s=.2,
            force_warning_only_n=12., coverage_force_exclusion_n=8.,
            coverage_and_contact_predicates_unchanged=True,
            inherited_execution='Original approach, slide, anchor bookkeeping, surface-normal speed cap, and recovery',
            reference_continuity='Target ranking reads existing contact references without reprojecting them',
            scheduler='Surface sweep until 97% or search exhaustion, then adaptive residual completion',
            source_trial='Fresh single fixed version from the common 6mm hover',
        )
        (self.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def tick(self, command, phase, obs=None):
        """Original guarded native tick with an explicit relaxed force stop rule."""
        h, d, p = self.h, self.data, self.planner
        obs = self.measure() if obs is None else obs
        f, safe = obs['f'], obs['safe']
        self.window.append(safe.copy())
        self.coverage.update(obs['bp']-h.object_position_m, obs['bn'], safe)
        self.peak = max(self.peak, float(f.max()))
        self.high_steps = self.high_steps+1 if f.max() > HARD_FORCE_N else 0
        self.force_warning_steps = self.force_warning_steps+1 if f.max() > 12. else 0
        self.force_warning_observations += int(f.max() > 12.)

        def reason(candidate):
            return (p.safety.command_collision_reason(joint_state(h, d), candidate)
                    or p.safety.command_collision_reason(self.last, candidate))
        rejection = reason(command)
        if rejection:
            self.geometry_backtracks += 1
            self.rejections += 1
            accepted = None
            for scale in (.5, .25, .125, 0.):
                trial = self.last+scale*(command-self.last)
                if reason(trial) is None:
                    accepted = trial
                    break
            if accepted is None:
                accepted = p.clearance_exit(joint_state(h, d), self.last)
                if accepted is None or reason(accepted) is not None:
                    raise RuntimeError('NO_GEOMETRY_SAFE_COMMAND:'+rejection)
                phase = 'CLEARANCE_EXIT'
            command = accepted
        else:
            self.rejections = 0
        assert np.all(command >= p.lower) and np.all(command <= p.upper)
        assert np.max(abs(command-self.last)) <= .0024+1e-10
        values = dict(
            time_s=d.time, q_rad=joint_state(h, d), qvel_rad_s=d.qvel.copy(), qacc_warmstart=d.qacc_warmstart.copy(),
            sensor_command_rad=self.last.copy(), command_rad=command.copy(), forces_n=f,
            contact_positions_world_m=obs['p'], normals_world=obs['n'], active=obs['a'],
            belly_forces_n=obs['bf'], belly_contact_positions_world_m=obs['bp'], belly_normals_world=obs['bn'],
            belly_active=obs['ba'], rejected_contact_force_n=obs['rejected'], safe_belly=safe,
            phase=phase, leader_finger=self.anchor, mcc_offset_m=self.offsets.copy(),
            coverage=self.coverage.area_fraction, visited_vertex_count=int(self.coverage.visited_vertices.sum()),
            goal_vertex=self.goal, unsafe_count=len(obs['unsafe']),
            arm_self_force_n=max([item['force_n'] for item in obs['arm']], default=0.),
            actual_penetration_m=obs['penetration'], reference_geometry_backtrack=rejection or '',
            controlled_fingers=np.ones(4, bool), finger_search_active=~safe, simultaneous_safe_contacts=int(safe.sum()),
            target_points_world=self.points.copy(), target_normals_world=self.normal_filter.copy(),
            target_clearance_m=self.clearance.copy(), tangential_seam_bias_m=self.bias.copy(),
            stroke_distance_m=self.route_progress,
            contact_tasks_enabled=phase not in ('NECESSARY_REORIENTATION', 'COOPERATIVE_WITHDRAWAL'),
            force_warning_12n_steps=self.force_warning_steps, force_stop_40n_steps=self.high_steps,
            traversal_stage=self.stage,
        )
        for key, value in values.items():
            self.logs.setdefault(key, []).append(value)
        self.states += 1
        if obs['unsafe'] or obs['arm']:
            raise RuntimeError('UNSAFE_NATIVE_CONTACT')
        if self.high_steps >= HARD_FORCE_STEPS:
            raise RuntimeError('SUSTAINED_DANGEROUS_FORCE')
        self.last = command.copy()
        set_control(h, d, command)
        mujoco.mj_step(h.model, d)
        mujoco.mj_forward(h.model, d)
        self.done_hold = self.done_hold+.002 if self.coverage.visited_vertices.all() and safe.any() else 0.
        if self.states % 10000 == 0:
            self.flush()
        if self.states % 500 == 0 and (self.out/'development_stop.json').exists():
            raise RuntimeError('DEVELOPMENT_STOP_REQUESTED')
        if perf_counter()-self.last_print > 15.:
            result = dict(time_s=d.time, area=self.coverage.area_fraction,
                          vertices=int(self.coverage.visited_vertices.sum()), goals=int(self.coverage.visited_goals.sum()),
                          routes=self.routes, lifts=self.lifts, phase=phase, stage=self.stage,
                          force_n=f.tolist(), safe=safe.tolist(), peak_n=self.peak, goal=self.goal)
            (self.out/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print('PROGRESS', json.dumps(result), flush=True)
            self.last_print = perf_counter()
        if self.stage == 'surface_sweep' and self.coverage.area_fraction >= .97:
            raise RuntimeError('SURFACE_SWEEP_TO_RESIDUAL_COMPLETION')
        return obs

    def run(self):
        # These inherited methods update the routing anchor and keep reference
        # continuity, unlike the failed matched_mcc overrides.
        result = RecoveryWholeHandTraversal.run(self)
        self.sweep_stop = result['stop_reason']
        recoverable = {'SURFACE_SWEEP_TO_RESIDUAL_COMPLETION', 'NO_REACHABLE_COOPERATIVE_ROUTE'}
        if self.stop in recoverable and self.data.time < self.duration:
            self.events.append(dict(kind='SCHEDULER_STAGE_SWITCH', time_s=float(self.data.time),
                                    from_stage=self.stage, to_stage='residual_completion', reason=self.stop,
                                    coverage=self.coverage.area_fraction))
            self.stage = 'residual_completion'
            self.stop = 'UNFINISHED'
            result = CompletionWholeHandTraversal.run(self)
        result.update(protocol=PROTOCOL, method='B-MCC', coverage_inherited=False,
                      fresh_single_version=True, resume=None, wall_elapsed_s=perf_counter()-self.wall_start,
                      surface_sweep_stop=self.sweep_stop, force_warning_observations=self.force_warning_observations,
                      force_releases_scope='Entire fresh single-version trajectory',
                      full_surface_complete=bool(self.done_hold >= .15),
                      strict_original_coverage=True)
        (self.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (self.out/'run_complete.json').write_text(
            json.dumps(dict(protocol=PROTOCOL, stop_reason=self.stop, final_s=float(self.data.time))), encoding='utf-8')
        print('FULL_B_RESULT', json.dumps({key: value for key, value in result.items() if key != 'trace_chunks'}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=9000.)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    install_asset_adapter()
    FullBTraversal(args.output, args.seconds).run()


if __name__ == '__main__':
    main()
