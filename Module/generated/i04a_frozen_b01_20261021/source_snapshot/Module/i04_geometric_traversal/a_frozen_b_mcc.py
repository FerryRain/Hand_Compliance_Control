"""A-MCC ablation under the user's frozen-B supervisory contract.

The original B files are never edited. The private supervisor retains B's
geometry proposals, routing anchor and predicted-contact gates, as explicitly
selected by the user. Only palm poses cross into the A reference constructor.
Consequently this is NOT the old strictly prediction-free A protocol.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .ab_frozen_b_planning import PlanningRecorder
from .b_full_mcc import FullBTraversal
from .contracts import WristTarget
from .matched_mcc import CommonSettings, MatchedMCCTraversal, WristOnlyReference, state_hash
from .natural_belly_traversal import unit
from .physics import joint_state
from .published_whole_hand import ROOT, install_asset_adapter
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_completion import CompletionWholeHandTraversal
from .whole_hand_recovery import RecoveryWholeHandTraversal
from .whole_hand_surface_traversal import WholeHandTraversal


PROTOCOL = 'i04.a.frozen-b-supervisor.mcc.v1'
BASELINE = ROOT/'Module/generated/i04b_full_force_grace01_20261020'


class FrozenBAMCC(FullBTraversal):
    _wrist = MatchedMCCTraversal._wrist
    _publish_wrist = MatchedMCCTraversal._publish_wrist
    _a_transfer = MatchedMCCTraversal._a_transfer

    def __init__(self, output, seconds=9000.):
        # All construction, priors, controllers, graph, geometry and seed data
        # are exactly the successful B constructor, not the failed matched B.
        super().__init__(output, seconds)
        self.cell = 'A-MCC'
        self.settings = CommonSettings(duration_s=seconds, hard_force_n=40., hard_force_s=.2)
        with np.load(BASELINE/'initial.npz') as initial:
            for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
                np.testing.assert_array_equal(getattr(self.data, name), initial[name])
                getattr(self.data, name)[:] = initial[name]
            assert self.data.time == float(initial['time'])
        self.last = joint_state(self.h, self.data)
        self.initial_q = self.last.copy()
        self.oracle_scratch = mujoco.MjData(self.h.model)
        self.a_reference = WristOnlyReference(self.h, self.initial_q, self.planner.lower,
                                             self.planner.upper, self.settings)
        self.reference_index = -1
        self.reference_file = self.out/'nominal_wrist_reference.jsonl'
        self.reference_file.write_text('', encoding='utf-8')
        self.public_wrist = self._wrist(self.initial_q)
        self.method_normals = self.a_reference.pad_reference(self.initial_q)[1]
        self._active_reference = None
        self._cached_key = None
        self._cached_reference = None
        self.recorder = PlanningRecorder(self.out/'planning_calls.jsonl', lambda: float(self.data.time))
        self.recorder.observe_least_squares()
        for name in ('fit_targets', 'fit_patch', 'find_patch', 'group_route', 'path_between'):
            self.recorder.wrap(self.planner, name, label='private_supervisor.'+name)
        self.recorder.wrap(self.a_reference, 'solve', label='A.wrist_IK')
        self.recorder.wrap(self, 'prepare_transfer', label='supervisor_and_A.prepare_transfer')
        self.recorder.wrap(self, 'pivot_route', label='private_supervisor.pivot_route')
        manifest = json.loads((self.out/'manifest.json').read_text())
        baseline_manifest = json.loads((BASELINE/'manifest.json').read_text())
        common_sources = {}
        for name, expected in baseline_manifest['source_hashes'].items():
            actual = sha256((ROOT/name).read_bytes()).hexdigest()
            assert actual == expected, ('FROZEN_B_SOURCE_CHANGED', name)
            common_sources[name] = actual
        manifest.update(
            protocol=PROTOCOL, method='A-MCC', cell='A-MCC',
            comparison_contract='Frozen B supervisor; palm-only A nominal reference; shared MCC executor',
            user_selected_supervisor_metadata=True,
            target_access='WRIST_POSE_ONLY_AT_A_REFERENCE_BOUNDARY',
            supervisor_access='Unchanged B private geometry, anchor, expected_contacts and search rules',
            strict_prediction_free_legacy_A=False,
            nominal_hand_rad=self.a_reference.nominal_hand.tolist(),
            nominal_hand_source='Fixed common initial hand; actual fingers still move under shared MCC',
            shared_nominal_wrist_route=False,
            reference_sources='A robot FK and measured normals; supervisor keeps private geometric references',
            normal_speed_cap='Identical per-finger reference-normal formula, using each method reference',
            frozen_b_case=str(BASELINE), frozen_b_manifest_sha256=sha256((BASELINE/'manifest.json').read_bytes()).hexdigest(),
            initial_state_sha256=state_hash(self.data), frozen_b_source_hashes=common_sources,
            source_hashes=snapshot_sources(self.out),
            planning_instrumentation='Nested wall/process CPU timings and actual least_squares costs; no mathematical changes',
            planning_timing_caveat='Observer overhead and system load can affect wall-clock search budgets',
        )
        (self.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    @contextmanager
    def using_reference(self, private):
        assert self._active_reference is None
        self._active_reference = private
        try:
            yield
        finally:
            self._active_reference = None

    def adapt_nominal(self, private_q, phase, *, root=None):
        wrist = self._wrist(private_q)
        target = self._publish_wrist(wrist, phase)[0]
        own = self.a_reference.solve(target, root=self.last if root is None else root)
        points, normals = self.a_reference.pad_reference(own)
        return own, points, normals

    def contact_step(self, points, normals, nominal, phase):
        # Own robot-FK references feed the unchanged four-finger control kernel.
        # Private surface references are restored for the next supervisor query;
        # they are not allowed to overwrite A's control targets or normal filter.
        private = self._active_reference
        if private is None:
            oracle = (points, normals, nominal)
            key = self._wrist(nominal).tobytes()
            if key != self._cached_key:
                self._cached_reference = self.adapt_nominal(nominal, phase)
                self._cached_key = key
            nominal, points, normals = self._cached_reference
        elif 'distance' in private:
            k = min(max(0, np.searchsorted(private['distance'], self.route_progress)-1), len(private['distance'])-2)
            fraction = (self.route_progress-private['distance'][k])/max(1e-9, private['distance'][k+1]-private['distance'][k])
            oracle = tuple((1-fraction)*private[key][k]+fraction*private[key][k+1]
                           for key in ('points', 'normals', 'q'))
            oracle = (oracle[0], np.asarray([unit(n) for n in oracle[1]]), oracle[2])
        else:
            oracle = private['points'], private['normals'], private['q']
        self.public_wrist = self._wrist(nominal)
        self.method_normals = np.asarray(normals).copy()
        try:
            return WholeHandTraversal.contact_step(self, points, normals, nominal, phase)
        finally:
            self.points, self.normals, self.nominal = [np.asarray(item).copy() for item in oracle]

    def approach(self, patch, clearance=.006, strict_four=False):
        try:
            with self.recorder.call('A.adapt_approach'):
                q, points, normals = self.adapt_nominal(patch['q'], 'APPROACH')
        except RuntimeError as error:
            self.events.append(dict(kind='A_WRIST_APPROACH_REJECTED', time_s=float(self.data.time), reason=str(error)))
            return False
        own = dict(patch, q=q, points=points, normals=normals)
        with self.using_reference(patch):
            return WholeHandTraversal.approach(self, own, clearance, strict_four)

    def slide(self, route, phase='COOPERATIVE_SLIDE'):
        qs, points, normals = [], [], []
        try:
            with self.recorder.call('A.adapt_route', goal=int(route.get('goal', -1))):
                previous = self.last.copy()
                for q in route['q']:
                    own, pp, nn = self.adapt_nominal(q, 'SLIDE', root=previous)
                    qs.append(own); points.append(pp); normals.append(nn); previous = own
        except RuntimeError as error:
            self.events.append(dict(kind='A_WRIST_ROUTE_REJECTED', time_s=float(self.data.time), reason=str(error)))
            return False, 'A_WRIST_IK_UNREACHABLE'
        own = dict(route, q=np.asarray(qs), points=np.asarray(points), normals=np.asarray(normals))
        with self.using_reference(route):
            return WholeHandTraversal.slide(self, own, phase)

    def prepare_transfer(self, target):
        private = RecoveryWholeHandTraversal.prepare_transfer(self, target)
        if private is None:
            return None
        first, middle, hover, clearance = private
        try:
            own_first = self._a_transfer(first, self.last)
            if own_first is None:
                return None
            own_middle = self._a_transfer(middle, own_first.waypoints[-1])
        except RuntimeError as error:
            self.events.append(dict(kind='A_WRIST_TRANSFER_REJECTED', time_s=float(self.data.time), reason=str(error)))
            return None
        if own_middle is None:
            return None
        return own_first, own_middle, hover, clearance

    def refresh_surface(self):
        RecoveryWholeHandTraversal.refresh_surface(self)
        q = self.last.copy(); q[7:] = self.a_reference.nominal_hand
        self.normal_filter = self.a_reference.pad_reference(q)[1]

    def tick(self, command, phase, obs=None):
        obs = self.measure() if obs is None else obs
        source = np.full(4, 'A_ROBOT_NOMINAL', dtype='U32')
        raw = obs['a'] & (np.sum(obs['n']*self.method_normals, axis=1) > .7)
        belly = obs['ba'] & (np.sum(obs['bn']*self.method_normals, axis=1) > .7)
        source[raw] = 'MEASURED_RAW'; source[belly] = 'MEASURED_BELLY'
        extra = dict(public_wrist_pose_world=self.public_wrist.copy(), public_reference_index=self.reference_index,
                     reference_is_robot_nominal=True, input_normal_source=source)
        for key, value in extra.items():
            self.logs.setdefault(key, []).append(value)
        before = self.states
        try:
            return FullBTraversal.tick(self, command, phase, obs)
        except BaseException:
            if self.states == before:
                for key in extra:
                    self.logs[key].pop()
            raise

    def run(self):
        try:
            # Same two unmodified scheduler functions as B. Publish completion
            # once, after the A summary is final, so following audits cannot
            # observe an intermediate sweep summary as the finished trial.
            result = RecoveryWholeHandTraversal.run(self)
            self.sweep_stop = result['stop_reason']
            if self.stop in {'SURFACE_SWEEP_TO_RESIDUAL_COMPLETION', 'NO_REACHABLE_COOPERATIVE_ROUTE'} and self.data.time < self.duration:
                self.events.append(dict(kind='SCHEDULER_STAGE_SWITCH', time_s=float(self.data.time),
                    from_stage=self.stage, to_stage='residual_completion', reason=self.stop,
                    coverage=self.coverage.area_fraction))
                self.stage = 'residual_completion'; self.stop = 'UNFINISHED'
                result = CompletionWholeHandTraversal.run(self)
            result.update(protocol=PROTOCOL, method='A-MCC', frozen_b_supervisor=True,
                          coverage_inherited=False, fresh_single_version=True, resume=None,
                          wall_elapsed_s=perf_counter()-self.wall_start, surface_sweep_stop=self.sweep_stop,
                          force_warning_observations=self.force_warning_observations,
                          force_releases_scope='Entire fresh single-version trajectory',
                          full_surface_complete=bool(self.done_hold >= .15), strict_original_coverage=True,
                          planning_calls=self.recorder.next_id,
                          nominal_wrist_reference_sha256=sha256(self.reference_file.read_bytes()).hexdigest())
            (self.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            (self.out/'run_complete.json').write_text(json.dumps(dict(protocol=PROTOCOL,
                stop_reason=self.stop, final_s=float(self.data.time))), encoding='utf-8')
            return result
        finally:
            self.recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=9000.)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    install_asset_adapter()
    FrozenBAMCC(args.output, args.seconds).run()


if __name__ == '__main__':
    main()
