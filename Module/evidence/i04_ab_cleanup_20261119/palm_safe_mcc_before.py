"""New B-MCC version with real palm/object collision and planning buffers.

Historical A/B releases remain immutable. The original B is not a valid palm
collision-free baseline: its rigid palm collision masks were zero.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from Module.fr3_leap.model import compile_model_xml
from .b_full_mcc import FullBTraversal
from .fixtures import ObjectPlant
from .natural_belly_traversal import set_kinematics
from .belly_contact import belly_ik_terms
from .palm_collision import enable_palm_collisions, PalmMonitor, PALM_NAMES, QUERY_MARGIN_M
from .physics import joint_state
from .published_whole_hand import install_asset_adapter
from .whole_hand_adaptive_completion import AdaptiveCompletionPlanner
from .whole_hand_belly_probe import snapshot_sources


PROTOCOL = 'i04.b.palm-collision.mcc.v1'
PALM_STOP_CLEARANCE_M = .001


class PalmSafePlanner(AdaptiveCompletionPlanner):
    def _fit_with_buffer(self, root, points, normals, clearance, iterations, tangent):
        """All four pad tasks plus the same body clearance terms as patch IK."""
        cached = None
        value = None
        def terms(q):
            nonlocal cached, value
            if cached is not None and np.array_equal(q, cached):
                return value
            set_kinematics(self.h, self.data, q)
            errors, rows = [], []
            for finger in range(4):
                normal = normals[finger]
                error, jac, _ = belly_ik_terms(self.h.model, self.data, int(self.h.tip_geom_ids[finger]),
                    points[finger], normal, clearance_m=clearance, orientation_weight_m=.04,
                    dof_indices=self.order)
                metric = np.outer(normal, normal) + tangent * (np.eye(3)-np.outer(normal, normal))
                errors.append(np.r_[metric @ error[:3], error[3:]])
                rows.append(np.vstack((metric @ jac[:3], jac[3:])))
            pe, pj = self.posture_terms(q, root)
            ge, gj = self.fixed_buffer_terms(q, margin=QUERY_MARGIN_M)
            value = np.concatenate([*errors, pe, ge]), np.vstack([*rows, pj, gj])
            cached = q.copy()
            return value
        fit = least_squares(lambda q: terms(q)[0], np.clip(root, self.lower+1e-9, self.upper-1e-9),
                            jac=lambda q: terms(q)[1], bounds=(self.lower, self.upper),
                            max_nfev=iterations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        return fit

    def fit(self, root, *, fingers=(0, 1, 2, 3), clearance=.003, outer=6, iterations=120):
        if tuple(fingers) != (0, 1, 2, 3):
            raise ValueError('This planner initializes the complete hand')
        original = np.asarray(root).copy()
        q = np.clip(original, self.lower+1e-9, self.upper-1e-9)
        points, normals = self.project_pads(q)
        evaluations = 0
        for _ in range(outer):
            fit = self._fit_with_buffer(q, points, normals, clearance, iterations, .12)
            q = fit.x
            evaluations += fit.nfev
            points, normals = self.project_pads(q)
        set_kinematics(self.h, self.data, q)
        details = []
        for i in range(4):
            e, _, d = belly_ik_terms(self.h.model, self.data, int(self.h.tip_geom_ids[i]),
                points[i], normals[i], clearance_m=clearance, dof_indices=self.order)
            details.append(dict(finger=i, normal_error_m=float(e[:3] @ normals[i]),
                                angle_deg=d['alignment_angle_deg'], qualified_patch=d['geometry_qualified']))
        reason = self.safety.collision_reason(q)
        accepted = reason is None and all(abs(d['normal_error_m']) < .0008 and
                    d['angle_deg'] < 6. and d['qualified_patch'] for d in details)
        return q, dict(accepted=accepted, collision_reason=reason, fingers=details,
                       evaluations=int(evaluations), palm_buffer_in_initial_fit=True), points, normals

    def fit_targets(self, root, points, normals, *, clearance=.0002, iterations=100):
        fit = self._fit_with_buffer(np.asarray(root), points, normals, clearance, iterations, .35)
        return fit.x, dict(collision_reason=self.safety.collision_reason(fit.x), evaluations=int(fit.nfev))


@contextmanager
def new_run_model_and_planner():
    """Use existing B scheduling with new-run plant/IK, without editing B sources."""
    from . import whole_hand_surface_traversal as traversal
    from . import b_full_mcc
    original_scene = traversal.build_scene
    original_planner = traversal.WholeHandSurfacePlanner
    original_final_planner = b_full_mcc.AdaptiveCompletionPlanner
    def scene(directory):
        h, graph, home, quality = original_scene(directory)
        xml = enable_palm_collisions(h.xml)
        model = compile_model_xml(xml)
        base = replace(h.base, model=model, xml=xml)
        updated = ObjectPlant(base, h.object_geom_ids, h.fixture)
        quality = dict(quality, palm_collision_geometry_enabled=list(PALM_NAMES),
                       palm_collision_backend='Existing six rigid-part mesh/cylinder convex geoms')
        return updated, graph, home, quality
    traversal.build_scene = scene
    traversal.WholeHandSurfacePlanner = PalmSafePlanner
    b_full_mcc.AdaptiveCompletionPlanner = PalmSafePlanner
    try:
        yield
    finally:
        traversal.build_scene = original_scene
        traversal.WholeHandSurfacePlanner = original_planner
        b_full_mcc.AdaptiveCompletionPlanner = original_final_planner


class PalmSafeTraversal(FullBTraversal):
    def __init__(self, output, seconds=9000.):
        self.minimum_palm_clearance = QUERY_MARGIN_M
        self.palm_monitor = None
        with new_run_model_and_planner():
            super().__init__(output, seconds)
        self.palm_monitor = PalmMonitor(self.h.model)
        initial = self.palm_monitor.observe(self.data)
        if initial['clearance_m'] < .003:
            raise RuntimeError('INITIAL_PALM_CLEARANCE')
        manifest = json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol=PROTOCOL, runtime_entrypoint='Module/i04_geometric_traversal/palm_safe_mcc.py',
            original_collision_geometry=False, original_object_collision_geometry=True,
            palm_collision_parts=list(PALM_NAMES), palm_contact_query_margin_m=QUERY_MARGIN_M,
            palm_query_force_activation_distance_m=0., palm_stop_clearance_m=PALM_STOP_CLEARANCE_M,
            palm_planning_buffer_m=QUERY_MARGIN_M, palm_command_buffer_m=.003,
            initial_palm_clearance=initial, source_hashes=snapshot_sources(self.out),
            scope='Fresh palm-collision repair; not the frozen historical A/B comparison')
        (self.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def measure(self):
        obs = super().measure()
        if self.palm_monitor is not None:
            record = self.palm_monitor.observe(self.data)
            obs['palm'] = record
            self.minimum_palm_clearance = min(self.minimum_palm_clearance, record['clearance_m'])
        return obs

    def tick(self, command, phase, obs=None):
        obs = self.measure() if obs is None else obs
        palm = obs['palm']
        if (palm['clearance_m'] < PALM_STOP_CLEARANCE_M
                and not self.permit_protective_stop_exit(command, phase, obs)):
            raise RuntimeError('PALM_CLEARANCE_STOP')
        for key, value in dict(palm_clearance_m=palm['clearance_m'],
                               nearest_palm_geom=palm['nearest_geom']).items():
            self.logs.setdefault(key, []).append(value)
        try:
            return super().tick(command, phase, obs)
        except Exception:
            # The parent may reject a command before committing its state to
            # the ledger. Keep this observation only if that state was logged.
            # A successful parent tick may also have flushed/cleared logs.
            count = len(self.logs.get('time_s', []))
            for key in ('palm_clearance_m', 'nearest_palm_geom'):
                values = self.logs.get(key, [])
                while len(values) > count:
                    values.pop()
                if not values:
                    self.logs.pop(key, None)
            raise

    def permit_protective_stop_exit(self, command, phase, obs):
        """Only a subclass with an independently replayable escape may opt in."""
        return False

    def run(self):
        result = super().run()
        result.update(protocol=PROTOCOL, minimum_palm_clearance_m=self.minimum_palm_clearance,
                      palm_collision_parts=list(PALM_NAMES), historical_b_modified=False)
        (self.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (self.out/'run_complete.json').write_text(json.dumps(dict(protocol=PROTOCOL,
            stop_reason=self.stop, final_s=float(self.data.time))), encoding='utf-8')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=9000.)
    args = parser.parse_args()
    install_asset_adapter()
    PalmSafeTraversal(args.output, args.seconds).run()


if __name__ == '__main__':
    main()
