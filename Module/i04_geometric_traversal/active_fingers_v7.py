"""Independent V6-equivalent candidate with conservative native query pruning.

Only closest-pair lookup is replaced. The two strategy methods below are exact
copies of the frozen V5 orientation-target and V6 finite-step methods; tests
check their AST identity. V6 update/history/limits/force/contact rules are
inherited unchanged. No global function is monkeypatched.
"""
from __future__ import annotations

import numpy as np

from .active_fingers_v6 import ActiveFingerSearchV6, bounded_damped_normal_step
from .full_surface import pad_support_vector
from .native_closest_pair import closest_finger_pair_pruned as closest_finger_pair
from .physics import joint_state, set_joint_state


class ActiveFingerSearchV7(ActiveFingerSearchV6):
    """Same controller, fewer provably irrelevant native distance queries."""

    def _update_orientation_targets(self, data):
        for finger in np.flatnonzero(self._query_mask):
            pair = closest_finger_pair(self.h, data, finger)
            self.target_valid[finger] = False
            self._surface_target_valid[finger] = False
            if pair is None: continue
            geom = int(self.h.tip_geom_ids[finger]);normal = pair['normal_world']
            support = pad_support_vector(self.h.model, geom, data.geom_xmat[geom].reshape(3, 3), normal)
            point = pair['object_point_world_m']
            self.surface_points[finger] = point
            self.surface_face_ids[finger] = -1  # Collision geom, never an original mesh face claim.
            self.closest_gaps[finger] = pair['gap_m']
            self.closest_object_geoms[finger] = pair['object_geom_id']
            self.target_normals[finger] = normal
            self.targets[finger] = point+support+self.config.target_clearance_m*normal
            self.target_valid[finger] = bool(np.linalg.norm(self.targets[finger]-data.geom_xpos[geom])
                                            <= self.config.maximum_target_distance_m)
            self._surface_target_valid[finger] = self.target_valid[finger]
        if np.any(self._query_mask): self.refresh_count += 1

    def _normal_search_step(self, data, finger, center, support, normal,
                            nominal, previous_bias, lower, upper, **bounds):
        delta, diag = bounded_damped_normal_step(center, support, normal,
            nominal, previous_bias, lower, upper, **bounds)
        if not diag['feasible'] or not np.any(delta): return delta, diag
        q = joint_state(self.h, data)
        columns = self.columns[finger]+7
        previous_reference = q.copy(); previous_reference[columns] = nominal+previous_bias
        actual_pair = closest_finger_pair(self.h, data, finger)
        set_joint_state(self.h, self._gap_scratch, previous_reference)
        reference_pair = closest_finger_pair(self.h, self._gap_scratch, finger)
        lo, hi = np.asarray(diag['lower_step_rad']), np.asarray(diag['upper_step_rad'])
        zero_ok = bool(np.all(lo <= 0.) and np.all(hi >= 0.))
        trials = []
        if actual_pair is not None and reference_pair is not None:
            for index in range(self.backtracking_steps):
                scale = .5**index; candidate = delta*scale
                if np.any(candidate < lo-1e-12) or np.any(candidate > hi+1e-12):
                    trials.append(dict(scale=scale, reason='REQUIRED_REPAIR_WOULD_BE_BROKEN'))
                    continue
                proposed = q.copy(); proposed[columns] += candidate
                set_joint_state(self.h, self._gap_scratch, proposed)
                actual_next = closest_finger_pair(self.h, self._gap_scratch, finger)
                proposed = previous_reference.copy(); proposed[columns] += candidate
                set_joint_state(self.h, self._gap_scratch, proposed)
                reference_next = closest_finger_pair(self.h, self._gap_scratch, finger)
                if actual_next is None or reference_next is None: continue
                actual_gain = actual_pair['gap_m']-actual_next['gap_m']
                reference_gain = reference_pair['gap_m']-reference_next['gap_m']
                trials.append(dict(scale=scale, actual_posture_gap_gain_m=actual_gain,
                                   previous_reference_gap_gain_m=reference_gain))
                # Both finite geometry checks must show meaningful progress.
                # Sensor contacts and force guards still decide all success.
                if min(actual_gain, reference_gain) >= self.minimum_gap_improvement_m:
                    center_step, support_step = float(np.linalg.norm(center@candidate)), float(np.linalg.norm(support@candidate))
                    return candidate, dict(diag, reason='DAMPED_NORMAL_FINITE_PROGRESS',
                        optimality_confirmed=False, backtracking_trials=trials, accepted_scale=scale,
                        minimum_gap_improvement_m=self.minimum_gap_improvement_m,
                        center_step_m=center_step, support_step_m=support_step, cartesian_step_m=support_step,
                        charged_step_m=max(center_step, support_step),
                        predicted_gap_change_m=float(normal@(support@candidate)),
                        maximum_joint_step_rad=float(np.max(abs(candidate))))
        if zero_ok:
            return np.zeros(4), dict(diag, reason='NO_FINITE_LOCAL_PROGRESS_HOLD', optimality_confirmed=False,
                backtracking_trials=trials, minimum_gap_improvement_m=self.minimum_gap_improvement_m,
                center_step_m=0., support_step_m=0., charged_step_m=0., predicted_gap_change_m=0.)
        return np.zeros(4), dict(diag, feasible=False, reason='NO_FINITE_PROGRESS_LIMIT_REPAIR',
                                optimality_confirmed=False, backtracking_trials=trials)
