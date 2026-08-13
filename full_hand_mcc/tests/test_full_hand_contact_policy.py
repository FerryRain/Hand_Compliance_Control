import re
import unittest

import mujoco
import numpy as np

from mjlab.tasks.leaphand.full_hand_mcc_planner_diagnostics import (
    central_difference_clearance_gradient,
    self_separation_ascent_seeds,
)

from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    ARM_COLLISION_GEOM_PATTERN,
    FivePointReachabilitySolver,
)
from mjlab.tasks.leaphand.leaphand_direct_force_env import (
    MCC_NON_TIP_HAND_GEOM_PATTERN,
    direct_force_contact_sensor_cfgs,
)


class FullHandContactPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = FivePointReachabilitySolver()

    def test_fr3_hand_and_tip_geometry_groups_are_disjoint(self) -> None:
        solver = self.solver
        arm_names = {
            solver.model.geom(int(geom_id)).name
            for geom_id in solver.arm_geom_ids
        }
        hand_names = {
            solver.model.geom(int(geom_id)).name
            for geom_id in solver.hand_non_tip_geom_ids
        }
        tip_names = {
            solver.model.geom(int(geom_id)).name
            for geom_id in solver.tip_geom_ids
        }

        self.assertEqual(len(arm_names), 8)
        self.assertEqual(len(hand_names), 13)
        self.assertEqual(len(tip_names), 4)
        self.assertFalse(arm_names & hand_names)
        self.assertFalse(arm_names & tip_names)
        self.assertFalse(hand_names & tip_names)
        self.assertTrue(
            all(ARM_COLLISION_GEOM_PATTERN.fullmatch(name) for name in arm_names)
        )
        self.assertTrue(
            all(
                re.fullmatch(MCC_NON_TIP_HAND_GEOM_PATTERN, name)
                for name in hand_names
            )
        )

    def test_runtime_sensors_separate_strict_arm_depth_and_hand_force(self) -> None:
        sensors = {
            cfg.name: cfg for cfg in direct_force_contact_sensor_cfgs()
        }

        self.assertEqual(sensors["arm_object_collision"].reduce, "mindist")
        self.assertEqual(
            sensors["incidental_hand_object_contact_depth"].reduce,
            "mindist",
        )
        self.assertEqual(
            sensors["incidental_hand_object_contact_force"].reduce,
            "maxforce",
        )
        self.assertNotIn("non_tip_hand_object_collision", sensors)

    def test_seed42_ring_pair_uses_locally_stable_self_clearance_fd(self) -> None:
        # Exact last-feasible q at 45.0 mm from the f744b43 seed42 prefix.
        # At this state a 1e-4-rad negative sample crosses a MuJoCo distance
        # branch and spuriously returns zero.  A 1e-5-rad stencil is also
        # polluted on a task-invariant arm joint after joint-margin clipping.
        # The production 1e-6-rad stencil
        # must instead generate seeds that measurably increase clearance.
        q = np.asarray(
            [
                0.30152822558771303,
                0.5657588407509574,
                -0.2956066584411353,
                -1.7131566033307912,
                -0.04367009081551091,
                1.6545650243831471,
                -0.4498343549933965,
                1.8533426595484093,
                -0.06006810779095744,
                -0.012382935213124505,
                -0.36073795491591654,
                1.486225440376297,
                0.029687221204753433,
                0.2750057773298135,
                -0.07524590616644747,
                1.7882759376683333,
                0.11860867016277592,
                0.08120677735566584,
                -0.3657552129324721,
                0.7145855792544306,
                2.0642836550777326,
                0.4179374171850195,
                0.030918979386621595,
            ],
            dtype=np.float64,
        )
        solver = self.solver
        pair = (
            mujoco.mj_name2id(
                solver.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "mcp_joint_3_geom",
            ),
            mujoco.mj_name2id(
                solver.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "dip_3_geom",
            ),
        )
        pairs = (pair,)
        lower = np.where(
            np.isfinite(solver.lower),
            solver.lower + 1.0e-7 + 0.0005,
            -20.0,
        )
        upper = np.where(
            np.isfinite(solver.upper),
            solver.upper - 1.0e-7 - 0.0005,
            20.0,
        )
        q = np.clip(q, lower, upper)
        source_clearance = float(solver.geometry_pair_distances(q, pairs)[0])

        def sampled_gradient(fd_step_rad: float) -> np.ndarray:
            plus = np.zeros(q.size, dtype=np.float64)
            minus = np.zeros(q.size, dtype=np.float64)
            span = np.zeros(q.size, dtype=np.float64)
            for joint in range(q.size):
                plus_q = q.copy()
                minus_q = q.copy()
                plus_q[joint] = min(
                    plus_q[joint] + fd_step_rad,
                    upper[joint],
                )
                minus_q[joint] = max(
                    minus_q[joint] - fd_step_rad,
                    lower[joint],
                )
                span[joint] = plus_q[joint] - minus_q[joint]
                plus[joint] = solver.geometry_pair_distances(plus_q, pairs)[0]
                minus[joint] = solver.geometry_pair_distances(minus_q, pairs)[0]
            return central_difference_clearance_gradient(plus, minus, span)

        unstable_gradient = sampled_gradient(1.0e-4)
        contaminated_gradient = sampled_gradient(1.0e-5)
        gradient = sampled_gradient(1.0e-6)
        self.assertGreater(float(unstable_gradient[16]), 0.0)
        self.assertLess(float(gradient[16]), 0.0)
        self.assertGreater(
            float(np.linalg.norm(contaminated_gradient)),
            10.0 * float(np.linalg.norm(gradient)),
        )
        seeds = self_separation_ascent_seeds(
            q,
            gradient,
            lower,
            upper,
            maximum_step_rad=0.005,
        )
        self.assertEqual(len(seeds), 2)
        improved = [
            float(solver.geometry_pair_distances(seed, pairs)[0])
            for seed in seeds
        ]
        self.assertGreater(source_clearance, 0.0)
        self.assertTrue(
            all(clearance > source_clearance + 1.0e-6 for clearance in improved),
            (source_clearance, improved),
        )


if __name__ == "__main__":
    unittest.main()
