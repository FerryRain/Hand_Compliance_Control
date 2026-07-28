import re
import unittest

from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    ARM_COLLISION_GEOM_PATTERN,
    FivePointReachabilitySolver,
)
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    MCC_NON_TIP_HAND_GEOM_PATTERN,
    _tip_sensor_cfgs,
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
        sensors = {cfg.name: cfg for cfg in _tip_sensor_cfgs()}

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


if __name__ == "__main__":
    unittest.main()
