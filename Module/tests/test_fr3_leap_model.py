from __future__ import annotations

import unittest

import mujoco
import numpy as np

from Module.e05_physics.scene import Q_NOMINAL
from Module.fr3_leap import (
  ARM_HOME_Q,
  FullRobotModelConfig,
  build_full_robot,
  model_audit,
)
from Module.module_1_oracle_surface_model import (
  FullRobotGeometryAdapter,
  OracleSurfaceModel,
  Plane,
)


class FR3LeapModelTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.handles = build_full_robot(
      FullRobotModelConfig(surface="plane", gravity_m_s2=0.0)
    )

  def data_at_home(self) -> mujoco.MjData:
    data = mujoco.MjData(self.handles.model)
    data.qpos[self.handles.arm_qpos_adrs] = ARM_HOME_Q
    data.qpos[self.handles.hand_qpos_adrs] = Q_NOMINAL
    data.ctrl[self.handles.arm_actuator_ids] = ARM_HOME_Q
    data.ctrl[self.handles.hand_actuator_ids] = Q_NOMINAL
    mujoco.mj_forward(self.handles.model, data)
    return data

  def test_plant_is_exactly_seven_plus_sixteen_actuated_dofs(self) -> None:
    model = self.handles.model
    self.assertEqual((model.nq, model.nv, model.nu), (23, 23, 23))
    self.assertEqual(len(self.handles.arm_actuator_ids), 7)
    self.assertEqual(len(self.handles.hand_actuator_ids), 16)
    self.assertEqual(model.body_mocapid[self.handles.object_body_id], -1)
    self.assertEqual(model.nmocap, 0)

  def test_physical_belly_pads_are_tip_children_and_face_down(self) -> None:
    audit = model_audit(self.handles)
    self.assertTrue(audit["all_pads_face_down"])
    self.assertEqual(
      audit["pad_parent_body_names"],
      ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"],
    )

  def test_m01_adapter_uses_live_arm_state_and_physics_narrow_phase(self) -> None:
    data = self.data_at_home()
    adapter = FullRobotGeometryAdapter(self.handles)
    capsules_before = adapter.world_capsules(data)
    data.qpos[self.handles.arm_qpos_adrs[0]] += 0.15
    mujoco.mj_forward(self.handles.model, data)
    capsules_after = adapter.world_capsules(data)
    self.assertFalse(np.allclose(capsules_before[-1].end, capsules_after[-1].end))
    top_z = float(self.handles.object_position_m[2] + 0.01)
    oracle = OracleSurfaceModel(
      Plane([0.0, 0.0, top_z], [0.0, 0.0, 1.0]),
      version="fr3-plane-v1",
    )
    result = adapter.query_oracle_clearance(oracle, data)
    self.assertTrue(np.isfinite(result.clearance))
    distance, witness = adapter.physics_pad_object_distance(data)
    self.assertTrue(np.isfinite(distance))
    self.assertEqual(witness.shape, (6,))

  def test_home_hold_is_dynamically_finite(self) -> None:
    data = self.data_at_home()
    for _ in range(50):
      mujoco.mj_step(self.handles.model, data)
    self.assertTrue(np.all(np.isfinite(data.qpos)))
    self.assertLess(np.max(np.abs(data.qpos[self.handles.arm_qpos_adrs] - ARM_HOME_Q)), 1e-9)


if __name__ == "__main__":
  unittest.main()
