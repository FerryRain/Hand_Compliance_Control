from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

from Module.fr3_leap import (
  ARM_HOME_Q,
  HAND_NATURAL_Q,
  FullRobotModelConfig,
  build_full_robot,
)
from Module.i01_bunny_physics.runner import I01BunnyConfig, run_i01_bunny
from Module.i01_bunny_physics.surface import canonical_bunny_heightfield


class I01BunnySurfaceTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.bunny = canonical_bunny_heightfield()

  def test_canonical_mesh_and_upper_envelope_are_auditable(self) -> None:
    self.assertEqual(self.bunny.height_m.shape, (181, 181))
    np.testing.assert_allclose(
      self.bunny.extents_m,
      [0.3, 0.2973691672991965, 0.23251266678617022],
      atol=1e-9,
    )
    self.assertEqual(
      self.bunny.source_sha256,
      "7fb5395ff0bdfcab05a61e03748db28556cff2484d2fd6b3c81845a29b8886ef",
    )
    self.assertGreater(self.bunny.coverage_fraction, 0.59)
    self.assertLess(self.bunny.coverage_fraction, 0.62)

  def test_bunny_full_robot_scene_compiles_and_home_footprint_is_valid(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      mesh_path = self.bunny.export_visual_mesh(Path(directory) / "bunny.obj")
      handles = build_full_robot(
        FullRobotModelConfig(
          surface="bunny",
          gravity_m_s2=0.0,
          object_offset_x_m=0.002,
          object_offset_y_m=-0.005,
          object_offset_z_m=-0.003,
          bunny_visual_mesh_path=str(mesh_path),
        )
      )
      self.assertEqual((handles.model.nq, handles.model.nv, handles.model.nu), (23, 23, 23))
      self.assertEqual(handles.model.nhfield, 1)
      data = mujoco.MjData(handles.model)
      data.qpos[handles.arm_qpos_adrs] = ARM_HOME_Q
      data.qpos[handles.hand_qpos_adrs] = HAND_NATURAL_Q
      mujoco.mj_forward(handles.model, data)
      for point in data.site_xpos[handles.tip_site_ids]:
        local = point - handles.object_position_m
        _, _, valid = self.bunny.query(float(local[0]), float(local[1]))
        self.assertTrue(valid)

  def test_short_physics_acquisition_stays_finite(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      mesh_path = self.bunny.export_visual_mesh(Path(directory) / "bunny.obj")
      trace, metrics, _ = run_i01_bunny(
        I01BunnyConfig(
          cell="fixed",
          seed=7,
          duration_s=3.2,
          visual_mesh_path=str(mesh_path),
        )
      )
      self.assertTrue(np.all(np.isfinite(trace.arm_q_rad)))
      self.assertTrue(np.all(np.isfinite(trace.finger_q_rad)))
      self.assertEqual(metrics["authority_violation_count"], 0)
      self.assertEqual(metrics["non_tip_contact_ticks"], 0)


if __name__ == "__main__":
  unittest.main()
