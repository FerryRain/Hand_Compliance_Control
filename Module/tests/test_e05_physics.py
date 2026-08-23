from __future__ import annotations

import unittest

import mujoco
import numpy as np

from Module.e05_physics.benchmark import (
  evaluate_contact_handover,
  evaluate_extreme_surface,
  protocol_sha256,
)
from Module.e05_physics.extreme_surface import query_surface
from Module.e05_physics.runner import PhysicsConfig, run_scenario
from Module.e05_physics.scene import (
  FINGERS,
  PAD_HALF_SIZE_M,
  Q_NOMINAL,
  build_scene,
)


class E05PhysicsTest(unittest.TestCase):
  def test_clean_scene_has_only_four_distal_belly_contact_pads(self) -> None:
    handles = build_scene("plane")
    collidable = set(np.flatnonzero(handles.model.geom_contype).tolist())
    expected = set(handles.tip_geom_ids.tolist()) | {handles.object_geom_id}

    self.assertEqual(collidable, expected)
    self.assertEqual(float(handles.object_spec.size[0]), 0.15)
    self.assertEqual(float(handles.object_spec.size[1]), 0.15)
    self.assertTrue(np.allclose(PAD_HALF_SIZE_M, [0.012, 0.008, 0.002]))
    data = mujoco.MjData(handles.model)
    data.qpos[handles.joint_qpos_adrs] = Q_NOMINAL
    mujoco.mj_forward(handles.model, data)
    for finger_index, finger in enumerate(FINGERS):
      body_id = int(handles.tip_body_ids[finger_index])
      geom_id = int(handles.tip_geom_ids[finger_index])
      site_id = int(handles.tip_site_ids[finger_index])
      fsr_body_id = mujoco.mj_name2id(
        handles.model,
        mujoco.mjtObj.mjOBJ_BODY,
        finger.fsr_body,
      )
      pad_outward = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
      design_head_clearance = (
        finger.pad_local_position_m[1]
        - PAD_HALF_SIZE_M[0]
        - finger.distal_head_y_m
      )
      self.assertEqual(int(handles.model.site_bodyid[site_id]), body_id)
      self.assertNotEqual(int(handles.model.site_bodyid[site_id]), fsr_body_id)
      self.assertTrue(
        np.allclose(handles.model.site_pos[site_id], finger.pad_local_position_m)
      )
      self.assertGreaterEqual(design_head_clearance, 0.012)
      self.assertGreater(float(np.dot(pad_outward, [0.0, 0.0, -1.0])), 0.999)
      self.assertEqual(handles.model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    self.assertTrue(np.allclose(Q_NOMINAL[12:16], [0.53481, 1.57006, 0.10087, -0.63505]))
    self.assertTrue(np.allclose(handles.model.opt.gravity, 0.0))
    self.assertEqual(
      mujoco.mj_name2id(
        handles.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "object_body",
      ),
      -1,
    )

  def test_short_physics_run_uses_stable_nonzero_contacts(self) -> None:
    trace, metrics = run_scenario(
      PhysicsConfig(
        scenario="maintenance_translation",
        duration_s=1.2,
        settling_time_s=0.6,
      )
    )

    self.assertTrue(np.all(np.isfinite(trace.joint_positions_rad)))
    self.assertEqual(metrics["contact_continuity_probability"], 1.0)
    self.assertGreaterEqual(metrics["average_contact_count"], 3.98)
    self.assertGreaterEqual(metrics["thumb_contact_probability"], 0.99)
    self.assertGreaterEqual(metrics["contact_distal_head_clearance_min_m"], 0.010)
    self.assertGreater(float(np.max(trace.fingertip_forces_n)), 1.0)
    self.assertLessEqual(metrics["max_tip_force_n"], 8.0)
    self.assertEqual(metrics["non_tip_contact_count"], 0)
    self.assertEqual(metrics["joint_limit_probability"], 0.0)

  def test_handover_confirms_real_replacement_without_zero_contact(self) -> None:
    result, trace = evaluate_contact_handover()

    self.assertTrue(result["thresholds_met"])
    self.assertTrue(result["ordered_contact_sets"])
    self.assertEqual(result["zero_contact_time_s"], 0.0)
    self.assertLessEqual(result["make_recovery_time_s"], 0.25)
    self.assertTrue(np.all(trace.actual_contacts[-250:] == [True, True, False, True]))

  def test_protocol_is_frozen_and_dp_is_not_an_input(self) -> None:
    self.assertEqual(len(protocol_sha256()), 64)
    self.assertNotIn("dp", PhysicsConfig.__dataclass_fields__)

  def test_extreme_hfield_matches_analytic_surface(self) -> None:
    handles = build_scene("extreme")
    data = mujoco.MjData(handles.model)
    data.mocap_pos[handles.object_mocap_id] = handles.object_spec.initial_position
    data.mocap_quat[handles.object_mocap_id] = [1.0, 0.0, 0.0, 0.0]
    handles.model.geom_group[handles.object_geom_id] = 5
    mujoco.mj_forward(handles.model, data)
    geom_group = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    errors = []
    for x_m in (-0.2373, -0.0711, 0.1137, 0.2471):
      for y_m in (-0.3571, -0.2473, -0.0837, 0.1031, 0.1977, 0.3479):
        expected_z = handles.object_spec.initial_position[2] + query_surface(
          x_m,
          y_m,
        )[0]
        ray_origin = np.array(
          [
            handles.object_spec.initial_position[0] + x_m,
            handles.object_spec.initial_position[1] + y_m,
            0.10,
          ]
        )
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
          handles.model,
          data,
          ray_origin,
          np.array([0.0, 0.0, -1.0]),
          geom_group,
          1,
          -1,
          geom_id,
        )
        errors.append((0.10 - distance) - expected_z)
    self.assertLessEqual(float(np.max(np.abs(errors))), 5e-5)

  def test_extreme_surface_recovers_contact_and_reports_unmet_force_thresholds(self) -> None:
    result, _ = evaluate_extreme_surface()
    recovery = result["pose_step_recovery"]

    self.assertFalse(result["thresholds_met"])
    self.assertEqual(result["config"]["duration_s"], 15.0)
    self.assertGreaterEqual(result["continuous_sweep"]["relative_path_length_m"], 0.30)
    # The current finger-heterogeneous relief intentionally lifts individual
    # pads; requiring 95% thumb retention would restore the retired, much
    # easier surface.  Whole-hand continuity and post-step recovery remain the
    # safety properties, while 80% prevents a permanently missing thumb.
    self.assertGreaterEqual(result["continuous_sweep"]["hand_contact_probability"], 0.99)
    self.assertGreaterEqual(result["continuous_sweep"]["thumb_contact_probability"], 0.80)
    self.assertGreaterEqual(
      result["continuous_sweep"]["contact_distal_head_clearance_min_m"],
      0.010,
    )
    self.assertLessEqual(recovery["any_contact_recovery_s"], 0.10)
    self.assertLessEqual(recovery["all_finger_contact_recovery_s"], 0.25)
    self.assertEqual(recovery["final_contact_set"], [1, 2, 3, 4])
    self.assertGreater(result["continuous_sweep"]["max_tip_force_n"], 8.0)
    self.assertTrue(
      recovery["force_settling_s"] is None
      or recovery["force_settling_s"] > 0.75
    )


if __name__ == "__main__":
  unittest.main()
