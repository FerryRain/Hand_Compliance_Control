from __future__ import annotations

import unittest

import mujoco
import numpy as np

from Module.e05_physics.scene import Q_NOMINAL
from Module.fr3_leap import ARM_HOME_Q, FullRobotModelConfig, build_full_robot
from Module.module_4_whole_hand_mcc import (
  ContactForceCoordinator,
  CoordinatorConfig,
  JointTorqueWrenchEstimator,
  PalmPoseIK,
  WristMCC,
  WristMCCConfig,
)


class CoordinatorTest(unittest.TestCase):
  def test_resultant_internal_reconstruction_and_leakage(self) -> None:
    coordinator = ContactForceCoordinator(
      CoordinatorConfig(transition_blend_steps=1, damping=0.0)
    )
    positions = np.array(
      [[-0.04, -0.03, 0.0], [0.04, -0.03, 0.0], [-0.04, 0.03, 0.0], [0.04, 0.03, 0.0]]
    )
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    output = coordinator.step(
      positions,
      normals,
      [2.0, 2.0, 2.0, 2.0],
      [1.0, 3.0, 1.0, 3.0],
      [1, 1, 1, 1],
      [0.0, 0.0, 0.0],
    )
    self.assertEqual(output.rank, 3)
    self.assertLess(output.internal_wrench_leakage_norm, 1e-8)
    self.assertLess(output.reconstruction_error_norm, 1e-12)
    self.assertTrue(
      np.allclose(
        output.force_error_n,
        output.resultant_force_error_n + output.internal_force_error_n,
      )
    )

  def test_actual_contact_mask_has_authority(self) -> None:
    coordinator = ContactForceCoordinator(CoordinatorConfig(transition_blend_steps=1))
    positions = np.zeros((4, 3))
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    output = coordinator.step(
      positions,
      normals,
      np.full(4, 2.0),
      np.zeros(4),
      [1, 0, 0, 0],
      np.zeros(3),
    )
    self.assertEqual(output.active_indices, (0,))
    self.assertTrue(np.all(output.force_error_n[1:] == 0.0))


class WristAndRobotControlTest(unittest.TestCase):
  def test_wrist_mcc_moves_into_surface_when_reaction_is_too_low(self) -> None:
    controller = WristMCC(
      WristMCCConfig(
        virtual_mass=np.ones(6),
        damping=np.ones(6),
        stiffness=np.zeros(6),
        dt_s=0.01,
      )
    )
    selection = np.zeros((6, 6))
    selection[2, 2] = 1.0
    command = controller.step(
      [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
      np.zeros(6),
      selection,
    )
    self.assertLess(command.offset[2], 0.0)
    self.assertEqual(command.offset[0], 0.0)

  def test_joint_torque_estimator_recovers_known_world_wrench(self) -> None:
    handles = build_full_robot(FullRobotModelConfig(surface="plane", gravity_m_s2=0.0))
    data = mujoco.MjData(handles.model)
    data.qpos[handles.arm_qpos_adrs] = ARM_HOME_Q
    data.qpos[handles.hand_qpos_adrs] = Q_NOMINAL
    mujoco.mj_forward(handles.model, data)
    jac_p = np.zeros((3, handles.model.nv))
    jac_r = np.zeros((3, handles.model.nv))
    mujoco.mj_jacSite(handles.model, data, jac_p, jac_r, handles.palm_site_id)
    jacobian = np.vstack((jac_p[:, handles.arm_dof_adrs], jac_r[:, handles.arm_dof_adrs]))
    expected = np.array([1.0, -2.0, 6.0, 0.2, -0.1, 0.3])
    data.qfrc_constraint[handles.arm_dof_adrs] = jacobian.T @ expected
    estimate = JointTorqueWrenchEstimator(handles).estimate(data)
    self.assertTrue(np.allclose(estimate.wrench_world, expected, atol=1e-8))
    self.assertLess(estimate.residual_norm_nm, 1e-9)

  def test_palm_pose_ik_reduces_small_position_error(self) -> None:
    handles = build_full_robot(FullRobotModelConfig(surface="plane", gravity_m_s2=0.0))
    data = mujoco.MjData(handles.model)
    data.qpos[handles.arm_qpos_adrs] = ARM_HOME_Q
    data.qpos[handles.hand_qpos_adrs] = Q_NOMINAL
    mujoco.mj_forward(handles.model, data)
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[handles.palm_site_id])
    target = np.concatenate((data.site_xpos[handles.palm_site_id] + [0.005, 0.0, 0.0], quaternion))
    initial = np.linalg.norm(target[:3] - data.site_xpos[handles.palm_site_id])
    solver = PalmPoseIK(handles)
    for _ in range(20):
      data.qpos[handles.arm_qpos_adrs] = solver.solve(data, target)
      mujoco.mj_forward(handles.model, data)
    final = np.linalg.norm(target[:3] - data.site_xpos[handles.palm_site_id])
    self.assertLess(final, initial * 0.1)


if __name__ == "__main__":
  unittest.main()
