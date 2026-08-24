"""FR3 pose IK and joint-torque wrist-wrench estimation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.fr3_leap import ARM_HOME_Q, FullRobotHandles


@dataclass(frozen=True, slots=True)
class PalmPoseIKConfig:
  damping: float = 0.02
  gain: float = 0.35
  posture_gain: float = 0.005
  max_joint_step_rad: float = 0.025
  joint_margin_rad: float = 0.03
  orientation_weight_m_per_rad: float = 0.12

  def __post_init__(self) -> None:
    for name, value in (
      ("damping", self.damping),
      ("gain", self.gain),
      ("max_joint_step_rad", self.max_joint_step_rad),
      ("joint_margin_rad", self.joint_margin_rad),
      ("orientation_weight_m_per_rad", self.orientation_weight_m_per_rad),
    ):
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(self.posture_gain) or self.posture_gain < 0.0:
      raise ValueError("posture_gain must be finite and non-negative")


def _quaternion_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  matrix = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)


class PalmPoseIK:
  def __init__(
    self,
    handles: FullRobotHandles,
    config: PalmPoseIKConfig | None = None,
  ) -> None:
    self.handles = handles
    self.config = config or PalmPoseIKConfig()

  def solve(self, data: mujoco.MjData, target_pose_world: ArrayLike) -> NDArray[np.float64]:
    target = np.asarray(target_pose_world, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
      raise ValueError("target_pose_world must be finite with shape (7,)")
    if not np.isclose(np.linalg.norm(target[3:]), 1.0, atol=1e-6):
      raise ValueError("target quaternion must be unit length")
    jac_position = np.zeros((3, self.handles.model.nv), dtype=np.float64)
    jac_rotation = np.zeros((3, self.handles.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
      self.handles.model,
      data,
      jac_position,
      jac_rotation,
      self.handles.palm_site_id,
    )
    current_rotation = data.site_xmat[self.handles.palm_site_id].reshape(3, 3)
    target_rotation = _quaternion_matrix(target[3:])
    orientation_error = 0.5 * sum(
      np.cross(current_rotation[:, axis], target_rotation[:, axis])
      for axis in range(3)
    )
    weight = self.config.orientation_weight_m_per_rad
    dofs = self.handles.arm_dof_adrs
    jacobian = np.vstack(
      (jac_position[:, dofs], weight * jac_rotation[:, dofs])
    )
    error = np.concatenate(
      (
        target[:3] - data.site_xpos[self.handles.palm_site_id],
        weight * orientation_error,
      )
    )
    regularized = jacobian @ jacobian.T + self.config.damping**2 * np.eye(6)
    delta = jacobian.T @ np.linalg.solve(regularized, error)
    current = np.array(data.qpos[self.handles.arm_qpos_adrs], dtype=np.float64)
    delta = self.config.gain * delta + self.config.posture_gain * (ARM_HOME_Q - current)
    delta = np.clip(
      delta,
      -self.config.max_joint_step_rad,
      self.config.max_joint_step_rad,
    )
    command = current + delta
    lower = self.handles.arm_joint_ranges_rad[:, 0] + self.config.joint_margin_rad
    upper = self.handles.arm_joint_ranges_rad[:, 1] - self.config.joint_margin_rad
    return np.clip(command, lower, upper)


@dataclass(frozen=True, slots=True)
class WrenchEstimate:
  wrench_world: NDArray[np.float64]
  joint_external_torque_nm: NDArray[np.float64]
  residual_norm_nm: float
  jacobian_rank: int
  condition_number: float
  source: str = "FR3_JOINT_CONSTRAINT_TORQUE"


class JointTorqueWrenchEstimator:
  """Estimate object-on-hand wrench from FR3 constraint joint torques.

  MuJoCo's ``qfrc_constraint`` isolates constraint/contact generalized forces;
  gravity and actuator forces are not silently folded into this channel.  The
  least-squares residual is logged so a rank/other-contact failure is visible.
  """

  def __init__(self, handles: FullRobotHandles, *, rcond: float = 1e-5) -> None:
    if not np.isfinite(rcond) or rcond <= 0.0:
      raise ValueError("rcond must be finite and positive")
    self.handles = handles
    self.rcond = float(rcond)

  def estimate(self, data: mujoco.MjData) -> WrenchEstimate:
    jac_position = np.zeros((3, self.handles.model.nv), dtype=np.float64)
    jac_rotation = np.zeros((3, self.handles.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
      self.handles.model,
      data,
      jac_position,
      jac_rotation,
      self.handles.palm_site_id,
    )
    jacobian = np.vstack(
      (
        jac_position[:, self.handles.arm_dof_adrs],
        jac_rotation[:, self.handles.arm_dof_adrs],
      )
    )
    torque = np.array(
      data.qfrc_constraint[self.handles.arm_dof_adrs],
      dtype=np.float64,
      copy=True,
    )
    wrench, _, rank, singular_values = np.linalg.lstsq(
      jacobian.T,
      torque,
      rcond=self.rcond,
    )
    residual = float(np.linalg.norm(jacobian.T @ wrench - torque))
    condition = (
      float(singular_values[0] / singular_values[-1])
      if singular_values[-1] > self.rcond * singular_values[0]
      else float("inf")
    )
    wrench.setflags(write=False)
    torque.setflags(write=False)
    return WrenchEstimate(
      wrench_world=wrench,
      joint_external_torque_nm=torque,
      residual_norm_nm=residual,
      jacobian_rank=int(rank),
      condition_number=condition,
    )
