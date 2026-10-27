"""Privileged non-MCC local trajectory repair for dataset generation only."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize

from Module.module_4_finger_dp.contracts import NUM_FINGER_JOINTS


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


def _matrix(value: ArrayLike, name: str, columns: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim != 2 or result.shape[1] != columns or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape (N,{columns})")
  return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class PrivilegedRepairConfig:
  joint_lower_rad: ArrayLike
  joint_upper_rad: ArrayLike
  contact_stiffness_n_per_m: float = 1200.0
  proposal_weight: float = 1.0
  force_weight: float = 4.0
  smoothness_weight: float = 0.5
  max_joint_step_rad: float = 0.03
  joint_margin_rad: float = 0.02
  solver_max_iterations: int = 120
  solver_tolerance: float = 1e-9

  def __post_init__(self) -> None:
    lower = _vector(self.joint_lower_rad, "joint_lower_rad", NUM_FINGER_JOINTS)
    upper = _vector(self.joint_upper_rad, "joint_upper_rad", NUM_FINGER_JOINTS)
    if np.any(lower >= upper):
      raise ValueError("joint bounds must satisfy lower < upper")
    object.__setattr__(self, "joint_lower_rad", lower)
    object.__setattr__(self, "joint_upper_rad", upper)
    for name in (
      "contact_stiffness_n_per_m",
      "proposal_weight",
      "force_weight",
      "smoothness_weight",
      "max_joint_step_rad",
      "joint_margin_rad",
      "solver_tolerance",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.solver_max_iterations < 1:
      raise ValueError("solver_max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class PrivilegedRepairResult:
  repaired_command_rad: NDArray[np.float64]
  predicted_normal_force_n: NDArray[np.float64]
  success: bool
  status: str
  objective: float
  proposal_deviation_norm_rad: float
  force_error_before_n: float
  force_error_after_n: float
  iterations: int
  latency_s: float


class PrivilegedContactRepairOracle:
  """Offline local MPC using GT contact Jacobians and a linear force model.

  This is deliberately not Fingertip MCC: it optimizes a complete command
  horizon jointly and is available only while generating simulator labels.
  Physical replay must still verify every repaired trajectory.
  """

  def __init__(self, config: PrivilegedRepairConfig) -> None:
    self.config = config

  def repair(
    self,
    *,
    current_q_rad: ArrayLike,
    proposal_command_rad: ArrayLike,
    signed_compression_jacobian_m_per_rad: ArrayLike,
    measured_normal_force_n: ArrayLike,
    desired_normal_force_n: ArrayLike,
  ) -> PrivilegedRepairResult:
    start_time = perf_counter()
    current = _vector(current_q_rad, "current_q_rad", NUM_FINGER_JOINTS)
    proposal = _matrix(proposal_command_rad, "proposal_command_rad", NUM_FINGER_JOINTS)
    jacobian = _matrix(
      signed_compression_jacobian_m_per_rad,
      "signed_compression_jacobian_m_per_rad",
      NUM_FINGER_JOINTS,
    )
    measured = _vector(measured_normal_force_n, "measured_normal_force_n", jacobian.shape[0])
    desired = _vector(desired_normal_force_n, "desired_normal_force_n", jacobian.shape[0])
    if np.any(measured < 0.0) or np.any(desired < 0.0):
      raise ValueError("normal force magnitudes must be non-negative")
    horizon = proposal.shape[0]
    if horizon < 1:
      raise ValueError("proposal horizon must be non-empty")
    c = self.config
    lower_joint = c.joint_lower_rad + c.joint_margin_rad
    upper_joint = c.joint_upper_rad - c.joint_margin_rad
    initial = np.empty_like(proposal)
    previous = current.copy()
    for step in range(horizon):
      candidate = np.clip(proposal[step], lower_joint, upper_joint)
      candidate = np.clip(
        candidate,
        previous - c.max_joint_step_rad,
        previous + c.max_joint_step_rad,
      )
      initial[step] = candidate
      previous = candidate

    force_map = c.contact_stiffness_n_per_m * jacobian

    def predicted_force(commands: NDArray[np.float64]) -> NDArray[np.float64]:
      return measured[None, :] + (commands - current[None, :]) @ force_map.T

    def objective(flat: NDArray[np.float64]) -> float:
      commands = flat.reshape(horizon, NUM_FINGER_JOINTS)
      proposal_error = commands - proposal
      force_error = predicted_force(commands) - desired[None, :]
      previous_commands = np.vstack((current[None, :], commands[:-1]))
      smooth = commands - previous_commands
      return float(
        c.proposal_weight * np.sum(np.square(proposal_error))
        + c.force_weight * np.sum(np.square(force_error))
        + c.smoothness_weight * np.sum(np.square(smooth))
      )

    def gradient(flat: NDArray[np.float64]) -> NDArray[np.float64]:
      commands = flat.reshape(horizon, NUM_FINGER_JOINTS)
      grad = 2.0 * c.proposal_weight * (commands - proposal)
      force_error = predicted_force(commands) - desired[None, :]
      grad += 2.0 * c.force_weight * (force_error @ force_map)
      differences = commands - np.vstack((current[None, :], commands[:-1]))
      grad += 2.0 * c.smoothness_weight * differences
      if horizon > 1:
        grad[:-1] -= 2.0 * c.smoothness_weight * differences[1:]
      return grad.ravel()

    dimension = horizon * NUM_FINGER_JOINTS
    rate_matrix = np.zeros((horizon * NUM_FINGER_JOINTS, dimension))
    rate_offset = np.zeros(horizon * NUM_FINGER_JOINTS)
    for step in range(horizon):
      row = slice(step * NUM_FINGER_JOINTS, (step + 1) * NUM_FINGER_JOINTS)
      column = slice(step * NUM_FINGER_JOINTS, (step + 1) * NUM_FINGER_JOINTS)
      rate_matrix[row, column] = np.eye(NUM_FINGER_JOINTS)
      if step == 0:
        rate_offset[row] = current
      else:
        previous_column = slice((step - 1) * NUM_FINGER_JOINTS, step * NUM_FINGER_JOINTS)
        rate_matrix[row, previous_column] = -np.eye(NUM_FINGER_JOINTS)
    lower_rate = rate_offset - c.max_joint_step_rad
    upper_rate = rate_offset + c.max_joint_step_rad
    lower_bounds = np.tile(lower_joint, horizon)
    upper_bounds = np.tile(upper_joint, horizon)
    result = minimize(
      objective,
      initial.ravel(),
      jac=gradient,
      method="SLSQP",
      bounds=Bounds(lower_bounds, upper_bounds),
      constraints=[LinearConstraint(rate_matrix, lower_rate, upper_rate)],
      options={"maxiter": c.solver_max_iterations, "ftol": c.solver_tolerance, "disp": False},
    )
    commands = np.asarray(result.x, dtype=np.float64).reshape(horizon, NUM_FINGER_JOINTS)
    forces = predicted_force(commands)
    before = float(np.linalg.norm(predicted_force(proposal) - desired[None, :]))
    after = float(np.linalg.norm(forces - desired[None, :]))
    commands.setflags(write=False)
    forces.setflags(write=False)
    return PrivilegedRepairResult(
      repaired_command_rad=commands,
      predicted_normal_force_n=forces,
      success=bool(result.success),
      status=str(result.message),
      objective=float(result.fun),
      proposal_deviation_norm_rad=float(np.linalg.norm(commands - proposal)),
      force_error_before_n=before,
      force_error_after_n=after,
      iterations=int(getattr(result, "nit", 0)),
      latency_s=perf_counter() - start_time,
    )
