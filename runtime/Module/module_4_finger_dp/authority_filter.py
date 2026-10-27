"""Contact-normal DP action authority filter and opposition metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
import daqp

from Module.module_4_finger_dp.contracts import NUM_FINGER_JOINTS


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


def _limit_vector(value: ArrayLike | float, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim == 0:
    result = np.full(NUM_FINGER_JOINTS, float(result), dtype=np.float64)
  if result.shape != (NUM_FINGER_JOINTS,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite scalar or ({NUM_FINGER_JOINTS},)")
  if np.any(result <= 0.0):
    raise ValueError(f"{name} must be strictly positive")
  return np.array(result, dtype=np.float64, copy=True)


def _contact_matrix(value: ArrayLike, name: str, columns: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim != 2 or result.shape[1] != columns or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape (M,{columns})")
  return np.array(result, dtype=np.float64, copy=True)


def _skew(vector: NDArray[np.float64]) -> NDArray[np.float64]:
  x, y, z = vector
  return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def contact_normal_wrist_map(
  contact_positions_palm_m: ArrayLike,
  outward_normals_palm: ArrayLike,
) -> NDArray[np.float64]:
  """Return ``B_H`` such that ``nu_contact = B_H [v,omega]``."""

  positions = _contact_matrix(contact_positions_palm_m, "contact_positions_palm_m", 3)
  normals = _contact_matrix(outward_normals_palm, "outward_normals_palm", 3)
  if positions.shape != normals.shape:
    raise ValueError("contact positions and normals must have the same shape")
  if len(normals):
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(lengths - 1.0) > 1e-5):
      raise ValueError("active outward normals must be unit vectors")
  result = np.zeros((len(normals), 6), dtype=np.float64)
  for index, (position, normal) in enumerate(zip(positions, normals)):
    result[index, :3] = normal
    result[index, 3:] = -normal @ _skew(position)
  return result


def _orthogonal_range_projector(
  matrix: NDArray[np.float64],
  relative_tolerance: float,
) -> tuple[NDArray[np.float64], int, NDArray[np.float64], float]:
  rows = matrix.shape[0]
  if rows == 0 or matrix.shape[1] == 0 or not np.any(np.abs(matrix) > 0.0):
    return np.zeros((rows, rows)), 0, np.zeros(0), 1.0
  left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
  threshold = relative_tolerance * float(singular[0])
  rank = int(np.sum(singular > threshold))
  if rank == 0:
    return np.zeros((rows, rows)), 0, singular, 1.0
  basis = left[:, :rank]
  projector = basis @ basis.T
  retained = singular[:rank]
  condition = float(retained[0] / retained[-1]) if rank > 1 else 1.0
  return projector, rank, singular, condition


@dataclass(frozen=True, slots=True)
class AuthorityFilterConfig:
  joint_lower_rad: ArrayLike
  joint_upper_rad: ArrayLike
  max_abs_delta_rad: ArrayLike | float = 0.025
  max_velocity_rad_s: ArrayLike | float = 1.5
  max_acceleration_rad_s2: ArrayLike | float = 25.0
  max_seam_rad: ArrayLike | float = 0.012
  objective_weight: ArrayLike | float = 1.0
  collective_limit_m: float = 0.0005
  joint_margin_rad: float = 0.02
  svd_relative_tolerance: float = 1e-6
  solver_max_iterations: int = 40
  solver_tolerance: float = 1e-9
  feasibility_tolerance: float = 1e-7

  def __post_init__(self) -> None:
    lower = _vector(self.joint_lower_rad, "joint_lower_rad", NUM_FINGER_JOINTS)
    upper = _vector(self.joint_upper_rad, "joint_upper_rad", NUM_FINGER_JOINTS)
    if np.any(lower >= upper):
      raise ValueError("joint bounds must satisfy lower < upper")
    object.__setattr__(self, "joint_lower_rad", lower)
    object.__setattr__(self, "joint_upper_rad", upper)
    for name in (
      "max_abs_delta_rad",
      "max_velocity_rad_s",
      "max_acceleration_rad_s2",
      "max_seam_rad",
      "objective_weight",
    ):
      object.__setattr__(self, name, _limit_vector(getattr(self, name), name))
    for name in (
      "collective_limit_m",
      "joint_margin_rad",
      "svd_relative_tolerance",
      "solver_tolerance",
      "feasibility_tolerance",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.solver_max_iterations < 1:
      raise ValueError("solver_max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class AuthorityFilterResult:
  safe_delta_rad: NDArray[np.float64]
  safe_command_rad: NDArray[np.float64]
  wrist_contact_normal_map: NDArray[np.float64]
  compliance_contact_normal_map: NDArray[np.float64]
  collective_projector: NDArray[np.float64]
  nominal_collective_motion_m: NDArray[np.float64]
  safe_collective_motion_m: NDArray[np.float64]
  projector_rank: int
  projector_singular_values: NDArray[np.float64]
  projector_condition_number: float
  intervened: bool
  intervention_norm_rad: float
  solver_success: bool
  solver_status: str
  solver_iterations: int
  maximum_constraint_violation: float
  latency_s: float


class DPActionAuthorityFilter:
  """Project one interpolated DP command into its deterministic authority set."""

  def __init__(self, config: AuthorityFilterConfig) -> None:
    self.config = config

  def _bounds(
    self,
    current_q: NDArray[np.float64],
    previous_command: NDArray[np.float64],
    previous_velocity: NDArray[np.float64],
    dt_s: float,
  ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    c = self.config
    lower = np.maximum(c.joint_lower_rad + c.joint_margin_rad - current_q, -c.max_abs_delta_rad)
    upper = np.minimum(c.joint_upper_rad - c.joint_margin_rad - current_q, c.max_abs_delta_rad)
    previous_delta = previous_command - current_q
    lower = np.maximum(lower, previous_delta - c.max_seam_rad)
    upper = np.minimum(upper, previous_delta + c.max_seam_rad)
    lower = np.maximum(lower, previous_delta - c.max_velocity_rad_s * dt_s)
    upper = np.minimum(upper, previous_delta + c.max_velocity_rad_s * dt_s)
    lower = np.maximum(
      lower,
      previous_delta + dt_s * (previous_velocity - c.max_acceleration_rad_s2 * dt_s),
    )
    upper = np.minimum(
      upper,
      previous_delta + dt_s * (previous_velocity + c.max_acceleration_rad_s2 * dt_s),
    )
    return lower, upper

  def step(
    self,
    *,
    current_q_rad: ArrayLike,
    nominal_delta_rad: ArrayLike,
    previous_executed_command_rad: ArrayLike,
    previous_executed_velocity_rad_s: ArrayLike,
    finger_normal_jacobian_m_per_rad: ArrayLike,
    active_contact_positions_palm_m: ArrayLike,
    active_outward_normals_palm: ArrayLike,
    wrist_compliance_selection: ArrayLike,
    dt_s: float,
    collective_limit_m: float | None = None,
  ) -> AuthorityFilterResult:
    start = perf_counter()
    current_q = _vector(current_q_rad, "current_q_rad", NUM_FINGER_JOINTS)
    nominal = _vector(nominal_delta_rad, "nominal_delta_rad", NUM_FINGER_JOINTS)
    previous_command = _vector(
      previous_executed_command_rad,
      "previous_executed_command_rad",
      NUM_FINGER_JOINTS,
    )
    previous_velocity = _vector(
      previous_executed_velocity_rad_s,
      "previous_executed_velocity_rad_s",
      NUM_FINGER_JOINTS,
    )
    jacobian = _contact_matrix(
      finger_normal_jacobian_m_per_rad,
      "finger_normal_jacobian_m_per_rad",
      NUM_FINGER_JOINTS,
    )
    positions = _contact_matrix(
      active_contact_positions_palm_m,
      "active_contact_positions_palm_m",
      3,
    )
    normals = _contact_matrix(
      active_outward_normals_palm,
      "active_outward_normals_palm",
      3,
    )
    if jacobian.shape[0] != positions.shape[0] or positions.shape != normals.shape:
      raise ValueError("active-contact arrays must share the same row count")
    selection = np.asarray(wrist_compliance_selection, dtype=np.float64)
    if selection.ndim != 2 or selection.shape[0] != 6 or not np.all(np.isfinite(selection)):
      raise ValueError("wrist_compliance_selection must be finite with shape (6,R)")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
      raise ValueError("dt_s must be finite and positive")
    collective_limit = (
      self.config.collective_limit_m
      if collective_limit_m is None
      else float(collective_limit_m)
    )
    if not np.isfinite(collective_limit) or collective_limit <= 0.0:
      raise ValueError("collective_limit_m must be finite and positive")

    wrist_map = contact_normal_wrist_map(positions, normals)
    compliance_map = wrist_map @ selection
    projector, rank, singular, condition = _orthogonal_range_projector(
      compliance_map,
      self.config.svd_relative_tolerance,
    )
    collective_operator = projector @ jacobian
    lower, upper = self._bounds(current_q, previous_command, previous_velocity, dt_s)
    previous_delta = previous_command - current_q
    bounds_feasible = bool(np.all(lower <= upper + self.config.feasibility_tolerance))
    nominal_bounded = np.clip(nominal, np.minimum(lower, upper), np.maximum(lower, upper))

    def collective_violation(value: NDArray[np.float64]) -> float:
      if collective_operator.size == 0:
        return 0.0
      # Authority applies to newly issued motion, not to the servo's existing
      # position-target preload.  Removing the latter would actively unload a
      # stable contact whenever measured q lags the position command.
      command_increment = value - previous_delta
      return float(
        np.max(
          np.maximum(
            np.abs(collective_operator @ command_increment) - collective_limit,
            0.0,
          )
        )
      )

    nominal_is_feasible = (
      bounds_feasible
      and np.all(nominal >= lower - self.config.feasibility_tolerance)
      and np.all(nominal <= upper + self.config.feasibility_tolerance)
      and collective_violation(nominal) <= self.config.feasibility_tolerance
    )
    success = nominal_is_feasible
    status = "NOMINAL_FEASIBLE" if success else "NOT_SOLVED"
    iterations = 0
    safe = nominal.copy() if success else nominal_bounded

    # Projection onto a box with diagonal weights is component-wise clipping.
    # If that exact box projection also satisfies the collective inequalities,
    # it is already the global QP solution; invoking SLSQP only adds numerical
    # failure modes at contact-set transitions.
    bounded_nominal_is_feasible = (
      bounds_feasible
      and collective_violation(nominal_bounded) <= self.config.feasibility_tolerance
    )
    if not success and bounded_nominal_is_feasible:
      success = True
      status = "BOUNDED_NOMINAL_FEASIBLE"

    if not success and bounds_feasible:
      weights_squared = np.square(self.config.objective_weight)
      quadratic = np.diag(weights_squared)
      linear = -weights_squared * nominal
      inequality_matrix = None
      inequality_upper = None
      if collective_operator.size:
        previous_collective = collective_operator @ previous_delta
        inequality_matrix = np.vstack(
          (collective_operator, -collective_operator)
        )
        inequality_upper = np.concatenate(
          (
            previous_collective + collective_limit,
            -previous_collective + collective_limit,
          )
        )
      # DAQP is a deterministic active-set solver for this small convex QP.
      # Unlike generic nonlinear optimizers it handles the redundant rows of
      # a contact-space projector without a singular-Jacobian fallback.
      constraint_matrix = (
        np.zeros((0, NUM_FINGER_JOINTS), dtype=np.float64)
        if inequality_matrix is None
        else inequality_matrix
      )
      inequality_count = constraint_matrix.shape[0]
      daqp_upper = np.concatenate(
        (
          upper,
          np.zeros(0, dtype=np.float64)
          if inequality_upper is None
          else inequality_upper,
        )
      )
      daqp_lower = np.concatenate(
        (lower, np.full(inequality_count, -1e30, dtype=np.float64))
      )
      daqp_sense = np.zeros(
        NUM_FINGER_JOINTS + inequality_count,
        dtype=np.int32,
      )
      try:
        candidate, _, exit_flag, solver_info = daqp.solve(
          quadratic,
          linear,
          constraint_matrix,
          daqp_upper,
          daqp_lower,
          daqp_sense,
          primal_start=nominal_bounded,
          iter_limit=self.config.solver_max_iterations,
          primal_tol=self.config.feasibility_tolerance,
          dual_tol=self.config.solver_tolerance,
        )
      except Exception as error:  # pragma: no cover - backend failure path
        candidate = np.zeros(NUM_FINGER_JOINTS, dtype=np.float64)
        exit_flag = -100
        solver_info = {}
        status = f"QP_BACKEND_ERROR:{type(error).__name__}"
      iterations = int(solver_info.get("iterations", 0))
      if exit_flag > 0:
        candidate = np.asarray(candidate, dtype=np.float64)
        candidate_violation = max(
          float(np.max(np.maximum(lower - candidate, 0.0))),
          float(np.max(np.maximum(candidate - upper, 0.0))),
          collective_violation(candidate),
        )
        if candidate_violation <= self.config.feasibility_tolerance:
          safe = candidate
          success = True
          status = "QP_SOLVED:DAQP"
        else:
          status = "QP_FAILED:DAQP_CONSTRAINT_VIOLATION"
      elif status == "NOT_SOLVED":
        status = f"QP_FAILED:DAQP_EXIT_{exit_flag}"

    if not success:
      # Fail closed by retaining the previous position target.  This has zero
      # newly commanded motion and, unlike measured-q hold, does not silently
      # release an existing contact preload.
      safe = np.minimum(
        np.maximum(previous_delta, np.minimum(lower, upper)),
        np.maximum(lower, upper),
      )
      if collective_violation(safe) > self.config.feasibility_tolerance:
        safe[:] = 0.0
      status = "INFEASIBLE_SAFE_HOLD" if not bounds_feasible else status

    lower_violation = float(np.max(np.maximum(lower - safe, 0.0)))
    upper_violation = float(np.max(np.maximum(safe - upper, 0.0)))
    maximum_violation = max(lower_violation, upper_violation, collective_violation(safe))
    nominal_collective = collective_operator @ (nominal - previous_delta)
    safe_collective = collective_operator @ (safe - previous_delta)
    intervention = float(np.linalg.norm(safe - nominal))
    safe_command = np.asarray(current_q + safe)
    frozen = [
      safe,
      safe_command,
      wrist_map,
      compliance_map,
      projector,
      nominal_collective,
      safe_collective,
      singular,
    ]
    for value in frozen:
      value.setflags(write=False)
    return AuthorityFilterResult(
      safe_delta_rad=safe,
      safe_command_rad=safe_command,
      wrist_contact_normal_map=wrist_map,
      compliance_contact_normal_map=compliance_map,
      collective_projector=projector,
      nominal_collective_motion_m=nominal_collective,
      safe_collective_motion_m=safe_collective,
      projector_rank=rank,
      projector_singular_values=singular,
      projector_condition_number=condition,
      intervened=intervention > 1e-10,
      intervention_norm_rad=intervention,
      solver_success=success,
      solver_status=status,
      solver_iterations=iterations,
      maximum_constraint_violation=maximum_violation,
      latency_s=perf_counter() - start,
    )


@dataclass(frozen=True, slots=True)
class OppositionMetrics:
  opposition_rate: float
  opposition_energy: float
  valid_frame_count: int
  conflict_frame_count: int


def opposition_metrics(
  finger_collective_velocity: ArrayLike,
  wrist_collective_velocity: ArrayLike,
  *,
  dt_s: float,
  finger_norm_threshold: float,
  wrist_norm_threshold: float,
  negative_dot_threshold: float = 0.0,
) -> OppositionMetrics:
  finger = np.asarray(finger_collective_velocity, dtype=np.float64)
  wrist = np.asarray(wrist_collective_velocity, dtype=np.float64)
  if finger.ndim != 2 or wrist.shape != finger.shape or not np.all(np.isfinite(finger)) or not np.all(np.isfinite(wrist)):
    raise ValueError("finger and wrist velocities must be finite with equal shape (T,M)")
  for name, value in {
    "dt_s": dt_s,
    "finger_norm_threshold": finger_norm_threshold,
    "wrist_norm_threshold": wrist_norm_threshold,
  }.items():
    if not np.isfinite(value) or value <= 0.0:
      raise ValueError(f"{name} must be finite and positive")
  if not np.isfinite(negative_dot_threshold) or negative_dot_threshold < 0.0:
    raise ValueError("negative_dot_threshold must be finite and non-negative")
  finger_norm = np.linalg.norm(finger, axis=1)
  wrist_norm = np.linalg.norm(wrist, axis=1)
  valid = (finger_norm > finger_norm_threshold) & (wrist_norm > wrist_norm_threshold)
  dot = np.einsum("ti,ti->t", finger, wrist)
  conflict = valid & (dot < -negative_dot_threshold)
  valid_count = int(np.count_nonzero(valid))
  conflict_count = int(np.count_nonzero(conflict))
  rate = float(conflict_count / valid_count) if valid_count else 0.0
  duration = max(len(dot) * dt_s, dt_s)
  energy = float(np.sum(np.maximum(-dot, 0.0)) * dt_s / duration)
  return OppositionMetrics(rate, energy, valid_count, conflict_count)
