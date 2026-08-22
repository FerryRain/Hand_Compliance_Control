"""Resultant-wrench/internal-force decomposition in realizable normal basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _array(value: ArrayLike, name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape {shape}")
  return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
  svd_relative_tolerance: float = 1e-4
  damping: float = 1e-8
  transition_blend_steps: int = 20

  def __post_init__(self) -> None:
    if not np.isfinite(self.svd_relative_tolerance) or self.svd_relative_tolerance <= 0.0:
      raise ValueError("svd_relative_tolerance must be finite and positive")
    if not np.isfinite(self.damping) or self.damping < 0.0:
      raise ValueError("damping must be finite and non-negative")
    if self.transition_blend_steps < 1:
      raise ValueError("transition_blend_steps must be at least one")


@dataclass(frozen=True, slots=True)
class CoordinatorOutput:
  active_indices: tuple[int, ...]
  grasp_normal_map: NDArray[np.float64]
  projector_resultant: NDArray[np.float64]
  projector_internal: NDArray[np.float64]
  force_error_n: NDArray[np.float64]
  resultant_force_error_n: NDArray[np.float64]
  internal_force_error_n: NDArray[np.float64]
  desired_hand_wrench_world: NDArray[np.float64]
  measured_hand_wrench_from_contacts_world: NDArray[np.float64]
  hand_wrench_error_from_contacts_world: NDArray[np.float64]
  rank: int
  singular_values: NDArray[np.float64]
  condition_number: float
  internal_wrench_leakage_norm: float
  reconstruction_error_norm: float
  transition_alpha: float


class ContactForceCoordinator:
  """Split scalar normal-force errors using only the actual contact set.

  Surface normals point out of the object.  A positive scalar force means
  compression, so each hand-on-object basis column is ``-normal``.  The hand
  reaction wrench is therefore ``-H lambda``.
  """

  def __init__(self, config: CoordinatorConfig | None = None) -> None:
    self.config = config or CoordinatorConfig()
    self._active_indices: tuple[int, ...] = ()
    self._transition_step = self.config.transition_blend_steps
    self._previous_internal = np.zeros(4, dtype=np.float64)

  def reset(self) -> None:
    self._active_indices = ()
    self._transition_step = self.config.transition_blend_steps
    self._previous_internal[:] = 0.0

  def step(
    self,
    contact_positions_world_m: ArrayLike,
    outward_normals_world: ArrayLike,
    desired_normal_forces_n: ArrayLike,
    measured_normal_forces_n: ArrayLike,
    actual_contact_mask: ArrayLike,
    wrench_reference_world_m: ArrayLike,
    reliability_weights: ArrayLike | None = None,
  ) -> CoordinatorOutput:
    positions = _array(contact_positions_world_m, "contact_positions_world_m", (4, 3))
    normals = _array(outward_normals_world, "outward_normals_world", (4, 3))
    desired = _array(desired_normal_forces_n, "desired_normal_forces_n", (4,))
    measured = _array(measured_normal_forces_n, "measured_normal_forces_n", (4,))
    reference = _array(wrench_reference_world_m, "wrench_reference_world_m", (3,))
    active = np.asarray(actual_contact_mask, dtype=np.bool_)
    if active.shape != (4,):
      raise ValueError("actual_contact_mask must have shape (4,)")
    if np.any(desired < 0.0) or np.any(measured < 0.0):
      raise ValueError("normal force magnitudes must be non-negative")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(lengths - 1.0) > 1e-5):
      raise ValueError("outward_normals_world rows must be unit vectors")
    indices = tuple(int(value) for value in np.flatnonzero(active))
    if not indices:
      raise ValueError("the coordinator requires at least one actual contact")
    active_array = np.asarray(indices, dtype=np.int32)
    if reliability_weights is None:
      weights = np.ones(len(indices), dtype=np.float64)
    else:
      all_weights = _array(reliability_weights, "reliability_weights", (4,))
      weights = all_weights[active_array]
      if np.any(weights <= 0.0):
        raise ValueError("active reliability weights must be positive")

    columns: list[NDArray[np.float64]] = []
    for index in indices:
      hand_on_object_direction = -normals[index]
      moment_arm = positions[index] - reference
      columns.append(
        np.concatenate(
          (hand_on_object_direction, np.cross(moment_arm, hand_on_object_direction))
        )
      )
    normal_map = np.column_stack(columns)
    singular_values = np.linalg.svd(normal_map, compute_uv=False)
    threshold = self.config.svd_relative_tolerance * float(singular_values[0])
    rank = int(np.sum(singular_values > threshold))
    nonzero = singular_values[singular_values > threshold]
    condition = (
      float(nonzero[0] / nonzero[-1]) if len(nonzero) > 1 else 1.0
    )

    weight_inverse = np.diag(1.0 / weights)
    gram = normal_map @ weight_inverse @ normal_map.T
    gram_pinv = np.linalg.pinv(
      gram + self.config.damping * np.eye(6),
      rcond=self.config.svd_relative_tolerance,
    )
    generalized_inverse = weight_inverse @ normal_map.T @ gram_pinv
    p_resultant = generalized_inverse @ normal_map
    p_internal = np.eye(len(indices)) - p_resultant
    active_error = desired[active_array] - measured[active_array]
    resultant_active = p_resultant @ active_error
    internal_active_raw = p_internal @ active_error

    if indices != self._active_indices:
      self._active_indices = indices
      self._transition_step = 0
    self._transition_step = min(
      self._transition_step + 1,
      self.config.transition_blend_steps,
    )
    alpha = self._transition_step / self.config.transition_blend_steps
    raw_full = np.zeros(4, dtype=np.float64)
    raw_full[active_array] = internal_active_raw
    internal_full = (1.0 - alpha) * self._previous_internal + alpha * raw_full
    internal_full[~active] = 0.0
    self._previous_internal[:] = internal_full
    resultant_full = np.zeros(4, dtype=np.float64)
    resultant_full[active_array] = active_error - internal_full[active_array]
    error_full = desired - measured
    error_full[~active] = 0.0

    desired_hand_wrench = -(normal_map @ desired[active_array])
    measured_hand_wrench = -(normal_map @ measured[active_array])
    hand_error = desired_hand_wrench - measured_hand_wrench
    leakage = float(np.linalg.norm(normal_map @ internal_full[active_array]))
    reconstruction = float(
      np.linalg.norm(error_full - resultant_full - internal_full)
    )
    frozen_arrays = [
      normal_map,
      p_resultant,
      p_internal,
      error_full,
      resultant_full,
      internal_full,
      desired_hand_wrench,
      measured_hand_wrench,
      hand_error,
      singular_values,
    ]
    for value in frozen_arrays:
      value.setflags(write=False)
    return CoordinatorOutput(
      active_indices=indices,
      grasp_normal_map=normal_map,
      projector_resultant=p_resultant,
      projector_internal=p_internal,
      force_error_n=error_full,
      resultant_force_error_n=resultant_full,
      internal_force_error_n=internal_full,
      desired_hand_wrench_world=desired_hand_wrench,
      measured_hand_wrench_from_contacts_world=measured_hand_wrench,
      hand_wrench_error_from_contacts_world=hand_error,
      rank=rank,
      singular_values=singular_values,
      condition_number=condition,
      internal_wrench_leakage_norm=leakage,
      reconstruction_error_norm=reconstruction,
      transition_alpha=float(alpha),
    )
