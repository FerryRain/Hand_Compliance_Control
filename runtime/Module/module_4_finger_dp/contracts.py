"""Causal observation contract for Finger DP controller v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


DP_SCHEMA_VERSION = "fr3-leap-finger-dp.v1"
NUM_FINGERS = 4
JOINTS_PER_FINGER = 4
NUM_FINGER_JOINTS = NUM_FINGERS * JOINTS_PER_FINGER
FORCE_HISTORY_STEPS = 20
WRIST_HISTORY_STEPS = 20
ACTION_HORIZON_STEPS = 20


def _float_array(
  value: ArrayLike,
  name: str,
  shape: tuple[int, ...],
) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
  result = np.array(result, dtype=np.float64, copy=True)
  result.setflags(write=False)
  return result


def _bool_array(
  value: ArrayLike,
  name: str,
  shape: tuple[int, ...],
) -> NDArray[np.bool_]:
  result = np.asarray(value, dtype=np.bool_)
  if result.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
  result = np.array(result, dtype=np.bool_, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class FingerDPObservation:
  """One causal policy observation at a DP replan boundary.

  Geometry is expressed in the current palm frame.  ``geometry_from_contact``
  is the frozen ``m_geom`` flag: true means measured contact geometry; false
  means a SurfaceModel prediction.  ``surface_geometry_valid`` separately
  prevents an invalid prediction from being aliased with a valid free-finger
  query.
  """

  timestamp_s: float
  surface_model_version: str
  finger_q_rad: ArrayLike
  finger_dq_rad_s: ArrayLike
  force_history_normalized: ArrayLike
  contact_history: ArrayLike
  force_valid_history: ArrayLike
  contact_position_palm_m: ArrayLike
  contact_normal_palm: ArrayLike
  surface_distance_m: ArrayLike
  surface_uncertainty_m: ArrayLike
  geometry_from_contact: ArrayLike
  surface_geometry_valid: ArrayLike
  desired_force_n: ArrayLike
  wrist_real_twist_history: ArrayLike
  wrist_mcc_offset_history: ArrayLike
  wrist_mcc_velocity_history: ArrayLike
  future_wrist_plan_twist: ArrayLike
  previous_executed_finger_command_rad: ArrayLike
  force_sample_dt_s: float = 0.01
  policy_dt_s: float = 0.02
  schema_version: ClassVar[str] = DP_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
      raise ValueError("timestamp_s must be finite and non-negative")
    if not self.surface_model_version:
      raise ValueError("surface_model_version must be non-empty")
    for name in ("force_sample_dt_s", "policy_dt_s"):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    float_shapes = {
      "finger_q_rad": (NUM_FINGERS, JOINTS_PER_FINGER),
      "finger_dq_rad_s": (NUM_FINGERS, JOINTS_PER_FINGER),
      "force_history_normalized": (NUM_FINGERS, FORCE_HISTORY_STEPS),
      "contact_position_palm_m": (NUM_FINGERS, 3),
      "contact_normal_palm": (NUM_FINGERS, 3),
      "surface_distance_m": (NUM_FINGERS,),
      "surface_uncertainty_m": (NUM_FINGERS,),
      "desired_force_n": (NUM_FINGERS,),
      "wrist_real_twist_history": (WRIST_HISTORY_STEPS, 6),
      "wrist_mcc_offset_history": (WRIST_HISTORY_STEPS, 6),
      "wrist_mcc_velocity_history": (WRIST_HISTORY_STEPS, 6),
      "future_wrist_plan_twist": (ACTION_HORIZON_STEPS, 6),
      "previous_executed_finger_command_rad": (NUM_FINGER_JOINTS,),
    }
    bool_shapes = {
      "contact_history": (NUM_FINGERS, FORCE_HISTORY_STEPS),
      "force_valid_history": (NUM_FINGERS, FORCE_HISTORY_STEPS),
      "geometry_from_contact": (NUM_FINGERS,),
      "surface_geometry_valid": (NUM_FINGERS,),
    }
    for name, shape in float_shapes.items():
      object.__setattr__(self, name, _float_array(getattr(self, name), name, shape))
    for name, shape in bool_shapes.items():
      object.__setattr__(self, name, _bool_array(getattr(self, name), name, shape))

    if np.any(self.force_history_normalized < 0.0):
      raise ValueError("normalized normal-force magnitudes must be non-negative")
    if np.any(self.surface_uncertainty_m < 0.0):
      raise ValueError("surface_uncertainty_m must be non-negative")
    if np.any(self.desired_force_n < 0.0):
      raise ValueError("desired_force_n must be non-negative")
    if np.any(self.geometry_from_contact & ~self.surface_geometry_valid):
      raise ValueError("measured contact geometry must also be marked valid")

    normals = self.contact_normal_palm
    lengths = np.linalg.norm(normals, axis=1)
    valid = self.surface_geometry_valid
    if np.any(np.abs(lengths[valid] - 1.0) > 1e-5):
      raise ValueError("valid contact_normal_palm rows must be unit vectors")
    if np.any(np.abs(normals[~valid]) > 1e-12):
      raise ValueError("invalid geometry must use a zero normal")

  @property
  def actual_contact_mask(self) -> NDArray[np.bool_]:
    result = np.array(self.contact_history[:, -1], copy=True)
    result.setflags(write=False)
    return result
  @property
  def current_force_valid(self) -> NDArray[np.bool_]:
    result = np.array(self.force_valid_history[:, -1], copy=True)
    result.setflags(write=False)
    return result

  def force_encoder_input(self) -> NDArray[np.float32]:
    """Return ``[finger,time,(force,contact,valid)]`` for the shared TCN."""

    result = np.stack(
      (
        self.force_history_normalized,
        self.contact_history.astype(np.float64),
        self.force_valid_history.astype(np.float64),
      ),
      axis=-1,
    ).astype(np.float32)
    return result

  def per_finger_state_geometry(self) -> NDArray[np.float32]:
    """Return non-temporal per-finger state and geometry features."""

    result = np.concatenate(
      (
        self.finger_q_rad,
        self.finger_dq_rad_s,
        self.contact_position_palm_m,
        self.contact_normal_palm,
        self.surface_distance_m[:, None],
        self.surface_uncertainty_m[:, None],
        self.geometry_from_contact[:, None].astype(np.float64),
        self.surface_geometry_valid[:, None].astype(np.float64),
        self.desired_force_n[:, None],
        self.current_force_valid[:, None].astype(np.float64),
      ),
      axis=1,
    ).astype(np.float32)
    return result
