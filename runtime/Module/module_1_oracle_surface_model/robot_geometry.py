"""M01-FR3 adapter from live MuJoCo state to Oracle link clearances."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.fr3_leap import FullRobotHandles
from Module.module_1_oracle_surface_model.surface_model import (
  CapsuleLink,
  ClearanceResult,
  OracleSurfaceModel,
)


@dataclass(frozen=True, slots=True)
class BodyCapsuleSpec:
  name: str
  start_body: str
  end_body: str
  radius_m: float

  def __post_init__(self) -> None:
    if not self.name or not self.start_body or not self.end_body:
      raise ValueError("capsule names must be non-empty")
    if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
      raise ValueError("radius_m must be finite and positive")


DEFAULT_FR3_CAPSULES = tuple(
  BodyCapsuleSpec(
    name=f"fr3_link_{index}",
    start_body=f"fr3v2_link{index}",
    end_body=f"fr3v2_link{index + 1}",
    radius_m=0.055 if index < 4 else 0.045,
  )
  for index in range(7)
)


class FullRobotGeometryAdapter:
  """Create frame-correct planner capsules from the actual full-robot state."""

  def __init__(
    self,
    handles: FullRobotHandles,
    specs: tuple[BodyCapsuleSpec, ...] = DEFAULT_FR3_CAPSULES,
  ) -> None:
    self.handles = handles
    self.specs = tuple(specs)
    self._body_pairs = tuple(
      (
        mujoco.mj_name2id(handles.model, mujoco.mjtObj.mjOBJ_BODY, spec.start_body),
        mujoco.mj_name2id(handles.model, mujoco.mjtObj.mjOBJ_BODY, spec.end_body),
      )
      for spec in self.specs
    )
    if any(first < 0 or second < 0 for first, second in self._body_pairs):
      raise ValueError("a capsule body is missing from the full-robot model")

  def world_capsules(self, data: mujoco.MjData) -> tuple[CapsuleLink, ...]:
    if data.qpos.shape != (self.handles.model.nq,):
      raise ValueError("MjData does not belong to the configured full-robot model")
    return tuple(
      CapsuleLink(
        start=np.array(data.xpos[first], dtype=np.float64, copy=True),
        end=np.array(data.xpos[second], dtype=np.float64, copy=True),
        radius=spec.radius_m,
        name=spec.name,
      )
      for spec, (first, second) in zip(self.specs, self._body_pairs)
    )

  def query_oracle_clearance(
    self,
    surface: OracleSurfaceModel,
    data: mujoco.MjData,
  ) -> ClearanceResult:
    return surface.query_clearance(self.world_capsules(data))

  def physics_pad_object_distance(
    self,
    data: mujoco.MjData,
    *,
    distance_cap_m: float = 1.0,
  ) -> tuple[float, NDArray[np.float64]]:
    """Return MuJoCo narrow-phase pad/object distance and witness points."""

    if not np.isfinite(distance_cap_m) or distance_cap_m <= 0.0:
      raise ValueError("distance_cap_m must be finite and positive")
    best = float(distance_cap_m)
    witness = np.zeros(6, dtype=np.float64)
    trial = np.zeros(6, dtype=np.float64)
    for geom_id in self.handles.tip_geom_ids:
      distance = float(
        mujoco.mj_geomDistance(
          self.handles.model,
          data,
          int(geom_id),
          self.handles.object_geom_id,
          float(distance_cap_m),
          trial,
        )
      )
      if distance < best:
        best = distance
        witness[:] = trial
    return best, witness
