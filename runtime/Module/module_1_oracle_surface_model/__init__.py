"""Ground-truth analytic SurfaceModel implementation."""

from Module.module_1_oracle_surface_model.geometry import (
  AnalyticShape,
  Box,
  Cylinder,
  Plane,
  RoundedBox,
  Sphere,
  SurfaceProjection,
)
from Module.module_1_oracle_surface_model.mesh_surface import MeshScalePolicy, MeshSurface
from Module.module_1_oracle_surface_model.surface_model import (
  CapsuleLink,
  ClearanceResult,
  ContactCandidate,
  ContactCandidateRequest,
  OracleSurfaceModel,
)
from Module.module_1_oracle_surface_model.robot_geometry import (
  DEFAULT_FR3_CAPSULES,
  BodyCapsuleSpec,
  FullRobotGeometryAdapter,
)

__all__ = [
  "AnalyticShape",
  "Box",
  "CapsuleLink",
  "ClearanceResult",
  "ContactCandidate",
  "ContactCandidateRequest",
  "Cylinder",
  "MeshSurface",
  "MeshScalePolicy",
  "OracleSurfaceModel",
  "Plane",
  "RoundedBox",
  "Sphere",
  "SurfaceProjection",
  "BodyCapsuleSpec",
  "DEFAULT_FR3_CAPSULES",
  "FullRobotGeometryAdapter",
]
