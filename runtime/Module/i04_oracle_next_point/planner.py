"""State-conditioned Explicit MCC planner used by the I04 physical runner.

M09 supplies locally optimized contact candidates to M11.  M12 filters terminal
dead ends.  The selected edge is then refined with the live nonlinear FR3 model,
audited by M10 and is executable only through M06.  Prediction suffixes are
recorded but never enter the command path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp

from Module.i01_bunny_physics.runner import (
  _finger_ik,
  _local_kinematics,
  _logical_q,
  _pad_support_radius,
  _quaternion_from_matrix,
)
from Module.i01_bunny_physics.surface import BunnyHeightField
from Module.i04_oracle_next_point.surface_graph import BunnySurfaceGraph
from Module.module_1_oracle_surface_model import (
  MeshScalePolicy,
  MeshSurface,
  OracleSurfaceModel,
)
from Module.module_4_whole_hand_mcc.robot_control import PalmPoseIK, PalmPoseIKConfig
from Module.module_6_prefix_executor import (
  PlannedPrefix,
  PrefixSample,
  PrefixSource,
  TransactionType,
)
from Module.module_7_contact_mode_graph import (
  CommitContext,
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)
from Module.module_8_cheap_cert import CheapCert, CheapCertConfig, CheapCertInput
from Module.module_9_continuous_optimize import (
  ContinuousOptimizer,
  OptimizationConfig,
  OptimizationRequest,
  PlannerState,
)
from Module.module_10_exact_prefix_audit import (
  AuditConfig,
  AuditEnvironment,
  AuditRequest,
  ExactPrefixAuditor,
)
from Module.module_11_lazy_beam_search import (
  BeamSearchConfig,
  LazyBeamSearch,
  PlanningCandidate,
  SearchWeights,
)
from Module.module_12_shadow_viability import ShadowViabilityEvaluator


SURFACE_MODEL_VERSION = "oracle-bunny-full-mesh-sdf.i04.v1"


@dataclass(frozen=True, slots=True)
class I04PlannerConfig:
  maximum_translation_per_prefix_m: float = 0.003
  maximum_rotation_per_prefix_rad: float = 0.005
  nominal_translation_speed_m_s: float = 0.010
  nominal_rotation_speed_rad_s: float = 0.050
  minimum_prefix_duration_s: float = 0.30
  prefix_samples: int = 9
  arm_ik_iterations: int = 28
  arm_position_tolerance_m: float = 0.0045
  arm_orientation_tolerance_rad: float = np.deg2rad(4.0)
  minimum_joint_margin_rad: float = 0.008
  audit_subdivisions: int = 3
  beam_horizon: int = 1
  beam_width: int = 8
  per_mode_quota: int = 2
  wrist_adjust_step_m: float = 0.0015
  wrist_adjust_interval_prefixes: int = 3

  def __post_init__(self) -> None:
    for name in (
      "maximum_translation_per_prefix_m",
      "maximum_rotation_per_prefix_rad",
      "nominal_translation_speed_m_s",
      "nominal_rotation_speed_rad_s",
      "minimum_prefix_duration_s",
      "arm_position_tolerance_m",
      "arm_orientation_tolerance_rad",
      "minimum_joint_margin_rad",
      "wrist_adjust_step_m",
    ):
      if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.prefix_samples < 3 or self.arm_ik_iterations < 1:
      raise ValueError("prefix_samples >=3 and positive arm_ik_iterations are required")
    if self.audit_subdivisions < 3:
      raise ValueError("audit_subdivisions must be >=3")
    if self.wrist_adjust_interval_prefixes < 1:
      raise ValueError("wrist_adjust_interval_prefixes must be positive")


@dataclass(frozen=True, slots=True)
class PoseCandidate:
  finger_id: int
  yaw_offset_rad: float
  target_position_m: NDArray[np.float64]
  target_rotation: NDArray[np.float64]
  terminal_arm_q_rad: NDArray[np.float64]
  terminal_tip_position_m: NDArray[np.float64]
  position_error_m: float
  orientation_error_rad: float
  joint_margin_rad: float
  target_tip_distance_m: float
  score: float

  @property
  def feasible(self) -> bool:
    return np.isfinite(self.score)


@dataclass(frozen=True, slots=True)
class I04PlanResult:
  prefix: PlannedPrefix
  certificate: Any
  selected_finger: int | None
  selected_primitive: str
  target_vertex: int
  target_pose_world: NDArray[np.float64]
  evidence: dict[str, Any]


def _whole_hand_roles(
  actual_contact_set: frozenset[int],
  primitive_kind: PrimitiveKind,
  selected_finger: int | None,
) -> dict[str, str]:
  """Expose the complete four-finger reference semantics of one transaction."""

  roles = {
    str(finger): ("ANCHOR" if finger in actual_contact_set else "FREE")
    for finger in range(1, 5)
  }
  if primitive_kind is PrimitiveKind.WRIST_ADJUST:
    return roles
  if selected_finger is None:
    raise ValueError("a finger primitive requires selected_finger")
  if primitive_kind is PrimitiveKind.SLIDE:
    roles[str(selected_finger)] = "EXPLORER"
  elif primitive_kind is PrimitiveKind.MAKE:
    roles[str(selected_finger)] = "REPLACEMENT_MAKE"
  elif primitive_kind is PrimitiveKind.REPOSITION:
    roles[str(selected_finger)] = "REPLACEMENT_STAGING"
  elif primitive_kind is PrimitiveKind.BREAK:
    roles[str(selected_finger)] = "RELEASE"
  else:
    roles[str(selected_finger)] = "PARTICIPANT"
  return roles


def _rotation_angle(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
  relative = left.T @ right
  return float(
    np.arccos(np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0))
  )


def _align_axis(
  source: NDArray[np.float64],
  target: NDArray[np.float64],
) -> NDArray[np.float64]:
  a = np.asarray(source, dtype=np.float64) / np.linalg.norm(source)
  b = np.asarray(target, dtype=np.float64) / np.linalg.norm(target)
  cross = np.cross(a, b)
  sine = float(np.linalg.norm(cross))
  cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
  if sine > 1e-10:
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * cross / sine).as_matrix()
  if cosine > 0.0:
    return np.eye(3)
  basis = np.array([1.0, 0.0, 0.0])
  if abs(float(np.dot(a, basis))) > 0.9:
    basis = np.array([0.0, 1.0, 0.0])
  axis = np.cross(a, basis)
  axis /= np.linalg.norm(axis)
  return Rotation.from_rotvec(np.pi * axis).as_matrix()


def _clip_pose(
  current_position: NDArray[np.float64],
  current_rotation: NDArray[np.float64],
  target_position: NDArray[np.float64],
  target_rotation: NDArray[np.float64],
  config: I04PlannerConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  displacement = target_position - current_position
  distance = float(np.linalg.norm(displacement))
  if distance > config.maximum_translation_per_prefix_m:
    target_position = current_position + (
      config.maximum_translation_per_prefix_m / distance
    ) * displacement
  relative = Rotation.from_matrix(current_rotation.T @ target_rotation)
  vector = relative.as_rotvec()
  angle = float(np.linalg.norm(vector))
  if angle > config.maximum_rotation_per_prefix_rad:
    relative = Rotation.from_rotvec(
      config.maximum_rotation_per_prefix_rad * vector / angle
    )
    target_rotation = current_rotation @ relative.as_matrix()
  return np.asarray(target_position), np.asarray(target_rotation)


class _ExactMuJoCoKinematics:
  """M10 backend that recomputes every fingertip from the audited 23-DoF q."""

  def __init__(self, handles: Any) -> None:
    self.handles = handles
    self.reference_q_rad = np.zeros(23, dtype=np.float64)
    self.joint_lower_rad = np.concatenate(
      (handles.arm_joint_ranges_rad[:, 0], handles.hand_joint_ranges_rad[:, 0])
    )
    self.joint_upper_rad = np.concatenate(
      (handles.arm_joint_ranges_rad[:, 1], handles.hand_joint_ranges_rad[:, 1])
    )
    self.scratch = mujoco.MjData(handles.model)

  def forward(
    self,
    q_rad: ArrayLike,
    _wrist_position_m: ArrayLike,
  ) -> NDArray[np.float64]:
    q = np.asarray(q_rad, dtype=np.float64)
    self.scratch.qpos[self.handles.arm_qpos_adrs] = q[:7]
    self.scratch.qpos[self.handles.hand_qpos_adrs] = q[7:]
    mujoco.mj_forward(self.handles.model, self.scratch)
    return np.array(
      self.scratch.site_xpos[self.handles.tip_site_ids],
      dtype=np.float64,
      copy=True,
    )

  def joint_margin(self, q_rad: ArrayLike) -> float:
    q = np.asarray(q_rad, dtype=np.float64)
    return float(
      np.min(np.minimum(q - self.joint_lower_rad, self.joint_upper_rad - q))
    )

  def self_collision_clearance(self, fingertip_positions_m: ArrayLike) -> float:
    tips = np.asarray(fingertip_positions_m, dtype=np.float64)
    distances = [
      float(np.linalg.norm(tips[left] - tips[right]))
      for left in range(4)
      for right in range(left + 1, 4)
    ]
    return min(distances) - 0.012


class ExplicitI04Planner:
  def __init__(
    self,
    handles: Any,
    bunny: BunnyHeightField,
    surface_graph: BunnySurfaceGraph,
    config: I04PlannerConfig | None = None,
  ) -> None:
    self.handles = handles
    self.bunny = bunny
    self.surface_graph = surface_graph
    self.config = config or I04PlannerConfig()
    self.graph = ContactModeGraph()
    mesh = bunny.mesh.copy()
    mesh.apply_translation(handles.object_position_m)
    self.surface_model = OracleSurfaceModel(
      MeshSurface(
        mesh,
        source_path=bunny.source_path,
        source_up_axis="y",
        scale_policy=MeshScalePolicy(0.30, 0.18),
        scale_factor=1.0,
      ),
      version=SURFACE_MODEL_VERSION,
    )
    self._exact_kinematics = _ExactMuJoCoKinematics(handles)
    self._link_clearance = self._make_sdf_link_clearance()
    self._pose_ik = PalmPoseIK(
      handles,
      PalmPoseIKConfig(
        gain=0.78,
        damping=0.012,
        posture_gain=0.0,
        max_joint_step_rad=0.08,
        joint_margin_rad=0.018,
        orientation_weight_m_per_rad=0.18,
      ),
    )

  def _make_sdf_link_clearance(self) -> Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    float,
  ]:
    """Return an exact SDF no-penetration audit for FR3 collision links.

    The execution plant intentionally permits only fingertip-pad/object
    contacts.  A private audit model additionally enables object collision for
    FR3 link geoms, making MuJoCo's own non-convex SDF narrow phase the M10
    rejection oracle.  Positive separation has no signed-distance value in
    MuJoCo's public SDF pair query, so the function returns a conservative
    +1 mm sentinel when no collision contact exists and the actual negative
    contact distance when penetration is present.
    """

    model = mujoco.MjModel.from_xml_string(self.handles.xml)
    scratch = mujoco.MjData(model)
    object_geom = mujoco.mj_name2id(
      model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "fr3_e05_object_geom",
    )
    link_geoms = {
      mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"fr3v2_link{index}_collision",
      )
      for index in range(8)
    }
    link_geoms.discard(-1)
    # Object contype=2/conaffinity=1 and arm links contype=4/conaffinity=4 in
    # the plant.  Enabling bit 4 only in this private model exposes link/SDF
    # overlaps without altering execution dynamics.
    model.geom_conaffinity[object_geom] = int(
      model.geom_conaffinity[object_geom]
    ) | 4

    def clearance(
      q: NDArray[np.float64],
      _wrist: NDArray[np.float64],
      _tips: NDArray[np.float64],
    ) -> float:
      scratch.qpos[self.handles.arm_qpos_adrs] = q[:7]
      scratch.qpos[self.handles.hand_qpos_adrs] = q[7:]
      mujoco.mj_forward(model, scratch)
      minimum = 0.001
      for contact_index in range(scratch.ncon):
        contact = scratch.contact[contact_index]
        geom_1 = int(contact.geom1)
        geom_2 = int(contact.geom2)
        if object_geom not in (geom_1, geom_2):
          continue
        other = geom_2 if geom_1 == object_geom else geom_1
        if other in link_geoms:
          minimum = min(minimum, float(contact.dist))
      return minimum

    return clearance

  def _solve_arm_pose(
    self,
    root_q: NDArray[np.float64],
    target_position: NDArray[np.float64],
    target_rotation: NDArray[np.float64],
    *,
    iterations: int | None = None,
  ) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
  ]:
    scratch = mujoco.MjData(self.handles.model)
    scratch.qpos[self.handles.arm_qpos_adrs] = root_q[:7]
    scratch.qpos[self.handles.hand_qpos_adrs] = root_q[7:]
    mujoco.mj_forward(self.handles.model, scratch)
    pose = np.concatenate(
      (target_position, _quaternion_from_matrix(target_rotation))
    )
    for _ in range(iterations or self.config.arm_ik_iterations):
      scratch.qpos[self.handles.arm_qpos_adrs] = self._pose_ik.solve(scratch, pose)
      mujoco.mj_forward(self.handles.model, scratch)
    terminal_q = np.concatenate(
      (
        np.asarray(scratch.qpos[self.handles.arm_qpos_adrs]),
        np.asarray(scratch.qpos[self.handles.hand_qpos_adrs]),
      )
    )
    position = np.array(scratch.site_xpos[self.handles.palm_site_id], copy=True)
    rotation = np.array(
      scratch.site_xmat[self.handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)
    return (
      terminal_q,
      np.array(scratch.site_xpos[self.handles.tip_site_ids], copy=True),
      position,
      float(np.linalg.norm(position - target_position)),
      _rotation_angle(rotation, target_rotation),
    )

  def pose_candidates(
    self,
    data: mujoco.MjData,
    target_vertex: int,
    actual_contact_set: frozenset[int],
    *,
    full_yaw_search: bool = False,
  ) -> dict[int, PoseCandidate]:
    root_q = _logical_q(self.handles, data)
    palm_position = np.array(data.site_xpos[self.handles.palm_site_id], copy=True)
    palm_rotation = np.array(
      data.site_xmat[self.handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)
    tips = np.array(data.site_xpos[self.handles.tip_site_ids], copy=True)
    local_offsets = (palm_rotation.T @ (tips - palm_position).T).T
    target_surface = (
      self.handles.object_position_m
      + self.surface_graph.vertices_m[int(target_vertex)]
    )
    target_normal = np.array(
      self.surface_graph.normals[int(target_vertex)],
      copy=True,
    )
    target_normal /= np.linalg.norm(target_normal)
    aligned = _align_axis(palm_rotation[:, 2], target_normal) @ palm_rotation
    yaw_values = (
      (0.0, -np.pi / 6.0, np.pi / 6.0, -np.pi / 3.0, np.pi / 3.0, np.pi)
      if full_yaw_search
      else (0.0,)
    )
    result: dict[int, PoseCandidate] = {}
    for finger in range(1, 5):
      support = _pad_support_radius(
        self.handles,
        data,
        finger - 1,
        target_normal,
      )
      desired_tip = target_surface + support * target_normal
      for yaw in yaw_values:
        rotation = (
          Rotation.from_rotvec(yaw * target_normal).as_matrix() @ aligned
        )
        desired_palm = desired_tip - rotation @ local_offsets[finger - 1]
        clipped_position, clipped_rotation = _clip_pose(
          palm_position,
          palm_rotation,
          desired_palm,
          rotation,
          self.config,
        )
        q, terminal_tips, _, position_error, orientation_error = self._solve_arm_pose(
          root_q,
          clipped_position,
          clipped_rotation,
        )
        margin = self._exact_kinematics.joint_margin(q)
        target_tip_distance = float(
          np.linalg.norm(terminal_tips[finger - 1] - desired_tip)
        )
        feasible = (
          position_error <= self.config.arm_position_tolerance_m
          and orientation_error <= self.config.arm_orientation_tolerance_rad
          and margin >= self.config.minimum_joint_margin_rad
        )
        score = (
          position_error
          + 0.12 * orientation_error
          + 0.06 * target_tip_distance
          + 0.003 * abs(yaw)
          + (0.012 if finger not in actual_contact_set else 0.0)
          + 0.001 / max(margin, 1e-4)
          if feasible
          else float("inf")
        )
        candidate = PoseCandidate(
          finger_id=finger,
          yaw_offset_rad=float(yaw),
          target_position_m=np.asarray(clipped_position),
          target_rotation=np.asarray(clipped_rotation),
          terminal_arm_q_rad=np.asarray(q[:7]),
          terminal_tip_position_m=np.asarray(terminal_tips[finger - 1]),
          position_error_m=position_error,
          orientation_error_rad=orientation_error,
          joint_margin_rad=margin,
          target_tip_distance_m=target_tip_distance,
          score=float(score),
        )
        previous = result.get(finger)
        if previous is None or candidate.score < previous.score:
          result[finger] = candidate
    return result

  def goal_feasibility_score(
    self,
    data: mujoco.MjData,
    goal_vertex: int,
    actual_contact_set: frozenset[int],
  ) -> float | None:
    candidates = self.pose_candidates(
      data,
      goal_vertex,
      actual_contact_set,
      full_yaw_search=True,
    )
    finite = [candidate.score for candidate in candidates.values() if candidate.feasible]
    return None if not finite else 0.05 * min(finite)

  def _state_rooted_finger_candidates(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    goal_vertex: int,
    contact_positions_world_m: NDArray[np.float64],
    contact_normals_world: NDArray[np.float64],
    maximum_surface_step_m: float,
  ) -> tuple[
    dict[int, PoseCandidate],
    dict[int, int],
    dict[int, float | None],
  ]:
    """Expand one no-finger-ID goal into per-finger geodesic bridges.

    Each measured contact owns a different mesh root.  The Oracle still emits
    only ``goal_vertex``; this expansion belongs to the explicit MCC planner
    and is deliberately recomputed from the fresh barrier on every prefix.
    Free fingers receive a local MAKE candidate at their nearest surface point
    so I03 can restore viability without the Oracle assigning a finger.
    """

    if maximum_surface_step_m <= 0.0:
      raise ValueError("maximum_surface_step_m must be positive")
    positions = np.asarray(contact_positions_world_m, dtype=np.float64)
    normals = np.asarray(contact_normals_world, dtype=np.float64)
    if positions.shape != (4, 3) or normals.shape != (4, 3):
      raise ValueError("contact positions/normals must both have shape (4,3)")

    root_q = _logical_q(self.handles, data)
    palm_position = np.array(data.site_xpos[self.handles.palm_site_id], copy=True)
    palm_rotation = np.array(
      data.site_xmat[self.handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)
    tips = np.array(data.site_xpos[self.handles.tip_site_ids], copy=True)
    margin = self._exact_kinematics.joint_margin(root_q)
    nearest_free, _ = self.surface_graph.nearest_vertices(
      tips - self.handles.object_position_m
    )

    poses: dict[int, PoseCandidate] = {}
    targets: dict[int, int] = {}
    remaining: dict[int, float | None] = {}
    for finger in range(1, 5):
      if finger in actual_contact_set:
        roots, _ = self.surface_graph.oriented_nearest_vertices(
          positions[finger - 1 : finger] - self.handles.object_position_m,
          normals[finger - 1 : finger],
        )
        root = int(roots[0])
        distances, predecessors = self.surface_graph.distances_from(
          root,
          return_predecessors=True,
        )
        path = self.surface_graph.reconstruct_path(
          root,
          int(goal_vertex),
          predecessors,
        )
        bridge = self.surface_graph.decimate_path(
          path,
          maximum_step_m=maximum_surface_step_m,
        )
        target_vertex = int(bridge[1] if len(bridge) > 1 else bridge[0])
        route_remaining = float(distances[int(goal_vertex)])
      else:
        target_vertex = int(nearest_free[finger - 1])
        route_remaining = None

      target_surface = (
        self.handles.object_position_m
        + self.surface_graph.vertices_m[target_vertex]
      )
      target_normal = np.array(
        self.surface_graph.normals[target_vertex],
        dtype=np.float64,
        copy=True,
      )
      target_normal /= np.linalg.norm(target_normal)
      support = _pad_support_radius(
        self.handles,
        data,
        finger - 1,
        target_normal,
      )
      # MuJoCo's compiled mesh SDF and the visual mesh differ by roughly one
      # grid cell on high-curvature patches.  A free-finger MAKE therefore
      # targets 4 mm behind the analytic pad-center surface; M03/M10 still
      # bound the realized force and swept motion.  Existing SLIDE contacts
      # retain the much smaller 0.3 mm preload.
      preload = 0.0003 if finger in actual_contact_set else 0.0040
      desired_tip = target_surface + (support - preload) * target_normal
      distance = float(np.linalg.norm(tips[finger - 1] - desired_tip))
      scratch = mujoco.MjData(self.handles.model)
      scratch.qpos[self.handles.arm_qpos_adrs] = root_q[:7]
      scratch.qpos[self.handles.hand_qpos_adrs] = root_q[7:]
      mujoco.mj_forward(self.handles.model, scratch)
      for _ in range(32):
        command = _finger_ik(
          self.handles,
          scratch,
          finger - 1,
          desired_tip,
          -target_normal,
          damping=0.009,
          gain=0.48,
          posture_gain=0.0,
        )
        scratch.qpos[self.handles.finger_qpos_adrs[finger - 1]] = command
        mujoco.mj_forward(self.handles.model, scratch)
      achieved_tip = np.array(
        scratch.site_xpos[self.handles.tip_site_ids[finger - 1]],
        copy=True,
      )
      nonlinear_residual = float(np.linalg.norm(achieved_tip - desired_tip))
      nonlinear_motion = float(np.linalg.norm(achieved_tip - tips[finger - 1]))
      residual_limit = 0.0030 if finger in actual_contact_set else 0.0060
      motion_limit = 0.010 if finger in actual_contact_set else 0.012
      feasible = (
        nonlinear_residual <= residual_limit
        and nonlinear_motion <= motion_limit
        and margin >= self.config.minimum_joint_margin_rad
      )
      # Palm motion is not part of a finger transaction.  The pose record is
      # an M08/M09/M11 scoring carrier; exact finger realization follows only
      # after M11 has selected the primitive/finger.
      poses[finger] = PoseCandidate(
        finger_id=finger,
        yaw_offset_rad=0.0,
        target_position_m=palm_position.copy(),
        target_rotation=palm_rotation.copy(),
        terminal_arm_q_rad=root_q[:7].copy(),
        terminal_tip_position_m=achieved_tip,
        position_error_m=0.0,
        orientation_error_rad=0.0,
        joint_margin_rad=margin,
        target_tip_distance_m=nonlinear_residual,
        score=(
          (
            (0.002 if finger in actual_contact_set else 0.025)
            + distance
            + 2.0 * nonlinear_residual
            + (
              0.15 * float(route_remaining)
              if finger in actual_contact_set
              else 0.0
            )
          )
          if feasible
          else float("inf")
        ),
      )
      targets[finger] = target_vertex
      remaining[finger] = route_remaining
    return poses, targets, remaining

  def _planner_state(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
  ) -> PlannerState:
    return PlannerState(
      joint_positions_rad=_logical_q(self.handles, data),
      wrist_position_m=data.site_xpos[self.handles.palm_site_id],
      fingertip_positions_m=data.site_xpos[self.handles.tip_site_ids],
      actual_contact_set=actual_contact_set,
      surface_model_version=SURFACE_MODEL_VERSION,
    )

  def _physical_finger_target(
    self,
    data: mujoco.MjData,
    finger: int,
    target_vertex: int,
    primitive_kind: PrimitiveKind,
  ) -> NDArray[np.float64]:
    """Convert a mesh point into the oriented collision-pad center target."""

    normal = np.array(
      self.surface_graph.normals[int(target_vertex)],
      dtype=np.float64,
      copy=True,
    )
    normal /= np.linalg.norm(normal)
    surface = (
      self.handles.object_position_m
      + self.surface_graph.vertices_m[int(target_vertex)]
    )
    support = _pad_support_radius(
      self.handles,
      data,
      finger - 1,
      normal,
    )
    if primitive_kind is PrimitiveKind.REPOSITION:
      offset = support + 0.004
    elif primitive_kind is PrimitiveKind.MAKE:
      offset = support - 0.004
    elif primitive_kind is PrimitiveKind.SLIDE:
      offset = support - 0.0003
    else:
      raise ValueError("I04 physical target supports SLIDE/MAKE/REPOSITION")
    return surface + offset * normal

  def _search_finger(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    target_vertex: int,
    pose_candidates: dict[int, PoseCandidate],
    target_vertex_by_finger: dict[int, int],
    sequence: int,
    forced_break_finger: int | None = None,
  ) -> tuple[int, PrimitiveKind, dict[str, Any]]:
    state = self._planner_state(data, actual_contact_set)
    kinematics = _local_kinematics(self.handles, data)
    optimizer = ContinuousOptimizer(
      self.graph,
      self.surface_model,
      kinematics,
      OptimizationConfig(
        waypoint_count=7,
        # M09 may see a farther surface look-ahead/contact target, but it may
        # optimize only the next closed-loop micro-step.  The remaining
        # displacement has prediction value only and is replanned from the
        # next measured M06 barrier.
        max_commit_displacement_m=self.config.maximum_translation_per_prefix_m,
        target_tolerance_m=0.0015,
        anchor_tolerance_m=0.0025,
        minimum_collision_clearance_m=0.0,
        minimum_joint_margin_rad=0.0,
        reach_radius_m=0.11,
        release_distance_m=0.006,
        free_surface_clearance_m=0.003,
        nominal_speed_m_s=0.035,
        damping=3e-4,
        ik_iterations=14,
      ),
    )
    cheap = CheapCert(self.graph, CheapCertConfig(reject_joint_below_rad=-0.01))
    shadow = ShadowViabilityEvaluator(self.graph, cheap)
    def factory(
      planner_state: PlannerState,
      primitive: ContactPrimitive,
      depth: int,
    ) -> PlanningCandidate | None:
      if primitive.finger_id is None:
        return None
      finger = primitive.finger_id
      pose = pose_candidates.get(finger)
      if pose is None:
        return None
      forced_break_is_active = (
        forced_break_finger is not None
        and forced_break_finger in planner_state.actual_contact_set
      )
      if forced_break_is_active:
        if (
          finger != forced_break_finger
          or primitive.kind is not PrimitiveKind.BREAK
        ):
          return None
        expected_kind = PrimitiveKind.BREAK
      elif finger in planner_state.actual_contact_set:
        expected_kind = PrimitiveKind.SLIDE
      else:
        expected_kind = (
          PrimitiveKind.MAKE if pose.feasible else PrimitiveKind.REPOSITION
        )
      if primitive.kind is not expected_kind:
        return None
      if len(planner_state.actual_contact_set) == 1 and primitive.kind is not PrimitiveKind.MAKE:
        # I03 terminal-viability semantics: a singleton first restores a real
        # replacement contact; it may not keep sliding the last anchor.
        return None
      if (
        not pose.feasible
        and primitive.kind
        not in {PrimitiveKind.REPOSITION, PrimitiveKind.BREAK}
      ):
        return None
      candidate_vertex = int(target_vertex_by_finger.get(finger, target_vertex))
      if primitive.kind is PrimitiveKind.BREAK:
        target = np.array(
          planner_state.fingertip_positions_m[finger - 1],
          copy=True,
        )
      else:
        target = self._physical_finger_target(
          data,
          finger,
          candidate_vertex,
          primitive.kind,
        )
      distance = float(
        np.linalg.norm(
          planner_state.fingertip_positions_m[finger - 1] - target
        )
      )
      margin = kinematics.joint_margin(planner_state.joint_positions_rad)
      cheap_input = CheapCertInput(
        mode=planner_state.mode,
        primitive=primitive,
        surface_model_version=SURFACE_MODEL_VERSION,
        anchor_margin_m=0.002,
        joint_margin_rad=margin,
        collision_margin_m=0.004,
        reach_margin_m=0.11 - distance,
        uncertainty_margin=1.0,
        trust_margin_m=(
          self.config.maximum_translation_per_prefix_m
          - min(distance, self.config.maximum_translation_per_prefix_m)
        ),
        metadata={
          "depth": float(depth),
          "pose_score": pose.score if np.isfinite(pose.score) else 1.0,
        },
      )
      return PlanningCandidate(
        cheap_input=cheap_input,
        optimization_request=OptimizationRequest(
          state=planner_state,
          primitive=primitive,
          target_position_m=target,
          prefix_id=f"i04-search-{sequence:05d}-d{depth}-{primitive.key}",
          progress_gain_m=(
            0.0
            if primitive.kind is PrimitiveKind.BREAK
            else min(
              distance,
              self.config.maximum_translation_per_prefix_m,
            )
          ),
          metadata=(
            {
              "pose_score": pose.score if np.isfinite(pose.score) else 1.0,
              "forced_workspace_break": 1.0,
            }
            if primitive.kind is PrimitiveKind.BREAK
            else {
              "pose_score": pose.score if np.isfinite(pose.score) else 1.0,
              "physical_pad_center_target": 1.0,
            }
          ),
        ),
        motion_cost=0.05 * distance,
        risk_cost=(
          pose.score
          if np.isfinite(pose.score)
          else 0.02
        ) + (0.03 if primitive.kind is PrimitiveKind.MAKE else 0.0),
      )

    started = perf_counter()
    search = LazyBeamSearch(
      self.graph,
      cheap,
      optimizer,
      BeamSearchConfig(
        horizon=self.config.beam_horizon,
        beam_width=self.config.beam_width,
        per_mode_quota=self.config.per_mode_quota,
      ),
      SearchWeights(progress=8.0, contact=0.2, motion=1.0, risk=1.5, switch=0.08),
    ).search(
      state,
      factory,
      terminal_viability=shadow.predicate(factory),
    )
    wall = perf_counter() - started
    if not search.found or search.committed_prefix_candidate is None:
      raise RuntimeError(
        "M11 found no M08/M09/M12-surviving I04 edge "
        f"[enumerated={search.enumerated_edges},"
        f" cheap={search.cheap_survivors},"
        f" optimized={search.optimized_edges},"
        f" retained={list(search.retained_nodes_per_depth)}]"
      )
    prefix = search.committed_prefix_candidate
    if prefix.finger_id is None:
      raise AssertionError("I04 search selected a fingerless primitive")
    terminal_shadow = shadow.evaluate(search.best_node.state, factory)  # type: ignore[union-attr]
    return (
      int(prefix.finger_id),
      PrimitiveKind(prefix.primitive_kind),
      {
        "m11_latency_s": search.latency_s,
        "m11_wall_latency_s": wall,
        "m11_expanded_nodes": search.expanded_nodes,
        "m11_enumerated_edges": search.enumerated_edges,
        "m11_cheap_survivors": search.cheap_survivors,
        "m11_optimized_edges": search.optimized_edges,
        "m11_selected_sequence": list(search.best_node.sequence_key),  # type: ignore[union-attr]
        "prediction_suffix_count": len(search.prediction_suffix),
        "prediction_suffix_execution_authority": False,
        "m12_status": terminal_shadow.status.value,
        "m12_reason": terminal_shadow.reason,
        "m12_execution_authority": terminal_shadow.execution_authority,
        "m12_successor_fingers": list(terminal_shadow.distinct_successor_fingers),
        "m12_latency_s": terminal_shadow.latency_s,
      },
    )

  def _search_wrist(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    wrist_target_m: NDArray[np.float64],
    pose_candidates: dict[int, PoseCandidate],
    target_vertex_by_finger: dict[int, int],
    sequence: int,
  ) -> dict[str, Any]:
    """Select WRIST_ADJUST through the same M07-M12 stack.

    Finger candidates remain in the factory so M12 can prove a nontrivial
    continuation from the wrist-adjusted terminal state.  Their root-search
    risk is intentionally high in this dedicated recenter transaction; only
    M11's selected WRIST prefix is eligible for subsequent exact refinement.
    """

    state = self._planner_state(data, actual_contact_set)
    kinematics, optimizer = self._wrist_optimizer(data)
    cheap = CheapCert(self.graph, CheapCertConfig(reject_joint_below_rad=-0.01))
    shadow = ShadowViabilityEvaluator(self.graph, cheap)

    def factory(
      planner_state: PlannerState,
      primitive: ContactPrimitive,
      depth: int,
    ) -> PlanningCandidate | None:
      margin = kinematics.joint_margin(planner_state.joint_positions_rad)
      if primitive.kind is PrimitiveKind.WRIST_ADJUST:
        displacement = float(
          np.linalg.norm(wrist_target_m - planner_state.wrist_position_m)
        )
        return PlanningCandidate(
          cheap_input=CheapCertInput(
            mode=planner_state.mode,
            primitive=primitive,
            surface_model_version=SURFACE_MODEL_VERSION,
            anchor_margin_m=0.002,
            joint_margin_rad=margin,
            collision_margin_m=0.004,
            reach_margin_m=0.11,
            uncertainty_margin=1.0,
            trust_margin_m=(
              self.config.wrist_adjust_step_m
              - min(displacement, self.config.wrist_adjust_step_m)
            ),
            metadata={"depth": float(depth), "i04_wrist_recenter": 1.0},
          ),
          optimization_request=OptimizationRequest(
            state=planner_state,
            primitive=primitive,
            target_wrist_position_m=wrist_target_m,
            prefix_id=f"i04-search-{sequence:05d}-d{depth}-WRIST",
            progress_gain_m=displacement,
            metadata={"i04_wrist_recenter": True},
          ),
          motion_cost=0.05 * displacement,
          risk_cost=0.0,
        )

      if primitive.finger_id is None:
        return None
      finger = primitive.finger_id
      expected_kind = (
        PrimitiveKind.SLIDE
        if finger in planner_state.actual_contact_set
        else PrimitiveKind.MAKE
      )
      if primitive.kind is not expected_kind:
        return None
      if len(planner_state.actual_contact_set) == 1 and primitive.kind is not PrimitiveKind.MAKE:
        return None
      pose = pose_candidates.get(finger)
      if pose is None or not pose.feasible:
        return None
      target_vertex = int(target_vertex_by_finger[finger])
      target = self._physical_finger_target(
        data,
        finger,
        target_vertex,
        primitive.kind,
      )
      distance = float(
        np.linalg.norm(
          planner_state.fingertip_positions_m[finger - 1] - target
        )
      )
      return PlanningCandidate(
        cheap_input=CheapCertInput(
          mode=planner_state.mode,
          primitive=primitive,
          surface_model_version=SURFACE_MODEL_VERSION,
          anchor_margin_m=0.002,
          joint_margin_rad=margin,
          collision_margin_m=0.004,
          reach_margin_m=0.11 - distance,
          uncertainty_margin=1.0,
          trust_margin_m=(
            self.config.maximum_translation_per_prefix_m
            - min(distance, self.config.maximum_translation_per_prefix_m)
          ),
          metadata={"depth": float(depth), "shadow_successor": 1.0},
        ),
        optimization_request=OptimizationRequest(
          state=planner_state,
          primitive=primitive,
          target_position_m=target,
          prefix_id=f"i04-shadow-{sequence:05d}-d{depth}-{primitive.key}",
          progress_gain_m=min(
            distance,
            self.config.maximum_translation_per_prefix_m,
          ),
          metadata={
            "shadow_successor": True,
            "physical_pad_center_target": 1.0,
          },
        ),
        motion_cost=0.05 * distance,
        risk_cost=100.0,
      )

    started = perf_counter()
    search = LazyBeamSearch(
      self.graph,
      cheap,
      optimizer,
      BeamSearchConfig(
        horizon=1,
        beam_width=self.config.beam_width,
        per_mode_quota=self.config.per_mode_quota,
      ),
      SearchWeights(progress=8.0, contact=0.2, motion=1.0, risk=1.5, switch=0.08),
    ).search(
      state,
      factory,
      terminal_viability=shadow.predicate(factory),
    )
    wall = perf_counter() - started
    candidate = search.committed_prefix_candidate
    if (
      not search.found
      or candidate is None
      or candidate.primitive_kind != PrimitiveKind.WRIST_ADJUST.value
    ):
      raise RuntimeError(
        "M11 found no M08/M09/M12-surviving WRIST_ADJUST edge "
        f"[enumerated={search.enumerated_edges},"
        f" cheap={search.cheap_survivors},"
        f" optimized={search.optimized_edges},"
        f" retained={list(search.retained_nodes_per_depth)}]"
      )
    terminal_shadow = shadow.evaluate(search.best_node.state, factory)  # type: ignore[union-attr]
    return {
      "m11_latency_s": search.latency_s,
      "m11_wall_latency_s": wall,
      "m11_expanded_nodes": search.expanded_nodes,
      "m11_enumerated_edges": search.enumerated_edges,
      "m11_cheap_survivors": search.cheap_survivors,
      "m11_optimized_edges": search.optimized_edges,
      "m11_selected_sequence": list(search.best_node.sequence_key),  # type: ignore[union-attr]
      "prediction_suffix_count": len(search.prediction_suffix),
      "prediction_suffix_execution_authority": False,
      "m12_status": terminal_shadow.status.value,
      "m12_reason": terminal_shadow.reason,
      "m12_execution_authority": terminal_shadow.execution_authority,
      "m12_successor_fingers": list(terminal_shadow.distinct_successor_fingers),
      "m12_latency_s": terminal_shadow.latency_s,
    }

  def _wrist_optimizer(
    self,
    data: mujoco.MjData,
  ) -> tuple[Any, ContinuousOptimizer]:
    """Build the shared M09 model used for WRIST selection and M11."""

    kinematics = _local_kinematics(self.handles, data)
    optimizer = ContinuousOptimizer(
      self.graph,
      self.surface_model,
      kinematics,
      OptimizationConfig(
        waypoint_count=7,
        max_commit_displacement_m=max(
          self.config.maximum_translation_per_prefix_m,
          self.config.wrist_adjust_step_m,
        ),
        target_tolerance_m=0.0015,
        anchor_tolerance_m=0.0025,
        minimum_collision_clearance_m=0.0,
        # Match the downstream M10 joint-margin requirement.  The previous
        # zero-margin M09 model could call a direction feasible only by
        # saturating an anchor joint, which M10 could never authorize.
        minimum_joint_margin_rad=self.config.minimum_joint_margin_rad,
        reach_radius_m=0.11,
        release_distance_m=0.006,
        free_surface_clearance_m=0.003,
        nominal_speed_m_s=0.035,
        damping=3e-4,
        ik_iterations=14,
      ),
    )
    return kinematics, optimizer

  def _select_wrist_target(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    desired_direction_world: NDArray[np.float64],
    outward_normal_world: NDArray[np.float64],
    sequence: int,
  ) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Choose a progress-positive WRIST direction that preserves anchors.

    The exact geodesic tangent can demand joint motion beyond an anchor's
    remaining workspace.  This deterministic local search rotates within the
    local tangent plane while retaining strictly positive progress toward the
    Oracle waypoint, then evaluates every candidate with the same M09
    formulation used by M11.  It deliberately never adds a surface-normal
    component: normal-force correction remains the shared MCC's job.  This
    screen does not issue a certificate or bypass M11/M12/M10.
    """

    desired = np.asarray(desired_direction_world, dtype=np.float64)
    desired /= np.linalg.norm(desired)
    normal = np.asarray(outward_normal_world, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    desired -= float(np.dot(desired, normal)) * normal
    desired_norm = float(np.linalg.norm(desired))
    if desired_norm <= 1e-9:
      raise RuntimeError("WRIST desired tangent is degenerate")
    desired /= desired_norm
    binormal = np.cross(normal, desired)
    binormal /= np.linalg.norm(binormal)

    state = self._planner_state(data, actual_contact_set)
    _kinematics, optimizer = self._wrist_optimizer(data)
    primitive = ContactPrimitive(PrimitiveKind.WRIST_ADJUST)
    root_wrist = np.array(
      data.site_xpos[self.handles.palm_site_id],
      dtype=np.float64,
      copy=True,
    )
    direction_keys: set[tuple[float, float, float]] = set()
    directions: list[NDArray[np.float64]] = []
    for side_angle_deg in range(-80, 81, 10):
      side_angle = np.deg2rad(float(side_angle_deg))
      direction = (
        np.cos(side_angle) * desired
        + np.sin(side_angle) * binormal
      )
      # Numerical projection makes the planner/MCC authority split explicit.
      direction -= float(np.dot(direction, normal)) * normal
      direction /= np.linalg.norm(direction)
      key = tuple(np.round(direction, decimals=10))
      if key not in direction_keys:
        direction_keys.add(key)
        directions.append(direction)

    rejection_counts: dict[str, int] = {}
    feasible: list[tuple[float, float, float, NDArray[np.float64], Any]] = []
    evaluated = 0
    for step_scale in (1.0, 0.75, 0.50, 0.25):
      step_m = self.config.wrist_adjust_step_m * step_scale
      for candidate_index, direction in enumerate(directions):
        progress_cosine = float(np.dot(direction, desired))
        if progress_cosine <= 1e-6:
          continue
        evaluated += 1
        target = root_wrist + step_m * direction
        result = optimizer.optimize(
          OptimizationRequest(
            state=state,
            primitive=primitive,
            target_wrist_position_m=target,
            prefix_id=(
              f"i04-wrist-select-{sequence:05d}-"
              f"s{step_scale:.2f}-c{candidate_index:03d}"
            ),
            progress_gain_m=step_m * progress_cosine,
            metadata={"i04_wrist_direction_screen": True},
          )
        )
        if not result.feasible or result.prefix is None:
          for reason in result.reasons or ("UNKNOWN",):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
          continue
        feasible.append(
          (
            step_m * progress_cosine,
            result.minimum_joint_margin_rad,
            result.anchor_margin_m,
            direction,
            result,
          )
        )
    if not feasible:
      raise RuntimeError(
        "M09 found no anchor-preserving WRIST direction "
        f"[evaluated={evaluated}, reasons={rejection_counts}]"
      )

    # Prefer candidates with a useful workspace buffer when one exists.  This
    # is stricter than M10's 8 mrad safety floor and prevents the next prefix
    # from immediately returning to the same joint-limit blocker.
    preferred_margin = max(0.020, self.config.minimum_joint_margin_rad)
    buffered = [row for row in feasible if row[1] >= preferred_margin]
    pool = buffered or feasible
    selected = max(
      pool,
      key=lambda row: (
        row[0],
        row[1],
        row[2],
        tuple(-abs(float(value)) for value in row[3]),
      ),
    )
    progress_m, joint_margin, anchor_margin, direction, result = selected
    target = root_wrist + float(
      np.linalg.norm(result.prefix.samples[-1].wrist_position_m - root_wrist)
    ) * direction
    return target, {
      "wrist_direction_candidates_evaluated": evaluated,
      "wrist_direction_feasible_count": len(feasible),
      "wrist_direction_buffered_count": len(buffered),
      "wrist_direction_rejection_counts": rejection_counts,
      "wrist_selected_direction_world": direction.tolist(),
      "wrist_selected_step_m": float(np.linalg.norm(target - root_wrist)),
      "wrist_selected_progress_m": progress_m,
      "wrist_selected_progress_cosine": float(np.dot(direction, desired)),
      "wrist_selected_surface_normal_component": float(
        np.dot(direction, normal)
      ),
      "wrist_direction_is_tangent_only": True,
      "wrist_selected_m09_joint_margin_rad": joint_margin,
      "wrist_selected_m09_anchor_error_m": (
        optimizer.config.anchor_tolerance_m - anchor_margin
      ),
      "wrist_direction_selection_execution_authority": False,
    }

  def _refine_wrist_adjust(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    contact_normals_world: NDArray[np.float64],
    target_wrist_position_m: NDArray[np.float64],
    target_wrist_rotation: NDArray[np.float64],
    sequence: int,
  ) -> tuple[PlannedPrefix, NDArray[np.float64], dict[str, Any]]:
    """Exact nonlinear arm recenter while measured fingertip anchors stay fixed."""

    root_q = _logical_q(self.handles, data)
    root_wrist = np.array(data.site_xpos[self.handles.palm_site_id], copy=True)
    root_rotation = np.array(
      data.site_xmat[self.handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)
    root_tips = np.array(data.site_xpos[self.handles.tip_site_ids], copy=True)
    displacement = np.asarray(target_wrist_position_m) - root_wrist
    length = float(np.linalg.norm(displacement))
    if length > self.config.wrist_adjust_step_m:
      displacement *= self.config.wrist_adjust_step_m / length
    target = root_wrist + displacement
    target_rotation = np.asarray(target_wrist_rotation, dtype=np.float64)
    rotation_amount = _rotation_angle(root_rotation, target_rotation)
    duration = max(
      0.40,
      float(np.linalg.norm(displacement)) / 0.004,
      rotation_amount / self.config.nominal_rotation_speed_rad_s,
    )
    slerp = Slerp(
      [0.0, 1.0],
      Rotation.from_matrix(np.stack((root_rotation, target_rotation))),
    )
    samples: list[PrefixSample] = []
    seed_q = root_q.copy()
    terminal_rotation = root_rotation
    for index, alpha_linear in enumerate(
      np.linspace(0.0, 1.0, self.config.prefix_samples)
    ):
      alpha = float(alpha_linear**2 * (3.0 - 2.0 * alpha_linear))
      desired_wrist = root_wrist + alpha * displacement
      desired_rotation = slerp([alpha]).as_matrix()[0]
      if index == 0:
        scratch = data
        q = root_q.copy()
      else:
        q, _, _, _, _ = self._solve_arm_pose(
          seed_q,
          desired_wrist,
          desired_rotation,
          iterations=max(18, self.config.arm_ik_iterations // 2),
        )
        scratch = mujoco.MjData(self.handles.model)
        scratch.qpos[self.handles.arm_qpos_adrs] = q[:7]
        scratch.qpos[self.handles.hand_qpos_adrs] = q[7:]
        mujoco.mj_forward(self.handles.model, scratch)
        for finger in sorted(actual_contact_set):
          normal = np.asarray(contact_normals_world[finger - 1], dtype=np.float64)
          normal /= np.linalg.norm(normal)
          for _ in range(48):
            command = _finger_ik(
              self.handles,
              scratch,
              finger - 1,
              root_tips[finger - 1],
              -normal,
              damping=0.009,
              gain=0.46,
              orientation_weight=0.003,
              posture_gain=0.0,
            )
            scratch.qpos[self.handles.finger_qpos_adrs[finger - 1]] = command
            mujoco.mj_forward(self.handles.model, scratch)
        q = np.concatenate(
          (
            np.asarray(scratch.qpos[self.handles.arm_qpos_adrs]),
            np.asarray(scratch.qpos[self.handles.hand_qpos_adrs]),
          )
        )
        seed_q = q.copy()
      wrist = np.array(scratch.site_xpos[self.handles.palm_site_id], copy=True)
      tips = np.array(scratch.site_xpos[self.handles.tip_site_ids], copy=True)
      terminal_rotation = np.array(
        scratch.site_xmat[self.handles.palm_site_id],
        copy=True,
      ).reshape(3, 3)
      samples.append(
        PrefixSample(
          time_s=float(alpha_linear * duration),
          wrist_position_m=wrist,
          fingertip_positions_m=tips,
          joint_positions_rad=np.array(q, copy=True),
        )
      )
    anchor_indices = np.asarray(
      [finger - 1 for finger in sorted(actual_contact_set)],
      dtype=np.int32,
    )
    terminal_anchor_error = float(
      np.max(
        np.linalg.norm(
          samples[-1].fingertip_positions_m[anchor_indices]
          - root_tips[anchor_indices],
          axis=1,
        )
      )
    )
    prefix = PlannedPrefix(
      prefix_id=f"i04-{sequence:05d}-wrist-adjust",
      transaction_type=TransactionType.WRIST_ADJUST,
      primitive_kind=PrimitiveKind.WRIST_ADJUST.value,
      surface_model_version=SURFACE_MODEL_VERSION,
      root_contact_set=actual_contact_set,
      expected_terminal_contact_set=actual_contact_set,
      samples=tuple(samples),
      participating_fingers=(),
      anchor_fingers=tuple(sorted(actual_contact_set)),
      finger_id=None,
      topology_change_count=0,
      source=PrefixSource.OPTIMIZER_COMMIT_CANDIDATE,
      metadata={
        "i04_wrist_recenter": True,
        "nonlinear_refinement": True,
        "whole_hand_roles": _whole_hand_roles(
          actual_contact_set,
          PrimitiveKind.WRIST_ADJUST,
          None,
        ),
        "local_surface_micro_step": True,
        "micro_step_limit_m": self.config.wrist_adjust_step_m,
        "fresh_barrier_replan_required": True,
        "oracle_waypoint_direct_execution": False,
      },
    )
    terminal_pose = np.concatenate(
      (
        samples[-1].wrist_position_m,
        _quaternion_from_matrix(terminal_rotation),
      )
    )
    return (
      prefix,
      terminal_pose,
      {
        "prefix_duration_s": duration,
        "prefix_translation_m": float(
          np.linalg.norm(samples[-1].wrist_position_m - root_wrist)
        ),
        "prefix_rotation_rad": _rotation_angle(root_rotation, terminal_rotation),
        "terminal_anchor_error_m": terminal_anchor_error,
        "i04_wrist_recenter": True,
      },
    )

  def _refine_prefix(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    primitive_kind: PrimitiveKind,
    pose: PoseCandidate,
    sequence: int,
    target_vertex: int,
  ) -> tuple[PlannedPrefix, NDArray[np.float64], dict[str, Any]]:
    root_q = _logical_q(self.handles, data)
    root_position = np.array(data.site_xpos[self.handles.palm_site_id], copy=True)
    root_rotation = np.array(
      data.site_xmat[self.handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)
    translation = float(np.linalg.norm(pose.target_position_m - root_position))
    rotation = _rotation_angle(root_rotation, pose.target_rotation)
    duration = max(
      self.config.minimum_prefix_duration_s,
      translation / self.config.nominal_translation_speed_m_s,
      rotation / self.config.nominal_rotation_speed_rad_s,
    )
    rotations = Rotation.from_matrix(
      np.stack((root_rotation, pose.target_rotation))
    )
    slerp = Slerp([0.0, 1.0], rotations)
    samples: list[PrefixSample] = []
    seed_q = root_q.copy()
    terminal_rotation = root_rotation
    for index, alpha_linear in enumerate(
      np.linspace(0.0, 1.0, self.config.prefix_samples)
    ):
      alpha = float(alpha_linear**2 * (3.0 - 2.0 * alpha_linear))
      desired_position = (
        (1.0 - alpha) * root_position + alpha * pose.target_position_m
      )
      desired_rotation = slerp([alpha]).as_matrix()[0]
      if index == 0:
        q = root_q.copy()
        scratch = data
      else:
        q, _, _, _, _ = self._solve_arm_pose(
          seed_q,
          desired_position,
          desired_rotation,
          iterations=max(18, self.config.arm_ik_iterations // 2),
        )
        scratch = mujoco.MjData(self.handles.model)
        scratch.qpos[self.handles.arm_qpos_adrs] = q[:7]
        scratch.qpos[self.handles.hand_qpos_adrs] = q[7:]
        mujoco.mj_forward(self.handles.model, scratch)
        seed_q = q
      wrist = np.array(scratch.site_xpos[self.handles.palm_site_id], copy=True)
      tips = np.array(scratch.site_xpos[self.handles.tip_site_ids], copy=True)
      terminal_rotation = np.array(
        scratch.site_xmat[self.handles.palm_site_id],
        copy=True,
      ).reshape(3, 3)
      samples.append(
        PrefixSample(
          time_s=float(alpha_linear * duration),
          wrist_position_m=wrist,
          fingertip_positions_m=tips,
          joint_positions_rad=q,
        )
      )
    expected = self.graph.apply_predictive(
      self._planner_state(data, actual_contact_set).mode,
      ContactPrimitive(primitive_kind, pose.finger_id),
    ).contacts
    make_progress = False
    if primitive_kind is PrimitiveKind.MAKE:
      target_surface = (
        self.handles.object_position_m
        + self.surface_graph.vertices_m[int(target_vertex)]
      )
      remaining = float(
        np.linalg.norm(samples[-1].fingertip_positions_m[pose.finger_id - 1] - target_surface)
      )
      make_progress = remaining > 0.006
      if make_progress:
        expected = actual_contact_set
    prefix = PlannedPrefix(
      prefix_id=f"i04-{sequence:05d}-{primitive_kind.value.lower()}-{pose.finger_id}",
      transaction_type=TransactionType.FINGER_RECONFIGURE,
      primitive_kind=primitive_kind.value,
      surface_model_version=SURFACE_MODEL_VERSION,
      root_contact_set=actual_contact_set,
      expected_terminal_contact_set=expected,
      samples=tuple(samples),
      participating_fingers=(pose.finger_id,),
      # I04's selected fingertip is the goal participant; other measured pads
      # are MCC-supported sliding contacts, not stationary Cartesian anchors.
      anchor_fingers=(),
      finger_id=pose.finger_id,
      topology_change_count=(
        0 if primitive_kind is PrimitiveKind.SLIDE or make_progress else 1
      ),
      source=PrefixSource.OPTIMIZER_COMMIT_CANDIDATE,
      metadata={
        "make_progress": make_progress,
        "nonlinear_refinement": True,
        "target_vertex": int(target_vertex),
        "yaw_offset_rad": pose.yaw_offset_rad,
        "planned_terminal_rotation_wxyz": _quaternion_from_matrix(
          terminal_rotation
        ).tolist(),
      },
    )
    return (
      prefix,
      np.concatenate(
        (samples[-1].wrist_position_m, _quaternion_from_matrix(terminal_rotation))
      ),
      {
        "prefix_duration_s": duration,
        "prefix_translation_m": translation,
        "prefix_rotation_rad": rotation,
        "nonlinear_terminal_tip_m": samples[-1].fingertip_positions_m[
          pose.finger_id - 1
        ].tolist(),
      },
    )

  def _refine_finger_motion(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    pose: PoseCandidate,
    sequence: int,
    target_vertex: int,
    primitive_kind: PrimitiveKind,
    contact_normal_world: NDArray[np.float64] | None = None,
  ) -> tuple[PlannedPrefix, NDArray[np.float64], dict[str, Any]]:
    """Nonlinear one-finger micro-step with measured peers fixed."""

    if primitive_kind not in {
      PrimitiveKind.SLIDE,
      PrimitiveKind.MAKE,
      PrimitiveKind.REPOSITION,
      PrimitiveKind.BREAK,
    }:
      raise ValueError(
        "finger-only refinement supports SLIDE/MAKE/REPOSITION/BREAK"
      )

    finger = pose.finger_id
    root_q = _logical_q(self.handles, data)
    root_tips = np.array(data.site_xpos[self.handles.tip_site_ids], copy=True)
    normal = np.array(
      (
        contact_normal_world
        if primitive_kind is PrimitiveKind.BREAK
        and contact_normal_world is not None
        else self.surface_graph.normals[int(target_vertex)]
      ),
      dtype=np.float64,
      copy=True,
    )
    normal /= np.linalg.norm(normal)
    surface = (
      self.handles.object_position_m
      + self.surface_graph.vertices_m[int(target_vertex)]
    )
    support = _pad_support_radius(
      self.handles,
      data,
      finger - 1,
      normal,
    )
    if primitive_kind is PrimitiveKind.BREAK:
      requested = (
        root_tips[finger - 1]
        + self.config.maximum_translation_per_prefix_m * normal
      )
    elif primitive_kind is PrimitiveKind.REPOSITION:
      requested = surface + (support + 0.004) * normal
    else:
      preload = 0.0040 if primitive_kind is PrimitiveKind.MAKE else 0.0003
      requested = surface + (support - preload) * normal
    displacement = requested - root_tips[finger - 1]
    length = float(np.linalg.norm(displacement))
    terminal_target = requested.copy()
    # Every transaction commits only one state-rooted Cartesian micro-step.
    # Farther staging/contact targets are approached through REPOSITION or
    # MAKE_PROGRESS prefixes separated by fresh M06 barriers.  An active
    # SLIDE likewise follows only the local surface direction; the final
    # Oracle waypoint is never an open-loop execution target.
    movement_limit = self.config.maximum_translation_per_prefix_m
    if length > movement_limit:
      terminal_target = (
        root_tips[finger - 1] + movement_limit * displacement / length
      )
    scratch = mujoco.MjData(self.handles.model)
    scratch.qpos[self.handles.arm_qpos_adrs] = root_q[:7]
    scratch.qpos[self.handles.hand_qpos_adrs] = root_q[7:]
    mujoco.mj_forward(self.handles.model, scratch)
    samples: list[PrefixSample] = []
    duration = max(0.45, float(np.linalg.norm(terminal_target - root_tips[finger - 1])) / 0.025)
    for index, alpha_linear in enumerate(
      np.linspace(0.0, 1.0, self.config.prefix_samples)
    ):
      alpha = float(alpha_linear**2 * (3.0 - 2.0 * alpha_linear))
      desired = (
        root_tips[finger - 1]
        + alpha * (terminal_target - root_tips[finger - 1])
      )
      if index:
        for _ in range(32):
          command = _finger_ik(
            self.handles,
            scratch,
            finger - 1,
            desired,
            -normal,
            damping=0.009,
            gain=0.48,
            posture_gain=0.0,
          )
          scratch.qpos[self.handles.finger_qpos_adrs[finger - 1]] = command
          mujoco.mj_forward(self.handles.model, scratch)
      q = np.concatenate(
        (
          np.asarray(scratch.qpos[self.handles.arm_qpos_adrs]),
          np.asarray(scratch.qpos[self.handles.hand_qpos_adrs]),
        )
      )
      samples.append(
        PrefixSample(
          time_s=float(alpha_linear * duration),
          # ``site_xpos`` is MuJoCo-owned mutable storage.  Every prefix sample
          # must be an immutable measured/realized snapshot; otherwise later
          # IK iterations silently overwrite the root sample and M10 rightly
          # reports a root-state and kinematic-trajectory mismatch.
          wrist_position_m=np.array(
            scratch.site_xpos[self.handles.palm_site_id],
            dtype=np.float64,
            copy=True,
          ),
          fingertip_positions_m=np.array(
            scratch.site_xpos[self.handles.tip_site_ids],
            dtype=np.float64,
            copy=True,
          ),
          joint_positions_rad=np.array(q, dtype=np.float64, copy=True),
        )
      )
    raw_maximum_displacement = max(
      float(
        np.linalg.norm(
          sample.fingertip_positions_m[finger - 1]
          - root_tips[finger - 1]
        )
      )
      for sample in samples
    )
    nonlinear_path_scale = 1.0
    # Resolved-rate orientation tracking can move the physical site farther
    # than its 3 mm Cartesian target even when the target position itself was
    # clipped.  Scale the exact MuJoCo joint path about the measured root; do
    # not relax M10's trust radius.  A 10% buffer covers nonlinear FK between
    # the sparse prefix knots that M10 subsequently subdivides and audits.
    buffered_limit = 0.90 * movement_limit
    if raw_maximum_displacement > buffered_limit:
      nonlinear_path_scale = min(
        1.0,
        buffered_limit / raw_maximum_displacement,
      )
      unscaled_samples = tuple(samples)
      for _ in range(8):
        scaled_samples: list[PrefixSample] = []
        scale_scratch = mujoco.MjData(self.handles.model)
        for sample in unscaled_samples:
          scaled_q = root_q + nonlinear_path_scale * (
            sample.joint_positions_rad - root_q
          )
          scale_scratch.qpos[self.handles.arm_qpos_adrs] = scaled_q[:7]
          scale_scratch.qpos[self.handles.hand_qpos_adrs] = scaled_q[7:]
          mujoco.mj_forward(self.handles.model, scale_scratch)
          scaled_samples.append(
            PrefixSample(
              time_s=sample.time_s,
              wrist_position_m=np.array(
                scale_scratch.site_xpos[self.handles.palm_site_id],
                dtype=np.float64,
                copy=True,
              ),
              fingertip_positions_m=np.array(
                scale_scratch.site_xpos[self.handles.tip_site_ids],
                dtype=np.float64,
                copy=True,
              ),
              joint_positions_rad=np.array(
                scaled_q,
                dtype=np.float64,
                copy=True,
              ),
            )
          )
        scaled_maximum = max(
          float(
            np.linalg.norm(
              sample.fingertip_positions_m[finger - 1]
              - root_tips[finger - 1]
            )
          )
          for sample in scaled_samples
        )
        if scaled_maximum <= buffered_limit + 1e-12:
          samples = scaled_samples
          break
        nonlinear_path_scale *= 0.80
      else:
        raise RuntimeError(
          "nonlinear finger path could not satisfy the micro-step bound"
        )
    terminal_error = float(
      np.linalg.norm(
        samples[-1].fingertip_positions_m[finger - 1] - requested
      )
    )
    realized_displacement = (
      samples[-1].fingertip_positions_m[finger - 1]
      - root_tips[finger - 1]
    )
    commanded_displacement = terminal_target - root_tips[finger - 1]
    commanded_length = float(np.linalg.norm(commanded_displacement))
    realized_length = float(np.linalg.norm(realized_displacement))
    directional_progress = 0.0
    directional_cosine = 1.0
    if commanded_length > 1e-12:
      commanded_direction = commanded_displacement / commanded_length
      directional_progress = float(
        np.dot(realized_displacement, commanded_direction)
      )
      if realized_length > 1e-12:
        directional_cosine = directional_progress / realized_length
      required_progress = min(0.00025, 0.10 * commanded_length)
      if directional_progress < required_progress:
        raise RuntimeError(
          "nonlinear finger refinement made no certified local progress "
          f"[finger={finger}, primitive={primitive_kind.value},"
          f" progress={directional_progress:.6g}m,"
          f" required={required_progress:.6g}m]"
        )
    desired_tangent = commanded_displacement - float(
      np.dot(commanded_displacement, normal)
    ) * normal
    realized_tangent = realized_displacement - float(
      np.dot(realized_displacement, normal)
    ) * normal
    desired_tangent_length = float(np.linalg.norm(desired_tangent))
    tangential_progress = 0.0
    if desired_tangent_length > 1e-12:
      tangential_progress = float(
        np.dot(realized_tangent, desired_tangent / desired_tangent_length)
      )
    if (
      primitive_kind is PrimitiveKind.SLIDE
      and desired_tangent_length > 1e-6
      and tangential_progress < min(0.00020, 0.10 * desired_tangent_length)
    ):
      raise RuntimeError(
        "nonlinear SLIDE failed to follow the local surface direction "
        f"[finger={finger}, progress={tangential_progress:.6g}m]"
      )
    contact_center = surface + support * normal
    terminal_normal_clearance = float(
      np.dot(
        (
          samples[-1].fingertip_positions_m[finger - 1]
          - root_tips[finger - 1]
          if primitive_kind is PrimitiveKind.BREAK
          else samples[-1].fingertip_positions_m[finger - 1]
          - contact_center
        ),
        normal,
      )
    )
    # A MAKE may change topology only if its exact nonlinear endpoint reaches
    # behind the zero-clearance pad-center surface.  Tangential Cartesian
    # residual alone cannot certify physical contact on a curved mesh.
    make_progress = (
      primitive_kind is PrimitiveKind.MAKE
      and terminal_normal_clearance > -0.0002
    )
    expected = actual_contact_set
    topology_change_count = 0
    if primitive_kind is PrimitiveKind.BREAK:
      expected = self.graph.apply_predictive(
        self._planner_state(data, actual_contact_set).mode,
        ContactPrimitive(PrimitiveKind.BREAK, finger),
      ).contacts
      topology_change_count = 1
    elif primitive_kind is PrimitiveKind.MAKE and not make_progress:
      expected = self.graph.apply_predictive(
        self._planner_state(data, actual_contact_set).mode,
        ContactPrimitive(PrimitiveKind.MAKE, finger),
      ).contacts
      topology_change_count = 1
    prefix = PlannedPrefix(
      prefix_id=(
        f"i04-{sequence:05d}-{primitive_kind.value.lower()}-{finger}"
      ),
      transaction_type=TransactionType.FINGER_RECONFIGURE,
      primitive_kind=primitive_kind.value,
      surface_model_version=SURFACE_MODEL_VERSION,
      root_contact_set=actual_contact_set,
      expected_terminal_contact_set=expected,
      samples=tuple(samples),
      participating_fingers=(finger,),
      anchor_fingers=tuple(
        sorted(
          actual_contact_set
          if primitive_kind is PrimitiveKind.MAKE
          or primitive_kind is PrimitiveKind.REPOSITION
          else actual_contact_set - {finger}
        )
      ),
      finger_id=finger,
      topology_change_count=topology_change_count,
      source=PrefixSource.OPTIMIZER_COMMIT_CANDIDATE,
      metadata={
        "make_progress": make_progress,
        "nonlinear_refinement": True,
        "finger_only_make": primitive_kind is PrimitiveKind.MAKE,
        "finger_only_slide": primitive_kind is PrimitiveKind.SLIDE,
        "finger_only_reposition": primitive_kind is PrimitiveKind.REPOSITION,
        "finger_only_break": primitive_kind is PrimitiveKind.BREAK,
        "target_vertex": int(target_vertex),
        "target_normal": normal.tolist(),
        "terminal_normal_clearance_m": terminal_normal_clearance,
        "commanded_micro_step_m": commanded_length,
        "realized_micro_step_m": realized_length,
        "directional_progress_m": directional_progress,
        "directional_cosine": directional_cosine,
        "surface_tangential_progress_m": tangential_progress,
        "raw_maximum_participant_displacement_m": (
          raw_maximum_displacement
        ),
        "nonlinear_path_scale": nonlinear_path_scale,
        "whole_hand_roles": _whole_hand_roles(
          actual_contact_set,
          primitive_kind,
          finger,
        ),
        "local_surface_micro_step": True,
        "micro_step_limit_m": movement_limit,
        "fresh_barrier_replan_required": True,
        "oracle_waypoint_direct_execution": False,
      },
    )
    palm_rotation = np.array(
      scratch.site_xmat[self.handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)
    terminal_pose = np.concatenate(
      (
        samples[-1].wrist_position_m,
        _quaternion_from_matrix(palm_rotation),
      )
    )
    return (
      prefix,
      terminal_pose,
      {
        "prefix_duration_s": duration,
        "prefix_translation_m": 0.0,
        "prefix_rotation_rad": 0.0,
        "nonlinear_terminal_tip_m": samples[-1].fingertip_positions_m[
          finger - 1
        ].tolist(),
        "make_terminal_error_m": terminal_error,
        "terminal_normal_clearance_m": terminal_normal_clearance,
        "commanded_micro_step_m": commanded_length,
        "realized_micro_step_m": realized_length,
        "directional_progress_m": directional_progress,
        "directional_cosine": directional_cosine,
        "surface_tangential_progress_m": tangential_progress,
        "raw_maximum_participant_displacement_m": (
          raw_maximum_displacement
        ),
        "nonlinear_path_scale": nonlinear_path_scale,
        "finger_only_make": primitive_kind is PrimitiveKind.MAKE,
        "finger_only_slide": primitive_kind is PrimitiveKind.SLIDE,
        "finger_only_reposition": primitive_kind is PrimitiveKind.REPOSITION,
        "finger_only_break": primitive_kind is PrimitiveKind.BREAK,
      },
    )

  def plan_prefix(
    self,
    data: mujoco.MjData,
    actual_contact_set: frozenset[int],
    goal_vertex: int,
    *,
    contact_positions_world_m: NDArray[np.float64],
    contact_normals_world: NDArray[np.float64],
    maximum_surface_step_m: float,
    force_wrist_adjust: bool = False,
    forced_break_finger: int | None = None,
    excluded_fingers: frozenset[int] = frozenset(),
    timestamp_s: float,
    replacement_confirmation_s: dict[int, float],
    sequence: int,
  ) -> I04PlanResult:
    if not actual_contact_set:
      raise ValueError("I04 planning requires a measured nonempty root mode")
    if forced_break_finger is not None:
      if forced_break_finger not in actual_contact_set:
        raise ValueError("forced_break_finger must be a measured contact")
      if len(actual_contact_set) < 2:
        raise ValueError("cannot BREAK the last measured contact")
    started = perf_counter()
    poses, target_vertex_by_finger, route_remaining_by_finger = (
      self._state_rooted_finger_candidates(
        data,
        actual_contact_set,
        int(goal_vertex),
        contact_positions_world_m,
        contact_normals_world,
        maximum_surface_step_m,
      )
    )
    # Preserve the unmasked physical candidates for WRIST terminal viability.
    # The finger-only branch may later set active-finger scores to infinity to
    # force redundancy recovery; leaking that search preference into M12 made
    # valid post-WRIST SLIDE successors disappear.
    wrist_viability_poses = dict(poses)
    invalid_exclusions = set(excluded_fingers) - {1, 2, 3, 4}
    if invalid_exclusions:
      raise ValueError("excluded_fingers contains an invalid finger id")
    available = [
      finger
      for finger, pose_candidate in poses.items()
      if finger not in excluded_fingers and pose_candidate.feasible
    ]
    # An execution timeout is negative evidence for the next transaction, not
    # a permanent reachability claim.  Apply it only when at least one
    # alternative survives the current measured state.
    if available:
      poses = {
        finger: (
          replace(pose_candidate, score=float("inf"))
          if finger in excluded_fingers
          else pose_candidate
        )
        for finger, pose_candidate in poses.items()
      }
    free_make_candidates = [
      finger
      for finger, pose_candidate in poses.items()
      if finger not in actual_contact_set and pose_candidate.feasible
    ]
    # Only a singleton must restore redundancy before surface traversal: it
    # cannot slide its last real anchor.  With two confirmed contacts, one
    # finger may be the explorer while the other remains the anchor.  The
    # previous <=2 rule suppressed both legal SLIDE edges and repeatedly
    # chased a third MAKE at the edge of the free fingers' workspaces.
    restore_contact_redundancy = (
      len(actual_contact_set) == 1 and bool(free_make_candidates)
    )
    stage_free_finger = False
    if restore_contact_redundancy or stage_free_finger:
      poses = {
        finger: (
          replace(pose_candidate, score=float("inf"))
          if finger in actual_contact_set
          else pose_candidate
        )
        for finger, pose_candidate in poses.items()
      }
    # Two independently confirmed contacts are sufficient for an audited
    # recenter transaction.  This is essential after singleton recovery: the
    # newly formed pair may leave both free fingers outside their local IK
    # workspaces, so another MAKE is impossible until the palm is recentered.
    use_wrist = bool(
      forced_break_finger is None
      and force_wrist_adjust
      and len(actual_contact_set) >= 2
    )
    pose: PoseCandidate | None = None
    if use_wrist:
      route_fingers = [
        finger
        for finger in sorted(actual_contact_set)
        if route_remaining_by_finger[finger] is not None
        and np.isfinite(float(route_remaining_by_finger[finger]))
      ]
      if not route_fingers:
        raise RuntimeError("WRIST_ADJUST has no measured geodesic root")
      route_finger = min(
        route_fingers,
        key=lambda finger: float(route_remaining_by_finger[finger]),
      )
      realization_target_vertex = target_vertex_by_finger[route_finger]
      target_surface = (
        self.handles.object_position_m
        + self.surface_graph.vertices_m[realization_target_vertex]
      )
      root_surface = np.asarray(contact_positions_world_m[route_finger - 1])
      root_normal = np.asarray(contact_normals_world[route_finger - 1])
      root_normal = root_normal / np.linalg.norm(root_normal)
      direction = target_surface - root_surface
      direction -= float(np.dot(direction, root_normal)) * root_normal
      direction_norm = float(np.linalg.norm(direction))
      if direction_norm <= 1e-9:
        goal_surface = (
          self.handles.object_position_m
          + self.surface_graph.vertices_m[int(goal_vertex)]
        )
        direction = goal_surface - root_surface
        direction -= float(np.dot(direction, root_normal)) * root_normal
        direction_norm = float(np.linalg.norm(direction))
      if direction_norm <= 1e-9:
        raise RuntimeError("WRIST_ADJUST geodesic tangent is degenerate")
      current_wrist = np.array(
        data.site_xpos[self.handles.palm_site_id],
        copy=True,
      )
      target_wrist, direction_evidence = self._select_wrist_target(
        data,
        actual_contact_set,
        direction / direction_norm,
        root_normal,
        sequence,
      )
      current_rotation = np.array(
        data.site_xmat[self.handles.palm_site_id],
        copy=True,
      ).reshape(3, 3)
      target_normal = np.array(
        self.surface_graph.normals[realization_target_vertex],
        copy=True,
      )
      target_normal /= np.linalg.norm(target_normal)
      aligned_rotation = _align_axis(
        current_rotation[:, 2],
        target_normal,
      ) @ current_rotation
      _, target_rotation = _clip_pose(
        current_wrist,
        current_rotation,
        target_wrist,
        aligned_rotation,
        self.config,
      )
      search_evidence = {
        **direction_evidence,
        **self._search_wrist(
          data,
          actual_contact_set,
          target_wrist,
          wrist_viability_poses,
          target_vertex_by_finger,
          sequence,
        ),
      }
      prefix, terminal_pose, refine_evidence = self._refine_wrist_adjust(
        data,
        actual_contact_set,
        contact_normals_world,
        target_wrist,
        target_rotation,
        sequence,
      )
      finger: int | None = None
      primitive = PrimitiveKind.WRIST_ADJUST
      selected_route_remaining = route_remaining_by_finger[route_finger]
    else:
      try:
        finger, primitive, search_evidence = self._search_finger(
          data,
          actual_contact_set,
          int(goal_vertex),
          poses,
          target_vertex_by_finger,
          sequence,
          forced_break_finger,
        )
      except RuntimeError as finger_error:
        if len(actual_contact_set) < 2 or forced_break_finger is not None:
          raise
        # Do not spend a 200 ms physical hold after a rejected finger edge:
        # a newly established second anchor can disappear during that wait.
        # Recenter immediately from the same measured barrier and retain the
        # same Oracle waypoint.  This recursive call can take the WRIST branch
        # only once because force_wrist_adjust=True.
        wrist_result = self.plan_prefix(
          data,
          actual_contact_set,
          int(goal_vertex),
          contact_positions_world_m=contact_positions_world_m,
          contact_normals_world=contact_normals_world,
          maximum_surface_step_m=maximum_surface_step_m,
          force_wrist_adjust=True,
          forced_break_finger=None,
          excluded_fingers=excluded_fingers,
          timestamp_s=timestamp_s,
          replacement_confirmation_s=replacement_confirmation_s,
          sequence=sequence,
        )
        wrist_result.evidence["automatic_wrist_fallback"] = True
        wrist_result.evidence["finger_edge_rejection_before_wrist"] = str(
          finger_error
        )
        wrist_result.evidence["planning_wall_latency_s"] = (
          perf_counter() - started
        )
        return wrist_result
      pose = poses[finger]
      realization_target_vertex = target_vertex_by_finger[finger]
      selected_route_remaining = route_remaining_by_finger[finger]
      if primitive in {
        PrimitiveKind.MAKE,
        PrimitiveKind.SLIDE,
        PrimitiveKind.REPOSITION,
        PrimitiveKind.BREAK,
      }:
        prefix, terminal_pose, refine_evidence = self._refine_finger_motion(
          data,
          actual_contact_set,
          pose,
          sequence,
          realization_target_vertex,
          primitive,
          contact_normals_world[finger - 1],
        )
      else:
        prefix, terminal_pose, refine_evidence = self._refine_prefix(
          data,
          actual_contact_set,
          primitive,
          pose,
          sequence,
          realization_target_vertex,
        )
    state = self._planner_state(data, actual_contact_set)
    auditor = ExactPrefixAuditor(
      self.graph,
      AuditEnvironment(
        self.surface_model,
        self._exact_kinematics,  # type: ignore[arg-type]
        link_clearance_fn=self._link_clearance,
      ),
      AuditConfig(
        audit_version="exact-prefix-audit.i04-full-bunny.v1",
        subdivisions_per_segment=self.config.audit_subdivisions,
        minimum_self_collision_clearance_m=0.0,
        minimum_link_clearance_m=0.0,
        minimum_joint_margin_rad=self.config.minimum_joint_margin_rad,
        anchor_tolerance_m=0.0025,
        # Executor Cartesian interpolation and exact nonlinear joint
        # interpolation differ slightly between sparse prefix knots.  The
        # 0.5 mm agreement bound remains far below the 4 mm pad/completion
        # scale while rejecting genuinely inconsistent trajectories.
        kinematic_consistency_tolerance_m=(
          0.0015
          if primitive is PrimitiveKind.WRIST_ADJUST
          else 0.0020
        ),
        # The final waypoint is never execution input.  M10 independently
        # enforces the same participant micro-step bound as M09/refinement.
        max_commit_displacement_m=(
          self.config.wrist_adjust_step_m
          if primitive is PrimitiveKind.WRIST_ADJUST
          else self.config.maximum_translation_per_prefix_m
        ),
      ),
    )
    audit = auditor.audit(
      AuditRequest(
        prefix=prefix,
        current_state=state,
        commit_context=CommitContext(
          actual_contact_set=actual_contact_set,
          replacement_confirmation_s=replacement_confirmation_s,
          minimum_confirmation_s=0.05,
        ),
        issued_at_s=timestamp_s,
      )
    )
    if not audit.certified or audit.certificate is None:
      raise RuntimeError(
        "M10 rejected I04 nonlinear prefix: "
        + ",".join(audit.reasons)
        + (
          f" [kin={audit.maximum_kinematic_error_m:.6g}m,"
          f" anchor={audit.maximum_anchor_error_m:.6g}m,"
          f" trust={audit.maximum_trust_displacement_m:.6g}m,"
          f" joint={audit.minimum_joint_margin_rad:.6g}rad,"
          f" link={audit.minimum_link_clearance_m:.6g}m]"
        )
      )
    root_sample = prefix.samples[0]
    terminal_sample = prefix.samples[-1]
    if primitive is PrimitiveKind.WRIST_ADJUST:
      committed_participant_displacement_m = float(
        np.linalg.norm(
          terminal_sample.wrist_position_m - root_sample.wrist_position_m
        )
      )
      committed_micro_step_limit_m = self.config.wrist_adjust_step_m
    else:
      assert finger is not None
      committed_participant_displacement_m = float(
        np.linalg.norm(
          terminal_sample.fingertip_positions_m[finger - 1]
          - root_sample.fingertip_positions_m[finger - 1]
        )
      )
      committed_micro_step_limit_m = self.config.maximum_translation_per_prefix_m
    if (
      committed_participant_displacement_m
      > committed_micro_step_limit_m + 1e-9
    ):
      raise AssertionError("M10 certified a prefix beyond the I04 micro-step limit")
    evidence = {
      **search_evidence,
      **refine_evidence,
      "m10_latency_s": audit.latency_s,
      "m10_swept_samples": audit.swept_samples,
      "m10_minimum_joint_margin_rad": audit.minimum_joint_margin_rad,
      "m10_minimum_self_clearance_m": audit.minimum_self_collision_clearance_m,
      "m10_minimum_link_clearance_m": audit.minimum_link_clearance_m,
      "m10_maximum_kinematic_error_m": audit.maximum_kinematic_error_m,
      "certificate_id": audit.certificate.certificate_id,
      "selected_finger": finger,
      "selected_primitive": primitive.value,
      "pose_yaw_offset_rad": 0.0 if pose is None else pose.yaw_offset_rad,
      "pose_position_error_m": 0.0 if pose is None else pose.position_error_m,
      "pose_orientation_error_rad": 0.0 if pose is None else pose.orientation_error_rad,
      "pose_joint_margin_rad": (
        self._exact_kinematics.joint_margin(state.joint_positions_rad)
        if pose is None
        else pose.joint_margin_rad
      ),
      "pose_target_tip_distance_m": (
        0.0 if pose is None else pose.target_tip_distance_m
      ),
      "planning_wall_latency_s": perf_counter() - started,
      "realization_target_vertex": realization_target_vertex,
      "goal_vertex": int(goal_vertex),
      "bridge_target_vertex": realization_target_vertex,
      "committed_participant_displacement_m": (
        committed_participant_displacement_m
      ),
      "committed_micro_step_limit_m": committed_micro_step_limit_m,
      "local_surface_micro_step": True,
      "fresh_barrier_replan_required": True,
      "oracle_waypoint_direct_execution": False,
      "whole_hand_roles": prefix.metadata.get("whole_hand_roles", {}),
      "selected_finger_route_remaining_m": selected_route_remaining,
      "singleton_make_recovery": (
        len(actual_contact_set) == 1 and primitive is PrimitiveKind.MAKE
      ),
      "wrist_adjust_forced": use_wrist,
      "contact_redundancy_make_forced": restore_contact_redundancy,
      "free_finger_reposition_forced": stage_free_finger,
      "forced_workspace_break": forced_break_finger is not None,
    }
    return I04PlanResult(
      prefix=prefix,
      certificate=audit.certificate,
      selected_finger=finger,
      selected_primitive=primitive.value,
      target_vertex=realization_target_vertex,
      target_pose_world=terminal_pose,
      evidence=evidence,
    )
