"""Validate and deploy the fingertip DP in the object-fixed inverse environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from active_capsule_palm_planner import (
    ActiveCapsulePalmPlanner,
    ActiveCapsulePalmPlannerConfig,
)
from dp_dataset import ACTION_DIM, ENV_STATE_DIM, ROBOT_STATE_DIM
from dp_motion_features import (
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    MOTION_SCHEMA,
    MOTION_SCHEMAS,
    TASK_EST_V4_SCHEMA,
    causal_motion_features,
    kinematic_q_baseline,
)
from dp_chunk_scheduler import DPChunkScheduler, DPChunkSchedulerConfig
from fingertip_impedance import (
    FingertipImpedanceConfig,
    FingertipImpedanceController,
)
from palm_planner_features import future_palm_delta_pose_palm
from object_catalog import MeshNormalOracle, ObjectConfig, load_object_config
from surface_manifold_gp import GPManifoldConfig, local_gp_point_features
from train_surface_pointnet import SurfacePointNet
from replay_inverted import (
    CONTACT_SOLIMP,
    CONTACT_SOLREF,
    MCC_TIP_NAMES,
    REPLAY_PHYSICS_SUBSTEPS,
    replay_env_cfg,
)
from surface_mcc_finger import (
    FullHandMCCFingerConfig,
    FullHandMCCFingerController,
    PrivilegedCapsuleSurfaceOracle,
)
from train_dp import build_policy


Mode = Literal["offline_teacher", "teacher_dp", "live_dp"]
ACTION_SCALE = 0.08
GEOMETRY_STATE_SCHEMAS = (
    "contact_geometry",
    "contact_geometry_planner",
    "contact_geometry_planner_manifold",
    MOTION_SCHEMA,
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    TASK_EST_V4_SCHEMA,
)
PLANNER_STATE_SCHEMAS = (
    "contact_geometry_planner",
    "contact_geometry_planner_manifold",
    MOTION_SCHEMA,
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    TASK_EST_V4_SCHEMA,
)


@dataclass
class ContactAwareReplanConfig:
    """Keep live DP causal while preventing contact loss from poisoning history."""

    min_fingers: int = 3
    force_threshold: float = 0.05
    bad_grace_steps: int = 5


@dataclass(frozen=True)
class ReplayObjectMetadata:
    """Physical replay object inferred from the trajectory H5 contract."""

    object_id: str | None
    object_scale: float


def load_replay_object_metadata(path: Path) -> ReplayObjectMetadata:
    """Read only environment-construction metadata, never controller inputs."""

    with h5py.File(path, "r") as file:
        object_id = str(
            file.attrs.get(
                "object_id",
                file.attrs.get("planner_object_id", ""),
            )
        ).strip()
        object_scale = float(
            file.attrs.get(
                "object_scale",
                file.attrs.get("planner_object_scale", 1.0),
            )
        )
    if object_scale <= 0.0:
        raise ValueError(f"Invalid object_scale={object_scale} in {path}")
    return ReplayObjectMetadata(object_id or None, object_scale)


def audit_collection_execution_contract(
    path: Path, args: argparse.Namespace
) -> None:
    """Fail fast when a collection-matched run is not physically matched."""

    if args.mcc_preset != "collection_matched_sensor":
        print(
            "[ALIGNMENT] mcc_preset=current: controller/physics transfer "
            "is intentional and is not a collection-matched result"
        )
        return
    with h5py.File(path, "r") as file:
        attrs = file.attrs
        checks = (
            ("contact_stiffness", -CONTACT_SOLREF[0]),
            ("contact_damping", -CONTACT_SOLREF[1]),
            ("contact_transition_width_m", CONTACT_SOLIMP[2]),
            ("physics_substeps", REPLAY_PHYSICS_SUBSTEPS),
            ("fullhand_finger_stiffness", args.hand_servo_stiffness),
            ("fullhand_finger_damping", args.hand_servo_damping),
            ("fullhand_finger_effort_limit", args.hand_servo_effort_limit),
            ("fullhand_force_servo_gain", args.mcc_force_servo_integral_gain),
        )
        mismatches: list[str] = []
        for name, deployment_value in checks:
            if name not in attrs:
                continue
            collection_value = float(attrs[name])
            if not np.isclose(
                collection_value,
                float(deployment_value),
                rtol=1.0e-6,
                atol=1.0e-9,
            ):
                mismatches.append(
                    f"{name}: collection={collection_value:g}, "
                    f"deployment={float(deployment_value):g}"
                )
    if mismatches:
        raise RuntimeError(
            "collection_matched_sensor contract mismatch:\n  "
            + "\n  ".join(mismatches)
        )
    print(
        "[ALIGNMENT] collection physics matched: "
        f"solref={CONTACT_SOLREF} solimp_width={CONTACT_SOLIMP[2]:g}m "
        f"substeps={REPLAY_PHYSICS_SUBSTEPS} finger_servo="
        f"{args.hand_servo_stiffness:g}/{args.hand_servo_damping:g}/"
        f"{args.hand_servo_effort_limit:g}"
    )


@dataclass(frozen=True)
class MCCPrecontactConfig:
    """Per-finger Cartesian contact search copied from FullHandMCC."""

    force_threshold: float = 0.10
    desired_force_per_finger: tuple[float, float, float, float] | None = None
    settle_frames: int = 3
    cartesian_step_m: float = 0.00015
    joint_step_rad: float = 0.02
    joint_limit_rad: float = 0.30
    servo_load_scale: float = 0.0
    trajectory_tracking_gain: float = 0.0
    runtime_loss_frames: int = 5
    sensor_normal_memory_frames: int = 20
    recovery_confirm_frames: int = 3
    runtime_recovery_limit_rad: float = 0.08
    command_rate_limit_rad: float = 0.02
    command_ema_alpha: float = 0.65
    recovery_offset_decay: float = 0.999
    recovery_decay_force_ratio: float = 1.5
    # A geometry hit with only a tiny force is not a settled contact.  Such a
    # finger must continue the sensor-driven Jacobian precontact search.
    settle_force_ratio: float = 0.80
    # Without an analytic surface projection, the tactile loop must absorb
    # both force error and the site-to-contact geometric residual. ``None``
    # retains the controller's mode-specific fallback; the CLI validated
    # default supplies 20 mm explicitly.
    max_normal_offset_m: float | None = None
    thumb_max_inward_offset_m: float | None = None
    thumb_max_outward_offset_m: float | None = None
    posture_cost: float = 0.08
    nominal_surface_preload_m: float = 0.0
    flexion_synergy_gain: float = 0.0
    flexion_synergy_hard_gain: float = 0.0
    flexion_synergy_max_step_rad: float = 0.03
    normal_synergy_control: bool = False
    normal_synergy_max_step_rad: float = 0.035
    force_magnitude_only: bool = False
    use_contact_point_jacobian: bool = True
    project_nominal_normal_motion: bool = False
    overforce_trigger_ratio: float = 1.20
    overforce_release_ratio: float = 0.90
    overforce_hard_ratio: float = 1.4
    thumb_overforce_hard_ratio: float | None = 1.8
    overforce_retreat_step_m: float = 0.00008
    overforce_recovery_step_m: float = 0.00002
    overforce_max_offset_m: float = 0.020
    virtual_stiffness: float = 0.0
    max_normal_speed_m_s: float = 0.020
    max_normal_acceleration_m_s2: float = 2.0
    use_direct_force_servo: bool = True
    force_servo_integral_gain: float = 0.005
    force_servo_deadband: float = 0.05
    force_servo_max_step_m: float = 0.00008
    force_servo_hard_step_m: float = 0.00020
    thumb_force_servo_hard_step_m: float | None = 0.00010
    force_servo_search_step_m: float = 0.00050
    thumb_force_servo_search_step_m: float | None = 0.00025
    force_servo_weak_contact_step_m: float = 0.00020
    force_filter_alpha: float = 0.25
    enable_loss_state_machine: bool = False
    transient_loss_frames: int = 6
    transient_search_step_m: float = 0.00020
    transient_release_step_m: float = 0.00010
    # The collection-side controller was built with a distal-flexion floor
    # (-0.30 rad under manifold/inverse planning, -0.10 otherwise) so the
    # differential surface planner could unfold fingers without taking a
    # folded IK branch.  Deployment must clamp on the same floor or the
    # DP-intended unfolding can travel further than the training domain.
    natural_flexion_floor: float | None = None


def _episode(file: h5py.File, episode_id: int, name: str) -> np.ndarray:
    ids = np.asarray(file["episode_id"], dtype=np.int64)
    locations = np.argwhere(ids == episode_id)
    if not locations.size:
        available = np.unique(ids)
        raise ValueError(
            f"episode_id={episode_id} not found; available IDs include "
            f"{available[:20].tolist()}"
        )
    steps = np.asarray(file["episode_step"])
    order = np.argsort(
        np.asarray([steps[t, e] for t, e in locations], dtype=np.int64)
    )
    locations = locations[order]
    dataset = file[name]
    return np.stack(
        [dataset[t, e] for t, e in locations], axis=0
    ).astype(np.float32)


def load_episode(
    path: Path,
    episode_id: int,
    *,
    include_teacher_tactile: bool,
) -> dict[str, np.ndarray]:
    # Live deployment must not even load recorded fingertip force/normal/pose
    # channels.  Those channels are retained only for the explicit
    # ``teacher_dp`` and ``offline_teacher`` evaluation modes.
    required = ["palm_pose_object", "q_hand", "palm_twist_object"]
    if include_teacher_tactile:
        required.extend(
            (
                "fingertip_force_object",
                "fingertip_contact_normal_object",
                "fingertip_pose_object",
            )
        )
    with h5py.File(path, "r") as file:
        names = tuple(required)
        if include_teacher_tactile:
            names += tuple(
                name
                for name in (
                    "fingertip_contact_pos_object",
                    "fingertip_contact",
                )
                if name in file
            )
        return {name: _episode(file, episode_id, name) for name in names}


class DPRuntime:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        inference_steps: int | None,
        seed: int,
        samples: int = 1,
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = dict(checkpoint["config"])
        if inference_steps is not None:
            config["inference_steps"] = inference_steps
        self.config = SimpleNamespace(**config)
        self.policy = build_policy(self.config, device)
        self.policy.load_state_dict(checkpoint["model"])
        self.policy.eval()
        self.policy.diffusion.num_inference_steps = int(
            config["inference_steps"]
        )
        self.input_frame = str(checkpoint.get("input_frame", "object"))
        if self.input_frame not in ("object", "palm"):
            raise ValueError(
                f"Unsupported checkpoint input_frame={self.input_frame!r}"
            )
        self.action_representation = str(
            checkpoint.get(
                "action_representation",
                config.get("action_representation", "delta_q"),
            )
        )
        if self.action_representation not in (
            "delta_q",
            "absolute_q",
            "kinematic_residual_q",
        ):
            raise ValueError(
                "Unsupported checkpoint action_representation="
                f"{self.action_representation!r}"
            )
        self.action_dim = int(
            checkpoint.get(
                "action_dim",
                config.get("action_dim", ACTION_DIM),
            )
        )
        self.action_field = str(
            checkpoint.get(
                "action_field",
                config.get("action_field", "q_hand"),
            )
        )
        if self.action_field not in (
            "q_hand",
            "q_ref",
            "tip_delta_tangent_palm",
            "tip_motion_tangent_palm",
            "tip_motion_palm",
            "tip_target_palm",
        ):
            raise ValueError(
                "Unsupported checkpoint action_field="
                f"{self.action_field!r}"
            )
        if self.action_dim == 12:
            if self.action_field not in (
                "tip_delta_tangent_palm",
                "tip_motion_tangent_palm",
                "tip_motion_palm",
                "tip_target_palm",
            ):
                raise ValueError(
                    "12D action_dim requires a supported fingertip action; "
                    f"checkpoint has {self.action_field!r}"
                )
        elif self.action_dim != ACTION_DIM:
            raise ValueError(
                f"Unsupported checkpoint action_dim={self.action_dim}"
            )
        self.state_schema = str(
            checkpoint.get(
                "state_schema",
                config.get("state_schema", "force_normal"),
            )
        )
        if self.state_schema not in ("force_normal", *GEOMETRY_STATE_SCHEMAS):
            raise ValueError(f"Unsupported state_schema={self.state_schema!r}")
        self.robot_state_dim = int(
            checkpoint.get(
                "robot_state_dim",
                config.get("robot_state_dim", ROBOT_STATE_DIM),
            )
        )
        self.environment_state_dim = int(
            checkpoint.get(
                "environment_state_dim",
                config.get("environment_state_dim", ENV_STATE_DIM),
            )
        )
        self.state_dim = self.robot_state_dim + self.environment_state_dim
        self.contact_normal_polarity = str(
            checkpoint.get("contact_normal_polarity", "")
        )
        if not self.contact_normal_polarity:
            dataset_path = Path(str(config.get("file", "")))
            if dataset_path.is_file():
                with h5py.File(dataset_path, "r") as dataset_file:
                    self.contact_normal_polarity = str(
                        dataset_file.attrs.get(
                            "contact_normal_polarity",
                            "primary_fingertip_to_object",
                        )
                    )
            else:
                # Legacy capsule checkpoints were trained directly from the
                # ContactSensor primary->secondary normal.
                self.contact_normal_polarity = "primary_fingertip_to_object"
        if self.contact_normal_polarity not in (
            "primary_fingertip_to_object",
            "source_mesh_outward",
            "analytic_outward",
        ):
            raise ValueError(
                "Unsupported checkpoint contact_normal_polarity="
                f"{self.contact_normal_polarity!r}"
            )
        normalization = checkpoint["normalization"]
        self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(normalization["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(
            normalization["action_mean"], dtype=np.float32
        )
        self.action_std = np.asarray(
            normalization["action_std"], dtype=np.float32
        )
        self.state_input_mask = np.asarray(
            checkpoint.get("state_input_mask", np.ones(self.state_dim)),
            dtype=np.float32,
        ).reshape(-1)
        if self.state_mean.shape != (self.state_dim,):
            raise ValueError(
                f"Checkpoint state dim {self.state_mean.shape} "
                f"!= {(self.state_dim,)}"
            )
        if self.state_input_mask.shape != (self.state_dim,):
            raise ValueError(
                f"Checkpoint input mask {self.state_input_mask.shape} "
                f"!= {(self.state_dim,)}"
            )
        self.device = device
        self.samples = int(samples)
        if self.samples < 1:
            raise ValueError(f"samples must be >= 1, got {self.samples}")
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.inference_seconds: list[float] = []
        self.surface_pointnet = None
        self.surface_point_mean = None
        self.surface_point_std = None
        self.surface_gp_config = None
        if self.state_schema == "contact_geometry_planner_manifold":
            dataset_path = Path(config["file"])
            with h5py.File(dataset_path, "r") as dataset_file:
                pointnet_path = Path(
                    str(dataset_file.attrs["surface_manifold_pointnet"])
                )
            pointnet_checkpoint = torch.load(
                pointnet_path, map_location="cpu", weights_only=False
            )
            self.surface_pointnet = SurfacePointNet(
                latent_dim=int(pointnet_checkpoint["config"]["latent_dim"])
            ).to(device)
            self.surface_pointnet.load_state_dict(pointnet_checkpoint["model"])
            self.surface_pointnet.eval()
            self.surface_point_mean = np.asarray(
                pointnet_checkpoint["point_mean"], dtype=np.float32
            )
            self.surface_point_std = np.asarray(
                pointnet_checkpoint["point_std"], dtype=np.float32
            )
            self.surface_gp_config = GPManifoldConfig(
                **json.loads(pointnet_checkpoint["gp_config"])
            )

    @property
    def stride(self) -> int:
        return int(self.config.stride)

    @property
    def obs_horizon(self) -> int:
        return int(self.config.obs_horizon)

    @property
    def pred_horizon(self) -> int:
        return int(self.config.pred_horizon)

    @property
    def planner_waypoints(self) -> int:
        return int(getattr(self.config, "planner_waypoints", 0))

    @property
    def planner_step_frames(self) -> int:
        return int(getattr(self.config, "planner_step_frames", 0))

    @property
    def motion_feature_step_frames(self) -> int:
        return int(getattr(self.config, "motion_feature_step_frames", self.stride))

    @property
    def action_waypoint_dt(self) -> float:
        return float(getattr(self.config, "action_waypoint_dt", self.stride * 0.01))

    @property
    def kinematic_velocity_clip(self) -> float:
        return float(getattr(self.config, "kinematic_velocity_clip_rad_s", 1.0))

    def absolute_action(
        self,
        prediction: np.ndarray,
        current_q: np.ndarray,
        current_q_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reconstruct the absolute-q chunk for every action representation."""
        if self.action_representation == "absolute_q":
            return np.asarray(prediction, dtype=np.float32)
        if self.action_representation == "delta_q":
            return np.asarray(current_q, dtype=np.float32)[None, :] + prediction
        if current_q_velocity is None:
            raise ValueError("kinematic residual reconstruction requires q velocity")
        baseline = kinematic_q_baseline(
            current_q,
            current_q_velocity,
            self.pred_horizon,
            self.action_waypoint_dt,
            self.kinematic_velocity_clip,
        )
        return baseline + prediction

    @torch.no_grad()
    def encode_surface_manifold(self, points: np.ndarray) -> np.ndarray:
        if self.surface_pointnet is None:
            raise RuntimeError("This checkpoint has no surface PointNet")
        normalized = (points - self.surface_point_mean) / self.surface_point_std
        value = torch.as_tensor(
            normalized, device=self.device, dtype=torch.float32
        )
        if value.ndim == 3:
            value = value.unsqueeze(0)
        return self.surface_pointnet.encode(value).cpu().numpy()

    @torch.no_grad()
    def predict(self, history: np.ndarray) -> np.ndarray:
        if history.shape != (self.obs_horizon, self.state_dim):
            raise ValueError(
                f"history shape {history.shape} != "
                f"{(self.obs_horizon, self.state_dim)}"
            )
        normalized = (
            (history - self.state_mean) / self.state_std
        ) * self.state_input_mask
        state = torch.as_tensor(
            normalized[:, : self.robot_state_dim],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(self.samples, -1, -1)
        environment = torch.as_tensor(
            normalized[:, self.robot_state_dim :],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(self.samples, -1, -1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        global_condition = self.policy.diffusion._prepare_global_conditioning(
            {
                "observation.state": state,
                "observation.environment_state": environment,
            }
        )
        prediction = self.policy.diffusion.conditional_sample(
            self.samples,
            global_cond=global_condition,
            generator=self.generator,
        ).mean(dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.inference_seconds.append(time.perf_counter() - start)
        normalized_action = prediction.detach().cpu().numpy()
        return (
            normalized_action * self.action_std[None, :]
            + self.action_mean[None, :]
        ).astype(np.float32)


def _wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping palm-frame vectors into the object frame."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix.astype(np.float32)


def _vectors_object_to_palm(
    vectors: np.ndarray, palm_quaternion_object_wxyz: np.ndarray
) -> np.ndarray:
    palm_from_object = np.swapaxes(
        _wxyz_to_matrix(palm_quaternion_object_wxyz), -1, -2
    )
    if vectors.ndim == palm_from_object.ndim - 1:
        return np.einsum("...ij,...j->...i", palm_from_object, vectors)
    return np.einsum("...ij,...fj->...fi", palm_from_object, vectors)


def _points_object_to_palm(
    points: np.ndarray, palm_pose_object: np.ndarray
) -> np.ndarray:
    palm_from_object = np.swapaxes(
        _wxyz_to_matrix(palm_pose_object[..., 3:7]), -1, -2
    )
    return np.einsum(
        "...ij,...fj->...fi",
        palm_from_object,
        points - palm_pose_object[..., None, :3],
    )


def _causal_surface_gp_points(
    positions_object: np.ndarray,
    normals_object: np.ndarray,
    masks: np.ndarray,
    palm_pose_object: np.ndarray,
    current: int,
    stride: int,
    config: GPManifoldConfig,
) -> np.ndarray:
    indices = current - stride * np.arange(
        config.history_steps - 1, -1, -1, dtype=np.int64
    )
    indices = np.maximum(indices, 0)
    current_pose = palm_pose_object[current]
    positions_palm = _points_object_to_palm(
        positions_object[indices],
        np.broadcast_to(current_pose, (len(indices), 7)),
    )
    normals_palm = _vectors_object_to_palm(
        normals_object[indices],
        np.broadcast_to(current_pose[3:7], (len(indices), 4)),
    )
    return np.stack(
        [
            local_gp_point_features(
                positions_palm[:, finger],
                normals_palm[:, finger],
                masks[indices, finger] > 0.5,
                config,
            )
            for finger in range(4)
        ],
        axis=0,
    )


def causal_surface_embeddings(
    data: dict[str, np.ndarray], runtime: DPRuntime
) -> np.ndarray:
    if runtime.surface_gp_config is None:
        raise RuntimeError("Surface GP configuration is unavailable")
    point_sets = np.stack(
        [
            _causal_surface_gp_points(
                data["fingertip_contact_pos_object"],
                data["fingertip_contact_normal_object"],
                data["fingertip_contact"],
                data["palm_pose_object"],
                current,
                runtime.stride,
                runtime.surface_gp_config,
            )
            for current in range(len(data["q_hand"]))
        ],
        axis=0,
    )
    chunks = []
    for start in range(0, len(point_sets), 512):
        chunks.append(runtime.encode_surface_manifold(point_sets[start : start + 512]))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def teacher_state(
    data: dict[str, np.ndarray],
    input_frame: str = "object",
    state_schema: str = "force_normal",
    planner_waypoints: int = 0,
    planner_step_frames: int = 0,
    surface_embeddings: np.ndarray | None = None,
    motion_feature_step_frames: int = 5,
    control_dt: float = 0.01,
) -> np.ndarray:
    if state_schema == DUAL_TRACK_V3_SCHEMA:
        if "teacher_state_dual_track" not in data:
            raise ValueError("Dual-track teacher evaluation requires the aligned training observation fields")
        return data["teacher_state_dual_track"]
    normals = data["fingertip_contact_normal_object"]
    twist = data["palm_twist_object"]
    if state_schema == "force_normal":
        tactile = data["fingertip_force_object"]
        contact_mask = None
    elif state_schema in GEOMETRY_STATE_SCHEMAS:
        tactile = data["fingertip_contact_pos_object"]
        contact_mask = data["fingertip_contact"]
    else:
        raise ValueError(f"Unsupported state_schema={state_schema!r}")
    if input_frame == "palm":
        palm_pose = data["palm_pose_object"]
        palm_quaternion = palm_pose[:, 3:7]
        tactile = (
            _vectors_object_to_palm(tactile, palm_quaternion)
            if state_schema == "force_normal"
            else _points_object_to_palm(tactile, palm_pose)
        )
        normals = _vectors_object_to_palm(normals, palm_quaternion)
        twist = np.concatenate(
            (
                _vectors_object_to_palm(twist[:, :3], palm_quaternion),
                _vectors_object_to_palm(twist[:, 3:], palm_quaternion),
            ),
            axis=-1,
        )
    if contact_mask is not None:
        tactile = tactile.copy()
        normals = normals.copy()
        valid = contact_mask > 0.5
        for finger in range(4):
            valid_indices = np.flatnonzero(valid[:, finger])
            if not len(valid_indices):
                continue
            last = int(valid_indices[0])
            tactile[:last, finger] = tactile[last, finger]
            normals[:last, finger] = normals[last, finger]
            for index in range(last + 1, len(tactile)):
                if valid[index, finger]:
                    last = index
                else:
                    tactile[index, finger] = tactile[last, finger]
                    normals[index, finger] = normals[last, finger]
    motion_parts: list[np.ndarray] = []
    if state_schema in (MOTION_SCHEMA, TASK_EST_V4_SCHEMA):
        q_velocity, point_velocity, normal_rate = causal_motion_features(
            data["q_hand"],
            tactile,
            normals,
            contact_mask,
            np.zeros(len(data["q_hand"]), dtype=np.int32),
            control_dt=control_dt,
            step_frames=motion_feature_step_frames,
        )
        motion_parts = [
            q_velocity.reshape(-1, 16),
            point_velocity.reshape(-1, 12),
            normal_rate.reshape(-1, 12),
        ]
    parts = [
        data["q_hand"],
        tactile.reshape(-1, 12),
        normals.reshape(-1, 12),
    ]
    if contact_mask is not None:
        parts.append(contact_mask.reshape(-1, 4))
    parts.extend(motion_parts)
    parts.append(twist.reshape(-1, 6))
    if state_schema in PLANNER_STATE_SCHEMAS:
        if planner_waypoints <= 0 or planner_step_frames <= 0:
            raise ValueError("Planner-conditioned state requires planner metadata")
        planner = future_palm_delta_pose_palm(
            data["palm_pose_object"],
            np.zeros(len(data["palm_pose_object"]), dtype=np.int32),
            waypoint_count=planner_waypoints,
            step_frames=planner_step_frames,
        )
        parts.append(planner.reshape(len(planner), -1))
    if state_schema == "contact_geometry_planner_manifold":
        if surface_embeddings is None:
            raise ValueError("Manifold state requires causal PointNet embeddings")
        parts.append(np.asarray(surface_embeddings, dtype=np.float32))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def history_indices(t: int, stride: int, horizon: int) -> np.ndarray:
    return t - stride * np.arange(horizon - 1, -1, -1)


def tip_delta_to_absolute_q(
    controller: FullHandMCCFingerController,
    prediction: np.ndarray,
    q_base: np.ndarray,
    pred_horizon: int,
    *,
    per_waypoint_increment: bool = False,
) -> np.ndarray:
    """Rebuild a 16D absolute-q chunk from the 12D tangent-intent output
    (Variant B, action_field='tip_delta_tangent_palm').

    Each waypoint carries the per-finger fingertip displacement the policy
    wants at that future step relative to the current executed tip, with the
    normal component removed (the deployment MCC force loop owns normal
    approach).  The absolute command is

        tip_des[h] = FK(q_base) + delta[h]
        q_des[h]   = solve_fingertip_targets(tip_des[h], seed=q_des[h-1])

    Seeding each solve from the previous waypoint makes the resolved-rate IK
    act as a temporal branch selector, the same convention used offline
    (seed_q = preceding trajectory frame).
    """
    deltas = np.asarray(prediction, dtype=np.float64).reshape(
        pred_horizon, 4, 3
    )
    # The task-motion fields store displacement between consecutive DP
    # waypoints. Reconstruct the H-step path by integration; the legacy v2
    # tip-delta label is already interpreted as an offset from one base.
    offsets = np.cumsum(deltas, axis=0) if per_waypoint_increment else deltas
    tip_base = controller.tip_positions_palm(q_base)
    seed = np.asarray(q_base, dtype=np.float64).reshape(16)
    chunk = np.empty((pred_horizon, 16), dtype=np.float32)
    for waypoint in range(pred_horizon):
        seed, _, _ = controller.solve_fingertip_targets(
            tip_base + offsets[waypoint],
            seed_q=seed,
        )
        chunk[waypoint] = seed.astype(np.float32)
    return chunk


def tip_target_to_absolute_q(
    controller: FullHandMCCFingerController,
    prediction: np.ndarray,
    q_base: np.ndarray,
    pred_horizon: int,
) -> np.ndarray:
    """Resolve absolute palm-frame fingertip targets without accumulation."""
    targets = np.asarray(prediction, dtype=np.float64).reshape(
        pred_horizon, 4, 3
    )
    nominal = np.asarray(q_base, dtype=np.float64).reshape(16)
    seed = nominal.copy()
    chunk = np.empty((pred_horizon, 16), dtype=np.float32)
    for waypoint in range(pred_horizon):
        seed, _, _ = controller.solve_fingertip_targets(
            targets[waypoint],
            seed_q=seed,
            nominal_q=nominal,
        )
        chunk[waypoint] = seed.astype(np.float32)
    return chunk


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_closed_loop_rollout(
    path: Path,
    arrays: dict[str, list[np.ndarray]],
    *,
    source_file: Path,
    source_episode_id: int,
    mode: str,
    teacher_action_source: str,
    control_dt: float,
    bootstrap_frames: int,
    input_frame: str,
    state_schema: str,
    stride: int,
    obs_horizon: int,
    pred_horizon: int,
    dp_history_q_source: str,
    teacher_observation_source: str,
    dp_tactile_normal_source: str,
    live_teacher_takeover_frame: int,
) -> None:
    """Write causally aligned deployment observations for later relabelling."""

    if not arrays:
        return
    lengths = {key: len(values) for key, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Rollout fields have inconsistent lengths: {lengths}")
    with h5py.File(source_file, "r") as source:
        source_metadata = {
            key: source.attrs[key]
            for key in (
                "object_id",
                "object_scale",
                "contact_normal_polarity",
            )
            if key in source.attrs
        }
        if "object_scale" not in source_metadata:
            source_metadata["object_scale"] = float(
                source.attrs.get("planner_object_scale", 1.0)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for key, values in arrays.items():
            data = np.stack(values)
            data = data.astype(
                np.int32 if key == "episode_step" else np.float32
            )
            file.create_dataset(
                key,
                data=data,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        length = next(iter(lengths.values()))
        file.create_dataset(
            "episode_id",
            data=np.full((length,), source_episode_id, dtype=np.int32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        file.attrs["format"] = "mcc_closed_loop_observation_v1"
        file.attrs["source_file"] = str(source_file)
        file.attrs["source_episode_id"] = int(source_episode_id)
        file.attrs["mode"] = mode
        file.attrs["teacher_action_source"] = teacher_action_source
        file.attrs["control_dt"] = float(control_dt)
        file.attrs["bootstrap_frames"] = int(bootstrap_frames)
        file.attrs["input_frame"] = input_frame
        file.attrs["state_schema"] = state_schema
        file.attrs["stride"] = int(stride)
        file.attrs["obs_horizon"] = int(obs_horizon)
        file.attrs["pred_horizon"] = int(pred_horizon)
        file.attrs["dp_history_q_source"] = str(dp_history_q_source)
        file.attrs["teacher_observation_source"] = str(
            teacher_observation_source
        )
        file.attrs["dp_tactile_normal_source"] = str(
            dp_tactile_normal_source
        )
        file.attrs["live_teacher_takeover_frame"] = int(
            live_teacher_takeover_frame
        )
        for key, value in source_metadata.items():
            file.attrs[key] = value
        file.attrs["alignment"] = (
            "q_live/contact at t follow q_cmd_applied/q_prior_applied from t-1; "
            "q_prior_next/q_cmd_next are commands computed at t"
        )
        file.attrs["action_label"] = "teacher_q_hand; future labels use source time"


def offline_teacher(
    data: dict[str, np.ndarray],
    runtime: DPRuntime,
    max_dp_calls: int,
    report: Path,
) -> None:
    surface_embeddings = (
        causal_surface_embeddings(data, runtime)
        if runtime.state_schema == "contact_geometry_planner_manifold"
        else None
    )
    state = teacher_state(
        data,
        runtime.input_frame,
        runtime.state_schema,
        runtime.planner_waypoints,
        runtime.planner_step_frames,
        surface_embeddings,
        runtime.motion_feature_step_frames,
        float(getattr(runtime.config, "control_dt", 0.01)),
    )
    q = data["q_hand"]
    first = (runtime.obs_horizon - 1) * runtime.stride
    last = len(q) - runtime.pred_horizon * runtime.stride - 1
    rows: list[dict[str, float | int | str]] = []
    variant_b = runtime.action_field in (
        "tip_delta_tangent_palm",
        "tip_motion_tangent_palm",
        "tip_motion_palm",
        "tip_target_palm",
    )
    # Offline FK+IK rebuild uses the same fixed-hand offline model as the
    # dual-track exporter, not the physics-hand controller.
    variant_b_controller = (
        FullHandMCCFingerController() if variant_b else None
    )
    for call, t in enumerate(range(first, last + 1, runtime.stride), start=1):
        if max_dp_calls > 0 and call > max_dp_calls:
            break
        prediction = runtime.predict(
            state[history_indices(t, runtime.stride, runtime.obs_horizon)]
        )
        target_indices = t + runtime.stride * np.arange(
            1, runtime.pred_horizon + 1
        )
        teacher_future = q[target_indices]
        if variant_b:
            if runtime.action_field == "tip_target_palm":
                predicted_future = tip_target_to_absolute_q(
                    variant_b_controller,
                    prediction,
                    q[t],
                    runtime.pred_horizon,
                )
            else:
                predicted_future = tip_delta_to_absolute_q(
                    variant_b_controller,
                    prediction,
                    q[t],
                    runtime.pred_horizon,
                    per_waypoint_increment=(
                        runtime.action_field in (
                            "tip_motion_tangent_palm",
                            "tip_motion_palm",
                        )
                    ),
                )
        else:
            predicted_future = runtime.absolute_action(
                prediction,
                q[t],
                (
                    state[t, 44:60]
                    if runtime.action_representation == "kinematic_residual_q"
                    else None
                ),
            )
        error = predicted_future - teacher_future
        hold_error = q[t][None, :] - teacher_future
        rows.append(
            {
                "mode": "offline_teacher",
                "call": call,
                "frame": t,
                "horizon_mae_rad": float(np.abs(error).mean()),
                "first_step_mae_rad": float(np.abs(error[0]).mean()),
                "final_step_mae_rad": float(np.abs(error[-1]).mean()),
                "hold_q_mae_rad": float(np.abs(hold_error).mean()),
            }
        )
        if call == 1 or call % 25 == 0:
            print(
                f"[OFFLINE] call={call:4d} frame={t:4d} "
                f"first={rows[-1]['first_step_mae_rad']:.6f}rad "
                f"horizon={rows[-1]['horizon_mae_rad']:.6f}rad"
            )
    write_report(report, rows)
    values = np.asarray([row["horizon_mae_rad"] for row in rows], dtype=float)
    first_values = np.asarray(
        [row["first_step_mae_rad"] for row in rows], dtype=float
    )
    baseline = np.asarray([row["hold_q_mae_rad"] for row in rows], dtype=float)
    print(
        f"[RESULT] mode=offline_teacher calls={len(rows)} "
        f"horizon_mae={values.mean():.6f}rad "
        f"first_step_mae={first_values.mean():.6f}rad "
        f"hold_q={baseline.mean():.6f}rad report={report}"
    )


def live_tip_observation(
    env: ManagerBasedRlEnv,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forces = np.zeros((4, 3), dtype=np.float32)
    normals = np.zeros((4, 3), dtype=np.float32)
    positions = np.zeros((4, 3), dtype=np.float32)
    loaded = np.zeros(4, dtype=bool)
    distances = np.zeros(4, dtype=np.float32)
    for tip_index, site_name in enumerate(MCC_TIP_NAMES):
        sensor = env.scene[f"{site_name}_contact"]
        if not isinstance(sensor, ContactSensor):
            raise TypeError(type(sensor))
        sensor.update(0.0)
        sensor_data = sensor.data
        contact_found = (
            sensor_data.found is not None
            and bool((sensor_data.found[0] > 0).any())
        )
        if contact_found:
            found_slot = sensor_data.found[0] > 0
            slot = int(torch.nonzero(found_slot, as_tuple=False)[0, 0])
            loaded[tip_index] = True
        else:
            continue
        if sensor_data.force is not None:
            slot_force = sensor_data.force[0]
            forces[tip_index] = (
                torch.where(
                    found_slot[:, None],
                    slot_force,
                    torch.zeros_like(slot_force),
                )
                .sum(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
        if sensor_data.normal is not None:
            normals[tip_index] = (
                sensor_data.normal[0, slot].detach().cpu().numpy()
            )
        if sensor_data.pos is not None:
            positions[tip_index] = (
                sensor_data.pos[0, slot].detach().cpu().numpy()
            )
        if sensor_data.dist is not None:
            distances[tip_index] = float(sensor_data.dist[0, slot])
    return forces, normals, positions, loaded, distances


def run_inverse(
    data: dict[str, np.ndarray],
    runtime: DPRuntime,
    mode: Literal["teacher_dp", "live_dp"],
    viewer: Literal["headless", "native", "viser", "video"],
    device: torch.device,
    max_steps: int,
    max_dp_calls: int,
    contact_threshold: float,
    impedance_config: FingertipImpedanceConfig | None,
    chunk_config: DPChunkSchedulerConfig | None,
    contact_guard_config: ContactAwareReplanConfig | None,
    execution_layer: Literal["joint_position", "fullhand_mcc"],
    mcc_direction_source: Literal[
        "oracle", "sensor_normal", "grasp_closure", "hybrid"
    ],
    mcc_desired_force: float,
    mcc_precontact_config: MCCPrecontactConfig,
    dp_history_q_source: Literal["nominal", "live"],
    teacher_observation_source: Literal[
        "teacher", "live_tactile", "teacher_tactile"
    ],
    dp_tactile_normal_source: Literal["contact_sensor", "source_mesh_oracle"],
    teacher_action_source: Literal["dp", "recorded"],
    live_teacher_takeover_frame: int,
    rollout_h5: Path | None,
    rollout_source_file: Path,
    rollout_source_episode_id: int,
    active_palm_planner_config: ActiveCapsulePalmPlannerConfig | None,
    hand_servo_stiffness: float,
    hand_servo_damping: float,
    hand_servo_effort_limit: float,
    thumb_servo_stiffness: float | None,
    thumb_servo_damping: float | None,
    thumb_servo_effort_limit: float | None,
    highlight_contacts: bool,
    report: Path,
    video_output: Path | None,
    video_fps: int,
    video_width: int,
    video_height: int,
    video_camera_distance: float,
    video_camera_azimuth: float,
    video_camera_elevation: float,
    replay_object_config: ObjectConfig | None,
    replay_object_scale: float,
) -> None:
    frames = len(data["q_hand"])
    if max_steps > 0:
        frames = min(frames, max_steps)
    bootstrap_end = (runtime.obs_horizon - 1) * runtime.stride
    if frames <= bootstrap_end + runtime.stride:
        raise ValueError(
            f"Need more than {bootstrap_end + runtime.stride} frames, got {frames}"
        )
    needs_teacher_state = (
        mode == "teacher_dp"
        or teacher_observation_source == "teacher_tactile"
    )
    teacher_surface_embeddings = (
        causal_surface_embeddings(data, runtime)
        if needs_teacher_state
        and runtime.state_schema == "contact_geometry_planner_manifold"
        else None
    )
    teacher = (
        teacher_state(
            data,
            runtime.input_frame,
            runtime.state_schema,
            runtime.planner_waypoints,
            runtime.planner_step_frames,
            teacher_surface_embeddings,
            runtime.motion_feature_step_frames,
            float(getattr(runtime.config, "control_dt", 0.01)),
        )
        if needs_teacher_state
        else None
    )
    planner_features = (
        future_palm_delta_pose_palm(
            data["palm_pose_object"],
            np.zeros(len(data["palm_pose_object"]), dtype=np.int32),
            waypoint_count=runtime.planner_waypoints,
            step_frames=runtime.planner_step_frames,
        ).reshape(len(data["palm_pose_object"]), -1)
        if active_palm_planner_config is None
        and runtime.state_schema in PLANNER_STATE_SCHEMAS
        else None
    )
    if execution_layer == "fullhand_mcc" and impedance_config is not None:
        raise ValueError(
            "--finger-impedance and --execution-layer fullhand_mcc are "
            "alternative low-level controllers; enable only one."
        )
    env_cfg = replay_env_cfg(
        hand_stiffness=hand_servo_stiffness,
        hand_damping=hand_servo_damping,
        hand_effort_limit=hand_servo_effort_limit,
        thumb_stiffness=thumb_servo_stiffness,
        thumb_damping=thumb_servo_damping,
        thumb_effort_limit=thumb_servo_effort_limit,
        object_config=replay_object_config,
        object_scale=replay_object_scale,
    )
    if viewer == "video":
        env_cfg.viewer.width = video_width
        env_cfg.viewer.height = video_height
        env_cfg.viewer.distance = video_camera_distance
        env_cfg.viewer.azimuth = video_camera_azimuth
        env_cfg.viewer.elevation = video_camera_elevation
        env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.ASSET_BODY
        env_cfg.viewer.entity_name = "robot"
        env_cfg.viewer.body_name = "palm_lower"
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=str(device),
        render_mode="rgb_array" if viewer == "video" else None,
    )
    wrapped = RslRlVecEnvWrapper(env)
    robot = env.scene["robot"]
    dp_mesh_normal_oracle = (
        MeshNormalOracle.from_config(
            replay_object_config,
            scale=replay_object_scale,
        )
        if (
            dp_tactile_normal_source == "source_mesh_oracle"
            and replay_object_config is not None
        )
        else None
    )
    if (
        dp_tactile_normal_source == "source_mesh_oracle"
        and dp_mesh_normal_oracle is None
    ):
        raise ValueError(
            "--dp-tactile-normal-source source_mesh_oracle requires a mesh "
            "object with a source visual mesh"
        )

    class DPReplayPolicy:
        def __init__(self):
            self.frame = 0
            self.dp_calls = 0
            self.live_history: deque[np.ndarray] = deque(
                maxlen=runtime.obs_horizon
            )
            # DP owns a nominal joint trajectory.  The high-rate fingertip
            # controller may move the physical joints around that trajectory,
            # but its correction must not be integrated into the next DP
            # reference through q_live.
            self.nominal_q = data["q_hand"][0].copy()
            self.segment_start = data["q_hand"][0].copy()
            self.segment_target = data["q_hand"][0].copy()
            self.segment_plan_frame = bootstrap_end
            self.dp_first_step_mae = float("nan")
            self.dp_horizon_mae = float("nan")
            self.rows: list[dict[str, float | int | str]] = []
            self.contact3_frames = 0
            self.contact4_frames = 0
            self.per_tip_found_frames = np.zeros(4, dtype=np.int64)
            self.per_tip_loaded_frames = np.zeros(4, dtype=np.int64)
            self.force_max = 0.0
            self.impedance = (
                FingertipImpedanceController(impedance_config)
                if impedance_config is not None
                else None
            )
            # DP remains the nominal geometric planner.  This execution
            # layer converts its q prediction into four FK tip targets and
            # lets the shared full-hand MCC apply high-rate force control and
            # four-site IK around those targets.
            self.fullhand_mcc = (
                FullHandMCCFingerController(
                    FullHandMCCFingerConfig(
                        desired_force=mcc_desired_force,
                        desired_force_per_finger=(
                            mcc_precontact_config.desired_force_per_finger
                        ),
                        force_filter_alpha=(
                            mcc_precontact_config.force_filter_alpha
                        ),
                        virtual_stiffness=(
                            mcc_precontact_config.virtual_stiffness
                        ),
                        max_normal_speed=(
                            mcc_precontact_config.max_normal_speed_m_s
                        ),
                        max_normal_acceleration=(
                            mcc_precontact_config.max_normal_acceleration_m_s2
                        ),
                        use_direct_force_servo=(
                            mcc_precontact_config.use_direct_force_servo
                        ),
                        force_servo_integral_gain=(
                            mcc_precontact_config.force_servo_integral_gain
                        ),
                        force_servo_deadband=(
                            mcc_precontact_config.force_servo_deadband
                        ),
                        force_servo_max_step=(
                            mcc_precontact_config.force_servo_max_step_m
                        ),
                        force_servo_hard_step=(
                            mcc_precontact_config.force_servo_hard_step_m
                        ),
                        thumb_force_servo_hard_step=(
                            mcc_precontact_config.thumb_force_servo_hard_step_m
                        ),
                        force_servo_search_step=(
                            mcc_precontact_config.force_servo_search_step_m
                        ),
                        thumb_force_servo_search_step=(
                            mcc_precontact_config.thumb_force_servo_search_step_m
                        ),
                        force_servo_weak_contact_step=(
                            mcc_precontact_config.force_servo_weak_contact_step_m
                        ),
                        max_normal_offset=(
                            mcc_precontact_config.max_normal_offset_m
                            if mcc_precontact_config.max_normal_offset_m
                            is not None
                            else (
                                0.003
                                if mcc_direction_source == "oracle"
                                else 0.006
                            )
                        ),
                        thumb_max_inward_offset=(
                            mcc_precontact_config.thumb_max_inward_offset_m
                        ),
                        thumb_max_outward_offset=(
                            mcc_precontact_config.thumb_max_outward_offset_m
                        ),
                        posture_cost=mcc_precontact_config.posture_cost,
                        nominal_surface_preload=(
                            mcc_precontact_config.nominal_surface_preload_m
                        ),
                        flexion_synergy_gain=(
                            mcc_precontact_config.flexion_synergy_gain
                        ),
                        flexion_synergy_hard_gain=(
                            mcc_precontact_config.flexion_synergy_hard_gain
                        ),
                        flexion_synergy_max_step=(
                            mcc_precontact_config.flexion_synergy_max_step_rad
                        ),
                        normal_synergy_control=(
                            mcc_precontact_config.normal_synergy_control
                        ),
                        normal_synergy_max_step=(
                            mcc_precontact_config.normal_synergy_max_step_rad
                        ),
                        action_rate_limit=(
                            mcc_precontact_config.command_rate_limit_rad
                        ),
                        command_ema_alpha=(
                            mcc_precontact_config.command_ema_alpha
                        ),
                        project_nominal_normal_motion=(
                            mcc_precontact_config.project_nominal_normal_motion
                        ),
                        overforce_trigger_ratio=(
                            mcc_precontact_config.overforce_trigger_ratio
                        ),
                        overforce_release_ratio=(
                            mcc_precontact_config.overforce_release_ratio
                        ),
                        overforce_hard_ratio=(
                            mcc_precontact_config.overforce_hard_ratio
                        ),
                        thumb_overforce_hard_ratio=(
                            mcc_precontact_config.thumb_overforce_hard_ratio
                        ),
                        overforce_retreat_step=(
                            mcc_precontact_config.overforce_retreat_step_m
                        ),
                        overforce_recovery_step=(
                            mcc_precontact_config.overforce_recovery_step_m
                        ),
                        overforce_max_offset=(
                            mcc_precontact_config.overforce_max_offset_m
                        ),
                        enable_loss_state_machine=(
                            mcc_precontact_config.enable_loss_state_machine
                        ),
                        transient_loss_frames=(
                            mcc_precontact_config.transient_loss_frames
                        ),
                        recovery_contact_confirm_frames=(
                            mcc_precontact_config.recovery_confirm_frames
                        ),
                        transient_search_step=(
                            mcc_precontact_config.transient_search_step_m
                        ),
                        transient_release_step=(
                            mcc_precontact_config.transient_release_step_m
                        ),
                        # Match the collection-side distal-flexion floor so
                        # the DP-intended unfolding is clamped to the same
                        # domain the teacher was trained in.
                        natural_flexion_floor=(
                            mcc_precontact_config.natural_flexion_floor
                        ),
                    )
                )
                if execution_layer == "fullhand_mcc"
                else None
            )
            self.surface_oracle = (
                PrivilegedCapsuleSurfaceOracle(radius=0.15, half_height=0.08)
                if (
                    self.fullhand_mcc is not None
                    and mcc_direction_source == "oracle"
                )
                else None
            )
            # Every action decoder already produces absolute joint targets.
            # Enable force regulation from startup for each finger; neither
            # recorded q nor reconstructed DP q needs a new grasp anchor.
            self.fullhand_mcc_calibrated = self.fullhand_mcc is not None
            self.fullhand_precontact_closure = np.zeros(16, dtype=np.float32)
            self.fullhand_contact_anchor_q = data["q_hand"][0].copy()
            self.fullhand_dp_anchor_q = data["q_hand"][0].copy()
            self.fullhand_servo_offset = np.zeros(16, dtype=np.float32)
            self.fullhand_search_delta = np.zeros(16, dtype=np.float32)
            self.fullhand_contact_settle_streak = 0
            self.fullhand_loss_streak = np.zeros(4, dtype=np.int64)
            self.fullhand_recovery_confirm_streak = np.zeros(4, dtype=np.int64)
            self.fullhand_recovery_active = np.zeros(4, dtype=bool)
            self.fullhand_runtime_recovery_offset = np.zeros(
                16, dtype=np.float32
            )
            self.fullhand_last_command_q = data["q_hand"][0].copy()
            self.fullhand_phase = "bootstrap"
            self.active_palm_planner = (
                ActiveCapsulePalmPlanner(
                    data["palm_pose_object"][0], active_palm_planner_config
                )
                if active_palm_planner_config is not None
                else None
            )
            self.current_palm_pose = data["palm_pose_object"][0].copy()
            self.current_palm_twist = np.zeros(6, dtype=np.float32)
            self.current_planner_feature = (
                np.zeros(
                    runtime.planner_waypoints * 6,
                    dtype=np.float32,
                )
                if self.active_palm_planner is not None
                else None
            )
            # FullHandMCC-style viewer state.  Planning targets retain each
            # finger's colour; live markers encode physical contact state.
            self.visual_surface_targets = np.full((4, 3), np.nan)
            self.visual_normals = np.full((4, 3), np.nan)
            self.visual_contact_points = np.full((4, 3), np.nan)
            self.visual_tip_points = np.full((4, 3), np.nan)
            self.visual_found = np.zeros(4, dtype=bool)
            self.visual_loaded = np.zeros(4, dtype=bool)
            # The scheduler always executes a reconstructed 16-DoF joint
            # chunk.  For Cartesian 12D policies, action_std is expressed in
            # metres and cannot define joint-space DTW distances (nor does it
            # have the right shape).  Use the executed-q observation scale,
            # which is exactly the state track on which the dual-track model
            # was trained.
            scheduler_scale = (
                runtime.action_std
                if runtime.action_dim == ACTION_DIM
                else runtime.state_std[:ACTION_DIM]
            )
            self.chunk_scheduler = (
                DPChunkScheduler(scheduler_scale, chunk_config)
                if chunk_config is not None
                else None
            )
            self.chunk_drop_index = 0
            self.contact_guard_blocked = False
            self.guard_contact_count = 4
            self.contact_guard_bad_steps = 0
            self.held_force_state = np.zeros((4, 3), dtype=np.float32)
            self.held_normal_state = np.zeros((4, 3), dtype=np.float32)
            self.held_tactile_valid = np.zeros(4, dtype=bool)
            # Preserve the sensor's native normal convention for the DP
            # observation.  The MCC control history below deliberately uses
            # the opposite (outward-surface) convention.
            self.state_normal_history = np.zeros((4, 3), dtype=np.float32)
            self.state_point_history = np.zeros((4, 3), dtype=np.float32)
            self.state_normal_valid = np.zeros(4, dtype=bool)
            self.state_point_valid = np.zeros(4, dtype=bool)
            self.surface_position_buffer: deque[np.ndarray] = deque(
                maxlen=(runtime.obs_horizon - 1) * runtime.stride + 1
            )
            self.surface_normal_buffer: deque[np.ndarray] = deque(
                maxlen=(runtime.obs_horizon - 1) * runtime.stride + 1
            )
            self.surface_mask_buffer: deque[np.ndarray] = deque(
                maxlen=(runtime.obs_horizon - 1) * runtime.stride + 1
            )
            self.surface_pose_buffer: deque[np.ndarray] = deque(
                maxlen=(runtime.obs_horizon - 1) * runtime.stride + 1
            )
            motion_buffer_length = runtime.motion_feature_step_frames + 1
            self.motion_q_buffer: deque[np.ndarray] = deque(
                maxlen=motion_buffer_length
            )
            self.motion_prior_buffer: deque[np.ndarray] = deque(
                maxlen=motion_buffer_length
            )
            self.motion_position_buffer: deque[np.ndarray] = deque(
                maxlen=motion_buffer_length
            )
            self.motion_normal_buffer: deque[np.ndarray] = deque(
                maxlen=motion_buffer_length
            )
            self.motion_mask_buffer: deque[np.ndarray] = deque(
                maxlen=motion_buffer_length
            )
            # FullHandMCC receives the same contact-anchored tactile geometry
            # as DP.  The normal may be locally lifted to the original source
            # surface, but only after the MuJoCo sensor reports a real
            # fingertip contact; the surface oracle cannot create/search a
            # contact or expose distance/future geometry to the controller.
            self.sensor_normal_history = np.zeros((4, 3), dtype=np.float32)
            self.sensor_point_history = np.zeros((4, 3), dtype=np.float32)
            self.sensor_normal_valid = np.zeros(4, dtype=bool)
            self.sensor_point_valid = np.zeros(4, dtype=bool)
            self.sensor_normal_age = np.full(4, 10_000, dtype=np.int32)
            self.fullhand_control_outward = np.zeros((4, 3), dtype=np.float32)
            self.fullhand_control_direction_valid = False
            self.last_dp_nominal_q: np.ndarray | None = None
            self.applied_prior_q = data["q_hand"][0].copy()
            self.applied_command_q = data["q_hand"][0].copy()
            self.latest_dp_observation_state: np.ndarray | None = None
            self.rollout_arrays: dict[str, list[np.ndarray]] = {}

        def _append_rollout(
            self,
            key: str,
            value: np.ndarray | list[float] | tuple[float, ...],
        ) -> None:
            if rollout_h5 is None:
                return
            self.rollout_arrays.setdefault(key, []).append(
                np.asarray(value, dtype=np.float32).copy()
            )

        def _set_palm(self, t: int) -> None:
            if self.active_palm_planner is not None:
                self.current_palm_pose = (
                    self.active_palm_planner.pose_object
                )
                self.current_palm_twist = (
                    self.active_palm_planner.twist_object
                )
                self.current_planner_feature = (
                    self.active_palm_planner.planner_feature(
                        waypoint_count=runtime.planner_waypoints,
                        step_frames=runtime.planner_step_frames,
                    )
                )
            else:
                self.current_palm_pose = data["palm_pose_object"][t].copy()
                self.current_palm_twist = data["palm_twist_object"][t].copy()
            pose = torch.as_tensor(
                self.current_palm_pose,
                device=env.device,
                dtype=torch.float32,
            )
            root_state = torch.cat(
                (pose, torch.zeros(6, device=env.device))
            ).unsqueeze(0)
            robot.write_root_state_to_sim(root_state)
            if env.sim.model.nmocap:
                env.sim.data.mocap_pos[:, 0, :] = 0.0
                env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
                    (1.0, 0.0, 0.0, 0.0), device=env.device
                )

        def _live_state(
            self, t: int
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            int,
            float,
            float,
        ]:
            forces, sensor_normals, positions, found, distances = (
                live_tip_observation(env)
            )
            observation_normals = sensor_normals.copy()
            if dp_mesh_normal_oracle is not None and found.any():
                found_indices = np.flatnonzero(found)
                point_world = positions[found_indices]
                if env.sim.model.nmocap:
                    object_position = (
                        env.sim.data.mocap_pos[0, 0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    object_quaternion = (
                        env.sim.data.mocap_quat[0, 0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    object_position = np.zeros(3, dtype=np.float32)
                    object_quaternion = np.asarray(
                        (1.0, 0.0, 0.0, 0.0), dtype=np.float32
                    )
                outward = dp_mesh_normal_oracle.query_world(
                    point_world,
                    np.repeat(object_position[None, :], len(found_indices), axis=0),
                    np.repeat(
                        object_quaternion[None, :], len(found_indices), axis=0
                    ),
                )
                # The checkpoint convention is ContactSensor primary
                # fingertip -> secondary object, i.e. source-mesh inward.
                observation_normals[found_indices] = -outward.astype(np.float32)
            # Geometry is measured by the contact sensors.  Hold the last
            # valid point/normal across a short sensor dropout so the policy
            # does not receive an artificial all-zero surface observation.
            normal_valid = found & (
                np.linalg.norm(observation_normals, axis=-1) > 1.0e-6
            )
            point_valid = found & (np.linalg.norm(positions, axis=-1) > 1.0e-9)
            self.state_normal_history[normal_valid] = observation_normals[
                normal_valid
            ]
            self.state_point_history[point_valid] = positions[point_valid]
            self.state_normal_valid[normal_valid] = True
            self.state_point_valid[point_valid] = True
            state_normals_live = observation_normals.copy()
            state_positions_live = positions.copy()
            held_normals = (~normal_valid) & self.state_normal_valid
            held_points = (~point_valid) & self.state_point_valid
            state_normals_live[held_normals] = self.state_normal_history[held_normals]
            state_positions_live[held_points] = self.state_point_history[held_points]
            q_live = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            state_forces = forces
            state_normals = state_normals_live
            if runtime.contact_normal_polarity in (
                "source_mesh_outward",
                "analytic_outward",
            ):
                # Live ContactSensor normals are primary fingertip -> object.
                state_normals = -state_normals
            state_positions = state_positions_live
            state_twist = self.current_palm_twist
            if runtime.input_frame == "palm":
                palm_pose = self.current_palm_pose
                palm_quaternion = palm_pose[3:7]
                state_forces = _vectors_object_to_palm(
                    forces, palm_quaternion
                )
                state_normals = _vectors_object_to_palm(
                    state_normals_live, palm_quaternion
                )
                state_positions = _points_object_to_palm(
                    state_positions_live[None, ...],
                    palm_pose[None, ...],
                )[0]
                state_twist = np.concatenate(
                    (
                        _vectors_object_to_palm(
                            state_twist[:3], palm_quaternion
                        ),
                        _vectors_object_to_palm(
                            state_twist[3:], palm_quaternion
                        ),
                    )
                )
            motion_parts: list[np.ndarray] = []
            q_prior_velocity = np.zeros(16, dtype=np.float32)
            q_live_velocity_fd = np.zeros(16, dtype=np.float32)
            point_velocity = np.zeros((4, 3), dtype=np.float32)
            normal_rate = np.zeros((4, 3), dtype=np.float32)
            applied_delta_q_comp = (
                self.applied_command_q - self.applied_prior_q
            ).astype(np.float32)
            q_task_est = (q_live - applied_delta_q_comp).astype(np.float32)
            if runtime.state_schema in MOTION_SCHEMAS:
                motion_q = (
                    q_task_est
                    if runtime.state_schema == TASK_EST_V4_SCHEMA
                    else q_live
                )
                self.motion_q_buffer.append(motion_q.copy())
                self.motion_prior_buffer.append(self.applied_prior_q.copy())
                self.motion_position_buffer.append(state_positions.copy())
                self.motion_normal_buffer.append(state_normals.copy())
                self.motion_mask_buffer.append(found.copy())
                if len(self.motion_q_buffer) == self.motion_q_buffer.maxlen:
                    duration = runtime.motion_feature_step_frames * float(
                        getattr(runtime.config, "control_dt", 0.01)
                    )
                    previous_q = self.motion_q_buffer[0]
                    previous_position = self.motion_position_buffer[0]
                    previous_normal = self.motion_normal_buffer[0]
                    previous_mask = self.motion_mask_buffer[0]
                    valid_motion = found & previous_mask
                    q_live_velocity_fd = (
                        (motion_q - previous_q) / duration
                    ).astype(np.float32)
                    q_prior_velocity = (
                        (self.applied_prior_q - self.motion_prior_buffer[0])
                        / duration
                    ).astype(np.float32)
                    point_velocity = np.where(
                        valid_motion[:, None],
                        (state_positions - previous_position) / duration,
                        0.0,
                    ).astype(np.float32)
                    normal_rate = np.where(
                        valid_motion[:, None],
                        np.cross(previous_normal, state_normals) / duration,
                        0.0,
                    ).astype(np.float32)
                motion_parts = [point_velocity.reshape(-1), normal_rate.reshape(-1)]
            tactile_state = (
                state_forces if runtime.state_schema == "force_normal"
                else state_positions
            )
            if runtime.state_schema == TASK_EST_V4_SCHEMA:
                state_parts = [
                    q_task_est,
                    tactile_state.reshape(-1),
                    state_normals.reshape(-1),
                    found.astype(np.float32),
                    q_live_velocity_fd,
                    *motion_parts,
                    state_twist,
                ]
            elif runtime.state_schema == DUAL_TRACK_V3_SCHEMA:
                joint_velocity = (
                    robot.data.joint_vel[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1)[-16:]
                )
                state_parts = [
                    self.applied_prior_q,
                    tactile_state.reshape(-1),
                    state_normals.reshape(-1),
                    found.astype(np.float32),
                    q_prior_velocity,
                    *motion_parts,
                    q_live,
                    joint_velocity,
                    joint_velocity - q_prior_velocity,
                    self.applied_command_q - self.applied_prior_q,
                    q_live - self.applied_command_q,
                    state_twist,
                ]
            elif runtime.state_schema == DUAL_TRACK_SCHEMA:
                state_parts = [
                    self.applied_prior_q,
                    tactile_state.reshape(-1),
                    state_normals.reshape(-1),
                    found.astype(np.float32),
                    q_prior_velocity,
                    *motion_parts,
                    q_live,
                    self.applied_command_q - self.applied_prior_q,
                    q_live - self.applied_command_q,
                    state_twist,
                ]
            else:
                state_parts = [
                    q_live,
                    tactile_state.reshape(-1),
                    state_normals.reshape(-1),
                ]
                if runtime.state_schema in GEOMETRY_STATE_SCHEMAS:
                    state_parts.append(found.astype(np.float32))
                if runtime.state_schema == MOTION_SCHEMA:
                    state_parts.extend([q_live_velocity_fd, *motion_parts])
                state_parts.append(state_twist)
            if self.current_planner_feature is not None:
                state_parts.append(self.current_planner_feature)
            elif planner_features is not None:
                state_parts.append(planner_features[t])
            if runtime.state_schema == "contact_geometry_planner_manifold":
                self.surface_position_buffer.append(state_positions_live.copy())
                self.surface_normal_buffer.append(state_normals_live.copy())
                self.surface_mask_buffer.append(found.astype(np.float32))
                self.surface_pose_buffer.append(self.current_palm_pose.copy())
                surface_points = _causal_surface_gp_points(
                    np.asarray(self.surface_position_buffer),
                    np.asarray(self.surface_normal_buffer),
                    np.asarray(self.surface_mask_buffer),
                    np.asarray(self.surface_pose_buffer),
                    len(self.surface_pose_buffer) - 1,
                    runtime.stride,
                    runtime.surface_gp_config,
                )
                state_parts.append(runtime.encode_surface_manifold(surface_points)[0])
            state = np.concatenate(state_parts).astype(np.float32)
            magnitude = np.linalg.norm(forces, axis=-1)
            found_count = int(found.sum())
            loaded_count = int(np.sum(found & (magnitude >= contact_threshold)))
            return (
                state,
                forces,
                observation_normals,
                state_positions_live,
                found,
                found_count,
                loaded_count,
                float(magnitude.max(initial=0.0)),
                float(distances.min(initial=0.0)),
            )

        def _plan(self, t: int, live_state: np.ndarray) -> None:
            if self.dp_calls >= max_dp_calls > 0:
                return
            if mode == "teacher_dp" and teacher_observation_source == "teacher":
                indices = history_indices(
                    t, runtime.stride, runtime.obs_horizon
                )
                if teacher is None:
                    raise RuntimeError("teacher history is unavailable")
                history = teacher[indices]
                q_base = data["q_hand"][t]
            else:
                if len(self.live_history) != runtime.obs_horizon:
                    raise RuntimeError(
                        f"live history has {len(self.live_history)} frames"
                    )
                history = np.stack(self.live_history)
                q_base = (
                    data["q_hand"][t]
                    if mode == "teacher_dp"
                    else history[-1, :16].copy()
                )
            prediction = runtime.predict(history)
            if runtime.action_field in (
                "tip_delta_tangent_palm",
                "tip_motion_tangent_palm",
                "tip_motion_palm",
                "tip_target_palm",
            ):
                # Cartesian policy: rebuild the 16D absolute joint command
                # with FK + per-finger IK. Legacy tip-delta is a one-base
                # offset; task-motion actions are per-waypoint increments.
                if runtime.action_field == "tip_target_palm":
                    predicted_absolute = tip_target_to_absolute_q(
                        self.fullhand_mcc,
                        prediction,
                        q_base,
                        runtime.pred_horizon,
                    )
                else:
                    predicted_absolute = tip_delta_to_absolute_q(
                        self.fullhand_mcc,
                        prediction,
                        q_base,
                        runtime.pred_horizon,
                        per_waypoint_increment=(
                            runtime.action_field in (
                                "tip_motion_tangent_palm",
                                "tip_motion_palm",
                            )
                        ),
                    )
            else:
                predicted_absolute = runtime.absolute_action(
                    prediction,
                    q_base,
                    (
                        history[-1, 44:60]
                        if runtime.action_representation
                        == "kinematic_residual_q"
                        else None
                    ),
                )
            self.segment_start = self.nominal_q.copy()
            self.segment_target = predicted_absolute[0]
            self.segment_plan_frame = t
            if self.chunk_scheduler is not None:
                self.chunk_drop_index = self.chunk_scheduler.install(
                    predicted_absolute
                )
            self.dp_calls += 1
            teacher_target = (
                self.nominal_q
                if self.active_palm_planner is not None
                else data["q_hand"][
                    min(t + runtime.stride, len(data["q_hand"]) - 1)
                ]
            )
            prediction_error = float(
                np.abs(self.segment_target - teacher_target).mean()
            )
            if self.active_palm_planner is None:
                teacher_horizon = np.stack(
                    [
                        data["q_hand"][
                            min(
                                t + (waypoint + 1) * runtime.stride,
                                len(data["q_hand"]) - 1,
                            )
                        ]
                        for waypoint in range(runtime.pred_horizon)
                    ]
                )
                horizon_error = np.abs(predicted_absolute - teacher_horizon)
                self.dp_first_step_mae = float(horizon_error[0].mean())
                self.dp_horizon_mae = float(horizon_error.mean())
            if self.dp_calls == 1 or self.dp_calls % 25 == 0:
                print(
                    f"[DP] mode={mode} call={self.dp_calls:4d} frame={t:4d} "
                    f"target_error={prediction_error:.6f}rad"
                )

        def _state_for_dp(
            self,
            live_state: np.ndarray,
            live_forces: np.ndarray,
            live_found: np.ndarray,
        ) -> np.ndarray:
            """Build causal live feedback and hold only unreliable tactile channels."""
            state = live_state.copy()
            if (
                runtime.state_schema == TASK_EST_V4_SCHEMA
                and dp_history_q_source != "live"
            ):
                raise ValueError(
                    "task-est v4 must use live q_task_est feedback; nominal "
                    "history would recreate the removed autoregressive loop"
                )
            if dp_history_q_source == "nominal":
                state[:16] = self.nominal_q
            if (
                runtime.state_schema == MOTION_SCHEMA
                and dp_history_q_source == "nominal"
            ):
                if self.last_dp_nominal_q is None:
                    state[44:60] = 0.0
                else:
                    sample_dt = runtime.stride * float(
                        getattr(runtime.config, "control_dt", 0.01)
                    )
                    state[44:60] = (
                        self.nominal_q - self.last_dp_nominal_q
                    ) / sample_dt
                self.last_dp_nominal_q = self.nominal_q.copy()
            force_state = state[16:28].reshape(4, 3)
            normal_state = state[28:40].reshape(4, 3)
            magnitude = np.linalg.norm(live_forces, axis=-1)
            threshold = (
                contact_guard_config.force_threshold
                if contact_guard_config is not None
                else 0.0
            )
            reliable = (
                live_found
                if runtime.state_schema in GEOMETRY_STATE_SCHEMAS
                else live_found & (magnitude >= threshold)
            )
            for finger in range(4):
                if reliable[finger]:
                    self.held_force_state[finger] = force_state[finger]
                    self.held_normal_state[finger] = normal_state[finger]
                    self.held_tactile_valid[finger] = True
                elif self.held_tactile_valid[finger]:
                    force_state[finger] = self.held_force_state[finger]
                    normal_state[finger] = self.held_normal_state[finger]
            return state

        def __call__(self, _observation: dict[str, torch.Tensor]) -> torch.Tensor:
            t = min(self.frame, frames - 1)
            self._set_palm(t)
            if t <= bootstrap_end:
                bootstrap_target = (
                    data["q_hand"][0]
                    if self.active_palm_planner is not None
                    else data["q_hand"][t]
                )
                # Write the initial state only once, then use actuators. Start
                # from the recorded grasp for every action contract: a generic
                # open posture can collide at this fixed palm/object pose (V4
                # note), and it also leaves q_hand-contract fingers so far from
                # the surface that the precontact search cannot settle and the
                # episode is stuck in precontact with growing force. The
                # recorded start does not certify the initial pose as
                # penetration-free; force regulation is active immediately.
                if self.fullhand_mcc is not None:
                    initial_q = data["q_hand"][0]
                    progress = float(t) / max(float(bootstrap_end), 1.0)
                    smooth = progress * progress * (3.0 - 2.0 * progress)
                    bootstrap_q = (
                        (1.0 - smooth) * initial_q
                        + smooth * bootstrap_target
                    )
                    if t == 0:
                        # Controller states may be float64; sim qpos is float32.
                        q_open = torch.as_tensor(
                            initial_q,
                            device=env.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        robot.write_joint_state_to_sim(
                            position=q_open,
                            velocity=torch.zeros_like(q_open),
                        )
                else:
                    # Preserve the legacy exact-state bootstrap for execution
                    # layers that do not own contact-force initialization.
                    bootstrap_q = bootstrap_target
                    q_teacher = torch.as_tensor(
                        bootstrap_q, device=env.device
                    ).unsqueeze(0)
                    robot.write_joint_state_to_sim(
                        position=q_teacher,
                        velocity=torch.zeros_like(q_teacher),
                    )
                self.nominal_q = bootstrap_q.copy()
            env.sim.forward()
            (
                live_state,
                live_forces,
                live_normals,
                live_contact_positions,
                live_found,
                found_contacts,
                loaded_contacts,
                force_max,
                min_distance,
            ) = self._live_state(t)
            guard_contact = live_found & (
                np.linalg.norm(live_forces, axis=-1)
                >= (
                    contact_guard_config.force_threshold
                    if contact_guard_config is not None
                    else 0.0
                )
            )
            self.guard_contact_count = int(guard_contact.sum())
            guard_contact_ok = (
                contact_guard_config is None
                or mode != "live_dp"
                or self.guard_contact_count >= contact_guard_config.min_fingers
            )
            if guard_contact_ok:
                self.contact_guard_bad_steps = 0
            else:
                self.contact_guard_bad_steps += 1
            if (
                contact_guard_config is not None
                and mode == "live_dp"
                and self.dp_calls > 0
                and not guard_contact_ok
                and not self.contact_guard_blocked
                and self.contact_guard_bad_steps
                >= contact_guard_config.bad_grace_steps
            ):
                self.contact_guard_blocked = True
                print(
                    f"[TACTILE-HOLD] active at frame={t}: "
                    f"loaded={self.guard_contact_count}/4; "
                    "DP nominal continues with last reliable tactile samples"
                )
            if self.chunk_scheduler is not None:
                self.chunk_scheduler.observe(self.nominal_q)

            if t % runtime.stride == 0:
                self.latest_dp_observation_state = self._state_for_dp(
                    live_state,
                    live_forces,
                    live_found,
                )
                if (
                    mode == "live_dp"
                    and teacher_observation_source == "teacher_tactile"
                ):
                    # Pure q-feedback diagnostic: retain the live executed
                    # q/qdot stream, but replace contact geometry and its
                    # rates with the time-aligned recorded teacher values.
                    # This is intentionally privileged and must never be used
                    # as a deployable result.
                    if teacher is None:
                        raise RuntimeError("teacher tactile state is unavailable")
                    self.latest_dp_observation_state[16:44] = teacher[t, 16:44]
                    if runtime.state_schema in (MOTION_SCHEMA, TASK_EST_V4_SCHEMA):
                        self.latest_dp_observation_state[60:84] = teacher[t, 60:84]
                if (
                    mode == "teacher_dp"
                    and teacher_observation_source == "live_tactile"
                ):
                    # Isolate tactile domain shift: q/qdot remain on the
                    # recorded teacher trajectory while contact point/normal,
                    # mask and their causal rates come from current physics.
                    self.latest_dp_observation_state[:16] = data["q_hand"][t]
                    if runtime.state_schema in (MOTION_SCHEMA, TASK_EST_V4_SCHEMA):
                        previous_t = max(
                            0, t - runtime.motion_feature_step_frames
                        )
                        duration = max(
                            1, t - previous_t
                        ) * float(getattr(runtime.config, "control_dt", 0.01))
                        self.latest_dp_observation_state[44:60] = (
                            data["q_hand"][t] - data["q_hand"][previous_t]
                        ) / duration
                self.live_history.append(self.latest_dp_observation_state)
            if (
                self.contact_guard_blocked
                and guard_contact_ok
            ):
                self.contact_guard_blocked = False
                print(
                    f"[TACTILE-HOLD] released at frame={t}: "
                    "live fingertip measurements are reliable again"
                )
            replan_interval = (
                self.chunk_scheduler.config.replan_interval
                if self.chunk_scheduler is not None
                else runtime.stride
            )
            if (
                t >= bootstrap_end
                and (t - bootstrap_end) % replan_interval == 0
                and mode != "collect_executed"
            ):
                history_ready = len(self.live_history) == runtime.obs_horizon
                if history_ready:
                    self._plan(t, live_state)

            if t <= bootstrap_end:
                desired = bootstrap_q
            elif self.chunk_scheduler is not None:
                desired = self.chunk_scheduler.next_command()
            else:
                alpha = min(
                    1.0,
                    (t - self.segment_plan_frame + 1) / runtime.stride,
                )
                desired = (
                    (1.0 - alpha) * self.segment_start
                    + alpha * self.segment_target
                )
            if mode == "teacher_dp" and teacher_action_source == "recorded":
                # Diagnostic path: retain teacher-conditioned DP inference and
                # its reported prediction error, but send the exact recorded
                # teacher q through the same position actuator/physics stack.
                # This separates model error from closed-loop servo error.
                desired = data["q_hand"][t]
            if mode == "collect_executed":
                # Data-collection mode: execute the recorded teacher q through
                # the same actuator/MCC stack used at deployment (frozen §32
                # MCC settings passed on the CLI).  DP inference is skipped;
                # q_live is the new training label, so the recorded closed-loop
                # state carries the deployment compensation distribution.
                desired = data["q_hand"][t]
            if (
                mode == "live_dp"
                and live_teacher_takeover_frame >= 0
                and t >= live_teacher_takeover_frame
            ):
                # Recovery-dataset diagnostic: preserve the exact physical
                # state reached by live DP up to this frame, then switch only
                # the task prior to the successful time-aligned teacher.  DP
                # inference continues for diagnostics, but its failed action
                # is never used as the recovery label or physical command.
                desired = data["q_hand"][t]
                if t == live_teacher_takeover_frame:
                    print(
                        "[TEACHER-TAKEOVER] "
                        f"frame={t}: live physical state retained; "
                        "executing time-aligned recorded teacher q"
                    )
            dp_desired = np.asarray(desired, dtype=np.float32).copy()
            self.nominal_q = dp_desired.copy()
            q_live = robot.data.joint_pos[0].detach().cpu().numpy().copy()
            applied_prior_q = self.applied_prior_q.copy()
            applied_command_q = self.applied_command_q.copy()
            mcc_normal_force = np.zeros(4, dtype=np.float32)
            mcc_force_error = np.zeros(4, dtype=np.float32)
            mcc_contact_active = np.zeros(4, dtype=bool)
            mcc_tip_target_error = np.zeros(4, dtype=np.float32)
            mcc_normal_offset = np.zeros(4, dtype=np.float32)
            mcc_normal_velocity = np.zeros(4, dtype=np.float32)
            mcc_overforce_active = np.zeros(4, dtype=bool)
            mcc_pad_contact_valid = np.zeros(4, dtype=bool)
            mcc_collision_safety_active = np.zeros(4, dtype=bool)
            mcc_safety_offset = np.zeros(4, dtype=np.float32)
            if self.fullhand_mcc is not None:
                self.fullhand_search_delta[:] = 0.0
                palm_pose = self.current_palm_pose
                force_magnitude = np.linalg.norm(live_forces, axis=-1)
                normal_valid = live_found & (
                    np.linalg.norm(live_normals, axis=-1) > 1.0e-6
                )
                point_valid = live_found & (
                    np.linalg.norm(live_contact_positions, axis=-1) > 1.0e-9
                )
                # ContactSensor.normal is primary -> secondary.  Here the
                # fingertip geom is primary and the object is secondary, so
                # the measured vector points into the object.  Negate it once
                # to obtain the outward unloading direction expected by MCC.
                measured_outward = -live_normals
                self.sensor_normal_history[normal_valid] = measured_outward[
                    normal_valid
                ]
                self.sensor_point_history[point_valid] = live_contact_positions[
                    point_valid
                ]
                self.sensor_normal_valid[normal_valid] = True
                self.sensor_point_valid[point_valid] = True
                self.sensor_normal_age += 1
                self.sensor_normal_age[normal_valid] = 0
                closure_inward_palm = (
                    self.fullhand_mcc.grasp_closure_directions_palm(q_live)
                )
                closure_inward_world = self.fullhand_mcc.vectors_palm_to_world(
                    closure_inward_palm, palm_pose
                ).astype(np.float32)
                closure_outward_world = -closure_inward_world
                # A fresh tactile normal is authoritative.  Across a brief
                # contact dropout retain the last measured normal, because
                # the local surface cannot rotate discontinuously.  Once the
                # measurement is stale, fall back to the kinematic closing
                # direction rather than extrapolating unknown geometry.
                short_normal_memory = self.sensor_normal_valid & (
                    self.sensor_normal_age
                    <= mcc_precontact_config.sensor_normal_memory_frames
                )
                sensor_outward_world = closure_outward_world.copy()
                sensor_outward_world[short_normal_memory] = (
                    self.sensor_normal_history[short_normal_memory]
                )
                if mcc_direction_source == "oracle":
                    oracle_query_world = self.fullhand_mcc.points_palm_to_world(
                        self.fullhand_mcc.tip_positions_palm(q_live), palm_pose
                    )
                    control_outward_world = self.surface_oracle.observe(
                        oracle_query_world
                    ).normals_world
                elif mcc_direction_source == "sensor_normal":
                    control_outward_world = sensor_outward_world.copy()
                elif mcc_direction_source == "grasp_closure":
                    control_outward_world = closure_outward_world.copy()
                else:
                    # Use the measured ContactSensor normal whenever a
                    # geometry contact exists.  Missing fingers fall back to
                    # their robot-kinematic grasp-closing direction.  No
                    # object pose, mesh query, or analytic surface normal is
                    # used in this hybrid deployment path.
                    raw_outward = closure_outward_world.copy()
                    raw_outward[short_normal_memory] = sensor_outward_world[
                        short_normal_memory
                    ]
                    if self.fullhand_control_direction_valid:
                        direction_alpha = 0.35
                        raw_outward = (
                            (1.0 - direction_alpha)
                            * self.fullhand_control_outward
                            + direction_alpha * raw_outward
                        )
                    direction_norm = np.linalg.norm(
                        raw_outward, axis=-1, keepdims=True
                    )
                    control_outward_world = raw_outward / np.maximum(
                        direction_norm, 1.0e-8
                    )
                    self.fullhand_control_outward[:] = control_outward_world
                    self.fullhand_control_direction_valid = True
                control_inward_world = -control_outward_world
                force_target = self.fullhand_mcc.nominal_force_setpoint.astype(
                    np.float32,
                    copy=True,
                )
                if mcc_direction_source == "oracle":
                    real_contact = live_found & (
                        force_magnitude
                        >= mcc_precontact_config.force_threshold
                    )
                else:
                    settle_force = np.maximum(
                        float(mcc_precontact_config.force_threshold),
                        mcc_precontact_config.settle_force_ratio * force_target,
                    )
                    real_contact = live_found & (
                        force_magnitude >= settle_force
                    )
                if t <= bootstrap_end:
                    self.fullhand_phase = "bootstrap"
                    plan_q = self.fullhand_mcc.clamp_joint_positions(dp_desired)
                elif not self.fullhand_mcc_calibrated:
                    self.fullhand_phase = "precontact"
                    # Precontact owns only geometry acquisition.  Requiring
                    # every finger to reach 0.8*F_des before enabling MCC kept
                    # the system in an unregulated closing phase for hundreds
                    # of frames and over-loaded fingers that had already
                    # touched.  Force convergence belongs to the force loop.
                    missing = ~live_found
                    # One independent 3x4 Jacobian solve per missing finger.
                    # Blocks are [0:4], [4:8], [8:12], [12:16], so the thumb
                    # always uses all four of its joints and never shares a
                    # Jacobian with the three parallel fingers.
                    search_q = self.fullhand_mcc.clamp_joint_positions(
                        dp_desired + self.fullhand_precontact_closure
                    )
                    if mcc_direction_source == "oracle":
                        search_points_world = (
                            self.fullhand_mcc.points_palm_to_world(
                                self.fullhand_mcc.tip_positions_palm(search_q),
                                palm_pose,
                            )
                        )
                        search_surface = self.surface_oracle.observe(
                            search_points_world
                        )
                        self.fullhand_search_delta = (
                            self.fullhand_mcc.normal_search_delta(
                                q_action_order=q_live,
                                palm_pose_world=palm_pose,
                                surface_normals_world=(
                                    search_surface.normals_world
                                ),
                                missing=missing,
                                inward_step=(
                                    mcc_precontact_config.cartesian_step_m
                                ),
                                max_joint_step=(
                                    mcc_precontact_config.joint_step_rad
                                ),
                            )
                        )
                    else:
                        self.fullhand_search_delta = (
                            self.fullhand_mcc.directional_search_delta(
                            q_action_order=q_live,
                            palm_pose_world=palm_pose,
                            inward_directions_world=control_inward_world,
                            missing=missing,
                            inward_step=mcc_precontact_config.cartesian_step_m,
                            max_joint_step=mcc_precontact_config.joint_step_rad,
                            contact_points_world=live_contact_positions,
                            contact_point_found=point_valid,
                        )
                        )
                    self.fullhand_precontact_closure = np.clip(
                        self.fullhand_precontact_closure
                        + self.fullhand_search_delta,
                        -mcc_precontact_config.joint_limit_rad,
                        mcc_precontact_config.joint_limit_rad,
                    )
                    plan_q = self.fullhand_mcc.clamp_joint_positions(
                        dp_desired + self.fullhand_precontact_closure
                    )
                    # Match FullHandMCC's outer precontact rate limit.  The
                    # accumulated closure may be large, but the physical
                    # command advances by at most one search step per frame.
                    desired = self.fullhand_mcc.clamp_joint_positions(
                        q_live
                        + np.clip(
                            plan_q - q_live,
                            -mcc_precontact_config.joint_step_rad,
                            mcc_precontact_config.joint_step_rad,
                        )
                    )
                    # A finger that has reached the settling band must not be
                    # pushed farther while the other fingers continue their
                    # independent search.  Hold its previous actuator target.
                    for finger in np.flatnonzero(live_found):
                        block = slice(4 * finger, 4 * finger + 4)
                        desired[block] = self.fullhand_last_command_q[block]

                    # Precontact previously had no force loop at all.  Apply
                    # an immediate per-finger outward Jacobian step before the
                    # full four-finger settle condition when raw force exceeds
                    # 1.5 times the requested load.
                    precontact_overforce = live_found & (
                        force_magnitude >= 1.5 * force_target
                    )
                    if np.any(precontact_overforce):
                        retreat_delta = (
                            self.fullhand_mcc.directional_search_delta(
                                q_action_order=q_live,
                                palm_pose_world=palm_pose,
                                inward_directions_world=control_outward_world,
                                missing=precontact_overforce,
                                inward_step=max(
                                    mcc_precontact_config.cartesian_step_m,
                                    mcc_precontact_config.force_servo_hard_step_m,
                                ),
                                max_joint_step=(
                                    mcc_precontact_config.joint_step_rad
                                ),
                                contact_points_world=live_contact_positions,
                                contact_point_found=point_valid,
                            )
                        )
                        for finger in np.flatnonzero(precontact_overforce):
                            block = slice(4 * finger, 4 * finger + 4)
                            # Safety owns this block: cancel the inward target
                            # and execute only a bounded outward Cartesian
                            # escape step from the measured posture.
                            desired[block] = (
                                q_live[block] + retreat_delta[block]
                            )
                        desired = self.fullhand_mcc.clamp_joint_positions(
                            desired
                        )
                    self.fullhand_mcc.previous_command = desired.copy()
                    if bool(np.all(live_found) and not np.any(precontact_overforce)):
                        self.fullhand_contact_settle_streak += 1
                    else:
                        self.fullhand_contact_settle_streak = 0
                    if (
                        self.fullhand_contact_settle_streak
                        >= mcc_precontact_config.settle_frames
                    ):
                        if mcc_direction_source == "oracle":
                            self.fullhand_mcc.calibrate_force_setpoint(
                                live_forces,
                                live_found,
                                search_surface.normals_world,
                            )
                        elif mcc_direction_source != "grasp_closure":
                            self.fullhand_mcc.calibrate_force_sign(
                                live_forces,
                                live_found,
                                control_outward_world,
                            )
                            self.fullhand_mcc.force_setpoint[:] = (
                                self.fullhand_mcc.nominal_force_setpoint
                            )
                        else:
                            self.fullhand_mcc.force_setpoint[:] = (
                                self.fullhand_mcc.nominal_force_setpoint
                            )
                        # FullHandMCC changes coordinates at contact: the
                        # loaded physical posture becomes the new planning
                        # anchor, while only command-to-loaded servo
                        # deflection is retained as feed-forward.  The local
                        # precontact closure is not a permanent trajectory
                        # offset.
                        self.fullhand_contact_anchor_q = q_live.copy()
                        self.fullhand_dp_anchor_q = dp_desired.copy()
                        # Build the posture/synergy reference from the first
                        # stable *executed* grasp.  This is causal and uses
                        # only joint/contact sensors.  Do not read the
                        # object-specific pregrasp from ObjectConfig here:
                        # that is teacher-side privileged information.
                        self.fullhand_mcc.grasp_closure_q = q_live.copy()
                        self.fullhand_servo_offset = (
                            mcc_precontact_config.servo_load_scale
                            * (self.fullhand_last_command_q - q_live)
                        )
                        self.fullhand_precontact_closure[:] = 0.0
                        self.fullhand_mcc_calibrated = True
                        self.fullhand_phase = "track"
                        plan_q = self.fullhand_contact_anchor_q.copy()
                        print(
                            "[DP->FULLHAND-MCC] per-finger contact settled; "
                            "force_setpoint="
                            f"{np.round(self.fullhand_mcc.force_setpoint, 2).tolist()}N "
                            "servo_offset="
                            f"{np.round(self.fullhand_servo_offset, 3).tolist()}"
                        )
                else:
                    # Weak force with a valid geometry contact remains the
                    # admittance loop's job.  Cartesian recovery starts only
                    # after consecutive true geometry-loss frames.
                    self.fullhand_loss_streak = np.where(
                        live_found, 0, self.fullhand_loss_streak + 1
                    )
                    newly_lost = (
                        ~self.fullhand_recovery_active
                        & (
                            self.fullhand_loss_streak
                            >= mcc_precontact_config.runtime_loss_frames
                        )
                    )
                    if np.any(newly_lost):
                        self.fullhand_recovery_active[newly_lost] = True
                        self.fullhand_recovery_confirm_streak[newly_lost] = 0
                        self.fullhand_mcc.reset_admittance_fingers(
                            newly_lost, preserve_offset=True
                        )

                    self.fullhand_recovery_confirm_streak = np.where(
                        self.fullhand_recovery_active & real_contact,
                        self.fullhand_recovery_confirm_streak + 1,
                        0,
                    )
                    recovered = (
                        self.fullhand_recovery_active
                        & (
                            self.fullhand_recovery_confirm_streak
                            >= mcc_precontact_config.recovery_confirm_frames
                        )
                    )
                    if np.any(recovered):
                        self.fullhand_recovery_active[recovered] = False
                        self.fullhand_recovery_confirm_streak[recovered] = 0
                        self.fullhand_loss_streak[recovered] = 0
                        self.fullhand_mcc.reset_admittance_fingers(
                            recovered, preserve_offset=True
                        )

                    # FullHandMCC establishes the loaded planning anchor only
                    # once.  Runtime re-contact must not replace it with
                    # q_live, otherwise every small tracking error is
                    # integrated into the future DP trajectory.  Preserve the
                    # bounded recovery correction and relax it only when that
                    # finger already carries its calibrated target force.
                    force_supported = (
                        ~self.fullhand_recovery_active
                        & live_found
                        & (
                            force_magnitude
                            >= (
                                mcc_precontact_config.recovery_decay_force_ratio
                                * self.fullhand_mcc.force_setpoint
                            )
                        )
                    )
                    for finger in np.flatnonzero(force_supported):
                        block = slice(4 * finger, 4 * finger + 4)
                        self.fullhand_runtime_recovery_offset[block] *= (
                            mcc_precontact_config.recovery_offset_decay
                        )

                    self.fullhand_phase = (
                        "recover"
                        if bool(np.any(self.fullhand_recovery_active))
                        else "track"
                    )
                    # Decoders already return absolute targets. Adding a
                    # loaded-grasp anchor here moves even exact recorded q
                    # away from its intended contact geometry.
                    base_plan_q = self.fullhand_mcc.clamp_joint_positions(
                        dp_desired
                    )
                    if bool(np.any(self.fullhand_recovery_active)):
                        if mcc_direction_source == "oracle":
                            base_points_world = (
                                self.fullhand_mcc.points_palm_to_world(
                                    self.fullhand_mcc.tip_positions_palm(
                                        base_plan_q
                                    ),
                                    palm_pose,
                                )
                            )
                            base_surface = self.surface_oracle.observe(
                                base_points_world
                            )
                            self.fullhand_search_delta = (
                                self.fullhand_mcc.normal_search_delta(
                                    q_action_order=q_live,
                                    palm_pose_world=palm_pose,
                                    surface_normals_world=(
                                        base_surface.normals_world
                                    ),
                                    missing=self.fullhand_recovery_active,
                                    inward_step=(
                                        mcc_precontact_config.cartesian_step_m
                                    ),
                                    max_joint_step=(
                                        mcc_precontact_config.joint_step_rad
                                    ),
                                )
                            )
                        else:
                            self.fullhand_search_delta = (
                                self.fullhand_mcc.directional_search_delta(
                                q_action_order=q_live,
                                palm_pose_world=palm_pose,
                                inward_directions_world=control_inward_world,
                                missing=self.fullhand_recovery_active,
                                inward_step=(
                                    mcc_precontact_config.cartesian_step_m
                                ),
                                max_joint_step=(
                                    mcc_precontact_config.joint_step_rad
                                ),
                                contact_points_world=live_contact_positions,
                                contact_point_found=point_valid,
                            )
                            )
                        self.fullhand_runtime_recovery_offset = np.clip(
                            self.fullhand_runtime_recovery_offset
                            + self.fullhand_search_delta,
                            -mcc_precontact_config.runtime_recovery_limit_rad,
                            mcc_precontact_config.runtime_recovery_limit_rad,
                        )
                    plan_q = self.fullhand_mcc.clamp_joint_positions(
                        base_plan_q + self.fullhand_runtime_recovery_offset
                    )
                planned_tip_world = self.fullhand_mcc.points_palm_to_world(
                    self.fullhand_mcc.tip_positions_palm(plan_q),
                    palm_pose,
                )
                if mcc_direction_source == "oracle":
                    surface = self.surface_oracle.observe(planned_tip_world)
                    control_outward_world = surface.normals_world
                    self.fullhand_mcc.calibrate_force_sign(
                        live_forces,
                        live_found,
                        surface.normals_world,
                    )
                elif mcc_direction_source != "grasp_closure":
                    self.fullhand_mcc.calibrate_force_sign(
                        live_forces, live_found, control_outward_world
                    )
                if t <= bootstrap_end and not self.fullhand_mcc_calibrated:
                    desired = plan_q
                    self.fullhand_mcc.previous_command = desired.copy()
                elif self.fullhand_mcc_calibrated:
                    # Match the collection-side recovery-state input
                    # (collect_trajectories.py ``contact_observed=loaded``):
                    # the contact state machine counts only loaded contacts
                    # (found AND force >= threshold) as re-established, not
                    # mere geometric grazing.  Without this the deployment
                    # loop confirms a transient touch earlier than the
                    # teacher did.
                    mcc_loaded = live_found & (
                        np.linalg.norm(live_forces, axis=-1) >= contact_threshold
                    )
                    joint_reference_q = self.fullhand_mcc.clamp_joint_positions(
                        plan_q
                        + self.fullhand_servo_offset
                        + mcc_precontact_config.trajectory_tracking_gain
                        * (plan_q - q_live)
                    )
                    desired, mcc_debug = self.fullhand_mcc.update(
                        q_live=q_live,
                        palm_pose_world=palm_pose,
                        force_world=live_forces,
                        found=live_found,
                        # This finger-only port has one position task, unlike
                        # full_hand_mcc which keeps separate surface and
                        # unloaded kinematic targets.  Track the body-fixed
                        # DP site here and use the projection only for its
                        # outward normal; forcing the internal site itself
                        # onto the object surface over-penetrates the pad.
                        # ContactSensor ``pos`` is the geom-to-geom contact
                        # point, not the fingertip site center.  Keep the DP
                        # FK site as the position target and use the live
                        # sensor normal/force for the normal loop.
                        surface_points_world=planned_tip_world,
                        surface_normals_world=control_outward_world,
                        nominal_posture_q=joint_reference_q,
                        force_magnitude_only=(
                            mcc_precontact_config.force_magnitude_only
                        ),
                        contact_points_world=live_contact_positions,
                        use_contact_point_jacobian=(
                            mcc_precontact_config.use_contact_point_jacobian
                        ),
                        contact_observed=mcc_loaded,
                    )
                    mcc_normal_force = mcc_debug["normal_force"]
                    mcc_force_error = mcc_debug["force_error"]
                    mcc_contact_active = mcc_debug["contact_active"]
                    mcc_tip_target_error = mcc_debug["surface_error"]
                    mcc_normal_offset = mcc_debug["normal_offset"]
                    mcc_normal_velocity = mcc_debug["normal_velocity"]
                    mcc_overforce_active = mcc_debug["overforce_active"]
                    mcc_pad_contact_valid = mcc_debug["pad_contact_valid"]
                    mcc_collision_safety_active = mcc_debug[
                        "collision_safety_active"
                    ]
                    mcc_safety_offset = mcc_debug[
                        "overforce_outward_offset"
                    ]
                self.fullhand_last_command_q = np.asarray(
                    desired, dtype=np.float32
                ).copy()
                self.visual_surface_targets[:] = (
                    surface.points_world
                    if mcc_direction_source == "oracle"
                    else planned_tip_world
                )
                self.visual_normals[:] = control_outward_world
                self.visual_contact_points[:] = np.nan
                self.visual_contact_points[live_found] = live_contact_positions[
                    live_found
                ]
                self.visual_tip_points[:] = self.fullhand_mcc.points_palm_to_world(
                    self.fullhand_mcc.tip_positions_palm(q_live), palm_pose
                )
                self.visual_found[:] = live_found
                self.visual_loaded[:] = live_found & (
                    np.linalg.norm(live_forces, axis=-1) >= contact_threshold
                )
            impedance_offset = np.zeros(4, dtype=np.float32)
            impedance_joint = np.zeros(16, dtype=np.float32)
            impedance_force = np.zeros(4, dtype=np.float32)
            impedance_normal = np.zeros((4, 3), dtype=np.float32)
            impedance_predicted_tip = np.zeros((4, 3), dtype=np.float32)
            impedance_predicted_normal = np.zeros(4, dtype=np.float32)
            impedance_contact_state = np.zeros(4, dtype=bool)
            impedance_contact_mode = np.zeros(4, dtype=np.int8)
            impedance_dp_normal_step = np.zeros(4, dtype=np.float32)
            impedance_offset_step = np.zeros(4, dtype=np.float32)
            impedance_nominal_guard = np.zeros(16, dtype=np.float32)
            impedance_nominal_frozen = np.zeros(4, dtype=bool)
            impedance_recovery_steps = np.zeros(4, dtype=np.int64)
            if self.impedance is not None:
                impedance_inputs = {
                    "force_world": live_forces,
                    "normal_world": live_normals,
                    "contact_pos_world": live_contact_positions,
                    "found": live_found,
                    "palm_position_world": self.current_palm_pose[:3],
                    "palm_quaternion_wxyz": self.current_palm_pose[3:7],
                }
                if t <= bootstrap_end:
                    self.impedance.prime(q_nominal=desired, **impedance_inputs)
                else:
                    desired, impedance_debug = self.impedance.update(
                        q_nominal=desired,
                        **impedance_inputs,
                    )
                    impedance_offset = impedance_debug["normal_offset"]
                    impedance_joint = impedance_debug["joint_correction"]
                    impedance_force = impedance_debug["force_magnitude"]
                    impedance_normal = impedance_debug["normal_local"]
                    impedance_predicted_tip = impedance_debug[
                        "predicted_tip_displacement"
                    ]
                    impedance_predicted_normal = impedance_debug[
                        "predicted_normal_displacement"
                    ]
                    impedance_contact_state = impedance_debug["contact_state"]
                    impedance_contact_mode = impedance_debug["contact_mode"]
                    impedance_dp_normal_step = impedance_debug["dp_normal_step"]
                    impedance_offset_step = impedance_debug["offset_step"]
                    impedance_nominal_guard = impedance_debug[
                        "nominal_guard_correction"
                    ]
                    impedance_nominal_frozen = impedance_debug["nominal_frozen"]
                    impedance_recovery_steps = impedance_debug[
                        "recovery_contact_steps"
                    ]
            raw_action = (desired - q_live) / ACTION_SCALE
            raw_action = np.clip(raw_action, -2.0, 2.0)
            q_reference = (
                self.nominal_q
                if self.active_palm_planner is not None
                else data["q_hand"][t]
            )
            q_error = float(np.abs(q_live - q_reference).mean())
            if t >= bootstrap_end:
                self.contact3_frames += int(found_contacts >= 3)
                self.contact4_frames += int(found_contacts >= 4)
                self.per_tip_found_frames += live_found.astype(np.int64)
                self.per_tip_loaded_frames += (
                    live_found
                    & (
                        np.linalg.norm(live_forces, axis=-1)
                        >= contact_threshold
                    )
                ).astype(np.int64)
                self.force_max = max(self.force_max, force_max)
            row: dict[str, float | int | str] = {
                    "mode": mode,
                    "execution_layer": execution_layer,
                    "mcc_direction_source": mcc_direction_source,
                    "teacher_observation_source": teacher_observation_source,
                    "dp_tactile_normal_source": dp_tactile_normal_source,
                    "frame": t,
                    "dp_calls": self.dp_calls,
                    "dp_first_step_mae_rad": self.dp_first_step_mae,
                    "dp_horizon_mae_rad": self.dp_horizon_mae,
                    "q_teacher_mae_rad": q_error,
                    "palm_source": (
                        "active_capsule"
                        if self.active_palm_planner is not None
                        else "teacher"
                    ),
                    "active_palm_progress_m": (
                        self.active_palm_planner.progress_m
                        if self.active_palm_planner is not None
                        else 0.0
                    ),
                    "active_palm_speed_m_s": (
                        self.active_palm_planner.surface_velocity
                        if self.active_palm_planner is not None
                        else 0.0
                    ),
                    "active_palm_contact_paused": int(
                        self.active_palm_planner is not None
                        and self.active_palm_planner.paused_for_contact
                    ),
                    "found_contacts": found_contacts,
                    "loaded_contacts": loaded_contacts,
                    "force_max_N": force_max,
                    "min_contact_distance_m": min_distance,
                    "impedance_offset_max_mm": float(
                        np.abs(impedance_offset).max() * 1000.0
                    ),
                    "impedance_joint_max_rad": float(
                        np.abs(impedance_joint).max()
                    ),
                    "chunk_drop_index": self.chunk_drop_index,
                    "contact_guard_contacts": self.guard_contact_count,
                    "contact_guard_blocked": int(self.contact_guard_blocked),
                    "contact_guard_history": len(self.live_history),
                    "contact_guard_bad_steps": self.contact_guard_bad_steps,
                    "contact_guard_held_fingers": int(
                        np.sum(self.held_tactile_valid & ~guard_contact)
                    ),
                    "fullhand_mcc_calibrated": int(self.fullhand_mcc_calibrated),
                    "fullhand_phase": self.fullhand_phase,
                    "fullhand_contact_settle_streak": (
                        self.fullhand_contact_settle_streak
                    ),
                    "fullhand_search_delta_max_rad": float(
                        np.abs(self.fullhand_search_delta).max()
                    ),
                    "fullhand_servo_offset_max_rad": float(
                        np.abs(self.fullhand_servo_offset).max()
                    ),
                    "fullhand_recovery_fingers": int(
                        self.fullhand_recovery_active.sum()
                    ),
                    "fullhand_runtime_recovery_offset_max_rad": float(
                        np.abs(self.fullhand_runtime_recovery_offset).max()
                    ),
                }
            tip_labels = ("index", "middle", "ring", "thumb")
            for finger, label in enumerate(tip_labels):
                dofs = self.impedance.active_dofs[finger] if self.impedance else ()
                correction_max = (
                    float(np.abs(impedance_joint[dofs]).max())
                    if len(dofs) > 0
                    else 0.0
                )
                row.update(
                    {
                        f"{label}_found": int(live_found[finger]),
                        f"{label}_force_raw_N": float(
                            np.linalg.norm(live_forces[finger])
                        ),
                        f"{label}_force_filtered_N": float(
                            impedance_force[finger]
                        ),
                        f"{label}_contact_state": int(
                            impedance_contact_state[finger]
                        ),
                        f"{label}_contact_mode": int(
                            impedance_contact_mode[finger]
                        ),
                        f"{label}_normal_norm": float(
                            np.linalg.norm(impedance_normal[finger])
                        ),
                        f"{label}_normal_x": float(impedance_normal[finger, 0]),
                        f"{label}_normal_y": float(impedance_normal[finger, 1]),
                        f"{label}_normal_z": float(impedance_normal[finger, 2]),
                        f"{label}_sensor_normal_norm": float(
                            np.linalg.norm(live_normals[finger])
                        ),
                        f"{label}_sensor_normal_x": float(
                            live_normals[finger, 0]
                        ),
                        f"{label}_sensor_normal_y": float(
                            live_normals[finger, 1]
                        ),
                        f"{label}_sensor_normal_z": float(
                            live_normals[finger, 2]
                        ),
                        f"{label}_offset_mm": float(
                            impedance_offset[finger] * 1000.0
                        ),
                        f"{label}_joint_correction_max_rad": correction_max,
                        f"{label}_predicted_normal_mm": float(
                            impedance_predicted_normal[finger] * 1000.0
                        ),
                        f"{label}_predicted_tip_norm_mm": float(
                            np.linalg.norm(impedance_predicted_tip[finger]) * 1000.0
                        ),
                        f"{label}_dp_normal_step_mm": float(
                            impedance_dp_normal_step[finger] * 1000.0
                        ),
                        f"{label}_offset_step_mm": float(
                            impedance_offset_step[finger] * 1000.0
                        ),
                        f"{label}_nominal_frozen": int(
                            impedance_nominal_frozen[finger]
                        ),
                        f"{label}_recovery_contact_steps": int(
                            impedance_recovery_steps[finger]
                        ),
                        f"{label}_mcc_normal_force_N": float(
                            mcc_normal_force[finger]
                        ),
                        f"{label}_mcc_force_error_N": float(
                            mcc_force_error[finger]
                        ),
                        f"{label}_mcc_contact_active": int(
                            mcc_contact_active[finger]
                        ),
                        f"{label}_mcc_tip_target_error_mm": float(
                            mcc_tip_target_error[finger] * 1000.0
                        ),
                        f"{label}_mcc_normal_offset_mm": float(
                            mcc_normal_offset[finger] * 1000.0
                        ),
                        f"{label}_mcc_normal_velocity_mm_s": float(
                            mcc_normal_velocity[finger] * 1000.0
                        ),
                        f"{label}_mcc_overforce_active": int(
                            mcc_overforce_active[finger]
                        ),
                        f"{label}_mcc_pad_contact_valid": int(
                            mcc_pad_contact_valid[finger]
                        ),
                        f"{label}_mcc_collision_safety_active": int(
                            mcc_collision_safety_active[finger]
                        ),
                        f"{label}_mcc_safety_offset_mm": float(
                            mcc_safety_offset[finger] * 1000.0
                        ),
                        f"{label}_mcc_search_delta_max_rad": float(
                            np.abs(
                                self.fullhand_search_delta[
                                    4 * finger : 4 * finger + 4
                                ]
                            ).max()
                        ),
                        f"{label}_mcc_recovery_active": int(
                            self.fullhand_recovery_active[finger]
                        ),
                        f"{label}_mcc_loss_streak": int(
                            self.fullhand_loss_streak[finger]
                        ),
                        f"{label}_mcc_recovery_confirm_streak": int(
                            self.fullhand_recovery_confirm_streak[finger]
                        ),
                        f"{label}_mcc_servo_offset_max_rad": float(
                            np.abs(
                                self.fullhand_servo_offset[
                                    4 * finger : 4 * finger + 4
                                ]
                            ).max()
                        ),
                        f"{label}_mcc_runtime_recovery_offset_max_rad": float(
                            np.abs(
                                self.fullhand_runtime_recovery_offset[
                                    4 * finger : 4 * finger + 4
                                ]
                            ).max()
                        ),
                    }
                )
                for local_index, dof in enumerate(dofs):
                    row[f"{label}_dq{local_index}_rad"] = float(
                        impedance_joint[dof]
                    )
                    row[f"{label}_nominal_guard_dq{local_index}_rad"] = float(
                        impedance_nominal_guard[dof]
                    )
            for joint in range(16):
                row[f"q_live_{joint}"] = float(q_live[joint])
                row[f"q_dp_{joint}"] = float(dp_desired[joint])
                row[f"q_cmd_{joint}"] = float(desired[joint])
                row[f"q_teacher_{joint}"] = float(q_reference[joint])
            self.rows.append(row)
            if rollout_h5 is not None:
                if runtime.input_frame != "palm":
                    raise ValueError(
                        "Closed-loop rollout export currently requires palm-frame DP"
                    )
                self._append_rollout("episode_step", [t])
                self._append_rollout("q_prior_applied", applied_prior_q)
                self._append_rollout("q_cmd_applied", applied_command_q)
                self._append_rollout("q_live", q_live)
                self._append_rollout(
                    "delta_q_comp_applied",
                    applied_command_q - applied_prior_q,
                )
                self._append_rollout(
                    "e_servo",
                    q_live - applied_command_q,
                )
                self._append_rollout("q_prior_next", dp_desired)
                self._append_rollout("q_cmd_next", np.asarray(desired))
                self._append_rollout("teacher_q_hand", data["q_hand"][t])
                # This is the exact state inserted into the causal DP history
                # at the most recent stride-aligned observation.  It differs
                # from live_dp_state when nominal q history is selected.
                if self.latest_dp_observation_state is None:
                    raise RuntimeError("DP observation state was not initialized")
                self._append_rollout(
                    "dp_observation_state",
                    self.latest_dp_observation_state,
                )
                self._append_rollout("live_dp_state", live_state)
                if runtime.state_schema in GEOMETRY_STATE_SCHEMAS:
                    self._append_rollout(
                        "fingertip_contact_pos_palm",
                        live_state[16:28].reshape(4, 3),
                    )
                    self._append_rollout(
                        "fingertip_contact_normal_palm",
                        live_state[28:40].reshape(4, 3),
                    )
                self._append_rollout(
                    "fingertip_contact_mask",
                    live_found.astype(np.float32),
                )
                self._append_rollout(
                    "fingertip_force_palm",
                    _vectors_object_to_palm(
                        live_forces,
                        self.current_palm_pose[3:7],
                    ),
                )
                self._append_rollout(
                    "palm_pose_object",
                    self.current_palm_pose,
                )
                self._append_rollout(
                    "palm_twist_object",
                    self.current_palm_twist,
                )
                if runtime.state_schema in PLANNER_STATE_SCHEMAS:
                    self._append_rollout(
                        "planner_palm_delta_pose_palm",
                        live_state[-runtime.planner_waypoints * 6 :],
                    )
            self.applied_prior_q = dp_desired.copy()
            self.applied_command_q = np.asarray(desired, dtype=np.float32).copy()
            if t % 100 == 0:
                if self.fullhand_mcc is not None:
                    thumb_summary = (
                        f"mcc(Fn={mcc_normal_force[3]:.3f}N "
                        f"err={mcc_force_error[3]:+.3f}N "
                        f"active={int(mcc_contact_active[3])} "
                        f"tip_err={mcc_tip_target_error[3]*1000.0:.2f}mm)"
                    )
                else:
                    thumb_dofs = (
                        self.impedance.active_dofs[3] if self.impedance else ()
                    )
                    thumb_dq = (
                        np.round(impedance_joint[thumb_dofs], 4).tolist()
                        if len(thumb_dofs) > 0
                        else []
                    )
                    thumb_summary = (
                        f"n_norm={np.linalg.norm(impedance_normal[3]):.2f} "
                        f"offset={impedance_offset[3]*1000.0:+.2f}mm "
                        f"pred_n={impedance_predicted_normal[3]*1000.0:+.2f}mm "
                        f"dq={thumb_dq}"
                    )
                print(
                    f"[REPLAY] mode={mode} frame={t:4d} "
                    f"q_mae={q_error:.5f}rad "
                    f"found={found_contacts}/4 loaded={loaded_contacts}/4 "
                    f"force_max={force_max:.2f}N | "
                    f"thumb(found={int(live_found[3])} "
                    f"F={np.linalg.norm(live_forces[3]):.3f}N "
                    f"{thumb_summary})"
                )
            if self.active_palm_planner is not None:
                self.active_palm_planner.step(
                    found_contacts,
                    enabled=(
                        t >= bootstrap_end
                        and (
                            self.fullhand_mcc is None
                            or self.fullhand_mcc_calibrated
                        )
                    ),
                )
            self.frame += 1
            return torch.as_tensor(
                raw_action, device=env.device, dtype=torch.float32
            ).unsqueeze(0)

    policy = DPReplayPolicy()
    if highlight_contacts and execution_layer == "fullhand_mcc":
        base_update_visualizers = env.update_visualizers

        def update_contact_visualizers(visualizer) -> None:
            base_update_visualizers(visualizer)
            if not np.all(np.isfinite(policy.visual_surface_targets)):
                return
            target_colors = (
                (1.0, 0.30, 0.20, 0.95),  # index plan
                (0.20, 0.70, 1.0, 0.95),  # middle plan
                (1.0, 0.80, 0.15, 0.95),  # ring plan
                (0.80, 0.30, 1.0, 0.95),  # thumb plan
            )
            for finger, color in enumerate(target_colors):
                target = policy.visual_surface_targets[finger]
                normal = policy.visual_normals[finger]
                visualizer.add_sphere(target, radius=0.005, color=color)
                visualizer.add_arrow(
                    target,
                    target + 0.025 * normal,
                    color=color,
                    width=0.0025,
                )
                if policy.visual_found[finger]:
                    contact = policy.visual_contact_points[finger]
                    if np.all(np.isfinite(contact)):
                        visualizer.add_sphere(
                            contact,
                            radius=0.009,
                            color=(
                                (0.15, 1.0, 0.20, 1.0)
                                if policy.visual_loaded[finger]
                                else (1.0, 0.50, 0.05, 1.0)
                            ),
                        )
                else:
                    visualizer.add_sphere(
                        policy.visual_tip_points[finger],
                        radius=0.007,
                        color=(1.0, 0.05, 0.05, 1.0),
                    )

        env.update_visualizers = update_contact_visualizers
    try:
        if viewer == "headless":
            for _ in range(frames):
                wrapped.step(policy(wrapped.get_observations()))
        elif viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        elif viewer == "viser":
            ViserPlayViewer(wrapped, policy).run()
        else:
            if video_output is None:
                raise ValueError("video_output is required for video rendering")
            video_output.parent.mkdir(parents=True, exist_ok=True)
            control_dt = float(
                env_cfg.decimation * env_cfg.sim.mujoco.timestep
            )
            next_frame_time = 0.0
            frames_written = 0
            with imageio.get_writer(
                video_output,
                fps=video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            ) as writer:
                for _ in range(frames):
                    wrapped.step(policy(wrapped.get_observations()))
                    sim_time = policy.frame * control_dt
                    if sim_time + 1.0e-9 >= next_frame_time:
                        frame = env.render()
                        if frame is not None:
                            writer.append_data(
                                np.asarray(frame, dtype=np.uint8)
                            )
                            frames_written += 1
                        next_frame_time += 1.0 / video_fps
            print(
                f"[VIDEO] wrote {frames_written} frames to {video_output}",
                flush=True,
            )
    finally:
        write_report(report, policy.rows)
        if rollout_h5 is not None:
            write_closed_loop_rollout(
                rollout_h5,
                policy.rollout_arrays,
                source_file=rollout_source_file,
                source_episode_id=rollout_source_episode_id,
                mode=mode,
                teacher_action_source=teacher_action_source,
                control_dt=float(getattr(runtime.config, "control_dt", 0.01)),
                bootstrap_frames=bootstrap_end,
                input_frame=runtime.input_frame,
                state_schema=runtime.state_schema,
                stride=runtime.stride,
                obs_horizon=runtime.obs_horizon,
                pred_horizon=runtime.pred_horizon,
                dp_history_q_source=dp_history_q_source,
                teacher_observation_source=teacher_observation_source,
                dp_tactile_normal_source=dp_tactile_normal_source,
                live_teacher_takeover_frame=live_teacher_takeover_frame,
            )
            print(f"[ROLLOUT] wrote {rollout_h5}", flush=True)
        active = max(1, frames - bootstrap_end)
        q_errors = np.asarray(
            [
                row["q_teacher_mae_rad"]
                for row in policy.rows
                if int(row["frame"]) >= bootstrap_end
            ],
            dtype=float,
        )
        q_mean = float(q_errors.mean()) if q_errors.size else float("nan")
        q_p95 = (
            float(np.percentile(q_errors, 95))
            if q_errors.size
            else float("nan")
        )
        active_planner_summary = ""
        if policy.active_palm_planner is not None:
            active_planner_summary = (
                " palm_progress="
                f"{1000.0 * policy.active_palm_planner.progress_m:.1f}mm"
                " palm_motion_frames="
                f"{policy.active_palm_planner.motion_steps}"
                " palm_contact_pause_frames="
                f"{policy.active_palm_planner.pause_steps}"
            )
        inference_ms = 1000.0 * np.asarray(
            runtime.inference_seconds, dtype=np.float64
        )
        inference_mean_ms = (
            float(inference_ms.mean()) if inference_ms.size else float("nan")
        )
        inference_p95_ms = (
            float(np.percentile(inference_ms, 95))
            if inference_ms.size
            else float("nan")
        )
        print(
            f"[RESULT] mode={mode} frames={frames} calls={policy.dp_calls} "
            f"q_mae={q_mean:.6f}rad "
            f"q_p95={q_p95:.6f}rad "
            f"contact3={100*policy.contact3_frames/active:.1f}% "
            f"contact4={100*policy.contact4_frames/active:.1f}% "
            f"force_max={policy.force_max:.2f}N "
            f"dp_samples={runtime.samples} "
            f"inference_mean={inference_mean_ms:.1f}ms "
            f"inference_p95={inference_p95_ms:.1f}ms "
            f"tip_found={np.round(100*policy.per_tip_found_frames/active,1).tolist()}% "
            f"tip_loaded={np.round(100*policy.per_tip_loaded_frames/active,1).tolist()}% "
            f"report={report}"
            f"{active_planner_summary}"
        )
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("offline_teacher", "teacher_dp", "live_dp", "collect_executed"),
        default="offline_teacher",
    )
    parser.add_argument(
        "--viewer",
        choices=("headless", "native", "viser", "video"),
        default="headless",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help=(
            "MP4 output used by --viewer video. Defaults to "
            "mcc_finger_compliance_control/outputs/<mode>_episode<ID>.mp4."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera-distance", type=float, default=0.45)
    parser.add_argument("--video-camera-azimuth", type=float, default=45.0)
    parser.add_argument("--video-camera-elevation", type=float, default=-10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument(
        "--dp-samples",
        type=int,
        default=1,
        help=(
            "Independent diffusion samples per replan. Their normalized action "
            "chunks are averaged before denormalization and execution."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-dp-calls", type=int, default=0)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument(
        "--palm-source",
        choices=("teacher", "active_capsule"),
        default="teacher",
        help=(
            "teacher replays palm_pose_object from H5; active_capsule uses "
            "only its first pose and then advances a contact-gated FullHand "
            "meridian palm plan online."
        ),
    )
    parser.add_argument("--active-palm-surface-speed-mm-s", type=float, default=8.0)
    parser.add_argument("--active-palm-travel-mm", type=float, default=40.0)
    parser.add_argument(
        "--active-palm-direction", type=int, choices=(-1, 1), default=1
    )
    parser.add_argument(
        "--active-palm-max-acceleration-mm-s2", type=float, default=40.0
    )
    parser.add_argument("--active-palm-min-contact-fingers", type=int, default=3)
    parser.add_argument(
        "--execution-layer",
        choices=("joint_position", "fullhand_mcc"),
        default="joint_position",
        help=(
            "joint_position sends DP q directly; fullhand_mcc converts DP q "
            "to FK fingertip targets and executes them through the shared "
            "normal-admittance + four-site IK layer."
        ),
    )
    parser.add_argument(
        "--mcc-preset",
        choices=("current", "collection_matched_sensor"),
        default="collection_matched_sensor",
        help=(
            "Named deployment-controller profile. collection_matched_sensor "
            "matches the non-privileged low-level dynamics and force-loop "
            "settings used by the trajectory collector, while retaining "
            "causal tactile normals instead of its mesh oracle. Explicit "
            "CLI options override preset defaults."
        ),
    )
    parser.add_argument(
        "--mcc-desired-force",
        type=float,
        default=1.0,
        help="Fallback desired normal force before per-finger bootstrap calibration.",
    )
    parser.add_argument(
        "--mcc-desired-force-per-finger",
        type=float,
        nargs=4,
        default=None,
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=(
            "Optional per-finger force setpoints in newtons. When omitted, "
            "--mcc-desired-force is used for all four fingertips."
        ),
    )
    parser.add_argument(
        "--mcc-direction-source",
        choices=("oracle", "sensor_normal", "grasp_closure", "hybrid"),
        default="hybrid",
        help=(
            "MCC contact axis: oracle restores the privileged analytic capsule normal; "
            "sensor_normal always uses measured contact "
            "normals; grasp_closure uses only the Jacobian direction toward "
            "the default grasp; hybrid uses live tactile contact normals when "
            "available and grasp closure while contact is missing."
        ),
    )
    parser.add_argument(
        "--allow-privileged-surface-oracle",
        action="store_true",
        help=(
            "Explicitly allow analytic object-surface geometry in MCC or the "
            "legacy active_capsule planner. Disabled by default."
        ),
    )
    parser.add_argument(
        "--mcc-surface-preload-mm",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. Fixed Cartesian preload is no "
            "longer used; precontact is established per finger by Jacobian search."
        ),
    )
    parser.add_argument("--mcc-contact-force-threshold", type=float, default=0.10)
    parser.add_argument("--mcc-contact-settle-frames", type=int, default=3)
    parser.add_argument("--mcc-contact-search-step-mm", type=float, default=0.15)
    parser.add_argument("--mcc-contact-search-step-rad", type=float, default=0.02)
    parser.add_argument("--mcc-contact-search-limit-rad", type=float, default=0.30)
    parser.add_argument("--mcc-finger-servo-load-scale", type=float, default=0.0)
    parser.add_argument("--mcc-finger-tracking-gain", type=float, default=0.0)
    parser.add_argument("--mcc-runtime-loss-frames", type=int, default=5)
    parser.add_argument(
        "--mcc-sensor-normal-memory-frames",
        type=int,
        default=20,
        help=(
            "Frames to retain the last measured tactile normal during contact "
            "loss. This is causal sensor memory, not a surface oracle."
        ),
    )
    parser.add_argument("--mcc-recovery-confirm-frames", type=int, default=3)
    parser.add_argument(
        "--mcc-runtime-recovery-limit-rad", type=float, default=0.08
    )
    parser.add_argument(
        "--mcc-max-normal-offset-mm",
        type=float,
        default=20.0,
        help=(
            "Bidirectional tactile admittance range. The 20 mm default "
            "retains enough inward recovery travel for the current DP/MCC "
            "deployment when the nominal fingertip target misses the surface."
        ),
    )
    parser.add_argument(
        "--mcc-thumb-max-outward-offset-mm", type=float, default=None
    )
    parser.add_argument(
        "--mcc-thumb-max-inward-offset-mm", type=float, default=None
    )
    parser.add_argument("--mcc-posture-cost", type=float, default=0.08)
    parser.add_argument(
        "--mcc-nominal-surface-preload-mm", type=float, default=0.0
    )
    parser.add_argument("--mcc-flexion-synergy-gain", type=float, default=0.0)
    parser.add_argument(
        "--mcc-flexion-synergy-hard-gain", type=float, default=0.0
    )
    parser.add_argument(
        "--mcc-flexion-synergy-max-step-rad", type=float, default=0.03
    )
    parser.add_argument(
        "--mcc-normal-synergy-control",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--mcc-normal-synergy-max-step-rad", type=float, default=0.035
    )
    parser.add_argument(
        "--mcc-force-magnitude-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--mcc-use-contact-point-jacobian",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mcc-command-rate-limit-rad",
        type=float,
        default=0.02,
        help="Maximum per-frame FullHandMCC joint-command change.",
    )
    parser.add_argument(
        "--mcc-command-ema-alpha",
        type=float,
        default=0.65,
        help=(
            "EMA weight of the newest rate-limited MCC command; 1 disables "
            "filtering and smaller values suppress recovery-transition jitter."
        ),
    )
    parser.add_argument(
        "--mcc-recovery-offset-decay",
        type=float,
        default=0.999,
        help="Per-frame decay applied to a supported finger's re-contact offset.",
    )
    parser.add_argument(
        "--mcc-recovery-decay-force-ratio",
        type=float,
        default=1.5,
        help="Only decay re-contact offset above this multiple of force setpoint.",
    )
    parser.add_argument("--hand-servo-stiffness", type=float, default=8.0)
    parser.add_argument("--hand-servo-damping", type=float, default=1.3)
    parser.add_argument("--hand-servo-effort-limit", type=float, default=12.0)
    parser.add_argument("--thumb-servo-stiffness", type=float, default=None)
    parser.add_argument("--thumb-servo-damping", type=float, default=None)
    parser.add_argument("--thumb-servo-effort-limit", type=float, default=None)
    parser.add_argument(
        "--mcc-project-nominal-normal-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--mcc-overforce-trigger-ratio", type=float, default=1.20)
    parser.add_argument("--mcc-overforce-release-ratio", type=float, default=0.90)
    parser.add_argument("--mcc-overforce-hard-ratio", type=float, default=1.4)
    parser.add_argument(
        "--mcc-thumb-overforce-hard-ratio", type=float, default=1.8
    )
    parser.add_argument("--mcc-overforce-retreat-step-mm", type=float, default=0.08)
    parser.add_argument("--mcc-overforce-recovery-step-mm", type=float, default=0.02)
    parser.add_argument("--mcc-overforce-max-offset-mm", type=float, default=20.0)
    parser.add_argument("--mcc-virtual-stiffness", type=float, default=0.0)
    parser.add_argument("--mcc-max-normal-speed-mm-s", type=float, default=20.0)
    parser.add_argument("--mcc-max-normal-acceleration-m-s2", type=float, default=2.0)
    parser.add_argument(
        "--mcc-direct-force-servo",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mcc-force-servo-integral-gain", type=float, default=0.005)
    parser.add_argument("--mcc-force-servo-deadband", type=float, default=0.05)
    parser.add_argument("--mcc-force-servo-max-step-mm", type=float, default=0.08)
    parser.add_argument("--mcc-force-servo-hard-step-mm", type=float, default=0.20)
    parser.add_argument(
        "--mcc-thumb-force-servo-hard-step-mm", type=float, default=0.10
    )
    parser.add_argument("--mcc-force-servo-search-step-mm", type=float, default=0.50)
    parser.add_argument(
        "--mcc-thumb-force-servo-search-step-mm", type=float, default=0.25
    )
    parser.add_argument(
        "--mcc-force-servo-weak-contact-step-mm", type=float, default=0.20
    )
    parser.add_argument("--mcc-force-filter-alpha", type=float, default=0.25)
    parser.add_argument(
        "--mcc-loss-state-machine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the collection controller's transient-loss state machine, "
            "so the force integrator does not independently search while "
            "contact is absent."
        ),
    )
    parser.add_argument("--mcc-transient-loss-frames", type=int, default=6)
    parser.add_argument("--mcc-transient-search-step-mm", type=float, default=0.20)
    parser.add_argument("--mcc-transient-release-step-mm", type=float, default=0.10)
    parser.add_argument(
        "--mcc-natural-flexion-floor",
        type=float,
        default=None,
        help=(
            "Distal-flexion floor passed to the MCC clamp, matching the "
            "collection-side value (-0.30 rad under manifold/inverse "
            "planning).  None disables the floor."
        ),
    )
    parser.add_argument(
        "--highlight-contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Draw DP fingertip plans/normals and live contact markers in "
            "FullHandMCC native, viser, or video viewers."
        ),
    )
    parser.add_argument(
        "--finger-impedance",
        action="store_true",
        help="Enable bounded fingertip-force compliance around the DP pose.",
    )
    parser.add_argument("--force-min", type=float, default=2.2)
    parser.add_argument("--force-max", type=float, default=3.5)
    parser.add_argument(
        "--impedance-stiffness",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. It is ignored because a spring "
            "to zero offset creates steady-state force error."
        ),
    )
    parser.add_argument(
        "--impedance-mass",
        type=float,
        default=0.20,
        help="Deprecated compatibility option; ignored by the first-order force loop.",
    )
    parser.add_argument(
        "--impedance-damping",
        type=float,
        default=25.0,
        help="Deprecated compatibility option; ignored by the first-order force loop.",
    )
    parser.add_argument(
        "--force-error-full-scale",
        type=float,
        default=2.2,
        help="Force error in newtons that commands the maximum normal-offset rate.",
    )
    parser.add_argument("--jacobian-damping", type=float, default=0.01)
    parser.add_argument("--max-normal-offset-mm", type=float, default=6.0)
    parser.add_argument(
        "--max-retreat-offset-mm",
        type=float,
        default=3.0,
        help="Maximum outward normal retreat relative to the DP nominal pose.",
    )
    parser.add_argument("--max-offset-rate-mm", type=float, default=0.05)
    parser.add_argument(
        "--recovery-offset-rate-mm",
        type=float,
        default=0.15,
        help=(
            "Minimum inward fingertip motion per control step during weak/lost "
            "contact, before Jacobian mapping."
        ),
    )
    parser.add_argument(
        "--max-recovery-offset-step-mm",
        type=float,
        default=3.0,
        help=(
            "Maximum per-step offset change used to replace DP's normal "
            "motion with force-controlled motion."
        ),
    )
    parser.add_argument("--max-joint-correction-rad", type=float, default=0.08)
    parser.add_argument("--max-joint-rate-rad", type=float, default=0.03)
    parser.add_argument(
        "--finger-nominal-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze one finger's nominal q during contact loss. By default "
            "this is disabled for contact-geometry DP and enabled for legacy "
            "force-input DP."
        ),
    )
    parser.add_argument(
        "--nominal-release-rate-rad",
        type=float,
        default=0.003,
        help=(
            "Maximum per-joint step when a recovered finger unfreezes and "
            "returns to the current DP nominal."
        ),
    )
    parser.add_argument(
        "--recovery-confirm-steps",
        type=int,
        default=3,
        help="Consecutive target-force frames required before per-finger unfreezing.",
    )
    parser.add_argument(
        "--chunk-execution",
        action="store_true",
        help=(
            "Execute complete DP chunks with DTW alignment and C2 interpolation "
            "instead of discarding every prediction after its first waypoint."
        ),
    )
    parser.add_argument(
        "--dp-replan-interval",
        type=int,
        default=10,
        help="Control frames between DP calls in chunk mode; 10 means 10 Hz.",
    )
    parser.add_argument("--dtw-history-points", type=int, default=6)
    parser.add_argument("--dtw-max-drop", type=int, default=4)
    parser.add_argument(
        "--contact-aware-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In live chunk mode, retain the last reliable per-finger tactile "
            "sample during contact loss while keeping the DP nominal moving."
        ),
    )
    parser.add_argument("--contact-guard-min-fingers", type=int, default=3)
    parser.add_argument(
        "--dp-history-q-source",
        choices=("live", "nominal"),
        default="nominal",
        help=(
            "Joint state fed back to live DP. Dual-track checkpoints "
            "(tip_target_palm / q_ref / tip-* action fields) require 'live': "
            "physical execution/task-est feedback is the trained contract. "
            "'nominal' is the legacy pre-dual-track contract, only reachable "
            "with --allow-dual-track-nominal-history to reproduce the "
            "known-invalid historical A/B."
        ),
    )
    parser.add_argument(
        "--allow-dual-track-nominal-history",
        action="store_true",
        help=(
            "Permit the historical mismatched A/B where a dual-track q_ref "
            "or fingertip-delta checkpoint is fed nominal rather than live "
            "joint history. This is diagnostic only and is rejected by "
            "default in live_dp."
        ),
    )
    parser.add_argument(
        "--teacher-observation-source",
        choices=("teacher", "live_tactile", "teacher_tactile"),
        default="teacher",
        help=(
            "teacher_dp diagnostic: use the complete recorded teacher state, "
            "or keep teacher q/qdot while replacing contact geometry with "
            "current ContactSensor observations. teacher_tactile is the "
            "privileged inverse diagnostic: live q/qdot with recorded tactile."
        ),
    )
    parser.add_argument(
        "--dp-tactile-normal-source",
        choices=("contact_sensor", "source_mesh_oracle"),
        default="source_mesh_oracle",
        help=(
            "Contact-anchored tactile normal. source_mesh_oracle queries the "
            "undecomposed high-resolution surface only at an actual MuJoCo "
            "contact point and feeds the same corrected normal to DP and MCC; "
            "it cannot create/search contact or expose surface distance."
        ),
    )
    parser.add_argument(
        "--teacher-action-source",
        choices=("dp", "recorded"),
        default="dp",
        help=(
            "teacher_dp diagnostic: execute the DP prediction or the exact "
            "recorded teacher q through the same actuator/physics stack."
        ),
    )
    parser.add_argument(
        "--live-teacher-takeover-frame",
        type=int,
        default=-1,
        help=(
            "Recovery-data diagnostic for live_dp: retain the physical state "
            "reached by DP before this frame, then execute the time-aligned "
            "recorded teacher q. Negative disables takeover."
        ),
    )
    parser.add_argument(
        "--rollout-h5",
        type=Path,
        default=None,
        help=(
            "Optional causally aligned closed-loop observation H5. Records "
            "applied prior/command, current live state/tactile, and teacher q."
        ),
    )
    parser.add_argument(
        "--contact-guard-force-threshold",
        type=float,
        default=0.05,
        help="Per-finger force threshold in newtons used by the DP replan guard.",
    )
    parser.add_argument(
        "--contact-guard-bad-grace-steps",
        type=int,
        default=5,
        help="Consecutive weak-contact frames required before blocking replans.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=None)

    # Apply named profiles as parser defaults rather than mutating ``args``
    # after parsing.  This keeps every explicitly supplied CLI option an
    # override of the profile and makes controller A/B tests reproducible.
    requested_preset = "collection_matched_sensor"
    for index, token in enumerate(sys.argv[1:]):
        if token == "--mcc-preset" and index + 2 <= len(sys.argv[1:]):
            requested_preset = sys.argv[1:][index + 1]
        elif token.startswith("--mcc-preset="):
            requested_preset = token.split("=", 1)[1]
    if requested_preset == "collection_matched_sensor":
        parser.set_defaults(
            mcc_desired_force=3.0,
            mcc_desired_force_per_finger=(3.0, 3.0, 3.0, 4.0),
            mcc_contact_search_step_mm=0.15,
            mcc_runtime_loss_frames=6,
            mcc_recovery_confirm_frames=4,
            mcc_max_normal_offset_mm=3.0,
            mcc_thumb_max_inward_offset_mm=6.0,
            mcc_posture_cost=0.15,
            mcc_nominal_surface_preload_mm=3.0,
            mcc_flexion_synergy_gain=0.18,
            mcc_flexion_synergy_hard_gain=0.75,
            mcc_flexion_synergy_max_step_rad=0.025,
            mcc_normal_synergy_control=True,
            mcc_normal_synergy_max_step_rad=0.025,
            mcc_force_magnitude_only=True,
            mcc_use_contact_point_jacobian=True,
            mcc_command_rate_limit_rad=0.18,
            mcc_command_ema_alpha=0.65,
            hand_servo_stiffness=35.0,
            hand_servo_damping=2.5,
            hand_servo_effort_limit=35.0,
            # Collection disabled hard overforce rejection (ratio=1000) to
            # maximize teacher contact coverage.  Replaying that setting in a
            # learned-policy loop is unsafe: an infeasible DP/IK target can
            # load the hard contact into the kN range before MCC retreats.
            # Keep the matched nominal force law, but restore an independent
            # deployment safety threshold.
            mcc_overforce_hard_ratio=1.4,
            mcc_thumb_overforce_hard_ratio=1.8,
            mcc_force_servo_integral_gain=0.003,
            mcc_thumb_force_servo_hard_step_mm=0.20,
            mcc_thumb_force_servo_search_step_mm=0.50,
            mcc_loss_state_machine=True,
            mcc_transient_loss_frames=6,
            mcc_transient_search_step_mm=0.20,
            mcc_transient_release_step_mm=0.25,
            # Collection ran the differential surface planner in
            # manifold/inverse mode, which set natural_flexion_floor=-0.30.
            mcc_natural_flexion_floor=-0.30,
        )
    args = parser.parse_args()
    audit_collection_execution_contract(args.file, args)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    data = load_episode(
        args.file,
        args.episode_id,
        include_teacher_tactile=(
            args.mode != "live_dp"
            or args.teacher_observation_source == "teacher_tactile"
        ),
    )
    replay_object = load_replay_object_metadata(args.file)
    replay_object_config = (
        load_object_config(replay_object.object_id)
        if replay_object.object_id is not None
        else None
    )
    runtime = DPRuntime(
        args.model,
        device,
        args.inference_steps,
        args.seed,
        samples=args.dp_samples,
    )
    if runtime.state_schema == DUAL_TRACK_V3_SCHEMA and (
        args.mode != "live_dp"
        or args.teacher_observation_source == "teacher_tactile"
    ):
        # Filtering can change episode numbering. Match the complete executed
        # joint sequence, not just an ID or initial pose, before loading the
        # exact observation contract used by this checkpoint.
        training_path = Path(runtime.config.file)
        with h5py.File(training_path, "r") as teacher_file:
            ids = np.asarray(teacher_file["episode_id"])
            steps = np.asarray(teacher_file["episode_step"])
            candidates = []
            for row, env_id in np.argwhere(steps == 0):
                if np.allclose(teacher_file["q_hand"][row, env_id], data["q_hand"][0], atol=1e-6, rtol=0):
                    candidates.append(int(ids[row, env_id]))
            matches = []
            for candidate in candidates:
                q = _episode(teacher_file, candidate, "q_hand")
                if q.shape == data["q_hand"].shape and np.allclose(q, data["q_hand"], atol=1e-6, rtol=0):
                    matches.append(candidate)
            if len(matches) != 1:
                raise ValueError(f"Expected one exact teacher episode match in {training_path}; got {matches}")
            fields = str(teacher_file.attrs["state_fields"]).split(",")
            state = np.concatenate([
                _episode(teacher_file, matches[0], field).reshape(len(data["q_hand"]), -1)
                for field in fields
            ], axis=-1)
            if state.shape[1] != 242:
                raise ValueError(f"Invalid dual-track teacher shape: {state.shape}")
            data["teacher_state_dual_track"] = state
            print(f"[TEACHER] exact training fields: {training_path} episode={matches[0]} shape={state.shape}")
    if (
        args.mode == "live_dp"
        and runtime.action_field in (
            "q_ref",
            "tip_delta_tangent_palm",
            "tip_motion_tangent_palm",
            "tip_motion_palm",
            "tip_target_palm",
        )
        and args.dp_history_q_source != "live"
        and not args.allow_dual_track_nominal_history
    ):
        raise ValueError(
            "Dual-track deployment requires --dp-history-q-source live: "
            "these checkpoints require physical execution/task-est feedback. "
            "Pass --allow-dual-track-nominal-history only to reproduce the "
            "known-invalid historical nominal-history A/B."
        )
    if args.live_teacher_takeover_frame >= 0 and args.mode != "live_dp":
        raise ValueError(
            "--live-teacher-takeover-frame is only valid with --mode live_dp"
        )
    if (
        args.teacher_observation_source == "live_tactile"
        and args.mode != "teacher_dp"
    ):
        raise ValueError(
            "--teacher-observation-source live_tactile is only valid with "
            "--mode teacher_dp"
        )
    if (
        args.teacher_observation_source == "teacher_tactile"
        and args.mode != "live_dp"
    ):
        raise ValueError(
            "--teacher-observation-source teacher_tactile is only valid with "
            "--mode live_dp"
        )
    if args.mode == "collect_executed":
        if args.rollout_h5 is None:
            raise ValueError("--mode collect_executed requires --rollout-h5")
        if args.viewer not in ("headless",):
            raise ValueError("--mode collect_executed requires --viewer headless")
        if args.live_teacher_takeover_frame >= 0:
            raise ValueError(
                "--live-teacher-takeover-frame is not valid with "
                "--mode collect_executed"
            )
    if (
        args.mcc_direction_source == "oracle"
        and not args.allow_privileged_surface_oracle
    ):
        raise ValueError(
            "--mcc-direction-source oracle uses privileged analytic surface "
            "geometry; pass --allow-privileged-surface-oracle only for an "
            "explicit oracle A/B test."
        )
    if (
        args.teacher_observation_source == "teacher_tactile"
        and not args.allow_privileged_surface_oracle
    ):
        raise ValueError(
            "--teacher-observation-source teacher_tactile replays privileged "
            "recorded tactile geometry; pass --allow-privileged-surface-oracle "
            "only for an explicit diagnostic A/B test."
        )
    if args.palm_source == "active_capsule":
        if not args.allow_privileged_surface_oracle:
            raise ValueError(
                "The legacy active_capsule planner uses known capsule radius, "
                "half-height and analytic projection. It is disabled in the "
                "sensor-only deployment path."
            )
        if args.mode != "live_dp":
            raise ValueError("--palm-source active_capsule requires --mode live_dp")
        if runtime.state_schema not in PLANNER_STATE_SCHEMAS:
            raise ValueError(
                "Active palm deployment requires a planner-conditioned DP checkpoint"
            )
        if args.execution_layer != "fullhand_mcc":
            raise ValueError(
                "Active palm deployment currently requires "
                "--execution-layer fullhand_mcc"
            )
        if not 1 <= args.active_palm_min_contact_fingers <= 4:
            raise ValueError("--active-palm-min-contact-fingers must be in [1, 4]")
    report = args.report or args.model.parent / (
        f"deploy_{args.mode}_episode{args.episode_id}.csv"
    )
    video_output = args.video_output
    if args.viewer == "video" and video_output is None:
        video_output = Path("mcc_finger_compliance_control/outputs") / (
            f"{args.mode}_episode{args.episode_id}.mp4"
        )
    print(
        f"[INFO] mode={args.mode} episode={args.episode_id} frames={len(data['q_hand'])} "
        f"device={device} input_frame={runtime.input_frame} "
        f"state_schema={runtime.state_schema} state_dim={runtime.state_dim} "
        f"action={runtime.action_representation} "
        f"action_field={runtime.action_field} action_dim={runtime.action_dim} "
        f"stride={runtime.stride} obs={runtime.obs_horizon} "
        f"pred={runtime.pred_horizon} inference={runtime.policy.diffusion.num_inference_steps} "
        f"dp_samples={runtime.samples} "
        f"execution_layer={args.execution_layer} "
        f"palm_source={args.palm_source} "
        f"object={replay_object.object_id or 'capsule_medium_legacy'} "
        f"object_scale={replay_object.object_scale:g} "
        f"mcc_preset={args.mcc_preset} "
        f"mcc_direction={args.mcc_direction_source} "
        f"dp_history_q={args.dp_history_q_source} "
        f"teacher_observation={args.teacher_observation_source} "
        f"dp_tactile_normal={args.dp_tactile_normal_source} "
        f"teacher_action={args.teacher_action_source} "
        f"teacher_takeover={args.live_teacher_takeover_frame} "
        "surface_source="
        + (
            "analytic_capsule_oracle_PRIVILEGED"
            if args.mcc_direction_source == "oracle"
            else (
                "actual_contact_anchored_source_surface_normal"
                if args.dp_tactile_normal_source == "source_mesh_oracle"
                else "live_contact_sensor"
            )
        )
    )
    if args.mode == "live_dp" and not args.allow_privileged_surface_oracle:
        print(
            "[INFORMATION-CONTRACT] CAUSAL_TACTILE: DP/MCC tactile="
            f"{args.dp_tactile_normal_source}; MCC direction="
            f"{args.mcc_direction_source}; joints=live encoders; "
            "the source-surface normal is allowed only after actual contact; "
            "no object pose/surface distance/future geometry/object YAML grasp "
            "is available to DP or FullHandMCC. The H5 palm path is treated "
            "only as an external upper-planner command."
        )
    if args.impedance_stiffness != 0.0:
        print(
            "[WARN] --impedance-stiffness is deprecated and ignored; "
            "persistent contact offset is required for zero steady-state force error."
        )
    if (
        args.impedance_mass != parser.get_default("impedance_mass")
        or args.impedance_damping != parser.get_default("impedance_damping")
    ):
        print(
            "[WARN] --impedance-mass/--impedance-damping are ignored by the "
            "bounded first-order normal-force loop."
        )
    if args.mode == "offline_teacher":
        offline_teacher(data, runtime, args.max_dp_calls, report)
    else:
        run_inverse(
            data,
            runtime,
            args.mode,
            args.viewer,
            device,
            args.max_steps,
            args.max_dp_calls,
            args.contact_threshold,
            (
                FingertipImpedanceConfig(
                    force_min=args.force_min,
                    force_max=args.force_max,
                    force_error_full_scale=args.force_error_full_scale,
                    jacobian_damping=args.jacobian_damping,
                    max_normal_offset=args.max_normal_offset_mm / 1000.0,
                    max_retreat_offset=args.max_retreat_offset_mm / 1000.0,
                    max_offset_rate=args.max_offset_rate_mm / 1000.0,
                    recovery_offset_rate=args.recovery_offset_rate_mm / 1000.0,
                    max_recovery_offset_step=(
                        args.max_recovery_offset_step_mm / 1000.0
                    ),
                    max_joint_correction=args.max_joint_correction_rad,
                    max_joint_rate=args.max_joint_rate_rad,
                    nominal_guard_enabled=(
                        args.finger_nominal_guard
                        if args.finger_nominal_guard is not None
                        else runtime.state_schema not in GEOMETRY_STATE_SCHEMAS
                    ),
                    nominal_release_rate=args.nominal_release_rate_rad,
                    recovery_confirm_steps=args.recovery_confirm_steps,
                )
                if args.finger_impedance
                else None
            ),
            (
                DPChunkSchedulerConfig(
                    control_dt=0.01,
                    waypoint_dt=runtime.stride * 0.01,
                    replan_interval=args.dp_replan_interval,
                    history_points=args.dtw_history_points,
                    max_drop=args.dtw_max_drop,
                )
                if args.chunk_execution
                else None
            ),
            (
                ContactAwareReplanConfig(
                    min_fingers=args.contact_guard_min_fingers,
                    force_threshold=args.contact_guard_force_threshold,
                    bad_grace_steps=args.contact_guard_bad_grace_steps,
                )
                if (
                    args.contact_aware_replan
                    and args.chunk_execution
                    and args.finger_impedance
                )
                else None
            ),
            args.execution_layer,
            args.mcc_direction_source,
            args.mcc_desired_force,
            MCCPrecontactConfig(
                force_threshold=args.mcc_contact_force_threshold,
                desired_force_per_finger=(
                    None
                    if args.mcc_desired_force_per_finger is None
                    else tuple(args.mcc_desired_force_per_finger)
                ),
                settle_frames=args.mcc_contact_settle_frames,
                max_normal_offset_m=(
                    None
                    if args.mcc_max_normal_offset_mm is None
                    else args.mcc_max_normal_offset_mm / 1000.0
                ),
                thumb_max_inward_offset_m=(
                    None
                    if args.mcc_thumb_max_inward_offset_mm is None
                    else args.mcc_thumb_max_inward_offset_mm / 1000.0
                ),
                thumb_max_outward_offset_m=(
                    None
                    if args.mcc_thumb_max_outward_offset_mm is None
                    else args.mcc_thumb_max_outward_offset_mm / 1000.0
                ),
                posture_cost=args.mcc_posture_cost,
                nominal_surface_preload_m=(
                    args.mcc_nominal_surface_preload_mm / 1000.0
                ),
                flexion_synergy_gain=args.mcc_flexion_synergy_gain,
                flexion_synergy_hard_gain=(
                    args.mcc_flexion_synergy_hard_gain
                ),
                flexion_synergy_max_step_rad=(
                    args.mcc_flexion_synergy_max_step_rad
                ),
                normal_synergy_control=args.mcc_normal_synergy_control,
                normal_synergy_max_step_rad=(
                    args.mcc_normal_synergy_max_step_rad
                ),
                force_magnitude_only=args.mcc_force_magnitude_only,
                use_contact_point_jacobian=(
                    args.mcc_use_contact_point_jacobian
                ),
                cartesian_step_m=args.mcc_contact_search_step_mm / 1000.0,
                joint_step_rad=args.mcc_contact_search_step_rad,
                joint_limit_rad=args.mcc_contact_search_limit_rad,
                servo_load_scale=args.mcc_finger_servo_load_scale,
                trajectory_tracking_gain=args.mcc_finger_tracking_gain,
                runtime_loss_frames=args.mcc_runtime_loss_frames,
                sensor_normal_memory_frames=(
                    args.mcc_sensor_normal_memory_frames
                ),
                recovery_confirm_frames=args.mcc_recovery_confirm_frames,
                runtime_recovery_limit_rad=(
                    args.mcc_runtime_recovery_limit_rad
                ),
                command_rate_limit_rad=args.mcc_command_rate_limit_rad,
                command_ema_alpha=args.mcc_command_ema_alpha,
                recovery_offset_decay=args.mcc_recovery_offset_decay,
                recovery_decay_force_ratio=(
                    args.mcc_recovery_decay_force_ratio
                ),
                project_nominal_normal_motion=(
                    args.mcc_project_nominal_normal_motion
                ),
                overforce_trigger_ratio=args.mcc_overforce_trigger_ratio,
                overforce_release_ratio=args.mcc_overforce_release_ratio,
                overforce_hard_ratio=args.mcc_overforce_hard_ratio,
                thumb_overforce_hard_ratio=(
                    args.mcc_thumb_overforce_hard_ratio
                ),
                overforce_retreat_step_m=(
                    args.mcc_overforce_retreat_step_mm / 1000.0
                ),
                overforce_recovery_step_m=(
                    args.mcc_overforce_recovery_step_mm / 1000.0
                ),
                overforce_max_offset_m=(
                    args.mcc_overforce_max_offset_mm / 1000.0
                ),
                virtual_stiffness=args.mcc_virtual_stiffness,
                max_normal_speed_m_s=(
                    args.mcc_max_normal_speed_mm_s / 1000.0
                ),
                max_normal_acceleration_m_s2=(
                    args.mcc_max_normal_acceleration_m_s2
                ),
                use_direct_force_servo=args.mcc_direct_force_servo,
                force_servo_integral_gain=(
                    args.mcc_force_servo_integral_gain
                ),
                force_servo_deadband=args.mcc_force_servo_deadband,
                force_servo_max_step_m=(
                    args.mcc_force_servo_max_step_mm / 1000.0
                ),
                force_servo_hard_step_m=(
                    args.mcc_force_servo_hard_step_mm / 1000.0
                ),
                thumb_force_servo_hard_step_m=(
                    args.mcc_thumb_force_servo_hard_step_mm / 1000.0
                ),
                force_servo_search_step_m=(
                    args.mcc_force_servo_search_step_mm / 1000.0
                ),
                thumb_force_servo_search_step_m=(
                    args.mcc_thumb_force_servo_search_step_mm / 1000.0
                ),
                force_servo_weak_contact_step_m=(
                    args.mcc_force_servo_weak_contact_step_mm / 1000.0
                ),
                force_filter_alpha=args.mcc_force_filter_alpha,
                enable_loss_state_machine=args.mcc_loss_state_machine,
                transient_loss_frames=args.mcc_transient_loss_frames,
                transient_search_step_m=(
                    args.mcc_transient_search_step_mm / 1000.0
                ),
                transient_release_step_m=(
                    args.mcc_transient_release_step_mm / 1000.0
                ),
                natural_flexion_floor=args.mcc_natural_flexion_floor,
            ),
            args.dp_history_q_source,
            args.teacher_observation_source,
            args.dp_tactile_normal_source,
            args.teacher_action_source,
            args.live_teacher_takeover_frame,
            args.rollout_h5,
            args.file,
            args.episode_id,
            (
                ActiveCapsulePalmPlannerConfig(
                    surface_speed_m_s=(
                        args.active_palm_surface_speed_mm_s / 1000.0
                    ),
                    travel_m=args.active_palm_travel_mm / 1000.0,
                    direction=args.active_palm_direction,
                    max_surface_acceleration_m_s2=(
                        args.active_palm_max_acceleration_mm_s2 / 1000.0
                    ),
                    min_contact_fingers=(
                        args.active_palm_min_contact_fingers
                    ),
                )
                if args.palm_source == "active_capsule"
                else None
            ),
            args.hand_servo_stiffness,
            args.hand_servo_damping,
            args.hand_servo_effort_limit,
            args.thumb_servo_stiffness,
            args.thumb_servo_damping,
            args.thumb_servo_effort_limit,
            args.highlight_contacts,
            report,
            video_output,
            args.video_fps,
            args.video_width,
            args.video_height,
            args.video_camera_distance,
            args.video_camera_azimuth,
            args.video_camera_elevation,
            replay_object_config,
            replay_object.object_scale,
        )


if __name__ == "__main__":
    main()
