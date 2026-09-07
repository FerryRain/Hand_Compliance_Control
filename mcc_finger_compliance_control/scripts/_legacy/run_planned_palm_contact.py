"""Test a planned palm path with a fixed object in the inverse workspace.

This experiment is deliberately separate from ``collect_trajectories.py``.
The object is fixed, the floating Leap Hand palm follows the *forward*
``palm_pose_object`` path, and the fingers start from the controller's open
grasp.  Finger closure is contact-triggered, so a closed posture is never
teleported into the object at initialization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from object_catalog import load_object_config
from replay_inverted import (
    MCC_TIP_NAMES,
    _find_site_index,
    replay_env_cfg,
)
from surface_mcc_finger import (
    FullHandMCCFingerConfig,
    FullHandMCCFingerController,
    GeometrySurfaceOracle,
)


# True reset-open posture in the action/model order
# [1,0,2,3, 5,4,6,7, 9,8,10,11, 12,13,14,15].  The controller's historical
# ``open_grasp_q`` is already partly curled and is unsuitable for collision-
# free placement at a close palm-path frame zero.
RESET_OPEN_Q = np.asarray(
    (
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.57, 0.0, 0.0,
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class PlannedContactTrajectory:
    palm_pose_object: np.ndarray
    object_id: str
    object_scale: float
    finger_q: np.ndarray | None = None
    fingertip_surface_object: np.ndarray | None = None
    fingertip_normal_object: np.ndarray | None = None


def _single_env_dataset(
    value: np.ndarray,
    *,
    trailing_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    expected_ndim = 1 + len(trailing_shape)
    if array.ndim == expected_ndim + 1:
        if array.shape[1] != 1:
            raise ValueError(
                f"{name} accepts one planner environment, got {array.shape}"
            )
        array = array[:, 0]
    if array.ndim != expected_ndim or array.shape[1:] != trailing_shape:
        raise ValueError(
            f"{name} must have shape (T,{','.join(map(str, trailing_shape))}), "
            f"got {array.shape}"
        )
    return array


def _load_plan(path: Path) -> PlannedContactTrajectory:
    with h5py.File(path, "r") as file:
        if "palm_pose_object" not in file:
            raise KeyError(f"{path} has no palm_pose_object dataset")
        pose = _single_env_dataset(
            file["palm_pose_object"],
            trailing_shape=(7,),
            name="palm_pose_object",
        )
        object_id = str(file.attrs.get("object_id", "ycb_mustard"))
        object_scale = float(file.attrs.get("object_scale", 1.0))
        finger_q = (
            _single_env_dataset(
                file["finger_q_plan"],
                trailing_shape=(16,),
                name="finger_q_plan",
            )
            if "finger_q_plan" in file
            else None
        )
        surface = (
            _single_env_dataset(
                file["finger_tip_surface_object"],
                trailing_shape=(4, 3),
                name="finger_tip_surface_object",
            )
            if "finger_tip_surface_object" in file
            else None
        )
        normal = (
            _single_env_dataset(
                file["finger_surface_normal_object"],
                trailing_shape=(4, 3),
                name="finger_surface_normal_object",
            )
            if "finger_surface_normal_object" in file
            else None
        )
    if len(pose) < 2:
        raise ValueError("palm_pose_object must contain at least two frames")
    for name, value in (
        ("finger_q_plan", finger_q),
        ("finger_tip_surface_object", surface),
        ("finger_surface_normal_object", normal),
    ):
        if value is not None and len(value) != len(pose):
            raise ValueError(
                f"{name} has {len(value)} frames but palm path has {len(pose)}"
            )
    return PlannedContactTrajectory(
        palm_pose_object=pose,
        object_id=object_id,
        object_scale=object_scale,
        finger_q=finger_q,
        fingertip_surface_object=surface,
        fingertip_normal_object=normal,
    )


def _world_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for site_name in MCC_TIP_NAMES:
        sensor = env.scene[f"{site_name}_contact"]
        assert isinstance(sensor, ContactSensor)
        force = sensor.data.force
        if force is None:
            columns.append(torch.zeros((env.num_envs, 3), device=env.device))
            continue
        if sensor.data.found is not None:
            force = torch.where(
                sensor.data.found.unsqueeze(-1) > 0,
                force,
                torch.zeros_like(force),
            )
        columns.append(force.sum(dim=1))
    return torch.stack(columns, dim=1)


def _contacts(
    env: ManagerBasedRlEnv,
) -> tuple[np.ndarray, np.ndarray]:
    found = np.zeros(4, dtype=bool)
    positions = np.zeros((4, 3), dtype=np.float64)
    for finger, site_name in enumerate(MCC_TIP_NAMES):
        sensor = env.scene[f"{site_name}_geometry_contact"]
        assert isinstance(sensor, ContactSensor)
        if sensor.data.found is None:
            continue
        found[finger] = bool(torch.any(sensor.data.found[0] > 0).item())
        if found[finger] and sensor.data.pos is not None:
            positions[finger] = sensor.data.pos[0, 0].detach().cpu().numpy()
    return found, positions


def _set_root_pose(
    env: ManagerBasedRlEnv,
    pose_object: np.ndarray,
    root_velocity: np.ndarray | None = None,
) -> None:
    """Object is fixed at identity, so object-frame pose is the world pose."""

    pose = np.asarray(pose_object, dtype=np.float32).reshape(7).copy()
    pose[:3] += env.scene.env_origins[0].detach().cpu().numpy()
    velocity = np.zeros(6, dtype=np.float32)
    if root_velocity is not None:
        velocity[:] = np.asarray(root_velocity, dtype=np.float32).reshape(6)
    state = torch.as_tensor(
        np.concatenate((pose, velocity))[None],
        device=env.device,
        dtype=torch.float32,
    )
    env.scene["robot"].write_root_state_to_sim(state)


def _path_twist_world(path: np.ndarray, dt: float) -> np.ndarray:
    """Return a smooth world-frame feed-forward twist for a pose path."""

    pose = np.asarray(path, dtype=np.float64)
    twist = np.zeros((len(pose), 6), dtype=np.float64)
    twist[:, :3] = np.gradient(pose[:, :3], float(dt), axis=0)
    rotation = R.from_quat(pose[:, (4, 5, 6, 3)])
    if len(pose) > 1:
        # R[k+1] * R[k]^-1 is a world-frame incremental rotation.
        edge = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / float(dt)
        twist[:-1, 3:] = edge
        twist[-1, 3:] = edge[-1]
    return twist


class PalmWrenchTracker:
    """Compliant 6-D root tracker driven by a world-frame PD wrench.

    Only reset initialization writes the free root pose.  During closure and
    motion the palm is a dynamic body: the planned path is a reference and
    contacts can displace it.  This is the inverse-workspace analogue of an
    arm Cartesian servo, without introducing a fictitious kinematic teleport.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        path: np.ndarray,
        *,
        dt: float,
        position_stiffness: float,
        position_damping: float,
        rotation_stiffness: float,
        rotation_damping: float,
        max_force: float,
        max_torque: float,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.path = np.asarray(path, dtype=np.float64)
        self.twist = _path_twist_world(self.path, dt)
        self.kp = float(position_stiffness)
        self.kd = float(position_damping)
        self.kr = float(rotation_stiffness)
        self.dr = float(rotation_damping)
        self.max_force = float(max_force)
        self.max_torque = float(max_torque)
        body_names = [body.name or "" for body in self.robot.data.indexing.bodies]
        root_matches = [
            index
            for index, name in enumerate(body_names)
            if name == "palm_lower" or name.endswith("/palm_lower")
        ]
        if len(root_matches) != 1:
            raise ValueError(
                f"palm_lower missing from robot bodies: {body_names}"
            )
        self.root_body_local = root_matches[0]
        self.last_position_error = np.zeros(3, dtype=np.float64)
        self.last_rotation_error = np.zeros(3, dtype=np.float64)
        self.last_force = np.zeros(3, dtype=np.float64)
        self.last_torque = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if limit > 0.0 and norm > limit:
            return value * (limit / max(norm, 1.0e-12))
        return value

    def apply(self, path_index: int) -> None:
        index = int(np.clip(path_index, 0, len(self.path) - 1))
        target = self.path[index].copy()
        target[:3] += self.env.scene.env_origins[0].detach().cpu().numpy()
        target_twist = self.twist[index]
        actual = (
            self.robot.data.root_link_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        actual_twist = (
            self.robot.data.root_link_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        position_error = target[:3] - actual[:3]
        current_rotation = R.from_quat(actual[[4, 5, 6, 3]])
        target_rotation = R.from_quat(target[[4, 5, 6, 3]])
        rotation_error = (target_rotation * current_rotation.inv()).as_rotvec()
        force = (
            self.kp * position_error
            + self.kd * (target_twist[:3] - actual_twist[:3])
        )
        torque = (
            self.kr * rotation_error
            + self.dr * (target_twist[3:] - actual_twist[3:])
        )
        force = self._clip_norm(force, self.max_force)
        torque = self._clip_norm(torque, self.max_torque)
        force_t = torch.as_tensor(
            force.reshape(1, 1, 3), device=self.env.device, dtype=torch.float32
        )
        torque_t = torch.as_tensor(
            torque.reshape(1, 1, 3), device=self.env.device, dtype=torch.float32
        )
        self.robot.write_external_wrench_to_sim(
            force_t,
            torque_t,
            body_ids=[self.root_body_local],
        )
        self.last_position_error[:] = position_error
        self.last_rotation_error[:] = rotation_error
        self.last_force[:] = force
        self.last_torque[:] = torque


def _closure_targets(
    controller: FullHandMCCFingerController,
    oracle: GeometrySurfaceOracle,
    palm_pose: np.ndarray,
    samples: int,
    fallback_fraction: float,
    q_open: np.ndarray = RESET_OPEN_Q,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the first site-path/surface crossing for each finger."""

    q_open = np.asarray(q_open, dtype=np.float64).reshape(16)
    q_close = controller.grasp_closure_q
    fractions = np.linspace(0.0, 1.0, max(3, samples))
    selected = np.full(4, fallback_fraction, dtype=np.float64)
    hit = np.zeros(4, dtype=bool)
    points = np.zeros((4, 3), dtype=np.float64)
    normals = np.zeros_like(points)
    previous_sd: np.ndarray | None = None
    for fraction in fractions:
        q = q_open + fraction * (q_close - q_open)
        tip_palm = controller.tip_positions_palm(q)
        tip_world = controller.points_palm_to_world(tip_palm, palm_pose)
        observation = oracle.observe(tip_world)
        sd = np.asarray(observation.signed_distance, dtype=np.float64)
        crossed = sd <= 0.0 if previous_sd is None else (
            (previous_sd > 0.0) & (sd <= 0.0)
        )
        for finger in np.flatnonzero(crossed & ~hit):
            hit[finger] = True
            selected[finger] = float(fraction)
            points[finger] = observation.points_world[finger]
            normals[finger] = observation.normals_world[finger]
        previous_sd = sd

    q_target = q_open.copy()
    for finger in range(4):
        block = slice(4 * finger, 4 * finger + 4)
        q_target[block] = q_open[block] + selected[finger] * (
            q_close[block] - q_open[block]
        )
    tip_world = controller.points_palm_to_world(
        controller.tip_positions_palm(q_target), palm_pose
    )
    fallback = oracle.observe(tip_world)
    for finger in np.flatnonzero(~hit):
        points[finger] = fallback.points_world[finger]
        normals[finger] = fallback.normals_world[finger]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return q_target, points, normals, hit


class PlannedPalmContactPolicy:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        trajectory: PlannedContactTrajectory,
        object_config,
        *,
        palm_control: str,
        finger_plan: str,
        manifold_lookahead: int,
        manifold_nominal_alpha: float,
        open_steps: int,
        closure_steps: int,
        stable_contact_frames: int,
        closure_samples: int,
        fallback_fraction: float,
        contact_threshold: float,
        surface_preload_m: float,
        transient_loss_frames: int,
        transient_search_step_m: float,
        palm_position_stiffness: float,
        palm_position_damping: float,
        palm_rotation_stiffness: float,
        palm_rotation_damping: float,
        palm_max_force: float,
        palm_max_torque: float,
        output: Path | None,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.trajectory = trajectory
        self.path = np.asarray(trajectory.palm_pose_object, dtype=np.float32)
        self.finger_q_plan = trajectory.finger_q
        self.fingertip_surface_plan = trajectory.fingertip_surface_object
        self.fingertip_normal_plan = trajectory.fingertip_normal_object
        self.palm_control = str(palm_control)
        self.finger_plan = str(finger_plan)
        self.manifold_lookahead = max(1, int(manifold_lookahead))
        self.manifold_nominal_alpha = float(manifold_nominal_alpha)
        self.open_steps = int(open_steps)
        self.closure_steps = int(closure_steps)
        self.stable_contact_frames = int(stable_contact_frames)
        self.closure_samples = int(closure_samples)
        self.fallback_fraction = float(fallback_fraction)
        self.contact_threshold = float(contact_threshold)
        self.surface_preload_m = float(surface_preload_m)
        self.output = output
        object_grasp = np.asarray(
            object_config.collection.get(
                "pregrasp_q", FullHandMCCFingerConfig().grasp_closure_q
            ),
            dtype=np.float64,
        ).reshape(16)
        self.controller = FullHandMCCFingerController(
            FullHandMCCFingerConfig(
                control_dt=0.01,
                desired_force_per_finger=(3.0, 3.0, 3.0, 4.0),
                enable_loss_state_machine=True,
                transient_loss_frames=int(transient_loss_frames),
                recovery_contact_confirm_frames=2,
                transient_search_step=float(transient_search_step_m),
                transient_release_step=float(transient_search_step_m) * 0.5,
                overforce_hard_ratio=1000.0,
                thumb_overforce_hard_ratio=1000.0,
                force_servo_integral_gain=0.003,
                nominal_surface_preload=0.002,
                grasp_closure_q=tuple(float(value) for value in object_grasp),
                use_lateral_reference_regularizer=True,
                flexion_synergy_gain=0.18,
                flexion_synergy_hard_gain=0.75,
                flexion_synergy_spread_threshold=0.75,
                flexion_synergy_max_step=0.025,
            )
        )
        self.oracle = GeometrySurfaceOracle(
            object_config, scale=trajectory.object_scale
        )
        self.oracle.set_pose(
            np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0))
        )
        self.tip_indices = [_find_site_index(env, name) for name in MCC_TIP_NAMES]
        self.palm_tracker = PalmWrenchTracker(
            env,
            self.path,
            dt=0.01,
            position_stiffness=palm_position_stiffness,
            position_damping=palm_position_damping,
            rotation_stiffness=palm_rotation_stiffness,
            rotation_damping=palm_rotation_damping,
            max_force=palm_max_force,
            max_torque=palm_max_torque,
        )
        self.records: dict[str, list[np.ndarray]] = {
            "palm_pose_object": [],
            "palm_pose_live_object": [],
            "palm_tracking_error": [],
            "q_hand": [],
            "fingertip_force_world": [],
            "fingertip_contact": [],
        }
        self.reset()

    def reset(self) -> None:
        self.phase = "open"
        self.phase_step = 0
        self.path_index = 0
        self.loaded_streak = 0
        self.ever_contact = np.zeros(4, dtype=bool)
        self.contact_q = RESET_OPEN_Q.copy()
        self.command_q = self.contact_q.copy()
        self.controller.reset()
        for values in self.records.values():
            values.clear()
        self.saved = False
        self.done = False
        self.path_reference_index = 0
        _set_root_pose(self.env, self.path[0])
        q = torch.as_tensor(
            RESET_OPEN_Q[None],
            device=self.env.device,
            dtype=torch.float32,
        )
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.env.sim.data.mocap_pos[:, 0, :] = 0.0
        self.env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.env.device
        )
        self.env.sim.forward()
        print(
            f"[PHASE] open hand at palm plan frame 0 for {self.open_steps} steps"
        )

    def _palm_pose_world(self) -> np.ndarray:
        return (
            self.robot.data.root_link_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def _reference_palm_pose_world(self, index: int | None = None) -> np.ndarray:
        frame = self.path_index if index is None else int(index)
        pose = self.path[frame].copy()
        pose[:3] += self.env.scene.env_origins[0].detach().cpu().numpy()
        return pose.astype(np.float64)

    def _apply_palm_control(self) -> None:
        if self.palm_control == "teleport":
            _set_root_pose(self.env, self.path[self.path_index])
            self.env.sim.forward()
        else:
            self.palm_tracker.apply(self.path_index)

    def _geometric_targets(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return privileged q/surface/normal targets for one path frame."""

        index = int(np.clip(index, 0, len(self.path) - 1))
        use_offline = (
            self.finger_plan in ("auto", "offline")
            and self.finger_q_plan is not None
            and self.fingertip_surface_plan is not None
            and self.fingertip_normal_plan is not None
        )
        if self.finger_plan == "offline" and not use_offline:
            raise ValueError(
                "--finger-plan offline requires finger_q_plan and surface "
                "datasets; run optimize_contact_plan.py first"
            )
        if use_offline:
            q = np.asarray(self.finger_q_plan[index], dtype=np.float64)
            surface = np.asarray(
                self.fingertip_surface_plan[index], dtype=np.float64
            ).copy()
            surface += self.env.scene.env_origins[0].detach().cpu().numpy()
            normal = np.asarray(
                self.fingertip_normal_plan[index], dtype=np.float64
            ).copy()
            hit = np.ones(4, dtype=bool)
            return q, surface, normal, hit
        return _closure_targets(
            self.controller,
            self.oracle,
            self._reference_palm_pose_world(index),
            self.closure_samples,
            self.fallback_fraction,
        )

    def _moving_manifold_nominal(
        self,
        q_live: np.ndarray,
        palm_pose_live: np.ndarray,
        surface_world: np.ndarray,
        normal_world: np.ndarray,
        base_nominal_q: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
        """Predict finger motion induced by the known future palm path.

        Surface points are compared in their respective palm frames.  The
        difference is therefore exactly the joint-space motion needed to
        remain on the contact manifold while the palm translates and rotates,
        rather than a delayed response after a force sensor has unloaded.
        """

        future_index = min(
            self.path_index + self.manifold_lookahead,
            len(self.path) - 1,
        )
        future_q, future_surface, _, _ = self._geometric_targets(future_index)
        future_pose = self._reference_palm_pose_world(future_index)
        current_target_palm = self.controller.points_world_to_palm(
            surface_world, palm_pose_live
        )
        future_target_palm = self.controller.points_world_to_palm(
            future_surface, future_pose
        )
        horizon_frames = max(1, future_index - self.path_index)
        target_velocity_palm = (
            future_target_palm - current_target_palm
        ) / (0.01 * horizon_frames)
        normal_palm = self.controller.vectors_world_to_palm(
            normal_world, palm_pose_live
        )
        # The future IK posture stabilizes the QP null space without replacing
        # the differential manifold velocity objective.
        nominal = 0.5 * (
            np.asarray(base_nominal_q, dtype=np.float64)
            + np.asarray(future_q, dtype=np.float64)
        )
        return self.controller.solve_contact_velocity_qp(
            q_live=q_live,
            target_velocity_palm=target_velocity_palm,
            surface_normals_palm=normal_palm,
            nominal_posture_q=nominal,
        )

    def _save(self) -> None:
        if self.saved or self.output is None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output, "w") as file:
            for name, values in self.records.items():
                file.create_dataset(name, data=np.asarray(values))
            file.attrs["control_dt"] = 0.01
            file.attrs["motion_mode"] = "fixed_object_forward_palm"
        self.saved = True
        print(f"[SAVED] {len(self.records['q_hand'])} motion frames -> {self.output}")

    def __call__(self, _obs: dict[str, torch.Tensor]) -> torch.Tensor:
        self._apply_palm_control()
        self.env.sim.data.mocap_pos[:, 0, :] = 0.0
        self.env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.env.device
        )
        q_live = self.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
        if self.done:
            action = np.clip((self.command_q - q_live) / 0.08, -1.0, 1.0)
            return torch.as_tensor(
                action[None], device=self.env.device, dtype=torch.float32
            )
        force_world_t = _world_forces(self.env)
        force_world = force_world_t[0].detach().cpu().numpy().astype(np.float64)
        found, contact_pos = _contacts(self.env)
        magnitude = np.linalg.norm(force_world, axis=1)
        loaded = found & (magnitude >= self.contact_threshold)
        palm_pose = self._palm_pose_world()

        if self.phase == "open":
            self.command_q = RESET_OPEN_Q.copy()
            self.phase_step += 1
            if self.phase_step >= self.open_steps:
                self.phase = "close"
                self.phase_step = 0
                self.controller.reset()
                print("[PHASE] contact-triggered open->grasp closure")
        elif self.phase == "close":
            progress = min(1.0, (self.phase_step + 1) / max(self.closure_steps, 1))
            scheduled = RESET_OPEN_Q + progress * (
                self.controller.grasp_closure_q - RESET_OPEN_Q
            )
            for finger in range(4):
                block = slice(4 * finger, 4 * finger + 4)
                if loaded[finger]:
                    if not self.ever_contact[finger]:
                        self.contact_q[block] = q_live[block]
                    self.ever_contact[finger] = True
                else:
                    # A missing finger alone continues closing. Fingers that
                    # already touch do not follow the remaining closure path
                    # into the mesh.
                    self.contact_q[block] = scheduled[block]
                self.command_q[block] = self.contact_q[block]
            self.loaded_streak = self.loaded_streak + 1 if loaded.all() else 0
            self.phase_step += 1
            if self.loaded_streak >= self.stable_contact_frames:
                self.phase = "move"
                self.phase_step = 0
                self.path_index = 0
                self.controller.reset()
                self.controller.previous_command = q_live.copy()
                print(
                    "[PHASE] four fingertips stable; forward palm motion and recording start"
                )
            elif (
                self.phase_step >= self.closure_steps
                and not loaded.all()
                and self.phase_step % 100 == 0
            ):
                print(
                    "[WARN] closure horizon exhausted without four-tip contact; "
                    f"loaded={loaded.astype(int).tolist()}"
                )
        else:
            q_plan, surface, normals, hit = self._geometric_targets(
                self.path_index
            )
            if np.any(found):
                measured_normals = self.oracle.normals_at_world(contact_pos[found])
                normals[found] = measured_normals
                surface[found] = contact_pos[found]
            q_manifold, manifold_debug = self._moving_manifold_nominal(
                q_live,
                palm_pose,
                surface,
                normals,
                q_plan,
            )
            # The privileged differential plan supplies feed-forward before
            # contact is lost.  Loaded fingers retain a small temporal filter;
            # a missing finger accepts the fresh reachable target immediately.
            for finger in range(4):
                block = slice(4 * finger, 4 * finger + 4)
                alpha = self.manifold_nominal_alpha if loaded[finger] else 1.0
                self.contact_q[block] += alpha * (
                    q_manifold[block] - self.contact_q[block]
                )
            target_surface = surface - self.surface_preload_m * normals
            self.controller.update_contact_point_anchors(
                q_live, palm_pose, contact_pos, loaded
            )
            self.command_q, debug = self.controller.update(
                q_live=q_live,
                palm_pose_world=palm_pose,
                force_world=force_world,
                found=found,
                surface_points_world=target_surface,
                surface_normals_world=normals,
                nominal_posture_q=self.contact_q,
                force_magnitude_only=True,
                contact_points_world=contact_pos,
                use_contact_point_jacobian=True,
                manage_contact_state=True,
                contact_observed=loaded,
            )
            persistent = np.asarray(debug["persistent_loss"], dtype=bool) & hit
            if np.any(persistent):
                self.command_q, _ = self.controller.recover_surface_contacts(
                    q_live=q_live,
                    base_command_q=self.command_q,
                    palm_pose_world=palm_pose,
                    surface_points_world=surface,
                    surface_normals_world=normals,
                    persistent_loss=persistent,
                    surface_preload_m=self.surface_preload_m,
                    nominal_posture_q=self.contact_q,
                )
                self.controller.previous_command = self.command_q.copy()
            self.records["palm_pose_object"].append(self.path[self.path_index].copy())
            live_pose_object = palm_pose.copy()
            live_pose_object[:3] -= (
                self.env.scene.env_origins[0].detach().cpu().numpy()
            )
            self.records["palm_pose_live_object"].append(
                live_pose_object.astype(np.float32)
            )
            if self.palm_control == "pd_wrench":
                tracking_error = np.concatenate(
                    (
                        self.palm_tracker.last_position_error,
                        self.palm_tracker.last_rotation_error,
                    )
                )
            else:
                tracking_error = np.zeros(6, dtype=np.float64)
            self.records["palm_tracking_error"].append(
                tracking_error.astype(np.float32)
            )
            self.records["q_hand"].append(q_live.astype(np.float32))
            self.records["fingertip_force_world"].append(force_world.astype(np.float32))
            self.records["fingertip_contact"].append(loaded.astype(np.float32))
            if self.path_index == len(self.path) - 1:
                self.done = True
                self._save()
            else:
                self.path_index += 1

        if (
            self.phase != "move" and self.phase_step % 100 == 0
        ) or (
            self.phase == "move" and self.path_index % 100 == 0
        ):
            print(
                f"[RUN] phase={self.phase} path={self.path_index}/{len(self.path)-1} "
                f"loaded={loaded.astype(int).tolist()} "
                f"force={np.round(magnitude, 2).tolist()} "
                f"palm_err_mm={1000*np.linalg.norm(self.palm_tracker.last_position_error):.1f} "
                f"palm_err_deg={np.degrees(np.linalg.norm(self.palm_tracker.last_rotation_error)):.1f}"
            )
        action = np.clip((self.command_q - q_live) / 0.08, -1.0, 1.0)
        return torch.as_tensor(action[None], device=self.env.device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-object forward-palm contact-planning experiment"
    )
    parser.add_argument("--planner-file", type=Path, required=True)
    parser.add_argument("--viewer", choices=("headless", "native", "viser"), default="native")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--palm-control",
        choices=("pd_wrench", "teleport"),
        default="pd_wrench",
        help="Dynamic 6-D wrench tracking (default) or legacy pose overwrite.",
    )
    parser.add_argument(
        "--finger-plan",
        choices=("auto", "offline", "online"),
        default="auto",
        help="Use optimized H5 targets when present, or online closure geometry.",
    )
    parser.add_argument("--manifold-lookahead", type=int, default=10)
    parser.add_argument("--manifold-nominal-alpha", type=float, default=0.35)
    parser.add_argument("--open-steps", type=int, default=100)
    parser.add_argument("--closure-steps", type=int, default=300)
    parser.add_argument("--stable-contact-frames", type=int, default=20)
    parser.add_argument("--closure-samples", type=int, default=17)
    parser.add_argument("--fallback-fraction", type=float, default=0.70)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--surface-preload-m", type=float, default=0.002)
    parser.add_argument("--transient-loss-frames", type=int, default=6)
    parser.add_argument("--transient-search-step-m", type=float, default=0.00020)
    parser.add_argument("--physics-substeps", type=int, default=10)
    parser.add_argument("--palm-position-stiffness", type=float, default=2500.0)
    parser.add_argument("--palm-position-damping", type=float, default=70.0)
    parser.add_argument("--palm-rotation-stiffness", type=float, default=60.0)
    parser.add_argument("--palm-rotation-damping", type=float, default=5.0)
    parser.add_argument("--palm-max-force", type=float, default=180.0)
    parser.add_argument("--palm-max-torque", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    trajectory = _load_plan(args.planner_file)
    palm_path = trajectory.palm_pose_object
    object_config = load_object_config(trajectory.object_id)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.physics_substeps <= 0:
        raise ValueError("--physics-substeps must be positive")
    if args.manifold_lookahead <= 0:
        raise ValueError("--manifold-lookahead must be positive")
    if not 0.0 < args.manifold_nominal_alpha <= 1.0:
        raise ValueError("--manifold-nominal-alpha must be in (0,1]")
    env_cfg = replay_env_cfg(
        hand_stiffness=35.0,
        hand_damping=2.5,
        hand_effort_limit=35.0,
        object_config=object_config,
        object_scale=trajectory.object_scale,
    )
    # Preserve the 10 ms control period while resolving mesh seams at the
    # same 1 ms physics step used by the collection environment.
    env_cfg.decimation = args.physics_substeps
    env_cfg.sim.mujoco.timestep = 0.01 / args.physics_substeps
    # The floating-hand inverse workspace represents a wrist pose already
    # supported by an upstream arm controller.  Applying gravity to the free
    # hand here adds an artificial unmodelled arm-load torque and contaminates
    # contact-planning results, so only contact and tracking dynamics remain.
    env_cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    policy = PlannedPalmContactPolicy(
        env,
        trajectory,
        object_config,
        palm_control=args.palm_control,
        finger_plan=args.finger_plan,
        manifold_lookahead=args.manifold_lookahead,
        manifold_nominal_alpha=args.manifold_nominal_alpha,
        open_steps=args.open_steps,
        closure_steps=args.closure_steps,
        stable_contact_frames=args.stable_contact_frames,
        closure_samples=args.closure_samples,
        fallback_fraction=args.fallback_fraction,
        contact_threshold=args.contact_threshold,
        surface_preload_m=args.surface_preload_m,
        transient_loss_frames=args.transient_loss_frames,
        transient_search_step_m=args.transient_search_step_m,
        palm_position_stiffness=args.palm_position_stiffness,
        palm_position_damping=args.palm_position_damping,
        palm_rotation_stiffness=args.palm_rotation_stiffness,
        palm_rotation_damping=args.palm_rotation_damping,
        palm_max_force=args.palm_max_force,
        palm_max_torque=args.palm_max_torque,
        output=args.output,
    )
    try:
        if args.viewer == "headless":
            total = args.open_steps + args.closure_steps + len(palm_path) + 1000
            observation = wrapped.get_observations()
            for _ in range(total):
                action = policy(observation)
                observation, *_ = wrapped.step(action)
                if policy.done:
                    break
            contact = np.asarray(policy.records["fingertip_contact"], dtype=bool)
            if len(contact):
                print(
                    f"[RESULT] frames={len(contact)} all4={100*contact.all(1).mean():.1f}% "
                    f"per_tip={np.round(contact.mean(0), 3).tolist()}"
                )
        elif args.viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        else:
            ViserPlayViewer(wrapped, policy).run()
    finally:
        policy._save()
        wrapped.close()


if __name__ == "__main__":
    main()
