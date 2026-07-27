"""Validate and deploy the fingertip DP in the object-fixed inverse environment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import h5py
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from dp_dataset import ENV_STATE_DIM, ROBOT_STATE_DIM
from dp_chunk_scheduler import DPChunkScheduler, DPChunkSchedulerConfig
from fingertip_impedance import (
    FingertipImpedanceConfig,
    FingertipImpedanceController,
)
from replay_inverted import MCC_TIP_NAMES, replay_env_cfg
from train_dp import build_policy


Mode = Literal["offline_teacher", "teacher_dp", "live_dp"]
ACTION_SCALE = 0.08


@dataclass
class ContactAwareReplanConfig:
    """Keep live DP causal while preventing contact loss from poisoning history."""

    min_fingers: int = 3
    force_threshold: float = 0.05
    bad_grace_steps: int = 5


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


def load_episode(path: Path, episode_id: int) -> dict[str, np.ndarray]:
    required = (
        "palm_pose_object",
        "q_hand",
        "fingertip_force_object",
        "fingertip_contact_normal_object",
        "palm_twist_object",
        "fingertip_pose_object",
    )
    with h5py.File(path, "r") as file:
        names = required + tuple(
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
        if self.action_representation not in ("delta_q", "absolute_q"):
            raise ValueError(
                "Unsupported checkpoint action_representation="
                f"{self.action_representation!r}"
            )
        self.state_schema = str(
            checkpoint.get(
                "state_schema",
                config.get("state_schema", "force_normal"),
            )
        )
        if self.state_schema not in ("force_normal", "contact_geometry"):
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
        normalization = checkpoint["normalization"]
        self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(normalization["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(
            normalization["action_mean"], dtype=np.float32
        )
        self.action_std = np.asarray(
            normalization["action_std"], dtype=np.float32
        )
        if self.state_mean.shape != (self.state_dim,):
            raise ValueError(
                f"Checkpoint state dim {self.state_mean.shape} "
                f"!= {(self.state_dim,)}"
            )
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @property
    def stride(self) -> int:
        return int(self.config.stride)

    @property
    def obs_horizon(self) -> int:
        return int(self.config.obs_horizon)

    @property
    def pred_horizon(self) -> int:
        return int(self.config.pred_horizon)

    @torch.no_grad()
    def predict(self, history: np.ndarray) -> np.ndarray:
        if history.shape != (self.obs_horizon, self.state_dim):
            raise ValueError(
                f"history shape {history.shape} != "
                f"{(self.obs_horizon, self.state_dim)}"
            )
        normalized = (history - self.state_mean) / self.state_std
        state = torch.as_tensor(
            normalized[:, : self.robot_state_dim],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        environment = torch.as_tensor(
            normalized[:, self.robot_state_dim :],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        global_condition = self.policy.diffusion._prepare_global_conditioning(
            {
                "observation.state": state,
                "observation.environment_state": environment,
            }
        )
        prediction = self.policy.diffusion.conditional_sample(
            1,
            global_cond=global_condition,
            generator=self.generator,
        )[0]
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


def teacher_state(
    data: dict[str, np.ndarray],
    input_frame: str = "object",
    state_schema: str = "force_normal",
) -> np.ndarray:
    normals = data["fingertip_contact_normal_object"]
    twist = data["palm_twist_object"]
    if state_schema == "force_normal":
        tactile = data["fingertip_force_object"]
        contact_mask = None
    elif state_schema == "contact_geometry":
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
    parts = [
        data["q_hand"],
        tactile.reshape(-1, 12),
        normals.reshape(-1, 12),
    ]
    if contact_mask is not None:
        parts.append(contact_mask.reshape(-1, 4))
    parts.append(twist.reshape(-1, 6))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def history_indices(t: int, stride: int, horizon: int) -> np.ndarray:
    return t - stride * np.arange(horizon - 1, -1, -1)


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def offline_teacher(
    data: dict[str, np.ndarray],
    runtime: DPRuntime,
    max_dp_calls: int,
    report: Path,
) -> None:
    state = teacher_state(
        data,
        runtime.input_frame,
        runtime.state_schema,
    )
    q = data["q_hand"]
    first = (runtime.obs_horizon - 1) * runtime.stride
    last = len(q) - runtime.pred_horizon * runtime.stride - 1
    rows: list[dict[str, float | int | str]] = []
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
        predicted_future = (
            q[t][None, :] + prediction
            if runtime.action_representation == "delta_q"
            else prediction
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
        force_sensor = env.scene[f"{site_name}_contact"]
        geometry_sensor = env.scene[f"{site_name}_geometry_contact"]
        if not isinstance(force_sensor, ContactSensor) or not isinstance(
            geometry_sensor, ContactSensor
        ):
            raise TypeError((type(force_sensor), type(geometry_sensor)))
        force_sensor.update(0.0)
        geometry_sensor.update(0.0)
        force_data = force_sensor.data
        geometry_data = geometry_sensor.data
        if force_data.force is None or force_data.found is None:
            continue
        found = force_data.found[0] > 0
        if not bool(found.any()):
            continue
        slot_force = force_data.force[0]
        forces[tip_index] = (
            torch.where(found[:, None], slot_force, torch.zeros_like(slot_force))
            .sum(dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        magnitudes = torch.linalg.vector_norm(slot_force, dim=-1)
        magnitudes = torch.where(
            found, magnitudes, torch.full_like(magnitudes, -1.0)
        )
        slot = int(torch.argmax(magnitudes))
        loaded[tip_index] = True
        if geometry_data.normal is not None:
            normals[tip_index] = (
                geometry_data.normal[0, 0].detach().cpu().numpy()
            )
        if geometry_data.pos is not None:
            positions[tip_index] = (
                geometry_data.pos[0, 0].detach().cpu().numpy()
            )
        if geometry_data.dist is not None:
            distances[tip_index] = float(geometry_data.dist[0, 0])
    return forces, normals, positions, loaded, distances


def run_inverse(
    data: dict[str, np.ndarray],
    runtime: DPRuntime,
    mode: Literal["teacher_dp", "live_dp"],
    viewer: Literal["headless", "native", "viser"],
    device: torch.device,
    max_steps: int,
    max_dp_calls: int,
    contact_threshold: float,
    impedance_config: FingertipImpedanceConfig | None,
    chunk_config: DPChunkSchedulerConfig | None,
    contact_guard_config: ContactAwareReplanConfig | None,
    report: Path,
) -> None:
    frames = len(data["q_hand"])
    if max_steps > 0:
        frames = min(frames, max_steps)
    bootstrap_end = (runtime.obs_horizon - 1) * runtime.stride
    if frames <= bootstrap_end + runtime.stride:
        raise ValueError(
            f"Need more than {bootstrap_end + runtime.stride} frames, got {frames}"
        )
    teacher = teacher_state(
        data,
        runtime.input_frame,
        runtime.state_schema,
    )
    env = ManagerBasedRlEnv(cfg=replay_env_cfg(), device=str(device))
    wrapped = RslRlVecEnvWrapper(env)
    robot = env.scene["robot"]

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
            self.chunk_scheduler = (
                DPChunkScheduler(runtime.action_std, chunk_config)
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

        def _set_palm(self, t: int) -> None:
            pose = torch.as_tensor(
                data["palm_pose_object"][t],
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
            forces, normals, positions, found, distances = live_tip_observation(env)
            q_live = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            state_forces = forces
            state_normals = normals
            state_positions = positions
            state_twist = data["palm_twist_object"][t]
            if runtime.input_frame == "palm":
                palm_pose = data["palm_pose_object"][t]
                palm_quaternion = palm_pose[3:7]
                state_forces = _vectors_object_to_palm(
                    forces, palm_quaternion
                )
                state_normals = _vectors_object_to_palm(
                    normals, palm_quaternion
                )
                state_positions = _points_object_to_palm(
                    positions[None, ...],
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
            tactile_state = (
                state_forces if runtime.state_schema == "force_normal"
                else state_positions
            )
            state_parts = [
                q_live,
                tactile_state.reshape(-1),
                state_normals.reshape(-1),
            ]
            if runtime.state_schema == "contact_geometry":
                state_parts.append(found.astype(np.float32))
            state_parts.append(state_twist)
            state = np.concatenate(state_parts).astype(np.float32)
            magnitude = np.linalg.norm(forces, axis=-1)
            found_count = int(found.sum())
            loaded_count = int(np.sum(found & (magnitude >= contact_threshold)))
            return (
                state,
                forces,
                normals,
                positions,
                found,
                found_count,
                loaded_count,
                float(magnitude.max(initial=0.0)),
                float(distances.min(initial=0.0)),
            )

        def _plan(self, t: int, live_state: np.ndarray) -> None:
            if self.dp_calls >= max_dp_calls > 0:
                return
            if mode == "teacher_dp":
                indices = history_indices(
                    t, runtime.stride, runtime.obs_horizon
                )
                history = teacher[indices]
                q_base = data["q_hand"][t]
            else:
                if len(self.live_history) != runtime.obs_horizon:
                    raise RuntimeError(
                        f"live history has {len(self.live_history)} frames"
                    )
                history = np.stack(self.live_history)
                q_base = self.nominal_q.copy()
            prediction = runtime.predict(history)
            predicted_absolute = (
                q_base[None, :] + prediction
                if runtime.action_representation == "delta_q"
                else prediction
            )
            self.segment_start = self.nominal_q.copy()
            self.segment_target = predicted_absolute[0]
            self.segment_plan_frame = t
            if self.chunk_scheduler is not None:
                self.chunk_drop_index = self.chunk_scheduler.install(
                    predicted_absolute
                )
            self.dp_calls += 1
            teacher_target = data["q_hand"][
                min(t + runtime.stride, len(data["q_hand"]) - 1)
            ]
            prediction_error = float(
                np.abs(self.segment_target - teacher_target).mean()
            )
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
            """Keep nominal q moving and hold only unreliable tactile channels."""
            state = live_state.copy()
            state[:16] = self.nominal_q
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
                if runtime.state_schema == "contact_geometry"
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
                q_teacher = torch.as_tensor(
                    data["q_hand"][t], device=env.device
                ).unsqueeze(0)
                robot.write_joint_state_to_sim(
                    position=q_teacher, velocity=torch.zeros_like(q_teacher)
                )
                self.nominal_q = data["q_hand"][t].copy()
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
                self.live_history.append(
                    self._state_for_dp(
                        live_state,
                        live_forces,
                        live_found,
                    )
                )
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
            ):
                history_ready = len(self.live_history) == runtime.obs_horizon
                if history_ready:
                    self._plan(t, live_state)

            if t <= bootstrap_end:
                desired = data["q_hand"][t]
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
            dp_desired = np.asarray(desired, dtype=np.float32).copy()
            self.nominal_q = dp_desired.copy()
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
                    "palm_position_world": data["palm_pose_object"][t, :3],
                    "palm_quaternion_wxyz": data["palm_pose_object"][t, 3:7],
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
            q_live = robot.data.joint_pos[0].detach().cpu().numpy()
            raw_action = (desired - q_live) / ACTION_SCALE
            raw_action = np.clip(raw_action, -2.0, 2.0)
            q_error = float(np.abs(q_live - data["q_hand"][t]).mean())
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
                    "frame": t,
                    "dp_calls": self.dp_calls,
                    "q_teacher_mae_rad": q_error,
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
                row[f"q_teacher_{joint}"] = float(data["q_hand"][t, joint])
            self.rows.append(row)
            if t % 100 == 0:
                thumb_dofs = (
                    self.impedance.active_dofs[3] if self.impedance else ()
                )
                thumb_dq = (
                    np.round(impedance_joint[thumb_dofs], 4).tolist()
                    if len(thumb_dofs) > 0
                    else []
                )
                print(
                    f"[REPLAY] mode={mode} frame={t:4d} "
                    f"q_mae={q_error:.5f}rad "
                    f"found={found_contacts}/4 loaded={loaded_contacts}/4 "
                    f"force_max={force_max:.2f}N | "
                    f"thumb(found={int(live_found[3])} "
                    f"F={np.linalg.norm(live_forces[3]):.3f}N "
                    f"n_norm={np.linalg.norm(impedance_normal[3]):.2f} "
                    f"offset={impedance_offset[3]*1000.0:+.2f}mm "
                    f"pred_n={impedance_predicted_normal[3]*1000.0:+.2f}mm "
                    f"dq={thumb_dq})"
                )
            self.frame += 1
            return torch.as_tensor(
                raw_action, device=env.device, dtype=torch.float32
            ).unsqueeze(0)

    policy = DPReplayPolicy()
    try:
        if viewer == "headless":
            for _ in range(frames):
                wrapped.step(policy(wrapped.get_observations()))
        elif viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        else:
            ViserPlayViewer(wrapped, policy).run()
    finally:
        write_report(report, policy.rows)
        active = max(1, frames - bootstrap_end)
        q_errors = np.asarray(
            [
                row["q_teacher_mae_rad"]
                for row in policy.rows
                if int(row["frame"]) >= bootstrap_end
            ],
            dtype=float,
        )
        print(
            f"[RESULT] mode={mode} frames={frames} calls={policy.dp_calls} "
            f"q_mae={q_errors.mean():.6f}rad "
            f"q_p95={np.percentile(q_errors,95):.6f}rad "
            f"contact3={100*policy.contact3_frames/active:.1f}% "
            f"contact4={100*policy.contact4_frames/active:.1f}% "
            f"force_max={policy.force_max:.2f}N "
            f"tip_found={np.round(100*policy.per_tip_found_frames/active,1).tolist()}% "
            f"tip_loaded={np.round(100*policy.per_tip_loaded_frames/active,1).tolist()}% "
            f"report={report}"
        )
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("offline_teacher", "teacher_dp", "live_dp"),
        default="offline_teacher",
    )
    parser.add_argument(
        "--viewer", choices=("headless", "native", "viser"), default="headless"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-dp-calls", type=int, default=0)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
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
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    data = load_episode(args.file, args.episode_id)
    runtime = DPRuntime(
        args.model, device, args.inference_steps, args.seed
    )
    report = args.report or args.model.parent / (
        f"deploy_{args.mode}_episode{args.episode_id}.csv"
    )
    print(
        f"[INFO] mode={args.mode} episode={args.episode_id} frames={len(data['q_hand'])} "
        f"device={device} input_frame={runtime.input_frame} "
        f"state_schema={runtime.state_schema} state_dim={runtime.state_dim} "
        f"action={runtime.action_representation} "
        f"stride={runtime.stride} obs={runtime.obs_horizon} "
        f"pred={runtime.pred_horizon} inference={runtime.policy.diffusion.num_inference_steps}"
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
                        else runtime.state_schema != "contact_geometry"
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
            report,
        )


if __name__ == "__main__":
    main()
