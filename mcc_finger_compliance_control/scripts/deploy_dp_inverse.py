"""Validate and deploy the fingertip DP in the object-fixed inverse environment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
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

from dp_dataset import ENV_STATE_DIM, ROBOT_STATE_DIM, STATE_DIM
from replay_inverted import MCC_TIP_NAMES, replay_env_cfg
from train_dp import build_policy


Mode = Literal["offline_teacher", "teacher_dp", "live_dp"]
ACTION_SCALE = 0.08


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
    names = (
        "palm_pose_object",
        "q_hand",
        "fingertip_force_object",
        "fingertip_contact_normal_object",
        "palm_twist_object",
        "fingertip_pose_object",
    )
    with h5py.File(path, "r") as file:
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
        normalization = checkpoint["normalization"]
        self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(normalization["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(
            normalization["action_mean"], dtype=np.float32
        )
        self.action_std = np.asarray(
            normalization["action_std"], dtype=np.float32
        )
        if self.state_mean.shape != (STATE_DIM,):
            raise ValueError(
                f"Checkpoint state dim {self.state_mean.shape} != {(STATE_DIM,)}"
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
        if history.shape != (self.obs_horizon, STATE_DIM):
            raise ValueError(
                f"history shape {history.shape} != "
                f"{(self.obs_horizon, STATE_DIM)}"
            )
        normalized = (history - self.state_mean) / self.state_std
        state = torch.as_tensor(
            normalized[:, :ROBOT_STATE_DIM],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        environment = torch.as_tensor(
            normalized[:, ROBOT_STATE_DIM:],
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


def teacher_state(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            data["q_hand"],
            data["fingertip_force_object"].reshape(-1, 12),
            data["fingertip_contact_normal_object"].reshape(-1, 12),
            data["palm_twist_object"].reshape(-1, 6),
        ),
        axis=-1,
    ).astype(np.float32)


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
    state = teacher_state(data)
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
        teacher_delta = q[target_indices] - q[t]
        error = prediction - teacher_delta
        rows.append(
            {
                "mode": "offline_teacher",
                "call": call,
                "frame": t,
                "horizon_mae_rad": float(np.abs(error).mean()),
                "first_step_mae_rad": float(np.abs(error[0]).mean()),
                "final_step_mae_rad": float(np.abs(error[-1]).mean()),
                "zero_delta_mae_rad": float(np.abs(teacher_delta).mean()),
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
    baseline = np.asarray([row["zero_delta_mae_rad"] for row in rows], dtype=float)
    print(
        f"[RESULT] mode=offline_teacher calls={len(rows)} "
        f"horizon_mae={values.mean():.6f}rad "
        f"first_step_mae={first_values.mean():.6f}rad "
        f"zero_delta={baseline.mean():.6f}rad report={report}"
    )


def live_tip_observation(
    env: ManagerBasedRlEnv,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forces = np.zeros((4, 3), dtype=np.float32)
    normals = np.zeros((4, 3), dtype=np.float32)
    loaded = np.zeros(4, dtype=bool)
    distances = np.zeros(4, dtype=np.float32)
    for tip_index, site_name in enumerate(MCC_TIP_NAMES):
        sensor = env.scene[f"{site_name}_contact"]
        if not isinstance(sensor, ContactSensor):
            raise TypeError(type(sensor))
        sensor.update(0.0)
        sensor_data = sensor.data
        if sensor_data.force is None or sensor_data.found is None:
            continue
        found = sensor_data.found[0] > 0
        if not bool(found.any()):
            continue
        slot_force = sensor_data.force[0]
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
        if sensor_data.normal is not None:
            normals[tip_index] = (
                sensor_data.normal[0, slot].detach().cpu().numpy()
            )
        if sensor_data.dist is not None:
            distances[tip_index] = float(sensor_data.dist[0, slot])
    return forces, normals, loaded, distances


def run_inverse(
    data: dict[str, np.ndarray],
    runtime: DPRuntime,
    mode: Literal["teacher_dp", "live_dp"],
    viewer: Literal["headless", "native", "viser"],
    device: torch.device,
    max_steps: int,
    max_dp_calls: int,
    contact_threshold: float,
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
    teacher = teacher_state(data)
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
            self.segment_start = data["q_hand"][0].copy()
            self.segment_target = data["q_hand"][0].copy()
            self.segment_plan_frame = bootstrap_end
            self.rows: list[dict[str, float | int | str]] = []
            self.contact3_frames = 0
            self.contact4_frames = 0
            self.force_max = 0.0

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
        ) -> tuple[np.ndarray, int, int, float, float]:
            forces, normals, found, distances = live_tip_observation(env)
            q_live = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            state = np.concatenate(
                (
                    q_live,
                    forces.reshape(-1),
                    normals.reshape(-1),
                    data["palm_twist_object"][t],
                )
            ).astype(np.float32)
            magnitude = np.linalg.norm(forces, axis=-1)
            found_count = int(found.sum())
            loaded_count = int(np.sum(found & (magnitude >= contact_threshold)))
            return (
                state,
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
                q_base = live_state[:16]
            prediction = runtime.predict(history)
            self.segment_start = (
                robot.data.joint_pos[0].detach().cpu().numpy().copy()
            )
            self.segment_target = q_base + prediction[0]
            self.segment_plan_frame = t
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
            env.sim.forward()
            (
                live_state,
                found_contacts,
                loaded_contacts,
                force_max,
                min_distance,
            ) = self._live_state(t)

            if t % runtime.stride == 0:
                self.live_history.append(live_state.copy())
            if (
                t >= bootstrap_end
                and t % runtime.stride == bootstrap_end % runtime.stride
            ):
                self._plan(t, live_state)

            if t <= bootstrap_end:
                desired = data["q_hand"][t]
            else:
                alpha = min(
                    1.0,
                    (t - self.segment_plan_frame + 1) / runtime.stride,
                )
                desired = (
                    (1.0 - alpha) * self.segment_start
                    + alpha * self.segment_target
                )
            q_live = robot.data.joint_pos[0].detach().cpu().numpy()
            raw_action = (desired - q_live) / ACTION_SCALE
            raw_action = np.clip(raw_action, -2.0, 2.0)
            q_error = float(np.abs(q_live - data["q_hand"][t]).mean())
            if t >= bootstrap_end:
                self.contact3_frames += int(found_contacts >= 3)
                self.contact4_frames += int(found_contacts >= 4)
                self.force_max = max(self.force_max, force_max)
            self.rows.append(
                {
                    "mode": mode,
                    "frame": t,
                    "dp_calls": self.dp_calls,
                    "q_teacher_mae_rad": q_error,
                    "found_contacts": found_contacts,
                    "loaded_contacts": loaded_contacts,
                    "force_max_N": force_max,
                    "min_contact_distance_m": min_distance,
                }
            )
            if t % 100 == 0:
                print(
                    f"[REPLAY] mode={mode} frame={t:4d} "
                    f"q_mae={q_error:.5f}rad "
                    f"found={found_contacts}/4 loaded={loaded_contacts}/4 "
                    f"force_max={force_max:.2f}N"
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
            f"force_max={policy.force_max:.2f}N report={report}"
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
        f"device={device} stride={runtime.stride} obs={runtime.obs_horizon} "
        f"pred={runtime.pred_horizon} inference={runtime.policy.diffusion.num_inference_steps}"
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
            report,
        )


if __name__ == "__main__":
    main()
