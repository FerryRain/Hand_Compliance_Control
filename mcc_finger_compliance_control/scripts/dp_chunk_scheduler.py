"""Low-frequency DP chunk execution with DTW alignment and C2 interpolation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DPChunkSchedulerConfig:
    control_dt: float = 0.01
    waypoint_dt: float = 0.05
    replan_interval: int = 10
    history_points: int = 6
    max_drop: int = 4
    position_weight: float = 1.0
    velocity_weight: float = 0.05
    max_joint_step: float = 0.02


class DPChunkScheduler:
    """Compose delayed DP chunks into a smooth control-rate joint path.

    The implementation follows the deployment idea in Force Policy: retain a
    recent executed trajectory, align a new action chunk against it, drop
    obsolete waypoints, and interpolate the retained chunk.  Here DP waypoints
    arrive at ``waypoint_dt`` while the position/force loop runs at
    ``control_dt``.
    """

    def __init__(
        self,
        action_scale: np.ndarray,
        config: DPChunkSchedulerConfig | None = None,
    ):
        self.config = config or DPChunkSchedulerConfig()
        if self.config.replan_interval < 1:
            raise ValueError("replan_interval must be positive")
        if self.config.waypoint_dt < self.config.control_dt:
            raise ValueError("waypoint_dt must be >= control_dt")
        self.scale = np.maximum(np.asarray(action_scale, dtype=np.float64), 0.02)
        self.history: deque[np.ndarray] = deque(
            maxlen=max(
                3,
                self.config.history_points
                * int(round(self.config.waypoint_dt / self.config.control_dt)),
            )
        )
        self.path = np.empty((0, len(self.scale)), dtype=np.float64)
        self.path_index = 0
        self.last_drop_index = 0

    def observe(self, q_actual: np.ndarray) -> None:
        self.history.append(np.asarray(q_actual, dtype=np.float64).copy())

    def _history_at_waypoint_rate(self) -> np.ndarray:
        if not self.history:
            raise RuntimeError("No executed trajectory is available")
        step = max(1, int(round(self.config.waypoint_dt / self.config.control_dt)))
        history = np.stack(self.history)
        sampled = history[::-step][::-1]
        return sampled[-self.config.history_points :]

    @staticmethod
    def _velocity(path: np.ndarray, dt: float) -> np.ndarray:
        velocity = np.zeros_like(path)
        if len(path) < 2:
            return velocity
        velocity[0] = (path[1] - path[0]) / dt
        velocity[-1] = (path[-1] - path[-2]) / dt
        if len(path) > 2:
            velocity[1:-1] = (path[2:] - path[:-2]) / (2.0 * dt)
        return velocity

    def _dtw_transition_index(self, prediction: np.ndarray) -> int:
        history = self._history_at_waypoint_rate()
        if len(history) < 2 or len(prediction) < 2 or self.config.max_drop <= 0:
            return 0
        predicted = prediction[: min(len(prediction), self.config.max_drop + 1)]
        h_velocity = self._velocity(history, self.config.waypoint_dt)
        p_velocity = self._velocity(predicted, self.config.waypoint_dt)
        position = (
            (history[:, None, :] - predicted[None, :, :]) / self.scale
        )
        velocity_scale = self.scale / self.config.waypoint_dt
        velocity = (
            (h_velocity[:, None, :] - p_velocity[None, :, :]) / velocity_scale
        )
        cost = (
            self.config.position_weight * np.mean(position * position, axis=-1)
            + self.config.velocity_weight * np.mean(velocity * velocity, axis=-1)
        )
        rows, cols = cost.shape
        accumulated = np.full((rows, cols), np.inf, dtype=np.float64)
        path_length = np.ones((rows, cols), dtype=np.int32)
        accumulated[0, 0] = cost[0, 0]
        for i in range(rows):
            for j in range(cols):
                if i == 0 and j == 0:
                    continue
                candidates: list[tuple[float, int]] = []
                if i > 0:
                    candidates.append((accumulated[i - 1, j], path_length[i - 1, j]))
                if j > 0:
                    candidates.append((accumulated[i, j - 1], path_length[i, j - 1]))
                if i > 0 and j > 0:
                    candidates.append(
                        (accumulated[i - 1, j - 1], path_length[i - 1, j - 1])
                    )
                previous_cost, previous_length = min(candidates, key=lambda item: item[0])
                accumulated[i, j] = previous_cost + cost[i, j]
                path_length[i, j] = previous_length + 1
        normalized = accumulated[-1] / path_length[-1]
        # The endpoint must also be close to the current executed state.  This
        # prevents a similar but spatially distant section from being selected.
        endpoint = np.mean(
            ((predicted - history[-1]) / self.scale) ** 2,
            axis=-1,
        )
        return int(np.argmin(normalized + endpoint))

    def _current_derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.history) < 2:
            zero = np.zeros_like(self.scale)
            return zero, zero
        values = np.stack(self.history)
        velocity = (values[-1] - values[-2]) / self.config.control_dt
        if len(values) < 3:
            return velocity, np.zeros_like(velocity)
        acceleration = (
            values[-1] - 2.0 * values[-2] + values[-3]
        ) / (self.config.control_dt**2)
        return velocity, acceleration

    @staticmethod
    def _quintic_segment(
        p0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        p1: np.ndarray,
        v1: np.ndarray,
        a1: np.ndarray,
        duration: float,
        steps: int,
    ) -> np.ndarray:
        v0s, v1s = v0 * duration, v1 * duration
        a0s, a1s = a0 * duration**2, a1 * duration**2
        delta = p1 - p0
        c0 = p0
        c1 = v0s
        c2 = 0.5 * a0s
        c3 = 10 * delta - 6 * v0s - 4 * v1s - 1.5 * a0s + 0.5 * a1s
        c4 = -15 * delta + 8 * v0s + 7 * v1s + 1.5 * a0s - a1s
        c5 = 6 * delta - 3 * v0s - 3 * v1s - 0.5 * a0s + 0.5 * a1s
        s = (np.arange(1, steps + 1, dtype=np.float64) / steps)[:, None]
        return c0 + c1 * s + c2 * s**2 + c3 * s**3 + c4 * s**4 + c5 * s**5

    def install(self, predicted_absolute: np.ndarray) -> int:
        prediction = np.asarray(predicted_absolute, dtype=np.float64)
        if prediction.ndim != 2 or prediction.shape[1] != len(self.scale):
            raise ValueError(f"Invalid prediction shape {prediction.shape}")
        drop = self._dtw_transition_index(prediction)
        retained = prediction[drop:]
        current = np.asarray(self.history[-1], dtype=np.float64)
        waypoints = np.vstack((current, retained))
        waypoint_velocity = self._velocity(waypoints, self.config.waypoint_dt)
        waypoint_acceleration = np.zeros_like(waypoints)
        if len(waypoints) > 2:
            waypoint_acceleration[1:-1] = (
                waypoints[2:] - 2.0 * waypoints[1:-1] + waypoints[:-2]
            ) / (self.config.waypoint_dt**2)
        current_velocity, current_acceleration = self._current_derivatives()
        waypoint_velocity[0] = current_velocity
        waypoint_acceleration[0] = current_acceleration
        steps = max(
            1, int(round(self.config.waypoint_dt / self.config.control_dt))
        )
        segments = [
            self._quintic_segment(
                waypoints[index],
                waypoint_velocity[index],
                waypoint_acceleration[index],
                waypoints[index + 1],
                waypoint_velocity[index + 1],
                waypoint_acceleration[index + 1],
                self.config.waypoint_dt,
                steps,
            )
            for index in range(len(waypoints) - 1)
        ]
        path = np.concatenate(segments, axis=0)
        # A final per-frame guard protects against spline overshoot.  It does
        # not change the DP waypoint sequence itself.
        limited = np.empty_like(path)
        previous = current.copy()
        for index, command in enumerate(path):
            command = np.clip(
                command,
                previous - self.config.max_joint_step,
                previous + self.config.max_joint_step,
            )
            limited[index] = command
            previous = command
        self.path = limited
        self.path_index = 0
        self.last_drop_index = drop
        return drop

    def next_command(self) -> np.ndarray:
        if len(self.path) == 0:
            return np.asarray(self.history[-1], dtype=np.float32)
        index = min(self.path_index, len(self.path) - 1)
        command = self.path[index]
        self.path_index += 1
        return command.astype(np.float32)
