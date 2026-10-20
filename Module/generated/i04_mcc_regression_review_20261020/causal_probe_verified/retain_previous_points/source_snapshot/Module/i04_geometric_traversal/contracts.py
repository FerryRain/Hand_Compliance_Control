"""Frozen geometric target and experiment contracts for the new I04."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Cell = Literal['A-MCC', 'B-MCC']
PROTOCOL_VERSION = 'i04-geometric-mcc-contact-recovery.v2'
SURFACE_VERSION = 'fixed-object-source-mesh.v2'


def frozen_array(value, shape):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'Expected finite array with shape {shape}')
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class WristTarget:
    """A exposes only a geometric palm pose; no finger or contact answers."""
    pose_world: NDArray[np.float64]

    def __post_init__(self):
        pose = frozen_array(self.pose_world, (7,))
        if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-7):
            raise ValueError('Scalar-first quaternion must be unit length')
        object.__setattr__(self, 'pose_world', pose)


@dataclass(frozen=True, slots=True)
class WholeHandTarget:
    """B adds only Cartesian fingertip positions to the same wrist target."""
    wrist: WristTarget
    fingertip_positions_world_m: NDArray[np.float64]

    def __post_init__(self):
        object.__setattr__(self, 'fingertip_positions_world_m',
                           frozen_array(self.fingertip_positions_world_m, (4, 3)))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    object_name: str = 'mustard'
    task: str = 'traverse'
    recovery_enabled: bool = True
    seed: int = 7
    dt_s: float = 0.002
    acquisition_s: float = 5.0
    duration_s: float = 60.0
    coverage_radius_m: float = 0.025
    contact_threshold_n: float = 0.20
    mesh_residual_limit_m: float = 0.005
    normal_alignment_cos: float = 0.5
    desired_force_n: float = 2.0
    hard_force_n: float = 12.0
    hard_force_duration_s: float = 0.02
    joint_margin_rad: float = 0.015
    max_joint_command_speed_rad_s: float = 1.2
    wrist_speed_m_s: float = 0.012
    wrist_rotation_speed_rad_s: float = 0.18
    trajectory_joint_speed_rad_s: float = 0.6
    goal_position_tolerance_m: float = 0.002
    goal_rotation_tolerance_rad: float = 0.04
    fingertip_tolerance_m: float = 0.003
    prefix_s: float = 0.10
    log_stride: int = 5
    contact_loss_debounce_s: float = 0.10
    recovery_confirmation_s: float = 0.15
    recovery_timeout_s: float = 8.0
    recovery_cooldown_s: float = 0.50
    recovery_joint_speed_rad_s: float = 0.25
    recovery_wrist_speed_m_s: float = 0.012
    force_relief_n: float = 8.0
    force_relief_speed_m_s: float = 0.030
    withdrawal_at_s: float = 0.0
    withdrawal_distance_m: float = 0.025

    def __post_init__(self):
        if not np.isclose(self.dt_s, 0.002):
            raise ValueError('Physics timestep is frozen at 2 ms')
        if self.duration_s <= self.acquisition_s:
            raise ValueError('Duration must exceed acquisition')
        for key, value in self.__dataclass_fields__.items():
            if key not in ('seed', 'normal_alignment_cos', 'object_name', 'task', 'recovery_enabled', 'withdrawal_at_s') and (not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0):
                raise ValueError(f'{key} must be finite and positive')
        if not 0 < self.normal_alignment_cos <= 1:
            raise ValueError('Normal alignment must be in (0,1]')
        if (self.object_name not in ('mustard', 'bunny') or self.task not in ('traverse', 'hold')
                or not np.isfinite(self.withdrawal_at_s) or self.withdrawal_at_s < 0):
            raise ValueError('Invalid object or withdrawal time')


def array_digest(value) -> str:
    return sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def json_digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
