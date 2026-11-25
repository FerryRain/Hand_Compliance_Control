"""Shared units, fixed experimental settings, and artifact identities."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
FINGERS = ("index", "middle", "ring", "thumb")
VARIANTS = ("A", "B", "C", "D")
DT = .01
PREVIEW = 20


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[..., [4, 5, 6, 3]])


def pose(position, attitude):
    return np.concatenate((position, attitude.as_quat()[..., [3, 0, 1, 2]]), axis=-1)


def unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class Settings:
    cell_m: float = .0045
    surface_epsilon_m: float = .0025
    variance_fraction: float = .025
    max_projection_m: float = .015
    local_radius_m: float = .060
    workspace_radius_m: float = .45
    max_wrist_step_m: float = .135
    max_wrist_turn_rad: float = 1.65
    max_attitude_from_start_rad: float = 3.05
    collision_radius_m: float = .024
    clearance_m: float = .0008
    cluster_m: float = .010
    target_count: int = 8
    scored_points: int = 64
    roll_rad: tuple = (-.40, 0., .40)
    orientation_weight: float = .075
    motion_weight: float = .055
    translation_scale_m: float = .048
    rotation_scale_rad: float = .70
    translation_speed_m_s: float = .046
    rotation_speed_rad_s: float = .95
    replan_ticks: int = 55
    lead_ticks: int = 12
    event_cells: int = 3
    improvement: float = .045
    global_after: int = 3

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_bytes()) if path else {}
        result = cls(**values)
        numbers = np.asarray([v for v in asdict(result).values() if np.isscalar(v)], float)
        if not np.isfinite(numbers).all() or min(numbers) < 0:
            raise ValueError("Settings must be finite and nonnegative")
        for value in (result.cell_m, result.cluster_m, result.target_count, result.scored_points,
                      result.translation_scale_m, result.rotation_scale_rad, result.translation_speed_m_s,
                      result.rotation_speed_rad_s, result.replan_ticks, result.lead_ticks, result.global_after):
            if value <= 0:
                raise ValueError("Distances, sampling budgets and periods must be positive")
        if not result.roll_rad or not np.isfinite(result.roll_rad).all():
            raise ValueError("roll_rad must contain finite angles")
        return result

