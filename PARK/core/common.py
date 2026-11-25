import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUTPUT = ROOT / "outputs/PARK"
HISTORY = ROOT / "outputs/PARK/training-episodes"
CONFIG = HERE / "config/planner.json"
FINGERS = ("index", "middle", "ring", "thumb")
DT = 0.01
PREVIEW = 20


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def configuration(path=CONFIG):
    cfg = read(path)
    if cfg["version"] != 1:
        raise ValueError("Unsupported PARK configuration")
    return cfg


def planning(cfg):
    values = cfg["planner"]
    for name, value in values.items():
        if not np.isfinite(value).all() or (name != "roll_rad" and np.min(value) < 0):
            raise ValueError(f"Invalid planner parameter {name}")
    positive = set(values) - {"roll_rad", "orientation_weight", "motion_weight", "clearance_m", "gamma",
                              "switch_margin", "variance_m2", "normal_agreement", "anchor_width_fraction"}
    if any(np.min(values[k]) <= 0 for k in positive):
        raise ValueError("Sampling counts, periods, distances and rate limits must be positive")
    return SimpleNamespace(**values)


def rot(pose):
    return Rotation.from_quat(np.asarray(pose)[..., [4, 5, 6, 3]])


def pack(xyz, attitude):
    return np.concatenate((xyz, attitude.as_quat()[..., [3, 0, 1, 2]]), axis=-1)


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def dispersed(points, count):
    """Deterministic farthest-point subset, using the original order for ties."""
    if len(points) <= count:
        return np.arange(len(points))
    chosen = np.empty(count, int)
    nearest = np.full(len(points), np.inf)
    index = np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1))
    for i in range(count):
        chosen[i] = index
        nearest = np.minimum(nearest, np.square(points - points[index]).sum(axis=1))
        nearest[index] = -1
        index = nearest.argmax()
    return chosen
