"""Read verified physical replay tracks and select the camera shared by PARK films."""
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ImageFont
from PARK.core.common import read
from PARK.evaluation.settings import digest


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

@lru_cache(maxsize=32)
def face(size, bold=False):
    windows = Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc")
    if windows.is_file():
        return ImageFont.truetype(str(windows), size)
    return ImageFont.truetype("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc", size)


def load_view():
    from Module.wrist_active.supported_field_video import FixedView
    return FixedView


class Track:
    def __init__(self, folder, obj, seed, seconds, method):
        meta = read(folder / "trial.json")
        native = (folder / meta["source"].replace(chr(92), "/")).resolve()
        if meta["method"].casefold() != method.casefold():
            raise ValueError(f"Wrong method at {folder}")
        rec = read(native / "run.json")
        self.sha = digest(native / "trace.npz")
        if self.sha != rec["trace_sha256"] or self.sha != meta["trace_sha256"]:
            raise ValueError(f"Trace checksum mismatch at {native}")
        if (meta["object"], meta["seed"], meta["seconds"], rec["steps"]) != (obj, seed, seconds, seconds * 100):
            raise ValueError(f"Wrong trial at {native}")
        self.initial, self.model = rec["initial_sha256"], rec["model_sha256"]
        with np.load(native / "trace.npz", allow_pickle=False) as data:
            self.q, self.points, self.live = data["qpos"], data["points_m"], data["pad_valid"]
            self.light = self.live & (data["region_load_n"].max(axis=1) < 8)[:, None]
        if self.q.shape != (seconds * 100 + 1, 23) or not np.isfinite(self.q).all():
            raise ValueError(f"Incomplete hand posture at {native}")
        coverage = read_csv(folder / "coverage-motion.csv")
        np.testing.assert_array_equal([float(x["seconds"]) for x in coverage], np.arange(seconds + 1))
        self.c25 = np.array([float(x["c25"]) for x in coverage])
        end = read(folder / "metrics.json")["final"]
        np.testing.assert_allclose(self.c25[-1], end["c25"], atol=1e-9)
        self.f5 = {int(x["seconds"]): float(x["f5"]) for x in read_csv(folder / "reconstruction.csv")}
        np.testing.assert_allclose(self.f5[seconds], end["f5"], atol=1e-9)
        self.events, self.endpoints = [], {}

    def state(self, tick):
        return None, None

    def goal_status(self, event, tick):
        return "none"
