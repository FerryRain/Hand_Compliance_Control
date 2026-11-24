"""Video typography and prefix-only signed-GPIS evaluation."""
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import ImageFont
import trimesh

from ..acitive_02.full_signed import signed_surface_for_output
from ..i04_geometric_traversal import fingertip_r_gpis as ridge
from ..i04_geometric_traversal.fingertip_surface_report import MESH_SAMPLES, METRIC_SEED, mesh_accuracy

BG, PANEL, INK, MUTED, HAIR = "#101922", "#17242e", "#f2f5f6", "#b0c1c8", "#354550"
TIP_NAMES = ("fingertip_geom", "fingertip_2_geom", "fingertip_3_geom", "thumb_fingertip_geom")


@lru_cache(maxsize=64)
def font(size, bold, mono):
    windows_name = ("consolab.ttf" if bold else "consola.ttf") if mono else ("msyhbd.ttc" if bold else "msyh.ttc")
    sans = "NotoSansCJK-" + ("Bold" if bold else "Regular") + ".ttc"
    candidates = [Path("C:/Windows/Fonts") / windows_name, Path("/usr/share/fonts/opentype/noto") / sans]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise OSError("Install Microsoft YaHei or the Noto Sans CJK fonts to render Chinese video labels")


def label(draw, xy, value, size=18, color=INK, *, bold=False, mono=False, anchor="lm"):
    draw.text(xy, value, font=font(size, bold, mono), fill=color, anchor=anchor)


def read_model(path):
    return mujoco.MjModel.from_binary_path("scene.mjb", {"scene.mjb": Path(path).read_bytes()})


def reference_samples(path):
    mesh = trimesh.load_mesh(path, process=False)
    return trimesh.sample.sample_surface(mesh, MESH_SAMPLES, seed=METRIC_SEED + 1)[0]


def prefix_mesh(points, normals, truth_samples, output):
    ids = ridge.voxel_representatives(points) if len(points) else np.empty(0, dtype=int)
    if len(ids) < 4:
        return None
    if len(ids) > ridge.CONTACT_BUDGET:
        ids = ids[ridge.farthest_indices(points[ids], ridge.CONTACT_BUDGET)]
    try:
        mesh, _, _ = signed_surface_for_output(points[ids], normals[ids])
    except (ValueError, RuntimeError) as exc:
        if any(term in str(exc).lower() for term in ("level", "surface", "ridge", "no supported")):
            return None
        raise
    if not len(mesh.faces):
        return None
    mesh.export(output)
    return mesh_accuracy(mesh, truth_samples)["at_mm"]["5"]["fscore"]
