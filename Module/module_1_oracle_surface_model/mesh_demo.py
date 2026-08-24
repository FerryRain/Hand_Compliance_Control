"""Large Stanford Bunny / user-supplied YCB mesh showcase for Module 1."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from Module.module_1_oracle_surface_model import (
  ContactCandidateRequest,
  MeshSurface,
  OracleSurfaceModel,
)
from Module.module_1_oracle_surface_model.mesh_surface import MeshScalePolicy


DEFAULT_BUNNY = Path(__file__).resolve().parents[1] / "assets" / "stanford_bunny.ply"


def _render_preview(
  surface: MeshSurface,
  candidate_points: np.ndarray,
  output_path: Path,
) -> None:
  matplotlib_cache = Path(tempfile.gettempdir()) / "handcomp-matplotlib"
  matplotlib_cache.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection

  mesh = surface.mesh
  triangles = mesh.triangles
  light_direction = np.array([0.35, -0.45, 1.0])
  light_direction /= np.linalg.norm(light_direction)
  intensity = 0.35 + 0.65 * np.clip(mesh.face_normals @ light_direction, 0.0, 1.0)
  base_color = np.array([0.31, 0.64, 0.86])
  face_colors = np.column_stack(
    [intensity[:, None] * base_color[None, :], np.full(len(intensity), 0.98)]
  )

  figure = plt.figure(figsize=(8, 8), dpi=150)
  axis = figure.add_subplot(111, projection="3d")
  collection = Poly3DCollection(
    triangles,
    facecolors=face_colors,
    edgecolor="none",
    linewidth=0.0,
    alpha=0.96,
  )
  axis.add_collection3d(collection)
  if len(candidate_points):
    axis.scatter(
      candidate_points[:, 0],
      candidate_points[:, 1],
      candidate_points[:, 2],
      c="#ff4f70",
      edgecolors="white",
      linewidths=0.9,
      s=75,
      depthshade=False,
      zorder=10,
      label="contact candidates",
    )

  axis.plot(
    [-0.05, 0.05],
    [-0.145, -0.145],
    [0.008, 0.008],
    color="#ff9f1c",
    linewidth=5.0,
    label="10 cm hand span reference",
  )

  bounds = surface.bounds
  center = np.mean(bounds, axis=0)
  radius = 0.55 * float(np.max(surface.extents))
  axis.set_xlim(center[0] - radius, center[0] + radius)
  axis.set_ylim(center[1] - radius, center[1] + radius)
  axis.set_zlim(0.0, 2.0 * radius)
  axis.set_box_aspect((1.0, 1.0, 1.0))
  axis.view_init(elev=18.0, azim=-58.0)
  axis.set_xlabel("x [m]")
  axis.set_ylabel("y [m]")
  axis.set_zlabel("z [m]")
  extents_text = " x ".join(f"{extent:.3f}" for extent in surface.extents)
  axis.set_title(f"Stanford Bunny scaled for whole-hand motion\n{extents_text} m")
  axis.legend(loc="upper right")
  figure.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, bbox_inches="tight")
  plt.close(figure)


def run_demo(
  mesh_path: Path = DEFAULT_BUNNY,
  *,
  source_up_axis: str = "y",
  target_extent_m: float = 0.30,
  minimum_second_extent_m: float = 0.18,
  preview_path: Path | None = None,
  export_path: Path | None = None,
  seed: int = 7,
) -> dict[str, Any]:
  policy = MeshScalePolicy(target_extent_m, minimum_second_extent_m)
  surface = MeshSurface.from_file(
    mesh_path,
    source_up_axis=source_up_axis,
    scale_policy=policy,
  )
  model = OracleSurfaceModel(surface, version=f"mesh-{mesh_path.stem}-v1")
  rng = np.random.default_rng(seed)
  samples = surface.sample_surface(64, rng)
  residuals: list[float] = []
  outside_correct = 0
  inside_correct = 0
  for point in samples:
    query = model.query_surface(point)
    residuals.append(abs(query.signed_distance))
    outside = model.query_surface(query.point + 0.001 * query.normal)
    inside = model.query_surface(query.point - 0.001 * query.normal)
    outside_correct += int(outside.signed_distance > 0.0)
    inside_correct += int(inside.signed_distance < 0.0)

  request = ContactCandidateRequest(
    finger_id=1,
    workspace_center=[0.22, -0.25, 0.14],
    reach_radius=0.35,
    count=12,
    seed=seed,
  )
  candidates = model.sample_contact_candidates(request)
  candidate_points = np.array(
    [
      candidate.position + 0.008 * candidate.outward_normal
      for candidate in candidates
    ]
  )
  if preview_path is not None:
    _render_preview(surface, candidate_points, preview_path)
  if export_path is not None:
    surface.export(export_path)

  sorted_extents = np.sort(surface.extents)[::-1]
  local_sign_accuracy = (outside_correct + inside_correct) / (2.0 * len(samples))
  passed = (
    abs(sorted_extents[0] - target_extent_m) <= 1e-9
    and sorted_extents[1] >= minimum_second_extent_m - 1e-9
    and abs(float(surface.bounds[0, 2])) <= 1e-12
    and max(residuals) <= 1e-8
    and local_sign_accuracy >= 0.95
    and len(candidates) == request.count
  )
  return {
    "module": "M01_MESH_SHOWCASE",
    "passed": passed,
    "asset": str(mesh_path),
    "source_up_axis": source_up_axis,
    "scale_policy": {
      "target_longest_extent_m": target_extent_m,
      "minimum_second_extent_m": minimum_second_extent_m,
      "uniform_scale_factor": surface.scale_factor,
    },
    "metrics": {
      "vertex_count": surface.vertex_count,
      "face_count": surface.face_count,
      "extents_m": surface.extents.tolist(),
      "ground_z_m": float(surface.bounds[0, 2]),
      "is_watertight": surface.is_watertight,
      "signed_distance_mode": "closest_face_normal_local",
      "local_sign_accuracy": local_sign_accuracy,
      "max_surface_residual_m": max(residuals),
      "candidate_count": len(candidates),
    },
    "outputs": {
      "preview": str(preview_path) if preview_path is not None else None,
      "scaled_mesh": str(export_path) if export_path is not None else None,
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mesh", type=Path, default=DEFAULT_BUNNY)
  parser.add_argument("--source-up-axis", choices=("x", "y", "z"), default="y")
  parser.add_argument("--target-extent-m", type=float, default=0.30)
  parser.add_argument("--minimum-second-extent-m", type=float, default=0.18)
  parser.add_argument(
    "--preview",
    type=Path,
    default=Path("Module/generated/stanford_bunny_preview.png"),
  )
  parser.add_argument("--export", type=Path)
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args()
  result = run_demo(
    args.mesh,
    source_up_axis=args.source_up_axis,
    target_extent_m=args.target_extent_m,
    minimum_second_extent_m=args.minimum_second_extent_m,
    preview_path=args.preview,
    export_path=args.export,
    seed=args.seed,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
