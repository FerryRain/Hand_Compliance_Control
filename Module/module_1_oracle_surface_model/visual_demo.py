"""Visual explanation of surface, normal, clearance, and candidate queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Module.module_1_oracle_surface_model import (
  CapsuleLink,
  ContactCandidateRequest,
  MeshScalePolicy,
  MeshSurface,
  OracleSurfaceModel,
  Plane,
  Sphere,
)
from Module.module_1_oracle_surface_model.mesh_demo import DEFAULT_BUNNY
from Module.visualization import COLORS, get_pyplot, save_figure


def _add_bunny_panel(axis, surface: MeshSurface, model: OracleSurfaceModel) -> int:
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection

  mesh = surface.mesh
  light = np.array([0.35, -0.45, 1.0])
  light /= np.linalg.norm(light)
  intensity = 0.30 + 0.70 * np.clip(mesh.face_normals @ light, 0.0, 1.0)
  base = np.array([0.25, 0.60, 0.83])
  face_colors = np.column_stack(
    [intensity[:, None] * base[None, :], np.full(len(intensity), 0.97)]
  )
  axis.add_collection3d(
    Poly3DCollection(
      mesh.triangles,
      facecolors=face_colors,
      edgecolor="none",
      linewidth=0.0,
    )
  )

  request = ContactCandidateRequest(
    finger_id=1,
    workspace_center=[0.22, -0.25, 0.14],
    reach_radius=0.35,
    count=12,
    seed=7,
  )
  candidates = model.sample_contact_candidates(request)
  points = np.asarray([candidate.position for candidate in candidates])
  normals = np.asarray([candidate.outward_normal for candidate in candidates])
  axis.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    c=COLORS["pink"],
    edgecolors="white",
    linewidths=0.8,
    s=54,
    depthshade=False,
    label="MAKE candidates",
  )
  axis.quiver(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    normals[:, 0],
    normals[:, 1],
    normals[:, 2],
    length=0.025,
    normalize=True,
    color=COLORS["orange"],
    linewidth=1.1,
    arrow_length_ratio=0.35,
    label="outward normals",
  )
  axis.plot(
    [-0.05, 0.05],
    [-0.145, -0.145],
    [0.008, 0.008],
    color=COLORS["orange"],
    linewidth=4.0,
    label="10 cm hand span",
  )

  bounds = surface.bounds
  center = np.mean(bounds, axis=0)
  radius = 0.57 * float(np.max(surface.extents))
  axis.set_xlim(center[0] - radius, center[0] + radius)
  axis.set_ylim(center[1] - radius, center[1] + radius)
  axis.set_zlim(0.0, 2.0 * radius)
  axis.set_box_aspect((1.0, 1.0, 1.0))
  axis.view_init(elev=18.0, azim=-58.0)
  axis.set_xlabel("x [m]")
  axis.set_ylabel("y [m]")
  axis.set_zlabel("z [m]")
  axis.set_title("A  Large object + reachable contact candidates", loc="left")
  axis.legend(loc="upper right", fontsize=8)
  return len(candidates)


def _add_surface_query_panel(axis) -> None:
  model = OracleSurfaceModel(Sphere([0.0, 0.0, 0.0], 0.10), version="sphere-v1")
  angles = np.linspace(0.0, 2.0 * np.pi, 361)
  axis.fill(
    1000.0 * 0.10 * np.cos(angles),
    1000.0 * 0.10 * np.sin(angles),
    color="#DDEFF7",
    edgecolor=COLORS["blue"],
    linewidth=2.0,
  )
  query_points = (
    np.array([0.145, 0.045, 0.0]),
    np.array([0.050, -0.020, 0.0]),
  )
  for index, point in enumerate(query_points):
    query = model.query_surface(point)
    color = COLORS["orange"] if query.signed_distance > 0.0 else COLORS["purple"]
    axis.scatter(1000.0 * point[0], 1000.0 * point[1], s=54, c=color, zorder=4)
    axis.scatter(
      1000.0 * query.point[0],
      1000.0 * query.point[1],
      s=45,
      c=COLORS["pink"],
      edgecolors="white",
      zorder=5,
    )
    axis.annotate(
      "",
      xy=(1000.0 * query.point[0], 1000.0 * query.point[1]),
      xytext=(1000.0 * point[0], 1000.0 * point[1]),
      arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.8},
    )
    normal_end = query.point + 0.035 * query.normal
    axis.annotate(
      "",
      xy=(1000.0 * normal_end[0], 1000.0 * normal_end[1]),
      xytext=(1000.0 * query.point[0], 1000.0 * query.point[1]),
      arrowprops={"arrowstyle": "-|>", "color": COLORS["green"], "linewidth": 1.4},
    )
    sign = "+" if query.signed_distance > 0.0 else "−"
    axis.text(
      1000.0 * point[0] + 4.0,
      1000.0 * point[1] + (5.0 if index == 0 else -12.0),
      f"query\nd = {sign}{abs(1000.0 * query.signed_distance):.1f} mm",
      fontsize=8,
      color=color,
    )
  axis.text(-97.0, 87.0, "pink: closest point", fontsize=8, color=COLORS["pink"])
  axis.text(-97.0, 75.0, "green: outward normal", fontsize=8, color=COLORS["green"])
  axis.set_aspect("equal")
  axis.set_xlim(-115.0, 175.0)
  axis.set_ylim(-120.0, 120.0)
  axis.set_xlabel("x [mm]")
  axis.set_ylabel("y [mm]")
  axis.set_title("B  querySurface(x) + queryNormal(x)", loc="left")


def _add_clearance_panel(axis) -> float:
  model = OracleSurfaceModel(
    Plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    version="plane-v1",
  )
  link = CapsuleLink([-0.075, 0.0, 0.040], [0.080, 0.0, 0.026], 0.010, "middle_link")
  result = model.query_clearance(link)
  axis.axhspan(-8.0, 0.0, color="#DDEFF7", alpha=0.9)
  axis.axhline(0.0, color=COLORS["blue"], linewidth=2.2, label="known surface")
  axis.plot(
    1000.0 * np.array([link.start[0], link.end[0]]),
    1000.0 * np.array([link.start[2], link.end[2]]),
    color=COLORS["navy"],
    linewidth=14.0,
    solid_capstyle="round",
    label="capsule link",
  )
  axis.plot(
    [1000.0 * result.link_point[0], 1000.0 * result.surface_point[0]],
    [1000.0 * (result.link_point[2] - link.radius), 1000.0 * result.surface_point[2]],
    color=COLORS["pink"],
    linestyle="--",
    linewidth=2.0,
  )
  midpoint = 0.5 * (result.link_point + result.surface_point)
  axis.text(
    1000.0 * midpoint[0] + 4.0,
    10.0,
    f"clearance = {1000.0 * result.clearance:.1f} mm",
    color=COLORS["pink"],
    fontsize=9,
    weight="bold",
  )
  axis.set_xlim(-100.0, 105.0)
  axis.set_ylim(-8.0, 58.0)
  axis.set_xlabel("x [mm]")
  axis.set_ylabel("z [mm]")
  axis.set_title("C  queryClearance(link)", loc="left")
  axis.legend(loc="upper right", fontsize=8)
  return result.clearance


def render_visual_demo(
  output_path: Path,
  *,
  mesh_path: Path = DEFAULT_BUNNY,
  source_up_axis: str = "y",
) -> dict[str, Any]:
  """Render the Module 1 visual board and return the displayed facts."""

  surface = MeshSurface.from_file(
    mesh_path,
    source_up_axis=source_up_axis,
    scale_policy=MeshScalePolicy(),
  )
  model = OracleSurfaceModel(surface, version=f"mesh-{mesh_path.stem}-v1")
  plt = get_pyplot()
  figure = plt.figure(figsize=(15.0, 8.3))
  grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0), hspace=0.34, wspace=0.19)
  mesh_axis = figure.add_subplot(grid[:, 0], projection="3d")
  query_axis = figure.add_subplot(grid[0, 1])
  clearance_axis = figure.add_subplot(grid[1, 1])
  candidate_count = _add_bunny_panel(mesh_axis, surface, model)
  _add_surface_query_panel(query_axis)
  clearance = _add_clearance_panel(clearance_axis)
  extents = surface.extents
  figure.suptitle(
    "Module 1 — Oracle SurfaceModel: what geometry the planner can ask",
    fontsize=16,
    weight="bold",
    x=0.04,
    ha="left",
  )
  figure.text(
    0.04,
    0.02,
    (
      f"Bunny extent: {extents[0]:.3f} × {extents[1]:.3f} × {extents[2]:.3f} m  |  "
      f"candidates: {candidate_count}  |  Oracle uncertainty U(x) = 0  |  version is immutable"
    ),
    fontsize=10,
    color=COLORS["gray"],
  )
  save_figure(figure, output_path)
  plt.close(figure)
  return {
    "module": "M01",
    "passed": Path(output_path).is_file() and candidate_count == 12 and clearance > 0.0,
    "output": str(output_path),
    "mesh_extent_m": extents.tolist(),
    "candidate_count": candidate_count,
    "clearance_m": clearance,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("Module/generated/visual_demo/module_1_surface_model.png"),
  )
  parser.add_argument("--mesh", type=Path, default=DEFAULT_BUNNY)
  parser.add_argument("--source-up-axis", choices=("x", "y", "z"), default="y")
  args = parser.parse_args()
  result = render_visual_demo(
    args.output,
    mesh_path=args.mesh,
    source_up_axis=args.source_up_axis,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
