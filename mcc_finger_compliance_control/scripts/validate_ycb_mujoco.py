"""Compile YCB meshes in MuJoCo and verify mesh-sphere collision contacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import trimesh


DEFAULT_ROOT = Path("assets_external/ycb")


@dataclass
class ValidationResult:
    object_id: str
    mesh_file: str
    source_variant: str
    vertices: int = 0
    faces: int = 0
    extent_x_m: float = 0.0
    extent_y_m: float = 0.0
    extent_z_m: float = 0.0
    mujoco_loaded: bool = False
    contact_found: bool = False
    contact_count: int = 0
    min_contact_distance_m: float = float("nan")
    contact_force_N: float = float("nan")
    probe_radius_m: float = float("nan")
    size_warning: str = ""
    error: str = ""


def _load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force=None, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("empty trimesh scene")
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(loaded).__name__}")
    if not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError("mesh has no triangle surface")
    if not np.all(np.isfinite(loaded.vertices)):
        raise ValueError("mesh contains non-finite vertices")
    return loaded


def _probe_from_convex_hull(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Place a sphere halfway through a broad convex-hull face.

    MuJoCo's built-in mesh collision uses the mesh convex hull. Positioning the
    probe from that same hull tests the collision geometry MuJoCo actually
    uses, independent of visual concavities in the source OBJ.
    """

    hull = mesh.convex_hull
    if not len(hull.faces):
        raise ValueError("cannot construct a convex collision hull")
    face_index = int(np.argmax(hull.area_faces))
    face_center = hull.vertices[hull.faces[face_index]].mean(axis=0)
    normal = np.array(hull.face_normals[face_index], dtype=np.float64, copy=True)
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
    positive_extents = mesh.extents[mesh.extents > 1.0e-6]
    if not len(positive_extents):
        raise ValueError("mesh has degenerate extents")
    radius = float(np.clip(np.min(positive_extents) * 0.05, 0.001, 0.01))
    # Half of the probe radius lies inside the collision hull.
    return face_center + normal * (0.5 * radius), radius


def _mujoco_collision_test(
    mesh_path: Path, probe_position: np.ndarray, probe_radius: float
) -> tuple[int, float, float]:
    mesh_file = escape(str(mesh_path.resolve()))
    probe_pos = " ".join(f"{value:.12g}" for value in probe_position)
    xml = f"""
<mujoco model="ycb_collision_validation">
  <option gravity="0 0 0" timestep="0.001"/>
  <asset>
    <mesh name="target_mesh" file="{mesh_file}"/>
  </asset>
  <worldbody>
    <geom name="target" type="mesh" mesh="target_mesh"
          contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
    <body name="probe" pos="{probe_pos}">
      <freejoint/>
      <geom name="probe" type="sphere" size="{probe_radius:.12g}"
            density="1000" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    count = int(data.ncon)
    minimum = min((float(data.contact[i].dist) for i in range(count)), default=float("nan"))
    contact_force = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(count):
        mujoco.mj_contactForce(model, data, index, wrench)
        contact_force = max(contact_force, float(np.linalg.norm(wrench[:3])))
    return count, minimum, contact_force


def validate_one(
    root: Path, record: dict[str, object]
) -> ValidationResult:
    object_id = str(record["object_id"])
    mesh_files = list(record.get("mesh_files", []))
    result = ValidationResult(
        object_id=object_id,
        mesh_file=str(mesh_files[0]) if mesh_files else "",
        source_variant=str(record.get("source_variant", "unknown")),
    )
    if not mesh_files:
        result.error = "manifest contains no usable triangle mesh"
        return result
    path = root / str(mesh_files[0])
    try:
        mesh = _load_triangle_mesh(path)
        result.vertices = int(len(mesh.vertices))
        result.faces = int(len(mesh.faces))
        extents = np.asarray(mesh.extents, dtype=np.float64)
        result.extent_x_m, result.extent_y_m, result.extent_z_m = map(float, extents)
        maximum = float(np.max(extents))
        minimum = float(np.min(extents))
        if maximum > 0.35:
            result.size_warning = "larger_than_0.35m_review_for_in_hand_use"
        elif maximum < 0.005:
            result.size_warning = "smaller_than_5mm"
        elif minimum < 0.0005:
            result.size_warning = "nearly_planar_or_thin"
        position, radius = _probe_from_convex_hull(mesh)
        result.probe_radius_m = radius
        count, distance, contact_force = _mujoco_collision_test(path, position, radius)
        result.mujoco_loaded = True
        result.contact_count = count
        result.min_contact_distance_m = distance
        result.contact_force_N = contact_force
        result.contact_found = (
            count > 0
            and np.isfinite(distance)
            and distance <= 0.0
            and contact_force > 0.0
        )
    except Exception as error:  # report all asset-specific failures
        result.error = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--all-meshes",
        action="store_true",
        help="Validate every OBJ variant instead of the manifest's preferred mesh.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    if args.all_meshes:
        expanded: list[dict[str, object]] = []
        for record in records:
            object_dir = root / "models" / str(record["object_id"])
            mesh_files = sorted(object_dir.rglob("*.obj")) if object_dir.is_dir() else []
            if not mesh_files:
                expanded.append(record)
                continue
            for mesh_path in mesh_files:
                clone = dict(record)
                clone["mesh_files"] = [str(mesh_path.relative_to(root))]
                expanded.append(clone)
        records = expanded
    if args.limit:
        records = records[: args.limit]
    results: list[ValidationResult] = []
    for index, record in enumerate(records, start=1):
        result = validate_one(root, record)
        results.append(result)
        state = "CONTACT" if result.contact_found else (
            "LOAD_ONLY" if result.mujoco_loaded else "FAILED"
        )
        print(
            f"[YCB-MJ] {index:03d}/{len(records):03d} "
            f"{result.object_id:<31} {state:<9} "
            f"faces={result.faces:<7d} "
            f"extent=({result.extent_x_m:.3f},"
            f"{result.extent_y_m:.3f},{result.extent_z_m:.3f})m"
        )

    report = args.report or root / "mujoco_validation.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    loaded = sum(result.mujoco_loaded for result in results)
    contacted = sum(result.contact_found for result in results)
    warnings = sum(bool(result.size_warning) for result in results)
    print(
        f"[RESULT] total={len(results)} loaded={loaded} contacts={contacted} "
        f"size_warnings={warnings} report={report}"
    )
    failures = [
        result for result in results if not result.mujoco_loaded or (
            result.mujoco_loaded and not result.contact_found
        )
    ]
    if failures:
        print("[RESULT] failures:")
        for result in failures:
            print(f"  - {result.object_id}: {result.error or 'no contact'}")


if __name__ == "__main__":
    main()
