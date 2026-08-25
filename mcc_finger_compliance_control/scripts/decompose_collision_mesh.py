"""Build and validate convex collision parts for a visual triangle mesh.

The source mesh is never modified.  The output directory contains convex OBJ
parts, a machine-readable manifest, and an MJCF preview which renders the
source mesh while using only the convex parts for collision.

V-HACD is the conservative default because it has a hard hull-count limit and
is stable on the non-watertight YCB/LeapHand meshes used in this repository.
CoACD is also supported when its optional Python package is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import trimesh


@dataclass(frozen=True)
class SurfaceError:
    mean_mm: float
    median_mm: float
    p95_mm: float
    p99_mm: float
    max_mm: float


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"{path} contains no geometry")
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected triangle mesh, got {type(loaded).__name__}")
    if not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"{path} has no triangle surface")
    if not np.all(np.isfinite(loaded.vertices)):
        raise ValueError(f"{path} contains non-finite vertices")
    loaded.remove_unreferenced_vertices()
    loaded.fix_normals()
    return loaded


def _vhacd(mesh: trimesh.Trimesh, args: argparse.Namespace) -> list[trimesh.Trimesh]:
    result = mesh.convex_decomposition(
        maxConvexHulls=args.max_hulls,
        resolution=args.vhacd_resolution,
        minimumVolumePercentErrorAllowed=args.vhacd_volume_error_percent,
        maxRecursionDepth=args.vhacd_recursion_depth,
        shrinkWrap=True,
        fillMode="flood",
        maxNumVerticesPerCH=args.max_vertices,
        asyncACD=False,
        minEdgeLength=1,
        findBestPlane=True,
    )
    return list(result) if isinstance(result, list) else [result]


def _coacd(mesh: trimesh.Trimesh, args: argparse.Namespace) -> list[trimesh.Trimesh]:
    try:
        import coacd
    except ImportError as error:
        raise RuntimeError(
            "CoACD is not installed. Run `python -m pip install coacd`, "
            "or use `--backend vhacd`."
        ) from error

    source = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    # CoACD's max_convex_hull is implemented by its merge pass.  Version
    # 1.0.11 can terminate inside native code while aggressively merging some
    # repaired, non-watertight scans (the YCB bowl is one example).  Therefore
    # merging is opt-in and the reliable default keeps every generated part.
    raw_parts = coacd.run_coacd(
        source,
        threshold=args.coacd_threshold_m,
        real_metric=True,
        max_convex_hull=args.max_hulls,
        preprocess_mode="auto",
        preprocess_resolution=args.coacd_preprocess_resolution,
        resolution=args.coacd_resolution,
        mcts_nodes=args.coacd_mcts_nodes,
        mcts_iterations=args.coacd_mcts_iterations,
        mcts_max_depth=args.coacd_mcts_depth,
        merge=args.coacd_merge,
        decimate=True,
        max_ch_vertex=args.max_vertices,
        seed=args.seed,
    )
    return [trimesh.Trimesh(vertices, faces, process=True) for vertices, faces in raw_parts]


def _clean_parts(parts: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    cleaned: list[trimesh.Trimesh] = []
    for part in parts:
        if not isinstance(part, trimesh.Trimesh) or not len(part.faces):
            continue
        # MuJoCo will use a convex hull for every part.  Export that exact
        # surface so visual inspection and offline metrics match simulation.
        hull = part.convex_hull
        hull.remove_unreferenced_vertices()
        hull.fix_normals()
        if len(hull.vertices) >= 4 and abs(float(hull.volume)) > 1.0e-15:
            cleaned.append(hull)
    if not cleaned:
        raise RuntimeError("Convex decomposition returned no valid solid parts")
    return cleaned


def _surface_error(reference: trimesh.Trimesh, query_points: np.ndarray) -> SurfaceError:
    _, distances, _ = trimesh.proximity.closest_point(reference, query_points)
    values = np.asarray(distances, dtype=np.float64) * 1000.0
    return SurfaceError(
        mean_mm=float(np.mean(values)),
        median_mm=float(np.median(values)),
        p95_mm=float(np.percentile(values, 95)),
        p99_mm=float(np.percentile(values, 99)),
        max_mm=float(np.max(values)),
    )


def _measure_surface_fit(
    source: trimesh.Trimesh,
    parts: list[trimesh.Trimesh],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_points, _ = trimesh.sample.sample_surface(source, sample_count, seed=rng)
    collision = trimesh.util.concatenate(parts)
    collision_points, _ = trimesh.sample.sample_surface(
        collision, sample_count, seed=rng
    )
    return {
        # Measures missing/under-approximated source surface.
        "visual_to_collision": asdict(_surface_error(collision, source_points)),
        # This is deliberately conservative: convex-part cut faces and overlap
        # faces are included, so it can overstate visible outward error.
        "collision_all_faces_to_visual": asdict(
            _surface_error(source, collision_points)
        ),
        "collision_metric_note": (
            "collision_all_faces_to_visual includes internal cut/overlap faces; "
            "use visual_to_collision plus preview inspection for acceptance"
        ),
    }


def _write_preview(
    output_dir: Path,
    source: Path,
    part_paths: list[Path],
    max_vertices: int,
) -> Path:
    assets = [
        f'    <mesh name="visual" file="{escape(str(source.resolve()))}"/>'
    ]
    assets.extend(
        f'    <mesh name="collision_{index:03d}" '
        f'file="{escape(str(path.resolve()))}" '
        f'maxhullvert="{max(4, max_vertices)}"/>'
        for index, path in enumerate(part_paths)
    )
    geoms = [
        '      <geom name="visual" type="mesh" mesh="visual" '
        'contype="0" conaffinity="0" density="0" rgba="0.65 0.7 0.8 0.35"/>'
    ]
    geoms.extend(
        f'      <geom name="collision_{index:03d}" type="mesh" '
        f'mesh="collision_{index:03d}" density="0" '
        'rgba="0.95 0.25 0.1 0.16"/>'
        for index in range(len(part_paths))
    )
    xml = "\n".join(
        [
            '<mujoco model="convex_collision_preview">',
            '  <compiler angle="radian" autolimits="true"/>',
            '  <option gravity="0 0 0"/>',
            '  <asset>',
            *assets,
            '  </asset>',
            '  <worldbody>',
            '    <light pos="0 0 1" dir="0 0 -1"/>',
            '    <body name="object">',
            *geoms,
            '    </body>',
            '  </worldbody>',
            '</mujoco>',
            '',
        ]
    )
    preview = output_dir / "preview.xml"
    preview.write_text(xml, encoding="utf-8")
    # Compilation verifies all exported hulls are accepted by MuJoCo.
    model = mujoco.MjModel.from_xml_path(str(preview))
    if model.nmesh != len(part_paths) + 1 or model.ngeom != len(part_paths) + 1:
        raise RuntimeError("MuJoCo preview did not contain every exported mesh")
    return preview


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source OBJ/STL visual mesh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("vhacd", "coacd"), default="vhacd")
    parser.add_argument("--max-hulls", type=int, default=64)
    parser.add_argument("--max-vertices", type=int, default=64)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--pre-scale",
        type=float,
        default=1.0,
        help=(
            "Bake this uniform scale into the source before decomposition. "
            "The transformed mesh is exported as visual_scaled.obj."
        ),
    )
    parser.add_argument(
        "--pre-center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Subtract this source-frame point before applying --pre-scale.",
    )
    parser.add_argument(
        "--accept-p95-mm",
        type=float,
        default=None,
        help="Fail after writing diagnostics when visual-to-collision p95 exceeds this.",
    )
    parser.add_argument(
        "--accept-max-parts",
        type=int,
        default=None,
        help="Fail after writing diagnostics when the generated part count exceeds this.",
    )

    parser.add_argument("--vhacd-resolution", type=int, default=800_000)
    parser.add_argument("--vhacd-volume-error-percent", type=float, default=0.2)
    parser.add_argument("--vhacd-recursion-depth", type=int, default=12)

    parser.add_argument("--coacd-threshold-m", type=float, default=0.001)
    parser.add_argument("--coacd-preprocess-resolution", type=int, default=50)
    parser.add_argument("--coacd-resolution", type=int, default=3_000)
    parser.add_argument("--coacd-mcts-nodes", type=int, default=20)
    parser.add_argument("--coacd-mcts-iterations", type=int, default=100)
    parser.add_argument("--coacd-mcts-depth", type=int, default=3)
    parser.add_argument(
        "--coacd-merge",
        action="store_true",
        help="Enable CoACD merge/max-hull pass; less robust on repaired scans.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source = _load_mesh(source_path)
    if args.pre_scale <= 0.0:
        raise ValueError("--pre-scale must be positive")
    pre_center = (
        np.zeros(3, dtype=np.float64)
        if args.pre_center is None
        else np.asarray(args.pre_center, dtype=np.float64)
    )
    transformed = args.pre_center is not None or args.pre_scale != 1.0
    if transformed:
        # Bake the final physical dimensions into the mesh *before* V-HACD.
        # This makes voxelization/error metrics correspond to the dimensions
        # MuJoCo will actually simulate and prevents accidental double scale.
        source.vertices = (
            np.asarray(source.vertices, dtype=np.float64) - pre_center
        ) * float(args.pre_scale)
        source.remove_unreferenced_vertices()
        source.fix_normals()
        visual_path = output_dir / "visual_scaled.obj"
        source.export(visual_path)
        # Reload the serialized OBJ before decomposition.  Besides making the
        # documented PLY -> scaled OBJ -> V-HACD order literal, this ensures
        # V-HACD sees exactly the same vertex/face asset that MuJoCo renders,
        # including any indexing changes introduced by OBJ serialization.
        source = _load_mesh(visual_path)
    else:
        visual_path = source_path
    print(
        f"[INPUT] {source_path} vertices={len(source.vertices)} "
        f"faces={len(source.faces)} watertight={source.is_watertight} "
        f"extent_m={np.asarray(source.extents).round(6).tolist()}"
    )
    started = time.perf_counter()
    raw_parts = _vhacd(source, args) if args.backend == "vhacd" else _coacd(source, args)
    parts = _clean_parts(raw_parts)
    elapsed = time.perf_counter() - started

    part_paths: list[Path] = []
    for index, part in enumerate(parts):
        path = output_dir / f"collision_part_{index:03d}.obj"
        part.export(path)
        part_paths.append(path)

    fit = _measure_surface_fit(source, parts, args.samples, args.seed)
    preview = _write_preview(output_dir, visual_path, part_paths, args.max_vertices)
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in {"input", "output_dir", "overwrite"}
    }
    manifest = {
        "schema_version": 1,
        "source_mesh": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "source_watertight": bool(source.is_watertight),
        "source_extent_m": np.asarray(source.extents, dtype=float).tolist(),
        "processed_visual_mesh": visual_path.name if transformed else str(source_path),
        "pre_center": pre_center.tolist(),
        "pre_scale": float(args.pre_scale),
        "backend": args.backend,
        "parameters": parameters,
        "elapsed_s": elapsed,
        "part_count": len(parts),
        "part_vertex_counts": [int(len(part.vertices)) for part in parts],
        "part_face_counts": [int(len(part.faces)) for part in parts],
        "collision_parts": [path.name for path in part_paths],
        "surface_fit": fit,
        "preview_mjcf": preview.name,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    visual_fit = fit["visual_to_collision"]
    if len(parts) > args.max_hulls:
        print(
            f"[WARNING] backend={args.backend} generated {len(parts)} parts, "
            f"above requested max_hulls={args.max_hulls}. This is expected for "
            "CoACD without --coacd-merge; reject or rerun with a coarser threshold."
        )
    print(
        f"[RESULT] backend={args.backend} parts={len(parts)} "
        f"elapsed={elapsed:.2f}s visual->collision "
        f"p95={visual_fit['p95_mm']:.3f}mm "
        f"p99={visual_fit['p99_mm']:.3f}mm "
        f"max={visual_fit['max_mm']:.3f}mm"
    )
    print(f"[OUTPUT] manifest={manifest_path}")
    print(f"[OUTPUT] preview={preview}")
    failures: list[str] = []
    if (
        args.accept_p95_mm is not None
        and visual_fit["p95_mm"] > args.accept_p95_mm
    ):
        failures.append(
            f"p95={visual_fit['p95_mm']:.3f}mm > "
            f"accept_p95={args.accept_p95_mm:.3f}mm"
        )
    if (
        args.accept_max_parts is not None
        and len(parts) > args.accept_max_parts
    ):
        failures.append(
            f"parts={len(parts)} > accept_max_parts={args.accept_max_parts}"
        )
    if failures:
        raise SystemExit("[REJECTED] " + "; ".join(failures))


if __name__ == "__main__":
    main()
