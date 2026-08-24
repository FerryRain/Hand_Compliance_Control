"""Config-driven MuJoCo object library for fingertip-compliance datasets.

The catalog deliberately separates reusable geometry from collection logic.
Each object YAML is resolved in this order::

    base.yaml <- families/<family>.yaml <- objects/<object_id>.yaml

Collection and motion code can therefore consume the same resolved metadata
without duplicating MuJoCo geometry definitions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
OBJECT_CONFIG_DIR = CONFIG_ROOT / "objects"
FAMILY_CONFIG_DIR = CONFIG_ROOT / "families"

_GEOM_TYPES = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
}
_SIZE_LENGTHS = {
    "sphere": 1,
    "capsule": 2,
    "cylinder": 2,
    "ellipsoid": 3,
    "box": 3,
    "rounded_box": 3,
    # A mesh geom's size is an isotropic per-axis scale factor of the source
    # OBJ, not a physical half-extent.
    "mesh": 3,
}
_SUPPORTED_GEOM_TYPES = {*_GEOM_TYPES, "rounded_box", "mesh"}
# Files outside the repository root (e.g. /tmp YCB exports) can be addressed
# with an absolute path in the YAML; everything else is resolved relative to
# the repository root so configs stay machine independent.
_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
OBJECT_CONFIG_DIR = CONFIG_ROOT / "objects"
FAMILY_CONFIG_DIR = CONFIG_ROOT / "families"


def _resolve_asset_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = _REPO_ROOT / candidate
    return resolved.resolve()


@dataclass(frozen=True)
class ObjectGeomConfig:
    geom_type: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    rgba: tuple[float, float, float, float]
    mass_fraction: float
    rounding_radius: float = 0.0
    mesh_subdivisions: int = 0
    # mesh-only: source visual OBJ and convex collision-part directory.  The
    # mesh is translated so ``pos`` lands on the origin and scaled by ``size``.
    file: str = ""
    collision_dir: str = ""


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    display_name: str
    family: str
    body_name: str
    mocap: bool
    total_mass_kg: float
    initial_pos: tuple[float, float, float]
    initial_rot: tuple[float, float, float, float]
    contact: dict[str, Any]
    collection: dict[str, Any]
    motion: dict[str, Any]
    geoms: tuple[ObjectGeomConfig, ...]
    resolved: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def list_object_ids() -> tuple[str, ...]:
    """Return every concrete object configuration in stable name order."""

    return tuple(
        path.stem
        for path in sorted(OBJECT_CONFIG_DIR.glob("*.yaml"))
        if path.name != "base.yaml"
    )


def _float_tuple(
    value: Any,
    length: int,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(float(item) for item in array)


def load_object_config(object_id: str) -> ObjectConfig:
    """Load, merge and validate one object configuration."""

    object_path = OBJECT_CONFIG_DIR / f"{object_id}.yaml"
    base = _read_yaml(OBJECT_CONFIG_DIR / "base.yaml")
    concrete = _read_yaml(object_path)
    configured_id = str(concrete.get("id", ""))
    if configured_id != object_id:
        raise ValueError(
            f"Object id mismatch: requested {object_id!r}, "
            f"but {object_path} declares {configured_id!r}"
        )
    family = str(concrete.get("family", ""))
    if not family:
        raise ValueError(f"Object {object_id!r} does not declare a family")
    resolved = _deep_merge(base, _read_yaml(FAMILY_CONFIG_DIR / f"{family}.yaml"))
    resolved = _deep_merge(resolved, concrete)

    body = resolved.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"Object {object_id!r} has no body mapping")
    total_mass = float(body.get("total_mass_kg", 1.0))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError(f"Object {object_id!r} must have positive total mass")
    initial_pos = _float_tuple(
        body.get("initial_pos", (0.7007, 0.0003, 0.8377)),
        3,
        "body.initial_pos",
    )
    initial_rot_array = np.asarray(
        _float_tuple(
            body.get("initial_rot", (1.0, 0.0, 0.0, 0.0)),
            4,
            "body.initial_rot",
        )
    )
    initial_rot_norm = float(np.linalg.norm(initial_rot_array))
    if initial_rot_norm < 1.0e-9:
        raise ValueError(f"Object {object_id!r} body.initial_rot cannot be zero")
    initial_rot = tuple(float(item) for item in initial_rot_array / initial_rot_norm)
    default_rgba = _float_tuple(
        resolved.get("rgba", (0.5, 0.5, 0.5, 1.0)), 4, "rgba"
    )

    raw_geoms = resolved.get("geoms")
    if not isinstance(raw_geoms, list) or not raw_geoms:
        raise ValueError(f"Object {object_id!r} must contain at least one geom")
    explicit_fractions = [geom.get("mass_fraction") for geom in raw_geoms]
    if any(value is not None for value in explicit_fractions):
        if any(value is None for value in explicit_fractions):
            raise ValueError(
                f"Object {object_id!r} must set mass_fraction for every geom or none"
            )
        mass_fractions = np.asarray(explicit_fractions, dtype=np.float64)
    else:
        mass_fractions = np.full(len(raw_geoms), 1.0 / len(raw_geoms))
    if np.any(mass_fractions <= 0.0) or not np.isclose(
        mass_fractions.sum(), 1.0, atol=1.0e-6
    ):
        raise ValueError(
            f"Object {object_id!r} geom mass fractions must be positive and sum to 1"
        )

    geoms: list[ObjectGeomConfig] = []
    for index, (raw, mass_fraction) in enumerate(
        zip(raw_geoms, mass_fractions, strict=True)
    ):
        if not isinstance(raw, dict):
            raise ValueError(f"geoms[{index}] must be a mapping")
        geom_type = str(raw.get("type", ""))
        if geom_type not in _SUPPORTED_GEOM_TYPES:
            raise ValueError(
                f"Unsupported geom type {geom_type!r}; "
                f"available={sorted(_SUPPORTED_GEOM_TYPES)}"
            )
        expected_size = _SIZE_LENGTHS[geom_type]
        compact_size = _float_tuple(
            raw.get("size"), expected_size, f"geoms[{index}].size"
        )
        if min(compact_size) <= 0.0:
            raise ValueError(f"geoms[{index}].size must be positive")
        padded_size = (*compact_size, *(0.0,) * (3 - expected_size))
        quat_array = np.asarray(
            _float_tuple(
                raw.get("quat", (1.0, 0.0, 0.0, 0.0)),
                4,
                f"geoms[{index}].quat",
            )
        )
        quat_norm = float(np.linalg.norm(quat_array))
        if quat_norm < 1.0e-9:
            raise ValueError(f"geoms[{index}].quat cannot be zero")
        quat = tuple(float(item) for item in quat_array / quat_norm)
        rounding_radius = float(raw.get("rounding_radius", 0.0))
        mesh_subdivisions = int(raw.get("mesh_subdivisions", 0))
        if geom_type == "rounded_box":
            if not 0.0 < rounding_radius < min(compact_size):
                raise ValueError(
                    f"geoms[{index}].rounding_radius must lie between zero "
                    "and the smallest box half-size"
                )
            if not 1 <= mesh_subdivisions <= 4:
                raise ValueError(
                    f"geoms[{index}].mesh_subdivisions must be in [1, 4]"
                )
        elif geom_type == "mesh":
            if rounding_radius != 0.0 or mesh_subdivisions != 0:
                raise ValueError(
                    f"geoms[{index}] rounding options require type=rounded_box"
                )
            file_value = str(raw.get("file", ""))
            collision_value = str(raw.get("collision_dir", ""))
            if not file_value:
                raise ValueError(f"geoms[{index}] type=mesh requires 'file'")
            file_path = _resolve_asset_path(file_value)
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"geoms[{index}] visual mesh not found: {file_path}"
                )
            if collision_value:
                collision_path = _resolve_asset_path(collision_value)
                if not collision_path.is_dir():
                    raise NotADirectoryError(
                        f"geoms[{index}] collision_dir not found: {collision_path}"
                    )
            else:
                collision_path = None
        else:
            if rounding_radius != 0.0 or mesh_subdivisions != 0:
                raise ValueError(
                    f"geoms[{index}] rounding options require type=rounded_box"
                )
            file_path = None
            collision_path = None
        geoms.append(
            ObjectGeomConfig(
                geom_type=geom_type,
                size=tuple(float(item) for item in padded_size),
                pos=_float_tuple(
                    raw.get("pos", (0.0, 0.0, 0.0)),
                    3,
                    f"geoms[{index}].pos",
                ),
                quat=quat,
                rgba=_float_tuple(
                    raw.get("rgba", default_rgba),
                    4,
                    f"geoms[{index}].rgba",
                ),
                mass_fraction=float(mass_fraction),
                rounding_radius=rounding_radius,
                mesh_subdivisions=mesh_subdivisions,
                file=str(file_path) if file_path is not None else "",
                collision_dir=(
                    str(collision_path) if collision_path is not None else ""
                ),
            )
        )

    contact = resolved.get("contact", {})
    collection = resolved.get("collection", {})
    motion = resolved.get("motion", {})
    for name, mapping in (
        ("contact", contact),
        ("collection", collection),
        ("motion", motion),
    ):
        if not isinstance(mapping, dict):
            raise ValueError(f"Object {object_id!r} {name} must be a mapping")

    return ObjectConfig(
        object_id=object_id,
        display_name=str(resolved.get("display_name", object_id)),
        family=family,
        body_name=str(body.get("name", "target_ball")),
        mocap=bool(body.get("mocap", True)),
        total_mass_kg=total_mass,
        initial_pos=initial_pos,
        initial_rot=initial_rot,
        contact=deepcopy(contact),
        collection=deepcopy(collection),
        motion=deepcopy(motion),
        geoms=tuple(geoms),
        resolved=deepcopy(resolved),
    )


class MeshNormalOracle:
    """Smooth contact normals from the high-resolution source OBJ vertices.

    MuJoCo contacts collide against the convex-decomposed collision parts,
    whose seams introduce artificial normal discontinuities (measured up to
    ~90 deg across seams).  This oracle ignores the collision mesh for
    normals: it re-samples the contact point on the visual mesh point cloud
    and fits a local least-squares plane (PCA) inside a fixed radius, which
    both removes seam jumps and low-pass filters the discrete face normals.
    """

    def __init__(
        self,
        geom_configs: list[ObjectGeomConfig],
        scale: float = 1.0,
        radius_m: float = 0.01,
        min_neighbours: int = 16,
    ) -> None:
        meshes = [
            _load_scaled_mesh(Path(geom.file), geom, scale)
            for geom in geom_configs
            if geom.geom_type == "mesh"
        ]
        if not meshes:
            raise ValueError("MeshNormalOracle requires at least one mesh geom")
        self.vertices = np.vstack([mesh.vertices for mesh in meshes])
        self.tree = cKDTree(self.vertices)
        # Outward sign reference: the source mesh's own (consistently wound)
        # face normals.  A center-based rule fails on horizontal bands whose
        # outward normal is perpendicular to the point-center ray (cap lip,
        # neck taper), which made the PCA sign flip ~180 deg between probes.
        self.face_centroids = np.vstack(
            [np.asarray(mesh.triangles_center) for mesh in meshes]
        )
        self.face_normals = np.vstack(
            [np.asarray(mesh.face_normals) for mesh in meshes]
        )
        self.face_tree = cKDTree(self.face_centroids)
        self.radius_m = float(radius_m)
        # Coarse patches of the source mesh (e.g. the label back of the
        # mustard bottle) can leave fewer than a handful of vertices inside
        # the ball; a 3-point fit is degenerate and its normal arbitrary.
        self.min_neighbours = int(min_neighbours)
        # Reference direction for outward signs: vertices already live in a
        # frame where the object sits roughly at the origin.
        self.center = self.vertices.mean(axis=0)

    @classmethod
    def from_config(
        cls,
        config: ObjectConfig,
        scale: float = 1.0,
        radius_m: float = 0.01,
    ) -> "MeshNormalOracle | None":
        mesh_geoms = [geom for geom in config.geoms if geom.geom_type == "mesh"]
        if not mesh_geoms:
            return None
        return cls(mesh_geoms, scale=scale, radius_m=radius_m)

    def query_object_frame(self, points: np.ndarray) -> np.ndarray:
        """Estimate outward normals for points given in the object frame."""

        points = np.asarray(points, dtype=np.float64)
        single = points.ndim == 1
        batch = points[None, :] if single else points
        normals = np.zeros_like(batch)
        for index, point in enumerate(batch):
            neighbours = self.tree.query_ball_point(point, self.radius_m)
            if len(neighbours) < self.min_neighbours:
                # Sparse source-mesh patch: grow the query to a fixed k
                # neighbours so the plane fit stays well-posed.
                _, nearest = self.tree.query(point, k=self.min_neighbours)
                neighbours = nearest.tolist()
            local = self.vertices[neighbours]
            if len(neighbours) >= 3:
                # Distance-weighted least-squares plane fit: near vertices
                # dominate so ridges and adjacent walls cannot tilt the fit,
                # and the fitted normal stays a genuine local surface normal.
                deltas = local - point
                dist2 = np.einsum("ij,ij->i", deltas, deltas)
                # k-neighbour fallback grows the neighbourhood beyond the
                # radius; widen sigma with it so weights never underflow.
                sigma = max(self.radius_m, 0.5 * np.sqrt(dist2.max()))
                weights = np.exp(-0.5 * dist2 / sigma**2)
                if not np.isfinite(weights).all() or weights.sum() <= 0.0:
                    weights = np.ones_like(dist2)
                weights /= weights.sum()
                weighted_mean = weights @ local
                centered = local - weighted_mean
                covariance = (centered * weights[:, None]).T @ centered
                _, eigenvectors = np.linalg.eigh(covariance)
                normal = eigenvectors[:, 0]
            else:
                normal = local[0] - point
            if np.linalg.norm(normal) < 1.0e-12:
                normal = np.array([0.0, 0.0, 1.0])
            normal = normal / np.linalg.norm(normal)
            # Outward sign: majority vote of the nearest source-mesh face
            # normals (the source OBJ is consistently wound, unlike the
            # collision parts).  A point-center rule breaks on horizontal
            # bands whose outward normal is perpendicular to the radial ray.
            _, nearest_faces = self.face_tree.query(point, k=5)
            dots = self.face_normals[nearest_faces] @ normal
            if np.max(np.abs(dots)) < 0.35:
                # PCA fit degenerate here (normal nearly tangential to every
                # local face -- sparse neck patches, double-walled regions):
                # fall back to the nearest face normal, which is already
                # outward by the mesh winding.
                normal = self.face_normals[nearest_faces[0]].copy()
            else:
                if np.sum(dots > 0.0) < 2.5:
                    normal = -normal
            normals[index] = normal
        return normals[0] if single else normals

    def query_world(
        self,
        points_world: np.ndarray,
        object_pos_world: np.ndarray,
        object_quat_world: np.ndarray,
    ) -> np.ndarray:
        """Estimate outward world-frame normals for world-frame points.

        ``object_quat_world`` is wxyz.  The object frame used here matches
        ``_load_scaled_mesh``: config ``pos`` on the origin, scaled by
        ``size * scale``.
        """

        quats = np.asarray(object_quat_world, dtype=np.float64).reshape(-1, 4)
        # scipy uses xyzw; our wxyz goes in as [x, y, z, w].
        rotmats = (
            Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            if quats.shape[0]
            else np.empty((0, 3, 3))
        )
        points_object = np.einsum(
            "ij,ikj->ik",
            np.asarray(points_world) - np.asarray(object_pos_world),
            rotmats,
        )
        normals_object = self.query_object_frame(points_object)
        return np.einsum("ij,ijk->ik", normals_object, rotmats)


def _load_scaled_mesh(
    path: Path,
    config: ObjectGeomConfig,
    scale: float,
) -> trimesh.Trimesh:
    """Load an OBJ, move ``config.pos`` to the origin and apply size*scale."""

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"empty trimesh scene: {path}")
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(loaded).__name__}: {path}")
    if not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"mesh has no triangle surface: {path}")
    mesh = loaded.copy()
    total_scale = np.asarray(config.size, dtype=np.float64) * scale
    mesh.vertices = (mesh.vertices - np.asarray(config.pos)) * total_scale
    return mesh


def _add_mesh_geom(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    geom_name: str,
    mesh: trimesh.Trimesh,
    rgba: tuple[float, float, float, float],
    mass: float,
    *,
    contact: dict[str, Any],
    solref: tuple[float, float],
    solimp: tuple[float, float, float, float, float],
    friction: tuple[float, float, float],
    contype: int = 1,
    conaffinity: int = 1,
) -> None:
    mesh_name = f"{geom_name}_mesh"
    spec.add_mesh(
        name=mesh_name,
        uservert=np.asarray(mesh.vertices, dtype=np.float64).ravel(),
        userface=np.asarray(mesh.faces, dtype=np.int32).ravel(),
        smoothnormal=1,
    )
    geom = body.add_geom(
        name=geom_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        rgba=rgba,
        mass=mass,
        contype=contype,
        conaffinity=conaffinity,
    )
    geom.solref[:] = solref
    geom.solimp[:] = solimp
    geom.friction[:] = friction
    geom.margin = float(contact.get("margin_m", 0.0))
    geom.gap = float(contact.get("gap_m", 0.0))
    geom.priority = int(contact.get("priority", 10))
    geom.condim = int(contact.get("condim", 3))


def add_object_body(
    spec: mujoco.MjSpec,
    config: ObjectConfig,
    *,
    body_name: str | None = None,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mocap: bool | None = None,
    scale: float = 1.0,
) -> mujoco.MjsBody:
    """Add one configured object to an existing ``MjSpec``.

    ``scale`` multiplies mesh geometry only (``collection.size_scale_range``);
    primitive sizes are physical and stay untouched.
    """

    name = body_name or config.body_name
    body = spec.worldbody.add_body(
        name=name,
        pos=pos,
        quat=quat,
        mocap=config.mocap if mocap is None else mocap,
    )
    contact = config.contact
    solref = _float_tuple(contact["solref"], 2, "contact.solref")
    solimp = _float_tuple(contact["solimp"], 5, "contact.solimp")
    friction = _float_tuple(contact["friction"], 3, "contact.friction")
    for index, geom_config in enumerate(config.geoms):
        geom_name = f"{name}_{index}_{geom_config.geom_type}"
        geom_kwargs: dict[str, Any] = {
            "name": geom_name,
            "pos": geom_config.pos,
            "quat": geom_config.quat,
            "rgba": geom_config.rgba,
            "mass": config.total_mass_kg * geom_config.mass_fraction,
        }
        if geom_config.geom_type == "rounded_box":
            half_size = np.asarray(geom_config.size, dtype=np.float64)
            core = half_size - geom_config.rounding_radius
            sphere = trimesh.creation.icosphere(
                subdivisions=geom_config.mesh_subdivisions,
                radius=1.0,
            )
            directions = np.asarray(sphere.vertices, dtype=np.float64)
            support_points = (
                np.sign(directions) * core
                + geom_config.rounding_radius * directions
            )
            # Recompute the convex hull after the support mapping.  This
            # produces one watertight collision mesh without overlapping
            # primitive seams at the rounded edges.
            rounded = trimesh.convex.convex_hull(support_points)
            rounded.fix_normals()
            _add_mesh_geom(
                spec, body, geom_name, rounded, geom_config.rgba,
                config.total_mass_kg * geom_config.mass_fraction,
                contact=contact, solref=solref, solimp=solimp, friction=friction,
            )
        elif geom_config.geom_type == "mesh":
            visual = _load_scaled_mesh(
                Path(geom_config.file), geom_config, scale
            )
            # Visual-only copy: never collides, carries no mass.
            _add_mesh_geom(
                spec, body, f"{geom_name}_visual", visual, geom_config.rgba,
                0.0,
                contact=contact, solref=solref, solimp=solimp, friction=friction,
                contype=0, conaffinity=0,
            )
            part_paths = (
                sorted(
                    Path(geom_config.collision_dir).glob(
                        "collision_part_*.obj"
                    )
                )
                if geom_config.collision_dir
                else []
            )
            if not part_paths:
                raise ValueError(
                    f"mesh geom {geom_name!r} has no collision parts in "
                    f"{geom_config.collision_dir or '(none)'}"
                )
            mass_per_part = (
                config.total_mass_kg * geom_config.mass_fraction
                / len(part_paths)
            )
            for part_index, part_path in enumerate(part_paths):
                part = _load_scaled_mesh(part_path, geom_config, scale)
                _add_mesh_geom(
                    spec, body,
                    f"{geom_name}_collision_{part_index}", part,
                    geom_config.rgba, mass_per_part,
                    contact=contact, solref=solref, solimp=solimp,
                    friction=friction,
                )
        else:
            geom_kwargs.update(
                type=_GEOM_TYPES[geom_config.geom_type],
                size=geom_config.size,
            )
            geom = body.add_geom(**geom_kwargs)
            geom.solref[:] = solref
            geom.solimp[:] = solimp
            geom.friction[:] = friction
            geom.margin = float(contact.get("margin_m", 0.0))
            geom.gap = float(contact.get("gap_m", 0.0))
            geom.priority = int(contact.get("priority", 10))
            geom.condim = int(contact.get("condim", 3))
    return body


def build_object_spec(
    object_id: str,
    *,
    body_name: str | None = None,
    mocap: bool | None = None,
) -> mujoco.MjSpec:
    """Build the single-body spec expected by the MJLab target entity."""

    spec = mujoco.MjSpec()
    config = load_object_config(object_id)
    add_object_body(spec, config, body_name=body_name, mocap=mocap)
    return spec


def object_local_aabb(
    config: ObjectConfig,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative local-frame AABB for gallery and motion limits.

    ``scale`` must match the scale passed to :func:`add_object_body`; mesh
    extents are measured from the scaled source OBJ.
    """

    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for geom in config.geoms:
        if geom.geom_type == "sphere":
            extent = np.full(3, geom.size[0])
        elif geom.geom_type == "capsule":
            extent = np.asarray(
                (geom.size[0], geom.size[0], geom.size[0] + geom.size[1])
            )
        elif geom.geom_type == "cylinder":
            extent = np.asarray((geom.size[0], geom.size[0], geom.size[1]))
        elif geom.geom_type in ("ellipsoid", "box", "rounded_box"):
            extent = np.asarray(geom.size)
        elif geom.geom_type == "mesh":
            mesh = _load_scaled_mesh(Path(geom.file), geom, scale)
            if not np.all(np.isfinite(mesh.vertices)):
                raise ValueError(f"mesh {geom.file!r} contains non-finite vertices")
            extent = 0.5 * (mesh.bounds[1] - mesh.bounds[0])
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, np.asarray(geom.quat))
            rotated_extent = np.abs(rotation.reshape(3, 3)) @ extent
            center = 0.5 * (mesh.bounds[1] + mesh.bounds[0])
            lower = np.minimum(lower, center - rotated_extent)
            upper = np.maximum(upper, center + rotated_extent)
            continue
        else:
            raise AssertionError(f"Unhandled geom type {geom.geom_type!r}")
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, np.asarray(geom.quat))
        rotated_extent = np.abs(rotation.reshape(3, 3)) @ extent
        center = np.asarray(geom.pos)
        lower = np.minimum(lower, center - rotated_extent)
        upper = np.maximum(upper, center + rotated_extent)
    return lower, upper


def get_motion_config(config: ObjectConfig) -> dict[str, Any]:
    """Merge family-level ``motion_defaults`` with object-level ``motion``.

    Returns a dict with keys ``translation``, ``rotation`` and ``orbit``,
    each containing the fully-resolved parameters for that motion axis.
    """
    defaults = config.resolved.get("motion_defaults", {})
    specific = config.resolved.get("motion", {})
    result: dict[str, Any] = {}
    for axis in ("translation", "rotation", "orbit"):
        base = defaults.get(axis, {})
        override = specific.get(axis, {})
        if not isinstance(base, dict) or not isinstance(override, dict):
            raise ValueError(
                f"Object {config.object_id!r} motion.{axis} must be a mapping"
            )
        result[axis] = _deep_merge(base, override)
    return result
