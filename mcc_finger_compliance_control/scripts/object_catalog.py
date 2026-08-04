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
}
_SUPPORTED_GEOM_TYPES = {*_GEOM_TYPES, "rounded_box"}


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


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    display_name: str
    family: str
    body_name: str
    mocap: bool
    total_mass_kg: float
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
        elif rounding_radius != 0.0 or mesh_subdivisions != 0:
            raise ValueError(
                f"geoms[{index}] rounding options require type=rounded_box"
            )
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
        contact=deepcopy(contact),
        collection=deepcopy(collection),
        motion=deepcopy(motion),
        geoms=tuple(geoms),
        resolved=deepcopy(resolved),
    )


def add_object_body(
    spec: mujoco.MjSpec,
    config: ObjectConfig,
    *,
    body_name: str | None = None,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mocap: bool | None = None,
) -> mujoco.MjsBody:
    """Add one configured object to an existing ``MjSpec``."""

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
            mesh_name = f"{geom_name}_mesh"
            spec.add_mesh(
                name=mesh_name,
                uservert=np.asarray(rounded.vertices, dtype=np.float64).ravel(),
                userface=np.asarray(rounded.faces, dtype=np.int32).ravel(),
                smoothnormal=1,
            )
            geom_kwargs.update(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
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


def object_local_aabb(config: ObjectConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative local-frame AABB for gallery and motion limits."""

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

    Returns a dict with keys ``translation`` and ``rotation``, each containing
    the fully-resolved parameters for that motion axis.
    """
    defaults = config.resolved.get("motion_defaults", {})
    specific = config.resolved.get("motion", {})
    result: dict[str, Any] = {}
    for axis in ("translation", "rotation"):
        base = defaults.get(axis, {})
        override = specific.get(axis, {})
        if not isinstance(base, dict) or not isinstance(override, dict):
            raise ValueError(
                f"Object {config.object_id!r} motion.{axis} must be a mapping"
            )
        result[axis] = _deep_merge(base, override)
    return result
