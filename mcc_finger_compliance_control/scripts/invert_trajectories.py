from __future__ import annotations

import argparse
from functools import lru_cache
import glob
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import trimesh


CAPSULE_RADIUS = 0.15
CAPSULE_HALF_LENGTH = 0.08
BACK_CONTACT_X_LIMIT_M = np.asarray(
    (0.012, 0.012, 0.012, 0.016), dtype=np.float64
)
CONTACT_POINT_RADIUS_LIMIT_M = 0.05

# The old positive-X cutoff is useful as a cheap candidate test, but it is
# not sufficient by itself: the rounded lateral edge of the real fingertip
# mesh also reaches positive X.  Classify a point as rear-shell contact only
# when the nearest original-mesh normal is predominantly +X in the MCC site
# frame.  This keeps legitimate side contacts in the DP history.
BACK_NORMAL_MIN_X = 0.5
_TIP_ASSET_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/mjlab/asset_zoo/robots/xarm6_leap_hand/assets"
)
_TIP_MESH_SPECS = (
    (
        "fingertip.stl",
        (0.0132864, -0.00611424, 0.0145),
        (0.0, 1.0, 0.0, 0.0),
        (-0.0106151, -0.0326103, 0.0141088),
    ),
    (
        "fingertip.stl",
        (0.0132864, -0.00611424, 0.0145),
        (0.0, 1.0, 0.0, 0.0),
        (-0.0106151, -0.0326103, 0.0144487),
    ),
    (
        "fingertip.stl",
        (0.0132864, -0.00611424, 0.0145),
        (0.0, 1.0, 0.0, 0.0),
        (-0.0106151, -0.0326103, 0.0140386),
    ),
    (
        "thumb_fingertip.stl",
        (0.0625595, 0.0784597, 0.0489929),
        (1.0, 0.0, 0.0, 0.0),
        (-0.0106383, -0.0453895, -0.0144321),
    ),
)

SELECTED_RAW_FIELDS = (
    "episode_step",
    "q_hand",
    "object_pose_world",
    "palm_pose_world",
    "fingertip_pose_world",
    "fingertip_force_world",
    "fingertip_collision_found",
    "fingertip_contact_pos_world",
    "fingertip_contact_normal_world",
)
SELECTED_RAW_OPTIONAL_FIELDS = (
    # FullHandMCC records its smooth outward normal from the undecomposed
    # source mesh.  Keep it when every selected input provides it; this is a
    # much better tactile-normal teacher than V-HACD contact-face normals.
    "oracle_surface_normal_world",
    # Dual-track v2 execution contract.  These fields are copied only when
    # every selected trajectory provides them, preventing a mixed legacy/v2
    # bundle from silently fabricating missing controller state.
    "q_prior",
    "q_cmd",
    "delta_q_comp",
    "e_servo",
    "execution_perturbation_q",
    "execution_perturbation_target_norm_rad",
)


def _pose_to_rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat_xyzw = pose[..., 3:7][..., [1, 2, 3, 0]]
    rotation = R.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    return pose[..., :3], rotation


@lru_cache(maxsize=1)
def _tip_surface_indices() -> tuple[tuple[cKDTree, np.ndarray], ...]:
    """Build nearest-vertex indices for the original fingertip meshes."""

    indices: list[tuple[cKDTree, np.ndarray]] = []
    for mesh_name, geom_pos, geom_quat_wxyz, site_pos in _TIP_MESH_SPECS:
        mesh = trimesh.load_mesh(_TIP_ASSET_DIR / mesh_name, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"expected one mesh in {mesh_name}, got {type(mesh)!r}")
        geom_rotation = R.from_quat(
            np.asarray(geom_quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
        ).as_matrix()
        vertices = (
            np.asarray(mesh.vertices, dtype=np.float64) @ geom_rotation.T
            + np.asarray(geom_pos, dtype=np.float64)
            - np.asarray(site_pos, dtype=np.float64)
        )
        normals = (
            np.asarray(mesh.vertex_normals, dtype=np.float64) @ geom_rotation.T
        )
        indices.append((cKDTree(vertices), normals))
    return tuple(indices)


def _rear_shell_contact(
    contact_pos_tip: np.ndarray,
    loaded_contact: np.ndarray,
) -> np.ndarray:
    """Detect true rear-shell contacts while deliberately accepting sides."""

    rear = np.zeros_like(loaded_contact, dtype=bool)
    for finger, (tree, vertex_normals) in enumerate(_tip_surface_indices()):
        points = contact_pos_tip[..., finger, :].reshape(-1, 3)
        loaded = loaded_contact[..., finger].reshape(-1)
        candidates = loaded & (
            points[:, 0] > BACK_CONTACT_X_LIMIT_M[finger]
        )
        if not np.any(candidates):
            continue
        candidate_indices = np.flatnonzero(candidates)
        _, nearest = tree.query(points[candidates])
        normals = vertex_normals[nearest]
        normal_x = normals[:, 0]
        is_rear = (
            (normal_x >= BACK_NORMAL_MIN_X)
            & (normal_x >= np.abs(normals[:, 1]))
            & (normal_x >= np.abs(normals[:, 2]))
        )
        rear[..., finger].reshape(-1)[candidate_indices] = is_rear
    return rear


def _rt_to_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(rotation.reshape(-1, 3, 3)).as_quat().reshape(*position.shape[:-1], 4)
    return np.concatenate((position, quat_xyzw[..., [3, 0, 1, 2]]), axis=-1).astype(np.float32)


def _relative_pose(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    ref_p, ref_r = _pose_to_rt(reference)
    pos, rot = _pose_to_rt(pose)
    ref_rt = np.swapaxes(ref_r, -1, -2)
    relative_p = np.einsum("...ij,...j->...i", ref_rt, pos - ref_p)
    relative_r = ref_rt @ rot
    return _rt_to_pose(relative_p, relative_r)


def _capsule_surface_features(
    contact_pos_object: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic normal and principal-curvature proxy for the task capsule."""
    closest = np.zeros_like(contact_pos_object)
    closest[..., 2] = np.clip(
        contact_pos_object[..., 2], -CAPSULE_HALF_LENGTH, CAPSULE_HALF_LENGTH
    )
    radial = contact_pos_object - closest
    norm = np.linalg.norm(radial, axis=-1, keepdims=True)
    normal = radial / np.maximum(norm, 1.0e-9)
    on_cap = np.abs(contact_pos_object[..., 2]) > CAPSULE_HALF_LENGTH
    curvature = np.empty((*contact_pos_object.shape[:-1], 2), dtype=np.float64)
    curvature[..., 0] = 1.0 / CAPSULE_RADIUS
    curvature[..., 1] = np.where(on_cap, 1.0 / CAPSULE_RADIUS, 0.0)
    normal = np.where(valid[..., None], normal, 0.0)
    curvature = np.where(valid[..., None], curvature, 0.0)
    return normal.astype(np.float32), curvature.astype(np.float32)


def _enforce_normal_sign_continuity(
    normal: np.ndarray,
    valid: np.ndarray,
    episode_id: np.ndarray,
) -> np.ndarray:
    """Remove impossible one-frame normal sign flips within each episode.

    This changes only polarity, never the normal direction modulo sign.  A
    smooth physical trajectory cannot rotate its surface normal by more than
    90 degrees in one 10 ms step, whereas PCA/mesh winding ambiguities can.
    """

    output = np.asarray(normal, dtype=np.float64).copy()
    flat_normal = output.reshape(-1, 4, 3)
    flat_valid = np.asarray(valid, dtype=bool).reshape(-1, 4)
    flat_episode = np.asarray(episode_id).reshape(-1)
    for eid in np.unique(flat_episode):
        indices = np.flatnonzero(flat_episode == eid)
        for finger in range(4):
            previous: np.ndarray | None = None
            for index in indices:
                if not flat_valid[index, finger]:
                    continue
                current = flat_normal[index, finger]
                if previous is not None and float(np.dot(previous, current)) < 0.0:
                    current *= -1.0
                previous = current.copy()
    return output


def _backward_palm_twist(
    palm_pose_object: np.ndarray,
    episode_id: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute causal object-frame palm linear/angular velocity per episode."""
    position, rotation = _pose_to_rt(palm_pose_object)
    twist = np.zeros((*palm_pose_object.shape[:-1], 6), dtype=np.float32)
    flat_episode = episode_id.reshape(-1)
    flat_position = position.reshape(-1, 3)
    flat_rotation = rotation.reshape(-1, 3, 3)
    flat_twist = twist.reshape(-1, 6)
    for eid in np.unique(flat_episode):
        indices = np.flatnonzero(flat_episode == eid)
        if indices.size < 2:
            continue
        linear = (flat_position[indices[1:]] - flat_position[indices[:-1]]) / dt
        # R maps palm-frame vectors into the object frame.  R_next R_prev^T
        # is therefore the incremental rotation expressed in object
        # coordinates (spatial angular velocity), matching the dataset name.
        relative_rotation = (
            flat_rotation[indices[1:]]
            @ np.swapaxes(flat_rotation[indices[:-1]], -1, -2)
        )
        angular = R.from_matrix(relative_rotation).as_rotvec() / dt
        flat_twist[indices[1:], :3] = linear.astype(np.float32)
        flat_twist[indices[1:], 3:] = angular.astype(np.float32)
        # Avoid an artificial zero-velocity impulse at the first recorded frame.
        flat_twist[indices[0]] = flat_twist[indices[1]]
    return twist


def _backward_object_angular_velocity(
    object_pose_world: np.ndarray,
    episode_id: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Recover object angular velocity in its local frame from pose history.

    This avoids trusting legacy collection files whose commanded local axis
    was incorrectly labelled as a world-frame angular velocity.
    """
    rotation = _pose_to_rt(object_pose_world)[1]
    angular = np.zeros((*object_pose_world.shape[:-1], 3), dtype=np.float32)
    flat_episode = episode_id.reshape(-1)
    flat_rotation = rotation.reshape(-1, 3, 3)
    flat_angular = angular.reshape(-1, 3)
    for eid in np.unique(flat_episode):
        indices = np.flatnonzero(flat_episode == eid)
        if indices.size < 2:
            continue
        # R_prev^T R_next is the incremental rotation expressed in the
        # object's local/body frame.
        relative_rotation = (
            np.swapaxes(flat_rotation[indices[:-1]], -1, -2)
            @ flat_rotation[indices[1:]]
        )
        value = R.from_matrix(relative_rotation).as_rotvec() / dt
        flat_angular[indices[1:]] = value.astype(np.float32)
        flat_angular[indices[0]] = flat_angular[indices[1]]
    return angular


def invert(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
        object_pose = np.asarray(source["object_pose_world"], dtype=np.float64)
        palm_pose = np.asarray(source["palm_pose_world"], dtype=np.float64)
        tip_pose = np.asarray(source["fingertip_pose_world"], dtype=np.float64)

        palm_pose_object = _relative_pose(object_pose, palm_pose)
        target.create_dataset("palm_pose_object", data=palm_pose_object)
        episode_id = np.asarray(source["episode_id"], dtype=np.int64)
        control_dt = float(source.attrs.get("control_dt", 0.01))
        target.create_dataset(
            "palm_twist_object",
            data=_backward_palm_twist(
                palm_pose_object,
                episode_id,
                control_dt,
            ),
        )
        target.create_dataset(
            "fingertip_pose_object",
            data=_relative_pose(object_pose[..., None, :], tip_pose),
        )
        fixed_object = np.zeros_like(object_pose, dtype=np.float32)
        fixed_object[..., 3] = 1.0
        target.create_dataset("object_pose_world", data=fixed_object)

        object_r_t = np.swapaxes(_pose_to_rt(object_pose)[1], -1, -2)
        force_world = np.asarray(source["fingertip_force_world"], dtype=np.float64)
        force_object = np.einsum("...ij,...fj->...fi", object_r_t, force_world)
        target.create_dataset("fingertip_force_object", data=force_object.astype(np.float32))

        angular_object = _backward_object_angular_velocity(
            object_pose,
            episode_id,
            control_dt,
        )
        # In object-fixed replay the hand/palm relative angular velocity has the opposite sign.
        target.create_dataset("planned_palm_angular_velocity_object", data=(-angular_object).astype(np.float32))

        if "fingertip_contact_pos_world" in source:
            contact_pos_world = np.asarray(source["fingertip_contact_pos_world"], dtype=np.float64)
            tip_p, tip_r = _pose_to_rt(tip_pose)
            contact_pos_tip = np.einsum(
                "...fji,...fj->...fi",
                tip_r,
                contact_pos_world - tip_p,
            )
            force_norm = np.linalg.norm(force_world, axis=-1)
            if "fingertip_collision_found" in source:
                collision_found = np.asarray(
                    source["fingertip_collision_found"]
                ).astype(bool)
            else:
                collision_found = force_norm > 0.0
            contact_threshold = float(source.attrs.get("contact_threshold", 0.05))
            plausible_point = (
                np.isfinite(contact_pos_tip).all(axis=-1)
                & (
                    np.linalg.norm(contact_pos_tip, axis=-1)
                    <= CONTACT_POINT_RADIUS_LIMIT_M
                )
            )
            loaded_contact = (
                collision_found
                & (force_norm >= contact_threshold)
                & plausible_point
            )
            rear_shell_contact = _rear_shell_contact(
                contact_pos_tip, loaded_contact
            )
            valid = loaded_contact & ~rear_shell_contact
            target.create_dataset(
                "fingertip_contact",
                data=valid.astype(np.float32),
            )
            target.create_dataset(
                "fingertip_contact_pos_tip",
                data=contact_pos_tip.astype(np.float32),
            )
            object_p = object_pose[..., :3]
            contact_pos_object = np.einsum(
                "...ij,...fj->...fi",
                object_r_t,
                contact_pos_world - object_p[..., None, :],
            )
            contact_pos_object = np.where(valid[..., None], contact_pos_object, 0.0)
            target.create_dataset(
                "fingertip_contact_pos_object",
                data=contact_pos_object.astype(np.float32),
            )
            analytic_normal, curvature_object = _capsule_surface_features(
                contact_pos_object, valid
            )
            if "oracle_surface_normal_world" in source:
                # Source-mesh oracles use the conventional object-outward
                # normal.  The deployable fingertip ContactSensor reports
                # primary->secondary (fingertip->object), so store the
                # opposite, object-inward direction in every DP dataset.
                # This keeps primitive and mesh objects on one sensor-native
                # convention and avoids an object-dependent sign bit.
                normal_world = -np.asarray(
                    source["oracle_surface_normal_world"], dtype=np.float64
                )
                normal_norm = np.linalg.norm(
                    normal_world, axis=-1, keepdims=True
                )
                normal_world = normal_world / np.maximum(normal_norm, 1.0e-12)
                normal_object = np.einsum(
                    "...ij,...fj->...fi",
                    object_r_t,
                    normal_world,
                )
                normal_object = np.where(valid[..., None], normal_object, 0.0)
                normal_object = normal_object.astype(np.float32)
                normal_source = "undecomposed_source_mesh_oracle_inward"
                normal_polarity = "primary_fingertip_to_object"
            elif "fingertip_contact_normal_world" in source:
                normal_world = np.asarray(
                    source["fingertip_contact_normal_world"], dtype=np.float64
                )
                normal_object = np.einsum(
                    "...ij,...fj->...fi",
                    object_r_t,
                    normal_world,
                )
                normal_norm = np.linalg.norm(
                    normal_object, axis=-1, keepdims=True
                )
                normal_object = normal_object / np.maximum(normal_norm, 1.0e-12)
                normal_object = np.where(valid[..., None], normal_object, 0.0)
                normal_object = normal_object.astype(np.float32)
                normal_source = "recorded_contact_sensor"
                normal_polarity = "primary_fingertip_to_object"
            else:
                normal_object = -analytic_normal
                normal_source = "analytic_capsule_fallback_inward"
                normal_polarity = "primary_fingertip_to_object"
            normal_object = _enforce_normal_sign_continuity(
                normal_object, valid, episode_id
            ).astype(np.float32)
            target.create_dataset(
                "fingertip_contact_normal_object",
                data=normal_object,
            )
            target.create_dataset(
                "fingertip_curvature_object",
                data=curvature_object,
            )

        replaced = {
            "object_pose_world",
            "palm_pose_world",
            "fingertip_pose_world",
            "fingertip_force_world",
            "object_angular_velocity_world",
            "fingertip_contact_pos_world",
            "fingertip_contact_normal_world",
            "oracle_surface_normal_world",
            "fingertip_contact",
            "fingertip_contact_pos_tip",
        }
        for name, dataset in source.items():
            # Selected single-episode files may already carry derived fields
            # such as palm_pose_object.  Keep the freshly recomputed version.
            if name not in replaced and name not in target:
                target.create_dataset(name, data=np.asarray(dataset))
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["inverted"] = True
        target.attrs["pose_frame"] = "object"
        target.attrs["force_frame"] = "object"
        target.attrs["source_file"] = str(input_path)
        target.attrs["surface_features"] = "analytic_capsule_from_contact_position"
        target.attrs["contact_normal_source"] = normal_source
        target.attrs["contact_normal_normalized"] = True
        target.attrs["contact_normal_polarity"] = normal_polarity
        target.attrs["contact_normal_temporal_sign_continuity"] = True
        target.attrs["contact_mask_source"] = (
            "same_frame_collision_force_and_tip_surface_geometry"
        )
        target.attrs["contact_point_radius_limit_m"] = (
            CONTACT_POINT_RADIUS_LIMIT_M
        )
        target.attrs["back_contact_x_limit_m"] = BACK_CONTACT_X_LIMIT_M
        target.attrs["back_contact_classifier"] = (
            "positive_site_x_and_nearest_original_mesh_normal_dominant_positive_x"
        )
        target.attrs["back_contact_normal_min_x"] = BACK_NORMAL_MIN_X
        target.attrs["side_contact_allowed"] = True
        target.attrs["palm_twist"] = "causal_backward_difference_in_object_frame"
        target.attrs["planned_angular_velocity_source"] = (
            "causal_object_pose_difference_in_object_frame"
        )
        target.attrs["capsule_radius"] = CAPSULE_RADIUS
        target.attrs["capsule_half_length"] = CAPSULE_HALF_LENGTH
    print(f"[SUCCESS] inverted trajectory saved to {output_path}")


def _selected_paths(selected_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in selected_dirs:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        paths.extend(sorted(directory.glob("*.h5")))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise RuntimeError("no selected trajectory H5 files found")
    return paths


def _h5_contains(path: Path, name: str) -> bool:
    with h5py.File(path, "r") as source:
        return name in source


def _bundle_selected_raw(
    paths: list[Path], bundle_path: Path
) -> list[str]:
    """Build a minimal raw bundle from single- or multi-env H5 files."""
    raw_fields = list(SELECTED_RAW_FIELDS)
    for optional_name in SELECTED_RAW_OPTIONAL_FIELDS:
        if all(
            _h5_contains(path, optional_name)
            for path in paths
        ):
            raw_fields.append(optional_name)
    episodes: list[tuple[Path, int, int]] = []
    frame_counts: list[int] = []
    control_dts: list[float] = []
    object_ids: list[str] = []
    layouts: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    first_attrs: dict[str, object] = {}
    for path_index, path in enumerate(paths):
        with h5py.File(path, "r") as source:
            if path_index == 0:
                first_attrs = dict(source.attrs)
            if "q_hand" not in source:
                raise KeyError(f"{path}: missing 'q_hand'")
            q_shape = source["q_hand"].shape
            if len(q_shape) < 3 or q_shape[1] < 1:
                raise ValueError(f"{path}: expected (T,E,...) H5, got {q_shape}")
            frames = int(q_shape[0])
            env_count = int(q_shape[1])
            for env in range(env_count):
                episodes.append((path, env, env_count))
                frame_counts.append(frames)
                control_dts.append(float(source.attrs.get("control_dt", 0.01)))
                object_ids.append(str(source.attrs.get("object_id", "")))
            for name in raw_fields:
                if name == "episode_step" and name not in source:
                    continue
                if name not in source:
                    raise KeyError(f"{path}: missing required raw field {name!r}")
                dataset = source[name]
                if dataset.shape[0] != frames or dataset.shape[1] != env_count:
                    raise ValueError(
                        f"{path}:{name} must begin with (T,E), got {dataset.shape}"
                    )
                layout = ((1, *dataset.shape[2:]), dataset.dtype)
                previous = layouts.setdefault(name, layout)
                if previous != layout:
                    raise ValueError(
                        f"inconsistent {name} layout: {previous} versus {layout} in {path}"
                    )
    control_dt = control_dts[0]
    if not np.allclose(control_dts, control_dt, rtol=0.0, atol=1.0e-12):
        raise ValueError("selected trajectories use different control_dt values")
    if len(set(object_ids)) != 1:
        raise ValueError(
            "selected trajectories contain multiple object_id values; invert each "
            "object separately before constructing a multi-object DP dataset"
        )

    total = int(sum(frame_counts))
    with h5py.File(bundle_path, "w") as target:
        outputs: dict[str, h5py.Dataset] = {}
        for name, (tail_shape, dtype) in layouts.items():
            outputs[name] = target.create_dataset(
                name,
                shape=(total, *tail_shape),
                dtype=dtype,
                chunks=(min(4096, total), *tail_shape),
            )
        episode_out = target.create_dataset(
            "episode_id",
            shape=(total, 1),
            dtype="i4",
            chunks=(min(4096, total), 1),
        )
        if "episode_step" not in outputs:
            outputs["episode_step"] = target.create_dataset(
                "episode_step",
                shape=(total, 1),
                dtype="i4",
                chunks=(min(4096, total), 1),
            )

        cursor = 0
        for episode, ((path, env, _), frames) in enumerate(
            zip(episodes, frame_counts, strict=True)
        ):
            selection = slice(cursor, cursor + frames)
            with h5py.File(path, "r") as source:
                for name in raw_fields:
                    if name in source:
                        outputs[name][selection] = source[name][:, env : env + 1]
                if "episode_step" not in source:
                    outputs["episode_step"][selection, 0] = np.arange(
                        frames, dtype=np.int32
                    )
            episode_out[selection, 0] = episode
            cursor += frames

        for key, value in first_attrs.items():
            target.attrs[key] = value
        target.attrs["control_dt"] = control_dt
        target.attrs["object_id"] = object_ids[0]
        target.attrs["num_trajectories"] = len(episodes)
        target.attrs["selected_raw_bundle"] = True
    return [
        str(path) if env_count == 1 else f"{path}#env={env}"
        for path, env, env_count in episodes
    ]


def invert_selected(
    selected_dirs: list[Path], raw_files: list[Path], output_path: Path
) -> None:
    """Explicitly invert selected passive trajectories into one replay H5."""
    paths = _selected_paths(selected_dirs) if selected_dirs else []
    for raw_file in raw_files:
        if not raw_file.is_file():
            raise FileNotFoundError(raw_file)
        paths.append(raw_file.resolve())
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("no selected trajectory H5 files found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="selected_raw_bundle_",
            suffix=".h5",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        source_labels = _bundle_selected_raw(paths, temporary_path)
        invert(temporary_path, output_path)
        with h5py.File(output_path, "a") as target:
            source_names = target.create_dataset(
                "source_trajectory",
                shape=(len(source_labels),),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            source_names[:] = source_labels
            target.attrs["source_trajectory_count"] = len(source_labels)
            target.attrs["source_file"] = "selected trajectory bundle"
            target.attrs["inversion_pipeline"] = (
                "passive_object_motion_to_fixed_object_active_palm_motion"
            )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(
        f"[SUCCESS] explicitly inverted {len(source_labels)} selected episodes into "
        f"{output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert MCC fingertip data into the object frame.")
    parser.add_argument("--file", default=None)
    parser.add_argument(
        "--selected-dir",
        type=Path,
        action="append",
        default=[],
        help="Selected single-episode directory; repeat to merge directories.",
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        action="append",
        default=[],
        help="Raw (T,E,...) H5; repeat to append every environment as an episode.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.selected_dir or args.raw_file:
        if args.file is not None:
            parser.error("--file and --selected-dir/--raw-file are mutually exclusive")
        if args.output is None:
            parser.error("--output is required with --selected-dir/--raw-file")
        invert_selected(args.selected_dir, args.raw_file, Path(args.output))
        return
    if args.file is None:
        candidates = glob.glob("mcc_finger_compliance_control/data/trajectories/*.h5")
        if not candidates:
            raise FileNotFoundError("No trajectory H5 found")
        input_path = Path(max(candidates, key=lambda path: Path(path).stat().st_mtime))
    else:
        input_path = Path(args.file)
    output_path = Path(args.output) if args.output else (
        Path("mcc_finger_compliance_control/data/inverted") / f"{input_path.stem}_inverted.h5"
    )
    invert(input_path, output_path)


if __name__ == "__main__":
    main()
