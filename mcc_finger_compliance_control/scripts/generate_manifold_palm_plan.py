"""Generate an inverse-compatible palm-only manifold trajectory.

The first recorded contact pose of a raw H5 is used as the reference.  A
smooth rigid orbit is generated in the object frame, while the object is
fixed for replay.  This is intended for visualizing the planner itself, not
for collecting teacher data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d

from object_catalog import MeshNormalOracle, load_object_config


DEFAULT_PALM_POS_WORLD = np.asarray((0.707417, -0.029887, 0.635323))
DEFAULT_PALM_ROTVEC_WORLD = np.asarray((-np.pi, 0.0, 0.0))
# ``palm_lower`` object-frame perimeter.  These are collision-outline points,
# not FSR centres: a cap or handle can hit this boundary long before the palm
# centre approaches the mesh.
PALM_OUTLINE_LOCAL = np.asarray(
    (
        (-0.100095, -0.027242, -0.0347224),
        (-0.100095, -0.054761, -0.0347224),
        (-0.093899, -0.080485, -0.0347224),
        (-0.071635, -0.093574, -0.0347224),
        (-0.044283, -0.096601, -0.0347224),
        (-0.036095, -0.078225, -0.0347224),
        (-0.036095, 0.004332, -0.0347224),
        (-0.042189, 0.025758, -0.0347224),
        (-0.065295, 0.015398, -0.0347224),
        (-0.082695, -0.005922, -0.0347224),
        (-0.067000, -0.041000, -0.0347224),
    ),
    dtype=np.float64,
)


def resample_pose_uniform_arc_length(pose: np.ndarray, frames: int) -> np.ndarray:
    """Reparameterize a pose path by spatial arc length.

    Mesh nearest-point projection can make equally spaced time samples move
    fast/slow even after Gaussian filtering.  Interpolating by cumulative
    palm-position arc length removes that artifact while using Slerp for the
    orientation, so the object sees a nearly constant tangential speed.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if len(pose) < 2 or frames < 2:
        return pose.astype(np.float32)
    position = pose[:, :3]
    distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(distance)))
    keep = np.concatenate(([True], distance > 1.0e-10))
    if int(keep.sum()) < 2:
        return pose.astype(np.float32)
    position = position[keep]
    pose = pose[keep]
    arc = arc[keep]
    total = float(arc[-1])
    if total < 1.0e-10:
        return pose.astype(np.float32)
    target = np.linspace(0.0, total, int(frames))
    out = np.zeros((int(frames), 7), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(target, arc, position[:, axis])
    quat_wxyz = pose[:, 3:7].copy()
    for i in range(1, len(quat_wxyz)):
        if float(quat_wxyz[i - 1] @ quat_wxyz[i]) < 0.0:
            quat_wxyz[i] *= -1.0
    rotations = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    interp = Slerp(arc, rotations)(target).as_quat()
    out[:, 3:] = interp[:, [3, 0, 1, 2]]
    return out.astype(np.float32)


def pose_to_rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat = pose[..., 3:7][..., [1, 2, 3, 0]]
    return pose[..., :3], R.from_quat(quat).as_matrix()


def rt_to_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    quat = R.from_matrix(rotation).as_quat()
    return np.concatenate((position, quat[..., [3, 0, 1, 2]]), axis=-1).astype(
        np.float32
    )


def relative_pose(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    ref_pos, ref_rot = pose_to_rt(reference)
    pos, rot = pose_to_rt(pose)
    inv_ref = np.swapaxes(ref_rot, -1, -2)
    return rt_to_pose(
        np.einsum("...ij,...j->...i", inv_ref, pos - ref_pos),
        inv_ref @ rot,
    )


def align_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotation that maps one nonzero vector onto another."""
    source = source / max(np.linalg.norm(source), 1.0e-12)
    target = target / max(np.linalg.norm(target), 1.0e-12)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1.0e-10:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.array((1.0, 0.0, 0.0))
        if abs(source @ basis) > 0.9:
            basis = np.array((0.0, 1.0, 0.0))
        axis = np.cross(source, basis)
        return R.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return R.from_rotvec(np.arctan2(sine, cosine) * cross / sine).as_matrix()


def enforce_palm_outline_clearance(
    position: np.ndarray,
    rotation: np.ndarray,
    oracle: MeshNormalOracle,
    clearance_m: float,
    max_iterations: int = 6,
) -> tuple[np.ndarray, float]:
    """Translate a planned palm until its entire collision outline is clear.

    Signed clearance is approximated from the nearest high-resolution mesh
    vertex and its smooth outward normal.  It is a planning guard, not a
    replacement for MuJoCo's final collision check.
    """
    position = position.copy()
    min_clearance = -np.inf
    for _ in range(max_iterations):
        outline_world = position + (rotation @ PALM_OUTLINE_LOCAL.T).T
        _, nearest = oracle.tree.query(outline_world)
        surface = oracle.vertices[np.asarray(nearest, dtype=np.int64)]
        normals = oracle.query_object_frame(surface)
        signed = np.einsum("ij,ij->i", outline_world - surface, normals)
        worst = int(np.argmin(signed))
        min_clearance = float(signed[worst])
        if min_clearance >= clearance_m:
            break
        position += (clearance_m - min_clearance) * normals[worst]
    return position, min_clearance


def palm_outline_clearance_stats(
    position: np.ndarray,
    rotation: np.ndarray,
    oracle: MeshNormalOracle,
) -> tuple[float, float, np.ndarray]:
    """Return mean/min palm-outline clearance and the sampled normals."""
    outline = position + (rotation @ PALM_OUTLINE_LOCAL.T).T
    _, nearest = oracle.tree.query(outline)
    surface = oracle.vertices[np.asarray(nearest, dtype=np.int64)]
    normals = oracle.query_object_frame(surface)
    signed = np.einsum("ij,ij->i", outline - surface, normals)
    return float(np.mean(signed)), float(np.min(signed)), normals


def generate(source: Path | None, output: Path, episode_id: int, frames: int,
             angle_deg: float, axis: np.ndarray, direction: int,
             control_dt: float, object_id: str, palm_pos_world: np.ndarray,
             palm_rotvec_world: np.ndarray, path_mode: str,
             path_length_m: float, palm_outline_clearance_m: float,
             smoothing_sigma_frames: float,
             palm_mean_clearance_m: float,
             object_rotation_mode: str = "fixed",
             seed: int = 0,
             ellipse_azimuth_deg: float = 0.0) -> None:
    if source is not None:
        with h5py.File(source, "r") as src:
            ids = np.asarray(src["episode_id"], dtype=np.int64)
            locations = np.argwhere(ids == episode_id)
            if locations.size == 0:
                raise ValueError(f"episode_id={episode_id} not found in {source}")
            locations = locations[np.argsort([src["record_step"][tuple(x)] for x in locations])]
            t0, e0 = locations[0]
            object0 = np.asarray(src["object_pose_world"][t0, e0], dtype=np.float64)
            palm0 = np.asarray(src["palm_pose_world"][t0, e0], dtype=np.float64)
            q0 = np.asarray(src["q_hand"][t0, e0], dtype=np.float32)
            tip0 = np.asarray(src["fingertip_pose_world"][t0, e0], dtype=np.float64)
            object_id = str(src.attrs.get("object_id", object_id))
            object_scale = float(src.attrs.get("object_scale", 1.0))
    else:
        config = load_object_config(object_id)
        object0 = np.concatenate((config.initial_pos, config.initial_rot)).astype(np.float64)
        if object_rotation_mode == "uniform_so3":
            rng = np.random.default_rng(seed)
            random_rot = R.random(random_state=rng).as_quat()
            object0[3:7] = random_rot[[3, 0, 1, 2]]
        palm0 = np.concatenate(
            (palm_pos_world, R.from_rotvec(palm_rotvec_world).as_quat()[[3, 0, 1, 2]])
        )
        q0 = np.asarray(config.collection["pregrasp_q"], dtype=np.float32)
        tip0 = np.zeros((4, 7), dtype=np.float64)
        tip0[:, 3] = 1.0
        scale_range = np.asarray(
            config.collection.get("size_scale_range", (1.0, 1.0)), dtype=np.float64
        )
        object_scale = float(scale_range.mean())

    _, object_rot0 = pose_to_rt(object0)
    palm_pos0, palm_rot0 = pose_to_rt(palm0)
    palm_pos_obj0 = object_rot0.T @ (palm_pos0 - object0[:3])
    palm_rot_obj0 = object_rot0.T @ palm_rot0
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1.0e-12)
    skew = np.array(((0.0, -axis[2], axis[1]),
                     (axis[2], 0.0, -axis[0]),
                     (-axis[1], axis[0], 0.0)))

    palm_object = np.zeros((frames, 7), dtype=np.float32)
    outline_clearance = np.full(frames, np.nan, dtype=np.float32)
    outline_mean_clearance = np.full(frames, np.nan, dtype=np.float32)
    if path_mode == "rigid_orbit":
        for i in range(frames):
            u = i / max(frames - 1, 1)
            smooth = u * u * (3.0 - 2.0 * u)
            angle = direction * np.deg2rad(angle_deg) * smooth
            delta = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
            palm_object[i] = rt_to_pose(delta @ palm_pos_obj0, delta @ palm_rot_obj0)
    elif path_mode == "ellipse_clearance":
        config = load_object_config(object_id)
        oracle = MeshNormalOracle.from_config(config, scale=object_scale)
        if oracle is None:
            raise ValueError("ellipse_clearance planning requires a mesh object")

        # Build one analytic ellipse in the cross-section normal to the PCA
        # long axis.  Unlike pointwise nearest-mesh projection, its position,
        # tangent and normal are continuous by construction.
        centered = oracle.vertices - oracle.vertices.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        long_axis = vectors[0] / np.linalg.norm(vectors[0])
        center = oracle.vertices.mean(axis=0)
        _, nearest_index = oracle.tree.query(palm_pos_obj0)
        initial_surface = oracle.vertices[int(nearest_index)].copy()
        height0 = float((initial_surface - center) @ long_axis)
        radial0 = initial_surface - center - height0 * long_axis
        radial0 /= max(float(np.linalg.norm(radial0)), 1.0e-12)
        tangent0 = np.cross(long_axis, radial0)
        tangent0 /= max(float(np.linalg.norm(tangent0)), 1.0e-12)
        # Select the ellipse plane explicitly in the object's local frame.
        # The plane is the cross-section spanned by radial_ref/tangent_ref;
        # rotating this basis around the PCA long axis changes the contact
        # route, while the clearance solve below still uses the full palm
        # outline and the source mesh.
        azimuth = np.deg2rad(float(ellipse_azimuth_deg))
        radial_ref = R.from_rotvec(long_axis * azimuth).apply(radial0)
        radial_ref /= max(float(np.linalg.norm(radial_ref)), 1.0e-12)
        tangent_ref = np.cross(long_axis, radial_ref)
        tangent_ref /= max(float(np.linalg.norm(tangent_ref)), 1.0e-12)

        height = centered @ long_axis
        half_span = max(0.01, 0.06 * float(np.ptp(height)))
        section = centered[np.abs(height - height0) <= half_span]
        if len(section) < 32:
            section = centered
        radius_a = float(
            np.percentile(np.abs(section @ radial0), 97.5)
        )
        radius_b = float(
            np.percentile(np.abs(section @ tangent0), 97.5)
        )
        radius_a = max(radius_a, 1.0e-3)
        radius_b = max(radius_b, 1.0e-3)

        base_position = np.zeros((frames, 3), dtype=np.float64)
        ellipse_normal = np.zeros_like(base_position)
        rotations = np.zeros((frames, 3, 3), dtype=np.float64)
        for i in range(frames):
            u = i / max(frames - 1, 1)
            # Quintic time scaling gives zero velocity and acceleration at
            # both endpoints; final arc-length resampling removes speed
            # variation caused by the unequal ellipse radii.
            smooth = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
            theta = direction * np.deg2rad(angle_deg) * smooth
            latitude = direction * path_length_m * smooth
            radial = (
                radius_a * np.cos(theta) * radial_ref
                + radius_b * np.sin(theta) * tangent_ref
            )
            normal = (
                np.cos(theta) / radius_a * radial_ref
                + np.sin(theta) / radius_b * tangent_ref
            )
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            base_position[i] = center + (height0 + latitude) * long_axis + radial
            ellipse_normal[i] = normal
            rotations[i] = align_vector(radial_ref, normal) @ palm_rot_obj0

        # Solve only one smooth scalar offset along the analytic ellipse
        # normal.  The objective is the mean distance of the complete palm
        # outline, rather than a discontinuous nearest vertex controlling the
        # whole pose.  Filtering the scalar correction preserves the ellipse.
        offset = np.full(frames, float(palm_mean_clearance_m), dtype=np.float64)
        sigma = max(1.0, float(smoothing_sigma_frames))
        for _ in range(5):
            correction = np.zeros(frames, dtype=np.float64)
            for i in range(frames):
                position = base_position[i] + offset[i] * ellipse_normal[i]
                mean_clearance, _, sampled_normals = palm_outline_clearance_stats(
                    position, rotations[i], oracle
                )
                sensitivity = float(
                    np.mean(sampled_normals @ ellipse_normal[i])
                )
                if abs(sensitivity) < 0.15:
                    sensitivity = np.copysign(0.15, sensitivity or 1.0)
                correction[i] = np.clip(
                    (palm_mean_clearance_m - mean_clearance) / sensitivity,
                    -0.02,
                    0.02,
                )
            correction = gaussian_filter1d(
                correction, sigma=sigma, mode="nearest"
            )
            offset += correction

        for i in range(frames):
            position = base_position[i] + offset[i] * ellipse_normal[i]
            palm_object[i] = rt_to_pose(position, rotations[i])
            mean_clearance, min_clearance, _ = palm_outline_clearance_stats(
                position, rotations[i], oracle
            )
            outline_mean_clearance[i] = mean_clearance
            outline_clearance[i] = min_clearance

        palm_object = resample_pose_uniform_arc_length(palm_object, frames)
        # Report clearances after resampling as these are the poses consumed
        # by the inverse collector.
        resampled_position, resampled_rotation = pose_to_rt(palm_object)
        for i in range(frames):
            mean_clearance, min_clearance, _ = palm_outline_clearance_stats(
                resampled_position[i], resampled_rotation[i], oracle
            )
            outline_mean_clearance[i] = mean_clearance
            outline_clearance[i] = min_clearance
    else:
        config = load_object_config(object_id)
        oracle = MeshNormalOracle.from_config(config, scale=object_scale)
        if oracle is None:
            raise ValueError("mesh_latlon planning requires a mesh object")
        _, nearest_index = oracle.tree.query(palm_pos_obj0)
        surface = oracle.vertices[int(nearest_index)].copy()
        normal = oracle.query_object_frame(surface)
        normal_sign = 1.0 if (palm_pos_obj0 - surface) @ normal >= 0.0 else -1.0
        clearance = abs(float((palm_pos_obj0 - surface) @ normal))
        clearance = max(clearance, 0.012)
        # Object-local latitude/longitude chart: the first PCA direction is
        # the bottle's long axis.  Longitude rotates around it; latitude
        # moves along it.  Every chart query is reprojected onto the mesh, so
        # a tapered bottle/cap remains a surface path rather than a cylinder.
        centered_vertices = oracle.vertices - oracle.vertices.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered_vertices, full_matrices=False)
        long_axis = vectors[0] / np.linalg.norm(vectors[0])
        center = oracle.vertices.mean(axis=0)
        height0 = float((surface - center) @ long_axis)
        radial0 = surface - center - height0 * long_axis
        radius0 = max(float(np.linalg.norm(radial0)), 1.0e-5)
        radial0 /= radius0
        tangent0 = np.cross(long_axis, radial0)
        tangent0 /= max(np.linalg.norm(tangent0), 1.0e-12)
        orientation = palm_rot_obj0.copy()
        previous_normal = normal_sign * normal
        position, outline_clearance[0] = enforce_palm_outline_clearance(
            surface + clearance * previous_normal,
            orientation,
            oracle,
            palm_outline_clearance_m,
        )
        palm_object[0] = rt_to_pose(position, orientation)
        for i in range(1, frames):
            u = i / max(frames - 1, 1)
            smooth = u * u * (3.0 - 2.0 * u)
            longitude = direction * np.deg2rad(angle_deg) * smooth
            latitude = direction * path_length_m * smooth
            radial = np.cos(longitude) * radial0 + np.sin(longitude) * tangent0
            candidate = center + (height0 + latitude) * long_axis + radius0 * radial
            _, nearest_index = oracle.tree.query(candidate)
            surface = oracle.vertices[int(nearest_index)].copy()
            normal = oracle.query_object_frame(surface)
            desired_normal = normal_sign * normal
            orientation = align_vector(previous_normal, desired_normal) @ orientation
            position, outline_clearance[i] = enforce_palm_outline_clearance(
                surface + clearance * desired_normal,
                orientation,
                oracle,
                palm_outline_clearance_m,
            )
            palm_object[i] = rt_to_pose(position, orientation)
            previous_normal = desired_normal

        if smoothing_sigma_frames > 0.0:
            position = gaussian_filter1d(
                palm_object[:, :3],
                sigma=smoothing_sigma_frames,
                axis=0,
                mode="nearest",
            )
            quat = palm_object[:, 3:7].copy()
            # Quaternion signs are equivalent but must be continuous before
            # filtering; otherwise a sign flip appears as a false 360 deg turn.
            for i in range(1, len(quat)):
                if float(quat[i - 1] @ quat[i]) < 0.0:
                    quat[i] *= -1.0
            quat = gaussian_filter1d(
                quat, sigma=smoothing_sigma_frames, axis=0, mode="nearest"
            )
            quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(1.0e-12)
            rotations = np.stack(
                [R.from_quat(q[[1, 2, 3, 0]]).as_matrix() for q in quat],
                axis=0,
            )
            # A pointwise projection is geometrically conservative but can
            # introduce a step when the nearest mesh vertex changes.  Filter
            # the *correction field* in time, then project again.  The small
            # extra clearance margin leaves room for the final filtered pass
            # while preserving the requested minimum after convergence.
            projection_target = float(palm_outline_clearance_m) + 0.002
            for _ in range(4):
                correction = np.zeros_like(position)
                for i in range(frames):
                    safe_position, _ = enforce_palm_outline_clearance(
                        position[i], rotations[i], oracle, projection_target
                    )
                    correction[i] = safe_position - position[i]
                correction = gaussian_filter1d(
                    correction,
                    sigma=max(1.0, smoothing_sigma_frames * 0.5),
                    axis=0,
                    mode="nearest",
                )
                position += correction
            # One final low-pass pass removes residual vertex-switch kinks.
            # The projection target above is intentionally inflated by 2 mm;
            # this pass therefore retains the requested 60 mm final margin in
            # practice without reintroducing pointwise discontinuities.
            position = gaussian_filter1d(
                position,
                sigma=max(2.0, smoothing_sigma_frames * 2.0),
                axis=0,
                mode="nearest",
            )
            for i in range(frames):
                palm_object[i] = rt_to_pose(position[i], rotations[i])
                _, outline_clearance[i] = enforce_palm_outline_clearance(
                    position[i], rotations[i], oracle, palm_outline_clearance_m
                )

        # The final path is sampled at constant spatial arc length.  This is
        # intentionally after the clearance projection: projecting first and
        # then resampling prevents mesh-vertex switches from becoming speed
        # spikes in the inverse object trajectory.
        palm_object = resample_pose_uniform_arc_length(palm_object, frames)

    tip_object0 = relative_pose(object0, tip0)
    tip_object = np.repeat(tip_object0[None, ...], frames, axis=0)
    q_hand = np.repeat(q0[None, :], frames, axis=0)
    qvel = np.zeros_like(q_hand)
    twist = np.zeros((frames, 6), dtype=np.float32)
    if frames > 1:
        p, r = pose_to_rt(palm_object)
        twist[1:, :3] = ((p[1:] - p[:-1]) / control_dt).astype(np.float32)
        twist[1:, 3:] = (
            R.from_matrix(r[1:] @ np.swapaxes(r[:-1], -1, -2)).as_rotvec()
            / control_dt
        ).astype(np.float32)
        twist[0] = twist[1]

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as dst:
        dst.create_dataset("palm_pose_object", data=palm_object[:, None])
        dst.create_dataset(
            "planner_palm_outline_min_clearance_object",
            data=outline_clearance[:, None],
        )
        dst.create_dataset(
            "planner_palm_outline_mean_clearance_object",
            data=outline_mean_clearance[:, None],
        )
        dst.create_dataset("palm_twist_object", data=twist[:, None])
        dst.create_dataset("q_hand", data=q_hand[:, None])
        dst.create_dataset("qvel", data=qvel[:, None])
        dst.create_dataset("fingertip_pose_object", data=tip_object[:, None])
        fixed_object = np.zeros((frames, 1, 7), dtype=np.float32)
        fixed_object[..., 3] = 1.0
        dst.create_dataset("object_pose_world", data=fixed_object)
        dst.create_dataset("episode_id", data=np.zeros((frames, 1), dtype=np.int64))
        dst.create_dataset("record_step", data=np.arange(frames)[:, None])
        dst.attrs["object_id"] = object_id
        dst.attrs["object_scale"] = object_scale
        dst.attrs["control_dt"] = float(control_dt)
        dst.attrs["inverted"] = True
        dst.attrs["pose_frame"] = "object"
        dst.attrs["planner_reference"] = (
            "first measured palm/object contact pose" if source is not None
            else "configured initial object pose and calibrated free-palm pose"
        )
        dst.attrs["planner_axis_local"] = axis
        dst.attrs["planner_angle_deg"] = float(angle_deg)
        dst.attrs["planner_path_mode"] = path_mode
        dst.attrs["planner_path_length_m"] = float(path_length_m)
        dst.attrs["planner_palm_outline_clearance_m"] = float(
            palm_outline_clearance_m
        )
        dst.attrs["planner_smoothing_sigma_frames"] = float(
            smoothing_sigma_frames
        )
        dst.attrs["planner_palm_mean_clearance_m"] = float(
            palm_mean_clearance_m
        )
        dst.attrs["planner_object_rotation_mode"] = object_rotation_mode
        dst.attrs["planner_seed"] = int(seed)
        dst.attrs["planner_ellipse_azimuth_deg"] = float(ellipse_azimuth_deg)
    print(f"[SUCCESS] palm manifold plan saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Optional raw H5 reference. Omit for direct object-frame planning.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="ycb_mustard")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--angle-deg", type=float, default=120.0)
    parser.add_argument("--axis", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--path-mode",
        choices=("mesh_latlon", "rigid_orbit", "ellipse_clearance"),
        default="mesh_latlon",
    )
    parser.add_argument(
        "--path-length-m", type=float, default=0.0,
        help="Signed latitude travel along the object's PCA long axis.",
    )
    parser.add_argument(
        "--palm-outline-clearance-m",
        type=float,
        default=0.060,
        help="Minimum mesh clearance for every palm-outline sample (default: 60 mm).",
    )
    parser.add_argument(
        "--smoothing-sigma-frames", type=float, default=8.0,
        help="Gaussian smoothing width applied before the final clearance pass.",
    )
    parser.add_argument(
        "--palm-mean-clearance-m", type=float, default=0.030,
        help=(
            "Target mean distance from the complete palm outline to the mesh "
            "for ellipse_clearance mode (default: 30 mm)."
        ),
    )
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument(
        "--object-rotation-mode",
        choices=("fixed", "uniform_so3"),
        default="fixed",
        help="Randomize the configured object's initial SO(3) pose for direct planning.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ellipse-azimuth-deg",
        type=float,
        default=0.0,
        help=(
            "Object-local rotation of the clearance ellipse basis around the "
            "PCA long axis; this selects the ellipse plane."
        ),
    )
    parser.add_argument(
        "--initial-palm-pos", nargs=3, type=float,
        default=DEFAULT_PALM_POS_WORLD,
    )
    parser.add_argument(
        "--initial-palm-rotvec", nargs=3, type=float,
        default=DEFAULT_PALM_ROTVEC_WORLD,
    )
    args = parser.parse_args()
    generate(
        args.source, args.output, args.episode_id, args.frames,
        args.angle_deg, np.asarray(args.axis), args.direction, args.control_dt,
        args.object_id, np.asarray(args.initial_palm_pos),
        np.asarray(args.initial_palm_rotvec), args.path_mode,
        args.path_length_m, args.palm_outline_clearance_m,
        args.smoothing_sigma_frames, args.palm_mean_clearance_m,
        args.object_rotation_mode, args.seed, args.ellipse_azimuth_deg,
    )


if __name__ == "__main__":
    main()
