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
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline

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
PALM_CENTER_LOCAL = np.asarray(
    (-0.0559703, -0.04142053, -0.0340008), dtype=np.float64
)


def resample_pose_uniform_arc_length(
    pose: np.ndarray,
    frames: int,
    local_control_point: np.ndarray | None = None,
    ramp_fraction: float = 0.0,
) -> np.ndarray:
    """Reparameterize a pose path by spatial arc length.

    Mesh nearest-point projection can make equally spaced time samples move
    fast/slow even after Gaussian filtering.  Interpolating by cumulative
    palm-position arc length removes that artifact while using Slerp for the
    orientation, so the object sees a nearly constant tangential speed.

    ``ramp_fraction`` (0..0.5) additionally cosine-ramps the first/last
    fraction of the trajectory down to zero speed: a fully uniform profile
    would start the object's equivalent rotation at full rate on the very
    first physics step, which sheds the weakest fingertip contact before the
    contact-manifold QP can adapt.  With ramp the middle arc runs slightly
    above the mean while both ends settle smoothly.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if len(pose) < 2 or frames < 2:
        return pose.astype(np.float32)
    position = pose[:, :3]
    if local_control_point is None:
        arc_position = position
    else:
        _, rotation = pose_to_rt(pose)
        arc_position = position + np.einsum(
            "nij,j->ni",
            rotation,
            np.asarray(local_control_point, dtype=np.float64),
        )
    distance = np.linalg.norm(np.diff(arc_position, axis=0), axis=1)
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
    if ramp_fraction > 0.0:
        # Map uniform time u onto arc fraction s whose speed profile is a
        # cosine ramp at both ends and flat in the middle (same 10% ramp as
        # the pre-resample planner).  s = cumsum(w)/sum(w) with w(u) the
        # desired speed envelope.
        u = np.linspace(0.0, 1.0, int(frames))
        r = min(float(ramp_fraction), 0.5)
        w = np.ones_like(u)
        lo = u < r
        hi = u > 1.0 - r
        w[lo] = 0.5 * (1.0 - np.cos(np.pi * u[lo] / r))
        w[hi] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - u[hi]) / r))
        s = np.cumsum(w)
        s /= float(s[-1])
        target = s * total
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


def resample_pose_uniform_blend(
    pose: np.ndarray,
    frames: int,
    rotation_weight: float = 0.5,
    ramp_fraction: float = 0.0,
) -> np.ndarray:
    """Reparameterize by a blended translation+rotation arc.

    Pure position arc length equalizes palm speed but the equivalent object
    rotation can still swing ~1.8x along it (the fast segments shed the
    weakest fingertip).  Pure rotation arc explodes position speed.  This
    blends both: each step contributes its position arc and its geodesic
    rotation angle, each normalized to its own total, so the two components
    have equal weight and neither blows up.  ``rotation_weight`` shifts the
    balance (0 = pure position, 1 = pure rotation).
    """
    pose = np.asarray(pose, dtype=np.float64)
    if len(pose) < 2 or frames < 2:
        return pose.astype(np.float32)
    position = pose[:, :3]
    distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
    quat = pose[:, 3:7].copy()
    for i in range(1, len(quat)):
        if float(quat[i - 1] @ quat[i]) < 0.0:
            quat[i] *= -1.0
    dots = np.clip(np.sum(quat[:-1] * quat[1:], axis=1), -1.0, 1.0)
    rot_step = 2.0 * np.arccos(np.abs(dots))
    # Normalize each component to unit total so the blend is scale-free.
    dpos_n = distance / max(float(distance.sum()), 1.0e-12)
    drot_n = rot_step / max(float(rot_step.sum()), 1.0e-12)
    blend = (1.0 - rotation_weight) * dpos_n + rotation_weight * drot_n
    arc = np.concatenate(([0.0], np.cumsum(blend)))
    keep = np.concatenate(([True], blend > 1.0e-9))
    if int(keep.sum()) < 2:
        return pose.astype(np.float32)
    position = position[keep]
    quat = quat[keep]
    arc = arc[keep]
    total = float(arc[-1])
    if total < 1.0e-9:
        return pose.astype(np.float32)
    target = np.linspace(0.0, total, int(frames))
    if ramp_fraction > 0.0:
        u = np.linspace(0.0, 1.0, int(frames))
        r = min(float(ramp_fraction), 0.5)
        w = np.ones_like(u)
        lo = u < r
        hi = u > 1.0 - r
        w[lo] = 0.5 * (1.0 - np.cos(np.pi * u[lo] / r))
        w[hi] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - u[hi]) / r))
        s = np.cumsum(w)
        s /= float(s[-1])
        target = s * total
    out = np.zeros((int(frames), 7), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(target, arc, position[:, axis])
    rotations = R.from_quat(quat[:, [1, 2, 3, 0]])
    interp = Slerp(arc, rotations)(target).as_quat()
    out[:, 3:] = interp[:, [3, 0, 1, 2]]
    return out.astype(np.float32)


def resample_pose_uniform_rotation(
    pose: np.ndarray,
    frames: int,
) -> np.ndarray:
    """Reparameterize a pose path by cumulative orientation distance.

    Arc-length resampling equalizes palm *position* speed, but the executed
    equivalent object rotation is the palm orientation path, and on an
    ellipse (especially an oblique or longitudinal section) the rotation
    rate varies by several times along it.  Interpolating by cumulative
    geodesic rotation angle keeps the object's equivalent angular speed
    nearly constant, so no trajectory segment exceeds the contact manifold
    QP's adaptation rate.  Positions are linearly interpolated along the
    same rotation-arc parameter.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if len(pose) < 2 or frames < 2:
        return pose.astype(np.float32)
    quat = pose[:, 3:7].copy()
    # De-sign so consecutive quaternions move continuously.
    for i in range(1, len(quat)):
        if float(quat[i - 1] @ quat[i]) < 0.0:
            quat[i] *= -1.0
    dots = np.clip(np.sum(quat[:-1] * quat[1:], axis=1), -1.0, 1.0)
    delta = 2.0 * np.arccos(np.abs(dots))
    arc = np.concatenate(([0.0], np.cumsum(delta)))
    keep = np.concatenate(([True], delta > 1.0e-9))
    if int(keep.sum()) < 2:
        return pose.astype(np.float32)
    position = pose[:, :3][keep]
    quat = quat[keep]
    arc = arc[keep]
    total = float(arc[-1])
    if total < 1.0e-9:
        return pose.astype(np.float32)
    target = np.linspace(0.0, total, int(frames))
    out = np.zeros((int(frames), 7), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(target, arc, position[:, axis])
    # Use the de-signed quaternions so Slerp never walks the 360-theta
    # long arc through a sign flip.
    rotations = R.from_quat(quat[:, [1, 2, 3, 0]])
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


def palm_plan_motion_metrics(pose: np.ndarray) -> dict[str, float]:
    """Measure the *realized* SE(3) path, not only its nominal arc input."""

    pose = np.asarray(pose, dtype=np.float64).reshape(-1, 7)
    if len(pose) < 2:
        return {
            "translation_path_m": 0.0,
            "rotation_path_deg": 0.0,
            "max_translation_step_m": 0.0,
            "max_rotation_step_deg": 0.0,
        }
    position, rotation = pose_to_rt(pose)
    translation_step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    rotation_step = R.from_matrix(
        rotation[1:] @ np.swapaxes(rotation[:-1], -1, -2)
    ).magnitude()
    return {
        "translation_path_m": float(np.sum(translation_step)),
        "rotation_path_deg": float(np.rad2deg(np.sum(rotation_step))),
        "max_translation_step_m": float(np.max(translation_step)),
        "max_rotation_step_deg": float(np.rad2deg(np.max(rotation_step))),
    }


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


def minimum_enclosing_ellipse_2d(
    points: np.ndarray,
    tolerance: float = 1.0e-7,
    max_iterations: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centre, orthonormal axes and radii of the 2-D MVEE.

    Khachiyan's algorithm is applied only to the convex-hull vertices.  The
    returned ellipse contains the complete projected source mesh, rather
    than fitting a percentile that may cut through a cap or shoulder.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("minimum enclosing ellipse requires >=3 finite points")
    hull = ConvexHull(points)
    boundary = points[hull.vertices]
    q = np.vstack((boundary.T, np.ones(len(boundary))))
    u = np.full(len(boundary), 1.0 / len(boundary), dtype=np.float64)
    dimension = 2
    for _ in range(max_iterations):
        x = (q * u[None, :]) @ q.T
        x_inv = np.linalg.pinv(x, rcond=1.0e-12)
        leverage = np.einsum("in,ij,jn->n", q, x_inv, q)
        index = int(np.argmax(leverage))
        maximum = float(leverage[index])
        if maximum <= dimension + 1.0 + tolerance:
            break
        step = (maximum - dimension - 1.0) / (
            (dimension + 1.0) * max(maximum - 1.0, 1.0e-12)
        )
        next_u = (1.0 - step) * u
        next_u[index] += step
        if np.linalg.norm(next_u - u) <= tolerance:
            u = next_u
            break
        u = next_u
    center = boundary.T @ u
    covariance = (boundary.T * u[None, :]) @ boundary - np.outer(center, center)
    shape = np.linalg.pinv(covariance, rcond=1.0e-12) / dimension
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    radii = 1.0 / np.sqrt(np.maximum(eigenvalues, 1.0e-12))
    order = np.argsort(radii)[::-1]
    axes = eigenvectors[:, order]
    radii = radii[order]
    # Numerical termination can leave a tiny containment violation.  Inflate
    # once by the exact maximum quadratic value so every projected vertex is
    # guaranteed to lie on or inside the returned ellipse.
    local = (points - center) @ axes
    normalized = np.sum((local / radii[None, :]) ** 2, axis=1)
    radii *= np.sqrt(max(1.0, float(np.max(normalized))))
    return center, axes, radii


def palm_facing_rotation(
    reference_rotation: np.ndarray,
    inward_normal: np.ndarray,
) -> np.ndarray:
    """Preserve wrist roll while turning palm -Z toward the object."""
    reference_facing = reference_rotation @ np.array((0.0, 0.0, -1.0))
    return align_vector(reference_facing, inward_normal) @ reference_rotation


def palm_surface_tangent_rotation(
    surface_outward_normal: np.ndarray,
    motion_tangent: np.ndarray,
    tangent_axis: str,
    tangent_sign: int = 1,
) -> np.ndarray:
    """Build a palm frame from the measured mesh normal and path tangent.

    Palm ``-Z`` faces the object, hence palm ``+Z`` follows the outward mesh
    normal.  The selected palm +X/+Y axis follows the direction of travel and
    removes the otherwise arbitrary wrist-roll degree of freedom.
    """
    z_axis = np.asarray(surface_outward_normal, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)
    tangent = float(tangent_sign) * np.asarray(
        motion_tangent, dtype=np.float64
    )
    tangent -= z_axis * float(tangent @ z_axis)
    if float(np.linalg.norm(tangent)) < 1.0e-9:
        fallback = np.array((1.0, 0.0, 0.0))
        if abs(float(fallback @ z_axis)) > 0.9:
            fallback = np.array((0.0, 1.0, 0.0))
        tangent = fallback - z_axis * float(fallback @ z_axis)
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
    if tangent_axis == "x":
        x_axis = tangent
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
        x_axis = np.cross(y_axis, z_axis)
    elif tangent_axis == "y":
        y_axis = tangent
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
        y_axis = np.cross(z_axis, x_axis)
    else:
        raise ValueError(f"Unsupported palm tangent axis: {tangent_axis!r}")
    return np.column_stack((x_axis, y_axis, z_axis))


def smooth_surface_aligned_poses(
    control_position: np.ndarray,
    oracle: MeshNormalOracle,
    tangent_axis: str,
    anchor_spacing_frames: float,
    tangent_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a continuous palm pose sequence with B-spline mesh normals.

    The analytic ellipse is already smooth.  Only its desired surface normal
    was discontinuous because a nearest triangle changes abruptly.  Sample a
    small set of anchor normals from local points of the original high-detail
    mesh, interpolate them over trajectory arc length with a cubic B-spline,
    and normalize the result.  The analytic control-point path is untouched.
    """

    control = np.asarray(control_position, dtype=np.float64).reshape(-1, 3)
    segment = np.linalg.norm(np.diff(control, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    if float(arc[-1]) <= 1.0e-12:
        arc = np.linspace(0.0, 1.0, len(control))
    else:
        arc /= arc[-1]
    spacing = max(4, int(round(float(anchor_spacing_frames))))
    anchor_count = int(np.clip(np.ceil((len(control) - 1) / spacing) + 1, 6, 40))
    anchor_arc = np.linspace(0.0, 1.0, anchor_count)
    anchor_control = np.column_stack(
        [np.interp(anchor_arc, arc, control[:, axis]) for axis in range(3)]
    )
    # ``anchor_control`` is the palm control point, intentionally several
    # centimetres outside the mesh.  Estimating a local PCA normal around
    # that remote point makes the neighbourhood depend on the standoff and
    # can jump between the bottle wall, shoulder and cap.  This produced a
    # 251 degree wrist excursion for a nominal 12 degree cap arc.  Project
    # each anchor onto the source mesh first, then estimate the normal at the
    # actual material surface point.  The collision decomposition is not
    # involved in this query.
    _, anchor_nearest = oracle.tree.query(anchor_control)
    anchor_surface = oracle.vertices[
        np.asarray(anchor_nearest, dtype=np.int64)
    ]
    anchor_normals = np.asarray(
        oracle.query_object_frame(anchor_surface), dtype=np.float64
    )
    # Protect against an isolated winding/sign defect at an anchor.
    for index in range(1, len(anchor_normals)):
        if float(anchor_normals[index - 1] @ anchor_normals[index]) < 0.0:
            anchor_normals[index] *= -1.0
    spline = make_interp_spline(
        anchor_arc,
        anchor_normals,
        k=min(3, anchor_count - 1),
        axis=0,
    )
    normals = np.asarray(spline(arc), dtype=np.float64)
    normals /= np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12
    )

    tangent = float(tangent_sign) * np.gradient(control, axis=0)
    # Fix the wrist-roll branch from the first reliable route tangent, then
    # parallel-transport it with the changing surface normal.  Do *not*
    # continuously blend it back toward the instantaneous route tangent.
    #
    # On a non-convex/strongly tapered mesh, that projected route tangent can
    # rotate through several complete turns even though the palm control point
    # follows one short arc.  The old per-frame blend was locally smooth, but
    # accumulated 120--550 degrees of unnecessary wrist roll on some Mustard
    # cuts.  Exact minimal rotation between consecutive normals gives the
    # rotation-minimizing (Bishop) frame wanted by the passive-compliance
    # collector: palm +Z follows the surface normal while wrist yaw changes
    # only as much as the normal field requires.
    transported_axes = np.zeros_like(normals)
    previous_axis: np.ndarray | None = None
    previous_normal: np.ndarray | None = None
    for index, (normal, direction) in enumerate(
        zip(normals, tangent, strict=True)
    ):
        projected = direction - normal * float(direction @ normal)
        projected_norm = float(np.linalg.norm(projected))
        desired = (
            projected / projected_norm
            if projected_norm > 1.0e-10
            else None
        )
        if previous_axis is None:
            if desired is None:
                fallback = np.array((1.0, 0.0, 0.0), dtype=np.float64)
                if abs(float(fallback @ normal)) > 0.9:
                    fallback = np.array((0.0, 1.0, 0.0), dtype=np.float64)
                axis = fallback - normal * float(fallback @ normal)
                axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
            else:
                axis = desired
        else:
            assert previous_normal is not None
            normal_step = align_vector(previous_normal, normal)
            axis = normal_step @ previous_axis
            # Remove round-off accumulated over a long trajectory and retain
            # a unit tangent vector.  This projection is not used to choose a
            # new wrist branch.
            axis -= normal * float(axis @ normal)
            axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        transported_axes[index] = axis
        previous_axis = axis
        previous_normal = normal

    rotations = np.zeros((len(control), 3, 3), dtype=np.float64)
    for index, (normal, selected_axis) in enumerate(
        zip(normals, transported_axes, strict=True)
    ):
        if tangent_axis == "x":
            x_axis = selected_axis
            y_axis = np.cross(normal, x_axis)
            y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
            x_axis = np.cross(y_axis, normal)
        elif tangent_axis == "y":
            y_axis = selected_axis
            x_axis = np.cross(y_axis, normal)
            x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
            y_axis = np.cross(normal, x_axis)
        else:
            raise ValueError(f"Unsupported palm tangent axis: {tangent_axis!r}")
        rotations[index] = np.column_stack((x_axis, y_axis, normal))
    body_position = control - np.einsum(
        "nij,j->ni", rotations, PALM_CENTER_LOCAL
    )
    return body_position, rotations, anchor_count


def _generate_from_recovery_state(
    recovery_state: Path,
    output: Path,
    frames: int,
    control_dt: float,
    max_plan_translation_path_m: float,
    max_plan_rotation_path_deg: float,
    max_plan_translation_step_m: float,
    max_plan_rotation_step_deg: float,
) -> None:
    """Continue a DP-failure rollout's own palm trajectory past state A,
    instead of deriving a fresh analytic path (rigid_orbit / ellipse / ...).

    ``DAgger_Recovery_Data_Guide.md``'s continuity requirement is
    ``p_new(0)=p_A``, ``v_new(0)~=v_original(A)``. Under the deployment
    default ``--palm-source teacher`` -- the only mode
    ``collect_dagger_rollouts.py`` actually uses -- the rollout's palm
    motion is already a verbatim replay of the source episode's own
    recorded trajectory regardless of the finger contact outcome, so the
    "recovery" palm path *is* that recorded continuation: already exactly
    continuous at A by construction, no blend or re-plan needed. See
    ``extract_recovery_state.py``'s ``--palm-source`` caveat for the one
    case (``--palm-source active_capsule``) where this would not hold --
    that case is out of scope here, same as it is there.
    """

    with h5py.File(recovery_state, "r") as state:
        if str(state.attrs.get("format", "")) != "mcc_recovery_state_v1":
            raise ValueError(
                f"{recovery_state}: expected attrs['format']="
                f"'mcc_recovery_state_v1' (extract_recovery_state.py output), "
                f"got {state.attrs.get('format')!r}"
            )
        continuation_pose = np.asarray(
            state["continuation_palm_pose_object"], dtype=np.float32
        )
        continuation_twist = np.asarray(
            state["continuation_palm_twist_object"], dtype=np.float32
        )
        q0 = np.asarray(state["q_live"], dtype=np.float32)
        object_id = str(state.attrs.get("object_id", ""))
        object_scale = float(state.attrs.get("object_scale", 1.0))
        failure_frame = int(state.attrs.get("failure_frame", -1))
        source_rollout_file = str(state.attrs.get("source_rollout_file", ""))

    available = len(continuation_pose)
    if available == 0:
        raise ValueError(
            f"{recovery_state}: empty continuation (state A was already the "
            "last frame of its source episode)"
        )
    if frames > available:
        print(
            f"[WARN] --frames={frames} exceeds the {available} frames "
            f"available past state A in {recovery_state}; truncating to "
            f"{available}."
        )
        frames = available

    palm_object = continuation_pose[:frames].copy()
    twist = continuation_twist[:frames].copy()
    tip0 = np.zeros((4, 7), dtype=np.float64)
    tip0[:, 3] = 1.0  # identity quaternion placeholder: state A does not
    # record fingertip pose, only contact pos/normal/force; downstream
    # (optimize_contact_plan.py) re-solves finger geometry from scratch and
    # does not read this field, same placeholder convention already used
    # above for the no-`--source` config branch.
    tip_object = np.repeat(tip0[None, ...], frames, axis=0)
    q_hand = np.repeat(q0[None, :], frames, axis=0)
    qvel = np.zeros_like(q_hand)
    outline_clearance = np.full(frames, np.nan, dtype=np.float32)
    outline_mean_clearance = np.full(frames, np.nan, dtype=np.float32)

    motion_metrics = palm_plan_motion_metrics(palm_object)
    limits = (
        ("translation_path_m", max_plan_translation_path_m),
        ("rotation_path_deg", max_plan_rotation_path_deg),
        ("max_translation_step_m", max_plan_translation_step_m),
        ("max_rotation_step_deg", max_plan_rotation_step_deg),
    )
    violations = [
        f"{name}={motion_metrics[name]:.6g} > {float(limit):.6g}"
        for name, limit in limits
        if float(limit) > 0.0 and motion_metrics[name] > float(limit)
    ]
    if violations:
        raise ValueError(
            "rejecting kinematically irregular recovery continuation: "
            + "; ".join(violations)
        )

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
        dst.attrs["planner_reference"] = "recovery continuation from state A"
        dst.attrs["planner_path_mode"] = "recovery_state"
        dst.attrs["planner_recovery_state_file"] = str(recovery_state)
        dst.attrs["planner_recovery_source_rollout_file"] = source_rollout_file
        dst.attrs["planner_recovery_failure_frame"] = failure_frame
        for name, value in motion_metrics.items():
            dst.attrs[f"planner_realized_{name}"] = float(value)
    print(
        f"[SUCCESS] palm manifold plan (recovery continuation) saved to "
        f"{output} ({frames} frames continuing from failure_frame="
        f"{failure_frame} of {source_rollout_file})"
    )


def generate(source: Path | None, output: Path, episode_id: int, frames: int,
             angle_deg: float, axis: np.ndarray, direction: int,
             control_dt: float, object_id: str, palm_pos_world: np.ndarray,
             palm_rotvec_world: np.ndarray, path_mode: str,
             path_length_m: float, palm_outline_clearance_m: float,
             smoothing_sigma_frames: float,
             palm_mean_clearance_m: float,
             object_rotation_mode: str = "fixed",
             seed: int = 0,
             ellipse_azimuth_deg: float = 0.0,
             ellipse_meridian_deg: float = 0.0,
             max_palm_surface_distance_m: float = 0.130,
             ellipse_section_fraction: float = -1.0,
             ellipse_plane_tilt_deg: float = 0.0,
             ellipse_arc_region: str = "calibrated",
             ellipse_arc_center_offset_deg: float = 0.0,
             palm_tangent_axis: str = "x",
             palm_tangent_sign: int = 1,
             max_plan_translation_path_m: float = 0.0,
             max_plan_rotation_path_deg: float = 0.0,
             max_plan_translation_step_m: float = 0.0,
             max_plan_rotation_step_deg: float = 0.0,
             time_parameterization: str = "arc_length",
             recovery_state: Path | None = None) -> None:
    if palm_tangent_sign not in (-1, 1):
        raise ValueError("palm_tangent_sign must be -1 or 1")
    if recovery_state is not None:
        _generate_from_recovery_state(
            recovery_state,
            output,
            frames,
            control_dt,
            max_plan_translation_path_m,
            max_plan_rotation_path_deg,
            max_plan_translation_step_m,
            max_plan_rotation_step_deg,
        )
        return
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
    planner_geometry_attrs: dict[str, np.ndarray | float | int | str] = {}
    if path_mode == "rigid_orbit":
        for i in range(frames):
            u = i / max(frames - 1, 1)
            smooth = u * u * (3.0 - 2.0 * u)
            angle = direction * np.deg2rad(angle_deg) * smooth
            delta = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
            palm_object[i] = rt_to_pose(delta @ palm_pos_obj0, delta @ palm_rot_obj0)
    elif path_mode == "minimum_enclosing_ellipse":
        config = load_object_config(object_id)
        oracle = MeshNormalOracle.from_config(config, scale=object_scale)
        if oracle is None:
            raise ValueError(
                "minimum_enclosing_ellipse planning requires a mesh object"
            )

        mesh_center = np.mean(oracle.vertices, axis=0)
        palm_center_obj0 = palm_pos_obj0 + palm_rot_obj0 @ PALM_CENTER_LOCAL
        section_attrs: dict[str, np.ndarray | float | int | str] = {}
        if 0.0 <= ellipse_section_fraction <= 1.0:
            # General object slice.  ``tilt=0`` is a transverse section
            # normal to the PCA long axis, ``tilt=90`` is a longitudinal
            # section, and intermediate values produce oblique sections.
            # For the current Mustard source, fraction 0 is the bottle bottom
            # and 1 is the cap end.
            centered = oracle.vertices - mesh_center
            _, _, vectors = np.linalg.svd(centered, full_matrices=False)
            long_axis = vectors[0]
            long_axis /= max(float(np.linalg.norm(long_axis)), 1.0e-12)
            dominant = int(np.argmax(np.abs(long_axis)))
            if long_axis[dominant] < 0.0:
                long_axis = -long_axis
            height = centered @ long_axis
            height_min = float(height.min())
            height_max = float(height.max())
            target_height = height_min + float(ellipse_section_fraction) * (
                height_max - height_min
            )
            section_origin = mesh_center + target_height * long_axis
            radial = palm_center_obj0 - section_origin
            radial -= long_axis * float(radial @ long_axis)
            if float(np.linalg.norm(radial)) < 1.0e-8:
                radial = np.cross(long_axis, np.array((1.0, 0.0, 0.0)))
                if float(np.linalg.norm(radial)) < 1.0e-8:
                    radial = np.cross(long_axis, np.array((0.0, 1.0, 0.0)))
            radial /= max(float(np.linalg.norm(radial)), 1.0e-12)
            if abs(float(ellipse_azimuth_deg)) > 1.0e-12:
                radial = R.from_rotvec(
                    long_axis * np.deg2rad(float(ellipse_azimuth_deg))
                ).apply(radial)
            tilt_rad = np.deg2rad(float(ellipse_plane_tilt_deg))
            plane_normal = R.from_rotvec(radial * tilt_rad).apply(long_axis)
            plane_normal /= max(float(np.linalg.norm(plane_normal)), 1.0e-12)
            second_axis = np.cross(plane_normal, radial)
            second_axis /= max(float(np.linalg.norm(second_axis)), 1.0e-12)
            plane_basis = np.column_stack((radial, second_axis))
            plane_distance = (oracle.vertices - section_origin) @ plane_normal
            half_width = max(0.010, 0.025 * (height_max - height_min))
            section_mask = np.abs(plane_distance) <= half_width
            while int(section_mask.sum()) < 64 and half_width < 0.20:
                half_width *= 1.5
                section_mask = np.abs(plane_distance) <= half_width
            section_vertices = oracle.vertices[section_mask]
            if len(section_vertices) < 3:
                raise ValueError(
                    "too few mesh vertices near requested ellipse section"
                )
            projected = (section_vertices - section_origin) @ plane_basis
            projection_origin = section_origin
            section_attrs = {
                "planner_ellipse_section_fraction": float(
                    ellipse_section_fraction
                ),
                "planner_ellipse_section_height_object_m": target_height,
                "planner_ellipse_section_half_width_m": half_width,
                "planner_ellipse_section_vertices": int(len(section_vertices)),
                "planner_object_long_axis": long_axis,
                "planner_ellipse_plane_tilt_deg": float(
                    ellipse_plane_tilt_deg
                ),
            }
        else:
            # Global longitudinal orbit defined by sampled object SO(3) and
            # the calibrated palm frame.
            centered_global = oracle.vertices - mesh_center
            _, _, global_vectors = np.linalg.svd(
                centered_global, full_matrices=False
            )
            long_axis_global = global_vectors[0]
            long_axis_global /= max(
                float(np.linalg.norm(long_axis_global)), 1.0e-12
            )
            dominant_global = int(np.argmax(np.abs(long_axis_global)))
            if long_axis_global[dominant_global] < 0.0:
                long_axis_global = -long_axis_global
            radial = palm_center_obj0 - mesh_center
            radial /= max(float(np.linalg.norm(radial)), 1.0e-12)
            if abs(float(ellipse_meridian_deg)) > 1.0e-12:
                # Meridian selection: rotate the orbit plane around the
                # object's PCA long axis so the cap-pole arc descends along
                # a different meridian.  (Legacy --ellipse-azimuth-deg
                # instead spins the palm plane around the view radial.)
                meridian_rot = R.from_rotvec(
                    long_axis_global * np.deg2rad(float(ellipse_meridian_deg))
                )
                radial = meridian_rot.apply(radial)
            palm_tangent = palm_rot_obj0 @ np.array((1.0, 0.0, 0.0))
            tangent = palm_tangent - radial * float(palm_tangent @ radial)
            if float(np.linalg.norm(tangent)) < 1.0e-8:
                palm_tangent = palm_rot_obj0 @ np.array((0.0, 1.0, 0.0))
                tangent = palm_tangent - radial * float(palm_tangent @ radial)
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
            if abs(float(ellipse_meridian_deg)) > 1.0e-12:
                tangent = meridian_rot.apply(tangent)
                tangent -= radial * float(tangent @ radial)
                tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
            if abs(float(ellipse_azimuth_deg)) > 1.0e-12:
                tangent = R.from_rotvec(
                    radial * np.deg2rad(float(ellipse_azimuth_deg))
                ).apply(tangent)
                tangent -= radial * float(tangent @ radial)
                tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
            plane_normal = np.cross(radial, tangent)
            plane_normal /= max(float(np.linalg.norm(plane_normal)), 1.0e-12)
            plane_basis = np.column_stack((radial, tangent))
            projected = (oracle.vertices - mesh_center) @ plane_basis
            projection_origin = mesh_center

        center_2d, ellipse_axes_2d, base_radii = minimum_enclosing_ellipse_2d(
            projected
        )
        ellipse_center = projection_origin + plane_basis @ center_2d
        ellipse_axis_a = plane_basis @ ellipse_axes_2d[:, 0]
        ellipse_axis_b = plane_basis @ ellipse_axes_2d[:, 1]
        ellipse_axis_a /= max(float(np.linalg.norm(ellipse_axis_a)), 1.0e-12)
        ellipse_axis_b /= max(float(np.linalg.norm(ellipse_axis_b)), 1.0e-12)

        if ellipse_arc_region in ("bottom", "cap"):
            # Centre the executed longitudinal arc on the requested bottle
            # end.  This is the same global ellipse used for cap trajectories,
            # not a transverse slice near the bottom.
            centered = oracle.vertices - mesh_center
            _, _, vectors = np.linalg.svd(centered, full_matrices=False)
            object_long_axis = vectors[0]
            object_long_axis /= max(
                float(np.linalg.norm(object_long_axis)), 1.0e-12
            )
            dominant = int(np.argmax(np.abs(object_long_axis)))
            if object_long_axis[dominant] < 0.0:
                object_long_axis = -object_long_axis
            phase_grid = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
            ellipse_grid = (
                ellipse_center[None, :]
                + base_radii[0] * np.cos(phase_grid)[:, None] * ellipse_axis_a
                + base_radii[1] * np.sin(phase_grid)[:, None] * ellipse_axis_b
            )
            longitudinal = (ellipse_grid - mesh_center) @ object_long_axis
            target_index = int(
                np.argmin(longitudinal)
                if ellipse_arc_region == "bottom"
                else np.argmax(longitudinal)
            )
            target_phase = float(phase_grid[target_index])
            center_phase = target_phase + np.deg2rad(
                float(ellipse_arc_center_offset_deg)
            )
            theta0 = center_phase - 0.5 * direction * np.deg2rad(angle_deg)
            section_attrs["planner_ellipse_arc_target_phase_rad"] = target_phase
            section_attrs["planner_ellipse_arc_center_phase_rad"] = center_phase
            section_attrs["planner_ellipse_arc_center_offset_deg"] = float(
                ellipse_arc_center_offset_deg
            )
            section_attrs["planner_object_long_axis"] = object_long_axis
        else:
            # Start from the ellipse point nearest the calibrated palm.
            palm_2d = (palm_center_obj0 - ellipse_center) @ np.column_stack(
                (ellipse_axis_a, ellipse_axis_b)
            )
            theta0 = float(
                np.arctan2(
                    palm_2d[1] / max(base_radii[1], 1.0e-12),
                    palm_2d[0] / max(base_radii[0], 1.0e-12),
                )
            )
        section_attrs["planner_ellipse_arc_region"] = ellipse_arc_region

        def ellipse_poses(
            theta: np.ndarray, axis_inflation: np.ndarray | float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            theta = np.asarray(theta, dtype=np.float64).reshape(-1)
            inflation_2d = np.broadcast_to(
                np.asarray(axis_inflation, dtype=np.float64), (2,)
            )
            radii = base_radii + inflation_2d
            cosine, sine = np.cos(theta)[:, None], np.sin(theta)[:, None]
            control_position = (
                ellipse_center[None, :]
                + radii[0] * cosine * ellipse_axis_a[None, :]
                + radii[1] * sine * ellipse_axis_b[None, :]
            )
            motion_tangent = direction * (
                -radii[0] * sine * ellipse_axis_a[None, :]
                + radii[1] * cosine * ellipse_axis_b[None, :]
            )
            _, nearest_face = oracle.face_tree.query(control_position)
            nearest_face = np.asarray(nearest_face, dtype=np.int64)
            surface_point = oracle.face_centroids[nearest_face]
            surface_normal = np.asarray(
                oracle.face_normals[nearest_face], dtype=np.float64
            ).copy()
            flip = np.einsum(
                "ij,ij->i", control_position - surface_point, surface_normal
            ) < 0.0
            surface_normal[flip] *= -1.0
            rotation = np.stack(
                [
                    palm_surface_tangent_rotation(
                        normal,
                        tangent,
                        palm_tangent_axis,
                        palm_tangent_sign,
                    )
                    for normal, tangent in zip(
                        surface_normal, motion_tangent, strict=True
                    )
                ],
                axis=0,
            )
            body_position = control_position - np.einsum(
                "nij,j->ni", rotation, PALM_CENTER_LOCAL
            )
            return body_position, rotation, control_position

        def ellipse_pose(
            theta: float, axis_inflation: np.ndarray | float
        ) -> tuple[np.ndarray, np.ndarray]:
            position, rotation, _ = ellipse_poses(
                np.asarray((theta,)), axis_inflation
            )
            return position[0], rotation[0]

        # Find independent major/minor-axis changes whose *executed arc*
        # satisfies the requested clearance.
        # The old implementation added the same offset to both axes.  On a
        # long thin object, a clearance required near the minor-axis side was
        # therefore also added to the already-long major axis and could put
        # the entire object outside fingertip reach.  Independent constant
        # axis offsets still produce one smooth analytic ellipse.  Because
        # only the executed arc must be collision-safe, the major axis may
        # also shrink below the global MVEE while the minor axis grows.  This
        # lowers eccentricity and avoids paying for unvisited mesh regions.
        executed_angles = np.linspace(
            theta0,
            theta0 + direction * np.deg2rad(angle_deg),
            181,
        )

        def ellipse_clearance_metrics(
            axis_inflation: np.ndarray | float,
        ) -> tuple[float, float]:
            position, rotation, control_position = ellipse_poses(
                executed_angles, axis_inflation
            )
            outline = position[:, None, :] + np.einsum(
                "nij,kj->nki", rotation, PALM_OUTLINE_LOCAL
            )
            flat_outline = outline.reshape(-1, 3)
            distance, _ = oracle.tree.query(flat_outline)
            _, nearest_face = oracle.face_tree.query(flat_outline)
            nearest_face = np.asarray(nearest_face, dtype=np.int64)
            face_center = oracle.face_centroids[nearest_face]
            face_normal = oracle.face_normals[nearest_face]
            sign = np.sign(
                np.einsum(
                    "ij,ij->i", flat_outline - face_center, face_normal
                )
            )
            signed = np.asarray(distance, dtype=np.float64) * np.where(
                sign == 0.0, 1.0, sign
            )
            center_distance = np.asarray(
                oracle.tree.query(control_position)[0], dtype=np.float64
            )
            return float(np.min(signed)), float(np.max(center_distance))

        # Keep a 1 mm numerical reserve for pose resampling/Slerp.  The file
        # still reports the requested clearance separately from this internal
        # planning target.
        planning_clearance = float(palm_outline_clearance_m) + 0.001
        lower, upper = 0.0, max(0.02, planning_clearance)
        while ellipse_clearance_metrics(upper)[0] < planning_clearance:
            upper *= 1.5
            if upper > 0.50:
                raise RuntimeError(
                    "could not construct a safe enclosing palm ellipse within "
                    "500 mm axis inflation"
                )
        for _ in range(24):
            middle = 0.5 * (lower + upper)
            if ellipse_clearance_metrics(middle)[0] >= planning_clearance:
                upper = middle
            else:
                lower = middle

        # The common inflation above is a guaranteed feasible upper bound.
        # Sweep the major-axis change (including up to 30% shrinkage) and
        # binary-search the least feasible minor-axis inflation.  Select the
        # feasible ellipse with the lowest
        # worst-case palm-centre/surface distance; total inflation breaks
        # near ties.  This directly favours fingertip reach without weakening
        # the complete-palm collision clearance.
        candidates: list[tuple[float, float, np.ndarray]] = []
        common = np.full(2, upper, dtype=np.float64)
        common_clearance, common_reach = ellipse_clearance_metrics(common)
        candidates.append((common_reach, float(common.sum()), common))
        minimum_major_radius = max(
            0.70 * base_radii[0], 1.05 * base_radii[1]
        )
        minimum_major_change = minimum_major_radius - base_radii[0]
        for inflation_a in np.linspace(minimum_major_change, upper, 41):
            high = upper
            trial = np.asarray((inflation_a, high), dtype=np.float64)
            while (
                ellipse_clearance_metrics(trial)[0] < planning_clearance
                and high < 0.50
            ):
                high = min(0.50, max(high * 1.5, high + 0.01))
                trial[1] = high
            if ellipse_clearance_metrics(trial)[0] < planning_clearance:
                continue
            low = 0.0
            for _ in range(16):
                middle = 0.5 * (low + high)
                trial[1] = middle
                if ellipse_clearance_metrics(trial)[0] >= planning_clearance:
                    high = middle
                else:
                    low = middle
            trial = np.asarray((inflation_a, high), dtype=np.float64)
            clearance, reach = ellipse_clearance_metrics(trial)
            if clearance >= planning_clearance - 1.0e-6:
                candidates.append((reach, float(trial.sum()), trial.copy()))
        if not candidates:
            raise RuntimeError("no safe independently inflated palm ellipse found")
        best_reach = min(item[0] for item in candidates)
        # Treat sub-millimetre reach differences as equivalent, then prefer
        # the smaller ellipse to avoid numerical over-expansion.
        near_best = [item for item in candidates if item[0] <= best_reach + 0.001]
        max_center_surface_distance, _, inflation = min(
            near_best, key=lambda item: item[1]
        )
        final_radii = base_radii + inflation
        if (
            max_palm_surface_distance_m > 0.0
            and max_center_surface_distance > max_palm_surface_distance_m
        ):
            raise ValueError(
                "unreachable enclosing ellipse: worst palm-centre/surface "
                f"distance={max_center_surface_distance * 1000.0:.1f} mm > "
                f"limit={max_palm_surface_distance_m * 1000.0:.1f} mm; "
                "reject this SO(3) orientation and resample"
            )

        # Generate the FSR-centre route directly from the analytic ellipse.
        # Never recover it from a body pose whose old triangle-normal frame
        # may contain angular jumps.  Reparameterize the dense analytic curve
        # by arc length with a cubic B-spline, then use short cosine velocity
        # ramps at both ends and constant speed through the middle.
        dense_count = max(2000, 4 * frames)
        dense_theta = np.linspace(
            theta0,
            theta0 + direction * np.deg2rad(angle_deg),
            dense_count,
        )
        dense_cosine = np.cos(dense_theta)[:, None]
        dense_sine = np.sin(dense_theta)[:, None]
        dense_control = (
            ellipse_center[None, :]
            + final_radii[0] * dense_cosine * ellipse_axis_a[None, :]
            + final_radii[1] * dense_sine * ellipse_axis_b[None, :]
        )
        dense_segment = np.linalg.norm(np.diff(dense_control, axis=0), axis=1)
        dense_arc = np.concatenate(([0.0], np.cumsum(dense_segment)))
        keep = np.concatenate(([True], dense_segment > 1.0e-12))
        dense_arc = dense_arc[keep]
        dense_control = dense_control[keep]
        route_spline = make_interp_spline(
            dense_arc,
            dense_control,
            k=min(3, len(dense_arc) - 1),
            axis=0,
        )
        time_u = np.linspace(0.0, 1.0, frames)
        ramp_fraction = 0.10
        velocity = np.ones(frames, dtype=np.float64)
        start = time_u < ramp_fraction
        end = time_u > 1.0 - ramp_fraction
        velocity[start] = 0.5 * (
            1.0 - np.cos(np.pi * time_u[start] / ramp_fraction)
        )
        velocity[end] = 0.5 * (
            1.0
            - np.cos(np.pi * (1.0 - time_u[end]) / ramp_fraction)
        )
        progress = np.zeros(frames, dtype=np.float64)
        if frames > 1:
            progress[1:] = np.cumsum(
                0.5 * (velocity[:-1] + velocity[1:]) * np.diff(time_u)
            )
        progress /= max(float(progress[-1]), 1.0e-12)
        resampled_control_position = np.asarray(
            route_spline(progress * dense_arc[-1]), dtype=np.float64
        )
        # About 25 normal anchors independent of trajectory duration.  The
        # source mesh determines those anchors; the B-spline
        # only fills the physically realizable continuous wrist trajectory
        # between them.
        spline_anchor_spacing = max(
            4.0, float(frames - 1) / 24.0
        )
        resampled_position, resampled_rotation, normal_anchor_count = (
            smooth_surface_aligned_poses(
                resampled_control_position,
                oracle,
                palm_tangent_axis,
                spline_anchor_spacing,
                palm_tangent_sign,
            )
        )
        # B-spline wrist orientation changes the palm outline even though the
        # FSR-centre route is unchanged.  Restore the requested safety reserve
        # with one smooth global offset along the interpolated surface normal,
        # rather than pointwise corrections that would reintroduce jitter.
        smooth_outward_padding = 0.0
        for _ in range(4):
            outline = resampled_position[:, None, :] + np.einsum(
                "nij,kj->nki", resampled_rotation, PALM_OUTLINE_LOCAL
            )
            flat_outline = outline.reshape(-1, 3)
            _, nearest_vertex = oracle.tree.query(flat_outline)
            surface = oracle.vertices[
                np.asarray(nearest_vertex, dtype=np.int64)
            ]
            guard_normals = oracle.query_object_frame(surface)
            signed = np.einsum(
                "ij,ij->i", flat_outline - surface, guard_normals
            )
            deficit = planning_clearance - float(np.min(signed))
            if deficit <= 1.0e-6:
                break
            padding = deficit + 0.00025
            smooth_outward_padding += padding
            resampled_control_position += (
                padding * resampled_rotation[:, :, 2]
            )
            resampled_position, resampled_rotation, normal_anchor_count = (
                smooth_surface_aligned_poses(
                    resampled_control_position,
                    oracle,
                    palm_tangent_axis,
                    spline_anchor_spacing,
                    palm_tangent_sign,
                )
            )
        for i in range(frames):
            palm_object[i] = rt_to_pose(
                resampled_position[i], resampled_rotation[i]
            )
        if time_parameterization == "rotation":
            palm_object = resample_pose_uniform_rotation(palm_object, frames)
        elif time_parameterization == "blend":
            palm_object = resample_pose_uniform_blend(
                palm_object, frames, rotation_weight=0.5, ramp_fraction=0.10
            )
        else:
            palm_object = resample_pose_uniform_arc_length(
                palm_object, frames, ramp_fraction=0.10
            )
        resampled_position, resampled_rotation = pose_to_rt(palm_object)
        for i in range(frames):
            mean_clearance, min_clearance, _ = palm_outline_clearance_stats(
                resampled_position[i], resampled_rotation[i], oracle
            )
            outline_mean_clearance[i] = mean_clearance
            outline_clearance[i] = min_clearance
        planner_geometry_attrs = {
            "planner_ellipse_center_object": ellipse_center,
            "planner_ellipse_plane_normal_object": plane_normal,
            "planner_ellipse_axis_a_object": ellipse_axis_a,
            "planner_ellipse_axis_b_object": ellipse_axis_b,
            "planner_ellipse_base_radii_m": base_radii,
            "planner_ellipse_safe_radii_m": final_radii,
            "planner_ellipse_axis_inflation_m": inflation,
            "planner_ellipse_max_center_surface_distance_m": float(
                max_center_surface_distance
            ),
            "planner_ellipse_internal_clearance_target_m": planning_clearance,
            "planner_ellipse_start_phase_rad": float(theta0),
            "planner_projected_mesh_vertices": int(len(projected)),
            "planner_palm_control_point_local": PALM_CENTER_LOCAL,
            "planner_palm_tangent_axis": palm_tangent_axis,
            "planner_palm_tangent_sign": int(palm_tangent_sign),
            "planner_palm_normal_source": "source_mesh_local_pca_bspline",
            "planner_palm_normal_spline_anchors": int(normal_anchor_count),
            "planner_palm_normal_spline_anchor_spacing_frames": float(
                spline_anchor_spacing
            ),
            "planner_smooth_outward_padding_m": float(smooth_outward_padding),
            "planner_time_parameterization": "arc_bspline_cosine_ramp_10pct",
            **section_attrs,
        }
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
        # The ellipse tangent defines the translational route, but do not
        # impose a hard wrist-roll constraint.  Preserve the initial palm
        # roll and only turn the palm's facing axis toward the surface.  This
        # keeps the intended ``palm toward object`` behavior while avoiding a
        # brittle tangent-frame orientation that can twist the wrist on thin
        # or asymmetric objects.
        palm_facing0 = palm_rot_obj0 @ np.array((0.0, 0.0, -1.0))
        normal_sign = 1.0 if float(palm_facing0 @ radial_ref) >= 0.0 else -1.0
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
            desired_facing = normal_sign * normal
            ellipse_normal[i] = desired_facing
            rotations[i] = (
                align_vector(palm_facing0, desired_facing) @ palm_rot_obj0
            )

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

        if time_parameterization == "rotation":
            palm_object = resample_pose_uniform_rotation(palm_object, frames)
        elif time_parameterization == "blend":
            palm_object = resample_pose_uniform_blend(
                palm_object, frames, rotation_weight=0.5, ramp_fraction=0.10
            )
        else:
            palm_object = resample_pose_uniform_arc_length(
                palm_object, frames, ramp_fraction=0.10
            )
        # Report clearances after resampling as these are the poses consumed
        # by the inverse collector.
        resampled_position, resampled_rotation = pose_to_rt(palm_object)
        for i in range(frames):
            mean_clearance, min_clearance, _ = palm_outline_clearance_stats(
                resampled_position[i], resampled_rotation[i], oracle
            )
    else:
        raise ValueError(
            f"unsupported path_mode {path_mode!r}: only rigid_orbit,"
            " minimum_enclosing_ellipse and ellipse_clearance remain"
        )

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

    motion_metrics = palm_plan_motion_metrics(palm_object)
    limits = (
        ("translation_path_m", max_plan_translation_path_m),
        ("rotation_path_deg", max_plan_rotation_path_deg),
        ("max_translation_step_m", max_plan_translation_step_m),
        ("max_rotation_step_deg", max_plan_rotation_step_deg),
    )
    violations = [
        f"{name}={motion_metrics[name]:.6g} > {float(limit):.6g}"
        for name, limit in limits
        if float(limit) > 0.0 and motion_metrics[name] > float(limit)
    ]
    if violations:
        raise ValueError(
            "rejecting kinematically irregular palm plan: "
            + "; ".join(violations)
        )

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
        # Store the actual sampled pose, not only the RNG seed.  This makes
        # SO(3) orientation sweeps auditable and prevents confusing the
        # object's initial orientation with path coverage or ellipse azimuth.
        dst.attrs["planner_object_initial_pose_world"] = object0
        dst.attrs["planner_ellipse_azimuth_deg"] = float(ellipse_azimuth_deg)
        dst.attrs["planner_ellipse_meridian_deg"] = float(
            ellipse_meridian_deg
        )
        dst.attrs["planner_max_palm_surface_distance_m"] = float(
            max_palm_surface_distance_m
        )
        for name, value in motion_metrics.items():
            dst.attrs[f"planner_realized_{name}"] = float(value)
        for key, value in planner_geometry_attrs.items():
            dst.attrs[key] = value
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
        choices=(
            "rigid_orbit",
            "ellipse_clearance",
            "minimum_enclosing_ellipse",
        ),
    )
    parser.add_argument(
        "--path-length-m", type=float, default=0.0,
        help="Signed latitude travel along the object's PCA long axis.",
    )
    parser.add_argument(
        "--palm-outline-clearance-m",
        type=float,
        default=0.030,
        help="Minimum mesh clearance for every palm-outline sample (default: 30 mm).",
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
        "--ellipse-meridian-deg",
        type=float,
        default=0.0,
        help=(
            "Rotate the global longitudinal orbit plane around the PCA long "
            "axis so the cap-pole arc descends along a selected meridian. "
            "Unlike --ellipse-azimuth-deg (which spins the palm plane around "
            "the view radial), the cap extremum stays at the cap pole and "
            "the arc endpoint remains on the cap top (default: 0)."
        ),
    )
    parser.add_argument(
        "--max-palm-surface-distance-m",
        type=float,
        default=0.130,
        help=(
            "Reject a minimum-enclosing ellipse when the palm centre is "
            "farther than this from the nearest surface anywhere on the "
            "executed arc; <=0 disables the reachability gate "
            "(default: 130 mm)."
        ),
    )
    parser.add_argument(
        "--ellipse-section-fraction",
        type=float,
        default=-1.0,
        help=(
            "Fit a regional ellipse to a mesh cross-section normal to its "
            "PCA long axis: 0 selects one end (Mustard bottom), 1 the cap; "
            "negative keeps the global longitudinal ellipse."
        ),
    )
    parser.add_argument(
        "--ellipse-plane-tilt-deg",
        type=float,
        default=0.0,
        help=(
            "Slice-plane tilt around its radial axis: 0 is transverse, "
            "90 is longitudinal, and intermediate values are oblique "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--ellipse-arc-region",
        choices=("calibrated", "bottom", "cap"),
        default="calibrated",
        help=(
            "For a global longitudinal ellipse, select the calibrated arc "
            "or centre the executed arc on the Mustard bottom/cap."
        ),
    )
    parser.add_argument(
        "--ellipse-arc-center-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Shift the executed arc centre relative to the selected bottom/cap "
            "extremum. For an arc of A degrees, offset=-direction*A/2 makes "
            "the trajectory end at that extremum instead of crossing it."
        ),
    )
    parser.add_argument(
        "--palm-tangent-axis",
        choices=("x", "y"),
        default="x",
        help=(
            "Palm positive axis aligned with the direction of travel; palm "
            "-Z always faces the nearest source-mesh surface (default: x)."
        ),
    )
    parser.add_argument(
        "--palm-tangent-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help=(
            "Select which end of the chosen palm tangent axis faces the "
            "direction of travel. -1 rotates the palm 180 degrees around "
            "its surface-facing normal so the wrist leads and fingers trail."
        ),
    )
    parser.add_argument(
        "--max-plan-translation-path-m", type=float, default=0.0,
        help="Reject plans whose realized palm translation path exceeds this; <=0 disables.",
    )
    parser.add_argument(
        "--max-plan-rotation-path-deg", type=float, default=0.0,
        help="Reject plans whose accumulated palm rotation exceeds this; <=0 disables.",
    )
    parser.add_argument(
        "--max-plan-translation-step-m", type=float, default=0.0,
        help="Reject plans with a larger one-frame palm translation; <=0 disables.",
    )
    parser.add_argument(
        "--max-plan-rotation-step-deg", type=float, default=0.0,
        help="Reject plans with a larger one-frame palm rotation; <=0 disables.",
    )
    parser.add_argument(
        "--recovery-state",
        type=Path,
        default=None,
        help=(
            "extract_recovery_state.py output (state A). When given, the "
            "palm plan is the recorded continuation of the failed rollout's "
            "own source episode past state A instead of a fresh analytic "
            "path; --source/--episode-id/--object-id/--path-mode and the "
            "ellipse/orbit options are all ignored."
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
    parser.add_argument(
        "--time-parameterization",
        choices=("arc_length", "rotation", "blend"),
        default="arc_length",
        help=(
            "arc_length (default): uniform palm position speed. rotation: "
            "uniform equivalent object angular speed (reparameterize by "
            "cumulative orientation distance) -- the executed rotation on an "
            "oblique/longitudinal ellipse otherwise speeds up and slows down "
            "several-fold, and the fast segments shed contact."
        ),
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
        args.ellipse_meridian_deg,
        args.max_palm_surface_distance_m, args.ellipse_section_fraction,
        args.ellipse_plane_tilt_deg,
        args.ellipse_arc_region,
        args.ellipse_arc_center_offset_deg,
        args.palm_tangent_axis,
        args.palm_tangent_sign,
        args.max_plan_translation_path_m,
        args.max_plan_rotation_path_deg,
        args.max_plan_translation_step_m,
        args.max_plan_rotation_step_deg,
        args.time_parameterization,
        args.recovery_state,
    )


if __name__ == "__main__":
    main()
