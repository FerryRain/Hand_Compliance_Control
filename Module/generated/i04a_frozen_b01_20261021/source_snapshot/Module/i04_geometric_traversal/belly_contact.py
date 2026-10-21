"""Anatomical belly-face geometry for the unchanged registered ellipsoid pads.

The geom's +Z axis is the distal body's -X (palmar) axis, as registered by
``PAD_LOCAL_ROTATION``. It is NOT the direction from a finger to the palm center.
An object's outward normal must oppose this axis. A collision involving the
pad geom alone does not establish belly contact: its rim and back also collide.

These helpers only read FK/contacts and supply geometric IK terms. They neither
step physics nor authorize contact, force safety, recovery, or coverage. The
thresholds below are explicit proposed planning/classification parameters, not
changes to the physical 12 x 8 x 2 mm ellipsoid or to force/collision limits.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from Module.e05_physics.scene import PAD_LOCAL_ROTATION
from .finger_reconfiguration_planner import ellipsoid_support_derivative


@dataclass(frozen=True)
class BellyContactConfig:
    maximum_alignment_angle_deg: float = 30.
    maximum_patch_radius: float = .8
    surface_tolerance_m: float = .0001

    def __post_init__(self):
        if not np.isfinite(self.maximum_alignment_angle_deg) or not 0. < self.maximum_alignment_angle_deg < 90.:
            raise ValueError('Belly alignment angle must be finite and strictly between 0 and 90 degrees')
        if not np.isfinite(self.maximum_patch_radius) or not 0. < self.maximum_patch_radius < 1.:
            raise ValueError('The central belly patch must have a finite normalized radius strictly between 0 and 1')
        if not np.isfinite(self.surface_tolerance_m) or self.surface_tolerance_m < 0.:
            raise ValueError('Surface tolerance must be finite and nonnegative')


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite 3-vector')
    return value


def _normal(value):
    value = _vector(value, 'Object outward normal')
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        raise ValueError('Object outward normal must be nonzero')
    return value / length


def _frame(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-8, rtol=0.)
            or abs(np.linalg.det(value)-1.) > 1e-8):
        raise ValueError('Pad frame must be a finite proper orthonormal rotation')
    return value


def _skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def registered_belly_frame(model, data, geom_id):
    """Read the registered pad frame, rejecting unrelated or reoriented geoms."""
    geom = int(geom_id)
    if not 0 <= geom < model.ngeom or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        raise ValueError('Belly geometry requires a registered ellipsoid pad')
    local = np.empty(9)
    mujoco.mju_quat2Mat(local, model.geom_quat[geom])
    if not np.allclose(local.reshape(3, 3), PAD_LOCAL_ROTATION, atol=1e-8, rtol=0.):
        raise ValueError('Geom orientation does not match the anatomical PAD_LOCAL_ROTATION')
    sizes = np.asarray(model.geom_size[geom], float)
    if not np.isfinite(sizes).all() or np.any(sizes <= 0.):
        raise ValueError('Pad half sizes must be finite and positive')
    return _frame(data.geom_xmat[geom].reshape(3, 3)).copy()


def belly_alignment_terms(frame, object_outward_normal, angular_jacobian=None):
    """Return dimensionless chord residual b+n and its world-angular Jacobian.

    The desired anatomical belly axis is -n. Unlike b cross n, this residual
    is nonzero at the wrong, antipodal orientation. An exactly antipodal start
    can still be stationary for local least squares; final angle/patch checks
    are mandatory, and a solver termination flag alone must never accept it.
    Twist about the belly axis is intentionally unconstrained.
    """
    frame, normal = _frame(frame), _normal(object_outward_normal)
    belly = frame[:, 2]
    residual = belly + normal
    jacobian = None
    if angular_jacobian is not None:
        angular = np.asarray(angular_jacobian, dtype=float)
        if angular.ndim != 2 or angular.shape[0] != 3 or not np.isfinite(angular).all():
            raise ValueError('Angular Jacobian must be finite with shape (3, N)')
        jacobian = -_skew(belly) @ angular
    cosine = float(np.clip(belly @ -normal, -1., 1.))
    return residual, jacobian, dict(belly_axis_world=belly.tolist(),
        desired_belly_axis_world=(-normal).tolist(), alignment_cosine=cosine,
        alignment_angle_deg=float(np.rad2deg(np.arccos(cosine))))


def classify_belly_patch(center, frame, half_sizes, point_world, object_outward_normal, config=None):
    """Classify a supplied point; this function does not establish real contact.

    Radially project the supplied point onto the unchanged ellipsoid solely to
    identify its patch. The unprojected radial shell error is checked separately
    so projection cannot hide a center or a point beyond the shell tolerance.
    Patch radius is
    sqrt((x/a)^2+(y/b)^2) on that shell: .8 means the central 80% semiaxes, NOT an
    80-degree cone. On a thin pad it is stricter than the axis-angle test.
    """
    cfg = config or BellyContactConfig()
    center, point = _vector(center, 'Pad center'), _vector(point_world, 'Contact point')
    frame, sizes = _frame(frame), _vector(half_sizes, 'Pad half sizes')
    if np.any(sizes <= 0.):
        raise ValueError('Pad half sizes must be positive')
    _, _, alignment = belly_alignment_terms(frame, object_outward_normal)
    local = frame.T @ (point-center)
    normalized = local / sizes
    radius = float(np.linalg.norm(normalized))
    reasons = []
    if radius <= 1e-12:
        return dict(alignment, geometry_qualified=False, rejection_reasons=['POINT_AT_PAD_CENTER'],
                    contact_local_m=local.tolist(), normalized_ellipsoid_radius=radius,
                    projected_patch_radius=None, radial_shell_error_m=None,
                    authority='GEOMETRY_ONLY_NOT_CONTACT_OR_COVERAGE')
    shell = local / radius
    patch_radius = float(np.linalg.norm(shell[:2] / sizes[:2]))
    shell_error = float(np.linalg.norm(local-shell))
    if alignment['alignment_angle_deg'] > cfg.maximum_alignment_angle_deg + 1e-10:
        reasons.append('BELLY_AXIS_MISALIGNED')
    if shell[2] <= 0.:
        reasons.append('NOT_PALMAR_HEMISPHERE')
    if patch_radius > cfg.maximum_patch_radius + 1e-12:
        reasons.append('OUTSIDE_CENTRAL_BELLY_PATCH')
    if shell_error > cfg.surface_tolerance_m + 1e-12:
        reasons.append('POINT_NOT_ON_PAD_SURFACE')
    local_surface_normal = shell / (sizes*sizes)
    local_surface_normal /= np.linalg.norm(local_surface_normal)
    surface_normal = frame @ local_surface_normal
    contact_normal_cosine = float(surface_normal @ -_normal(object_outward_normal))
    if contact_normal_cosine <= 0.:
        reasons.append('CONTACT_NORMAL_OPPOSES_PAD_SURFACE_NORMAL')
    return dict(alignment, geometry_qualified=not reasons, rejection_reasons=reasons,
                contact_local_m=local.tolist(), normalized_ellipsoid_radius=radius,
                projected_contact_local_m=shell.tolist(), projected_patch_radius=patch_radius,
                radial_shell_error_m=shell_error,
                ellipsoid_outward_normal_world=surface_normal.tolist(),
                contact_normal_cosine=contact_normal_cosine,
                authority='GEOMETRY_ONLY_NOT_CONTACT_OR_COVERAGE')


def evaluate_belly_contact(model, data, geom_id, point_world, object_outward_normal, config=None):
    """Classify an observed or proposed point using current native pad FK."""
    frame = registered_belly_frame(model, data, geom_id)
    return classify_belly_patch(data.geom_xpos[geom_id], frame, model.geom_size[geom_id],
                               point_world, object_outward_normal, config)


def belly_ik_terms(model, data, geom_id, surface_point_world, object_outward_normal,
                   *, clearance_m=0., orientation_weight_m=.01, dof_indices=None, config=None):
    """Six residuals/Jacobian rows: support position and weighted belly direction.

    ``data`` must already contain FK at the candidate q. This does not update q,
    FK, model parameters or controls. The normal/mesh point are held fixed when
    differentiating. Position is c-support(n), against p+clearance*n; the
    orientation weight converts the dimensionless direction error into meters.
    Returned diagnostics classify the actual geometric support point. Acceptance
    also needs the caller's position tolerance, collision/path and native guards.
    """
    if not np.isfinite(clearance_m) or not np.isfinite(orientation_weight_m) or orientation_weight_m <= 0.:
        raise ValueError('Clearance must be finite and orientation weight finite and positive')
    geom = int(geom_id)
    frame = registered_belly_frame(model, data, geom)
    normal = _normal(object_outward_normal)
    target = _vector(surface_point_world, 'Surface point') + clearance_m*normal
    sizes = np.asarray(model.geom_size[geom])
    local_normal = frame.T @ normal
    support = frame @ (sizes*sizes*local_normal / np.linalg.norm(sizes*local_normal))
    point = data.geom_xpos[geom]-support
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacGeom(model, data, jp, jr, geom)
    if dof_indices is not None:
        indices = np.asarray(dof_indices)
        if (indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= model.nv)):
            raise ValueError('DOF indices must be a one-dimensional integer array inside model.nv')
        jp, jr = jp[:, indices], jr[:, indices]
    position_jac = jp - ellipsoid_support_derivative(frame, sizes, normal, jr)
    orientation_residual, orientation_jac, _ = belly_alignment_terms(frame, normal, jr)
    residual = np.r_[point-target, orientation_weight_m*orientation_residual]
    jacobian = np.vstack([position_jac, orientation_weight_m*orientation_jac])
    diagnostic = classify_belly_patch(data.geom_xpos[geom], frame, sizes, point, normal, config)
    diagnostic.update(support_point_world_m=point.tolist(), target_support_world_m=target.tolist(),
                      position_residual_m=float(np.linalg.norm(residual[:3])),
                      orientation_weight_m=float(orientation_weight_m))
    return residual, jacobian, diagnostic


def measured_belly_contacts(handles, data, config=None):
    """Per-contact native evidence, without averaging away a rim/back contact.

    Returns records only, with unchanged mj_contactForce normal force. The caller
    retains the existing contact/force thresholds and must choose how to handle
    mixed valid/invalid contacts. No active flags or coverage masks are rewritten.
    MuJoCo contact.pos is the witness midpoint; signed dist reconstructs the pad
    witness. Both are reported. This is not a penetration/force safety check.
    """
    model = handles.model
    objects = set(map(int, getattr(handles, 'object_geom_ids', [handles.object_geom_id])))
    fingers = {int(g): i for i, g in enumerate(handles.tip_geom_ids)}
    records, wrench = [], np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        a, b = int(contact.geom1), int(contact.geom2)
        if a in objects and b in fingers:
            geom, normal = b, np.asarray(contact.frame[:3]).copy()
        elif b in objects and a in fingers:
            geom, normal = a, -np.asarray(contact.frame[:3]).copy()
        else:
            continue
        normal = _normal(normal)
        point = np.asarray(contact.pos) + .5*float(contact.dist)*normal
        mujoco.mj_contactForce(model, data, index, wrench)
        diagnostic = evaluate_belly_contact(model, data, geom, point, normal, config)
        records.append(dict(diagnostic, contact_index=index, finger=fingers[geom], geom_id=geom,
                            normal_force_n=max(0., float(wrench[0])), signed_distance_m=float(contact.dist),
                            midpoint_world_m=np.asarray(contact.pos).tolist(), pad_witness_world_m=point.tolist(),
                            object_outward_normal_world=normal.tolist(), authority='MEASURED_NATIVE_CONTACT_GEOMETRY'))
    return records


def belly_contact_measurement(handles, data, threshold=.2, config=None):
    """Filter each real contact before force-weighted aggregation, never after.

    Returns ``forces, points, normals, active, diag``. Points retain the original
    MuJoCo midpoint convention used by ``physics.contact_measurement``; the patch
    classifier uses reconstructed pad witnesses. All raw force and each rejected
    contact remain in diag. The caller MUST still use the original unfiltered
    measurement for high-force, non-tip, penetration and self-collision guards.
    A mixed contact can have qualified force, but is explicitly marked mixed.
    """
    if not np.isfinite(threshold) or threshold <= 0.:
        raise ValueError('Contact threshold must be finite and positive')
    count = len(handles.tip_geom_ids)
    forces, raw, rejected = np.zeros(count), np.zeros(count), np.zeros(count)
    points, normals = np.zeros((count, 3)), np.zeros((count, 3))
    active = np.zeros(count, dtype=bool)
    records = measured_belly_contacts(handles, data, config)
    for record in records:
        finger, force = record['finger'], record['normal_force_n']
        raw[finger] += force
        if not record['geometry_qualified']:
            rejected[finger] += force
            continue
        forces[finger] += force
        points[finger] += force*np.asarray(record['midpoint_world_m'])
        normals[finger] += force*np.asarray(record['object_outward_normal_world'])
    for finger in range(count):
        normal_length = float(np.linalg.norm(normals[finger]))
        if forces[finger] >= threshold and normal_length > 1e-10:
            points[finger] /= forces[finger]
            normals[finger] /= normal_length
            active[finger] = True
        else:
            points[finger] = 0.
            normals[finger] = 0.
    diagnostic = dict(records=records, raw_contact_force_n=raw.tolist(),
                      rejected_contact_force_n=rejected.tolist(),
                      mixed_qualified_and_rejected=(np.logical_and(forces > 0., rejected > 0.)).tolist(),
                      authority='FILTERED_MEASURED_BELLY_CONTACTS_NOT_RAW_SAFETY_FORCE',
                      coverage_modified=False, model_or_state_modified=False)
    return forces, points, normals, active, diagnostic
