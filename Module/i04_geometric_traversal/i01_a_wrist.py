"""Geometry-only wrist reference and bounded tactile feedback for i01-a."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, RotationSpline

from .fingertip_hand import DT


# Object-frame XY offsets of the grasp footprint on the upper bottle face.
# The first turn approaches the shoulder, and subsequent passes traverse the body.
SCAN_OFFSETS_M = np.array([
    [0., 0.], [.004, -.021], [.018, -.036], [.034, -.026],
    [.048, .014], [.047, .077], [.035, .135], [.011, .166],
    [-.013, .144], [-.019, .086], [-.017, .027], [-.007, -.005],
])
STARTUP_S = .9
SETTLE_S = .15
GRID_M = .003
KNOT_M = .003
HEIGHT_SMOOTH_M = .005
PROXY_TOLERANCE_M = .007
MAX_SPEED_M_S = .016
MAX_ACCEL_M_S2 = .045
MOTION_STD_WIDTH = 2.5
UPDATE_STEPS = round(.1 / DT)


def wrist_digest(poses):
    return sha256(np.ascontiguousarray(poses, dtype='<f8').tobytes()).hexdigest()


def as_wxyz(rotations):
    return rotations.as_quat()[..., [3, 0, 1, 2]]


class UpperSurface:
    """Sample and smooth the known surface on the side of the initial grasp."""

    def __init__(self, mesh, bounds):
        axes = tuple(np.arange(lo, hi + GRID_M, GRID_M) for lo, hi in zip(*bounds))
        xx, yy = np.meshgrid(*axes, indexing='ij')
        origins = np.column_stack((xx.ravel(), yy.ravel(),
                                   np.full(xx.size, mesh.bounds[1, 2] + .1)))
        hits, indices, _ = mesh.ray.intersects_location(
            origins, np.broadcast_to((0., 0., -1.), origins.shape), multiple_hits=True)
        top = np.full(len(origins), -np.inf)
        np.maximum.at(top, indices, hits[:, 2])
        top = top.reshape(xx.shape)
        valid = np.isfinite(top)
        if not valid.any():
            raise ValueError('Wrist route has no support on the object mesh')
        width = HEIGHT_SMOOTH_M / GRID_M
        smooth_z = gaussian_filter(np.where(valid, top, 0.), width)
        smooth_z /= np.maximum(gaussian_filter(valid.astype(float), width), 1.e-10)
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        self.height = RegularGridInterpolator(axes, smooth_z[tuple(nearest)],
                                              bounds_error=False, fill_value=None)
        self.visible = RegularGridInterpolator(axes, valid.astype(float), method='nearest',
                                               bounds_error=False, fill_value=0.)


@dataclass
class WristTape:
    poses: np.ndarray
    arc_m: np.ndarray
    scan_end: int
    footprint_in_wrist: np.ndarray
    points_in_wrist: np.ndarray
    geometry: dict


def time_parameterize(arc, position, orientation, calibration, *, max_speed=MAX_SPEED_M_S):
    rot = orientation(arc)
    linear = rot.inv().apply(position(arc, 1))
    secant = (rot[:-1].inv() * rot[1:]).as_rotvec() / np.diff(arc)[:, None]
    angular = np.vstack((secant[0], .5 * (secant[:-1] + secant[1:]), secant[-1]))
    derivative = np.column_stack((linear, angular))
    mean, std = np.asarray(calibration['mean']), np.asarray(calibration['std'])
    # One bound for both the measured wrist twist and the 200 ms relative motion.
    lower = np.minimum(mean - MOTION_STD_WIDTH * std, 0.)
    upper = np.maximum(mean + MOTION_STD_WIDTH * std, 0.)
    negative = np.minimum(-lower[:6], -lower[6:] / .2)
    positive = np.minimum(upper[:6], upper[6:] / .2)
    budget = np.where(derivative >= 0., positive, negative)
    ratio = np.divide(budget, abs(derivative), out=np.full_like(derivative, np.inf),
                      where=abs(derivative) > 1.e-9)
    speed = np.minimum(max_speed, .8 * ratio.min(axis=1))
    speed[[0, -1]] = 0.
    s2 = speed**2
    limit = 2. * MAX_ACCEL_M_S2 * np.diff(arc)
    for idx in range(1, len(s2)):
        s2[idx] = min(s2[idx], s2[idx - 1] + limit[idx - 1])
    for idx in range(len(s2) - 2, -1, -1):
        s2[idx] = min(s2[idx], s2[idx + 1] + limit[idx])
    speed = np.sqrt(s2)
    if np.any(speed[1:-1] <= 0.):
        raise ValueError('Wrist route requires motion outside the calibrated directions')
    times = np.r_[0., np.cumsum(2. * np.diff(arc) / (speed[1:] + speed[:-1]))]
    return times, speed


def plan_wrist(mesh, start_position, start_rotation, initial_points, calibration):
    """Fit one rigid grasp footprint by moving only the wrist's height and tilt."""
    center = initial_points.mean(axis=0)
    footprint = start_rotation.inv().apply(center - start_position)
    offsets = start_rotation.inv().apply(initial_points - center)
    query_bounds = (initial_points[:, :2].min(axis=0) + SCAN_OFFSETS_M.min(axis=0) - .04,
                    initial_points[:, :2].max(axis=0) + SCAN_OFFSETS_M.max(axis=0) + .04)
    surface = UpperSurface(mesh, query_bounds)
    initial_gap = initial_points[:, 2] - surface.height(initial_points[:, :2])
    control_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(SCAN_OFFSETS_M, axis=0), axis=1))]
    translated = CubicSpline(control_s, SCAN_OFFSETS_M, bc_type='natural')(
        np.linspace(0., control_s[-1], int(np.ceil(control_s[-1] / KNOT_M)) + 1))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(translated, axis=0), axis=1))]
    centers = np.broadcast_to(center, (len(arc), 3)).copy()
    centers[:, :2] += translated
    attitudes = [start_rotation]
    height_shift = 0.
    residuals = [np.zeros(len(offsets))]
    proxy_valid = [surface.visible(initial_points[:, :2]) > .5]
    # The point pattern stays rigid and is used only for this geometric fit.
    # It is never exported as a finger target or used to solve joint angles.
    for idx in range(1, len(arc)):
        anchor = centers[idx].copy()
        previous = attitudes[-1]

        def candidate(values):
            pose = previous * Rotation.from_rotvec(np.r_[values[1:] / .07, 0.])
            points = pose.apply(offsets) + anchor + np.array([0., 0., values[0]])
            gap = points[:, 2] - surface.height(points[:, :2]) - initial_gap
            return pose, points, gap

        def residual(values):
            _, _, gap = candidate(values)
            return np.r_[np.where(gap < 0., 1.7 * gap, gap), .35 * values[1:]]

        solved = least_squares(residual, np.r_[height_shift, 0., 0.], max_nfev=20,
                              bounds=([-.08, -.016, -.016], [.08, .016, .016]))
        attitude, points, gap = candidate(solved.x)
        height_shift = solved.x[0]
        centers[idx, 2] += height_shift
        attitudes.append(attitude)
        residuals.append(gap)
        proxy_valid.append(surface.visible(points[:, :2]) > .5)
    errors = np.asarray(residuals)
    supported = (abs(errors) <= PROXY_TOLERANCE_M) & np.asarray(proxy_valid)
    quorum = supported.sum(axis=1)
    if quorum.min() < 3:
        i = int(np.argmin(quorum))
        raise ValueError(f'Fewer than three footprint samples fit at {arc[i]:.3f} m '
                         f'(offset {translated[i]}, residual {errors[i]})')
    rots = Rotation.concatenate(attitudes)
    xyz = centers - rots.apply(footprint)
    xyz[0] = start_position
    position, orientation = CubicSpline(arc, xyz), RotationSpline(arc, rots)
    times, speed = time_parameterize(arc, position, orientation, calibration)
    settle_at = int(np.ceil((STARTUP_S + times[-1]) / DT))
    ticks = np.arange(settle_at + round(SETTLE_S / DT) + 1)
    at = np.clip(ticks * DT - STARTUP_S, 0., times[-1])
    progress = PchipInterpolator(times, arc)(at)
    progress[at == times[-1]] = arc[-1]
    poses = np.column_stack((position(progress), as_wxyz(orientation(progress))))
    poses[at == 0.] = np.r_[start_position, as_wxyz(start_rotation)]
    return WristTape(poses, progress, settle_at, footprint, offsets,
                     dict(path_length_m=float(arc[-1]), planning_knots=len(arc),
                          fit_quorum_min=int(quorum.min()), fit_four_fraction=float(np.mean(quorum == 4)),
                          max_abs_fit_residual_m=float(abs(errors).max()),
                          wrist_path_length_m=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()),
                          max_wrist_tilt_rad=float((start_rotation.inv() * rots).magnitude().max()),
                          scan_time_s=float(times[-1]), scan_end_s=settle_at * DT,
                          nominal_total_s=len(poses) * DT, maximum_scan_speed_m_s=float(speed.max())))


class ContactWristSchedule:
    """Extend a committed wrist horizon from actual contacts, without changing its route."""

    def __init__(self, tape, budget, horizon):
        self.tape = tape
        self.budget = budget
        self.horizon = horizon
        self.published = 0
        self.poses = np.empty((budget, 7))
        self.trace = np.empty((budget, 5))  # Nominal time, rate, normal offset, two tilt components.
        self.cursor = -1.
        self.state = np.array([1., 0., 0., 0.])
        self.forces = self.loads = None
        offsets = tape.points_in_wrist
        lever = np.column_stack((np.ones(len(offsets)), offsets[:, 1] / .07, -offsets[:, 0] / .07))
        self.tilt_fit = np.linalg.solve(lever.T @ lever + np.diag([.1, .8, .8]), lever.T)

    def extend(self, step, touch):
        # Cover the lookahead of every control tick until the next extension.
        end = min(step + self.horizon + UPDATE_STEPS - 1, self.budget)
        if self.forces is None:
            self.forces, self.loads = touch.forces.copy(), touch.loads.copy()
        self.forces += .5 * (touch.forces - self.forces)
        self.loads += .5 * (touch.loads - self.loads)
        target = np.array([1., 0., 0., 0.])
        if step * DT >= STARTUP_S:
            third = np.sort(self.forces)[-3]
            contact_rate = np.interp(third, [.0, .2, .45], [.55, .80, 1.])
            load_rate = np.interp(self.loads.max(), [4., 8., 25.], [1., .75, .45])
            relief = np.clip((self.loads[:4] - 1.8) * .0008, 0., .004)
            adjust = self.tilt_fit @ relief
            if np.count_nonzero(self.forces < .2) >= 2 and self.loads.max() < 1.8:
                adjust[0] = -.0012
            adjust[0] += np.clip((self.loads[-1] - 2.) * .001, 0., .003)
            target = np.r_[min(contact_rate, load_rate), np.clip(adjust[0], -.0015, .004),
                           np.clip(adjust[1:] / .07, -.035, .035)]
        # Keep published commands fixed. Each new tail is ramped from the previous one.
        envelope = DT * np.array([2., .004, .055, .055])
        for i in range(self.published, end):
            self.state += np.clip(target - self.state, -envelope, envelope)
            self.cursor = min(self.cursor + self.state[0], len(self.tape.poses) - 1.)
            self.trace[i] = np.r_[self.cursor, self.state]
        next_rows = slice(self.published, end)
        rows = self.trace[next_rows]
        lo = rows[:, 0].astype(int)
        hi = np.minimum(lo + 1, len(self.tape.poses) - 1)
        blend = rows[:, 0] - lo
        nominal = self.tape.poses[lo] + blend[:, None] * (self.tape.poses[hi] - self.tape.poses[lo])
        rotation = Rotation.from_quat(nominal[:, [4, 5, 6, 3]])
        tilt = Rotation.from_rotvec(np.column_stack((rows[:, 3:], np.zeros(len(rows)))))
        corrected = rotation * tilt
        center = nominal[:, :3] + rotation.apply(self.tape.footprint_in_wrist)
        delta = rotation.apply(np.column_stack((np.zeros((len(rows), 2)), rows[:, 2])))
        self.poses[next_rows, :3] = center + delta - corrected.apply(self.tape.footprint_in_wrist)
        self.poses[next_rows, 3:] = as_wxyz(corrected)
        self.published = end
