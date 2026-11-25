"""Follow the tactile surface under the palm while preserving supporting fingertips."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion
from . import paired_support, palm_landing as landing
from .contact_first import signed_gap
from .palm_surface import OUT, normalize, on_plane
from .sweep import DT, LOAD_LIMIT
from .ycb_fixtures import OBJECTS


def following_twist(state, row, result, cap_points, forces):
    """Fit the next height and two tilts to measured support and nearby distal skin."""
    active = np.flatnonzero(state['hits'])
    if not len(active) or row['mode'] != 'explore' or max(state['loads']) > 4.5:
        return
    surface = state['surface']
    pivot = np.asarray(row['palm_pivot_m'])
    attitude = state['wrist_rotation']
    target = Rotation.from_quat(row['palm_target_xyzw'])
    surface_n = np.asarray(row['palm_patch_outward'])
    hand_axes = attitude.as_matrix()[:, :2]
    desired_n = target.apply(OUT)
    error = (target * attitude.inv()).as_rotvec()
    omega = limit(2.65 * on_plane(error, desired_n) + 1.15 * (error @ desired_n) * desired_n, 1.25)
    horizon = .16
    turn = Rotation.from_rotvec(horizon * omega)
    trim = np.asarray(row['palm_trim_velocity'])
    speed = row['tangent_speed_m_s'] * result['direction'] + row['height_speed_m_s'] * result['normal'] + trim
    leverage = .06
    rows, residuals, present = [], [], []
    for f, skin in enumerate(cap_points):
        measured = state['hits'][f]
        if measured:
            at, hint = state['pads'][f], state['normals'][f]
        else:
            if state['loads'][f] > 1.2:
                continue
            distance, near = surface.tree.query(skin)
            ns = surface.normals[near]
            gap = np.sum((skin - surface.points[near]) * ns, axis=1)
            valid = (distance < .022) & (gap > -.001) & (gap < .012) & (ns @ surface_n > .42)
            if not valid.any():
                continue
            j = np.argmin(np.where(valid, distance + .35 * np.abs(gap), np.inf))
            at, hint = skin[j], ns[j]
        patch = surface.patch(at, preferred_normal=hint)
        if patch is None:
            continue
        start, _ = signed_gap(patch, at)
        if not measured and not -.001 < start < .012:
            continue
        offset = turn.apply(at - pivot)
        forecast, n = signed_gap(patch, pivot + horizon * speed + offset)
        if measured:
            rest = start - np.clip(.00045 * (.9 - forces[f]), -.0007, .00045)
            weight = 1.
        else:
            rest, weight = max(.0002, .45 * start), .54
        rows.append(weight * np.r_[n @ surface_n, np.cross(offset, n) @ hand_axes / leverage])
        residuals.append(weight * (forecast - rest))
        present.append(f)
    if not rows:
        return
    j = np.asarray(rows)
    correction = np.linalg.solve(j.T @ j + np.diag([.15, .26, .26]), -j.T @ residuals)
    height = np.clip(correction[0], -.0027, .0035) / horizon
    if state['loads'][-1] >= .32 or max(state['loads']) > 3.:
        height = max(0., height)
    adjustment = limit(hand_axes @ correction[1:] / (horizon * leverage), .64)
    velocity = limit(omega + adjustment, 1.25)
    # Invert the two gains of the 100 Hz palm servo. An increment from this
    # support fit can change tilt by at most a few degrees per planning interval.
    reference_error = on_plane(velocity, desired_n) / 2.65 + (velocity @ desired_n) * desired_n / 1.15
    next_ref = Rotation.from_rotvec(reference_error) * attitude
    row.update(palm_target_xyzw=next_ref.as_quat().tolist(),
               palm_trim_velocity=(trim + height * surface_n).tolist(),
               palm_fitted_fingers=present, palm_height_fit_mm=float(height * horizon * 1000),
               palm_support_turn_rad_s=adjustment.tolist())


class SupportFollower(landing.LandingFollower):
    def face_surface(self, state, row, result):
        # Blend the landing pose as the measured palm site and global target approach.
        landing.first.PalmFollower.face_surface(self, state, row, result)
        if not row['palm_reference_valid']:
            return
        goal = self.guide.palm_goal
        att, wrist = state['wrist_rotation'], state['wrist_position']
        if isinstance(goal, landing.Landing) and goal.region is self.guide.region:
            site = wrist + att.apply(goal.site)
            delta = goal.patch - site
            lateral = np.linalg.norm(on_plane(delta, goal.normal))
            tracking_error = goal.region.tracking_error(state)
            blend = np.exp(-.5 * ((lateral / .048)**2 + (tracking_error / .038)**2))
            # Also limit roll/yaw around the palm normal while transferring
            # between patches. Prefer the rotation supported at the current site.
            here = Rotation.from_quat(row['palm_target_xyzw'])
            local = landing.align(att.apply(OUT), here.apply(OUT)) * att
            target = Rotation.from_rotvec(blend * (goal.rotation * local.inv()).as_rotvec()) * local
            drift = goal.patch + goal.height * goal.rotation.apply(OUT) - site
            trim = np.asarray(row['palm_trim_velocity']) + limit(.34 * blend * on_plane(drift, goal.normal), .009)
            row.update(palm_target_xyzw=target.as_quat().tolist(), palm_trim_velocity=trim.tolist(),
                       palmar_site_local_m=goal.site.tolist(), palm_landing_blend=float(blend),
                       palm_endpoint_m=goal.patch.tolist())
        following_twist(state, row, result, self.cap_points, self.tip_forces)
        pivot = np.asarray(row['palm_pivot_m'])
        target = Rotation.from_quat(row['palm_target_xyzw'])
        turn = Rotation.from_rotvec(limit((target * att.inv()).as_rotvec(), .048))
        shift = row['tangent_speed_m_s'] * result['direction'] + np.asarray(row['palm_trim_velocity'])
        result.update(position=pivot + turn.apply(wrist - pivot) + .05 * shift, rotation=turn * att)


class SupportWrist(landing.LandingWrist):
    def __init__(self, *args, **opts):
        super().__init__(*args, **opts)
        base = self.selector.base
        base.global_model = base.guide.model = self.selector.global_model.base
        self.selector = SupportFollower(base, self.selector.sites)
        self.cap_material = paired_support.pad_caps(self.hand)
        self.release_normal = None

    def observe(self, data, touch, tick):
        super().observe(data, touch, tick)
        if touch.loads.max() < 3.5:
            self.release_normal = None
        if self.high_since is None or tick - self.high_since < 18 or tick % 5 or self.surface.tree is None:
            return
        region = np.argmax(touch.loads)
        if region < 4 and touch.measured_heads[region] and touch.forces[region] > .55 * touch.loads[region]:
            self.release_normal = touch.normals[region].copy()
        elif region < 4:
            hull = self.link_hull
            indices = hull.regions == region
            geoms = hull.ids[indices]
            vertices = data.geom_xpos[geoms] + np.einsum('vij,vj->vi', data.geom_xmat[geoms].reshape(-1,3,3), hull.vertices[indices])
            d, near = self.surface.tree.query(vertices)
            k = np.argmin(d)
            if d[k] < .040:
                normals = self.surface.normals[near]
                valid = (d < d[k] + .007) & (normals @ normals[k] > .65)
                self.release_normal = normalize(normals[valid].mean(0))
        else:
            row = self.transition or {}
            if row.get('palm_reference_valid'):
                self.release_normal = np.asarray(row['palm_patch_outward'])

    def command(self, touch):
        if (self.unloading and self.high_since is not None and self.tick - self.high_since >= 22
                and self.release_normal is not None and max(touch.loads) >= 4.):
            # A previous safe pose may now be occupied by a bent finger. Seek
            # actual load relief along the surface beside the stressed hand link.
            delta = DT * .086 * self.release_normal
            position = self.measured_position + limit(self.position - self.measured_position + delta, .012)
            self.position = self.workspace_center + limit(position - self.workspace_center, .45)
            cmd = np.r_[self.position, quaternion(self.attitude)]
            return cmd, cmd.copy()
        return super().command(touch)


class SupportWorker(landing.first.previous.ReturnWorker):
    def submit(self, data, touch, tick, lead):
        s = self.servo
        s.selector.cap_points = [o + skin @ r.T for o, r, skin in zip(s.origins, s.matrices, s.cap_material)]
        return super().submit(data, touch, tick, lead)


def run(folder, obj, seconds=240, device='cuda', prior=None):
    first = landing.first
    prev = first.previous
    return landing.field.run(folder, obj, seconds, device, prior, wrist_cls=SupportWrist, worker_cls=SupportWorker,
        source_files=(__file__, paired_support.__file__, landing.__file__, first.__file__, first.palm_acquisition.__file__,
                      first.palm_surface.__file__, prev.__file__, prev.ct.__file__, prev.ct.entry_frontier.__file__,
                      prev.ct.transition_compute.__file__, prev.ct.delivery.__file__, prev.ct.tactile_reentry.__file__,
                      prev.ct.tactile_stall.__file__), release='mesh-palm-contact-following')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=OBJECTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=240)
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args()
    print(run(args.out, args.object, args.seconds, prior=args.prior), flush=True)
