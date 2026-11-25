"""The active decision is a single wrist transform scored over the uncertain zero set."""
from time import perf_counter

import numpy as np

from .belief import Belief
from .common import rot, unit
from .geometry import Clearance, anchors, candidates, uncertain_surface
from .probe import Probe, scores


def ms(start):
    return (perf_counter()-start)*1000


def scale(values):
    return (values-values.min())/max(float(np.ptp(values)), 1e-12)


class Planner:
    def __init__(self, models, cfg, variant, origin, plate):
        self.gp = Belief.load(models / "gpis.pt")
        self.probe = Probe(models, variant)
        self.cfg, self.variant, self.origin, self.plate = cfg, variant, origin, plate

    def plan(self, tick, splice, path, actual, pads, hull, points, normals, incumbent):
        start = perf_counter()
        cfg = self.cfg
        row = dict(tick=tick, cells=len(points), splice_tick=splice, variant=self.variant)

        def finish(reason, tail=None):
            row.update(reason=reason, planning_ms=ms(start))
            return row, tail

        if not len(points):
            return finish("awaiting_contact")
        t = perf_counter()
        row["update_mode"] = self.gp.condition(points, normals)
        row["gpis_update_ms"] = ms(t) if row["update_mode"] != "cached" else 0.
        t = perf_counter()
        refs = anchors(self.plate, pads, cfg.anchor_width_fraction)
        focus = actual[:3]+rot(actual).apply(refs[0]) if incumbent is None else np.array(incumbent["point"])
        active = incumbent is not None and splice < path.end
        changed = False
        if active:
            mean, var, grad = self.gp.query(focus[None])
            changed = (var[0] <= cfg.variance_retained*incumbent["variance"]
                       or abs(mean[0]) >= cfg.surface_band_m
                       or unit(grad[0]) @ incumbent["normal"] < np.cos(np.deg2rad(cfg.normal_change_deg)))
            row.update(incumbent_variance_m2=float(var[0]), incumbent_geometry_changed=bool(changed))
        surface = uncertain_surface(self.gp, points, normals, focus, cfg)
        row["gpis_query_ms"] = ms(t)
        if surface is None:
            return finish("no_supported_crossing")
        t = perf_counter()
        poses, owners = candidates(surface, actual, self.origin, refs, cfg)
        clearance = Clearance(points, normals, hull, actual, cfg)
        if active:
            poses = np.vstack((poses, path.at(path.end)))
            owners = np.r_[owners, np.linalg.norm(surface.points-focus, axis=1).argmin()]
        if not len(poses):
            return finish("outside_workspace")
        ok = clearance.valid(poses)
        incumbent_valid = active and ok[-1]
        poses, owners = poses[ok], owners[ok]
        row.update(candidate_ms=ms(t), feasible=len(poses), regions_count=len(surface.regions))
        if not len(poses):
            return finish("no_clear_endpoint")
        t = perf_counter()
        attitudes = rot(poses)
        matrices = attitudes.as_matrix()
        local = np.einsum("cpi,cij->cpj", surface.points[None]-poses[:, None, :3], matrices)
        weights = surface.variance/(surface.variance.max()+1e-12)
        k = perf_counter()
        if self.variant == "A":
            raw = np.exp(-np.square(local - refs[0]).sum(2)/(2*cfg.uncertainty_aperture_m**2)) @ weights
            comparisons, by_finger = {}, None
        else:
            masses = self.probe.fingers(local)
            comparisons, by_finger = scores(masses, weights)
            raw = comparisons.get(self.variant, comparisons["D"])
        row["kde_ms"] = ms(k) if self.variant not in ("A", "FK", "Hull") else 0.
        row["representation_ms"] = ms(k)
        # A palm's inward axis is -Z. An outward normal parallel to +Z gives maximal alignment.
        alignment = ((1 + np.clip(matrices[:, :, 2] @ surface.normals.T, -1, 1))/2) @ weights/weights.sum()
        translation = np.linalg.norm(poses[:, :3]-actual[:3], axis=1)/cfg.translation_scale_m
        angle = (rot(actual).inv()*attitudes).magnitude()/cfg.rotation_scale_rad
        motion = translation**2 + cfg.gamma*angle**2
        value = scale(raw) - cfg.motion_weight*scale(motion)
        if self.variant not in ("A", "B"):
            value += cfg.orientation_weight*alignment
        order = np.argsort(-value, kind="stable")
        for j in order:
            if incumbent_valid and (j == len(poses)-1 or value[j] <= value[-1] + (0 if changed else cfg.switch_margin)):
                row["evaluation_ms"] = ms(t)
                return finish("keep_current_target")
            tail = path.revised(splice, poses[j], cfg)
            if tail is None:
                continue
            # Cover angular sweeps as well as translations. The dense 100 Hz command
            # samples are checked in blocks to bound temporary arrays.
            within = (np.linalg.norm(tail.samples[:, :3]-self.origin[:3], axis=1) <= cfg.workspace_radius_m).all()
            if not within or not all(clearance.valid(block).all() for block in np.array_split(
                    tail.samples, 1 + len(tail.samples)//96)):
                continue
            owner = owners[j]
            row.update(selected=int(j), target=poses[j].tolist(), end_tick=tail.end,
                       point=surface.points[owner].tolist(), normal=surface.normals[owner].tolist(),
                       variance=float(surface.variance[owner]), predicted=float(raw[j]/weights.sum()),
                       surface=surface.points.tolist(), normals=surface.normals.tolist(),
                       variance_m2=surface.variance.tolist(), regions=surface.regions.tolist(),
                       poses=poses.tolist(), scores=value.tolist(), probe_sum=raw.tolist(),
                       orientation=alignment.tolist(), motion=motion.tolist(),
                       aggregation={key: (v/weights.sum()).tolist() for key, v in comparisons.items()},
                       finger_mass=None if by_finger is None else (by_finger[j]/weights.sum()).tolist(),
                       trigger="geometry" if active else "segment_end", evaluation_ms=ms(t))
            return finish("new_target", tail)
        row["evaluation_ms"] = ms(t)
        return finish("no_clear_trajectory")
