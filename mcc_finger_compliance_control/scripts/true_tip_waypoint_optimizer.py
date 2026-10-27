"""Physical split-hull waypoint refinement with a soft distal-tip preference."""
from pathlib import Path
import json
import numpy as np
import mujoco
import trimesh
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from surface_mcc_finger import (HAND_XML, TIP_BODY_NAMES, TIP_NAMES,
    TIP_SITE_LOCAL_POSITIONS, HAND_JOINT_NAMES, _quat_wxyz_to_matrix)
from split_fingertip_collision import replace_tip_collision, ASSETS, GEOMS
from object_catalog import add_object_body


class SplitHullGeometry:
    def __init__(self, config, scale=1.):
        spec = mujoco.MjSpec.from_file(str(HAND_XML))
        spec.option.ccd_iterations = 200
        for ex in list(spec.excludes):
            if ex.bodyname1 == 'thumb_pip' and ex.bodyname2 == 'pip4': spec.delete(ex)
        # Retain the free joint: welding the palm to the world changes MuJoCo
        # parent collision filtering and creates false assembly collisions.
        palm = spec.body('palm_lower'); palm.pos[:] = 0; palm.quat[:] = (1, 0, 0, 0)
        palm.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        present = {s.name for s in spec.sites}
        for body, site, pos in zip(TIP_BODY_NAMES, TIP_NAMES, TIP_SITE_LOCAL_POSITIONS):
            if site not in present: spec.body(body).add_site(name=site, pos=pos, size=(.004, 0, 0))
        native = [(np.array(spec.geom(name).pos), _quat_wxyz_to_matrix(np.array(spec.geom(name).quat))) for name in GEOMS]
        replace_tip_collision(spec)
        add_object_body(spec, config, body_name='waypoint_target', mocap=True, scale=scale)
        self.model = spec.compile(); self.data = mujoco.MjData(self.model)
        self.hand_qpos = np.asarray([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in HAND_JOINT_NAMES])
        self.pads = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name+'_pad_collision') for name in GEOMS]
        self.connectors = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name+'_connector_collision') for name in GEOMS]
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'waypoint_target')
        self.mocap = int(self.model.body_mocapid[object_id])
        # The source mesh used for rendering is an inactive, combined hull.
        # Only the actual colliding decomposition parts may define contact.
        self.objects = np.where((self.model.geom_bodyid == object_id)
            & ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)))[0]
        self.native = native
        self.parts = []
        for kind in ('fingertip_two_part', 'thumb_fingertip_two_part'):
            split = json.loads((ASSETS/kind/'split_manifest.json').read_text())
            mesh = trimesh.load(ASSETS/kind/'pad_convex.obj', force='mesh', process=False)
            sign = 1 if split['pad_side'] == 'positive' else -1
            axis = sign*np.asarray(split['plane_normal_source_mesh'])
            offset = sign*split['plane_offset_m']
            span = float((mesh.vertices @ axis).max()-offset)
            self.parts.append((axis, offset, span, ConvexHull(mesh.vertices).equations))
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'palm_lower')
        descendants = set()
        for b in range(1, self.model.nbody):
            parent = b
            while parent > 0 and parent != root: parent = int(self.model.body_parentid[parent])
            if parent == root: descendants.add(b)
        self.robot_geoms = {g for g, b in enumerate(self.model.geom_bodyid) if b in descendants}
        self.branch = {}
        for b in descendants:
            parent = b
            while int(self.model.body_parentid[parent]) not in (root, 0):
                parent = int(self.model.body_parentid[parent])
            self.branch[b] = parent
        self.root_body = root
        # Collision-active assembly parts only. Visual hulls and the accepted
        # true fingertip pads must never become forbidden contact surfaces.
        self.non_tip_geoms = np.asarray(sorted(
            g for g in self.robot_geoms if g not in self.pads
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])), dtype=int)
        self.non_tip_names = tuple(self.model.geom(int(g)).name for g in self.non_tip_geoms)

    def position(self, q, palm_pose):
        palm_pose = np.asarray(palm_pose,dtype=np.float64).reshape(7)
        quaternion = palm_pose[3:]/np.linalg.norm(palm_pose[3:])
        r = _quat_wxyz_to_matrix(quaternion)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.hand_qpos] = q
        self.data.mocap_pos[self.mocap] = -r.T @ palm_pose[:3]
        # MuJoCo's collision queries require a normalized mocap quaternion.
        # Float32 recording error otherwise makes repeated clearance queries
        # disagree with the normalized pose used by the planner's transforms.
        self.data.mocap_quat[self.mocap] = quaternion*np.array([1,-1,-1,-1])
        mujoco.mj_kinematics(self.model, self.data)
        return r

    def evaluate(self, q, palm_pose, self_collision=False, non_tip_clearance=False):
        r = self.position(q,palm_pose)
        distances, connector_dist, points, top, cap, normals = [], [], [], [], [], []
        hand_points_palm = []
        def nearest(geom, witness=False):
            # Bounding spheres give conservative lower bounds. Sorting them
            # avoids running GJK against all 257 object decomposition parts
            # for every finite-difference joint perturbation.
            lower = (np.linalg.norm(self.data.geom_xpos[self.objects]
                -self.data.geom_xpos[geom], axis=1)
                -self.model.geom_rbound[self.objects]-self.model.geom_rbound[geom])
            best, best_line = .2, np.zeros(6)
            for index in np.argsort(lower):
                if lower[index] > best:
                    break
                line = np.zeros(6)
                d = mujoco.mj_geomDistance(self.model, self.data, geom,
                    int(self.objects[index]), .2, line)
                if d == 0. and not np.any(line):
                    # Failed mesh CCD queries can return zero and leave the
                    # witness buffer untouched. This is not a contact point.
                    continue
                # Some disjoint mesh pairs report zero with separated witness
                # points. They cannot define touching or penetrating contact.
                if d == 0. and np.linalg.norm(line[:3]-line[3:]) > 1e-6:
                    d = float(np.linalg.norm(line[:3]-line[3:]))
                if d < best:
                    best = d
                    if witness: best_line = line.copy()
            return best, best_line
        for f in range(4):
            best, best_line = nearest(self.pads[f], witness=True)
            distances.append(best)
            connector_dist.append(nearest(self.connectors[f])[0])
            bid = int(self.model.geom_bodyid[self.pads[f]])
            br = self.data.xmat[bid].reshape(3, 3)
            body = br.T @ (best_line[:3]-self.data.xpos[bid])
            gp, gr = self.native[f]; local = gr.T @ (body-gp)
            axis, offset, span, planes = self.parts[1 if f == 3 else 0]
            top.append(float(np.clip((local @ axis-offset)/span, 0, 1)))
            facet = int(np.argmax(planes[:, :3] @ local+planes[:, 3]))
            cap.append(bool(planes[facet, :3] @ axis < -.999))
            normals.append(r @ br @ gr @ planes[facet, :3])
            points.append(r @ best_line[3:]+palm_pose[:3])
            hand_points_palm.append(best_line[:3].copy())
        depth = 0.
        if self_collision:
            mujoco.mj_collision(self.model, self.data)
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if int(c.geom[0]) in self.robot_geoms and int(c.geom[1]) in self.robot_geoms:
                    b0, b1 = (int(self.model.geom_bodyid[int(g)]) for g in c.geom)
                    # Intrafinger housing intersections are part of this
                    # model's mechanical assembly, not cross-finger posture.
                    if (b0 != self.root_body and b1 != self.root_body
                            and self.branch[b0] != self.branch[b1]):
                        depth = max(depth, -float(c.dist))
        result = dict(distance=np.asarray(distances), connector_distance=np.asarray(connector_dist),
                    object_points=np.asarray(points), top_fraction=np.asarray(top),
                    pad_surface_normals=np.asarray(normals), cut_cap=np.asarray(cap), self_penetration=depth,
                    hand_points_palm=np.asarray(hand_points_palm))
        if non_tip_clearance:
            # Reuse the four connector queries; all other links receive the
            # same exact active-object decomposition distance calculation.
            connectors = dict(zip(self.connectors, connector_dist))
            result['non_tip_distance'] = np.asarray([
                connectors[int(g)] if int(g) in connectors else nearest(int(g))[0]
                for g in self.non_tip_geoms])
        return result


class TrueTipWaypointController:
    def __init__(self, base, config, scale=1., top_weight=.05, top_fraction=.75,
                 non_tip_clearance_m=None, max_cross_finger_penetration_m=.0001,
                 clearance_polish=True, preferred_non_tip_clearance_m=None):
        self.base = base; self.geometry = SplitHullGeometry(config, scale)
        self.top_weight, self.preferred_fraction = top_weight, top_fraction
        self.last_geometry = None
        self.measured_seed_mode = False
        self.health_limits = None
        # Opt-in for measured-palm initialization. Existing trajectory solves
        # retain their original objective unless explicitly enabled.
        self.kinematic_health_weight = 0.
        if non_tip_clearance_m is not None and not 0 <= non_tip_clearance_m <= .02:
            raise ValueError('non-tip clearance must be 0..20 mm')
        if max_cross_finger_penetration_m < 0:
            raise ValueError('cross-finger penetration tolerance must be nonnegative')
        self.non_tip_clearance_m = non_tip_clearance_m
        if preferred_non_tip_clearance_m is not None and (
                non_tip_clearance_m is None or not np.isfinite(preferred_non_tip_clearance_m)
                or preferred_non_tip_clearance_m < non_tip_clearance_m or preferred_non_tip_clearance_m > .02):
            raise ValueError('preferred non-tip clearance requires and must exceed the hard margin')
        self.preferred_non_tip_clearance_m = (non_tip_clearance_m
            if preferred_non_tip_clearance_m is None else preferred_non_tip_clearance_m)
        self.max_cross_finger_penetration_m = max_cross_finger_penetration_m
        self.clearance_polish = clearance_polish

    def geometry_valid(self, g):
        valid = (np.isfinite(g['distance']).all() and (g['distance'] <= .004).all()
                 and (g['distance'] >= -.012).all()
                 and np.isfinite(g['connector_distance']).all()
                 and (g['connector_distance'] >= -.0005).all() and not g['cut_cap'].any())
        if self.non_tip_clearance_m is not None:
            valid = (valid and np.isfinite(g['non_tip_distance']).all()
                     and (g['non_tip_distance'] >= self.non_tip_clearance_m-1e-6).all()
                     and g['self_penetration'] <= self.max_cross_finger_penetration_m+1e-6)
        return bool(valid)

    def __getattr__(self, name): return getattr(self.base, name)

    def solve_geometry_contact_posture(self, oracle, palm_pose_world, seed_q, nominal_q, **kwargs):
        # Structured seed/health search remains; fixed front-normal preference
        # is removed. Distal top is preferred softly, never an acceptance gate.
        kwargs['pad_alignment_gain'] = 0.
        lower = self.base.lower+.03; upper = self.base.upper-.03
        if self.health_limits is not None:
            bound = self.health_limits['posture']-1e-6
            lower = np.maximum(lower, nominal_q-bound)
            upper = np.minimum(upper, nominal_q+bound)
        preload = float(kwargs.get('preload_m', .002))
        maps = self.base._kinematic_health_maps()
        def residual(value):
            g = self.geometry.evaluate(value, palm_pose_world, self_collision=True,
                non_tip_clearance=self.non_tip_clearance_m is not None)
            health = np.zeros(0)
            if self.kinematic_health_weight > 0:
                tip = self.base.tip_positions_palm(value)
                eta, mu = np.asarray([self.base._finger_kinematic_health(
                    f, tip[f], maps[f]) for f in range(4)]).T
                health = np.r_[
                    np.sqrt(self.kinematic_health_weight*.004)*(1/(mu+.05)-1/1.05),
                    np.sqrt(self.kinematic_health_weight*.002)*(1/(eta+.02)-1/1.02)]
            if self.health_limits is not None:
                tip = self.base.tip_positions_palm(value)
                eta, mu = np.asarray([self.base._finger_kinematic_health(f,tip[f],maps[f]) for f in range(4)]).T
                h = self.health_limits
                lateral = np.asarray([h['order'][i]*h['axis'] @ (tip[left]-tip[right])
                    for i,(left,right) in enumerate(((0,1),(1,2)))])
                health = np.r_[health, np.maximum(h['eta']-eta,0)/.01,
                    np.maximum(h['mu']-mu,0)/.01,
                    np.maximum(h['lateral']-lateral,0)/.002]
            clearance = np.zeros(0)
            if self.non_tip_clearance_m is not None:
                clearance = np.r_[
                    4*np.maximum(self.preferred_non_tip_clearance_m-g['non_tip_distance'], 0)/.002,
                    4*max(g['self_penetration']-self.max_cross_finger_penetration_m, 0)/.002]
            return np.concatenate(((g['distance']+preload)/.002,
                2*np.maximum(.0005-g['connector_distance'], 0)/.002,
                np.sqrt(self.top_weight)*np.maximum(self.preferred_fraction-g['top_fraction'], 0),
                np.sqrt(.08)*(value-seed_q)/.3,
                np.sqrt(float(kwargs.get('posture_gain', .03)))*(value-nominal_q)/.6,
                health, clearance))
        # First reuse an already feasible measured/temporal hand posture.
        # A site-based solve can discard that basin because its contact proxy
        # belongs to the old combined mesh rather than the real pad hull.
        q = np.clip(seed_q, lower, upper)
        initial = self.geometry.evaluate(q, palm_pose_world)
        if not self.measured_seed_mode and ((initial['distance'] > .004).any()
                or (initial['distance'] < -.012).any() or initial['cut_cap'].any()):
            candidate, _ = self.base.solve_geometry_contact_posture(
                oracle, palm_pose_world, seed_q, nominal_q, **kwargs)
            candidate = np.clip(candidate, lower, upper)
            if np.sum(residual(candidate)**2) < np.sum(residual(q)**2):
                q = candidate
        result = least_squares(residual, q, bounds=(lower, upper), max_nfev=40,
                               diff_step=1e-3,
                               ftol=1e-5, xtol=1e-5, gtol=1e-5)
        q = result.x
        g = self.geometry.evaluate(q, palm_pose_world, self_collision=True,
            non_tip_clearance=self.non_tip_clearance_m is not None)
        polish_iterations = 0
        if self.non_tip_clearance_m is not None and self.clearance_polish and not self.geometry_valid(g):
            # Penalties alone generally converge just outside a hard boundary.
            # Keep the least-squares warm start, then enforce signed distances
            # in a constrained solve. Acceptance still rechecks exact geometry.
            def constraints(value):
                value_g = self.geometry.evaluate(value, palm_pose_world,
                    self_collision=True, non_tip_clearance=True)
                return np.r_[value_g['non_tip_distance']-self.non_tip_clearance_m,
                    .004-value_g['distance'], value_g['distance']+.012,
                    self.max_cross_finger_penetration_m-value_g['self_penetration']]/.002
            def objective(value):
                errors = residual(value)
                return float(errors @ errors)
            polished = minimize(objective, q, method='SLSQP', bounds=list(zip(lower,upper)),
                constraints={'type':'ineq','fun':constraints},
                options={'maxiter':60,'ftol':1e-6,'eps':1e-4})
            polish_iterations = int(polished.nit)
            candidate_g = self.geometry.evaluate(polished.x, palm_pose_world,
                self_collision=True, non_tip_clearance=True)
            if (self.geometry_valid(candidate_g)
                    or np.minimum(constraints(polished.x),0) @ np.minimum(constraints(polished.x),0)
                    < np.minimum(constraints(q),0) @ np.minimum(constraints(q),0)):
                q, g = polished.x, candidate_g
        if self.non_tip_clearance_m is not None:
            # The dataset stores float32 waypoints. Certify that exact saved
            # representation rather than only the optimizer's float64 state.
            q = np.asarray(q,dtype=np.float32).astype(np.float64)
            g = self.geometry.evaluate(q,palm_pose_world,self_collision=True,non_tip_clearance=True)
        self.last_geometry = g
        self.base._set_q(self.base.data, q)
        tip = self.base.tip_positions_palm(q)
        maps = self.base._kinematic_health_maps()
        eta, mu = zip(*(self.base._finger_kinematic_health(f, tip[f], maps[f]) for f in range(4)))
        spread, synergy = self.base.flexion_synergy_metrics(q)
        normal = oracle.normals_at_world(g['object_points'])
        # No hard angular gate tied to a fixed fingertip axis. Actual convex
        # geometry supplies contact existence; report top location separately.
        valid = np.isfinite(q).all() and self.geometry_valid(g)
        angle = np.arccos(np.clip((g['pad_surface_normals']*-normal).sum(-1), -1, 1))
        diagnostic = dict(surface_point_world=g['object_points'], surface_normal_world=normal,
            signed_distance=g['distance'], pad_normal_error=angle,
            synergy_spread=spread, synergy_residual=synergy,
            isotropic_reachability=np.asarray(eta), manipulability=np.asarray(mu),
            top_fraction=g['top_fraction'], connector_distance=g['connector_distance'],
            self_penetration=g['self_penetration'], valid=bool(valid), score=float(np.sum(residual(q)**2)))
        if 'non_tip_distance' in g:
            diagnostic['non_tip_distance'] = g['non_tip_distance']
            diagnostic['clearance_polish_iterations'] = polish_iterations
        return q, diagnostic

    def pad_normal_errors(self, q, palm_pose_world, normals):
        # Legacy helper is called during dense diagnostics. It must not
        # reintroduce the removed fixed-front angular constraint.
        g = self.geometry.evaluate(q, palm_pose_world)
        return np.arccos(np.clip((g['pad_surface_normals']*-normals).sum(-1), -1, 1))
