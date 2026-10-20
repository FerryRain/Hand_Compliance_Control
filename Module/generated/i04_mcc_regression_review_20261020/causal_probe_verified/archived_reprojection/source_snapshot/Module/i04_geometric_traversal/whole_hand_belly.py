"""Joint palmar-surface fitting for an actively shaped, cooperative hand.

The former single-leader experiment remains frozen. All selected fingers share
the wrist solve, with independent surface targets and a flexed posture prior.
Geometry never grants contact or coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
import trimesh

from Module.fr3_leap import ARM_HOME_Q
from .natural_belly_traversal import NaturalBellyPlanner, set_kinematics, unit
from .belly_contact import belly_ik_terms
from .full_surface import build_scene


def cooperative_hand_reference():
    """All long fingers flex; the thumb actively opposes the same surface patch."""
    return np.r_[np.tile([0., .22, .35, .22], 3), [.08, 1.1, .35, .25]]


def contact_clearance_handoff(clearance, offset, maximum_offset=.004):
    """Transfer approach clearance into MCC memory without moving its target."""
    shift = min(float(clearance), max(0., float(offset)+maximum_offset))
    return float(clearance)-shift, float(offset)-shift


def cooperative_window_ready(safe_history):
    """Require all four to participate, while tolerating brief individual gaps.

    The 300 ms window must contain at least three qualified contacts for 90%
    of observations and each finger for 60%; the current observation has four.
    This is a motion handoff condition, never a contact or coverage classifier.
    """
    safe=np.asarray(safe_history,dtype=bool)
    if safe.shape!=(150,4):return False
    return bool(safe[-1].all() and np.mean(safe.sum(axis=1)>=3)>=.90 and np.all(safe.mean(axis=0)>=.60))


class WholeHandBellyPlanner(NaturalBellyPlanner):
    def __init__(self, handles, graph):
        super().__init__(handles, graph)
        # Natural hand shape, intersected with the unchanged physical ranges.
        # Positive PIP/DIP minima prevent the old extended waiting posture.
        self.lower[7:19] = np.maximum(self.safety.lower[7:19], np.tile([-.18, -.12, .10, .07], 3))
        self.upper[7:19] = np.minimum(self.safety.upper[7:19], np.tile([.18, .85, .95, .75], 3))
        self.reference_hand = cooperative_hand_reference()

    def project_pads(self, q):
        set_kinematics(self.h, self.data, q)
        centers = self.data.geom_xpos[self.h.tip_geom_ids].copy()
        points, _, _ = trimesh.proximity.closest_point(self.graph.mesh, centers-self.h.object_position_m)
        normals = np.array([self.smooth_normal(p) for p in points])
        return points+self.h.object_position_m, normals

    def posture_terms(self, q, anchor, *, arm_weight=.0015, finger_weight=.0025):
        preferred = np.r_[anchor[:7], self.reference_hand]
        weights = np.r_[np.full(7, arm_weight), np.full(16, finger_weight)]
        rows = [np.diag(weights)]; errors = [weights*(q-preferred)]
        # Soft tendon-like coupling allows curvature adaptation, not rigid fingers.
        for finger in range(3):
            row = np.zeros(23); row[10+4*finger] = .009; row[9+4*finger] = -.009*.65
            rows.append(row[None]); errors.append(np.array([row@q-.009*.015]))
        for finger in (1, 2):
            for joint in (1, 2, 3):
                row = np.zeros(23); row[7+4*finger+joint] = .0015; row[7+joint] = -.0015
                rows.append(row[None]); errors.append(np.array([row@q]))
        return np.concatenate(errors), np.vstack(rows)

    def object_buffer_terms(self, q, margin=.006):
        """Keep proximal links clear while distal bellies approach together."""
        self.minimum_nonpad_clearance(q, query_m=margin+1e-6)
        m=self.h.model; scratch=self.safety.scratch
        objects=set(map(int,self.h.object_geom_ids));pads=set(map(int,self.h.tip_geom_ids))
        nearest={}
        for contact in scratch.contact:
            a,b=int(contact.geom1),int(contact.geom2)
            if a in objects and b not in objects:other=b;normal=np.asarray(contact.frame[:3])
            elif b in objects and a not in objects:other=a;normal=-np.asarray(contact.frame[:3])
            else:continue
            if other in pads or contact.dist>=margin:continue
            if other in nearest and nearest[other][0]<=contact.dist:continue
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
            mujoco.mj_jac(m,scratch,jp,jr,np.asarray(contact.pos),int(m.geom_bodyid[other]))
            nearest[other]=(float(contact.dist),normal@jp[:,self.order])
        rows=[3.*j for distance,j in nearest.values()]
        errors=[3.*(distance-margin) for distance,j in nearest.values()]
        return np.asarray(errors),np.asarray(rows).reshape(-1,23)

    def fit(self, root, *, fingers=(0, 1, 2, 3), clearance=.003, outer=4, iterations=90):
        """Refit independent surface points, then solve one shared 23-DoF task."""
        root = np.asarray(root, float)
        q = np.clip(root, self.lower+1e-9, self.upper-1e-9)
        anchors, normals = self.project_pads(q)
        total_evaluations = 0
        for _ in range(outer):
            def terms(trial):
                set_kinematics(self.h, self.data, trial)
                errors = []; rows = []
                for finger in fingers:
                    n = normals[finger]
                    e, j, _ = belly_ik_terms(self.h.model, self.data, int(self.h.tip_geom_ids[finger]),
                        anchors[finger], n, clearance_m=clearance, orientation_weight_m=.04, dof_indices=self.order)
                    # Normal distance and belly orientation dominate; tangential
                    # freedom lets neighbours share a comfortable hand shape.
                    metric = np.outer(n, n)+.12*(np.eye(3)-np.outer(n, n))
                    errors.append(np.r_[metric@e[:3], e[3:]])
                    rows.append(np.vstack((metric@j[:3], j[3:])))
                pe, pj = self.posture_terms(trial, root)
                errors.append(pe); rows.append(pj)
                return np.concatenate(errors), np.vstack(rows)
            fit = least_squares(lambda x: terms(x)[0], q, jac=lambda x: terms(x)[1],
                bounds=(self.lower, self.upper), max_nfev=iterations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            q = fit.x; total_evaluations += fit.nfev
            anchors, normals = self.project_pads(q)
        set_kinematics(self.h, self.data, q)
        details = []
        for finger in range(4):
            e, _, d = belly_ik_terms(self.h.model, self.data, int(self.h.tip_geom_ids[finger]),
                anchors[finger], normals[finger], clearance_m=clearance, dof_indices=self.order)
            details.append(dict(finger=finger, normal_error_m=float(e[:3]@normals[finger]),
                angle_deg=d['alignment_angle_deg'], qualified_patch=d['geometry_qualified']))
        reason = self.safety.collision_reason(q)
        accepted = reason is None and all(abs(details[i]['normal_error_m'])<.0008 and
            details[i]['angle_deg']<6. and details[i]['qualified_patch'] for i in fingers)
        return q, dict(accepted=accepted, collision_reason=reason, fingers=details,
            evaluations=int(total_evaluations), hand_deg=np.degrees(q[7:]).reshape(4, 4).tolist()), anchors, normals

    def fit_targets(self, root, points, normals, *, clearance=.0002, iterations=80):
        """A common nominal posture follows the four moving surface targets."""
        root=np.asarray(root);q=np.clip(root,self.lower+1e-9,self.upper-1e-9)
        def terms(trial):
            set_kinematics(self.h,self.data,trial);errors=[];rows=[]
            for finger in range(4):
                n=normals[finger]
                e,j,_=belly_ik_terms(self.h.model,self.data,int(self.h.tip_geom_ids[finger]),points[finger],n,
                    clearance_m=clearance,orientation_weight_m=.04,dof_indices=self.order)
                metric=np.outer(n,n)+.35*(np.eye(3)-np.outer(n,n))
                errors.append(np.r_[metric@e[:3],e[3:]]);rows.append(np.vstack((metric@j[:3],j[3:])))
            pe,pj=self.posture_terms(trial,root,arm_weight=.0015,finger_weight=.0025)
            errors.append(pe);rows.append(pj)
            return np.concatenate(errors),np.vstack(rows)
        fit=least_squares(lambda x:terms(x)[0],q,jac=lambda x:terms(x)[1],bounds=(self.lower,self.upper),
            max_nfev=iterations,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q=fit.x
        return q,dict(collision_reason=self.safety.collision_reason(q),evaluations=int(fit.nfev))


def survey(output, clearance=.003):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    h, graph, home, quality = build_scene(out)
    planner = WholeHandBellyPlanner(h, graph)
    surface, _, _ = trimesh.proximity.closest_point(graph.mesh, np.array([[0., 0., .25]]))
    p = surface[0]+h.object_position_m; n = planner.smooth_normal(surface[0])
    records = []; qs = []; points = []; normals = []
    for roll in (0., np.pi/2, -np.pi/2, np.pi, np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4):
        target, hand = planner.target(p, n, np.array([0., 1., 0.]), finger=1, clearance=.015,
            roll=roll, root_hand=cooperative_hand_reference())
        root = home.copy(); root[7:] = hand
        seed, ep, er, _ = planner.safety.solve(root, target)
        for fingers in ((0, 1, 2, 3), (0, 1, 2)):
            q, diag, pts, ns = planner.fit(seed, fingers=fingers,clearance=clearance)
            diag.update(roll=roll, controlled_fingers=list(fingers), seed_position_error_m=ep)
            records.append(diag); qs.append(q); points.append(pts); normals.append(ns)
            print(json.dumps(diag), flush=True)
    np.savez_compressed(out/'geometry.npz', q=qs, points_world=points, normals_world=normals)
    (out/'model.xml').write_text(h.xml, encoding='utf-8')
    (out/'survey.json').write_text(json.dumps(dict(geometry_only=True, coverage_granted=False, records=records), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--survey', required=True)
    parser.add_argument('--clearance',type=float,default=.003);args=parser.parse_args()
    survey(args.survey,args.clearance)
