"""Select translations of the physical palm by the expected information of a policy-reachable contact footprint."""

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from . import palm_unload, policy_reach
from .contact_first import palm_clear
from .informative_actions import MASK_THRESHOLDS, joint_gain
from .multirate_wrist import GLOBAL_TICKS, SPEED_M_S
from .palm_landing import Landing
from .palm_surface import OUT, normalize, on_plane
from .policy_reach import HORIZONS
from .sweep import DT, LOAD_LIMIT
from .ycb_fixtures import OBJECTS


@dataclass
class PalmMove:
    center: np.ndarray
    site: np.ndarray
    normal: np.ndarray

    def tracking_error(self, state):
        center = state['wrist_position'] + state['wrist_rotation'].apply(self.site)
        return np.linalg.norm(on_plane(self.center - center, self.normal))


def palm_headings(attitude, normal):
    axes = on_plane(attitude.as_matrix().T[:2], normal)
    along = normalize(axes[np.argmax(np.linalg.norm(axes, axis=1))])
    angles = np.arange(12)[:, None] * (np.pi / 6)
    ring = np.cos(angles) * along + np.sin(angles) * np.cross(normal, along)
    return np.vstack((np.zeros(3), ring))


class PalmDirectionGuide:
    def __init__(self, previous, reach, *, single_risk=None):
        self.model, self.patches = previous.model, previous.patches
        self.history = previous.history  # Used by the existing one-contact local support controller.
        self.site, self.reach = previous.sites[0], reach
        self.single_risk = single_risk
        self.region = self.palm_goal = self.direction = self.normal = None
        self.batch, self.last_tick, self.next_tick = 0, -GLOBAL_TICKS, 0
        self.holding = False
        self.hold_start_cells = 0
        self.unsuccessful_hold = None

    def evaluate(self, state, pivot, normal, headings, speed, scale):
        horizon = HORIZONS[scale]
        surface = state['surface']
        attitude, wrist = state['wrist_rotation'], state['wrist_position']
        center = wrist + attitude.apply(self.site)
        shift = headings * (speed * horizon)
        # Transport a whole-hand tangent step over the observed sheet. The local
        # controller still determines the executed normal height, yaw and inclination.
        support = pivot + shift
        hints = np.broadcast_to(normal, support.shape)
        patch = self.patches.query(self.model, surface, support, hints, .028)
        axes = np.cross(hints, patch.normals)
        angle = np.arctan2(np.linalg.norm(axes, axis=1), patch.normals @ normal)
        turn = normalize(axes) * np.minimum(angle, .65 * horizon)[:, None]
        rise = np.clip((patch.points - support) @ normal, -.012, .012)
        turn[~patch.valid] = 0.
        rise[~patch.valid] = 0.
        turn[0], rise[0] = 0., 0.  # The stationary comparator has no commanded motion.
        shift += rise[:, None] * normal
        sweep = np.array([.25, .50, .75, 1.])
        rotations = Rotation.from_rotvec((turn[:, None] * sweep[None, :, None]).reshape(-1, 3)) * attitude
        matrix = rotations.as_matrix().reshape(len(headings), len(sweep), 3, 3)
        centers = center + sweep[None, :, None] * shift[:, None]
        origins = centers - np.einsum('atij,j->ati', matrix, self.site)
        actions = np.c_[attitude.inv().apply(origins[:, -1] - wrist), attitude.inv().apply(turn)]
        pads_local = attitude.inv().apply(state['pads'] - wrist)
        local_normals = attitude.inv().apply(state['normals'])
        sites, contact_normals, light, weights = self.reach.predict(
            state['q'], state['hits'], local_normals, pads_local, actions, scale)
        world = origins[:, None, 1::2, None] + np.einsum('atij,ahtfj->ahtfi', matrix[:, 1::2], sites)
        facing = np.einsum('atij,ahtfj->ahtfi', matrix[:, 1::2], contact_normals)
        # A reachable location can lie beside two faces of a corner. Associate
        # the predicted contact with a nearby plane its material normal can touch.
        dist, near = surface.tree.query(world, k=np.arange(1, min(4, len(surface.points)) + 1))
        ns = surface.normals[near]
        gap = np.sum((world[..., None, :] - surface.points[near]) * ns, axis=-1)
        match = np.sum(facing[..., None, :] * ns, axis=-1)
        lateral2 = np.maximum(0., dist**2 - gap**2)
        confidence = np.exp(-.5 * ((gap / .008)**2 + lateral2 / .026**2))
        confidence *= np.clip((match - .25) / .55, 0., 1.) * (dist < .038)
        face = np.argmax(confidence, axis=-1, keepdims=True)
        projection = np.take_along_axis(gap[..., None] * ns, face[..., None], axis=-2)[..., 0, :]
        probability = light * confidence.max(axis=-1)
        gain, fresh = joint_gain(world - projection, probability, weights, surface, self.model)
        masks = probability[..., None, :] > MASK_THRESHOLDS[:, :4]
        count = masks.sum(axis=-1)
        pair = (weights * (count >= 2).mean(axis=(2, 3))).sum(axis=1)
        survival = (weights * (count > 0).all(axis=2).mean(axis=2)).sum(axis=1)
        pair_both = (weights * (count >= 2).all(axis=2).mean(axis=2)).sum(axis=1)
        geometry = attitude.inv().apply(state['palm_vertices'] - wrist)
        plate = origins[:, :, None] + np.einsum('atij,vj->atvi', matrix, geometry)
        legal = palm_clear(surface, state['palm_vertices'], plate)
        legal &= (np.linalg.norm(origins - state['workspace_center'], axis=-1) <= .45).all(axis=1)
        # A hold which just produced no new cells at this location has observed zero yield.
        hold_gain = gain[0] if self.unsuccessful_hold is None else 0.
        score = self.score_actions(gain, hold_gain, survival, pair_both, legal, horizon)
        audit = None
        if self.single_risk is not None and state['hits'].sum() == 1 and max(state['loads']) < LOAD_LIMIT:
            risk = self.single_risk.predict(state['q'], state['hits'], local_normals, actions, scale)
            unfiltered = int(np.argmax(score))
            allowed = risk <= self.single_risk.max_risk
            eligible = np.isfinite(score[1:])
            audit = dict(owner=int(np.flatnonzero(state['hits'])[0]), horizon_s=float(horizon),
                         unfiltered_sector=unfiltered, unfiltered_score=float(score[unfiltered]),
                         risk_by_sector=risk.tolist(), eligible_moves=int(eligible.sum()),
                         accepted_moves=int((eligible & allowed[1:]).sum()))
            # Index zero leaves the tangential goal stationary while local control adapts.
            score[1:] = np.where(allowed[1:], score[1:], -np.inf)
        k = int(np.argmax(score))
        result = dict(score=float(score[k]), gain=float(gain[k]), hold_gain=float(hold_gain), hold=not k,
                    hold_forecast=float(gain[0]), incremental=float(gain[k] - hold_gain) if k else 0.,
                    pair=float(pair[k]), survival=float(survival[k]), fresh=float(fresh[k]),
                    seconds=float(horizon), legal=int(legal[1:].sum()),
                    center=centers[k, -1], wrist=origins[k, -1], rotation=Rotation.from_matrix(matrix[k, -1]),
                    support=pivot + shift[k], normal=patch.normals[k] if patch.valid[k] else normal)
        if audit is not None:
            audit['sector'] = k
            result['single_support_risk'] = audit
        return result

    def score_actions(self, gain, hold_gain, survival, pair_both, legal, horizon):
        support_reference = survival[legal].max(initial=0.) if self.unsuccessful_hold is None else survival[1:][legal[1:]].max(initial=0.)
        supported = legal & (survival >= min(.70, .8 * support_reference))
        rate = (gain - hold_gain) * survival * (.25 + .75 * pair_both) / (horizon + .25)
        score = np.where(supported, rate, -np.inf)
        score[0] = 0.
        return score

    def choose(self, state, ids, normal, tick, event):
        att = state['wrist_rotation']
        center = state['wrist_position'] + att.apply(self.site)
        pivot = state['pads'][ids].mean(axis=0)
        headings = palm_headings(att, normal)
        speed = SPEED_M_S * (1. if len(ids) >= 2 else .55)
        options = [self.evaluate(state, pivot, normal, headings, speed, i) for i in range(len(HORIZONS))]
        best = max(options, key=lambda option: option['score'])
        if 'single_support_risk' in best:
            records = [option['single_support_risk'] for option in options]
            original = max(records, key=lambda record: record['unfiltered_score'])
            chosen = best['single_support_risk']
            event['single_support_risk'] = dict(owner=chosen['owner'], max_risk=self.single_risk.max_risk,
                    sector=chosen['sector'], horizon_s=chosen['horizon_s'],
                    predicted_loss=chosen['risk_by_sector'][chosen['sector']],
                    original_sector=original['unfiltered_sector'], original_horizon_s=original['horizon_s'],
                    options=records)
        self.holding = best['hold']
        self.hold_start_cells = len(state['surface'].points)
        self.region = PalmMove(best['center'], self.site, normal.copy())
        self.palm_goal = None
        if not self.holding:
            # Supply a terminal plate reference only when a measured sheet exists
            # under the winning palm. The local controller blends it by palm progress.
            n = best['normal']
            below = best['center'] - n * ((best['center'] - best['support']) @ n)
            under = self.patches.query(self.model, state['surface'], below, n, .044)
            height = (best['center'] - under.points[0]) @ best['rotation'].apply(OUT)
            if under.valid[0] and .018 <= height <= .18:
                self.palm_goal = Landing(self.region, best['wrist'], best['rotation'], under.points[0],
                                         under.normals[0], 1., self.site, float(height))
        self.next_tick = tick + round(best['seconds'] / DT)
        event.update(source='palm_hold' if self.holding else 'palm_reach_information',
                     palm_start_m=center.tolist(), palm_target_m=best['center'].tolist(), horizon_s=best['seconds'],
                     candidates=len(options) * (len(headings) - 1), legal_candidates=sum(o['legal'] for o in options),
                     acquisition_nat=best['gain'], stationary_gain_nat=best['hold_gain'],
                     stationary_forecast_nat=best['hold_forecast'], incremental_gain_nat=best['incremental'],
                     predicted_new_cells=best['fresh'],
                     predicted_light2=best['pair'], predicted_support_both_knots=best['survival'],
                     selected_rate_nat_s=best['score'])

    def update(self, surface, tick, pads, ids, normal, wrist_rotation):
        state = self.state
        center = state['wrist_position'] + wrist_rotation.apply(self.site)
        self.normal, self.direction = normal.copy(), None
        if self.unsuccessful_hold is not None:
            held_at, cells = self.unsuccessful_hold
            if len(surface.points) != cells or np.linalg.norm(center - held_at) > .008:
                self.unsuccessful_hold = None
        if not len(ids):
            self.region = self.palm_goal = None
            self.holding = False
        due = tick >= self.next_tick
        if self.region is not None and not self.holding:
            due |= self.region.tracking_error(state) < .006 or normal @ self.region.normal < .55
        plan = len(ids) and (self.region is None or due) and tick - self.last_tick >= 25
        event = None
        if surface.tree is not None and (plan or tick - self.last_tick >= GLOBAL_TICKS):
            old = self.model.count
            self.model.update(surface, tick)
            self.batch += 1
            self.last_tick = tick
            event = dict(batch=self.batch, observed_tick=tick, old_cells=old, total_cells=self.model.count,
                         new_cells=self.model.count-old, mesh_version=self.model.mesh_version,
                         mesh_faces=len(self.model.mesh.faces), direction_chosen=False)
            if len(ids):
                if self.holding and len(surface.points) == self.hold_start_cells:
                    self.unsuccessful_hold = center.copy(), self.hold_start_cells
                began = perf_counter()
                self.choose(state, ids, normal, tick, event)
                event['palm_selection_ms'] = 1000 * (perf_counter() - began)
        if len(ids) and self.region is not None:
            self.direction = normalize(on_plane(self.region.center - center, normal)) if not self.holding else np.zeros(3)
            if event is not None:
                event.update(direction_chosen=not self.holding, direction=self.direction.tolist(), normal=normal.tolist())
        return event


class PalmReachWrist(palm_unload.UnloadWrist):
    def __init__(self, *args, reach, guide_cls=PalmDirectionGuide, **options):
        super().__init__(*args, **options)
        follower = self.selector
        follower.guide = follower.base.guide = guide_cls(follower.guide, reach)
        follower.finger_reentry = False

    def observe(self, data, touch, tick):
        super().observe(data, touch, tick)
        # During a sustained contact drought reuse the measured wrist return
        # corridor. Reacquisition never installs a finger frontier as a global goal.
        if (self.in_drought and not touch.measured_heads.any() and not self.unloading and self.anchor is None
                and tick % 80 == 0 and tick - self.last_return >= 160):
            self.return_from_dead_end(tick, self.measured_position, self.live_attitude)
            if self.anchor is not None:
                self.began_return, self.start_cells = tick, len(self.surface.samples)


def run(folder, obj, seconds=240, device='cuda', prior=None, reach_path=policy_reach.MODEL, *, risk_path=None):
    reach = policy_reach.PolicyReach(reach_path)
    if reach.metadata['checkpoint'] != json.loads(policy_reach.PROTOCOL.read_text(encoding='utf-8'))['checkpoint']:
        raise ValueError('Reach model and deployed single_all checkpoint do not match')
    guide = PalmDirectionGuide
    sources = (__file__, policy_reach.__file__, reach_path)
    release = 'palm-reach-information'
    if risk_path is not None:
        from . import single_support_risk
        risk = single_support_risk.SingleSupportRisk(risk_path, reach_path)
        guide = partial(PalmDirectionGuide, single_risk=risk)
        sources += (single_support_risk.__file__, risk_path)
        release += '-single-support-risk'
    return palm_unload.run(folder, obj, seconds, device, prior,
                           wrist_cls=partial(PalmReachWrist, reach=reach, guide_cls=guide),
                           source_files=sources, release=release)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=OBJECTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=240)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--reach-model', type=Path, default=policy_reach.MODEL)
    parser.add_argument('--single-support-risk', type=Path, help='Optional fitted last-pad loss model')
    args = parser.parse_args()
    print(run(args.out, args.object, args.seconds, args.device, args.prior, args.reach_model,
              risk_path=args.single_support_risk), flush=True)
