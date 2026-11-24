"""Follow an information-selected curved contact strip, turning the wrist before the next patch."""
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import tangent, unit
from ..acitive_02.surface_scan import limit
from . import informative_run, multirate_wrist, online_tactile_mesh, posterior_route, strip_information
from .contact_first import signed_gap, palm_clear
from .experiment import digest
from .informative_wrist import InformativeWrist
from .multirate_wrist import LOCAL_S, MultirateWorker, contact_frame
from .online_tactile_mesh import ContactBand, StreamingGPIS
from .sweep import DT, LOAD_LIMIT, PLAN_TICKS
from .ycb_fixtures import OBJECTS


class PursuitActions:
    def __init__(self, touch, rotation, device):
        ids = np.flatnonzero(touch.measured_heads)
        normal = unit(touch.normals[ids].mean(0))
        self.global_model = StreamingGPIS(device, ContactBand(.95))
        self.guide = posterior_route.PosteriorRoute(self.global_model, touch.points, normal, rotation)
        self.hand_normal = rotation.inv().apply(normal)
        self.last_normal = normal
        self.tip_forces = np.zeros(4)

    def select(self, *, surface, hits, pads, normals, loads, wrist_position, wrist_rotation,
               palm_vertices, tick, observed_phase, **_):
        ids = np.flatnonzero(hits)
        row = self.guide.row
        pivot = pads[row].mean(0)
        live_n = self.last_normal
        if len(ids):
            same, _, live_n = contact_frame(pads, normals, ids, self.last_normal)
        light = len(ids) > 0 and max(loads) < LOAD_LIMIT and observed_phase == 'scan'
        start = perf_counter()
        event = self.guide.update(surface, tick, pads, live_n, wrist_rotation, light)
        elapsed = 1000*(perf_counter()-start) if event is not None else 0.
        info = dict(batch=self.guide.batch, global_event=event, global_ms=elapsed, local_period_s=LOCAL_S,
                    gpis_device=str(self.global_model.device), contact_iso=self.global_model.band.level,
                    tangent_speed_m_s=0., height_speed_m_s=0., angular_velocity=[0., 0., 0.],
                    mode='withdraw' if max(loads) >= LOAD_LIMIT or observed_phase == 'withdraw' else 'recontact')
        result = dict(position=wrist_position.copy(), rotation=wrist_rotation, normal=live_n,
                      direction=np.zeros(3), diagnostics=info)
        if not light or self.guide.path is None:
            return result
        self.last_normal = live_n
        lead = .031 if len(ids) >= 2 else .017
        target, future_n = self.guide.waypoint(pivot, lead)
        patch = surface.patch(target, preferred_normal=future_n)
        if patch is not None and surface.tree.query(target)[0] < .022:
            gap, n = signed_gap(patch, target)
            affinity = np.clip((n@future_n-.30)/.55, 0., 1.)
            target = target-affinity*np.clip(gap, -.009, .009)*n
            future_n = unit(future_n+.85*affinity*n)
        hand_n = wrist_rotation.apply(self.hand_normal)
        # The leading posture rolls the pad row round an edge. Tilt and height
        # are corrected against the real local contacts while the route is held.
        desired_n = unit(.70*future_n+.30*live_n)
        angular = 5.*np.cross(hand_n, desired_n)
        moment = np.cross(pads-pivot, live_n).T@np.minimum(self.tip_forces, 5.)
        angular += 3.*moment
        displacement = target-pivot
        speed = .094 if len(ids)>=2 else .044
        along = limit(3.4*tangent(displacement, live_n), speed)
        across = unit(np.cross(live_n, along))
        row_axis = unit(tangent(pads[row[-1]]-pads[row[0]], live_n))
        if row_axis@across < 0:
            across *= -1
        angular += np.clip(np.cross(row_axis, across)@live_n, -.32, .32)*live_n
        angular = limit(angular, 1.15 if len(ids)>=2 else .72)
        # Hold measured pads near their signed local surfaces. Penalize predicted
        # gap change at the supported footprint; no fingertip joint target is sent.
        turn = Rotation.from_rotvec(LOCAL_S*angular)
        center = pads[same].mean(0)
        dz, weights = [], []
        for i in same:
            patch = surface.patch(pads[i], preferred_normal=normals[i])
            if patch is None:
                continue
            base = patch.predict(pads[i])[0][0]
            probe = center+LOCAL_S*along+turn.apply(pads[i]-center)
            gap, n = signed_gap(patch, probe, base)
            if n@live_n > .40:
                dz.append(-gap/(n@live_n))
                weights.append(1./(.25+abs(self.tip_forces[i]-1.)))
        height = np.clip(np.average(dz, weights=weights)/LOCAL_S, -.018, .022) if dz else 0.
        if max(loads)>=4. or loads[-1]>=.35:
            height = max(0., height)
        center_v = along+height*live_n
        hull = center+LOCAL_S*center_v+turn.apply(palm_vertices-center)
        if not palm_clear(surface, palm_vertices, hull[None, None])[0]:
            center_v = .40*along+.016*live_n
            angular *= .45
            turn = Rotation.from_rotvec(LOCAL_S*angular)
        info.update(mode='explore', pivot_velocity=center_v.tolist(), pivot_ids=same.tolist(),
                    tangent_speed_m_s=float(np.linalg.norm(along)), height_speed_m_s=float(height),
                    angular_velocity=angular.tolist(), target_m=target.tolist(), target_normal=future_n.tolist(),
                    route_index=self.guide.index)
        result.update(position=center+LOCAL_S*center_v+turn.apply(wrist_position-center),
                      rotation=turn*wrist_rotation, direction=unit(along))
        return result


class PursuitWrist(InformativeWrist):
    def __init__(self, hand, data, touch, *, device, **options):
        super().__init__(hand, data, touch, **options)
        self.selector = PursuitActions(touch, self.attitude, device)
        self.flow = None

    def observe(self, data, touch, tick):
        super().observe(data, touch, tick)
        if not self.unloading and touch.loads.max() >= LOAD_LIMIT:
            i = int(np.argmax(touch.loads))
            self.relief = touch.normals[i] if i < len(touch.measured_heads) and touch.measured_heads[i] else self.normal
            self.unloading, self.rescue, self.clear_ticks, self.good_ticks = True, None, 0, 0
            self.escape_count += 1

    def command(self, touch):
        if self.flow is None or self.phase() != 'scan' or not touch.measured_heads.any():
            self.goal, self.goal_attitude = self.position.copy(), self.attitude
            return super().command(touch)
        row, frame_normal = self.flow
        ids = np.array(row['pivot_ids'], dtype=int)
        ids = ids[touch.measured_heads[ids]]
        if not len(ids):
            ids = np.flatnonzero(touch.measured_heads)
        _, pivot, n = contact_frame(touch.points, touch.normals, ids, frame_normal)
        v = np.asarray(row['pivot_velocity'])
        omega = np.asarray(row['angular_velocity'])
        lever = self.measured_position-pivot
        remain = max(1, self.apply_tick+PLAN_TICKS-self.tick)*DT
        wrist_v = v+(Rotation.from_rotvec(DT*omega).apply(lever)-lever)/DT
        self.goal = self.position+remain*wrist_v
        self.goal_attitude = Rotation.from_rotvec(remain*omega)*self.attitude
        self.normal = n
        return super().command(touch)


def run(folder, obj, seconds, device):
    modules = [multirate_wrist, online_tactile_mesh, posterior_route, strip_information]
    sources = informative_run.source_hashes()
    for path in [Path(__file__), Path(__file__).with_name('contact_first.py'), *(Path(m.__file__) for m in modules)]:
        sources[path.name] = digest(path)
    with patch.multiple(informative_run, InformativeWrist=partial(PursuitWrist, device=device),
                        InformativeWorker=MultirateWorker, RELEASE=f'posterior-route-{device}-global400-local5',
                        source_hashes=lambda: sources):
        return informative_run.run(folder, obj, seconds)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=OBJECTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=240.)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    print(run(args.out, args.object, args.seconds, args.device), flush=True)
