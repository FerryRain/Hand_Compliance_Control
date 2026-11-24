"""Unload a stationary, stressed hand using measured contact sides and the tactile map."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion
from . import palm_support as support
from .palm_surface import normalize
from .sweep import DT
from .ycb_fixtures import OBJECTS


class UnloadWrist(support.SupportWrist):
    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.pressure_since = None
        self.recent_wrist = deque(maxlen=28)
        self.release_velocity = None
        self.attempted = deque(maxlen=5)
        self.last_test = -1000
        self.peak_at_test = 0.
        self.cap_surfaces = []
        m = self.hand.model
        for geom in self.hand.pads:
            mesh = m.geom_dataid[geom]
            start = m.mesh_vertadr[mesh]
            verts = m.mesh_vert[start:start + m.mesh_vertnum[mesh]]
            # Include the sides of the pad collision solid. Only its distal
            # portion reports an eligible point, but the entire solid loads the finger.
            self.cap_surfaces.append(verts[::max(1,len(verts)//140)].astype(float))

    def observe(self, data, touch, tick):
        super().observe(data, touch, tick)
        peak = float(touch.loads.max())
        self.recent_wrist.append(self.measured_position.copy())
        if peak < 4. or (peak < 8. and self.release_velocity is None):
            self.pressure_since = self.release_velocity = None
            self.attempted.clear()
            return
        if self.pressure_since is None:
            self.pressure_since = tick
        if tick - self.pressure_since < 85:
            return
        if self.release_velocity is None and np.linalg.norm(self.measured_position - self.recent_wrist[0]) > .0025:
            return
        if self.release_velocity is not None and tick - self.last_test < 18:
            return
        if self.release_velocity is not None and peak > .88 * self.peak_at_test:
            self.attempted.append(normalize(self.release_velocity[0]))
        region = int(np.argmax(touch.loads))
        current = self.measured_position
        witnesses, normals, weights = [], [], []
        surface = self.surface
        if region < 4 and surface.tree is not None:
            h = self.link_hull
            select = h.regions == region
            ids = h.ids[select]
            links = data.geom_xpos[ids] + np.einsum('nij,nj->ni', data.geom_xmat[ids].reshape(-1,3,3), h.vertices[select])
            cap = self.origins[region] + self.cap_surfaces[region] @ self.matrices[region].T
            vertices = np.vstack([cap, links])
            dist, near = surface.tree.query(vertices)
            used = []
            for k in np.argsort(dist):
                if dist[k] > .048 or len(used) >= 3:
                    break
                n = surface.normals[near[k]]
                if any(n @ old > .82 for old in used):
                    continue
                used.append(n)
                witnesses.append(vertices[k])
                normals.append(n)
                weights.append(1.3 * np.exp(-.5*(dist[k]/.024)**2))
        for f in np.flatnonzero(touch.measured_heads):
            witnesses.append(touch.points[f])
            normals.append(touch.normals[f])
            weights.append(1.5 if f == region else .60)
        if not normals:
            for _, prior, _ in reversed(self.low_load_history):
                delta = prior-current
                if np.linalg.norm(delta) > .005:
                    witnesses.append(current)
                    normals.append(normalize(delta))
                    weights.append(.65)
                    break
        if not normals:
            return
        normals = np.asarray(normals)
        w = np.asarray(weights)
        offsets = np.asarray(witnesses) - current
        torques = np.cross(offsets,normals)
        desired = normalize(np.vstack([np.average(normals,axis=0,weights=w), normals]))
        rates = []
        for axis in desired:
            aligned = w * np.maximum(normals @ axis, 0.)**2
            if aligned.sum() < 1.e-8:
                continue
            omega = limit(.75 * np.average(torques,axis=0,weights=aligned)/.11, .70)
            velocity = .080 * axis
            # Evaluate what the bounded workspace actually permits, including a
            # rotation that can unload a link with little wrist translation.
            attainable = self.workspace_center + limit(current + .18*velocity - self.workspace_center,.448)
            velocity = limit((attainable-current)/.18, .085)
            opening = normals @ velocity + torques @ omega
            merit = np.average(opening,weights=w) + .32 * np.min(opening)
            if self.attempted:
                merit -= .050 * max(0., np.max(np.asarray(self.attempted) @ normalize(velocity)))**3
            rates.append((merit,velocity,omega))
        if rates:
            _, velocity, omega = max(rates,key=lambda r:r[0])
            self.release_velocity = velocity,omega
            self.last_test,self.peak_at_test = tick,peak

    def command(self,touch):
        if self.release_velocity is None:
            return super().command(touch)
        velocity,omega = self.release_velocity
        self.position = self.workspace_center + limit(self.position + DT*velocity-self.workspace_center,.45)
        self.attitude = Rotation.from_rotvec(DT*omega)*self.attitude
        cmd = np.r_[self.position,quaternion(self.attitude)]
        return cmd,cmd.copy()


class UnloadWorker(support.SupportWorker):
    def install(self,tick,*,initial=False):
        row = super().install(tick,initial=initial)
        if row is not None and self.servo.release_velocity is not None:
            v,omega = self.servo.release_velocity
            row.update(stationary_load_relief=True,release_m_s=v.tolist(),release_rad_s=omega.tolist())
        return row


def run(folder,obj,seconds=240,device='cuda',prior=None):
    l = support.landing
    first, prev = l.first,l.first.previous
    return l.field.run(folder,obj,seconds,device,prior,wrist_cls=UnloadWrist,worker_cls=UnloadWorker,
        source_files=(__file__,support.__file__,support.paired_support.__file__,l.__file__,first.__file__,
                      first.palm_acquisition.__file__,first.palm_surface.__file__,prev.__file__,prev.ct.__file__,
                      prev.ct.entry_frontier.__file__,prev.ct.transition_compute.__file__,prev.ct.delivery.__file__,
                      prev.ct.tactile_reentry.__file__,prev.ct.tactile_stall.__file__),release='mesh-palm-contact-load-feedback')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',choices=OBJECTS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=240)
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,prior=args.prior),flush=True)
