"""Reconstruct sensor loads using native MuJoCo C, not Warp's force helper."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import mujoco
import numpy as np


class NativeContactDecoder:
    def __init__(self):
        library = next(Path(mujoco.__file__).parent.glob('libmujoco.so*'))
        self.library = ctypes.CDLL(str(library))
        self.pointer = ctypes.POINTER(ctypes.c_double)
        self.decode = self.library.mju_decodePyramid
        self.decode.argtypes = [self.pointer, self.pointer, self.pointer, ctypes.c_int]
        self.decode.restype = None

    def force(self, cone, dim, addresses, friction, constraint_force):
        force = np.zeros(6, dtype=np.float64)
        addresses = np.asarray(addresses).reshape(-1)
        first = int(addresses[0])
        if first < 0:
            return force
        if int(cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
            edges = np.zeros(10, dtype=np.float64)
            n = max(1, 2*(dim-1))
            if first+n > len(constraint_force):
                raise ValueError('Invalid pyramid constraint address')
            edges[:n] = constraint_force[first:first+n]
            mu = np.ascontiguousarray(friction, dtype=np.float64)
            self.decode(force.ctypes.data_as(self.pointer), edges.ctypes.data_as(self.pointer),
                        mu.ctypes.data_as(self.pointer), int(dim))
        else:
            for j, address in enumerate(addresses[:dim]):
                if address >= 0:
                    force[j] = constraint_force[int(address)]
        return force


def stop_load(pad, connector):
    """Same declared criterion, independently evaluated after reconstruction."""
    components = np.stack((np.linalg.norm(pad, axis=-1),
        np.linalg.norm(connector, axis=-1), np.linalg.norm(pad+connector, axis=-1)), -1)
    return components.max(axis=(-1, -2)), components


class IndependentContactForceAudit:
    def __init__(self, stride=20):
        self.stride = stride
        self.decoder = NativeContactDecoder()
        self.samples = []
        self.max_error = 0.
        self.max_stream_error = 0.
        self.failures = []
        self.events = []
        self.peak = 0.
        self.last_threshold = 0
        self.previously_overforce = False
        self.map = None
        self.object_step_mm = None
        self.previous_object_position = None

    def begin_tick(self, env):
        pose = env.sim.data.mocap_pos.detach().cpu().numpy().copy()
        mid = int(env.scene['target'].data.indexing.mocap_id)
        position = pose[:, mid]
        self.object_step_mm = (np.zeros(env.num_envs) if self.previous_object_position is None else
            1000*np.linalg.norm(position-self.previous_object_position, axis=-1))
        self.previous_object_position = position.copy()

    def observe(self, env, observer, step, substep):
        if step < 1200:
            return
        import warp as wp
        observed_peak = float(observer.safety_peak_force_n.max().detach().cpu())
        onset = observed_peak>100 and not self.previously_overforce
        self.previously_overforce = observed_peak>100
        threshold = 3 if observed_peak>100 else 2 if observed_peak>50 else 1 if observed_peak>10 else 0
        peak_event = (observed_peak > max(60., self.peak+10.)
                      or threshold > self.last_threshold or onset)
        if not peak_event and not (step % self.stride == 0 and substep == env.cfg.decimation):
            return
        self.peak = max(self.peak, observed_peak)
        self.last_threshold = max(self.last_threshold, threshold)
        early = {k:observer.latest[k].detach().cpu().numpy().copy()
                 for k in ('pad_force', 'connector_force')}
        wp.synchronize_stream(wp.get_stream(env.sim.wp_device))
        fenced = {k:observer.latest[k].detach().cpu().numpy().copy() for k in early}
        self.max_stream_error = max(self.max_stream_error,
            max(float(np.max(np.abs(early[k]-fenced[k]))) for k in early))
        d = env.sim.wp_data; model = env.sim.mj_model
        if self.map is None:
            from split_fingertip_collision import GEOMS
            names = [model.geom(g).name for g in range(model.ngeom)]
            mapping = {}
            for g, name in enumerate(names):
                for f, base in enumerate(GEOMS):
                    for role in ('pad', 'connector'):
                        if name.endswith(base+'_'+role+'_collision'):
                            mapping[g] = (role, f)
            self.map = dict(names=names, mapping=mapping,
                            objects={g for g,n in enumerate(names) if n.startswith('target/')})
        count = int(d.nacon.numpy()[0])
        geom = d.contact.geom.numpy()[:count]
        world = d.contact.worldid.numpy()[:count]
        frame = d.contact.frame.numpy()[:count]
        friction = d.contact.friction.numpy()[:count]
        dim = d.contact.dim.numpy()[:count]
        addresses = d.contact.efc_address.numpy()[:count]
        efc = d.efc.force.numpy()
        position = d.contact.pos.numpy()[:count]
        distance = d.contact.dist.numpy()[:count]
        reconstructed = {role:np.zeros((env.num_envs,4,3)) for role in ('pad', 'connector')}
        details = []
        for c in range(count):
            a,b = map(int, geom[c]); sign = 1.
            if a in self.map['mapping'] and b in self.map['objects']:
                hand, obj = a,b
            elif b in self.map['mapping'] and a in self.map['objects']:
                hand, obj = b,a; sign = -1.
            else:
                continue
            lane = int(world[c]); role,f = self.map['mapping'][hand]
            local = self.decoder.force(int(model.opt.cone), int(dim[c]), addresses[c],
                                       friction[c], efc[lane])
            vector = sign*(frame[c].T @ local[:3])
            reconstructed[role][lane,f] += vector
            if np.linalg.norm(vector)>.01:
                details.append(dict(lane=lane, finger=f, role=role, contact_id=c,
                    hand_geom=self.map['names'][hand], object_geom=self.map['names'][obj],
                    force_world=vector.tolist(), force_local=local.tolist(),
                    magnitude_n=float(np.linalg.norm(vector)), position_world=position[c].tolist(),
                    normal_outward_world=(-sign*frame[c,0]).tolist(),
                    penetration_mm=float(max(0., -distance[c])*1000),
                    constraint_addresses=addresses[c,:max(1,int(dim[c]))].tolist()))
        peak, components = stop_load(reconstructed['pad'], reconstructed['connector'])
        errors = {role:float(np.max(np.abs(reconstructed[role]-fenced[role+'_force'])))
                  for role in reconstructed}
        error = max(errors.values()); self.max_error = max(self.max_error, error)
        observed = observer.safety_peak_force_n.detach().cpu().numpy().copy()
        peak_error = float(np.max(np.abs(peak-observed)))
        tolerance = .002+3.e-5*float(max(peak.max(),observed.max()))
        passed = bool(error<=tolerance and peak_error<=2*tolerance)
        row = dict(control_step=step, task_frame=step-1200, physical_substep=substep,
            contact_count=count, vector_errors_n=errors, stop_reference_n=peak.tolist(),
            stop_observed_n=observed.tolist(), stop_error_n=peak_error,
            tolerance_n=tolerance, passed=passed,
            stop_decision_equal=bool(np.array_equal(peak>100, observed>100)))
        self.samples.append(row)
        if not passed or not row['stop_decision_equal']:
            self.failures.append(row)
        if (peak_event and len(self.events)<400) or not passed:
            self.events.append(dict(**row,
                components_pad_connector_combined_n=components.tolist(),
                object_command_step_mm=self.object_step_mm.tolist(),
                contacts=sorted(details,key=lambda x:-x['magnitude_n']),
                body_positions_world=d.xpos.numpy().tolist(),
                body_rotations_world=d.xmat.numpy().tolist(),
                actuator_force=d.actuator_force.numpy().tolist()))

    def write(self, output):
        result = dict(method='Native libmujoco mju_decodePyramid with explicit condim / raw elliptic constraint rows; CPU frame transform and per-geom/world vector sum',
            shared_warp_force_helper=False, samples=self.samples, failures=self.failures,
            max_vector_error_n=self.max_error, max_pre_post_stream_fence_difference_n=self.max_stream_error,
            events=self.events, passed=bool(self.samples and not self.failures),
            detailed_event_snapshot_limit=400,
            caveat='Checks force extraction/classification/aggregation/stop criterion. Physical validity of contact/actuator parameters requires separate tests.')
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
        return result
