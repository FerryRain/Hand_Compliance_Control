"""Collection-only matched contact readout, independent of sensor slot sorting.

The installed MJWarp maxforce sorter reorders contact IDs without reordering
primary/secondary signs. Read the selected geom pair and its frame together.
No force/normal is smoothed or flipped to agree with a desired motion.
The physics solver and other deployment sensors are unchanged.
"""
import mujoco
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import ContactType, vec5
from collection_force_regulation import CollectionForceObserver
from split_fingertip_collision import GEOMS


@wp.kernel
def matched_contacts(
    cone: int, count: wp.array[int], world: wp.array[int],
    geom: wp.array[wp.vec2i], kind: wp.array[int],
    frame: wp.array[wp.mat33], pos: wp.array[wp.vec3], dist: wp.array[float],
    friction: wp.array[vec5], dim: wp.array[int], address: wp.array2d[int],
    efc: wp.array2d[float], njmax: int,
    roles: wp.array[int], target: wp.array[int], hand: wp.array[int],
    found: wp.array2d[int], ids: wp.array2d[int], pairs: wp.array2d[wp.vec2i],
    force: wp.array2d[wp.vec3], normal: wp.array2d[wp.vec3],
    tangent: wp.array2d[wp.vec3], point: wp.array2d[wp.vec3], distance: wp.array2d[float],
    net: wp.array2d[wp.vec3], centroid: wp.array2d[wp.vec3]):
    lane, role = wp.tid()
    best = int(-1)
    best_sign = float(1.0)
    best_mag = float(-1.0)
    best_force = wp.vec3(0.0)
    matches = int(0)
    total = wp.vec3(0.0)
    weighted_point = wp.vec3(0.0)
    weights = float(0.0)
    for c in range(count[0]):
        if world[c] != lane or not (kind[c] & int(ContactType.CONSTRAINT)):
            continue
        a, b = geom[c][0], geom[c][1]
        if a < 0 or b < 0:
            continue
        sign = float(0.0)
        if (roles[a] == role or (role == 8 and hand[a] > 0)) and target[b] > 0:
            sign = 1.0
        elif (roles[b] == role or (role == 8 and hand[b] > 0)) and target[a] > 0:
            sign = -1.0
        if sign == 0.0:
            continue
        wrench = contact_force_fn(cone, frame, friction, dim, address, efc,
                                  njmax, count, lane, c, False)
        local_force = wp.spatial_top(wrench)
        magnitude = wp.length(local_force)
        # Canonical force-on-object convention, matching the old net sensor.
        total += sign * (wp.transpose(frame[c]) * local_force)
        weighted_point += magnitude * pos[c]
        weights += magnitude
        matches += 1
        if magnitude > best_mag:
            best = c
            best_mag = magnitude
            best_sign = sign
            best_force = wp.vec3(local_force[0], local_force[1], sign * local_force[2])
    found[lane, role] = matches
    ids[lane, role] = best
    net[lane, role] = total
    centroid[lane, role] = weighted_point / wp.max(weights, 1.e-12)
    pairs[lane, role] = wp.vec2i(-1, -1)
    force[lane, role] = wp.vec3(0.0)
    normal[lane, role] = wp.vec3(0.0)
    tangent[lane, role] = wp.vec3(0.0)
    point[lane, role] = wp.vec3(0.0)
    distance[lane, role] = 0.0
    if best >= 0:
        pairs[lane, role] = geom[best]
        force[lane, role] = best_force
        normal[lane, role] = best_sign * frame[best][0]
        tangent[lane, role] = best_sign * frame[best][1]
        point[lane, role] = pos[best]
        distance[lane, role] = dist[best]


class MatchedCollectionObserver(CollectionForceObserver):
    """Retains legacy readings for audit, feeds raw matched readings to MCC."""
    def _bind_raw(self, env):
        roles = np.full(env.sim.mj_model.ngeom, -1, np.int32)
        target = np.zeros_like(roles)
        hand = np.zeros_like(roles)
        names = {}
        for g in range(len(roles)):
            name = mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            names[g] = name
            target[g] = name.startswith('target/')
            hand[g] = name.startswith('robot/')
            for offset, part in ((0, 'pad'), (4, 'connector')):
                for f, base in enumerate(GEOMS):
                    if name.endswith('/'+base+'_'+part+'_collision'):
                        roles[g] = offset+f
        if set(roles[roles >= 0]) != set(range(8)):
            raise ValueError('Expected four split pads and four connectors')
        self.geom_names = names
        self.raw = {}
        with wp.ScopedDevice(env.sim.wp_device):
            self.roles = wp.array(roles, dtype=int)
            self.targets = wp.array(target, dtype=int)
            self.hand = wp.array(hand, dtype=int)
            for key, dtype in (('found',int),('ids',int),('pairs',wp.vec2i),
                               ('force',wp.vec3),('normal',wp.vec3),('tangent',wp.vec3),
                               ('point',wp.vec3),('distance',float),('net',wp.vec3),('centroid',wp.vec3)):
                self.raw[key] = wp.zeros((env.num_envs,9),dtype=dtype)

    def read(self, env, control_entries):
        super().read(env, control_entries)
        if not hasattr(self, 'raw'):
            self._bind_raw(env)
        d=env.sim.wp_data
        # No silently truncated contact/constraint buffers are valid labels.
        if int(wp.to_torch(d.nacon)[0]) >= d.naconmax or bool((wp.to_torch(d.nefc) >= d.njmax).any()):
            raise RuntimeError('physics_buffer_capacity_exhausted')
        stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with wp.ScopedDevice(env.sim.wp_device):
            wp.launch(matched_contacts,dim=(env.num_envs,9),inputs=[
                env.sim.wp_model.opt.cone,d.nacon,d.contact.worldid,d.contact.geom,d.contact.type,
                d.contact.frame,d.contact.pos,d.contact.dist,d.contact.friction,d.contact.dim,
                d.contact.efc_address,d.efc.force,d.njmax,self.roles,self.targets,self.hand],
                outputs=list(self.raw.values()))
        torch.cuda.current_stream().wait_stream(stream)
        values={k:wp.to_torch(v).clone() for k,v in self.raw.items()}
        for offset, role in ((0,'pad'),(4,'connector')):
            s=slice(offset,offset+4)
            legacy_n=self.latest[role+'_physical_inward']
            legacy_f=self.latest[role+'_force']
            self.latest[role+'_legacy_normal_dot_raw']=(legacy_n*values['normal'][:,s]).sum(-1)
            self.latest[role+'_legacy_net_error_n']=(legacy_f-values['net'][:,s]).norm(dim=-1)
            self.latest.update({
                role+'_found':values['found'][:,s]>0,
                role+'_force':values['net'][:,s],
                role+'_points':values['centroid'][:,s],
                role+'_physical_found':values['found'][:,s]>0,
                role+'_physical_inward':values['normal'][:,s],
                role+'_physical_tangent':values['tangent'][:,s],
                role+'_physical_force':values['force'][:,s],
                role+'_physical_points':values['point'][:,s],
                role+'_physical_geom_ids':values['pairs'][:,s],
                role+'_physical_contact_id':values['ids'][:,s],
                role+'_physical_distance':values['distance'][:,s]})
        self.safety_peak_force_n=torch.maximum(
            values['net'][:,:8].norm(dim=-1).amax(-1),
            (values['net'][:,:4]+values['net'][:,4:8]).norm(dim=-1).amax(-1))
        self.latest.update(object_physical_geom_ids=values['pairs'][:,8],
            object_physical_force=values['force'][:,8],object_physical_point=values['point'][:,8],
            object_physical_normal=values['normal'][:,8],object_physical_distance=values['distance'][:,8])
