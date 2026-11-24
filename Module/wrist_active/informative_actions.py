"""Select a wrist motion by expected information at the responding fingertips.

All coordinates are in the fixture frame, with positions in metres. The only
object geometry supplied to the acquisition is the on-trial tactile map.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import NOISE_VARIANCE, VOXEL_M
from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import align, limit
from .history import GENERATED, HandHistory, normal_spread
from .global_gpis import ANGLES, STROKE_M, GlobalGPIS, representatives, tangents


HORIZON = 0.50
EXECUTE = 0.20
SWEEP = STROKE_M
HISTORY = GENERATED / "gpis_wrist_active_20261121/hand-history.npz"
FRACTIONS = np.array([0.5, 1.0])
# Four deterministic, stratified joint contact-mask samples for each complete
# empirical hand response. Use identical samples across competing actions.
MASK_THRESHOLDS = ((np.arange(4)[:, None] + np.array([0, 1, 2, 3, 2, 3, 0, 1])) % 4 + 0.5) / 4


def basis(normal):
    first = unit(np.cross(np.eye(3)[np.argmin(abs(normal))], normal))
    return np.stack([first, np.cross(normal, first)])


class IndexedHandHistory(HandHistory):
    """Exact six neighbours in HandHistory's normalized state/action metric."""
    def __init__(self, path):
        super().__init__(path)
        self.scale = np.r_[np.full(self.q.shape[1], .35 * np.sqrt(self.q.shape[1])),
                           np.full(4, np.sqrt(4. / .7)), .35,
                           np.sqrt(6) * np.array([.010] * 3 + [.085] * 3)]
        self.index = cKDTree(np.column_stack([self.q, self.hits, self.normal_spread, self.action]) / self.scale)

    def predict(self, q, hits, spread, actions):
        state = np.broadcast_to(np.r_[q, hits, spread], (len(actions), len(q) + 5))
        distance, near = self.index.query(np.column_stack([state, actions]) / self.scale, k=6)
        square = distance**2
        weight = np.exp(-(square - square[:, :1]) / .35)
        weight /= weight.sum(axis=1, keepdims=True)
        return self.pad_delta[near], self.low_hits[near], weight


def joint_gain(projected, probability, weights, surface, model):
    """Expected I(f(Q); y_C | D) over empirical joint hand responses.

    C is represented at the evaluation voxel resolution. Computing the two
    determinants on C keeps the expensive part at most eight by eight.
    """
    count, examples = weights.shape
    cell = np.floor(projected.reshape(count, examples, 8, 3) / VOXEL_M).astype(np.int64)
    # First sample stored in surface.samples may be off-centre in its cell:
    # exact key membership is required; the Euclidean nearest need not own it.
    width = 2**20
    def keys(index):
        return (index[..., 0] * width + index[..., 1]) * width + index[..., 2]
    existing = getattr(surface, "_information_cell_cache", None)
    if existing is None or existing[0] is not surface.points:
        existing = (surface.points, np.sort(keys(np.floor(surface.points / VOXEL_M).astype(np.int64))))
        surface._information_cell_cache = existing
    hashed = keys(cell)
    lookup = np.searchsorted(existing[1], hashed).clip(0, len(existing[1])-1)
    unseen = existing[1][lookup] != hashed
    present = (probability.reshape(count, examples, 1, 8) > MASK_THRESHOLDS) & unseen[:, :, None]
    repeated = np.tril(hashed[..., :, None] == hashed[..., None, :], k=-1)
    present &= ~np.any(repeated[:, :, None] & present[..., None, :], axis=-1)
    active = present.any(axis=2)
    if not active.any():
        return np.zeros(count), np.zeros(count)
    unique, inverse = np.unique(cell[active], axis=0, return_inverse=True)
    sites = (unique + .5) * VOXEL_M
    prior, conditional = model.observation_covariances(sites)
    ids = np.zeros(active.shape, dtype=int)
    ids[active] = inverse
    pair = (ids[..., :, None], ids[..., None, :])
    keep = present[..., :, None] & present[..., None, :]
    def logdet(cov):
        sign, value = np.linalg.slogdet(np.eye(8) + cov[pair][:, :, None] * keep / NOISE_VARIANCE)
        if np.any(sign <= 0):
            raise FloatingPointError("Invalid conditional tactile covariance")
        return value
    information = .5 * (logdet(prior) - logdet(conditional))
    gain = np.sum(weights * np.maximum(0., information).mean(axis=2), axis=1)
    new_count = np.sum(weights * present.sum(axis=-1).mean(axis=2), axis=1)
    return gain, new_count


class InformativeActions:
    def __init__(self, history):
        self.history = history
        self.global_model = GlobalGPIS()
        self.graph_version = -1
        self.graph = None

    @classmethod
    def from_default_history(cls):
        return cls(IndexedHandHistory(HISTORY))

    def remote_frontiers(self, surface, center, normal, tick):
        """Propose remote boundary strokes reachable on a measured-neighbour graph."""
        if len(surface.points) < 30:
            return []
        if self.graph_version != self.global_model.version:
            ids = representatives(surface.points, 96)
            points, normals = surface.points[ids], surface.normals[ids]
            tree = cKDTree(points)
            pair = tree.query_pairs(.045, output_type="ndarray")
            i, j = pair.T
            chord = points[j] - points[i]
            length = np.linalg.norm(chord, axis=1)
            allowed = ((normals[i] * normals[j]).sum(axis=1) > .15) & (
                np.maximum(abs((chord*normals[i]).sum(axis=1)), abs((chord*normals[j]).sum(axis=1))) < .75*length)
            i, j = i[allowed], j[allowed]
            graph = coo_matrix((np.linalg.norm(points[i]-points[j], axis=1), (i,j)),
                               shape=(len(points), len(points))).tocsr()
            dirs = tangents(normals)
            proposed = points[:,None] + SWEEP * dirs
            projected, supported = self.global_model.project(proposed, normals)
            missing = surface.tree.query(projected.reshape(-1,3))[0].reshape(-1,8)
            valid = supported & (missing > .006) & (missing < .036)
            merit = np.zeros_like(missing)
            if valid.any():
                merit[valid] = self.global_model.regional_gain(projected[valid])
            best = merit.argmax(axis=1)
            rows = np.arange(len(points))
            self.graph = (points, normals, tree, graph, dirs[rows,best], merit[rows,best])
            self.graph_version = self.global_model.version
        points,normals,tree,graph,dirs,information = self.graph
        source = int(tree.query(center)[1])
        if np.linalg.norm(points[source]-center) > .060:
            return []
        travel, prev = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
        viable = (travel > .038) & (travel < .32) & (information > 0.)
        merit = np.where(viable, information / (HORIZON + travel/.065), -np.inf)
        result = []
        for _ in range(3):
            k = int(merit.argmax())
            if not np.isfinite(merit[k]):
                break
            waypoint = k
            while prev[waypoint] >= 0 and travel[prev[waypoint]] > .025:
                waypoint = prev[waypoint]
            heading = points[waypoint] - center
            heading -= (heading @ normal) * normal
            if np.linalg.norm(heading) > .002:
                result.append((points[k], normals[k], dirs[k], unit(heading), travel[k]))
            merit[np.linalg.norm(points - points[k], axis=1) < .045] = -np.inf
        return result

    def select(self, *, surface, q, hits, normals, wrist_position, wrist_rotation,
               pads, palm_vertices, loads, tick, workspace_center=None, observed_phase="scan"):
        hits = np.asarray(hits, dtype=bool)
        pads, normals, wrist_position = map(np.asarray, (pads,normals,wrist_position))
        # Use the observed mean outward direction and tactile support centroid.
        center = np.mean(pads[hits],axis=0) if hits.any() else np.mean(pads,axis=0)
        if hits.any():
            normal = unit(normals[hits].sum(axis=0))
        elif surface.tree is not None:
            normal = surface.normals[surface.tree.query(center)[1]]
        else:
            normal = unit(wrist_position-center)
        if np.linalg.norm(normal) < .5:
            normal = unit(wrist_position-center)
        if surface.tree is None or not hits.any() or observed_phase != "scan":
            result = self.recontact(wrist_position,wrist_rotation,normal,loads,0)
            if observed_phase == "withdraw":
                result["diagnostics"]["mode"] = "withdraw"
            return result

        self.global_model.update(surface, tick)
        axes = basis(normal)
        directions = np.cos(ANGLES)[:,None]*axes[0] + np.sin(ANGLES)[:,None]*axes[1]
        # Locally measured change of normal along a tangent estimates curvature.
        ids = surface.tree.query(center,k=min(40,len(surface.points)))[1]
        near_points = surface.points[np.atleast_1d(ids)]
        near_normals = surface.normals[np.atleast_1d(ids)]
        keep = (near_normals@normal > .25) & (np.linalg.norm(near_points-center,axis=1)<.080)
        xy = (near_points[keep]-center)@axes.T
        dn = (near_normals[keep]-normal)@axes.T
        shape = np.linalg.solve(xy.T@xy + .018**2*np.eye(2), xy.T@dn)
        shape = (shape+shape.T)/2
        eigen, vec = np.linalg.eigh(shape)
        shape = (vec * eigen.clip(-12,40)) @ vec.T
        first_dirs = np.repeat(directions,3,axis=0)
        dirs = first_dirs.copy()
        travel = np.zeros(len(dirs))
        offsets = np.tile([-.002,0,.002],len(directions))
        extra_tilt = np.tile([.045,0,-.045],len(directions))
        centers = np.broadcast_to(center,(len(dirs),3)).copy()
        norm = np.broadcast_to(normal,(len(dirs),3)).copy()
        remote = self.remote_frontiers(surface,center,normal,tick)
        for point, n, stroke, lead, distance in remote:
            centers = np.vstack([centers,point])
            norm = np.vstack([norm,n])
            dirs = np.vstack([dirs,stroke])
            first_dirs = np.vstack([first_dirs,lead])
            travel = np.r_[travel,distance]
        offsets = np.r_[offsets,np.zeros(len(remote))]
        extra_tilt = np.r_[extra_tilt,np.zeros(len(remote))]
        count = len(dirs)
        bend = SWEEP*(dirs@axes.T)@shape@axes
        bend[travel>0]=0
        turn = np.cross(norm,bend + extra_tilt[:,None]*dirs)
        turn *= np.minimum(1,.23/np.maximum(1e-10,np.linalg.norm(turn,axis=1)))[:,None]
        transport = np.broadcast_to(np.eye(3),(count,3,3)).copy()
        for a in np.flatnonzero(travel>0):
            transport[a] = align(normal,norm[a]).as_matrix()
        frame = transport @ wrist_rotation.as_matrix()
        localpads = wrist_rotation.inv().apply(pads - wrist_position)
        pivot = wrist_rotation.inv().apply(center - wrist_position)
        start_w = centers - frame@pivot
        rot_end = Rotation.from_rotvec(turn).as_matrix()@frame
        depth = offsets - .5*SWEEP*(dirs*bend).sum(axis=1)
        end_center = centers + SWEEP*dirs + depth[:,None]*norm
        end_w = end_center - rot_end@pivot
        actions = np.column_stack([np.einsum("aji,aj->ai",frame,end_w-start_w),
                                   np.einsum("aji,aj->ai",frame,turn)])
        delta, low, weights = self.history.predict(q,hits,normal_spread(normals,hits),actions)
        knots = FRACTIONS[None,:,None]*turn[:,None]
        rotations = Rotation.from_rotvec(knots.reshape(-1,3)).as_matrix().reshape(count,2,3,3)@frame[:,None]
        wpos = start_w[:,None] + FRACTIONS[None,:,None]*(end_w-start_w)[:,None]
        proposed = np.einsum("atij,ahtfj->ahtfi",rotations,localpads + delta) + wpos[:,None,:,None]
        mean = np.einsum("ah,ahtfj->atfj",weights,proposed)
        gaps = np.zeros(proposed.shape[:-1])
        outnormal = np.empty_like(mean)
        supported = np.zeros((count,4),dtype=bool)
        # Predict signed geometry at each knot; linearize over the small
        # within-action variation in the six historical finger responses.
        groups = [np.arange(count-len(remote)),*[np.array([k]) for k in np.flatnonzero(travel>0)]]
        for group in groups:
            k = group[0]
            at = centers[k] + (localpads-pivot)@frame[k].T
            for finger, anchor in enumerate(at):
                n = transport[k] @ normals[finger]
                if np.linalg.norm(n)<.5: n=norm[k]
                patch = surface.patch(anchor,preferred_normal=n)
                if patch is None:
                    outnormal[group,:,finger] = n
                    continue
                query = mean[group,:,finger]
                val, grad = patch.predict(query.reshape(-1,3))
                unitgrad = grad / np.maximum(np.linalg.norm(grad,axis=1,keepdims=True),1e-8)
                zero = patch.predict(anchor)[0][0] if hits[finger] and travel[k]==0 else 0.
                gap = (val-zero)/np.maximum(np.linalg.norm(grad,axis=1),.5)
                nn = unitgrad.reshape(len(group),2,3)
                outnormal[group,:,finger] = nn
                difference = proposed[group,:,:,finger]-query[:,None]
                gaps[group,:,:,finger] = gap.reshape(len(group),1,2) + (difference*nn[:,None]).sum(axis=-1)
                supported[group,finger] = True
        projected = proposed - gaps[...,None]*outnormal[:,None]
        distance = surface.tree.query(projected.reshape(-1,3))[0].reshape(gaps.shape)
        proximity = np.exp(-.5*((gaps+.0007)/.006)**2)
        # low is the observed light-contact fraction at each knot. Retain
        # empirical losses and overloads in the outcome probabilities.
        probability = proximity*low*(distance < .046)*supported[:,None,None,:]
        gain, new_count = joint_gain(projected,probability,weights,surface,self.global_model)
        quorum = (probability[...,None,:] > MASK_THRESHOLDS[:,:4]).sum(axis=-1)>=3
        light_three = np.sum(weights*quorum.mean(axis=(2,3)),axis=1)
        expected_contacts = np.sum(weights*probability.sum(axis=-1).mean(axis=2),axis=1)
        # Only approaching a nearby measured plane beyond the already reached
        # depth is unsafe. Check both the first command and the end of the stroke.
        first_turn = turn.copy()
        first_delta = end_w - wrist_position
        for a in np.flatnonzero(travel>0):
            nmove = SWEEP*(first_dirs[a]@axes.T)@shape@axes
            first_turn[a] = limit(np.cross(normal,nmove),.23)
            first_delta[a] = center+SWEEP*first_dirs[a] - (Rotation.from_rotvec(first_turn[a])*wrist_rotation).apply(pivot) - wrist_position
        f = EXECUTE/HORIZON
        first_rot = Rotation.from_rotvec(f*first_turn)*wrist_rotation
        first_pos = wrist_position + f*first_delta
        palm = wrist_rotation.inv().apply(palm_vertices-wrist_position)
        safe = (np.linalg.norm(first_pos - wrist_position,axis=1) <= .04) & (f*np.linalg.norm(first_turn,axis=1) <= .25)
        if workspace_center is not None:
            safe &= np.linalg.norm(end_w - workspace_center, axis=1) <= .45
        now_dist,now_ids = surface.tree.query(palm_vertices)
        now_gap = ((palm_vertices-surface.points[now_ids])*surface.normals[now_ids]).sum(axis=-1)
        for pos,rot in ((first_pos,first_rot.as_matrix()),(end_w,rot_end)):
            world = pos[:,None]+np.einsum("aij,pj->api",rot,palm)
            dist, idx = surface.tree.query(world)
            gap = np.sum((world-surface.points[idx])*surface.normals[idx],axis=-1)
            same = (now_dist[None]<.025) & (idx==now_ids[None])
            floor = np.where(same,np.minimum(0,now_gap)[None]-.002,-.003)
            safe &= ~np.any((dist<.020)&(gap<floor),axis=1)
        duration = HORIZON+travel/.065
        rate = np.where(safe & (expected_contacts>.05), gain/duration, -np.inf)
        valid = np.flatnonzero(np.isfinite(rate))
        if len(valid)==0 or np.max(rate[valid])<=0:
            return self.recontact(wrist_position,wrist_rotation,normal,loads,count)
        # Within 2% of the best information rate prefer three-finger contact;
        # resolve exact ties by less wrist motion, then stable candidate order.
        near = valid[rate[valid] >= .98*np.max(rate[valid])]
        motion = np.linalg.norm(first_delta,axis=1)+.065*np.linalg.norm(first_turn,axis=1)
        choice = int(near[np.lexsort((near,motion[near],-light_three[near]))[0]])
        other = valid[valid!=choice]
        runner = int(other[np.argmax(rate[other])]) if len(other) else None
        def detail(i):
            return dict(direction=first_dirs[i].tolist(),gain_nat=float(gain[i]),rate=float(rate[i]),
                        light3=float(light_three[i]),expected_new_cells=float(new_count[i]),
                        frontier=bool(travel[i]>0),duration_s=float(duration[i]))
        local_rates = rate[:3*len(ANGLES)].reshape(-1, 3).max(axis=1)
        diagnostics = dict(mode="explore",candidate_count=count,feasible_count=len(valid),
                           regional_candidates=len(remote),target_m=end_center[choice].tolist(),
                           local_rate_nat_s=[float(v) if np.isfinite(v) else None for v in local_rates],
                           winner=detail(choice),runner_up=detail(runner) if runner is not None else None,
                           history_effective_samples=float(1/np.sum(weights[choice]**2)),
                           direction=first_dirs[choice].tolist(),gain_nat=float(gain[choice]),choice=choice)
        return dict(position=first_pos[choice],rotation=first_rot[choice],direction=first_dirs[choice],
                    normal=normal,diagnostics=diagnostics)

    @staticmethod
    def recontact(pos,rotation,normal,loads,count):
        step = .005 if np.max(loads)>4 else -.004
        return dict(position=pos+step*normal,rotation=rotation,normal=normal,direction=np.zeros(3),
                    diagnostics=dict(mode="recontact",candidate_count=count,feasible_count=0,
                                     gain_nat=0.,winner=None,runner_up=None,history_effective_samples=0.))

