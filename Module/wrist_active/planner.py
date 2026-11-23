"""Choose a short wrist motion using measured surface patches and whole-hand motion examples."""
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import ObservedSurface, VOXEL_M
from ..acitive_02.aude import tangent, unit
from ..acitive_02.surface_scan import PadSupport, limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from .history import normal_spread


HORIZON = .5
PERIOD = .25
LOAD_LIMIT = 8.
STEP_M = .008
ANGLES = np.deg2rad([0, 45, -45, 90, -90, 145, -145, 180])
PIVOTS = np.array([0., .050, -.050])


def pose(position, attitude):
    return np.r_[position, quaternion(attitude)]


def interpolate(first, last, weights):
    w = np.asarray(weights)[:, None]
    ori = rotation(first[3:])
    turn = (rotation(last[3:]) * ori.inv()).as_rotvec()
    attitudes = (Rotation.from_rotvec(w * turn) * ori).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((first[:3] + w * (last[:3]-first[:3]), attitudes))


def frontier_direction(surface, point, normal, heading):
    """Find a sparsely observed surface boundary reachable through measured neighbors."""
    if len(surface.points) < 14:
        return heading
    step = max(1, len(surface.points)//240)
    p, n = surface.points[::step], surface.normals[::step]
    local = cKDTree(p)
    edges = local.query_pairs(.038, output_type='ndarray')
    if len(edges) == 0:
        return heading
    u, v = edges.T
    usable = np.einsum('ij,ij->i', n[u], n[v]) > .32
    u, v = u[usable], v[usable]
    graph = coo_matrix((np.linalg.norm(p[u]-p[v], axis=1), (u,v)), shape=(len(p),len(p))).tocsr()
    source = local.query(point)[1]
    dist, pred = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
    axes = np.eye(3)[np.argmin(abs(n), axis=1)]
    e = np.cross(n, axes)
    e /= np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-8)
    f = np.cross(n,e)
    a = np.arange(8)*np.pi/4
    dirs = e[:,None]*np.cos(a)[None,:,None]+f[:,None]*np.sin(a)[None,:,None]
    probes = p[:,None]+.015*dirs
    missing = surface.tree.query(probes.reshape(-1,3))[0].reshape(len(p),8)
    score = np.minimum(missing/.015,1.) - .9*dist[:,None] + .14*np.einsum('paj,j->pa',dirs,heading)
    at, angle = np.unravel_index(np.argmax(score),score.shape)
    goal = probes[at,angle]
    while pred[at] >= 0 and dist[at] > .040:
        goal = p[at]
        at = pred[at]
    direct = tangent(goal-point, normal)
    return unit(direct) if np.linalg.norm(direct) > .002 else heading


class WristPlanner:
    def __init__(self, robot, library, mode):
        if robot.objects:
            raise ValueError('Wrist planning requires a hand-only kinematics model')
        self.robot, self.library, self.mode = robot, library, mode
        self.surface = ObservedSurface()
        self.vertices = PadSupport(robot).vertices
        g = robot.palm_geom
        mesh = robot.model.geom_dataid[g]
        start, count = robot.model.mesh_vertadr[mesh], robot.model.mesh_vertnum[mesh]
        raw = robot.model.mesh_vert[start:start+count]
        axes = np.vstack((np.eye(3),-np.eye(3),[[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)]))
        self.palm = raw[np.unique(np.argmax(raw@axes.T,axis=0))]
        self.last_normals = np.zeros((4,3))
        self.stability = np.zeros(4)
        self.heading = self.guide = self.safe = None
        self.next_frontier = 0
        self.recover_count = 0

    def observe(self, touch, tick):
        hits = touch.measured_heads
        self.stability += .04*(hits*(touch.loads.max()<LOAD_LIMIT)-self.stability)
        self.last_normals[hits] = touch.normals[hits]
        if touch.loads.max() < LOAD_LIMIT:
            self.surface.add(touch,tick)

    def fingertips(self, data, touch, normal):
        points = touch.points.copy()
        for i in np.flatnonzero(~touch.measured_heads):
            g = self.robot.pads[i]
            local = self.vertices[int(g)]
            vertices = data.geom_xpos[g]+local@data.geom_xmat[g].reshape(3,3).T
            n = self.last_normals[i] if np.linalg.norm(self.last_normals[i])>.5 else normal
            height = vertices@n
            points[i] = vertices[height<=height.min()+.0005].mean(axis=0)
        return points

    def recovery(self, origin, live, touch, normal, reason):
        self.recover_count += 1
        disp = normal * (.0025 if reason=='relief' else -.0008)
        if reason != 'relief' and self.safe is not None and self.recover_count>2:
            disp = limit(self.safe[:3]-origin[:3],.004) - .0005*normal
        target = origin.copy()
        target[:3] += disp
        if reason!='relief' and not touch.measured_heads.any() and self.safe is not None:
            target[3:] = interpolate(origin,self.safe,[.16])[0,3:]
        return target, dict(mode=reason, choice=-1, pair=[], gain=0., extra_gain=0., feasible=0, candidates=[])

    def plan(self, data, touch, origin, tick, forced=None):
        surf = self.surface
        surf.refresh()
        live = data.qpos[:7].copy()
        hits = touch.measured_heads
        n = unit(touch.normals[hits].sum(axis=0)) if hits.any() else unit(self.last_normals.sum(axis=0))
        if np.linalg.norm(n)<.5:
            n = rotation(origin[3:]).apply([0.,0.,1.])
        if touch.loads.max()>=LOAD_LIMIT:
            return self.recovery(origin,live,touch,n,'relief')
        if hits.sum()<2 or surf.tree is None:
            return self.recovery(origin,live,touch,n,'recontact')
        points = self.fingertips(data,touch,n)
        pairs = list(combinations(np.flatnonzero(hits),2))
        def pair_quality(pair):
            i,j = pair
            return (min(self.stability[i],self.stability[j]) + .3*(touch.normals[i]@touch.normals[j])
                    + 3.*min(np.linalg.norm(points[i]-points[j]),.085))
        a,b = max(pairs,key=pair_quality)
        anchors = [a,b]
        center = points[anchors].mean(axis=0)
        n = unit(touch.normals[a]+touch.normals[b])
        if np.linalg.norm(n)<.5:
            n = touch.normals[a]
        line = unit(points[b]-points[a])
        r0, measured = rotation(origin[3:]), rotation(live[3:])
        if self.heading is None or np.linalg.norm(tangent(self.heading,n))<.15:
            self.heading = max((tangent(r0.apply(e),n) for e in np.eye(3)[:2]), key=np.linalg.norm)
        heading = unit(tangent(self.heading,n))
        if tick>=self.next_frontier or self.guide is None:
            self.guide = frontier_direction(surf,center,n,heading)
            self.next_frontier = tick+200
        guide = unit(tangent(self.guide,n))
        side = unit(np.cross(n,heading))
        directions = np.repeat(np.cos(ANGLES)[:,None]*heading+np.sin(ANGLES)[:,None]*side,len(PIVOTS),axis=0)
        pivots = np.tile(PIVOTS,len(ANGLES))
        arm = points[b]-points[a]
        curve = np.clip((touch.normals[b]-touch.normals[a])@arm/max(arm@arm,1e-7),0.,22.)
        rv = np.cross(n,directions)*(STEP_M*curve) + pivots[:,None]*line
        turns = Rotation.from_rotvec(rv)
        center_local = measured.inv().apply(center-live[:3])
        pivot = origin[:3]+r0.apply(center_local)
        end_ori = turns*r0
        end_pos = pivot+STEP_M*directions-end_ori.apply(np.broadcast_to(center_local,directions.shape))
        # Fit the palm standoff to the two support patches. Tilt is determined by
        # transported normals and the pivot candidates, all solely wrist variables.
        local_tips = measured.inv().apply(points-live[:3])
        patches = [surf.patch(points[i],preferred_normal=self.last_normals[i] if hits[i] else n) for i in range(4)]
        gaps = []
        for i in anchors:
            patch=patches[i]
            if patch is None:
                return self.recovery(origin,live,touch,n,'recontact')
            pp=end_pos+end_ori.apply(np.broadcast_to(local_tips[i],directions.shape))
            value,gradient=patch.predict(pp)
            ref=patch.predict(points[i])[0][0]
            gaps.append((value-ref)/np.maximum(gradient@n,.55))
        correction=np.clip(-np.mean(gaps,axis=0),-.0025,.0025)
        # A measured force correction is common to all candidate-scoring variants.
        correction+=np.clip((max(touch.forces[a],touch.forces[b])-1.3)*.00026,-.0003,.0011)
        end_pos+=correction[:,None]*n
        actions=np.c_[r0.inv().apply(end_pos-origin[:3]),r0.inv().apply(rv)]
        if self.mode=='rigid-dp':
            delta=np.zeros((len(rv),1,2,4,3))
            hit_prior=np.ones(delta.shape[:-1])
            weights=np.ones((len(rv),1))
        else:
            delta,hit_prior,weights=self.library.predict(data.qpos[self.robot.q],hits,normal_spread(touch.normals,hits),actions)
        positions=np.empty_like(delta)
        for k,fraction in enumerate((.5,1.)):
            rr=(Rotation.from_rotvec(fraction*rv)*r0).as_matrix()
            positions[:,:,k]=np.einsum('cij,csfj->csfi',rr,local_tips+delta[:,:,k])
            positions[:,:,k]+=((1-fraction)*origin[:3]+fraction*end_pos)[:,None,None]
        gap=np.full(positions.shape[:-1],.1)
        uncertainty=np.zeros_like(gap)
        projected=positions.copy()
        for i,patch in enumerate(patches):
            if patch is None:
                continue
            xyz=positions[:,:,:,i].reshape(-1,3)
            value,grad=patch.predict(xyz)
            magnitude=np.maximum(np.linalg.norm(grad,axis=1),.5)
            zero=patch.predict(points[i])[0][0] if hits[i] else 0.
            distance=(value-zero)/magnitude
            gshape=gap[:,:,:,i].shape
            gap[:,:,:,i]=distance.reshape(gshape)
            surface_xyz=xyz-grad*(distance/magnitude)[:,None]
            projected[:,:,:,i]=surface_xyz.reshape((*gshape,3))
            uncertainty[:,:,:,i]=patch.predict(surface_xyz,variance=True).reshape(gshape)
        distance=surf.tree.query(projected.reshape(-1,3))[0].reshape(gap.shape)
        prob=np.exp(-.5*(gap/.0038)**2)*(.30+.70*hit_prior)*(distance<.034)
        support=np.sum(weights*np.min(prob[:,:,:,anchors],axis=(2,3)),axis=1)
        # Candidate palm hull against local GPIS near the two supports.
        geom=self.robot.palm_geom
        palm_world=data.geom_xpos[geom]+self.palm@data.geom_xmat[geom].reshape(3,3).T
        p_local=measured.inv().apply(palm_world-live[:3])
        hull=end_pos[:,None]+np.einsum('cij,pj->cpi',end_ori.as_matrix(),p_local)
        clearance=np.ones(len(rv))
        for i in anchors:
            value=patches[i].predict(hull.reshape(-1,3))[0].reshape(hull.shape[:2])
            near=np.linalg.norm(hull-points[i],axis=-1)<.055
            clearance=np.minimum(clearance,np.min(np.where(near,value,1.),axis=1))
        feasible=(support>=.16)&(clearance>=.0008)&(np.max(np.abs(np.stack(gaps)+correction),axis=0)<.0065)
        if not feasible.any():
            return self.recovery(origin,live,touch,n,'recontact' if clearance.max()>.0008 else 'relief')
        cells=np.floor(projected/VOXEL_M).astype(int)
        # Probability of at least one observation in each cell for this one
        # synchronous four-finger example. Preserve correlations across examples.
        novelty=np.minimum(1.,distance/VOXEL_M)*np.sqrt(np.maximum(uncertainty,0.))
        def union_gain(fingers):
            g=novelty[:,:,:,fingers].reshape(*weights.shape,-1)
            p=prob[:,:,:,fingers].reshape(g.shape)
            coords=cells[:,:,:,fingers,:].reshape(*g.shape,3)
            value=np.zeros(weights.shape)
            for j in range(g.shape[-1]):
                same=np.all(coords[:,:,:j]==coords[:,:,j:j+1],axis=-1)
                unique=np.prod(np.where(same,1.-p[:,:,:j],1.),axis=-1)
                value+=g[:,:,j]*p[:,:,j]*unique
            return np.sum(value*weights,axis=1)
        gain=union_gain([0,1,2,3])
        extra=np.maximum(gain-union_gain(anchors),0.)
        utility=.10*(directions@heading)+.13*(directions@guide)+.05*support-.020*abs(pivots)/.05
        if self.mode!='no-gain-dp':
            utility+=gain
        choice=int(np.argmax(np.where(feasible,utility,-np.inf))) if forced is None else int(forced)
        if not feasible[choice]:
            raise ValueError('A forced branch must satisfy the original feasibility test')
        self.heading=directions[choice]
        self.safe=live
        self.recover_count=0
        candidates=np.c_[np.arange(len(rv)),feasible,support,gain,extra,utility]
        return pose(end_pos[choice],end_ori[choice]),dict(mode='scan',choice=choice,pair=[int(a),int(b)],
            gain=float(gain[choice]),extra_gain=float(extra[choice]),feasible=int(feasible.sum()),
            candidates=candidates.tolist())
