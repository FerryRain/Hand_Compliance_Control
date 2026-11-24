"""A palmar frame and contact-supported estimates of the surface beneath it."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from ..acitive_02.surface_scan import align
from ..i04_geometric_traversal.physics import rotation


OUT = np.array([0., 0., 1.])


def normalize(x):
    x = np.asarray(x)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.e-12)


def clamp_length(x, maximum):
    return x * np.minimum(1., maximum / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.e-12))


def on_plane(x, n):
    return x - np.sum(x * n, axis=-1, keepdims=True) * n


def palm_sites(hand):
    """Center and eight area samples of the actual inward-facing plane in the palm body frame."""
    m, g = hand.model, hand.palm_geom
    mesh = m.geom_dataid[g]
    v0, f0 = m.mesh_vertadr[mesh], m.mesh_faceadr[mesh]
    vertices = rotation(m.geom_quat[g]).apply(m.mesh_vert[v0:v0+m.mesh_vertnum[mesh]]) + m.geom_pos[g]
    faces = m.mesh_face[f0:f0+m.mesh_facenum[mesh]]
    tri = vertices[faces]
    tri = tri[tri[:, :, 2].max(1) < vertices[:, 2].min() + .00005]
    mass = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    if not len(tri) or mass.sum() < .002:
        raise ValueError('The planning hand has no supported palmar plate in its -Z contact frame')
    center = np.average(tri.mean(1), axis=0, weights=mass)
    # Fixed quadrature samples the hand geometry alone. Oscillations and paths
    # in object coordinates are never used to generate these locations.
    uv = [(i/6, j/6) for i in range(1, 6) for j in range(1, 6-i)]
    pool = np.concatenate([(1-u-v)*tri[:, 0]+u*tri[:, 1]+v*tri[:, 2] for u,v in uv])
    sites = [center]
    for _ in range(8):
        distances = np.linalg.norm(pool[:, None]-np.array(sites), axis=2).min(1)
        sites.append(pool[np.argmax(distances)])
    return np.array(sites)


@dataclass
class Evidence:
    points: np.ndarray
    normals: np.ndarray
    valid: np.ndarray
    mesh: np.ndarray
    components: np.ndarray


class TactilePatches:
    def __init__(self):
        self.mesh = self.labels = None

    def query(self, model, surface, xyz, hints, radius, components=None):
        """Choose one oriented sheet per query on the tactile isosurface, with measured fallback."""
        xyz, hints = np.atleast_2d(xyz), normalize(np.atleast_2d(hints))
        count = len(xyz)
        pts, ns = xyz.copy(), hints.copy()
        valid = np.zeros(count, dtype=bool)
        from_mesh = valid.copy()
        labels = np.full(count, -1)
        mesh = model.mesh
        if mesh is not None and len(mesh.faces):
            if self.mesh is not mesh:
                self.mesh = mesh
                _, self.labels = connected_components(mesh.graph, directed=False)
            distance, face = mesh.tree.query(xyz, k=min(32, len(mesh.faces)))
            face, distance = face.reshape(count,-1), distance.reshape(count,-1)
            normals = mesh.normals[face]
            match = np.einsum('qki,qi->qk', normals, hints)
            allowed = (distance < radius) & (match > .30)
            if components is not None:
                comp = np.asarray(components)[:, None]
                allowed &= (comp < 0) | (self.labels[face] == comp)
            first = np.argmax(allowed, axis=1)
            near_face = face[np.arange(count), first]
            local = self.labels[near_face]
            agree = np.einsum('qki,qi->qk', normals, mesh.normals[near_face]) > .71
            weights = np.exp(-.5*((distance-distance.min(1)[:, None])/.013)**2)*allowed*agree*(self.labels[face]==local[:,None])
            mass = weights.sum(1)
            ok = mass > 0
            if ok.any():
                points = mesh.centers[face] - model.band.half_width*normals
                smoothed = np.einsum('qk,qki->qi', weights, normals)
                # Keep the representative on the sheet. Averaging positions
                # across a curved face can put an apparent target inside it.
                pts[ok] = points[np.arange(count), first][ok]
                ns[ok] = normalize(smoothed[ok])
                labels[ok] = local[ok]
                valid[ok] = from_mesh[ok] = True
        missing = np.flatnonzero(~valid)
        if len(missing) and surface.tree is not None:
            dist, idx = surface.tree.query(xyz[missing], k=min(16,len(surface.points)))
            dist, idx = dist.reshape(len(missing),-1), idx.reshape(len(missing),-1)
            nearby = surface.normals[idx]
            allowed = (dist < min(radius,.042)) & (np.einsum('qki,qi->qk', nearby,hints[missing]) > .55)
            first = np.argmax(allowed, axis=1)
            selected = idx[np.arange(len(missing)), first]
            use = allowed.any(1)
            pts[missing[use]] = surface.points[selected[use]]
            ns[missing[use]] = surface.normals[selected[use]]
            valid[missing[use]] = True
        return Evidence(pts,ns,valid,from_mesh,labels)


def absolute_attitudes(current, outwards):
    old = current.apply(OUT)
    return Rotation.from_quat(np.stack([(align(old,n)*current).as_quat() for n in outwards]))


def footprint(patches, model, surface, sites, plane, normal, outward, labels):
    """Estimate how much of the physical palm looks onto tactile-supported object surface."""
    count, width, _ = sites.shape
    probes = sites - np.sum((sites-plane[:,None])*normal[:,None],axis=-1,keepdims=True)*normal[:,None]
    local = patches.query(model,surface,probes.reshape(-1,3),np.repeat(normal,width,axis=0),
                          .082,np.repeat(labels,width))
    skin = local.points.reshape(count,width,3)
    n = local.normals.reshape(count,width,3)
    denom = np.einsum('qsi,qi->qs',n,outward)
    distance = np.sum((sites-skin)*n,axis=2)/np.maximum(denom,.15)
    projected = sites-distance[:,:,None]*outward[:,None]
    lateral = np.linalg.norm(projected-skin,axis=2)
    good = local.valid.reshape(count,width)&(denom>.35)&(distance>.002)&(distance<.21)
    evidence = np.exp(-.5*(lateral/.022)**2)*good
    return evidence.mean(1),good & (lateral<.016),skin[:,0],n[:,0],distance[:,0],local.mesh.reshape(count,width)[:,0]
