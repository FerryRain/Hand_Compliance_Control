"""Deterministic geometry pool for contact dynamics, separate from DP assets.

Convex objects use exact convex collision meshes. Nonconvex local surfaces use
native heightfields, never the convex hull of the complete wavy surface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from functools import lru_cache

FAMILIES=('plane','ellipsoid','cylinder','rounded_box','tapered','waves','saddle','irregular')


@lru_cache(maxsize=32)
def collision_planes(folder,files):
    from scipy.spatial import ConvexHull
    return [ConvexHull(trimesh.load(Path(folder)/f,process=False).vertices).equations for f in files]


@lru_cache(maxsize=32)
def _packed_collision_planes(folder, files):
    parts = collision_planes(folder, files)
    packed = np.zeros((len(parts), max(map(len, parts)), 4))
    valid = np.zeros(packed.shape[:2], dtype=bool)
    for i, planes in enumerate(parts):
        packed[i, :len(planes)] = planes
        valid[i, :len(planes)] = True
    return packed, valid


def collision_line_exit(root, row, origin, direction):
    """Topmost exit of a LINE through the actual convex collision union.

    Privileged collection-motion geometry, not a fingertip contact measurement.
    Distances use the supplied unit direction; negative exits are allowed.
    Intersecting the whole-object convex hull would incorrectly fill concavities.
    """
    if row['backend'] not in ('convex_parts', 'convex_mesh'):
        raise ValueError('Line following requires closed convex collision pieces')
    files = tuple(row['collision_parts']) if row['backend'] == 'convex_parts' else ('surface.obj',)
    planes, valid = _packed_collision_planes(str(Path(root)/'objects'/row['object_id']), files)
    origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    if not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError('Nonfinite line')
    length = np.linalg.norm(direction)
    if length < 1e-12:
        raise ValueError('Zero line direction')
    direction = direction / length
    signed = planes[..., :3] @ origin + planes[..., 3]
    slope = planes[..., :3] @ direction
    upper, lower = valid & (slope > 1e-10), valid & (slope < -1e-10)
    bounds = np.divide(-signed, slope, out=np.zeros_like(signed), where=np.abs(slope)>1e-10)
    high = np.min(np.where(upper, bounds, np.inf), axis=1)
    low = np.max(np.where(lower, bounds, -np.inf), axis=1)
    feasible = (high >= low - 1e-9) & np.isfinite(high)
    feasible &= ~np.any(valid & ~(upper | lower) & (signed > 1e-9), axis=1)
    if not feasible.any():
        raise ValueError('No collision-union line intersection')
    part = int(np.argmax(np.where(feasible, high, -np.inf)))
    face = int(np.argmin(np.where(upper[part], bounds[part], np.inf)))
    return float(high[part]), planes[part, face, :3].copy()


def solve_clearance_height(clearance, guess, center, *, target=2e-5, travel=.06, tolerance=1e-6):
    """Bounded scalar root on the separated side of a fixed fingertip.

    A signed distance's penetration magnitude can be discontinuous at convex
    feature switches. Bisection only uses its sign relative to a positive gap;
    it never commands a displacement proportional to that penetration depth.
    ``clearance(z)`` is a static geometry query, not a live simulator action.
    """
    if target <= 0 or tolerance <= 0:
        raise ValueError('Positive separation and tolerance required')
    lo=hi=float(guess);step=.0005;queries=0
    left=right=float(clearance(guess))-target;queries+=1
    if not np.isfinite(left):raise ValueError('Nonfinite geometry distance')
    while left < 0 or right > 0:
        if left < 0:
            lo=max(center-travel,lo-step);left=float(clearance(lo))-target;queries+=1
        if right > 0:
            hi=min(center+travel,hi+step);right=float(clearance(hi))-target;queries+=1
        if not np.isfinite(left+right):raise ValueError('Nonfinite geometry distance')
        if (lo<=center-travel and left<0) or (hi>=center+travel and right>0):
            raise ValueError('No object-height contact bracket within allowed travel')
        step*=2
    while hi-lo > tolerance:
        mid=(lo+hi)/2;value=float(clearance(mid))-target;queries+=1
        if not np.isfinite(value):raise ValueError('Nonfinite geometry distance')
        if value>0:lo=mid
        else:hi=mid
    return (lo+hi)/2,queries


def build(family, seed):
    rng=np.random.default_rng(seed);p=dict(seed=seed,family=family)
    if family in ('waves','saddle','irregular'):
        half=.035; n=7; grid=np.linspace(-half,half,n);x,y=np.meshgrid(grid,grid)
        a=float(rng.uniform(.001,.003));w=float(rng.uniform(.025,.05))
        if family=='waves':z=a*np.sin(2*np.pi*x/w)+.45*a*np.cos(2*np.pi*y/(w*1.3))
        elif family=='saddle':z=a*((x/half)**2-(y/half)**2)
        else:
            z=a*.4*np.sin(2*np.pi*x/w)*np.cos(2*np.pi*y/(w*1.2))
            for _ in range(5):
                cx,cy=rng.uniform(-.02,.02,2);sigma=rng.uniform(.008,.018);amp=rng.uniform(-.002,.002)
                z+=amp*np.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma*sigma))
        z-=z[n//2,n//2];lo=float(z.min());span=float(z.max()-lo)
        vertices=np.c_[x.ravel(),y.ravel(),z.ravel()];faces=[]
        for j in range(n-1):
            for i in range(n-1):
                k=j*n+i;faces.extend(((k,k+1,k+n),(k+1,k+n+1,k+n)))
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        gy,gx=np.gradient(z,grid,grid);slope=np.sqrt(gx*gx+gy*gy)
        p.update(backend='hfield',half_extent_m=half,n=n,z_min_m=lo,z_span_m=span,
                 amplitude_m=a,wavelength_m=w,max_slope=float(slope.max()),anchor=[0.,0.,0.],
                 vertical_range_m=float(np.ptp(z)),grid_spacing_m=float(grid[1]-grid[0]),
                 collision_resolution_note='7x7 native heightfield; no whole-surface convex hull')
        if p['max_slope']>1.:raise ValueError('Surface too steep for this pilot')
        return mesh,p,(z-lo)/span
    if family=='plane':
        # Keep the first nominal plane reproducible; other instances vary in
        # footprint/thickness so identical meshes do not cross data splits.
        half=np.array([.025,.025,.006]) if seed==20261109 else np.r_[rng.uniform(.022,.04,2),rng.uniform(.004,.009)]
        mesh=trimesh.creation.box(extents=2*half)
    elif family=='ellipsoid':
        half=rng.uniform(.018,.035,3);mesh=trimesh.creation.icosphere(subdivisions=3);mesh.vertices*=half
    elif family=='cylinder':
        half=np.array([rng.uniform(.018,.030)]*2+[rng.uniform(.015,.03)])
        mesh=trimesh.creation.cylinder(radius=half[0],height=2*half[2],sections=64)
    elif family=='rounded_box':
        half=rng.uniform(.018,.030,3);radius=float(rng.uniform(.004,.008))
        sphere=trimesh.creation.icosphere(subdivisions=2).vertices*radius
        corners=np.array(np.meshgrid(*[[-1,1]]*3)).T.reshape(-1,3)*(half-radius)
        mesh=trimesh.convex.convex_hull((sphere[None]+corners[:,None]).reshape(-1,3))
        p['rounding_radius_m']=radius
    elif family=='tapered':
        r0=float(rng.uniform(.022,.032));r1=float(rng.uniform(.011,.018));hz=float(rng.uniform(.020,.035))
        t=np.linspace(0,2*np.pi,64,endpoint=False)
        vertices=np.r_[np.c_[r0*np.cos(t),r0*np.sin(t),np.full(64,-hz)],np.c_[r1*np.cos(t),r1*np.sin(t),np.full(64,hz)]]
        mesh=trimesh.convex.convex_hull(vertices);half=np.array([r0,r0,hz]);p.update(radius_bottom_m=r0,radius_top_m=r1)
    else:raise ValueError(family)
    # Curved side regions rather than only flat end caps for cylinders/cones.
    if family=='cylinder':anchor=np.array([half[0],0,0]);normal=np.array([1.,0,0])
    elif family=='tapered':
        anchor=np.array([(r0+r1)/2,0,0]);normal=np.array([1.,0.,(r0-r1)/(2*hz)]);normal/=np.linalg.norm(normal)
    else:anchor=np.array([0.,0.,half[2]]);normal=np.array([0.,0.,1.])
    from scipy.spatial.transform import Rotation
    rotation=Rotation.align_vectors([[0,0,1]],[normal])[0].as_matrix()
    mesh.vertices=mesh.vertices@rotation.T
    p.update(backend='convex_mesh',half_size_m=half.tolist(),anchor=(rotation@anchor).tolist(),
             watertight=bool(mesh.is_watertight),convex=bool(mesh.is_convex),max_slope=None)
    if not mesh.is_watertight or not mesh.is_convex:raise ValueError('Convex source expected')
    return mesh,p,None


def generate(output,per_family=4):
    output=Path(output)
    if (output/'object_manifest.json').exists():raise FileExistsError('Use a new pool revision; never overwrite geometry used by recorded data')
    objects=output/'objects';objects.mkdir(parents=True,exist_ok=True);rows=[]
    for fi,family in enumerate(FAMILIES):
        for j in range(per_family):
            seed=20261109+fi*100+j;mesh,p,heights=build(family,seed)
            name=f'{family}_{j:03d}';folder=objects/name;folder.mkdir(exist_ok=True)
            mesh.export(folder/'surface.obj')
            if heights is not None:np.save(folder/'height.npy',heights.astype(np.float32))
            p.update(object_id=name,split=('train' if j<per_family-2 else 'validation' if j==per_family-2 else 'test_id'),
                     mesh_sha256=hashlib.sha256((folder/'surface.obj').read_bytes()).hexdigest(),units='metres',
                     geometry_ood=False,notes='local contact-scale object; not a complete graspable-object distribution')
            (folder/'geometry.json').write_text(json.dumps(p,indent=2));rows.append(p)
    manifest=dict(schema='contact_wm_geometry_v1',objects=rows,
        split_rule='immutable object instance; all episodes and branches inherit this split',
        geometry_ood='not generated yet; cannot claim unseen-family generalization from test_id')
    assert len({r['mesh_sha256'] for r in rows})==len(rows),'Duplicate geometry instances: do not split identical meshes'
    (output/'object_manifest.json').write_text(json.dumps(manifest,indent=2))
    gallery(output,rows)
    return manifest


def gallery(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(16,9))
    for i,family in enumerate(FAMILIES):
        row=next(r for r in rows if r['family']==family);mesh=trimesh.load(output/'objects'/row['object_id']/'surface.obj',process=False)
        ax=fig.add_subplot(2,4,i+1,projection='3d');verts=mesh.vertices*1000;faces=mesh.faces
        coll=Poly3DCollection(verts[faces],facecolor=plt.cm.viridis(.15+i*.09),edgecolor='none',alpha=.94);ax.add_collection3d(coll)
        center=(verts.max(0)+verts.min(0))/2;radius=max(np.ptp(verts,axis=0))/2*1.08
        for k,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):setter(center[k]-radius,center[k]+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(25,-55);ax.set_title(family+'\n'+row['backend']);ax.set_xlabel('mm');ax.set_ylabel('mm')
    fig.suptitle('Contact WM P0: procedural geometries (true nonconvex heightfields)',fontsize=16)
    fig.subplots_adjust(top=.88,bottom=.06,hspace=.42,wspace=.1);fig.savefig(output/'geometry_gallery.png',dpi=140);plt.close(fig)


def surface_anchor(root,row,uv):
    """Local top-surface point and normal, used only to initialize contact."""
    x,y=np.asarray(uv,dtype=float)
    if row['backend']=='hfield':
        h=np.load(Path(root)/'objects'/row['object_id']/'height.npy')*row['z_span_m']+row['z_min_m']
        spacing=2*row['half_extent_m']/(row['n']-1)
        a=(x+row['half_extent_m'])/spacing;b=(y+row['half_extent_m'])/spacing
        i=int(np.floor(a));j=int(np.floor(b));a-=i;b-=j
        z00,z10,z01,z11=h[j,i],h[j,i+1],h[j+1,i],h[j+1,i+1]
        # Match the triangulation used to export the analytic heightfield.
        if a+b<=1:z=z00+a*(z10-z00)+b*(z01-z00);dx=(z10-z00)/spacing;dy=(z01-z00)/spacing
        else:z=z11+(1-a)*(z01-z11)+(1-b)*(z10-z11);dx=(z11-z01)/spacing;dy=(z11-z10)/spacing
        normal=np.array([-dx,-dy,1.])
    elif row['backend']=='convex_parts':
        # Initialize on the actual collision union, not on an approximate
        # visual surface whose position may differ by millimetres.
        candidates=[]
        for e in collision_planes(str(Path(root)/'objects'/row['object_id']),tuple(row['collision_parts'])):
            xy=e[:,0]*x+e[:,1]*y+e[:,3];upper=e[:,2]>1e-9;lower=e[:,2]<-1e-9
            if np.any(xy[~(upper|lower)]>1e-8) or not upper.any():continue
            eu=e[upper];heights=-xy[upper]/eu[:,2];k=np.argmin(heights);zu=heights[k]
            zl=np.max(-xy[lower]/e[lower,2]) if lower.any() else -np.inf
            if zu>=zl-1e-8:candidates.append((zu,eu[k,:3]))
        if not candidates:raise ValueError('No collision-surface ray intersection')
        z,normal=max(candidates,key=lambda p:p[0])
    else:
        from scipy.spatial import ConvexHull
        mesh=trimesh.load(Path(root)/'objects'/row['object_id']/'surface.obj',process=False)
        e=ConvexHull(mesh.vertices).equations;e=e[e[:,2]>1e-7]
        heights=-(e[:,0]*x+e[:,1]*y+e[:,3])/e[:,2];k=np.argmin(heights);z=heights[k];normal=e[k,:3]
    return np.array([x,y,z]),normal/np.linalg.norm(normal)


def target_spec(root,row,friction=1.,contact_multiplier=1.):
    import mujoco
    s=mujoco.MjSpec();dynamic=row.get('motion_backend')=='mocap_weld'
    body=s.worldbody.add_body(name='target_ball',mocap=not dynamic)
    if dynamic:
        body.add_freejoint(name='object_free')
        driver=s.worldbody.add_body(name='object_driver',mocap=True)
        s.add_equality(name='object_drive_weld',type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,name1=body.name,name2=driver.name,
            data=[0,0,0,0,0,0,1,0,0,0,1],solref=[.005,1.],solimp=[.95,.99,.001,.5,2.])
    folder=Path(root)/'objects'/row['object_id']
    kwargs=dict(name='wm_surface',friction=[friction,.005,.0001],solref=[-20000*contact_multiplier,-400*contact_multiplier**.5],
                solimp=[.9,.98,.002,.5,2.],priority=10,condim=3,rgba=[.55,.7,.9,1.],mass=.2)
    if row['backend']=='convex_parts':
        s.add_mesh(name='surface_visual',file=str((folder/'surface.obj').resolve()))
        body.add_geom(name='wm_visual',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='surface_visual',
            contype=0,conaffinity=0,group=1,mass=0,rgba=[.45,.7,.9,1.])
        for i,file in enumerate(row['collision_parts']):
            name=f'part_{i}';s.add_mesh(name=name,file=str((folder/file).resolve()))
            part_kwargs={**kwargs,'name':f'wm_collision_{i}','mass':.2/len(row['collision_parts']),'rgba':[.4,.7,.9,0.]}
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,group=3,**part_kwargs)
    elif row['backend']=='hfield':
        a=np.load(folder/'height.npy');s.add_hfield(name='surface',size=[row['half_extent_m']]*2+[row['z_span_m'],.006],nrow=row['n'],ncol=row['n'],userdata=a.ravel())
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname='surface',pos=[0,0,row['z_min_m']],**kwargs)
    else:
        s.add_mesh(name='surface',file=str((folder/'surface.obj').resolve()))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname='surface',**kwargs)
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--per-family',type=int,default=4);a=p.parse_args()
    print(json.dumps({'objects':len(generate(a.output,a.per_family)['objects']),'output':str(a.output)}))
