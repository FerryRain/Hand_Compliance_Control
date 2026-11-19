"""Object-frame feature traversals; no finger waypoints or force feedback.

The deadband is a collection design, not a certified reachable set. Geometric
incompatibility and residual violations are reported, never hidden by teleport.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from contact_wm_geometry import collision_line_exit, collision_planes
from contact_wm_distance_band import surface_heights, smooth_band


def blend(u):
    u = np.clip(u, 0., 1.)
    return u**3*(10.+u*(-15.+6.*u))


def feature_path(root, row, home, anchor, palm_xy, palm_bottom, frames,
                 *, half_band=.012, clearance=.008, edge_sign=1):
    from pathlib import Path
    home=np.asarray(home,float);anchor=np.asarray(anchor,float)
    if edge_sign not in (-1,1):raise ValueError('edge_sign must be -1 or 1')
    initial=Rotation.from_quat(home[[4,5,6,3]])
    local=initial.inv().apply(anchor-home[:3])
    mesh=trimesh.load(Path(root)/'objects'/row['object_id']/'surface.obj',process=False)
    bounds=mesh.bounds
    t=np.linspace(0,1,frames)
    points=np.zeros((frames,3));angles=np.zeros(frames);phase=np.zeros(frames,dtype=np.int32)
    if row['object_id']=='ycb_013_apple':
        # Convex shoulder -> depression boundary -> opposite shoulder; no loop.
        xs=local[0]+(.075-local[0])*blend(t)
        for i,x in enumerate(xs):
            z,_=collision_line_exit(root,row,[x,local[1],0],[0,0,1])
            points[i]=[x,local[1],z]
        phase=np.where(xs<-.025,0,np.where(xs<.025,1,2))
        labels=['convex shoulder','stem-depression crossing','opposite shoulder']
    else:
        # Traverse the long-axis end: top surface, rounded rim/edge, end face.
        # Keep the ray origin inside the body, not outside a recessed can lid.
        corner=np.array([bounds[1,0]-.030 if edge_sign==1 else bounds[0,0]+.030,local[1],bounds[1,2]-.030])
        for i,u in enumerate(t):
            if u<.32:
                x=local[0]+(corner[0]-local[0])*blend(u/.32)
                z,_=collision_line_exit(root,row,[x,local[1],0],[0,0,1])
                points[i]=[x,local[1],z]
            elif u<.76:
                theta=edge_sign*np.pi/2*blend((u-.32)/.44);normal=np.array([np.sin(theta),0,np.cos(theta)])
                distance,_=collision_line_exit(root,row,corner,normal)
                points[i]=corner+distance*normal;angles[i]=-theta;phase[i]=1
            else:
                z=corner[2]*(1-blend((u-.76)/.24))
                distance,_=collision_line_exit(root,row,[0,local[1],z],[edge_sign,0,0])
                points[i]=[edge_sign*distance,local[1],z];angles[i]=-edge_sign*np.pi/2;phase[i]=2
        labels=['side/top interior','rounded rim/edge','end/adjacent face']
    rotations=initial*Rotation.from_euler('y',angles)
    # Preserve initial command exactly. XY transports material features under
    # the contact neighborhood; Z is NOT pointwise tracking of that feature.
    delta=rotations.apply(points)-initial.apply(points[0])
    prescribed=np.tile(home,(frames,1));prescribed[:,:2]-=delta[:,:2]
    prescribed[:,3:]=rotations.as_quat()[:,[3,0,1,2]];prescribed[0]=home
    planes=collision_planes(str(Path(root)/'objects'/row['object_id']),tuple(row['collision_parts']))
    patch=anchor[:2]+np.array([[0,0],[.004,0],[-.004,0],[0,.004],[0,-.004]])
    indices=np.unique(np.r_[np.arange(0,frames,10),frames-1])
    # Chunked geometry queries bound memory; interpolation is disclosed.
    heights=np.concatenate([surface_heights(planes,prescribed[ids],np.vstack([patch,palm_xy]))
                            for ids in np.array_split(indices,max(1,len(indices)//24))])
    if np.any(~np.isfinite(heights[:,:5]).any(1)):
        raise ValueError('Feature path leaves the contact neighborhood entirely')
    low=np.nanmin(heights[:,:5],axis=1);high=np.nanmax(heights[:,:5],axis=1)
    # A finite fingertip needs ONE reachable contact patch, not simultaneous
    # contact with the deepest point and the rim. Using the minimum here would
    # drive the object upward when a groove passes beside a loaded rim.
    lower=np.interp(np.arange(frames),indices,high[0]-half_band-high)
    distance_upper=np.interp(np.arange(frames),indices,high[0]+half_band-high)
    palm_height=np.max(np.where(np.isfinite(heights[:,5:]),heights[:,5:],-np.inf),axis=1)
    palm_limit=np.interp(np.arange(frames),indices,palm_bottom-clearance-palm_height)
    upper=np.minimum(distance_upper,palm_limit)
    offset,velocity,trigger=smooth_band(lower,upper,vmax=.025,amax=.12)
    commanded=prescribed.copy();commanded[:,2]+=offset
    linear=np.linalg.norm(np.diff(commanded[:,:3],axis=0),axis=1)/.01
    angular=(rotations[1:]*rotations[:-1].inv()).magnitude()/.01
    residual=np.maximum(lower-offset,offset-upper).clip(0)
    report=dict(phase_labels=labels,initial_material_point=points[0].tolist(),
        final_material_point=points[-1].tolist(),material_path_length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
        normal_reorientation_deg=float(np.rad2deg(abs(angles[-1]))),
        half_band_m=half_band,palm_clearance_m=clearance,edge_sign=edge_sign,
        incompatible_frames=int((lower>upper).sum()),remaining_band_violation_m=float(residual.max()),
        height_offset_range_m=[float(offset.min()),float(offset.max())],
        height_trigger_fraction=float((trigger!=0).mean()),
        palm_limit_tighter_fraction=float((palm_limit<distance_upper).mean()),
        palm_driven_lowering_fraction=float(((trigger<0)&(palm_limit<distance_upper)).mean()),
        max_translation_m_s=float(linear.max()),max_angular_rad_s=float(angular.max()),
        surface_speed_upper_bound_m_s=float((linear+angular*max(np.linalg.norm(mesh.vertices,axis=1))).max()),
        geometric_sampling_interval_s=.1,physical_admission=False,
        limitation='Sampled palm footprint and initialization-relative band; not a certified FK reachability/collision guarantee.')
    return dict(commanded=commanded,prescribed=prescribed,material_feature_point=points,
                region_phase=phase,lower=lower,upper=upper,height_offset=offset,
                distance_upper=distance_upper,palm_limit=palm_limit),report
