"""Bounded, static placement checks; no future hand or object planning."""
import numpy as np
from scipy.spatial.transform import Rotation
from contact_wm_geometry import surface_anchor


def hand_qpos_indices(model):
    import mujoco
    return [int(model.jnt_qposadr[j]) for j in range(model.njnt)
            if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').startswith('robot/')
            and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]


def set_palm_static(model,data,mocap_id):
    """Native geometry-check root only; never called in the live physics loop."""
    import mujoco
    data.mocap_pos[mocap_id]=[0,0,.6];data.mocap_quat[mocap_id]=[1,0,0,0]
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/palm_lower')
    for j in range(model.njnt):
        if model.jnt_bodyid[j]==body and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            adr=int(model.jnt_qposadr[j]);data.qpos[adr:adr+7]=[0,0,.6,1,0,0,0]


def set_object_static(model,data,mocap_id,pos,quat=None):
    """Place both driver and free object for geometry checks ONLY."""
    import mujoco
    data.mocap_pos[mocap_id]=pos
    if quat is not None:data.mocap_quat[mocap_id]=quat
    joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'target/object_free')
    if joint>=0:
        adr=int(model.jnt_qposadr[joint]);data.qpos[adr:adr+3]=pos
        data.qpos[adr+3:adr+7]=data.mocap_quat[mocap_id]


def place(model, root, row, q, finger, palm_id, target_id, seed):
    if row.get('initialization')=='under_palm':
        return place_under_palm(model,root,row,q,finger,palm_id,target_id,seed)
    import mujoco
    from split_fingertip_collision import GEOMS
    from surface_mcc_finger import TIP_NAMES
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,k) or '' for k in range(model.ngeom)]
    pad=next(k for k,n in enumerate(names) if n.endswith(GEOMS[finger]+'_pad_collision'))
    targets={k for k,n in enumerate(names) if n.startswith('target/') and (model.geom_contype[k] or model.geom_conaffinity[k])}
    sites=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_SITE,k) or '' for k in range(model.nsite)]
    sid=next(k for k,n in enumerate(sites) if n.endswith('/'+TIP_NAMES[finger]))
    mid=model.geom_dataid[pad]
    vertices=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
    rng=np.random.default_rng(seed+10000);d=mujoco.MjData(model)
    set_palm_static(model,d,palm_id)
    original=q.copy();rejections=[]
    for attempt in range(160):
        q=original.copy()
        if attempt>=20:
            q[4*finger:4*finger+4]=[rng.uniform(.25,.9),rng.uniform(-.1,.1) if finger<3 else rng.uniform(1.15,1.6),rng.uniform(.15,.65),rng.uniform(.10,.55)]
        d.qpos[hand_qpos_indices(model)]=q;set_object_static(model,d,target_id,[10,10,10]);mujoco.mj_forward(model,d)
        if any(c.dist < -1e-5 for c in d.contact[:d.ncon]):rejections.append('initial_self_penetration');continue
        rtip=d.site_xmat[sid].reshape(3,3)
        angle=0. if attempt==0 else rng.uniform(0,np.pi/2)
        phi=rng.uniform(-np.pi,np.pi)
        inward=rtip@np.array([-np.cos(angle),np.sin(angle)*np.cos(phi),np.sin(angle)*np.sin(phi)])
        world_vertices=vertices@d.geom_xmat[pad].reshape(3,3).T+d.geom_xpos[pad]
        point=world_vertices[np.argmax(world_vertices@inward)]
        anchor,normal=surface_anchor(root,row,rng.uniform(-.004,.004,2))
        rotation=Rotation.align_vectors([-inward],[normal])[0].as_matrix()
        rotation=Rotation.from_rotvec(-inward*rng.uniform(-np.pi,np.pi)).as_matrix()@rotation
        quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        # Test a straight 6mm approach with the hand held still. Only a tiny
        # 50um probe at the end is allowed, never used as the live initial state.
        accepted=True;touch=False;min_other=1.
        for gap in (.006,.003,0.,-.00005):
            pos=point+inward*gap-rotation@anchor
            set_object_static(model,d,target_id,pos,quat);mujoco.mj_forward(model,d)
            for c in d.contact[:d.ncon]:
                pair={int(c.geom1),int(c.geom2)}
                if not (targets & pair):continue
                others=pair-targets
                if not others:continue
                other=next(iter(others))
                if other==pad and c.dist<1e-5:touch=True
                if other!=pad:
                    min_other=min(min_other,float(c.dist))
                    if c.dist < -1e-5:accepted=False
            if not accepted:break
        if accepted and touch:
            from collections import Counter
            return q,point+inward*.006-rotation@anchor,rotation,inward,anchor,dict(
                attempts=attempt+1,rejections=dict(Counter(rejections)),surface_anchor=anchor.tolist(),
                object_normal=normal.tolist(),approach_axis=inward.tolist(),initial_gap_m=.006,
                endpoint_probe_penetration_m=.00005,other_contact_min_dist_m=min_other,
                scope='static initial placement and straight contact approach only; no trajectory screening')
        rejections.append('non_tip_first' if not accepted else 'no_tip_at_probe')
    raise RuntimeError(f'No legal single-finger initial placement within 160 trials: {row["object_id"]} finger {finger}')


def place_under_palm(model,root,row,q,finger,palm_id,target_id,seed):
    """Object below the palm, active tip first. Static start only, no path plan.

    Palm/object are mocap bodies: their collision may be omitted by MuJoCo's
    welded-body filter. Explicit distance checks must cover the palm as well.
    """
    import mujoco
    from split_fingertip_collision import GEOMS
    from collections import Counter
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,k) or '' for k in range(model.ngeom)]
    pad=next(k for k,n in enumerate(names) if n.endswith(GEOMS[finger]+'_pad_collision'))
    targets=[k for k,n in enumerate(names) if n.startswith('target/') and (model.geom_contype[k] or model.geom_conaffinity[k])]
    palm=next(k for k,n in enumerate(names) if n=='robot/palm_lower_collision')
    mid=model.geom_dataid[pad]
    vertices=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
    d=mujoco.MjData(model);set_palm_static(model,d,palm_id)
    rng=np.random.default_rng(seed+10000);original=q.copy();rejections=[]
    def minimum_distance(hand_geom, cutoff):
        return min(mujoco.mj_geomDistance(model,d,hand_geom,t,cutoff,None) for t in targets)
    for attempt in range(96):
        q=original.copy()
        if attempt:
            q[4*finger:4*finger+4]=[rng.uniform(.45,.9),rng.uniform(-.08,.08) if finger<3 else rng.uniform(1.1,1.5),rng.uniform(.20,.55),rng.uniform(.15,.45)]
        d.qpos[hand_qpos_indices(model)]=q;set_object_static(model,d,target_id,[10,10,10]);mujoco.mj_forward(model,d)
        if any(c.dist < -1e-5 for c in d.contact[:d.ncon]):rejections.append('initial_self_penetration');continue
        center_xy=d.geom_xpos[palm,:2]+rng.uniform(-.012,.012,2)
        rotation=Rotation.from_euler('z',rng.uniform(-np.pi,np.pi)).as_matrix()
        world_vertices=vertices@d.geom_xmat[pad].reshape(3,3).T+d.geom_xpos[pad]
        point=world_vertices[np.argmin(world_vertices[:,2])]
        uv=(rotation.T@np.r_[point[:2]-center_xy,0.])[:2]
        try:anchor,normal=surface_anchor(root,row,uv)
        except ValueError:rejections.append('surface_ray_missed');continue
        if normal[2]<.7:rejections.append('initial_surface_too_steep');continue
        pos=np.r_[center_xy,point[2]-anchor[2]]
        quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        set_object_static(model,d,target_id,pos,quat)
        # Locate the actual collision surface, not the visual mesh. Never
        # spawn inside it: the returned live root is 6mm above first contact.
        lo,hi=pos[2]-.012,pos[2]+.012
        for _ in range(15):
            set_object_static(model,d,target_id,[pos[0],pos[1],(lo+hi)/2]);mujoco.mj_forward(model,d)
            if minimum_distance(pad,.02)>0:lo=(lo+hi)/2
            else:hi=(lo+hi)/2
        pos[2]=(lo+hi)/2
        accepted=True;touch=False;min_other=1.;palm_clearance=1.
        for gap in (.006,.003,0.,-.00005):
            set_object_static(model,d,target_id,pos-[0,0,gap]);mujoco.mj_forward(model,d)
            palm_clearance=min(palm_clearance,minimum_distance(palm,.10))
            if palm_clearance<.015:accepted=False;break
            for c in d.contact[:d.ncon]:
                pair={int(c.geom1),int(c.geom2)}
                if not(set(targets)&pair):continue
                other=next(iter(pair-set(targets)),None)
                if other==pad and c.dist<1e-5:touch=True
                elif other is not None:
                    min_other=min(min_other,float(c.dist))
                    if c.dist < -1e-5:accepted=False
            if not accepted:break
        if accepted and touch:
            return q,pos-[0,0,.006],rotation,np.array([0.,0.,-1.]),anchor,dict(
                attempts=attempt+1,rejections=dict(Counter(rejections)),surface_anchor=anchor.tolist(),
                object_normal=normal.tolist(),approach_axis=[0.,0.,-1.],initial_gap_m=.006,
                endpoint_probe_penetration_m=.00005,other_contact_min_dist_m=min_other,
                palm_surface_clearance_m=palm_clearance,object_xy_offset_from_palm_m=(center_xy-d.geom_xpos[palm,:2]).tolist(),
                layout='object_under_palm; palm faces downward; upward object approach',
                scope='bounded static initial geometry only; no future waypoint planning')
        rejections.append('non_tip_first_or_palm_gap' if not accepted else 'no_tip_at_probe')
    raise RuntimeError(f'No under-palm tip-first root within 96 trials: {row["object_id"]} finger {finger}; {dict(Counter(rejections))}')
