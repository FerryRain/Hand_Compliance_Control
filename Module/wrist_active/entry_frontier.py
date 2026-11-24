"""Continue from a measured support even when its GPIS mesh component is open."""

import numpy as np
import torch

from ..acitive_02.active_gpis import NOISE_VARIANCE
from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import limit
from .contact_first import signed_gap
from .online_tactile_mesh import GRID


def chart_options(surface, model, stations, wrist, attitude, empty):
    """A short local chart at a measured pad; rank its tip using the global posterior."""
    anchors=[]
    bound=min(.25,.055+.0018*empty)
    for s in stations:
        travel=np.linalg.norm(wrist-s.position)
        angle=(s.attitude*attitude.inv()).magnitude()
        if travel>bound or angle>min(1.80,.50+.018*empty):
            continue
        for j,(p,n) in enumerate(zip(s.points,s.normals)):
            anchors.append((s,j,p,n,travel,angle))
    if not anchors or not model.count:
        return []
    xyz=np.stack([a[2] for a in anchors])
    normals=np.stack([a[3] for a in anchors])
    # The coverage gradient points out of the locally observed patch. In its
    # normal plane a dense, already sampled strip cannot nominate its own middle.
    dist,idx=surface.tree.query(xyz,k=min(32,len(surface.points)))
    dist=np.asarray(dist).reshape(len(xyz),-1)
    idx=np.asarray(idx).reshape(len(xyz),-1)
    w=np.exp(-.5*(dist/.019)**2)*(np.einsum('aki,ai->ak',surface.normals[idx],normals)>.60)
    slope=np.einsum('ak,aki->ai',w,xyz[:,None]-surface.points[idx])/np.maximum(w.sum(1)[:,None],1.)
    delta=slope-(slope*normals).sum(1)[:,None]*normals
    size=np.linalg.norm(delta,axis=1)
    end=xyz+np.clip(size[:,None]*4.,.010,.022)*delta/np.maximum(size[:,None],1.e-9)
    missing=surface.tree.query(end)[0]
    good=np.flatnonzero((size>.0005)&(missing>.0015)&(missing<.024))
    if not len(good):
        return []
    # Predict the information of all nominated points at once, using exactly
    # the same conditional covariance as the triangle-based route score.
    with torch.inference_mode():
        white=torch.linalg.solve_triangular(model.chol,model.features(end[good]).T,upper=False)
        prior=NOISE_VARIANCE+white.square().sum(0)
        gain=(.5*torch.log(prior/(prior-(model.conditional@white).square().sum(0)).clamp_min(NOISE_VARIANCE))).cpu().numpy()
    out=[]
    for k,ig in zip(good,gain):
        s,j,p,n,travel,angle=anchors[k]
        value=float(ig*np.clip(missing[k]/.018,0.,1.))
        quick=value*(.54+.12*len(s.fingers))/(.44+travel/.09+angle)
        if value>1.e-7:
            out.append((quick,s,j,(end[k],value)))
    return out


def verify_chart(surface, point, normal, option):
    proposal,value=option
    patch=surface.patch(point,preferred_normal=normal)
    if patch is None:
        return None
    gap,direction=signed_gap(patch,proposal,patch.predict(point)[0][0])
    if abs(gap)>.010 or normal@direction<.50:
        return None
    end=proposal-gap*direction
    return end,direction,end[None],float(np.linalg.norm(end-point)),value


def unload_link(surface, sensed, info, state, donor):
    """Rotate about the live pad to clear a measured bearing link on a mapped patch."""
    vertices=sensed
    if vertices is None or not len(vertices) or surface.tree is None:
        return
    near,idx=surface.tree.query(vertices)
    normals=surface.normals[idx]
    diff=vertices-surface.points[idx]
    gap=np.einsum('vi,vi->v',diff,normals)
    supported=(near<.020)&(np.abs(gap)<.010)
    if not supported.any():
        return
    # Estimate the contacting part of the link; keep close normals on the same
    # observed sheet so opposing faces cannot cancel the separation direction.
    best=int(np.argmin(np.where(supported,near,np.inf)))
    take=supported&(normals@normals[best]>.55)&(near<near[best]+.005)
    pivot=state['pads'][donor]
    direction=unit(normals[take].mean(0))
    jac=np.cross(vertices[take].mean(0)-pivot,direction)
    if jac@jac<.014**2:
        return
    speed=info['tangent_speed_m_s']
    heading=np.asarray(state['bearing'])
    omega=np.asarray(info['angular_velocity'])
    closure=direction@(speed*heading)+jac@omega
    extra=float(state['side_load'])
    separation=np.clip(.007*(extra-1.2),.003,.026)
    if closure>=separation:
        return
    increment=limit((separation-closure)*jac/(jac@jac+.016**2),.32)
    if 'nominal_flow' not in info:
        info['nominal_flow']=dict(tangent_speed_m_s=speed,height_speed_m_s=info['height_speed_m_s'],
                                  angular_velocity=info['angular_velocity'],selected_direction=heading.tolist())
    info.update(handoff_choice='support_roll',handoff_anchor=donor,side_load_n=extra,
                angular_velocity=(omega+increment).tolist(),link_clearance_gain_m_s=float(jac@increment))


class BatchPosterior:
    def __init__(self,model):
        self.base=model
        self.enabled=False

    def __getattr__(self,key):
        return getattr(self.base,key)

    def update(self,surface,tick):
        base=self.base
        new=len(surface.points)-base.count
        if (self.enabled and base.mesh is not None and len(base.mesh.faces) and 0<new<24 and
                tick-base.last_fit_tick<105):
            d,idx=base.mesh.tree.query(surface.points[base.count:])
            if not np.any((d>1.5*GRID)|((base.mesh.normals[idx]*surface.normals[base.count:]).sum(1)<.38)):
                return False
        return base.update(surface,tick)
