"""Object-side deadband height control; independent of finger/MCC commands."""
import numpy as np
from scipy.spatial.transform import Rotation


def surface_heights(planes, poses, xy):
    """Vertical ray upper exits of convex union at fixed world XY samples.

    Missing coverage returns NaN. Spatial samples are not a full collision proof.
    """
    rot=Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
    origins=np.c_[xy,np.zeros(len(xy))]
    local=np.einsum('tji,tpj->tpi',rot,origins[None]-poses[:,None,:3])
    direction=rot[:,2,:]
    top=np.full(local.shape[:2],-np.inf)
    for e in planes:
        val=np.einsum('tpi,fi->tpf',local,e[:,:3])+e[:,3]
        den=direction@e[:,:3].T
        with np.errstate(divide='ignore',invalid='ignore'):
            crossing=-val/den[:,None,:]
        hi=np.min(np.where(den[:,None,:]>1e-9,crossing,np.inf),axis=-1)
        lo=np.max(np.where(den[:,None,:]<-1e-9,crossing,-np.inf),axis=-1)
        valid=(hi>=lo-1e-8)&np.all((abs(den[:,None,:])>1e-9)|(val<=1e-8),axis=-1)
        top=np.maximum(top,np.where(valid,hi,-np.inf))
    return np.where(np.isfinite(top),top,np.nan)


def smooth_band(lower,upper,dt=.01,hysteresis=.001,vmax=.04,amax=.4):
    """Hold height offset inside band; latch nearest interior edge outside.

    Finite-rate motion may temporarily violate a moving band: report it rather
    than teleport or claim safety. Empty bands prioritize the near/palm edge.
    """
    lower=np.asarray(lower);upper=np.asarray(upper)
    if lower.shape!=upper.shape or lower.ndim!=1 or not np.isfinite(lower+upper).all():raise ValueError('Invalid band')
    if lower[0]>1e-6 or upper[0]<-1e-6:raise ValueError('Initial pose outside distance band')
    n=len(lower);offset=np.zeros(n);velocity=np.zeros(n);trigger=np.zeros(n,dtype=np.int8);target=0.
    for t in range(1,n):
        x=offset[t-1];v=velocity[t-1]
        width=max(0.,upper[t]-lower[t]);margin=min(hysteresis,width*.25)
        if x>upper[t]:target=upper[t]-margin;trigger[t]=-1
        elif x<lower[t] and width>0:target=lower[t]+margin;trigger[t]=1
        error=target-x
        desired=np.clip(20.*error,-vmax,vmax)
        v+=np.clip(desired-v,-amax*dt,amax*dt)
        offset[t]=x+v*dt;velocity[t]=v
    return offset,velocity,trigger
