"""Vectorized geometric Jacobian for four simultaneous native pad tasks.

Contact qualification and all guards remain in their original modules. This
helper only removes repeated scalar derivative work from the IK residual.
"""
import mujoco
import numpy as np
from .belly_contact import registered_belly_frame


def skew(vector):
    x,y,z=vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def support_derivative(frame,sizes,normal,angular):
    shape=(frame*(sizes*sizes)[None,:])@frame.T
    vector=shape@normal;radius=float(np.sqrt(normal@vector))
    if radius<=1e-12:raise ValueError('Ellipsoid support radius must be positive')
    change=(shape@skew(normal)-skew(vector))@angular
    return change/radius-vector[:,None]*(normal@change)[None,:]/(2.*radius**3)


def fast_belly_ik_terms(model,data,geom_id,surface_point_world,object_outward_normal,
                       *,clearance_m=0.,orientation_weight_m=.01,dof_indices=None):
    """Same six residuals/Jacobian; no classification diagnostic is generated."""
    normal=np.asarray(object_outward_normal,float);normal=normal/np.linalg.norm(normal)
    geom=int(geom_id);frame=registered_belly_frame(model,data,geom)
    sizes=np.asarray(model.geom_size[geom]);local=frame.T@normal
    support=frame@(sizes*sizes*local/np.linalg.norm(sizes*local))
    point=data.geom_xpos[geom]-support
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacGeom(model,data,jp,jr,geom)
    if dof_indices is not None:jp,jr=jp[:,dof_indices],jr[:,dof_indices]
    residual=np.r_[point-np.asarray(surface_point_world)-clearance_m*normal,
                   orientation_weight_m*(frame[:,2]+normal)]
    jac=np.vstack((jp-support_derivative(frame,sizes,normal,jr),-orientation_weight_m*skew(frame[:,2])@jr))
    return residual,jac,None
