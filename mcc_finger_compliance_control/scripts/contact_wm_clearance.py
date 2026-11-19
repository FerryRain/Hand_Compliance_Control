"""Collection geometry checks, independent of live force/contact observations.

MuJoCo native mesh distance can return zero for separated convex pairs. A
positive separating-plane bound proves separation without trusting that query.
This bound is never substituted for a sensor measurement or a dynamics label.
"""
import numpy as np
from scipy.spatial import ConvexHull


def separation_lower_bound(a, b, planes_a, planes_b):
    """Positive result is a rigorous distance lower bound for convex sets.

    Nonpositive results do NOT prove overlap: edge-edge axes are not included.
    """
    return float(max(np.min(a @ planes_b[:, :3].T + planes_b[:, 3], axis=0).max(),
                     np.min(b @ planes_a[:, :3].T + planes_a[:, 3], axis=0).max()))


def world_convex(model, data, geom, local=None):
    if local is None:
        mesh=int(model.geom_dataid[geom])
        if mesh<0:raise ValueError('Expected a convex mesh geometry')
        vertices=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]].astype(float)
        local=(vertices,ConvexHull(vertices).equations)
    vertices,planes=local;R=data.geom_xmat[geom].reshape(3,3);p=data.geom_xpos[geom]
    normal=planes[:,:3]@R.T
    return vertices@R.T+p,np.c_[normal,planes[:,3]-normal@p]


class FixedTipClearance:
    def __init__(self,model,data,pad,targets):
        self.model,self.data,self.pad=model,data,pad
        self.targets=np.asarray(targets);self.local={}
        for g in [pad,*targets]:
            mesh=int(model.geom_dataid[g])
            vertices=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]].astype(float)
            self.local[g]=(vertices,ConvexHull(vertices).equations)
        self.corrected_queries=0;self.max_proven_false_contact_gap=0.

    def __call__(self):
        import mujoco
        m,d,pad=self.model,self.data,self.pad
        lower=np.linalg.norm(d.geom_xpos[self.targets]-d.geom_xpos[pad],axis=-1)
        lower-=m.geom_rbound[self.targets]+m.geom_rbound[pad]
        nearest=.05
        a,pa=world_convex(m,d,pad,self.local[pad])
        for idx in np.argsort(lower):
            if lower[idx]>max(nearest,0.):break
            geom=int(self.targets[idx])
            value=mujoco.mj_geomDistance(m,d,pad,geom,.05,None)
            if value<=1.e-5:
                b,pb=world_convex(m,d,geom,self.local[geom])
                bound=separation_lower_bound(a,b,pa,pb)
                if bound>max(value,0.)+1.e-8:
                    self.corrected_queries+=1
                    self.max_proven_false_contact_gap=max(self.max_proven_false_contact_gap,bound)
                    value=bound
            nearest=min(nearest,value)
        return float(nearest)
