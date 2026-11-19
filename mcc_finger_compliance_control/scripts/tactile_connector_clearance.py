"""Known robot connector geometry against measured local tactile planes.

No object mesh/pose or non-tip collision observation is consumed. A tangent
plane is a local approximation; this does not certify clearance from unseen
curvature, other robot links, or self collision.
"""
import numpy as np
import torch
import mujoco

class TactileConnectorClearance:
    def __init__(self,model,device):
        names=['fingertip_geom_connector_collision','fingertip_2_geom_connector_collision',
               'fingertip_3_geom_connector_collision','thumb_fingertip_geom_connector_collision']
        groups=[]
        for name in names:
            g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+name)
            if g<0:raise ValueError('Missing known connector geometry: '+name)
            mesh=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Expected mesh connector')
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.geom_quat[g])
            vertices=model.mesh_vert[start:start+n]@rotation.reshape(3,3).T+model.geom_pos[g]
            groups.append(vertices.astype(np.float32))
        width=max(map(len,groups))
        self.vertices=torch.tensor(np.stack([np.concatenate((v,np.repeat(v[:1],width-len(v),axis=0))) for v in groups]),device=device)

    def minimum(self,fk,q,point_local,outward_local,valid,radius=.025):
        _,_,origin,rot,_=fk.contact_kinematics(q)
        vertices=origin[:,:,None]+torch.einsum('bfij,fvj->bfvi',rot,self.vertices)
        diff=vertices-point_local[:,:,None]
        distance=(diff*outward_local[:,:,None]).sum(-1)
        tangent=diff-distance[...,None]*outward_local[:,:,None]
        patch=(tangent.norm(dim=-1)<radius)&valid[...,None]
        candidate=torch.where(patch,distance,torch.inf)
        values,indices=candidate.min(-1)
        anchors=self.vertices[torch.arange(4,device=q.device)[None],indices]
        available=patch.any(-1)
        return torch.where(available,values,0.),anchors,available

    @torch.no_grad()
    def fit(self,fk,q,preferred,target_sites,lower,upper,point_local,outward_local,valid,iterations=60,clearance=.003):
        x=torch.maximum(torch.minimum(q,upper),lower);eye=torch.eye(4,device=q.device)
        for _ in range(iterations):
            sites,_,_,_,jac=fk.contact_kinematics(x);jt=jac.transpose(-1,-2)
            distance,anchor,has=self.minimum(fk,x,point_local,outward_local,valid)
            _,_,_,_,cj=fk.contact_kinematics(x,anchor,has)
            jn=torch.einsum('bfi,bfij->bfj',outward_local,cj)
            active=has&(distance<clearance)
            weight=20*active.to(q.dtype)
            a=jt@jac+weight[...,None,None]*jn[...,None]*jn[...,None,:]+1e-5*eye
            rhs=(jt@(target_sites-sites)[...,None]).squeeze(-1)+1e-5*(preferred-x).reshape(-1,4,4)
            rhs+=weight[...,None]*jn*(clearance-distance).clamp_min(0)[...,None]
            step=torch.linalg.solve(a,rhs[...,None]).squeeze(-1).reshape(-1,16).clamp(-.04,.04)
            x=torch.maximum(torch.minimum(x+step,upper),lower)
        return x
