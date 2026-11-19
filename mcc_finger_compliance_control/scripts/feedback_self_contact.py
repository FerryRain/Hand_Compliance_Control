"""Read-only privileged robot self-contact point force for reward/audit only."""
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn


@wp.kernel
def self_contact_max(cone:int,frame:wp.array(dtype=wp.mat33),
    friction:wp.array(dtype=wp.types.vector(length=5,dtype=float)),dim:wp.array(dtype=int),
    address:wp.array2d(dtype=int),efc:wp.array2d(dtype=float),njmax:int,
    count:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),world:wp.array(dtype=int),
    hand:wp.array(dtype=int),peak:wp.array(dtype=float)):
    c=wp.tid()
    if c>=count[0]:return
    a=geom[c][0];b=geom[c][1]
    if a<0 or b<0:return
    if hand[a]==0 or hand[b]==0:return
    f=contact_force_fn(cone,frame,friction,dim,address,efc,njmax,count,world[c],c,True)
    wp.atomic_max(peak,world[c],wp.length(wp.spatial_top(f)))


class SelfContactDiagnostic:
    def __init__(self,env,hand_mask):
        self.env=env
        with wp.ScopedDevice(env.sim.wp_device):
            self.hand=wp.array(np.asarray(hand_mask,dtype=np.int32),dtype=wp.int32)
            self.peak=wp.zeros(env.num_envs,dtype=float)

    def read(self):
        env=self.env;d=env.sim.wp_data
        stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with wp.ScopedDevice(env.sim.wp_device):
            self.peak.zero_()
            wp.launch(self_contact_max,dim=d.naconmax,inputs=[env.sim.wp_model.opt.cone,
                d.contact.frame,d.contact.friction,d.contact.dim,d.contact.efc_address,
                d.efc.force,d.njmax,d.nacon,d.contact.geom,d.contact.worldid,self.hand],outputs=[self.peak])
        torch.cuda.current_stream().wait_stream(stream)
        return wp.to_torch(self.peak).clone()
