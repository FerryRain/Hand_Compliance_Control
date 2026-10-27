"""CUDA form of the existing causal constant-velocity prediction baseline."""
import torch


def absolute_kinematic_chunk(q,velocity,horizon,waypoint_dt,velocity_clip):
    if horizon<=0 or waypoint_dt<=0 or velocity_clip<=0:raise ValueError('invalid kinematic baseline settings')
    if q.ndim!=2 or q.shape!=velocity.shape or q.shape[-1]!=16:raise ValueError('expected matching [batch,16] state and velocity')
    q=q.float();velocity=velocity.to(device=q.device,dtype=q.dtype).clamp(-velocity_clip,velocity_clip)
    times=waypoint_dt*torch.arange(1,horizon+1,device=q.device,dtype=q.dtype)
    return q[:,None]+times[None,:,None]*velocity[:,None]
