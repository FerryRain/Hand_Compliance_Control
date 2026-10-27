"""Opt-in graph replay of stateless control kernels, preserving original math.

Prepare after scheduler priming and profile/oracle setup, before DP workers
submit their first request. Stateful MCC filtering/recovery stays in Python;
only its pure final projection is captured. No algorithm or safety change.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch


def _clone(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {k:_clone(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_clone(v) for v in value)
    raise TypeError(type(value))


class GraphFunction:
    def __init__(self,fn,inputs,device):
        self.inputs=inputs
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(),torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph):
            try:self.outputs=fn()
            except Exception:
                # capture_end() can replace the original exception with a
                # generic invalidated-capture error. Preserve the root cause.
                import traceback
                traceback.print_exc()
                raise

    def __call__(self,*inputs):
        for target,value in zip(self.inputs,inputs):target.copy_(value)
        self.graph.replay()
        return _clone(self.outputs)


class GraphChunkInstall:
    def __init__(self,scheduler,horizon,frame_counts):
        if scheduler.count!=scheduler.capacity:
            raise ValueError('Prime the full scheduler history before capture')
        self.scheduler=scheduler
        self.original=scheduler.install
        self.config=tuple(vars(scheduler.config).items())
        self.history_ptr=scheduler.history.data_ptr();self.scale_ptr=scheduler.scale.data_ptr()
        self.graphs={}
        saved=(scheduler.path,scheduler.path_index,scheduler.last_drop_index)
        try:
            for frames in sorted(set(frame_counts)):
                prediction=torch.zeros(scheduler.batch,horizon,len(scheduler.scale),dtype=torch.float64,device=scheduler.device)
                def run():
                    drop=self.original(prediction,execution_frames=frames)
                    return scheduler.path,drop
                self.graphs[frames]=GraphFunction(run,(prediction,),scheduler.device)
        finally:
            scheduler.path,scheduler.path_index,scheduler.last_drop_index=saved
        self.bound_buffers=(scheduler.history,scheduler.scale,scheduler._history_indices)

    def __call__(self,prediction,*,execution_frames=None):
        s=self.scheduler;frames=s.config.replan_interval if execution_frames is None else int(execution_frames)
        if (frames not in self.graphs or s.count!=s.capacity or s.history.data_ptr()!=self.history_ptr
                or s.scale.data_ptr()!=self.scale_ptr or tuple(vars(s.config).items())!=self.config
                or prediction.shape!=self.graphs[frames].inputs[0].shape):
            return self.original(prediction,execution_frames=execution_frames)
        path,drop=self.graphs[frames](prediction)
        s.path=path;s.path_index=0;s.last_drop_index=drop
        return drop


class GraphMeshQuery:
    def __init__(self,oracle,point_count):
        self.oracle=oracle;self.original=oracle.query_surface_object_frame
        self.geometry=(oracle.triangles,oracle.faces,oracle.vertex_normals,oracle.face_normals)
        self.geometry_ptrs=tuple(value.data_ptr() for value in self.geometry)
        self.chunk_triangles=oracle.chunk_triangles
        points=torch.zeros(point_count,3,device=oracle.device)
        self.function=GraphFunction(lambda:self.original(points),(points,),oracle.device)

    def __call__(self,points):
        buffers=(self.oracle.triangles,self.oracle.faces,self.oracle.vertex_normals,self.oracle.face_normals)
        if (points.shape!=self.function.inputs[0].shape or torch.is_grad_enabled()
                or tuple(value.data_ptr() for value in buffers)!=self.geometry_ptrs
                or self.oracle.chunk_triangles!=self.chunk_triangles):
            return self.original(points)
        return self.function(points)


class GraphContactProjection:
    def __init__(self,projection):
        self.projection=projection;self.original=projection.project
        chain=projection.chain;device=chain.device;n=chain.batch
        self.config_key=self._key()
        self.bound_buffers=(chain.mcc.lower,chain.mcc.upper,chain.mcc.force_setpoint)
        self.buffer_ptrs=tuple(value.data_ptr() for value in self.bound_buffers)
        self.fk=chain.mcc.fk
        q=torch.zeros(n,16,device=device);command=q.clone();previous=q.clone()
        pose=torch.zeros(n,7,device=device);pose[:,3]=1
        twist=torch.zeros(n,6,device=device)
        self.observations={
            'physical_contact_point':torch.zeros(n,4,3,device=device),
            'physical_normal_inward':torch.zeros(n,4,3,device=device),
            'physical_normal_force':torch.zeros(n,4,device=device),
            'physical_normal_found':torch.zeros(n,4,dtype=torch.bool,device=device)}
        original_observer=projection.observer
        projection.observer=SimpleNamespace(latest=self.observations)
        fk=chain.mcc.fk
        original_cache_active=getattr(fk,'cache_active',None)
        if original_cache_active is not None:fk.cache_active=False
        try:
            self.function=GraphFunction(lambda:self.original(q,command,pose,twist,previous),
                (q,command,pose,twist,previous,*self.observations.values()),device)
        finally:
            projection.observer=original_observer
            if original_cache_active is not None:fk.cache_active=original_cache_active
        self.bound_hard_ratios=projection._hard_ratios

    def __call__(self,q,command,pose,twist,previous=None):
        mcc=self.projection.chain.mcc
        if (self._key()!=self.config_key or mcc.fk is not self.fk
                or tuple(value.data_ptr() for value in (mcc.lower,mcc.upper,mcc.force_setpoint))!=self.buffer_ptrs):
            return self.original(q,command,pose,twist,previous)
        x=self.projection.observer.latest
        return self.function(q,command,pose,twist,q if previous is None else previous,
                             *(x[name] for name in self.observations))

    def _key(self):
        cfg=self.projection.chain.mcc.config
        fields=('natural_flexion_floor','contact_anchor_max_site_distance',
            'solver_contact_safety_force_ratio','overforce_hard_ratio','thumb_overforce_hard_ratio',
            'solver_contact_escape_step','action_rate_limit','control_dt')
        return (self.projection.include_palm_motion,self.projection.max_live_joint_step_rad,self.projection.bound_all_commands,
                *(getattr(cfg,name) for name in fields))


class ControlGraphKernels:
    def __init__(self,chain):
        # No inference may be running during graph capture.
        if chain.async_planner is not None and chain.async_planner.busy:
            raise RuntimeError('Prepare control graphs before asynchronous inference')
        counts=[chain.scheduler.config.replan_interval]
        if chain.async_planner is not None:counts.append(chain.async_planner.buffer_frames)
        print('[CONTROL GRAPHS] Preparing scheduler',flush=True)
        self.chunk=GraphChunkInstall(chain.scheduler,chain.runtime.pred_horizon,counts)
        chain.scheduler.install=self.chunk
        self.mesh=None;self.projection=None
        if chain.mesh_oracle is not None:
            print('[CONTROL GRAPHS] Preparing source-mesh query',flush=True)
            self.mesh=GraphMeshQuery(chain.mesh_oracle,chain.batch*4)
            chain.mesh_oracle.query_surface_object_frame=self.mesh
        if hasattr(chain,'paired_contact_projection'):
            print('[CONTROL GRAPHS] Preparing matched-contact projection',flush=True)
            self.projection=GraphContactProjection(chain.paired_contact_projection)
            chain.paired_contact_projection.project=self.projection
