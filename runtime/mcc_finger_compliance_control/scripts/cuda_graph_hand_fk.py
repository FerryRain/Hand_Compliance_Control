"""Optional CUDA-graph replay of the unchanged TorchHandFK operations.

Build before starting an inference worker: graph capture and concurrent CUDA
allocation must not overlap. Clone returned outputs because the next replay
overwrites captured storage. Training/autograd always use the original FK.
"""
from __future__ import annotations
import torch
from torch import nn


class CUDAGraphHandFK(nn.Module):
    def __init__(self, original, batch, device, *, frame_cache=False):
        super().__init__()
        self.original = original
        self.batch = int(batch)
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDAGraphHandFK requires CUDA')
        if self.device.index is None:
            self.device = torch.device('cuda',torch.cuda.current_device())
        self.graphs = {}
        self.frame_cache = frame_cache
        self.cache_active = False
        self.cached_kinematics = None
        self.cached_key = None
        self.cache_hits = 0
        with torch.no_grad():
            for kind in ('sites','kinematics','anchor'):
                q = torch.zeros(batch,16,device=device)
                anchor = torch.zeros(batch,4,3,device=device)
                valid = torch.zeros(batch,4,dtype=torch.bool,device=device)
                def run():
                    if kind == 'sites':return original(q)
                    if kind == 'kinematics':return original.contact_kinematics(q)
                    return original.contact_kinematics(q,anchor,valid)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):run()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output = run()
                self.graphs[kind] = (graph,q,anchor,valid,output)

    def _replay(self, kind, q, anchor=None, valid=None):
        graph,static_q,static_anchor,static_valid,output = self.graphs[kind]
        static_q.copy_(q.reshape(self.batch,16))
        if kind == 'anchor':
            static_anchor.copy_(anchor)
            static_valid.copy_(valid)
        graph.replay()
        if isinstance(output,tuple):return tuple(value.clone() for value in output)
        return output.clone()

    def forward(self, q):
        if (torch.is_grad_enabled() or q.numel() != self.batch*16 or q.device != self.device
                or torch.cuda.is_current_stream_capturing()):
            return self.original(q)
        prefix = q.shape[:-1]
        return self._replay('sites',q).reshape(*prefix,4,3)

    def contact_kinematics(self, q, anchor_body_local=None, anchor_valid=None):
        if (torch.is_grad_enabled() or q.shape != (self.batch,16) or q.device != self.device
                or torch.cuda.is_current_stream_capturing()):
            return self.original.contact_kinematics(q,anchor_body_local,anchor_valid)
        kind = 'anchor' if anchor_body_local is not None and anchor_valid is not None else 'kinematics'
        if kind == 'kinematics' and self.frame_cache and self.cache_active:
            key=(q.data_ptr(),None if q.is_inference() else q._version,q.dtype,tuple(q.shape),tuple(q.stride()))
            if key == self.cached_key:
                self.cache_hits += 1
                # Current consumers treat these geometry tensors as read-only.
                # They are independent clones of captured storage, and remain
                # valid through all anchored and site-only replays this frame.
                return self.cached_kinematics
            result=self._replay(kind,q)
            self.cached_key=key;self.cached_kinematics=result
            return result
        return self._replay(kind,q,anchor_body_local,anchor_valid)

    def begin_frame(self):
        # Warp writes shared q memory without increasing Torch's _version.
        # An explicit per-frame reset is therefore mandatory.
        self.cached_key=None;self.cached_kinematics=None;self.cache_active=True
