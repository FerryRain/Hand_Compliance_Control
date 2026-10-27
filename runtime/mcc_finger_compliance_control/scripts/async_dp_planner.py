"""Bounded asynchronous inference for the accepted anchored-delta DP sampler.

Inference owns the model and RNG; the control thread owns DTW, MCC and physics.
Snapshots are immutable and their q anchor is retained through decoding.
Artificial delays are delivery gates in CONTROL FRAMES, not wall-clock sleeps:
they remain meaningful when simulation itself runs slower than real time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import time

import torch


@dataclass
class DPPlan:
    request_frame: int
    history: torch.Tensor
    normalized: torch.Tensor
    anchor: torch.Tensor
    observation: torch.Tensor | None
    prediction: torch.Tensor
    chunk: torch.Tensor
    wall_ms: float
    gpu_span_ms: float | None
    noise_before: str
    noise_after: str
    event: object = None
    worker_start_frame: int | None = None
    worker_finish_frame: int | None = None


class AsyncDPPlanner:
    """One outstanding request; requests never queue behind stale observations."""

    def __init__(self, chain, *, inline=False, minimum_latency_frames=0,
                 max_age_frames=20, buffer_frames=30):
        if minimum_latency_frames < 0 or max_age_frames < 0:
            raise ValueError('Latency and maximum age must be nonnegative')
        if inline and minimum_latency_frames:
            raise ValueError('Inline mode is the zero-delay parity probe')
        self.chain = chain
        self.inline = inline
        self.minimum_latency_frames = int(minimum_latency_frames)
        self.max_age_frames = int(max_age_frames)
        # One extra waypoint determines derivatives at the last boundary.
        capacity = (chain.runtime.pred_horizon-chain.scheduler.config.max_drop-1)*chain.scheduler.step
        self.buffer_frames = min(int(buffer_frames), capacity)
        if self.buffer_frames < chain.scheduler.config.replan_interval:
            raise ValueError('Checkpoint horizon cannot buffer one replan interval')
        self.device = chain.device
        self.stream = (None if inline or self.device.type != 'cuda' else torch.cuda.Stream(device=self.device))
        self.executor = None if inline else ThreadPoolExecutor(max_workers=1,thread_name_prefix='dp-inference')
        self.pending = None
        self.request_frame = None
        self.closed = False
        self.submitted = 0
        self.busy_requests = 0
        self.expired = 0
        self.results = []
        self.installs = []
        self.pending_control_frames = []
        self.running_control_frames = []

    @property
    def busy(self):
        return self.pending is not None

    def submit(self, frame, q, history, normalized, observation=None):
        if self.closed:
            raise RuntimeError('Planner has been closed')
        if self.busy:
            self.busy_requests += 1
            return False
        # Warp writes q in place; the clone is necessary even if the history
        # tensor already came from a gather into independent storage.
        snapshot = dict(frame=int(frame), q=q.detach().clone(), history=history.detach().clone(),
                        normalized=normalized.detach().clone(),
                        observation=None if observation is None else observation.detach().clone())
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        self.request_frame = int(frame)
        self.submitted += 1
        if self.inline:
            self.pending = self._infer(snapshot,ready)
        else:
            self.pending = self.executor.submit(self._infer,snapshot,ready)
        return True

    @torch.no_grad()
    def _infer(self, snapshot, ready):
        if self.stream is None:
            return self._sample(snapshot)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            return self._sample(snapshot)

    def _sample(self, snapshot):
        runtime = self.chain.runtime
        # The generator is accessed only by this single worker.
        before = hashlib.sha256(runtime.generator.get_state().numpy().tobytes()).hexdigest()
        start = time.perf_counter()
        start_frame = int(getattr(self.chain,'control_frame',snapshot['frame']))
        begin = end = None
        if self.device.type == 'cuda' and not self.inline:
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
        condition = runtime.policy.diffusion._prepare_global_conditioning({
            'observation.state':snapshot['normalized'][:,:,:runtime.robot_state_dim],
            'observation.environment_state':snapshot['normalized'][:,:,runtime.robot_state_dim:]})
        prediction = runtime.policy.diffusion.conditional_sample(
            self.chain.batch,global_cond=condition,generator=runtime.generator)
        chunk = prediction*self.chain.action_std+self.chain.action_mean+snapshot['q'][:,None]
        if end is not None:
            end.record()
            # Synchronize the WORKER stream only. The control loop never waits
            # for this result; it polls the Future and executes old commands.
            end.synchronize()
        after = hashlib.sha256(runtime.generator.get_state().numpy().tobytes()).hexdigest()
        return DPPlan(snapshot['frame'],snapshot['history'],snapshot['normalized'],snapshot['q'],
            snapshot['observation'],prediction,chunk,(time.perf_counter()-start)*1000,
            None if end is None else begin.elapsed_time(end),before,after,end,
            start_frame,int(getattr(self.chain,'control_frame',snapshot['frame'])))

    def poll(self, frame):
        if not self.busy:
            return None
        age = int(frame)-self.request_frame
        self.pending_control_frames.append(int(frame))
        # An inline result is immediately available. Ordinary futures are
        # inspected without result()/wait() until they have completed.
        if not self.inline and not self.pending.done():
            self.running_control_frames.append(int(frame))
            return None
        if age < self.minimum_latency_frames:
            return None
        result = self.pending if self.inline else self.pending.result()
        self.pending = None
        row = dict(request_frame=result.request_frame,delivery_frame=int(frame),age_frames=age,
                   compute_wall_ms=result.wall_ms,gpu_span_ms=result.gpu_span_ms,
                   noise_before=result.noise_before,noise_after=result.noise_after)
        row.update(worker_start_frame=result.worker_start_frame,worker_finish_frame=result.worker_finish_frame,
                   sim_frames_advanced_during_inference=result.worker_finish_frame-result.worker_start_frame)
        row['expired'] = age > self.max_age_frames
        self.results.append(row)
        if row['expired']:
            self.expired += 1
            return None
        if self.device.type == 'cuda' and result.event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(result.event)
            # These allocations were produced on a different stream; keep
            # them alive until the control-stream DTW/gather/trace finishes.
            for value in (result.prediction,result.chunk):
                value.record_stream(current)
        return result

    def installed(self, frame, plan, drop):
        self.installs.append(dict(request_frame=plan.request_frame,install_frame=int(frame),
            age_frames=int(frame)-plan.request_frame,dtw_drop=drop.detach().cpu().tolist()))

    def report(self):
        return dict(mode='inline' if self.inline else 'thread_and_cuda_stream',
            minimum_latency_frames=self.minimum_latency_frames,max_age_frames=self.max_age_frames,
            buffer_frames=self.buffer_frames,submitted=self.submitted,installed=len(self.installs),
            busy_requests=self.busy_requests,expired=self.expired,outstanding_on_exit=self.busy,
            results=list(self.results),installs=list(self.installs),
            pending_control_frames=list(self.pending_control_frames),
            running_control_frames=list(self.running_control_frames),
            anchor='Frozen measured q at the observation snapshot, never reanchored at delivery',
            alignment='Existing DTW against issued nominal history, unchanged; no fixed elapsed-time waypoint truncation',
            delay_semantics='Delivery floor in simulated control frames; compute_wall_ms is measured worker latency, including GPU contention')

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.executor is not None:
            self.executor.shutdown(wait=True,cancel_futures=True)
        self.pending = None
