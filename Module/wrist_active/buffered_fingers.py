"""Serve the released single_all policy without a GPU barrier in the paced control loop."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np

from ..i04_geometric_traversal.fingertip_dp_worker import CHUNK_OUTPUT_BYTES, WIRE
from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy


class BufferedFingers(LocalFingerPolicy):
    lead_ticks = 5

    def __init__(self, seed, frames, reference, limits, *, paced):
        super().__init__(seed, frames)
        self.lower = np.maximum(reference-.48, limits[:,0]+.01)
        self.upper = np.minimum(reference+.48, limits[:,1]-.01)
        self.nominal = reference.copy()
        self.pending = None
        self.request_tick = -1
        self.frames = frames
        self.paced = paced
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='single-all-rpc')
        self.missed = self.underrun = self.skipped = self.installs = 0
        self.info.update(schedule='100ms chunks, 50ms buffered handoff', lead_s=.05,
                         execution_envelope_rad=.48, nominal_rate_rad_s=1.8)

    def query(self, packet):
        started = perf_counter()
        self.worker.stdin.write(packet)
        self.worker.stdin.flush()
        reply = self.worker.stdout.read(CHUNK_OUTPUT_BYTES)
        if len(reply) != CHUNK_OUTPUT_BYTES:
            raise RuntimeError("Incomplete single_all action chunk")
        chunk = np.frombuffer(reply, dtype=WIRE)
        if not np.isfinite(chunk).all():
            raise ValueError("Nonfinite single_all action chunk")
        return chunk[:-1].reshape(16,16), float(chunk[-1]), 1000*(perf_counter()-started)

    def command(self, frame, packet):
        started = perf_counter()
        measured = self.observation.push(packet)
        self.scheduler.observe(self.nominal)
        self.local_ms.append(1000*(perf_counter()-started))
        if self.pending is not None and frame >= self.request_tick+self.lead_ticks:
            # Offline simulation uses a fixed logical latency. Paced deployment
            # consumes the previous buffer while the worker is outstanding.
            if not self.paced or self.pending.done():
                chunk, compute, rpc = self.pending.result()
                self.inference_ms.append(compute)
                self.rpc_ms.append(rpc)
                started = perf_counter()
                self.scheduler.install(chunk)
                self.install_ms.append(1000*(perf_counter()-started))
                self.installs += 1
                self.pending = None
            elif frame == self.request_tick+self.lead_ticks:
                self.missed += 1
        if (frame >= self.bootstrap and frame+self.lead_ticks < self.frames and
                (frame-self.bootstrap) % self.scheduler.config.replan_interval == 0):
            if self.pending is None:
                request = np.r_[np.float32(frame), measured, self.observation.history().ravel()].astype(WIRE)
                self.pending = self.pool.submit(self.query, request.tobytes())
                self.request_tick = frame
            else:
                self.skipped += 1
        if self.scheduler.path_index < len(self.scheduler.path):
            target = np.clip(self.scheduler.next_command(), self.lower, self.upper)
            self.nominal = self.nominal + np.clip(target-self.nominal,-.018,.018)
        elif frame >= self.bootstrap+self.lead_ticks:
            self.underrun += 1
        return self.nominal

    def __exit__(self, *exc):
        try:
            self.pool.shutdown(wait=True)
            if self.pending is not None:
                self.pending.result()
        finally:
            super().__exit__(*exc)
