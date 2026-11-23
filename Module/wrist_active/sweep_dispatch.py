"""Release the 5 Hz wrist computation after the current 100 Hz servo update."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch

import torch

from . import sweep_experiment as runtime, wrist_worker
from .experiment import digest, save_json
from .load_guard import GuardedWrist
from .sweep_paced import timer_resolution
from .sweep_score import score


class FrameRelease:
    def __init__(self):
        self.pending = None
        self.count = 0

    def arm(self):
        if self.pending is not None:
            raise RuntimeError('The preceding servo frame has not finished')
        self.pending = Event()
        self.count += 1

    def release(self):
        if self.pending is not None:
            self.pending.set()
            self.pending = None


class AfterFramePool(ThreadPoolExecutor):
    def __init__(self, clock, **kwargs):
        super().__init__(**kwargs)
        self.clock = clock

    def submit(self, fn, /, *args, **kwargs):
        event = self.clock.pending
        if event is None:
            return super().submit(fn, *args, **kwargs)

        def after_frame():
            event.wait()
            return fn(*args, **kwargs)

        return super().submit(after_frame)


@contextmanager
def phased_dispatch(clock):
    class WristAfterFrame(wrist_worker.WristWorker):
        def submit(self, data, touch, tick, lead):
            if lead:
                clock.arm()
            return super().submit(data, touch, tick, lead)

        def __exit__(self, *exc):
            clock.release()
            super().__exit__(*exc)

    # robot_state is the last sensor update in a frame. The background worker
    # already owns the earlier, immutable snapshot; it never reads this packet.
    update_sensors = runtime.robot_state

    def finish_frame(*args):
        update_sensors(*args)
        clock.release()

    original_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(.0005)
        with (patch.object(runtime, 'WristWorker', WristAfterFrame),
              patch.object(wrist_worker, 'ThreadPoolExecutor', partial(AfterFramePool, clock)),
              patch.object(runtime, 'robot_state', finish_frame), timer_resolution()):
            yield
    finally:
        clock.release()
        sys.setswitchinterval(original_interval)


def trial(root):
    guard = root / 'force-priority'
    frozen = json.loads((guard / 'protocol.json').read_bytes())
    source = frozen['source']
    here = Path(__file__)
    if any(digest(here.with_name(k)) != v for k, v in source.items()):
        raise ValueError('Guarded wrist source differs from the registered comparison')
    source = {**source, here.name: digest(here), 'sweep_paced.py': digest(here.with_name('sweep_paced.py'))}
    folder = guard / 'paced-dispatch/pear/active-02/sweep-dp'
    clock = FrameRelease()
    with (phased_dispatch(clock), patch.object(runtime, 'SweepWrist', GuardedWrist),
          patch.object(runtime, 'hashes', return_value=source)):
        result = runtime.run(folder, 'pear', 'active-02', 'sweep-dp', 120., paced=True)
    base = json.loads((guard / 'paced/pear/active-02/sweep-dp/run.json').read_bytes())
    for key in ('initial_sha256', 'model_sha256', 'seed'):
        if result[key] != base[key]:
            raise ValueError(f'Unmatched timing trial: {key}')
    memo = folder / 'dispatch.json'
    if clock.count:
        save_json(memo, dict(deferred=clock.count, switch_interval_s=.0005,
                             comparison_trace_sha256=base['trace_sha256'], trace_sha256=result['trace_sha256']))
    elif not memo.exists():
        raise ValueError('Missing dispatch receipt for an existing trace')
    score(folder, folder, 'pear', 'active-02', [60.,120.])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=runtime.OUT)
    torch.set_num_threads(1)
    trial(parser.parse_args().root)
