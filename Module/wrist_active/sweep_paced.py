"""Run the frozen paced trial with a one-millisecond Windows timer period."""
import argparse
from contextlib import contextmanager, nullcontext
import ctypes
import os
from pathlib import Path
from unittest.mock import patch

from . import sweep_experiment
from .sweep_experiment import OUT, run
from .sweep_suite import freeze


@contextmanager
def waitable_clock():
    """Use a process-owned high resolution timer for sub-frame release waits."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateWaitableTimerExW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong]
    kernel.CreateWaitableTimerExW.restype=ctypes.c_void_p
    kernel.SetWaitableTimer.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_longlong),
                                      ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateWaitableTimerExW(None,None,2,0x001F0003)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    def wait(seconds):
        due=ctypes.c_longlong(-max(1,round(seconds*10_000_000)))
        if not kernel.SetWaitableTimer(handle,ctypes.byref(due),0,None,None,0):
            raise ctypes.WinError(ctypes.get_last_error())
        if kernel.WaitForSingleObject(handle,0xFFFFFFFF)!=0:
            raise ctypes.WinError(ctypes.get_last_error())
    try:
        with patch.object(sweep_experiment,'sleep',wait):
            yield
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def timer_resolution(*, high_resolution=True):
    timer = ctypes.WinDLL('winmm') if os.name=='nt' else None
    if timer and timer.timeBeginPeriod(1)!=0:
        raise OSError('The one-millisecond Windows timer period was unavailable')
    try:
        with (waitable_clock() if timer and high_resolution else nullcontext()):
            yield
    finally:
        if timer:
            timer.timeEndPeriod(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    args=p.parse_args()
    spec=freeze(args.root)
    trial=next(r for r in spec['jobs'] if r['group']=='paced')
    # Match the original load-stress trial's ordinary Python sleep semantics.
    with timer_resolution(high_resolution=False):
        run(args.root/'paced'/trial['obj']/trial['start']/trial['arm'],trial['obj'],trial['start'],
            trial['arm'],trial['seconds'],paced=True)
