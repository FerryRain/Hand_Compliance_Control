"""Keep per-tick history and execution native-side; send only DP queries."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from mcc_finger_compliance_control.scripts.dp_chunk_scheduler import DPChunkScheduler
from .fingertip_dp_worker import CHUNK_OUTPUT_BYTES, FIELD_WIDTHS, ROOT, WIRE


class ObservationRing:
    stride = 5
    horizon = 16
    control_dt = .01

    def __init__(self):
        self.count = 0
        self.states = torch.zeros((76, 96), dtype=torch.float32)
        self.history_offsets = torch.arange(self.horizon - 1, -1, -1) * self.stride
        self.motion_q = torch.zeros((6, 16))
        self.motion_point = torch.zeros((6, 4, 3))
        self.motion_normal = torch.zeros_like(self.motion_point)
        self.motion_mask = torch.zeros((6, 4), dtype=torch.bool)
        self.last_point = torch.zeros((1, 4, 3))
        self.last_normal = torch.zeros_like(self.last_point)
        self.point_valid = torch.zeros((1, 4), dtype=torch.bool)
        self.normal_valid = torch.zeros_like(self.point_valid)

    @staticmethod
    def palm_rotation(pose):
        quaternion = pose[:, 3:7]
        w, x, y, z = (quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)).unbind(-1)
        return torch.stack((
            1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
            2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
            2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
        ), dim=-1).reshape(-1, 3, 3)

    def push(self, packet):
        q, pose, point, normal, found, twist, delta = torch.from_numpy(packet.astype(WIRE)).split(FIELD_WIDTHS)
        q, pose = q[None], pose[None]
        point, normal, found = point.reshape(1, 4, 3), normal.reshape(1, 4, 3), found[None].bool()
        point_now = found & (torch.linalg.vector_norm(point, dim=-1) > 1e-9)
        normal_now = found & (torch.linalg.vector_norm(normal, dim=-1) > 1e-6)
        self.last_point = torch.where(point_now[..., None], point, self.last_point)
        self.last_normal = torch.where(normal_now[..., None], normal, self.last_normal)
        self.point_valid |= point_now
        self.normal_valid |= normal_now
        held_point = torch.where((~point_now & self.point_valid)[..., None], self.last_point, point)
        held_normal = torch.where((~normal_now & self.normal_valid)[..., None], self.last_normal, normal)
        inv = self.palm_rotation(pose).transpose(-1, -2)
        point_palm = torch.einsum('nij,nfj->nfi', inv, held_point - pose[:, None, :3])
        normal_palm = torch.einsum('nij,nfj->nfi', inv, held_normal)
        linear = torch.einsum('nij,nj->ni', inv, twist[None, :3])
        angular = torch.einsum('nij,nj->ni', inv, twist[None, 3:])
        if self.count >= self.stride:
            old = (self.count - self.stride) % len(self.motion_q)
            duration = self.stride * self.control_dt
            valid = (found & self.motion_mask[old])[..., None]
            q_rate = (q - self.motion_q[old]) / duration
            point_rate = torch.where(valid, (point_palm - self.motion_point[old]) / duration, 0)
            normal_rate = torch.where(valid, torch.linalg.cross(self.motion_normal[old][None], normal_palm) / duration, 0)
        else:
            q_rate, point_rate, normal_rate = torch.zeros_like(q), torch.zeros_like(point_palm), torch.zeros_like(normal_palm)
        slot = self.count % len(self.motion_q)
        self.motion_q[slot], self.motion_point[slot], self.motion_normal[slot], self.motion_mask[slot] = (
            q[0], point_palm[0], normal_palm[0], found[0])
        state = torch.cat((q, point_palm.flatten(1), normal_palm.flatten(1), found.float(), q_rate,
                           point_rate.flatten(1), normal_rate.flatten(1), linear, angular, delta[None]), dim=-1)
        if self.count == 0:
            self.states[:] = state
        self.states[self.count % len(self.states)] = state[0]
        self.count += 1
        return q.numpy().ravel()

    def history(self):
        indices = (self.count - 1 - self.history_offsets) % len(self.states)
        return self.states[indices].numpy()


class LocalFingerPolicy:
    def __init__(self, seed, frames):
        started = perf_counter()
        torch.set_num_threads(1)
        env = os.environ.copy()
        # The released CUDA provider imports its newer mjWarp/MuJoCo stack.
        env['PYTHONPATH'] = os.pathsep.join(str(ROOT / name) for name in (
            'mcc_finger_compliance_control/scripts', 'finger_policy/scripts', 'src', '.'))
        self.worker = subprocess.Popen([sys.executable, '-u', '-m',
            'Module.i04_geometric_traversal.fingertip_dp_worker', '--device', 'cuda:0',
            '--seed', str(seed), '--frames', str(frames), '--sampler', 'cached-ddpm', '--chunks-only'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=ROOT, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            self.info = json.loads(self.worker.stdout.readline())
        except (ValueError, OSError):
            self.__exit__(*sys.exc_info())
            raise
        scale = np.asarray(self.info.pop('scale'), dtype=np.float32)
        self.scheduler = DPChunkScheduler(scale)
        self.observation = ObservationRing()
        self.bootstrap = (self.observation.horizon - 1) * self.observation.stride
        self.nominal = None
        self.rpc_ms, self.inference_ms, self.install_ms, self.local_ms = [], [], [], []
        self.info.update(schedule='sync-100ms', control_s=.01, action_points=16, waypoint_s=.05,
                         observation_history='native_control_ring_76x96', ipc='action_chunk_queries_only',
                         process_startup_s=perf_counter() - started)

    def __enter__(self):
        return self

    def __exit__(self, error_type, *_):
        if error_type is not None and self.worker.poll() is None:
            self.worker.terminate()
        self.worker.stdin.close()
        self.worker.stdout.close()
        code = self.worker.wait(timeout=40)
        if code and error_type is None:
            raise RuntimeError(f'Finger DP chunk provider exited with status {code}')

    def command(self, frame, packet):
        started = perf_counter()
        measured = self.observation.push(packet)
        if self.nominal is None:
            self.initial = self.nominal = measured
        self.scheduler.observe(self.nominal)
        self.local_ms.append((perf_counter() - started) * 1000.)
        if frame >= self.bootstrap and (frame - self.bootstrap) % self.scheduler.config.replan_interval == 0:
            started = perf_counter()
            request = np.r_[np.float32(frame), measured, self.observation.history().ravel()].astype(WIRE, copy=False)
            self.worker.stdin.write(request.tobytes())
            self.worker.stdin.flush()
            reply = self.worker.stdout.read(CHUNK_OUTPUT_BYTES)
            if len(reply) != CHUNK_OUTPUT_BYTES:
                raise RuntimeError('Missing Finger DP prediction')
            chunk = np.frombuffer(reply, dtype=WIRE)
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite Finger DP prediction')
            self.rpc_ms.append((perf_counter() - started) * 1000.)
            self.inference_ms.append(float(chunk[-1]))
            started = perf_counter()
            self.scheduler.install(chunk[:-1].reshape(16, 16))
            self.install_ms.append((perf_counter() - started) * 1000.)
        self.nominal = self.initial if frame <= self.bootstrap else self.scheduler.next_command()
        return self.nominal

    def report(self, output):
        from .fingertip_traversal import latency_ms

        signals = dict(inference_ms=self.inference_ms, chunk_rpc_ms=self.rpc_ms,
                       chunk_install_ms=self.install_ms, local_observation_ms=self.local_ms)
        np.savez_compressed(output / 'timing.npz', **signals)
        return {key: latency_ms(data) for key, data in signals.items()}
