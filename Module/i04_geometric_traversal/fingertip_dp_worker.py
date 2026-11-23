"""Finger Policy inference in its deployment environment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICY = "single_all_seed20261116"
WIRE = np.dtype("<f4")
FIELD_WIDTHS = (16, 7, 12, 12, 4, 6, 6)
OBS_BYTES = sum(FIELD_WIDTHS) * WIRE.itemsize
RELEASE = 'release-100ms'
ROLLING = 'rolling-16-8'
ASYNC_100 = 'async-100ms'
SCHEDULES = (RELEASE, ROLLING, ASYNC_100)
SAMPLERS = ('release', 'cached-ddpm')
ROLLING_EXECUTION_POINTS = 8
PREFETCH_FRAMES = 10
WORKER_FIELDS = ('inference_ms', 'scheduler_install_ms', 'worker_total_ms',
                 'query_submitted', 'installed_from_frame', 'overlap_frame',
                 'switch_deadline_miss', 'buffer_underrun')
ACTION_BYTES = (16 + len(WORKER_FIELDS)) * WIRE.itemsize
CHUNK_INPUT_BYTES = (1 + 16 + 16 * 96) * WIRE.itemsize
CHUNK_OUTPUT_BYTES = (16 * 16 + 1) * WIRE.itemsize


class RollingChunks:
    """Run one inference ahead while the controller consumes the installed path."""
    def __init__(self, scheduler, infer, executor, bootstrap_end, total_frames, buffer_frames,
                 *, lead_frames=PREFETCH_FRAMES):
        self.scheduler = scheduler
        self.infer = infer
        self.executor = executor
        self.lead_frames = lead_frames
        self.next_switch = bootstrap_end + lead_frames
        self.total_frames = total_frames
        self.buffer_frames = buffer_frames
        self.pending = self.ready = None
        self.request_frame = -1

    def update(self, frame, q, observation, nominal):
        inference_ms = schedule_ms = installed = -1.
        submitted = missed = underrun = overlap = False
        if self.pending is not None and self.pending.done():
            self.ready, inference_ms = self.pending.result()
            self.pending = None
        if frame >= self.next_switch and self.pending is not None:
            missed = frame == self.next_switch
        if frame >= self.next_switch and self.ready is not None:
            started = perf_counter()
            self.scheduler.install(self.ready, execution_frames=self.buffer_frames)
            schedule_ms = (perf_counter() - started) * 1000.
            self.ready = None
            installed = self.request_frame
            self.next_switch = frame + self.scheduler.config.replan_interval
        if (frame >= self.next_switch - self.lead_frames and self.next_switch < self.total_frames
                and self.pending is None and self.ready is None):
            # History and the measured joint anchor belong to the request frame.
            self.request_frame = frame
            self.pending = self.executor.submit(self.infer, frame, q.clone(), observation.history())
            submitted = True
        if self.scheduler.path is not None:
            if self.scheduler.path_index < self.scheduler.path.shape[1]:
                nominal = self.scheduler.next_command()
                overlap = self.pending is not None
            else:
                underrun = self.next_switch < self.total_frames
        return nominal, inference_ms, schedule_ms, submitted, installed, overlap, missed, underrun


def load_runtime(args):
    import torch
    import controlled_policy_ablation as provider

    started = perf_counter()
    torch.set_num_threads(1)
    directory = ROOT / "finger_policy"
    bundle = json.loads((directory / "checkpoints/manifest.json").read_bytes())[POLICY]
    checkpoint = directory / bundle["path"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != bundle["sha256"]:
        raise ValueError("Finger DP checkpoint does not match its release manifest")
    runtime = provider.ControlledRuntime(dict(comparison_checkpoint=str(checkpoint), record_latency=True),
                                         args.device, args.seed)
    if args.sampler == 'cached-ddpm':
        # Load the provider's noise-key dependency during startup.
        import_module('evaluate_matched_regrasp')
        from .fingertip_ddpm import CachedDDPM
        noise_scheduler = next(iter(runtime.models.values())).diffusion.noise_scheduler
        provider.sample = CachedDDPM(noise_scheduler, runtime.denoise_steps, runtime.device,
                                     conditioning=provider.conditioning, predict_noise=provider.predict_noise)
    runtime.supervisor = SimpleNamespace(cursor=[0], phase=[0], age=[0], lengths=[args.frames])
    request = directory / "data/evaluation_inputs_full/requests/long_natural.json"
    scale = json.loads(request.read_bytes())["scheduler_scale"]
    info = dict(policy=POLICY, checkpoint_path=bundle["path"], checkpoint_sha256=bundle["sha256"],
                training_step=runtime.step, inference_seed=args.seed, torch=torch.__version__,
                device=(torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda"
                        else "CPU"), tensor_device=str(runtime.device),
                denoise_steps=runtime.denoise_steps, sampler=args.sampler, cuda_graph=runtime.denoise_graph is not None,
                worker_setup_s=perf_counter() - started)

    def infer(frame, q, history):
        runtime.supervisor.cursor[0] = frame
        started = perf_counter()
        chunk = runtime(frame, q.to(runtime.device), history.to(runtime.device)).cpu()
        return chunk, (perf_counter() - started) * 1000

    return runtime, infer, scale, info


def serve_chunks(args, stdin, stdout):
    import torch

    _, infer, scale, info = load_runtime(args)
    stdout.write(json.dumps(dict(**info, scale=scale)).encode('utf-8') + b'\n')
    stdout.flush()
    while packet := stdin.read(CHUNK_INPUT_BYTES):
        if len(packet) != CHUNK_INPUT_BYTES:
            raise ValueError('Incomplete action-chunk request')
        buffer = np.frombuffer(packet, dtype=WIRE).copy()
        q = torch.from_numpy(buffer[1:17]).reshape(1, 16)
        history = torch.from_numpy(buffer[17:]).reshape(1, 16, 96)
        chunk, elapsed = infer(int(buffer[0]), q, history)
        stdout.write(np.r_[chunk.numpy().ravel(), np.float32(elapsed)].astype(WIRE, copy=False).tobytes())
        stdout.flush()


def serve(args, stdin, stdout):
    import torch
    from batched_b_control_chain import BatchedBObservation, BatchedDPChunkScheduler
    from dp_chunk_scheduler import DPChunkSchedulerConfig

    runtime, infer, scale, info = load_runtime(args)
    scale = torch.tensor(scale)
    config = DPChunkSchedulerConfig()
    asynchronous = args.schedule != RELEASE
    stride = round(config.waypoint_dt / config.control_dt)
    if args.schedule == ROLLING:
        config = replace(config, replan_interval=ROLLING_EXECUTION_POINTS * stride)
    scheduler = BatchedDPChunkScheduler(1, scale, config)
    observation = BatchedBObservation(1, "cpu")
    bootstrap_end = (observation.horizon - 1) * observation.stride
    prediction_points = runtime.norm['scale'].numel() // scale.numel()
    buffer_frames = (prediction_points - config.max_drop - 1) * stride if asynchronous else config.replan_interval
    lead_frames = PREFETCH_FRAMES if args.schedule == ROLLING else (3 if args.sampler == 'cached-ddpm' else 7)
    info.update(history_samples=observation.horizon, history_spacing_s=.05,
                bootstrap_frames=bootstrap_end + 1, schedule=args.schedule,
                prediction_points=prediction_points, prediction_point_dt_s=config.waypoint_dt,
                planned_execution_points=config.replan_interval // stride,
                query_spacing_s=config.replan_interval * config.control_dt,
                prefetch_s=lead_frames * config.control_dt if asynchronous else 0.,
                command_buffer_s=buffer_frames * config.control_dt, scheduler_control_dt=config.control_dt)
    stdout.write(json.dumps(info).encode("utf-8") + b"\n")
    stdout.flush()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='i04c-finger-dp') as executor:
        pipeline = (RollingChunks(scheduler, infer, executor, bootstrap_end, args.frames, buffer_frames,
                                  lead_frames=lead_frames) if asynchronous else None)
        control_frames(args.frames, stdin, stdout, torch, observation, scheduler, bootstrap_end, infer, pipeline)


def control_frames(frames, stdin, stdout, torch, observation, scheduler, bootstrap_end, infer, pipeline):
    nominal = None
    for frame in range(frames):
        packet = stdin.read(OBS_BYTES)
        if not packet:
            return
        if len(packet) != OBS_BYTES:
            raise ValueError("Incomplete policy observation")
        tick = perf_counter()
        q, pose, points, normals, found, twist, delta = torch.from_numpy(
            np.frombuffer(packet, dtype=WIRE).copy()).split(FIELD_WIDTHS)
        q = q.reshape(1, 16)
        if nominal is None:
            initial_q = nominal = q
        observation.push(q=q, palm_pose_world=pose.reshape(1, 7),
                         contact_points_world=points.reshape(1, 4, 3),
                         contact_normals_inward_world=normals.reshape(1, 4, 3),
                         found=found.reshape(1, 4).bool(), palm_twist_world=twist.reshape(1, 6),
                         planner_command_palm=delta.reshape(1, 6))
        scheduler.observe(nominal)
        if pipeline is not None:
            nominal, infer_ms, schedule_ms, submitted, installed, overlap, missed, underrun = pipeline.update(
                frame, q, observation, nominal)
        else:
            infer_ms = schedule_ms = installed = -1.
            submitted = overlap = missed = underrun = False
            if observation.history_ready and (frame - bootstrap_end) % scheduler.config.replan_interval == 0:
                chunk, infer_ms = infer(frame, q, observation.history())
                started = perf_counter()
                scheduler.install(chunk)
                schedule_ms = (perf_counter() - started) * 1000
                submitted, installed = True, frame
            nominal = initial_q if frame <= bootstrap_end else scheduler.next_command()
        elapsed_ms = (perf_counter() - tick) * 1000
        reply = np.concatenate((nominal.numpy().ravel(), [infer_ms, schedule_ms, elapsed_ms, submitted,
                               installed, overlap, missed, underrun]))
        stdout.write(reply.astype(WIRE).tobytes())
        stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument('--schedule', choices=SCHEDULES, default=RELEASE)
    parser.add_argument('--sampler', choices=SAMPLERS, default='release')
    parser.add_argument('--chunks-only', action='store_true')
    args = parser.parse_args()
    protocol_out = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        worker = serve_chunks if args.chunks_only else serve
        worker(args, sys.stdin.buffer, protocol_out)


if __name__ == "__main__":
    main()
