"""Compare Finger DP against the audited I04-c run at a fixed wrist command."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter, sleep

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import ExperimentConfig
from .fingertip_dp_worker import ACTION_BYTES, RELEASE, ROOT, SAMPLERS, SCHEDULES, WIRE, WORKER_FIELDS
from .fingertip_hand import DT
from .fingertip_replay import chunks, load_case
from .fingertip_traversal import ForceFeedback, Measurements, STATE, latency_ms, surface_graph, write_json
from .oracle import MeasuredCoverage

CONTROL_STEPS = 5
LOOKAHEAD_STEPS = 100
TRACK_PHASES = ("slide", "hold", "approach", "recontact")
TRACE_FIELDS = ("qpos", "palm_command", "finger_command", "normal_force_n",
                "min_head_fraction", "region_load_n", "incidental_contact", "phase")
TIMING_FIELDS = ("observation_ms", "policy_rpc_ms", *WORKER_FIELDS)


@contextmanager
def control_pacer(enabled):
    """Give the background inference real control deadlines at 100 Hz."""
    if not enabled:
        yield lambda _: None
        return
    if os.name != 'nt':
        yield lambda deadline: sleep(max(0., deadline - perf_counter()))
        return
    import ctypes as ct
    from ctypes import wintypes as wt

    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.CreateWaitableTimerExW.argtypes = (wt.LPVOID, wt.LPCWSTR, wt.DWORD, wt.DWORD)
    kernel.CreateWaitableTimerExW.restype = wt.HANDLE
    kernel.SetWaitableTimer.argtypes = (wt.HANDLE, ct.POINTER(ct.c_longlong), wt.LONG,
                                        wt.LPVOID, wt.LPVOID, wt.BOOL)
    kernel.SetWaitableTimer.restype = wt.BOOL
    kernel.WaitForSingleObject.argtypes = (wt.HANDLE, wt.DWORD)
    kernel.WaitForSingleObject.restype = wt.DWORD
    kernel.CloseHandle.argtypes = (wt.HANDLE,)
    high_resolution, timer_modify_and_synchronize = 2, 0x00100002
    timer = kernel.CreateWaitableTimerExW(None, None, high_resolution, timer_modify_and_synchronize)
    if not timer:
        raise ct.WinError(ct.get_last_error())

    def wait_until(deadline):
        remaining = deadline - perf_counter()
        if remaining > 0.:
            ticks = ct.c_longlong(-max(1, int(remaining * 10_000_000)))
            if not kernel.SetWaitableTimer(timer, ct.byref(ticks), 0, None, None, False):
                raise ct.WinError(ct.get_last_error())
            if kernel.WaitForSingleObject(timer, 0xFFFFFFFF) != 0:
                raise ct.WinError(ct.get_last_error())

    try:
        yield wait_until
    finally:
        kernel.CloseHandle(timer)


class PolicyProcess:
    """Keep release inference and native MuJoCo 3.6 in their own environments."""
    def __init__(self, python, output, device, seed, frames, schedule, sampler):
        self.command = [str(python), "-u", "-m",
                        "Module.i04_geometric_traversal.fingertip_dp_worker",
                        "--device", device, "--seed", str(seed), "--frames", str(frames),
                        "--schedule", schedule, "--sampler", sampler]
        self.worker_log = output / "finger_dp_worker.log"

    def __enter__(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(ROOT / path) for path in (
            "mcc_finger_compliance_control/scripts", "finger_policy/scripts", "src", "."))
        started = perf_counter()
        with self.worker_log.open("wb") as error_log:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=error_log, cwd=ROOT, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            header = self.process.stdout.readline()
            self.info = json.loads(header)
        except (ValueError, OSError) as error:
            self.__exit__(type(error), error, None)
            raise RuntimeError(f"Finger DP startup failed. Check {self.worker_log}") from error
        self.info["process_startup_s"] = perf_counter() - started
        return self

    def command_at(self, observation):
        started = perf_counter()
        self.process.stdin.write(observation.astype(WIRE).tobytes())
        self.process.stdin.flush()
        reply = self.process.stdout.read(ACTION_BYTES)
        if len(reply) != ACTION_BYTES:
            raise RuntimeError(f"Finger DP stopped before replying. Check {self.worker_log}")
        data = np.frombuffer(reply, dtype=WIRE).astype(np.float64)
        if not np.isfinite(data).all():
            raise ValueError("Finger DP returned a non-finite command")
        return data[:16], (perf_counter() - started) * 1000, data[16:]

    def __exit__(self, error_type, error, trace):
        if error_type is not None and self.process.poll() is None:
            self.process.terminate()
        self.process.stdin.close()
        self.process.stdout.close()
        code = self.process.wait(timeout=30)
        if error_type is None and code != 0:
            raise RuntimeError(f"Finger DP worker quit with exit {code}. Check {self.worker_log}")


def policy_observation(hand, data, future_palm):
    """Map live pad collision pairs into the Finger Policy sensor contract."""
    point, inward = np.zeros((4, 3)), np.zeros((4, 3))
    loads, wrench = np.zeros(4), np.zeros(6)
    for index, contact in enumerate(data.contact):
        a, b = int(contact.geom1), int(contact.geom2)
        if a in hand.objects:
            finger, sign = hand.pad_index.get(b), -1.
        elif b in hand.objects:
            finger, sign = hand.pad_index.get(a), 1.
        else:
            continue
        if finger is None or contact.dist > 0:
            continue
        mujoco.mj_contactForce(hand.model, data, index, wrench)
        load = max(wrench[0], 0.)
        loads[finger] += load
        point[finger] += load * contact.pos
        inward[finger] += sign * load * contact.frame[:3]
    normal_size = np.linalg.norm(inward, axis=1)
    found = (loads >= .05) & (normal_size > 1e-9)
    point[found] /= loads[found, None]
    inward[found] /= normal_size[found, None]
    point[~found] = 0.
    inward[~found] = 0.
    pose = np.r_[data.xpos[hand.palm], data.xquat[hand.palm]]
    local = Rotation.from_quat(pose[[4, 5, 6, 3]]).inv()
    target = Rotation.from_quat(future_palm[[4, 5, 6, 3]])
    delta = np.r_[local.apply(future_palm[:3] - pose[:3]), (local * target).as_rotvec()]
    twist = np.empty(6)
    mujoco.mj_objectVelocity(hand.model, data, mujoco.mjtObj.mjOBJ_BODY, hand.palm, twist, 0)
    return np.concatenate((data.qpos[hand.q], pose, point.ravel(), inward.ravel(), found,
                           twist[[3, 4, 5, 0, 1, 2]], delta))


def run_comparison(case, output, policy_python, device, seed, seconds, schedule=RELEASE, sampler='release'):
    if seconds is not None and (not np.isfinite(seconds) or seconds < DT):
        raise ValueError("Duration must be at least one physics step")
    audit = json.loads((case / "native_audit.json").read_bytes())
    if not audit["native_replay_matched"]:
        raise ValueError("Use an audited I04-c reference case")
    reference = list(chunks(case))
    palm_commands = np.concatenate([item["palm_command"] for item in reference])
    phase = np.concatenate([item["phase"] for item in reference])
    total_steps = len(palm_commands)
    if total_steps != audit["states"]:
        raise ValueError("Wrist trajectory differs in length from its audit")
    steps = total_steps if seconds is None else min(total_steps, int(round(seconds / DT)))
    palm_commands = palm_commands[:steps]
    prefix = sha256(np.ascontiguousarray(palm_commands, dtype="<f8").tobytes()).hexdigest()
    comparison_frames = (steps + CONTROL_STEPS - 1) // CONTROL_STEPS
    rolling = schedule != RELEASE
    hand = load_case(case)
    model = hand.model
    data = mujoco.MjData(model)
    with np.load(case / "initial.npz", allow_pickle=False) as initial:
        mujoco.mj_setState(model, data, initial["integration_state"], STATE)
    mujoco.mj_forward(model, data)
    touch = hand.measure(data)
    graph = surface_graph()
    stats = Measurements(graph)
    forces_only = MeasuredCoverage(graph, ExperimentConfig())
    feedback = ForceFeedback()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("initial.npz", "model.mjb"):
        shutil.copyfile(case / name, output / name)
    manifest = json.loads((case / "manifest.json").read_bytes())
    records, trace_index = [], []
    policy_timing = np.empty((comparison_frames, len(TIMING_FIELDS)))
    control_start_s = np.empty(comparison_frames)
    mcc_timing = np.zeros(comparison_frames)
    first_pose_failure = first_force_stop = None
    force_excluded = longest_high = 0
    wrist_error = np.empty((steps, 2))
    started = perf_counter()

    def save_batch(batch):
        path = output / f"trace_{len(trace_index):04d}.npz"
        np.savez_compressed(path, **dict(zip(TRACE_FIELDS, map(np.asarray, zip(*batch)))))
        trace_index.append(dict(file=path.name, states=len(batch),
                                sha256=sha256(path.read_bytes()).hexdigest()))
        write_json(output / "trace_chunks.json", trace_index)

    writes = []
    with (ThreadPoolExecutor(max_workers=1, thread_name_prefix='i04c-trace') as writer,
          PolicyProcess(policy_python, output, device, seed, comparison_frames, schedule, sampler) as policy,
          control_pacer(rolling) as wait_until):
        def flush():
            nonlocal records
            if records:
                if rolling:
                    writes.append(writer.submit(save_batch, records))
                else:
                    save_batch(records)
                records = []

        manifest.update(experiment="I04-c Finger DP on fixed wrist", baseline_case=case.name,
                        wrist_sha256=prefix, wrist_steps=steps, policy=policy.info,
                        controller=dict(implementation="FingertipMCC", desired_normal_force_n=1.,
                                        parameters=asdict(feedback.fingers[0].config)),
                        wrist_feature_horizon_s=LOOKAHEAD_STEPS * DT,
                        posture_and_force_trip_behavior="log_and_continue_offline")
        write_json(output / "manifest.json", manifest)
        active_start = perf_counter()
        for i, palm in enumerate(palm_commands):
            frame = i // CONTROL_STEPS
            if i % CONTROL_STEPS == 0:
                wait_until(active_start + i * DT)
                sample_start = perf_counter()
                control_start_s[frame] = sample_start - active_start
                future = palm_commands[min(i + LOOKAHEAD_STEPS - 1, steps - 1)]
                packet = policy_observation(hand, data, future)
                observation_ms = (perf_counter() - sample_start) * 1000
                nominal, rpc_ms, worker_times = policy.command_at(packet)
                policy_timing[frame] = np.r_[observation_ms, rpc_ms, worker_times]
            before = perf_counter()
            # For a lost pad, use the current commanded palm's approach
            # direction. DP receives no geometry-derived finger targets.
            if i % CONTROL_STEPS == 0:
                outer = Rotation.from_quat(palm[[4, 5, 6, 3]]).apply(-hand.inward_in_palm)
                normals = np.broadcast_to(outer, (4, 3))
            command = feedback.command(hand, data, nominal, normals, touch,
                                       str(phase[i]) in TRACK_PHASES)
            mcc_timing[frame] += (perf_counter() - before) * 1000
            data.mocap_pos[hand.driver], data.mocap_quat[hand.driver] = palm[:3], palm[3:]
            data.ctrl[:] = command
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or abs(data.time - (i + 1) * DT) > 1e-6:
                raise RuntimeError(f"Simulation became invalid at step {i + 1}")
            touch = hand.measure(data)
            stats.update(touch)
            high = touch.loads.max() >= 8.
            force_excluded += int(high)
            longest_high = max(longest_high, stats.high_steps)
            if not high:
                forces_only.update(touch.points, touch.normals, touch.measured_heads)
            if first_pose_failure is None and not touch.legal_posture:
                first_pose_failure = dict(step=i + 1, time_s=(i + 1) * DT,
                    strict_area=stats.coverage.area_fraction, posture=touch.posture_metrics.tolist())
            if first_force_stop is None and stats.high_steps >= 100:
                first_force_stop = dict(step=i + 1, time_s=(i + 1) * DT,
                    strict_area=stats.coverage.area_fraction, region_force_n=touch.loads.tolist())
            position_error = np.linalg.norm(data.qpos[:3] - palm[:3]) * 1000
            angle_error = np.degrees(2 * np.arccos(np.clip(abs(data.qpos[3:7] @ palm[3:]), 0., 1.)))
            wrist_error[i] = position_error, angle_error
            records.append((data.qpos.copy(), palm, command, touch.forces.astype("f4"),
                            touch.min_fraction.astype("f4"), touch.loads.astype("f4"),
                            bool(touch.incidental), phase[i]))
            if (i + 1) % 5000 == 0:
                flush()
                progress = stats.summary()
                progress["force_only_area_fraction"] = min(1., forces_only.area_fraction)
                write_json(output / "progress.json", progress)
                print(f"Finger DP {i + 1}/{steps} steps, head area below the 8 N cutoff "
                      f"{forces_only.area_fraction:.2%}", flush=True)
        native_wall_s = perf_counter() - active_start
        flush()
    for write in writes:
        write.result()
    cov = stats.coverage
    np.savez_compressed(output / "coverage.npz", visited_vertices=cov.visited_vertices,
                        visited_goals=cov.visited_goals, required_vertices=graph.required_vertices,
                        force_only_vertices=forces_only.visited_vertices)
    channels = dict(zip(TIMING_FIELDS, policy_timing.T))
    np.savez_compressed(output / "timing.npz", **channels, shared_mcc_ms=mcc_timing,
                        control_start_s=control_start_s)
    control_dt = CONTROL_STEPS * DT
    switches = np.flatnonzero(channels['installed_from_frame'] >= 0)
    scheduling = dict(mode=schedule, wall_clock_paced=rolling, queries_submitted=int(np.sum(channels['query_submitted'])),
        switches=len(switches), first_switch_s=float(switches[0] * control_dt) if len(switches) else None,
        inter_switch_simulated_ms=latency_ms(np.diff(switches) * control_dt * 1000),
        overlap_control_frames=int(np.sum(channels['overlap_frame'])),
        missed_switch_deadlines=int(np.sum(channels['switch_deadline_miss'])),
        command_buffer_underrun_frames=int(np.sum(channels['buffer_underrun'])))
    if rolling:
        scheduling['control_tick_lateness_ms'] = latency_ms(
            np.maximum(0., control_start_s - np.arange(comparison_frames) * control_dt) * 1000.)
    summary = dict(experiment="I04-c Finger DP on fixed wrist", stop_reason="FROZEN_WRIST_HORIZON",
        wrist_sha256=prefix, baseline_case=case.name, full_wrist=steps == total_steps,
        first_illegal_grasp=first_pose_failure, first_sustained_overload_stop=first_force_stop,
        final_force_only_coverage=min(1., forces_only.area_fraction),
        force_only_covered_vertices=int(forces_only.visited_vertices.sum()),
        force_excluded_states=force_excluded, longest_over_40n_s=longest_high * DT,
        p95_wrist_tracking_mm_deg=np.percentile(wrist_error, 95, axis=0).tolist(),
        native_wall_s=native_wall_s, total_wall_s=perf_counter() - started, scheduling=scheduling,
        timing=dict(shared_mcc_per_control_interval=latency_ms(mcc_timing),
                    all_control_per_interval=latency_ms(mcc_timing + policy_timing[:, :2].sum(axis=1)),
                    policy_inference_per_query=latency_ms(policy_timing[policy_timing[:, 2] >= 0, 2]),
                    scheduler_install_per_query=latency_ms(policy_timing[policy_timing[:, 3] >= 0, 3]),
                    policy_rpc_per_control_interval=latency_ms(policy_timing[:, 1])),
        **stats.summary())
    write_json(output / "summary.json", summary)
    return summary


def add_arguments(parser):
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path,
                        default=ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--seconds", type=float, help="Optional prefix for a quick integration check")
    parser.add_argument('--schedule', choices=SCHEDULES, default=RELEASE,
                        help='The asynchronous schedules use a paced 100 Hz control loop')
    parser.add_argument('--sampler', choices=SAMPLERS, default='release')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    result = run_comparison(args.case, args.output, args.policy_python, args.device, args.seed,
                            args.seconds, args.schedule, args.sampler)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
