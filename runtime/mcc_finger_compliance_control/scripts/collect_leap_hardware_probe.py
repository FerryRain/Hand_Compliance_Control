"""Evaluate a hardware-limited teacher chain with optional 1ms object motion.

This process-local adapter keeps the existing collector and healthy initializer.
Actuator settings apply from model creation. MCC targets, trajectory duration,
contact geometry, execution tube, and overforce record mode remain explicit.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from leap_hardware_profile import PROFILES, profile_metadata


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--leap-profile", choices=tuple(PROFILES), default="continuous")
    parser.add_argument("--object-substep-interpolation", action="store_true")
    parser.add_argument("--leap-kp", type=float, default=None)
    parser.add_argument("--leap-kd", type=float, default=None)
    own, rest = parser.parse_known_args()
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--output", type=Path, required=True)
    paths, _ = known.parse_known_args(rest)
    paths.output.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[own.leap_profile]
    if own.leap_kp is not None:
        if own.leap_kp <= 0:
            parser.error("--leap-kp must be positive")
        profile = replace(profile, kp_nm_per_rad=own.leap_kp)
    if own.leap_kd is not None:
        if own.leap_kd < 0:
            parser.error("--leap-kd must be nonnegative")
        profile = replace(profile, kd_nm_s_per_rad=own.leap_kd)
    forbidden = {"--finger-stiffness", "--finger-damping", "--finger-effort-limit"}
    if forbidden.intersection(rest):
        parser.error("Actuator values are owned by --leap-profile")
    sys.argv = [sys.argv[0], *rest, "--finger-stiffness", str(profile.kp_nm_per_rad),
                "--finger-damping", str(profile.kd_nm_s_per_rad),
                "--finger-effort-limit", str(profile.effort_limit_nm)]
    from mjlab.envs import ManagerBasedRlEnv
    import torch
    import warp as wp
    import collect_trajectories as collector
    import collection_initialization as initialization
    from collect_substep_object_probe import substep_poses

    original_step = ManagerBasedRlEnv.step
    original_teacher = collector.FullHandMCCCollectionPolicy.__call__
    original_initialization = initialization.StableContactInitializer.update
    previous = None
    physical_config = None
    initialization_records = []

    def stable_initialization(initializer, *values, **kwargs):
        command, nominal, ready = original_initialization(initializer, *values, **kwargs)
        q = np.asarray(kwargs.get("q_live", values[0] if values else None), float)
        history = getattr(initializer, "_leap_stationary_history", None)
        if history is None:
            history = deque(maxlen=50)
            initializer._leap_stationary_history = history
        history.append(q.copy())
        span = float(np.ptp(np.asarray(history), axis=0).max())
        ready = bool(ready and len(history) == 50 and span <= .005)
        initializer.ready = ready
        initializer.debug.update(ready=ready, stationary_q_span_max_rad=span,
                                 stationary_window_frames=len(history), stationary_limit_rad=.005)
        return command, nominal, ready

    def teacher(policy, *values, **kwargs):
        nonlocal physical_config
        motion = policy.motion_preview_controller
        policy.env._leap_motion_active = motion is not None and bool(motion.motion_active.any())
        if physical_config is None:
            m = policy.env.sim.mj_model
            ids = [i for i in range(m.nu) if m.actuator(i).name.split("/")[-1].isdigit()]
            if len(ids) != 16:
                raise RuntimeError("Expected sixteen LEAP joint actuators")
            wp.synchronize_stream(wp.get_stream(policy.env.sim.wp_device))
            actual_gain = policy.env.sim.wp_model.actuator_gainprm.numpy()[:, ids, 0]
            actual_bias = policy.env.sim.wp_model.actuator_biasprm.numpy()[:, ids]
            actual_limits = policy.env.sim.wp_model.actuator_forcerange.numpy()[:, ids]
            if not (np.allclose(actual_gain, profile.kp_nm_per_rad)
                    and np.allclose(actual_bias[..., 1], -profile.kp_nm_per_rad)
                    and np.allclose(actual_bias[..., 2], -profile.kd_nm_s_per_rad)
                    and np.allclose(actual_limits[..., 0], -profile.effort_limit_nm)
                    and np.allclose(actual_limits[..., 1], profile.effort_limit_nm)):
                raise RuntimeError("Actual Warp gains/force limits disagree with the requested profile")
            physical_config = dict(actuator_ids=ids, kp=actual_gain.tolist(),
                                   kd=(-actual_bias[..., 2]).tolist(), limits=actual_limits.tolist(),
                                   gear=m.actuator_gear[ids].tolist(), verified_in_warp=True)
            (paths.output / "actual_actuators.json").write_text(json.dumps(physical_config, indent=2)+"\n")
            import mujoco
            mujoco.mj_saveModel(m, str(paths.output / "scene.mjb"), None)
        result = original_teacher(policy, *values, **kwargs)
        count = getattr(policy.env, "_long_probe_steps", 0)
        if count >= 800 and not policy.env._leap_motion_active:
            q = policy.env.scene["robot"].data.joint_pos[:, -16:].detach().cpu().numpy().copy()
            initialization_records.append(dict(step=count, q=q,
                command=q+.08*result[:, -16:].detach().cpu().numpy(),
                actuator_force=policy.env.sim.data.actuator_force[:, -16:].detach().cpu().numpy().copy(),
                qvel=policy.env.sim.data.qvel[:, -16:].detach().cpu().numpy().copy(),
                nominal=np.stack([x.nominal_q if x is not None else q[i]
                                  for i,x in enumerate(policy.initializers)])))
        return result

    def step(env, *values, **kwargs):
        nonlocal previous
        mid = int(env.scene["target"].data.indexing.mocap_id)
        position = env.sim.data.mocap_pos[:, mid].detach().cpu().numpy().copy()
        quaternion = env.sim.data.mocap_quat[:, mid].detach().cpu().numpy().copy()
        active = bool(getattr(env, "_leap_motion_active", False))
        if not own.object_substep_interpolation or not active or previous is None:
            previous = (position, quaternion)
            return original_step(env, *values, **kwargs)
        poses = np.stack([substep_poses(previous[0][i], previous[1][i], position[i],
                                         quaternion[i], env.cfg.decimation)
                          for i in range(env.num_envs)], axis=1)
        poses = torch.as_tensor(poses, device=env.device, dtype=torch.float32)
        physical = env.sim.step
        index = 0

        def substep():
            nonlocal index
            stream = torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
            with torch.cuda.stream(stream):
                env.sim.data.mocap_pos[:, mid] = poses[index, :, :3]
                env.sim.data.mocap_quat[:, mid] = poses[index, :, 3:]
            index += 1
            physical()

        env.sim.step = substep
        try:
            return original_step(env, *values, **kwargs)
        finally:
            env.sim.step = physical
            previous = (position, quaternion)

    protocol = dict(**profile_metadata(own.leap_profile),
                    object_substep_interpolation=own.object_substep_interpolation,
                    interpolation="linear translation and shortest-arc SLERP; same 100Hz endpoints; motion only",
                    activation="Actuator parameters apply from model creation, including initialization",
                    initialization_stationarity_gate=dict(frames=50, q_span_max_rad=.005,
                        meaning="Existing posthoc stability requirement is enforced before motion"),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol.update(kp_nm_per_rad=profile.kp_nm_per_rad, kd_nm_s_per_rad=profile.kd_nm_s_per_rad)
    (paths.output / "leap_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    ManagerBasedRlEnv.step = step
    collector.FullHandMCCCollectionPolicy.__call__ = teacher
    initialization.StableContactInitializer.update = stable_initialization
    try:
        from collect_regional_healthy_reference import main as collect
        collect()
    finally:
        ManagerBasedRlEnv.step = original_step
        collector.FullHandMCCCollectionPolicy.__call__ = original_teacher
        initialization.StableContactInitializer.update = original_initialization
        if initialization_records:
            np.savez_compressed(paths.output / "initialization_execution.npz",
                                **{key: np.asarray([r[key] for r in initialization_records])
                                   for key in initialization_records[0]})
        report = paths.output / "summary.json"
        if report.exists():
            value = json.loads(report.read_text())
            value.update(leap_profile=protocol, actual_finger_actuators=physical_config)
            report.write_text(json.dumps(value, indent=2)+"\n")


if __name__ == "__main__":
    main()
