"""Long-duration contact ratio test with full collection parameters.

Mirrors collect_trajectories.py exactly: motion_start=350, motion_length=1800,
per-family angular speed from YAML, random axis, SO(3) uniform initial pose.
Reports the fraction of frames where all four fingertips stay in loaded
contact (|F| >= 0.05 N) during the motion window.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from object_catalog import get_motion_config, list_object_ids, load_object_config
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    MCCLeapHandPositionControlCfg,
    mcc_finger_contact_env_cfg,
)

TIP_SITES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
THRESHOLD = 0.05  # N


def _force_mags(env: ManagerBasedRlEnv) -> np.ndarray:
    mags = np.zeros(4, dtype=np.float64)
    for i, site_name in enumerate(TIP_SITES):
        sensor = env.scene[f"{site_name}_contact"]
        if not isinstance(sensor, ContactSensor):
            continue
        force = sensor.data.force
        if force is None:
            continue
        if sensor.data.found is not None:
            force = torch.where(
                sensor.data.found.unsqueeze(-1) > 0,
                force, torch.zeros_like(force),
            )
        mags[i] = float(torch.linalg.vector_norm(force.sum(dim=1)).cpu())
    return mags


def _wxyz_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def main() -> None:
    device = "cuda:0"
    motion_start = 350
    motion_length = 1800
    total_steps = motion_start + motion_length

    print(
        f"Motion window: step {motion_start}..{motion_start+motion_length} "
        f"({motion_length*0.01:.0f}s)  threshold={THRESHOLD}N"
    )
    print(f"{'Object':<22} {'Family':<14} {'speed':>9} {'all4%':>7} "
          f"{'IF%':>6} {'MF%':>6} {'RF%':>6} {'TH%':>6} {'minF/N':>7} {'maxF/N':>7}")
    print("-" * 95)

    for oid in list_object_ids():
        config = load_object_config(oid)
        motion = get_motion_config(config)
        speed_range = motion["rotation"]["angular_speed_range_rad_s"]
        speed = float(np.mean(speed_range))  # midpoint of family range

        env_cfg = mcc_finger_contact_env_cfg(
            num_envs=1, play=True, object_config=config,
        )
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        rl_cfg = MCCLeapHandPositionControlCfg()
        kw = asdict(rl_cfg)
        pc = kw.pop("policy_class")
        kw.pop("device", None)
        policy = pc(device=device, num_envs=1, **kw)
        policy.reset()

        mocap_id = int(env.scene["target"].data.indexing.mocap_id)
        dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)

        torch.manual_seed(42)
        np.random.seed(42)
        obs, _ = env.reset()

        # SO(3) uniform initial orientation
        init_q = torch.randn(4, device=device)
        init_q = init_q / torch.linalg.vector_norm(init_q).clamp_min(1e-8)
        env.sim.data.mocap_quat[0, mocap_id] = init_q

        # random rotation axis
        axis = torch.randn(3, device=device)
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-8)

        loaded = np.zeros((motion_length, 4), dtype=bool)
        min_force = np.full(4, np.inf)
        max_force = np.zeros(4)

        for step in range(total_steps):
            moving = motion_start <= step < motion_start + motion_length
            if moving:
                angle = torch.tensor(speed * dt, device=device)
                delta = torch.cat((
                    torch.cos(angle / 2).unsqueeze(0),
                    axis * torch.sin(angle / 2).unsqueeze(0),
                ))
                quat = env.sim.data.mocap_quat[0, mocap_id].clone()
                quat = _wxyz_mul(quat, delta)
                quat = quat / torch.linalg.vector_norm(quat)
                env.sim.data.mocap_quat[0, mocap_id] = quat

            obs, *_ = env.step(policy(obs))

            if moving:
                fm = _force_mags(env)
                frame_idx = step - motion_start
                loaded[frame_idx] = fm >= THRESHOLD
                min_force = np.minimum(min_force, fm)
                max_force = np.maximum(max_force, fm)

        env.close()

        all4 = loaded.all(axis=1).mean()
        per_tip = loaded.mean(axis=0)

        # Find max consecutive loss run
        loss = ~loaded.all(axis=1)
        max_run = 0
        current = 0
        for is_loss in loss:
            if is_loss:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0

        flag = ""
        if all4 < 0.95:
            flag = "  <-- LOW"
        print(
            f"{oid:<22} {config.family:<14} {speed:>8.3f} {all4:>6.1%} "
            f"{per_tip[0]:>5.1%} {per_tip[1]:>5.1%} {per_tip[2]:>5.1%} {per_tip[3]:>5.1%} "
            f"{min_force.min():>6.3f} {max_force.max():>6.3f}{flag}"
        )

    print("\nNote: single-seed test; real collection should sample more seeds.")


if __name__ == "__main__":
    main()
