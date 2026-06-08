from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import torch
import warp as wp

wp.config.quiet = True

import mujoco

mujoco.set_mju_user_warning(lambda *_args: None)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from collect_data_headless import (
    HeadlessCollectConfig,
    _adapt_action_dim,
    _build_registered_policy,
    _compute_contact_quality,
    _read_fsr_data,
    _reset_target_pose,
    _resolve_palm_body_local_idx,
    _resolve_target_mocap_idx,
)
from train_goal_conditioned import GoalConditionedGRUPolicy


@contextmanager
def _suppress_mujoco_output():
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


@dataclass
class LoadedPolicy:
    model: GoalConditionedGRUPolicy
    norm_x_mean: np.ndarray
    norm_x_std: np.ndarray
    norm_y_mean: np.ndarray
    norm_y_std: np.ndarray
    goal_vec: np.ndarray
    predict_residual: bool


def _load_policy(model_path: str, norm_path: str, device: torch.device) -> LoadedPolicy:
    ckpt = torch.load(model_path, map_location=device)
    in_dim = int(ckpt["in_dim"])
    goal_dim = int(ckpt["goal_dim"])
    action_dim = int(ckpt["action_dim"])
    quality_dim = int(ckpt["quality_dim"])
    hidden_dim = int(ckpt["config"].get("hidden_dim", 256))

    model = GoalConditionedGRUPolicy(
        in_dim=in_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        quality_dim=quality_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = np.load(norm_path)
    return LoadedPolicy(
        model=model,
        norm_x_mean=norm["x_mean"].astype(np.float32),
        norm_x_std=norm["x_std"].astype(np.float32),
        norm_y_mean=norm["y_mean"].astype(np.float32),
        norm_y_std=norm["y_std"].astype(np.float32),
        goal_vec=norm["goal_vec"].astype(np.float32),
        predict_residual=bool(ckpt["config"].get("predict_residual", False)),
    )


def _q_hand_from_env(env: ManagerBasedRlEnv) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos
    return q[:, -16:]


def _prepare_x_seq(
    fsr_hist: list[torch.Tensor],
    q_hist: list[torch.Tensor],
    prev_action_hist: list[torch.Tensor],
    norm_x_mean: np.ndarray,
    norm_x_std: np.ndarray,
) -> torch.Tensor:
    fsr_seq = torch.stack(fsr_hist, dim=1)
    q_seq = torch.stack(q_hist, dim=1)
    pa_seq = torch.stack(prev_action_hist, dim=1)
    x_seq = torch.cat([fsr_seq, q_seq, pa_seq], dim=-1)

    mean = torch.as_tensor(norm_x_mean, device=x_seq.device, dtype=x_seq.dtype)
    std = torch.as_tensor(norm_x_std, device=x_seq.device, dtype=x_seq.dtype)
    return (x_seq - mean) / std


def _policy_action(
    policy: LoadedPolicy,
    x_seq: torch.Tensor,
    prev_action: torch.Tensor,
) -> torch.Tensor:
    goal = torch.as_tensor(policy.goal_vec, device=x_seq.device, dtype=x_seq.dtype)
    goal = goal.unsqueeze(0).repeat(x_seq.shape[0], 1)

    with torch.no_grad():
        y_pred_z, _ = policy.model(x_seq, goal)

    y_mean = torch.as_tensor(policy.norm_y_mean, device=x_seq.device, dtype=x_seq.dtype)
    y_std = torch.as_tensor(policy.norm_y_std, device=x_seq.device, dtype=x_seq.dtype)
    y_pred = y_pred_z * y_std + y_mean

    if policy.predict_residual:
        y_pred = y_pred + prev_action

    return torch.clamp(y_pred, min=-1.0, max=1.0)


def _evaluate_once(
    task_id: str,
    policy_type: Literal["learned", "manual"],
    policy: LoadedPolicy | None,
    device: torch.device,
    num_envs: int,
    episodes: int,
    episode_steps: int,
    window: int,
    drop_palm_fsr: bool,
    fsr_source: Literal["policy", "sensor"],
    contact_threshold: float,
    stability_window: int,
    randomize_object_orientation: bool,
    target_anchor: Literal["origin", "palm"],
    target_offset: tuple[float, float, float],
) -> dict[str, float]:
    with _suppress_mujoco_output():
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = num_envs
        env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))

    cfg = HeadlessCollectConfig(
        num_envs=num_envs,
        device=str(device),
        contact_threshold=contact_threshold,
        stability_window=stability_window,
        randomize_object_orientation=randomize_object_orientation,
        target_anchor=target_anchor,
        target_offset=target_offset,
        fsr_source=fsr_source,
    )
    manual_policy = None
    action_dim = int(env.action_manager.total_action_dim)
    if policy_type == "manual":
        manual_policy = _build_registered_policy(task_id, cfg, env.num_envs)
    elif policy is None:
        raise ValueError("Learned policy requested but no model is loaded.")

    target_mocap_idx = _resolve_target_mocap_idx(env)
    palm_body_local_idx = _resolve_palm_body_local_idx(env)
    base_target_pos = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()

    full_contact_hits = 0.0
    stability_sum = 0.0
    contact_ratio_sum = 0.0
    frames = 0
    success_episodes = 0

    for _ in range(episodes):
        with _suppress_mujoco_output():
            obs, _ = env.reset()
        _reset_target_pose(
            env,
            target_mocap_idx,
            base_target_pos,
            palm_body_local_idx,
            cfg,
            str(device),
        )

        prev_fsr = torch.zeros((num_envs, 16), device=device)
        full_contact_run = torch.zeros((num_envs,), dtype=torch.int32, device=device)

        fsr0 = _read_fsr_data(env, obs, fsr_dims=16, source=cfg.fsr_source)
        fsr_feat0 = fsr0[:, 4:] if drop_palm_fsr else fsr0
        q0 = _q_hand_from_env(env)
        pa0 = torch.zeros((num_envs, 16), device=device)

        fsr_hist = [fsr_feat0.clone() for _ in range(window)]
        q_hist = [q0.clone() for _ in range(window)]
        pa_hist = [pa0.clone() for _ in range(window)]

        any_full_contact = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        prev_action = pa0

        for _step in range(episode_steps):
            if policy_type == "manual":
                assert manual_policy is not None
                action = _adapt_action_dim(manual_policy(obs), action_dim)
            else:
                assert policy is not None
                x_seq = _prepare_x_seq(
                    fsr_hist,
                    q_hist,
                    pa_hist,
                    norm_x_mean=policy.norm_x_mean,
                    norm_x_std=policy.norm_x_std,
                )
                action = _policy_action(policy, x_seq, prev_action)

            with _suppress_mujoco_output():
                obs, _, _, _, _ = env.step(action)

            fsr_now = _read_fsr_data(env, obs, fsr_dims=16, source=cfg.fsr_source)
            fsr_feat = fsr_now[:, 4:] if drop_palm_fsr else fsr_now
            q_now = _q_hand_from_env(env)

            quality, full_contact_run = _compute_contact_quality(
                fsr_now,
                prev_fsr,
                full_contact_run,
                cfg,
            )
            prev_fsr = fsr_now.clone()

            full_contact = quality["full_contact"].squeeze(-1)
            stability = quality["contact_stability"].squeeze(-1)
            any_full_contact = torch.logical_or(any_full_contact, full_contact > 0.5)

            full_contact_hits += float((full_contact > 0.5).float().mean().item())
            stability_sum += float(stability.mean().item())
            contact_ratio_sum += float((quality["finger_contact"] > 0.5).float().mean().item())
            frames += 1

            fsr_hist.pop(0)
            q_hist.pop(0)
            pa_hist.pop(0)
            fsr_hist.append(fsr_feat.clone())
            q_hist.append(q_now.clone())
            pa_hist.append(action.clone())
            prev_action = action

        success_episodes += int(any_full_contact.float().mean().item() > 0.5)

    env.close()

    denom = max(frames, 1)
    return {
        "full_contact_ratio": full_contact_hits / denom,
        "contact_stability_mean": stability_sum / denom,
        "finger_contact_ratio": contact_ratio_sum / denom,
        "episode_success_rate": float(success_episodes) / float(max(episodes, 1)),
    }


def _play_once(
    task_id: str,
    policy_type: Literal["learned", "manual"],
    policy: LoadedPolicy | None,
    device: torch.device,
    num_envs: int,
    window: int,
    drop_palm_fsr: bool,
    fsr_source: Literal["policy", "sensor"],
    contact_threshold: float,
    stability_window: int,
    randomize_object_orientation: bool,
    target_anchor: Literal["origin", "palm"],
    target_offset: tuple[float, float, float],
    viewer: Literal["native", "viser"],
) -> None:
    with _suppress_mujoco_output():
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = num_envs
        env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))

    cfg = HeadlessCollectConfig(
        num_envs=num_envs,
        device=str(device),
        contact_threshold=contact_threshold,
        stability_window=stability_window,
        randomize_object_orientation=randomize_object_orientation,
        target_anchor=target_anchor,
        target_offset=target_offset,
        fsr_source=fsr_source,
    )
    manual_policy = None
    action_dim = int(env.action_manager.total_action_dim)
    if policy_type == "manual":
        manual_policy = _build_registered_policy(task_id, cfg, env.num_envs)
    elif policy is None:
        raise ValueError("Learned policy requested but no model is loaded.")

    target_mocap_idx = _resolve_target_mocap_idx(env)
    palm_body_local_idx = _resolve_palm_body_local_idx(env)
    base_target_pos = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()

    state: dict[str, list[torch.Tensor] | torch.Tensor | bool] = {
        "fsr_hist": [],
        "q_hist": [],
        "pa_hist": [],
        "prev_action": torch.zeros((num_envs, 16), device=device),
        "initialized": False,
    }

    raw_reset = env.reset

    def _init_histories(obs_dict) -> None:
        fsr0 = _read_fsr_data(env, obs_dict, fsr_dims=16, source=cfg.fsr_source)
        fsr_feat0 = fsr0[:, 4:] if drop_palm_fsr else fsr0
        q0 = _q_hand_from_env(env)
        pa0 = torch.zeros((num_envs, 16), device=device)

        state["fsr_hist"] = [fsr_feat0.clone() for _ in range(window)]
        state["q_hist"] = [q0.clone() for _ in range(window)]
        state["pa_hist"] = [pa0.clone() for _ in range(window)]
        state["prev_action"] = pa0
        state["initialized"] = True

    def reset_with_target_pose(*args, **kwargs):
        obs, info = raw_reset(*args, **kwargs)
        _reset_target_pose(
            env,
            target_mocap_idx,
            base_target_pos,
            palm_body_local_idx,
            cfg,
            str(device),
        )
        _init_histories(obs)
        return obs, info

    class GoalPolicyAdapter:
        def __call__(self, obs):
            if policy_type == "manual":
                assert manual_policy is not None
                return _adapt_action_dim(manual_policy(obs), action_dim)

            if not bool(state["initialized"]):
                _init_histories(obs)

            fsr_now = _read_fsr_data(env, obs, fsr_dims=16, source=cfg.fsr_source)
            fsr_feat = fsr_now[:, 4:] if drop_palm_fsr else fsr_now
            q_now = _q_hand_from_env(env)

            fsr_hist = state["fsr_hist"]
            q_hist = state["q_hist"]
            pa_hist = state["pa_hist"]
            prev_action = state["prev_action"]

            assert isinstance(fsr_hist, list)
            assert isinstance(q_hist, list)
            assert isinstance(pa_hist, list)
            assert isinstance(prev_action, torch.Tensor)

            fsr_hist.pop(0)
            q_hist.pop(0)
            pa_hist.pop(0)
            fsr_hist.append(fsr_feat.clone())
            q_hist.append(q_now.clone())
            pa_hist.append(prev_action.clone())

            x_seq = _prepare_x_seq(
                fsr_hist,
                q_hist,
                pa_hist,
                norm_x_mean=policy.norm_x_mean,  # type: ignore[union-attr]
                norm_x_std=policy.norm_x_std,  # type: ignore[union-attr]
            )
            action = _policy_action(policy, x_seq, prev_action)  # type: ignore[arg-type]
            state["prev_action"] = action
            return action

    try:
        env.reset = reset_with_target_pose
        viewer_env = RslRlVecEnvWrapper(env)
        policy_adapter = GoalPolicyAdapter()

        if viewer == "native":
            NativeMujocoViewer(viewer_env, policy_adapter).run()
        else:
            ViserPlayViewer(viewer_env, policy_adapter).run()
        viewer_env.close()
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online A/B policy evaluation with collection-aligned object settings."
    )
    p.add_argument("--task-id", type=str, default="Leaphand-Finger-Compliance-Control")
    p.add_argument("--policy-a-type", type=str, choices=("learned", "manual"), default="learned")
    p.add_argument("--policy-b-type", type=str, choices=("learned", "manual"), default="learned")
    p.add_argument("--model-a", type=str, default=None)
    p.add_argument("--norm-a", type=str, default=None)
    p.add_argument("--model-b", type=str, default=None)
    p.add_argument("--norm-b", type=str, default=None)
    p.add_argument("--viewer", type=str, choices=("headless", "native", "viser"), default="headless")
    p.add_argument("--play-model", type=str, choices=("a", "b"), default="a")
    p.add_argument("--play-num-envs", type=int, default=1)
    p.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-steps", type=int, default=400)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--drop-palm-fsr", action="store_true")
    p.add_argument("--fsr-source", type=str, choices=("policy", "sensor"), default="policy")
    p.add_argument("--contact-threshold", type=float, default=0.2)
    p.add_argument("--stability-window", type=int, default=20)
    p.add_argument("--randomize-object-orientation", action="store_true")
    p.add_argument("--target-anchor", type=str, choices=("origin", "palm"), default="origin")
    p.add_argument("--target-offset", type=float, nargs=3, default=(0.10, 0.0, 0.0))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-dir",
        type=str,
        default="./finger_compliance_control/data/models/logs",
        help="Directory to save evaluation summary logs.",
    )
    p.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Optional log filename (without directory). If omitted, use timestamped name.",
    )
    return p.parse_args()


def _save_eval_log(
    args: argparse.Namespace,
    model_a_desc: str,
    model_b_desc: str,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
) -> Path:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.log_name:
        filename = args.log_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"online_ab_{ts}.log"

    log_path = log_dir / filename

    delta = {k: metrics_b[k] - metrics_a[k] for k in metrics_a}
    lines = [
        "=" * 72,
        "Online A/B Evaluation Summary",
        "=" * 72,
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"task_id={args.task_id}",
        f"device={args.device}",
        f"num_envs={args.num_envs}",
        f"episodes={args.episodes}",
        f"episode_steps={args.episode_steps}",
        f"window={args.window}",
        f"drop_palm_fsr={args.drop_palm_fsr}",
        f"fsr_source={args.fsr_source}",
        f"contact_threshold={args.contact_threshold}",
        f"stability_window={args.stability_window}",
        f"target_anchor={args.target_anchor}",
        f"target_offset={tuple(args.target_offset)}",
        f"randomize_object_orientation={args.randomize_object_orientation}",
        f"seed={args.seed}",
        "",
        "[Model A]",
        f"policy_type={args.policy_a_type}",
        f"model={model_a_desc}",
        (
            f"full_contact_ratio={metrics_a['full_contact_ratio']:.4f}, "
            f"contact_stability_mean={metrics_a['contact_stability_mean']:.4f}, "
            f"finger_contact_ratio={metrics_a['finger_contact_ratio']:.4f}, "
            f"episode_success_rate={metrics_a['episode_success_rate']:.4f}"
        ),
        "",
        "[Model B]",
        f"policy_type={args.policy_b_type}",
        f"model={model_b_desc}",
        (
            f"full_contact_ratio={metrics_b['full_contact_ratio']:.4f}, "
            f"contact_stability_mean={metrics_b['contact_stability_mean']:.4f}, "
            f"finger_contact_ratio={metrics_b['finger_contact_ratio']:.4f}, "
            f"episode_success_rate={metrics_b['episode_success_rate']:.4f}"
        ),
        "",
        "[Delta B - A]",
        f"full_contact_ratio={delta['full_contact_ratio']:+.4f}",
        f"contact_stability_mean={delta['contact_stability_mean']:+.4f}",
        f"finger_contact_ratio={delta['finger_contact_ratio']:+.4f}",
        f"episode_success_rate={delta['episode_success_rate']:+.4f}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def _load_if_needed(policy_type: Literal["learned", "manual"], model_path: str | None, norm_path: str | None) -> LoadedPolicy | None:
        if policy_type == "manual":
            return None
        if model_path is None or norm_path is None:
            raise ValueError("Learned policy requires both --model-* and --norm-* arguments.")
        return _load_policy(model_path, norm_path, device)

    policy_a = _load_if_needed(args.policy_a_type, args.model_a, args.norm_a)
    policy_b = _load_if_needed(args.policy_b_type, args.model_b, args.norm_b)

    model_a_desc = (
        f"{args.model_a} | {args.norm_a}" if args.policy_a_type == "learned" else "registered manual compliance policy"
    )
    model_b_desc = (
        f"{args.model_b} | {args.norm_b}" if args.policy_b_type == "learned" else "registered manual compliance policy"
    )

    if args.viewer != "headless":
        selected = policy_a if args.play_model == "a" else policy_b
        selected_type = args.policy_a_type if args.play_model == "a" else args.policy_b_type
        selected_name = model_a_desc if args.play_model == "a" else model_b_desc
        print("=" * 72)
        print("Policy Play Mode")
        print("=" * 72)
        print(f"viewer={args.viewer} policy_type={selected_type} model={selected_name} envs={args.play_num_envs}")
        _play_once(
            task_id=args.task_id,
            policy_type=selected_type,
            policy=selected,
            device=device,
            num_envs=args.play_num_envs,
            window=args.window,
            drop_palm_fsr=args.drop_palm_fsr,
            fsr_source=args.fsr_source,
            contact_threshold=args.contact_threshold,
            stability_window=args.stability_window,
            randomize_object_orientation=args.randomize_object_orientation,
            target_anchor=args.target_anchor,
            target_offset=tuple(args.target_offset),
            viewer=args.viewer,
        )
        return

    eval_kwargs = dict(
        task_id=args.task_id,
        device=device,
        num_envs=args.num_envs,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        window=args.window,
        drop_palm_fsr=args.drop_palm_fsr,
        fsr_source=args.fsr_source,
        contact_threshold=args.contact_threshold,
        stability_window=args.stability_window,
        randomize_object_orientation=args.randomize_object_orientation,
        target_anchor=args.target_anchor,
        target_offset=tuple(args.target_offset),
    )

    print("=" * 72)
    print("Online A/B Evaluation (Collection-aligned target setup)")
    print("=" * 72)
    print(f"task={args.task_id} device={device.type} envs={args.num_envs} episodes={args.episodes}")
    print(f"policy_a={model_a_desc}")
    print(f"policy_b={model_b_desc}")
    print(
        "target_setup: "
        f"anchor={args.target_anchor}, offset={tuple(args.target_offset)}, "
        f"randomize_orientation={args.randomize_object_orientation}"
    )

    metrics_a = _evaluate_once(policy_type=args.policy_a_type, policy=policy_a, **eval_kwargs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metrics_b = _evaluate_once(policy_type=args.policy_b_type, policy=policy_b, **eval_kwargs)

    def _fmt(m: dict[str, float]) -> str:
        return (
            f"full_contact_ratio={m['full_contact_ratio']:.4f}, "
            f"contact_stability_mean={m['contact_stability_mean']:.4f}, "
            f"finger_contact_ratio={m['finger_contact_ratio']:.4f}, "
            f"episode_success_rate={m['episode_success_rate']:.4f}"
        )

    print("\n[Model A]")
    print(_fmt(metrics_a))
    print("\n[Model B]")
    print(_fmt(metrics_b))

    print("\n[Delta B - A]")
    for k in metrics_a:
        print(f"{k}: {metrics_b[k] - metrics_a[k]:+.4f}")

    log_path = _save_eval_log(args, model_a_desc, model_b_desc, metrics_a, metrics_b)
    print(f"\n[Saved] Evaluation log: {log_path}")


if __name__ == "__main__":
    main()
