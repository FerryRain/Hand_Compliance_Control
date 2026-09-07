"""Export semantically causal fingertip task motion for DP.

The source is the causally aligned dual-track v3 H5.  The policy observation
keeps the complete 242-D dual-track B2 state by default. The action is the
teacher's absolute future 3-D fingertip target in the palm frame:

    a[t] = tip_x_des[t]

The exporter separately checks the stride-S target motion and rejects episodes
with discontinuous jumps. Deployment resolves each absolute target directly;
there is no cross-replan integration. MCC owns the small normal force offset.
This is intentionally different from the legacy ``tip_delta_tangent_palm``
tracking residual, which cannot be accumulated as future task motion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from dp_motion_features import (
    DUAL_TRACK_V3_SCHEMA,
    TASK_EST_V4_SCHEMA,
    causal_motion_features,
)


COPY_FIELDS = (
    "episode_id",
    "episode_step",
    "fingertip_contact_pos_palm",
    "fingertip_contact_normal_palm",
    "fingertip_contact_mask",
    "palm_relative_twist_palm",
    "planner_palm_delta_pose_palm",
    "episode_domain_id",
)

DUAL_TRACK_FIELDS = (
    "q_prior",
    "q_hand",
    "q_prior_velocity",
    "q_live_velocity",
    "e_qdot",
    "delta_q_comp",
    "e_servo",
    "fingertip_contact_point_velocity_palm",
    "fingertip_contact_normal_angular_rate_palm",
)


def _flat(value: h5py.Dataset) -> np.ndarray:
    array = np.asarray(value)
    return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])


def _task_tip_motion(
    target: np.ndarray,
    episode_id: np.ndarray,
    step_frames: int,
) -> np.ndarray:
    output = np.zeros_like(target, dtype=np.float32)
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        if len(indices) <= step_frames:
            continue
        current = indices[step_frames:]
        previous = indices[:-step_frames]
        output[current] = target[current] - target[previous]
    return output.astype(np.float32)


def _write(file: h5py.File, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    file.create_dataset(
        name,
        data=array,
        chunks=(min(2048, len(array)), *array.shape[1:]),
        compression="gzip",
        compression_opts=1,
        shuffle=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--max-action-step-mm",
        type=float,
        default=5.0,
        help=(
            "Reject complete episodes containing a per-finger task-tip "
            "increment above this value; such jumps violate the local-motion "
            "action contract. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--state-contract",
        choices=("dual_track", "task_est"),
        default="dual_track",
        help=(
            "Keep the 242-D B2 observation by default. task_est is a later "
            "observation ablation, not part of the action-contract fix."
        ),
    )
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source:
        source_schema = str(source.attrs.get("dp_state_schema", ""))
        if source_schema != "contact_geometry_planner_motion_dual_track_v3":
            raise ValueError(
                f"expected dual-track v3 source, got {source_schema!r}"
            )
        required = set(COPY_FIELDS) | {
            "q_hand",
            "delta_q_comp",
            "tip_x_des_palm",
        }
        if args.state_contract == "dual_track":
            required.update(DUAL_TRACK_FIELDS)
        missing = sorted(required - set(source.keys()))
        if missing:
            raise KeyError(f"source is missing fields: {missing}")

        episode_id = _flat(source["episode_id"]).reshape(-1).astype(np.int64)
        q_live = _flat(source["q_hand"]).astype(np.float32)
        delta_q_comp = _flat(source["delta_q_comp"]).astype(np.float32)
        q_task_est = (q_live - delta_q_comp).astype(np.float32)
        position = _flat(source["fingertip_contact_pos_palm"]).astype(np.float32)
        normal = _flat(source["fingertip_contact_normal_palm"]).astype(np.float32)
        mask = _flat(source["fingertip_contact_mask"]).astype(np.float32)
        control_dt = float(source.attrs.get("control_dt", 0.01))
        q_task_velocity, point_velocity, normal_rate = causal_motion_features(
            q_task_est,
            position,
            normal,
            mask,
            episode_id,
            control_dt=control_dt,
            step_frames=args.stride,
        )
        tip_target = _flat(source["tip_x_des_palm"]).astype(np.float32)
        task_motion = _task_tip_motion(
            tip_target,
            episode_id,
            args.stride,
        )
        action_norm = np.linalg.norm(task_motion, axis=-1)
        rejected_episodes: list[int] = []
        if args.max_action_step_mm > 0.0:
            limit_m = args.max_action_step_mm / 1000.0
            rejected_episodes = [
                int(episode)
                for episode in np.unique(episode_id)
                if np.any(action_norm[episode_id == episode] > limit_m)
            ]
        keep = ~np.isin(episode_id, rejected_episodes)
        if not np.any(keep):
            raise RuntimeError("action-step gate rejected every episode")

        def selected(dataset: h5py.Dataset) -> np.ndarray:
            value = _flat(dataset)[keep]
            return value[:, None, ...]

        with h5py.File(args.output, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            for name in COPY_FIELDS:
                _write(target, name, selected(source[name]))
            shaped_action = tip_target[keep, None, ...]
            if args.state_contract == "dual_track":
                for name in DUAL_TRACK_FIELDS:
                    _write(target, name, selected(source[name]))
                target.attrs["dp_state_schema"] = DUAL_TRACK_V3_SCHEMA
                target.attrs["state_fields"] = str(source.attrs["state_fields"])
                state_dim = 242
            else:
                shaped_task = q_task_est[keep, None, ...]
                shaped_velocity = q_task_velocity[keep, None, ...]
                _write(target, "q_hand", selected(source["q_hand"]))
                _write(target, "q_task_est", shaped_task)
                _write(target, "q_task_est_velocity", shaped_velocity)
                # Recompute these with the same stride used by the policy so
                # all causal rates share one temporal support.
                _write(
                    target,
                    "fingertip_contact_point_velocity_palm",
                    point_velocity.reshape(
                        -1, 1, 4, 3
                    )[keep],
                )
                _write(
                    target,
                    "fingertip_contact_normal_angular_rate_palm",
                    normal_rate.reshape(
                        -1, 1, 4, 3
                    )[keep],
                )
                target.attrs["dp_state_schema"] = TASK_EST_V4_SCHEMA
                target.attrs["state_fields"] = (
                    "q_task_est,fingertip_contact_pos_palm,"
                    "fingertip_contact_normal_palm,fingertip_contact_mask,"
                    "q_task_est_velocity,fingertip_contact_point_velocity_palm,"
                    "fingertip_contact_normal_angular_rate_palm,"
                    "palm_relative_twist_palm,planner_palm_delta_pose_palm"
                )
                state_dim = 162
            _write(target, "tip_x_des_palm", selected(source["tip_x_des_palm"]))
            _write(target, "tip_target_palm", shaped_action)

            target.attrs["action_field"] = "tip_target_palm"
            target.attrs["action_dim"] = 12
            target.attrs["action_representation"] = "absolute_q"
            target.attrs["action_coordinate_space"] = "palm_3d_position_m"
            target.attrs["action_step_frames"] = int(args.stride)
            target.attrs["motion_feature_step_frames"] = int(args.stride)
            target.attrs["motion_feature_dt"] = args.stride * control_dt
            target.attrs["task_state_contract"] = (
                "full dual-track B2 observation"
                if args.state_contract == "dual_track"
                else "q_task_est=q_hand-delta_q_comp; no DP nominal feedback"
            )
            target.attrs["task_action_contract"] = (
                "absolute future teacher tip_x_des in palm frame; no temporal "
                "accumulation; MCC adds an independent normal force offset"
            )
            target.attrs["task_action_max_step_m"] = (
                args.max_action_step_mm / 1000.0
                if args.max_action_step_mm > 0.0
                else np.inf
            )
            target.attrs["task_action_rejected_episode_ids"] = np.asarray(
                rejected_episodes, dtype=np.int32
            )

    task_motion = task_motion[keep]
    episode_id = episode_id[keep]
    active = np.linalg.norm(task_motion.reshape(-1, 3), axis=-1)
    active = active[active > 1.0e-9]
    print(
        f"[SUCCESS] {args.output} episodes={len(np.unique(episode_id))} "
        f"frames={len(episode_id)} state={state_dim} action=12"
    )
    if len(active):
        print(
            "[TASK-MOTION] stride displacement norm mm p50/p90/p95/p99/max="
            + "/".join(
                f"{1000.0 * value:.3f}"
                for value in np.quantile(active, [0.5, 0.9, 0.95, 0.99, 1.0])
            )
        )
    if rejected_episodes:
        print(
            f"[ACTION-GATE] rejected {len(rejected_episodes)} episodes: "
            + ",".join(str(value) for value in rejected_episodes)
        )


if __name__ == "__main__":
    main()
