"""Compare local future-q predictability across palm-frame DP datasets.

This diagnostic deliberately avoids the diffusion network.  It asks whether a
validation observation has similar training observations whose demonstrated
future joint trajectories agree.  If a nearest-neighbour regressor cannot beat
holding the current q, the dataset is locally ambiguous, poorly covered, or its
command horizon is insufficient; adding network capacity alone is unlikely to
fix that failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from dp_dataset import load_episodes, split_episode_ids


FEATURE_SLICES = {
    "q": slice(0, 16),
    "q_geometry_mask": slice(0, 44),
    "q_geometry_motion": slice(0, 84),
    "all": slice(None),
}


def _windows(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: list[int],
    obs_horizon: int,
    pred_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation, future, current_q = [], [], []
    for episode_id in episode_ids:
        state, q = episodes[episode_id]
        indices = np.arange(obs_horizon - 1, len(q) - pred_horizon)
        # Match the policy contract: condition on the complete causal history,
        # not only the final observation.  Using one frame here materially
        # overstates ambiguity for trajectories whose phase is visible from
        # recent motion.
        observation.append(
            np.stack(
                [state[i - obs_horizon + 1 : i + 1] for i in indices]
            )
        )
        future.append(np.stack([q[i + 1 : i + 1 + pred_horizon] for i in indices]))
        current_q.append(q[indices])
    return (
        np.concatenate(observation),
        np.concatenate(future),
        np.concatenate(current_q),
    )


def audit(path: Path, args: argparse.Namespace) -> None:
    episodes = load_episodes(path, args.stride)
    train_ids, val_ids = split_episode_ids(
        list(episodes), args.val_ratio, args.seed
    )
    train_x, train_y, _ = _windows(
        episodes, train_ids, args.obs_horizon, args.pred_horizon
    )
    val_x, val_y, val_q = _windows(
        episodes, val_ids, args.obs_horizon, args.pred_horizon
    )
    rng = np.random.default_rng(args.seed)
    if len(train_x) > args.max_train_windows:
        selected = rng.choice(len(train_x), args.max_train_windows, replace=False)
        train_x, train_y = train_x[selected], train_y[selected]
    if len(val_x) > args.max_val_windows:
        selected = rng.choice(len(val_x), args.max_val_windows, replace=False)
        val_x, val_y, val_q = val_x[selected], val_y[selected], val_q[selected]

    hold = np.abs(val_y - val_q[:, None])
    print(
        f"\n{path.name}: episodes={len(episodes)} train/val={len(train_ids)}/{len(val_ids)} "
        f"windows={len(train_x)}/{len(val_x)} hold_first4={hold[:, :4].mean():.6f} "
        f"hold_full={hold.mean():.6f} rad"
    )
    for name, feature_slice in FEATURE_SLICES.items():
        x_train = train_x[..., feature_slice].reshape(len(train_x), -1).astype(
            np.float64
        )
        x_val = val_x[..., feature_slice].reshape(len(val_x), -1).astype(np.float64)
        mean = x_train.mean(axis=0)
        scale = np.maximum(x_train.std(axis=0), 1.0e-5)
        x_train = (x_train - mean) / scale
        x_val = (x_val - mean) / scale
        neighbours = NearestNeighbors(
            n_neighbors=args.neighbours, algorithm="auto", n_jobs=-1
        ).fit(x_train)
        distance, index = neighbours.kneighbors(x_val)
        prediction = train_y[index].mean(axis=1)
        error = np.abs(prediction - val_y)
        print(
            f"  {name:16s} d_med={np.median(distance[:, 0]):7.3f} "
            f"knn_first4={error[:, :4].mean():.6f} "
            f"knn_full={error.mean():.6f} "
            f"full/hold={error.mean() / hold.mean():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=50_000)
    parser.add_argument("--max-val-windows", type=int, default=2_000)
    args = parser.parse_args()
    for path in args.files:
        audit(path, args)


if __name__ == "__main__":
    main()
