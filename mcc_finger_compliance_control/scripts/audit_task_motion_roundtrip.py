"""Verify that ground-truth task-tip actions survive deployment decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from deploy_dp_inverse import tip_target_to_absolute_q
from surface_mcc_finger import FullHandMCCFingerController


def _flat(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset)
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--window-step", type=int, default=50)
    parser.add_argument("--max-p95-error-mm", type=float, default=1.0)
    args = parser.parse_args()

    with h5py.File(args.file, "r") as file:
        action_field = str(file.attrs.get("action_field", ""))
        if action_field != "tip_target_palm":
            raise ValueError(
                f"round-trip audit requires tip_target_palm, got {action_field!r}"
            )
        episode_id = _flat(file["episode_id"]).reshape(-1).astype(np.int64)
        q_prior = _flat(file["q_prior"]).astype(np.float32)
        action = _flat(file[action_field]).astype(np.float32)

    available = np.unique(episode_id)
    episodes = available if args.episodes is None else np.asarray(args.episodes)
    unknown = sorted(set(episodes.tolist()) - set(available.tolist()))
    if unknown:
        raise ValueError(f"episodes not present in file: {unknown}")

    controller = FullHandMCCFingerController()
    errors: list[np.ndarray] = []
    q_changes: list[np.ndarray] = []
    windows = 0
    first = (16 - 1) * args.stride
    last_margin = args.pred_horizon * args.stride
    for episode in episodes:
        indices = np.flatnonzero(episode_id == episode)
        for local_t in range(first, len(indices) - last_margin, args.window_step):
            current = indices[local_t]
            future = indices[
                local_t
                + args.stride
                * np.arange(1, args.pred_horizon + 1)
            ]
            base_q = q_prior[current]
            target = action[future]
            q_chunk = tip_target_to_absolute_q(
                controller,
                action[future],
                base_q,
                args.pred_horizon,
            )
            decoded_tip = np.asarray(
                [controller.tip_positions_palm(q) for q in q_chunk]
            )
            errors.append(np.linalg.norm(decoded_tip - target, axis=-1))
            q_changes.append(np.abs(q_chunk - base_q[None, :]))
            windows += 1

    if not errors:
        raise RuntimeError("no valid round-trip windows")
    error_mm = 1000.0 * np.concatenate(errors).reshape(-1)
    q_change = np.concatenate(q_changes).reshape(-1)
    print(
        "[ROUNDTRIP] "
        f"episodes={len(episodes)} windows={windows} "
        "tip_error_mm p50/p90/p95/p99/max="
        + "/".join(
            f"{value:.3f}"
            for value in np.percentile(error_mm, (50, 90, 95, 99, 100))
        )
        + " q_change_rad_p95="
        + f"{np.percentile(q_change, 95):.4f}"
    )
    p95 = float(np.percentile(error_mm, 95))
    if p95 > args.max_p95_error_mm:
        raise RuntimeError(
            f"round-trip P95 {p95:.3f}mm exceeds "
            f"{args.max_p95_error_mm:.3f}mm"
        )
    print("[PASS] ground-truth task motion is executable by the deploy decoder")


if __name__ == "__main__":
    main()
