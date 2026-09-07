"""Measure the DP model's local Jacobian sensitivity (G-delta test).

The "thin manifold" hypothesis: training data sits almost exactly on the
teacher manifold (delta_q ~ 0 relative to the intended trajectory), so the
network's on-manifold error can be small while its gradient in the
off-manifold (transverse) directions is uncontrolled.  A tiny execution
deviation in the observation then amplifies into a large action error --
which is exactly the closed-loop failure signature.

For each sampled observation window, perturb the joint part of the history
by delta along a random 16-D unit direction (all obs_horizon frames, a
persistent deviation), and measure:

  base_err  = ||pi(o)        - a*||      (physical units)
  pert_err  = ||pi(o + d*u)  - a*||
  R(delta)  = pert_err / base_err        (amplification ratio; >1 = worse)
  G(delta)  = ||pi(o+d*u) - pi(o)|| / d  (output drift per rad of joint
                                           deviation, physical units)

Two perturbation placements (--perturb-mode):
  last : only the observation-end frame is displaced (history stays on the
         teacher trajectory).  This is the closed-loop information structure:
         the finger was pushed off-trajectory *now*, contact geometry
         unchanged, and the model must emit a corrective intent.  A model
         with G ~ 0 here cannot close the loop on a sticking finger.
  all  : every history frame shifted by the same offset (absolute-q gauge
         direction; tip_delta targets are roughly invariant to it, so this
         isolates the shape/velocity response).

Usage:
  python audit_local_sensitivity.py --file data/dp/*.h5 \
      --model data/models/*/best.pt --deltas 0.003 0.005 0.01 0.015 0.02
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dp_dataset import (
    ACTION_DIM,
    FingertipDiffusionDataset,
    Normalization,
    load_episodes,
    state_dimensions,
    state_fields,
    state_schema,
)
from train_dp import build_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", default=None)
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[0.003, 0.005, 0.01, 0.015, 0.02])
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--perturb-mode", choices=["last", "all"], default="last")
    parser.add_argument(
        "--perturb-target",
        choices=["auto", "live_consistent", "prior", "first_q"],
        default="auto",
        help=(
            "auto uses a physically consistent q_live/e_servo perturbation for "
            "the new dual-track schema and first_q for legacy files"
        ),
    )
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "independent diffusion samples per (window, input).  With K>1 the "
            "report contains two blocks: 'single' (K=1 slice, deployment-like "
            "one-shot behavior) and 'mean-of-K' (distribution-level response). "
            "A much smaller mean-of-K G than single-sample G means the model's "
            "conditional distribution IS contracting but one-shot sampling "
            "noise dominates the point-wise gain."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional machine-readable copy of the complete G(delta) table.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} is unavailable")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    runtime_args = SimpleNamespace(**config)
    policy = build_policy(runtime_args, device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    action_dim = int(
        checkpoint.get("action_dim", config.get("action_dim", ACTION_DIM))
    )
    action_field = str(
        checkpoint.get("action_field", config.get("action_field", "q_hand"))
    )
    action_representation = str(checkpoint.get("action_representation", "absolute_q"))
    dataset_fields = tuple(state_fields(args.file))
    if dataset_fields != tuple(
        tuple(value) for value in checkpoint["state_fields"]
    ):
        raise ValueError("state fields differ between data and checkpoint")
    robot_dim, _, _ = state_dimensions(args.file)
    episodes = load_episodes(args.file, int(config["stride"]), action_field, action_dim)
    episode_ids = args.episode_ids or [
        int(value) for value in checkpoint["val_episode_ids"]
    ]
    missing = sorted(set(episode_ids) - set(episodes))
    if missing:
        raise ValueError(f"unknown episode IDs: {missing}")
    value = checkpoint["normalization"]
    normalization = Normalization(
        **{field.name: np.asarray(value[field.name]) for field in fields(Normalization)}
    )
    dataset = FingertipDiffusionDataset(
        episodes,
        episode_ids,
        normalization,
        int(config["obs_horizon"]),
        int(config["pred_horizon"]),
        robot_dim,
        state_schema(args.file),
        0.0,
        int(config["max_contact_dropout_steps"]),
        action_representation,
        float(config.get("action_waypoint_dt", int(config["stride"]) * 0.01)),
        float(config.get("kinematic_velocity_clip_rad_s", 1.0)),
        action_dim=action_dim,
    )
    if len(dataset) == 0:
        raise ValueError(f"no windows for episodes {episode_ids}")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        len(dataset), min(args.windows, len(dataset)), replace=False
    )
    obs_horizon = int(config["obs_horizon"])
    state_mean = normalization.state_mean
    state_std = normalization.state_std
    action_mean = normalization.action_mean
    action_std = normalization.action_std
    state_input_mask = np.asarray(
        checkpoint.get("state_input_mask", np.ones_like(state_mean)),
        dtype=np.float32,
    )
    field_slices: dict[str, slice] = {}
    cursor = 0
    for name, size in dataset_fields:
        field_slices[name] = slice(cursor, cursor + size)
        cursor += size
    perturb_target = args.perturb_target
    if perturb_target == "auto":
        perturb_target = (
            "live_consistent"
            if "q_prior" in field_slices and "e_servo" in field_slices
            else "first_q"
        )
    if perturb_target == "live_consistent":
        missing_fields = {"q_hand", "e_servo"} - set(field_slices)
        if missing_fields:
            raise ValueError(
                "live_consistent perturbation requires dual-track fields: "
                f"missing {sorted(missing_fields)}"
            )
        perturb_slices = (field_slices["q_hand"], field_slices["e_servo"])
    elif perturb_target == "prior":
        if "q_prior" not in field_slices:
            raise ValueError("prior perturbation requires q_prior")
        perturb_slices = (field_slices["q_prior"],)
    else:
        perturb_slices = (slice(0, ACTION_DIM),)

    def forward_many(
        histories: np.ndarray, seed: int, repeats: int
    ) -> np.ndarray:
        """histories: (N, horizon, state_dim) unnormalized -> physical actions.

        Each of the N perturbed variants is sampled ``repeats`` times (the
        histories are tiled so the batch carries independent per-sample noise
        in one denoise pass).  The *same* seed is reused for the baseline and
        every perturbed input of one window; only the input changes, isolating
        the model's deterministic input sensitivity per sample.
        """
        tiled = np.tile(histories, (repeats, 1, 1))  # (N*K, H, D)
        normalized = ((tiled - state_mean) / state_std) * state_input_mask
        # One denoise pass over N*K rows can exceed small GPUs (K=16 x 21
        # inputs -> 336 rows OOM'd an 8 GiB card).  Chunk the batch and keep
        # a single generator stream so the samples are exactly those a single
        # call would have drawn.
        #
        # no_grad is load-bearing here: parameters keep requires_grad=True,
        # so every conditional_sample otherwise builds an autograd graph that
        # stays alive (holding ~2.3 GiB) until the *next* iteration's
        # assignment frees it -- i.e. during the next chunk's forward, one
        # stale graph overlaps the new one and two chunks of 16 already OOM.
        chunk_rows = 16
        generator = torch.Generator(device=device).manual_seed(seed)
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tiled), chunk_rows):
                stop = start + chunk_rows
                batch = {
                    "observation.state": torch.from_numpy(
                        normalized[start:stop, :, :robot_dim].astype(np.float32)
                    ).to(device),
                    "observation.environment_state": torch.from_numpy(
                        normalized[start:stop, :, robot_dim:].astype(np.float32)
                    ).to(device),
                }
                global_cond = policy.diffusion._prepare_global_conditioning(batch)
                prediction = policy.diffusion.conditional_sample(
                    stop - start, global_cond=global_cond, generator=generator
                )
                chunks.append(prediction.detach().cpu())
        physical = (
            torch.cat(chunks, dim=0).numpy() * action_std + action_mean
        )  # (N*K, pred_horizon, action_dim)
        # np.tile repeats the whole batch K times, so rows [k*N:(k+1)*N] are
        # K independent samples of the same N inputs.
        return physical.reshape(
            repeats, len(histories), int(config["pred_horizon"]), action_dim
        )  # (K, N, pred_horizon, action_dim)

    print(
        f"[SENSITIVITY] model={args.model} episodes={episode_ids} "
        f"windows={len(indices)} action_dim={action_dim} "
        f"action={action_field!r} deltas={args.deltas} "
        f"perturb_mode={args.perturb_mode} "
        f"perturb_target={perturb_target} directions={args.directions}"
    )
    # Two measurement levels, indexed 0/1:
    #   single     = one independent sample (deployment-like one-shot)
    #   mean-of-K  = average over args.samples independent samples
    #                (distribution-level conditional response)
    samples_k = max(1, args.samples)
    deltas = np.asarray(args.deltas)
    base_errors = np.zeros((2, len(indices)))
    pert_errors = np.zeros((2, len(indices), len(deltas)))
    ratios = np.zeros((2, len(indices), len(deltas)))
    gains = np.zeros((2, len(indices), len(deltas)))
    directions = rng.standard_normal((args.directions, ACTION_DIM))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for row, (window_index, window_seed) in enumerate(
        zip(indices, rng.integers(2**31, size=len(indices)))
    ):
        eid, current = dataset.windows[int(window_index)]
        state, action_q = dataset.episodes[eid]
        history = state[current - obs_horizon + 1 : current + 1].copy()
        future = action_q[current + 1 : current + 1 + int(config["pred_horizon"])]
        teacher = future  # absolute_q representation for both A and B
        # Batch: baseline + directions x deltas in one sampled pass.
        perturbed_histories = [history]
        for direction in directions:
            for delta in deltas:
                perturbed = history.copy()
                if args.perturb_mode == "last":
                    for field_slice in perturb_slices:
                        perturbed[-1, field_slice] += delta * direction
                else:
                    for field_slice in perturb_slices:
                        perturbed[:, field_slice] += delta * direction
                perturbed_histories.append(perturbed)
        raw = forward_many(
            np.stack(perturbed_histories), int(window_seed), samples_k
        )  # (K, 1+Ndir*Ndeltas, pred_horizon, action_dim)
        # Level 0: the first independent sample; level 1: mean over K.
        samples = [raw[0], raw.mean(axis=0)]
        for level, outputs in enumerate(samples):
            a0 = outputs[0]
            base_errors[level, row] = float(
                np.linalg.norm(a0 - teacher, axis=-1).mean()
            )
            # Mean over directions so anisotropies are averaged.
            pert_dir = outputs[1:].reshape(
                args.directions, len(deltas), int(config["pred_horizon"]), action_dim
            )
            pert_norms = np.linalg.norm(
                pert_dir - teacher[None, None], axis=-1
            ).mean(axis=-1)  # (directions, deltas)
            gain_norms = np.linalg.norm(
                pert_dir - a0[None, None], axis=-1
            ).mean(axis=-1)  # (directions, deltas)
            row_pert = pert_norms.mean(axis=0)
            row_gain = (gain_norms / deltas).mean(axis=0)
            pert_errors[level, row] = row_pert
            ratios[level, row] = row_pert / max(
                base_errors[level, row], 1e-12
            )
            gains[level, row] = row_gain

    unit = "m" if action_dim != ACTION_DIM else "rad"
    level_names = ["single-sample", f"mean-of-{samples_k}"]
    payload_rows: dict[str, list[dict[str, float]]] = {}
    for level in range(2):
        print(
            f"\n== {level_names[level]} "
            f"(level {level}: {'one independent sample' if level == 0 else 'average over independent samples'}) =="
        )
        print(f"{'delta(rad)':>10} {'base_err':>10} {'pert_err':>10} "
              f"{'R p50':>8} {'R p90':>8} {'R p99':>8} {'G p50':>10} {'G p90':>10}")
        rows: list[dict[str, float]] = []
        for i, delta in enumerate(deltas):
            row = {
                "delta_rad": float(delta),
                "base_err_mean": float(base_errors[level].mean()),
                "pert_err_mean": float(pert_errors[level, :, i].mean()),
                "R_p50": float(np.median(ratios[level, :, i])),
                "R_p90": float(np.quantile(ratios[level, :, i], 0.9)),
                "R_p99": float(np.quantile(ratios[level, :, i], 0.99)),
                "G_p50": float(np.median(gains[level, :, i])),
                "G_p90": float(np.quantile(gains[level, :, i], 0.9)),
            }
            rows.append(row)
            print(
                f"{delta:>10.4f} {base_errors[level].mean():>10.6f} "
                f"{pert_errors[level, :, i].mean():>10.6f} "
                f"{np.median(ratios[level, :, i]):>8.2f} "
                f"{np.quantile(ratios[level, :, i], 0.9):>8.2f} "
                f"{np.quantile(ratios[level, :, i], 0.99):>8.2f} "
                f"{np.median(gains[level, :, i]):>10.4f} "
                f"{np.quantile(gains[level, :, i], 0.9):>10.4f}"
            )
        payload_rows[level_names[level]] = rows
    print(f"\nunit: action error in {unit}; G in {unit}/rad of joint deviation")
    print(f"base_err p50 (single): {np.median(base_errors[0]):.6f} {unit} "
          f"(teacher-manifold error at delta=0)")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model),
            "file": str(args.file),
            "episode_ids": episode_ids,
            "windows": int(len(indices)),
            "directions": int(args.directions),
            "samples": int(samples_k),
            "perturb_mode": args.perturb_mode,
            "perturb_target": perturb_target,
            "action_unit": unit,
            "base_err_p50_single": float(np.median(base_errors[0])),
            "base_err_p50_mean": float(np.median(base_errors[1])),
            "rows": payload_rows,
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"[SENSITIVITY] JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
