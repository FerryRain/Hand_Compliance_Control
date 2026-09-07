"""Combine compatible palm-frame DP H5 files without crossing episode boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


CONTRACT_ATTRS = (
    "schema_version",
    "dp_input_frame",
    "palm_frame_body",
    "dp_state_schema",
    "state_fields",
    "action_field",
    "action_representation",
    "action_coordinate_space",
    "planner_waypoints",
    "planner_step_frames",
    "planner_horizon_frames",
    "planner_waypoint_dt",
    "planner_horizon_seconds",
    "motion_feature_step_frames",
    "motion_feature_dt",
    "motion_feature_convention",
    "control_dt",
    "contact_normal_polarity",
)


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be DOMAIN=PATH")
    domain, path = value.split("=", 1)
    if not domain:
        raise argparse.ArgumentTypeError("input domain cannot be empty")
    return domain, Path(path)


def _attribute(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def combine(
    inputs: list[tuple[str, Path]],
    output: Path,
    train_only_domains: set[str] | None = None,
) -> None:
    if len(inputs) < 2:
        raise ValueError("at least two --input DOMAIN=PATH values are required")
    domain_names = [domain for domain, _ in inputs]
    if len(domain_names) != len(set(domain_names)):
        raise ValueError(f"domain names must be unique: {domain_names}")
    train_only_domains = set(train_only_domains or ())
    unknown_train_only = train_only_domains - set(domain_names)
    if unknown_train_only:
        raise ValueError(
            f"unknown --train-only-domain values: {sorted(unknown_train_only)}"
        )

    metadata: list[dict[str, object]] = []
    reference_contract: dict[str, object] | None = None
    reference_layout: dict[str, tuple[tuple[int, ...], np.dtype]] | None = None
    total_frames = 0
    total_episodes = 0
    for domain, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as source:
            contract = {
                key: _attribute(source.attrs[key])
                for key in CONTRACT_ATTRS
                if key in source.attrs
            }
            if reference_contract is None:
                reference_contract = contract
            elif contract != reference_contract:
                differences = {
                    key: (reference_contract.get(key), contract.get(key))
                    for key in sorted(set(reference_contract) | set(contract))
                    if reference_contract.get(key) != contract.get(key)
                }
                raise ValueError(f"{path}: incompatible DP contract: {differences}")
            layout = {
                name: (dataset.shape[1:], dataset.dtype)
                for name, dataset in source.items()
                if name not in ("episode_id", "episode_domain_id", "episode_source_id")
            }
            if reference_layout is None:
                reference_layout = layout
            elif layout != reference_layout:
                raise ValueError(f"{path}: dataset layout differs from first input")
            episode_id = np.asarray(source["episode_id"]).reshape(-1)
            unique_episode = np.unique(episode_id)
            frames = len(episode_id)
            metadata.append(
                {
                    "domain": domain,
                    "path": str(path.resolve()),
                    "frames": frames,
                    "episodes": len(unique_episode),
                    "episode_ids": unique_episode,
                    "normal_source": str(source.attrs.get("contact_normal_source", "")),
                }
            )
            total_frames += frames
            total_episodes += len(unique_episode)

    assert reference_contract is not None and reference_layout is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as target:
        outputs: dict[str, h5py.Dataset] = {}
        for name, (tail, dtype) in reference_layout.items():
            outputs[name] = target.create_dataset(
                name,
                shape=(total_frames, *tail),
                dtype=dtype,
                chunks=(min(4096, total_frames), *tail),
            )
        episode_out = target.create_dataset(
            "episode_id", shape=(total_frames, 1), dtype="i4", chunks=True
        )
        domain_out = target.create_dataset(
            "episode_domain_id", shape=(total_frames, 1), dtype="i2", chunks=True
        )
        source_episode_out = target.create_dataset(
            "episode_source_id", shape=(total_frames, 1), dtype="i4", chunks=True
        )

        frame_cursor = 0
        episode_cursor = 0
        for domain_id, ((_, path), info) in enumerate(zip(inputs, metadata, strict=True)):
            frames = int(info["frames"])
            selection = slice(frame_cursor, frame_cursor + frames)
            with h5py.File(path, "r") as source:
                for name in outputs:
                    outputs[name][selection] = source[name]
                original = np.asarray(source["episode_id"]).reshape(-1).astype(np.int64)
                unique = np.asarray(info["episode_ids"], dtype=np.int64)
                remap = {int(value): episode_cursor + i for i, value in enumerate(unique)}
                remapped = np.fromiter(
                    (remap[int(value)] for value in original),
                    dtype=np.int32,
                    count=frames,
                )
                episode_out[selection, 0] = remapped
                domain_out[selection, 0] = domain_id
                source_episode_out[selection, 0] = original.astype(np.int32)
            frame_cursor += frames
            episode_cursor += len(unique)

        for key, value in reference_contract.items():
            target.attrs[key] = value
        target.attrs["combined_dataset"] = True
        target.attrs["domain_names_json"] = json.dumps(domain_names)
        target.attrs["source_files_json"] = json.dumps(
            [str(info["path"]) for info in metadata]
        )
        target.attrs["source_normal_conventions_json"] = json.dumps(
            {str(info["domain"]): str(info["normal_source"]) for info in metadata}
        )
        target.attrs["contact_normal_source"] = "domain-specific smooth inward surface normal"
        target.attrs["num_trajectories"] = total_episodes
        target.attrs["train_only_domains_json"] = json.dumps(
            sorted(train_only_domains)
        )

    with h5py.File(output, "r") as check:
        ids = np.asarray(check["episode_id"]).reshape(-1)
        if not np.array_equal(np.unique(ids), np.arange(total_episodes)):
            raise RuntimeError("combined episode IDs are not contiguous")
    print(
        f"[SUCCESS] combined {len(inputs)} domains, {total_episodes} episodes and "
        f"{total_frames} frames -> {output}"
    )
    for domain_id, info in enumerate(metadata):
        print(
            f"  domain={domain_id}:{info['domain']} episodes={info['episodes']} "
            f"frames={info['frames']} normals={info['normal_source']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-only-domain",
        action="append",
        default=[],
        help="Domain excluded from validation and always placed in training.",
    )
    args = parser.parse_args()
    combine(args.input, args.output, set(args.train_only_domain))


if __name__ == "__main__":
    main()
