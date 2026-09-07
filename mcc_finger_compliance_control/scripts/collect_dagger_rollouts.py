"""Collect resumable nominal-history DP rollouts for iterative DAgger."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py


def _complete(path: Path, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as file:
            return (
                str(file.attrs.get("dp_history_q_source", "")) == "nominal"
                and "dp_observation_state" in file
                and len(file["dp_observation_state"]) >= expected_frames
            )
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    # deploy_dp_inverse.py changes into its own script directory while importing
    # the local environment.  Resolve every user-facing path before spawning it,
    # otherwise repository-relative paths silently become script-relative.
    args.file = args.file.expanduser().resolve(strict=True)
    args.model = args.model.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deploy = Path(__file__).with_name("deploy_dp_inverse.py").resolve(strict=True)
    records: list[dict[str, object]] = []
    for index, episode_id in enumerate(args.episodes, start=1):
        stem = f"ep{episode_id:03d}"
        rollout = args.output_dir / f"{stem}.h5"
        report = args.output_dir / f"{stem}.csv"
        if not args.overwrite and _complete(rollout, args.max_steps):
            print(
                f"[DAGGER-COLLECT] {index}/{len(args.episodes)} "
                f"ep={episode_id} already complete",
                flush=True,
            )
            records.append(
                {"episode_id": episode_id, "rollout": str(rollout), "skipped": True}
            )
            continue

        command = [
            sys.executable,
            str(deploy),
            "--file",
            str(args.file),
            "--model",
            str(args.model),
            "--episode-id",
            str(episode_id),
            "--mode",
            "live_dp",
            "--viewer",
            "headless",
            "--device",
            args.device,
            "--inference-steps",
            "10",
            "--execution-layer",
            "fullhand_mcc",
            "--mcc-direction-source",
            "hybrid",
            "--mcc-preset",
            "collection_matched_sensor",
            "--dp-history-q-source",
            "nominal",
            "--chunk-execution",
            "--dp-replan-interval",
            "10",
            "--max-steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
            "--rollout-h5",
            str(rollout),
            "--report",
            str(report),
        ]
        print(
            f"[DAGGER-COLLECT] {index}/{len(args.episodes)} ep={episode_id}",
            flush=True,
        )
        subprocess.run(command, check=True)
        if not _complete(rollout, args.max_steps):
            raise RuntimeError(f"Incomplete rollout after successful process: {rollout}")
        records.append(
            {"episode_id": episode_id, "rollout": str(rollout), "skipped": False}
        )

    manifest = {
        "file": str(args.file),
        "model": str(args.model),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "device": args.device,
        "seed": args.seed,
        "dp_history_q_source": "nominal",
        "records": records,
    }
    manifest_path = args.output_dir / "rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DAGGER-COLLECT] complete -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
