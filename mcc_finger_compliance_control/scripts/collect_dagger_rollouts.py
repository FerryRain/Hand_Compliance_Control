"""Collect resumable closed-loop DP rollouts for iterative DAgger.

For the released legacy 96-D q policy, ``nominal`` isolates policy
autoregression from MCC correction.  Dual-track checkpoints instead require
``live`` (enforced by deploy_dp_inverse.py).  Run this script
from the repository root: the hand XML is loaded with a repository-relative
path, and file/model paths are resolved here before the subprocess is spawned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py


def _complete(path: Path, expected_frames: int, q_source: str) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as file:
            return (
                str(file.attrs.get("dp_history_q_source", "")) == q_source
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
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--dp-replan-interval", type=int, default=10)
    parser.add_argument("--dp-samples", type=int, default=1)
    parser.add_argument(
        "--dp-history-q-source",
        choices=("nominal", "live"),
        default="nominal",
        help="Use nominal for the released legacy q-policy DAgger pilot; dual-track models require live.",
    )
    parser.add_argument(
        "--dp-tactile-normal-source",
        choices=("contact_sensor", "source_mesh_oracle"),
        default="source_mesh_oracle",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    # deploy_dp_inverse.py resolves the hand XML from the process cwd
    # (repository-relative 'src/mjlab/...'), so this script must be launched
    # from the repository root.  Resolve user-facing paths anyway so the
    # spawned subprocess is insensitive to how --file/--model were typed.
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
        if not args.overwrite and _complete(
            rollout, args.max_steps, args.dp_history_q_source
        ):
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
            str(args.inference_steps),
            "--execution-layer",
            "fullhand_mcc",
            "--mcc-direction-source",
            "hybrid",
            "--mcc-preset",
            "collection_matched_sensor",
            "--dp-history-q-source",
            args.dp_history_q_source,
            "--dp-tactile-normal-source",
            args.dp_tactile_normal_source,
            "--chunk-execution",
            "--dp-replan-interval",
            str(args.dp_replan_interval),
            "--dp-samples",
            str(args.dp_samples),
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
        if not _complete(rollout, args.max_steps, args.dp_history_q_source):
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
        "dp_history_q_source": args.dp_history_q_source,
        "dp_tactile_normal_source": args.dp_tactile_normal_source,
        "inference_steps": args.inference_steps,
        "dp_replan_interval": args.dp_replan_interval,
        "dp_samples": args.dp_samples,
        "records": records,
    }
    manifest_path = args.output_dir / "rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DAGGER-COLLECT] complete -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
