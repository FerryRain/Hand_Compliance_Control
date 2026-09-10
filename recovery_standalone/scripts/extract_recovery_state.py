"""Extract a single "state A" (failure state) from a DP closed-loop rollout.

This is the first of the two tools flagged as missing in
``DAgger_Recovery_Data_Guide.md`` (section "尚需补充的两个工具"). It reads a
rollout H5 produced by ``deploy_dp_inverse.py --rollout-h5`` (optionally via
the batch wrapper ``collect_dagger_rollouts.py``), takes a human-chosen
failure frame, and writes a small "state A" H5 that a recovery planner can
warm-start from.

Scope of this tool (see the guide and the repo plan file for the full
picture):

- Reads only. Never touches the rollout or the original source file.
- Does NOT auto-detect the failure frame; ``--frame`` is required. The guide
  proposes a "valid contacts < 3 for 20-30 frames" heuristic for a future
  ``--auto-detect-frame`` option; that is intentionally not built here.
- The "remaining planner task" continuation is only correct when the rollout
  was produced with the default ``--palm-source teacher`` (i.e.
  ``palm_pose_object``/``palm_twist_object`` in the rollout are a verbatim
  replay of the original source episode). The rollout attrs do not record
  which ``--palm-source`` was used, so this is a documented precondition,
  not something this script can verify -- see the printed warning below.
- ``qvel_live`` is not recorded anywhere in the rollout format
  (``mcc_closed_loop_observation_v1``); it is approximated here by a single
  finite difference over ``control_dt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


ROLLOUT_FORMAT = "mcc_closed_loop_observation_v1"
STATE_A_FORMAT = "mcc_recovery_state_v1"

# Fields copied verbatim from the rollout at the failure frame, keyed by the
# name they get in state_A.h5. All are 1D per-frame vectors in the rollout.
ALWAYS_PRESENT_FIELDS = (
    "q_live",
    "palm_pose_object",
    "palm_twist_object",
    "fingertip_contact_mask",
    "fingertip_force_palm",
)
# Only present in the rollout when its state_schema is a GEOMETRY schema
# (dp_dataset.GEOMETRY_STATE_SCHEMAS) -- see deploy_dp_inverse.py's
# ``_append_rollout`` gate around the ``runtime.state_schema in
# GEOMETRY_STATE_SCHEMAS`` check.
GEOMETRY_GATED_FIELDS = (
    "fingertip_contact_pos_palm",
    "fingertip_contact_normal_palm",
)
# Only present when the rollout's state_schema is a PLANNER schema.
PLANNER_GATED_FIELD = "planner_palm_delta_pose_palm"
# Diagnostics-only: kept for provenance/debugging, never treated as
# authoritative "current state" by a downstream planner.
DIAGNOSTIC_FIELDS = (
    "q_cmd_applied",
    "delta_q_comp_applied",
    "e_servo",
    "dp_observation_state",
    "live_dp_state",
)


def _episode_rows(file: h5py.File, episode_id: int, name: str) -> np.ndarray:
    """Return dataset ``name`` for ``episode_id``, sorted by episode_step.

    Mirrors ``deploy_dp_inverse.py``'s private ``_episode`` helper, but is
    reimplemented locally to avoid importing that module (it pulls in
    mjlab/torch/imageio for live simulation, which this offline tool has no
    need for).
    """

    ids = np.asarray(file["episode_id"], dtype=np.int64)
    locations = np.argwhere(ids == episode_id)
    if not locations.size:
        available = np.unique(ids)
        raise ValueError(
            f"episode_id={episode_id} not found in {file.filename!r}; "
            f"available IDs include {available[:20].tolist()}"
        )
    steps = np.asarray(file["episode_step"])
    order = np.argsort(np.asarray([steps[tuple(loc)] for loc in locations], dtype=np.int64))
    locations = locations[order]
    dataset = file[name]
    return np.stack([dataset[tuple(loc)] for loc in locations], axis=0)


def extract(rollout_path: Path, frame: int, output_path: Path) -> None:
    with h5py.File(rollout_path, "r") as rollout:
        rollout_format = str(rollout.attrs.get("format", ""))
        if rollout_format != ROLLOUT_FORMAT:
            raise ValueError(
                f"{rollout_path}: attrs['format']={rollout_format!r}, expected "
                f"{ROLLOUT_FORMAT!r}. This tool only reads "
                "deploy_dp_inverse.py --rollout-h5 output."
            )
        num_frames = int(rollout["episode_step"].shape[0])
        if not (0 <= frame < num_frames):
            raise ValueError(
                f"--frame={frame} out of range; {rollout_path} has "
                f"{num_frames} frames (valid range [0, {num_frames - 1}])"
            )

        state_schema = str(rollout.attrs.get("state_schema", ""))
        control_dt = float(rollout.attrs.get("control_dt", 0.01))
        if control_dt <= 0.0:
            raise ValueError(f"{rollout_path}: control_dt must be positive")

        fields: dict[str, np.ndarray] = {}
        for name in ALWAYS_PRESENT_FIELDS:
            fields[name] = np.asarray(rollout[name][frame], dtype=np.float32)

        # These two are gated on the rollout's own state_schema; only copy
        # what the rollout actually wrote.
        for name in GEOMETRY_GATED_FIELDS:
            if name in rollout:
                fields[name] = np.asarray(rollout[name][frame], dtype=np.float32)
        if PLANNER_GATED_FIELD in rollout:
            fields[PLANNER_GATED_FIELD] = np.asarray(
                rollout[PLANNER_GATED_FIELD][frame], dtype=np.float32
            )
        for name in DIAGNOSTIC_FIELDS:
            if name in rollout:
                fields[name] = np.asarray(rollout[name][frame], dtype=np.float32)

        # qvel_live: not recorded anywhere in this rollout format. Finite
        # difference is used instead of patching deploy_dp_inverse.py (see
        # module docstring / plan file for the rationale).
        if frame > 0:
            q_prev = np.asarray(rollout["q_live"][frame - 1], dtype=np.float64)
            q_curr = np.asarray(rollout["q_live"][frame], dtype=np.float64)
            qvel_live = ((q_curr - q_prev) / control_dt).astype(np.float32)
        else:
            qvel_live = np.zeros_like(fields["q_live"])
            print(
                "[WARN] --frame=0: no previous frame to finite-difference "
                "qvel_live from; writing zeros."
            )

        source_file = str(rollout.attrs.get("source_file", ""))
        source_episode_id = int(rollout.attrs.get("source_episode_id", -1))
        if not source_file or source_episode_id < 0:
            raise ValueError(
                f"{rollout_path}: missing source_file/source_episode_id attrs; "
                "cannot reconstruct the remaining planner task."
            )

        rollout_attrs = {
            "state_schema": state_schema,
            "control_dt": control_dt,
            "contact_normal_polarity": str(
                rollout.attrs.get("contact_normal_polarity", "")
            ),
            "dp_tactile_normal_source": str(
                rollout.attrs.get("dp_tactile_normal_source", "")
            ),
            "object_id": str(rollout.attrs.get("object_id", "")),
            "object_scale": float(rollout.attrs.get("object_scale", 1.0)),
        }

    print(
        "[ASSUMPTION] Continuation trajectory assumes this rollout was produced "
        "with the default --palm-source teacher (palm_pose_object/palm_twist_object "
        "verbatim replay the original source episode). If it was produced with "
        "--palm-source active_capsule instead, the continuation below is WRONG -- "
        "the rollout attrs do not record which --palm-source was used, so this "
        "cannot be checked automatically."
    )

    source_path = Path(source_file)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"source_file={source_file!r} referenced by {rollout_path} does not exist"
        )
    with h5py.File(source_path, "r") as source:
        pose = _episode_rows(source, source_episode_id, "palm_pose_object").astype(
            np.float32
        )
        twist = _episode_rows(source, source_episode_id, "palm_twist_object").astype(
            np.float32
        )
        steps = _episode_rows(source, source_episode_id, "episode_step").astype(
            np.int32
        )
    if frame >= len(pose):
        raise ValueError(
            f"--frame={frame} exceeds the source episode length ({len(pose)}); "
            "the rollout and source_file/source_episode_id appear inconsistent."
        )
    continuation_pose = pose[frame:]
    continuation_twist = twist[frame:]
    continuation_steps = steps[frame:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        for name, value in fields.items():
            out.create_dataset(name, data=value)
        out.create_dataset("qvel_live", data=qvel_live)
        out.create_dataset("continuation_palm_pose_object", data=continuation_pose)
        out.create_dataset("continuation_palm_twist_object", data=continuation_twist)
        out.create_dataset("continuation_episode_step", data=continuation_steps)

        out.attrs["format"] = STATE_A_FORMAT
        out.attrs["source_rollout_file"] = str(rollout_path)
        out.attrs["failure_frame"] = int(frame)
        out.attrs["source_file"] = source_file
        out.attrs["source_episode_id"] = source_episode_id
        for key, value in rollout_attrs.items():
            out.attrs[key] = value
        out.attrs["palm_source_assumption"] = (
            "teacher (verbatim replay); NOT verified, rollout attrs do not "
            "record --palm-source"
        )

    print(
        f"[SUCCESS] state A extracted from {rollout_path} frame={frame} "
        f"(continuation length={len(continuation_pose)}) -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument(
        "--frame", type=int, required=True, help="Failure frame (== episode_step)."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.rollout, args.frame, args.output)


if __name__ == "__main__":
    main()
