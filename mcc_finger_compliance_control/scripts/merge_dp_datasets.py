"""Merge several per-object *_dp.h5 files (from export_palm_dp.py) into one
combined file that train_dp.py / dp_dataset.py can train on across objects.

Why this exists: invert_trajectories.py, export_palm_dp.py and train_dp.py
all take exactly one ``--file`` and dp_dataset.load_episodes() reads a single
h5py.File. There is no built-in step that combines several objects into one
cross-object training set. This script is that missing step.

It concatenates every dataset present in the source files along axis 0 and
renumbers ``episode_id`` so episodes from different objects/files never
collide (object i's episodes become ``i * 1_000_000 + original_id``). It also
writes a ``source_object`` per-frame integer array plus a small
``<output>.objects.json`` sidecar mapping those integers back to object ids,
so you can still slice or audit the merged set by object later.

All inputs must share the same ``dp_input_frame`` / ``dp_state_schema`` (and
``state_fields``, when present) — mixing schemas would silently produce a
state vector whose columns mean different things per source file.

Usage:

    python merge_dp_datasets.py \\
        --inputs mcc_finger_compliance_control/data/inverted/sphere_medium_*_dp.h5 \\
                 mcc_finger_compliance_control/data/inverted/capsule_medium_*_dp.h5 \\
                 mcc_finger_compliance_control/data/inverted/rounded_box_medium_*_dp.h5 \\
        --output mcc_finger_compliance_control/data/inverted/multiobject_dp.h5

Shell globs also work directly if your shell expands them before Python sees
them; --inputs accepts any number of paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

EPISODE_OFFSET = 1_000_000
ATTRS_THAT_MUST_MATCH = ("dp_input_frame", "dp_state_schema", "state_fields")


def _object_id_from_filename(path: Path) -> str:
    # collect_trajectories.py / run_full_pipeline.py name files
    # "<object_id>_<motion_mode>_<timestamp>...": keep everything before the
    # motion-mode token as a best-effort label. Falls back to the stem.
    stem = path.stem
    for token in ("_rotation_", "_translation_", "_combined_"):
        if token in stem:
            return stem.split(token)[0]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("Need at least two input files to merge.")

    reference_attrs: dict[str, str] | None = None
    per_file_data: list[dict[str, np.ndarray]] = []
    per_file_object_id: list[str] = []
    dataset_keys: list[str] | None = None

    for path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        with h5py.File(path, "r") as handle:
            attrs = {
                key: handle.attrs[key]
                for key in ATTRS_THAT_MUST_MATCH
                if key in handle.attrs
            }
            if reference_attrs is None:
                reference_attrs = attrs
            elif attrs != reference_attrs:
                raise SystemExit(
                    f"{path} has attrs {attrs}, expected {reference_attrs}. "
                    "Refusing to merge incompatible schemas."
                )
            keys = sorted(handle.keys())
            if dataset_keys is None:
                dataset_keys = keys
            elif keys != dataset_keys:
                raise SystemExit(
                    f"{path} has datasets {keys}, expected {dataset_keys}."
                )
            per_file_data.append({key: np.asarray(handle[key]) for key in keys})
        per_file_object_id.append(_object_id_from_filename(path))

    assert dataset_keys is not None and reference_attrs is not None

    merged: dict[str, np.ndarray] = {}
    source_object_chunks: list[np.ndarray] = []
    for file_index, data in enumerate(per_file_data):
        n_frames = len(data["episode_id"])
        source_object_chunks.append(np.full(n_frames, file_index, dtype=np.int32))
        data["episode_id"] = data["episode_id"].astype(np.int64) + (
            file_index * EPISODE_OFFSET
        )

    for key in dataset_keys:
        merged[key] = np.concatenate([data[key] for data in per_file_data], axis=0)
    merged["source_object"] = np.concatenate(source_object_chunks, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        for key, array in merged.items():
            out.create_dataset(key, data=array)
        for key, value in reference_attrs.items():
            out.attrs[key] = value

    sidecar = {
        "object_ids_by_index": per_file_object_id,
        "episode_offset": EPISODE_OFFSET,
        "source_files": [str(p) for p in args.inputs],
        "total_episodes": int(len(np.unique(merged["episode_id"]))),
        "total_frames": int(len(merged["episode_id"])),
    }
    sidecar_path = args.output.with_suffix(args.output.suffix + ".objects.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print(f"[MERGE] wrote {args.output}")
    print(f"[MERGE] wrote {sidecar_path}")
    print(f"[MERGE] {sidecar['total_episodes']} episodes, "
          f"{sidecar['total_frames']} frames, "
          f"{len(per_file_object_id)} objects: {per_file_object_id}")


if __name__ == "__main__":
    main()