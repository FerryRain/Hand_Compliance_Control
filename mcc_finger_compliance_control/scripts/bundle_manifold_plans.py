"""Bundle equal-length single-environment palm plans for parallel collection."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


STACK_DATASETS = (
    "palm_pose_object",
    "palm_twist_object",
    "q_hand",
    "qvel",
    "fingertip_pose_object",
    "planner_palm_outline_min_clearance_object",
    "planner_palm_outline_mean_clearance_object",
)

# Sparse grasp boundary conditions have no environment dimension in a
# single-plan file.  A bundle inserts that dimension at axis 1, mirroring the
# (T, E, ...) convention used by palm_pose_object.
KEYFRAME_STACK_DATASETS = (
    "grasp_keyframe_frame_index",
    "grasp_keyframe_q",
    "grasp_keyframe_contact_point_object",
    "grasp_keyframe_normal_object",
    "grasp_keyframe_signed_distance",
    "grasp_keyframe_pad_normal_error",
    "grasp_keyframe_synergy_residual",
    "grasp_keyframe_isotropic_reachability",
    "grasp_keyframe_manipulability",
    "grasp_keyframe_lateral_margin",
    "grasp_keyframe_posture_deviation",
    "grasp_keyframe_valid",
)


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def bundle(paths: list[Path], output: Path) -> None:
    if len(paths) < 2:
        raise ValueError("At least two plan files are required for a bundle")
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in STACK_DATASETS}
    keyframe_arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in KEYFRAME_STACK_DATASETS
    }
    names: list[str] = []
    sources: list[str] = []
    regions: list[str] = []
    tangent_axes: list[str] = []
    seeds: list[int] = []
    section_fractions: list[float] = []
    reference_attrs: dict[str, object] | None = None
    frame_count: int | None = None
    keyframe_count: int | None = None

    for path in paths:
        with h5py.File(path, "r") as file:
            if reference_attrs is None:
                reference_attrs = {
                    "object_id": _decode(file.attrs["object_id"]),
                    "object_scale": float(file.attrs.get("object_scale", 1.0)),
                    "control_dt": float(file.attrs.get("control_dt", 0.01)),
                    "pose_frame": _decode(file.attrs.get("pose_frame", "object")),
                }
                frame_count = int(file["palm_pose_object"].shape[0])
            else:
                if _decode(file.attrs["object_id"]) != reference_attrs["object_id"]:
                    raise ValueError("All plans in a bundle must use the same object")
                if int(file["palm_pose_object"].shape[0]) != frame_count:
                    raise ValueError("All plans in a bundle must have equal frame counts")
            for dataset_name in STACK_DATASETS:
                if dataset_name not in file:
                    continue
                value = np.asarray(file[dataset_name])
                if value.ndim < 2 or value.shape[1] != 1:
                    raise ValueError(
                        f"Expected single-environment {dataset_name} in {path}, "
                        f"got {value.shape}"
                    )
                arrays[dataset_name].append(value[:, 0])
            for dataset_name in KEYFRAME_STACK_DATASETS:
                if dataset_name not in file:
                    raise KeyError(
                        f"{path} has no required sparse dataset {dataset_name}"
                    )
                value = np.asarray(file[dataset_name])
                if keyframe_count is None:
                    keyframe_count = int(value.shape[0])
                elif int(value.shape[0]) != keyframe_count:
                    raise ValueError(
                        "All plans in a bundle must have equal keyframe counts"
                    )
                keyframe_arrays[dataset_name].append(value)
            names.append(path.stem)
            sources.append(str(path))
            regions.append(
                _decode(file.attrs.get("planner_ellipse_arc_region", "section"))
            )
            tangent_axes.append(
                _decode(file.attrs.get("planner_palm_tangent_axis", "unknown"))
            )
            seeds.append(int(file.attrs.get("planner_seed", -1)))
            section_fractions.append(
                float(file.attrs.get("planner_ellipse_section_fraction", -1.0))
            )

    assert reference_attrs is not None and frame_count is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output, "w") as target:
        for dataset_name, values in arrays.items():
            if values:
                target.create_dataset(dataset_name, data=np.stack(values, axis=1))
        for dataset_name, values in keyframe_arrays.items():
            target.create_dataset(dataset_name, data=np.stack(values, axis=1))
        environment_count = len(paths)
        target.create_dataset(
            "episode_id",
            data=np.broadcast_to(
                np.arange(environment_count, dtype=np.int64)[None, :],
                (frame_count, environment_count),
            ),
        )
        target.create_dataset(
            "record_step",
            data=np.broadcast_to(
                np.arange(frame_count, dtype=np.int64)[:, None],
                (frame_count, environment_count),
            ),
        )
        target.create_dataset("plan_name", data=np.asarray(names, dtype=string_dtype))
        target.create_dataset(
            "source_plan", data=np.asarray(sources, dtype=string_dtype)
        )
        target.create_dataset("plan_region", data=np.asarray(regions, dtype=string_dtype))
        target.create_dataset(
            "palm_tangent_axis", data=np.asarray(tangent_axes, dtype=string_dtype)
        )
        target.create_dataset("planner_seed", data=np.asarray(seeds, dtype=np.int64))
        target.create_dataset(
            "ellipse_section_fraction",
            data=np.asarray(section_fractions, dtype=np.float32),
        )
        for key, value in reference_attrs.items():
            target.attrs[key] = value
        target.attrs["inverted"] = True
        target.attrs["planner_batch_size"] = environment_count
        target.attrs["planner_bundle"] = True
    print(
        f"[SUCCESS] bundled {len(paths)} plans x {frame_count} frames into {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("plans", nargs="+", type=Path)
    args = parser.parse_args()
    bundle(args.plans, args.output)


if __name__ == "__main__":
    main()
