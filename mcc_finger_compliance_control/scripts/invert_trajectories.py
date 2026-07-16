from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


CAPSULE_RADIUS = 0.15
CAPSULE_HALF_LENGTH = 0.08


def _pose_to_rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat_xyzw = pose[..., 3:7][..., [1, 2, 3, 0]]
    rotation = R.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    return pose[..., :3], rotation


def _rt_to_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(rotation.reshape(-1, 3, 3)).as_quat().reshape(*position.shape[:-1], 4)
    return np.concatenate((position, quat_xyzw[..., [3, 0, 1, 2]]), axis=-1).astype(np.float32)


def _relative_pose(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    ref_p, ref_r = _pose_to_rt(reference)
    pos, rot = _pose_to_rt(pose)
    ref_rt = np.swapaxes(ref_r, -1, -2)
    relative_p = np.einsum("...ij,...j->...i", ref_rt, pos - ref_p)
    relative_r = ref_rt @ rot
    return _rt_to_pose(relative_p, relative_r)


def _capsule_surface_features(
    contact_pos_object: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic normal and principal-curvature proxy for the task capsule."""
    closest = np.zeros_like(contact_pos_object)
    closest[..., 2] = np.clip(
        contact_pos_object[..., 2], -CAPSULE_HALF_LENGTH, CAPSULE_HALF_LENGTH
    )
    radial = contact_pos_object - closest
    norm = np.linalg.norm(radial, axis=-1, keepdims=True)
    normal = radial / np.maximum(norm, 1.0e-9)
    on_cap = np.abs(contact_pos_object[..., 2]) > CAPSULE_HALF_LENGTH
    curvature = np.empty((*contact_pos_object.shape[:-1], 2), dtype=np.float64)
    curvature[..., 0] = 1.0 / CAPSULE_RADIUS
    curvature[..., 1] = np.where(on_cap, 1.0 / CAPSULE_RADIUS, 0.0)
    normal = np.where(valid[..., None], normal, 0.0)
    curvature = np.where(valid[..., None], curvature, 0.0)
    return normal.astype(np.float32), curvature.astype(np.float32)


def _backward_palm_twist(
    palm_pose_object: np.ndarray,
    episode_id: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute causal object-frame palm linear/angular velocity per episode."""
    position, rotation = _pose_to_rt(palm_pose_object)
    twist = np.zeros((*palm_pose_object.shape[:-1], 6), dtype=np.float32)
    flat_episode = episode_id.reshape(-1)
    flat_position = position.reshape(-1, 3)
    flat_rotation = rotation.reshape(-1, 3, 3)
    flat_twist = twist.reshape(-1, 6)
    for eid in np.unique(flat_episode):
        indices = np.flatnonzero(flat_episode == eid)
        if indices.size < 2:
            continue
        linear = (flat_position[indices[1:]] - flat_position[indices[:-1]]) / dt
        relative_rotation = (
            np.swapaxes(flat_rotation[indices[:-1]], -1, -2)
            @ flat_rotation[indices[1:]]
        )
        angular = R.from_matrix(relative_rotation).as_rotvec() / dt
        flat_twist[indices[1:], :3] = linear.astype(np.float32)
        flat_twist[indices[1:], 3:] = angular.astype(np.float32)
        # Avoid an artificial zero-velocity impulse at the first recorded frame.
        flat_twist[indices[0]] = flat_twist[indices[1]]
    return twist


def invert(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
        object_pose = np.asarray(source["object_pose_world"], dtype=np.float64)
        palm_pose = np.asarray(source["palm_pose_world"], dtype=np.float64)
        tip_pose = np.asarray(source["fingertip_pose_world"], dtype=np.float64)

        palm_pose_object = _relative_pose(object_pose, palm_pose)
        target.create_dataset("palm_pose_object", data=palm_pose_object)
        episode_id = np.asarray(source["episode_id"], dtype=np.int64)
        control_dt = float(source.attrs.get("control_dt", 0.01))
        target.create_dataset(
            "palm_twist_object",
            data=_backward_palm_twist(
                palm_pose_object,
                episode_id,
                control_dt,
            ),
        )
        target.create_dataset(
            "fingertip_pose_object",
            data=_relative_pose(object_pose[..., None, :], tip_pose),
        )
        fixed_object = np.zeros_like(object_pose, dtype=np.float32)
        fixed_object[..., 3] = 1.0
        target.create_dataset("object_pose_world", data=fixed_object)

        object_r_t = np.swapaxes(_pose_to_rt(object_pose)[1], -1, -2)
        force_world = np.asarray(source["fingertip_force_world"], dtype=np.float64)
        force_object = np.einsum("...ij,...fj->...fi", object_r_t, force_world)
        target.create_dataset("fingertip_force_object", data=force_object.astype(np.float32))

        angular_world = np.asarray(source["object_angular_velocity_world"], dtype=np.float64)
        angular_object = np.einsum("...ij,...j->...i", object_r_t, angular_world)
        # In object-fixed replay the hand/palm relative angular velocity has the opposite sign.
        target.create_dataset("planned_palm_angular_velocity_object", data=(-angular_object).astype(np.float32))

        if "fingertip_contact_pos_world" in source:
            contact_pos_world = np.asarray(source["fingertip_contact_pos_world"], dtype=np.float64)
            object_p = object_pose[..., :3]
            contact_pos_object = np.einsum(
                "...ij,...fj->...fi",
                object_r_t,
                contact_pos_world - object_p[..., None, :],
            )
            valid = (
                np.asarray(source["fingertip_contact"]).astype(bool)
                if "fingertip_contact" in source
                else np.ones(contact_pos_object.shape[:-1], dtype=bool)
            )
            contact_pos_object = np.where(valid[..., None], contact_pos_object, 0.0)
            target.create_dataset(
                "fingertip_contact_pos_object",
                data=contact_pos_object.astype(np.float32),
            )
            analytic_normal, curvature_object = _capsule_surface_features(
                contact_pos_object, valid
            )
            if "fingertip_contact_normal_world" in source:
                normal_world = np.asarray(
                    source["fingertip_contact_normal_world"], dtype=np.float64
                )
                normal_object = np.einsum(
                    "...ij,...fj->...fi",
                    object_r_t,
                    normal_world,
                )
                normal_object = np.where(valid[..., None], normal_object, 0.0)
                normal_object = normal_object.astype(np.float32)
                normal_source = "recorded_contact_sensor"
            else:
                normal_object = analytic_normal
                normal_source = "analytic_capsule_fallback"
            target.create_dataset(
                "fingertip_contact_normal_object",
                data=normal_object,
            )
            target.create_dataset(
                "fingertip_curvature_object",
                data=curvature_object,
            )

        replaced = {
            "object_pose_world",
            "palm_pose_world",
            "fingertip_pose_world",
            "fingertip_force_world",
            "object_angular_velocity_world",
            "fingertip_contact_pos_world",
            "fingertip_contact_normal_world",
        }
        for name, dataset in source.items():
            if name not in replaced:
                target.create_dataset(name, data=np.asarray(dataset))
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["inverted"] = True
        target.attrs["pose_frame"] = "object"
        target.attrs["force_frame"] = "object"
        target.attrs["source_file"] = str(input_path)
        target.attrs["surface_features"] = "analytic_capsule_from_contact_position"
        target.attrs["contact_normal_source"] = normal_source
        target.attrs["palm_twist"] = "causal_backward_difference_in_object_frame"
        target.attrs["capsule_radius"] = CAPSULE_RADIUS
        target.attrs["capsule_half_length"] = CAPSULE_HALF_LENGTH
    print(f"[SUCCESS] inverted trajectory saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert MCC fingertip data into the object frame.")
    parser.add_argument("--file", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.file is None:
        candidates = glob.glob("mcc_finger_compliance_control/data/trajectories/*.h5")
        if not candidates:
            raise FileNotFoundError("No trajectory H5 found")
        input_path = Path(max(candidates, key=lambda path: Path(path).stat().st_mtime))
    else:
        input_path = Path(args.file)
    output_path = Path(args.output) if args.output else (
        Path("mcc_finger_compliance_control/data/inverted") / f"{input_path.stem}_inverted.h5"
    )
    invert(input_path, output_path)


if __name__ == "__main__":
    main()
