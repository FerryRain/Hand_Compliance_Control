"""Slice raw MCC collection H5 files into DP training samples.

Implements the DP_DATA_COLLECTION_GUIDE sample contract:

    O_t = [q_{t-H:t}, p^c_{t-H:t}, n^c_{t-H:t}, f^c_{t-H:t},
           m^c_{t-H:t}, delta_T^palm_{t:t+K}]   ->   A_t = q_{t:t+K}

where q* is the privileged teacher reference (q_ref) and q_nom is the plan's
nominal grasp q (plan q_hand[0], stored as ``q_nominal``).  Privileged
curvature metadata (contact_curvature_k1/k2) is stored beside each sample for
dataset balancing and curvature-space evaluation only -- never as input.

Output H5 layout (one dataset per tensor, samples on axis 0):
  obs_hand_q       (S, H, 16)     hand joint history (executed)
  obs_hand_qref    (S, H, 16)     teacher reference history (future-blind)
  obs_contact_pos  (S, H, 4, 3)   contact positions relative to palm
  obs_contact_nrm  (S, H, 4, 3)   oracle contact normals (palm frame)
  obs_contact_frc  (S, H, 4, 3)   fingertip forces (palm frame)
  obs_contact_mask (S, H, 4)      bool contact mask (collision + force)
  obs_palm_future  (S, K, 6)      future delta pose in current palm frame
  act_q_absolute   (S, K, 16)     future absolute hand qpos
  q_nominal        (16,)          per-file nominal grasp
  meta_curv_k1     (S, 4), meta_curv_k2 (S, 4)   mean contact curvature
  meta_contact_rate(S,)           fraction of valid contacts in the window
  meta_episode     (S,)           episode id of the window start

Usage:
  python scripts/build_dp_samples.py raw.h5 -o out.h5 [-H 30] [-K 30] \
      [--stride 5] [--min-contact-rate 0.8] [--curvature-report report.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from palm_planner_features import future_palm_delta_pose_palm


BACK_CONTACT_X_LIMIT_M = np.asarray(
    (0.012, 0.012, 0.012, 0.016), dtype=np.float64
)


def _wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def _window_mean(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean over the time axis of ``data`` where ``valid`` is True."""
    value = np.asarray(data, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if value.shape != mask.shape:
        raise ValueError(f"window mean shape mismatch: {value.shape} vs {mask.shape}")
    count = mask.sum(axis=0)
    total = np.where(mask, value, 0.0).sum(axis=0)
    return np.divide(
        total,
        np.maximum(count, 1),
        out=np.zeros_like(total, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)


def build_samples(
    path: Path,
    out_path: Path,
    history_frames: int = 30,
    action_frames: int = 30,
    stride: int = 5,
    min_contact_rate: float = 0.8,
) -> dict:
    """Slice one raw H5 into windowed samples; returns a stats dict."""
    with h5py.File(path, "r") as src:
        q = np.asarray(src["q_hand"][:, 0])                      # (T,16)
        qref = np.asarray(src["q_ref"][:, 0])                    # diagnostics only
        cpos = np.asarray(src["fingertip_contact_pos_world"][:, 0])   # (T,4,3)
        cnrm = np.asarray(src["fingertip_contact_normal_world"][:, 0])
        cfrc = np.asarray(src["fingertip_force_world"][:, 0])
        cfnd = np.asarray(src["fingertip_collision_found"][:, 0])     # (T,4)
        force = np.linalg.norm(cfrc, axis=-1)
        palm = np.asarray(src["palm_pose_world"][:, 0])          # (T,7) wxyz
        if "palm_pose_object" in src:
            palm_obj = np.asarray(src["palm_pose_object"][:, 0])
        elif "palm_command_pose_object" in src:
            palm_obj = np.asarray(src["palm_command_pose_object"])
            if palm_obj.ndim == 3:
                palm_obj = palm_obj[:, 0]
        else:
            raise KeyError(
                "build_dp_samples requires palm_pose_object or "
                "palm_command_pose_object"
            )
        tip_pose = np.asarray(src["fingertip_pose_world"][:, 0])
        q_nom = np.asarray(src["q_nominal"]).astype(np.float32)
        k1 = np.asarray(src["contact_curvature_k1"][:, 0])
        k2 = np.asarray(src["contact_curvature_k2"][:, 0])
        episode = np.asarray(src["episode_id"][:, 0])
        motion_start = int(src.attrs.get("actual_motion_start_step", 0))
        obj_id = src.attrs.get("object_id", "unknown")
        contact_threshold = float(src.attrs.get("contact_threshold", 0.05))

    # Window body: only frames within the motion window with at least
    # ``history_frames + action_frames`` remaining.
    T = q.shape[0]
    lo = max(motion_start + 1, history_frames)
    hi = T - action_frames
    starts = list(range(lo, hi, stride))

    # Every Cartesian DP feature is expressed in the instantaneous
    # ``palm_lower`` frame.  No rotation is deferred to the trainer.
    world_from_palm = _wxyz_to_matrix(palm[:, 3:7])
    palm_from_world = np.swapaxes(world_from_palm, -1, -2)
    cpos_palm = np.einsum(
        "tij,tfj->tfi", palm_from_world, cpos - palm[:, None, :3]
    )
    cnrm_palm = np.einsum("tij,tfj->tfi", palm_from_world, cnrm)
    cfrc_palm = np.einsum("tij,tfj->tfi", palm_from_world, cfrc)

    world_from_tip = _wxyz_to_matrix(tip_pose[..., 3:7])
    contact_tip = np.einsum(
        "tfji,tfj->tfi", world_from_tip, cpos - tip_pose[..., :3]
    )
    plausible = (
        np.isfinite(contact_tip).all(axis=-1)
        & (np.linalg.norm(contact_tip, axis=-1) <= 0.05)
        & (contact_tip[..., 0] <= BACK_CONTACT_X_LIMIT_M[None, :])
    )
    cmask = (
        cfnd.astype(bool) & (force >= contact_threshold) & plausible
    ).astype(np.float32)
    cpos_palm = np.where(cmask[..., None] > 0.5, cpos_palm, 0.0)
    cnrm_palm = np.where(cmask[..., None] > 0.5, cnrm_palm, 0.0)
    cfrc_palm = np.where(cmask[..., None] > 0.5, cfrc_palm, 0.0)
    planner_palm = future_palm_delta_pose_palm(
        palm_obj,
        episode,
        waypoint_count=action_frames,
        step_frames=1,
    )

    records: dict[str, list[np.ndarray]] = {
        k: [] for k in (
            "obs_hand_q", "obs_hand_qref", "obs_contact_pos", "obs_contact_nrm",
            "obs_contact_frc", "obs_contact_mask", "obs_palm_future",
            "act_q_absolute", "meta_curv_k1", "meta_curv_k2",
            "meta_contact_rate", "meta_episode",
        )
    }
    for t in starts:
        qh = q[t - history_frames : t]                            # history excludes t
        qrh = qref[t - history_frames : t]
        cp = cpos_palm[t - history_frames : t]
        cnn = cnrm_palm[t - history_frames : t]
        cf = cfrc_palm[t - history_frames : t]
        cm = cmask[t - history_frames : t]
        # Sample only windows whose history has sustained multi-finger
        # contact -- guide: learn from contact, recover from contact loss.
        rate = float(cm.mean())
        if rate < min_contact_rate:
            continue
        fut_palm = planner_palm[t]                               # (K,6)
        absolute_q = q[t : t + action_frames]                    # (K,16)
        records["obs_hand_q"].append(qh.astype(np.float32))
        records["obs_hand_qref"].append(qrh.astype(np.float32))
        records["obs_contact_pos"].append(cp.astype(np.float32))
        records["obs_contact_nrm"].append(cnn.astype(np.float32))
        records["obs_contact_frc"].append(cf.astype(np.float32))
        records["obs_contact_mask"].append(cm.astype(np.float32))
        records["obs_palm_future"].append(fut_palm.astype(np.float32))
        records["act_q_absolute"].append(absolute_q.astype(np.float32))
        # Privileged metadata (balancing/eval only).
        valid = cm > 0
        records["meta_curv_k1"].append(_window_mean(k1[t - history_frames : t], valid))
        records["meta_curv_k2"].append(_window_mean(k2[t - history_frames : t], valid))
        records["meta_contact_rate"].append(np.float32(rate))
        records["meta_episode"].append(np.int32(episode[t]))

    if not records["obs_hand_q"]:
        raise RuntimeError(f"no windows passed contact-rate filter ({min_contact_rate})")

    with h5py.File(out_path, "w") as dst:
        for key, values in records.items():
            dst.create_dataset(key, data=np.stack(values))
        dst.create_dataset("q_nominal", data=q_nom)
        dst.attrs["object_id"] = obj_id
        dst.attrs["history_frames"] = history_frames
        dst.attrs["action_frames"] = action_frames
        dst.attrs["stride"] = stride
        dst.attrs["min_contact_rate"] = min_contact_rate
        dst.attrs["source"] = str(path)
        dst.attrs["n_samples"] = len(records["obs_hand_q"])
        dst.attrs["dp_input_frame"] = "palm"
        dst.attrs["palm_frame_body"] = "palm_lower"
        dst.attrs["action_representation"] = "absolute_q"
        dst.attrs["coordinate_contract"] = (
            "contact point/normal/force and planner delta pose are all in "
            "the instantaneous palm_lower frame"
        )

    # Curvature distribution report for dataset balancing (guide sec. 7).
    k1_all = np.concatenate(records["meta_curv_k1"]).ravel()
    k2_all = np.concatenate(records["meta_curv_k2"]).ravel()
    # Cross-finger normal disagreement: max pair angle over sampled windows
    # (guide: the four fingers form a spatial tactile array; heterogeneous
    # contact state is the mode DP must learn to anticipate).
    nrm = np.concatenate(records["obs_contact_nrm"])            # (S,H,4,3)
    sample = nrm[np.linspace(0, len(nrm) - 1, min(2000, len(nrm)), dtype=int)]
    dots = sample @ sample.transpose(0, 2, 1)                   # (S,H,4,4)
    pair_angles = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    cross_max = float(pair_angles.max()) if pair_angles.size else 0.0
    report = {
        "n_samples": len(records["obs_hand_q"]),
        "object_id": obj_id,
        "curvature": {
            "k1": {
                "p05": float(np.percentile(k1_all, 5)),
                "p50": float(np.percentile(k1_all, 50)),
                "p95": float(np.percentile(k1_all, 95)),
            },
            "k2": {
                "p05": float(np.percentile(k2_all, 5)),
                "p50": float(np.percentile(k2_all, 50)),
                "p95": float(np.percentile(k2_all, 95)),
            },
            "cross_finger_normal_angle_max_deg": cross_max,
        },
        "contact_rate": {
            "p50": float(np.median(records["meta_contact_rate"])),
            "min": float(min(records["meta_contact_rate"])),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-H", "--history-frames", type=int, default=30)
    parser.add_argument("-K", "--action-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min-contact-rate", type=float, default=0.8)
    parser.add_argument("--curvature-report", type=Path, default=None)
    args = parser.parse_args()

    report = build_samples(
        args.input,
        args.output,
        history_frames=args.history_frames,
        action_frames=args.action_frames,
        stride=args.stride,
        min_contact_rate=args.min_contact_rate,
    )
    print(f"[DP-SAMPLES] {report['n_samples']} samples -> {args.output}")
    if args.curvature_report is not None:
        import json
        args.curvature_report.write_text(json.dumps(report, indent=2))
        print(f"[CURVATURE-REPORT] {args.curvature_report}")


if __name__ == "__main__":
    main()
