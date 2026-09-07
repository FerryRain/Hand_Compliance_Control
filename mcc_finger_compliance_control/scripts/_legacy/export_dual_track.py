"""Build the dual-track DP training file: execution obs + intent labels.

Background
----------
The closed-loop failure mechanism (SPEED_UNIFORM_PROGRESS §31.3/§42) is that
training labels were the executed q_live, so the model learned "executed pose
-> executed pose continuation".  At deployment the MCC modifies the DP output,
so the DP output no longer becomes the next input and the autoregressive loop
drifts.  The fix (authorized in §42) is to decouple the meaning of input and
output:

    obs   (input):  executed joint history + palm tactile + planner   (deploy-able)
    label (output): the pre-execution INTENT (what to ask for), not the
                    executed state

The collection-era controller (collect_trajectories.py) already recorded the
intent channel: ``q_ref`` = q_command_t, the QP/planning reference sent to the
execution layer, alongside the executed ``q_hand``.  The invert / build_dp_samples
pipeline dropped it ("diagnostics only").  This exporter restores it from the
original raw trajectory files via the per-episode ``source_trajectory`` mapping
stored in the inverted H5.

This export uses the ORIGINAL 239 trajectories (executed q_hand under the
collection-era controller, same observation distribution as the D1 motion96
file), NOT a 1.5 N MCC re-collection: a pilot re-collection showed episodes 0-3
(the 4x3700 strong-press batch) destabilize under the frozen MCC (thumb driven
to 2.19 rad, force spikes 500 N), so executed-q_live relabeling was dropped
(2026-09-01, §42).  The obs channel is therefore identical to D1; only the
label semantics change to pre-execution intent.

Two label variants are written so the DP output format can be A/B'd:

  Variant A (absolute q-target):   label field  = q_ref       (16D joint intent)
  Variant B (tangent-intent delta):label field  = tip_delta_tangent_palm
      delta_t = proj_tan(tip_x_des_palm[t+1] - FK(q_hand)[t], n[t])
      i.e. the desired fingertip displacement from the current executed tip,
      projected onto the contact tangent plane; the normal component is owned
      by the MCC force control (F_n -> 1.5N) at deployment, so the policy only
      ever learns tangential intent.  Contact-less frames get a zero delta.

The output file has the same palm-frame schema as the D1 motion96 file
(dp_state_schema=contact_geometry_planner_motion), so the existing dataset /
training pipeline applies; the action channel is selected with
file.attrs["action_field"] == "q_ref" | "tip_delta_tangent_palm" and the
action dimensionality with file.attrs["action_dim"] (16 | 12).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_motion_features import MOTION_SCHEMA
from export_palm_dp import export as export_palm_dp, validate_palm_dp_file
from surface_mcc_finger import FullHandMCCFingerController


def _flat(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as file:
        return np.asarray(file[key], dtype=np.float32)


def _parse_source(source: str, repo_root: Path) -> tuple[Path, int | None]:
    """Split 'path.h5#env=N' into (path, env) and tolerate the nested
    'scripts/mcc_finger_compliance_control/...' prefix recorded for some
    episodes by resolving relative to the repo root."""
    env: int | None = None
    path_str = source
    if "#env=" in path_str:
        path_str, env_str = path_str.split("#env=", 1)
        env = int(env_str)
    candidate = Path(path_str)
    if not candidate.exists() and str(candidate).startswith(
        str(repo_root / "mcc_finger_compliance_control" / "scripts")
    ):
        candidate = repo_root / str(candidate).replace(
            str(repo_root / "mcc_finger_compliance_control" / "scripts"), ""
        ).lstrip("/")
    if not candidate.exists():
        raise FileNotFoundError(f"source trajectory missing: {path_str} ({candidate})")
    return candidate, env


def _tip_delta_tangent(
    tip_target: np.ndarray,  # (T,4,3) palm-frame desired tip (raw tip_x_des_palm)
    tip_actual: np.ndarray,  # (T,4,3) palm-frame executed tip = FK(q_hand)
    normal: np.ndarray,      # (T,4,3) palm-frame contact normal
    mask: np.ndarray,        # (T,4)   contact mask
) -> np.ndarray:
    """Per-frame tangential intent: desired displacement from the current
    executed tip, with the normal component removed.

    delta_t[t] = (tip_target[t+1] - tip_actual[t])  projected onto the
    tangent plane spanned by normal[t].  Frames without a valid contact get a
    zero delta (the MCC owns normal approach; a missing normal makes the
    projection meaningless, and a zero tangential intent is the conservative
    hold)."""
    t_total = len(tip_target)
    delta = np.zeros((t_total, 4, 3), dtype=np.float32)
    valid = mask > 0.5
    target_next = tip_target[1:]
    for finger in range(4):
        base = tip_actual[:-1, finger]
        raw = target_next[:, finger] - base
        nrm = normal[:-1, finger]
        nrm_norm = np.linalg.norm(nrm, axis=-1, keepdims=True)
        safe_n = np.divide(
            nrm,
            np.maximum(nrm_norm, 1.0e-12),
            out=np.zeros_like(nrm),
            where=nrm_norm > 1.0e-6,
        )
        normal_component = np.sum(raw * safe_n, axis=-1, keepdims=True) * safe_n
        tangent = raw - normal_component
        valid_row = valid[:-1, finger]
        delta[:-1, finger] = np.where(valid_row[:, None], tangent, 0.0)
    return delta


def export(
    inverted_path: Path,
    output_path: Path,
    planner_waypoints: int = 12,
    planner_step_frames: int = 6,
    motion_feature_step_frames: int = 5,
) -> dict[str, object]:
    """Export the dual-track file from the inverted H5 + original raw files.

    The palm-frame observation block (q_hand, contact geometry, twist,
    planner, motion features) is produced by export_palm_dp.export() — exactly
    the D1 motion96 pipeline — so the obs distribution matches D1 by
    construction.  Intent channels (q_ref, tip target/delta) are then merged
    per episode from the raw source files.
    """
    repo_root = Path(__file__).resolve().parents[2]

    with h5py.File(inverted_path, "r") as file:
        episode_ids = np.asarray(file["episode_id"], dtype=np.int64)[:, 0]
        sources = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in np.asarray(file["source_trajectory"]).tolist()
        ]
        inverted_q = np.asarray(file["q_hand"], dtype=np.float32).reshape(-1, 16)
    n_episodes = len(sources)
    total = len(episode_ids)
    print(
        f"[DUAL] {n_episodes} episodes, {total} frames, obs = original q_hand "
        "(same as D1); intent labels restored from raw files",
        flush=True,
    )

    # Step 1: palm-frame observation block — identical to D1.
    with tempfile.TemporaryDirectory(dir=str(output_path.parent)) as tmp_dir:
        base_path = Path(tmp_dir) / "dual_track_base.h5"
        export_palm_dp(
            inverted_path,
            base_path,
            state_schema=MOTION_SCHEMA,
            planner_waypoints=planner_waypoints,
            planner_step_frames=planner_step_frames,
            motion_feature_step_frames=motion_feature_step_frames,
        )
        with h5py.File(base_path, "r") as base:
            base_attrs = {key: value for key, value in base.attrs.items()}
            base_data = {
                name: np.asarray(base[name]) for name in base.keys()
            }

    # Step 2: intent channels from the raw source files, row-aligned with the
    # inverted q_hand (verified: inverted q_hand == raw q_hand, diff 0.0).
    fk = FullHandMCCFingerController()
    merged_qref = np.empty((total, 16), dtype=np.float32)
    merged_tip_target = np.empty((total, 4, 3), dtype=np.float32)
    merged_tip_actual = np.empty((total, 4, 3), dtype=np.float32)
    merged_tip_delta = np.empty((total, 4, 3), dtype=np.float32)
    merged_mask = base_data["fingertip_contact_mask"].reshape(-1, 4)
    merged_normal = base_data["fingertip_contact_normal_palm"].reshape(-1, 4, 3)

    per_episode: list[dict[str, object]] = []
    for episode_id in range(n_episodes):
        rows = np.flatnonzero(episode_ids == episode_id)
        if len(rows) == 0:
            print(f"[WARN] ep {episode_id}: no rows in inverted; skipped", flush=True)
            continue
        count = len(rows)
        source, env = _parse_source(sources[episode_id], repo_root)
        with h5py.File(source, "r") as file:
            qref_raw = np.asarray(file["q_ref"], dtype=np.float32)
            tip_raw = np.asarray(file["tip_x_des_palm"], dtype=np.float32)
            qh_raw = np.asarray(file["q_hand"], dtype=np.float32)
        # raw layout: (T, env, ...) for batched files, (T, 1, ...) / (T, ...) for
        # single-env files; tip channels may lack the env dim when it is 1.
        if qref_raw.ndim == 3:
            qref = qref_raw[:, env if env is not None else 0]
        else:
            qref = qref_raw
        qref = qref.reshape(-1, 16)
        if tip_raw.ndim == 4:
            tip_target = tip_raw[:, env if env is not None else 0]
        else:
            tip_target = tip_raw
        tip_target = tip_target.reshape(-1, 4, 3)
        if qh_raw.ndim == 3:
            qh_raw_ep = qh_raw[:, env if env is not None else 0]
        else:
            qh_raw_ep = qh_raw
        qh_raw_ep = qh_raw_ep.reshape(-1, 16)
        if len(qref) < count:
            raise ValueError(
                f"ep{episode_id}: raw q_ref {len(qref)} < inverted rows {count}"
            )
        if len(tip_target) < count:
            raise ValueError(
                f"ep{episode_id}: raw tip_x_des_palm {len(tip_target)} < rows {count}"
            )

        q_hand = inverted_q[rows]
        qref = qref[:count]
        tip_target = tip_target[:count]
        qh_raw_ep = qh_raw_ep[:count]
        align = float(np.sqrt(np.mean(np.square(q_hand - qh_raw_ep))))
        if align > 1.0e-4:
            raise ValueError(
                f"ep{episode_id}: inverted vs raw q_hand misaligned "
                f"(rms {align:.6f} rad); source mapping is wrong"
            )

        tip_actual = np.stack(
            [fk.tip_positions_palm(row) for row in q_hand], axis=0
        ).astype(np.float32)
        tip_delta = _tip_delta_tangent(
            tip_target, tip_actual, merged_normal[rows], merged_mask[rows]
        )

        merged_qref[rows] = qref
        merged_tip_target[rows] = tip_target
        merged_tip_actual[rows] = tip_actual
        merged_tip_delta[rows] = tip_delta

        q_dev = q_hand - qref
        per_episode.append(
            {
                "episode_id": episode_id,
                "frames": count,
                "source": sources[episode_id],
                "q_hand_vs_qref_rms_rad": float(np.sqrt(np.mean(np.square(q_dev)))),
                "q_hand_vs_qref_max_rad": float(np.max(np.abs(q_dev))),
                "tip_delta_norm_mm": float(
                    np.mean(np.linalg.norm(tip_delta, axis=-1)) * 1000.0
                ),
            }
        )
        print(
            f"[DUAL] ep {episode_id:3d}: n={count} "
            f"q_rms={per_episode[-1]['q_hand_vs_qref_rms_rad']:.4f}rad "
            f"q_max={per_episode[-1]['q_hand_vs_qref_max_rad']:.4f}rad "
            f"tip_delta={per_episode[-1]['tip_delta_norm_mm']:.2f}mm",
            flush=True,
        )

    # Step 3: write the merged file (base obs block + intent channels).
    with h5py.File(output_path, "w") as target:
        chunk = min(4096, total)
        for name, data in base_data.items():
            target.create_dataset(name, data=data, chunks=(chunk, *data.shape[1:]))
        target.create_dataset(
            "q_ref", data=merged_qref[:, None, :], chunks=(chunk, 1, 16)
        )
        target.create_dataset(
            "tip_target_palm",
            data=merged_tip_target[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )
        target.create_dataset(
            "tip_actual_palm",
            data=merged_tip_actual[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )
        target.create_dataset(
            "tip_delta_tangent_palm",
            data=merged_tip_delta[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )

        for key, value in base_attrs.items():
            target.attrs[key] = value
        target.attrs["schema_version"] = "mcc_dual_track_palm_v2"
        target.attrs["dual_track"] = True
        # This export uses the ORIGINAL collection-era executed q_hand (the D1
        # distribution), not a 1.5N MCC re-collection (§42, 2026-09-01).
        target.attrs["obs_q_source"] = "original_collection_executed_q_hand"
        target.attrs["relabel_source"] = "original_raw_trajectories"
        target.attrs["relabel_rollout_dir"] = ""
        target.attrs["relabel_q_live"] = False
        target.attrs["action_field"] = "q_ref"
        target.attrs["action_dim"] = 16
        target.attrs["action_representation"] = "absolute_q"
        target.attrs["action_coordinate_space"] = "joint_position_rad"
        target.attrs["tip_delta_tangent_convention"] = (
            "proj_tan(tip_x_des_palm[t+1] - FK(q_hand)[t], normal[t]); "
            "normal component owned by deployment MCC force control; "
            "zero where contact mask is invalid"
        )
        target.attrs["tip_target_source"] = "collection-era tip_x_des_palm"
        target.attrs["q_ref_source"] = "collection-era q_command_t (QP/planning intent)"

    validate_palm_dp_file(output_path)
    print(f"[SUCCESS] dual-track DP data saved to {output_path}", flush=True)

    q_rms = np.asarray([row["q_hand_vs_qref_rms_rad"] for row in per_episode])
    q_max = np.asarray([row["q_hand_vs_qref_max_rad"] for row in per_episode])
    report = {
        "inverted_file": str(inverted_path),
        "output_file": str(output_path),
        "obs_source": "original collection-era executed q_hand (D1 distribution)",
        "label_variant_a": "q_ref (16D absolute joint intent)",
        "label_variant_b": "tip_delta_tangent_palm (12D tangential intent)",
        "episodes": len(per_episode),
        "frames": total,
        "q_hand_vs_qref_rad": {
            "rms_median": float(np.median(q_rms)),
            "rms_p90": float(np.quantile(q_rms, 0.90)),
            "max_median": float(np.median(q_max)),
            "max_p90": float(np.quantile(q_max, 0.90)),
        },
        "per_episode": per_episode,
    }
    report_path = output_path.with_suffix(".dual_track_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DUAL] report -> {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inverted", type=Path, required=True,
                        help="Inverted H5 carrying the source_trajectory mapping")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--planner-waypoints", type=int, default=12)
    parser.add_argument("--planner-step-frames", type=int, default=6)
    parser.add_argument("--motion-feature-step-frames", type=int, default=5)
    args = parser.parse_args()
    output = args.output or args.inverted.with_name(
        f"{args.inverted.stem}_dual_track_dp.h5"
    )
    export(
        args.inverted,
        output,
        args.planner_waypoints,
        args.planner_step_frames,
        args.motion_feature_step_frames,
    )


if __name__ == "__main__":
    main()
