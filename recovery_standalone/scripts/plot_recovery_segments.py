"""Headless (no display, no mujoco) sanity plot for a ctx+recovery splice.

Reads a ctx rollout segment (mcc_closed_loop_observation_v1) and a recovery
segment (either the same schema, or the 'inverted' schema from
invert_trajectories.py), concatenates them at the given ctx cutoff frame, and
plots:

  1. per-fingertip contact mask over time (search/recovery should show a
     visible drop-then-recover pattern instead of already-stable contact)
  2. palm position (object frame) over time
  3. finite-differenced palm linear speed over time, with the splice marked

Saves one PNG, no GUI/X11/mujoco required.

Usage:
    python plot_recovery_segments.py \
      --ctx mcc_finger_compliance_control/data/closed_loop_rollouts/recovery_round1/ep040.h5 \
      --ctx-cutoff 859 \
      --recovery /tmp/recovery_fix_test_output/_intermediate/ep040_recovery_inverted.h5 \
      --output /tmp/ep040_recovery_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _first_present(f: h5py.File, names: list[str]) -> str | None:
    for name in names:
        if name in f:
            return name
    return None


def _load_segment(path: Path, cutoff: int | None) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        print(f"[{path.name}] keys: {sorted(f.keys())}")
        print(f"[{path.name}] attrs: {dict(f.attrs)}")

        contact_name = _first_present(
            f, ["fingertip_contact_mask", "fingertip_contact"]
        )
        pose_name = _first_present(f, ["palm_pose_object"])
        twist_name = _first_present(f, ["palm_twist_object"])

        out: dict[str, np.ndarray] = {}
        n = None
        if contact_name is not None:
            contact = np.asarray(f[contact_name])
            contact = contact.reshape(contact.shape[0], -1)[:, :4].astype(np.float64)
            out["contact"] = contact
            n = contact.shape[0]
        if pose_name is not None:
            pose = np.asarray(f[pose_name])
            pose = pose.reshape(pose.shape[0], -1)[:, :7].astype(np.float64)
            out["pose"] = pose
            n = pose.shape[0] if n is None else n
        if twist_name is not None:
            twist = np.asarray(f[twist_name])
            twist = twist.reshape(twist.shape[0], -1)[:, :6].astype(np.float64)
            out["twist"] = twist

    if n is None:
        raise ValueError(f"{path}: found neither contact nor pose fields")
    if cutoff is not None:
        for k in list(out.keys()):
            out[k] = out[k][:cutoff]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx", type=Path, required=True)
    parser.add_argument("--ctx-cutoff", type=int, default=None)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ctx = _load_segment(args.ctx, args.ctx_cutoff)
    rec = _load_segment(args.recovery, None)

    splice = next(iter(ctx.values())).shape[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    if "contact" in ctx and "contact" in rec:
        contact = np.concatenate([ctx["contact"], rec["contact"]], axis=0)
        for tip in range(contact.shape[1]):
            axes[0].step(
                np.arange(contact.shape[0]),
                contact[:, tip] + tip * 1.2,
                where="post",
                label=f"tip{tip}",
            )
        axes[0].set_ylabel("contact mask\n(per tip, offset)")
        axes[0].legend(loc="upper right", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "no contact field found in one of the files", ha="center")

    if "pose" in ctx and "pose" in rec:
        pose = np.concatenate([ctx["pose"], rec["pose"]], axis=0)
        for i, label in enumerate(["x", "y", "z"]):
            axes[1].plot(pose[:, i], label=f"palm_pos_{label}")
        axes[1].set_ylabel("palm position (m)\nobject frame")
        axes[1].legend(loc="upper right", fontsize=8)

        vel = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) / args.control_dt
        axes[2].plot(vel)
        axes[2].set_ylabel("|d(palm_pos)/dt|\n(m/s, finite-diff)")
        axes[2].set_xlabel("frame (concatenated ctx + recovery)")
    else:
        axes[1].text(0.5, 0.5, "no pose field found in one of the files", ha="center")

    for ax in axes:
        ax.axvline(splice, color="red", linestyle="--", linewidth=1.5)
        ax.text(
            splice, ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0,
            " splice", color="red", fontsize=8, va="top",
        )

    fig.suptitle(f"{args.ctx.name} (0:{splice}) + {args.recovery.name}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
