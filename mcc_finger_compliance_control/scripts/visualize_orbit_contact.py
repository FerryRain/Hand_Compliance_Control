"""Diagnose low contact fraction in orbit_palm collection H5 files.

Reads a trajectory H5 and renders a 3x2 panel figure:

  1. |fingertip force| vs orbit phase  -- which tip unloads, at which phase
  2. contact distance vs phase        -- gap (>=0) or penetration (<0) at unload
  3. geometry: tips / contact points / palm orbit projected on the plane
     perpendicular to the orbit axis, unloaded frames highlighted red
  4. phase + gate states over time    -- where the contact gate parks
  5. hand joint angles over time      -- confirm fingers stay static
  6. phase over time                  -- full triangle wave shape

Usage:
  python scripts/visualize_orbit_contact.py [path/to/traj.h5] [--out img.png]

The default contact threshold (0.05 N) matches the collection gate.
"""
import argparse
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIP_NAMES = ["index", "middle", "ring", "little"]
CONTACT_THRESHOLD_N = 0.05


def _proj_basis(axis):
    """Return orthonormal (u, v) perpendicular to axis, for projection."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=os.path.join(
        "mcc_finger_compliance_control", "data", "trajectories",
        "orbit_pinch_verify4.h5"))
    ap.add_argument("--out", default=None, help="output PNG (default: next to input)")
    ap.add_argument("--threshold", type=float, default=CONTACT_THRESHOLD_N)
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        n = f["time"].shape[0]
        force = f["fingertip_force_world"][:, 0]          # (T, 4, 3)
        cdist = f["fingertip_contact_dist"][:, 0]         # (T, 4)
        contact = f["fingertip_contact"][:, 0]            # (T, 4)
        cpos = f["fingertip_contact_pos_world"][:, 0]     # (T, 4, 3)
        tip = f["fingertip_pose_world"][:, 0, :, :3]      # (T, 4, 3)
        palm = f["palm_pose_world"][:, 0, :3]             # (T, 3)
        obj = f["object_pose_world"][:, 0]                # (T, 7)
        phase = f["orbit_phase_rad"][:, 0]
        moving = f["orbit_moving"][:, 0] > 0.5
        axis = f["orbit_axis_world"][:, 0]
        amp = f["orbit_amplitude_rad"][:, 0]
        speed = f["orbit_speed_target_rad_s"][:, 0]
        qhand = f["q_hand"][:, 0]                         # (T, 16)

    mag = np.linalg.norm(force, axis=-1)                  # (T, 4)
    loaded = mag >= args.threshold
    all4 = loaded.all(axis=-1)
    obj_c = obj[:, :3]

    # Projection plane perpendicular to the orbit axis (bottle long axis).
    u, v = _proj_basis(axis[0])
    def proj(p):
        p = np.asarray(p, dtype=float)
        return np.stack([p @ u, p @ v], axis=-1)

    tip_p = proj(tip)            # (T, 4, 2)
    palm_p = proj(palm)
    cpos_p = proj(cpos)
    obj_p = proj(obj_c)

    fig, axs = plt.subplots(3, 2, figsize=(16, 13))
    ph_deg = np.degrees(phase)

    # 1. |F| vs phase
    ax = axs[0, 0]
    for i, name in enumerate(TIP_NAMES):
        ax.plot(ph_deg, mag[:, i] * 1000, lw=0.8, label=f"{name} tip")
    ax.axhline(args.threshold * 1000, color="k", ls="--", lw=1,
               label=f"gate {args.threshold:.2f} N")
    ax.set_xlabel("orbit phase [deg]")
    ax.set_ylabel("|F| [mN]")
    ax.set_title("1) tip force vs phase  (which tip unloads, where)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. contact distance vs phase
    ax = axs[0, 1]
    for i, name in enumerate(TIP_NAMES):
        ax.plot(ph_deg, cdist[:, i] * 1000, lw=0.8, label=f"{name} tip")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xlabel("orbit phase [deg]")
    ax.set_ylabel("contact dist [mm]  (<0 penetration, >0 gap)")
    ax.set_title("2) sensor gap/penetration vs phase")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. geometry: tips + surface + palm orbit in projection plane
    ax = axs[1, 0]
    for i, name in enumerate(TIP_NAMES):
        ax.plot(tip_p[:, i, 0] * 1000, tip_p[:, i, 1] * 1000, lw=0.7,
                label=f"{name} tip path")
    for i, name in enumerate(TIP_NAMES):
        ok = contact[:, i] > 0.5
        ax.scatter(cpos_p[ok, i, 0] * 1000, cpos_p[ok, i, 1] * 1000,
                   s=3, alpha=0.5, marker=".", label=f"{name} contact pt" if i == 0 else None)
    ax.plot(palm_p[:, 0] * 1000, palm_p[:, 1] * 1000, "k-", lw=1,
            label="palm orbit")
    unl = ~all4
    if unl.any():
        ax.scatter(tip_p[unl, :, 0] * 1000, tip_p[unl, :, 1] * 1000,
                   s=10, c="red", alpha=0.4, label="unloaded frames (tips)")
    ax.plot(obj_p[:, 0] * 1000, obj_p[:, 1] * 1000, "k+", ms=4,
            label="object center")
    ax.set_xlabel("proj along u [mm]")
    ax.set_ylabel("proj along v [mm]")
    ax.set_aspect("equal")
    ax.set_title("3) geometry in orbit-axis-normal plane (red = unloaded)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. phase + gate states over time
    ax = axs[1, 1]
    t = np.arange(n)
    ax.plot(t, ph_deg, lw=1, label="phase [deg]")
    ax.plot(t, moving * np.nanmax(np.abs(ph_deg)) * 0.9, "b", lw=0.8, alpha=0.6,
            label="gate moving")
    ax.plot(t, all4 * np.nanmax(np.abs(ph_deg)) * 0.9, "g", lw=0.8, alpha=0.6,
            label="all4 loaded")
    ax.set_xlabel("step")
    ax.set_ylabel("phase [deg] / state")
    ax.set_title("4) phase & contact gate over time")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # 5. hand joints over time (should stay static)
    ax = axs[2, 0]
    ax.plot(t, np.degrees(qhand), lw=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("q_hand [deg]")
    ax.set_title("5) hand joint angles over time")
    ax.grid(alpha=0.3)

    # 6. phase vs time (triangle wave)
    ax = axs[2, 1]
    ax.plot(t, ph_deg, lw=1)
    ax.axhline(np.degrees(amp[0]), color="r", ls="--", lw=0.8, label="+amp")
    ax.axhline(-np.degrees(amp[0]), color="r", ls="--", lw=0.8, label="-amp")
    ax.set_xlabel("step")
    ax.set_ylabel("phase [deg]")
    ax.set_title("6) orbit phase (triangle wave)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{os.path.basename(args.path)}  |  "
                 f"amp={np.degrees(amp[0]):.1f} deg  speed={speed[0]:.3f} rad/s  |  "
                 f"all4 loaded {all4.mean():.1%}  ({all4.sum()}/{n} frames)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or os.path.splitext(args.path)[0] + "_diag.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"frames={n}  all4_loaded={all4.mean():.3f}")
    print(f"per-tip unloaded fractions: "
          + "  ".join(f"{name}={1 - loaded[:, i].mean():.3f}"
                      for i, name in enumerate(TIP_NAMES)))
    for i, name in enumerate(TIP_NAMES):
        un = ~loaded[:, i]
        if un.any():
            ph_un = ph_deg[un]
            print(f"  {name} tip unloaded phase range: "
                  f"[{ph_un.min():.1f}, {ph_un.max():.1f}] deg over {un.sum()} frames")


if __name__ == "__main__":
    main()
