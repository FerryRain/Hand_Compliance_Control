"""Plot fixed-reference LEAP force/contact comparisons, including failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py


def motion_statistics(folder, row):
    """Use zero-based physics ticks, including the first motion tick.

    The observer's physics trace records the counter before env.step returns;
    trace.npz records it afterwards. Their task masks therefore differ by one.
    Keep the original execution report intact and save corrected offline
    motion statistics alongside the comparison.
    """
    summary = json.loads((folder/"summary.json").read_text())
    start = summary.get("actual_motion_start_step")
    frames = int(row["requested_frames"])
    if start is None:
        return None
    with np.load(folder/"physics_trace.npz") as archive:
        active = (archive["step"] >= start) & (archive["step"] < start+frames)
        parts = np.maximum(archive["part_net"][active, 0],
                           archive["part_single"][active, 0])
        peak = np.maximum(archive["peak_force"][active, 0], parts.max(-1))
        result = dict(physics_samples=int(active.sum()), expected_samples=frames*10,
                      first_physics_tick_included=True,
                      peak_all_parts_n=float(peak.max()),
                      over100_ms=int((peak > 100.).sum()),
                      part_peak_force_n=dict(zip(("true_fingertip", "connector", "other_link", "palm"),
                                                parts.max(0).tolist())))
        if row.get("status") == "completed" and result["physics_samples"] != frames*10:
            raise RuntimeError(f"Incomplete physics trace: {folder}: {result}")
        if "actuator_force" in archive:
            tau = archive["actuator_force"][active, 0, -16:]
            result["max_motor_torque_nm"] = float(abs(tau).max())
            result["per_joint_rms_nm"] = np.sqrt((tau*tau).mean(0)).tolist()
    return result


def plot_ring_execution(directory):
    folders = [directory/f"candidate_0062_{v}" for v in ("continuous_interpolated", "peak_interpolated")]
    if not all((folder/"collected.h5").exists() for folder in folders):
        return
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex="col")
    for column, folder in enumerate(folders):
        summary = json.loads((folder/"summary.json").read_text())
        start = summary.get("actual_motion_start_step")
        if start is None:
            plt.close(fig)
            return
        with h5py.File(folder/"collected.h5") as handle:
            step = handle["episode_step"][:, 0]
            q = handle["q_hand"][:, 0, 11]
            target = handle["q_cmd"][:, 0, 11]
            phase = handle["fullhand_contact_phase"][:, 0, 2]
            contact = handle["true_fingertip_contact_valid"][:, 0, 2]
        axes[0, column].plot((step-start+1)*.01, q, label="actual ring distal")
        axes[0, column].plot((step-start)*.01, target, label="held actuator target", alpha=.7)
        axes[0, column].set_title(folder.name)
        axes[1, column].step((step-start)*.01, phase, label="contact state", where="post")
        axes[1, column].step((step-start+1)*.01, contact, label="ring strict contact", where="post", alpha=.6)
        with np.load(folder/"physics_trace.npz") as archive:
            active = archive["step"] >= start
            t = (archive["step"][active]-start+(archive["substep"][active]-1)/10)*.01
            force = np.linalg.norm(archive["pad_force"][active, 0, 2], axis=-1)
            tau = archive["actuator_force"][active, 0, 14:18]
        axes[2, column].plot(t, force, label="ring fingertip force,1ms")
        axes[2, column].axhline(100., color="red", ls="--", lw=1)
        axes[3, column].plot(t, np.max(abs(tau), axis=-1), label="max ring motor torque")
        axes[3, column].axhline(.186 if column == 0 else .8, color="red", ls="--", lw=1)
        axes[3, column].set_xlabel("task time(s), settling excluded")
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            ax.set_xlim(0, len(step)*.01)
    for ax, label in zip(axes[:, 0], ("angle(rad)", "phase / contact", "force(N)", "motor torque(Nm)")):
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(directory/"ring_control_execution.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.directory/"analysis.json").read_text())
    rows = report["rows"]
    for row in rows:
        if row.get("baseline_reused") and row.get("source_output"):
            row["output"] = row["source_output"]
        folder = Path(row["output"])
        corrected = (motion_statistics(folder, row) if row.get("status") == "completed"
                     and (folder/"physics_trace.npz").exists() else None)
        row["offline_motion_statistics"] = corrected
        row["peak_all_parts_n"] = (corrected["peak_all_parts_n"] if corrected is not None else
            max([row.get("peak_stop_sensor_force_n", 0.)]
                + list((row.get("part_peak_force_n") or {}).values())))
    # The stop sensor covers fingertip/connector. Include other finger links
    # and palm, measured independently at every physical substep, in this plot.
    for variant, group in report["aggregates"].items():
        group["max_force_n"] = max([r["peak_all_parts_n"] for r in rows if r["variant"] == variant]+[0.])
        group["over100_cases"] = sum(
            (r.get("offline_motion_statistics") or r.get("force_events") or {}).get(
                "over100_ms", (r.get("force_events") or {}).get("total_overforce_ms", 0)) > 0
            for r in rows if r["variant"] == variant)
    variants = ["legacy", "continuous_interpolated", "peak_interpolated"]
    ids = sorted({row["candidate_id"] for row in rows})
    lookup = {(r["candidate_id"], r["variant"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    panels = [("strict_healthy_four_rate_full", "Strict true-tip contact (%)", "YlGn", 100),
              ("found_four_rate_full", "Found contact (%)", "YlGn", 100),
              ("peak_all_parts_n", "Peak all-hand-part force (N)", "YlOrRd", 1)]
    for ax, (key, title, color, scale) in zip(axes, panels):
        values = np.array([[lookup.get((cid, v), {}).get(key, 0.)*scale for v in variants] for cid in ids])
        shown = values if scale == 100 else np.minimum(values, 150.)
        image = ax.imshow(shown, cmap=color, aspect="auto", vmin=0, vmax=100 if scale == 100 else 150.)
        for i, cid in enumerate(ids):
            for j, variant in enumerate(variants):
                row = lookup.get((cid, variant), {})
                text = f"{values[i,j]:.1f}" if row.get("status") == "completed" else "init/error"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), ["Legacy", "0.186 Nm", "0.8 Nm"], rotation=20)
        ax.set_yticks(range(len(ids)), [f"candidate {i}" for i in ids])
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=.7)
    fig.suptitle("Mustard full-reference evaluation; new profiles use 1ms object interpolation\n"
                 "Legacy results reused; settling states can differ; 0.8Nm is a peak-capability diagnostic")
    fig.tight_layout()
    fig.savefig(args.directory/"contact_force_comparison.png", dpi=160)
    plt.close(fig)
    comparisons = []
    for variant in variants[1:]:
        pairs = []
        for cid in ids:
            baseline = lookup.get((cid, "legacy"), {})
            candidate = lookup.get((cid, variant), {})
            if not candidate:
                continue
            pairs.append(dict(candidate_id=cid,
                initial_state_comparison="Different actuator gains/limits during settling; not identical-state branches",
                legacy_strict=baseline.get("strict_healthy_four_rate_full", 0.),
                strict=candidate.get("strict_healthy_four_rate_full", 0.),
                delta_strict_pp=100*(candidate.get("strict_healthy_four_rate_full", 0.)-baseline.get("strict_healthy_four_rate_full", 0.)),
                delta_found_pp=100*(candidate.get("found_four_rate_full", 0.)-baseline.get("found_four_rate_full", 0.)),
                stable_start=candidate.get("stable_start", False),
                full_execution=candidate.get("status") == "completed",
                torque=candidate.get("torque"),
                offline_motion_statistics=candidate.get("offline_motion_statistics"),
                over100_ms=(candidate.get("offline_motion_statistics") or {}).get("over100_ms",
                    (candidate.get("force_events") or {}).get("total_overforce_ms"))))
        healthy = [p for p in pairs if p["legacy_strict"] >= .95]
        difficult = [p for p in pairs if p["legacy_strict"] < .95]
        comparisons.append(dict(variant=variant, pairs=pairs,
            healthy_reference_count=len(healthy),
            healthy_mean_delta_strict_pp=float(np.mean([p["delta_strict_pp"] for p in healthy])) if healthy else None,
            difficult_reference_count=len(difficult),
            difficult_mean_delta_strict_pp=float(np.mean([p["delta_strict_pp"] for p in difficult])) if difficult else None,
            mean_delta_strict_pp=float(np.mean([p["delta_strict_pp"] for p in pairs])) if pairs else None))
    (args.directory/"comparison.json").write_text(json.dumps(dict(aggregates=report["aggregates"],
        comparisons=comparisons,
        physics_time_indexing="Physics ticks are zero-based: start <= step < start+frames; all10 substeps including the first task tick are included",
        limitations="One run per reference/configuration; no statistical or hardware certification claim"), indent=2)+"\n")
    print(json.dumps(report["aggregates"], indent=2))
    plot_ring_execution(args.directory)


if __name__ == "__main__":
    main()
