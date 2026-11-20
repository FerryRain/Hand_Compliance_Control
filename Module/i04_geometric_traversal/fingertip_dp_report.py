"""Profile the archived I04-c controller and compare the two measured rollouts."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .fingertip_dp_compare import CONTROL_STEPS, TRACK_PHASES
from .fingertip_hand import DT
from .fingertip_plan import JointPatchPlanner
from .fingertip_replay import chunks, load_case
from .fingertip_traversal import ForceFeedback, STATE, latency_ms, surface_graph, write_json


def read_json(case, name):
    return json.loads((case / name).read_bytes())


def tape(case, *names):
    trace = list(chunks(case))
    return {name: np.concatenate([item[name] for item in trace]) for name in names}


def mocap_error(data):
    planned = data["palm_command"]
    measured = data["qpos"][:, :7]
    error = np.linalg.norm(planned[:, :3] - measured[:, :3], axis=1) * 1000
    cosine = np.clip(abs(np.einsum("ni,ni->n", planned[:, 3:], measured[:, 3:])), 0., 1.)
    return np.percentile(np.c_[error, np.degrees(2 * np.arccos(cosine))], 95, axis=0).tolist()


def profile_reference(case, reference):
    """Time controller work on recorded states, with FK restored outside the timer."""
    hand = load_case(case)
    model = hand.model
    data = mujoco.MjData(model)
    with np.load(case / "initial.npz", allow_pickle=False) as initial:
        mujoco.mj_setState(model, data, initial["integration_state"], STATE)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_normals = hand.measure(data).normals
    patches = read_json(case, "planned_patches.json")
    start_steps = [patch["start_step"] for patch in patches]
    feedback = ForceFeedback()
    samples = np.arange(0, len(reference["qpos"]), 5 * CONTROL_STEPS)
    latencies = np.empty(len(samples))
    for idx, step in enumerate(samples):
        if step:
            state = step - 1
            data.qpos[:] = reference["qpos"][state]
            palm = reference["palm_command"][state]
            data.mocap_pos[hand.driver], data.mocap_quat[hand.driver] = palm[:3], palm[3:]
            data.ctrl[:] = reference["finger_command"][state]
            previous = reference["qpos"][state - 1] if state else initial_qpos
            mujoco.mj_differentiatePos(model, data.qvel, DT, previous, data.qpos)
            mujoco.mj_forward(model, data)
        loaded = hand.measure(data)
        patch = int(np.searchsorted(start_steps, step, side="right") - 1)
        normals = np.asarray(patches[patch]["surface_normals"]) if patch >= 0 else initial_normals
        if step:
            # The original load is archived at every step. FK provides the
            # contact geometry for this standalone timing sample.
            missing_direction = np.linalg.norm(loaded.normals, axis=1) < .99
            loaded.normals[missing_direction] = normals[missing_direction]
            loaded.forces = reference["normal_force_n"][step - 1]
        end = min(step + CONTROL_STEPS, len(reference["qpos"]))
        commands = reference["finger_command"][step:end]
        phases = reference["phase"][step:end]
        start = perf_counter()
        for control, phase in zip(commands, phases):
            feedback.command(hand, data, control, normals, loaded, phase in TRACK_PHASES)
        latencies[idx] = (perf_counter() - start) * 1000
    planner = JointPatchPlanner(hand, surface_graph())
    fit_ms = []
    successful_fits = 0
    for patch in patches:
        orient = Rotation.from_quat(np.asarray(patch["wrist_wxyz"])[[1, 2, 3, 0]])
        start = perf_counter()
        result = planner.fit(patch["target_site"], patch["anchor"], orient)
        fit_ms.append((perf_counter() - start) * 1000)
        successful_fits += result is not None
    return dict(mcc_per_control_interval=latency_ms(latencies),
                one_candidate_wrist_finger_ik=latency_ms(fit_ms),
                candidate_fits_accepted=successful_fits,
                mcc_sample_every_s=DT * 5 * CONTROL_STEPS,
                mcc_over_10ms_fraction=float(np.mean(latencies > 10.)),
                mcc_total_s_estimate=float(latencies.mean() * len(reference["qpos"]) / CONTROL_STEPS / 1000))


def branch_metrics(summary, audit):
    quorum = summary["simultaneous_qualified_steps_0_to_4"]
    states = summary["states"]
    return dict(area_fraction=summary["area_fraction"],
        covered_vertices=summary["covered_vertices"], covered_goals=summary["covered_goals"],
        any_eligible_contact_rate=1. - quorum[0] / states, four_eligible_contact_rate=quorum[4] / states,
        incidental_contact_steps=summary["incidental_contact_steps"],
        incidental_episodes_by_part=summary["incidental_episodes_by_part"],
        illegal_grasp_rate=summary["illegal_grasp_states"] / states,
        first_illegal_grasp=summary.get("first_illegal_grasp"),
        first_sustained_overload_stop=summary.get("first_sustained_overload_stop"),
        peak_regional_load_n=max(summary["peak_load_n_fingers_then_palm"]),
        force_excluded_rate=audit["force_excluded_states"] / states,
        longest_over_40n_s=audit["longest_over_40n_s"])


def plot_comparison(case, dp_case, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    basemap = "#14786f"
    learned = "#a6502a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10,
                         "axes.labelcolor": "#202733", "text.color": "#202733",
                         "axes.titlepad": 11, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.78), gridspec_kw={"width_ratios": [1.35, 1.]},
                             layout="constrained")
    for path, label, color in ((case, "Geometric planning + MCC", basemap),
                               (dp_case, "Finger DP + MCC", learned)):
        preview_path = path / "native_preview.npz"
        if sha256(preview_path.read_bytes()).hexdigest() != read_json(path, "native_audit.json")["preview_sha256"]:
            raise ValueError(f"Unverified preview in {path}")
        with np.load(preview_path, allow_pickle=False) as preview:
            axes[0].step(preview["step"] * DT, preview["area_fraction"], where="post",
                         label=label, color=color, linewidth=2.)
    axes[0].set(xlabel="Simulated time (s)", ylabel="Qualified surface coverage",
                ylim=(0., 1.04), xlim=(0., result["wrist_time_s"]), title="Coverage along the same wrist commands")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.))
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(axis="y", color="#e3e7e7")
    stats = [result["mcc_profile"]["mcc_per_control_interval"],
             result["dp_timing"]["all_control_per_interval"],
             result["dp_timing"]["policy_inference_per_query"]]
    labels = ["MCC\n10 ms interval", "DP + MCC\n10 ms interval", "DP inference\n100 ms query"]
    for i, (entry, color) in enumerate(zip(stats, (basemap, learned, learned))):
        axes[1].plot([i, i], [entry["p50_ms"], entry["p95_ms"]], color=color, linewidth=2)
        axes[1].scatter(i, entry["p50_ms"], color=color, s=34, zorder=3,
                        label="Median" if i == 0 else None)
        axes[1].scatter(i, entry["p95_ms"], color=color, marker="_", s=160, zorder=3,
                        label="95th percentile" if i == 0 else None)
    axes[1].axhline(10., color="#838b8e", ls="--", lw=1., label="10 ms control period")
    axes[1].set(yscale="log", ylabel="Elapsed time (ms)", title="Control latency",
                xticks=np.arange(len(labels)), xticklabels=labels, xlim=(-.4, 2.4))
    axes[1].grid(axis="y", color="#e3e7e7")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def run_report(case, dp_case):
    base_audit = read_json(case, "native_audit.json")
    dp_audit = read_json(dp_case, "native_audit.json")
    if not (base_audit["native_replay_matched"] and dp_audit["native_replay_matched"]):
        raise ValueError("Both cases must have passed native command replay")
    baseline = read_json(case, "summary.json")
    learned = read_json(dp_case, "summary.json")
    if not learned["full_wrist"]:
        raise ValueError("The comparison requires the full archived trajectory")
    base = tape(case, "qpos", "palm_command", "finger_command", "normal_force_n", "phase")
    dp = tape(dp_case, "qpos", "palm_command", "phase")
    np.testing.assert_array_equal(base["phase"], dp["phase"])
    np.testing.assert_array_equal(base["palm_command"], dp["palm_command"])
    digest = sha256(np.ascontiguousarray(base["palm_command"], dtype="<f8").tobytes()).hexdigest()
    if digest != learned["wrist_sha256"]:
        raise ValueError("DP wrist digest mismatch")
    if (case / "initial.npz").read_bytes() != (dp_case / "initial.npz").read_bytes():
        raise ValueError("Initial physical state differs between the two branches")
    mcc_manifest = read_json(case, "manifest.json")
    dp_manifest = read_json(dp_case, "manifest.json")
    if mcc_manifest["model_sha256"] != dp_manifest["model_sha256"]:
        raise ValueError("The two branches must use the same native model")
    profile = profile_reference(case, base)
    info = dp_manifest["policy"]
    with np.load(dp_case / "timing.npz", allow_pickle=False) as timings:
        control_ms = timings["shared_mcc_ms"] + timings["observation_ms"] + timings["policy_rpc_ms"]
    dp_over_10ms_fraction = float(np.mean(control_ms > 10.))
    report = dict(physics_dt_s=DT, wrist_steps=len(base["palm_command"]),
                  wrist_time_s=len(base["palm_command"]) * DT, wrist_sha256=digest,
                  initial_physical_state_equal=True, wrist_commands_equal=True,
                  baseline_case=case.name, dp_case=dp_case.name, policy=info,
                  mcc=branch_metrics(baseline, base_audit), dp=branch_metrics(learned, dp_audit),
                  dp_force_only_coverage=learned["final_force_only_coverage"],
                  mcc_p95_wrist_mm_deg=mocap_error(base), dp_p95_wrist_mm_deg=mocap_error(dp),
                  mcc_profile=profile, dp_timing=learned["timing"], dp_over_10ms_fraction=dp_over_10ms_fraction,
                  dp_rollout_wall_s=learned["native_wall_s"])
    write_json(dp_case / "comparison.json", report)
    plot_comparison(case, dp_case, report, dp_case / "comparison.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--dp-case", type=Path, required=True)
    args = parser.parse_args()
    result = run_report(args.case, args.dp_case)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
