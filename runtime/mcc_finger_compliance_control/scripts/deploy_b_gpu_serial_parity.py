"""Batched GPU deployment of the B checkpoint, ported from the serial live_dp path.

This is a port, not a reimplementation.  Every block below names the serial
source range it mirrors, and the acceptance test is the serial report itself:

    mcc_finger_compliance_control/scripts/deploy_dp_inverse.py \
      --file  .../mustard_fullsurface_v2_221_inverted.h5 \
      --model .../mustard_singlepoint_qhand_cmd_B_5k_to_25k/best.pt \
      --episode-id 9 --mode live_dp --nominal-policy dp --seed 42 \
      --palm-source teacher --recenter-palm-substeps --chunk-execution \
      --dp-history-q-source live --execution-layer fullhand_mcc \
      --mcc-preset collection_matched_sensor --max-safe-force-n 100 \
      --report <serial.csv>

Run this script with the same episode/seed and `--report <batched.csv>`, then
diff the two reports frame by frame (`q_dp_J` is the command handed to MCC).

What runs on the GPU here is the whole control chain; what stays on the CPU is
one Python loop dispatching a single `torch` call per control frame.

    serial                                    this file
    ---------------------------------------------------------------------
    2220  __call__ entry, settling flag         main loop, `settling`
    2241  bootstrap smoothstep                  `_bootstrap_nominal`
    1698  _set_palm (pose + twist)              `_palm_root_target`
    1728  step_with_palm_recenter               `_recenter_patch`
    2496  closure/sensor normal blend           BatchedHybridMCCDirection
    2547  track: recovery state machine         BatchedBControlChain._orchestrate
    2874  plan_q = clamp(base + recovery)       same, inside `_orchestrate`
    3076  raw_action = (desired - q)/0.08       same, in `step`
    3457  settle bookkeeping / history clear    `begin_settle` + reset
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from batched_b_control_chain import BatchedBControlChain, ZeroResidualPolicy  # noqa: E402
from deploy_dp_inverse import (                                   # noqa: E402
    DPRuntime, attach_commanded_planner_features, load_episode)
from object_catalog import load_object_config                     # noqa: E402
from replay_inverted import replay_env_cfg                        # noqa: E402
from surface_mcc_finger import FullHandMCCFingerConfig            # noqa: E402
import hashlib
from typing import Any, Sequence
import h5py




SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


SOURCE = "mcc_finger_compliance_control/data/inverted/mustard_fullsurface_v2_221_inverted.h5"
MODEL = "mcc_finger_compliance_control/data/models/mustard_singlepoint_qhand_cmd_B_5k_to_25k/best.pt"
ACTION_SCALE = 0.08          # deploy_dp_inverse.ACTION_SCALE
CONTACT_THRESHOLD = 0.05     # --contact-threshold default



def mcc_config() -> FullHandMCCFingerConfig:
    """`--mcc-preset collection_matched_sensor`, deploy_dp_inverse.py:4276-4326."""
    return FullHandMCCFingerConfig(
        desired_force=10.0,
        desired_force_per_finger=(10.0, 10.0, 10.0, 12.0),
        max_normal_offset=0.002,              # --mcc-max-normal-offset-mm 2.0
        thumb_max_inward_offset=0.002,
        posture_cost=0.15,
        nominal_surface_preload=0.002,
        flexion_synergy_gain=0.0,
        flexion_synergy_hard_gain=0.0,
        flexion_synergy_max_step=0.025,
        normal_synergy_control=False,
        normal_synergy_max_step=0.025,
        nominal_joint_correction_limit=0.04,   # --mcc-local-joint-correction-limit-rad
        thumb_nominal_joint_correction_limit=0.04,
        action_rate_limit=0.02,
        command_ema_alpha=0.65,
        overforce_hard_ratio=4.0,
        thumb_overforce_hard_ratio=4.0,
        overforce_max_offset=0.002,
        overforce_recontact_hold_frames=20,
        force_servo_integral_gain=0.003,
        thumb_force_servo_hard_step=0.00010,
        thumb_force_servo_search_step=0.00022,
        enable_loss_state_machine=True,
        transient_loss_frames=6,
        recovery_contact_confirm_frames=4,
        transient_search_step=0.00020,
        transient_release_step=0.00025,
        natural_flexion_floor=-0.30,
    )


def orchestration_config() -> dict:
    """The MCCPrecontactConfig values the orchestrator needs (deploy_dp_inverse.py:187-254).

    These are not `FullHandMCCFingerConfig` fields; the serial loop reads them
    off `mcc_precontact_config` directly, so they are parked on the chain here
    instead of being guessed from unrelated defaults.
    """
    return dict(
        runtime_loss_frames=2,                # --mcc-runtime-loss-frames
        recovery_confirm_frames=4,            # --mcc-recovery-confirm-frames
        runtime_recovery_limit_rad=0.04,      # --mcc-runtime-recovery-limit-rad
        recovery_offset_decay=0.999,
        recovery_decay_force_ratio=1.5,
        search_cartesian_step_m=0.00015,      # cartesian_step_m
        search_joint_step_rad=0.02,           # joint_step_rad
        settle_force_ratio=0.80,
        force_threshold=0.10,
    )


def run(batch: int, episode: int, frames: int, device: str, seed: int,
        report_csv: Path | None = None, dp_trace_h5: Path | None = None,
        max_safe_force_n: float = 100.0, settle_steps: int = 100,
        residual_mode: str = "off", per_env_safety: bool = False,
        episode_data: dict | None = None, chain_factory: Any = None,
        contact_observer: Any = None,
        initial_joint_position: Any = None,
        initial_joint_velocity: Any = None,
        initial_joint_target: Any = None,
        stop_when_all_inactive: bool = False,
        palm_driver: Any = None) -> dict:
    from mjlab.envs import ManagerBasedRlEnv

    torch_device = torch.device(device)
    model = Path(MODEL)

    # --- data + policy (deploy_dp_inverse.py:4333+) -------------------------
    runtime = DPRuntime(model, torch_device, None, seed)
    if episode_data is None:
        data = load_episode(Path(SOURCE), episode, include_teacher_tactile=False)
        attach_commanded_planner_features(data, runtime, episode)
    else:
        # Calibration may supply different, correctly paired paths/q0 per lane.
        # Shapes are [T,B,...], command [T,W,B,6]; ordinary replay uses [T,...].
        data = episode_data
        for field, width in (("q_hand", 16), ("palm_pose_object", 7), ("palm_twist_object", 6)):
            if np.asarray(data[field]).shape[1:] != (batch, width):
                raise ValueError(f"batched episode field {field} has invalid shape")
        if np.asarray(data["planner_palm_delta_pose_palm_commanded"]).shape[2:] != (batch, 6):
            raise ValueError("batched preissued palm command must have shape [T,W,B,6]")
    if "planner_palm_delta_pose_palm_commanded" not in data:
        raise ValueError("frozen preissued palm command missing")
    n = min(frames, len(data["q_hand"]))

    cfg = replay_env_cfg(num_envs=batch, hand_stiffness=35.0, hand_damping=2.5,
                         hand_effort_limit=35.0,
                         object_config=load_object_config("ycb_mustard"))
    if palm_driver is not None:
        palm_driver.configure(cfg)
    if getattr(chain_factory, 'needs_stabilizer_diagnostics', False):
        from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import _tip_sensor_cfgs
        diagnostic = next(s for s in _tip_sensor_cfgs(include_arm_guard=False)
                          if s.name == 'incidental_hand_object_contact_force')
        cfg.scene.sensors = (*cfg.scene.sensors, diagnostic)
    if contact_observer is not None:
        contact_observer.configure(cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    chain = None
    try:
        env.reset()
        robot = env.scene["robot"]
        if palm_driver is not None:
            palm_driver.bind(env)

        def write_palm_root(state):
            if palm_driver is None:
                robot.write_root_state_to_sim(state)
            else:
                palm_driver.write_root_state(state)
        q0 = torch.as_tensor(data["q_hand"][0] if initial_joint_position is None else initial_joint_position,
                             device=device, dtype=torch.float32).expand(batch, -1).clone()
        # Matches the serial initialisation exactly: only qpos is written, so
        # the first physics step fires the same ~976 N transient the serial
        # settle trace shows at its step 0, and the 100 warm-up frames absorb
        # it.  Pre-setting the actuator target would remove that transient and
        # put this chain on a different trajectory from frame 0.
        velocity0=(torch.zeros_like(q0) if initial_joint_velocity is None else
                   torch.as_tensor(initial_joint_velocity,device=device,dtype=torch.float32).expand_as(q0).clone())
        if palm_driver is not None and hasattr(palm_driver, 'initialize_hand'):
            palm_driver.initialize_hand(q0, velocity0, initial_joint_target)
        else:
            robot.write_joint_state_to_sim(q0, velocity0)
        if initial_joint_target is not None and not hasattr(palm_driver, 'initialize_hand'):
            robot.set_joint_position_target(torch.as_tensor(initial_joint_target,device=device,dtype=torch.float32).expand_as(q0))
            robot.write_data_to_sim()

        palm_path = torch.as_tensor(data["palm_pose_object"][:n], device=device)
        twist_path = torch.as_tensor(data["palm_twist_object"][:n], device=device)
        command_path = torch.as_tensor(
            data["planner_palm_delta_pose_palm_commanded"][:n, 0], device=device)

        # --- _set_palm / step_with_palm_recenter (serial 1698-1740, 1338) ----
        origins = env.scene.env_origins
        palm_root = [None]

        def recentered_step() -> None:
            env.sim.step_orig()
            if hasattr(chain, 'record_base_substep'):
                chain.record_base_substep(env, palm_root[0])
            if hasattr(chain, 'record_substep_force'):
                env.sim.sense()
                read_sensors()
                chain.record_substep_force()
            if palm_root[0] is not None and (palm_driver is None or palm_driver.recenter_substeps):
                write_palm_root(palm_root[0])

        env.sim.step_orig = env.sim.step
        env.sim.step = recentered_step

        sensors = [env.scene[f"{name}_contact"]
                   for name in ("if_tip", "mf_tip", "rf_tip", "th_tip")]

        def read_sensors():
            """Refresh each ContactSensor and take its first live slot (serial 1768+)."""
            for sensor in sensors:
                sensor.update(0.0)
            entries = [sensor.data for sensor in sensors]
            # Optional read-only evaluation hook. Preserve the controller's
            # first-live-slot point and aggregate force exactly as before.
            if contact_observer is not None:
                contact_observer.read(env, entries)
            live = [entry.found > 0 for entry in entries]
            found = torch.stack([m.any(dim=1) for m in live], dim=1)
            force = torch.stack([(e.force * m[..., None]).sum(dim=1)
                                 for e, m in zip(entries, live)], dim=1)
            slot = [m.long().argmax(dim=1) for m in live]
            row = torch.arange(batch, device=device)
            points = torch.stack([e.pos[row, i] for e, i in zip(entries, slot)], dim=1)
            inward = torch.stack([e.normal[row, i] for e, i in zip(entries, slot)], dim=1)
            return found, force, points, inward

        chain = (chain_factory or BatchedBControlChain)(model, batch, torch_device,
                                     seed=seed, mcc_config=mcc_config(),
                                     max_safe_force_n=(0.0 if per_env_safety else max_safe_force_n))
        for key, value in orchestration_config().items():
            setattr(chain, key, value)
        if runtime.state_schema is not None:
            # --dp-tactile-normal-source defaults to source_mesh_oracle, so the
            # serial run corrects every contact normal against the undecomposed
            # source mesh before DP and MCC see it.
            chain.mesh_oracle = TorchMeshNormalOracle(
                load_object_config("ycb_mustard"), scale=1.0, device=device)
            print("[MESH-ORACLE] source-mesh normal correction enabled", flush=True)
        chain.prime(q0, repeats=30)
        chain.begin_settle(q0, settle_steps)
        post_step_safety=bool(getattr(chain,'post_step_safety',False))
        if residual_mode == "zero":
            chain.residual_policy = ZeroResidualPolicy()
        elif residual_mode != "off":
            raise ValueError(f"unsupported residual mode: {residual_mode}")

        trace = None
        if dp_trace_h5 is not None or report_csv is not None:
            trace = BatchedDPTrace(
                model, batch, torch_device, max_frames=n + settle_steps,
                control_dt=float(getattr(runtime.config, "control_dt", 0.01)),
                scheduler_config=chain.scheduler.config, runtime=chain.runtime,
                note=f"deploy_b_gpu_serial_parity episode={episode} batch={batch}")
            chain.trace = trace

        # --- main loop (serial 2220-3480): ONE loop, settling is a branch ----
        bootstrap_end = chain.bootstrap_end
        max_force_n = 0.0
        max_pre_step_force_n = 0.0
        found_sum = 0
        stage_found = torch.zeros((3, 4), device=device)
        stage_count = [0, 0, 0]
        stop_frame = None
        reason = None
        completed = 0
        active = torch.ones(batch, device=device, dtype=torch.bool)
        stopped_at = torch.full((batch,), -1, device=device, dtype=torch.long)
        initial_rejected = torch.zeros(batch,device=device,dtype=torch.bool)
        hold_pose = torch.zeros((batch, 7), device=device)
        torch.cuda.synchronize(torch_device)
        started = time.perf_counter()

        for step_index in range(n + settle_steps):
            settling = step_index < settle_steps
            frame = 0 if settling else step_index - settle_steps
            newly_stopped = torch.zeros(batch, device=device, dtype=torch.bool)
            # warm-up is recorded under negative frame numbers: the serial
            # report has no such rows, but its force trace is the thing the
            # first real frame inherits
            if trace is not None and settling:
                chain.trace = trace
                chain.trace.frame_offset = -(settle_steps - step_index)

            pose = palm_path[frame].expand(batch, -1).clone()
            pose[:, :3] += origins
            if per_env_safety and not settling:
                pose = torch.where(active[:, None], pose, hold_pose)
            # _set_palm: the prescribed root twist is zeroed only while settling
            twist = (torch.zeros((batch, 6), device=device) if settling
                     else twist_path[frame].expand(batch, -1).clone())
            if per_env_safety and not settling:
                twist = torch.where(active[:, None], twist, torch.zeros_like(twist))
            palm_root[0] = torch.cat((pose, twist), dim=1)
            write_palm_root(palm_root[0])
            if env.sim.model.nmocap:
                # Target mocap index is not necessarily0 once a supported
                # robot also has a mocap base. Never reset the palm as target.
                if palm_driver is None:
                    env.sim.data.mocap_pos[:, 0, :] = 0.0
                    env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
                        (1.0, 0.0, 0.0, 0.0), device=device)
                else:
                    target_mocap_id = int(env.scene['target'].data.indexing.mocap_id)
                    env.sim.data.mocap_pos[:, target_mocap_id, :] = 0.0
                    env.sim.data.mocap_quat[:, target_mocap_id, :] = torch.tensor(
                        (1.0, 0.0, 0.0, 0.0), device=device)
            env.sim.forward()
            # MJWarp refreshes sensor buffers in sense(); forward() alone can
            # leave the previous frame's readings in place, which shows up as a
            # spurious 976 N (the settle step-0 impact) reported on frame 0.
            # Sensor values themselves are identical either way -- verified
            # against the serial report -- so this only fixes staleness.
            env.sim.sense()
            found, force, points, inward = read_sensors()
            peak = float(torch.linalg.vector_norm(force, dim=-1).max().item())
            if contact_observer is not None and contact_observer.safety_peak_force_n is not None:
                peak = max(peak, float(contact_observer.safety_peak_force_n.max()))

            if not settling:
                max_pre_step_force_n = max(max_pre_step_force_n, peak)
                if not post_step_safety:max_force_n = max(max_force_n, peak)
                if not post_step_safety and per_env_safety and max_safe_force_n > 0:
                    per_env_peak = torch.linalg.vector_norm(force, dim=-1).amax(dim=1)
                    if contact_observer is not None and contact_observer.safety_peak_force_n is not None:
                        per_env_peak = torch.maximum(per_env_peak, contact_observer.safety_peak_force_n)
                    rejected = torch.zeros_like(active)
                    if hasattr(chain,'initial_state_gate'):
                        rejected=chain.initial_state_gate(frame=frame,q=robot.data.joint_pos[:, -16:])
                        if rejected.shape!=active.shape:raise ValueError('Initial-state gate must return[batch]')
                    rejected=active & rejected
                    initial_rejected|=rejected
                    newly_stopped = active & ((per_env_peak > max_safe_force_n)|rejected)
                    stopped_at = torch.where(newly_stopped, frame, stopped_at)
                    # A terminated lane no longer contributes data. Park its
                    # palm away from the object before the next physics step
                    # so the batch can continue without loading that contact.
                    parked_pose = pose.clone()
                    parked_pose[:, 2] += 2.0
                    hold_pose = torch.where(newly_stopped[:, None], parked_pose, hold_pose)
                    active = active & ~newly_stopped
                elif not post_step_safety and max_safe_force_n > 0 and peak > max_safe_force_n:
                    stop_frame, reason = frame, f"force safety stop: {peak:.2f} N"
                    break
                found_sum += int(found.sum().item())
                stage = 0 if frame < 75 else 1 if frame < 250 else 2
                stage_found[stage] += found.float().mean(dim=0)
                stage_count[stage] += 1

            q = robot.data.joint_pos[:, -16:]
            command = command_path[frame].expand(batch, -1)
            bootstrap_nominal = (
                (1.0 - _smoothstep(frame, bootstrap_end))
                * torch.as_tensor(data["q_hand"][0], device=device)
                + _smoothstep(frame, bootstrap_end)
                * torch.as_tensor(data["q_hand"][frame], device=device)
            ).expand(batch, -1)
            try:
                if hasattr(chain, "set_training_context"):
                    stabilizer_context = {}
                    if getattr(chain, 'needs_stabilizer_diagnostics', False):
                        incidental_sensor = env.scene['incidental_hand_object_contact_force']
                        incidental_sensor.update(0.0)
                        incidental = incidental_sensor.data
                        stabilizer_context = dict(
                            incidental_hand_force=incidental.force,
                            incidental_hand_found=incidental.found,
                            joint_pos_limits=robot.data.joint_pos_limits)
                    chain.set_training_context(frame=frame, settling=settling,
                        active=active, newly_stopped=newly_stopped,
                        q_velocity=robot.data.joint_vel[:, -16:], palm_pose=pose,
                        previous_actuator_target=robot.data.joint_pos_target[:, -16:],
                        points=points, force=force, found=found,
                        last_observation=(not settling and frame == n - 1),
                        **stabilizer_context)
                # warm-up already set a negative offset; do not clobber it
                if chain.trace is not None and not settling:
                    chain.trace.frame_offset = frame - chain.control_frame
                action, _ = chain.step(
                    q=q, palm_pose_world=pose, contact_force_world=force,
                    contact_found=found, contact_points_world=points,
                    contact_normals_inward_world=inward, palm_twist_world=twist,
                    planner_command_palm=command,
                    contact_observed=found & (
                        torch.linalg.vector_norm(force, dim=-1) >= CONTACT_THRESHOLD),
                    bootstrap_nominal=bootstrap_nominal)
            except RuntimeError as error:
                stop_frame, reason = frame, str(error)
                break
            if per_env_safety and not settling:
                action = torch.where(active[:, None], action, torch.zeros_like(action))
                if bool(newly_stopped.any()):
                    parked_root = torch.cat((hold_pose, torch.zeros((batch, 6), device=device)), dim=1)
                    palm_root[0] = torch.where(active[:, None], palm_root[0], parked_root)
                    write_palm_root(palm_root[0])
            env.step(palm_driver.augment_action(action) if hasattr(palm_driver, "augment_action") else action)
            if post_step_safety and not settling:
                if not per_env_safety:raise ValueError('post-step safety requires lane-wise termination')
                env.sim.sense()
                read_sensors()
                per_env_peak=contact_observer.safety_peak_force_n
                if hasattr(chain,'substep_peak'):per_env_peak=torch.maximum(per_env_peak,chain.substep_peak[frame])
                max_force_n=max(max_force_n,float(per_env_peak.max()))
                newly_stopped=active & (per_env_peak>max_safe_force_n)
                stopped_at=torch.where(newly_stopped,frame,stopped_at)
                active=active & ~newly_stopped
                if hasattr(chain,'record_post_physics'):
                    chain.record_post_physics(frame=frame,q=robot.data.joint_pos[:, -16:],palm_pose=pose,
                        active=active,newly_stopped=newly_stopped,settling=False)
                if bool(newly_stopped.any()):
                    parked_pose=pose.clone();parked_pose[:,2]+=2.
                    hold_pose=torch.where(newly_stopped[:,None],parked_pose,hold_pose)
                    parked_root=torch.cat((hold_pose,torch.zeros((batch,6),device=device)),1)
                    palm_root[0]=torch.where(active[:,None],palm_root[0],parked_root)
                    write_palm_root(palm_root[0])
            if not settling:
                completed += 1
                if stop_when_all_inactive and not bool(active.any()):
                    break

        torch.cuda.synchronize(torch_device)
        duration_s = time.perf_counter() - started

        trace_path = None
        if trace is not None:
            trace.add_series("q_hand_reference", list(data["q_hand"][:n]))
            if "q_cmd" in data:
                trace.add_series("q_cmd_reference", list(data["q_cmd"][:n]))
            if dp_trace_h5 is not None:
                trace.write(dp_trace_h5)
                trace_path = str(dp_trace_h5)
            if report_csv is not None:
                trace.write_csv(report_csv)
            print(f"[REPORT] {trace_path or report_csv}"
                  f"{' + ' + str(report_csv) if trace_path and report_csv else ''}", flush=True)

        return {
            "schema": "deploy_b_gpu_serial_parity_v1",
            "episode_id": episode, "batch": batch, "seed": seed,
            "residual_mode": residual_mode,
            "per_env_safety": per_env_safety,
            "per_env_stop_frame": stopped_at.cpu().tolist(),
            "settle_steps": settle_steps, "bootstrap_end": bootstrap_end,
            "requested_frames": n, "completed_frames": completed,
            "stop_frame": stop_frame, "stop_reason": reason,
            "duration_s": duration_s,
            "env_control_steps_per_second": completed * batch / max(duration_s, 1e-9),
            "found_fraction": found_sum / max(completed * batch * 4, 1),
            "peak_force_n": max_force_n,
            "pre_step_forward_peak_force_n": max_pre_step_force_n,
            "safety_observation_phase": ('actual_physics_substeps_and_post_step' if post_step_safety else 'pre_step_forward'),
            "found_per_finger_by_stage": [
                (stage_found[i] / c).tolist() if c else None
                for i, c in enumerate(stage_count)],
            "dp_trace_h5": trace_path,
            "report_csv": str(report_csv) if report_csv else None,
            **({'palm_support':palm_driver.report()} if palm_driver is not None else {}),
            **({'async_dp':chain.async_planner.report()} if chain.async_planner is not None else {}),
            **({'per_env_initial_rejected':initial_rejected.cpu().tolist(),
                'all_inactive_early_exit':completed<n and not bool(active.any())}
               if stop_when_all_inactive or hasattr(chain,'initial_state_gate') else {}),
        }
    finally:
        # Drain the inference worker before destroying its environment or
        # issuing device-wide synchronization. It never updates the env.
        if chain is not None:
            chain.close()
        # close() only tears down the renderer in this mjlab version. CUDA
        # graphs and the palm-recenter closure otherwise survive in cycles.
        # Release graphs outside capture, after all submitted work completes.
        torch.cuda.synchronize(torch_device)
        if hasattr(env.sim, 'step_orig'):
            env.sim.step = env.sim.step_orig
            del env.sim.step_orig
        for graph_name in ('step_graph', 'forward_graph', 'reset_graph', 'sense_graph'):
            setattr(env.sim, graph_name, None)
        env.close()


def _smoothstep(frame: int, end: int) -> float:
    """deploy_dp_inverse.py:2243-2246."""
    progress = min(float(frame) / max(float(end), 1.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--episode-id", type=int, default=9)
    parser.add_argument("--frames", type=int, default=2500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual-mode", choices=("off", "zero"), default="off")
    parser.add_argument("--per-env-safety", action="store_true",
                        help="freeze only the lane exceeding the force limit; continue other lanes")
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--max-safe-force-n", type=float, default=100.0)
    parser.add_argument("--report", type=Path,
                        help="per-frame CSV in the serial report's column schema")
    parser.add_argument("--dp-trace-h5", type=Path,
                        help="full DP trace: per-round input/output and the pre-MCC command")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.batch, args.episode_id, args.frames, args.device, args.seed,
                 args.report, args.dp_trace_h5, args.max_safe_force_n, args.settle_steps,
                 args.residual_mode, args.per_env_safety)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "found_per_finger_by_stage"}, indent=2))




# --------------------------------------------------------------------------
# merged from batched_dp_trace.py
# """Full-fidelity DP deployment trace for the batched B -> chunk scheduler -> MCC chain.
#
# Records the DP interface (observation history in, action chunk out, plus the
# anchor) and the MCC interface (the scheduler command handed to the controller
# each frame, and the controller's answer), together with the 96-D input layout
# and every normalisation tensor, so a trace can be read back without the
# checkpoint that produced it.
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checkpoint_meta(path: Path) -> dict[str, Any]:
    """Scalars and the declared state layout, without materialising the weights.

    ``mmap=True`` maps the weight storages lazily, so reading ``step`` and
    ``state_fields`` off a 1 GB checkpoint costs a header parse rather than a
    full load.  The returned dict holds only extracted values, so the mapping is
    released as soon as this returns.
    """
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return {}

    def get(key, default=None):
        try:
            value = blob.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    return {
        "step": int(get("step", -1)),
        "architecture": str(get("architecture", "")),
        "state_fields": get("state_fields"),
    }


def _as_numpy(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).numpy().astype(dtype)
    return np.asarray(value, dtype=dtype)


def _scalar(value: Any) -> Any:
    """HDF5 attributes accept only plain Python scalars and strings."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item()) if value.numel() == 1 else value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return json.dumps([_scalar(v) for v in value])
    if isinstance(value, dict):
        return json.dumps({k: _scalar(v) for k, v in value.items()})
    return str(value)


class BatchedDPTrace:
    """Capture every DP round and every control frame of a batched deployment.

    All buffers are preallocated on the compute device and copied to the host
    once, in :meth:`write`, so the hot loop pays only a slice assignment.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        batch: int,
        device: torch.device | str,
        *,
        max_frames: int,
        control_dt: float,
        scheduler_config: Any,
        action_scale: float = 0.08,
        runtime: Any = None,
        note: str = "",
    ):
        from deploy_dp_inverse import DPRuntime  # local import keeps this module import-light

        self.device = torch.device(device)
        self.batch = int(batch)
        self.checkpoint = Path(checkpoint)
        self.runtime = runtime if runtime is not None else DPRuntime(
            self.checkpoint, self.device, None, 42
        )
        runtime = self.runtime
        # The runtime stores the schema name but not the per-field layout, so
        # read the layout straight off the checkpoint that declares it.
        self.checkpoint_meta = _read_checkpoint_meta(self.checkpoint)

        self.obs_horizon = int(runtime.obs_horizon)
        self.pred_horizon = int(runtime.pred_horizon)
        self.state_dim = int(runtime.state_dim)
        self.action_dim = int(runtime.action_dim)
        self.robot_state_dim = int(runtime.robot_state_dim)
        self.stride = int(runtime.stride)
        self.control_dt = float(control_dt)
        self.action_scale = float(action_scale)
        self.note = str(note)

        self.scheduler_config = scheduler_config
        replan = int(getattr(scheduler_config, "replan_interval", 10))
        self.replan_interval = replan
        # A round may land on the last frame, hence the +2.
        self.max_rounds = max(1, max_frames // max(replan, 1) + 2)
        self.max_frames = int(max_frames)

        dev = self.device
        f32 = torch.float32
        # The runtime keeps normalisation in numpy; the chain moves it to the
        # device itself.  Keep a device copy here for the same reason.
        self.action_mean_t = torch.as_tensor(runtime.action_mean, device=dev, dtype=f32)
        self.action_std_t = torch.as_tensor(runtime.action_std, device=dev, dtype=f32)

        # --- DP rounds -----------------------------------------------------
        self.plan_frame = torch.full((self.max_rounds,), -1, device=dev, dtype=torch.long)
        self.plan_history = torch.zeros(
            (self.max_rounds, self.batch, self.obs_horizon, self.state_dim), device=dev, dtype=f32)
        self.plan_history_normalized = torch.zeros_like(self.plan_history)
        self.plan_anchor_q = torch.zeros((self.max_rounds, self.batch, self.action_dim), device=dev, dtype=f32)
        self.plan_anchor_from_history = torch.zeros_like(self.plan_anchor_q)
        self.plan_pred_normalized = torch.zeros(
            (self.max_rounds, self.batch, self.pred_horizon, self.action_dim), device=dev, dtype=f32)
        self.plan_pred_delta = torch.zeros_like(self.plan_pred_normalized)
        self.plan_absolute_q = torch.zeros_like(self.plan_pred_normalized)
        self.plan_drop_index = torch.zeros((self.max_rounds, self.batch), device=dev, dtype=torch.long)
        self.plan_path = None          # allocated on the first round (replan_interval is known)
        # The full (batch, capacity, 96) sliding buffer as it stood at the plan,
        # not just the sampled history, so the stride-5 gather can be replayed.
        self.plan_states = None
        self.plan_count = 0

        # --- control frames ------------------------------------------------
        self.frame_t = torch.full((self.max_frames,), -1, device=dev, dtype=torch.long)
        self.frame_replanned = torch.zeros((self.max_frames, self.batch), device=dev, dtype=torch.bool)
        self.frame_q_live = torch.zeros((self.max_frames, self.batch, self.action_dim), device=dev, dtype=f32)
        self.frame_nominal_q = torch.zeros_like(self.frame_q_live)
        self.frame_command_q = torch.zeros_like(self.frame_q_live)
        self.frame_action = torch.zeros_like(self.frame_q_live)
        self.frame_palm_pose = torch.zeros((self.max_frames, self.batch, 7), device=dev, dtype=f32)
        self.frame_palm_twist = torch.zeros((self.max_frames, self.batch, 6), device=dev, dtype=f32)
        self.frame_planner_command = torch.zeros((self.max_frames, self.batch, 6), device=dev, dtype=f32)
        self.frame_force = torch.zeros((self.max_frames, self.batch, 4, 3), device=dev, dtype=f32)
        self.frame_found = torch.zeros((self.max_frames, self.batch, 4), device=dev, dtype=torch.bool)
        self.frame_points = torch.zeros((self.max_frames, self.batch, 4, 3), device=dev, dtype=f32)
        self.frame_normals_inward = torch.zeros((self.max_frames, self.batch, 4, 3), device=dev, dtype=f32)
        self.frame_state = torch.zeros((self.max_frames, self.batch, self.state_dim), device=dev, dtype=f32)
        self.frame_control_normal = torch.zeros((self.max_frames, self.batch, 4, 3), device=dev, dtype=f32)
        self.frame_count = 0

        self.extra: dict[str, list] = {}
        # The chain counts its own control steps from zero, but a trace is
        # compared against a serial report keyed by roll-out frame.  The caller
        # sets this to (rollout frame - chain.control_frame) so both use the
        # roll-out frame.
        self.frame_offset = 0

    # ------------------------------------------------------------------ hooks
    @torch.no_grad()
    def record_plan(
        self, *, frame: int, history: torch.Tensor, normalized: torch.Tensor,
        prediction: torch.Tensor, chunk: torch.Tensor, anchor: torch.Tensor,
        drop_index: torch.Tensor, path: torch.Tensor | None,
        observation: torch.Tensor,
    ) -> None:
        """One DP inference round, recorded at the moment the chunk is installed."""
        if self.plan_count >= self.max_rounds:
            raise RuntimeError("DP trace round buffer exhausted; raise max_frames")
        i = self.plan_count
        self.plan_frame[i] = int(frame) + self.frame_offset
        self.plan_history[i] = history
        self.plan_history_normalized[i] = normalized
        self.plan_pred_normalized[i] = prediction
        self.plan_pred_delta[i] = prediction * self.action_std_t + self.action_mean_t
        self.plan_absolute_q[i] = chunk
        self.plan_anchor_q[i] = anchor
        self.plan_anchor_from_history[i] = history[:, -1, : self.action_dim]
        if self.plan_path is None and path is not None:
            # path is (batch, replan_interval, action_dim); keep every axis but
            # the round index, so the batch dimension is not dropped.
            self.plan_path = torch.zeros(
                (self.max_rounds, *path.shape), device=self.device, dtype=torch.float32)
        if self.plan_path is not None and path is not None:
            self.plan_path[i] = path.to(torch.float32)
        self.plan_drop_index[i] = drop_index
        if observation is not None:
            if self.plan_states is None:
                self.plan_states = torch.zeros(
                    (self.max_rounds, *observation.shape), device=self.device, dtype=torch.float32)
            self.plan_states[i] = observation
        self.plan_count += 1

    @torch.no_grad()
    def record_frame(
        self, *, frame: int, replanned: bool, q_live: torch.Tensor,
        nominal_q: torch.Tensor, command_q: torch.Tensor, action: torch.Tensor,
        palm_pose_world: torch.Tensor, palm_twist_world: torch.Tensor,
        planner_command_palm: torch.Tensor, force_world: torch.Tensor,
        found: torch.Tensor, contact_points_world: torch.Tensor,
        contact_normals_inward_world: torch.Tensor, state: torch.Tensor,
        control_normal_world: torch.Tensor | None = None,
    ) -> None:
        """One control frame: what the scheduler asked for and what MCC returned."""
        if self.frame_count >= self.max_frames:
            raise RuntimeError("DP trace frame buffer exhausted; raise max_frames")
        i = self.frame_count
        self.frame_t[i] = int(frame) + self.frame_offset
        self.frame_replanned[i] = replanned
        self.frame_q_live[i] = q_live
        self.frame_nominal_q[i] = nominal_q
        self.frame_command_q[i] = command_q
        self.frame_action[i] = action
        self.frame_palm_pose[i] = palm_pose_world
        self.frame_palm_twist[i] = palm_twist_world
        self.frame_planner_command[i] = planner_command_palm
        self.frame_force[i] = force_world
        self.frame_found[i] = found
        self.frame_points[i] = contact_points_world
        self.frame_normals_inward[i] = contact_normals_inward_world
        self.frame_state[i] = state
        if control_normal_world is not None:
            self.frame_control_normal[i] = control_normal_world
        self.frame_count += 1

    def add_series(self, name: str, values: Sequence[Any]) -> None:
        """Align a trajectory series to recorded frames, including warm-up."""
        if len(values) == 0:
            raise ValueError(f"{name}: empty reference series")
        source = [_as_numpy(v) for v in values]
        self.extra[name] = [source[min(max(int(self.frame_t[i]), 0), len(source) - 1)]
                            for i in range(self.frame_count)]

    # ------------------------------------------------------------------ layout
    def state_layout(self) -> list[tuple[str, int, int]]:
        """``(field, start, stop)`` for the 96-D input, from the checkpoint contract."""
        fields = self.checkpoint_meta.get("state_fields")
        if not fields:
            raise RuntimeError(
                f"checkpoint {self.checkpoint} does not declare state_fields; "
                "cannot describe the 96-D input layout"
            )
        layout, start = [], 0
        for name, dim in fields:
            stop = start + int(dim)
            layout.append((str(name), start, stop))
            start = stop
        if start != self.state_dim:
            raise RuntimeError(f"state_fields cover {start} dims, expected {self.state_dim}")
        return layout

    def _attrs(self) -> dict[str, Any]:
        runtime = self.runtime
        config = runtime.config
        scheduler_config = getattr(self, "scheduler_config", None)
        attrs = {
            "format": "batched_b_dp_deployment_trace_v1",
            "note": self.note,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_step": self.checkpoint_meta.get("step", -1),
            "dataset_file": str(getattr(config, "file", "")),
            "batch": self.batch,
            # --- input contract ---
            "state_schema": runtime.state_schema,
            "state_dim": self.state_dim,
            "robot_state_dim": self.robot_state_dim,
            "environment_state_dim": int(runtime.environment_state_dim),
            "obs_horizon": self.obs_horizon,
            "stride": self.stride,
            "control_dt": self.control_dt,
            "input_frame": runtime.input_frame,
            "contact_normal_polarity": runtime.contact_normal_polarity,
            "active_input_fields": json.dumps(
                [str(v) for v in getattr(runtime, "active_input_fields", []) or []]
            ),
            "state_layout_note": (
                "history is (batch, obs_horizon, state_dim); index j samples control "
                "frame t-(obs_horizon-1-j)*stride, so index -1 is the newest. Columns "
                "are the concatenation of the state_layout rows, in order."
            ),
            # --- output contract ---
            "action_field": runtime.action_field,
            "action_representation": runtime.action_representation,
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "action_unit": "rad",
            "action_waypoint_dt": float(getattr(runtime, "action_waypoint_dt", 0.0)),
            "action_scale_actuator": self.action_scale,
            "action_pipeline_note": (
                "plan_pred_normalized is the raw conditional_sample output; "
                "plan_pred_delta = pred_normalized*action_std + action_mean; "
                "plan_absolute_q = plan_pred_delta + plan_anchor_q[:, :, None]; "
                "the scheduler turns absolute_q into frame_nominal_q."
            ),
            # --- inference ---
            "inference_steps": int(getattr(config, "inference_steps", -1)),
            "noise_scheduler": str(getattr(config, "noise_scheduler", "")),
            "diffusion_steps": int(getattr(config, "diffusion_steps", -1)),
            "architecture": self.checkpoint_meta.get("architecture", ""),
            "num_inference_steps_effective": int(
                getattr(runtime.policy.diffusion, "num_inference_steps", -1)
            ),
            # --- controller interface ---
            "replan_interval": self.replan_interval,
            "frames_recorded": self.frame_count,
            "plan_rounds_recorded": self.plan_count,
            "mcc_interface_note": (
                "frame_nominal_q is the command handed to MCC before any force "
                "compensation; frame_command_q is MCC's answer; frame_action is "
                "((command_q - q_live)/action_scale).clamp(-2,2)."
            ),
        }
        if scheduler_config is not None:
            for key, value in sorted(vars(scheduler_config).items()):
                attrs[f"scheduler_{key}"] = _scalar(value)
        return attrs

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        layout = self.state_layout()
        n_rounds, n_frames = self.plan_count, self.frame_count

        def put(group, name, tensor, length=None):
            block = tensor[:length] if length is not None else tensor
            group.create_dataset(
                name, data=_as_numpy(block),
                compression="gzip", compression_opts=4, shuffle=True,
            )

        with h5py.File(path, "w") as file:
            for key, value in self._attrs().items():
                file.attrs[key] = value

            layout_group = file.create_group("state_layout")
            layout_group.create_dataset(
                "field", data=np.asarray([row[0] for row in layout], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            layout_group.create_dataset("start", data=np.asarray([row[1] for row in layout], dtype=np.int32))
            layout_group.create_dataset("stop", data=np.asarray([row[2] for row in layout], dtype=np.int32))
            layout_group.create_dataset(
                "is_robot_state",
                data=np.asarray([row[2] <= self.robot_state_dim for row in layout]),
            )

            norm = file.create_group("normalization")
            put(norm, "state_mean", torch.as_tensor(self.runtime.state_mean))
            put(norm, "state_std", torch.as_tensor(self.runtime.state_std))
            put(norm, "state_input_mask", torch.as_tensor(self.runtime.state_input_mask))
            put(norm, "action_mean", torch.as_tensor(self.runtime.action_mean))
            put(norm, "action_std", torch.as_tensor(self.runtime.action_std))

            plan = file.create_group("plan")
            put(plan, "frame", self.plan_frame, n_rounds)
            put(plan, "history", self.plan_history, n_rounds)
            put(plan, "history_normalized", self.plan_history_normalized, n_rounds)
            put(plan, "anchor_q", self.plan_anchor_q, n_rounds)
            put(plan, "anchor_q_from_history", self.plan_anchor_from_history, n_rounds)
            put(plan, "pred_normalized", self.plan_pred_normalized, n_rounds)
            put(plan, "pred_delta", self.plan_pred_delta, n_rounds)
            put(plan, "absolute_q", self.plan_absolute_q, n_rounds)
            put(plan, "scheduler_drop_index", self.plan_drop_index, n_rounds)
            if self.plan_states is not None:
                put(plan, "states_buffer_at_plan", self.plan_states, n_rounds)
            if self.plan_path is not None:
                put(plan, "scheduler_path", self.plan_path, n_rounds)
            plan.attrs["states_buffer_at_plan"] = (
                "the observation module's full (batch, capacity, 96) sliding buffer at "
                "the moment of the plan; history is this buffer gathered at "
                "capacity-1-arange(obs_horizon-1,-1,-1)*stride"
            )
            plan.attrs["anchor_check"] = (
                "anchor_q is the anchor the chain actually used; anchor_q_from_history "
                "is history[:, -1, :16]. They must match for the anchored-delta "
                "contract to hold."
            )

            frames = file.create_group("frame")
            for name, tensor in (
                ("control_frame", self.frame_t),
                ("replanned", self.frame_replanned),
                ("q_live", self.frame_q_live),
                ("nominal_q", self.frame_nominal_q),
                ("command_q", self.frame_command_q),
                ("action", self.frame_action),
                ("palm_pose_world", self.frame_palm_pose),
                ("palm_twist_world", self.frame_palm_twist),
                ("planner_command_palm", self.frame_planner_command),
                ("force_world", self.frame_force),
                ("found", self.frame_found),
                ("contact_points_world", self.frame_points),
                ("contact_normals_inward_world", self.frame_normals_inward),
                ("observation_state", self.frame_state),
                ("control_normal_world", self.frame_control_normal),
            ):
                put(frames, name, tensor, n_frames)
            for name, values in self.extra.items():
                block = values[:n_frames]
                if not block:
                    continue  # a run that stopped before the DP phase has no frames to pair
                frames.create_dataset(
                    name, data=np.stack(block),
                    compression="gzip", compression_opts=4, shuffle=True,
                )
        self._write_layout_csv(path.with_suffix(".layout.csv"))
        return path

    def write_csv(self, path: str | Path, *, contact_threshold: float = 0.05) -> Path:
        """Per-frame CSV in the serial `--report` column schema, for direct diffing.

        One row per (control frame, environment).  Column names match the serial
        deployment report so the two can be joined on `frame` without renaming.
        Columns the batched chain has no counterpart for (the recovery
        orchestrator) are emitted as zeros rather than omitted, so their absence
        shows up as a flat line in the diff instead of a missing column.
        """
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fingers = ("index", "middle", "ring", "thumb")
        n_frames = self.frame_count
        n_batch = self.batch
        bootstrap_end = (self.obs_horizon - 1) * self.stride

        q_dp = self.frame_nominal_q[:n_frames]
        q_live = self.frame_q_live[:n_frames]
        q_cmd = self.frame_command_q[:n_frames]
        force = self.frame_force[:n_frames]
        found = self.frame_found[:n_frames]
        force_mag = torch.linalg.vector_norm(force, dim=-1)
        loaded = found & (force_mag >= contact_threshold)
        ref = self.extra.get("q_hand_reference")

        header = (
            ["frame", "env", "dp_calls", "chunk_drop_index",
             "found_contacts", "loaded_contacts", "force_max_N",
             "fullhand_phase", "fullhand_mcc_calibrated",
             "fullhand_recovery_fingers",
             "fullhand_runtime_recovery_offset_max_rad"]
            + [f"{f}_found" for f in fingers]
            + [f"{f}_force_raw_N" for f in fingers]
            + [f"q_dp_{j}" for j in range(self.action_dim)]
            + [f"q_live_{j}" for j in range(self.action_dim)]
            + [f"q_cmd_{j}" for j in range(self.action_dim)]
            + [f"q_teacher_{j}" for j in range(self.action_dim)]
        )

        def num(value, digits=6):
            return f"{float(value):.{digits}f}"

        plan_frames = {int(self.plan_frame[k]): k for k in range(self.plan_count)}
        rounds_so_far = 0
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i in range(n_frames):
                t = int(self.frame_t[i])
                if t in plan_frames:
                    rounds_so_far += 1
                dpi = self.plan_drop_index[plan_frames[t]][0] if t in plan_frames else 0
                for e in range(n_batch):
                    row = [
                        t, e, rounds_so_far, int(dpi),
                        int(found[i, e].sum()), int(loaded[i, e].sum()),
                        num(force_mag[i, e].max()),
                        "bootstrap" if t <= bootstrap_end else "track", 1,
                        0, "0.000000",
                    ]
                    row += [int(v) for v in found[i, e]]
                    row += [num(v) for v in force_mag[i, e]]
                    row += [num(v) for v in q_dp[i, e]]
                    row += [num(v) for v in q_live[i, e]]
                    row += [num(v) for v in q_cmd[i, e]]
                    row += ([num(v) for v in ref[i]] if ref is not None and i < len(ref)
                            else [""] * self.action_dim)
                    writer.writerow(row)
        return path

    @staticmethod
    def csv_from_h5(h5_path: str | Path, csv_path: str | Path,
                    *, contact_threshold: float = 0.05) -> Path:
        """Re-emit the CSV from a written trace, without rerunning the rollout.

        Lets the report column set be revised against an existing trace instead
        of paying for another deployment.
        """
        import csv

        h5_path, csv_path = Path(h5_path), Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fingers = ("index", "middle", "ring", "thumb")
        with h5py.File(h5_path, "r") as f:
            obs = int(f.attrs["obs_horizon"])
            stride = int(f.attrs["stride"])
            action_dim = int(f.attrs["action_dim"])
            bootstrap_end = (obs - 1) * stride
            t = f["frame/control_frame"][:].astype(int)
            nom = f["frame/nominal_q"][:]
            q = f["frame/q_live"][:]
            cmd = f["frame/command_q"][:]
            force = f["frame/force_world"][:]
            found = f["frame/found"][:].astype(bool)
            pframe = f["plan/frame"][:].astype(int)
            pdrop = f["plan/scheduler_drop_index"][:]
            ref = f["frame/q_hand_reference"][:] if "frame/q_hand_reference" in f else None
        mag = np.linalg.norm(force, axis=-1)
        loaded = found & (mag >= contact_threshold)
        n_frames, n_batch = len(t), q.shape[1]

        header = (
            ["frame", "env", "dp_calls", "chunk_drop_index",
             "found_contacts", "loaded_contacts", "force_max_N",
             "fullhand_phase", "fullhand_mcc_calibrated",
             "fullhand_recovery_fingers",
             "fullhand_runtime_recovery_offset_max_rad"]
            + [f"{x}_found" for x in fingers]
            + [f"{x}_force_raw_N" for x in fingers]
            + [f"q_dp_{j}" for j in range(action_dim)]
            + [f"q_live_{j}" for j in range(action_dim)]
            + [f"q_cmd_{j}" for j in range(action_dim)]
            + [f"q_teacher_{j}" for j in range(action_dim)]
        )
        plan_at = {int(pframe[k]): k for k in range(len(pframe))}
        rounds = 0
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i in range(n_frames):
                frame = int(t[i])
                if frame in plan_at:
                    rounds += 1
                drop = int(pdrop[plan_at[frame]][0]) if frame in plan_at else 0
                for e in range(n_batch):
                    row = [frame, e, rounds, drop,
                           int(found[i, e].sum()), int(loaded[i, e].sum()),
                           f"{mag[i, e].max():.6f}",
                           "bootstrap" if frame <= bootstrap_end else "track", 1,
                           0, "0.000000"]
                    row += [int(v) for v in found[i, e]]
                    row += [f"{v:.6f}" for v in mag[i, e]]
                    row += [f"{v:.6f}" for v in nom[i, e]]
                    row += [f"{v:.6f}" for v in q[i, e]]
                    row += [f"{v:.6f}" for v in cmd[i, e]]
                    row += ([f"{v:.6f}" for v in ref[i]] if ref is not None
                            else [""] * action_dim)
                    writer.writerow(row)
        return csv_path

    def _write_layout_csv(self, path: Path) -> None:
        lines = ["field,start,stop,dim,block"]
        for name, start, stop in self.state_layout():
            block = "robot_state" if stop <= self.robot_state_dim else "environment_state"
            lines.append(f"{name},{start},{stop},{stop - start},{block}")
        path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# merged from torch_mesh_normal.py
# """GPU port of the source-mesh contact normal used by the serial deployment.
#
# The serial controller does not feed the DP or MCC the MuJoCo contact normal.
# `--dp-tactile-normal-source` defaults to `source_mesh_oracle`
# (deploy_dp_inverse.py:4179): the collision point is used only as a query
# anchor, is projected onto the exact closest triangle of the *undecomposed*
# source mesh, and the three source vertex normals are interpolated
# barycentrically there.  That removes the seam discontinuities the convex
# collision decomposition introduces.
# --------------------------------------------------------------------------
class TorchMeshNormalOracle:
    """Batched closest-triangle projection + barycentric vertex-normal blend."""

    def __init__(self, config, *, scale: float = 1.0,
                 device: torch.device | str = "cuda:0",
                 chunk_triangles: int = 4096):
        from object_catalog import MeshNormalOracle

        oracle = MeshNormalOracle.from_config(config, scale=scale)
        if oracle is None:
            raise ValueError("object config exposes no source mesh")
        self.oracle = oracle
        self.device = torch.device(device)
        self.chunk_triangles = int(chunk_triangles)
        f32 = torch.float32
        self.triangles = torch.as_tensor(
            np.asarray(oracle.mesh.triangles), device=self.device, dtype=f32)      # (T,3,3)
        self.faces = torch.as_tensor(
            np.asarray(oracle.mesh.faces), device=self.device, dtype=torch.long)   # (T,3)
        self.vertex_normals = torch.as_tensor(
            np.asarray(oracle.vertex_normals), device=self.device, dtype=f32)      # (V,3)
        self.face_normals = torch.as_tensor(
            np.asarray(oracle.mesh.face_normals), device=self.device, dtype=f32)   # (T,3)

    @torch.no_grad()
    def query_surface_object_frame(self, points: torch.Tensor):
        """Return ``(closest, inward_normal, distance)`` for object-frame anchors.

        Points are assumed to sit close to the surface (they are contact
        points), so the closest point is found by projecting onto each
        triangle plane, clamping the barycentric coordinates into the
        triangle, and keeping the smallest distance.  Triangle iteration is
        chunked purely to bound peak memory; every query point is still
        tested against every triangle.
        """
        single = points.ndim == 1
        p = (points[None] if single else points).to(self.device, torch.float32)
        n_points = p.shape[0]
        best_d2 = torch.full((n_points,), float("inf"), device=self.device)
        best_closest = torch.zeros((n_points, 3), device=self.device)
        best_tri = torch.zeros(n_points, device=self.device, dtype=torch.long)

        for start in range(0, self.triangles.shape[0], self.chunk_triangles):
            tri = self.triangles[start:start + self.chunk_triangles]      # (C,3,3)
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            ab, ac = b - a, c - a                                          # (C,3)
            normal = torch.linalg.cross(ab, ac)
            normal = normal / torch.linalg.vector_norm(
                normal, dim=-1, keepdim=True).clamp_min(1e-12)
            rel = p[:, None, :] - a[None, :, :]                            # (M,C,3)
            height = torch.einsum("mci,ci->mc", rel, normal)
            projected = p[:, None, :] - height[..., None] * normal[None]   # (M,C,3)
            # barycentric of the projected point
            ap = projected - a[None]
            d00 = torch.einsum("ci,ci->c", ab, ab)
            d01 = torch.einsum("ci,ci->c", ab, ac)
            d11 = torch.einsum("ci,ci->c", ac, ac)
            d20 = torch.einsum("mci,ci->mc", ap, ab)
            d21 = torch.einsum("mci,ci->mc", ap, ac)
            denom = (d00 * d11 - d01 * d01).clamp_min(1e-18)[None]
            v = (d11[None] * d20 - d01[None] * d21) / denom
            w = (d00[None] * d21 - d01[None] * d20) / denom
            # clamp into the triangle: first to the vertex ranges, then to the
            # hypotenuse v + w <= 1
            v = v.clamp(0.0, 1.0)
            w = w.clamp(0.0, 1.0)
            over = (v + w) > 1.0
            s = (v + w).clamp_min(1e-18)
            v = torch.where(over, v / s, v)
            w = torch.where(over, w / s, w)
            closest = a[None] + v[..., None] * ab[None] + w[..., None] * ac[None]
            d2 = ((p[:, None, :] - closest) ** 2).sum(-1)                  # (M,C)
            chunk_best, chunk_idx = d2.min(dim=1)
            better = chunk_best < best_d2
            best_d2 = torch.where(better, chunk_best, best_d2)
            best_closest = torch.where(better[:, None], closest[
                torch.arange(n_points, device=self.device), chunk_idx], best_closest)
            best_tri = torch.where(better, chunk_idx + start, best_tri)

        tri = self.triangles[best_tri]                                     # (M,3,3)
        bary = _points_to_barycentric(tri, best_closest)                   # (M,3)
        corner = self.vertex_normals[self.faces[best_tri]]                 # (M,3,3)
        normals = torch.einsum("mi,mij->mj", bary, corner)
        norm = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        degenerate = norm[:, 0] < 1e-10
        normals = normals / norm.clamp_min(1e-10)
        face_normal = self.face_normals[best_tri]
        normals = torch.where(degenerate[:, None], face_normal, normals)
        distance = best_d2.clamp_min(0).sqrt()
        if single:
            return best_closest[0], normals[0], distance[0]
        return best_closest, normals, distance


def _points_to_barycentric(triangles: torch.Tensor,
                           points: torch.Tensor) -> torch.Tensor:
    """Barycentric coordinates of `points` inside `triangles` (trimesh's formula)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = torch.einsum("mi,mi->m", v0, v0)
    d01 = torch.einsum("mi,mi->m", v0, v1)
    d11 = torch.einsum("mi,mi->m", v1, v1)
    d20 = torch.einsum("mi,mi->m", v2, v0)
    d21 = torch.einsum("mi,mi->m", v2, v1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-18)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return torch.stack((1.0 - v - w, v, w), dim=-1)

if __name__ == "__main__":
    main()
