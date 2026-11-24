"""Pair the Active-1-B wrist planner and torque controller with frozen single_all fingers."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mujoco
import numpy as np
import torch

from ..acitive_02 import active_a, aude, author_control, route_free
from ..acitive_02.experiment import digest, read_json, write_json
from ..acitive_02.route_free_inputs import OBJECTS, load_trial
from ..acitive_02.route_free_score import score
from ..i04_geometric_traversal import fingertip_ddpm, fingertip_dp_compare, fingertip_dp_worker, fingertip_local_policy


def wrist_forecast(executor):
    """Forecast the commanded palm using only the wrist goal already issued this tick."""
    position, rotation, goal = executor.position.copy(), executor.rotation, executor.goal
    for _ in range(active_a.LOOKAHEAD_STEPS):
        position, rotation = author_control.wrist_step(position, rotation, goal, executor.dt)
    while True:
        yield np.r_[position, active_a.quaternion(rotation)]
        for _ in range(active_a.CONTROL_STEPS):
            position, rotation = author_control.wrist_step(position, rotation, goal, executor.dt)


def rollout(obj, fixtures, output, seconds):
    paired, initial, case, mesh = load_trial(fixtures, obj)
    plant, data, robot, sensed, touch, state, _ = initial
    if robot.objects or robot.surface_queries or plant.surface_queries:
        raise ValueError("The online planner must see only the robot and its measured contacts")
    if (sha256(state.tobytes()).hexdigest() != paired["integration_state"]
            or digest(case / "model.mjb") != paired["model.mjb"] or digest(mesh) != paired["surface.obj"]):
        raise ValueError("Trial differs from the registered Active-1-B fixture")
    seed = paired["seed"]
    ticks = seconds * 100
    period = author_control.PERIOD
    sources = [Path(source.__file__) for source in
               (aude, author_control, route_free, fingertip_dp_compare, fingertip_local_policy,
                fingertip_dp_worker, fingertip_ddpm)]
    sources.append(Path(__file__))
    release = read_json(fingertip_dp_worker.ROOT / "finger_policy/checkpoints/manifest.json")
    signature = dict(object=obj, seed=seed, seconds=seconds, initial=paired["integration_state"],
                     model=paired["model.mjb"], offline_mesh=paired["surface.obj"],
                     code={p.name: digest(p) for p in sources}, source=author_control.SOURCE,
                     checkpoint_sha256=release[fingertip_dp_worker.POLICY]["sha256"],
                     planner_feedback="physical", period=period, servo_hz=500, contact_log_hz=100,
                     planner_variables=22, planning_object_geometries=0, wrist_mcc=False, finger_mcc=False,
                     finger_policy=fingertip_dp_worker.POLICY, history="current rollout only",
                     wrist_lookahead_s=active_a.LOOKAHEAD_STEPS * data.model.opt.timestep)
    output = output.resolve()
    if output.exists():
        archived = read_json(output / "run.json")
        if archived["signature"] != signature or any(digest(output / n) != h for n, h in archived["hashes"].items()):
            raise ValueError(f"Existing output differs from this trial: {output}")
        return

    current = aude.state_posture(robot, sensed)
    grasp = active_a.GraspEnvelope(current.joints, robot.limits)
    ik = route_free.GraspIK(robot, grasp, solver=author_control.stable_trf)
    explorer = aude.AudeExplorer(robot, ik, seed, aude.DEFAULT_STEP_M)
    explorer.directions = route_free.TimedDirections(seed)
    executor = author_control.AuthorExecutor(plant, robot, data, grasp)
    executor.set_goal(current)
    observed = executor.physical_observation(touch)
    rows = [executor.sample(touch, "initial", 0.)]
    physical = [observed.measured_heads.copy()]
    raw_commands = [current.joints.copy()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        with fingertip_local_policy.LocalFingerPolicy(seed, ticks) as policy, (stage / "plans.jsonl").open(
                "w", encoding="utf-8") as plans:
            start = perf_counter()
            for tick in range(ticks):
                explorer.observe(sensed, observed)
                if tick % period == 0:
                    began = perf_counter()
                    ik.center = route_free.touch_frame(observed, ik.center)
                    ik.last_fit_ms = []
                    explorer.directions.elapsed_ms = 0.
                    plan = explorer.plan(sensed, observed)
                    # Only the wrist part of the joint solution goes to the executor.
                    target = plan.posture
                    executor.set_goal(active_a.Posture(target.position, target.orientation, executor.goal.joints))
                    forecast = wrist_forecast(executor)
                    row = dict(tick=tick, mode=plan.mode, live=plan.valid.tolist(), wrist=executor.target_wrist.tolist(),
                               aude_joints_unexecuted_rad=target.joints.tolist(), targets_m=plan.targets.tolist(),
                               directions=plan.direction.tolist(), target_error_mm=plan.error_mm,
                               fit_calls=len(ik.last_fit_ms), ik_ms=sum(ik.last_fit_ms),
                               gp_direction_ms=explorer.directions.elapsed_ms, plan_ms=(perf_counter()-began)*1000.)
                    plans.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

                packet = fingertip_dp_compare.policy_observation(plant, data, next(forecast))
                nominal = np.asarray(policy.command(tick, packet), dtype=float)
                raw_commands.append(nominal.copy())
                executor.set_goal(active_a.Posture(target.position, target.orientation, nominal))
                for _ in range(active_a.CONTROL_STEPS):
                    executor.step()
                mujoco.mj_forward(plant.model, data)
                touch = plant.measure(data)
                active_a.robot_state(plant, robot, data, sensed)
                observed = executor.physical_observation(touch)
                physical.append(observed.measured_heads.copy())
                rows.append(executor.sample(touch, plan.mode, perf_counter() - start))
                if (tick + 1) % 6000 == 0:
                    print(f"{obj} {(tick + 1) / 100:.0f} s, {len(explorer.gp.surface)} contact cells", flush=True)
            online_wall = rows[-1][-1]
            if policy.info["checkpoint_sha256"] != signature["checkpoint_sha256"]:
                raise ValueError("The worker loaded a different finger policy")
            query_count = len(range(policy.bootstrap, ticks, policy.scheduler.config.replan_interval))
            if len(policy.inference_ms) != query_count:
                raise ValueError("Missing online finger-policy inference")
            timing = policy.report(stage)
            policy_info = dict(policy.info, bootstrap_frames=policy.bootstrap, queries=query_count)

        trace = {k: np.asarray(v) for k, v in zip(author_control.TRACE_FIELDS, zip(*rows), strict=True)}
        trace["planner_pad_valid"] = np.asarray(physical)
        trace["single_all_raw_rad"] = np.asarray(raw_commands)
        if (abs(data.time - seconds) > 1.e-6 or not np.isfinite(trace["qpos"]).all()
                or not np.isfinite(trace["finger_torque_nm"]).all() or not np.isfinite(raw_commands).all()):
            raise ValueError("Invalid simulated trajectory")
        np.testing.assert_allclose(trace["goal_joints_rad"], grasp.bound(trace["single_all_raw_rad"]), atol=1.e-10, rtol=0)
        np.savez_compressed(stage / "trace.npz", **trace)
        hashes = {name: digest(stage / name) for name in ("trace.npz", "plans.jsonl", "timing.npz")}
        write_json(stage / "run.json", dict(signature=signature, hashes=hashes, trace_sha256=hashes["trace.npz"],
                   initial_sha256=paired["integration_state"], model_sha256=paired["model.mjb"],
                   offline_mesh_sha256=paired["surface.obj"], seed=seed, steps=ticks, wall_s=online_wall,
                   policy=policy_info, policy_timing=timing, actuation="Active-1-B wrist DS and fingertip torque",
                   planning_object_geometries=0, rest_rad=executor.rest.tolist()))
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    torch.set_num_threads(1)
    rollout(args.object, args.fixtures, args.out, args.seconds)
    print(score(args.out, args.fixtures / args.object / "surface.obj"), flush=True)


if __name__ == "__main__":
    main()
