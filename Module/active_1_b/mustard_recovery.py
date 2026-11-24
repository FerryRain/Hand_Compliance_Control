"""Test finger step limits and lost-contact retries on registered Active-1-B fixtures."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
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
from ..i04_geometric_traversal import fingertip_dp_compare, fingertip_dp_worker, fingertip_local_policy
from . import single_all


class LastAttemptIK(route_free.GraspIK):
    """Remember the exact targets of the last normal IK attempt, including its preload."""

    def fit(self, start, offsets, targets):
        self.last_targets = targets.copy()
        return super().fit(start, offsets, targets)


class RecoveryExplorer(aude.AudeExplorer):
    def __init__(self, hand, ik, seed, radius_m):
        super().__init__(hand, ik, seed, aude.DEFAULT_STEP_M)
        self.radius_m = radius_m

    def plan(self, data, touch):
        original = super().plan(data, touch)
        self.retry = None
        if not self.radius_m or original.mode != "recontact" or not original.valid.any():
            return original
        targets = self.ik.last_targets.copy()
        axes = data.xmat[self.hand.tip_bodies].reshape(4, 3, 3)
        material_points = data.xpos[self.hand.tip_bodies] + np.einsum("ijk,ik->ij", axes, self.offsets)
        delta = targets - material_points
        distance = np.linalg.norm(delta, axis=1)
        lost = ~original.valid & (np.linalg.norm(self.last_normals, axis=1) > .5)
        limited = lost & (distance > self.radius_m)
        if not limited.any():
            return original
        # A temporary IK reference. Only actual contacts enter the GP or refresh last_points.
        targets[limited] = material_points[limited] + delta[limited] * (self.radius_m / distance[limited, None])
        pose, error, gap = self.ik.fit(aude.state_posture(self.hand, data), self.offsets, targets)
        accepted = error <= 13. and gap >= -.001
        self.retry = dict(limited=limited.tolist(), distance_before_mm=(1000*distance).tolist(),
                          original_error_mm=original.error_mm, retry_error_mm=error, min_self_gap_m=gap,
                          accepted=bool(accepted))
        if accepted:
            return aude.Plan(pose, "explore", targets, original.direction, original.valid, error, gap)
        return original


def run(fixtures, output, fingers, radius_mm, seed, seconds, max_finger_step_rad=None, object_name="mustard"):
    paired, initial, case, surface = load_trial(fixtures, object_name)
    plant, data, robot, sensed, touch, state, _ = initial
    if (robot.objects or robot.surface_queries or plant.surface_queries
            or sha256(state.tobytes()).hexdigest() != paired["integration_state"]
            or digest(case/"model.mjb") != paired["model.mjb"] or digest(surface) != paired["surface.obj"]):
        raise ValueError(f"{object_name} trial does not match its registered initial state and input boundary")
    seed = paired["seed"] if seed is None else seed
    ticks, period = seconds * 100, author_control.PERIOD
    sources = [Path(__file__), *(Path(m.__file__) for m in
               (aude, author_control, route_free, single_all, fingertip_local_policy, fingertip_dp_worker))]
    signature = dict(object=object_name, initial=paired["integration_state"], seed=seed, seconds=seconds,
                     source=author_control.SOURCE, code={p.name:digest(p) for p in sources},
                     planner_feedback="physical", planner_variables=22, planning_object_geometries=0,
                     period=period, servo_hz=500, contact_log_hz=100, wrist_mcc=False, finger_mcc=False,
                     lost_target_retry_radius_mm=radius_mm, finger_policy=fingers)
    if fingers == "single_all":
        release = read_json(fingertip_dp_worker.ROOT / "finger_policy/checkpoints/manifest.json")
        signature["checkpoint_sha256"] = release[fingertip_dp_worker.POLICY]["sha256"]
    if max_finger_step_rad is not None:
        signature["max_finger_step_rad"] = max_finger_step_rad
    output = output.resolve()
    if output.exists():
        saved = read_json(output / "run.json")
        expected = signature
        # These two archived drivers ran the identical loop with the fixture fixed to mustard.
        old_driver = saved["signature"]["code"].get(Path(__file__).name)
        if object_name == "mustard" and old_driver in (
            "d749941c043eee1bbcc5cf38d5d6b34f9329f8b9f5da9acc18322f17feb95d3e",
            "e753f9af7e4710d827f95d05c6c42296a78b159f61728ab1f8a8ee0cc256969d",
        ):
            expected = dict(signature, code={**signature["code"], Path(__file__).name: old_driver})
        if saved["signature"] != expected or any(digest(output/k) != v for k,v in saved["hashes"].items()):
            raise ValueError(f"Different trial already exists at {output}")
        return
    initial_pose = aude.state_posture(robot, sensed)
    grasp = active_a.GraspEnvelope(initial_pose.joints, robot.limits)
    ik = LastAttemptIK(robot, grasp, solver=author_control.stable_trf)
    explorer = RecoveryExplorer(robot, ik, seed, radius_mm/1000.)
    explorer.directions = route_free.TimedDirections(seed)
    execute = author_control.AuthorExecutor(plant, robot, data, grasp)
    execute.set_goal(initial_pose)
    observed = execute.physical_observation(touch)
    rows = [execute.sample(touch, "initial", 0.)]
    physical = [observed.measured_heads.copy()]
    raw = [initial_pose.joints.copy()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        provider = fingertip_local_policy.LocalFingerPolicy(seed, ticks) if fingers == "single_all" else nullcontext()
        policy_record = {}
        with provider as policy, (stage / "plans.jsonl").open("w", encoding="utf-8") as file:
            if max_finger_step_rad is not None:
                policy.scheduler.config.max_joint_step = max_finger_step_rad
                policy.info["max_joint_step_rad_per_control_tick"] = max_finger_step_rad
            started = perf_counter()
            for tick in range(ticks):
                explorer.observe(sensed, observed)
                if tick % period == 0:
                    began = perf_counter()
                    ik.center = route_free.touch_frame(observed, ik.center)
                    ik.last_fit_ms = []
                    explorer.directions.elapsed_ms = 0.
                    plan = explorer.plan(sensed, observed)
                    target = plan.posture
                    if policy is None:
                        execute.set_goal(target)
                    else:
                        execute.set_goal(active_a.Posture(target.position, target.orientation, execute.goal.joints))
                        forecast = single_all.wrist_forecast(execute)
                    record = dict(tick=tick, mode=plan.mode, live=plan.valid.tolist(), wrist=execute.target_wrist.tolist(),
                                  targets_m=plan.targets.tolist(), directions=plan.direction.tolist(),
                                  aude_joints_rad=target.joints.tolist(), target_error_mm=plan.error_mm,
                                  fit_calls=len(ik.last_fit_ms), ik_ms=sum(ik.last_fit_ms),
                                  gp_direction_ms=explorer.directions.elapsed_ms,
                                  plan_ms=(perf_counter()-began)*1000., retry=explorer.retry)
                    file.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")
                if policy is not None:
                    packet = fingertip_dp_compare.policy_observation(plant, data, next(forecast))
                    command = np.asarray(policy.command(tick, packet), dtype=float)
                    raw.append(command.copy())
                    execute.set_goal(active_a.Posture(target.position, target.orientation, command))
                for _ in range(active_a.CONTROL_STEPS):
                    execute.step()
                mujoco.mj_forward(plant.model, data)
                touch = plant.measure(data)
                active_a.robot_state(plant, robot, data, sensed)
                observed = execute.physical_observation(touch)
                physical.append(observed.measured_heads.copy())
                rows.append(execute.sample(touch, plan.mode, perf_counter()-started))
                if (tick+1) % 6000 == 0:
                    print(f"{object_name} {fingers} radius={radius_mm:g} seed={seed} {(tick+1)/100:.0f} s", flush=True)
            if policy is not None:
                queries = len(range(policy.bootstrap, ticks, policy.scheduler.config.replan_interval))
                if len(policy.inference_ms) != queries or policy.info["checkpoint_sha256"] != signature["checkpoint_sha256"]:
                    raise ValueError("Policy checkpoint or query count mismatch")
                policy_record = dict(policy=dict(policy.info, queries=queries, bootstrap_frames=policy.bootstrap),
                                     policy_timing=policy.report(stage))
        trace = {k:np.asarray(v) for k,v in zip(author_control.TRACE_FIELDS, zip(*rows), strict=True)}
        trace["planner_pad_valid"] = np.asarray(physical)
        if policy_record:
            trace["single_all_raw_rad"] = np.asarray(raw)
            np.testing.assert_allclose(trace["goal_joints_rad"], grasp.bound(trace["single_all_raw_rad"]), atol=1e-10, rtol=0)
        if abs(data.time-seconds)>1e-6 or not all(np.isfinite(trace[k]).all() for k in ("qpos","finger_torque_nm")):
            raise ValueError("Invalid simulated trajectory")
        np.savez_compressed(stage/"trace.npz", **trace)
        names = ["trace.npz", "plans.jsonl"] + (["timing.npz"] if policy_record else [])
        hashes = {name:digest(stage/name) for name in names}
        write_json(stage/"run.json", dict(signature=signature, hashes=hashes, trace_sha256=hashes["trace.npz"],
                   initial_sha256=paired["integration_state"], model_sha256=paired["model.mjb"],
                   offline_mesh_sha256=paired["surface.obj"], seed=seed, steps=ticks, wall_s=float(trace["wall_s"][-1]),
                   planning_object_geometries=0, rest_rad=execute.rest.tolist(), **policy_record))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, default="mustard")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fingers", choices=("aude", "single_all"), required=True)
    parser.add_argument("--radius-mm", type=float, required=True, help="0 disables the extra retry")
    parser.add_argument("--max-finger-step-rad", type=float, help="single_all only, per 10 ms command tick")
    parser.add_argument("--seed", type=int, help="Defaults to the paired fixture seed")
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args()
    if not np.isfinite(args.radius_mm) or args.radius_mm < 0 or args.seconds < 1:
        parser.error("Use a nonnegative radius and positive duration")
    if args.max_finger_step_rad is not None and (args.fingers != "single_all"
            or not np.isfinite(args.max_finger_step_rad) or args.max_finger_step_rad <= 0):
        parser.error("A finger step limit must be positive and requires single_all")
    torch.set_num_threads(1)
    run(args.fixtures, args.out, args.fingers, args.radius_mm, args.seed, args.seconds,
        args.max_finger_step_rad, args.object)
    print(score(args.out, args.fixtures/args.object/"surface.obj"), flush=True)
