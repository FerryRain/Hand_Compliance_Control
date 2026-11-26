"""Execute AUDE's wrist goal with the released single_all policy and fingertip MCC."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mujoco
import numpy as np

from PARK.identity import digest, read
from ..acitive_02 import active_a, aude, author_control, route_free, route_free_inputs
from ..active_1_b.single_all import wrist_forecast
from ..i04_geometric_traversal import fingertip_dp_compare, fingertip_dp_worker, fingertip_local_policy
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback
from ..wrist_active.buffered_fingers import BufferedFingers


RELEASED_MCC_SHA256 = "8064f6ea0f8a46835784385ae9571c3b0c1fdf35906e02e5bd50e2f1ec11584f"


class FingerMCCExecutor(author_control.AuthorExecutor):
    """Share AUDE's physical contact observer and wrist DS; retain the native position motors."""

    def __init__(self, plant, data, initial):
        # AuthorExecutor.__init__ converts the LEAP position motors to torque sources.
        # This condition uses the original MJB actuator gains and bias for joint control.
        self.plant, self.data = plant, data
        self.dt = plant.model.opt.timestep
        if (self.dt != .002 or plant.model.nu != 16 or np.any(plant.model.actuator_gainprm[:, 0] <= 0)
                or np.any(plant.model.actuator_biasprm[:, 1] >= 0)):
            raise ValueError("Expected the registered LEAP position servos at 500 Hz")
        self.position = data.mocap_pos[plant.driver].copy()
        self.rotation = active_a.rotation(data.mocap_quat[plant.driver])
        self.origin = data.qpos[:3].copy()
        self.wrench = np.empty(6)
        self.feedback = ForceFeedback()
        self.goal = initial
        self.set_wrist(initial)

    def set_wrist(self, planned):
        position = self.origin + route_free.limit(planned.position - self.origin, .45)
        self.goal = active_a.Posture(position, planned.orientation, self.goal.joints)
        self.target_wrist = np.r_[position, active_a.quaternion(planned.orientation)]

    def advance(self, nominal, normals, touch):
        self.goal = active_a.Posture(self.goal.position, self.goal.orientation, nominal.copy())
        for _ in range(active_a.CONTROL_STEPS):
            self.position, self.rotation = author_control.wrist_step(self.position, self.rotation, self.goal, self.dt)
            self.data.mocap_pos[self.plant.driver] = self.position
            self.data.mocap_quat[self.plant.driver] = active_a.quaternion(self.rotation)
            self.data.ctrl[:] = self.feedback.command(self.plant, self.data, nominal, normals, touch, True)
            mujoco.mj_step(self.plant.model, self.data)
        mujoco.mj_forward(self.plant.model, self.data)


def run(obj, fixtures, out, seconds, *, constrain_palm=False):
    paired, (plant, data, hand, sensed, touch, state, warm), case, mesh = route_free_inputs.load_trial(fixtures, obj)
    if (hand.objects or hand.surface_queries or plant.surface_queries or warm is not None
            or sha256(state.tobytes()).hexdigest() != paired["integration_state"]
            or digest(case/"model.mjb") != paired["model.mjb"] or digest(mesh) != paired["surface.obj"]):
        raise ValueError("Unmatched start, physical model or planning inputs")
    seed, ticks = paired["seed"], seconds * 100
    checkpoint = read(fixtures / "fixtures.json")["checkpoint"]
    sources = [Path(m.__file__) for m in (aude, author_control, route_free, route_free_inputs,
                                          fingertip_local_policy, fingertip_dp_compare, fingertip_dp_worker)]
    sources.extend(Path(fn.__code__.co_filename) for fn in (
        BufferedFingers.command, ForceFeedback.command, active_a.ForceFeedback.__init__, wrist_forecast))
    sources.append(Path(__file__))
    limit = None
    if constrain_palm:
        from . import fingerall_height, height

        limit = fingerall_height.PalmDistance(ticks)
        sources.extend((Path(fingerall_height.__file__), Path(height.__file__)))
    condition = "aude_mcc_height" if constrain_palm else "aude_mcc"
    signature = dict(object=obj, condition=condition, seconds=seconds, seed=seed, initial=paired["integration_state"],
                     source=author_control.SOURCE, code={p.name: digest(p) for p in sources}, start=paired["start"],
                     period=author_control.PERIOD, servo_hz=500, contact_log_hz=100, planner_feedback="physical",
                     planner_variables=22, planning_object_geometries=0, execute_planned_fingers=False,
                     finger_mcc=True, wrist_mcc=False, finger_policy=fingertip_dp_worker.POLICY,
                     checkpoint_sha256=checkpoint, wrist_control="AUDE wrist_step", history="current rollout only")
    if limit is not None:
        signature.update(limit.signature())
    out = Path(out).resolve()
    if out.exists():
        saved = read(out / "run.json")
        expected = signature
        # Accept the published unconstrained implementation with its original acquisition hash.
        if not constrain_palm and saved["signature"]["code"].get("mcc.py") == RELEASED_MCC_SHA256:
            expected = {**signature, "code": {**signature["code"], "mcc.py": RELEASED_MCC_SHA256}}
        if saved["signature"] != expected or any(digest(out / f) != h for f, h in saved["hashes"].items()):
            raise ValueError(f"A different trial exists at {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    initial = aude.state_posture(hand, sensed)
    grasp = active_a.GraspEnvelope(initial.joints, hand.limits)
    ik = route_free.GraspIK(hand, grasp, solver=author_control.stable_trf)
    explorer = aude.AudeExplorer(hand, ik, seed, aude.DEFAULT_STEP_M)
    explorer.directions = route_free.TimedDirections(seed)
    execute = FingerMCCExecutor(plant, data, initial)
    if limit is not None:
        limit.observe(0, touch, data.qpos[:7])
    observed = execute.physical_observation(touch)
    normals = touch.normals.copy()
    fields = (*author_control.TRACE_FIELDS, "planner_pad_valid", "finger_nominal_rad", "finger_command_rad",
              "mcc_bias_rad", "policy_wrist_preview")
    first = (*execute.sample(touch, "initial", 0.), observed.measured_heads, initial.joints, data.ctrl,
             execute.feedback.bias, execute.target_wrist)
    trace = {k: np.empty((ticks+1, *np.shape(v)), dtype="U12" if k == "mode" else np.asarray(v).dtype)
             for k, v in zip(fields, first, strict=True)}
    for k, v in zip(fields, first, strict=True):
        trace[k][0] = v
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        with (BufferedFingers(seed, ticks, initial.joints, hand.limits, paced=False) as policy,
              (stage / "plans.jsonl").open("w", encoding="utf-8") as stream):
            start = perf_counter()
            for tick in range(ticks):
                explorer.observe(sensed, observed)
                if tick % author_control.PERIOD == 0:
                    began = perf_counter()
                    ik.center = route_free.touch_frame(observed, ik.center)
                    ik.last_fit_ms = []
                    explorer.directions.elapsed_ms = 0.
                    plan = explorer.plan(sensed, observed)
                    height_record = {}
                    if limit is None:
                        execute.set_wrist(plan.posture)
                    else:
                        height_record = limit.issue(execute, plan.posture, tick)
                    forecast = wrist_forecast(execute)
                    record = dict(tick=tick, mode=plan.mode, live=plan.valid.tolist(), wrist=execute.target_wrist.tolist(),
                                  aude_joints_unexecuted_rad=plan.posture.joints.tolist(),
                                  targets_m=plan.targets.tolist(), directions=plan.direction.tolist(),
                                  target_error_mm=plan.error_mm, fit_calls=len(ik.last_fit_ms), ik_ms=sum(ik.last_fit_ms),
                                  gp_direction_ms=explorer.directions.elapsed_ms, plan_ms=(perf_counter()-began)*1000,
                                  **height_record)
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                ahead = next(forecast)
                packet = fingertip_dp_compare.policy_observation(plant, data, ahead)
                nominal = policy.command(tick, packet)
                normals[touch.measured_heads] = touch.normals[touch.measured_heads]
                execute.advance(nominal, normals, touch)
                touch = plant.measure(data)
                if limit is not None:
                    limit.observe(tick + 1, touch, data.qpos[:7])
                active_a.robot_state(plant, hand, data, sensed)
                observed = execute.physical_observation(touch)
                values = (*execute.sample(touch, plan.mode, perf_counter()-start), observed.measured_heads, nominal,
                          data.ctrl, execute.feedback.bias, ahead)
                for k, v in zip(fields, values, strict=True):
                    trace[k][tick+1] = v
                if (tick+1) % 6000 == 0:
                    print(f"{obj} {condition} {(tick+1)/100:g} s", flush=True)
            queries = len(range(policy.bootstrap, ticks-policy.lead_ticks, policy.scheduler.config.replan_interval))
            if (policy.info["checkpoint_sha256"] != checkpoint or policy.installs != queries
                    or len(policy.inference_ms) != queries or policy.skipped or policy.missed):
                raise ValueError("Unmatched deployed finger controller or missing action chunks")
            info = dict(policy.info, queries=queries, bootstrap_frames=policy.bootstrap, handoff_misses=policy.missed,
                        skipped_queries=policy.skipped, underrun_ticks=policy.underrun)
            timing = policy.report(stage)
        if abs(data.time-seconds) > 1e-6 or not np.isfinite(trace["qpos"]).all():
            raise ValueError("Incomplete simulation")
        limits = plant.limits
        applied = np.clip(trace["finger_nominal_rad"][1:] + trace["mcc_bias_rad"][1:],
                          limits[:, 0] + .01, limits[:, 1] - .01)
        np.testing.assert_allclose(applied, trace["finger_command_rad"][1:], atol=1e-10, rtol=0)
        if limit is not None:
            limit.add_trace(trace, stage)
        np.savez_compressed(stage / "trace.npz", **trace)
        hashes = {f: digest(stage / f) for f in ("plans.jsonl", "trace.npz", "timing.npz")}
        if limit is not None:
            hashes["height_calibration.json"] = digest(stage / "height_calibration.json")
        receipt = dict(signature=signature, hashes=hashes, seed=seed, steps=ticks, trace_sha256=hashes["trace.npz"],
                       initial_sha256=paired["integration_state"], model_sha256=paired["model.mjb"],
                       offline_mesh_sha256=paired["surface.obj"], robot_sha256=digest(fixtures / "hand.mjb"),
                       initial_file_sha256=paired["initial.npz"], wall_s=float(trace["wall_s"][-1]),
                       joint_limits_rad=limits.tolist(), policy=info, policy_timing=timing,
                       planning_object_geometries=0, actuation="LEAP position servos with fingertip ForceFeedback")
        if limit is not None:
            receipt["height_summary"] = limit.summary()
        active_a.write_json(stage / "run.json", receipt)
        stage.rename(out)
