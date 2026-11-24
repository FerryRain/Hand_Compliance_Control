"""Execute Aude's wrist and contact torque law without a Cartesian finger position reference."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ..acitive_02 import active_a, aude, author_control, route_free, route_free_inputs
from ..acitive_02.experiment import digest, read_json, write_json
from ..acitive_02.route_free_score import score


class WristGoal(NamedTuple):
    position: np.ndarray
    orientation: Rotation


class NoTipTargetExecutor(author_control.AuthorExecutor):
    """Use the registered direct-torque plant, tactile sensors and intrinsic rest posture."""

    def __init__(self, plant, robot, data, grasp):
        super().__init__(plant, robot, data, grasp)
        # Zero displacement disables only the position-dependent term in the shared torque law.
        self.cartesian_error = np.zeros((4, 3))
        self.cartesian_error.setflags(write=False)
        self.servo_ticks = 0

    def set_wrist(self, position, orientation):
        position = self.origin + route_free.limit(position - self.origin, .45)
        self.goal = WristGoal(position, orientation)
        self.target_wrist = np.r_[position, active_a.quaternion(orientation)]

    def set_goal(self, posture):
        raise TypeError("This executor accepts wrist targets only")

    def step(self):
        model, data, hand = self.plant.model, self.data, self.plant
        self.position, self.rotation = author_control.wrist_step(self.position, self.rotation, self.goal, self.dt)
        data.mocap_pos[hand.driver] = self.position
        data.mocap_quat[hand.driver] = active_a.quaternion(self.rotation)
        mujoco.mj_forward(model, data)
        live, points, forces = self.tactile()
        delta = data.geom_xpos[hand.pads][live] - points[live]
        self.normals[live] = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1.e-10)
        for i, pad in enumerate(hand.pads):
            mujoco.mj_jacGeom(model, data, self.full_jac[i], None, pad)
        velocity = self.full_jac @ data.qvel
        rest_term = .004 * (self.rest - data.qpos[hand.q]) - .000008 * data.qvel[hand.dof]
        data.ctrl[:] = author_control.finger_torque(
            self.full_jac[:, :, hand.dof], velocity, self.cartesian_error, self.normals, live,
            forces, rest_term, data.qfrc_bias[hand.dof])
        mujoco.mj_step(model, data)
        self.servo_ticks += 1

    def sample(self, touch, mode, clock):
        data, driver = self.data, self.plant.driver
        # A zero-width channel explicitly represents an absent external finger command.
        return (data.qpos.copy(), np.r_[data.mocap_pos[driver], data.mocap_quat[driver]], self.target_wrist,
                np.empty(0), data.actuator_force.copy(), touch.points.copy(), touch.normals.copy(),
                touch.forces.copy(), touch.loads.copy(), touch.measured_heads, touch.legal_posture,
                mode, clock, self.cartesian_error, data.ctrl.copy())


def run(obj, fixtures, out, seconds):
    paired, initial, case, surface = route_free_inputs.load_trial(fixtures, obj)
    plant, data, robot, sensed, touch, state, _ = initial
    if (robot.objects or robot.surface_queries or plant.surface_queries or
            sha256(state.tobytes()).hexdigest() != paired["integration_state"] or
            digest(case/"model.mjb") != paired["model.mjb"] or digest(surface) != paired["surface.obj"]):
        raise ValueError("Expected the registered tactile-only LEAP fixture")
    files = [Path(__file__), *(Path(m.__file__) for m in (active_a, aude, author_control, route_free, route_free_inputs))]
    sig = dict(object=obj, seed=paired["seed"], initial=paired["integration_state"], source=author_control.SOURCE,
               code={p.name:digest(p) for p in files}, seconds=seconds, period=author_control.PERIOD,
               servo_hz=500, contact_log_hz=100, planner_variables=22, planning_object_geometries=0,
               planner_feedback="physical", wrist_mcc=False, finger_mcc=False, execute_planned_fingers=False,
               finger_reference_source=None, actuator="Aude direct joint torque", cartesian_tracking=False,
               intrinsic_nullspace_rest=True, policy_queries=0)
    out = out.resolve()
    if out.exists():
        rec = read_json(out/"run.json")
        if sig != rec["signature"] or any(digest(out/k) != v for k,v in rec["hashes"].items()):
            raise ValueError(f"Existing run does not match requested acquisition at {out}")
        return
    start = aude.state_posture(robot, sensed)
    grasp = active_a.GraspEnvelope(start.joints, robot.limits)
    ik = route_free.GraspIK(robot, grasp, solver=author_control.stable_trf)
    explorer = aude.AudeExplorer(robot, ik, paired["seed"], aude.DEFAULT_STEP_M)
    explorer.directions = route_free.TimedDirections(paired["seed"])
    execute = NoTipTargetExecutor(plant, robot, data, grasp)
    execute.set_wrist(start.position, start.orientation)
    observed = execute.physical_observation(touch)
    ticks = seconds * 100
    fields = (*author_control.TRACE_FIELDS, "cartesian_tracking_error_m", "issued_finger_torque_nm", "planner_pad_valid")
    first = (*execute.sample(touch, "initial", 0.), observed.measured_heads)
    trace = {k:np.empty((ticks+1,*np.shape(v)), dtype="U12" if k=="mode" else np.asarray(v).dtype)
             for k,v in zip(fields,first,strict=True)}
    for k,v in zip(fields,first,strict=True):
        trace[k][0] = v
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)/"run"
        stage.mkdir()
        with (stage/"plans.jsonl").open("w", encoding="utf-8") as stream:
            began = perf_counter()
            for tick in range(ticks):
                explorer.observe(sensed,observed)
                if tick % author_control.PERIOD == 0:
                    t0 = perf_counter()
                    ik.center = route_free.touch_frame(observed,ik.center)
                    ik.last_fit_ms = []
                    explorer.directions.elapsed_ms = 0.
                    plan = explorer.plan(sensed,observed)
                    execute.set_wrist(plan.posture.position,plan.posture.orientation)
                    row = dict(tick=tick,mode=plan.mode,live=plan.valid.tolist(),wrist=execute.target_wrist.tolist(),
                               aude_joints_unexecuted_rad=plan.posture.joints.tolist(), targets_m=plan.targets.tolist(),
                               directions=plan.direction.tolist(),target_error_mm=plan.error_mm,fit_calls=len(ik.last_fit_ms),
                               ik_ms=sum(ik.last_fit_ms),gp_direction_ms=explorer.directions.elapsed_ms,
                               plan_ms=(perf_counter()-t0)*1000)
                    stream.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
                for _ in range(active_a.CONTROL_STEPS):
                    execute.step()
                mujoco.mj_forward(plant.model,data)
                touch = plant.measure(data)
                active_a.robot_state(plant,robot,data,sensed)
                observed = execute.physical_observation(touch)
                values = (*execute.sample(touch,plan.mode,perf_counter()-began), observed.measured_heads)
                for k,v in zip(fields,values,strict=True):
                    trace[k][tick+1] = v
                if (tick+1) % 6000 == 0:
                    print(obj, (tick+1)/100, "s", flush=True)
        if (abs(data.time-seconds)>1e-6 or execute.servo_ticks!=ticks*5 or np.any(trace["cartesian_tracking_error_m"])
                or trace["goal_joints_rad"].shape!=(ticks+1,0)
                or not all(np.isfinite(trace[k]).all() for k in ("qpos","finger_torque_nm","issued_finger_torque_nm"))
                or not np.any(trace["finger_torque_nm"][1:])):
            raise ValueError("Incomplete rollout or invalid torque command")
        np.savez_compressed(stage/"trace.npz",**trace)
        hashes = {name:digest(stage/name) for name in ("trace.npz","plans.jsonl")}
        write_json(stage/"run.json",dict(signature=sig,hashes=hashes,trace_sha256=hashes["trace.npz"],
                   initial_sha256=paired["integration_state"],model_sha256=paired["model.mjb"],
                   offline_mesh_sha256=paired["surface.obj"],seed=paired["seed"],steps=ticks,servo_steps=execute.servo_ticks,
                   rest_rad=execute.rest.tolist(),planning_object_geometries=0,wall_s=float(trace["wall_s"][-1])))
        stage.rename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object",choices=route_free_inputs.OBJECTS,required=True)
    ap.add_argument("--fixtures",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seconds",type=int,default=240)
    args = ap.parse_args()
    if args.seconds<=0:
        ap.error("A positive simulation duration is required")
    torch.set_num_threads(1)
    run(args.object,args.fixtures,args.out,args.seconds)
    print(score(args.out,args.fixtures/args.object/"surface.obj"),flush=True)
