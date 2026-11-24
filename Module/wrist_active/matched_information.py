"""Compare wrist planners on the same action and finger-control interface."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ..acitive_02 import active_a as aude
from ..acitive_02.active_gpis import NOISE_VARIANCE, VOXEL_M
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback
from ..i04_geometric_traversal.physics import quaternion, rotation
from .buffered_fingers import BufferedFingers
from .experiment import ACTIVE2, digest, save_json, seed_for
from .load_guard import GuardedWrist
from .mcc_trials import RESULTS, read
from .planner import interpolate
from .sweep import DT, PLAN_TICKS, PREVIEW_S
from .sweep_experiment import OUT, fixture, hashes
from .sweep_suite import jobs
from .wrist_worker import WristWorker


SECONDS = 240
STEP_M = .002
STEP_RAD = .020
LENGTHS_M = (.025, .018, .039)
CONTACT_BUDGETS = (500, 1000)
PEAR_SEEDS = (20261120, 20261122)
METHODS = ("wrist-dp", "aude-dp", "wrist-mcc", "aude-mcc")
CASES = [(r["obj"], r["start"]) for r in jobs() if r["series"] == "runs" and r["arm"] == "sweep-dp"]


def register(root):
    root.mkdir(exist_ok=True, parents=True)
    frozen_wrist = read(OUT / "force-priority/protocol.json")["source"]
    checked = hashes()
    checked["load_guard.py"] = digest(Path(__file__).with_name("load_guard.py"))
    if any(frozen_wrist.get(name) != value for name, value in checked.items()):
        raise ValueError("A frozen wrist planning dependency changed")
    for name, value in read(ACTIVE2 / "protocol.json")["source_sha256"].items():
        if digest(Path(aude.__file__).with_name(name)) != value:
            raise ValueError(f"Frozen Aude dependency changed: {name}")
    spec = dict(steps=round(SECONDS / DT), control_step_s=DT, wrist_segment_ticks=PLAN_TICKS,
                max_step_m=STEP_M, max_step_rad=STEP_RAD, workspace="ball of initial palm radius + 0.13 m",
                preview_s=PREVIEW_S, finger_controller="ForceFeedback, five substeps per tick",
                dp="BufferedFingers, 50 ms handoff, same initial warm history and frozen checkpoint",
                pear_dp_seeds=PEAR_SEEDS, cases={f"{o}/{s}":seed_for(o,s) for o,s in CASES},
                primary_length_m=LENGTHS_M[0], sensitivity_lengths_m=LENGTHS_M[1:],
                noise_variance=NOISE_VARIANCE, unique_voxel_m=VOXEL_M, new_contact_budgets=CONTACT_BUDGETS,
                metrics="conditional log-det information at 60/120/240 s and at fixed counts of new cells",
                note="Only measured zero-level touches. Condition on the four initial touches, include each voxel once.",
                evaluation_prior="unit-variance Matern 3/2; a common stationary evaluation model for both arms",
                planner_inputs="object-free hand kinematics and physical tactile observations",
                frozen_wrist_protocol=digest(OUT / "force-priority/protocol.json"),
                frozen_baseline_protocol=digest(ACTIVE2 / "protocol.json"),
                sources={name: digest(Path(__file__).with_name(name)) for name in
                         ("buffered_fingers.py", "sweep_experiment.py", "load_guard.py", "wrist_worker.py")},
                runner_sha256=digest(Path(__file__)))
    # Store lists in the same form as their on-disk JSON representation.
    spec = json.loads(json.dumps(spec))
    path = root / "protocol.json"
    if path.exists():
        if read(path) != spec:
            raise ValueError("Registered settings differ from this runner")
    else:
        save_json(path, spec)
    return spec


class DirectWrist:
    """Both planners command the same position-driven physical wrist."""

    @staticmethod
    def forecast(pose):
        return pose


class AudeWristPlan:
    def __init__(self, hand, sensed, seed):
        self.hand = hand
        live = aude.state_posture(hand, sensed)
        self.grasp = aude.GraspEnvelope(live.joints, hand.limits)
        ik = aude.JointIK(hand, limits=self.grasp.limits)
        self.explorer = aude.AudeExplorer(hand, ik, seed, aude.DEFAULT_STEP_M)
        self.planner = aude.ActiveSurfacePlanner(ik, self.grasp, live, 3, DT)
        self.planner.plan_dt = PLAN_TICKS * DT
        self.raw = np.r_[live.position, quaternion(live.orientation)]
        self.wrist = DirectWrist()
        self.segments = 0

    def command(self, tick, sensed, touch):
        self.explorer.observe(sensed, touch)
        self.planner.observe(sensed, touch, tick)
        if tick % aude.PLAN_TICKS == 0:
            self.proposal = self.explorer.plan(sensed, touch)
        phase = tick % PLAN_TICKS
        if phase == 0:
            live = aude.state_posture(self.hand, sensed)
            goal, mode, self.normal = self.planner.plan(live, self.proposal.posture, touch, tick * DT)
            if self.planner.done:
                raise ValueError("Aude finished all three passes before the paired budget")
            start = aude.Posture(self.raw[:3], rotation(self.raw[3:]), self.grasp.bound(live.joints))
            positions, quats, _ = aude.Leg(start, goal, mode).samples(PLAN_TICKS)
            self.path = np.column_stack((positions, quats))
            self.segments += 1
        self.raw = self.path[phase]
        joints = sensed.qpos[self.hand.q]
        now = self.planner.stabilize(self.raw, joints, touch, self.normal, self.wrist, tick % aude.SERVO_TICKS == 0)
        ahead = self.planner.stabilize(self.path[-1], joints, touch, self.normal, self.wrist, False)
        return now, ahead


def bounded(origin, target, radius, steps):
    result = target.copy()
    position = result[:3]
    norm = np.linalg.norm(position)
    if norm > radius:
        position *= radius / norm
    delta = position - origin[:3]
    distance = np.linalg.norm(delta)
    if distance > steps * STEP_M:
        position[:] = origin[:3] + delta * (steps * STEP_M / distance)
    start, end = rotation(origin[3:]), rotation(target[3:])
    turn = (end * start.inv()).as_rotvec()
    angle = np.linalg.norm(turn)
    if angle > steps * STEP_RAD:
        result[3:] = quaternion(Rotation.from_rotvec(turn * (steps * STEP_RAD / angle)) * start)
    return result, bool(norm > radius or distance > steps * STEP_M or angle > steps * STEP_RAD)


def run(root, method, seed, obj, start):
    protocol = root / "protocol.json"
    folder = root / "matched" / obj / start / f"seed-{seed}" / method
    mode, fingers = method.split("-")
    receipt = folder / "run.json"
    if receipt.exists():
        saved = read(receipt)
        if (saved["protocol_sha256"] != digest(protocol) or saved["method"] != method or saved["seed"] != seed
                or saved["trace_sha256"] != digest(folder / "trace.npz")):
            raise ValueError(f"Existing matched trial differs: {folder}")
        recorded = read(protocol)["runner_sha256"]
        if recorded != digest(Path(__file__)):
            original = root / "registered-source/matched_information.py"
            if not original.is_file() or digest(original) != recorded:
                raise ValueError("Registered acquisition source unavailable")
        return
    spec = register(root)
    folder.mkdir(exist_ok=True, parents=True)
    (plant, data, hand, sensed, touch, initial, warm), case, _ = fixture(obj, start)
    frozen = read(OUT / "protocol.json")["fixtures"][f"{obj}/{start}"]
    if digest(case / "model.mjb") != frozen["model.mjb"] or digest(case / "initial.npz") != frozen["initial.npz"]:
        raise ValueError("Wrong comparison fixture")
    n = spec["steps"]
    wrist = data.qpos[:7].copy()
    radius = np.linalg.norm(wrist[:3]) + .13
    nominal = data.qpos[plant.q].copy()
    normals = touch.normals.copy()
    feedback = ForceFeedback()
    fractions = np.arange(1, aude.CONTROL_STEPS + 1) / aude.CONTROL_STEPS
    def sample():
        return dict(qpos=data.qpos, palm_command=wrist, finger_nominal=nominal, points_m=touch.points,
                    normals_outward=touch.normals, pad_valid=touch.measured_heads,
                    pad_force_n=touch.forces, region_load_n=touch.loads)
    arrays = {k: np.empty((n+1, *np.shape(v)), dtype=v.dtype) for k, v in sample().items()}
    for k, value in sample().items():
        arrays[k][0] = value
    clips = segments = 0
    with ExitStack() as stack:
        policy = None
        if fingers == "dp":
            history = 0 if warm is None else len(warm)
            policy = stack.enter_context(BufferedFingers(seed, n+history, nominal, hand.limits, paced=False))
            if warm is not None:
                for idx, packet in enumerate(warm):
                    policy.command(idx, packet)
        if mode == "wrist":
            planner = GuardedWrist(hand, sensed, touch)
            worker = stack.enter_context(WristWorker(planner, False))
            worker.observe(sensed, touch, 0)
            worker.submit(sensed, touch, 0, 0)
            worker.install(0, initial=True)
            segments = 1
        else:
            planner = AudeWristPlan(hand, sensed, seed_for(obj, start))
        for tick in range(n):
            if mode == "wrist":
                if tick and worker.update(sensed, touch, tick, n) is not None:
                    segments += 1
                command, ahead = planner.command(touch)
            else:
                command, ahead = planner.command(tick, sensed, touch)
            command, clipped = bounded(wrist, command, radius, 1)
            clips += clipped
            if policy is not None:
                ahead, _ = bounded(command, ahead, radius, round(PREVIEW_S / DT))
                nominal = policy.command(history+tick, aude.policy_observation(plant, data, ahead))
            normals[touch.measured_heads] = touch.normals[touch.measured_heads]
            for pose in interpolate(wrist, command, fractions):
                data.ctrl[:] = feedback.command(plant, data, nominal, normals, touch, True)
                data.mocap_pos[plant.driver] = pose[:3]
                data.mocap_quat[plant.driver] = pose[3:]
                mujoco.mj_step(plant.model, data)
            wrist = command
            mujoco.mj_forward(plant.model, data)
            touch = plant.measure(data)
            aude.robot_state(plant, hand, data, sensed)
            for k, value in sample().items():
                arrays[k][tick+1] = value
            if (tick+1) % 6000 == 0:
                print(f"{obj}/{start}/{method}/{seed}: {(tick+1)*DT:g} s", flush=True)
        if policy is not None:
            policy.report(folder)
    if mode == "aude":
        segments = planner.segments
    if segments != n//PLAN_TICKS or abs(data.time-SECONDS) > 1.e-6 or not np.isfinite(arrays["qpos"]).all():
        raise ValueError("Matched action budget or physical time differs")
    np.savez_compressed(folder / "trace.npz", **arrays)
    save_json(receipt, dict(method=method, seed=seed, planner_seed=seed_for(obj, start), steps=n,
              segments=segments, clipped_wrist_steps=clips, initial_sha256=sha256(initial.tobytes()).hexdigest(),
              model_sha256=frozen["model.mjb"], maximum_radius_m=float(radius),
              protocol_sha256=digest(protocol), trace_sha256=digest(folder/"trace.npz"),
              policy=policy.info if policy is not None else None, finger_controller="ForceFeedback"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path)
    cli.add_argument("--method", choices=METHODS)
    cli.add_argument("--case", choices=[f"{o}/{s}" for o,s in CASES], default="pear/active-02")
    cli.add_argument("--seed", type=int)
    args = cli.parse_args()
    torch.set_num_threads(1)
    obj, start = args.case.split("/")
    allowed = PEAR_SEEDS if obj == "pear" else (seed_for(obj, start),)
    seed = args.seed if args.seed is not None else allowed[0]
    if seed not in allowed:
        cli.error(f"Unregistered sampling seed for {args.case}")
    root = args.root or (RESULTS if obj == "pear" else RESULTS/"transfer-extension")
    if args.method:
        run(root, args.method, seed, obj, start)
    else:
        if (root/"registered-source/matched_information.py").is_file() and (root/"protocol.json").is_file():
            if digest(root/"registered-source/matched_information.py") != read(root/"protocol.json")["runner_sha256"]:
                raise ValueError("Archived primary protocol differs")
        else:
            register(root)
