from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Literal

import h5py
import mujoco
import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointRelativePositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import NativeMujocoViewer, ViewerConfig, ViserPlayViewer
from mjlab.utils.lab_api.math import quat_apply_inverse


HAND_XML = Path("src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml")
MCC_TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
MCC_TIP_BODY_NAMES = (
    "fingertip",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
)
MCC_TIP_GEOM_NAMES = (
    r"fingertip_geom",
    r"fingertip_2_geom",
    r"fingertip_3_geom",
    r"thumb_fingertip_geom",
)
MCC_TIP_SITE_LOCAL_POSITIONS = (
    (-0.0106151, -0.0326103, 0.0141088),
    (-0.0106151, -0.0326103, 0.0144487),
    (-0.0106151, -0.0326103, 0.0140386),
    (-0.0106383, -0.0453895, -0.0144321),
)
# Keep deployment contact close to FullHandMCC while allowing a slightly more
# compliant fingertip transition.  The previous collection-style contact
# (-200, -18) with an 18 mm width was far too soft; this midpoint retains a
# millimetre-scale transition without making the hand visually rigid.
CONTACT_SOLREF = (-10_000.0, -280.0)
CONTACT_SOLIMP = (0.90, 0.98, 0.002, 0.5, 2.0)


def _apply_contact_material(spec: mujoco.MjSpec) -> None:
    """Apply the moderately compliant FullHandMCC deployment material."""
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            continue
        geom.solref[:] = CONTACT_SOLREF
        geom.solimp[:] = CONTACT_SOLIMP
        geom.margin = 0.0
        geom.gap = 0.0
        geom.priority = 10


def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos


def fingertip_force_3d(env: ManagerBasedRlEnv) -> torch.Tensor:
    forces = []
    for site_name in MCC_TIP_NAMES:
        sensor = env.scene[f"{site_name}_contact"]
        assert isinstance(sensor, ContactSensor)
        force = sensor.data.force
        if force is None:
            forces.append(torch.zeros((env.num_envs, 3), device=env.device))
            continue
        if sensor.data.found is not None:
            force = torch.where(
                sensor.data.found.unsqueeze(-1) > 0,
                force,
                torch.zeros_like(force),
            )
        forces.append(force.sum(dim=1))
    force_world = torch.stack(forces, dim=1)
    robot = env.scene["robot"]
    names = [site.name or "" for site in robot.data.indexing.sites]
    indices = [
        next(
            index
            for index, name in enumerate(names)
            if name == wanted or name.endswith(f"/{wanted}")
        )
        for wanted in MCC_TIP_NAMES
    ]
    quat = robot.data.site_pose_w[:, indices, 3:7]
    return quat_apply_inverse(
        quat.reshape(-1, 4), force_world.reshape(-1, 3)
    ).reshape(env.num_envs, 12)


def _hand_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(HAND_XML))
    for exclude in list(spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            spec.delete(exclude)
    # Match the data-collection environment exactly.  Leaving the tactile XML
    # defaults here would make replay joint damping about 33x larger and
    # friction loss about 1000x larger than the teacher environment.
    for joint in spec.joints:
        joint_name = joint.name or ""
        if joint_name.isdigit() and 0 <= int(joint_name) < 16:
            joint.damping = (0.03, 0.0, 0.0)
            joint.frictionloss = 0.001
    existing = {site.name for site in spec.sites}
    for body_name, site_name, site_pos in zip(
        MCC_TIP_BODY_NAMES, MCC_TIP_NAMES, MCC_TIP_SITE_LOCAL_POSITIONS
    ):
        if site_name not in existing:
            spec.body(body_name).add_site(
                name=site_name,
                pos=site_pos,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.004, 0.0, 0.0),
                rgba=(1.0, 0.8, 0.1, 0.8),
            )
    _apply_contact_material(spec)
    return spec


def _target_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    body.add_geom(
        name="target_capsule_medium_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(0.15, 0.08, 0.0),
        rgba=(0.2, 0.6, 1.0, 1.0),
        mass=1.0,
    )
    _apply_contact_material(spec)
    return spec


def _sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    sensors: list[ContactSensorCfg] = []
    for site_name, geom_name in zip(MCC_TIP_NAMES, MCC_TIP_GEOM_NAMES):
        primary = ContactMatch(
            mode="geom", pattern=f"^{geom_name}$", entity="robot"
        )
        secondary = ContactMatch(
            mode="body", pattern="^target_ball$", entity="target"
        )
        sensors.append(
            ContactSensorCfg(
                name=f"{site_name}_contact",
                primary=primary,
                secondary=secondary,
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
            )
        )
        sensors.append(
            ContactSensorCfg(
                name=f"{site_name}_geometry_contact",
                primary=primary,
                secondary=secondary,
                fields=("found", "dist", "pos", "normal", "tangent"),
                reduce="maxforce",
                num_slots=1,
            )
        )
    return tuple(sensors)


def replay_env_cfg(
    hand_stiffness: float = 8.0,
    hand_damping: float = 1.3,
    hand_effort_limit: float = 12.0,
    thumb_stiffness: float | None = None,
    thumb_damping: float | None = None,
    thumb_effort_limit: float | None = None,
) -> ManagerBasedRlEnvCfg:
    thumb_stiffness = (
        hand_stiffness if thumb_stiffness is None else thumb_stiffness
    )
    thumb_damping = hand_damping if thumb_damping is None else thumb_damping
    thumb_effort_limit = (
        hand_effort_limit
        if thumb_effort_limit is None
        else thumb_effort_limit
    )
    robot = EntityCfg(
        spec_fn=_hand_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(?:[0-9]|1[01])$",),
                    # Keep the joint servo compliant enough that the 100 Hz
                    # normal-force loop can retreat before a tracking residual
                    # becomes a contact impulse.  (8, 1.3, 12) is the strict
                    # force-safety validation setting.
                    stiffness=float(hand_stiffness),
                    damping=float(hand_damping),
                    effort_limit=float(hand_effort_limit),
                    armature=0.0,
                    frictionloss=0.001,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^1[2-5]$",),
                    stiffness=float(thumb_stiffness),
                    damping=float(thumb_damping),
                    effort_limit=float(thumb_effort_limit),
                    armature=0.0,
                    frictionloss=0.001,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"13": 1.57}),
    )
    target = EntityCfg(spec_fn=_target_spec, init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 0)))
    observations = {
        "policy": ObservationGroupCfg(
            {
                "fingertip_force_3d": ObservationTermCfg(func=fingertip_force_3d),
                "joint_pos": ObservationTermCfg(
                    func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
                ),
            }
        )
    }
    actions: dict[str, ActionTermCfg] = {
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.08,
            use_default_offset=False,
        )
    }
    return ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
            terrain=None,
            entities={"robot": robot, "target": target},
            sensors=_sensor_cfgs(),
            num_envs=1,
            env_spacing=2.0,
        ),
        observations=observations,
        actions=actions,
        rewards={},
        terminations={},
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                gravity=(0.0, 0.0, -9.81),
                ccd_iterations=200,
                solver="newton",
            ),
            njmax=1000,
            nconmax=500,
        ),
        viewer=ViewerConfig(entity_name="robot", body_name="palm_lower", distance=0.8),
        episode_length_s=1e10,
    )


def _find_site_index(env: ManagerBasedRlEnv, suffix: str) -> int:
    for index, site in enumerate(env.scene["robot"].data.indexing.sites):
        name = site.name or ""
        if name == suffix or name.endswith(f"/{suffix}"):
            return index
    raise ValueError(f"Site {suffix!r} not found")


def replay(
    path: Path,
    episode_id: int,
    viewer: Literal["headless", "native", "viser"],
    mode: Literal["teacher"],
    device: str | None,
    max_steps: int,
    contact_threshold: float,
    state_velocity: Literal["zero", "trajectory"],
) -> None:
    if mode != "teacher":
        raise ValueError("v1 replay only implements exact teacher geometry")
    with h5py.File(path, "r") as file:
        ids = np.asarray(file["episode_id"], dtype=np.int64)
        locations = np.argwhere(ids == episode_id)
        if locations.size == 0:
            available = np.unique(ids)
            raise ValueError(
                f"episode_id={episode_id} not found; available range="
                f"[{available.min()}, {available.max()}]"
            )
        time_indices = locations[:, 0]
        env_indices = locations[:, 1]

        def episode(name: str) -> np.ndarray:
            dataset = file[name]
            return np.stack(
                [dataset[t, e] for t, e in zip(time_indices, env_indices)], axis=0
            ).astype(np.float32)

        palm = episode("palm_pose_object")
        palm_twist = episode("palm_twist_object")
        q_hand = episode("q_hand")
        tip_teacher = episode("fingertip_pose_object")
        control_dt = float(file.attrs.get("control_dt", 0.01))
    qvel_hand = np.zeros_like(q_hand)
    qvel_hand[1:] = (q_hand[1:] - q_hand[:-1]) / control_dt
    if len(qvel_hand) > 1:
        qvel_hand[0] = qvel_hand[1]
    frames = len(palm) if max_steps <= 0 else min(len(palm), max_steps)

    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=replay_env_cfg(), device=run_device)
    wrapped = RslRlVecEnvWrapper(env)
    robot = env.scene["robot"]
    tip_indices = [_find_site_index(env, name) for name in MCC_TIP_NAMES]

    class ReplayPolicy:
        def __init__(self):
            self.frame = 0
            self.contact_frames = 0
            self.contact3_frames = 0
            self.found3_frames = 0
            self.found4_frames = 0
            self.per_tip_found = np.zeros(4, dtype=np.int64)
            self.per_tip_loaded = np.zeros(4, dtype=np.int64)
            self.force_max = 0.0
            self.tip_errors: list[float] = []

        def __call__(self, _obs):
            t = self.frame % frames
            pose = torch.as_tensor(palm[t], device=env.device)
            root_velocity = (
                torch.as_tensor(palm_twist[t], device=env.device)
                if state_velocity == "trajectory"
                else torch.zeros(6, device=env.device)
            )
            root_state = torch.cat((pose, root_velocity)).unsqueeze(0)
            robot.write_root_state_to_sim(root_state)
            q = torch.as_tensor(q_hand[t], device=env.device).unsqueeze(0)
            qvel = (
                torch.as_tensor(qvel_hand[t], device=env.device).unsqueeze(0)
                if state_velocity == "trajectory"
                else torch.zeros_like(q)
            )
            robot.write_joint_state_to_sim(position=q, velocity=qvel)
            if env.sim.model.nmocap:
                env.sim.data.mocap_pos[:, 0, :] = 0.0
                env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
                    (1.0, 0.0, 0.0, 0.0), device=env.device
                )
            env.sim.forward()
            found = np.zeros(4, dtype=bool)
            for finger, site_name in enumerate(MCC_TIP_NAMES):
                sensor = env.scene[f"{site_name}_contact"]
                sensor.update(0.0)
                if sensor.data.found is not None:
                    found[finger] = bool((sensor.data.found[0] > 0).any())
            force = fingertip_force_3d(env).reshape(1, 4, 3)[0]
            magnitude = torch.linalg.vector_norm(force, dim=-1)
            loaded = magnitude.cpu().numpy() >= contact_threshold
            found_count = int(found.sum())
            loaded_count = int(loaded.sum())
            self.contact_frames += int(loaded_count > 0)
            self.contact3_frames += int(loaded_count >= 3)
            self.found3_frames += int(found_count >= 3)
            self.found4_frames += int(found_count == 4)
            self.per_tip_found += found.astype(np.int64)
            self.per_tip_loaded += loaded.astype(np.int64)
            self.force_max = max(self.force_max, float(magnitude.max()))

            live_pose = robot.data.site_pose_w[0]
            live_tip = torch.stack([live_pose[index, :3] for index in tip_indices]).cpu().numpy()
            self.tip_errors.append(float(np.linalg.norm(live_tip - tip_teacher[t, :, :3], axis=-1).mean()))
            if self.frame % 100 == 0:
                print(
                    f"[REPLAY] frame={t:4d} found={found_count}/4 "
                    f"loaded={loaded_count}/4 "
                    f"forces={magnitude.cpu().numpy().round(3).tolist()} "
                    f"tip_err={self.tip_errors[-1] * 1000:.3f}mm"
                )
            self.frame += 1
            return torch.zeros((1, 16), device=env.device)

    policy = ReplayPolicy()
    try:
        if viewer == "headless":
            for _ in range(frames):
                action = policy(wrapped.get_observations())
                wrapped.step(action)
            errors = np.asarray(policy.tip_errors)
            print(
                f"[RESULT] frames={frames} any_contact={100*policy.contact_frames/frames:.1f}% "
                f"loaded3={100*policy.contact3_frames/frames:.1f}% "
                f"found3={100*policy.found3_frames/frames:.1f}% "
                f"found4={100*policy.found4_frames/frames:.1f}% "
                f"tip_found={np.round(100*policy.per_tip_found/frames,1).tolist()}% "
                f"tip_loaded={np.round(100*policy.per_tip_loaded/frames,1).tolist()}% "
                f"force_max={policy.force_max:.3f}N "
                f"tip_error_mean={errors.mean()*1000:.3f}mm "
                f"tip_error_p95={np.percentile(errors,95)*1000:.3f}mm"
            )
        elif viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        else:
            ViserPlayViewer(wrapped, policy).run()
    finally:
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay object-frame MCC fingertip data.")
    parser.add_argument("--file", default=None)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--viewer", choices=("headless", "native", "viser"), default="native")
    parser.add_argument("--mode", choices=("teacher",), default="teacher")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument(
        "--state-velocity",
        choices=("zero", "trajectory"),
        default="zero",
        help="Replay zero velocity or the trajectory's palm twist and finite-difference qdot.",
    )
    args = parser.parse_args()
    if args.file:
        path = Path(args.file)
    else:
        candidates = glob.glob("mcc_finger_compliance_control/data/inverted/*_inverted.h5")
        if not candidates:
            raise FileNotFoundError("No inverted trajectory H5 found")
        path = Path(max(candidates, key=lambda item: Path(item).stat().st_mtime))
    replay(
        path,
        args.episode_id,
        args.viewer,
        args.mode,
        args.device,
        args.max_steps,
        args.contact_threshold,
        args.state_velocity,
    )


if __name__ == "__main__":
    main()
