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
    return spec


def _sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    return tuple(
        ContactSensorCfg(
            name=f"{site_name}_contact",
            primary=ContactMatch(mode="geom", pattern=f"^{geom_name}$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="^target_ball$", entity="target"),
            fields=("found", "force", "dist", "pos", "normal", "tangent"),
            reduce="netforce",
            num_slots=1,
        )
        for site_name, geom_name in zip(MCC_TIP_NAMES, MCC_TIP_GEOM_NAMES)
    )


def _replay_env_cfg() -> ManagerBasedRlEnvCfg:
    robot = EntityCfg(
        spec_fn=_hand_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=500.0,
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
        q_hand = episode("q_hand")
        tip_teacher = episode("fingertip_pose_object")
    frames = len(palm) if max_steps <= 0 else min(len(palm), max_steps)

    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=_replay_env_cfg(), device=run_device)
    wrapped = RslRlVecEnvWrapper(env)
    robot = env.scene["robot"]
    tip_indices = [_find_site_index(env, name) for name in MCC_TIP_NAMES]

    class ReplayPolicy:
        def __init__(self):
            self.frame = 0
            self.contact_frames = 0
            self.contact3_frames = 0
            self.force_max = 0.0
            self.tip_errors: list[float] = []

        def __call__(self, _obs):
            t = self.frame % frames
            pose = torch.as_tensor(palm[t], device=env.device)
            root_state = torch.cat((pose, torch.zeros(6, device=env.device))).unsqueeze(0)
            robot.write_root_state_to_sim(root_state)
            q = torch.as_tensor(q_hand[t], device=env.device).unsqueeze(0)
            robot.write_joint_state_to_sim(position=q, velocity=torch.zeros_like(q))
            if env.sim.model.nmocap:
                env.sim.data.mocap_pos[:, 0, :] = 0.0
                env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
                    (1.0, 0.0, 0.0, 0.0), device=env.device
                )
            env.sim.forward()
            for site_name in MCC_TIP_NAMES:
                env.scene[f"{site_name}_contact"].update(0.0)
            force = fingertip_force_3d(env).reshape(1, 4, 3)[0]
            magnitude = torch.linalg.vector_norm(force, dim=-1)
            count = int((magnitude >= contact_threshold).sum())
            self.contact_frames += int(count > 0)
            self.contact3_frames += int(count >= 3)
            self.force_max = max(self.force_max, float(magnitude.max()))

            live_pose = robot.data.site_pose_w[0]
            live_tip = torch.stack([live_pose[index, :3] for index in tip_indices]).cpu().numpy()
            self.tip_errors.append(float(np.linalg.norm(live_tip - tip_teacher[t, :, :3], axis=-1).mean()))
            if self.frame % 100 == 0:
                print(
                    f"[REPLAY] frame={t:4d} contacts={count}/4 "
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
                f"contact3={100*policy.contact3_frames/frames:.1f}% "
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
    )


if __name__ == "__main__":
    main()
