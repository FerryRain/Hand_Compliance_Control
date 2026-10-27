"""Shared LEAP Hand primitives for Baseline-2 direct-force control.

This module owns the simulation contract used by the FR3 + LEAP Hand
Baseline-2 controller: four contact sensors measure the resultant force on
the four visible fingertip geoms, and those measurements are exposed as four
3-D vectors in the corresponding fingertip-site frames.  It deliberately
contains no legacy finger controller, policy, or surface planner.

The environment factory is robot-agnostic.  The caller supplies the robot,
target, and action configurations, which keeps the FR3 task independent of
the retired xArm finger and palm environments.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer import ViewerConfig


LEAP_HAND_TACTILE_XML = (
    Path(__file__).resolve().parents[2]
    / "asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml"
)

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
MCC_NON_TIP_HAND_GEOM_PATTERN = (
    r"^(?:palm_lower_collision|"
    r"mcp_joint(?:_[23])?_geom|"
    r"pip(?:_[234])?_geom|"
    r"dip(?:_[23])?_geom|"
    r"thumb_(?:pip|dip)_geom)$"
)
MCC_TIP_SITE_LOCAL_POSITIONS = (
    (-0.0106151, -0.0326103, 0.0141088),
    (-0.0106151, -0.0326103, 0.0144487),
    (-0.0106151, -0.0326103, 0.0140386),
    (-0.0106383, -0.0453895, -0.0144321),
)

# Conservative nominal LEAP pose used to initialize the direct-force loop.
DEFAULT_PREGRASP_Q = (
    0.85,
    0.00,
    0.45,
    0.55,
    0.85,
    0.00,
    0.45,
    0.55,
    0.85,
    0.00,
    0.45,
    0.55,
    0.85,
    1.57,
    0.45,
    0.55,
)

# MuJoCo has no perfectly rigid, zero-penetration constraint.  These values
# are the validated high-priority approximation used on the object surface.
HARD_CONTACT_SOLREF = (-20_000.0, -400.0)
HARD_CONTACT_SOLIMP = (0.90, 0.98, 0.001, 0.5, 2.0)
HARD_CONTACT_MARGIN = 0.0


def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return absolute robot joint positions."""

    return env.scene[asset_cfg.name].data.joint_pos


def _apply_hard_contact(geom: mujoco.MjsGeom) -> None:
    if geom.contype == 0 and geom.conaffinity == 0:
        return
    geom.solref[:] = HARD_CONTACT_SOLREF
    geom.solimp[:] = HARD_CONTACT_SOLIMP
    geom.margin = HARD_CONTACT_MARGIN
    geom.gap = 0.0
    geom.priority = 10


def make_hard_contact_target_spec() -> mujoco.MjSpec:
    """Create the high-priority capsule target used by the full-hand task."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    body.add_geom(
        name="target_capsule_medium_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(0.15, 0.08, 0.0),
        rgba=(0.2, 0.6, 1.0, 1.0),
        mass=1.0,
    )
    for geom in spec.geoms:
        _apply_hard_contact(geom)
    return spec


def load_fixed_palm_direct_force_hand_spec() -> mujoco.MjSpec:
    """Build the fixed-palm 16-DoF model used by fingertip Cartesian IK."""

    spec = mujoco.MjSpec.from_file(str(LEAP_HAND_TACTILE_XML))
    for exclude in list(spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            spec.delete(exclude)
    palm_free_joint = spec.joint("palm_base")
    if palm_free_joint is not None:
        spec.delete(palm_free_joint)
    palm = spec.body("palm_lower")
    if palm is None:
        raise ValueError(f"palm_lower is missing from {LEAP_HAND_TACTILE_XML}")
    palm.pos[:] = (0.0, 0.0, 0.0)
    palm.quat[:] = (1.0, 0.0, 0.0, 0.0)
    existing_sites = {site.name for site in spec.sites}
    for body_name, site_name, site_pos in zip(
        MCC_TIP_BODY_NAMES,
        MCC_TIP_NAMES,
        MCC_TIP_SITE_LOCAL_POSITIONS,
        strict=True,
    ):
        if site_name in existing_sites:
            continue
        body = spec.body(body_name)
        if body is None:
            raise ValueError(
                f"Fingertip body {body_name!r} is missing from "
                f"{LEAP_HAND_TACTILE_XML}"
            )
        body.add_site(
            name=site_name,
            pos=site_pos,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.004, 0.0, 0.0),
            rgba=(1.0, 0.8, 0.1, 0.8),
        )
    return spec


def direct_force_contact_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    """Return four tip-force sensors and independent collision audit sensors."""

    tip_sensors = tuple(
        ContactSensorCfg(
            name=f"{site_name}_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=f"^{geom_name}$",
                entity="robot",
            ),
            secondary=ContactMatch(
                mode="body",
                pattern="^target_ball$",
                entity="target",
            ),
            fields=("found", "force", "dist", "pos", "normal", "tangent"),
            reduce="netforce",
            num_slots=1,
        )
        for site_name, geom_name in zip(
            MCC_TIP_NAMES,
            MCC_TIP_GEOM_NAMES,
            strict=True,
        )
    )
    arm_object_guard = ContactSensorCfg(
        name="arm_object_collision",
        primary=ContactMatch(
            mode="geom",
            pattern=(
                r"^(?:base_collision|link[1-6]_collision|"
                r"fr3v2_link[0-7]_collision)$"
            ),
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern=r"^target_ball$",
            entity="target",
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="mindist",
        num_slots=1,
    )
    incidental_hand_primary = ContactMatch(
        mode="geom",
        pattern=MCC_NON_TIP_HAND_GEOM_PATTERN,
        entity="robot",
    )
    incidental_hand_secondary = ContactMatch(
        mode="body",
        pattern=r"^target_ball$",
        entity="target",
    )
    incidental_hand_depth = ContactSensorCfg(
        name="incidental_hand_object_contact_depth",
        primary=incidental_hand_primary,
        secondary=incidental_hand_secondary,
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="mindist",
        num_slots=1,
    )
    incidental_hand_force = ContactSensorCfg(
        name="incidental_hand_object_contact_force",
        primary=ContactMatch(
            mode="geom",
            pattern=MCC_NON_TIP_HAND_GEOM_PATTERN,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern=r"^target_ball$",
            entity="target",
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="maxforce",
        num_slots=1,
    )
    return (
        *tip_sensors,
        arm_object_guard,
        incidental_hand_depth,
        incidental_hand_force,
    )


def fingertip_force_3d_world(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return exact 3-D resultants on the four fingertip geoms in world axes."""

    forces: list[torch.Tensor] = []
    for site_name in MCC_TIP_NAMES:
        sensor = env.scene[f"{site_name}_contact"]
        if not isinstance(sensor, ContactSensor):
            raise TypeError(
                f"{site_name}_contact must be ContactSensor, "
                f"got {type(sensor).__name__}"
            )
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
    return torch.cat(forces, dim=-1)


def fingertip_force_3d(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return four direct tip forces expressed in their local site frames."""

    force_world = fingertip_force_3d_world(env).reshape(env.num_envs, 4, 3)
    robot = env.scene["robot"]
    site_names = [site.name or "" for site in robot.data.indexing.sites]
    site_indices: list[int] = []
    for wanted in MCC_TIP_NAMES:
        matches = [
            index
            for index, name in enumerate(site_names)
            if name == wanted or name.endswith(f"/{wanted}")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one site {wanted!r}, got {matches}")
        site_indices.append(matches[0])
    site_quat = robot.data.site_pose_w[:, site_indices, 3:7]
    return quat_apply_inverse(
        site_quat.reshape(-1, 4),
        force_world.reshape(-1, 3),
    ).reshape(env.num_envs, 12)


def direct_force_contact_env_cfg(
    *,
    robot_cfg: EntityCfg,
    target_cfg: EntityCfg,
    actions: dict[str, ActionTermCfg],
    num_envs: int = 1,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic direct-fingertip-force environment shell."""

    robot = SceneEntityCfg("robot")
    observations = {
        "palm": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=joint_pos,
                    params={"asset_cfg": robot},
                ),
            }
        ),
        "finger": ObservationGroupCfg(
            {
                "fingertip_force_3d": ObservationTermCfg(func=fingertip_force_3d),
                "joint_pos": ObservationTermCfg(
                    func=joint_pos,
                    params={"asset_cfg": robot},
                ),
            }
        ),
    }
    return ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=direct_force_contact_sensor_cfgs(),
            num_envs=num_envs,
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
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="palm_lower",
            distance=1.2,
        ),
        episode_length_s=1e10 if play else 50.0,
    )
