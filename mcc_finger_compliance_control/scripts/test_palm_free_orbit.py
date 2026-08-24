#!/usr/bin/env python3
"""Standalone palm-free orbit test for the LEAP hand (mjlab).

Inverted-space layout (replay_inverted.py convention):
  * the object sits at the world origin (mocap pinned every step),
  * the palm is a mocap-driven RIGID body -- the palm_base free joint is
    deleted and the palm body is mocap, so the teacher writes
    mocap_pos/mocap_quat each step and the palm is held EXACTLY at the
    commanded world pose.  Contact reactions cannot shove the hand around
    (the palm-free equivalent of the arm holding the palm rigidly),
  * the fingers keep their 16-DoF dynamics with the soft pregrasp servo
    (stiffness 5, effort limit 10) and slide passively over the surface,
  * the initial pose puts the palm NEXT TO the object at the surface
    clearance with the palm normal facing the object centre.

Motion: ``FacingCenterOrbitController`` (imported from collect_trajectories.py)
-- one-way phase advance, palm position on the object's elliptical
cross-section at constant surface clearance, orientation rotated about the
long axis so the palm normal keeps facing the object centre.

Reference pose (object frame): orbit_pinch_verify4.h5 frame 0, all four
tips loaded with 4.1-10.6 N -- the palm hovers at the surface clearance
with the fingertips on the bottle.

Metrics reported:
  * four-tip contact fraction (all fingertips loaded)
  * palm clearance constancy (radial distance - ellipse radius vs target)
  * palm normal-facing error (angle between facing axis and object centre)
  * one-way phase advance (no triangle-wave reflection)

Usage (headless smoke):
  python mcc_finger_compliance_control/scripts/test_palm_free_orbit.py

Usage (live visualization, user check):
  python mcc_finger_compliance_control/scripts/test_palm_free_orbit.py --viewer native
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    DEFAULT_PREGRASP_Q,
    MCC_TIP_GEOM_NAMES,
    MCC_TIP_NAMES,
    _get_hard_contact_target_spec,
    _load_palm_free_leaphand_spec,
    fingertip_force_3d_world,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import NativeMujocoViewer, ViewerConfig

from collect_trajectories import FacingCenterOrbitController
from object_catalog import load_object_config, object_local_aabb

TIP_NAMES = MCC_TIP_NAMES

# Reference palm pose in the object frame (object at the origin with the
# yaml rotation): orbit_pinch_verify4.h5 frame 0, all four tips loaded
# (4.1-10.6 N).  The palm hovers at the surface clearance beside the bottle
# with the palm normal facing the object centre.
REF_PALM_POS_OBJECT = (0.170055, -0.028994, 0.009911)
REF_PALM_ROTVEC_OBJECT = (2.230724, 0.004096, 2.208227)
# Finger configuration at the same frame (frame 0, all four tips loaded
# 4.1-10.6 N).  The pregrasp servo targets are this contact posture -- with
# a rigid palm the pregrasp FK would bury the fingertips deep into the
# surface, so the servo target is the surface-adapted configuration.
REF_Q_HAND = (
    -0.3433, 0.2563, 0.6403, 0.8118,
    -0.3587, 0.0460, 0.7832, 0.8471,
    0.0307, -0.2482, 0.8539, 0.8681,
    0.0147, 1.3001, 0.7279, 0.8023,
)


def _palm_mocap_spec() -> mujoco.MjSpec:
    """Hand-only spec with the palm as a mocap-driven rigid body.

    Deletes the ``palm_base`` free joint and marks the palm body as mocap:
    the palm's world pose is then fully kinematic (mocap_pos/mocap_quat),
    while the finger joints keep their dynamics and soft servos.
    """
    spec = _load_palm_free_leaphand_spec()
    for joint in list(spec.joints):
        if joint.name == "palm_base":
            spec.delete(joint)
    base = spec.body("palm_lower")
    if base is None:
        raise ValueError("palm_lower body not found in the hand-only XML")
    base.mocap = True
    return spec


def _full_qpos_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Full entity qpos (16 hand joints; the palm is mocap-driven)."""
    return env.scene[asset_cfg.name].data.data.qpos


def build_palm_free_env_cfg(
    object_id: str,
    object_scale: float,
    init_hand_q: tuple[float, ...],
    num_envs: int,
    viewer: str,
) -> ManagerBasedRlEnvCfg:
    """Hand-only env: mocap-rigid palm, dynamic fingers, object at origin."""
    object_config = load_object_config(object_id)
    hand_init = {
        str(index): float(value) for index, value in enumerate(init_hand_q)
    }
    robot_cfg = EntityCfg(
        spec_fn=_palm_mocap_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=5.0,
                    damping=0.5,
                    effort_limit=10.0,
                    armature=0.0,
                    frictionloss=0.001,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos=hand_init,
        ),
    )
    target_cfg = EntityCfg(
        spec_fn=lambda: _get_hard_contact_target_spec(object_config, object_scale),
        # Inverted space: the object sits at the world origin.
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=object_config.initial_rot,
        ),
    )
    # Fingertip contact sensors only -- the arm_object_guard in the arm
    # version's _tip_sensor_cfgs() references xArm geoms that do not exist
    # in the hand-only model.
    tip_sensors = tuple(
        ContactSensorCfg(
            name=f"{site}_contact",
            primary=ContactMatch(mode="geom", pattern=f"^{geom}$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern=r"^target_ball$", entity="target"),
            fields=("found", "force", "dist", "pos", "normal", "tangent"),
            reduce="netforce",
            num_slots=1,
        )
        for site, geom in zip(MCC_TIP_NAMES, MCC_TIP_GEOM_NAMES)
    )
    observations = {
        "palm": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=_full_qpos_obs,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
            }
        ),
        "finger": ObservationGroupCfg(
            {
                "fingertip_force_3d": ObservationTermCfg(func=fingertip_force_3d_world),
                "joint_pos": ObservationTermCfg(
                    func=_full_qpos_obs,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
            }
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^[0-9]+$",),
            scale=0.08,
            use_default_offset=False,
        ),
    }
    return ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=tip_sensors,
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
        viewer=ViewerConfig(entity_name="robot", body_name="palm_lower", distance=1.2),
        episode_length_s=1e10,
    )


class PalmFreeOrbitTeacher:
    """Drives the mocap-rigid palm on the facing-centre orbit.

    Each step: pin the object mocap at the origin, advance the orbit
    (contact-gated), write the commanded pose into the palm's mocap pose,
    and hold the fingers at their pregrasp soft-servo targets (zero hand
    action => default offset = pregrasp).  Metrics are accumulated for the
    final report.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        orbit: FacingCenterOrbitController,
        contact_threshold: float,
        gate: bool,
        device: str,
        object_quat_wxyz: tuple[float, float, float, float],
        palm_mocap_idx: int,
    ) -> None:
        self.env = env
        self.orbit = orbit
        self.robot = env.scene["robot"]
        self.contact_threshold = float(contact_threshold)
        self.gate = bool(gate)
        self.device = device
        self.object_quat_wxyz = object_quat_wxyz
        self.palm_mocap_idx = int(palm_mocap_idx)
        self.step_count = 0
        # Lazily derived at capture: which palm-local axis faces the object
        # centre in the reference pose.
        self.facing_local: np.ndarray | None = None
        # Metric buffers.
        self.n_frames = 0
        self.loaded_frames = np.zeros(4, dtype=np.int64)
        self.all4_frames = 0
        self.motion_frames = 0
        self.clearance_errors: list[float] = []
        self.normal_errors: list[float] = []
        self.dphase: list[float] = []
        self.tracking_errors: list[float] = []
        self.travel_rad = 0.0

    def _write_palm_mocap(self, pose: torch.Tensor) -> None:
        """Write a (B,6) world pos+rotvec into the palm mocap."""
        pose_np = pose.detach().cpu().numpy()
        quat_xyzw = R.from_rotvec(pose_np[:, 3:6]).as_quat()
        sim = self.env.sim.data
        sim.mocap_pos[:, self.palm_mocap_idx, :] = torch.as_tensor(
            pose_np[:, :3], device=self.device
        )
        sim.mocap_quat[:, self.palm_mocap_idx, :] = torch.as_tensor(
            quat_xyzw[:, [3, 0, 1, 2]], device=self.device
        )

    def reset(self) -> None:
        self.orbit.reset()
        self.step_count = 0
        self.facing_local = None
        self.n_frames = 0
        self.loaded_frames.fill(0)
        self.all4_frames = 0
        self.motion_frames = 0
        self.clearance_errors.clear()
        self.normal_errors.clear()
        self.dphase.clear()
        self.tracking_errors.clear()
        self.travel_rad = 0.0
        # Put the palm at the reference pose right away so the first
        # rendered frame shows the hand beside the object, not at the
        # origin.
        self._write_palm_mocap(self.orbit.current_x_des)

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        forces = fingertip_force_3d_world(self.env).reshape(
            self.env.num_envs, 4, 3
        )
        loaded = (
            torch.linalg.vector_norm(forces, dim=-1) >= self.contact_threshold
        )
        contact_ready = None if not self.gate else loaded.all(dim=-1)
        phase_before = self.orbit.phase.clone()

        # Inverted space: pin the object mocap at the origin every step.
        self.env.sim.data.mocap_pos[:, self.orbit.mocap_idx, :] = 0.0
        self.env.sim.data.mocap_quat[:, self.orbit.mocap_idx, :] = (
            torch.as_tensor(
                self.object_quat_wxyz, device=self.device, dtype=torch.float32
            )
        )

        moved = self.orbit.step(self.step_count, contact_ready=contact_ready)
        self.step_count += 1

        if moved:
            self.motion_frames += 1
            dth = (self.orbit.phase - phase_before).detach().cpu().numpy()
            self.dphase.extend(dth[dth > 0].tolist())
            self.travel_rad = float(
                (self.orbit.phase - self.orbit.theta0)[0].detach().cpu()
            )

        # Hold the palm exactly at the commanded pose (mocap-rigid).
        self._write_palm_mocap(self.orbit.current_x_des)
        # Fingers hold pregrasp targets: zero delta on the relative action.
        action = torch.zeros((self.env.num_envs, 16), device=self.device)
        return action

    def observe(self, obs: dict[str, torch.Tensor]) -> None:
        """Post-step metrics from the fresh observations."""
        forces = fingertip_force_3d_world(self.env).reshape(
            self.env.num_envs, 4, 3
        )
        loaded = (
            torch.linalg.vector_norm(forces, dim=-1)
            >= self.contact_threshold
        )
        self.n_frames += 1
        self.loaded_frames += loaded[0].detach().cpu().numpy().astype(np.int64)
        self.all4_frames += int(bool(torch.all(loaded[0])))

        sim = self.env.sim.data
        palm_pos = (
            sim.mocap_pos[0, self.palm_mocap_idx, :].detach().cpu().numpy()
        )
        palm_quat = (
            sim.mocap_quat[0, self.palm_mocap_idx, :].detach().cpu().numpy()
        )  # wxyz
        palm_rotmat = R.from_quat(
            (palm_quat[1], palm_quat[2], palm_quat[3], palm_quat[0])
        ).as_matrix()
        center = (
            sim.mocap_pos[0, self.orbit.mocap_idx, :].detach().cpu().numpy()
        )
        axis = self.orbit.orbit_axis_world[0].detach().cpu().numpy()

        # Tracking error: commanded vs actual palm position.
        cmd_pos = self.orbit.current_x_des[0, :3].detach().cpu().numpy()
        self.tracking_errors.append(float(np.linalg.norm(cmd_pos - palm_pos)))

        # Clearance: |radial| - ellipse radius along the actual direction.
        offset = palm_pos - center
        radial = offset - axis * float(offset @ axis)
        n_actual = radial / (np.linalg.norm(radial) + 1e-9)
        r_surf = float(
            self.orbit._ellipse_radius(
                torch.as_tensor(n_actual, device=self.device)[None]
            )[0]
        )
        clearance = float(np.linalg.norm(radial)) - r_surf
        self.clearance_errors.append(clearance)

        # Normal-facing error: which palm-local axis faces the object centre.
        if self.facing_local is None:
            best = None
            best_align = -1.0
            for axis_idx in range(3):
                for sign in (1.0, -1.0):
                    local = np.zeros(3)
                    local[axis_idx] = sign
                    world_dir = palm_rotmat @ local
                    align = float(world_dir @ (center - palm_pos)) / (
                        np.linalg.norm(center - palm_pos) + 1e-9
                    )
                    if align > best_align:
                        best_align = align
                        best = local
            self.facing_local = best
        facing_world = palm_rotmat @ self.facing_local
        to_center = center - palm_pos
        cos_angle = float(facing_world @ to_center) / (
            np.linalg.norm(facing_world) * np.linalg.norm(to_center) + 1e-9
        )
        self.normal_errors.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))))

    def report(self, surface_clearance_target: float | None) -> str:
        n = max(self.n_frames, 1)
        ce = np.asarray(self.clearance_errors)
        ne = np.asarray(self.normal_errors)
        te = np.asarray(self.tracking_errors)
        dp = np.asarray(self.dphase)
        lines = [
            f"[PALM-FREE-TEST] frames={self.n_frames} "
            f"all4_loaded={self.all4_frames / n:.1%} "
            f"per_tip={[(self.loaded_frames[i] / n).round(3) for i in range(4)]}",
            f"[PALM-FREE-TEST] motion_frames={self.motion_frames} "
            f"travel={np.degrees(self.travel_rad):.1f} deg "
            f"one_way={bool((dp >= -1e-6).all()) if dp.size else 'n/a'} "
            f"dphase_min={dp.min():.5f} rad" if dp.size else
            f"[PALM-FREE-TEST] motion_frames={self.motion_frames} (no motion ran)",
            (
                f"[PALM-FREE-TEST] palm tracking |cmd-actual| max="
                f"{te.max() * 1000:.2f} mm"
            ) if te.size else
            "[PALM-FREE-TEST] palm tracking |cmd-actual| n/a (no frames)",
        ]
        if ce.size:
            target = (
                surface_clearance_target
                if surface_clearance_target is not None
                else float(self.orbit.surface_clearance[0].detach().cpu())
            )
            lines.append(
                f"[PALM-FREE-TEST] clearance target={target * 1000:.1f} mm "
                f"actual mean={ce.mean() * 1000:.1f} std={ce.std() * 1000:.2f} "
                f"max_dev={np.abs(ce - target).max() * 1000:.2f} mm"
            )
        if ne.size:
            lines.append(
                f"[PALM-FREE-TEST] normal-facing error "
                f"mean={ne.mean():.2f} deg max={ne.max():.2f} deg "
                f"(facing axis={self.facing_local})"
            )
        return "\n".join(lines)


class _ViewerPolicy:
    """Minimal NativeMujocoViewer policy wrapper around the teacher."""

    def __init__(self, teacher: PalmFreeOrbitTeacher, env) -> None:
        self.teacher = teacher
        self.env = env

    def reset(self) -> None:
        self.teacher.reset()

    def __call__(self, obs) -> torch.Tensor:
        action = self.teacher(obs)
        self.teacher.observe(obs)
        return action


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object-id", default="ycb_mustard")
    ap.add_argument("--object-scale", type=float, default=None,
                    help="fixed scale (default: sampled from the object yaml range)")
    ap.add_argument("--viewer", choices=("headless", "native"), default="headless")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--motion-start", type=int, default=200)
    ap.add_argument("--motion-length", type=int, default=900)
    ap.add_argument("--speed-min", type=float, default=0.04)
    ap.add_argument("--speed-max", type=float, default=0.07)
    ap.add_argument("--surface-clearance-m", type=float, default=None,
                    help="palm hover distance from the object surface in m "
                         "(default: derived from the reference pose)")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--gate", action="store_true", default=True,
                    help="gate orbit motion on all-four-tip contact")
    ap.add_argument("--no-gate", action="store_false", dest="gate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--viewer-fps", type=float, default=60.0)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--debug-tips", action="store_true",
                    help="print per-tip force/dist in the periodic status line")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    object_config = load_object_config(args.object_id)
    if args.object_scale is None:
        scale_lo, scale_hi = object_config.collection["size_scale_range"]
        args.object_scale = float(
            np.random.default_rng(args.seed).uniform(scale_lo, scale_hi)
        )
    object_scale = args.object_scale
    lower, upper = object_local_aabb(object_config, scale=object_scale)
    object_extent = upper - lower

    # Reference palm pose in world coordinates: the object frame is rotated
    # by the yaml initial_rot in the env, so rotate the reference pose out.
    # With --surface-clearance-m, the pose is moved along the reference
    # radial direction to that clearance from the ellipse surface.
    obj_quat = np.asarray(object_config.initial_rot, dtype=np.float64)  # wxyz
    obj_rotmat = R.from_quat((*obj_quat[1:], obj_quat[0])).as_matrix()
    palm_pos0 = obj_rotmat @ np.asarray(REF_PALM_POS_OBJECT, dtype=np.float64)
    palm_rotvec0 = (
        R.from_matrix(obj_rotmat)
        * R.from_rotvec(np.asarray(REF_PALM_ROTVEC_OBJECT, dtype=np.float64))
    ).as_rotvec()
    if args.surface_clearance_m is not None:
        h = float(args.surface_clearance_m)
        long_idx = int(np.argmax(np.abs(object_extent)))
        axis_local = np.zeros(3)
        axis_local[long_idx] = 1.0
        axis_w = obj_rotmat @ axis_local
        ref_obj = np.asarray(REF_PALM_POS_OBJECT, dtype=np.float64)
        radial = ref_obj - axis_w * float(ref_obj @ axis_w)
        n0 = radial / np.linalg.norm(radial)
        ell = [i for i in range(3) if i != long_idx]
        s1 = obj_rotmat @ np.eye(3)[ell[0]]
        s2 = obj_rotmat @ np.eye(3)[ell[1]]
        half1 = 0.5 * abs(float(object_extent[ell[0]]))
        half2 = 0.5 * abs(float(object_extent[ell[1]]))
        n1 = float(n0 @ s1)
        n2 = float(n0 @ s2)
        r0 = half1 * half2 / np.sqrt((half2 * n1) ** 2 + (half1 * n2) ** 2)
        palm_pos0 = n0 * (r0 + h)
    print(
        f"[PALM-INIT] world pos={np.round(palm_pos0, 4)}m "
        f"rotvec={np.round(palm_rotvec0, 3)}rad"
    )

    env_cfg = build_palm_free_env_cfg(
        args.object_id, object_scale, REF_Q_HAND, args.num_envs, args.viewer
    )
    env_cfg.viewer.env_idx = 0
    print(
        f"[OBJECT] id={args.object_id} extent={np.round(object_extent, 3)}m "
        f"scale={object_scale:.3f}"
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    robot_mocap = env.scene["robot"].data.indexing.mocap_id
    palm_mocap_idx = int(
        robot_mocap[0] if hasattr(robot_mocap, "__len__") else robot_mocap
    )

    fixed_target = torch.as_tensor(
        [*palm_pos0, *palm_rotvec0],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).repeat(args.num_envs, 1)
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)
    orbit = FacingCenterOrbitController(
        env,
        fixed_target=fixed_target,
        target_mocap_idx=mocap_idx,
        object_extent_local=object_extent,
        angular_speed_min=args.speed_min,
        angular_speed_max=args.speed_max,
        surface_clearance_m=args.surface_clearance_m,
        motion_start=args.motion_start,
        motion_length=args.motion_length,
        dt=dt,
        device=device,
    )
    teacher = PalmFreeOrbitTeacher(
        env,
        orbit,
        contact_threshold=args.threshold,
        gate=args.gate,
        device=device,
        object_quat_wxyz=tuple(float(v) for v in object_config.initial_rot),
        palm_mocap_idx=palm_mocap_idx,
    )
    # Put the palm at the reference pose before any frame is rendered.
    teacher.reset()
    print(
        f"[MOTION] mode=palm_orbit one_way facing_center gate={'ON' if args.gate else 'OFF'} "
        f"window=steps{args.motion_start}+{args.motion_length} "
        f"speed=[{args.speed_min}, {args.speed_max}] rad/s "
        f"dt={dt:.4f} s"
    )

    if args.viewer == "native":
        wrapped = RslRlVecEnvWrapper(env)
        NativeMujocoViewer(
            wrapped,
            _ViewerPolicy(teacher, env),
            frame_rate=args.viewer_fps,
        ).run(num_steps=args.steps)
    else:
        obs, _ = env.reset()
        teacher.reset()
        for step in range(args.steps):
            action = teacher(obs)
            obs, _, _, _, _ = env.step(action)
            teacher.observe(obs)
            if (step + 1) % args.print_every == 0:
                loaded = teacher.all4_frames / max(teacher.n_frames, 1)
                line = (
                    f"[step {step + 1}/{args.steps}] all4={loaded:.1%} "
                    f"travel={np.degrees(teacher.travel_rad):.1f} deg"
                )
                if args.debug_tips:
                    parts = []
                    for tip in TIP_NAMES:
                        s = env.scene[f"{tip}_contact"].data
                        f = float(
                            np.linalg.norm(
                                np.asarray(s.force[0].detach().cpu())
                            )
                        )
                        d = float(
                            np.asarray(s.dist[0].detach().cpu()).ravel()[0]
                        )
                        parts.append(
                            f"{tip}:F={f * 1000:.0f}mN d={d * 1000:.1f}mm"
                        )
                    line += "  |  " + "  ".join(parts)
                print(line)

    print(teacher.report(args.surface_clearance_m))


if __name__ == "__main__":
    main()
