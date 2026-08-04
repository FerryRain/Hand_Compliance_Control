"""Evaluate finger-only MCC in the object-fixed inverse environment.

The palm follows the recorded teacher pose.  After a short teacher warm-up,
the four finger commands come from an MCC controller; there is no DP model.
``--controller fullhand_admittance`` is the finger path used by full_hand_mcc:
surface plan -> normal admittance -> four-site Mink IK -> rate limit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from replay_inverted import replay_env_cfg
from surface_mcc_finger import (
    FullHandMCCFingerConfig,
    FullHandMCCFingerController,
    PrivilegedCapsuleSurfaceOracle,
    SurfaceMCCFingerConfig,
    SurfaceMCCFingerController,
    TIP_NAMES,
)


ACTION_SCALE = 0.08
TIP_LABELS = ("index", "middle", "ring", "thumb")


def live_tip_observation(
    env: ManagerBasedRlEnv,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Read force and geometry contact states independently for each tip.

    ``geometry_found`` is the raw collision-pair state and is the only state
    used for geometric coverage.  ``force_found`` belongs to the net-force
    sensor and is kept as a diagnostic: it must agree with geometry on a
    well-formed contact pair, but it must never decide geometry coverage.
    """
    forces = np.zeros((4, 3), dtype=np.float32)
    normals = np.zeros((4, 3), dtype=np.float32)
    positions = np.zeros((4, 3), dtype=np.float32)
    geometry_found_tips = np.zeros(4, dtype=bool)
    force_found_tips = np.zeros(4, dtype=bool)
    distances = np.zeros(4, dtype=np.float32)
    for tip_index, site_name in enumerate(TIP_NAMES):
        force_sensor = env.scene[f"{site_name}_contact"]
        geometry_sensor = env.scene[f"{site_name}_geometry_contact"]
        if not isinstance(force_sensor, ContactSensor) or not isinstance(
            geometry_sensor, ContactSensor
        ):
            raise TypeError((type(force_sensor), type(geometry_sensor)))
        force_sensor.update(0.0)
        geometry_sensor.update(0.0)
        force_data = force_sensor.data
        geometry_data = geometry_sensor.data
        force_found = (
            force_data.found is not None
            and bool((force_data.found[0] > 0).any())
        )
        geometry_found = (
            geometry_data.found is not None
            and bool((geometry_data.found[0] > 0).any())
        )
        force_found_tips[tip_index] = force_found
        geometry_found_tips[tip_index] = geometry_found
        if force_found and force_data.force is not None:
            found_slots = force_data.found[0] > 0
            forces[tip_index] = (
                torch.where(
                    found_slots[:, None],
                    force_data.force[0],
                    torch.zeros_like(force_data.force[0]),
                )
                .sum(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
        if geometry_found and geometry_data.normal is not None:
            normals[tip_index] = (
                geometry_data.normal[0, 0].detach().cpu().numpy()
            )
        if geometry_found and geometry_data.pos is not None:
            positions[tip_index] = (
                geometry_data.pos[0, 0].detach().cpu().numpy()
            )
        if geometry_found and geometry_data.dist is not None:
            distances[tip_index] = float(geometry_data.dist[0, 0])
    return (
        forces,
        normals,
        positions,
        geometry_found_tips,
        force_found_tips,
        distances,
    )


def _episode(file: h5py.File, episode_id: int, name: str) -> np.ndarray:
    ids = np.asarray(file["episode_id"], dtype=np.int64)
    locations = np.argwhere(ids == episode_id)
    if not locations.size:
        available = np.unique(ids)
        raise ValueError(
            f"episode_id={episode_id} not found; available IDs include "
            f"{available[:20].tolist()}"
        )
    steps = np.asarray(file["episode_step"])
    locations = locations[
        np.argsort([steps[t, e] for t, e in locations])
    ]
    return np.stack(
        [file[name][t, e] for t, e in locations], axis=0
    ).astype(np.float32)


def load_episode(path: Path, episode_id: int) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as file:
        return {
            name: _episode(file, episode_id, name)
            for name in (
                "palm_pose_object",
                "q_hand",
                "fingertip_pose_object",
            )
        }


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    data: dict[str, np.ndarray],
    episode_id: int,
    viewer: Literal["headless", "native", "viser", "video"],
    device: torch.device,
    max_steps: int,
    warmup_steps: int,
    contact_threshold: float,
    controller_config: SurfaceMCCFingerConfig | FullHandMCCFingerConfig,
    capsule_radius: float,
    capsule_half_height: float,
    target_source: Literal["teacher_tip", "nearest_surface"],
    controller_kind: Literal["reference_dynamics", "fullhand_admittance"],
    teacher_nominal: bool,
    surface_preload: float,
    finger_servo_load_scale: float,
    finger_tracking_gain: float,
    contact_settle_frames: int,
    precontact_force: float,
    contact_search_step: float,
    contact_search_step_rad: float,
    contact_search_limit_rad: float,
    max_calibrated_force: float,
    highlight_contacts: bool,
    report: Path,
    video_output: Path | None,
    video_fps: int,
) -> None:
    frames = len(data["q_hand"])
    if max_steps > 0:
        frames = min(frames, max_steps)
    if not 0 <= warmup_steps < frames:
        raise ValueError("warmup_steps must lie inside the episode")

    env_cfg = replay_env_cfg()
    if viewer == "video":
        env_cfg.viewer.width = 960
        env_cfg.viewer.height = 720
        env_cfg.viewer.distance = 0.45
        env_cfg.viewer.azimuth = 45.0
        env_cfg.viewer.elevation = -10.0
        env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.ASSET_BODY
        env_cfg.viewer.entity_name = "robot"
        env_cfg.viewer.body_name = "palm_lower"
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=str(device),
        render_mode="rgb_array" if viewer == "video" else None,
    )
    wrapped = RslRlVecEnvWrapper(env)
    robot = env.scene["robot"]
    if controller_kind == "fullhand_admittance":
        if not isinstance(controller_config, FullHandMCCFingerConfig):
            raise TypeError("fullhand_admittance requires FullHandMCCFingerConfig")
        controller = FullHandMCCFingerController(controller_config)
    else:
        if not isinstance(controller_config, SurfaceMCCFingerConfig):
            raise TypeError("reference_dynamics requires SurfaceMCCFingerConfig")
        controller = SurfaceMCCFingerController(controller_config)
    oracle = PrivilegedCapsuleSurfaceOracle(
        radius=capsule_radius,
        half_height=capsule_half_height,
    )

    class ReplayPolicy:
        def __init__(self) -> None:
            self.frame = 0
            self.rows: list[dict[str, float | int | str]] = []
            self.site_standoff_m: np.ndarray | None = None
            self.contact3_frames = 0
            self.contact4_frames = 0
            self.tip_found_frames = np.zeros(4, dtype=np.int64)
            self.tip_loaded_frames = np.zeros(4, dtype=np.int64)
            self.force_max = 0.0
            # Full-hand demo preparation state.  The inverse task has no arm
            # planning DOFs, so its five-point planner reduces to a
            # fixed-palm four-tip planner; the teacher palm trajectory remains
            # common to both DP and MCC A/B tests.
            self.contact_calibrated = controller_kind != "fullhand_admittance"
            self.contact_settle_streak = 0
            self.precontact_closure = np.zeros(16, dtype=np.float32)
            self.contact_servo_offset = np.zeros(16, dtype=np.float32)
            self.force_setpoint = np.zeros(4, dtype=np.float32)
            # Debug geometry follows the full_hand_mcc demo convention:
            # small target spheres/arrows, large live-contact spheres, and a
            # red fingertip marker whenever contact is absent.
            self.visual_surface_targets = np.full((4, 3), np.nan)
            self.visual_normals = np.full((4, 3), np.nan)
            self.visual_contact_points = np.full((4, 3), np.nan)
            self.visual_tip_points = np.full((4, 3), np.nan)
            self.visual_found = np.zeros(4, dtype=bool)
            self.visual_loaded = np.zeros(4, dtype=bool)

        def _set_palm(self, t: int) -> None:
            pose = torch.as_tensor(
                data["palm_pose_object"][t],
                device=env.device,
                dtype=torch.float32,
            )
            state = torch.cat(
                (pose, torch.zeros(6, device=env.device))
            ).unsqueeze(0)
            robot.write_root_state_to_sim(state)
            if env.sim.model.nmocap:
                env.sim.data.mocap_pos[:, 0, :] = 0.0
                env.sim.data.mocap_quat[:, 0, :] = torch.tensor(
                    (1.0, 0.0, 0.0, 0.0), device=env.device
                )

        def __call__(self, _observation: dict[str, torch.Tensor]) -> torch.Tensor:
            t = min(self.frame, frames - 1)
            self._set_palm(t)
            if t < warmup_steps:
                q_teacher_t = torch.as_tensor(
                    data["q_hand"][t],
                    device=env.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                robot.write_joint_state_to_sim(
                    position=q_teacher_t,
                    velocity=torch.zeros_like(q_teacher_t),
                )
            env.sim.forward()
            (
                forces,
                _,
                contact_points,
                geometry_found,
                force_found,
                distances,
            ) = live_tip_observation(env)
            found = geometry_found
            q_live = (
                robot.data.joint_pos[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            palm_pose = data["palm_pose_object"][t]
            tip_palm = controller.tip_positions_palm(q_live)
            tip_world = controller.points_palm_to_world(
                tip_palm, palm_pose
            )
            actual_surface = oracle.observe(tip_world)
            if target_source == "teacher_tip":
                target_site_points_world = data[
                    "fingertip_pose_object"
                ][t, :, :3].copy()
                target_surface = oracle.observe(
                    target_site_points_world
                )
                target_normals_world = target_surface.normals_world
            else:
                target_surface = actual_surface
                target_normals_world = actual_surface.normals_world
                target_site_points_world = actual_surface.points_world.copy()
            if (
                target_source == "nearest_surface"
                and self.site_standoff_m is None
                and t >= warmup_steps
            ):
                # The MCC sites lie inside the fingertip bodies, not on the
                # collision surface.  Calibrate that geometric standoff from
                # the settled teacher contact, then transport it along the
                # privileged surface normal just like full_hand_mcc.
                self.site_standoff_m = actual_surface.signed_distance.astype(
                    np.float64
                )
                print(
                    "[SURFACE-MCC] calibrated site standoff="
                    f"{np.round(self.site_standoff_m*1000, 2).tolist()}mm"
                )
            if (
                target_source == "nearest_surface"
                and self.site_standoff_m is not None
            ):
                target_site_points_world += (
                    self.site_standoff_m[:, None]
                    * target_normals_world
                )

            # The fullhand planner distinguishes a surface point from the
            # actual kinematic fingertip task.  Its nominal 2 mm inward
            # preload is absent from a raw inverse tip-site trajectory, so
            # restore it here before its standard admittance/IK branch.
            kinematic_target_world = target_site_points_world.copy()
            if controller_kind == "fullhand_admittance":
                kinematic_target_world -= (
                    surface_preload * target_normals_world
                )

            force_magnitude = np.linalg.norm(forces, axis=-1)
            if controller_kind == "fullhand_admittance":
                controller.calibrate_force_sign(
                    forces,
                    found,
                    target_normals_world,
                )

            if t < warmup_steps:
                q_command = data["q_hand"][t].copy()
                debug = {
                    "normal_force": np.abs(
                        np.einsum(
                            "fi,fi->f",
                            forces,
                            target_normals_world,
                        )
                    ),
                    "force_error": np.zeros(4, dtype=np.float32),
                    "contact_active": found.copy(),
                    "reference_speed": np.zeros(4, dtype=np.float32),
                    "surface_error": np.linalg.norm(
                        kinematic_target_world - tip_world, axis=-1
                    ),
                }
                if t == warmup_steps - 1:
                    controller.reset()
            elif controller_kind == "fullhand_admittance" and not self.contact_calibrated:
                # Identical to demo_surface_slide: do not start admittance
                # until all pads have established settled contact.  Each
                # missing finger searches only along its own inward normal.
                settled_contact = found & (force_magnitude >= precontact_force)
                if bool(np.all(settled_contact)):
                    self.contact_settle_streak += 1
                else:
                    self.contact_settle_streak = 0
                    self.precontact_closure += controller.normal_search_delta(
                        q_live,
                        palm_pose,
                        target_normals_world,
                        ~settled_contact,
                        inward_step=contact_search_step,
                        max_joint_step=contact_search_step_rad,
                    )
                    self.precontact_closure = np.clip(
                        self.precontact_closure,
                        -contact_search_limit_rad,
                        contact_search_limit_rad,
                    )
                q_command = data["q_hand"][t] + self.precontact_closure
                debug = {
                    "normal_force": np.abs(np.einsum(
                        "fi,fi->f", forces, target_normals_world
                    )),
                    "force_error": np.zeros(4, dtype=np.float32),
                    "contact_active": settled_contact,
                    "reference_speed": np.zeros(4, dtype=np.float32),
                    "surface_error": np.linalg.norm(
                        kinematic_target_world - tip_world, axis=-1
                    ),
                }
                if self.contact_settle_streak >= contact_settle_frames:
                    self.contact_calibrated = True
                    self.force_setpoint = controller.calibrate_force_setpoint(
                        forces,
                        settled_contact,
                        target_normals_world,
                        maximum_force=max_calibrated_force,
                    ).astype(np.float32)
                    self.contact_servo_offset = (
                        data["q_hand"][t] - q_live
                    ).astype(np.float32)
                    print(
                        "[FULLHAND-PREP] contact settled at "
                        f"frame={t}; force_setpoint="
                        f"{np.round(self.force_setpoint, 2).tolist()}N "
                        f"servo_offset_max="
                        f"{float(np.max(np.abs(self.contact_servo_offset))):.4f}rad"
                    )
            else:
                if controller_kind == "fullhand_admittance":
                    # Same external plan interface as full_hand_mcc:
                    # reachable-q + calibrated loaded-servo bias + lead term.
                    if teacher_nominal:
                        planner_q = data["q_hand"][t]
                        nominal_q = (
                            planner_q
                            + finger_servo_load_scale * self.contact_servo_offset
                            + finger_tracking_gain * (planner_q - q_live)
                        )
                    else:
                        nominal_q = None
                else:
                    nominal_q = (
                        data["q_hand"][t] if teacher_nominal else None
                    )
                q_command, debug = controller.update(
                    q_live=q_live,
                    palm_pose_world=palm_pose,
                    force_world=forces,
                    found=found,
                    surface_points_world=kinematic_target_world,
                    surface_normals_world=target_normals_world,
                    nominal_posture_q=nominal_q,
                )

            raw_action = np.clip(
                (q_command - q_live) / ACTION_SCALE, -2.0, 2.0
            )
            found_count = int(found.sum())
            loaded = found & (force_magnitude >= contact_threshold)
            loaded_count = int(loaded.sum())
            self.visual_surface_targets[:] = target_surface.points_world
            self.visual_normals[:] = target_normals_world
            self.visual_contact_points[:] = np.nan
            self.visual_contact_points[found] = contact_points[found]
            self.visual_tip_points[:] = tip_world
            self.visual_found[:] = found
            self.visual_loaded[:] = loaded
            q_teacher_error = float(
                np.abs(q_live - data["q_hand"][t]).mean()
            )
            if t >= warmup_steps:
                self.contact3_frames += int(found_count >= 3)
                self.contact4_frames += int(found_count == 4)
                self.tip_found_frames += found.astype(np.int64)
                self.tip_loaded_frames += loaded.astype(np.int64)
                self.force_max = max(
                    self.force_max,
                    float(force_magnitude.max(initial=0.0)),
                )

            row: dict[str, float | int | str] = {
                "mode": f"surface_mcc_{controller_kind}",
                "target_source": target_source,
                "teacher_nominal": int(teacher_nominal),
                "fullhand_contact_calibrated": int(self.contact_calibrated),
                "fullhand_contact_settle_streak": self.contact_settle_streak,
                "episode_id": episode_id,
                "frame": t,
                "warmup": int(t < warmup_steps),
                "q_teacher_mae_rad": q_teacher_error,
                "found_contacts": found_count,
                "geometry_found_contacts": found_count,
                "force_sensor_found_contacts": int(force_found.sum()),
                "sensor_found_mismatch_count": int(
                    np.count_nonzero(geometry_found != force_found)
                ),
                "loaded_contacts": loaded_count,
                "force_max_N": float(force_magnitude.max(initial=0.0)),
                "min_contact_distance_m": float(distances.min(initial=0.0)),
                "oracle_abs_distance_max_mm": float(
                    np.abs(actual_surface.signed_distance).max() * 1000.0
                ),
                "site_standoff_error_max_mm": float(
                    np.max(
                        np.abs(
                            actual_surface.signed_distance
                            - (
                                self.site_standoff_m
                                if self.site_standoff_m is not None
                                else actual_surface.signed_distance
                            )
                        )
                    )
                    * 1000.0
                ),
            }
            for finger, label in enumerate(TIP_LABELS):
                row.update(
                    {
                        f"{label}_found": int(found[finger]),
                        f"{label}_geometry_found": int(geometry_found[finger]),
                        f"{label}_force_sensor_found": int(force_found[finger]),
                        f"{label}_sensor_found_mismatch": int(
                            geometry_found[finger] != force_found[finger]
                        ),
                        f"{label}_force_raw_N": float(force_magnitude[finger]),
                        f"{label}_normal_force_N": float(
                            debug["normal_force"][finger]
                        ),
                        f"{label}_force_error_N": float(
                            debug["force_error"][finger]
                        ),
                        f"{label}_contact_active": int(
                            debug["contact_active"][finger]
                        ),
                        f"{label}_surface_distance_mm": float(
                            actual_surface.signed_distance[finger] * 1000.0
                        ),
                        f"{label}_site_standoff_target_mm": float(
                            (
                                self.site_standoff_m[finger]
                                if self.site_standoff_m is not None
                                else target_surface.signed_distance[finger]
                            )
                            * 1000.0
                        ),
                        f"{label}_surface_error_mm": float(
                            debug["surface_error"][finger] * 1000.0
                        ),
                        f"{label}_reference_speed_mps": float(
                            debug["reference_speed"][finger]
                        ),
                    }
                )
            for joint in range(16):
                row[f"q_live_{joint}"] = float(q_live[joint])
                row[f"q_cmd_{joint}"] = float(q_command[joint])
                row[f"q_teacher_{joint}"] = float(
                    data["q_hand"][t, joint]
                )
            self.rows.append(row)
            if t % 100 == 0:
                print(
                    f"[SURFACE-MCC] frame={t:4d} "
                    f"q_teacher={q_teacher_error:.4f}rad "
                    f"found={found_count}/4 loaded={loaded_count}/4 "
                    f"F={np.round(force_magnitude, 2).tolist()}N "
                    f"dist={np.round(actual_surface.signed_distance*1000, 2).tolist()}mm"
                )
            self.frame += 1
            return torch.as_tensor(
                raw_action, device=env.device, dtype=torch.float32
            ).unsqueeze(0)

    policy = ReplayPolicy()
    if highlight_contacts:
        base_update_visualizers = env.update_visualizers

        def update_contact_visualizers(visualizer) -> None:
            base_update_visualizers(visualizer)
            if not np.all(np.isfinite(policy.visual_surface_targets)):
                return
            target_colors = (
                (1.0, 0.30, 0.20, 0.95),  # index
                (0.20, 0.70, 1.0, 0.95),  # middle
                (1.0, 0.80, 0.15, 0.95),  # ring
                (0.80, 0.30, 1.0, 0.95),  # thumb
            )
            for finger, color in enumerate(target_colors):
                surface = policy.visual_surface_targets[finger]
                normal = policy.visual_normals[finger]
                visualizer.add_sphere(surface, radius=0.005, color=color)
                visualizer.add_arrow(
                    surface,
                    surface + 0.025 * normal,
                    color=color,
                    width=0.0025,
                )
                if policy.visual_found[finger]:
                    contact = policy.visual_contact_points[finger]
                    if np.all(np.isfinite(contact)):
                        # Green: geometrically contacted and above the force
                        # threshold. Orange: geometry contact but weak force.
                        live_color = (
                            (0.15, 1.0, 0.20, 1.0)
                            if policy.visual_loaded[finger]
                            else (1.0, 0.50, 0.05, 1.0)
                        )
                        visualizer.add_sphere(
                            contact, radius=0.009, color=live_color
                        )
                else:
                    # A red sphere on the actual fingertip shows precisely
                    # which finger has left the object, rather than merely
                    # showing an absent contact marker.
                    visualizer.add_sphere(
                        policy.visual_tip_points[finger],
                        radius=0.007,
                        color=(1.0, 0.05, 0.05, 1.0),
                    )

        env.update_visualizers = update_contact_visualizers
    try:
        if viewer == "headless":
            for _ in range(frames):
                wrapped.step(policy(wrapped.get_observations()))
        elif viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        elif viewer == "viser":
            ViserPlayViewer(wrapped, policy).run()
        else:
            if video_output is None:
                raise ValueError("video_output is required for video mode")
            video_output.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(
                video_output,
                fps=video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            ) as writer:
                control_dt = float(
                    env_cfg.decimation * env_cfg.sim.mujoco.timestep
                )
                next_frame_time = 0.0
                for _ in range(frames):
                    wrapped.step(policy(wrapped.get_observations()))
                    if policy.frame * control_dt + 1.0e-9 >= next_frame_time:
                        frame = env.render()
                        if frame is not None:
                            writer.append_data(
                                np.asarray(frame, dtype=np.uint8)
                            )
                        next_frame_time += 1.0 / video_fps
    finally:
        write_report(report, policy.rows)
        active = max(1, frames - warmup_steps)
        q_errors = np.asarray(
            [
                row["q_teacher_mae_rad"]
                for row in policy.rows
                if int(row["frame"]) >= warmup_steps
            ],
            dtype=float,
        )
        q_mae = float(q_errors.mean()) if len(q_errors) else float("nan")
        q_p95 = float(np.percentile(q_errors, 95)) if len(q_errors) else float("nan")
        print(
            f"[RESULT] mode=surface_mcc_{controller_kind} episode={episode_id} "
            f"frames={frames} q_mae={q_mae:.6f}rad "
            f"q_p95={q_p95:.6f}rad "
            f"contact3={100*policy.contact3_frames/active:.1f}% "
            f"contact4={100*policy.contact4_frames/active:.1f}% "
            f"force_max={policy.force_max:.2f}N "
            f"tip_found={np.round(100*policy.tip_found_frames/active,1).tolist()}% "
            f"tip_loaded={np.round(100*policy.tip_loaded_frames/active,1).tolist()}% "
            f"report={report}"
        )
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--viewer",
        choices=("headless", "native", "viser", "video"),
        default="headless",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=75)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--desired-force", type=float, default=1.0)
    parser.add_argument(
        "--controller",
        choices=("reference_dynamics", "fullhand_admittance"),
        default="fullhand_admittance",
        help=(
            "fullhand_admittance ports full_hand_mcc's normal-admittance and "
            "Mink fingertip IK path."
        ),
    )
    parser.add_argument("--force-kp", type=float, default=0.004)
    parser.add_argument("--force-ki", type=float, default=0.001)
    parser.add_argument("--normal-position-kp", type=float, default=8.0)
    parser.add_argument("--max-tip-speed", type=float, default=0.04)
    parser.add_argument("--max-reference-offset-mm", type=float, default=35.0)
    parser.add_argument("--finger-virtual-mass", type=float, default=0.08)
    parser.add_argument("--finger-virtual-damping", type=float, default=18.0)
    parser.add_argument("--finger-virtual-stiffness", type=float, default=1000.0)
    parser.add_argument("--finger-force-gain", type=float, default=1.0)
    parser.add_argument("--finger-force-filter-alpha", type=float, default=0.25)
    parser.add_argument("--finger-contact-on-force", type=float, default=0.15)
    parser.add_argument("--finger-contact-off-force", type=float, default=0.08)
    parser.add_argument("--finger-max-normal-offset-mm", type=float, default=3.0)
    parser.add_argument("--finger-max-normal-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--finger-max-normal-acceleration", type=float, default=0.2)
    parser.add_argument(
        "--surface-preload-mm",
        type=float,
        default=2.0,
        help="fullhandMCC planner's inward kinematic target preload.",
    )
    parser.add_argument(
        "--finger-servo-load-scale",
        type=float,
        default=1.5,
        help="Scale for the calibrated loaded-position joint bias.",
    )
    parser.add_argument(
        "--finger-tracking-gain",
        type=float,
        default=0.5,
        help="Fullhand planner's joint-space lead tracking gain.",
    )
    parser.add_argument("--contact-settle-frames", type=int, default=3)
    parser.add_argument("--precontact-force", type=float, default=0.10)
    parser.add_argument("--contact-search-step-mm", type=float, default=0.15)
    parser.add_argument("--contact-search-step-rad", type=float, default=0.02)
    parser.add_argument("--contact-search-limit-rad", type=float, default=0.30)
    parser.add_argument("--finger-max-calibrated-force", type=float, default=12.0)
    parser.add_argument(
        "--highlight-contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw planner targets and live fingertip contact state in viewers.",
    )
    parser.add_argument(
        "--nominal-normal-preload-mm",
        type=float,
        default=0.0,
        help="Constant inward site displacement around privileged nominal q.",
    )
    parser.add_argument(
        "--teacher-nominal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the recorded reachable hand posture as full_hand_mcc's "
            "external surface-planner nominal. Disable this for the later "
            "no-teacher pure-reactive experiment."
        ),
    )
    parser.add_argument(
        "--nominal-force-compliance-mm-per-n",
        type=float,
        default=0.35,
        help="Normal displacement per newton of force error.",
    )
    parser.add_argument(
        "--nominal-preload-scales",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 5.0, 3.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help="Per-finger multipliers for nominal normal preload.",
    )
    parser.add_argument("--capsule-radius", type=float, default=0.15)
    parser.add_argument("--capsule-half-height", type=float, default=0.08)
    parser.add_argument(
        "--target-source",
        choices=("teacher_tip", "nearest_surface"),
        default="teacher_tip",
        help=(
            "Temporary privileged planner output. teacher_tip supplies the "
            "recorded Cartesian tip-site path; nearest_surface supplies only "
            "the analytic closest surface point."
        ),
    )
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    data = load_episode(args.file, args.episode_id)
    report = args.report or Path(
        "mcc_finger_compliance_control/data/models/deployment_ab"
    ) / f"surface_mcc_privileged_ep{args.episode_id}.csv"
    video_output = args.video_output
    if args.viewer == "video" and video_output is None:
        video_output = Path(
            "mcc_finger_compliance_control/outputs"
        ) / f"surface_mcc_privileged_ep{args.episode_id}.mp4"
    print(
        f"[INFO] surface MCC controller={args.controller} episode={args.episode_id} "
        f"frames={len(data['q_hand'])} device={device} "
        f"warmup={args.warmup_steps} target_force={args.desired_force:.2f}N "
        f"target={args.target_source} teacher_nominal={args.teacher_nominal}"
    )
    controller_config: SurfaceMCCFingerConfig | FullHandMCCFingerConfig
    if args.controller == "fullhand_admittance":
        controller_config = FullHandMCCFingerConfig(
            desired_force=args.desired_force,
            virtual_mass=args.finger_virtual_mass,
            virtual_damping=args.finger_virtual_damping,
            virtual_stiffness=args.finger_virtual_stiffness,
            force_gain=args.finger_force_gain,
            force_filter_alpha=args.finger_force_filter_alpha,
            contact_on_force=args.finger_contact_on_force,
            contact_off_force=args.finger_contact_off_force,
            max_normal_offset=args.finger_max_normal_offset_mm / 1000.0,
            max_normal_speed=args.finger_max_normal_speed_mm_s / 1000.0,
            max_normal_acceleration=args.finger_max_normal_acceleration,
        )
    else:
        controller_config = SurfaceMCCFingerConfig(
            desired_force=args.desired_force,
            force_kp=args.force_kp,
            force_ki=args.force_ki,
            normal_position_kp=args.normal_position_kp,
            max_reference_speed=args.max_tip_speed,
            max_reference_offset=args.max_reference_offset_mm / 1000.0,
            nominal_normal_preload=(
                args.nominal_normal_preload_mm / 1000.0
            ),
            nominal_preload_scales=tuple(args.nominal_preload_scales),
            nominal_force_compliance=(
                args.nominal_force_compliance_mm_per_n / 1000.0
            ),
        )
    run(
        data=data,
        episode_id=args.episode_id,
        viewer=args.viewer,
        device=device,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        contact_threshold=args.contact_threshold,
        controller_config=controller_config,
        capsule_radius=args.capsule_radius,
        capsule_half_height=args.capsule_half_height,
        target_source=args.target_source,
        controller_kind=args.controller,
        teacher_nominal=args.teacher_nominal,
        surface_preload=args.surface_preload_mm / 1000.0,
        finger_servo_load_scale=args.finger_servo_load_scale,
        finger_tracking_gain=args.finger_tracking_gain,
        contact_settle_frames=args.contact_settle_frames,
        precontact_force=args.precontact_force,
        contact_search_step=args.contact_search_step_mm / 1000.0,
        contact_search_step_rad=args.contact_search_step_rad,
        contact_search_limit_rad=args.contact_search_limit_rad,
        max_calibrated_force=args.finger_max_calibrated_force,
        highlight_contacts=args.highlight_contacts,
        report=report,
        video_output=video_output,
        video_fps=args.video_fps,
    )


if __name__ == "__main__":
    main()
