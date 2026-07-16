from __future__ import annotations

import argparse
import glob
import os
from collections import deque
from pathlib import Path
from typing import Literal

import h5py
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

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
from mjlab.viewer import NativeMujocoViewer, ViewerConfig, ViserPlayViewer
from mjlab.tasks.leaphand.leaphand_finger_env_cfg import (
    LeapHandComplianceController,
    fsr_force_and_visual_logic,
)
from palm_compliance_control.controllers import (
    MultiContactFingerImpedanceController,
)


LEAP_HAND_XML = Path(
    "./src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand.xml"
)
DEFAULT_DP_DATASET = Path(
    "./palm_compliance_control/data/pipeline_smoke/fixed2500_hierarchical_env1_overfit.h5"
)
DEFAULT_DP_MODEL = Path(
    "./palm_compliance_control/data/pipeline_smoke/"
    "overfit_hierarchical_full_sample/pretrained"
)


def _rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6d[:3], dtype=np.float64)
    a2 = np.asarray(rot6d[3:6], dtype=np.float64)
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    a2 = a2 - b1 * float(np.dot(b1, a2))
    b2 = a2 / max(np.linalg.norm(a2), 1e-8)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-1)


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = R.from_matrix(matrix).as_quat()
    return xyzw[[3, 0, 1, 2]].astype(np.float32)


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def _wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()


def _load_hand_only_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LEAP_HAND_XML))


def _get_fixed_target_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_ball")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.15, 0.08],
        rgba=[0.2, 0.6, 1.0, 1.0],
        mass=1.0,
    )
    return spec


def _dummy_joint_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos


def _build_hand_replay_env_cfg() -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=_load_hand_only_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=500.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"13": 1.57}),
    )

    target_cfg = EntityCfg(
        spec_fn=_get_fixed_target_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    fsr_contact_cfg = ContactSensorCfg(
        name="fsr_contact",
        primary=ContactMatch(
            mode="geom", pattern=r".*_fsr_geom$", entity="robot"
        ),
        secondary=ContactMatch(
            mode="body", pattern="target_ball", entity="target"
        ),
        fields=("found", "force", "dist"),
        reduce="netforce",
        num_slots=1,
    )
    penetration_cfg = ContactSensorCfg(
        name="hand_object_penetration",
        primary=ContactMatch(
            mode="subtree", pattern="palm_lower", entity="robot"
        ),
        secondary=ContactMatch(
            mode="body", pattern="target_ball", entity="target"
        ),
        fields=("dist",),
        reduce="mindist",
        num_slots=1,
    )

    observations = {
        "policy": ObservationGroupCfg(
            {
                "fsr_forces": ObservationTermCfg(
                    func=fsr_force_and_visual_logic,
                    params={
                        "sensor_name": "fsr_contact",
                        "fsr_regex": r".*_fsr_geom$",
                        "display_forces": False,
                    },
                ),
                "joint_pos": ObservationTermCfg(
                    func=_dummy_joint_obs,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                )
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
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=(fsr_contact_cfg, penetration_cfg),
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
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="palm_lower",
            distance=0.8,
        ),
        episode_length_s=1e10,
    )


def _select_hand_joint_slice(q_step: np.ndarray) -> np.ndarray:
    # Recorded q can be either 22D (xarm6 + hand) or 16D (hand only).
    if q_step.shape[-1] == 22:
        return q_step[..., -16:]
    if q_step.shape[-1] == 16:
        return q_step
    raise ValueError(f"Unsupported q dimension {q_step.shape[-1]}; expected 16 or 22.")


def _decode_str_attr(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _build_q_mapper(
    source_joint_names: list[str] | None,
    target_joint_names: tuple[str, ...],
    q_dim: int,
):
    if q_dim == 16:
        print("[INFO] q has 16 dims; using direct hand joint replay.")
        return lambda q_step: q_step

    if q_dim == 22 and source_joint_names:
        name_to_src = {name: i for i, name in enumerate(source_joint_names)}
        missing = [name for name in target_joint_names if name not in name_to_src]
        if not missing:
            src_ids = [name_to_src[name] for name in target_joint_names]
            print("[INFO] Using name-based q remapping from source to replay hand model.")
            return lambda q_step: q_step[src_ids]

    if q_dim == 22:
        print("[WARN] Missing source joint metadata; fallback to last 16 dims.")
        return lambda q_step: _select_hand_joint_slice(q_step)

    raise ValueError(f"Unsupported q dimension {q_dim}; expected 16 or 22.")


def run_replay(
    file_path: str,
    device: str | None = None,
    env_idx: int = 0,
    viewer: Literal["headless", "native", "viser"] = "native",
    replay_mode: Literal[
        "teacher", "dp-palm", "teacher-palm-dp-qpre", "dp-full"
    ] = "teacher",
    dp_dataset: str | None = None,
    dp_model: str | None = None,
    dp_inference_steps: int = 20,
    dp_interval: int = 5,
    max_steps: int = 0,
    dp_state_source: Literal["teacher", "live"] = "teacher",
    finger_controller_type: Literal["legacy", "impedance"] = "impedance",
    palm_controller_type: Literal["root_pose", "pd_wrench"] = "root_pose",
    kp_pos: float = 500.0,
    kd_pos: float = 40.0,
    kp_rot: float = 50.0,
    kd_rot: float = 5.0,
    max_force: float = 100.0,
    max_torque: float = 10.0,
    gravity_compensation: bool = True,
    seed: int = 42,
) -> None:
    if not file_path:
        print("[ERROR] No inverted H5 file provided.")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    with h5py.File(file_path, "r") as f:
        q_traj = np.array(f["q"], dtype=np.float32)
        palm_traj = np.array(f["palm_pose_world"], dtype=np.float32)
        obj_traj = np.array(f["obj_pose_world"], dtype=np.float32)
        fsr_traj = (
            np.array(f["fsr"], dtype=np.float32) if "fsr" in f else None
        )
        num_steps = int(q_traj.shape[0])
        if max_steps > 0:
            num_steps = min(num_steps, int(max_steps))
        source_joint_names = None
        if "source_robot_joint_names" in f.attrs:
            source_joint_names = _decode_str_attr(
                np.asarray(f.attrs["source_robot_joint_names"])
            )

    num_envs_in_file = int(q_traj.shape[1])
    if not (0 <= env_idx < num_envs_in_file):
        raise ValueError(
            f"env_idx={env_idx} out of range [0, {num_envs_in_file - 1}]"
        )

    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = _build_hand_replay_env_cfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=run_device)
    viewer_env = RslRlVecEnvWrapper(env)
    action_dim = int(env.action_manager.total_action_dim)
    target_joint_names = tuple(env.scene["robot"].joint_names)
    robot = env.scene["robot"]
    robot_body_ids = robot.data.indexing.body_ids
    body_mass = env.sim.model.body_mass
    if body_mass.ndim == 2:
        body_mass = body_mass[0]
    robot_body_mass = body_mass[robot_body_ids].to(
        device=env.device, dtype=torch.float32
    )
    q_mapper = _build_q_mapper(
        source_joint_names=source_joint_names,
        target_joint_names=target_joint_names,
        q_dim=int(q_traj.shape[-1]),
    )

    dist = np.linalg.norm(palm_traj[:, env_idx, :3] - obj_traj[:, env_idx, :3], axis=-1)
    print(
        "[INFO] palm-object distance stats "
        f"mean={float(dist.mean()):.4f}, min={float(dist.min()):.4f}, max={float(dist.max()):.4f}"
    )

    dp_runtime = None
    if replay_mode != "teacher":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        dataset_path = Path(dp_dataset) if dp_dataset else DEFAULT_DP_DATASET
        model_path = Path(dp_model) if dp_model else DEFAULT_DP_MODEL
        with h5py.File(dataset_path, "r") as data:
            dp_state = np.asarray(data["state"][:, 0], dtype=np.float32)
            dp_action = np.asarray(data["action"][:, 0], dtype=np.float32)
        usable = min(num_steps, len(dp_state), len(dp_action))
        dp_state = dp_state[:usable]
        dp_action = dp_action[:usable]

        print(f"[INFO] Loading DP palm model: {model_path}")
        policy = DiffusionPolicy.from_pretrained(
            model_path, local_files_only=True
        ).to(run_device)
        policy.diffusion.num_inference_steps = int(dp_inference_steps)
        policy.eval()
        policy_cfg = PreTrainedConfig.from_pretrained(
            model_path, local_files_only=True
        )
        try:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg, pretrained_path=model_path
            )
        except Exception:
            checkpoint = torch.load(
                model_path.parent / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg, dataset_stats=checkpoint["stats"]
            )

        # DP predicts the palm FSR-center control point, whereas play_inverse
        # writes the palm_lower body origin.  Recover their fixed local offset
        # directly from aligned teacher labels.
        offsets = []
        for i in range(min(200, usable)):
            root_r = _wxyz_to_matrix(palm_traj[i, env_idx, 3:7])
            offsets.append(
                root_r.T
                @ (dp_action[i, :3] - palm_traj[i, env_idx, :3])
            )
        control_offset_local = np.median(np.stack(offsets), axis=0)
        dp_runtime = {
            "policy": policy,
            "pre": preprocessor,
            "post": postprocessor,
            "state": dp_state,
            "action": dp_action,
            "usable": usable,
            "interval": max(1, int(dp_interval)),
            "offset": control_offset_local,
            "state_source": dp_state_source,
            "fsr_max": np.maximum(dp_state[:, 3:19].max(axis=0), 1e-3),
        }
        print(
            f"[INFO] replay_mode={replay_mode} | usable={usable} | "
            f"interval={dp_runtime['interval']} sim steps | "
            f"inference_steps={dp_inference_steps} | "
            f"state_source={dp_state_source} | "
            f"control_offset_local={np.round(control_offset_local, 5)}"
        )

    finger_mode = replay_mode in ("teacher-palm-dp-qpre", "dp-full")
    finger_controller = None
    if finger_mode:
        if finger_controller_type == "legacy":
            finger_controller = LeapHandComplianceController(
                device=str(env.device),
                num_envs=1,
                K_pre_free=0.2,
                K_pre_contact=0.05,
            )
            # Match the original controller used by the current collector.
            for name, value in {
                "S_min": 0.6,
                "S_max": 1.5,
                "K_prox": 0.8,
                "K_mid": 0.32,
                "K_dist": 0.2,
                "D_force": 2.8,
                "K_limit_spring": 0.3,
                "q_pre_grasp_list": [0.6, 0.4, 0.3],
                "S_contact_threshold": 0.15,
                "reset_speed": 0.1,
                "alpha_obs": 0.2,
                "alpha_ctrl": 0.15,
                "contact_on_threshold": 0.20,
                "contact_off_threshold": 0.12,
                "error_deadband": 0.03,
                "ds_clip": 0.10,
                "action_rate_limit": 0.15,
            }.items():
                setattr(finger_controller, name, value)
        else:
            finger_controller = MultiContactFingerImpedanceController(
                device=env.device, num_envs=1
            )
        print(
            f"[INFO] Finger validation enabled: mode={replay_mode} | "
            f"controller={finger_controller_type} | "
            "teacher q_actual seeds frames 0..35, then DP q_pre + live FSR controller"
        )
    print(
        f"[INFO] Palm executor: {palm_controller_type} | "
        f"gravity_comp={gravity_compensation} | seed={seed}"
    )

    class ReplayPolicy:
        def __init__(self):
            self.t = 0
            self.dp_calls = 0
            self.segment_start = palm_traj[0, env_idx].copy()
            self.segment_target = self.segment_start.copy()
            initial_qpre = (
                dp_runtime["action"][0, 9:25].copy()
                if dp_runtime is not None
                else q_mapper(q_traj[0, env_idx]).copy()
            )
            self.segment_qpre_start = initial_qpre.copy()
            self.segment_qpre_target = initial_qpre.copy()
            self.current_qpre = initial_qpre.copy()
            self.active_steps = 0
            self.contact3_steps = 0
            self.max_fsr_norm = 0.0
            self.penetration_samples: list[float] = []
            self.dp_pos_errors: list[float] = []
            self.dp_qpre_errors: list[float] = []
            self.palm_tracking_pos_errors: list[float] = []
            self.palm_tracking_ori_errors: list[float] = []
            bootstrap = np.minimum(
                np.arange(8) * 5,
                int(dp_runtime["usable"]) - 1,
            ) if dp_runtime is not None else np.zeros(8, dtype=np.int64)
            self.live_history = deque(
                [dp_runtime["state"][i].copy() for i in bootstrap]
                if dp_runtime is not None
                else [],
                maxlen=8,
            )
            self.geometry_pos_history: deque[np.ndarray] = deque(maxlen=3)
            self.geometry_normal_history: deque[np.ndarray] = deque(maxlen=3)

        def _live_dp_state(self, t: int) -> np.ndarray:
            assert dp_runtime is not None
            sensor_data = env.scene["fsr_contact"].data
            if sensor_data.force is None:
                fsr = np.zeros(16, dtype=np.float32)
            else:
                force = torch.linalg.vector_norm(sensor_data.force[0], dim=-1)
                if sensor_data.found is not None:
                    force = torch.where(
                        sensor_data.found[0] > 0,
                        force,
                        torch.zeros_like(force),
                    )
                fsr = force.detach().cpu().numpy()
            fsr = np.clip(fsr, 0.0, dp_runtime["fsr_max"])
            q_hand = (
                env.scene["robot"].data.joint_pos[0, :16]
                .detach()
                .cpu()
                .numpy()
            )
            root = (
                env.scene["robot"].data.root_link_pose_w[0]
                .detach()
                .cpu()
                .numpy()
            )
            root_r = _wxyz_to_matrix(root[3:7])
            control_pos = root[:3] + root_r @ dp_runtime["offset"]
            axis_point = np.array(
                [0.0, 0.0, np.clip(control_pos[2], -0.08, 0.08)],
                dtype=np.float64,
            )
            normal = control_pos - axis_point
            normal /= max(np.linalg.norm(normal), 1e-8)
            self.geometry_pos_history.append(control_pos.copy())
            self.geometry_normal_history.append(normal.copy())
            curvature = 0.0
            if len(self.geometry_pos_history) == 3:
                positions = np.stack(self.geometry_pos_history)
                normals = np.stack(self.geometry_normal_history)
                distance = np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum()
                if distance > 1e-3:
                    angle = np.arccos(
                        np.clip(float(np.dot(normals[-1], normals[0])), -1.0, 1.0)
                    )
                    curvature = min(angle / distance, 50.0)
            direction = dp_runtime["state"][
                min(t, int(dp_runtime["usable"]) - 1), :3
            ]
            state = np.concatenate(
                (
                    direction,
                    fsr,
                    q_hand,
                    control_pos,
                    _matrix_to_rotation_6d(root_r),
                    normal,
                    np.array([curvature], dtype=np.float32),
                )
            ).astype(np.float32)
            if state.shape != (48,):
                raise AssertionError(state.shape)
            return state

        @torch.no_grad()
        def _predict_dp_target(self, t: int) -> tuple[np.ndarray, np.ndarray]:
            assert dp_runtime is not None
            interval = int(dp_runtime["interval"])
            if dp_runtime["state_source"] == "teacher":
                indices = t - interval * np.arange(7, -1, -1)
                indices = np.clip(indices, 0, int(dp_runtime["usable"]) - 1)
                state_np = dp_runtime["state"][indices]
            else:
                # The first live plan intentionally uses the exact recorded
                # bootstrap [0, 5, ..., 35].  For later plans, append the
                # current live frame *before* inference so the history ends at
                # t rather than being one DP interval stale.  Appending after
                # inference also duplicated frame 35 on the second plan.
                if self.dp_calls > 0:
                    self.live_history.append(self._live_dp_state(t))
                state_np = np.stack(self.live_history)
            state = torch.as_tensor(
                state_np, device=env.device, dtype=torch.float32
            ).unsqueeze(0)
            batch = {
                "observation.state": state,
                "observation.environment_state": torch.zeros(
                    (1, 8, 1), device=env.device, dtype=torch.float32
                ),
            }
            batch = dp_runtime["pre"](batch)
            prediction = dp_runtime["policy"].diffusion.generate_actions(batch)
            prediction = dp_runtime["post"](prediction)[0].detach().cpu().numpy()
            action = prediction[min(1, len(prediction) - 1)]
            control_r = _rotation_6d_to_matrix(action[3:9])
            root_pos = action[:3] - control_r @ dp_runtime["offset"]
            root_pose = np.concatenate(
                (root_pos, _matrix_to_wxyz(control_r))
            ).astype(np.float32)
            self.dp_calls += 1

            teacher_idx = min(t + interval, int(dp_runtime["usable"]) - 1)
            pos_error = np.linalg.norm(action[:3] - dp_runtime["action"][teacher_idx, :3])
            q_error = np.mean(
                np.abs(action[9:25] - dp_runtime["action"][teacher_idx, 9:25])
            )
            self.dp_pos_errors.append(float(pos_error))
            self.dp_qpre_errors.append(float(q_error))
            print(
                f"[DP] call={self.dp_calls:4d} frame={t:4d} "
                f"teacher_target={teacher_idx:4d} "
                f"palm_error={pos_error*1000:6.2f}mm "
                f"qpre_error={q_error:7.4f}rad"
            )
            return root_pose, action[9:25].copy()

        def _update_dp_segment(self, t: int) -> None:
            if dp_runtime is None or t < 35:
                return
            interval = int(dp_runtime["interval"])
            if (t - 35) % interval != 0:
                return
            teacher_pose = palm_traj[t, env_idx].copy()
            if t == 35:
                self.segment_start = teacher_pose
                self.segment_qpre_start = dp_runtime["action"][t, 9:25].copy()
                self.current_qpre = self.segment_qpre_start.copy()
            else:
                self.segment_start = self.segment_target.copy()
                self.segment_qpre_start = self.current_qpre.copy()
            self.segment_target, self.segment_qpre_target = self._predict_dp_target(t)

        def _root_pose_for_frame(self, t: int) -> np.ndarray:
            teacher_pose = palm_traj[t, env_idx].copy()
            if replay_mode not in ("dp-palm", "dp-full") or t < 35:
                return teacher_pose

            interval = int(dp_runtime["interval"])
            alpha = ((t - 35) % interval + 1) / interval
            p = (
                (1.0 - alpha) * self.segment_start[:3]
                + alpha * self.segment_target[:3]
            )
            r0 = _wxyz_to_matrix(self.segment_start[3:7])
            r1 = _wxyz_to_matrix(self.segment_target[3:7])
            rotation = Slerp(
                [0.0, 1.0], R.from_matrix(np.stack((r0, r1)))
            )([alpha]).as_matrix()[0]
            return np.concatenate((p, _matrix_to_wxyz(rotation))).astype(np.float32)

        def _qpre_for_frame(self, t: int) -> np.ndarray:
            if dp_runtime is None or t < 35:
                return self.current_qpre
            interval = int(dp_runtime["interval"])
            alpha = ((t - 35) % interval + 1) / interval
            self.current_qpre = (
                (1.0 - alpha) * self.segment_qpre_start
                + alpha * self.segment_qpre_target
            ).astype(np.float32)
            return self.current_qpre

        def _write_root_pose(self, palm_pose: torch.Tensor) -> None:
            root_state = torch.cat(
                [palm_pose, torch.zeros(6, device=env.device, dtype=palm_pose.dtype)]
            ).unsqueeze(0)
            robot.write_root_state_to_sim(root_state)

        def _apply_pd_wrench(self, palm_pose: torch.Tensor) -> tuple[float, float]:
            root = robot.data.root_link_pose_w[0]
            velocity = robot.data.root_link_vel_w[0]
            target_pos = palm_pose[:3]
            target_r = _wxyz_to_matrix(palm_pose[3:7].detach().cpu().numpy())
            root_r = _wxyz_to_matrix(root[3:7].detach().cpu().numpy())

            pos_error = target_pos - root[:3]
            force = kp_pos * pos_error - kd_pos * velocity[:3]
            ori_error = R.from_matrix(target_r @ root_r.T).as_rotvec()
            ori_error_t = torch.as_tensor(
                ori_error, device=env.device, dtype=torch.float32
            )
            torque = kp_rot * ori_error_t - kd_rot * velocity[3:6]

            force_norm = torch.linalg.vector_norm(force).clamp(min=1e-8)
            torque_norm = torch.linalg.vector_norm(torque).clamp(min=1e-8)
            force = force * torch.clamp(max_force / force_norm, max=1.0)
            torque = torque * torch.clamp(max_torque / torque_norm, max=1.0)

            n_bodies = int(robot_body_mass.numel())
            forces = torch.zeros((1, n_bodies, 3), device=env.device)
            torques = torch.zeros_like(forces)
            if gravity_compensation:
                forces[0, :, 2] = robot_body_mass * 9.81
            forces[0, 0] += force
            torques[0, 0] += torque
            robot.write_external_wrench_to_sim(forces, torques)
            return float(torch.linalg.vector_norm(pos_error)), float(np.linalg.norm(ori_error))

        def __call__(self, obs):
            _ = obs
            t = self.t % num_steps
            env_idx_in_h5 = env_idx

            self._update_dp_segment(t)

            if env.sim.model.nmocap > 0:
                obj_pose = torch.from_numpy(
                    obj_traj[t, env_idx_in_h5]
                ).to(env.device)
                env.sim.data.mocap_pos[:, 0, :] = obj_pose[:3]
                env.sim.data.mocap_quat[:, 0, :] = obj_pose[3:]

            palm_pose = torch.from_numpy(self._root_pose_for_frame(t)).to(env.device)
            if palm_controller_type == "root_pose":
                self._write_root_pose(palm_pose)
                track_pos_error, track_ori_error = 0.0, 0.0
            else:
                # Initialize at the exact first teacher pose, then let dynamics
                # and the bounded 6-D wrench execute all subsequent targets.
                if self.t == 0:
                    self._write_root_pose(palm_pose)
                    env.sim.forward()
                track_pos_error, track_ori_error = self._apply_pd_wrench(palm_pose)
            if t >= 35:
                self.palm_tracking_pos_errors.append(track_pos_error)
                self.palm_tracking_ori_errors.append(track_ori_error)

            # Seed the finger dynamics from teacher q_actual, then stop
            # overwriting joints once the q_pre/FSR controller takes over.
            if not finger_mode or t <= 35:
                q_vals = torch.from_numpy(
                    q_mapper(q_traj[t, env_idx_in_h5])
                ).to(env.device)
                zero_vel = torch.zeros_like(q_vals).unsqueeze(0)
                env.scene["robot"].write_joint_state_to_sim(
                    position=q_vals.unsqueeze(0),
                    velocity=zero_vel,
                )

            env.sim.forward()

            # Read the contact sensor for the current replay geometry.  Mask
            # force slots without an actual geom-target match so numerical
            # residuals are never reported as FSR contact.
            fsr_sensor = env.scene["fsr_contact"]
            fsr_sensor.update(0.0)
            sensor_data = fsr_sensor.data
            fsr_live = torch.zeros(16, device=env.device)
            if sensor_data.force is not None:
                fsr_live = torch.linalg.vector_norm(sensor_data.force[0], dim=-1)
                if sensor_data.found is not None:
                    fsr_live = torch.where(
                        sensor_data.found[0] > 0,
                        fsr_live,
                        torch.zeros_like(fsr_live),
                    )
                finger_count = sum(
                    int(float(fsr_live[ids].max()) > 0.01)
                    for ids in ([4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15])
                )
                if t >= 35:
                    self.active_steps += 1
                    self.contact3_steps += int(finger_count >= 3)
                    self.max_fsr_norm = max(
                        self.max_fsr_norm,
                        float(torch.linalg.vector_norm(fsr_live)),
                    )
                if t % 100 == 0:
                    active = (
                        torch.nonzero(fsr_live > 0.01, as_tuple=False)
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                    recorded_norm = (
                        float(np.linalg.norm(fsr_traj[t, env_idx_in_h5]))
                        if fsr_traj is not None
                        else float("nan")
                    )
                    impedance_diag = ""
                    if isinstance(
                        finger_controller, MultiContactFingerImpedanceController
                    ):
                        diag = finger_controller.diagnostics()
                        impedance_diag = (
                            f" regions={diag['contact_regions']:.0f} "
                            f"dq={diag['max_contact_offset']:.3f} "
                            f"dqdot={diag['max_contact_offset_rate']:.3f} "
                            f"action={diag['max_action']:.2f}"
                        )
                    print(
                        f"[FSR] frame={t:4d} active={active} "
                        f"fingers={finger_count} "
                        f"live_norm={float(torch.linalg.vector_norm(fsr_live)):.2f} "
                        f"recorded_norm={recorded_norm:.2f}{impedance_diag}"
                    )

            penetration = 0.0
            penetration_dist = env.scene["hand_object_penetration"].data.dist
            if penetration_dist is not None:
                penetration = max(0.0, -float(penetration_dist.min()))
            if t >= 35:
                self.penetration_samples.append(penetration)
            if t % 100 == 0:
                print(
                    f"[Palm] frame={t:4d} executor={palm_controller_type} "
                    f"track=({track_pos_error*1000:.2f}mm,{track_ori_error:.4f}rad) "
                    f"penetration={penetration*1000:.3f}mm"
                )

            self.t += 1
            if finger_controller is None or t < 35:
                return torch.zeros((1, action_dim), device=env.device)

            qpre = torch.as_tensor(
                self._qpre_for_frame(t), device=env.device, dtype=torch.float32
            ).unsqueeze(0)
            finger_controller.set_q_pre(qpre)
            q_hand = env.scene["robot"].data.joint_pos[:, :16]
            qd_hand = env.scene["robot"].data.joint_vel[:, :16]
            if isinstance(
                finger_controller, MultiContactFingerImpedanceController
            ):
                return finger_controller(
                    fsr_live.unsqueeze(0), q_hand, qd_hand
                )
            tau_hand = env.scene["robot"].data.qfrc_actuator[:, :16]
            q_fake = torch.cat(
                (torch.zeros((1, 6), device=env.device), q_hand), dim=-1
            )
            finger_obs = torch.cat((fsr_live.unsqueeze(0), q_fake, tau_hand), dim=-1)
            return finger_controller({"policy": finger_obs})

    print(f"[INFO] Replaying from: {os.path.basename(file_path)}")
    replay_policy = ReplayPolicy()
    try:
        if viewer == "headless":
            for _ in range(num_steps):
                obs = viewer_env.get_observations()
                action = replay_policy(obs)
                viewer_env.step(action)
            penetration = np.asarray(replay_policy.penetration_samples)
            dp_pos = np.asarray(replay_policy.dp_pos_errors)
            dp_qpre = np.asarray(replay_policy.dp_qpre_errors)
            track_pos = np.asarray(replay_policy.palm_tracking_pos_errors)
            track_ori = np.asarray(replay_policy.palm_tracking_ori_errors)
            print(
                f"[RESULT] mode={replay_mode} state={dp_state_source} "
                f"palm={palm_controller_type} finger={finger_controller_type} "
                f"active_steps={replay_policy.active_steps} "
                f"contact3={100.0*replay_policy.contact3_steps/max(replay_policy.active_steps,1):.1f}% "
                f"max_fsr_norm={replay_policy.max_fsr_norm:.2f} "
                f"dp_palm_mean={1000*float(dp_pos.mean()) if dp_pos.size else 0.0:.2f}mm "
                f"dp_palm_p95={1000*float(np.percentile(dp_pos,95)) if dp_pos.size else 0.0:.2f}mm "
                f"dp_qpre_mean={float(dp_qpre.mean()) if dp_qpre.size else 0.0:.4f}rad "
                f"track_pos_p95={1000*float(np.percentile(track_pos,95)) if track_pos.size else 0.0:.2f}mm "
                f"track_ori_p95={float(np.percentile(track_ori,95)) if track_ori.size else 0.0:.4f}rad "
                f"penetration_p95={1000*float(np.percentile(penetration,95)) if penetration.size else 0.0:.3f}mm "
                f"penetration_max={1000*float(penetration.max()) if penetration.size else 0.0:.3f}mm"
            )
        elif viewer == "native":
            NativeMujocoViewer(viewer_env, replay_policy).run()
        else:
            ViserPlayViewer(viewer_env, replay_policy).run()
    finally:
        viewer_env.close()


def _resolve_input_file(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    inverted_files = glob.glob("./finger_compliance_control/data/headless/*_inverted.h5")
    inverted_files += glob.glob("./finger_compliance_control/data/*_inverted.h5")
    if not inverted_files:
        return None
    return max(inverted_files, key=os.path.getctime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay inverted hand trajectory with leap_hand.xml (hand-only)."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to *_inverted.h5. If omitted, use latest one in data folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:0 or cpu. Default: auto-select.",
    )
    parser.add_argument(
        "--env-idx",
        type=int,
        default=0,
        help="Environment index from multi-env H5 to replay.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        choices=("headless", "native", "viser"),
        default="native",
        help="Viewer backend. Use 'viser' to avoid GLX/OpenGL context issues.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=(
            "teacher",
            "dp-palm",
            "teacher-palm-dp-qpre",
            "dp-full",
        ),
        default="teacher",
        help=(
            "teacher: exact pose/q; dp-palm: DP palm + teacher q; "
            "teacher-palm-dp-qpre: teacher palm + DP q_pre/FSR control; "
            "dp-full: DP palm + DP q_pre/FSR control"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--dp-dataset", type=str, default=str(DEFAULT_DP_DATASET))
    parser.add_argument("--dp-model", type=str, default=str(DEFAULT_DP_MODEL))
    parser.add_argument("--dp-inference-steps", type=int, default=20)
    parser.add_argument(
        "--dp-state-source",
        choices=("teacher", "live"),
        default="teacher",
        help="Use recorded DP history or online FSR/q/palm/geometry history",
    )
    parser.add_argument(
        "--dp-interval",
        type=int,
        default=5,
        help="Simulation frames per DP target (5 at 100 Hz gives 20 Hz DP)",
    )
    parser.add_argument(
        "--finger-controller",
        choices=("legacy", "impedance"),
        default="impedance",
        help="Low-level q_pre executor used by DP finger replay",
    )
    parser.add_argument(
        "--palm-controller",
        choices=("root_pose", "pd_wrench"),
        default="root_pose",
        help="Execute DP palm targets by exact root writes or a physical 6-D PD wrench",
    )
    parser.add_argument("--kp-pos", type=float, default=500.0)
    parser.add_argument("--kd-pos", type=float, default=40.0)
    parser.add_argument("--kp-rot", type=float, default=50.0)
    parser.add_argument("--kd-rot", type=float, default=5.0)
    parser.add_argument("--max-force", type=float, default=100.0)
    parser.add_argument("--max-torque", type=float, default=10.0)
    parser.add_argument(
        "--gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    file_path = _resolve_input_file(args.file)
    if file_path is None:
        print("[ERROR] No *_inverted.h5 files found in ./finger_copliance_control/data/")
        return

    run_replay(
        file_path=file_path,
        device=args.device,
        env_idx=args.env_idx,
        viewer=args.viewer,
        replay_mode=args.replay_mode,
        dp_dataset=args.dp_dataset,
        dp_model=args.dp_model,
        dp_inference_steps=args.dp_inference_steps,
        dp_interval=args.dp_interval,
        max_steps=args.max_steps,
        dp_state_source=args.dp_state_source,
        finger_controller_type=args.finger_controller,
        palm_controller_type=args.palm_controller,
        kp_pos=args.kp_pos,
        kd_pos=args.kd_pos,
        kp_rot=args.kp_rot,
        kd_rot=args.kd_rot,
        max_force=args.max_force,
        max_torque=args.max_torque,
        gravity_compensation=args.gravity_compensation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
