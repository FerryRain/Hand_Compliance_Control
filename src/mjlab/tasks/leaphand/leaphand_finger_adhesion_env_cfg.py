from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlMotorActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointRelativePositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_LEAPHAND_XML = Path("/home/rimlab/Code/Hand_Compliance_Control/src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_adhesion.xml")
_ENABLE_HAND_OBJECT_ONLY_COLLISION = False
_HAND_CONTYPE = 2
_HAND_CONAFFINITY = 4
_OBJECT_CONTYPE = 4
_OBJECT_CONAFFINITY = 2

_FSR_CACHE = {}
_FSR_COLOR_FIELDS_READY = set()


def _load_leaphand_spec(enable_hand_object_only_collision: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_LEAPHAND_XML))
    if not enable_hand_object_only_collision:
        return spec

    hand_geom_regex = re.compile(
        r"^(?:"
        r"palm_.*|mcp_.*|pip(?:_\d+)?_geom|dip(?:_\d+)?_geom|"
        r"fingertip(?:_\d+)?_geom|thumb_.*|tip_\d+_fsr_geom|"
        r"pip_\d+_fsr_geom|dip_\d+_fsr_geom"
        r")$"
    )


    for geom in spec.geoms:
        geom_name = geom.name or ""
        if hand_geom_regex.fullmatch(geom_name):
            geom.contype = _HAND_CONTYPE
            geom.conaffinity = _HAND_CONAFFINITY

    return spec


def fsr_force_and_visual_logic(
    env: ManagerBasedRlEnv,
    sensor_name: str = "fsr_contact",
    fsr_regex: str = r".*_fsr_geom$",
    contact_rgba: tuple[float, float, float, float] = (0.2, 1.0, 0.2, 0.9),
    default_rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
    display_forces: bool = True,
    display_every: int = 5,
    display_top_k: int = 8,
    expected_num_fsrs: int = 16,
) -> torch.Tensor:
    """Read FSR forces from ContactSensor and keep optional FSR geom coloring."""
    _ = (display_forces, display_every, display_top_k)

    m = env.sim.mj_model
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor), (
        f"{sensor_name} must be ContactSensor, got {type(sensor).__name__}"
    )
    sensor_data = sensor.data
    assert sensor_data.force is not None, "ContactSensor must expose 'force' field"

    # force: [B, N, 3], convert to per-FSR magnitude [B, N].
    forces = torch.linalg.vector_norm(sensor_data.force, dim=-1)

    num_envs = forces.shape[0]
    num_fsrs = forces.shape[1]
    forces_tensor = torch.zeros(
        (num_envs, expected_num_fsrs),
        device=forces.device,
        dtype=forces.dtype,
    )
    copy_count = min(expected_num_fsrs, num_fsrs)
    forces_tensor[:, :copy_count] = forces[:, :copy_count]

    env_ptr = id(env)

    if env_ptr not in _FSR_COLOR_FIELDS_READY:
        env.sim.expand_model_fields(("geom_rgba",))
        _FSR_COLOR_FIELDS_READY.add(env_ptr)

    if env_ptr not in _FSR_CACHE:
        pattern = re.compile(fsr_regex)
        _FSR_CACHE[env_ptr] = [
            gid for gid in range(m.ngeom)
            if (name := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)) and pattern.match(name)
        ]
    fsr_ids = _FSR_CACHE[env_ptr]

    if num_envs > 0 and len(fsr_ids) > 0:
        sim_geom_rgba = env.sim.model.geom_rgba
        c_rgba_t = torch.as_tensor(contact_rgba, device=forces.device, dtype=sim_geom_rgba.dtype)
        d_rgba_t = torch.as_tensor(default_rgba, device=forces.device, dtype=sim_geom_rgba.dtype)

        for fsr_idx, gid in enumerate(fsr_ids):
            if fsr_idx >= expected_num_fsrs:
                break

            if sim_geom_rgba.ndim == 2:
                # Shared visual buffer: use env-0 state as fallback.
                color = c_rgba_t if forces_tensor[0, fsr_idx] > 0.0 else d_rgba_t
                sim_geom_rgba[gid] = color
            else:
                # Per-env visual buffer: color each environment independently.
                active_env_mask = forces_tensor[:, fsr_idx] > 0.0
                sim_geom_rgba[active_env_mask, gid] = c_rgba_t
                sim_geom_rgba[~active_env_mask, gid] = d_rgba_t

    return forces_tensor

def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return absolute robot joint positions with shape [num_envs, num_joints]."""
    asset = env.scene[asset_cfg.name]

    return asset.data.joint_pos


def qfrc_actuator_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return actuator force for hand joints (index 6..21) with shape [num_envs, 16].

    Used by the Torque2FSR module to estimate contact forces from motor torque
    when real FSR sensors are unavailable (sim2real bridge).
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.qfrc_actuator[:, 6:22]

# --- Entity 配置 ---

def _get_target_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    # Mocap 胶囊：右键拖动定位，纯碰撞体，无 gap。吸附器在 FSR 上。
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    target_geoms = (
        body.add_geom(
            name="target_capsule_medium_geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.15, 0.08],
            rgba=[0.2, 0.6, 1.0, 1.0],
            mass=1,
        ),
    )
    # conaffinity=9: bit0=1 正常碰撞, bit3=8 给 FSR(contype=8) 做 gap 吸附检测
    for geom in target_geoms:
        geom.conaffinity = 9
    return spec

# --- 环境配置构建 ---

def _make_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(
            enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    # LeapHand joints are named "0".."15" in this XML.
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=500.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.183,
                "joint3": -3.1416,
                "joint4": 3.1415,
                "joint5": 1.183,
                "joint6": -1.569,
                "13": 1.57},
        ),
    )
    
    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.7007, 0.0003, 0.8377),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    fsr_contact_cfg = ContactSensorCfg(
        name="fsr_contact",
        primary=ContactMatch(mode="geom", pattern=r".*_fsr_geom$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="target_ball", entity="target"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    observations = {
        "policy": ObservationGroupCfg({
            "fsr_forces": ObservationTermCfg(
                func=fsr_force_and_visual_logic,
                params={
                    "sensor_name": "fsr_contact",
                    "fsr_regex": r".*_fsr_geom$",
                    "display_forces": True,
                    "display_every": 5,
                    "display_top_k": 8,
                },
            ),
            "joint_pos": ObservationTermCfg(
                func=joint_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "qfrc_actuator_hand": ObservationTermCfg(
                func=qfrc_actuator_hand,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
        })
    }

    actions: dict[str, ActionTermCfg] = {
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.08,
            use_default_offset=False
        )
    }

    return ManagerBasedRlEnvCfg(
        decimation=5, # type: ignore
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=(fsr_contact_cfg,),
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
            nconmax=2000,
        ),
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base_link",
            distance=2.0,
        ),
        episode_length_s=1e10 if play else 50.0,
    )

def leaphand_adhesion_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_env_cfg(num_envs=num_envs, play=play)

class LeapHandComplianceController:
    """
    针对 22 维架构 (6臂+16手) 的底层顺应性控制器

    修改说明：
    1. 解锁逻辑：仅当手掌 (FSR 0-3) 检测到压力时，手指才启动顺应性调节。
    2. 拇指统一：拇指根部关节 (18, 19) 采用与其他手指相同的回缩/下压逻辑。
    3. Torque2FSR 模式：指定 torque2fsr_model_path 时使用 MLP 从关节力矩
       估计 FSR，绕过真实 FSR 传感器 (sim2real bridge)。
    """
    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs

        self.S_min = 0.6
        self.S_max = 1.5
        self.K_prox = 0.8
        self.K_mid = 0.32
        self.K_dist = 0.2
        self.D_force = 1.2
        self.K_limit_spring = 0.3

        self.q_pre_grasp_list = [0.8, 0.4, 0.3]
        self.S_contact_threshold = 0.15
        self.reset_speed = 0.1

        # Stabilization parameters to reduce glittering while keeping fast response.
        self.alpha_obs = 0.2
        self.alpha_ctrl = 0.15
        self.contact_on_threshold = 0.20
        self.contact_off_threshold = 0.12
        self.error_deadband = 0.03
        self.ds_clip = 0.2
        self.action_rate_limit = 0.15

        self.q_nom = torch.zeros((num_envs, 22), device=device)
        self.fsr_obs = torch.zeros((num_envs, 16), device=device)
        self.fsr_ctrl = torch.zeros((num_envs, 16), device=device)
        self.contact_state = torch.zeros((num_envs, 4), dtype=torch.bool, device=device)
        self.prev_action = torch.zeros((num_envs, 16), device=device)
        self.is_init = False

        self.finger_configs = [
            {"name": "index", "j": [6, 8, 9], "p_fsr": [4, 5], "d_fsr": [6]},
            {"name": "middle", "j": [10, 12, 13], "p_fsr": [7, 8], "d_fsr": [9]},
            {"name": "ring",   "j": [14, 16, 17], "p_fsr": [10, 11], "d_fsr": [12]},
            {"name": "thumb", "j": [18, 20, 21], "p_fsr": [13, 14], "d_fsr": [15]},
        ]

        # Per-hand-joint action normalization for joints [6..21].
        # Proximal joints use larger scale; distal joints use smaller scale.
        self.action_scale_hand = torch.tensor(
            [
                0.12, 0.08, 0.08, 0.05,
                0.12, 0.08, 0.08, 0.05,
                0.12, 0.08, 0.08, 0.05,
                0.12, 0.08, 0.08, 0.05,
            ],
            device=device,
            dtype=torch.float32,
        )

        # ── Torque-to-FSR mode (sim2real bridge) ──
        self.torque2fsr: "Torque2FSRInference | None" = None  # type: ignore[name-defined]
        torque2fsr_path = kwargs.get("torque2fsr_model_path", None)
        if torque2fsr_path is not None:
            self._init_torque2fsr(torque2fsr_path)

    def _init_torque2fsr(self, model_path: str):
        """Lazy-load the Torque2FSR estimator."""
        import sys
        _proj_root = Path(__file__).resolve().parents[4]
        _models_dir = str(_proj_root / "finger_compliance_control" / "models")
        if _models_dir not in sys.path:
            sys.path.insert(0, _models_dir)

        from finger_compliance_control.models.torque2fsr import Torque2FSRInference
        self.torque2fsr = Torque2FSRInference(model_path, device=self.device)
        print(f"[INFO] Torque2FSR mode enabled, model: {model_path}")

    def _compute_interval_error(self, s: torch.Tensor) -> torch.Tensor:
        """Interval error: below S_min pushes (+), above S_max retracts (-)."""
        error = torch.zeros_like(s)
        low_mask = s < self.S_min
        high_mask = s > self.S_max
        error[low_mask] = self.S_min - s[low_mask]
        error[high_mask] = self.S_max - s[high_mask]
        return error

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        policy_obs = obs["policy"]
        fsr_raw = policy_obs[:, :16]
        q_curr = policy_obs[:, 16:38]

        # ── Torque-to-FSR: replace real FSR with MLP estimate ──
        if self.torque2fsr is not None:
            tau_hand = policy_obs[:, 38:54]        # [B, 16] qfrc_actuator hand
            q_hand = q_curr[:, 6:22]               # [B, 16] hand joints only
            # Match batch size (prev_action may be initialized for different num_envs)
            if self.prev_action.shape[0] != q_hand.shape[0]:
                self.prev_action = self.prev_action[:1].repeat(q_hand.shape[0], 1)
            fsr_estimated = self.torque2fsr(q_hand, tau_hand, self.prev_action)
            fsr_raw = fsr_estimated                # replace with torque-based estimate

        if not self.is_init:
            self.q_nom[:] = q_curr
            self.is_init = True
            self.fsr_obs[:] = fsr_raw
            self.fsr_ctrl[:] = fsr_raw

        last_ctrl = self.fsr_ctrl.clone()
        self.fsr_obs = self.alpha_obs * self.fsr_obs + (1 - self.alpha_obs) * fsr_raw
        self.fsr_ctrl = self.alpha_ctrl * self.fsr_ctrl + (1 - self.alpha_ctrl) * fsr_raw
        df_ctrl = self.fsr_ctrl - last_ctrl

        delta_comp = torch.zeros_like(q_curr)

        for finger_idx, config in enumerate(self.finger_configs):
            j_idx = config["j"]
            f_ids = config["p_fsr"] + config["d_fsr"]

            max_finger_force = torch.max(self.fsr_obs[:, f_ids], dim=1)[0]
            prev_contact = self.contact_state[:, finger_idx]
            has_contact = torch.where(
                prev_contact,
                max_finger_force >= self.contact_off_threshold,
                max_finger_force >= self.contact_on_threshold,
            )
            self.contact_state[:, finger_idx] = has_contact

            s_p = torch.mean(self.fsr_ctrl[:, config["p_fsr"]], dim=1)
            s_d = self.fsr_ctrl[:, config["d_fsr"]].squeeze(-1)
            ds_p = torch.mean(df_ctrl[:, config["p_fsr"]], dim=1)
            ds_p = torch.clamp(ds_p, min=-self.ds_clip, max=self.ds_clip)

            ds_d = df_ctrl[:, config["d_fsr"]].squeeze(-1)
            ds_d = torch.clamp(ds_d, min=-self.ds_clip, max=self.ds_clip)

            e_p = self._compute_interval_error(s_p)
            e_p = torch.where(
                torch.abs(e_p) < self.error_deadband,
                torch.zeros_like(e_p),
                e_p,
            )
            comp_p = self.K_prox * e_p - self.D_force * ds_p

            e_d = self._compute_interval_error(s_d)
            e_d = torch.where(
                torch.abs(e_d) < self.error_deadband,
                torch.zeros_like(e_d),
                e_d,
            )
            wrapping_factor = torch.clamp(s_d - s_p, min=0)
            adj_e_d = e_d - 0.5 * wrapping_factor

            comp_m = self.K_mid * adj_e_d - 0.6*self.D_force * ds_d
            comp_d = self.K_dist * adj_e_d - 0.2*self.D_force * ds_d

            for joint_idx, target_comp, limit_val in zip(
                j_idx,
                (comp_p, comp_m, comp_d),
                self.q_pre_grasp_list,
            ):

                current_q = q_curr[:, joint_idx]
                dist_to_limit = limit_val - current_q

                limit_active = (~has_contact) & (current_q > limit_val) & (target_comp > 0)
                spring_delta = dist_to_limit * self.K_limit_spring
                safe_delta = torch.where(limit_active, spring_delta, target_comp)
                delta_comp[:, joint_idx] = safe_delta

        unused_hand_joints = [7, 11, 15, 19]
        for uj in unused_hand_joints:
            delta_comp[:, uj] = self.reset_speed * (self.q_nom[:, uj] - q_curr[:, uj])

        hand_delta = delta_comp[:, 6:]
        raw_action = hand_delta / self.action_scale_hand.unsqueeze(0)
        action_cmd = torch.tanh(raw_action)

        # Rate-limit action changes to suppress high-frequency chattering.
        action_delta = torch.clamp(
            action_cmd - self.prev_action,
            min=-self.action_rate_limit,
            max=self.action_rate_limit,
        )
        action_out = self.prev_action + action_delta
        self.prev_action = action_out

        return action_out

class NullComplianceController:
    """一个不做任何补偿的控制器，用于对比测试"""
    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = obs["policy"].shape[0]
        return torch.zeros((batch_size, 16), device=self.device)

@dataclass
class LeapHandAdhesionControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    """用于传递给采集脚本的配置"""
    policy_class: type = NullComplianceController
    # policy_class: type = LeapHandComplianceController
    amplitude: float = 0.5
