from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
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

_LEAPHAND_XML = Path("/home/rimlab/Code/Hand_Compliance_Control/src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_0.xml")
_ENABLE_HAND_OBJECT_ONLY_COLLISION = False
_HAND_CONTYPE = 2
_HAND_CONAFFINITY = 4
_OBJECT_CONTYPE = 4
_OBJECT_CONAFFINITY = 2

_FSR_CACHE = {}
_FSR_COLOR_FIELDS_READY = set()


def _finger_geom_contact_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    """Privileged whole-finger contact sensors for matched A/B/C evaluation."""
    body_patterns = {
        "index": r"^(mcp_joint|pip|pip_1_fsr|dip|dip_1_fsr|fingertip|tip_1_fsr)$",
        "middle": r"^(mcp_joint_2|pip_2|pip_2_fsr|dip_2|dip_2_fsr|fingertip_2|tip_2_fsr)$",
        "ring": r"^(mcp_joint_3|pip_3|pip_3_fsr|dip_3|dip_3_fsr|fingertip_3|tip_3_fsr)$",
        "thumb": r"^(pip_4|thumb_pip|thumb_dip|thumb_fingertip|thumb_(pip|dip|tip)_fsr)$",
    }
    return tuple(
        ContactSensorCfg(
            name=f"finger_geom_contact_{finger_name}",
            primary=ContactMatch(mode="body", pattern=body_pattern, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="target_ball", entity="target"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        )
        for finger_name, body_pattern in body_patterns.items()
    )


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
    # Mocap body is kinematic: mouse perturbation can place it directly.
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    target_geoms = (
        body.add_geom(
            name="target_capsule_medium_geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.15, 0.08],
            rgba=[0.2, 0.6, 1.0, 1.0],
            mass=1,
        ),
        # body.add_geom(
        #     name="target_box_medium_geom",
        #     type=mujoco.mjtGeom.mjGEOM_BOX,
        #     size=[0.15, 0.12, 0.12],
        #     rgba=[0.2, 1.0, 0.5, 1.0],
        #     mass=1,
        # ),
        # body.add_geom(
        #     name="target_cylinder_medium_geom",
        #     type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        #     size=[0.12, 0.15],
        #     rgba=[0.3, 0.7, 1.0, 1.0],
        #     mass=1,
        # ),
        # body.add_geom(
        #     name="target_ellipsoid_medium_geom",
        #     type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        #     size=[0.18, 0.15, 0.12],
        #     rgba=[0.25, 0.8, 0.9, 1.0],
        #     mass=1,
        # ),
    )
    for geom in target_geoms:
        if _ENABLE_HAND_OBJECT_ONLY_COLLISION:
            geom.contype = _OBJECT_CONTYPE
            geom.conaffinity = _OBJECT_CONAFFINITY
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
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.0003, 0.8377)),
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
            sensors=(fsr_contact_cfg, *_finger_geom_contact_sensor_cfgs()),
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
            body_name="base_link",
            distance=2.0,
        ),
        episode_length_s=1e10 if play else 50.0,
    )

def leaphand_contact_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
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
        self.K_prox = 1.2
        self.K_mid = 0.5
        self.K_dist = 0.35
        self.D_force = 1.8
        self.K_limit_spring = 0.3

        self.q_pre_grasp_list = [0.8, 0.4, 0.3]
        # Optional high-level 16-D finger pre-shape.  When unset the controller
        # keeps its original fixed-grasp behaviour.  A hierarchical policy can
        # update this target online while the FSR loop supplies fast contact
        # corrections around it.
        self.external_q_pre: torch.Tensor | None = None
        self.K_pre_free = float(kwargs.get("K_pre_free", 0.2))
        self.K_pre_contact = float(kwargs.get("K_pre_contact", 0.05))
        self.S_contact_threshold = 0.15
        self.reset_speed = 0.1

        # Stabilization parameters to reduce glittering while keeping fast response.
        self.alpha_obs = 0.2
        self.alpha_ctrl = 0.15
        self.contact_on_threshold = 0.20
        self.contact_off_threshold = 0.12
        self.error_deadband = 0.03
        self.ds_clip = 0.2
        self.action_rate_limit = 0.25

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

    def set_q_pre(self, q_pre: torch.Tensor | None) -> None:
        """Set a per-environment 16-D pre-shape target in hand joint order."""
        if q_pre is None:
            self.external_q_pre = None
            return
        q_pre = q_pre.to(device=self.device, dtype=torch.float32)
        if q_pre.ndim == 1:
            q_pre = q_pre.unsqueeze(0)
        expected = (self.num_envs, 16)
        if tuple(q_pre.shape) != expected:
            raise ValueError(f"q_pre must have shape {expected}, got {tuple(q_pre.shape)}")
        self.external_q_pre = q_pre.detach().clone()

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

            for joint_idx, target_comp, fallback_limit in zip(
                j_idx,
                (comp_p, comp_m, comp_d),
                self.q_pre_grasp_list,
            ):

                current_q = q_curr[:, joint_idx]
                if self.external_q_pre is None:
                    limit_val = torch.full_like(current_q, fallback_limit)
                else:
                    limit_val = self.external_q_pre[:, joint_idx - 6]
                dist_to_limit = limit_val - current_q

                limit_active = (~has_contact) & (current_q > limit_val) & (target_comp > 0)
                spring_delta = dist_to_limit * self.K_limit_spring
                safe_delta = torch.where(limit_active, spring_delta, target_comp)
                delta_comp[:, joint_idx] = safe_delta

        if self.external_q_pre is None:
            unused_hand_joints = [7, 11, 15, 19]
            for uj in unused_hand_joints:
                delta_comp[:, uj] = self.reset_speed * (self.q_nom[:, uj] - q_curr[:, uj])
        else:
            # No-contact fingers follow the high-level geometry more strongly;
            # contacted fingers retain only a weak anchor so FSR compliance wins.
            q_pre_error = self.external_q_pre - q_curr[:, 6:22]
            contact_per_joint = self.contact_state.repeat_interleave(4, dim=1)
            k_pre = torch.where(
                contact_per_joint,
                torch.full_like(q_pre_error, self.K_pre_contact),
                torch.full_like(q_pre_error, self.K_pre_free),
            )
            delta_comp[:, 6:22] += k_pre * q_pre_error

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

class FingertipForceComplianceController:
    """Compliance controller using fingertip 3D force instead of 16D FSR.

    Designed for sim2real — real hardware only needs 4 fingertip force sensors
    (3D force each, e.g. vision-based tactile), not 16 discrete FSRs.

    Key differences from LeapHandComplianceController:
    - Contact signal: |fingertip_force_3d| replaces distal FSR (s_d)
    - Wrapping: s_p ≈ 0 for index/middle/ring (proximal FSR was mostly zero),
      so wrapping ≈ s_d everywhere — same behavior for 3 fingers,
      thumb loses minor pad-contact modulation.
    - New: force direction used for proximal joint alignment (compensates
      for loss of proximal FSR signal on thumb).
    - No dependency on 16D FSR in observation — reads ContactSensor directly.

    Requires env passed via kwargs: FingertipForceComplianceController(..., env=env)
    """

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs
        self._env = kwargs.get("env", None)

        # ── Compliance parameters (same as original) ──
        self.S_min = 0.6
        self.S_max = 1.5
        self.K_prox = 0.8
        self.K_mid = 0.32
        self.K_dist = 0.2
        self.D_force = 1.2
        self.K_limit_spring = 0.3

        self.q_pre_grasp_list = [0.8, 0.4, 0.3]
        self.reset_speed = 0.1

        # ── Smoothing ──
        self.alpha_obs = 0.2
        self.alpha_ctrl = 0.15
        self.contact_on_threshold = 0.20
        self.contact_off_threshold = 0.12
        self.error_deadband = 0.03
        self.ds_clip = 0.2
        self.action_rate_limit = 0.15

        # ── State ──
        self.q_nom = torch.zeros((num_envs, 22), device=device)
        self.ft_obs = torch.zeros((num_envs, 12), device=device)   # [B, 12] smoothed force
        self.ft_ctrl = torch.zeros((num_envs, 12), device=device)  # [B, 12] control force
        self.contact_state = torch.zeros((num_envs, 4), dtype=torch.bool, device=device)
        self.prev_action = torch.zeros((num_envs, 16), device=device)
        self.is_init = False

        # Finger configs: joints + which fingertip force index
        # ft_idx → slice ft_ctrl[:, ft_idx*3 : (ft_idx+1)*3]
        self.finger_configs = [
            {"name": "index",  "j": [6, 8, 9],   "ft_idx": 0},
            {"name": "middle", "j": [10, 12, 13], "ft_idx": 1},
            {"name": "ring",   "j": [14, 16, 17], "ft_idx": 2},
            {"name": "thumb",  "j": [18, 20, 21], "ft_idx": 3},
        ]

        # Per-hand-joint action scale (same as original)
        self.action_scale_hand = torch.tensor(
            [0.12, 0.08, 0.08, 0.05] * 4,
            device=device, dtype=torch.float32,
        )

    def _read_fingertip_force(self) -> torch.Tensor:
        """Read raw 3D force from ContactSensor distal sensors (6,9,12,15).
        Returns [B, 12]: index(0:3), middle(3:6), ring(6:9), thumb(9:12).
        """
        if self._env is None:
            return torch.zeros(self.num_envs, 12, device=self.device)
        try:
            sensor = self._env.scene["fsr_contact"]
            force = sensor.data.force  # [B, N, 3]
            if force is None:
                return torch.zeros(self.num_envs, 12, device=self.device)
        except Exception:
            return torch.zeros(self.num_envs, 12, device=self.device)

        B = force.shape[0]
        n = force.shape[1]
        result = torch.zeros(B, 12, device=force.device, dtype=force.dtype)
        for i, idx in enumerate((6, 9, 12, 15)):
            if idx < n:
                result[:, i * 3 : (i + 1) * 3] = force[:, idx, :]
        return result

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
        q_curr = policy_obs[:, 16:38]  # joint positions [B, 22]

        # ── Read fingertip 3D force from ContactSensor ──
        ft_raw = self._read_fingertip_force()  # [B, 12]

        # ── Joint actuator torque for blind collision detection ──
        # policy_obs[:, 38:54] = qfrc_actuator hand [B, 16]
        tau_hand = policy_obs[:, 38:54] if policy_obs.shape[-1] >= 54 else None

        if not self.is_init:
            self.q_nom[:] = q_curr
            self.is_init = True
            self.ft_obs[:] = ft_raw
            self.ft_ctrl[:] = ft_raw

        last_ctrl = self.ft_ctrl.clone()
        self.ft_obs = self.alpha_obs * self.ft_obs + (1 - self.alpha_obs) * ft_raw
        self.ft_ctrl = self.alpha_ctrl * self.ft_ctrl + (1 - self.alpha_ctrl) * ft_raw
        dft_ctrl = self.ft_ctrl - last_ctrl  # [B, 12]

        delta_comp = torch.zeros_like(q_curr)

        for fi, config in enumerate(self.finger_configs):
            j_idx = config["j"]         # [proximal, middle, distal] joint indices
            ft_i = config["ft_idx"]     # fingertip force index

            # ── Fingertip force (3D vector) ──
            f_vec = self.ft_ctrl[:, ft_i * 3 : (ft_i + 1) * 3]  # [B, 3]
            f_mag = torch.linalg.vector_norm(f_vec, dim=-1)      # [B]
            df_vec = dft_ctrl[:, ft_i * 3 : (ft_i + 1) * 3]      # [B, 3]
            df_mag = torch.linalg.vector_norm(df_vec, dim=-1)
            df_mag = torch.clamp(df_mag, min=-self.ds_clip, max=self.ds_clip)

            # ── Force direction ──
            f_norm = f_mag.clamp(min=1e-8)
            f_dir = f_vec / f_norm.unsqueeze(-1)  # [B, 3] unit direction
            f_dir_z = f_dir[:, 2]                  # alignment with fingertip Z

            # ── Contact detection (fingertip) ──
            f_mag_obs = torch.linalg.vector_norm(
                self.ft_obs[:, ft_i * 3 : (ft_i + 1) * 3], dim=-1
            )
            prev_contact = self.contact_state[:, fi]
            has_contact = torch.where(
                prev_contact,
                f_mag_obs >= self.contact_off_threshold,
                f_mag_obs >= self.contact_on_threshold,
            )
            self.contact_state[:, fi] = has_contact

            # ── Blind collision detection ──
            # 指尖无力 + 近端关节力矩大 → 手指根部撞到不规则物体 → 需避让
            # 只用近端关节(actuator index 1)：正常指尖按压时力矩集中在
            # 中/远关节，近端力矩低；盲碰撞时根部被推挤，近端力矩高。
            blind_collision = torch.zeros_like(f_mag, dtype=torch.bool)
            if tau_hand is not None:
                # 每指 4 actuator: [abduction, proximal, middle, distal]
                tau_proximal = tau_hand[:, fi * 4 + 1]  # proximal joint only
                blind_collision = (
                    (f_mag_obs < self.contact_on_threshold)
                    & (tau_proximal.abs() > 0.8)
                )

            # ── s_d: force magnitude (replaces distal FSR) ──
            # Blind collision → 伪造高 s_d 触发回撤
            s_d = torch.where(
                blind_collision,
                torch.full_like(f_mag, self.S_max + 0.5),
                f_mag,
            )
            ds_d = df_mag

            # ── s_p: proximal pressure (replaces proximal FSR mean) ──
            s_p = f_mag * (1.0 - f_dir_z.abs()).clamp(min=0) * 0.5
            ds_p = df_mag * (1.0 - f_dir_z.abs()).clamp(min=0) * 0.5
            ds_p = torch.clamp(ds_p, min=-self.ds_clip, max=self.ds_clip)

            # ── Proximal joint compliance ──
            # Blind collision → 近端也回撤
            e_p = self._compute_interval_error(s_p)
            e_p = torch.where(
                torch.abs(e_p) < self.error_deadband, torch.zeros_like(e_p), e_p
            )
            comp_p = torch.where(
                blind_collision,
                -self.K_prox * 0.5,  # 近端回撤
                self.K_prox * e_p - self.D_force * ds_p,
            )

            # ── Middle + Distal joint compliance ──
            e_d = self._compute_interval_error(s_d)
            e_d = torch.where(
                torch.abs(e_d) < self.error_deadband, torch.zeros_like(e_d), e_d
            )
            wrapping_factor = torch.clamp(s_d - s_p, min=0)
            adj_e_d = e_d - 0.5 * wrapping_factor

            comp_m = self.K_mid * adj_e_d - 0.6 * self.D_force * ds_d
            comp_d = self.K_dist * adj_e_d - 0.2 * self.D_force * ds_d

            # ── Apply with joint limits ──
            # Blind collision 时不触发 limit_spring（手指已经在回撤了）
            for joint_idx, target_comp, limit_val in zip(
                j_idx, (comp_p, comp_m, comp_d), self.q_pre_grasp_list,
            ):
                current_q = q_curr[:, joint_idx]
                dist_to_limit = limit_val - current_q
                limit_active = (
                    (~has_contact)
                    & (~blind_collision)
                    & (current_q > limit_val)
                    & (target_comp > 0)
                )
                spring_delta = dist_to_limit * self.K_limit_spring
                safe_delta = torch.where(limit_active, spring_delta, target_comp)
                delta_comp[:, joint_idx] = safe_delta

        # ── Sidesway recovery (same as original) ──
        unused_hand_joints = [7, 11, 15, 19]
        for uj in unused_hand_joints:
            delta_comp[:, uj] = self.reset_speed * (self.q_nom[:, uj] - q_curr[:, uj])

        # ── Action normalization + rate limiting (same as original) ──
        hand_delta = delta_comp[:, 6:]
        raw_action = hand_delta / self.action_scale_hand.unsqueeze(0)
        action_cmd = torch.tanh(raw_action)

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
class LeapHandControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    """用于传递给采集脚本的配置"""
    # policy_class: type = NullComplianceController
    policy_class: type = LeapHandComplianceController
    # policy_class: type = FingertipForceComplianceController
    amplitude: float = 0.5
