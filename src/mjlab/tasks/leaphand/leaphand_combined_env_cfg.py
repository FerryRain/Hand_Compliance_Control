from __future__ import annotations

import csv
import os
import re
import time as _time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mink
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointRelativePositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# ── MCC 核心数学组件（override __call__ 中需要）──
from minimalist_compliance_control.minimalist_compliance_control.wrench_estimation import estimate_wrench
from minimalist_compliance_control.minimalist_compliance_control.utils import get_damping_matrix

# ── Sub-controllers ──
from mjlab.tasks.leaphand.leaphand_palm_mcc_env_cfg import (
    MCCPalmComplianceController,
    _get_or_build_observer,
)
from mjlab.tasks.leaphand.leaphand_finger_env_cfg import (
    LeapHandComplianceController,
    FingertipForceComplianceController,
)

# ── Palm observation functions ──
from mjlab.tasks.leaphand.leaphand_palm_mcc_env_cfg import (
    joint_pos,
    joint_vel_arm,
    qfrc_actuator_arm,
    qfrc_bias_arm,
    palm_jacobian,
    palm_jacobian_rot,
    palm_pos,
    palm_rot,
    target_body_pos,
    target_body_rot,
    _load_leaphand_spec,
    _get_target_box_spec,
)

# ── Finger observation functions ──
from mjlab.tasks.leaphand.leaphand_finger_env_cfg import (
    fsr_force_and_visual_logic,
    qfrc_actuator_hand,
)

# ==============================================================================
#  常量
# ==============================================================================

_LEAPHAND_XML = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/src/mjlab/asset_zoo/robots/"
    "xarm6_leap_hand/xarm6_leap_hand_nolimit.xml"
)
_ENABLE_HAND_OBJECT_ONLY_COLLISION = False
_HAND_CONTYPE = 2
_HAND_CONAFFINITY = 4
_OBJECT_CONTYPE = 4
_OBJECT_CONAFFINITY = 2


def _finger_geom_contact_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    """Privileged whole-finger contact sensors used only for simulation labels."""
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

# ==============================================================================
#  环境配置构建
# ==============================================================================


def _make_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建组合环境配置：手掌 MCC + 手指柔顺控制。

    观测分为两个命名组：
      - "palm"  (88-D): 与 MCC 手掌控制器布局完全一致
      - "finger" (54-D): 与手指柔顺控制器布局完全一致

    动作分为两个命名项：
      - "arm_pos"    (6-D):  arm 关节绝对位置 (JointPositionActionCfg)
      - "hand_delta" (16-D): hand 关节相对增量 (JointRelativePositionActionCfg)
    """
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(
            enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(joint[1-6])$",),
                    stiffness=3000.0,
                    damping=300.0,
                    effort_limit=500.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=5.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                # ── Arm joints: 与手掌 MCC 环境一致的起始姿态 ──
                "joint1": 0.0,
                "joint2": 1.183,
                "joint3": -1.541,
                "joint4": 3.1415,
                "joint5": 2.742,
                "joint6": -1.569,
                # ── Hand joints: 拇指对掌预抓取 ──
                "13": 1.57,
            },
        ),
    )

    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.6007, 0.0003, 0.7000)),
    )

    # ── FSR 接触传感器（手指控制器需要）──
    fsr_contact_cfg = ContactSensorCfg(
        name="fsr_contact",
        primary=ContactMatch(mode="geom", pattern=r".*_fsr_geom$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="target_ball", entity="target"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    # ── 双命名观测组：palm (88-D) + finger (54-D) ──
    observations: dict[str, ObservationGroupCfg] = {
        "palm": ObservationGroupCfg({
            "joint_pos": ObservationTermCfg(
                func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "joint_vel_arm": ObservationTermCfg(
                func=joint_vel_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_actuator_arm": ObservationTermCfg(
                func=qfrc_actuator_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_bias_arm": ObservationTermCfg(
                func=qfrc_bias_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "palm_jacobian": ObservationTermCfg(
                func=palm_jacobian,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_jacobian_rot": ObservationTermCfg(
                func=palm_jacobian_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_pos": ObservationTermCfg(
                func=palm_pos,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_rot": ObservationTermCfg(
                func=palm_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "target_pos": ObservationTermCfg(
                func=target_body_pos,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "target_ball"},
            ),
            "target_rot": ObservationTermCfg(
                func=target_body_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "target_ball"},
            ),
        }),
        # palm 布局: joint_pos(0:22), joint_vel_arm(22:28), qfrc_actuator_arm(28:34),
        #   qfrc_bias_arm(34:40), palm_jacobian(40:58), palm_jacobian_rot(58:76),
        #   palm_pos(76:79), palm_rot(79:82), target_pos(82:85), target_rot(85:88) = 88-D

        "finger": ObservationGroupCfg({
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
                func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_actuator_hand": ObservationTermCfg(
                func=qfrc_actuator_hand,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
        }),
        # finger 布局: fsr_forces(0:16), joint_pos(16:38), qfrc_actuator_hand(38:54) = 54-D
    }

    # ── 双命名动作项：arm 绝对位置 + hand 相对增量 ──
    actions: dict[str, ActionTermCfg] = {
        "arm_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^joint[1-6]$",),
            use_default_offset=False,
        ),
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


def mcc_palm_fsr_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    """纯手掌环境 + 手掌 FSR 力传感器。观测 92-D（88-D palm + 4-D palm FSR）。"""
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(
            enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(joint[1-6])$",),
                    stiffness=3000.0,
                    damping=300.0,
                    effort_limit=500.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=5.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                "joint1": 0.0, "joint2": 1.183, "joint3": -1.541,
                "joint4": 3.1415, "joint5": 2.742, "joint6": -1.569,
                "13": 1.57,
            },
        ),
    )

    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.0003, 0.7000)),
    )

    # FSR 接触传感器（只取手掌 4 个 FSR）
    fsr_contact_cfg = ContactSensorCfg(
        name="fsr_contact",
        primary=ContactMatch(mode="geom", pattern=r"^palm_[1-4]_fsr_geom$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="target_ball", entity="target"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    observations = {
        "policy": ObservationGroupCfg({
            "joint_pos": ObservationTermCfg(
                func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "joint_vel_arm": ObservationTermCfg(
                func=joint_vel_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_actuator_arm": ObservationTermCfg(
                func=qfrc_actuator_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_bias_arm": ObservationTermCfg(
                func=qfrc_bias_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "palm_jacobian": ObservationTermCfg(
                func=palm_jacobian,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_jacobian_rot": ObservationTermCfg(
                func=palm_jacobian_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_pos": ObservationTermCfg(
                func=palm_pos,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_rot": ObservationTermCfg(
                func=palm_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "target_pos": ObservationTermCfg(
                func=target_body_pos,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "target_ball"},
            ),
            "target_rot": ObservationTermCfg(
                func=target_body_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "target_ball"},
            ),
            "palm_fsr": ObservationTermCfg(
                func=fsr_force_and_visual_logic,
                params={
                    "sensor_name": "fsr_contact",
                    "fsr_regex": r"^palm_[1-4]_fsr_geom$",
                    "display_forces": False,
                    "display_every": 0,
                    "display_top_k": 0,
                },
            ),
        })
        # 布局: 0:88 同 palm MCC, 88:92 = palm_fsr (4,)
    }

    actions: dict[str, ActionTermCfg] = {
        "arm_delta": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            use_default_offset=False,
        )
    }

    return ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
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
            nconmax=500,
        ),
        viewer=ViewerConfig(entity_name="robot", body_name="base_link", distance=2.0),
        episode_length_s=1e10 if play else 50.0,
    )


def combined_mcc_finger_env_cfg(
    num_envs: int = 1, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """组合 MCC 手掌 + 手指柔顺控制环境配置入口。"""
    return _make_env_cfg(num_envs=num_envs, play=play)


# ==============================================================================
#  MCCPalmMinkController — 用 mink IK (daqp) 替代 DLS 的手掌控制器
# ==============================================================================

class MCCPalmMinkController(MCCPalmComplianceController):
    """MCC 手掌导纳控制器 + mink IK（daqp 非线性优化）。

    与父类 MCCPalmComplianceController 的区别：
      - IK 层：mink.solve_ik() (daqp QP) 替代 DLS 伪逆
        → 单步 <0.1mm 精度，不再需要 Kp_task*dt 的慢速跟踪
      - 去除 IK 层的冗余力 PI（力控仅在导纳层完成，对齐 MCC 设计）
      - 去除虚拟 q_ref 积分（IK 从实际关节状态直接求解绝对位置）
      - 接触检测简化为纯力阈值判断
    """

    _PREP_TARGET = np.array(
        [0.0, 1.183, -1.541, 3.1415, 2.742, -1.569], dtype=np.float32,
    )

    def __init__(self, device: str, num_envs: int, **kwargs):
        # 父类初始化：observer、导纳参数、力估计配置等全部复用
        super().__init__(device, num_envs, **kwargs)

        # ── mink IK 设置（替代父类的 DLS）──
        self._mink_config = mink.Configuration(self._obs_model)
        try:
            self._mink_config.update_from_keyframe("home")
        except Exception:
            pass
        self._mink_palm_task = mink.FrameTask(
            frame_name="palm_lower",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        self._mink_posture_task = mink.PostureTask(self._obs_model, cost=0.1)
        self._mink_limits = [mink.ConfigurationLimit(self._obs_model)]
        self._mink_damping = float(kwargs.get("mink_damping", 0.1))
        self._mink_num_iter = int(kwargs.get("mink_num_iter", 3))

        # ── Joint6 旋转阻尼：限制 wrist yaw 速度，防止手掌水平 180° 翻转 ──
        self._max_ori_error = float(kwargs.get("max_ori_error", 0.3))  # ~17°/step
        self._joint6_damping = float(kwargs.get("joint6_damping", 0.7))

        # ── 接触稳定性参数 ──
        # 接触后法向保留一小部分刚度（K_force * contact_K_ratio），
        # 防止零刚度导纳纯积分漂移→过冲→振荡
        self._contact_K_ratio = float(kwargs.get("contact_K_ratio", 0.15))
        # 迟滞双阈值 + debounce：进入/退出接触用不同阈值，且需持续 N 步才切换
        _enter = float(kwargs.get("contact_enter_threshold", 0.0))
        _exit = float(kwargs.get("contact_exit_threshold", 0.0))
        self._contact_enter_thr = _enter if _enter > 0 else self.contact_threshold
        self._contact_exit_thr = _exit if _exit > 0 else self.contact_threshold * 0.5
        self._contact_debounce_steps = int(kwargs.get("contact_debounce_steps", 10))

        # 覆盖为简化状态（无 q_ref / force_error_integral / was_in_contact）
        self._init_states()

        print(
            f"[MCC-Palm-Mink] Init | f_desired_normal={self.f_desired_normal:.1f}N | "
            f"K_force={self.K_force:.0f} K_position={self.K_position:.0f} | "
            f"K_contact_ratio={self._contact_K_ratio:.2f} kd_n={self._kd_normal:.0f} | "
            f"enter>{self._contact_enter_thr:.1f}N exit<{self._contact_exit_thr:.1f}N "
            f"debounce={self._contact_debounce_steps} | "
            f"mink_damping={self._mink_damping:.2f} mink_iter={self._mink_num_iter}"
        )

    # ------------------------------------------------------------------
    #  简化状态（无虚拟 q_ref、无力 PI 状态）
    # ------------------------------------------------------------------

    def _init_states(self) -> None:
        self.states: list[dict] = []
        for _ in range(self.num_envs):
            self.states.append({
                "initialized": False,
                "prep_counter": 0,
                "q_init": None,
                "x_des": None,
                "x_ref": None,
                "v_ref": np.zeros(6, dtype=np.float32),
                "tau_smoothed": np.zeros(6, dtype=np.float32),
                "bias_smoothed": np.zeros(6, dtype=np.float32),
                "contact_normal": self._default_contact_normal.copy(),
                # 接触迟滞状态
                "in_contact": False,
                "contact_debounce": 0,
            })

    # ------------------------------------------------------------------
    #  主调用入口（覆盖父类 __call__，用 mink IK 替代 DLS+力PI）
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        f_cmd: torch.Tensor | None = None,
        x_des: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_obs = obs["policy"]
        B = policy_obs.shape[0]
        if B != self.num_envs:
            self.num_envs = B
            self._init_states()

        self._step_count += 1

        # 从 policy 观测中提取所需物理量 (CPU)
        joint_pos_all = policy_obs[:, 0:22].cpu().numpy().astype(np.float32)
        qfrc_actuator_arm = policy_obs[:, 28:34].cpu().numpy().astype(np.float32)
        # 关键：用仿真自身的 qfrc_bias（含惯性力补偿），而非 observer 的纯重力偏置
        # observer 用 qvel=0 只算重力偏置 → 高速运动时惯性力被误判为外力 → 正反馈失控
        qfrc_bias_arm = policy_obs[:, 34:40].cpu().numpy().astype(np.float32)

        # 外部 x_des 覆盖
        if x_des is not None:
            x_des_np = x_des.cpu().numpy().astype(np.float32)
            if x_des_np.ndim == 1:
                x_des_np = x_des_np.reshape(1, -1)
            if x_des_np.shape[1] == 3:
                x_des_np = np.pad(x_des_np, ((0, 0), (0, 3)), constant_values=0.0)
            B_des = x_des_np.shape[0]
            if B_des == 1 and B > 1:
                x_des_np = np.tile(x_des_np, (B, 1))
            _use_external_x_des = True
        else:
            x_des_np = np.zeros((B, 6), dtype=np.float32)
            _use_external_x_des = False

        if f_cmd is not None:
            f_cmd_np = f_cmd.cpu().numpy().astype(np.float32)
            if f_cmd_np.ndim == 1:
                f_cmd_np = np.tile(f_cmd_np.reshape(1, 3), (B, 1))
            _use_explicit_f_cmd = True
        else:
            f_cmd_np = np.zeros((B, 3), dtype=np.float32)
            _use_explicit_f_cmd = False

        output = np.zeros_like(joint_pos_all)

        for i in range(B):
            state = self.states[i]
            qpos_full = joint_pos_all[i]

            # ---- Step 1: 物理同步 ──
            # observer 仅用于运动学（Jacobian、palm 位姿），偏置力矩用仿真自身的值
            _bias_obs, jp, jr, palm_pos, palm_rotmat = self._sync_observer(qpos_full)
            # 用仿真 qfrc_bias（包含惯性+科氏力补偿），正确分离外力
            bias_sim = qfrc_bias_arm[i]

            # ---- Step 2: Preparation 阶段 ----
            if state["prep_counter"] < self.prep_steps:
                if state["prep_counter"] == 0:
                    state["q_init"] = qpos_full[:6].copy()
                state["prep_counter"] += 1
                t = state["prep_counter"] / self.prep_steps
                output[i, :6] = (
                    (1.0 - t) * state["q_init"] + t * self._PREP_TARGET
                )
                output[i, 6:] = qpos_full[6:]
                continue

            # ---- Step 3: Alignment 阶段（首次初始化状态）----
            if not state["initialized"]:
                state["tau_smoothed"] = qfrc_actuator_arm[i].copy()
                state["bias_smoothed"] = bias_sim.copy()
                palm_rotvec = (
                    R.from_matrix(palm_rotmat).as_rotvec().astype(np.float32)
                )
                state["x_des"] = np.concatenate([palm_pos, palm_rotvec])
                state["x_ref"] = state["x_des"].copy()
                state["v_ref"] = np.zeros(6, dtype=np.float32)
                state["initialized"] = True

            # ---- Step 3.5: 外部 x_des 覆盖 ----
            if _use_external_x_des:
                state["x_des"][:3] = x_des_np[i, :3]
                if not np.allclose(x_des_np[i, 3:6], 0.0):
                    state["x_des"][3:6] = x_des_np[i, 3:6]

            # ---- Step 4: 力矩 EMA 滤波 ----
            state["tau_smoothed"] = (
                self.alpha_tau * qfrc_actuator_arm[i]
                + (1.0 - self.alpha_tau) * state["tau_smoothed"]
            )
            state["bias_smoothed"] = (
                self.alpha_tau * bias_sim
                + (1.0 - self.alpha_tau) * state["bias_smoothed"]
            )
            tau_ext = -(state["tau_smoothed"] - state["bias_smoothed"])

            # ---- Step 5: MCC wrench estimation ----
            wrench = estimate_wrench(
                jp, jr, tau_ext, palm_rotmat, self._wrench_config,
            )
            f_ext = wrench[:3].astype(np.float32)
            tau_ext_wrench = wrench[3:6].astype(np.float32)

            # ---- Step 6: 各向异性 Admittance ----
            # 6a. 接触法向估计
            f_ext_norm = float(np.linalg.norm(f_ext))
            if f_ext_norm > self.contact_threshold:
                raw_normal = f_ext / f_ext_norm
            else:
                raw_normal = palm_rotmat[:, self._default_normal_idx].astype(np.float32)
            state["contact_normal"] = (
                self.alpha_normal * raw_normal
                + (1.0 - self.alpha_normal) * state["contact_normal"]
            )
            n = state["contact_normal"] / np.linalg.norm(state["contact_normal"])

            # 6b. 接触平滑过渡（迟滞 + smooth blend，防振荡）──
            # 用外力幅值计算一个 0→1 的接触因子 α，连续过渡：
            #  α=0: 纯接近（K=K_force, f_net=f_ext）
            #  α=1: 纯接触（K=K_force*ratio, f_net=f_ext+f_cmd）
            # 迟滞：上升沿用 enter 阈值，下降沿用 exit 阈值 + debounce
            if f_ext_norm > self._contact_enter_thr:
                want_contact = True
            elif f_ext_norm < self._contact_exit_thr:
                want_contact = False
            else:
                want_contact = state["in_contact"]

            if want_contact != state["in_contact"]:
                state["contact_debounce"] += 1
                if state["contact_debounce"] >= self._contact_debounce_steps:
                    state["in_contact"] = want_contact
                    state["contact_debounce"] = 0
            else:
                state["contact_debounce"] = 0

            # α 平滑过渡：在 enter↔exit 力区间内线性插值
            # 上升阶段（进入接触）：force 从 enter_thr 到 enter_thr*1.3 之间 α 从 0→1
            # 下降阶段（退出接触）：force 从 enter_thr*0.8 到 exit_thr 之间 α 从 1→0
            if f_ext_norm <= self._contact_exit_thr:
                contact_alpha = 0.0
            elif f_ext_norm >= self._contact_enter_thr * 1.3:
                contact_alpha = 1.0
            elif f_ext_norm >= self._contact_enter_thr:
                # 上升过渡带：enter_thr → enter_thr*1.3
                contact_alpha = (f_ext_norm - self._contact_enter_thr) / (
                    self._contact_enter_thr * 0.3
                )
            else:
                # 下降过渡带：exit_thr → enter_thr
                contact_alpha = (f_ext_norm - self._contact_exit_thr) / (
                    self._contact_enter_thr - self._contact_exit_thr
                )
            contact_alpha = float(np.clip(contact_alpha, 0.0, 1.0))
            in_contact = state["in_contact"]  # 用于日志

            # 6c. 刚度/阻尼矩阵（α 连续过渡）──
            nnT = np.outer(n, n)
            I3 = np.eye(3, dtype=np.float32)
            # 法向刚度：从 K_force（接近）平滑过渡到 K_force*ratio（接触）
            K_normal_approach = self.K_force
            K_normal_contact = self.K_force * self._contact_K_ratio
            K_normal = K_normal_approach + contact_alpha * (
                K_normal_contact - K_normal_approach
            )
            kp_pos_dyn = (
                self.K_position * I3 + (K_normal - self.K_position) * nnT
            ).astype(np.float32)
            kd_pos_dyn = get_damping_matrix(kp_pos_dyn, I3 * self.mass)

            # 6d. f_cmd
            if _use_explicit_f_cmd:
                f_cmd_i = f_cmd_np[i]
            elif self.f_desired_normal != 0.0:
                f_cmd_i = -self.f_desired_normal * n
            else:
                f_cmd_i = self.f_cmd_default

            # 6e. 平动 admittance 积分 ──
            # f_cmd 随 α 平滑激活：α=0 时只有弹簧驱动接近，α=1 时全力控
            pos_prev = state["x_ref"][:3]
            vel_prev = state["v_ref"][:3]
            pos_des = state["x_des"][:3]
            pos_error = pos_des - pos_prev

            kp_term = kp_pos_dyn @ pos_error
            kd_term = kd_pos_dyn @ vel_prev
            f_net = f_ext + contact_alpha * f_cmd_i
            lin_acc = (kp_term - kd_term + f_net) / self.mass
            vel_next = vel_prev + lin_acc * self.control_dt
            pos_next = pos_prev + vel_next * self.control_dt

            # 安全钳位：x_ref 不得偏离 x_des 超过 max_ref_offset（防力估计噪声→漂移→失控）
            _max_offset = 0.15  # 15cm
            _offset_vec = pos_next - pos_des
            _offset_dist = float(np.linalg.norm(_offset_vec))
            if _offset_dist > _max_offset:
                _offset_dir = _offset_vec / _offset_dist
                pos_next = pos_des + _offset_dir * _max_offset
                # 同时削掉背离 x_des 方向的速度分量
                _v_proj = float(np.dot(vel_next, _offset_dir))
                if _v_proj > 0:
                    vel_next = vel_next - _v_proj * _offset_dir

            # 6f. 转动 admittance 积分
            ori_prev = R.from_rotvec(state["x_ref"][3:6])
            omega_prev = state["v_ref"][3:6]
            ori_des = R.from_rotvec(state["x_des"][3:6])
            ori_error = (ori_des * ori_prev.inv()).as_rotvec().astype(np.float32)

            # ── 方向误差钳位：防止 x_des 突变时产生大范围旋转目标 ──
            _ori_err_norm = float(np.linalg.norm(ori_error))
            if _ori_err_norm > self._max_ori_error:
                ori_error = ori_error * (self._max_ori_error / _ori_err_norm)

            kp_term_rot = self.kp_rot @ ori_error
            kd_term_rot = self.kd_rot @ omega_prev
            ang_acc = (
                (kp_term_rot - kd_term_rot + tau_ext_used) / self.inertia_diag
            )
            omega_next = omega_prev + ang_acc * self.control_dt
            ori_next = (
                R.from_rotvec(omega_next * self.control_dt) * ori_prev
            ).as_rotvec().astype(np.float32)

            state["x_ref"][:3] = pos_next
            state["x_ref"][3:6] = ori_next
            state["v_ref"][:3] = vel_next
            state["v_ref"][3:6] = omega_next

            # ---- Step 7: mink IK（替代 DLS+力PI，对齐 MCC MinkIK）----
            # 从实际关节状态出发求解，输出绝对关节位置（无虚拟积分）
            self._mink_config.data.qpos[:] = qpos_full.copy()
            mujoco.mj_forward(self._mink_config.model, self._mink_config.data)

            self._mink_posture_task.set_target_from_configuration(self._mink_config)

            target_pos = state["x_ref"][:3]
            target_rotvec = state["x_ref"][3:6]
            target_rotmat = R.from_rotvec(target_rotvec).as_matrix()
            target_rot = mink.SO3.from_matrix(target_rotmat)
            target = mink.SE3.from_rotation_and_translation(target_rot, target_pos)
            self._mink_palm_task.set_target(target)

            for _ in range(self._mink_num_iter):
                vel = mink.solve_ik(
                    self._mink_config,
                    [self._mink_posture_task, self._mink_palm_task],
                    self.control_dt,
                    solver="daqp",
                    damping=self._mink_damping,
                    limits=self._mink_limits,
                )
                self._mink_config.integrate_inplace(vel, self.control_dt)

            arm_joint_pos = (
                self._mink_config.data.qpos[self._arm_dof_idx].copy().astype(np.float32)
            )
            # 关节限位 clip（安全冗余，mink ConfigurationLimit 已处理）
            arm_joint_pos = np.clip(
                arm_joint_pos,
                self._arm_jnt_ranges[:, 0],
                self._arm_jnt_ranges[:, 1],
            )

            # ── Joint6 旋转阻尼：限制 wrist yaw 速度，防止手掌水平 180° 翻转 ──
            # 只影响 joint6 (wrist yaw)，不影响 pitch/roll (由 joint2-5 顺应表面)
            if self._joint6_damping > 0 and state["initialized"]:
                _j6_idx = 5
                _j6_ik = arm_joint_pos[_j6_idx]
                _j6_actual = qpos_full[_j6_idx]
                _j6_diff = np.arctan2(
                    np.sin(_j6_ik - _j6_actual),
                    np.cos(_j6_ik - _j6_actual),
                )
                arm_joint_pos[_j6_idx] = _j6_actual + (1.0 - self._joint6_damping) * _j6_diff

            # ---- Step 8: 输出 ----
            output[i, :6] = arm_joint_pos
            output[i, 6:] = qpos_full[6:]

            # ---- 周期日志 ----
            if self._step_count % 300 == 0 and i == 0:
                f_norm = float(np.linalg.norm(f_ext))
                ik_err = float(np.linalg.norm(target_pos - palm_pos))
                f_ext_n_log = float(np.dot(f_ext, n))
                f_desired = (
                    float(np.linalg.norm(f_cmd_i))
                    if not _use_explicit_f_cmd
                    else self.f_desired_normal
                )
                print(
                    f"[MCC-Palm-Mink] Step={self._step_count} | "
                    f"|F_ext|={f_norm:.2f}N F_ext_n={f_ext_n_log:+.2f}N "
                    f"F_des={f_desired:.1f}N | "
                    f"IK_err={ik_err*1000:.2f}mm | "
                    f"α={contact_alpha:.2f} K_n={K_normal:.1f} "
                    f"{'CONTACT' if in_contact else 'APPROACH'} | "
                    f"dz_ref={state['x_ref'][2]-state['x_des'][2]:+.4f} "
                    f"dz_palm={palm_pos[2]-state['x_des'][2]:+.4f}"
                )

        return torch.as_tensor(output, device=self.device, dtype=torch.float32)


# ==============================================================================
#  MCCPalmStrictController — 严格对齐原版 MCC 行为的手掌控制器
# ==============================================================================

class MCCPalmStrictController:
    """手掌 MCC 柔顺控制器（严格模式，对齐原版 MCC 行为）。

    与 MCCPalmMinkController 的关键区别：
      - 力指令：f_net = f_ext + f_cmd 始终激活（无 α 调制）
      - 刚度：各向异性但接触后不切换，法向/切向比例固定
      - 法向：由配置 contact_normal_world 固定，不由 f_ext 更新
      - 偏置：observer qvel=0 静态重力偏置（同 MCC WrenchSim）
      - IK：site task（palm_control_site，位于 FSR 中心加局部可调偏移）
      - posture target：preparation 完成后用真实关节状态初始化
      - 测试模式：f_desired_normal=0 + f_ext=0 隔离跟踪链
    """

    _PREP_TARGET = np.array(
        [0.0, 1.183, -1.541, 3.1415, 2.742, -1.569], dtype=np.float32,
    )

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs

        # ── 控制频率 ──
        self.control_dt = float(kwargs.get("control_dt", 0.01))

        # ── 重力补偿积分增益（缓慢消除残余稳态误差）──
        self._ki_joint = float(kwargs.get("ki_joint", 0.3))
        self._grav_comp_gain = float(kwargs.get("grav_comp_gain", 1.0))

        # ── 测试模式：f_ext=0 + f_desired=0，隔离纯跟踪链 ──
        self._test_mode = bool(kwargs.get("test_mode", False))

        # ── 手指柔顺控制（可选）──
        self._enable_finger = bool(kwargs.get("enable_finger_control", False))
        self._palm_fsr_threshold = float(kwargs.get("palm_fsr_threshold", 0.3))
        self._finger_gain_scale = float(kwargs.get("finger_gain_scale", 0.3))
        # 手指只在机械臂 prep 结束且手掌控制 site 接近目标后启用。
        self._finger_activation_distance = float(
            kwargs.get("finger_activation_distance", 0.02)
        )
        self._finger_activation_blend_steps = max(
            1, int(kwargs.get("finger_activation_blend_steps", 50))
        )

        # ── 导纳参数（MCC 基线：mass=1.0, Kp_pos=100, Kp_rot=10）──
        self.mass = float(kwargs.get("mass_trans", 1.0))
        self.inertia_diag = np.array(
            kwargs.get("inertia_diag", (1.0, 1.0, 1.0)), dtype=np.float32,
        )
        self.K_force = float(kwargs.get("K_force", 100.0))
        self.K_position = float(kwargs.get("K_position", 100.0))
        self.K_rot = float(kwargs.get("K_rot", 10.0))

        # ── 转动刚度/阻尼 ──
        self.kp_rot = np.eye(3, dtype=np.float32) * self.K_rot
        self.kd_rot = get_damping_matrix(
            self.kp_rot, np.diag(self.inertia_diag),
        )

        # ── 固定接触法向（严格模式：由配置/任务输入，不由 f_ext 更新）──
        contact_normal = kwargs.get("contact_normal_world", None)
        if contact_normal is not None and not np.allclose(contact_normal, 0.0):
            self._contact_normal = np.array(contact_normal, dtype=np.float32).reshape(3)
            self._contact_normal = self._contact_normal / np.linalg.norm(self._contact_normal)
        else:
            normal_axis = str(kwargs.get("normal_axis", "z")).lower()
            axis_map = {"x": 0, "y": 1, "z": 2}
            default_idx = axis_map.get(normal_axis, 2)
            self._contact_normal = np.zeros(3, dtype=np.float32)
            self._contact_normal[default_idx] = 1.0

        # ── 固定各向异性刚度（不随接触切换）──
        K_normal = self.K_force
        self._build_stiffness(self._contact_normal, K_normal)

        # ── 力控参数 ──
        self.f_desired_normal = float(kwargs.get("f_desired_normal", 0.0))
        default_force = kwargs.get("f_cmd_default", (0.0, 0.0, 0.0))
        self.f_cmd_default = np.array(default_force, dtype=np.float32).reshape(3)

        # ── 力矩 EMA 滤波 ──
        self.alpha_tau = float(kwargs.get("alpha_tau", 0.1))

        # ── 力估计配置 ──
        from minimalist_compliance_control.minimalist_compliance_control.wrench_estimation import (
            WrenchEstimateConfig,
        )
        self._wrench_config = WrenchEstimateConfig(
            force_reg=float(kwargs.get("lambda_force", 1e-3)),
            torque_reg=float(kwargs.get("lambda_torque", 1e-2)),
        )

        # ── 准备阶段 ──
        self.prep_duration_s = float(kwargs.get("prep_duration_s", 1.5))
        self.prep_steps = max(1, int(self.prep_duration_s / self.control_dt))

        # ── 手掌控制点局部偏移 ──
        # palm_lower 局部 +X 指向主手指根部。正 X 偏移会把被跟踪的 site
        # 从 4 个 palm FSR 的几何中心向手指方向移动。
        self._palm_control_offset_local = np.asarray(
            kwargs.get("palm_control_offset_local", (0.0, 0.0, 0.0)),
            dtype=np.float32,
        ).reshape(3)

        # ── 构建 observer 模型（含 palm_control_site，位于 FSR 中心）──
        self._build_observer()

        # ── 构建 mink IK（site task，posture 延迟到 alignment 设置）──
        self._mink_config = mink.Configuration(self._obs_model)
        self._mink_orientation_cost = np.asarray(
            kwargs.get("mink_orientation_cost", (10.0, 10.0, 10.0)),
            dtype=np.float64,
        ).reshape(3)
        self._mink_site_task = mink.FrameTask(
            frame_name="palm_control_site",
            frame_type="site",
            position_cost=10.0,
            # Mink 的旋转误差在 site 局部坐标系表达：[X, Y, Z]。
            # 将 Z 权重设为 0 即只约束 palm-Z 法向，不约束绕法向 yaw。
            orientation_cost=self._mink_orientation_cost,
            lm_damping=1.0,
        )
        self._mink_posture_task = mink.PostureTask(self._obs_model, cost=0.1)
        self._posture_target_set = False  # 延迟到 alignment 设置
        self._mink_limits = [mink.ConfigurationLimit(self._obs_model)]
        self._mink_damping = float(kwargs.get("mink_damping", 0.1))
        self._mink_num_iter = int(kwargs.get("mink_num_iter", 1))

        # ── 方向追踪钳位：限制每步方向误差和 joint6 旋转，防止手掌 180° 翻转 ──
        # max_ori_error: 最大方向误差 (rad)，超过此值则钳位
        self._max_ori_error = float(kwargs.get("max_ori_error", 0.3))  # ~17°/step
        # joint6_damping: wrist yaw 阻尼 (0=跟随IK, 1=冻结)，只影响水平旋转不影响 pitch/roll
        self._joint6_damping = float(kwargs.get("joint6_damping", 0.7))

        # ── x_ref 安全钳位 ──
        self._max_ref_offset = float(kwargs.get("max_ref_offset", 0.0))

        # ── 手指柔顺控制器（可选）──
        self._finger_ctrl = None
        self._finger_delta_ema = None  # 手指输出平滑
        self._finger_smooth_alpha = float(kwargs.get("finger_smooth_alpha", 0.3))
        if self._enable_finger:
            from mjlab.tasks.leaphand.leaphand_finger_env_cfg import (
                LeapHandComplianceController,
            )
            self._finger_ctrl = LeapHandComplianceController(
                device=device, num_envs=num_envs, **kwargs,
            )

        # ── 状态 + 日志 ──
        self._init_states()
        self._step_count = 0
        self.last_debug: dict[str, torch.Tensor] = {}
        self._setup_logging()

        print(
            f"[MCC-Palm-Strict] Init | "
            f"mass={self.mass:.1f} K_pos={self.K_position:.0f} K_rot={self.K_rot:.0f} | "
            f"f_desired={self.f_desired_normal:.1f}N normal={self._contact_normal} | "
            f"test_mode={self._test_mode} finger={self._enable_finger} | "
            f"alpha_tau={self.alpha_tau:.2f} force_reg={self._wrench_config.force_reg:.0e} | "
            f"mink_damping={self._mink_damping:.2f} mink_iter={self._mink_num_iter} | "
            f"ori_cost={self._mink_orientation_cost.tolist()} | "
            f"max_ref_offset={self._max_ref_offset:.2f} | "
            f"max_ori_err={self._max_ori_error:.2f}rad j6_damp={self._joint6_damping:.1f}"
        )

    # ------------------------------------------------------------------
    #  构建 observer（palm_control_site 位于 FSR 中心）
    # ------------------------------------------------------------------

    def _build_observer(self) -> None:
        """构建 observer，并在 FSR 中心加局部偏移后放置控制 site。"""
        spec = mujoco.MjSpec.from_file(str(_LEAPHAND_XML))

        # 找到 palm_lower body
        palm_body = None
        for body in spec.bodies:
            if body.name == "palm_lower":
                palm_body = body
                break
        if palm_body is None:
            raise ValueError("palm_lower body not found in XML spec.")

        site = palm_body.add_site()
        site.name = "palm_control_site"

        self._obs_model = spec.compile()
        self._obs_data = mujoco.MjData(self._obs_model)

        # 提取 arm 关节 DOF 索引和限位
        dof_indices: list[int] = []
        jnt_ranges: list[np.ndarray] = []
        for joint_name in [f"joint{i}" for i in range(1, 7)]:
            jid = mujoco.mj_name2id(
                self._obs_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
            )
            if jid < 0:
                raise ValueError(f"Joint '{joint_name}' not found in observer model.")
            dof_indices.append(int(self._obs_model.jnt_dofadr[jid]))
            jnt_ranges.append(self._obs_model.jnt_range[jid].copy())
        self._arm_dof_idx = np.array(dof_indices, dtype=np.int32)
        self._arm_jnt_ranges = np.array(jnt_ranges, dtype=np.float32)

        # palm body 和 site ID
        self._palm_bid = -1
        for name in ("palm_lower", "palm", "robot/palm_lower"):
            bid = mujoco.mj_name2id(
                self._obs_model, mujoco.mjtObj.mjOBJ_BODY, name,
            )
            if bid >= 0:
                self._palm_bid = int(bid)
                break
        if self._palm_bid < 0:
            raise ValueError("Palm body not found in observer model.")

        sid = mujoco.mj_name2id(
            self._obs_model, mujoco.mjtObj.mjOBJ_SITE, "palm_control_site",
        )
        if sid < 0:
            raise ValueError("palm_control_site not found in observer model.")
        self._palm_site_id = int(sid)

        # ── 计算 FSR 中心在 palm_lower 本地坐标系中的位置 ──
        # 用默认 qpos 做一次 FK，取 4 个 palm FSR body 的平均位置
        self._obs_data.qpos[:] = 0.0
        mujoco.mj_forward(self._obs_model, self._obs_data)

        fsr_positions_world = []
        for fsr_name in ["palm_1_fsr", "palm_2_fsr", "palm_3_fsr", "palm_4_fsr"]:
            fsr_bid = mujoco.mj_name2id(
                self._obs_model, mujoco.mjtObj.mjOBJ_BODY, fsr_name,
            )
            if fsr_bid >= 0:
                fsr_positions_world.append(
                    self._obs_data.xpos[fsr_bid].copy()
                )

        if fsr_positions_world:
            fsr_center_world = np.mean(fsr_positions_world, axis=0)
            palm_pos = self._obs_data.xpos[self._palm_bid].copy()
            palm_rotmat = self._obs_data.xmat[self._palm_bid].reshape(3, 3).copy()
            # 转到 palm_lower 局部坐标
            fsr_center_local = palm_rotmat.T @ (fsr_center_world - palm_pos)
            control_site_local = (
                fsr_center_local + self._palm_control_offset_local
            )
            # 更新 MjSpec 中 site 的位置
            for s in spec.sites:
                if s.name == "palm_control_site":
                    s.pos = control_site_local
                    break
            # 重新编译模型以应用 site 位置
            self._obs_model = spec.compile()
            self._obs_data = mujoco.MjData(self._obs_model)
            # 重新获取 ID（编译后 ID 可能变化）
            sid = mujoco.mj_name2id(
                self._obs_model, mujoco.mjtObj.mjOBJ_SITE, "palm_control_site",
            )
            self._palm_site_id = int(sid)
            print(
                f"[MCC-Palm-Strict] palm_control_site: "
                f"fsr_center_local={fsr_center_local} "
                f"offset_local={self._palm_control_offset_local} "
                f"site_local={control_site_local}"
            )
        else:
            print("[MCC-Palm-Strict] WARNING: no FSR bodies found, site at body origin")

    # ------------------------------------------------------------------
    #  固定各向异性刚度
    # ------------------------------------------------------------------

    def _build_stiffness(self, n: np.ndarray, K_normal: float) -> None:
        nnT = np.outer(n, n)
        I3 = np.eye(3, dtype=np.float32)
        self._kp_pos = (
            self.K_position * I3 + (K_normal - self.K_position) * nnT
        ).astype(np.float32)
        self._kd_pos = get_damping_matrix(
            self._kp_pos, I3 * self.mass,
        )

    # ------------------------------------------------------------------
    #  状态
    # ------------------------------------------------------------------

    def _init_states(self) -> None:
        self.states: list[dict] = []
        for _ in range(self.num_envs):
            self.states.append({
                "initialized": False,
                "prep_counter": 0,
                "q_init": None,
                "x_des": None,
                "x_ref": None,
                "v_ref": np.zeros(6, dtype=np.float32),
                "tau_smoothed": np.zeros(6, dtype=np.float32),
                "q_error_integral": np.zeros(6, dtype=np.float32),
            })
        self._finger_gate_active = np.zeros(self.num_envs, dtype=bool)
        self._finger_gate_blend_counter = np.zeros(
            self.num_envs, dtype=np.int32,
        )

    # ------------------------------------------------------------------
    #  日志
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = _time.strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"mcc_strict_{ts}.csv")
        self._log_file = open(self._log_path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            "step", "time_s",
            "x_des_x", "x_des_y", "x_des_z",
            "x_ref_x", "x_ref_y", "x_ref_z",
            "site_x", "site_y", "site_z",
            "mink_fk_x", "mink_fk_y", "mink_fk_z",
            "q_ref_1", "q_ref_2", "q_ref_3", "q_ref_4", "q_ref_5", "q_ref_6",
            "q_act_1", "q_act_2", "q_act_3", "q_act_4", "q_act_5", "q_act_6",
            "qvel_1", "qvel_2", "qvel_3", "qvel_4", "qvel_5", "qvel_6",
            "f_ext_x", "f_ext_y", "f_ext_z", "f_ext_norm", "f_ext_n",
            "f_cmd_x", "f_cmd_y", "f_cmd_z",
            "tracking_err_mm", "FK_res_mm", "qacc_est",
            "dz_ref", "dz_site",
        ])
        self._log_file.flush()
        print(f"[MCC-Palm-Strict] Log: {self._log_path}")

    def _log_step(
        self, step: int, t: float,
        x_des: np.ndarray, x_ref: np.ndarray, site_pos: np.ndarray,
        mink_fk_pos: np.ndarray,
        q_ref: np.ndarray, q_act: np.ndarray, qvel: np.ndarray,
        f_ext: np.ndarray, f_ext_norm: float, f_ext_n: float,
        f_cmd: np.ndarray,
        tracking_err: float, fk_res: float, qacc: float,
        dz_ref: float, dz_site: float,
    ) -> None:
        self._log_writer.writerow([
            step, f"{t:.4f}",
            f"{x_des[0]:.6f}", f"{x_des[1]:.6f}", f"{x_des[2]:.6f}",
            f"{x_ref[0]:.6f}", f"{x_ref[1]:.6f}", f"{x_ref[2]:.6f}",
            f"{site_pos[0]:.6f}", f"{site_pos[1]:.6f}", f"{site_pos[2]:.6f}",
            f"{mink_fk_pos[0]:.6f}", f"{mink_fk_pos[1]:.6f}", f"{mink_fk_pos[2]:.6f}",
            f"{q_ref[0]:.4f}", f"{q_ref[1]:.4f}", f"{q_ref[2]:.4f}",
            f"{q_ref[3]:.4f}", f"{q_ref[4]:.4f}", f"{q_ref[5]:.4f}",
            f"{q_act[0]:.4f}", f"{q_act[1]:.4f}", f"{q_act[2]:.4f}",
            f"{q_act[3]:.4f}", f"{q_act[4]:.4f}", f"{q_act[5]:.4f}",
            f"{qvel[0]:.4f}", f"{qvel[1]:.4f}", f"{qvel[2]:.4f}",
            f"{qvel[3]:.4f}", f"{qvel[4]:.4f}", f"{qvel[5]:.4f}",
            f"{f_ext[0]:.4f}", f"{f_ext[1]:.4f}", f"{f_ext[2]:.4f}",
            f"{f_ext_norm:.4f}", f"{f_ext_n:.4f}",
            f"{f_cmd[0]:.4f}", f"{f_cmd[1]:.4f}", f"{f_cmd[2]:.4f}",
            f"{tracking_err:.4f}", f"{fk_res:.4f}", f"{qacc:.4f}",
            f"{dz_ref:.6f}", f"{dz_site:.6f}",
        ])
        if step % 100 == 0:
            self._log_file.flush() # type: ignore

    def close_log(self) -> None:
        if hasattr(self, "_log_file") and self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
            print(f"[MCC-Palm-Strict] Log closed: {self._log_path}")

    # ------------------------------------------------------------------
    #  Observer 同步
    # ------------------------------------------------------------------

    def _sync_observer(
        self, qpos_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """同步 observer：qvel=0 静态重力偏置 + 运动学。

        Returns: (bias(6,), jac_p(3,6), jac_r(3,6), site_pos(3,), site_rotmat(3,3))
        """
        self._obs_data.qpos[:] = qpos_np
        self._obs_data.qvel[:] = 0.0
        mujoco.mj_forward(self._obs_model, self._obs_data)

        # 静态重力偏置（qvel=0，同 MCC）
        bias = self._obs_data.qfrc_bias[self._arm_dof_idx].copy().astype(np.float32)

        # site 位姿 Jacobian
        jac_p = np.zeros((3, self._obs_model.nv), dtype=np.float64)
        jac_r = np.zeros((3, self._obs_model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self._obs_model, self._obs_data,
            jac_p, jac_r, self._palm_site_id,
        )
        jac_p_arm = jac_p[:, self._arm_dof_idx].copy().astype(np.float32)
        jac_r_arm = jac_r[:, self._arm_dof_idx].copy().astype(np.float32)

        # site 位姿（用于导纳位置反馈和 x_ref 初始化，与 IK 目标一致）
        site_pos = self._obs_data.site_xpos[self._palm_site_id].copy().astype(np.float32)
        site_rotmat = (
            self._obs_data.site_xmat[self._palm_site_id].reshape(3, 3).copy().astype(np.float32)
        )

        return bias, jac_p_arm, jac_r_arm, site_pos, site_rotmat

    # ------------------------------------------------------------------
    #  主控制循环
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        f_cmd: torch.Tensor | None = None,
        x_des: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 支持单组 (obs["policy"]) 和多组 (obs["palm"] + obs["finger"])
        if "palm" in obs:
            policy_obs = obs["palm"]
            finger_obs = obs.get("finger", None)
        else:
            policy_obs = obs["policy"]
            finger_obs = None

        B = policy_obs.shape[0]
        if B != self.num_envs:
            self.num_envs = B
            self._init_states()

        self._step_count += 1

        joint_pos_all = policy_obs[:, 0:22].cpu().numpy().astype(np.float32)
        qfrc_actuator_arm = policy_obs[:, 28:34].cpu().numpy().astype(np.float32)
        joint_vel_arm = policy_obs[:, 22:28].cpu().numpy().astype(np.float32)
        qfrc_bias_arm = policy_obs[:, 34:40].cpu().numpy().astype(np.float32)

        # 外部 x_des
        if x_des is not None:
            x_des_np = x_des.cpu().numpy().astype(np.float32)
            if x_des_np.ndim == 1:
                x_des_np = x_des_np.reshape(1, -1)
            if x_des_np.shape[1] == 3:
                x_des_np = np.pad(x_des_np, ((0, 0), (0, 3)), constant_values=0.0)
            B_des = x_des_np.shape[0]
            if B_des == 1 and B > 1:
                x_des_np = np.tile(x_des_np, (B, 1))
            _use_external_x_des = True
        else:
            x_des_np = np.zeros((B, 6), dtype=np.float32)
            _use_external_x_des = False

        if f_cmd is not None:
            f_cmd_np = f_cmd.cpu().numpy().astype(np.float32)
            if f_cmd_np.ndim == 1:
                f_cmd_np = np.tile(f_cmd_np.reshape(1, 3), (B, 1))
            _use_explicit_f_cmd = True
        else:
            f_cmd_np = np.zeros((B, 3), dtype=np.float32)
            _use_explicit_f_cmd = False

        output = np.zeros_like(joint_pos_all)
        debug_np = {
            "palm_x_des": np.zeros((B, 6), dtype=np.float32),
            "palm_x_ref": np.zeros((B, 6), dtype=np.float32),
            "palm_site_pos": np.zeros((B, 3), dtype=np.float32),
            "palm_site_rotvec": np.zeros((B, 3), dtype=np.float32),
            "palm_arm_q_ref": np.zeros((B, 6), dtype=np.float32),
            "palm_arm_q_actual": joint_pos_all[:, :6].copy(),
            "palm_f_ext": np.zeros((B, 3), dtype=np.float32),
            "palm_f_cmd": np.zeros((B, 3), dtype=np.float32),
            "palm_fk_residual": np.zeros((B, 1), dtype=np.float32),
            "palm_tracking_error": np.zeros((B, 1), dtype=np.float32),
            "palm_in_prep": np.zeros((B, 1), dtype=np.float32),
            "controller_finger_gate_active": np.zeros((B, 1), dtype=np.float32),
            "controller_finger_gate_blend": np.zeros((B, 1), dtype=np.float32),
            "controller_finger_arrival_error": np.zeros((B, 1), dtype=np.float32),
        }

        for i in range(B):
            state = self.states[i]
            qpos_full = joint_pos_all[i]

            # ---- Step 1: Observer 同步（静态重力偏置 + 运动学）----
            bias, jp, jr, site_pos, site_rotmat = (
                self._sync_observer(qpos_full)
            )
            debug_np["palm_site_pos"][i] = site_pos
            debug_np["palm_site_rotvec"][i] = (
                R.from_matrix(site_rotmat).as_rotvec().astype(np.float32)
            )

            # ---- Step 2: Preparation ----
            if state["prep_counter"] < self.prep_steps:
                if state["prep_counter"] == 0:
                    state["q_init"] = qpos_full[:6].copy()
                state["prep_counter"] += 1
                t = state["prep_counter"] / self.prep_steps
                output[i, :6] = (
                    (1.0 - t) * state["q_init"] + t * self._PREP_TARGET
                    + self._grav_comp_gain * qfrc_bias_arm[i] / 1000.0
                )
                output[i, 6:] = qpos_full[6:]
                debug_np["palm_in_prep"][i, 0] = 1.0
                debug_np["palm_x_des"][i, :3] = site_pos
                debug_np["palm_x_des"][i, 3:6] = debug_np["palm_site_rotvec"][i]
                debug_np["palm_x_ref"][i] = debug_np["palm_x_des"][i]
                debug_np["palm_arm_q_ref"][i] = output[i, :6]
                continue

            # ---- Step 3: Alignment（首次初始化 + posture target）----
            if not state["initialized"]:
                state["tau_smoothed"] = qfrc_actuator_arm[i].copy()
                site_rotvec = (
                    R.from_matrix(site_rotmat).as_rotvec().astype(np.float32)
                )
                state["x_des"] = np.concatenate([site_pos, site_rotvec])
                state["x_ref"] = state["x_des"].copy()
                state["v_ref"] = np.zeros(6, dtype=np.float32)
                state["initialized"] = True

                # 用真实关节状态设置 posture target（对齐 MCC）
                self._mink_config.data.qpos[:] = qpos_full.copy()
                mujoco.mj_forward(self._mink_config.model, self._mink_config.data)
                self._mink_posture_task.set_target_from_configuration(self._mink_config)
                self._posture_target_set = True

            # ---- Step 3.5: 外部 x_des 覆盖 ----
            if _use_external_x_des:
                state["x_des"][:3] = x_des_np[i, :3]
                if not np.allclose(x_des_np[i, 3:6], 0.0):
                    state["x_des"][3:6] = x_des_np[i, 3:6]

            # ---- Step 4: 力矩 EMA 滤波 + 外力估计（observer 静态偏置，同 MCC）----
            state["tau_smoothed"] = (
                self.alpha_tau * qfrc_actuator_arm[i]
                + (1.0 - self.alpha_tau) * state["tau_smoothed"]
            )
            tau_ext = -(state["tau_smoothed"] - bias)

            # ---- Step 5: 力估计（site 坐标系）----
            wrench = estimate_wrench(
                jp, jr, tau_ext, site_rotmat, self._wrench_config,
            )
            f_ext_wrench = wrench[:3].astype(np.float32)
            tau_ext_wrench = wrench[3:6].astype(np.float32)

            # ---- Step 6: 导纳积分 ----
            n = self._contact_normal

            if self._test_mode:
                f_ext_used = np.zeros(3, dtype=np.float32)
                f_cmd_i = np.zeros(3, dtype=np.float32)
                tau_ext_used = np.zeros(3, dtype=np.float32)
            else:
                f_ext_used = f_ext_wrench
                tau_ext_used = tau_ext_wrench
                if _use_explicit_f_cmd:
                    f_cmd_i = f_cmd_np[i]
                elif self.f_desired_normal != 0.0:
                    f_cmd_i = -self.f_desired_normal * n
                else:
                    f_cmd_i = self.f_cmd_default

            vel_prev = state["v_ref"][:3]
            pos_des = state["x_des"][:3]
            pos_error = pos_des - state["x_ref"][:3]

            kp_term = self._kp_pos @ pos_error
            kd_term = self._kd_pos @ vel_prev
            f_net = f_ext_used + f_cmd_i
            lin_acc = (kp_term - kd_term + f_net) / self.mass
            vel_next = vel_prev + lin_acc * self.control_dt
            pos_next = state["x_ref"][:3] + vel_next * self.control_dt
            if self._max_ref_offset > 0:
                _offset_vec = pos_next - pos_des
                _offset_dist = float(np.linalg.norm(_offset_vec))
                if _offset_dist > self._max_ref_offset:
                    _offset_dir = _offset_vec / _offset_dist
                    pos_next = pos_des + _offset_dir * self._max_ref_offset
                    _v_proj = float(np.dot(vel_next, _offset_dir))
                    if _v_proj > 0:
                        vel_next = vel_next - _v_proj * _offset_dir
                    if self._step_count % 300 == 0:
                        print(
                            f"[MCC-Palm-Strict] WARNING x_ref clamped: "
                            f"dist={_offset_dist:.3f}m > max={self._max_ref_offset:.2f}m"
                        )

            # 转动导纳
            ori_prev = R.from_rotvec(state["x_ref"][3:6])
            omega_prev = state["v_ref"][3:6]
            ori_des = R.from_rotvec(state["x_des"][3:6])
            ori_error = (ori_des * ori_prev.inv()).as_rotvec().astype(np.float32)

            # ── 方向误差钳位：防止 x_des 突变时产生大范围旋转目标 ──
            # 与位置 max_ref_offset 对应，限制每步方向变化幅度
            _ori_err_norm = float(np.linalg.norm(ori_error))
            if _ori_err_norm > self._max_ori_error:
                ori_error = ori_error * (self._max_ori_error / _ori_err_norm)

            kp_term_rot = self.kp_rot @ ori_error
            kd_term_rot = self.kd_rot @ omega_prev
            ang_acc = (
                (kp_term_rot - kd_term_rot + tau_ext_used) / self.inertia_diag
            )
            omega_next = omega_prev + ang_acc * self.control_dt
            ori_next = (
                R.from_rotvec(omega_next * self.control_dt) * ori_prev
            ).as_rotvec().astype(np.float32)

            state["x_ref"][:3] = pos_next
            state["x_ref"][3:6] = ori_next
            state["v_ref"][:3] = vel_next
            state["v_ref"][3:6] = omega_next
            debug_np["palm_x_des"][i] = state["x_des"]
            debug_np["palm_x_ref"][i] = state["x_ref"]
            debug_np["palm_f_ext"][i] = f_ext_used
            debug_np["palm_f_cmd"][i] = f_cmd_i

            # ---- Step 7: mink IK ----
            self._mink_config.data.qpos[:] = qpos_full.copy()
            mujoco.mj_forward(self._mink_config.model, self._mink_config.data)

            target_pos = state["x_ref"][:3]
            target_rotvec = state["x_ref"][3:6]
            target_rotmat = R.from_rotvec(target_rotvec).as_matrix()
            target_rot = mink.SO3.from_matrix(target_rotmat)
            target = mink.SE3.from_rotation_and_translation(target_rot, target_pos)
            self._mink_site_task.set_target(target)

            for _ in range(self._mink_num_iter):
                vel = mink.solve_ik(
                    self._mink_config,
                    [self._mink_posture_task, self._mink_site_task],
                    self.control_dt,
                    solver="daqp",
                    damping=self._mink_damping,
                    limits=self._mink_limits,
                )
                self._mink_config.integrate_inplace(vel, self.control_dt)

            arm_joint_pos = (
                self._mink_config.data.qpos[self._arm_dof_idx].copy().astype(np.float32)
            )
            arm_joint_pos = np.clip(
                arm_joint_pos,
                self._arm_jnt_ranges[:, 0],
                self._arm_jnt_ranges[:, 1],
            )

            # ── Joint6 旋转阻尼：限制 wrist yaw 速度，防止手掌水平 180° 翻转 ──
            # joint6 只控制 wrist yaw（手掌水平朝向），不影响 pitch/roll（由 joint2-5 负责顺应表面）
            if self._joint6_damping > 0 and state["initialized"]:
                _j6_idx = 5
                _j6_ik = arm_joint_pos[_j6_idx]
                _j6_actual = qpos_full[_j6_idx]
                # 最短角度差（正确处理 ±π 环绕）
                _j6_diff = np.arctan2(
                    np.sin(_j6_ik - _j6_actual),
                    np.cos(_j6_ik - _j6_actual),
                )
                arm_joint_pos[_j6_idx] = _j6_actual + (1.0 - self._joint6_damping) * _j6_diff

            # Mink FK 残差：求解后 site 的实际位姿 vs 目标
            _mink_sid = mujoco.mj_name2id(
                self._mink_config.model, mujoco.mjtObj.mjOBJ_SITE, "palm_control_site",
            )
            mink_site_pos = self._mink_config.data.site_xpos[_mink_sid].copy()
            mink_fk_residual = float(np.linalg.norm(mink_site_pos - target_pos))
            debug_np["palm_fk_residual"][i, 0] = mink_fk_residual

            # ---- Step 8: 输出（重力前馈 + 积分修正）----
            # 关节跟踪误差积分（缓慢收敛，消除残余稳态误差）
            _q_err = arm_joint_pos - qpos_full[:6]
            state["q_error_integral"] += _q_err * self.control_dt * self._ki_joint
            _int_clamp = 0.1  # 防积分饱和
            state["q_error_integral"] = np.clip(
                state["q_error_integral"], -_int_clamp, _int_clamp,
            )
            output[i, :6] = (
                arm_joint_pos
                + self._grav_comp_gain * qfrc_bias_arm[i] / 1000.0
                + state["q_error_integral"]
            )
            output[i, 6:] = qpos_full[6:]
            debug_np["palm_arm_q_ref"][i] = arm_joint_pos
            debug_np["palm_tracking_error"][i, 0] = float(
                np.linalg.norm(state["x_ref"][:3] - site_pos)
            )

            # |qvel| 用作运动检测
            _qvel_norm = float(np.linalg.norm(joint_vel_arm[i]))

            # ---- 逐帧日志（env 0，CSV 文件）----
            if i == 0:
                self._log_step(
                    step=self._step_count,
                    t=self._step_count * self.control_dt,
                    x_des=state["x_des"],
                    x_ref=state["x_ref"],
                    site_pos=site_pos,
                    mink_fk_pos=mink_site_pos,
                    q_ref=arm_joint_pos,
                    q_act=qpos_full[:6],
                    qvel=joint_vel_arm[i],
                    f_ext=f_ext_used,
                    f_ext_norm=float(np.linalg.norm(f_ext_used)),
                    f_ext_n=float(np.dot(f_ext_used, n)),
                    f_cmd=f_cmd_i,
                    tracking_err=float(np.linalg.norm(state["x_ref"][:3] - site_pos)),
                    fk_res=mink_fk_residual,
                    qacc=_qvel_norm,
                    dz_ref=float(pos_next[2] - state["x_des"][2]),
                    dz_site=float(site_pos[2] - state["x_des"][2]),
                )

            # ---- 周期日志（控制台）----
            if self._step_count % 300 == 0 and i == 0:
                f_raw_norm = float(np.linalg.norm(f_ext_used))
                f_raw_n = float(np.dot(f_ext_used, n))
                tracking_err = float(np.linalg.norm(state["x_ref"][:3] - site_pos))
                print(
                    f"[MCC-Palm-Strict] Step={self._step_count} | "
                    f"|F_ext|={f_raw_norm:.2f}N F_ext_n={f_raw_n:+.2f}N "
                    f"({'OFF' if self._test_mode else 'ON'}) | "
                    f"tracking_err={tracking_err*1000:.1f}mm "
                    f"FK_res={mink_fk_residual*1000:.2f}mm | "
                    f"|qvel|={_qvel_norm:.2f} | "
                    f"x_des=({state['x_des'][0]:.3f},{state['x_des'][1]:.3f},{state['x_des'][2]:.3f}) "
                    f"site=({site_pos[0]:.3f},{site_pos[1]:.3f},{site_pos[2]:.3f})"
                )

        output_t = torch.as_tensor(output, device=self.device, dtype=torch.float32)

        # ── 手指延迟启用门控 ──
        # prep 阶段绝不启动手指；prep 结束后，实际控制 site 接近 x_des
        # 才锁存启用。锁存后不因物体运动造成的瞬时误差而关闭。
        in_prep = debug_np["palm_in_prep"][:, 0] > 0.5
        arrival_error = np.linalg.norm(
            debug_np["palm_x_des"][:, :3] - debug_np["palm_site_pos"],
            axis=1,
        )
        finger_ready = (
            (~in_prep)
            & (arrival_error <= self._finger_activation_distance)
        )
        self._finger_gate_active |= finger_ready
        self._finger_gate_blend_counter = np.where(
            self._finger_gate_active,
            np.minimum(
                self._finger_gate_blend_counter + 1,
                self._finger_activation_blend_steps,
            ),
            0,
        ).astype(np.int32)
        blend_raw = (
            self._finger_gate_blend_counter.astype(np.float32)
            / float(self._finger_activation_blend_steps)
        )
        finger_gate_blend = blend_raw * blend_raw * (3.0 - 2.0 * blend_raw)
        debug_np["controller_finger_gate_active"][:, 0] = (
            self._finger_gate_active.astype(np.float32)
        )
        debug_np["controller_finger_gate_blend"][:, 0] = finger_gate_blend
        debug_np["controller_finger_arrival_error"][:, 0] = arrival_error

        # ── 手指柔顺控制（门控渐入 + 增益缩放 + EMA）──
        if self._enable_finger and self._finger_ctrl is not None and finger_obs is not None:
            # hand_delta 是相对动作；未到位时零动作保持当前手型。
            output_t[:, 6:] = 0.0
            if bool(np.any(self._finger_gate_active)):
                finger_delta = self._finger_ctrl({"policy": finger_obs})  # [B, 16]
                finger_delta = finger_delta * self._finger_gain_scale

                if self._finger_delta_ema is None:
                    self._finger_delta_ema = finger_delta.clone()
                else:
                    alpha = self._finger_smooth_alpha
                    self._finger_delta_ema = (
                        alpha * self._finger_delta_ema
                        + (1.0 - alpha) * finger_delta
                    )
                gate_blend_t = torch.as_tensor(
                    finger_gate_blend,
                    device=self.device,
                    dtype=output_t.dtype,
                ).unsqueeze(-1)
                output_t[:, 6:] = self._finger_delta_ema * gate_blend_t

        self.last_debug = {
            key: torch.as_tensor(value, device=self.device, dtype=torch.float32)
            for key, value in debug_np.items()
        }

        return output_t

    def reset(self) -> None:
        """Reset palm states and delayed finger activation."""
        self._init_states()
        self._step_count = 0
        self._finger_delta_ema = None
        if self._finger_ctrl is not None:
            self._finger_ctrl.is_init = False
            self._finger_ctrl.q_nom.zero_()
            self._finger_ctrl.fsr_obs.zero_()
            self._finger_ctrl.fsr_ctrl.zero_()
            self._finger_ctrl.contact_state.zero_()
            self._finger_ctrl.prev_action.zero_()


# ==============================================================================
#  CombinedMCCFingerController
# ==============================================================================

class CombinedMCCFingerController:
    """组合控制器：MCC 手掌导纳控制 + 手指 FSR 柔顺控制。

    阶段机：
      Phase 0 (APPROACH):      手掌通过 MCC 接近表面，手指保持预抓取姿态不动
      Phase 1 (PALM_STABLE):   手掌已稳定接触，手指通过 smoothstep 平滑激活
      Phase 2 (FULL_COMPLIANCE): 两者完全激活

    阶段分离解决手指接触力→手掌力估计的耦合问题：
    手指仅在手掌已建立稳定接触后才激活，降低力估计被污染的风险。
    """

    # ── 阶段常量 ──
    PHASE_APPROACH = 0
    PHASE_PALM_STABLE = 1
    PHASE_FULL_COMPLIANCE = 2

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs

        # ── 阶段机参数 ──
        # 注：使用 phase_contact_force_threshold 区分于手掌控制器的 contact_threshold
        # （前者判断 arm 关节力矩幅值，后者判断 external force 幅值用于法向估计）
        self.contact_threshold = float(
            kwargs.get("phase_contact_force_threshold",
                       kwargs.get("contact_threshold", 4.0))
        )
        self.settle_vel_threshold = float(kwargs.get("settle_vel_threshold", 0.01))
        self.sustain_steps = int(kwargs.get("sustain_steps", 50))
        self.blend_duration_steps = int(kwargs.get("blend_duration_steps", 100))
        self.lost_contact_timeout = int(kwargs.get("lost_contact_timeout", 50))
        self.palm_lost_steps = int(kwargs.get("palm_lost_steps", 10))

        # ── FSR 门控参数 ──
        # 手掌 FSR 超过此阈值才确认手掌已贴合（直接接触信号，不被手指力污染）
        self.palm_fsr_threshold = float(kwargs.get("palm_fsr_threshold", 0.3))
        # 手指 FSR 超过此阈值判定手指已接触到物体
        self.finger_fsr_threshold = float(kwargs.get("finger_fsr_threshold", 0.2))
        # 手指预接触时张开手指的力度（归一化动作值，负=张开）
        self.finger_open_magnitude = float(kwargs.get("finger_open_magnitude", 0.4))
        # 手指柔顺整体增益缩放（<1.0 减弱抓握力度，实现被动贴合而非强行抓取）
        self.finger_gain_scale = float(kwargs.get("finger_gain_scale", 0.5))

        # 手指屈曲关节索引（在 hand 16-D 动作空间中）
        # 每指: [prox, abd, mid, dist] — 正值=握拢, 负值=张开
        self._finger_flexion_indices = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15]

        # ── 每环境阶段状态（GPU 张量，支持向量化）──
        self.env_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_sustain_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_blend_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_lost_contact_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._palm_lost_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._prev_palm_z = torch.zeros(num_envs, device=device)   # 诊断：上帧手掌Z位置

        # ── 构建 MCC 手掌子控制器 ──
        strict_mcc = bool(kwargs.get("strict_mcc", False))
        if strict_mcc:
            self.palm_controller = MCCPalmStrictController(
                device=device, num_envs=num_envs, **kwargs,
            )
        else:
            self.palm_controller = MCCPalmMinkController(
                device=device, num_envs=num_envs, **kwargs,
            )

        # ── 构建手指柔顺子控制器 ──
        finger_class_name = kwargs.get("finger_class", "LeapHandComplianceController")
        if finger_class_name == "FingertipForceComplianceController":
            env_ref = kwargs.get("env", None)
            self.finger_controller = FingertipForceComplianceController(
                device=device, num_envs=num_envs, env=env_ref, **kwargs,
            )
        else:
            self.finger_controller = LeapHandComplianceController(
                device=device, num_envs=num_envs, **kwargs,
            )

        # ── 手指力矩前馈补偿：预计算 FSR → arm 关节力矩映射矩阵 ──
        self._fsr_torque_matrix = self._setup_finger_torque_compensation()

        self._step_count = 0
        self.last_debug: dict[str, torch.Tensor] = {}
        print(
            f"[Combined-MCC-Finger] Init | "
            f"strict_mcc={strict_mcc} | "
            f"contact_thresh={self.contact_threshold:.1f}N "
            f"sustain={self.sustain_steps} steps "
            f"blend={self.blend_duration_steps} steps | "
            f"palm_fsr>{self.palm_fsr_threshold:.1f}N "
            f"finger_fsr>{self.finger_fsr_threshold:.1f}N "
            f"open={self.finger_open_magnitude:.1f} "
            f"gain={self.finger_gain_scale:.1f} | "
            f"finger_ctrl={type(self.finger_controller).__name__}"
        )

    # ------------------------------------------------------------------
    #  手指力矩前馈补偿
    # ------------------------------------------------------------------

    def _setup_finger_torque_compensation(self) -> torch.Tensor:
        """预计算 FSR → arm 关节力矩的线性映射矩阵 W (6×16)。

        手指接触力通过运动链传到 arm 关节，被 MCC 误判为手掌外力。
        这里用指尖雅可比预计算 W，使得每步可以从 qfrc_actuator_arm
        中扣掉 τ_finger = W @ fsr 来消除手指贡献。

        使用远端 FSR（每指最后一个）作为主要力信号，
        力方向取指尖 body 的 local Z 轴。
        """
        obs_model, obs_data, arm_dof_idx, _, _ = _get_or_build_observer()

        # 指尖 body 名称 → 对应的远端 FSR 索引（16 通道中）
        # FSR 布局: 手掌[0:4], index[4:7], middle[7:10], ring[10:13], thumb[13:16]
        # 每指: [proximal, proximal, distal]
        fingertips: list[tuple[str, int]] = [
            ("fingertip",       6),   # index 远端 FSR
            ("fingertip_2",     9),   # middle 远端 FSR
            ("fingertip_3",     12),  # ring 远端 FSR
            ("thumb_fingertip", 15),  # thumb 远端 FSR
        ]

        # 使用 observer 模型默认 qpos（XML keyframe 姿态）
        mujoco.mj_forward(obs_model, obs_data)

        W = np.zeros((6, 16), dtype=np.float32)  # 6 arm joints × 16 FSR channels

        for body_name, fsr_idx in fingertips:
            bid = mujoco.mj_name2id(
                obs_model, mujoco.mjtObj.mjOBJ_BODY, body_name,
            )
            if bid < 0:
                print(f"  [WARN] fingertip body '{body_name}' not found")
                continue

            # 指尖位置雅可比 w.r.t. arm 关节 (3 × 6)
            jac_p = np.zeros((3, obs_model.nv), dtype=np.float64)
            jac_r = np.zeros((3, obs_model.nv), dtype=np.float64)
            pos = obs_data.xpos[bid].copy()
            mujoco.mj_jac(obs_model, obs_data, jac_p, jac_r, pos, bid)
            J_arm = jac_p[:, arm_dof_idx].astype(np.float32)  # 3×6

            # 力方向：指尖 body 的 local Z 轴（指向抓取方向）
            rotmat = obs_data.xmat[bid].reshape(3, 3).astype(np.float32)
            force_dir = rotmat[:, 2]  # local Z in world frame

            # τ_arm = J^T · f_dir × |f|，FSR 读数即 |f| 的近似
            contribution = J_arm.T @ force_dir  # (6,)
            W[:, fsr_idx] = contribution

        print(
            f"[Combined-MCC-Finger] Finger torque compensation matrix: "
            f"W shape=({W.shape[0]}, {W.shape[1]}), "
            f"active FSR cols=[6,9,12,15]"
        )
        return torch.tensor(W, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    #  主调用入口
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        f_cmd: torch.Tensor | None = None,
        x_des: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """每控制周期调用，返回 22-D 动作张量 [arm_abs(6), hand_delta(16)]。

        Args:
            obs: 观测字典，包含 "palm" (88-D) 和 "finger" (54-D) 两个键
            f_cmd: 可选外部力指令，透传给手掌控制器
            x_des: 可选外部位置目标，透传给手掌控制器
        """
        B = obs["palm"].shape[0]
        if B != self.num_envs:
            self.num_envs = B
            self.env_phase = torch.zeros(B, dtype=torch.long, device=self.device)
            self.env_sustain_counter = torch.zeros(B, dtype=torch.long, device=self.device)
            self.env_blend_counter = torch.zeros(B, dtype=torch.long, device=self.device)
            self.env_lost_contact_counter = torch.zeros(B, dtype=torch.long, device=self.device)
            self._palm_lost_counter = torch.zeros(B, dtype=torch.long, device=self.device)

        self._step_count += 1

        # ── 1. 手指力矩前馈补偿 ──
        # τ_finger = W @ fsr：估算手指接触力在 arm 关节上的力矩贡献
        # 从 qfrc_actuator_arm 中扣除，避免手指力被 MCC 误判为手掌外力
        fsr_all = obs["finger"][:, 0:16]  # [B, 16]
        tau_finger = fsr_all @ self._fsr_torque_matrix.T  # [B, 6]

        # 构建修正后的手掌观测（在 qfrc_actuator_arm 位置减去手指贡献）
        palm_obs_tensor = obs["palm"].clone()
        palm_obs_tensor[:, 28:34] = palm_obs_tensor[:, 28:34] - tau_finger
        palm_obs = {"policy": palm_obs_tensor}

        # ── 2. 运行手掌 MCC 控制器（始终活跃）──
        palm_output_22d = self.palm_controller(palm_obs, f_cmd=f_cmd, x_des=x_des)
        arm_action_abs = palm_output_22d[:, :6]  # [B, 6]

        # ── 3. 运行手指控制器（每步调用，保持内部 EMA 状态温热）──
        finger_obs = {"policy": obs["finger"]}
        finger_action_norm = self.finger_controller(finger_obs)  # [B, 16]

        # ── 4. 阶段状态机（FSR 增强版）──
        # 4a. 读取 arm 状态（关节力矩 + 速度）
        qfrc_arm = obs["palm"][:, 28:34]
        vel_arm = obs["palm"][:, 22:28]
        arm_force_mag = torch.linalg.vector_norm(qfrc_arm, dim=-1)
        arm_vel_mag = torch.linalg.vector_norm(vel_arm, dim=-1)

        # 4b. 分离手掌/手指 FSR（fsr_all 已在第 1 步读取）
        palm_fsr = fsr_all[:, 0:4]          # [B, 4] 手掌 FSR
        finger_fsr = fsr_all[:, 4:16]       # [B, 12] 手指 FSR
        palm_fsr_max = palm_fsr.max(dim=-1).values    # [B]
        finger_fsr_max = finger_fsr.max(dim=-1).values  # [B]

        # 4c. 手掌接触检测：仅用 FSR 直接信号
        # arm 关节力矩（qfrc_actuator）包含重力补偿，自由空间中就很大，
        # 不能可靠区分"接触"和"悬空"——只用 FSR 作为接触真值
        palm_contact = palm_fsr_max > self.palm_fsr_threshold

        # 4d. 手指预接触检测：手指碰到物体但手掌还没贴上
        # 这是需要主动抑制的情况 — 手指接触力会污染 MCC 力估计
        finger_pre_contact = (
            (finger_fsr_max > self.finger_fsr_threshold)
            & (~palm_contact)
            & (self.env_phase == self.PHASE_APPROACH)
        )

        # 4e. 更新 sustain 计数器 (APPROACH → PALM_STABLE)
        entering_contact = palm_contact & (self.env_phase == self.PHASE_APPROACH)
        self.env_sustain_counter = torch.where(
            entering_contact,
            self.env_sustain_counter + 1,
            torch.where(
                self.env_phase == self.PHASE_APPROACH,
                torch.zeros_like(self.env_sustain_counter),
                self.env_sustain_counter,
            ),
        )

        # Phase 0 → 1: 手掌接触持续确认
        transition_to_stable = (
            (self.env_phase == self.PHASE_APPROACH)
            & (self.env_sustain_counter >= self.sustain_steps)
        )
        self.env_phase = torch.where(
            transition_to_stable,
            torch.full_like(self.env_phase, self.PHASE_PALM_STABLE),
            self.env_phase,
        )
        self.env_blend_counter = torch.where(
            transition_to_stable,
            torch.zeros_like(self.env_blend_counter),
            self.env_blend_counter,
        )

        # Phase 1 中更新 blend 计数器
        in_stable = self.env_phase == self.PHASE_PALM_STABLE
        self.env_blend_counter = torch.where(
            in_stable & (self.env_blend_counter < self.blend_duration_steps),
            self.env_blend_counter + 1,
            self.env_blend_counter,
        )

        # Phase 1 → 2: blend 完成
        transition_to_full = in_stable & (
            self.env_blend_counter >= self.blend_duration_steps
        )
        self.env_phase = torch.where(
            transition_to_full,
            torch.full_like(self.env_phase, self.PHASE_FULL_COMPLIANCE),
            self.env_phase,
        )

        # Phase 2 中检测接触丢失（仍用 palm_contact 综合判断）
        in_full = self.env_phase == self.PHASE_FULL_COMPLIANCE
        lost_contact = in_full & (~palm_contact)
        self.env_lost_contact_counter = torch.where(
            lost_contact,
            self.env_lost_contact_counter + 1,
            torch.where(
                in_full,
                torch.zeros_like(self.env_lost_contact_counter),
                self.env_lost_contact_counter,
            ),
        )

        # Phase 2 → 1: 持续失去接触
        revert_to_stable = in_full & (
            self.env_lost_contact_counter >= self.lost_contact_timeout
        )
        self.env_phase = torch.where(
            revert_to_stable,
            torch.full_like(self.env_phase, self.PHASE_PALM_STABLE),
            self.env_phase,
        )
        self.env_blend_counter = torch.where(
            revert_to_stable,
            torch.full_like(self.env_blend_counter, self.blend_duration_steps),
            self.env_blend_counter,
        )

        # ── 5. Smoothstep blend + 手指预接触抑制 + 手掌接触优先级 ──
        blend_raw = (
            self.env_blend_counter.float() / max(self.blend_duration_steps, 1)
        )
        # smoothstep: f(t) = 3t² - 2t³, 两端导数为零
        blend = blend_raw * blend_raw * (3.0 - 2.0 * blend_raw)
        blend = torch.clamp(blend, 0.0, 1.0)

        # Phase 0: blend = 0（完全抑制手指柔顺）
        blend = torch.where(
            self.env_phase == self.PHASE_APPROACH,
            torch.zeros_like(blend),
            blend,
        )
        # Phase 2: blend = 1（完全激活手指柔顺）
        blend = torch.where(
            self.env_phase == self.PHASE_FULL_COMPLIANCE,
            torch.ones_like(blend),
            blend,
        )

        # 手掌脱离持续检测：FSR 低于阈值时计数，持续 N 步后才抑制手指
        # 避免 FSR 瞬时波动导致手指频繁开关
        palm_fsr_low = (palm_fsr_max < self.palm_fsr_threshold) & (
            self.env_phase == self.PHASE_FULL_COMPLIANCE
        )
        self._palm_lost_counter = torch.where(
            palm_fsr_low,
            self._palm_lost_counter + 1,
            torch.where(
                self.env_phase == self.PHASE_FULL_COMPLIANCE,
                torch.zeros_like(self._palm_lost_counter),
                self._palm_lost_counter,
            ),
        )
        palm_sustained_lost = self._palm_lost_counter >= self.palm_lost_steps

        # 手指预接触抑制：手指碰到物体但手掌未贴合时，主动张开手指
        # 避免手指接触力通过运动链污染 MCC 手掌力估计
        finger_open_action = torch.zeros_like(finger_action_norm)
        finger_open_action[:, self._finger_flexion_indices] = -self.finger_open_magnitude

        # APPROACH 阶段：预接触 → 张开手指；正常 → 零增量
        approach_open = finger_pre_contact.unsqueeze(-1)  # [B, 1]
        approach_action = torch.where(
            approach_open,
            finger_open_action,
            torch.zeros_like(finger_action_norm),
        )

        # PALM_STABLE/FULL 阶段：用 blend 混合手指柔顺输出
        compliance_action = finger_action_norm * blend.unsqueeze(-1)

        # 最终手指动作：APPROACH 用预接触抑制，其他阶段用柔顺混合
        in_approach = (self.env_phase == self.PHASE_APPROACH).unsqueeze(-1)
        hand_action = torch.where(in_approach, approach_action, compliance_action)

        # 手掌持续脱离时（FSR 低于阈值持续 sustain_steps 步）→ 强制张开手指
        palm_lost = palm_sustained_lost.unsqueeze(-1)
        hand_action = torch.where(palm_lost, finger_open_action, hand_action)

        # 手指整体增益缩放：减弱抓握力度，实现被动贴合
        hand_action = hand_action * self.finger_gain_scale

        # ── 6. 拼接并返回 ──
        action = torch.cat([arm_action_abs, hand_action], dim=-1)  # [B, 22]
        palm_debug = getattr(self.palm_controller, "last_debug", {})
        self.last_debug = {
            **palm_debug,
            "controller_phase": self.env_phase.to(torch.float32).unsqueeze(-1),
            "controller_blend": blend.unsqueeze(-1),
            "controller_palm_contact": palm_contact.to(torch.float32).unsqueeze(-1),
            "controller_finger_pre_contact": finger_pre_contact.to(torch.float32).unsqueeze(-1),
            "controller_palm_lost": palm_sustained_lost.to(torch.float32).unsqueeze(-1),
            "controller_tau_finger": tau_finger,
            "controller_finger_action_raw": finger_action_norm,
            "controller_finger_action_blended": hand_action,
            "controller_arm_action_abs": arm_action_abs,
        }

        # ── 周期日志 ──
        if self._step_count % 300 == 0 and B > 0:
            phase_names = {0: "APPROACH", 1: "PALM_STABLE", 2: "FULL_COMPLIANCE"}
            phase_counts = {
                phase_names[p]: (self.env_phase == p).sum().item()
                for p in range(3)
            }
            f_norm = arm_force_mag[0].item()
            v_norm = arm_vel_mag[0].item()
            b = blend[0].item()
            pfsr = palm_fsr_max[0].item()
            ffsr = finger_fsr_max[0].item()
            tcomp = tau_finger[0].norm().item()
            lost_cnt = self._palm_lost_counter[0].item()
            pre = "PRE-CTACT" if finger_pre_contact[0] else (f"LOST{lost_cnt}" if palm_lost[0] else "ok")
            # x_des vs palm 位置跟踪误差
            palm_pos_actual = obs["palm"][0, 76:79]
            if x_des is not None:
                xd = x_des[0, :3]
                track_err = (xd - palm_pos_actual).norm().item()
            else:
                track_err = 0.0
            # 诊断：手掌Z位移（正=靠近物体，负=被推开）
            palm_z_now = obs["palm"][0, 78].item()  # palm_pos Z
            palm_dz = palm_z_now - self._prev_palm_z[0].item()
            self._prev_palm_z[0] = palm_z_now
            print(
                f"[Combined] Step={self._step_count} | "
                f"Phases: A={phase_counts['APPROACH']} "
                f"S={phase_counts['PALM_STABLE']} "
                f"F={phase_counts['FULL_COMPLIANCE']} | "
                f"|F_arm|={f_norm:.2f}N |vel|={v_norm:.4f} | "
                f"FSR_p={pfsr:.2f} FSR_f={ffsr:.2f} "
                f"τcmp={tcomp:.3f} lc={lost_cnt} "
                f"err={track_err:.3f} dz={palm_dz:+.4f} | "
                f"blend={b:.3f} {pre}"
            )

        return action

    # ------------------------------------------------------------------
    #  状态管理
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """重置阶段状态 + 手掌控制器状态（当环境 episode 重置时调用）。"""
        if env_ids is None:
            self.env_phase[:] = self.PHASE_APPROACH
            self.env_sustain_counter[:] = 0
            self.env_blend_counter[:] = 0
            self.env_lost_contact_counter[:] = 0
            self._palm_lost_counter[:] = 0
            self.last_debug = {}
            # 重置手掌控制器状态（x_ref/v_ref/EMA 等）
            self.palm_controller._init_states()
            self.palm_controller._step_count = 0
        else:
            self.env_phase[env_ids] = self.PHASE_APPROACH
            self.env_sustain_counter[env_ids] = 0
            self.env_blend_counter[env_ids] = 0
            self.env_lost_contact_counter[env_ids] = 0
            self._palm_lost_counter[env_ids] = 0

    def set_env(self, env: ManagerBasedRlEnv) -> None:
        """注入环境引用（FingertipForceComplianceController 需要）。"""
        if isinstance(self.finger_controller, FingertipForceComplianceController):
            self.finger_controller._env = env


# ==============================================================================
#  RL Config
# ==============================================================================


@dataclass
class CombinedMCCFingerControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    policy_class: type = MCCPalmStrictController
    amplitude: float = 0.5

    # ── 阶段机参数 ──
    # phase_contact_force_threshold: |qfrc_actuator_arm| 阈值，判断手掌是否接触
    phase_contact_force_threshold: float = 4.0
    settle_vel_threshold: float = 0.01
    sustain_steps: int = 10
    blend_duration_steps: int = 100
    lost_contact_timeout: int = 50

    # ── FSR 门控参数 ──
    # 手掌 FSR 直接接触信号阈值（不受手指力污染）
    palm_fsr_threshold: float = 0.3
    # 手指 FSR 预接触检测阈值
    finger_fsr_threshold: float = 0.2
    # 手指预接触时张开力度（归一化值，负=张开屈曲关节）
    finger_open_magnitude: float = 0.4
    # 手指柔顺整体增益缩放（<1.0 减弱抓握力度，被动贴合；=1.0 原始力度）
    finger_gain_scale: float = 0.5

    # ── 手指控制器选择 ──
    finger_class: str = "LeapHandComplianceController"

    # ── 手掌柔顺模式 ──
    strict_mcc: bool = False           # True=严格MCC模式, False=当前模式(带接触状态机)
    contact_normal_world: tuple = (0.0, 0.0, 0.0)  # 严格模式下固定接触法向(0=用normal_axis)
    max_ref_offset: float = 0.0        # x_ref钳位距离(0=禁用,严格模式建议0.15作为安全网)
    # ── 方向追踪限制：防止手掌 180° 翻转 ──
    max_ori_error: float = 0.3         # 最大方向误差 (rad/step)
    joint6_damping: float = 0.7        # wrist yaw 阻尼 (0=跟随IK, 1=冻结)

    # ═══════════════════════════════════════════════════════════
    #  MCC 手掌控制器参数（透传给手掌子控制器）
    # ═══════════════════════════════════════════════════════════
    control_dt: float = 0.01
    mass_trans: float = 3.0            # 导纳质量 (strict模式建议1.0)
    inertia_diag: tuple = (0.3, 0.3, 0.3)  # (strict模式建议(1,1,1))
    K_force: float = 20.0              # 法向刚度 (strict模式建议100)
    K_position: float = 100.0          # 切向刚度
    K_rot: float = 10.0                # 姿态跟踪刚度
    normal_axis: str = "z"             # 手掌局部法向轴（strict模式用contact_normal_world）
    f_desired_normal: float = 1
    f_cmd_default: tuple = (0.0, 0.0, 0.0)
    kd_normal: float = 80.0
    contact_threshold: float = 4.0
    # ── 接触稳定性（仅非严格模式生效）──
    contact_K_ratio: float = 0.15      # 接触后法向保留 K_force 的比例（0=纯力控/易振荡，1=位置控制）
    contact_enter_threshold: float = 0.0  # 进入接触阈值（0=跟随 contact_threshold）
    contact_exit_threshold: float = 0.0   # 退出接触阈值（0=contact_threshold*0.5）
    contact_debounce_steps: int = 10   # 状态切换需持续的步数
    alpha_normal: float = 0.1       # 接触法向慢平滑，不跳动
    alpha_tau: float = 0.3              # 力矩EMA (strict模式建议0.1)
    lambda_force: float = 1e-2          # 力估计正则化 (strict模式建议1e-3)
    lambda_torque: float = 1e-2         # 力矩估计正则化
    prep_duration_s: float = 1.5
    # ── mink IK 参数 ──
    mink_damping: float = 0.1           # IK 求解器阻尼
    mink_num_iter: int = 3              # 每步 IK 迭代次数 (strict模式建议1)
    # ── 已废弃（新 mink IK 忽略，保留向后兼容）──
    Kp_task: float = 0.8
    dls_lambda: float = 0.3
    k_posture: float = 0.0
    Kf_vel: float = 0.03
    Kif_vel: float = 0.002
    force_int_max_n: float = 5.0

    # ═══════════════════════════════════════════════════════════
    #  手指控制器参数（透传给 LeapHandComplianceController）
    # ═══════════════════════════════════════════════════════════
    torque2fsr_model_path: str | None = None


# ==============================================================================
#  纯手掌严格 MCC 模式 Config（无手指，直接用 MCCPalmStrictController）
# ==============================================================================

@dataclass
class MCCPalmStrictControlCfg(RslRlOnPolicyRunnerCfg):
    """纯手掌严格 MCC 控制配置（不包含手指）。"""
    seed: int = 42
    device: str = "cuda:0"
    policy_class: type = MCCPalmStrictController
    amplitude: float = 0.5

    # ── 导纳参数（MCC 基线）──
    control_dt: float = 0.01
    mass_trans: float = 1.0
    inertia_diag: tuple = (1.0, 1.0, 1.0)
    K_force: float = 100.0
    K_position: float = 100.0
    K_rot: float = 10.0
    normal_axis: str = "z"
    contact_normal_world: tuple = (0.0, 0.0, -1.0)  # 底面外法向 = 世界 -Z（手在物体下方，+Z 压入）
    f_desired_normal: float = 5.0
    f_cmd_default: tuple = (0.0, 0.0, 0.0)

    # ── 力估计（MCC 基线）──
    alpha_tau: float = 0.1
    lambda_force: float = 1e-3
    lambda_torque: float = 1e-2

    # ── mink IK（MCC 基线）──
    mink_damping: float = 0.1
    mink_num_iter: int = 3     # 至少 3 次迭代才能收敛到亚毫米精度
    # site 局部旋转误差权重 [X, Y, Z]；Z=0 表示 wrist yaw 自由。
    mink_orientation_cost: tuple = (10.0, 10.0, 10.0)

    # ── 测试模式：f_desired_normal=0 + f_ext=0，隔离跟踪链 ──
    test_mode: bool = False
    # ── 重力补偿调参 ──
    ki_joint: float = 0.3
    grav_comp_gain: float = 1.0
    # ── 方向追踪限制：防止手掌 180° 翻转 ──
    max_ori_error: float = 0.3     # 最大方向误差 (rad/step)，~17°/步
    joint6_damping: float = 0.7    # wrist yaw 阻尼 (0=跟随IK, 1=冻结)，不影响 pitch/roll
    # palm_lower 局部坐标偏移；+X 指向主手指根部。
    palm_control_offset_local: tuple = (0.0, 0.0, 0.0)
    # ── 手指柔顺控制 ──
    enable_finger_control: bool = False
    palm_fsr_threshold: float = 0.3
    finger_gain_scale: float = 0.3
    finger_smooth_alpha: float = 0.3
    # prep 结束后，控制 site 距离 x_des 小于该值才启用手指柔顺。
    finger_activation_distance: float = 0.02
    # 启用后用 smoothstep 渐入，避免手指突然闭合。
    finger_activation_blend_steps: int = 50
    max_ref_offset: float = 0.15
    prep_duration_s: float = 1.5
