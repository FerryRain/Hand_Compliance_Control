"""Fair Exp. 2 nominal-reference sources.

All three sources feed the same deterministic role interpreter and coordinated
MCC stack.  They differ only in how a nominal finger chunk and role intention
are proposed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
import torch

from Module.module_4_finger_dp.contracts import ACTION_HORIZON_STEPS
from Module.module_4_finger_dp.dpref_policy import DPREF_INPUT_NAMES
from Module.module_4_finger_dp.dpref_train import load_dpref_policy
from Module.module_4_finger_dp.force_history import CausalForcePreprocessor
from Module.module_4_finger_dp.gpu_runtime import synchronize_cuda
from Module.module_4_finger_dp.track_d_dataset import (
  finger_state_geometry_features,
  pose_twist_series,
  relative_plan_twists,
)
from Module.module_4_whole_hand_mcc.reference_interpreter import ContactRole


@dataclass(frozen=True, slots=True)
class ReferenceSourceContext:
  step: int
  timestamp_s: float
  dt_s: float
  policy_period_steps: int
  desired_force_n: float
  finger_q_rad: NDArray[np.float64]
  finger_dq_rad_s: NDArray[np.float64]
  fingertip_force_n: NDArray[np.float64]
  actual_contact_mask: NDArray[np.bool_]
  fingertip_positions_world_m: NDArray[np.float64]
  contact_positions_world_m: NDArray[np.float64]
  contact_normals_world: NDArray[np.float64]
  palm_pose_world: NDArray[np.float64]
  future_plan_poses_world: NDArray[np.float64]
  object_position_world_m: NDArray[np.float64]
  wrist_mcc_offset_world: NDArray[np.float64]
  wrist_mcc_velocity_world: NDArray[np.float64]
  fingertip_jacobian_world: NDArray[np.float64]
  previous_nominal_command_rad: NDArray[np.float64]
  previous_mcc_correction_rad: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceSourceOutput:
  nominal_command_chunk_rad: NDArray[np.float64]
  requested_roles: NDArray[np.int64]
  role_probabilities: NDArray[np.float64]
  inference_latency_s: float
  inference_executed: bool
  nominal_free_authority_mask: NDArray[np.bool_]
  nominal_tangent_authority_mask: NDArray[np.bool_]


def _hold_output(
  *,
  use_nominal: bool = False,
  anchor_q_rad: NDArray[np.float64] | None = None,
) -> ReferenceSourceOutput:
  probability = np.zeros((4, 4), dtype=np.float64)
  probability[:, int(ContactRole.KEEP)] = 1.0
  return ReferenceSourceOutput(
    nominal_command_chunk_rad=np.repeat(
      np.asarray(
        np.zeros(16) if anchor_q_rad is None else anchor_q_rad,
        dtype=np.float64,
      )[None, :],
      ACTION_HORIZON_STEPS,
      axis=0,
    ),
    requested_roles=np.full(4, int(ContactRole.KEEP), dtype=np.int64),
    role_probabilities=probability,
    inference_latency_s=0.0,
    inference_executed=False,
    nominal_free_authority_mask=np.full(4, use_nominal, dtype=np.bool_),
    nominal_tangent_authority_mask=np.full(4, use_nominal, dtype=np.bool_),
  )


def _not_replanned(output: ReferenceSourceOutput) -> ReferenceSourceOutput:
  return ReferenceSourceOutput(
    nominal_command_chunk_rad=output.nominal_command_chunk_rad,
    requested_roles=output.requested_roles,
    role_probabilities=output.role_probabilities,
    inference_latency_s=0.0,
    inference_executed=False,
    nominal_free_authority_mask=output.nominal_free_authority_mask,
    nominal_tangent_authority_mask=output.nominal_tangent_authority_mask,
  )


class PassiveHoldReferenceSource:
  name = "PASSIVE_HOLD"

  def reset(self) -> None:
    return None

  def step(self, context: ReferenceSourceContext) -> ReferenceSourceOutput:
    return _hold_output(use_nominal=False, anchor_q_rad=context.finger_q_rad)


class ReactiveHeuristicReferenceSource:
  """Causal velocity compensation with no future wrist-plan access."""

  name = "REACTIVE_HEURISTIC"

  def __init__(self) -> None:
    self._last_palm_position: NDArray[np.float64] | None = None
    self._last_output = _hold_output(use_nominal=False)

  def reset(self) -> None:
    self._last_palm_position = None
    self._last_output = _hold_output(use_nominal=False)

  def step(self, context: ReferenceSourceContext) -> ReferenceSourceOutput:
    if self._last_palm_position is None:
      palm_velocity = np.zeros(3, dtype=np.float64)
    else:
      palm_velocity = (
        context.palm_pose_world[:3] - self._last_palm_position
      ) / context.dt_s
    self._last_palm_position = np.array(context.palm_pose_world[:3], copy=True)
    if context.step % context.policy_period_steps:
      return _not_replanned(self._last_output)
    first_offset = np.zeros(16, dtype=np.float64)
    horizon_s = context.dt_s * context.policy_period_steps
    for finger in range(4):
      if not context.actual_contact_mask[finger]:
        continue
      normal = context.contact_normals_world[finger]
      tangent = np.eye(3) - np.outer(normal, normal)
      desired_tip_delta = -tangent @ palm_velocity * horizon_s
      columns = slice(4 * finger, 4 * finger + 4)
      jacobian = context.fingertip_jacobian_world[finger, :, columns]
      damping = 1e-5 * np.eye(3)
      delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping,
        desired_tip_delta,
      )
      first_offset[columns] = np.clip(delta, -0.00003, 0.00003)
    chunk = np.stack(
      [first_offset * (index + 1) for index in range(ACTION_HORIZON_STEPS)]
    )
    output = _hold_output(use_nominal=False, anchor_q_rad=context.finger_q_rad)
    self._last_output = ReferenceSourceOutput(
      nominal_command_chunk_rad=context.finger_q_rad[None, :] + chunk,
      requested_roles=output.requested_roles,
      role_probabilities=output.role_probabilities,
      inference_latency_s=0.0,
      inference_executed=True,
      nominal_free_authority_mask=np.zeros(4, dtype=np.bool_),
      nominal_tangent_authority_mask=np.ones(4, dtype=np.bool_),
    )
    return self._last_output


class DPRefReferenceSource:
  name = "DPREF_TWO_HEAD"

  def __init__(
    self,
    checkpoint_path: str,
    *,
    device: str = "cuda:0",
    make_probability_threshold: float = 0.30,
  ) -> None:
    self.model = load_dpref_policy(
      checkpoint_path,
      device=device,
      require_training_pass=False,
    )
    self.device = next(self.model.parameters()).device
    self.make_probability_threshold = make_probability_threshold
    self.force = CausalForcePreprocessor()
    self.reset()

  def reset(self) -> None:
    self.force.reset()
    self.real_twist: deque[NDArray[np.float64]] = deque(maxlen=20)
    self.wrist_offset: deque[NDArray[np.float64]] = deque(maxlen=20)
    self.wrist_velocity: deque[NDArray[np.float64]] = deque(maxlen=20)
    self._last_history_pose: NDArray[np.float64] | None = None
    self._ever_contact = np.zeros(4, dtype=np.bool_)
    self._last_output = _hold_output(use_nominal=True)

  def _append_wrist_history(self, context: ReferenceSourceContext) -> None:
    if self._last_history_pose is None:
      twist = np.zeros(6, dtype=np.float64)
    else:
      twist = pose_twist_series(
        np.stack((self._last_history_pose, context.palm_pose_world)),
        context.dt_s * 5,
      )[-1]
    self._last_history_pose = np.array(context.palm_pose_world, copy=True)
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, context.palm_pose_world[3:])
    rotation = rotation.reshape(3, 3)
    offset = np.array(context.wrist_mcc_offset_world, copy=True)
    velocity = np.array(context.wrist_mcc_velocity_world, copy=True)
    offset[:3] = offset[:3] @ rotation
    offset[3:] = offset[3:] @ rotation
    velocity[:3] = velocity[:3] @ rotation
    velocity[3:] = velocity[3:] @ rotation
    self.real_twist.append(twist)
    self.wrist_offset.append(offset)
    self.wrist_velocity.append(velocity)

  def step(self, context: ReferenceSourceContext) -> ReferenceSourceOutput:
    self._ever_contact |= context.actual_contact_mask
    self.force.push(
      context.fingertip_force_n,
      context.actual_contact_mask,
      np.ones(4, dtype=np.bool_),
    )
    if (context.step + 1) % 5 == 0:
      self._append_wrist_history(context)
    if context.step % context.policy_period_steps:
      return _not_replanned(self._last_output)
    if not self.force.ready or len(self.real_twist) < 20:
      self._last_output = _hold_output(
        use_nominal=True,
        anchor_q_rad=context.finger_q_rad,
      )
      return self._last_output
    state = finger_state_geometry_features(
      finger_q_rad=context.finger_q_rad,
      finger_dq_rad_s=context.finger_dq_rad_s,
      fingertip_positions_world_m=context.fingertip_positions_world_m,
      measured_contact_positions_world_m=context.contact_positions_world_m,
      actual_contact_mask=context.actual_contact_mask,
      palm_pose_world=context.palm_pose_world,
      object_position_world_m=context.object_position_world_m,
      desired_force_n=context.desired_force_n,
    )
    horizons = (
      context.dt_s
      * context.policy_period_steps
      * np.arange(1, ACTION_HORIZON_STEPS + 1, dtype=np.float64)
    )
    values: dict[str, NDArray[np.float32]] = {
      "force_history": self.force.window().encoder_input(),
      "finger_state_geometry": state,
      "wrist_real_twist_history": np.asarray(self.real_twist, dtype=np.float32),
      "wrist_mcc_offset_history": np.asarray(self.wrist_offset, dtype=np.float32),
      "wrist_mcc_velocity_history": np.asarray(self.wrist_velocity, dtype=np.float32),
      "future_wrist_plan_twist": relative_plan_twists(
        context.palm_pose_world,
        context.future_plan_poses_world,
        horizons,
      ).astype(np.float32),
      "q_meas_rad": context.finger_q_rad.astype(np.float32),
      "previous_nominal_command_rad": context.previous_nominal_command_rad.astype(np.float32),
      "previous_mcc_correction_rad": context.previous_mcc_correction_rad.astype(np.float32),
    }
    if set(values) != set(DPREF_INPUT_NAMES):
      raise AssertionError("online DPRef condition contract drifted")
    inputs = {
      name: torch.from_numpy(np.array(value, copy=True)).unsqueeze(0).to(
        device=self.device,
        non_blocking=True,
      )
      for name, value in values.items()
    }
    latent = torch.zeros(
      1,
      ACTION_HORIZON_STEPS,
      16,
      device=self.device,
    )
    synchronize_cuda(self.device)
    begin = perf_counter()
    offsets, roles, probabilities = self.model.sample(
      inputs,
      initial_noise=latent,
      deterministic=True,
    )
    synchronize_cuda(self.device)
    latency = perf_counter() - begin
    role = roles[0].cpu().numpy().astype(np.int64)
    probability = probabilities[0].cpu().numpy().astype(np.float64)
    # A low-frequency categorical head may propose MAKE slightly before it
    # becomes the argmax.  The deterministic geometry/contact interpreter is
    # still the sole authority that can approve the transition.
    free = ~context.actual_contact_mask
    role[free & (probability[:, int(ContactRole.MAKE)] >= self.make_probability_threshold)] = int(
      ContactRole.MAKE
    )
    # INITIALIZE owns first-contact acquisition.  Before a finger has ever
    # established a physical contact, a learned FREE prediction cannot cancel
    # the deterministic MAKE already in progress.
    role[~self._ever_contact] = int(ContactRole.KEEP)
    self._last_output = ReferenceSourceOutput(
      nominal_command_chunk_rad=(
        context.finger_q_rad[None, :]
        + offsets[0].cpu().numpy().astype(np.float64)
      ),
      requested_roles=role,
      role_probabilities=probability,
      inference_latency_s=latency,
      inference_executed=True,
      nominal_free_authority_mask=self._ever_contact.copy(),
      nominal_tangent_authority_mask=np.ones(4, dtype=np.bool_),
    )
    return self._last_output


__all__ = [
  "DPRefReferenceSource",
  "PassiveHoldReferenceSource",
  "ReactiveHeuristicReferenceSource",
  "ReferenceSourceContext",
  "ReferenceSourceOutput",
]
