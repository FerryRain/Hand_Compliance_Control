from __future__ import annotations

import unittest

import numpy as np
import torch

from Module.module_3_runtime_guards import (
  CommandContinuityConfig,
  CommandContinuityLimiter,
)

from Module.module_4_finger_dp import (
  AuthorityFilterConfig,
  CausalForcePreprocessor,
  ContactHysteresisConfig,
  DPActionAuthorityFilter,
  DPGuardConfig,
  DPGuardState,
  DPRuntimeGuardExecutor,
  DiffusionPolicyConfig,
  FingerDPObservation,
  FingerDiffusionPolicy,
  ForceHistoryConfig,
  MeasuredAnchoredActionChunk,
  MeasuredContactHysteresis,
  build_teacher_command_chunks,
  contact_normal_wrist_map,
  observation_to_tensors,
  opposition_metrics,
)
from Module.module_4_finger_dp.dpref_policy import (
  DPREF_INPUT_NAMES,
  DPRefPolicyConfig,
  FingerDPRefPolicy,
)
from Module.module_4_whole_hand_mcc import (
  ContactRole,
  ContactRoleInterpreter,
  RoleInterpreterConfig,
)


def valid_observation() -> FingerDPObservation:
  normals = np.tile([0.0, 0.0, 1.0], (4, 1))
  return FingerDPObservation(
    timestamp_s=1.0,
    surface_model_version="oracle-v1",
    finger_q_rad=np.zeros((4, 4)),
    finger_dq_rad_s=np.zeros((4, 4)),
    force_history_normalized=np.full((4, 20), 0.5),
    contact_history=np.ones((4, 20), dtype=bool),
    force_valid_history=np.ones((4, 20), dtype=bool),
    contact_position_palm_m=np.zeros((4, 3)),
    contact_normal_palm=normals,
    surface_distance_m=np.zeros(4),
    surface_uncertainty_m=np.zeros(4),
    geometry_from_contact=np.ones(4, dtype=bool),
    surface_geometry_valid=np.ones(4, dtype=bool),
    desired_force_n=np.full(4, 2.0),
    wrist_real_twist_history=np.zeros((20, 6)),
    wrist_mcc_offset_history=np.zeros((20, 6)),
    wrist_mcc_velocity_history=np.zeros((20, 6)),
    future_wrist_plan_twist=np.zeros((20, 6)),
    previous_executed_finger_command_rad=np.zeros(16),
  )


class ObservationAndForceHistoryTest(unittest.TestCase):
  def test_observation_distinguishes_force_and_geometry_validity(self) -> None:
    observation = valid_observation()
    self.assertEqual(observation.force_encoder_input().shape, (4, 20, 3))
    self.assertEqual(observation.per_finger_state_geometry().shape, (4, 20))
    self.assertTrue(np.all(observation.actual_contact_mask))
    with self.assertRaises(ValueError):
      FingerDPObservation(
        **{
          name: getattr(observation, name)
          for name in observation.__dataclass_fields__
          if name not in {"geometry_from_contact", "surface_geometry_valid"}
          and not name.startswith("schema_")
        },
        geometry_from_contact=np.ones(4, dtype=bool),
        surface_geometry_valid=np.zeros(4, dtype=bool),
      )

  def test_force_history_is_filtered_decimated_and_explicitly_valid(self) -> None:
    processor = CausalForcePreprocessor(
      ForceHistoryConfig(lowpass_cutoff_hz=20.0, history_steps=20)
    )
    emitted = 0
    for step in range(100):
      raw = np.full(4, 10.0 if step % 2 else 0.0)
      valid = np.ones(4, dtype=bool)
      if step >= 95:
        valid[2] = False
      emitted += int(processor.push(raw, raw > 0.2, valid))
    self.assertEqual(emitted, 20)
    self.assertTrue(processor.ready)
    window = processor.window()
    self.assertEqual(window.encoder_input().shape, (4, 20, 3))
    self.assertFalse(bool(window.force_valid_history[2, -1]))
    self.assertGreater(window.filtered_force_n[0, -1], 1.0)
    self.assertLess(window.filtered_force_n[0, -1], 9.0)
    processor.reset()
    self.assertFalse(processor.ready)

  def test_contact_set_requires_time_confirmed_make_and_break(self) -> None:
    contact = MeasuredContactHysteresis(
      ContactHysteresisConfig(
        enter_force_n=0.2,
        exit_force_n=0.1,
        confirm_steps=3,
        release_steps=2,
      )
    )
    self.assertFalse(contact.update([0.3, 0.0, 0.0, 0.0]).actual_contact_mask[0])
    self.assertFalse(contact.update([0.3, 0.0, 0.0, 0.0]).actual_contact_mask[0])
    made = contact.update([0.3, 0.0, 0.0, 0.0])
    self.assertTrue(made.actual_contact_mask[0])
    self.assertTrue(made.make_mask[0])
    self.assertTrue(contact.update([0.0, 0.0, 0.0, 0.0]).actual_contact_mask[0])
    broken = contact.update([0.0, 0.0, 0.0, 0.0])
    self.assertFalse(broken.actual_contact_mask[0])
    self.assertTrue(broken.break_mask[0])


class RelativeChunkTest(unittest.TestCase):
  def test_teacher_chunk_is_command_imitation_with_one_measured_anchor(self) -> None:
    measured = np.arange(8 * 16, dtype=np.float64).reshape(8, 16) * 0.001
    teacher = measured + 0.1
    usable = np.ones(8, dtype=bool)
    usable[6] = False
    chunks = build_teacher_command_chunks(
      measured,
      teacher,
      horizon=3,
      usable_mask=usable,
    )
    self.assertEqual([chunk.start_index for chunk in chunks], [0, 1, 2])
    expected = teacher[1:4] - measured[0]
    self.assertTrue(np.allclose(chunks[0].target_offsets_rad, expected))
    action = chunks[0].as_action_chunk()
    self.assertTrue(np.allclose(action.nominal_commands_rad, teacher[1:4]))

  def test_chunk_reanchors_and_blends_at_the_boundary(self) -> None:
    chunk = MeasuredAnchoredActionChunk(np.ones(16), np.full((2, 16), 0.2))
    self.assertTrue(np.allclose(chunk.nominal_command(0), 1.2))
    blended = chunk.seam_blended_command(0, np.full(16, 0.8), blend_steps=4)
    self.assertTrue(np.allclose(blended, 0.9))


class AuthorityFilterTest(unittest.TestCase):
  def setUp(self) -> None:
    self.config = AuthorityFilterConfig(
      joint_lower_rad=-np.full(16, 2.0),
      joint_upper_rad=np.full(16, 2.0),
      max_abs_delta_rad=0.2,
      max_velocity_rad_s=20.0,
      max_acceleration_rad_s2=1000.0,
      max_seam_rad=0.2,
      collective_limit_m=2e-4,
    )
    self.filter = DPActionAuthorityFilter(self.config)

  def _step(self, nominal: np.ndarray):
    jacobian = np.zeros((2, 16))
    jacobian[0, 0] = 0.01
    jacobian[1, 4] = 0.01
    selection = np.zeros((6, 1))
    selection[2, 0] = 1.0
    return self.filter.step(
      current_q_rad=np.zeros(16),
      nominal_delta_rad=nominal,
      previous_executed_command_rad=np.zeros(16),
      previous_executed_velocity_rad_s=np.zeros(16),
      finger_normal_jacobian_m_per_rad=jacobian,
      active_contact_positions_palm_m=np.array([[-0.03, 0, 0], [0.03, 0, 0]]),
      active_outward_normals_palm=np.tile([0.0, 0.0, 1.0], (2, 1)),
      wrist_compliance_selection=selection,
      dt_s=0.02,
    )

  def test_wrist_map_includes_rotation(self) -> None:
    mapping = contact_normal_wrist_map([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    self.assertTrue(np.allclose(mapping, [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]))

  def test_filter_removes_collective_but_preserves_differential_motion(self) -> None:
    collective = np.zeros(16)
    collective[[0, 4]] = 0.1
    result = self._step(collective)
    self.assertTrue(result.solver_success)
    self.assertEqual(result.solver_status, "QP_SOLVED:DAQP")
    self.assertGreater(result.solver_iterations, 0)
    self.assertTrue(result.intervened)
    self.assertLessEqual(np.max(np.abs(result.safe_collective_motion_m)), 2.001e-4)

    differential = np.zeros(16)
    differential[0] = 0.05
    differential[4] = -0.05
    allowed = self._step(differential)
    self.assertFalse(allowed.intervened)
    self.assertTrue(np.allclose(allowed.safe_delta_rad, differential))

  def test_existing_collective_preload_is_not_counted_as_new_motion(self) -> None:
    preload = np.zeros(16)
    preload[[0, 4]] = 0.1
    jacobian = np.zeros((2, 16))
    jacobian[0, 0] = 0.01
    jacobian[1, 4] = 0.01
    selection = np.zeros((6, 1))
    selection[2, 0] = 1.0
    result = self.filter.step(
      current_q_rad=np.zeros(16),
      nominal_delta_rad=preload,
      previous_executed_command_rad=preload,
      previous_executed_velocity_rad_s=np.zeros(16),
      finger_normal_jacobian_m_per_rad=jacobian,
      active_contact_positions_palm_m=np.array([[-0.03, 0, 0], [0.03, 0, 0]]),
      active_outward_normals_palm=np.tile([0.0, 0.0, 1.0], (2, 1)),
      wrist_compliance_selection=selection,
      dt_s=0.02,
    )
    self.assertTrue(result.solver_success)
    self.assertFalse(result.intervened)
    self.assertTrue(np.allclose(result.safe_collective_motion_m, 0.0))

  def test_opposition_reports_rate_and_magnitude(self) -> None:
    finger = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    wrist = np.array([[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    metrics = opposition_metrics(
      finger,
      wrist,
      dt_s=0.01,
      finger_norm_threshold=0.1,
      wrist_norm_threshold=0.1,
    )
    self.assertEqual(metrics.valid_frame_count, 2)
    self.assertEqual(metrics.conflict_frame_count, 1)
    self.assertAlmostEqual(metrics.opposition_rate, 0.5)
    self.assertGreater(metrics.opposition_energy, 0.0)

  def test_box_projection_bypasses_unnecessary_numeric_qp(self) -> None:
    nominal = np.zeros(16)
    nominal[2] = 0.5
    result = self._step(nominal)
    self.assertTrue(result.solver_success)
    self.assertEqual(result.solver_status, "BOUNDED_NOMINAL_FEASIBLE")
    self.assertAlmostEqual(result.safe_delta_rad[2], 0.2)
    self.assertEqual(result.solver_iterations, 0)


class GuardStateMachineTest(unittest.TestCase):
  def setUp(self) -> None:
    self.guard = DPRuntimeGuardExecutor(
      DPGuardConfig(
        joint_lower_rad=-np.ones(16),
        joint_upper_rad=np.ones(16),
        dt_s=0.01,
        stable_time_s=0.02,
        hard_timeout_s=0.10,
      )
    )
    self.jacobian = np.zeros((4, 16))
    for finger in range(4):
      self.jacobian[finger, 4 * finger] = 1.0

  def _step(self, forces, ready=True):
    return self.guard.step(
      fingertip_force_n=forces,
      force_valid_mask=np.ones(4, dtype=bool),
      history_ready=ready,
      current_q_rad=np.zeros(16),
      signed_compression_jacobian=self.jacobian,
    )

  def test_hard_release_reduces_signed_compression_and_resets_history(self) -> None:
    self.assertEqual(self._step(np.zeros(4), False).state, DPGuardState.BUFFER_FILL)
    ready = self._step(np.zeros(4), True)
    self.assertEqual(ready.state, DPGuardState.REENTRY_RAMP)
    self.assertEqual(ready.dp_authority_scale, 0.0)
    for _ in range(20):
      active = self._step(np.zeros(4), True)
      if active.state is DPGuardState.DP_ACTIVE:
        break
    self.assertEqual(active.state, DPGuardState.DP_ACTIVE)
    hard = self._step([9.0, 0.0, 0.0, 0.0])
    self.assertEqual(hard.state, DPGuardState.HARD_RELEASE)
    assert hard.override_delta_rad is not None
    self.assertLess(float(self.jacobian[0] @ hard.override_delta_rad), 0.0)
    hold = self._step(np.zeros(4))
    self.assertEqual(hold.state, DPGuardState.SAFE_HOLD)
    self._step(np.zeros(4))
    reset = self._step(np.zeros(4))
    self.assertEqual(reset.state, DPGuardState.BUFFER_RESET)
    self.assertTrue(reset.reset_history)
    fill = self._step(np.zeros(4), False)
    self.assertEqual(fill.state, DPGuardState.BUFFER_FILL)
    self.assertEqual(fill.dp_authority_scale, 0.0)

  def test_rapid_loading_triggers_soft_damping_before_absolute_hard_limit(self) -> None:
    self._step(np.zeros(4), False)
    self._step(np.zeros(4), True)
    for _ in range(20):
      active = self._step(np.zeros(4), True)
      if active.state is DPGuardState.DP_ACTIVE:
        break
    rapid = self._step([5.5, 0.0, 0.0, 0.0])
    self.assertEqual(rapid.state, DPGuardState.SOFT_RECOVERY)
    self.assertLess(rapid.dp_authority_scale, 1.0)
    self.assertEqual(rapid.affected_fingers, (0,))
    self.assertIsNotNone(rapid.override_delta_rad)
    self.assertLess(5.5, self.guard.config.hard_force_n)


class CommandContinuityTest(unittest.TestCase):
  def test_finger_and_wrist_targets_are_rate_limited(self) -> None:
    limiter = CommandContinuityLimiter(
      CommandContinuityConfig(
        max_finger_step_rad=0.004,
        max_wrist_translation_step_m=0.0005,
        max_wrist_rotation_step_rad=0.01,
      )
    )
    limiter.reset(
      finger_command_rad=np.zeros(16),
      wrist_pose_world=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    finger = limiter.limit_finger(np.full(16, 0.1))
    self.assertLessEqual(float(np.max(np.abs(finger))), 0.004000001)
    wrist = limiter.limit_wrist(
      np.array([0.03, 0.04, 0.0, np.cos(0.1), 0.0, 0.0, np.sin(0.1)])
    )
    self.assertLessEqual(float(np.linalg.norm(wrist[:3])), 0.000500001)
    angle = 2.0 * np.arccos(np.clip(abs(wrist[3]), 0.0, 1.0))
    self.assertLessEqual(float(angle), 0.010000001)

  def test_make_fingers_can_use_a_stricter_per_joint_limit(self) -> None:
    limiter = CommandContinuityLimiter()
    limiter.reset(
      finger_command_rad=np.zeros(16),
      wrist_pose_world=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    limits = np.full(16, 0.001)
    limits[12:] = 0.0002
    command = limiter.limit_finger(np.full(16, 0.1), limits)
    np.testing.assert_allclose(command[:12], 0.001)
    np.testing.assert_allclose(command[12:], 0.0002)

  def test_wrist_normal_and_tangent_steps_are_bounded_separately(self) -> None:
    limiter = CommandContinuityLimiter(
      CommandContinuityConfig(max_wrist_translation_step_m=0.00008)
    )
    limiter.reset(
      finger_command_rad=np.zeros(16),
      wrist_pose_world=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    pose = limiter.limit_wrist(
      np.array([0.01, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0]),
      normal_direction_world=np.array([0.0, 0.0, 1.0]),
      max_normal_step_m=0.00001,
    )
    self.assertLessEqual(abs(float(pose[2])), 0.000010001)
    self.assertLessEqual(abs(float(pose[0])), 0.000080001)


class ContactRoleInterpreterTest(unittest.TestCase):
  def setUp(self) -> None:
    self.interpreter = ContactRoleInterpreter(
      RoleInterpreterConfig(
        dt_s=0.01,
        make_confirm_time_s=0.02,
        break_confirm_time_s=0.02,
        load_ramp_time_s=0.04,
        release_ramp_time_s=0.04,
        request_confirm_time_s=0.01,
      )
    )
    self.keep = np.full(4, int(ContactRole.KEEP))
    self.target = np.full(4, 2.0)

  def _step(self, request: np.ndarray, force: list[float]):
    return self.interpreter.step(
      requested_roles=request,
      measured_force_n=np.asarray(force, dtype=np.float64),
      target_force_n=self.target,
    )

  def test_make_confirmation_and_load_ramp_are_time_confirmed(self) -> None:
    first = self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    self.assertEqual(first.roles[0], int(ContactRole.MAKE))
    self.assertEqual(first.reference_motion_scale[0], 0.5)
    second = self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    self.assertEqual(second.roles[0], int(ContactRole.KEEP))
    self.assertEqual(second.desired_force_n[0], 1.0)
    third = self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    self.assertGreater(third.desired_force_n[0], 1.0)
    self.assertLess(third.desired_force_n[0], 2.0)

  def test_release_preserves_one_anchor_and_ramps_force_before_free(self) -> None:
    self._step(self.keep, [1.0, 1.0, 0.0, 0.0])
    self._step(self.keep, [1.0, 1.0, 0.0, 0.0])
    request = self.keep.copy()
    request[:2] = int(ContactRole.RELEASE)
    release = self._step(request, [1.0, 1.0, 0.0, 0.0])
    self.assertEqual(release.roles[0], int(ContactRole.RELEASE))
    self.assertEqual(release.roles[1], int(ContactRole.KEEP))
    self.assertEqual(release.transition_reason[1], "RELEASE_DENIED_MINIMUM_CONTACT")
    self.assertEqual(release.tangential_scale[0], 0.0)
    self._step(request, [0.0, 1.0, 0.0, 0.0])
    cleared = self._step(request, [0.0, 1.0, 0.0, 0.0])
    self.assertEqual(cleared.roles[0], int(ContactRole.FREE))

  def test_unexpected_loss_does_not_invent_a_learned_break_transition(self) -> None:
    self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    for _ in range(4):
      lost = self._step(self.keep, [0.0, 0.0, 0.0, 0.0])
    self.assertEqual(lost.roles[0], int(ContactRole.KEEP))
    self.assertFalse(lost.mcc_enabled[0])
    self.assertFalse(lost.physical_contact_confirmed[0])
    self.assertTrue(lost.full_reference_authority[0])
    self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    recovered = self._step(self.keep, [1.0, 0.0, 0.0, 0.0])
    self.assertEqual(recovered.roles[0], int(ContactRole.KEEP))
    self.assertTrue(recovered.physical_contact_confirmed[0])
    self.assertTrue(recovered.mcc_enabled[0])


class DiffusionPolicySmokeTest(unittest.TestCase):
  def test_shared_encoder_loss_and_sampling_are_finite(self) -> None:
    torch.manual_seed(3)
    observation = valid_observation()
    inputs = observation_to_tensors(observation)
    policy = FingerDiffusionPolicy(
      DiffusionPolicyConfig(action_horizon_steps=4, diffusion_steps=4)
    )
    target = torch.zeros((1, 4, 16), dtype=torch.float32)
    loss = policy.diffusion_loss(inputs, target)
    self.assertTrue(torch.isfinite(loss))
    loss.backward()
    self.assertIsNotNone(policy.condition_encoder.force_encoder.input_projection.weight.grad)
    sampled = policy.sample(inputs)
    self.assertEqual(tuple(sampled.shape), (1, 4, 16))
    self.assertTrue(torch.all(torch.isfinite(sampled)))

  def test_deterministic_sampling_reuses_fixed_latent_and_returns_radians(self) -> None:
    torch.manual_seed(5)
    inputs = observation_to_tensors(valid_observation())
    config = DiffusionPolicyConfig(
      action_horizon_steps=4,
      diffusion_steps=4,
      action_scale_rad=0.05,
      max_abs_action_offset_rad=0.08,
    )
    policy = FingerDiffusionPolicy(config)
    latent = torch.zeros((1, 4, 16), dtype=torch.float32)
    first = policy.sample(inputs, initial_noise=latent, deterministic=True)
    second = policy.sample(inputs, initial_noise=latent, deterministic=True)
    self.assertTrue(torch.equal(first, second))
    self.assertLessEqual(float(torch.max(torch.abs(first))), 0.080001)


class DPRefTwoHeadSmokeTest(unittest.TestCase):
  def test_continuous_and_categorical_heads_share_one_encoder(self) -> None:
    torch.manual_seed(9)
    batch = 2
    inputs = {
      "force_history": torch.zeros(batch, 4, 20, 3),
      "finger_state_geometry": torch.zeros(batch, 4, 20),
      "wrist_real_twist_history": torch.zeros(batch, 20, 6),
      "wrist_mcc_offset_history": torch.zeros(batch, 20, 6),
      "wrist_mcc_velocity_history": torch.zeros(batch, 20, 6),
      "future_wrist_plan_twist": torch.zeros(batch, 20, 6),
      "q_meas_rad": torch.zeros(batch, 16),
      "previous_nominal_command_rad": torch.zeros(batch, 16),
      "previous_mcc_correction_rad": torch.zeros(batch, 16),
    }
    self.assertEqual(set(inputs), set(DPREF_INPUT_NAMES))
    policy = FingerDPRefPolicy(DPRefPolicyConfig(diffusion_steps=4))
    target = torch.zeros(batch, 20, 16)
    role = torch.tensor([[0, 1, 2, 3], [0, 0, 2, 3]])
    valid = torch.tensor([[True, True, True, True], [True, False, True, True]])
    loss = policy.loss(inputs, target, role, valid)
    self.assertTrue(torch.isfinite(loss.total))
    self.assertEqual(loss.valid_role_count, 7)
    loss.total.backward()
    offsets, roles, probabilities = policy.sample(
      inputs,
      initial_noise=torch.zeros(batch, 20, 16),
    )
    self.assertEqual(tuple(offsets.shape), (batch, 20, 16))
    self.assertEqual(tuple(roles.shape), (batch, 4))
    self.assertEqual(tuple(probabilities.shape), (batch, 4, 4))


if __name__ == "__main__":
  unittest.main()
