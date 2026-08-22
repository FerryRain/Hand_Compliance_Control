from __future__ import annotations

import unittest

from Module.module_3_runtime_guards import (
  GuardObservation,
  GuardReason,
  GuardSeverity,
  RuntimeGuardConfig,
  RuntimeGuards,
)


def observation(
  *,
  q=(0.0, 0.0),
  command=(0.1, 0.0),
  actual=(0.1, 0.0),
  forces=(0.0, 0.0, 0.0, 0.0),
  collision_distance=0.02,
) -> GuardObservation:
  return GuardObservation(
    q_rad=q,
    qd_command_rad_s=command,
    qd_actual_rad_s=actual,
    fingertip_forces_n=forces,
    contact_states=("FREE", "FREE", "FREE", "FREE"),
    min_self_collision_distance_m=collision_distance,
  )


class RuntimeGuardsTest(unittest.TestCase):
  def setUp(self) -> None:
    self.config = RuntimeGuardConfig(
      joint_lower_rad=[-1.0, -1.0],
      joint_upper_rad=[1.0, 1.0],
    )

  def test_free_motion_has_no_false_positive(self) -> None:
    guards = RuntimeGuards(self.config)
    decisions = [guards.evaluate(observation()) for _ in range(200)]
    self.assertTrue(all(decision.reason is GuardReason.NONE for decision in decisions))
    self.assertEqual(guards.stall_duration_s, 0.0)

  def test_suspected_object_blockage_detection_latency(self) -> None:
    guards = RuntimeGuards(self.config)
    decision = None
    detection_frame = None
    for frame in range(1, 100):
      decision = guards.evaluate(observation(actual=(0.0, 0.0)))
      if decision.reason is GuardReason.SUSPECTED_OBJECT_BLOCKAGE:
        detection_frame = frame
        break

    self.assertIsNotNone(detection_frame)
    detection_s = detection_frame * self.config.dt_s
    self.assertGreaterEqual(detection_s, 0.15)
    self.assertLessEqual(detection_s, 0.17)
    assert decision is not None
    self.assertEqual(decision.severity, GuardSeverity.SOFT_STOP)
    self.assertTrue(decision.evidence.local_observation_only)
    self.assertNotIn("collision_point", decision.evidence.to_dict())
    self.assertNotIn("collision_normal", decision.evidence.to_dict())

  def test_nonquiet_stall_is_no_progress(self) -> None:
    guards = RuntimeGuards(self.config)
    decision = None
    for _ in range(20):
      decision = guards.evaluate(
        observation(actual=(0.0, 0.0), forces=(1.0, 0.0, 0.0, 0.0))
      )
    assert decision is not None
    self.assertEqual(decision.reason, GuardReason.NO_PROGRESS)

  def test_joint_limit_only_when_outside_or_commanding_toward_limit(self) -> None:
    guards = RuntimeGuards(self.config)
    toward = guards.evaluate(
      observation(q=(0.99, 0.0), command=(0.1, 0.0), actual=(0.0, 0.0))
    )
    self.assertEqual(toward.reason, GuardReason.JOINT_LIMIT)
    self.assertEqual(toward.evidence.joint_indices, (0,))
    self.assertEqual(toward.severity, GuardSeverity.HARD_STOP)

    guards.reset()
    away = guards.evaluate(
      observation(q=(0.99, 0.0), command=(-0.1, 0.0), actual=(-0.1, 0.0))
    )
    self.assertEqual(away.reason, GuardReason.NONE)

  def test_overforce_and_known_self_collision_respond_immediately(self) -> None:
    guards = RuntimeGuards(self.config)
    force = guards.evaluate(observation(forces=(3.6, 0.0, 0.0, 0.0)))
    self.assertEqual(force.reason, GuardReason.TIP_OVERFORCE)
    self.assertEqual(force.severity, GuardSeverity.HARD_STOP)

    collision = guards.evaluate(observation(collision_distance=0.001))
    self.assertEqual(collision.reason, GuardReason.SELF_COLLISION)
    self.assertEqual(collision.severity, GuardSeverity.HARD_STOP)

  def test_real_progress_resets_stall_timer(self) -> None:
    guards = RuntimeGuards(self.config)
    for _ in range(10):
      guards.evaluate(observation(actual=(0.0, 0.0)))
    self.assertAlmostEqual(guards.stall_duration_s, 0.1)
    guards.evaluate(observation(actual=(0.1, 0.0)))
    self.assertEqual(guards.stall_duration_s, 0.0)
    for _ in range(10):
      decision = guards.evaluate(observation(actual=(0.0, 0.0)))
    self.assertEqual(decision.reason, GuardReason.NONE)

  def test_dimension_mismatch_is_rejected(self) -> None:
    guards = RuntimeGuards(self.config)
    bad = GuardObservation(
      q_rad=[0.0],
      qd_command_rad_s=[0.1],
      qd_actual_rad_s=[0.1],
      fingertip_forces_n=[0.0],
      contact_states=("FREE",),
    )
    with self.assertRaisesRegex(ValueError, "dimension"):
      guards.evaluate(bad)


if __name__ == "__main__":
  unittest.main()
