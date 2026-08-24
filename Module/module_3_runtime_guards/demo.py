"""Reproduce free-motion, blockage, joint, force, and collision guard cases."""

from __future__ import annotations

import json
from typing import Any

from Module.module_3_runtime_guards import (
  GuardObservation,
  GuardReason,
  RuntimeGuardConfig,
  RuntimeGuards,
)


def _observation(
  *,
  q=(0.0, 0.0),
  command=(0.1, 0.0),
  actual=(0.1, 0.0),
  forces=(0.0, 0.0, 0.0, 0.0),
  contacts=("FREE", "FREE", "FREE", "FREE"),
  collision_distance=0.02,
) -> GuardObservation:
  return GuardObservation(
    q_rad=q,
    qd_command_rad_s=command,
    qd_actual_rad_s=actual,
    fingertip_forces_n=forces,
    contact_states=contacts,
    min_self_collision_distance_m=collision_distance,
  )


def run_demo() -> dict[str, Any]:
  config = RuntimeGuardConfig(joint_lower_rad=[-1.0, -1.0], joint_upper_rad=[1.0, 1.0])

  free_guard = RuntimeGuards(config)
  free_decisions = [free_guard.evaluate(_observation()) for _ in range(200)]
  false_positives = sum(decision.should_stop for decision in free_decisions)

  blocked_guard = RuntimeGuards(config)
  blockage_detection_s: float | None = None
  blockage_decision = None
  for frame in range(1, 101):
    blockage_decision = blocked_guard.evaluate(_observation(actual=(0.0, 0.0)))
    if blockage_decision.reason is GuardReason.SUSPECTED_OBJECT_BLOCKAGE:
      blockage_detection_s = frame * config.dt_s
      break

  joint_guard = RuntimeGuards(config)
  joint_decision = joint_guard.evaluate(
    _observation(q=(0.99, 0.0), command=(0.1, 0.0), actual=(0.0, 0.0))
  )

  force_guard = RuntimeGuards(config)
  force_decision = force_guard.evaluate(_observation(forces=(3.6, 0.0, 0.0, 0.0)))

  collision_guard = RuntimeGuards(config)
  collision_decision = collision_guard.evaluate(_observation(collision_distance=0.001))

  reset_guard = RuntimeGuards(config)
  for _ in range(10):
    reset_guard.evaluate(_observation(actual=(0.0, 0.0)))
  reset_guard.evaluate(_observation(actual=(0.1, 0.0)))
  stall_after_recovery_s = reset_guard.stall_duration_s

  passed = (
    false_positives == 0
    and blockage_detection_s is not None
    and 0.15 <= blockage_detection_s <= 0.17
    and blockage_decision is not None
    and blockage_decision.evidence.local_observation_only
    and joint_decision.reason is GuardReason.JOINT_LIMIT
    and force_decision.reason is GuardReason.TIP_OVERFORCE
    and collision_decision.reason is GuardReason.SELF_COLLISION
    and stall_after_recovery_s == 0.0
  )
  return {
    "module": "M03",
    "passed": passed,
    "config": {
      "dt_s": config.dt_s,
      "stall_time_s": config.stall_time_s,
      "max_tip_force_n": config.max_tip_force_n,
      "joint_limit_margin_rad": config.joint_limit_margin_rad,
      "min_self_collision_distance_m": config.min_self_collision_distance_m,
    },
    "metrics": {
      "free_motion_frames": len(free_decisions),
      "blocked_false_positives": false_positives,
      "blockage_detection_s": blockage_detection_s,
      "joint_limit_response_s": config.dt_s,
      "overforce_response_s": config.dt_s,
      "self_collision_response_s": config.dt_s,
      "stall_after_recovery_s": stall_after_recovery_s,
    },
    "decisions": {
      "blockage": blockage_decision.to_dict() if blockage_decision else None,
      "joint_limit": joint_decision.to_dict(),
      "overforce": force_decision.to_dict(),
      "self_collision": collision_decision.to_dict(),
    },
  }


def main() -> None:
  result = run_demo()
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
