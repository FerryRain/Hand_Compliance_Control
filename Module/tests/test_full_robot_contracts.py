from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Module.common import (
  FullRobotJsonlLogger,
  FullRobotStateSnapshot,
  load_full_robot_jsonl,
)


def snapshot(step: int = 0) -> FullRobotStateSnapshot:
  return FullRobotStateSnapshot(
    timestamp_s=0.002 * step,
    episode_id="fr3-contract-smoke",
    step=step,
    seed=7,
    surface_model_version="oracle-extreme-v1",
    palm_pose_world=[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
    palm_twist_world=np.zeros(6),
    wrist_wrench=[0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    arm_q_rad=np.zeros(7),
    arm_dq_rad_s=np.zeros(7),
    arm_command_rad=np.zeros(7),
    arm_external_torque_nm=np.zeros(7),
    finger_q_rad=np.zeros(16),
    finger_dq_rad_s=np.zeros(16),
    finger_command_rad=np.zeros(16),
    fingertip_positions_world_m=np.zeros((4, 3)),
    fingertip_force_vectors=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]] * 2),
    fingertip_normal_forces_n=[2.0, 0.0, 2.0, 0.0],
    contact_positions_world_m=np.zeros((4, 3)),
    contact_normals_world=np.tile([0.0, 0.0, 1.0], (4, 1)),
    contact_states=("CONTACT", "FREE", "CONTACT", "FREE"),
    contact_position_valid=(True, False, True, False),
    sensor_validity={"wrist_wrench": True, "tip_force": True, "joint_state": True},
  )


class FullRobotContractTest(unittest.TestCase):
  def test_round_trip_preserves_groups_frames_and_actual_contacts(self) -> None:
    original = snapshot()
    restored = FullRobotStateSnapshot.from_dict(original.to_dict())
    self.assertEqual(restored.to_dict(), original.to_dict())
    self.assertEqual(restored.q_rad.shape, (23,))
    self.assertEqual(restored.actual_contact_set, frozenset({1, 3}))
    self.assertEqual(restored.wrench_acting_on, "HAND")
    forged = original.to_dict()
    forged["actual_contact_set"] = [2, 4]
    self.assertEqual(
      FullRobotStateSnapshot.from_dict(forged).actual_contact_set,
      frozenset({1, 3}),
    )

  def test_jsonl_round_trip(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "full.jsonl"
      with FullRobotJsonlLogger(path) as logger:
        logger.append(snapshot(0))
        logger.append(snapshot(1))
      restored = load_full_robot_jsonl(path)
    self.assertEqual([item.step for item in restored], [0, 1])

  def test_rejects_ambiguous_or_invalid_full_robot_data(self) -> None:
    payload = snapshot().to_dict()
    payload["wrench_acting_on"] = "OBJECT"
    with self.assertRaisesRegex(ValueError, "HAND"):
      FullRobotStateSnapshot.from_dict(payload)
    payload = snapshot().to_dict()
    payload["contact_position_valid"][0] = False
    with self.assertRaisesRegex(ValueError, "valid contact"):
      FullRobotStateSnapshot.from_dict(payload)
    payload = snapshot().to_dict()
    payload["arm_joint_names"] = list(reversed(payload["arm_joint_names"]))
    with self.assertRaisesRegex(ValueError, "order"):
      FullRobotStateSnapshot.from_dict(payload)


if __name__ == "__main__":
  unittest.main()
