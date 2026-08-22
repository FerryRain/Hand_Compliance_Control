from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Module.common import (
  ContactState,
  JsonlEpisodeLogger,
  StateSnapshot,
  load_jsonl_episode,
)


def make_snapshot(step: int = 0) -> StateSnapshot:
  return StateSnapshot(
    timestamp_s=0.01 * step,
    episode_id="mock-episode",
    step=step,
    seed=7,
    frame_id="world",
    evaluator_version="module-evaluator.v1",
    sampling_period_s=0.01,
    wrist_pose=[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
    wrist_twist=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    wrist_wrench=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    q=np.linspace(-0.2, 0.2, 8),
    dq=np.zeros(8),
    previous_action=np.zeros(8),
    current_command=np.full(8, 0.01),
    fingertip_positions=[
      [0.02, 0.02, 0.1],
      [0.02, -0.02, 0.1],
      [-0.02, 0.02, 0.1],
      [-0.02, -0.02, 0.1],
    ],
    fingertip_forces=[1.0, 0.0, 2.0, 0.0],
    contact_states=("CONTACT", "FREE", "CONTACT", "FREE"),
    predicted_contact_set=(1, 3),
    surface_model_version="oracle-demo-v1",
    planned_trajectory=({"t": 0.0, "wrist": [0.0, 0.0, 0.2]},),
    committed_prefix=({"transaction": "tx-1", "duration_s": 0.02},),
    prediction_suffix=({"prediction_only": True, "duration_s": 0.05},),
    transaction_id="tx-1",
    micro_barrier_state="OPEN",
    blocked_evidence={"reason": None},
    latencies_s={"planning": 0.0, "execution": 0.001},
    collision_distance_m=0.03,
    joint_margin_rad=0.2,
    anchor_margin_m=0.01,
    reach_margin_m=0.02,
    safety_override=None,
    contact_event="CONTACT_STABLE",
    certificate_id="demo-certificate",
  )


class StateSnapshotTest(unittest.TestCase):
  def test_dict_round_trip_and_authoritative_contact_set(self) -> None:
    original = make_snapshot()
    restored = StateSnapshot.from_dict(original.to_dict())

    self.assertEqual(original.to_dict(), restored.to_dict())
    self.assertEqual(restored.actual_contact_set, frozenset({1, 3}))
    self.assertEqual(restored.contact_states[0], ContactState.CONTACT)
    self.assertEqual(restored.transaction_id, "tx-1")
    self.assertEqual(restored.prediction_suffix[0]["prediction_only"], True)

    forged = original.to_dict()
    forged["actual_contact_set"] = [2, 4]
    self.assertEqual(
      StateSnapshot.from_dict(forged).actual_contact_set,
      frozenset({1, 3}),
    )

  def test_jsonl_episode_round_trip(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "episode.jsonl"
      expected = [make_snapshot(step) for step in range(3)]
      with JsonlEpisodeLogger(path) as logger:
        for snapshot in expected:
          logger.append(snapshot)
      actual = load_jsonl_episode(path)

    self.assertEqual(
      [snapshot.to_dict() for snapshot in expected],
      [snapshot.to_dict() for snapshot in actual],
    )

  def test_rejects_invalid_numeric_contracts(self) -> None:
    payload = make_snapshot().to_dict()
    payload["wrist_pose"] = [0.0, 0.0, 0.2, 2.0, 0.0, 0.0, 0.0]
    with self.assertRaisesRegex(ValueError, "unit length"):
      StateSnapshot.from_dict(payload)

    payload = make_snapshot().to_dict()
    payload["dq"] = [0.0]
    with self.assertRaisesRegex(ValueError, "identical shapes"):
      StateSnapshot.from_dict(payload)

    payload = make_snapshot().to_dict()
    payload["fingertip_forces"][0] = float("nan")
    with self.assertRaisesRegex(ValueError, "finite"):
      StateSnapshot.from_dict(payload)


if __name__ == "__main__":
  unittest.main()
