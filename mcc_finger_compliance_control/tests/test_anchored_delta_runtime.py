from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from deploy_dp_inverse import (  # noqa: E402
    DPRuntime,
    attach_commanded_planner_features,
)


class AnchoredDeltaRuntimeTest(unittest.TestCase):
    def test_action_reconstruction_uses_one_current_q_anchor(self) -> None:
        runtime = DPRuntime.__new__(DPRuntime)
        runtime.action_representation = "anchored_delta_q"
        q = np.arange(16, dtype=np.float32) / 10.0
        delta = np.arange(4 * 16, dtype=np.float32).reshape(4, 16) / 1000.0
        result = runtime.absolute_action(delta, q)
        np.testing.assert_array_equal(result, q[None, :] + delta)

    def test_commanded_planner_feature_requires_exact_episode_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dp.h5"
            with h5py.File(path, "w") as file:
                file.attrs["planner_feature_source"] = "preissued_palm_command_pose_object"
                file.create_dataset("episode_id", data=np.zeros((3, 1), dtype=np.int32))
                file.create_dataset("episode_step", data=np.arange(3)[:, None])
                file.create_dataset("q_hand", data=np.zeros((3, 1, 16), dtype=np.float32))
                feature = np.arange(18, dtype=np.float32).reshape(3, 1, 1, 6)
                file.create_dataset("planner_palm_delta_pose_palm", data=feature)
            runtime = SimpleNamespace(
                config=SimpleNamespace(file=path),
                state_schema="contact_geometry_planner_motion",
                planner_waypoints=1,
            )
            data = {"q_hand": np.zeros((3, 16), dtype=np.float32)}
            attach_commanded_planner_features(data, runtime, 0)
            np.testing.assert_array_equal(
                data["planner_palm_delta_pose_palm_commanded"], feature[:, 0]
            )
            bad = {"q_hand": np.ones((3, 16), dtype=np.float32)}
            with self.assertRaisesRegex(ValueError, "does not match"):
                attach_commanded_planner_features(bad, runtime, 0)


if __name__ == "__main__":
    unittest.main()
