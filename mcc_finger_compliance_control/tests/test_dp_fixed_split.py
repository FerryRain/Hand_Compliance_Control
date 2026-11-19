from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_dp import load_fixed_episode_split  # noqa: E402


class FixedSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "data.h5"
        self.manifest = self.root / "split.json"
        self.payload = {
            "dp_h5": str(self.dataset),
            "action_field": "q_hand",
            "episodes": [
                {"episode_id": 0, "split": "train", "related_plan_group_id": "a"},
                {"episode_id": 1, "split": "val", "related_plan_group_id": "b"},
                {"episode_id": 2, "split": "test", "related_plan_group_id": "c"},
                {
                    "episode_id": 3,
                    "split": "excluded_unmapped",
                    "related_plan_group_id": "d",
                },
            ],
        }

    def _load(self) -> tuple[list[int], list[int], list[int], list[int]]:
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")
        return load_fixed_episode_split(self.manifest, self.dataset, {0, 1, 2, 3})

    def test_valid_split_keeps_test_and_excluded_out_of_training(self) -> None:
        self.assertEqual(self._load(), ([0], [1], [2], [3]))

    def test_wrong_dataset_is_rejected(self) -> None:
        self.payload["dp_h5"] = str(self.root / "other.h5")
        with self.assertRaisesRegex(ValueError, "different DP H5"):
            self._load()

    def test_group_leak_is_rejected(self) -> None:
        self.payload["episodes"][1]["related_plan_group_id"] = "a"
        with self.assertRaisesRegex(ValueError, "groups cross"):
            self._load()

    def test_missing_episode_is_rejected(self) -> None:
        self.payload["episodes"].pop()
        with self.assertRaisesRegex(ValueError, "differ from H5"):
            self._load()


if __name__ == "__main__":
    unittest.main()
