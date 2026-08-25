from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Module.e05_strategy_review import build_review


def _metric(mean: float, maximum: float | None = None) -> dict[str, float]:
  return {"mean": mean, "min": mean, "max": mean if maximum is None else maximum}


class E05StrategyReviewTest(unittest.TestCase):
  def test_combines_six_strategies_without_strategy_verdict(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      exp1 = root / "exp1"
      exp2 = root / "exp2"
      output = root / "review"
      exp1.mkdir()
      exp2.mkdir()

      exp1_metrics = {
        "contact_continuity_probability": _metric(0.8),
        "average_contact_count": _metric(2.0),
        "force_rmse_n": _metric(1.0),
        "max_tip_force_n": _metric(7.0, 9.0),
        "traversal_y_m": _metric(0.1),
        "zero_contact_time_s": _metric(0.2),
        "controller_latency_p95_s": _metric(0.001),
        "wrist_force_z_rmse_n": _metric(1.5),
        "force_violation_time_s": _metric(0.01),
        "hard_guard_frames": _metric(2.0),
      }
      (exp1 / "summary.json").write_text(json.dumps({
        "cells": {
          "E05-H-MCC": {"numeric_metrics": exp1_metrics},
          "E05-H-DP": {"numeric_metrics": exp1_metrics},
        }
      }), encoding="utf-8")

      exp2_metrics = {
        "contact_continuity_probability": _metric(0.9),
        "average_contact_count": _metric(1.5),
        "contact_count_ge2_probability": _metric(0.55),
        "contact_count_ge3_probability": _metric(0.25),
        "four_contact_probability": _metric(0.10),
        "force_rmse_n": _metric(1.2),
        "max_tip_force_n": _metric(6.0, 7.0),
        "traversal_y_m": _metric(0.12),
        "supported_y_traversal_ge2_m": _metric(0.05),
        "zero_contact_time_s": _metric(0.1),
        "reference_inference_latency_p95_s": _metric(0.01),
        "force_violation_time_s": _metric(0.0),
        "hard_guard_frames": _metric(0.0),
      }
      (exp2 / "summary.json").write_text(json.dumps({
        "aggregates": {
          "PLAIN_WHOLE_HAND_MCC": exp2_metrics,
          "PASSIVE_HOLD_MCC": exp2_metrics,
          "REACTIVE_HEURISTIC_MCC": exp2_metrics,
          "DPREF_MCC": exp2_metrics,
        }
      }), encoding="utf-8")

      for directory, names in (
        (exp1, (
          "e05_h_mcc_vs_dp_dashboard.png", "e05_h_mcc_nominal.mp4",
          "e05_h_dp_nominal.mp4", "e05_h_mcc_vs_dp_side_by_side.mp4",
        )),
        (exp2, (
          "exp2_comparison.png", "plain_whole_hand_mcc_video.mp4",
          "passive_hold_mcc_video.mp4",
          "reactive_heuristic_mcc_video.mp4", "dpref_mcc_video.mp4",
        )),
      ):
        for name in names:
          (directory / name).write_bytes(b"asset")

      summary = build_review(exp1, exp2, output)
      self.assertEqual(
        summary["evaluation_semantics"],
        "DESCRIPTIVE_ONLY_NO_STRATEGY_PASS_FAIL",
      )
      strategies = [
        strategy
        for experiment in summary["experiments"].values()
        for strategy in experiment["strategies"]
      ]
      self.assertEqual(len(strategies), 6)
      self.assertNotIn("g1b", summary)
      self.assertNotIn("performance_verdict", json.dumps(summary))
      self.assertAlmostEqual(strategies[0]["limit_observation"]["excess_n"], 1.0)
      self.assertTrue((output / "index.html").is_file())
      self.assertTrue((output / "metrics.csv").is_file())


if __name__ == "__main__":
  unittest.main()
