"""Run the static, sliding, cylinder, and sphere MCC demos."""

from __future__ import annotations

import json
from typing import Any

from Module.module_2_fingertip_mcc.benchmarks import (
  DEFAULT_CONFIG,
  run_curved_surface,
  run_static_contact,
  run_tangential_sliding,
)


def run_demo() -> dict[str, Any]:
  static = {
    str(force): run_static_contact(force).to_dict()
    for force in (1.0, 2.0, 3.0)
  }
  sliding = run_tangential_sliding().to_dict()
  curved = {
    surface: run_curved_surface(surface).to_dict()
    for surface in ("cylinder", "sphere")
  }

  static_pass = all(
    metrics["force_rmse_n"] <= 0.05
    and metrics["overshoot_n"] <= 0.20
    and metrics["force_violation_probability"] == 0.0
    for metrics in static.values()
  )
  sliding_pass = (
    sliding["force_rmse_n"] <= 0.05
    and sliding["max_tangential_error_m"] <= 1e-9
    and sliding["contact_loss_count_after_settling"] == 0
  )
  curved_pass = all(
    metrics["force_rmse_n"] <= 0.06
    and metrics["max_tangential_error_m"] <= 1e-6
    and metrics["contact_loss_count_after_settling"] == 0
    for metrics in curved.values()
  )
  limit_pass = all(
    metrics["max_abs_offset_m"] <= DEFAULT_CONFIG.max_offset_m
    and metrics["max_abs_velocity_m_s"] <= DEFAULT_CONFIG.max_velocity_m_s
    and metrics["max_abs_acceleration_m_s2"] <= DEFAULT_CONFIG.max_acceleration_m_s2
    for metrics in [*static.values(), sliding, *curved.values()]
  )
  return {
    "module": "M02",
    "passed": static_pass and sliding_pass and curved_pass and limit_pass,
    "config": {
      "dt_s": DEFAULT_CONFIG.dt_s,
      "virtual_mass": DEFAULT_CONFIG.virtual_mass,
      "damping": DEFAULT_CONFIG.damping,
      "stiffness": DEFAULT_CONFIG.stiffness,
      "max_offset_m": DEFAULT_CONFIG.max_offset_m,
      "max_velocity_m_s": DEFAULT_CONFIG.max_velocity_m_s,
      "max_acceleration_m_s2": DEFAULT_CONFIG.max_acceleration_m_s2,
    },
    "static": static,
    "sliding": sliding,
    "curved": curved,
  }


def main() -> None:
  result = run_demo()
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
