"""Deterministic command-line demo for Module 1."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from Module.module_1_oracle_surface_model import (
  Box,
  CapsuleLink,
  ContactCandidateRequest,
  Cylinder,
  OracleSurfaceModel,
  Plane,
  RoundedBox,
  Sphere,
)


def _dense_clearance(model: OracleSurfaceModel, link: CapsuleLink) -> float:
  alphas = np.linspace(0.0, 1.0, 20_001)
  direction = link.end - link.start
  return min(
    model.shape.signed_distance(link.start + float(alpha) * direction) - link.radius
    for alpha in alphas
  )


def run_demo(seed: int = 7) -> dict[str, Any]:
  rng = np.random.default_rng(seed)
  shapes = {
    "plane": Plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    "sphere": Sphere([0.0, 0.0, 0.0], 0.1),
    "cylinder": Cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.08, 0.12),
    "box": Box([0.0, 0.0, 0.0], [0.08, 0.06, 0.05]),
    "rounded_box": RoundedBox(
      [0.0, 0.0, 0.0],
      [0.07, 0.05, 0.04],
      0.01,
    ),
  }

  max_projection_residual = 0.0
  max_normal_unit_error = 0.0
  max_clearance_error = 0.0
  clearance_cases = 0
  for name, shape in shapes.items():
    model = OracleSurfaceModel(shape, version=f"oracle-{name}-v1")
    points = rng.uniform(-0.18, 0.18, size=(200, 3))
    for point in points:
      query = model.query_surface(point)
      projection_residual = abs(model.shape.signed_distance(query.point))
      max_projection_residual = max(max_projection_residual, projection_residual)
      max_normal_unit_error = max(
        max_normal_unit_error,
        abs(float(np.linalg.norm(query.normal)) - 1.0),
      )

    for case in range(10):
      link = CapsuleLink(
        rng.uniform(-0.16, 0.16, size=3),
        rng.uniform(-0.16, 0.16, size=3),
        radius=0.003 + 0.001 * (case % 3),
        name=f"{name}-{case}",
      )
      predicted = model.query_clearance(link).clearance
      dense = _dense_clearance(model, link)
      max_clearance_error = max(max_clearance_error, abs(predicted - dense))
      clearance_cases += 1

  sphere_model = OracleSurfaceModel(shapes["sphere"], version="oracle-sphere-v1")
  request = ContactCandidateRequest(
    finger_id=1,
    workspace_center=[0.0, 0.0, 0.14],
    reach_radius=0.22,
    count=8,
    seed=seed,
  )
  candidates = sphere_model.sample_contact_candidates(request)
  max_candidate_residual = max(
    abs(sphere_model.shape.signed_distance(candidate.position))
    for candidate in candidates
  )
  max_candidate_reach = max(candidate.reach_distance for candidate in candidates)

  thresholds = {
    "projection_residual_m": 1e-9,
    "normal_unit_error": 1e-12,
    "clearance_error_m": 5e-5,
    "candidate_surface_residual_m": 1e-9,
  }
  metrics = {
    "seed": seed,
    "shape_count": len(shapes),
    "point_queries": 200 * len(shapes),
    "clearance_cases": clearance_cases,
    "max_projection_residual_m": max_projection_residual,
    "max_normal_unit_error": max_normal_unit_error,
    "max_clearance_error_m": max_clearance_error,
    "candidate_count": len(candidates),
    "max_candidate_surface_residual_m": max_candidate_residual,
    "max_candidate_reach_m": max_candidate_reach,
  }
  passed = (
    metrics["max_projection_residual_m"] <= thresholds["projection_residual_m"]
    and metrics["max_normal_unit_error"] <= thresholds["normal_unit_error"]
    and metrics["max_clearance_error_m"] <= thresholds["clearance_error_m"]
    and metrics["candidate_count"] == request.count
    and metrics["max_candidate_surface_residual_m"]
    <= thresholds["candidate_surface_residual_m"]
    and metrics["max_candidate_reach_m"] <= request.reach_radius
  )
  return {"module": "M01", "passed": passed, "thresholds": thresholds, "metrics": metrics}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args()
  result = run_demo(args.seed)
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
