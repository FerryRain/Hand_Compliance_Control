from __future__ import annotations

import unittest

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


def box_sdf(point: np.ndarray, half_extents: np.ndarray) -> float:
  q = np.abs(point) - half_extents
  return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))


def cylinder_sdf(point: np.ndarray, radius: float, half_length: float) -> float:
  d = np.array([np.linalg.norm(point[:2]) - radius, abs(point[2]) - half_length])
  return float(np.linalg.norm(np.maximum(d, 0.0)) + min(float(np.max(d)), 0.0))


def box_normal(point: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
  closest = np.clip(point, -half_extents, half_extents)
  outside_delta = point - closest
  outside_distance = float(np.linalg.norm(outside_delta))
  if outside_distance > 1e-12:
    return outside_delta / outside_distance
  gaps = half_extents - np.abs(point)
  axis = int(np.argmin(gaps))
  normal = np.zeros(3)
  normal[axis] = 1.0 if point[axis] >= 0.0 else -1.0
  return normal


def cylinder_normal(point: np.ndarray, radius: float, half_length: float) -> np.ndarray:
  radial_distance = float(np.linalg.norm(point[:2]))
  radial_direction = point[:2] / radial_distance
  radial_excess = radial_distance - radius
  axial_excess = abs(point[2]) - half_length
  if radial_excess > 0.0 or axial_excess > 0.0:
    closest = np.array(
      [
        radial_direction[0] * min(radial_distance, radius),
        radial_direction[1] * min(radial_distance, radius),
        np.clip(point[2], -half_length, half_length),
      ]
    )
    delta = point - closest
    return delta / np.linalg.norm(delta)
  side_gap = radius - radial_distance
  cap_gap = half_length - abs(point[2])
  if side_gap <= cap_gap:
    return np.array([radial_direction[0], radial_direction[1], 0.0])
  return np.array([0.0, 0.0, 1.0 if point[2] >= 0.0 else -1.0])


def dense_clearance(model: OracleSurfaceModel, link: CapsuleLink) -> float:
  direction = link.end - link.start
  return min(
    model.shape.signed_distance(link.start + alpha * direction) - link.radius
    for alpha in np.linspace(0.0, 1.0, 20_001)
  )


class OracleSurfaceModelTest(unittest.TestCase):
  def setUp(self) -> None:
    self.rng = np.random.default_rng(7)
    self.shapes = {
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

  def test_point_distance_against_closed_form_reference(self) -> None:
    for _ in range(200):
      point = self.rng.uniform(-0.2, 0.2, size=3)
      expected = {
        "plane": point[2],
        "sphere": np.linalg.norm(point) - 0.1,
        "cylinder": cylinder_sdf(point, 0.08, 0.12),
        "box": box_sdf(point, np.array([0.08, 0.06, 0.05])),
        "rounded_box": box_sdf(point, np.array([0.07, 0.05, 0.04])) - 0.01,
      }
      expected_normals = {
        "plane": np.array([0.0, 0.0, 1.0]),
        "sphere": point / np.linalg.norm(point),
        "cylinder": cylinder_normal(point, 0.08, 0.12),
        "box": box_normal(point, np.array([0.08, 0.06, 0.05])),
        "rounded_box": box_normal(point, np.array([0.07, 0.05, 0.04])),
      }
      for name, shape in self.shapes.items():
        with self.subTest(shape=name):
          query = shape.query(point)
          self.assertLessEqual(abs(query.signed_distance - expected[name]), 1e-10)
          cosine = float(np.clip(np.dot(query.normal, expected_normals[name]), -1.0, 1.0))
          self.assertLessEqual(float(np.arccos(cosine)), 1e-7)

  def test_projection_and_canonical_normals(self) -> None:
    canonical = {
      "plane": (np.array([0.03, -0.02, 0.1]), np.array([0.0, 0.0, 1.0])),
      "sphere": (np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
      "cylinder": (np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
      "box": (np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
      "rounded_box": (np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    }
    for name, shape in self.shapes.items():
      point, expected_normal = canonical[name]
      query = shape.query(point)
      residual = abs(shape.signed_distance(query.point))
      cosine = float(np.clip(np.dot(query.normal, expected_normal), -1.0, 1.0))
      angle = float(np.arccos(cosine))
      with self.subTest(shape=name):
        self.assertLessEqual(residual, 1e-9)
        self.assertLessEqual(angle, 1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(query.normal)), 1.0, places=12)

    cap_query = self.shapes["cylinder"].query([0.0, 0.0, 0.2])
    self.assertTrue(np.allclose(cap_query.normal, [0.0, 0.0, 1.0]))

  def test_projection_residual_for_random_queries(self) -> None:
    for name, shape in self.shapes.items():
      for point in self.rng.uniform(-0.18, 0.18, size=(200, 3)):
        with self.subTest(shape=name):
          query = shape.query(point)
          self.assertLessEqual(abs(shape.signed_distance(query.point)), 1e-9)

  def test_capsule_clearance_matches_dense_reference(self) -> None:
    max_error = 0.0
    for name, shape in self.shapes.items():
      model = OracleSurfaceModel(shape, version=f"oracle-{name}-v1")
      for case in range(2):
        link = CapsuleLink(
          self.rng.uniform(-0.16, 0.16, size=3),
          self.rng.uniform(-0.16, 0.16, size=3),
          0.004 + 0.001 * case,
          name=f"{name}-{case}",
        )
        predicted = model.query_clearance(link)
        reference = dense_clearance(model, link)
        max_error = max(max_error, abs(predicted.clearance - reference))
        self.assertEqual(predicted.model_version, model.version)
    self.assertLessEqual(max_error, 5e-5)

  def test_contact_candidates_are_on_surface_and_reachable(self) -> None:
    model = OracleSurfaceModel(self.shapes["sphere"], version="oracle-sphere-v1")
    request = ContactCandidateRequest(
      finger_id=2,
      workspace_center=[0.0, 0.0, 0.14],
      reach_radius=0.22,
      count=8,
      seed=7,
    )
    candidates = model.sample_contact_candidates(request)
    self.assertEqual(len(candidates), request.count)
    for candidate in candidates:
      self.assertEqual(candidate.finger_id, 2)
      self.assertEqual(candidate.model_version, model.version)
      self.assertLessEqual(abs(model.shape.signed_distance(candidate.position)), 1e-9)
      self.assertLessEqual(candidate.reach_distance, request.reach_radius)
      self.assertAlmostEqual(float(np.linalg.norm(candidate.outward_normal)), 1.0, places=12)

  def test_uncertainty_is_zero_and_version_is_fixed(self) -> None:
    model = OracleSurfaceModel(self.shapes["plane"], version="oracle-plane-v1")
    self.assertEqual(model.version, "oracle-plane-v1")
    self.assertEqual(model.query_uncertainty([0.1, 0.2, 0.3]), 0.0)


if __name__ == "__main__":
  unittest.main()
