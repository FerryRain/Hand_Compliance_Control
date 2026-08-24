from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from Module.module_1_oracle_surface_model import (
  ContactCandidateRequest,
  MeshScalePolicy,
  MeshSurface,
  OracleSurfaceModel,
)


BUNNY_ASSET = Path(__file__).resolve().parents[1] / "assets" / "stanford_bunny.ply"


class MeshSurfaceTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.surface = MeshSurface.from_file(
      BUNNY_ASSET,
      source_up_axis="y",
      scale_policy=MeshScalePolicy(0.30, 0.18),
    )
    cls.model = OracleSurfaceModel(cls.surface, version="mesh-bunny-test-v1")

  def test_bunny_is_large_grounded_and_traceable(self) -> None:
    sorted_extents = np.sort(self.surface.extents)[::-1]
    self.assertAlmostEqual(float(sorted_extents[0]), 0.30, places=12)
    self.assertGreaterEqual(float(sorted_extents[1]), 0.18)
    self.assertAlmostEqual(float(self.surface.bounds[0, 2]), 0.0, places=12)
    self.assertEqual(self.surface.face_count, 69_451)
    self.assertEqual(self.surface.source_path, BUNNY_ASSET)

  def test_local_surface_sign_and_candidates(self) -> None:
    rng = np.random.default_rng(7)
    points = self.surface.sample_surface(24, rng)
    for point in points:
      query = self.model.query_surface(point)
      self.assertLessEqual(abs(query.signed_distance), 1e-8)
      outside = self.model.query_surface(query.point + 0.001 * query.normal)
      inside = self.model.query_surface(query.point - 0.001 * query.normal)
      self.assertGreater(outside.signed_distance, 0.0)
      self.assertLess(inside.signed_distance, 0.0)

    candidates = self.model.sample_contact_candidates(
      ContactCandidateRequest(
        finger_id=1,
        workspace_center=[0.0, 0.0, 0.14],
        reach_radius=0.5,
        count=8,
        seed=7,
      )
    )
    self.assertEqual(len(candidates), 8)

  def test_same_scale_policy_accepts_ycb_style_mesh_path(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "ycb_like_box.ply"
      trimesh.creation.box(extents=[0.08, 0.04, 0.12]).export(path)
      surface = MeshSurface.from_file(
        path,
        source_up_axis="z",
        scale_policy=MeshScalePolicy(0.30, 0.18),
      )
    sorted_extents = np.sort(surface.extents)[::-1]
    self.assertAlmostEqual(float(sorted_extents[0]), 0.30, places=12)
    self.assertGreaterEqual(float(sorted_extents[1]), 0.18)
    self.assertAlmostEqual(float(surface.bounds[0, 2]), 0.0, places=12)


if __name__ == "__main__":
  unittest.main()
