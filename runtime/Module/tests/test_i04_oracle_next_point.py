from __future__ import annotations

import unittest

import numpy as np
import trimesh

from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point import BunnySurfaceGraph, CoverageLedger
from Module.i04_oracle_next_point.planner import _whole_hand_roles
from Module.module_7_contact_mode_graph import PrimitiveKind


class I04SurfaceGraphTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.graph = BunnySurfaceGraph(
      canonical_bunny_heightfield().mesh,
      coverage_radius_m=0.025,
    )

  def test_fixed_required_set_covers_the_complete_connected_mesh(self) -> None:
    self.assertGreater(self.graph.required_goal_count, 200)
    self.assertLessEqual(self.graph.realized_cover_radius_m, 0.025 + 1e-12)
    self.assertAlmostEqual(
      self.graph.total_area_m2,
      canonical_bunny_heightfield().mesh.area,
      places=10,
    )
    self.assertEqual(
      len(np.unique(self.graph.vertex_goal_owner)),
      self.graph.required_goal_count,
    )

  def test_next_goal_is_replanned_from_measured_contact_position(self) -> None:
    ledger_a = CoverageLedger(self.graph)
    ledger_b = CoverageLedger(self.graph)
    goal_a = int(self.graph.required_vertices[0])
    goal_b = int(self.graph.required_vertices[-1])
    selection_a = ledger_a.select_from_measured_contacts(
      self.graph.vertices_m[[goal_a]],
      maximum_candidates=8,
    )
    selection_b = ledger_b.select_from_measured_contacts(
      self.graph.vertices_m[[goal_b]],
      maximum_candidates=8,
    )
    self.assertEqual(selection_a.root_vertex, goal_a)
    self.assertEqual(selection_b.root_vertex, goal_b)
    self.assertNotEqual(selection_a.goal.goal_id, selection_b.goal.goal_id)
    self.assertFalse(hasattr(selection_a.goal, "finger_id"))

  def test_infeasible_near_goal_is_not_deleted_from_ledger(self) -> None:
    ledger = CoverageLedger(self.graph)
    root = int(self.graph.required_vertices[0])
    initial = ledger.remaining_goal_ids.copy()
    selection = ledger.select_from_measured_contacts(
      self.graph.vertices_m[[root]],
      feasibility_score=lambda goal_id, _distance: None if goal_id == 0 else 0.0,
      maximum_candidates=8,
    )
    self.assertNotEqual(selection.goal.goal_id, 0)
    self.assertTrue(np.array_equal(initial, ledger.remaining_goal_ids))

  def test_arrival_uses_mesh_geodesic_and_real_contact_normal(self) -> None:
    ledger = CoverageLedger(self.graph)
    root = int(self.graph.required_vertices[0])
    selection = ledger.select_from_measured_contacts(self.graph.vertices_m[[root]])
    goal = selection.goal
    arrived = ledger.arrival_fingers(
      goal,
      goal.position_local_m[None, :],
      goal.normal_local[None, :],
    )
    wrong_normal = ledger.arrival_fingers(
      goal,
      goal.position_local_m[None, :],
      -goal.normal_local[None, :],
    )
    self.assertEqual(arrived, (1,))
    self.assertEqual(wrong_normal, ())


class I04WholeHandReferenceSemanticsTest(unittest.TestCase):
  def test_only_selected_finger_receives_surface_traversal_role(self) -> None:
    roles = _whole_hand_roles(
      frozenset({1, 2, 4}),
      PrimitiveKind.SLIDE,
      1,
    )
    self.assertEqual(
      roles,
      {
        "1": "EXPLORER",
        "2": "ANCHOR",
        "3": "FREE",
        "4": "ANCHOR",
      },
    )
    self.assertEqual(tuple(roles.values()).count("EXPLORER"), 1)

  def test_wrist_adjust_does_not_assign_four_finger_directions(self) -> None:
    roles = _whole_hand_roles(
      frozenset({1, 2, 4}),
      PrimitiveKind.WRIST_ADJUST,
      None,
    )
    self.assertEqual(
      roles,
      {
        "1": "ANCHOR",
        "2": "ANCHOR",
        "3": "FREE",
        "4": "ANCHOR",
      },
    )

  def test_break_releases_only_the_stalled_explorer(self) -> None:
    roles = _whole_hand_roles(
      frozenset({1, 2, 4}),
      PrimitiveKind.BREAK,
      1,
    )
    self.assertEqual(roles["1"], "RELEASE")
    self.assertEqual(roles["2"], "ANCHOR")
    self.assertEqual(roles["4"], "ANCHOR")


if __name__ == "__main__":
  unittest.main()
