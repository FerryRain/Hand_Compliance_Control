from __future__ import annotations

import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/mjlab/tasks/leaphand/full_hand_mcc_core.py"
)
SPEC = importlib.util.spec_from_file_location("full_hand_mcc_core", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)

GEOMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/mjlab/tasks/leaphand/full_hand_mcc_geometry.py"
)
GEOMETRY_SPEC = importlib.util.spec_from_file_location(
    "full_hand_mcc_geometry", GEOMETRY_PATH
)
assert GEOMETRY_SPEC is not None and GEOMETRY_SPEC.loader is not None
GEOMETRY = importlib.util.module_from_spec(GEOMETRY_SPEC)
sys.modules[GEOMETRY_SPEC.name] = GEOMETRY
GEOMETRY_SPEC.loader.exec_module(GEOMETRY)

DIAGNOSTICS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/mjlab/tasks/leaphand/full_hand_mcc_planner_diagnostics.py"
)
DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "full_hand_mcc_planner_diagnostics", DIAGNOSTICS_PATH
)
assert DIAGNOSTICS_SPEC is not None and DIAGNOSTICS_SPEC.loader is not None
DIAGNOSTICS = importlib.util.module_from_spec(DIAGNOSTICS_SPEC)
sys.modules[DIAGNOSTICS_SPEC.name] = DIAGNOSTICS
DIAGNOSTICS_SPEC.loader.exec_module(DIAGNOSTICS)

DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/demo_surface_slide.py"
)
ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/mjlab/tasks/leaphand/leaphand_full_hand_mcc_env_cfg.py"
)
RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_baseline2_capsule_level2.ps1"
)


def load_demo_pure_function(name: str):
    """Load one top-level pure helper without importing MJLab or MuJoCo."""

    source = DEMO_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace = {
        "np": np,
        "smoothstep_joint_interpolation": (
            DIAGNOSTICS.smoothstep_joint_interpolation
        ),
    }
    exec(compile(module, str(DEMO_PATH), "exec"), namespace)
    return namespace[name]


class PlannerDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _candidate_rank(
        angle_deg: float,
        *,
        hard_feasible: bool = True,
        task_error: float = 0.0,
        self_clearance_mm: float = 0.2,
    ) -> tuple[float, ...]:
        return DIAGNOSTICS.orientation_aware_candidate_rank(
            hard_feasible=hard_feasible,
            hard_violation_score=0.0,
            minimum_pad_alignment=float(np.cos(np.deg2rad(angle_deg))),
            hard_pad_alignment=float(np.cos(np.deg2rad(40.0))),
            soft_pad_alignment=float(np.cos(np.deg2rad(35.0))),
            task_error_score=task_error,
            continuity_error=0.0,
            solver_cost=0.0,
            minimum_protected_self_clearance_m=(
                self_clearance_mm / 1000.0
            ),
            soft_self_clearance_target_m=0.1 / 1000.0,
        )

    def test_positive_self_clearance_residual_is_one_sided(self) -> None:
        residual = DIAGNOSTICS.positive_self_clearance_residual(
            np.asarray([-0.00002, 0.00004, 0.00020]),
            target_clearance_m=0.00010,
        )
        np.testing.assert_allclose(
            residual,
            np.asarray([0.00012, 0.00006, 0.0]),
        )

    def test_central_fd_self_separation_seeds_follow_clearance_ascent(
        self,
    ) -> None:
        plus = np.asarray([0.3, -0.1, 0.2]) * 2.0e-4
        minus = -plus
        gradient = DIAGNOSTICS.central_difference_clearance_gradient(
            plus,
            minus,
            np.full(3, 2.0e-4),
        )
        np.testing.assert_allclose(gradient, np.asarray([0.6, -0.2, 0.4]))
        seeds = DIAGNOSTICS.self_separation_ascent_seeds(
            np.zeros(3),
            gradient,
            np.full(3, -1.0),
            np.full(3, 1.0),
            maximum_step_rad=0.005,
        )
        self.assertEqual(len(seeds), 2)
        self.assertAlmostEqual(float(np.linalg.norm(seeds[0])), 0.002)
        self.assertAlmostEqual(float(np.linalg.norm(seeds[1])), 0.005)
        self.assertGreater(float(np.dot(seeds[0], gradient)), 0.0)
        self.assertGreater(float(np.dot(seeds[1], gradient)), 0.0)

    def test_suffix_seed_cap_prioritizes_protected_self_and_cache(self) -> None:
        kinds = (
            "previous",
            "extrapolated",
            "nullspace_combined",
            "nullspace_combined",
            "nullspace_joint_0",
            "nullspace_joint_1",
            "protected_self_small",
            "protected_self_large",
            "certified_cache",
        )
        indices = DIAGNOSTICS.prioritized_suffix_seed_indices(
            kinds,
            maximum_seeds=6,
        )
        self.assertEqual(indices, (0, 1, 6, 7, 8, 2))
        retained = tuple(kinds[index] for index in indices)
        self.assertIn("previous", retained)
        self.assertIn("extrapolated", retained)
        self.assertEqual(
            sum(kind.startswith("protected_self") for kind in retained),
            2,
        )
        self.assertIn("certified_cache", retained)

    def test_suffix_seed_tighter_cap_still_retains_certified_cache(self) -> None:
        kinds = (
            "previous",
            "extrapolated",
            "nullspace_combined",
            "nullspace_joint_0",
            "nullspace_joint_1",
            "protected_self_small",
            "protected_self_large",
            "certified_cache",
        )
        indices = DIAGNOSTICS.prioritized_suffix_seed_indices(
            kinds,
            maximum_seeds=5,
        )
        self.assertEqual(indices, (0, 1, 5, 6, 7))
        retained = tuple(kinds[index] for index in indices)
        self.assertIn("certified_cache", retained)
        self.assertEqual(
            sum(kind.startswith("protected_self") for kind in retained),
            2,
        )

    def test_suffix_seed_cap_without_cache_still_keeps_two_protected(self) -> None:
        kinds = (
            "previous",
            "extrapolated",
            "nullspace_combined",
            "nullspace_combined",
            "nullspace_joint_0",
            "nullspace_joint_1",
            "protected_self_small",
            "protected_self_large",
        )
        indices = DIAGNOSTICS.prioritized_suffix_seed_indices(
            kinds,
            maximum_seeds=6,
        )
        self.assertEqual(indices, (0, 1, 6, 7, 2, 3))

    def test_transported_suffix_seed_preserves_motion_and_anchor_delta(
        self,
    ) -> None:
        base = np.asarray([[0.1, 0.2], [0.2, 0.4], [0.4, 0.7]])
        anchor = np.asarray([0.05, 0.1])
        modified = np.asarray([0.03, 0.14])
        transported = DIAGNOSTICS.transported_suffix_seed_rows(
            base,
            anchor,
            modified,
        )
        np.testing.assert_allclose(transported, base + [-0.02, 0.04])
        np.testing.assert_allclose(
            np.diff(transported, axis=0),
            np.diff(base, axis=0),
        )

    def test_suffix_rollout_without_cache_keeps_old_protected_order(self) -> None:
        kinds = (
            "previous",
            "extrapolated",
            "protected_self",
            "protected_self",
            "nullspace_combined",
            "nullspace_combined",
        )
        indices = DIAGNOSTICS.prioritized_suffix_rollout_indices(
            kinds,
            (5, 4, 1, 2, 0, 3),
            maximum_sources=3,
        )
        self.assertEqual(indices, (5, 2, 3))
        self.assertEqual(
            sum(kinds[index].startswith("protected_self") for index in indices),
            2,
        )

    def test_suffix_rollout_keeps_best_cache_and_two_protected(self) -> None:
        kinds = (
            "previous",
            "protected_self_small",
            "nullspace_combined",
            "certified_cache",
            "protected_self_large",
        )
        indices = DIAGNOSTICS.prioritized_suffix_rollout_indices(
            kinds,
            (2, 0, 3, 4, 1),
            maximum_sources=4,
        )
        self.assertEqual(indices, (2, 3, 4, 1))

    def test_suffix_rollout_deduplicates_best_special_source(self) -> None:
        cases = (
            (
                ("certified_cache", "generic", "protected_self_a"),
                (0, 1, 2),
                (0, 2, 1),
            ),
            (
                ("protected_self_a", "generic", "certified_cache"),
                (0, 1, 2),
                (0, 2, 1),
            ),
        )
        for kinds, ranked, expected in cases:
            with self.subTest(best=kinds[ranked[0]]):
                indices = DIAGNOSTICS.prioritized_suffix_rollout_indices(
                    kinds,
                    ranked,
                    maximum_sources=3,
                )
                self.assertEqual(indices, expected)
                self.assertEqual(len(indices), len(set(indices)))

    def test_suffix_rollout_respects_small_source_caps(self) -> None:
        kinds = ("previous", "protected_self_a", "protected_self_b")
        ranked = (0, 2, 1)
        self.assertEqual(
            DIAGNOSTICS.prioritized_suffix_rollout_indices(
                kinds, ranked, maximum_sources=1
            ),
            (0,),
        )
        self.assertEqual(
            DIAGNOSTICS.prioritized_suffix_rollout_indices(
                kinds, ranked, maximum_sources=2
            ),
            (0, 2),
        )

    def test_bridge_multistart_keeps_previous_and_all_unique_seeds(
        self,
    ) -> None:
        previous = np.asarray([0.1, -0.2, 0.3])
        separation_a = np.asarray([0.11, -0.2, 0.3])
        separation_b = np.asarray([0.1, -0.19, 0.3])
        seeds = DIAGNOSTICS.deduplicated_bridge_multistart_seeds(
            previous,
            (
                separation_a,
                previous.copy(),
                separation_a.copy(),
                separation_b,
            ),
        )
        self.assertEqual(len(seeds), 3)
        np.testing.assert_array_equal(seeds[0], previous)
        np.testing.assert_array_equal(seeds[1], separation_a)
        np.testing.assert_array_equal(seeds[2], separation_b)

    def test_bridge_tip_residual_wires_target_and_inner_blocks(self) -> None:
        clearance = np.asarray([-0.0002, -0.0009, 0.0, -0.0005])
        target = np.asarray([-0.00025, -0.00025, -0.0005, -0.00025])
        residual = DIAGNOSTICS.moving_bridge_tip_geometry_residual(
            clearance,
            target,
            inner_cap_m=-0.0008,
            target_weight=2200.0,
            target_scale=0.5,
            inner_weight=18000.0,
        )
        np.testing.assert_allclose(
            residual[:4],
            0.5 * 2200.0 * (clearance - target),
        )
        np.testing.assert_allclose(
            residual[4:],
            18000.0 * np.minimum(clearance + 0.0008, 0.0),
        )
        self.assertEqual(np.count_nonzero(residual[4:]), 1)

    def test_bridge_rank_prefers_strict_then_recovery_then_tip_buffer(
        self,
    ) -> None:
        common = {
            "collision_hard_feasible": True,
            "failed_condition_count": 0,
            "minimum_tip_clearance_m": -0.0007,
            "tip_inner_cap_m": -0.0008,
            "minimum_protected_self_clearance_m": 0.0002,
            "soft_self_clearance_target_m": 0.0001,
            "minimum_pad_alignment": 0.9,
            "soft_pad_alignment": 0.8,
            "task_error_score": 0.0,
            "continuity_error": 0.0,
            "solver_cost": 0.0,
        }
        strict = DIAGNOSTICS.moving_bridge_candidate_rank(
            strict_hard_feasible=True,
            recovery_hard_feasible=False,
            **common,
        )
        recovery = DIAGNOSTICS.moving_bridge_candidate_rank(
            strict_hard_feasible=False,
            recovery_hard_feasible=True,
            **common,
        )
        rejected = DIAGNOSTICS.moving_bridge_candidate_rank(
            strict_hard_feasible=False,
            recovery_hard_feasible=False,
            **(common | {"failed_condition_count": 1}),
        )
        unsafe_tip = DIAGNOSTICS.moving_bridge_candidate_rank(
            strict_hard_feasible=True,
            recovery_hard_feasible=False,
            **(common | {"minimum_tip_clearance_m": -0.0009}),
        )
        self.assertLess(strict, recovery)
        self.assertLess(recovery, rejected)
        self.assertLess(strict, unsafe_tip)

    def test_bridge_rejection_rank_keeps_collision_safe_prefix(self) -> None:
        common = {
            "strict_hard_feasible": False,
            "recovery_hard_feasible": False,
            "minimum_tip_clearance_m": -0.0007,
            "tip_inner_cap_m": -0.0008,
            "minimum_protected_self_clearance_m": 0.0002,
            "soft_self_clearance_target_m": 0.0001,
            "minimum_pad_alignment": 0.9,
            "soft_pad_alignment": 0.8,
            "task_error_score": 0.0,
            "continuity_error": 0.0,
            "solver_cost": 0.0,
        }
        collision_safe_two_task_failures = (
            DIAGNOSTICS.moving_bridge_candidate_rank(
                collision_hard_feasible=True,
                failed_condition_count=2,
                **common,
            )
        )
        collision_unsafe_one_failure = (
            DIAGNOSTICS.moving_bridge_candidate_rank(
                collision_hard_feasible=False,
                failed_condition_count=1,
                **common,
            )
        )
        self.assertLess(
            collision_safe_two_task_failures,
            collision_unsafe_one_failure,
        )

    def test_seed42_bridge_tip_probe_fixture_passes_every_named_gate(
        self,
    ) -> None:
        # Frozen CPU measurements from the 45.9375 mm seed-42 failure
        # snapshot after adding the 0.5x physical-tip target residual.
        strict = DIAGNOSTICS.evaluate_bridge_conditions(
            progress_error_m=np.asarray([0.0, 0.004615616]),
            progress_limit_m=0.004631122,
            normal_ok=bool(
                np.all(
                    np.asarray([3.0179, 2.6063, 2.7050, 1.7653])
                    <= np.asarray([3.6311, 3.0, 3.0, 3.0])
                )
            ),
            tangential_error_m=(
                np.asarray([1.6000, 1.2152, 1.9785, 1.6467])
                / 1000.0
            ),
            tangential_limit_m=(
                np.asarray([2.2705, 2.0, 2.0, 2.0]) / 1000.0
            ),
            monotonic_error_m=np.zeros(5),
            monotonic_limit_m=0.0002,
            palm_error_m=0.0130195,
            palm_limit_m=0.030,
            collision_ok=bool(
                0.015332 >= 0.002
                and -0.0000385 >= -0.001
                and -0.0009255 >= -0.001
                and 36.147 <= 40.0
            ),
            joint_ok=bool(0.003128 <= 0.03 and 0.003608 >= 0.0),
            motion_ok=bool(
                np.count_nonzero(
                    np.asarray([0.1053, 0.1259, 0.1644, 0.0426])
                    >= 0.015625
                )
                >= 3
            ),
            budget_ok=True,
        )
        self.assertEqual(tuple(strict), DIAGNOSTICS.BRIDGE_CONDITION_NAMES)
        self.assertTrue(all(strict.values()))
        candidate_rank = DIAGNOSTICS.moving_bridge_candidate_rank(
            strict_hard_feasible=True,
            recovery_hard_feasible=False,
            collision_hard_feasible=True,
            failed_condition_count=0,
            minimum_tip_clearance_m=-0.0009255,
            tip_inner_cap_m=-0.0008,
            minimum_protected_self_clearance_m=0.0,
            soft_self_clearance_target_m=0.0001,
            minimum_pad_alignment=float(np.cos(np.deg2rad(36.147))),
            soft_pad_alignment=float(np.cos(np.deg2rad(35.0))),
            task_error_score=0.004615616,
            continuity_error=0.003128,
            solver_cost=1.0,
        )
        rejected_tip_rank = DIAGNOSTICS.moving_bridge_candidate_rank(
            strict_hard_feasible=False,
            recovery_hard_feasible=False,
            collision_hard_feasible=False,
            failed_condition_count=1,
            minimum_tip_clearance_m=-0.001041958,
            tip_inner_cap_m=-0.0008,
            minimum_protected_self_clearance_m=0.0,
            soft_self_clearance_target_m=0.0001,
            minimum_pad_alignment=float(np.cos(np.deg2rad(36.0))),
            soft_pad_alignment=float(np.cos(np.deg2rad(35.0))),
            task_error_score=0.0,
            continuity_error=0.0,
            solver_cost=0.0,
        )
        self.assertLess(candidate_rank, rejected_tip_rank)

    def test_soft_self_clearance_precedes_pad_and_task_rank(self) -> None:
        near_contact_20_deg = self._candidate_rank(
            20.0,
            task_error=0.0,
            self_clearance_mm=0.01,
        )
        separated_34_deg = self._candidate_rank(
            34.0,
            task_error=1000.0,
            self_clearance_mm=0.11,
        )
        self.assertLess(separated_34_deg, near_contact_20_deg)

    def test_soft_pad_residual_has_gradient_before_hard_cone(self) -> None:
        target = float(np.cos(np.deg2rad(35.0)))
        alignment_38 = float(np.cos(np.deg2rad(38.0)))
        alignment_39 = float(np.cos(np.deg2rad(39.0)))
        residual = DIAGNOSTICS.smooth_pad_alignment_residual(
            np.asarray([alignment_38, alignment_39]),
            target_alignment=target,
            tau=0.02,
        )
        self.assertGreater(residual[1], residual[0])
        epsilon = 1.0e-6
        plus = DIAGNOSTICS.smooth_pad_alignment_residual(
            np.asarray([alignment_39 + epsilon]),
            target_alignment=target,
            tau=0.02,
        )[0]
        minus = DIAGNOSTICS.smooth_pad_alignment_residual(
            np.asarray([alignment_39 - epsilon]),
            target_alignment=target,
            tau=0.02,
        )[0]
        self.assertLess((plus - minus) / (2.0 * epsilon), -0.5)

    def test_posture_candidate_beats_privileged_raw_outside_soft_cone(
        self,
    ) -> None:
        raw_39 = self._candidate_rank(39.0, task_error=0.0)
        posture_34 = self._candidate_rank(34.0, task_error=100.0)
        self.assertLess(posture_34, raw_39)

    def test_34_degree_fallback_beats_39_9_degree_fallback(self) -> None:
        fallback_39_9 = self._candidate_rank(39.9, task_error=0.0)
        fallback_34 = self._candidate_rank(34.0, task_error=1000.0)
        self.assertLess(fallback_34, fallback_39_9)

    def test_task_error_regains_priority_inside_soft_cone(self) -> None:
        task_accurate_34 = self._candidate_rank(34.0, task_error=0.1)
        task_worse_20 = self._candidate_rank(20.0, task_error=0.2)
        self.assertLess(task_accurate_34, task_worse_20)

    def test_midsegment_41_degree_pad_state_is_hard_infeasible(self) -> None:
        endpoint_safe = self._candidate_rank(39.0, task_error=10.0)
        midpoint_unsafe = self._candidate_rank(41.0, task_error=0.0)
        self.assertLess(endpoint_safe, midpoint_unsafe)
        self.assertEqual(midpoint_unsafe[0], 1.0)

    def test_midsegment_tip_penetration_rejects_safe_endpoints(self) -> None:
        tip_clearance = np.full((3, 4), -0.0005, dtype=np.float64)
        tip_clearance[1, 2] = -0.0012
        passed, minimum, index = (
            DIAGNOSTICS.segment_tip_clearance_status(
                tip_clearance,
                maximum_penetration_m=0.001,
            )
        )
        self.assertFalse(passed)
        self.assertAlmostEqual(minimum, -0.0012)
        self.assertEqual(index, (1, 2))
        tip_clearance[1, 2] = -0.001
        self.assertTrue(
            DIAGNOSTICS.segment_tip_clearance_status(
                tip_clearance,
                maximum_penetration_m=0.001,
            )[0]
        )

    @staticmethod
    def _stationary_plan_inputs(
        sample_count: int,
        *,
        route_length_m: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = np.linspace(0.0, route_length_m, sample_count)
        progress = np.zeros((sample_count, 5), dtype=np.float64)
        points = np.zeros((sample_count, 5, 3), dtype=np.float64)
        axial = target.copy()
        return progress, points, target, axial

    def test_low_motion_window_allows_19_intervals_and_rejects_20(
        self,
    ) -> None:
        progress, points, target, axial = self._stationary_plan_inputs(20)
        allowed = DIAGNOSTICS.find_unmarked_low_motion_windows(
            progress,
            points,
            target,
            np.zeros(20, dtype=bool),
            axial,
            window_frames=20,
        )
        self.assertEqual(allowed, [])

        progress, points, target, axial = self._stationary_plan_inputs(21)
        rejected = DIAGNOSTICS.find_unmarked_low_motion_windows(
            progress,
            points,
            target,
            np.zeros(21, dtype=bool),
            axial,
            window_frames=20,
        )
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0]["worst_window"]["forward_finger_count"],
            0,
        )

    def test_nonzero_joint_motion_does_not_hide_tip_stall_evidence(
        self,
    ) -> None:
        progress, points, target, axial = self._stationary_plan_inputs(21)
        joint_positions = np.zeros((21, 23), dtype=np.float64)
        joint_positions[:, 0] = np.linspace(0.0, 0.02, 21)
        regions = DIAGNOSTICS.find_unmarked_low_motion_windows(
            progress,
            points,
            target,
            np.zeros(21, dtype=bool),
            axial,
            window_frames=20,
        )
        self.assertEqual(len(regions), 1)
        self.assertGreater(
            float(np.max(np.abs(np.diff(joint_positions, axis=0)))),
            0.0,
        )
        self.assertEqual(
            regions[0]["first_window"]["forward_finger_count"],
            0,
        )

    def test_explicit_static_or_recovery_frame_exempts_window(self) -> None:
        progress, points, target, axial = self._stationary_plan_inputs(21)
        for bridge_kind in ("static", "recovery"):
            with self.subTest(bridge_kind=bridge_kind):
                marked = np.zeros(21, dtype=bool)
                marked[10] = True
                regions = DIAGNOSTICS.find_unmarked_low_motion_windows(
                    progress,
                    points,
                    target,
                    marked,
                    axial,
                    window_frames=20,
                )
                self.assertEqual(regions, [])

    def test_seed42_historical_350mm_platform_is_rejected(self) -> None:
        progress, points, target, axial = self._stationary_plan_inputs(
            33,
            route_length_m=0.35407792207792205 - 0.35033766233766234,
        )
        target += 0.35033766233766234
        axial += 0.35033766233766234
        regions = DIAGNOSTICS.find_unmarked_low_motion_windows(
            progress,
            points,
            target,
            np.zeros(33, dtype=bool),
            axial,
            window_frames=20,
        )
        self.assertEqual(len(regions), 1)
        worst = regions[0]["worst_window"]
        self.assertEqual(worst["forward_finger_count"], 0)
        self.assertAlmostEqual(
            worst["required_tip_progress_m"],
            0.1 * worst["route_delta_m"],
        )

    def test_seed42_4125_to_425mm_prefix_stall_is_rejected(self) -> None:
        target = np.linspace(0.0411875, 0.0424375, 21, dtype=np.float64)
        progress = np.zeros((21, 5), dtype=np.float64)
        progress[:, 1:] = np.linspace(
            np.zeros(4, dtype=np.float64),
            np.asarray(
                (0.000033727, 0.000165903, 0.000064609, 0.000203714)
            ),
            21,
        )
        points = np.zeros((21, 5, 3), dtype=np.float64)
        axial = np.min(progress[:, 1:], axis=1)
        regions = DIAGNOSTICS.find_unmarked_low_motion_windows(
            progress,
            points,
            target,
            np.zeros(21, dtype=bool),
            axial,
            window_frames=20,
        )
        self.assertEqual(len(regions), 1)
        first = regions[0]["first_window"]
        self.assertEqual(first["forward_finger_count"], 2)
        self.assertEqual(first["forward_mask"], [False, True, False, True])
        self.assertAlmostEqual(first["route_delta_m"], 0.00125)
        self.assertAlmostEqual(first["required_tip_progress_m"], 0.000125)

    def test_prospective_low_motion_gate_precedes_coarse_commit(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        helper_start = source.index("def prospective_low_motion_failures")
        loop_start = source.index("keyframe = 1", helper_start)
        helper = source[helper_start:loop_start]
        self.assertIn("smoothstep_joint_interpolation(", helper)
        self.assertIn("[-LOW_MOTION_DEFAULT_WINDOW_FRAMES:]", helper)
        self.assertIn("find_unmarked_low_motion_windows(", helper)
        self.assertIn("coarse_static_feasibility_bridge", helper)
        self.assertIn("coarse_recovery_bridge", helper)

        pre_rephase = source.index(
            "pre_rephase_low_motion_failures = (",
            loop_start,
        )
        auto_rephase = source.index("auto_rephase_needed = bool(", pre_rephase)
        self.assertLess(pre_rephase, auto_rephase)
        self.assertIn(
            "or bool(pre_rephase_low_motion_failures)",
            source[auto_rephase : auto_rephase + 1200],
        )
        rephase_gate = source.index(
            "if rephased_hard_ok:",
            auto_rephase,
        )
        self.assertIn(
            "prospective_low_motion_failures(",
            source[rephase_gate : rephase_gate + 800],
        )
        final_gate = source.index(
            "selected_low_motion_failures = (",
            rephase_gate,
        )
        coarse_commit = source.index("coarse_q[keyframe] = q", final_gate)
        self.assertLess(final_gate, coarse_commit)
        final_source = source[final_gate:coarse_commit]
        self.assertIn('reason="unmarked_low_motion"', final_source)
        self.assertIn("insert_auto_refinement(", final_source)
        self.assertIn("raise_adaptive_planner_failure(", final_source)
        self.assertIn("[PROSPECTIVE-LOW-MOTION]", final_source)

    def test_low_motion_gate_precedes_plan_publish_and_success_archive(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        gate = source.index(
            "low_motion_regions = find_unmarked_low_motion_windows"
        )
        publish = source.index("self.plan_surface = surface_plan", gate)
        success_archive = source.index(
            "np.savez_compressed(\n                args.plan_output",
            gate,
        )
        self.assertLess(gate, publish)
        self.assertLess(gate, success_archive)
        gate_source = source[gate:publish]
        self.assertIn("save_npz_no_overwrite", gate_source)
        self.assertIn("failure_prefix_path.with_name", gate_source)
        self.assertIn(
            "if args.mpc_failure_prefix_output is not None",
            gate_source,
        )
        self.assertIn("evidence_joint_positions_rad", gate_source)
        self.assertIn("first_window_tip_progress_delta_m", gate_source)
        self.assertIn("raise RuntimeError", gate_source)
        self.assertIn(
            "static_bridge_mask_plan | recovery_bridge_mask_plan",
            source,
        )


    def test_bridge_conditions_and_json_keep_all_named_gates(self) -> None:
        strict = DIAGNOSTICS.evaluate_bridge_conditions(
            progress_error_m=np.asarray([0.0, 0.003]),
            progress_limit_m=0.002,
            normal_ok=False,
            tangential_error_m=np.asarray([0.001, 0.003]),
            tangential_limit_m=np.asarray([0.002, 0.002]),
            monotonic_error_m=np.asarray([0.0003]),
            monotonic_limit_m=0.0002,
            palm_error_m=0.004,
            palm_limit_m=0.003,
            collision_ok=False,
            joint_ok=False,
            motion_ok=False,
            budget_ok=False,
        )
        self.assertEqual(
            tuple(strict),
            DIAGNOSTICS.BRIDGE_CONDITION_NAMES,
        )
        self.assertFalse(any(strict.values()))
        recovery = {name: True for name in strict}
        record = DIAGNOSTICS.make_bridge_rejection_record(
            keyframe=8,
            keyframe_count=42,
            distance_m=0.2698,
            fallback="static_bridge",
            strict_conditions=strict,
            recovery_conditions=recovery,
            metrics={
                "progress_margin_m": -0.001,
                "bridge_active_fingers": np.asarray(
                    [True, True, True, False]
                ),
            },
        )
        decoded = json.loads(
            DIAGNOSTICS.format_bridge_rejection_record(record)
        )
        self.assertEqual(decoded["keyframe"], 8)
        self.assertEqual(decoded["fallback"], "static_bridge")
        self.assertEqual(set(decoded["strict"]), set(strict))
        self.assertEqual(
            decoded["metrics"]["bridge_active_fingers"],
            [True, True, True, False],
        )

    def test_palm_guide_multistart_spans_bounded_drift_layers(self) -> None:
        maximum_drift_m = 0.030
        offsets, arm_patterns = (
            DIAGNOSTICS.build_palm_guide_multistart_specs(
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]),
                maximum_drift_m,
            )
        )
        self.assertEqual(offsets.shape, (14, 3))
        self.assertEqual(arm_patterns.shape, (14, 7))
        radii = np.linalg.norm(offsets, axis=1)
        self.assertTrue(np.all(radii < maximum_drift_m))
        self.assertGreater(float(radii.max()), 0.020)
        np.testing.assert_allclose(
            np.unique(np.round(radii / maximum_drift_m, 2)),
            np.asarray([0.20, 0.50, 0.85]),
        )
        self.assertTrue(np.any(np.abs(arm_patterns) > 0.0))
        self.assertTrue(
            np.allclose(arm_patterns[:, [1, 3, 4, 5]], 0.0)
        )

    def test_moving_bridge_always_requires_three_forward_fingers(
        self,
    ) -> None:
        tip_motion_m = np.asarray([0.001, 0.001, 0.001, 0.0])
        active_fingers = np.asarray([True, True, True, False])
        for configured_contact_count in (2, 4):
            with self.subTest(
                min_planner_contact_fingers=configured_contact_count
            ):
                motion_ok, progressing_count = (
                    DIAGNOSTICS.evaluate_moving_bridge_motion(
                        max_joint_motion_rad=0.01,
                        tip_motion_m=tip_motion_m,
                        minimum_tip_motion_m=0.0005,
                        active_fingers=active_fingers,
                    )
                )
                self.assertEqual(
                    DIAGNOSTICS.MOVING_BRIDGE_FORWARD_FINGER_COUNT,
                    3,
                )
                self.assertEqual(progressing_count, 3)
                self.assertTrue(motion_ok)
        two_tip_ok, progressing_count = (
            DIAGNOSTICS.evaluate_moving_bridge_motion(
                max_joint_motion_rad=0.01,
                tip_motion_m=np.asarray([0.001, 0.001, 0.0, 0.0]),
                minimum_tip_motion_m=0.0005,
                active_fingers=np.asarray([True, True, False, False]),
            )
        )
        self.assertEqual(progressing_count, 2)
        self.assertFalse(two_tip_ok)

    def test_moving_bridge_local_targets_advance_without_overshoot(self) -> None:
        forward = DIAGNOSTICS.bounded_incremental_arc_targets(
            current_arc_m=np.asarray([0.10, 0.20, 0.30, 0.40]),
            desired_arc_m=np.asarray([0.10005, 0.25, 0.30, 0.39]),
            direction=1.0,
            interval_m=0.0002,
        )
        np.testing.assert_allclose(
            forward,
            np.asarray([0.10005, 0.2002, 0.30, 0.40]),
        )
        reverse = DIAGNOSTICS.bounded_incremental_arc_targets(
            current_arc_m=np.asarray([0.40, 0.30]),
            desired_arc_m=np.asarray([0.39995, 0.20]),
            direction=-1.0,
            interval_m=0.0002,
        )
        np.testing.assert_allclose(reverse, np.asarray([0.39995, 0.2998]))
        with self.assertRaisesRegex(ValueError, "direction"):
            DIAGNOSTICS.bounded_incremental_arc_targets(
                current_arc_m=np.zeros(1),
                desired_arc_m=np.ones(1),
                direction=0.0,
                interval_m=0.1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            DIAGNOSTICS.bounded_incremental_arc_targets(
                current_arc_m=np.asarray([np.nan]),
                desired_arc_m=np.zeros(1),
                direction=1.0,
                interval_m=0.1,
            )

    def test_progress_aware_targets_enter_strict_inner_band(self) -> None:
        current = np.asarray((0.0, 0.100, 0.101), dtype=np.float64)
        desired = np.asarray((0.0, 0.104816928, 0.104), dtype=np.float64)
        target = DIAGNOSTICS.progress_aware_arc_targets(
            current_arc_m=current,
            desired_arc_m=desired,
            direction=1.0,
            nominal_advance_m=0.00015625,
            hard_progress_limit_m=0.004588452,
            interior_guard_m=0.00005,
        )
        signed_error = desired - target
        self.assertLessEqual(float(signed_error.max()), 0.004538452 + 1e-12)
        self.assertTrue(np.all(target >= current - 1e-12))
        self.assertGreater(target[1] - current[1], 0.00015625)

    def test_progress_aware_targets_support_negative_direction(self) -> None:
        current = np.asarray((0.2, 0.1), dtype=np.float64)
        desired = np.asarray((0.19, 0.09), dtype=np.float64)
        target = DIAGNOSTICS.progress_aware_arc_targets(
            current_arc_m=current,
            desired_arc_m=desired,
            direction=-1.0,
            nominal_advance_m=0.002,
            hard_progress_limit_m=0.004,
            interior_guard_m=0.001,
        )
        self.assertTrue(np.all(target <= current + 1e-12))
        self.assertTrue(np.all(np.abs(target - desired) <= 0.003 + 1e-12))

    def test_strict_suffix_hinge_keeps_task_gates_separate_and_three_move(
        self,
    ) -> None:
        residual = DIAGNOSTICS.strict_suffix_task_hinge_residual(
            progress_error_m=np.asarray((0.9, 0.8, 0.7, 0.6)) * 1.0e-3,
            progress_limit_m=1.0e-3,
            normal_error_m=np.asarray((0.4, 0.5, 0.6, 0.7)) * 1.0e-3,
            normal_tolerance_m=np.full(4, 0.8e-3),
            tangent_error_m=np.asarray((0.1, 0.2, 0.3, 0.4)) * 1.0e-3,
            tangent_tolerance_m=np.full(4, 0.5e-3),
            monotonic_error_m=np.asarray((0.0, 0.1, 0.2, 0.3)) * 1.0e-3,
            monotonic_tolerance_m=0.4e-3,
            tip_motion_m=np.asarray((0.11, 0.10, 0.09, -0.50)) * 1.0e-3,
            minimum_tip_motion_m=0.1e-3,
            interior_guard_m=0.2e-3,
            weight=1000.0,
        )
        self.assertEqual(residual.shape, (19,))
        np.testing.assert_allclose(
            residual[:4],
            np.asarray((0.1, 0.0, 0.0, 0.0)),
            atol=1.0e-12,
        )
        # The worst paused finger is the allowed 1-of-4 exemption.
        np.testing.assert_allclose(residual[-3:], (0.0, 0.0, 0.01))
        with self.assertRaisesRegex(ValueError, "shape"):
            DIAGNOSTICS.strict_suffix_task_hinge_residual(
                progress_error_m=np.zeros(3),
                progress_limit_m=1.0,
                normal_error_m=np.zeros(4),
                normal_tolerance_m=np.ones(4),
                tangent_error_m=np.zeros(4),
                tangent_tolerance_m=np.ones(4),
                monotonic_error_m=np.zeros(4),
                monotonic_tolerance_m=1.0,
                tip_motion_m=np.zeros(4),
                minimum_tip_motion_m=0.0,
                interior_guard_m=0.0,
                weight=1.0,
            )

    def test_suffix_solve_guard_adds_headroom_without_changing_audit(self) -> None:
        required = 0.00005
        solve_guard = DIAGNOSTICS.suffix_optimization_guard(required)
        self.assertAlmostEqual(solve_guard, 0.000075)
        self.assertGreater(solve_guard, required)
        polish_guard = DIAGNOSTICS.suffix_optimization_guard(solve_guard)
        self.assertAlmostEqual(polish_guard, 0.0001125)
        self.assertGreater(polish_guard, solve_guard)
        self.assertAlmostEqual(
            DIAGNOSTICS.suffix_optimization_guard(0.0),
            0.000025,
        )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_optimization_guard(float("nan"))
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_optimization_guard(-1.0e-6)

    def test_suffix_interior_polish_requires_an_exact_safe_prefix(self) -> None:
        conditions = np.ones((5, 11), dtype=bool)
        conditions[4, -1] = False
        kwargs = {
            "node_condition_ok": conditions,
            "node_index": 4,
            "publisher_first_failure_distance_m": np.nan,
            "node_distance_m": 0.0469375,
            "low_motion_ok": True,
        }
        self.assertTrue(
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(**kwargs)
        )

        hard_failed = conditions.copy()
        hard_failed[3, 2] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                **{**kwargs, "node_condition_ok": hard_failed}
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                **{
                    **kwargs,
                    "publisher_first_failure_distance_m": 0.0469375,
                }
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                **{**kwargs, "low_motion_ok": False}
            )
        )
        all_interior = np.ones((5, 11), dtype=bool)
        self.assertFalse(
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                **{**kwargs, "node_condition_ok": all_interior}
            )
        )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                **{**kwargs, "node_condition_ok": np.ones(5, dtype=bool)}
            )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                **{**kwargs, "node_index": 5}
            )

    def test_suffix_interior_polish_scale_ladder_is_deterministic(self) -> None:
        self.assertEqual(
            DIAGNOSTICS.suffix_interior_polish_scale_ladder(4.0),
            (4.0, 8.0, 16.0, 32.0),
        )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_interior_polish_scale_ladder(0.0)
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_interior_polish_scale_ladder(float("inf"))

    def test_suffix_explicit_task_constraints_keep_each_margin_separate(
        self,
    ) -> None:
        margins = DIAGNOSTICS.strict_suffix_task_constraint_margins(
            progress_error_m=np.full(4, 0.0008),
            progress_limit_m=0.001,
            normal_error_m=np.asarray((0.001, 0.002, 0.003, 0.004)),
            normal_tolerance_m=np.asarray((0.002, 0.003, 0.004, 0.005)),
            tangent_error_m=np.full(4, 0.0004),
            tangent_tolerance_m=np.full(4, 0.0008),
            monotonic_error_m=np.full(4, 0.0001),
            monotonic_tolerance_m=0.0004,
            interior_guard_m=0.0001,
        )
        self.assertEqual(margins.shape, (16,))
        np.testing.assert_allclose(margins[:4], 0.0001)
        np.testing.assert_allclose(margins[4:8], 0.0009)
        np.testing.assert_allclose(margins[8:12], 0.0003)
        np.testing.assert_allclose(margins[12:], 0.0002)
        self.assertAlmostEqual(
            DIAGNOSTICS.suffix_explicit_constraint_guard(50.0e-6),
            51.0e-6,
        )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.strict_suffix_task_constraint_margins(
                progress_error_m=np.zeros(3),
                progress_limit_m=0.001,
                normal_error_m=np.zeros(4),
                normal_tolerance_m=np.ones(4),
                tangent_error_m=np.zeros(4),
                tangent_tolerance_m=np.ones(4),
                monotonic_error_m=np.zeros(4),
                monotonic_tolerance_m=0.001,
                interior_guard_m=0.0,
            )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_explicit_constraint_guard(-1.0e-6)

    def test_suffix_explicit_support_sets_are_fixed_and_deterministic(
        self,
    ) -> None:
        motion_indices, contact_indices = (
            DIAGNOSTICS.suffix_explicit_support_indices(
                tip_motion_m=np.asarray((0.0002, 0.0004, 0.0003, 0.00005)),
                minimum_tip_motion_m=0.0001,
                normal_error_m=np.asarray((0.002, 0.001, 0.0025, 0.004)),
                nominal_normal_tolerance_m=0.003,
                required_motion_fingers=3,
                required_contact_fingers=3,
            )
        )
        np.testing.assert_array_equal(motion_indices, (1, 2, 0))
        np.testing.assert_array_equal(contact_indices, (1, 0, 2))

        short_motion, short_contact = (
            DIAGNOSTICS.suffix_explicit_support_indices(
                tip_motion_m=np.asarray((0.0, 0.0, 0.0002, 0.0003)),
                minimum_tip_motion_m=0.0001,
                normal_error_m=np.asarray((0.001, 0.004, 0.004, 0.004)),
                nominal_normal_tolerance_m=0.003,
                required_motion_fingers=3,
                required_contact_fingers=3,
            )
        )
        self.assertEqual(short_motion.size, 2)
        self.assertEqual(short_contact.size, 1)
        repair_motion, repair_contact = (
            DIAGNOSTICS.suffix_explicit_support_indices(
                tip_motion_m=np.asarray((0.0002, 0.0004, 0.0003, 0.00005)),
                minimum_tip_motion_m=0.0001,
                normal_error_m=np.asarray((0.002, 0.001, 0.0025, 0.00301)),
                nominal_normal_tolerance_m=0.003,
                required_motion_fingers=3,
                required_contact_fingers=4,
                include_all_contacts=True,
            )
        )
        np.testing.assert_array_equal(repair_motion, (1, 2, 0))
        np.testing.assert_array_equal(repair_contact, (1, 0, 2, 3))
        # Opt-in contact repair never changes the fixed 3-finger motion set.
        self.assertEqual(repair_motion.size, 3)
        with self.assertRaisesRegex(ValueError, "requires four"):
            DIAGNOSTICS.suffix_explicit_support_indices(
                tip_motion_m=np.ones(4),
                minimum_tip_motion_m=0.0,
                normal_error_m=np.zeros(4),
                nominal_normal_tolerance_m=0.003,
                required_motion_fingers=3,
                required_contact_fingers=3,
                include_all_contacts=True,
            )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_explicit_support_indices(
                tip_motion_m=np.zeros(3),
                minimum_tip_motion_m=0.0,
                normal_error_m=np.zeros(4),
                nominal_normal_tolerance_m=0.003,
                required_motion_fingers=3,
                required_contact_fingers=3,
            )

    def test_terminal_contact_repair_matches_only_the_47_34375mm_shape(
        self,
    ) -> None:
        conditions = np.ones((5, 11), dtype=np.bool_)
        conditions[4, [1, 2, 10]] = False
        metrics_m = np.full((5, 8), 0.001, dtype=np.float64)
        # Lightweight regression values from the selected 47.34375 mm failure
        # candidate's terminal node (all values are exact-audit margins).
        metrics_m[4] = np.asarray(
            (
                49.7753125e-6,
                -1.54329887e-6,
                53.4199999e-6,
                200.0e-6,
                18.4490409e-3,
                14.8720644e-3,
                725.641081e-6,
                96.860199e-6,
            )
        )
        metrics_rad = np.full((5, 2), 0.001, dtype=np.float64)
        metrics_rad[4] = np.asarray((0.000000000076, 0.0196339499))
        contact_count = np.asarray((4, 4, 4, 4, 3), dtype=np.int8)
        kwargs = {
            "node_condition_ok": conditions,
            "node_metric_margin_m": metrics_m,
            "node_metric_margin_rad": metrics_rad,
            "node_contact_count": contact_count,
            "node_index": 4,
            "publisher_first_failure_distance_m": np.nan,
            "node_distance_m": 0.04796875,
            "terminal_start_m": 0.0469375,
            "low_motion_ok": True,
            "task_guard_m": 50.0e-6,
        }
        self.assertTrue(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(**kwargs)
        )
        # The ordinary interior entry remains closed because a hard gate fails.
        self.assertFalse(
            DIAGNOSTICS.suffix_prefix_needs_interior_polish(
                node_condition_ok=conditions,
                node_index=4,
                publisher_first_failure_distance_m=np.nan,
                node_distance_m=0.04796875,
                low_motion_ok=True,
            )
        )

        negative_cases = {
            "preterminal": {"node_distance_m": 0.0469},
            "publisher": {
                "publisher_first_failure_distance_m": 0.04796875
            },
            "low_motion": {"low_motion_ok": False},
            "two_contacts": {
                "node_contact_count": np.asarray((4, 4, 4, 4, 2))
            },
            "four_contacts": {
                "node_contact_count": np.asarray((4, 4, 4, 4, 4))
            },
        }
        for name, replacement in negative_cases.items():
            with self.subTest(name=name):
                self.assertFalse(
                    DIAGNOSTICS.suffix_terminal_contact_repair_required(
                        **{**kwargs, **replacement}
                    )
                )

        prior_failed = conditions.copy()
        prior_failed[3, -1] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_condition_ok": prior_failed}
            )
        )
        wrong_current_failure = conditions.copy()
        wrong_current_failure[4, 3] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_condition_ok": wrong_current_failure}
            )
        )
        normal_too_far = metrics_m.copy()
        normal_too_far[4, 1] = -50.01e-6
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_metric_margin_m": normal_too_far}
            )
        )
        normal_at_limit = metrics_m.copy()
        normal_at_limit[4, 1] = -50.0e-6
        self.assertTrue(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_metric_margin_m": normal_at_limit}
            )
        )
        normal_not_failed = metrics_m.copy()
        normal_not_failed[4, 1] = 1.0e-9
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_metric_margin_m": normal_not_failed}
            )
        )
        inconsistent_task_gate = metrics_m.copy()
        inconsistent_task_gate[4, 2] = -1.0e-9
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_metric_margin_m": inconsistent_task_gate}
            )
        )
        for metric_index in range(5, 8):
            clearance_failed = metrics_m.copy()
            clearance_failed[4, metric_index] = 49.99e-6
            self.assertFalse(
                DIAGNOSTICS.suffix_terminal_contact_repair_required(
                    **{**kwargs, "node_metric_margin_m": clearance_failed}
                )
            )
        palm_failed = metrics_m.copy()
        palm_failed[4, 4] = -1.0e-9
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_metric_margin_m": palm_failed}
            )
        )
        joint_failed = metrics_rad.copy()
        joint_failed[4, 0] = -2.0e-12
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_required(
                **{**kwargs, "node_metric_margin_rad": joint_failed}
            )
        )

        restart_kwargs = {
            **kwargs,
            "q_rad": np.zeros(23),
            "expected_dof": 23,
            "explicit_prefix_ok": False,
        }
        self.assertTrue(
            DIAGNOSTICS.suffix_terminal_contact_repair_restart_required(
                **restart_kwargs
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_restart_required(
                **{**restart_kwargs, "explicit_prefix_ok": True}
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_restart_required(
                **{**restart_kwargs, "q_rad": np.full(23, np.nan)}
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_terminal_contact_repair_restart_required(
                **{
                    **restart_kwargs,
                    "node_condition_ok": wrong_current_failure,
                }
            )
        )

    def test_suffix_explicit_task_polish_requires_only_current_task_miss(
        self,
    ) -> None:
        conditions = np.ones((5, 11), dtype=bool)
        conditions[4, -1] = False
        metrics = np.full((5, 8), 0.001, dtype=np.float64)
        metrics[4, 1] = 37.83e-6
        kwargs = {
            "node_condition_ok": conditions,
            "node_metric_margin_m": metrics,
            "node_index": 4,
            "task_guard_m": 50.0e-6,
        }
        self.assertTrue(
            DIAGNOSTICS.suffix_node_needs_explicit_task_polish(**kwargs)
        )

        prior_interior_failed = conditions.copy()
        prior_interior_failed[3, -1] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_node_needs_explicit_task_polish(
                **{
                    **kwargs,
                    "node_condition_ok": prior_interior_failed,
                }
            )
        )
        current_hard_failed = conditions.copy()
        current_hard_failed[4, 2] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_node_needs_explicit_task_polish(
                **{**kwargs, "node_condition_ok": current_hard_failed}
            )
        )
        task_safe_metrics = metrics.copy()
        task_safe_metrics[4, :4] = 0.001
        self.assertFalse(
            DIAGNOSTICS.suffix_node_needs_explicit_task_polish(
                **{**kwargs, "node_metric_margin_m": task_safe_metrics}
            )
        )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_node_needs_explicit_task_polish(
                **{**kwargs, "node_metric_margin_m": np.zeros((4, 8))}
            )

    def test_suffix_explicit_restart_requires_one_finite_exact_task_miss(
        self,
    ) -> None:
        conditions = np.ones((2, 11), dtype=bool)
        conditions[1, -1] = False
        metrics = np.full((2, 8), 0.001, dtype=np.float64)
        metrics[1, 1] = 48.394703e-6
        metrics[1, 5:8] = 50.0e-6
        kwargs = {
            "q_rad": np.zeros(23, dtype=np.float64),
            "expected_dof": 23,
            "explicit_prefix_ok": False,
            "node_condition_ok": conditions,
            "node_metric_margin_m": metrics,
            "node_index": 1,
            "publisher_first_failure_distance_m": np.nan,
            "node_distance_m": 0.04703125,
            "low_motion_ok": True,
            "task_guard_m": 50.0e-6,
        }
        self.assertTrue(
            DIAGNOSTICS.suffix_explicit_restart_required(**kwargs)
        )
        palm_below_guard = metrics.copy()
        palm_below_guard[1, 4] = 40.0e-6
        self.assertTrue(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "node_metric_margin_m": palm_below_guard}
            )
        )
        for collision_index, collision_name in enumerate(
            ("arm", "hand", "tip"), start=5
        ):
            with self.subTest(collision_margin=collision_name):
                task_and_collision_miss = metrics.copy()
                task_and_collision_miss[1, collision_index] = 40.0e-6
                self.assertFalse(
                    DIAGNOSTICS.suffix_explicit_restart_required(
                        **{
                            **kwargs,
                            "node_metric_margin_m": task_and_collision_miss,
                        }
                    )
                )
        with self.assertRaisesRegex(ValueError, "collision margins"):
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "node_metric_margin_m": metrics[:, :7]}
            )

        # Exact audit is authoritative: a solver status such as SLSQP 9 is
        # deliberately not an input, and an exact pass must not restart.
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "explicit_prefix_ok": True}
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "q_rad": np.full(23, np.nan)}
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "q_rad": np.zeros(22)}
            )
        )

        prior_hard_failed = conditions.copy()
        prior_hard_failed[0, 2] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "node_condition_ok": prior_hard_failed}
            )
        )
        current_hard_failed = conditions.copy()
        current_hard_failed[1, 2] = False
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "node_condition_ok": current_hard_failed}
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{
                    **kwargs,
                    "publisher_first_failure_distance_m": 0.04703125,
                }
            )
        )
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "low_motion_ok": False}
            )
        )
        non_task_miss = metrics.copy()
        non_task_miss[1, :4] = 0.001
        non_task_miss[1, 4] = -1.0e-6
        self.assertFalse(
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "node_metric_margin_m": non_task_miss}
            )
        )
        with self.assertRaises(ValueError):
            DIAGNOSTICS.suffix_explicit_restart_required(
                **{**kwargs, "expected_dof": 0}
            )

    def test_suffix_rollout_prefix_rank_preserves_only_exact_prefixes(
        self,
    ) -> None:
        conditions = np.ones((3, 11), dtype=bool)
        metric_m = np.full((3, 8), 0.001, dtype=np.float64)
        metric_rad = np.full((3, 2), 0.002, dtype=np.float64)
        pad_margin = np.full(3, 0.05, dtype=np.float64)
        passed, rank = DIAGNOSTICS.suffix_rollout_prefix_rank(
            node_condition_ok=conditions,
            node_metric_margin_m=metric_m,
            node_metric_margin_rad=metric_rad,
            node_pad_alignment_margin=pad_margin,
            node_index=0,
            publisher_first_failure_distance_m=0.2,
            node_distance_m=0.1,
            low_motion_ok=True,
        )
        self.assertTrue(passed)
        self.assertEqual(rank[:2], (0.0, 0.0))

        # Future node failure is irrelevant to a certified node-0 prefix.
        conditions[2, 0] = False
        future_failed, _ = DIAGNOSTICS.suffix_rollout_prefix_rank(
            node_condition_ok=conditions,
            node_metric_margin_m=metric_m,
            node_metric_margin_rad=metric_rad,
            node_pad_alignment_margin=pad_margin,
            node_index=0,
            publisher_first_failure_distance_m=np.nan,
            node_distance_m=0.1,
            low_motion_ok=True,
        )
        self.assertTrue(future_failed)

        # A publisher failure exactly at the current node is part of the
        # prefix, and a failed node condition cannot be hidden by margins.
        conditions[0, 8] = False
        failed, failed_rank = DIAGNOSTICS.suffix_rollout_prefix_rank(
            node_condition_ok=conditions,
            node_metric_margin_m=metric_m,
            node_metric_margin_rad=metric_rad,
            node_pad_alignment_margin=pad_margin,
            node_index=0,
            publisher_first_failure_distance_m=0.1,
            node_distance_m=0.1,
            low_motion_ok=True,
        )
        self.assertFalse(failed)
        self.assertEqual(failed_rank[0], 1.0)
        self.assertEqual(failed_rank[1], 2.0)

        conditions[0, 8] = True
        low_motion_failed, low_motion_rank = (
            DIAGNOSTICS.suffix_rollout_prefix_rank(
                node_condition_ok=conditions,
                node_metric_margin_m=metric_m,
                node_metric_margin_rad=metric_rad,
                node_pad_alignment_margin=pad_margin,
                node_index=0,
                publisher_first_failure_distance_m=np.nan,
                node_distance_m=0.1,
                low_motion_ok=False,
            )
        )
        self.assertFalse(low_motion_failed)
        self.assertEqual(low_motion_rank[1], 1.0)

    def test_terminal_start_matches_published_last_fifty_frames(self) -> None:
        start = DIAGNOSTICS.terminal_contact_start_distance(0.05, 800, 50)
        self.assertAlmostEqual(start, 0.0469375, places=12)
        route = np.linspace(0.0, 0.05, 801, dtype=np.float64)[1:]
        mask = DIAGNOSTICS.terminal_contact_sample_mask(
            route,
            terminal_start_m=start,
        )
        self.assertEqual(int(np.flatnonzero(mask)[0]), 750)
        self.assertFalse(bool(mask[749]))
        self.assertTrue(bool(mask[750]))
        self.assertEqual(int(np.count_nonzero(mask)), 50)

    def test_horizon_grid_inserts_exact_terminal_sentinel(self) -> None:
        distances = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.04609375,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        self.assertEqual(distances.shape, (5,))
        self.assertAlmostEqual(float(distances[0]), 0.04609375, places=12)
        self.assertAlmostEqual(float(distances[-1]), 0.0469375, places=12)
        self.assertTrue(np.all(np.diff(distances) > 0.0))

    def test_horizon_grid_previews_terminal_from_first_short_bridge(self) -> None:
        distances = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.04515625,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        self.assertEqual(distances.shape, (5,))
        self.assertAlmostEqual(float(distances[0]), 0.04515625, places=12)
        self.assertAlmostEqual(float(distances[-1]), 0.0469375, places=12)
        self.assertTrue(np.all(np.diff(distances) > 0.0))

    def test_damped_nullspace_projection_is_generic_and_task_preserving(
        self,
    ) -> None:
        jacobian = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
        projected = DIAGNOSTICS.damped_task_nullspace_directions(
            jacobian,
            np.asarray(((1.0, 1.0, 1.0), (0.0, 0.0, 2.0))),
            damping=1.0e-12,
        )
        self.assertEqual(projected.shape, (2, 3))
        np.testing.assert_allclose(
            np.max(np.abs(projected), axis=1),
            np.ones(2),
            atol=1.0e-12,
        )
        self.assertLess(
            float(np.max(np.abs(jacobian @ projected.T))),
            1.0e-9,
        )
        self.assertTrue(np.all(projected[:, 2] > 0.99))
        with self.assertRaisesRegex(ValueError, "one column per joint"):
            DIAGNOSTICS.damped_task_nullspace_directions(
                jacobian,
                np.ones((1, 2)),
            )

    def test_horizon_grid_inserts_route_endpoint_when_in_reach(self) -> None:
        distances = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.0494,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        self.assertAlmostEqual(float(distances[-1]), 0.05, places=12)
        self.assertLessEqual(
            float(np.max(np.diff(distances))),
            0.00015625 + 1.0e-12,
        )

    def test_horizon_grid_stays_local_after_terminal_boundary(self) -> None:
        distances = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.04703125,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        np.testing.assert_allclose(
            distances,
            np.asarray(
                (0.04703125, 0.0471875, 0.04734375, 0.0475, 0.04765625),
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertLessEqual(
            float(np.max(np.diff(distances))),
            0.00015625 + 1.0e-12,
        )
        self.assertLess(float(distances[-1]), 0.05)

    def test_horizon_grid_endpoint_local_reach_boundary(self) -> None:
        just_outside = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.049374,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        self.assertLess(float(just_outside[-1]), 0.05)
        self.assertLessEqual(
            float(np.max(np.diff(just_outside))),
            0.00015625 + 1.0e-12,
        )
        exact_reach = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.049375,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        self.assertAlmostEqual(float(exact_reach[-1]), 0.05, places=12)
        one_node_before_endpoint = (
            DIAGNOSTICS.build_receding_horizon_distances(
                first_distance_m=0.049,
                nominal_step_m=0.00015625,
                horizon_nodes=1,
                route_end_m=0.05,
                terminal_start_m=0.0469375,
            )
        )
        np.testing.assert_array_equal(
            one_node_before_endpoint, np.asarray([0.049])
        )
        endpoint_only = DIAGNOSTICS.build_receding_horizon_distances(
            first_distance_m=0.05,
            nominal_step_m=0.00015625,
            horizon_nodes=5,
            route_end_m=0.05,
            terminal_start_m=0.0469375,
        )
        np.testing.assert_array_equal(endpoint_only, np.asarray([0.05]))

    def test_smoothstep_interpolation_matches_publisher_off_midpoint(self) -> None:
        distance = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
        q = np.asarray(((0.0,), (2.0,), (4.0,)), dtype=np.float64)
        sample = np.asarray(
            (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
            dtype=np.float64,
        )
        interpolated = DIAGNOSTICS.smoothstep_joint_interpolation(
            distance,
            q,
            sample,
        )
        np.testing.assert_allclose(
            interpolated[:, 0],
            (0.0, 0.3125, 1.0, 2.0, 3.0, 4.0),
        )
        self.assertAlmostEqual(float(interpolated[1, 0] / 2.0), 0.15625)

    def test_horizon_joint_residuals_cover_all_nodes_and_steps(self) -> None:
        q = np.asarray(((0.2, 0.8), (0.25, 0.75)), dtype=np.float64)
        lower = np.zeros(2, dtype=np.float64)
        upper = np.ones(2, dtype=np.float64)
        margin = DIAGNOSTICS.horizon_joint_margin_residual(
            q,
            lower,
            upper,
            minimum_margin_rad=0.21,
            weight=10.0,
        )
        self.assertEqual(margin.shape, (4,))
        self.assertAlmostEqual(float(margin[0]), 0.1)
        step = DIAGNOSTICS.horizon_joint_step_residual(
            q,
            np.asarray((0.19, 0.81), dtype=np.float64),
            maximum_step_rad=0.06,
            interior_guard_rad=0.01,
            weight=100.0,
        )
        self.assertEqual(step.shape, (4,))
        self.assertAlmostEqual(float(step.max()), 0.0)

    def test_moving_bridge_local_residual_is_tip_only_and_wraps_azimuth(
        self,
    ) -> None:
        epsilon = 1.0e-4
        residual = DIAGNOSTICS.moving_bridge_local_residual(
            arc_m=np.asarray([1.0, 2.0, 3.0, 4.0]),
            target_arc_m=np.asarray([0.9, 2.0, 3.1, 4.0]),
            standoff_m=np.asarray([0.01, 0.02, 0.03, 0.04]),
            anchor_standoff_m=np.asarray([0.01, 0.01, 0.03, 0.05]),
            azimuth_rad=np.asarray(
                [np.pi - epsilon, 0.1, -0.2, -np.pi + epsilon]
            ),
            anchor_azimuth_rad=np.asarray(
                [-np.pi + epsilon, 0.0, -0.1, np.pi - epsilon]
            ),
            q_rad=np.ones(23),
            anchor_q_rad=np.zeros(23),
            capsule_radius_m=0.1,
            task_weight=3200.0,
        )
        self.assertEqual(residual.shape, (12 + 23,))
        np.testing.assert_allclose(
            residual[:4],
            3200.0 * np.asarray([0.1, 0.0, -0.1, 0.0]),
        )
        np.testing.assert_allclose(
            residual[4:8],
            3200.0 * np.asarray([0.0, 0.01, 0.0, -0.01]),
        )
        np.testing.assert_allclose(
            residual[8:12],
            3200.0
            * 0.1
            * np.asarray([-2.0 * epsilon, 0.1, -0.1, 2.0 * epsilon]),
            atol=1.0e-10,
        )
        np.testing.assert_allclose(residual[12:], 1.0e-4)
        with self.assertRaisesRegex(ValueError, "shape"):
            DIAGNOSTICS.moving_bridge_local_residual(
                arc_m=np.zeros(3),
                target_arc_m=np.zeros(4),
                standoff_m=np.zeros(4),
                anchor_standoff_m=np.zeros(4),
                azimuth_rad=np.zeros(4),
                anchor_azimuth_rad=np.zeros(4),
                q_rad=np.zeros(23),
                anchor_q_rad=np.zeros(23),
                capsule_radius_m=0.1,
                task_weight=3200.0,
            )

    def test_moving_bridge_trust_radius_never_exceeds_plan_step(self) -> None:
        self.assertEqual(
            DIAGNOSTICS.bounded_moving_bridge_trust_radius(0.05, 0.03),
            0.03,
        )
        self.assertEqual(
            DIAGNOSTICS.bounded_moving_bridge_trust_radius(0.02, 0.03),
            0.02,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            DIAGNOSTICS.bounded_moving_bridge_trust_radius(0.0, 0.03)
        with self.assertRaisesRegex(ValueError, "positive"):
            DIAGNOSTICS.bounded_moving_bridge_trust_radius(np.nan, 0.03)

    def test_seed42_380mm_bridge_regression_has_three_forward_targets(
        self,
    ) -> None:
        # Exact fingertip arcs from the 380.103896 mm failure prefix.  The
        # previous implementation returned an approximately zero-motion
        # least-squares state even though fingers 1/2/3 each had a feasible
        # 0.155844 mm local increment.
        anchor_arc = np.asarray(
            [
                0.4835175065198217,
                0.47134945670750483,
                0.47752682248738154,
                0.5058588851551499,
            ]
        )
        hard_desired_arc = np.asarray(
            [
                0.48343811,
                0.47628215,
                0.48273124,
                0.50985539,
            ]
        )
        interval_m = 0.00015584415584413147
        target = DIAGNOSTICS.bounded_incremental_arc_targets(
            current_arc_m=anchor_arc,
            desired_arc_m=hard_desired_arc,
            direction=1.0,
            interval_m=interval_m,
        )
        forward = target - anchor_arc
        np.testing.assert_allclose(forward[0], 0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            forward[1:],
            interval_m,
            atol=1.0e-15,
        )
        self.assertEqual(int(np.count_nonzero(forward >= 0.1 * interval_m)), 3)
        self.assertTrue(np.all(target[1:] <= hard_desired_arc[1:]))

    def test_failure_prefix_never_overwrites_and_loads_without_pickle(
        self,
    ) -> None:
        conditions = {
            name: name != "progress"
            for name in DIAGNOSTICS.BRIDGE_CONDITION_NAMES
        }
        bridge_record = DIAGNOSTICS.make_bridge_rejection_record(
            keyframe=2,
            keyframe_count=5,
            distance_m=0.2698,
            fallback="planner_failure",
            strict_conditions=conditions,
            recovery_conditions=conditions,
            metrics={"progress_margin_m": -0.001},
        )
        restart_source_q = np.full((2, 5, 23), np.nan, dtype=np.float64)
        restart_source_q[0, 4] = 0.25
        primary_eps = np.full((2, 5), np.nan, dtype=np.float64)
        primary_eps[0, 4] = 1.0e-5
        restart_q = np.full((2, 5, 23), np.nan, dtype=np.float64)
        restart_q[0, 4] = 0.251
        restart_eps = np.full((2, 5), np.nan, dtype=np.float64)
        restart_eps[0, 4] = 1.0e-6
        restart_objective_anchor_q = np.full(
            (2, 5, 23), np.nan, dtype=np.float64
        )
        restart_objective_anchor_q[0, 4] = 0.2505
        restart_attempt_count = np.zeros((2, 5), dtype=np.int16)
        restart_attempt_count[0, 4] = 1
        restart_status = np.full((2, 5), -999, dtype=np.int16)
        restart_status[0, 4] = 9
        restart_solver_success = np.zeros((2, 5), dtype=np.bool_)
        restart_prefix_ok = np.zeros((2, 5), dtype=np.bool_)
        restart_prefix_ok[0, 4] = True
        restart_nfev = np.zeros((2, 5), dtype=np.int32)
        restart_nfev[0, 4] = 3316
        restart_constraint_margin = np.full(
            (2, 5), np.nan, dtype=np.float64
        )
        restart_constraint_margin[0, 4] = 0.067217e-6
        repair_mode = np.zeros((2, 5), dtype=np.bool_)
        repair_mode[0, 4] = True
        repair_source_q = np.full((2, 5, 23), np.nan, dtype=np.float64)
        repair_source_q[0, 4] = 0.249
        repair_source_contact_count = np.full((2, 5), -1, dtype=np.int8)
        repair_source_contact_count[0, 4] = 3
        repair_source_normal_error = np.full(
            (2, 5, 4), np.nan, dtype=np.float64
        )
        repair_source_normal_error[0, 4] = np.asarray(
            (0.0029, 0.0028, 0.003001543, 0.0027)
        )
        repair_motion_support = np.full((2, 5, 3), -1, dtype=np.int8)
        repair_motion_support[0, 4] = np.asarray((3, 1, 0), dtype=np.int8)
        repair_contact_support = np.full((2, 5, 4), -1, dtype=np.int8)
        repair_contact_support[0, 4] = np.asarray(
            (3, 1, 0, 2), dtype=np.int8
        )
        repair_bound_headroom = np.full((2, 5), np.nan, dtype=np.float64)
        repair_bound_headroom[0, 4] = 1.0e-6
        repair_objective_anchor_q = np.full(
            (2, 5, 23), np.nan, dtype=np.float64
        )
        repair_objective_anchor_q[0, 4] = 0.2495
        coarse_auto_rephase_all = np.asarray(
            (
                (0.0, 0.0, 0.0, 0.0),
                (-3.0e-5, -2.0e-5, 0.0, -1.0e-5),
                (999.0, 999.0, 999.0, 999.0),
            ),
            dtype=np.float64,
        )
        coarse_target_progress_all = np.vstack(
            (
                np.arange(10, dtype=np.float64).reshape(2, 5),
                np.full((1, 5), 999.0),
            )
        )
        committed_rows = 2
        coarse_auto_rephase = coarse_auto_rephase_all[:committed_rows]
        coarse_target_progress = coarse_target_progress_all[:committed_rows]
        coarse_provenance = {
            "auto_rephase_offset_m": coarse_auto_rephase,
            "progress_m": coarse_target_progress,
            "target_progress_m": coarse_target_progress,
            "feasibility_bridge": np.asarray(
                (False, True, True), dtype=np.bool_
            )[:committed_rows],
            "suffix_horizon": np.asarray(
                (False, True, False), dtype=np.bool_
            )[:committed_rows],
            "static_feasibility_bridge": np.asarray(
                (False, False, True), dtype=np.bool_
            )[:committed_rows],
            "static_bridge_dwell_m": np.asarray((0.0, 0.0001, 999.0))[
                :committed_rows
            ],
            "recovery_bridge": np.asarray(
                (False, True, True), dtype=np.bool_
            )[:committed_rows],
            "recovery_bridge_dwell_m": np.asarray((0.0, 0.0002, 999.0))[
                :committed_rows
            ],
            "normal_error_m": np.vstack(
                (np.zeros((2, 5)), np.full((1, 5), 999.0))
            )[:committed_rows],
            "palm_target_m": np.vstack(
                (np.zeros((2, 3)), np.full((1, 3), 999.0))
            )[:committed_rows],
            "palm_position_error_m": np.asarray((0.0, 0.001, 999.0))[
                :committed_rows
            ],
            "cost": np.asarray((1.0, 2.0, 999.0))[:committed_rows],
            "nfev": np.asarray((10, 20, 999), dtype=np.int32)[
                :committed_rows
            ],
        }
        output = Path("failure.npz")
        values = {
            "reason": "longitudinal_progress",
            "keyframe": 2,
            "keyframe_count": 5,
            "failure_distance_m": 0.2698,
            "last_feasible_distance_m": np.asarray([0.0, 0.25]),
            "last_feasible_q_rad": np.zeros((2, 23)),
            "last_feasible_points_m": np.zeros((2, 5, 3)),
            "last_feasible_arcs_m": np.zeros((2, 5)),
            "final_best_desired_arcs_m": np.ones(5),
            "final_best_q_rad": np.ones(23),
            "final_best_points_m": np.ones((5, 3)),
            "final_best_arcs_m": np.ones(5),
            "rephase_offset_m": np.zeros(4),
            "budget_values": {
                "recovery_remaining_m": 0.001,
                "feasibility_bridge_tip_target_scale": 0.5,
                "last_suffix_horizon_candidate_rollout_explicit_polish_source_q_rad": (
                    restart_source_q
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_eps": (
                    primary_eps
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_attempt_count": (
                    restart_attempt_count
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_status": (
                    restart_status
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_solver_success": (
                    restart_solver_success
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_prefix_ok": (
                    restart_prefix_ok
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_nfev": (
                    restart_nfev
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_constraint_margin_m": (
                    restart_constraint_margin
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_q_rad": (
                    restart_q
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_eps": (
                    restart_eps
                ),
                "last_suffix_horizon_candidate_rollout_explicit_polish_restart_objective_anchor_q_rad": (
                    restart_objective_anchor_q
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_mode": (
                    repair_mode
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_source_q_rad": (
                    repair_source_q
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_source_contact_count": (
                    repair_source_contact_count
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_source_normal_error_m": (
                    repair_source_normal_error
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_motion_support_indices": (
                    repair_motion_support
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_contact_support_indices": (
                    repair_contact_support
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_bound_headroom_rad": (
                    repair_bound_headroom
                ),
                "last_suffix_horizon_candidate_rollout_terminal_contact_repair_objective_anchor_q_rad": (
                    repair_objective_anchor_q
                ),
            },
            "failure_metrics": {
                "progress_error_m": np.asarray([0.0, 0.006])
            },
            "coarse_provenance": coarse_provenance,
            "refinement_provenance": {
                "inserted_distance_m": np.asarray((0.125,), dtype=np.float64),
                "inserted_reason": np.asarray(("fingertip_support",)),
            },
            "rolling_provenance": {
                "frame_target_distance_m": np.linspace(0.01, 0.25, 25),
                "window_frames": np.asarray(20, dtype=np.int32),
                "forward_progress_ratio": np.asarray(0.1, dtype=np.float64),
                "required_forward_fingers": np.asarray(3, dtype=np.int8),
            },
            "bridge_record": bridge_record,
            "rejected_moving_bridge": (
                DIAGNOSTICS.RejectedMovingBridgeCandidate(
                    q_rad=np.full(23, 2.0),
                    points_m=np.full((5, 3), 3.0),
                    arcs_m=np.full(5, 4.0),
                    desired_arcs_m=np.full(5, 5.0),
                )
            ),
        }
        saved_payloads: list[dict[str, np.ndarray]] = []

        def capture_payload(
            _output_file: object,
            **payload: np.ndarray,
        ) -> None:
            saved_payloads.append(payload)

        first_context = mock.MagicMock()
        first_context.__enter__.return_value = io.BytesIO()
        with (
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(Path, "open", return_value=first_context),
            mock.patch.object(
                DIAGNOSTICS.np,
                "savez_compressed",
                side_effect=capture_payload,
            ),
        ):
            first = DIAGNOSTICS.save_mpc_failure_prefix(
                output,
                **values,
            )
        second_context = mock.MagicMock()
        second_context.__enter__.return_value = io.BytesIO()
        with (
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(
                Path,
                "open",
                side_effect=(FileExistsError, second_context),
            ),
            mock.patch.object(
                DIAGNOSTICS.np,
                "savez_compressed",
                side_effect=capture_payload,
            ),
        ):
            second = DIAGNOSTICS.save_mpc_failure_prefix(
                output,
                **values,
            )
        self.assertEqual(first, output)
        self.assertEqual(second, output.with_name("failure_001.npz"))
        self.assertEqual(len(saved_payloads), 2)
        self.assertTrue(
            all(array.dtype != object for array in saved_payloads[0].values())
        )
        archive = io.BytesIO()
        np.savez_compressed(archive, **saved_payloads[0])
        archive.seek(0)
        with np.load(archive, allow_pickle=False) as saved:
            self.assertEqual(
                str(saved["reason"]),
                "longitudinal_progress",
            )
            self.assertEqual(
                saved["last_feasible_coarse_q_rad"].shape,
                (2, 23),
            )
            self.assertEqual(
                int(saved["schema_version"]),
                3,
            )
            self.assertEqual(
                int(saved["committed_coarse_row_count"]),
                committed_rows,
            )
            self.assertEqual(
                saved["committed_coarse_row_count"].dtype,
                np.dtype(np.int32),
            )
            self.assertTrue(
                bool(saved["last_feasible_coarse_provenance_available"])
            )
            expected_coarse_arrays = {
                "auto_rephase_offset_m": (np.dtype(np.float64), (2, 4)),
                "progress_m": (np.dtype(np.float64), (2, 5)),
                "target_progress_m": (np.dtype(np.float64), (2, 5)),
                "feasibility_bridge": (np.dtype(np.bool_), (2,)),
                "suffix_horizon": (np.dtype(np.bool_), (2,)),
                "static_feasibility_bridge": (np.dtype(np.bool_), (2,)),
                "static_bridge_dwell_m": (np.dtype(np.float64), (2,)),
                "recovery_bridge": (np.dtype(np.bool_), (2,)),
                "recovery_bridge_dwell_m": (np.dtype(np.float64), (2,)),
                "normal_error_m": (np.dtype(np.float64), (2, 5)),
                "palm_target_m": (np.dtype(np.float64), (2, 3)),
                "palm_position_error_m": (np.dtype(np.float64), (2,)),
                "cost": (np.dtype(np.float64), (2,)),
                "nfev": (np.dtype(np.int32), (2,)),
            }
            for suffix, (expected_dtype, expected_shape) in (
                expected_coarse_arrays.items()
            ):
                array = saved[f"last_feasible_coarse_{suffix}"]
                self.assertEqual(array.dtype, expected_dtype)
                self.assertEqual(array.shape, expected_shape)
                self.assertEqual(
                    array.shape[0],
                    saved["last_feasible_distance_m"].size,
                )
                self.assertNotEqual(array.dtype, np.dtype(object))
                if array.dtype == np.dtype(np.float64):
                    self.assertFalse(np.any(array == 999.0))
            np.testing.assert_allclose(
                saved["last_feasible_coarse_auto_rephase_offset_m"],
                coarse_auto_rephase,
            )
            np.testing.assert_allclose(
                saved["last_feasible_coarse_target_progress_m"],
                coarse_target_progress,
            )
            np.testing.assert_array_equal(
                saved["last_feasible_coarse_nfev"],
                (10, 20),
            )
            self.assertTrue(bool(saved["refinement_provenance_available"]))
            self.assertEqual(
                saved["auto_refine_inserted_distance_m"].dtype,
                np.dtype(np.float64),
            )
            self.assertEqual(saved["auto_refine_inserted_reason"].dtype.kind, "U")
            self.assertEqual(saved["auto_refine_inserted_reason"].shape, (1,))
            self.assertTrue(bool(saved["rolling_provenance_available"]))
            self.assertEqual(
                saved["rolling_frame_target_distance_m"].shape,
                (25,),
            )
            self.assertEqual(
                saved["rolling_frame_target_distance_m"].dtype,
                np.dtype(np.float64),
            )
            self.assertEqual(
                saved["rolling_window_frames"].dtype,
                np.dtype(np.int32),
            )
            self.assertEqual(int(saved["rolling_window_frames"]), 20)
            self.assertEqual(
                saved["rolling_forward_progress_ratio"].dtype,
                np.dtype(np.float64),
            )
            self.assertEqual(
                float(saved["rolling_forward_progress_ratio"]),
                0.1,
            )
            self.assertEqual(
                saved["rolling_required_forward_fingers"].dtype,
                np.dtype(np.int8),
            )
            self.assertEqual(
                int(saved["rolling_required_forward_fingers"]),
                3,
            )
            restart_prefix = (
                "budget_last_suffix_horizon_candidate_rollout_"
                "explicit_polish_restart_"
            )
            source_q_array = saved[
                "budget_last_suffix_horizon_candidate_rollout_"
                "explicit_polish_source_q_rad"
            ]
            self.assertEqual(source_q_array.shape, (2, 5, 23))
            self.assertEqual(source_q_array.dtype, np.dtype(np.float64))
            self.assertTrue(np.isnan(source_q_array[1, 0]).all())
            primary_eps_array = saved[
                "budget_last_suffix_horizon_candidate_rollout_"
                "explicit_polish_eps"
            ]
            self.assertEqual(primary_eps_array.shape, (2, 5))
            self.assertEqual(primary_eps_array.dtype, np.dtype(np.float64))
            self.assertEqual(float(primary_eps_array[0, 4]), 1.0e-5)
            self.assertTrue(np.isnan(primary_eps_array[1, 0]))
            expected_restart_arrays = {
                "attempt_count": (np.dtype(np.int16), (2, 5)),
                "status": (np.dtype(np.int16), (2, 5)),
                "solver_success": (np.dtype(np.bool_), (2, 5)),
                "prefix_ok": (np.dtype(np.bool_), (2, 5)),
                "nfev": (np.dtype(np.int32), (2, 5)),
                "constraint_margin_m": (np.dtype(np.float64), (2, 5)),
                "q_rad": (np.dtype(np.float64), (2, 5, 23)),
                "eps": (np.dtype(np.float64), (2, 5)),
                "objective_anchor_q_rad": (
                    np.dtype(np.float64),
                    (2, 5, 23),
                ),
            }
            for suffix, (expected_dtype, expected_shape) in (
                expected_restart_arrays.items()
            ):
                array = saved[f"{restart_prefix}{suffix}"]
                self.assertEqual(array.dtype, expected_dtype)
                self.assertEqual(array.shape, expected_shape)
                self.assertNotEqual(array.dtype, np.dtype(object))
            self.assertEqual(
                int(saved[f"{restart_prefix}attempt_count"][0, 4]),
                1,
            )
            self.assertEqual(
                int(saved[f"{restart_prefix}status"][0, 4]),
                9,
            )
            self.assertFalse(
                bool(saved[f"{restart_prefix}solver_success"][0, 4])
            )
            self.assertTrue(
                bool(saved[f"{restart_prefix}prefix_ok"][0, 4])
            )
            self.assertEqual(
                int(saved[f"{restart_prefix}status"][1, 0]),
                -999,
            )
            self.assertTrue(
                np.isnan(saved[f"{restart_prefix}q_rad"][1, 0]).all()
            )
            self.assertEqual(
                float(saved[f"{restart_prefix}eps"][0, 4]),
                1.0e-6,
            )
            self.assertTrue(
                np.isnan(saved[f"{restart_prefix}eps"][1, 0])
            )
            np.testing.assert_allclose(
                saved[f"{restart_prefix}objective_anchor_q_rad"][0, 4],
                0.2505,
            )
            self.assertTrue(
                np.isnan(
                    saved[f"{restart_prefix}objective_anchor_q_rad"][1, 0]
                ).all()
            )
            repair_prefix = (
                "budget_last_suffix_horizon_candidate_rollout_"
                "terminal_contact_repair_"
            )
            expected_repair_arrays = {
                "mode": (np.dtype(np.bool_), (2, 5)),
                "source_q_rad": (np.dtype(np.float64), (2, 5, 23)),
                "source_contact_count": (np.dtype(np.int8), (2, 5)),
                "source_normal_error_m": (
                    np.dtype(np.float64),
                    (2, 5, 4),
                ),
                "motion_support_indices": (np.dtype(np.int8), (2, 5, 3)),
                "contact_support_indices": (np.dtype(np.int8), (2, 5, 4)),
                "bound_headroom_rad": (np.dtype(np.float64), (2, 5)),
                "objective_anchor_q_rad": (
                    np.dtype(np.float64),
                    (2, 5, 23),
                ),
            }
            for suffix, (expected_dtype, expected_shape) in (
                expected_repair_arrays.items()
            ):
                array = saved[f"{repair_prefix}{suffix}"]
                self.assertEqual(array.dtype, expected_dtype)
                self.assertEqual(array.shape, expected_shape)
                self.assertNotEqual(array.dtype, np.dtype(object))
            self.assertTrue(bool(saved[f"{repair_prefix}mode"][0, 4]))
            self.assertFalse(bool(saved[f"{repair_prefix}mode"][1, 0]))
            self.assertEqual(
                int(saved[f"{repair_prefix}source_contact_count"][0, 4]),
                3,
            )
            np.testing.assert_array_equal(
                saved[f"{repair_prefix}motion_support_indices"][0, 4],
                (3, 1, 0),
            )
            np.testing.assert_array_equal(
                saved[f"{repair_prefix}contact_support_indices"][0, 4],
                (3, 1, 0, 2),
            )
            self.assertEqual(
                float(saved[f"{repair_prefix}bound_headroom_rad"][0, 4]),
                1.0e-6,
            )
            self.assertTrue(
                np.isnan(saved[f"{repair_prefix}source_q_rad"][1, 0]).all()
            )
            self.assertEqual(
                saved["failure_final_best_q_rad"].shape,
                (23,),
            )
            self.assertEqual(
                saved["failure_final_best_points_m"].shape,
                (5, 3),
            )
            self.assertEqual(
                saved["failure_final_best_arcs_m"].shape,
                (5,),
            )
            self.assertEqual(
                saved["failure_final_best_desired_arcs_m"].shape,
                (5,),
            )
            self.assertEqual(saved["bridge_rejected_q_rad"].shape, (23,))
            self.assertEqual(
                saved["bridge_rejected_points_m"].shape,
                (5, 3),
            )
            self.assertEqual(
                saved["bridge_rejected_arcs_m"].shape,
                (5,),
            )
            self.assertEqual(
                saved["bridge_rejected_desired_arcs_m"].shape,
                (5,),
            )
            np.testing.assert_allclose(
                saved["failure_final_best_points_m"],
                1.0,
            )
            np.testing.assert_allclose(
                saved["bridge_rejected_points_m"],
                3.0,
            )
            self.assertNotIn("failure_candidate_points_m", saved.files)
            self.assertEqual(
                saved["bridge_strict_conditions"].shape,
                (9,),
            )
            self.assertAlmostEqual(
                float(
                    saved[
                        "budget_feasibility_bridge_tip_target_scale"
                    ]
                ),
                0.5,
            )

        mispaired_values = dict(values)
        mispaired_values.pop("rejected_moving_bridge")
        with self.assertRaisesRegex(ValueError, "provided together"):
            DIAGNOSTICS.save_mpc_failure_prefix(
                output,
                **mispaired_values,
            )

        unsafe_values = dict(values)
        unsafe_values["failure_metrics"] = {"unsafe": None}
        with (
            mock.patch.object(Path, "mkdir"),
            self.assertRaisesRegex(TypeError, "object dtype"),
        ):
            DIAGNOSTICS.save_mpc_failure_prefix(
                output,
                **unsafe_values,
            )

    def test_failure_prefix_legacy_provenance_uses_fixed_sentinels(
        self,
    ) -> None:
        output = Path("legacy_failure.npz")
        values = {
            "reason": "legacy",
            "keyframe": 2,
            "keyframe_count": 5,
            "failure_distance_m": 0.3,
            "last_feasible_distance_m": np.asarray((0.0, 0.25)),
            "last_feasible_q_rad": np.zeros((2, 23)),
            "last_feasible_points_m": np.zeros((2, 5, 3)),
            "last_feasible_arcs_m": np.zeros((2, 5)),
            "final_best_desired_arcs_m": np.zeros(5),
            "final_best_q_rad": np.zeros(23),
            "final_best_points_m": np.zeros((5, 3)),
            "final_best_arcs_m": np.zeros(5),
            "rephase_offset_m": np.zeros(4),
            "budget_values": {},
            "failure_metrics": {},
        }
        with mock.patch.object(
            DIAGNOSTICS,
            "save_npz_no_overwrite",
            return_value=output,
        ) as save:
            self.assertEqual(
                DIAGNOSTICS.save_mpc_failure_prefix(output, **values),
                output,
            )
        payload = save.call_args.args[1]
        self.assertEqual(int(payload["schema_version"]), 3)
        self.assertEqual(int(payload["committed_coarse_row_count"]), 2)
        self.assertEqual(
            payload["committed_coarse_row_count"].dtype,
            np.dtype(np.int32),
        )
        self.assertFalse(
            bool(payload["last_feasible_coarse_provenance_available"])
        )
        for name in (
            "auto_rephase_offset_m",
            "progress_m",
            "target_progress_m",
            "static_bridge_dwell_m",
            "recovery_bridge_dwell_m",
            "normal_error_m",
            "palm_target_m",
            "palm_position_error_m",
            "cost",
        ):
            array = payload[f"last_feasible_coarse_{name}"]
            self.assertEqual(array.shape[0], 2)
            self.assertEqual(array.dtype, np.dtype(np.float64))
            self.assertTrue(np.isnan(array).all())
        for name in (
            "feasibility_bridge",
            "suffix_horizon",
            "static_feasibility_bridge",
            "recovery_bridge",
        ):
            array = payload[f"last_feasible_coarse_{name}"]
            self.assertEqual(array.shape, (2,))
            self.assertEqual(array.dtype, np.dtype(np.bool_))
            self.assertFalse(np.any(array))
        nfev = payload["last_feasible_coarse_nfev"]
        self.assertEqual(nfev.shape, (2,))
        self.assertEqual(nfev.dtype, np.dtype(np.int32))
        self.assertTrue(np.all(nfev == -1))
        self.assertFalse(bool(payload["refinement_provenance_available"]))
        self.assertEqual(payload["auto_refine_inserted_distance_m"].shape, (0,))
        self.assertEqual(payload["auto_refine_inserted_reason"].shape, (0,))
        self.assertNotEqual(
            payload["auto_refine_inserted_reason"].dtype,
            np.dtype(object),
        )
        self.assertFalse(bool(payload["rolling_provenance_available"]))
        self.assertEqual(payload["rolling_frame_target_distance_m"].shape, (0,))
        self.assertEqual(int(payload["rolling_window_frames"]), -1)
        self.assertTrue(np.isnan(payload["rolling_forward_progress_ratio"]))
        self.assertEqual(int(payload["rolling_required_forward_fingers"]), -1)

        valid_coarse = {
            "auto_rephase_offset_m": np.zeros((2, 4)),
            "progress_m": np.zeros((2, 5)),
            "target_progress_m": np.zeros((2, 5)),
            "feasibility_bridge": np.zeros(2, dtype=np.bool_),
            "suffix_horizon": np.zeros(2, dtype=np.bool_),
            "static_feasibility_bridge": np.zeros(2, dtype=np.bool_),
            "static_bridge_dwell_m": np.zeros(2),
            "recovery_bridge": np.zeros(2, dtype=np.bool_),
            "recovery_bridge_dwell_m": np.zeros(2),
            "normal_error_m": np.zeros((2, 5)),
            "palm_target_m": np.zeros((2, 3)),
            "palm_position_error_m": np.zeros(2),
            "cost": np.zeros(2),
            "nfev": np.zeros(2, dtype=np.int32),
        }
        valid_coarse["auto_rephase_offset_m"] = np.zeros((1, 4))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            DIAGNOSTICS.save_mpc_failure_prefix(
                output,
                **values,
                coarse_provenance=valid_coarse,
            )

        wrong_commit_count = dict(values)
        wrong_commit_count["keyframe"] = 1
        with self.assertRaisesRegex(ValueError, "committed last-feasible"):
            DIAGNOSTICS.save_mpc_failure_prefix(
                output,
                **wrong_commit_count,
            )

    def test_failure_prefix_rejects_malformed_authoritative_prefix(
        self,
    ) -> None:
        output = Path("malformed_prefix.npz")
        values = {
            "reason": "test",
            "keyframe": 2,
            "keyframe_count": 5,
            "failure_distance_m": 0.3,
            "last_feasible_distance_m": np.asarray((0.0, 0.25)),
            "last_feasible_q_rad": np.zeros((2, 23)),
            "last_feasible_points_m": np.zeros((2, 5, 3)),
            "last_feasible_arcs_m": np.zeros((2, 5)),
            "final_best_desired_arcs_m": np.zeros(5),
            "final_best_q_rad": np.zeros(23),
            "final_best_points_m": np.zeros((5, 3)),
            "final_best_arcs_m": np.zeros(5),
            "rephase_offset_m": np.zeros(4),
            "budget_values": {},
            "failure_metrics": {},
        }
        malformed = (
            (
                "nonzero origin",
                "last_feasible_distance_m",
                np.asarray((1.0e-4, 0.25)),
                "start at zero",
            ),
            (
                "duplicate distance",
                "last_feasible_distance_m",
                np.asarray((0.0, 0.0)),
                "strictly increasing",
            ),
            (
                "failure before prefix end",
                "failure_distance_m",
                0.25,
                "end before the failure distance",
            ),
            (
                "wrong joint width",
                "last_feasible_q_rad",
                np.zeros((2, 22)),
                "row-aligned arrays",
            ),
            (
                "wrong point count",
                "last_feasible_arcs_m",
                np.zeros((2, 4)),
                "row-aligned arrays",
            ),
            (
                "invalid keyframe ledger",
                "keyframe_count",
                1,
                "committed prefix",
            ),
        )
        for label, name, value, message in malformed:
            with self.subTest(label=label):
                trial = dict(values)
                trial[name] = value
                with self.assertRaisesRegex(ValueError, message):
                    DIAGNOSTICS.save_mpc_failure_prefix(output, **trial)


class BaselineTwoAdmittanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planned = np.zeros((1, 4, 3), dtype=np.float64)
        self.planned[..., 0] = np.asarray([[0.01, 0.02, 0.03, 0.04]])
        self.planned[..., 1] = np.asarray([[-0.01, -0.02, -0.03, -0.04]])
        self.normals = np.zeros_like(self.planned)
        self.normals[..., 2] = 1.0

    def test_low_direct_force_moves_inward_and_preserves_tangent(self) -> None:
        controller = CORE.FingertipNormalAdmittance()
        result = controller.step(
            self.planned,
            self.normals,
            np.zeros_like(self.planned),
        )
        self.assertTrue(np.all(result.normal_offset > 0.0))
        self.assertTrue(np.all(result.command_points[..., 2] < 0.0))
        np.testing.assert_allclose(
            result.command_points[..., :2],
            self.planned[..., :2],
            atol=1.0e-12,
        )

    def test_high_force_releases_normal_offset(self) -> None:
        gains = CORE.FingertipAdmittanceGains(
            virtual_mass=0.08,
            virtual_damping=3.0,
            virtual_stiffness=100.0,
            force_filter_alpha=1.0,
            max_normal_acceleration=5.0,
        )
        controller = CORE.FingertipNormalAdmittance(gains)
        low_force = np.zeros_like(self.planned)
        for _ in range(20):
            low_result = controller.step(
                self.planned,
                self.normals,
                low_force,
            )
        peak_offset = low_result.normal_offset.copy()
        high_force = np.zeros_like(self.planned)
        high_force[..., 2] = 3.0
        for _ in range(60):
            high_result = controller.step(
                self.planned,
                self.normals,
                high_force,
            )
        self.assertTrue(np.all(high_result.normal_offset < peak_offset))

    def test_fingertip_offset_speed_and_contact_hysteresis_are_bounded(
        self,
    ) -> None:
        controller = CORE.FingertipNormalAdmittance()
        result = None
        for _ in range(1000):
            result = controller.step(
                self.planned,
                self.normals,
                np.zeros_like(self.planned),
            )
        assert result is not None
        self.assertLessEqual(
            float(np.max(np.abs(result.normal_offset))),
            controller.gains.max_normal_offset + 1.0e-12,
        )
        self.assertLessEqual(
            float(np.max(np.abs(result.normal_velocity))),
            controller.gains.max_normal_speed + 1.0e-12,
        )

        hysteresis = CORE.FingertipNormalAdmittance(
            CORE.FingertipAdmittanceGains(force_filter_alpha=1.0)
        )
        force = np.zeros_like(self.planned)
        force[..., 2] = 0.16
        self.assertTrue(
            np.all(
                hysteresis.step(
                    self.planned, self.normals, force
                ).contact_active
            )
        )
        force[..., 2] = 0.10
        self.assertTrue(
            np.all(
                hysteresis.step(
                    self.planned, self.normals, force
                ).contact_active
            )
        )
        force[..., 2] = 0.0
        self.assertFalse(
            np.any(
                hysteresis.step(
                    self.planned, self.normals, force
                ).contact_active
            )
        )

    def test_wrist_admittance_is_wrench_driven_and_bounded(self) -> None:
        controller = CORE.WristCartesianAdmittance()
        normal = np.asarray([[0.0, 0.0, 1.0]])
        wrench = np.asarray([[0.0, 0.0, 8.0, 0.0, 0.0, 0.0]])
        first = controller.step(wrench, normal)
        self.assertGreater(float(first.reference_offset[0, 2]), 0.0)
        self.assertLessEqual(
            float(np.linalg.norm(first.filtered_wrench_error[0, :3])),
            controller.gains.max_force_error + 1.0e-12,
        )
        result = first
        for _ in range(1000):
            result = controller.step(wrench, normal)
        self.assertLessEqual(
            float(np.linalg.norm(result.reference_offset[0, :3])),
            controller.gains.max_translation_offset + 1.0e-12,
        )
        self.assertLessEqual(
            float(np.linalg.norm(result.reference_velocity[0, :3])),
            controller.gains.max_translation_speed + 1.0e-12,
        )
        torque_result = controller.step(
            np.asarray([[0.0, 0.0, 0.0, 8.0, 8.0, 8.0]]),
            normal,
        )
        self.assertLessEqual(
            float(
                np.linalg.norm(
                    torque_result.filtered_wrench_error[0, 3:]
                )
            ),
            controller.gains.max_torque_error + 1.0e-12,
        )

    def test_adapter_uses_direct_forces_and_separate_wrist_loop(self) -> None:
        core_source = MODULE_PATH.read_text(encoding="utf-8")
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        demo_source = DEMO_PATH.read_text(encoding="utf-8")
        self.assertIn("class FingertipForceFingerMCCController", source)
        self.assertIn("finger_obs[:, :12]", source)
        self.assertIn("direct_forces_local_batch", source)
        self.assertIn("FingertipNormalAdmittance", source)
        self.assertIn("WristCartesianAdmittance", source)
        self.assertIn("--max-tip-contact-force-n", demo_source)
        self.assertIn("--max-tip-raw-force-n", demo_source)
        self.assertIn("max_tactile_force", demo_source)
        self.assertIn("max_filtered_normal_force", demo_source)
        self.assertIn("tip_force_from_motors_diagnostic", source)
        self.assertIn(
            "12 + ARM_DOF : 12 + TOTAL_DOF",
            source,
        )
        self.assertIn("tip_normal_force_signed_raw", demo_source)
        self.assertIn(
            "calibrate_fingertip_force_sign",
            demo_source,
        )
        self.assertIn(
            "calibrate_fingertip_force_setpoint",
            demo_source,
        )
        self.assertNotIn('["tip_force_from_motors"]', demo_source)
        self.assertNotIn(
            "calibrate_motor_force_setpoint",
            demo_source,
        )
        self.assertNotIn("calibrate_motor_force_setpoint", source)
        self.assertNotIn("MotorForceFingerMCCController", source)
        self.assertNotIn("normal_preload_m =", demo_source)
        self.assertNotIn("--variant", demo_source)
        self.assertNotIn("MCC_VARIANTS", core_source)
        self.assertNotIn("FullHandMCCCore", core_source)
        for retired_label in (
            "independent_mcc",
            "motor_torque_mcc",
            "hierarchical_mcc",
            "hybrid_force_position",
            "passivity_tank",
        ):
            self.assertNotIn(retired_label, core_source)
            self.assertNotIn(retired_label, source)
            self.assertNotIn(retired_label, demo_source)
        self.assertIn("action - self._previous_action", source)


class SurfaceGeometryTest(unittest.TestCase):
    def test_capsule_projection_and_slide_stay_on_surface(self) -> None:
        points = np.asarray(
            [
                [0.20, 0.02, 0.00],
                [0.02, -0.30, 0.03],
                [0.01, 0.02, 0.30],
                [-0.20, 0.01, -0.25],
                [0.30, 0.20, 0.05],
            ]
        )
        center = np.zeros(3)
        rotation = np.eye(3)
        radius = 0.15
        half_height = 0.08
        surface, normals = GEOMETRY.capsule_project(
            points, center, rotation, radius, half_height
        )
        axis_points = np.zeros_like(surface)
        axis_points[:, 2] = np.clip(
            surface[:, 2], -half_height, half_height
        )
        distance = np.linalg.norm(surface - axis_points, axis=1)
        np.testing.assert_allclose(distance, radius, atol=1.0e-6)
        np.testing.assert_allclose(
            np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-6
        )

        moved = GEOMETRY.rotate_about_capsule_axis(
            surface, center, rotation, angle=0.2
        )
        moved_surface, _ = GEOMETRY.capsule_project(
            moved, center, rotation, radius, half_height
        )
        np.testing.assert_allclose(moved, moved_surface, atol=1.0e-6)

    def test_capsule_meridian_plan_round_trip_and_frames(self) -> None:
        radius = 0.07
        half_height = 0.10
        total = np.pi * radius + 2.0 * half_height
        arc = np.linspace(0.01, total - 0.01, 9)
        azimuth = np.linspace(-2.4, 2.4, 9)
        center = np.asarray([0.3, -0.2, 0.8])
        rotation = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        points, normals, frames = GEOMETRY.capsule_meridian_targets(
            arc,
            azimuth,
            center,
            rotation,
            radius,
            half_height,
        )
        recovered_arc, recovered_azimuth = (
            GEOMETRY.capsule_meridian_coordinates(
                points,
                center,
                rotation,
                radius,
                half_height,
            )
        )
        np.testing.assert_allclose(recovered_arc, arc, atol=1.0e-6)
        np.testing.assert_allclose(recovered_azimuth, azimuth, atol=1.0e-6)
        np.testing.assert_allclose(normals, frames[:, :, 0], atol=1.0e-7)
        for frame in frames:
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1.0e-6)

    def test_ellipsoid_meridian_round_trip_and_frames(self) -> None:
        radial_radius = 0.15
        axial_radius = 0.28
        total = GEOMETRY.ellipsoid_meridian_total_length(
            radial_radius,
            axial_radius,
        )
        arc = np.linspace(0.01, total - 0.01, 17)
        azimuth = np.linspace(-2.0, 2.0, 17)
        center = np.asarray([0.3, -0.2, 0.8])
        rotation = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        points, normals, frames = GEOMETRY.ellipsoid_meridian_targets(
            arc,
            azimuth,
            center,
            rotation,
            radial_radius,
            axial_radius,
        )
        recovered_arc, recovered_azimuth = (
            GEOMETRY.ellipsoid_meridian_coordinates(
                points,
                center,
                rotation,
                radial_radius,
                axial_radius,
            )
        )
        np.testing.assert_allclose(recovered_arc, arc, atol=2.0e-6)
        np.testing.assert_allclose(recovered_azimuth, azimuth, atol=1.0e-6)
        np.testing.assert_allclose(normals, frames[:, :, 0], atol=1.0e-7)
        for frame in frames:
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1.0e-6)

    def test_ellipsoid_has_large_continuous_curvature_change(self) -> None:
        radial_radius = 0.15
        axial_radius = 0.28
        total = GEOMETRY.ellipsoid_meridian_total_length(
            radial_radius,
            axial_radius,
        )
        curvature = GEOMETRY.ellipsoid_meridian_curvature(
            np.linspace(0.0, 0.5 * total, 1001),
            radial_radius,
            axial_radius,
        )
        self.assertTrue(np.all(np.isfinite(curvature)))
        self.assertGreater(
            float(curvature.max() / curvature.min()),
            6.0,
        )


class CertifiedSuffixCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.apply_cache_event = load_demo_pure_function(
            "apply_suffix_cache_event"
        )
        self.validate_cache = load_demo_pure_function(
            "validated_suffix_cache_seed"
        )
        self.cache = {
            "anchor_distance_m": 0.0,
            "anchor_q_rad": np.asarray([0.0, 0.0]),
            "distance_m": np.asarray([1.0, 3.0]),
            "q_rad": np.asarray([[1.0, 10.0], [3.0, 30.0]]),
        }

    def test_certified_cache_resamples_new_future_midpoint(self) -> None:
        horizon = np.asarray([0.5, 2.0, 4.0])
        seed, evidence = self.validate_cache(
            self.cache,
            current_anchor_distance_m=0.0,
            current_anchor_q_rad=np.asarray([0.0, 0.0]),
            horizon_distance_m=horizon,
            total_dof=2,
        )
        self.assertIsNotNone(seed)
        np.testing.assert_allclose(
            seed,
            np.asarray([[0.5, 5.0], [2.0, 20.0], [3.0, 30.0]]),
        )
        self.assertTrue(bool(evidence["cache_available"]))
        self.assertTrue(bool(evidence["cache_schema_valid"]))
        self.assertTrue(bool(evidence["cache_anchor_valid"]))
        self.assertTrue(bool(evidence["cache_valid"]))
        np.testing.assert_array_equal(
            evidence["cache_distance_m"],
            self.cache["distance_m"],
        )
        np.testing.assert_array_equal(
            evidence["cache_q_rad"],
            self.cache["q_rad"],
        )
        np.testing.assert_array_equal(
            evidence["cache_seed_distance_m"],
            horizon,
        )
        np.testing.assert_allclose(evidence["cache_seed_q_rad"], seed)
        for value in evidence.values():
            self.assertNotEqual(np.asarray(value).dtype, np.dtype("O"))

    def test_terminal_local_horizon_preserves_certified_knots(self) -> None:
        cache = {
            "anchor_distance_m": 0.046875,
            "anchor_q_rad": np.asarray([0.0, 0.0]),
            "distance_m": np.asarray(
                (0.04703125, 0.0471875, 0.04734375, 0.0475),
                dtype=np.float64,
            ),
            "q_rad": np.asarray(
                ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)),
                dtype=np.float64,
            ),
        }
        horizon = np.asarray(
            (0.04703125, 0.0471875, 0.04734375, 0.0475, 0.04765625),
            dtype=np.float64,
        )
        seed, evidence = self.validate_cache(
            cache,
            current_anchor_distance_m=0.046875,
            current_anchor_q_rad=np.asarray([0.0, 0.0]),
            horizon_distance_m=horizon,
            total_dof=2,
        )
        self.assertIsNotNone(seed)
        np.testing.assert_array_equal(seed[:4], cache["q_rad"])
        np.testing.assert_array_equal(seed[4], cache["q_rad"][-1])
        self.assertTrue(bool(evidence["cache_valid"]))
        np.testing.assert_array_equal(
            evidence["cache_seed_distance_m"], horizon
        )

    def test_cache_lifecycle_preserves_resamples_replaces_and_clears(
        self,
    ) -> None:
        retained = self.apply_cache_event(
            self.cache,
            event="auto_refine",
        )
        self.assertIs(retained, self.cache)
        seed, evidence = self.validate_cache(
            retained,
            current_anchor_distance_m=0.0,
            current_anchor_q_rad=np.asarray([0.0, 0.0]),
            horizon_distance_m=np.asarray([0.25, 2.0]),
            total_dof=2,
        )
        self.assertIsNotNone(seed)
        self.assertTrue(bool(evidence["cache_valid"]))
        np.testing.assert_allclose(seed[0], np.asarray([0.15625, 1.5625]))

        replacement = {
            "anchor_distance_m": 0.25,
            "anchor_q_rad": np.asarray(seed[0]).copy(),
            "distance_m": np.asarray([2.0]),
            "q_rad": np.asarray(seed[1:]).copy(),
        }
        committed = self.apply_cache_event(
            retained,
            event="suffix_commit",
            pending_cache=replacement,
        )
        self.assertIs(committed, replacement)
        cleared = self.apply_cache_event(
            committed,
            event="non_suffix_commit",
        )
        self.assertIsNone(cleared)

    def test_cache_lifecycle_rejects_unknown_or_incomplete_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown suffix cache event"):
            self.apply_cache_event(self.cache, event="typo")
        with self.assertRaisesRegex(ValueError, "requires a pending cache"):
            self.apply_cache_event(self.cache, event="suffix_commit")

    def test_certified_cache_anchor_mismatch_fails_closed(self) -> None:
        for mismatch in ("distance", "q"):
            with self.subTest(mismatch=mismatch):
                seed, evidence = self.validate_cache(
                    self.cache,
                    current_anchor_distance_m=(
                        2.0e-12 if mismatch == "distance" else 0.0
                    ),
                    current_anchor_q_rad=(
                        np.asarray([0.0, 2.0e-10])
                        if mismatch == "q"
                        else np.asarray([0.0, 0.0])
                    ),
                    horizon_distance_m=np.asarray([0.5, 2.0]),
                    total_dof=2,
                )
                self.assertIsNone(seed)
                self.assertTrue(bool(evidence["cache_available"]))
                self.assertTrue(bool(evidence["cache_schema_valid"]))
                self.assertFalse(bool(evidence["cache_anchor_valid"]))
                self.assertFalse(bool(evidence["cache_valid"]))
                np.testing.assert_array_equal(
                    evidence["cache_distance_m"],
                    self.cache["distance_m"],
                )
                np.testing.assert_array_equal(
                    evidence["cache_q_rad"],
                    self.cache["q_rad"],
                )

    def test_certified_cache_schema_mismatch_fails_closed(self) -> None:
        malformed_values = (
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, np.nan]),
            np.asarray([1.0, np.inf]),
            np.asarray([3.0, 1.0]),
        )
        for malformed_distance in malformed_values:
            with self.subTest(distance=malformed_distance):
                malformed_cache = dict(self.cache)
                malformed_cache["distance_m"] = malformed_distance
                seed, evidence = self.validate_cache(
                    malformed_cache,
                    current_anchor_distance_m=0.0,
                    current_anchor_q_rad=np.asarray([0.0, 0.0]),
                    horizon_distance_m=np.asarray([0.5, 2.0]),
                    total_dof=2,
                )
                self.assertIsNone(seed)
                self.assertTrue(bool(evidence["cache_available"]))
                self.assertFalse(bool(evidence["cache_schema_valid"]))
                self.assertFalse(bool(evidence["cache_valid"]))

    def test_current_anchor_distance_must_be_finite_scalar(self) -> None:
        malformed_values = (
            np.asarray([0.0]),
            np.asarray([0.0, 0.0]),
            np.nan,
            np.inf,
            "not-a-number",
        )
        for malformed_anchor in malformed_values:
            with self.subTest(anchor=malformed_anchor):
                with self.assertRaisesRegex(ValueError, "finite scalar"):
                    self.validate_cache(
                        self.cache,
                        current_anchor_distance_m=malformed_anchor,
                        current_anchor_q_rad=np.asarray([0.0, 0.0]),
                        horizon_distance_m=np.asarray([0.5, 2.0]),
                        total_dof=2,
                    )

    def test_cached_anchor_distance_must_be_numeric_finite_scalar(self) -> None:
        malformed_values = (
            np.asarray([0.0]),
            np.asarray([0.0, 0.0]),
            np.nan,
            np.inf,
            "not-a-number",
        )
        for malformed_anchor in malformed_values:
            with self.subTest(anchor=malformed_anchor):
                malformed_cache = dict(self.cache)
                malformed_cache["anchor_distance_m"] = malformed_anchor
                seed, evidence = self.validate_cache(
                    malformed_cache,
                    current_anchor_distance_m=0.0,
                    current_anchor_q_rad=np.asarray([0.0, 0.0]),
                    horizon_distance_m=np.asarray([0.5, 2.0]),
                    total_dof=2,
                )
                self.assertIsNone(seed)
                self.assertFalse(bool(evidence["cache_schema_valid"]))
                self.assertFalse(bool(evidence["cache_valid"]))

    def test_horizon_must_be_finite_ordered_and_after_anchor(self) -> None:
        malformed_values = (
            np.asarray([0.5, np.nan]),
            np.asarray([0.5, np.inf]),
            np.asarray([0.5, 0.4]),
            np.asarray([0.0, 0.5]),
            np.asarray([1.0e-12, 0.5]),
        )
        for malformed_horizon in malformed_values:
            with self.subTest(horizon=malformed_horizon):
                with self.assertRaisesRegex(ValueError, "after anchor"):
                    self.validate_cache(
                        self.cache,
                        current_anchor_distance_m=0.0,
                        current_anchor_q_rad=np.asarray([0.0, 0.0]),
                        horizon_distance_m=malformed_horizon,
                        total_dof=2,
                    )


class AdaptiveMPCSourceStructureTest(unittest.TestCase):
    def test_auto_refinement_retains_only_validated_certified_cache(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        refinement_node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "insert_auto_refinement"
        )
        refinement = ast.get_source_segment(source, refinement_node)
        self.assertIsNotNone(refinement)
        self.assertIn("suffix_cache_retained=", refinement)

        lifecycle_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_suffix_cache_event"
        ]
        self.assertEqual(len(lifecycle_calls), 2)
        lifecycle_events = {
            constant.value
            for call in lifecycle_calls
            for keyword in call.keywords
            if keyword.arg == "event"
            for constant in ast.walk(keyword.value)
            if isinstance(constant, ast.Constant)
            and isinstance(constant.value, str)
        }
        self.assertEqual(
            lifecycle_events,
            {"auto_refine", "suffix_commit", "non_suffix_commit"},
        )

        cache_helper_start = source.index("def validated_suffix_cache_seed")
        cache_helper_end = source.index(
            "def build_mpc_distance_grid", cache_helper_start
        )
        cache_helper = source[cache_helper_start:cache_helper_end]
        for evidence_name in (
            '"cache_anchor_distance_m"',
            '"cache_anchor_q_rad"',
            '"cache_distance_m"',
            '"cache_q_rad"',
            '"cache_seed_distance_m"',
            '"cache_seed_q_rad"',
        ):
            self.assertIn(evidence_name, cache_helper)
        self.assertIn("anchor_distance_error <= 1.0e-12", cache_helper)
        self.assertIn("anchor_q_error <= 1.0e-10", cache_helper)

        cache_use_start = source.index(
            "cache_seed, cache_evidence = ("
        )
        cache_use_end = source.index("def audit_suffix", cache_use_start)
        cache_use = source[cache_use_start:cache_use_end]
        self.assertIn("validated_suffix_cache_seed(", cache_use)
        self.assertIn(
            "last_suffix_horizon_evidence.update(cache_evidence)",
            cache_use,
        )
        self.assertLess(
            cache_use.index(
                'append_suffix_seed(cache_seed, "certified_cache")'
            ),
            cache_use.index("prioritized_suffix_seed_indices("),
        )

        final_evidence_start = source.index(
            '"status": np.asarray("completed")',
            cache_use_end,
        )
        final_evidence_end = source.index(
            'print(\n                                "[SUFFIX-HORIZON] "',
            final_evidence_start,
        )
        final_evidence = source[
            final_evidence_start:final_evidence_end
        ]
        self.assertIn(
            "last_suffix_horizon_evidence.update(",
            final_evidence,
        )
        self.assertIn("cache_evidence", final_evidence)

        self.assertIn("maximum_sources=4", source)

    def test_fallback_candidates_share_soft_pad_first_rank(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        fallback_start = source.index("def fallback_orientation_rank")
        fallback_end = source.index(
            "# Before launching another non-convex",
            fallback_start,
        )
        fallback = source[fallback_start:fallback_end]
        self.assertIn("orientation_aware_candidate_rank(", fallback)
        self.assertIn("soft_pad_alignment=planner_soft_pad_alignment", fallback)
        self.assertGreaterEqual(
            source.count("fallback_orientation_rank("),
            4,
        )
        self.assertIn(
            "static_bridge_candidate = (\n"
            "                            fallback_orientation_rank(",
            source,
        )
        self.assertIn(
            "key=lambda item: item[0]",
            source,
        )

    def test_all_central_23dof_solves_use_explicit_finite_difference_step(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        for start_marker, end_marker in (
            ("result = least_squares(", "result.candidate_kind"),
            ("repaired = least_squares(", "repaired.candidate_kind"),
            ("rephased = least_squares(", "rephased.candidate_kind"),
        ):
            start = source.index(start_marker)
            end = source.index(end_marker, start)
            self.assertIn("diff_step=1.0e-5", source[start:end])

    def test_circumferential_plan_runs_full_robot_audit_before_save(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        start = source.index("def _build_circumferential_surface_mpc_plan")
        end = source.index("def _build_adaptive_surface_mpc_plan", start)
        planner = source[start:end]
        audit = planner.index("self._validate_full_robot_plan_clearance(")
        save = planner.index("np.savez(")
        self.assertLess(audit, save)
        self.assertIn('label="circumferential_surface_mpc"', planner)

    def test_orientation_candidates_share_rank_without_raw_privilege(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        ordinary_start = source.index(
            "candidates = []",
            source.index("rigid_arm_seed = rigid_arm_result.joint_position"),
        )
        ordinary_end = source.index(
            "preliminary_progress =",
            ordinary_start,
        )
        ordinary = source[ordinary_start:ordinary_end]
        self.assertNotIn("(-2.0,", ordinary)
        self.assertNotIn("(-1.0,", ordinary)
        self.assertGreaterEqual(
            ordinary.count("orientation_aware_candidate_rank("),
            3,
        )
        for required_seed in (
            '"orientation_posture_surface"',
            '"orientation_posture_extrapolated"',
            '"orientation_posture_previous"',
            "4.0 * args.planner_soft_pad_weight",
        ):
            self.assertIn(required_seed, ordinary)
        posture_start = ordinary.index("orientation_posture_seed_specs")
        posture_end = ordinary.index(
            "local_seed_specs.extend(orientation_posture_seed_specs)",
            posture_start,
        )
        posture_block = ordinary[posture_start:posture_end]
        self.assertNotIn("feasibility_bridge_selected = True", posture_block)
        self.assertNotIn("recovery_bridge_selected = True", posture_block)
        self.assertNotIn("static_feasibility_bridge_selected = True", posture_block)

    def test_segment_audit_samples_pad_and_physical_tips_nine_times(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        segment_start = source.index("def segment_collision_status")
        segment_end = source.index("transported_offset =", segment_start)
        segment = source[segment_start:segment_end]
        self.assertIn("sample_count = max(\n                        9,", segment)
        self.assertIn("reachability.geometry_group_clearances", segment)
        self.assertIn("segment_tip_clearance_status", segment)
        self.assertIn("args.max_contact_penetration_mm", segment)
        self.assertIn("active_self_pairs.update(sample_self_pairs)", segment)
        self.assertIn("self_pair_sample_occurrences", segment)
        self.assertIn("minimum_protected_self_clearance", segment)
        self.assertNotIn("max_runtime_self_penetration_mm", segment)

    def test_level2_runner_freezes_both_planner_cones_at_40_degrees(
        self,
    ) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        acceptance_start = source.index('if ($Mode -eq "Acceptance")')
        diagnostic_start = source.index("else {", acceptance_start)
        arguments_start = source.index("$pythonArguments", diagnostic_start)
        acceptance = source[acceptance_start:diagnostic_start]
        diagnostic = source[diagnostic_start:arguments_start]
        self.assertIn('$maxPadAngleDeg = "45"', acceptance)
        self.assertIn('$plannerPadAngleMarginDeg = "5"', acceptance)
        self.assertIn('$maxPadAngleDeg = "50"', diagnostic)
        self.assertIn('$plannerPadAngleMarginDeg = "10"', diagnostic)
        self.assertIn('"--planner-soft-pad-angle-deg", "35"', source)
        self.assertIn('"--planner-soft-pad-weight", "24"', source)
        self.assertIn(
            '"--planner-soft-pad-softplus-tau", "0.02"',
            source,
        )
        self.assertIn(
            '"--planner-tip-geom-target-mm", "-0.25", "-0.25", '
            '"-0.50", "-0.25"',
            source,
        )
        self.assertIn('"--planner-tip-geom-weight", "2200"', source)
        self.assertIn(
            '"--planner-tip-geom-inner-cap-mm", "-0.8"',
            source,
        )
        self.assertIn(
            '"--planner-tip-geom-inner-weight", "18000"',
            source,
        )
        self.assertIn(
            '"--planner-protected-self-clearance-mm", "0.10"',
            source,
        )
        self.assertIn(
            '"--planner-protected-self-clearance-weight", "4000"',
            source,
        )
        self.assertIn(
            '"--planner-self-separation-seed-step-rad", "0.005"',
            source,
        )
        self.assertIn(
            '"--mpc-feasibility-bridge-tip-target-scale", "0.5"',
            source,
        )

    def test_physical_tip_objective_and_full_plan_audit_are_distinct(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        residual_start = source.index("def residual(")
        residual_end = source.index("candidates = []", residual_start)
        residual = source[residual_start:residual_end]
        for required_term in (
            "tip_geom_clearance",
            "planner_tip_geom_target_m",
            "planner_tip_geom_inner_cap_m",
            "args.planner_tip_geom_weight * tip_geom_error",
            "args.planner_tip_geom_inner_weight",
            "positive_self_clearance_residual",
            "args.planner_protected_self_clearance_weight",
            "protected_self_pairs",
        ):
            self.assertIn(required_term, residual)
        audit_start = source.index(
            "def _validate_full_robot_plan_clearance"
        )
        audit_end = source.index("def _object_pose", audit_start)
        audit = source[audit_start:audit_end]
        self.assertIn("minimum_tip = np.full(4", audit)
        self.assertIn("minimum_tip < -allowed_tip_penetration", audit)
        self.assertIn("per_tip_minimum_mm", audit)
        self.assertIn("self.min_planned_tip_clearance_m", audit)
        self.assertIn(
            "args.max_pad_angle_deg\n"
            "                        - args.planner_pad_angle_margin_deg",
            audit,
        )
        self.assertIn("planner_limit_deg=", audit)
        self.assertIn("if self_pairs:", audit)
        self.assertIn("minimum_protected_self_clearance", audit)
        self.assertIn("[INITIAL-PHYSICAL-TIP-AUDIT]", source)
        self.assertIn("[INITIAL-PROTECTED-SELF-AUDIT]", source)

    def test_protected_self_separation_multistart_is_triggered_and_ranked(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        planner_start = source.index("def _build_adaptive_surface_mpc_plan")
        planner = source[planner_start:]
        self.assertIn("PROTECTED_SELF_PAIR_NAMES", source)
        for required in (
            "protected_self_clearance_state",
            "protected_self_separation_seeds",
            "transported_suffix_seed_rows(",
            "central_difference_clearance_gradient",
            "self_separation_ascent_seeds",
            "for fd_step_rad in (1.0e-6, 5.0e-7)",
            "if improving_seeds:",
            '("surface", surface_ik_seed)',
            '("extrapolated", extrapolated_seed)',
            '("previous", previous_q)',
            "previous_q - args.max_plan_joint_step_rad",
            "previous_q + args.max_plan_joint_step_rad",
            'f"task_{separation_kind}"',
            'f"orientation_posture_{separation_kind}"',
            "minimum_protected_self_clearance_m=",
            "soft_self_clearance_target_m=",
        ):
            self.assertIn(required, planner)

    def test_dynamic_refinement_uses_a_mutable_keyframe_loop(self) -> None:
        tree = ast.parse(DEMO_PATH.read_text(encoding="utf-8"))
        planner = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_adaptive_surface_mpc_plan"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "keyframe"
                for node in ast.walk(planner)
            ),
            "A for-loop skips midpoint rows inserted into the MPC grid",
        )
        dynamic_loops = [
            node
            for node in ast.walk(planner)
            if isinstance(node, ast.While)
            and "keyframe" in ast.dump(node.test)
            and "keyframe_count" in ast.dump(node.test)
        ]
        self.assertEqual(len(dynamic_loops), 1)
        self.assertTrue(
            any(
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "keyframe"
                and isinstance(node.op, ast.Add)
                for node in ast.walk(dynamic_loops[0])
            ),
            "The mutable MPC keyframe loop must advance after acceptance",
        )

    def test_moving_bridge_rechecks_constraints_and_bounds_static_pauses(
        self,
    ) -> None:
        tree = ast.parse(DEMO_PATH.read_text(encoding="utf-8"))
        planner = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_adaptive_surface_mpc_plan"
        )
        source = ast.get_source_segment(
            DEMO_PATH.read_text(encoding="utf-8"),
            planner,
        )
        self.assertIsNotNone(source)
        for required_term in (
            "bridge_candidate = least_squares",
            "moving_bridge_candidates",
            "moving_bridge_multistart_rank",
            "deduplicated_bridge_multistart_seeds",
            "bridge_candidate.bridge_multistart_rank",
            "moving_bridge_candidate_rank",
            "bridge_lower",
            "bridge_upper",
            "moving_bridge_motion_ok",
            "moving_tip_motion_m",
            "moving_progressing_finger_count",
            "progressing_finger_count_required",
            "bridge_active_fingers",
            "minimum_tip_motion_m",
            "moving_bridge_target_arc",
            "progress_aware_arc_targets",
            "bounded_moving_bridge_trust_radius",
            "moving_bridge_residual",
            "moving_bridge_local_residual",
            "bridge_anchor_standoff_m",
            "bridge_anchor_azimuth_rad",
            "diff_step=1.0e-5",
            "bridge_active_fingers",
            "mpc_feasibility_bridge_trust_radius_rad",
            "mpc_feasibility_bridge_min_progress_ratio",
            "mpc_feasibility_bridge_target_weight",
            "mpc_feasibility_bridge_tip_target_scale",
            "bridge_result = SimpleNamespace",
            "x=previous_q.copy()",
            "bridge_interval_short",
            "mpc_auto_refine_min_step_mm",
            "mpc_feasibility_bridge_max_mm",
            "bridge_progress_error",
            "bridge_normal_ok",
            "bridge_tangential_error",
            "bridge_monotonic_error",
            "bridge_palm_error",
            "bridge_collision_ok",
            "bridge_joint_limits_ok",
            "coarse_feasibility_bridge",
            "coarse_static_feasibility_bridge",
            "coarse_static_bridge_dwell_m",
            "static_bridge_total_m",
            "mpc_static_bridge_max_dwell_mm",
            "mpc_static_bridge_max_total_ratio",
            "mpc_static_bridge_progress_tolerance_mm",
            "mpc_coarse_feasibility_bridge",
            "mpc_coarse_static_feasibility_bridge",
            "mpc_coarse_static_bridge_dwell_m",
            "STATIC-FEASIBILITY-BRIDGE",
            "MOVING-FEASIBILITY-BRIDGE",
        ):
            self.assertIn(required_term, source)
        moving_residual_start = source.index("def moving_bridge_residual")
        moving_residual_end = source.index(
            "def moving_bridge_multistart_rank",
            moving_residual_start,
        )
        moving_residual_source = source[
            moving_residual_start:moving_residual_end
        ]
        self.assertIn(
            "bridge_base_residual = moving_bridge_local_residual(",
            moving_residual_source,
        )
        self.assertIn("return np.concatenate", moving_residual_source)
        self.assertIn(
            "args.planner_protected_self_clearance_weight",
            moving_residual_source,
        )
        self.assertIn(
            "positive_self_clearance_residual",
            moving_residual_source,
        )
        self.assertIn(
            "moving_bridge_tip_geometry_residual",
            moving_residual_source,
        )
        self.assertIn("planner_tip_geom_target_m", moving_residual_source)
        self.assertIn("planner_tip_geom_inner_cap_m", moving_residual_source)
        self.assertIn(
            "args.mpc_feasibility_bridge_tip_target_scale",
            moving_residual_source,
        )
        self.assertIn("moving_separation_seeds", source)
        self.assertNotIn(
            "moving_bridge_seed = max(",
            source,
        )
        self.assertNotIn("progress_target_arc", source)
        self.assertNotIn("palm_target", moving_residual_source)
        self.assertNotIn("desired_azimuth", moving_residual_source)
        self.assertNotIn("start_q", moving_residual_source)
        self.assertIn(
            "moving_bridge_arc - bridge_desired_arc",
            source,
        )
        self.assertIn(
            "moving_joint_motion_rad\n"
            "                            <= args.max_plan_joint_step_rad",
            source,
        )
        self.assertNotIn(
            "bridge_arc[1:] + direction * bridge_interval_m",
            source,
        )
        preaudit_start = source.index(
            "def moving_bridge_multistart_rank"
        )
        preaudit_end = source.index(
            "moving_bridge_seed = np.minimum",
            preaudit_start,
        )
        preaudit = source[preaudit_start:preaudit_end]
        for hard_gate in (
            "segment_collision_status(candidate.x)",
            "scheduled_contact_status(",
            "recovery_contact_status(",
            "evaluate_moving_bridge_motion(",
            "candidate_strict = evaluate_bridge_conditions(",
            "candidate_recovery = evaluate_bridge_conditions(",
            "strict_hard_feasible=strict_ok",
            "recovery_hard_feasible=recovery_ok",
            "minimum_tip_clearance_m=",
            "minimum_protected_self_clearance_m=",
            "minimum_pad_alignment=",
        ):
            self.assertIn(hard_gate, preaudit)
        solve_loop = source[
            source.index("moving_bridge_candidates = []") :
            source.index("moving_bridge = min(")
        ]
        self.assertIn("for bridge_seed_index, bridge_seed in enumerate(", solve_loop)
        self.assertIn("moving_bridge_multistart_rank(", solve_loop)

    def test_bridge_tip_scale_is_validated_and_saved_in_both_artifacts(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '"--mpc-feasibility-bridge-tip-target-scale"',
            source,
        )
        self.assertIn(
            "not np.isfinite(args.mpc_feasibility_bridge_tip_target_scale)",
            source,
        )
        self.assertIn(
            '"feasibility_bridge_tip_target_scale": (',
            source,
        )
        self.assertIn(
            "mpc_feasibility_bridge_tip_target_scale=np.asarray(",
            source,
        )

    def test_suffix_horizon_is_atomic_terminal_aware_and_audited(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        start = source.index("def build_suffix_horizon_candidate")
        end = source.index("def moving_bridge_residual", start)
        horizon = source[start:end]
        for required in (
            "build_receding_horizon_distances(",
            "suffix_terminal_start_m",
            "scheduled_fingertip_targets(",
            "progress_aware_arc_targets(",
            "strict_suffix_task_hinge_residual(",
            "solve_guard_m = suffix_optimization_guard(",
            "polish_guard_m = suffix_optimization_guard(",
            "suffix_interior_polish_scale_ladder(4.0)",
            "explicit_constraint_guard_m = (",
            "suffix_explicit_constraint_guard(task_guard_m)",
            "interior_guard_m=min(",
            "suffix_prefix_needs_interior_polish(",
            "suffix_node_needs_explicit_task_polish(",
            "suffix_explicit_restart_required(",
            "polish_source_trials = [",
            "rollout_node_polish_residual",
            '"interior_polish_"',
            "[SUFFIX-INTERIOR-POLISH]",
            "rollout_node_explicit_constraints",
            "strict_suffix_task_constraint_margins(",
            "suffix_explicit_support_indices(",
            'method="SLSQP"',
            "Bounds(",
            "explicit_objective_anchor_q",
            '"explicit_constraint_restart"',
            "[SUFFIX-EXPLICIT-CONSTRAINT-POLISH]",
            "suffix_transition_fractions",
            "suffix_node_residual(",
            "least_squares(",
            "diff_step=1.0e-5",
            "segment_collision_status(",
            "segment_start_q=prior_q",
            "minimum_joint_margin_rad",
            "critical_joint_indices",
            "inward_sign",
            "damped_task_nullspace_directions(",
            "suffix_seed_task_feature",
            '"nullspace_combined"',
            "prioritized_suffix_seed_indices(",
            "prioritized_suffix_rollout_indices(",
            "suffix_horizon_cache",
            "args.max_plan_joint_step_rad",
            "args.max_contact_penetration_mm",
            "args.min_arm_clearance_mm",
            "args.max_incidental_hand_penetration_mm",
            "planner_pad_alignment",
            "self_count == 0",
            "sample_distance\n                                            >= suffix_terminal_start_m",
            "np.all(\n                                                    sample_normal_error",
            "prospective_low_motion_failures(",
            "LOW_MOTION_DEFAULT_WINDOW_FRAMES",
            "prefix_frame_distance",
            "sample_collision_ok",
            "reachability.geometry_group_clearances(",
            "reachability.self_collision_contacts(",
            "sample_pad_alignment",
            "published_backtrack",
            "sample_palm_ok",
            'candidate_kind="suffix_horizon"',
            '"candidate_node_condition_ok"',
            '"candidate_publisher_first_failure_gate_ok"',
            '"candidate_low_motion_first_window"',
            "rollout_source_indices",
            "local_lower = np.maximum(",
            "suffix_rollout_prefix_rank(",
            "source_prefix_ok",
            'trial_kind="source_preserved"',
            '"extrapolated_ls"',
            "feasibility_weight_scale=4.0",
            '"rollout_partial_"',
            "[SUFFIX-ROLLOUT-PRUNED]",
            '"candidate_rollout_reached_node"',
            '"candidate_rollout_prune_node"',
            '"candidate_rollout_prune_reason"',
            '"candidate_rollout_attempt_count"',
            '"candidate_rollout_interior_polish_attempt_count"',
            '"candidate_rollout_interior_polish_max_scale"',
            '"candidate_rollout_explicit_polish_attempt_count"',
            '"candidate_rollout_explicit_polish_success_count"',
            '"candidate_rollout_explicit_polish_min_margin_m"',
            '"candidate_rollout_explicit_polish_source_q_rad"',
            '"candidate_rollout_explicit_polish_status"',
            '"candidate_rollout_explicit_polish_solver_success"',
            '"candidate_rollout_explicit_polish_prefix_ok"',
            '"candidate_rollout_explicit_polish_nfev"',
            '"candidate_rollout_explicit_polish_constraint_margin_m"',
            '"candidate_rollout_explicit_polish_q_rad"',
            '"candidate_rollout_explicit_polish_eps"',
            '"candidate_rollout_explicit_polish_restart_attempt_count"',
            '"candidate_rollout_explicit_polish_restart_status"',
            '"candidate_rollout_explicit_polish_restart_solver_success"',
            '"candidate_rollout_explicit_polish_restart_prefix_ok"',
            '"candidate_rollout_explicit_polish_restart_nfev"',
            '"candidate_rollout_explicit_polish_restart_constraint_margin_m"',
            '"candidate_rollout_explicit_polish_restart_q_rad"',
            '"candidate_rollout_explicit_polish_restart_eps"',
            '"candidate_rollout_explicit_polish_restart_objective_anchor_q_rad"',
            '"explicit_constraint_guard_m"',
            '"interior_polish_scale_ladder"',
            '"rollout_"',
        ):
            self.assertIn(required, horizon)
        self.assertIn(
            "progress_margin\n                                        >= task_guard_m",
            horizon,
        )
        self.assertNotIn(
            "progress_margin\n                                        >= solve_guard_m",
            horizon,
        )
        self.assertNotIn(
            "progress_margin\n                                        >= polish_guard_m",
            horizon,
        )
        rollout = horizon[horizon.index("rollout_source_indices") :]
        self.assertLess(
            rollout.index('trial_kind="source_preserved"'),
            rollout.index("local_result = least_squares("),
        )
        self.assertLess(
            rollout.index("if not source_prefix_ok:"),
            rollout.index("local_result = least_squares("),
        )
        self.assertLess(
            rollout.index("local_result = least_squares("),
            rollout.index("suffix_prefix_needs_interior_polish("),
        )
        self.assertLess(
            rollout.index("suffix_prefix_needs_interior_polish("),
            rollout.index("suffix_node_needs_explicit_task_polish("),
        )
        self.assertLess(
            rollout.index("suffix_node_needs_explicit_task_polish("),
            rollout.index("selected_node_trial = min("),
        )
        continuation = rollout[
            rollout.index("for (\n                                            polish_stage,") :
            rollout.index("selected_node_trial = min(")
        ]
        self.assertIn("if any(", continuation)
        self.assertIn("trial.prefix_ok", continuation)
        self.assertIn("break", continuation)
        self.assertIn("feasibility_weight_scale=(", continuation)
        self.assertIn("_polish_scale", continuation)
        for forbidden_mutation in (
            "coarse_q[keyframe] =",
            "coarse_progress[keyframe] =",
            "static_bridge_total_m +=",
            "recovery_bridge_total_m +=",
            "auto_rephase_offset_m =",
        ):
            self.assertNotIn(forbidden_mutation, horizon)
        solve_start = source.index(
            "suffix_horizon_candidate = (",
            end,
        )
        bridge_solve = source.index(
            "bridge_candidate = least_squares(",
            solve_start,
        )
        self.assertLess(solve_start, bridge_solve)
        selection = source[
            source.index("moving_bridge = min(", bridge_solve) :
            source.index("moving_bridge_points", bridge_solve)
        ]
        self.assertIn('== "suffix_horizon"', selection)
        self.assertIn("candidate.bridge_multistart_rank[0]", selection)
        commit = source.index("coarse_q[keyframe] = q", bridge_solve)
        self.assertLess(
            source.index(
                "coarse_suffix_horizon[keyframe] = suffix_horizon_selected",
                bridge_solve,
            ),
            commit,
        )
        self.assertIn("mpc_coarse_suffix_horizon=", source)
        self.assertIn("mpc_suffix_horizon_attempt_count=np.asarray(", source)
        self.assertIn("mpc_suffix_horizon_success_count=np.asarray(", source)
        fail_close = source[
            source.index("suffix_horizon_failed_closed = bool(") :
            source.index("desired_arc[:] = nominal_desired_arc", solve_start)
        ]
        self.assertIn("[SUFFIX-HORIZON-FAIL-CLOSED]", fail_close)
        self.assertIn(
            "and not suffix_horizon_failed_closed",
            fail_close,
        )
        self.assertIn(
            '"myopic_bridge_commit_allowed=False"',
            fail_close,
        )

    def test_terminal_contact_repair_is_separate_bounded_and_evidenced(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        start = source.index("def build_suffix_horizon_candidate")
        end = source.index("def moving_bridge_residual", start)
        horizon = source[start:end]
        tree = ast.parse(horizon)

        ordinary_start = horizon.index("ordinary_explicit_source_trials = [")
        repair_start = horizon.index(
            "terminal_contact_repair_source_trials = [", ordinary_start
        )
        selection_start = horizon.index(
            "explicit_repair_mode = bool(", repair_start
        )
        ordinary_entry = horizon[ordinary_start:repair_start]
        repair_entry = horizon[repair_start:selection_start]
        self.assertIn("suffix_prefix_needs_interior_polish(", ordinary_entry)
        self.assertIn(
            "suffix_node_needs_explicit_task_polish(", ordinary_entry
        )
        self.assertNotIn("suffix_terminal_contact_repair_required(", ordinary_entry)
        self.assertIn("suffix_terminal_contact_repair_required(", repair_entry)
        self.assertIn(
            "not ordinary_explicit_source_trials\n"
            "                                                and "
            "terminal_contact_repair_source_trials",
            horizon[selection_start : selection_start + 500],
        )

        support_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "suffix_explicit_support_indices"
        ]
        self.assertEqual(len(support_calls), 1)
        support_keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in support_calls[0].keywords
        }
        self.assertEqual(
            support_keywords["include_all_contacts"], "explicit_repair_mode"
        )
        self.assertEqual(
            support_keywords["required_motion_fingers"],
            "MOVING_BRIDGE_FORWARD_FINGER_COUNT",
        )

        headroom_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "explicit_bound_headroom_rad"
        ]
        self.assertEqual(len(headroom_assignments), 1)
        headroom_value = headroom_assignments[0].value
        self.assertIsInstance(headroom_value, ast.IfExp)
        self.assertEqual(ast.unparse(headroom_value.test), "explicit_repair_mode")
        self.assertEqual(ast.literal_eval(headroom_value.body), 1.0e-6)
        self.assertEqual(ast.literal_eval(headroom_value.orelse), 0.0)

        bounds_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Bounds"
        ]
        self.assertEqual(len(bounds_calls), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in bounds_calls[0].args],
            ["explicit_lower", "explicit_upper"],
        )
        self.assertIn(
            "explicit_lower = (\n"
            "                                                    local_lower\n"
            "                                                    + "
            "explicit_bound_headroom_rad",
            horizon,
        )
        self.assertIn(
            "if explicit_repair_mode\n"
            "                                                        else "
            "explicit_source_q.copy()",
            horizon,
        )
        self.assertIn(
            "if explicit_repair_mode\n"
            "                                                            else "
            "explicit_attempt_q.copy()",
            horizon,
        )
        self.assertIn(
            "suffix_terminal_contact_repair_restart_required(", horizon
        )
        self.assertIn("if not explicit_restart_required:", horizon)

        for evidence_name in (
            "mode",
            "source_q_rad",
            "source_contact_count",
            "source_normal_error_m",
            "motion_support_indices",
            "contact_support_indices",
            "bound_headroom_rad",
            "objective_anchor_q_rad",
        ):
            self.assertIn(
                '"candidate_rollout_terminal_contact_repair_'
                f'{evidence_name}"',
                horizon,
            )
        # Solver headroom never replaces the original exact audit.
        audit_start = horizon.index("def audit_suffix(")
        audit_end = horizon.index(
            "horizon_candidates: list[SimpleNamespace]", audit_start
        )
        exact_audit = horizon[audit_start:audit_end]
        self.assertIn("q_node - lower", exact_audit)
        self.assertIn("upper - q_node", exact_audit)
        self.assertNotIn("explicit_lower", exact_audit)
        self.assertNotIn("explicit_upper", exact_audit)

    def test_suffix_explicit_restart_is_once_reanchored_and_reuses_solve(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        start = source.index("def build_suffix_horizon_candidate")
        end = source.index("def moving_bridge_residual", start)
        horizon = source[start:end]
        tree = ast.parse(horizon)
        restart_loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "explicit_attempt_index"
        ]
        self.assertEqual(len(restart_loops), 1)
        restart_loop = restart_loops[0]
        self.assertEqual(ast.unparse(restart_loop.iter), "range(2)")

        loop_calls = [
            node for node in ast.walk(restart_loop) if isinstance(node, ast.Call)
        ]
        minimize_calls = [
            node
            for node in loop_calls
            if isinstance(node.func, ast.Name) and node.func.id == "minimize"
        ]
        restart_gate_calls = [
            node
            for node in loop_calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "suffix_explicit_restart_required"
        ]
        repair_restart_gate_calls = [
            node
            for node in loop_calls
            if isinstance(node.func, ast.Name)
            and node.func.id
            == "suffix_terminal_contact_repair_restart_required"
        ]
        self.assertEqual(len(minimize_calls), 1)
        self.assertEqual(len(restart_gate_calls), 1)
        self.assertEqual(len(repair_restart_gate_calls), 1)
        minimize_call = minimize_calls[0]
        self.assertEqual(ast.unparse(minimize_call.args[1]), "explicit_attempt_q")
        minimize_keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in minimize_call.keywords
        }
        self.assertEqual(minimize_keywords["bounds"], "explicit_bounds")
        self.assertEqual(
            minimize_keywords["constraints"],
            "explicit_constraint_specs",
        )
        self.assertEqual(minimize_keywords["options"], "explicit_options")

        loop_source = ast.get_source_segment(horizon, restart_loop)
        assert loop_source is not None
        self.assertNotIn("if explicit_result.success", loop_source)
        self.assertIn("if explicit_trial_prefix_ok", loop_source)
        self.assertIn("rollout_explicit_polish_attempt_count += 1", loop_source)
        self.assertIn("rollout_nfev += explicit_nfev", loop_source)
        self.assertIn('"explicit_constraint_restart"', loop_source)
        self.assertIn('f"eps={explicit_attempt_eps:.1e} "', loop_source)

        restart_break_ifs = [
            node
            for node in ast.walk(restart_loop)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and ast.unparse(node.test.operand) == "explicit_restart_required"
        ]
        self.assertEqual(len(restart_break_ifs), 1)
        self.assertEqual(len(restart_break_ifs[0].body), 1)
        self.assertIsInstance(restart_break_ifs[0].body[0], ast.Break)
        for restart_gate_call in (
            *restart_gate_calls,
            *repair_restart_gate_calls,
        ):
            gate_keywords = {
                keyword.arg for keyword in restart_gate_call.keywords
            }
            self.assertNotIn("status", gate_keywords)
            self.assertNotIn("success", gate_keywords)
        control_tests = [
            ast.unparse(node.test)
            for node in ast.walk(restart_loop)
            if isinstance(node, (ast.If, ast.While))
        ]
        self.assertFalse(
            any(
                "explicit_result.status" in test
                or "explicit_result.success" in test
                for test in control_tests
            )
        )

        loop_assignments = [
            node
            for node in ast.walk(restart_loop)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ]
        assignment_values = {
            node.targets[0].id: ast.unparse(node.value)
            for node in loop_assignments
        }
        self.assertEqual(
            assignment_values["explicit_objective_anchor_q"],
            "np.clip(explicit_attempt_q, explicit_lower, explicit_upper) "
            "if explicit_repair_mode else explicit_attempt_q.copy()",
        )
        self.assertEqual(
            assignment_values["explicit_attempt_q"],
            "np.clip(explicit_q, explicit_lower, explicit_upper) "
            "if explicit_repair_mode else explicit_q.copy()",
        )
        self.assertEqual(
            assignment_values["explicit_attempt_eps"],
            "float(explicit_options['eps'])",
        )
        self.assertIn(
            "(explicit_q - explicit_source_q) / objective_scale",
            assignment_values["explicit_selection_delta"],
        )
        objective_functions = [
            node
            for node in ast.walk(restart_loop)
            if isinstance(node, ast.FunctionDef)
            and node.name == "rollout_node_explicit_objective"
        ]
        self.assertEqual(len(objective_functions), 1)
        objective = objective_functions[0]
        self.assertEqual(
            ast.unparse(objective.args.kw_defaults[0]),
            "explicit_objective_anchor_q",
        )
        self.assertEqual(
            ast.unparse(objective.args.kw_defaults[1]),
            "objective_scale",
        )
        self.assertIn("q_node - _anchor_q", ast.unparse(objective))

        loop_offset = horizon.index("for explicit_attempt_index")
        for frozen_setup in (
            "motion_constraint_indices",
            "contact_constraint_indices",
            "def rollout_node_explicit_constraints",
            "explicit_bounds = Bounds",
            "explicit_constraint_specs =",
            "explicit_options =",
        ):
            self.assertLess(horizon.index(frozen_setup), loop_offset)
        constraint_start = horizon.index("def rollout_node_explicit_constraints")
        constraint_end = horizon.index("if explicit_masks_valid", constraint_start)
        constraint_source = horizon[constraint_start:constraint_end]
        self.assertIn("explicit_constraint_guard_m", constraint_source)
        self.assertIn("motion_constraint_indices", constraint_source)
        self.assertIn("contact_constraint_indices", constraint_source)
        options_source = horizon[
            horizon.index("explicit_options =") : loop_offset
        ]
        self.assertIn("args.mpc_suffix_max_nfev", options_source)
        self.assertIn("100", options_source)
        self.assertIn('"ftol": 1.0e-12', options_source)
        self.assertIn('"eps": 1.0e-5', options_source)

        explicit_options_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "explicit_options"
        ]
        self.assertEqual(len(explicit_options_assignments), 1)
        explicit_options_value = explicit_options_assignments[0].value
        self.assertIsInstance(explicit_options_value, ast.Dict)
        explicit_option_values = {
            ast.literal_eval(key): value
            for key, value in zip(
                explicit_options_value.keys,
                explicit_options_value.values,
                strict=True,
            )
        }
        self.assertEqual(
            set(explicit_option_values),
            {"maxiter", "ftol", "eps", "disp"},
        )
        self.assertEqual(
            ast.literal_eval(explicit_option_values["eps"]),
            1.0e-5,
        )
        self.assertEqual(
            ast.literal_eval(explicit_option_values["ftol"]),
            1.0e-12,
        )
        self.assertEqual(
            ast.unparse(explicit_option_values["maxiter"]),
            "min(args.mpc_suffix_max_nfev, 100)",
        )

        option_mutations = [
            node
            for node in ast.walk(restart_loop)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "explicit_options"
        ]
        self.assertEqual(len(option_mutations), 1)
        eps_mutation = option_mutations[0]
        self.assertEqual(ast.literal_eval(eps_mutation.targets[0].slice), "eps")
        self.assertEqual(ast.literal_eval(eps_mutation.value), 1.0e-6)
        restart_eps_guards = [
            node
            for node in ast.walk(restart_loop)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "explicit_attempt_index > 0"
            and eps_mutation in ast.walk(node)
        ]
        self.assertEqual(len(restart_eps_guards), 1)

        self.assertLess(
            loop_source.index("explicit_result = minimize("),
            loop_source.index(
                "explicit_restart_required = "
                "suffix_terminal_contact_repair_restart_required("
            ),
        )
        self.assertLess(
            loop_source.index("if not explicit_restart_required:"),
            loop_source.rindex("explicit_attempt_q ="),
        )

    def test_failure_prefix_receives_last_suffix_horizon_evidence(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        failure_start = source.index("def raise_adaptive_planner_failure")
        failure_end = source.index("def insert_auto_refinement", failure_start)
        failure = source[failure_start:failure_end]
        self.assertIn("last_suffix_horizon_evidence", failure)
        self.assertIn('f"last_suffix_horizon_{evidence_name}"', failure)
        compact_failure = "".join(failure.split())
        palm_error_initializers = [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "coarse_palm_position_error"
                for target in node.targets
            )
        ]
        self.assertIn(
            "np.zeros(keyframe_count + 1, dtype=np.float64)",
            tuple(ast.unparse(value) for value in palm_error_initializers),
        )
        for expected in (
            "coarse_provenance=",
            '"auto_rephase_offset_m"',
            "coarse_auto_rephase_offset_m[:keyframe].copy()",
            '"target_progress_m"',
            "coarse_target_progress[:keyframe].copy()",
            "coarse_feasibility_bridge[:keyframe].copy()",
            "coarse_suffix_horizon[:keyframe].copy()",
            "coarse_static_bridge_dwell_m[:keyframe].copy()",
            "coarse_recovery_bridge[:keyframe].copy()",
            "refinement_provenance=",
            "auto_refine_inserted_distance_m",
            "auto_refine_inserted_reason",
            "rolling_provenance=",
            "planner_frame_target_distance.copy()",
            "LOW_MOTION_REQUIRED_FORWARD_FINGERS",
        ):
            self.assertIn(expected, failure)
        for expected in (
            "coarse_auto_rephase_offset_m[:keyframe].copy()",
            "coarse_progress[:keyframe].copy()",
            "coarse_target_progress[:keyframe].copy()",
            "coarse_feasibility_bridge[:keyframe].copy()",
            "coarse_suffix_horizon[:keyframe].copy()",
            "coarse_static_feasibility_bridge[:keyframe].copy()",
            "coarse_static_bridge_dwell_m[:keyframe].copy()",
            "coarse_recovery_bridge[:keyframe].copy()",
            "coarse_recovery_bridge_dwell_m[:keyframe].copy()",
            "coarse_normal_error[:keyframe].copy()",
            "coarse_palm_target[:keyframe].copy()",
            "coarse_palm_position_error[:keyframe].copy()",
            "coarse_cost[:keyframe].copy()",
            "coarse_nfev[:keyframe].copy()",
        ):
            self.assertIn(expected, compact_failure)
        for expected in (
            '"candidate_q_rad"',
            '"seed_kind"',
            '"node_condition_names"',
            '"candidate_node_metric_margin_m"',
            '"candidate_node_metric_margin_rad"',
            '"candidate_publisher_first_failure_distance_m"',
            '"candidate_rollout_explicit_polish_source_q_rad"',
            '"candidate_rollout_explicit_polish_eps"',
            '"candidate_rollout_explicit_polish_restart_attempt_count"',
            '"candidate_rollout_explicit_polish_restart_status"',
            '"candidate_rollout_explicit_polish_restart_solver_success"',
            '"candidate_rollout_explicit_polish_restart_prefix_ok"',
            '"candidate_rollout_explicit_polish_restart_nfev"',
            '"candidate_rollout_explicit_polish_restart_constraint_margin_m"',
            '"candidate_rollout_explicit_polish_restart_q_rad"',
            '"candidate_rollout_explicit_polish_restart_eps"',
            '"candidate_rollout_explicit_polish_restart_objective_anchor_q_rad"',
            '"selected_index"',
        ):
            self.assertIn(expected, source)

    def test_level2_runner_freezes_suffix_horizon_contract(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        for expected in (
            '"--mpc-suffix-horizon-nodes", "5"',
            '"--mpc-suffix-min-joint-margin-mrad", "0.5"',
            '"--mpc-suffix-min-task-margin-mm", "0.05"',
            '"--mpc-suffix-max-nfev", "160"',
        ):
            self.assertIn(expected, source)

    def test_bridge_failure_diagnostics_and_prefix_are_failure_scoped(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        diagnostics_source = DIAGNOSTICS_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count('"[BRIDGE-REJECTION] "'), 1)
        for required_term in (
            "evaluate_bridge_conditions",
            "make_bridge_rejection_record",
            "format_bridge_rejection_record",
            "not moving_bridge_hard_ok",
            "not moving_recovery_hard_ok",
            "raise_adaptive_planner_failure",
            "save_mpc_failure_prefix",
            "RejectedMovingBridgeCandidate",
            "q_rad=moving_bridge.x.copy()",
            "points_m=moving_bridge_points.copy()",
            "arcs_m=moving_bridge_arc.copy()",
            "last_rejected_moving_bridge",
            "final_best_q=best.x",
            "[MPC-FAILURE-PREFIX]",
            "--mpc-failure-prefix-output",
        ):
            self.assertIn(required_term, source)
        for required_metric in (
            '"strict_progress_margin_m"',
            '"recovery_progress_margin_m"',
            '"normal_margin_m"',
            '"tangential_margin_m"',
            '"monotonic_margin_m"',
            '"palm_margin_m"',
            '"arm_clearance_margin_m"',
            '"joint_min_margin_rad"',
            '"recovery_dwell_margin_m"',
            '"recovery_total_margin_m"',
            '"self_collision_unique_pair_count"',
            '"self_collision_sample_occurrence_count"',
            '"protected_self_clearance_margin_m"',
            '"protected_self_nearest_pair"',
        ):
            self.assertIn(required_metric, diagnostics_source)
        for explicit_snapshot_field in (
            '"failure_final_best_q_rad"',
            '"failure_final_best_points_m"',
            '"failure_final_best_arcs_m"',
            '"failure_final_best_desired_arcs_m"',
            '"bridge_rejected_q_rad"',
            '"bridge_rejected_points_m"',
            '"bridge_rejected_arcs_m"',
            '"bridge_rejected_desired_arcs_m"',
        ):
            self.assertIn(explicit_snapshot_field, diagnostics_source)
        self.assertNotIn('"failure_candidate_q_rad"', diagnostics_source)
        self.assertLess(
            source.index('"--mpc-failure-prefix-output"'),
            source.index("args = parser.parse_args()"),
        )
        for failure_reason in (
            "palm_drift",
            "monotonic_progress",
            "longitudinal_progress",
            "fingertip_support",
            "tangential_gait",
            "contact_policy",
            "unmarked_low_motion",
        ):
            self.assertIn(
                f'reason="{failure_reason}"',
                source,
            )
        coarse_failure_start = source.index(
            "candidate_failure_metrics ="
        )
        coarse_failure_end = source.index(
            "q = best.x",
            coarse_failure_start,
        )
        coarse_failure_source = source[
            coarse_failure_start:coarse_failure_end
        ]
        self.assertEqual(
            coarse_failure_source.count("raise_adaptive_planner_failure("),
            7,
        )
        self.assertNotIn("raise RuntimeError(", coarse_failure_source)
        self.assertIn("coarse-shooting prefix", source)

    def test_palm_guide_multistart_uses_guide_drift_not_three_mm_ball(
        self,
    ) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        seed_start = source.index(
            "if args.palm_guide_only:",
            source.index("palm_multistart_surface_seeds"),
        )
        guide_branch_end = source.index("else:", seed_start)
        guide_branch = source[seed_start:guide_branch_end]
        self.assertIn(
            "build_palm_guide_multistart_specs",
            guide_branch,
        )
        self.assertIn("args.palm_guide_max_drift_mm", guide_branch)
        self.assertNotIn(
            "args.mpc_palm_position_tolerance_mm",
            guide_branch,
        )
        self.assertIn("shifted_arm_seed[:ARM_DOF]", source)
        self.assertIn("segment_collision_status", source)

    def test_recovery_bridge_is_moving_bounded_and_auditable(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        for required_term in (
            "MOVING-RECOVERY-BRIDGE",
            "moving_recovery_hard_ok",
            "moving_bridge_motion_ok",
            "moving_tip_motion_m",
            "moving_bridge_collision_ok",
            "moving_bridge_joint_limits_ok",
            "moving_bridge_monotonic_error",
            "moving_bridge_palm_error",
            "recovery_bridge_budget_ok",
            "mpc_recovery_bridge_max_span_mm",
            "mpc_recovery_bridge_max_total_ratio",
            "mpc_recovery_bridge_progress_tolerance_mm",
            "mpc_recovery_bridge_normal_tolerance_mm",
            "mpc_recovery_bridge_min_contact_fingers",
            "mpc_recovery_bridge_terminal_margin_mm",
            "coarse_recovery_bridge",
            "coarse_recovery_bridge_dwell_m",
            "recovery_bridge_total_m",
            "recovery_bridge_mask_plan",
            "mpc_coarse_recovery_bridge",
            "mpc_coarse_recovery_bridge_dwell_m",
            "mpc_recovery_bridge_total_m",
            "planned_majority_contact_ratio",
            "planned_average_contact_fingers",
            "planned_contact_ratio",
        ):
            self.assertIn(required_term, source)

    def test_fixed_palm_frame_path_defines_transport_frame(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "else:\n"
            "                    transported_contact_frame = "
            "initial_contact_frame\n"
            "                    desired_palm_rotation = "
            "initial_palm_rotation",
            source,
        )

    def test_runtime_contact_acceptance_is_aggregate_and_bounded(self) -> None:
        source = DEMO_PATH.read_text(encoding="utf-8")
        for required_term in (
            "majority_contact_frames",
            "simultaneous_contact_sum",
            "majority_contact_ratio",
            "average_simultaneous_contacts",
            "min_majority_contact_ratio",
            "min_average_contact_fingers",
            "max_zero_contact_frames",
            "max_zero_contact_streak",
            "max_bad_contact_streak",
            "final_all_contact_streak",
        ):
            self.assertIn(required_term, source)
        self.assertNotIn(
            "Minimum simultaneous contact count was below",
            source,
        )
        self.assertNotIn(
            "Incidental palm/finger-link contact displaced",
            source,
        )

    def test_headless_execution_does_not_render_or_encode(self) -> None:
        source_text = DEMO_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        main = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "main"
        )
        headless_branch = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "args.viewer == 'headless'"
        )
        branch_source = ast.get_source_segment(
            source_text,
            headless_branch,
        )
        self.assertIsNotNone(branch_source)
        self.assertIn("wrapped.step(action)", branch_source)
        headless_body_source = "\n".join(
            ast.get_source_segment(source_text, node) or ""
            for node in headless_branch.body
        )
        self.assertNotIn("env.render", headless_body_source)
        self.assertNotIn("imageio.get_writer", headless_body_source)


if __name__ == "__main__":
    unittest.main()
