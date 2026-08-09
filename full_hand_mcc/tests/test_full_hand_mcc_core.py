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
            "budget_values": {"recovery_remaining_m": 0.001},
            "failure_metrics": {
                "progress_error_m": np.asarray([0.0, 0.006])
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
                2,
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


class AdaptiveMPCSourceStructureTest(unittest.TestCase):
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
            "central_difference_clearance_gradient",
            "self_separation_ascent_seeds",
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
            "moving_bridge = least_squares",
            "bridge_lower",
            "bridge_upper",
            "moving_bridge_motion_ok",
            "moving_tip_motion_m",
            "moving_progressing_finger_count",
            "progressing_finger_count_required",
            "bridge_active_fingers",
            "minimum_tip_motion_m",
            "moving_bridge_target_arc",
            "bounded_incremental_arc_targets",
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
            "moving_bridge = least_squares",
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
        self.assertIn("moving_separation_seeds", source)
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
            6,
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
