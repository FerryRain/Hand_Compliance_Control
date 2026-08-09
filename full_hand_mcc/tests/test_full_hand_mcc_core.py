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


class PlannerDiagnosticsTest(unittest.TestCase):
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
            "moving_bridge_residual",
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
