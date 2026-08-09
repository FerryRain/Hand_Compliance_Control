from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "full_hand_mcc/scripts/audit_level2_plan.py"
SPEC = importlib.util.spec_from_file_location("audit_level2_plan", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _capsule_projection(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    local = (points - center) @ rotation
    axis = np.zeros_like(local)
    axis[:, 2] = np.clip(local[:, 2], -half_height, half_height)
    radial = local - axis
    lengths = np.linalg.norm(radial, axis=1, keepdims=True)
    normals_local = radial / np.maximum(lengths, 1.0e-12)
    surface_local = axis + radius * normals_local
    return center + surface_local @ rotation.T, normals_local @ rotation.T


class _FakeReachabilitySolver:
    def __init__(
        self,
        q_plan: np.ndarray,
        points: np.ndarray,
        pad_normals: np.ndarray,
    ) -> None:
        self.q_plan = np.asarray(q_plan, dtype=np.float64)
        self.points = np.asarray(points, dtype=np.float64)
        self.pad_normals = np.asarray(pad_normals, dtype=np.float64)
        self.lower = np.full(23, -1.0)
        self.upper = np.full(23, 1.0)

    def _frame(self, q: np.ndarray) -> int:
        return int(np.argmin(np.abs(self.q_plan[:, 0] - float(q[0]))))

    def forward_points(self, q: np.ndarray) -> np.ndarray:
        return self.points[self._frame(q)].copy()

    def fingertip_pad_normals(self, q: np.ndarray) -> np.ndarray:
        return self.pad_normals[self._frame(q)].copy()


def _make_valid_inputs(
    frame_count: int = 61,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], _FakeReachabilitySolver]:
    radius = 0.10
    half_height = 0.17
    route_length = 0.48
    target = np.linspace(0.0, route_length, frame_count + 1)[1:]
    arc = 0.16 + target
    lower_join = 0.5 * np.pi * radius
    upper_join = lower_join + 2.0 * half_height
    x = np.full(frame_count, radius)
    z = arc - lower_join - half_height
    upper = arc > upper_join
    upper_angle = (arc[upper] - upper_join) / radius
    x[upper] = radius * np.cos(upper_angle)
    z[upper] = half_height + radius * np.sin(upper_angle)
    path = np.stack((x, np.zeros_like(x), z), axis=1)
    surface = np.repeat(path[:, None, :], 5, axis=1)
    seed_path = np.asarray(
        [radius, 0.0, 0.16 - lower_join - half_height],
        dtype=np.float64,
    )
    seed_surface = np.repeat(seed_path[None, :], 5, axis=0)
    _, outward = _capsule_projection(
        surface.reshape(-1, 3),
        np.zeros(3),
        np.eye(3),
        radius,
        half_height,
    )
    outward = outward.reshape(frame_count, 5, 3)
    pad_normals = -outward[:, 1:]
    _, seed_outward = _capsule_projection(
        seed_surface,
        np.zeros(3),
        np.eye(3),
        radius,
        half_height,
    )
    seed_pad_normals = -seed_outward[1:]

    q_plan = np.zeros((frame_count, 23), dtype=np.float64)
    q_plan[:, 0] = np.linspace(-0.10, 0.10, frame_count)
    seed_q = q_plan[0].copy()
    seed_q[0] -= q_plan[1, 0] - q_plan[0, 0]
    progress = np.zeros((frame_count, 5), dtype=np.float64)
    progress[:] = target[:, None]
    scheduled_mask = np.ones((frame_count, 4), dtype=np.bool_)
    coarse_distance = np.asarray([0.0, route_length])
    coarse_false = np.zeros(2, dtype=np.bool_)
    initial_q = seed_q.copy()
    maximum_step = float(
        np.max(np.abs(np.diff(np.vstack((initial_q[None], q_plan)), axis=0)))
    )
    plan = {
        "surface_points_m": surface,
        "kinematic_points_m": surface.copy(),
        "joint_positions_rad": q_plan,
        "progress_m": progress,
        "progress_residual_m": np.zeros_like(progress),
        "normal_error_m": np.zeros_like(progress),
        "scheduled_contact_mask": scheduled_mask,
        "scheduled_contact_count": np.full(frame_count, 4, dtype=np.int8),
        "recovery_bridge_mask": np.zeros(frame_count, dtype=np.bool_),
        "mpc_coarse_distance_m": coarse_distance,
        "mpc_coarse_static_feasibility_bridge": coarse_false.copy(),
        "mpc_coarse_recovery_bridge": coarse_false.copy(),
        "coarse_joint_positions_rad": np.vstack((seed_q, q_plan[-1])),
        "coarse_progress_m": np.vstack(
            (np.zeros(5), np.full(5, route_length))
        ),
        "mpc_recovery_bridge_min_contact_fingers": np.asarray(2),
        "final_contact_recovery_frames": np.asarray(20),
        "min_planner_contact_fingers": np.asarray(3),
        "transient_contact_finger": np.asarray(0),
        "transient_contact_start_m": np.asarray(0.10),
        "transient_contact_end_m": np.asarray(0.20),
        "object_shape": np.asarray("capsule"),
        "object_radius_m": np.asarray(radius),
        "object_half_height_m": np.asarray(half_height),
        "max_joint_step_rad": np.asarray(maximum_step),
        "axial_distance_m": target.copy(),
        "axial_direction": np.asarray(1),
        "start_surface_local_m": surface[0].copy(),
        "end_surface_local_m": surface[-1].copy(),
    }
    grasp = {
        "joint_position_rad": initial_q,
        "object_center_m": np.zeros(3),
        "object_rotation": np.eye(3),
    }
    solver = _FakeReachabilitySolver(
        np.vstack((seed_q, q_plan)),
        np.concatenate((seed_surface[None], surface), axis=0),
        np.concatenate((seed_pad_normals[None], pad_normals), axis=0),
    )
    return plan, grasp, solver


class Level2PlanAuditTest(unittest.TestCase):
    def test_load_rejects_object_dtype_without_pickle(self) -> None:
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.files = ["safe", "unsafe"]
        archive.__getitem__.side_effect = (
            np.asarray([1.0]),
            ValueError("Object arrays cannot be loaded when allow_pickle=False"),
        )
        with (
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(AUDIT.np, "load", return_value=archive) as load,
        ):
            with self.assertRaisesRegex(
                AUDIT.AuditInputError,
                "allow_pickle=False",
            ):
                AUDIT.load_npz_no_pickle(Path("unsafe.npz"))
        load.assert_called_once_with(Path("unsafe.npz"), allow_pickle=False)

    def test_synthetic_plan_passes_all_three_conclusions(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        report = AUDIT.audit_loaded_plan(
            plan,
            grasp,
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertEqual(
            report["conclusions"],
            {
                "SCHEMA": "PASS",
                "DIAGNOSTIC": "PASS",
                "LEVEL2_GEOMETRY": "PASS",
            },
        )
        self.assertEqual(report["unmarked_low_motion"]["region_count"], 0)
        self.assertLess(
            report["kinematics"]["pad_angle"]["maximum_deg"],
            1.0e-5,
        )
        self.assertGreater(
            report["kinematics"]["joint_limit"]["minimum_margin_rad"],
            0.0,
        )

    def test_low_motion_window_is_ignored_only_when_bridge_marked(self) -> None:
        plan, _, _ = _make_valid_inputs()
        target = AUDIT.reconstruct_frame_target_distance(
            len(plan["progress_m"]),
            plan["mpc_coarse_distance_m"],
        )
        progress = plan["progress_m"].copy()
        progress[10:41, 1:] = progress[10, 1:]
        unmarked = AUDIT.find_unmarked_low_motion_windows(
            progress,
            plan["kinematic_points_m"],
            target,
            np.zeros(len(target), dtype=bool),
            plan["axial_distance_m"],
            window_frames=20,
        )
        self.assertGreater(len(unmarked), 0)
        self.assertLess(
            unmarked[0]["worst_window"]["forward_finger_count"],
            3,
        )
        marked = np.zeros(len(target), dtype=bool)
        marked[10:41] = True
        excluded = AUDIT.find_unmarked_low_motion_windows(
            progress,
            plan["kinematic_points_m"],
            target,
            marked,
            plan["axial_distance_m"],
            window_frames=20,
        )
        self.assertEqual(excluded, [])

    def test_overlapping_low_motion_windows_merge_despite_unflagged_gap(self) -> None:
        frame_count = 23
        target = np.arange(frame_count, dtype=np.float64) * 0.001
        progress = np.zeros((frame_count, 5), dtype=np.float64)
        progress[:, 1] = target
        progress[:, 2] = target
        progress[21, 3] = 0.003
        regions = AUDIT.find_unmarked_low_motion_windows(
            progress,
            np.zeros((frame_count, 5, 3), dtype=np.float64),
            target,
            np.zeros(frame_count, dtype=bool),
            target,
            window_frames=20,
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["frame_start"], 0)
        self.assertEqual(regions[0]["frame_end"], 22)
        self.assertEqual(regions[0]["overlapping_window_count"], 2)

    def test_schema_rejects_inconsistent_frame_recovery_mask(self) -> None:
        plan, grasp, _ = _make_valid_inputs()
        plan["recovery_bridge_mask"][7] = True
        schema = AUDIT.validate_schema(plan, grasp, window_frames=20)
        self.assertFalse(schema["passed"])
        self.assertTrue(
            any("recovery_bridge_mask disagrees" in error for error in schema["errors"])
        )

    def test_schema_rejects_nonfinite_and_nonscalar_configuration(self) -> None:
        plan, grasp, _ = _make_valid_inputs()
        plan["object_radius_m"] = np.asarray(np.nan)
        plan["max_joint_step_rad"] = np.asarray([0.01])
        plan["start_surface_local_m"] = np.full((5, 3), "bad", dtype=np.str_)
        plan["progress_residual_m"][3, 2] = -0.001
        schema = AUDIT.validate_schema(plan, grasp, window_frames=20)
        self.assertFalse(schema["passed"])
        self.assertTrue(
            any("object_radius_m contains NaN or Inf" in error for error in schema["errors"])
        )
        self.assertTrue(
            any("max_joint_step_rad" in error and "scalar" in error for error in schema["errors"])
        )
        self.assertTrue(
            any("start_surface_local_m must have a numeric dtype" in error for error in schema["errors"])
        )
        self.assertTrue(
            any("progress_residual_m must be non-negative" in error for error in schema["errors"])
        )

    def test_bridge_mapping_matches_right_endpoint_boundary_semantics(self) -> None:
        target = np.asarray([0.10, 0.48])
        coarse = np.asarray([0.0, 0.10, 0.30, 0.48])
        static, recovery = AUDIT.derive_frame_bridge_masks(
            target,
            coarse,
            np.asarray([False, True, False, True]),
            np.asarray([False, False, True, True]),
        )
        np.testing.assert_array_equal(static, np.asarray([False, True]))
        np.testing.assert_array_equal(recovery, np.asarray([True, True]))

    def test_nominal_three_of_four_runs_are_classified_per_finger(self) -> None:
        plan, _, _ = _make_valid_inputs()
        target = AUDIT.reconstruct_frame_target_distance(
            len(plan["progress_m"]),
            plan["mpc_coarse_distance_m"],
        )
        mask = plan["scheduled_contact_mask"].copy()
        mask[5:30, 0] = False
        count = np.count_nonzero(mask, axis=1)
        expected = AUDIT.audit_nominal_three_of_four_runs(
            mask,
            count,
            plan["recovery_bridge_mask"],
            target,
            transient_finger=0,
            transient_start_m=float(target[4]),
            transient_end_m=float(target[30]),
            minimum_run_frames=20,
        )
        self.assertEqual(expected["unexpected_long_run_count"], 0)
        self.assertEqual(
            expected["per_finger"]["if_tip"][0]["classification"],
            "configured_transient",
        )

        mask = plan["scheduled_contact_mask"].copy()
        mask[5:30, 1] = False
        count = np.count_nonzero(mask, axis=1)
        unexpected = AUDIT.audit_nominal_three_of_four_runs(
            mask,
            count,
            plan["recovery_bridge_mask"],
            target,
            transient_finger=0,
            transient_start_m=float(target[4]),
            transient_end_m=float(target[30]),
            minimum_run_frames=20,
        )
        self.assertEqual(unexpected["unexpected_long_run_count"], 1)
        self.assertEqual(
            unexpected["unexpected_long_runs"][0]["finger"],
            "mf_tip",
        )

    def test_geometry_rejects_recovery_in_final_twenty_millimeters(self) -> None:
        plan, grasp, _ = _make_valid_inputs()
        target = AUDIT.reconstruct_frame_target_distance(
            len(plan["progress_m"]),
            plan["mpc_coarse_distance_m"],
        )
        recovery = np.zeros(len(target), dtype=bool)
        recovery[target >= target[-1] - 0.020] = True
        geometry = AUDIT.audit_level2_geometry(
            plan,
            grasp,
            target,
            recovery,
        )
        self.assertFalse(geometry["passed"])
        self.assertFalse(
            geometry["checks"]["final_20mm_has_no_recovery"]
        )
        self.assertGreater(geometry["final_tail_recovery_frame_count"], 0)

    def test_geometry_requires_directed_crossing_canonical_route_and_progress(self) -> None:
        plan, grasp, _ = _make_valid_inputs()
        frame_count = len(plan["surface_points_m"])
        target = AUDIT.reconstruct_frame_target_distance(
            frame_count, plan["mpc_coarse_distance_m"]
        )
        order = np.concatenate(
            (
                np.linspace(frame_count - 1, 0, 31, dtype=int),
                np.linspace(0, frame_count - 1, 31, dtype=int)[1:],
            )
        )
        plan["surface_points_m"] = plan["surface_points_m"][order]
        plan["start_surface_local_m"] = plan["surface_points_m"][0].copy()
        plan["end_surface_local_m"] = plan["surface_points_m"][-1].copy()
        geometry = AUDIT.audit_level2_geometry(
            plan,
            grasp,
            target,
            plan["recovery_bridge_mask"],
        )
        self.assertFalse(geometry["checks"]["all_tips_cross_upper_seam"])
        self.assertEqual(
            geometry["maximum_saved_keyframe_backtracking_step_m"], 0.0
        )
        self.assertGreater(
            geometry["maximum_interpolated_backtracking_step_m"], 0.0002
        )

        plan, grasp, _ = _make_valid_inputs()
        plan["mpc_coarse_distance_m"][-1] = 0.20
        target = AUDIT.reconstruct_frame_target_distance(
            len(plan["surface_points_m"]), plan["mpc_coarse_distance_m"]
        )
        plan["progress_m"][-1, 1:] = 0.465
        plan["progress_residual_m"][-1, 1:] = 0.015
        geometry = AUDIT.audit_level2_geometry(
            plan,
            grasp,
            target,
            plan["recovery_bridge_mask"],
        )
        self.assertFalse(geometry["checks"]["canonical_route_length"])
        self.assertFalse(
            geometry["checks"]["terminal_planned_tip_progress_reaches_route"]
        )
        self.assertFalse(
            geometry["checks"][
                "terminal_planned_tip_progress_residual_within_4mm"
            ]
        )

        plan, grasp, _ = _make_valid_inputs()
        plan["progress_m"][-1, 1:] = 0.490
        plan["progress_residual_m"][-1, 1:] = -0.010
        target = AUDIT.reconstruct_frame_target_distance(
            len(plan["surface_points_m"]), plan["mpc_coarse_distance_m"]
        )
        geometry = AUDIT.audit_level2_geometry(
            plan,
            grasp,
            target,
            plan["recovery_bridge_mask"],
        )
        self.assertFalse(
            geometry["checks"]["terminal_planned_tip_progress_reaches_route"]
        )
        self.assertFalse(
            geometry["checks"]["terminal_progress_residual_is_consistent"]
        )

    def test_ordinary_two_of_four_support_is_never_a_valid_exception(self) -> None:
        plan, _, _ = _make_valid_inputs()
        target = AUDIT.reconstruct_frame_target_distance(
            len(plan["surface_points_m"]), plan["mpc_coarse_distance_m"]
        )
        mask = plan["scheduled_contact_mask"].copy()
        mask[:10, :2] = False
        support = AUDIT.audit_nominal_support_policy(
            mask,
            np.count_nonzero(mask, axis=1),
            plan["recovery_bridge_mask"],
            target,
            transient_finger=0,
            transient_start_m=0.10,
            transient_end_m=0.20,
            minimum_planner_contacts=3,
            minimum_recovery_contacts=2,
        )
        self.assertFalse(support["passed"])
        self.assertEqual(support["violation_frame_count"], 10)
        self.assertEqual(support["violation_regions"][0]["region"], "ordinary")

    def test_acceptance_requires_saved_and_observed_half_second_terminal_support(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        report = AUDIT.audit_loaded_plan(
            plan,
            grasp,
            mode="Acceptance",
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertEqual(report["conclusions"]["DIAGNOSTIC"], "FAIL")
        self.assertFalse(
            report["terminal_nominal_support"]["checks"][
                "saved_configuration_covers_mode_requirement"
            ]
        )
        self.assertEqual(report["terminal_nominal_support"]["required_frames"], 50)

        plan["final_contact_recovery_frames"] = np.asarray(50)
        plan["scheduled_contact_mask"][-50, 0] = False
        plan["scheduled_contact_count"] = np.count_nonzero(
            plan["scheduled_contact_mask"], axis=1
        ).astype(np.int8)
        report = AUDIT.audit_loaded_plan(
            plan,
            grasp,
            mode="Acceptance",
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertFalse(
            report["terminal_nominal_support"]["checks"][
                "terminal_nominal_support_is_four_of_four"
            ]
        )

    def test_diagnostic_and_acceptance_pad_limits_are_not_mixed(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        inward = solver.pad_normals.copy()
        perpendicular = np.zeros_like(inward)
        perpendicular[:, :, 1] = 1.0
        solver.pad_normals = (
            np.cos(np.deg2rad(45.0)) * inward
            + np.sin(np.deg2rad(45.0)) * perpendicular
        )
        names = tuple(f"joint_{index}" for index in range(23))
        diagnostic = AUDIT.audit_kinematics(
            plan,
            grasp,
            mode="Diagnostic",
            solver=solver,
            joint_names=names,
            projection_fn=_capsule_projection,
        )
        acceptance = AUDIT.audit_kinematics(
            plan,
            grasp,
            mode="Acceptance",
            solver=solver,
            joint_names=names,
            projection_fn=_capsule_projection,
        )
        self.assertTrue(diagnostic["passed"])
        self.assertFalse(acceptance["passed"])
        self.assertAlmostEqual(
            diagnostic["pad_angle"]["maximum_deg"], 45.0, places=6
        )
        self.assertEqual(diagnostic["pad_angle"]["mode_limit_deg"], 50.0)
        self.assertEqual(acceptance["pad_angle"]["mode_limit_deg"], 40.0)

    def test_joint_step_uses_calibrated_coarse_seed_not_grasp_q(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        grasp["joint_position_rad"] = grasp["joint_position_rad"].copy()
        grasp["joint_position_rad"][0] -= 0.005
        result = AUDIT.audit_kinematics(
            plan,
            grasp,
            mode="Diagnostic",
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertTrue(result["checks"]["saved_joint_step_consistent"])
        self.assertEqual(
            result["joint_step"]["seed_source"],
            "coarse_joint_positions_rad[0]",
        )

    def test_nonfinite_pad_normals_and_surface_mismatch_cannot_pass(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        solver.pad_normals[:] = np.nan
        report = AUDIT.audit_loaded_plan(
            plan,
            grasp,
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertEqual(report["conclusions"]["DIAGNOSTIC"], "ERROR")
        self.assertNotIn("Infinity", json.dumps(report, default=AUDIT._json_default))

        plan, grasp, solver = _make_valid_inputs()
        plan["surface_points_m"] = plan["surface_points_m"].copy()
        plan["surface_points_m"][:, :, 1] += 0.001
        plan["start_surface_local_m"] = plan["surface_points_m"][0].copy()
        plan["end_surface_local_m"] = plan["surface_points_m"][-1].copy()
        result = AUDIT.audit_kinematics(
            plan,
            grasp,
            mode="Diagnostic",
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertFalse(result["checks"]["projected_surface_consistent"])

    def test_coarse_keyframes_participate_in_pad_joint_and_progress_audit(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        inward = solver.pad_normals[0].copy()
        perpendicular = np.zeros_like(inward)
        perpendicular[:, 1] = 1.0
        solver.pad_normals[0] = (
            np.cos(np.deg2rad(45.0)) * inward
            + np.sin(np.deg2rad(45.0)) * perpendicular
        )
        seed_q = plan["coarse_joint_positions_rad"][0]
        solver.lower[0] = seed_q[0] - 1.0e-5
        plan["coarse_progress_m"][-1, 1:] -= 0.002
        result = AUDIT.audit_kinematics(
            plan,
            grasp,
            mode="Diagnostic",
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertEqual(result["pad_angle"]["source"], "keyframe")
        self.assertAlmostEqual(result["pad_angle"]["maximum_deg"], 45.0, places=6)
        self.assertLess(
            result["pad_angle"]["maximum_interpolated_frame_deg"], 1.0e-5
        )
        self.assertAlmostEqual(
            result["pad_angle"]["maximum_keyframe_deg"], 45.0, places=6
        )
        self.assertEqual(result["joint_limit"]["source"], "keyframe")
        self.assertAlmostEqual(
            result["joint_limit"]["minimum_margin_rad"], 1.0e-5, places=9
        )
        self.assertFalse(result["checks"]["coarse_progress_geometry_consistent"])

    def test_keyframe_backtracking_uses_recomputed_coarse_geometry(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        q_plan = plan["joint_positions_rad"]
        plan["coarse_joint_positions_rad"] = np.vstack(
            (
                plan["coarse_joint_positions_rad"][0],
                q_plan[20],
                q_plan[19],
                q_plan[-1],
            )
        )
        plan["coarse_progress_m"] = np.repeat(
            np.asarray([[0.0], [0.15], [0.16], [0.48]]),
            5,
            axis=1,
        )
        plan["mpc_coarse_distance_m"] = np.asarray(
            [0.0, 0.15, 0.16, 0.48]
        )
        result = AUDIT.audit_kinematics(
            plan,
            grasp,
            mode="Diagnostic",
            solver=solver,
            joint_names=tuple(f"joint_{index}" for index in range(23)),
            projection_fn=_capsule_projection,
        )
        self.assertFalse(
            result["checks"][
                "recomputed_keyframe_backtracking_within_0_2mm"
            ]
        )
        self.assertGreater(
            result["coarse_progress_geometry"][
                "maximum_recomputed_backtracking_step_m"
            ],
            0.0002,
        )

    def test_cpu_loader_routes_import_noise_to_stderr(self) -> None:
        fake_backend = (object(), tuple(), _capsule_projection)

        def noisy_factory():
            print("Warp path: synthetic")
            return fake_backend

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                AUDIT, "_create_cpu_kinematics", side_effect=noisy_factory
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            loaded = AUDIT._load_cpu_kinematics()
        self.assertIs(loaded, fake_backend)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Warp path: synthetic", stderr.getvalue())

    def test_cli_help_does_not_require_loading_mjlab_environment(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(AUDIT_PATH), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--initial-grasp", completed.stdout)
        self.assertIn("no GPU environment", completed.stdout)

    def test_module_import_emits_no_stdout(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import runpy; "
                    f"runpy.run_path({str(AUDIT_PATH)!r}, "
                    "run_name='audit_import_only')"
                ),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_cli_emits_json_and_zero_only_when_all_conclusions_pass(self) -> None:
        plan, grasp, solver = _make_valid_inputs()
        output = io.StringIO()
        with (
            mock.patch.object(
                AUDIT,
                "load_npz_no_pickle",
                side_effect=(plan, grasp),
            ),
            mock.patch.object(
                AUDIT,
                "_load_cpu_kinematics",
                return_value=(
                    solver,
                    tuple(f"joint_{index}" for index in range(23)),
                    _capsule_projection,
                ),
            ),
            redirect_stdout(output),
        ):
            exit_code = AUDIT.main(
                (
                    "--plan",
                    "synthetic_plan.npz",
                    "--initial-grasp",
                    "synthetic_grasp.npz",
                    "--json-indent",
                    "0",
                )
            )
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(
            all(status == "PASS" for status in report["conclusions"].values())
        )

    def test_cli_returns_json_exit_two_for_malformed_numeric_dtype(self) -> None:
        plan, grasp, _ = _make_valid_inputs()
        plan["surface_points_m"] = np.full(
            plan["surface_points_m"].shape, "bad", dtype=np.str_
        )
        output = io.StringIO()
        with (
            mock.patch.object(
                AUDIT,
                "load_npz_no_pickle",
                side_effect=(plan, grasp),
            ),
            redirect_stdout(output),
        ):
            exit_code = AUDIT.main(
                (
                    "--plan",
                    "malformed_plan.npz",
                    "--initial-grasp",
                    "synthetic_grasp.npz",
                    "--json-indent",
                    "0",
                )
            )
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["conclusions"]["SCHEMA"], "FAIL")

    def test_cli_replaces_nonfinite_internal_report_with_valid_json_error(self) -> None:
        malformed_report = {
            "conclusions": {
                "SCHEMA": "PASS",
                "DIAGNOSTIC": "PASS",
                "LEVEL2_GEOMETRY": "PASS",
            },
            "bad": np.inf,
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                AUDIT,
                "load_npz_no_pickle",
                side_effect=({}, {}),
            ),
            mock.patch.object(
                AUDIT, "audit_loaded_plan", return_value=malformed_report
            ),
            redirect_stdout(output),
        ):
            exit_code = AUDIT.main(
                (
                    "--plan",
                    "synthetic_plan.npz",
                    "--initial-grasp",
                    "synthetic_grasp.npz",
                )
            )
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["conclusions"]["DIAGNOSTIC"], "ERROR")


if __name__ == "__main__":
    unittest.main()
