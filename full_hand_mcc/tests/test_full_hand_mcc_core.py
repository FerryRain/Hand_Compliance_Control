from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

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

DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/demo_surface_slide.py"
)


class FullHandMCCCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.actual = np.zeros((2, 5, 3), dtype=np.float64)
        self.desired = self.actual.copy()
        self.normals = np.zeros_like(self.actual)
        self.normals[..., 2] = 1.0
        self.forces = np.zeros_like(self.actual)

    def test_all_variants_are_finite_and_speed_limited(self) -> None:
        self.desired[..., 0] = 0.1
        for variant in CORE.MCC_VARIANTS:
            controller = CORE.FullHandMCCCore(variant=variant)
            result = controller.step(
                self.actual, self.desired, self.normals, self.forces
            )
            self.assertTrue(np.all(np.isfinite(result.reference_points)))
            speed = np.linalg.norm(result.reference_velocity, axis=-1)
            self.assertLessEqual(
                float(speed.max()),
                controller.gains.max_reference_speed + 1.0e-12,
            )

    def test_low_force_moves_inward(self) -> None:
        controller = CORE.FullHandMCCCore(variant="hybrid_force_position")
        # Activate contact first, then drop below the desired force while
        # remaining above the hysteresis off threshold.
        contact_force = self.forces.copy()
        contact_force[..., 2] = 0.2
        controller.step(
            self.actual, self.desired, self.normals, contact_force
        )
        result = controller.step(
            self.actual, self.desired, self.normals, contact_force
        )
        self.assertTrue(np.all(result.reference_velocity[..., 2] < 0.0))

    def test_contact_hysteresis(self) -> None:
        controller = CORE.FullHandMCCCore()
        force = self.forces.copy()
        force[..., 2] = 0.16
        first = controller.step(
            self.actual, self.desired, self.normals, force
        )
        self.assertTrue(np.all(first.contact_active))
        force[..., 2] = 0.10
        second = controller.step(
            self.actual, self.desired, self.normals, force
        )
        self.assertTrue(np.all(second.contact_active))
        force[..., 2] = 0.01
        third = controller.step(
            self.actual, self.desired, self.normals, force
        )
        self.assertFalse(np.any(third.contact_active))

    def test_hybrid_load_balance_pushes_low_force_finger_inward(self) -> None:
        controller = CORE.FullHandMCCCore(variant="hybrid_force_position")
        force = self.forces[:1].copy()
        force[:, 0, 2] = 3.0
        force[:, 1:, 2] = np.asarray([0.2, 1.2, 1.2, 1.2])
        controller.step(
            self.actual[:1], self.desired[:1], self.normals[:1], force
        )
        result = controller.step(
            self.actual[:1], self.desired[:1], self.normals[:1], force
        )
        low_force_velocity = result.reference_velocity[0, 1, 2]
        loaded_velocity = result.reference_velocity[0, 2:, 2]
        self.assertLess(low_force_velocity, float(loaded_velocity.min()))

    def test_passivity_tank_never_drops_below_floor(self) -> None:
        controller = CORE.FullHandMCCCore(variant="passivity_tank")
        self.desired[..., 2] = -0.5
        self.forces[..., 2] = 100.0
        result = None
        for _ in range(300):
            result = controller.step(
                self.actual, self.desired, self.normals, self.forces
            )
        assert result is not None
        self.assertGreaterEqual(
            float(result.energy_tank.min()),
            controller.gains.energy_tank_floor - 1.0e-12,
        )
        self.assertLessEqual(float(result.passivity_scale.max()), 1.0)

    def test_bad_shape_is_rejected(self) -> None:
        controller = CORE.FullHandMCCCore()
        with self.assertRaises(ValueError):
            controller.step(
                np.zeros((4, 3)),
                np.zeros((4, 3)),
                np.zeros((4, 3)),
                np.zeros((4, 3)),
            )


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


if __name__ == "__main__":
    unittest.main()
