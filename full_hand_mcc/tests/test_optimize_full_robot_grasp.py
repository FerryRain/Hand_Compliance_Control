"""Pure-CPU regression tests for the full-robot grasp optimizer guards."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "full_hand_mcc" / "scripts" / "optimize_full_robot_grasp.py"
)
SPEC = importlib.util.spec_from_file_location(
    "optimize_full_robot_grasp_under_test",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
OPTIMIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTIMIZER)


class GraspOptimizerGuardTest(unittest.TestCase):
    def test_seed_only_uses_an_independent_copy_of_seed_as_reference(self) -> None:
        nominal = np.asarray((1.0, 2.0, 3.0))
        seed = np.asarray((4.0, 5.0, 6.0))

        selected = OPTIMIZER._select_physical_reference_q(
            nominal,
            seed_only=True,
            seed_q=seed,
        )
        np.testing.assert_array_equal(selected, seed)
        selected[0] = -1.0
        self.assertEqual(seed[0], 4.0)

        ordinary = OPTIMIZER._select_physical_reference_q(
            nominal,
            seed_only=False,
            seed_q=None,
        )
        np.testing.assert_array_equal(ordinary, nominal)
        with self.assertRaisesRegex(ValueError, "requires a seed q"):
            OPTIMIZER._select_physical_reference_q(
                nominal,
                seed_only=True,
                seed_q=None,
            )

    def test_joint_margin_soft_hinge_and_hard_gate(self) -> None:
        lower = np.asarray((0.0, -1.0, 0.0))
        upper = np.asarray((1.0, 1.0, 2.0))
        q = np.asarray((0.1, 0.0, 1.9))

        margin = OPTIMIZER._joint_margin_rad(q, lower, upper)
        np.testing.assert_allclose(margin, (0.1, 1.0, 0.1))
        np.testing.assert_allclose(
            OPTIMIZER._joint_margin_soft_residual(
                q,
                lower,
                upper,
                0.2,
            ),
            (2.0, 0.0, 2.0),
        )
        self.assertTrue(OPTIMIZER._joint_margin_hard_ok(margin, 0.1))
        self.assertFalse(OPTIMIZER._joint_margin_hard_ok(margin, 0.1001))
        self.assertFalse(
            OPTIMIZER._joint_margin_hard_ok(
                np.asarray((0.2, np.nan)),
                0.0,
            )
        )

    def test_zero_joint_margin_targets_preserve_legacy_boundary(self) -> None:
        lower = np.asarray((0.0, -1.0))
        upper = np.asarray((1.0, 1.0))
        q_on_limit = np.asarray((0.0, 1.0))

        np.testing.assert_array_equal(
            OPTIMIZER._joint_margin_soft_residual(
                q_on_limit,
                lower,
                upper,
                0.0,
            ),
            np.zeros(0),
        )
        self.assertTrue(
            OPTIMIZER._joint_margin_hard_ok(
                OPTIMIZER._joint_margin_rad(
                    q_on_limit,
                    lower,
                    upper,
                ),
                0.0,
            )
        )

    def test_optimizer_result_accepts_success_or_finite_max_nfev(self) -> None:
        valid = SimpleNamespace(
            success=True,
            status=1,
            x=np.asarray((0.1, 0.2)),
            fun=np.asarray((0.0, 0.1)),
            cost=0.01,
            optimality=1.0e-6,
        )
        self.assertTrue(
            OPTIMIZER._least_squares_result_is_acceptable(valid)
        )
        max_nfev = SimpleNamespace(**vars(valid))
        max_nfev.success = False
        max_nfev.status = 0
        self.assertTrue(
            OPTIMIZER._least_squares_result_is_acceptable(max_nfev)
        )

        failed = SimpleNamespace(**vars(valid))
        failed.success = False
        failed.status = -1
        self.assertFalse(
            OPTIMIZER._least_squares_result_is_acceptable(failed)
        )
        for field, value in (
            ("x", np.asarray((np.nan, 0.2))),
            ("fun", np.asarray((0.0, np.inf))),
            ("cost", np.nan),
            ("optimality", np.inf),
        ):
            invalid = SimpleNamespace(**vars(valid))
            setattr(invalid, field, value)
            self.assertFalse(
                OPTIMIZER._least_squares_result_is_acceptable(invalid),
                msg=f"field={field}",
            )

    def test_final_optimizer_joint_gate_rejects_insufficient_margin(
        self,
    ) -> None:
        converged = SimpleNamespace(
            success=True,
            status=1,
            x=np.asarray((0.1, 0.2)),
            fun=np.asarray((0.0, 0.1)),
            cost=0.01,
            optimality=1.0e-6,
        )
        self.assertFalse(
            OPTIMIZER._optimizer_and_joint_margin_hard_ok(
                converged,
                np.asarray((0.08, 0.049)),
                0.05,
            )
        )
        self.assertTrue(
            OPTIMIZER._optimizer_and_joint_margin_hard_ok(
                converged,
                np.asarray((0.08, 0.05)),
                0.05,
            )
        )

        max_nfev = SimpleNamespace(**vars(converged))
        max_nfev.success = False
        max_nfev.status = 0
        self.assertTrue(
            OPTIMIZER._optimizer_and_joint_margin_hard_ok(
                max_nfev,
                np.asarray((0.08, 0.05)),
                0.05,
            )
        )


class GraspOptimizerSourceStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_joint_margin_cli_defaults_remain_zero(self) -> None:
        defaults: dict[str, object] = {}
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                continue
            argument = node.args[0].value
            if argument not in {
                "--optimization-joint-margin-rad",
                "--minimum-accepted-joint-margin-rad",
            }:
                continue
            default_keyword = next(
                keyword
                for keyword in node.keywords
                if keyword.arg == "default"
            )
            defaults[str(argument)] = ast.literal_eval(default_keyword.value)

        self.assertEqual(
            defaults,
            {
                "--optimization-joint-margin-rad": 0.0,
                "--minimum-accepted-joint-margin-rad": 0.0,
            },
        )

    def test_saved_npz_contains_actual_and_target_joint_margins(self) -> None:
        saved_fields: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "savez_compressed"
            ):
                continue
            saved_fields.update(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None
            )

        self.assertTrue(
            {
                "joint_margin_rad",
                "minimum_joint_margin_rad",
                "optimization_joint_margin_rad",
                "minimum_accepted_joint_margin_rad",
                "optimizer_success",
                "optimizer_status",
                "optimizer_hard_feasible_max_nfev_override",
            }.issubset(saved_fields)
        )


if __name__ == "__main__":
    unittest.main()
