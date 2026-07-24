"""Compare fixed-palm MCC Jacobians with the full robot fingertip Jacobians."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mjlab.tasks.leaphand.full_hand_mcc_geometry import capsule_project
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    ARM_DOF,
    FIXED_TO_ATTACHED_PALM_ROTATION,
    TOTAL_DOF,
    FivePointReachabilitySolver,
    MotorForceFingerMCCController,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grasp", type=Path)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--half-height", type=float, default=0.26)
    args = parser.parse_args()

    grasp = np.load(args.grasp)
    q = np.asarray(grasp["joint_position_rad"], dtype=np.float64)
    center = np.asarray(grasp["object_center_m"], dtype=np.float64)
    rotation = np.eye(3)

    full = FivePointReachabilitySolver()
    points = full.forward_points(q)
    _, surface_normals = capsule_project(
        points,
        center,
        rotation,
        args.radius,
        args.half_height,
    )
    full_jacobian = full._stacked_jacobian()

    fixed = MotorForceFingerMCCController(
        device="cpu",
        num_envs=1,
        variant="hybrid_force_position",
    )
    fixed._set_hand_q(q[ARM_DOF:TOTAL_DOF])
    fixed_positions, fixed_jacobians = (
        fixed._tip_positions_and_jacobians()
    )
    palm_origin, palm_rotation = fixed._world_palm_pose(q)
    full_positions_local = (
        FIXED_TO_ATTACHED_PALM_ROTATION.T
        @ palm_rotation.T
        @ (points[1:] - palm_origin).T
    ).T
    print(
        f"fixed_positions_m={fixed_positions.round(6).tolist()}\n"
        f"full_positions_local_m={full_positions_local.round(6).tolist()}\n"
        "fixed_minus_full_local_m="
        f"{(fixed_positions - full_positions_local).round(6).tolist()}"
    )

    for finger in range(4):
        outward_world = surface_normals[finger + 1]
        outward_local = (
            FIXED_TO_ATTACHED_PALM_ROTATION.T
            @ palm_rotation.T
            @ outward_world
        )
        base = 4 * finger
        fixed_j = fixed_jacobians[finger][:, base : base + 4]
        correction = fixed_j.T @ np.linalg.solve(
            fixed_j @ fixed_j.T + 1.0e-3 * np.eye(3),
            -outward_local,
        )
        full_j = full_jacobian[
            3 * (finger + 1) : 3 * (finger + 2),
            ARM_DOF + base : ARM_DOF + base + 4,
        ]
        predicted_world = full_j @ correction
        predicted_local = fixed_j @ correction
        fixed_predicted_world = (
            palm_rotation
            @ FIXED_TO_ATTACHED_PALM_ROTATION
            @ predicted_local
        )
        print(
            f"finger={finger} "
            f"fixed_inward_dot={float(predicted_local @ outward_local):+.6f} "
            f"full_inward_dot={float(predicted_world @ outward_world):+.6f} "
            f"fixed_vs_full_cos="
            f"{float(fixed_predicted_world @ predicted_world / max(np.linalg.norm(fixed_predicted_world) * np.linalg.norm(predicted_world), 1.0e-12)):+.6f} "
            f"full_displacement={predicted_world.round(6).tolist()} "
            f"joint_correction={correction.round(6).tolist()}"
        )


if __name__ == "__main__":
    main()
