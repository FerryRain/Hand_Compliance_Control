"""Search a collision-free five-point pre-contact pose for the thick capsule."""

from __future__ import annotations

import argparse

import numpy as np

from mjlab.tasks.leaphand.full_hand_mcc_geometry import capsule_project
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    FR3_HOME_Q,
    FivePointReachabilitySolver,
)
from mjlab.tasks.leaphand.leaphand_direct_force_env import (
    DEFAULT_PREGRASP_Q,
)
import mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg as env_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--half-height", type=float, default=0.26)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--center-x-min", type=float, default=0.55)
    parser.add_argument("--center-x-max", type=float, default=1.02)
    parser.add_argument("--max-tip-standoff", type=float, default=0.09)
    args = parser.parse_args()

    env_module.FULL_HAND_CAPSULE_RADIUS = args.radius
    env_module.FULL_HAND_CAPSULE_HALF_HEIGHT = args.half_height
    solver = FivePointReachabilitySolver(
        tolerance=0.005,
        max_iterations=120,
        palm_weight=5.0,
    )
    rotation = np.eye(3)
    candidates: list[tuple[float, ...]] = []
    evaluated = 0
    ik_evaluated = 0
    grasp_side_count = 0
    clearance_count = 0
    best_prefilter: list[tuple[float, float, float, float, float, float, str]] = []

    for roll in np.linspace(2.25, 4.05, 13):
        q = np.concatenate(
            (
                FR3_HOME_Q.astype(np.float64),
                np.asarray(DEFAULT_PREGRASP_Q, dtype=np.float64),
            )
        )
        q[5] += roll
        points = solver.forward_points(q)
        palm_position, palm_rotation = solver.forward_palm_pose(q)
        for x in np.linspace(
            args.center_x_min, args.center_x_max, 20
        ):
            for y in np.linspace(-0.30, 0.20, 21):
                for z in np.linspace(0.60, 0.82, 12):
                    evaluated += 1
                    center = np.asarray((x, y, z))
                    center_in_palm = palm_rotation.T @ (
                        center - palm_position
                    )
                    if center_in_palm[0] >= -0.02:
                        continue
                    grasp_side_count += 1
                    clearance, geom_name = solver.minimum_non_tip_clearance(
                        q, center, rotation
                    )
                    if clearance < 0.012:
                        continue
                    clearance_count += 1
                    surface, _ = capsule_project(
                        points,
                        center,
                        rotation,
                        args.radius,
                        args.half_height,
                    )
                    tip_standoff = np.linalg.norm(
                        points[1:] - surface[1:], axis=1
                    )
                    max_standoff = float(tip_standoff.max())
                    best_prefilter.append(
                        (
                            max_standoff,
                            clearance,
                            roll,
                            x,
                            y,
                            z,
                            geom_name,
                        )
                    )
                    if max_standoff > args.max_tip_standoff:
                        continue
                    targets = surface.copy()
                    targets[0] = points[0]
                    ik_evaluated += 1
                    result = solver.solve(targets, q)
                    if not result.accepted:
                        continue
                    solved_clearance, solved_geom = (
                        solver.minimum_non_tip_clearance(
                            result.joint_position,
                            center,
                            rotation,
                        )
                    )
                    if solved_clearance < 0.008:
                        continue
                    interpolation_clearance = min(
                        solver.minimum_non_tip_clearance(
                            (1.0 - phase) * q
                            + phase * result.joint_position,
                            center,
                            rotation,
                        )[0]
                        for phase in np.linspace(0.0, 1.0, 9)
                    )
                    if interpolation_clearance < 0.008:
                        continue
                    score = (
                        2.0 * interpolation_clearance
                        + solved_clearance
                        + clearance
                        - 0.2 * float(tip_standoff.max())
                        - 0.1 * float(result.residual_m.max())
                    )
                    candidates.append(
                        (
                            score,
                            roll,
                            x,
                            y,
                            z,
                            clearance,
                            solved_clearance,
                            interpolation_clearance,
                            float(tip_standoff.max()),
                            float(result.residual_m.max()),
                            geom_name,
                            solved_geom,
                        )
                    )

    candidates.sort(key=lambda item: item[0], reverse=True)
    print(
        f"evaluated={evaluated} ik_evaluated={ik_evaluated} "
        f"accepted={len(candidates)} grasp_side={grasp_side_count} "
        f"clearance_pass={clearance_count}"
    )
    best_prefilter.sort(key=lambda item: item[0])
    for item in best_prefilter[:5]:
        print(
            "prefilter max_tip_standoff_mm={:.2f} clearance_mm={:.2f} "
            "roll={:.5f} center=({:.4f},{:.4f},{:.4f}) nearest={}".format(
                item[0] * 1000.0,
                item[1] * 1000.0,
                item[2],
                item[3],
                item[4],
                item[5],
                item[6],
            )
        )
    for candidate in candidates[: args.top_k]:
        print(
            "score={:.5f} roll={:.5f} center=({:.4f},{:.4f},{:.4f}) "
            "clearance_mm=({:.2f},{:.2f},{:.2f}) "
            "max_tip_standoff_mm={:.2f} max_ik_residual_mm={:.2f} "
            "nearest=({},{})".format(
                candidate[0],
                candidate[1],
                candidate[2],
                candidate[3],
                candidate[4],
                candidate[5] * 1000.0,
                candidate[6] * 1000.0,
                candidate[7] * 1000.0,
                candidate[8] * 1000.0,
                candidate[9] * 1000.0,
                candidate[10],
                candidate[11],
            )
        )


if __name__ == "__main__":
    main()
