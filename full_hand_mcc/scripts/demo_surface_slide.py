"""FR3 + LEAP Baseline-2 direct-force surface-sliding demo.

Run from the repository root:

    python full_hand_mcc/scripts/demo_surface_slide.py \
        --viewer native

The five target points are kept on the capsule analytically.  Every proposed
slide increment is then accepted only after the real 23-DoF model reaches all
five points within the configured tolerance.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

# The 23-DoF least-squares planner is highly non-convex near the lower
# capsule end cap.  Multi-threaded BLAS reductions may perturb a candidate
# enough to select a different local branch, so keep the small dense planner
# deterministic.  GPU physics remains on the requested CUDA device.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.leaphand.full_hand_mcc_geometry import (
    capsule_meridian_curvature,
    capsule_meridian_coordinates,
    capsule_meridian_targets,
    capsule_project,
    ellipsoid_meridian_curvature,
    ellipsoid_meridian_coordinates,
    ellipsoid_meridian_targets,
    ellipsoid_meridian_total_length,
    ellipsoid_project,
)
from mjlab.tasks.leaphand.full_hand_mcc_planner_diagnostics import (
    LOW_MOTION_DEFAULT_WINDOW_FRAMES,
    LOW_MOTION_FORWARD_PROGRESS_RATIO,
    MOVING_BRIDGE_FORWARD_FINGER_COUNT,
    RejectedMovingBridgeCandidate,
    bounded_moving_bridge_trust_radius,
    build_receding_horizon_distances,
    build_bridge_rejection_metrics,
    build_candidate_failure_metrics,
    build_palm_guide_multistart_specs,
    central_difference_clearance_gradient,
    damped_task_nullspace_directions,
    deduplicated_bridge_multistart_seeds,
    evaluate_bridge_conditions,
    evaluate_moving_bridge_motion,
    find_unmarked_low_motion_windows,
    format_bridge_rejection_record,
    make_bridge_rejection_record,
    moving_bridge_candidate_rank,
    moving_bridge_local_residual,
    moving_bridge_tip_geometry_residual,
    orientation_aware_candidate_rank,
    positive_self_clearance_residual,
    prioritized_suffix_rollout_indices,
    prioritized_suffix_seed_indices,
    progress_aware_arc_targets,
    save_mpc_failure_prefix,
    save_npz_no_overwrite,
    segment_tip_clearance_status,
    self_separation_ascent_seeds,
    smooth_pad_alignment_residual,
    smoothstep_joint_interpolation,
    suffix_rollout_prefix_rank,
    strict_suffix_task_hinge_residual,
    terminal_contact_sample_mask,
    terminal_contact_start_distance,
    transported_suffix_seed_rows,
)
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    ARM_DOF,
    ARM_JOINT_NAMES,
    FULL_HAND_CAPSULE_HALF_HEIGHT,
    FULL_HAND_CAPSULE_RADIUS,
    HAND_DOF,
    TOTAL_DOF,
    FivePointReachabilitySolver,
    FullHandMCCControlCfg,
    full_hand_mcc_env_cfg,
)
from mjlab.tasks.leaphand.leaphand_direct_force_env import MCC_TIP_NAMES
import mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg as full_hand_env_module
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


CAPSULE_RADIUS = FULL_HAND_CAPSULE_RADIUS
CAPSULE_HALF_HEIGHT = FULL_HAND_CAPSULE_HALF_HEIGHT
SURFACE_TOTAL_LENGTH = np.pi * CAPSULE_RADIUS + 2.0 * CAPSULE_HALF_HEIGHT
surface_meridian_curvature = capsule_meridian_curvature

PROTECTED_SELF_PAIR_NAMES = (
    ("mcp_joint_geom", "dip_geom"),
    ("mcp_joint_2_geom", "dip_2_geom"),
    ("mcp_joint_3_geom", "dip_3_geom"),
)


def build_mpc_distance_grid(
    total_distance_m: float,
    base_keyframes: int,
    local_refine_start_m: float,
    local_refine_end_m: float,
    local_refine_factor: int,
    local_refine_windows: tuple[tuple[float, float, int], ...] = (),
) -> np.ndarray:
    """Build a deterministic distance grid with repeatable local refinement."""

    base_grid = np.linspace(0.0, total_distance_m, base_keyframes + 1)
    refine_windows = list(local_refine_windows)
    if local_refine_factor > 1:
        refine_windows.insert(
            0,
            (
                local_refine_start_m,
                local_refine_end_m,
                local_refine_factor,
            ),
        )
    if not refine_windows:
        return base_grid

    refined_distance = [0.0]
    for left, right in zip(base_grid[:-1], base_grid[1:]):
        factor = max(
            (
                refine_factor
                for refine_start, refine_end, refine_factor in refine_windows
                if right > refine_start and left < refine_end
            ),
            default=1,
        )
        refined_distance.extend(
            np.linspace(left, right, factor + 1)[1:].tolist()
        )
    return np.asarray(refined_distance, dtype=np.float64)


def main() -> None:
    global CAPSULE_RADIUS, CAPSULE_HALF_HEIGHT, SURFACE_TOTAL_LENGTH
    global capsule_project, capsule_meridian_coordinates
    global capsule_meridian_targets, surface_meridian_curvature
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        choices=("native", "viser", "headless", "video"),
        default="native",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic environment/reset seed for reproducible planning.",
    )
    parser.add_argument(
        "--planner-state-quantization-rad",
        type=float,
        default=0.0,
        help=(
            "Quantize the GPU-loaded contact state relative to its servo "
            "command before using it as the non-convex MPC seed. This does "
            "not alter the measured MCC/contact state; a small value such "
            "as 0.0005 rad makes repeated CUDA runs select the same planning "
            "branch."
        ),
    )
    parser.add_argument("--axial-travel-m", type=float, default=0.20)
    parser.add_argument(
        "--palm-travel-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of fingertip surface travel assigned to the non-contact "
            "palm point. Values below one force active finger articulation "
            "instead of accepting rigid whole-hand transport."
        ),
    )
    parser.add_argument(
        "--palm-late-follow-start-m",
        type=float,
        default=1.0,
        help=(
            "Fingertip surface progress at which the non-contact palm starts "
            "a smooth late-route slowdown. The default is beyond ordinary "
            "routes and therefore disables the slowdown."
        ),
    )
    parser.add_argument(
        "--palm-late-follow-ratio",
        type=float,
        default=1.0,
        help=(
            "Palm-to-fingertip velocity ratio after the late-follow ramp. "
            "This preserves a rigid, collision-safe grasp early in the "
            "route, then assigns the remaining travel to finger articulation."
        ),
    )
    parser.add_argument(
        "--palm-late-follow-ramp-m",
        type=float,
        default=0.04,
        help=(
            "Fingertip surface distance over which palm velocity transitions "
            "smoothly to --palm-late-follow-ratio."
        ),
    )
    parser.add_argument(
        "--palm-follow-surface-frame",
        action="store_true",
        help=(
            "Rotate the non-contact palm target with the mean physical "
            "fingertip contact frame. This is intended for long routes that "
            "cross a curved end cap."
        ),
    )
    parser.add_argument(
        "--palm-surface-frame-gain",
        type=float,
        default=1.0,
        help=(
            "Fraction of the local contact-frame rotation transported to "
            "the palm when --palm-follow-surface-frame is enabled."
        ),
    )
    parser.add_argument(
        "--palm-surface-frame-late-gain",
        type=float,
        default=None,
        help=(
            "Optional late-route surface-frame gain. The planner smoothly "
            "transitions from --palm-surface-frame-gain to this value, "
            "allowing the lower-cap branch to be preserved before reducing "
            "palm rotation near the cylinder."
        ),
    )
    parser.add_argument(
        "--palm-surface-frame-late-start-m",
        type=float,
        default=1.0,
        help="Surface progress where the late frame-gain transition starts.",
    )
    parser.add_argument(
        "--palm-surface-frame-late-ramp-m",
        type=float,
        default=0.04,
        help="Surface distance used for the smooth late frame-gain transition.",
    )
    parser.add_argument(
        "--palm-surface-frame-terminal-gain",
        type=float,
        default=None,
        help=(
            "Optional third-stage surface-frame gain for the terminal "
            "recovery region. It preserves the validated early/late branch "
            "and changes palm rotation only near the configured terminal "
            "transition."
        ),
    )
    parser.add_argument(
        "--palm-surface-frame-terminal-start-m",
        type=float,
        default=1.0,
        help=(
            "Surface progress where the terminal frame-gain transition "
            "starts."
        ),
    )
    parser.add_argument(
        "--palm-surface-frame-terminal-ramp-m",
        type=float,
        default=0.04,
        help=(
            "Surface distance used for the terminal frame-gain transition."
        ),
    )
    parser.add_argument(
        "--palm-clearance-lift-m",
        type=float,
        default=0.0,
        help=(
            "Maximum collision-avoidance displacement of the palm away from "
            "the initial fingertip contact patch while the fingertips stay "
            "on their planned surface route."
        ),
    )
    parser.add_argument(
        "--palm-clearance-ramp-m",
        type=float,
        default=0.04,
        help=(
            "Surface distance over which --palm-clearance-lift-m is smoothly "
            "introduced; the lift remains active for the rest of the route."
        ),
    )
    parser.add_argument(
        "--palm-clearance-use-local-normal",
        action="store_true",
        help=(
            "Move the non-contact palm away from the object's surface along "
            "the palm target's own projected surface normal. This is more "
            "direct than the mean fingertip-patch normal on curved end caps."
        ),
    )
    parser.add_argument(
        "--palm-clearance-tilt-deg",
        type=float,
        default=0.0,
        help=(
            "Maximum collision-avoidance palm tilt. Positive values rotate "
            "the lower palm edge away from the initial fingertip contact "
            "patch, reducing the radial lift demanded from the fingers."
        ),
    )
    parser.add_argument(
        "--palm-clearance-tilt-start-m",
        type=float,
        default=0.0,
        help=(
            "Surface progress before collision-avoidance palm tilt begins. "
            "Delaying tilt preserves the calibrated finger-pad pose while "
            "leaving a highly curved lower end cap."
        ),
    )
    parser.add_argument(
        "--palm-clearance-tilt-ramp-m",
        type=float,
        default=0.04,
        help="Surface distance over which delayed palm tilt ramps in.",
    )
    parser.add_argument(
        "--palm-clearance-tilt-release-start-m",
        type=float,
        default=1.0,
        help=(
            "Surface progress at which the collision-avoidance palm tilt "
            "starts returning smoothly to zero. The default disables release "
            "for ordinary routes."
        ),
    )
    parser.add_argument(
        "--palm-clearance-tilt-release-ramp-m",
        type=float,
        default=0.04,
        help="Surface distance over which the late palm tilt is released.",
    )
    parser.add_argument(
        "--palm-clearance-secondary-lift-m",
        type=float,
        default=0.0,
        help=(
            "Additional late-route outward palm displacement. This staged "
            "lift avoids over-extending the fingers during the initial cap "
            "transition while protecting the palm near the widest section."
        ),
    )
    parser.add_argument(
        "--palm-clearance-secondary-start-m",
        type=float,
        default=0.10,
        help="Surface progress at which the staged secondary lift starts.",
    )
    parser.add_argument(
        "--palm-clearance-secondary-ramp-m",
        type=float,
        default=0.04,
        help="Surface distance over which the staged secondary lift ramps in.",
    )
    parser.add_argument(
        "--palm-terminal-local-offset-mm",
        type=float,
        nargs=3,
        metavar=("NORMAL", "AZIMUTH", "MERIDIAN"),
        default=(0.0, 0.0, 0.0),
        help=(
            "Bounded terminal correction of the non-contact palm input "
            "point in the local surface basis. This moves the five-point "
            "target itself. In ordinary mode the real palm must still satisfy "
            "--mpc-palm-position-tolerance-mm; in --palm-guide-only mode the "
            "target is only a weak directional reference."
        ),
    )
    parser.add_argument(
        "--palm-terminal-local-offset-start-m",
        type=float,
        default=1.0,
        help="Surface progress where the terminal palm-point correction starts.",
    )
    parser.add_argument(
        "--palm-terminal-local-offset-ramp-m",
        type=float,
        default=0.02,
        help="Surface distance used to establish the terminal palm correction.",
    )
    parser.add_argument(
        "--palm-terminal-second-local-offset-mm",
        type=float,
        nargs=3,
        metavar=("NORMAL", "AZIMUTH", "MERIDIAN"),
        default=(0.0, 0.0, 0.0),
        help=(
            "Additive second-stage correction of the non-contact palm input "
            "point in the same local surface basis. The first stage must be "
            "fully established before this stage starts, and their combined "
            "offset must remain inside the same 3 mm input ball."
        ),
    )
    parser.add_argument(
        "--palm-terminal-second-local-offset-start-m",
        type=float,
        default=1.0,
        help="Surface progress where the second palm correction starts.",
    )
    parser.add_argument(
        "--palm-terminal-second-local-offset-ramp-m",
        type=float,
        default=0.02,
        help="Surface distance used to establish the second palm correction.",
    )
    parser.add_argument(
        "--finger-gait-amplitude-m",
        type=float,
        default=0.0,
        help=(
            "Peak monotonic lead/lag of each fingertip around the palm's main "
            "surface trajectory. Alternating signs force active per-finger "
            "surface sliding while all tips still travel end to end."
        ),
    )
    parser.add_argument(
        "--finger-meridian-gait-mm",
        type=float,
        default=0.0,
        help=(
            "Peak asynchronous fingertip lead along the meridian during a "
            "bounded curvature-transition window. The offset returns to zero "
            "after the window, so all fingertips retain the same endpoint."
        ),
    )
    parser.add_argument(
        "--finger-meridian-gait-start-m",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--finger-meridian-gait-end-m",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--finger-meridian-gait-scales",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help="Per-fingertip scale for the bounded meridian lead profile.",
    )
    parser.add_argument(
        "--finger-meridian-correction-mm",
        type=float,
        default=0.0,
        help=(
            "Peak amplitude of a second, later bounded meridian correction. "
            "It preserves an already feasible early branch while correcting "
            "individual fingertips near a later curvature transition."
        ),
    )
    parser.add_argument(
        "--finger-meridian-correction-start-m",
        type=float,
        default=0.034,
    )
    parser.add_argument(
        "--finger-meridian-correction-end-m",
        type=float,
        default=0.060,
    )
    parser.add_argument(
        "--finger-meridian-correction-scales",
        type=float,
        nargs=4,
        default=(0.0, 1.0, 0.0, 0.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
    )
    parser.add_argument(
        "--finger-meridian-terminal-correction-mm",
        type=float,
        default=0.0,
        help=(
            "Peak amplitude of a terminal zero-endpoint meridian target "
            "phase. This rewrites explicit per-fingertip surface plan points "
            "inside a bounded late window without changing acceptance bands."
        ),
    )
    parser.add_argument(
        "--finger-meridian-terminal-correction-start-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--finger-meridian-terminal-correction-end-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--finger-meridian-terminal-correction-scales",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 0.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=(
            "Signed per-fingertip scales for the terminal meridian target "
            "phase."
        ),
    )
    parser.add_argument(
        "--finger-meridian-terminal-tail-correction-mm",
        type=float,
        default=0.0,
        help=(
            "Peak amplitude of a second zero-endpoint meridian target phase "
            "used to correct only the tail of terminal fingertip recovery."
        ),
    )
    parser.add_argument(
        "--finger-meridian-terminal-tail-correction-start-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--finger-meridian-terminal-tail-correction-end-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--finger-meridian-terminal-tail-correction-scales",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 0.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=(
            "Signed per-fingertip scales for the terminal tail meridian "
            "target phase."
        ),
    )
    parser.add_argument(
        "--finger-meridian-local-phase",
        type=float,
        nargs=7,
        action="append",
        default=[],
        metavar=(
            "AMP_MM",
            "START_M",
            "END_M",
            "INDEX",
            "MIDDLE",
            "RING",
            "THUMB",
        ),
        help=(
            "Repeatable zero-endpoint local meridian plan phase: peak "
            "amplitude, start/end surface distance, and four signed finger "
            "scales. This changes explicit plan points, not acceptance bands."
        ),
    )
    parser.add_argument(
        "--runtime-finger-gait-rad",
        type=float,
        default=0.12,
        help=(
            "Maximum joint correction used by the Jacobian-projected "
            "fingertip gait."
        ),
    )
    parser.add_argument(
        "--runtime-tip-gait-mm",
        type=float,
        default=3.0,
        help=(
            "Peak per-finger displacement along the capsule tangent. The "
            "MuJoCo site Jacobian converts this to joint motion while "
            "suppressing normal separation."
        ),
    )
    parser.add_argument(
        "--runtime-gait-cycles",
        type=float,
        default=2.0,
        help="Number of slow URDF-valid fingertip gait cycles over the slide.",
    )
    parser.add_argument(
        "--runtime-gait-finger-scales",
        type=float,
        nargs=4,
        default=(0.85, 1.0, 1.0, 1.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=(
            "Per-finger gait scaling. The index default is slightly lower "
            "because its measured contact margin is the smallest."
        ),
    )
    parser.add_argument(
        "--object-shape",
        choices=("capsule", "ellipsoid"),
        default="capsule",
        help=(
            "Axisymmetric contact surface. Ellipsoid gives continuously "
            "varying meridian curvature; capsule gives a zero-to-1/r "
            "curvature jump at each cylinder/end-cap transition."
        ),
    )
    parser.add_argument(
        "--object-radius-m",
        type=float,
        default=0.15,
        help="Capsule radius or ellipsoid radial semi-axis.",
    )
    parser.add_argument(
        "--object-half-height-m",
        type=float,
        default=0.26,
        help="Capsule half-height or ellipsoid axial semi-axis.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("tip_only", "full_robot"),
        default="full_robot",
        help=(
            "Use tip_only while developing fingertip surface sliding, then "
            "full_robot for final arm/palm collision-aware validation."
        ),
    )
    parser.add_argument(
        "--object-center-x-m",
        type=float,
        default=0.74,
    )
    parser.add_argument(
        "--object-center-y-m",
        type=float,
        default=-0.05,
    )
    parser.add_argument(
        "--object-center-z-m",
        type=float,
        default=0.7077,
    )
    parser.add_argument(
        "--tool-roll-rad",
        type=float,
        default=float(np.pi),
        help=(
            "Additional FR3 joint-7 tool roll. Keep it inside the FR3 limit; "
            "palmar fingertip side toward an object outside the arm workspace."
        ),
    )
    parser.add_argument(
        "--planner",
        choices=(
            "staged_inward",
            "continuous_inward",
            "meridian_inward",
            "adaptive_surface_mpc",
            "circumferential_surface_mpc",
        ),
        default="adaptive_surface_mpc",
    )
    parser.add_argument("--waypoint-count", type=int, default=5)
    parser.add_argument("--stage-move-fraction", type=float, default=0.78)
    parser.add_argument("--surface-preload-mm", type=float, default=2.0)
    parser.add_argument("--max-plan-joint-step-rad", type=float, default=0.03)
    parser.add_argument(
        "--min-arm-clearance-mm",
        "--min-non-tip-clearance-mm",
        dest="min_arm_clearance_mm",
        type=float,
        default=2.0,
        help=(
            "Minimum planned FR3/object distance in full_robot mode. "
            "--min-non-tip-clearance-mm remains as a deprecated alias."
        ),
    )
    parser.add_argument(
        "--max-incidental-hand-penetration-mm",
        type=float,
        default=1.0,
        help=(
            "Maximum planned/runtime penetration for allowed palm, finger-"
            "link, or finger-back object contact. These contacts never count "
            "as fingertip-pad support."
        ),
    )
    parser.add_argument(
        "--max-incidental-hand-contact-force-n",
        type=float,
        default=24.0,
        help=(
            "Maximum instantaneous force magnitude on any allowed non-tip "
            "LEAP Hand contact geometry."
        ),
    )
    parser.add_argument(
        "--max-incidental-hand-total-force-n",
        type=float,
        default=36.0,
        help=(
            "Maximum sum of strongest per-geometry forces across all allowed "
            "non-tip LEAP Hand contacts."
        ),
    )
    parser.add_argument(
        "--max-pad-angle-deg",
        type=float,
        default=45.0,
        help=(
            "Hard maximum angle between every physical finger-pad normal "
            "and its contact point's local inward object-surface normal."
        ),
    )
    parser.add_argument(
        "--planner-pad-angle-margin-deg",
        type=float,
        default=5.0,
        help=(
            "Extra fingertip-pad angle margin enforced at every MPC segment "
            "to leave room for dynamic force-control and gait corrections."
        ),
    )
    parser.add_argument(
        "--planner-soft-pad-angle-deg",
        type=float,
        default=35.0,
        help=(
            "Preferred interpolated finger-pad cone used by the smooth MPC "
            "posture objective. The hard planner cone is unchanged."
        ),
    )
    parser.add_argument(
        "--planner-soft-pad-weight",
        type=float,
        default=24.0,
        help="Default weight of the smooth preferred-pad-cone residual.",
    )
    parser.add_argument(
        "--planner-soft-pad-softplus-tau",
        type=float,
        default=0.02,
        help="Softplus temperature in dimensionless pad-alignment units.",
    )
    parser.add_argument(
        "--planner-tip-geom-target-mm",
        type=float,
        nargs=4,
        default=(-0.25, -0.25, -0.50, -0.25),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=(
            "Preferred signed distances from the four physical fingertip "
            "collision geoms to the object surface."
        ),
    )
    parser.add_argument(
        "--planner-tip-geom-weight",
        type=float,
        default=2200.0,
        help="Weight of the four physical fingertip distance residuals.",
    )
    parser.add_argument(
        "--planner-tip-geom-inner-cap-mm",
        type=float,
        default=-0.8,
        help=(
            "Soft inner signed-distance cap, kept inside the immutable hard "
            "tip penetration limit."
        ),
    )
    parser.add_argument(
        "--planner-tip-geom-inner-weight",
        type=float,
        default=18000.0,
        help="Weight of physical-tip penetration beyond the soft inner cap.",
    )
    parser.add_argument(
        "--planner-protected-self-clearance-mm",
        type=float,
        default=0.10,
        help=(
            "Preferred positive clearance for the three protected "
            "non-adjacent MCP-to-DIP geometry pairs."
        ),
    )
    parser.add_argument(
        "--planner-protected-self-clearance-weight",
        type=float,
        default=4000.0,
        help="Weight of the protected MCP-to-DIP clearance barrier.",
    )
    parser.add_argument(
        "--planner-self-separation-seed-step-rad",
        type=float,
        default=0.005,
        help=(
            "Maximum normalized central-FD ascent nudge used when a "
            "protected self pair starts below its preferred clearance."
        ),
    )
    parser.add_argument("--motion-start", type=int, default=1000)
    parser.add_argument("--ik-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--ik-max-iterations", type=int, default=160)
    parser.add_argument("--palm-ik-weight", type=float, default=5.0)
    parser.add_argument(
        "--palm-path-tolerance-mm",
        type=float,
        default=50.0,
        help=(
            "Maximum deviation from the nominal non-contact palm path. "
            "The accepted palm plan point is always replaced by the actual "
            "URDF-reachable point."
        ),
    )
    parser.add_argument("--mpc-keyframes", type=int, default=40)
    parser.add_argument(
        "--mpc-local-refine-start-m",
        type=float,
        default=0.0,
        help=(
            "Start surface distance of the optional locally refined MPC "
            "grid. The refinement is disabled when its factor is one."
        ),
    )
    parser.add_argument(
        "--mpc-local-refine-end-m",
        type=float,
        default=0.0,
        help="End surface distance of the optional locally refined MPC grid.",
    )
    parser.add_argument(
        "--mpc-local-refine-factor",
        type=int,
        default=1,
        help=(
            "Subdivision factor for base MPC intervals intersecting the "
            "local refinement window."
        ),
    )
    parser.add_argument(
        "--mpc-local-refine-window",
        type=float,
        nargs=3,
        action="append",
        default=[],
        metavar=("START_M", "END_M", "FACTOR"),
        help=(
            "Additional repeatable local MPC refinement window. FACTOR must "
            "be an integer >= 2. Overlapping windows use the largest factor."
        ),
    )
    parser.add_argument(
        "--mpc-auto-rephase-max-mm",
        type=float,
        default=0.0,
        help=(
            "Maximum per-fingertip meridian target offset available to the "
            "failure-triggered bounded joint-rephasing shooting pass. Zero "
            "disables the pass; all ordinary hard constraints remain active."
        ),
    )
    parser.add_argument(
        "--mpc-feasibility-bridge-max-mm",
        type=float,
        default=None,
        help=(
            "Optional phase bound used only by the short-interval predecessor "
            "feasibility bridge. The ordinary non-convex rephase search keeps "
            "--mpc-auto-rephase-max-mm, preserving its validated branch."
        ),
    )
    parser.add_argument(
        "--mpc-feasibility-bridge-trust-radius-rad",
        type=float,
        default=0.05,
        help=(
            "Per-joint trust radius for a moving short-interval feasibility "
            "bridge centred on the last accepted URDF state."
        ),
    )
    parser.add_argument(
        "--mpc-feasibility-bridge-min-progress-ratio",
        type=float,
        default=0.10,
        help=(
            "Minimum fraction of the short shooting interval that at least "
            "three fingertips must physically advance for a moving bridge."
        ),
    )
    parser.add_argument(
        "--mpc-feasibility-bridge-target-weight",
        type=float,
        default=3200.0,
        help=(
            "Dedicated along-surface target weight for the moving short-step "
            "bridge. This prevents a feasibility-band solution from remaining "
            "at the predecessor state."
        ),
    )
    parser.add_argument(
        "--mpc-feasibility-bridge-tip-target-scale",
        type=float,
        default=0.5,
        help=(
            "Scale applied to the ordinary physical-tip target residual "
            "inside each moving feasibility-bridge solve. The immutable "
            "physical-tip penetration gate is unchanged."
        ),
    )
    parser.add_argument(
        "--mpc-suffix-horizon-nodes",
        type=int,
        default=0,
        help=(
            "Number of strict moving-suffix nodes jointly optimized before "
            "accepting the first node. Zero disables the receding horizon."
        ),
    )
    parser.add_argument(
        "--mpc-suffix-min-joint-margin-mrad",
        type=float,
        default=0.5,
        help=(
            "Soft minimum joint-limit margin for every new suffix node. "
            "The historical anchor is exempt; formal joint limits remain hard."
        ),
    )
    parser.add_argument(
        "--mpc-suffix-min-task-margin-mm",
        type=float,
        default=0.05,
        help=(
            "Interior progress/normal/tangent/physical-tip margin targeted "
            "by the strict suffix solver without changing any hard limit."
        ),
    )
    parser.add_argument(
        "--mpc-suffix-max-nfev",
        type=int,
        default=160,
        help="Maximum evaluations for each H-node suffix least-squares solve.",
    )
    parser.add_argument(
        "--mpc-static-bridge-max-dwell-mm",
        type=float,
        default=1.50,
        help=(
            "Maximum consecutive route distance that short static feasibility "
            "bridges may cover before real fingertip motion must resume."
        ),
    )
    parser.add_argument(
        "--mpc-static-bridge-max-total-ratio",
        type=float,
        default=0.02,
        help=(
            "Maximum fraction of the full route covered by static feasibility "
            "bridges. This keeps physical motion dominant over brief pauses."
        ),
    )
    parser.add_argument(
        "--mpc-static-bridge-progress-tolerance-mm",
        type=float,
        default=6.00,
        help=(
            "Temporary fingertip progress tolerance used only by a bounded "
            "static bridge. Ordinary moving and final tolerances are unchanged."
        ),
    )
    parser.add_argument(
        "--mpc-recovery-bridge-max-span-mm",
        type=float,
        default=3.00,
        help=(
            "Maximum consecutive route distance occupied by static or relaxed "
            "moving recovery. A strict moving state must then be recovered."
        ),
    )
    parser.add_argument(
        "--mpc-recovery-bridge-max-total-ratio",
        type=float,
        default=0.03,
        help=(
            "Maximum fraction of the full route occupied by bounded recovery "
            "intervals, including both static and genuinely moving recovery."
        ),
    )
    parser.add_argument(
        "--mpc-recovery-bridge-progress-tolerance-mm",
        type=float,
        default=6.00,
        help=(
            "Temporary longitudinal tolerance used only by a bounded moving "
            "recovery bridge. Ordinary and terminal tolerances stay strict."
        ),
    )
    parser.add_argument(
        "--mpc-recovery-bridge-normal-tolerance-mm",
        type=float,
        default=6.50,
        help=(
            "Temporary all-finger surface-distance bound during a bounded "
            "recovery bridge; nominal contact is still measured by the "
            "ordinary normal tolerance."
        ),
    )
    parser.add_argument(
        "--mpc-recovery-bridge-min-contact-fingers",
        type=int,
        default=2,
        help=(
            "Minimum nominal fingertip contacts allowed only inside a bounded "
            "recovery bridge. Route-level and terminal contact checks remain "
            "stricter."
        ),
    )
    parser.add_argument(
        "--mpc-recovery-bridge-terminal-margin-mm",
        type=float,
        default=20.0,
        help=(
            "Final route distance in which recovery bridges are forbidden so "
            "the plan must finish on the ordinary four-fingertip branch."
        ),
    )
    parser.add_argument(
        "--mpc-auto-rephase-step-mm",
        type=float,
        default=0.05,
        help="Smallest target-offset increment tried by automatic rephasing.",
    )
    parser.add_argument(
        "--mpc-auto-rephase-decay-mm",
        type=float,
        default=0.02,
        help=(
            "Maximum offset removed per accepted keyframe so an automatic "
            "rephase returns continuously to the nominal fingertip route."
        ),
    )
    parser.add_argument(
        "--mpc-auto-rephase-margin-mm",
        type=float,
        default=0.5,
        help=(
            "Include fingertips this far inside the progress limit in a "
            "coupled rephasing trial when another fingertip violates it."
        ),
    )
    parser.add_argument(
        "--mpc-auto-refine-min-step-mm",
        type=float,
        default=0.0,
        help=(
            "Failure-triggered midpoint shooting refinement stops before a "
            "new interval would be shorter than this value. Zero disables "
            "automatic refinement."
        ),
    )
    parser.add_argument(
        "--mpc-auto-refine-max-insertions",
        type=int,
        default=0,
        help=(
            "Maximum midpoint shooting keyframes inserted after hard "
            "feasibility failures. Zero disables automatic refinement."
        ),
    )
    parser.add_argument(
        "--mpc-failure-prefix-output",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/debug/20_fr3_planning/"
            "adaptive_mpc_failure_prefix.npz"
        ),
        help=(
            "Stable NPZ written only when adaptive surface MPC coarse "
            "shooting exhausts its local refinement/recovery options. It "
            "contains the last feasible coarse prefix, rejected candidate, "
            "constraint metrics, and current rephase/recovery budgets; no "
            "video is produced. An existing file is never overwritten."
        ),
    )
    parser.add_argument("--mpc-max-nfev", type=int, default=120)
    parser.add_argument("--mpc-progress-tolerance-mm", type=float, default=4.0)
    parser.add_argument(
        "--mpc-intermediate-progress-tolerance-mm",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--mpc-monotonic-tolerance-mm",
        type=float,
        default=0.2,
        help="Maximum numerical backtracking allowed between MPC keyframes.",
    )
    parser.add_argument("--mpc-normal-tolerance-mm", type=float, default=3.0)
    parser.add_argument(
        "--min-planner-contact-fingers",
        type=int,
        default=4,
        help=(
            "Minimum number of physical fingertip pads that must remain "
            "inside the nominal MPC contact-standoff band at every planned "
            "frame."
        ),
    )
    parser.add_argument(
        "--transient-contact-finger",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help=(
            "Fingertip allowed to use the larger transient standoff band "
            "inside the bounded recovery window: 0=index, 1=middle, "
            "2=ring, 3=thumb."
        ),
    )
    parser.add_argument(
        "--transient-contact-start-m",
        type=float,
        default=0.0,
        help="Surface progress where the bounded one-finger swing starts.",
    )
    parser.add_argument(
        "--transient-contact-end-m",
        type=float,
        default=0.0,
        help=(
            "Surface progress where the swing finger must have recovered. "
            "A value not greater than the start disables transient contact."
        ),
    )
    parser.add_argument(
        "--transient-contact-recovery-start-m",
        type=float,
        default=None,
        help=(
            "Optional progress where the swing-finger tolerances begin "
            "smoothly narrowing back to their nominal values. Recovery "
            "finishes at --transient-contact-end-m."
        ),
    )
    parser.add_argument(
        "--transient-progress-recovery-end-m",
        type=float,
        default=None,
        help=(
            "Optional later endpoint for fingertip meridian-progress phase "
            "synchronization. Normal/tangential contact still recovers at "
            "--transient-contact-end-m; the progress band alone continues "
            "shrinking to its unchanged nominal value at this endpoint."
        ),
    )
    parser.add_argument(
        "--transient-contact-normal-recovery-start-m",
        type=float,
        default=None,
        help=(
            "Optional later recovery start for the swing-finger normal "
            "standoff only. This supports a lift-move-place schedule: "
            "meridian/tangential alignment can begin first, then the "
            "finger closes back onto the surface. The default reuses "
            "--transient-contact-recovery-start-m."
        ),
    )
    parser.add_argument(
        "--mpc-transient-normal-tolerance-mm",
        type=float,
        default=6.0,
        help=(
            "Maximum planned standoff error of the one scheduled swing "
            "finger. All support fingers keep --mpc-normal-tolerance-mm."
        ),
    )
    parser.add_argument(
        "--mpc-transient-tangential-tolerance-mm",
        type=float,
        default=None,
        help=(
            "Optional tangential tolerance for the one scheduled swing "
            "finger inside its bounded recovery window. Support fingers and "
            "all fingertips outside the window keep "
            "--mpc-tangential-tolerance-mm."
        ),
    )
    parser.add_argument(
        "--mpc-transient-progress-tolerance-mm",
        type=float,
        default=4.0,
        help=(
            "Intermediate meridian-progress tolerance used only inside the "
            "bounded one-finger recovery window. The ordinary intermediate "
            "and final tolerances remain unchanged outside it."
        ),
    )
    parser.add_argument(
        "--mpc-palm-position-tolerance-mm",
        type=float,
        default=3.0,
        help=(
            "Spherical feasibility tolerance for the non-contact palm-root "
            "MPC point. The optimizer may move inside this ball to preserve "
            "all four physical fingertip contacts and collision clearance."
        ),
    )
    parser.add_argument(
        "--palm-guide-only",
        action="store_true",
        help=(
            "Treat the non-contact palm-root target as a weak directional "
            "guide instead of a fingertip-like hard position target. Four "
            "fingertip constraints, joint limits, FR3 clearance, and hand "
            "contact limits remain hard."
        ),
    )
    parser.add_argument(
        "--palm-guide-max-drift-mm",
        type=float,
        default=30.0,
        help=(
            "Safety-only maximum palm-root drift from the directional guide "
            "when --palm-guide-only is active. The measured drift is always "
            "recorded and does not replace fingertip acceptance."
        ),
    )
    parser.add_argument(
        "--mpc-tangential-tolerance-mm",
        type=float,
        default=2.0,
        help="Maximum fingertip error along the circumferential surface tangent.",
    )
    parser.add_argument(
        "--contact-failure-window",
        type=int,
        default=20,
        help=(
            "Maximum consecutive evaluated frames below the configured "
            "majority-contact threshold."
        ),
    )
    parser.add_argument("--min-contact-force-n", type=float, default=0.10)
    parser.add_argument(
        "--max-tip-contact-force-n",
        type=float,
        default=25.0,
        help=(
            "Hard per-fingertip filtered normal-force limit after contact "
            "calibration. Exceeding it aborts the run."
        ),
    )
    parser.add_argument(
        "--max-tip-raw-force-n",
        type=float,
        default=40.0,
        help=(
            "Emergency cutoff for an unfiltered 3-D fingertip force sample. "
            "Raw peaks below this cutoff are still reported."
        ),
    )
    parser.add_argument("--min-contact-ratio", type=float, default=0.75)
    parser.add_argument(
        "--min-runtime-contact-fingers",
        type=int,
        default=3,
        help=(
            "Number of simultaneous tactile fingertip contacts counted as "
            "majority support. Brief drops below it are allowed."
        ),
    )
    parser.add_argument(
        "--min-majority-contact-ratio",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of evaluated motion frames with at least "
            "--min-runtime-contact-fingers physical pad contacts."
        ),
    )
    parser.add_argument(
        "--min-average-contact-fingers",
        type=float,
        default=3.0,
        help=(
            "Minimum time-average number of simultaneously contacting "
            "physical fingertip pads over the evaluated motion."
        ),
    )
    parser.add_argument(
        "--max-zero-contact-frames",
        type=int,
        default=10,
        help=(
            "Maximum consecutive evaluated frames with no physical fingertip "
            "pad contact. A brief complete release is allowed but must recover."
        ),
    )
    parser.add_argument(
        "--max-individual-contact-loss-frames",
        type=int,
        default=20,
        help=(
            "Maximum consecutive evaluated frames that any one fingertip "
            "may remain below the tactile contact-force threshold."
        ),
    )
    parser.add_argument(
        "--final-contact-recovery-frames",
        type=int,
        default=20,
        help=(
            "Required consecutive four-fingertip contact frames at the end "
            "of the route, proving that every transient loss recovered."
        ),
    )
    parser.add_argument(
        "--min-tip-surface-travel-m",
        type=float,
        default=0.17,
        help=(
            "Minimum measured physical fingertip-site meridian travel for "
            "every continuously contacting fingertip."
        ),
    )
    parser.add_argument(
        "--min-tip-relative-travel-m",
        type=float,
        default=0.004,
        help=(
            "Minimum change of each real contact point in the palm frame; "
            "rejects videos produced by rigid arm transport."
        ),
    )
    parser.add_argument(
        "--min-finger-joint-excursion-rad",
        type=float,
        default=0.08,
        help="Minimum peak-to-peak excursion of at least one joint per finger.",
    )
    parser.add_argument(
        "--max-contact-penetration-mm",
        type=float,
        default=1.0,
        help=(
            "Reject the video if MuJoCo reports deeper fingertip/object "
            "penetration at any time."
        ),
    )
    parser.add_argument(
        "--max-runtime-self-penetration-mm",
        type=float,
        default=0.01,
        help=(
            "Numerical penetration tolerance for robot/robot contacts during "
            "GPU execution. Object collisions remain governed by separate "
            "zero-frame collision guards."
        ),
    )
    parser.add_argument(
        "--min-meridian-curvature-ratio",
        type=float,
        default=1.0,
        help=(
            "Reject a planned route whose maximum/minimum positive meridian "
            "curvature ratio is smaller. A capsule route crossing a "
            "cylinder/end-cap boundary is reported as infinite variation."
        ),
    )
    parser.add_argument("--contact-settle-frames", type=int, default=3)
    parser.add_argument("--contact-calibration-start", type=int, default=15)
    parser.add_argument("--preshape-frames", type=int, default=0)
    parser.add_argument("--object-approach-frames", type=int, default=300)
    parser.add_argument("--object-retreat-distance-m", type=float, default=0.20)
    parser.add_argument(
        "--object-retreat-azimuth-deg",
        type=float,
        default=0.0,
        help=(
            "World-XY direction of the collision-checked object approach "
            "path, measured in degrees from +X."
        ),
    )
    parser.add_argument(
        "--object-retreat-direction-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Side used to initialize the object away from the hand.",
    )
    parser.add_argument(
        "--planning-center-shift-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--axial-direction",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="-1 follows the opposite object meridian direction.",
    )
    parser.add_argument("--contact-tracking-radius-rad", type=float, default=0.05)
    parser.add_argument(
        "--precontact-tracking-radius-rad",
        type=float,
        default=0.02,
    )
    parser.add_argument("--contact-search-step-rad", type=float, default=0.02)
    parser.add_argument(
        "--contact-search-step-mm",
        type=float,
        default=0.15,
        help=(
            "Per-frame Cartesian inward search step for a fingertip that has "
            "not contacted. Search starts only after object approach ends."
        ),
    )
    parser.add_argument("--contact-search-limit-rad", type=float, default=0.30)
    parser.add_argument("--finger-force-n", type=float, default=3.0)
    parser.add_argument(
        "--finger-max-calibrated-force-n",
        type=float,
        default=12.0,
        help=(
            "Maximum per-finger loaded direct-force operating setpoint "
            "captured immediately before surface motion."
        ),
    )
    parser.add_argument(
        "--finger-admittance-mass-kg",
        type=float,
        default=0.08,
        help="Virtual mass of each scalar fingertip normal admittance loop.",
    )
    parser.add_argument(
        "--finger-admittance-damping-n-s-m",
        type=float,
        default=18.0,
        help="Virtual damping of each fingertip normal admittance loop.",
    )
    parser.add_argument(
        "--finger-admittance-stiffness-n-m",
        type=float,
        default=1000.0,
        help="Virtual stiffness of each fingertip normal admittance loop.",
    )
    parser.add_argument("--finger-force-gain", type=float, default=1.0)
    parser.add_argument("--finger-force-filter-alpha", type=float, default=0.25)
    parser.add_argument("--finger-contact-on-force-n", type=float, default=0.15)
    parser.add_argument("--finger-contact-off-force-n", type=float, default=0.08)
    parser.add_argument(
        "--finger-max-normal-offset-mm",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--finger-max-normal-speed-mm-s",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--finger-max-normal-acceleration-m-s2",
        type=float,
        default=0.2,
    )
    # Accepted only so archived experiment commands remain parseable.  These
    # static displacement parameters are not part of Baseline-2 admittance.
    parser.add_argument(
        "--finger-normal-preload-mm",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finger-normal-preload-scales",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finger-normal-compliance-mm-per-n",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finger-max-release-correction-rad",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--palm-force-n", type=float, default=0.0)
    parser.add_argument("--arm-mcc-correction-rad", type=float, default=0.003)
    parser.add_argument("--wrist-update-decimation", type=int, default=4)
    parser.add_argument("--wrist-damping-ratio", type=float, default=1.0)
    parser.add_argument("--wrist-max-force-error-n", type=float, default=5.0)
    parser.add_argument("--wrist-max-torque-error-nm", type=float, default=0.8)
    parser.add_argument(
        "--wrist-max-translation-offset-mm",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--wrist-max-rotation-offset-rad",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--arm-trajectory-tracking-gain",
        type=float,
        default=2.0,
        help="Joint-space lead compensation for the loaded moving arm.",
    )
    parser.add_argument(
        "--finger-trajectory-tracking-gain",
        type=float,
        default=0.5,
        help="Joint-space lead compensation for loaded finger servos.",
    )
    parser.add_argument(
        "--arm-servo-load-scale",
        type=float,
        default=1.0,
        help="Scale the calibrated arm command-to-loaded-position offset.",
    )
    parser.add_argument(
        "--finger-servo-load-scale",
        type=float,
        default=1.5,
        help="Scale the calibrated finger contact preload offset.",
    )
    parser.add_argument("--print-every", type=int, default=300)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance-m", type=float, default=0.82)
    parser.add_argument("--camera-elevation-deg", type=float, default=-18.0)
    parser.add_argument("--camera-azimuth-deg", type=float, default=145.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/debug/20_fr3_planning/"
            "baseline2_candidate.mp4"
        ),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/debug/20_fr3_planning/"
            "baseline2_plan.npz"
        ),
    )
    parser.add_argument(
        "--reuse-plan",
        type=Path,
        default=None,
        help="Reuse a previously validated NPZ plan for controller tuning.",
    )
    parser.add_argument(
        "--initial-grasp",
        type=Path,
        default=None,
        help=(
            "Load an optimized collision-free 23-DoF initial pose and object "
            "center from an NPZ produced by optimize_full_robot_grasp.py."
        ),
    )
    args = parser.parse_args()
    if args.steps <= args.motion_start:
        raise ValueError("--steps must be greater than --motion-start")
    if args.seed < 0:
        raise ValueError("--seed cannot be negative")
    if args.planner_state_quantization_rad < 0.0:
        raise ValueError("--planner-state-quantization-rad cannot be negative")
    if args.axial_travel_m <= 0.0:
        raise ValueError("--axial-travel-m must be positive")
    if not 0.0 <= args.palm_travel_ratio <= 1.0:
        raise ValueError("--palm-travel-ratio must be in [0, 1]")
    if args.palm_late_follow_start_m < 0.0:
        raise ValueError("--palm-late-follow-start-m cannot be negative")
    if not 0.0 <= args.palm_late_follow_ratio <= 1.0:
        raise ValueError("--palm-late-follow-ratio must be in [0, 1]")
    if args.palm_late_follow_ramp_m <= 0.0:
        raise ValueError("--palm-late-follow-ramp-m must be positive")
    if args.finger_gait_amplitude_m < 0.0:
        raise ValueError("--finger-gait-amplitude-m cannot be negative")
    if args.finger_meridian_gait_mm < 0.0:
        raise ValueError("--finger-meridian-gait-mm cannot be negative")
    if args.finger_meridian_gait_start_m < 0.0:
        raise ValueError(
            "--finger-meridian-gait-start-m cannot be negative"
        )
    if (
        args.finger_meridian_gait_end_m
        <= args.finger_meridian_gait_start_m
    ):
        raise ValueError(
            "--finger-meridian-gait-end-m must be greater than "
            "--finger-meridian-gait-start-m"
        )
    if any(scale < 0.0 for scale in args.finger_meridian_gait_scales):
        raise ValueError(
            "--finger-meridian-gait-scales cannot be negative"
        )
    if args.finger_meridian_correction_mm < 0.0:
        raise ValueError(
            "--finger-meridian-correction-mm cannot be negative"
        )
    if args.finger_meridian_correction_start_m < 0.0:
        raise ValueError(
            "--finger-meridian-correction-start-m cannot be negative"
        )
    if (
        args.finger_meridian_correction_end_m
        <= args.finger_meridian_correction_start_m
    ):
        raise ValueError(
            "--finger-meridian-correction-end-m must be greater than "
            "--finger-meridian-correction-start-m"
        )
    if any(
        scale < 0.0
        for scale in args.finger_meridian_correction_scales
    ):
        raise ValueError(
            "--finger-meridian-correction-scales cannot be negative"
        )
    if args.finger_meridian_terminal_correction_mm < 0.0:
        raise ValueError(
            "--finger-meridian-terminal-correction-mm cannot be negative"
        )
    if (
        args.finger_meridian_terminal_correction_mm > 0.0
        and (
            args.finger_meridian_terminal_correction_start_m < 0.0
            or args.finger_meridian_terminal_correction_end_m
            <= args.finger_meridian_terminal_correction_start_m
            or args.finger_meridian_terminal_correction_end_m
            > args.axial_travel_m
        )
    ):
        raise ValueError(
            "Terminal finger-meridian correction requires an ordered "
            "window inside --axial-travel-m"
        )
    if args.finger_meridian_terminal_tail_correction_mm < 0.0:
        raise ValueError(
            "--finger-meridian-terminal-tail-correction-mm cannot be "
            "negative"
        )
    if (
        args.finger_meridian_terminal_tail_correction_mm > 0.0
        and (
            args.finger_meridian_terminal_tail_correction_start_m < 0.0
            or args.finger_meridian_terminal_tail_correction_end_m
            <= args.finger_meridian_terminal_tail_correction_start_m
            or args.finger_meridian_terminal_tail_correction_end_m
            > args.axial_travel_m
        )
    ):
        raise ValueError(
            "Terminal tail finger-meridian correction requires an ordered "
            "window inside --axial-travel-m"
        )
    for local_phase_spec in args.finger_meridian_local_phase:
        amplitude_mm, start_m, end_m, *_ = local_phase_spec
        if amplitude_mm < 0.0:
            raise ValueError(
                "--finger-meridian-local-phase amplitude cannot be negative"
            )
        if (
            start_m < 0.0
            or end_m <= start_m
            or end_m > args.axial_travel_m
        ):
            raise ValueError(
                "--finger-meridian-local-phase requires an ordered window "
                "inside --axial-travel-m"
            )
    if args.runtime_finger_gait_rad < 0.0:
        raise ValueError("--runtime-finger-gait-rad cannot be negative")
    if args.runtime_tip_gait_mm < 0.0:
        raise ValueError("--runtime-tip-gait-mm cannot be negative")
    if args.runtime_gait_cycles < 0.0:
        raise ValueError("--runtime-gait-cycles cannot be negative")
    if any(scale < 0.0 for scale in args.runtime_gait_finger_scales):
        raise ValueError("--runtime-gait-finger-scales cannot be negative")
    if args.object_radius_m <= 0.0:
        raise ValueError("--object-radius-m must be positive")
    if args.object_half_height_m < 0.0:
        raise ValueError("--object-half-height-m cannot be negative")
    if args.object_shape == "ellipsoid" and args.object_half_height_m <= 0.0:
        raise ValueError("Ellipsoid --object-half-height-m must be positive")
    if args.min_meridian_curvature_ratio < 1.0:
        raise ValueError("--min-meridian-curvature-ratio must be at least one")
    if args.waypoint_count < 1:
        raise ValueError("--waypoint-count must be at least one")
    if not 0.0 < args.stage_move_fraction <= 1.0:
        raise ValueError("--stage-move-fraction must be in (0, 1]")
    if args.surface_preload_mm < 0.0:
        raise ValueError("--surface-preload-mm cannot be negative")
    if args.mpc_keyframes < 2:
        raise ValueError("--mpc-keyframes must be at least two")
    if args.mpc_local_refine_factor < 1:
        raise ValueError("--mpc-local-refine-factor must be at least one")
    if args.mpc_local_refine_start_m < 0.0:
        raise ValueError("--mpc-local-refine-start-m cannot be negative")
    if args.mpc_local_refine_end_m > args.axial_travel_m:
        raise ValueError(
            "--mpc-local-refine-end-m cannot exceed --axial-travel-m"
        )
    if (
        args.mpc_local_refine_factor > 1
        and args.mpc_local_refine_end_m
        <= args.mpc_local_refine_start_m
    ):
        raise ValueError(
            "Local MPC refinement requires its end to be greater than "
            "its start"
        )
    mpc_local_refine_windows: list[tuple[float, float, int]] = []
    for refine_start_m, refine_end_m, refine_factor_raw in (
        args.mpc_local_refine_window
    ):
        refine_factor = int(round(refine_factor_raw))
        if not np.isclose(refine_factor_raw, refine_factor):
            raise ValueError(
                "--mpc-local-refine-window FACTOR must be an integer"
            )
        if refine_factor < 2:
            raise ValueError(
                "--mpc-local-refine-window FACTOR must be at least two"
            )
        if (
            refine_start_m < 0.0
            or refine_end_m <= refine_start_m
            or refine_end_m > args.axial_travel_m
        ):
            raise ValueError(
                "--mpc-local-refine-window requires an ordered window "
                "inside --axial-travel-m"
            )
        mpc_local_refine_windows.append(
            (refine_start_m, refine_end_m, refine_factor)
        )
    args.mpc_local_refine_window = mpc_local_refine_windows
    if args.mpc_auto_rephase_max_mm < 0.0:
        raise ValueError("--mpc-auto-rephase-max-mm cannot be negative")
    if args.mpc_feasibility_bridge_max_mm is None:
        args.mpc_feasibility_bridge_max_mm = (
            args.mpc_auto_rephase_max_mm
        )
    if args.mpc_feasibility_bridge_max_mm < 0.0:
        raise ValueError(
            "--mpc-feasibility-bridge-max-mm cannot be negative"
        )
    if args.mpc_feasibility_bridge_trust_radius_rad <= 0.0:
        raise ValueError(
            "--mpc-feasibility-bridge-trust-radius-rad must be positive"
        )
    if not (
        0.0
        <= args.mpc_feasibility_bridge_min_progress_ratio
        <= 1.0
    ):
        raise ValueError(
            "--mpc-feasibility-bridge-min-progress-ratio must lie in [0, 1]"
        )
    if args.mpc_feasibility_bridge_target_weight <= 0.0:
        raise ValueError(
            "--mpc-feasibility-bridge-target-weight must be positive"
        )
    if (
        not np.isfinite(args.mpc_feasibility_bridge_tip_target_scale)
        or args.mpc_feasibility_bridge_tip_target_scale < 0.0
    ):
        raise ValueError(
            "--mpc-feasibility-bridge-tip-target-scale must be finite and "
            "non-negative"
        )
    if args.mpc_suffix_horizon_nodes < 0:
        raise ValueError("--mpc-suffix-horizon-nodes cannot be negative")
    if args.mpc_suffix_horizon_nodes == 1:
        raise ValueError(
            "--mpc-suffix-horizon-nodes must be zero or at least two"
        )
    if (
        not np.isfinite(args.mpc_suffix_min_joint_margin_mrad)
        or args.mpc_suffix_min_joint_margin_mrad < 0.0
    ):
        raise ValueError(
            "--mpc-suffix-min-joint-margin-mrad must be finite and non-negative"
        )
    if (
        not np.isfinite(args.mpc_suffix_min_task_margin_mm)
        or args.mpc_suffix_min_task_margin_mm < 0.0
    ):
        raise ValueError(
            "--mpc-suffix-min-task-margin-mm must be finite and non-negative"
        )
    if args.mpc_suffix_max_nfev <= 0:
        raise ValueError("--mpc-suffix-max-nfev must be positive")
    if args.mpc_static_bridge_max_dwell_mm < 0.0:
        raise ValueError("--mpc-static-bridge-max-dwell-mm cannot be negative")
    if not 0.0 <= args.mpc_static_bridge_max_total_ratio <= 1.0:
        raise ValueError(
            "--mpc-static-bridge-max-total-ratio must lie in [0, 1]"
        )
    if args.mpc_static_bridge_progress_tolerance_mm <= 0.0:
        raise ValueError(
            "--mpc-static-bridge-progress-tolerance-mm must be positive"
        )
    if args.mpc_recovery_bridge_max_span_mm <= 0.0:
        raise ValueError("--mpc-recovery-bridge-max-span-mm must be positive")
    if not 0.0 <= args.mpc_recovery_bridge_max_total_ratio <= 1.0:
        raise ValueError(
            "--mpc-recovery-bridge-max-total-ratio must lie in [0, 1]"
        )
    if args.mpc_recovery_bridge_progress_tolerance_mm <= 0.0:
        raise ValueError(
            "--mpc-recovery-bridge-progress-tolerance-mm must be positive"
        )
    if (
        args.mpc_recovery_bridge_normal_tolerance_mm
        < args.mpc_normal_tolerance_mm
    ):
        raise ValueError(
            "--mpc-recovery-bridge-normal-tolerance-mm cannot be below "
            "--mpc-normal-tolerance-mm"
        )
    if not 1 <= args.mpc_recovery_bridge_min_contact_fingers <= 4:
        raise ValueError(
            "--mpc-recovery-bridge-min-contact-fingers must be in [1, 4]"
        )
    if args.mpc_recovery_bridge_terminal_margin_mm < 0.0:
        raise ValueError(
            "--mpc-recovery-bridge-terminal-margin-mm cannot be negative"
        )
    if args.mpc_auto_rephase_step_mm <= 0.0:
        raise ValueError("--mpc-auto-rephase-step-mm must be positive")
    if args.mpc_auto_rephase_decay_mm < 0.0:
        raise ValueError("--mpc-auto-rephase-decay-mm cannot be negative")
    if args.mpc_auto_rephase_margin_mm < 0.0:
        raise ValueError("--mpc-auto-rephase-margin-mm cannot be negative")
    if (
        args.mpc_auto_rephase_max_mm > 0.0
        and args.mpc_auto_rephase_step_mm
        > args.mpc_auto_rephase_max_mm
    ):
        raise ValueError(
            "--mpc-auto-rephase-step-mm cannot exceed "
            "--mpc-auto-rephase-max-mm"
        )
    if args.mpc_auto_refine_min_step_mm < 0.0:
        raise ValueError("--mpc-auto-refine-min-step-mm cannot be negative")
    if args.mpc_auto_refine_max_insertions < 0:
        raise ValueError(
            "--mpc-auto-refine-max-insertions cannot be negative"
        )
    if (
        (args.mpc_auto_refine_min_step_mm > 0.0)
        != (args.mpc_auto_refine_max_insertions > 0)
    ):
        raise ValueError(
            "Automatic MPC refinement requires both "
            "--mpc-auto-refine-min-step-mm and "
            "--mpc-auto-refine-max-insertions, or neither"
        )
    if args.mpc_normal_tolerance_mm <= 0.0:
        raise ValueError("--mpc-normal-tolerance-mm must be positive")
    if not 1 <= args.min_planner_contact_fingers <= 4:
        raise ValueError("--min-planner-contact-fingers must be in [1, 4]")
    if args.transient_contact_start_m < 0.0:
        raise ValueError("--transient-contact-start-m cannot be negative")
    if (
        args.min_planner_contact_fingers < 4
        and args.transient_contact_end_m
        <= args.transient_contact_start_m
    ):
        raise ValueError(
            "Three-support planning requires "
            "--transient-contact-end-m greater than its start"
        )
    if args.transient_contact_end_m > args.axial_travel_m:
        raise ValueError(
            "--transient-contact-end-m cannot exceed --axial-travel-m"
        )
    if (
        args.transient_contact_recovery_start_m is not None
        and not (
            args.transient_contact_start_m
            < args.transient_contact_recovery_start_m
            < args.transient_contact_end_m
        )
    ):
        raise ValueError(
            "--transient-contact-recovery-start-m must lie strictly inside "
            "the transient contact window"
        )
    if (
        args.transient_progress_recovery_end_m is not None
        and (
            args.transient_progress_recovery_end_m
            < args.transient_contact_end_m
            or args.transient_progress_recovery_end_m
            > args.axial_travel_m
            or (
                args.transient_contact_recovery_start_m is not None
                and args.transient_progress_recovery_end_m
                <= args.transient_contact_recovery_start_m
            )
        )
    ):
        raise ValueError(
            "--transient-progress-recovery-end-m must be at least the "
            "contact end, greater than the recovery start, and no greater "
            "than --axial-travel-m"
        )
    if (
        args.transient_contact_normal_recovery_start_m is not None
        and not (
            args.transient_contact_start_m
            < args.transient_contact_normal_recovery_start_m
            < args.transient_contact_end_m
        )
    ):
        raise ValueError(
            "--transient-contact-normal-recovery-start-m must lie strictly "
            "inside the transient contact window"
        )
    if args.mpc_transient_normal_tolerance_mm < args.mpc_normal_tolerance_mm:
        raise ValueError(
            "--mpc-transient-normal-tolerance-mm cannot be smaller than "
            "--mpc-normal-tolerance-mm"
        )
    if (
        args.mpc_transient_tangential_tolerance_mm is not None
        and args.mpc_transient_tangential_tolerance_mm
        < args.mpc_tangential_tolerance_mm
    ):
        raise ValueError(
            "--mpc-transient-tangential-tolerance-mm cannot be smaller "
            "than --mpc-tangential-tolerance-mm"
        )
    if (
        args.mpc_transient_progress_tolerance_mm
        < args.mpc_intermediate_progress_tolerance_mm
    ):
        raise ValueError(
            "--mpc-transient-progress-tolerance-mm cannot be smaller than "
            "--mpc-intermediate-progress-tolerance-mm"
        )
    if args.mpc_monotonic_tolerance_mm < 0.0:
        raise ValueError("--mpc-monotonic-tolerance-mm cannot be negative")
    if args.mpc_palm_position_tolerance_mm <= 0.0:
        raise ValueError(
            "--mpc-palm-position-tolerance-mm must be positive"
        )
    if args.palm_guide_max_drift_mm <= 0.0:
        raise ValueError("--palm-guide-max-drift-mm must be positive")
    if args.palm_path_tolerance_mm < args.ik_tolerance_mm:
        raise ValueError(
            "--palm-path-tolerance-mm must be at least --ik-tolerance-mm"
        )
    if args.contact_search_step_rad <= 0.0:
        raise ValueError("--contact-search-step-rad must be positive")
    if args.contact_search_step_mm <= 0.0:
        raise ValueError("--contact-search-step-mm must be positive")
    if args.contact_search_limit_rad <= 0.0:
        raise ValueError("--contact-search-limit-rad must be positive")
    if not 1 <= args.min_runtime_contact_fingers <= 4:
        raise ValueError("--min-runtime-contact-fingers must be in [1, 4]")
    if not 0.0 <= args.min_contact_ratio <= 1.0:
        raise ValueError("--min-contact-ratio must lie in [0, 1]")
    if not 0.0 <= args.min_majority_contact_ratio <= 1.0:
        raise ValueError("--min-majority-contact-ratio must lie in [0, 1]")
    if not 0.0 <= args.min_average_contact_fingers <= 4.0:
        raise ValueError("--min-average-contact-fingers must lie in [0, 4]")
    if args.contact_failure_window < 0:
        raise ValueError("--contact-failure-window cannot be negative")
    if args.max_zero_contact_frames < 0:
        raise ValueError("--max-zero-contact-frames cannot be negative")
    if args.max_individual_contact_loss_frames < 0:
        raise ValueError(
            "--max-individual-contact-loss-frames cannot be negative"
        )
    if args.final_contact_recovery_frames < 1:
        raise ValueError("--final-contact-recovery-frames must be positive")
    if args.finger_max_release_correction_rad < 0.0:
        raise ValueError(
            "--finger-max-release-correction-rad cannot be negative"
        )
    if args.finger_normal_preload_mm < 0.0:
        raise ValueError("--finger-normal-preload-mm cannot be negative")
    if any(scale < 0.0 for scale in args.finger_normal_preload_scales):
        raise ValueError("--finger-normal-preload-scales cannot be negative")
    if args.finger_normal_compliance_mm_per_n < 0.0:
        raise ValueError(
            "--finger-normal-compliance-mm-per-n cannot be negative"
        )
    if min(
        args.finger_admittance_mass_kg,
        args.finger_admittance_damping_n_s_m,
        args.finger_admittance_stiffness_n_m,
        args.finger_force_gain,
        args.finger_max_normal_offset_mm,
        args.finger_max_normal_speed_mm_s,
        args.finger_max_normal_acceleration_m_s2,
    ) <= 0.0:
        raise ValueError("Fingertip admittance gains and limits must be positive")
    if args.finger_max_calibrated_force_n < args.finger_force_n:
        raise ValueError(
            "--finger-max-calibrated-force-n must be at least "
            "--finger-force-n"
        )
    if args.max_tip_contact_force_n < args.finger_max_calibrated_force_n:
        raise ValueError(
            "--max-tip-contact-force-n must be at least "
            "--finger-max-calibrated-force-n"
        )
    if args.max_tip_raw_force_n < args.max_tip_contact_force_n:
        raise ValueError(
            "--max-tip-raw-force-n must be at least "
            "--max-tip-contact-force-n"
        )
    if not 0.0 <= args.finger_force_filter_alpha <= 1.0:
        raise ValueError("--finger-force-filter-alpha must lie in [0, 1]")
    if not (
        0.0
        <= args.finger_contact_off_force_n
        <= args.finger_contact_on_force_n
    ):
        raise ValueError(
            "Finger contact thresholds require 0 <= off <= on"
        )
    if args.arm_mcc_correction_rad < 0.0:
        raise ValueError("--arm-mcc-correction-rad cannot be negative")
    if args.wrist_update_decimation < 1:
        raise ValueError("--wrist-update-decimation must be positive")
    if min(
        args.wrist_damping_ratio,
        args.wrist_max_force_error_n,
        args.wrist_max_torque_error_nm,
        args.wrist_max_translation_offset_mm,
        args.wrist_max_rotation_offset_rad,
    ) <= 0.0:
        raise ValueError("Wrist admittance damping and limits must be positive")
    if args.arm_trajectory_tracking_gain < 0.0:
        raise ValueError("--arm-trajectory-tracking-gain cannot be negative")
    if args.finger_trajectory_tracking_gain < 0.0:
        raise ValueError(
            "--finger-trajectory-tracking-gain cannot be negative"
        )
    if args.arm_servo_load_scale < 0.0:
        raise ValueError("--arm-servo-load-scale cannot be negative")
    if args.finger_servo_load_scale < 0.0:
        raise ValueError("--finger-servo-load-scale cannot be negative")
    if args.max_contact_penetration_mm < 0.0:
        raise ValueError("--max-contact-penetration-mm cannot be negative")
    if args.max_runtime_self_penetration_mm < 0.0:
        raise ValueError(
            "--max-runtime-self-penetration-mm cannot be negative"
        )
    if args.min_arm_clearance_mm < 0.0:
        raise ValueError("--min-arm-clearance-mm cannot be negative")
    if args.max_incidental_hand_penetration_mm < 0.0:
        raise ValueError(
            "--max-incidental-hand-penetration-mm cannot be negative"
        )
    if args.max_incidental_hand_contact_force_n <= 0.0:
        raise ValueError(
            "--max-incidental-hand-contact-force-n must be positive"
        )
    if args.max_incidental_hand_total_force_n <= 0.0:
        raise ValueError(
            "--max-incidental-hand-total-force-n must be positive"
        )
    if (
        args.max_incidental_hand_total_force_n
        < args.max_incidental_hand_contact_force_n
    ):
        raise ValueError(
            "--max-incidental-hand-total-force-n cannot be smaller than "
            "--max-incidental-hand-contact-force-n"
        )
    if not 0.0 < args.max_pad_angle_deg < 90.0:
        raise ValueError("--max-pad-angle-deg must be in (0, 90)")
    if not 0.0 <= args.planner_pad_angle_margin_deg < args.max_pad_angle_deg:
        raise ValueError(
            "--planner-pad-angle-margin-deg must be in "
            "[0, --max-pad-angle-deg)"
        )
    planner_hard_pad_angle_deg = (
        args.max_pad_angle_deg - args.planner_pad_angle_margin_deg
    )
    if (
        not np.isfinite(args.planner_soft_pad_angle_deg)
        or not 0.0
        < args.planner_soft_pad_angle_deg
        <= planner_hard_pad_angle_deg
    ):
        raise ValueError(
            "--planner-soft-pad-angle-deg must be positive and no larger "
            "than the hard planner cone"
        )
    if (
        not np.isfinite(args.planner_soft_pad_weight)
        or args.planner_soft_pad_weight <= 0.0
    ):
        raise ValueError("--planner-soft-pad-weight must be positive")
    if (
        not np.isfinite(args.planner_soft_pad_softplus_tau)
        or args.planner_soft_pad_softplus_tau <= 0.0
    ):
        raise ValueError(
            "--planner-soft-pad-softplus-tau must be positive"
        )
    planner_tip_geom_target_mm = np.asarray(
        args.planner_tip_geom_target_mm,
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(planner_tip_geom_target_mm))
        or np.any(planner_tip_geom_target_mm > 0.0)
        or np.any(
            planner_tip_geom_target_mm
            < args.planner_tip_geom_inner_cap_mm
        )
    ):
        raise ValueError(
            "--planner-tip-geom-target-mm must be finite and lie between "
            "the soft inner cap and zero"
        )
    if (
        not np.isfinite(args.planner_tip_geom_inner_cap_mm)
        or args.planner_tip_geom_inner_cap_mm > 0.0
        or args.planner_tip_geom_inner_cap_mm
        < -args.max_contact_penetration_mm
    ):
        raise ValueError(
            "--planner-tip-geom-inner-cap-mm must stay inside the hard "
            "tip penetration band"
        )
    for option, value in (
        ("--planner-tip-geom-weight", args.planner_tip_geom_weight),
        (
            "--planner-tip-geom-inner-weight",
            args.planner_tip_geom_inner_weight,
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{option} must be finite and positive")
    if (
        not np.isfinite(args.planner_protected_self_clearance_mm)
        or args.planner_protected_self_clearance_mm < 0.0
    ):
        raise ValueError(
            "--planner-protected-self-clearance-mm must be finite and "
            "non-negative"
        )
    for option, value in (
        (
            "--planner-protected-self-clearance-weight",
            args.planner_protected_self_clearance_weight,
        ),
        (
            "--planner-self-separation-seed-step-rad",
            args.planner_self_separation_seed_step_rad,
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{option} must be finite and positive")
    if not 0.0 <= args.palm_surface_frame_gain <= 1.0:
        raise ValueError("--palm-surface-frame-gain must be in [0, 1]")
    if (
        args.palm_surface_frame_late_gain is not None
        and not 0.0 <= args.palm_surface_frame_late_gain <= 1.0
    ):
        raise ValueError("--palm-surface-frame-late-gain must be in [0, 1]")
    if args.palm_surface_frame_late_start_m < 0.0:
        raise ValueError(
            "--palm-surface-frame-late-start-m cannot be negative"
        )
    if args.palm_surface_frame_late_ramp_m <= 0.0:
        raise ValueError(
            "--palm-surface-frame-late-ramp-m must be positive"
        )
    if (
        args.palm_surface_frame_terminal_gain is not None
        and not 0.0 <= args.palm_surface_frame_terminal_gain <= 1.0
    ):
        raise ValueError(
            "--palm-surface-frame-terminal-gain must be in [0, 1]"
        )
    if args.palm_surface_frame_terminal_start_m < 0.0:
        raise ValueError(
            "--palm-surface-frame-terminal-start-m cannot be negative"
        )
    if args.palm_surface_frame_terminal_ramp_m <= 0.0:
        raise ValueError(
            "--palm-surface-frame-terminal-ramp-m must be positive"
        )
    if args.palm_clearance_lift_m < 0.0:
        raise ValueError("--palm-clearance-lift-m cannot be negative")
    if args.palm_clearance_ramp_m <= 0.0:
        raise ValueError("--palm-clearance-ramp-m must be positive")
    if args.palm_clearance_tilt_deg < 0.0:
        raise ValueError("--palm-clearance-tilt-deg cannot be negative")
    if args.palm_clearance_tilt_start_m < 0.0:
        raise ValueError(
            "--palm-clearance-tilt-start-m cannot be negative"
        )
    if args.palm_clearance_tilt_ramp_m <= 0.0:
        raise ValueError(
            "--palm-clearance-tilt-ramp-m must be positive"
        )
    if args.palm_clearance_tilt_release_start_m < 0.0:
        raise ValueError(
            "--palm-clearance-tilt-release-start-m cannot be negative"
        )
    if args.palm_clearance_tilt_release_ramp_m <= 0.0:
        raise ValueError(
            "--palm-clearance-tilt-release-ramp-m must be positive"
        )
    if args.palm_clearance_secondary_lift_m < 0.0:
        raise ValueError(
            "--palm-clearance-secondary-lift-m cannot be negative"
        )
    if args.palm_clearance_secondary_start_m < 0.0:
        raise ValueError(
            "--palm-clearance-secondary-start-m cannot be negative"
        )
    if args.palm_clearance_secondary_ramp_m <= 0.0:
        raise ValueError(
            "--palm-clearance-secondary-ramp-m must be positive"
        )
    if args.palm_terminal_local_offset_start_m < 0.0:
        raise ValueError(
            "--palm-terminal-local-offset-start-m cannot be negative"
        )
    if args.palm_terminal_local_offset_ramp_m <= 0.0:
        raise ValueError(
            "--palm-terminal-local-offset-ramp-m must be positive"
        )
    if (
        np.linalg.norm(
            np.asarray(
                args.palm_terminal_local_offset_mm,
                dtype=np.float64,
            )
        )
        > 3.0
    ):
        raise ValueError(
            "--palm-terminal-local-offset-mm must remain inside a 3 mm ball"
        )
    if args.palm_terminal_second_local_offset_start_m < 0.0:
        raise ValueError(
            "--palm-terminal-second-local-offset-start-m cannot be negative"
        )
    if args.palm_terminal_second_local_offset_ramp_m <= 0.0:
        raise ValueError(
            "--palm-terminal-second-local-offset-ramp-m must be positive"
        )
    terminal_palm_offset_mm = np.asarray(
        args.palm_terminal_local_offset_mm,
        dtype=np.float64,
    )
    terminal_second_palm_offset_mm = np.asarray(
        args.palm_terminal_second_local_offset_mm,
        dtype=np.float64,
    )
    if np.linalg.norm(terminal_second_palm_offset_mm) > 0.0:
        first_stage_end_m = (
            args.palm_terminal_local_offset_start_m
            + args.palm_terminal_local_offset_ramp_m
        )
        if (
            args.palm_terminal_second_local_offset_start_m
            < first_stage_end_m
        ):
            raise ValueError(
                "The second terminal palm correction must start after the "
                "first correction is fully established"
            )
    if (
        np.linalg.norm(
            terminal_palm_offset_mm + terminal_second_palm_offset_mm
        )
        > 3.0
    ):
        raise ValueError(
            "Combined terminal palm corrections must remain inside a 3 mm "
            "ball"
        )

    CAPSULE_RADIUS = args.object_radius_m
    CAPSULE_HALF_HEIGHT = args.object_half_height_m
    if args.object_shape == "ellipsoid":
        capsule_project = ellipsoid_project
        capsule_meridian_coordinates = ellipsoid_meridian_coordinates
        capsule_meridian_targets = ellipsoid_meridian_targets
        surface_meridian_curvature = ellipsoid_meridian_curvature
        SURFACE_TOTAL_LENGTH = ellipsoid_meridian_total_length(
            CAPSULE_RADIUS,
            CAPSULE_HALF_HEIGHT,
        )
    else:
        SURFACE_TOTAL_LENGTH = (
            np.pi * CAPSULE_RADIUS + 2.0 * CAPSULE_HALF_HEIGHT
        )
    full_hand_env_module.FULL_HAND_CAPSULE_RADIUS = CAPSULE_RADIUS
    full_hand_env_module.FULL_HAND_CAPSULE_HALF_HEIGHT = CAPSULE_HALF_HEIGHT
    full_hand_env_module.FULL_HAND_OBJECT_SHAPE = args.object_shape
    full_hand_env_module.FULL_HAND_COLLISION_MODE = args.collision_mode

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = full_hand_mcc_env_cfg(num_envs=1, play=True)
    env_cfg.seed = args.seed
    if args.initial_grasp is not None:
        optimized_grasp = np.load(args.initial_grasp)
        optimized_q = np.asarray(
            optimized_grasp["joint_position_rad"],
            dtype=np.float64,
        ).reshape(TOTAL_DOF)
        optimized_center = np.asarray(
            optimized_grasp["object_center_m"],
            dtype=np.float64,
        ).reshape(3)
        robot_joint_pos = env_cfg.scene.entities[
            "robot"
        ].init_state.joint_pos
        for joint_name, value in zip(
            ARM_JOINT_NAMES,
            optimized_q[:ARM_DOF],
            strict=True,
        ):
            robot_joint_pos[f"^{joint_name}$"] = float(value)
        for joint_name, value in zip(
            full_hand_env_module.HAND_QPOS_NAMES,
            optimized_q[ARM_DOF:TOTAL_DOF],
            strict=True,
        ):
            robot_joint_pos[f"^{joint_name}$"] = float(value)
        (
            args.object_center_x_m,
            args.object_center_y_m,
            args.object_center_z_m,
        ) = optimized_center.tolist()
        print(
            "[INITIAL-GRASP] loaded collision-aware pose "
            f"source={args.initial_grasp.resolve()} "
            f"center_m={optimized_center.round(5).tolist()}",
            flush=True,
        )
    else:
        joint7_init = env_cfg.scene.entities[
            "robot"
        ].init_state.joint_pos["^fr3v2_joint7$"]
        env_cfg.scene.entities["robot"].init_state.joint_pos[
            "^fr3v2_joint7$"
        ] = (
            float(joint7_init) + args.tool_roll_rad
        )
    retreat_azimuth = np.deg2rad(args.object_retreat_azimuth_deg)
    retreat_direction = args.object_retreat_direction_sign * np.asarray(
        (
            np.cos(retreat_azimuth),
            np.sin(retreat_azimuth),
            0.0,
        ),
        dtype=np.float64,
    )
    retreat_center = (
        np.asarray(
            (
                args.object_center_x_m,
                args.object_center_y_m,
                args.object_center_z_m,
            ),
            dtype=np.float64,
        )
        + args.object_retreat_distance_m * retreat_direction
    )
    env_cfg.scene.entities["target"].init_state.pos = tuple(
        retreat_center.tolist()
    )
    env_cfg.viewer.distance = args.camera_distance_m
    env_cfg.viewer.elevation = args.camera_elevation_deg
    env_cfg.viewer.azimuth = args.camera_azimuth_deg
    if args.viewer == "video":
        env_cfg.viewer.width = args.width
        env_cfg.viewer.height = args.height
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if args.viewer == "video" else None,
    )
    wrapped = RslRlVecEnvWrapper(env)
    cfg = FullHandMCCControlCfg(
        device=device,
        finger_desired_force=args.finger_force_n,
        finger_virtual_mass=args.finger_admittance_mass_kg,
        finger_virtual_damping=args.finger_admittance_damping_n_s_m,
        finger_virtual_stiffness=args.finger_admittance_stiffness_n_m,
        finger_force_gain=args.finger_force_gain,
        finger_max_calibrated_force=(
            args.finger_max_calibrated_force_n
        ),
        finger_force_filter_alpha=args.finger_force_filter_alpha,
        finger_contact_on_force=args.finger_contact_on_force_n,
        finger_contact_off_force=args.finger_contact_off_force_n,
        finger_max_normal_offset_m=(
            args.finger_max_normal_offset_mm / 1000.0
        ),
        max_tip_speed=args.finger_max_normal_speed_mm_s / 1000.0,
        finger_max_normal_acceleration=(
            args.finger_max_normal_acceleration_m_s2
        ),
        palm_desired_force=args.palm_force_n,
        arm_mcc_correction_limit=args.arm_mcc_correction_rad,
        wrist_update_decimation=args.wrist_update_decimation,
        wrist_damping_ratio=args.wrist_damping_ratio,
        wrist_max_force_error_n=args.wrist_max_force_error_n,
        wrist_max_torque_error_nm=args.wrist_max_torque_error_nm,
        wrist_max_translation_offset_m=(
            args.wrist_max_translation_offset_mm / 1000.0
        ),
        wrist_max_rotation_offset_rad=(
            args.wrist_max_rotation_offset_rad
        ),
    )
    kwargs = asdict(cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    controller = policy_class(device=device, num_envs=1, **kwargs)
    controller.fingers.nominal_tracking_radius = (
        args.precontact_tracking_radius_rad
    )
    reachability = FivePointReachabilitySolver(
        tolerance=args.ik_tolerance_mm / 1000.0,
        max_iterations=args.ik_max_iterations,
        palm_weight=args.palm_ik_weight,
    )
    protected_self_pairs = tuple(
        (
            mujoco.mj_name2id(
                reachability.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                first,
            ),
            mujoco.mj_name2id(
                reachability.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                second,
            ),
        )
        for first, second in PROTECTED_SELF_PAIR_NAMES
    )
    if any(first < 0 or second < 0 for first, second in protected_self_pairs):
        raise ValueError("One or more protected MCP-to-DIP geoms are missing")
    protected_self_pair_labels = tuple(
        f"{first}::{second}" for first, second in PROTECTED_SELF_PAIR_NAMES
    )
    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)

    class SurfaceSlidePolicy:
        def __init__(self) -> None:
            self.step = 0
            self.targets: np.ndarray | None = None
            self.kinematic_targets: np.ndarray | None = None
            self.normals: np.ndarray | None = None
            self.reachable_q: np.ndarray | None = None
            self.last_residual = np.full(5, np.inf)
            self.rejected_steps = 0
            self.actual_points = np.zeros((5, 3))
            self.tracking_error = np.full(5, np.inf)
            self.surface_error = np.full(5, np.inf)
            self.joint_error = np.full(TOTAL_DOF, np.inf)
            self.tactile_force = np.zeros(4)
            self.max_tactile_force = np.zeros(4)
            self.max_filtered_normal_force = np.zeros(4)
            self.contact_distance_m = np.full(4, np.nan)
            self.actual_contact_points = np.full((4, 3), np.nan)
            self.contact_start_arc = np.full(4, np.nan)
            self.contact_current_arc = np.full(4, np.nan)
            self.contact_surface_travel_m = np.zeros(4)
            self.contact_start_in_palm = np.full((4, 3), np.nan)
            self.contact_current_in_palm = np.full((4, 3), np.nan)
            self.contact_relative_travel_m = np.zeros(4)
            self.finger_q_min = np.full((4, 4), np.inf)
            self.finger_q_max = np.full((4, 4), -np.inf)
            self.max_penetration_m = np.zeros(4)
            self.max_runtime_self_penetration_m = 0.0
            self.runtime_self_near_contact_frames = 0
            self.arm_collision_frames = 0
            self.incidental_hand_contact_frames = 0
            self.incidental_hand_contact_streak = 0
            self.max_incidental_hand_contact_streak = 0
            self.max_incidental_hand_contact_force_n = 0.0
            self.max_incidental_hand_total_force_n = 0.0
            self.max_incidental_hand_penetration_m = 0.0
            self.incidental_hand_contact_evaluated_frames = 0
            self.min_tip_contacts_during_incidental_contact = 4
            self.contact_frames = np.zeros(4, dtype=np.int64)
            self.evaluated_frames = 0
            self.bad_contact_streak = 0
            self.max_bad_contact_streak = 0
            self.zero_contact_streak = 0
            self.max_zero_contact_streak = 0
            self.majority_contact_frames = 0
            self.simultaneous_contact_sum = 0
            self.contact_loss_streak = np.zeros(4, dtype=np.int64)
            self.max_contact_loss_streak = np.zeros(4, dtype=np.int64)
            self.min_simultaneous_contacts = 4
            self.final_all_contact_streak = 0
            self.contact_settle_streak = 0
            self.contact_calibrated = False
            self.fingertip_force_recalibrated = False
            self.max_fingertip_admittance_joint_correction_rad = 0.0
            self.max_arm_force_correction_rad = 0.0
            self.precontact_closure = np.zeros(16, dtype=np.float32)
            self.last_command_q: np.ndarray | None = None
            self.contact_servo_offset_q = np.zeros(TOTAL_DOF, dtype=np.float32)
            self.plan_surface: np.ndarray | None = None
            self.plan_kinematic: np.ndarray | None = None
            self.plan_normals: np.ndarray | None = None
            self.plan_q: np.ndarray | None = None
            self.plan_residual: np.ndarray | None = None
            self.plan_distance: np.ndarray | None = None
            self.plan_index = -1
            self.plan_direction = 1.0
            self.planned_axial_travel = 0.0
            self.executed_axial_travel = 0.0
            self.object_final_center: np.ndarray | None = None
            self.object_retreat_center: np.ndarray | None = None
            self.min_planned_arm_clearance_m = np.inf
            self.nearest_planned_arm_geom = ""
            self.min_planned_hand_clearance_m = np.inf
            self.nearest_planned_hand_geom = ""
            self.min_planned_tip_clearance_m = np.full(4, np.inf)
            self.min_planned_tip_clearance_frame = np.full(4, -1)
            self.min_planned_protected_self_clearance_m = np.inf
            self.min_planned_protected_self_clearance_frame = -1
            self.min_planned_protected_self_pair_name = ""
            self.min_planned_pad_alignment = 1.0
            self.min_runtime_pad_alignment = 1.0
            self.planned_curvature_min_inv_m = 0.0
            self.planned_curvature_max_inv_m = 0.0
            self.planned_curvature_ratio = 1.0

        def _audit_planned_surface_curvature(
            self,
            center: np.ndarray,
            rotation: np.ndarray,
        ) -> None:
            """Measure curvature actually traversed by the four planned tips."""

            assert self.plan_surface is not None
            finger_surface = self.plan_surface[:, 1:, :].reshape(-1, 3)
            arc, _ = capsule_meridian_coordinates(
                finger_surface,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            curvature = surface_meridian_curvature(
                arc,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            minimum = float(curvature.min())
            maximum = float(curvature.max())
            if minimum <= 1.0e-9 < maximum:
                ratio = np.inf
            elif maximum <= 1.0e-9:
                ratio = 1.0
            else:
                ratio = maximum / minimum
            self.planned_curvature_min_inv_m = minimum
            self.planned_curvature_max_inv_m = maximum
            self.planned_curvature_ratio = ratio
            print(
                "[CURVATURE-AUDIT] "
                f"shape={args.object_shape} "
                f"min_inv_m={minimum:.3f} max_inv_m={maximum:.3f} "
                f"ratio={'inf' if np.isinf(ratio) else f'{ratio:.3f}'}",
                flush=True,
            )
            if ratio + 1.0e-9 < args.min_meridian_curvature_ratio:
                raise RuntimeError(
                    "Planned route does not contain enough meridian "
                    "curvature variation: "
                    f"ratio={ratio:.3f} "
                    f"required={args.min_meridian_curvature_ratio:.3f}"
                )

        def _validate_full_robot_plan_clearance(
            self,
            center: np.ndarray,
            rotation: np.ndarray,
            joint_plan: np.ndarray,
            *,
            label: str,
        ) -> None:
            if args.collision_mode != "full_robot":
                return
            minimum_arm = np.inf
            nearest_arm = ""
            minimum_arm_frame = -1
            minimum_hand = np.inf
            nearest_hand = ""
            minimum_hand_frame = -1
            minimum_tip = np.full(4, np.inf, dtype=np.float64)
            minimum_tip_frame = np.full(4, -1, dtype=np.int32)
            minimum_pad_alignment = 1.0
            minimum_pad_frame = -1
            minimum_pad_finger = -1
            minimum_protected_self_clearance = np.inf
            minimum_protected_self_frame = -1
            minimum_protected_self_pair_name = ""
            for frame, q in enumerate(joint_plan):
                (
                    tip_clearances,
                    arm_clearances,
                    arm_geom_names,
                    hand_clearances,
                    hand_geom_names,
                ) = reachability.geometry_group_clearances(
                    q,
                    center,
                    rotation,
                )
                arm_index = int(np.argmin(arm_clearances))
                arm_clearance = float(arm_clearances[arm_index])
                arm_geom_name = arm_geom_names[arm_index]
                if arm_clearance < minimum_arm:
                    minimum_arm = arm_clearance
                    nearest_arm = arm_geom_name
                    minimum_arm_frame = frame
                hand_index = int(np.argmin(hand_clearances))
                hand_clearance = float(hand_clearances[hand_index])
                hand_geom_name = hand_geom_names[hand_index]
                if hand_clearance < minimum_hand:
                    minimum_hand = hand_clearance
                    nearest_hand = hand_geom_name
                    minimum_hand_frame = frame
                tip_update = tip_clearances < minimum_tip
                minimum_tip[tip_update] = tip_clearances[tip_update]
                minimum_tip_frame[tip_update] = frame
                points = reachability.forward_points(q)
                _, surface_normals = capsule_project(
                    points,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                pad_normals = reachability.fingertip_pad_normals(q)
                pad_alignment = np.einsum(
                    "ij,ij->i",
                    pad_normals,
                    -surface_normals[1:],
                )
                finger = int(np.argmin(pad_alignment))
                if pad_alignment[finger] < minimum_pad_alignment:
                    minimum_pad_alignment = float(pad_alignment[finger])
                    minimum_pad_frame = frame
                    minimum_pad_finger = finger
                self_pairs, self_distances = (
                    reachability.self_collision_contacts(q)
                )
                protected_distances = reachability.geometry_pair_distances(
                    q,
                    protected_self_pairs,
                )
                protected_index = int(np.argmin(protected_distances))
                protected_clearance = float(
                    protected_distances[protected_index]
                )
                if protected_clearance < minimum_protected_self_clearance:
                    minimum_protected_self_clearance = protected_clearance
                    minimum_protected_self_frame = frame
                    minimum_protected_self_pair_name = (
                        protected_self_pair_labels[protected_index]
                    )
                if self_pairs:
                    unique_self_pairs = set(self_pairs)
                    raise RuntimeError(
                        "Full-robot trajectory contains active robot self "
                        "contacts: "
                        f"label={label} frame={frame}/{len(joint_plan)} "
                        f"unique_pairs={len(unique_self_pairs)} "
                        f"sample_occurrences={len(self_pairs)} "
                        f"deepest_mm={self_distances.min() * 1000:.3f} "
                        "minimum_protected_clearance_mm="
                        f"{minimum_protected_self_clearance * 1000:.3f} "
                        f"nearest_protected_pair="
                        f"{minimum_protected_self_pair_name}"
                    )
            required_arm = args.min_arm_clearance_mm / 1000.0
            if minimum_arm < required_arm:
                raise RuntimeError(
                    "Full-robot trajectory violates strict FR3/object "
                    f"clearance: label={label} frame={minimum_arm_frame}/"
                    f"{len(joint_plan)} clearance_mm="
                    f"{minimum_arm * 1000:.3f} "
                    f"required_mm={args.min_arm_clearance_mm:.3f} "
                    f"nearest={nearest_arm}"
                )
            allowed_hand_penetration = (
                args.max_incidental_hand_penetration_mm / 1000.0
            )
            if minimum_hand < -allowed_hand_penetration:
                raise RuntimeError(
                    "Full-robot trajectory exceeds allowed incidental "
                    f"LEAP Hand/object penetration: label={label} "
                    f"frame={minimum_hand_frame}/{len(joint_plan)} "
                    f"penetration_mm={-minimum_hand * 1000:.3f} "
                    "allowed_mm="
                    f"{args.max_incidental_hand_penetration_mm:.3f} "
                    f"nearest={nearest_hand}"
                )
            allowed_tip_penetration = (
                args.max_contact_penetration_mm / 1000.0
            )
            if np.any(minimum_tip < -allowed_tip_penetration):
                finger = int(np.argmin(minimum_tip))
                tip_name = (
                    reachability.model.geom(
                        int(reachability.tip_geom_ids[finger])
                    ).name
                    or ""
                )
                raise RuntimeError(
                    "Full-robot trajectory exceeds allowed physical "
                    "fingertip/object penetration: "
                    f"label={label} frame={minimum_tip_frame[finger]}/"
                    f"{len(joint_plan)} finger={finger} "
                    f"penetration_mm={-minimum_tip[finger] * 1000:.3f} "
                    f"allowed_mm={args.max_contact_penetration_mm:.3f} "
                    f"nearest={tip_name} per_tip_minimum_mm="
                    f"{(minimum_tip * 1000).round(3).tolist()}"
                )
            required_pad_alignment = float(
                np.cos(
                    np.deg2rad(
                        args.max_pad_angle_deg
                        - args.planner_pad_angle_margin_deg
                    )
                )
            )
            if minimum_pad_alignment < required_pad_alignment:
                raise RuntimeError(
                    "Full-robot trajectory turns a fingertip onto its "
                    "outer/nail side: "
                    f"label={label} frame={minimum_pad_frame}/"
                    f"{len(joint_plan)} finger={minimum_pad_finger} "
                    f"pad_angle_deg="
                    f"{np.degrees(np.arccos(np.clip(minimum_pad_alignment, -1, 1))):.2f} "
                    "planner_limit_deg="
                    f"{args.max_pad_angle_deg - args.planner_pad_angle_margin_deg:.2f}"
                )
            self.min_planned_arm_clearance_m = float(minimum_arm)
            self.nearest_planned_arm_geom = nearest_arm
            self.min_planned_hand_clearance_m = float(minimum_hand)
            self.nearest_planned_hand_geom = nearest_hand
            self.min_planned_tip_clearance_m = minimum_tip.copy()
            self.min_planned_tip_clearance_frame = minimum_tip_frame.copy()
            self.min_planned_protected_self_clearance_m = float(
                minimum_protected_self_clearance
            )
            self.min_planned_protected_self_clearance_frame = (
                minimum_protected_self_frame
            )
            self.min_planned_protected_self_pair_name = (
                minimum_protected_self_pair_name
            )
            self.min_planned_pad_alignment = minimum_pad_alignment
            print(
                "[FULL-ROBOT-PLAN-CLEARANCE] "
                f"label={label} frames={len(joint_plan)} "
                f"minimum_arm_mm={minimum_arm * 1000:.3f} "
                f"required_arm_mm={args.min_arm_clearance_mm:.3f} "
                f"arm_frame={minimum_arm_frame} "
                f"nearest_arm={nearest_arm} "
                f"minimum_hand_mm={minimum_hand * 1000:.3f} "
                "allowed_hand_penetration_mm="
                f"{args.max_incidental_hand_penetration_mm:.3f} "
                f"hand_frame={minimum_hand_frame} "
                f"nearest_hand={nearest_hand} "
                f"minimum_tip_mm="
                f"{(minimum_tip * 1000).round(3).tolist()} "
                f"allowed_tip_penetration_mm="
                f"{args.max_contact_penetration_mm:.3f} "
                f"tip_frames={minimum_tip_frame.tolist()} "
                "minimum_protected_self_clearance_mm="
                f"{minimum_protected_self_clearance * 1000:.3f} "
                f"protected_self_frame={minimum_protected_self_frame} "
                f"nearest_protected_self_pair="
                f"{minimum_protected_self_pair_name} "
                f"max_pad_angle_deg="
                f"{np.degrees(np.arccos(np.clip(minimum_pad_alignment, -1, 1))):.2f} "
                f"planner_limit_deg="
                f"{args.max_pad_angle_deg - args.planner_pad_angle_margin_deg:.2f} "
                f"pad_frame={minimum_pad_frame} "
                f"pad_finger={minimum_pad_finger}",
                flush=True,
            )

        def _object_pose(self, obs) -> tuple[np.ndarray, np.ndarray]:
            _ = obs
            center = (
                env.sim.data.mocap_pos[0, target_mocap_idx]
                .detach()
                .cpu()
                .numpy()
            )
            quat_wxyz = (
                env.sim.data.mocap_quat[0, target_mocap_idx]
                .detach()
                .cpu()
                .numpy()
            )
            quat_xyzw = np.roll(quat_wxyz, -1)
            return center, R.from_quat(quat_xyzw).as_matrix()

        def _initialize(self, obs) -> None:
            q = obs["palm"][0, :TOTAL_DOF].detach().cpu().numpy()
            retreat_center, rotation = self._object_pose(obs)
            center = np.asarray(
                (
                    args.object_center_x_m,
                    args.object_center_y_m,
                    args.object_center_z_m,
                ),
                dtype=np.float64,
            )
            if args.collision_mode == "full_robot":
                initial_arm_clearance, initial_arm_nearest = (
                    reachability.minimum_arm_clearance(
                        q, retreat_center, rotation
                    )
                )
                final_arm_clearance, final_arm_nearest = (
                    reachability.minimum_arm_clearance(q, center, rotation)
                )
                initial_hand_clearance, initial_hand_nearest = (
                    reachability.minimum_hand_clearance(
                        q, retreat_center, rotation
                    )
                )
                final_hand_clearance, final_hand_nearest = (
                    reachability.minimum_hand_clearance(q, center, rotation)
                )
                print(
                    "[FULL-ROBOT-INITIAL-STATE] "
                    f"live_q={q.round(5).tolist()} "
                    f"retreat_center={retreat_center.round(5).tolist()} "
                    f"final_center={center.round(5).tolist()} "
                    f"cpu_retreat_arm_clearance_mm="
                    f"{initial_arm_clearance * 1000:.3f} "
                    f"cpu_retreat_arm_nearest={initial_arm_nearest} "
                    f"cpu_final_arm_clearance_mm="
                    f"{final_arm_clearance * 1000:.3f} "
                    f"cpu_final_arm_nearest={final_arm_nearest} "
                    f"cpu_retreat_hand_clearance_mm="
                    f"{initial_hand_clearance * 1000:.3f} "
                    f"cpu_retreat_hand_nearest={initial_hand_nearest} "
                    f"cpu_final_hand_clearance_mm="
                    f"{final_hand_clearance * 1000:.3f} "
                    f"cpu_final_hand_nearest={final_hand_nearest}",
                    flush=True,
                )
            self.object_final_center = center.copy()
            live_points = reachability.forward_points(q)
            self.object_retreat_center = retreat_center.copy()
            surface_targets, normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            approach_targets = surface_targets.copy()
            approach_targets[0] = live_points[0]
            approach = reachability.solve(approach_targets, q)
            if not approach.accepted:
                raise RuntimeError(
                    "Tactile pre-contact IK is unreachable: "
                    f"residual_mm="
                    f"{(approach.residual_m * 1000).round(2).tolist()}"
                )
            self.targets = surface_targets
            # Hold the live arm pose while the object approaches.  The
            # projected five-point IK above is a feasibility check only; using
            # it as a first-frame command can jump the high-stiffness FR3
            # servo and destabilize a wrist-rolled configuration.
            self.kinematic_targets = live_points.copy()
            self.normals = normals
            self.reachable_q = q.copy()
            self.last_residual = approach.residual_m.copy()

        def _update_object_approach(self) -> None:
            if (
                self.object_final_center is None
                or self.object_retreat_center is None
            ):
                return
            phase = min(
                max(self.step - args.preshape_frames, 0)
                / float(max(args.object_approach_frames, 1)),
                1.0,
            )
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            center = (
                (1.0 - smooth_phase) * self.object_retreat_center
                + smooth_phase * self.object_final_center
            )
            env.sim.data.mocap_pos[0, target_mocap_idx] = torch.as_tensor(
                center,
                device=env.sim.data.mocap_pos.device,
                dtype=env.sim.data.mocap_pos.dtype,
            )

        def _capture_contact_calibration(
            self,
            obs,
            live_q: np.ndarray,
            live_points: np.ndarray,
        ) -> None:
            center, rotation = self._object_pose(obs)
            center = (
                center
                + args.planning_center_shift_m * rotation[:, 2]
            )
            env.sim.data.mocap_pos[0, target_mocap_idx] = torch.as_tensor(
                center,
                device=env.sim.data.mocap_pos.device,
                dtype=env.sim.data.mocap_pos.dtype,
            )
            self.object_final_center = center.copy()
            self.object_retreat_center = center.copy()
            palm_position, palm_rotation = reachability.forward_palm_pose(
                live_q
            )
            object_center_in_palm = palm_rotation.T @ (
                center - palm_position
            )
            _, calibration_surface_normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            calibration_pad_normals = (
                reachability.fingertip_pad_normals(live_q)
            )
            calibration_pad_alignment = np.einsum(
                "ij,ij->i",
                calibration_pad_normals,
                -calibration_surface_normals[1:],
            )
            required_pad_alignment = float(
                np.cos(np.deg2rad(args.max_pad_angle_deg))
            )
            if float(calibration_pad_alignment.min()) < required_pad_alignment:
                raise RuntimeError(
                    "Physical finger-pad orientation validation failed at "
                    "contact calibration: "
                    f"alignment={calibration_pad_alignment.round(4).tolist()} "
                    f"angle_deg="
                    f"{np.degrees(np.arccos(np.clip(calibration_pad_alignment, -1, 1))).round(2).tolist()} "
                    f"limit_deg={args.max_pad_angle_deg:.2f} "
                    f"object_center_in_palm_m="
                    f"{object_center_in_palm.round(4).tolist()}"
                )
            surface_targets, normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            # Preserve the servo deflection that produced the measured
            # contact forces as feed-forward, while planning from the true
            # loaded configuration.  Using the unloaded command itself as the
            # physical MPC state corrupts surface progress by several mm.
            contact_command_q = (
                self.last_command_q.copy()
                if self.last_command_q is not None
                else live_q.copy()
            )
            planning_anchor_q = (
                self.reachable_q.copy()
                if self.reachable_q is not None
                else contact_command_q.copy()
            )
            planning_q = live_q.copy()
            if args.planner_state_quantization_rad > 0.0:
                quantum = args.planner_state_quantization_rad
                planning_q = planning_anchor_q + quantum * np.round(
                    (live_q - planning_anchor_q) / quantum
                )
                planning_q = np.minimum(
                    np.maximum(planning_q, reachability.lower),
                    reachability.upper,
                )
            planning_points = reachability.forward_points(planning_q)
            planning_surface_targets, planning_normals = capsule_project(
                planning_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            calibrated_offset = contact_command_q - live_q
            self.contact_servo_offset_q[:ARM_DOF] = (
                args.arm_servo_load_scale * calibrated_offset[:ARM_DOF]
            )
            self.contact_servo_offset_q[ARM_DOF:TOTAL_DOF] = (
                args.finger_servo_load_scale
                * calibrated_offset[ARM_DOF:TOTAL_DOF]
            )
            self.targets = planning_surface_targets
            self.kinematic_targets = planning_points
            self.normals = planning_normals
            self.reachable_q = planning_q
            self.last_residual = np.zeros(5)
            self.surface_error = np.linalg.norm(
                live_points - surface_targets, axis=1
            )
            # ContactSensor.pos may switch between equally valid contact
            # points on a rounded pad (or between manifold slots), producing
            # a discontinuous apparent displacement.  The physical
            # fingertip sites are stable body-fixed references; project those
            # sites to the object meridian for the travel audit while the
            # tactile/contact sensors independently prove continuous contact.
            contact_surface, _ = capsule_project(
                live_points[1:],
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            contact_arc, _ = capsule_meridian_coordinates(
                contact_surface,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            contact_in_palm = (
                palm_rotation.T
                @ (live_points[1:] - palm_position).T
            ).T
            self.contact_start_arc = contact_arc.copy()
            self.contact_current_arc = contact_arc.copy()
            self.contact_start_in_palm = contact_in_palm.copy()
            self.contact_current_in_palm = contact_in_palm.copy()
            finger_q = live_q[ARM_DOF:TOTAL_DOF].reshape(4, 4)
            self.finger_q_min = finger_q.copy()
            self.finger_q_max = finger_q.copy()
            print(
                "[PLANNER-STATE] "
                f"quantization_rad="
                f"{args.planner_state_quantization_rad:.6f} "
                f"max_adjustment_rad="
                f"{float(np.max(np.abs(planning_q - live_q))):.6f} "
                f"planning_q={planning_q.round(6).tolist()}",
                flush=True,
            )
            self._build_axial_plan(center, rotation)
            self._audit_planned_surface_curvature(center, rotation)
            self.contact_calibrated = True
            direct_normal_baseline = controller.last_debug[
                "tip_normal_force_signed_raw"
            ][0].detach().cpu().numpy()
            controller.reset()
            controller.calibrate_arm_force_setpoint(obs["palm"])
            controller.fingers.calibrate_fingertip_force_sign(
                direct_normal_baseline
            )
            controller.fingers.nominal_tracking_radius = (
                args.contact_tracking_radius_rad
            )
            print(
                "[CONTACT-CALIBRATION] captured collision-consistent "
                f"planner_quantization_rad="
                f"{args.planner_state_quantization_rad:.6f} "
                f"planner_q_adjustment_rad="
                f"{float(np.max(np.abs(planning_q - live_q))):.6f} "
                f"site_standoff_mm="
                f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                f"tactile_force_N={self.tactile_force.round(2).tolist()} "
                f"direct_normal_force_raw_N="
                f"{direct_normal_baseline.round(2).tolist()} "
                f"force_target_N="
                f"{controller.fingers.fingertip_force_setpoint.round(2).tolist()} "
                f"servo_deflection_rad="
                f"{self.contact_servo_offset_q.round(3).tolist()}",
                flush=True,
            )

        def _build_axial_plan(
            self,
            center: np.ndarray,
            rotation: np.ndarray,
        ) -> None:
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            if args.planner == "circumferential_surface_mpc":
                self._build_circumferential_surface_mpc_plan(
                    center=center,
                    rotation=rotation,
                    frame_count=args.steps - args.motion_start,
                )
                return
            if args.reuse_plan is not None:
                cached = np.load(args.reuse_plan)
                cached_shape = (
                    str(cached["object_shape"])
                    if "object_shape" in cached.files
                    else "capsule"
                )
                if cached_shape != args.object_shape:
                    raise RuntimeError(
                        "Cached plan object-shape mismatch: "
                        f"cached={cached_shape} requested={args.object_shape}"
                    )
                joint_plan = np.asarray(
                    cached["joint_positions_rad"], dtype=np.float32
                )
                expected_frames = args.steps - args.motion_start
                if joint_plan.shape != (expected_frames, TOTAL_DOF):
                    raise RuntimeError(
                        "Cached plan frame shape mismatch: "
                        f"got={joint_plan.shape} "
                        f"expected={(expected_frames, TOTAL_DOF)}"
                    )
                self.plan_surface = np.asarray(
                    cached["surface_points_m"], dtype=np.float32
                )
                self.plan_kinematic = np.asarray(
                    cached["kinematic_points_m"], dtype=np.float32
                )
                _, cached_normals = capsule_project(
                    self.plan_kinematic.reshape(-1, 3),
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                self.plan_normals = cached_normals.reshape(
                    expected_frames, 5, 3
                )
                self.plan_q = joint_plan
                residual_key = (
                    "progress_residual_m"
                    if "progress_residual_m" in cached.files
                    else "residual_m"
                )
                self.plan_residual = np.asarray(
                    cached[residual_key], dtype=np.float32
                )
                self.plan_distance = np.asarray(
                    cached["axial_distance_m"], dtype=np.float32
                )
                self.plan_direction = float(
                    cached["axial_direction"]
                    if "axial_direction" in cached.files
                    else args.axial_direction
                )
                self.planned_axial_travel = float(self.plan_distance[-1])
                start_delta = float(
                    np.max(np.abs(joint_plan[0] - self.reachable_q))
                )
                if start_delta > args.max_plan_joint_step_rad:
                    raise RuntimeError(
                        "Cached plan does not match calibrated start pose: "
                        f"max_joint_delta={start_delta:.5f}rad"
                    )
                self._validate_full_robot_plan_clearance(
                    center,
                    rotation,
                    joint_plan,
                    label="cached",
                )
                print(
                    "[AXIAL-PLAN] reused validated plan | "
                    f"frames={expected_frames} "
                    f"travel_m={self.planned_axial_travel:.4f} "
                    f"start_joint_delta_rad={start_delta:.5f} "
                    f"source={args.reuse_plan.resolve()}",
                    flush=True,
                )
                return
            start_arc, azimuth = capsule_meridian_coordinates(
                self.targets,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            start_surface, _, start_frames = capsule_meridian_targets(
                start_arc,
                azimuth,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            standoff_components = np.einsum(
                "nji,nj->ni",
                start_frames,
                self.kinematic_targets - start_surface,
            )
            total_arc = SURFACE_TOTAL_LENGTH
            # The palm root is a non-contact arm/MCC reference.  Choose and
            # bound the surface direction from the four physical contacts.
            direction = float(args.axial_direction)
            available = (
                total_arc - float(start_arc[1:].max())
                if direction > 0.0
                else float(start_arc[1:].min())
            )
            if args.axial_travel_m > available - 1.0e-6:
                raise RuntimeError(
                    "Requested end-to-end travel exceeds the object surface: "
                    f"requested={args.axial_travel_m:.4f}m "
                    f"available={available:.4f}m"
                )
            frame_count = args.steps - args.motion_start
            surface_plan = np.zeros((frame_count, 5, 3), dtype=np.float32)
            kinematic_plan = np.zeros_like(surface_plan)
            normal_plan = np.zeros_like(surface_plan)
            joint_plan = np.zeros((frame_count, TOTAL_DOF), dtype=np.float32)
            residual_plan = np.zeros((frame_count, 5), dtype=np.float32)
            distance_plan = np.zeros(frame_count, dtype=np.float32)
            palm_pose_error_plan = np.zeros((frame_count, 2), dtype=np.float32)
            seed = self.reachable_q.copy()
            start_kinematic = self.kinematic_targets.copy()
            axis_world = rotation[:, 2]
            preload = args.surface_preload_mm / 1000.0
            start_palm_position, start_palm_rotation = (
                reachability.forward_palm_pose(self.reachable_q)
            )

            if args.planner == "adaptive_surface_mpc":
                self._build_adaptive_surface_mpc_plan(
                    center=center,
                    rotation=rotation,
                    start_arc=start_arc,
                    start_azimuth=azimuth,
                    start_surface=start_surface,
                    start_frames=start_frames,
                    direction=direction,
                    available=available,
                    frame_count=frame_count,
                )
                return

            for frame in range(frame_count):
                phase = (frame + 1) / float(frame_count)
                if args.planner == "staged_inward":
                    segment_phase = min(
                        phase * args.waypoint_count,
                        float(args.waypoint_count),
                    )
                    segment = min(
                        int(segment_phase),
                        args.waypoint_count - 1,
                    )
                    local_phase = segment_phase - segment
                    move_phase = min(
                        local_phase / args.stage_move_fraction,
                        1.0,
                    )
                    smooth_local = move_phase * move_phase * (
                        3.0 - 2.0 * move_phase
                    )
                    smooth_phase = (
                        segment + smooth_local
                    ) / args.waypoint_count
                else:
                    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
                distance = args.axial_travel_m * smooth_phase
                if args.planner == "meridian_inward":
                    arc = start_arc + direction * distance
                    surface, normals, frames = capsule_meridian_targets(
                        arc,
                        azimuth,
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                    kinematic = (
                        surface
                        + np.einsum(
                            "nij,nj->ni",
                            frames,
                            standoff_components,
                        )
                        - preload * normals
                    )
                else:
                    translation = direction * distance * axis_world
                    surface, normals = capsule_project(
                        start_surface + translation,
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                    # Preserve the calibrated hand shape and translate it as a
                    # rigid five-site group.  Projected surface points remain
                    # exact visualization/input targets; the kinematic sites
                    # are intentionally biased inward to maintain contact.
                    kinematic = (
                        start_kinematic
                        + translation
                        - preload * normals
                    )
                palm_translation = direction * distance * axis_world
                palm_pose = reachability.solve_palm_pose(
                    start_palm_position + palm_translation,
                    start_palm_rotation,
                    seed,
                )
                # This arm-only pose solve is a seed, not an acceptance
                # constraint.  Even its best sub-millimetre result can unlock
                # the strict simultaneous five-point solve.
                result = reachability.solve(
                    kinematic,
                    palm_pose.joint_position,
                )
                if not result.accepted:
                    result = reachability.solve(kinematic, seed)
                if not result.accepted:
                    raise RuntimeError(
                        "End-to-end plan is not simultaneously reachable: "
                        f"frame={frame}/{frame_count} "
                        f"distance_m={distance:.4f} "
                        f"residual_mm="
                        f"{(result.residual_m * 1000).round(2).tolist()}"
                    )
                q = result.joint_position
                if np.any(q < reachability.lower - 1.0e-8) or np.any(
                    q > reachability.upper + 1.0e-8
                ):
                    raise RuntimeError(
                        f"Planner produced a joint-limit violation at frame {frame}"
                    )
                surface_plan[frame] = surface
                kinematic_plan[frame] = kinematic
                normal_plan[frame] = normals
                joint_plan[frame] = q
                residual_plan[frame] = result.residual_m
                distance_plan[frame] = distance
                palm_pose_error_plan[frame] = (
                    palm_pose.position_error_m,
                    palm_pose.orientation_error_rad,
                )
                seed = q

            self.plan_surface = surface_plan
            self.plan_kinematic = kinematic_plan
            self.plan_normals = normal_plan
            self.plan_q = joint_plan
            self.plan_residual = residual_plan
            self.plan_distance = distance_plan
            self.plan_direction = direction
            self.planned_axial_travel = float(distance_plan[-1])
            q_with_seed = np.vstack((self.reachable_q[None], joint_plan))
            max_joint_step = float(np.max(np.abs(np.diff(q_with_seed, axis=0))))
            if max_joint_step > args.max_plan_joint_step_rad:
                raise RuntimeError(
                    "Planned joint step exceeds the configured bound: "
                    f"observed={max_joint_step:.5f}rad "
                    f"limit={args.max_plan_joint_step_rad:.5f}rad"
                )
            self._validate_full_robot_plan_clearance(
                center,
                rotation,
                joint_plan,
                label=args.planner,
            )
            start_local = (rotation.T @ (surface_plan[0] - center).T).T
            end_local = (rotation.T @ (surface_plan[-1] - center).T).T
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.plan_output,
                surface_points_m=surface_plan,
                kinematic_points_m=kinematic_plan,
                joint_positions_rad=joint_plan,
                residual_m=residual_plan,
                axial_distance_m=distance_plan,
                axial_direction=np.asarray(direction),
                planner=np.asarray(args.planner),
                seed=np.asarray(args.seed),
                planner_state_quantization_rad=np.asarray(
                    args.planner_state_quantization_rad
                ),
                surface_preload_mm=np.asarray(args.surface_preload_mm),
                palm_travel_ratio=np.asarray(args.palm_travel_ratio),
                finger_gait_amplitude_m=np.asarray(
                    args.finger_gait_amplitude_m
                ),
                finger_meridian_gait_mm=np.asarray(
                    args.finger_meridian_gait_mm
                ),
                finger_meridian_gait_start_m=np.asarray(
                    args.finger_meridian_gait_start_m
                ),
                finger_meridian_gait_end_m=np.asarray(
                    args.finger_meridian_gait_end_m
                ),
                finger_meridian_gait_scales=np.asarray(
                    args.finger_meridian_gait_scales
                ),
                object_shape=np.asarray(args.object_shape),
                object_radius_m=np.asarray(CAPSULE_RADIUS),
                object_half_height_m=np.asarray(CAPSULE_HALF_HEIGHT),
                max_joint_step_rad=np.asarray(max_joint_step),
                palm_pose_error=palm_pose_error_plan,
                start_surface_local_m=start_local,
                end_surface_local_m=end_local,
            )
            print(
                "[AXIAL-PLAN] all frames simultaneously reachable | "
                f"planner={args.planner} "
                f"frames={frame_count} travel_m={self.planned_axial_travel:.4f} "
                f"preload_mm={args.surface_preload_mm:.2f} "
                f"max_joint_step_rad={max_joint_step:.5f} "
                f"direction={direction:+.0f} "
                f"start_z_m={start_local[:, 2].round(4).tolist()} "
                f"end_z_m={end_local[:, 2].round(4).tolist()} "
                f"max_residual_mm="
                f"{float(residual_plan.max() * 1000):.2f} "
                f"saved={args.plan_output.resolve()}",
                flush=True,
            )

        def _build_circumferential_surface_mpc_plan(
            self,
            *,
            center: np.ndarray,
            rotation: np.ndarray,
            frame_count: int,
        ) -> None:
            """Move the rigid five-point constellation around a thick capsule.

            The four tactile pads remain on their initial cylindrical rings.
            The palm-root point is rotated with the hand as a kinematic
            reference, but is never treated as a physical contact.
            """

            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            start_surface = self.targets.copy()
            start_points = self.kinematic_targets.copy()
            start_local = (rotation.T @ (start_surface - center).T).T
            if np.any(
                np.abs(start_local[1:, 2])
                > CAPSULE_HALF_HEIGHT - 1.0e-4
            ):
                raise RuntimeError(
                    "Circumferential surface MPC requires all fingertip "
                    "contacts on the capsule cylinder, not an end cap: "
                    f"local_z_m={start_local[1:, 2].round(4).tolist()}"
                )

            keyframe_count = min(args.mpc_keyframes, frame_count)
            coarse_distance = np.linspace(
                0.0,
                args.axial_travel_m,
                keyframe_count + 1,
            )
            coarse_q = np.zeros(
                (keyframe_count + 1, TOTAL_DOF), dtype=np.float64
            )
            coarse_q[0] = self.reachable_q
            coarse_residual = np.zeros((keyframe_count + 1, 5), dtype=np.float64)
            coarse_nfev = np.zeros(keyframe_count + 1, dtype=np.int32)
            planner_pad_alignment = float(
                np.cos(
                    np.deg2rad(
                        args.max_pad_angle_deg
                        - args.planner_pad_angle_margin_deg
                    )
                )
            )
            previous_q = self.reachable_q.copy()
            preload = args.surface_preload_mm / 1000.0

            def rotated_targets(
                distance: float,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                theta = (
                    float(args.axial_direction)
                    * distance
                    / CAPSULE_RADIUS
                )
                world_rotation = (
                    rotation
                    @ R.from_rotvec((0.0, 0.0, theta)).as_matrix()
                    @ rotation.T
                )
                surface = (
                    center
                    + (world_rotation @ (start_surface - center).T).T
                )
                points = (
                    center
                    + (world_rotation @ (start_points - center).T).T
                )
                _, normals = capsule_project(
                    surface,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                preload_fraction = min(
                    distance / max(0.025, args.axial_travel_m),
                    1.0,
                )
                points[1:] -= (
                    preload_fraction * preload * normals[1:]
                )
                return surface, points, normals

            for keyframe in range(1, keyframe_count + 1):
                distance = float(coarse_distance[keyframe])
                _, desired_points, _ = rotated_targets(distance)
                result = reachability.solve(desired_points, previous_q)
                tip_error = float(result.residual_m[1:].max())
                palm_error = float(result.residual_m[0])
                if (
                    tip_error > args.ik_tolerance_mm / 1000.0
                    or palm_error > args.palm_path_tolerance_mm / 1000.0
                ):
                    raise RuntimeError(
                        "Circumferential surface MPC found an unreachable "
                        f"five-point state at keyframe={keyframe}/"
                        f"{keyframe_count}, distance_m={distance:.4f}, "
                        f"residual_mm="
                        f"{(result.residual_m * 1000).round(2).tolist()}"
                    )
                coarse_q[keyframe] = result.joint_position
                coarse_residual[keyframe, 1:] = result.residual_m[1:]
                coarse_nfev[keyframe] = result.iterations
                previous_q = result.joint_position
                print(
                    "[CIRCUMFERENTIAL-MPC] "
                    f"keyframe={keyframe:02d}/{keyframe_count} "
                    f"surface_travel_m={distance:.4f} "
                    f"five_point_error_mm="
                    f"{(result.residual_m * 1000).round(2).tolist()} "
                    f"iterations={result.iterations}",
                    flush=True,
                )

            sample_coordinate = np.linspace(
                0.0,
                float(keyframe_count),
                frame_count,
            )
            joint_plan = np.column_stack(
                [
                    np.interp(
                        sample_coordinate,
                        np.arange(keyframe_count + 1),
                        coarse_q[:, joint],
                    )
                    for joint in range(TOTAL_DOF)
                ]
            ).astype(np.float32)
            distance_plan = np.linspace(
                0.0,
                args.axial_travel_m,
                frame_count,
                dtype=np.float32,
            )
            surface_plan = np.zeros((frame_count, 5, 3), dtype=np.float32)
            kinematic_plan = np.zeros_like(surface_plan)
            normal_plan = np.zeros_like(surface_plan)
            residual_plan = np.zeros((frame_count, 5), dtype=np.float32)
            for frame, distance in enumerate(distance_plan):
                surface, desired_points, normals = rotated_targets(
                    float(distance)
                )
                achieved_points = reachability.forward_points(
                    joint_plan[frame]
                )
                # The palm root is a non-contact load-feedback coordinate.
                # Replace its nominal rigid orbit with the point actually
                # achieved by this joint-limit-valid arm state.  Fingertip
                # surface targets remain unchanged and strictly checked.
                desired_points[0] = achieved_points[0]
                surface[0] = achieved_points[0]
                surface_plan[frame] = surface
                kinematic_plan[frame] = desired_points
                normal_plan[frame] = normals
                residual_plan[frame] = np.linalg.norm(
                    achieved_points - desired_points,
                    axis=1,
                )

            max_joint_step = float(
                np.max(np.abs(np.diff(joint_plan, axis=0)))
            )
            if max_joint_step > args.max_plan_joint_step_rad:
                raise RuntimeError(
                    "Circumferential surface MPC joint step exceeds bound: "
                    f"{max_joint_step:.5f}rad > "
                    f"{args.max_plan_joint_step_rad:.5f}rad"
                )
            if (
                float(residual_plan[-1, 1:].max())
                > args.ik_tolerance_mm / 1000.0
            ):
                raise RuntimeError(
                    "Circumferential surface MPC interpolation failed final "
                    f"fingertip reachability: "
                    f"{(residual_plan[-1] * 1000).round(2).tolist()}mm"
                )

            self._validate_full_robot_plan_clearance(
                center,
                rotation,
                joint_plan,
                label="circumferential_surface_mpc",
            )

            self.plan_surface = surface_plan
            self.plan_kinematic = kinematic_plan
            self.plan_normals = normal_plan
            self.plan_q = joint_plan
            self.plan_residual = residual_plan
            self.plan_distance = distance_plan
            self.planned_axial_travel = float(distance_plan[-1])
            end_local = (
                rotation.T @ (surface_plan[-1] - center).T
            ).T
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.plan_output,
                surface_points_m=surface_plan,
                kinematic_points_m=kinematic_plan,
                joint_positions_rad=joint_plan,
                progress_m=np.repeat(
                    distance_plan[:, None], 5, axis=1
                ),
                progress_residual_m=residual_plan,
                normal_error_m=np.zeros_like(residual_plan),
                axial_distance_m=distance_plan,
                planner=np.asarray(args.planner),
                seed=np.asarray(args.seed),
                planner_state_quantization_rad=np.asarray(
                    args.planner_state_quantization_rad
                ),
                surface_preload_mm=np.asarray(args.surface_preload_mm),
                max_joint_step_rad=np.asarray(max_joint_step),
                coarse_joint_positions_rad=coarse_q,
                coarse_progress_m=np.repeat(
                    coarse_distance[:, None], 5, axis=1
                ),
                coarse_normal_error_m=np.zeros_like(coarse_residual),
                coarse_cost=np.zeros(keyframe_count + 1),
                coarse_nfev=coarse_nfev,
                start_surface_local_m=start_local,
                end_surface_local_m=end_local,
            )
            print(
                "[CIRCUMFERENTIAL-PLAN] passed | "
                f"frames={frame_count} keyframes={keyframe_count} "
                f"surface_travel_m={self.planned_axial_travel:.4f} "
                f"rotation_deg="
                f"{np.degrees(args.axial_travel_m / CAPSULE_RADIUS):.2f} "
                f"max_joint_step_rad={max_joint_step:.5f} "
                f"max_five_point_error_mm="
                f"{float(residual_plan.max() * 1000):.2f} "
                f"saved={args.plan_output.resolve()}",
                flush=True,
            )

        def _build_adaptive_surface_mpc_plan(
            self,
            *,
            center: np.ndarray,
            rotation: np.ndarray,
            start_arc: np.ndarray,
            start_azimuth: np.ndarray,
            start_surface: np.ndarray,
            start_frames: np.ndarray,
            direction: float,
            available: float,
            frame_count: int,
        ) -> None:
            """Optimize a reachable contact path directly in joint space.

            Unlike rigid Cartesian transport, every plan state is a real URDF
            configuration.  Per-contact meridian progress is a soft MPC
            objective, so fingers may lead or lag slightly when the coupled
            kinematics require it while all contacts still cross the object.
            """

            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            palm_tracking_limit_m = (
                args.palm_guide_max_drift_mm
                if args.palm_guide_only
                else args.mpc_palm_position_tolerance_mm
            ) / 1000.0
            if args.axial_travel_m > available - 1.0e-6:
                raise RuntimeError(
                    "Requested adaptive-MPC travel exceeds the capsule surface: "
                    f"requested={args.axial_travel_m:.4f}m "
                    f"available={available:.4f}m"
                )

            initial_site_offset = self.kinematic_targets - start_surface
            initial_signed_standoff = np.einsum(
                "ni,ni->n",
                initial_site_offset,
                start_frames[:, :, 0],
            )
            # Express the calibrated site-to-surface offset in each local
            # contact frame.  Transporting these components to the desired
            # surface state gives the optimizer a seed in which the fingers
            # have already advanced along the curved object.  This matters
            # when the palm is fixed: the old rigid seed was exactly the
            # previous state and repeatedly converged to the no-motion local
            # minimum.
            initial_frame_offset = np.einsum(
                "nji,nj->ni",
                start_frames,
                initial_site_offset,
            )
            target_signed_standoff = (
                initial_signed_standoff - args.surface_preload_mm / 1000.0
            )
            lower = np.where(
                np.isfinite(reachability.lower),
                reachability.lower + 1.0e-7,
                -20.0,
            )
            upper = np.where(
                np.isfinite(reachability.upper),
                reachability.upper - 1.0e-7,
                20.0,
            )
            base_keyframe_count = min(args.mpc_keyframes, frame_count)
            coarse_distance = build_mpc_distance_grid(
                args.axial_travel_m,
                base_keyframe_count,
                args.mpc_local_refine_start_m,
                args.mpc_local_refine_end_m,
                args.mpc_local_refine_factor,
                tuple(args.mpc_local_refine_window),
            )
            keyframe_count = len(coarse_distance) - 1
            if keyframe_count > frame_count:
                raise RuntimeError(
                    "Refined MPC keyframe count exceeds runtime frame count: "
                    f"keyframes={keyframe_count} frames={frame_count}"
                )
            coarse_q = np.zeros(
                (keyframe_count + 1, TOTAL_DOF), dtype=np.float64
            )
            coarse_q[0] = np.minimum(
                np.maximum(self.reachable_q, lower),
                upper,
            )
            coarse_progress = np.zeros((keyframe_count + 1, 5), dtype=np.float64)
            coarse_target_progress = np.zeros_like(coarse_progress)
            coarse_auto_rephase_offset_m = np.zeros(
                (keyframe_count + 1, 4),
                dtype=np.float64,
            )
            coarse_feasibility_bridge = np.zeros(
                keyframe_count + 1,
                dtype=np.bool_,
            )
            coarse_suffix_horizon = np.zeros(
                keyframe_count + 1,
                dtype=np.bool_,
            )
            coarse_static_feasibility_bridge = np.zeros(
                keyframe_count + 1,
                dtype=np.bool_,
            )
            coarse_static_bridge_dwell_m = np.zeros(
                keyframe_count + 1,
                dtype=np.float64,
            )
            coarse_recovery_bridge = np.zeros(
                keyframe_count + 1,
                dtype=np.bool_,
            )
            coarse_recovery_bridge_dwell_m = np.zeros(
                keyframe_count + 1,
                dtype=np.float64,
            )
            coarse_normal_error = np.zeros_like(coarse_progress)
            coarse_palm_target = np.zeros(
                (keyframe_count + 1, 3), dtype=np.float64
            )
            coarse_palm_target[0] = self.kinematic_targets[0]
            coarse_palm_position_error = np.zeros(
                keyframe_count + 1, dtype=np.float64
            )
            coarse_cost = np.zeros(keyframe_count + 1, dtype=np.float64)
            coarse_nfev = np.zeros(keyframe_count + 1, dtype=np.int32)
            auto_refine_inserted_distance_m: list[float] = []
            auto_refine_inserted_reason: list[str] = []
            suffix_horizon_attempt_count = 0
            suffix_horizon_success_count = 0
            suffix_horizon_cache: dict[str, np.ndarray | float] | None = None
            last_suffix_horizon_evidence: dict[str, object] | None = None
            planner_frame_target_distance = np.linspace(
                0.0,
                args.axial_travel_m,
                frame_count + 1,
                dtype=np.float64,
            )[1:]
            suffix_terminal_start_m = terminal_contact_start_distance(
                args.axial_travel_m,
                frame_count,
                min(args.final_contact_recovery_frames, frame_count),
            )
            print(
                "[MPC-GRID] "
                f"base_keyframes={base_keyframe_count} "
                f"actual_keyframes={keyframe_count} "
                f"local_refine_m="
                f"[{args.mpc_local_refine_start_m:.4f},"
                f"{args.mpc_local_refine_end_m:.4f}] "
                f"factor={args.mpc_local_refine_factor} "
                f"extra_windows={args.mpc_local_refine_window} "
                "auto_refine="
                f"{args.mpc_auto_refine_min_step_mm:.3f}mm/"
                f"{args.mpc_auto_refine_max_insertions}",
                flush=True,
            )
            planner_pad_alignment = float(
                np.cos(
                    np.deg2rad(
                        args.max_pad_angle_deg
                        - args.planner_pad_angle_margin_deg
                    )
                )
            )
            planner_soft_pad_alignment = float(
                np.cos(np.deg2rad(args.planner_soft_pad_angle_deg))
            )
            planner_tip_geom_target_m = (
                np.asarray(
                    args.planner_tip_geom_target_mm,
                    dtype=np.float64,
                )
                / 1000.0
            )
            planner_tip_geom_inner_cap_m = (
                args.planner_tip_geom_inner_cap_mm / 1000.0
            )
            planner_protected_self_clearance_m = (
                args.planner_protected_self_clearance_mm / 1000.0
            )

            def protected_self_clearance_state(
                q: np.ndarray,
            ) -> tuple[float, str, np.ndarray]:
                clearances = reachability.geometry_pair_distances(
                    q,
                    protected_self_pairs,
                )
                index = int(np.argmin(clearances))
                return (
                    float(clearances[index]),
                    protected_self_pair_labels[index],
                    clearances,
                )

            def protected_self_separation_seeds(
                seed_q: np.ndarray,
                seed_lower: np.ndarray = lower,
                seed_upper: np.ndarray = upper,
            ) -> tuple[np.ndarray, ...]:
                """Nudge the worst protected pair along its measured ascent."""

                if args.collision_mode != "full_robot":
                    return ()
                minimum_clearance, _, clearances = (
                    protected_self_clearance_state(seed_q)
                )
                if minimum_clearance >= planner_protected_self_clearance_m:
                    return ()
                worst_pair = int(np.argmin(clearances))
                maximum_seed_step = min(
                    args.planner_self_separation_seed_step_rad,
                    0.25 * args.max_plan_joint_step_rad,
                )
                # 1e-4 and 1e-5 rad cross different MuJoCo geom-distance
                # branches at the seed42 45 mm ring MCP--DIP boundary.  Start
                # at the empirically stable double-precision scale, fall back
                # once, and return only seeds whose measured distance
                # actually increases.  The validation makes this a general
                # ascent guard rather than relying on one finite-difference
                # stencil being smooth everywhere.
                for fd_step_rad in (1.0e-6, 5.0e-7):
                    plus_distance = np.zeros(TOTAL_DOF, dtype=np.float64)
                    minus_distance = np.zeros(TOTAL_DOF, dtype=np.float64)
                    sample_span = np.zeros(TOTAL_DOF, dtype=np.float64)
                    for joint in range(TOTAL_DOF):
                        plus_q = seed_q.copy()
                        minus_q = seed_q.copy()
                        plus_q[joint] = min(
                            plus_q[joint] + fd_step_rad,
                            seed_upper[joint],
                        )
                        minus_q[joint] = max(
                            minus_q[joint] - fd_step_rad,
                            seed_lower[joint],
                        )
                        sample_span[joint] = plus_q[joint] - minus_q[joint]
                        if sample_span[joint] <= 0.0:
                            sample_span[joint] = 1.0
                            plus_distance[joint] = clearances[worst_pair]
                            minus_distance[joint] = clearances[worst_pair]
                            continue
                        plus_distance[joint] = (
                            reachability.geometry_pair_distances(
                                plus_q,
                                protected_self_pairs,
                            )[worst_pair]
                        )
                        minus_distance[joint] = (
                            reachability.geometry_pair_distances(
                                minus_q,
                                protected_self_pairs,
                            )[worst_pair]
                        )
                    gradient = central_difference_clearance_gradient(
                        plus_distance,
                        minus_distance,
                        sample_span,
                    )
                    for step_scale in (1.0, 0.5, 0.25):
                        candidate_seeds = self_separation_ascent_seeds(
                            seed_q,
                            gradient,
                            seed_lower,
                            seed_upper,
                            maximum_step_rad=(
                                step_scale * maximum_seed_step
                            ),
                        )
                        improving_seeds = tuple(
                            candidate_seed
                            for candidate_seed in candidate_seeds
                            if reachability.geometry_pair_distances(
                                candidate_seed,
                                protected_self_pairs,
                            )[worst_pair]
                            > minimum_clearance + 1.0e-12
                        )
                        if improving_seeds:
                            return improving_seeds
                return ()

            def contact_state(
                q: np.ndarray,
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]:
                points = reachability.forward_points(q)
                surface, normals = capsule_project(
                    points,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                arc, azimuth = capsule_meridian_coordinates(
                    surface,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                # The palm root is not a contact.  Its fifth planning
                # coordinate is exact axial URDF displacement rather than a
                # projection around an irrelevant capsule end cap.
                palm_axis_displacement = float(
                    np.dot(
                        points[0] - self.kinematic_targets[0],
                        rotation[:, 2],
                    )
                )
                arc[0] = start_arc[0] + palm_axis_displacement
                signed_standoff = np.einsum(
                    "ni,ni->n",
                    points - surface,
                    normals,
                )
                return points, surface, normals, arc, np.column_stack(
                    (azimuth, signed_standoff)
                )

            start_q = coarse_q[0].copy()
            if args.collision_mode == "full_robot":
                initial_tip_clearance = (
                    reachability.geometry_group_clearances(
                        start_q,
                        center,
                        rotation,
                    )[0]
                )
                if np.any(
                    initial_tip_clearance
                    < -args.max_contact_penetration_mm / 1000.0
                ):
                    raise RuntimeError(
                        "Initial grasp violates the physical fingertip "
                        "penetration invariant: per_tip_mm="
                        f"{(initial_tip_clearance * 1000).round(3).tolist()} "
                        f"allowed_mm={args.max_contact_penetration_mm:.3f}"
                    )
                print(
                    "[INITIAL-PHYSICAL-TIP-AUDIT] "
                    f"clearance_mm="
                    f"{(initial_tip_clearance * 1000).round(3).tolist()} "
                    f"allowed_penetration_mm="
                    f"{args.max_contact_penetration_mm:.3f}",
                    flush=True,
                )
                initial_self_pairs, initial_self_distances = (
                    reachability.self_collision_contacts(start_q)
                )
                (
                    initial_protected_clearance,
                    initial_protected_pair_name,
                    initial_protected_clearances,
                ) = protected_self_clearance_state(start_q)
                if initial_self_pairs:
                    raise RuntimeError(
                        "Initial grasp contains active robot self contacts: "
                        f"unique_pairs={len(set(initial_self_pairs))} "
                        f"sample_occurrences={len(initial_self_pairs)} "
                        f"deepest_mm={initial_self_distances.min() * 1000:.3f}"
                    )
                print(
                    "[INITIAL-PROTECTED-SELF-AUDIT] "
                    "clearance_mm="
                    f"{(initial_protected_clearances * 1000).round(6).tolist()} "
                    f"minimum_mm={initial_protected_clearance * 1000:.6f} "
                    f"nearest={initial_protected_pair_name} "
                    "soft_target_mm="
                    f"{args.planner_protected_self_clearance_mm:.3f}",
                    flush=True,
                )
            initial_palm_position, initial_palm_rotation = (
                reachability.forward_palm_pose(start_q)
            )
            palm_site_offset_local = (
                initial_palm_rotation.T
                @ (
                    self.kinematic_targets[0]
                    - initial_palm_position
                )
            )
            (
                initial_surface_points,
                _,
                initial_contact_frames,
            ) = capsule_meridian_targets(
                start_arc,
                start_azimuth,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )

            def mean_fingertip_contact_frame(
                frames: np.ndarray,
            ) -> np.ndarray:
                """Build a proper frame for the four-contact surface patch."""

                normal = np.mean(frames[1:, :, 0], axis=0)
                normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
                meridian = np.mean(frames[1:, :, 2], axis=0)
                meridian -= normal * float(np.dot(normal, meridian))
                meridian /= max(
                    float(np.linalg.norm(meridian)),
                    1.0e-12,
                )
                azimuth = np.cross(meridian, normal)
                azimuth /= max(float(np.linalg.norm(azimuth)), 1.0e-12)
                meridian = np.cross(normal, azimuth)
                return np.column_stack((normal, azimuth, meridian))

            initial_contact_frame = mean_fingertip_contact_frame(
                initial_contact_frames
            )
            initial_patch_center = np.mean(
                initial_surface_points[1:],
                axis=0,
            )
            initial_palm_patch_offset = (
                initial_contact_frame.T
                @ (
                    self.kinematic_targets[0]
                    - initial_patch_center
                )
            )
            previous_q = start_q.copy()
            previous_delta = np.zeros(TOTAL_DOF, dtype=np.float64)
            auto_rephase_offset_m = np.zeros(4, dtype=np.float64)
            static_bridge_total_m = 0.0
            recovery_bridge_total_m = 0.0

            def emit_bridge_rejection(
                record: dict[str, object],
            ) -> None:
                """Emit one compact record only at a real fallback boundary."""

                print(
                    "[BRIDGE-REJECTION] "
                    + format_bridge_rejection_record(record),
                    flush=True,
                )

            def raise_adaptive_planner_failure(
                *,
                reason: str,
                message: str,
                keyframe: int,
                desired_distance: float,
                desired_arc: np.ndarray,
                final_best_q: np.ndarray,
                final_best_points: np.ndarray,
                final_best_arc: np.ndarray,
                failure_metrics: dict[str, object],
                bridge_record: dict[str, object] | None,
                rejected_moving_bridge: (
                    RejectedMovingBridgeCandidate | None
                ),
            ) -> None:
                """Save an exhausted coarse-shooting prefix, then raise."""

                # TODO: define a separate snapshot schema for interpolation
                # and route-level audits; this artifact intentionally captures
                # the non-convex coarse shooting bottleneck and its candidate.

                prefix_q = coarse_q[:keyframe].copy()
                prefix_points: list[np.ndarray] = []
                prefix_arcs: list[np.ndarray] = []
                for feasible_q in prefix_q:
                    feasible_state = contact_state(feasible_q)
                    prefix_points.append(feasible_state[0])
                    prefix_arcs.append(feasible_state[3])
                previous_index = max(keyframe - 1, 0)
                recovery_total_limit_m = (
                    args.mpc_recovery_bridge_max_total_ratio
                    * args.axial_travel_m
                )
                static_total_limit_m = (
                    args.mpc_static_bridge_max_total_ratio
                    * args.axial_travel_m
                )
                budget_values: dict[str, object] = {
                    "auto_rephase_offset_m": auto_rephase_offset_m.copy(),
                    "auto_rephase_max_m": (
                        args.mpc_auto_rephase_max_mm / 1000.0
                    ),
                    "feasibility_bridge_rephase_max_m": (
                        args.mpc_feasibility_bridge_max_mm / 1000.0
                    ),
                    "feasibility_bridge_tip_target_scale": (
                        args.mpc_feasibility_bridge_tip_target_scale
                    ),
                    "static_bridge_dwell_m": float(
                        coarse_static_bridge_dwell_m[previous_index]
                    ),
                    "static_bridge_dwell_max_m": (
                        args.mpc_static_bridge_max_dwell_mm / 1000.0
                    ),
                    "static_bridge_total_m": static_bridge_total_m,
                    "static_bridge_total_max_m": static_total_limit_m,
                    "static_bridge_total_remaining_m": (
                        static_total_limit_m - static_bridge_total_m
                    ),
                    "recovery_bridge_dwell_m": float(
                        coarse_recovery_bridge_dwell_m[previous_index]
                    ),
                    "recovery_bridge_dwell_max_m": (
                        args.mpc_recovery_bridge_max_span_mm / 1000.0
                    ),
                    "recovery_bridge_total_m": recovery_bridge_total_m,
                    "recovery_bridge_total_max_m": recovery_total_limit_m,
                    "recovery_bridge_total_remaining_m": (
                        recovery_total_limit_m - recovery_bridge_total_m
                    ),
                    "recovery_terminal_cutoff_m": (
                        args.axial_travel_m
                        - args.mpc_recovery_bridge_terminal_margin_mm
                        / 1000.0
                    ),
                    "auto_refine_insertions_used": len(
                        auto_refine_inserted_distance_m
                    ),
                    "auto_refine_insertions_max": (
                        args.mpc_auto_refine_max_insertions
                    ),
                    "suffix_horizon_nodes": (
                        args.mpc_suffix_horizon_nodes
                    ),
                    "suffix_horizon_attempt_count": (
                        suffix_horizon_attempt_count
                    ),
                    "suffix_horizon_success_count": (
                        suffix_horizon_success_count
                    ),
                    "suffix_min_joint_margin_mrad": (
                        args.mpc_suffix_min_joint_margin_mrad
                    ),
                    "suffix_min_task_margin_mm": (
                        args.mpc_suffix_min_task_margin_mm
                    ),
                    "suffix_terminal_start_m": (
                        suffix_terminal_start_m
                    ),
                }
                if last_suffix_horizon_evidence is not None:
                    for evidence_name, evidence_value in (
                        last_suffix_horizon_evidence.items()
                    ):
                        budget_values[
                            f"last_suffix_horizon_{evidence_name}"
                        ] = evidence_value
                if (
                    bridge_record is not None
                    and bridge_record.get("fallback") == "planner_failure"
                ):
                    emit_bridge_rejection(bridge_record)
                try:
                    saved_path = save_mpc_failure_prefix(
                        args.mpc_failure_prefix_output,
                        reason=reason,
                        keyframe=keyframe,
                        keyframe_count=keyframe_count,
                        failure_distance_m=desired_distance,
                        last_feasible_distance_m=coarse_distance[:keyframe],
                        last_feasible_q_rad=prefix_q,
                        last_feasible_points_m=np.stack(prefix_points),
                        last_feasible_arcs_m=np.stack(prefix_arcs),
                        final_best_desired_arcs_m=desired_arc,
                        final_best_q_rad=final_best_q,
                        final_best_points_m=final_best_points,
                        final_best_arcs_m=final_best_arc,
                        rephase_offset_m=auto_rephase_offset_m,
                        budget_values=budget_values,
                        failure_metrics=failure_metrics,
                        bridge_record=bridge_record,
                        rejected_moving_bridge=rejected_moving_bridge,
                    )
                    print(
                        "[MPC-FAILURE-PREFIX] "
                        f"reason={reason} saved={saved_path.resolve()}",
                        flush=True,
                    )
                except Exception as error:
                    print(
                        "[MPC-FAILURE-PREFIX-WRITE-ERROR] "
                        f"reason={reason} error={error!r}",
                        flush=True,
                    )
                raise RuntimeError(message)

            def insert_auto_refinement(
                *,
                keyframe: int,
                desired_distance: float,
                reason: str,
            ) -> bool:
                """Insert a bounded midpoint shooting state after a failure."""

                nonlocal coarse_distance
                nonlocal coarse_q
                nonlocal coarse_progress
                nonlocal coarse_target_progress
                nonlocal coarse_auto_rephase_offset_m
                nonlocal coarse_feasibility_bridge
                nonlocal coarse_suffix_horizon
                nonlocal coarse_static_feasibility_bridge
                nonlocal coarse_static_bridge_dwell_m
                nonlocal coarse_recovery_bridge
                nonlocal coarse_recovery_bridge_dwell_m
                nonlocal coarse_normal_error
                nonlocal coarse_palm_target
                nonlocal coarse_palm_position_error
                nonlocal coarse_cost
                nonlocal coarse_nfev
                nonlocal keyframe_count
                nonlocal auto_rephase_offset_m
                nonlocal suffix_horizon_cache

                if (
                    args.mpc_auto_refine_min_step_mm <= 0.0
                    or args.mpc_auto_refine_max_insertions <= 0
                    or len(auto_refine_inserted_distance_m)
                    >= args.mpc_auto_refine_max_insertions
                    or keyframe_count >= frame_count
                ):
                    return False
                left_distance = float(coarse_distance[keyframe - 1])
                interval_m = desired_distance - left_distance
                midpoint_m = left_distance + 0.5 * interval_m
                minimum_step_m = (
                    args.mpc_auto_refine_min_step_mm / 1000.0
                )
                if (
                    interval_m <= 0.0
                    or midpoint_m - left_distance
                    < minimum_step_m - 1.0e-12
                ):
                    return False

                coarse_distance = np.insert(
                    coarse_distance,
                    keyframe,
                    midpoint_m,
                )
                coarse_q = np.insert(
                    coarse_q,
                    keyframe,
                    np.zeros_like(coarse_q[0]),
                    axis=0,
                )
                coarse_progress = np.insert(
                    coarse_progress,
                    keyframe,
                    np.zeros_like(coarse_progress[0]),
                    axis=0,
                )
                coarse_target_progress = np.insert(
                    coarse_target_progress,
                    keyframe,
                    np.zeros_like(coarse_target_progress[0]),
                    axis=0,
                )
                coarse_auto_rephase_offset_m = np.insert(
                    coarse_auto_rephase_offset_m,
                    keyframe,
                    np.zeros_like(coarse_auto_rephase_offset_m[0]),
                    axis=0,
                )
                coarse_feasibility_bridge = np.insert(
                    coarse_feasibility_bridge,
                    keyframe,
                    False,
                )
                coarse_suffix_horizon = np.insert(
                    coarse_suffix_horizon,
                    keyframe,
                    False,
                )
                coarse_static_feasibility_bridge = np.insert(
                    coarse_static_feasibility_bridge,
                    keyframe,
                    False,
                )
                coarse_static_bridge_dwell_m = np.insert(
                    coarse_static_bridge_dwell_m,
                    keyframe,
                    0.0,
                )
                coarse_recovery_bridge = np.insert(
                    coarse_recovery_bridge,
                    keyframe,
                    False,
                )
                coarse_recovery_bridge_dwell_m = np.insert(
                    coarse_recovery_bridge_dwell_m,
                    keyframe,
                    0.0,
                )
                coarse_normal_error = np.insert(
                    coarse_normal_error,
                    keyframe,
                    np.zeros_like(coarse_normal_error[0]),
                    axis=0,
                )
                coarse_palm_target = np.insert(
                    coarse_palm_target,
                    keyframe,
                    np.zeros_like(coarse_palm_target[0]),
                    axis=0,
                )
                coarse_palm_position_error = np.insert(
                    coarse_palm_position_error,
                    keyframe,
                    0.0,
                )
                coarse_cost = np.insert(coarse_cost, keyframe, 0.0)
                coarse_nfev = np.insert(coarse_nfev, keyframe, 0)
                keyframe_count += 1
                auto_rephase_offset_m = (
                    coarse_auto_rephase_offset_m[keyframe - 1].copy()
                )
                auto_refine_inserted_distance_m.append(midpoint_m)
                auto_refine_inserted_reason.append(reason)
                suffix_horizon_cache = None
                print(
                    "[AUTO-REFINE] "
                    f"reason={reason} keyframe={keyframe}/{keyframe_count} "
                    f"interval_m=[{left_distance:.6f},"
                    f"{desired_distance:.6f}] "
                    f"inserted_m={midpoint_m:.6f} "
                    f"new_step_mm={(midpoint_m - left_distance) * 1000:.3f} "
                    f"insertions="
                    f"{len(auto_refine_inserted_distance_m)}/"
                    f"{args.mpc_auto_refine_max_insertions}",
                    flush=True,
                )
                return True

            def transient_contact_active(
                surface_distance: float,
            ) -> bool:
                return bool(
                    args.min_planner_contact_fingers < 4
                    and args.transient_contact_end_m
                    > args.transient_contact_start_m
                    and args.transient_contact_start_m
                    < surface_distance
                    < args.transient_contact_end_m
                )

            def transient_recovery_phase(
                surface_distance: float,
                recovery_start_m: float | None = None,
            ) -> float:
                active_recovery_start_m = (
                    args.transient_contact_recovery_start_m
                    if recovery_start_m is None
                    else recovery_start_m
                )
                if (
                    not transient_contact_active(surface_distance)
                    or active_recovery_start_m is None
                    or surface_distance
                    <= active_recovery_start_m
                ):
                    return 0.0
                phase = float(
                    np.clip(
                        (
                            surface_distance
                            - active_recovery_start_m
                        )
                        / (
                            args.transient_contact_end_m
                            - active_recovery_start_m
                        ),
                        0.0,
                        1.0,
                    )
                )
                return phase * phase * (3.0 - 2.0 * phase)

            def transient_progress_recovery_end() -> float:
                if args.transient_progress_recovery_end_m is None:
                    return args.transient_contact_end_m
                return args.transient_progress_recovery_end_m

            def transient_progress_active(
                surface_distance: float,
            ) -> bool:
                return bool(
                    args.min_planner_contact_fingers < 4
                    and transient_progress_recovery_end()
                    > args.transient_contact_start_m
                    and args.transient_contact_start_m
                    < surface_distance
                    < transient_progress_recovery_end()
                )

            def transient_progress_recovery_phase(
                surface_distance: float,
            ) -> float:
                recovery_start_m = (
                    args.transient_contact_recovery_start_m
                )
                if (
                    not transient_progress_active(surface_distance)
                    or recovery_start_m is None
                    or surface_distance <= recovery_start_m
                ):
                    return 0.0
                phase = float(
                    np.clip(
                        (
                            surface_distance - recovery_start_m
                        )
                        / (
                            transient_progress_recovery_end()
                            - recovery_start_m
                        ),
                        0.0,
                        1.0,
                    )
                )
                return phase * phase * (3.0 - 2.0 * phase)

            def scheduled_tip_normal_tolerances(
                surface_distance: float,
            ) -> np.ndarray:
                tolerances = np.full(
                    4,
                    args.mpc_normal_tolerance_mm / 1000.0,
                    dtype=np.float64,
                )
                if transient_contact_active(surface_distance):
                    recovery_phase = transient_recovery_phase(
                        surface_distance,
                        args.transient_contact_normal_recovery_start_m,
                    )
                    tolerances[args.transient_contact_finger] = (
                        (
                            args.mpc_transient_normal_tolerance_mm
                            + recovery_phase
                            * (
                                args.mpc_normal_tolerance_mm
                                - args.mpc_transient_normal_tolerance_mm
                            )
                        )
                        / 1000.0
                    )
                return tolerances

            def scheduled_contact_status(
                tip_normal_error: np.ndarray,
                surface_distance: float,
            ) -> tuple[bool, np.ndarray, np.ndarray]:
                tolerances = scheduled_tip_normal_tolerances(
                    surface_distance
                )
                nominal_contact = (
                    tip_normal_error
                    <= args.mpc_normal_tolerance_mm / 1000.0
                )
                accepted = bool(
                    int(np.count_nonzero(nominal_contact))
                    >= args.min_planner_contact_fingers
                    and np.all(tip_normal_error <= tolerances)
                )
                return accepted, nominal_contact, tolerances

            def recovery_contact_status(
                tip_normal_error: np.ndarray,
            ) -> tuple[bool, np.ndarray, np.ndarray]:
                """Audit a short recovery without redefining nominal contact."""

                nominal_contact = (
                    tip_normal_error
                    <= args.mpc_normal_tolerance_mm / 1000.0
                )
                tolerances = np.full(
                    4,
                    args.mpc_recovery_bridge_normal_tolerance_mm / 1000.0,
                    dtype=np.float64,
                )
                accepted = bool(
                    int(np.count_nonzero(nominal_contact))
                    >= args.mpc_recovery_bridge_min_contact_fingers
                    and np.all(tip_normal_error <= tolerances)
                )
                return accepted, nominal_contact, tolerances

            def scheduled_tip_tangential_tolerances(
                surface_distance: float,
            ) -> np.ndarray:
                tolerances = np.full(
                    4,
                    args.mpc_tangential_tolerance_mm / 1000.0,
                    dtype=np.float64,
                )
                if (
                    transient_contact_active(surface_distance)
                    and args.mpc_transient_tangential_tolerance_mm
                    is not None
                ):
                    recovery_phase = transient_recovery_phase(
                        surface_distance
                    )
                    tolerances[args.transient_contact_finger] = (
                        (
                            args.mpc_transient_tangential_tolerance_mm
                            + recovery_phase
                            * (
                                args.mpc_tangential_tolerance_mm
                                - args.mpc_transient_tangential_tolerance_mm
                            )
                        )
                        / 1000.0
                    )
                return tolerances

            def palm_follow_distance(fingertip_distance: float) -> float:
                """Integrate a smooth late-route palm velocity schedule."""

                base_distance = (
                    args.palm_travel_ratio * fingertip_distance
                )
                late_distance = max(
                    fingertip_distance - args.palm_late_follow_start_m,
                    0.0,
                )
                ramp = args.palm_late_follow_ramp_m
                if late_distance < ramp:
                    phase = late_distance / ramp
                    slowdown_integral = ramp * (
                        phase**3 - 0.5 * phase**4
                    )
                else:
                    # Integral of smoothstep(phase) over [0, 1] is 1/2.
                    slowdown_integral = late_distance - 0.5 * ramp
                return max(
                    base_distance
                    - (1.0 - args.palm_late_follow_ratio)
                    * slowdown_integral,
                    0.0,
                )

            def terminal_rephase_envelope_at(
                surface_distance: float,
            ) -> float:
                coordinate = float(
                    np.clip(
                        (
                            args.axial_travel_m - surface_distance
                        )
                        / min(0.020, args.axial_travel_m),
                        0.0,
                        1.0,
                    )
                )
                return coordinate * coordinate * (3.0 - 2.0 * coordinate)

            def scheduled_fingertip_targets(
                surface_distance: float,
                rephase_offset_m: np.ndarray,
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray,
                np.ndarray,
            ]:
                """Rebuild immutable fingertip targets for a lookahead node."""

                distance = float(surface_distance)
                if not np.isfinite(distance):
                    raise ValueError("suffix target distance must be finite")
                offset = np.asarray(rephase_offset_m, dtype=np.float64)
                if offset.shape != (4,) or not np.all(np.isfinite(offset)):
                    raise ValueError("suffix rephase offset must have shape (4,)")
                target_arc = start_arc + direction * distance
                target_arc[0] = (
                    start_arc[0]
                    + direction * palm_follow_distance(distance)
                )

                def sine_window(start_m: float, end_m: float) -> float:
                    if not start_m < distance < end_m:
                        return 0.0
                    coordinate = (distance - start_m) / (end_m - start_m)
                    return float(np.sin(np.pi * coordinate))

                target_arc[1:] += (
                    direction
                    * args.finger_meridian_gait_mm
                    / 1000.0
                    * sine_window(
                        args.finger_meridian_gait_start_m,
                        args.finger_meridian_gait_end_m,
                    )
                    * np.asarray(args.finger_meridian_gait_scales)
                )
                target_arc[1:] += (
                    direction
                    * args.finger_meridian_correction_mm
                    / 1000.0
                    * sine_window(
                        args.finger_meridian_correction_start_m,
                        args.finger_meridian_correction_end_m,
                    )
                    * np.asarray(args.finger_meridian_correction_scales)
                )
                target_arc[1:] += (
                    direction
                    * args.finger_meridian_terminal_correction_mm
                    / 1000.0
                    * sine_window(
                        args.finger_meridian_terminal_correction_start_m,
                        args.finger_meridian_terminal_correction_end_m,
                    )
                    * np.asarray(
                        args.finger_meridian_terminal_correction_scales
                    )
                )
                target_arc[1:] += (
                    direction
                    * args.finger_meridian_terminal_tail_correction_mm
                    / 1000.0
                    * sine_window(
                        args.finger_meridian_terminal_tail_correction_start_m,
                        args.finger_meridian_terminal_tail_correction_end_m,
                    )
                    * np.asarray(
                        args.finger_meridian_terminal_tail_correction_scales
                    )
                )
                for local_phase_spec in args.finger_meridian_local_phase:
                    (
                        local_amplitude_mm,
                        local_start_m,
                        local_end_m,
                        *local_scales,
                    ) = local_phase_spec
                    target_arc[1:] += (
                        direction
                        * local_amplitude_mm
                        / 1000.0
                        * sine_window(local_start_m, local_end_m)
                        * np.asarray(local_scales)
                    )
                active_rephase_limit_m = (
                    max(
                        args.mpc_auto_rephase_max_mm,
                        args.mpc_feasibility_bridge_max_mm,
                    )
                    / 1000.0
                    * terminal_rephase_envelope_at(distance)
                )
                target_arc[1:] += direction * np.clip(
                    offset,
                    -active_rephase_limit_m,
                    active_rephase_limit_m,
                )

                target_azimuth = start_azimuth.copy()
                target_azimuth[1:] += (
                    args.finger_gait_amplitude_m
                    * np.sin(np.pi * distance / args.axial_travel_m)
                    * np.asarray((1.0, -1.0, 0.75, -0.75))
                    / CAPSULE_RADIUS
                )
                preload_fraction = min(
                    distance / max(0.025, args.axial_travel_m),
                    1.0,
                )
                target_standoff = (
                    initial_signed_standoff
                    + preload_fraction
                    * (target_signed_standoff - initial_signed_standoff)
                )
                progress_tolerance_mm = (
                    args.mpc_progress_tolerance_mm
                    if distance
                    >= args.axial_travel_m - 1.0e-12
                    else args.mpc_intermediate_progress_tolerance_mm
                )
                if (
                    distance < args.axial_travel_m - 1.0e-12
                    and transient_progress_active(distance)
                ):
                    progress_tolerance_mm = (
                        args.mpc_transient_progress_tolerance_mm
                        + transient_progress_recovery_phase(distance)
                        * (
                            args.mpc_intermediate_progress_tolerance_mm
                            - args.mpc_transient_progress_tolerance_mm
                        )
                    )
                return (
                    target_arc,
                    target_azimuth,
                    target_standoff,
                    progress_tolerance_mm / 1000.0,
                    scheduled_tip_normal_tolerances(distance),
                    scheduled_tip_tangential_tolerances(distance),
                )

            def prospective_low_motion_failures(
                candidate_distance_m: np.ndarray,
                candidate_q_rad: np.ndarray,
                *,
                candidate_keyframe: int,
                candidate_marked: bool = False,
            ) -> list[dict[str, object]]:
                """Audit a candidate against the rolling publisher window."""

                knot_distance = np.asarray(
                    candidate_distance_m,
                    dtype=np.float64,
                ).reshape(-1)
                knot_q = np.asarray(
                    candidate_q_rad,
                    dtype=np.float64,
                )
                if knot_q.ndim == 1:
                    knot_q = knot_q.reshape(1, -1)
                if (
                    candidate_keyframe <= 0
                    or candidate_keyframe > keyframe_count
                    or knot_distance.size == 0
                    or knot_q.shape != (knot_distance.size, TOTAL_DOF)
                    or not np.all(np.isfinite(knot_distance))
                    or not np.all(np.isfinite(knot_q))
                ):
                    raise ValueError(
                        "prospective low-motion candidate has invalid shape"
                    )
                anchor_distance = float(
                    coarse_distance[candidate_keyframe - 1]
                )
                if (
                    knot_distance[0] <= anchor_distance + 1.0e-12
                    or np.any(np.diff(knot_distance) <= 0.0)
                ):
                    raise ValueError(
                        "prospective low-motion knots must advance the route"
                    )
                candidate_frame_distance = planner_frame_target_distance[
                    (planner_frame_target_distance
                     > anchor_distance + 1.0e-12)
                    & (planner_frame_target_distance
                       <= knot_distance[-1] + 1.0e-12)
                ]
                if candidate_frame_distance.size == 0:
                    return []
                candidate_frame_q = smoothstep_joint_interpolation(
                    np.concatenate(([anchor_distance], knot_distance)),
                    np.vstack((coarse_q[candidate_keyframe - 1], knot_q)),
                    candidate_frame_distance,
                )
                prefix_frame_distance = planner_frame_target_distance[
                    planner_frame_target_distance
                    <= anchor_distance + 1.0e-12
                ][-LOW_MOTION_DEFAULT_WINDOW_FRAMES:]
                if prefix_frame_distance.size:
                    prefix_frame_q = smoothstep_joint_interpolation(
                        coarse_distance[:candidate_keyframe],
                        coarse_q[:candidate_keyframe],
                        prefix_frame_distance,
                    )
                    prefix_right = np.searchsorted(
                        coarse_distance[:candidate_keyframe],
                        prefix_frame_distance,
                        side="right",
                    )
                    prefix_right = np.clip(
                        prefix_right,
                        1,
                        candidate_keyframe - 1,
                    )
                    prefix_marked = (
                        coarse_static_feasibility_bridge[prefix_right]
                        | coarse_recovery_bridge[prefix_right]
                    )
                else:
                    prefix_frame_q = np.zeros(
                        (0, TOTAL_DOF), dtype=np.float64
                    )
                    prefix_marked = np.zeros(0, dtype=np.bool_)
                audit_distance = np.concatenate(
                    (prefix_frame_distance, candidate_frame_distance)
                )
                audit_q = np.vstack((prefix_frame_q, candidate_frame_q))
                audit_progress: list[np.ndarray] = []
                audit_points: list[np.ndarray] = []
                audit_axial: list[float] = []
                for sample_q in audit_q:
                    sample_state = contact_state(sample_q)
                    sample_progress = direction * (
                        sample_state[3] - start_arc
                    )
                    audit_progress.append(sample_progress)
                    audit_points.append(sample_state[0])
                    audit_axial.append(
                        float(np.min(sample_progress[1:]))
                    )
                audit_marked = np.concatenate(
                    (
                        prefix_marked,
                        np.full(
                            candidate_frame_distance.size,
                            candidate_marked,
                            dtype=np.bool_,
                        ),
                    )
                )
                return find_unmarked_low_motion_windows(
                    np.stack(audit_progress),
                    np.stack(audit_points),
                    audit_distance,
                    audit_marked,
                    np.asarray(audit_axial, dtype=np.float64),
                    window_frames=LOW_MOTION_DEFAULT_WINDOW_FRAMES,
                    forward_progress_ratio=(
                        LOW_MOTION_FORWARD_PROGRESS_RATIO
                    ),
                )

            keyframe = 1
            while keyframe <= keyframe_count:
                last_bridge_rejection_record: dict[str, object] | None = None
                last_rejected_moving_bridge: (
                    RejectedMovingBridgeCandidate | None
                ) = None
                feasibility_bridge_selected = False
                suffix_horizon_selected = False
                pending_suffix_horizon: dict[
                    str, np.ndarray | float
                ] | None = None
                static_feasibility_bridge_selected = False
                recovery_bridge_selected = False
                selected_static_bridge_dwell_m = 0.0
                selected_recovery_bridge_dwell_m = 0.0
                bridge_tip_motion_m = np.zeros(4, dtype=np.float64)
                bridge_joint_motion_rad = 0.0
                desired_distance = float(coarse_distance[keyframe])
                desired_palm_distance = palm_follow_distance(
                    desired_distance
                )
                palm_path_ratio = desired_palm_distance / max(
                    desired_distance,
                    1.0e-12,
                )
                desired_arc = start_arc + direction * desired_distance
                desired_arc[0] = (
                    start_arc[0]
                    + direction
                    * desired_palm_distance
                )
                if (
                    args.finger_meridian_gait_start_m
                    < desired_distance
                    < args.finger_meridian_gait_end_m
                ):
                    meridian_gait_coordinate = (
                        desired_distance
                        - args.finger_meridian_gait_start_m
                    ) / (
                        args.finger_meridian_gait_end_m
                        - args.finger_meridian_gait_start_m
                    )
                    meridian_gait_phase = np.sin(
                        np.pi * meridian_gait_coordinate
                    )
                else:
                    meridian_gait_phase = 0.0
                desired_arc[1:] += (
                    direction
                    * args.finger_meridian_gait_mm
                    / 1000.0
                    * meridian_gait_phase
                    * np.asarray(args.finger_meridian_gait_scales)
                )
                if (
                    args.finger_meridian_correction_start_m
                    < desired_distance
                    < args.finger_meridian_correction_end_m
                ):
                    correction_coordinate = (
                        desired_distance
                        - args.finger_meridian_correction_start_m
                    ) / (
                        args.finger_meridian_correction_end_m
                        - args.finger_meridian_correction_start_m
                    )
                    correction_phase = np.sin(
                        np.pi * correction_coordinate
                    )
                else:
                    correction_phase = 0.0
                desired_arc[1:] += (
                    direction
                    * args.finger_meridian_correction_mm
                    / 1000.0
                    * correction_phase
                    * np.asarray(
                        args.finger_meridian_correction_scales
                    )
                )
                if (
                    args.finger_meridian_terminal_correction_start_m
                    < desired_distance
                    < args.finger_meridian_terminal_correction_end_m
                ):
                    terminal_correction_coordinate = (
                        desired_distance
                        - args.finger_meridian_terminal_correction_start_m
                    ) / (
                        args.finger_meridian_terminal_correction_end_m
                        - args.finger_meridian_terminal_correction_start_m
                    )
                    terminal_correction_phase = np.sin(
                        np.pi * terminal_correction_coordinate
                    )
                else:
                    terminal_correction_phase = 0.0
                desired_arc[1:] += (
                    direction
                    * args.finger_meridian_terminal_correction_mm
                    / 1000.0
                    * terminal_correction_phase
                    * np.asarray(
                        args.finger_meridian_terminal_correction_scales
                    )
                )
                if (
                    args.finger_meridian_terminal_tail_correction_start_m
                    < desired_distance
                    < args.finger_meridian_terminal_tail_correction_end_m
                ):
                    terminal_tail_coordinate = (
                        desired_distance
                        - args.finger_meridian_terminal_tail_correction_start_m
                    ) / (
                        args.finger_meridian_terminal_tail_correction_end_m
                        - args.finger_meridian_terminal_tail_correction_start_m
                    )
                    terminal_tail_phase = np.sin(
                        np.pi * terminal_tail_coordinate
                    )
                else:
                    terminal_tail_phase = 0.0
                desired_arc[1:] += (
                    direction
                    * args.finger_meridian_terminal_tail_correction_mm
                    / 1000.0
                    * terminal_tail_phase
                    * np.asarray(
                        args.finger_meridian_terminal_tail_correction_scales
                    )
                )
                for local_phase_spec in args.finger_meridian_local_phase:
                    (
                        local_amplitude_mm,
                        local_start_m,
                        local_end_m,
                        *local_scales,
                    ) = local_phase_spec
                    if local_start_m < desired_distance < local_end_m:
                        local_coordinate = (
                            desired_distance - local_start_m
                        ) / (local_end_m - local_start_m)
                        local_phase = np.sin(np.pi * local_coordinate)
                    else:
                        local_phase = 0.0
                    desired_arc[1:] += (
                        direction
                        * local_amplitude_mm
                        / 1000.0
                        * local_phase
                        * np.asarray(local_scales)
                    )
                auto_rephase_limit_m = (
                    args.mpc_auto_rephase_max_mm / 1000.0
                )
                if auto_rephase_limit_m > 0.0:
                    decay_m = args.mpc_auto_rephase_decay_mm / 1000.0
                    decay_allowed = True
                    if keyframe > 1:
                        previous_progress_error = np.abs(
                            coarse_progress[keyframe - 1, 1:]
                            - coarse_target_progress[keyframe - 1, 1:]
                        )
                        previous_progress_tolerance_mm = (
                            args.mpc_intermediate_progress_tolerance_mm
                        )
                        previous_distance = float(
                            coarse_distance[keyframe - 1]
                        )
                        if transient_progress_active(previous_distance):
                            previous_progress_tolerance_mm = (
                                args.mpc_transient_progress_tolerance_mm
                                + transient_progress_recovery_phase(
                                    previous_distance
                                )
                                * (
                                    args.mpc_intermediate_progress_tolerance_mm
                                    - args.mpc_transient_progress_tolerance_mm
                                )
                            )
                        decay_allowed = bool(
                            float(previous_progress_error.max())
                            <= max(
                                previous_progress_tolerance_mm
                                - args.mpc_auto_rephase_margin_mm,
                                0.0,
                            )
                            / 1000.0
                        )
                    if (
                        keyframe > 1
                        and decay_m > 0.0
                        and decay_allowed
                    ):
                        auto_rephase_offset_m -= (
                            np.sign(auto_rephase_offset_m)
                            * np.minimum(
                                np.abs(auto_rephase_offset_m),
                                decay_m,
                            )
                        )
                    # Every automatically introduced asynchronous offset
                    # must return continuously to zero before the common
                    # route endpoint.  The 20 mm terminal envelope is a
                    # target constraint, not an acceptance-band relaxation.
                    terminal_rephase_coordinate = np.clip(
                        (
                            args.axial_travel_m - desired_distance
                        )
                        / min(0.020, args.axial_travel_m),
                        0.0,
                        1.0,
                    )
                    terminal_rephase_envelope = (
                        terminal_rephase_coordinate
                        * terminal_rephase_coordinate
                        * (3.0 - 2.0 * terminal_rephase_coordinate)
                    )
                    auto_rephase_limit_m *= terminal_rephase_envelope
                    auto_rephase_offset_m = np.clip(
                        auto_rephase_offset_m,
                        -auto_rephase_limit_m,
                        auto_rephase_limit_m,
                    )
                    desired_arc[1:] += (
                        direction * auto_rephase_offset_m
                    )
                    coarse_auto_rephase_offset_m[keyframe] = (
                        auto_rephase_offset_m
                    )
                gait_phase = np.sin(
                    np.pi * desired_distance / args.axial_travel_m
                )
                gait_pattern = np.asarray(
                    (1.0, -1.0, 0.75, -0.75)
                )
                desired_azimuth = start_azimuth.copy()
                desired_azimuth[1:] += (
                    args.finger_gait_amplitude_m
                    * gait_phase
                    * gait_pattern
                    / CAPSULE_RADIUS
                )
                active_progress_tolerance_mm = (
                    args.mpc_progress_tolerance_mm
                    if keyframe == keyframe_count
                    else args.mpc_intermediate_progress_tolerance_mm
                )
                if (
                    keyframe != keyframe_count
                    and transient_progress_active(desired_distance)
                ):
                    active_progress_tolerance_mm = (
                        args.mpc_transient_progress_tolerance_mm
                        + transient_progress_recovery_phase(desired_distance)
                        * (
                            args.mpc_intermediate_progress_tolerance_mm
                            - args.mpc_transient_progress_tolerance_mm
                        )
                    )
                tip_normal_tolerances = (
                    scheduled_tip_normal_tolerances(desired_distance)
                )
                tip_tangential_tolerances = (
                    scheduled_tip_tangential_tolerances(
                        desired_distance
                    )
                )
                minimum_progress = (
                    coarse_progress[keyframe - 1]
                )
                preload_fraction = min(
                    desired_distance / max(0.025, args.axial_travel_m),
                    1.0,
                )
                desired_standoff = (
                    initial_signed_standoff
                    + preload_fraction
                    * (target_signed_standoff - initial_signed_standoff)
                )
                desired_surface, _, desired_frames = (
                    capsule_meridian_targets(
                        desired_arc,
                        desired_azimuth,
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                )
                if args.palm_follow_surface_frame:
                    desired_contact_frame = mean_fingertip_contact_frame(
                        desired_frames
                    )
                    full_palm_frame_transport = (
                        desired_contact_frame
                        @ initial_contact_frame.T
                    )
                    active_surface_frame_gain = (
                        args.palm_surface_frame_gain
                    )
                    if args.palm_surface_frame_late_gain is not None:
                        late_gain_phase = float(
                            np.clip(
                                (
                                    desired_distance
                                    - args.palm_surface_frame_late_start_m
                                )
                                / args.palm_surface_frame_late_ramp_m,
                                0.0,
                                1.0,
                            )
                        )
                        late_gain_phase = (
                            late_gain_phase
                            * late_gain_phase
                            * (3.0 - 2.0 * late_gain_phase)
                        )
                        active_surface_frame_gain += (
                            late_gain_phase
                            * (
                                args.palm_surface_frame_late_gain
                                - args.palm_surface_frame_gain
                            )
                        )
                    if (
                        args.palm_surface_frame_terminal_gain
                        is not None
                    ):
                        terminal_gain_phase = float(
                            np.clip(
                                (
                                    desired_distance
                                    - args.palm_surface_frame_terminal_start_m
                                )
                                / args.palm_surface_frame_terminal_ramp_m,
                                0.0,
                                1.0,
                            )
                        )
                        terminal_gain_phase = (
                            terminal_gain_phase
                            * terminal_gain_phase
                            * (3.0 - 2.0 * terminal_gain_phase)
                        )
                        active_surface_frame_gain += (
                            terminal_gain_phase
                            * (
                                args.palm_surface_frame_terminal_gain
                                - active_surface_frame_gain
                            )
                        )
                    palm_frame_transport = R.from_rotvec(
                        active_surface_frame_gain
                        * R.from_matrix(
                            full_palm_frame_transport
                        ).as_rotvec()
                    ).as_matrix()
                    transported_contact_frame = (
                        palm_frame_transport @ initial_contact_frame
                    )
                    desired_palm_rotation = (
                        palm_frame_transport @ initial_palm_rotation
                    )
                else:
                    transported_contact_frame = initial_contact_frame
                    desired_palm_rotation = initial_palm_rotation
                desired_patch_center = np.mean(
                    desired_surface[1:],
                    axis=0,
                )
                transported_palm_target = (
                    desired_patch_center
                    + transported_contact_frame
                    @ initial_palm_patch_offset
                    if args.palm_follow_surface_frame
                    else self.kinematic_targets[0]
                    + direction
                    * desired_distance
                    * rotation[:, 2]
                )
                palm_target = (
                    self.kinematic_targets[0]
                    + palm_path_ratio
                    * (
                        transported_palm_target
                        - self.kinematic_targets[0]
                    )
                )
                clearance_phase = min(
                    desired_distance / args.palm_clearance_ramp_m,
                    1.0,
                )
                clearance_phase = (
                    clearance_phase
                    * clearance_phase
                    * (3.0 - 2.0 * clearance_phase)
                )
                tilt_phase = float(
                    np.clip(
                        (
                            desired_distance
                            - args.palm_clearance_tilt_start_m
                        )
                        / args.palm_clearance_tilt_ramp_m,
                        0.0,
                        1.0,
                    )
                )
                tilt_phase = (
                    tilt_phase
                    * tilt_phase
                    * (3.0 - 2.0 * tilt_phase)
                )
                tilt_release_phase = float(
                    np.clip(
                        (
                            desired_distance
                            - args.palm_clearance_tilt_release_start_m
                        )
                        / args.palm_clearance_tilt_release_ramp_m,
                        0.0,
                        1.0,
                    )
                )
                tilt_release_phase = (
                    tilt_release_phase
                    * tilt_release_phase
                    * (3.0 - 2.0 * tilt_release_phase)
                )
                clearance_tilt = R.from_rotvec(
                    -np.deg2rad(args.palm_clearance_tilt_deg)
                    * tilt_phase
                    * (1.0 - tilt_release_phase)
                    * initial_contact_frame[:, 1]
                ).as_matrix()
                desired_palm_rotation = (
                    clearance_tilt @ desired_palm_rotation
                )
                secondary_clearance_phase = np.clip(
                    (
                        desired_distance
                        - args.palm_clearance_secondary_start_m
                    )
                    / args.palm_clearance_secondary_ramp_m,
                    0.0,
                    1.0,
                )
                secondary_clearance_phase = (
                    secondary_clearance_phase
                    * secondary_clearance_phase
                    * (3.0 - 2.0 * secondary_clearance_phase)
                )
                palm_clearance_direction = (
                    transported_contact_frame[:, 0]
                )
                if args.palm_clearance_use_local_normal:
                    _, palm_local_normals = capsule_project(
                        palm_target[None],
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                    palm_clearance_direction = np.asarray(
                        palm_local_normals[0],
                        dtype=np.float64,
                    )
                palm_target += (
                    (
                        args.palm_clearance_lift_m
                        * clearance_phase
                        + args.palm_clearance_secondary_lift_m
                        * secondary_clearance_phase
                    )
                    * palm_clearance_direction
                )
                palm_target_local_normal = (
                    palm_clearance_direction.copy()
                )
                palm_target_local_normal /= max(
                    float(np.linalg.norm(palm_target_local_normal)),
                    1.0e-12,
                )
                palm_target_local_azimuth = (
                    transported_contact_frame[:, 1].copy()
                )
                palm_target_local_azimuth -= (
                    palm_target_local_normal
                    * float(
                        np.dot(
                            palm_target_local_normal,
                            palm_target_local_azimuth,
                        )
                    )
                )
                palm_target_local_azimuth /= max(
                    float(np.linalg.norm(palm_target_local_azimuth)),
                    1.0e-12,
                )
                palm_target_local_meridian = np.cross(
                    palm_target_local_normal,
                    palm_target_local_azimuth,
                )
                palm_target_local_meridian /= max(
                    float(np.linalg.norm(palm_target_local_meridian)),
                    1.0e-12,
                )
                if (
                    float(
                        np.dot(
                            palm_target_local_meridian,
                            transported_contact_frame[:, 2],
                        )
                    )
                    < 0.0
                ):
                    palm_target_local_meridian *= -1.0
                terminal_palm_offset_phase = float(
                    np.clip(
                        (
                            desired_distance
                            - args.palm_terminal_local_offset_start_m
                        )
                        / args.palm_terminal_local_offset_ramp_m,
                        0.0,
                        1.0,
                    )
                )
                terminal_palm_offset_phase = (
                    terminal_palm_offset_phase
                    * terminal_palm_offset_phase
                    * (3.0 - 2.0 * terminal_palm_offset_phase)
                )
                terminal_palm_offset_m = (
                    np.asarray(
                        args.palm_terminal_local_offset_mm,
                        dtype=np.float64,
                    )
                    / 1000.0
                )
                terminal_second_palm_offset_phase = float(
                    np.clip(
                        (
                            desired_distance
                            - args.palm_terminal_second_local_offset_start_m
                        )
                        / args.palm_terminal_second_local_offset_ramp_m,
                        0.0,
                        1.0,
                    )
                )
                terminal_second_palm_offset_phase = (
                    terminal_second_palm_offset_phase
                    * terminal_second_palm_offset_phase
                    * (3.0 - 2.0 * terminal_second_palm_offset_phase)
                )
                terminal_second_palm_offset_m = (
                    np.asarray(
                        args.palm_terminal_second_local_offset_mm,
                        dtype=np.float64,
                    )
                    / 1000.0
                )
                combined_terminal_palm_offset_m = (
                    terminal_palm_offset_phase * terminal_palm_offset_m
                    + terminal_second_palm_offset_phase
                    * terminal_second_palm_offset_m
                )
                palm_target += (
                    combined_terminal_palm_offset_m[0]
                    * palm_target_local_normal
                    + combined_terminal_palm_offset_m[1]
                    * palm_target_local_azimuth
                    + combined_terminal_palm_offset_m[2]
                    * palm_target_local_meridian
                )
                desired_arc[0] = (
                    start_arc[0]
                    + float(
                        np.dot(
                            palm_target - self.kinematic_targets[0],
                            rotation[:, 2],
                        )
                    )
                )
                coarse_target_progress[keyframe] = direction * (
                    desired_arc - start_arc
                )
                coarse_palm_target[keyframe] = palm_target
                desired_palm_body_position = (
                    palm_target
                    - desired_palm_rotation @ palm_site_offset_local
                )

                def residual(
                    q: np.ndarray,
                    *,
                    joint_regularization: float,
                    progress_scale: float,
                    normal_scale: float,
                    monotonic_scale: float = 150.0,
                    pad_scale: float | None = None,
                    palm_scale: float = 30.0,
                ) -> np.ndarray:
                    points, _, surface_normals, arc, auxiliary = contact_state(q)
                    azimuth = auxiliary[:, 0]
                    signed_standoff = auxiliary[:, 1]
                    azimuth_error = (
                        azimuth - desired_azimuth + np.pi
                    ) % (2.0 * np.pi) - np.pi
                    progress_error = direction * (arc - desired_arc)
                    # The palm root is a non-contact Cartesian/MCC reference.
                    # Its projection onto the object meridian is not a
                    # physical surface-progress constraint.
                    progress_error[0] = 0.0
                    normal_error = signed_standoff - desired_standoff
                    # Treat tracking bounds as a feasibility region instead
                    # of forcing progress and normal errors to compete in a
                    # single weighted sum.  Once an error enters the inner
                    # band, smoothness and azimuth drift select the solution.
                    progress_band = (
                        0.65
                        * active_progress_tolerance_mm
                        / 1000.0
                    )
                    normal_band = 0.55 * tip_normal_tolerances
                    progress_violation = np.sign(progress_error) * np.maximum(
                        np.abs(progress_error) - progress_band,
                        0.0,
                    )
                    progress_violation[0] = 0.0
                    tip_normal_error = normal_error[1:]
                    normal_violation = np.sign(tip_normal_error) * np.maximum(
                        np.abs(tip_normal_error) - normal_band,
                        0.0,
                    )
                    achieved_progress = direction * (arc - start_arc)
                    monotonic_violation = np.maximum(
                        minimum_progress - achieved_progress,
                        0.0,
                    )
                    monotonic_violation[0] = 0.0
                    _, palm_rotation = reachability.forward_palm_pose(q)
                    palm_orientation_error = R.from_matrix(
                        desired_palm_rotation @ palm_rotation.T
                    ).as_rotvec()
                    palm_position_error = points[0] - palm_target
                    palm_position_error_norm = float(
                        np.linalg.norm(palm_position_error)
                    )
                    if args.palm_guide_only:
                        # The FR3/MCC palm motion is passive with respect to
                        # the four hard fingertip tasks. Keep only a weak
                        # Cartesian guide so it moves in the requested broad
                        # direction; do not let palm precision displace a tip.
                        palm_position_violation = np.zeros(3, dtype=np.float64)
                        palm_guide_error = 0.25 * palm_position_error
                    else:
                        palm_position_band = (
                            0.8
                            * args.mpc_palm_position_tolerance_mm
                            / 1000.0
                        )
                        palm_position_violation = (
                            palm_position_error
                            * max(
                                palm_position_error_norm - palm_position_band,
                                0.0,
                            )
                            / max(palm_position_error_norm, 1.0e-12)
                        )
                        palm_guide_error = palm_position_error
                    clearance_violation = np.zeros(1, dtype=np.float64)
                    tip_geom_error = np.zeros(4, dtype=np.float64)
                    tip_geom_inner_violation = np.zeros(
                        4,
                        dtype=np.float64,
                    )
                    protected_self_violation = np.zeros(
                        len(protected_self_pairs),
                        dtype=np.float64,
                    )
                    if args.collision_mode == "full_robot":
                        (
                            tip_geom_clearance,
                            arm_clearance,
                            _,
                            hand_clearance,
                            _,
                        ) = (
                            reachability.geometry_group_clearances(
                                q, center, rotation
                            )
                        )
                        # Aim slightly inside both feasible sets. FR3 stays
                        # outside the object. LEAP Hand contact is allowed,
                        # but the solver is penalized before it reaches the
                        # configured incidental penetration limit.
                        arm_clearance_objective_m = (
                            args.min_arm_clearance_mm + 0.25
                        ) / 1000.0
                        hand_penetration_objective_m = max(
                            args.max_incidental_hand_penetration_mm - 0.25,
                            0.0,
                        ) / 1000.0
                        clearance_violation = np.concatenate(
                            (
                                np.maximum(
                                    arm_clearance_objective_m
                                    - arm_clearance,
                                    0.0,
                                ),
                                np.maximum(
                                    -hand_penetration_objective_m
                                    - hand_clearance,
                                    0.0,
                                ),
                            )
                        )
                        tip_geom_error = (
                            tip_geom_clearance - planner_tip_geom_target_m
                        )
                        tip_geom_inner_violation = np.minimum(
                            tip_geom_clearance
                            - planner_tip_geom_inner_cap_m,
                            0.0,
                        )
                        protected_self_violation = (
                            positive_self_clearance_residual(
                                reachability.geometry_pair_distances(
                                    q,
                                    protected_self_pairs,
                                ),
                                target_clearance_m=(
                                    planner_protected_self_clearance_m
                                ),
                            )
                        )
                    pad_alignment = np.einsum(
                        "ij,ij->i",
                        reachability.fingertip_pad_normals(q),
                        -surface_normals[1:],
                    )
                    pad_alignment_violation = smooth_pad_alignment_residual(
                        pad_alignment,
                        target_alignment=planner_soft_pad_alignment,
                        tau=args.planner_soft_pad_softplus_tau,
                    )
                    active_pad_scale = (
                        args.planner_soft_pad_weight
                        if pad_scale is None
                        else pad_scale
                    )
                    return np.concatenate(
                        (
                            progress_scale * progress_violation
                            + 1.0 * progress_error,
                            normal_scale * normal_violation
                            + 0.3 * tip_normal_error,
                            monotonic_scale * monotonic_violation,
                            palm_scale * palm_position_violation
                            + palm_guide_error,
                            0.02 * palm_orientation_error,
                            0.01
                            * (
                                q[ARM_DOF:TOTAL_DOF]
                                - start_q[ARM_DOF:TOTAL_DOF]
                            ),
                            1000.0 * CAPSULE_RADIUS * azimuth_error,
                            joint_regularization * (q - previous_q),
                            0.0008 * (q - previous_q - previous_delta),
                            1000.0 * clearance_violation,
                            args.planner_tip_geom_weight * tip_geom_error,
                            args.planner_tip_geom_inner_weight
                            * tip_geom_inner_violation,
                            args.planner_protected_self_clearance_weight
                            * protected_self_violation,
                            active_pad_scale * pad_alignment_violation,
                        )
                    )

                candidates = []
                extrapolated_seed = np.minimum(
                    np.maximum(previous_q + previous_delta, lower),
                    upper,
                )

                def segment_collision_status(
                    candidate_q: np.ndarray,
                    segment_start_q: np.ndarray | None = None,
                ) -> tuple[
                    float,
                    str,
                    float,
                    str,
                    float,
                    str,
                    int,
                    int,
                    float,
                    str,
                    float,
                ]:
                    """Audit distinct arm, hand, tip, self, and pad gates."""

                    segment_start = (
                        previous_q
                        if segment_start_q is None
                        else np.asarray(segment_start_q, dtype=np.float64)
                    )
                    if segment_start.shape != candidate_q.shape:
                        raise ValueError(
                            "segment_start_q must match candidate_q"
                        )
                    if not np.all(np.isfinite(segment_start)):
                        raise ValueError("segment_start_q must be finite")

                    if args.collision_mode != "full_robot":
                        return (
                            np.inf,
                            "",
                            np.inf,
                            "",
                            np.inf,
                            "",
                            0,
                            0,
                            planner_protected_self_clearance_m,
                            "",
                            1.0,
                        )
                    sample_count = max(
                        9,
                        int(np.ceil(frame_count / keyframe_count)),
                    )
                    minimum_arm = np.inf
                    nearest_arm = ""
                    minimum_hand = np.inf
                    nearest_hand = ""
                    sampled_tip_clearances: list[np.ndarray] = []
                    active_self_pairs: set[tuple[int, int]] = set()
                    self_pair_sample_occurrences = 0
                    minimum_protected_self_clearance = np.inf
                    minimum_protected_self_pair_name = ""
                    minimum_pad_alignment = 1.0
                    for fraction in np.linspace(
                        0.0,
                        1.0,
                        sample_count + 1,
                    )[1:]:
                        sample_q = (
                            (1.0 - fraction) * segment_start
                            + fraction * candidate_q
                        )
                        (
                            tip_clearances,
                            arm_clearances,
                            arm_geom_names,
                            hand_clearances,
                            hand_geom_names,
                        ) = reachability.geometry_group_clearances(
                            sample_q,
                            center,
                            rotation,
                        )
                        arm_index = int(np.argmin(arm_clearances))
                        arm_clearance = float(arm_clearances[arm_index])
                        arm_geom_name = arm_geom_names[arm_index]
                        if arm_clearance < minimum_arm:
                            minimum_arm = arm_clearance
                            nearest_arm = arm_geom_name
                        hand_index = int(np.argmin(hand_clearances))
                        hand_clearance = float(hand_clearances[hand_index])
                        hand_geom_name = hand_geom_names[hand_index]
                        if hand_clearance < minimum_hand:
                            minimum_hand = hand_clearance
                            nearest_hand = hand_geom_name
                        sampled_tip_clearances.append(tip_clearances.copy())
                        sample_self_pairs, sample_self_distances = (
                            reachability.self_collision_contacts(sample_q)
                        )
                        _ = sample_self_distances
                        active_self_pairs.update(sample_self_pairs)
                        self_pair_sample_occurrences += len(
                            sample_self_pairs
                        )
                        (
                            sample_protected_clearance,
                            sample_protected_pair_name,
                            _,
                        ) = protected_self_clearance_state(sample_q)
                        if (
                            sample_protected_clearance
                            < minimum_protected_self_clearance
                        ):
                            minimum_protected_self_clearance = (
                                sample_protected_clearance
                            )
                            minimum_protected_self_pair_name = (
                                sample_protected_pair_name
                            )
                        _, _, sample_normals, _, _ = contact_state(sample_q)
                        sample_pad_alignment = np.einsum(
                            "ij,ij->i",
                            reachability.fingertip_pad_normals(sample_q),
                            -sample_normals[1:],
                        )
                        minimum_pad_alignment = min(
                            minimum_pad_alignment,
                            float(sample_pad_alignment.min()),
                        )
                    (
                        _,
                        minimum_tip,
                        (_, nearest_tip_index),
                    ) = segment_tip_clearance_status(
                        np.stack(sampled_tip_clearances),
                        maximum_penetration_m=(
                            args.max_contact_penetration_mm / 1000.0
                        ),
                    )
                    nearest_tip = (
                        reachability.model.geom(
                            int(reachability.tip_geom_ids[nearest_tip_index])
                        ).name
                        or ""
                    )
                    return (
                        minimum_arm,
                        nearest_arm,
                        minimum_hand,
                        nearest_hand,
                        minimum_tip,
                        nearest_tip,
                        len(active_self_pairs),
                        self_pair_sample_occurrences,
                        minimum_protected_self_clearance,
                        minimum_protected_self_pair_name,
                        minimum_pad_alignment,
                    )

                transported_offset = initial_frame_offset.copy()
                transported_offset[:, 0] = desired_standoff
                surface_ik_points = (
                    desired_surface
                    + np.einsum(
                        "nij,nj->ni",
                        desired_frames,
                        transported_offset,
                    )
                )
                # The palm root is a five-point planning/MCC reference, not a
                # physical surface contact.  Hold or translate it according
                # to the explicit palm ratio while the four fingertip sites
                # follow the variable-curvature surface.
                surface_ik_points[0] = palm_target
                surface_arm_result = reachability.solve_palm_pose(
                    desired_palm_body_position,
                    desired_palm_rotation,
                    previous_q,
                    position_tolerance=2.5e-4,
                    orientation_tolerance=1.0e-3,
                    max_iterations=args.mpc_max_nfev,
                )
                surface_ik_result = (
                    reachability.solve_fingertips_fixed_arm(
                        surface_ik_points,
                        surface_arm_result.joint_position,
                        tolerance=2.5e-4,
                    )
                )
                surface_ik_seed = np.minimum(
                    np.maximum(
                        surface_ik_result.joint_position,
                        lower,
                    ),
                    upper,
                )
                # A single exact-palm hierarchical IK seed can sit outside the
                # coupled fingertip workspace even when another collision-safe
                # FR3 branch is feasible. Ordinary mode samples only inside its
                # hard palm ball. Guide-only mode instead spans bounded layers
                # of the actual guide-drift region and pairs them with small
                # FR3 redundancy seeds. Every optimized endpoint is still
                # checked against the original guide, joint, collision, pad,
                # and fingertip constraints.
                palm_multistart_surface_seeds: list[np.ndarray] = []
                palm_ball_normal = palm_target_local_normal
                palm_ball_azimuth = palm_target_local_azimuth
                palm_ball_meridian = palm_target_local_meridian
                if args.palm_guide_only:
                    (
                        palm_multistart_offsets_m,
                        palm_multistart_arm_deltas_rad,
                    ) = build_palm_guide_multistart_specs(
                        palm_ball_normal,
                        palm_ball_azimuth,
                        palm_ball_meridian,
                        args.palm_guide_max_drift_mm / 1000.0,
                    )
                else:
                    palm_ball_radius_m = (
                        args.mpc_palm_position_tolerance_mm / 1000.0
                    )
                    palm_ball_offset_fractions = (
                        -0.25 * palm_ball_normal,
                        -0.50 * palm_ball_normal,
                        -0.75 * palm_ball_normal,
                        0.50 * palm_ball_normal,
                        0.50 * palm_ball_azimuth,
                        -0.50 * palm_ball_azimuth,
                        0.50 * palm_ball_meridian,
                        -0.50 * palm_ball_meridian,
                        -0.40 * palm_ball_normal
                        + 0.50 * palm_ball_azimuth,
                        -0.40 * palm_ball_normal
                        - 0.50 * palm_ball_azimuth,
                    )
                    if terminal_palm_offset_phase > 0.0:
                        palm_ball_offset_fractions += (
                            0.75 * palm_ball_normal,
                            0.90 * palm_ball_normal,
                            0.40 * palm_ball_normal
                            + 0.50 * palm_ball_azimuth,
                            0.40 * palm_ball_normal
                            - 0.50 * palm_ball_azimuth,
                            0.40 * palm_ball_normal
                            + 0.50 * palm_ball_meridian,
                            0.40 * palm_ball_normal
                            - 0.50 * palm_ball_meridian,
                            0.50 * palm_ball_normal
                            + 0.35 * palm_ball_azimuth
                            + 0.35 * palm_ball_meridian,
                            0.50 * palm_ball_normal
                            - 0.35 * palm_ball_azimuth
                            + 0.35 * palm_ball_meridian,
                            0.75 * palm_ball_normal
                            + 0.30 * palm_ball_azimuth,
                            0.75 * palm_ball_normal
                            - 0.30 * palm_ball_azimuth,
                            0.75 * palm_ball_normal
                            + 0.30 * palm_ball_meridian,
                            0.75 * palm_ball_normal
                            - 0.30 * palm_ball_meridian,
                            0.70 * palm_ball_normal
                            + 0.35 * palm_ball_azimuth
                            + 0.35 * palm_ball_meridian,
                            0.70 * palm_ball_normal
                            + 0.35 * palm_ball_azimuth
                            - 0.35 * palm_ball_meridian,
                            0.70 * palm_ball_normal
                            - 0.35 * palm_ball_azimuth
                            + 0.35 * palm_ball_meridian,
                            0.70 * palm_ball_normal
                            - 0.35 * palm_ball_azimuth
                            - 0.35 * palm_ball_meridian,
                        )
                    palm_multistart_offsets_m = palm_ball_radius_m * np.stack(
                        palm_ball_offset_fractions
                    )
                    palm_multistart_arm_deltas_rad = np.zeros(
                        (len(palm_ball_offset_fractions), ARM_DOF),
                        dtype=np.float64,
                    )
                for palm_offset_m, arm_seed_delta_rad in zip(
                    palm_multistart_offsets_m,
                    palm_multistart_arm_deltas_rad,
                    strict=True,
                ):
                    shifted_palm_target = palm_target + palm_offset_m
                    shifted_palm_body_position = (
                        shifted_palm_target
                        - desired_palm_rotation @ palm_site_offset_local
                    )
                    shifted_arm_seed = previous_q.copy()
                    shifted_arm_seed[:ARM_DOF] += arm_seed_delta_rad
                    shifted_arm_seed = np.minimum(
                        np.maximum(shifted_arm_seed, lower),
                        upper,
                    )
                    shifted_arm_result = reachability.solve_palm_pose(
                        shifted_palm_body_position,
                        desired_palm_rotation,
                        shifted_arm_seed,
                        position_tolerance=2.5e-4,
                        orientation_tolerance=1.0e-3,
                        max_iterations=args.mpc_max_nfev,
                    )
                    shifted_surface_points = surface_ik_points.copy()
                    shifted_surface_points[0] = shifted_palm_target
                    shifted_finger_result = (
                        reachability.solve_fingertips_fixed_arm(
                            shifted_surface_points,
                            shifted_arm_result.joint_position,
                            tolerance=2.5e-4,
                        )
                    )
                    palm_multistart_surface_seeds.append(
                        np.minimum(
                            np.maximum(
                                shifted_finger_result.joint_position,
                                lower,
                            ),
                            upper,
                        )
                    )
                (
                    surface_ik_arm_clearance,
                    surface_ik_arm_nearest,
                    surface_ik_hand_clearance,
                    surface_ik_hand_nearest,
                    surface_ik_tip_clearance,
                    surface_ik_tip_nearest,
                    surface_ik_self_count,
                    surface_ik_self_occurrences,
                    surface_ik_protected_clearance,
                    surface_ik_protected_pair_name,
                    surface_ik_pad_alignment,
                ) = segment_collision_status(surface_ik_seed)
                endpoint_self_pairs, endpoint_self_distances = (
                    reachability.self_collision_contacts(
                        surface_ik_seed
                    )
                )
                endpoint_self_pair_names = [
                    (
                        reachability.model.geom(pair[0]).name,
                        reachability.model.geom(pair[1]).name,
                    )
                    for pair in endpoint_self_pairs
                ]
                print(
                    "[SURFACE-IK-SEED] "
                    f"keyframe={keyframe}/{keyframe_count} "
                    f"five_point_error_mm="
                    f"{(surface_ik_result.residual_m * 1000).round(2).tolist()} "
                    f"max_arm_step_rad="
                    f"{float(np.max(np.abs(surface_ik_seed[:ARM_DOF] - previous_q[:ARM_DOF]))):.5f} "
                    f"arm_clearance_mm="
                    f"{surface_ik_arm_clearance * 1000:.2f} "
                    f"nearest_arm={surface_ik_arm_nearest or 'none'} "
                    f"hand_clearance_mm="
                    f"{surface_ik_hand_clearance * 1000:.2f} "
                    f"nearest_hand={surface_ik_hand_nearest or 'none'} "
                    f"tip_clearance_mm="
                    f"{surface_ik_tip_clearance * 1000:.2f} "
                    f"nearest_tip={surface_ik_tip_nearest or 'none'} "
                    f"self_collision_unique_pairs={surface_ik_self_count} "
                    "self_collision_sample_occurrences="
                    f"{surface_ik_self_occurrences} "
                    "protected_self_clearance_mm="
                    f"{surface_ik_protected_clearance * 1000:.3f} "
                    f"protected_self_pair={surface_ik_protected_pair_name} "
                    f"max_pad_angle_deg="
                    f"{np.degrees(np.arccos(np.clip(surface_ik_pad_alignment, -1, 1))):.2f}",
                    f"endpoint_self_pairs={endpoint_self_pair_names} "
                    f"endpoint_self_distances_mm="
                    f"{(endpoint_self_distances * 1000).round(6).tolist()}",
                    flush=True,
                )

                rigid_seed_input = previous_q.copy()
                rigid_seed_input[ARM_DOF:] = start_q[ARM_DOF:]
                rigid_arm_result = reachability.solve_palm_pose(
                    desired_palm_body_position,
                    desired_palm_rotation,
                    rigid_seed_input,
                    position_tolerance=1.0e-3,
                    orientation_tolerance=3.0e-3,
                    max_iterations=args.mpc_max_nfev,
                )
                rigid_arm_seed = rigid_arm_result.joint_position
                candidates = []
                (
                    surface_ik_points_actual,
                    _,
                    _,
                    surface_ik_arc,
                    surface_ik_aux,
                ) = contact_state(surface_ik_seed)
                surface_ik_progress_error = np.abs(
                    direction * (surface_ik_arc - desired_arc)
                )
                surface_ik_progress_error[0] = 0.0
                surface_ik_normal_error = np.abs(
                    surface_ik_aux[:, 1] - desired_standoff
                )
                surface_ik_tangential_error = (
                    (
                        surface_ik_aux[:, 0]
                        - desired_azimuth
                        + np.pi
                    )
                    % (2.0 * np.pi)
                    - np.pi
                ) * CAPSULE_RADIUS
                surface_ik_monotonic_error = np.maximum(
                    minimum_progress
                    - direction * (surface_ik_arc - start_arc),
                    0.0,
                )
                surface_ik_monotonic_error[0] = 0.0
                surface_ik_palm_error = float(
                    np.linalg.norm(surface_ik_points_actual[0] - palm_target)
                )
                surface_ik_joint_step = float(
                    np.max(np.abs(surface_ik_seed - previous_q))
                )
                surface_ik_collision_safe = bool(
                    surface_ik_arm_clearance
                    >= args.min_arm_clearance_mm / 1000.0
                    and surface_ik_hand_clearance
                    >= -args.max_incidental_hand_penetration_mm / 1000.0
                    and surface_ik_tip_clearance
                    >= -args.max_contact_penetration_mm / 1000.0
                    and surface_ik_self_count == 0
                    and surface_ik_pad_alignment
                    >= planner_pad_alignment
                )
                surface_ik_normal_ok, _, _ = scheduled_contact_status(
                    surface_ik_normal_error[1:],
                    desired_distance,
                )
                surface_ik_hard_feasible = bool(
                    float(surface_ik_progress_error.max())
                    <= active_progress_tolerance_mm / 1000.0
                    and surface_ik_normal_ok
                    and np.all(
                        np.abs(surface_ik_tangential_error[1:])
                        <= tip_tangential_tolerances
                    )
                    and float(surface_ik_monotonic_error.max())
                    <= args.mpc_monotonic_tolerance_mm / 1000.0
                    and surface_ik_palm_error <= palm_tracking_limit_m
                    and surface_ik_joint_step
                    <= args.max_plan_joint_step_rad + 1.0e-12
                    and surface_ik_collision_safe
                )
                if surface_ik_hard_feasible:
                    # A tightly solved hierarchical IK state already
                    # satisfies the hard path constraints.  Keep it as a raw
                    # candidate so the subsequent unconstrained local least
                    # squares pass cannot destroy a collision-free solution.
                    surface_ik_cost = float(
                        np.sum(surface_ik_result.residual_m**2)
                    )
                    surface_ik_task_score = (
                        8.0 * float(surface_ik_progress_error.max())
                        + 5.0
                        * float(surface_ik_normal_error[1:].max())
                        + 5.0
                        * float(
                            np.abs(
                                surface_ik_tangential_error[1:]
                            ).max()
                        )
                    )
                    candidates.append(
                        (
                            orientation_aware_candidate_rank(
                                hard_feasible=True,
                                hard_violation_score=0.0,
                                minimum_pad_alignment=(
                                    surface_ik_pad_alignment
                                ),
                                hard_pad_alignment=planner_pad_alignment,
                                soft_pad_alignment=(
                                    planner_soft_pad_alignment
                                ),
                                task_error_score=surface_ik_task_score,
                                continuity_error=surface_ik_joint_step,
                                solver_cost=surface_ik_cost,
                                minimum_protected_self_clearance_m=(
                                    surface_ik_protected_clearance
                                ),
                                soft_self_clearance_target_m=(
                                    planner_protected_self_clearance_m
                                ),
                            ),
                            SimpleNamespace(
                                x=surface_ik_seed,
                                cost=surface_ik_cost,
                                nfev=surface_ik_result.iterations,
                                candidate_kind="raw_surface_ik",
                            ),
                            surface_ik_progress_error,
                            surface_ik_normal_error,
                            surface_ik_arc,
                        )
                    )
                (
                    rigid_points,
                    _,
                    _,
                    rigid_arc,
                    rigid_aux,
                ) = contact_state(rigid_arm_seed)
                rigid_progress_error = np.abs(
                    direction * (rigid_arc - desired_arc)
                )
                rigid_progress_error[0] = 0.0
                rigid_normal_error = np.abs(
                    rigid_aux[:, 1] - desired_standoff
                )
                rigid_tangential_error = (
                    (
                        rigid_aux[:, 0] - desired_azimuth + np.pi
                    )
                    % (2.0 * np.pi)
                    - np.pi
                ) * CAPSULE_RADIUS
                rigid_monotonic_error = np.maximum(
                    minimum_progress
                    - direction * (rigid_arc - start_arc),
                    0.0,
                )
                rigid_monotonic_error[0] = 0.0
                rigid_palm_error = float(
                    np.linalg.norm(rigid_points[0] - palm_target)
                )
                rigid_joint_step = float(
                    np.max(np.abs(rigid_arm_seed - previous_q))
                )
                (
                    rigid_arm_clearance,
                    _,
                    rigid_hand_clearance,
                    _,
                    rigid_tip_clearance,
                    _,
                    rigid_self_collision_count,
                    rigid_self_occurrences,
                    rigid_protected_clearance,
                    rigid_protected_pair_name,
                    rigid_pad_alignment,
                ) = segment_collision_status(rigid_arm_seed)
                rigid_collision_safe = bool(
                    rigid_arm_clearance
                    >= args.min_arm_clearance_mm / 1000.0
                    and rigid_hand_clearance
                    >= -args.max_incidental_hand_penetration_mm / 1000.0
                    and rigid_tip_clearance
                    >= -args.max_contact_penetration_mm / 1000.0
                    and rigid_self_collision_count == 0
                    and rigid_pad_alignment >= planner_pad_alignment
                )
                rigid_normal_ok, _, _ = scheduled_contact_status(
                    rigid_normal_error[1:],
                    desired_distance,
                )
                if (
                    args.finger_gait_amplitude_m <= 0.0
                    and
                    float(rigid_progress_error.max())
                    <= active_progress_tolerance_mm / 1000.0
                    and rigid_normal_ok
                    and np.all(
                        np.abs(rigid_tangential_error[1:])
                        <= tip_tangential_tolerances
                    )
                    and float(rigid_monotonic_error.max())
                    <= args.mpc_monotonic_tolerance_mm / 1000.0
                    and rigid_palm_error <= palm_tracking_limit_m
                    and rigid_joint_step
                    <= args.max_plan_joint_step_rad + 1.0e-12
                    and rigid_collision_safe
                ):
                    rigid_task_score = (
                        8.0 * float(rigid_progress_error.max())
                        + 5.0 * float(rigid_normal_error[1:].max())
                        + 5.0
                        * float(
                            np.abs(rigid_tangential_error[1:]).max()
                        )
                    )
                    candidates.append(
                        (
                            orientation_aware_candidate_rank(
                                hard_feasible=True,
                                hard_violation_score=0.0,
                                minimum_pad_alignment=rigid_pad_alignment,
                                hard_pad_alignment=planner_pad_alignment,
                                soft_pad_alignment=(
                                    planner_soft_pad_alignment
                                ),
                                task_error_score=rigid_task_score,
                                continuity_error=rigid_joint_step,
                                solver_cost=0.0,
                                minimum_protected_self_clearance_m=(
                                    rigid_protected_clearance
                                ),
                                soft_self_clearance_target_m=(
                                    planner_protected_self_clearance_m
                                ),
                            ),
                            SimpleNamespace(
                                x=rigid_arm_seed,
                                cost=0.0,
                                nfev=rigid_arm_result.iterations,
                                candidate_kind="rigid_palm_seed",
                            ),
                            rigid_progress_error,
                            rigid_normal_error,
                            rigid_arc,
                        )
                    )
                else:
                    print(
                        "[RIGID-SEED-REJECTED] "
                        f"keyframe={keyframe}/{keyframe_count} "
                        f"progress_error_mm="
                        f"{(rigid_progress_error * 1000).round(2).tolist()} "
                        f"tip_normal_error_mm="
                        f"{(rigid_normal_error[1:] * 1000).round(2).tolist()} "
                        f"palm_pos_error_mm="
                        f"{rigid_arm_result.position_error_m * 1000:.2f} "
                        f"palm_rot_error_rad="
                        f"{rigid_arm_result.orientation_error_rad:.4f} "
                        f"arm_clearance_mm="
                        f"{rigid_arm_clearance * 1000:.2f} "
                        f"hand_clearance_mm="
                        f"{rigid_hand_clearance * 1000:.2f} "
                        f"tip_clearance_mm="
                        f"{rigid_tip_clearance * 1000:.2f} "
                        f"self_collision_unique_pairs="
                        f"{rigid_self_collision_count} "
                        "self_collision_sample_occurrences="
                        f"{rigid_self_occurrences} "
                        "protected_self_clearance_mm="
                        f"{rigid_protected_clearance * 1000:.3f} "
                        f"protected_self_pair={rigid_protected_pair_name}",
                        flush=True,
                    )
                local_seed_specs = [
                    (
                        "task_surface_balanced",
                        surface_ik_seed,
                        0.0001,
                        32.0,
                        32.0,
                        args.planner_soft_pad_weight,
                    ),
                    (
                        "task_surface_strict",
                        surface_ik_seed,
                        0.0,
                        48.0,
                        48.0,
                        args.planner_soft_pad_weight,
                    ),
                    (
                        "task_rigid_seed",
                        rigid_arm_seed,
                        0.0001,
                        24.0,
                        24.0,
                        args.planner_soft_pad_weight,
                    ),
                    (
                        "task_extrapolated",
                        extrapolated_seed,
                        0.0003,
                        24.0,
                        24.0,
                        args.planner_soft_pad_weight,
                    ),
                    (
                        "task_previous_progress",
                        previous_q,
                        0.0001,
                        32.0,
                        20.0,
                        args.planner_soft_pad_weight,
                    ),
                    (
                        "task_previous_normal",
                        previous_q,
                        0.0001,
                        20.0,
                        32.0,
                        args.planner_soft_pad_weight,
                    ),
                    (
                        "task_previous_strict",
                        previous_q,
                        0.0,
                        40.0,
                        40.0,
                        args.planner_soft_pad_weight,
                    ),
                ]
                local_seed_specs.extend(
                    (
                        "task_palm_multistart",
                        seed,
                        0.0001,
                        48.0,
                        48.0,
                        args.planner_soft_pad_weight,
                    )
                    for seed in palm_multistart_surface_seeds
                )
                # These are independent full-23-DoF posture solves, not a
                # moving/static bridge.  Their stronger smooth cone objective
                # gives the arm, palm, and finger joints room to rotate the
                # pads before the immutable 40-degree segment gate is hit.
                orientation_posture_seed_specs = [
                    (
                        "orientation_posture_surface",
                        surface_ik_seed,
                        0.0001,
                        32.0,
                        32.0,
                        4.0 * args.planner_soft_pad_weight,
                    ),
                    (
                        "orientation_posture_extrapolated",
                        extrapolated_seed,
                        0.0003,
                        32.0,
                        32.0,
                        4.0 * args.planner_soft_pad_weight,
                    ),
                    (
                        "orientation_posture_previous",
                        previous_q,
                        0.0001,
                        32.0,
                        32.0,
                        4.0 * args.planner_soft_pad_weight,
                    ),
                ]
                seen_self_separation_seeds: list[np.ndarray] = []
                for separation_source, separation_base in (
                    ("surface", surface_ik_seed),
                    ("extrapolated", extrapolated_seed),
                    ("previous", previous_q),
                ):
                    separation_lower = np.maximum(
                        lower,
                        previous_q - args.max_plan_joint_step_rad,
                    )
                    separation_upper = np.minimum(
                        upper,
                        previous_q + args.max_plan_joint_step_rad,
                    )
                    bounded_separation_base = np.clip(
                        separation_base,
                        separation_lower,
                        separation_upper,
                    )
                    source_clearance, source_pair_name, _ = (
                        protected_self_clearance_state(
                            bounded_separation_base
                        )
                    )
                    for separation_index, separation_seed in enumerate(
                        protected_self_separation_seeds(
                            bounded_separation_base,
                            separation_lower,
                            separation_upper,
                        )
                    ):
                        if any(
                            np.allclose(
                                separation_seed,
                                old_seed,
                                atol=1.0e-12,
                                rtol=0.0,
                            )
                            for old_seed in seen_self_separation_seeds
                        ):
                            continue
                        seen_self_separation_seeds.append(separation_seed)
                        seed_clearance, _, _ = (
                            protected_self_clearance_state(separation_seed)
                        )
                        separation_kind = (
                            f"self_separation_{separation_source}_"
                            f"{separation_index}"
                        )
                        local_seed_specs.append(
                            (
                                f"task_{separation_kind}",
                                separation_seed,
                                0.0001,
                                48.0,
                                48.0,
                                args.planner_soft_pad_weight,
                            )
                        )
                        orientation_posture_seed_specs.append(
                            (
                                f"orientation_posture_{separation_kind}",
                                separation_seed,
                                0.0001,
                                48.0,
                                48.0,
                                4.0 * args.planner_soft_pad_weight,
                            )
                        )
                        print(
                            "[SELF-SEPARATION-SEED] "
                            f"keyframe={keyframe}/{keyframe_count} "
                            f"source={separation_source} "
                            f"index={separation_index} "
                            f"pair={source_pair_name} "
                            "source_clearance_mm="
                            f"{source_clearance * 1000:.6f} "
                            f"seed_clearance_mm="
                            f"{seed_clearance * 1000:.6f}",
                            flush=True,
                        )
                local_seed_specs.extend(orientation_posture_seed_specs)
                for (
                    candidate_kind,
                    seed,
                    regularization,
                    progress_scale,
                    normal_scale,
                    pad_scale,
                ) in local_seed_specs:
                    result = least_squares(
                        lambda q, reg=regularization, ps=progress_scale, ns=normal_scale, pads=pad_scale: residual(
                            q,
                            joint_regularization=reg,
                            progress_scale=ps,
                            normal_scale=ns,
                            pad_scale=pads,
                        ),
                        seed,
                        bounds=(lower, upper),
                        max_nfev=args.mpc_max_nfev,
                        xtol=1.0e-8,
                        ftol=1.0e-8,
                        gtol=1.0e-8,
                        x_scale="jac",
                        diff_step=1.0e-5,
                    )
                    result.candidate_kind = candidate_kind
                    (
                        candidate_points,
                        _,
                        _,
                        candidate_arc,
                        candidate_aux,
                    ) = contact_state(result.x)
                    progress_error = np.abs(
                        direction * (candidate_arc - desired_arc)
                    )
                    progress_error[0] = 0.0
                    normal_error = np.abs(
                        candidate_aux[:, 1] - desired_standoff
                    )
                    candidate_tangential_error = (
                        (
                            candidate_aux[:, 0]
                            - desired_azimuth
                            + np.pi
                        )
                        % (2.0 * np.pi)
                        - np.pi
                    ) * CAPSULE_RADIUS
                    monotonic_error = np.maximum(
                        minimum_progress
                        - direction * (candidate_arc - start_arc),
                        0.0,
                    )
                    monotonic_error[0] = 0.0
                    candidate_task_score = (
                        1000.0 * float(monotonic_error.max())
                        +
                        1000.0
                        * max(
                            float(progress_error.max())
                            - active_progress_tolerance_mm / 1000.0,
                            0.0,
                        )
                        + 1000.0
                        * float(
                            np.max(
                                np.maximum(
                                    normal_error[1:]
                                    - tip_normal_tolerances,
                                    0.0,
                                )
                            )
                        )
                        + 1000.0
                        * float(
                            np.max(
                                np.maximum(
                                    np.abs(
                                        candidate_tangential_error[1:]
                                    )
                                    - tip_tangential_tolerances,
                                    0.0,
                                )
                            )
                        )
                        + 8.0 * float(progress_error.max())
                        + 5.0 * float(normal_error[1:].max())
                        + 5.0
                        * float(
                            np.abs(
                                candidate_tangential_error[1:]
                            ).max()
                        )
                    )
                    candidate_joint_step = float(
                        np.max(np.abs(result.x - previous_q))
                    )
                    score = (
                        candidate_task_score
                        + 0.02 * candidate_joint_step
                        + 1.0e-3 * float(result.cost)
                    )
                    candidate_hard_feasible = True
                    progress_excess = max(
                        float(progress_error.max())
                        - active_progress_tolerance_mm / 1000.0,
                        0.0,
                    )
                    (
                        candidate_normal_ok,
                        candidate_contact_mask,
                        candidate_normal_tolerances,
                    ) = scheduled_contact_status(
                        normal_error[1:],
                        desired_distance,
                    )
                    normal_excess = float(
                        np.max(
                            np.maximum(
                                normal_error[1:]
                                - candidate_normal_tolerances,
                                0.0,
                            )
                        )
                    )
                    tangential_excess = float(
                        np.max(
                            np.maximum(
                                np.abs(candidate_tangential_error[1:])
                                - tip_tangential_tolerances,
                                0.0,
                            )
                        )
                    )
                    monotonic_excess = max(
                        float(monotonic_error.max())
                        - args.mpc_monotonic_tolerance_mm / 1000.0,
                        0.0,
                    )
                    for hard_excess in (
                        progress_excess,
                        tangential_excess,
                        monotonic_excess,
                    ):
                        if hard_excess > 0.0:
                            candidate_hard_feasible = False
                            score += 1.0e6 + 1.0e6 * hard_excess
                    if not candidate_normal_ok:
                        candidate_hard_feasible = False
                        score += (
                            1.0e6
                            + 1.0e6 * normal_excess
                            + 1000.0
                            * max(
                                args.min_planner_contact_fingers
                                - int(
                                    np.count_nonzero(
                                        candidate_contact_mask
                                    )
                                ),
                                0,
                            )
                        )
                    candidate_palm_error = float(
                        np.linalg.norm(candidate_points[0] - palm_target)
                    )
                    if (
                        candidate_palm_error
                        > palm_tracking_limit_m
                    ):
                        candidate_hard_feasible = False
                        score += (
                            1.0e6
                            + 1.0e6
                            * (
                                candidate_palm_error
                                - palm_tracking_limit_m
                            )
                        )
                    candidate_joint_step_excess = max(
                        candidate_joint_step
                        - args.max_plan_joint_step_rad,
                        0.0,
                    )
                    if candidate_joint_step_excess > 0.0:
                        candidate_hard_feasible = False
                        score += (
                            1.0e6
                            + 1.0e6 * candidate_joint_step_excess
                        )
                    candidate_pad_alignment = 1.0
                    candidate_tip_clearance = np.inf
                    candidate_protected_clearance = (
                        planner_protected_self_clearance_m
                    )
                    if args.collision_mode == "full_robot":
                        (
                            arm_clearance,
                            _,
                            hand_clearance,
                            _,
                            candidate_tip_clearance,
                            _,
                            candidate_self_count,
                            _,
                            candidate_protected_clearance,
                            _,
                            candidate_pad_alignment,
                        ) = segment_collision_status(result.x)
                        arm_clearance_violation = max(
                            args.min_arm_clearance_mm / 1000.0
                            - arm_clearance,
                            0.0,
                        )
                        hand_penetration_violation = max(
                            -args.max_incidental_hand_penetration_mm
                            / 1000.0
                            - hand_clearance,
                            0.0,
                        )
                        tip_penetration_violation = max(
                            -args.max_contact_penetration_mm / 1000.0
                            - candidate_tip_clearance,
                            0.0,
                        )
                        for collision_violation in (
                            arm_clearance_violation,
                            hand_penetration_violation,
                            tip_penetration_violation,
                        ):
                            if collision_violation > 0.0:
                                candidate_hard_feasible = False
                                score += (
                                    1.0e6
                                    + 1.0e6 * collision_violation
                                )
                        if candidate_self_count:
                            candidate_hard_feasible = False
                            score += 1.0e6 + candidate_self_count
                        if candidate_pad_alignment < planner_pad_alignment:
                            candidate_hard_feasible = False
                            score += (
                                1.0e6
                                + 1.0e6
                                * (
                                    planner_pad_alignment
                                    - candidate_pad_alignment
                                )
                            )
                    candidates.append(
                        (
                            orientation_aware_candidate_rank(
                                hard_feasible=candidate_hard_feasible,
                                hard_violation_score=max(
                                    score - candidate_task_score,
                                    0.0,
                                ),
                                minimum_pad_alignment=(
                                    candidate_pad_alignment
                                ),
                                hard_pad_alignment=planner_pad_alignment,
                                soft_pad_alignment=(
                                    planner_soft_pad_alignment
                                ),
                                task_error_score=candidate_task_score,
                                continuity_error=candidate_joint_step,
                                solver_cost=float(result.cost),
                                minimum_protected_self_clearance_m=(
                                    candidate_protected_clearance
                                ),
                                soft_self_clearance_target_m=(
                                    planner_protected_self_clearance_m
                                ),
                            ),
                            result,
                            progress_error,
                            normal_error,
                            candidate_arc,
                        )
                    )
                (
                    _,
                    best,
                    progress_error,
                    normal_error,
                    achieved_arc,
                ) = min(candidates, key=lambda item: item[0])
                preliminary_progress = direction * (
                    achieved_arc - start_arc
                )
                preliminary_monotonic_error = np.maximum(
                    minimum_progress - preliminary_progress,
                    0.0,
                )
                preliminary_monotonic_error[0] = 0.0
                preliminary_pad_alignment = 1.0
                preliminary_arm_clearance = np.inf
                preliminary_hand_clearance = np.inf
                preliminary_tip_clearance = np.inf
                preliminary_self_count = 0
                if args.collision_mode == "full_robot":
                    (
                        preliminary_arm_clearance,
                        _,
                        preliminary_hand_clearance,
                        _,
                        preliminary_tip_clearance,
                        _,
                        preliminary_self_count,
                        _,
                        _,
                        _,
                        preliminary_pad_alignment,
                    ) = segment_collision_status(best.x)
                (
                    preliminary_points,
                    _,
                    _,
                    _,
                    best_auxiliary,
                ) = contact_state(best.x)
                preliminary_palm_error = float(
                    np.linalg.norm(preliminary_points[0] - palm_target)
                )
                tangential_error = (
                    (
                        best_auxiliary[:, 0]
                        - desired_azimuth
                        + np.pi
                    )
                    % (2.0 * np.pi)
                    - np.pi
                ) * CAPSULE_RADIUS
                preliminary_normal_ok, _, _ = scheduled_contact_status(
                    normal_error[1:],
                    desired_distance,
                )
                if (
                    float(progress_error.max())
                    > active_progress_tolerance_mm / 1000.0
                    or not preliminary_normal_ok
                    or np.any(
                        np.abs(tangential_error[1:])
                        > tip_tangential_tolerances
                    )
                    or float(preliminary_monotonic_error.max())
                    > args.mpc_monotonic_tolerance_mm / 1000.0
                    or preliminary_palm_error
                    > palm_tracking_limit_m
                    or preliminary_arm_clearance
                    < args.min_arm_clearance_mm / 1000.0
                    or preliminary_hand_clearance
                    < -args.max_incidental_hand_penetration_mm / 1000.0
                    or preliminary_tip_clearance
                    < -args.max_contact_penetration_mm / 1000.0
                    or float(np.max(np.abs(best.x - previous_q)))
                    > args.max_plan_joint_step_rad + 1.0e-12
                    or preliminary_self_count > 0
                    or preliminary_pad_alignment < planner_pad_alignment
                ):
                    # Constraint-repair pass: warm-start from the best
                    # compromise and sharply penalize only band violations.
                    # This avoids loosening tolerances for a single contact
                    # that misses the feasible set by a fraction of a mm.
                    repair_seed = best.x.copy()
                    for (
                        progress_scale,
                        normal_scale,
                        monotonic_scale,
                        pad_scale,
                        palm_scale,
                    ) in (
                        (100.0, 60.0, 450.0, 16.0, 60.0),
                        (60.0, 100.0, 700.0, 24.0, 100.0),
                        (120.0, 120.0, 1000.0, 36.0, 160.0),
                        (180.0, 140.0, 1400.0, 56.0, 240.0),
                        (300.0, 180.0, 1800.0, 80.0, 280.0),
                        (500.0, 240.0, 2400.0, 100.0, 360.0),
                        (500.0, 500.0, 3000.0, 120.0, 420.0),
                        (700.0, 800.0, 4000.0, 140.0, 500.0),
                        (800.0, 1200.0, 5000.0, 170.0, 600.0),
                        (1000.0, 1600.0, 6500.0, 200.0, 750.0),
                        # In passive-palm mode, preserve the strict fingertip
                        # recovery endpoint before considering palm accuracy.
                        # These stages do not change any final threshold.
                        (1600.0, 1600.0, 7500.0, 240.0, 900.0),
                        (2400.0, 1800.0, 9000.0, 280.0, 1000.0),
                    ):
                        repaired = least_squares(
                            lambda q, ps=progress_scale, ns=normal_scale,
                            ms=monotonic_scale, pads=pad_scale,
                            palms=palm_scale: residual(
                                q,
                                joint_regularization=0.0,
                                progress_scale=ps,
                                normal_scale=ns,
                                monotonic_scale=ms,
                                pad_scale=pads,
                                palm_scale=palms,
                            ),
                            repair_seed,
                            bounds=(lower, upper),
                            max_nfev=args.mpc_max_nfev,
                            xtol=1.0e-9,
                            ftol=1.0e-9,
                            gtol=1.0e-9,
                            x_scale="jac",
                            diff_step=1.0e-5,
                        )
                        repaired.candidate_kind = "constraint_repair"
                        (
                            repaired_points,
                            _,
                            _,
                            repaired_arc,
                            repaired_aux,
                        ) = contact_state(repaired.x)
                        repaired_progress_error = np.abs(
                            direction * (repaired_arc - desired_arc)
                        )
                        repaired_progress_error[0] = 0.0
                        repaired_normal_error = np.abs(
                            repaired_aux[:, 1] - desired_standoff
                        )
                        repaired_tangential_error = (
                            (
                                repaired_aux[:, 0]
                                - desired_azimuth
                                + np.pi
                            )
                            % (2.0 * np.pi)
                            - np.pi
                        ) * CAPSULE_RADIUS
                        repaired_monotonic_error = np.maximum(
                            minimum_progress
                            - direction * (repaired_arc - start_arc),
                            0.0,
                        )
                        repaired_monotonic_error[0] = 0.0
                        repaired_task_score = (
                            1000.0
                            * float(repaired_monotonic_error.max())
                            +
                            1000.0
                            * max(
                                float(repaired_progress_error.max())
                                - active_progress_tolerance_mm / 1000.0,
                                0.0,
                            )
                            + 1000.0
                            * float(
                                np.max(
                                    np.maximum(
                                        repaired_normal_error[1:]
                                        - tip_normal_tolerances,
                                        0.0,
                                    )
                                )
                            )
                            + 1000.0
                            * float(
                                np.max(
                                    np.maximum(
                                        np.abs(
                                            repaired_tangential_error[1:]
                                        )
                                        - tip_tangential_tolerances,
                                        0.0,
                                    )
                                )
                            )
                            + 8.0 * float(repaired_progress_error.max())
                            + 5.0
                            * float(repaired_normal_error[1:].max())
                            + 5.0
                            * float(
                                np.abs(
                                    repaired_tangential_error[1:]
                                ).max()
                            )
                        )
                        repaired_joint_step = float(
                            np.max(np.abs(repaired.x - previous_q))
                        )
                        repaired_score = (
                            repaired_task_score
                            + 0.02 * repaired_joint_step
                            + 1.0e-3 * float(repaired.cost)
                        )
                        repaired_hard_feasible = True
                        repaired_progress_excess = max(
                            float(repaired_progress_error.max())
                            - active_progress_tolerance_mm / 1000.0,
                            0.0,
                        )
                        (
                            repaired_normal_ok,
                            repaired_contact_mask,
                            repaired_normal_tolerances,
                        ) = scheduled_contact_status(
                            repaired_normal_error[1:],
                            desired_distance,
                        )
                        repaired_normal_excess = float(
                            np.max(
                                np.maximum(
                                    repaired_normal_error[1:]
                                    - repaired_normal_tolerances,
                                    0.0,
                                )
                            )
                        )
                        repaired_tangential_excess = float(
                            np.max(
                                np.maximum(
                                    np.abs(
                                        repaired_tangential_error[1:]
                                    )
                                    - tip_tangential_tolerances,
                                    0.0,
                                )
                            )
                        )
                        repaired_monotonic_excess = max(
                            float(repaired_monotonic_error.max())
                            - args.mpc_monotonic_tolerance_mm / 1000.0,
                            0.0,
                        )
                        for hard_excess in (
                            repaired_progress_excess,
                            repaired_tangential_excess,
                            repaired_monotonic_excess,
                        ):
                            if hard_excess > 0.0:
                                repaired_hard_feasible = False
                                repaired_score += (
                                    1.0e6 + 1.0e6 * hard_excess
                                )
                        if not repaired_normal_ok:
                            repaired_hard_feasible = False
                            repaired_score += (
                                1.0e6
                                + 1.0e6 * repaired_normal_excess
                                + 1000.0
                                * max(
                                    args.min_planner_contact_fingers
                                    - int(
                                        np.count_nonzero(
                                            repaired_contact_mask
                                        )
                                    ),
                                    0,
                                )
                            )
                        repaired_palm_error = float(
                            np.linalg.norm(repaired_points[0] - palm_target)
                        )
                        if (
                            repaired_palm_error
                            > palm_tracking_limit_m
                        ):
                            repaired_hard_feasible = False
                            repaired_score += (
                                1.0e6
                                + 1.0e6
                                * (
                                    repaired_palm_error
                                    - palm_tracking_limit_m
                                )
                            )
                        repaired_joint_step_excess = max(
                            repaired_joint_step
                            - args.max_plan_joint_step_rad,
                            0.0,
                        )
                        if repaired_joint_step_excess > 0.0:
                            repaired_hard_feasible = False
                            repaired_score += (
                                1.0e6
                                + 1.0e6 * repaired_joint_step_excess
                            )
                        repaired_pad_alignment = 1.0
                        repaired_tip_clearance = np.inf
                        repaired_protected_clearance = (
                            planner_protected_self_clearance_m
                        )
                        if args.collision_mode == "full_robot":
                            (
                                arm_clearance,
                                _,
                                hand_clearance,
                                _,
                                repaired_tip_clearance,
                                _,
                                repaired_self_count,
                                _,
                                repaired_protected_clearance,
                                _,
                                repaired_pad_alignment,
                            ) = segment_collision_status(repaired.x)
                            arm_clearance_violation = max(
                                args.min_arm_clearance_mm / 1000.0
                                - arm_clearance,
                                0.0,
                            )
                            hand_penetration_violation = max(
                                -args.max_incidental_hand_penetration_mm
                                / 1000.0
                                - hand_clearance,
                                0.0,
                            )
                            tip_penetration_violation = max(
                                -args.max_contact_penetration_mm / 1000.0
                                - repaired_tip_clearance,
                                0.0,
                            )
                            for collision_violation in (
                                arm_clearance_violation,
                                hand_penetration_violation,
                                tip_penetration_violation,
                            ):
                                if collision_violation > 0.0:
                                    repaired_hard_feasible = False
                                    repaired_score += (
                                        1.0e6
                                        + 1.0e6 * collision_violation
                                    )
                            if repaired_self_count:
                                repaired_hard_feasible = False
                                repaired_score += (
                                    1.0e6 + repaired_self_count
                                )
                            if repaired_pad_alignment < planner_pad_alignment:
                                repaired_hard_feasible = False
                                repaired_score += (
                                    1.0e6
                                    + 1.0e6
                                    * (
                                        planner_pad_alignment
                                        - repaired_pad_alignment
                                    )
                                )
                        candidates.append(
                            (
                                orientation_aware_candidate_rank(
                                    hard_feasible=repaired_hard_feasible,
                                    hard_violation_score=max(
                                        repaired_score
                                        - repaired_task_score,
                                        0.0,
                                    ),
                                    minimum_pad_alignment=(
                                        repaired_pad_alignment
                                    ),
                                    hard_pad_alignment=(
                                        planner_pad_alignment
                                    ),
                                    soft_pad_alignment=(
                                        planner_soft_pad_alignment
                                    ),
                                    task_error_score=(
                                        repaired_task_score
                                    ),
                                    continuity_error=repaired_joint_step,
                                    solver_cost=float(repaired.cost),
                                    minimum_protected_self_clearance_m=(
                                        repaired_protected_clearance
                                    ),
                                    soft_self_clearance_target_m=(
                                        planner_protected_self_clearance_m
                                    ),
                                ),
                                repaired,
                                repaired_progress_error,
                                repaired_normal_error,
                                repaired_arc,
                            )
                        )
                        repair_seed = repaired.x
                    (
                        _,
                        best,
                        progress_error,
                        normal_error,
                        achieved_arc,
                    ) = min(candidates, key=lambda item: item[0])
                (
                    pre_rephase_points,
                    _,
                    _,
                    _,
                    pre_rephase_auxiliary,
                ) = contact_state(best.x)
                pre_rephase_tangential_error = (
                    (
                        pre_rephase_auxiliary[:, 0]
                        - desired_azimuth
                        + np.pi
                    )
                    % (2.0 * np.pi)
                    - np.pi
                ) * CAPSULE_RADIUS
                pre_rephase_progress = direction * (
                    achieved_arc - start_arc
                )
                pre_rephase_monotonic_error = np.maximum(
                    minimum_progress - pre_rephase_progress,
                    0.0,
                )
                pre_rephase_monotonic_error[0] = 0.0
                pre_rephase_normal_ok, _, _ = scheduled_contact_status(
                    normal_error[1:],
                    desired_distance,
                )
                pre_rephase_palm_error = float(
                    np.linalg.norm(
                        pre_rephase_points[0] - palm_target
                    )
                )
                pre_rephase_collision_ok = True
                if args.collision_mode == "full_robot":
                    (
                        pre_rephase_arm_clearance,
                        _,
                        pre_rephase_hand_clearance,
                        _,
                        pre_rephase_tip_clearance,
                        _,
                        pre_rephase_self_count,
                        _,
                        _,
                        _,
                        pre_rephase_pad_alignment,
                    ) = segment_collision_status(best.x)
                    pre_rephase_collision_ok = bool(
                        pre_rephase_arm_clearance
                        >= args.min_arm_clearance_mm / 1000.0
                        and pre_rephase_hand_clearance
                        >= -args.max_incidental_hand_penetration_mm
                        / 1000.0
                        and pre_rephase_tip_clearance
                        >= -args.max_contact_penetration_mm / 1000.0
                        and pre_rephase_self_count == 0
                        and pre_rephase_pad_alignment
                        >= planner_pad_alignment
                    )
                pre_rephase_low_motion_failures = (
                    prospective_low_motion_failures(
                        np.asarray((desired_distance,), dtype=np.float64),
                        best.x,
                        candidate_keyframe=keyframe,
                    )
                )
                auto_rephase_needed = bool(
                    float(progress_error.max())
                    > active_progress_tolerance_mm / 1000.0
                    or not pre_rephase_normal_ok
                    or np.any(
                        np.abs(pre_rephase_tangential_error[1:])
                        > tip_tangential_tolerances
                    )
                    or float(pre_rephase_monotonic_error.max())
                    > args.mpc_monotonic_tolerance_mm / 1000.0
                    or pre_rephase_palm_error
                    > palm_tracking_limit_m
                    or not pre_rephase_collision_ok
                    or bool(pre_rephase_low_motion_failures)
                )
                if (
                    auto_rephase_limit_m > 0.0
                    and auto_rephase_needed
                ):
                    # Failure-triggered bounded joint rephasing.  The
                    # ordinary and repair passes keep the nominal fingertip
                    # targets fixed.  Only if both leave any hard feasibility
                    # band do we shoot a small set of coupled, continuous
                    # per-finger target phases.  Every trial is still checked
                    # against contact, tangent, monotonicity, FR3,
                    # incidental-hand, self-collision, and pad-angle
                    # constraints before it can replace the nominal result.
                    nominal_desired_arc = desired_arc.copy()
                    starting_rephase_offset_m = (
                        auto_rephase_offset_m.copy()
                    )
                    progress_limit_m = (
                        active_progress_tolerance_mm / 1000.0
                    )
                    near_progress_limit = (
                        progress_error[1:]
                        >= (
                            active_progress_tolerance_mm
                            - args.mpc_auto_rephase_margin_mm
                        )
                        / 1000.0
                    )
                    if (
                        float(progress_error.max())
                        <= active_progress_tolerance_mm / 1000.0
                    ):
                        # A normal/tangent/collision/pad-angle failure may be
                        # relieved by any finger's phase, so
                        # do not restrict the coupled search to the current
                        # progress-active subset.
                        near_progress_limit[:] = True
                    if not np.any(near_progress_limit):
                        near_progress_limit[
                            int(np.argmax(progress_error[1:]))
                        ] = True
                    rephase_patterns: list[np.ndarray] = []
                    for finger_index in np.flatnonzero(
                        near_progress_limit
                    ):
                        unit_pattern = np.zeros(4, dtype=np.float64)
                        unit_pattern[finger_index] = 1.0
                        rephase_patterns.append(unit_pattern)
                    rephase_patterns.append(
                        near_progress_limit.astype(np.float64)
                    )
                    weighted_pattern = np.maximum(
                        progress_error[1:]
                        - (
                            active_progress_tolerance_mm
                            - args.mpc_auto_rephase_margin_mm
                        )
                        / 1000.0,
                        0.0,
                    )
                    if float(weighted_pattern.max()) > 0.0:
                        rephase_patterns.append(
                            weighted_pattern
                            / float(weighted_pattern.max())
                        )
                    if near_progress_limit[0] or near_progress_limit[1]:
                        # Measured v89-v92 response showed that the leading
                        # two fingers often need this coupled ratio.
                        rephase_patterns.append(
                            np.asarray((1.0, 2.0 / 3.0, 0.0, 0.0))
                        )
                    unique_patterns: list[np.ndarray] = []
                    seen_patterns: set[tuple[float, ...]] = set()
                    for pattern in rephase_patterns:
                        pattern_key = tuple(np.round(pattern, 6))
                        if pattern_key in seen_patterns:
                            continue
                        seen_patterns.add(pattern_key)
                        unique_patterns.append(pattern)

                    step_m = args.mpc_auto_rephase_step_mm / 1000.0
                    amplitude_candidates_m: list[float] = []
                    trial_amplitude_m = step_m
                    while (
                        trial_amplitude_m
                        < auto_rephase_limit_m - 1.0e-12
                    ):
                        amplitude_candidates_m.append(
                            trial_amplitude_m
                        )
                        trial_amplitude_m *= 2.0
                    amplitude_candidates_m.append(
                        auto_rephase_limit_m
                    )
                    accepted_rephase_candidates = []

                    def fallback_orientation_rank(
                        *,
                        minimum_pad_alignment: float,
                        offset_norm_m: float,
                        progress_error_m: float,
                        solver_cost: float,
                        minimum_protected_self_clearance_m: float,
                    ) -> tuple[float, ...]:
                        """Rank hard-feasible fallback states by pad first."""

                        return orientation_aware_candidate_rank(
                            hard_feasible=True,
                            hard_violation_score=0.0,
                            minimum_pad_alignment=minimum_pad_alignment,
                            hard_pad_alignment=planner_pad_alignment,
                            soft_pad_alignment=planner_soft_pad_alignment,
                            # Preserve the historical offset/progress/cost
                            # tie-break once the entire segment is <=35 deg.
                            task_error_score=offset_norm_m,
                            continuity_error=progress_error_m,
                            solver_cost=solver_cost,
                            minimum_protected_self_clearance_m=(
                                minimum_protected_self_clearance_m
                            ),
                            soft_self_clearance_target_m=(
                                planner_protected_self_clearance_m
                            ),
                        )

                    # Before launching another non-convex least-squares solve,
                    # audit the last accepted URDF state itself.  Across a
                    # very short shooting interval it can remain physically
                    # feasible even when the optimizer immediately jumps to a
                    # different branch.  Project only the signed progress
                    # outside a 0.05 mm inner band into the target phase; no
                    # joint state or physical hard limit is relaxed.
                    bridge_interval_m = (
                        desired_distance
                        - float(coarse_distance[keyframe - 1])
                    )
                    bridge_interval_limit_m = max(
                        0.25 / 1000.0,
                        2.0
                        * args.mpc_auto_refine_min_step_mm
                        / 1000.0,
                    )
                    bridge_interval_short = bool(
                        bridge_interval_m
                        <= bridge_interval_limit_m + 1.0e-12
                    )
                    (
                        bridge_points,
                        _,
                        _,
                        bridge_arc,
                        bridge_aux,
                    ) = contact_state(previous_q)
                    bridge_signed_progress_error = direction * (
                        bridge_arc - desired_arc
                    )
                    bridge_signed_progress_error[0] = 0.0
                    bridge_inner_progress_limit_m = max(
                        progress_limit_m - 0.05 / 1000.0,
                        0.0,
                    )
                    bridge_offset_delta_m = (
                        bridge_signed_progress_error[1:]
                        - np.clip(
                            bridge_signed_progress_error[1:],
                            -bridge_inner_progress_limit_m,
                            bridge_inner_progress_limit_m,
                        )
                    )
                    bridge_rephase_offset_m = np.clip(
                        starting_rephase_offset_m
                        + bridge_offset_delta_m,
                        -(
                            args.mpc_feasibility_bridge_max_mm
                            / 1000.0
                            * terminal_rephase_envelope
                        ),
                        (
                            args.mpc_feasibility_bridge_max_mm
                            / 1000.0
                            * terminal_rephase_envelope
                        ),
                    )
                    bridge_desired_arc = nominal_desired_arc.copy()
                    bridge_desired_arc[1:] += (
                        direction
                        * (
                            bridge_rephase_offset_m
                            - starting_rephase_offset_m
                        )
                    )
                    bridge_progress_error = np.abs(
                        direction * (bridge_arc - bridge_desired_arc)
                    )
                    bridge_progress_error[0] = 0.0
                    bridge_normal_error = np.abs(
                        bridge_aux[:, 1] - desired_standoff
                    )
                    bridge_tangential_error = (
                        (
                            bridge_aux[:, 0]
                            - desired_azimuth
                            + np.pi
                        )
                        % (2.0 * np.pi)
                        - np.pi
                    ) * CAPSULE_RADIUS
                    bridge_progress = direction * (
                        bridge_arc - start_arc
                    )
                    bridge_monotonic_error = np.maximum(
                        minimum_progress - bridge_progress,
                        0.0,
                    )
                    bridge_monotonic_error[0] = 0.0
                    bridge_normal_ok, _, _ = scheduled_contact_status(
                        bridge_normal_error[1:],
                        desired_distance,
                    )
                    bridge_recovery_normal_ok, _, _ = (
                        recovery_contact_status(
                            bridge_normal_error[1:]
                        )
                    )
                    bridge_palm_error = float(
                        np.linalg.norm(
                            bridge_points[0] - palm_target
                        )
                    )
                    bridge_collision_ok = True
                    bridge_pad_alignment = 1.0
                    bridge_protected_clearance = (
                        planner_protected_self_clearance_m
                    )
                    if args.collision_mode == "full_robot":
                        (
                            bridge_arm_clearance,
                            _,
                            bridge_hand_clearance,
                            _,
                            bridge_tip_clearance,
                            _,
                            bridge_self_count,
                            _,
                            bridge_protected_clearance,
                            _,
                            bridge_pad_alignment,
                        ) = segment_collision_status(previous_q)
                        bridge_collision_ok = bool(
                            bridge_arm_clearance
                            >= args.min_arm_clearance_mm / 1000.0
                            and bridge_hand_clearance
                            >= -args.max_incidental_hand_penetration_mm
                            / 1000.0
                            and bridge_tip_clearance
                            >= -args.max_contact_penetration_mm / 1000.0
                            and bridge_self_count == 0
                            and bridge_pad_alignment
                            >= planner_pad_alignment
                        )
                    bridge_joint_limits_ok = bool(
                        np.all(previous_q >= lower - 1.0e-12)
                        and np.all(previous_q <= upper + 1.0e-12)
                    )
                    proposed_static_bridge_dwell_m = (
                        float(coarse_static_bridge_dwell_m[keyframe - 1])
                        + bridge_interval_m
                    )
                    proposed_static_bridge_total_m = (
                        static_bridge_total_m + bridge_interval_m
                    )
                    proposed_recovery_bridge_dwell_m = (
                        float(
                            coarse_recovery_bridge_dwell_m[keyframe - 1]
                        )
                        + bridge_interval_m
                    )
                    proposed_recovery_bridge_total_m = (
                        recovery_bridge_total_m + bridge_interval_m
                    )
                    recovery_bridge_budget_ok = bool(
                        proposed_recovery_bridge_dwell_m
                        <= args.mpc_recovery_bridge_max_span_mm / 1000.0
                        + 1.0e-12
                        and proposed_recovery_bridge_total_m
                        <= (
                            args.mpc_recovery_bridge_max_total_ratio
                            * args.axial_travel_m
                        )
                        + 1.0e-12
                        and desired_distance
                        <= (
                            args.axial_travel_m
                            - args.mpc_recovery_bridge_terminal_margin_mm
                            / 1000.0
                        )
                        + 1.0e-12
                    )
                    static_bridge_progress_limit_m = max(
                        progress_limit_m,
                        args.mpc_static_bridge_progress_tolerance_mm / 1000.0,
                    )
                    bridge_hard_ok = bool(
                        bridge_interval_short
                        and float(bridge_progress_error.max())
                        <= static_bridge_progress_limit_m
                        and bridge_recovery_normal_ok
                        and np.all(
                            np.abs(bridge_tangential_error[1:])
                            <= tip_tangential_tolerances
                        )
                        and float(bridge_monotonic_error.max())
                        <= args.mpc_monotonic_tolerance_mm / 1000.0
                        and bridge_palm_error <= palm_tracking_limit_m
                        and bridge_collision_ok
                        and bridge_joint_limits_ok
                        and proposed_static_bridge_dwell_m
                        <= args.mpc_static_bridge_max_dwell_mm / 1000.0
                        + 1.0e-12
                        and proposed_static_bridge_total_m
                        <= (
                            args.mpc_static_bridge_max_total_ratio
                            * args.axial_travel_m
                        )
                        + 1.0e-12
                        and recovery_bridge_budget_ok
                    )
                    static_bridge_candidate = None
                    if bridge_hard_ok:
                        bridge_result = SimpleNamespace(
                            x=previous_q.copy(),
                            cost=0.0,
                            nfev=0,
                            candidate_kind="static_bridge",
                        )
                        static_offset_norm_m = float(
                            np.linalg.norm(
                                bridge_rephase_offset_m
                                - starting_rephase_offset_m
                            )
                        )
                        static_progress_error_m = float(
                            bridge_progress_error.max()
                        )
                        static_bridge_candidate = (
                            fallback_orientation_rank(
                                minimum_pad_alignment=(
                                    bridge_pad_alignment
                                ),
                                offset_norm_m=static_offset_norm_m,
                                progress_error_m=(
                                    static_progress_error_m
                                ),
                                solver_cost=0.0,
                                minimum_protected_self_clearance_m=(
                                    bridge_protected_clearance
                                ),
                            ),
                            static_offset_norm_m,
                            static_progress_error_m,
                            0.0,
                            bridge_result,
                            bridge_progress_error,
                            bridge_normal_error,
                            bridge_arc,
                            bridge_rephase_offset_m.copy(),
                            bridge_desired_arc.copy(),
                        )
                    desired_arc[:] = nominal_desired_arc
                    for trial_amplitude_m in amplitude_candidates_m:
                        for pattern in unique_patterns:
                            for trial_sign in (1.0, -1.0):
                                trial_rephase_offset_m = np.clip(
                                    starting_rephase_offset_m
                                    + trial_sign
                                    * trial_amplitude_m
                                    * pattern,
                                    -auto_rephase_limit_m,
                                    auto_rephase_limit_m,
                                )
                                if np.allclose(
                                    trial_rephase_offset_m,
                                    starting_rephase_offset_m,
                                    atol=1.0e-12,
                                    rtol=0.0,
                                ):
                                    continue
                                desired_arc[:] = nominal_desired_arc
                                desired_arc[1:] += (
                                    direction
                                    * (
                                        trial_rephase_offset_m
                                        - starting_rephase_offset_m
                                    )
                                )
                                for rephase_seed in (
                                    best.x,
                                    previous_q,
                                ):
                                    rephased = least_squares(
                                        lambda q: residual(
                                            q,
                                            joint_regularization=0.0,
                                            progress_scale=2400.0,
                                            normal_scale=1800.0,
                                            monotonic_scale=9000.0,
                                            pad_scale=280.0,
                                            palm_scale=1000.0,
                                        ),
                                        rephase_seed,
                                        bounds=(lower, upper),
                                        max_nfev=args.mpc_max_nfev,
                                        xtol=1.0e-9,
                                        ftol=1.0e-9,
                                        gtol=1.0e-9,
                                        x_scale="jac",
                                        diff_step=1.0e-5,
                                    )
                                    rephased.candidate_kind = (
                                        "single_finger_rephase"
                                    )
                                    (
                                        rephased_points,
                                        _,
                                        _,
                                        rephased_arc,
                                        rephased_aux,
                                    ) = contact_state(rephased.x)
                                    rephased_progress_error = np.abs(
                                        direction
                                        * (rephased_arc - desired_arc)
                                    )
                                    rephased_progress_error[0] = 0.0
                                    rephased_normal_error = np.abs(
                                        rephased_aux[:, 1]
                                        - desired_standoff
                                    )
                                    rephased_tangential_error = (
                                        (
                                            rephased_aux[:, 0]
                                            - desired_azimuth
                                            + np.pi
                                        )
                                        % (2.0 * np.pi)
                                        - np.pi
                                    ) * CAPSULE_RADIUS
                                    rephased_progress = direction * (
                                        rephased_arc - start_arc
                                    )
                                    rephased_monotonic_error = np.maximum(
                                        minimum_progress
                                        - rephased_progress,
                                        0.0,
                                    )
                                    rephased_monotonic_error[0] = 0.0
                                    (
                                        rephased_normal_ok,
                                        _,
                                        _,
                                    ) = scheduled_contact_status(
                                        rephased_normal_error[1:],
                                        desired_distance,
                                    )
                                    rephased_palm_error = float(
                                        np.linalg.norm(
                                            rephased_points[0]
                                            - palm_target
                                        )
                                    )
                                    rephased_collision_ok = True
                                    rephased_pad_alignment = 1.0
                                    rephased_protected_clearance = (
                                        planner_protected_self_clearance_m
                                    )
                                    if args.collision_mode == "full_robot":
                                        (
                                            rephased_arm_clearance,
                                            _,
                                            rephased_hand_clearance,
                                            _,
                                            rephased_tip_clearance,
                                            _,
                                            rephased_self_count,
                                            _,
                                            rephased_protected_clearance,
                                            _,
                                            rephased_pad_alignment,
                                        ) = segment_collision_status(
                                            rephased.x
                                        )
                                        rephased_collision_ok = bool(
                                            rephased_arm_clearance
                                            >= args.min_arm_clearance_mm
                                            / 1000.0
                                            and rephased_hand_clearance
                                            >= -args.max_incidental_hand_penetration_mm
                                            / 1000.0
                                            and rephased_tip_clearance
                                            >= -args.max_contact_penetration_mm
                                            / 1000.0
                                            and rephased_self_count == 0
                                            and rephased_pad_alignment
                                            >= planner_pad_alignment
                                        )
                                    rephased_hard_ok = bool(
                                        float(
                                            rephased_progress_error.max()
                                        )
                                        <= progress_limit_m
                                        and rephased_normal_ok
                                        and np.all(
                                            np.abs(
                                                rephased_tangential_error[1:]
                                            )
                                            <= tip_tangential_tolerances
                                        )
                                        and float(
                                            rephased_monotonic_error.max()
                                        )
                                        <= args.mpc_monotonic_tolerance_mm
                                        / 1000.0
                                        and rephased_palm_error
                                        <= palm_tracking_limit_m
                                        and float(
                                            np.max(
                                                np.abs(
                                                    rephased.x - previous_q
                                                )
                                            )
                                        )
                                        <= args.max_plan_joint_step_rad
                                        + 1.0e-12
                                        and rephased_collision_ok
                                    )
                                    if rephased_hard_ok:
                                        rephased_hard_ok = not bool(
                                            prospective_low_motion_failures(
                                                np.asarray(
                                                    (desired_distance,),
                                                    dtype=np.float64,
                                                ),
                                                rephased.x,
                                                candidate_keyframe=keyframe,
                                            )
                                        )
                                    if not rephased_hard_ok:
                                        continue
                                    rephased_offset_norm_m = float(
                                        np.linalg.norm(
                                            trial_rephase_offset_m
                                            - starting_rephase_offset_m
                                        )
                                    )
                                    rephased_progress_max_m = float(
                                        rephased_progress_error.max()
                                    )
                                    accepted_rephase_candidates.append(
                                        (
                                            fallback_orientation_rank(
                                                minimum_pad_alignment=(
                                                    rephased_pad_alignment
                                                ),
                                                offset_norm_m=(
                                                    rephased_offset_norm_m
                                                ),
                                                progress_error_m=(
                                                    rephased_progress_max_m
                                                ),
                                                solver_cost=float(
                                                    rephased.cost
                                                ),
                                                minimum_protected_self_clearance_m=(
                                                    rephased_protected_clearance
                                                ),
                                            ),
                                            rephased_offset_norm_m,
                                            rephased_progress_max_m,
                                            float(rephased.cost),
                                            rephased,
                                            rephased_progress_error,
                                            rephased_normal_error,
                                            rephased_arc,
                                            trial_rephase_offset_m.copy(),
                                            desired_arc.copy(),
                                        )
                                    )
                        if accepted_rephase_candidates:
                            break
                    if (
                        not accepted_rephase_candidates
                        and bridge_interval_short
                    ):
                        # The expanded short-step phase range must first be
                        # used by a genuinely moving, locally bounded solve.
                        # This preserves the continuous predecessor branch
                        # without accumulating zero-motion planning pauses.
                        desired_arc[:] = bridge_desired_arc
                        moving_bridge_target_arc = (
                            progress_aware_arc_targets(
                                current_arc_m=bridge_arc[1:],
                                desired_arc_m=bridge_desired_arc[1:],
                                direction=direction,
                                nominal_advance_m=bridge_interval_m,
                                hard_progress_limit_m=progress_limit_m,
                                interior_guard_m=min(
                                    args.mpc_suffix_min_task_margin_mm
                                    / 1000.0,
                                    progress_limit_m,
                                ),
                            )
                        )
                        bridge_anchor_standoff_m = bridge_aux[1:, 1].copy()
                        bridge_anchor_azimuth_rad = bridge_aux[1:, 0].copy()
                        bridge_trust_radius = (
                            bounded_moving_bridge_trust_radius(
                                args.mpc_feasibility_bridge_trust_radius_rad,
                                args.max_plan_joint_step_rad,
                            )
                        )
                        bridge_lower = np.maximum(
                            lower,
                            previous_q - bridge_trust_radius,
                        )
                        bridge_upper = np.minimum(
                            upper,
                            previous_q + bridge_trust_radius,
                        )

                        def build_suffix_horizon_candidate(
                        ) -> SimpleNamespace | None:
                            """Solve and exact-audit an immutable H-node suffix."""

                            nonlocal suffix_horizon_attempt_count
                            nonlocal last_suffix_horizon_evidence
                            if args.mpc_suffix_horizon_nodes <= 0:
                                return None
                            suffix_horizon_attempt_count += 1
                            last_suffix_horizon_evidence = {
                                "invocation_distance_m": np.asarray(
                                    desired_distance, dtype=np.float64
                                ),
                                "anchor_distance_m": np.asarray(
                                    coarse_distance[keyframe - 1],
                                    dtype=np.float64,
                                ),
                                "terminal_start_m": np.asarray(
                                    suffix_terminal_start_m,
                                    dtype=np.float64,
                                ),
                                "status": np.asarray("building"),
                            }
                            horizon_distance = build_receding_horizon_distances(
                                first_distance_m=desired_distance,
                                nominal_step_m=bridge_interval_m,
                                horizon_nodes=args.mpc_suffix_horizon_nodes,
                                route_end_m=args.axial_travel_m,
                                terminal_start_m=suffix_terminal_start_m,
                            )
                            last_suffix_horizon_evidence[
                                "node_distance_m"
                            ] = horizon_distance.copy()
                            if horizon_distance.size < 2:
                                last_suffix_horizon_evidence["status"] = (
                                    np.asarray("insufficient_nodes")
                                )
                                return None
                            node_count = int(horizon_distance.size)
                            target_arc_rows: list[np.ndarray] = []
                            target_azimuth_rows: list[np.ndarray] = []
                            target_standoff_rows: list[np.ndarray] = []
                            progress_limit_rows: list[float] = []
                            normal_tolerance_rows: list[np.ndarray] = []
                            tangent_tolerance_rows: list[np.ndarray] = []
                            local_target_rows: list[np.ndarray] = []
                            prior_target_arc = bridge_arc[1:].copy()
                            prior_distance = float(
                                coarse_distance[keyframe - 1]
                            )
                            task_guard_m = (
                                args.mpc_suffix_min_task_margin_mm
                                / 1000.0
                            )
                            for node_distance in horizon_distance:
                                (
                                    node_arc,
                                    node_azimuth,
                                    node_standoff,
                                    node_progress_limit,
                                    node_normal_tolerance,
                                    node_tangent_tolerance,
                                ) = scheduled_fingertip_targets(
                                    float(node_distance),
                                    bridge_rephase_offset_m,
                                )
                                node_interval = float(
                                    node_distance - prior_distance
                                )
                                local_target = progress_aware_arc_targets(
                                    current_arc_m=prior_target_arc,
                                    desired_arc_m=node_arc[1:],
                                    direction=direction,
                                    nominal_advance_m=node_interval,
                                    hard_progress_limit_m=node_progress_limit,
                                    interior_guard_m=min(
                                        task_guard_m,
                                        node_progress_limit,
                                    ),
                                )
                                target_arc_rows.append(node_arc)
                                target_azimuth_rows.append(node_azimuth)
                                target_standoff_rows.append(node_standoff)
                                progress_limit_rows.append(node_progress_limit)
                                normal_tolerance_rows.append(
                                    node_normal_tolerance
                                )
                                tangent_tolerance_rows.append(
                                    node_tangent_tolerance
                                )
                                local_target_rows.append(local_target)
                                prior_target_arc = local_target
                                prior_distance = float(node_distance)
                            target_arc_array = np.stack(target_arc_rows)
                            target_azimuth_array = np.stack(
                                target_azimuth_rows
                            )
                            target_standoff_array = np.stack(
                                target_standoff_rows
                            )
                            progress_limit_array = np.asarray(
                                progress_limit_rows,
                                dtype=np.float64,
                            )
                            normal_tolerance_array = np.stack(
                                normal_tolerance_rows
                            )
                            tangent_tolerance_array = np.stack(
                                tangent_tolerance_rows
                            )
                            local_target_array = np.stack(local_target_rows)
                            current_schedule_arc = scheduled_fingertip_targets(
                                desired_distance,
                                bridge_rephase_offset_m,
                            )[0]
                            if not np.allclose(
                                current_schedule_arc[1:],
                                bridge_desired_arc[1:],
                                atol=1.0e-8,
                                rtol=0.0,
                            ):
                                print(
                                    "[SUFFIX-HORIZON-SCHEDULE-MISMATCH] "
                                    f"distance_m={desired_distance:.9f} "
                                    "max_arc_delta_mm="
                                    f"{float(np.max(np.abs(current_schedule_arc[1:] - bridge_desired_arc[1:]))) * 1000:.6f}",
                                    flush=True,
                                )
                                last_suffix_horizon_evidence["status"] = (
                                    np.asarray("schedule_mismatch")
                                )
                                last_suffix_horizon_evidence[
                                    "schedule_mismatch_max_arc_delta_m"
                                ] = np.asarray(
                                    np.max(
                                        np.abs(
                                            current_schedule_arc[1:]
                                            - bridge_desired_arc[1:]
                                        )
                                    ),
                                    dtype=np.float64,
                                )
                                return None

                            minimum_joint_margin_rad = (
                                args.mpc_suffix_min_joint_margin_mrad
                                / 1000.0
                            )
                            node_lower = lower + minimum_joint_margin_rad
                            node_upper = upper - minimum_joint_margin_rad
                            if np.any(node_lower >= node_upper):
                                last_suffix_horizon_evidence["status"] = (
                                    np.asarray("empty_joint_interior")
                                )
                                return None
                            flat_lower = np.tile(node_lower, node_count)
                            flat_upper = np.tile(node_upper, node_count)
                            hand_inner_limit_m = -(
                                args.max_incidental_hand_penetration_mm
                                - args.mpc_suffix_min_task_margin_mm
                            ) / 1000.0
                            arm_inner_limit_m = (
                                args.min_arm_clearance_mm
                                + args.mpc_suffix_min_task_margin_mm
                            ) / 1000.0

                            suffix_task_hinge_weight = max(
                                20000.0,
                                6.25
                                * args.mpc_feasibility_bridge_target_weight,
                            )
                            suffix_collision_hinge_weight = 20000.0
                            suffix_protected_self_weight = max(
                                16000.0,
                                4.0
                                * args.planner_protected_self_clearance_weight,
                            )
                            suffix_transition_fractions = np.asarray(
                                (0.25, 0.50, 0.75),
                                dtype=np.float64,
                            )

                            def suffix_node_residual(
                                q_node: np.ndarray,
                                *,
                                node_index: int,
                                prior_q: np.ndarray,
                                prior_arc: np.ndarray,
                                prior_distance: float,
                                feasibility_weight_scale: float = 1.0,
                            ) -> np.ndarray:
                                """Shape one node and its segment; hard audit remains authoritative."""

                                if (
                                    not np.isfinite(feasibility_weight_scale)
                                    or feasibility_weight_scale <= 0.0
                                ):
                                    raise ValueError(
                                        "feasibility_weight_scale must be finite and positive"
                                    )

                                (
                                    node_points,
                                    _,
                                    node_surface_normals,
                                    node_arc,
                                    node_aux,
                                ) = contact_state(q_node)
                                rows: list[np.ndarray] = [
                                    moving_bridge_local_residual(
                                        arc_m=node_arc[1:],
                                        target_arc_m=(
                                            local_target_array[node_index]
                                        ),
                                        standoff_m=node_aux[1:, 1],
                                        anchor_standoff_m=(
                                            target_standoff_array[
                                                node_index, 1:
                                            ]
                                        ),
                                        azimuth_rad=node_aux[1:, 0],
                                        anchor_azimuth_rad=(
                                            target_azimuth_array[
                                                node_index, 1:
                                            ]
                                        ),
                                        q_rad=q_node,
                                        anchor_q_rad=prior_q,
                                        capsule_radius_m=CAPSULE_RADIUS,
                                        task_weight=(
                                            args.mpc_feasibility_bridge_target_weight
                                        ),
                                    )
                                ]

                                node_progress_error = np.abs(
                                    direction
                                    * (
                                        node_arc[1:]
                                        - target_arc_array[node_index, 1:]
                                    )
                                )
                                node_normal_error = np.abs(
                                    node_aux[1:, 1]
                                    - target_standoff_array[node_index, 1:]
                                )
                                node_normal_tolerance = (
                                    np.full(
                                        4,
                                        args.mpc_normal_tolerance_mm
                                        / 1000.0,
                                        dtype=np.float64,
                                    )
                                    if horizon_distance[node_index]
                                    >= suffix_terminal_start_m - 1.0e-12
                                    else normal_tolerance_array[node_index]
                                )
                                node_tangent_error = np.abs(
                                    (
                                        (
                                            node_aux[1:, 0]
                                            - target_azimuth_array[
                                                node_index, 1:
                                            ]
                                            + np.pi
                                        )
                                        % (2.0 * np.pi)
                                        - np.pi
                                    )
                                    * CAPSULE_RADIUS
                                )
                                node_monotonic_error = np.maximum(
                                    direction
                                    * (prior_arc[1:] - node_arc[1:]),
                                    0.0,
                                )
                                node_interval_m = float(
                                    horizon_distance[node_index]
                                    - prior_distance
                                )
                                node_motion_m = direction * (
                                    node_arc[1:] - prior_arc[1:]
                                )
                                minimum_node_motion_m = (
                                    args.mpc_feasibility_bridge_min_progress_ratio
                                    * node_interval_m
                                )
                                rows.append(
                                    strict_suffix_task_hinge_residual(
                                        progress_error_m=node_progress_error,
                                        progress_limit_m=(
                                            progress_limit_array[node_index]
                                        ),
                                        normal_error_m=node_normal_error,
                                        normal_tolerance_m=(
                                            node_normal_tolerance
                                        ),
                                        tangent_error_m=node_tangent_error,
                                        tangent_tolerance_m=(
                                            tangent_tolerance_array[node_index]
                                        ),
                                        monotonic_error_m=(
                                            node_monotonic_error
                                        ),
                                        monotonic_tolerance_m=(
                                            args.mpc_monotonic_tolerance_mm
                                            / 1000.0
                                        ),
                                        tip_motion_m=node_motion_m,
                                        minimum_tip_motion_m=(
                                            minimum_node_motion_m
                                        ),
                                        interior_guard_m=task_guard_m,
                                        weight=(
                                            feasibility_weight_scale
                                            * suffix_task_hinge_weight
                                        ),
                                    )
                                )
                                pad_alignment = np.einsum(
                                    "ij,ij->i",
                                    reachability.fingertip_pad_normals(q_node),
                                    -node_surface_normals[1:],
                                )
                                rows.append(
                                    args.planner_soft_pad_weight
                                    * smooth_pad_alignment_residual(
                                        pad_alignment,
                                        target_alignment=(
                                            planner_soft_pad_alignment
                                        ),
                                        tau=(
                                            args.planner_soft_pad_softplus_tau
                                        ),
                                    )
                                )

                                def append_collision_guidance(
                                    sample_q: np.ndarray,
                                    sample_surface_normals: np.ndarray,
                                    *,
                                    include_pad: bool,
                                ) -> None:
                                    if args.collision_mode != "full_robot":
                                        return
                                    (
                                        tip_clearance,
                                        arm_clearance,
                                        _,
                                        hand_clearance,
                                        _,
                                    ) = reachability.geometry_group_clearances(
                                        sample_q,
                                        center,
                                        rotation,
                                    )
                                    rows.append(
                                        feasibility_weight_scale
                                        * moving_bridge_tip_geometry_residual(
                                            tip_clearance,
                                            planner_tip_geom_target_m,
                                            inner_cap_m=(
                                                planner_tip_geom_inner_cap_m
                                            ),
                                            target_weight=(
                                                args.planner_tip_geom_weight
                                            ),
                                            target_scale=(
                                                args.mpc_feasibility_bridge_tip_target_scale
                                            ),
                                            inner_weight=(
                                                args.planner_tip_geom_inner_weight
                                            ),
                                        )
                                    )
                                    rows.append(
                                        feasibility_weight_scale
                                        * suffix_protected_self_weight
                                        * positive_self_clearance_residual(
                                            reachability.geometry_pair_distances(
                                                sample_q,
                                                protected_self_pairs,
                                            ),
                                            target_clearance_m=(
                                                planner_protected_self_clearance_m
                                            ),
                                        )
                                    )
                                    rows.append(
                                        feasibility_weight_scale
                                        * suffix_collision_hinge_weight
                                        * np.maximum(
                                            arm_inner_limit_m - arm_clearance,
                                            0.0,
                                        )
                                    )
                                    rows.append(
                                        feasibility_weight_scale
                                        * suffix_collision_hinge_weight
                                        * np.maximum(
                                            hand_inner_limit_m - hand_clearance,
                                            0.0,
                                        )
                                    )
                                    if include_pad:
                                        sample_pad_alignment = np.einsum(
                                            "ij,ij->i",
                                            reachability.fingertip_pad_normals(
                                                sample_q
                                            ),
                                            -sample_surface_normals[1:],
                                        )
                                        rows.append(
                                            args.planner_soft_pad_weight
                                            * smooth_pad_alignment_residual(
                                                sample_pad_alignment,
                                                target_alignment=(
                                                    planner_soft_pad_alignment
                                                ),
                                                tau=(
                                                    args.planner_soft_pad_softplus_tau
                                                ),
                                            )
                                        )

                                append_collision_guidance(
                                    q_node,
                                    node_surface_normals,
                                    include_pad=False,
                                )

                                transition_prior_arc = prior_arc
                                for fraction in suffix_transition_fractions:
                                    smooth_fraction = fraction * fraction * (
                                        3.0 - 2.0 * fraction
                                    )
                                    sample_q = (
                                        (1.0 - smooth_fraction) * prior_q
                                        + smooth_fraction * q_node
                                    )
                                    sample_distance = float(
                                        prior_distance
                                        + fraction * node_interval_m
                                    )
                                    (
                                        _,
                                        _,
                                        sample_surface_normals,
                                        sample_arc,
                                        sample_aux,
                                    ) = contact_state(sample_q)
                                    (
                                        sample_target_arc,
                                        sample_target_azimuth,
                                        sample_target_standoff,
                                        sample_progress_limit,
                                        sample_normal_tolerance,
                                        sample_tangent_tolerance,
                                    ) = scheduled_fingertip_targets(
                                        sample_distance,
                                        bridge_rephase_offset_m,
                                    )
                                    if (
                                        sample_distance
                                        >= suffix_terminal_start_m - 1.0e-12
                                    ):
                                        sample_normal_tolerance = np.full(
                                            4,
                                            args.mpc_normal_tolerance_mm
                                            / 1000.0,
                                            dtype=np.float64,
                                        )
                                    sample_progress_error = np.abs(
                                        direction
                                        * (
                                            sample_arc[1:]
                                            - sample_target_arc[1:]
                                        )
                                    )
                                    sample_normal_error = np.abs(
                                        sample_aux[1:, 1]
                                        - sample_target_standoff[1:]
                                    )
                                    sample_tangent_error = np.abs(
                                        (
                                            (
                                                sample_aux[1:, 0]
                                                - sample_target_azimuth[1:]
                                                + np.pi
                                            )
                                            % (2.0 * np.pi)
                                            - np.pi
                                        )
                                        * CAPSULE_RADIUS
                                    )
                                    sample_monotonic_error = np.maximum(
                                        direction
                                        * (
                                            transition_prior_arc[1:]
                                            - sample_arc[1:]
                                        ),
                                        0.0,
                                    )
                                    rows.append(
                                        strict_suffix_task_hinge_residual(
                                            progress_error_m=(
                                                sample_progress_error
                                            ),
                                            progress_limit_m=(
                                                sample_progress_limit
                                            ),
                                            normal_error_m=(
                                                sample_normal_error
                                            ),
                                            normal_tolerance_m=(
                                                sample_normal_tolerance
                                            ),
                                            tangent_error_m=(
                                                sample_tangent_error
                                            ),
                                            tangent_tolerance_m=(
                                                sample_tangent_tolerance
                                            ),
                                            monotonic_error_m=(
                                                sample_monotonic_error
                                            ),
                                            monotonic_tolerance_m=(
                                                args.mpc_monotonic_tolerance_mm
                                                / 1000.0
                                            ),
                                            tip_motion_m=np.zeros(
                                                4, dtype=np.float64
                                            ),
                                            minimum_tip_motion_m=0.0,
                                            interior_guard_m=task_guard_m,
                                            weight=(
                                                feasibility_weight_scale
                                                * suffix_task_hinge_weight
                                            ),
                                            minimum_progressing_fingers=0,
                                        )
                                    )
                                    append_collision_guidance(
                                        sample_q,
                                        sample_surface_normals,
                                        include_pad=True,
                                    )
                                    transition_prior_arc = sample_arc

                                approximate_palm_target = (
                                    palm_target
                                    + direction
                                    * (
                                        horizon_distance[node_index]
                                        - desired_distance
                                    )
                                    * args.palm_travel_ratio
                                    * rotation[:, 2]
                                )
                                rows.append(
                                    20.0
                                    * (node_points[0] - approximate_palm_target)
                                )
                                margin = np.minimum(
                                    q_node - lower,
                                    upper - q_node,
                                )
                                rows.append(
                                    feasibility_weight_scale
                                    * suffix_task_hinge_weight
                                    * np.maximum(
                                        minimum_joint_margin_rad - margin,
                                        0.0,
                                    )
                                )
                                rows.append(
                                    feasibility_weight_scale
                                    * suffix_task_hinge_weight
                                    * np.maximum(
                                        np.abs(q_node - prior_q)
                                        - (
                                            args.max_plan_joint_step_rad
                                            - 5.0e-5
                                        ),
                                        0.0,
                                    )
                                )
                                return np.concatenate(rows)

                            def suffix_residual(flat_q: np.ndarray) -> np.ndarray:
                                q_rows = np.asarray(
                                    flat_q,
                                    dtype=np.float64,
                                ).reshape(node_count, TOTAL_DOF)
                                rows: list[np.ndarray] = []
                                prior_q = previous_q
                                prior_arc = bridge_arc
                                prior_distance = float(
                                    coarse_distance[keyframe - 1]
                                )
                                for node_index, q_node in enumerate(q_rows):
                                    rows.append(
                                        suffix_node_residual(
                                            q_node,
                                            node_index=node_index,
                                            prior_q=prior_q,
                                            prior_arc=prior_arc,
                                            prior_distance=prior_distance,
                                        )
                                    )
                                    prior_q = q_node
                                    prior_arc = contact_state(q_node)[3]
                                    prior_distance = float(
                                        horizon_distance[node_index]
                                    )
                                if node_count > 1:
                                    rows.append(
                                        0.02 * np.diff(q_rows, n=2, axis=0).ravel()
                                        if node_count > 2
                                        else np.zeros(0, dtype=np.float64)
                                    )
                                return np.concatenate(rows)

                            suffix_seeds: list[np.ndarray] = []
                            suffix_seed_kinds: list[str] = []

                            def append_suffix_seed(
                                q_rows: np.ndarray,
                                seed_kind: str,
                            ) -> None:
                                candidate = np.asarray(
                                    q_rows,
                                    dtype=np.float64,
                                ).reshape(node_count, TOTAL_DOF)
                                candidate = np.clip(
                                    candidate,
                                    node_lower,
                                    node_upper,
                                )
                                if any(
                                    np.allclose(
                                        candidate,
                                        existing,
                                        atol=1.0e-12,
                                        rtol=0.0,
                                    )
                                    for existing in suffix_seeds
                                ):
                                    return
                                suffix_seeds.append(candidate)
                                suffix_seed_kinds.append(str(seed_kind))

                            append_suffix_seed(
                                np.repeat(previous_q[None, :], node_count, axis=0),
                                "previous",
                            )
                            extrapolated_rows = []
                            extrapolated_q = previous_q.copy()
                            capped_delta = np.clip(
                                previous_delta,
                                -0.020,
                                0.020,
                            )
                            for _ in range(node_count):
                                extrapolated_q = np.clip(
                                    extrapolated_q + capped_delta,
                                    node_lower,
                                    node_upper,
                                )
                                extrapolated_rows.append(extrapolated_q.copy())
                            extrapolated_seed = np.stack(extrapolated_rows)
                            append_suffix_seed(extrapolated_seed, "extrapolated")
                            anchor_joint_margin = np.minimum(
                                previous_q - lower,
                                upper - previous_q,
                            )
                            critical_joint_indices = np.argsort(
                                anchor_joint_margin
                            )[: min(4, TOTAL_DOF)]
                            centered_q = np.clip(
                                previous_q,
                                node_lower,
                                node_upper,
                            )
                            raw_inward_directions: list[np.ndarray] = []
                            combined_inward = np.zeros(
                                TOTAL_DOF, dtype=np.float64
                            )
                            for joint_index in critical_joint_indices:
                                lower_distance = (
                                    centered_q[joint_index]
                                    - lower[joint_index]
                                )
                                upper_distance = (
                                    upper[joint_index]
                                    - centered_q[joint_index]
                                )
                                inward_sign = (
                                    1.0
                                    if lower_distance <= upper_distance
                                    else -1.0
                                )
                                combined_inward[joint_index] = inward_sign
                                individual_inward = np.zeros(
                                    TOTAL_DOF, dtype=np.float64
                                )
                                individual_inward[joint_index] = inward_sign
                                raw_inward_directions.append(individual_inward)
                            raw_inward_directions.insert(0, combined_inward)

                            def suffix_seed_task_feature(
                                q_seed: np.ndarray,
                            ) -> np.ndarray:
                                _, _, _, seed_arc, seed_aux = contact_state(q_seed)
                                seed_tip_clearance = np.zeros(
                                    4, dtype=np.float64
                                )
                                if args.collision_mode == "full_robot":
                                    seed_tip_clearance = (
                                        reachability.geometry_group_clearances(
                                            q_seed,
                                            center,
                                            rotation,
                                        )[0]
                                    )
                                seed_azimuth = seed_aux[1:, 0]
                                return np.concatenate(
                                    (
                                        seed_arc[1:],
                                        seed_aux[1:, 1],
                                        CAPSULE_RADIUS * np.cos(seed_azimuth),
                                        CAPSULE_RADIUS * np.sin(seed_azimuth),
                                        seed_tip_clearance,
                                    )
                                )

                            fd_step_rad = 1.0e-5
                            anchor_feature = suffix_seed_task_feature(centered_q)
                            task_jacobian = np.zeros(
                                (anchor_feature.size, TOTAL_DOF),
                                dtype=np.float64,
                            )
                            for joint_index in range(TOTAL_DOF):
                                plus_q = centered_q.copy()
                                minus_q = centered_q.copy()
                                plus_q[joint_index] = min(
                                    plus_q[joint_index] + fd_step_rad,
                                    node_upper[joint_index],
                                )
                                minus_q[joint_index] = max(
                                    minus_q[joint_index] - fd_step_rad,
                                    node_lower[joint_index],
                                )
                                sample_span = (
                                    plus_q[joint_index]
                                    - minus_q[joint_index]
                                )
                                if sample_span <= 0.0:
                                    continue
                                task_jacobian[:, joint_index] = (
                                    suffix_seed_task_feature(plus_q)
                                    - suffix_seed_task_feature(minus_q)
                                ) / sample_span
                            projected_directions = (
                                damped_task_nullspace_directions(
                                    task_jacobian,
                                    np.stack(raw_inward_directions),
                                    damping=1.0e-6,
                                )
                            )
                            for direction_index, projected_direction in enumerate(
                                projected_directions
                            ):
                                if not np.any(projected_direction):
                                    continue
                                seed_steps = (
                                    (0.002, 0.005)
                                    if direction_index == 0
                                    else (0.005,)
                                )
                                for seed_step in seed_steps:
                                    projected_rows = extrapolated_seed.copy()
                                    projected_rows += (
                                        seed_step * projected_direction[None, :]
                                    )
                                    append_suffix_seed(
                                        projected_rows,
                                        "nullspace_combined"
                                        if direction_index == 0
                                        else f"nullspace_joint_{direction_index - 1}",
                                    )
                            separation_anchor_q = np.clip(
                                previous_q,
                                node_lower,
                                node_upper,
                            )
                            for separation_seed in protected_self_separation_seeds(
                                separation_anchor_q,
                                node_lower,
                                node_upper,
                            ):
                                append_suffix_seed(
                                    transported_suffix_seed_rows(
                                        extrapolated_seed,
                                        separation_anchor_q,
                                        separation_seed,
                                    ),
                                    "protected_self",
                                )
                            cache_seed: np.ndarray | None = None
                            if suffix_horizon_cache is not None:
                                cached_anchor_distance = float(
                                    suffix_horizon_cache["anchor_distance_m"]
                                )
                                cached_anchor_q = np.asarray(
                                    suffix_horizon_cache["anchor_q_rad"]
                                )
                                cached_distance = np.asarray(
                                    suffix_horizon_cache["distance_m"]
                                )
                                cached_q = np.asarray(
                                    suffix_horizon_cache["q_rad"]
                                )
                                if (
                                    abs(
                                        cached_anchor_distance
                                        - float(coarse_distance[keyframe - 1])
                                    )
                                    <= 1.0e-12
                                    and np.allclose(
                                        cached_anchor_q,
                                        previous_q,
                                        atol=1.0e-10,
                                        rtol=0.0,
                                    )
                                    and cached_distance.ndim == 1
                                    and cached_q.shape
                                    == (cached_distance.size, TOTAL_DOF)
                                    and cached_distance.size >= 1
                                ):
                                    cache_rows: list[np.ndarray] = []
                                    cache_knots_d = np.concatenate(
                                        (
                                            [cached_anchor_distance],
                                            cached_distance,
                                        )
                                    )
                                    cache_knots_q = np.vstack(
                                        (cached_anchor_q, cached_q)
                                    )
                                    for node_distance in horizon_distance:
                                        if node_distance <= cache_knots_d[-1] + 1.0e-12:
                                            cache_rows.append(
                                                smoothstep_joint_interpolation(
                                                    cache_knots_d,
                                                    cache_knots_q,
                                                    np.asarray([node_distance]),
                                                )[0]
                                            )
                                        else:
                                            cache_rows.append(cache_knots_q[-1])
                                    cache_seed = np.stack(cache_rows)
                                    append_suffix_seed(cache_seed, "certified_cache")
                            retained_seed_indices = prioritized_suffix_seed_indices(
                                suffix_seed_kinds,
                                maximum_seeds=6,
                            )
                            suffix_seeds = [
                                suffix_seeds[index]
                                for index in retained_seed_indices
                            ]
                            suffix_seed_kinds = [
                                suffix_seed_kinds[index]
                                for index in retained_seed_indices
                            ]

                            def audit_suffix(
                                q_rows: np.ndarray,
                            ) -> tuple[bool, tuple[float, ...], dict[str, object]]:
                                prior_q = previous_q
                                prior_arc = bridge_arc
                                prior_distance = float(
                                    coarse_distance[keyframe - 1]
                                )
                                minimum_task_slacks_m: list[float] = []
                                minimum_joint_slacks_rad: list[float] = []
                                minimum_pad_slacks: list[float] = []
                                failed_gate_count = 0
                                node_contact_counts: list[int] = []
                                node_motion_counts: list[int] = []
                                node_progress_margins: list[float] = []
                                node_normal_margins: list[float] = []
                                node_tangent_margins: list[float] = []
                                node_joint_margins: list[float] = []
                                node_collision_ok: list[bool] = []
                                node_condition_rows: list[np.ndarray] = []
                                node_metric_rows_m: list[np.ndarray] = []
                                node_metric_rows_rad: list[np.ndarray] = []
                                node_pad_margins: list[float] = []
                                node_self_counts: list[int] = []
                                node_interior_ok: list[bool] = []
                                for node_index, q_node in enumerate(q_rows):
                                    node_distance = float(
                                        horizon_distance[node_index]
                                    )
                                    (
                                        node_points,
                                        _,
                                        _,
                                        node_arc,
                                        node_aux,
                                    ) = contact_state(q_node)
                                    progress_error = np.abs(
                                        direction
                                        * (
                                            node_arc
                                            - target_arc_array[node_index]
                                        )
                                    )
                                    progress_error[0] = 0.0
                                    progress_margin = float(
                                        progress_limit_array[node_index]
                                        - np.max(progress_error)
                                    )
                                    normal_error = np.abs(
                                        node_aux[1:, 1]
                                        - target_standoff_array[
                                            node_index, 1:
                                        ]
                                    )
                                    nominal_contact = normal_error <= (
                                        args.mpc_normal_tolerance_mm
                                        / 1000.0
                                    )
                                    if (
                                        node_distance
                                        >= suffix_terminal_start_m - 1.0e-12
                                    ):
                                        normal_tolerance = np.full(
                                            4,
                                            args.mpc_normal_tolerance_mm
                                            / 1000.0,
                                            dtype=np.float64,
                                        )
                                        contact_ok = bool(
                                            np.all(nominal_contact)
                                        )
                                    else:
                                        normal_tolerance = (
                                            normal_tolerance_array[node_index]
                                        )
                                        contact_ok = bool(
                                            np.count_nonzero(nominal_contact)
                                            >= args.min_planner_contact_fingers
                                            and np.all(
                                                normal_error
                                                <= normal_tolerance
                                            )
                                        )
                                    normal_margin = float(
                                        np.min(
                                            normal_tolerance - normal_error
                                        )
                                    )
                                    tangent_error = np.abs(
                                        (
                                            (
                                                node_aux[1:, 0]
                                                - target_azimuth_array[
                                                    node_index, 1:
                                                ]
                                                + np.pi
                                            )
                                            % (2.0 * np.pi)
                                            - np.pi
                                        )
                                        * CAPSULE_RADIUS
                                    )
                                    tangent_margin = float(
                                        np.min(
                                            tangent_tolerance_array[node_index]
                                            - tangent_error
                                        )
                                    )
                                    achieved_progress = direction * (
                                        node_arc - start_arc
                                    )
                                    prior_progress = direction * (
                                        prior_arc - start_arc
                                    )
                                    monotonic_error = float(
                                        np.max(
                                            np.maximum(
                                                prior_progress
                                                - achieved_progress,
                                                0.0,
                                            )
                                        )
                                    )
                                    monotonic_margin = (
                                        args.mpc_monotonic_tolerance_mm
                                        / 1000.0
                                        - monotonic_error
                                    )
                                    approximate_palm_target = (
                                        palm_target
                                        + direction
                                        * (node_distance - desired_distance)
                                        * args.palm_travel_ratio
                                        * rotation[:, 2]
                                    )
                                    palm_error = float(
                                        np.linalg.norm(
                                            node_points[0]
                                            - approximate_palm_target
                                        )
                                    )
                                    joint_margin = float(
                                        np.min(
                                            np.minimum(
                                                q_node - lower,
                                                upper - q_node,
                                            )
                                        )
                                    )
                                    joint_step = float(
                                        np.max(np.abs(q_node - prior_q))
                                    )
                                    tip_motion = direction * (
                                        node_arc[1:] - prior_arc[1:]
                                    )
                                    minimum_motion = (
                                        args.mpc_feasibility_bridge_min_progress_ratio
                                        * (node_distance - prior_distance)
                                    )
                                    motion_ok, motion_count = (
                                        evaluate_moving_bridge_motion(
                                        max_joint_motion_rad=joint_step,
                                        tip_motion_m=tip_motion,
                                        minimum_tip_motion_m=minimum_motion,
                                        active_fingers=np.zeros(4, dtype=bool),
                                        )
                                    )
                                    collision_ok = True
                                    arm_margin = np.inf
                                    hand_margin = np.inf
                                    tip_margin = np.inf
                                    pad_margin = np.inf
                                    self_count = 0
                                    if args.collision_mode == "full_robot":
                                        (
                                            arm_clearance,
                                            _,
                                            hand_clearance,
                                            _,
                                            tip_clearance,
                                            _,
                                            self_count,
                                            _,
                                            _,
                                            _,
                                            pad_alignment,
                                        ) = segment_collision_status(
                                            q_node,
                                            segment_start_q=prior_q,
                                        )
                                        arm_margin = (
                                            arm_clearance
                                            - args.min_arm_clearance_mm
                                            / 1000.0
                                        )
                                        hand_margin = (
                                            hand_clearance
                                            + args.max_incidental_hand_penetration_mm
                                            / 1000.0
                                        )
                                        tip_margin = (
                                            tip_clearance
                                            + args.max_contact_penetration_mm
                                            / 1000.0
                                        )
                                        pad_margin = float(
                                            pad_alignment
                                            - planner_pad_alignment
                                        )
                                        collision_ok = bool(
                                            arm_margin >= 0.0
                                            and hand_margin >= 0.0
                                            and tip_margin >= 0.0
                                            and self_count == 0
                                            and pad_margin >= 0.0
                                        )
                                    node_hard_ok = bool(
                                        progress_margin >= 0.0
                                        and contact_ok
                                        and normal_margin >= 0.0
                                        and tangent_margin >= 0.0
                                        and monotonic_margin >= 0.0
                                        and palm_error
                                        <= palm_tracking_limit_m
                                        and joint_margin
                                        >= minimum_joint_margin_rad
                                        - 1.0e-12
                                        and joint_step
                                        <= args.max_plan_joint_step_rad
                                        + 1.0e-12
                                        and motion_ok
                                        and collision_ok
                                    )
                                    interior_ok = bool(
                                        progress_margin
                                        >= task_guard_m - 1.0e-12
                                        and normal_margin
                                        >= task_guard_m - 1.0e-12
                                        and tangent_margin
                                        >= task_guard_m - 1.0e-12
                                        and monotonic_margin
                                        >= task_guard_m - 1.0e-12
                                        and arm_margin
                                        >= task_guard_m - 1.0e-12
                                        and hand_margin
                                        >= task_guard_m - 1.0e-12
                                        and tip_margin
                                        >= task_guard_m - 1.0e-12
                                    )
                                    if not node_hard_ok:
                                        failed_gate_count += 1
                                    if not interior_ok:
                                        failed_gate_count += 1
                                    palm_margin = (
                                        palm_tracking_limit_m - palm_error
                                    )
                                    joint_target_margin = (
                                        joint_margin - minimum_joint_margin_rad
                                    )
                                    joint_step_margin = (
                                        args.max_plan_joint_step_rad
                                        - joint_step
                                    )
                                    minimum_task_slacks_m.extend(
                                        (
                                            progress_margin,
                                            normal_margin,
                                            tangent_margin,
                                            monotonic_margin,
                                            palm_margin,
                                            arm_margin,
                                            hand_margin,
                                            tip_margin,
                                        )
                                    )
                                    minimum_joint_slacks_rad.extend(
                                        (
                                            joint_target_margin,
                                            joint_step_margin,
                                        )
                                    )
                                    minimum_pad_slacks.append(pad_margin)
                                    node_contact_counts.append(
                                        int(np.count_nonzero(nominal_contact))
                                    )
                                    node_motion_counts.append(motion_count)
                                    node_progress_margins.append(progress_margin)
                                    node_normal_margins.append(normal_margin)
                                    node_tangent_margins.append(tangent_margin)
                                    node_joint_margins.append(joint_margin)
                                    node_collision_ok.append(collision_ok)
                                    node_condition_rows.append(
                                        np.asarray(
                                            (
                                                progress_margin >= 0.0,
                                                contact_ok,
                                                normal_margin >= 0.0,
                                                tangent_margin >= 0.0,
                                                monotonic_margin >= 0.0,
                                                palm_margin >= 0.0,
                                                joint_target_margin >= -1.0e-12,
                                                joint_step_margin >= -1.0e-12,
                                                motion_ok,
                                                collision_ok,
                                                interior_ok,
                                            ),
                                            dtype=np.bool_,
                                        )
                                    )
                                    node_metric_rows_m.append(
                                        np.asarray(
                                            (
                                                progress_margin,
                                                normal_margin,
                                                tangent_margin,
                                                monotonic_margin,
                                                palm_margin,
                                                arm_margin,
                                                hand_margin,
                                                tip_margin,
                                            ),
                                            dtype=np.float64,
                                        )
                                    )
                                    node_metric_rows_rad.append(
                                        np.asarray(
                                            (
                                                joint_target_margin,
                                                joint_step_margin,
                                            ),
                                            dtype=np.float64,
                                        )
                                    )
                                    node_pad_margins.append(pad_margin)
                                    node_self_counts.append(int(self_count))
                                    node_interior_ok.append(interior_ok)
                                    prior_q = q_node
                                    prior_arc = node_arc
                                    prior_distance = node_distance

                                anchor_distance = float(
                                    coarse_distance[keyframe - 1]
                                )
                                candidate_frame_distance = (
                                    planner_frame_target_distance[
                                        (planner_frame_target_distance
                                         > anchor_distance + 1.0e-12)
                                        & (planner_frame_target_distance
                                           <= horizon_distance[-1] + 1.0e-12)
                                    ]
                                )
                                frame_hard_ok = True
                                low_motion_ok = True
                                low_motion_failures: list[
                                    dict[str, object]
                                ] = []
                                publisher_gate_names = (
                                    "progress",
                                    "contact",
                                    "tangent",
                                    "palm",
                                    "collision",
                                    "monotonic",
                                )
                                publisher_first_failure_index = -1
                                publisher_first_failure_distance_m = np.nan
                                publisher_first_failure_gate_ok = np.ones(
                                    len(publisher_gate_names), dtype=np.bool_
                                )
                                if candidate_frame_distance.size:
                                    knot_distance = np.concatenate(
                                        ([anchor_distance], horizon_distance)
                                    )
                                    knot_q = np.vstack((previous_q, q_rows))
                                    candidate_frame_q = (
                                        smoothstep_joint_interpolation(
                                            knot_distance,
                                            knot_q,
                                            candidate_frame_distance,
                                        )
                                    )
                                    prefix_frame_distance = (
                                        planner_frame_target_distance[
                                            planner_frame_target_distance
                                            <= anchor_distance + 1.0e-12
                                        ][-LOW_MOTION_DEFAULT_WINDOW_FRAMES:]
                                    )
                                    if prefix_frame_distance.size:
                                        prefix_frame_q = (
                                            smoothstep_joint_interpolation(
                                                coarse_distance[:keyframe],
                                                coarse_q[:keyframe],
                                                prefix_frame_distance,
                                            )
                                        )
                                    else:
                                        prefix_frame_q = np.zeros(
                                            (0, TOTAL_DOF), dtype=np.float64
                                        )
                                    audit_distance = np.concatenate(
                                        (
                                            prefix_frame_distance,
                                            candidate_frame_distance,
                                        )
                                    )
                                    audit_q = np.vstack(
                                        (prefix_frame_q, candidate_frame_q)
                                    )
                                    audit_progress: list[np.ndarray] = []
                                    for sample_distance, sample_q in zip(
                                        audit_distance,
                                        audit_q,
                                        strict=True,
                                    ):
                                        (
                                            sample_points,
                                            _,
                                            sample_surface_normals,
                                            sample_arc,
                                            sample_aux,
                                        ) = contact_state(sample_q)
                                        sample_progress = direction * (
                                            sample_arc - start_arc
                                        )
                                        audit_progress.append(sample_progress)
                                        if sample_distance <= anchor_distance + 1.0e-12:
                                            continue
                                        (
                                            sample_target_arc,
                                            sample_target_azimuth,
                                            sample_target_standoff,
                                            sample_progress_limit,
                                            sample_normal_tolerance,
                                            sample_tangent_tolerance,
                                        ) = scheduled_fingertip_targets(
                                            float(sample_distance),
                                            bridge_rephase_offset_m,
                                        )
                                        sample_progress_error = np.abs(
                                            direction
                                            * (sample_arc - sample_target_arc)
                                        )
                                        sample_progress_error[0] = 0.0
                                        sample_normal_error = np.abs(
                                            sample_aux[1:, 1]
                                            - sample_target_standoff[1:]
                                        )
                                        sample_tangent_error = np.abs(
                                            (
                                                (
                                                    sample_aux[1:, 0]
                                                    - sample_target_azimuth[1:]
                                                    + np.pi
                                                )
                                                % (2.0 * np.pi)
                                                - np.pi
                                            )
                                            * CAPSULE_RADIUS
                                        )
                                        sample_palm_target = (
                                            palm_target
                                            + direction
                                            * (
                                                sample_distance
                                                - desired_distance
                                            )
                                            * args.palm_travel_ratio
                                            * rotation[:, 2]
                                        )
                                        sample_palm_ok = bool(
                                            np.linalg.norm(
                                                sample_points[0]
                                                - sample_palm_target
                                            )
                                            <= palm_tracking_limit_m
                                        )
                                        sample_collision_ok = True
                                        if args.collision_mode == "full_robot":
                                            (
                                                sample_tip_clearance,
                                                sample_arm_clearance,
                                                _,
                                                sample_hand_clearance,
                                                _,
                                            ) = reachability.geometry_group_clearances(
                                                sample_q,
                                                center,
                                                rotation,
                                            )
                                            sample_self_pairs, _ = (
                                                reachability.self_collision_contacts(
                                                    sample_q
                                                )
                                            )
                                            sample_pad_alignment = np.einsum(
                                                "ij,ij->i",
                                                reachability.fingertip_pad_normals(
                                                    sample_q
                                                ),
                                                -sample_surface_normals[1:],
                                            )
                                            sample_collision_ok = bool(
                                                float(
                                                    np.min(
                                                        sample_arm_clearance
                                                    )
                                                )
                                                >= args.min_arm_clearance_mm
                                                / 1000.0
                                                and float(
                                                    np.min(
                                                        sample_hand_clearance
                                                    )
                                                )
                                                >= -args.max_incidental_hand_penetration_mm
                                                / 1000.0
                                                and float(
                                                    np.min(
                                                        sample_tip_clearance
                                                    )
                                                )
                                                >= -args.max_contact_penetration_mm
                                                / 1000.0
                                                and not sample_self_pairs
                                                and float(
                                                    np.min(
                                                        sample_pad_alignment
                                                    )
                                                )
                                                >= planner_pad_alignment
                                            )
                                        if (
                                            sample_distance
                                            >= suffix_terminal_start_m - 1.0e-12
                                        ):
                                            sample_contact_ok = bool(
                                                np.all(
                                                    sample_normal_error
                                                    <= args.mpc_normal_tolerance_mm
                                                    / 1000.0
                                                )
                                            )
                                        else:
                                            sample_contact_ok = bool(
                                                np.count_nonzero(
                                                    sample_normal_error
                                                    <= args.mpc_normal_tolerance_mm
                                                    / 1000.0
                                                )
                                                >= args.min_planner_contact_fingers
                                                and np.all(
                                                    sample_normal_error
                                                    <= sample_normal_tolerance
                                                )
                                            )
                                        sample_gate_ok = np.asarray(
                                            (
                                                float(
                                                    np.max(
                                                        sample_progress_error
                                                    )
                                                )
                                                <= sample_progress_limit
                                                + 1.0e-12,
                                                sample_contact_ok,
                                                bool(
                                                    np.all(
                                                        sample_tangent_error
                                                        <= sample_tangent_tolerance
                                                        + 1.0e-12
                                                    )
                                                ),
                                                sample_palm_ok,
                                                sample_collision_ok,
                                                True,
                                            ),
                                            dtype=np.bool_,
                                        )
                                        if not bool(np.all(sample_gate_ok)):
                                            frame_hard_ok = False
                                            if publisher_first_failure_index < 0:
                                                publisher_first_failure_index = (
                                                    len(audit_progress) - 1
                                                )
                                                publisher_first_failure_distance_m = float(
                                                    sample_distance
                                                )
                                                publisher_first_failure_gate_ok = (
                                                    sample_gate_ok.copy()
                                                )
                                    audit_progress_array = np.stack(
                                        audit_progress
                                    )
                                    if audit_progress_array.shape[0] > 1:
                                        published_backtrack = np.maximum(
                                            audit_progress_array[:-1]
                                            - audit_progress_array[1:],
                                            0.0,
                                        )
                                        published_backtrack[:, 0] = 0.0
                                        if float(
                                            np.max(published_backtrack)
                                        ) > (
                                            args.mpc_monotonic_tolerance_mm
                                            / 1000.0
                                            + 1.0e-12
                                        ):
                                            frame_hard_ok = False
                                            if publisher_first_failure_index < 0:
                                                worst_backtrack = int(
                                                    np.argmax(
                                                        np.max(
                                                            published_backtrack,
                                                            axis=1,
                                                        )
                                                    )
                                                )
                                                publisher_first_failure_index = (
                                                    worst_backtrack + 1
                                                )
                                                publisher_first_failure_distance_m = float(
                                                    audit_distance[
                                                        worst_backtrack + 1
                                                    ]
                                                )
                                                publisher_first_failure_gate_ok[
                                                    -1
                                                ] = False
                                    low_motion_failures = (
                                        prospective_low_motion_failures(
                                            horizon_distance,
                                            q_rows,
                                            candidate_keyframe=keyframe,
                                        )
                                    )
                                    low_motion_ok = not bool(
                                        low_motion_failures
                                    )
                                if not frame_hard_ok:
                                    failed_gate_count += 1
                                if not low_motion_ok:
                                    failed_gate_count += 1
                                minimum_task_slack = float(
                                    np.min(minimum_task_slacks_m)
                                )
                                minimum_joint_slack = float(
                                    np.min(minimum_joint_slacks_rad)
                                )
                                minimum_pad_slack = float(
                                    np.min(minimum_pad_slacks)
                                )
                                passed = bool(
                                    failed_gate_count == 0
                                    and frame_hard_ok
                                    and low_motion_ok
                                )
                                rank = (
                                    0.0 if passed else 1.0,
                                    float(failed_gate_count),
                                    -minimum_task_slack,
                                    -minimum_joint_slack,
                                    -minimum_pad_slack,
                                )
                                details: dict[str, object] = {
                                    "passed": passed,
                                    "minimum_slack_m": minimum_task_slack,
                                    "minimum_task_slack_m": minimum_task_slack,
                                    "minimum_joint_slack_rad": (
                                        minimum_joint_slack
                                    ),
                                    "minimum_pad_alignment_slack": (
                                        minimum_pad_slack
                                    ),
                                    "failed_gate_count": failed_gate_count,
                                    "node_condition_names": np.asarray(
                                        (
                                            "progress",
                                            "contact",
                                            "normal",
                                            "tangent",
                                            "monotonic",
                                            "palm",
                                            "joint_margin",
                                            "joint_step",
                                            "motion",
                                            "collision",
                                            "interior",
                                        )
                                    ),
                                    "node_condition_ok": np.stack(
                                        node_condition_rows
                                    ),
                                    "node_metric_names_m": np.asarray(
                                        (
                                            "progress",
                                            "normal",
                                            "tangent",
                                            "monotonic",
                                            "palm",
                                            "arm_clearance",
                                            "hand_clearance",
                                            "tip_clearance",
                                        )
                                    ),
                                    "node_metric_margin_m": np.stack(
                                        node_metric_rows_m
                                    ),
                                    "node_metric_names_rad": np.asarray(
                                        ("joint_margin", "joint_step")
                                    ),
                                    "node_metric_margin_rad": np.stack(
                                        node_metric_rows_rad
                                    ),
                                    "node_contact_count": np.asarray(
                                        node_contact_counts, dtype=np.int8
                                    ),
                                    "node_progress_margin_m": np.asarray(
                                        node_progress_margins
                                    ),
                                    "node_normal_margin_m": np.asarray(
                                        node_normal_margins
                                    ),
                                    "node_tangent_margin_m": np.asarray(
                                        node_tangent_margins
                                    ),
                                    "node_joint_margin_rad": np.asarray(
                                        node_joint_margins
                                    ),
                                    "node_collision_ok": np.asarray(
                                        node_collision_ok, dtype=bool
                                    ),
                                    "node_motion_count": np.asarray(
                                        node_motion_counts, dtype=np.int8
                                    ),
                                    "node_self_contact_count": np.asarray(
                                        node_self_counts, dtype=np.int16
                                    ),
                                    "node_pad_alignment_margin": np.asarray(
                                        node_pad_margins, dtype=np.float64
                                    ),
                                    "node_interior_ok": np.asarray(
                                        node_interior_ok, dtype=np.bool_
                                    ),
                                    "publisher_hard_ok": frame_hard_ok,
                                    "publisher_gate_names": np.asarray(
                                        publisher_gate_names
                                    ),
                                    "publisher_first_failure_index": (
                                        publisher_first_failure_index
                                    ),
                                    "publisher_first_failure_distance_m": (
                                        publisher_first_failure_distance_m
                                    ),
                                    "publisher_first_failure_gate_ok": (
                                        publisher_first_failure_gate_ok
                                    ),
                                    "low_motion_ok": low_motion_ok,
                                    "low_motion_first_window_start": (
                                        int(
                                            low_motion_failures[0][
                                                "first_window"
                                            ]["start"]
                                        )
                                        if low_motion_failures
                                        else -1
                                    ),
                                    "low_motion_first_window_end": (
                                        int(
                                            low_motion_failures[0][
                                                "first_window"
                                            ]["end"]
                                        )
                                        if low_motion_failures
                                        else -1
                                    ),
                                }
                                return passed, rank, details

                            horizon_candidates: list[SimpleNamespace] = []
                            for seed_index, (suffix_seed, suffix_seed_kind) in enumerate(
                                zip(
                                    suffix_seeds,
                                    suffix_seed_kinds,
                                    strict=True,
                                )
                            ):
                                result = least_squares(
                                    suffix_residual,
                                    suffix_seed.ravel(),
                                    bounds=(flat_lower, flat_upper),
                                    max_nfev=args.mpc_suffix_max_nfev,
                                    xtol=1.0e-9,
                                    ftol=1.0e-9,
                                    gtol=1.0e-9,
                                    x_scale="jac",
                                    diff_step=1.0e-5,
                                )
                                q_rows = result.x.reshape(
                                    node_count, TOTAL_DOF
                                )
                                passed, audit_rank, audit_details = (
                                    audit_suffix(q_rows)
                                )
                                horizon_candidates.append(
                                    SimpleNamespace(
                                        x=q_rows[0].copy(),
                                        cost=float(result.cost),
                                        nfev=int(result.nfev),
                                        candidate_kind="suffix_horizon",
                                        suffix_seed_kind=suffix_seed_kind,
                                        suffix_q_rad=q_rows.copy(),
                                        suffix_distance_m=(
                                            horizon_distance.copy()
                                        ),
                                        suffix_audit=audit_details,
                                        suffix_passed=passed,
                                        suffix_rank=(
                                            *audit_rank,
                                            float(result.cost),
                                            float(seed_index),
                                        ),
                                    )
                                )
                            if horizon_candidates and not any(
                                candidate.suffix_passed
                                for candidate in horizon_candidates
                            ):
                                # The block solve is a basin finder.  Repair its
                                # best basins one node at a time with bounds that
                                # make the formal 0.03-rad transition limit
                                # impossible to violate.  Every repaired prefix
                                # is exact-audited before it may seed the next
                                # node; no planner state is mutated here.
                                ranked_rollout_source_indices = sorted(
                                    range(len(horizon_candidates)),
                                    key=lambda index: horizon_candidates[
                                        index
                                    ].suffix_rank,
                                )
                                rollout_source_indices = (
                                    prioritized_suffix_rollout_indices(
                                        tuple(
                                            candidate.suffix_seed_kind
                                            for candidate in horizon_candidates
                                        ),
                                        ranked_rollout_source_indices,
                                        maximum_sources=3,
                                    )
                                )
                                for source_index in rollout_source_indices:
                                    source = horizon_candidates[source_index]
                                    rollout_rows = source.suffix_q_rad.copy()
                                    rollout_nfev = 0
                                    rollout_cost = 0.0
                                    rollout_ok = True
                                    rollout_reached_node = -1
                                    rollout_prune_node = -1
                                    rollout_prune_reason = "completed"
                                    rollout_attempt_count = 0
                                    rollout_prior_q = previous_q.copy()
                                    rollout_prior_arc = bridge_arc.copy()
                                    rollout_prior_delta = previous_delta.copy()
                                    rollout_prior_distance = float(
                                        coarse_distance[keyframe - 1]
                                    )
                                    rollout_audit_details = source.suffix_audit
                                    rollout_audit_rank = source.suffix_rank[:5]
                                    for node_index in range(node_count):
                                        step_inner_limit = (
                                            args.max_plan_joint_step_rad
                                            - 5.0e-5
                                        )
                                        local_lower = np.maximum(
                                            node_lower,
                                            rollout_prior_q - step_inner_limit,
                                        )
                                        local_upper = np.minimum(
                                            node_upper,
                                            rollout_prior_q + step_inner_limit,
                                        )
                                        if np.any(
                                            local_lower
                                            >= local_upper - 1.0e-12
                                        ):
                                            rollout_ok = False
                                            rollout_prune_node = node_index
                                            rollout_prune_reason = "invalid_bounds"
                                            (
                                                _,
                                                rollout_audit_rank,
                                                rollout_audit_details,
                                            ) = audit_suffix(rollout_rows)
                                            break
                                        local_seed = np.clip(
                                            rollout_rows[node_index],
                                            local_lower + 1.0e-10,
                                            local_upper - 1.0e-10,
                                        )

                                        def rollout_node_residual(
                                            q_node: np.ndarray,
                                            *,
                                            _node_index: int = node_index,
                                            _prior_q: np.ndarray = (
                                                rollout_prior_q.copy()
                                            ),
                                            _prior_arc: np.ndarray = (
                                                rollout_prior_arc.copy()
                                            ),
                                            _prior_distance: float = (
                                                rollout_prior_distance
                                            ),
                                        ) -> np.ndarray:
                                            return suffix_node_residual(
                                                q_node,
                                                node_index=_node_index,
                                                prior_q=_prior_q,
                                                prior_arc=_prior_arc,
                                                prior_distance=(
                                                    _prior_distance
                                                ),
                                                feasibility_weight_scale=4.0,
                                            )

                                        node_trials: list[SimpleNamespace] = []

                                        def append_rollout_node_trial(
                                            q_node: np.ndarray,
                                            *,
                                            trial_kind: str,
                                            trial_cost: float,
                                            trial_nfev: int,
                                            trial_order: int,
                                        ) -> None:
                                            nonlocal rollout_attempt_count
                                            trial_rows = rollout_rows.copy()
                                            trial_rows[node_index] = q_node
                                            (
                                                _,
                                                _,
                                                trial_audit,
                                            ) = audit_suffix(trial_rows)
                                            (
                                                trial_prefix_ok,
                                                trial_prefix_rank,
                                            ) = suffix_rollout_prefix_rank(
                                                node_condition_ok=(
                                                    trial_audit[
                                                        "node_condition_ok"
                                                    ]
                                                ),
                                                node_metric_margin_m=(
                                                    trial_audit[
                                                        "node_metric_margin_m"
                                                    ]
                                                ),
                                                node_metric_margin_rad=(
                                                    trial_audit[
                                                        "node_metric_margin_rad"
                                                    ]
                                                ),
                                                node_pad_alignment_margin=(
                                                    trial_audit[
                                                        "node_pad_alignment_margin"
                                                    ]
                                                ),
                                                node_index=node_index,
                                                publisher_first_failure_distance_m=float(
                                                    trial_audit[
                                                        "publisher_first_failure_distance_m"
                                                    ]
                                                ),
                                                node_distance_m=float(
                                                    horizon_distance[node_index]
                                                ),
                                            )
                                            rollout_attempt_count += 1
                                            node_trials.append(
                                                SimpleNamespace(
                                                    q_rad=q_node.copy(),
                                                    rows=trial_rows,
                                                    audit=trial_audit,
                                                    prefix_ok=trial_prefix_ok,
                                                    rank=(
                                                        *trial_prefix_rank,
                                                        float(trial_cost),
                                                        float(trial_order),
                                                    ),
                                                    kind=trial_kind,
                                                    cost=float(trial_cost),
                                                    nfev=int(trial_nfev),
                                                )
                                            )

                                        # Preserve an already certified block-prefix
                                        # row. Re-solving it can move a safe node
                                        # into a different, invalid local basin.
                                        append_rollout_node_trial(
                                            local_seed,
                                            trial_kind="source_preserved",
                                            trial_cost=0.0,
                                            trial_nfev=0,
                                            trial_order=0,
                                        )
                                        source_prefix_ok = bool(
                                            node_trials[0].prefix_ok
                                        )
                                        if not source_prefix_ok:
                                            extrapolated_seed = np.clip(
                                                rollout_prior_q
                                                + rollout_prior_delta,
                                                local_lower + 1.0e-10,
                                                local_upper - 1.0e-10,
                                            )
                                            rollout_seed_specs = [
                                                ("source_ls", local_seed)
                                            ]
                                            if not np.allclose(
                                                extrapolated_seed,
                                                local_seed,
                                                atol=1.0e-12,
                                                rtol=0.0,
                                            ):
                                                rollout_seed_specs.append(
                                                    (
                                                        "extrapolated_ls",
                                                        extrapolated_seed,
                                                    )
                                                )
                                            for (
                                                trial_order,
                                                (
                                                    rollout_seed_kind,
                                                    rollout_seed,
                                                ),
                                            ) in enumerate(
                                                rollout_seed_specs,
                                                start=1,
                                            ):
                                                local_result = least_squares(
                                                    rollout_node_residual,
                                                    rollout_seed,
                                                    bounds=(
                                                        local_lower,
                                                        local_upper,
                                                    ),
                                                    max_nfev=min(
                                                        args.mpc_suffix_max_nfev,
                                                        100,
                                                    ),
                                                    xtol=1.0e-9,
                                                    ftol=1.0e-9,
                                                    gtol=1.0e-9,
                                                    x_scale="jac",
                                                    diff_step=1.0e-5,
                                                )
                                                rollout_nfev += int(
                                                    local_result.nfev
                                                )
                                                append_rollout_node_trial(
                                                    local_result.x,
                                                    trial_kind=(
                                                        rollout_seed_kind
                                                    ),
                                                    trial_cost=float(
                                                        local_result.cost
                                                    ),
                                                    trial_nfev=int(
                                                        local_result.nfev
                                                    ),
                                                    trial_order=trial_order,
                                                )

                                        selected_node_trial = min(
                                            node_trials,
                                            key=lambda trial: trial.rank,
                                        )
                                        rollout_rows = (
                                            selected_node_trial.rows.copy()
                                        )
                                        rollout_audit_details = (
                                            selected_node_trial.audit
                                        )
                                        rollout_cost += float(
                                            selected_node_trial.cost
                                        )
                                        if not selected_node_trial.prefix_ok:
                                            rollout_ok = False
                                            rollout_prune_node = node_index
                                            rollout_prune_reason = (
                                                "prefix_gates"
                                            )
                                            break

                                        rollout_reached_node = node_index
                                        rollout_prior_delta = (
                                            selected_node_trial.q_rad
                                            - rollout_prior_q
                                        )
                                        rollout_prior_q = (
                                            selected_node_trial.q_rad.copy()
                                        )
                                        rollout_prior_arc = contact_state(
                                            rollout_prior_q
                                        )[3]
                                        rollout_prior_distance = float(
                                            horizon_distance[node_index]
                                        )
                                    if not rollout_ok:
                                        partial_seed_kind = (
                                            "rollout_partial_"
                                            + source.suffix_seed_kind
                                        )
                                        partial_rank = audit_suffix(
                                            rollout_rows
                                        )[1]
                                        horizon_candidates.append(
                                            SimpleNamespace(
                                                x=rollout_rows[0].copy(),
                                                cost=rollout_cost,
                                                nfev=rollout_nfev,
                                                candidate_kind=(
                                                    "suffix_horizon"
                                                ),
                                                suffix_seed_kind=(
                                                    partial_seed_kind
                                                ),
                                                suffix_q_rad=(
                                                    rollout_rows.copy()
                                                ),
                                                suffix_distance_m=(
                                                    horizon_distance.copy()
                                                ),
                                                suffix_audit=(
                                                    rollout_audit_details
                                                ),
                                                suffix_passed=False,
                                                suffix_rank=(
                                                    *partial_rank,
                                                    rollout_cost,
                                                    float(source_index),
                                                ),
                                                rollout_reached_node=(
                                                    rollout_reached_node
                                                ),
                                                rollout_prune_node=(
                                                    rollout_prune_node
                                                ),
                                                rollout_prune_reason=(
                                                    rollout_prune_reason
                                                ),
                                                rollout_attempt_count=(
                                                    rollout_attempt_count
                                                ),
                                            )
                                        )
                                        first_failure_distance = float(
                                            rollout_audit_details[
                                                "publisher_first_failure_distance_m"
                                            ]
                                        )
                                        print(
                                            "[SUFFIX-ROLLOUT-PRUNED] "
                                            f"source={source.suffix_seed_kind} "
                                            f"reached_node={rollout_reached_node} "
                                            f"prune_node={rollout_prune_node} "
                                            f"reason={rollout_prune_reason} "
                                            "publisher_first_failure_m="
                                            f"{first_failure_distance:.9f}",
                                            flush=True,
                                        )
                                        continue
                                    (
                                        rollout_passed,
                                        rollout_audit_rank,
                                        rollout_audit_details,
                                    ) = audit_suffix(rollout_rows)
                                    horizon_candidates.append(
                                        SimpleNamespace(
                                            x=rollout_rows[0].copy(),
                                            cost=rollout_cost,
                                            nfev=rollout_nfev,
                                            candidate_kind=(
                                                "suffix_horizon"
                                            ),
                                            suffix_seed_kind=(
                                                "rollout_"
                                                + source.suffix_seed_kind
                                            ),
                                            suffix_q_rad=(
                                                rollout_rows.copy()
                                            ),
                                            suffix_distance_m=(
                                                horizon_distance.copy()
                                            ),
                                            suffix_audit=(
                                                rollout_audit_details
                                            ),
                                            suffix_passed=(
                                                rollout_passed
                                            ),
                                            suffix_rank=(
                                                *rollout_audit_rank,
                                                rollout_cost,
                                                float(source_index),
                                            ),
                                            rollout_reached_node=(
                                                node_count - 1
                                            ),
                                            rollout_prune_node=-1,
                                            rollout_prune_reason=(
                                                "completed"
                                            ),
                                            rollout_attempt_count=(
                                                rollout_attempt_count
                                            ),
                                        )
                                    )
                            if not horizon_candidates:
                                last_suffix_horizon_evidence["status"] = (
                                    np.asarray("no_candidates")
                                )
                                return None
                            selected_index = min(
                                range(len(horizon_candidates)),
                                key=lambda candidate_index: (
                                    horizon_candidates[
                                        candidate_index
                                    ].suffix_rank
                                ),
                            )
                            selected = horizon_candidates[selected_index]
                            last_suffix_horizon_evidence = {
                                "status": np.asarray("completed"),
                                "invocation_distance_m": np.asarray(
                                    desired_distance, dtype=np.float64
                                ),
                                "anchor_distance_m": np.asarray(
                                    coarse_distance[keyframe - 1],
                                    dtype=np.float64,
                                ),
                                "terminal_start_m": np.asarray(
                                    suffix_terminal_start_m,
                                    dtype=np.float64,
                                ),
                                "node_distance_m": horizon_distance.copy(),
                                "seed_kind": np.asarray(
                                    [
                                        candidate.suffix_seed_kind
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_q_rad": np.stack(
                                    [
                                        candidate.suffix_q_rad
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_cost": np.asarray(
                                    [
                                        candidate.cost
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.float64,
                                ),
                                "candidate_nfev": np.asarray(
                                    [
                                        candidate.nfev
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int32,
                                ),
                                "candidate_rollout_reached_node": np.asarray(
                                    [
                                        getattr(
                                            candidate,
                                            "rollout_reached_node",
                                            -1,
                                        )
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int16,
                                ),
                                "candidate_rollout_prune_node": np.asarray(
                                    [
                                        getattr(
                                            candidate,
                                            "rollout_prune_node",
                                            -1,
                                        )
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int16,
                                ),
                                "candidate_rollout_prune_reason": np.asarray(
                                    [
                                        getattr(
                                            candidate,
                                            "rollout_prune_reason",
                                            "not_rollout",
                                        )
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_rollout_attempt_count": np.asarray(
                                    [
                                        getattr(
                                            candidate,
                                            "rollout_attempt_count",
                                            0,
                                        )
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int16,
                                ),
                                "candidate_passed": np.asarray(
                                    [
                                        candidate.suffix_passed
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.bool_,
                                ),
                                "candidate_failed_gate_count": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "failed_gate_count"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int16,
                                ),
                                "candidate_minimum_task_slack_m": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "minimum_task_slack_m"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.float64,
                                ),
                                "candidate_minimum_joint_slack_rad": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "minimum_joint_slack_rad"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.float64,
                                ),
                                "candidate_minimum_pad_alignment_slack": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "minimum_pad_alignment_slack"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.float64,
                                ),
                                "node_condition_names": np.asarray(
                                    selected.suffix_audit[
                                        "node_condition_names"
                                    ]
                                ),
                                "candidate_node_condition_ok": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_condition_ok"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "node_metric_names_m": np.asarray(
                                    selected.suffix_audit[
                                        "node_metric_names_m"
                                    ]
                                ),
                                "candidate_node_metric_margin_m": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_metric_margin_m"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "node_metric_names_rad": np.asarray(
                                    selected.suffix_audit[
                                        "node_metric_names_rad"
                                    ]
                                ),
                                "candidate_node_metric_margin_rad": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_metric_margin_rad"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_node_contact_count": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_contact_count"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_node_motion_count": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_motion_count"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_node_collision_ok": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_collision_ok"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_node_self_contact_count": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_self_contact_count"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_node_pad_alignment_margin": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "node_pad_alignment_margin"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "publisher_gate_names": np.asarray(
                                    selected.suffix_audit[
                                        "publisher_gate_names"
                                    ]
                                ),
                                "candidate_publisher_hard_ok": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "publisher_hard_ok"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.bool_,
                                ),
                                "candidate_publisher_first_failure_index": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "publisher_first_failure_index"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int32,
                                ),
                                "candidate_publisher_first_failure_distance_m": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "publisher_first_failure_distance_m"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.float64,
                                ),
                                "candidate_publisher_first_failure_gate_ok": np.stack(
                                    [
                                        candidate.suffix_audit[
                                            "publisher_first_failure_gate_ok"
                                        ]
                                        for candidate in horizon_candidates
                                    ]
                                ),
                                "candidate_low_motion_ok": np.asarray(
                                    [
                                        candidate.suffix_audit[
                                            "low_motion_ok"
                                        ]
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.bool_,
                                ),
                                "candidate_low_motion_first_window": np.asarray(
                                    [
                                        (
                                            candidate.suffix_audit[
                                                "low_motion_first_window_start"
                                            ],
                                            candidate.suffix_audit[
                                                "low_motion_first_window_end"
                                            ],
                                        )
                                        for candidate in horizon_candidates
                                    ],
                                    dtype=np.int32,
                                ),
                                "selected_index": np.asarray(
                                    selected_index, dtype=np.int32
                                ),
                                "selected_passed": np.asarray(
                                    selected.suffix_passed, dtype=np.bool_
                                ),
                            }
                            print(
                                "[SUFFIX-HORIZON] "
                                f"distance_m={desired_distance:.9f} "
                                f"nodes={node_count} attempts={len(horizon_candidates)} "
                                f"passed={selected.suffix_passed} "
                                "minimum_slack_mm="
                                f"{float(selected.suffix_audit['minimum_task_slack_m']) * 1000:.6f} "
                                "minimum_joint_slack_mrad="
                                f"{float(selected.suffix_audit['minimum_joint_slack_rad']) * 1000:.6f} "
                                "failed_gates="
                                f"{int(selected.suffix_audit['failed_gate_count'])}",
                                flush=True,
                            )
                            return selected if selected.suffix_passed else None

                        def moving_bridge_residual(
                            q: np.ndarray,
                        ) -> np.ndarray:
                            _, _, _, moving_arc, moving_aux = contact_state(q)
                            bridge_base_residual = moving_bridge_local_residual(
                                arc_m=moving_arc[1:],
                                target_arc_m=moving_bridge_target_arc,
                                standoff_m=moving_aux[1:, 1],
                                anchor_standoff_m=(
                                    bridge_anchor_standoff_m
                                ),
                                azimuth_rad=moving_aux[1:, 0],
                                anchor_azimuth_rad=(
                                    bridge_anchor_azimuth_rad
                                ),
                                q_rad=q,
                                anchor_q_rad=previous_q,
                                capsule_radius_m=CAPSULE_RADIUS,
                                task_weight=(
                                    args.mpc_feasibility_bridge_target_weight
                                ),
                            )
                            bridge_self_residual = np.zeros(
                                len(protected_self_pairs),
                                dtype=np.float64,
                            )
                            if args.collision_mode == "full_robot":
                                bridge_self_residual = (
                                    positive_self_clearance_residual(
                                        reachability.geometry_pair_distances(
                                            q,
                                            protected_self_pairs,
                                        ),
                                        target_clearance_m=(
                                            planner_protected_self_clearance_m
                                        ),
                                    )
                                )
                            bridge_tip_residual = np.zeros(
                                8,
                                dtype=np.float64,
                            )
                            if args.collision_mode == "full_robot":
                                bridge_tip_clearance = (
                                    reachability.geometry_group_clearances(
                                        q,
                                        center,
                                        rotation,
                                    )[0]
                                )
                                bridge_tip_residual = (
                                    moving_bridge_tip_geometry_residual(
                                        bridge_tip_clearance,
                                        planner_tip_geom_target_m,
                                        inner_cap_m=(
                                            planner_tip_geom_inner_cap_m
                                        ),
                                        target_weight=(
                                            args.planner_tip_geom_weight
                                        ),
                                        target_scale=(
                                            args.mpc_feasibility_bridge_tip_target_scale
                                        ),
                                        inner_weight=(
                                            args.planner_tip_geom_inner_weight
                                        ),
                                    )
                                )
                            return np.concatenate(
                                (
                                    bridge_base_residual,
                                    args.planner_protected_self_clearance_weight
                                    * bridge_self_residual,
                                    bridge_tip_residual,
                                )
                            )

                        def moving_bridge_multistart_rank(
                            candidate: object,
                        ) -> tuple[float, ...]:
                            """Run every immutable hard gate for one seed."""

                            (
                                candidate_points,
                                _,
                                _,
                                candidate_arc,
                                candidate_aux,
                            ) = contact_state(candidate.x)
                            candidate_progress_error = np.abs(
                                direction
                                * (candidate_arc - bridge_desired_arc)
                            )
                            candidate_progress_error[0] = 0.0
                            candidate_normal_error = np.abs(
                                candidate_aux[:, 1] - desired_standoff
                            )
                            candidate_tangential_error = (
                                (
                                    candidate_aux[:, 0]
                                    - desired_azimuth
                                    + np.pi
                                )
                                % (2.0 * np.pi)
                                - np.pi
                            ) * CAPSULE_RADIUS
                            candidate_progress = direction * (
                                candidate_arc - start_arc
                            )
                            candidate_monotonic_error = np.maximum(
                                minimum_progress - candidate_progress,
                                0.0,
                            )
                            candidate_monotonic_error[0] = 0.0
                            (
                                candidate_normal_ok,
                                _,
                                _,
                            ) = scheduled_contact_status(
                                candidate_normal_error[1:],
                                desired_distance,
                            )
                            (
                                candidate_recovery_normal_ok,
                                _,
                                _,
                            ) = recovery_contact_status(
                                candidate_normal_error[1:]
                            )
                            candidate_palm_error = float(
                                np.linalg.norm(
                                    candidate_points[0] - palm_target
                                )
                            )
                            candidate_collision_ok = True
                            candidate_tip_clearance = (
                                planner_tip_geom_inner_cap_m
                            )
                            candidate_protected_clearance = (
                                planner_protected_self_clearance_m
                            )
                            candidate_pad_alignment = 1.0
                            if args.collision_mode == "full_robot":
                                (
                                    candidate_arm_clearance,
                                    _,
                                    candidate_hand_clearance,
                                    _,
                                    candidate_tip_clearance,
                                    _,
                                    candidate_self_count,
                                    _,
                                    candidate_protected_clearance,
                                    _,
                                    candidate_pad_alignment,
                                ) = segment_collision_status(candidate.x)
                                candidate_collision_ok = bool(
                                    candidate_arm_clearance
                                    >= args.min_arm_clearance_mm / 1000.0
                                    and candidate_hand_clearance
                                    >= -args.max_incidental_hand_penetration_mm
                                    / 1000.0
                                    and candidate_tip_clearance
                                    >= -args.max_contact_penetration_mm
                                    / 1000.0
                                    and candidate_self_count == 0
                                    and candidate_pad_alignment
                                    >= planner_pad_alignment
                                )
                            candidate_joint_motion = float(
                                np.max(np.abs(candidate.x - previous_q))
                            )
                            candidate_joint_ok = bool(
                                np.all(candidate.x >= lower - 1.0e-12)
                                and np.all(candidate.x <= upper + 1.0e-12)
                                and candidate_joint_motion
                                <= args.max_plan_joint_step_rad + 1.0e-12
                            )
                            candidate_tip_motion = direction * (
                                candidate_arc[1:] - bridge_arc[1:]
                            )
                            candidate_minimum_motion = (
                                args.mpc_feasibility_bridge_min_progress_ratio
                                * bridge_interval_m
                            )
                            candidate_motion_ok, _ = (
                                evaluate_moving_bridge_motion(
                                    max_joint_motion_rad=(
                                        candidate_joint_motion
                                    ),
                                    tip_motion_m=candidate_tip_motion,
                                    minimum_tip_motion_m=(
                                        candidate_minimum_motion
                                    ),
                                    active_fingers=(
                                        np.abs(bridge_offset_delta_m)
                                        > 1.0e-12
                                    ),
                                )
                            )
                            candidate_strict = evaluate_bridge_conditions(
                                progress_error_m=(
                                    candidate_progress_error
                                ),
                                progress_limit_m=progress_limit_m,
                                normal_ok=candidate_normal_ok,
                                tangential_error_m=(
                                    candidate_tangential_error[1:]
                                ),
                                tangential_limit_m=(
                                    tip_tangential_tolerances
                                ),
                                monotonic_error_m=(
                                    candidate_monotonic_error
                                ),
                                monotonic_limit_m=(
                                    args.mpc_monotonic_tolerance_mm
                                    / 1000.0
                                ),
                                palm_error_m=candidate_palm_error,
                                palm_limit_m=palm_tracking_limit_m,
                                collision_ok=candidate_collision_ok,
                                joint_ok=candidate_joint_ok,
                                motion_ok=candidate_motion_ok,
                                budget_ok=True,
                            )
                            candidate_recovery = evaluate_bridge_conditions(
                                progress_error_m=(
                                    candidate_progress_error
                                ),
                                progress_limit_m=max(
                                    progress_limit_m,
                                    args.mpc_recovery_bridge_progress_tolerance_mm
                                    / 1000.0,
                                ),
                                normal_ok=(
                                    candidate_recovery_normal_ok
                                ),
                                tangential_error_m=(
                                    candidate_tangential_error[1:]
                                ),
                                tangential_limit_m=(
                                    tip_tangential_tolerances
                                ),
                                monotonic_error_m=(
                                    candidate_monotonic_error
                                ),
                                monotonic_limit_m=(
                                    args.mpc_monotonic_tolerance_mm
                                    / 1000.0
                                ),
                                palm_error_m=candidate_palm_error,
                                palm_limit_m=palm_tracking_limit_m,
                                collision_ok=candidate_collision_ok,
                                joint_ok=candidate_joint_ok,
                                motion_ok=candidate_motion_ok,
                                budget_ok=recovery_bridge_budget_ok,
                            )
                            strict_ok = all(candidate_strict.values())
                            recovery_ok = bool(
                                not strict_ok
                                and all(candidate_recovery.values())
                            )
                            failed_condition_count = min(
                                sum(
                                    not passed
                                    for passed in candidate_strict.values()
                                ),
                                sum(
                                    not passed
                                    for passed in candidate_recovery.values()
                                ),
                            )
                            task_error_score = float(
                                np.max(candidate_progress_error)
                                + np.max(candidate_normal_error[1:])
                                + np.max(
                                    np.abs(
                                        candidate_tangential_error[1:]
                                    )
                                )
                                + np.max(candidate_monotonic_error)
                                + candidate_palm_error
                            )
                            return moving_bridge_candidate_rank(
                                strict_hard_feasible=strict_ok,
                                recovery_hard_feasible=recovery_ok,
                                collision_hard_feasible=(
                                    candidate_collision_ok
                                ),
                                failed_condition_count=(
                                    failed_condition_count
                                ),
                                minimum_tip_clearance_m=(
                                    candidate_tip_clearance
                                ),
                                tip_inner_cap_m=(
                                    planner_tip_geom_inner_cap_m
                                ),
                                minimum_protected_self_clearance_m=(
                                    candidate_protected_clearance
                                ),
                                soft_self_clearance_target_m=(
                                    planner_protected_self_clearance_m
                                ),
                                minimum_pad_alignment=(
                                    candidate_pad_alignment
                                ),
                                soft_pad_alignment=(
                                    planner_soft_pad_alignment
                                ),
                                task_error_score=task_error_score,
                                continuity_error=candidate_joint_motion,
                                solver_cost=float(candidate.cost),
                            )

                        moving_bridge_seed = np.minimum(
                            np.maximum(previous_q, bridge_lower),
                            bridge_upper,
                        )
                        moving_separation_seeds = (
                            protected_self_separation_seeds(
                                moving_bridge_seed,
                                bridge_lower,
                                bridge_upper,
                            )
                        )
                        moving_bridge_seeds = (
                            deduplicated_bridge_multistart_seeds(
                                moving_bridge_seed,
                                moving_separation_seeds,
                            )
                        )
                        moving_bridge_candidates = []
                        suffix_horizon_required = bool(
                            args.mpc_suffix_horizon_nodes > 0
                        )
                        suffix_horizon_candidate = (
                            build_suffix_horizon_candidate()
                        )
                        suffix_horizon_failed_closed = bool(
                            suffix_horizon_required
                            and suffix_horizon_candidate is None
                        )
                        if suffix_horizon_failed_closed:
                            if last_suffix_horizon_evidence is not None:
                                last_suffix_horizon_evidence[
                                    "fail_closed"
                                ] = np.asarray(True, dtype=np.bool_)
                            print(
                                "[SUFFIX-HORIZON-FAIL-CLOSED] "
                                f"distance_m={desired_distance:.9f} "
                                "myopic_bridge_commit_allowed=False",
                                flush=True,
                            )
                        if suffix_horizon_candidate is not None:
                            suffix_horizon_candidate.bridge_seed_index = -1
                            suffix_horizon_candidate.bridge_multistart_rank = (
                                moving_bridge_multistart_rank(
                                    suffix_horizon_candidate
                                )
                            )
                            moving_bridge_candidates.append(
                                suffix_horizon_candidate
                            )
                        for bridge_seed_index, bridge_seed in enumerate(
                            moving_bridge_seeds
                        ):
                            bridge_candidate = least_squares(
                                moving_bridge_residual,
                                bridge_seed,
                                bounds=(bridge_lower, bridge_upper),
                                max_nfev=args.mpc_max_nfev,
                                xtol=1.0e-10,
                                ftol=1.0e-10,
                                gtol=1.0e-10,
                                x_scale="jac",
                                diff_step=1.0e-5,
                            )
                            bridge_candidate.candidate_kind = (
                                "moving_bridge"
                            )
                            bridge_candidate.bridge_seed_index = (
                                bridge_seed_index
                            )
                            bridge_candidate.bridge_multistart_rank = (
                                moving_bridge_multistart_rank(
                                    bridge_candidate
                                )
                            )
                            moving_bridge_candidates.append(
                                bridge_candidate
                            )
                        moving_bridge = min(
                            moving_bridge_candidates,
                            key=lambda candidate: (
                                candidate.bridge_multistart_rank[0],
                                0.0
                                if getattr(
                                    candidate,
                                    "candidate_kind",
                                    "",
                                )
                                == "suffix_horizon"
                                else 1.0,
                                *candidate.bridge_multistart_rank[1:],
                            ),
                        )
                        if (
                            getattr(
                                moving_bridge,
                                "candidate_kind",
                                "",
                            )
                            == "suffix_horizon"
                        ):
                            suffix_horizon_selected = True
                            pending_suffix_horizon = {
                                "anchor_distance_m": float(
                                    desired_distance
                                ),
                                "anchor_q_rad": moving_bridge.x.copy(),
                                "distance_m": np.asarray(
                                    moving_bridge.suffix_distance_m[1:],
                                    dtype=np.float64,
                                ).copy(),
                                "q_rad": np.asarray(
                                    moving_bridge.suffix_q_rad[1:],
                                    dtype=np.float64,
                                ).copy(),
                            }
                        (
                            moving_bridge_points,
                            _,
                            _,
                            moving_bridge_arc,
                            moving_bridge_aux,
                        ) = contact_state(moving_bridge.x)
                        moving_bridge_progress_error = np.abs(
                            direction
                            * (moving_bridge_arc - bridge_desired_arc)
                        )
                        moving_bridge_progress_error[0] = 0.0
                        moving_bridge_normal_error = np.abs(
                            moving_bridge_aux[:, 1] - desired_standoff
                        )
                        moving_bridge_tangential_error = (
                            (
                                moving_bridge_aux[:, 0]
                                - desired_azimuth
                                + np.pi
                            )
                            % (2.0 * np.pi)
                            - np.pi
                        ) * CAPSULE_RADIUS
                        moving_bridge_progress = direction * (
                            moving_bridge_arc - start_arc
                        )
                        moving_bridge_monotonic_error = np.maximum(
                            minimum_progress - moving_bridge_progress,
                            0.0,
                        )
                        moving_bridge_monotonic_error[0] = 0.0
                        (
                            moving_bridge_normal_ok,
                            moving_bridge_contact_mask,
                            moving_bridge_normal_tolerances,
                        ) = scheduled_contact_status(
                            moving_bridge_normal_error[1:],
                            desired_distance,
                        )
                        (
                            moving_bridge_recovery_normal_ok,
                            moving_bridge_recovery_contact_mask,
                            moving_bridge_recovery_normal_tolerances,
                        ) = recovery_contact_status(
                            moving_bridge_normal_error[1:]
                        )
                        moving_bridge_palm_error = float(
                            np.linalg.norm(
                                moving_bridge_points[0] - palm_target
                            )
                        )
                        moving_bridge_collision_ok = True
                        moving_bridge_arm_clearance = np.inf
                        moving_bridge_hand_clearance = np.inf
                        moving_bridge_tip_clearance = np.inf
                        moving_bridge_self_count = 0
                        moving_bridge_self_occurrences = 0
                        moving_bridge_protected_clearance = (
                            planner_protected_self_clearance_m
                        )
                        moving_bridge_protected_pair_name = ""
                        moving_bridge_pad_alignment = 1.0
                        if args.collision_mode == "full_robot":
                            (
                                moving_bridge_arm_clearance,
                                _,
                                moving_bridge_hand_clearance,
                                _,
                                moving_bridge_tip_clearance,
                                _,
                                moving_bridge_self_count,
                                moving_bridge_self_occurrences,
                                moving_bridge_protected_clearance,
                                moving_bridge_protected_pair_name,
                                moving_bridge_pad_alignment,
                            ) = segment_collision_status(moving_bridge.x)
                            moving_bridge_collision_ok = bool(
                                moving_bridge_arm_clearance
                                >= args.min_arm_clearance_mm / 1000.0
                                and moving_bridge_hand_clearance
                                >= -args.max_incidental_hand_penetration_mm
                                / 1000.0
                                and moving_bridge_tip_clearance
                                >= -args.max_contact_penetration_mm / 1000.0
                                and moving_bridge_self_count == 0
                                and moving_bridge_pad_alignment
                                >= planner_pad_alignment
                            )
                        moving_bridge_within_joint_limits = bool(
                            np.all(moving_bridge.x >= lower - 1.0e-12)
                            and np.all(moving_bridge.x <= upper + 1.0e-12)
                        )
                        moving_bridge_joint_margin_rad = float(
                            np.min(
                                np.minimum(
                                    moving_bridge.x - lower,
                                    upper - moving_bridge.x,
                                )
                            )
                        )
                        moving_tip_motion_m = direction * (
                            moving_bridge_arc[1:] - bridge_arc[1:]
                        )
                        minimum_bridge_motion_m = (
                            args.mpc_feasibility_bridge_min_progress_ratio
                            * bridge_interval_m
                        )
                        bridge_active_fingers = (
                            np.abs(bridge_offset_delta_m) > 1.0e-12
                        )
                        moving_joint_motion_rad = float(
                            np.max(
                                np.abs(moving_bridge.x - previous_q)
                            )
                        )
                        moving_bridge_joint_limits_ok = bool(
                            moving_bridge_within_joint_limits
                            and moving_joint_motion_rad
                            <= args.max_plan_joint_step_rad + 1.0e-12
                        )
                        (
                            moving_bridge_motion_ok,
                            moving_progressing_finger_count,
                        ) = evaluate_moving_bridge_motion(
                            max_joint_motion_rad=moving_joint_motion_rad,
                            tip_motion_m=moving_tip_motion_m,
                            minimum_tip_motion_m=minimum_bridge_motion_m,
                            active_fingers=bridge_active_fingers,
                        )
                        moving_recovery_progress_limit_m = max(
                            progress_limit_m,
                            args.mpc_recovery_bridge_progress_tolerance_mm
                            / 1000.0,
                        )
                        moving_bridge_conditions = evaluate_bridge_conditions(
                            progress_error_m=moving_bridge_progress_error,
                            progress_limit_m=progress_limit_m,
                            normal_ok=moving_bridge_normal_ok,
                            tangential_error_m=(
                                moving_bridge_tangential_error[1:]
                            ),
                            tangential_limit_m=tip_tangential_tolerances,
                            monotonic_error_m=(
                                moving_bridge_monotonic_error
                            ),
                            monotonic_limit_m=(
                                args.mpc_monotonic_tolerance_mm / 1000.0
                            ),
                            palm_error_m=moving_bridge_palm_error,
                            palm_limit_m=palm_tracking_limit_m,
                            collision_ok=moving_bridge_collision_ok,
                            joint_ok=moving_bridge_joint_limits_ok,
                            motion_ok=moving_bridge_motion_ok,
                            budget_ok=True,
                        )
                        moving_recovery_conditions = evaluate_bridge_conditions(
                            progress_error_m=moving_bridge_progress_error,
                            progress_limit_m=moving_recovery_progress_limit_m,
                            normal_ok=moving_bridge_recovery_normal_ok,
                            tangential_error_m=(
                                moving_bridge_tangential_error[1:]
                            ),
                            tangential_limit_m=tip_tangential_tolerances,
                            monotonic_error_m=(
                                moving_bridge_monotonic_error
                            ),
                            monotonic_limit_m=(
                                args.mpc_monotonic_tolerance_mm / 1000.0
                            ),
                            palm_error_m=moving_bridge_palm_error,
                            palm_limit_m=palm_tracking_limit_m,
                            collision_ok=moving_bridge_collision_ok,
                            joint_ok=moving_bridge_joint_limits_ok,
                            motion_ok=moving_bridge_motion_ok,
                            budget_ok=recovery_bridge_budget_ok,
                        )
                        moving_bridge_hard_ok = all(
                            moving_bridge_conditions.values()
                        )
                        moving_recovery_hard_ok = bool(
                            not moving_bridge_hard_ok
                            and all(moving_recovery_conditions.values())
                        )
                        if (
                            not moving_bridge_hard_ok
                            and not moving_recovery_hard_ok
                        ):
                            bridge_metrics = build_bridge_rejection_metrics(
                                bridge_interval_m=bridge_interval_m,
                                bridge_interval_limit_m=(
                                    bridge_interval_limit_m
                                ),
                                progress_error_m=(
                                    moving_bridge_progress_error
                                ),
                                strict_progress_limit_m=progress_limit_m,
                                recovery_progress_limit_m=(
                                    moving_recovery_progress_limit_m
                                ),
                                normal_error_m=(
                                    moving_bridge_normal_error[1:]
                                ),
                                strict_normal_tolerance_m=(
                                    moving_bridge_normal_tolerances
                                ),
                                strict_contact_mask=(
                                    moving_bridge_contact_mask
                                ),
                                strict_contact_count_required=(
                                    args.min_planner_contact_fingers
                                ),
                                recovery_normal_tolerance_m=(
                                    moving_bridge_recovery_normal_tolerances
                                ),
                                recovery_contact_mask=(
                                    moving_bridge_recovery_contact_mask
                                ),
                                recovery_contact_count_required=(
                                    args.mpc_recovery_bridge_min_contact_fingers
                                ),
                                tangential_error_m=(
                                    moving_bridge_tangential_error[1:]
                                ),
                                tangential_tolerance_m=(
                                    tip_tangential_tolerances
                                ),
                                monotonic_error_m=(
                                    moving_bridge_monotonic_error
                                ),
                                monotonic_limit_m=(
                                    args.mpc_monotonic_tolerance_mm / 1000.0
                                ),
                                palm_error_m=moving_bridge_palm_error,
                                palm_limit_m=palm_tracking_limit_m,
                                collision_mode=args.collision_mode,
                                arm_clearance_m=(
                                    moving_bridge_arm_clearance
                                ),
                                arm_clearance_limit_m=(
                                    args.min_arm_clearance_mm / 1000.0
                                ),
                                hand_clearance_m=(
                                    moving_bridge_hand_clearance
                                ),
                                hand_clearance_limit_m=-(
                                    args.max_incidental_hand_penetration_mm
                                    / 1000.0
                                ),
                                tip_clearance_m=(
                                    moving_bridge_tip_clearance
                                ),
                                tip_clearance_limit_m=-(
                                    args.max_contact_penetration_mm / 1000.0
                                ),
                                self_collision_count=(
                                    moving_bridge_self_count
                                ),
                                self_collision_sample_occurrence_count=(
                                    moving_bridge_self_occurrences
                                ),
                                minimum_protected_self_clearance_m=(
                                    moving_bridge_protected_clearance
                                ),
                                protected_self_clearance_target_m=(
                                    planner_protected_self_clearance_m
                                ),
                                minimum_protected_self_pair_name=(
                                    moving_bridge_protected_pair_name
                                ),
                                pad_alignment=(
                                    moving_bridge_pad_alignment
                                ),
                                pad_alignment_limit=planner_pad_alignment,
                                joint_min_margin_rad=(
                                    moving_bridge_joint_margin_rad
                                ),
                                max_joint_motion_rad=(
                                    moving_joint_motion_rad
                                ),
                                tip_motion_m=moving_tip_motion_m,
                                minimum_tip_motion_m=(
                                    minimum_bridge_motion_m
                                ),
                                bridge_active_fingers=(
                                    bridge_active_fingers
                                ),
                                progressing_finger_count=(
                                    moving_progressing_finger_count
                                ),
                                progressing_finger_count_required=(
                                    MOVING_BRIDGE_FORWARD_FINGER_COUNT
                                ),
                                recovery_dwell_m=(
                                    proposed_recovery_bridge_dwell_m
                                ),
                                recovery_dwell_limit_m=(
                                    args.mpc_recovery_bridge_max_span_mm
                                    / 1000.0
                                ),
                                recovery_total_m=(
                                    proposed_recovery_bridge_total_m
                                ),
                                recovery_total_limit_m=(
                                    args.mpc_recovery_bridge_max_total_ratio
                                    * args.axial_travel_m
                                ),
                                distance_m=desired_distance,
                                recovery_terminal_cutoff_m=(
                                    args.axial_travel_m
                                    - args.mpc_recovery_bridge_terminal_margin_mm
                                    / 1000.0
                                ),
                                solver_cost=float(moving_bridge.cost),
                                solver_nfev=int(moving_bridge.nfev),
                            )
                            last_bridge_rejection_record = (
                                make_bridge_rejection_record(
                                    keyframe=keyframe,
                                    keyframe_count=keyframe_count,
                                    distance_m=desired_distance,
                                    fallback=(
                                        "static_bridge"
                                        if static_bridge_candidate is not None
                                        else "planner_failure"
                                    ),
                                    strict_conditions=(
                                        moving_bridge_conditions
                                    ),
                                    recovery_conditions=(
                                        moving_recovery_conditions
                                    ),
                                    metrics=bridge_metrics,
                                )
                            )
                            last_rejected_moving_bridge = (
                                RejectedMovingBridgeCandidate(
                                    q_rad=moving_bridge.x.copy(),
                                    points_m=moving_bridge_points.copy(),
                                    arcs_m=moving_bridge_arc.copy(),
                                    desired_arcs_m=(
                                        bridge_desired_arc.copy()
                                    ),
                                )
                            )
                            if static_bridge_candidate is not None:
                                emit_bridge_rejection(
                                    last_bridge_rejection_record
                                )
                        moving_offset_norm_m = float(
                            np.linalg.norm(
                                bridge_rephase_offset_m
                                - starting_rephase_offset_m
                            )
                        )
                        moving_progress_max_m = float(
                            moving_bridge_progress_error.max()
                        )
                        moving_orientation_rank = fallback_orientation_rank(
                            minimum_pad_alignment=(
                                moving_bridge_pad_alignment
                            ),
                            offset_norm_m=moving_offset_norm_m,
                            progress_error_m=moving_progress_max_m,
                            solver_cost=float(moving_bridge.cost),
                            minimum_protected_self_clearance_m=(
                                moving_bridge_protected_clearance
                            ),
                        )
                        if (
                            moving_bridge_hard_ok
                            and not suffix_horizon_failed_closed
                        ):
                            accepted_rephase_candidates.append(
                                (
                                    moving_orientation_rank,
                                    moving_offset_norm_m,
                                    moving_progress_max_m,
                                    float(moving_bridge.cost),
                                    moving_bridge,
                                    moving_bridge_progress_error,
                                    moving_bridge_normal_error,
                                    moving_bridge_arc,
                                    bridge_rephase_offset_m.copy(),
                                    bridge_desired_arc.copy(),
                                )
                            )
                            feasibility_bridge_selected = True
                            bridge_tip_motion_m = (
                                moving_tip_motion_m.copy()
                            )
                            bridge_joint_motion_rad = float(
                                np.max(
                                    np.abs(
                                        moving_bridge.x - previous_q
                                    )
                                )
                            )
                        elif (
                            moving_recovery_hard_ok
                            and not suffix_horizon_failed_closed
                        ):
                            accepted_rephase_candidates.append(
                                (
                                    moving_orientation_rank,
                                    moving_offset_norm_m,
                                    moving_progress_max_m,
                                    float(moving_bridge.cost),
                                    moving_bridge,
                                    moving_bridge_progress_error,
                                    moving_bridge_normal_error,
                                    moving_bridge_arc,
                                    bridge_rephase_offset_m.copy(),
                                    bridge_desired_arc.copy(),
                                )
                            )
                            feasibility_bridge_selected = True
                            recovery_bridge_selected = True
                            selected_recovery_bridge_dwell_m = (
                                proposed_recovery_bridge_dwell_m
                            )
                            bridge_tip_motion_m = moving_tip_motion_m.copy()
                            bridge_joint_motion_rad = float(
                                np.max(
                                    np.abs(
                                        moving_bridge.x - previous_q
                                    )
                                )
                            )
                        elif (
                            static_bridge_candidate is not None
                            and not suffix_horizon_failed_closed
                        ):
                            accepted_rephase_candidates.append(
                                static_bridge_candidate
                            )
                            feasibility_bridge_selected = True
                            static_feasibility_bridge_selected = True
                            selected_static_bridge_dwell_m = (
                                proposed_static_bridge_dwell_m
                            )
                            recovery_bridge_selected = True
                            selected_recovery_bridge_dwell_m = (
                                proposed_recovery_bridge_dwell_m
                            )
                        desired_arc[:] = nominal_desired_arc
                    if accepted_rephase_candidates:
                        (
                            _,
                            _,
                            _,
                            _,
                            best,
                            progress_error,
                            normal_error,
                            achieved_arc,
                            auto_rephase_offset_m,
                            accepted_desired_arc,
                        ) = min(
                            accepted_rephase_candidates,
                            key=lambda item: item[0],
                        )
                        desired_arc[:] = accepted_desired_arc
                        coarse_target_progress[keyframe] = direction * (
                            desired_arc - start_arc
                        )
                        coarse_auto_rephase_offset_m[keyframe] = (
                            auto_rephase_offset_m
                        )
                        suffix_horizon_selected = bool(
                            getattr(best, "candidate_kind", "")
                            == "suffix_horizon"
                        )
                        if not suffix_horizon_selected:
                            pending_suffix_horizon = None
                        if suffix_horizon_selected:
                            event_name = "STRICT-SUFFIX-HORIZON"
                        elif static_feasibility_bridge_selected:
                            event_name = "STATIC-FEASIBILITY-BRIDGE"
                        elif recovery_bridge_selected:
                            event_name = "MOVING-RECOVERY-BRIDGE"
                        elif feasibility_bridge_selected:
                            event_name = "MOVING-FEASIBILITY-BRIDGE"
                        else:
                            event_name = "AUTO-REPHASE"
                        bridge_motion_summary = (
                            " tip_motion_mm="
                            f"{(bridge_tip_motion_m * 1000).round(3).tolist()} "
                            "max_joint_motion_rad="
                            f"{bridge_joint_motion_rad:.6f}"
                            if (
                                feasibility_bridge_selected
                                and not static_feasibility_bridge_selected
                            )
                            else ""
                        )
                        print(
                            f"[{event_name}] "
                            f"keyframe={keyframe}/{keyframe_count} "
                            f"distance_m={desired_distance:.4f} "
                            "offset_mm="
                            f"{(auto_rephase_offset_m * 1000).round(3).tolist()} "
                            "progress_error_mm="
                            f"{(progress_error * 1000).round(2).tolist()}"
                            f"{bridge_motion_summary}",
                            flush=True,
                        )
                    else:
                        desired_arc[:] = nominal_desired_arc
                (
                    best_points,
                    _,
                    _,
                    _,
                    best_auxiliary,
                ) = contact_state(best.x)
                best_palm_position_error = float(
                    np.linalg.norm(best_points[0] - palm_target)
                )
                palm_error_vector = best_points[0] - palm_target
                palm_error_local = np.asarray(
                    (
                        np.dot(palm_error_vector, palm_ball_normal),
                        np.dot(palm_error_vector, palm_ball_azimuth),
                        np.dot(palm_error_vector, palm_ball_meridian),
                    ),
                    dtype=np.float64,
                )
                tangential_error = (
                    (
                        best_auxiliary[:, 0]
                        - desired_azimuth
                        + np.pi
                    )
                    % (2.0 * np.pi)
                    - np.pi
                ) * CAPSULE_RADIUS
                achieved_progress = direction * (
                    achieved_arc - start_arc
                )
                monotonic_error = np.maximum(
                    minimum_progress - achieved_progress,
                    0.0,
                )
                monotonic_error[0] = 0.0
                accepted_progress_limit_m = (
                    max(
                        args.mpc_static_bridge_progress_tolerance_mm,
                        args.mpc_recovery_bridge_progress_tolerance_mm,
                    )
                    / 1000.0
                    if recovery_bridge_selected
                    else active_progress_tolerance_mm / 1000.0
                )
                if recovery_bridge_selected:
                    (
                        normal_contact_ok,
                        nominal_contact_mask,
                        active_normal_tolerances,
                    ) = recovery_contact_status(normal_error[1:])
                    required_contact_fingers = (
                        args.mpc_recovery_bridge_min_contact_fingers
                    )
                else:
                    (
                        normal_contact_ok,
                        nominal_contact_mask,
                        active_normal_tolerances,
                    ) = scheduled_contact_status(
                        normal_error[1:],
                        desired_distance,
                    )
                    required_contact_fingers = (
                        args.min_planner_contact_fingers
                    )
                best_arm_clearance = np.inf
                best_arm_nearest = ""
                best_hand_clearance = np.inf
                best_hand_nearest = ""
                best_tip_clearance = np.inf
                best_tip_nearest = ""
                best_self_count = 0
                best_self_occurrences = 0
                best_protected_self_clearance = (
                    planner_protected_self_clearance_m
                )
                best_protected_self_pair_name = ""
                best_pad_alignment = 1.0
                if args.collision_mode == "full_robot":
                    (
                        best_arm_clearance,
                        best_arm_nearest,
                        best_hand_clearance,
                        best_hand_nearest,
                        best_tip_clearance,
                        best_tip_nearest,
                        best_self_count,
                        best_self_occurrences,
                        best_protected_self_clearance,
                        best_protected_self_pair_name,
                        best_pad_alignment,
                    ) = segment_collision_status(best.x)
                arm_clearance_limit_m = (
                    args.min_arm_clearance_mm / 1000.0
                )
                hand_clearance_limit_m = -(
                    args.max_incidental_hand_penetration_mm / 1000.0
                )
                tip_clearance_limit_m = -(
                    args.max_contact_penetration_mm / 1000.0
                )
                best_joint_margin_rad = float(
                    np.min(
                        np.minimum(
                            best.x - lower,
                            upper - best.x,
                        )
                    )
                )
                candidate_failure_metrics = (
                    build_candidate_failure_metrics(
                        progress_error_m=progress_error,
                        progress_limit_m=accepted_progress_limit_m,
                        normal_error_m=normal_error[1:],
                        normal_tolerance_m=active_normal_tolerances,
                        contact_mask=nominal_contact_mask,
                        contact_count_required=required_contact_fingers,
                        tangential_error_m=tangential_error[1:],
                        tangential_tolerance_m=(
                            tip_tangential_tolerances
                        ),
                        monotonic_error_m=monotonic_error,
                        monotonic_limit_m=(
                            args.mpc_monotonic_tolerance_mm / 1000.0
                        ),
                        palm_error_m=best_palm_position_error,
                        palm_limit_m=palm_tracking_limit_m,
                        palm_error_world_m=palm_error_vector,
                        palm_error_local_m=palm_error_local,
                        collision_mode=args.collision_mode,
                        arm_clearance_m=best_arm_clearance,
                        arm_clearance_limit_m=arm_clearance_limit_m,
                        arm_nearest_geometry=best_arm_nearest,
                        hand_clearance_m=best_hand_clearance,
                        hand_clearance_limit_m=hand_clearance_limit_m,
                        hand_nearest_geometry=best_hand_nearest,
                        tip_clearance_m=best_tip_clearance,
                        tip_clearance_limit_m=tip_clearance_limit_m,
                        tip_nearest_geometry=best_tip_nearest,
                        self_collision_count=best_self_count,
                        self_collision_sample_occurrence_count=(
                            best_self_occurrences
                        ),
                        minimum_protected_self_clearance_m=(
                            best_protected_self_clearance
                        ),
                        protected_self_clearance_target_m=(
                            planner_protected_self_clearance_m
                        ),
                        minimum_protected_self_pair_name=(
                            best_protected_self_pair_name
                        ),
                        pad_alignment=best_pad_alignment,
                        pad_alignment_limit=planner_pad_alignment,
                        joint_min_margin_rad=best_joint_margin_rad,
                        solver_cost=float(best.cost),
                        solver_nfev=int(best.nfev),
                    )
                )
                best_collision_ok = bool(
                    candidate_failure_metrics["condition_collision_ok"]
                )
                if (
                    best_palm_position_error
                    > palm_tracking_limit_m
                ):
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="palm_drift",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="palm_drift",
                        message=(
                            "Adaptive surface MPC exceeded the non-contact "
                            "palm "
                            f"{'guide drift guard' if args.palm_guide_only else 'feasibility ball'}: "
                            f"keyframe={keyframe}/{keyframe_count} "
                            f"error_mm={best_palm_position_error * 1000:.3f} "
                            f"limit_mm={palm_tracking_limit_m * 1000:.3f} "
                            "offset_world_mm="
                            f"{(palm_error_vector * 1000).round(3).tolist()} "
                            "offset_normal_azimuth_meridian_mm="
                            f"{(palm_error_local * 1000).round(3).tolist()}"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                if (
                    float(monotonic_error.max())
                    > args.mpc_monotonic_tolerance_mm / 1000.0
                ):
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="monotonic_progress",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="monotonic_progress",
                        message=(
                            "Adaptive surface MPC violated monotonic "
                            f"progress: keyframe={keyframe}/{keyframe_count} "
                            "error_mm="
                            f"{(monotonic_error * 1000).round(2).tolist()}"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                if float(progress_error.max()) > accepted_progress_limit_m:
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="longitudinal_progress",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="longitudinal_progress",
                        message=(
                            "Adaptive surface MPC missed longitudinal "
                            f"progress: keyframe={keyframe}/{keyframe_count} "
                            f"distance_m={desired_distance:.4f} "
                            "error_mm="
                            f"{(progress_error * 1000).round(2).tolist()}"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                if not normal_contact_ok:
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="fingertip_support",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="fingertip_support",
                        message=(
                            "Adaptive surface MPC violated the scheduled "
                            "fingertip support set: "
                            f"keyframe={keyframe}/{keyframe_count} "
                            f"distance_m={desired_distance:.4f} "
                            "contacts="
                            f"{int(np.count_nonzero(nominal_contact_mask))}/4 "
                            f"required={required_contact_fingers} "
                            "error_mm="
                            f"{(normal_error[1:] * 1000).round(2).tolist()} "
                            "tolerance_mm="
                            f"{(active_normal_tolerances * 1000).round(2).tolist()}"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                if np.any(
                    np.abs(tangential_error[1:])
                    > tip_tangential_tolerances
                ):
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="tangential_gait",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="tangential_gait",
                        message=(
                            "Adaptive surface MPC missed fingertip "
                            f"tangential gait: keyframe={keyframe}/"
                            f"{keyframe_count} "
                            f"distance_m={desired_distance:.4f} "
                            "error_mm="
                            f"{(np.abs(tangential_error[1:]) * 1000).round(2).tolist()} "
                            "tolerance_mm="
                            f"{(tip_tangential_tolerances * 1000).round(2).tolist()}"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                if (
                    args.collision_mode == "full_robot"
                    and not best_collision_ok
                ):
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="contact_policy",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="contact_policy",
                        message=(
                            "Adaptive surface MPC has no contact-policy-"
                            f"safe candidate at keyframe={keyframe}/"
                            f"{keyframe_count}: arm_clearance_mm="
                            f"{best_arm_clearance * 1000:.3f} "
                            "required_arm_mm="
                            f"{args.min_arm_clearance_mm:.3f} "
                            f"nearest_arm={best_arm_nearest} "
                            f"hand_clearance_mm="
                            f"{best_hand_clearance * 1000:.3f} "
                            "allowed_hand_penetration_mm="
                            f"{args.max_incidental_hand_penetration_mm:.3f} "
                            f"nearest_hand={best_hand_nearest} "
                            f"tip_clearance_mm="
                            f"{best_tip_clearance * 1000:.3f} "
                            "allowed_tip_penetration_mm="
                            f"{args.max_contact_penetration_mm:.3f} "
                            f"nearest_tip={best_tip_nearest} "
                            "self_collision_unique_pairs="
                            f"{best_self_count} "
                            "self_collision_sample_occurrences="
                            f"{best_self_occurrences} "
                            "protected_self_clearance_mm="
                            f"{best_protected_self_clearance * 1000:.6f} "
                            f"protected_self_pair="
                            f"{best_protected_self_pair_name} "
                            "max_pad_angle_deg="
                            f"{np.degrees(np.arccos(np.clip(best_pad_alignment, -1, 1))):.2f} "
                            "planner_limit_deg="
                            f"{args.max_pad_angle_deg - args.planner_pad_angle_margin_deg:.2f}"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                selected_low_motion_failures = (
                    prospective_low_motion_failures(
                        np.asarray((desired_distance,), dtype=np.float64),
                        best.x,
                        candidate_keyframe=keyframe,
                        candidate_marked=bool(
                            recovery_bridge_selected
                            or static_feasibility_bridge_selected
                        ),
                    )
                )
                if selected_low_motion_failures:
                    first_low_motion_window = selected_low_motion_failures[0][
                        "first_window"
                    ]
                    candidate_failure_metrics[
                        "condition_low_motion_ok"
                    ] = np.asarray(False, dtype=np.bool_)
                    candidate_failure_metrics[
                        "low_motion_window_start_m"
                    ] = np.asarray(
                        first_low_motion_window[
                            "target_distance_start_m"
                        ],
                        dtype=np.float64,
                    )
                    candidate_failure_metrics[
                        "low_motion_window_end_m"
                    ] = np.asarray(
                        first_low_motion_window["target_distance_end_m"],
                        dtype=np.float64,
                    )
                    candidate_failure_metrics[
                        "low_motion_forward_finger_count"
                    ] = np.asarray(
                        first_low_motion_window["forward_finger_count"],
                        dtype=np.int32,
                    )
                    candidate_failure_metrics[
                        "low_motion_required_tip_progress_m"
                    ] = np.asarray(
                        first_low_motion_window[
                            "required_tip_progress_m"
                        ],
                        dtype=np.float64,
                    )
                    candidate_failure_metrics[
                        "low_motion_tip_progress_delta_m"
                    ] = np.asarray(
                        first_low_motion_window["tip_progress_delta_m"],
                        dtype=np.float64,
                    )
                    print(
                        "[PROSPECTIVE-LOW-MOTION] "
                        f"keyframe={keyframe}/{keyframe_count} "
                        f"distance_m={desired_distance:.9f} "
                        "window_m="
                        f"[{float(first_low_motion_window['target_distance_start_m']):.9f},"
                        f"{float(first_low_motion_window['target_distance_end_m']):.9f}] "
                        "forward_fingers="
                        f"{int(first_low_motion_window['forward_finger_count'])}/4 "
                        "required=3 tip_delta_mm="
                        f"{(np.asarray(first_low_motion_window['tip_progress_delta_m']) * 1000).round(3).tolist()}",
                        flush=True,
                    )
                    if insert_auto_refinement(
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        reason="unmarked_low_motion",
                    ):
                        continue
                    raise_adaptive_planner_failure(
                        reason="unmarked_low_motion",
                        message=(
                            "Adaptive surface MPC would commit an unmarked "
                            "20-frame fingertip stall: "
                            f"keyframe={keyframe}/{keyframe_count} "
                            f"distance_m={desired_distance:.6f} "
                            "window_m="
                            f"[{float(first_low_motion_window['target_distance_start_m']):.6f},"
                            f"{float(first_low_motion_window['target_distance_end_m']):.6f}] "
                            "forward_fingers="
                            f"{int(first_low_motion_window['forward_finger_count'])}/4"
                        ),
                        keyframe=keyframe,
                        desired_distance=desired_distance,
                        desired_arc=desired_arc,
                        final_best_q=best.x,
                        final_best_points=best_points,
                        final_best_arc=achieved_arc,
                        failure_metrics=candidate_failure_metrics,
                        bridge_record=last_bridge_rejection_record,
                        rejected_moving_bridge=(
                            last_rejected_moving_bridge
                        ),
                    )
                q = best.x
                coarse_feasibility_bridge[keyframe] = (
                    feasibility_bridge_selected
                )
                coarse_suffix_horizon[keyframe] = suffix_horizon_selected
                coarse_static_feasibility_bridge[keyframe] = (
                    static_feasibility_bridge_selected
                )
                coarse_static_bridge_dwell_m[keyframe] = (
                    selected_static_bridge_dwell_m
                    if static_feasibility_bridge_selected
                    else 0.0
                )
                coarse_recovery_bridge[keyframe] = (
                    recovery_bridge_selected
                )
                coarse_recovery_bridge_dwell_m[keyframe] = (
                    selected_recovery_bridge_dwell_m
                    if recovery_bridge_selected
                    else 0.0
                )
                if static_feasibility_bridge_selected:
                    static_bridge_total_m += (
                        desired_distance
                        - float(coarse_distance[keyframe - 1])
                    )
                if recovery_bridge_selected:
                    recovery_bridge_total_m += (
                        desired_distance
                        - float(coarse_distance[keyframe - 1])
                    )
                coarse_q[keyframe] = q
                coarse_progress[keyframe] = direction * (
                    achieved_arc - start_arc
                )
                coarse_normal_error[keyframe] = normal_error
                coarse_palm_position_error[keyframe] = (
                    best_palm_position_error
                )
                coarse_cost[keyframe] = float(best.cost)
                coarse_nfev[keyframe] = int(best.nfev)
                if suffix_horizon_selected:
                    suffix_horizon_success_count += 1
                suffix_horizon_cache = (
                    pending_suffix_horizon
                    if suffix_horizon_selected
                    else None
                )
                previous_delta = q - previous_q
                previous_q = q
                print(
                    "[ADAPTIVE-MPC] "
                    f"keyframe={keyframe:02d}/{keyframe_count} "
                    f"travel_m={desired_distance:.4f} "
                    f"progress_mm={(coarse_progress[keyframe] * 1000).round(1).tolist()} "
                    f"tip_normal_error_mm="
                    f"{(normal_error[1:] * 1000).round(2).tolist()} "
                    f"contacts="
                    f"{int(np.count_nonzero(nominal_contact_mask))}/4 "
                    f"tip_tangential_error_mm="
                    f"{(np.abs(tangential_error[1:]) * 1000).round(2).tolist()} "
                    f"palm_position_error_mm="
                    f"{best_palm_position_error * 1000:.2f} "
                    f"candidate_kind="
                    f"{getattr(best, 'candidate_kind', 'rephase_or_bridge')} "
                    f"segment_tip_clearance_mm="
                    f"{best_tip_clearance * 1000:.3f} "
                    f"segment_max_pad_angle_deg="
                    f"{np.degrees(np.arccos(np.clip(best_pad_alignment, -1, 1))):.2f} "
                    f"nfev={best.nfev}",
                    flush=True,
                )
                keyframe += 1

            frame_target_distance = np.linspace(
                0.0,
                args.axial_travel_m,
                frame_count + 1,
            )[1:]
            surface_plan = np.zeros((frame_count, 5, 3), dtype=np.float32)
            kinematic_plan = np.zeros_like(surface_plan)
            normal_plan = np.zeros_like(surface_plan)
            joint_plan = np.zeros((frame_count, TOTAL_DOF), dtype=np.float32)
            residual_plan = np.zeros((frame_count, 5), dtype=np.float32)
            distance_plan = np.zeros(frame_count, dtype=np.float32)
            progress_plan = np.zeros((frame_count, 5), dtype=np.float32)
            normal_error_plan = np.zeros_like(progress_plan)
            scheduled_contact_mask_plan = np.zeros(
                (frame_count, 4),
                dtype=bool,
            )
            scheduled_contact_count_plan = np.zeros(
                frame_count,
                dtype=np.int8,
            )
            recovery_bridge_mask_plan = np.zeros(
                frame_count,
                dtype=np.bool_,
            )
            static_bridge_mask_plan = np.zeros(
                frame_count,
                dtype=np.bool_,
            )
            palm_position_error_plan = np.zeros(
                frame_count, dtype=np.float32
            )
            for frame, desired_distance in enumerate(frame_target_distance):
                left = int(
                    np.searchsorted(
                        coarse_distance,
                        desired_distance,
                        side="right",
                    )
                    - 1
                )
                left = min(max(left, 0), keyframe_count - 1)
                interval = coarse_distance[left + 1] - coarse_distance[left]
                blend = (
                    desired_distance - coarse_distance[left]
                ) / max(interval, 1.0e-12)
                blend = float(np.clip(blend, 0.0, 1.0))
                blend = blend * blend * (3.0 - 2.0 * blend)
                q = (1.0 - blend) * coarse_q[left] + blend * coarse_q[left + 1]
                points, surface, normals, arc, auxiliary = contact_state(q)
                progress = direction * (arc - start_arc)
                desired_progress = np.full(5, desired_distance)
                desired_progress[0] = (
                    (1.0 - blend) * coarse_progress[left, 0]
                    + blend * coarse_progress[left + 1, 0]
                )
                surface_plan[frame] = surface
                kinematic_plan[frame] = points
                normal_plan[frame] = normals
                joint_plan[frame] = q
                residual_plan[frame] = np.abs(
                    progress - desired_progress
                )
                residual_plan[frame, 0] = 0.0
                progress_plan[frame] = progress
                normal_error_plan[frame] = np.abs(
                    auxiliary[:, 1]
                    - (
                        initial_signed_standoff
                        + min(
                            desired_distance
                            / max(0.025, args.axial_travel_m),
                            1.0,
                        )
                        * (
                            target_signed_standoff
                            - initial_signed_standoff
                        )
                    )
                )
                interpolation_recovery_bridge = bool(
                    coarse_recovery_bridge[left + 1]
                )
                interpolation_static_bridge = bool(
                    coarse_static_feasibility_bridge[left + 1]
                )
                recovery_bridge_mask_plan[frame] = (
                    interpolation_recovery_bridge
                )
                static_bridge_mask_plan[frame] = (
                    interpolation_static_bridge
                )
                if interpolation_recovery_bridge:
                    (
                        interpolation_contact_ok,
                        interpolation_contact_mask,
                        interpolation_normal_tolerances,
                    ) = recovery_contact_status(
                        normal_error_plan[frame, 1:]
                    )
                    interpolation_required_contacts = (
                        args.mpc_recovery_bridge_min_contact_fingers
                    )
                else:
                    (
                        interpolation_contact_ok,
                        interpolation_contact_mask,
                        interpolation_normal_tolerances,
                    ) = scheduled_contact_status(
                        normal_error_plan[frame, 1:],
                        desired_distance,
                    )
                    interpolation_required_contacts = (
                        args.min_planner_contact_fingers
                    )
                scheduled_contact_mask_plan[frame] = (
                    interpolation_contact_mask
                )
                scheduled_contact_count_plan[frame] = int(
                    np.count_nonzero(interpolation_contact_mask)
                )
                if not interpolation_contact_ok:
                    raise RuntimeError(
                        "Adaptive surface MPC interpolation violated the "
                        "scheduled fingertip support set: "
                        f"frame={frame + 1}/{frame_count} "
                        f"distance_m={desired_distance:.4f} "
                        f"contacts="
                        f"{scheduled_contact_count_plan[frame]}/4 "
                        f"required={interpolation_required_contacts} "
                        f"error_mm="
                        f"{(normal_error_plan[frame, 1:] * 1000).round(2).tolist()} "
                        f"tolerance_mm="
                        f"{(interpolation_normal_tolerances * 1000).round(2).tolist()}"
                    )
                palm_target = (
                    (1.0 - blend) * coarse_palm_target[left]
                    + blend * coarse_palm_target[left + 1]
                )
                palm_position_error_plan[frame] = float(
                    np.linalg.norm(points[0] - palm_target)
                )
                distance_plan[frame] = float(np.min(progress[1:]))

            marked_bridge_mask_plan = (
                static_bridge_mask_plan | recovery_bridge_mask_plan
            )
            low_motion_regions = find_unmarked_low_motion_windows(
                progress_plan,
                kinematic_plan,
                frame_target_distance,
                marked_bridge_mask_plan,
                distance_plan,
                window_frames=LOW_MOTION_DEFAULT_WINDOW_FRAMES,
                forward_progress_ratio=LOW_MOTION_FORWARD_PROGRESS_RATIO,
            )
            if low_motion_regions:
                first_window = low_motion_regions[0]["first_window"]
                assert isinstance(first_window, dict)
                first_start = int(first_window["start"])
                first_end = int(first_window["end"])
                first_tip_delta_m = np.asarray(
                    first_window["tip_progress_delta_m"]
                )
                first_forward_mask = np.asarray(
                    first_window["forward_mask"], dtype=np.bool_
                )
                evidence_path = None
                evidence_error = "disabled"
                try:
                    if args.mpc_failure_prefix_output is not None:
                        failure_prefix_path = Path(
                            args.mpc_failure_prefix_output
                        )
                        low_motion_evidence_path = (
                            failure_prefix_path.with_name(
                                f"{failure_prefix_path.stem}_low_motion"
                                f"{failure_prefix_path.suffix or '.npz'}"
                            )
                        )
                        evidence_start = max(first_start - 1, 0)
                        evidence_stop = min(first_end + 2, frame_count)
                        evidence_slice = slice(evidence_start, evidence_stop)
                        evidence_payload: dict[str, object] = {
                            "schema_version": np.asarray(1, np.int32),
                            "reason": np.asarray("unmarked_low_motion"),
                            "window_frames": np.asarray(
                                LOW_MOTION_DEFAULT_WINDOW_FRAMES, np.int32
                            ),
                            "forward_progress_ratio": np.asarray(
                                LOW_MOTION_FORWARD_PROGRESS_RATIO
                            ),
                            "region_frame_ranges": np.asarray(
                                [
                                    [
                                        int(region["frame_start"]),
                                        int(region["frame_end"]),
                                    ]
                                    for region in low_motion_regions
                                ],
                                dtype=np.int32,
                            ),
                            "first_window_frame_start": np.asarray(
                                first_start, np.int32
                            ),
                            "first_window_frame_end": np.asarray(
                                first_end, np.int32
                            ),
                            "first_window_route_delta_m": np.asarray(
                                first_window["route_delta_m"]
                            ),
                            "first_window_required_tip_progress_m": (
                                np.asarray(
                                    first_window["required_tip_progress_m"]
                                )
                            ),
                            "first_window_tip_progress_delta_m": (
                                first_tip_delta_m
                            ),
                            "first_window_tip_cartesian_delta_m": np.asarray(
                                first_window["tip_cartesian_delta_m"]
                            ),
                            "first_window_forward_mask": first_forward_mask,
                            "evidence_frame_start": np.asarray(
                                evidence_start, np.int32
                            ),
                            "evidence_frame_stop_exclusive": np.asarray(
                                evidence_stop, np.int32
                            ),
                            "evidence_joint_positions_rad": joint_plan[
                                evidence_slice
                            ],
                        }
                        for name, values in (
                            (
                                "frame_target_distance_m",
                                frame_target_distance,
                            ),
                            ("axial_distance_m", distance_plan),
                            ("kinematic_points_m", kinematic_plan),
                            ("progress_m", progress_plan),
                            ("static_bridge_mask", static_bridge_mask_plan),
                            ("recovery_bridge_mask", recovery_bridge_mask_plan),
                        ):
                            evidence_payload[f"evidence_{name}"] = values[
                                evidence_slice
                            ]
                        for name, values in (
                            ("distance_m", coarse_distance),
                            ("joint_positions_rad", coarse_q),
                            (
                                "static_bridge_mask",
                                coarse_static_feasibility_bridge,
                            ),
                            ("recovery_bridge_mask", coarse_recovery_bridge),
                        ):
                            evidence_payload[f"coarse_{name}"] = values
                        evidence_path = save_npz_no_overwrite(
                            low_motion_evidence_path,
                            evidence_payload,
                            field_label="Low-motion evidence",
                        )
                except Exception as exc:
                    evidence_error = f"{type(exc).__name__}: {exc}"
                evidence_summary = (
                    f"saved={evidence_path}"
                    if evidence_path is not None
                    else f"save_error={evidence_error}"
                )
                print(
                    "[MPC-LOW-MOTION-FAILURE] "
                    f"regions={len(low_motion_regions)} "
                    f"frame_window={first_start}:{first_end} "
                    "target_distance_m="
                    f"[{first_window['target_distance_start_m']:.9f},"
                    f"{first_window['target_distance_end_m']:.9f}] "
                    f"route_delta_mm="
                    f"{first_window['route_delta_m'] * 1000:.6f} "
                    f"required_tip_progress_mm="
                    f"{first_window['required_tip_progress_m'] * 1000:.6f} "
                    "tip_progress_delta_mm="
                    f"{(first_tip_delta_m * 1000).round(6).tolist()} "
                    f"forward_mask={first_forward_mask.tolist()} "
                    f"{evidence_summary}",
                    flush=True,
                )
                raise RuntimeError(
                    "Adaptive surface MPC rejected an unmarked low-motion "
                    "plan before dynamics: "
                    f"regions={len(low_motion_regions)} "
                    f"first_frame_window={first_start}:{first_end} "
                    f"forward_fingers="
                    f"{int(np.count_nonzero(first_forward_mask))}/4 "
                    f"evidence={evidence_path or evidence_error}"
                )

            planned_contact_ratio = np.mean(
                scheduled_contact_mask_plan,
                axis=0,
            )
            planned_majority_contact_ratio = float(
                np.mean(
                    scheduled_contact_count_plan
                    >= args.min_runtime_contact_fingers
                )
            )
            planned_average_contact_fingers = float(
                np.mean(scheduled_contact_count_plan)
            )
            if (
                planned_majority_contact_ratio
                < args.min_majority_contact_ratio
            ):
                raise RuntimeError(
                    "Adaptive surface MPC planned majority-contact ratio "
                    "below the route-level requirement: "
                    f"observed={planned_majority_contact_ratio:.4f} "
                    f"required={args.min_majority_contact_ratio:.4f}"
                )
            if (
                planned_average_contact_fingers
                < args.min_average_contact_fingers
            ):
                raise RuntimeError(
                    "Adaptive surface MPC planned average contact count "
                    "below the route-level requirement: "
                    f"observed={planned_average_contact_fingers:.4f} "
                    f"required={args.min_average_contact_fingers:.4f}"
                )
            if np.any(planned_contact_ratio < args.min_contact_ratio):
                raise RuntimeError(
                    "Adaptive surface MPC planned per-finger contact ratio "
                    "below the route-level requirement: "
                    f"observed={planned_contact_ratio.round(4).tolist()} "
                    f"required={args.min_contact_ratio:.4f}"
                )
            terminal_contact_frames = min(
                args.final_contact_recovery_frames,
                frame_count,
            )
            if (
                terminal_contact_frames < args.final_contact_recovery_frames
                or not np.all(
                    scheduled_contact_count_plan[-terminal_contact_frames:]
                    == 4
                )
            ):
                raise RuntimeError(
                    "Adaptive surface MPC did not restore four-fingertip "
                    "terminal contact: "
                    f"required_frames={args.final_contact_recovery_frames}"
                )

            final_target_progress = coarse_target_progress[-1]
            final_progress_error = np.abs(
                progress_plan[-1] - final_target_progress
            )
            final_progress_error[0] = 0.0
            if float(final_progress_error.max()) > (
                args.mpc_progress_tolerance_mm / 1000.0
            ):
                raise RuntimeError(
                    "Adaptive surface MPC interpolation failed final progress: "
                    f"error_mm="
                    f"{(final_progress_error * 1000).round(2).tolist()}"
                )
            q_with_seed = np.vstack((start_q[None], joint_plan))
            max_joint_step = float(np.max(np.abs(np.diff(q_with_seed, axis=0))))
            if max_joint_step > args.max_plan_joint_step_rad:
                raise RuntimeError(
                    "Adaptive surface MPC joint step exceeds bound: "
                    f"observed={max_joint_step:.5f}rad "
                    f"limit={args.max_plan_joint_step_rad:.5f}rad"
                )
            max_palm_position_error = float(
                palm_position_error_plan.max()
            )
            if (
                max_palm_position_error
                > palm_tracking_limit_m
            ):
                raise RuntimeError(
                    "Adaptive surface MPC interpolation exceeded the "
                    f"non-contact palm {'guide drift guard' if args.palm_guide_only else 'feasibility ball'}: "
                    f"error_mm={max_palm_position_error * 1000:.3f} "
                    f"limit_mm={palm_tracking_limit_m * 1000:.3f}"
                )

            self.plan_surface = surface_plan
            self.plan_kinematic = kinematic_plan
            self.plan_normals = normal_plan
            self.plan_q = joint_plan
            self.plan_residual = residual_plan
            self.plan_distance = distance_plan
            self.plan_direction = direction
            self.planned_axial_travel = float(distance_plan[-1])
            self._validate_full_robot_plan_clearance(
                center,
                rotation,
                joint_plan,
                label="adaptive_surface_mpc",
            )
            start_local = (rotation.T @ (surface_plan[0] - center).T).T
            end_local = (rotation.T @ (surface_plan[-1] - center).T).T
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.plan_output,
                surface_points_m=surface_plan,
                kinematic_points_m=kinematic_plan,
                joint_positions_rad=joint_plan,
                progress_m=progress_plan,
                progress_residual_m=residual_plan,
                normal_error_m=normal_error_plan,
                scheduled_contact_mask=scheduled_contact_mask_plan,
                scheduled_contact_count=scheduled_contact_count_plan,
                recovery_bridge_mask=recovery_bridge_mask_plan,
                planned_contact_ratio=planned_contact_ratio,
                planned_majority_contact_ratio=np.asarray(
                    planned_majority_contact_ratio
                ),
                planned_average_contact_fingers=np.asarray(
                    planned_average_contact_fingers
                ),
                min_planner_contact_fingers=np.asarray(
                    args.min_planner_contact_fingers
                ),
                transient_contact_finger=np.asarray(
                    args.transient_contact_finger
                ),
                transient_contact_start_m=np.asarray(
                    args.transient_contact_start_m
                ),
                transient_contact_end_m=np.asarray(
                    args.transient_contact_end_m
                ),
                transient_contact_recovery_start_m=np.asarray(
                    np.nan
                    if args.transient_contact_recovery_start_m is None
                    else args.transient_contact_recovery_start_m
                ),
                transient_progress_recovery_end_m=np.asarray(
                    np.nan
                    if args.transient_progress_recovery_end_m is None
                    else args.transient_progress_recovery_end_m
                ),
                transient_contact_normal_recovery_start_m=np.asarray(
                    np.nan
                    if args.transient_contact_normal_recovery_start_m is None
                    else args.transient_contact_normal_recovery_start_m
                ),
                mpc_transient_normal_tolerance_mm=np.asarray(
                    args.mpc_transient_normal_tolerance_mm
                ),
                mpc_transient_tangential_tolerance_mm=np.asarray(
                    np.nan
                    if args.mpc_transient_tangential_tolerance_mm is None
                    else args.mpc_transient_tangential_tolerance_mm
                ),
                mpc_transient_progress_tolerance_mm=np.asarray(
                    args.mpc_transient_progress_tolerance_mm
                ),
                palm_guide_only=np.asarray(args.palm_guide_only),
                palm_guide_max_drift_mm=np.asarray(
                    args.palm_guide_max_drift_mm
                ),
                mpc_palm_position_tolerance_mm=np.asarray(
                    args.mpc_palm_position_tolerance_mm
                ),
                mpc_base_keyframes=np.asarray(base_keyframe_count),
                mpc_actual_keyframes=np.asarray(keyframe_count),
                mpc_coarse_distance_m=coarse_distance,
                mpc_coarse_auto_rephase_offset_m=(
                    coarse_auto_rephase_offset_m
                ),
                mpc_coarse_feasibility_bridge=(
                    coarse_feasibility_bridge
                ),
                mpc_coarse_suffix_horizon=coarse_suffix_horizon,
                mpc_coarse_static_feasibility_bridge=(
                    coarse_static_feasibility_bridge
                ),
                mpc_coarse_static_bridge_dwell_m=(
                    coarse_static_bridge_dwell_m
                ),
                mpc_coarse_recovery_bridge=coarse_recovery_bridge,
                mpc_coarse_recovery_bridge_dwell_m=(
                    coarse_recovery_bridge_dwell_m
                ),
                mpc_local_refine_start_m=np.asarray(
                    args.mpc_local_refine_start_m
                ),
                mpc_local_refine_end_m=np.asarray(
                    args.mpc_local_refine_end_m
                ),
                mpc_local_refine_factor=np.asarray(
                    args.mpc_local_refine_factor
                ),
                mpc_local_refine_windows=np.asarray(
                    args.mpc_local_refine_window,
                    dtype=np.float64,
                ).reshape(-1, 3),
                mpc_auto_rephase_max_mm=np.asarray(
                    args.mpc_auto_rephase_max_mm
                ),
                mpc_feasibility_bridge_max_mm=np.asarray(
                    args.mpc_feasibility_bridge_max_mm
                ),
                mpc_feasibility_bridge_trust_radius_rad=np.asarray(
                    args.mpc_feasibility_bridge_trust_radius_rad
                ),
                mpc_feasibility_bridge_min_progress_ratio=np.asarray(
                    args.mpc_feasibility_bridge_min_progress_ratio
                ),
                mpc_feasibility_bridge_target_weight=np.asarray(
                    args.mpc_feasibility_bridge_target_weight
                ),
                mpc_feasibility_bridge_tip_target_scale=np.asarray(
                    args.mpc_feasibility_bridge_tip_target_scale
                ),
                mpc_suffix_horizon_nodes=np.asarray(
                    args.mpc_suffix_horizon_nodes
                ),
                mpc_suffix_min_joint_margin_mrad=np.asarray(
                    args.mpc_suffix_min_joint_margin_mrad
                ),
                mpc_suffix_min_task_margin_mm=np.asarray(
                    args.mpc_suffix_min_task_margin_mm
                ),
                mpc_suffix_max_nfev=np.asarray(
                    args.mpc_suffix_max_nfev
                ),
                mpc_suffix_horizon_attempt_count=np.asarray(
                    suffix_horizon_attempt_count
                ),
                mpc_suffix_horizon_success_count=np.asarray(
                    suffix_horizon_success_count
                ),
                mpc_suffix_terminal_start_m=np.asarray(
                    suffix_terminal_start_m
                ),
                mpc_static_bridge_max_dwell_mm=np.asarray(
                    args.mpc_static_bridge_max_dwell_mm
                ),
                mpc_static_bridge_max_total_ratio=np.asarray(
                    args.mpc_static_bridge_max_total_ratio
                ),
                mpc_static_bridge_progress_tolerance_mm=np.asarray(
                    args.mpc_static_bridge_progress_tolerance_mm
                ),
                mpc_static_bridge_total_m=np.asarray(static_bridge_total_m),
                mpc_recovery_bridge_max_span_mm=np.asarray(
                    args.mpc_recovery_bridge_max_span_mm
                ),
                mpc_recovery_bridge_max_total_ratio=np.asarray(
                    args.mpc_recovery_bridge_max_total_ratio
                ),
                mpc_recovery_bridge_progress_tolerance_mm=np.asarray(
                    args.mpc_recovery_bridge_progress_tolerance_mm
                ),
                mpc_recovery_bridge_normal_tolerance_mm=np.asarray(
                    args.mpc_recovery_bridge_normal_tolerance_mm
                ),
                mpc_recovery_bridge_min_contact_fingers=np.asarray(
                    args.mpc_recovery_bridge_min_contact_fingers
                ),
                mpc_recovery_bridge_terminal_margin_mm=np.asarray(
                    args.mpc_recovery_bridge_terminal_margin_mm
                ),
                mpc_recovery_bridge_total_m=np.asarray(
                    recovery_bridge_total_m
                ),
                mpc_auto_rephase_step_mm=np.asarray(
                    args.mpc_auto_rephase_step_mm
                ),
                mpc_auto_rephase_decay_mm=np.asarray(
                    args.mpc_auto_rephase_decay_mm
                ),
                mpc_auto_rephase_margin_mm=np.asarray(
                    args.mpc_auto_rephase_margin_mm
                ),
                mpc_auto_refine_min_step_mm=np.asarray(
                    args.mpc_auto_refine_min_step_mm
                ),
                mpc_auto_refine_max_insertions=np.asarray(
                    args.mpc_auto_refine_max_insertions
                ),
                mpc_auto_refine_inserted_distance_m=np.asarray(
                    auto_refine_inserted_distance_m,
                    dtype=np.float64,
                ),
                mpc_auto_refine_inserted_reason=np.asarray(
                    auto_refine_inserted_reason,
                    dtype=np.str_,
                ),
                min_runtime_contact_fingers=np.asarray(
                    args.min_runtime_contact_fingers
                ),
                max_individual_contact_loss_frames=np.asarray(
                    args.max_individual_contact_loss_frames
                ),
                final_contact_recovery_frames=np.asarray(
                    args.final_contact_recovery_frames
                ),
                min_arm_clearance_mm=np.asarray(
                    args.min_arm_clearance_mm
                ),
                max_incidental_hand_penetration_mm=np.asarray(
                    args.max_incidental_hand_penetration_mm
                ),
                max_contact_penetration_mm=np.asarray(
                    args.max_contact_penetration_mm
                ),
                planner_soft_pad_angle_deg=np.asarray(
                    args.planner_soft_pad_angle_deg
                ),
                planner_soft_pad_weight=np.asarray(
                    args.planner_soft_pad_weight
                ),
                planner_soft_pad_softplus_tau=np.asarray(
                    args.planner_soft_pad_softplus_tau
                ),
                planner_tip_geom_target_mm=np.asarray(
                    args.planner_tip_geom_target_mm,
                    dtype=np.float64,
                ),
                planner_tip_geom_weight=np.asarray(
                    args.planner_tip_geom_weight
                ),
                planner_tip_geom_inner_cap_mm=np.asarray(
                    args.planner_tip_geom_inner_cap_mm
                ),
                planner_tip_geom_inner_weight=np.asarray(
                    args.planner_tip_geom_inner_weight
                ),
                planner_protected_self_clearance_mm=np.asarray(
                    args.planner_protected_self_clearance_mm
                ),
                planner_protected_self_clearance_weight=np.asarray(
                    args.planner_protected_self_clearance_weight
                ),
                planner_self_separation_seed_step_rad=np.asarray(
                    args.planner_self_separation_seed_step_rad
                ),
                planner_protected_self_pair_names=np.asarray(
                    protected_self_pair_labels,
                    dtype=np.str_,
                ),
                planned_tip_geom_minimum_clearance_m=(
                    self.min_planned_tip_clearance_m
                ),
                planned_tip_geom_minimum_clearance_frame=(
                    self.min_planned_tip_clearance_frame
                ),
                planned_protected_self_minimum_clearance_m=np.asarray(
                    self.min_planned_protected_self_clearance_m
                ),
                planned_protected_self_minimum_clearance_frame=np.asarray(
                    self.min_planned_protected_self_clearance_frame
                ),
                planned_protected_self_nearest_pair=np.asarray(
                    self.min_planned_protected_self_pair_name
                ),
                max_incidental_hand_contact_force_n=np.asarray(
                    args.max_incidental_hand_contact_force_n
                ),
                max_incidental_hand_total_force_n=np.asarray(
                    args.max_incidental_hand_total_force_n
                ),
                axial_distance_m=distance_plan,
                axial_direction=np.asarray(direction),
                planner=np.asarray(args.planner),
                seed=np.asarray(args.seed),
                planner_state_quantization_rad=np.asarray(
                    args.planner_state_quantization_rad
                ),
                surface_preload_mm=np.asarray(args.surface_preload_mm),
                palm_travel_ratio=np.asarray(args.palm_travel_ratio),
                palm_clearance_use_local_normal=np.asarray(
                    args.palm_clearance_use_local_normal
                ),
                palm_surface_frame_gain=np.asarray(
                    args.palm_surface_frame_gain
                ),
                palm_surface_frame_late_gain=np.asarray(
                    np.nan
                    if args.palm_surface_frame_late_gain is None
                    else args.palm_surface_frame_late_gain
                ),
                palm_surface_frame_late_start_m=np.asarray(
                    args.palm_surface_frame_late_start_m
                ),
                palm_surface_frame_late_ramp_m=np.asarray(
                    args.palm_surface_frame_late_ramp_m
                ),
                palm_surface_frame_terminal_gain=np.asarray(
                    np.nan
                    if args.palm_surface_frame_terminal_gain is None
                    else args.palm_surface_frame_terminal_gain
                ),
                palm_surface_frame_terminal_start_m=np.asarray(
                    args.palm_surface_frame_terminal_start_m
                ),
                palm_surface_frame_terminal_ramp_m=np.asarray(
                    args.palm_surface_frame_terminal_ramp_m
                ),
                palm_terminal_local_offset_mm=np.asarray(
                    args.palm_terminal_local_offset_mm,
                    dtype=np.float64,
                ),
                palm_terminal_local_offset_start_m=np.asarray(
                    args.palm_terminal_local_offset_start_m
                ),
                palm_terminal_local_offset_ramp_m=np.asarray(
                    args.palm_terminal_local_offset_ramp_m
                ),
                palm_terminal_second_local_offset_mm=np.asarray(
                    args.palm_terminal_second_local_offset_mm,
                    dtype=np.float64,
                ),
                palm_terminal_second_local_offset_start_m=np.asarray(
                    args.palm_terminal_second_local_offset_start_m
                ),
                palm_terminal_second_local_offset_ramp_m=np.asarray(
                    args.palm_terminal_second_local_offset_ramp_m
                ),
                finger_gait_amplitude_m=np.asarray(
                    args.finger_gait_amplitude_m
                ),
                finger_meridian_gait_mm=np.asarray(
                    args.finger_meridian_gait_mm
                ),
                finger_meridian_gait_start_m=np.asarray(
                    args.finger_meridian_gait_start_m
                ),
                finger_meridian_gait_end_m=np.asarray(
                    args.finger_meridian_gait_end_m
                ),
                finger_meridian_gait_scales=np.asarray(
                    args.finger_meridian_gait_scales
                ),
                finger_meridian_correction_mm=np.asarray(
                    args.finger_meridian_correction_mm
                ),
                finger_meridian_correction_start_m=np.asarray(
                    args.finger_meridian_correction_start_m
                ),
                finger_meridian_correction_end_m=np.asarray(
                    args.finger_meridian_correction_end_m
                ),
                finger_meridian_correction_scales=np.asarray(
                    args.finger_meridian_correction_scales
                ),
                finger_meridian_terminal_correction_mm=np.asarray(
                    args.finger_meridian_terminal_correction_mm
                ),
                finger_meridian_terminal_correction_start_m=np.asarray(
                    args.finger_meridian_terminal_correction_start_m
                ),
                finger_meridian_terminal_correction_end_m=np.asarray(
                    args.finger_meridian_terminal_correction_end_m
                ),
                finger_meridian_terminal_correction_scales=np.asarray(
                    args.finger_meridian_terminal_correction_scales
                ),
                finger_meridian_terminal_tail_correction_mm=np.asarray(
                    args.finger_meridian_terminal_tail_correction_mm
                ),
                finger_meridian_terminal_tail_correction_start_m=np.asarray(
                    args.finger_meridian_terminal_tail_correction_start_m
                ),
                finger_meridian_terminal_tail_correction_end_m=np.asarray(
                    args.finger_meridian_terminal_tail_correction_end_m
                ),
                finger_meridian_terminal_tail_correction_scales=np.asarray(
                    args.finger_meridian_terminal_tail_correction_scales
                ),
                finger_meridian_local_phase=np.asarray(
                    args.finger_meridian_local_phase,
                    dtype=np.float64,
                ).reshape((-1, 7)),
                object_shape=np.asarray(args.object_shape),
                object_radius_m=np.asarray(CAPSULE_RADIUS),
                object_half_height_m=np.asarray(CAPSULE_HALF_HEIGHT),
                max_joint_step_rad=np.asarray(max_joint_step),
                coarse_joint_positions_rad=coarse_q,
                coarse_progress_m=coarse_progress,
                coarse_normal_error_m=coarse_normal_error,
                coarse_palm_target_m=coarse_palm_target,
                coarse_palm_position_error_m=coarse_palm_position_error,
                palm_position_error_m=palm_position_error_plan,
                coarse_cost=coarse_cost,
                coarse_nfev=coarse_nfev,
                start_surface_local_m=start_local,
                end_surface_local_m=end_local,
            )
            print(
                "[AXIAL-PLAN] adaptive surface MPC passed | "
                f"frames={frame_count} "
                f"keyframes={keyframe_count} "
                f"min_travel_m={self.planned_axial_travel:.4f} "
                f"per_contact_travel_m={progress_plan[-1].round(4).tolist()} "
                f"max_joint_step_rad={max_joint_step:.5f} "
                f"max_progress_error_mm="
                f"{float(residual_plan.max() * 1000):.2f} "
                f"max_tip_normal_error_mm="
                f"{float(normal_error_plan[:, 1:].max() * 1000):.2f} "
                f"min_scheduled_contacts="
                f"{int(scheduled_contact_count_plan.min())}/4 "
                f"planned_majority_contact_ratio="
                f"{planned_majority_contact_ratio:.4f} "
                f"planned_average_contacts="
                f"{planned_average_contact_fingers:.3f}/4 "
                f"recovery_frames="
                f"{int(np.count_nonzero(recovery_bridge_mask_plan))}/"
                f"{frame_count} "
                f"max_palm_position_error_mm="
                f"{max_palm_position_error * 1000:.2f} "
                f"min_arm_clearance_mm="
                f"{self.min_planned_arm_clearance_m * 1000:.2f} "
                f"min_hand_clearance_mm="
                f"{self.min_planned_hand_clearance_m * 1000:.2f} "
                f"start_z_m={start_local[:, 2].round(4).tolist()} "
                f"end_z_m={end_local[:, 2].round(4).tolist()} "
                f"saved={args.plan_output.resolve()}",
                flush=True,
            )

        def _advance_axial_plan(self) -> None:
            assert self.plan_surface is not None
            assert self.plan_kinematic is not None
            assert self.plan_normals is not None
            assert self.plan_q is not None
            assert self.plan_residual is not None
            assert self.plan_distance is not None
            index = min(
                max(self.step - args.motion_start, 0),
                self.plan_surface.shape[0] - 1,
            )
            self.plan_index = index
            self.targets = self.plan_surface[index]
            self.kinematic_targets = self.plan_kinematic[index]
            self.normals = self.plan_normals[index]
            self.reachable_q = self.plan_q[index]
            self.last_residual = self.plan_residual[index]
            self.executed_axial_travel = float(self.plan_distance[index])

        def _apply_runtime_finger_gait(
            self,
            center: np.ndarray,
            rotation: np.ndarray,
        ) -> None:
            """Add visible per-finger motion without violating the URDF.

            The main arm trajectory remains the validated end-to-end plan.
            Each fingertip target is regenerated by forward kinematics after
            a slow joint-space gait, so the five points always correspond to
            one real 23-DoF robot configuration.
            """

            if (
                args.runtime_finger_gait_rad <= 0.0
                or self.plan_q is None
                or self.plan_index < 0
            ):
                return
            phase = self.plan_index / float(
                max(self.plan_q.shape[0] - 1, 1)
            )
            gait_m = (
                args.runtime_tip_gait_mm
                / 1000.0
                * np.sin(
                    2.0
                    * np.pi
                    * args.runtime_gait_cycles
                    * phase
                )
            )
            gait_pattern = np.asarray((1.0, -1.0, 1.0, -1.0))
            q = self.reachable_q.copy()
            baseline_points = reachability.forward_points(q)
            _, baseline_normals = capsule_project(
                baseline_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            jacobian = reachability._stacked_jacobian()
            capsule_axis = rotation[:, 2]
            for finger, (sign, gait_scale) in enumerate(
                zip(
                    gait_pattern,
                    args.runtime_gait_finger_scales,
                    strict=True,
                )
            ):
                normal = baseline_normals[finger + 1]
                tangent = np.cross(capsule_axis, normal)
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm < 1.0e-8:
                    continue
                tangent /= tangent_norm
                row = 3 * (finger + 1)
                col = ARM_DOF + 4 * finger
                finger_jacobian = jacobian[
                    row : row + 3, col : col + 4
                ]
                target_displacement = (
                    sign * gait_scale * gait_m * tangent
                )
                lhs = (
                    finger_jacobian @ finger_jacobian.T
                    + 1.0e-5 * np.eye(3)
                )
                dq = finger_jacobian.T @ np.linalg.solve(
                    lhs, target_displacement
                )
                dq = np.clip(
                    dq,
                    -args.runtime_finger_gait_rad,
                    args.runtime_finger_gait_rad,
                )
                q[col : col + 4] += dq
            q = np.minimum(
                np.maximum(q, reachability.lower),
                reachability.upper,
            )
            points = reachability.forward_points(q)
            surface, normals = capsule_project(
                points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            # The palm root is a planning point, not a physical contact.
            surface[0] = self.targets[0]
            normals[0] = self.normals[0]
            points[0] = self.kinematic_targets[0]
            self.reachable_q = q
            self.kinematic_targets = points
            self.targets = surface
            self.normals = normals
            self.last_residual = np.zeros(5)

        def __call__(self, obs):
            if self.targets is None:
                self._initialize(obs)
            self._update_object_approach()
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.normals is not None
            assert self.reachable_q is not None
            live_q = obs["palm"][0, :TOTAL_DOF].detach().cpu().numpy()
            self.tactile_force = (
                torch.linalg.vector_norm(
                    obs["finger"][0, :12].reshape(4, 3), dim=-1
                )
                .detach()
                .cpu()
                .numpy()
            )
            if self.contact_calibrated:
                self.max_tactile_force = np.maximum(
                    self.max_tactile_force, self.tactile_force
                )
                if bool(
                    np.any(
                        self.tactile_force
                        > args.max_tip_raw_force_n
                    )
                ):
                    raise RuntimeError(
                        "Raw fingertip force exceeded emergency limit "
                        f"{args.max_tip_raw_force_n:.2f}N: "
                        f"{self.tactile_force.round(3).tolist()}"
                    )
            for finger, site_name in enumerate(MCC_TIP_NAMES):
                sensor_data = env.scene[f"{site_name}_contact"].data
                if (
                    sensor_data.found is not None
                    and sensor_data.dist is not None
                    and bool(sensor_data.found[0, 0].item())
                ):
                    distance = float(sensor_data.dist[0, 0].item())
                    self.contact_distance_m[finger] = distance
                    self.max_penetration_m[finger] = max(
                        self.max_penetration_m[finger],
                        max(-distance, 0.0),
                    )
                    if sensor_data.pos is not None:
                        self.actual_contact_points[finger] = (
                            sensor_data.pos[0, 0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    self.contact_distance_m[finger] = np.nan
            incidental_hand_contact_active = False
            if args.collision_mode == "full_robot":
                arm_guard = env.scene["arm_object_collision"].data
                if (
                    arm_guard.found is not None
                    and bool(torch.any(arm_guard.found[0] > 0).item())
                ):
                    self.arm_collision_frames += 1
                    guard_distance_mm = (
                        arm_guard.dist[0].detach().cpu().numpy() * 1000.0
                        if arm_guard.dist is not None
                        else np.asarray([])
                    )
                    raise RuntimeError(
                        "Arm-object collision guard triggered: an FR3 "
                        "link0..link7 collision geom touched the target. "
                        f"contact_slots="
                        f"{arm_guard.found[0].detach().cpu().numpy().tolist()} "
                        f"dist_mm={guard_distance_mm.round(3).tolist()}"
                    )
                hand_depth = env.scene[
                    "incidental_hand_object_contact_depth"
                ].data
                hand_force = env.scene[
                    "incidental_hand_object_contact_force"
                ].data
                depth_found = (
                    hand_depth.found is not None
                    and bool(torch.any(hand_depth.found[0] > 0).item())
                )
                force_found = (
                    hand_force.found is not None
                    and bool(torch.any(hand_force.found[0] > 0).item())
                )
                incidental_hand_contact_active = depth_found or force_found
                if incidental_hand_contact_active:
                    self.incidental_hand_contact_frames += 1
                    self.incidental_hand_contact_streak += 1
                    self.max_incidental_hand_contact_streak = max(
                        self.max_incidental_hand_contact_streak,
                        self.incidental_hand_contact_streak,
                    )
                    deepest_penetration_m = 0.0
                    hand_distance_mm = np.asarray([])
                    if hand_depth.dist is not None:
                        hand_distance = (
                            hand_depth.dist[0].detach().cpu().numpy()
                        )
                        hand_distance_mm = hand_distance * 1000.0
                        if hand_depth.found is not None:
                            depth_mask = (
                                hand_depth.found[0]
                                .detach()
                                .cpu()
                                .numpy()
                                > 0
                            )
                            if bool(np.any(depth_mask)):
                                deepest_penetration_m = max(
                                    -float(np.min(hand_distance[depth_mask])),
                                    0.0,
                                )
                    max_force_n = 0.0
                    total_force_n = 0.0
                    force_magnitudes = np.asarray([])
                    if hand_force.force is not None:
                        force_vectors = (
                            hand_force.force[0].detach().cpu().numpy()
                        )
                        force_magnitudes = np.linalg.norm(
                            force_vectors,
                            axis=-1,
                        )
                        if hand_force.found is not None:
                            force_mask = (
                                hand_force.found[0]
                                .detach()
                                .cpu()
                                .numpy()
                                > 0
                            )
                            active_forces = force_magnitudes[force_mask]
                            if active_forces.size:
                                max_force_n = float(active_forces.max())
                                total_force_n = float(active_forces.sum())
                    self.max_incidental_hand_penetration_m = max(
                        self.max_incidental_hand_penetration_m,
                        deepest_penetration_m,
                    )
                    self.max_incidental_hand_contact_force_n = max(
                        self.max_incidental_hand_contact_force_n,
                        max_force_n,
                    )
                    self.max_incidental_hand_total_force_n = max(
                        self.max_incidental_hand_total_force_n,
                        total_force_n,
                    )
                    if (
                        deepest_penetration_m * 1000.0
                        > args.max_incidental_hand_penetration_mm
                    ):
                        raise RuntimeError(
                            "Allowed LEAP Hand/object contact exceeded the "
                            "penetration limit: deepest_mm="
                            f"{deepest_penetration_m * 1000:.3f} "
                            f"limit_mm="
                            f"{args.max_incidental_hand_penetration_mm:.3f} "
                            f"dist_mm={hand_distance_mm.round(3).tolist()}"
                        )
                    if (
                        max_force_n
                        > args.max_incidental_hand_contact_force_n
                        or total_force_n
                        > args.max_incidental_hand_total_force_n
                    ):
                        raise RuntimeError(
                            "Allowed LEAP Hand/object contact force became "
                            "too large: max_per_geom_N="
                            f"{max_force_n:.3f} limit_per_geom_N="
                            f"{args.max_incidental_hand_contact_force_n:.3f} "
                            f"total_N={total_force_n:.3f} "
                            "limit_total_N="
                            f"{args.max_incidental_hand_total_force_n:.3f} "
                            f"per_geom_N="
                            f"{force_magnitudes.round(3).tolist()}"
                        )
                else:
                    self.incidental_hand_contact_streak = 0
            if (
                not self.contact_calibrated
                and self.step
                >= args.preshape_frames + args.object_approach_frames
                and bool(
                    np.any(self.tactile_force < args.min_contact_force_n)
                )
            ):
                # Search along each physical pad's object normal.  A generic
                # positive-flexion "natural closure" can move curved LeapHand
                # fingertips away from the capsule and was the source of the
                # earlier visually invalid grasp.
                search_q = self.reachable_q.copy()
                search_q[ARM_DOF:TOTAL_DOF] += self.precontact_closure
                search_points = reachability.forward_points(search_q)
                search_surface, search_normals = capsule_project(
                    search_points,
                    *self._object_pose(obs),
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                _ = search_surface
                stacked_jacobian = reachability._stacked_jacobian()
                for finger, tactile_force in enumerate(self.tactile_force):
                    if tactile_force >= args.min_contact_force_n:
                        continue
                    row = 3 * (finger + 1)
                    col = ARM_DOF + 4 * finger
                    finger_jacobian = stacked_jacobian[
                        row : row + 3,
                        col : col + 4,
                    ]
                    target_displacement = (
                        -args.contact_search_step_mm
                        / 1000.0
                        * search_normals[finger + 1]
                    )
                    lhs = (
                        finger_jacobian @ finger_jacobian.T
                        + 1.0e-5 * np.eye(3)
                    )
                    delta_q = finger_jacobian.T @ np.linalg.solve(
                        lhs,
                        target_displacement,
                    )
                    delta_q = np.clip(
                        delta_q,
                        -args.contact_search_step_rad,
                        args.contact_search_step_rad,
                    )
                    base = 4 * finger
                    self.precontact_closure[base : base + 4] = np.clip(
                        self.precontact_closure[base : base + 4] + delta_q,
                        -args.contact_search_limit_rad,
                        args.contact_search_limit_rad,
                    )
            self.actual_points = reachability.forward_points(live_q)
            if (
                not self.contact_calibrated
                and self.step >= args.contact_calibration_start
            ):
                if bool(np.all(self.tactile_force >= args.min_contact_force_n)):
                    self.contact_settle_streak += 1
                else:
                    self.contact_settle_streak = 0
                if self.contact_settle_streak >= args.contact_settle_frames:
                    self._capture_contact_calibration(
                        obs, live_q, self.actual_points
                    )
            if self.step >= args.motion_start:
                if not self.contact_calibrated:
                    raise RuntimeError(
                        "All four fingertips did not establish real contact "
                        "before surface sliding"
                    )
                self._advance_axial_plan()
                center, rotation = self._object_pose(obs)
                self._apply_runtime_finger_gait(center, rotation)
            elif (
                self.contact_calibrated
                and not self.fingertip_force_recalibrated
                and self.step == args.motion_start - 1
            ):
                settled_direct_normal_force = controller.last_debug[
                    "tip_normal_force_signed_raw"
                ][0].detach().cpu().numpy()
                controller.fingers.calibrate_fingertip_force_setpoint(
                    settled_direct_normal_force
                )
                controller.calibrate_arm_force_setpoint(obs["palm"])
                self.fingertip_force_recalibrated = True
                print(
                    "[FINGERTIP-FORCE-RECALIBRATION] settled direct sensor "
                    f"raw_normal_N="
                    f"{settled_direct_normal_force.round(2).tolist()} "
                    f"target_N="
                    f"{controller.fingers.fingertip_force_setpoint.round(2).tolist()}",
                    flush=True,
                )
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            self.joint_error = self.reachable_q - live_q
            self.tracking_error = np.linalg.norm(
                self.actual_points - self.kinematic_targets, axis=1
            )
            center, rotation = self._object_pose(obs)
            projected_actual, _ = capsule_project(
                self.actual_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            self.surface_error = np.linalg.norm(
                self.actual_points - projected_actual, axis=1
            )
            if self.step >= args.motion_start:
                _, actual_surface_normals = capsule_project(
                    self.actual_points,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                actual_pad_normals = (
                    reachability.fingertip_pad_normals(live_q)
                )
                runtime_pad_alignment = np.einsum(
                    "ij,ij->i",
                    actual_pad_normals,
                    -actual_surface_normals[1:],
                )
                self.min_runtime_pad_alignment = min(
                    self.min_runtime_pad_alignment,
                    float(runtime_pad_alignment.min()),
                )
                required_pad_alignment = float(
                    np.cos(np.deg2rad(args.max_pad_angle_deg))
                )
                if float(runtime_pad_alignment.min()) < required_pad_alignment:
                    finger = int(np.argmin(runtime_pad_alignment))
                    raise RuntimeError(
                        "Runtime fingertip left the physical finger-pad "
                        "orientation cone: "
                        f"finger={finger} alignment="
                        f"{runtime_pad_alignment[finger]:.4f} "
                        f"angle_deg="
                        f"{np.degrees(np.arccos(np.clip(runtime_pad_alignment[finger], -1, 1))):.2f} "
                        f"limit_deg={args.max_pad_angle_deg:.2f}"
                    )
                runtime_self_pairs, runtime_self_distances = (
                    reachability.self_collision_contacts(live_q)
                )
                if runtime_self_pairs:
                    runtime_self_penetration_m = max(
                        -float(runtime_self_distances.min()),
                        0.0,
                    )
                    self.max_runtime_self_penetration_m = max(
                        self.max_runtime_self_penetration_m,
                        runtime_self_penetration_m,
                    )
                    self.runtime_self_near_contact_frames += 1
                    runtime_self_pair_names = [
                        (
                            reachability.model.geom(pair[0]).name,
                            reachability.model.geom(pair[1]).name,
                        )
                        for pair in runtime_self_pairs
                    ]
                    if (
                        runtime_self_penetration_m * 1000.0
                        > args.max_runtime_self_penetration_mm
                    ):
                        raise RuntimeError(
                            "Runtime robot self-collision exceeded numerical "
                            "penetration tolerance: "
                            f"pairs={len(runtime_self_pairs)} deepest_mm="
                            f"{runtime_self_distances.min() * 1000:.6f} "
                            f"limit_mm="
                            f"{args.max_runtime_self_penetration_mm:.6f} "
                            f"pair_names={runtime_self_pair_names} "
                            f"distances_mm="
                            f"{(runtime_self_distances * 1000).round(6).tolist()}"
                        )
                tip_contact = self.tactile_force >= args.min_contact_force_n
                self.contact_frames += tip_contact.astype(np.int64)
                self.evaluated_frames += 1
                if bool(np.all(np.isfinite(self.actual_points[1:]))):
                    contact_surface, _ = capsule_project(
                        self.actual_points[1:],
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                    self.contact_current_arc, _ = (
                        capsule_meridian_coordinates(
                            contact_surface,
                            center,
                            rotation,
                            CAPSULE_RADIUS,
                            CAPSULE_HALF_HEIGHT,
                        )
                    )
                    self.contact_surface_travel_m = np.maximum(
                        self.contact_surface_travel_m,
                        self.plan_direction
                        * (
                            self.contact_current_arc
                            - self.contact_start_arc
                        ),
                    )
                    palm_position, palm_rotation = (
                        reachability.forward_palm_pose(live_q)
                    )
                    self.contact_current_in_palm = (
                        palm_rotation.T
                        @ (
                            self.actual_points[1:]
                            - palm_position
                        ).T
                    ).T
                    self.contact_relative_travel_m = np.maximum(
                        self.contact_relative_travel_m,
                        np.linalg.norm(
                            self.contact_current_in_palm
                            - self.contact_start_in_palm,
                            axis=1,
                        ),
                    )
                finger_q = live_q[ARM_DOF:TOTAL_DOF].reshape(4, 4)
                self.finger_q_min = np.minimum(
                    self.finger_q_min, finger_q
                )
                self.finger_q_max = np.maximum(
                    self.finger_q_max, finger_q
                )
                simultaneous_contacts = int(np.count_nonzero(tip_contact))
                self.simultaneous_contact_sum += simultaneous_contacts
                self.min_simultaneous_contacts = min(
                    self.min_simultaneous_contacts,
                    simultaneous_contacts,
                )
                if (
                    simultaneous_contacts
                    >= args.min_runtime_contact_fingers
                ):
                    self.majority_contact_frames += 1
                if simultaneous_contacts == 0:
                    self.zero_contact_streak += 1
                else:
                    self.zero_contact_streak = 0
                self.max_zero_contact_streak = max(
                    self.max_zero_contact_streak,
                    self.zero_contact_streak,
                )
                if (
                    self.zero_contact_streak
                    > args.max_zero_contact_frames
                ):
                    raise RuntimeError(
                        "All fingertip-pad contacts were lost beyond the "
                        "allowed brief release window: "
                        f"zero_contact_streak={self.zero_contact_streak} "
                        f"limit={args.max_zero_contact_frames} "
                        f"tactile_force_N="
                        f"{self.tactile_force.round(2).tolist()}"
                    )
                if incidental_hand_contact_active:
                    self.incidental_hand_contact_evaluated_frames += 1
                    self.min_tip_contacts_during_incidental_contact = min(
                        self.min_tip_contacts_during_incidental_contact,
                        simultaneous_contacts,
                    )
                self.contact_loss_streak = np.where(
                    tip_contact,
                    0,
                    self.contact_loss_streak + 1,
                )
                self.max_contact_loss_streak = np.maximum(
                    self.max_contact_loss_streak,
                    self.contact_loss_streak,
                )
                if bool(np.all(tip_contact)):
                    self.final_all_contact_streak += 1
                else:
                    self.final_all_contact_streak = 0
                if (
                    simultaneous_contacts
                    >= args.min_runtime_contact_fingers
                ):
                    self.bad_contact_streak = 0
                else:
                    self.bad_contact_streak += 1
                self.max_bad_contact_streak = max(
                    self.max_bad_contact_streak,
                    self.bad_contact_streak,
                )
                if self.bad_contact_streak > args.contact_failure_window:
                    raise RuntimeError(
                        "Minimum simultaneous fingertip support failed: "
                        f"contacts={simultaneous_contacts}/4 "
                        f"required={args.min_runtime_contact_fingers} "
                        f"site_standoff_mm="
                        f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                        f"kinematic_tracking_error_mm="
                        f"{(self.tracking_error[1:] * 1000).round(2).tolist()} "
                        f"tactile_force_N="
                        f"{self.tactile_force.round(2).tolist()} "
                        f"arm_joint_error_rad="
                        f"{self.joint_error[:ARM_DOF].round(3).tolist()} "
                        f"finger_joint_error_rad="
                        f"{self.joint_error[ARM_DOF:].round(3).tolist()}"
                    )
                if np.any(
                    self.contact_loss_streak
                    > args.max_individual_contact_loss_frames
                ):
                    raise RuntimeError(
                        "A swing fingertip did not recover contact within "
                        "the allowed window: "
                        f"current_loss_frames="
                        f"{self.contact_loss_streak.tolist()} "
                        f"max_loss_frames="
                        f"{args.max_individual_contact_loss_frames} "
                        f"tactile_force_N="
                        f"{self.tactile_force.round(2).tolist()}"
                    )
            target_t = torch.as_tensor(
                self.targets[None], device=device, dtype=torch.float32
            )
            normal_t = torch.as_tensor(
                self.normals[None], device=device, dtype=torch.float32
            )
            kinematic_target_t = torch.as_tensor(
                self.kinematic_targets[None], device=device, dtype=torch.float32
            )
            command_q = self.reachable_q.copy()
            if not self.contact_calibrated:
                command_q[ARM_DOF:TOTAL_DOF] += self.precontact_closure
                command_q = np.minimum(
                    np.maximum(command_q, reachability.lower),
                    reachability.upper,
                )
                command_q[ARM_DOF:TOTAL_DOF] = live_q[
                    ARM_DOF:TOTAL_DOF
                ] + np.clip(
                    command_q[ARM_DOF:TOTAL_DOF]
                    - live_q[ARM_DOF:TOTAL_DOF],
                    -args.contact_search_step_rad,
                    args.contact_search_step_rad,
                )
            else:
                command_q += self.contact_servo_offset_q
                command_q[:ARM_DOF] += (
                    args.arm_trajectory_tracking_gain
                    * (
                        self.reachable_q[:ARM_DOF]
                        - live_q[:ARM_DOF]
                    )
                )
                command_q[ARM_DOF:TOTAL_DOF] += (
                    args.finger_trajectory_tracking_gain
                    * (
                        self.reachable_q[ARM_DOF:TOTAL_DOF]
                        - live_q[ARM_DOF:TOTAL_DOF]
                    )
                )
                command_q = np.minimum(
                    np.maximum(command_q, reachability.lower),
                    reachability.upper,
                )
            self.last_command_q = command_q.copy()
            joint_reference_t = torch.as_tensor(
                command_q[None],
                device=device,
                dtype=torch.float32,
            )
            action = controller(
                obs,
                contact_points=target_t,
                surface_normals=normal_t,
                joint_reference=joint_reference_t,
                kinematic_points=kinematic_target_t,
            )
            if self.contact_calibrated:
                filtered_normal_force = (
                    controller.last_debug["tip_normal_force_filtered"][0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                self.max_filtered_normal_force = np.maximum(
                    self.max_filtered_normal_force,
                    np.abs(filtered_normal_force),
                )
                if bool(
                    np.any(
                        np.abs(filtered_normal_force)
                        > args.max_tip_contact_force_n
                    )
                ):
                    raise RuntimeError(
                        "Filtered fingertip normal force exceeded hard limit "
                        f"{args.max_tip_contact_force_n:.2f}N: "
                        f"{filtered_normal_force.round(3).tolist()}"
                    )
                force_correction = controller.last_debug[
                    "finger_force_joint_correction"
                ]
                self.max_fingertip_admittance_joint_correction_rad = max(
                    self.max_fingertip_admittance_joint_correction_rad,
                    float(torch.max(torch.abs(force_correction)).item()),
                )
                arm_force_correction = controller.last_debug[
                    "arm_force_joint_correction"
                ]
                self.max_arm_force_correction_rad = max(
                    self.max_arm_force_correction_rad,
                    float(
                        torch.max(
                            torch.abs(arm_force_correction)
                        ).item()
                    ),
                )
            if not self.contact_calibrated:
                # Before sensor-sign calibration, contact establishment uses
                # absolute position hold plus independent tactile search.
                action[:, :ARM_DOF] = joint_reference_t[:, :ARM_DOF]
                action[:, ARM_DOF:TOTAL_DOF] = joint_reference_t[
                    :, ARM_DOF:TOTAL_DOF
                ]
            if self.step % max(args.print_every, 1) == 0:
                debug = controller.last_debug
                direct_tip_force = debug["tip_normal_force_filtered"][0]
                motor_force_diagnostic = torch.linalg.vector_norm(
                    debug["tip_force_from_motors_diagnostic"][0], dim=-1
                )
                finger_offset_mm = (
                    debug["finger_normal_admittance_offset_m"][0] * 1000.0
                )
                wrist_offset_mm = (
                    torch.linalg.vector_norm(
                        debug["wrist_admittance_reference_offset"][0, :3]
                    )
                    * 1000.0
                )
                wrist_wrench_error = debug["arm_wrench_error"][0]
                print(
                    f"[FULL-HAND-MCC] step={self.step:05d} "
                    f"reachable_err_mm={(self.last_residual * 1000).round(2).tolist()} "
                    f"actual_tip_surface_mm="
                    f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                    f"actual_tip_kinematic_target_mm="
                    f"{(self.tracking_error[1:] * 1000).round(2).tolist()} "
                    f"max_joint_error_rad="
                    f"[{np.max(np.abs(self.joint_error[:ARM_DOF])):.3f},"
                    f"{np.max(np.abs(self.joint_error[ARM_DOF:])):.3f}] "
                    f"tip_force_N="
                    f"{direct_tip_force.cpu().numpy().round(2).tolist()} "
                    f"motor_force_diag_N="
                    f"{motor_force_diagnostic.cpu().numpy().round(2).tolist()} "
                    f"tactile_force_N={self.tactile_force.round(2).tolist()} "
                    f"contact_dist_mm="
                    f"{(self.contact_distance_m * 1000).round(3).tolist()} "
                    f"contact_surface_travel_m="
                    f"{self.contact_surface_travel_m.round(3).tolist()} "
                    f"tip_in_palm_travel_mm="
                    f"{(self.contact_relative_travel_m * 1000).round(1).tolist()} "
                    f"finger_admittance_offset_mm="
                    f"{finger_offset_mm.cpu().numpy().round(3).tolist()} "
                    f"wrist_admittance_offset_mm="
                    f"{float(wrist_offset_mm):.3f} "
                    f"wrist_wrench_error="
                    f"{wrist_wrench_error.cpu().numpy().round(2).tolist()} "
                    f"axial_travel_m={self.executed_axial_travel:.4f} "
                    f"plan_frame={self.plan_index}",
                    flush=True,
                )
            self.step += 1
            return action

    policy = SurfaceSlidePolicy()

    base_update_visualizers = env.update_visualizers

    def update_demo_visualizers(visualizer) -> None:
        base_update_visualizers(visualizer)
        if policy.targets is None or policy.normals is None:
            return
        colors = (
            (0.2, 1.0, 0.2, 0.95),
            (1.0, 0.3, 0.2, 0.95),
            (0.2, 0.7, 1.0, 0.95),
            (1.0, 0.8, 0.15, 0.95),
            (0.8, 0.3, 1.0, 0.95),
        )
        for point, normal, color in zip(policy.targets, policy.normals, colors):
            visualizer.add_sphere(point, radius=0.007, color=color)
            visualizer.add_arrow(
                point,
                point + 0.035 * normal,
                color=color,
                width=0.003,
            )
        for point, color in zip(
            policy.actual_contact_points, colors[1:]
        ):
            if np.all(np.isfinite(point)):
                visualizer.add_sphere(
                    point,
                    radius=0.009,
                    color=(color[0], color[1], color[2], 1.0),
                )

    env.update_visualizers = update_demo_visualizers
    print(
        f"[INFO] Baseline-2 full-hand MCC demo | "
        f"device={device} viewer={args.viewer} "
        f"object_shape={args.object_shape} "
        f"object_radius_m={CAPSULE_RADIUS:.4f} "
        f"object_half_height_m={CAPSULE_HALF_HEIGHT:.4f}"
    )
    print("[INFO] Targets are surface-projected and URDF/joint-limit checked every step")
    try:
        if args.viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        elif args.viewer == "viser":
            ViserPlayViewer(wrapped, policy).run()
        else:
            obs, _ = wrapped.reset()
            frames_written = 0
            if args.viewer == "headless":
                for step in range(args.steps):
                    action = policy(obs)
                    obs, _, _, _ = wrapped.step(action)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                next_frame_time = 0.0
                with imageio.get_writer(
                    args.output,
                    fps=args.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                ) as writer:
                    for step in range(args.steps):
                        action = policy(obs)
                        obs, _, _, _ = wrapped.step(action)
                        sim_time = (step + 1) * dt
                        if sim_time + 1.0e-9 >= next_frame_time:
                            frame = env.render()
                            if frame is not None:
                                writer.append_data(
                                    np.asarray(frame, dtype=np.uint8)
                                )
                                frames_written += 1
                            next_frame_time += 1.0 / args.fps
            if policy.evaluated_frames <= 0:
                raise RuntimeError("No sliding frames were evaluated for contact")
            if (
                policy.executed_axial_travel + 1.0e-6
                < 0.99 * args.axial_travel_m
            ):
                raise RuntimeError(
                    "End-to-end trajectory was not fully executed: "
                    f"executed={policy.executed_axial_travel:.4f}m "
                    f"requested={args.axial_travel_m:.4f}m"
                )
            contact_ratio = (
                policy.contact_frames / float(policy.evaluated_frames)
            )
            majority_contact_ratio = (
                policy.majority_contact_frames
                / float(policy.evaluated_frames)
            )
            average_simultaneous_contacts = (
                policy.simultaneous_contact_sum
                / float(policy.evaluated_frames)
            )
            if majority_contact_ratio < args.min_majority_contact_ratio:
                raise RuntimeError(
                    "Majority fingertip-contact ratio below required "
                    f"{args.min_majority_contact_ratio:.1%}: "
                    f"observed={majority_contact_ratio:.1%} "
                    f"majority_threshold="
                    f"{args.min_runtime_contact_fingers}/4"
                )
            if (
                average_simultaneous_contacts
                < args.min_average_contact_fingers
            ):
                raise RuntimeError(
                    "Average simultaneous fingertip contacts below required "
                    f"{args.min_average_contact_fingers:.2f}/4: "
                    f"observed={average_simultaneous_contacts:.3f}/4"
                )
            if (
                policy.final_all_contact_streak
                < args.final_contact_recovery_frames
            ):
                raise RuntimeError(
                    "The route ended before all four fingertips recovered "
                    "stable contact: "
                    f"final_all_contact_streak="
                    f"{policy.final_all_contact_streak} "
                    f"required={args.final_contact_recovery_frames}"
                )
            if np.any(contact_ratio < args.min_contact_ratio):
                raise RuntimeError(
                    "Fingertip continuous-contact ratio below required "
                    f"{args.min_contact_ratio:.1%}: "
                    f"{contact_ratio.round(4).tolist()}"
                )
            if np.any(
                policy.contact_surface_travel_m
                < args.min_tip_surface_travel_m
            ):
                raise RuntimeError(
                    "Measured physical fingertip-site surface travel is below "
                    f"{args.min_tip_surface_travel_m:.3f}m: "
                    f"{policy.contact_surface_travel_m.round(4).tolist()}"
                )
            if np.any(
                policy.contact_relative_travel_m
                < args.min_tip_relative_travel_m
            ):
                raise RuntimeError(
                    "Fingertip motion relative to the palm is too small; "
                    "this is rigid whole-hand transport, not active surface "
                    f"sliding: {policy.contact_relative_travel_m.round(4).tolist()}"
                )
            finger_joint_excursion = (
                policy.finger_q_max - policy.finger_q_min
            )
            per_finger_joint_excursion = np.max(
                finger_joint_excursion, axis=1
            )
            if np.any(
                per_finger_joint_excursion
                < args.min_finger_joint_excursion_rad
            ):
                raise RuntimeError(
                    "Active finger-joint excursion is below "
                    f"{args.min_finger_joint_excursion_rad:.3f}rad: "
                    f"{per_finger_joint_excursion.round(4).tolist()}"
                )
            max_penetration_mm = policy.max_penetration_m * 1000.0
            if np.any(max_penetration_mm > args.max_contact_penetration_mm):
                raise RuntimeError(
                    "Fingertip/object penetration exceeded required limit "
                    f"{args.max_contact_penetration_mm:.3f}mm: "
                    f"{max_penetration_mm.round(3).tolist()}"
                )
            if (
                args.collision_mode == "full_robot"
                and policy.arm_collision_frames > 0
            ):
                raise RuntimeError(
                    "Strict FR3/object collision was observed: "
                    f"arm_frames={policy.arm_collision_frames}"
                )
            run_summary = (
                f"saved={args.output.resolve()} frames={frames_written}"
                if args.viewer == "video"
                else "rendered=false frames=0"
            )
            print(
                f"[{args.viewer.upper()}] {run_summary} "
                f"duration_s={args.steps * dt:.2f} fps={args.fps:.1f} "
                f"collision_mode={args.collision_mode} "
                f"tip_contact_ratio={contact_ratio.round(4).tolist()} "
                f"majority_contact_ratio={majority_contact_ratio:.4f} "
                "average_simultaneous_contacts="
                f"{average_simultaneous_contacts:.4f}/4 "
                f"min_simultaneous_contacts="
                f"{policy.min_simultaneous_contacts}/4 "
                "max_below_majority_streak_frames="
                f"{policy.max_bad_contact_streak} "
                "max_zero_contact_streak_frames="
                f"{policy.max_zero_contact_streak} "
                f"max_contact_loss_streak_frames="
                f"{policy.max_contact_loss_streak.tolist()} "
                f"final_all_contact_streak_frames="
                f"{policy.final_all_contact_streak} "
                f"axial_travel_m={policy.executed_axial_travel:.4f} "
                f"max_fingertip_admittance_joint_correction_rad="
                f"{policy.max_fingertip_admittance_joint_correction_rad:.6f} "
                f"max_arm_force_correction_rad="
                f"{policy.max_arm_force_correction_rad:.6f} "
                f"max_contact_penetration_mm="
                f"{max_penetration_mm.round(3).tolist()} "
                f"max_raw_tip_force_N="
                f"{policy.max_tactile_force.round(3).tolist()} "
                f"max_filtered_tip_normal_force_N="
                f"{policy.max_filtered_normal_force.round(3).tolist()} "
                f"max_runtime_self_penetration_mm="
                f"{policy.max_runtime_self_penetration_m * 1000:.6f} "
                f"runtime_self_near_contact_frames="
                f"{policy.runtime_self_near_contact_frames} "
                f"contact_surface_travel_m="
                f"{policy.contact_surface_travel_m.round(4).tolist()} "
                f"tip_in_palm_travel_m="
                f"{policy.contact_relative_travel_m.round(4).tolist()} "
                f"per_finger_joint_excursion_rad="
                f"{per_finger_joint_excursion.round(4).tolist()} "
                f"arm_collision_frames={policy.arm_collision_frames} "
                "incidental_hand_contact_frames="
                f"{policy.incidental_hand_contact_frames} "
                "max_incidental_hand_contact_streak_frames="
                f"{policy.max_incidental_hand_contact_streak} "
                "max_incidental_hand_contact_force_N="
                f"{policy.max_incidental_hand_contact_force_n:.3f} "
                "max_incidental_hand_total_force_N="
                f"{policy.max_incidental_hand_total_force_n:.3f} "
                "max_incidental_hand_penetration_mm="
                f"{policy.max_incidental_hand_penetration_m * 1000:.3f} "
                "incidental_hand_contact_evaluated_frames="
                f"{policy.incidental_hand_contact_evaluated_frames} "
                "min_tip_contacts_during_incidental_contact="
                f"{policy.min_tip_contacts_during_incidental_contact}/4 "
                f"max_runtime_pad_angle_deg="
                f"{np.degrees(np.arccos(np.clip(policy.min_runtime_pad_alignment, -1, 1))):.2f} "
                f"planned_curvature_inv_m="
                f"[{policy.planned_curvature_min_inv_m:.3f},"
                f"{policy.planned_curvature_max_inv_m:.3f}] "
                f"planned_curvature_ratio="
                f"{'inf' if np.isinf(policy.planned_curvature_ratio) else f'{policy.planned_curvature_ratio:.3f}'} "
                f"final_tip_site_standoff_mm="
                f"{(policy.surface_error[1:] * 1000).round(2).tolist()}",
                flush=True,
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
    horizon_joint_margin_residual,
    horizon_joint_step_residual,
