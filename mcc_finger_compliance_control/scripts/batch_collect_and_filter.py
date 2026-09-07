"""并行仿真采集 + 离线筛选 ≥98% 四指接触率(数据采集第二步)。

输入:第一步输出的 passing_plans.txt(或扫描 *_opt.h5 全离线判据通过的 plan)。
流程:
  1. 把通过离线检查的 plans 打包成若干 bundle(默认 16 env/批,复用
     bundle_manifold_plans.bundle,要求等长帧数)。
  2. 逐批 subprocess 调用 collect_trajectories.py(planner_inverse,
     同批 env 并行仿真,批间串行以免单卡过载)。
  3. 分析每 env 的正面 pad all4 接触率、单指接触率、最长失触和 q 跳变。
  4. 筛选合格轨迹并逐条导出到 --selected-dir，保留标准 (T,1,...)
     环境维，写 contact_rates.csv 汇总。

用法:
  python batch_collect_and_filter.py \
      --passing-list ../data/plans/mustard_so3_v1/passing_plans.txt \
      --output-dir ../data/trajectories/mustard_so3_v1 \
      --selected-dir ../data/trajectories/mustard_so3_v1/selected \
      --envs-per-batch 16 --min-contact-rate 0.98
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from bundle_manifold_plans import bundle

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent.parent
PYTHON = sys.executable
COLLECT = SCRIPTS / "collect_trajectories.py"
# collect_trajectories.py 把输出写到 <cwd>/mcc_finger_compliance_control/
# data/trajectories/<filename>.h5(filename 无扩展名时自动追加 .h5)。
COLLECT_OUT_DIR = PROJECT_ROOT / "mcc_finger_compliance_control" / "data" / "trajectories"
BACK_CONTACT_X_LIMIT_M = np.asarray(
    (0.012, 0.012, 0.012, 0.016), dtype=np.float64
)
CONTACT_POINT_RADIUS_LIMIT_M = 0.05

# 与回归 aude_s601_s606_s607_s608_s609_full 一致的采集参数(见其 attrs)。
# fixed_motion_start=True 对应 attrs motion_start_semantics=fixed_schedule
# 与 wait_for_four_tip_prep=False:motion 按固定时刻启动,不等待四指就绪。
# no_contact_gate=True 对应 attrs contact_gated_motion=False。
COLLECT_DEFAULTS = {
    "trajectory_length": 3700,
    "motion_start": 800,
    "motion_length": 2500,
    "record_start_step": 800,
    "max_prep_wait_steps": 0,
    "planner_settle_steps": 400,
    "physics_substeps": 10,
    "seed": 42,
    "contact_threshold": 0.05,
    "fixed_motion_start": True,
    "no_contact_gate": True,
}

EXPORT_DATASETS = (
    "episode_id",
    "record_step",
    "q_hand",
    "qvel",
    "palm_pose_object",
    "palm_twist_object",
    "fingertip_pose_object",
    "fingertip_force_world",
    "fingertip_collision_found",
    "object_pose_world",
    "object_angular_velocity_world",
    "fingertip_pad_contact_valid",
    "fingertip_contact_normal_oracle",
)


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _contact_points_tip_local(
    fingertip_pose_world: np.ndarray,
    contact_points_world: np.ndarray,
) -> np.ndarray:
    """Transform same-frame world contact points into wxyz tip frames."""
    pose = np.asarray(fingertip_pose_world, dtype=np.float64)
    points = np.asarray(contact_points_world, dtype=np.float64)
    quaternion = pose[..., 3:7].copy()
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    rotation[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[..., 0, 1] = 2.0 * (x * y - z * w)
    rotation[..., 0, 2] = 2.0 * (x * z + y * w)
    rotation[..., 1, 0] = 2.0 * (x * y + z * w)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[..., 1, 2] = 2.0 * (y * z - x * w)
    rotation[..., 2, 0] = 2.0 * (x * z - y * w)
    rotation[..., 2, 1] = 2.0 * (y * z + x * w)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.einsum("...ji,...j->...i", rotation, points - pose[..., :3])


def _contact_rates(batch_h5: Path) -> dict[int, dict]:
    """逐 env 计算正面指腹接触和关节平滑性指标。"""
    with h5py.File(batch_h5, "r") as source:
        episode_id = np.asarray(source["episode_id"])  # (T, E)
        force = np.asarray(source["fingertip_force_world"])  # (T, E, 4, 3)
        force_norm = np.linalg.norm(force, axis=-1)  # (T, E, 4)
        if "fingertip_collision_found" in source:
            found = np.asarray(source["fingertip_collision_found"]) > 0.5
        else:
            found = force_norm > 0.0
        threshold = float(
            source.attrs.get("contact_threshold", COLLECT_DEFAULTS["contact_threshold"])
        )
        loaded = found & (force_norm >= threshold)
        q_hand = np.asarray(source["q_hand"])
        if (
            "fingertip_pose_world" in source
            and "fingertip_contact_pos_world" in source
        ):
            contact_pos_tip = _contact_points_tip_local(
                np.asarray(source["fingertip_pose_world"]),
                np.asarray(source["fingertip_contact_pos_world"]),
            )
        elif "fingertip_contact_pos_tip" in source:
            contact_pos_tip = np.asarray(source["fingertip_contact_pos_tip"])
        else:
            contact_pos_tip = None
        if contact_pos_tip is not None:
            plausible = (
                np.isfinite(contact_pos_tip).all(axis=-1)
                & (
                    np.linalg.norm(contact_pos_tip, axis=-1)
                    <= CONTACT_POINT_RADIUS_LIMIT_M
                )
            )
            front_or_side = (
                contact_pos_tip[..., 0]
                <= BACK_CONTACT_X_LIMIT_M[None, None, :]
            )
            loaded &= plausible & front_or_side
        elif "fingertip_pad_contact_valid" in source:
            loaded &= np.asarray(source["fingertip_pad_contact_valid"]) > 0.5
    rates: dict[int, dict] = {}
    for eid in np.unique(episode_id):
        frame_mask, env_mask = np.where(episode_id == eid)
        env_index = int(env_mask[0])
        episode_loaded = loaded[frame_mask, env_index]  # (T, 4)
        all4 = episode_loaded.all(axis=1)
        episode_q = q_hand[frame_mask, env_index]
        q_step = (
            np.max(np.abs(np.diff(episode_q, axis=0)), axis=1)
            if len(episode_q) > 1
            else np.zeros(1, dtype=np.float32)
        )
        contact_x_p95_mm = np.full(4, np.nan, dtype=np.float64)
        if contact_pos_tip is not None:
            episode_pos = contact_pos_tip[frame_mask, env_index]
            for finger in range(4):
                values = episode_pos[episode_loaded[:, finger], finger, 0]
                if values.size:
                    contact_x_p95_mm[finger] = np.percentile(values, 95) * 1000.0
        rates[eid] = {
            "all4": float(all4.mean()),
            "tip_ratios": tuple(float(v) for v in episode_loaded.mean(axis=0)),
            "frames": int(len(frame_mask)),
            "max_loss_run": _longest_true_run(~all4),
            "q_step_p99": float(np.percentile(q_step, 99)),
            "q_step_max": float(q_step.max()),
            "contact_x_p95_mm": tuple(float(v) for v in contact_x_p95_mm),
        }
    return rates


def _export_episode(batch_h5: Path, env: int, plan_name: str, out: Path) -> None:
    """导出单轨迹并保留标准环境轴，供 filter/export/DP 工具直接读取。"""
    with h5py.File(batch_h5, "r") as source:
        frame_count = int(source["episode_id"].shape[0])
        env_count = int(np.asarray(source["episode_id"]).shape[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as target:
            for name, dataset in source.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                value = np.asarray(dataset)
                if value.ndim >= 2 and value.shape[:2] == (frame_count, env_count):
                    # (T,E,...) -> (T,1,...)
                    target.create_dataset(name, data=value[:, env : env + 1])
                elif value.ndim >= 1 and value.shape[0] == env_count:
                    # 每 env 一个的向量/矩阵,如 q_nominal (E, 16)。
                    target.create_dataset(name, data=value[env : env + 1])
                else:
                    target.create_dataset(name, data=value)
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs["plan_name"] = plan_name
            target.attrs["selected_all4"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passing-list",
        type=Path,
        default=None,
        help="第一步的 passing_plans.txt(默认扫描 --plans-dir 下全部 _opt.h5)。",
    )
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=None,
        help="扫描 *_opt.h5 并仅取 grasp_keyframe_valid 全 True 的 plan。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="采集输出目录(bundle 与 batch h5 存放处)。",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=None,
        help="筛选后单轨迹导出目录(默认 <output-dir>/selected)。",
    )
    parser.add_argument(
        "--envs-per-batch",
        type=int,
        default=16,
        help="每批并行环境数(默认 16)。",
    )
    parser.add_argument(
        "--min-contact-rate",
        type=float,
        default=0.98,
        help="筛选阈值:all4 四指接触率下限(默认 0.98)。",
    )
    parser.add_argument(
        "--min-per-tip-rate",
        type=float,
        default=0.98,
        help="每根手指的正面接触率下限(默认 0.98)。",
    )
    parser.add_argument(
        "--max-loss-run",
        type=int,
        default=10,
        help="允许的最长连续四指失触帧数(默认 10)。",
    )
    parser.add_argument(
        "--max-q-step-rad",
        type=float,
        default=0.03,
        help="实际 q_hand 单步最大跳变上限(默认 0.03 rad)。",
    )
    parser.add_argument(
        "--min-plan-rotation-deg",
        type=float,
        default=30.0,
        help="采集前剔除实际姿态变化不足的 plan(默认 30deg)。",
    )
    parser.add_argument(
        "--max-plan-posture-deviation-rad",
        type=float,
        default=0.90,
        help="采集前的关键帧手型偏差上限(默认 0.90rad)。",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="采集设备(默认 cuda:0)。"
    )
    parser.add_argument("--seed", type=int, default=COLLECT_DEFAULTS["seed"])
    parser.add_argument(
        "--execution-randomization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "采集成功 teacher 周围的物理执行扰动管，并保留 "
            "q_prior/q_cmd/q_live 三轨。"
        ),
    )
    parser.add_argument(
        "--execution-perturbation-min-rad", type=float, default=0.003
    )
    parser.add_argument(
        "--execution-perturbation-max-rad", type=float, default=0.020
    )
    parser.add_argument(
        "--execution-clean-env-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--execution-perturbation-time-constant-s",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--execution-perturbation-ramp-steps", type=int, default=50
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 plan 与 bundle,不启动仿真。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用已完成的 batch_*_collected.h5；适合中断后继续。",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = Path(args.selected_dir) if args.selected_dir else output_dir / "selected"

    # 1. 收集通过离线检查的 plans
    if args.passing_list is not None:
        plans = [
            Path(line.strip())
            for line in args.passing_list.read_text().splitlines()
            if line.strip()
        ]
    elif args.plans_dir is not None:
        plans = []
        for opt_h5 in sorted(args.plans_dir.glob("*_opt.h5")):
            try:
                with h5py.File(opt_h5, "r") as handle:
                    if bool(handle["grasp_keyframe_valid"][:].all()):
                        plans.append(opt_h5)
            except Exception:
                continue
    else:
        parser.error("需要 --passing-list 或 --plans-dir 之一")
    if not plans:
        print("[COLLECT] 没有通过离线检查的 plan,退出")
        return
    # 路径可从项目根或 passing-list 所在目录解释，避免依赖启动 cwd。
    resolved_plans: list[Path] = []
    for plan in plans:
        candidates = [plan]
        if args.passing_list is not None:
            candidates.append(args.passing_list.parent / plan)
        candidates.append(PROJECT_ROOT / plan)
        existing = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if existing is None:
            raise FileNotFoundError(f"plan 不存在: {plan}")
        resolved_plans.append(existing)
    plans = resolved_plans

    # 关键帧 26/26 只是必要条件；再剔除几乎静止和手型偏差过大的轨迹。
    preflight_rows: list[tuple[str, float, float, str]] = []
    qualified: list[Path] = []
    for plan in plans:
        with h5py.File(plan, "r") as handle:
            rotation = float(handle.attrs.get("planner_realized_rotation_path_deg", 0.0))
            posture = float(np.max(handle["grasp_keyframe_posture_deviation"][:]))
            valid = bool(handle["grasp_keyframe_valid"][:].all())
        reasons = []
        if not valid:
            reasons.append("invalid_keyframe")
        if rotation < args.min_plan_rotation_deg:
            reasons.append("low_motion")
        if posture > args.max_plan_posture_deviation_rad:
            reasons.append("posture")
        status = "qualified" if not reasons else "+".join(reasons)
        preflight_rows.append((plan.stem, rotation, posture, status))
        if not reasons:
            qualified.append(plan)
    plans = qualified
    preflight_path = output_dir / "collection_preflight.csv"
    with preflight_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["plan_name", "rotation_path_deg", "posture_max_rad", "status"])
        writer.writerows(preflight_rows)
    if not plans:
        raise RuntimeError("没有 plan 通过采集前运动量/手型门控")
    print(
        f"[COLLECT] preflight={len(plans)}/{len(preflight_rows)} -> {preflight_path}"
    )

    if len(plans) == 1:
        print(
            "[COLLECT] 警告:只有 1 个 plan,直接按单 env bundle 处理"
        )
    print(f"[COLLECT] {len(plans)} plans 通过离线检查")

    # 2. 分批 bundle
    batches = [
        plans[i : i + args.envs_per_batch]
        for i in range(0, len(plans), args.envs_per_batch)
    ]
    batch_files: list[tuple[Path, Path]] = []
    for batch_id, batch in enumerate(batches):
        bundle_h5 = output_dir / f"batch_{batch_id:03d}.h5"
        bundle(batch, bundle_h5)
        batch_files.append((bundle_h5, batch))
    print(
        f"[COLLECT] {len(batches)} batches "
        f"({', '.join(str(p.name) for p, _ in batch_files)})"
    )

    if args.dry_run:
        print("[COLLECT] dry-run:不启动仿真")
        return

    # 3. 逐批并行采集
    summary: list[tuple] = []
    for batch_id, (bundle_h5, batch) in enumerate(batch_files):
        batch_out_name = bundle_h5.stem + "_collected"
        batch_out = output_dir / f"{batch_out_name}.h5"
        # A fixed seed for every subprocess would repeat the same clean-env
        # assignment, amplitudes and correlated disturbance directions in the
        # same env slots of all batches.  Derive a deterministic independent
        # seed per bundle so a large collection actually thickens the teacher
        # manifold with diverse recovery states while remaining reproducible.
        batch_seed = int(args.seed + 104729 * batch_id)
        command = [
            PYTHON, str(COLLECT),
            "--viewer", "headless",
            "--device", args.device,
            "--object-id", "ycb_mustard",
            "--teacher-controller", "fullhand_mcc",
            "--motion-mode", "planner_inverse",
            "--planner-file", str(bundle_h5),
            "--initial-orientation-mode", "fixed",
            "--differential-contact-qp",
            "--manifold-qp-blend", "1.0",
            "--trajectory-length", str(COLLECT_DEFAULTS["trajectory_length"]),
            "--motion-start", str(COLLECT_DEFAULTS["motion_start"]),
            "--motion-length", str(COLLECT_DEFAULTS["motion_length"]),
            "--record-start-step", str(COLLECT_DEFAULTS["record_start_step"]),
            "--max-prep-wait-steps", str(COLLECT_DEFAULTS["max_prep_wait_steps"]),
            "--planner-settle-steps", str(COLLECT_DEFAULTS["planner_settle_steps"]),
            "--physics-substeps", str(COLLECT_DEFAULTS["physics_substeps"]),
            "--fixed-motion-start",
            "--no-contact-gate",
            "--num-envs", str(len(batch)),
            "--max-trajectories", str(len(batch)),
            "--seed", str(batch_seed),
            "--filename", batch_out_name,
        ]
        if args.execution_randomization:
            command.extend(
                [
                    "--execution-randomization",
                    "--execution-perturbation-min-rad",
                    str(args.execution_perturbation_min_rad),
                    "--execution-perturbation-max-rad",
                    str(args.execution_perturbation_max_rad),
                    "--execution-clean-env-fraction",
                    str(args.execution_clean_env_fraction),
                    "--execution-perturbation-time-constant-s",
                    str(args.execution_perturbation_time_constant_s),
                    "--execution-perturbation-ramp-steps",
                    str(args.execution_perturbation_ramp_steps),
                ]
            )
        if args.resume and batch_out.exists():
            print(f"[COLLECT] 批 {bundle_h5.name}:复用 {batch_out.name}")
            return_code = 0
        else:
            batch_out.unlink(missing_ok=True)
            print(
                f"[COLLECT] 批 {bundle_h5.name} ({len(batch)} env, "
                f"seed={batch_seed}):采集开始"
            )
            sys.stdout.flush()
            result = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
            return_code = result.returncode
            saved_raw = COLLECT_OUT_DIR / f"{batch_out_name}.h5"
            if saved_raw.exists():
                saved_raw.replace(batch_out)
        if return_code != 0 or not batch_out.exists():
            print(
                f"[COLLECT] 批 {bundle_h5.name} 失败(exit={return_code}, "
                f"out={batch_out})"
            )
            for plan_h5 in batch:
                summary.append(
                    (
                        plan_h5.stem,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -1,
                        np.nan,
                        np.nan,
                        "collect_error",
                    )
                )
            continue
        # 4. 分析接触率
        rates = _contact_rates(batch_out)
        with h5py.File(bundle_h5, "r") as handle:
            plan_names = [str(v, "utf-8") for v in handle["plan_name"][:]]
        for env, plan_h5 in enumerate(batch):
            plan_name = plan_h5.stem
            if env in rates:
                row = rates[env]
                all4 = row["all4"]
                selected = bool(
                    all4 >= args.min_contact_rate
                    and min(row["tip_ratios"]) >= args.min_per_tip_rate
                    and row["max_loss_run"] <= args.max_loss_run
                    and row["q_step_max"] <= args.max_q_step_rad
                )
                if selected:
                    _export_episode(
                        batch_out,
                        env,
                        plan_name,
                        selected_dir / f"{plan_name}.h5",
                    )
                summary.append(
                    (plan_name, all4, row["tip_ratios"][0], row["tip_ratios"][1],
                     row["tip_ratios"][2], row["tip_ratios"][3],
                     row["max_loss_run"], row["q_step_p99"], row["q_step_max"],
                     "selected" if selected else "rejected")
                )
            else:
                summary.append((plan_name, -1.0, 0, 0, 0, 0, -1, np.nan, np.nan, "no_data"))
        print(f"[COLLECT] 批 {bundle_h5.name} 完成,{len(batch)} env")

    # 5. 汇总 CSV
    csv_path = output_dir / "contact_rates.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["plan_name", "all4_contact_rate", "tip0", "tip1", "tip2",
             "tip3", "max_loss_run", "q_step_p99_rad", "q_step_max_rad",
             "status"]
        )
        writer.writerows(summary)
    selected_count = sum(1 for row in summary if row[-1] == "selected")
    print(
        f"[COLLECT] done: {selected_count}/{len(summary)} trajectories >= "
        f"{args.min_contact_rate:.0%} -> {selected_dir}"
    )
    print(f"[COLLECT] summary -> {csv_path}")


if __name__ == "__main__":
    main()
