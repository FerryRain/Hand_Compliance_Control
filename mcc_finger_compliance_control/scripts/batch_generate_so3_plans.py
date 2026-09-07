"""批量生成 SO(3) 随机环绕轨迹 + 离线可达性筛选(数据采集第一步)。

默认网格:16 个环绕区域 (ellipse-section-fraction x plane-tilt x azimuth)
x 16 个 SO(3) 随机初始姿态 seed = 256 条候选轨迹。每条并行执行:
  1. generate_manifold_palm_plan.py  -- 物体系手掌椭圆轨迹(uniform_so3)
  2. optimize_contact_plan.py --keyframes-only -- 26 关键帧离线判据
只保留 grasp_keyframe_valid 全部为 True(每条轨迹的每个关键帧都通过
离线可达性检查:距离/法线朝向/各向同性可达性/可操纵性/横向间距/姿态偏差)
的轨迹,写入 passing 列表供第二步并行采集。

--target-passing N 模式:生成一小批 --batch-size 条(默认 12)候选就做一轮
离线检测,再生成下一批,直到累计通过离线检查的轨迹达到 N 条。候选按
seed 主序扩展(每个 seed 覆盖全部 16 区域,seed 601、602… 递增),参考
12 条分层重跑通过率 ~42%,256 条通过约需 600+ 候选。该模式下保留全部
产物(便于重跑幂等)。

用法:
  python batch_generate_so3_plans.py --output-dir ../data/plans/mustard_so3_v1 \
      --workers 12 [--target-passing 256] [--batch-size 12] [--count 256]
      [--regions ...] [--seeds ...] [--keep-all]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py

SCRIPTS = Path(__file__).resolve().parent
PYTHON = sys.executable
GENERATE = SCRIPTS / "generate_manifold_palm_plan.py"
OPTIMIZE = SCRIPTS / "optimize_contact_plan.py"

# 16 个环绕区域:(section_fraction, tilt_deg, azimuth_deg)。f 沿瓶子 PCA 长轴
# 0=瓶底端 1=瓶盖端,tilt 截面倾斜,azimuth 椭圆平面绕长轴旋转。继承
# mustard_scale_sweep_v3 已验证网格并补充 4 组填充中间区域。
DEFAULT_REGIONS = (
    (0.18, 20, 30), (0.25, 40, 80), (0.32, 55, 130), (0.40, 25, 180),
    (0.48, 50, 230), (0.56, 65, 280), (0.65, 20, 330), (0.72, 40, 45),
    (0.80, 55, 100), (0.86, 20, 155), (0.91, 35, 215), (0.96, 15, 275),
    (0.28, 30, 60), (0.60, 10, 200), (0.75, 45, 120), (0.90, 25, 300),
)
DEFAULT_SEEDS = tuple(range(601, 617))  # 16 个 SO(3) 初始姿态 seed


def _plan_name(f: float, t: float, a: float, seed: int) -> str:
    return f"f{f:.2f}_t{int(t)}_a{int(a)}_s{seed}"


def _evaluate_one(args: tuple) -> tuple[str, int, float, str]:
    """生成 + 关键帧离线判据评估;返回 (name, valid_count, ratio, status)。"""
    name, f, t, a, seed, output_dir, keep_all = args
    plan_h5 = output_dir / f"{name}.h5"
    opt_h5 = output_dir / f"{name}_opt.h5"
    if opt_h5.exists():
        try:
            with h5py.File(opt_h5, "r") as handle:
                valid = handle["grasp_keyframe_valid"][:]
            if "grasp_keyframe_valid" in handle and valid.size:
                # 缓存命中:判定与首跑一致,避免累计通过数漏计。
                ratio = float(valid.mean())
                status = "passed" if bool(valid.all()) else "failed"
                return name, int(valid.sum()), ratio, status
        except Exception:
            pass  # 缓存损坏则重新生成
    generate_cmd = [
        PYTHON, str(GENERATE),
        "--output", str(plan_h5),
        "--object-id", "ycb_mustard", "--frames", "2500", "--angle-deg", "55",
        "--path-mode", "minimum_enclosing_ellipse",
        "--palm-tangent-sign", "-1",
        "--ellipse-arc-region", "calibrated",
        "--ellipse-section-fraction", str(f),
        "--ellipse-plane-tilt-deg", str(int(t)),
        "--ellipse-azimuth-deg", str(int(a)),
        "--object-rotation-mode", "uniform_so3",
        "--seed", str(seed),
    ]
    result = subprocess.run(
        generate_cmd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return name, 0, 0.0, f"generate_error: {result.stderr.strip()[:120]}"
    optimize_cmd = [
        PYTHON, str(OPTIMIZE),
        "--input", str(plan_h5),
        "--output", str(opt_h5),
        "--object-id", "ycb_mustard",
        "--keyframes-only",
    ]
    result = subprocess.run(
        optimize_cmd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return name, 0, 0.0, f"optimize_error: {result.stderr.strip()[:120]}"
    with h5py.File(opt_h5, "r") as handle:
        valid = handle["grasp_keyframe_valid"][:]
    ratio = float(valid.mean())
    status = "passed" if bool(valid.all()) else "failed"
    return name, int(valid.sum()), ratio, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Plan 输出目录(幂等:已有 _opt.h5 则跳过)。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="并行进程数(默认 12)。",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="候选总数;默认 16 区域 x 16 seed = 256。",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        type=float,
        default=None,
        help="区域列表(f t a 三连);默认 DEFAULT_REGIONS。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="SO(3) seed 列表;默认 601-616。",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="保留全部候选的 _opt.h5(默认只保留通过离线检查的产物)。",
    )
    parser.add_argument(
        "--target-passing",
        type=int,
        default=None,
        help=(
            "目标通过数:每生成一批就做一轮离线检测,直到累计通过的轨迹 "
            ">= N 条。默认 None = 只跑给定网格一次。该模式下自动保留全部"
            "产物(--keep-all),便于重跑幂等。"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="每批候选条数(默认 12):生成一批 -> 离线检测 -> 再生成下一批。",
    )
    args = parser.parse_args()

    regions = (
        tuple(
            (args.regions[i], args.regions[i + 1], args.regions[i + 2])
            for i in range(0, len(args.regions), 3)
        )
        if args.regions
        else DEFAULT_REGIONS
    )
    base_seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    target = args.target_passing
    # target 模式保留全部产物:重跑时缓存命中直接跳过,不重复生成失败候选。
    keep_all = args.keep_all or target is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple] = []
    batch_no = 0
    # 候选队列按 seed 主序:每个 seed 覆盖全部区域,保证每批区域分布均匀。
    seed_low, seed_high = base_seeds[0], base_seeds[-1]
    queue = [
        (f, t, a, seed)
        for seed in range(seed_low, seed_high + 1)
        for f, t, a in regions
    ]
    if args.count is not None and target is None:
        queue = queue[: args.count]
    while queue:
        chunk = queue[: args.batch_size]
        queue = queue[args.batch_size:]
        jobs = [
            (_plan_name(f, t, a, seed), f, t, a, seed, args.output_dir,
             keep_all)
            for f, t, a, seed in chunk
        ]
        print(
            f"[BATCH-GEN] batch {batch_no}: {len(jobs)} candidates "
            f"(seeds {chunk[0][3]}-{chunk[-1][3]}), workers={args.workers}, "
            f"dir={args.output_dir}",
            flush=True,
        )
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_evaluate_one, job): job[0] for job in jobs}
            done = 0
            for future in as_completed(futures):
                name = futures[future]
                try:
                    name, valid_count, ratio, status = future.result()
                except Exception as exc:  # 单条失败不中断整批
                    name, valid_count, ratio, status = (
                        name, 0, 0.0, f"exception: {exc}"
                    )
                done += 1
                rows.append((name, valid_count, ratio, status))
                print(
                    f"[{done}/{len(jobs)}] {name}: keyframe_valid "
                    f"{valid_count}/26 ({ratio:.1%}) [{status}]",
                    flush=True,
                )
                if status == "failed" and not keep_all:
                    (args.output_dir / f"{name}.h5").unlink(missing_ok=True)
                    (args.output_dir / f"{name}_opt.h5").unlink(missing_ok=True)
        all_rows.extend(rows)
        passing = [row for row in all_rows if row[3] == "passed"]
        batch_no += 1
        print(
            f"[BATCH-GEN] batch {batch_no} done: cumulative passing "
            f"{len(passing)} (target={target if target is not None else 'n/a'}, "
            f"{len(all_rows)} candidates attempted)",
            flush=True,
        )
        if target is None or len(passing) >= target:
            break
        if not queue:  # target 模式:扩展下一组 seed,继续逐批生成
            if len(all_rows) >= 4096:  # 安全阀:防止通过率过低时无限扩展
                print(
                    f"[BATCH-GEN] 达到 {len(all_rows)} 候选仍不足 {target} 条通过,"
                    f"停止扩展"
                )
                break
            seed_low, seed_high = seed_high + 1, seed_high + 16
            queue = [
                (f, t, a, seed)
                for seed in range(seed_low, seed_high + 1)
                for f, t, a in regions
            ]

    all_rows.sort(key=lambda row: (row[3] == "passed", -row[2], row[0]))
    summary_path = args.output_dir / "offline_screening.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["plan_name", "keyframe_valid_count", "keyframe_valid_ratio",
             "status"]
        )
        writer.writerows(all_rows)
    passing = [row for row in all_rows if row[3] == "passed"]
    if target is not None:
        passing = sorted(passing, key=lambda row: row[0])[:target]
    passing_path = args.output_dir / "passing_plans.txt"
    passing_path.write_text(
        "\n".join(
            f"{args.output_dir / row[0]}_opt.h5" for row in passing
        )
        + ("\n" if passing else "")
    )
    print(
        f"[BATCH-GEN] done: {len(passing)} plans passed offline screening "
        f"(of {len(all_rows)} candidates) -> {passing_path}"
    )
    print(f"[BATCH-GEN] summary -> {summary_path}")


if __name__ == "__main__":
    main()
