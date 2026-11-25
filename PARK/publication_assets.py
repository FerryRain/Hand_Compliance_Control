"""Export the verified PARK scores, physical replays and reconstruction evidence for the atlas."""
import argparse
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import trimesh

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORKSPACE / "PalmReach-3"), str(WORKSPACE / "PARK-3")]

from Module.palmreach6.__main__ import FIXTURES
from Module.wrist_active.evaluate import Reference, mesh_accuracy, reconstruct
from baseline_comparison import draw_mesh, reference_view
from comparison import collect
from park3.common import digest, read, table, write
import replays

TRIALS = WORKSPACE / "outputs/PalmReach-3-v3"
FOUR_OBJECTS = ("coffee", "cracker", "mustard", "bleach")
OBJECTS = (*FOUR_OBJECTS, "pear")
BUDGETS = (240, 600)
METHODS = ("PARK", "PalmReach-6", "AUDE")
NAMES = dict(zip(replays.METHODS, METHODS, strict=True))
METRICS = ("c25_pct", "c10_pct", "f5_pct", "precision5_pct", "recall5_pct",
           "surface_error_mm", "chamfer_mm2")


def media_path(output, seconds, obj):
    return output / "media/park" / f"{seconds}s" / f"{obj}.mp4"


def render_movies(output, budgets, objects):
    view = replays.load_view()
    for seconds in budgets:
        for obj in objects:
            movie = media_path(output, seconds, obj)
            replays.replay(TRIALS, obj, seconds, view, baselines=True,
                           primary_label="PARK", destination=movie)
            with imageio.get_reader(str(movie), format="FFMPEG") as stream:
                last = stream.get_data(stream.count_frames() - 1)
            Image.fromarray(last).save(movie.with_suffix(".webp"), quality=88, method=5)


def render_endpoint(output, obj, seconds, truth, endpoints, scores, basis, center, radius):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = ("#949DA5", "#5FB7A5", "#7CA6D7", "#DDAD6C")
    with plt.rc_context({"font.family": "Microsoft YaHei"}):
        fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.35), dpi=140)
        fig.subplots_adjust(left=.025, right=.975, top=.78, bottom=.13, wspace=.025)
        fig.text(.025, .956, f"{replays.NAMES[obj]}  ·  {seconds} 秒表面重建",
                 ha="left", va="top", color="#233646", fontsize=15, weight="bold")
        for ax, title, mesh, color in zip(axes, ("参考真值", *METHODS),
                                          (truth, *endpoints), colors, strict=True):
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(title, fontsize=12, weight="semibold", pad=2)
            draw_mesh(ax, mesh, np.asarray(mesh.vertices) @ basis, basis, color)
            if title in METHODS:
                row = scores[seconds, obj, title]
                ax.text(.5, -.03, f"C25 {row['c25_pct']:.1f}%   ·   F@5 {row['f5_pct']:.1f}%",
                        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#344253")
        out = output / "images/park" / f"{seconds}s" / f"{obj}.webp"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor="white", pil_kwargs={"quality": 92, "method": 5})
        plt.close(fig)


def zip_entry(archive, name, content):
    entry = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    entry.compress_type = ZIP_DEFLATED
    entry.external_attr = 0o100644 << 16
    archive.writestr(entry, content, compresslevel=6)


def wrist_csv(pose):
    sample = pose[::10, :7]
    seconds = np.arange(len(sample), dtype=float) / 10
    out = io.StringIO()
    np.savetxt(out, np.column_stack((seconds, sample)), delimiter=",",
               fmt=["%.1f", *(["%.17g"] * 7)], comments="",
               header="time_s,x_m,y_m,z_m,qw,qx,qy,qz")
    return out.getvalue().encode("ascii")


def precision_recall(archived, f5, native, reference, mesh_path, stage):
    if np.isclose(archived["fscore"], f5, rtol=0, atol=1e-13):
        return archived, "saved_mesh"
    # The evaluator originally scored its float64 vertices immediately before PLY encoding.
    # Recover its exact point counts when float32 serialization moves a sample across 5 mm.
    with np.load(native / "trace.npz", allow_pickle=False) as tr:
        valid = tr["pad_valid"] & (tr["region_load_n"].max(1) < 8.)[:, None]
        points, normals = tr["points_m"][valid], tr["normals_outward"][valid]
    with TemporaryDirectory(dir=stage) as temporary:
        rebuilt = Path(temporary) / "signed.ply"
        original = reconstruct(points, normals, reference, rebuilt, "signed", signed_device="cuda")
        if (original["status"] != "ok" or digest(rebuilt) != digest(mesh_path)
                or not np.isclose(original["fscore"], f5, rtol=0, atol=1e-13)):
            raise ValueError(f"Could not reproduce the original precision and recall for {native}")
    return original, "original_evaluator_before_ply_encoding"


def summarize(trials):
    means = []
    for cohort, objects in (("four", FOUR_OBJECTS), ("five", OBJECTS)):
        for seconds in BUDGETS:
            for method in METHODS:
                rows = [r for r in trials if r["object"] in objects and
                        (r["seconds"], r["method"]) == (seconds, method)]
                if {row["object"] for row in rows} != set(objects) or len(rows) != len(objects):
                    raise ValueError(f"Incomplete {cohort}-object mean for {method}, {seconds}s")
                means.append(dict(cohort=cohort, seconds=seconds, method=method, n=len(objects),
                                  **{key: float(np.mean([r[key] for r in rows])) for key in METRICS}))
    return means


def build_dataset(output):
    rows, motion, _, sources = collect(TRIALS, BUDGETS, "v3", objects=OBJECTS, repeats=[0])
    scores, inputs, provenance = {}, {}, {}
    curves, curve_rows = [], []
    for row, source in zip(rows, sources, strict=True):
        seconds, obj, original = (row[k] for k in ("budget_s", "object", "method"))
        method = NAMES[original]
        key = (seconds, obj, method)
        score_path = (TRIALS / source["score"]).resolve()
        native = (TRIALS / source["native"]).resolve()
        mesh = score_path.with_name(f"signed-{seconds}s.ply")
        signature = read(score_path)["signature"]
        if (source["executed_steps"] != 100 * seconds or not mesh.is_file() or
                source["trace"] != signature["trace"]):
            raise ValueError(f"Invalid time budget, mesh or trace for {key}")
        scores[key] = dict(object=obj, method=method, seconds=seconds, seed=row["seed"],
                           c25_pct=100 * row["c25"], c10_pct=100 * row["c10"], f5_pct=100 * row["f5"],
                           surface_error_mm=row["surface_error_mm"], chamfer_mm2=row["chamfer_mm2"])
        inputs[key] = (mesh, native)
        provenance[key] = dict(object=obj, method=method, seconds=seconds, seed=row["seed"],
                               steps=source["executed_steps"], trace_sha256=source["trace"],
                               score_sha256=digest(score_path), evaluator_sha256=signature["metrics"],
                               plan_sha256=signature["plans"], initial_state_sha256=source["initial"],
                               model_sha256=source["model"], reference_sha256=source["scoring_mesh"],
                               endpoint_mesh_sha256=digest(mesh), scoring_gpis_sha256=signature["gpis"])
        if seconds == 600:
            series = [p for p in motion if (p["budget_s"], p["object"], p["method"]) ==
                      (seconds, obj, original)]
            coverage = [[int(p["seconds"]), 100 * p["c25"]] for p in series]
            reconstruction = [[int(p["seconds"]), 100 * p["f5"]] for p in series if p["f5"] is not None]
            if len(coverage) != seconds + 1 or reconstruction[-1][0] != seconds:
                raise ValueError(f"Incomplete score curve for {key}")
            curves.append(dict(object=obj, method=method, seconds=seconds, seed=row["seed"],
                               coverage=coverage, reconstruction=reconstruction))
            for p in series:
                curve_rows.append(dict(object=obj, method=method, seed=row["seed"], budget_s=seconds,
                                       time_s=int(p["seconds"]), c25_pct=100 * p["c25"],
                                       c10_pct=100 * p["c10"], path_m=p["path_m"],
                                       f5_pct=100 * p["f5"] if p["f5"] is not None else None,
                                       surface_error_mm=p["surface_error_mm"]))

    data = output / "data/park"
    data.mkdir(parents=True, exist_ok=True)
    specs = read(FIXTURES / "fixtures.json")["objects"]
    with TemporaryDirectory(dir=data) as tmp:
        stage = Path(tmp)
        with ZipFile(stage / "scored-meshes.zip", "w") as meshes, ZipFile(stage / "wrist-trajectories.zip", "w") as wrists:
            for obj in OBJECTS:
                reference_path = FIXTURES / obj / "surface.obj"
                truth = trimesh.load_mesh(reference_path, process=False)
                ref = Reference(reference_path)
                basis = reference_view(np.asarray(truth.vertices))
                xy = [(truth.vertices @ basis)[:, :2]]
                endpoint_meshes = {}
                for method in METHODS:
                    for seconds in BUDGETS:
                        key = (seconds, obj, method)
                        mesh_path, native = inputs[key]
                        reconstructed = trimesh.load_mesh(mesh_path, process=False)
                        endpoint_meshes[seconds, method] = reconstructed
                        xy.append((reconstructed.vertices @ basis)[:, :2])
                        accuracy = mesh_accuracy(reconstructed, ref.truth)
                        row = scores[key]
                        at5, source = precision_recall(accuracy["at_mm"]["5"], row["f5_pct"] / 100,
                                                       native, ref, mesh_path, stage)
                        provenance[key]["precision_recall_source"] = source
                        np.testing.assert_allclose(accuracy["symmetric_mean_mm"], row["surface_error_mm"],
                                                   atol=.001, rtol=0)
                        np.testing.assert_allclose(accuracy["chamfer_mm2"], row["chamfer_mm2"], rtol=1e-10)
                        row.update(precision5_pct=100 * at5["precision"], recall5_pct=100 * at5["recall"])
                        entry = f"{seconds}s/{obj}/{method}.ply"
                        zip_entry(meshes, entry, mesh_path.read_bytes())
                        provenance[key]["mesh_entry"] = entry
                        if digest(reference_path) != provenance[key]["reference_sha256"]:
                            raise ValueError(f"Wrong reference surface for {key}")
                    short_key, long_key = (240, obj, method), (600, obj, method)
                    with np.load(inputs[short_key][1] / "trace.npz", allow_pickle=False) as earlier, \
                         np.load(inputs[long_key][1] / "trace.npz", allow_pickle=False) as later:
                        fields = ("qpos", "points_m", "normals_outward", "pad_valid", "pad_force_n", "region_load_n")
                        matched = all(np.array_equal(earlier[field], later[field][:24001], equal_nan=True)
                                      for field in fields)
                        long_entry = f"600s/{obj}/{method}.csv"
                        zip_entry(wrists, long_entry, wrist_csv(later["qpos"]))
                        short_entry = long_entry if matched else f"240s/{obj}/{method}.csv"
                        if not matched:
                            zip_entry(wrists, short_entry, wrist_csv(earlier["qpos"]))
                        provenance[short_key].update(wrist_entry=short_entry, same_physical_prefix_240s=matched)
                        provenance[long_key].update(wrist_entry=long_entry, same_physical_prefix_240s=matched)
                points = np.concatenate(xy)
                low, high = points.min(0), points.max(0)
                center, radius = .5 * (low + high), .56 * max(high - low)
                for seconds in BUDGETS:
                    ordered = [endpoint_meshes[seconds, method] for method in METHODS]
                    render_endpoint(output, obj, seconds, truth, ordered, scores, basis, center, radius)
                print(f"Verified endpoint meshes, 5 mm scores and trajectories for {obj}", flush=True)
        trials = list(scores.values())
        dataset = dict(methods=METHODS, budgets=BUDGETS, cohorts={"four": FOUR_OBJECTS, "five": OBJECTS},
                       objects=[dict(id=obj, name=replays.NAMES[obj], seed=specs[obj]["seed"]) for obj in OBJECTS],
                       trials=trials, means=summarize(trials))
        write(stage / "results.json", dataset)
        table(stage / "scores.csv", trials)
        write(stage / "curves.json", dict(methods=METHODS, series=curves))
        table(stage / "curves.csv", curve_rows)
        write(stage / "provenance.json", dict(
            physics_hz=100, wrist_export_hz=10, wrist_pose="Measured x,y,z (m) and MuJoCo quaternion w,x,y,z",
            trajectory_archive="wrist-trajectories.zip", endpoint_archive="scored-meshes.zip",
            f5_threshold_mm=5, curve_budget_s=600, coverage_curve_period_s=1,
            reconstruction_checkpoints_s=[1, 10, 30, 60, 120, 240, 360, 480, 600],
            files_sha256={name: digest(stage / name) for name in (
                "results.json", "scores.csv", "curves.json", "curves.csv", "scored-meshes.zip", "wrist-trajectories.zip")},
            trials=list(provenance.values())))
        for artifact in stage.iterdir():
            artifact.replace(data / artifact.name)


def verify_movies(output):
    data = output / "data/park"
    evidence = read(data / "provenance.json")["trials"]
    sources = {(e["seconds"], e["object"], e["method"]): e for e in evidence}
    movies = []
    for seconds in BUDGETS:
        stride = 50 if seconds == 240 else 100
        speed = replays.FPS * stride // 100
        frames = seconds * 100 // stride + 1 + replays.FPS
        for obj in OBJECTS:
            movie = media_path(output, seconds, obj)
            poster = movie.with_suffix(".webp")
            receipt = read(movie.with_suffix(".json"))
            info = receipt["signature"]
            traces = [sources[seconds, obj, method]["trace_sha256"] for method in METHODS]
            seed = sources[seconds, obj, "PARK"]["seed"]
            if (info["methods"] != list(METHODS) or info["traces"] != traces or
                    info["frame_stride_ticks"] != stride or info["seed"] != seed or
                    info["frames"] != frames or info["budget_s"] != seconds or
                    receipt["sha256"] != digest(movie)):
                raise ValueError(f"Unmatched replay receipt for {movie}")
            header = replays.header(obj, seconds, seed, speed, METHODS, True, replays.SIZE, replays.COLORS)
            with imageio.get_reader(str(movie), format="FFMPEG") as stream:
                metadata = stream.get_meta_data()
                if (stream.count_frames() != frames or metadata["codec"] != "h264" or
                        metadata["fps"] != replays.FPS or tuple(metadata["size"]) != replays.SIZE):
                    raise ValueError(f"Invalid browser video at {movie}")
                for time in (0, seconds // 2, seconds):
                    index = time * 100 // stride
                    expected = header.copy()
                    replays.label(ImageDraw.Draw(expected), (replays.SIZE[0] // 2, 28),
                                  f"{time} / {seconds} s", 31, anchor="mt")
                    decoded = stream.get_data(index).astype(np.int16)
                    patches = [(16, 82, 280, 123), (660, 23, 875, 72)]
                    for left, top, right, bottom in patches:
                        patch = np.asarray(expected.crop((left, top, right, bottom)), dtype=np.int16)
                        if np.abs(decoded[top:bottom, left:right] - patch).mean() > 3.5:
                            raise ValueError(f"Incorrect PARK heading or clock at {movie}, {time}s")
                last = stream.get_data(frames - 1).astype(np.int16)
            with Image.open(poster) as thumbnail:
                if np.abs(np.asarray(thumbnail, dtype=np.int16) - last).mean() > 3:
                    raise ValueError(f"Poster differs from the physical endpoint at {movie}")
            mesh_image = output / "images/park" / f"{seconds}s" / f"{obj}.webp"
            with Image.open(mesh_image) as mesh:
                mesh.verify()
            movies.append(dict(object=obj, seconds=seconds, seed=seed, fps=replays.FPS, frames=frames,
                               speed=speed, frame_stride_ticks=stride, methods=METHODS, trace_sha256=traces,
                               video=movie.relative_to(output).as_posix(), sha256=receipt["sha256"],
                               poster=poster.relative_to(output).as_posix(), poster_sha256=digest(poster),
                               mesh=mesh_image.relative_to(output).as_posix(), mesh_sha256=digest(mesh_image)))
    write(data / "media.json", dict(width=replays.SIZE[0], height=replays.SIZE[1], codec="h264", movies=movies))
    print(f"Verified the PARK heading, shared clock, source hashes and end frames of {len(movies)} videos.", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("media", "data", "verify", "all"))
    cli.add_argument("--output", type=Path, default=WORKSPACE / ".tmp/park-final-assets")
    cli.add_argument("--objects", nargs="+", choices=OBJECTS, help="limit the media action")
    cli.add_argument("--seconds", type=int, choices=BUDGETS, help="limit the media action")
    args = cli.parse_args()
    if args.action != "media" and (args.objects or args.seconds):
        cli.error("A subset can only be selected for media")
    output = args.output.resolve()
    if args.action in ("data", "all"):
        build_dataset(output)
    if args.action in ("media", "all"):
        render_movies(output, [args.seconds] if args.seconds else BUDGETS, args.objects or OBJECTS)
    if args.action in ("verify", "all"):
        verify_movies(output)


if __name__ == "__main__":
    main()
