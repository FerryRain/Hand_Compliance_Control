"""Paired CPU/CUDA offline reconstruction on every released five-object checkpoint."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import gpytorch
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from ..acitive_02 import full_signed, signed_cuda
from ..i04_geometric_traversal import fingertip_r_gpis as ridge
from ..i04_geometric_traversal.fingertip_surface_report import (
  MESH_SAMPLES,
  METRIC_SEED,
  mesh_accuracy,
)
from .experiment import digest
from .face_regression_suite import REPO, read, write
from .frontier_study import ROOT
from .ycb_fixtures import fixture

STUDY = ROOT / "validated"
OUT = STUDY / "gpu-reconstruction"
WINDOWS = (60, 120, 240, 600)


def reconstruct(trace, seconds, device, output):
  if device == "cuda" and torch.cuda.is_initialized():
    torch.cuda.synchronize()
  began = perf_counter()
  stop = seconds * 100 + 1
  with np.load(trace, allow_pickle=False) as f:
    valid = f["pad_valid"][:stop] & (f["region_load_n"][:stop].max(axis=1) < 8)[:, None]
    points = f["points_m"][:stop][valid]
    normals = f["normals_outward"][:stop][valid]
  ids = ridge.voxel_representatives(points)
  if len(ids) > ridge.CONTACT_BUDGET:
    ids = ids[ridge.farthest_indices(points[ids], ridge.CONTACT_BUDGET)]
  if len(ids) < 4:
    raise ValueError("Insufficient contact representatives")
  prepared = perf_counter()
  mesh, length = full_signed.signed_gpis(points[ids], normals[ids], device=device)
  if device == "cuda":
    torch.cuda.synchronize()
  surface = perf_counter()
  mesh.export(output)
  end = perf_counter()
  return mesh, dict(
    total_s=end - began,
    read_and_select_s=prepared - began,
    posterior_and_surface_s=surface - prepared,
    ply_write_s=end - surface,
    contacts=len(ids),
    length_m=length,
    vertices=len(mesh.vertices),
    faces=len(mesh.faces),
  )


def discrepancy(a, b):
  samples = [np.concatenate((m.vertices, m.triangles_center)) for m in (a, b)]
  distances = [
    cKDTree(target).query(source)[0] * 1000
    for source, target in (samples, samples[::-1])
  ]
  return dict(
    p95_mm=max(float(np.quantile(x, 0.95)) for x in distances),
    max_mm=max(float(x.max()) for x in distances),
    same_faces=np.array_equal(a.faces, b.faces),
  )


def run_pair(obj, arm, folder, seconds, order, spec, truth, old):
  dest = OUT / obj / arm
  dest.mkdir(parents=True, exist_ok=True)
  fname = f"signed-{seconds}s.ply"
  published = dest / fname
  reference = dest / "cpu-reference" / fname
  native = folder / fname
  receipt = dest / f"{seconds}s.json"
  if receipt.exists():
    data = read(receipt)
    if (
      data["protocol_sha256"] != spec
      or digest(published) != data["cuda_mesh_sha256"]
      or digest(reference) != data["cpu_mesh_sha256"]
    ):
      raise ValueError(f"Different saved validation for {obj}/{arm}/{seconds}")
    return data
  mechanisms = {}
  geometry = {}
  with TemporaryDirectory(dir=dest) as tmp:
    paths = {device: Path(tmp) / f"{device}.ply" for device in order}
    for device in order:
      geometry[device], mechanisms[device] = reconstruct(
        folder / "trace.npz", seconds, device, paths[device]
      )
    if digest(native) != digest(paths["cpu"]):
      raise ValueError(
        f"CPU did not reproduce its released PLY for {obj}/{arm}/{seconds}"
      )
    error = discrepancy(geometry["cpu"], geometry["cuda"])
    for device, mesh in geometry.items():
      metric = mesh_accuracy(mesh, truth)
      mechanisms[device].update(
        f5=metric["at_mm"]["5"]["fscore"], symmetric_mean_mm=metric["symmetric_mean_mm"]
      )
    if abs(mechanisms["cpu"]["f5"] - old["fscore"]) > 1e-12:
      raise ValueError(
        f"CPU did not reproduce its released F@5 for {obj}/{arm}/{seconds}"
      )
    cpu, gpu = (mechanisms[d] for d in ("cpu", "cuda"))
    if (cpu["contacts"], cpu["length_m"]) != (gpu["contacts"], gpu["length_m"]):
      raise ValueError("CPU and CUDA received different inputs or hyperparameters")
    data = dict(
      obj=obj,
      arm=arm,
      sim_s=seconds,
      order=list(order),
      protocol_sha256=spec,
      cpu=cpu,
      cuda=gpu,
      abs_f5_difference=abs(cpu["f5"] - gpu["f5"]),
      geometry=error,
      cpu_mesh_sha256=digest(native),
      cuda_mesh_sha256=digest(paths["cuda"]),
    )
    reference.parent.mkdir(exist_ok=True)
    paths["cpu"].replace(reference)
    paths["cuda"].replace(published)
    write(receipt, data)
  print(
    f"{obj}/{arm}/{seconds}s CPU {cpu['total_s']:.2f}s, CUDA {gpu['total_s']:.2f}s, "
    f"F@5 change {100 * data['abs_f5_difference']:.5f} pp, p95 {error['p95_mm']:.6f} mm",
    flush=True,
  )
  return data


def benchmark():
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is required to compare CPU and CUDA")
  torch.set_num_threads(1)
  OUT.mkdir(parents=True, exist_ok=True)
  hardware = torch.cuda.get_device_properties(0)
  release = read(STUDY / "results.json")
  spec = dict(
    study_sha256=digest(STUDY / "results.json"),
    cpu_and_grid_sha256=digest(Path(full_signed.__file__)),
    cuda_sha256=digest(Path(signed_cuda.__file__)),
    benchmark_sha256=digest(Path(__file__)),
    windows_s=list(WINDOWS),
    hardware=hardware.name,
    torch_version=str(torch.__version__),
    gpytorch_version=gpytorch.__version__,
    dtype="float64",
    cpu_torch_threads=torch.get_num_threads(),
    parity=dict(max_abs_f5=0.001, max_pair_p95_mm=0.1, max_pair_distance_mm=0.5),
    primary_speed="Mean complete time from reading recorded touches through PLY export across the ten 600 s outputs, CPU and CUDA in alternating order. Device discovery precedes timing. The first CUDA model and prediction are timed in its checkpoint.",
    geometry="Conservative bidirectional surface check: nearest of opposing vertices and triangle centroids, pooled within each direction; report the larger directional p95 and maximum.",
    inputs="Native measured fingertips; all five region loads below 8 N. Identical representatives, signed labels, covariance, regularization, length, 4 mm grid, support mask and zero-surface extraction.",
  )
  protocol = OUT / "protocol.json"
  if protocol.exists() and read(protocol) != spec:
    raise ValueError("Use a fresh directory for a changed offline benchmark")
  if not protocol.exists():
    write(protocol, spec)
  key = digest(protocol)
  results = []
  for j, (obj, item) in enumerate(release["objects"].items()):
    mesh = trimesh.load_mesh(fixture(obj)[1], process=False)
    truth = trimesh.sample.sample_surface(mesh, MESH_SAMPLES, seed=METRIC_SEED + 1)[0]
    for k, arm in enumerate(("wrist-dp", "native-aude")):
      row = item["native"][arm]
      folder = REPO / row["source"]
      if digest(folder / "trace.npz") != row["trace_sha256"]:
        raise ValueError("Native trajectory differs from registered study")
      archived = {
        int(s["s"]): s["signed"] for s in read(folder / "scores.json")["snapshots"]
      }
      for i, seconds in enumerate(WINDOWS):
        order = ("cpu", "cuda") if (i + j + k) % 2 == 0 else ("cuda", "cpu")
        results.append(
          run_pair(obj, arm, folder, seconds, order, key, truth, archived[seconds])
        )
  if len(results) != 40:
    raise ValueError("Expected all 40 reconstructions")
  tolerance = spec["parity"]
  passed = [
    r
    for r in results
    if (
      r["abs_f5_difference"] <= tolerance["max_abs_f5"]
      and r["geometry"]["p95_mm"] <= tolerance["max_pair_p95_mm"]
      and r["geometry"]["max_mm"] <= tolerance["max_pair_distance_mm"]
    )
  ]

  def summarize(rows):
    cpu, gpu = (
      float(np.mean([r[d]["total_s"] for r in rows])) for d in ("cpu", "cuda")
    )
    return dict(
      n=len(rows),
      cpu_s=cpu,
      cuda_s=gpu,
      speedup=cpu / gpu,
      cuda_faster=sum(r["cuda"]["total_s"] < r["cpu"]["total_s"] for r in rows),
    )

  final = summarize([r for r in results if r["sim_s"] == 600])
  result = dict(
    protocol=spec,
    protocol_sha256=key,
    all=summarize(results),
    final=final,
    matched=len(passed),
    same_ply=sum(r["cpu_mesh_sha256"] == r["cuda_mesh_sha256"] for r in results),
    max_abs_f5=max(r["abs_f5_difference"] for r in results),
    worst_p95_mm=max(r["geometry"]["p95_mm"] for r in results),
    worst_max_mm=max(r["geometry"]["max_mm"] for r in results),
    accept_cuda=len(passed) == len(results) and final["speedup"] > 1,
    runs=results,
  )
  write(OUT / "comparison.json", result)
  print(
    json.dumps(
      {k: v for k, v in result.items() if k not in {"runs", "protocol"}},
      ensure_ascii=False,
    ),
    flush=True,
  )


if __name__ == "__main__":
  benchmark()
