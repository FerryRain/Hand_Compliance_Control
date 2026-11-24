"""Freeze common, measured starts before running either Active system on the five YCB objects."""

import json
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

from ..acitive_02.active_a import save_model
from ..i04_geometric_traversal.fingertip_hand import FingertipHand, build_model
from ..i04_geometric_traversal.fingertip_traversal import STATE
from ..i04_geometric_traversal.fixtures import SurfaceGraph
from ..i04_geometric_traversal.i01_a_fair import initial_grasp
from ..i04_geometric_traversal.i01_a_full_wrist import EVALUATION, REVIEW
from ..i04_geometric_traversal.physics import rotation
from . import experiment
from .experiment import digest, seed_for
from .face_regression_suite import REPO, read, write
from .pear_mustard import FIXTURE
from .ycb_active import ROOT

OBJECTS = {
  "pear": ("016", "梨", "active-02"),
  "mustard": ("006", "芥末瓶", "start-01"),
  "cracker": ("003", "饼干盒", "start-01"),
  "bleach": ("021", "漂白清洁剂瓶", "start-01"),
  "coffee": ("002", "咖啡罐", "start-01"),
}
MODEL_IDS = {"bleach": "ycb_021_bleach_cleanser", "coffee": "ycb_002_master_chef_can"}
STUDY = ROOT / "five-objects"


def fixture(obj):
  if obj == "pear":
    return (
      experiment.ACTIVE2,
      experiment.GENERATED / "i01_a_policy_wrist_20261120/pear/plan/surface.obj",
    )
  if obj in ("mustard", "cracker"):
    folder = FIXTURE.parent.parent / obj / "fixed-dp_r01"
  else:
    folder = STUDY / "fixtures" / obj
  return folder, folder / "surface.obj"


def logged_grasp(hand, path):
  with np.load(path, allow_pickle=False) as f:
    force = np.linalg.norm(f["input_tip_force_world"][:, 0], axis=-1)
    valid = f["observed"][:, 0].astype(bool) & (f["reset_phase"][:, 0] == 0)
    tips = f["input_tip_found"][:, 0].astype(bool) & (force >= 0.2)
    pos, palm, q = (
      f[n][:, 0]
      for n in ("measured_object_pose_world", "input_palm_pose_world", "input_q_live")
    )
  candidates = np.flatnonzero(valid & tips.all(axis=1) & (force.max(axis=1) < 8))
  # Old recordings seed the common fixture only. Check all candidates chronologically in native physics.
  for frame in candidates:
    reference = rotation(pos[frame, 3:].astype(float)).inv()
    grasp = (
      reference.apply(palm[frame, :3].astype(float) - pos[frame, :3].astype(float)),
      reference * rotation(palm[frame, 3:].astype(float)),
      q[frame].astype(float),
    )
    touch = hand.measure(hand.initialize(*grasp))
    if touch.measured_heads.all() and touch.loads.max() < 8:
      return grasp, dict(
        method="first physically valid four-light-pad frame",
        frame=int(frame),
        record=path.relative_to(REPO).as_posix(),
        record_sha256=digest(path),
      )
  return None


def build(obj):
  folder, mesh = fixture(obj)
  if (folder / "initial.npz").exists() and (folder / "model.mjb").exists():
    return
  name = MODEL_IDS[obj]
  source = REVIEW / "ycb_expansion_v2/objects" / name / "surface.obj"
  parts = sorted(
    (REVIEW / "physics_contract_v1/ycb_policy_stress_v1" / name / "collision").glob(
      "collision_part_*.obj"
    )
  )
  if len(parts) != 128:
    raise ValueError(f"Missing original contact geometry for {obj}")
  model = build_model(
    surface=source,
    collision_parts=parts,
    object_name=obj,
    model_name=f"ycb-active-{obj}",
  )
  if (folder / "initial.npz").exists():
    if read(folder / "origin.json")["scoring_mesh_sha256"] != digest(source):
      raise ValueError("Cannot resume a fixture built with different geometry")
    save_model(model, folder / "model.mjb.tmp")
    (folder / "model.mjb.tmp").replace(folder / "model.mjb")
    return
  hand = FingertipHand(model, object_name=obj)
  hand.inward_in_palm[:] = (0.0, 0.0, -1.0)
  found = logged_grasp(hand, EVALUATION / (name + ".npz"))
  if found is None:
    graph = SurfaceGraph(
      trimesh.load_mesh(source, process=False), coverage_radius_m=0.025
    )
    grasp, _, receipt = initial_grasp(hand, graph, np.array([0.0, 0.0, 1.0]))
  else:
    grasp, receipt = found
  data = hand.initialize(*grasp)
  touch = hand.measure(data)
  if not touch.measured_heads.all() or touch.loads.max() >= 8:
    raise ValueError("Common start failed the low-load test")
  state = np.empty(mujoco.mj_stateSize(model, STATE))
  mujoco.mj_getState(model, data, state, STATE)
  folder.parent.mkdir(parents=True, exist_ok=True)
  with TemporaryDirectory(dir=folder.parent) as temp:
    trial = Path(temp) / "fixture"
    trial.mkdir()
    copyfile(source, trial / "surface.obj")
    save_model(model, trial / "model.mjb")
    np.savez(trial / "initial.npz", integration_state=state)
    write(
      trial / "origin.json",
      dict(
        receipt,
        object_id=name,
        parts=len(parts),
        forces_n=touch.forces.tolist(),
        region_load_n=touch.loads.tolist(),
        scoring_mesh_sha256=digest(mesh if mesh.exists() else source),
      ),
    )
    trial.rename(folder)


def prepare():
  for obj in MODEL_IDS:
    build(obj)
  fixtures = {}
  for obj, (_, _, start) in OBJECTS.items():
    folder, mesh = fixture(obj)
    with patch.object(experiment, "scenario", return_value=(folder, mesh)):
      (plant, data, hand, sensed, touch, state, warm) = experiment.setup(obj, start)
    fixtures[obj + "/" + start] = {
      "model.mjb": digest(folder / "model.mjb"),
      "initial.npz": digest(folder / "initial.npz"),
      "surface.obj": digest(mesh),
      "integration_state": sha256(state.tobytes()).hexdigest(),
      "pad_force_n": touch.forces.tolist(),
      "region_load_n": touch.loads.tolist(),
      "seed": seed_for(obj, start),
      "source": folder.relative_to(REPO).as_posix(),
      "origin": read(folder / "origin.json")
      if (folder / "origin.json").exists()
      else {"method": "archived four-pad start"},
    }
  base = read(ROOT / "protocol.json")
  release = read(ROOT / "release.json")
  spec = dict(
    objects=OBJECTS,
    fixtures=fixtures,
    checkpoint=base["checkpoint"],
    primary=base["primary"],
    thresholds=base["thresholds"],
    controller=release["chosen"],
    release_sha256=digest(ROOT / "release.json"),
    budget_s=600,
    primary_budget_s=240,
    scope="Five specified YCB objects, one paired four-pad start each. Mustard used for revision selection. Other four did not select the revision; pear had been evaluated previously. All objects evaluate exploration from established contact.",
    aggregate="Object-level arithmetic mean; for each time threshold report hit count and mean min(time,600); non-hits contribute 600 seconds, never presented as hits.",
  )
  path = STUDY / "protocol.json"
  spec = json.loads(json.dumps(spec))
  if path.exists() and read(path) != spec:
    raise ValueError("Frozen five-object conditions changed")
  if not path.exists():
    write(path, spec)
  return spec


if __name__ == "__main__":
  for key, row in prepare()["fixtures"].items():
    print(key, row["pad_force_n"], row["origin"]["method"], flush=True)
