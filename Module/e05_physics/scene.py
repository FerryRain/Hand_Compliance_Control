"""Build a clean Leap Hand contact scene from the repository's visual asset."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.e05_physics.extreme_surface import (
  BASE_DEPTH_M,
  N_COL,
  N_ROW,
  X_HALF_M,
  Y_HALF_M,
  hfield_elevation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_XML = REPO_ROOT / "src/mjlab/asset_zoo/robots/leaphand_only.xml"
PAD_HALF_SIZE_M = np.array([0.012, 0.008, 0.002], dtype=np.float64)
# The pads are registered in each physical fingertip-body frame.  Local +Y is
# proximal, local -X is the palmar/outward direction, and local Z spans the
# finger width.  The pad long axis is therefore body +Y and its thin/outward
# axis is body -X.  The width axis is body -Z so the three axes form a proper
# right-handed frame. No world-frame rotation is baked into the geom.
PAD_LOCAL_ROTATION = np.array(
  [
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
  ],
  dtype=np.float64,
)
Q_NOMINAL = np.array(
  [
    0.00005,
    0.68161,
    -0.40600,
    -0.26600,
    0.00003,
    0.68161,
    -0.40600,
    -0.26600,
    0.00013,
    0.68161,
    -0.40600,
    -0.26600,
    0.53481,
    1.57006,
    0.10087,
    -0.63505,
  ],
  dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class FingerSpec:
  finger_id: int
  name: str
  tip_body: str
  fsr_body: str
  proxy_geom: str
  pad_site: str
  joint_names: tuple[str, ...]
  color: tuple[float, float, float, float]
  pad_local_position_m: tuple[float, float, float]
  distal_head_y_m: float


FINGERS = (
  FingerSpec(1, "index", "fingertip", "tip_1_fsr", "e05_tip_1", "e05_pad_site_1", ("0", "1", "2", "3"), (0.16, 0.58, 0.86, 1.0), (-0.0126, -0.0240, 0.0141088), -0.0495),
  FingerSpec(2, "middle", "fingertip_2", "tip_2_fsr", "e05_tip_2", "e05_pad_site_2", ("4", "5", "6", "7"), (0.24, 0.72, 0.60, 1.0), (-0.0126, -0.0240, 0.0144487), -0.0495),
  FingerSpec(3, "ring", "fingertip_3", "tip_3_fsr", "e05_tip_3", "e05_pad_site_3", ("8", "9", "10", "11"), (0.95, 0.57, 0.15, 1.0), (-0.0126, -0.0240, 0.0140386), -0.0495),
  FingerSpec(4, "thumb", "thumb_fingertip", "thumb_tip_fsr", "e05_tip_4", "e05_pad_site_4", ("12", "13", "14", "15"), (0.93, 0.32, 0.47, 1.0), (-0.0126, -0.0300, -0.0144321), -0.0621),
)


@dataclass(frozen=True, slots=True)
class ObjectSpec:
  shape: str
  initial_position: NDArray[np.float64]
  initial_quaternion: NDArray[np.float64]
  size: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SceneHandles:
  model: mujoco.MjModel
  object_spec: ObjectSpec
  object_body_id: int
  object_geom_id: int
  object_mocap_id: int
  tip_body_ids: NDArray[np.int32]
  tip_geom_ids: NDArray[np.int32]
  tip_site_ids: NDArray[np.int32]
  finger_dof_adrs: tuple[NDArray[np.int32], ...]
  finger_qpos_adrs: tuple[NDArray[np.int32], ...]
  joint_qpos_adrs: NDArray[np.int32]
  joint_dof_adrs: NDArray[np.int32]
  joint_ranges_rad: NDArray[np.float64]


def _object_spec(shape: str) -> ObjectSpec:
  if shape == "plane":
    return ObjectSpec(
      shape="plane",
      initial_position=np.array([-0.025, -0.025, -0.010]),
      initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
      size=np.array([0.15, 0.15, 0.01]),
    )
  if shape == "sphere":
    return ObjectSpec(
      shape="sphere",
      initial_position=np.array([0.005, -0.005, -0.443]),
      initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
      size=np.array([0.45, 0.0, 0.0]),
    )
  if shape == "extreme":
    _, minimum, span = hfield_elevation()
    return ObjectSpec(
      shape="extreme",
      initial_position=np.array([-0.015, 0.245, -0.006]),
      initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
      size=np.array([X_HALF_M, Y_HALF_M, span, BASE_DEPTH_M, minimum]),
    )
  raise ValueError("shape must be 'plane', 'sphere', or 'extreme'")


def _set_clean_collision_groups(root: ET.Element) -> None:
  for geom in root.iter("geom"):
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    # The source model paints every FSR mesh red.  That made the new collision
    # pads visually ambiguous, so keep sensors as neutral visual geometry and
    # reserve the four vivid colors exclusively for the physical pad proxies.
    if "_fsr_geom" in geom.get("name", ""):
      geom.set("rgba", "0.48 0.50 0.54 1")


def _remove_legacy_object(root: ET.Element) -> None:
  palm = root.find(".//body[@name='palm_lower']")
  if palm is None:
    raise RuntimeError("palm_lower body is missing from the Leap Hand XML")
  legacy_object = palm.find("./body[@name='object_body']")
  if legacy_object is not None:
    palm.remove(legacy_object)


def _matrix_to_quaternion(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
  quaternion = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
  return quaternion


def _mesh_registered_pad_transforms(
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
  """Return pad frames fixed to the actual fingertip mesh/body coordinates."""

  quaternion = _matrix_to_quaternion(PAD_LOCAL_ROTATION)
  return tuple(
    (
      np.asarray(finger.pad_local_position_m, dtype=np.float64),
      quaternion.copy(),
    )
    for finger in FINGERS
  )


def _add_tip_proxies(
  root: ET.Element,
  transforms: tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...],
) -> None:
  for finger, (position, quaternion) in zip(FINGERS, transforms):
    body = root.find(f".//body[@name='{finger.tip_body}']")
    if body is None:
      raise RuntimeError(f"missing physical fingertip body: {finger.tip_body}")
    ET.SubElement(
      body,
      "geom",
      {
        "name": finger.proxy_geom,
        "type": "ellipsoid",
        "size": " ".join(str(value) for value in PAD_HALF_SIZE_M),
        "pos": " ".join(str(value) for value in position),
        "quat": " ".join(str(value) for value in quaternion),
        "density": "0",
        "contype": "1",
        "conaffinity": "2",
        "friction": "0.9 0.01 0.001",
        "solref": "0.02 1",
        "solimp": "0.90 0.95 0.001",
        "rgba": " ".join(str(value) for value in finger.color),
      },
    )
    ET.SubElement(
      body,
      "site",
      {
        "name": finger.pad_site,
        "type": "sphere",
        "size": "0.001",
        "pos": " ".join(str(value) for value in position),
        "quat": " ".join(str(value) for value in quaternion),
        "rgba": "0 0 0 0",
      },
    )
    outward = PAD_LOCAL_ROTATION[:, 2]
    marker_position = position + 0.008 * outward
    ET.SubElement(
      body,
      "site",
      {
        "name": f"{finger.pad_site}_normal_marker",
        "type": "cylinder",
        "size": "0.0012 0.006",
        "pos": " ".join(str(value) for value in marker_position),
        "quat": " ".join(str(value) for value in quaternion),
        "rgba": " ".join(str(value) for value in finger.color),
      },
    )


def _add_object(root: ET.Element, spec: ObjectSpec) -> None:
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise RuntimeError("worldbody is missing from the Leap Hand XML")
  body = ET.SubElement(
    worldbody,
    "body",
    {
      "name": "e05_object",
      "mocap": "true",
      "pos": " ".join(str(value) for value in spec.initial_position),
      "quat": "1 0 0 0",
    },
  )
  attributes = {
    "name": "e05_object_geom",
    "density": "0",
    "contype": "2",
    "conaffinity": "1",
    "friction": "0.9 0.01 0.001",
    "solref": "0.02 1",
    "solimp": "0.90 0.95 0.001",
    "rgba": "0.16 0.44 0.90 0.92",
  }
  if spec.shape == "plane":
    attributes.update(
      {
        "type": "box",
        "size": " ".join(str(value) for value in spec.size),
      }
    )
  elif spec.shape == "sphere":
    attributes.update({"type": "sphere", "size": str(float(spec.size[0]))})
  else:
    asset = root.find("asset")
    if asset is None:
      raise RuntimeError("asset section is missing from the Leap Hand XML")
    elevation, minimum, span = hfield_elevation()
    ET.SubElement(
      asset,
      "hfield",
      {
        "name": "e05_extreme_hfield",
        "nrow": str(N_ROW),
        "ncol": str(N_COL),
        "size": f"{X_HALF_M} {Y_HALF_M} {span} {BASE_DEPTH_M}",
        "elevation": " ".join(str(value) for value in elevation.ravel()),
      },
    )
    attributes.update(
      {
        "type": "hfield",
        "hfield": "e05_extreme_hfield",
        "pos": f"0 0 {minimum}",
        "rgba": "0.18 0.38 0.78 0.96",
      }
    )
  ET.SubElement(body, "geom", attributes)


def build_scene(shape: str = "plane", *, timestep_s: float = 0.002) -> SceneHandles:
  if timestep_s <= 0.0:
    raise ValueError("timestep_s must be positive")
  root = ET.parse(SOURCE_XML).getroot()
  compiler = root.find("compiler")
  if compiler is None:
    raise RuntimeError("compiler element is missing from the Leap Hand XML")
  compiler.set("meshdir", str(SOURCE_XML.parent.resolve()))
  option = root.find("option")
  if option is None:
    raise RuntimeError("option element is missing from the Leap Hand XML")
  option.set("timestep", str(timestep_s))
  option.set("gravity", "0 0 0")
  option.set("iterations", "80")
  _set_clean_collision_groups(root)
  _remove_legacy_object(root)
  pad_transforms = _mesh_registered_pad_transforms()
  _add_tip_proxies(root, pad_transforms)
  spec = _object_spec(shape)
  _add_object(root, spec)
  xml = ET.tostring(root, encoding="unicode")
  model = mujoco.MjModel.from_xml_string(xml)

  object_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "e05_object",
  )
  object_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "e05_object_geom",
  )
  object_mocap_id = int(model.body_mocapid[object_body_id])
  tip_body_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger.tip_body)
      for finger in FINGERS
    ],
    dtype=np.int32,
  )
  tip_geom_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, finger.proxy_geom)
      for finger in FINGERS
    ],
    dtype=np.int32,
  )
  tip_site_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, finger.pad_site)
      for finger in FINGERS
    ],
    dtype=np.int32,
  )
  joint_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(index))
      for index in range(16)
    ],
    dtype=np.int32,
  )
  joint_qpos_adrs = np.array(
    [model.jnt_qposadr[joint_id] for joint_id in joint_ids],
    dtype=np.int32,
  )
  joint_dof_adrs = np.array(
    [model.jnt_dofadr[joint_id] for joint_id in joint_ids],
    dtype=np.int32,
  )
  finger_qpos_adrs: list[NDArray[np.int32]] = []
  finger_dof_adrs: list[NDArray[np.int32]] = []
  for finger in FINGERS:
    ids = np.array(
      [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in finger.joint_names
      ],
      dtype=np.int32,
    )
    finger_qpos_adrs.append(
      np.array([model.jnt_qposadr[joint_id] for joint_id in ids], dtype=np.int32)
    )
    finger_dof_adrs.append(
      np.array([model.jnt_dofadr[joint_id] for joint_id in ids], dtype=np.int32)
    )
  return SceneHandles(
    model=model,
    object_spec=spec,
    object_body_id=object_body_id,
    object_geom_id=object_geom_id,
    object_mocap_id=object_mocap_id,
    tip_body_ids=tip_body_ids,
    tip_geom_ids=tip_geom_ids,
    tip_site_ids=tip_site_ids,
    finger_dof_adrs=tuple(finger_dof_adrs),
    finger_qpos_adrs=tuple(finger_qpos_adrs),
    joint_qpos_adrs=joint_qpos_adrs,
    joint_dof_adrs=joint_dof_adrs,
    joint_ranges_rad=np.array(model.jnt_range[joint_ids], dtype=np.float64, copy=True),
  )
