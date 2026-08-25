"""Deterministically assemble a controllable FR3 and Leap Hand MuJoCo plant.

The repository's FR3 XML contains only seven passive joints, while the Leap
Hand XML is a fixed-palm demo containing a legacy free object.  This builder
keeps the upstream assets immutable and creates the actual 23-DoF robot needed
by M0--M4:

* 7 actuated FR3 joints and 16 actuated Leap Hand joints;
* a fixed, explicit flange-to-palm transform;
* four fingertip-body belly-pad collision proxies and sites;
* a fixed world object (the hand moves; the object does not);
* wrist/palm sites and sensor channels.

The mount orientation is chosen so the nominal hand pose has the same
downward-facing physical fingertip bellies as the audited E05-PHY-v3 hand.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

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
from Module.e05_physics.scene import (
  FINGERS,
  PAD_HALF_SIZE_M,
  PAD_LOCAL_ROTATION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_XML = REPO_ROOT / "src/mjlab/asset_zoo/robots/fr3_leap_hand/fr3v2_collision.xml"
HAND_XML = REPO_ROOT / "src/mjlab/asset_zoo/robots/leaphand_only.xml"

ARM_JOINT_NAMES = tuple(f"fr3v2_joint{index}" for index in range(1, 8))
HAND_JOINT_NAMES = tuple(str(index) for index in range(16))
ARM_HOME_Q = np.array(
  [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
  dtype=np.float64,
)
# Exact measured q_hand at t=2.000 s in the already published raw-DP video.
# It was re-extracted deterministically from commit c5090d6 and checkpoint
# SHA-256 89044a...ef0, rather than approximated by the checkpoint mean.  The
# three long fingers are naturally curled and the thumb is opposed.  E05-MCC
# uses this as its posture null-space reference; it is not a DP command and
# does not make DP part of the MCC controller.
HAND_NATURAL_Q = np.array(
  [
    0.21093411785976737,
    0.005229204743344924,
    0.4035944678008984,
    0.5287342369050559,
    0.20371661194285254,
    0.00995644973119566,
    0.40966995188205774,
    0.5316287796080648,
    0.2090436955272734,
    0.012743144261669894,
    0.40656648217964514,
    0.528036463232157,
    -0.05238612040473543,
    1.5615033691412608,
    0.3896256643566552,
    0.5183713092524893,
  ],
  dtype=np.float64,
)
FULL_HOME_Q = np.concatenate((ARM_HOME_Q, HAND_NATURAL_Q)).astype(np.float64)
HAND_MODEL_JOINT_ORDER = (1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15)
MODEL_HOME_QPOS = np.concatenate(
  (
    ARM_HOME_Q,
    HAND_NATURAL_Q[np.asarray(HAND_MODEL_JOINT_ORDER, dtype=np.int32)],
  )
).astype(np.float64)

# The Leap XML body origin lies near a motor/knuckle.  Mounting that origin to
# link8 made the kinematic tree valid but put the physical connector on the
# motor cluster.  The explicit mount point below lies on the central palm mesh
# (about 50/54 percent through its X/Y bounding box).  The palm root transform
# is solved so this point, rather than the body origin, meets the FR3 axis.
# Local Z is the flange-facing exterior of the palm mesh.  Using the opposite
# face would bury the adapter about 23 mm inside the palm even though its axis
# passed through the right X/Y location.
PALM_MOUNT_LOCAL_M = np.array(
  [-0.048, -0.032, 0.0112776],
  dtype=np.float64,
)
MOUNT_INTERFACE_POSITION_LINK8_M = np.array([0.0, 0.0, 0.0112], dtype=np.float64)
MOUNT_ADAPTER_RADIUS_M = 0.028
MOUNT_INTERFACE_TOLERANCE_M = 0.001
MOUNT_QUATERNION_LINK8 = np.array(
  [0.0, 0.38272878, 0.92386075, 0.0],
  dtype=np.float64,
)


def _quaternion_to_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  matrix = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)


MOUNT_PALM_ROOT_POSITION_LINK8_M = (
  MOUNT_INTERFACE_POSITION_LINK8_M
  - _quaternion_to_matrix(MOUNT_QUATERNION_LINK8) @ PALM_MOUNT_LOCAL_M
)

# The fixed surface is a rigid translation of the old fixed-palm coordinate
# system.  This preserves the already audited initial contact geometry while
# reversing the execution authority: FR3 moves over a fixed world object.
SOURCE_PALM_WORLD_POSITION_M = np.array([0.0, 0.0, 0.08], dtype=np.float64)
SOURCE_OBJECT_POSITIONS_M = {
  "plane": np.array([-0.025, -0.025, -0.010], dtype=np.float64),
  "sphere": np.array([0.005, -0.005, -0.443], dtype=np.float64),
  # Raised 11 mm relative to the retired stretched-finger seed so the exact
  # t=2.0 s DP-video posture starts within a small physical MAKE distance.
  "extreme": np.array([-0.015, 0.245, 0.005], dtype=np.float64),
  # I01 lays the scaled Bunny on its side.  The upper envelope underneath the
  # natural fingertip footprint is about 0.12 m high, so this body origin
  # preserves the same ~18 mm physical MAKE gap as the plane baseline.
  "bunny": np.array([-0.0115, -0.052, -0.120], dtype=np.float64),
}


@dataclass(frozen=True, slots=True)
class FullRobotModelConfig:
  """Configuration frozen by the MCC-only E05 protocol."""

  surface: str = "extreme"
  timestep_s: float = 0.002
  gravity_m_s2: float = -9.81
  arm_kp: float = 600.0
  arm_damping_ratio: float = 1.0
  hand_kp: float = 22.0
  hand_damping_ratio: float = 1.5
  hand_actuator_force_limit_nm: float | None = None
  # Dataset-I uses translated crops of the same frozen large hfield as distinct
  # object-region variants.  Defaults remain exactly the MCC/E05 geometry.
  object_offset_x_m: float = 0.0
  object_offset_y_m: float = 0.0
  object_offset_z_m: float = 0.0
  # Optional canonical, already transformed Bunny mesh used only for visual
  # rendering in the legacy hfield scene.  I04 can opt into MuJoCo's native
  # non-convex mesh SDF, in which case the same canonical file is the physical
  # collision geometry as well as the rendered geometry.
  bunny_visual_mesh_path: str | None = None
  bunny_collision_mode: str = "hfield"

  def __post_init__(self) -> None:
    if self.surface not in {"plane", "sphere", "extreme", "bunny"}:
      raise ValueError("surface must be 'plane', 'sphere', 'extreme', or 'bunny'")
    positive = {
      "timestep_s": self.timestep_s,
      "arm_kp": self.arm_kp,
      "arm_damping_ratio": self.arm_damping_ratio,
      "hand_kp": self.hand_kp,
      "hand_damping_ratio": self.hand_damping_ratio,
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.hand_actuator_force_limit_nm is not None:
      limit = float(self.hand_actuator_force_limit_nm)
      if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("hand_actuator_force_limit_nm must be finite and positive")
    if not np.isfinite(self.gravity_m_s2):
      raise ValueError("gravity_m_s2 must be finite")
    if (
      not np.isfinite(self.object_offset_x_m)
      or not np.isfinite(self.object_offset_y_m)
      or not np.isfinite(self.object_offset_z_m)
    ):
      raise ValueError("object offsets must be finite")
    if self.bunny_visual_mesh_path is not None:
      mesh_path = Path(self.bunny_visual_mesh_path).expanduser().resolve()
      if not mesh_path.is_file():
        raise ValueError(f"bunny_visual_mesh_path does not exist: {mesh_path}")
      object.__setattr__(self, "bunny_visual_mesh_path", str(mesh_path))
    if self.bunny_collision_mode not in {"hfield", "sdf"}:
      raise ValueError("bunny_collision_mode must be 'hfield' or 'sdf'")
    if self.surface != "bunny" and self.bunny_collision_mode != "hfield":
      raise ValueError("bunny_collision_mode='sdf' requires surface='bunny'")
    if self.bunny_collision_mode == "sdf" and self.bunny_visual_mesh_path is None:
      raise ValueError(
        "bunny_collision_mode='sdf' requires bunny_visual_mesh_path"
      )


@dataclass(frozen=True, slots=True)
class FullRobotHandles:
  model: mujoco.MjModel
  xml: str
  config: FullRobotModelConfig
  arm_joint_ids: NDArray[np.int32]
  hand_joint_ids: NDArray[np.int32]
  arm_qpos_adrs: NDArray[np.int32]
  hand_qpos_adrs: NDArray[np.int32]
  arm_dof_adrs: NDArray[np.int32]
  hand_dof_adrs: NDArray[np.int32]
  arm_actuator_ids: NDArray[np.int32]
  hand_actuator_ids: NDArray[np.int32]
  finger_qpos_adrs: tuple[NDArray[np.int32], ...]
  finger_dof_adrs: tuple[NDArray[np.int32], ...]
  tip_body_ids: NDArray[np.int32]
  tip_geom_ids: NDArray[np.int32]
  tip_site_ids: NDArray[np.int32]
  palm_body_id: int
  palm_site_id: int
  palm_mount_site_id: int
  wrist_site_id: int
  mount_interface_site_id: int
  object_body_id: int
  object_geom_id: int
  object_mocap_id: int
  robot_geom_ids: NDArray[np.int32]
  arm_joint_ranges_rad: NDArray[np.float64]
  hand_joint_ranges_rad: NDArray[np.float64]
  object_position_m: NDArray[np.float64]


def _absolute_mesh_files(root: ET.Element, source: Path) -> None:
  asset = root.find("asset")
  if asset is None:
    raise RuntimeError(f"asset section missing from {source}")
  for mesh in asset.findall("mesh"):
    filename = mesh.get("file")
    if filename is None:
      raise RuntimeError(f"mesh without a file in {source}")
    mesh.set("file", str((source.parent / filename).resolve()))


def _matrix_to_quaternion(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
  quaternion = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
  return quaternion


def _vector_text(values: NDArray[np.float64] | tuple[float, ...]) -> str:
  return " ".join(f"{float(value):.12g}" for value in values)


def _merge_assets(arm_root: ET.Element, hand_root: ET.Element) -> None:
  arm_asset = arm_root.find("asset")
  hand_asset = hand_root.find("asset")
  if arm_asset is None or hand_asset is None:
    raise RuntimeError("both robot sources must contain an asset section")
  existing_names = {element.get("name") for element in arm_asset}
  for element in hand_asset:
    name = element.get("name")
    if name in existing_names:
      raise RuntimeError(f"duplicate merged asset name: {name}")
    arm_asset.append(deepcopy(element))
    existing_names.add(name)


def _add_leap_default(root: ET.Element, config: FullRobotModelConfig) -> None:
  default = root.find("default")
  if default is None:
    raise RuntimeError("FR3 source is missing its default section")
  leap = ET.SubElement(default, "default", {"class": "fr3_leap_hand"})
  ET.SubElement(
    leap,
    "joint",
    {"damping": "0.4", "frictionloss": "0.05", "armature": "0.001"},
  )
  ET.SubElement(
    leap,
    "geom",
    {
      "friction": "1.3 0.01 0.001",
      "solimp": "0.99 0.995 0.0005",
      "solref": "0.003 1",
    },
  )
  ET.SubElement(
    leap,
    "position",
    {
      "kp": str(config.hand_kp),
      "dampratio": str(config.hand_damping_ratio),
      "ctrlrange": "-3.2 3.2",
    },
  )


def _prepare_palm(hand_root: ET.Element) -> ET.Element:
  palm_source = hand_root.find(".//body[@name='palm_lower']")
  if palm_source is None:
    raise RuntimeError("palm_lower is missing from the Leap Hand source")
  palm = deepcopy(palm_source)
  legacy_object = palm.find("./body[@name='object_body']")
  if legacy_object is not None:
    palm.remove(legacy_object)
  palm.set("pos", _vector_text(MOUNT_PALM_ROOT_POSITION_LINK8_M))
  palm.set("quat", _vector_text(MOUNT_QUATERNION_LINK8))
  palm.set("childclass", "fr3_leap_hand")

  # The detailed hand meshes remain visible.  Only the registered physical
  # belly pads can contact the object in E05, making the contact source
  # auditable and preventing a rounded fingertip head from substituting for
  # the requested finger-belly contact.
  for geom in palm.iter("geom"):
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("group", "0")
    if "_fsr_geom" in geom.get("name", ""):
      geom.set("rgba", "0.48 0.50 0.54 1")

  pad_quaternion = _matrix_to_quaternion(PAD_LOCAL_ROTATION)
  for finger in FINGERS:
    body = palm.find(f".//body[@name='{finger.tip_body}']")
    if body is None:
      raise RuntimeError(f"missing fingertip body: {finger.tip_body}")
    position = np.asarray(finger.pad_local_position_m, dtype=np.float64)
    ET.SubElement(
      body,
      "geom",
      {
        "name": finger.proxy_geom.replace("e05_", "fr3_"),
        "type": "ellipsoid",
        "size": _vector_text(PAD_HALF_SIZE_M),
        "pos": _vector_text(position),
        "quat": _vector_text(pad_quaternion),
        "density": "0",
        "contype": "1",
        "conaffinity": "2",
        "friction": "0.9 0.01 0.001",
        "solref": "0.028 1",
        "solimp": "0.90 0.95 0.001",
        "rgba": _vector_text(finger.color),
        "group": "2",
      },
    )
    site_name = finger.pad_site.replace("e05_", "fr3_")
    ET.SubElement(
      body,
      "site",
      {
        "name": site_name,
        "type": "ellipsoid",
        "size": _vector_text(PAD_HALF_SIZE_M),
        "pos": _vector_text(position),
        "quat": _vector_text(pad_quaternion),
        "rgba": "0 0 0 0",
      },
    )
    outward = PAD_LOCAL_ROTATION[:, 2]
    ET.SubElement(
      body,
      "site",
      {
        "name": f"{site_name}_normal_marker",
        "type": "cylinder",
        "size": "0.0012 0.006",
        "pos": _vector_text(position + 0.008 * outward),
        "quat": _vector_text(pad_quaternion),
        "rgba": _vector_text(finger.color),
      },
    )

  # This site is the sole pose/wrench reference used by contracts, IK and the
  # coordinator.  It is intentionally distinct from the source palm_center.
  ET.SubElement(
    palm,
    "site",
    {
      "name": "fr3_palm_control_site",
      "pos": "-0.048 -0.032 -0.002",
      "size": "0.005",
      "rgba": "0.15 0.95 0.30 0.8",
    },
  )
  ET.SubElement(
    palm,
    "site",
    {
      "name": "fr3_palm_mount_site",
      "pos": _vector_text(PALM_MOUNT_LOCAL_M),
      "type": "sphere",
      "size": "0.0045",
      "rgba": "0.10 0.95 0.35 0.9",
      "group": "3",
    },
  )
  return palm


def _home_palm_position_without_surface(root: ET.Element) -> NDArray[np.float64]:
  """Compile one cheap pass to obtain the exact mounted palm world position."""

  probe = deepcopy(root)
  keyframe = probe.find("keyframe")
  if keyframe is not None:
    probe.remove(keyframe)
  model = mujoco.MjModel.from_xml_string(ET.tostring(probe, encoding="unicode"))
  data = mujoco.MjData(model)
  data.qpos[:] = MODEL_HOME_QPOS
  mujoco.mj_forward(model, data)
  palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower")
  return np.array(data.xpos[palm_id], dtype=np.float64, copy=True)


def _add_fixed_surface(
  root: ET.Element,
  config: FullRobotModelConfig,
  palm_home_position: NDArray[np.float64],
) -> NDArray[np.float64]:
  worldbody = root.find("worldbody")
  asset = root.find("asset")
  if worldbody is None or asset is None:
    raise RuntimeError("FR3 model is missing worldbody or asset")
  translation = palm_home_position - SOURCE_PALM_WORLD_POSITION_M
  object_position = SOURCE_OBJECT_POSITIONS_M[config.surface] + translation
  object_position[:2] += np.array(
    [config.object_offset_x_m, config.object_offset_y_m],
    dtype=np.float64,
  )
  object_position[2] += config.object_offset_z_m
  body = ET.SubElement(
    worldbody,
    "body",
    {"name": "fr3_e05_object", "pos": _vector_text(object_position)},
  )
  attributes = {
    "name": "fr3_e05_object_geom",
    "density": "0",
    "contype": "2",
    "conaffinity": "1",
    "friction": "0.9 0.01 0.001",
    "solref": "0.028 1",
    "solimp": "0.90 0.95 0.001",
    "rgba": "0.18 0.38 0.78 0.96",
    "group": "2",
  }
  if config.surface == "plane":
    attributes.update({"type": "box", "size": "0.30 0.42 0.01"})
  elif config.surface == "sphere":
    attributes.update({"type": "sphere", "size": "0.45"})
  elif config.surface == "extreme":
    elevation, minimum, span = hfield_elevation()
    ET.SubElement(
      asset,
      "hfield",
      {
        "name": "fr3_e05_extreme_hfield",
        "nrow": str(N_ROW),
        "ncol": str(N_COL),
        "size": f"{X_HALF_M} {Y_HALF_M} {span} {BASE_DEPTH_M}",
        "elevation": " ".join(str(value) for value in elevation.ravel()),
      },
    )
    attributes.update(
      {
        "type": "hfield",
        "hfield": "fr3_e05_extreme_hfield",
        "pos": f"0 0 {minimum}",
      }
    )
  else:
    # Import lazily so the standard M0--M4 plant does not pay the mesh/raycast
    # construction cost.  I01--I03 retain their deterministic upper-envelope
    # hfield.  I04 instead uses MuJoCo's mesh-SDF collision path so contact is
    # possible on the Bunny's sides, underside and concavities without silently
    # replacing it by a convex hull.
    from Module.i01_bunny_physics.surface import (
      BUNNY_BASE_DEPTH_M,
      BUNNY_HFIELD_NCOL,
      BUNNY_HFIELD_NROW,
      canonical_bunny_heightfield,
    )

    if config.bunny_collision_mode == "sdf":
      ET.SubElement(
        asset,
        "mesh",
        {
          "name": "fr3_i04_bunny_sdf_mesh",
          "file": config.bunny_visual_mesh_path,
        },
      )
      attributes.update(
        {
          "type": "sdf",
          "mesh": "fr3_i04_bunny_sdf_mesh",
          "pos": "0 0 0",
          "rgba": "0.18 0.38 0.78 0.96",
        }
      )
    else:
      bunny = canonical_bunny_heightfield()
      ET.SubElement(
        asset,
        "hfield",
        {
          "name": "fr3_i01_bunny_hfield",
          "nrow": str(BUNNY_HFIELD_NROW),
          "ncol": str(BUNNY_HFIELD_NCOL),
          "size": (
            f"{bunny.x_half_m} {bunny.y_half_m} "
            f"{bunny.height_span_m} {BUNNY_BASE_DEPTH_M}"
          ),
          "elevation": " ".join(
            str(value) for value in bunny.mujoco_elevation().ravel()
          ),
        },
      )
      attributes.update(
        {
          "type": "hfield",
          "hfield": "fr3_i01_bunny_hfield",
          "pos": "0 0 0",
          "rgba": "0.20 0.54 0.82 0.14",
          "group": "4",
        }
      )
      if config.bunny_visual_mesh_path is not None:
        ET.SubElement(
          asset,
          "mesh",
          {
            "name": "fr3_i01_bunny_visual_mesh",
            "file": config.bunny_visual_mesh_path,
          },
        )
        ET.SubElement(
          body,
          "geom",
          {
            "name": "fr3_i01_bunny_visual_geom",
            "type": "mesh",
            "mesh": "fr3_i01_bunny_visual_mesh",
            "density": "0",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.18 0.38 0.78 0.96",
            "group": "2",
          },
        )
  ET.SubElement(body, "geom", attributes)
  return object_position


def _add_actuators_and_sensors(
  root: ET.Element,
  hand_root: ET.Element,
  config: FullRobotModelConfig,
) -> None:
  actuator = ET.SubElement(root, "actuator")
  arm_source_joints = {
    joint.get("name"): joint for joint in root.findall(".//joint")
  }
  for index, joint_name in enumerate(ARM_JOINT_NAMES, start=1):
    joint = arm_source_joints[joint_name]
    force_range = joint.get("actuatorfrcrange", "-12 12")
    joint_range = joint.get("range", "-3.2 3.2")
    ET.SubElement(
      actuator,
      "position",
      {
        "name": f"fr3_arm_act_{index}",
        "joint": joint_name,
        "kp": str(config.arm_kp),
        "dampratio": str(config.arm_damping_ratio),
        "ctrlrange": joint_range,
        "forcerange": force_range,
      },
    )
  source_actuator = hand_root.find("actuator")
  if source_actuator is None:
    raise RuntimeError("Leap Hand source is missing actuators")
  for element in source_actuator:
    copied = deepcopy(element)
    copied.set("class", "fr3_leap_hand")
    if config.hand_actuator_force_limit_nm is not None:
      limit = float(config.hand_actuator_force_limit_nm)
      copied.set("forcerange", f"{-limit:.12g} {limit:.12g}")
      copied.set("forcelimited", "true")
    actuator.append(copied)

  sensor = ET.SubElement(root, "sensor")
  ET.SubElement(sensor, "framepos", {"name": "palm_position", "objtype": "site", "objname": "fr3_palm_control_site"})
  ET.SubElement(sensor, "framequat", {"name": "palm_quaternion", "objtype": "site", "objname": "fr3_palm_control_site"})
  ET.SubElement(sensor, "framelinvel", {"name": "palm_linear_velocity", "objtype": "site", "objname": "fr3_palm_control_site"})
  ET.SubElement(sensor, "frameangvel", {"name": "palm_angular_velocity", "objtype": "site", "objname": "fr3_palm_control_site"})
  ET.SubElement(sensor, "force", {"name": "fr3_wrist_force", "site": "fr3_palm_control_site"})
  ET.SubElement(sensor, "torque", {"name": "fr3_wrist_torque", "site": "fr3_palm_control_site"})
  for index, joint_name in enumerate(ARM_JOINT_NAMES, start=1):
    ET.SubElement(sensor, "jointpos", {"name": f"arm_joint_{index}_position", "joint": joint_name})
    ET.SubElement(sensor, "jointvel", {"name": f"arm_joint_{index}_velocity", "joint": joint_name})
    ET.SubElement(sensor, "actuatorfrc", {"name": f"arm_joint_{index}_actuator_force", "actuator": f"fr3_arm_act_{index}"})


def _add_scene_visuals(root: ET.Element) -> None:
  visual = ET.Element("visual")
  ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
  ET.SubElement(visual, "quality", {"shadowsize": "4096"})
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise RuntimeError("FR3 model is missing worldbody")
  insertion = list(root).index(worldbody)
  root.insert(insertion, visual)
  ET.SubElement(worldbody, "light", {"name": "key", "pos": "0.5 -0.8 1.6", "dir": "0 0.3 -1", "diffuse": "0.9 0.9 0.9"})
  ET.SubElement(worldbody, "light", {"name": "fill", "pos": "-0.3 0.8 1.2", "dir": "0 -0.4 -1", "diffuse": "0.45 0.45 0.5"})
  ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "2 2 0.05", "pos": "0 0 -0.02", "rgba": "0.88 0.90 0.92 1", "contype": "0", "conaffinity": "0"})


def _assemble_xml(config: FullRobotModelConfig) -> tuple[str, NDArray[np.float64]]:
  arm_root = ET.parse(ARM_XML).getroot()
  hand_root = ET.parse(HAND_XML).getroot()
  _absolute_mesh_files(arm_root, ARM_XML)
  _absolute_mesh_files(hand_root, HAND_XML)
  _merge_assets(arm_root, hand_root)

  compiler = arm_root.find("compiler")
  if compiler is None:
    raise RuntimeError("FR3 source is missing compiler")
  compiler.set("eulerseq", "xyz")
  option = ET.Element(
    "option",
    {
      "timestep": str(config.timestep_s),
      "gravity": f"0 0 {config.gravity_m_s2}",
      "integrator": "implicitfast",
      "iterations": "80",
    },
  )
  arm_root.insert(list(arm_root).index(arm_root.find("asset")), option)
  _add_leap_default(arm_root, config)

  for geom in arm_root.findall(".//geom"):
    geom.set("contype", "4")
    geom.set("conaffinity", "4")
    geom.set("group", "1")

  link8 = arm_root.find(".//body[@name='fr3v2_link8']")
  if link8 is None:
    raise RuntimeError("FR3 source is missing fr3v2_link8")
  ET.SubElement(
    link8,
    "geom",
    {
      "name": "fr3_leap_mount_adapter",
      "type": "cylinder",
      "size": _vector_text(
        (
          MOUNT_ADAPTER_RADIUS_M,
          float(MOUNT_INTERFACE_POSITION_LINK8_M[2] / 2.0),
        )
      ),
      "pos": _vector_text(
        (0.0, 0.0, float(MOUNT_INTERFACE_POSITION_LINK8_M[2] / 2.0))
      ),
      "density": "0",
      "contype": "0",
      "conaffinity": "0",
      "group": "0",
      "rgba": "0.22 0.24 0.28 1",
    },
  )
  ET.SubElement(
    link8,
    "site",
    {"name": "fr3_wrist_site", "size": "0.006", "rgba": "0.95 0.2 0.2 0.8"},
  )
  ET.SubElement(
    link8,
    "site",
    {
      "name": "fr3_mount_interface_site",
      "pos": _vector_text(MOUNT_INTERFACE_POSITION_LINK8_M),
      "size": "0.0045",
      "rgba": "0.98 0.70 0.10 0.9",
      "group": "3",
    },
  )
  link8.append(_prepare_palm(hand_root))

  palm_home = _home_palm_position_without_surface(arm_root)
  object_position = _add_fixed_surface(arm_root, config, palm_home)
  _add_actuators_and_sensors(arm_root, hand_root, config)
  _add_scene_visuals(arm_root)

  keyframe = arm_root.find("keyframe")
  if keyframe is None:
    keyframe = ET.SubElement(arm_root, "keyframe")
  keyframe.clear()
  ET.SubElement(
    keyframe,
    "key",
    {"name": "fr3_leap_home", "qpos": _vector_text(MODEL_HOME_QPOS)},
  )
  return ET.tostring(arm_root, encoding="unicode"), object_position


def _ids(model: mujoco.MjModel, obj: mujoco.mjtObj, names: tuple[str, ...]) -> NDArray[np.int32]:
  result = np.array(
    [mujoco.mj_name2id(model, obj, name) for name in names],
    dtype=np.int32,
  )
  if np.any(result < 0):
    missing = [name for name, identifier in zip(names, result) if identifier < 0]
    raise RuntimeError(f"compiled model is missing names: {missing}")
  return result


def build_full_robot(config: FullRobotModelConfig | None = None) -> FullRobotHandles:
  """Build and audit one full robot model in memory."""

  resolved = config or FullRobotModelConfig()
  xml, object_position = _assemble_xml(resolved)
  model = mujoco.MjModel.from_xml_string(xml)
  if (model.nq, model.nv, model.nu) != (23, 23, 23):
    raise RuntimeError(
      f"FR3+Leap plant must be 23/23/23, got {model.nq}/{model.nv}/{model.nu}"
    )

  arm_joint_ids = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINT_NAMES)
  hand_joint_ids = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, HAND_JOINT_NAMES)
  arm_actuator_ids = _ids(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    tuple(f"fr3_arm_act_{index}" for index in range(1, 8)),
  )
  hand_actuator_ids = _ids(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    tuple(f"act_{index}" for index in range(16)),
  )
  tip_body_ids = _ids(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    tuple(finger.tip_body for finger in FINGERS),
  )
  tip_geom_ids = _ids(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    tuple(finger.proxy_geom.replace("e05_", "fr3_") for finger in FINGERS),
  )
  tip_site_ids = _ids(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    tuple(finger.pad_site.replace("e05_", "fr3_") for finger in FINGERS),
  )
  finger_qpos: list[NDArray[np.int32]] = []
  finger_dof: list[NDArray[np.int32]] = []
  for finger in FINGERS:
    joint_ids = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, finger.joint_names)
    finger_qpos.append(np.array(model.jnt_qposadr[joint_ids], dtype=np.int32))
    finger_dof.append(np.array(model.jnt_dofadr[joint_ids], dtype=np.int32))

  robot_geom_names = tuple(
    name
    for name in (
      *(f"fr3v2_link{index}_collision" for index in range(8)),
      *(finger.proxy_geom.replace("e05_", "fr3_") for finger in FINGERS),
    )
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
  )
  robot_geom_ids = _ids(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_names)
  return FullRobotHandles(
    model=model,
    xml=xml,
    config=resolved,
    arm_joint_ids=arm_joint_ids,
    hand_joint_ids=hand_joint_ids,
    arm_qpos_adrs=np.array(model.jnt_qposadr[arm_joint_ids], dtype=np.int32),
    hand_qpos_adrs=np.array(model.jnt_qposadr[hand_joint_ids], dtype=np.int32),
    arm_dof_adrs=np.array(model.jnt_dofadr[arm_joint_ids], dtype=np.int32),
    hand_dof_adrs=np.array(model.jnt_dofadr[hand_joint_ids], dtype=np.int32),
    arm_actuator_ids=arm_actuator_ids,
    hand_actuator_ids=hand_actuator_ids,
    finger_qpos_adrs=tuple(finger_qpos),
    finger_dof_adrs=tuple(finger_dof),
    tip_body_ids=tip_body_ids,
    tip_geom_ids=tip_geom_ids,
    tip_site_ids=tip_site_ids,
    palm_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower"),
    palm_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr3_palm_control_site"),
    palm_mount_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr3_palm_mount_site"),
    wrist_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr3_wrist_site"),
    mount_interface_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr3_mount_interface_site"),
    object_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_e05_object"),
    object_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fr3_e05_object_geom"),
    object_mocap_id=int(
      model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_e05_object")
      ]
    ),
    robot_geom_ids=robot_geom_ids,
    arm_joint_ranges_rad=np.array(model.jnt_range[arm_joint_ids], dtype=np.float64, copy=True),
    hand_joint_ranges_rad=np.array(model.jnt_range[hand_joint_ids], dtype=np.float64, copy=True),
    object_position_m=np.array(object_position, dtype=np.float64, copy=True),
  )


def export_model_xml(path: str | Path, config: FullRobotModelConfig | None = None) -> Path:
  """Export the exact generated XML used by an experiment for provenance."""

  handles = build_full_robot(config)
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(handles.xml, encoding="utf-8")
  return output


def model_audit(handles: FullRobotHandles) -> dict[str, object]:
  """Return deterministic structural and initial-pose audit evidence."""

  data = mujoco.MjData(handles.model)
  data.qpos[handles.arm_qpos_adrs] = ARM_HOME_Q
  data.qpos[handles.hand_qpos_adrs] = HAND_NATURAL_Q
  data.ctrl[handles.arm_actuator_ids] = ARM_HOME_Q
  data.ctrl[handles.hand_actuator_ids] = HAND_NATURAL_Q
  mujoco.mj_forward(handles.model, data)
  pad_normals = np.array(
    [data.site_xmat[site].reshape(3, 3)[:, 2] for site in handles.tip_site_ids],
    dtype=np.float64,
  )
  link8_body_id = mujoco.mj_name2id(
    handles.model, mujoco.mjtObj.mjOBJ_BODY, "fr3v2_link8"
  )
  palm_geom_id = mujoco.mj_name2id(
    handles.model, mujoco.mjtObj.mjOBJ_GEOM, "palm_lower_collision"
  )
  adapter_geom_id = mujoco.mj_name2id(
    handles.model, mujoco.mjtObj.mjOBJ_GEOM, "fr3_leap_mount_adapter"
  )
  mount_witness = np.zeros(6, dtype=np.float64)
  adapter_palm_mesh_distance = float(
    mujoco.mj_geomDistance(
      handles.model,
      data,
      adapter_geom_id,
      palm_geom_id,
      1.0,
      mount_witness,
    )
  )
  mount_origin_gap = float(
    np.linalg.norm(data.xpos[handles.palm_body_id] - data.xpos[link8_body_id])
  )
  mount_parent_is_link8 = bool(
    handles.model.body_parentid[handles.palm_body_id] == link8_body_id
  )
  mount_center_error = float(
    np.linalg.norm(
      data.site_xpos[handles.palm_mount_site_id]
      - data.site_xpos[handles.mount_interface_site_id]
    )
  )
  mesh_id = int(handles.model.geom_dataid[palm_geom_id])
  vertex_start = int(handles.model.mesh_vertadr[mesh_id])
  vertex_count = int(handles.model.mesh_vertnum[mesh_id])
  vertices = np.array(
    handles.model.mesh_vert[vertex_start : vertex_start + vertex_count],
    dtype=np.float64,
    copy=True,
  )
  geom_rotation = _quaternion_to_matrix(handles.model.geom_quat[palm_geom_id])
  vertices = vertices @ geom_rotation.T + handles.model.geom_pos[palm_geom_id]
  lower = np.min(vertices[:, :2], axis=0)
  upper = np.max(vertices[:, :2], axis=0)
  center_fraction = (PALM_MOUNT_LOCAL_M[:2] - lower) / (upper - lower)
  mount_is_central = bool(np.all(np.abs(center_fraction - 0.5) <= 0.12))
  mount_geometrically_closed = bool(
    mount_parent_is_link8
    and adapter_geom_id >= 0
    and mount_center_error <= 1e-9
    and mount_is_central
    and abs(adapter_palm_mesh_distance) <= MOUNT_INTERFACE_TOLERANCE_M
  )
  return {
    "model": "FR3_LEAP_NATURAL_ROUGH_LOCAL_REVIEW",
    "nq": int(handles.model.nq),
    "nv": int(handles.model.nv),
    "nu": int(handles.model.nu),
    "arm_dof": len(handles.arm_joint_ids),
    "hand_dof": len(handles.hand_joint_ids),
    "fixed_object": bool(handles.object_mocap_id < 0),
    "object_mocap_id": handles.object_mocap_id,
    "pad_parent_body_names": [
      mujoco.mj_id2name(handles.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
      for body_id in handles.tip_body_ids
    ],
    "pad_normal_world_z": pad_normals[:, 2].tolist(),
    "all_pads_face_down": bool(np.all(pad_normals[:, 2] < -0.50)),
    "natural_hand_q_rad": HAND_NATURAL_Q.tolist(),
    "natural_pose_source": "published raw-DP video q_hand at t=2.000 s",
    "mount_parent_is_link8": mount_parent_is_link8,
    "mount_adapter_present": bool(adapter_geom_id >= 0),
    "mount_origin_gap_m": mount_origin_gap,
    "mount_center_alignment_error_m": mount_center_error,
    "mount_center_xy_fraction": center_fraction.tolist(),
    "mount_is_central": mount_is_central,
    "adapter_palm_mesh_distance_m": adapter_palm_mesh_distance,
    "mount_interface_tolerance_m": MOUNT_INTERFACE_TOLERANCE_M,
    "mount_geometrically_closed": mount_geometrically_closed,
    "mount_witness_world_m": mount_witness.tolist(),
    "palm_position_m": data.site_xpos[handles.palm_site_id].tolist(),
    "object_position_m": handles.object_position_m.tolist(),
    "gravity_m_s2": handles.config.gravity_m_s2,
  }
