"""Compile the robot XML and collision meshes used by the hand experiments."""

from pathlib import Path

import mujoco
import pytest

from mjlab import MJLAB_SRC_PATH


@pytest.mark.parametrize("model", ["leaphand_only.xml", "fr3_leap_hand/fr3v2_collision.xml"])
def test_robot_compiles_parametrized(model: str) -> None:
  path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / model
  if path.is_relative_to(Path.cwd()):
    path = path.relative_to(Path.cwd())
  robot = mujoco.MjModel.from_xml_path(str(path))
  assert robot.nq > 0
  assert robot.nmesh > 0
