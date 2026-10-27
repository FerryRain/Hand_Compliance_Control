"""Headless-safe launcher for the formal FR3 + Leap Hand video demo."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

# This module lives above module_4_whole_hand_mcc, so these variables are set
# before that package imports MuJoCo and chooses an OpenGL platform.
os.environ["MUJOCO_GL"] = "osmesa"
cache = Path(tempfile.gettempdir()) / "handcomp-fr3-mesa"
cache.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(cache)
os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")

from Module.module_4_whole_hand_mcc.visual_demo import main


if __name__ == "__main__":
  main()
