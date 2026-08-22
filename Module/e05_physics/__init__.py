"""MuJoCo physical validation for the E05 MCC baseline arm only.

The public runner objects are loaded lazily so ``visual_demo`` can select a
headless OpenGL backend before importing MuJoCo.
"""

from __future__ import annotations

from typing import Any


__all__ = ["PhysicsConfig", "PhysicsTrace", "run_scenario"]


def __getattr__(name: str) -> Any:
  if name in __all__:
    from Module.e05_physics.runner import PhysicsConfig, PhysicsTrace, run_scenario

    return {
      "PhysicsConfig": PhysicsConfig,
      "PhysicsTrace": PhysicsTrace,
      "run_scenario": run_scenario,
    }[name]
  raise AttributeError(name)
