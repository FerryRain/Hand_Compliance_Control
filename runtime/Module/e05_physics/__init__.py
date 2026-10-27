"""Shared rough-surface geometry and MuJoCo scene primitives.

Current FR3+LEAP MCC evaluation reuses :mod:`extreme_surface` and :mod:`scene`
from this package. This package does not define a standalone evaluator.
"""

from Module.e05_physics.extreme_surface import query_surface

__all__ = ["query_surface"]
