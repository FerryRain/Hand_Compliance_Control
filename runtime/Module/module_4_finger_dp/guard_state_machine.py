"""Compatibility names for the shared M03 force-safety executor.

The execution authority is controller-independent and lives in M03. These
aliases preserve the frozen M4-DP API without giving Finger DP its own safety
implementation.
"""

from Module.module_3_runtime_guards.force_safety_executor import (
  ForceSafetyConfig,
  ForceSafetyExecutor,
  ForceSafetyOutput,
  ForceSafetyState,
)


DPGuardConfig = ForceSafetyConfig
DPGuardOutput = ForceSafetyOutput
DPGuardState = ForceSafetyState


class DPRuntimeGuardExecutor(ForceSafetyExecutor):
  """Backward-compatible alias of M03's shared safety executor."""


__all__ = [
  "DPGuardConfig",
  "DPGuardOutput",
  "DPGuardState",
  "DPRuntimeGuardExecutor",
]
