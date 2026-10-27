"""Shared, versioned state and logging contracts."""

from Module.common.contracts import (
  SCHEMA_VERSION,
  ContactState,
  ExecutorState,
  JsonlEpisodeLogger,
  StateSnapshot,
  load_jsonl_episode,
)

__all__ = [
  "SCHEMA_VERSION",
  "ContactState",
  "ExecutorState",
  "JsonlEpisodeLogger",
  "StateSnapshot",
  "load_jsonl_episode",
  "FULL_ROBOT_SCHEMA_VERSION",
  "FullRobotJsonlLogger",
  "FullRobotStateSnapshot",
  "load_full_robot_jsonl",
]


_FULL_ROBOT_EXPORTS = {
  "FULL_ROBOT_SCHEMA_VERSION",
  "FullRobotJsonlLogger",
  "FullRobotStateSnapshot",
  "load_full_robot_jsonl",
}


def __getattr__(name: str):
  """Load the MuJoCo-backed full-robot contract only when requested.

  This keeps ``python -m Module.fr3_visual_demo`` able to select OSMesa before
  MuJoCo is imported, while preserving the public ``from Module.common import
  FullRobotStateSnapshot`` API.
  """

  if name in _FULL_ROBOT_EXPORTS:
    from Module.common import full_robot_contracts

    return getattr(full_robot_contracts, name)
  raise AttributeError(name)
