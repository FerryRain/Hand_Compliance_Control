"""Collect trajectory data by running the policy registered on a task."""

from dataclasses import asdict, dataclass
from datetime import datetime
import os
import sys
from typing import Any, Literal

import h5py
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, types as env_types
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class CollectConfig:
  output_dir: str = "data/collected"
  filename: str | None = None
  device: str | None = None
  viewer: Literal["native", "viser"] = "native"
  collect: bool = True
  record_forces: bool = True
  fsr_dims: int = 20


class H5DataLogger:
  def __init__(self, filepath: str):
    self.file = h5py.File(filepath, "w")
    self.group = self.file.create_group("data")
    self.step_idx = 0

  def log(
    self,
    obs: env_types.VecEnvObs,
    action: torch.Tensor,
    reward: torch.Tensor,
    forces: torch.Tensor | None = None,
  ) -> None:
    step_grp = self.group.create_group(f"step_{self.step_idx}")
    policy_obs = _policy_obs_tensor(obs)
    step_grp.create_dataset("obs", data=policy_obs.detach().cpu().numpy())
    step_grp.create_dataset("action", data=action.detach().cpu().numpy())
    step_grp.create_dataset("reward", data=reward.detach().cpu().numpy())
    if forces is not None:
      step_grp.create_dataset("fsr_forces", data=forces.detach().cpu().numpy())
    self.step_idx += 1

  def close(self) -> None:
    self.file.close()


def _policy_obs_tensor(obs: env_types.VecEnvObs) -> torch.Tensor:
  policy_obs = obs["policy"]
  if isinstance(policy_obs, torch.Tensor):
    return policy_obs

  if "fsr_forces" in policy_obs:
    return policy_obs["fsr_forces"]

  if len(policy_obs) == 1:
    return next(iter(policy_obs.values()))

  keys = ", ".join(sorted(policy_obs.keys()))
  raise ValueError(
    "Cannot infer policy tensor from non-concatenated policy observations. "
    f"Available terms: {keys}."
  )


def _build_registered_policy(task_id: str, cfg: CollectConfig, num_envs: int) -> Any:
  policy_cfg = load_rl_cfg(task_id)
  policy_class = getattr(policy_cfg, "policy_class", None)
  if policy_class is None:
    raise ValueError(
      f"Task '{task_id}' has no 'policy_class' in rl_cfg. "
      "Register a control policy in the task's rl_cfg first."
    )

  cfg_dict = asdict(policy_cfg)
  cfg_dict.pop("policy_class", None)
  cfg_dict.pop("device", None)

  policy_device = cfg.device or getattr(policy_cfg, "device", None)
  if policy_device is None:
    policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"

  print(f"[INFO] Loading registered policy: {policy_class.__name__}")
  return policy_class(device=policy_device, num_envs=num_envs, **cfg_dict)


def _adapt_action_dim(action: torch.Tensor, target_dim: int) -> torch.Tensor:
  current_dim = int(action.shape[-1])
  if current_dim == target_dim:
    return action

  if target_dim == 0:
    return action.new_zeros((action.shape[0], 0))

  if current_dim > target_dim:
    return action[:, :target_dim]

  pad = action.new_zeros((action.shape[0], target_dim - current_dim))
  return torch.cat((action, pad), dim=-1)


def _log_joint_action_mapping(env: ManagerBasedRlEnv) -> None:
  """Log action index -> joint target mapping when available."""
  try:
    joint_action = env.action_manager.get_term("hand_delta")
  except Exception:
    print("[INFO] Action term 'hand_delta' not found; skipping action map log")
    return

  target_names = getattr(joint_action, "target_names", None)
  target_ids = getattr(joint_action, "target_ids", None)
  if target_names is None or target_ids is None:
    print("[INFO] hand_delta has no target metadata; skipping action map log")
    return

  if hasattr(target_ids, "tolist"):
    target_ids = target_ids.tolist()

  rows = [
    (i, str(name), int(jid))
    for i, (name, jid) in enumerate(zip(target_names, target_ids, strict=False))
  ]

  idx_w = max(len("idx"), max((len(str(i)) for i, _, _ in rows), default=1))
  name_w = max(
    len("joint_name"), max((len(name) for _, name, _ in rows), default=1)
  )
  jid_w = max(
    len("joint_id"), max((len(str(jid)) for _, _, jid in rows), default=1)
  )

  sep = f"+{'-' * (idx_w + 2)}+{'-' * (name_w + 2)}+{'-' * (jid_w + 2)}+"
  header = (
    f"| {'idx'.rjust(idx_w)} | {'joint_name'.ljust(name_w)} | "
    f"{'joint_id'.rjust(jid_w)} |"
  )

  print("[INFO] hand_delta action mapping")
  print(sep)
  print(header)
  print(sep)
  for i, name, jid in rows:
    print(f"| {str(i).rjust(idx_w)} | {name.ljust(name_w)} | {str(jid).rjust(jid_w)} |")
  print(sep)


def _log_observation_action_dims(
  env: ManagerBasedRlEnv,
  obs: env_types.VecEnvObs,
) -> int:
  """Log observation/action dimensions and return action dimension."""
  policy_obs = _policy_obs_tensor(obs)
  action_dim = env.action_manager.total_action_dim

  print(f"[INFO] Policy observation dim: {int(policy_obs.shape[-1])}")
  print("[INFO] Observation group dims:")
  for group_name, group_dim in env.observation_manager.group_obs_dim.items():
    print(f"[INFO]   {group_name}: {group_dim}")
    term_names = env.observation_manager.active_terms[group_name]
    term_dims = env.observation_manager.group_obs_term_dim[group_name]
    for term_name, term_dim in zip(term_names, term_dims, strict=False):
      print(f"[INFO]     {term_name}: {term_dim}")

  print(f"[INFO] Environment action dim: {action_dim}")
  print("[INFO] Action term dims:")
  for term_name, term_dim in zip(
    env.action_manager.active_terms,
    env.action_manager.action_term_dim,
    strict=False,
  ):
    print(f"[INFO]   {term_name}: {term_dim}")

  return action_dim


def run_collect(task_id: str, cfg: CollectConfig) -> None:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  _log_joint_action_mapping(env)

  logger: H5DataLogger | None = None
  save_path: str | None = None
  if cfg.collect:
    os.makedirs(cfg.output_dir, exist_ok=True)
    filename = cfg.filename or f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    save_path = os.path.join(cfg.output_dir, filename)
    logger = H5DataLogger(save_path)

  raw_policy = _build_registered_policy(task_id, cfg, env.num_envs)

  original_step = env.step
  current_obs, _ = env.reset()
  action_dim = _log_observation_action_dims(env, current_obs)

  def step_with_logging(action: torch.Tensor):
    nonlocal current_obs
    next_obs, reward, terminated, truncated, info = original_step(action)

    forces = None
    if cfg.record_forces and cfg.fsr_dims > 0:
      policy_obs = _policy_obs_tensor(next_obs)
      forces = policy_obs[:, : cfg.fsr_dims]

    if logger is not None:
      logger.log(current_obs, action, reward, forces)
    current_obs = next_obs
    return next_obs, reward, terminated, truncated, info

  env.step = step_with_logging
  viewer_env = RslRlVecEnvWrapper(
    env, clip_actions=getattr(agent_cfg, "clip_actions", None)
  )

  class PolicyWithActionAdapter:
    def __call__(self, obs):
      return _adapt_action_dim(raw_policy(obs), action_dim)

  policy = PolicyWithActionAdapter()

  if save_path is not None:
    print(f"[INFO] Collecting data to: {save_path}")
  else:
    print("[INFO] Running without data collection")
  try:
    if cfg.viewer == "native":
      NativeMujocoViewer(viewer_env, policy).run()
    else:
      ViserPlayViewer(viewer_env, policy).run()
  finally:
    if logger is not None:
      logger.close()
    viewer_env.close()
    saved_steps = logger.step_idx if logger is not None else 0
    print(f"[SUCCESS] Saved {saved_steps} steps")


def main() -> None:
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    CollectConfig,
    args=remaining_args,
    default=CollectConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_collect(chosen_task, args)


if __name__ == "__main__":
  main()


"""
PYTHONPATH=src python -m mjlab.scripts.collect_data Leaphand-Contact-Relocation --collect False --viewer native
"""