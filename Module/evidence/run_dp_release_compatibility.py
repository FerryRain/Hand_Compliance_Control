"""Run the historical dp-capsule-v1 policy as a raw E05-F compatibility trial.

This is deliberately not the formal M04-DP or E05-F-DP evaluator.  The
released policy was trained as a nominal absolute-q generator whose documented
deployment layer is FullHandMCC.  Here we remove that analytical finger MCC on
purpose and ask a narrower question: can the raw policy be inserted into the
current FR3 + Leap Hand E05-F plant without changing its initial grasp?

LeRobot 0.4.4 and the release checkpoint are external audit inputs.  See
``2026-08-23_DP_STRATEGY_AUDIT.md`` for setup and interpretation.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import types
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
_CACHE = Path(tempfile.gettempdir()) / "handcomp-dp-audit-mesa"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_CACHE)

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_whole_hand_mcc.robot_control import PalmPoseIK, PalmPoseIKConfig
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  _contact_state,
  _initialize_data,
  _planned_palm_pose,
  _quaternion_from_matrix,
  _surface_reference,
)


HISTORICAL_COMMIT = "2742f39918d20c594cc6a5d4b5df95fc86511a67"
EXPECTED_CHECKPOINT_SHA256 = (
  "89044a1045ae44e28bec129c71998d3b389f08e4349e8a18441ba10bdd073ef0"
)
EXPECTED_STATE_FIELDS = (
  ("q_hand", 16),
  ("fingertip_contact_pos_palm", 12),
  ("fingertip_contact_normal_palm", 12),
  ("fingertip_contact_mask", 4),
  ("palm_relative_twist_palm", 6),
  ("planner_palm_delta_pose_palm", 6),
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / filename
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = np.array([0.34, 0.10, 0.43])
  camera.distance = 1.25
  camera.azimuth = 133.0
  camera.elevation = -22.0
  return camera


def _trial_overlay(
  frame: np.ndarray,
  timestamp_s: float,
  contacts: np.ndarray,
  forces: np.ndarray,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  draw = ImageDraw.Draw(image)
  draw.rounded_rectangle((18, 14, image.width - 18, 112), 13, fill=(8, 22, 38, 225))
  draw.text(
    (34, 28),
    "E05-F-DP RAW COMPATIBILITY TRIAL",
    font=_font(23, bold=True),
    fill="white",
  )
  draw.text(
    (34, 67),
    "EVIDENCE ONLY · no Fingertip MCC · not a formal DP score",
    font=_font(17, bold=True),
    fill=(255, 189, 102),
  )
  draw.text(
    (image.width - 175, 30),
    f"t = {timestamp_s:3.1f} s",
    font=_font(20, bold=True),
    fill=(255, 222, 117),
  )
  panel_top = image.height - 132
  draw.rounded_rectangle((18, panel_top, image.width - 18, image.height - 16), 13, fill=(8, 22, 38, 225))
  members = [str(index + 1) for index, active in enumerate(contacts) if active]
  contact_text = "{" + ",".join(members) + "}" if members else "EMPTY"
  draw.text(
    (34, panel_top + 14),
    f"actual contact set = {contact_text}",
    font=_font(19, bold=True),
    fill=(151, 231, 209) if members else (255, 112, 125),
  )
  draw.text(
    (34, panel_top + 55),
    "tip forces [N]  " + "   ".join(f"F{i + 1} {force:5.2f}" for i, force in enumerate(forces)),
    font=_font(17),
    fill=(213, 226, 238),
  )
  if not members:
    draw.rounded_rectangle((image.width - 320, panel_top + 16, image.width - 36, panel_top + 58), 9, fill=(183, 42, 61, 235))
    draw.text(
      (image.width - 298, panel_top + 27),
      "ALL FINGERTIP CONTACTS LOST",
      font=_font(15, bold=True),
      fill="white",
    )
  return np.asarray(image.convert("RGB"))


def _render_trial_video(
  handles: Any,
  arm_q: np.ndarray,
  finger_q: np.ndarray,
  contacts: np.ndarray,
  forces: np.ndarray,
  output_path: Path,
  *,
  dt_s: float,
  fps: int = 12,
) -> Path:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  data = mujoco.MjData(handles.model)
  renderer = mujoco.Renderer(handles.model, width=960, height=540)
  writer = imageio.get_writer(
    output_path,
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  frame_times = np.arange(0.0, len(arm_q) * dt_s, 1.0 / fps)
  indices = np.unique(np.clip(np.round(frame_times / dt_s).astype(int), 0, len(arm_q) - 1))
  try:
    for index in indices:
      data.qpos[handles.arm_qpos_adrs] = arm_q[index]
      data.qpos[handles.hand_qpos_adrs] = finger_q[index]
      mujoco.mj_forward(handles.model, data)
      renderer.update_scene(data, camera=_camera())
      writer.append_data(
        _trial_overlay(
          renderer.render().copy(),
          float(index * dt_s),
          contacts[index],
          forces[index],
        )
      )
  finally:
    writer.close()
    renderer.close()
  return output_path


def _historical_scheduler() -> tuple[type[Any], type[Any]]:
  """Load the exact release scheduler without restoring the retired tree."""

  source = subprocess.run(
    [
      "git",
      "show",
      f"{HISTORICAL_COMMIT}:mcc_finger_compliance_control/scripts/dp_chunk_scheduler.py",
    ],
    check=True,
    capture_output=True,
    text=True,
  ).stdout
  name = "historical_dp_chunk_scheduler"
  module = types.ModuleType(name)
  sys.modules[name] = module
  exec(compile(source, f"{name}.py", "exec"), module.__dict__)
  return module.DPChunkScheduler, module.DPChunkSchedulerConfig


class ReleasedPolicy:
  """Strict loader for the release's 56-D / absolute-q diffusion policy."""

  def __init__(self, checkpoint_path: Path, *, seed: int = 42) -> None:
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
      raise ValueError(
        f"checkpoint SHA-256 {checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
      )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["state_fields"]) != EXPECTED_STATE_FIELDS:
      raise ValueError(f"unexpected state fields: {checkpoint['state_fields']!r}")
    if checkpoint["state_schema"] != "contact_geometry_planner":
      raise ValueError(f"unexpected state schema: {checkpoint['state_schema']!r}")
    if checkpoint["action_representation"] != "absolute_q":
      raise ValueError(
        f"unexpected action representation: {checkpoint['action_representation']!r}"
      )
    if (checkpoint["robot_state_dim"], checkpoint["environment_state_dim"]) != (
      44,
      12,
    ):
      raise ValueError("released state split is not 44 + 12")

    args = SimpleNamespace(**dict(checkpoint["config"]))
    args.inference_steps = 10
    config = DiffusionConfig(
      input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(44,)),
        "observation.environment_state": PolicyFeature(
          type=FeatureType.ENV,
          shape=(12,),
        ),
      },
      output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))
      },
      n_obs_steps=16,
      horizon=32,
      n_action_steps=1,
      device="cpu",
      down_dims=tuple(args.down_dims),
      kernel_size=args.kernel_size,
      n_groups=args.n_groups,
      diffusion_step_embed_dim=args.diffusion_step_embed_dim,
      noise_scheduler_type=args.noise_scheduler,
      num_train_timesteps=args.diffusion_steps,
      beta_schedule="squaredcos_cap_v2",
      prediction_type="epsilon",
      clip_sample=True,
      clip_sample_range=1.0,
      num_inference_steps=10,
    )
    self.policy = DiffusionPolicy(config)
    load_result = self.policy.load_state_dict(checkpoint["model"])
    if load_result.missing_keys or load_result.unexpected_keys:
      raise RuntimeError(
        f"state_dict mismatch: missing={load_result.missing_keys}, "
        f"unexpected={load_result.unexpected_keys}"
      )
    self.policy.eval()
    self.policy.diffusion.num_inference_steps = 10
    self.generator = torch.Generator(device="cpu").manual_seed(seed)
    normalization = checkpoint["normalization"]
    self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
    self.state_std = np.asarray(normalization["state_std"], dtype=np.float32)
    self.action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    self.action_std = np.asarray(normalization["action_std"], dtype=np.float32)
    self.parameter_count = int(sum(value.numel() for value in checkpoint["model"].values()))

  @torch.no_grad()
  def predict(self, history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    if history.shape != (16, 56):
      raise ValueError(f"history shape {history.shape} != (16, 56)")
    normalized = (history - self.state_mean) / self.state_std
    state = torch.as_tensor(normalized[:, :44]).unsqueeze(0)
    environment = torch.as_tensor(normalized[:, 44:]).unsqueeze(0)
    condition = self.policy.diffusion._prepare_global_conditioning(
      {
        "observation.state": state,
        "observation.environment_state": environment,
      }
    )
    prediction = self.policy.diffusion.conditional_sample(
      1,
      global_cond=condition,
      generator=self.generator,
    )[0].numpy()
    return prediction * self.action_std[None, :] + self.action_mean[None, :]


def run_trial(
  checkpoint_path: Path,
  *,
  video_path: Path | None = None,
) -> dict[str, Any]:
  """Execute the fixed 3 s compatibility trial and return diagnostic metrics."""

  runtime = ReleasedPolicy(checkpoint_path)
  scheduler_type, scheduler_config_type = _historical_scheduler()
  config = E05MCCConfig(
    mode="E05-F-MCC",
    duration_s=3.0,
    settling_time_s=1.0,
    pose_step_time_s=2.0,
    traversal_y_m=0.03,
    lateral_primary_amplitude_m=0.004,
    lateral_secondary_amplitude_m=0.002,
  )
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="extreme",
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
    )
  )
  handles.model.geom_friction[handles.object_geom_id, 0] = 0.9
  handles.model.geom_friction[handles.tip_geom_ids, 0] = 0.9
  data = _initialize_data(handles, config)
  initial_q = np.asarray(data.qpos[handles.hand_qpos_adrs]).copy()
  initial_pose = np.concatenate(
    (
      data.site_xpos[handles.palm_site_id].copy(),
      _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
    )
  )
  palm_ik = PalmPoseIK(
    handles,
    PalmPoseIKConfig(gain=0.32, damping=0.018, max_joint_step_rad=0.02),
  )
  scheduler = scheduler_type(
    runtime.action_std,
    scheduler_config_type(
      control_dt=0.01,
      waypoint_dt=0.05,
      replan_interval=20,
      history_points=6,
      max_drop=4,
      max_joint_step=0.02,
    ),
  )

  history: deque[np.ndarray] = deque(maxlen=16)
  held_points = np.zeros((4, 3))
  held_normals = np.zeros((4, 3))
  point_valid = np.zeros(4, dtype=np.bool_)
  normal_valid = np.zeros(4, dtype=np.bool_)
  measured_forces = np.zeros(4)
  measured_positions = np.zeros((4, 3))
  active = np.zeros(4, dtype=np.bool_)
  command = np.asarray(data.ctrl[handles.hand_actuator_ids]).copy()
  installed = False

  sample_count = int(round(config.duration_s / config.dt_s))
  contacts = np.zeros((sample_count, 4), dtype=np.bool_)
  force_log = np.zeros((sample_count, 4))
  palm_log = np.zeros((sample_count, 3))
  arm_q_log = np.zeros((sample_count, 7))
  finger_q_log = np.zeros((sample_count, 16))
  inference_latency: list[float] = []
  first_target: np.ndarray | None = None
  non_tip_contact_samples = 0
  bootstrap_step = 375  # 15 intervals x 0.05 s / 0.002 s

  for step in range(sample_count):
    timestamp_s = step * config.dt_s
    planned_pose = _planned_palm_pose(initial_pose, config, timestamp_s)
    data.ctrl[handles.arm_actuator_ids] = palm_ik.solve(data, planned_pose)
    sensor_normals_world = np.zeros((4, 3))
    for finger, site_id in enumerate(handles.tip_site_ids):
      _, outward, _ = _surface_reference(
        "extreme",
        handles.object_position_m,
        np.asarray(data.site_xpos[int(site_id)]),
      )
      # Historical ContactSensor convention was fingertip -> object.
      sensor_normals_world[finger] = -outward

    if step % 5 == 0:  # historical DP command loop: 100 Hz
      valid_measured_point = active & (
        np.linalg.norm(measured_positions, axis=1) > 1.0e-9
      )
      held_points[valid_measured_point] = measured_positions[valid_measured_point]
      point_valid[valid_measured_point] = True
      held_normals[active] = sensor_normals_world[active]
      normal_valid[active] = True
      points_world = np.zeros((4, 3))
      normals_world = np.zeros((4, 3))
      points_world[point_valid] = held_points[point_valid]
      normals_world[normal_valid] = held_normals[normal_valid]

      rotation = np.asarray(data.site_xmat[handles.palm_site_id]).reshape(3, 3)
      palm_position = np.asarray(data.site_xpos[handles.palm_site_id])
      points_palm = (points_world - palm_position) @ rotation
      normals_palm = normals_world @ rotation
      future_pose = _planned_palm_pose(
        initial_pose,
        config,
        min(timestamp_s + 0.2, config.duration_s - config.dt_s),
      )
      planner_translation = rotation.T @ (future_pose[:3] - planned_pose[:3])
      next_pose = _planned_palm_pose(
        initial_pose,
        config,
        min(timestamp_s + 0.01, config.duration_s - config.dt_s),
      )
      palm_linear_velocity = rotation.T @ (
        (next_pose[:3] - planned_pose[:3]) / 0.01
      )
      state = np.concatenate(
        (
          np.asarray(data.qpos[handles.hand_qpos_adrs]),
          points_palm.ravel(),
          normals_palm.ravel(),
          active.astype(np.float64),
          palm_linear_velocity,
          np.zeros(3),
          planner_translation,
          np.zeros(3),
        )
      ).astype(np.float32)
      if step % 25 == 0:  # 0.05 s observation waypoint
        history.append(state)
      scheduler.observe(np.asarray(data.qpos[handles.hand_qpos_adrs]))
      if (
        step >= bootstrap_step
        and (step - bootstrap_step) % 100 == 0
        and len(history) == 16
      ):
        tic = perf_counter()
        prediction = runtime.predict(np.stack(history))
        inference_latency.append(perf_counter() - tic)
        scheduler.install(prediction)
        installed = True
        if first_target is None:
          first_target = prediction[0].copy()
      if installed:
        command = scheduler.next_command().astype(np.float64)
      command = np.clip(
        command,
        handles.hand_joint_ranges_rad[:, 0] + 0.05,
        handles.hand_joint_ranges_rad[:, 1] - 0.05,
      )
      data.ctrl[handles.hand_actuator_ids] = command

    mujoco.mj_step(handles.model, data)
    measured_forces, _, measured_positions, non_tip = _contact_state(handles, data)
    non_tip_contact_samples += non_tip
    active = np.where(
      active,
      measured_forces >= 0.10,
      measured_forces >= 0.20,
    )
    contacts[step] = active
    force_log[step] = measured_forces
    palm_log[step] = data.site_xpos[handles.palm_site_id]
    arm_q_log[step] = data.qpos[handles.arm_qpos_adrs]
    finger_q_log[step] = data.qpos[handles.hand_qpos_adrs]

  evaluated = np.arange(sample_count) * config.dt_s >= config.settling_time_s
  contact_count = np.sum(contacts[evaluated], axis=1)
  any_contact = contact_count > 0
  rendered_video = None
  if video_path is not None:
    rendered_video = str(
      _render_trial_video(
        handles,
        arm_q_log,
        finger_q_log,
        contacts,
        force_log,
        video_path,
        dt_s=config.dt_s,
      )
    )
  return {
    "trial": "E05-F-DP-RAW-3S-COMPATIBILITY",
    "execution_status": "EVIDENCE_ONLY",
    "formal_e05": False,
    "postprocess_mcc": False,
    "video": rendered_video,
    "duration_s": config.duration_s,
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "parameter_count": runtime.parameter_count,
    "dp_calls": len(inference_latency),
    "dp_inference_mean_s": float(np.mean(inference_latency)),
    "dp_inference_p95_s": float(np.percentile(inference_latency, 95.0)),
    "first_target_max_delta_from_initial_rad": float(
      np.max(np.abs(first_target - initial_q))
    ),
    "contact_continuity_probability": float(np.mean(any_contact)),
    "average_contact_count": float(np.mean(contact_count)),
    "minimum_contact_count": int(np.min(contact_count)),
    "zero_contact_time_s": float(np.sum(~any_contact) * config.dt_s),
    "force_rmse_n": float(np.sqrt(np.mean((force_log[evaluated] - 2.0) ** 2))),
    "max_tip_force_n": float(np.max(force_log[evaluated])),
    "traversal_y_m": float(
      palm_log[evaluated][-1, 1] - palm_log[evaluated][0, 1]
    ),
    "final_contacts": contacts[-1].astype(int).tolist(),
    "non_tip_contact_count_accumulated": int(non_tip_contact_samples),
    "interpretation": (
      "Raw absolute-q compatibility trial only. The released deployment "
      "contract expects a DP nominal followed by FullHandMCC."
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--video", type=Path, default=None)
  args = parser.parse_args()
  result = run_trial(args.checkpoint, video_path=args.video)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  print(encoded)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
