from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
XML_PATH = REPO_ROOT / "src/mjlab/asset_zoo/robots/leaphand_only.xml"
TACTILE_GEOM_GROUPS = {
  "palm_lower_collision": [0, 1, 2, 3],
  "pip_geom": [4],
  "dip_geom": [5],
  "fingertip_geom": [6],
  "pip_2_geom": [7],
  "dip_2_geom": [8],
  "fingertip_2_geom": [9],
  "pip_3_geom": [10],
  "dip_3_geom": [11],
  "fingertip_3_geom": [12],
  "thumb_pip_geom": [13],
  "thumb_dip_geom": [14],
  "thumb_fingertip_geom": [15],
}
HAND_JOINT_NAMES = [str(i) for i in range(16)]
ACTIVE_FINGERS = [
  {"name": "index", "j": [0, 2, 3], "p_fsr": [4, 5], "d_fsr": [6]},
  {"name": "middle", "j": [4, 6, 7], "p_fsr": [7, 8], "d_fsr": [9]},
  {"name": "ring", "j": [8, 10, 11], "p_fsr": [10, 11], "d_fsr": [12]},
  {"name": "thumb", "j": [12, 14, 15], "p_fsr": [13, 14], "d_fsr": [15]},
]
UNUSED_HAND_JOINTS = [1, 5, 9, 13]


@dataclass(frozen=True)
class DemoConfig:
  duration_s: float = 8.0
  sim_dt: float = 0.002
  control_decimation: int = 10
  video_fps: int = 25
  width: int = 1280
  height: int = 720
  seed: int = 7
  output_root: str = "data/hand_only_compliance"
  save_h5: bool = True
  save_npz: bool = True


class HandComplianceController:
  def __init__(self, q_nom: np.ndarray):
    self.q_nom = q_nom.astype(np.float64).copy()
    self.prev_fsr = np.zeros(16, dtype=np.float64)
    self.S_min = 0.6
    self.S_max = 1.5
    self.K_prox = 0.15
    self.K_mid = 0.08
    self.K_dist = 0.04
    self.D_force = 0.03
    self.S_palm_threshold = 0.2
    self.reset_speed = 0.1

  def _interval_error(self, s: np.ndarray) -> np.ndarray:
    err = np.zeros_like(s)
    low = s < self.S_min
    high = s > self.S_max
    err[low] = self.S_min - s[low]
    err[high] = self.S_max - s[high]
    return err

  def __call__(self, q_curr: np.ndarray, fsr: np.ndarray) -> np.ndarray:
    dot_fsr = fsr - self.prev_fsr
    self.prev_fsr = fsr.copy()

    delta = np.zeros_like(q_curr)
    palm_force = float(np.mean(fsr[:4]))
    unlocked = palm_force > self.S_palm_threshold

    for cfg in ACTIVE_FINGERS:
      s_p = float(np.mean(fsr[cfg["p_fsr"]]))
      s_d = float(np.mean(fsr[cfg["d_fsr"]]))
      ds_p = float(np.mean(dot_fsr[cfg["p_fsr"]]))

      e_p = float(self._interval_error(np.array([s_p]))[0])
      e_d = float(self._interval_error(np.array([s_d]))[0])
      wrapping_factor = max(s_d - s_p, 0.0)
      adj_e_d = e_d - 0.5 * wrapping_factor

      comps = np.array(
        [
          self.K_prox * e_p - self.D_force * ds_p,
          self.K_mid * adj_e_d,
          self.K_dist * adj_e_d,
        ],
        dtype=np.float64,
      )

      for joint_idx, comp in zip(cfg["j"], comps, strict=False):
        reset_delta = self.reset_speed * (self.q_nom[joint_idx] - q_curr[joint_idx])
        delta[joint_idx] = comp if unlocked else reset_delta

    for joint_idx in UNUSED_HAND_JOINTS:
      delta[joint_idx] = self.reset_speed * (self.q_nom[joint_idx] - q_curr[joint_idx])

    return delta


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
  quat = quat.astype(np.float64)
  quat = quat / np.linalg.norm(quat)
  w, x, y, z = quat
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def matrix_to_transform(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
  T = np.eye(4, dtype=np.float64)
  T[:3, :3] = rot
  T[:3, 3] = pos
  return T


def invert_transform(T: np.ndarray) -> np.ndarray:
  R = T[:3, :3]
  t = T[:3, 3]
  T_inv = np.eye(4, dtype=np.float64)
  T_inv[:3, :3] = R.T
  T_inv[:3, 3] = -R.T @ t
  return T_inv


def make_camera() -> mujoco.MjvCamera:
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance = 0.42
  cam.azimuth = 132.0
  cam.elevation = -28.0
  cam.lookat[:] = np.array([-0.02, -0.02, 0.07], dtype=np.float64)
  return cam


def render_frame(
  renderer: mujoco.Renderer,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  camera: mujoco.MjvCamera,
) -> np.ndarray:
  renderer.update_scene(data, camera=camera)
  return renderer.render().copy()


def tactile_forces(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  tactile_geom_slots: dict[int, list[int]],
  object_geom_id: int,
) -> np.ndarray:
  forces = np.zeros(16, dtype=np.float64)
  contact_force = np.zeros(6, dtype=np.float64)
  for i in range(data.ncon):
    contact = data.contact[i]
    geom1 = int(contact.geom1)
    geom2 = int(contact.geom2)
    if object_geom_id not in (geom1, geom2):
      continue
    hand_geom = geom2 if geom1 == object_geom_id else geom1
    slots = tactile_geom_slots.get(hand_geom)
    if slots is None:
      continue
    mujoco.mj_contactForce(model, data, i, contact_force)
    force_norm = float(np.linalg.norm(contact_force[:3]))
    for slot in slots:
      forces[slot] += force_norm
  return forces


def hand_joint_values(data: mujoco.MjData, qpos_adrs: np.ndarray) -> np.ndarray:
  return np.array([data.qpos[int(adr)] for adr in qpos_adrs], dtype=np.float64)


def hand_joint_velocities(data: mujoco.MjData, dof_adrs: np.ndarray) -> np.ndarray:
  return np.array([data.qvel[int(adr)] for adr in dof_adrs], dtype=np.float64)


def apply_random_object_rotation(
  data: mujoco.MjData,
  dof_adr: int,
  rng: np.random.Generator,
  step: int,
) -> np.ndarray:
  phase = 2.0 * math.pi * step / 300.0
  smooth_bias = np.array(
    [
      0.18 * math.sin(phase),
      0.14 * math.cos(0.7 * phase + 0.6),
      0.16 * math.sin(1.3 * phase + 1.2),
    ],
    dtype=np.float64,
  )
  noise = rng.normal(0.0, 0.08, size=3)
  torque = smooth_bias + noise
  data.qfrc_applied[dof_adr : dof_adr + 3] = torque
  return torque


def save_metadata(output_dir: Path, cfg: DemoConfig, summary: dict) -> None:
  meta = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "xml_path": str(XML_PATH),
    "repo_root": str(REPO_ROOT),
    "config": asdict(cfg),
    **summary,
  }
  (output_dir / "metadata.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Hand-only compliance demo and data collection")
  parser.add_argument("--duration-s", type=float, default=DemoConfig.duration_s)
  parser.add_argument("--sim-dt", type=float, default=DemoConfig.sim_dt)
  parser.add_argument("--control-decimation", type=int, default=DemoConfig.control_decimation)
  parser.add_argument("--video-fps", type=int, default=DemoConfig.video_fps)
  parser.add_argument("--width", type=int, default=DemoConfig.width)
  parser.add_argument("--height", type=int, default=DemoConfig.height)
  parser.add_argument("--seed", type=int, default=DemoConfig.seed)
  parser.add_argument("--output-root", type=str, default=DemoConfig.output_root)
  parser.add_argument("--no-h5", action="store_true")
  parser.add_argument("--no-npz", action="store_true")
  args = parser.parse_args()

  cfg = DemoConfig(
    duration_s=args.duration_s,
    sim_dt=args.sim_dt,
    control_decimation=args.control_decimation,
    video_fps=args.video_fps,
    width=args.width,
    height=args.height,
    seed=args.seed,
    output_root=args.output_root,
    save_h5=not args.no_h5,
    save_npz=not args.no_npz,
  )

  os.environ.setdefault("MUJOCO_GL", "egl")
  rng = np.random.default_rng(cfg.seed)

  output_dir = REPO_ROOT / cfg.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
  output_dir.mkdir(parents=True, exist_ok=True)

  model = mujoco.MjModel.from_xml_path(str(XML_PATH))
  model.opt.timestep = cfg.sim_dt
  data = mujoco.MjData(model)

  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  camera = make_camera()

  hand_joint_ids = np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINT_NAMES],
    dtype=np.int32,
  )
  qpos_adrs = np.array([model.jnt_qposadr[jid] for jid in hand_joint_ids], dtype=np.int32)
  dof_adrs = np.array([model.jnt_dofadr[jid] for jid in hand_joint_ids], dtype=np.int32)
  tactile_geom_slots = {
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name): slots
    for geom_name, slots in TACTILE_GEOM_GROUPS.items()
  }
  hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower")
  object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_body")
  object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
  object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_ball")
  object_dof_adr = int(model.jnt_dofadr[object_joint_id])

  q_nom = np.array(
    [
      0.05,
      0.00,
      0.95,
      0.80,
      0.08,
      0.00,
      0.98,
      0.82,
      0.12,
      0.00,
      1.02,
      0.86,
      0.85,
      0.20,
      0.78,
      0.55,
    ],
    dtype=np.float64,
  )

  for idx, adr in enumerate(qpos_adrs.tolist()):
    data.qpos[adr] = q_nom[idx]
  data.ctrl[:] = q_nom
  mujoco.mj_forward(model, data)

  controller = HandComplianceController(q_nom=q_nom)

  num_steps = int(cfg.duration_s / cfg.sim_dt)
  render_interval = max(1, int(round(1.0 / (cfg.video_fps * cfg.sim_dt))))

  times = []
  qpos_log = []
  qvel_log = []
  ctrl_log = []
  fsr_log = []
  object_torque_log = []
  T_HO_log = []
  T_OH_log = []
  hand_pose_w_log = []
  object_pose_w_log = []
  frames = []
  screenshot_frames = {}

  screenshot_steps = {
    "start": 0,
    "mid": num_steps // 2,
    "end": max(0, num_steps - 1),
  }

  for step in range(num_steps):
    q_curr = hand_joint_values(data, qpos_adrs)
    fsr = tactile_forces(model, data, tactile_geom_slots, object_geom_id)

    if step % cfg.control_decimation == 0:
      delta = controller(q_curr=q_curr, fsr=fsr)
      data.ctrl[:] = np.clip(q_nom + delta, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    applied_torque = apply_random_object_rotation(
      data=data,
      dof_adr=object_dof_adr,
      rng=rng,
      step=step,
    )

    mujoco.mj_step(model, data)

    hand_pos = data.xpos[hand_body_id].copy()
    hand_rot = data.xmat[hand_body_id].reshape(3, 3).copy()
    object_pos = data.xpos[object_body_id].copy()
    object_rot = data.xmat[object_body_id].reshape(3, 3).copy()

    T_WH = matrix_to_transform(hand_pos, hand_rot)
    T_WO = matrix_to_transform(object_pos, object_rot)
    T_HO = invert_transform(T_WH) @ T_WO
    T_OH = invert_transform(T_HO)

    times.append(float(data.time))
    qpos_log.append(hand_joint_values(data, qpos_adrs))
    qvel_log.append(hand_joint_velocities(data, dof_adrs))
    ctrl_log.append(data.ctrl.copy())
    fsr_log.append(fsr)
    object_torque_log.append(applied_torque)
    T_HO_log.append(T_HO)
    T_OH_log.append(T_OH)
    hand_pose_w_log.append(T_WH)
    object_pose_w_log.append(T_WO)

    if step % render_interval == 0 or step in screenshot_steps.values():
      frame = render_frame(renderer, model, data, camera)
      frames.append(frame)
      for name, shot_step in screenshot_steps.items():
        if step == shot_step and name not in screenshot_frames:
          screenshot_frames[name] = frame.copy()

  times_arr = np.asarray(times, dtype=np.float64)
  qpos_arr = np.asarray(qpos_log, dtype=np.float64)
  qvel_arr = np.asarray(qvel_log, dtype=np.float64)
  ctrl_arr = np.asarray(ctrl_log, dtype=np.float64)
  fsr_arr = np.asarray(fsr_log, dtype=np.float64)
  torque_arr = np.asarray(object_torque_log, dtype=np.float64)
  T_HO_arr = np.asarray(T_HO_log, dtype=np.float64)
  T_OH_arr = np.asarray(T_OH_log, dtype=np.float64)
  T_WH_arr = np.asarray(hand_pose_w_log, dtype=np.float64)
  T_WO_arr = np.asarray(object_pose_w_log, dtype=np.float64)

  reversed_idx = slice(None, None, -1)
  T_HO_reversed = T_HO_arr[reversed_idx].copy()
  T_OH_reversed = T_OH_arr[reversed_idx].copy()

  if cfg.save_npz:
    np.savez_compressed(
      output_dir / "trajectory_forward.npz",
      time=times_arr,
      qpos=qpos_arr,
      qvel=qvel_arr,
      ctrl=ctrl_arr,
      fsr_forces=fsr_arr,
      object_torque=torque_arr,
      T_WH=T_WH_arr,
      T_WO=T_WO_arr,
      T_HO=T_HO_arr,
      T_OH=T_OH_arr,
    )
    np.savez_compressed(
      output_dir / "trajectory_inverted.npz",
      time=times_arr[::-1].copy(),
      qpos=qpos_arr[::-1].copy(),
      qvel=qvel_arr[::-1].copy(),
      ctrl=ctrl_arr[::-1].copy(),
      fsr_forces=fsr_arr[::-1].copy(),
      object_torque=torque_arr[::-1].copy(),
      T_HO=T_HO_reversed,
      T_OH=T_OH_reversed,
    )

  if cfg.save_h5:
    with h5py.File(output_dir / "trajectory_forward.h5", "w") as h5:
      h5.create_dataset("time", data=times_arr)
      h5.create_dataset("qpos", data=qpos_arr)
      h5.create_dataset("qvel", data=qvel_arr)
      h5.create_dataset("ctrl", data=ctrl_arr)
      h5.create_dataset("fsr_forces", data=fsr_arr)
      h5.create_dataset("object_torque", data=torque_arr)
      h5.create_dataset("T_WH", data=T_WH_arr)
      h5.create_dataset("T_WO", data=T_WO_arr)
      h5.create_dataset("T_HO", data=T_HO_arr)
      h5.create_dataset("T_OH", data=T_OH_arr)

  for name, frame in screenshot_frames.items():
    imageio.imwrite(output_dir / f"screenshot_{name}.png", frame)

  imageio.mimwrite(output_dir / "demo.mp4", frames, fps=cfg.video_fps, quality=8)

  summary = {
    "num_steps": num_steps,
    "num_video_frames": len(frames),
    "mean_palm_force": float(np.mean(fsr_arr[:, :4])),
    "max_palm_force": float(np.max(fsr_arr[:, :4])),
    "mean_object_torque_norm": float(np.mean(np.linalg.norm(torque_arr, axis=1))),
    "artifacts": {
      "video": "demo.mp4",
      "screenshots": [
        "screenshot_start.png",
        "screenshot_mid.png",
        "screenshot_end.png",
      ],
      "forward_npz": "trajectory_forward.npz" if cfg.save_npz else None,
      "inverted_npz": "trajectory_inverted.npz" if cfg.save_npz else None,
      "forward_h5": "trajectory_forward.h5" if cfg.save_h5 else None,
    },
  }
  save_metadata(output_dir, cfg, summary)

  print(f"[DONE] Output directory: {output_dir}")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
