from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

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
SCREENSHOT_NAMES = ("start", "mid", "end")


@dataclass(frozen=True)
class DemoConfig:
    mode: str = "grasp_maintain"
    duration_s: float = 4.0
    sim_dt: float = 0.002
    control_decimation: int = 10
    video_fps: int = 20
    width: int = 960
    height: int = 720
    seed: int = 7
    save_h5: bool = True
    save_npz: bool = True
    skip_inversion: bool = False
    output_tag: str = "random_inhand"


class FingerComplianceController:
    def __init__(self, q_nom: np.ndarray):
        self.q_nom = q_nom.astype(np.float64).copy()
        self.prev_fsr = np.zeros(16, dtype=np.float64)
        self.s_min = 0.6
        self.s_max = 1.5
        self.k_prox = 0.15
        self.k_mid = 0.08
        self.k_dist = 0.04
        self.d_force = 0.03
        self.s_palm_threshold = 0.2
        self.reset_speed = 0.1

    def _interval_error(self, s: np.ndarray) -> np.ndarray:
        err = np.zeros_like(s)
        low = s < self.s_min
        high = s > self.s_max
        err[low] = self.s_min - s[low]
        err[high] = self.s_max - s[high]
        return err

    def __call__(self, q_curr: np.ndarray, fsr: np.ndarray) -> np.ndarray:
        dot_fsr = fsr - self.prev_fsr
        self.prev_fsr = fsr.copy()

        delta = np.zeros_like(q_curr)
        palm_force = float(np.mean(fsr[:4]))
        unlocked = palm_force > self.s_palm_threshold

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
                    self.k_prox * e_p - self.d_force * ds_p,
                    self.k_mid * adj_e_d,
                    self.k_dist * adj_e_d,
                ],
                dtype=np.float64,
            )

            for joint_idx, comp in zip(cfg["j"], comps, strict=False):
                reset_delta = self.reset_speed * (self.q_nom[joint_idx] - q_curr[joint_idx])
                delta[joint_idx] = comp if unlocked else reset_delta

        for joint_idx in UNUSED_HAND_JOINTS:
            delta[joint_idx] = self.reset_speed * (self.q_nom[joint_idx] - q_curr[joint_idx])

        return delta


def matrix_to_transform(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    pos = transform[:3, 3]
    transform_inv = np.eye(4, dtype=np.float64)
    transform_inv[:3, :3] = rot.T
    transform_inv[:3, 3] = -rot.T @ pos
    return transform_inv


def invert_transform_trajectory(transforms: np.ndarray) -> np.ndarray:
    inverted = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], transforms.shape[0], axis=0)
    rot = transforms[:, :3, :3]
    pos = transforms[:, :3, 3]
    rot_t = np.swapaxes(rot, 1, 2)
    inverted[:, :3, :3] = rot_t
    inverted[:, :3, 3] = -np.einsum("nij,nj->ni", rot_t, pos)
    return inverted


def summarize_inversion(forward: np.ndarray, inverse_: np.ndarray) -> dict[str, float | int | bool]:
    identity = np.matmul(forward, inverse_)
    translation_error = np.linalg.norm(identity[:, :3, 3], axis=1)
    rotation_error = np.linalg.norm(identity[:, :3, :3] - np.eye(3), axis=(1, 2))
    return {
        "reverse_time": False,
        "num_steps": int(forward.shape[0]),
        "mean_translation_error_m": float(np.mean(translation_error)),
        "max_translation_error_m": float(np.max(translation_error)),
        "mean_rotation_error_fro": float(np.mean(rotation_error)),
        "max_rotation_error_fro": float(np.max(rotation_error)),
    }


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_h5(path: Path, datasets: dict[str, np.ndarray], attrs: dict[str, object]) -> None:
    ensure_parent(path)
    with h5py.File(path, "w") as h5:
        for name, value in datasets.items():
            h5.create_dataset(name, data=value)
        for key, value in attrs.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                h5.attrs[key] = value
            else:
                h5.attrs[key] = json.dumps(value, ensure_ascii=False)


def build_run_paths(run_stem: str) -> dict[str, Path]:
    dataset_root = REPO_ROOT / "artifacts" / "datasets"
    video_root = REPO_ROOT / "artifacts" / "videos"
    screenshot_root = REPO_ROOT / "screenshots"
    log_root = REPO_ROOT / "logs"
    for root in (dataset_root, video_root, screenshot_root, log_root):
        root.mkdir(parents=True, exist_ok=True)
    return {
        "forward_npz": dataset_root / f"{run_stem}_trajectory_forward.npz",
        "forward_h5": dataset_root / f"{run_stem}_trajectory_forward.h5",
        "forward_json": dataset_root / f"{run_stem}_trajectory_forward.json",
        "inversion_npz": dataset_root / f"{run_stem}_trajectory_inversion.npz",
        "inversion_h5": dataset_root / f"{run_stem}_trajectory_inversion.h5",
        "inversion_json": dataset_root / f"{run_stem}_trajectory_inversion.json",
        "video": video_root / f"{run_stem}_demo.mp4",
        "screenshot_start": screenshot_root / f"{run_stem}_start.png",
        "screenshot_mid": screenshot_root / f"{run_stem}_mid.png",
        "screenshot_end": screenshot_root / f"{run_stem}_end.png",
        "summary": log_root / f"{run_stem}_summary.json",
        "latest_status": log_root / "latest_status.json",
    }


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-only compliance demo and data collection")
    parser.add_argument("--mode", choices=("grasp_maintain",), default=DemoConfig.mode)
    parser.add_argument("--duration-s", type=float, default=DemoConfig.duration_s)
    parser.add_argument("--sim-dt", type=float, default=DemoConfig.sim_dt)
    parser.add_argument("--control-decimation", type=int, default=DemoConfig.control_decimation)
    parser.add_argument("--video-fps", type=int, default=DemoConfig.video_fps)
    parser.add_argument("--width", type=int, default=DemoConfig.width)
    parser.add_argument("--height", type=int, default=DemoConfig.height)
    parser.add_argument("--seed", type=int, default=DemoConfig.seed)
    parser.add_argument("--output-tag", type=str, default=DemoConfig.output_tag)
    parser.add_argument("--no-h5", action="store_true")
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--skip-inversion", action="store_true")
    args = parser.parse_args()

    cfg = DemoConfig(
        mode=args.mode,
        duration_s=args.duration_s,
        sim_dt=args.sim_dt,
        control_decimation=args.control_decimation,
        video_fps=args.video_fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        save_h5=not args.no_h5,
        save_npz=not args.no_npz,
        skip_inversion=args.skip_inversion,
        output_tag=args.output_tag,
    )

    rng = np.random.default_rng(cfg.seed)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_stem = f"{timestamp}_{cfg.output_tag}_{cfg.mode}"
    run_paths = build_run_paths(run_stem)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.timestep = cfg.sim_dt
    data = mujoco.MjData(model)

    gravity = model.opt.gravity.copy()
    if not np.allclose(gravity, 0.0):
        raise RuntimeError(f"Expected gravity to be disabled, got {gravity.tolist()}")

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
    object_size = model.geom_size[object_geom_id].copy()
    object_mass = float(model.body_mass[object_body_id])

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

    controller = FingerComplianceController(q_nom=q_nom)

    num_steps = int(cfg.duration_s / cfg.sim_dt)
    render_interval = max(1, int(round(1.0 / (cfg.video_fps * cfg.sim_dt))))
    screenshot_steps = {
        "start": 0,
        "mid": num_steps // 2,
        "end": max(0, num_steps - 1),
    }

    times: list[float] = []
    qpos_log: list[np.ndarray] = []
    qvel_log: list[np.ndarray] = []
    ctrl_log: list[np.ndarray] = []
    fsr_log: list[np.ndarray] = []
    object_torque_log: list[np.ndarray] = []
    object_angvel_log: list[np.ndarray] = []
    t_ho_log: list[np.ndarray] = []
    t_oh_log: list[np.ndarray] = []
    t_wh_log: list[np.ndarray] = []
    t_wo_log: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    screenshot_frames: dict[str, np.ndarray] = {}

    for step in range(num_steps):
        q_curr = hand_joint_values(data, qpos_adrs)
        fsr = tactile_forces(model, data, tactile_geom_slots, object_geom_id)

        if step % cfg.control_decimation == 0:
            delta = controller(q_curr=q_curr, fsr=fsr)
            data.ctrl[:] = np.clip(
                q_nom + delta,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )

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
        object_angvel = data.qvel[object_dof_adr : object_dof_adr + 3].copy()

        t_wh = matrix_to_transform(hand_pos, hand_rot)
        t_wo = matrix_to_transform(object_pos, object_rot)
        t_ho = invert_transform(t_wh) @ t_wo
        t_oh = invert_transform(t_ho)

        times.append(float(data.time))
        qpos_log.append(hand_joint_values(data, qpos_adrs))
        qvel_log.append(hand_joint_velocities(data, dof_adrs))
        ctrl_log.append(data.ctrl.copy())
        fsr_log.append(fsr)
        object_torque_log.append(applied_torque)
        object_angvel_log.append(object_angvel)
        t_ho_log.append(t_ho)
        t_oh_log.append(t_oh)
        t_wh_log.append(t_wh)
        t_wo_log.append(t_wo)

        if step % render_interval == 0 or step in screenshot_steps.values():
            frame = render_frame(renderer, data, camera)
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
    angvel_arr = np.asarray(object_angvel_log, dtype=np.float64)
    t_ho_arr = np.asarray(t_ho_log, dtype=np.float64)
    t_oh_arr = np.asarray(t_oh_log, dtype=np.float64)
    t_wh_arr = np.asarray(t_wh_log, dtype=np.float64)
    t_wo_arr = np.asarray(t_wo_log, dtype=np.float64)

    forward_datasets = {
        "time": times_arr,
        "qpos": qpos_arr,
        "qvel": qvel_arr,
        "ctrl": ctrl_arr,
        "fsr_forces": fsr_arr,
        "object_torque": torque_arr,
        "object_angular_velocity": angvel_arr,
        "T_WH": t_wh_arr,
        "T_WO": t_wo_arr,
        "T_HO": t_ho_arr,
        "T_OH": t_oh_arr,
    }
    common_attrs = {
        "mode": cfg.mode,
        "xml_path": str(XML_PATH),
        "python_path": sys.executable,
        "gravity": gravity.tolist(),
        "control_decimation": cfg.control_decimation,
        "sim_dt": cfg.sim_dt,
        "output_tag": cfg.output_tag,
        "seed": cfg.seed,
        "object_size": object_size.tolist(),
        "object_mass": object_mass,
        "finger_compliance_controller": "active",
    }

    if cfg.save_npz:
        np.savez_compressed(run_paths["forward_npz"], **forward_datasets)
    if cfg.save_h5:
        write_h5(run_paths["forward_h5"], forward_datasets, common_attrs)

    for name in SCREENSHOT_NAMES:
        imageio.imwrite(run_paths[f"screenshot_{name}"], screenshot_frames[name])
    imageio.mimwrite(run_paths["video"], frames, fps=cfg.video_fps, quality=8)

    palm_force = fsr_arr[:, :4]
    forward_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "xml_path": str(XML_PATH),
        "python_path": sys.executable,
        "mode": cfg.mode,
        "gravity": gravity.tolist(),
        "config": asdict(cfg),
        "num_steps": int(num_steps),
        "num_video_frames": int(len(frames)),
        "mean_palm_force": float(np.mean(palm_force)),
        "max_palm_force": float(np.max(palm_force)),
        "mean_object_torque_norm": float(np.mean(np.linalg.norm(torque_arr, axis=1))),
        "max_object_torque_norm": float(np.max(np.linalg.norm(torque_arr, axis=1))),
        "mean_object_angvel_norm": float(np.mean(np.linalg.norm(angvel_arr, axis=1))),
        "max_object_angvel_norm": float(np.max(np.linalg.norm(angvel_arr, axis=1))),
        "artifacts": {
            "video": relative_to_repo(run_paths["video"]),
            "screenshots": [relative_to_repo(run_paths[f"screenshot_{name}"]) for name in SCREENSHOT_NAMES],
            "forward_npz": relative_to_repo(run_paths["forward_npz"]) if cfg.save_npz else None,
            "forward_h5": relative_to_repo(run_paths["forward_h5"]) if cfg.save_h5 else None,
            "forward_json": relative_to_repo(run_paths["forward_json"]),
        },
        "notes": [
            "Hand-only MuJoCoLab setup only; no arm or legacy full-hand MCC path is used.",
            "Palm-up configuration comes from src/mjlab/asset_zoo/robots/leaphand_only.xml.",
            "Gravity is disabled in the XML and revalidated at runtime before collection.",
            f"The object is a relatively large box inside the palm with half-sizes {object_size.tolist()} m and mass {object_mass:.3f} kg.",
            "Finger compliance controller remains active during random in-hand object rotation.",
            "Forward trajectory records T_HO explicitly; inversion writes T_OH and numerical consistency checks.",
        ],
    }
    write_json(run_paths["forward_json"], forward_summary)

    inversion_summary: dict[str, object] | None = None
    if not cfg.skip_inversion:
        inverted_t_oh = invert_transform_trajectory(t_ho_arr)
        inversion_summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_forward_json": relative_to_repo(run_paths["forward_json"]),
            **summarize_inversion(t_ho_arr, inverted_t_oh),
        }
        inversion_datasets = {
            "time": times_arr,
            "T_HO_source": t_ho_arr,
            "T_OH_inverted": inverted_t_oh,
        }
        if cfg.save_npz:
            np.savez_compressed(run_paths["inversion_npz"], **inversion_datasets)
        if cfg.save_h5:
            write_h5(run_paths["inversion_h5"], inversion_datasets, common_attrs)
        write_json(run_paths["inversion_json"], inversion_summary)

    final_summary = {
        "run_stem": run_stem,
        "forward_summary": forward_summary,
        "inversion_summary": inversion_summary,
        "artifacts": {
            "summary_json": relative_to_repo(run_paths["summary"]),
            "latest_status_json": relative_to_repo(run_paths["latest_status"]),
            "inversion_npz": relative_to_repo(run_paths["inversion_npz"]) if (inversion_summary and cfg.save_npz) else None,
            "inversion_h5": relative_to_repo(run_paths["inversion_h5"]) if (inversion_summary and cfg.save_h5) else None,
            "inversion_json": relative_to_repo(run_paths["inversion_json"]) if inversion_summary else None,
        },
    }
    write_json(run_paths["summary"], final_summary)
    write_json(run_paths["latest_status"], final_summary)

    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
