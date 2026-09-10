"""Replay a recovery episode in the fixed-object inverse frame, without the xArm.

The inverse frame is the one ``invert_trajectories.py`` produces: the object is
pinned at the origin and the palm carries all of the relative motion.  Because
``palm_pose_object`` is recorded, nothing has to be guessed -- unlike the
world-frame replay, which needs an object placement, here the object *is* the
origin and the palm pose comes straight from the file.

The hand is the repository's existing hand-only model: the ``palm_base`` free
joint is deleted and ``palm_lower`` is marked mocap, so its world pose is fully
kinematic (``mocap_pos``/``mocap_quat``) while the 16 finger joints are written
from the recording.  This mirrors ``_legacy/test_palm_free_orbit.py``'s
``_palm_mocap_spec`` and ``view_inverse_palm_plans.py``'s scene construction.

No physics and no controller run: each frame writes the palm pose and the finger
joints, calls ``mj_forward`` and draws.  The displayed motion is therefore the
H5 itself.

Two schemas are understood, and any number of files may be concatenated so a
whole episode can be replayed across its segments:

``inverted`` (``invert_trajectories.py`` output)
    Carries ``palm_pose_object``, ``fingertip_pose_object``,
    ``fingertip_contact_pos_object`` and ``fingertip_contact``.
``mcc_closed_loop_observation_v1`` (a DP rollout)
    Carries ``palm_pose_object`` and ``q_live``, but states its contact in the
    *palm* frame, so ``fingertip_contact_pos_palm`` is rotated into the object
    frame with the same ``palm_pose_object`` that drives the hand.  It has no
    ``fingertip_pose_object``, so those frames are skipped by the self-check.

This is what makes the ctx segment visible: a rollout holds the pre-takeover
frames (including the contact loss), an inverted file holds the recovery -- and
both state the palm pose in the object frame, so they join seamlessly.  The
finger joints do *not* join seamlessly (the recovery dataset is built across an
unrecorded settle gap), which the junction report prints.

Usage::

    PY=/home/rimlab/miniconda3/envs/mjlab/bin/python
    $PY recovery_standalone/scripts/visualize_inverse_trajectory.py \\
        --file recovery_standalone/data/e2e_legacy_liveq_20260910/\\
ep040_recovery_inverted.h5

    # whole episode: the 80 ctx frames that end at the failure, then recovery
    $PY recovery_standalone/scripts/visualize_inverse_trajectory.py \\
        --file <rollout>.h5:1050:1130 --file <recovery_inverted>.h5

A ``--file`` entry may carry a ``:START:STOP`` suffix to take a frame window
from that file; without it the file is used whole.

Green sphere = recorded contact point, red sphere = recorded contact loss,
small coloured sphere = fingertip site.  ``mj_forward`` fingertip positions are
compared against the file's own ``fingertip_pose_object`` as a self-check.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import _add_tip_sites

from object_catalog import add_object_body, load_object_config


ROOT = Path(__file__).resolve().parents[2]
HAND_XML = (
    ROOT / "src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml"
)

# qpos order of the hand joints in the hand-only XML (same convention as
# view_inverse_palm_plans.py).  ``q_hand`` rows are stored in this order.
HAND_JOINT_NAMES = (
    "1", "0", "2", "3",
    "5", "4", "6", "7",
    "9", "8", "10", "11",
    "12", "13", "14", "15",
)
TIP_SITE_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")

TIP_COLORS = (
    (1.0, 0.30, 0.20, 0.95),
    (0.20, 0.70, 1.0, 0.95),
    (1.0, 0.80, 0.15, 0.95),
    (0.80, 0.30, 1.0, 0.95),
)
CONTACT_COLOR = (0.10, 1.0, 0.15, 1.0)
LOST_COLOR = (1.0, 0.05, 0.05, 1.0)


ROLLOUT_FORMAT = "mcc_closed_loop_observation_v1"


def _rows(file: h5py.File, name: str, env_axis: bool) -> np.ndarray:
    """Return one dataset's ``(T, ...)`` rows.

    Inverted files keep the collector's single env axis and rollouts do not, so
    the schema (not the ndim) says whether to drop an axis -- guessing would
    silently mis-slice a ``(T, 4)`` mask.
    """

    if name not in file:
        raise KeyError(f"{file.filename}: missing dataset {name!r}")
    value = np.asarray(file[name], dtype=np.float64)
    if env_axis:
        if value.ndim < 2:
            raise ValueError(f"{name}: expected at least 2 dims, got {value.shape}")
        value = value[:, 0]
    if value.ndim < 1:
        raise ValueError(f"{name}: expected at least 1 dim, got {value.shape}")
    return value


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """``(T, 4)`` wxyz quaternions to ``(T, 3, 3)`` rotation matrices."""

    out = np.empty((len(quat), 9), dtype=np.float64)
    for index, value in enumerate(quat):
        mujoco.mju_quat2Mat(out[index], np.asarray(value, dtype=np.float64))
    return out.reshape(-1, 3, 3)


class _Segment(NamedTuple):
    """One file's contribution to the replay, already object-frame."""

    label: str
    object_id: str | None
    object_scale: float | None
    palm_pose: np.ndarray  # (T, 7)
    q_hand: np.ndarray  # (T, 16)
    tip_pose: np.ndarray | None  # (T, 4, 3) reference, None when unavailable
    contact_pos: np.ndarray  # (T, 4, 3)
    contact: np.ndarray  # (T, 4)


def _parse_file_spec(text: str) -> tuple[Path, int | None, int | None]:
    """Split ``PATH`` or ``PATH:START:STOP`` into its three parts."""

    parts = text.split(":")
    window: list[int] = []
    while len(parts) > 1:
        try:
            window.insert(0, int(parts[-1]))
        except ValueError:
            break
        parts.pop()
    path = Path(":".join(parts))
    if not window:
        return path, None, None
    if len(window) != 2:
        raise ValueError(
            f"--file {text!r}: a frame window needs both bounds, as PATH:START:STOP"
        )
    return path, window[0], window[1]


def _load_segment(path: Path, start: int | None, stop: int | None) -> _Segment:
    if not path.is_file():
        raise FileNotFoundError(f"--file not found: {path}")
    with h5py.File(path, "r") as file:
        if bool(file.attrs.get("inverted", False)):
            palm_pose = _rows(file, "palm_pose_object", True)
            q_hand = _rows(file, "q_hand", True)
            tip_pose = _rows(file, "fingertip_pose_object", True)[..., :3]
            contact_pos = _rows(file, "fingertip_contact_pos_object", True)
            contact = _rows(file, "fingertip_contact", True) > 0.5
        elif str(file.attrs.get("format", "")) == ROLLOUT_FORMAT:
            palm_pose = _rows(file, "palm_pose_object", False)
            q_hand = _rows(file, "q_live", False)
            tip_pose = None
            # A rollout states contact in the palm frame; the very same
            # ``palm_pose_object`` drives the hand, so it is also the transform
            # that carries these points into the object frame.
            contact_palm = _rows(file, "fingertip_contact_pos_palm", False)
            rot = _quat_to_mat(palm_pose[:, 3:7])
            contact_pos = (
                np.einsum("tij,tfj->tfi", rot, contact_palm)
                + palm_pose[:, None, :3]
            )
            contact = _rows(file, "fingertip_contact_mask", False) > 0.5
        else:
            raise ValueError(
                f"{path}: unrecognized schema -- expected attrs['inverted']=True "
                f"or attrs['format']={ROLLOUT_FORMAT!r}."
            )
        object_id = file.attrs.get("object_id")
        object_scale = file.attrs.get("object_scale")

    total = len(q_hand)
    if total == 0:
        raise ValueError(f"{path}: no frames")
    begin = 0 if start is None else start
    end = total if stop is None else stop
    if not 0 <= begin < end <= total:
        raise ValueError(
            f"{path}: window [{begin}, {end}) is empty or outside the file's "
            f"{total} frames"
        )
    window = slice(begin, end)
    label = path.name if (begin, end) == (0, total) else f"{path.name}[{begin}:{end}]"
    return _Segment(
        label=label,
        object_id=None if object_id is None else str(object_id),
        object_scale=None if object_scale is None else float(object_scale),
        palm_pose=palm_pose[window],
        q_hand=q_hand[window],
        tip_pose=None if tip_pose is None else tip_pose[window],
        contact_pos=contact_pos[window],
        contact=contact[window],
    )


def _resolve_object_scale(segments: list[_Segment], object_id: str) -> float:
    for segment in segments:
        if segment.object_scale is not None:
            return segment.object_scale
    config = load_object_config(object_id)
    scale_range = np.asarray(
        config.collection.get("size_scale_range", (1.0, 1.0)), dtype=np.float64
    )
    if scale_range.shape != (2,) or not np.isclose(scale_range[0], scale_range[1]):
        raise ValueError(
            f"{object_id}: collection uses a randomized size scale and the H5 "
            "records no object_scale, so the replay scale is ambiguous."
        )
    return float(scale_range[0])


def _hand_mocap_spec() -> mujoco.MjSpec:
    """Hand-only spec with ``palm_lower`` as a mocap-driven rigid body.

    Same construction as ``view_inverse_palm_plans.py`` (meshes absolutized
    before the spec is attached) plus ``test_palm_free_orbit.py``'s trick of
    deleting the free joint and marking the palm mocap, so its world pose is
    set kinematically instead of by dynamics.
    """

    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    # Attached specs lose their source directory, so mesh paths must be made
    # absolute before merging them into the inverse viewer scene.
    for mesh in hand.meshes:
        if mesh.file:
            mesh.file = str((HAND_XML.parent / mesh.file).resolve())
    for key in list(hand.keys):
        hand.delete(key)
    for joint in list(hand.joints):
        if joint.name == "palm_base":
            hand.delete(joint)
    palm = hand.body("palm_lower")
    if palm is None:
        raise ValueError(f"palm_lower is missing from {HAND_XML}")
    palm.mocap = True
    palm.pos[:] = (0.0, 0.0, 0.0)
    palm.quat[:] = (1.0, 0.0, 0.0, 0.0)
    # The bare hand XML carries no sites; the four MCC fingertip reference
    # sites are added by the environment.  Reuse that same routine (and its
    # constants) so ``fingertip_pose_object`` and these sites stay the same
    # frame -- the self-check below would otherwise compare different points.
    _add_tip_sites(hand, HAND_XML)
    # The scene is a geometric check, so the hand must not push the object.
    for geom in hand.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    return hand


def _scene_spec(object_id: str, object_scale: float) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.gravity[:] = (0.0, 0.0, 0.0)
    spec.worldbody.add_light(
        name="key_light",
        pos=(0.4, -0.6, 1.0),
        dir=(-0.3, 0.4, -1.0),
        diffuse=(0.9, 0.9, 0.9),
    )
    # The inverse frame puts the object at the origin by definition.
    add_object_body(
        spec,
        load_object_config(object_id),
        body_name="inverse_object",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        mocap=False,
        scale=object_scale,
    )
    # MJCF frames are coordinate systems, not bodies, so attaching to a frame
    # declared on the world body still makes the hand root a direct child of
    # ``worldbody`` -- which is what MuJoCo requires of a mocap body.  The
    # frame itself stays at identity; ``mocap_pos``/``mocap_quat`` set the pose.
    frame = spec.worldbody.add_frame(name="hand_frame")
    spec.attach(_hand_mocap_spec(), prefix="hand/", frame=frame)
    return spec


def _add_sphere(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius, 0.0, 0.0)),
        np.asarray(center, dtype=np.float64),
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kinematic replay of recovery/rollout frames in the fixed-object "
            "inverse frame (no xArm, no physics).  Several --file entries are "
            "concatenated in the order given."
        )
    )
    parser.add_argument(
        "--file",
        action="append",
        nargs="+",
        required=True,
        metavar="PATH[:START:STOP]",
        help="Repeatable; the files play back to back, in the order given.",
    )
    parser.add_argument("--object-id", default=None, help="Defaults to the H5 attr.")
    parser.add_argument(
        "--fps",
        type=float,
        default=100.0,
        help="Playback rate; the recording itself is 100 Hz (control_dt=0.01).",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Zero means through the end."
    )
    args = parser.parse_args()

    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.stride < 1:
        raise ValueError("--stride must be at least one")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")

    specs = [item for group in args.file for item in group]
    segments = [_load_segment(*_parse_file_spec(text)) for text in specs]

    object_id = args.object_id or next(
        (item.object_id for item in segments if item.object_id), "ycb_mustard"
    )
    object_scale = _resolve_object_scale(segments, object_id)

    palm_pose = np.concatenate([item.palm_pose for item in segments], axis=0)
    q_hand = np.concatenate([item.q_hand for item in segments], axis=0)
    contact_pos = np.concatenate([item.contact_pos for item in segments], axis=0)
    contact = np.concatenate([item.contact for item in segments], axis=0)
    # Only the inverted schema carries the fingertip reference the self-check
    # needs; rollout frames are padded with NaN and simply excluded from it.
    if any(item.tip_pose is not None for item in segments):
        tip_pose: np.ndarray | None = np.concatenate(
            [
                item.tip_pose
                if item.tip_pose is not None
                else np.full((len(item.q_hand), 4, 3), np.nan)
                for item in segments
            ],
            axis=0,
        )
    else:
        tip_pose = None

    if q_hand.shape[1] != len(HAND_JOINT_NAMES):
        raise ValueError(
            f"q_hand: expected {len(HAND_JOINT_NAMES)} hand joints, got {q_hand.shape}"
        )
    if contact_pos.shape[1:] != (4, 3):
        raise ValueError(f"contact positions: expected (T, 4, 3), got {contact_pos.shape}")

    total_frames = len(q_hand)
    if args.start_frame >= total_frames:
        raise ValueError(
            f"--start-frame={args.start_frame} exceeds length {total_frames}"
        )
    stop = total_frames
    if args.max_frames:
        stop = min(stop, args.start_frame + args.max_frames)
    # Segment boundaries are computed before the global window is applied, so
    # they stay correct however the replay is trimmed.
    boundaries = [
        int(value)
        for value in np.cumsum([len(item.q_hand) for item in segments])[:-1]
    ]
    sel = slice(args.start_frame, stop)
    palm_pose, q_hand = palm_pose[sel], q_hand[sel]
    contact_pos, contact = contact_pos[sel], contact[sel]
    if tip_pose is not None:
        tip_pose = tip_pose[sel]
    frames = len(q_hand)
    junctions = [
        boundary - args.start_frame
        for boundary in boundaries
        if args.start_frame < boundary < stop
    ]

    model = _scene_spec(object_id, object_scale).compile()
    data = mujoco.MjData(model)

    palm_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "hand/palm_lower"
    )
    if palm_body < 0:
        raise ValueError("hand/palm_lower body not found after attaching the hand")
    mocap_id = int(model.body_mocapid[palm_body])
    if mocap_id < 0:
        raise ValueError("hand/palm_lower is not a mocap body")
    joint_addr = []
    for name in HAND_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"hand/{name}"
        )
        if joint_id < 0:
            raise ValueError(f"hand joint not found: hand/{name}")
        joint_addr.append(int(model.jnt_qposadr[joint_id]))
    tip_site = []
    for name in TIP_SITE_NAMES:
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"hand/{name}"
        )
        if site_id < 0:
            raise ValueError(f"tip site not found: hand/{name}")
        tip_site.append(site_id)

    print(
        f"[INVERSE-REPLAY] frames={frames} object={object_id}@{object_scale:.4f} "
        f"mocap_id={mocap_id} "
        f"all4={float(np.mean(np.all(contact, axis=1))):.1%} "
        f"ge3={float(np.mean(np.sum(contact, axis=1) >= 3)):.1%}"
    )
    for item in segments:
        print(f"[INVERSE-REPLAY] segment {item.label}: {len(item.q_hand)} frames")
    print(
        "[INVERSE-REPLAY] object pinned at the origin; palm_pose_object drives "
        "the hand. green=recorded contact, red=recorded loss"
    )
    print(
        f"[INVERSE-REPLAY] palm_pose_object travel over the episode: "
        f"{np.linalg.norm(palm_pose[-1, :3] - palm_pose[0, :3]) * 1000:.2f} mm"
    )
    # The reason several files may be given at once: the segments were recorded
    # either side of an unrecorded gap, so the joint state is expected to jump
    # while the palm does not.  Print both, so the discontinuity is measured
    # rather than merely seen.
    for junction in junctions:
        joint_gap = float(np.abs(q_hand[junction] - q_hand[junction - 1]).max())
        palm_gap = float(
            np.linalg.norm(palm_pose[junction, :3] - palm_pose[junction - 1, :3])
        )
        print(
            f"[INVERSE-REPLAY] junction {junction - 1}->{junction}: "
            f"max|dq|={joint_gap:.5f} rad, palm move={palm_gap * 1000:.3f} mm"
        )

    tip_error = np.full((frames, 4), np.nan, dtype=np.float64)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = palm_pose[:, :3].mean(axis=0)
            viewer.cam.distance = 0.45
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            hold = 1.0 / args.fps
            for index in range(0, frames, args.stride):
                data.mocap_pos[mocap_id, :] = palm_pose[index, :3]
                data.mocap_quat[mocap_id, :] = palm_pose[index, 3:7]
                for addr, value in zip(joint_addr, q_hand[index]):
                    data.qpos[addr] = float(value)
                mujoco.mj_forward(model, data)

                tip_world = data.site_xpos[np.asarray(tip_site)].copy()
                if tip_pose is not None:
                    tip_error[index] = np.linalg.norm(
                        tip_world - tip_pose[index], axis=-1
                    )

                scene = viewer.user_scn
                scene.ngeom = 0
                for finger in range(4):
                    if contact[index, finger]:
                        _add_sphere(
                            scene, contact_pos[index, finger], 0.009, CONTACT_COLOR
                        )
                    else:
                        _add_sphere(
                            scene, tip_world[finger], 0.007, LOST_COLOR
                        )
                    _add_sphere(scene, tip_world[finger], 0.003, TIP_COLORS[finger])
                viewer.sync()
                time.sleep(hold)
    finally:
        # mj_forward must reproduce the file's own fingertip sites exactly;
        # anything above a fraction of a millimetre means the joint order or
        # the palm pose is being applied wrongly.
        finite = np.isfinite(tip_error)
        if finite.any():
            worst = float(np.nanmax(tip_error))
            median = float(np.nanmedian(tip_error[finite])) * 1000
            verdict = "ok" if worst < 1.0e-3 else "OFF -- check joint order/palm pose"
            print(
                f"[INVERSE-REPLAY] mj_forward fingertips vs fingertip_pose_object: "
                f"max={worst * 1000:.4f} mm, median={median:.4f} mm over "
                f"{int(finite[:, 0].sum())}/{frames} frames ({verdict})"
            )
        else:
            print(
                "[INVERSE-REPLAY] self-check skipped: no frame in this window "
                "carried fingertip_pose_object (rollout segments do not record it)"
            )
        # Same GLX grace period as the other viewers in this bundle.
        time.sleep(2.0)
        print("[INVERSE-REPLAY] viewer resources released")


if __name__ == "__main__":
    main()
