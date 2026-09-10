"""Splice a DP-failure rollout segment with an expert recovery segment.

This is the second of the two tools flagged as missing in
``DAgger_Recovery_Data_Guide.md`` (section "尚需补充的两个工具"). Given:

- a DP closed-loop rollout (``deploy_dp_inverse.py --rollout-h5``) and the
  frame where it failed ("state A"),
- an expert recovery episode that was executed from state A and already
  passed through the existing ``invert_trajectories.py`` ->
  ``export_palm_dp.py`` pipeline (this tool does NOT reimplement that
  geometry -- see the plan/Guide's per-script division of responsibility),

it produces one training-ready recovery episode H5 for
``train_dp.py --dagger-file``, auditing continuity at the splice boundary and
refusing to write anything if any check fails.

Scope / known limitations (see the plan file for the full rationale):

- Only supports the schemas whose action label is the same dataset as the
  state's ``q_hand`` (i.e. every schema ``export_palm_dp.py`` can produce
  except the dual-track ones): ``force_normal``, ``contact_geometry``,
  ``contact_geometry_planner``, ``contact_geometry_planner_motion``. This is
  not a workaround -- for these schemas ``export_palm_dp.py`` itself never
  exports the expert's *intended* ``q_ref``, only the *executed* ``q_hand``,
  for ANY episode in this repo's pipeline (not just recovery ones). So the
  recovery segment's action label here is exactly as faithful as every other
  DP-direct episode already in the training set.
- ``--recovery`` must already be an ``export_palm_dp.py`` output with the
  same ``dp_state_schema``/``dp_input_frame`` as ``--reference-dp``. This
  tool is a pure splice/audit tool; it never re-derives contact geometry.
- The "planner phase/direction consistency" check the Guide lists is not
  implemented here: it requires provenance from the (not yet built)
  ``generate_manifold_palm_plan.py --recovery-state`` hook. This tool logs
  that check as skipped rather than silently omitting it.
- Only one recovery episode is produced per invocation (single
  ``episode_id``); pass multiple output files to
  ``train_dp.py --dagger-file`` directly, or merge them with the existing
  ``combine_dp_datasets.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from dp_motion_features import MOTION_SCHEMA, causal_motion_features
from export_palm_dp import _wxyz_to_matrix
from palm_planner_features import planner_feature_dim


ROLLOUT_FORMAT = "mcc_closed_loop_observation_v1"
OUTPUT_FORMAT = "mcc_recovery_dataset_v1"

# Restricted to schemas whose action_field == "q_hand" in export_palm_dp.py
# (i.e. every non-dual-track schema it can produce). See module docstring.
SUPPORTED_SCHEMAS = (
    "force_normal",
    "contact_geometry",
    "contact_geometry_planner",
    MOTION_SCHEMA,
)

# Mirrors dp_dataset.STATE_FIELDS_BY_SCHEMA, restricted to SUPPORTED_SCHEMAS.
# planner_palm_delta_pose_palm's dim is resolved at runtime from the
# reference file's planner_waypoints attr via planner_feature_dim().
_PLANNER = "planner_palm_delta_pose_palm"
STATE_FIELDS_BY_SCHEMA: dict[str, tuple[tuple[str, int | None], ...]] = {
    "force_normal": (
        ("q_hand", 16),
        ("fingertip_force_palm", 12),
        ("fingertip_contact_normal_palm", 12),
        ("palm_relative_twist_palm", 6),
    ),
    "contact_geometry": (
        ("q_hand", 16),
        ("fingertip_contact_pos_palm", 12),
        ("fingertip_contact_normal_palm", 12),
        ("fingertip_contact_mask", 4),
        ("palm_relative_twist_palm", 6),
    ),
    "contact_geometry_planner": (
        ("q_hand", 16),
        ("fingertip_contact_pos_palm", 12),
        ("fingertip_contact_normal_palm", 12),
        ("fingertip_contact_mask", 4),
        ("palm_relative_twist_palm", 6),
        (_PLANNER, None),
    ),
    MOTION_SCHEMA: (
        ("q_hand", 16),
        ("fingertip_contact_pos_palm", 12),
        ("fingertip_contact_normal_palm", 12),
        ("fingertip_contact_mask", 4),
        ("q_velocity", 16),
        ("fingertip_contact_point_velocity_palm", 12),
        ("fingertip_contact_normal_angular_rate_palm", 12),
        ("palm_relative_twist_palm", 6),
        (_PLANNER, None),
    ),
}
# Field tail shapes (natural, pre-flatten) matching export_palm_dp.py's own
# dataset shapes -- these are what actually get written/read, not the flat
# sizes above (dp_dataset._flat_feature flattens on load).
FIELD_TAIL_SHAPE = {
    "q_hand": (16,),
    "fingertip_force_palm": (4, 3),
    "fingertip_contact_pos_palm": (4, 3),
    "fingertip_contact_normal_palm": (4, 3),
    "fingertip_contact_mask": (4,),
    "palm_relative_twist_palm": (6,),
    "q_velocity": (16,),
    "fingertip_contact_point_velocity_palm": (4, 3),
    "fingertip_contact_normal_angular_rate_palm": (4, 3),
    # planner field's tail is (waypoints, 6), resolved at runtime.
}
MOTION_ONLY_FIELDS = (
    "q_velocity",
    "fingertip_contact_point_velocity_palm",
    "fingertip_contact_normal_angular_rate_palm",
)

# Same list export_palm_dp.py's sibling combine_dp_datasets.py copies
# verbatim from the reference file; kept identical for consistency.
CONTRACT_ATTRS = (
    "schema_version",
    "dp_input_frame",
    "palm_frame_body",
    "dp_state_schema",
    "state_fields",
    "action_field",
    "action_representation",
    "action_coordinate_space",
    "planner_waypoints",
    "planner_step_frames",
    "planner_horizon_frames",
    "planner_waypoint_dt",
    "planner_horizon_seconds",
    "motion_feature_step_frames",
    "motion_feature_dt",
    "motion_feature_convention",
    "control_dt",
    "contact_normal_polarity",
    "action_dim",
    "planner_feature",
    "dp_coordinate_contract",
    "palm_frame_transform",
)

# dp_tactile_normal_source (deploy_dp_inverse.py) and contact_normal_source
# (invert_trajectories.py) use different vocabularies for the same
# underlying normal-source family -- see invert_trajectories.py:389-410 and
# deploy_dp_inverse.py's --dp-tactile-normal-source choices. Map both onto a
# shared family key so the audit compares like with like instead of
# rejecting every recovery episode on a naming mismatch.
_NORMAL_SOURCE_FAMILY = {
    "source_mesh_oracle": "mesh_oracle",
    "undecomposed_source_mesh_oracle_inward": "mesh_oracle",
    "contact_sensor": "contact_sensor",
    "recorded_contact_sensor": "contact_sensor",
    "analytic_capsule_fallback_inward": "capsule_fallback",
}


class ContinuityViolation(RuntimeError):
    """A splice-boundary audit check exceeded its threshold."""


def _episode_rows(file: h5py.File, episode_id: int, name: str) -> np.ndarray:
    """Return dataset ``name`` for ``episode_id``, sorted by episode_step.

    Duplicated from extract_recovery_state.py rather than shared, matching
    this repo's convention of small self-contained scripts (see e.g.
    deploy_dp_inverse.py's own private ``_episode``, which every other
    reader of the same H5 convention reimplements locally instead of
    importing deploy_dp_inverse.py's heavy simulation dependencies).
    """

    ids = np.asarray(file["episode_id"], dtype=np.int64)
    locations = np.argwhere(ids == episode_id)
    if not locations.size:
        available = np.unique(ids)
        raise ValueError(
            f"episode_id={episode_id} not found in {file.filename!r}; "
            f"available IDs include {available[:20].tolist()}"
        )
    steps = np.asarray(file["episode_step"])
    order = np.argsort(np.asarray([steps[tuple(loc)] for loc in locations], dtype=np.int64))
    locations = locations[order]
    dataset = file[name]
    return np.stack([dataset[tuple(loc)] for loc in locations], axis=0)


def _unique_episode_id(file: h5py.File, explicit: int | None, label: str) -> int:
    if explicit is not None:
        return explicit
    ids = np.unique(np.asarray(file["episode_id"]).reshape(-1).astype(np.int64))
    if len(ids) != 1:
        raise ValueError(
            f"{label} ({file.filename}) contains {len(ids)} episodes "
            f"{ids[:10].tolist()}...; pass an explicit episode id."
        )
    return int(ids[0])


def _palm_relative_twist(pose_object: np.ndarray, twist_object: np.ndarray) -> np.ndarray:
    """Rotate [linear,angular] object-frame twist into the current palm frame.

    Identical formula to export_palm_dp.py's per-block transform
    (object_from_palm = R(quat); palm_from_object = its transpose).
    """

    object_from_palm = _wxyz_to_matrix(np.asarray(pose_object, dtype=np.float64)[..., 3:7])
    palm_from_object = np.swapaxes(object_from_palm, -1, -2)
    twist = np.asarray(twist_object, dtype=np.float64)
    linear = np.einsum("...ij,...j->...i", palm_from_object, twist[..., :3])
    angular = np.einsum("...ij,...j->...i", palm_from_object, twist[..., 3:])
    return np.concatenate([linear, angular], axis=-1).astype(np.float32)


def _quat_angle_deg(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    a = np.asarray(quat_a, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1.0e-12)
    b = np.asarray(quat_b, dtype=np.float64)
    b = b / max(float(np.linalg.norm(b)), 1.0e-12)
    dot = min(1.0, abs(float(np.dot(a, b))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _normal_source_family(value: str) -> str | None:
    return _NORMAL_SOURCE_FAMILY.get(value)


def _read_reference_contract(reference_dp: Path) -> dict[str, object]:
    with h5py.File(reference_dp, "r") as file:
        if str(file.attrs.get("dp_input_frame", "")) != "palm":
            raise ValueError(f"{reference_dp}: dp_input_frame must be 'palm'")
        if str(file.attrs.get("palm_frame_body", "")) != "palm_lower":
            raise ValueError(f"{reference_dp}: palm_frame_body must be 'palm_lower'")
        schema = str(file.attrs.get("dp_state_schema", "force_normal"))
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f"{reference_dp}: dp_state_schema={schema!r} is not supported by "
                f"build_recovery_dataset.py (supported: {SUPPORTED_SCHEMAS}). "
                "Dual-track/task-estimate schemas use action_field='q_prior', "
                "which this tool does not handle -- see module docstring."
            )
        action_field = str(file.attrs.get("action_field", "q_hand"))
        if action_field != "q_hand":
            raise ValueError(
                f"{reference_dp}: action_field={action_field!r}, expected 'q_hand'. "
                "This should be unreachable for a supported schema; the reference "
                "file's contract looks inconsistent."
            )
        control_dt = float(file.attrs.get("control_dt", 0.01))
        fields = list(STATE_FIELDS_BY_SCHEMA[schema])
        planner_waypoints = None
        if any(name == _PLANNER for name, _ in fields):
            planner_waypoints = int(file.attrs["planner_waypoints"])
            dim = planner_feature_dim(planner_waypoints)
            fields = [
                (_PLANNER, dim) if name == _PLANNER else (name, size)
                for name, size in fields
            ]
        contract_attrs = {
            key: file.attrs[key] for key in CONTRACT_ATTRS if key in file.attrs
        }
    return {
        "schema": schema,
        "control_dt": control_dt,
        "fields": tuple(fields),
        "planner_waypoints": planner_waypoints,
        "contract_attrs": contract_attrs,
    }


def _field_tail_shape(name: str, planner_waypoints: int | None) -> tuple[int, ...]:
    if name == _PLANNER:
        assert planner_waypoints is not None
        return (planner_waypoints, 6)
    return FIELD_TAIL_SHAPE[name]


def _rollout_field(
    name: str,
    rollout: h5py.File,
    dp_slice: slice,
    planner_waypoints: int,
) -> np.ndarray:
    if name == "q_hand":
        return np.asarray(rollout["q_live"][dp_slice], dtype=np.float32)
    if name == "palm_relative_twist_palm":
        pose = np.asarray(rollout["palm_pose_object"][dp_slice], dtype=np.float64)
        twist = np.asarray(rollout["palm_twist_object"][dp_slice], dtype=np.float64)
        return _palm_relative_twist(pose, twist)
    if name in MOTION_ONLY_FIELDS:
        # Recomputed once on the final spliced array so the strictly-causal
        # convention is correct across the splice boundary; see module
        # docstring / plan file.
        raise KeyError(name)
    if name not in rollout:
        raise KeyError(
            f"rollout is missing {name!r}; its state_schema may not actually "
            "match --reference-dp's schema."
        )
    value = np.asarray(rollout[name][dp_slice], dtype=np.float32)
    if name == "planner_palm_delta_pose_palm" and value.ndim == 2:
        # The rollout writes this flattened, (N, waypoints*6) -- see
        # deploy_dp_inverse.py's ``live_state[-runtime.planner_waypoints * 6:]``
        # -- while export_palm_dp.py (and hence --recovery/--reference-dp)
        # keeps the waypoint axis, (N, waypoints, 6). Reshape to match so the
        # two segments concatenate; dp_dataset.py's reader flattens the
        # trailing axes back down regardless, so this is a no-op for it.
        value = value.reshape(value.shape[0], planner_waypoints, 6)
    return value


def _recovery_field(
    name: str,
    recovery: h5py.File,
    episode_id: int,
) -> np.ndarray:
    if name in MOTION_ONLY_FIELDS:
        raise KeyError(name)
    if name not in recovery:
        raise KeyError(
            f"--recovery is missing {name!r}; it was not exported with the same "
            "--state-schema as --reference-dp."
        )
    return _episode_rows(recovery, episode_id, name).astype(np.float32)


def build(
    rollout_path: Path,
    failure_frame: int,
    recovery_path: Path,
    reference_dp: Path,
    output_path: Path,
    *,
    recovery_inverted_path: Path | None,
    recovery_episode_id: int | None,
    episode_id: int,
    obs_horizon: int,
    stride: int,
    pred_horizon: int,
    recovery_confirm_frame: int | None,
    recovery_confirm_min_frames: int,
    tail_frames: int,
    max_q_jump_rad: float,
    max_qvel_jump_rad_s: float,
    max_palm_position_error_m: float,
    max_palm_orientation_error_deg: float,
    max_palm_linear_velocity_jump_m_s: float,
    max_palm_angular_velocity_jump_rad_s: float,
    min_recovery_contact_rate: float,
    dry_run: bool,
) -> None:
    contract = _read_reference_contract(reference_dp)
    schema = contract["schema"]
    control_dt = contract["control_dt"]
    fields = contract["fields"]
    planner_waypoints = contract["planner_waypoints"]
    print(f"[INFO] target schema={schema} control_dt={control_dt} fields={[n for n, _ in fields]}")

    # ---- rollout: validate + build the DP-context segment -----------------
    with h5py.File(rollout_path, "r") as rollout:
        if str(rollout.attrs.get("format", "")) != ROLLOUT_FORMAT:
            raise ValueError(f"{rollout_path}: not a deploy_dp_inverse.py --rollout-h5 file")
        rollout_schema = str(rollout.attrs.get("state_schema", ""))
        if rollout_schema != schema:
            raise ValueError(
                f"{rollout_path}: state_schema={rollout_schema!r} != "
                f"--reference-dp schema {schema!r}"
            )
        rollout_dt = float(rollout.attrs.get("control_dt", 0.01))
        if abs(rollout_dt - control_dt) > 1.0e-9:
            raise ValueError(
                f"{rollout_path}: control_dt={rollout_dt} != --reference-dp control_dt={control_dt}"
            )
        num_rollout_frames = int(rollout["episode_step"].shape[0])
        if not (0 <= failure_frame < num_rollout_frames):
            raise ValueError(
                f"--failure-frame={failure_frame} out of range "
                f"[0, {num_rollout_frames - 1}]"
            )

        context_prefix = min(failure_frame, obs_horizon * stride)
        context_prefix = (context_prefix // stride) * stride
        start = failure_frame - context_prefix
        dp_slice = slice(start, failure_frame)
        if context_prefix < obs_horizon * stride:
            print(
                f"[WARN] only {context_prefix} raw frames of DP-drift context "
                f"available (recommended {obs_horizon * stride}); "
                f"--failure-frame={failure_frame} is close to the start of the rollout."
            )

        dp_context: dict[str, np.ndarray] = {}
        for name, _ in fields:
            try:
                dp_context[name] = _rollout_field(
                    name, rollout, dp_slice, planner_waypoints
                )
            except KeyError:
                pass  # motion-only fields, filled after splicing

        rollout_q_at_a = np.asarray(rollout["q_live"][failure_frame], dtype=np.float64)
        if failure_frame > 0:
            rollout_qvel_at_a = (
                rollout_q_at_a
                - np.asarray(rollout["q_live"][failure_frame - 1], dtype=np.float64)
            ) / control_dt
        else:
            rollout_qvel_at_a = np.zeros_like(rollout_q_at_a)
        rollout_palm_pose_at_a = np.asarray(
            rollout["palm_pose_object"][failure_frame], dtype=np.float64
        )
        rollout_palm_twist_at_a = np.asarray(
            rollout["palm_twist_object"][failure_frame], dtype=np.float64
        )
        rollout_tactile_source = str(rollout.attrs.get("dp_tactile_normal_source", ""))

    # ---- recovery: validate + read the full episode ------------------------
    with h5py.File(recovery_path, "r") as recovery:
        recovery_input_frame = str(recovery.attrs.get("dp_input_frame", ""))
        recovery_schema = str(recovery.attrs.get("dp_state_schema", ""))
        recovery_dt = float(recovery.attrs.get("control_dt", 0.01))
        recovery_palm_frame_body = str(recovery.attrs.get("palm_frame_body", ""))
        if recovery_input_frame != "palm" or recovery_schema != schema:
            raise ValueError(
                f"{recovery_path}: dp_input_frame={recovery_input_frame!r} "
                f"dp_state_schema={recovery_schema!r}; expected 'palm'/{schema!r} "
                "(same as --reference-dp). Run invert_trajectories.py then "
                f"export_palm_dp.py --state-schema {schema} on the raw recovery "
                "episode first."
            )
        if abs(recovery_dt - control_dt) > 1.0e-9:
            raise ValueError(
                f"{recovery_path}: control_dt={recovery_dt} != --reference-dp control_dt={control_dt}"
            )
        if recovery_palm_frame_body != "palm_lower":
            raise ValueError(f"{recovery_path}: palm_frame_body must be 'palm_lower'")

        rec_episode_id = _unique_episode_id(recovery, recovery_episode_id, "--recovery")
        recovery_frames: dict[str, np.ndarray] = {}
        for name, _ in fields:
            try:
                recovery_frames[name] = _recovery_field(name, recovery, rec_episode_id)
            except KeyError:
                pass
        recovery_length = len(recovery_frames["q_hand"])
        if recovery_length < 2:
            raise ValueError(f"{recovery_path}: recovery episode has < 2 frames")

    has_contact_mask = "fingertip_contact_mask" in recovery_frames
    if recovery_confirm_frame is None:
        if not has_contact_mask:
            raise ValueError(
                f"schema={schema!r} has no fingertip_contact_mask "
                "(force_normal); pass --recovery-confirm-frame explicitly."
            )
        mask = recovery_frames["fingertip_contact_mask"] > 0.5
        full_contact = mask.sum(axis=-1) == mask.shape[-1]
        recovery_confirm_frame = None
        run_length = 0
        for index, ok in enumerate(full_contact):
            run_length = run_length + 1 if ok else 0
            if run_length >= recovery_confirm_min_frames:
                recovery_confirm_frame = index - recovery_confirm_min_frames + 1
                break
        if recovery_confirm_frame is None:
            raise ValueError(
                f"{recovery_path}: never sustained full contact for "
                f"{recovery_confirm_min_frames} consecutive frames; recovery "
                "did not stabilize, or pass --recovery-confirm-frame manually."
            )
        print(f"[INFO] auto-detected recovery_confirm_frame={recovery_confirm_frame}")

    truncate_at = min(recovery_length, recovery_confirm_frame + tail_frames)
    if has_contact_mask:
        tail_mask = recovery_frames["fingertip_contact_mask"][
            max(0, truncate_at - tail_frames) : truncate_at
        ]
        tail_rate = float((tail_mask > 0.5).all(axis=-1).mean()) if len(tail_mask) else 0.0
        if tail_rate < min_recovery_contact_rate:
            raise ContinuityViolation(
                f"tail contact rate {tail_rate:.4f} < --min-recovery-contact-rate "
                f"{min_recovery_contact_rate}"
            )
        print(f"[INFO] tail contact rate={tail_rate:.4f}")
    for name in list(recovery_frames):
        recovery_frames[name] = recovery_frames[name][:truncate_at]
    recovery_length = truncate_at

    # ---- continuity audit ---------------------------------------------------
    failures: list[str] = []
    checks: list[str] = []

    q_jump = float(np.max(np.abs(rollout_q_at_a - recovery_frames["q_hand"][0].astype(np.float64))))
    checks.append(f"q_jump={q_jump:.5f} rad (limit {max_q_jump_rad})")
    if q_jump > max_q_jump_rad:
        failures.append(checks[-1])

    recovery_qvel0 = (
        recovery_frames["q_hand"][1].astype(np.float64)
        - recovery_frames["q_hand"][0].astype(np.float64)
    ) / control_dt
    qvel_jump = float(np.max(np.abs(rollout_qvel_at_a - recovery_qvel0)))
    checks.append(f"qvel_jump={qvel_jump:.5f} rad/s (limit {max_qvel_jump_rad_s})")
    if qvel_jump > max_qvel_jump_rad_s:
        failures.append(checks[-1])

    recovery_inverted_contact_source: str | None = None
    if recovery_inverted_path is not None:
        with h5py.File(recovery_inverted_path, "r") as inverted:
            inv_episode_id = _unique_episode_id(inverted, recovery_episode_id, "--recovery-inverted")
            inv_dt = float(inverted.attrs.get("control_dt", 0.01))
            if abs(inv_dt - control_dt) > 1.0e-9:
                raise ValueError(
                    f"{recovery_inverted_path}: control_dt={inv_dt} != {control_dt}"
                )
            inv_pose = _episode_rows(inverted, inv_episode_id, "palm_pose_object").astype(np.float64)
            inv_twist = _episode_rows(inverted, inv_episode_id, "palm_twist_object").astype(np.float64)
            recovery_inverted_contact_source = str(
                inverted.attrs.get("contact_normal_source", "")
            )
        inv_pose0 = inv_pose[0]
        inv_twist0 = inv_twist[0]

        position_error = float(np.linalg.norm(rollout_palm_pose_at_a[:3] - inv_pose0[:3]))
        orientation_error_deg = _quat_angle_deg(rollout_palm_pose_at_a[3:7], inv_pose0[3:7])
        position_check = f"palm_position_error={position_error:.5f} m (limit {max_palm_position_error_m})"
        orientation_check = (
            f"palm_orientation_error={orientation_error_deg:.3f} deg "
            f"(limit {max_palm_orientation_error_deg})"
        )
        checks.append(f"{position_check}, {orientation_check}")
        if position_error > max_palm_position_error_m:
            failures.append(position_check)
        if orientation_error_deg > max_palm_orientation_error_deg:
            failures.append(orientation_check)

        linear_jump = float(np.linalg.norm(rollout_palm_twist_at_a[:3] - inv_twist0[:3]))
        angular_jump = float(np.linalg.norm(rollout_palm_twist_at_a[3:] - inv_twist0[3:]))
        linear_check = (
            f"palm_linear_velocity_jump={linear_jump:.5f} m/s "
            f"(limit {max_palm_linear_velocity_jump_m_s})"
        )
        angular_check = (
            f"palm_angular_velocity_jump={angular_jump:.5f} rad/s "
            f"(limit {max_palm_angular_velocity_jump_rad_s})"
        )
        checks.append(f"{linear_check}, {angular_check}")
        if linear_jump > max_palm_linear_velocity_jump_m_s:
            failures.append(linear_check)
        if angular_jump > max_palm_angular_velocity_jump_rad_s:
            failures.append(angular_check)

        rollout_family = _normal_source_family(rollout_tactile_source)
        recovery_family = _normal_source_family(recovery_inverted_contact_source)
        checks.append(
            f"tactile_normal_family: rollout={rollout_tactile_source!r}"
            f"({rollout_family}) vs recovery={recovery_inverted_contact_source!r}"
            f"({recovery_family})"
        )
        if rollout_family is None or recovery_family is None or rollout_family != recovery_family:
            failures.append(checks[-1])
    else:
        print(
            "[WARN] --recovery-inverted not given: skipping palm pose/twist "
            "continuity and tactile-source-family checks (only q/qvel jump "
            "and tail contact rate are audited)."
        )

    print("[AUDIT]")
    for line in checks:
        print(f"  {line}")
    print(
        "[AUDIT] planner phase/direction consistency: SKIPPED (requires the "
        "not-yet-built generate_manifold_palm_plan.py --recovery-state provenance)"
    )

    if failures:
        message = "continuity audit failed:\n" + "\n".join(f"  - {line}" for line in failures)
        print(f"[RECOVERY-DATASET-RESULT] REJECTED\n{message}")
        raise ContinuityViolation(message)

    # ---- splice ---------------------------------------------------------
    spliced: dict[str, np.ndarray] = {}
    for name, _ in fields:
        if name in MOTION_ONLY_FIELDS:
            continue
        spliced[name] = np.concatenate([dp_context[name], recovery_frames[name]], axis=0)
    total_length = context_prefix + recovery_length
    expert_start_frame = context_prefix

    if schema in (MOTION_SCHEMA,):
        control_source_q = spliced["q_hand"]
        episode_id_flat = np.zeros((total_length,), dtype=np.int64)
        q_velocity, point_velocity, normal_rate = causal_motion_features(
            control_source_q,
            spliced["fingertip_contact_pos_palm"],
            spliced["fingertip_contact_normal_palm"],
            spliced["fingertip_contact_mask"],
            episode_id_flat,
            control_dt=control_dt,
            step_frames=int(
                contract["contract_attrs"].get("motion_feature_step_frames", 5)
            ),
        )
        spliced["q_velocity"] = q_velocity
        spliced["fingertip_contact_point_velocity_palm"] = point_velocity
        spliced["fingertip_contact_normal_angular_rate_palm"] = normal_rate

    # ---- window-validity self-check --------------------------------------
    expert_start_strided = expert_start_frame // stride
    if expert_start_frame % stride != 0:
        raise RuntimeError(
            f"internal error: expert_start_frame={expert_start_frame} is not a "
            f"multiple of --stride={stride}; context_prefix computation is buggy."
        )
    strided_length = len(range(0, total_length, stride))
    violations = [
        current
        for current in range(obs_horizon - 1, strided_length - pred_horizon)
        if current + 1 < expert_start_strided
    ]
    if violations:
        raise RuntimeError(
            f"window-validity self-check FAILED: {len(violations)} window(s) "
            f"(first at strided index {violations[0]}) would draw a prediction "
            f"target from the DP-failure segment (expert_start_strided="
            f"{expert_start_strided}, obs_horizon={obs_horizon}). This should be "
            "unreachable given how context_prefix is computed -- please report."
        )
    print(
        f"[INFO] window self-check OK: expert_start_strided={expert_start_strided} "
        f"<= obs_horizon={obs_horizon}, strided_length={strided_length}"
    )
    total_windows = len(range(obs_horizon - 1, strided_length - pred_horizon))
    if total_windows <= 0:
        print(
            f"[WARN] this episode has ZERO usable training windows for "
            f"obs_horizon={obs_horizon}/pred_horizon={pred_horizon}/stride={stride} "
            f"(strided_length={strided_length}). The file will still be written "
            "(it is schema-valid and safe to concatenate with other episodes/"
            "combine_dp_datasets.py), but by itself it contributes nothing to "
            "train_dp.py --dagger-file unless --obs-horizon/--pred-horizon "
            "match the checkpoint you actually intend to fine-tune (they must "
            "match what you pass to train_dp.py, not this tool's defaults)."
        )

    print(
        f"[RECOVERY-DATASET-RESULT] PASSED: {total_length} frames "
        f"({context_prefix} DP-context + {recovery_length} expert), "
        f"expert_start_frame={expert_start_frame}, "
        f"recovery_confirm_frame(raw, within recovery)={recovery_confirm_frame}"
    )

    if dry_run:
        print("[INFO] --dry-run: not writing output")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        for name, _ in fields:
            tail_shape = _field_tail_shape(name, planner_waypoints)
            data = spliced[name].reshape(total_length, 1, *tail_shape).astype(np.float32)
            out.create_dataset(name, data=data)
        out.create_dataset(
            "episode_id",
            data=np.full((total_length, 1), episode_id, dtype=np.int32),
        )
        out.create_dataset(
            "episode_step",
            data=np.arange(total_length, dtype=np.int32).reshape(total_length, 1),
        )
        controller_source = np.zeros((total_length, 1), dtype=np.uint8)
        controller_source[expert_start_frame:] = 1
        out.create_dataset("controller_source", data=controller_source)
        expert_label_valid = np.zeros((total_length, 1), dtype=np.uint8)
        expert_label_valid[expert_start_frame:] = 1
        out.create_dataset("expert_label_valid", data=expert_label_valid)

        for key, value in contract["contract_attrs"].items():
            out.attrs[key] = value
        out.attrs["schema_version"] = OUTPUT_FORMAT
        out.attrs["source_rollout_file"] = str(rollout_path)
        out.attrs["source_recovery_file"] = str(recovery_path)
        if recovery_inverted_path is not None:
            out.attrs["source_recovery_inverted_file"] = str(recovery_inverted_path)
        out.attrs["source_reference_dp_file"] = str(reference_dp)
        out.attrs["failure_frame"] = int(failure_frame)
        out.attrs["expert_start_frame"] = int(expert_start_frame)
        out.attrs["recovery_confirm_frame"] = int(recovery_confirm_frame)
        out.attrs["built_for_obs_horizon"] = int(obs_horizon)
        out.attrs["built_for_stride"] = int(stride)
        out.attrs["built_for_pred_horizon"] = int(pred_horizon)
        out.attrs["action_label_caveat"] = (
            "action_field='q_hand' is the expert's EXECUTED joint position, "
            "matching export_palm_dp.py's existing convention for this schema "
            "(it never exports the intended q_ref for any non-dual-track "
            "schema in this repo) -- not unique to recovery data."
        )
        out.attrs["planner_phase_consistency_audited"] = False
        out.attrs["audit_checks_json"] = json.dumps(checks)

    print(f"[SUCCESS] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--failure-frame", type=int, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument(
        "--recovery-inverted",
        type=Path,
        default=None,
        help="Pre-export_palm_dp.py, post-invert_trajectories.py file for the "
        "same recovery episode. Strongly recommended -- enables the palm "
        "pose/twist continuity and tactile-source-family audit checks.",
    )
    parser.add_argument("--reference-dp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-episode-id", type=int, default=None)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=32,
        help="Only used for the window-validity self-check replication.",
    )
    parser.add_argument("--recovery-confirm-frame", type=int, default=None)
    parser.add_argument("--recovery-confirm-min-frames", type=int, default=50)
    parser.add_argument("--tail-frames", type=int, default=50)
    parser.add_argument("--max-q-jump-rad", type=float, default=0.05)
    parser.add_argument("--max-qvel-jump-rad-s", type=float, default=2.0)
    parser.add_argument("--max-palm-position-error-m", type=float, default=0.003)
    parser.add_argument("--max-palm-orientation-error-deg", type=float, default=2.0)
    parser.add_argument("--max-palm-linear-velocity-jump-m-s", type=float, default=0.05)
    parser.add_argument("--max-palm-angular-velocity-jump-rad-s", type=float, default=0.3)
    parser.add_argument("--min-recovery-contact-rate", type=float, default=0.98)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build(
        args.rollout,
        args.failure_frame,
        args.recovery,
        args.reference_dp,
        args.output,
        recovery_inverted_path=args.recovery_inverted,
        recovery_episode_id=args.recovery_episode_id,
        episode_id=args.episode_id,
        obs_horizon=args.obs_horizon,
        stride=args.stride,
        pred_horizon=args.pred_horizon,
        recovery_confirm_frame=args.recovery_confirm_frame,
        recovery_confirm_min_frames=args.recovery_confirm_min_frames,
        tail_frames=args.tail_frames,
        max_q_jump_rad=args.max_q_jump_rad,
        max_qvel_jump_rad_s=args.max_qvel_jump_rad_s,
        max_palm_position_error_m=args.max_palm_position_error_m,
        max_palm_orientation_error_deg=args.max_palm_orientation_error_deg,
        max_palm_linear_velocity_jump_m_s=args.max_palm_linear_velocity_jump_m_s,
        max_palm_angular_velocity_jump_rad_s=args.max_palm_angular_velocity_jump_rad_s,
        min_recovery_contact_rate=args.min_recovery_contact_rate,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except ContinuityViolation:
        # build() already printed the [RECOVERY-DATASET-RESULT] REJECTED
        # banner and the specific failing checks; a rejected splice is
        # expected CLI behavior (bad input), not a crash, so exit quietly
        # with a non-zero status instead of dumping a traceback.
        sys.exit(1)
