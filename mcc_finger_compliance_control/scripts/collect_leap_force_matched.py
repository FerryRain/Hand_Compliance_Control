"""Match MCC load goals to LEAP actuator candidates and verify stable starts.

All changes are process-local. This is a new experimental configuration;
the former 6/8N hardware-limit-only results remain separately identified.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import h5py
import numpy as np


def matched_force_config(config, targets, recovery_handover=False):
    # The initializer also constructs auxiliary geometry controllers with
    # the loss state machine disabled. Only the actual force controller
    # owns runtime handover state.
    ratio = max(config.overforce_hard_ratio, 6. / min(targets[:3]))
    thumb = max(config.thumb_overforce_hard_ratio or config.overforce_hard_ratio,
                6. / targets[3])
    return replace(config, desired_force_per_finger=targets,
                   overforce_hard_ratio=ratio, thumb_overforce_hard_ratio=thumb,
                   enable_recovery_handover=bool(recovery_handover
                       and config.enable_loss_state_machine and config.use_direct_force_servo))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--leap-profile", choices=("continuous", "peak"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcc-force-targets", type=float, nargs=4)
    parser.add_argument("--recovery-handover", action="store_true")
    own, rest = parser.parse_known_args()
    targets = (tuple(own.mcc_force_targets) if own.mcc_force_targets is not None else
               ((.5, .5, .5, .5) if own.leap_profile == "continuous" else (3., 3., 3., 4.)))
    if not np.isfinite(targets).all() or min(targets) <= 0:
        parser.error("MCC force targets must be positive and finite")
    sys.argv = [sys.argv[0], *rest, "--leap-profile", own.leap_profile, "--output", str(own.output)]
    own.output.mkdir(parents=True, exist_ok=True)
    import surface_mcc_finger as mcc
    import collection_force_regulation as regulation
    import collect_trajectories as collector
    import collection_initialization as initialization
    from held_relative_target import install_held_relative_targets
    restore_targets = install_held_relative_targets()
    original_init = mcc.FullHandMCCFingerController.__init__
    original_config = regulation.collection_force_config
    original_guard = regulation.protect_collection_contacts
    original_teacher = collector.FullHandMCCCollectionPolicy.__call__
    original_mask = collector.FullHandMCCCollectionPolicy.motion_ready_mask
    original_initialization = initialization.StableContactInitializer.update
    initialization_source = textwrap.dedent(inspect.getsource(original_initialization))
    condition = "force <= 2.5*setpoint"
    if initialization_source.count(condition) != 1:
        raise RuntimeError("Expected exactly one initialization force-validity condition")
    patched_initialization = initialization_source.replace(condition,
        "force <= np.maximum(2.5*setpoint, 6.)")
    namespace = {}
    exec(compile(patched_initialization, str(own.output/"initialization_force_validity.py"), "exec"),
         initialization.__dict__, namespace)
    (own.output/"initialization_force_validity.py").write_text(patched_initialization)

    def matched_config(config, *, runtime=False):
        # Low setpoint must not turn normal grazing impulses into a hard
        # emergency release. Maintain an explicit 6N raw hard-warning floor.
        return matched_force_config(config, targets, own.recovery_handover and runtime)

    def initialize(controller, config=None, *values, **kwargs):
        if config is None:
            config = mcc.FullHandMCCFingerConfig()
        return original_init(controller, matched_config(config), *values, **kwargs)

    def configure(config, profile):
        return matched_config(original_config(config, profile), runtime=profile == 'regulated')

    def guard(controller, *values, **kwargs):
        actual = controller.force_setpoint
        controller.force_setpoint = np.maximum(actual, 2.4)
        try:
            return original_guard(controller, *values, **kwargs)
        finally:
            controller.force_setpoint = actual

    def teacher(policy, *values, **kwargs):
        result = original_teacher(policy, *values, **kwargs)
        for lane, controller in enumerate(policy.controllers):
            expected = bool(own.recovery_handover and policy.force_profile_active[lane])
            if controller.config.enable_recovery_handover != expected:
                raise RuntimeError('Runtime handover activation disagrees with accepted motion phase')
        raw_ready = original_mask.fget(policy)
        streak = getattr(policy, "_leap_calibrated_hold", np.zeros_like(raw_ready, dtype=int))
        policy._leap_calibrated_hold = np.where(raw_ready, streak+1, 0)
        return result

    def ready_mask(policy):
        return original_mask.fget(policy) & (getattr(policy, "_leap_calibrated_hold", 0) >= 50)

    mcc.FullHandMCCFingerController.__init__ = initialize
    regulation.collection_force_config = configure
    regulation.protect_collection_contacts = guard
    collector.FullHandMCCCollectionPolicy.__call__ = teacher
    collector.FullHandMCCCollectionPolicy.motion_ready_mask = property(ready_mask)
    initialization.StableContactInitializer.update = namespace["update"]
    protocol = dict(force_targets_n=targets,
                    force_meaning="Existing 3D resultant regulation; not measured normal-only force",
                    initialization_and_motion_targets_match=True,
                    final_tip_warning_n=[max(1.25*f, 3.) for f in targets],
                    final_tip_hard_n=[max(2.5*f, 6.) for f in targets],
                    calibrated_hold_frames_before_motion=50,
                    initial_force_ceiling_n=[max(2.5*f, 6.) for f in targets],
                    initialization_ceiling_reason="Do not tighten the old 7.5/10N absolute acceptance ceiling to1.25N merely because the regulation target is lower",
                    stationary_q_span_limit_rad=.005, stationary_frames=50,
                    mcc_recovery_ik_and_scheduler_unchanged=True,
                    recovery_handover=own.recovery_handover,
                    recovery_handover_revision=2,
                    recovery_handover_activation="regulated motion only, following accepted initialization; asserted on each teacher tick",
                    recovery_joint_step_rule="Existing live-relative recovery cap retained; adjacent-command slew limited only during contact confirmation",
                    finger_target_semantics="Absolute MCC target held through all10 physics substeps; external delta action unchanged",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (own.output/"force_matched_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    try:
        from collect_leap_hardware_probe import main as collect
        collect()
    finally:
        mcc.FullHandMCCFingerController.__init__ = original_init
        regulation.collection_force_config = original_config
        regulation.protect_collection_contacts = original_guard
        collector.FullHandMCCCollectionPolicy.__call__ = original_teacher
        collector.FullHandMCCCollectionPolicy.motion_ready_mask = original_mask
        initialization.StableContactInitializer.update = original_initialization
        restore_targets()
        summary_path = own.output/"summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["force_matched"] = protocol
            summary_path.write_text(json.dumps(summary, indent=2)+"\n")
        if (own.output/"collected.h5").exists():
            with h5py.File(own.output/"collected.h5", "r+") as handle:
                handle.attrs["force_matched_protocol_json"] = json.dumps(protocol)


if __name__ == "__main__":
    main()
