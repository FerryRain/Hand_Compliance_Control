"""Lift the original clean palm-DP data into the strict dual-track schema.

The legacy clean trajectories do not contain a trustworthy pre-MCC q_prior.
For the clean-manifold anchor we therefore use their successful executed hand
trajectory as the nominal task trajectory and encode an ideal execution:

    q_prior = q_cmd = q_live = clean q_hand
    delta_q_comp = e_servo = 0

This never reinterprets the legacy post-controller q_ref as task intent.  A
new randomized dual-track export is used only as a contract/layout template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FIELDS = (
    "episode_id",
    "episode_step",
    "q_prior",
    "q_hand",
    "q_cmd",
    "delta_q_comp",
    "e_servo",
    "fingertip_contact_pos_palm",
    "fingertip_contact_normal_palm",
    "fingertip_contact_mask",
    "q_prior_velocity",
    "fingertip_contact_point_velocity_palm",
    "fingertip_contact_normal_angular_rate_palm",
    "palm_relative_twist_palm",
    "planner_palm_delta_pose_palm",
)


def export(clean: Path, template: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(clean, "r") as source,
        h5py.File(template, "r") as contract,
        h5py.File(output, "w") as target,
    ):
        q_live = np.asarray(source["q_hand"], dtype=np.float32)
        zeros = np.zeros_like(q_live)
        source_values = {
            "episode_id": np.asarray(source["episode_id"]),
            "episode_step": np.asarray(source["episode_step"]),
            "q_prior": q_live,
            "q_hand": q_live,
            "q_cmd": q_live,
            "delta_q_comp": zeros,
            "e_servo": zeros,
            "q_prior_velocity": np.asarray(source["q_velocity"], dtype=np.float32),
        }
        for name in FIELDS:
            value = source_values.get(name)
            if value is None:
                if name not in source:
                    raise KeyError(f"{clean}: missing clean field {name!r}")
                value = np.asarray(source[name])
            expected_tail = contract[name].shape[1:]
            if value.shape[1:] != expected_tail:
                raise ValueError(
                    f"{name}: clean tail {value.shape[1:]} != template {expected_tail}"
                )
            target.create_dataset(
                name,
                data=value,
                chunks=(min(4096, len(value)), *value.shape[1:]),
            )
        for key, value in contract.attrs.items():
            target.attrs[key] = value
        target.attrs["source_file"] = str(clean)
        target.attrs["clean_dual_track_lift"] = True
        target.attrs["clean_dual_track_semantics"] = (
            "q_prior=q_cmd=q_live=successful clean q_hand; "
            "delta_q_comp=e_servo=0; legacy q_ref is not used"
        )
        target.attrs["execution_randomization"] = False

    print(f"[SUCCESS] clean dual-track data -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.clean, args.template, args.output)


if __name__ == "__main__":
    main()
