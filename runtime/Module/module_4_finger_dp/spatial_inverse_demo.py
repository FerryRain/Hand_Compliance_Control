"""Run and visualize the minimum real forward -> spatial inverse pair."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile

_CACHE = Path(tempfile.gettempdir()) / "handcomp-spatial-inverse-v1"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["XDG_CACHE_HOME"] = str(_CACHE)
os.environ["MESA_SHADER_CACHE_DIR"] = str(_CACHE / "mesa_shader_cache")
os.environ["MPLCONFIGDIR"] = str(_CACHE / "matplotlib")

from Module.module_4_finger_dp.spatial_inverse_data import (
  SpatialInverseConfig,
  audit_spatial_inverse_pair,
  run_spatial_inverse_physical_pair,
  save_spatial_inverse_pair,
)
from Module.module_4_finger_dp.spatial_inverse_visual import (
  render_spatial_inverse_dashboard,
  render_spatial_inverse_video,
)


DEFAULT_OUTPUT = Path("Module/generated/visual_demo/spatial_inverse_v1")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--duration", type=float, default=3.0)
  parser.add_argument("--traversal", type=float, default=-0.010)
  parser.add_argument("--lateral", type=float, default=0.0015)
  parser.add_argument("--fps", type=int, default=20)
  parser.add_argument("--skip-video", action="store_true")
  parser.add_argument("--require-accepted", action="store_true")
  arguments = parser.parse_args()
  config = replace(
    SpatialInverseConfig(),
    duration_s=float(arguments.duration),
    object_traversal_y_m=float(arguments.traversal),
    object_lateral_x_m=float(arguments.lateral),
  )
  output = arguments.output
  output.mkdir(parents=True, exist_ok=True)
  pair = run_spatial_inverse_physical_pair(config)
  audit = audit_spatial_inverse_pair(pair)
  save_spatial_inverse_pair(output / "forward_replay_pair.h5", pair)
  render_spatial_inverse_dashboard(pair, audit, output / "forward_replay_audit.png")
  if not arguments.skip_video:
    render_spatial_inverse_video(
      pair,
      audit,
      output / "forward_spatial_inverse_replay.mp4",
      fps=arguments.fps,
    )

  # This run validates the mechanics with an MCC-generated forward episode.
  # It is Dataset-D diagnostic evidence, not the formal non-MCC Dataset-I and
  # is therefore never silently exposed as a training-ready contribution.
  payload = {
    "artifact_type": "DATASET_D_SPATIAL_INVERSE_PIPELINE_DIAGNOSTIC",
    "config": asdict(config),
    "raw_spatial_replay_audit": asdict(audit),
    "provenance": {
      "forward": pair.forward_provenance,
      "inversion_mode": pair.inversion_mode,
      "time_mapping": pair.time_mapping,
      "finger_command_mapping": pair.finger_command_mapping,
      "replay_repair_policy": pair.replay_repair_policy,
      "replay_force_contact_source": "FRESH_PHYSICS_MEASUREMENT",
    },
    "dataset_class": "DATASET_D_DIAGNOSTIC",
    "formal_dataset_i_ready": False,
    "formal_dataset_i_blocker": "FORWARD_PROVENANCE_USES_FINGERTIP_MCC",
    "formal_dataset_i_blockers": [
      "FORWARD_PROVENANCE_USES_FINGERTIP_MCC",
      "SINGLE_THREE_SECOND_EPISODE_IS_NOT_A_TRAINING_DATASET",
      "PER_FINGER_REPLAY_EQUIVALENCE_NOT_YET_VALIDATED",
    ],
    "training_allowed": False,
    "training_started": False,
    "dp_evaluated": False,
  }
  (output / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  verdict = "PASSED" if audit.accepted else "FAILED"
  reasons = ", ".join(audit.reasons) if audit.reasons else "NONE"
  (output / "README.md").write_text(
    "# Forward → spatial inverse → physical replay\n\n"
    f"- raw replay gate: `{verdict}`\n"
    f"- gate reasons: `{reasons}`\n"
    "- inversion: `SPATIAL_ONLY`; sample `t` remains sample `t`\n"
    "- finger proposal: exact recorded forward `q_cmd[t]`\n"
    "- replay finger repair: `NONE`\n"
    "- replay force/contact: fresh simulator measurements\n"
    "- classification: `Dataset-D diagnostic`; **not DP training/evaluation**\n"
    "- formal Dataset-I remains blocked because this first forward collector uses Finger MCC\n\n"
    "Inspect `forward_spatial_inverse_replay.mp4` first. The complete paired raw data "
    "(`T_HO`, q/dq/q_cmd, fresh F/C/r/n, arm/palm state) is in "
    "`forward_replay_pair.h5`; plots and exact metrics are in "
    "`forward_replay_audit.png` and `summary.json`.\n",
    encoding="utf-8",
  )
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 2 if arguments.require_accepted and not audit.accepted else 0


if __name__ == "__main__":
  raise SystemExit(main())
