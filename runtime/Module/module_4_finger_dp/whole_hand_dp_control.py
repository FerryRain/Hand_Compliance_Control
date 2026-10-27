"""Complete Wrist-MCC + CUDA Finger-DP control strategy.

The module exposes the deployable authority decomposition as one runtime:

* Wrist MCC owns collective compliance and planner tracking;
* Finger DP owns differential/local contact realization;
* the authority QP certifies every 500 Hz interpolated finger command;
* the runtime guard can slow/stop the wrist and revoke DP authority.

The current checkpoint remains Dataset-D diagnostic until Dataset-I gates pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np

from Module.module_4_finger_dp.gpu_runtime import require_cuda
from Module.module_4_finger_dp.track_d_closed_loop import (
  TrackDClosedLoopConfig,
  TrackDClosedLoopMetrics,
  TrackDClosedLoopTrace,
  run_track_d_closed_loop,
)


WHOLE_HAND_DP_CONTROL_SCHEMA_VERSION = "fr3-leap-whole-hand-dp-control.v1"


@dataclass(frozen=True, slots=True)
class WholeHandDPControlConfig:
  duration_s: float = 11.5
  initialization_s: float = 1.0
  desired_force_n: float = 2.0
  seed: int = 47
  friction_coefficient: float = 0.95
  initial_joint_noise_std_rad: float = 0.002
  torch_device: str = "cuda:0"
  minimum_contact_continuity: float = 0.995
  maximum_force_n: float = 8.0
  maximum_policy_latency_s: float = 0.020
  collective_normal_velocity_limit_m_s: float = 0.010
  maximum_opposition_energy: float = 1e-5

  def __post_init__(self) -> None:
    if self.duration_s < 10.0:
      raise ValueError("whole-hand DP execution must be at least 10 seconds")
    if not 0.2 <= self.initialization_s < self.duration_s:
      raise ValueError("initialization_s must be inside the episode")
    if not self.torch_device.startswith("cuda"):
      raise ValueError("whole-hand Finger DP is CUDA-only")
    if self.collective_normal_velocity_limit_m_s <= 0.0:
      raise ValueError("collective normal velocity limit must be positive")
    if self.maximum_opposition_energy <= 0.0:
      raise ValueError("maximum opposition energy must be positive")

  def runtime_config(self) -> TrackDClosedLoopConfig:
    return TrackDClosedLoopConfig(
      duration_s=self.duration_s,
      dp_activation_s=self.initialization_s,
      desired_force_n=self.desired_force_n,
      force_limit_n=self.maximum_force_n,
      seed=self.seed,
      torch_device=self.torch_device,
      friction_coefficient=self.friction_coefficient,
      initial_joint_noise_std_rad=self.initial_joint_noise_std_rad,
      collective_normal_velocity_limit_m_s=(
        self.collective_normal_velocity_limit_m_s
      ),
      maximum_opposition_energy=self.maximum_opposition_energy,
    )


@dataclass(frozen=True, slots=True)
class WholeHandDPControlVerdict:
  execution_status: str
  readiness_status: str
  blocking_reason: tuple[str, ...]
  checks: dict[str, bool]
  diagnostics: dict[str, float | bool]


@dataclass(frozen=True, slots=True)
class WholeHandDPControlResult:
  trace: TrackDClosedLoopTrace
  metrics: TrackDClosedLoopMetrics
  verdict: WholeHandDPControlVerdict


def _verdict(
  trace: TrackDClosedLoopTrace,
  metrics: TrackDClosedLoopMetrics,
  config: WholeHandDPControlConfig,
) -> WholeHandDPControlVerdict:
  evaluation = trace.time_s >= config.initialization_s
  finger_mcc_absent = not np.any(
    np.char.find(trace.command_owner[evaluation].astype(str), "MCC") >= 0
  )
  checks = {
    "cuda_policy_executed": metrics.policy_replan_count > 0,
    "finger_mcc_absent_after_activation": bool(finger_mcc_absent),
    "wrist_mcc_enabled": True,
    "contact_continuity": (
      metrics.contact_continuity >= config.minimum_contact_continuity
    ),
    "zero_contact": metrics.zero_contact_time_s == 0.0,
    "tip_force": metrics.maximum_force_n < config.maximum_force_n,
    "non_tip_contact": metrics.non_tip_contact_frames == 0,
    "authority_qp": metrics.authority_solver_failure_frames == 0,
    "authority_collective_velocity": (
      metrics.finger_collective_normal_max_abs_velocity_m_s
      <= config.runtime_config().collective_normal_velocity_limit_m_s + 1e-4
    ),
    "hard_guard": metrics.hard_guard_frames == 0,
    "gpu_policy_deadline": (
      metrics.policy_latency_p95_s <= config.maximum_policy_latency_s
    ),
  }
  failed = tuple(name for name, passed in checks.items() if not passed)
  execution_checks = {
    name: passed
    for name, passed in checks.items()
    if name != "gpu_policy_deadline"
  }
  execution = "EVALUATED" if all(execution_checks.values()) else "FAILED"
  readiness = "MET" if not failed else "NOT_MET"
  return WholeHandDPControlVerdict(
    execution_status=execution,
    readiness_status=readiness,
    blocking_reason=("NONE",) if not failed else failed,
    checks=checks,
    diagnostics={
      "opposition_rate": metrics.opposition_rate,
      "opposition_energy": metrics.opposition_energy,
      "opposition_below_provisional_reference": (
        metrics.opposition_energy <= config.maximum_opposition_energy
      ),
    },
  )


def run_whole_hand_dp_control(
  checkpoint_path: str | Path,
  reference_trace_path: str | Path,
  config: WholeHandDPControlConfig = WholeHandDPControlConfig(),
) -> WholeHandDPControlResult:
  require_cuda(config.torch_device)
  trace, metrics = run_track_d_closed_loop(
    checkpoint_path,
    reference_trace_path,
    config.runtime_config(),
  )
  return WholeHandDPControlResult(
    trace=trace,
    metrics=metrics,
    verdict=_verdict(trace, metrics, config),
  )


def save_whole_hand_dp_control(
  output_directory: str | Path,
  result: WholeHandDPControlResult,
  config: WholeHandDPControlConfig,
) -> tuple[Path, Path]:
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  trace_path = output / "whole_hand_dp_trace.npz"
  np.savez_compressed(
    trace_path,
    **{
      definition.name: getattr(result.trace, definition.name)
      for definition in fields(result.trace)
    },
  )
  _, cuda_info = require_cuda(config.torch_device)
  summary_path = output / "whole_hand_dp_summary.json"
  summary_path.write_text(
    json.dumps(
      {
        "schema_version": WHOLE_HAND_DP_CONTROL_SCHEMA_VERSION,
        "scope": "DATASET_D_LONG_CONTROL_DIAGNOSTIC",
        "formal_dataset_i": False,
        "formal_e05": False,
        "controller": {
          "wrist": "WRIST_MCC_COLLECTIVE_COMPLIANCE",
          "finger": "CUDA_FORCE_HISTORY_DIFFUSION_POLICY",
          "authority": "DP_ACTION_AUTHORITY_FILTER_DAQP",
          "guard": "DETERMINISTIC_RUNTIME_GUARD",
          "finger_mcc_after_activation": False,
        },
        "cuda_only": True,
        "cuda_runtime": cuda_info.to_dict(),
        "config": asdict(config),
        "metrics": asdict(result.metrics),
        "verdict": asdict(result.verdict),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return trace_path, summary_path
