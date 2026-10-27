"""Deterministic analytic benchmarks for Fingertip MCC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from Module.module_1_oracle_surface_model import Cylinder, Plane, Sphere
from Module.module_2_fingertip_mcc.controller import FingertipMCC, MCCConfig


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
  force_rmse_n: float
  overshoot_n: float
  force_violation_probability: float
  settling_time_s: float
  contact_loss_count_after_settling: int
  max_tangential_error_m: float
  max_abs_offset_m: float
  max_abs_velocity_m_s: float
  max_abs_acceleration_m_s2: float
  saturation_count: int

  def to_dict(self) -> dict[str, float | int]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class Trace:
  time_s: NDArray[np.float64]
  measured_force_n: NDArray[np.float64]
  tangential_error_m: NDArray[np.float64]
  offset_m: NDArray[np.float64]
  velocity_m_s: NDArray[np.float64]
  acceleration_m_s2: NDArray[np.float64]
  planned_position_m: NDArray[np.float64]
  commanded_position_m: NDArray[np.float64]
  compliance_direction: NDArray[np.float64]
  saturation_count: int


DEFAULT_CONFIG = MCCConfig()
CONTACT_STIFFNESS_N_M = 1000.0
FORCE_LIMIT_N = 3.5
EVALUATION_WINDOW_S = 1.0
CONTACT_SETTLING_CUTOFF_S = 0.25


def _run_contact_trace(
  *,
  planned_kinematics: Callable[
    [float],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
  ],
  signed_distance: Callable[[NDArray[np.float64]], float],
  desired_force_n: float,
  duration_s: float = 3.0,
  config: MCCConfig = DEFAULT_CONFIG,
) -> Trace:
  controller = FingertipMCC(config)
  step_count = int(round(duration_s / config.dt_s))
  time_s = np.arange(step_count, dtype=np.float64) * config.dt_s
  forces = np.zeros(step_count, dtype=np.float64)
  tangential_errors = np.zeros(step_count, dtype=np.float64)
  offsets = np.zeros(step_count, dtype=np.float64)
  velocities = np.zeros(step_count, dtype=np.float64)
  accelerations = np.zeros(step_count, dtype=np.float64)
  planned_positions = np.zeros((step_count, 3), dtype=np.float64)
  commanded_positions = np.zeros((step_count, 3), dtype=np.float64)
  compliance_directions = np.zeros((step_count, 3), dtype=np.float64)
  measured_force = 0.0
  saturation_count = 0

  for index, timestamp in enumerate(time_s):
    plan, compliance_direction, tangent = planned_kinematics(float(timestamp))
    command = controller.step(
      plan,
      compliance_direction,
      desired_force_n,
      measured_force,
    )
    penetration = max(0.0, -signed_distance(command.position_command))
    measured_force = CONTACT_STIFFNESS_N_M * penetration
    displacement = command.position_command - plan
    tangential_errors[index] = abs(float(np.dot(displacement, tangent)))
    forces[index] = measured_force
    offsets[index] = command.offset_m
    velocities[index] = command.velocity_m_s
    accelerations[index] = command.acceleration_m_s2
    planned_positions[index] = plan
    commanded_positions[index] = command.position_command
    compliance_directions[index] = compliance_direction
    saturation_count += len(command.saturated_limits)

  return Trace(
    time_s=time_s,
    measured_force_n=forces,
    tangential_error_m=tangential_errors,
    offset_m=offsets,
    velocity_m_s=velocities,
    acceleration_m_s2=accelerations,
    planned_position_m=planned_positions,
    commanded_position_m=commanded_positions,
    compliance_direction=compliance_directions,
    saturation_count=saturation_count,
  )


def _metrics(trace: Trace, desired_force_n: float, config: MCCConfig) -> BenchmarkMetrics:
  evaluation_start = trace.time_s[-1] - EVALUATION_WINDOW_S + config.dt_s
  evaluation_mask = trace.time_s >= evaluation_start
  errors = trace.measured_force_n[evaluation_mask] - desired_force_n
  force_rmse = float(np.sqrt(np.mean(errors**2)))
  overshoot = max(0.0, float(np.max(trace.measured_force_n) - desired_force_n))
  violation_probability = float(np.mean(trace.measured_force_n > FORCE_LIMIT_N))

  tolerance = max(0.05 * desired_force_n, 0.02)
  within_tolerance = np.abs(trace.measured_force_n - desired_force_n) <= tolerance
  settling_time = float("inf")
  for index in range(len(within_tolerance)):
    if np.all(within_tolerance[index:]):
      settling_time = float(trace.time_s[index])
      break

  contact_mask = trace.time_s >= CONTACT_SETTLING_CUTOFF_S
  contact_loss_count = int(np.sum(trace.measured_force_n[contact_mask] <= 1e-9))
  return BenchmarkMetrics(
    force_rmse_n=force_rmse,
    overshoot_n=overshoot,
    force_violation_probability=violation_probability,
    settling_time_s=settling_time,
    contact_loss_count_after_settling=contact_loss_count,
    max_tangential_error_m=float(np.max(trace.tangential_error_m)),
    max_abs_offset_m=float(np.max(np.abs(trace.offset_m))),
    max_abs_velocity_m_s=float(np.max(np.abs(trace.velocity_m_s))),
    max_abs_acceleration_m_s2=float(np.max(np.abs(trace.acceleration_m_s2))),
    saturation_count=trace.saturation_count,
  )


def trace_static_contact(
  desired_force_n: float,
  config: MCCConfig = DEFAULT_CONFIG,
) -> Trace:
  plane = Plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

  def kinematics(_: float):
    return (
      np.array([0.0, 0.0, 0.0]),
      np.array([0.0, 0.0, -1.0]),
      np.array([1.0, 0.0, 0.0]),
    )

  return _run_contact_trace(
    planned_kinematics=kinematics,
    signed_distance=plane.signed_distance,
    desired_force_n=desired_force_n,
    config=config,
  )


def run_static_contact(
  desired_force_n: float,
  config: MCCConfig = DEFAULT_CONFIG,
) -> BenchmarkMetrics:
  return _metrics(trace_static_contact(desired_force_n, config), desired_force_n, config)


def trace_tangential_sliding(
  desired_force_n: float = 2.0,
  config: MCCConfig = DEFAULT_CONFIG,
) -> Trace:
  plane = Plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

  def kinematics(timestamp: float):
    x = -0.06 + 0.04 * timestamp
    return (
      np.array([x, 0.0, 0.0]),
      np.array([0.0, 0.0, -1.0]),
      np.array([1.0, 0.0, 0.0]),
    )

  return _run_contact_trace(
    planned_kinematics=kinematics,
    signed_distance=plane.signed_distance,
    desired_force_n=desired_force_n,
    config=config,
  )


def run_tangential_sliding(
  desired_force_n: float = 2.0,
  config: MCCConfig = DEFAULT_CONFIG,
) -> BenchmarkMetrics:
  return _metrics(
    trace_tangential_sliding(desired_force_n, config),
    desired_force_n,
    config,
  )


def trace_curved_surface(
  surface: str,
  desired_force_n: float = 2.0,
  config: MCCConfig = DEFAULT_CONFIG,
) -> Trace:
  if surface == "sphere":
    shape = Sphere([0.0, 0.0, 0.0], 0.1)

    def kinematics(timestamp: float):
      angle = -0.7 + 1.4 * timestamp / 3.0
      outward = np.array([np.cos(angle), np.sin(angle), 0.0])
      tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
      return 0.1 * outward, -outward, tangent

  elif surface == "cylinder":
    shape = Cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.08, 0.12)

    def kinematics(timestamp: float):
      angle = -0.8 + 1.6 * timestamp / 3.0
      outward = np.array([np.cos(angle), np.sin(angle), 0.0])
      tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
      plan = 0.08 * outward + np.array([0.0, 0.0, -0.06 + 0.04 * timestamp])
      return plan, -outward, tangent

  else:
    raise ValueError("surface must be 'sphere' or 'cylinder'")

  return _run_contact_trace(
    planned_kinematics=kinematics,
    signed_distance=shape.signed_distance,
    desired_force_n=desired_force_n,
    config=config,
  )


def run_curved_surface(
  surface: str,
  desired_force_n: float = 2.0,
  config: MCCConfig = DEFAULT_CONFIG,
) -> BenchmarkMetrics:
  return _metrics(
    trace_curved_surface(surface, desired_force_n, config),
    desired_force_n,
    config,
  )
