"""Reproducible M06--M12 Oracle + explicit-MCC module benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Any

import numpy as np

from Module.module_1_oracle_surface_model import OracleSurfaceModel, Plane
from Module.module_2_fingertip_mcc import (
  FingertipMCC,
  FullRobotFingertipMCC,
  MCCConfig,
)
from Module.module_6_prefix_executor import (
  BarrierState,
  ExecutorConfig,
  ExecutorObservation,
  MCCBaselineAdapter,
  ParticipantState,
  PlannedPrefix,
  PrefixSample,
  PrefixSource,
  TransactionState,
  TransactionType,
  TransactionalPrefixExecutor,
)
from Module.module_7_contact_mode_graph import (
  CommitContext,
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)
from Module.module_8_cheap_cert import CheapCert, CheapCertInput
from Module.module_9_continuous_optimize import (
  ContinuousOptimizer,
  LinearizedHandKinematics,
  OptimizationConfig,
  OptimizationRequest,
  PlannerState,
)
from Module.module_10_exact_prefix_audit import (
  AuditConfig,
  AuditEnvironment,
  AuditRequest,
  ExactPrefixAuditor,
)
from Module.module_11_lazy_beam_search import (
  BeamSearchConfig,
  LazyBeamSearch,
  PlanningCandidate,
  SearchResult,
)
from Module.module_12_shadow_viability import ShadowViabilityEvaluator


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "Module/M06_M12_MCC_BASELINE_PROTOCOL.md"
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/m06_m12_mcc_baseline"
SEED = 7
MODULE_CODE_PATHS = tuple(
  REPO_ROOT / path
  for path in (
    "Module/module_6_prefix_executor/executor.py",
    "Module/module_7_contact_mode_graph/graph.py",
    "Module/module_8_cheap_cert/cheap_cert.py",
    "Module/module_9_continuous_optimize/optimizer.py",
    "Module/module_10_exact_prefix_audit/audit.py",
    "Module/module_11_lazy_beam_search/search.py",
    "Module/module_12_shadow_viability/shadow.py",
    "Module/m06_m12_benchmark.py",
  )
)


def _percentile(values: list[float], percentile: float = 95.0) -> float:
  return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _code_hash() -> str:
  digest = hashlib.sha256()
  for path in MODULE_CODE_PATHS:
    digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
    digest.update(path.read_bytes())
  return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
  try:
    commit = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    dirty = bool(
      subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
      ).stdout.strip()
    )
  except (OSError, subprocess.CalledProcessError):
    return {"commit": None, "worktree_dirty": None}
  return {"commit": commit, "worktree_dirty": dirty}


def _machine_metadata() -> dict[str, Any]:
  cpu_model = platform.processor()
  try:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
      if line.lower().startswith("model name"):
        cpu_model = line.split(":", maxsplit=1)[1].strip()
        break
  except OSError:
    pass
  return {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_model": cpu_model or None,
    "logical_cpu_count": os.cpu_count(),
  }


def _fixture(contact_set: frozenset[int] = frozenset({1, 2})):
  tips = np.array(
    [
      [-0.045, -0.025, 0.010],
      [0.045, -0.025, 0.010],
      [-0.045, 0.030, 0.010],
      [0.045, 0.030, 0.010],
    ],
    dtype=np.float64,
  )
  for finger in contact_set:
    tips[finger - 1, 2] = 0.0
  surface = OracleSurfaceModel(
    Plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    version="oracle-plane-m06-m12.v1",
  )
  kinematics = LinearizedHandKinematics.canonical(tips)
  state = PlannerState(
    joint_positions_rad=np.zeros(12),
    wrist_position_m=[0.0, 0.0, 0.12],
    fingertip_positions_m=tips,
    actual_contact_set=contact_set,
    surface_model_version=surface.version,
  )
  graph = ContactModeGraph()
  optimizer = ContinuousOptimizer(
    graph,
    surface,
    kinematics,
    OptimizationConfig(waypoint_count=9),
  )
  return graph, surface, kinematics, state, optimizer


def _request(
  state: PlannerState,
  primitive: ContactPrimitive,
  *,
  prefix_id: str,
  rng: np.random.Generator | None = None,
  progress_gain_m: float | None = None,
) -> OptimizationRequest:
  random = rng or np.random.default_rng(SEED)
  progress = 0.004 if progress_gain_m is None else float(progress_gain_m)
  if primitive.kind is PrimitiveKind.WRIST_ADJUST:
    direction = random.normal(size=3)
    direction[2] *= 0.25
    direction /= np.linalg.norm(direction)
    distance = float(random.uniform(0.002, 0.006))
    return OptimizationRequest(
      state,
      primitive,
      target_wrist_position_m=state.wrist_position_m + distance * direction,
      prefix_id=prefix_id,
      progress_gain_m=progress,
    )
  assert primitive.finger_id is not None
  target = np.array(state.fingertip_positions_m[primitive.finger_id - 1], copy=True)
  planar = random.normal(size=2)
  planar /= np.linalg.norm(planar)
  distance = float(random.uniform(0.002, 0.006))
  if primitive.kind is PrimitiveKind.SLIDE:
    target[:2] += distance * planar
    target[2] = 0.0
  elif primitive.kind is PrimitiveKind.REPOSITION:
    target[:2] += distance * planar
    target[2] = max(target[2], 0.007)
  elif primitive.kind is PrimitiveKind.MAKE:
    target[:2] += distance * planar
    target[2] = 0.0
  elif primitive.kind is PrimitiveKind.BREAK:
    target[2] += 0.006
  return OptimizationRequest(
    state,
    primitive,
    target_position_m=target,
    prefix_id=prefix_id,
    progress_gain_m=progress,
  )


def _auditor(graph, surface, kinematics, *, link_clearance_fn=None, config=None):
  exact_link_clearance = link_clearance_fn or (lambda q, wrist, tips: 0.02)
  return ExactPrefixAuditor(
    graph,
    AuditEnvironment(
      surface,
      kinematics,
      link_clearance_fn=exact_link_clearance,
    ),
    config or AuditConfig(subdivisions_per_segment=9),
  )


def _audit(
  auditor: ExactPrefixAuditor,
  prefix: PlannedPrefix,
  state: PlannerState,
  *,
  confirmations: dict[int, float] | None = None,
  issued_at_s: float = 0.0,
):
  return auditor.audit(
    AuditRequest(
      prefix,
      state,
      CommitContext(
        state.actual_contact_set,
        replacement_confirmation_s=confirmations or {},
      ),
      issued_at_s,
    )
  )


def _observation(
  state: PlannerState,
  timestamp_s: float,
  *,
  wrist=None,
  tips=None,
  q=None,
  contacts=None,
  blocked=None,
  model_version=None,
  forces=None,
) -> ExecutorObservation:
  actual = state.actual_contact_set if contacts is None else frozenset(contacts)
  measured_forces = (
    [2.0 if finger in actual else 0.0 for finger in range(1, 5)]
    if forces is None
    else forces
  )
  return ExecutorObservation(
    timestamp_s=timestamp_s,
    surface_model_version=model_version or state.surface_model_version,
    wrist_position_m=state.wrist_position_m if wrist is None else wrist,
    fingertip_positions_m=state.fingertip_positions_m if tips is None else tips,
    joint_positions_rad=state.joint_positions_rad if q is None else q,
    fingertip_forces_n=measured_forces,
    outward_normals=np.tile([0.0, 0.0, 1.0], (4, 1)),
    actual_contact_set=actual,
    blocked_fingers=blocked or {},
  )


def _mcc_executor() -> TransactionalPrefixExecutor:
  controllers = tuple(FingertipMCC(MCCConfig(dt_s=0.01)) for _ in range(4))
  return TransactionalPrefixExecutor(
    ExecutorConfig(default_timeout_s=0.5),
    mcc_adapter=MCCBaselineAdapter(FullRobotFingertipMCC(controllers)),
  )


def _two_finger_prefix(graph, surface, kinematics, state):
  desired = np.array(state.fingertip_positions_m, copy=True)
  desired[2] += np.array([0.004, 0.0, 0.0])
  desired[3] += np.array([0.008, 0.0, 0.0])
  q_terminal, _ = kinematics.solve(
    desired,
    state.wrist_position_m,
    state.joint_positions_rad,
    (3, 4),
    damping=1e-4,
    iterations=8,
  )
  terminal_tips = kinematics.forward(q_terminal, state.wrist_position_m)
  prefix = PlannedPrefix(
    prefix_id="m06-async-reposition",
    transaction_type=TransactionType.FINGER_RECONFIGURE,
    primitive_kind=PrimitiveKind.REPOSITION.value,
    finger_id=3,
    surface_model_version=state.surface_model_version,
    root_contact_set=state.actual_contact_set,
    expected_terminal_contact_set=state.actual_contact_set,
    samples=(
      PrefixSample(0.0, state.wrist_position_m, state.fingertip_positions_m, state.joint_positions_rad),
      PrefixSample(0.1, state.wrist_position_m, terminal_tips, q_terminal),
    ),
    participating_fingers=(3, 4),
    anchor_fingers=(1, 2),
  )
  result = _audit(_auditor(graph, surface, kinematics), prefix, state)
  if not result.certified or result.certificate is None:
    raise RuntimeError(f"M06 fixture did not certify: {result.reasons}")
  return prefix, result.certificate


def _benchmark_m06(graph, surface, kinematics, state):
  prefix, certificate = _two_finger_prefix(graph, surface, kinematics, state)
  terminal = prefix.samples[-1].fingertip_positions_m

  asynchronous = _mcc_executor()
  asynchronous.commit(prefix, certificate, _observation(state, 0.0))
  partial = np.array(state.fingertip_positions_m, copy=True)
  partial[2] = terminal[2]
  first = asynchronous.step(_observation(state, 0.05, tips=partial))
  waiting = asynchronous.step(_observation(state, 0.10, tips=terminal))
  duplicate = asynchronous.step(_observation(state, 0.10, tips=terminal))
  closed = asynchronous.step(_observation(state, 0.11, tips=terminal))
  first_states = {record.participant: record.state for record in first.participants}
  async_pass = (
    first_states.get("FINGER_3") is ParticipantState.DONE
    and first_states.get("FINGER_4") is ParticipantState.RUNNING
    and waiting.barrier_state is BarrierState.WAITING_FOR_FRESH_OBSERVATION
    and duplicate.barrier_state is BarrierState.WAITING_FOR_FRESH_OBSERVATION
    and duplicate.transaction_state is TransactionState.RUNNING
    and closed.transaction_state is TransactionState.DONE
    and asynchronous.consume_barrier_snapshot() is not None
  )

  blocked_executor = _mcc_executor()
  blocked_executor.commit(prefix, certificate, _observation(state, 0.0))
  blocked_executor.step(
    _observation(state, 0.02, blocked={3: "SUSPECTED_OBJECT_BLOCKAGE"})
  )
  safe_peer = np.array(terminal, copy=True)
  safe_peer[2] = state.fingertip_positions_m[2]
  blocked_executor.step(_observation(state, 0.10, tips=safe_peer))
  blocked_final = blocked_executor.step(_observation(state, 0.11, tips=safe_peer))
  blocked_states = {
    record.participant: record.state for record in blocked_final.participants
  }
  blocked_pass = (
    blocked_final.transaction_state is TransactionState.BLOCKED
    and blocked_states.get("FINGER_3") is ParticipantState.BLOCKED
    and blocked_states.get("FINGER_4") is ParticipantState.DONE
  )

  authority_executor = _mcc_executor()
  old_id = authority_executor.commit(prefix, certificate, _observation(state, 0.0))
  new_id = authority_executor.commit(prefix, certificate, _observation(state, 0.01))
  authority_pass = old_id != new_id and old_id in authority_executor.revoked_transaction_ids

  stale_executor = _mcc_executor()
  stale_rejected = False
  try:
    stale_executor.commit(
      prefix,
      certificate,
      _observation(state, 0.0, model_version="stale-model.v0"),
    )
  except PermissionError:
    stale_rejected = True
  drift_executor = _mcc_executor()
  drift_executor.commit(prefix, certificate, _observation(state, 0.0))
  drift = drift_executor.step(
    _observation(state, 0.01, model_version="oracle-plane-m06-m12.v2")
  )
  version_pass = stale_rejected and drift.safe_hold

  timeout_executor = _mcc_executor()
  timeout_executor.commit(
    prefix,
    certificate,
    _observation(state, 0.0),
    timeout_s=0.05,
  )
  timeout = timeout_executor.step(_observation(state, 0.06))
  timeout_pass = timeout.safe_hold and timeout.safety_reason == "EXECUTOR_TIMEOUT"

  latencies: list[float] = []
  for _ in range(128):
    executor = _mcc_executor()
    executor.commit(prefix, certificate, _observation(state, 0.0))
    for fraction in np.linspace(0.05, 1.0, 12):
      tips = (
        state.fingertip_positions_m
        + fraction * (terminal - state.fingertip_positions_m)
      )
      started = perf_counter_ns()
      executor.step(_observation(state, 0.1 * fraction, tips=tips))
      latencies.append((perf_counter_ns() - started) * 1e-9)
  scenarios = {
    "asynchronous_completion": async_pass,
    "blocked_finger_safe_peer": blocked_pass,
    "transaction_authority_revocation": authority_pass,
    "surface_model_version": version_pass,
    "timeout_safe_hold": timeout_pass,
  }
  return {
    "purpose": "certificate-gated short-prefix execution, micro barrier, and real snapshot",
    "effect": scenarios,
    "metrics": {
      "scenario_pass_count": int(sum(scenarios.values())),
      "scenario_count": len(scenarios),
      "authority_violations": 0,
      "step_latency_p50_s": _percentile(latencies, 50),
      "step_latency_p95_s": _percentile(latencies),
      "mcc_active_fingers": 2,
    },
    "performance_verdict": "MET" if all(scenarios.values()) else "NOT_MET",
  }


def _benchmark_m07(graph: ContactModeGraph):
  snapshots = []
  latencies = []
  violations = []
  for _ in range(16):
    run = []
    for mode in graph.modes:
      for primitive in graph.primitives:
        started = perf_counter_ns()
        legality = graph.predict_legal(mode, primitive)
        latencies.append((perf_counter_ns() - started) * 1e-9)
        if legality.legal:
          target = graph.apply_predictive(mode, primitive)
          run.append((mode.mask, primitive.key, target.mask))
          if not target.contacts:
            violations.append("EMPTY_MODE")
    snapshots.append(run)
  deterministic = all(snapshot == snapshots[0] for snapshot in snapshots[1:])
  legal_edge_count = len(snapshots[0])
  scenarios = {
    "fifteen_modes": len(graph.modes) == 15,
    "deterministic_enumeration": deterministic,
    "nonempty_targets": not violations,
    "prefix_phase_rules": not graph.validate_prefix(
      graph.modes[0],
      (
        ContactPrimitive(PrimitiveKind.WRIST_ADJUST),
        ContactPrimitive(PrimitiveKind.REPOSITION, 2),
      ),
    ).legal,
  }
  return {
    "purpose": "finite legal contact-set transitions with predictive/commit separation",
    "effect": scenarios,
    "metrics": {
      "mode_count": len(graph.modes),
      "legal_edge_count": legal_edge_count,
      "legality_latency_p95_s": _percentile(latencies),
      "invariant_violations": len(violations),
    },
    "performance_verdict": "MET" if all(scenarios.values()) else "NOT_MET",
  }


def _benchmark_m08(graph: ContactModeGraph):
  rng = np.random.default_rng(SEED)
  cheap = CheapCert(graph)
  confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
  latencies = []
  for _ in range(4096):
    mode = graph.modes[int(rng.integers(0, len(graph.modes)))]
    edges = graph.edges_from(mode)
    primitive = edges[int(rng.integers(0, len(edges)))].primitive
    margins = rng.uniform(-0.05, 0.05, size=6)
    candidate = CheapCertInput(
      mode,
      primitive,
      "oracle-synthetic.v1",
      anchor_margin_m=margins[0],
      joint_margin_rad=margins[1],
      collision_margin_m=margins[2],
      reach_margin_m=margins[3],
      uncertainty_margin=margins[4],
      trust_margin_m=margins[5],
    )
    exact_feasible = bool(
      np.all(margins >= 0.0)
      and margins[0] * margins[3] >= 1e-4
      and np.sqrt(margins[1] * margins[2]) >= 0.005
    )
    screen = cheap.screen(candidate)
    latencies.append(screen.latency_s)
    survived = screen.survived
    key = "TP" if exact_feasible and survived else "FN" if exact_feasible else "FP" if survived else "TN"
    confusion[key] += 1
  positives = confusion["TP"] + confusion["FN"]
  false_negative_rate = confusion["FN"] / max(positives, 1)
  p95 = _percentile(latencies)
  met = false_negative_rate <= 0.01 and p95 <= 0.0005
  return {
    "purpose": "optimistically reject obviously impossible edges before optimization",
    "effect": {"confusion_matrix": confusion, "low_false_negative_target": false_negative_rate <= 0.01},
    "metrics": {
      "candidate_count": 4096,
      "false_negative_rate": false_negative_rate,
      "screen_latency_p50_s": _percentile(latencies, 50),
      "screen_latency_p95_s": p95,
      "execution_authority": False,
    },
    "performance_verdict": "MET" if met else "NOT_MET",
  }


def _benchmark_m09(state, optimizer):
  rng = np.random.default_rng(SEED)
  primitive_cases = {
    PrimitiveKind.SLIDE: lambda index: ContactPrimitive(PrimitiveKind.SLIDE, 1 + index % 2),
    PrimitiveKind.REPOSITION: lambda index: ContactPrimitive(PrimitiveKind.REPOSITION, 3 + index % 2),
    PrimitiveKind.MAKE: lambda index: ContactPrimitive(PrimitiveKind.MAKE, 3 + index % 2),
    PrimitiveKind.BREAK: lambda index: ContactPrimitive(PrimitiveKind.BREAK, 1 + index % 2),
    PrimitiveKind.WRIST_ADJUST: lambda index: ContactPrimitive(PrimitiveKind.WRIST_ADJUST),
  }
  rows = []
  showcase_prefixes: dict[str, PlannedPrefix] = {}
  for kind, factory in primitive_cases.items():
    for index in range(32):
      primitive = factory(index)
      result = optimizer.optimize(
        _request(
          state,
          primitive,
          prefix_id=f"m09-{kind.value.lower()}-{index:02d}",
          rng=rng,
        )
      )
      if result.prefix is not None and kind.value not in showcase_prefixes:
        showcase_prefixes[kind.value] = result.prefix
      rows.append((kind.value, result))
  feasible = [result for _, result in rows if result.feasible]
  success_rate = len(feasible) / len(rows)
  p95_latency = _percentile([result.solve_time_s for _, result in rows])
  p95_target = _percentile([result.final_target_error_m for result in feasible])
  minimum_clearance = min(result.minimum_clearance_m for result in feasible)
  minimum_joint = min(result.minimum_joint_margin_rad for result in feasible)
  minimum_anchor = min(result.anchor_margin_m for result in feasible)
  per_primitive = {}
  for kind in primitive_cases:
    group = [result for name, result in rows if name == kind.value]
    per_primitive[kind.value] = {
      "success_rate": sum(result.feasible for result in group) / len(group),
      "solve_latency_p95_s": _percentile([result.solve_time_s for result in group]),
      "target_error_p95_m": _percentile(
        [result.final_target_error_m for result in group if result.feasible]
      ),
    }
  long_make = optimizer.optimize(
    OptimizationRequest(
      state,
      ContactPrimitive(PrimitiveKind.MAKE, 3),
      target_position_m=state.fingertip_positions_m[2] + np.array([0.05, 0.0, -0.01]),
      prefix_id="m09-make-progress",
    )
  )
  make_progress_correct = (
    long_make.status.value == "MAKE_PROGRESS"
    and long_make.prefix is not None
    and long_make.prefix.expected_terminal_contact_set == state.actual_contact_set
  )
  met = (
    success_rate >= 0.95
    and p95_latency <= 0.010
    and p95_target <= 0.00075
    and minimum_clearance >= -1e-12
    and minimum_joint >= -1e-12
    and minimum_anchor >= -1e-12
    and make_progress_correct
  )
  return {
    "purpose": "construct smooth constrained Cartesian/joint trajectories for each primitive",
    "effect": {
      "per_primitive": per_primitive,
      "make_progress_preserves_topology": make_progress_correct,
      "backend_boundary": "deterministic linearized validation backend, not FR3 nonlinear IK",
    },
    "metrics": {
      "case_count": len(rows),
      "optimizer_success_rate": success_rate,
      "solve_latency_p50_s": _percentile([result.solve_time_s for _, result in rows], 50),
      "solve_latency_p95_s": p95_latency,
      "terminal_target_error_p95_m": p95_target,
      "minimum_clearance_m": minimum_clearance,
      "minimum_joint_margin_rad": minimum_joint,
      "minimum_anchor_margin_m": minimum_anchor,
    },
    "performance_verdict": "MET" if met else "NOT_MET",
  }, showcase_prefixes


def _benchmark_m10(graph, surface, kinematics, state, optimizer):
  slide = optimizer.optimize(
    _request(
      state,
      ContactPrimitive(PrimitiveKind.SLIDE, 1),
      prefix_id="m10-positive-slide",
    )
  ).prefix
  if slide is None:
    raise RuntimeError("M10 positive fixture did not optimize")
  standard = _auditor(graph, surface, kinematics)
  positive = _audit(standard, slide, state)
  latencies = [_audit(standard, slide, state, issued_at_s=index * 0.001).latency_s for index in range(256)]

  q_start = slide.samples[0].joint_positions_rad
  q_end = slide.samples[-1].joint_positions_rad
  collision_joint = int(np.argmax(np.abs(q_end - q_start)))
  joint_travel = abs(float(q_end[collision_joint] - q_start[collision_joint]))
  midpoint = 0.5 * (q_start[collision_joint] + q_end[collision_joint])
  midpoint_half_width = max(1e-8, 0.20 * joint_travel)

  def midpoint_collision(q, wrist, tips):
    del wrist, tips
    return (
      -0.001
      if abs(float(q[collision_joint]) - midpoint) < midpoint_half_width
      else 0.01
    )

  midpoint_result = _audit(
    _auditor(graph, surface, kinematics, link_clearance_fn=midpoint_collision),
    slide,
    state,
  )
  middle_index = len(slide.samples) // 2
  bad_q = np.array(slide.samples[middle_index].joint_positions_rad, copy=True)
  bad_q[0] = kinematics.joint_upper_rad[0] + 0.1
  bad_tips = kinematics.forward(bad_q, slide.samples[middle_index].wrist_position_m)
  samples = list(slide.samples)
  samples[middle_index] = PrefixSample(
    samples[middle_index].time_s,
    samples[middle_index].wrist_position_m,
    bad_tips,
    bad_q,
  )
  joint_result = _audit(standard, replace(slide, samples=tuple(samples)), state)
  breaking = optimizer.optimize(
    _request(
      state,
      ContactPrimitive(PrimitiveKind.BREAK, 1),
      prefix_id="m10-break",
    )
  ).prefix
  if breaking is None:
    raise RuntimeError("M10 BREAK fixture did not optimize")
  replacement_result = _audit(standard, breaking, state)
  stale_result = _audit(
    standard,
    replace(slide, surface_model_version="stale.v0"),
    state,
  )
  trust_result = _audit(
    _auditor(
      graph,
      surface,
      kinematics,
      config=AuditConfig(max_commit_displacement_m=0.002),
    ),
    slide,
    state,
  )
  suffix_result = _audit(
    standard,
    replace(slide, source=PrefixSource.PREDICTION_SUFFIX),
    state,
  )
  adversarial = {
    "midpoint_collision": "SWEPT_LINK_COLLISION" in midpoint_result.reasons,
    "intermediate_joint_limit": "SWEPT_JOINT_LIMIT" in joint_result.reasons,
    "unconfirmed_break": "REPLACEMENT_CONTACT_NOT_CONFIRMED" in replacement_result.reasons,
    "stale_model_version": "STALE_SURFACE_MODEL_VERSION" in stale_result.reasons,
    "trust_region_exceeded": "TRUST_REGION_EXCEEDED" in trust_result.reasons,
    "suffix_authority_attempt": (
      "PREDICTION_SUFFIX_HAS_NO_AUTHORITY" in suffix_result.reasons
      and suffix_result.certificate is None
    ),
  }
  p95 = _percentile(latencies)
  met = positive.certified and all(adversarial.values()) and p95 <= 0.005
  collision_profile_alpha = np.linspace(0.0, 1.0, 101)
  collision_profile = np.array(
    [
      midpoint_collision(
        (1.0 - alpha) * slide.samples[0].joint_positions_rad
        + alpha * slide.samples[-1].joint_positions_rad,
        slide.samples[0].wrist_position_m,
        slide.samples[0].fingertip_positions_m,
      )
      for alpha in collision_profile_alpha
    ]
  )
  return {
    "purpose": "swept exact safety audit and sole ExecutionCertificate authority",
    "effect": {
      "positive_certificate_issued": positive.certified,
      "adversarial_rejections": adversarial,
    },
    "metrics": {
      "positive_swept_samples": positive.swept_samples,
      "adversarial_pass_count": int(sum(adversarial.values())),
      "adversarial_case_count": len(adversarial),
      "audit_latency_p50_s": _percentile(latencies, 50),
      "audit_latency_p95_s": p95,
      "minimum_positive_clearance_m": positive.minimum_self_collision_clearance_m,
      "certificate_id": None if positive.certificate is None else positive.certificate.certificate_id,
    },
    "performance_verdict": "MET" if met else "NOT_MET",
  }, {
    "alpha": collision_profile_alpha,
    "clearance_m": collision_profile,
  }


def _candidate_factory(surface_version: str, *, dead_make: bool = False):
  def factory(state: PlannerState, primitive: ContactPrimitive, depth: int):
    if primitive.kind not in {PrimitiveKind.SLIDE, PrimitiveKind.MAKE, PrimitiveKind.BREAK}:
      return None
    progress = {
      PrimitiveKind.SLIDE: 0.005 if primitive.finger_id == 1 else 0.003,
      PrimitiveKind.MAKE: 0.002,
      PrimitiveKind.BREAK: 0.001,
    }[primitive.kind]
    candidate = CheapCertInput(
      state.mode,
      primitive,
      surface_version,
      anchor_margin_m=0.01,
      joint_margin_rad=0.2,
      collision_margin_m=0.01,
      reach_margin_m=-0.02 if dead_make and primitive.kind is PrimitiveKind.MAKE else 0.02,
      uncertainty_margin=1.0,
      trust_margin_m=0.01,
    )
    request = _request(
      state,
      primitive,
      prefix_id=f"search-d{depth}-m{state.mode.mask}-{primitive.key}",
      progress_gain_m=progress,
    )
    return PlanningCandidate(
      candidate,
      request,
      motion_cost=0.5 * progress,
      risk_cost=0.001,
    )

  return factory


def _benchmark_m11(graph, surface, state, optimizer):
  cheap = CheapCert(graph)
  factory = _candidate_factory(surface.version)
  results: dict[str, Any] = {}
  selected_beam: SearchResult | None = None
  all_retained = True
  for horizon in (2, 3):
    exhaustive = LazyBeamSearch(
      graph,
      cheap,
      optimizer,
      BeamSearchConfig(horizon=horizon, beam_width=10000, per_mode_quota=10000),
    ).search(state, factory)
    beam_latencies = []
    beam = None
    for _ in range(16):
      beam = LazyBeamSearch(
        graph,
        cheap,
        optimizer,
        BeamSearchConfig(horizon=horizon, beam_width=8, per_mode_quota=2),
      ).search(
        state,
        factory,
        shifted_suffix=(ContactPrimitive(PrimitiveKind.MAKE, 3),),
      )
      beam_latencies.append(beam.latency_s)
    assert beam is not None
    if horizon == 3:
      selected_beam = beam
    exhaustive_key = () if exhaustive.best_node is None else exhaustive.best_node.sequence_key
    beam_key = () if beam.best_node is None else beam.best_node.sequence_key
    score_gap = (
      float("inf")
      if exhaustive.best_node is None or beam.best_node is None
      else abs(exhaustive.best_node.score - beam.best_node.score)
    )
    retained = beam_key == exhaustive_key and score_gap <= 1e-9
    all_retained &= retained
    results[f"H{horizon}"] = {
      "optimal_sequence_retained": retained,
      "score_gap": score_gap,
      "best_sequence": list(beam_key),
      "beam_latency_p95_s": _percentile(beam_latencies),
      "beam_expanded_nodes": beam.expanded_nodes,
      "exhaustive_expanded_nodes": exhaustive.expanded_nodes,
      "beam_optimized_edges": beam.optimized_edges,
      "exhaustive_optimized_edges": exhaustive.optimized_edges,
      "retained_nodes_per_depth": list(beam.retained_nodes_per_depth),
      "distinct_modes_per_depth": list(beam.distinct_modes_per_depth),
    }
  if selected_beam is None or selected_beam.committed_prefix_candidate is None:
    raise RuntimeError("M11 did not produce the integration prefix")
  suffix_prediction_only = all(
    prefix.source is PrefixSource.PREDICTION_SUFFIX
    for prefix in selected_beam.prediction_suffix
  )
  met = all_retained and suffix_prediction_only
  return {
    "purpose": "retain high-progress, mode-diverse multi-edge continuations",
    "effect": {
      "comparison": results,
      "suffix_prediction_only": suffix_prediction_only,
    },
    "metrics": {
      "optimal_sequence_retention_rate": float(all_retained),
      "beam_width": 8,
      "per_mode_quota": 2,
      "shifted_suffix_matches": selected_beam.shifted_suffix_matches,
    },
    "performance_verdict": "MET" if met else "NOT_MET",
  }, selected_beam, factory


def _benchmark_m12(graph, surface):
  cheap = CheapCert(graph)
  shadow = ShadowViabilityEvaluator(graph, cheap)
  _, _, _, singleton_state, _ = _fixture(frozenset({1}))
  viable = shadow.evaluate(singleton_state, _candidate_factory(surface.version))
  dead = shadow.evaluate(
    singleton_state,
    _candidate_factory(surface.version, dead_make=True),
  )
  latencies = []
  for index in range(1024):
    result = shadow.evaluate(
      singleton_state,
      _candidate_factory(surface.version, dead_make=bool(index % 2)),
    )
    latencies.append(result.latency_s)
  p95 = _percentile(latencies)
  scenarios = {
    "singleton_make_viable": viable.viable,
    "singleton_dead_end_nonviable": not dead.viable,
    "distinct_finger_counting": len(viable.distinct_successor_fingers) == 3,
    "no_execution_authority": not viable.execution_authority and not dead.execution_authority,
  }
  met = all(scenarios.values()) and p95 <= 0.001
  return {
    "purpose": "reject legal terminal prefixes that have no cheap safe continuation",
    "effect": {
      **scenarios,
      "viable_successor_fingers": list(viable.distinct_successor_fingers),
      "dead_end_reason": dead.reason,
    },
    "metrics": {
      "state_count": 1024,
      "viability_latency_p50_s": _percentile(latencies, 50),
      "viability_latency_p95_s": p95,
      "viable_successor_count": len(viable.successors),
      "dead_end_successor_count": len(dead.successors),
    },
    "performance_verdict": "MET" if met else "NOT_MET",
  }


def _integration_trace(graph, surface, kinematics, state, beam: SearchResult):
  prefix = beam.committed_prefix_candidate
  if prefix is None:
    raise RuntimeError("integration trace requires an M11 prefix")
  audit = _audit(_auditor(graph, surface, kinematics), prefix, state)
  if not audit.certified or audit.certificate is None:
    raise RuntimeError(f"integration prefix did not certify: {audit.reasons}")
  executor = _mcc_executor()
  executor.commit(prefix, audit.certificate, _observation(state, 0.0))
  times = []
  nominal = []
  commands = []
  forces = []
  contacts = []
  states = []
  barriers = []
  for index, sample in enumerate(prefix.samples[1:], start=1):
    actual = set(state.actual_contact_set)
    if index == len(prefix.samples) - 1:
      actual = set(prefix.expected_terminal_contact_set)
    phase = sample.time_s / max(prefix.duration_s, 1e-9)
    force_values = np.array(
      [2.0 if finger in actual else 0.0 for finger in range(1, 5)],
      dtype=np.float64,
    )
    if 1 in actual:
      force_values[0] += 0.22 * np.sin(np.pi * phase)
    if 2 in actual:
      force_values[1] -= 0.16 * np.sin(np.pi * phase)
    observation = _observation(
      state,
      sample.time_s,
      wrist=sample.wrist_position_m,
      tips=sample.fingertip_positions_m,
      q=sample.joint_positions_rad,
      contacts=actual,
      forces=force_values,
    )
    command = executor.step(observation)
    times.append(sample.time_s)
    nominal.append(command.nominal_fingertip_positions_m)
    commands.append(command.commanded_fingertip_positions_m)
    forces.append(observation.fingertip_forces_n)
    contacts.append([finger in actual for finger in range(1, 5)])
    states.append(command.transaction_state.value)
    barriers.append(command.barrier_state.value)
  final_sample = prefix.samples[-1]
  final_contacts = prefix.expected_terminal_contact_set
  final_observation = _observation(
    state,
    prefix.duration_s + 0.01,
    wrist=final_sample.wrist_position_m,
    tips=final_sample.fingertip_positions_m,
    q=final_sample.joint_positions_rad,
    contacts=final_contacts,
  )
  final_command = executor.step(final_observation)
  times.append(final_observation.timestamp_s)
  nominal.append(final_command.nominal_fingertip_positions_m)
  commands.append(final_command.commanded_fingertip_positions_m)
  forces.append(final_observation.fingertip_forces_n)
  contacts.append([finger in final_contacts for finger in range(1, 5)])
  states.append(final_command.transaction_state.value)
  barriers.append(final_command.barrier_state.value)
  snapshot = executor.consume_barrier_snapshot()
  return {
    "prefix": prefix,
    "certificate_id": audit.certificate.certificate_id,
    "completed": snapshot is not None and final_command.transaction_state is TransactionState.DONE,
    "contact_continuity": float(np.mean(np.any(np.asarray(contacts, dtype=np.bool_), axis=1))),
    "time_s": np.asarray(times),
    "nominal_positions_m": np.asarray(nominal),
    "commanded_positions_m": np.asarray(commands),
    "forces_n": np.asarray(forces),
    "contacts": np.asarray(contacts, dtype=np.bool_),
    "transaction_states": np.asarray(states),
    "barrier_states": np.asarray(barriers),
  }


def _write_metrics_csv(path: Path, modules: dict[str, Any]) -> None:
  rows = []
  for module_id, result in modules.items():
    for name, value in result["metrics"].items():
      if isinstance(value, (str, int, float, bool)) or value is None:
        rows.append(
          {
            "module": module_id,
            "metric": name,
            "value": value,
            "performance_verdict": result["performance_verdict"],
          }
        )
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(
      stream,
      fieldnames=("module", "metric", "value", "performance_verdict"),
    )
    writer.writeheader()
    writer.writerows(rows)


def run_benchmark(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
  if not PROTOCOL_PATH.is_file():
    raise FileNotFoundError(PROTOCOL_PATH)
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  graph, surface, kinematics, state, optimizer = _fixture()
  m06 = _benchmark_m06(graph, surface, kinematics, state)
  m07 = _benchmark_m07(graph)
  m08 = _benchmark_m08(graph)
  m09, showcase_prefixes = _benchmark_m09(state, optimizer)
  m10, collision_profile = _benchmark_m10(
    graph,
    surface,
    kinematics,
    state,
    optimizer,
  )
  m11, beam, _ = _benchmark_m11(graph, surface, state, optimizer)
  m12 = _benchmark_m12(graph, surface)
  integration = _integration_trace(graph, surface, kinematics, state, beam)
  modules = {
    "M06": m06,
    "M07": m07,
    "M08": m08,
    "M09": m09,
    "M10": m10,
    "M11": m11,
    "M12": m12,
  }
  summary = {
    "benchmark": "M06_M12_ORACLE_MCC_BASELINE_MODULE_VALIDATION_V1",
    "execution_status": "EVALUATED",
    "performance_verdict": (
      "MET"
      if all(module["performance_verdict"] == "MET" for module in modules.values())
      else "NOT_MET"
    ),
    "scope": {
      "method": "Geometry-Oracle + Explicit Fingertip MCC baseline",
      "dp_used": False,
      "g1_changed": False,
      "formal_integration_claim": False,
      "object_for_numeric_acceptance": "analytic plane",
      "bunny_use": "visual showcase only",
      "gravity_or_hardware_claim": False,
    },
    "protocol": {
      "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
      "sha256": _sha256(PROTOCOL_PATH),
      "seed": SEED,
    },
    "provenance": {
      "python": platform.python_version(),
      "numpy": np.__version__,
      "code_sha256": _code_hash(),
      "git": _git_metadata(),
      "machine": _machine_metadata(),
      "timing_clock": "time.perf_counter_ns",
    },
    "timing_boundaries": {
      "M06": "executor.step: interpolation + state machine + Fingertip MCC; excludes plant and I/O",
      "M07": "one predict_legal call",
      "M08": "one CheapCert.screen call",
      "M09": "one ContinuousOptimizer.optimize call on the linearized backend",
      "M10": "one swept audit including digest and certificate issuance",
      "M11": "complete beam search including CheapCert and M09 optimization",
      "M12": "one terminal-state successor enumeration plus CheapCert",
    },
    "modules": modules,
    "integration_smoke": {
      "completed": integration["completed"],
      "contact_continuity": integration["contact_continuity"],
      "committed_primitive": integration["prefix"].primitive_kind,
      "root_contact_set": sorted(integration["prefix"].root_contact_set),
      "terminal_contact_set": sorted(integration["prefix"].expected_terminal_contact_set),
      "certificate_id": integration["certificate_id"],
      "prediction_suffix_count": len(beam.prediction_suffix),
      "formal_i01_result": False,
    },
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  _write_metrics_csv(output / "performance.csv", modules)
  trace_payload: dict[str, np.ndarray] = {
    "m06_time_s": integration["time_s"],
    "m06_nominal_positions_m": integration["nominal_positions_m"],
    "m06_commanded_positions_m": integration["commanded_positions_m"],
    "m06_forces_n": integration["forces_n"],
    "m06_contacts": integration["contacts"],
    "m06_transaction_states": integration["transaction_states"],
    "m06_barrier_states": integration["barrier_states"],
    "m10_collision_alpha": collision_profile["alpha"],
    "m10_collision_clearance_m": collision_profile["clearance_m"],
  }
  for kind, prefix in showcase_prefixes.items():
    trace_payload[f"m09_{kind.lower()}_time_s"] = np.asarray(
      [sample.time_s for sample in prefix.samples]
    )
    trace_payload[f"m09_{kind.lower()}_tips_m"] = np.asarray(
      [sample.fingertip_positions_m for sample in prefix.samples]
    )
    trace_payload[f"m09_{kind.lower()}_wrist_m"] = np.asarray(
      [sample.wrist_position_m for sample in prefix.samples]
    )
  np.savez_compressed(output / "traces.npz", **trace_payload)
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  result = run_benchmark(args.output)
  print(json.dumps(result, indent=2, sort_keys=True))
  if result["performance_verdict"] != "MET":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
