from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from Module.module_1_oracle_surface_model import OracleSurfaceModel, Plane
from Module.module_2_fingertip_mcc import (
  FingertipMCC,
  FullRobotFingertipMCC,
  MCCConfig,
)
from Module.module_6_prefix_executor import (
  BarrierState,
  ExecutionCertificate,
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
  ContactMode,
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
  OptimizationStatus,
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
)
from Module.module_12_shadow_viability import ShadowStatus, ShadowViabilityEvaluator


def _fixture(contact_set: frozenset[int] = frozenset({1, 2})):
  tips = np.array(
    [
      [-0.045, -0.025, 0.000],
      [0.045, -0.025, 0.000],
      [-0.045, 0.030, 0.010],
      [0.045, 0.030, 0.010],
    ],
    dtype=np.float64,
  )
  for finger in contact_set:
    tips[finger - 1, 2] = 0.0
  surface = OracleSurfaceModel(
    Plane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0]),
    version="oracle-plane.v1",
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
) -> OptimizationRequest:
  if primitive.kind is PrimitiveKind.WRIST_ADJUST:
    return OptimizationRequest(
      state,
      primitive,
      target_wrist_position_m=state.wrist_position_m + np.array([0.0, 0.004, 0.0]),
      prefix_id=prefix_id,
      progress_gain_m=0.004,
    )
  assert primitive.finger_id is not None
  target = np.array(state.fingertip_positions_m[primitive.finger_id - 1], copy=True)
  if primitive.kind is PrimitiveKind.SLIDE:
    target += np.array([0.006, 0.0, 0.0])
    target[2] = 0.0
  elif primitive.kind is PrimitiveKind.REPOSITION:
    target += np.array([0.004, 0.0, 0.003])
  elif primitive.kind is PrimitiveKind.MAKE:
    target[2] = 0.0
  elif primitive.kind is PrimitiveKind.BREAK:
    target += np.array([0.0, 0.0, 0.006])
  return OptimizationRequest(
    state,
    primitive,
    target_position_m=target,
    prefix_id=prefix_id,
    progress_gain_m=0.004,
  )


def _audit(
  graph: ContactModeGraph,
  surface: OracleSurfaceModel,
  kinematics: LinearizedHandKinematics,
  prefix: PlannedPrefix,
  state: PlannerState,
  *,
  confirmations: dict[int, float] | None = None,
  link_clearance_fn=None,
):
  exact_link_clearance = link_clearance_fn or (lambda q, wrist, tips: 0.02)
  auditor = ExactPrefixAuditor(
    graph,
    AuditEnvironment(surface, kinematics, link_clearance_fn=exact_link_clearance),
    AuditConfig(subdivisions_per_segment=9),
  )
  return auditor.audit(
    AuditRequest(
      prefix,
      state,
      CommitContext(
        state.actual_contact_set,
        replacement_confirmation_s=confirmations or {},
      ),
      issued_at_s=0.0,
    )
  )


def _observation(
  state: PlannerState,
  timestamp_s: float,
  *,
  tips=None,
  joints=None,
  contacts=None,
  blocked=None,
  model_version=None,
) -> ExecutorObservation:
  actual = state.actual_contact_set if contacts is None else frozenset(contacts)
  forces = np.array([2.0 if finger in actual else 0.0 for finger in range(1, 5)])
  return ExecutorObservation(
    timestamp_s=timestamp_s,
    surface_model_version=model_version or state.surface_model_version,
    wrist_position_m=state.wrist_position_m,
    fingertip_positions_m=state.fingertip_positions_m if tips is None else tips,
    joint_positions_rad=(
      state.joint_positions_rad if joints is None else joints
    ),
    fingertip_forces_n=forces,
    outward_normals=np.tile([0.0, 0.0, 1.0], (4, 1)),
    actual_contact_set=actual,
    blocked_fingers=blocked or {},
  )


class M07ContactModeGraphTest(unittest.TestCase):
  def test_all_modes_and_edges_are_deterministic_and_nonempty(self) -> None:
    graph = ContactModeGraph()
    self.assertEqual(len(graph.modes), 15)
    first = [(edge.key, edge.target.mask) for mode in graph.modes for edge in graph.edges_from(mode)]
    second = [(edge.key, edge.target.mask) for mode in graph.modes for edge in graph.edges_from(mode)]
    self.assertEqual(first, second)
    self.assertTrue(all(target > 0 for _, target in first))

    for mode in graph.modes:
      for edge in graph.edges_from(mode):
        if edge.primitive.kind is PrimitiveKind.MAKE:
          self.assertNotIn(edge.primitive.finger_id, mode.contacts)
        if edge.primitive.kind in {PrimitiveKind.BREAK, PrimitiveKind.SLIDE}:
          self.assertIn(edge.primitive.finger_id, mode.contacts)
        if edge.primitive.kind is PrimitiveKind.REPOSITION:
          self.assertNotIn(edge.primitive.finger_id, mode.contacts)

  def test_commit_break_requires_measured_replacement_confirmation(self) -> None:
    graph = ContactModeGraph()
    mode = ContactMode(frozenset({1, 2}))
    action = ContactPrimitive(PrimitiveKind.BREAK, 1)
    denied = graph.commit_legal(mode, action, CommitContext(mode.contacts))
    self.assertFalse(denied.legal)
    allowed = graph.commit_legal(
      mode,
      action,
      CommitContext(mode.contacts, replacement_confirmation_s={2: 0.05}),
    )
    self.assertTrue(allowed.legal)
    mixed = graph.validate_prefix(
      ContactMode(frozenset({1})),
      (
        ContactPrimitive(PrimitiveKind.MAKE, 2),
        ContactPrimitive(PrimitiveKind.BREAK, 1),
      ),
    )
    self.assertFalse(mixed.legal)
    self.assertEqual(mixed.reason, "MULTIPLE_TOPOLOGY_CHANGES")


class M08CheapCertTest(unittest.TestCase):
  def test_optimistic_screen_has_zero_false_negatives_on_seeded_reference(self) -> None:
    graph = ContactModeGraph()
    cheap = CheapCert(graph)
    rng = np.random.default_rng(7)
    confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for _ in range(4096):
      mode = graph.modes[int(rng.integers(0, len(graph.modes)))]
      edges = graph.edges_from(mode)
      primitive = edges[int(rng.integers(0, len(edges)))].primitive
      margins = rng.uniform(-0.05, 0.05, size=6)
      candidate = CheapCertInput(
        mode,
        primitive,
        "oracle.v1",
        anchor_margin_m=margins[0],
        joint_margin_rad=margins[1],
        collision_margin_m=margins[2],
        reach_margin_m=margins[3],
        uncertainty_margin=margins[4],
        trust_margin_m=margins[5],
      )
      exact = bool(
        np.all(margins >= 0.0)
        and margins[0] * margins[3] >= 1e-4
        and np.sqrt(margins[1] * margins[2]) >= 0.005
      )
      survived = cheap.screen(candidate).survived
      confusion[
        "TP" if exact and survived else "FN" if exact else "FP" if survived else "TN"
      ] += 1
    self.assertGreater(confusion["TP"], 0)
    self.assertEqual(confusion["FN"], 0)
    self.assertGreater(confusion["FP"], 0)


class M09ContinuousOptimizeTest(unittest.TestCase):
  def test_all_five_primitives_construct_safe_prefixes(self) -> None:
    graph, _, _, state, optimizer = _fixture()
    primitives = (
      ContactPrimitive(PrimitiveKind.SLIDE, 1),
      ContactPrimitive(PrimitiveKind.REPOSITION, 3),
      ContactPrimitive(PrimitiveKind.MAKE, 3),
      ContactPrimitive(PrimitiveKind.BREAK, 1),
      ContactPrimitive(PrimitiveKind.WRIST_ADJUST),
    )
    for index, primitive in enumerate(primitives):
      with self.subTest(primitive=primitive.key):
        result = optimizer.optimize(_request(state, primitive, prefix_id=f"p{index}"))
        self.assertTrue(result.feasible, result.reasons)
        self.assertIsNotNone(result.prefix)
        self.assertLessEqual(result.final_target_error_m, 0.00075)
        self.assertGreaterEqual(result.anchor_margin_m, -1e-12)
        self.assertGreaterEqual(result.minimum_clearance_m, -1e-12)
        self.assertLessEqual(result.trust_region_use, 1.0 + 1e-10)

  def test_long_make_is_progress_only_and_does_not_change_topology(self) -> None:
    _, _, _, state, optimizer = _fixture()
    primitive = ContactPrimitive(PrimitiveKind.MAKE, 3)
    target = state.fingertip_positions_m[2] + np.array([0.05, 0.0, -0.01])
    result = optimizer.optimize(
      OptimizationRequest(
        state,
        primitive,
        target_position_m=target,
        prefix_id="make-progress",
      )
    )
    self.assertEqual(result.status, OptimizationStatus.MAKE_PROGRESS)
    assert result.prefix is not None
    self.assertEqual(result.prefix.expected_terminal_contact_set, state.actual_contact_set)
    self.assertEqual(result.prefix.topology_change_count, 0)

  def test_explicit_physical_pad_center_target_is_not_reprojected(self) -> None:
    _, _, _, state, optimizer = _fixture()
    primitive = ContactPrimitive(PrimitiveKind.MAKE, 3)
    pad_center = np.array(state.fingertip_positions_m[2], copy=True)
    pad_center[2] = 0.004
    result = optimizer.optimize(
      OptimizationRequest(
        state,
        primitive,
        target_position_m=pad_center,
        prefix_id="physical-pad-center",
        metadata={"physical_pad_center_target": 1.0},
      )
    )
    self.assertTrue(result.feasible, result.reasons)
    assert result.prefix is not None
    self.assertAlmostEqual(
      result.prefix.samples[-1].fingertip_positions_m[2, 2],
      0.004,
      places=6,
    )
    self.assertEqual(
      result.prefix.metadata["physical_pad_center_target"],
      1.0,
    )


class M10ExactPrefixAuditTest(unittest.TestCase):
  def test_only_positive_swept_prefix_receives_certificate(self) -> None:
    graph, surface, kinematics, state, optimizer = _fixture()
    optimized = optimizer.optimize(
      _request(state, ContactPrimitive(PrimitiveKind.SLIDE, 1), prefix_id="slide")
    )
    assert optimized.prefix is not None
    result = _audit(graph, surface, kinematics, optimized.prefix, state)
    self.assertTrue(result.certified, result.reasons)
    self.assertIsNotNone(result.certificate)
    self.assertGreater(result.swept_samples, len(optimized.prefix.samples))
    assert result.certificate is not None
    with self.assertRaises(AttributeError):
      result.certificate.prefix_digest = "tampered"
    with self.assertRaises(PermissionError):
      ExecutionCertificate(
        certificate_id="forged",
        prefix_id=optimized.prefix.prefix_id,
        prefix_digest_value="forged",
        surface_model_version=state.surface_model_version,
        root_contact_set=state.actual_contact_set,
        audit_version="forged",
        issued_at_s=0.0,
      )

  def test_six_adversarial_authority_and_sweep_cases_are_rejected(self) -> None:
    graph, surface, kinematics, state, optimizer = _fixture()
    slide = optimizer.optimize(
      _request(state, ContactPrimitive(PrimitiveKind.SLIDE, 1), prefix_id="slide-adv")
    ).prefix
    assert slide is not None

    q_start = slide.samples[0].joint_positions_rad
    q_terminal = slide.samples[-1].joint_positions_rad
    collision_joint = int(np.argmax(np.abs(q_terminal - q_start)))
    joint_travel = abs(float(q_terminal[collision_joint] - q_start[collision_joint]))
    midpoint = 0.5 * (q_start[collision_joint] + q_terminal[collision_joint])
    midpoint_half_width = max(1e-8, 0.20 * joint_travel)

    def midpoint_collision(q, wrist, tips):
      del wrist, tips
      return (
        -0.001
        if abs(float(q[collision_joint]) - midpoint) < midpoint_half_width
        else 0.01
      )

    collision = _audit(
      graph,
      surface,
      kinematics,
      slide,
      state,
      link_clearance_fn=midpoint_collision,
    )
    self.assertIn("SWEPT_LINK_COLLISION", collision.reasons)

    middle_q = np.array(slide.samples[len(slide.samples) // 2].joint_positions_rad, copy=True)
    middle_q[0] = kinematics.joint_upper_rad[0] + 0.1
    middle_tips = kinematics.forward(
      middle_q,
      slide.samples[len(slide.samples) // 2].wrist_position_m,
    )
    samples = list(slide.samples)
    middle = len(samples) // 2
    samples[middle] = PrefixSample(
      samples[middle].time_s,
      samples[middle].wrist_position_m,
      middle_tips,
      middle_q,
    )
    joint = _audit(graph, surface, kinematics, replace(slide, samples=tuple(samples)), state)
    self.assertIn("SWEPT_JOINT_LIMIT", joint.reasons)

    breaking = optimizer.optimize(
      _request(state, ContactPrimitive(PrimitiveKind.BREAK, 1), prefix_id="break-adv")
    ).prefix
    assert breaking is not None
    no_replacement = _audit(graph, surface, kinematics, breaking, state)
    self.assertIn("REPLACEMENT_CONTACT_NOT_CONFIRMED", no_replacement.reasons)

    stale = _audit(
      graph,
      surface,
      kinematics,
      replace(slide, surface_model_version="stale.v0"),
      state,
    )
    self.assertIn("STALE_SURFACE_MODEL_VERSION", stale.reasons)

    trust_auditor = ExactPrefixAuditor(
      graph,
      AuditEnvironment(surface, kinematics, link_clearance_fn=lambda q, wrist, tips: 0.02),
      AuditConfig(max_commit_displacement_m=0.002),
    )
    trust = trust_auditor.audit(
      AuditRequest(slide, state, CommitContext(state.actual_contact_set), 0.0)
    )
    self.assertIn("TRUST_REGION_EXCEEDED", trust.reasons)

    suffix = _audit(
      graph,
      surface,
      kinematics,
      replace(slide, source=PrefixSource.PREDICTION_SUFFIX),
      state,
    )
    self.assertIn("PREDICTION_SUFFIX_HAS_NO_AUTHORITY", suffix.reasons)
    self.assertIsNone(suffix.certificate)


class M06TransactionalExecutorTest(unittest.TestCase):
  def _two_finger_prefix(self):
    graph, surface, kinematics, state, _ = _fixture()
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
      prefix_id="async-reposition",
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
    audit = _audit(graph, surface, kinematics, prefix, state)
    self.assertTrue(audit.certified, audit.reasons)
    assert audit.certificate is not None
    return state, prefix, audit.certificate

  def _executor(self):
    controllers = tuple(
      FingertipMCC(MCCConfig(dt_s=0.01)) for _ in range(4)
    )
    return TransactionalPrefixExecutor(
      ExecutorConfig(default_timeout_s=0.3),
      mcc_adapter=MCCBaselineAdapter(FullRobotFingertipMCC(controllers)),
    )

  def test_asynchronous_completion_waits_for_fresh_micro_barrier(self) -> None:
    state, prefix, certificate = self._two_finger_prefix()
    executor = self._executor()
    executor.commit(prefix, certificate, _observation(state, 0.0))
    terminal = prefix.samples[-1].fingertip_positions_m
    partial = np.array(state.fingertip_positions_m, copy=True)
    partial[2] = terminal[2]
    first = executor.step(_observation(state, 0.05, tips=partial))
    records = {record.participant: record.state for record in first.participants}
    self.assertEqual(records["FINGER_3"], ParticipantState.DONE)
    self.assertEqual(records["FINGER_4"], ParticipantState.RUNNING)
    self.assertEqual(first.mcc_active_mask.tolist(), [True, True, False, False])

    waiting = executor.step(_observation(state, 0.10, tips=terminal))
    self.assertEqual(waiting.barrier_state, BarrierState.WAITING_FOR_FRESH_OBSERVATION)
    duplicate = executor.step(_observation(state, 0.10, tips=terminal))
    self.assertEqual(duplicate.barrier_state, BarrierState.WAITING_FOR_FRESH_OBSERVATION)
    self.assertEqual(duplicate.transaction_state, TransactionState.RUNNING)
    closed = executor.step(_observation(state, 0.11, tips=terminal))
    self.assertEqual(closed.transaction_state, TransactionState.DONE)
    self.assertEqual(closed.barrier_state, BarrierState.CLOSED)
    snapshot = executor.consume_barrier_snapshot()
    self.assertIsNotNone(snapshot)
    assert snapshot is not None
    self.assertEqual(snapshot.actual_contact_set, state.actual_contact_set)

  def test_make_contact_must_persist_through_fresh_barrier(self) -> None:
    graph, surface, kinematics, state, optimizer = _fixture()
    optimized = optimizer.optimize(
      _request(
        state,
        ContactPrimitive(PrimitiveKind.MAKE, 3),
        prefix_id="persistent-make",
      )
    )
    self.assertTrue(optimized.feasible, optimized.reasons)
    assert optimized.prefix is not None
    audit = _audit(graph, surface, kinematics, optimized.prefix, state)
    self.assertTrue(audit.certified, audit.reasons)
    assert audit.certificate is not None
    executor = TransactionalPrefixExecutor(
      ExecutorConfig(
        default_timeout_s=0.5,
        make_contact_is_terminal=True,
      )
    )
    executor.commit(
      optimized.prefix,
      audit.certificate,
      _observation(state, 0.0),
    )
    touched = executor.step(
      _observation(state, 0.05, contacts={1, 2, 3})
    )
    self.assertEqual(
      touched.barrier_state,
      BarrierState.WAITING_FOR_FRESH_OBSERVATION,
    )
    dropped = executor.step(
      _observation(state, 0.06, contacts={1, 2})
    )
    self.assertEqual(dropped.barrier_state, BarrierState.OPEN)
    self.assertEqual(dropped.transaction_state, TransactionState.RUNNING)
    record = {item.participant: item.state for item in dropped.participants}
    self.assertEqual(record["FINGER_3"], ParticipantState.RUNNING)
    executor.step(_observation(state, 0.07, contacts={1, 2, 3}))
    closed = executor.step(
      _observation(state, 0.08, contacts={1, 2, 3})
    )
    self.assertEqual(closed.transaction_state, TransactionState.DONE)
    snapshot = executor.consume_barrier_snapshot()
    assert snapshot is not None
    self.assertIn(3, snapshot.actual_contact_set)

  def test_wrist_may_close_on_certified_joint_tracking_under_load(self) -> None:
    graph, surface, kinematics, state, optimizer = _fixture()
    optimized = optimizer.optimize(
      _request(
        state,
        ContactPrimitive(PrimitiveKind.WRIST_ADJUST),
        prefix_id="joint-tracked-wrist",
      )
    )
    self.assertTrue(optimized.feasible, optimized.reasons)
    assert optimized.prefix is not None
    audit = _audit(graph, surface, kinematics, optimized.prefix, state)
    self.assertTrue(audit.certified, audit.reasons)
    assert audit.certificate is not None
    executor = TransactionalPrefixExecutor(
      ExecutorConfig(
        default_timeout_s=0.5,
        wrist_completion_tolerance_m=0.0001,
        wrist_joint_completion_tolerance_rad=0.001,
        wrist_joint_dimension=len(state.joint_positions_rad),
      )
    )
    executor.commit(
      optimized.prefix,
      audit.certificate,
      _observation(state, 0.0),
    )
    terminal_q = optimized.prefix.samples[-1].joint_positions_rad
    waiting = executor.step(
      _observation(
        state,
        optimized.prefix.duration_s,
        # Deliberately keep the observed Cartesian wrist at its root.  The
        # certified joint terminal is nevertheless tracked under load.
        joints=terminal_q,
      )
    )
    self.assertEqual(
      waiting.barrier_state,
      BarrierState.WAITING_FOR_FRESH_OBSERVATION,
    )
    closed = executor.step(
      _observation(
        state,
        optimized.prefix.duration_s + 0.01,
        joints=terminal_q,
      )
    )
    self.assertEqual(closed.transaction_state, TransactionState.DONE)

  def test_blocked_finger_does_not_prevent_safe_peer_completion(self) -> None:
    state, prefix, certificate = self._two_finger_prefix()
    executor = self._executor()
    executor.commit(prefix, certificate, _observation(state, 0.0))
    executor.step(_observation(state, 0.02, blocked={3: "SUSPECTED_OBJECT_BLOCKAGE"}))
    terminal = prefix.samples[-1].fingertip_positions_m
    mixed = np.array(terminal, copy=True)
    mixed[2] = state.fingertip_positions_m[2]
    executor.step(_observation(state, 0.10, tips=mixed))
    final = executor.step(_observation(state, 0.11, tips=mixed))
    self.assertEqual(final.transaction_state, TransactionState.BLOCKED)
    records = {record.participant: record.state for record in final.participants}
    self.assertEqual(records["FINGER_3"], ParticipantState.BLOCKED)
    self.assertEqual(records["FINGER_4"], ParticipantState.DONE)

  def test_new_commit_revokes_old_and_timeout_or_version_drift_safe_holds(self) -> None:
    state, prefix, certificate = self._two_finger_prefix()
    executor = self._executor()
    first_id = executor.commit(prefix, certificate, _observation(state, 0.0))
    second_id = executor.commit(prefix, certificate, _observation(state, 0.01))
    self.assertNotEqual(first_id, second_id)
    self.assertIn(first_id, executor.revoked_transaction_ids)
    timeout = executor.step(_observation(state, 0.32))
    self.assertTrue(timeout.safe_hold)
    self.assertEqual(timeout.safety_reason, "EXECUTOR_TIMEOUT")

    other = self._executor()
    other.commit(prefix, certificate, _observation(state, 0.0))
    drift = other.step(_observation(state, 0.01, model_version="oracle-plane.v2"))
    self.assertTrue(drift.safe_hold)
    self.assertEqual(drift.safety_reason, "SURFACE_MODEL_VERSION_DRIFT")


def _candidate_factory(surface_version: str, *, force_dead_end: bool = False):
  def factory(state: PlannerState, primitive: ContactPrimitive, depth: int):
    if primitive.kind not in {PrimitiveKind.SLIDE, PrimitiveKind.MAKE, PrimitiveKind.BREAK}:
      return None
    if primitive.kind is PrimitiveKind.MAKE and force_dead_end:
      reach_margin = -0.02
    else:
      reach_margin = 0.02
    cheap_input = CheapCertInput(
      state.mode,
      primitive,
      surface_version,
      anchor_margin_m=0.01,
      joint_margin_rad=0.2,
      collision_margin_m=0.01,
      reach_margin_m=reach_margin,
      uncertainty_margin=1.0,
      trust_margin_m=0.01,
    )
    request = _request(
      state,
      primitive,
      prefix_id=f"d{depth}-m{state.mode.mask}-{primitive.key}",
    )
    progress = {
      PrimitiveKind.SLIDE: 0.005 if primitive.finger_id == 1 else 0.003,
      PrimitiveKind.MAKE: 0.002,
      PrimitiveKind.BREAK: 0.001,
    }[primitive.kind]
    request = replace(request, progress_gain_m=progress)
    return PlanningCandidate(
      cheap_input,
      request,
      motion_cost=0.5 * progress,
      risk_cost=0.001,
    )

  return factory


class M11LazyBeamSearchTest(unittest.TestCase):
  def test_beam_retains_exhaustive_best_and_suffix_has_no_authority(self) -> None:
    graph, surface, _, state, optimizer = _fixture()
    cheap = CheapCert(graph)
    factory = _candidate_factory(surface.version)
    for horizon in (2, 3):
      with self.subTest(horizon=horizon):
        exhaustive = LazyBeamSearch(
          graph,
          cheap,
          optimizer,
          BeamSearchConfig(horizon=horizon, beam_width=10000, per_mode_quota=10000),
        ).search(state, factory)
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
        self.assertTrue(exhaustive.found)
        self.assertTrue(beam.found)
        assert exhaustive.best_node is not None and beam.best_node is not None
        self.assertEqual(beam.best_node.sequence_key, exhaustive.best_node.sequence_key)
        self.assertAlmostEqual(beam.best_node.score, exhaustive.best_node.score, places=12)
        self.assertIsNotNone(beam.committed_prefix_candidate)
        self.assertTrue(
          all(prefix.source is PrefixSource.PREDICTION_SUFFIX for prefix in beam.prediction_suffix)
        )


class M12ShadowViabilityTest(unittest.TestCase):
  def test_singleton_requires_a_cheap_feasible_make(self) -> None:
    graph, surface, _, state, _ = _fixture(frozenset({1}))
    cheap = CheapCert(graph)
    shadow = ShadowViabilityEvaluator(graph, cheap)
    viable = shadow.evaluate(state, _candidate_factory(surface.version))
    self.assertEqual(viable.status, ShadowStatus.VIABLE)
    self.assertIn(2, viable.distinct_successor_fingers)
    self.assertTrue(all(successor.primitive.kind is PrimitiveKind.MAKE for successor in viable.successors))
    self.assertFalse(viable.execution_authority)

    dead = shadow.evaluate(
      state,
      _candidate_factory(surface.version, force_dead_end=True),
    )
    self.assertEqual(dead.status, ShadowStatus.NONVIABLE)
    self.assertEqual(dead.distinct_successor_fingers, ())


if __name__ == "__main__":
  unittest.main()
