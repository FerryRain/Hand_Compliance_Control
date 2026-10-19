"""Native regression for command-space preload checks.

The unittest fixture is constructed from repository models and requires no
generated experiment. The optional ``--middle205 SOURCE_CASE`` CLI separately
checks the exact previously observed mustard state, without touching it.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from Module.tests import test_i04_contact_handoff as native_fixture
from Module.i04_geometric_traversal.coordinated_surface_slide import checked_preload_path
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.physics import contact_measurement, set_joint_state, set_control
from Module.i04_geometric_traversal.sliding_safety import SlidingSafetyPlanner


class RecordingPlanner:
    def __init__(self, native):
        self.native = native
        self.lower, self.upper = native.lower.copy(), native.upper.copy()
        self.edges = []

    def edge_free(self, first, second):
        self.edges.append((first.copy(), second.copy()))
        return self.native.edge_free(first, second)


def snapshot(h, data):
    state = {key: getattr(data, key).copy() for key in
             ('qpos', 'qvel', 'ctrl', 'qacc', 'qacc_warmstart', 'geom_xpos', 'site_xpos')}
    state['time'] = data.time
    parameters = {key: getattr(h.model, key).copy() for key in
                  ('geom_margin', 'geom_solref', 'geom_solimp', 'dof_damping',
                   'actuator_gainprm', 'actuator_ctrlrange', 'body_pos', 'body_quat')}
    return state, parameters


def assert_plant_unchanged(h, data, before):
    for key, value in before[0].items():
        np.testing.assert_array_equal(getattr(data, key), value)
    for key, value in before[1].items():
        np.testing.assert_array_equal(getattr(h.model, key), value)


class CoordinatedHandoffNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.ContactHandoffNativeTests.setUpClass()

    def setUp(self):
        fixture = native_fixture.ContactHandoffNativeTests()
        fixture.setUp()
        self.h, self.data = fixture.h, fixture.data
        self.native = SlidingSafetyPlanner(self.h, ExperimentConfig())
        self.planner = RecordingPlanner(self.native)
        self.root = fixture.nominal.copy()
        # This models a real servo tracking error just outside the planner's
        # 15 mrad inward margin, still well inside the original native limit.
        self.root[8] = self.native.upper[8] + 15e-6
        set_joint_state(self.h, self.data, self.root)
        self.previous = self.root.copy()
        self.previous[19:23] += [0., .001, -.003, .002]
        set_control(self.h, self.data, self.previous)
        mujoco.mj_forward(self.h.model, self.data)
        self.forces, _, self.normals, self.active, unsafe, _ = contact_measurement(self.h, self.data, .2)
        self.assertFalse(unsafe)
        self.assertTrue(self.active[3])
        self.assertTrue(.2 <= self.forces[3] < 8.)
        goal = self.root.copy(); goal[11:15] += [.003, .012, .005, .002]
        self.path = SimpleNamespace(root_q=self.root.copy(), waypoints=np.stack([self.root, goal]))

    def test_native_safe_path_accepts_15_microradian_existing_joint_clip(self):
        passed, diag = checked_preload_path(self.planner, self.path, self.previous,
                                            self.active, self.forces)
        self.assertTrue(passed)
        self.assertAlmostEqual(diag['maximum_existing_joint_clip_rad'], 15e-6, places=12)
        self.assertEqual(diag['checked_states'], len(self.planner.edges))
        self.assertGreaterEqual(len(self.planner.edges), 4)
        for first, second in self.planner.edges:
            self.assertTrue(np.all(first >= self.native.lower))
            self.assertTrue(np.all(second <= self.native.upper))
            self.assertLessEqual(np.max(abs(second-first)), .004+1e-12)
            self.assertEqual(second[8], self.native.upper[8])
        # Physical joint margin was never widened to accommodate the error.
        native_upper = self.h.hand_joint_ranges_rad[1, 1]
        self.assertAlmostEqual(native_upper-self.root[8], .015-15e-6, places=12)

    def test_native_arm_collision_is_still_rejected_after_command_clipping(self):
        goal = self.root.copy()
        goal[:7] = [1.2254070302740296, -1.271202017932884, 1.4538529625885903,
                    -3.0271, -1.5317249192627793, 1.6445867946393036, -1.6841290859917004]
        self.assertEqual(self.native.collision_reason(goal), 'SHALLOW_ARM_SELF_COLLISION')
        path = SimpleNamespace(root_q=self.root.copy(), waypoints=np.stack([self.root, goal]))
        passed, diag = checked_preload_path(self.planner, path, self.previous,
                                            self.active, self.forces)
        self.assertFalse(passed)
        self.assertEqual(diag['reason'], 'PRELOADED_COMMAND_PATH_REJECTED')
        self.assertGreater(diag['checked_states'], 1)
        self.assertFalse(self.native.edge_free(*self.planner.edges[-1]))

    def test_preload_is_only_from_current_safe_measured_fingers_and_never_arm(self):
        previous = self.previous.copy()
        previous[:7] += .1
        previous[7:19] += .02
        # Native model-backed boundary test: flags and force values are
        # deliberately varied; these are inputs, not invented contact claims.
        active = np.array([True, True, False, True])
        forces = np.array([.199, 8., 2., self.forces[3]])
        passed, diag = checked_preload_path(self.planner, self.path, previous, active, forces)
        self.assertTrue(passed)
        bias = np.asarray(diag['held_hand_reference_bias_rad'])
        np.testing.assert_array_equal(bias[:12], np.zeros(12))
        np.testing.assert_array_equal(bias[12:], previous[19:23]-self.root[19:23])
        for first, second in self.planner.edges:
            np.testing.assert_array_equal(first[:7], self.root[:7])
            np.testing.assert_array_equal(second[:7], self.root[:7])

    def test_no_contact_preserves_no_preload_even_if_prior_command_differs(self):
        previous = self.previous.copy(); previous[7:] += .02
        passed, diag = checked_preload_path(self.planner, self.path, previous,
                                            np.zeros(4, bool), np.zeros(4))
        self.assertTrue(passed)
        np.testing.assert_array_equal(diag['held_hand_reference_bias_rad'], np.zeros(16))

    def test_scratch_geometry_check_never_mutates_plant_model_or_path(self):
        before = snapshot(self.h, self.data)
        inputs = [x.copy() for x in (self.previous, self.active, self.forces,
                                     self.path.root_q, self.path.waypoints)]
        checked_preload_path(self.planner, self.path, self.previous, self.active, self.forces)
        assert_plant_unchanged(self.h, self.data, before)
        for actual, expected in zip((self.previous, self.active, self.forces,
                                    self.path.root_q, self.path.waypoints), inputs):
            np.testing.assert_array_equal(actual, expected)


def check_middle205(source_directory):
    """Read-only exact-state integration check, independent of unittest fixtures."""
    from Module.i04_geometric_traversal.posture_recovery_probe import prepare_source, make_plant
    from Module.i04_geometric_traversal.finger_reconfiguration_planner import FingerReconfigurationPlanner
    source = prepare_source(Path(source_directory), 205., 1)
    h, graph, data, sensor_errors = make_plant(source)
    before = snapshot(h, data)
    forces, _, normals, active, unsafe, _ = contact_measurement(h, data, .2)
    assert not unsafe and max(sensor_errors.values()) == 0.
    proposal = FingerReconfigurationPlanner(h, graph)
    path, plan = proposal.plan(data, 1, active, forces, normals)
    assert path is not None, plan
    passed, diag = checked_preload_path(proposal.planner, path,
        source['frame']['sensor_command_rad'], active, forces)
    assert passed, diag
    assert abs(diag['maximum_existing_joint_clip_rad']-1.5127870929720899e-5) < 1e-9, diag
    assert_plant_unchanged(h, data, before)
    return dict(status='PASSED', scope='Exact run04 middle205 read-only command-path check',
        native_sensor_errors=sensor_errors, source_trace=source['provenance']['source_trace'],
        physical_path_duration_s=path.duration_s, preloaded_command_path=diag,
        plant_and_model_unchanged=True, physical_recovery_claim=False)


if __name__ == '__main__':
    import sys
    if '--middle205' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--middle205', type=Path, required=True)
        args = parser.parse_args()
        print(json.dumps(check_middle205(args.middle205), indent=2))
    else:
        unittest.main()
