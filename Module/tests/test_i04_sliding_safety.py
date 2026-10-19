"""Native collision regressions for shallow FR3 arm contact and safe exits."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml
from Module.e05_physics.scene import FINGERS
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.fixtures import MUSTARD
from Module.i04_geometric_traversal.full_surface import FullSurfacePlanner, acquire, build_scene
from Module.i04_geometric_traversal.physics import contact_measurement, joint_state, set_joint_state
from Module.i04_geometric_traversal.planning import SharedGeometricTrajectoryPlanner
from Module.i04_geometric_traversal.sliding_safety import (
    ARM_SELF_CLEARANCE, SHALLOW_ARM_SELF_COLLISION, SlidingSafetyPlanner,
    arm_self_contacts, is_nonadjacent_arm_pair,
)
from Module.i04_geometric_traversal.surface_slide import tracking_progress_scale


ROOT = Path(__file__).resolve().parents[2]
RUN02 = ROOT / 'Module/generated/mustard_active_sliding_run02_20261016/B-MCC_seed7'


def native_pair_fixture(*, hand_gap=None, hand_is_pad=False):
    bodies = []
    for i in range(23):
        name = ('fr3v2_link1' if i == 0 else 'fr3v2_link3' if i == 1 else f'dummy_{i}')
        x = 0. if i == 0 else .0205 if i == 1 else 20. + i
        if i == 22 and hand_gap is not None:
            x = 10. + .02 + hand_gap
        bodies.append(f'<body name="{name}" pos="{x} 0 0">'
            f'<joint name="j{i}" type="slide" axis="1 0 0" range="-1 1"/>'
            f'<geom name="g{i}" type="sphere" size=".01" mass="1"/></body>')
    xml = ('<mujoco><compiler autolimits="true"/><option gravity="0 0 0"/>'
        '<default><geom solref=".005 1"/></default><worldbody>'
        '<geom name="object" type="sphere" pos="10 0 0" size=".01"/>'
        + ''.join(bodies) + '</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    gid = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    tips = [gid('g22' if hand_is_pad else 'g21'), gid('g20'), gid('g19'), gid('g18')]
    return SimpleNamespace(model=model, arm_qpos_adrs=np.arange(7), hand_qpos_adrs=np.arange(7, 23),
        arm_joint_ranges_rad=model.jnt_range[:7], hand_joint_ranges_rad=model.jnt_range[7:],
        tip_geom_ids=np.array(tips), object_geom_ids=np.array([gid('object')]), object_geom_id=gid('object'))


class NativeArmSafetyTests(unittest.TestCase):
    def setUp(self):
        self.h = native_pair_fixture()
        self.planner = SlidingSafetyPlanner(self.h, ExperimentConfig())
        self.q = np.zeros(23)

    def test_positive_buffer_is_detected_without_changing_physical_contact(self):
        data = mujoco.MjData(self.h.model)
        mujoco.mj_forward(self.h.model, data)
        self.assertEqual(data.ncon, 0)
        self.assertEqual(FullSurfacePlanner(self.h, ExperimentConfig()).collision_reason(self.q), None)
        self.assertEqual(self.planner.collision_reason(self.q), ARM_SELF_CLEARANCE)
        self.assertAlmostEqual(min(self.planner.arm_clearance_state().values()), .0005, places=10)
        self.assertEqual(arm_self_contacts(self.h, data), [])
        mujoco.mj_forward(self.h.model, data)
        self.assertEqual(data.ncon, 0)

    def test_shallow_native_force_is_detected_below_old_two_mm_cutoff(self):
        q = self.q.copy(); q[1] = -.0006  # 0.1 mm overlap.
        data = mujoco.MjData(self.h.model)
        set_joint_state(self.h, data, q)
        contacts = arm_self_contacts(self.h, data)
        self.assertEqual(len(contacts), 1)
        self.assertAlmostEqual(contacts[0]['distance_m'], -.0001, places=10)
        self.assertGreater(contacts[0]['force_n'], .5)
        self.assertEqual(set(contacts[0]['body_names']), {'fr3v2_link1', 'fr3v2_link3'})
        self.assertEqual(arm_self_contacts(self.h, data, contacts[0]['force_n']), [])
        self.assertEqual(FullSurfacePlanner(self.h, ExperimentConfig()).collision_reason(q), None)
        self.assertEqual(self.planner.collision_reason(q), SHALLOW_ARM_SELF_COLLISION)
        clear = q.copy(); clear[1] = .002
        self.assertFalse(self.planner.edge_free(q, clear))
        self.assertEqual(self.planner.command_collision_reason(q, clear), SHALLOW_ARM_SELF_COLLISION)

    def test_positive_buffer_allows_separating_exit_but_rejects_entry_and_approach(self):
        outward = self.q.copy(); outward[1] = .0002
        clear = self.q.copy(); clear[1] = .002
        inward = self.q.copy(); inward[1] = -.0001
        self.assertTrue(self.planner.edge_free(self.q, outward))
        self.assertTrue(self.planner.edge_free(self.q, clear))
        self.assertEqual(self.planner.command_collision_reason(self.q, outward), None)
        self.assertFalse(self.planner.edge_free(clear, self.q))
        self.assertFalse(self.planner.edge_free(self.q, inward))
        self.assertEqual(self.planner.command_collision_reason(clear, self.q), ARM_SELF_CLEARANCE)
        self.assertEqual(self.planner.command_collision_reason(self.q, inward), ARM_SELF_CLEARANCE)

    def test_exit_must_not_exchange_one_buffered_pair_for_another(self):
        # Per-pair checks prevent a global-minimum-only clearance from hiding
        # the introduction of a second collision while the first separates.
        self.assertFalse(self.planner._arm_exit_allowed({(1, 3): .0003},
            {(1, 3): .0008, (1, 5): .0009}))
        self.assertTrue(self.planner._arm_exit_allowed({(1, 3): .0003, (1, 5): .0002},
            {(1, 3): .0008, (1, 5): .0007}))

    def test_model_and_measured_state_unchanged_even_if_nested_query_raises(self):
        model = self.h.model
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        fields = ('geom_margin', 'geom_gap', 'geom_solref', 'geom_solimp',
                  'geom_contype', 'geom_conaffinity')
        before = {key: getattr(model, key).copy() for key in fields}
        state = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
        self.planner.collision_reason(self.q)
        with patch.object(SharedGeometricTrajectoryPlanner, 'collision_reason', side_effect=RuntimeError('query failed')):
            with self.assertRaisesRegex(RuntimeError, 'query failed'):
                self.planner.collision_reason(self.q)
        for key in fields:
            np.testing.assert_array_equal(getattr(model, key), before[key])
        for current, saved in zip((data.qpos, data.qvel, data.ctrl, data.time), state):
            np.testing.assert_array_equal(current, saved)

    def test_object_hand_and_pad_thresholds_remain_identical(self):
        q = self.q.copy(); q[1] = .003  # Arm safely outside its buffer.
        cases = [(False, .002, 'NON_TIP_CLEARANCE'),
                 (False, -.0006, 'NON_TIP_OBJECT_COLLISION'),
                 (True, -.001, None), (True, -.0025, 'DEEP_PAD_PENETRATION')]
        for is_pad, gap, expected in cases:
            with self.subTest(pad=is_pad, gap=gap):
                h = native_pair_fixture(hand_gap=gap, hand_is_pad=is_pad)
                old = FullSurfacePlanner(h, ExperimentConfig())
                new = SlidingSafetyPlanner(h, ExperimentConfig())
                self.assertEqual(old.collision_reason(q), expected)
                self.assertEqual(new.collision_reason(q), expected)

    def test_same_adjacent_welded_and_non_arm_bodies_are_excluded(self):
        xml = '''<mujoco><worldbody><body name="fr3v2_link0"><geom size=".01"/>
          <body name="fr3v2_link1"><joint type="hinge"/><geom size=".01"/>
            <body name="fr3v2_link2"><joint type="hinge"/><geom size=".01"/>
              <body name="fr3v2_link3"><joint type="hinge"/><geom size=".01"/>
                <body name="fr3v2_link8"><geom size=".01"/></body>
              </body></body></body></body>
          <body name="hand"><joint type="hinge"/><geom size=".01"/></body>
          </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for a, b in [('fr3v2_link1', 'fr3v2_link1'), ('fr3v2_link1', 'fr3v2_link2'),
                     ('fr3v2_link3', 'fr3v2_link8'), ('fr3v2_link2', 'fr3v2_link8'),
                     ('fr3v2_link1', 'hand')]:
            self.assertFalse(is_nonadjacent_arm_pair(model, body(a), body(b)), (a, b))
        self.assertTrue(is_nonadjacent_arm_pair(model, body('fr3v2_link1'), body('fr3v2_link3')))
        self.assertTrue(is_nonadjacent_arm_pair(model, body('fr3v2_link1'), body('fr3v2_link8')))


class TrackingGateTests(unittest.TestCase):
    def test_stuck_arm_pauses_even_when_reference_is_moving_slowly(self):
        # A speed limiter alone misses a joint accumulating compliant error
        # against a hard obstacle. Even a 0.22 rad/s reference must stop here.
        measured = np.zeros(23)
        for tick in range(4):
            applied = np.zeros(23); applied[5] = .024 + tick * .22 * .002
            scale, lag = tracking_progress_scale(measured, applied, applied.copy())
            self.assertEqual(scale, 0.)
            self.assertGreaterEqual(lag, .024)

    def test_independent_finger_search_bias_does_not_pause_a_tracking_arm(self):
        measured = np.zeros(23)
        applied = measured.copy(); applied[7:] = np.linspace(-.5, .5, 16)
        nominal = measured.copy(); nominal[7:] = -applied[7:]
        scale, lag = tracking_progress_scale(measured, applied, nominal)
        self.assertEqual(scale, 1.)
        self.assertEqual(lag, 0.)

    def test_nominal_reference_cannot_run_ahead_of_collision_filtered_command(self):
        measured = np.zeros(23); applied = measured.copy(); nominal = measured.copy()
        nominal[2] = .0101
        scale, lag = tracking_progress_scale(measured, applied, nominal)
        self.assertEqual(scale, 0.)
        self.assertAlmostEqual(lag, .0101)


@unittest.skipUnless((MUSTARD / 'visual_scaled.obj').exists(), 'Mustard asset unavailable')
class MustardArmSafetyTests(unittest.TestCase):
    def test_fresh_acquisition_is_legal_and_retains_actual_contact(self):
        scratch_root = ROOT / '.runtime_tmp'; scratch_root.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix='arm_safety_', dir=scratch_root) as directory:
            self.assertEqual(Path(directory).resolve().parent, scratch_root.resolve())
            h, graph, q, _ = build_scene(directory)
            data, _ = acquire(h, graph, q, seed=7)
            self.assertTrue(contact_measurement(h, data, .2)[3].any())
            self.assertEqual(arm_self_contacts(h, data), [])
            planner = SlidingSafetyPlanner(h, ExperimentConfig())
            self.assertEqual(planner.collision_reason(joint_state(h, data)), None)

    @unittest.skipUnless((RUN02 / 'trace_chunks/0004.npz').exists(), 'Preserved run02 evidence unavailable')
    def test_reconstructed_run02_link5_link7_event_precedes_fingertip_spike(self):
        model = compile_model_xml((RUN02 / 'model.xml').read_text(encoding='utf-8'))
        ids = lambda kind, names: np.array([mujoco.mj_name2id(model, kind, n) for n in names])
        joints = ids(mujoco.mjtObj.mjOBJ_JOINT, (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES))
        actuators = ids(mujoco.mjtObj.mjOBJ_ACTUATOR,
            (*[f'fr3_arm_act_{i}' for i in range(1, 8)], *[f'act_{i}' for i in range(16)]))
        tips = ids(mujoco.mjtObj.mjOBJ_GEOM, [f.proxy_geom.replace('e05_', 'fr3_') for f in FINGERS])
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
        objects = np.flatnonzero(model.geom_bodyid == body)
        h = SimpleNamespace(model=model, arm_qpos_adrs=model.jnt_qposadr[joints[:7]],
            hand_qpos_adrs=model.jnt_qposadr[joints[7:]], arm_joint_ranges_rad=model.jnt_range[joints[:7]],
            hand_joint_ranges_rad=model.jnt_range[joints[7:]], tip_geom_ids=tips,
            object_geom_ids=objects, object_geom_id=int(objects[0]))
        with np.load(RUN02 / 'trace_chunks/0004.npz') as trace:
            row = int(np.argmin(abs(trace['time_s'] - 212.5)))
            self.assertAlmostEqual(float(trace['time_s'][row]), 212.5)
            data = mujoco.MjData(model)
            data.qpos[model.jnt_qposadr[joints]] = trace['q_rad'][row]
            data.qvel[model.jnt_dofadr[joints]] = trace['qvel_rad_s'][row]
            data.ctrl[actuators] = trace['sensor_command_rad'][row]
            data.qacc_warmstart[:] = trace['qacc_warmstart'][row]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(contact_measurement(h, data, .2)[0], trace['forces_n'][row], atol=1e-8)
            events = arm_self_contacts(h, data)
            link57 = [e for e in events if set(e['body_names']) == {'fr3v2_link5', 'fr3v2_link7'}]
            self.assertTrue(link57, events)
            self.assertGreater(max(e['force_n'] for e in link57), 50.)
            self.assertGreater(min(e['distance_m'] for e in link57), -.002)
            self.assertLess(float(np.max(trace['forces_n'][row])), 12.)
            q = trace['q_rad'][row]
            self.assertEqual(FullSurfacePlanner(h, ExperimentConfig()).collision_reason(q), None)
            self.assertEqual(SlidingSafetyPlanner(h, ExperimentConfig()).collision_reason(q), SHALLOW_ARM_SELF_COLLISION)


if __name__ == '__main__':
    unittest.main()
