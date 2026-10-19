"""Exercise real >8 N relief through both separate native runner entries.

The shared mustard acquisition is advanced by bounded wrist actuator commands
until an actual sensor crosses 8 N. That physical state is injected only as the
test's initial condition. No sensor is forged and no experiment success or
surface coverage follows from this short branch integration test.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.i04_geometric_traversal import normal_surface_slide, release_surface_slide
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.physics import contact_measurement, joint_state, position_jacobian, set_control
from Module.i04_geometric_traversal.reference_force_release import applied_reference_force_release
from Module.i04_geometric_traversal.sliding_safety import SlidingSafetyPlanner, arm_self_contacts
from Module.tests import test_i04_coordinated_combo_integration as combo_fixture


def copy_native(model, source):
    data = mujoco.MjData(model)
    for field in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
        getattr(data, field)[:] = getattr(source, field)
    data.time = source.time
    mujoco.mj_forward(model, data)
    return data


class ReleaseForceNativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        combo_fixture.CoordinatedComboRunnerTests.setUpClass()
        cls.fixture = combo_fixture.CoordinatedComboRunnerTests
        cls.results = {}
        cls.h = cls.fixture.h
        cls.guard = SlidingSafetyPlanner(cls.h, ExperimentConfig())
        data = copy_native(cls.h.model, cls.fixture.source)
        ids = np.r_[cls.h.arm_actuator_ids, cls.h.hand_actuator_ids]
        last = data.ctrl[ids].copy()
        force, _, normals, active, unsafe, _ = contact_measurement(cls.h, data, .2)
        anchor = int(np.argmax(force))
        assert active[anchor] and not unsafe
        inward = -normals[anchor].copy()
        start_time = data.time
        for step in range(1500):
            force, _, _, _, unsafe, _ = contact_measurement(cls.h, data, .2)
            if unsafe or arm_self_contacts(cls.h, data):
                raise AssertionError(f'Unsafe high-force fixture: {unsafe}')
            if force.max() >= 8.2:
                assert force.max() < 12., force
                cls.high_source = copy_native(cls.h.model, data)
                cls.preparation_steps = step
                cls.preparation_duration_s = data.time-start_time
                cls.high_force = force.copy()
                break
            jp, jr = position_jacobian(cls.h, data, cls.h.palm_site_id)
            jac = np.vstack((jp[:, :7], jr[:, :7]))
            delta = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6),
                                         np.r_[inward*.002*.002, np.zeros(3)])
            command = last.copy(); command[:7] += delta
            command = np.clip(command, last-.0024, last+.0024)
            command = np.clip(command, cls.guard.lower, cls.guard.upper)
            set_control(cls.h, data, command)
            mujoco.mj_step(cls.h.model, data); mujoco.mj_forward(cls.h.model, data)
            last = command
        else:
            raise AssertionError(f'Bounded native preparation did not reach 8 N: {force}')

    @classmethod
    def tearDownClass(cls):
        combo_fixture.CoordinatedComboRunnerTests.tearDownClass()

    def exercise(self, runner, label):
        h, fixture = self.h, self.fixture
        source = copy_native(h.model, self.high_source)
        ids = np.r_[h.arm_actuator_ids, h.hand_actuator_ids]
        calls, searches, commands = [], [], []
        search_update = runner.ActiveFingerSearch.update

        def observed_relief(handles, data, proposed, previous, forces, normals, config):
            actual_prior = data.ctrl[ids].copy()
            np.testing.assert_array_equal(previous, actual_prior)
            measured = contact_measurement(handles, data, .2)
            np.testing.assert_array_equal(forces, measured[0])
            np.testing.assert_array_equal(normals, measured[2])
            calls.append(dict(time=float(data.time), prior=actual_prior,
                measured_q=joint_state(handles, data), forces=forces.copy()))
            return applied_reference_force_release(handles, data, proposed, previous, forces, normals, config)

        def observed_search(search, data, nominal, forces, normals, active, **kwargs):
            searches.append(dict(time=float(data.time), enabled=kwargs['enabled'], forces=forces.copy()))
            return search_update(search, data, nominal, forces, normals, active, **kwargs)

        def observed_command(handles, data, command):
            prior = data.ctrl[ids].copy()
            commands.append(dict(time=float(data.time), q=data.qpos.copy(), qvel=data.qvel.copy(),
                prior=prior, command=command.copy(), force=contact_measurement(handles, data, .2)[0]))
            set_control(handles, data, command)

        output = fixture.root/label
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'build_scene', return_value=(h, fixture.graph, fixture.q, fixture.quality)))
            stack.enter_context(patch.object(runner, 'acquire', return_value=(source, deepcopy(fixture.initial))))
            # Hold expensive geometric scheduling only; native contact, search
            # enablement, relief, clipping and hardware guards are real.
            stack.enter_context(patch.object(runner.SurfaceStrokePlanner, 'plan', return_value=(None, [])))
            stack.enter_context(patch.object(runner, 'applied_reference_force_release', observed_relief))
            stack.enter_context(patch.object(runner.ActiveFingerSearch, 'update', observed_search))
            stack.enter_context(patch.object(runner, 'set_control', observed_command))
            stack.enter_context(redirect_stdout(io.StringIO()))
            summary = runner.run(output, budget_s=.06)

        self.assertGreater(self.preparation_steps, 0)
        self.assertAlmostEqual(self.preparation_duration_s, self.preparation_steps*.002, places=10)
        self.assertGreaterEqual(self.high_force.max(), 8.2)
        self.assertLess(self.high_force.max(), 12.)
        self.assertGreaterEqual(len(calls), 2)
        self.assertGreater(np.max(abs(calls[0]['prior']-calls[0]['measured_q'])), 1e-4)
        self.assertEqual(summary['stop_reason'], 'TIME_BUDGET_REACHED')
        self.assertLess(summary['peak_force_n'], 12.)
        self.assertEqual(summary['peak_arm_self_force_n'], 0.)
        search_by_time = {item['time']:item for item in searches}
        for item in calls:
            self.assertFalse(search_by_time[item['time']]['enabled'])
            self.assertGreaterEqual(item['forces'].max(), 4.)
        self.assertGreater(len(commands), 2)
        self.assertGreater(np.max(abs(commands[-1]['q']-commands[0]['q'])), 1e-7)
        # Replay every actual command from the identical native initial state.
        # A qpos teleport, stale previous command or hidden physics shortcut
        # cannot satisfy this comparison.
        replay = copy_native(h.model, self.high_source)
        for index, row in enumerate(commands):
            self.assertEqual(replay.time, row['time'])
            np.testing.assert_array_equal(replay.qpos, row['q'])
            np.testing.assert_array_equal(replay.qvel, row['qvel'])
            np.testing.assert_array_equal(replay.ctrl[ids], row['prior'])
            np.testing.assert_array_equal(contact_measurement(h, replay, .2)[0], row['force'])
            self.assertLessEqual(np.max(abs(row['command']-row['prior'])), .0024+1e-12)
            self.assertTrue(np.all(row['command'] >= self.guard.lower))
            self.assertTrue(np.all(row['command'] <= self.guard.upper))
            self.assertFalse(contact_measurement(h, replay, .2)[4])
            self.assertFalse(arm_self_contacts(h, replay))
            set_control(h, replay, row['command'])
            if index+1 < len(commands):
                mujoco.mj_step(h.model, replay); mujoco.mj_forward(h.model, replay)
        with np.load(output/'trace.npz') as trace:
            relief = trace['phase']=='FORCE_RELIEF'
            self.assertTrue(np.any(relief))
            self.assertFalse(np.any(trace['finger_search_enabled'][relief]))
            self.assertFalse(np.any(trace['finger_search_active'][relief]))
        events = json.loads((output/'events.json').read_text(encoding='utf-8'))
        self.assertFalse(any(event['kind']=='HARDWARE_STOP' for event in events))
        self.results[label] = dict(preparation_steps=self.preparation_steps,
            preparation_duration_s=self.preparation_duration_s,
            initial_measured_force_n=self.high_force.tolist(),
            relief_calls=len(calls), native_frames=len(commands), duration_s=summary['executed_s'],
            peak_force_n=summary['peak_force_n'], final_force_n=commands[-1]['force'].tolist(),
            peak_arm_self_force_n=summary['peak_arm_self_force_n'],
            maximum_command_step_rad=max(float(np.max(abs(row['command']-row['prior']))) for row in commands),
            prior_reference_sensor_command_max_error_rad=0., native_replay_max_error=0.,
            search_paused_during_relief=True, scope='Injected physical initial condition; branch test only')

    def test_release_entry_real_high_force_uses_actual_prior_control(self):
        self.exercise(release_surface_slide, 'release_force_branch')

    def test_normal_entry_real_high_force_uses_actual_prior_control(self):
        self.exercise(normal_surface_slide, 'normal_force_branch')


if __name__ == '__main__': unittest.main()
