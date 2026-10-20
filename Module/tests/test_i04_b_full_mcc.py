"""Force grace must preserve raw evidence and the independent collision stops."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from Module.i04_geometric_traversal import b_full_mcc as runtime


class ForceGraceTests(unittest.TestCase):
    def fixture(self):
        r = object.__new__(runtime.FullBTraversal)
        zero = np.zeros(23)
        r.data = NS(time=0., qvel=zero.copy(), qacc_warmstart=zero.copy())
        r.h = NS(model=None)
        r.last = zero.copy()
        self.guard_calls = []
        def guard(start, goal):
            self.guard_calls.append((start.copy(), goal.copy()))
            return None
        r.planner = NS(lower=np.full(23, -1.), upper=np.ones(23), safety=NS(command_collision_reason=guard))
        r.coverage = NS(visited_vertices=np.zeros(8192, bool), visited_goals=np.zeros(388, bool),
                        area_fraction=0., update=lambda *args: None)
        r.window = deque(maxlen=150)
        r.peak = r.high_steps = r.force_warning_steps = r.force_warning_observations = 0
        r.geometry_backtracks = r.rejections = r.states = r.done_hold = 0
        r.anchor = 1
        r.goal = -1
        r.offsets = np.zeros(4)
        r.points = r.normal_filter = r.bias = np.zeros((4, 3))
        r.clearance = np.zeros(4)
        r.route_progress = 0.
        r.stage = 'residual_completion'
        r.logs = {}
        r.out = Path('unused-test-no-files')
        r.last_print = float('inf')
        obs = dict(f=np.zeros(4), safe=np.zeros(4, bool), bp=np.zeros((4, 3)), bn=np.zeros((4, 3)),
                   p=np.zeros((4, 3)), n=np.zeros((4, 3)), a=np.zeros(4, bool), bf=np.zeros(4),
                   ba=np.zeros(4, bool), rejected=np.zeros(4), unsafe=[], arm=[], penetration=0.)
        r.h.object_position_m = np.zeros(3)
        self.stack = __import__('contextlib').ExitStack()
        self.stack.enter_context(patch.object(runtime, 'joint_state', lambda *args: zero.copy()))
        self.stack.enter_context(patch.object(runtime, 'set_control', lambda *args: None))
        def step(model, data):
            data.time += .002
        self.stack.enter_context(patch.object(runtime.mujoco, 'mj_step', step))
        self.stack.enter_context(patch.object(runtime.mujoco, 'mj_forward', lambda *args: None))
        self.addCleanup(self.stack.close)
        return r, obs, zero

    def test_sustained_15n_is_recorded_without_old_20ms_stop(self):
        runner, obs, command = self.fixture()
        obs['f'][3] = 15.
        for _ in range(100):
            runner.tick(command, 'FORCE_RELEASE', obs)
        self.assertEqual(runner.states, 100)
        self.assertAlmostEqual(runner.data.time, .2)
        self.assertEqual(runner.force_warning_observations, 100)
        self.assertEqual(len(self.guard_calls), 200)
        self.assertTrue(np.all(np.array(runner.logs['forces_n'])[:, 3] == 15.))
        self.assertFalse(np.any(runner.logs['safe_belly']))

    def test_40n_counter_resets_and_terminal_observation_is_not_applied(self):
        runner, obs, command = self.fixture()
        obs['f'][3] = 41.
        for _ in range(99):
            runner.tick(command, 'FORCE_RELEASE', obs)
        obs['f'] = np.array([0., 0., 0., 40.])
        runner.tick(command, 'FORCE_RELEASE', obs)
        self.assertEqual(runner.high_steps, 0)
        obs['f'] = np.array([0., 0., 0., 41.])
        for _ in range(99):
            runner.tick(command, 'FORCE_RELEASE', obs)
        before = runner.data.time
        with self.assertRaisesRegex(RuntimeError, 'SUSTAINED_DANGEROUS_FORCE'):
            runner.tick(command, 'FORCE_RELEASE', obs)
        self.assertEqual(runner.data.time, before)
        self.assertEqual(runner.logs['forces_n'][-1][3], 41.)

    def test_unsafe_collision_still_stops_on_first_observation(self):
        runner, obs, command = self.fixture()
        obs['unsafe'] = ['NONPAD_CONTACT']
        with self.assertRaisesRegex(RuntimeError, 'UNSAFE_NATIVE_CONTACT'):
            runner.tick(command, 'COOPERATIVE_SLIDE', obs)
        self.assertEqual(runner.data.time, 0.)
        self.assertEqual(runner.states, 1)


if __name__ == '__main__':
    unittest.main()
