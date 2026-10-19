"""Information boundary and measured-only gates for the new MCC comparison."""
from types import SimpleNamespace
import numpy as np
import unittest

from Module.i04_geometric_traversal.contracts import WristTarget, WholeHandTarget
from Module.i04_geometric_traversal.matched_mcc import (
    CommonSettings, MatchedMCCTraversal, WristOnlyReference, measured_ready, state_hash,
)


class ForbiddenGeometry:
    def __array__(self, *args, **kwargs):
        raise AssertionError('A accessed private finger geometry')


def test_a_adapter_never_reads_private_finger_points_or_normals():
    wrist = np.array([.1, .2, .3, 1., 0., 0., 0.])
    own = np.arange(23, dtype=float)/100
    received = []
    def solve(target, *, root):
        assert type(target) is WristTarget
        received.append(target.pose_world.copy())
        np.testing.assert_array_equal(root, own)
        return own.copy()
    adapter = SimpleNamespace(cell='A-MCC', last=own,
        _wrist=lambda q: wrist.copy(),
        _publish_wrist=lambda value, phase: [WristTarget(value)],
        a_reference=SimpleNamespace(solve=solve,
            pad_reference=lambda q: (np.ones((4, 3)), np.tile([0., 0., 1.], (4, 1)))))
    result = MatchedMCCTraversal._adapt_patch(adapter, ForbiddenGeometry(), ForbiddenGeometry(),
                                             ForbiddenGeometry(), 'TEST')
    np.testing.assert_array_equal(received[0], wrist)
    np.testing.assert_array_equal(result[2], own)


def test_a_wrist_solver_freezes_its_own_hand_and_rejects_whole_hand_targets():
    seen = []
    own_hand = np.linspace(.1, .4, 16)
    def solve(q, target, *, iterations):
        seen.append(q.copy())
        return q.copy(), 0., 0., 0.
    method = object.__new__(WristOnlyReference)
    method.previous = np.zeros(23)
    method.nominal_hand = own_hand
    method.safety = SimpleNamespace(solve=solve)
    target = WristTarget([0., 0., 0., 1., 0., 0., 0.])
    result = method.solve(target)
    np.testing.assert_array_equal(result[7:], own_hand)
    np.testing.assert_array_equal(seen[0][7:], own_hand)
    with unittest.TestCase().assertRaisesRegex(TypeError, 'only a WristTarget'):
        method.solve(WholeHandTarget(target, np.zeros((4, 3))))


def test_measured_quorum_tolerates_short_losses_but_requires_current_support():
    cfg = CommonSettings()
    history = np.ones((150, 4), dtype=bool)
    history[:30] = False
    assert measured_ready(history, cfg)
    history[30] = False
    assert not measured_ready(history, cfg)
    history[:] = True
    history[-1] = [True, False, False, False]
    assert not measured_ready(history, cfg)
    history[-1] = [False, True, False, True]  # no special anchor
    assert measured_ready(history, cfg)
    assert not measured_ready(history[:-1], cfg)


def test_initial_state_digest_covers_all_physics_state_not_only_qpos():
    state = SimpleNamespace(qpos=np.zeros(23), qvel=np.zeros(23), ctrl=np.zeros(23),
                            qacc_warmstart=np.zeros(23), time=0.)
    original = state_hash(state)
    for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
        getattr(state, name)[0] = .1
        assert state_hash(state) != original
        getattr(state, name)[0] = 0.
    state.time = .002
    assert state_hash(state) != original


def test_a_cache_key_ignores_private_geometry_changes():
    wrist = np.array([.1, .2, .3, 1., 0., 0., 0.])
    adaptations = []
    marker = (np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(23), wrist)
    def adapt(*args):
        adaptations.append(1)
        return marker
    runner = SimpleNamespace(cell='A-MCC', _last_private_key=None, _last_adapted=None,
        _wrist=lambda q: wrist, _adapt_patch=adapt, _execute=lambda *args: args)
    MatchedMCCTraversal.contact_step(runner, ForbiddenGeometry(), ForbiddenGeometry(), np.zeros(23), 'TEST')
    MatchedMCCTraversal.contact_step(runner, ForbiddenGeometry(), ForbiddenGeometry(), np.ones(23), 'TEST')
    assert len(adaptations) == 1


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
