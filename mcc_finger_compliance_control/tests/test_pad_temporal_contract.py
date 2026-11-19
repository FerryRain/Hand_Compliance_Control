"""Deterministic checks for causal DP motion/tactile time semantics."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dp_motion_features import causal_motion_features


def _features(q, position, normal, mask, episode_id, step_frames=2):
    return causal_motion_features(
        q,
        position,
        normal,
        mask,
        episode_id,
        control_dt=0.01,
        step_frames=step_frames,
    )


def test_future_changes_do_not_modify_past_causal_features():
    time, envs = 10, 2
    q = np.zeros((time, envs, 16), dtype=np.float32)
    position = np.zeros((time, envs, 4, 3), dtype=np.float32)
    normal = np.zeros_like(position)
    normal[..., 2] = 1.0
    mask = np.ones((time, envs, 4), dtype=np.float32)
    episode_id = np.broadcast_to(np.arange(envs), (time, envs))
    baseline = _features(q, position, normal, mask, episode_id)
    q[7:] = 100.0
    position[7:] = 100.0
    normal[7:, ..., 0] = 1.0
    changed = _features(q, position, normal, mask, episode_id)
    for before, after in zip(baseline, changed, strict=True):
        np.testing.assert_array_equal(before[:7], after[:7])


def test_backward_difference_is_episode_local_and_has_expected_scale():
    time, envs = 7, 2
    q = np.zeros((time, envs, 16), dtype=np.float32)
    q[:, 0] = np.arange(time, dtype=np.float32)[:, None] * 0.01
    q[:, 1] = 10.0 + np.arange(time, dtype=np.float32)[:, None] * 0.02
    position = np.zeros((time, envs, 4, 3), dtype=np.float32)
    normal = np.zeros_like(position)
    normal[..., 2] = 1.0
    mask = np.ones((time, envs, 4), dtype=np.float32)
    episode_id = np.broadcast_to(np.asarray((11, 29)), (time, envs))
    q_velocity, _, _ = _features(
        q, position, normal, mask, episode_id, step_frames=2
    )
    np.testing.assert_array_equal(q_velocity[:2], 0.0)
    np.testing.assert_allclose(q_velocity[2:, 0], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(q_velocity[2:, 1], 2.0, atol=1.0e-4)


def test_contact_rates_are_zero_unless_both_endpoints_are_valid():
    time, envs = 7, 1
    q = np.zeros((time, envs, 16), dtype=np.float32)
    position = np.zeros((time, envs, 4, 3), dtype=np.float32)
    position[:, :, :, 0] = np.arange(time, dtype=np.float32)[:, None, None]
    normal = np.zeros_like(position)
    normal[..., 2] = 1.0
    mask = np.ones((time, envs, 4), dtype=np.float32)
    mask[3, :, 0] = 0.0
    episode_id = np.zeros((time, envs), dtype=np.int64)
    _, point_velocity, normal_rate = _features(
        q, position, normal, mask, episode_id, step_frames=2
    )
    np.testing.assert_array_equal(point_velocity[3, :, 0], 0.0)
    np.testing.assert_array_equal(point_velocity[5, :, 0], 0.0)
    np.testing.assert_array_equal(normal_rate[3, :, 0], 0.0)
    np.testing.assert_array_equal(normal_rate[5, :, 0], 0.0)
    assert np.linalg.norm(point_velocity[4, :, 0]) > 0.0


def test_normal_rate_uses_past_cross_product_and_masks_recontact():
    time, envs = 6, 1
    q = np.zeros((time, envs, 16), dtype=np.float32)
    position = np.zeros((time, envs, 4, 3), dtype=np.float32)
    normal = np.zeros_like(position)
    normal[..., 0] = 1.0
    normal[2:, ..., 0] = 0.0
    normal[2:, ..., 1] = 1.0
    mask = np.ones((time, envs, 4), dtype=np.float32)
    episode_id = np.zeros((time, envs), dtype=np.int64)
    _, _, normal_rate = _features(
        q, position, normal, mask, episode_id, step_frames=2
    )
    assert (normal_rate[2, :, :, 2] > 0.0).all()
    mask[2] = 0.0
    _, _, masked_rate = _features(
        q, position, normal, mask, episode_id, step_frames=2
    )
    np.testing.assert_array_equal(masked_rate[2], 0.0)
    np.testing.assert_array_equal(masked_rate[4], 0.0)
