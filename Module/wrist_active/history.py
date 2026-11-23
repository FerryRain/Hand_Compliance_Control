"""Synchronous measured hand-motion examples, in the moving palm frame."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..i04_geometric_traversal.fingertip_hand import ROOT


GENERATED = ROOT / 'Module/generated'
DEVELOPMENT = GENERATED / 'i01_a_multicontact_20261120'
ROBOT = GENERATED / 'active-02-a_20261121/planning-hand.mjb'
KNOTS = np.array([.25, .50])
FIELDS = ('q', 'hits', 'normal_spread', 'action', 'pad_delta', 'low_hits')


def read_model(path):
    return mujoco.MjModel.from_binary_path('scene.mjb', {'scene.mjb': Path(path).read_bytes()})


def palm_rot(pose):
    return Rotation.from_quat(pose[..., [4, 5, 6, 3]])


def normal_spread(normals, hits):
    return 1. - np.linalg.norm(np.sum(normals * hits[..., None], axis=-2), axis=-1) / np.maximum(1, hits.sum(-1))


def build_history(path):
    from ..i04_geometric_traversal.fingertip_hand import FingertipHand
    from ..acitive_02.experiment import digest

    model = read_model(ROBOT)
    hand = FingertipHand(model, object_name='fixture', surface_queries=False)
    data = mujoco.MjData(model)
    rows, sources = [], {}
    for obj in ('mustard', 'cracker'):
        case = DEVELOPMENT / obj / 'fixed-dp_r01'
        with np.load(case / 'moving_contact.npz') as series:
            low = series['region_load_n'] < 8.
            moving = series['moving']
        n = len(moving)
        tips = np.zeros((n, 4), bool)
        normals = np.zeros((n, 4, 3), dtype=np.float32)
        with np.load(case / 'live_head_contacts.npz') as touch:
            ids = (touch['step'] - 1, touch['finger'])
            tips[ids] = True
            normals[ids] = touch['normal']
        blocks, commands = [], []
        for part in sorted(case.glob('trace_[0-9]*.npz')):
            with np.load(part) as block:
                blocks.append(block['qpos'])
                commands.append(block['palm_command'])
        states, tape = np.concatenate(blocks), np.concatenate(commands)
        np.testing.assert_equal(len(states), n)
        # Equal candidate counts per development object. Retain lost/high-force outcomes.
        starts = np.flatnonzero(moving & (np.arange(n) < n - 251))
        starts = np.unique(starts[np.linspace(0, len(starts)-1, 1400).astype(int)])
        times = starts[:, None] + np.array([0, 125, 250])
        pad = np.empty((len(starts), 3, 4, 3))
        for i, sample in enumerate(times):
            for j, idx in enumerate(sample):
                data.qpos[:7] = (0., 0., 0., 1., 0., 0., 0.)
                data.qpos[hand.q] = states[idx, 7:]
                mujoco.mj_kinematics(model, data)
                pad[i, j] = data.geom_xpos[hand.pads]
        frame = palm_rot(tape[starts]).inv()
        end = tape[starts+250]
        action = np.c_[frame.apply(end[:, :3] - tape[starts, :3]), (frame * palm_rot(end)).as_rotvec()]
        samples = starts[:, None, None] + np.array([125, 250])[None, :, None] + np.arange(-25, 1, 5)
        success = ((tips[samples] & low[samples, None]).mean(axis=2))
        rows.append((states[starts, 7:], tips[starts], normal_spread(normals[starts], tips[starts]),
                     action, pad[:, 1:] - pad[:, :1], success))
        sources[obj] = dict(windows=len(starts), trace_index_sha256=digest(case/'trace_chunks.json'),
                            contact_sha256=digest(case/'live_head_contacts.npz'))
    np.savez_compressed(path, **{k: np.concatenate(v) for k, v in zip(FIELDS, zip(*rows))})
    return sources


class HandHistory:
    def __init__(self, path):
        with np.load(path) as samples:
            for key in FIELDS:
                setattr(self, key, samples[key])

    def predict(self, q, hits, spread, actions):
        base = (np.mean(((self.q-q)/.35)**2, axis=1) + .7*np.mean(self.hits != hits, axis=1)
                + ((self.normal_spread-spread)/.35)**2)
        scale = np.array([.010]*3 + [.085]*3)
        distance = base[None, :] + np.mean(((actions[:, None]-self.action)/scale)**2, axis=-1)
        near = np.argpartition(distance, 5, axis=1)[:, :6]
        d = np.take_along_axis(distance, near, axis=1)
        weight = np.exp(-(d - d.min(axis=1, keepdims=True))/.35)
        weight /= weight.sum(axis=1, keepdims=True)
        return self.pad_delta[near], self.low_hits[near], weight
