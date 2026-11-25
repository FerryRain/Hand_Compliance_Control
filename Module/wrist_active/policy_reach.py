"""Fit the moving hand's conditional light-contact workspace from single_all rollouts."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


MODULE = Path(__file__).resolve().parents[1]
TRAINING = MODULE / 'generated/i01_a_smooth_20261121/runs'
FAIR_TRAINING = MODULE / 'generated/i01_a_fair_20261121/runs'
MODEL = MODULE / 'assets/single_all_palm_reach.npz'
PROTOCOL = MODULE / 'generated/ycb_frontier_active_20261123/validated/protocol.json'
HORIZONS = np.array([.5, 1.5])
FRACTIONS = np.array([.5, 1.])


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotations(poses):
    return Rotation.from_quat(poses[..., [4, 5, 6, 3]])


def state_features(joints, hits, local_normals):
    known = local_normals * hits[..., None]
    return np.concatenate((joints / (4 * .35), .5 * hits, known.reshape((*hits.shape[:-1], 12)) / 1.4), axis=-1)


def action_features(actions, horizon):
    scale = np.array([.012] * 3 + [.13] * 3) * np.sqrt(6 * horizon / .5)
    return actions / scale


def episode(folder, checkpoint):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    dt = manifest['dt_s']
    if (manifest['policy']['checkpoint_sha256'] != checkpoint or
            not manifest['policy']['policy'].startswith('single_all') or dt != .002):
        raise ValueError(f'Unexpected policy or sample period in {folder}')
    poses, commands = [], []
    for part in json.loads((folder / 'trace_chunks.json').read_text(encoding='utf-8')):
        path = folder / part['file']
        if file_hash(path) != part['sha256']:
            raise ValueError(f'Trace checksum mismatch at {path}')
        with np.load(path) as block:
            poses.append(block['qpos'])
            commands.append(block['palm_command'])
    poses, commands = np.concatenate(poses), np.concatenate(commands)
    count = len(poses)
    hits = np.zeros((count, 4), dtype=bool)
    local = np.zeros((count, 4, 3), dtype=np.float32)
    normals = np.zeros_like(local)
    with np.load(folder / 'live_head_contacts.npz') as touch:
        light = (touch['max_region_load_n'] < 8.) & (touch['normal_force_n'] >= .2)
        time = touch['step'][light] - 1
        digit = touch['finger'][light]
        body = rotations(poses[time]).inv()
        hits[time, digit] = True
        local[time, digit] = body.apply(touch['point_m'][light] - poses[time, :3])
        normals[time, digit] = body.apply(touch['normal'][light])
    # Condition on supported starting states. Empty and overloaded outcomes
    # after the start remain in the denominator of the reach probabilities.
    ticks = np.arange(0, count - round(max(HORIZONS) / dt), round(.20 / dt))
    ticks = ticks[hits[ticks].any(axis=1)]
    frame = rotations(poses[ticks]).inv()
    features = state_features(poses[ticks, 7:], hits[ticks], normals[ticks])
    actions, locations, orientations, probabilities = [], [], [], []
    for horizon in HORIZONS:
        end = ticks + round(horizon / dt)
        change = rotations(commands[end]) * rotations(commands[ticks]).inv()
        action = np.c_[frame.apply(commands[end, :3] - commands[ticks, :3]), frame.apply(change.as_rotvec())]
        sample = (ticks[:, None, None] + np.rint(horizon * FRACTIONS / dt).astype(int)[None, :, None]
                  + np.arange(-25, 1, 5))
        valid = hits[sample]
        mass = valid.sum(axis=2)
        points = local[sample].sum(axis=2) / np.maximum(mass[..., None], 1)
        facing = normals[sample].sum(axis=2)
        facing /= np.maximum(np.linalg.norm(facing, axis=-1, keepdims=True), 1.e-8)
        actions.append(action)
        locations.append(points)
        orientations.append(facing)
        probabilities.append(valid.mean(axis=2))
    return dict(features=features, hits=hits[ticks], start=local[ticks], start_normals=normals[ticks],
                actions=np.stack(actions, axis=1), normals=np.stack(orientations, axis=1),
                contacts=np.stack(locations, axis=1), probability=np.stack(probabilities, axis=1))


def fit(sources=None, output=MODEL):
    checkpoint = json.loads(PROTOCOL.read_text(encoding='utf-8'))['checkpoint']
    rows, records = [], {}
    for source in sources or (TRAINING, FAIR_TRAINING):
        folders = sorted(Path(source).glob('*/*/shared/fixed-dp_r01'))
        if not folders:
            raise ValueError(f'No fixed-DP training rollouts in {source}')
        for folder in folders:
            row = episode(folder, checkpoint)
            rows.append(row)
            records[folder.relative_to(source.parent.parent).as_posix()] = dict(windows=len(row['hits']),
                contacts_sha256=file_hash(folder / 'live_head_contacts.npz'),
                chunks_sha256=file_hash(folder / 'trace_chunks.json'))
    metadata = dict(version=2, checkpoint=checkpoint, episodes=records, horizons_s=HORIZONS.tolist(),
                    window_stride_s=.20, contact_gate_n=8.)
    arrays = {key: np.concatenate([row[key] for row in rows]).astype(bool if key == 'hits' else np.float32)
              for key in rows[0]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


class PolicyReach:
    """Empirical joint four-pad distribution conditional on state and a palm action."""

    def __init__(self, path=MODEL):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(data['metadata'].item())
            if self.metadata['version'] != 2 or self.metadata['horizons_s'] != HORIZONS.tolist():
                raise ValueError(f'Unsupported reach model in {path}')
            self.start, self.hits = data['start'], data['hits']
            self.normals, self.start_normals = data['normals'], data['start_normals']
            self.contacts, self.probability = data['contacts'], data['probability']
            features, actions = data['features'], data['actions']
        self.indices = [cKDTree(np.c_[features, action_features(actions[:, i], horizon)])
                        for i, horizon in enumerate(HORIZONS)]

    def predict(self, joints, hits, local_normals, pads_local, actions, horizon_index):
        features = state_features(joints, hits, local_normals)
        states = np.broadcast_to(features, (len(actions), len(features)))
        query = np.c_[states, action_features(actions, HORIZONS[horizon_index])]
        index = self.indices[horizon_index]
        distance, near = index.query(query, k=np.arange(1, min(8, index.n) + 1))
        weight = np.exp(-.5 * (distance**2 - distance[:, :1]**2))
        weight /= weight.sum(axis=1, keepdims=True)
        # Register each observed material contact to its live position. For a
        # newly contacting pad use its measured absolute location in the palm.
        measured = (hits & self.hits[near])[..., None]
        shift = (pads_local - self.start[near]) * measured
        contacts = self.contacts[near, horizon_index] + shift[:, :, None]
        normals = self.normals[near, horizon_index] + ((local_normals - self.start_normals[near]) * measured)[:, :, None]
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1.e-8)
        return contacts, normals, self.probability[near, horizon_index], weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', help='Training run root; repeat for multiple archives')
    parser.add_argument('--out', type=Path, default=MODEL)
    args = parser.parse_args()
    print(json.dumps(fit(args.source, args.out), indent=2))
