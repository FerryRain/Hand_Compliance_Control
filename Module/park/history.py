"""Read physical contact events and synchronise each one to the measured wrist."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import FINGERS, digest, rotation, write_json


@dataclass
class Episode:
    name: str
    group: str
    finger: np.ndarray
    palm: np.ndarray
    world: np.ndarray
    normals: np.ndarray
    tick: np.ndarray
    source: dict


def valid(points, normals, force, loads):
    length = np.linalg.norm(normals, axis=-1)
    return (np.isfinite(points).all(axis=-1) & np.isfinite(length) & (length > .99)
            & (length < 1.01) & np.isfinite(force) & (force >= .2) & (loads < 8.))


def split(obj):
    return {"ovoid": "test", "oblate": "validation"}.get(obj, "train")


def extract(name, obj, qpos, tick, finger, point, normal, force, peak, dt, source):
    if not len(tick) or tick.min() < 0 or tick.max() >= len(qpos) or not np.isin(finger, range(4)).all():
        raise ValueError(f"Contact index outside the measured trace in {name}")
    good = valid(point, normal, force, peak)
    # One real event per finger per 50 ms bin. Keep the original event timestamp at a boundary.
    ix = np.flatnonzero(good)
    bins = (tick[ix] // round(.05 / dt)) * 4 + finger[ix]
    _, first = np.unique(bins, return_index=True)
    ix = ix[first]
    wrist = qpos[tick[ix], :7]
    local = rotation(wrist).inv().apply(point[ix] - wrist[:, :3])
    if not np.isfinite(local).all():
        raise ValueError(f"Invalid measured wrist in {name}")
    return Episode(name, split(obj), finger[ix], local, point[ix], normal[ix], tick[ix],
                   dict(**source, raw_valid=int(good.sum()), sampled=len(ix), object=obj, dt_s=dt))


def read_i01(root, checkpoint):
    for manifest_path in sorted(Path(root).glob("i01_a_*/runs/*/*/shared/fixed-dp_r01/manifest.json")):
        folder = manifest_path.parent
        meta = json.loads(manifest_path.read_bytes())
        if (meta["policy"]["checkpoint_sha256"] != checkpoint or meta["fingers"] != list(FINGERS)
                or meta["dt_s"] != .002 or not meta["controller"]["mcc_feedback_enabled"]):
            raise ValueError(f"Different deployment schema in {folder}")
        name = folder.relative_to(root).as_posix()
        poses = []
        chunks = folder / "trace_chunks.json"
        for part in json.loads(chunks.read_bytes()):
            path = folder / part["file"]
            if digest(path) != part["sha256"]:
                raise ValueError(f"Trace hash differs in {path}")
            with np.load(path, allow_pickle=False) as block:
                if len(block["qpos"]) != part["states"]:
                    raise ValueError(f"Incomplete trace in {path}")
                poses.append(block["qpos"][:, :7])
        events = folder / "live_head_contacts.npz"
        with np.load(events, allow_pickle=False) as d:
            yield extract(name, folder.parents[2].name, np.concatenate(poses), d["step"] - 1,
                          d["finger"], d["point_m"], d["normal"], d["normal_force_n"],
                          d["max_region_load_n"], .002,
                          dict(contact_sha256=digest(events), chunks_sha256=digest(chunks)))


def read_sweeps(root, checkpoint):
    for log in sorted(Path(root).glob("*/*/sweep-dp/run.json")):
        folder = log.parent
        obj = folder.parents[1].name
        # Exclude all five formal benchmark objects, including starts different from release.
        if obj not in {"ellipsoid", "meat-can", "oblate", "ovoid", "superellipsoid"}:
            continue
        meta = json.loads(log.read_bytes())
        if meta["policy"]["checkpoint_sha256"] != checkpoint:
            raise ValueError(f"Unexpected Finger Policy at {log}")
        trace = folder / "trace.npz"
        if digest(trace) != meta["trace_sha256"]:
            raise ValueError(f"Changed trace at {trace}")
        with np.load(trace, allow_pickle=False) as d:
            tick, finger = np.nonzero(d["pad_valid"])
            yield extract("sweep/" + folder.relative_to(root).as_posix(), obj, d["qpos"], tick, finger,
                          d["points_m"][tick, finger], d["normals_outward"][tick, finger],
                          d["pad_force_n"][tick, finger], d["region_load_n"].max(1)[tick],
                          .01, dict(trace_sha256=meta["trace_sha256"]))


def save_episodes(directory, episodes, checkpoint):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, ep in enumerate(episodes):
        target = directory / f"episode-{idx:02d}.npz"
        with target.with_suffix(".partial").open("wb") as stream:
            np.savez_compressed(stream, finger=ep.finger, palm=ep.palm, world=ep.world,
                                normals=ep.normals, tick=ep.tick)
        target.with_suffix(".partial").replace(target)
        records.append(dict(file=target.name, sha256=digest(target), name=ep.name, group=ep.group, source=ep.source))
    write_json(directory / "index.json", dict(version=1, checkpoint=checkpoint, fingers=FINGERS,
               coordinates="palm and world XYZ in metres, outward world normals", records=records))


def load_episodes(directory, checkpoint):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_bytes())
    if index["version"] != 1 or index["checkpoint"] != checkpoint or index["fingers"] != list(FINGERS):
        raise ValueError("Different history schema or Finger Policy checkpoint")
    for row in index["records"]:
        path = directory / row["file"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Changed derived contact archive {path}")
        with np.load(path, allow_pickle=False) as data:
            yield Episode(row["name"], row["group"], source=row["source"], **dict(data))


if __name__ == "__main__":
    import argparse
    from itertools import chain
    from Module.palmreach6.__main__ import FIXTURES
    p = argparse.ArgumentParser(description="Export thinned measured contacts with raw-source provenance")
    p.add_argument("--i01", required=True, type=Path)
    p.add_argument("--sweeps", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    checkpoint = json.loads((FIXTURES / "fixtures.json").read_bytes())["checkpoint"]
    save_episodes(args.out, chain(read_i01(args.i01, checkpoint), read_sweeps(args.sweeps, checkpoint)), checkpoint)

