"""Recover the four Active-2 table metrics from the released trajectories ZIP."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


def trial(archive, obj, condition):
    prefix = f"{obj}/{condition}/"
    run = json.loads(archive.read(prefix + "run.json"))
    scores = json.loads(archive.read(prefix + "portable-scores.json"))
    raw = archive.read(prefix + "trace.npz")
    if hashlib.sha256(raw).hexdigest() != run["trace_sha256"]:
        raise ValueError(f"Trace checksum mismatch in {prefix}")
    with np.load(io.BytesIO(raw), allow_pickle=False) as trace:
        # Row zero holds the common initial state. Time fractions use 24,000 subsequent samples.
        valid = trace["pad_valid"][1:] & (trace["region_load_n"][1:].max(axis=1) < 8)[:, None]
        if valid.shape != (24000, 4):
            raise ValueError(f"Unexpected duration or finger count in {prefix}")
        count = valid.sum(axis=1)
    with np.load(io.BytesIO(archive.read(prefix + "coverage.npz")), allow_pickle=False) as coverage:
        c25 = float(coverage["c25"][-1])

    precision, recall = scores["precision5"], scores["recall5"]
    f5 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    values = (c25, f5, (count >= 2).mean(), (count >= 3).mean())
    expected = (scores["coverage25"], scores["f5"], scores["light2_s"] / 240, scores["light3_s"] / 240)
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-10, err_msg=prefix)
    fixture = (run["seed"], run["initial_sha256"], run["model_sha256"], run["offline_mesh_sha256"])
    return fixture, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()

    groups = {mode: [] for mode in ("planned", "single_all", "compliance")}
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("object", "method", "c25", "f5", "ge2", "ge3"))
    with ZipFile(args.archive) as archive:
        for obj in ("pear", "mustard", "cracker", "bleach", "coffee"):
            paired = set()
            for mode, rows in groups.items():
                fixture, values = trial(archive, obj, mode)
                paired.add(fixture)
                rows.append(values)
                writer.writerow((obj, mode, *values))
            if len(paired) != 1:
                raise ValueError(f"Nonmatching fixtures or inference seeds for {obj}")
    for mode, rows in groups.items():
        writer.writerow(("mean", mode, *np.mean(rows, axis=0)))


if __name__ == "__main__":
    main()
