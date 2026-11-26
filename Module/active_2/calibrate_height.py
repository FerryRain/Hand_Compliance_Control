"""Calibrate the AUDE wrist-height band from the released single_all training data.

Run with --mustard data/training/mustard222/prepared --expanded data/training/expanded/prepared
and --integrity data/training/settings/integrity.json from a FingerPolicy_Data download.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


PROBABILITIES = np.array([0.025, 0.5, 0.975])
INTEGRITY_SHA256 = "cfed57d5c7fa55bd83e971c8209216311cfd8ebfbf8b159a3b6c29d6c755b979"
CHECKPOINT = "9fbe34c3c36fd9f74ad5630ccb15c6200dfd3b44bdbf97da86f8d9afbcc55ec4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(file, declared, entry):
    if sha(file) != declared[entry]:
        raise ValueError(f"Released data digest differs for {entry}")
    return file


def touch_heights(observation):
    if observation.shape[-1] != 45:
        raise ValueError("Expected the released 45-feature palm_lower observation")
    flag = observation[..., 10]
    if not np.isin(flag, (0, 1)).all():
        raise ValueError("Expected a binary measured-contact flag")
    # Features 4:7 contain a measured or held palm-frame point. Only a live
    # measurement may enter the distribution. The palm faces local negative Z.
    z = observation[..., 6]
    if not np.isfinite(z[flag == 1]).all():
        raise ValueError("Nonfinite measured palm-frame contact")
    return np.where(flag == 1, -z.astype(np.float64), np.nan)


def frame_summary(depths, stats):
    count = np.isfinite(depths).sum(axis=1)
    stats.update(count.tolist())
    present = count > 0
    return np.nanmedian(depths[present], axis=1)


def summarize(values, finger_counts):
    return dict(all_frames=sum(finger_counts.values()), contact_frames=len(values),
                frames_by_contact_count={str(i): finger_counts[i] for i in range(5)},
                quantiles_m=np.quantile(values, PROBABILITIES, method="inverted_cdf").tolist())


def mustard_frames(directory, declared):
    manifest = checked(directory / "episode_split.json", declared, "data/old222/episode_split.json")
    rows = json.loads(manifest.read_bytes())["episodes"]
    training = [row for row in rows if row["split"] == "train"]
    ids = {row["episode_id"] for row in training}
    if len(ids) != len(training) or len({row["sha256"] for row in training}) != len(training):
        raise ValueError("Repeated training episode in the Mustard manifest")
    source = checked(directory / "single_finger_frames.npz", declared, "data/old222/single_finger_frames.npz")
    with np.load(source, allow_pickle=False) as data:
        ids_from_roots = set(data["roots"][data["roots"][:, 3] == 0, 0])
        if ids_from_roots != ids:
            raise ValueError("The training roots disagree with the episode split")
        use = np.isin(data["episode_id"], sorted(ids))
        depths = touch_heights(data["observation"][use])
    stats = Counter()
    values = frame_summary(depths, stats)
    info = dict(episodes=len(ids), split_counts=dict(Counter(row["split"] for row in rows)),
                split_sha256=sha(manifest), prepared_sha256=sha(source), **summarize(values, stats))
    return values, info


def expanded_frames(directory, declared):
    manifest = checked(directory / "manifest.json", declared, "data/expanded/manifest.json")
    rows = json.loads(manifest.read_bytes())["records"]
    train = [row for row in rows if row["assigned_split"] == "train"]
    fingers_by_object = defaultdict(set)
    for row in train:
        fingers_by_object[row["object_parent"]].add(row["finger"])
    # ExpandedWindows.choose can only train single_all on objects supplying all
    # four finger pools. Its pools can contain different physical trials.
    usable = {obj for obj, fingers in fingers_by_object.items() if fingers == {0, 1, 2, 3}}
    groups = defaultdict(list)
    for row in train:
        if row["object_parent"] in usable:
            groups[row["source_sha256"], row["lane"]].append(row)
    stats, lane_counts, arrays = Counter(), Counter(), []
    inputs = hashlib.sha256()
    for (source, lane), group in sorted(groups.items()):
        depth = np.full((group[0]["sampled_frames"], 4), np.nan)
        seen = set()
        for row in sorted(group, key=lambda r: r["finger"]):
            finger = row["finger"]
            if (finger in seen or row["sampled_frames"] != len(depth)
                    or row["record_key"] != f"{source}:{lane}:{finger}"):
                raise ValueError("Ambiguous synchronous finger lanes")
            seen.add(finger)
            key = "data/expanded/" + row["shard"]
            file = checked(directory / row["shard"], declared, key)
            with np.load(file, allow_pickle=False) as shard:
                if int(shard["finger"]) != finger:
                    raise ValueError(f"Finger mismatch at {file}")
                depth[:, finger] = touch_heights(shard["observation"])
            inputs.update(f"{row['record_key']} {declared[key]}\n".encode("ascii"))
        lane_counts[len(seen)] += 1
        arrays.append(frame_summary(depth, stats))
    values = np.concatenate(arrays)
    info = dict(split_counts=dict(Counter(row["assigned_split"] for row in rows)),
                usable_finger_shards=sum(map(len, groups.values())), source_rollouts=len({source for source, _ in groups}),
                physical_lanes=len(groups), lanes_by_recorded_finger_count=dict(sorted(lane_counts.items())),
                excluded_train_objects=sorted(set(fingers_by_object)-usable), used_objects=sorted(usable),
                manifest_sha256=sha(manifest), verified_shard_set_sha256=inputs.hexdigest(), **summarize(values, stats))
    return values, info


def calibrate(mustard, expanded, integrity):
    if sha(integrity) != INTEGRITY_SHA256:
        raise ValueError("Expected the published FingerPolicy_Data training integrity manifest")
    declared = json.loads(integrity.read_bytes())
    old, old_info = mustard_frames(mustard, declared)
    new, new_info = expanded_frames(expanded, declared)
    # Integer masses implement a 50/50 mixture of the two empirical CDFs.
    heights = np.concatenate((old, new))
    mass = np.repeat([len(new), len(old)], [len(old), len(new)]).astype(np.int64)
    order = np.argsort(heights, kind="stable")
    cumulative = np.cumsum(mass[order])
    lower, median, upper = heights[order][np.searchsorted(cumulative, PROBABILITIES*cumulative[-1])]
    for values, info in ((old, old_info), (new, new_info)):
        info["fraction_inside_band"] = float(((values >= lower) & (values <= upper)).mean())
    return dict(lower_m=float(lower), upper_m=float(upper), target_m=float(median), axis_palm=[0, 0, -1],
                frame="palm_lower", statistic="Within one physical lane and 50 ms frame, median of -contact_position_palm.z for fingers with contact_valid=1.",
                source_weight={"mustard222": 0.5, "expanded": 0.5}, probabilities=PROBABILITIES.tolist(),
                quantile_rule="Generalized inverse of the equal-source empirical CDF. Each unique measured contact frame within its source has equal mass.",
                sample_hz=20, checkpoint_sha256=CHECKPOINT, release_integrity_sha256=INTEGRITY_SHA256,
                release_folder="https://drive.google.com/drive/folders/1NNCVvIcW-SkmGx_RhgbIre4rlQlORIgY",
                mustard222=old_info, expanded=new_info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mustard", "expanded", "integrity"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("height_calibration.json"))
    args = parser.parse_args()
    record = calibrate(args.mustard, args.expanded, args.integrity)
    data = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    # The released calibration is byte-pinned in each run, including its CRLF line endings.
    args.output.write_bytes(data.replace("\n", "\r\n").encode("utf-8"))
    print(f"{record['lower_m']:.9f} to {record['upper_m']:.9f} m; median {record['target_m']:.9f} m")


if __name__ == "__main__":
    main()
