"""把 body→cap 规划轨迹反转为 cap→body 恢复轨迹(时间倒序)。

恢复数据语义:cap 起始帧为高内收捏取,后续在接触约束下逐步张开回到
body 正常抓握。反转只做时间轴倒序,关键帧判据按帧重排(不改变每帧
本身的判据值),attrs 记录 direction=cap_to_body 与来源。

用法:
  python reverse_plan_time.py \
      --input .../az300_55deg_s802_opt.h5 \
      --output .../cap_to_body_az300_55deg_s802_opt.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def reverse_time(input_path: Path, output_path: Path) -> int:
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
        total_frames = int(source["palm_pose_object"].shape[0])
        knot_frames = None
        if "grasp_keyframe_frame_index" in source:
            knot_frames = np.asarray(source["grasp_keyframe_frame_index"])
        for name, dataset in source.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            value = np.asarray(dataset)
            if name == "grasp_keyframe_frame_index":
                target.create_dataset(
                    name, data=(total_frames - 1) - value[::-1]
                )
            elif (
                value.ndim >= 1
                and value.shape[0] == total_frames
                or (knot_frames is not None and value.shape[0] == len(knot_frames))
            ):
                # (T, ...) 或 (N_knot, ...) 时间序列:整体倒序。
                target.create_dataset(name, data=value[::-1])
            else:
                target.create_dataset(name, data=value)
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["direction"] = "cap_to_body"
        target.attrs["time_reversed_from"] = str(
            source.attrs.get("plan_name", input_path.stem)
        )
        target.attrs["plan_name"] = f"{output_path.stem}"
    return int(total_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = reverse_time(args.input, args.output)
    print(
        f"[REVERSE] {args.input.name} -> {args.output.name} "
        f"({frames} frames, direction=cap_to_body)"
    )


if __name__ == "__main__":
    main()
