"""按时序反转已采集的轨迹 H5,得到反向运动的数据(如 body→cap 反转为 cap→body 恢复数据)。

时间反转规则:
- 所有时间维 (T, ...) 数组沿 T 反序;
- 速度/角速度/力/速度目标等字段取负(时间反转对称性);
- 姿态 quaternion / 位置 / 法线 / 接触标记等保持原值;
- 标量 attr(如 planner 参数、contact_threshold)原样保留。

用法:
    python reverse_collected_trajectory.py \
        --input  mcc_finger_compliance_control/data/trajectories/body_to_cap_mer270_A110_p12_i40_test.h5 \
        --output mcc_finger_compliance_control/data/trajectories/cap_to_body_mer270_A110_p12_i40_reversed.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# 需要取负的时间反转字段(含 vel/force/speed_target 关键词;qvel 由 "vel" 匹配)
NEGATE_KEYWORDS = ("vel", "force", "speed_target")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    with h5py.File(inp, "r") as src, h5py.File(out, "w") as dst:
        for k in src.keys():
            item = src[k]
            if isinstance(item, h5py.Group):
                print(f"skip group: {k}")
                continue
            data = item[()]
            if data.ndim >= 1 and data.shape[0] == 2500:
                lk = k.lower()
                if any(w in lk for w in NEGATE_KEYWORDS) and np.issubdtype(data.dtype, np.floating):
                    data = -data[::-1]
                    note = "negate+reverse"
                else:
                    data = data[::-1]
                    note = "reverse"
            elif item.attrs:
                print(f"keep scalar-with-attrs: {k}")
            dst.create_dataset(k, data=data, compression="gzip")
            dst[k].attrs.update({a: item.attrs[a] for a in item.attrs})
        dst.attrs.update({a: src.attrs[a] for a in src.attrs})
        dst.attrs["time_reversed_from"] = str(inp.name)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
