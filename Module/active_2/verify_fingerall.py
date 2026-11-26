"""Verify the 30 Active-3 recordings and their primary complete-hand wrist comparison."""

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

from . import report as base
from . import report_extended as extended
from . import report_fingerall as study
from .verify_extended import open_runs, verify_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=base.ROOT_OUT / "runs", help="Original 15 trajectories ZIP or runs")
    parser.add_argument("--extended", type=Path, default=extended.ROOT_OUT / "runs", help="Prior 10 trajectories ZIP or runs")
    parser.add_argument("--new", type=Path, default=study.ROOT_OUT / "runs", help="New five trajectories ZIP or runs")
    parser.add_argument("--report", type=Path, help="Also verify the combined results.json and trials.csv")
    parser.add_argument("--geometry", action="store_true", help="Recompute coverage and endpoint mesh scores independently")
    args = parser.parse_args()
    with ExitStack() as stack:
        sources = {p: open_runs(stack, p) for p in {args.base, args.extended, args.new}}
        rows = study.verified_rows(sources[args.base], sources[args.extended], sources[args.new], geometry=args.geometry)
    result = study.summarize(rows)
    if args.report:
        verify_tables(args.report, rows, result)
    fields = ("c25", "f5", "ge2", "ge3", "precision5", "recall5")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("object", "method", *fields))
    for row in rows:
        writer.writerow((row["object"], row["method"], *(row[k] for k in fields)))
    for k, mean in result["macro_mean"].items():
        writer.writerow(("mean", k, *(mean[key] for key in fields)))


if __name__ == "__main__":
    main()
