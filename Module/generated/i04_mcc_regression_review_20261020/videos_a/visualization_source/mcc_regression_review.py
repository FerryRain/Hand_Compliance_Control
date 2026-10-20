"""Read-only follow-up of the archived MCC trial; write to a separate folder.

The normal-speed render reuses the exact published renderer, changing only its
destination paths. Archived manifests, source snapshots and videos stay intact.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path

from . import matched_mcc_review as review


def render_complete(case: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    source = inspect.getsource(review.render)
    replacements = {
        "video = case/(stem+'.mp4')": "video = destination/(stem+'.mp4')",
        "canvas.save(case/filename)": "canvas.save(destination/filename)",
        "write(case/('video_manifest.json' if full else f'video_{stem}.json'), result)":
            "write(destination/f'video_{stem}.json', result)",
        "source = case/'visualization_source'; source.mkdir(exist_ok=True)":
            "source = destination/'visualization_source'; source.mkdir(exist_ok=True)",
        "(source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())":
            "(source/'matched_mcc_review.py').write_bytes(upstream_source.read_bytes())",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    name = f"i04_{review.read(case/'manifest.json')['method'].lower()}_complete_1x.mp4"
    if (output/name).exists():
        raise FileExistsError(output/name)
    namespace = dict(vars(review), destination=output, upstream_source=Path(review.__file__))
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    result = namespace["render"](case, speed=1.)
    result.update(
        source_case=str(case.resolve()),
        renderer_original_sha256=sha256(Path(review.__file__).read_bytes()).hexdigest(),
        transformation="Only output file locations; same 20fps renderer, native state selection, cameras and labels",
    )
    (output/f"video_{Path(name).stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output/'visualization_source'/'mcc_regression_review.py').write_bytes(Path(__file__).read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_complete(args.case, args.output), indent=2), flush=True)


if __name__ == '__main__':
    main()
