"""Use the frozen B audit/renderer for A; never write into the B case."""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path

from .a_frozen_b_mcc import BASELINE, PROTOCOL
from . import b_full_mcc_audit, b_full_video


def require_a(case):
    case = Path(case).resolve()
    if case == BASELINE.resolve():
        raise ValueError('The successful B case is immutable')
    manifest = json.loads((case/'manifest.json').read_text())
    assert manifest['protocol'] == PROTOCOL and manifest['method'] == 'A-MCC'
    return case


def audit(case, workers=4, follow=False, output_name='independent_native_audit'):
    case = require_a(case)
    namespace = dict(vars(b_full_mcc_audit), PROTOCOL=PROTOCOL, __file__=__file__)
    source = inspect.getsource(b_full_mcc_audit.audit)
    exec(compile(source, __file__, 'exec'), namespace)
    result = namespace['audit'](case, workers, follow, output_name)
    result['comparison_adapter_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    result['same_frozen_b_physics_coverage_and_guards'] = True
    (case/output_name/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def video(case, follow=False, output_name='complete_videos'):
    case = require_a(case)
    source = inspect.getsource(b_full_video.render)
    assert source.count('b_complete_') == 2 and source.count('I04B') == 1
    source = source.replace('b_complete_', 'a_complete_').replace('I04B', 'I04A')
    namespace = dict(vars(b_full_video), __file__=__file__)
    exec(compile(source, __file__, 'exec'), namespace)
    result = namespace['render'](case, output_name, follow)
    target = case/output_name/'effective_render.py'
    target.write_text(source, encoding='utf-8')
    result.update(effective_render_sha256=sha256(target.read_bytes()).hexdigest(),
                  frozen_b_renderer_changes='A title and output filename only')
    (case/output_name/'video_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'video'))
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    if args.action == 'audit':
        audit(args.case, args.workers, args.follow)
    else:
        video(args.case, args.follow)
