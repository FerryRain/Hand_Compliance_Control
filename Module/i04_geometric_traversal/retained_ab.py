"""Entry point for the retained, non-Active I04 A and palm-safe B experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'A': ROOT / 'Module/generated/i04a_frozen_b01_20261021',
    'B': ROOT / 'Module/generated/i04b_palm_recovery02_20261027',
}
CATALOG = ROOT / 'Module/evidence/2026-11-19_I04_AB_RETAINED_FILES.json'
CODE_CATALOG = ROOT / 'Module/evidence/2026-11-19_I04_AB_CODE_FILES.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(code_only=False):
    data = json.loads((CODE_CATALOG if code_only else CATALOG).read_text(encoding='utf-8'))
    errors = []
    for name, expected in data['sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append({'file': name, 'error': 'missing or outside repository'})
        elif digest(path) != expected:
            errors.append({'file': name, 'error': 'SHA256 mismatch'})
    if errors:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))
    return {'status': 'PASSED_RETAINED_FILE_INTEGRITY', 'files': len(data['sha256']),
            'scope': 'code_and_runtime_inputs' if code_only else 'full_retained_archive',
            'full_physics_reaudit': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'replay', 'verify', 'smoke', 'audit'))
    parser.add_argument('--method', choices=('A', 'B'), default='B')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=9000.)
    parser.add_argument('--speed', type=float, default=12.)
    parser.add_argument('--states', type=int, default=500)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output-name', default='retained_native_reaudit')
    parser.add_argument('--code-only', action='store_true', help='verify code/runtime inputs without Drive history or videos')
    args = parser.parse_args()
    case = CASES[args.method]
    if args.action == 'run':
        if args.output is None or args.output.exists() or args.seconds <= 0:
            parser.error('run requires a new --output directory and positive --seconds')
        from .retained_seed_inputs import install
        from .published_whole_hand import install_asset_adapter
        install()
        install_asset_adapter()
        if args.method == 'A':
            from .a_frozen_b_mcc import FrozenBAMCC
            result = FrozenBAMCC(args.output, args.seconds).run()
        else:
            from .palm_safe_recovery import PalmRecoveryTraversal
            result = PalmRecoveryTraversal(args.output, args.seconds).run()
        result = {k: v for k, v in result.items() if k != 'trace_chunks'}
    elif args.action == 'replay':
        sys.argv = ['trajectory_replay.py', '--case', str(case), '--speed', str(args.speed)]
        runpy.run_path(str(Path(__file__).with_name('trajectory_replay.py')), run_name='__main__')
        return
    elif args.action == 'verify':
        result = verify(args.code_only)
    elif args.action == 'smoke':
        from .b_full_portable import smoke
        result = smoke(case, args.states)
        result['method'] = args.method
    else:
        if (case / args.output_name).exists():
            parser.error('audit requires a new --output-name to preserve existing evidence')
        if Path(args.output_name).name != args.output_name or args.output_name in ('.', '..'):
            parser.error('--output-name must be one directory name')
        if args.method == 'A':
            from .b_full_portable import register, initialize_worker
            from . import ab_frozen_b_delivery as delivery
            register(case)
            delivery.b_full_mcc_audit.portable_guard_initializer = initialize_worker
            result = delivery.audit(case, args.workers, output_name=args.output_name)
        else:
            from .palm_safe_audit import audit
            result = audit(case, workers=args.workers, output_name=args.output_name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
