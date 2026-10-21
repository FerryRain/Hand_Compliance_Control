"""Portable A delivery and byte verification; published B remains read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .a_frozen_b_mcc import BASELINE
from .b_full_portable import register, smoke, initialize_worker
from .b_full_video import digest, read
from .published_whole_hand import ROOT


CATALOG = ROOT/'Module/evidence/2026-10-21_I04_FROZEN_B_AB.json'


def verify():
    catalog = read(CATALOG)
    total = 0
    for item in catalog['files']:
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT)
        assert path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], item['path']
        total += item['bytes']
    from .b_full_portable import verify_download
    return dict(status='PASSED_FROZEN_B_AB_DOWNLOAD_INTEGRITY', files=len(catalog['files']), bytes=total,
                original_B=verify_download(BASELINE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'assets', 'smoke', 'audit', 'video', 'report'))
    parser.add_argument('--case', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'verify':
        result = verify()
    else:
        if args.case is None or args.case.resolve() == BASELINE.resolve():
            parser.error('Use --case for the new A case; B is immutable')
        assets = register(args.case)
        if args.action == 'assets':
            result = assets
        elif args.action == 'smoke':
            result = smoke(args.case)
            result['status'] = 'PASSED_RELOCATED_A_NATIVE_SMOKE'
        elif args.action == 'audit':
            from . import ab_frozen_b_delivery as delivery
            delivery.b_full_mcc_audit.portable_guard_initializer = initialize_worker
            result = delivery.audit(args.case, args.workers, output_name='downloaded_native_reaudit')
        elif args.action == 'video':
            from .ab_frozen_b_delivery import video
            result = video(args.case, output_name='downloaded_complete_videos')
        else:
            if args.output is None:
                parser.error('--output is required for report')
            register(BASELINE)  # Existing verified catalog; no B file write.
            from .ab_frozen_b_report import report
            result = report(args.case, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
