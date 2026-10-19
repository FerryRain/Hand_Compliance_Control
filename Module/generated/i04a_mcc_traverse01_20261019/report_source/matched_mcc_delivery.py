"""Check and catalog the exact new MCC evidence; verify needs only stdlib."""
from __future__ import annotations
import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
CASES = [ROOT/f'Module/generated/i04{method}_mcc_traverse01_20261019' for method in ('a', 'b')]
COMPARISON = ROOT/'Module/generated/i04_mcc_comparison01_20261019'
CATALOG = ROOT/'Module/evidence/2026-10-19_I04_MATCHED_MCC.json'


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value:
                self.links.append(value)


def page_check(path):
    parser = Links(); parser.feed(path.read_text(encoding='utf-8'))
    count = 0
    for target in parser.links:
        parsed = urlparse(target)
        if parsed.scheme or not parsed.path:
            continue
        resolved = (path.parent/unquote(parsed.path)).resolve()
        assert resolved.is_relative_to(ROOT) and resolved.is_file(), (path, target)
        count += 1
    return dict(page=path.relative_to(ROOT).as_posix(), local_links_checked=count,
                scope='Static HTML reference validation; not browser layout testing')


def qa():
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    results = []
    for case in CASES:
        checked = read(case/'independent_native_audit/audit.json')
        assert checked['status'] == 'PASSED_NATIVE_EVIDENCE_AUDIT'
        for name, expected in checked['source_files'].items():
            assert digest(case/name) == expected, name
        for name, expected in read(case/'manifest.json')['source_hashes'].items():
            assert digest(case/'source_snapshot'/name) == digest(ROOT/name) == expected, name
        destination = case/'delivery_checks'; destination.mkdir(exist_ok=True)
        videos = []
        for manifest in [case/'video_manifest.json', *sorted(case.glob('video_i04_*_segment_*.json'))]:
            meta = read(manifest); video = case/meta['video']
            assert digest(video) == meta['video_sha256']
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
            assert frames == meta['frames'] and abs(seconds-frames/meta['fps']) < .1
            reader = imageio.get_reader(video)
            samples = []
            try:
                for capture in (meta['captures'][0], meta['captures'][2], meta['captures'][-1]):
                    index = int(Path(capture['file']).stem.rsplit('_', 1)[1])
                    decoded = reader.get_data(index)
                    original = np.asarray(Image.open(case/capture['file']).convert('RGB'))
                    assert decoded.shape == original.shape == (700, 2400, 3)
                    mae = float(np.mean(np.abs(decoded.astype(float)-original.astype(float))))
                    assert mae < 10., (video, index, mae)
                    name = f'decoded_{video.stem}_{index:05d}.png'
                    Image.fromarray(decoded).save(destination/name)
                    samples.append(dict(frame=index, time_s=capture['time_s'], encoding_mae=mae,
                                        file='delivery_checks/'+name))
            finally:
                reader.close()
            videos.append(dict(file=video.name, frames=frames, encoded_duration_s=seconds,
                               fps=meta['fps'], sha256=digest(video), decoded_samples=samples))
        result = dict(status='PASSED_ARTIFACT_CHECKS', native_states=checked['native_states'],
                      source_snapshots_checked=len(read(case/'manifest.json')['source_hashes']),
                      page=page_check(case/'review.html'), videos=videos,
                      visual_review='See manual_visual_review.json for human/model image inspection; checks here are automated')
        write(case/'delivery_checks.json', result)
        # Report logic evolved after rendering. Preserve both actual renderer
        # source (visualization_source) and the final report source separately.
        source = case/'report_source'; source.mkdir(exist_ok=True)
        source.joinpath('matched_mcc_review.py').write_bytes(Path(__file__).with_name('matched_mcc_review.py').read_bytes())
        source.joinpath('matched_mcc_delivery.py').write_bytes(Path(__file__).read_bytes())
        results.append(result)
    combined = dict(status='PASSED_ARTIFACT_CHECKS', cases=results, comparison=page_check(COMPARISON/'review.html'))
    write(COMPARISON/'delivery_checks.json', combined)
    return dict(status=combined['status'], native_states=sum(item['native_states'] for item in results),
                videos=sum(len(item['videos']) for item in results))


def artifact_paths():
    paths = []
    for case in [*CASES, COMPARISON]:
        referenced = set()
        if case in CASES:
            for item in [case/'video_manifest.json', *case.glob('video_i04_*_segment_*.json')]:
                referenced.update(capture['file'] for capture in read(item)['captures'])
        for path in sorted(case.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.name == 'comparison_horizon.json':
                assert read(path)['monitor_result'] == 'NOT_APPLIED_RUN_ALREADY_TERMINATED'
                continue
            # Earlier overwritten render samples are not part of the final video.
            if path.parent == case and path.name.startswith('i04_') and path.suffix == '.png' and path.name not in referenced:
                continue
            paths.append(path)
    return paths


def catalog():
    assert read(COMPARISON/'delivery_checks.json')['status'] == 'PASSED_ARTIFACT_CHECKS'
    files = [dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest(path))
             for path in artifact_paths()]
    result = dict(protocol='i04.ab.whole_hand.v2.mcc.1', files=files, file_count=len(files),
                  bytes=sum(item['bytes'] for item in files),
                  scope='Complete A trial and same-version B trial; all original 2ms chunks, reports, videos and audits; neither completed coverage',
                  support_assets_catalog='Module/evidence/whole_hand_release_assets_20261019.json',
                  historical_cooperative_geometry_seed_catalog='Module/evidence/2026-10-19_I04_WHOLE_HAND_SURFACE.json')
    write(CATALOG, result)
    return {key: value for key, value in result.items() if key != 'files'}


def verify():
    catalog = read(CATALOG)
    for item in catalog['files']:
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), item['path']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], item['path']
    return dict(status='PASSED_NEW_MCC_DOWNLOAD_INTEGRITY', files=len(catalog['files']), bytes=catalog['bytes'],
                scope='Artifact byte integrity; physics replay is a separate audit command')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('qa', 'catalog', 'verify'))
    args = parser.parse_args()
    print(json.dumps({'qa': qa, 'catalog': catalog, 'verify': verify}[args.command](), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
