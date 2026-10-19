"""Check that the Git index contains every published artifact with exact bytes."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'Module/generated/whole_hand_surface_run10_20261019'


def main():
    catalog = json.loads((ROOT / 'Module/evidence/2026-10-19_I04_WHOLE_HAND_SURFACE.json').read_text(encoding='utf-8'))
    expected = {item['path']: item['sha256'] for item in catalog['files']}
    expected.update(json.loads((CASE / 'manifest.json').read_text())['source_hashes'])
    assets = json.loads((ROOT / 'Module/evidence/whole_hand_release_assets_20261019.json').read_text(encoding='utf-8'))
    expected.update({item['path']: item['sha256'] for item in assets['assets']})
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT)
    index = {}
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        attributes, path = entry.split(b'\t', 1)
        mode, oid, stage = attributes.split()
        assert stage == b'0', 'Unmerged index entry'
        index[path.decode('utf-8')] = oid
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    lfs = {}
    try:
        for name, expected_hash in sorted(expected.items()):
            if name not in index:
                raise AssertionError(f'Published file missing from Git index: {name}')
            process.stdin.write(index[name] + b'\n')
            process.stdin.flush()
            header = process.stdout.readline().split()
            assert len(header) == 3 and header[1] == b'blob', (name, header)
            data = process.stdout.read(int(header[2]))
            assert process.stdout.read(1) == b'\n'
            if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                fields = dict(line.split(' ', 1) for line in data.decode().strip().splitlines()[1:])
                actual = fields['oid'].removeprefix('sha256:')
                assert int(fields['size']) == (ROOT / name).stat().st_size, name
                lfs[actual] = int(fields['size'])
            else:
                actual = sha256(data).hexdigest()
            assert actual == expected_hash, ('Git index bytes/hash differ from native evidence', name)
    finally:
        process.stdin.close()
        process.wait()
    assert process.returncode == 0
    print(json.dumps(dict(status='PASSED_PUBLISH_INDEX', files_checked=len(expected),
        unique_lfs_objects=len(lfs), unique_lfs_bytes=sum(lfs.values()),
        native_chunks=len(json.loads((CASE / 'summary.json').read_text())['trace_chunks'])), indent=2))


if __name__ == '__main__':
    main()
