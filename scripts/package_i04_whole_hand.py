"""Build the published result inventory without rewriting its native evidence."""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'Module/generated/whole_hand_surface_run10_20261019'
EVIDENCE = ROOT / 'Module/evidence'


def digest(path):
    result = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    assets = []
    for node in ET.parse(CASE / 'model.xml').findall('./asset/*[@file]'):
        original = Path(node.get('file'))
        if 'whole_hand_surface_run04_20261018' in original.parts:
            base = original.parts.index('whole_hand_surface_run04_20261018')
            actual = CASE.joinpath(*original.parts[base + 1:])
        else:
            actual = original
        assert original.is_file() and digest(original) == digest(actual), str(original)
        assets.append(dict(name=node.get('name'), original_file=node.get('file'),
                           path=actual.relative_to(ROOT).as_posix(), sha256=digest(actual)))
    (EVIDENCE / 'whole_hand_release_assets_20261019.json').write_text(json.dumps(dict(
        scope='Only path relocation; same mesh/texture bytes, immutable original model.xml',
        assets=assets), ensure_ascii=False, indent=2), encoding='utf-8')
    paths = set(p for p in CASE.rglob('*') if p.is_file())
    for relative in (
        'Module/generated/whole_hand_geometry_touch02_20261018/geometry.npz',
        'Module/generated/whole_hand_geometry_touch02_20261018/survey.json',
        'Module/generated/mustard_natural_fresh02_20261018/independent_native_audit/audit.json',
        'Module/generated/mustard_natural_fresh02_20261018/independent_native_audit/verified_preview.npz',
        'Module/generated/belly_contact_audit_20261017/audit_native_probe.py',
    ):
        paths.add(ROOT / relative)
    source = ROOT / 'assets_external/ycb/collision/006_mustard_bottle/vhacd_256_scaled2p8_objstage'
    paths.update(p for p in source.iterdir() if p.is_file() and p.name != 'preview.xml')
    files = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(paths)]
    audit = json.loads((CASE / 'independent_native_audit/audit.json').read_text())
    metrics = json.loads((CASE / 'whole_hand_metrics.json').read_text())
    summary = json.loads((CASE / 'summary.json').read_text())
    result = dict(case=CASE.relative_to(ROOT).as_posix(), status='PUBLISHED_CONTENT_CATALOG',
        artifact_bytes=sum(f['bytes'] for f in files), artifact_files=len(files), files=files,
        result=dict(stop_reason=summary['stop_reason'], executed_s=summary['executed_s'],
            native_states=audit['native_states'], covered_vertices=audit['covered_vertices'],
            covered_goals=audit['covered_goals'], coverage_definition='25 mm mesh-edge geodesic neighborhood of qualified native finger-belly contact, not pressure footprint',
            continuous_replay=True, physical_initializations=1, fresh_single_version=False,
            controller_join_times_s=[2260., 4102.92, 5273.],
            longest_whole_hand_safe_contact_gap_s=metrics['longest_safe_gap_s'][-1],
            dp='DEFERRED_BY_USER', physical_scope='fixed Mustard, zero gravity, MCC simulation'),
        data_storage='MP4 and large native NPZ files use Git LFS; all native steps retained',
        seed_preview_scope='Old single-leader preview is geometry-only; no old coverage is merged')
    (EVIDENCE / '2026-10-19_I04_WHOLE_HAND_SURFACE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(artifact_files=len(files), artifact_bytes=result['artifact_bytes'], model_assets=len(assets)), indent=2))


if __name__ == '__main__':
    main()
