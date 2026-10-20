"""Independent continuous audit, with the declared 40N/200ms simulation stop.

Only the force-stop predicate and live completion marker differ from the
published independent auditor. Contact/coverage and both geometry guards are
identical. The exact effective auditor and this adapter are archived.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path

from . import whole_hand_parallel_audit as upstream
from .b_full_mcc import PROTOCOL, HARD_FORCE_N, HARD_FORCE_STEPS
from .published_whole_hand import install_asset_adapter, portable_guard_initializer


def audit(case, workers=4, follow=False, output_name='independent_native_audit'):
    case = Path(case).resolve()
    if Path(output_name).name != output_name or output_name in ('.', '..'):
        raise ValueError('A new single-directory output name is required')
    output = case/output_name
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((case/'manifest.json').read_text())
    assert manifest['protocol'] == PROTOCOL
    policy = manifest['force_stop_policy']
    assert policy['stop_n'] == HARD_FORCE_N == 40.
    assert policy['consecutive_samples'] == HARD_FORCE_STEPS == 100
    assert manifest['coverage_force_exclusion_n'] == 8.
    install_asset_adapter()
    source = inspect.getsource(upstream.audit)
    replacements = {
        'high_steps=high_steps+1 if f.max()>12. else 0':
            'high_steps=high_steps+1 if f.max()>40. else 0',
        'assert not unsafe and arm<=.5 and high_steps<10':
            'assert not unsafe and arm<=.5 and high_steps<100',
        "finished=(case/'summary.json').exists()":
            "finished=(case/'run_complete.json').exists() or (not follow and (case/'summary.json').exists())",
        "endpoint=case/'final_state.npz' if (case/'final_state.npz').exists() else case/'checkpoint.npz'":
            "endpoint=case/'checkpoint.npz'",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(vars(upstream), initialize_guard_worker=portable_guard_initializer)
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    result = namespace['audit'](case, workers=workers, follow=follow, output_name=output_name)
    (output/'upstream_auditor.py').write_bytes(Path(upstream.__file__).read_bytes())
    (output/'auditor.py').write_text(source, encoding='utf-8')
    (output/'force_policy_adapter.py').write_bytes(Path(__file__).read_bytes())
    result.pop('sequential_audit_source_sha256', None)
    result.update(
        protocol=PROTOCOL, force_stop_policy=policy, original_8n_coverage_predicate=True,
        audit_source_sha256=sha256((output/'auditor.py').read_bytes()).hexdigest(),
        upstream_auditor_sha256=sha256((output/'upstream_auditor.py').read_bytes()).hexdigest(),
        force_policy_adapter_sha256=sha256((output/'force_policy_adapter.py').read_bytes()).hexdigest(),
    )
    (output/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--output-name', default='independent_native_audit')
    args = parser.parse_args()
    audit(args.case, args.workers, args.follow, args.output_name)
