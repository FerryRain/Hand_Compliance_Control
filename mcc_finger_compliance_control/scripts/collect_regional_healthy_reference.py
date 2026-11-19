"""Production collection with the validated contact-normal Jacobian mapping.

Only the regulated motion profile changes. Healthy initialization, force
targets, actuator limits, timing and contact arbitration remain unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import h5py


def jacobian_force_config(config, profile, original):
    value = original(config, profile)
    return replace(value, normal_synergy_control=False) if profile == 'regulated' else value


def main():
    import collection_force_regulation as regulation
    from collect_healthy_reference import main as collect
    output = Path(sys.argv[sys.argv.index('--output') + 1]) if '--output' in sys.argv else None
    original = regulation.collection_force_config
    regulation.collection_force_config = lambda config, profile: jacobian_force_config(config, profile, original)
    try:
        collect()
    finally:
        regulation.collection_force_config = original
        if output is not None:
            metadata = dict(collection_normal_mapping='contact_normal_jacobian',
                collection_normal_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            report = output / 'summary.json'
            if report.exists():
                value = json.loads(report.read_text())
                value.update(metadata)
                report.write_text(json.dumps(value, indent=2) + '\n')
            data = output / 'collected.h5'
            if data.exists():
                with h5py.File(data, 'r+') as handle:
                    handle.attrs.update(metadata)


if __name__ == '__main__':
    main()
