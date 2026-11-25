"""Launch the shared native executor on its registered initial state."""
import json
from pathlib import Path
from unittest.mock import patch

from Module.palmreach6 import __main__ as release


def run_baseline(obj, method, seconds, out, seed):
    from Module.acitive_02 import route_free_inputs
    from Module.wrist_active import informative_run, tactile_field

    manifest_path = release.FIXTURES / "fixtures.json"
    manifest = json.loads(manifest_path.read_bytes())
    native_seed = manifest["objects"][obj]["seed"]
    if seed == native_seed:
        return release.run(obj, out, seconds, method)
    # The release CLI binds each fixture to its published seed. Override only that
    # experimental parameter at the two native read boundaries, in this trial process.
    manifest["objects"][obj]["seed"] = seed
    manifest["fixtures"][obj + "/start-02"]["seed"] = seed
    def with_seed(original):
        def read(path):
            return manifest if Path(path).resolve() == manifest_path.resolve() else original(path)
        return read
    with (patch.object(informative_run, "read", with_seed(informative_run.read)),
          patch.object(route_free_inputs, "read_json", with_seed(route_free_inputs.read_json)),
          patch.object(informative_run, "seed_for", return_value=seed),
          patch.object(tactile_field, "seed_for", return_value=seed)):
        return release.run(obj, out, seconds, method)
