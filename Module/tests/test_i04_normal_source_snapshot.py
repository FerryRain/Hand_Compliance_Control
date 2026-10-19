"""Source provenance checks for the separate normal-search entry only."""
import ast
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Module.i04_geometric_traversal import normal_surface_slide as runner


ROOT = Path(__file__).resolve().parents[2]
ENTRY = 'Module/i04_geometric_traversal/normal_surface_slide.py'


class NormalSourceSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root = ROOT/'.runtime_tmp'
        cls.temporary_root.mkdir(exist_ok=True)

    def test_discovery_filters_namespace_resolved_location_and_python_suffix(self):
        scene = ROOT/'Module/e05_physics/scene.py'
        modules = {
            'Module.z_alias': SimpleNamespace(__file__=str(scene)),
            'unrelated.scene': SimpleNamespace(__file__=str(scene)),
            'Module.external': SimpleNamespace(__file__=str(ROOT.parent/'outside.py')),
            'Module.native': SimpleNamespace(__file__=str(ROOT/'Module/native.pyd')),
            'Module.bytecode': SimpleNamespace(__file__=str(ROOT/'Module/cache.pyc')),
            'Module.namespace': SimpleNamespace(),
            'Module.pending': None,
            'Module.a_scene': SimpleNamespace(__file__=str(scene.parent/'..'/'e05_physics'/'scene.py')),
        }
        result = runner.loaded_project_module_sources(ROOT, modules)
        self.assertEqual(list(result), ['Module.a_scene', 'Module.z_alias'])
        self.assertEqual(set(result.values()), {'Module/e05_physics/scene.py'})

    def test_wrong_expected_entry_hash_stops_before_scene_and_output_creation(self):
        with tempfile.TemporaryDirectory(prefix='i04_source_guard_', dir=self.temporary_root) as temporary:
            output = Path(temporary)/'never_created'
            with patch.object(runner, 'build_scene') as build:
                with self.assertRaisesRegex(RuntimeError, 'Controller changed'):
                    runner.run(output, budget_s=.002, expected_runtime_sha256='0'*64)
            build.assert_not_called()
            self.assertFalse(output.exists())

    def test_native_startup_manifest_snapshots_loaded_dependencies_exactly_once(self):
        # Geometry scheduling is suppressed solely to keep this provenance test
        # short. Acquisition, two native states, file bytes, hashing and manifest
        # writing remain the real runner. This is not traversal evidence.
        with tempfile.TemporaryDirectory(prefix='i04_source_native_', dir=self.temporary_root) as temporary:
            output = Path(temporary)/'case'
            expected = sha256((ROOT/ENTRY).read_bytes()).hexdigest()
            with patch.object(runner.SurfaceStrokePlanner, 'plan', return_value=(None, [])), \
                    redirect_stdout(io.StringIO()):
                runner.run(output, budget_s=.01, expected_runtime_sha256=expected)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            hashes = manifest['source_hashes']
            self.assertEqual(list(hashes), sorted(hashes))
            self.assertEqual(hashes[ENTRY], expected)
            for relative in ('Module/e05_physics/scene.py',
                             'Module/i04_oracle_next_point/surface_graph.py'):
                self.assertIn(relative, hashes)
            for relative, digest in hashes.items():
                frozen = (output/'source_snapshot'/relative).read_bytes()
                self.assertEqual(frozen, (ROOT/relative).read_bytes())
                self.assertEqual(sha256(frozen).hexdigest(), digest)
            actual_files = list((output/'source_snapshot').rglob('*.py'))
            self.assertEqual(len(actual_files), len(hashes))
            policy = manifest['source_snapshot_policy']
            self.assertEqual(policy['explicit_source_count'], 34)
            self.assertFalse(policy['future_dynamic_imports_included'])
            self.assertIn('before build_scene', policy['captured_at'])
            for relative in policy['loaded_module_sources'].values():
                self.assertIn(relative, hashes)

    def test_entry_run_has_no_hidden_project_or_dynamic_import(self):
        # This scope check is deliberate: the manifest describes startup-loaded
        # modules, so a future lazy import in run() needs a provenance decision.
        tree = ast.parse((ROOT/ENTRY).read_text(encoding='utf-8-sig'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        imports = [node for node in ast.walk(run) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports, [])
        dynamic = [node for node in ast.walk(run) if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name) and node.func.id == '__import__' or
            isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')]
        self.assertEqual(dynamic, [])


if __name__ == '__main__':
    unittest.main()
