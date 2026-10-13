"""Reproducible paired MCC-only new I04 benchmark and result manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

import mujoco
import numpy as np
import scipy
import trimesh

from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point.surface_graph import BunnySurfaceGraph
from .contracts import ExperimentConfig, PROTOCOL_VERSION, SURFACE_VERSION, json_digest
from .physics import make_plant
from .runner import prepare_initial_state, run_cell

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('smoke', 'formal'), default='smoke')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--cells', choices=('A-MCC', 'B-MCC'), nargs='+', default=['A-MCC', 'B-MCC'])
    parser.add_argument('--output', type=Path, default=ROOT / 'Module/generated/new_i04_mcc')
    args = parser.parse_args()
    seeds = args.seeds or ([7] if args.profile == 'smoke' else [7, 11, 19])
    duration = args.duration or (6.0 if args.profile == 'smoke' else 60.0)
    output = args.output.resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment {output}; choose another output')
    output.mkdir(parents=True, exist_ok=True)
    print('Building canonical full Bunny mesh and 23-DoF SDF plant...', flush=True)
    bunny = canonical_bunny_heightfield()
    mesh_path = bunny.export_visual_mesh(output / 'canonical_bunny.obj')
    config = ExperimentConfig(duration_s=duration)
    graph = BunnySurfaceGraph(bunny.mesh, coverage_radius_m=config.coverage_radius_m)
    h = make_plant(mesh_path, config.dt_s)
    manifest = {
        'protocol': PROTOCOL_VERSION, 'profile': args.profile, 'surface': SURFACE_VERSION,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seeds': seeds, 'cells': args.cells, 'config': asdict(config), 'dpref': 'DEFERRED_BY_USER',
        'environment': {'python': sys.version, 'python_executable': sys.executable,
            'platform': platform.platform(), 'mujoco': mujoco.__version__, 'numpy': np.__version__,
            'scipy': scipy.__version__, 'trimesh': trimesh.__version__,
            'physics_backend': 'MuJoCo native CPU; GPU OpenGL rendering'},
        'geometry': {'source_sha256': bunny.source_sha256,
            'canonical_mesh_sha256': sha256(mesh_path.read_bytes()).hexdigest(),
            'vertices': len(graph.vertices_m), 'triangles': len(graph.faces),
            'required_goals': graph.required_goal_count, 'whole_mesh_area_m2': graph.total_area_m2,
            'collision': 'native nonconvex mesh SDF', 'gravity_m_s2': 0.0},
        'method_boundaries': {'oracle_contact_labels': False, 'A_fingertip_targets': False,
            'B_fingertip_targets': True, 'contact_task_vetoes': False,
            'continuous_contact_hard_planning_constraint': False,
            'privileged_oracle_geometric_continuation': True,
            'coverage_source': 'measured physical pad contacts; geodesic area union'},
        'code_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    for relative in ('Module/fr3_leap/model.py', 'Module/i01_bunny_physics/surface.py',
                     'Module/i04_oracle_next_point/surface_graph.py',
                     'Module/module_2_fingertip_mcc/controller.py',
                     'Module/e05_physics/scene.py', 'Module/I04_GEOMETRIC_MCC_PROTOCOL.md'):
        manifest['code_hashes'][relative] = sha256((ROOT / relative).read_bytes()).hexdigest()
    manifest['model_xml_sha256'] = sha256(h.xml.encode()).hexdigest()
    manifest['planning_clock'] = 'Synchronous offline planning; wall latency reported separately from simulated physics time'
    manifest['manifest_digest'] = json_digest(manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'model.xml').write_text(h.xml, encoding='utf-8')
    rows = []
    for seed in seeds:
        episode_config = ExperimentConfig(seed=seed, duration_s=duration)
        print(f'Preparing identical acquisition state for seed {seed}...', flush=True)
        state, audit = prepare_initial_state(h, graph, episode_config)
        initial_path = output / 'initial_states'
        initial_path.mkdir(exist_ok=True)
        np.savez_compressed(initial_path / f'seed_{seed}.npz', **state)
        (initial_path / f'seed_{seed}.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print('Initial state:', json.dumps(audit), flush=True)
        if audit['initial_contact_count'] < 1 or audit['initial_non_tip_contacts']:
            raise RuntimeError('Invalid shared contact initialization; treatment not started')
        for cell in args.cells:
            row = run_cell(h, graph, episode_config, cell, state, audit, output / f'{cell}_seed{seed}')
            rows.append(row)
            print('RESULT', json.dumps({k: row[k] for k in ('cell', 'seed', 'stop_reason',
                  'executed_post_acquisition_s', 'mean_contacts', 'contact_continuity',
                  'final_covered_area_fraction', 'plan_count')}), flush=True)
    summary = {'profile': args.profile, 'protocol': PROTOCOL_VERSION, 'status': 'EVALUATED',
               'dpref': 'DEFERRED_BY_USER', 'episodes': rows}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Saved experiment: {output}', flush=True)


if __name__ == '__main__':
    main()
