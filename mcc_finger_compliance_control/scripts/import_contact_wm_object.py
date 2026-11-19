"""Import a pre-decomposed household asset without changing its collision shape.

The same rigid transform and scale are applied to the visual mesh and every
convex piece. Original assets remain untouched; provenance distinguishes CAD
from scanned meshes. This is geometry preparation, not physics admission.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--visual', default='visual_scaled.obj')
    p.add_argument('--parts', default='collision_part_*.obj')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--object-id', required=True)
    p.add_argument('--parent-object-id', required=True)
    p.add_argument('--source-type', choices=['scanned_mesh', 'CAD'], required=True)
    p.add_argument('--baked-scale', type=float, required=True)
    p.add_argument('--input-units', choices=['metres'], required=True)
    p.add_argument('--rotate-y-deg', type=float, default=-90.)
    a = p.parse_args()
    source = a.source.resolve(); root = a.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    if a.baked_scale <= 0 or not np.isfinite(a.baked_scale):
        raise ValueError('Invalid source scale')
    files = [source/a.visual] + sorted(source.glob(a.parts))
    if len(files) < 2:
        raise ValueError('Visual mesh and explicit convex pieces required')
    rotation = Rotation.from_euler('y', a.rotate_y_deg, degrees=True).as_matrix()
    folder = root/'objects'/a.object_id
    folder.mkdir(parents=True)
    records = []; meshes = []
    for i, path in enumerate(files):
        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
            raise ValueError('Invalid mesh: '+str(path))
        # Scale is already baked into source; never multiply by it again.
        vertices = mesh.vertices.copy()
        mesh.vertices = vertices @ rotation.T
        name = 'surface.obj' if i == 0 else f'collision_{i-1:03d}.obj'
        mesh.export(folder/name)
        saved = trimesh.load(folder/name, process=False)
        error = float(np.max(np.abs(saved.vertices - vertices @ rotation.T)))
        if error > 1e-7 or not np.array_equal(saved.faces, mesh.faces):
            raise ValueError('Geometry changed during export')
        records.append(dict(source=str(path), source_sha256=digest(path), output=name,
                            output_sha256=digest(folder/name), transform_error_m=error))
        meshes.append(saved)
    mesh = meshes[0]
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Invalid closed visual surface')
    row = dict(object_id=a.object_id, parent_object_id=a.parent_object_id,
               family='large_household', category='household_mesh_and_CAD',
               source_type=a.source_type, split='train', backend='convex_parts',
               initialization='under_palm', units='metres',
               collision_parts=[r['output'] for r in records[1:]],
               mesh_sha256=digest(folder/'surface.obj'),
               source_baked_scale=a.baked_scale, applied_scale=1.,
               source_to_collection_rotation=rotation.tolist(),
               extents_m=mesh.extents.tolist(), anchor=[0., 0., float(mesh.bounds[1,2])],
               assets=records, wm_contact_rate_cutoff=None,
               admission='geometry_transform_verified; physical_review_pending')
    for name in ('README.md', 'manifest.json'):
        if (source/name).exists():
            (folder/('source_'+name)).write_bytes((source/name).read_bytes())
    (folder/'geometry.json').write_text(json.dumps(row, indent=2))
    (root/'object_manifest.json').write_text(json.dumps(dict(
        schema='contact_wm_imported_object_v1', objects=[row],
        status='development_only; group all parent variants together; not heldout'), indent=2))
    print(json.dumps(dict(object=row['object_id'], extents_m=row['extents_m'],
                         collision_parts=len(records)-1,
                         max_transform_error_m=max(r['transform_error_m'] for r in records)), indent=2))


if __name__ == '__main__':
    main()
