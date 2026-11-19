"""Opt-in diagnostic housing hulls; no defaults or contact exclusions change."""
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh


def verify_housing_model(model, folder):
    """Check the compiled plant, not just a request flag or planner model."""
    folder = Path(folder)
    expected = 3 * sum(len(json.loads((folder/k/'manifest.json').read_text())['parts'])
                       for k in ('mcp', 'dip'))
    active = [model.geom(i).name or '' for i in range(model.ngeom)
              if model.geom_contype[i] or model.geom_conaffinity[i]]
    parts = [n for n in active if '_housing_collision_' in n]
    originals = {'mcp_joint_geom', 'dip_geom', 'mcp_joint_2_geom', 'dip_2_geom',
                 'mcp_joint_3_geom', 'dip_3_geom'}
    remaining = [n for n in active if n.split('/')[-1] in originals]
    if len(parts) != expected or remaining:
        raise ValueError(f'Housing model mismatch: expected{expected} active parts, '
                         f'found{len(parts)}, old active hulls={remaining}')
    return dict(expected_parts=expected, active_parts=len(parts), old_active_hulls=remaining,
                folder=str(folder.resolve()), verified=True)


def install_request_housing(task):
    """Shared capture hook; restore it on exit. Does not alter MCC/materials."""
    import deploy_b_gpu_serial_parity as deployment
    folder = task.get('diagnostic_housing_collision')
    planned = (task.get('optimizer_teacher') or {}).get('housing_collision')
    mismatch = bool(planned and (not folder or Path(planned).resolve() != Path(folder).resolve()))
    if mismatch and not task.get('diagnostic_allow_housing_mismatch', False):
        raise ValueError('Teacher and physical plant housing collision models differ')
    original = deployment.replay_env_cfg
    if folder:
        def configured(*args, **kwargs):
            result = original(*args, **kwargs)
            source = result.scene.entities['robot'].spec_fn
            result.scene.entities['robot'].spec_fn = lambda: replace_housing_collision(source(), folder)
            result.sim.njmax = max(result.sim.njmax or 0, 2048)
            result.sim.nconmax = max(result.sim.nconmax or 0, 1000)
            return result
        deployment.replay_env_cfg = configured
    return lambda: setattr(deployment, 'replay_env_cfg', original)


def replace_housing_collision(spec,folder):
    folder=Path(folder);assets={}
    for kind in ['mcp','dip']:
        manifest=json.loads((folder/kind/'manifest.json').read_text())
        names=[]
        for part in manifest['parts']:
            import hashlib
            path=folder/kind/part['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=part['sha256']:
                raise ValueError('Housing candidate asset changed')
            mesh=trimesh.load(path,force='mesh',process=False)
            name='diagnostic_housing_'+path.stem
            spec.add_mesh(name=name,uservert=np.asarray(mesh.vertices).ravel(),
                          userface=np.asarray(mesh.faces,np.int32).ravel())
            names.append(name)
        assets['mcp_joint_mesh' if kind=='mcp' else 'dip_mesh']=names
    replaced=[]
    for source in list(spec.geoms):
        if source.meshname not in assets or not (source.contype or source.conaffinity):continue
        for i,mesh in enumerate(assets[source.meshname]):
            g=source.parent.add_geom(name=source.name+f'_housing_collision_{i:02d}',
                type=mujoco.mjtGeom.mjGEOM_MESH,meshname=mesh,pos=tuple(source.pos),quat=tuple(source.quat),
                mass=0.,rgba=(0.,0.,0.,0.),contype=int(source.contype),conaffinity=int(source.conaffinity),
                condim=int(source.condim),group=3)
            for attr in ['friction','solref','solimp']:getattr(g,attr)[:]=getattr(source,attr)
            for attr in ['priority','solmix','margin','gap']:setattr(g,attr,getattr(source,attr))
        source.contype=0;source.conaffinity=0;replaced.append(source.name)
    # The three long fingers share these two meshes. The thumb has distinct
    # thumb_pip/thumb_dip assets and is not silently assigned long-finger hulls.
    if len(replaced)!=6:raise ValueError(f'Expected6 long-finger housing geoms,got{replaced}')
    return spec
