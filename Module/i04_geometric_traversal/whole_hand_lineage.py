"""Exact native continuation with an explicit, recursively preserved history."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def restore_verified_prefix(task,parent):
    parent=Path(parent).resolve();out=task.out
    pm=json.loads((parent/'manifest.json').read_text());ps=json.loads((parent/'summary.json').read_text())
    pa=json.loads((parent/'independent_native_audit/audit.json').read_text())
    assert pa['status']=='PASSED_NATIVE_EVIDENCE_AUDIT' and pa['continuous_replay']
    assert ps['trace_chunks'] and not ps['full_surface_complete']
    trees=[ET.fromstring((directory/'model.xml').read_text(encoding='utf-8')) for directory in (parent,out)]
    meshes=[{e.get('name'):e for e in tree.findall('./asset/mesh')} for tree in trees]
    assert meshes[0].keys()==meshes[1].keys()
    for name in meshes[0]:
        if name.startswith('mustard_collision_mesh_') or name=='mustard_visual_mesh':
            assert Path(meshes[0][name].get('file')).read_bytes()==Path(meshes[1][name].get('file')).read_bytes()
            for entries in meshes:entries[name].set('file','CASE_ASSET/'+name)
    assert ET.tostring(trees[0])==ET.tostring(trees[1])
    assert sha256((parent/'model.xml').read_bytes()).hexdigest()==pa['source_files']['model.xml']
    shutil.copy2(parent/'model.xml',out/'model.xml')
    provenance=out/'parent_provenance';provenance.mkdir()
    for name in ('manifest.json','summary.json','initial.npz','final_state.npz'):
        assert sha256((parent/name).read_bytes()).hexdigest()==pa['source_files'][name]
        shutil.copy2(parent/name,provenance/name)
    shutil.copy2(parent/'independent_native_audit/audit.json',provenance/'audit.json')
    shutil.copytree(parent/'source_snapshot',provenance/'source_snapshot')
    if (parent/'parent_provenance').exists():shutil.copytree(parent/'parent_provenance',provenance/'parent_provenance')
    for item in ps['trace_chunks']:
        source=parent/item['file'];assert sha256(source.read_bytes()).hexdigest()==item['sha256']
        try:os.link(source,out/item['file'])
        except OSError:shutil.copy2(source,out/item['file'])
    task.chunks=list(ps['trace_chunks']);task.states=int(ps['native_states'])
    task.events=json.loads((parent/'events.json').read_text())
    shutil.copy2(parent/'initial.npz',out/'initial.npz')
    with np.load(parent/'final_state.npz') as z:
        for key in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(task.data,key)[:]=z[key]
        task.data.time=float(z['time']);task.coverage.visited_vertices[:]=z['visited_vertices'];task.coverage.visited_goals[:]=z['visited_goals']
    mujoco.mj_forward(task.h.model,task.data)
    task.last=np.r_[task.data.ctrl[task.h.arm_actuator_ids],task.data.ctrl[task.h.hand_actuator_ids]]
    with np.load(parent/task.chunks[-1]['file']) as z:
        np.testing.assert_array_equal(task.last,z['command_rad'][-1])
        assert abs(task.data.time-float(z['time_s'][-1])-.002)<1e-9
        task.anchor=int(z['leader_finger'][-1]);task.goal=int(z['goal_vertex'][-1])
    task.refresh_surface();task.routes=int(ps['routes']);task.lifts=int(ps['necessary_lifts'])
    task.peak=float(ps['peak_force_n']);task.geometry_backtracks=int(ps['geometry_backtracks']);task.force_releases=0
    task.resume_info=dict(parent_case=str(parent),join_time_s=float(task.data.time),
        parent_manifest_sha256=sha256((parent/'manifest.json').read_bytes()).hexdigest(),
        parent_summary_sha256=sha256((parent/'summary.json').read_bytes()).hexdigest(),
        parent_audit_sha256=sha256((provenance/'audit.json').read_bytes()).hexdigest(),
        prefix_native_states=task.states,prefix_chunks=len(task.chunks),physical_state_reset=False,
        independent_trajectories_merged=False,previous_protocol=pm['protocol'],parent_resume=pm.get('resume'),
        controller_state='Reinitialized; physical state and previous command unchanged')
    task.events.append(dict(kind='EXACT_NATIVE_CONTINUATION',time_s=task.data.time,**task.resume_info))
    shutil.copy2(parent/'final_state.npz',out/'checkpoint.npz')
    (out/'trace_chunks.json').write_text(json.dumps(task.chunks,indent=2));(out/'events.json').write_text(json.dumps(task.events,indent=2))
