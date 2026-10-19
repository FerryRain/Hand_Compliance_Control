"""Independent complete replay and sensor-only coverage of terminal continuation."""
from pathlib import Path, PurePosixPath
from hashlib import sha256
from functools import lru_cache
import ast,json
import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra
from Module.i04_geometric_traversal.sliding_audit import _native_context,_native_arm_contacts
from Module.i04_geometric_traversal.physics import contact_measurement
from Module.i04_geometric_traversal.fixtures import SurfaceGraph
from Module.i04_geometric_traversal.full_review import task_mesh

import sys
P=Path(sys.argv[1]).resolve()
def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text());summary=json.loads((P/'summary.json').read_text())
assert manifest['protocol']=='i04-terminal-route-continuation.development.v2'
assert manifest['scope']=='INDEPENDENT_TERMINAL_CONTINUATION_NOT_FRESH_FULL_RUN'
assert not summary['fresh_full_traversal_success']
assert digest(P/'trace.npz')==summary['trace_sha256']
assert digest(P/'route.npz')==manifest['route_sha256']
source=Path(manifest['source_run']);provenance=manifest['source_provenance']
parent_manifest=provenance.get('parent_manifest_sha256',provenance.get('source_manifest_sha256'))
assert parent_manifest is not None
for name,expected in [('final_state.npz',manifest['source_final_state_sha256']),('manifest.json',parent_manifest)]:
    assert digest(source/name)==expected and digest(P/('source_'+name))==expected,name
assert digest(source/'trace.npz')==manifest['source_trace_sha256']
for name in ('model.xml','surface_graph.npz'):assert digest(P/name)==digest(source/name)
coverage_source=next(path for path in (source/'coverage_independent.npz',source/'trace.npz') if path.is_file() and digest(path)==manifest['source_coverage_sha256'])
source_mask=load(coverage_source)['covered_vertices'].astype(bool)
np.testing.assert_array_equal(source_mask,load(P/'source_coverage.npz')['covered_vertices'])
for name,expected in manifest['source_hashes'].items():
    relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
    assert digest(P/'source_snapshot'/relative)==expected,name
tree=ast.parse((P/'source_snapshot/Module/i04_geometric_traversal/terminal_route_probe.py').read_text())
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe')
for n in ast.walk(run):
    if isinstance(n,ast.Call):assert not(isinstance(n.func,ast.Name) and n.func.id=='set_joint_state')
    targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AugAssign,ast.AnnAssign)) else []
    assert not any(isinstance(a,ast.Attribute) and a.attr=='qpos' for t in targets for a in ast.walk(t))
ctx=_native_context(P);m,d,h=ctx.model,ctx.data,ctx.handles
assert m.opt.timestep==.002 and m.na==0
start,initial,old_final=load(P/'replay_start.npz'),load(P/'initial.npz'),load(P/'source_final_state.npz')
terminal_errors={}
if 'parent_manifest_sha256' in provenance:
    for key in ('qpos','qvel','ctrl','qacc_warmstart','time'):
        np.testing.assert_array_equal(start[key],initial[key]);np.testing.assert_array_equal(initial[key],old_final[key])
        terminal_errors[key]=0.
        if key!='time':getattr(d,key)[:]=initial[key]
    d.time=float(initial['time']);mujoco.mj_forward(m,d)
    with np.load(source/'trace.npz') as z:
        for native,logged,ids in [('qpos','q_rad',ctx.qpos),('qvel','qvel_rad_s',ctx.dof),('ctrl','sensor_command_rad',ctx.actuator_ids)]:
            np.testing.assert_array_equal(initial[native][ids],z[logged][-1])
        np.testing.assert_array_equal(initial['qacc_warmstart'],z['qacc_warmstart'][-1])
        assert initial['time']==float(z['native_time_s'][-1])==provenance['native_time_s']
else:
    # Original sliding run stopped between 100 Hz logs. Reproduce both final
    # native ticks rather than trusting a reset to the asserted endpoint.
    assert digest(P/'model.xml')==provenance['source_model_sha256']
    assert digest(P/'surface_graph.npz')==provenance['source_graph_sha256']
    origin=provenance['source_trace']
    with np.load(source/'trace.npz') as z:
        row=int(origin['row']);assert row==len(z['time_s'])-1
        assert float(z['time_s'][row])==origin['relative_time_s']
        for native,logged,ids in [('qpos','q_rad',ctx.qpos),('qvel','qvel_rad_s',ctx.dof),('ctrl','sensor_command_rad',ctx.actuator_ids)]:
            np.testing.assert_array_equal(start[native][ids],z[logged][row])
        np.testing.assert_array_equal(start['qacc_warmstart'],z['qacc_warmstart'][row])
        np.testing.assert_array_equal(start['first_control_canonical'],z['command_rad'][row])
    clock=0.
    for _ in range(provenance['acquisition_steps']+round(origin['relative_time_s']/.002)):clock+=.002
    assert clock==start['time']==provenance['native_time_s']
    for key in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(d,key)[:]=start[key]
    d.time=float(start['time']);mujoco.mj_forward(m,d)
    d.ctrl[ctx.actuator_ids]=start['first_control_canonical'];mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
    d.ctrl[:]=old_final['ctrl'];mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
    for key in ('qpos','qvel','ctrl','qacc_warmstart','time'):
        terminal_errors[key]=float(np.max(abs(np.asarray(getattr(d,key))-initial[key])))
        assert terminal_errors[key]==0.,(key,terminal_errors[key])
        if key in old_final:np.testing.assert_array_equal(initial[key],old_final[key])
route=load(P/'route.npz');goal=int(manifest['goal_vertex'])
np.testing.assert_array_equal(route['q'][0],initial['qpos'][ctx.qpos]);assert int(route['goal_vertex'])==goal
tr=load(P/'trace.npz');n=len(tr['time_s'])
np.testing.assert_allclose(np.diff(tr['time_s']),.002,atol=1e-12,rtol=0)
assert tr['time_s'][0]==0 and tr['time_s'][-1]==summary['executed_s']
errors={};active_errors=0;unsafe_frames=[];arm_violations=[];peak=0.;peak_arm=0.;maximum_high_run=0;high_run=0
for row in range(n):
    forces,points,normals,active,unsafe,_=contact_measurement(h,d,.2)
    arm=max([0.]+[c['force_n'] for c in _native_arm_contacts(m,d)])
    pairs={'q_rad':(d.qpos[ctx.qpos],tr['q_rad'][row]),'qvel_rad_s':(d.qvel[ctx.dof],tr['qvel_rad_s'][row]),
        'sensor_command_rad':(d.ctrl[ctx.actuator_ids],tr['sensor_command_rad'][row]),'qacc_warmstart':(d.qacc_warmstart,tr['qacc_warmstart'][row]),
        'native_time_s':(d.time,tr['native_time_s'][row]),'forces_n':(forces,tr['forces_n'][row]),
        'contact_positions_world_m':(points,tr['contact_positions_world_m'][row]),'normals_world':(normals,tr['normals_world'][row]),
        'tip_positions_world_m':(d.site_xpos[ctx.sites],tr['tip_positions_world_m'][row]),'arm_self_force_n':(arm,tr['arm_self_force_n'][row])}
    for key,(a,b) in pairs.items():errors[key]=max(errors.get(key,0.),float(np.max(abs(np.asarray(a)-b))))
    active_errors+=int(np.count_nonzero(active!=tr['active'][row]))
    if unsafe:unsafe_frames.append(row)
    if arm>.5:arm_violations.append(row)
    peak=max(peak,float(forces.max()));peak_arm=max(peak_arm,arm)
    high_run=high_run+1 if forces.max()>12 else 0;maximum_high_run=max(maximum_high_run,high_run)
    if row+1<n:
        d.ctrl[ctx.actuator_ids]=tr['command_rad'][row];mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
    if row%10000==0:print('NATIVE_REPLAY',row,n,flush=True)
assert max(errors.values())<1e-7 and active_errors==0,errors
final=load(P/'final_state.npz')
for key in ('qpos','qvel','ctrl','qacc_warmstart','time'):np.testing.assert_array_equal(getattr(d,key),final[key])
assert peak==summary['peak_force_n'] and peak_arm==summary['peak_arm_self_force_n']
max_command_step=float(np.max(abs(tr['command_rad']-tr['sensor_command_rad'])))
assert max_command_step<=.0024+1e-10
graph=SurfaceGraph(task_mesh(),coverage_radius_m=.025);saved=load(P/'surface_graph.npz')
np.testing.assert_array_equal(saved['vertices'],graph.vertices_m);np.testing.assert_array_equal(saved['faces'],graph.faces)
np.testing.assert_array_equal(saved['required_vertices'],graph.required_vertices)
assert len(graph.vertices_m)==8192 and len(graph.required_vertices)==388
area=np.zeros(8192);np.add.at(area,graph.faces.ravel(),np.repeat(graph.mesh.area_faces/3.,3))
original=source_mask;mask=original.copy()
np.testing.assert_array_equal(tr['source_covered_vertices'],original)
rows,fingers=np.nonzero(tr['active']);points=tr['contact_positions_world_m'][rows,fingers]-np.asarray(manifest['object_position_m'])
normals=tr['normals_world'][rows,fingers];dist,candidates=graph._tree.query(points,k=24)
alignment=np.einsum('nkj,nj->nk',graph.normals[candidates],normals/np.linalg.norm(normals,axis=1)[:,None])
col=np.argmax(np.where(dist<=dist[:,:1]+.0015,alignment,-np.inf),axis=1);vertices=candidates[np.arange(len(rows)),col]
valid=(dist[np.arange(len(rows)),col]<=.005)&(np.einsum('ij,ij->i',graph.normals[vertices],normals)>=.5)
@lru_cache(maxsize=8192)
def neighborhood(vertex):return np.flatnonzero(np.isfinite(dijkstra(graph.adjacency,directed=False,indices=int(vertex),limit=.025)))
curve=np.zeros(n);near=np.zeros(n,dtype=bool);cursor=0
for row in range(n):
    while cursor<len(rows) and rows[cursor]==row:
        if valid[cursor]:
            touched=neighborhood(int(vertices[cursor]));finger=fingers[cursor]
            if row%5==0:mask[touched]=True
            if .2<=tr['forces_n'][row,finger]<=8. and goal in touched:near[row]=True
        cursor+=1
    curve[row]=float(area@mask/area.sum())
    assert int(mask.sum())==tr['covered_vertex_count'][row]
    assert int(mask[graph.required_vertices].sum())==tr['covered_goals'][row]
np.testing.assert_allclose(curve,tr['coverage'],atol=1e-10,rtol=0)
np.testing.assert_array_equal(mask,tr['covered_vertices']);np.testing.assert_array_equal(near,tr['target_region_safe_contact'])
new=np.flatnonzero(mask&~original).tolist();remaining=np.flatnonzero(~mask).tolist()
assert new==summary['new_covered_vertices'] and remaining==summary['remaining_vertices']
assert int(mask.sum())==summary['combined_covered_vertex_count'] and int(mask[graph.required_vertices].sum())==summary['combined_visited_required_goals']
first=float(tr['time_s'][np.flatnonzero(near)[0]]) if near.any() else None
edges=np.diff(np.r_[False,near,False].astype(int));runs=[(int(a),int(b)-1) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
stable=[(a,b) for a,b in runs if tr['time_s'][b]-tr['time_s'][a]>=.15]
assert first==summary['first_goal_region_contact_s']
confirmation=summary['goal_region_stable_confirmation_s']
assert bool(stable)==(confirmation is not None)
if confirmation is not None:
    a,b=stable[0];assert .15<=confirmation-tr['time_s'][a]<=.152+1e-10
result=dict(status='PASSED',scope=manifest['scope'],source_snapshots_verified=len(manifest['source_hashes']),
    source_unchanged=True,parent_run=str(source),goal_vertex=goal,executed_s=float(tr['time_s'][-1]),exact_parent_endpoint_errors=terminal_errors,continuous_native_states=n,
    native_replay_max_errors=errors,active_disagreements=active_errors,unsafe_frames=unsafe_frames,
    arm_self_violation_frames=arm_violations,peak_force_n=peak,peak_arm_self_force_n=peak_arm,
    maximum_over_12N_native_duration_s=maximum_high_run*.002,maximum_command_step_rad=max_command_step,
    independently_recomputed_new_vertices=new,remaining_vertices=remaining,combined_vertices=int(mask.sum()),
    combined_required_goals=int(mask[graph.required_vertices].sum()),total_vertices=8192,total_required_goals=388,
    actual_goal_region_first_contact_s=first,actual_goal_region_stable_confirmation_s=confirmation,
    fresh_full_traversal_success=False,all_surface_vertices_covered=bool(mask.all()),stop_reason=summary['stop_reason'],
    trace_sha256=digest(P/'trace.npz'),audit_source_sha256=digest(__file__))
(P/'independent_native_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
