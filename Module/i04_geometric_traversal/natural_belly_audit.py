"""Continuous replay, independent palmar classification and whole coverage.

Imports the independently written contact auditor, not runtime classifiers.
Chunk boundaries never reset the physical replay or the coverage ledger.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
import time
from zipfile import BadZipFile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra

from .sliding_audit import _native_context, _native_arm_contacts
from .physics import contact_measurement
from .full_review import task_mesh
from .fixtures import SurfaceGraph
from .sliding_safety import SlidingSafetyPlanner
from .contracts import ExperimentConfig
from Module.fr3_leap.model import ARM_JOINT_NAMES,HAND_JOINT_NAMES

ROOT=Path(__file__).resolve().parents[2]


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def audit(directory,check_commands=True,follow=False):
    case=Path(directory).resolve();out=case/'independent_native_audit';out.mkdir(exist_ok=True)
    manifest=json.loads((case/'manifest.json').read_text())
    summary=json.loads((case/'summary.json').read_text()) if (case/'summary.json').exists() else None
    if summary is None and not follow:raise ValueError('No finished summary; use --follow for a live audit')
    for name,expected in manifest['source_hashes'].items():
        name=PurePosixPath(name);assert not name.is_absolute() and '..' not in name.parts
        assert digest(case/'source_snapshot'/name)==expected
    helper=ROOT/'Module/generated/belly_contact_audit_20261017/audit_native_probe.py'
    spec=importlib.util.spec_from_file_location('independent_original_palmar_auditor',helper)
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    (out/'auditor.py').write_bytes(Path(__file__).read_bytes());(out/'independent_contact_classifier.py').write_bytes(helper.read_bytes())
    c=_native_context(case);m,d=c.model,c.data
    with np.load(case/'initial.npz') as z:initial={k:z[k].copy() for k in z.files}
    final=None
    if (case/'final_state.npz').exists():
        with np.load(case/'final_state.npz') as z:final={k:z[k].copy() for k in z.files}
    for k in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(d,k)[:]=initial[k]
    d.time=float(initial['time']);mujoco.mj_forward(m,d)
    names=(*ARM_JOINT_NAMES,*HAND_JOINT_NAMES);jids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names])
    h=SimpleNamespace(model=m,object_geom_ids=c.handles.object_geom_ids,object_geom_id=c.handles.object_geom_ids[0],
        tip_geom_ids=c.handles.tip_geom_ids,arm_qpos_adrs=m.jnt_qposadr[jids[:7]],hand_qpos_adrs=m.jnt_qposadr[jids[7:]],
        arm_joint_ranges_rad=m.jnt_range[jids[:7]],hand_joint_ranges_rad=m.jnt_range[jids[7:]])
    safety=SlidingSafetyPlanner(h,ExperimentConfig()) if check_commands else None
    graph=SurfaceGraph(task_mesh(),coverage_radius_m=.025)
    assert len(graph.vertices_m)==8192 and len(graph.required_vertices)==388
    area=np.zeros(8192);np.add.at(area,graph.faces.ravel(),np.repeat(graph.mesh.area_faces/3,3))
    covered=np.zeros(8192,bool);first=np.full(8192,np.nan)
    obj=m.body_pos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fr3_e05_object')]
    @lru_cache(maxsize=8192)
    def neighborhood(v):return np.flatnonzero(np.isfinite(dijkstra(graph.adjacency,directed=False,indices=int(v),limit=.025)))
    maximum={};row=0;peak=np.zeros(4);armpeak=0.;qmin=np.full(23,np.inf);qmax=-qmin
    gaps=np.zeros(5);longest=np.zeros(5);active_sum=np.zeros(4);phase_count={};phase_active={}
    material=np.zeros(4);prev_vel=np.zeros(4);prev_safe=np.zeros(4,bool);last_time=None;full_hold=0.;unsafe_count=0;high_steps=0
    preview={k:[] for k in ('time_s','q_rad','forces_n','belly_active','belly_contact_positions_world_m','phase','coverage','leader_finger')}
    def check(name,actual,expected,tol=1e-9):
        error=float(np.max(abs(np.asarray(actual)-np.asarray(expected))))
        maximum[name]=max(maximum.get(name,0.),error)
        if error>tol:raise AssertionError((name,error,row))
    read_hashes={};reference_guard_steps=0
    def chunks():
        delivered=0
        while True:
            try:
                finished=(case/'summary.json').exists()
                catalog=(json.loads((case/'summary.json').read_text())['trace_chunks'] if finished
                         else json.loads((case/'trace_chunks.json').read_text()))
            except (FileNotFoundError,json.JSONDecodeError):
                if not follow:raise
                time.sleep(1.);continue
            while delivered<len(catalog):
                item=catalog[delivered];delivered+=1;yield item
            if finished:return
            if not follow:raise ValueError('Run is still active')
            time.sleep(1.)
    for chunk in chunks():
        observed_hash=digest(case/chunk['file']);read_hashes[chunk['file']]=observed_hash
        if 'sha256' in chunk:assert observed_hash==chunk['sha256']
        while True:
            try:
                endpoint=case/'final_state.npz' if (case/'final_state.npz').exists() else case/'checkpoint.npz'
                with np.load(endpoint) as z:advance_until=float(z['time'])
                break
            except (ValueError,OSError,EOFError,BadZipFile):
                if not follow:raise
                time.sleep(.2)
        with np.load(case/chunk['file']) as z:tr={k:z[k].copy() for k in z.files}
        if chunk.get('skip_last',0):tr={k:v[:-chunk['skip_last']] for k,v in tr.items()}
        for k,t in enumerate(tr['time_s']):
            if last_time is not None:assert abs(t-last_time-.002)<1e-9
            last_time=float(t)
            for name,value in (('q_rad',d.qpos[c.qpos]),('qvel_rad_s',d.qvel),('sensor_command_rad',d.ctrl[c.actuator_ids]),('qacc_warmstart',d.qacc_warmstart),('time_s',d.time)):
                check(name,value,tr[name][k],1e-7 if name=='qacc_warmstart' else 1e-9)
            f,p,n,a,unsafe,pen=contact_measurement(c.handles,d,.2)
            for name,value in (('forces_n',f),('contact_positions_world_m',p),('normals_world',n)):
                check(name,value,tr[name][k])
            np.testing.assert_array_equal(a,tr['active'][k])
            bf,bp,bn,ba,vel,records=independent.classify_contacts(c,manifest['belly_contact'])
            for name,value in (('belly_forces_n',bf),('belly_contact_positions_world_m',bp),('belly_normals_world',bn)):
                check(name,value,tr[name][k])
            rejected=np.zeros(4)
            for record in records:
                if not record[3]:rejected[record[0]]+=record[2]
            check('rejected_contact_force_n',rejected,tr['rejected_contact_force_n'][k])
            np.testing.assert_array_equal(ba,tr['belly_active'][k])
            arm=max([item['force_n'] for item in _native_arm_contacts(m,d)]+[0.])
            armpeak=max(armpeak,arm);unsafe_count+=bool(unsafe or arm>.5)
            safe=ba&(f<8.)&(rejected<.2)
            if unsafe or arm>.5 or f.max()>=8:safe[:]=False
            np.testing.assert_array_equal(safe,tr['safe_belly'][k])
            for finger in np.flatnonzero(safe):
                distances,idx=graph._tree.query(bp[finger]-obj,k=24)
                cos=graph.normals[idx]@bn[finger];eligible=distances<=distances[0]+.0015
                col=int(np.argmax(np.where(eligible,cos,-np.inf)));v=int(idx[col])
                if distances[col]<=.005 and cos[col]>=.5:
                    ns=neighborhood(v);first[ns[~covered[ns]]]=t;covered[ns]=True
            fraction=float(area@covered/area.sum());check('coverage',fraction,tr['coverage'][k])
            assert int(covered.sum())==tr['visited_vertex_count'][k]
            full_hold=full_hold+.002 if covered.all() and safe.any() else 0.
            live=np.r_[safe,safe.any()];gaps=np.where(live,0.,gaps+.002);longest=np.maximum(longest,gaps)
            active_sum+=safe;material+=.001*(prev_vel+vel)*(prev_safe&safe);prev_vel=vel;prev_safe=safe
            peak=np.maximum(peak,f);qmin=np.minimum(qmin,d.qpos[c.qpos]);qmax=np.maximum(qmax,d.qpos[c.qpos])
            high_steps=high_steps+1 if f.max()>12. else 0
            phase=str(tr['phase'][k]);phase_count[phase]=phase_count.get(phase,0)+1;phase_active[phase]=phase_active.get(phase,0)+int(safe.any())
            advance=advance_until>d.time+.001
            if advance:
                cmd=tr['command_rad'][k]
                assert np.max(abs(cmd-d.ctrl[c.actuator_ids]))<=.0024+1e-10
                assert np.all(cmd>=manifest['posture_lower_rad']) and np.all(cmd<=manifest['posture_upper_rad'])
                if safety:
                    reason=safety.command_collision_reason(d.qpos[c.qpos],cmd)
                    assert reason is None,(row,reason)
                    join=float((manifest.get('resume') or {}).get('join_time_s',0.))
                    if manifest.get('command_reference_guard_every_native_tick') and t>=join-1e-9:
                        assert safety.command_collision_reason(tr['sensor_command_rad'][k],cmd) is None
                        reference_guard_steps+=1
                assert not unsafe and arm<=.5 and high_steps<10
                if 'proactive_release_n' in manifest:
                    if f.max()>=8.:assert phase in ('FORCE_RELEASE','CLEARANCE_EXIT')
                else:assert f.max()<8.
                d.ctrl[c.actuator_ids]=cmd;mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            if row%25==0:
                for name in preview:preview[name].append(tr[name][k])
            row+=1
        print('AUDITED',row,'time',float(d.time),'area',fraction,flush=True)
        (out/'progress.json').write_text(json.dumps(dict(native_states=row,time_s=float(d.time),area_fraction=fraction,
            maximum_error=max(maximum.values()),status='CONTINUOUS_REPLAY_IN_PROGRESS'),indent=2))
    summary=json.loads((case/'summary.json').read_text())
    with np.load(case/'final_state.npz') as z:final={k:z[k].copy() for k in z.files}
    for chunk in summary['trace_chunks']:assert read_hashes[chunk['file']]==chunk['sha256']
    if preview['time_s'][-1]<last_time-1e-9:
        for name in preview:preview[name].append(tr[name][-1])
    for name in ('qpos','qvel','ctrl','qacc_warmstart','time'):
        check('final_'+name,getattr(d,name),final[name],1e-7 if name=='qacc_warmstart' else 1e-9)
    np.testing.assert_array_equal(covered,final['visited_vertices'])
    assert int(covered.sum())==summary['visited_vertices']
    complete=bool(covered.all() and covered[graph.required_vertices].all() and full_hold>=.15-1e-9)
    if complete:
        ff,_,_,_,uu,_=contact_measurement(c.handles,d,.2)
        _,_,_,bb,_,rr=independent.classify_contacts(c,manifest['belly_contact'])
        rejected=np.zeros(4)
        for r in rr:
            if not r[3]:rejected[r[0]]+=r[2]
        assert not uu and ff.max()<8. and np.any(bb&(rejected<.2))
    assert complete==summary['full_surface_complete']
    np.savez_compressed(out/'verified_preview.npz',**{k:np.asarray(v) for k,v in preview.items()},
                        covered_vertices=covered,first_touch_s=first)
    result=dict(status='PASSED_NATIVE_EVIDENCE_AUDIT',native_states=row,continuous_replay=True,physical_initializations=1,
        fresh_single_version=not bool(manifest.get('resume')),resume=manifest.get('resume'),
        source_snapshot_count=len(manifest['source_hashes']),maximum_errors=maximum,
        all_applied_commands_original_geometry_checked=check_commands,behavior_stop_reason=summary['stop_reason'],
        current_version_reference_guard_steps=reference_guard_steps,
        full_surface_complete=complete,covered_vertices=int(covered.sum()),total_vertices=8192,
        covered_goals=int(covered[graph.required_vertices].sum()),total_goals=388,area_fraction=fraction,
        raw_force_peak_n=peak.tolist(),arm_self_peak_n=armpeak,unsafe_states=unsafe_count,
        safe_belly_fraction=(active_sum/row).tolist(),longest_safe_belly_gap_s=longest.tolist(),
        actual_tangential_pad_material_motion_m=material.tolist(),phase_duration_s={p:c*.002 for p,c in phase_count.items()},
        phase_any_belly_fraction={p:phase_active[p]/c for p,c in phase_count.items()},
        actual_joint_min_deg=np.degrees(qmin).tolist(),actual_joint_max_deg=np.degrees(qmax).tolist(),
        actual_joint_range_deg=np.degrees(qmax-qmin).tolist(),
        source_files={name:digest(case/name) for name in ('manifest.json','summary.json','initial.npz','final_state.npz','model.xml')},
        audit_source_sha256=digest(out/'auditor.py'),independent_classifier_sha256=digest(out/'independent_contact_classifier.py'),
        preview_sha256=digest(out/'verified_preview.npz'))
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--skip-command-audit',action='store_true')
    p.add_argument('--follow',action='store_true')
    a=p.parse_args();audit(a.case,not a.skip_command_audit,a.follow)
