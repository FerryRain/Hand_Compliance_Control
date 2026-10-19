"""Independent continuous native replay and per-contact belly audit of local probes.

This intentionally does not import the runtime belly classifier or controller.
Measured tangential pad-material motion is separate from contact-centroid travel
and the requested stroke distance. No interval spanning a contact loss counts.
"""
from pathlib import Path, PurePosixPath
from hashlib import sha256
from functools import lru_cache
import argparse
import ast
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra

from Module.i04_geometric_traversal.sliding_audit import _native_context, _native_arm_contacts
from Module.i04_geometric_traversal.physics import contact_measurement
from Module.i04_geometric_traversal.full_review import task_mesh
from Module.i04_geometric_traversal.fixtures import SurfaceGraph

ROOT = Path(__file__).resolve().parents[3]
digest = lambda path: sha256(Path(path).read_bytes()).hexdigest()


def bouts(mask, time):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1
    return [dict(first_s=float(time[a]), last_s=float(time[b]),
                 endpoint_span_s=float(time[b]-time[a]), states=int(b-a+1))
            for a, b in zip(starts, ends)]


def classify_contacts(c, cfg):
    m, d = c.model, c.data
    objects = set(map(int, c.handles.object_geom_ids))
    fingers = {int(g): f for f, g in enumerate(c.handles.tip_geom_ids)}
    force, point, normal, velocity = np.zeros(4), np.zeros((4,3)), np.zeros((4,3)), np.zeros(4)
    records = []; wrench = np.zeros(6)
    for k in range(d.ncon):
        ct = d.contact[k]; a, b = int(ct.geom1), int(ct.geom2)
        if a in objects and b in fingers: geom, n = b, np.array(ct.frame[:3])
        elif b in objects and a in fingers: geom, n = a, -np.array(ct.frame[:3])
        else: continue
        mujoco.mj_contactForce(m,d,k,wrench)
        f = max(0.,float(wrench[0])); finger = fingers[geom]
        if f <= 0: continue
        r = d.geom_xmat[geom].reshape(3,3); size = m.geom_size[geom]
        witness = np.array(ct.pos)+.5*float(ct.dist)*n
        local = r.T@(witness-d.geom_xpos[geom]); radius = np.linalg.norm(local/size)
        shell = local/radius if radius > 1e-12 else np.zeros(3)
        rho = float(np.linalg.norm(shell[:2]/size[:2]))
        shell_error = float(np.linalg.norm(local-shell))
        cos = float(r[:,2]@-n); angle = float(np.degrees(np.arccos(np.clip(cos,-1,1))))
        grad = r@(shell/(size*size)); grad /= max(np.linalg.norm(grad),1e-20)
        normal_cosine = float(grad@-n)
        qualified = bool(radius>1e-12 and shell[2]>0 and
            angle<=cfg['maximum_alignment_angle_deg']+1e-10 and
            rho<=cfg['maximum_patch_radius']+1e-12 and
            shell_error<=cfg['surface_tolerance_m']+1e-12 and normal_cosine>0)
        speed = 0.
        if qualified:
            jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jac(m,d,jp,jr,witness,int(m.geom_bodyid[geom]))
            v = jp@d.qvel; tangent = v-n*(v@n); speed = float(np.linalg.norm(tangent))
            force[finger]+=f; point[finger]+=f*np.array(ct.pos); normal[finger]+=f*n
            velocity[finger]+=f*speed
        records.append((finger,k,f,qualified,angle,rho,shell_error,normal_cosine,*local,speed))
    active = np.zeros(4,dtype=bool)
    for f in range(4):
        length = np.linalg.norm(normal[f])
        if force[f]>=.2 and length>1e-10:
            active[f]=True; point[f]/=force[f]; normal[f]/=length; velocity[f]/=force[f]
        else: point[f]=0.;normal[f]=0.;velocity[f]=0.
    return force,point,normal,active,velocity,records


def audit(directory):
    run=Path(directory).resolve(); out=run/'independent_native_audit'; out.mkdir(exist_ok=True)
    (out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    manifest=json.loads((run/'manifest.json').read_text()); summary=json.loads((run/'summary.json').read_text())
    assert digest(run/'trace.npz')==summary['trace_sha256']
    hashes=manifest['source_hashes']
    for relative, expected in hashes.items():
        path=PurePosixPath(relative); assert not path.is_absolute() and '..' not in path.parts
        assert digest(run/'source_snapshot'/path)==expected, relative
    # Auditing loads these helpers locally. Verify the relevant frozen runtime
    # definitions, and separately record all auditor imports below.
    for relative in ('Module/fr3_leap/model.py','Module/i04_geometric_traversal/physics.py'):
        assert digest(ROOT/relative)==hashes[relative]
    entry='Module/i04_geometric_traversal/belly_surface_probe.py'
    tree=ast.parse((run/'source_snapshot'/entry).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    setters=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='set_joint_state']
    assert all(len(n.args)>=2 and isinstance(n.args[1],ast.Name) and n.args[1].id=='scratch' for n in setters)
    direct_writes=[]
    for n in ast.walk(fn):
        targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AugAssign,ast.AnnAssign)) else []
        for target in targets:
            if any(isinstance(x,ast.Attribute) and x.attr=='qpos' for x in ast.walk(target)):direct_writes.append(n.lineno)
    assert not direct_writes
    c=_native_context(run);m,d=c.model,c.data
    # This probe explicitly logs data.qvel in native DOF order, unlike older
    # whole-surface logs that reordered velocity into joint-name order.
    with np.load(run/'trace.npz') as z: tr={k:z[k].copy() for k in z.files}
    with np.load(run/'initial.npz') as z: initial={k:z[k].copy() for k in z.files}
    with np.load(run/'final_state.npz') as z: final={k:z[k].copy() for k in z.files}
    t=tr['time_s']; count=len(t); assert count
    np.testing.assert_allclose(np.diff(t),.002,atol=1e-10,rtol=0)
    mdt=float(m.opt.timestep); assert abs(mdt-.002)<1e-12
    d.qpos[:]=initial['qpos'];d.qvel[:]=initial['qvel'];d.ctrl[:]=initial['ctrl']
    d.qacc_warmstart[:]=initial['qacc_warmstart'];d.time=float(initial['time'])
    mujoco.mj_forward(m,d)
    graph=SurfaceGraph(task_mesh(),coverage_radius_m=.025)
    assert len(graph.vertices_m)==8192 and graph.required_goal_count==388
    object_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fr3_e05_object')
    object_position=m.body_pos[object_id]
    area=np.zeros(len(graph.vertices_m));np.add.at(area,graph.faces.ravel(),np.repeat(graph.mesh.area_faces/3,3))
    mask=np.zeros(len(area),bool); first=np.full(len(area),np.nan)
    @lru_cache(maxsize=8192)
    def neighborhood(v):return np.flatnonzero(np.isfinite(dijkstra(graph.adjacency,directed=False,indices=int(v),limit=.025)))
    maximum={};contact_rows=[];speeds=np.zeros((count,4));safe=np.zeros((count,4),bool);coverage=np.zeros(count)
    def check(name,actual,expected,tolerance=1e-9):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        maximum[name]=max(maximum.get(name,0.),error)
        assert error<=tolerance,(name,error,row)
    peak=np.zeros(4);arm_peak=0.;unsafe_states=[];high_count=0;maximum_high_count=0
    for row in range(count):
        check('q_rad',d.qpos[c.qpos],tr['q_rad'][row]);check('qvel_rad_s',d.qvel,tr['qvel_rad_s'][row])
        check('prior_control',d.ctrl[c.actuator_ids],tr['sensor_command_rad'][row])
        check('qacc_warmstart',d.qacc_warmstart,tr['qacc_warmstart'][row],1e-7)
        check('time_s',d.time-float(initial['time']),t[row])
        check('tip_positions',d.site_xpos[c.sites],tr['tip_positions_world_m'][row])
        check('pad_frames',d.geom_xmat[c.handles.tip_geom_ids],tr['pad_frames_world'][row])
        f,p,n,a,unsafe,pen=contact_measurement(c.handles,d,.2)
        for key,value in (('forces_n',f),('contact_positions_world_m',p),('normals_world',n)):
            check(key,value,tr[key][row])
        np.testing.assert_array_equal(a,tr['active'][row]);peak=np.maximum(peak,f)
        bf,bp,bn,ba,vel,rs=classify_contacts(c,manifest['belly_contact'])
        for key,value in (('belly_forces_n',bf),('belly_contact_positions_world_m',bp),('belly_normals_world',bn)):
            check(key,value,tr[key][row])
        np.testing.assert_array_equal(ba,tr['belly_active'][row])
        safe[row]=ba&(f<8);speeds[row]=vel
        contact_rows.extend([(float(t[row]),*r) for r in rs])
        arm=max([x['force_n'] for x in _native_arm_contacts(m,d)]+[0.]);arm_peak=max(arm_peak,arm)
        check('arm_self_force_n',arm,tr['arm_self_force_n'][row])
        if unsafe or arm>.5:unsafe_states.append(dict(time_s=float(t[row]),unsafe=unsafe,arm_force_n=arm))
        high_count=high_count+1 if f.max()>12 else 0;maximum_high_count=max(maximum_high_count,high_count)
        for finger in np.flatnonzero(safe[row]):
            distances,idx=graph._tree.query(bp[finger]-object_position,k=24)
            cos=graph.normals[idx]@bn[finger];eligible=distances<=distances[0]+.0015
            col=np.argmax(np.where(eligible,cos,-np.inf));v=int(idx[col])
            if distances[col]<=.005 and cos[col]>=.5:
                ns=neighborhood(v);first[ns[~mask[ns]]]=t[row];mask[ns]=True
        coverage[row]=float(area@mask/area.sum());check('coverage',coverage[row],tr['coverage'][row])
        advance = row<count-1 or float(final['time'])>d.time+mdt/2
        if advance:
            cmd=tr['command_rad'][row]
            assert np.max(abs(cmd-d.ctrl[c.actuator_ids]))<=.0024+1e-10
            assert not unsafe and arm<=.5 and f.max()<8
            d.ctrl[c.actuator_ids]=cmd;mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        if (row+1)%5000==0: print('AUDITED',row+1,count,flush=True)
    for name in ('qpos','qvel','ctrl','qacc_warmstart','time'):
        check('final_'+name,getattr(d,name),final[name],1e-7 if name=='qacc_warmstart' else 1e-9)
    assert abs(float(peak.max())-summary['peak_force_n'])<1e-9
    assert abs(float(coverage[-1])-summary['belly_coverage_fraction'])<1e-9
    dt=np.diff(t); slide=tr['phase']=='BELLY_SLIDE';statistics=[]
    for finger in range(4):
        keep=safe[:-1,finger]&safe[1:,finger]
        material=float(np.sum(.5*(speeds[:-1,finger]+speeds[1:,finger])*dt*keep))
        travel=np.linalg.norm(np.diff(tr['belly_contact_positions_world_m'][:,finger],axis=0),axis=1)
        bout=bouts(safe[:,finger],t)
        statistics.append(dict(finger=finger,safe_belly_fraction=float(safe[:,finger].mean()),
            slide_safe_belly_fraction=float(safe[slide,finger].mean()) if slide.any() else None,
            actual_tangential_material_motion_m=material,contact_centroid_path_m=float(np.sum(travel[keep])),
            actual_motion_by_phase={str(phase):dict(
                tangential_material_motion_m=float(np.sum(.5*(speeds[:-1,finger]+speeds[1:,finger])*dt*keep*
                    (tr['phase'][:-1]==phase)*(tr['phase'][1:]==phase))),
                contact_centroid_path_m=float(np.sum(travel[keep&(tr['phase'][:-1]==phase)&(tr['phase'][1:]==phase)])))
                for phase in np.unique(tr['phase'])},
            longest_safe_endpoint_span_s=max([x['endpoint_span_s'] for x in bout]+[0.]),
            stable_150ms_bouts=[x for x in bout if x['endpoint_span_s']>=.15-1e-10],
            joint_range_deg=np.degrees(np.ptp(tr['q_rad'][:,7+finger*4:11+finger*4],axis=0)).tolist()))
    np.savez_compressed(out/'independent_measurements.npz',time_s=t,belly_safe=safe,
        belly_tangential_material_speed_m_s=speeds,coverage=coverage,covered_vertices=mask,first_touch_time_s=first)
    np.savez_compressed(out/'per_contact_native.npz',records=np.asarray(contact_rows),columns=np.array([
        'time_s','finger','contact_index','force_n','belly_qualified','angle_deg','patch_radius',
        'shell_error_m','normal_gradient_cosine','local_x_m','local_y_m','local_z_m','tangential_material_speed_m_s']))
    assets={}
    for e in ET.fromstring((run/'model.xml').read_text()).iter():
        if 'file' in e.attrib:
            path=Path(e.attrib['file'])
            if path.is_file():assets[str(path)]=digest(path)
    result=dict(status='PASSED_NATIVE_EVIDENCE_AUDIT',source_run=str(run),protocol=manifest['protocol'],
        source_snapshot_count=len(hashes),source_snapshots_verified=True,
        scratch_only_joint_setters=len(setters),physical_qpos_writes=direct_writes,
        source_files={name:digest(run/name) for name in ('manifest.json','summary.json','trace.npz','initial.npz','final_state.npz','model.xml','events.json')},
        auditor_sha256=digest(Path(__file__)),asset_hashes_at_audit=assets,
        asset_hash_scope='Model-referenced assets hashed at audit; not retroactive pre-run asset immutability proof',
        native_states=count,continuous_replay=True,physical_initializations=1,
        recorded_duration_s=float(t[-1]-t[0]),executed_duration_s=float(final['time']-initial['time']),
        maximum_errors=maximum,raw_force_peak_n=peak.tolist(),arm_self_peak_n=arm_peak,
        unsafe_states=unsafe_states,maximum_consecutive_over12_native_states=maximum_high_count,
        behavior_stop_reason=summary['stop_reason'],whole_surface_complete=False,
        measured_belly_coverage_fraction=float(coverage[-1]),covered_vertices=int(mask.sum()),total_vertices=len(mask),
        measured_belly_goals=int(mask[graph.required_vertices].sum()),total_goals=388,
        actual_motion_scope='Trapezoidal integral of actual qvel/contact Jacobian tangential material speed, only adjacent safe belly states; centroid path is separate and may include witness migration',
        requested_stroke_distance_m=float(tr['stroke_distance_m'].max()),finger_statistics=statistics,
        per_contact_count=len(contact_rows),qualified_positive_force_contacts=sum(int(r[4]) for r in contact_rows),
        belly_slide_phase_present=bool(slide.any()),new_full_or_safe_success_claim=False,
        artifacts={name:digest(out/name) for name in ('independent_measurements.npz','per_contact_native.npz')})
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','native_states','maximum_errors','behavior_stop_reason','measured_belly_coverage_fraction','finger_statistics')},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();audit(args.directory)
