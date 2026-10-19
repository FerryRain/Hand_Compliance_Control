"""Independent native evidence for bounded local search geometry comparisons."""
from __future__ import annotations
import argparse,ast,json
from hashlib import sha256
from pathlib import Path,PurePosixPath
import mujoco
import numpy as np
from .sliding_audit import _native_context,_native_arm_contacts
from .physics import contact_measurement

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def intervals(mask,times):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [(int(a),int(b)-1,float(times[b-1]-times[a])) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
def skew(w):return np.array([[0.,-w[2],w[1]],[w[2],0.,-w[0]],[-w[1],w[0],0.]])

def audit(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());s=json.loads((directory/'summary.json').read_text())
    assert m['protocol']=='i04-support-point-search.paired.development.v1'
    assert m['coverage_authority']=='NONE' and not m['coverage_counted'] and not m['whole_traversal_evidence']
    assert not m['exact_full_controller_continuation']
    assert digest(directory/'trace.npz')==s['trace_sha256'] and digest(directory/'model.xml')==m['model_sha256']
    for relative,value in m['source_hashes'].items():
        name=PurePosixPath(relative);assert not name.is_absolute() and '..' not in name.parts
        assert digest(directory/'source_snapshot'/name)==value
    tree=ast.parse((directory/'source_snapshot/Module/i04_geometric_traversal/support_recovery_probe.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    for node in ast.walk(run):
        if isinstance(node,ast.Call):assert not(isinstance(node.func,ast.Name) and node.func.id=='set_joint_state')
        targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,(ast.AugAssign,ast.AnnAssign)) else []
        assert not any(isinstance(v,ast.Attribute) and v.attr=='qpos' for target in targets for v in ast.walk(target))
    provenance=m['source_state'];origin=provenance['source_trace'];source_frame=load(directory/'source_frame.npz')
    assert digest(directory/'source_run_manifest.json')==provenance['source_manifest_sha256']
    assert digest(directory/'surface_graph.npz')==provenance['source_graph_sha256']
    assert digest(directory/'source_clock_anchor.npz')==provenance['clock_anchor_sha256']
    assert digest(origin['path'])==origin['sha256']
    controller=load(directory/'controller_frame.npz');assert digest(directory/'controller_frame.npz')==m['controller_frame_sha256']
    initialization=m.get('controller_initialization_provenance')
    with np.load(origin['path']) as original:
        for key,value in source_frame.items():np.testing.assert_array_equal(value,original[key][origin['row']])
        if initialization is None:
            for key,value in controller.items():np.testing.assert_array_equal(value,original[key][origin['row']])
        else:
            # Explicit compatibility adapter for an older positive probe.
            # Newly supplied counters must not acquire "logged" authority.
            assert m['controller_initialization']=='SHARED_FRESH_LOCAL_COUNTERS_BIAS_MCC_TARGET_AGE_WITH_SENSOR_DERIVED_EPISODE_AGES'
            assert not initialization['original_controller_internals_recovered']
            original_trace=initialization['original_source_trace']
            assert digest(original_trace['path'])==original_trace['sha256']==digest(origin['path'])
            assert original_trace['row']==origin['row']
            row=int(origin['row']);active=original['active'][:row+1];force=original['forces_n'][:row+1]
            age,stable=np.zeros(4),np.zeros(4)
            np.testing.assert_allclose(np.diff(original['time_s'][:row+1]),.002,atol=1e-12,rtol=0)
            for f in range(4):
                if active[-1,f]:
                    gaps=np.flatnonzero(~active[:,f]|(force[:,f]>=8.))
                    stable[f]=(row-int(gaps[-1]) if len(gaps) else row+1)*.002
                else:
                    touch=np.flatnonzero(active[:,f]);age[f]=(row-int(touch[-1]) if len(touch) else row+1)*.002
            np.testing.assert_array_equal(controller['finger_search_loss_age_s'],age)
            np.testing.assert_array_equal(controller['finger_contact_stable_age_s'],stable)
            fresh=set(controller)-{'finger_search_loss_age_s','finger_contact_stable_age_s'}
            assert set(initialization['common_fresh_zero_fields'])==fresh
            for key in fresh:assert not np.any(controller[key]),key
            for key,value in controller.items():np.testing.assert_array_equal(value,initialization['initialized_controller_values'][key])
            adapter=m['source_adapter'];adapter_dir=Path(adapter['directory']);old_dir=Path(adapter['source'])
            for name,value in adapter['original_files'].items():
                assert digest(adapter_dir/name)==digest(old_dir/name)==value
            with np.load(old_dir/'final_state.npz') as old_final,np.load(directory/'source_clock_anchor.npz') as clock_anchor:
                assert float(old_final['time'])==float(clock_anchor['time'])
                old_summary=json.loads((old_dir/'summary.json').read_text())
                assert old_summary['executed_s']==float(clock_anchor['elapsed'])
            from .fixtures import SurfaceGraph
            import trimesh
            graph=SurfaceGraph(trimesh.load(old_dir/'canonical_task_surface.obj',force='mesh',process=False),coverage_radius_m=.025)
            with np.load(directory/'surface_graph.npz') as frozen_graph:
                for key,actual in (('vertices',graph.vertices_m),('faces',graph.faces),('required_vertices',graph.required_vertices)):
                    np.testing.assert_array_equal(frozen_graph[key],actual)
    anchor=load(directory/'source_clock_anchor.npz');clock=0.;native_clock=None
    source_step=provenance['acquisition_steps']+round(origin['relative_time_s']/.002)
    anchor_step=provenance['acquisition_steps']+round(float(anchor['elapsed'])/.002)
    for i in range(anchor_step+1):
        if i==source_step:native_clock=clock
        if i<anchor_step:clock+=.002
    assert clock==float(anchor['time']) and native_clock==provenance['native_time_s']
    initial=load(directory/'initial.npz');tr=load(directory/'trace.npz');ctx=_native_context(directory)
    model,data,h=ctx.model,ctx.data,ctx.handles
    assert model.opt.timestep==.002 and model.na==0
    for key in ('qpos','qvel','ctrl','qacc_warmstart'):getattr(data,key)[:]=initial[key]
    data.time=float(initial['time']);assert data.time==native_clock
    for native,logged,ids in [('qpos','q_rad',ctx.qpos),('qvel','qvel_rad_s',ctx.dof),('ctrl','sensor_command_rad',ctx.actuator_ids)]:
        np.testing.assert_array_equal(initial[native][ids],source_frame[logged])
    np.testing.assert_array_equal(initial['qacc_warmstart'],source_frame['qacc_warmstart'])
    handoff=load(directory/'controller_handoff.npz');assert digest(directory/'controller_handoff.npz')==m['controller_handoff_sha256']
    np.testing.assert_array_equal(handoff['effective_initial_reference'],source_frame['sensor_command_rad'])
    np.testing.assert_array_equal(handoff['restored_search_bias'],controller['finger_search_bias_rad'])
    np.testing.assert_allclose(handoff['base_reference']+np.r_[np.zeros(7),handoff['restored_search_bias']],source_frame['sensor_command_rad'],atol=1e-15,rtol=0)
    assert not np.any(handoff['initial_mcc_held_bias'])
    mujoco.mj_forward(model,data)
    times=tr['time_s'];assert times[0]==0. and times[-1]==s['executed_s']
    np.testing.assert_allclose(np.diff(times),.002,atol=1e-12,rtol=0)
    assert len(times)==s['native_recorded_states']
    finger=int(m['selected_finger']);geom=int(h.tip_geom_ids[finger]);columns=7+4*finger+np.arange(4)
    objects=[int(g) for g in h.object_geom_ids if model.geom_contype[g]!=0]
    errors={};active_errors=0;unsafe_rows=[];arm_violations=[];peak=armpeak=0.
    center_steps=[];support_steps=[];geometric_gap=[];geometric_error=[];target_points=[];support_points=[]
    implementation=m['search_implementation']
    assert implementation in ('ActiveFingerSearchV3','ActiveFingerSearchV4','ActiveFingerSearchV5','ActiveFingerSearchV6')
    gap_error=0.;previous_target=None;surface_point=None
    dual_support=implementation in ('ActiveFingerSearchV4','ActiveFingerSearchV5','ActiveFingerSearchV6')
    normal_priority=implementation in ('ActiveFingerSearchV5','ActiveFingerSearchV6')
    finite_progress=implementation=='ActiveFingerSearchV6'
    finite_progress_checks=[];geometry_scratch=mujoco.MjData(model) if finite_progress else None
    normal_step_max=0.;closest_pair_checks=0;closest_point_error=0.;closest_normal_error=0.
    for row in range(len(times)):
        force,points,normals,active,unsafe,_=contact_measurement(h,data,.2)
        arm=max([0.]+[c['force_n'] for c in _native_arm_contacts(model,data)])
        pairs=dict(q_rad=(data.qpos[ctx.qpos],tr['q_rad'][row]),qvel_rad_s=(data.qvel[ctx.dof],tr['qvel_rad_s'][row]),
            sensor_command_rad=(data.ctrl[ctx.actuator_ids],tr['sensor_command_rad'][row]),qacc_warmstart=(data.qacc_warmstart,tr['qacc_warmstart'][row]),
            native_time_s=(data.time,tr['native_time_s'][row]),forces_n=(force,tr['forces_n'][row]),
            contact_positions_world_m=(points,tr['contact_positions_world_m'][row]),normals_world=(normals,tr['normals_world'][row]),
            tip_positions_world_m=(data.site_xpos[ctx.sites],tr['tip_positions_world_m'][row]),arm_self_force_n=(arm,tr['arm_self_force_n'][row]))
        for key,(a,b) in pairs.items():errors[key]=max(errors.get(key,0.),float(np.max(abs(np.asarray(a)-b))))
        active_errors+=int(np.count_nonzero(active!=tr['active'][row]))
        if unsafe:unsafe_rows.append(row)
        if arm>.5:arm_violations.append(row)
        peak=max(peak,float(force.max()));armpeak=max(armpeak,arm)
        if np.isfinite(tr['measured_selected_collision_gap_m'][row]):
            distance=min(mujoco.mj_geomDistance(model,data,geom,g,.2,None) for g in objects)
            gap_error=max(gap_error,abs(distance-tr['measured_selected_collision_gap_m'][row]))
        normal=tr['finger_search_target_normals_world'][row,finger]
        target=tr['finger_search_target_positions_world_m'][row,finger]
        if tr['finger_search_target_valid'][row,finger]:
            frame=data.geom_xmat[geom].reshape(3,3);B=frame@np.diag(model.geom_size[geom]**2)@frame.T
            radius=np.sqrt(normal@B@normal);offset=B@normal/radius;point=data.geom_xpos[geom]-offset
            if dual_support:surface_point=tr['finger_search_surface_points_world_m'][row,finger]
            elif previous_target is None or not np.array_equal(previous_target,target):
                # V3 stores center targets unchanged until a 50 ms refresh.
                # Recover its real surface point once at refresh, not every tick.
                surface_point=target-offset-m['active_finger_search']['target_clearance_m']*normal
            previous_target=target.copy()
            geometric_gap.append(float(normal@(point-surface_point)))
            geometric_error.append(float(np.linalg.norm(surface_point+m['active_finger_search']['target_clearance_m']*normal-point)))
            target_points.append(surface_point.copy());support_points.append(point.copy())
            jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jacGeom(model,data,jp,jr,geom)
            support=jp[:,ctx.dof[columns]].copy()
            for j,column in enumerate(columns):
                W=skew(jr[:,ctx.dof[column]]);dB=W@B-B@W
                support[:,j]-=dB@normal/radius-(B@normal)*(normal@dB@normal)/(2*radius**3)
            delta=tr['finger_search_delta_rad'][row,4*finger:4*finger+4]
            center_step=float(np.linalg.norm(jp[:,ctx.dof[columns]]@delta));support_step=float(np.linalg.norm(support@delta))
            assert center_step<=8e-6+1e-12 and np.max(abs(delta))<=.001+1e-12
            if dual_support:assert support_step<=8e-6+1e-12
            if normal_priority and np.any(delta):
                normal_step=float(normal@(support@delta));normal_step_max=max(normal_step_max,normal_step)
                assert normal_step<=1e-12, 'Normal-priority proposal must not predict outward motion'
                if row%20==0:
                    # Independently recover the nearest collidable object pair.
                    # These are geometric targets, never sensor-contact credit.
                    candidates=[]
                    for obj in objects:
                        ends=np.zeros(6);distance=mujoco.mj_geomDistance(model,data,geom,obj,.2,ends)
                        candidates.append((distance,ends.copy()))
                    distance,ends=min(candidates,key=lambda item:item[0])
                    actual_normal=ends[:3]-ends[3:]
                    actual_normal/=np.linalg.norm(actual_normal)
                    if actual_normal@(data.geom_xpos[geom]-ends[3:])<0:actual_normal=-actual_normal
                    closest_point_error=max(closest_point_error,float(np.max(abs(surface_point-ends[3:]))))
                    closest_normal_error=max(closest_normal_error,float(np.max(abs(normal-actual_normal))))
                    closest_pair_checks+=1
                    if finite_progress:
                        # Both finite FK bases must improve by the declared
                        # candidate threshold. These remain geometric checks;
                        # the actual native trajectory is separately replayed.
                        def gap_for(q):
                            geometry_scratch.qpos[:]=data.qpos
                            geometry_scratch.qpos[ctx.qpos]=q
                            mujoco.mj_forward(model,geometry_scratch)
                            return min(mujoco.mj_geomDistance(model,geometry_scratch,geom,g,.2,None) for g in objects)
                        actual_q=data.qpos[ctx.qpos].copy();actual_next=actual_q.copy();actual_next[columns]+=delta
                        nominal=tr['search_nominal_rad'][row]
                        bias=tr['finger_search_bias_rad'][row,columns-7]
                        reference_q=actual_q.copy();reference_q[columns]=nominal[columns]+bias-delta
                        reference_next=reference_q.copy();reference_next[columns]+=delta
                        gains=(distance-gap_for(actual_next),gap_for(reference_q)-gap_for(reference_next))
                        assert min(gains)>=2e-8-1e-12, ('Finite V6 progress not reconstructed',row,gains)
                        finite_progress_checks.append(dict(row=row,time_s=float(times[row]),
                            actual_posture_gap_gain_m=float(gains[0]),previous_reference_gap_gain_m=float(gains[1])))
            if np.any(delta):
                np.testing.assert_allclose(center_step,tr['finger_search_center_predicted_step_m'][row,finger],atol=1e-12,rtol=0)
                charge=max(center_step,support_step) if dual_support else center_step
                np.testing.assert_allclose(charge,tr['finger_search_charged_step_m'][row,finger],atol=1e-12,rtol=0)
                if dual_support:np.testing.assert_allclose(support_step,tr['finger_search_support_predicted_step_m'][row,finger],atol=1e-12,rtol=0)
            center_steps.append(center_step);support_steps.append(support_step)
        else:
            previous_target=None;surface_point=None;center_steps.append(0.);support_steps.append(0.)
            geometric_gap.append(np.nan);geometric_error.append(np.nan);target_points.append(np.full(3,np.nan));support_points.append(np.full(3,np.nan))
        if row+1<len(times):
            data.ctrl[ctx.actuator_ids]=tr['command_rad'][row];mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
    assert max(errors.values())<1e-7 and active_errors==0 and gap_error<1e-12,errors
    assert closest_point_error<1e-12 and closest_normal_error<1e-12
    final=load(directory/'final_state.npz')
    for key in ('qpos','qvel','ctrl','qacc_warmstart','time'):np.testing.assert_array_equal(getattr(data,key),final[key])
    assert peak==s['peak_force_n'] and armpeak==s['peak_arm_self_force_n']
    assert np.max(abs(tr['command_rad']-tr['sensor_command_rad']))<=.0024+1e-10
    assert not tr['coverage'].any() and not tr['covered_goals'].any()
    assert not tr['finger_search_active'][:,np.arange(4)!=finger].any()
    assert not (tr['finger_search_active']&tr['active']).any()
    for field in ('finger_posture_attempt_id','finger_recovery_context_id'):
        np.testing.assert_array_equal(tr[field],np.tile(controller[field],(len(times),1)))
    safe=tr['active']&(tr['forces_n']>=.2)&(tr['forces_n']<8.)
    contact=tr['active'][:,finger];first=float(times[np.flatnonzero(contact)[0]]) if contact.any() else None
    runs=intervals(safe[:,finger],times);stable=[r for r in runs if r[2]>=.15-1e-12]
    confirmation=float(times[stable[0][0]]+.15) if stable else None
    assert first==s['first_recontact_delay_s']
    assert bool(stable)==s['stable_recontact_observed']
    if confirmation is not None:assert abs(confirmation-s['stable_confirmation_delay_s'])<.002+1e-10
    per_finger=[];dt=np.diff(times)
    for f in range(4):
        missing=intervals(~tr['active'][:,f],times)
        per_finger.append(dict(finger=f,safe_contact_fraction=float(dt@safe[:-1,f]/dt.sum()),
            longest_actual_gap_s=max([0.]+[r[2] for r in missing]),peak_force_n=float(tr['forces_n'][:,f].max()),
            joint_net_deg=np.rad2deg(tr['q_rad'][-1,7+4*f:11+4*f]-tr['q_rad'][0,7+4*f:11+4*f]).tolist()))
    np.savez_compressed(directory/'independent_support_geometry.npz',time_s=times,
        actual_support_point_world_m=support_points,recovered_target_surface_point_world_m=target_points,
        target_plane_gap_m=geometric_gap,support_target_error_m=geometric_error,
        predicted_center_step_m=center_steps,predicted_support_step_m=support_steps)
    result=dict(status='PASSED',scope='MATCHED_LOCAL_SUPPORT_GEOMETRY_ABLATION_NOT_EXACT_CONTROLLER_CONTINUATION',
        source_snapshots_verified=len(m['source_hashes']),continuous_native_states=len(times),native_replay_max_errors=errors,
        active_disagreements=active_errors,physical_collision_gap_max_error_m=gap_error,unsafe_frames=unsafe_rows,
        arm_self_violation_frames=arm_violations,peak_force_n=peak,peak_arm_self_force_n=armpeak,
        selected_actual_first_contact_s=first,selected_stable_confirmation_s=confirmation,
        longest_selected_safe_span_s=max([0.]+[r[2] for r in runs]),selected_actual_contact_frames=int(contact.sum()),
        initial_collision_gap_m=float(tr['measured_selected_collision_gap_m'][0]),
        final_collision_gap_m=float(tr['measured_selected_collision_gap_m'][-1]),
        minimum_sampled_collision_gap_m=float(np.nanmin(tr['measured_selected_collision_gap_m'])),
        maximum_predicted_center_step_m=max(center_steps),maximum_predicted_support_step_m=max(support_steps),
        bounded_support_required=dual_support,normal_priority_required=normal_priority,
        maximum_outward_normal_proposal_m=normal_step_max,
        independent_closest_pair_checks=closest_pair_checks,
        closest_point_max_error_m=closest_point_error,closest_normal_max_error=closest_normal_error,
        finite_progress_required=finite_progress,independent_finite_progress_checks=finite_progress_checks,
        search_budget_initial=dict(elapsed_s=controller['finger_search_elapsed_s'].tolist(),distance_m=controller['finger_search_distance_m'].tolist(),
            loss_age_s=controller['finger_search_loss_age_s'].tolist()),
        search_budget_final=dict(elapsed_s=tr['finger_search_elapsed_s'][-1].tolist(),distance_m=tr['finger_search_distance_m'][-1].tolist()),
        initial_preload_preserved=True,initial_counter_restore_verified=initialization is None,
        shared_fresh_counter_initialization_verified=initialization is not None,
        sensor_inferred_episode_ages_verified=initialization is not None,per_finger=per_finger,
        arm_command_change_rad=float(np.max(abs(tr['command_rad'][:,:7]-source_frame['sensor_command_rad'][:7]))),
        stop_reason=s['stop_reason'],whole_traversal_evidence=False,coverage_counted=False,
        trace_sha256=digest(directory/'trace.npz'),audit_source_sha256=digest(__file__))
    (directory/'independent_native_audit.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('pair',type=Path)
    parser.add_argument('--cases',nargs=2,choices=('V3','V4','V5','V6'),default=('V3','V4'))
    args=parser.parse_args();assert args.cases[0]!=args.cases[1]
    cases={name:audit(args.pair/name) for name in args.cases}
    manifests={name:json.loads((args.pair/name/'manifest.json').read_text()) for name in cases}
    left,right=args.cases
    assert manifests[left]['source_hashes']==manifests[right]['source_hashes']
    for name in ('initial.npz','controller_frame.npz','controller_handoff.npz'):
        a,b=load(args.pair/left/name),load(args.pair/right/name);assert set(a)==set(b)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    result=dict(status='PASSED',same_exact_initial_physics=True,same_logged_controller_initialization=True,
        same_frozen_sources=True,scope='LOCAL_SEARCH_GEOMETRY_ABLATION_NO_COVERAGE',cases=cases)
    (args.pair/'independent_pair_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
