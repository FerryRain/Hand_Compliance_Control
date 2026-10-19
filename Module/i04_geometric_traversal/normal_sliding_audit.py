"""Extra v3.7 geometry-proposal evidence; no contact or coverage authority."""
from __future__ import annotations
import ast
from hashlib import sha256
from pathlib import Path
import numpy as np
import mujoco

PROTOCOL='i04-mustard-coordinated-finger-sliding.development.v3.7-normal'
PREFIX='finger_search_'

def _skew(w):
    return np.array([[0.,-w[2],w[1]],[w[2],0.,-w[0]],[-w[1],w[0],0.]])

def normal_search_checks(directory,manifest,trace,maximum_native_samples=201):
    """Check all saved scalars and independently rebuild selected native FK.

    The stored trace is normally100Hz. This is explicitly a sample audit, not
    evidence that every2ms proposal or native interval was recorded/replayed.
    """
    from .sliding_audit import _native_context,_restore_native
    assert manifest['protocol']==PROTOCOL
    assert manifest['active_finger_source']=='Module/i04_geometric_traversal/active_fingers_v6.py'
    assert manifest['force_release_source']=='Module/i04_geometric_traversal/reference_force_release.py'
    required=('active_fingers.py','active_fingers_v3.py','active_fingers_v4.py','active_fingers_v5.py',
              'active_fingers_v6.py','bounded_finger_step.py','reference_force_release.py')
    assert all('Module/i04_geometric_traversal/'+name in manifest['source_hashes'] for name in required)
    cfg=manifest['normal_priority_search']
    assert cfg['gradient_magnitude_preserved'] and cfg['damping_m']==.001
    assert cfg['finite_fk_checks']==['actual_posture','previous_finger_reference']
    assert cfg['maximum_center_step_m']<=8e-6 and cfg['maximum_support_step_m']<=8e-6
    minimum=float(cfg['minimum_gap_improvement_m']);trials=int(cfg['backtracking_steps'])
    assert minimum>=2e-8 and 1<=trials<=7
    release=manifest['applied_reference_relief']
    assert release['reference']=='Actual prior guarded command'
    assert release['original_site_jacobian_and_damping'] and release['original_slew_range_force_guards']
    entry=ast.parse((directory/'source_snapshot'/manifest['runtime_entrypoint']).read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='applied_reference_force_release']
    assert len(calls)==1 and len(calls[0].args)>=4
    assert isinstance(calls[0].args[3],ast.Name) and calls[0].args[3].id=='last'
    search_source=ast.parse((directory/'source_snapshot'/manifest['active_finger_source']).read_text(encoding='utf-8'))
    # V6 finite geometry is allowed only on its separately allocated scratch.
    setters=[n for n in ast.walk(search_source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='set_joint_state']
    assert setters and all(len(n.args)>=2 and isinstance(n.args[1],ast.Attribute) and n.args[1].attr=='_gap_scratch' for n in setters)
    shape=trace['active'].shape
    scalars=('closest_collision_gap_m','center_step_m','support_step_m','charged_step_m',
             'predicted_gap_change_m','accepted_scale','actual_posture_gap_gain_m','previous_reference_gap_gain_m')
    for name in scalars:
        value=trace[PREFIX+name];assert value.shape==shape and np.isfinite(value).all(),name
    reasons=trace[PREFIX+'solver_reason'];assert reasons.shape==shape
    seeking=trace[PREFIX+'active'];accepted=reasons=='DAMPED_NORMAL_FINITE_PROGRESS'
    np.testing.assert_array_equal(accepted,seeking)
    np.testing.assert_array_equal(trace[PREFIX+'state']=='SEEK',seeking)
    assert trace[PREFIX+'requires_rebase'].shape==shape
    assert trace[PREFIX+'surface_points_world_m'].shape==shape+(3,)
    assert trace[PREFIX+'closest_object_geom_ids'].shape==shape
    assert np.issubdtype(trace[PREFIX+'closest_object_geom_ids'].dtype,np.integer)
    for name in ('center_step_m','support_step_m','charged_step_m'):
        value=trace[PREFIX+name]
        assert np.all(value>=0.) and np.all(value<=8e-6+1e-12),name
        assert not np.any(value[~accepted]), 'Rejected/unused proposal charged displacement'
    np.testing.assert_allclose(trace[PREFIX+'charged_step_m'],np.maximum(trace[PREFIX+'center_step_m'],trace[PREFIX+'support_step_m']),atol=1e-12,rtol=0)
    assert np.all(trace[PREFIX+'predicted_gap_change_m'][accepted]<0.)
    for name in ('actual_posture_gap_gain_m','previous_reference_gap_gain_m'):
        assert np.all(trace[PREFIX+name][accepted]>=minimum-1e-12)
        assert not np.any(trace[PREFIX+name][~accepted])
    scales=trace[PREFIX+'accepted_scale'][accepted]
    assert all(np.any(abs(scale-.5**np.arange(trials))<1e-12) for scale in scales)
    assert not np.any(trace[PREFIX+'accepted_scale'][~accepted])
    holds=trace[PREFIX+'state']=='NO_LEGAL_INWARD_STEP'
    for f in range(4):assert not np.any(trace[PREFIX+'delta_rad'][holds[:,f],4*f:4*f+4])
    pairs=np.argwhere(accepted)
    if not len(pairs):
        return dict(normal_priority_search_audited=True,normal_proposals_logged=0,
            normal_auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            normal_native_geometry_samples_checked=0,geometry_scope='No accepted normal search proposals in saved trace')
    indices=np.unique(np.r_[np.linspace(0,len(pairs)-1,min(maximum_native_samples,len(pairs)),dtype=int),
        np.argmin(trace[PREFIX+'actual_posture_gap_gain_m'][accepted]),
        np.argmin(trace[PREFIX+'previous_reference_gap_gain_m'][accepted]),
        np.argmax(trace[PREFIX+'center_step_m'][accepted]),np.argmax(trace[PREFIX+'support_step_m'][accepted])])
    ctx=_native_context(directory);model,data,h=ctx.model,ctx.data,ctx.handles
    scratch=mujoco.MjData(model);objects=[int(g) for g in h.object_geom_ids if model.geom_contype[g]!=0]
    errors=dict(closest_gap_m=0.,surface_point_m=0.,normal=0.,center_step_m=0.,support_step_m=0.,
                predicted_gap_change_m=0.,actual_posture_gap_gain_m=0.,previous_reference_gap_gain_m=0.)
    checked=[]
    for selection in indices:
        row,finger=map(int,pairs[selection]);columns=7+4*finger+np.arange(4);geom=int(h.tip_geom_ids[finger])
        _restore_native(ctx,{name:trace[name][row] for name in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart')})
        candidates=[]
        for obj in objects:
            ends=np.zeros(6);gap=mujoco.mj_geomDistance(model,data,geom,obj,.2,ends)
            candidates.append((gap,obj,ends.copy()))
        gap,obj,ends=min(candidates,key=lambda x:x[0]);normal=ends[:3]-ends[3:];normal/=np.linalg.norm(normal)
        if normal@(data.geom_xpos[geom]-ends[3:])<0:normal=-normal
        assert obj==trace[PREFIX+'closest_object_geom_ids'][row,finger]
        for key,actual,saved in (('closest_gap_m',gap,trace[PREFIX+'closest_collision_gap_m'][row,finger]),
            ('surface_point_m',ends[3:],trace[PREFIX+'surface_points_world_m'][row,finger]),
            ('normal',normal,trace[PREFIX+'target_normals_world'][row,finger])):
            errors[key]=max(errors[key],float(np.max(abs(np.asarray(actual)-saved))))
        R=data.geom_xmat[geom].reshape(3,3);B=R@np.diag(model.geom_size[geom]**2)@R.T;radius=np.sqrt(normal@B@normal)
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jacGeom(model,data,jp,jr,geom)
        center=jp[:,ctx.dof[columns]];support=center.copy()
        for j,col in enumerate(columns):
            W=_skew(jr[:,ctx.dof[col]]);dB=W@B-B@W
            support[:,j]-=dB@normal/radius-(B@normal)*(normal@dB@normal)/(2*radius**3)
        delta=trace[PREFIX+'delta_rad'][row,columns-7]
        for key,actual in (('center_step_m',np.linalg.norm(center@delta)),('support_step_m',np.linalg.norm(support@delta)),
                           ('predicted_gap_change_m',normal@(support@delta))):
            errors[key]=max(errors[key],abs(float(actual)-trace[PREFIX+key][row,finger]))
        def gap_at(q):
            scratch.qpos[:]=data.qpos;scratch.qpos[ctx.qpos]=q;mujoco.mj_forward(model,scratch)
            return min(mujoco.mj_geomDistance(model,scratch,geom,g,.2,None) for g in objects)
        q=trace['q_rad'][row].copy();after=q.copy();after[columns]+=delta
        reference=q.copy();reference[columns]=trace[PREFIX+'nominal_rad'][row,columns]+trace[PREFIX+'bias_rad'][row,columns-7]-delta
        next_reference=reference.copy();next_reference[columns]+=delta
        gains=dict(actual_posture_gap_gain_m=gap-gap_at(after),previous_reference_gap_gain_m=gap_at(reference)-gap_at(next_reference))
        for key,actual in gains.items():
            assert actual>=minimum-1e-12
            errors[key]=max(errors[key],abs(actual-trace[PREFIX+key][row,finger]))
        checked.append(dict(row=row,time_s=float(trace['time_s'][row]),finger=finger,**gains))
    assert max(errors.values())<1e-9,errors
    return dict(normal_priority_search_audited=True,normal_proposals_logged=int(accepted.sum()),
        normal_auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        explicit_no_progress_holds_logged=int(holds.sum()),normal_native_geometry_samples_checked=len(checked),
        normal_geometry_reconstruction_max_errors=errors,normal_native_geometry_samples=checked,
        geometry_scope='All saved proposal scalars checked; selected100Hz native states independently rebuild closest-pair/Jacobian/two finiteFK gains. This is not full2ms proposal replay.',
        normal_targets_count_as_measured_contact=False,normal_targets_award_coverage=False,
        relief_source_and_prior_applied_argument_verified=True)
