"""Audit no-search collection without excluding failed contact trajectories."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
import mujoco
from collect_contact_wm import job_fingers,selected_contacts


def longest(mask):
    x=np.r_[False,np.asarray(mask,dtype=bool),False].astype(int)
    return int((np.flatnonzero(np.diff(x)==-1)-np.flatnonzero(np.diff(x)==1)).max(initial=0))


def audit(folder):
    with h5py.File(folder/'rollout.h5') as h:
        meta=json.loads(h.attrs['metadata']);r={k:v[:] for k,v in h['raw_100hz'].items()}
    with np.load(folder/'substeps.npz') as a:
        d={k:a[k] for k in a.files}
    m=mujoco.MjModel.from_binary_path(str(folder/'physics_model.mjb'))
    hinge=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    qi=np.array([m.jnt_qposadr[j] for j in hinge]);vi=np.array([m.jnt_dofadr[j] for j in hinge])
    motion=d['phase'][:,0]==1
    balance=d['dynamics_mass_times_acc']-d['qfrc_smooth']-d['qfrc_constraint']
    assert all(np.isfinite(v).all() for v in d.values())
    eq_error=d['qpos_after'][:,:,qi]-d['qpos_before'][:,:,qi]-m.opt.timestep*d['qvel_after'][:,:,vi]
    integrity=dict(finite=True,max_position_discontinuity=float(np.abs(d['qpos_before'][1:]-d['qpos_after'][:-1]).max()),
        max_velocity_discontinuity=float(np.abs(d['qvel_before'][1:]-d['qvel_after'][:-1]).max()),
        max_hinge_integration_error_rad=float(np.abs(eq_error).max()),
        joint_torque_balance_nm_quantiles=np.quantile(np.abs(balance[motion][:,:,vi]),[.5,.95,.99,1]).tolist(),
        max_actuator_torque_nm=float(np.abs(d['qfrc_actuator'][:,:,vi]).max()),
        nominal_q_max_range_rad=float(np.ptp(r['pre_mcc_q'][1:],axis=0).max()),
        transient_search_max_m=float(np.abs(r['mcc_transient_offset']).max()),
        held_target_error_rad=float(r['held_error'].max()))
    rows=[]
    finger_jobs=[(lane,{**job,'finger':finger}) for lane,job in enumerate(meta['jobs']) for finger in job_fingers(job)]
    for lane,job in finger_jobs:
        f=job['finger'];off=r['mcc_force_offset'][:,lane,f]
        absent=~r['pad_physical_found'][:-1,lane,f].astype(bool)
        wrong=int(((np.abs(off[1:])-np.abs(off[:-1])>1e-7)&absent).sum())
        incremental=meta.get('mcc_candidate_overrides',{}).get('incremental_force_compliance',False)
        mask=r['strict'][1:,lane,f].astype(bool)
        force=np.linalg.norm(d['pad_force'][motion,lane,f],axis=-1)
        maximum=int(np.argmax(np.linalg.norm(d['object_physical_force'][:,lane],axis=-1)))
        pair=d['object_physical_geom_ids'][maximum,lane]
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in pair]
        row=dict(**job,lane=lane,strict=float(mask.mean()),found=float(r['found'][1:,lane,f].mean()),
            longest_loss_s=longest(~mask)*.01,force_peak_n=float(force.max()),force_p95_n=float(np.quantile(force,.95)),
            longest_force_above25n_s=longest(force>25)*.001,object_peak_n=float(r['object_peak_n'][1:,lane].max()),
            self_peak_n=float(r['self_peak_n'][1:,lane].max()),absent_contact_offset_growth_count=None if incremental else wrong,
            offset_semantics='current-Jacobian diagnostic projection, not integrated control memory' if incremental else 'scalar force integrator',
            force_max_contact_pair=names,force_max_contact_penetration_m=float(-d['object_physical_distance'][maximum,lane]),
            offset_min_max_mm=(1000*np.array([off.min(),off.max()])).tolist())
        row['teacher_contact_force_gate']=bool(row['strict']>=.95 and row['found']>=.95 and
            row['longest_loss_s']<=.1 and row['force_p95_n']<=15 and row['longest_force_above25n_s']<=.02 and
            row['object_peak_n']<=100 and row['self_peak_n']<=100)
        rows.append(row)
    lane_rows=[]
    for lane,job in enumerate(meta['jobs']):
        active=np.zeros(4,dtype=bool);active[job_fingers(job)]=True
        mask=selected_contacts(r['strict'][1:,lane].astype(bool),active)
        found=selected_contacts(r['found'][1:,lane].astype(bool),active)
        x=dict(lane=lane,active_fingers=job_fingers(job),strict_all_active=float(mask.mean()),
            found_all_active=float(found.mean()),strict_all_four=float(r['strict'][1:,lane].all(-1).mean()),
            found_all_four=float(r['found'][1:,lane].all(-1).mean()),longest_all_active_loss_s=longest(~mask)*.01)
        x['teacher_contact_force_gate']=bool(all(v['teacher_contact_force_gate'] for v in rows if v['lane']==lane)
            and x['strict_all_active']>=.95 and x['found_all_active']>=.95 and x['longest_all_active_loss_s']<=.1)
        lane_rows.append(x)
    return dict(object=meta['object_id'],scope=meta.get('collection_scope','single_finger'),
        mode=meta.get('contact_regulation'),integrity=integrity,rows=rows,
        lanes=lane_rows,teacher_contact_force_pass=sum(x['teacher_contact_force_gate'] for x in lane_rows),
        teacher_finger_contact_force_pass=sum(x['teacher_contact_force_gate'] for x in rows),
        warning='This gate uses contact and force only; no motion/sliding coverage cutoff is applied. Failed task trajectories remain WM observations pending physical-integrity review, not teacher labels.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports=[audit(f) for f in sorted((a.root/'episodes/single_finger'/a.tag).iterdir()) if (f/'rollout.h5').exists()]
    a.output.write_text(json.dumps(dict(tag=a.tag,reports=reports),indent=2))
    for r in reports:print(r['object'],r['integrity'],'teacher contact/force',r['teacher_contact_force_pass'],'/',len(r['rows']))
