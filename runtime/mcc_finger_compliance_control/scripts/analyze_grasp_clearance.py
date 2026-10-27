"""Paired teacher posture audit; strict/found and actual per-part solver loads."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from true_tip_waypoint_optimizer import SplitHullGeometry
from object_catalog import load_object_config


def smooth_keyframe_reference(frames, q, valid, total):
    """Same valid-knot filtering and smoothstep as the collector reference."""
    frames = np.asarray(frames).reshape(-1)[valid]
    q = np.asarray(q).reshape(-1,16)[valid]
    order = np.argsort(frames,kind='stable'); frames=frames[order];q=q[order]
    unique = np.r_[True,np.diff(frames)>0];frames=frames[unique];q=q[unique]
    if len(frames)<2: raise ValueError('Need two valid grasp boundaries')
    t=np.arange(total)
    right=np.clip(np.searchsorted(frames,t,side='right'),1,len(frames)-1);left=right-1
    phase=np.clip((t-frames[left])/np.maximum(frames[right]-frames[left],1),0,1)
    phase=phase*phase*(3-2*phase)
    return (1-phase[:,None])*q[left]+phase[:,None]*q[right]


def loss_stats(mask):
    streak=longest=0;first=None
    for frame,ok in enumerate(mask):
        streak=0 if ok else streak+1;longest=max(longest,streak)
        if first is None and streak==6:first=frame-5
    return dict(longest_observed_four_loss_frames=longest,first_loss_over5_frame=first)


def geometry_report(source, candidate, geometry, sample_stride=1):
    rows=[]
    with h5py.File(source) as old,h5py.File(candidate) as new:
        palm=old['palm_pose_object'][:,0]
        assert np.array_equal(old['palm_pose_object'][:],new['palm_pose_object'][:])
        assert np.array_equal(old['palm_twist_object'][:],new['palm_twist_object'][:])
        for label,h in (('baseline',old),('full',new)):
            frames=np.asarray(h['grasp_keyframe_frame_index']).reshape(-1)
            knots=np.asarray(h['grasp_keyframe_q']).reshape(-1,16)
            valid=np.asarray(h['grasp_keyframe_valid']).reshape(-1).astype(bool)
            accepted=[geometry.evaluate(q,palm[f],self_collision=True,non_tip_clearance=True)
                      for q,f in zip(knots[valid],frames[valid])]
            d=np.stack([x['non_tip_distance'] for x in accepted])
            self_depth=np.array([x['self_penetration'] for x in accepted])
            dense=smooth_keyframe_reference(frames,knots,valid,len(palm))
            sampled=np.arange(0,len(palm),sample_stride)
            measurement=[geometry.evaluate(dense[f],palm[f],self_collision=True,non_tip_clearance=True)
                         for f in sampled]
            dd=np.stack([x['non_tip_distance'] for x in measurement])
            ds=np.array([x['self_penetration'] for x in measurement])
            pad=np.stack([x['distance'] for x in measurement])
            margin=float(new.attrs['non_tip_clearance_m'])
            self_limit=float(new.attrs['max_cross_finger_penetration_m'])
            rows.append(dict(variant=label,accepted_knots=int(valid.sum()),total_knots=len(valid),
                accepted_non_tip_min_mm=float(d.min()*1000),
                accepted_clearance_recheck_pass_fraction=float((d.min(-1)>=margin-1e-6).mean()),
                accepted_cross_finger_max_mm=float(self_depth.max()*1000),
                accepted_knots_non_tip_contact=int((d.min(-1)<0).sum()),
                accepted_knots_cross_finger_over_tolerance=int((self_depth>self_limit+1e-6).sum()),
                reference_scope='smoothstep posture prior through accepted knots; not actuator commands or actual dynamics',
                sampled_control_frames=len(sampled),sample_stride=sample_stride,
                reference_non_tip_min_mm=float(dd.min()*1000),
                reference_cross_finger_max_mm=float(ds.max()*1000),
                reference_non_tip_contact_fraction=float((dd.min(-1)<0).mean()),
                reference_clearance_pass_fraction=float((dd.min(-1)>=margin-1e-6).mean()),
                reference_joint_step_max_rad=float(abs(np.diff(dense,axis=0)).max()),
                reference_tip_distance_outside_legacy_band_fraction=float(((pad>.004)|(pad<-.012)).any(-1).mean())))
    return dict(palm_reference_unchanged=True,non_tip_geom_names=geometry.non_tip_names,
                geometry_rows=rows)


def rollout_report(directory, total):
    summary=json.loads((directory/'summary.json').read_text())
    z=np.load(directory/'trace.npz')
    start=summary.get('actual_motion_start_step',1200)
    task=z['step']>start if start is not None else np.zeros(len(z['step']),bool)
    strict=z['strict'][task,0].copy();found=z['found'][task,0].copy()
    forces=z['force'][task,0]
    net=z['part_net_force'][task,0];single=z['part_single_force'][task,0]
    error=summary['error'];stop=bool(error and 'Independent collection safety stop' in error)
    if stop and len(strict):strict[-1]=False;found[-1]=False
    complete=bool(not error and len(strict)==total)
    initial=((z['step']>start-20)&(z['step']<=start) if start is not None
             else np.zeros(len(z['step']),bool))
    initial_rate=float(z['strict'][initial,0].all(-1).mean()) if initial.any() else None
    with h5py.File(summary['plan']) as h:
        planned=h['palm_pose_object'][:,0]
    actual=z['palm'][task,0]
    frame=np.clip(z['step'][task]-(start+1 if start is not None else 1201),0,len(planned)-1)
    position_error=np.linalg.norm(actual[:,:3]-planned[frame,:3],axis=-1)*1000
    aq=actual[:,3:]/np.linalg.norm(actual[:,3:],axis=-1,keepdims=True)
    pq=planned[frame,3:]/np.linalg.norm(planned[frame,3:],axis=-1,keepdims=True)
    angle_error=np.degrees(2*np.arccos(np.clip(np.abs((aq*pq).sum(-1)),0,1)))
    no_extra_load=(np.maximum(net[:,1:],single[:,1:]).max(-1)<.05)
    # Per-tick arrays retain every physics-substep peak. Summary maxima also
    # include initialization, so using them here can reject a clean motion.
    part_peaks=[max(float(net[:,j].max()) if len(net) else 0,
                    float(single[:,j].max()) if len(single) else 0)
                for j,_ in enumerate(summary['force_part_labels'])]
    row=dict(status='completed' if complete else 'diagnostic_prefix'
             if summary.get('diagnostic_prefix_completed',False) else 'force_stop' if stop else 'execution_error',
        requested_frames=total,observed_frames=len(strict),
        stop_task_frame=int(z['step'][task][-1]-start-1) if stop and task.any() else None,
        initial_strict_four_rate=initial_rate,
        strict_healthy_four_rate_full=float(strict.all(-1).sum()/total),
        found_four_rate_full=float(found.all(-1).sum()/total),
        strict_healthy_four_rate_observed=float(strict.all(-1).mean()) if len(strict) else None,
        found_four_rate_observed=float(found.all(-1).mean()) if len(found) else None,
        strict_per_finger_full=(strict.sum(0)/total).tolist(),found_per_finger_full=(found.sum(0)/total).tolist(),
        tip_only_four_rate_full=float((strict.all(-1)&no_extra_load).sum()/total),
        tip_only_definition='Post-step strict four fingertips AND no >0.05N net/single-contact load on any non-tip geom throughout the same 10ms control tick',
        peak_stop_sensor_force_n=float(forces.max()) if len(forces) else 0,
        part_peak_force_n=dict(zip(summary['force_part_labels'],part_peaks)),
        part_p95_force_n=dict(zip(summary['force_part_labels'],
                                 (np.quantile(np.maximum(net,single),.95,axis=0).tolist()
                                  if len(net) else [0.]*len(summary['force_part_labels'])))),
        peak_part_events=summary['peak_part_events'],error=error,
        complete_and_all_hand_parts_under100N=bool(complete and max(part_peaks)<=100),
        complete_strict90=bool(complete and strict.all(-1).mean()>=.90),
        collector_sha256=summary['collector_sha256'],
        palm_position_error_mm=(dict(p95=float(np.quantile(position_error,.95)),max=float(position_error.max()))
                                if len(position_error) else None),
        palm_rotation_error_deg=(dict(p95=float(np.quantile(angle_error,.95)),max=float(angle_error.max()))
                                if len(angle_error) else None),
        **loss_stats(strict.all(-1)))
    return row,dict(strict=strict.all(-1),found=found.all(-1),force=forces,
                    parts=np.maximum(net,single),q=z['q'][task,0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('mcc_finger_compliance_control/data/grasp_clearance_20261013'))
    p.add_argument('--source',type=Path,default=Path('mcc_finger_compliance_control/data/long_palm_oracle_geometry_20261013'))
    p.add_argument('--ids',type=int,nargs='+',default=[39,45])
    p.add_argument('--geometry-only',action='store_true')
    p.add_argument('--geometry-stride',type=int,default=1)
    a=p.parse_args();geometry=SplitHullGeometry(load_object_config('ycb_mustard'))
    rows=[];traces={}
    for eid in a.ids:
        source=a.source/f'plan_{eid}.h5';candidate=a.root/f'plan_{eid}_full.h5'
        row=dict(trajectory_id=eid,**geometry_report(source,candidate,geometry,a.geometry_stride))
        if not a.geometry_only:
            with h5py.File(source) as h:total=len(h['palm_pose_object'])
            for label in ('baseline','full'):
                row[label],traces[(eid,label)]=rollout_report(a.root/f'{label}_{eid}',total)
            assert row['baseline']['collector_sha256']==row['full']['collector_sha256']
            common=min(len(traces[(eid,x)]['q']) for x in ('baseline','full'))
            row['paired_difference']=dict(
                strict_pp=100*(row['full']['strict_healthy_four_rate_full']-row['baseline']['strict_healthy_four_rate_full']),
                found_pp=100*(row['full']['found_four_rate_full']-row['baseline']['found_four_rate_full']),
                strict_common_prefix_pp=100*(traces[(eid,'full')]['strict'][:common].mean()-traces[(eid,'baseline')]['strict'][:common].mean()),
                found_common_prefix_pp=100*(traces[(eid,'full')]['found'][:common].mean()-traces[(eid,'baseline')]['found'][:common].mean()),
                actual_q_mean_change_common_prefix_rad=(traces[(eid,'full')]['q'][:common]-traces[(eid,'baseline')]['q'][:common]).mean(0).tolist())
        rows.append(row)
        print('[GRASP-AUDIT]',eid,json.dumps(row.get('paired_difference',row['geometry_rows']),ensure_ascii=False),flush=True)
    name='geometry_continuity.json' if a.geometry_only else 'paired_analysis.json'
    result=dict(rows=rows,scope='Same palm reference, teacher/MCC/physics and safety; changed offline grasp boundary conditions. Single replay per variant, no policy training',
                candidate=dict(non_tip_clearance_m=.002,posture_gain=.18,max_cross_finger_penetration_m=.0001,
                               acceptance_scope='This physical canary uses v1 boundary snapshots. Recomputed serialized clearance and interpolated continuity are reported separately; no full-path clearance certification.'),
                stop_definition='100N actual solver-substep and post-step pad/connector/net-sum stop; other hand links and individual contact maxima are independently diagnosed',
                strict_definition='Existing full true-fingertip loaded contact; report found separately; tip_only is an additional diagnostic')
    (a.root/name).write_text(json.dumps(result,indent=2)+'\n')
    if not a.geometry_only:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,len(a.ids),figsize=(6*len(a.ids),9),squeeze=False)
        for col,eid in enumerate(a.ids):
            for label,color in (('baseline','tab:blue'),('full','tab:orange')):
                t=traces[(eid,label)]
                for j,kind in enumerate(('strict','found')):
                    padded=np.zeros(6000);padded[:len(t[kind])]=t[kind]
                    average=np.convolve(padded,np.ones(100)/100,mode='valid')
                    axes[j,col].plot(np.arange(len(average))*.01,average,label=label,color=color)
                    axes[j,col].set_ylabel(kind+' four (1s mean)');axes[j,col].set_ylim(-.03,1.03)
                tt=np.arange(len(t['parts']))*.01
                axes[2,col].plot(tt,t['parts'][:,0],color=color,label=label+' fingertip')
                axes[2,col].plot(tt,t['parts'][:,1:].max(-1),color=color,linestyle='--',label=label+' non-tip')
            axes[0,col].set_title(f'Reference {eid}');axes[2,col].axhline(100,color='red',alpha=.5)
            axes[2,col].set_ylabel('Actual per-part force peak N');axes[2,col].set_xlabel('Simulation task time (s)')
            for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(a.root/'paired_contact_force.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
