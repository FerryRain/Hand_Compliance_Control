"""Audit actual motion separately from nominal smoothing and reset completion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from surface_mcc_finger import FullHandMCCFingerConfig, FullHandMCCFingerController


def quantiles(x):
    return dict(zip(['p50','p95','p99','max'], np.quantile(np.abs(x),[.5,.95,.99,1]).tolist()))


def audit(path):
    result=json.loads(path.read_text()); trace=np.load(result['trace'])
    ctl=FullHandMCCFingerController(FullHandMCCFingerConfig(**result['source_config']))
    reports=[]
    for lane,row in enumerate(result['lanes']):
        n=row['frames']; q=trace['q'][:n,lane]
        margin=np.minimum(q-ctl.lower,ctl.upper-q).min(-1)
        force=trace['force'][:n,lane].max(-1); own=trace['self_force'][:n,lane]
        r=dict(episode=row['episode_id'],frames=n,complete=row['complete'],contact=row['full'],
               minimum_joint_margin_rad=float(margin.min()),outside_frames=int((margin < -1e-6).sum()),
               object_over100_frames=int((force>100).sum()),self_over100_frames=int((own>100).sum()),
               self_peak_n=float(own.max()),motion={})
        for name in ['nominal','command','q']:
            delta=np.diff(trace[name][100:n,lane],axis=0)
            r['motion'][name]=dict(speed_rad_s=quantiles(delta/.01),acceleration_rad_s2=quantiles(np.diff(delta,axis=0)/.0001))
        if 'reset_phase' in trace:
            phase=trace['reset_phase'][:n,lane];cursor=trace['reset_reference_frame'][:n,lane]
            frozen=(phase[:-1]!=0)&(phase[1:]!=0)
            events=[e for e in result['palm_reset']['events'] if e['lane']==lane]
            r['recovery']=dict(final_phase=result['palm_reset']['final_phase'][lane],
                final_cursor=int(cursor[-1]),paused_frames=int(((phase!=0)&(phase!=8)).sum()),
                pause_cursor_changes=int((np.diff(cursor)[frozen]!=0).sum()),events=events,
                resumed=sum(e['reason']=='resume_remaining_reference' for e in events),
                measured_reshapes=sum(e['reason']=='measured_pregrasp_reached_unloaded' for e in events))
        reports.append(r)
    output=dict(source=str(path.resolve()),lanes=reports,
                caveat='Acceleration is finite-difference diagnostic, not a torque/penetration certificate. '
                       'Overforce counts are control frames containing a physical-substep peak. '
                       'Contact includes recovery time; full completion requires remaining reference execution.')
    filtered=path.with_suffix('.reference_filter.npz')
    if filtered.exists():
        z=np.load(filtered);override=z['bound_override']
        output['reference_filter']=dict(speed=quantiles(z['velocity']),acceleration=quantiles(z['acceleration']),
            lag_rad=quantiles(z['lag']),bound_priority_overrides=int(override.sum()),
            bounded_velocity=bool(np.max(abs(z['velocity']))<=2.0001),
            bounded_acceleration_except_bound_priority=bool(np.max(abs(z['acceleration'][~override]))<=40.002))
    protocol=path.parent/'protocol.json'
    if protocol.exists():
        frozen=json.loads(protocol.read_text())
        if 'frozen_sha256' in frozen:
            hashes=frozen['frozen_sha256']
        elif 'fixed_weights_and_mcc' in frozen:
            hashes=frozen['fixed_weights_and_mcc']
        else:
            raise ValueError('Protocol has no frozen controller/weight manifest')
        output['frozen_changes']=[p for p,h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    path.with_suffix('.audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path)
    audit(parser.parse_args().result)
