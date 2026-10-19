"""Read-only thumb-loss, exhausted-budget and scheduler diagnosis."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def analyze(directory, output, end_s=600.):
    directory, output = Path(directory), Path(output)
    keys = ('time_s', 'q_rad', 'palm_pose_world', 'pad_centers_world_m', 'active', 'forces_n',
        'phase', 'command_phase', 'stroke_id', 'stroke_distance_m', 'finger_search_state',
        'finger_search_loss_age_s', 'finger_search_elapsed_s', 'finger_search_distance_m',
        'finger_search_target_positions_world_m', 'finger_search_target_normals_world',
        'finger_recovery_requests', 'finger_posture_attempts_this_episode', 'finger_posture_attempt_id')
    with np.load(directory/'trace.npz', allow_pickle=False) as saved:
        x = {key: saved[key] for key in keys}
    t = x['time_s']; finger = 3
    events = json.loads((directory/'events.json').read_text())
    windows = {}
    for start, end in ((18., 300.), (18., end_s), (93.4, end_s)):
        rows = np.flatnonzero((t >= start) & (t <= end)); a, b = rows[0], rows[-1]
        dt = np.diff(t[a:b+1]); state = x['finger_search_state'][a:b, finger]
        available = np.linalg.norm(x['finger_search_target_normals_world'][a:b+1, finger], axis=1) > .5
        dist = np.linalg.norm(x['finger_search_target_positions_world_m'][a:b+1, finger]
                              -x['pad_centers_world_m'][a:b+1, finger], axis=1)
        poses = x['palm_pose_world'][a:b+1]
        rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
        angles = (rotations[0].inv()*rotations).magnitude()
        windows[f'{start:g}-{end:g}'] = dict(duration_s=end-start,
            contact_samples=int(x['active'][a:b+1, finger].sum()),
            state_time_s={str(k): float(dt[state == k].sum()) for k in np.unique(state)},
            search_elapsed_start_end_s=x['finger_search_elapsed_s'][[a,b], finger].tolist(),
            search_distance_start_end_m=x['finger_search_distance_m'][[a,b], finger].tolist(),
            physical_loss_age_start_end_s=x['finger_search_loss_age_s'][[a,b], finger].tolist(),
            monitor_requests_start_end=x['finger_recovery_requests'][[a,b], finger].tolist(),
            accepted_posture_attempts_start_end=x['finger_posture_attempts_this_episode'][[a,b], finger].tolist(),
            target_available_s=float(dt[available[:-1]].sum()),
            target_distance_quantile_m=np.quantile(dist[available], [0,.25,.5,.75,1]).tolist() if available.any() else None,
            target_within_60mm_s=float(dt[available[:-1] & (dist[:-1] <= .06)].sum()),
            actual_thumb_joint_range_deg=np.rad2deg(np.ptp(x['q_rad'][a:b+1, 19:23], axis=0)).tolist(),
            maximum_actual_palm_displacement_m=float(np.linalg.norm(poses[:,:3]-poses[0,:3],axis=1).max()),
            maximum_actual_palm_rotation_deg=float(np.rad2deg(angles.max())))
    requests = []
    for event in events:
        if event['kind'] == 'COORDINATED_RECOVERY_PLAN' and event['request']['finger'] == finger:
            fp, wp = event['finger_plan'], event['wrist_plan'] or {}
            near = fp.get('nearest_attempt', {})
            requests.append(dict(time_s=event['time_s'], request=event['request'], accepted=event['accepted'],
                finger_reason=fp.get('reason'), targets=fp.get('attempted_target_count'), nearest_reason=near.get('reason'),
                nearest_best_residual_m=min((c.get('target_residual_m', float('inf')) for c in near.get('candidates',[])), default=None),
                neighbor_rejection_counts=dict(Counter(c.get('reason') for item in fp.get('neighbor_attempts',[])
                    for c in item.get('planner',{}).get('candidates',[]))),
                wrist_reason=wp.get('reason'),
                wrist_rejection_counts=dict(Counter(c.get('reason') for c in wp.get('candidates',[])))))
    transitions = []
    for event in events:
        if event['kind'] == 'LIFT_RECONTACT':
            start = max(item['time_s'] for item in events if item['kind'] == 'LIFT_COMMIT' and item['time_s'] < event['time_s'])
            a, b = np.searchsorted(t, start), np.searchsorted(t, event['time_s'])
            poses = x['palm_pose_world'][[a,b]]; rotations = Rotation.from_quat(poses[:,[4,5,6,3]])
            transitions.append(dict(commit_s=start, recontact_s=event['time_s'],
                measured_translation_m=float(np.linalg.norm(poses[1,:3]-poses[0,:3])),
                measured_rotation_deg=float(np.rad2deg((rotations[0].inv()*rotations[1]).magnitude())),
                thumb_monitor_requests=x['finger_recovery_requests'][[a,b],finger].tolist(),
                thumb_posture_attempts=x['finger_posture_attempts_this_episode'][[a,b],finger].tolist(),
                thumb_search_elapsed_s=x['finger_search_elapsed_s'][[a,b],finger].tolist(),
                thumb_search_state=x['finger_search_state'][[a,b],finger].tolist()))
    preemptions = []
    for event in events:
        if event['kind'] == 'COORDINATED_RECOVERY_BEGIN':
            b = max(0,np.searchsorted(t,event['time_s'])-1)
            a = max(0,np.searchsorted(t,t[b]-.5))
            same = x['stroke_id'][a:b+1] == x['stroke_id'][b]; ix = np.flatnonzero(same)+a
            preemptions.append(dict(time_s=event['time_s'], finger=event['selected_finger'],
                prior_phase=str(x['command_phase'][b]),
                safe_actual_fingers=np.flatnonzero(x['active'][b] & (x['forces_n'][b]<8)).tolist(),
                prior_max_force_n=float(x['forces_n'][b].max()),
                same_stroke_saved_arc_gain_m=float(x['stroke_distance_m'][ix[-1]]-x['stroke_distance_m'][ix[0]]),
                lookback_s=float(t[ix[-1]]-t[ix[0]]), planned_recovery_duration_s=event['duration_s']))
    result = dict(scope='Read-only 100Hz trace and exact native event times; no physics or controller edits',
        source_trace_sha256=json.loads((directory/'summary.json').read_text())['trace_sha256'],
        windows=windows, thumb_plans=requests,
        thumb_execution_events=[item for item in events if item.get('selected_finger') == finger and item['kind'] in
            ('COORDINATED_RECOVERY_BEGIN','COORDINATED_RECOVERY_FIRST_CONTACT','COORDINATED_RECOVERY_END')],
        completed_lifts=transitions, all_recovery_preemptions=preemptions)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end',type=float,default=600.)
    args=parser.parse_args(); result=analyze(args.directory,args.output,args.end)
    print(json.dumps({key:result[key] for key in ('windows','completed_lifts','all_recovery_preemptions')},indent=2))
