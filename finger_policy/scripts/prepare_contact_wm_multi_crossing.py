"""Prepare bounded cross-region programs from paired, previously loaded roots.

Initial pilot: tomato can with axial travel, own-axis roll and slow tilt.
MCC, initial joint state, friction and force targets are inherited unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from contact_wm_initialization import hand_qpos_indices, set_object_static, set_palm_static
from split_fingertip_collision import GEOMS
from contact_wm_clearance import FixedTipClearance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpolate(values, frames):
    x = np.linspace(0., len(values)-1, frames)
    i = np.minimum(x.astype(int), len(values)-2)
    t = x-i
    s = t**3*(10.-15.*t+6.*t*t)
    v = np.asarray(values)
    return v[i]+s*(v[i+1]-v[i])


def program(home, sign, frames):
    # One changing sequence, not repetitions of the same local sweep.
    axial = sign*interpolate([0., .035, -.025, .045, -.035], frames)
    roll = sign*np.deg2rad(interpolate([0., 45., 110., 170., 230.], frames))
    tilt = np.deg2rad(interpolate([0., 10., -8., 12., 0.], frames))
    r0 = Rotation.from_quat(home[[4,5,6,3]])
    r = r0*Rotation.from_rotvec(np.c_[roll, roll*0, roll*0])*Rotation.from_rotvec(np.c_[tilt*0, tilt, tilt*0])
    path = np.tile(home, (frames,1))
    path[:, :3] += axial[:,None]*r0.apply([1.,0.,0.])[None]
    path[:, 3:] = r.as_quat()[:, [3,0,1,2]]
    path[0] = home
    return path


def geometry_check(root, path):
    model = mujoco.MjModel.from_binary_path(root['geometry_model_path'])
    if sha(root['geometry_model_path']) != root['geometry_model_sha256']:
        raise ValueError('Geometry model changed')
    d = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
    pads = [next(i for i,n in enumerate(names) if n.endswith(g+'_pad_collision')) for g in GEOMS]
    active = [pads[f] for f in root['active_fingers']]
    nonpad = [i for i,n in enumerate(names) if n.startswith('robot/') and i not in active and (model.geom_contype[i] or model.geom_conaffinity[i])]
    targets = [i for i,n in enumerate(names) if n.startswith('target/') and (model.geom_contype[i] or model.geom_conaffinity[i])]
    palm = int(model.body_mocapid[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/palm_lower')])
    obj = int(model.body_mocapid[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'target/object_driver')])
    set_palm_static(model,d,palm)
    d.qpos[hand_qpos_indices(model)] = root['q']
    queries = {g: FixedTipClearance(model,d,g,targets) for g in nonpad}
    minimum = .05
    worst = None
    for frame in sorted(set(range(0,len(path),50)) | {len(path)-1}):
        pose = path[frame]
        set_object_static(model,d,obj,pose[:3],pose[3:])
        mujoco.mj_kinematics(model,d)
        for hand in nonpad:
            distance = queries[hand]()
            if distance < minimum:
                minimum = distance
                worst = dict(frame=frame,geom=names[hand],distance_m=distance)
    return dict(min_nonpad_gap_m=minimum,worst=worst,sampling_frames=50,
        fixed_initial_q_only=True,full_physics_certified=False,passed=minimum>=.002,
        native_distance_false_zero_corrections=sum(q.corrected_queries for q in queries.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--groups',nargs='+',default=['03','13','23','023','123'])
    a = ap.parse_args()
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[1]
    source = repo/'mcc_finger_compliance_control/data/contact_wm_review_20261110/physics_contract_v1/balanced_axis_collection_20261113'
    a.output = a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    jobs,checks = [],[]
    for group in a.groups:
        base = json.loads((source/f'axis_tomato_active{group}/axis_tomato_active{group}_5mm.protocol.json').read_text())
        for name,h in base['source_hashes'].items():
            if sha(scripts/name)!=h: raise ValueError('Frozen source changed: '+name)
        for name,h in base['input_hashes'].items():
            if sha(name)!=h: raise ValueError('Frozen input changed: '+name)
        roots = json.loads(Path(base['collector']['initial_roots']).read_text())['roots']
        roots = [r for r in roots if r['object_id']==base['object'] and r['active_fingers']==base['collector']['active_fingers']]
        if len(roots)!=1: raise ValueError('Exactly one matched root required')
        root = roots[0]
        for sign in [1,-1]:
            tag = f'multi_cross_{group}_{"pos" if sign==1 else "neg"}'
            path = program(np.asarray(root['collection_home_object_pose']),sign,4500)
            report = geometry_check(root,path)
            r = Rotation.from_quat(path[:,[4,5,6,3]])
            report.update(id=tag,max_speed_m_s=float(np.linalg.norm(np.diff(path[:,:3],axis=0),axis=-1).max()/.01),
                max_angular_speed_rad_s=float((r[:-1].inv()*r[1:]).magnitude().max()/.01),
                translation_span_m=np.ptp(path[:,:3],axis=0).tolist(),
                description='80mm axial span, 230deg own-axis rotation, -8..12deg tilt; one shared object; unchanged fixed virtual grasp')
            report['passed'] &= report['max_speed_m_s']<.05 and report['max_angular_speed_rad_s']<.4
            checks.append(report)
            print(tag,report,flush=True)
            if not report['passed']: continue
            ref = a.output/(tag+'.npz')
            np.savez_compressed(ref,commanded=path[:,None,:])
            ref.with_suffix('.json').write_text(json.dumps(report,indent=2))
            p = copy.deepcopy(base)
            p.update(id=tag,scope='Shared-object multi-finger cross-region pilot; not approved teacher data',expected_lanes=1,expected_transitions_20hz=900)
            p['collector'].update(frames=4500,behaviors=['slide'],randomize_motion=False,motion_rate=1.,
                motion_rate_range=None,run_tag=tag,object_waypoints=str(ref),object_waypoints_sha256=sha(ref))
            for name in ['collect_contact_wm_multi_waypoints.py','prepare_contact_wm_multi_crossing.py']:
                p['source_hashes'][name]=sha(scripts/name)
            p['input_hashes'][str(ref)]=sha(ref)
            p['input_hashes'][str(ref.with_suffix('.json'))]=sha(ref.with_suffix('.json'))
            p['motion']=report['description']
            protocol=a.output/(tag+'.protocol.json')
            protocol.write_text(json.dumps(p,indent=2))
            folder=Path(p['collector']['root'])/'episodes/multi_finger'/tag/(p['object']+'_mu1_k1_active'+group)
            jobs.append(dict(id=tag,protocol=str(protocol),folder=str(folder),expected_frames=4500,collector='collect_contact_wm_multi_waypoints.py'))
    (a.output/'preflight.json').write_text(json.dumps(checks,indent=2))
    (a.output/'queue.json').write_text(json.dumps(dict(jobs=jobs),indent=2))


if __name__=='__main__': main()
