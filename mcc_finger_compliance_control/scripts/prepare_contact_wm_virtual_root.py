"""Prepare a static inward command reserve, never a penetrating physical reset.

The point used is the closest true-pad collision witness, not a fixed tip site.
Only flexion coordinates move. Each depth is an independent paired experiment.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from contact_wm_initialization import hand_qpos_indices, set_object_static, set_palm_static
from split_fingertip_collision import GEOMS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(model, root, depth):
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in range(model.ngeom)]
    pads = [next(g for g, name in enumerate(names) if name.endswith(n+'_pad_collision')) for n in GEOMS]
    targets = [g for g, n in enumerate(names) if n.startswith('target/') and (model.geom_contype[g] or model.geom_conaffinity[g])]
    palm = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot/palm_lower')])
    target = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target/object_driver')])
    qids = hand_qpos_indices(model)
    joints = [int(np.flatnonzero(model.jnt_qposadr == q)[0]) for q in qids]
    dofs = model.jnt_dofadr[joints]
    limits = model.jnt_range[joints]
    q0 = np.array(root['q'], float)
    q = q0.copy()
    home = np.array(root['collection_home_object_pose'])
    set_palm_static(model, data, palm)
    set_object_static(model, data, target, home[:3], home[3:])
    data.qpos[qids] = q
    mujoco.mj_forward(model, data)
    witnesses = []
    for f in root['active_fingers']:
        choices = []
        for other in targets:
            line = np.zeros(6)
            gap = mujoco.mj_geomDistance(model, data, pads[f], other, .1, line)
            if 1e-8 < gap < .05:
                choices.append((gap, line.copy(), other))
        if not choices:
            raise ValueError('No separated pad witness for finger '+str(f))
        gap, line, other = min(choices, key=lambda x: x[0])
        inward = line[3:]-line[:3]
        inward /= np.linalg.norm(inward)
        local = data.geom_xmat[pads[f]].reshape(3,3).T @ (line[:3]-data.geom_xpos[pads[f]])
        witnesses.append(dict(finger=f, pad=pads[f], other=other, gap=gap,
                              local=local, initial=line[:3], inward=inward))
    for _ in range(20):
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        for w in witnesses:
            f, g = w['finger'], w['pad']
            point = data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3) @ w['local']
            error = depth+w['gap']-np.dot(point-w['initial'], w['inward'])
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, jr, point, int(model.geom_bodyid[g]))
            idx = np.array([4*f,4*f+2,4*f+3])
            jn = w['inward'] @ jp[:,dofs[idx]]
            q[idx] += np.clip(jn*error/max(float(jn@jn),1e-10), -.025, .025)
        q = np.clip(q, limits[:,0], limits[:,1])
    data.qpos[qids] = q
    mujoco.mj_forward(model, data)
    errors = []
    for w in witnesses:
        point = data.geom_xpos[w['pad']]+data.geom_xmat[w['pad']].reshape(3,3) @ w['local']
        errors.append(float(np.dot(point-w['initial'],w['inward'])-w['gap']-depth))
    # Target/object penetration is intentional only in the virtual command;
    # self contact is not. The physical state remains the separated root.
    self_contacts = [dict(geoms=[names[int(c.geom1)],names[int(c.geom2)]],distance=float(c.dist))
                     for c in data.contact[:data.ncon]
                     if c.dist < -1e-5 and all(names[int(g)].startswith('robot/') for g in (c.geom1,c.geom2))]
    report = dict(object=root['object_id'], physical_root_q=q0.tolist(), virtual_q=q.tolist(),
                  home_object_pose=home.tolist(), active_fingers=root['active_fingers'],
                  depth_m=depth, initial_gaps_m=[float(w['gap']) for w in witnesses],
                  normal_error_m=errors,max_q_delta_rad=float(abs(q-q0).max()),
                  max_side_delta_rad=float(abs((q-q0)[[1,5,9,13]]).max()),
                  virtual_self_contacts=self_contacts,actual_state_teleport=False)
    if max(map(abs,errors)) > 1e-5 or abs(q-q0).max() > .35 or self_contacts:
        raise ValueError(json.dumps(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depths-mm',type=float,nargs='+',default=[1.,3.,5.])
    a = p.parse_args()
    base = json.loads(a.protocol.read_text())
    roots = json.loads(Path(base['collector']['initial_roots']).read_text())['roots']
    matches = [r for r in roots if r['object_id']==base['object'] and r['active_fingers']==base['collector']['active_fingers']]
    if len(matches)!=1: raise ValueError('Pilot requires exactly one root')
    root = matches[0]
    if root['geometry_model_sha256']!=sha(a.model): raise ValueError('Model/root mismatch')
    model = mujoco.MjModel.from_binary_path(str(a.model))
    a.output.mkdir(parents=True,exist_ok=False)
    results = []
    for mm in a.depths_mm:
        if not 0 < mm <= 10: raise ValueError('Depth outside (0,10]mm')
        try:
            report = fit(model,root,mm*.001)
            report.update(model_sha256=sha(a.model),base_protocol_sha256=sha(a.protocol))
            ref = (a.output/f'virtual_{mm:g}mm.json').resolve()
            ref.write_text(json.dumps(report,indent=2))
            protocol = copy.deepcopy(base)
            tag = a.output.name+f'_{mm:g}mm'
            protocol['id'] = tag
            protocol['collector'].update(virtual_root_reference=str(ref),run_tag=tag)
            protocol['input_hashes'][str(ref)] = sha(ref)
            for name in ['collect_contact_wm_virtual_root.py','prepare_contact_wm_virtual_root.py']:
                protocol['source_hashes'][name] = sha(Path(__file__).parent/name)
            protocol['scope'] = 'Paired virtual preload test; original healthy physical root, MCC and object motion unchanged'
            path = (a.output/f'{tag}.protocol.json').resolve()
            path.write_text(json.dumps(protocol,indent=2))
            results.append(dict(depth_mm=mm,status='prepared_not_physically_verified',protocol=str(path),report=report))
        except ValueError as e:
            results.append(dict(depth_mm=mm,status='rejected',reason=str(e)))
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':main()
