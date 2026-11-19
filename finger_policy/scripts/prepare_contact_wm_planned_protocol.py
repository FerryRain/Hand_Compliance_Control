"""Prepare physical collection only after full offline teacher certification."""
import sys,json,hashlib,copy,argparse
from pathlib import Path
import numpy as np,mujoco
R=Path('/home/rimlab/Code/Hand_Compliance_Control');S=R/'mcc_finger_compliance_control/scripts';sys.path.insert(0,str(S));O=R/'mcc_finger_compliance_control/data/contact_wm_review_20261110/physics_contract_v1';parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--pool',type=Path,required=True);parser.add_argument('--object',required=True);parser.add_argument('--seed',type=int,default=20261310);parser.add_argument('--model-dir',type=Path,default=O/'ycb_multi_prepare_v1/models');args=parser.parse_args();D=args.plan.resolve();pool=args.pool.resolve();
if (D/'collection_protocol.json').exists():raise FileExistsError('Preserve existing protocol')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
from collect_contact_wm_planned import prepare_virtual_reference
from contact_wm_initialization import hand_qpos_indices,set_palm_static,set_object_static
assert json.loads((D/'summary.json').read_text())['collection_admissible']
depths=[];checks=[]
for d in [.001,.003,.006]:
 try:
  _,report=prepare_virtual_reference(D/'reference.h5',[d]);depths.append(d);checks.append(dict(depth=d,passed=True,report=report))
 except ValueError as e:checks.append(dict(depth=d,passed=False,reason=str(e)))
(D/'virtual_depth_checks.json').write_text(json.dumps(checks,indent=2));assert depths
arrays,report=prepare_virtual_reference(D/'reference.h5',depths);np.savez_compressed(D/'virtual_reference.npz',**arrays)
obj=next(x for x in json.loads((pool/'object_manifest.json').read_text())['objects'] if x['object_id']==args.object)
old=json.loads((O/'planned_whole_interior_v1_roots.json').read_text())['roots'][0];roots=[];m=mujoco.MjModel.from_binary_path(str(args.model_dir/f"{obj['object_id']}.mjb"))
model_manifest=json.loads((args.model_dir/'manifest.json').read_text());entry=next(x for x in model_manifest['models'] if x['object_id']==obj['object_id']);assert entry['mesh_sha256']==obj['mesh_sha256'];assert entry['sha256']==h(args.model_dir/f"{obj['object_id']}.mjb")
palm=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/palm_lower')]);target=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'target/object_driver')])
for depth in depths:
 r=copy.deepcopy(old);r.update(object_id=obj['object_id'],mesh_sha256=obj['mesh_sha256'],q=arrays['healthy_q'][0].tolist(),seed=args.seed,virtual_depth_m=depth)
 pose=arrays['object_command_pose'][0].copy();pose[2]-=.1;r['object_pose']=pose.tolist()
 data=mujoco.MjData(m);set_palm_static(m,data,palm);data.qpos[hand_qpos_indices(m)]=r['q'];set_object_static(m,data,target,pose[:3],pose[3:]);mujoco.mj_forward(m,data)
 bad=[float(c.dist) for c in data.contact[:data.ncon] if c.dist < -1e-5]
 if bad:raise ValueError(('initial penetration',bad))
 r['initial_penetration_check']=bad;roots.append(r)
(D/'roots.json').write_text(json.dumps(dict(roots=roots),indent=2))
p=json.loads((O/'planned_whole_friction_v1/planned_whole_friction_v1_large_organic_000_mu0.6.json').read_text());p.update(id=D.name,object=obj['object_id'],objects=[obj['object_id']])
p['collector'].update(root=str(pool),initial_roots=str(D/'roots.json'),objects=[obj['object_id']],frames=len(arrays['healthy_q']),friction=1.,run_tag=D.name,virtual_reference=str(D/'virtual_reference.npz'),seed=args.seed,virtual_depths_m=depths)
# Only previously audited collector-hook migration is allowed.
latest=json.loads((O/'distance_band_v1/bumps.protocol.json').read_text())
for n,expected in p['source_hashes'].items():
 now=h(S/n)
 if now!=expected:
  reviewed=json.loads((O/'wm_ycb_motion_expansion_v2/ycb_021_bleach_cleanser.protocol.json').read_text())
  assert (n=='collect_contact_wm.py' and latest['source_hashes'][n]==now) or (n=='prepare_contact_wm_interior.py' and reviewed['source_hashes'][n]==now),(n,expected,now)
  p['source_hashes'][n]=now
p['reference_provenance']=report;p['input_hashes']={str(f):h(f) for f in [pool/'object_manifest.json',D/'roots.json',D/'virtual_reference.npz',D/'reference.h5']};p['comparison']='Full variable-length verified geometric reference; physics not yet admitted'
(D/'collection_protocol.json').write_text(json.dumps(p,indent=2));print('READY',depths,flush=True)
