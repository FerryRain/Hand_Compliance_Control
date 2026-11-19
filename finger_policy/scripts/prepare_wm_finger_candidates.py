"""Index audited, active per-finger imitation candidates; retain rejected WM data.

No whole-hand contact requirement, and no passive finger labels. This stage
indexes evidence only; raw hash and causal conversion are verified at export.
"""
import argparse,json
from pathlib import Path

def index(root):
 candidates={};audits=0
 for path in sorted(root.rglob('*audit*.json')):
  if path.stat().st_size>20_000_000:continue
  try:d=json.loads(path.read_text())
  except (ValueError,OSError):continue
  if not isinstance(d,dict):continue
  for report in d.get('reports',[]):
   if not isinstance(report,dict) or 'folder' not in report or 'rows' not in report:continue
   folder=Path(report['folder'])
   if not (folder/'rollout.h5').exists():continue
   audits+=1
   for row in report['rows']:
    if 'lane' not in row or 'finger' not in row:continue
    key=(str(folder.resolve()),int(row['lane']),int(row['finger']))
    reasons=[]
    if not row.get('teacher_contact_force_gate',False):reasons.append('original_per_finger_teacher_gate_failed')
    if not report.get('physics',{}).get('finite',False):reasons.append('finite_physics_not_verified')
    for field in ['between_substeps_qpos_jump','between_substeps_qvel_jump']:
     if report.get('physics',{}).get(field)!=0:reasons.append(field)
    if not row.get('actual_actuator_matches_post_mcc_1e6',False):reasons.append('actuator_command_alignment')
    if not (folder/'object_path_plan.npz').exists():reasons.append('missing_published_command_plan')
    item=dict(folder=key[0],lane=key[1],finger=key[2],source_audit=str(path.resolve()),rollout_sha256=report.get('rollout_sha256'),object=report.get('object'),split=report.get('split'),strict=row.get('strict'),found=row.get('found'),joint_range_deg=row.get('measured_q_range_deg'),candidate=not reasons,reasons=reasons,training_admissible=False)
    # Same evidence source can be repeated by a batch wrapper. Prefer a
    # qualifying record, but never reinterpret a failing row as a pass.
    if key not in candidates or (item['candidate'] and not candidates[key]['candidate']):candidates[key]=item
 return dict(audit_reports=audits,unique_finger_trajectories=len(candidates),candidates=sum(x['candidate'] for x in candidates.values()),rows=list(candidates.values()),remaining='Verify raw hash, actual active mask, measured joint margins, deduplicate physical trials, split by object/path; then causal45D export. No admission from contact alone.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=index(a.root);a.output.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='rows'})
