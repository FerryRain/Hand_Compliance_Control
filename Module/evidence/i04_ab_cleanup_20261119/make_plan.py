from pathlib import Path
import ast,json,os,re,hashlib,subprocess
from functools import lru_cache
R=Path('D:/文档/Hand_Compliance_Control/i04_experiment_repo').resolve();W=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
inv=read(W/'inventory.json');keep=set(inv['source_closure'])
keep.add('Module/i04_geometric_traversal/retained_ab.py')
listed=subprocess.check_output(['rg','--files','--hidden','Module','scripts','tests','src','minimalist_compliance_control','mcc_finger_compliance_control','-g','*.py','-g','!generated/**','-g','!source_snapshot/**'],cwd=R).decode('utf-8').splitlines()
source_index={n.replace('\\','/') for n in listed}
print('INDEXED_SOURCES',len(source_index),flush=True)
geometric_names=[p.name for p in (R/'Module/i04_geometric_traversal').glob('*.py')]
# Keep source dependencies used by the independent DP/E05 projects as well.
@lru_cache(maxsize=None)
def deps(p):
 try:tree=ast.parse(p.read_text(encoding='utf-8-sig'))
 except (SyntaxError,UnicodeError):return set()
 pkg=list(p.relative_to(R).parts[:-1]);out=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):names=[x.name for x in node.names]
  elif isinstance(node,ast.ImportFrom):
   prefix=pkg[:len(pkg)-node.level+1] if node.level else []
   base='.'.join(prefix+([node.module] if node.module else []));names=[base]+[base+'.'+x.name for x in node.names]
  else:continue
  for n in names:
   for k in range(1,len(n.split('.'))+1):
    stem='/'.join(n.split('.')[:k])
    for f in [stem+'.py',stem+'/__init__.py']:
     if f in source_index:out.add(f)
 return out
shared=[]
for d in ['Module/dp_inverse_workflow','Module/coordination','Module/e05_physics','scripts','tests']:
 print('CHECKING_SHARED',d,flush=True)
 for name in sorted(n for n in source_index if n.startswith(d+'/') and '/source_snapshot/' not in n):
  p=R/name
  if 'i04' in p.name.lower():continue
  links={f for f in deps(p) if '/i04_' in f}
  if links:shared.append(dict(consumer=p.relative_to(R).as_posix(),sources=sorted(links)));keep.update(links)
  txt=p.read_text(encoding='utf-8-sig',errors='replace')
  if 'i04_geometric_traversal' in txt:
   for name in geometric_names:
    if name in txt:keep.add('Module/i04_geometric_traversal/'+name)
todo=list(keep)
while todo:
 for f in deps(R/todo.pop())-keep:keep.add(f);todo.append(f)
tests=[]
for p in (R/'Module/tests').glob('test_i04*.py'):
 ds=deps(p);is_i04={f for f in ds if '/i04_' in f}
 if is_i04 and is_i04<=keep:
  tests.append(p.relative_to(R).as_posix());keep.add(p.relative_to(R).as_posix())
support=set()
for e in read(R/'Module/evidence/whole_hand_release_assets_20261019.json')['assets']:
 support.add(Path(e['path']).as_posix())
for f in ['Module/generated/i04b_full_force_grace01_20261020/initial.npz','Module/generated/i04b_full_force_grace01_20261020/manifest.json','Module/generated/mustard_natural_fresh02_20261018/retained_ik_seeds.json','Module/generated/mustard_natural_fresh02_20261018/retained_surface_ik_seeds.npz','Module/generated/mustard_natural_fresh02_20261018/independent_native_audit/audit.json','Module/generated/whole_hand_geometry_touch02_20261018/geometry.npz']:
 support.add(f)
# DP audits also read run10's original model/provenance. Retain only its
# compact dependency/provenance files and shared meshes, not old media/results.
seed=R/'Module/generated/whole_hand_surface_run10_20261019'
for p in seed.rglob('*'):
 if p.is_file() and (p.suffix.lower() in {'.json','.py','.npz','.xml','.obj','.mtl'} and not p.name.startswith('trace_')):
  support.add(p.relative_to(R).as_posix())
protected_dirs={f'Module/generated/{n}' for n in inv['cases']}
protected_dirs.update('Module/generated/'+n for n in ['dp_inverse_workflow_20261019','e05_mcc_current','visual_demo','local_review'])
evidence_keep={'whole_hand_release_assets_20261019.json','2026-10-21_I04_FROZEN_B_AB.json','2026-10-21_I04_FROZEN_B_AB.md','2026-10-27_I04_PALM_COLLISION_FIX.json','2026-10-27_I04_PALM_COLLISION_FIX.md','2026-10-28_TRAJECTORY_RETENTION.json','2026-10-28_SPACE_CLEANUP.json','2026-10-19_I04_WHOLE_HAND_SURFACE.json','2026-10-20_I04_B_FULL.json'}
doc_keep={'I04_AB_WHOLE_HAND_PROTOCOL.md','I04_B_FULL_RUNBOOK.md','I04_FROZEN_B_AB_RUNBOOK.md','I04_PALM_COLLISION_FIX.md','I04_AB_RETAINED.md'}
deletes=[]
def prune(path,protect):
 rel=path.relative_to(R).as_posix()
 if rel in protect:return
 if any(x.startswith(rel+'/') for x in protect):
  if not path.is_dir():raise RuntimeError(rel)
  for c in path.iterdir():prune(c,protect)
 else:deletes.append(rel)
for p in (R/'Module/generated').iterdir():prune(p,support|protected_dirs)
for folder in ['Module/i04_geometric_traversal','Module/i04_gpis','Module/i04_er_gpis','Module/i04_oracle_next_point']:
 prune(R/folder,keep)
for p in (R/'Module/tests').glob('test_i04*'):
 if p.relative_to(R).as_posix() not in keep:deletes.append(p.relative_to(R).as_posix())
for p in (R/'Module/evidence').iterdir():
 if ('I04' in p.name or p.name in {'2026-10-27_ER_GPIS_PROBE_PROJECTION.json','demos'}) and p.name not in evidence_keep:deletes.append(p.relative_to(R).as_posix())
for p in (R/'Module').glob('I04*'):
 if p.is_file() and p.name not in doc_keep:deletes.append(p.relative_to(R).as_posix())
for p in (R/'scripts').glob('*'):
 if p.is_file() and ('i04' in p.name.lower() or p.name=='record_mustard_recovery_evidence.py') and p.name!='run_i04_ab.ps1':deletes.append(p.relative_to(R).as_posix())
deletes.extend(['.runtime_tmp','.runtime_tmpdebug'])
# Incomplete duplicate video exports inside selected B are not final results.
for n in ['incomplete_render_from_zero','incomplete_render_no_follow']:
 p=R/'Module/generated/i04b_palm_recovery02_20261027'/n
 if p.exists():deletes.append(p.relative_to(R).as_posix())
deletes=sorted(set(deletes))
for f in deletes:
 p=(R/f).resolve()
 assert p!=R and p.is_relative_to(R) and p.exists(),f
 assert not p.is_symlink(),f
 for k in keep|support:
  assert not (k==f or k.startswith(f+'/')),(f,k)
counts={};total=0;bytes_total=0;reparse=[]
print('SCANNING_DELETE_ROOTS',len(deletes),flush=True)
for index,f in enumerate(deletes):
 p=R/f;count=0;nbytes=0
 if p.is_file():count=1;nbytes=p.stat().st_size
 else:
  pending=[str(p)]
  while pending:
   with os.scandir(pending.pop()) as entries:
    for e in entries:
     s=e.stat(follow_symlinks=False)
     if s.st_file_attributes&0x400:reparse.append(e.path);continue
     if e.is_dir(follow_symlinks=False):pending.append(e.path)
     else:
      count+=1;nbytes+=s.st_size
      if count%250000==0:print('SCANNED',f,count,flush=True)
 counts[f]={'files':count,'logical_bytes':nbytes};total+=count;bytes_total+=nbytes
 if index%100==0:print('INVENTORIED',index,len(deletes),flush=True)
assert not reparse,reparse
plan=dict(root=str(R),delete_paths=deletes,delete_inventory=counts,delete_files=total,logical_bytes=bytes_total,reparse_points=reparse,source_keep=sorted(keep),support_keep=sorted(support),result_keep=sorted(inv['cases']),other_project_dirs=sorted(protected_dirs-{f'Module/generated/{n}' for n in inv['cases']}),shared_consumers=shared,tests=tests)
(W/'cleanup_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in plan.items() if k not in {'delete_paths','delete_inventory','support_keep','shared_consumers','source_keep','tests'}},ensure_ascii=False),flush=True)
print('SOURCE_KEEP',len(keep),'TESTS',len(tests),'SUPPORT_FILES',len(support),'DELETE_ROOTS',len(deletes),flush=True)
