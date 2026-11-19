from pathlib import Path
import json,hashlib,os,time
R=Path('D:/文档/Hand_Compliance_Control/i04_experiment_repo');W=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for base in (R/'Module/generated').iterdir():
 if not base.is_dir() or not any(t in base.name.lower() for t in ['i04','mustard','whole_hand']):continue
 if 'blind_fresh' in base.name or 'gpis' in base.name:continue
 for path,dirs,files in os.walk(base):
  depth=len(Path(path).relative_to(base).parts)
  dirs[:]=[d for d in dirs if depth<3 and not any(t in d for t in ['source','audit','collision','video','render','snapshot','provenance'])]
  if 'summary.json' not in files:continue
  p=Path(path);s=read(p/'summary.json');m=read(p/'manifest.json') if (p/'manifest.json').exists() else {}
  if not isinstance(s,dict) or not isinstance(m,dict):continue
  method=s.get('method',s.get('cell',m.get('method',m.get('cell',''))))
  if 'A-' not in str(method) and 'a_' not in base.name and 'A-' not in p.name:continue
  chunks=s.get('trace_chunks',[])
  rows.append(dict(path=p.relative_to(R).as_posix(),method=method,coverage=s.get('coverage_fraction'),complete=s.get('full_surface_complete'),seconds=s.get('executed_s'),native_states=s.get('native_states'),expected_chunks=len(chunks),present_chunks=sum((p/c['file']).is_file() for c in chunks),native_trace=(p/'trace.npz').is_file()))
(W/'all_a_candidates.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('A_CANDIDATES',json.dumps(rows),flush=True)
inventory=read(W/'inventory.json');checks=[];hashes={};failures=[]
for name in inventory['cases']:
 case=R/'Module/generated'/name;s=read(case/'summary.json');m=read(case/'manifest.json')
 for c in s['trace_chunks']:
  p=case/c['file'];h=digest(p);hashes[p.relative_to(R).as_posix()]=h
  if h!=c['sha256']:failures.append(str(p))
 for f,h in m['source_hashes'].items():
  p=case/'source_snapshot'/f
  if digest(p)!=h:failures.append(str(p))
 current_mismatch=[f for f,h in m['source_hashes'].items() if digest(R/f)!=h]
 for p in case.rglob('*'):
  if p.is_file() and p.relative_to(R).as_posix() not in hashes:hashes[p.relative_to(R).as_posix()]=digest(p)
 checks.append(dict(case=name,chunks=len(s['trace_chunks']),audit_status=read(case/'independent_native_audit/audit.json').get('status'),current_source_differences=current_mismatch))
 print('CASE_VERIFIED',json.dumps(checks[-1]),flush=True)
assets=read(R/'Module/evidence/whole_hand_release_assets_20261019.json')['assets']
for e in assets:
 p=R/e['path'];h=digest(p);hashes[e['path']]=h
 if h!=e['sha256']:failures.append(str(p))
baseline=read(R/'Module/generated/i04b_full_force_grace01_20261020/manifest.json')
bad=[f for f,h in baseline['source_hashes'].items() if digest(R/f)!=h]
if bad:failures.extend(bad)
result=dict(cases=checks,shared_assets=len(assets),frozen_A_baseline_source_mismatches=bad,failures=failures,file_sha256=hashes)
(W/'verified_before.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('VERIFIED',len(hashes),'FILES;FAILURES',len(failures),flush=True)
assert not failures
