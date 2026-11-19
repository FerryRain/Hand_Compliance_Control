from pathlib import Path
import ast,json,hashlib
R=Path('D:/文档/Hand_Compliance_Control/i04_experiment_repo')
W=Path(__file__).parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
cases=['i04a_frozen_b01_20261021','i04b_palm_recovery02_20261027']
roots=set()
caseinfo={}
for c in cases:
 p=R/'Module/generated'/c;m=read(p/'manifest.json');s=read(p/'summary.json')
 roots.update(m['source_hashes'])
 caseinfo[c]={'manifest_keys':list(m),'summary_keys':list(s),'files':[x.name for x in p.iterdir() if not x.name.startswith('trace_')],'sources':m['source_hashes'],'resume':{k:v for k,v in m.items() if any(t in k for t in ['parent','resume','prefix','continuation'])}}
roots.update('Module/i04_geometric_traversal/'+n+'.py' for n in ['a_frozen_b_mcc','palm_safe_recovery','palm_safe_audit','trajectory_replay','b_full_portable','b_full_video','b_full_review','ab_frozen_b_portable','retained_seed_inputs'])
def module_file(n):
 p=R/('/'.join(n.split('.'))+'.py')
 if p.is_file(): return p.relative_to(R).as_posix()
 p=R/('/'.join(n.split('.'))+'/__init__.py')
 return p.relative_to(R).as_posix() if p.is_file() else None
def deps(f):
 try:t=ast.parse((R/f).read_text(encoding='utf-8-sig'))
 except (SyntaxError,UnicodeError):return set()
 parts=f[:-3].split('/');pkg=parts[:-1]
 out=set()
 for n in ast.walk(t):
  if isinstance(n,ast.Import): mods=[a.name for a in n.names]
  elif isinstance(n,ast.ImportFrom):
   prefix=pkg[:len(pkg)-n.level+1] if n.level else []
   base='.'.join(prefix+([n.module] if n.module else []));mods=[base]+[base+'.'+a.name for a in n.names]
  else:continue
  for mod in mods:
   for i in range(1,len(mod.split('.'))+1):
    v=module_file('.'.join(mod.split('.')[:i]))
    if v:out.add(v)
 return out
seen=set();todo=list(roots)
while todo:
 f=todo.pop()
 if f in seen:continue
 if not (R/f).is_file():raise RuntimeError(f)
 seen.add(f);todo.extend(deps(f)-seen)
info={'cases':caseinfo,'source_closure':sorted(seen),'geometric_all':[x.name for x in (R/'Module/i04_geometric_traversal').glob('*.py')], 'generated_names':[x.name for x in (R/'Module/generated').iterdir()], 'runtime_nonfresh':[x.name for x in (R/'.runtime_tmp').iterdir() if not x.name.startswith('fresh')], 'evidence_names':[x.name for x in (R/'Module/evidence').iterdir()], 'scripts_names':[x.name for x in (R/'scripts').iterdir()]}
(W/'inventory.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'closure':sorted(seen),'case_meta':{c:{k:v for k,v in d.items() if k!='sources'} for c,d in caseinfo.items()},'other_runtime':info['runtime_nonfresh']},ensure_ascii=False,indent=2))
