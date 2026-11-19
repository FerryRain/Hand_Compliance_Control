from pathlib import Path
import hashlib,json,shutil,os
R=Path('D:/文档/Hand_Compliance_Control/i04_experiment_repo');W=Path(__file__).parent
E=R/'Module/evidence';D=E/'i04_ab_cleanup_20261119';D.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=read(W/'cleanup_plan.json');before=read(W/'verified_before.json');status=read(W/'cleanup_status.json')
assert status['status']=='DELETION_COMPLETE_PENDING_FINAL_VERIFICATION'
regenerated_caches=[n for n in plan['delete_paths'] if Path(n).name=='__pycache__' and (R/n).exists()]
remaining=[n for n in plan['delete_paths'] if (R/n).exists() and n not in regenerated_caches]
assert not remaining,remaining
missing=[n for n in plan['source_keep']+plan['support_keep'] if not (R/n).is_file()]
assert not missing,missing
hashes={};removed_selected=[]
for f,expected in before['file_sha256'].items():
 p=R/f
 if not p.exists():
  assert any(f==n or f.startswith(n+'/') for n in plan['delete_paths']),f
  removed_selected.append(f);continue
 actual=digest(p);assert actual==expected,f
 hashes[f]=actual
for f in plan['source_keep']+plan['support_keep']:
 p=R/f;hashes[f]=digest(p)
for folder in ['assets_external/ycb/collision/006_mustard_bottle/vhacd_256_scaled2p8_objstage','src/mjlab/asset_zoo/robots/fr3_leap_hand']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:hashes[p.relative_to(R).as_posix()]=digest(p)
for f in ['src/mjlab/asset_zoo/robots/leaphand_only.xml','scripts/run_i04_ab.ps1','Module/requirements-i04-whole-hand.txt','Module/I04_AB_RETAINED.md','Module/evidence/whole_hand_release_assets_20261019.json']:
 hashes[f]=digest(R/f)
cases={}
for name in plan['result_keep']:
 c=R/'Module/generated'/name;m=read(c/'manifest.json');s=read(c/'summary.json');a=read(c/'independent_native_audit/audit.json')
 bad=[f for f,h in m['source_hashes'].items() if digest(R/f)!=h or digest(c/'source_snapshot'/f)!=h]
 assert not bad,bad
 cases[name]=dict(path=c.relative_to(R).as_posix(),coverage_fraction=s['coverage_fraction'],complete=s['full_surface_complete'],native_states=s['native_states'],native_trace_chunks=len(s['trace_chunks']),native_seconds=s['executed_s'],audit_status=a['status'],current_runtime_matches_all_archived_sources=True,source_files=len(m['source_hashes']),model_sha256=digest(c/'model.xml'),proof_sha256=digest(c/'independent_native_audit/audit.json'))
 print('FINAL_DATA_VERIFIED',name,flush=True)
checks=read(W/'after_native_checks.json');assert len(checks)==6 and all(x['returncode']==0 for x in checks)
regression=read(W/'regression.json');assert regression['returncode']==0
for p in W.glob('*.log'):shutil.copy2(p,D/p.name)
for n in ['cleanup_plan.json','deletion_log.jsonl','initial_git_status.txt','all_a_candidates.json','before_native_checks.json','after_native_checks.json','regression.json']:
 shutil.copy2(W/n,D/n)
record=dict(date='2026-11-19',timezone='Asia/Shanghai',status='CLEANUP_COMPLETE_RETAINED_AB_VERIFIED',scope='I04 artifacts in this repository only; DP/E05/shared assets and git history preserved',cases=cases,
 original_traces_unchanged=True,full_new_experiment_rerun=False,old_A_100_percent_native_trajectories_were_already_deleted_on='2026-10-28',
 A_selection='Best remaining complete native palmar-contact recording. A is incomplete, not a full traversal success.',
 B_coverage_definition='Original 25mm mesh-geodesic neighborhood coverage; not exhaustive direct tactile area or unknown-object GPIS completion.',
 comparison_limit='A retains disabled rigid-palm collision model; B enables palm collisions. Not a same-physics paired AB comparison.',
 B_versioned_continuation_s=5774.600000258942,
 deleted_logical_files=plan['delete_files'],deleted_logical_bytes_with_hardlink_duplicates=plan['logical_bytes'],deleted_roots=len(plan['delete_paths']),
 free_before_bytes=status['free_before'],free_after_bytes=status['free_after'],observed_volume_space_released_bytes=status['free_after']-status['free_before'],
 removed_incomplete_selected_B_exports=len(removed_selected),retained_source_and_test_files=len(plan['source_keep']),retained_dependency_files=len(plan['support_keep']),
 native_checks=checks,regression=regression,detail_directory=D.relative_to(R).as_posix(),
 source_restore={'file':'Module/i04_geometric_traversal/palm_safe_mcc.py','previous_sha256':digest(W/'palm_safe_mcc_before.py'),'restored_successful_sha256':digest(R/'Module/i04_geometric_traversal/palm_safe_mcc.py')},
 new_entry='scripts/run_i04_ab.ps1',remaining_deletion_targets=remaining,missing_dependencies=missing,
 regenerated_bytecode_caches_from_verification=regenerated_caches)
(E/'2026-11-19_I04_AB_CLEANUP.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
catalog=dict(status='RETAINED_FILES_VERIFIED',scope='Retained A/B evidence plus current runtime and required inputs. Original evidence files unchanged; old failed runs are not included.',sha256=hashes)
(E/'2026-11-19_I04_AB_RETAINED_FILES.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ['native_checks','regression']},ensure_ascii=False,indent=2),flush=True)
