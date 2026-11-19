import os,sys,json,subprocess,time
from pathlib import Path
R=Path('D:/文档/Hand_Compliance_Control/i04_experiment_repo');W=Path(__file__).parent
env=os.environ.copy();env.update(PYTHONPATH=str(R/'.runtime')+';'+str(R),PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
py='D:/Environment/Anaconda/envs/handcomp/python.exe'
phase=sys.argv[1];results=[]
commands=[]
for method in ['A','B']:
 commands.append((method+'_startup',[py,'-u','-m','Module.i04_geometric_traversal.retained_ab','run','--method',method,'--seconds','0.02','--output',str(W/(phase+'_'+method+'_startup'))]))
 commands.append((method+'_replay',[py,str(R/'Module/i04_geometric_traversal/trajectory_replay.py'),'--case',method,'--check']))
 commands.append((method+'_native',[py,'-u','-m','Module.i04_geometric_traversal.retained_ab','smoke','--method',method,'--states','500']))
for name,cmd in commands:
 t=time.perf_counter();log=W/(phase+'_'+name+'.log')
 with log.open('w',encoding='utf-8') as f:p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,returncode=p.returncode,wall_seconds=time.perf_counter()-t,log=str(log))
 results.append(row);(W/(phase+'_native_checks.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
 print(json.dumps(row),flush=True)
 if p.returncode:print(log.read_text(encoding='utf-8')[-6000:],flush=True);raise SystemExit(1)
