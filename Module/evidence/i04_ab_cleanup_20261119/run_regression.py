from pathlib import Path
import json,subprocess,os,time
W=Path(__file__).parent;R=Path('D:/文档/Hand_Compliance_Control/i04_experiment_repo')
p=json.loads((W/'cleanup_plan.json').read_text(encoding='utf-8'))
names=[x[:-3].replace('/','.') for x in p['tests']]
env=os.environ.copy();env.update(PYTHONPATH=str(R/'.runtime')+';'+str(R),PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['D:/Environment/Anaconda/envs/handcomp/python.exe','-m','unittest','-v',*names]
t=time.perf_counter()
with (W/'retained_regression.log').open('w',encoding='utf-8') as log:
 r=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
result=dict(command=cmd,returncode=r.returncode,wall_seconds=time.perf_counter()-t,test_modules=len(names))
(W/'regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
print((W/'retained_regression.log').read_text(encoding='utf-8')[-4000:],flush=True)
raise SystemExit(r.returncode)
