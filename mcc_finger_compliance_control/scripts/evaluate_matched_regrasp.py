"""Controlled reset toggle, identical startup/history code, progress-keyed noise.

No edits to the frozen controller or DP implementation. Runtime monkeypatches
exist only in this process. Measured observations may diverge after a reset.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch


def noise_key(cursor, phase, age, length):
    # Normal path queries share a noise bank independent of reset wall time.
    # Recovery queries have a disjoint bank and do not consume path noise.
    return (cursor//10)*10 if phase==0 else ((length+100)//10)*10 + phase*1000000 + age


def main(request):
    task=json.loads(Path(request).read_text())
    enabled=task['midpath_regrasp_enabled']
    if type(enabled) is not bool or len(task['rows'])!=1:
        raise ValueError('Matched protocol requires Boolean toggle and batch=1')
    import smooth_regrasp_reference as outer
    factory=outer.regrasp_class
    def controlled_class():
        original=factory()
        class Controlled(original):
            def observe(self,q,pose,tactile):
                saved=self.cooldown.clone()
                # Gate only natural mid-path trigger eligibility. Do not skip
                # history, caching, startup, load authority or nominal shaping.
                if not enabled:
                    mask=self.initial_ready & (self.phase==0)
                    self.cooldown=torch.where(mask,self.cooldown.clamp_min(1),self.cooldown)
                super().observe(q,pose,tactile)
                if not enabled:self.cooldown=saved
            def report(self):
                return dict(super().report(),midpath_regrasp_enabled=enabled)
        return Controlled
    outer.regrasp_class=controlled_class
    from single_finger_dp import SingleFingerRuntime
    from batched_b_control_chain import BatchedBControlChain
    step= BatchedBControlChain.step
    call=SingleFingerRuntime.__call__
    records=[]
    def hooked_step(self,**kwargs):
        self.single_finger_runtime.matched_supervisor=self.command_supervisor
        return step(self,**kwargs)
    def digest(x):
        return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def sample(self,frame,q,history):
        s=self.matched_supervisor
        cursor,phase,age=int(s.cursor[0]),int(s.phase[0]),int(s.age[0])
        key=noise_key(cursor,phase,age,int(s.lengths[0]))
        row=dict(wall_frame=int(frame),cursor=cursor,phase=phase,age=age,noise_key=key,
                 q_sha256=digest(q),history_sha256=digest(history),families={})
        for family,(model,_,_) in self.models.items():
            original=model.diffusion.conditional_sample
            def recorded(*args,_family=family,_original=original,**kwargs):
                generator=kwargs['generator']
                before=digest(generator.get_state())
                output=_original(*args,**kwargs)
                row['families'][_family]=dict(rng_before=before,rng_after=digest(generator.get_state()))
                return output
            model.diffusion.conditional_sample=recorded
        try:
            result=call(self,key,q,history)
        finally:
            # Each wrapper holds the original callable in a keyword default.
            for model,_,_ in self.models.values():
                model.diffusion.conditional_sample=model.diffusion.conditional_sample.__kwdefaults__['_original']
        row['chunk_sha256']=digest(result);records.append(row)
        return result
    BatchedBControlChain.step=hooked_step
    SingleFingerRuntime.__call__=sample
    import evaluate_smooth_regrasp
    try:evaluate_smooth_regrasp.main(request)
    finally:
        Path(task['output']).with_suffix('.sampling.json').write_text(json.dumps(records,indent=2))
        BatchedBControlChain.step=step
        SingleFingerRuntime.__call__=call
        outer.regrasp_class=factory


if __name__=='__main__':main(sys.argv[1])
