"""Object-disjoint expansion without changing the frozen model or loss.

Only simultaneous four-active-finger records may train flat. Independent
finger histories are combined in batches ONLY for the independent model.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from controlled_policy_ablation import CommonWindows, sample


class ExpandedWindows:
    def __init__(self, protocol, group, device='cpu'):
        self.group=group;self.device=torch.device(device)
        self.old=CommonWindows(protocol['data_directory'],device)
        self.normalization=self.old.normalization;self.norm=self.old.norm;self.limits=self.old.limits
        manifest=json.loads(Path(protocol['expansion_manifest']).read_text())
        self.records=[r for r in manifest['records'] if r['assigned_split'] in ['train','val']]
        self.by_finger=defaultdict(lambda:defaultdict(list));self.whole=defaultdict(list)
        arrays=[];joints=[];self.entries={};cursor=0
        for r in self.records:
            with np.load(r['shard']) as z:
                x=z['observation'].copy();q=z['q'].copy();roots=z['roots'].copy()
            assert np.isfinite(x).all() and np.isfinite(q).all() and np.array_equal(x[:,:4],q)
            assert np.array_equal(roots,np.arange(15,len(q)-16))
            f=int(r['finger']);key=r['record_key']
            entry=dict(start=cursor,roots=roots,record=r)
            self.entries[key]=entry
            if r['assigned_split']=='train':self.by_finger[r['object_parent']][f].append(entry)
            arrays.append(x);joints.append(q);cursor+=len(q)
        self.obs=torch.as_tensor(np.concatenate(arrays),device=device)
        self.q=torch.as_tensor(np.concatenate(joints),device=device)
        self.validation_whole=[]
        for trial in manifest['whole_trials']:
            if trial['assigned_split'] not in ['train','val']:continue
            es=[self.entries[k] for k in trial['record_keys']]
            assert all(np.array_equal(e['roots'],es[0]['roots']) for e in es)
            if trial['assigned_split']=='train':self.whole[trial['object']].append(es)
            else:self.validation_whole.append(es)
        self.objects=sorted(self.whole if group!='single_all' else
                            {o:fs for o,fs in self.by_finger.items() if set(fs)=={0,1,2,3}})
        assert self.objects and self.validation_whole
        self.draw_counts=defaultdict(lambda:np.zeros(4,np.int64))

    def choose(self,count,rng):
        assert count%2==0
        old=self.old.choose(count//2,rng)
        selected=[]
        for _ in range(count//2):
            obj=self.objects[int(rng.integers(len(self.objects)))];self.draw_counts[obj]+=1
            if self.group=='single_all':
                indices=[]
                for f in range(4):
                    es=self.by_finger[obj][f];e=es[int(rng.integers(len(es)))];indices.append(e['start']+int(rng.choice(e['roots'])))
            else:
                es=self.whole[obj][int(rng.integers(len(self.whole[obj])))];root=int(rng.choice(es[0]['roots']))
                indices=[e['start']+root for e in es]
            selected.append(indices)
        return old,np.asarray(selected,dtype=np.int64)

    def raw_new(self,indices):
        roots=torch.as_tensor(indices,device=self.device)
        raw=self.obs[roots[:,:,None]+torch.arange(-15,1,device=self.device)].permute(0,2,1,3).clone()
        delta=(self.q[roots[:,:,None]+torch.arange(1,17,device=self.device)]-self.q[roots][:,:,None]).permute(0,2,1,3)
        return raw,delta

    def batch(self,selection,augmentation_rng=None):
        old,new=selection
        x,y,d,v=self.old.batch(old)
        raw,delta=self.raw_new(new)
        # Convert old normalized observations back only when augmentation is
        # requested, so the same causal contact-burst implementation serves all.
        joined=torch.cat([x*self.norm['std']+self.norm['mean'],raw])
        delta=torch.cat([d.reshape(-1,16,4,4),delta])
        if augmentation_rng is not None:
            for i,f in np.argwhere(augmentation_rng.random((len(joined),4))<.1):
                n=int(augmentation_rng.integers(1,4));stop=int(augmentation_rng.integers(n+1,17));start=stop-n
                joined[i,start:stop,f,4:10]=joined[i,start-1,f,4:10]
                joined[i,start:stop,f,10]=0;joined[i,start:min(stop+1,16),f,15:21]=0
        return (joined-self.norm['mean'])/self.norm['std'],(delta/self.norm['scale']).reshape(-1,16,16)

    def validation_batches(self):
        selected=[];ids=[]
        for i,es in enumerate(self.validation_whole):
            roots=es[0]['roots'];chosen=roots[np.linspace(0,len(roots)-1,min(16,len(roots))).astype(int)]
            selected.extend([[e['start']+int(r) for e in es] for r in chosen]);ids.extend([i]*len(chosen))
        return np.asarray(selected),np.asarray(ids)


@torch.no_grad()
def validate_expansion(models,arm,data):
    from controlled_policy_ablation import validate
    old=validate(models,arm,data.old)
    models.eval();roots,ids=data.validation_batches();errors=[]
    g=torch.Generator(device=data.device).manual_seed(77119)
    for start in range(0,len(roots),64):
        raw,truth=data.raw_new(roots[start:start+64])
        x=(raw-data.norm['mean'])/data.norm['std']
        pred=sample(models,arm,x,g)*data.norm['scale'].reshape(16,16)
        errors.append(abs(pred-truth.reshape(-1,16,16)).cpu().numpy())
    err=np.concatenate(errors)
    near=float(np.mean([err[ids==i,:4].mean() for i in np.unique(ids)]))
    models.train()
    return dict(old=old,new_whole_near4_mae_rad=near,new_per_finger_near4_rad=[float(err[:,:4,4*f:4*f+4].mean()) for f in range(4)],
        selection_score=.5*(old['near4_mae_rad']+near),new_validation_physical_trials=len(np.unique(ids)))
