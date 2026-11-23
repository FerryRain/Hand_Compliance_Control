"""Shared data, noise, loss and decoding contract for flat vs independent DP.

Every sampled full-hand window supplies exactly the same 16 joint labels to
both arms. Single = shared long-finger U-Net + separate thumb U-Net; flat has
all four local feature blocks and one copy of the common palm command.
"""
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from single_finger_dp import configuration, project_history, split_check, sha, save
from train_dp import build_policy


def configs(arm):
    if arm=='single':return {k:configuration() for k in ['long','thumb']}
    c=configuration()
    c.update(robot_state_dim=132,environment_state_dim=12,action_dim=16,down_dims=[64,128,288])
    return {'flat':c}


def build(arm,device):
    return torch.nn.ModuleDict({k:build_policy(SimpleNamespace(**v),torch.device(device))
                               for k,v in configs(arm).items()})


def conditions(x,arm):
    # x[B,T,F,45]; the shared palm features agree across all fingers.
    if arm=='flat':
        return {'flat':{'observation.state':x[...,:33].reshape(len(x),16,132),
                        'observation.environment_state':x[:,:,0,33:]}}
    return {k:{'observation.state':v[...,:33],'observation.environment_state':v[...,33:]}
            for k,v in {'long':x[:,:,:3].permute(0,2,1,3).reshape(-1,16,45),
                        'thumb':x[:,:,3]}.items()}


def predict_noise(models,arm,noisy,t,cond,*,thumb_stream=None):
    if t.ndim==0:t=t.expand(len(noisy))
    if arm=='flat':return models['flat'].diffusion.unet(noisy,t,global_cond=cond['flat'])
    b=len(noisy)
    local=noisy.reshape(b,16,4,4)
    long=local[:,:,:3].permute(0,2,1,3).reshape(-1,16,4)
    lt=t.repeat_interleave(3) if t.ndim else t
    if thumb_stream is not None:
        caller = torch.cuda.current_stream(noisy.device)
        thumb_stream.wait_stream(caller)

    lp=models['long'].diffusion.unet(long,lt,global_cond=cond['long']).reshape(b,3,16,4).permute(0,2,1,3)
    if thumb_stream is None:
        tp=models['thumb'].diffusion.unet(local[:,:,3],t,global_cond=cond['thumb'])
    else:
        try:
            with torch.cuda.stream(thumb_stream):
                tp = models['thumb'].diffusion.unet(local[:,:,3], t, global_cond=cond['thumb'])
        finally:
            # The scheduler consumes both predictions before starting the next step.
            caller.wait_stream(thumb_stream)
        tp.record_stream(caller)
    return torch.cat([lp,tp[:,:,None]],2).reshape(b,16,16)


def conditioning(models,arm,x):
    return {k:models[k].diffusion._prepare_global_conditioning(b) for k,b in conditions(x,arm).items()}


class SingleDenoiseGraph:
    """Capture the two-stream network forward for the deployed one-hand shape."""

    @torch.no_grad()
    def __init__(self, models, device, thumb_stream):
        dtype = next(models.parameters()).dtype
        self.noisy = torch.zeros((1, 16, 16), device=device, dtype=dtype)
        self.timestep = torch.zeros((), device=device, dtype=torch.long)
        self.conditions = conditioning(models, 'single', self.noisy.new_zeros((1, 16, 4, 45)))
        self.caller = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(self.caller)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                predict_noise(models, 'single', self.noisy, self.timestep, self.conditions,
                              thumb_stream=thumb_stream)
        self.caller.wait_stream(capture_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.prediction = predict_noise(
                models, 'single', self.noisy, self.timestep, self.conditions, thumb_stream=thumb_stream)

    def supports(self, observation):
        return (observation.shape == (1, 16, 4, 45)
                and observation.device == self.noisy.device
                and observation.dtype == self.noisy.dtype
                and torch.cuda.current_stream(observation.device) == self.caller)

    def update_conditions(self, conditions):
        for branch, target in self.conditions.items():
            target.copy_(conditions[branch])

    def __call__(self, noisy, timestep):
        self.noisy.copy_(noisy)
        self.timestep.copy_(timestep)
        self.graph.replay()
        return self.prediction


def shared_loss(models,arm,x,action,eps,t):
    scheduler=next(iter(models.values())).diffusion.noise_scheduler
    noisy=scheduler.add_noise(action,eps,t)
    pred=predict_noise(models,arm,noisy,t,conditioning(models,arm,x))
    # This equals .75 * long MSE + .25 * thumb MSE, with one global
    # gradient clip and optimizer step for the entire single-finger system.
    return (pred-eps).square().mean()


@torch.no_grad()
def sample(models,arm,x,generator,*,thumb_stream=None,denoise_graph=None,denoise_steps=10):
    scheduler=next(iter(models.values())).diffusion.noise_scheduler
    scheduler.set_timesteps(denoise_steps,device=x.device)
    cond=conditioning(models,arm,x)
    if denoise_graph is not None:
        denoise_graph.update_conditions(cond)
    value=torch.randn((len(x),16,16),device=x.device,generator=generator)
    for t in scheduler.timesteps:
        if denoise_graph is None:
            prediction=predict_noise(models,arm,value,t,cond,thumb_stream=thumb_stream)
        else:
            prediction=denoise_graph(value,t)
        value=scheduler.step(prediction,t,value,generator=generator).prev_sample
    return value


class CommonWindows:
    def __init__(self,directory,device='cpu',normalization=None):
        directory=Path(directory);self.device=torch.device(device)
        rows=json.loads((directory/'episode_split.json').read_text())['episodes'];split_check(rows)
        with np.load(directory/'single_finger_frames.npz') as z:
            self.roots=z['roots'].copy();self.limits=z['joint_limits'].copy()
            self.frame_ids=z['episode_id'].copy();raw=z['observation'].copy();q=z['q'].copy()
        assert raw.shape==(len(q),4,45) and q.shape[1:]==(4,4)
        assert np.isfinite(raw).all() and np.isfinite(q).all()
        assert np.array_equal(raw[:,:,:4],q)
        assert np.array_equal(raw[:,:,33:],np.broadcast_to(raw[:,0:1,33:],raw[:,:,33:].shape))
        self.train_roots=self.roots[self.roots[:,3]==0]
        self.val_roots=self.roots[self.roots[:,3]==1]
        self.by_episode=[r[:,1] for eid in np.unique(self.train_roots[:,0])
                         if len(r:=self.train_roots[self.train_roots[:,0]==eid])]
        for offset in [-15,16]:
            assert np.array_equal(self.frame_ids[self.roots[:,1]+offset],self.roots[:,0])
        if normalization is None:
            train=np.isin(self.frame_ids,np.unique(self.train_roots[:,0]))
            mean=np.empty((4,45),np.float32);std=mean.copy()
            for fingers in [(0,1,2),(3,)]:
                x=raw[train][:,fingers].astype(np.float64).reshape(-1,45)
                m=x.mean(0);s=np.maximum(x.std(0),1e-5)
                m[10]=s[10]=.5;m[29:33]=0;s[29:33]=1
                for f in fingers:mean[f]=m;std[f]=s
            # One physical common-palm normalization for both architectures.
            mean[:,33:]=mean[0,33:];std[:,33:]=std[0,33:]
            scale=np.full((16,4,4),1e-4,np.float32)
            for roots in np.array_split(self.train_roots[:,1],max(1,len(self.train_roots)//2048)):
                d=abs(q[roots[:,None]+np.arange(1,17)]-q[roots,None])
                scale=np.maximum(scale,d.max(0))
            scale[:,:3]=scale[:,:3].max(1,keepdims=True)
            normalization=dict(mean=mean,std=std,scale=scale)
        self.normalization=normalization
        self.norm={k:torch.as_tensor(v,device=device) for k,v in normalization.items()}
        self.obs=torch.as_tensor(raw,device=device);self.q=torch.as_tensor(q,device=device)

    def choose(self,count,rng):
        ids=rng.integers(len(self.by_episode),size=count)
        return np.array([v[rng.integers(len(v))] for i in ids for v in [self.by_episode[i]]])

    def batch(self,roots,augmentation_rng=None):
        roots=torch.as_tensor(roots,device=self.device)
        raw=self.obs[roots[:,None]+torch.arange(-15,1,device=self.device)].clone()
        if augmentation_rng is not None:
            # Apply exactly the same per-finger contact burst to the two arms.
            for i,f in np.argwhere(augmentation_rng.random((len(roots),4))<.1):
                n=int(augmentation_rng.integers(1,4));stop=int(augmentation_rng.integers(n+1,17));start=stop-n
                raw[i,start:stop,f,4:10]=raw[i,start-1,f,4:10]
                raw[i,start:stop,f,10]=0;raw[i,start:min(stop+1,16),f,15:21]=0
        delta=self.q[roots[:,None]+torch.arange(1,17,device=self.device)]-self.q[roots,None]
        return ((raw-self.norm['mean'])/self.norm['std'],
                (delta/self.norm['scale']).reshape(len(roots),16,16),
                delta.reshape(len(roots),16,16),raw[:,-1,:,11:15].reshape(len(roots),16))

    def validation(self):
        roots=[];ids=[]
        for eid in np.unique(self.val_roots[:,0]):
            r=self.val_roots[self.val_roots[:,0]==eid,1]
            selected=r[np.linspace(0,len(r)-1,min(16,len(r))).astype(int)]
            roots.extend(selected);ids.extend([int(eid)]*len(selected))
        return np.asarray(roots),np.asarray(ids)


@torch.no_grad()
def validate(models,arm,data):
    models.eval();roots,ids=data.validation();err=[];signed=[];hold=[];cv=[]
    g=torch.Generator(device=data.device).manual_seed(77119)
    for start in range(0,len(roots),64):
        x,_,truth,velocity=data.batch(roots[start:start+64])
        pred=sample(models,arm,x,g)*data.norm['scale'].reshape(16,16)
        e=pred-truth;signed.append(e.cpu().numpy());err.append(e.abs().cpu().numpy());hold.append(truth.abs().cpu().numpy())
        expected=velocity.clamp(-1,1)[:,None]*torch.arange(1,17,device=data.device)[None,:,None]*.05
        cv.append((expected-truth).abs().cpu().numpy())
    e=np.concatenate(err);bias=np.concatenate(signed);h=np.concatenate(hold);v=np.concatenate(cv)
    def avg(a):return float(np.mean([a[ids==i].mean() for i in np.unique(ids)]))
    result=dict(near4_mae_rad=avg(e[:,:4]),all16_mae_rad=avg(e),hold_near4_mae_rad=avg(h[:,:4]),
                velocity_near4_mae_rad=avg(v[:,:4]),per_finger_near4_rad=[avg(e[:,:4,4*f:4*f+4]) for f in range(4)],
                joint_bias_rad=bias.mean((0,1)).tolist(),windows=len(roots),episodes=len(np.unique(ids)))
    models.train();return result


class ControlledRuntime:
    def __init__(self,config,device,seed):
        self.path=Path(config['comparison_checkpoint']);self.device=torch.device(device);self.seed=int(seed)
        state=torch.load(self.path,map_location='cpu',weights_only=False)
        assert state['kind']=='controlled_flat_single_v1'
        self.arm=state['arm'];self.models=build(self.arm,self.device)
        if state['config']!=configs(self.arm):raise ValueError('Checkpoint architecture differs from frozen provider')
        self.models.load_state_dict(state['model']);self.models.eval().requires_grad_(False)
        self.denoise_steps = config.get('denoise_steps', 10)
        self.record_latency = config.get('record_latency', False)
        train_steps = next(iter(self.models.values())).diffusion.noise_scheduler.config.num_train_timesteps
        if type(self.denoise_steps) is not int or not 1 <= self.denoise_steps <= train_steps:
            raise ValueError(f'Denoise steps must be an integer in [1, {train_steps}]')
        self.thumb_stream = (torch.cuda.Stream(device=self.device)
                             if self.arm == 'single' and self.device.type == 'cuda' else None)
        self.denoise_graph = None
        if self.thumb_stream is not None:
            try:
                self.denoise_graph = SingleDenoiseGraph(self.models, self.device, self.thumb_stream)
            except RuntimeError as exc:
                warnings.warn(f'Finger DP CUDA graph capture failed; using eager CUDA streams: {exc}',
                              RuntimeWarning, stacklevel=2)
        self.norm={k:torch.as_tensor(v,device=device) for k,v in state['normalization'].items()}
        self.limits=state['joint_limits'];self.records=[];self.step=state['step']

    @torch.no_grad()
    def __call__(self,frame,q,history):
        started = perf_counter() if self.record_latency else None
        from evaluate_matched_regrasp import noise_key
        s=self.supervisor
        key=noise_key(int(s.cursor[0]),int(s.phase[0]),int(s.age[0]),int(s.lengths[0]))
        seed=int.from_bytes(hashlib.blake2b(f'{self.seed}:{key}:controlled_joint_noise_v1'.encode(),digest_size=8).digest(),'little')%(2**63-1)
        g=torch.Generator(device=self.device).manual_seed(seed)
        raw=project_history(history,range(4),self.limits).permute(0,2,1,3)
        x=(raw-self.norm['mean'])/self.norm['std']
        graph = self.denoise_graph
        if graph is not None and not graph.supports(x):
            graph = None
        delta=sample(self.models,self.arm,x,g,thumb_stream=self.thumb_stream,
                     denoise_graph=graph,denoise_steps=self.denoise_steps)*self.norm['scale'].reshape(16,16)
        mode = 'cuda_graph_streams' if graph is not None else (
            'cuda_streams' if self.thumb_stream is not None else 'serial')
        out=q[:,None]+delta
        if not torch.isfinite(out).all():raise ValueError('Nonfinite decoded chunk')
        record = dict(frame=int(frame),cursor=int(s.cursor[0]),phase=int(s.phase[0]),
                      noise_key=key,seed=seed,branch_execution=mode)
        if started is not None:
            # The finite check has already waited for the decoded CUDA output.
            record['dp_latency_ms'] = (perf_counter() - started) * 1000
        self.records.append(record)
        return out

    def report(self):
        return dict(kind='controlled_flat_single_v1',arm=self.arm,checkpoint=str(self.path.resolve()),
                    sha256=sha(self.path),step=self.step,calls=len(self.records),
                    noise='shared joint-space progress-keyed DDPM stream',noise_schedule=self.records,
                    anchor='same measured q',denoise_steps=self.denoise_steps)
