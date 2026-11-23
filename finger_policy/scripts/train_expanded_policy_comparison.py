"""Train unchanged matched flat/single architectures on frozen expansion pools."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from controlled_policy_ablation import build,configs,shared_loss,save,sha
from expanded_policy_data import ExpandedWindows,validate_expansion


def main(a):
    p=json.loads(a.protocol.read_text());arm='flat' if a.group=='flat_common' else 'single'
    if a.output.exists():raise FileExistsError(a.output)
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    for path,digest in p['frozen_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen input changed: '+path)
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    data=ExpandedWindows(p,a.group,a.device);models=build(arm,a.device)
    opt=torch.optim.AdamW(models.parameters(),lr=p['lr'],weight_decay=1e-6)
    rng=np.random.default_rng(a.seed);aug=np.random.default_rng(a.seed+100000)
    noise=torch.Generator(device=a.device).manual_seed(a.seed+200000)
    a.output.mkdir(parents=True);start=time.monotonic();best=float('inf');rows=[];losses=[];digest=hashlib.sha256()
    budget=a.canary_steps or p['updates'];every=budget if a.canary_steps else p['eval_every']
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in models.state_dict().values())).hexdigest()
    for step in range(1,budget+1):
        indices=data.choose(p['full_hand_windows_per_update'],rng)
        for i in indices:digest.update(i.tobytes())
        x,action=data.batch(indices,aug)
        eps=torch.randn(action.shape,device=a.device,generator=noise)
        t=torch.randint(100,(len(x),),device=a.device,generator=noise)
        loss=shared_loss(models,arm,x,action,eps,t)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(models.parameters(),1.,error_if_nonfinite=True);opt.step()
        losses.append(float(loss.detach()))
        if step%100==0:save(a.output/'progress.json',dict(stage='training',complete=False,step=step,total=budget,elapsed_s=time.monotonic()-start,loss=float(np.mean(losses[-100:]))))
        if step%every and step!=budget:continue
        metrics=validate_expansion(models,arm,data)
        row=dict(step=step,elapsed_s=time.monotonic()-start,train_loss=float(np.mean(losses)),**metrics);losses=[];rows.append(row);save(a.output/'metrics.json',rows)
        state=dict(kind='controlled_flat_single_v1',arm=arm,config=configs(arm),model={k:v.detach().cpu().clone() for k,v in models.state_dict().items()},
            normalization=data.normalization,joint_limits=data.limits,step=step,training_seed=a.seed,metrics=row,protocol_sha256=sha(a.protocol),
            window_stream_sha256=digest.hexdigest(),initialization='random',expansion_group=a.group,test_used=False)
        torch.save(state,a.output/'last.tmp.pt');(a.output/'last.tmp.pt').replace(a.output/'last.pt')
        if metrics['selection_score']<best:best=metrics['selection_score'];torch.save(state,a.output/'best.pt')
        torch.save(dict(optimizer=opt.state_dict(),window_rng=rng.bit_generator.state,augmentation_rng=aug.bit_generator.state,noise_rng=noise.get_state(),step=step),a.output/'resume_last.pt')
        save(a.output/'sampling_counts.json',{k:v.tolist() for k,v in data.draw_counts.items()})
        print(a.group,a.seed,json.dumps(row),flush=True)
    final=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in models.state_dict().values())).hexdigest()
    assert final!=initial and all(torch.isfinite(v).all() for v in models.state_dict().values())
    assert all(len(set(v.tolist()))==1 for v in data.draw_counts.values())
    save(a.output/'progress.json',dict(complete=True,step=budget,elapsed_s=time.monotonic()-start,parameters_changed=True,finite=True,
        best_score=best,parameters=sum(v.numel() for v in models.parameters()),window_stream_sha256=digest.hexdigest(),
        per_anatomical_finger_examples=budget*p['full_hand_windows_per_update'],family_loss='original .75 long + .25 thumb; no loss or architecture changes',canary=bool(a.canary_steps)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--group',choices=['flat_common','single_common','single_all'],required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--canary-steps',type=int,default=0)
    main(ap.parse_args())
