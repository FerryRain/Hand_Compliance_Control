"""Single-finger action learning from the existing grouped Mustard dataset.

This is not the old finger-token encoder: each example and each output contains
one finger only. Long fingers share weights; the thumb has a separate policy.
The full-hand controller is unchanged and receives the assembled absolute chunk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'mcc_finger_compliance_control/data'
DEFAULT_DATA = DATA / 'dp_stratified_222_20261022'
HISTORY = HORIZON = 16
STRIDE = 5
ROBOT_DIM = 33
STATE_DIM = 45
FAMILIES = {'long': (0, 1, 2), 'thumb': (3,)}
FIELDS = [('q', 4), ('contact_position_palm', 3), ('contact_normal_palm', 3),
          ('contact_valid', 1), ('q_velocity', 4), ('contact_point_velocity_palm', 3),
          ('contact_normal_rate_palm', 3), ('joint_lower_margin_rad', 4),
          ('joint_upper_margin_rad', 4), ('finger_identity', 4),
          ('palm_twist_palm', 6), ('published_palm_delta_pose_palm', 6)]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def robot_limits():
    import mujoco
    from surface_mcc_finger import HAND_JOINT_NAMES, _fixed_hand_model
    model = _fixed_hand_model()
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINT_NAMES]
    return np.asarray(model.jnt_range[ids], dtype=np.float32).reshape(4, 4, 2)


def project_history(history, finger_ids, limits):
    """One implementation for training and runtime; (...,time,96) -> (...,F,time,45).

    Palm coordinates are retained exactly. Identity conditions on mounting
    differences. No other finger's measured state or future label is exposed.
    Signed margins use physical robot limits, never demonstration extrema.
    """
    if history.shape[-1] != 96:
        raise ValueError('Expected the established 96-dimensional observation')
    limits = torch.as_tensor(limits, device=history.device, dtype=history.dtype)
    values = []
    for finger in finger_ids:
        q = history[..., 4*finger:4*finger+4]
        local = [q, history[..., 16+3*finger:19+3*finger],
                 history[..., 28+3*finger:31+3*finger], history[..., 40+finger:41+finger],
                 history[..., 44+4*finger:48+4*finger],
                 history[..., 60+3*finger:63+3*finger], history[..., 72+3*finger:75+3*finger],
                 q-limits[finger, :, 0], limits[finger, :, 1]-q]
        identity = torch.eye(4, device=history.device, dtype=history.dtype)[finger]
        local += [identity.expand(*q.shape[:-1], 4), history[..., 84:96]]
        values.append(torch.cat(local, -1))
    return torch.stack(values, -3)


def split_check(rows):
    parents = {}
    ids = set()
    for row in rows:
        if row['episode_id'] in ids:
            raise ValueError('Duplicate episode')
        ids.add(row['episode_id'])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        parents.setdefault(row['related_plan_group_id'], set()).add(row['split'])
    if any(len(s) != 1 for s in parents.values()):
        raise ValueError('Parent/tube split leakage')


def prepare(dataset, output):
    """Extract compact frame arrays and root indices, not overlapping windows."""
    from dp_dataset import load_episodes
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = dataset/'teachers_commanded_palm.h5'
    manifest = dataset/'episode_split.json'
    rows = json.loads(manifest.read_text())['episodes']
    split_check(rows)
    with h5py.File(source) as h:
        if str(h.attrs.get('dp_state_schema')) != 'contact_geometry_planner_motion':
            raise ValueError('Unexpected source schema')
        if str(h.attrs.get('contact_region_semantics')) != 'whole_true_fingertip_split_collision_geom':
            raise ValueError('Must use whole true-tip split collision data')
        if float(h.attrs['control_dt']) != .01 or int(h.attrs['planner_waypoints']) != 1:
            raise ValueError('Unexpected source time contract')
        initial_valid = []
        ids = h['episode_id'][:].reshape(-1)
        masks = h['fingertip_contact_mask'][:].reshape(-1, 4)
        for row in rows:
            initial_valid.append(bool((masks[np.flatnonzero(ids == row['episode_id'])[0]] > .5).all()))
    # The legacy loader forward-fills an initially missing contact. Refuse that
    # case rather than quietly introducing future tactile information.
    if not all(initial_valid):
        raise ValueError('Initial missing contacts require a causal source-loader adaptation')
    episodes = load_episodes(source, STRIDE, action_field='q_hand', action_dim=16)
    limits = robot_limits()
    observations, joints, frame_ids, indices = [], [], [], []
    cursor = 0
    for row in rows:
        eid = row['episode_id']; state, q = episodes[eid]
        if not np.array_equal(state[:, :16], q):
            raise ValueError('Labels must be actual measured q, not planner targets')
        if not np.isfinite(state).all() or not np.isfinite(q).all():
            raise ValueError('Nonfinite source data')
        projected = project_history(torch.from_numpy(state), range(4), limits).numpy()
        observations.append(projected.transpose(1, 0, 2))
        joints.append(q.reshape(-1, 4, 4))
        frame_ids.append(np.full(len(q), eid, np.int32))
        for t in range(HISTORY-1, len(q)-HORIZON):
            indices.append((eid, cursor+t, t*STRIDE, ('train','val','test').index(row['split'])))
        cursor += len(q)
    obs, q, frame_ids = np.concatenate(observations), np.concatenate(joints), np.concatenate(frame_ids)
    indices = np.asarray(indices, dtype=np.int64)
    for eid, root, _, _ in indices:
        if frame_ids[root-HISTORY+1] != eid or frame_ids[root+HORIZON] != eid:
            raise AssertionError('Window crosses an episode boundary')
    np.savez_compressed(output/'single_finger_frames.npz', observation=obs, q=q,
                        episode_id=frame_ids, roots=indices, joint_limits=limits)
    split_counts = {s: sum(r['split'] == s for r in rows) for s in ('train','val','test')}
    audit = dict(source=str(source.resolve()), source_sha256=sha(source),
                 split_source=str(manifest.resolve()), split_sha256=sha(manifest),
                 episodes=len(rows), split=split_counts, parent_split_leakage=False,
                 raw_frames=sum(r['frames'] for r in rows), sampled_frames=len(q),
                 histories=HISTORY, prediction=HORIZON, stride=STRIDE, control_dt=.01,
                 fields=FIELDS, state_dim=STATE_DIM, action_dim=4,
                 labels='future measured q_i minus SAME current measured q_i for all16 waypoints',
                 contact_region='whole true fingertip; no fixed pad point or added contact gate',
                 coordinate_frame='existing palm_lower frame; finger identity encodes mounting',
                 input_scope='only own finger state plus published/common palm motion',
                 split_preserved_for_all_fingers=True, test_used_for_normalization_or_selection=False,
                 window_counts={s:int((indices[:,3] == k).sum()) for k,s in enumerate(('train','val','test'))},
                 finger_window_counts={family:{s:int((indices[:,3]==k).sum())*len(fingers)
                                      for k,s in enumerate(('train','val','test'))}
                                      for family,fingers in FAMILIES.items()},
                 extraction='frame arrays plus index view; no duplicated H5 trajectories',
                 source_selection='existing accepted222 trajectories, no new filtering')
    save(output/'data_audit.json', audit)
    save(output/'episode_split.json', dict(episodes=rows))
    print(json.dumps(audit, indent=2), flush=True)
    return audit


class FingerWindows:
    def __init__(self, directory, family, device='cpu'):
        self.directory, self.family, self.device = Path(directory), family, torch.device(device)
        self.fingers = FAMILIES[family]
        with np.load(self.directory/'single_finger_frames.npz') as data:
            self.roots = data['roots'].copy()
            self.limits = data['joint_limits'].copy()
            self.observation = torch.as_tensor(data['observation'][:, self.fingers], device=device)
            self.q = torch.as_tensor(data['q'][:, self.fingers], device=device)
        self.train_roots = self.roots[self.roots[:,3] == 0]
        self.val_roots = self.roots[self.roots[:,3] == 1]
        self.by_episode = [r[:,1] for eid in np.unique(self.train_roots[:,0])
                           if len(r := self.train_roots[self.train_roots[:,0] == eid])]
        # Physical scales are fitted on TRAIN frames/windows only.
        all_train_frames = []
        with np.load(self.directory/'single_finger_frames.npz') as data:
            all_train_frames = np.isin(data['episode_id'], np.unique(self.train_roots[:,0]))
        x = self.observation[torch.as_tensor(all_train_frames, device=device)].double().reshape(-1, STATE_DIM)
        self.mean = x.mean(0).float(); self.std = x.std(0, correction=0).clamp_min(1e-5).float()
        self.mean[10] = self.std[10] = .5
        self.mean[29:33] = 0.; self.std[29:33] = 1.  # identity is not a statistical feature
        self.action_scale = torch.full((HORIZON,4), 1e-4, device=device)
        for rows in np.array_split(self.train_roots, max(1, len(self.train_roots)//2048)):
            root = torch.as_tensor(rows[:,1], device=device)
            future = self.q[root[:,None]+torch.arange(1,HORIZON+1,device=device)[None]]
            delta = future-self.q[root,None]
            self.action_scale = torch.maximum(self.action_scale, delta.abs().amax((0,2)))
        self.normalization = {k:v.detach().cpu().numpy() for k,v in
                              dict(state_mean=self.mean,state_std=self.std,action_std=self.action_scale).items()}

    def training_indices(self, batch, rng):
        e = rng.integers(len(self.by_episode), size=batch)
        root = np.asarray([self.by_episode[i][rng.integers(len(self.by_episode[i]))] for i in e])
        finger = rng.integers(len(self.fingers), size=batch)
        return np.c_[root, finger]

    def validation_indices(self, per_episode=16):
        pairs, episode_ids = [], []
        for eid in np.unique(self.val_roots[:,0]):
            roots = self.val_roots[self.val_roots[:,0] == eid,1]
            selected = roots[np.linspace(0,len(roots)-1,min(per_episode,len(roots))).astype(int)]
            for r in selected:
                for f in range(len(self.fingers)):
                    pairs.append((r,f)); episode_ids.append(int(eid))
        return np.asarray(pairs), np.asarray(episode_ids)

    def batch(self, pairs, dropout_rng=None):
        pairs = torch.as_tensor(pairs, device=self.device)
        root, finger = pairs[:,0], pairs[:,1]
        raw = self.observation[root[:,None]+torch.arange(-HISTORY+1,1,device=self.device),finger[:,None]].clone()
        future = self.q[root[:,None]+torch.arange(1,HORIZON+1,device=self.device),finger[:,None]]
        anchor = self.q[root,finger]
        delta = future-anchor[:,None]
        velocity = raw[:,-1,11:15].clone()
        if dropout_rng is not None:
            # Same 10% contact burst augmentation as the old DP, only this finger.
            for i in np.flatnonzero(dropout_rng.random(len(pairs)) < .1):
                n=int(dropout_rng.integers(1,4)); stop=int(dropout_rng.integers(n+1,HISTORY+1)); start=stop-n
                raw[i,start:stop,4:10]=raw[i,start-1,4:10]
                raw[i,start:stop,10]=0
                raw[i,start:min(stop+1,HISTORY),15:21]=0
        x = (raw-self.mean)/self.std
        return dict(**{'observation.state':x[:,:,:ROBOT_DIM],
                       'observation.environment_state':x[:,:,ROBOT_DIM:]},
                    action=delta/self.action_scale, action_is_pad=torch.zeros((len(pairs),HORIZON),
                    device=self.device,dtype=torch.bool)), delta, velocity


def configuration():
    return dict(robot_state_dim=ROBOT_DIM,environment_state_dim=12,action_dim=4,
                obs_horizon=HISTORY,pred_horizon=HORIZON,down_dims=[64,128,256],
                kernel_size=5,n_groups=8,diffusion_step_embed_dim=128,
                noise_scheduler='DDPM',diffusion_steps=100,inference_steps=10,
                conditioning_architecture='flat',policy_architecture='unet',action_sample_clip='none')


@torch.no_grad()
def validate(policy, dataset, seed=77119):
    policy.eval()
    pairs,eids=dataset.validation_indices()
    errors, hold, velocity, signed, losses = [],[],[],[],[]
    # Fixed validation diffusion noise; does not advance the training RNG.
    with torch.random.fork_rng(devices=[dataset.device.index or 0] if dataset.device.type=='cuda' else []):
        torch.manual_seed(seed)
        generator=torch.Generator(device=dataset.device).manual_seed(seed)
        for part in np.array_split(np.arange(len(pairs)),max(1,(len(pairs)+127)//128)):
            batch,truth,qd=dataset.batch(pairs[part])
            losses.append(float(policy.diffusion.compute_loss(batch)))
            cond=policy.diffusion._prepare_global_conditioning(batch)
            pred=policy.diffusion.conditional_sample(len(part),global_cond=cond,generator=generator)*dataset.action_scale
            cv=qd.clamp(-1,1)[:,None]*(.05*torch.arange(1,HORIZON+1,device=dataset.device)[None,:,None])
            errors.append((pred-truth).abs().cpu().numpy()); signed.append((pred-truth).cpu().numpy())
            hold.append(truth.abs().cpu().numpy()); velocity.append((cv-truth).abs().cpu().numpy())
    err=np.concatenate(errors); hold=np.concatenate(hold); cv=np.concatenate(velocity); signed=np.concatenate(signed)
    def equal_episode(x):
        return float(np.mean([x[eids==eid].mean() for eid in np.unique(eids)]))
    by_finger={str(f):dict(mae_rad=equal_episode(err[pairs[:,1]==k]) if len(dataset.fingers)==1 else float(err[pairs[:,1]==k].mean()),
                    near4_mae_rad=float(err[pairs[:,1]==k,:4].mean()),
                    joint_bias_rad=signed[pairs[:,1]==k].mean((0,1)).tolist())
               for k,f in enumerate(dataset.fingers) if np.any(pairs[:,1] == k)}
    result=dict(val_noise_loss=float(np.mean(losses)), val_mae_rad=equal_episode(err),
                val_near4_mae_rad=equal_episode(err[:,:4]),hold_mae_rad=equal_episode(hold),
                hold_near4_mae_rad=equal_episode(hold[:,:4]),velocity_mae_rad=equal_episode(cv),
                velocity_near4_mae_rad=equal_episode(cv[:,:4]),per_finger=by_finger,
                windows=len(pairs),episodes=len(np.unique(eids)),selection='episode-equal near4 MAE',
                joint_mae_rad=err.mean((0,1)).tolist(),joint_bias_rad=signed.mean((0,1)).tolist())
    policy.train()
    return result


def train(directory, family, steps=6000, batch_size=128, device='cuda:0', seed=20261109, eval_every=500):
    from train_dp import build_policy
    torch.set_num_threads(4)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires authorized CUDA; no CPU fallback')
    torch.manual_seed(seed); np.random.seed(seed)
    dataset=FingerWindows(directory,family,device)
    cfg=configuration(); policy=build_policy(SimpleNamespace(**cfg),torch.device(device))
    optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=1e-6)
    folder=Path(directory)/family;folder.mkdir(exist_ok=True)
    if (folder/'latest.pt').exists():raise ValueError('Preserve existing run; use a new output directory')
    rng=np.random.default_rng(seed)
    metrics=[];best=float('inf');started=time.monotonic();accum=[]
    parameters=sum(p.numel() for p in policy.parameters())
    save(folder/'config.json',dict(model=cfg,family=family,training_steps=steps,batch_size=batch_size,
          seed=seed,parameters=parameters,optimizer='AdamW',lr=3e-4,
          sampling='equal train episode, equal finger, uniform valid root',
          source_audit=json.loads((Path(directory)/'data_audit.json').read_text())))
    for step in range(1,steps+1):
        batch,_,_=dataset.batch(dataset.training_indices(batch_size,rng),rng)
        loss=policy.diffusion.compute_loss(batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step();accum.append(float(loss.detach()))
        if step % 100 == 0:
            save(folder/'progress.json',dict(state='training',step=step,total=steps,
                 elapsed_s=time.monotonic()-started,mean_loss=float(np.mean(accum[-100:]))))
        if step == 1 or step % eval_every == 0 or step == steps:
            val=validate(policy,dataset)
            row=dict(step=step,train_loss=float(np.mean(accum)),elapsed_s=time.monotonic()-started,**val)
            metrics.append(row);save(folder/'metrics.json',metrics);accum=[]
            state=dict(kind='single_finger_dp_v1',family=family,config=cfg,
                       model={k:v.detach().cpu() for k,v in policy.state_dict().items()},
                       normalization=dataset.normalization,joint_limits=dataset.limits,
                       finger_ids=dataset.fingers,step=step,metrics=row,
                       data_audit_sha256=sha(Path(directory)/'data_audit.json'),
                       action_representation='anchored_delta_q',fields=FIELDS,
                       training_seed=seed,test_used=False)
            torch.save(state,folder/'latest.pt')
            if row['val_near4_mae_rad'] < best:
                best=row['val_near4_mae_rad'];torch.save(state,folder/'best.pt')
            print('[SINGLE-FINGER]',family,json.dumps(row),flush=True)
    torch.save(optimizer.state_dict(),folder/'optimizer_last.pt')
    save(folder/'progress.json',dict(state='completed',step=steps,elapsed_s=time.monotonic()-started,
          best_near4_mae_rad=best,parameters=parameters))


class SingleFingerRuntime:
    """Opt-in chunk provider. Scheduler, MCC, bootstrap and palm timing are untouched."""
    def __init__(self, config, device, seed):
        from train_dp import build_policy
        self.device=torch.device(device);self.seed=int(seed);self.models={};self.calls=0
        self.paths={k:str(Path(config[k]).resolve()) for k in FAMILIES}
        self.mode=config.get('mode','all')
        if self.mode not in ('all','long_only','thumb_only'):raise ValueError('Unknown finger replacement mode')
        self.checkpoints={}
        for family in FAMILIES:
            state=torch.load(self.paths[family],map_location='cpu',weights_only=False)
            if state.get('kind')!='single_finger_dp_v1' or state['family']!=family:
                raise ValueError('Wrong finger checkpoint')
            model=build_policy(SimpleNamespace(**state['config']),self.device)
            model.load_state_dict(state['model']);model.eval().requires_grad_(False)
            norm={k:torch.as_tensor(v,device=device) for k,v in state['normalization'].items()}
            self.models[family]=(model,norm,torch.as_tensor(state['joint_limits'],device=device))
            self.checkpoints[family]=dict(path=self.paths[family],sha256=sha(self.paths[family]),step=state['step'])

    @torch.no_grad()
    def __call__(self, frame, q, history):
        if self.mode != 'all':raise ValueError('Partial replacement needs a baseline provider')
        out=q[:,None,:].expand(-1,HORIZON,-1).clone()
        for family,fingers in FAMILIES.items():
            model,norm,limits=self.models[family]
            x=project_history(history,fingers,limits).reshape(-1,HISTORY,STATE_DIM)
            x=(x-norm['state_mean'])/norm['state_std']
            batch={'observation.state':x[:,:,:ROBOT_DIM],'observation.environment_state':x[:,:,ROBOT_DIM:]}
            cond=model.diffusion._prepare_global_conditioning(batch)
            key=f'{self.seed}:{frame}:{family}:single_finger_v1'.encode()
            seed=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')%(2**63-1)
            generator=torch.Generator(device=self.device).manual_seed(seed)
            pred=model.diffusion.conditional_sample(len(x),global_cond=cond,generator=generator)
            delta=(pred*norm['action_std']).reshape(len(q),len(fingers),HORIZON,4)
            for slot,finger in enumerate(fingers):
                out[:,:,4*finger:4*finger+4]+=delta[:,slot]
        self.calls+=1
        if not torch.isfinite(out).all():raise ValueError('Nonfinite single-finger chunk')
        return out

    def report(self):
        return dict(kind='single_finger_dp_v1',checkpoints=self.checkpoints,calls=self.calls,
                    local_state_dim=STATE_DIM,robot_dim=ROBOT_DIM,mode=self.mode,
                    same_parent_split=True,anchor='same measured q for each future waypoint',
                    sampling='wall-frame keyed independent family streams; not identical noise to16D baseline')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','train'])
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=FAMILIES,default='long')
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261109)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.dataset,a.output)
    else:train(a.output,a.family,a.steps,a.batch_size,seed=a.seed,eval_every=a.eval_every)


if __name__=='__main__':main()
