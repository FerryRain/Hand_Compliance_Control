"""
Diffusion Policy Overfit 诊断

关键发现:
  - Training preprocessor (dataset_stats) 保留 windowed batch keys
  - Inference preprocessor (pretrained_path) 缓冲单步 obs

用法:
  python finger_compliance_control/scripts/dp_overfit_diagnosis.py --quick
"""

from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py, numpy as np, torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

H5_PATH = Path("/home/rimlab/Code/Hand_Compliance_Control/"
               "finger_compliance_control/data/headless/"
               "headless_train_20260414_175152.h5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OverfitConfig:
    name: str
    clip_fsr_pct: float
    stride: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    n_envs: int
    n_steps: int
    training_steps: int
    batch_size: int
    lr: float
    down_dims: tuple = (64, 128, 256)  # None = use default (512,1024,2048)


def load_subset(h5_path, n_steps, n_envs, stride, clip_fsr_pct):
    with h5py.File(h5_path, "r") as f:
        q = np.asarray(f["q"][:n_steps*stride:stride, :n_envs, :22], dtype=np.float32)
        fsr = np.asarray(f["fsr"][:n_steps*stride:stride, :n_envs, :16], dtype=np.float32)
        act = np.asarray(f["action"][:n_steps*stride:stride, :n_envs, :16], dtype=np.float32)
    if clip_fsr_pct < 100:
        fsr = np.clip(fsr, 0, float(np.percentile(fsr, clip_fsr_pct)))
    state = np.concatenate([q, fsr], axis=-1).astype(np.float32)
    return state, act


def compute_stats(state, action):
    return {
        "observation.state": {
            "min": torch.from_numpy(state.min(axis=(0,1)).astype(np.float32)),
            "max": torch.from_numpy(state.max(axis=(0,1)).astype(np.float32)),
        },
        "action": {
            "min": torch.from_numpy(action.min(axis=(0,1)).astype(np.float32)),
            "max": torch.from_numpy(action.max(axis=(0,1)).astype(np.float32)),
        },
    }


class WindowDataset(Dataset):
    """输出时序窗口: obs[n_obs_steps,38], action[horizon,16]"""
    def __init__(self, state, action, n_obs_steps, horizon):
        self.state, self.action = state, action
        self.T, self.N = state.shape[0], state.shape[1]
        self.n_obs, self.H = n_obs_steps, horizon
        t0, t1 = n_obs_steps - 1, self.T - horizon + n_obs_steps
        self.indices = [(t, e) for t in range(t0, t1) for e in range(self.N)]
        if not self.indices:
            raise ValueError(f"No valid windows: T={self.T} n_obs={n_obs_steps} H={horizon}")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t, e = self.indices[idx]
        return {
            "observation.state":
                torch.from_numpy(self.state[t-self.n_obs+1:t+1, e].astype(np.float32)),
            "observation.environment_state":
                torch.zeros(self.n_obs, 1, dtype=torch.float32),
            "action":
                torch.from_numpy(self.action[t-self.n_obs+1:t-self.n_obs+1+self.H, e].astype(np.float32)),
            "action_is_pad":
                torch.zeros(self.H, dtype=torch.bool),
        }


def train_and_eval(cfg: OverfitConfig) -> dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"[TEST] {cfg.name}  clip={cfg.clip_fsr_pct}% stride={cfg.stride} "
          f"H={cfg.horizon} obs={cfg.n_obs_steps} act={cfg.n_action_steps}")

    state, action = load_subset(H5_PATH, cfg.n_steps, cfg.n_envs,
                                cfg.stride, cfg.clip_fsr_pct)

    stats = compute_stats(state, action)
    print(f"  state={state.shape}  act={action.shape}  "
          f"fsr∈[{state[...,22:].min():.1f},{state[...,22:].max():.1f}]")

    input_f = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(38,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
    }
    output_f = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}

    policy_cfg = DiffusionConfig(
        input_features=input_f, output_features=output_f,
        n_obs_steps=cfg.n_obs_steps, horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps, device=DEVICE,
        down_dims=cfg.down_dims, kernel_size=5, n_groups=8,
        diffusion_step_embed_dim=128 if cfg.down_dims == (512, 1024, 2048) else 64,
    )
    policy = DiffusionPolicy(policy_cfg).to(DEVICE)
    # TRAINING preprocessor: with dataset_stats → preserves windowed keys
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg, dataset_stats=stats
    )

    ds = WindowDataset(state, action, cfg.n_obs_steps, cfg.horizon)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  params={n_params:,}  windows={len(ds)}  batches={len(dl)}")

    # ── Train ──
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=1e-5)
    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    policy.train()

    pbar = tqdm(total=cfg.training_steps, desc=f"  {cfg.name}", unit="step",
                dynamic_ncols=True) if tqdm is not None else None
    losses, step = [], 0
    try:
        while step < cfg.training_steps:
            for batch in dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                batch = preprocessor(batch)
                with torch.autocast(device_type=DEVICE, enabled=use_amp):
                    loss, _ = policy.forward(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                losses.append(float(loss.item()))
                step += 1
                if pbar is not None:
                    pbar.update(1)
                    if step % 100 == 0:
                        pbar.set_postfix(loss=f"{np.mean(losses[-100:]):.4f}")
                if step >= cfg.training_steps:
                    break
    finally:
        if pbar is not None:
            pbar.close()

    # ── Compute final training set loss as overfit metric ──
    policy.eval()
    eval_losses = []
    dl_eval = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        for batch in dl_eval:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            eval_losses.append(float(loss.item()))

    final_train_loss = float(np.mean(losses[-min(50, len(losses)):]))
    eval_loss = float(np.mean(eval_losses))

    status = "✅ OVERFIT" if eval_loss < 0.3 else ("⚠️ PARTIAL" if eval_loss < 0.6 else "❌ FAIL")
    print(f"  → train_loss(final)={final_train_loss:.5f}  eval_loss={eval_loss:.5f}  {status}")

    return {
        "config": cfg.name, "clip_fsr_pct": cfg.clip_fsr_pct,
        "stride": cfg.stride, "horizon": cfg.horizon,
        "final_train_loss": final_train_loss, "eval_loss": eval_loss,
        "n_windows": len(ds), "n_params": n_params, "status": status,
    }


def main():
    global DEVICE
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", type=str, default=DEVICE)
    args = p.parse_args()
    DEVICE = args.device

    if args.quick:
        configs = [
            OverfitConfig("1_baseline",      100.0, 1, 16, 2, 8, 1, 300, 300, 64, 3e-4),
            OverfitConfig("2_clip_fsr",       99.9, 1, 16, 2, 8, 1, 300, 300, 64, 3e-4),
            OverfitConfig("3_clip_stride5",   99.9, 5,  8, 2, 4, 1, 300, 300, 64, 3e-4),
            OverfitConfig("4_baseline_BIG",  100.0, 1, 16, 2, 8, 1, 300, 300, 32, 1e-4,
                          down_dims=(512, 1024, 2048)),
        ]
    else:
        configs = [
            OverfitConfig("baseline",      100.0, 1, 16, 2, 8, 1, 300, 300, 64, 3e-4),
            OverfitConfig("clip",           99.9, 1, 16, 2, 8, 1, 300, 300, 64, 3e-4),
            OverfitConfig("clip_s5",        99.9, 5, 16, 2, 8, 1, 300, 300, 64, 3e-4),
            OverfitConfig("clip_h8",        99.9, 1,  8, 2, 4, 1, 300, 300, 64, 3e-4),
            OverfitConfig("clip_h8_s5",     99.9, 5,  8, 2, 4, 1, 300, 300, 64, 3e-4),
        ]

    print(f"Running {len(configs)} tests on {DEVICE}")
    results = []
    for cfg in configs:
        try:
            results.append(train_and_eval(cfg))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"config": cfg.name, "error": str(e)[:120]})

    print(f"\n{'='*90}")
    print(f"{'Config':<23s} {'Clip':>5s} {'Strd':>4s} {'H':>3s} "
          f"{'TrainLoss':>11s} {'EvalLoss':>10s}  Status")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"  {r['config']:<23s}  ERROR: {r['error'][:65]}")
        else:
            print(f"  {r['config']:<23s} {r['clip_fsr_pct']:4.1f}% {r['stride']:4d} "
                  f"{r['horizon']:3d} {r['final_train_loss']:11.5f} {r['eval_loss']:10.5f}  "
                  f"{r['status']}")

    out = H5_PATH.parent / "dp_overfit_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")
    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda r: r["eval_loss"])
        print(f"Best: {best['config']}  eval_loss={best['eval_loss']:.5f}")


if __name__ == "__main__":
    main()
