"""
端到端验证: 小数据训练 DP → 保存 → 正确推理 → 评估 Action R²

用法:
  python finger_compliance_control/scripts/eval_trained_dp.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

H5_PATH = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
    "headless_train_20260414_175152.h5"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════ Dataset ═══════════════════════════════

class WindowDataset(Dataset):
    def __init__(self, state, action, n_obs_steps, horizon):
        self.state, self.action = state, action
        self.T, self.N = state.shape[0], state.shape[1]
        self.n_obs, self.H = n_obs_steps, horizon
        t0, t1 = n_obs_steps - 1, self.T - horizon + n_obs_steps
        self.indices = [(t, e) for t in range(t0, t1) for e in range(self.N)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, e = self.indices[idx]
        return {
            "observation.state":
                torch.from_numpy(self.state[t - self.n_obs + 1:t + 1, e]
                                 .astype(np.float32)),
            "observation.environment_state":
                torch.zeros(self.n_obs, 1, dtype=torch.float32),
            "action":
                torch.from_numpy(self.action[t - self.n_obs + 1:
                                             t - self.n_obs + 1 + self.H, e]
                                 .astype(np.float32)),
            "action_is_pad": torch.zeros(self.H, dtype=torch.bool),
        }


# ═════════════════════════ Step 1: Train + Save ═════════════════════════

def train_and_save(save_dir, n_steps=300, n_envs=1, horizon=16,
                   n_obs_steps=2, n_action_steps=8,
                   training_steps=500, batch_size=64, lr=3e-4):
    print(f"\n{'='*60}")
    print(f"[TRAIN] {n_steps} steps x {n_envs} envs, H={horizon}, "
          f"{training_steps} training steps")

    with h5py.File(H5_PATH, "r") as f:
        q = np.asarray(f["q"][:n_steps, :n_envs, :22], dtype=np.float32)
        fsr = np.asarray(f["fsr"][:n_steps, :n_envs, :16], dtype=np.float32)
        act = np.asarray(f["action"][:n_steps, :n_envs, :16], dtype=np.float32)
    state = np.concatenate([q, fsr], axis=-1).astype(np.float32)

    stats = {
        "observation.state": {
            "min": torch.from_numpy(state.min(axis=(0, 1)).astype(np.float32)),
            "max": torch.from_numpy(state.max(axis=(0, 1)).astype(np.float32)),
        },
        "action": {
            "min": torch.from_numpy(act.min(axis=(0, 1)).astype(np.float32)),
            "max": torch.from_numpy(act.max(axis=(0, 1)).astype(np.float32)),
        },
    }

    input_f = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(38,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
    }
    output_f = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}

    cfg = DiffusionConfig(
        input_features=input_f, output_features=output_f,
        n_obs_steps=n_obs_steps, horizon=horizon,
        n_action_steps=n_action_steps, device=DEVICE,
        down_dims=(512, 1024, 2048),
    )

    policy = DiffusionPolicy(cfg).to(DEVICE)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=stats)

    ds = WindowDataset(state, act, n_obs_steps, horizon)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  params={n_params:,}  windows={len(ds)}  batches={len(dl)}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    policy.train()

    pbar = tqdm(total=training_steps, desc="  train", unit="step",
                dynamic_ncols=True) if tqdm is not None else None
    losses, step = [], 0
    try:
        while step < training_steps:
            for batch in dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                batch = preprocessor(batch)
                with torch.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
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
                if step >= training_steps:
                    break
    finally:
        if pbar is not None:
            pbar.close()

    final_loss = float(np.mean(losses[-min(50, len(losses)):]))
    print(f"  final loss: {final_loss:.5f}")

    # Save pretrained format
    save_path = Path(save_dir)
    policy.save_pretrained(save_path)
    preprocessor.save_pretrained(save_path)
    _, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)
    postprocessor.save_pretrained(save_path)
    print(f"  saved -> {save_path}")
    return final_loss


# ═══════════════════ Step 2: Correct Inference + Eval ═════════════════

def evaluate_pretrained(model_dir, n_steps=500, n_envs=2):
    print(f"\n{'='*60}")
    print(f"[EVAL] Correct inference: single-step -> preprocessor -> select_action")

    # INFERENCE mode preprocessor (pretrained_path, NOT dataset_stats!)
    policy = DiffusionPolicy.from_pretrained(model_dir, local_files_only=True)
    cfg = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=model_dir
    )
    policy.to(DEVICE)
    policy.eval()

    n_obs = cfg.n_obs_steps
    print(f"  n_obs_steps={n_obs}  horizon={cfg.horizon}")

    # Load eval data
    with h5py.File(H5_PATH, "r") as f:
        q = np.asarray(f["q"][:n_steps, :n_envs, :22], dtype=np.float32)
        fsr = np.asarray(f["fsr"][:n_steps, :n_envs, :16], dtype=np.float32)
        action_gt = np.asarray(f["action"][:n_steps, :n_envs, :16], dtype=np.float32)
    state = np.concatenate([q, fsr], axis=-1).astype(np.float32)
    T, N = state.shape[0], state.shape[1]

    all_preds, all_trues = [], []

    for e in (tqdm(range(N), desc="  envs", unit="env", dynamic_ncols=True)
              if tqdm is not None else range(N)):
        env_preds, env_trues = [], []
        for t in range(T):
            obs = torch.from_numpy(state[t, e]).unsqueeze(0).to(DEVICE)
            env_t = torch.zeros(1, 1, device=DEVICE)
            batch = {"observation.state": obs, "observation.environment_state": env_t}
            batch = preprocessor(batch)       # single-step -> buffers internally
            pred = policy.select_action(batch)
            pred = postprocessor(pred)
            env_preds.append(pred.cpu().numpy()[0])
            env_trues.append(action_gt[t, e])
        all_preds.append(np.stack(env_preds))
        all_trues.append(np.stack(env_trues))

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)

    # Drop burn-in (first n_obs steps per env have no valid temporal window)
    y_pred = y_pred[N * n_obs:]
    y_true = y_true[N * n_obs:]

    action_names = [
        "j6_iR", "j7_iA", "j8_iM", "j9_iD",
        "j10_mR", "j11_mA", "j12_mM", "j13_mD",
        "j14_rR", "j15_rA", "j16_rM", "j17_rD",
        "j18_tR", "j19_tA", "j20_tM", "j21_tD",
    ]

    r2_per_dim = []
    for d in range(16):
        ss_res = ((y_true[:, d] - y_pred[:, d]) ** 2).sum()
        ss_tot = ((y_true[:, d] - y_true[:, d].mean()) ** 2).sum()
        r2_per_dim.append(1.0 - ss_res / max(ss_tot, 1e-10))

    r2_mean = float(np.mean(r2_per_dim))
    r2_median = float(np.median(r2_per_dim))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    print(f"\n  Action R2: mean={r2_mean:.4f}  median={r2_median:.4f}  MAE={mae:.4f}")
    print(f"  Samples: {y_pred.shape[0]} (after {n_obs}-step burn-in)")

    ranked = sorted(zip(r2_per_dim, action_names))
    print(f"  Best:  {', '.join(f'{n}={r:.3f}' for r, n in ranked[-4:][::-1])}")
    print(f"  Worst: {', '.join(f'{n}={r:.3f}' for r, n in ranked[:4])}")

    return r2_mean, r2_median, mae, r2_per_dim


# ═════════════════════════════ Main ═══════════════════════════════════

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--save-dir", type=str, default="/tmp/dp_overfit_test_model")
    args = parser.parse_args()
    DEVICE = args.device

    train_loss = train_and_save(
        args.save_dir,
        n_steps=300, n_envs=1,
        horizon=16, n_obs_steps=2, n_action_steps=8,
        training_steps=500,
    )

    r2_mean, r2_median, mae, r2_per_dim = evaluate_pretrained(
        args.save_dir,
        n_steps=500, n_envs=2,
    )

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"  Train loss (windowed preprocessor):  {train_loss:.5f}")
    print(f"  Eval  R2   (sequential inference):   mean={r2_mean:.4f}")
    if r2_mean > 0.5:
        print("  Pipeline works end-to-end!")
    elif r2_mean > 0.0:
        print("  Pipeline runs, predictions weak - needs more training")
    else:
        print("  Pipeline broken")


if __name__ == "__main__":
    main()
