"""
精确力矩分析 v2:
  1. PD 反推验证 (K_p=20, K_d=2 是否正确)
  2. FSR ↔ τ 的线性相关天花板
  3. 测试非线性 MLP 能否突破: (q, action, τ, dτ) → FSR
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter1d

H5_PATH = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
    "headless_train_20260414_175152.h5"
)
OUT_DIR = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
)
N_STEPS = 40000
ENV_IDX = 0

K_P = 20.0
K_D = 2.0
DT = 0.01

TORQUE_NAMES_SHORT = [
    "j06_iR","j07_iA","j08_iM","j09_iD",
    "j10_mR","j11_mA","j12_mM","j13_mD",
    "j14_rR","j15_rA","j16_rM","j17_rD",
    "j18_tR","j19_tA","j20_tM","j21_tD",
]
FSR_NAMES_SHORT = [
    "p0","p1","p2","p3",
    "i_p0","i_p1","i_d",
    "m_p0","m_p1","m_d",
    "r_p0","r_p1","r_d",
    "t_p0","t_p1","t_d",
]
FINGER_GROUPS = {
    "Index":  {"fsr": [4,5,6],    "torque": [0,2,3]},
    "Middle": {"fsr": [7,8,9],    "torque": [4,6,7]},
    "Ring":   {"fsr": [10,11,12], "torque": [8,10,11]},
    "Thumb":  {"fsr": [13,14,15], "torque": [12,14,15]},
}


def load_data(h5_path, n_steps, env_idx):
    with h5py.File(h5_path, "r") as f:
        n_load = min(f["fsr"].shape[0], n_steps)
        fsr    = np.asarray(f["fsr"][:n_load, env_idx, :], dtype=np.float64)
        tq     = np.asarray(f["qfrc_actuator"][:n_load, env_idx, :], dtype=np.float64)
        action = np.asarray(f["action"][:n_load, env_idx, :], dtype=np.float64)
        q_all  = np.asarray(f["q"][:n_load, env_idx, :], dtype=np.float64)
    return fsr, tq, action, q_all


def main():
    fsr, tq, action, q_all = load_data(H5_PATH, N_STEPS, ENV_IDX)
    q_hand = q_all[:, 6:22]  # hand joints
    T, D = q_hand.shape

    # Smooth velocity (stronger smoothing to reduce noise)
    qvel = np.zeros_like(q_hand)
    raw_qvel = np.gradient(q_hand, DT, axis=0)
    for j in range(D):
        qvel[:, j] = gaussian_filter1d(raw_qvel[:, j], sigma=8.0)

    print("========== 数据统计 (env0) ==========")
    print(f"  τ:         σ={tq.std(axis=0).mean():.4f}")
    print(f"  qvel:      σ={qvel.std(axis=0).mean():.6f} (smoothed)")
    print(f"  K_d·qvel:  σ={(K_D*qvel).std(axis=0).mean():.4f}")
    print(f"  K_d·qvel / τ σ ratio: {(K_D*qvel).std(axis=0).mean() / max(tq.std(axis=0).mean(), 1e-8):.2f}")

    # ====== 1. 线性相关: τ vs FSR (基准) ======
    print(f"\n{'='*80}")
    print("1. 线性相关基准: FSR ↔ τ (env0, per-finger contact mask)")
    print(f"{'='*80}")

    # dτ (smoothed difference)
    dtq = np.diff(tq, axis=0)
    dtq_smooth = np.zeros_like(tq)
    dtq_smooth[0] = dtq[0]
    for j in range(D):
        dtq_smooth[1:, j] = gaussian_filter1d(dtq[:, j], sigma=5.0)

    for fname, grp in FINGER_GROUPS.items():
        fi = grp["fsr"]
        tj = grp["torque"]
        mask = (fsr[:, fi] > 0.1).any(axis=1)
        prox_fsr = fsr[mask][:, fi[:2]].mean(axis=1)

        print(f"\n{fname}:")
        print(f"  {'Joint':<14s} {'τ_raw':>8s} {'τ_smooth':>8s} {'Δτ_smooth':>8s} {'pos_err*':>8s}")
        for t_idx in tj:
            tau_val = tq[mask, t_idx]
            # Apply light smoothing to tau as well
            tau_smooth = gaussian_filter1d(tq[:, t_idx], sigma=3.0)[mask]

            valid = np.isfinite(prox_fsr) & np.isfinite(tau_val)
            r_raw = np.corrcoef(prox_fsr[valid], tau_val[valid])[0,1] if valid.sum()>10 else np.nan
            r_smooth = np.corrcoef(prox_fsr[valid], tau_smooth[valid])[0,1] if valid.sum()>10 else np.nan
            r_dtau = np.corrcoef(prox_fsr[valid], dtq_smooth[mask, t_idx][valid])[0,1] if valid.sum()>10 else np.nan

            # Position error from PD inversion
            pos_err = (tq[:, t_idx] + K_D * qvel[:, t_idx]) / K_P
            r_err = np.corrcoef(prox_fsr[valid], pos_err[mask][valid])[0,1] if valid.sum()>10 else np.nan

            print(f"  {TORQUE_NAMES_SHORT[t_idx]:<14s} {r_raw:8.3f} {r_smooth:8.3f} {r_dtau:8.3f} {r_err:8.3f}")

    # ====== 2. 诊断: qvel 对 correlation 的影响 ======
    print(f"\n{'='*80}")
    print("2. 诊断: q̇ 与 τ 和 FSR 的关系")
    print(f"{'='*80}")

    for fname, grp in FINGER_GROUPS.items():
        fi = grp["fsr"]
        tj_root = grp["torque"][0]
        mask = (fsr[:, fi] > 0.1).any(axis=1)

        # Split by qvel sign
        qvel_root = qvel[mask, tj_root]
        tau_root = tq[mask, tj_root]
        fsr_root = fsr[mask][:, fi[:2]].mean(axis=1)

        closing = qvel_root > 0.01   # joint closing (finger bending)
        opening = qvel_root < -0.01  # joint opening
        steady = np.abs(qvel_root) <= 0.01

        for label, cond in [("closing", closing), ("steady", steady), ("opening", opening)]:
            if cond.sum() > 20:
                r = np.corrcoef(fsr_root[cond], tau_root[cond])[0,1]
                print(f"  {fname:<8s} {label:<8s} (N={cond.sum():5d}): "
                      f"corr(FSR, τ)={r:+.3f}  "
                      f"τ_mean={tau_root[cond].mean():.3f}  "
                      f"qvel_mean={qvel_root[cond].mean():.4f}")

    # ====== 3. 尝试非线性映射 ======
    print(f"\n{'='*80}")
    print("3. 非线性映射: MLP(q, τ, action, dτ) → FSR (逐指, 3-fold CV)")
    print(f"{'='*80}")

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    for fname, grp in FINGER_GROUPS.items():
        fi = grp["fsr"]
        tj = grp["torque"]
        mask = (fsr[:, fi] > 0.1).any(axis=1)

        # Features: per-finger q, τ, action, dτ
        X_q = q_hand[mask][:, tj]
        X_tau = tq[mask][:, tj]
        X_action = action[mask][:, tj]
        X_dtau = np.diff(tq, axis=0, prepend=tq[0:1])[mask][:, tj]

        X = np.concatenate([X_q, X_tau, X_action, X_dtau], axis=1)
        y = fsr[mask][:, fi]  # all FSR channels for this finger

        # Normalize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Small MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.001,
            batch_size=256,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        )

        # Cross-validated R² for each FSR channel
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        r2_scores = np.zeros((3, len(fi)))  # [fold, fsr_channel]

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
            mlp.fit(X_scaled[train_idx], y_scaled[train_idx])
            y_pred = mlp.predict(X_scaled[test_idx])
            for ch in range(len(fi)):
                ss_res = ((y_scaled[test_idx, ch] - y_pred[:, ch])**2).sum()
                ss_tot = ((y_scaled[test_idx, ch] - y_scaled[test_idx, ch].mean())**2).sum()
                r2_scores[fold, ch] = 1 - ss_res / max(ss_tot, 1e-10)

        r2_mean = r2_scores.mean(axis=0)
        r2_std = r2_scores.std(axis=0)
        print(f"\n{fname} (features: q[{len(tj)}], τ[{len(tj)}], act[{len(tj)}], dτ[{len(tj)}]):")
        for ch_idx, fsr_idx in enumerate(fi):
            # Compare with linear baseline
            r_linear = np.corrcoef(
                fsr[mask][:, fsr_idx],
                tq[mask, tj[0]]
            )[0,1]
            print(f"  FSR[{fsr_idx}] {FSR_NAMES_SHORT[fsr_idx]:6s}: "
                  f"linear r²={r_linear**2:.4f}  "
                  f"MLP R²={r2_mean[ch_idx]:.4f}±{r2_std[ch_idx]:.4f}  "
                  f"gain={r2_mean[ch_idx]/max(r_linear**2, 1e-10):.1f}x")

    # ====== 4. 散点图: 最佳线性 vs 非线性 ======
    print(f"\n========== 4. 可视化 ==========")
    colors = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (fname, grp) in zip(axes.flat, FINGER_GROUPS.items()):
        fi = grp["fsr"]
        tj_root = grp["torque"][0]
        mask = (fsr[:, fi] > 0.1).any(axis=1)
        x = fsr[mask][:, fi[:2]].mean(axis=1)
        y = tq[mask, tj_root]
        valid = np.isfinite(x) & np.isfinite(y)
        n_plot = min(5000, valid.sum())
        idx = np.random.choice(np.where(valid)[0], n_plot, replace=False)
        ax.scatter(x[idx], y[idx], s=2, alpha=0.4, c=colors[list(FINGER_GROUPS.keys()).index(fname)])
        r = np.corrcoef(x[valid], y[valid])[0,1]
        ax.set_xlabel('Proximal FSR (N)')
        ax.set_ylabel(f'τ_root (Nm) [{TORQUE_NAMES_SHORT[tj_root]}]')
        ax.set_title(f'{fname}: τ ↔ FSR  r={r:.3f}  r²={r**2:.3f}')
        ax.grid(alpha=0.3)
    plt.suptitle("env0: FSR vs Root Joint Torque (contact only, linear correlation)", fontsize=13)
    plt.tight_layout()
    out_sc = OUT_DIR / "fsr_vs_torque_env0_scatter.png"
    fig.savefig(out_sc, dpi=150)
    plt.close(fig)
    print(f"[DONE] {out_sc}")

    # ====== 5. 时间序列叠加 ======
    contact_density = np.convolve(
        (fsr > 0.1).any(axis=1).astype(float),
        np.ones(200) / 200, mode='same'
    )
    best_start = np.argmax(contact_density[2000:-2000]) + 2000
    win = slice(best_start, best_start + 500)

    fig, axes = plt.subplots(4, 1, figsize=(22, 16), sharex=True)
    colors_list = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    fig.suptitle(f"env0: FSR vs Torque Time Series (steps {best_start}-{best_start+500})", fontsize=13)

    for ax_idx, (fname, grp) in enumerate(FINGER_GROUPS.items()):
        ax = axes[ax_idx]
        fi = grp["fsr"]
        tj_root = grp["torque"][0]
        prox_fsr = fsr[win][:, fi[:2]].mean(axis=1)
        root_tq = tq[win, tj_root]
        ax2 = ax.twinx()
        ax.plot(prox_fsr, color=colors_list[ax_idx], alpha=0.7, linewidth=1.5,
                label=f'{fname} prox FSR')
        ax2.plot(root_tq, color='gray', alpha=0.9, linewidth=1.2,
                 linestyle='--', label=f'{fname} τ_root')
        ax.set_ylabel('FSR (N)', fontsize=9)
        ax2.set_ylabel('τ (Nm)', fontsize=9)
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
        ax.set_title(fname, fontsize=11)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    out_ts = OUT_DIR / "fsr_torque_timeseries_overlay_env0.png"
    fig.savefig(out_ts, dpi=150)
    plt.close(fig)
    print(f"[DONE] {out_ts}")

    # ====== 6. 数据质量汇总 ======
    print(f"\n{'='*80}")
    print("SUMMARY: 力矩 → FSR 的可行性评估")
    print(f"{'='*80}")
    for fname, grp in FINGER_GROUPS.items():
        fi = grp["fsr"]
        tj_root = grp["torque"][0]
        mask = (fsr[:, fi] > 0.1).any(axis=1)
        prox_fsr = fsr[mask][:, fi[:2]].mean(axis=1)
        tau_root = tq[mask, tj_root]
        valid = np.isfinite(prox_fsr) & np.isfinite(tau_root)

        r = np.corrcoef(prox_fsr[valid], tau_root[valid])[0,1]
        # SNR: contact vs no-contact torque distributions
        tau_nc = tq[~mask, tj_root]
        tau_c = tq[mask, tj_root]
        snr = abs(tau_c.mean() - tau_nc.mean()) / max(tau_c.std(), 1e-8)

        # Sensitivity: Δτ / ΔFSR (linear slope)
        slope, intercept = np.polyfit(prox_fsr[valid], tau_root[valid], 1)

        print(f"  {fname:<8s}: r={r:+.3f}  r²={r**2:.3f}  SNR={snr:.2f}  "
              f"sensitivity={slope:.4f} Nm/N  "
              f"τ_baseline={intercept:.4f} Nm  "
              f"contact_ratio={mask.mean():.1%}")

    print("\n[DONE] All analysis complete.")


if __name__ == "__main__":
    main()
