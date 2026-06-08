"""
深入分析: 为什么 FSR-Torque 相关性偏低？

排查方向:
  1. env0 单独分析 (排除跨环境混合)
  2. 逐指接触掩码 (排除非接触指干扰)
  3. 时间对齐检查 (是否有相位滞后)
  4. 非线性关系检查 (PD控制可能不是线性action→torque)
  5. 时间序列叠加可视化
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

H5_PATH = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
    "headless_train_20260414_175152.h5"
)
OUT_DIR = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
)
N_STEPS = 20000   # 分析前 2 万步
ENV_IDX = 0       # 只分析 env0

FSR_NAMES = [
    "p_0","p_1","p_2","p_3",
    "i_p0","i_p1","i_d",
    "m_p0","m_p1","m_d",
    "r_p0","r_p1","r_d",
    "t_p0","t_p1","t_d",
]
TORQUE_NAMES = [
    "j6_iR","j7_iA","j8_iM","j9_iD",
    "j10_mR","j11_mA","j12_mM","j13_mD",
    "j14_rR","j15_rA","j16_rM","j17_rD",
    "j18_tR","j19_tA","j20_tM","j21_tD",
]

FINGER_GROUPS = {
    "Index":  {"fsr": [4,5,6],    "torque": [0,2,3]},
    "Middle": {"fsr": [7,8,9],    "torque": [4,6,7]},
    "Ring":   {"fsr": [10,11,12], "torque": [8,10,11]},
    "Thumb":  {"fsr": [13,14,15], "torque": [12,14,15]},
}


def load_env0(h5_path, n_steps, env_idx=0):
    """Load single environment data."""
    with h5py.File(h5_path, "r") as f:
        total = f["fsr"].shape[0]
        n_load = min(total, n_steps)
        print(f"[LOAD] steps 0:{n_load}, env={env_idx}")

        fsr = np.asarray(f["fsr"][:n_load, env_idx, :], dtype=np.float64)
        torque = np.asarray(f["qfrc_actuator"][:n_load, env_idx, :], dtype=np.float64)
        action = np.asarray(f["action"][:n_load, env_idx, :], dtype=np.float64)
        q = np.asarray(f["q"][:n_load, env_idx, :], dtype=np.float64)
    return fsr, torque, action, q


def fit_torque_residual(torque, action, eps=1e-8):
    """Simple linear de-trend per joint."""
    torque = np.asarray(torque, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    a_mean = action.mean(axis=0)
    t_mean = torque.mean(axis=0)
    cov = ((action - a_mean) * (torque - t_mean)).mean(axis=0)
    var_a = ((action - a_mean) ** 2).mean(axis=0) + eps
    alpha = cov / var_a
    beta = t_mean - alpha * a_mean
    pred = action * alpha[None, :] + beta[None, :]
    residual = torque - pred
    return residual, alpha, beta


def main():
    fsr, torque, action, q = load_env0(H5_PATH, N_STEPS, ENV_IDX)
    T, D = fsr.shape

    # ---- 1. 基础统计 ----
    print("\n========== 基础统计 (env0) ==========")
    contact_per_fsr = (fsr > 0.1).mean(axis=0)
    print("FSR contact ratio (>0.1N) per channel:")
    for i in range(16):
        print(f"  FSR[{i:02d}] {FSR_NAMES[i]:6s}: {contact_per_fsr[i]:.1%}  mean={fsr[:,i].mean():.3f}  max={fsr[:,i].max():.2f}")

    # ---- 2. 残差分解 ----
    res, alpha, beta = fit_torque_residual(torque, action)
    raw_std = torque.std(axis=0)
    res_std = res.std(axis=0)
    print(f"\n[RESIDUAL] torque std={raw_std.mean():.3f}, residual std={res_std.mean():.3f}, "
          f"reduction per joint:")
    for j in range(16):
        print(f"  TQ[{j:02d}] {TORQUE_NAMES[j]:8s}: raw_std={raw_std[j]:.3f} → res_std={res_std[j]:.3f} "
              f"({1-res_std[j]/max(raw_std[j],1e-8):.0%})  alpha={alpha[j]:.3f}")

    d_res = np.diff(res, axis=0)
    fsr_d = fsr[1:, :]

    # ---- 3. 逐指相关 (只用该指自己的FSR>0.1来筛选) ----
    print("\n========== 逐指相关 (per-finger contact mask) ==========")
    for fname, grp in FINGER_GROUPS.items():
        fi = grp["fsr"]
        tj = grp["torque"]

        # Per-finger contact: any of this finger's FSRs > 0.1
        finger_contact = (fsr[:, fi] > 0.1).any(axis=1)
        finger_contact_d = (fsr_d[:, fi] > 0.1).any(axis=1)
        ratio = finger_contact.mean()

        print(f"\n--- {fname} (contact ratio={ratio:.1%}) ---")
        for f_idx in fi:
            for t_idx in tj:
                # Contact-only
                mask = finger_contact
                x = fsr[mask, f_idx]
                y = res[mask, t_idx]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() > 20:
                    r = np.corrcoef(x[valid], y[valid])[0,1]
                    rho, _ = stats.spearmanr(x[valid], y[valid])
                    print(f"  FSR[{f_idx}] {FSR_NAMES[f_idx]:6s} ↔ TQ[{t_idx}] {TORQUE_NAMES[t_idx]:8s}  "
                          f"Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}  N={valid.sum()}")

                # dTorque
                mask_d = finger_contact_d
                xd = fsr_d[mask_d, f_idx]
                yd = d_res[mask_d, t_idx]
                valid_d = np.isfinite(xd) & np.isfinite(yd)
                if valid_d.sum() > 20:
                    rd = np.corrcoef(xd[valid_d], yd[valid_d])[0,1]
                    print(f"  FSR[{f_idx}] {FSR_NAMES[f_idx]:6s} ↔ dTQ[{t_idx}] {TORQUE_NAMES[t_idx]:8s}  "
                          f"Pearson r={rd:+.3f}  N={valid_d.sum()}  [dTorque]")

    # ---- 4. 时间对齐: 滞后相关 ----
    print("\n========== 滞后相关 (cross-correlation lag analysis) ==========")
    # 对每根手指的 proximal FSR mean ↔ root torque
    max_lag = 20
    for fname, grp in FINGER_GROUPS.items():
        fi = grp["fsr"]
        tj_root = grp["torque"][0]
        prox_fsr = fsr[:, fi[:2]].mean(axis=1)  # proximal FSR mean
        root_tq = res[:, tj_root]

        # Cross-correlation at different lags
        contact_mask_finger = (fsr[:, fi] > 0.1).any(axis=1)
        prox_fsr_c = prox_fsr[contact_mask_finger]
        root_tq_c = root_tq[contact_mask_finger]

        print(f"\n{fname}: proximal FSR mean ↔ root residual torque")
        best_r, best_lag = 0, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x, y = prox_fsr_c[-lag:], root_tq_c[:lag]
            elif lag > 0:
                x, y = prox_fsr_c[:-lag], root_tq_c[lag:]
            else:
                x, y = prox_fsr_c, root_tq_c
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 20:
                continue
            r = np.corrcoef(x[valid], y[valid])[0, 1]
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        print(f"  Best lag: {best_lag:+d} steps → Pearson r={best_r:+.3f}")
        print(f"  (positive lag = torque lags behind FSR)")

    # ---- 5. 时间序列叠加可视化 (代表性片段) ----
    print("\n========== 绘制时间序列叠加图 ==========")
    # 找一个接触丰富的时间窗
    contact_density = np.convolve(
        (fsr > 0.1).any(axis=1).astype(float),
        np.ones(200) / 200, mode='same'
    )
    best_start = np.argmax(contact_density[1000:-1000]) + 1000
    window = slice(best_start, best_start + 600)

    fig, axes = plt.subplots(4, 1, figsize=(20, 18), sharex=True)
    fig.suptitle(f"env0: FSR vs Residual Torque — steps {best_start}-{best_start+600}", fontsize=14, weight="bold")

    colors_fsr = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    colors_tq = ['#0D47A1', '#1B5E20', '#E65100', '#880E4F']

    for ax_idx, (fname, grp) in enumerate(FINGER_GROUPS.items()):
        ax = axes[ax_idx]
        fi = grp["fsr"]
        tj = grp["torque"]
        prox_fsr = fsr[window, fi[:2]].mean(axis=1)
        dist_fsr = fsr[window, fi[2]]
        root_tq = res[window, tj[0]]
        mid_tq = res[window, tj[1]] if len(tj) > 1 else res[window, tj[0]]

        ax2 = ax.twinx()
        ax.plot(prox_fsr, color=colors_fsr[ax_idx], alpha=0.7, linewidth=1.2,
                label=f'{fname} prox FSR mean')
        ax.plot(dist_fsr, color=colors_fsr[ax_idx], alpha=0.3, linewidth=0.8,
                linestyle='--', label=f'{fname} dist FSR')
        ax.set_ylabel('FSR (N)', fontsize=9)
        ax.legend(loc='upper left', fontsize=7)

        ax2.plot(root_tq, color=colors_tq[ax_idx], alpha=0.9, linewidth=1.5,
                 label=f'{fname} root res torque')
        ax2.plot(mid_tq, color=colors_tq[ax_idx], alpha=0.5, linewidth=1.0,
                 linestyle=':', label=f'{fname} mid res torque')
        ax2.set_ylabel('Residual Torque (Nm)', fontsize=9)
        ax2.legend(loc='upper right', fontsize=7)

        # Also plot raw torque for comparison
        # ax2.plot(torque[window, tj[0]], color='gray', alpha=0.3, linewidth=0.5, label='raw torque')

        ax.set_title(fname, fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Step', fontsize=10)
    plt.tight_layout()
    out_ts = OUT_DIR / "fsr_torque_timeseries_env0.png"
    fig.savefig(out_ts, dpi=150)
    plt.close(fig)
    print(f"[DONE] {out_ts}")

    # ---- 6. 诊断：action与torque的非线性关系 ----
    print("\n========== 诊断: action → torque 的非线性程度 ==========")
    for fname, grp in FINGER_GROUPS.items():
        tj_root = grp["torque"][0]
        # 只用接触时的数据
        mask = (fsr[:, grp["fsr"]] > 0.1).any(axis=1)
        a = action[mask, tj_root]
        t = torque[mask, tj_root]

        # 线性拟合
        coeffs = np.polyfit(a, t, 1)
        pred_linear = np.polyval(coeffs, a)
        r2_linear = 1 - np.var(t - pred_linear) / np.var(t)

        # 二次拟合
        coeffs2 = np.polyfit(a, t, 2)
        pred_quad = np.polyval(coeffs2, a)
        r2_quad = 1 - np.var(t - pred_quad) / np.var(t)

        print(f"  {fname:<8s} tj_root: linear R²={r2_linear:.4f}  quadratic R²={r2_quad:.4f}  "
              f"Δ={(r2_quad-r2_linear)*100:+.1f}%")

    # ---- 7. 诊断：无接触时residual torque的baseline noise ----
    print("\n========== 诊断: 无接触 vs 有接触时 residual torque 分布 ==========")
    for fname, grp in FINGER_GROUPS.items():
        tj_root = grp["torque"][0]
        has_contact = (fsr[:, grp["fsr"]] > 0.1).any(axis=1)
        no_contact = ~has_contact

        res_nc = res[no_contact, tj_root]
        res_c = res[has_contact, tj_root]
        if len(res_nc) > 10 and len(res_c) > 10:
            print(f"  {fname:<8s} tj_root: no_contact μ={res_nc.mean():.4f} σ={res_nc.std():.4f}  "
                  f"contact μ={res_c.mean():.4f} σ={res_c.std():.4f}  "
                  f"Δμ={res_c.mean()-res_nc.mean():+.4f}  SNR={abs(res_c.mean()-res_nc.mean())/max(res_nc.std(),1e-8):.2f}")

    print("\n[DONE] All diagnostics complete.")


if __name__ == "__main__":
    main()
