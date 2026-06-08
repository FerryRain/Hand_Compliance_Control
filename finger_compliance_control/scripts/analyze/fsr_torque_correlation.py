"""
验证 FSR — residual_Torque / FSR — d_ResidualTorque 的相关性。

关键修正:
  1. qfrc_actuator 包含位置控制力矩 + 外部接触力矩, 必须先用 action 回归去除
     位置控制分量, 残差 ≈ 外部接触力矩。
  2. 只分析有接触的时间步 (FSR > threshold), 避免大量零值淹没信号。
  3. 同时算 Pearson r 和 Spearman ρ (后者对非线性单调关系更鲁棒)。

用法:
    python finger_compliance_control/scripts/analyze/fsr_torque_correlation.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# --------------- config ---------------
H5_PATH = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
    "headless_train_20260414_175152.h5"
)
OUT_DIR = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
)
N_STEPS = 60_000
N_ENVS = 4
SKIP = 1
FSR_THRESHOLD = 0.1

FSR_NAMES = [
    "palm_0", "palm_1", "palm_2", "palm_3",
    "idx_prox0", "idx_prox1", "idx_dist",
    "mid_prox0", "mid_prox1", "mid_dist",
    "rng_prox0", "rng_prox1", "rng_dist",
    "thb_prox0", "thb_prox1", "thb_dist",
]

TORQUE_NAMES = [
    "j6_idx_root", "j7_idx_abd",
    "j8_idx_mid",  "j9_idx_dist",
    "j10_mid_root","j11_mid_abd",
    "j12_mid_mid", "j13_mid_dist",
    "j14_rng_root","j15_rng_abd",
    "j16_rng_mid", "j17_rng_dist",
    "j18_thb_root","j19_thb_abd",
    "j20_thb_mid", "j21_thb_dist",
]

FINGER_GROUPS = {
    "Index":  {"fsr": [4, 5, 6],    "torque": [0, 2, 3]},
    "Middle": {"fsr": [7, 8, 9],    "torque": [4, 6, 7]},
    "Ring":   {"fsr": [10, 11, 12], "torque": [8, 10, 11]},
    "Thumb":  {"fsr": [13, 14, 15], "torque": [12, 14, 15]},
    "Palm":   {"fsr": [0, 1, 2, 3], "torque": []},
}


def fit_torque_action_residual(torque, action, eps=1e-8):
    """Per-joint linear de-trend: torque_j ≈ alpha_j * action_j + beta_j.
    Returns residual [N, D] which approximates external contact torque."""
    torque = np.asarray(torque, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    T, D = torque.shape
    if torque.shape != action.shape:
        raise ValueError(f"shape mismatch: {torque.shape} vs {action.shape}")
    a_mean = action.mean(axis=0)
    t_mean = torque.mean(axis=0)
    cov = ((action - a_mean) * (torque - t_mean)).mean(axis=0)
    var_a = ((action - a_mean) ** 2).mean(axis=0) + eps
    alpha = cov / var_a
    beta = t_mean - alpha * a_mean
    pred = action * alpha[None, :] + beta[None, :]
    residual = torque - pred
    return residual, alpha, beta


def load_and_preprocess(h5_path, n_steps, n_envs):
    """Load fsr, torque, action; compute residual torque; flatten to numpy."""
    with h5py.File(h5_path, "r") as f:
        total = f["fsr"].shape[0]
        idx = np.linspace(0, total - 1, min(total, n_steps), dtype=int)
        print(f"[LOAD] {len(idx)}/{total} steps x {n_envs} envs")

        # Explicit numpy conversion
        fsr = np.asarray(f["fsr"][idx, :n_envs, :], dtype=np.float64)
        torque = np.asarray(f["qfrc_actuator"][idx, :n_envs, :], dtype=np.float64)
        action = np.asarray(f["action"][idx, :n_envs, :], dtype=np.float64)

    T, E, D = fsr.shape
    print(f"[DATA] fsr [{fsr.min():.2f}, {fsr.max():.2f}]  "
          f"torque [{torque.min():.2f}, {torque.max():.2f}]  "
          f"action [{action.min():.2f}, {action.max():.2f}]")

    # Flat: [T*E, D]
    fsr_f = np.ascontiguousarray(fsr.reshape(-1, D))
    torque_f = np.ascontiguousarray(torque.reshape(-1, D))
    action_f = np.ascontiguousarray(action.reshape(-1, D))

    # Residual torque per joint
    res_f, alpha, beta = fit_torque_action_residual(torque_f, action_f)
    print(f"[RESIDUAL] alpha range [{alpha.min():.3f}, {alpha.max():.3f}]  "
          f"beta range [{beta.min():.3f}, {beta.max():.3f}]")
    raw_std = torque_f.std(axis=0).mean()
    res_std = res_f.std(axis=0).mean()
    print(f"[RESIDUAL] raw torque std={raw_std:.3f}  "
          f"residual std={res_std:.3f}  "
          f"(reduction={1 - res_std / max(raw_std, 1e-8):.1%})")

    # d_residual
    d_res = np.diff(res_f, axis=0)
    fsr_for_d = fsr_f[1:, :]

    return fsr_f, res_f, d_res, fsr_for_d


def contact_mask(fsr, threshold):
    """Return boolean mask [N] where any FSR channel > threshold."""
    return (fsr > threshold).any(axis=1)


def compute_corr_matrix(x, y):
    """Return Pearson r and Spearman ρ [x_dim, y_dim] matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Nx, Dx = x.shape
    Ny, Dy = y.shape
    pearson = np.full((Dx, Dy), np.nan, dtype=np.float32)
    spearman = np.full((Dx, Dy), np.nan, dtype=np.float32)
    for i in range(Dx):
        for j in range(Dy):
            xi = x[:, i]
            yi = y[:, j]
            valid = np.isfinite(xi) & np.isfinite(yi)
            if valid.sum() < 10:
                continue
            pearson[i, j] = np.corrcoef(xi[valid], yi[valid])[0, 1]
            spearman[i, j] = stats.spearmanr(xi[valid], yi[valid])[0]
    return pearson, spearman


def plot_heatmap(corr, title, out_path, xlabels, ylabels, cmap="RdBu_r"):
    """Plot and save a correlation heatmap."""
    Dx, Dy = corr.shape
    fig, ax = plt.subplots(figsize=(max(16, Dy * 1.3), max(9, Dx * 0.6)))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(Dy))
    ax.set_xticklabels(xlabels, rotation=75, ha="right", fontsize=6.5)
    ax.set_yticks(range(Dx))
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_title(title, fontsize=12, weight="bold")
    for i in range(Dx):
        for j in range(Dy):
            v = corr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if abs(v) < 0.65 else "white")
    plt.colorbar(im, ax=ax, shrink=0.82)
    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[HEAT] {out_path}")


def plot_scatter_grid(fsr_flat, torque_flat, fsr_ids, tq_ids, finger_name,
                      suffix, out_dir, color="steelblue", max_pts=3000):
    """Scatter grid: rows=FSR channels, cols=torque channels."""
    n_fsr, n_tq = len(fsr_ids), len(tq_ids)
    fig, axes = plt.subplots(n_fsr, n_tq, figsize=(3 * n_tq, 3 * n_fsr),
                              squeeze=False)
    title = f"{finger_name} — {suffix} (N={fsr_flat.shape[0]})"
    fig.suptitle(title, fontsize=12, weight="bold")

    N = fsr_flat.shape[0]
    mask = np.random.choice(N, min(max_pts, N), replace=False)
    for row, fi in enumerate(fsr_ids):
        for col, tj in enumerate(tq_ids):
            ax = axes[row][col]
            x = fsr_flat[mask, fi]
            y = torque_flat[mask, tj]
            ax.scatter(x, y, s=1, alpha=0.35, c=color, edgecolors="none")
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 10:
                r = np.corrcoef(x[valid], y[valid])[0, 1]
                rho, _ = stats.spearmanr(x[valid], y[valid])
            else:
                r, rho = np.nan, np.nan
            ax.set_xlabel(FSR_NAMES[fi], fontsize=7)
            ylabel = TORQUE_NAMES[tj] if "Residual" in suffix else "d" + TORQUE_NAMES[tj]
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_title(f"r={r:.3f} ρ={rho:.3f}", fontsize=8)
            ax.tick_params(labelsize=5)
    plt.tight_layout(pad=0.8)
    fname = (f"fsr_{suffix.lower().replace(' ','_').replace('(','').replace(')','')}"
             f"_scatter_{finger_name.lower()}.png")
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)
    print(f"[SCATTER] {out_dir / fname}")


def print_top_corrs(pearson, spearman, xnames, ynames, label, threshold):
    """Print significant correlation pairs."""
    print(f"\n{'='*80}")
    print(f"{label} — |Pearson r| > {threshold}")
    print(f"{'='*80}")
    Dx, Dy = pearson.shape
    pairs = []
    for i in range(Dx):
        for j in range(Dy):
            if not np.isnan(pearson[i, j]) and abs(pearson[i, j]) > threshold:
                pairs.append((abs(pearson[i, j]), i, j, pearson[i, j], spearman[i, j]))
    pairs.sort(reverse=True)
    if not pairs:
        print("  (none)")
    for _, i, j, r, rho in pairs[:30]:
        print(f"  {xnames[i]:<16s} ↔ {ynames[j]:<16s}  "
              f"Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}")


def main():
    # ---- 1. Load & residual decomposition ----
    fsr_f, res_f, d_res_f, fsr_for_d = load_and_preprocess(H5_PATH, N_STEPS, N_ENVS)

    # ---- 2. Contact mask ----
    mask_c = contact_mask(fsr_f, FSR_THRESHOLD)
    mask_d = contact_mask(fsr_for_d, FSR_THRESHOLD)
    print(f"\n[CONTACT] full: contact ratio={mask_c.mean():.2%}  "
          f"dTorque: contact ratio={mask_d.mean():.2%}")

    fsr_c = np.ascontiguousarray(fsr_f[mask_c])
    res_c = np.ascontiguousarray(res_f[mask_c])
    fsr_dc = np.ascontiguousarray(fsr_for_d[mask_d])
    d_res_c = np.ascontiguousarray(d_res_f[mask_d])
    print(f"[CONTACT] shapes: fsr_c={fsr_c.shape}, res_c={res_c.shape}")

    # ---- 3. Heatmaps: ALL vs contact-only ----
    for label, f_arr, t_arr, out_name in [
        ("FSR vs Residual Torque (ALL timesteps)",    fsr_f,     res_f,    "fsr_restorque_all"),
        ("FSR vs Residual Torque (contact only)",      fsr_c,     res_c,    "fsr_restorque_contact"),
        ("FSR vs d(Residual Torque) (ALL)",            fsr_for_d, d_res_f,  "fsr_drestorque_all"),
        ("FSR vs d(Residual Torque) (contact only)",   fsr_dc,    d_res_c,  "fsr_drestorque_contact"),
    ]:
        pearson, spearman = compute_corr_matrix(f_arr, t_arr)
        plot_heatmap(pearson, f"Pearson r — {label}", OUT_DIR / f"{out_name}_pearson.png",
                     TORQUE_NAMES, FSR_NAMES)
        plot_heatmap(spearman, f"Spearman ρ — {label}", OUT_DIR / f"{out_name}_spearman.png",
                     TORQUE_NAMES, FSR_NAMES, cmap="RdYlGn")
        print_top_corrs(pearson, spearman, FSR_NAMES, TORQUE_NAMES, label, 0.25)

    # ---- 4. Per-finger scatter grids (contact only) ----
    for finger_name, grp in FINGER_GROUPS.items():
        fsr_ids = grp["fsr"]
        tq_ids = grp["torque"] or list(range(16))
        plot_scatter_grid(fsr_c, res_c, fsr_ids, tq_ids, finger_name,
                          "Residual Torque (contact only)", OUT_DIR, "steelblue")
        plot_scatter_grid(fsr_dc, d_res_c, fsr_ids, tq_ids, finger_name,
                          "d(Residual Torque) (contact only)", OUT_DIR, "darkorange")

    # ---- 5. Summary: per-finger key pairs ----
    print("\n" + "=" * 80)
    print("Per-finger strongest FSR↔ResidualTorque pair (contact only)")
    print("=" * 80)
    for finger_name, grp in FINGER_GROUPS.items():
        if not grp["torque"]:
            continue
        fsr_ids = grp["fsr"]
        tq_ids = grp["torque"]
        pearson, spearman = compute_corr_matrix(fsr_c[:, fsr_ids], res_c[:, tq_ids])
        best = np.unravel_index(np.nanargmax(np.abs(pearson)), pearson.shape)
        fi, tj = fsr_ids[best[0]], tq_ids[best[1]]
        print(f"  {finger_name:<8s}: FSR[{fi}] {FSR_NAMES[fi]:<14s} ↔ "
              f"ResTq[{tj}] {TORQUE_NAMES[tj]:<16s}  "
              f"Pearson r={pearson[best]:+.3f}  Spearman ρ={spearman[best]:+.3f}")

    # ---- 6. Summary: proximal FSR mean ↔ root joint residual torque ----
    print("\n" + "=" * 80)
    print("Proximal FSR (mean) ↔ Root Joint Residual Torque (contact only)")
    print("=" * 80)
    for finger_name, grp in FINGER_GROUPS.items():
        if not grp["torque"]:
            continue
        prox_fsr = fsr_c[:, grp["fsr"][:2]].mean(axis=1)
        root_tq = res_c[:, grp["torque"][0]]
        valid = np.isfinite(prox_fsr) & np.isfinite(root_tq)
        if valid.sum() > 10:
            r = np.corrcoef(prox_fsr[valid], root_tq[valid])[0, 1]
            rho, _ = stats.spearmanr(prox_fsr[valid], root_tq[valid])
        else:
            r, rho = np.nan, np.nan
        print(f"  {finger_name:<8s}: mean(prox_fsr) ↔ root_torque  "
              f"Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}")

    print("\n[DONE] All analysis complete.")


if __name__ == "__main__":
    # Run the main analysis function
    main()
