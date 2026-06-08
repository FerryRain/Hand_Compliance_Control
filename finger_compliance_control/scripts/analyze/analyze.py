import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
import glob
import sys


def fit_torque_action_residual(
    torque: np.ndarray, action: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-joint linear de-trending: torque_i ~= alpha_i * action_i + beta_i.

    Returns:
        residual: [T, D]
        alpha: [D]
        beta: [D]
    """
    if torque.shape != action.shape:
        raise ValueError(
            f"torque/action shape mismatch: {torque.shape} vs {action.shape}"
        )

    a_mean = np.mean(action, axis=0)
    t_mean = np.mean(torque, axis=0)
    cov = np.mean((action - a_mean) * (torque - t_mean), axis=0)
    var_a = np.mean((action - a_mean) ** 2, axis=0)
    alpha = cov / (var_a + eps)
    beta = t_mean - alpha * a_mean
    pred = action * alpha[None, :] + beta[None, :]
    residual = torque - pred
    return residual, alpha.astype(np.float32), beta.astype(np.float32)


def point_biserial_corr(x: np.ndarray, y01: np.ndarray, eps: float = 1e-8) -> float:
    """Correlation between continuous x and binary y in {0,1}."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y01, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: x={x.shape}, y={y.shape}")
    x_std = np.std(x)
    p = np.mean(y)
    q = 1.0 - p
    if x_std < eps or p < eps or q < eps:
        return 0.0
    x1 = x[y > 0.5]
    x0 = x[y <= 0.5]
    return float((np.mean(x1) - np.mean(x0)) * np.sqrt(p * q) / (x_std + eps))


def as_binary_event_1d(event: np.ndarray, length: int, name: str) -> np.ndarray:
    """Convert event array to 1D int array of given length with values in {0,1}."""
    e = np.asarray(event)
    if e.ndim == 2 and e.shape[1] == 1:
        e = e[:, 0]
    elif e.ndim > 1:
        e = e.reshape(e.shape[0], -1)
        # Any activated channel counts as event at that timestep.
        e = (np.any(e > 0, axis=1)).astype(np.int32)
    e = e.reshape(-1)
    if e.shape[0] != length:
        raise ValueError(
            f"{name} length mismatch: expected {length}, got {e.shape[0]}"
        )
    return (e > 0).astype(np.int32)


def make_proxy_events_from_fsr(fsr: np.ndarray) -> dict[str, np.ndarray]:
    """Fallback event definitions from fsr when quality metrics are unavailable.

    Returns three binary arrays [T]:
    - contact_event: change in number of active contacts
    - slip_event: large fsr delta norm
    - instability_event: unstable contact count in short horizon
    """
    fsr = np.asarray(fsr, dtype=np.float32)
    t = fsr.shape[0]
    contact_mask = fsr > 0.1
    contact_count = np.sum(contact_mask, axis=1).astype(np.float32)

    # Contact event: contact count changes.
    contact_delta = np.abs(np.diff(contact_count, prepend=contact_count[:1]))
    contact_event = (contact_delta >= 1.0).astype(np.int32)

    # Slip proxy: strong change in tactile field.
    d_fsr = np.diff(fsr, axis=0, prepend=fsr[:1])
    fsr_delta_norm = np.linalg.norm(d_fsr, axis=1)
    slip_thr = np.quantile(fsr_delta_norm, 0.8)
    slip_event = (fsr_delta_norm >= slip_thr).astype(np.int32)

    # Instability proxy: high rolling std of contact count.
    w = 8
    csum = np.concatenate([[0.0], np.cumsum(contact_count)])
    csum2 = np.concatenate([[0.0], np.cumsum(contact_count**2)])
    mean = (csum[w:] - csum[:-w]) / w
    mean2 = (csum2[w:] - csum2[:-w]) / w
    rolling_std = np.sqrt(np.maximum(0.0, mean2 - mean**2))
    rolling_std = np.concatenate([np.zeros(w - 1, dtype=np.float32), rolling_std]).astype(
        np.float32
    )
    instab_thr = np.quantile(rolling_std, 0.8)
    instability_event = (rolling_std >= instab_thr).astype(np.int32)

    if contact_event.shape[0] != t:
        raise RuntimeError("proxy event length mismatch")

    return {
        "contact_event": contact_event,
        "slip_event": slip_event,
        "instability_event": instability_event,
    }


def compute_cei(fsr: np.ndarray, action: np.ndarray, top_k: int = 4, eps: float = 1e-6) -> np.ndarray:
    """Compliance Efficiency Index (CEI): top-k contact force per action effort."""
    k = min(top_k, fsr.shape[-1])
    topk = np.partition(fsr, -k, axis=-1)[:, -k:]
    contact_force = topk.mean(axis=-1)
    action_effort = np.mean(np.abs(action), axis=-1)
    return contact_force / (action_effort + eps)


def safe_corrcoef(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute correlation matrix without warnings for constant channels."""
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(x_centered, axis=0)
    valid = std > eps

    corr = np.zeros((x.shape[1], x.shape[1]), dtype=np.float32)
    if np.any(valid):
        x_valid = x_centered[:, valid] / std[valid]
        corr_valid = (x_valid.T @ x_valid) / max(1, x.shape[0] - 1)
        corr[np.ix_(valid, valid)] = corr_valid.astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    return corr


def load_h5_trajectory(h5_path: str, env_idx: int = 0) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        fsr = np.asarray(f["fsr"], dtype=np.float32)[:, env_idx]
        action = np.asarray(f["action"], dtype=np.float32)[:, env_idx]
        q = np.asarray(f["q"], dtype=np.float32)[:, env_idx]
        time = np.asarray(f["time"], dtype=np.float32)[:, env_idx]
        qfrc_actuator = np.asarray(f.get("qfrc_actuator", np.zeros((fsr.shape[0], 16))), dtype=np.float32)[:, env_idx]

    step = np.arange(fsr.shape[0], dtype=np.int32)
    cei = compute_cei(fsr, action)
    return {
        "step": step,
        "time": time,
        "fsr": fsr,
        "action": action,
        "q": q,
        "qfrc_actuator": qfrc_actuator,
        "cei": cei,
    }

def plot_compliance_analysis(h5_path: str, env_idx: int = 0) -> None:
    traj = load_h5_trajectory(h5_path, env_idx=env_idx)
    steps = traj["step"]
    fsr = traj["fsr"]
    action = traj["action"]
    q = traj["q"]
    cei = traj["cei"]
    
    # 定义手指映射（基于你的控制器配置）
    # FSR 映射: Palm(0-3), Index(4-6), Middle(7-9), Ring(10-12), Thumb(13-15)
    # Action 映射 (16维): Index(0,2,3), Middle(4,6,7), Ring(8,10,11), Thumb(12,14,15) 
    # 注：Action对应关节偏移，这里简化对应关系
    fingers = {
        "Index":  {"fsr": [4, 5, 6], "action": [0, 2, 3], "pos": [0, 2, 3]},
        "Middle": {"fsr": [7, 8, 9], "action": [4, 6, 7], "pos": [4, 6, 7]},
        "Ring":   {"fsr": [10, 11, 12], "action": [8, 10, 11], "pos": [8, 10, 11]},
        "Thumb":  {"fsr": [13, 14, 15], "action": [12, 14, 15], "pos": [12, 14, 15]}
    }

    # 1. 创建大图：展示每个手指的 动态关系
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i, (name, idxs) in enumerate(fingers.items()):
        ax = axes[i]
        ax2 = ax.twinx() # 右轴用于 Action/Pos
        
        # 提取数据
        fsr_sum = fsr[:, idxs["fsr"]].mean(axis=1)
        # 取该手指的主动作（比如中段关节的 Action）
        action_main = action[:, idxs["action"][1]]
        # q 是 22 维时，前 6 维是臂，后 16 维是手。
        q_hand = q[:, -16:] if q.shape[-1] >= 16 else q
        pos_main = q_hand[:, idxs["pos"][1]]
        
        # 绘制受力 (左轴 - 填充图)
        ax.fill_between(steps, fsr_sum, color="gray", alpha=0.3, label="Avg FSR Force")
        ax.set_ylabel('Force (N)', color='gray')
        
        # 绘制 Action 和 Position (右轴)
        ax2.plot(steps, action_main, label='Action (Delta)', color='blue', linewidth=1.5)
        ax2.plot(steps, (pos_main - pos_main[0]), label='Pos Offset', color='red', linestyle='--')
        ax2.set_ylabel('Action / Pos Delta', color='blue')

        ax.set_title(f"Finger: {name} - Force vs Control response", fontsize=14)
        if i == 0:
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

    fig_cei, ax_cei = plt.subplots(1, 1, figsize=(15, 4))
    ax_cei.plot(steps, cei, color="tab:green", linewidth=1.2)
    ax_cei.set_title(
        f"Compliance Efficiency Index (mean={float(np.mean(cei)):.3f})"
    )
    ax_cei.set_ylabel("CEI")
    ax_cei.set_xlabel("step")
    ax_cei.grid(True, alpha=0.3)

    stem = str(Path(h5_path).with_suffix("")) + f"_env{env_idx}"
    plt.suptitle(f"Trajectory Analysis: {Path(h5_path).name} (env {env_idx})", fontsize=16)
    fig.savefig(f"{stem}_analysis.png")
    fig_cei.savefig(f"{stem}_cei.png")
    print(f"Saved analysis plot to {stem}_analysis.png")
    print(f"Saved CEI plot to {stem}_cei.png")

    # 2. 相关性热力图 (Correlation Heatmap)
    # 使用所有环境数据计算 FSR 与 Action 的相关性
    plt.figure(figsize=(12, 10))
    with h5py.File(h5_path, "r") as f:
        fsr_all = np.asarray(f["fsr"], dtype=np.float32)      # [T, E, 16]
        action_all = np.asarray(f["action"], dtype=np.float32)  # [T, E, 16]

    # 展平为 [T*E, 16]
    fsr_flat = fsr_all.reshape(-1, fsr_all.shape[-1])
    action_flat = action_all.reshape(-1, action_all.shape[-1])

    fsr_single_env = fsr_all[:, env_idx, :].reshape(-1, fsr_all.shape[-1])
    action_single_env = action_all[:, env_idx, :].reshape(-1, action_all.shape[-1])

    x = np.concatenate([fsr_flat, action_flat], axis=1)
    # x = np.concatenate([fsr_single_env, action_single_env], axis=1)
    corr = safe_corrcoef(x)
    
    # 重点看 FSR(行) 和 Action(列) 的交叉区域
    fsr_action_corr = corr[:16, 16:32]
    sns.heatmap(fsr_action_corr, annot=False, cmap='RdBu_r', center=0)
    plt.title("Correlation: FSR vs Action")
    plt.xlabel("Actions")
    plt.ylabel("FSR Sensors")
    plt.savefig(f"{stem}_corr.png")
    print(f"Saved correlation plot to {stem}_corr.png")
    plt.close("all")


def plot_torque_analysis(h5_path: str, env_idx: int = 0) -> None:
    """Analyze joint torques: distribution, relationship with qpos and action."""
    traj = load_h5_trajectory(h5_path, env_idx=env_idx)
    steps = traj["step"]
    qfrc = traj["qfrc_actuator"]  # [T, 16]
    q = traj["q"]  # [T, 22]
    action = traj["action"]  # [T, 16]
    
    stem = str(Path(h5_path).with_suffix("")) + f"_torque_env{env_idx}"
    
    # 1. 关节力矩的分布
    fig1, axes1 = plt.subplots(4, 4, figsize=(16, 12))
    fig1.suptitle("Joint Torque Distribution (16 Fingers)")
    axes1 = axes1.flatten()
    
    torque_means = []
    torque_stds = []
    for i in range(16):
        ax = axes1[i]
        torque_i = qfrc[:, i]
        torque_means.append(np.mean(torque_i))
        torque_stds.append(np.std(torque_i))
        ax.hist(torque_i, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax.set_title(f"Joint {i}: μ={torque_means[-1]:.3f}, σ={torque_stds[-1]:.3f}")
        ax.set_xlabel("Torque (N·m)")
        ax.set_ylabel("Count")
    
    fig1.tight_layout()
    fig1.savefig(f"{stem}_distribution.png", dpi=100)
    print(f"Saved torque distribution to {stem}_distribution.png")
    
    # 2. 关节力矩 vs 时间序列对应关系（选 4 个代表关节）
    fig2, axes2 = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig2.suptitle(f"Torque vs Position & Action (representative joints)")
    
    selected_joints = [0, 5, 10, 15]  # Index, Middle, Ring, Thumb (主关节)
    for idx, joint_id in enumerate(selected_joints):
        ax = axes2[idx]
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # 力矩 (左轴)
        ax.plot(steps, qfrc[:, joint_id], label='Torque', color='red', linewidth=1.5)
        ax.set_ylabel('Torque (N·m)', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        
        # 关节位置 (中轴)
        q_hand = q[:, -16:] if q.shape[-1] >= 16 else q
        ax2.plot(steps, q_hand[:, joint_id], label='QPos', color='blue', linewidth=1, linestyle='--')
        ax2.set_ylabel('Position (rad)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # 动作 (右轴)
        ax3.plot(steps, action[:, joint_id], label='Action', color='green', linewidth=1, linestyle=':')
        ax3.set_ylabel('Action', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        ax.set_title(f"Joint {joint_id}: Torque ← → Position ← → Action")
        if idx == 0:
            ax.legend(loc='upper left')
            ax2.legend(loc='upper center')
            ax3.legend(loc='upper right')
    
    axes2[-1].set_xlabel('Step')
    fig2.tight_layout()
    fig2.savefig(f"{stem}_timeseries.png", dpi=100)
    print(f"Saved torque timeseries to {stem}_timeseries.png")
    
    # 3. 相关性分析：Torque vs QPos, Torque vs Action
    with h5py.File(h5_path, "r") as f:
        qfrc_all = np.asarray(f.get("qfrc_actuator", np.zeros((f["action"].shape[0], 16))), dtype=np.float32) # type: ignore
        q_all = np.asarray(f["q"], dtype=np.float32)
        action_all = np.asarray(f["action"], dtype=np.float32)
    
    # 展平
    qfrc_flat = qfrc_all.reshape(-1, qfrc_all.shape[-1])  # [T*E, 16]
    q_flat = q_all.reshape(-1, q_all.shape[-1])  # [T*E, 22]
    action_flat = action_all.reshape(-1, action_all.shape[-1])  # [T*E, 16]
    
    # 只取手部关节 (最后 16 维)
    q_hand_flat = q_flat[:, -16:]  # [T*E, 16]
    
    # 计算相关性矩阵
    torque_qpos_corr = safe_corrcoef(np.concatenate([qfrc_flat, q_hand_flat], axis=1))[:16, 16:32]
    torque_action_corr = safe_corrcoef(np.concatenate([qfrc_flat, action_flat], axis=1))[:16, 16:32]
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle("Correlation Analysis")
    
    sns.heatmap(torque_qpos_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Correlation'}, ax=ax3a)
    ax3a.set_title("Torque vs QPos (Hand Joints)")
    ax3a.set_xlabel("QPos Indices")
    ax3a.set_ylabel("Torque Indices")
    
    sns.heatmap(torque_action_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Correlation'}, ax=ax3b)
    ax3b.set_title("Torque vs Action")
    ax3b.set_xlabel("Action Indices")
    ax3b.set_ylabel("Torque Indices")
    
    fig3.tight_layout()
    fig3.savefig(f"{stem}_correlation.png", dpi=100)
    print(f"Saved torque correlation to {stem}_correlation.png")
    
    # 4. 统计摘要
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.axis('off')
    
    summary_text = f"""
    TORQUE ANALYSIS SUMMARY
    File: {Path(h5_path).name}
    Environment: {env_idx}
    
    DISTRIBUTION STATISTICS:
    Mean Torques (N·m):  {[f'{m:.4f}' for m in torque_means]}
    Std Torques (N·m):   {[f'{s:.4f}' for s in torque_stds]}
    
    TRAINABILITY ASSESSMENT:
    - Non-zero torque variance: {np.sum(np.array(torque_stds) > 1e-6)} / 16 joints
    - Avg signal-to-noise: {np.mean(np.array(torque_means) / (np.array(torque_stds) + 1e-8)):.3f}
    
    RELATIONSHIPS:
    - Avg |Torque-QPos correlation|: {np.mean(np.abs(torque_qpos_corr)):.3f}
    - Avg |Torque-Action correlation|: {np.mean(np.abs(torque_action_corr)):.3f}
    - Max |Torque-Action correlation|: {np.max(np.abs(torque_action_corr)):.3f}
    
    RECOMMENDATION:
    {"✓ Trainable: Strong torque signal and clear action correspondence" if np.mean(np.abs(torque_action_corr)) > 0.3 else "⚠ Check: Weak torque-action alignment"}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontfamily='monospace',
             verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig4.savefig(f"{stem}_summary.png", dpi=100, bbox_inches='tight')
    print(f"Saved torque summary to {stem}_summary.png")
    print(summary_text)


def plot_residual_event_analysis(h5_path: str, env_idx: int = 0) -> None:
    """Analyze epsilon_t vs contact/slip/instability events."""
    traj = load_h5_trajectory(h5_path, env_idx=env_idx)
    qfrc = traj["qfrc_actuator"]
    action = traj["action"]
    fsr = traj["fsr"]
    steps = traj["step"]

    residual, alpha, beta = fit_torque_action_residual(qfrc, action)
    eps_abs_mean = np.mean(np.abs(residual), axis=1)  # [T]
    eps_l2 = np.linalg.norm(residual, axis=1)  # [T]

    stem = str(Path(h5_path).with_suffix("")) + f"_residual_env{env_idx}"

    with h5py.File(h5_path, "r") as f:
        has_quality = all(
            k in f
            for k in ("full_contact", "contact_stability", "force_balance", "fsr_delta_norm")
        )
        if has_quality:
            full_contact = np.asarray(f["full_contact"], dtype=np.float32)[:, env_idx]
            contact_stability = np.asarray(f["contact_stability"], dtype=np.float32)[:, env_idx]
            fsr_delta_norm = np.asarray(f["fsr_delta_norm"], dtype=np.float32)[:, env_idx]
            # Convert metrics to events.
            contact_event = as_binary_event_1d(
                (full_contact > 0.5).astype(np.int32), len(steps), "contact_event"
            )
            slip_event = as_binary_event_1d(
                (fsr_delta_norm >= np.quantile(fsr_delta_norm, 0.8)).astype(np.int32),
                len(steps),
                "slip_event",
            )
            instability_event = as_binary_event_1d(
                (contact_stability <= np.quantile(contact_stability, 0.2)).astype(np.int32),
                len(steps),
                "instability_event",
            )
            events = {
                "contact_event": contact_event,
                "slip_event": slip_event,
                "instability_event": instability_event,
            }
            metric_source = "h5_metrics"
        else:
            raw_events = make_proxy_events_from_fsr(fsr)
            events = {
                k: as_binary_event_1d(v, len(steps), k) for k, v in raw_events.items()
            }
            metric_source = "fsr_proxy"

    # 1) Residual time series with event markers.
    fig1, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(steps, eps_abs_mean, color="tab:purple", linewidth=1.2)
    axes[0].set_ylabel("mean(|epsilon|)")
    axes[0].set_title("Residual torque magnitude")
    axes[0].grid(True, alpha=0.25)

    for i, (name, event) in enumerate(events.items(), start=1):
        axes[i].plot(steps, eps_l2, color="tab:gray", alpha=0.8, linewidth=1.0)
        ev_idx = np.flatnonzero(event > 0)
        axes[i].scatter(
            ev_idx,
            eps_l2[ev_idx],
            s=6,
            alpha=0.7,
            color="tab:red",
            label=name,
        )
        axes[i].set_ylabel("||epsilon||_2")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.2)
    axes[-1].set_xlabel("step")
    fig1.suptitle(f"Residual vs Events ({metric_source})")
    fig1.tight_layout()
    fig1.savefig(f"{stem}_timeseries_events.png", dpi=120)
    print(f"Saved residual-event timeseries to {stem}_timeseries_events.png")

    # 2) Distribution shift for each event (event=1 vs event=0).
    event_names = list(events.keys())
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    stats_rows = []
    for i, name in enumerate(event_names):
        event = events[name].astype(np.int32)
        x1 = eps_abs_mean[event == 1]
        x0 = eps_abs_mean[event == 0]
        m1 = float(np.mean(x1)) if x1.size > 0 else 0.0
        m0 = float(np.mean(x0)) if x0.size > 0 else 0.0
        s1 = float(np.std(x1)) if x1.size > 0 else 0.0
        s0 = float(np.std(x0)) if x0.size > 0 else 0.0
        pooled = np.sqrt((s1**2 + s0**2) / 2.0 + 1e-8)
        cohen_d = (m1 - m0) / pooled
        r_pb = point_biserial_corr(eps_abs_mean, event)

        axes2[i].boxplot(
            [x0, x1],
            tick_labels=["event=0", "event=1"],
            showfliers=False,
        )
        axes2[i].set_title(f"{name}\nΔmean={m1-m0:.3f}, d={cohen_d:.3f}, r={r_pb:.3f}")
        axes2[i].set_ylabel("mean(|epsilon|)")
        axes2[i].grid(True, axis="y", alpha=0.2)

        stats_rows.append((name, m0, m1, m1 - m0, cohen_d, r_pb, int(np.sum(event))))

    fig2.suptitle("Residual shift under events")
    fig2.tight_layout()
    fig2.savefig(f"{stem}_event_boxplots.png", dpi=120)
    print(f"Saved residual-event boxplots to {stem}_event_boxplots.png")

    # 3) Lag correlation: corr(epsilon_t, event_{t+lag})
    lags = np.arange(-20, 21)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4))
    for name, event in events.items():
        cvals = []
        event_f = event.astype(np.float32)
        for lag in lags:
            if lag < 0:
                x = eps_abs_mean[-lag:]
                y = event_f[: len(event_f) + lag]
            elif lag > 0:
                x = eps_abs_mean[: len(eps_abs_mean) - lag]
                y = event_f[lag:]
            else:
                x = eps_abs_mean
                y = event_f
            if x.size < 10:
                cvals.append(0.0)
            else:
                cvals.append(point_biserial_corr(x, y))
        ax3.plot(lags, cvals, label=name)
    ax3.axvline(0, color="k", linestyle="--", linewidth=1)
    ax3.set_xlabel("lag (event at t+lag)")
    ax3.set_ylabel("point-biserial corr")
    ax3.set_title("Lag relation: residual to event")
    ax3.grid(True, alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(f"{stem}_lag_corr.png", dpi=120)
    print(f"Saved residual-event lag correlation to {stem}_lag_corr.png")

    # 4) Text summary.
    summary_lines = [
        "RESIDUAL-EVENT ANALYSIS SUMMARY",
        f"File: {Path(h5_path).name}",
        f"Environment: {env_idx}",
        f"Metric source: {metric_source}",
        "",
        "Linear de-trending per joint: tau_i = alpha_i * action_i + beta_i",
        f"alpha (first 8): {[float(v) for v in alpha[:8]]}",
        f"beta  (first 8): {[float(v) for v in beta[:8]]}",
        f"Residual mean(|epsilon|): {float(np.mean(eps_abs_mean)):.4f}",
        f"Residual std(|epsilon|): {float(np.std(eps_abs_mean)):.4f}",
        "",
        "Event shift stats: name, mean(event=0), mean(event=1), delta, cohen_d, r_pb, event_count",
    ]
    for row in stats_rows:
        summary_lines.append(
            f"- {row[0]}: {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f}, {row[4]:.4f}, {row[5]:.4f}, {row[6]}"
        )
    summary_text = "\n".join(summary_lines)

    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ax4.axis("off")
    ax4.text(
        0.02,
        0.98,
        summary_text,
        transform=ax4.transAxes,
        va="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8),
    )
    fig4.tight_layout()
    fig4.savefig(f"{stem}_summary.png", dpi=120)
    print(f"Saved residual-event summary to {stem}_summary.png")
    print(summary_text)

if __name__ == "__main__":
    # 支持命令行参数指定 h5 文件
    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
        env_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if not Path(h5_file).exists():
            print(f"[ERROR] File not found: {h5_file}")
            sys.exit(1)
    else:
        # 默认查找最新的采集文件
        h5_files = sorted(
            glob.glob("./finger_compliance_control/data/headless/headless_train_20260414_175152.h5"),
            key=lambda p: Path(p).stat().st_mtime,
        )
        h5_files = [p for p in h5_files if not p.endswith("_inverted.h5")]
        if h5_files:
            h5_file = h5_files[-1]
            env_idx = 0
        else:
            print("No H5 found. Usage: python analyze.py <h5_file> [env_idx]")
            sys.exit(1)
    
    print(f"[INFO] Analyzing: {h5_file} (env {env_idx})")
    print("\n--- Compliance Analysis ---")
    plot_compliance_analysis(h5_file, env_idx=env_idx)
    print("\n--- Torque Analysis ---")
    plot_torque_analysis(h5_file, env_idx=env_idx)
    print("\n--- Residual vs Event Analysis ---")
    plot_residual_event_analysis(h5_file, env_idx=env_idx)
    print("\n[SUCCESS] All analyses completed!")