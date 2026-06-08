import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
import glob


def compute_cei(fsr: np.ndarray, action: np.ndarray, top_k: int = 4, eps: float = 1e-6) -> np.ndarray:
    k = min(top_k, fsr.shape[-1])
    topk = np.partition(fsr, -k, axis=-1)[:, -k:]
    contact_force = topk.mean(axis=-1)
    action_effort = np.mean(np.abs(action), axis=-1)
    return contact_force / (action_effort + eps)


def load_h5_trajectory(h5_path: str, env_idx: int = 0) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        fsr = np.asarray(f["fsr"], dtype=np.float32)[:, env_idx]
        action = np.asarray(f["action"], dtype=np.float32)[:, env_idx]
        q = np.asarray(f["q"], dtype=np.float32)[:, env_idx]

    return {
        "step": np.arange(fsr.shape[0], dtype=np.int32),
        "fsr": fsr,
        "action": action,
        "q": q,
        "cei": compute_cei(fsr, action),
    }

def plot_detailed_channels(h5_path: str, env_idx: int = 0) -> None:
    traj = load_h5_trajectory(h5_path, env_idx=env_idx)
    steps = traj["step"]
    fsr = traj["fsr"]
    action = traj["action"]
    q = traj["q"]
    cei = traj["cei"]

    stem = str(Path(h5_path).with_suffix("")) + f"_env{env_idx}"

    def create_grid_plot(values: np.ndarray, title: str, color: str, save_suffix: str) -> None:
        num_channels = int(values.shape[-1])
        fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
        fig.suptitle(title, fontsize=20)
        
        for i in range(num_channels):
            ax = axes[i // 4, i % 4]
            ax.plot(steps, values[:, i], color=color, linewidth=1.2)
            ax.set_title(f"ch_{i}")
            ax.grid(True, alpha=0.3)

        for i in range(num_channels, 16):
            ax = axes[i // 4, i % 4]
            ax.axis("off")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        save_path = f"{stem}_{save_suffix}.png"
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    # 1. 绘制所有 16 个 FSR
    create_grid_plot(fsr, "FSR Sensor Forces (0-15)", "tab:red", "fsr_detail")

    # 2. 绘制所有 16 个 Hand Actions
    create_grid_plot(action, "Joint Actions / Delta (0-15)", "tab:blue", "action_detail")

    # 3. 绘制手部关节位置 (q 的最后 16 维)
    q_hand = q[:, -16:] if q.shape[-1] >= 16 else q
    fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
    fig.suptitle("Joint Positions (Hand)", fontsize=20)
    for i in range(min(16, q_hand.shape[-1])):
        ax = axes[i // 4, i % 4]
        ax.plot(steps, q_hand[:, i], color="tab:orange")
        ax.set_title(f"q_hand_{i}")
        ax.grid(True, alpha=0.3)

    for i in range(q_hand.shape[-1], 16):
        ax = axes[i // 4, i % 4]
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
    plt.savefig(f"{stem}_pos_detail.png")
    print(f"Saved: {stem}_pos_detail.png")

    # 4. CEI 指标曲线
    fig_cei, ax_cei = plt.subplots(1, 1, figsize=(16, 4))
    ax_cei.plot(steps, cei, color="tab:green")
    ax_cei.set_title(f"Compliance Efficiency Index (mean={float(np.mean(cei)):.3f})")
    ax_cei.set_ylabel("CEI")
    ax_cei.set_xlabel("step")
    ax_cei.grid(True, alpha=0.3)
    fig_cei.tight_layout()
    fig_cei.savefig(f"{stem}_cei.png")
    print(f"Saved: {stem}_cei.png")
    
    plt.show()

if __name__ == "__main__":
    h5_files = sorted(
        glob.glob("./finger_copliance_control/data/*.h5"),
        key=lambda p: Path(p).stat().st_mtime,
    )
    h5_files = [p for p in h5_files if not p.endswith("_inverted.h5")]
    if h5_files:
        plot_detailed_channels(h5_files[-1], env_idx=0)
    else:
        print("No H5 found.")