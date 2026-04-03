import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_detailed_channels(csv_path):
    df = pd.read_csv(csv_path)
    steps = df['step']
    unlocked = df['is_unlocked']

    def create_grid_plot(data_prefix, num_channels, title, ylabel, color):
        fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
        fig.suptitle(title, fontsize=20)
        
        for i in range(num_channels):
            ax = axes[i // 4, i % 4]
            col_name = f"{data_prefix}_{i}"
            if col_name in df.columns:
                # 绘制主体信号
                ax.plot(steps, df[col_name], color=color, linewidth=1.5)
                # 底色高亮显示 "Unlocked" 状态（绿色区域代表控制器激活）
                ax.fill_between(steps, df[col_name].min(), df[col_name].max(), 
                                where=unlocked==1, color='green', alpha=0.1)
                ax.set_title(col_name)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        save_path = csv_path.replace(".csv", f"_{data_prefix}_detail.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    # 1. 绘制所有 16 个 FSR
    create_grid_plot("fsr", 16, "FSR Sensor Forces (0-15)", "Force (N)", "tab:red")

    # 2. 绘制所有 16 个 Hand Actions
    create_grid_plot("action", 16, "Joint Actions / Delta (0-15)", "Action", "tab:blue")

    # 3. 绘制手部 关节位置 (从 pos_6 到 pos_21)
    # 注意：你的 pos 数据从 0 开始计数，前 6 位是臂，手部是 6-21
    fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
    fig.suptitle("Joint Positions (Hand Only: pos_6 to pos_21)", fontsize=20)
    for i in range(16):
        ax = axes[i // 4, i % 4]
        col_name = f"pos_{i}" # 偏移 6 位
        if col_name in df.columns:
            ax.plot(steps, df[col_name], color="tab:orange")
            ax.set_title(col_name)
            ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
    plt.savefig(csv_path.replace(".csv", "_pos_detail.png"))
    
    plt.show()

if __name__ == "__main__":
    import glob
    csv_files = glob.glob("./finger_copliance_control/data/*.csv")
    if csv_files:
        plot_detailed_channels(csv_files[0])
    else:
        print("No CSV found.")