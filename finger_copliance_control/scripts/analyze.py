import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_compliance_analysis(csv_path):
    df = pd.read_csv(csv_path)
    
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
        fsr_sum = df[[f"fsr_{j}" for j in idxs["fsr"]]].mean(axis=1)
        # 取该手指的主动作（比如中段关节的 Action）
        action_main = df[f"action_{idxs['action'][1]}"] 
        pos_main = df[f"pos_{idxs['pos'][1]}"]
        
        # 绘制受力 (左轴 - 填充图)
        line1 = ax.fill_between(df['step'], fsr_sum, color='gray', alpha=0.3, label='Avg FSR Force')
        ax.set_ylabel('Force (N)', color='gray')
        
        # 绘制 Action 和 Position (右轴)
        line2, = ax2.plot(df['step'], action_main, label='Action (Delta)', color='blue', linewidth=1.5)
        line3, = ax2.plot(df['step'], (pos_main - pos_main.iloc[0]), label='Pos Offset', color='red', linestyle='--')
        ax2.set_ylabel('Action / Pos Delta', color='blue')

        # 标注解锁状态 (is_unlocked)
        unlocked_mask = df['is_unlocked'] == 1
        ax.scatter(df['step'][unlocked_mask], [fsr_sum.max()*1.1]*unlocked_mask.sum(), 
                   marker='|', color='green', alpha=0.5, label='Unlocked')

        ax.set_title(f"Finger: {name} - Force vs Control response", fontsize=14)
        if i == 0:
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

    plt.suptitle(f"Trajectory Analysis: {Path(csv_path).name}", fontsize=16)
    plt.savefig(csv_path.replace(".csv", "_analysis.png"))
    print(f"Saved analysis plot to {csv_path.replace('.csv', '_analysis.png')}")

    # 2. 相关性热力图 (Correlation Heatmap)
    # 我们选几个关键维度看看 FSR 和 Action 的直接耦合度
    plt.figure(figsize=(12, 10))
    cols_to_corr = [f"fsr_{i}" for i in range(16)] + [f"action_{i}" for i in range(16)]
    corr = df[cols_to_corr].corr()
    
    # 重点看 FSR(行) 和 Action(列) 的交叉区域
    fsr_action_corr = corr.iloc[:16, 16:]
    sns.heatmap(fsr_action_corr, annot=False, cmap='RdBu_r', center=0)
    plt.title("Correlation: FSR vs Action")
    plt.xlabel("Actions")
    plt.ylabel("FSR Sensors")
    plt.savefig(csv_path.replace(".csv", "_corr.png"))
    plt.show()

if __name__ == "__main__":
    # 替换为你实际的 CSV 文件路径
    import glob
    csv_files = glob.glob("./finger_copliance_control/data/2/*.csv")
    if csv_files:
        plot_compliance_analysis(csv_files[0]) # 分析最新的一个
    else:
        print("No CSV found.")