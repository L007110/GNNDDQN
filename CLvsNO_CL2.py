import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
FILE_CL = 'global_metrics_CL.csv'
FILE_NOCL = 'global_metrics_NoCL.csv'
WINDOW_SIZE = 50  # 平滑窗口


# ================= 数据预处理 =================
def load_and_fix_epochs(file_path):
    """自动处理 Epoch 重置问题，生成 cumulative_epoch"""
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        if df['epoch'].is_monotonic_increasing:
            df['cumulative_epoch'] = df['epoch']
        else:
            new_epochs = []
            prev_epoch = -1
            offset = 0
            for e in df['epoch']:
                if e < prev_epoch:
                    offset += prev_epoch
                new_epochs.append(e + offset)
                prev_epoch = e
            df['cumulative_epoch'] = new_epochs
        return df
    except FileNotFoundError:
        print(f"Error: Not found {file_path}")
        return None


# 加载数据
cl_df = load_and_fix_epochs(FILE_CL)
nocl_df = load_and_fix_epochs(FILE_NOCL)

if cl_df is not None and nocl_df is not None:
    plt.style.use('seaborn-v0_8-whitegrid')

    # 准备数据：按难度分箱
    bins = range(0, 160, 20)
    labels = [f"{i}-{i + 20}" for i in range(0, 140, 20)]

    cl_df['n_bin'] = pd.cut(cl_df['vehicle_count'], bins=bins, labels=labels)
    nocl_df['n_bin'] = pd.cut(nocl_df['vehicle_count'], bins=bins, labels=labels)

    # 计算分组均值
    cl_cap = cl_df.groupby('n_bin')['v2i_sum_capacity'].mean().fillna(0)
    nocl_cap = nocl_df.groupby('n_bin')['v2i_sum_capacity'].mean().fillna(0)

    cl_delay = cl_df.groupby('n_bin')['mean_delay'].mean().fillna(0)
    nocl_delay = nocl_df.groupby('n_bin')['mean_delay'].mean().fillna(0)

    # ==========================================
    # 图表 1: 容量与延迟对比 (Bar Charts)
    # ==========================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    x = np.arange(len(labels))
    width = 0.35

    # 子图 1: V2I 容量 (越高越好)
    ax1.bar(x - width / 2, cl_cap, width, label='CL (Curriculum)', color='#2ca02c', alpha=0.9)  # 绿色
    ax1.bar(x + width / 2, nocl_cap, width, label='No-CL (Baseline)', color='#d62728', alpha=0.9)  # 红色

    ax1.set_ylabel('V2I Sum Capacity (Mbps)', fontsize=12)
    ax1.set_title('Metric 1: V2I Capacity (Higher is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Difficulty (Vehicle Count)', fontsize=12)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # 子图 2: 平均延迟 (越低越好，但需满足约束)
    ax2.bar(x - width / 2, cl_delay, width, label='CL', color='#9467bd', alpha=0.9)  # 紫色
    ax2.bar(x + width / 2, nocl_delay, width, label='No-CL', color='#8c564b', alpha=0.9)  # 棕色

    ax2.set_ylabel('Mean Delay (s)', fontsize=12)
    ax2.set_title('Metric 2: Mean Delay (Constraint: < 0.1s)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Difficulty (Vehicle Count)', fontsize=12)
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, label='Constraint (100ms)')  # 画出约束线
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0, 0.12)  # 聚焦在低延迟区域

    plt.tight_layout()
    plt.savefig('Additional_Metrics_Analysis.png')
    print("图表 1 已保存: Additional_Metrics_Analysis.png")

    # ==========================================
    # 图表 2: 容量训练动态 (Sequential Line Chart)
    # ==========================================
    fig2, ax3 = plt.subplots(figsize=(14, 7), dpi=150)

    # 右轴：难度背景
    ax3_right = ax3.twinx()
    ax3_right.plot(cl_df['cumulative_epoch'], cl_df['vehicle_count'], color='gray', linestyle=':', alpha=0.5,
                   label='CL Difficulty')
    ax3_right.set_ylabel('Difficulty (N)', color='gray')

    # 左轴：容量曲线 (平滑)
    cl_cap_smooth = cl_df['v2i_sum_capacity'].rolling(WINDOW_SIZE, min_periods=1).mean()
    nocl_cap_smooth = nocl_df['v2i_sum_capacity'].rolling(WINDOW_SIZE, min_periods=1).mean()

    # 对于 No-CL，也使用它的 cumulative_epoch (如果之前修正过) 或 epoch
    x_nocl = nocl_df['cumulative_epoch'] if 'cumulative_epoch' in nocl_df.columns else nocl_df['epoch']

    ax3.plot(cl_df['cumulative_epoch'], cl_cap_smooth, label='CL Capacity', color='#2ca02c', linewidth=2.5)
    ax3.plot(x_nocl, nocl_cap_smooth, label='No-CL Capacity', color='#d62728', linewidth=2.5)

    ax3.set_xlabel('Cumulative Training Steps (Epochs)', fontsize=12)
    ax3.set_ylabel('V2I Sum Capacity (Mbps)', fontsize=12)
    ax3.set_title('Training Dynamics: Capacity Evolution', fontsize=14, fontweight='bold')

    # 合并图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Capacity_Dynamics_Sequential.png')
    print("图表 2 已保存: Capacity_Dynamics_Sequential.png")

    plt.show()