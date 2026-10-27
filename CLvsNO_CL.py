import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
FILE_CL = 'global_metrics_CL.csv'
FILE_NOCL = 'global_metrics_NoCL.csv'
WINDOW_SIZE = 50  # 平滑窗口大小


# ================= 数据预处理函数 =================
def load_and_fix_epochs(file_path):
    """
    加载数据，并自动检测/修复 Epoch 重置的问题（即从 1..500 变成 1..500...）
    将其转换为累积 Epoch (Cumulative Epoch)
    """
    try:
        df = pd.read_csv(file_path)

        # 1. 确保按时间排序 (如果有 timestamp)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # 2. 检测 Epoch 是否单调递增
        if df['epoch'].is_monotonic_increasing:
            df['cumulative_epoch'] = df['epoch']
        else:
            # 如果不单调，说明分阶段训练了，需要累加
            print(f"检测到 {file_path} 中 Epoch 非单调，正在转换为累积 Epoch...")
            new_epochs = []
            prev_epoch = -1
            offset = 0

            for e in df['epoch']:
                # 如果当前 epoch 小于上一个，说明进入了新阶段 (reset)
                if e < prev_epoch:
                    offset += prev_epoch
                new_epochs.append(e + offset)
                prev_epoch = e

            df['cumulative_epoch'] = new_epochs

        return df
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None


# ================= 主程序 =================

# 1. 加载数据
cl_df = load_and_fix_epochs(FILE_CL)
nocl_df = load_and_fix_epochs(FILE_NOCL)

if cl_df is not None and nocl_df is not None:
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')

    # ==========================================
    # 图表 1: 公平对比 (Bar Chart: Performance vs Difficulty)
    # ==========================================
    # 定义难度区间 (Bins)
    bins = range(0, 160, 20)
    labels = [f"{i}-{i + 20}" for i in range(0, 140, 20)]

    # 将数据分箱
    cl_df['n_bin'] = pd.cut(cl_df['vehicle_count'], bins=bins, labels=labels)
    nocl_df['n_bin'] = pd.cut(nocl_df['vehicle_count'], bins=bins, labels=labels)

    # 计算每个难度的平均 Feasible Success Rate
    cl_perf = cl_df.groupby('n_bin')['feasible_v2v_success_rate'].mean().fillna(0)
    nocl_perf = nocl_df.groupby('n_bin')['feasible_v2v_success_rate'].mean().fillna(0)

    # 绘图
    fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width / 2, cl_perf, width, label='CL (Curriculum)', color='#1f77b4', alpha=0.9)
    ax1.bar(x + width / 2, nocl_perf, width, label='No-CL (Baseline)', color='#ff7f0e', alpha=0.9)

    ax1.set_title('Fair Comparison: Performance vs Difficulty (N)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feasible Success Rate', fontsize=12)
    ax1.set_xlabel('Vehicle Count (Difficulty Bin)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Performance_vs_Difficulty.png')
    print("图表 1 已保存: Performance_vs_Difficulty.png")

    # ==========================================
    # 图表 2: 训练动态 (Line Chart: Sequential Training Dynamics)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(14, 7), dpi=150)

    # --- 右轴: 绘制难度 (背景) ---
    ax2_right = ax2.twinx()
    # 使用累积 Epoch 作为 X 轴
    ax2_right.plot(cl_df['cumulative_epoch'], cl_df['vehicle_count'], color='gray', linestyle=':', alpha=0.5,
                   label='CL Difficulty (N)')
    ax2_right.set_ylabel('Difficulty (Vehicle Count)', color='gray', fontsize=12)

    # --- 左轴: 绘制性能 (前景) ---
    # 计算平滑曲线
    cl_smooth = cl_df['feasible_v2v_success_rate'].rolling(WINDOW_SIZE, min_periods=1).mean()
    nocl_smooth = nocl_df['feasible_v2v_success_rate'].rolling(WINDOW_SIZE, min_periods=1).mean()

    ax2.plot(cl_df['cumulative_epoch'], cl_smooth, label='CL Feasible Rate', color='#1f77b4', linewidth=2)
    ax2.plot(nocl_df['cumulative_epoch'], nocl_smooth, label='No-CL Feasible Rate', color='#ff7f0e', linewidth=2)

    ax2.set_title('Training Dynamics: Sequential View (Corrected Epochs)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cumulative Training Steps (Epochs)', fontsize=12)
    ax2.set_ylabel('Feasible Success Rate', fontsize=12)

    # 合并图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Training_Dynamics_Sequential.png')
    print("图表 2 已保存: Training_Dynamics_Sequential.png")

    plt.show()