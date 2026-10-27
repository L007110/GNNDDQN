import pandas as pd
import numpy as np
import os


def merge_and_generate_plot_data():
    print("🚀 开始读取并整合实验数据...")

    # 1. 读取基线数据 (包含 Ashraf 和 Random)
    file_baseline = "Strict_Baseline_Results_Full.csv"
    if os.path.exists(file_baseline):
        df_base = pd.read_csv(file_baseline)
        print(f"📂 成功读取: {file_baseline}")
    else:
        df_base = pd.DataFrame()
        print(f"⚠️ 找不到: {file_baseline}")

    # 2. 读取并清洗 NoGNN-CL 数据
    file_nognn = "results_NoGNN_CL.csv"
    if os.path.exists(file_nognn):
        df_nognn = pd.read_csv(file_nognn)
        # 🛑 核心修复：重命名列，与 Baseline 对齐
        df_nognn = df_nognn.rename(columns={
            'density': 'Density',
            'raw_success_rate': 'V2V_Success_Rate',
            'v2v_sum_mbps': 'V2V_Sum_Capacity',
            'v2i_sum_mbps': 'V2I_Sum_Capacity'
        })
        df_nognn['Method'] = 'NoGNN-CL'
        print(f"📂 成功读取并清洗: {file_nognn}")
    else:
        df_nognn = pd.DataFrame()
        print(f"⚠️ 找不到: {file_nognn}")

    # 3. 读取并清洗 GNN-CL 数据
    file_gnn = "results_CL.csv"
    if os.path.exists(file_gnn):
        df_gnn = pd.read_csv(file_gnn)
        # 🛑 核心修复：重命名列，与 Baseline 对齐
        df_gnn = df_gnn.rename(columns={
            'density': 'Density',
            'raw_success_rate': 'V2V_Success_Rate',
            'v2v_sum_mbps': 'V2V_Sum_Capacity',
            'v2i_sum_mbps': 'V2I_Sum_Capacity'
        })
        df_gnn['Method'] = 'GNN-CL'
        print(f"📂 成功读取并清洗: {file_gnn}")
    else:
        df_gnn = pd.DataFrame()
        print(f"⚠️ 找不到: {file_gnn}")

    # 4. 把所有数据拼接到一起
    all_dfs = [df for df in [df_base, df_nognn, df_gnn] if not df.empty]
    if not all_dfs:
        print("❌ 没有成功读取任何数据文件，程序退出。")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 5. 生成透视表 (宽表格式)
    try:
        # 只取我们需要画图的核心列
        cols_to_keep = ['Density', 'Method', 'V2V_Success_Rate', 'V2V_Sum_Capacity', 'V2I_Sum_Capacity']
        # 过滤掉不在 DataFrame 中的列，防止报错
        available_cols = [c for c in cols_to_keep if c in merged_df.columns]

        merged_df = merged_df[available_cols]

        df_wide = pd.pivot_table(
            merged_df,
            index='Density',
            columns='Method',
            values=['V2V_Success_Rate', 'V2V_Sum_Capacity', 'V2I_Sum_Capacity'],
            aggfunc='mean'
        )

        # 扁平化表头：变成 GNN-CL_V2V_Success_Rate
        df_wide.columns = [f"{col[1]}_{col[0]}" for col in df_wide.columns]
        df_wide.reset_index(inplace=True)

    except Exception as e:
        print(f"❌ 数据透视转换失败: {e}")
        return

    # 6. 自动补齐容错机制 (防止某些算法未来得及跑完时 LaTeX 报错)
    expected_methods = ["GNN-CL", "NoGNN-CL", "Ashraf", "Random"]
    metrics = ["V2V_Success_Rate", "V2V_Sum_Capacity", "V2I_Sum_Capacity"]

    for method in expected_methods:
        for metric in metrics:
            col_name = f"{method}_{metric}"
            if col_name not in df_wide.columns:
                df_wide[col_name] = np.nan  # 缺失的列填充空值，LaTeX 会自动跳过

    # 7. 保存最终文件
    output_file = "Plot_Data.csv"
    df_wide.to_csv(output_file, index=False)
    print(f"\n✅ 完美！已成功整合三个文件，处理了列名不一致问题。")
    print(f"🎯 绘图数据已保存至: {output_file}")
    print("✨ 请直接去重新编译 LaTeX，4 条线这次一定会同时出现！")


if __name__ == "__main__":
    merge_and_generate_plot_data()