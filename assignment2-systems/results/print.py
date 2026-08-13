import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# 读取 CSV 文件
df = pd.read_csv("flash_attention_benchmark_results.csv")

# 过滤掉 OOM 的行（status 为 "OOM" 的行数值缺失）
df = df[df['status'] == 'ok'].copy()

# 确保数值列为浮点类型（避免类型问题）
numeric_cols = ['seq_len', 'd_model', 'triton_fwd_ms', 'triton_bwd_ms', 'triton_e2e_ms',
                'torch_fwd_ms', 'torch_bwd_ms', 'torch_e2e_ms',
                'speedup_fwd', 'speedup_bwd', 'speedup_e2e']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 删除可能存在的 NaN（如果有）
df.dropna(subset=numeric_cols, inplace=True)

# 将 dtype 设置为分类类型，方便图例顺序
df['dtype'] = pd.Categorical(df['dtype'], categories=['bfloat16', 'float32'])

# ---- 图1: 加速比对比（fwd, bwd, e2e） ----
# 使用分面网格，按 d_model 和 dtype 分组
g = sns.FacetGrid(df, row='d_model', col='dtype', height=3, aspect=1.2, sharex=False)
g.map_dataframe(sns.lineplot, x='seq_len', y='speedup_fwd', label='fwd', marker='o')
g.map_dataframe(sns.lineplot, x='seq_len', y='speedup_bwd', label='bwd', marker='s')
g.map_dataframe(sns.lineplot, x='seq_len', y='speedup_e2e', label='e2e', marker='^')
g.add_legend()
g.set_axis_labels("Sequence Length", "Speedup (Torch / Triton)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Speedup vs Sequence Length (by d_model and dtype)")
plt.savefig("speedup_compare.png", bbox_inches='tight')
plt.close()

# ---- 图2: 端到端运行时间对比（Triton vs PyTorch） ----
g2 = sns.FacetGrid(df, row='d_model', col='dtype', height=3, aspect=1.2, sharex=False)
g2.map_dataframe(sns.lineplot, x='seq_len', y='triton_e2e_ms', label='Triton', marker='o')
g2.map_dataframe(sns.lineplot, x='seq_len', y='torch_e2e_ms', label='PyTorch', marker='s')
g2.add_legend()
g2.set_axis_labels("Sequence Length", "Time (ms)")
g2.fig.subplots_adjust(top=0.9)
g2.fig.suptitle("End-to-End Time: Triton vs PyTorch")
plt.savefig("e2e_time_compare.png", bbox_inches='tight')
plt.close()

# ---- 图3: 按 d_model 分别绘制 e2e 加速比（不同 dtype 对比） ----
# 使用 seaborn 的 relplot，按 d_model 分面，并在同一图中区分 dtype
g3 = sns.relplot(data=df, x='seq_len', y='speedup_e2e', hue='dtype', style='dtype',
                 col='d_model', kind='line', marker=True, height=3, aspect=1.2)
g3.set_axis_labels("Sequence Length", "End-to-End Speedup")
g3.fig.subplots_adjust(top=0.9)
g3.fig.suptitle("End-to-End Speedup by d_model")
plt.savefig("speedup_e2e_by_dmodel.png", bbox_inches='tight')
plt.close()

# ---- 图4: 热力图（可选）：展示加速比随 seq_len 和 d_model 的变化（按 dtype 分开） ----
# 为每个 dtype 绘制一个热力图，显示 e2e 加速比
for dtype_name in df['dtype'].unique():
    sub = df[df['dtype'] == dtype_name].pivot(index='d_model', columns='seq_len', values='speedup_e2e')
    plt.figure(figsize=(10, 6))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Speedup'})
    plt.title(f"End-to-End Speedup Heatmap (dtype={dtype_name})")
    plt.xlabel("Sequence Length")
    plt.ylabel("d_model")
    plt.tight_layout()
    plt.savefig(f"heatmap_speedup_e2e_{dtype_name}.png")
    plt.close()

print("所有图表已生成完毕！")