import pandas as pd

df = pd.read_parquet("lab/runs/h1_cross_condition_physics_balanced_8389c3f2cd6d/artifacts/cross_condition/head_matrix.parquet")

print("Columns:", df.columns.tolist())
print(f"Shape: {df.shape}\n")

# Head 0:2 across all conditions
head_02 = df[(df['layer'] == 0) & (df['head'] == 2)]
print("Head 0:2 (single vs multi-token):")
print(head_02[['condition', 'metric', 'value']].drop_duplicates().sort_values(['condition', 'metric']))

# All layer-0 heads summary
print("\nLayer 0 suppressor trio comparison (logit_diff vs seq_logprob_diff):")
layer0 = df[(df['layer'] == 0) & (df['head'].isin([2, 4, 7]))]
for head in [2, 4, 7]:
    print(f"\n  Head 0:{head}:")
    h = layer0[layer0['head'] == head]
    for metric in ['logit_diff', 'seq_logprob_diff']:
        vals = h[h['metric'] == metric].groupby('condition')['value'].mean()
        print(f"    {metric}:\n{vals}")