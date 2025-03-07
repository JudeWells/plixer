import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the log likelihood results
log_likelihood_df = pd.read_csv("vox2smiles_evaluation2/log_likelihood_results.csv")
log_likelihood_df = log_likelihood_df.drop_duplicates(subset=['smiles'])
print("Number of unique smiles in log likelihood results: ", len(log_likelihood_df))
log_likelihood_df.reset_index(drop=True, inplace=True)
log_likelihood_df['ranking'] = log_likelihood_df.index

hit_label_df = pd.read_csv("/mnt/disk2/VoxelDiffOuter/VoxelDiff2/data/cache_round1_smiles_all_out_hits_and_others.csv")
hit_label_df = hit_label_df.drop_duplicates(subset=['smiles'])
print("Number of unique smiles in hit label results: ", len(hit_label_df))
# Merge the two dataframes on the smiles column
merged_df = pd.merge(log_likelihood_df, hit_label_df, on="smiles", how="inner")
print(f"Average hit: {merged_df[merged_df['hit']==1]['per_token_log_likelihood'].mean()}")
print(f"Average miss: {merged_df[merged_df['hit']==0]['per_token_log_likelihood'].mean()}")
print(f"Average hit ranking: {merged_df[merged_df['hit']==1]['ranking'].mean()}")
print(f"Average miss ranking: {merged_df[merged_df['hit']==0]['ranking'].mean()}")
# Save the merged dataframe
merged_df.to_csv("vox2smiles_evaluation2/log_likelihood_results_with_hit_label.csv", index=False)
n_hits = merged_df[merged_df['hit']==1].shape[0]
merged_df = merged_df.sort_values(by='ranking')
top_df = merged_df.head(n_hits)
proportion_of_hits_at_top = top_df['hit'].mean()
print(f"Proportion of hits at top {n_hits}: {proportion_of_hits_at_top}")
# Create a density plot to compare distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=merged_df, 
    x='per_token_log_likelihood', 
    hue='hit',
    fill=True,
    common_norm=False,
    palette={1: 'green', 0: 'red'},
    alpha=0.7,
    linewidth=2
)

plt.title('Distribution of Per-Token Log Likelihoods: Hits vs Non-Hits', fontsize=14)
plt.xlabel('Per-Token Log Likelihood', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(labels=['Hits', 'Non-Hits'], title='Category')
plt.grid(True, linestyle='--', alpha=0.7)

# Add vertical lines for the means
hit_mean = merged_df[merged_df['hit']==1]['per_token_log_likelihood'].mean()
miss_mean = merged_df[merged_df['hit']==0]['per_token_log_likelihood'].mean()
plt.axvline(hit_mean, color='green', linestyle='--', label=f'Hit Mean: {hit_mean:.3f}')
plt.axvline(miss_mean, color='red', linestyle='--', label=f'Non-Hit Mean: {miss_mean:.3f}')

# Save the plot
plt.tight_layout()
plt.savefig("vox2smiles_evaluation2/log_likelihood_distribution.png", dpi=300)
plt.show()


