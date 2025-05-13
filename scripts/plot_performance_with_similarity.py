import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
similarity_df = pd.read_csv("../hiqbind/similarity_analysis/test_similarities.csv")
res_path = "evaluation_results/CombinedHiQBindCkptFrmPrevCombined_2025-05-06_v3_member_zero_v3/combined_model_results.csv"
results_df = pd.read_csv(res_path)
save_dir = os.path.dirname(res_path)
results_df['system_id'] = results_df['name']
if "log_likelihood" not in results_df.columns:
    results_df["log_likelihood"] = -results_df["loss"]
# Merge similarity data with results data
merged_df = pd.merge(results_df, similarity_df, on="system_id", how="left")
merged_df['max_protein_similarity'] = merged_df['max_protein_similarity'].fillna(0)
plt.scatter(merged_df['max_protein_similarity'], merged_df['poc2mol_loss'], alpha=0.1)
plt.show()

metrics_for_box_plots = [
    "poc2mol_loss",
    "loss",
    "log_likelihood",
    "tanimoto_similarity",
    "prop_common_structure"
]

similarity_bins = [
    (0,15),
    (15,30),
    (30,40),
    (40,50),
    (50,70),
    (70,90),
    (90,100)
]

# Create similarity bin labels
bin_labels = [f"{start}-{end}%" for start, end in similarity_bins]

# Add similarity bin column to dataframe
merged_df['similarity_bin'] = pd.cut(
    merged_df['max_protein_similarity'],
    bins=[b[0] for b in similarity_bins] + [similarity_bins[-1][1]],
    labels=bin_labels,
    include_lowest=True
)

sns.set_palette("husl")

# Create box plots for each metric
for metric in metrics_for_box_plots:
    plt.figure(figsize=(12, 6))
    
    # Create box plot
    sns.boxplot(
        data=merged_df,
        x='similarity_bin',
        y=metric,
        showfliers=False  # Hide outliers for cleaner visualization
    )
    
    # Customize the plot
    plt.title(f'{metric} Distribution by Protein Similarity Bin', fontsize=14, pad=20)
    plt.xlabel('Protein Similarity Range (%)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'prot_similarity_boxplot_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create a summary statistics table
summary_stats = merged_df.groupby('similarity_bin')[metrics_for_box_plots].agg(['mean', 'std', 'median', 'count'])
summary_stats.to_csv(os.path.join(save_dir, 'similarity_bin_statistics.csv'))

