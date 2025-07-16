import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# df = pd.read_csv("/Users/judewells/Documents/dataScienceProgramming/binding_affinity/VoxelDiffOuter/plixer/evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/plinder/plinder_combined_model_results_20250516_000646.csv")
# df = pd.read_csv("/Users/judewells/Downloads/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv")
df = pd.read_csv( "/Users/judewells/Documents/dataScienceProgramming/binding_affinity/VoxelDiffOuter/plixer/evaluation_results/checkpoints_model_run_2025-07-02_batched/chrono_combined_model_results_20250702_140314.csv")
plt.style.use("dark_background")
df = df.dropna(subset='tanimoto_similarity')

# First plot: Tanimoto similarity
plt.scatter(df.poc2mol_loss.values, df.tanimoto_similarity.values, c='#00FF00', alpha=1.0)
plt.xlabel("pixel loss")
plt.ylabel("tanimoto_similarity")

# Calculate correlation and p-value
corr_coef, p_value = stats.pearsonr(df.poc2mol_loss.values, df.tanimoto_similarity.values)
print(f"Tanimoto correlation: {corr_coef:.4f}, p-value: {p_value:.4e}")

# Add line of best fit
z = np.polyfit(df.poc2mol_loss.values, df.tanimoto_similarity.values, 1)
p = np.poly1d(z)
plt.plot(df.poc2mol_loss.values, p(df.poc2mol_loss.values), "r--", alpha=0.8)

plt.savefig("outputs/loss_tanimoto_scatter.png")
plt.show()
plt.clf()

# Second plot: SMILES loss
plt.scatter(df.poc2mol_loss.values, df.loss.values, c='#00FF00', alpha=1.0)
plt.xlabel("pixel loss")
plt.ylabel("SMILES loss")

# Calculate correlation and p-value
corr_coef, p_value = stats.pearsonr(df.poc2mol_loss.values, df.loss.values)
print(f"SMILES loss correlation: {corr_coef:.4f}, p-value: {p_value:.4e}")

# Add line of best fit
z = np.polyfit(df.poc2mol_loss.values, df.loss.values, 1)
p = np.poly1d(z)
plt.plot(df.poc2mol_loss.values, p(df.poc2mol_loss.values), "r--", alpha=0.8)

plt.savefig("outputs/loss_loss_scatter.png")
plt.show()
bp=1