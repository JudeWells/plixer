import pandas as pd
import os
import glob
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def plot_poc2mol_loss_histogram(df: pd.DataFrame, output_dir: str, prefix: str = ""):
    """Create and save a histogram of the `poc2mol_loss` column.

    The style loosely follows the histogram section in `generate_plots_from_results_df` but
    shows only a single distribution.  Uses a Times New Roman–like serif font.
    """
    if "poc2mol_loss" not in df.columns:
        raise ValueError("`poc2mol_loss` column not found in DataFrame")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Configure font to be Times New Roman (fallbacks included)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    # Prepare the data – drop NaNs
    data = df["poc2mol_loss"].dropna()
    if data.empty:
        print("[plot_poc2mol_loss_histogram] No valid data to plot.")
        return

    # Determine number of bins (sqrt rule, minimum of 10)
    bins = max(10, int(np.sqrt(len(data))))
    bin_edges = np.linspace(data.min(), data.max(), bins + 1)

    # Plot
    plt.figure(figsize=(6, 4))
    sns.histplot(data, bins=bin_edges, color="royalblue", alpha=0.6)

    plt.title("POC2Mol loss")
    plt.xlabel("loss")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()

    outfile = os.path.join(output_dir, f"{prefix}poc2mol_loss_hist.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[plot_poc2mol_loss_histogram] Figure saved to → {outfile}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def copy_any(src: str, dst: str):
    if os.path.isdir(src):
        # Recursively copy directory contents
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        # Copy single file, preserving metadata
        shutil.copy2(src, dst)


if __name__ == "__main__":
    output_dir = "evaluation_results/checkpoints_model_run_2025-07-04_batched/plinder/select_visualisations"

    df = pd.read_csv("evaluation_results/checkpoints_model_run_2025-07-04_batched/plinder/plinder_combined_model_results_20250704_161909.csv")

    results_dir = "outputs/images_from_cluster/CombinedHiQBindCkptFrmPrevCombined_2025-05-06_v3_member_zero_v3/voxel_visualizations_all_angles_cluster"
    plot_poc2mol_loss_histogram(df, output_dir)
    output_dir = "evaluation_results/checkpoints_model_run_2025-07-04_batched/plinder/select_visualisations"
    os.makedirs(output_dir, exist_ok=True)
    for i, row in df.iterrows():
        name = row['name']
        matches = glob.glob(f"{results_dir}/{name}*")
        if len(matches) == 0:
            print(f"No matches found for {name}")
            continue
        if len(matches) == 2:
            for m in matches:
                if not m.endswith(".tar.gz"):
                    if os.path.exists(os.path.join(output_dir, os.path.basename(m))):
                        print(f"Skipping {m} because it already exists")
                        continue
                    copy_any(m, os.path.join(output_dir, os.path.basename(m)))
                    break
        else:
            if os.path.exists(os.path.join(output_dir, name)):
                print(f"Skipping {name} because it already exists")
                continue
            copy_any(matches[0], os.path.join(output_dir, name))




