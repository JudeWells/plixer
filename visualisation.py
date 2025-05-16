import os, math, random, re
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def generate_plots_from_results_df(df, output_dir, vis_deciles: bool = True, similarity_df: pd.DataFrame = None, prefix: str = ""):
    """
    Produce a collection of plots / molecule grids that we can directly
    drop in to the paper.

    1. Over‑laid histograms (sample vs. decoy) for
         • loss                       ↔ decoy_loss
         • tanimoto_similarity        ↔ decoy_tanimoto_similarity
         • mcs_num_atoms              ↔ decoy_mcs_num_atoms
         • prop_common_structure      ↔ decoy_prop_common_structure  (built here if absent)
    2. Bar‑plot of means with 95 % CI.
    3. For every decile of (i) tanimoto_similarity and (ii) prop_common_structure
       select two rows with a valid sampled SMILES, render the "true" and
       "sampled" molecule side‑by‑side and save the figure.
    """

    os.makedirs(output_dir, exist_ok=True)
    if "true_num_heavy_atoms" not in df.columns or df["true_num_heavy_atoms"].isna().all():
        def _num_heavy(smi: str):
            if not isinstance(smi, str):
                return np.nan
            # remove possible BOS / EOS tokens
            cleaned = re.sub(r"<[^>]+>", "", smi).strip()
            cleaned = cleaned.replace("[BOS]","").replace("[EOS]", "")
            m = Chem.MolFromSmiles(cleaned)
            return m.GetNumHeavyAtoms() if m else np.nan
        df["true_num_heavy_atoms"] = df["smiles"].apply(_num_heavy)

    # if "decoy_prop_common_structure" not in df.columns:
    #     df["decoy_prop_common_structure"] = (
    #         df["decoy_mcs_num_atoms"] / df["true_num_heavy_atoms"]
    #     )
    if "log_likelihood" not in df.columns:
        df["log_likelihood"] = -df["loss"]
        df["decoy_log_likelihood"] = -df["decoy_loss"]
    # Only work on numeric rows
    num_df = df.select_dtypes(include=[np.number])

    # ---------------------------------------------------------------------
    # 1) HISTOGRAMS
    # ---------------------------------------------------------------------
    hist_cfg = [
        ("decoy_loss", "loss", "Model loss"),
        ("decoy_log_likelihood","log_likelihood",  "SMILES Log likelihood"),
        ("decoy_tanimoto_similarity", "tanimoto_similarity",  "Tanimoto similarity"),
        # ("decoy_mcs_num_atoms", "mcs_num_atoms", "MCS – number of atoms"),
        # ("decoy_prop_common_structure", "prop_common_structure", "Proportion common sub‑structure"),
        ("decoy_smiles_true_label_teacher_forced_accuracy","smiles_teacher_forced_accuracy",  "SMILES token recovery rate (context is true label)"),
        ("decoy_smiles_decoy_label_teacher_forced_accuracy","smiles_teacher_forced_accuracy", "SMILES token recovery rate"),
    ]

    for left, right, title in hist_cfg:
        if left not in num_df.columns or right not in num_df.columns:
            continue
        plt.figure(figsize=(6, 4))
        # drop nan values
        left_data = num_df[left].dropna()
        right_data = num_df[right].dropna()
        
        # Calculate bins based on combined data range
        combined_data = pd.concat([left_data, right_data])
        bins = max(10, int(np.sqrt(len(combined_data))))
        bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)
        
        # Plot histograms with the same bins
        sns.histplot(right_data, bins=bin_edges, color="darkorange", alpha=0.5, label=right)
        sns.histplot(left_data, bins=bin_edges, color="royalblue", alpha=0.5, label=left)
        
        
        # means
        plt.axvline(left_data.mean(), color="royalblue", linestyle="--")
        plt.axvline(right_data.mean(), color="darkorange", linestyle="--")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}{left}_vs_{right}_hist.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------------------
    # 2) BARPLOT OF MEANS WITH 95 % CI
    # ---------------------------------------------------------------------
    bar_metrics: List[str] = [
        "loss",
        "decoy_loss",
        "tanimoto_similarity",
        "decoy_tanimoto_similarity",
        "smiles_teacher_forced_accuracy",
        "decoy_smiles_teacher_forced_accuracy",
    ]
    bar_vals, bar_errors = [], []
    for m in bar_metrics:
        vals = num_df[m].dropna()
        if len(vals) == 0:
            bar_vals.append(np.nan)
            bar_errors.append(0)
            continue
        mean = vals.mean()
        ci = 1.96 * vals.std(ddof=1) / math.sqrt(len(vals))
        bar_vals.append(mean)
        bar_errors.append(ci)

    # Matplotlib's bar handles y‑error bars directly
    plt.figure(figsize=(10, 4))
    indices = np.arange(len(bar_metrics))
    plt.bar(
        indices,
        bar_vals,
        yerr=bar_errors,
        align="center",
        alpha=0.8,
        ecolor="black",
        capsize=5,
        color=sns.color_palette("viridis", len(bar_metrics)),
    )
    plt.xticks(indices, bar_metrics, rotation=60, ha="right")
    plt.ylabel("mean ± 95 % CI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}metric_means_with_CI.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # 3) MOLECULE GRIDS FOR DECILES
    # ---------------------------------------------------------------------
    valid_rows = df[df["valid_smiles"] == 1].copy()
    if valid_rows.empty:
        return

    def _clean_smiles(s):
        if not isinstance(s, str):
            return ""
        s = re.sub(r"<[^>]+>", "", s)  # remove <s> tokens
        s = s.replace("[BOS]","").replace("[EOS]", "")
        return s.strip()

    def _draw_pair(row, prefix, idx):
        true_smi = _clean_smiles(row["smiles"])
        samp_smi = row["sampled_smiles"]
        true_mol = Chem.MolFromSmiles(true_smi)
        samp_mol = Chem.MolFromSmiles(samp_smi)
        if true_mol is None or samp_mol is None:
            return
        if "tanimoto" in prefix:
            legend=f"Sampled (TS={round(row.tanimoto_similarity, 2)})"
        elif "propCommon" in prefix:
            legend=f"Sampled (PropCommon={round(row.prop_common_structure, 2)})"

        img = Draw.MolsToGridImage(
            [true_mol, samp_mol],
            molsPerRow=2,
            subImgSize=(300, 300),
            legends=["True", legend],
        )
        img.save(os.path.join(output_dir, f"{prefix}_{idx}.png"))

    # helper for two metrics
    def _make_decile_imgs(metric, prefix):
        if metric not in valid_rows.columns:
            return
        bins = pd.qcut(valid_rows[metric], 100, labels=False, duplicates="drop")
        valid_rows[f"{metric}_bin"] = bins
        for b in sorted(valid_rows[f"{metric}_bin"].dropna().unique()):
            group = valid_rows[valid_rows[f"{metric}_bin"] == b]
            if group.empty:
                continue
            # pick up to two examples
            for idx, (_, r) in enumerate(group.sample(min(2, len(group)), random_state=42).iterrows()):
                _draw_pair(r, f"{prefix}_decile{int(b)}", idx)

    if vis_deciles:
        _make_decile_imgs("tanimoto_similarity", f"{prefix}tanimoto")
        # _make_decile_imgs("prop_common_structure", f"{prefix}propCommon")

    print(f"[generate_plots_for_paper] Figures written to → {output_dir}")