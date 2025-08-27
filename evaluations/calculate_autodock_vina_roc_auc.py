"""
Created by Jude Wells 2025-05-25
Used CBGBench pipeline to cross-dock
all ligands in each of the test sets.
For each system we dock the true binder
and all others in the test set as decoys.
eg results saved here:
CBGBench/outputs/vina_dock_decoys/plixer_plinder_split/docking_summary.csv
for each system calculate the ROC AUC score
and report the average ROC AUC score across all systems.
"""

import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_roc_auc(df):
    target_values = df.is_hit.values
    pred_values = df.vina_score.values * -1
    return roc_auc_score(target_values, pred_values)

if __name__ == "__main__":
    df_paths ={
        "plinder": "evaluation_results/autodock_vina/plixer_plinder_split/docking_summary.csv",
        "seq_sim": "evaluation_results/autodock_vina/plixer_seq_sim_split/docking_summary.csv",
    }
    for split_name, df_path in df_paths.items():
        summary_rows = []
        df = pd.read_csv(df_path)
        for system_id in df["system_id"].unique():
            system_df = df[df["system_id"] == system_id]
            auc = calculate_roc_auc(system_df)
            new_row = {
                "split_name": split_name,
                "system_id": system_id,
                "roc_auc": auc,
            }
            summary_rows.append(new_row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f"evaluation_results/autodock_vina/roc_auc_summary_{split_name}.csv", index=False)
        print(f"ROC AUC score for {split_name}: {summary_df['roc_auc'].mean()}")