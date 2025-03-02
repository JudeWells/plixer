"""
Uses the structrual VAE
to generate encodings of the Clinvar mutation site
then uses a gaussian process to classify the mutation as pathogenic or not
"""


import glob
import os
import pandas as pd
import torch
from src.data.poc2mol_datasets import StructuralPretrainDataset

def filter_for_best_scoring_model(paths):
    """
    filenames are like so:
    /clinvar_structures/NP_000007.1_A81T/pred_Mutant.model_idx_0_plddt_0.8807.cif
    we want to select the filepath with the highest plddt score
    """
    clinvar_ids = [
        p.split("/")[-2]
        for p in paths
    ]
    clinvar_ids = list(set(clinvar_ids))
    best_plddts = []
    for clinvar_id in clinvar_ids:
        clinvar_id_paths = [p for p in paths if clinvar_id in p]
        best_plddt_path = max(clinvar_id_paths, key=lambda x: float(x.split("plddt_")[-1].split(".cif")[0]))
        best_plddts.append(best_plddt_path)
    return best_plddts


if __name__ == "__main__":
    clinvar_mut_pattern = "/mnt/disk2/ab_bind_cofold/outputs/clinvar_structures/*/pred_Mutant.model_idx_*.cif"
    clinvar_wt_pattern = "/mnt/disk2/ab_bind_cofold/outputs/clinvar_structures/*/pred_WT.model_idx_*.cif"
    clinvar_mut_paths = glob.glob(clinvar_mut_pattern)
    clinvar_wt_paths = glob.glob(clinvar_wt_pattern)
    clinvar_mut_paths = filter_for_best_scoring_model(clinvar_mut_paths)
    clinvar_wt_paths = filter_for_best_scoring_model(clinvar_wt_paths)

    print(len(clinvar_mut_paths))
    print(len(clinvar_wt_paths))
    bp=1
