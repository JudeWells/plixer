import pandas as pd
import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem

"""
Calculate tanimoto similarity of morgan fingerprints of predicted and hit smiles
QED, logP,
"""

def calculate_tanimoto_similarity(pred_smiles, hit_smiles):
    pass

def calculate_qed(smiles):
    pass

def calculate_logP(smiles):
    pass

def smiles_is_valid(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

if __name__=="__main__":
    test_df_paths = [
        ("plinder", "evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/plinder/plinder_combined_model_results_20250516_000646.csv")
    ]
    pocket2mmol_results_dir = "evaluation_results/pocket2mol/pocket2mol_on_plinder_split"
    all_smiles = []
    for split_name, df_path in test_df_paths:
        df = pd.read_csv(df_path)
        for i, row in df.iterrows():
            system_id = row["name"]
            smiles_path = f"{pocket2mmol_results_dir}/{system_id}/SMILES.txt"
            if not os.path.exists(smiles_path):
                print(f"SMILES file not found for {system_id}")
                continue
            with open(smiles_path, "r") as f:
                smiles = f.readlines()[-1].strip()
            hit_smiles = row['smiles'].replace("[EOS]", "").replace("[BOS]", "")
            all_smiles.append(smiles)
            result = {
                "system_id": system_id,
                "smiles": smiles,
                "hit_smiles": hit_smiles,
                "tanimoto_similarity": calculate_tanimoto_similarity(smiles, hit_smiles),
                "qed": calculate_qed(smiles),
                "logP": calculate_logP(smiles),
                "is_valid": smiles_is_valid(smiles),
            }
