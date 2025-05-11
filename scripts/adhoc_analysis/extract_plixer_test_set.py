import os
import pandas as pd
import shutil

def process_systems(pdb_dir):
    pass

if __name__=="__main__":
    test_pdb_dir = "../hiqbind/test_pdbs/"
    os.makedirs(test_pdb_dir, exist_ok=True)
    df = pd.read_csv("evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv")
    for i, row in df.iterrows():
        pdb_id = row['name'].split("_")[0]
        system_dir = f"../hiqbind/raw_data_hiq_sm/{pdb_id}/{row['name']}"
        if not os.path.exists(system_dir):
            print(f"failed to find {system_dir}")
            continue
        shutil.copytree(system_dir, os.path.join(test_pdb_dir, row['name']))