import glob
import os
import pandas as pd
from scripts.create_hiqbind_dataset import (
    calculate_similarity_scores,
    extract_protein_sequence,
    batch_search_protein_similarities,
    build_mmseqs_database
)
from tqdm import tqdm
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_protein_sequences(
        train_dir: str = "../hiqbind/parquet/train", 
        test_dir: str = "../hiqbind/parquet/test"
        ):
    train_save_path = '../hiqbind/plixer_train_data.csv'
    test_save_path = '../hiqbind/plixer_test_data.csv'
    if os.path.exists(test_save_path) and os.path.exists(train_save_path):
        return pd.read_csv(train_save_path), pd.read_csv(test_save_path)
    train_files  = glob.glob(os.path.join(train_dir, "*.parquet"))
    test_files  = glob.glob(os.path.join(test_dir, "*.parquet"))
    new_rows_train = []
    new_rows_test = []
    
    print("Processing train files...")
    for file in tqdm(train_files):
        df = pd.read_parquet(file)
        for index, row in df.iterrows():
            pdb_path = os.path.join(
                "../hiqbind/raw_data_hiq_sm",
                row.system_id.split("_")[0],
                row.system_id,
                f"{row.system_id}_protein.pdb"
            )
            if os.path.exists(pdb_path):
                new_rows_train.append({
                    "system_id": row.system_id,
                    "protein_sequence": extract_protein_sequence(pdb_path),
                    "smiles": row.smiles
                })
    
    print("Processing test files...")
    for file in tqdm(test_files):
        df = pd.read_parquet(file)
        for index, row in df.iterrows():
            pdb_path = os.path.join(
                "../hiqbind/raw_data_hiq_sm",
                row.system_id.split("_")[0],
                row.system_id,
                f"{row.system_id}_protein.pdb"
            )
            if os.path.exists(pdb_path):
                new_rows_test.append({
                    "system_id": row.system_id,
                    "protein_sequence": extract_protein_sequence(pdb_path),
                    "smiles": row.smiles
                })
    
    return pd.DataFrame(new_rows_train), pd.DataFrame(new_rows_test)

def calculate_similarity_scores(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate maximum similarity scores for test set entries relative to training set using batch MMSEQS search."""
    with tempfile.TemporaryDirectory() as tmpd:
        # Build sequence dictionaries
        test_seqs = {row["system_id"]: row["protein_sequence"] for _, row in test_df.iterrows() if row["protein_sequence"]}
        train_seqs = {row["system_id"]: row["protein_sequence"] for _, row in train_df.iterrows() if row["protein_sequence"]}
        
        # Build MMSEQS databases
        test_db = build_mmseqs_database(test_seqs, tmpd, "test_db")
        train_db = build_mmseqs_database(train_seqs, tmpd, "train_db")
        
        # Perform batch search
        sim_df = batch_search_protein_similarities(test_db, train_db, tmpd)
        max_protein_sims = sim_df.groupby("query")["pident"].max().to_dict()
        
        # Pre-compute fingerprints for all molecules
        print("Computing fingerprints for train molecules...")
        train_fps = []
        train_smiles = []
        for _, row in tqdm(train_df.iterrows(), desc="Train fingerprints"):
            if row["smiles"]:
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    train_fps.append(fp)
                    train_smiles.append(row["smiles"])
        
        print("Computing fingerprints for test molecules...")
        test_fps = []
        test_smiles = []
        for _, row in tqdm(test_df.iterrows(), desc="Test fingerprints"):
            if row["smiles"]:
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    test_fps.append(fp)
                    test_smiles.append(row["smiles"])
        
        # Convert fingerprints to numpy arrays
        train_fp_array = np.array([list(fp.ToBitString()) for fp in train_fps], dtype=np.uint8)
        test_fp_array = np.array([list(fp.ToBitString()) for fp in test_fps], dtype=np.uint8)
        
        # Calculate Tanimoto similarities using vectorized operations
        print("Calculating ligand similarities...")
        # For each test molecule, calculate similarity with all train molecules
        max_ligand_sims = {}
        for i, test_smiles in enumerate(test_smiles):
            # Calculate intersection and union counts
            intersection = np.sum(np.logical_and(test_fp_array[i], train_fp_array), axis=1)
            union = np.sum(np.logical_or(test_fp_array[i], train_fp_array), axis=1)
            # Calculate Tanimoto similarity
            similarities = intersection / union
            max_ligand_sims[test_smiles] = float(np.max(similarities))
        
        # Add similarity scores to test dataframe
        test_df["max_protein_similarity"] = test_df["system_id"].map(max_protein_sims)
        test_df["max_ligand_similarity"] = test_df["smiles"].map(max_ligand_sims)
        
        return test_df

def main():
    # Get protein sequences for train and test sets
    train_df, test_df = get_protein_sequences()
    
    # Calculate similarity scores
    print("Calculating similarity scores...")
    test_df = calculate_similarity_scores(test_df, train_df)
    
    # Save results
    output_dir = "../hiqbind/similarity_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    test_df.to_csv(os.path.join(output_dir, "test_similarities.csv"), index=False)
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of test entries: {len(test_df)}")
    print("\nProtein Similarity Statistics:")
    print(test_df["max_protein_similarity"].describe())
    print("\nLigand Similarity Statistics:")
    print(test_df["max_ligand_similarity"].describe())
    
    # Save summary statistics
    summary_stats = {
        "protein_similarity": test_df["max_protein_similarity"].describe().to_dict(),
        "ligand_similarity": test_df["max_ligand_similarity"].describe().to_dict()
    }
    pd.DataFrame(summary_stats).to_csv(os.path.join(output_dir, "similarity_summary.csv"))

if __name__ == "__main__":
    main()