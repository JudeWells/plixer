import os
import glob
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
from tqdm import tqdm
from scripts.create_hiqbind_dataset import (
    calculate_similarity_scores,
    extract_protein_sequence,
    batch_search_protein_similarities,
    build_mmseqs_database
)

"""
This dataset contains 16 targets
each target has multiple protein files
and corresponding bound ligand files

First we calculate the maximum protein sequence and ligand similarity
for each protein/ligand pair in the dataset compared to the training set
and additionally we calculate the maximum similarity to for each ligand
in the training set and in the smiles files.

Save the similarity results to a csv file.
"""

def calculate_lit_pcba_similarities(train_df, lit_pcba_directory):
    """Calculate similarity scores for LIT-PCBA dataset relative to training set."""
    with tempfile.TemporaryDirectory() as tmpd:
        # Build sequence dictionaries for LIT-PCBA proteins
        lit_pcba_seqs = {}
        lit_pcba_smiles = []
        lit_pcba_system_ids = []
        
        target_names = os.listdir(lit_pcba_directory)
        for target_name in target_names:
            target_directory = os.path.join(lit_pcba_directory, target_name)
            if not os.path.isdir(target_directory):
                continue
                
            # Process active ligands
            actives_df = pd.concat([pd.read_csv(p, sep=' ', header=None) for p in glob.glob(f"{target_directory}/active_*.smi")])
            actives_df.columns = ["smiles", "id_num"]
            
            # Process inactive ligands
            inactive_df = pd.concat([pd.read_csv(p, sep=' ', header=None) for p in glob.glob(f"{target_directory}/inactive_*.smi")])
            inactive_df.columns = ["smiles", "id_num"]
            
            # Process protein files
            protein_files = glob.glob(f"{target_directory}/*protein.mol2")
            for protein_file in protein_files:
                system_id = os.path.basename(protein_file).replace("_protein.mol2", "")
                try:
                    protein_sequence = extract_protein_sequence(protein_file)
                    if protein_sequence:
                        lit_pcba_seqs[system_id] = protein_sequence
                except Exception as e:
                    print(f"Error processing {protein_file}: {e}")
                    continue
        
        # Build MMSEQS databases
        lit_pcba_db = build_mmseqs_database(lit_pcba_seqs, tmpd, "lit_pcba_db")
        train_seqs = {row["system_id"]: row["protein_sequence"] for _, row in train_df.iterrows() if row["protein_sequence"]}
        train_db = build_mmseqs_database(train_seqs, tmpd, "train_db")
        
        # Perform batch search for protein similarities
        sim_df = batch_search_protein_similarities(lit_pcba_db, train_db, tmpd)
        max_protein_sims = sim_df.groupby("query")["pident"].max().to_dict()
        
        # Calculate ligand similarities
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
        
        # Convert fingerprints to numpy arrays
        train_fp_array = np.array([list(fp.ToBitString()) for fp in train_fps], dtype=np.uint8)
        
        # Process LIT-PCBA ligands
        results = []
        for target_name in target_names:
            target_directory = os.path.join(lit_pcba_directory, target_name)
            if not os.path.isdir(target_directory):
                continue
                
            # Process active ligands
            actives_df = pd.concat([pd.read_csv(p, sep=' ', header=None) for p in glob.glob(f"{target_directory}/active_*.smi")])
            actives_df.columns = ["smiles", "id_num"]
            actives_df["activity"] = 1
            
            # Process inactive ligands
            inactive_df = pd.concat([pd.read_csv(p, sep=' ', header=None) for p in glob.glob(f"{target_directory}/inactive_*.smi")])
            inactive_df.columns = ["smiles", "id_num"]
            inactive_df["activity"] = 0
            
            # Combine active and inactive ligands
            ligands_df = pd.concat([actives_df, inactive_df])
            
            # Calculate ligand similarities
            for _, row in tqdm(ligands_df.iterrows(), desc=f"Processing {target_name} ligands"):
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    test_fp = np.array(list(fp.ToBitString()), dtype=np.uint8)
                    
                    # Calculate intersection and union counts
                    intersection = np.sum(np.logical_and(test_fp, train_fp_array), axis=1)
                    union = np.sum(np.logical_or(test_fp, train_fp_array), axis=1)
                    # Calculate Tanimoto similarity
                    similarities = intersection / union
                    max_ligand_sim = float(np.max(similarities))
                    
                    results.append({
                        "target": target_name,
                        "smiles": row["smiles"],
                        "id_num": row["id_num"],
                        "activity": row["activity"],
                        "max_ligand_similarity": max_ligand_sim
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add protein similarities
        results_df["max_protein_similarity"] = results_df["target"].map(max_protein_sims)
        
        return results_df

def main():
    # Load training data
    train_save_path = '../hiqbind/plixer_train_data.csv'
    train_df = pd.read_csv(train_save_path)
    
    # LIT-PCBA directory
    lit_pcba_directory = "../LIT-PCBA_AVE_UNBIASED"
    
    # Calculate similarities
    print("Calculating similarity scores...")
    results_df = calculate_lit_pcba_similarities(train_df, lit_pcba_directory)
    
    # Save results
    output_dir = "../hiqbind/lit_pcba_similarity_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_dir, "lit_pcba_similarities.csv"), index=False)
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of LIT-PCBA entries: {len(results_df)}")
    print("\nProtein Similarity Statistics:")
    print(results_df["max_protein_similarity"].describe())
    print("\nLigand Similarity Statistics:")
    print(results_df["max_ligand_similarity"].describe())
    
    # Save summary statistics
    summary_stats = {
        "protein_similarity": results_df["max_protein_similarity"].describe().to_dict(),
        "ligand_similarity": results_df["max_ligand_similarity"].describe().to_dict()
    }
    pd.DataFrame(summary_stats).to_csv(os.path.join(output_dir, "lit_pcba_similarity_summary.csv"))

if __name__ == "__main__":
    main()
    