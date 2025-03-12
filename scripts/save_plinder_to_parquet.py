"""
Iterates through all of the plinder files in:
/mnt/disk2/plinder/2024-06/v2/systems
saves the pdb file and all liagnds above min size
to row in the parquet file.
also tracks whether each system is in train or validation set.
"""

import os
import pandas as pd
import glob
from rdkit import Chem
import zipfile
import logging
import shutil
import argparse
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_system(system_dir, min_atoms=10):
    pdb_file = glob.glob(os.path.join(system_dir, "*.pdb"))
    assert len(pdb_file) == 1
    pdb_file = pdb_file[0]
    
    # Read PDB file content
    with open(pdb_file, 'r') as f:
        pdb_content = f.read()
    
    all_ligand_files = glob.glob(os.path.join(system_dir, "ligand_files/*.sdf"))
    keep_ligands = []
    for ligand_file in all_ligand_files:
        ligand_mol = Chem.MolFromMolFile(ligand_file)
        if ligand_mol is None:
            continue
            
        # filter based on number of heavy atoms
        n_atoms = ligand_mol.GetNumAtoms()
        if n_atoms > min_atoms:
            # Serialize the molecule
            mol_block = Chem.MolToMolBlock(ligand_mol).encode()
            keep_ligands.append({
                "smiles": Chem.MolToSmiles(ligand_mol),
                "mol_block": mol_block,
                "n_atoms": n_atoms,
            })
        
    return pdb_content, keep_ligands

def process_split_df(split_df, all_zip_files):
    """
    Breaks the df into the middle char codes for faster lookups
    """
    split_df["batch"] = split_df['system_id'].apply(lambda x: x.split("_")[0][1:3])
    all_zip_codes = [os.path.basename(f).split(".")[0] for f in all_zip_files]
    assert set(split_df["batch"].unique()).issubset(set(all_zip_codes)) # some zips are empty and not in df
    indexed_dfs = {}
    for batch in split_df["batch"].unique():
        indexed_dfs[batch] = split_df[split_df["batch"] == batch]
    return indexed_dfs

def compute_cluster_weights(split_df):
    """
    Computes the weights for each cluster based on the number of systems in each cluster.
    Returns weights that are inversely proportional to cluster size.
    """
    cluster_counts = split_df["cluster"].value_counts()
    # Convert to float before taking inverse to avoid integer power error
    cluster_weights = 1.0 / cluster_counts.astype(float)
    return cluster_weights

def save_to_parquet(data, output_file):
    """Save processed data to a parquet file."""
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df.to_parquet(output_file, index=False)
    log.info(f"Saved {len(data)} systems to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Process Plinder files into parquet format')
    parser.add_argument('--input_dir', type=str, 
                        default="/mnt/disk2/plinder/2024-06/v2/systems", 
                        help='Directory containing Plinder zip files')
    parser.add_argument('--output_dir', type=str, 
                        default="/mnt/disk2/plinder/2024-06/v2/parquet", 
                        help='Directory to save parquet files')
    parser.add_argument('--split_file', type=str,
                        default="/mnt/disk2/plinder/2024-06/v2/splits/split.parquet",
                        help='Path to the split parquet file')
    parser.add_argument('--min_atoms', type=int, default=10,
                        help='Minimum number of atoms for ligands to keep')
    parser.add_argument('--systems_per_batch', type=int, default=500,
                        help='Number of systems per parquet file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temporary directory for zip extraction
    zip_out_dir = os.path.join(args.input_dir, "zip_out")
    os.makedirs(zip_out_dir, exist_ok=True)
    
    # Load split dataframe
    split_df = pd.read_parquet(args.split_file)
    cluster_weights = compute_cluster_weights(split_df)
    
    # Get all zip files
    all_zip_files = glob.glob(os.path.join(args.input_dir, "*.zip"))
    indexed_dfs = process_split_df(split_df, all_zip_files)
    
    # Initialize data containers for each split
    train_data = []
    val_data = []
    test_data = []
    
    # Initialize batch counters
    train_batch_idx = 0
    val_batch_idx = 0
    test_batch_idx = 0
    
    # Track saved files
    saved_files = []
    
    # Process each batch (zip file)
    for batch, df in tqdm(indexed_dfs.items(), desc="Processing batches"):
        zip_path = os.path.join(args.input_dir, f"{batch}.zip")
        if not os.path.exists(zip_path):
            log.info(f"Skipping {batch} because it doesn't exist")
            continue
        
        # Extract zip file
        log.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(zip_out_dir)
        
        # Process each system in the batch
        for _, row in tqdm(df.iterrows(), desc=f"Processing systems in {batch}", total=len(df)):
            system_dir = os.path.join(zip_out_dir, row["system_id"])
            if not os.path.exists(system_dir):
                log.warning(f"System directory {system_dir} does not exist, skipping")
                continue
                
            try:
                pdb_content, keep_ligands = process_system(system_dir, min_atoms=args.min_atoms)
                
                if not keep_ligands:
                    log.warning(f"No ligands found for system {row['system_id']}, skipping")
                    continue
                
                # Create entry for each system
                for ligand in keep_ligands:
                    system_data = {
                        "system_id": row["system_id"],
                        "pdb_content": pdb_content.encode(),
                        "smiles": ligand["smiles"],
                        "mol_block": ligand["mol_block"],
                        "n_atoms": ligand["n_atoms"],
                        "split": row["split"],
                        "cluster": row["cluster"],
                        "weight": cluster_weights[row["cluster"]]
                    }
                    
                    # Add to appropriate split
                    if row["split"] == "train":
                        train_data.append(system_data)
                    elif row["split"] == "val":
                        val_data.append(system_data)
                    elif row["split"] == "test":
                        test_data.append(system_data)
            except Exception as e:
                log.error(f"Error processing system {row['system_id']}: {e}")
        
        # Save batches if they reach the threshold
        if len(train_data) >= args.systems_per_batch:
            output_file = os.path.join(args.output_dir, f'train_batch_{train_batch_idx:03d}.parquet')
            save_to_parquet(train_data, output_file)
            saved_files.append(output_file)
            train_data = []
            train_batch_idx += 1
            
        if len(val_data) >= args.systems_per_batch:
            output_file = os.path.join(args.output_dir, f'val_batch_{val_batch_idx:03d}.parquet')
            save_to_parquet(val_data, output_file)
            saved_files.append(output_file)
            val_data = []
            val_batch_idx += 1
            
        if len(test_data) >= args.systems_per_batch:
            output_file = os.path.join(args.output_dir, f'test_batch_{test_batch_idx:03d}.parquet')
            save_to_parquet(test_data, output_file)
            saved_files.append(output_file)
            test_data = []
            test_batch_idx += 1
        
        # Clean up extracted files after processing the batch
        log.info(f"Cleaning up extracted files for {batch}")
        shutil.rmtree(zip_out_dir)
        os.makedirs(zip_out_dir, exist_ok=True)
    
    # Save any remaining data
    if train_data:
        output_file = os.path.join(args.output_dir, f'train_batch_{train_batch_idx:03d}.parquet')
        save_to_parquet(train_data, output_file)
        saved_files.append(output_file)
        
    if val_data:
        output_file = os.path.join(args.output_dir, f'val_batch_{val_batch_idx:03d}.parquet')
        save_to_parquet(val_data, output_file)
        saved_files.append(output_file)
        
    if test_data:
        output_file = os.path.join(args.output_dir, f'test_batch_{test_batch_idx:03d}.parquet')
        save_to_parquet(test_data, output_file)
        saved_files.append(output_file)
    
    # Clean up the temporary directory
    shutil.rmtree(zip_out_dir)
    
    # Create an index file
    index_df = pd.DataFrame({
        'parquet_file': saved_files,
        'split': [f.split('/')[-1].split('_')[0] for f in saved_files]
    })
    index_df.to_csv(os.path.join(args.output_dir, 'index.csv'), index=False)
    
    log.info(f"Created index file with {len(saved_files)} parquet files")
    log.info(f"Train batches: {train_batch_idx + 1}, Val batches: {val_batch_idx + 1}, Test batches: {test_batch_idx + 1}")

if __name__=="__main__":
    main()
