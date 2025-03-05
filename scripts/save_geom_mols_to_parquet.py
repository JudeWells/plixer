import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from pathlib import Path

def load_mol_from_pickle(file_path):
    """Load a pickled RDKit molecule."""
    try:
        with open(file_path, "rb") as f:
            mol_data = pickle.load(f)
        return mol_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def serialize_molecule(mol):
    """Serialize a molecule to a binary format."""
    return Chem.MolToMolBlock(mol).encode()

def save_molecules_to_parquet(molecules, output_dir, batch_idx):
    """Save a list of molecules to a parquet file."""
    if not molecules:
        return None
    
    data = {
        'smiles': [],
        'mol_block': [],
        'source_file': [],
        'conformer_id': [],
        'conformer_energy': []
    }
    
    for mol_data in molecules:
        data['smiles'].append(mol_data['smiles'])
        data['mol_block'].append(mol_data['mol_block'])
        data['source_file'].append(mol_data['source_file'])
        data['conformer_id'].append(mol_data['conformer_id'])
        data['conformer_energy'].append(mol_data['conformer_energy'])
    
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, f'batch_{batch_idx:06d}.parquet')
    df.to_parquet(output_file, index=False)
    return output_file

def process_geom_pickle_file(file_path, max_conformers=None):
    """Process a GEOM pickle file and extract molecules with conformers.
    
    Args:
        file_path: Path to the pickle file
        max_conformers: Maximum number of conformers to keep per molecule (random selection)
    """
    mol_data = load_mol_from_pickle(file_path)
    if mol_data is None:
        return None
    
    try: 
        
        # Check if there are conformers
        if "conformers" in mol_data and len(mol_data["conformers"]) > 0:
            smiles = Chem.MolToSmiles(mol_data["conformers"][0]["rd_mol"])
            
            # Get conformers and randomly select if needed
            conformers = mol_data["conformers"]
            if max_conformers and len(conformers) > max_conformers:
                # Randomly select max_conformers conformers
                selected_indices = np.random.choice(len(conformers), max_conformers, replace=False)
                conformers = [conformers[i] for i in selected_indices]
            
            # Process each conformer as a separate molecule
            results = []
            for i, conf in enumerate(conformers):
                if "rd_mol" in conf:
                    conf_mol = conf["rd_mol"]
                    conf_mol_block = serialize_molecule(conf_mol)
                    
                    # Extract energy if available
                    energy = conf.get("energy", 0.0)
                    
                    results.append({
                        'smiles': smiles,
                        'mol_block': conf_mol_block,
                        'source_file': str(file_path),
                        'conformer_id': i,
                        'conformer_energy': energy
                    })
            
            return results
        else:
            # If no conformers, use the main molecule
            main_mol_block = serialize_molecule(main_mol)
            return [{
                'smiles': smiles,
                'mol_block': main_mol_block,
                'source_file': str(file_path),
                'conformer_id': 0,
                'conformer_energy': 0.0
            }]
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process GEOM pickle files into parquet format')
    parser.add_argument('--input_dir', type=str, 
                        default="/mnt/disk2/VoxelDiffOuter/geom/rdkit_folder/drugs/train", 
                        help='Directory containing GEOM pickle files')
    parser.add_argument('--output_dir', type=str, 
                        default="/mnt/disk2/VoxelDiffOuter/geom/rdkit_folder/drugs/train_parquet_v2", 
                        help='Directory to save parquet files')
    parser.add_argument('--max_files', type=int, default=None, 
                        help='Maximum number of pickle files to process')
    parser.add_argument('--max_mols_per_batch', type=int, default=1000, 
                        help='Maximum number of molecules per parquet file')
    parser.add_argument('--max_batches', type=int, default=None, 
                        help='Maximum number of parquet files to create')
    parser.add_argument('--max_conformers', type=int, default=10, 
                        help='Maximum number of conformers to keep per molecule (random selection)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(args.input_dir, "*.pickle"))
    
    if args.max_files:
        pickle_files = pickle_files[:args.max_files]
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Process files in batches
    batch_idx = 0
    saved_files = []
    current_batch = []
    
    for file_path in tqdm(pickle_files):
        mol_results = process_geom_pickle_file(file_path, args.max_conformers)
        
        if mol_results:
            # Add all conformers to the current batch
            current_batch.extend(mol_results)
            
            # If we've collected enough molecules, save the batch
            if len(current_batch) >= args.max_mols_per_batch:
                output_file = save_molecules_to_parquet(current_batch, args.output_dir, batch_idx)
                if output_file:
                    saved_files.append(output_file)
                    batch_idx += 1
                    current_batch = []
                
                if args.max_batches and batch_idx >= args.max_batches:
                    print(f"Reached maximum number of batches ({args.max_batches})")
                    break
    
    # Save any remaining molecules
    if current_batch:
        output_file = save_molecules_to_parquet(current_batch, args.output_dir, batch_idx)
        if output_file:
            saved_files.append(output_file)
    
    print(f"Saved {len(saved_files)} parquet files with up to {args.max_mols_per_batch} molecules each")
    
    # Create an index file
    index_df = pd.DataFrame({'parquet_file': saved_files})
    index_df.to_csv(os.path.join(args.output_dir, 'index.csv'), index=False)
    
    print(f"Created index file with {len(saved_files)} parquet files")

if __name__ == "__main__":
    main()
